# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Block verification from raw logits without materializing [g, V] tensors.

Same algorithm as ``block_verification.block_verify`` (see there), sharing its
acceptance math (``acceptance_weights``, ``stop_probabilities``,
``emit_distribution``) and its rounding (fp32 log-probs, exponentiated in fp64),
computed as: one logsumexp per row of p and q, the draft-token ratios and b on
[g] vectors, and one Triton pass per row for r_i = sum_v max(b_i p_i - q_i, 0).
Only the residual row that is actually sampled from is materialized.

``block_verify_stats_lazy`` is the lazy variant (see
``block_verification.lazy_stop_index``): it needs only per-position statistics
(log p(x_i), log q(x_i), logsumexps) plus full rows at the few positions the
backward scan visits, and is pathwise identical to ``block_verify_logits``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from thundersync_rl.block_verification import (
    acceptance_weights,
    check_null_m,
    emit_distribution,
    emitted_token,
    lazy_slack,
    lazy_stop_index,
    null_root_residual,
    stop_probabilities,
    stop_probability,
)


@triton.jit
def _residual_mass_kernel(
    P, Q, lse_p, lse_q, b, out, sp, sq, vocab, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    lp0 = tl.load(lse_p + row)
    lq0 = tl.load(lse_q + row)
    bi = tl.load(b + row)
    acc = tl.zeros([BLOCK], tl.float64)
    for off in range(0, vocab, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < vocab
        lp = tl.load(P + row.to(tl.int64) * sp + cols, mask=mask, other=float("-inf"))
        lq = tl.load(Q + row.to(tl.int64) * sq + cols, mask=mask, other=float("-inf"))
        # Same rounding as the reference: fp32 log-probs, exponentiated in fp64.
        p = tl.exp((lp.to(tl.float32) - lp0).to(tl.float64))
        q = tl.exp((lq.to(tl.float32) - lq0).to(tl.float64))
        d = bi * p - q
        acc += tl.where(d > 0, d, 0.0)
    tl.store(out + row, tl.sum(acc, 0))


def residual_masses(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    lse_p: torch.Tensor,
    lse_q: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """r_k = sum_v max(b_k p_k(v) - q_k(v), 0) for rows k (fp64 [k]).

    One Triton program per row with a fixed reduction order, so a row's r does
    not depend on which other rows are in the call.
    """
    k = b.numel()  # rows 0..k-1 (p_logits may carry the extra bonus row)
    if not p_logits.is_cuda:
        # CPU reference (tests): same rounding, fp32 log-probs exponentiated in fp64.
        p = (p_logits[:k].float() - lse_p[:k, None]).double().exp()
        q = (q_logits[:k].float() - lse_q[:k, None]).double().exp()
        return (b[:, None] * p - q).clamp_min(0).sum(-1)
    r = torch.empty(k, dtype=torch.float64, device=p_logits.device)
    if k:
        if p_logits.stride(-1) != 1:
            p_logits = p_logits.contiguous()
        if q_logits.stride(-1) != 1:
            q_logits = q_logits.contiguous()
        _residual_mass_kernel[(k,)](
            p_logits,
            q_logits,
            lse_p.contiguous(),
            lse_q.contiguous(),
            b.contiguous(),
            r,
            p_logits.stride(0),
            q_logits.stride(0),
            p_logits.shape[-1],
            BLOCK=2048,
        )
    return r


def _probs(row: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    return (row.float() - lse).double().exp()


def _emit(p_row, q_row, lse_p_tau, lse_q_tau, b_tau, generator, null_m=1.0):
    """Sample the emitted token (bonus if ``q_row`` is None, else the residual;
    ``null_m`` > 1 only for the ⊥-BV root residual: ``NULL_TOKEN`` = BOT)."""
    q = None if q_row is None else _probs(q_row, lse_q_tau)
    dist = emit_distribution(_probs(p_row, lse_p_tau), q, b_tau, null_m)
    y = int(torch.multinomial((dist / dist.sum()).float(), 1, generator=generator))
    return emitted_token(y, p_row.shape[-1])


def block_verify_logits(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    draft: torch.Tensor,
    generator: torch.Generator | None = None,
    *,
    with_logq: bool = False,
) -> tuple[int, int, torch.Tensor] | tuple[int, int, torch.Tensor, torch.Tensor]:
    """Sample (tau, y) from raw logits; also return log p of the kept tokens.

    ``p_logits``: [g+1, V] target logits, ``q_logits``: [g, V] draft logits (any
    float dtype; last dim contiguous), ``draft``: [g] token ids.
    Returns (tau, y, logp) with logp = log p(kept[i]) for kept = draft[:tau] + [y];
    with ``with_logq`` also logq = log q(draft[i]) for i < tau ([tau], fp32, the
    values the acceptance ratios used).
    """
    g = draft.numel()
    dev = p_logits.device
    if p_logits.stride(-1) != 1:
        p_logits = p_logits.contiguous()
    if q_logits.stride(-1) != 1:
        q_logits = q_logits.contiguous()
    lse_p = torch.logsumexp(p_logits.float(), -1)  # [g+1]
    lse_q = torch.logsumexp(q_logits.float(), -1) if g else lse_p[:0]
    ar = torch.arange(g, device=dev)
    lpx = p_logits[ar, draft].float() - lse_p[:g]
    lqx = q_logits[ar, draft].float() - lse_q
    b = acceptance_weights(lpx - lqx)  # [g+1]
    r = residual_masses(p_logits, q_logits, lse_p[:g], lse_q, b[:g])
    h = stop_probabilities(r, b)
    eta = torch.rand(g + 1, generator=generator, dtype=torch.float64, device=dev)
    tau = int(torch.nonzero(eta <= h).flatten().max())
    q_row = None if tau == g else q_logits[tau]
    lse_q_tau = None if tau == g else lse_q[tau]
    y = _emit(p_logits[tau], q_row, lse_p[tau], lse_q_tau, b[tau], generator)
    kept = torch.cat([draft[:tau], torch.tensor([y], device=dev)])
    logp = p_logits[torch.arange(tau + 1, device=dev), kept].float() - lse_p[: tau + 1]
    if with_logq:
        return tau, y, logp, lqx[:tau]
    return tau, y, logp


def block_verify_stats_lazy(
    lpx: torch.Tensor,
    lqx: torch.Tensor,
    lse_p: torch.Tensor,
    lse_q: torch.Tensor,
    p_rows,
    q_rows,
    generator: torch.Generator | None = None,
    null_m: float = 1.0,
) -> tuple[int, int, torch.Tensor, int]:
    """Lazy block verification from per-position statistics and on-demand rows.

    ``lpx``/``lqx``: [g] fp32 log p(x_i) / log q(x_i) (draft-token logit minus
    logsumexp); ``lse_p``: [g+1], ``lse_q``: [g] fp32 logsumexps.
    ``p_rows(idx)`` / ``q_rows(idx)``: raw logits rows [k, V] at positions ``idx``
    (LongTensor), bitwise the rows the statistics were computed from.

    Given the same inputs and generator state, (tau, y, logp) and the generator
    state afterwards equal ``block_verify_logits`` on the full logits: the same
    ``rand(g+1)`` and ``multinomial`` calls, the same b, and r_i (same Triton
    program per row) and h_i (same formula) wherever ``lazy_stop_index`` needs
    them. Full rows are read only at positions where eta_i <= b_i (1 + slack)
    above tau, and at tau itself. Returns (tau, y, logp, rows evaluated).

    ``null_m`` = M > 1: ⊥-BV (see ``block_verification``). y = ``NULL_TOKEN``
    means BOT (the draft yields no output); logp is then log p of the kept
    prefix only (empty, since BOT happens only at tau = 0). M = 1 takes exactly
    the plain path.
    """
    g = lpx.numel()
    check_null_m(null_m, g)
    dev = lse_p.device
    b = acceptance_weights(lpx - lqx, null_m)

    def mass(idx, bi):
        return residual_masses(p_rows(idx), q_rows(idx), lse_p[idx], lse_q[idx], bi)

    eta = torch.rand(g + 1, generator=generator, dtype=torch.float64, device=dev)
    tau, n_eval = lazy_stop_index(
        b, eta, null_root_residual(mass, b, null_m), lazy_slack(lse_p[:g])
    )
    at = torch.tensor([tau], device=dev)
    p_row = p_rows(at)[0]
    q_row = None if tau == g else q_rows(at)[0]
    lse_q_tau = None if tau == g else lse_q[tau]
    m = null_m if tau == 0 and tau < g else 1.0
    y = _emit(p_row, q_row, lse_p[tau], lse_q_tau, b[tau], generator, m)
    if y < 0:
        return tau, y, lpx[:tau], n_eval
    logp = torch.cat([lpx[:tau], (p_row[y].float() - lse_p[tau]).view(1)])
    return tau, y, logp, n_eval


def sample_by_inversion(dist: torch.Tensor, u: float) -> int:
    """Index sampled from the unnormalized ``dist`` by inverting its CDF at ``u``."""
    c = torch.cumsum(dist.double(), 0)
    i = int(torch.searchsorted(c, torch.tensor([u], dtype=c.dtype, device=c.device) * c[-1], right=True))
    return min(i, dist.numel() - 1)


def block_verify_stats_lazy_grid(
    lpx: torch.Tensor,
    lqx: torch.Tensor,
    lse_p: torch.Tensor,
    lse_q: torch.Tensor,
    p_rows,
    q_rows,
    generator: torch.Generator | None,
    envelopes: tuple[float, ...],
    chunk: int = 512,
) -> tuple[dict[float, tuple[int, int, float | None]], int]:
    """``block_verify_stats_lazy`` at every envelope M of ``envelopes``, one set of draws.

    One ``rand(g+1)`` (eta) and one ``rand(1)`` (u) per draft. For each M:
    tau_M is the lazy scan's stop index with M's weights b_M and the shared
    eta, and y_M is drawn from M's emission law by inverting its CDF at u.
    Marginally in M this is the M sampler (eta uniform, emission from the same
    law), so taking the outcome at an M chosen independently of this draft's
    draws is exact (``null_bv.decide_pools``). Returns ({M: (tau, y, log p(y)
    or None for BOT)}, rows evaluated); y = ``NULL_TOKEN``: BOT.

    Rows are shared across envelopes and fetched in batches: positions are
    scanned from the top in chunks of ``chunk`` candidate positions (the union
    over the envelopes still scanning); each chunk's p and q rows are built
    once (one batched head GEMM per chunk) and every envelope's residual masses
    at its own candidates come from them. tau_M = the highest candidate with
    eta <= h, as in ``lazy_stop_index``; evaluating r at more candidates than
    the lazy scan would does not change it (r is a function of the position
    and b), and q rows / residual masses are row-count invariant (batch-invariant
    head GEMM, one Triton program per row), so outcomes are bitwise those of
    the per-envelope scan.
    """
    g = lpx.numel()
    dev = lse_p.device
    eta = torch.rand(g + 1, generator=generator, dtype=torch.float64, device=dev)
    u = float(torch.rand(1, generator=generator, dtype=torch.float64, device=dev))
    slack = lazy_slack(lse_p[:g])
    slack_h = slack.cpu() if isinstance(slack, torch.Tensor) else slack
    eta_h = eta.cpu()
    weights: dict[float, torch.Tensor] = {}
    tau: dict[float, int] = {}
    cand: dict[float, torch.Tensor] = {}  # descending candidate positions per M
    for M in envelopes:
        check_null_m(M, g)
        weights[M] = b = acceptance_weights(lpx - lqx, M)
        b_h = b.cpu()
        if bool(eta_h[g] <= b_h[g]):
            tau[M] = g
        else:
            cand[M] = torch.nonzero(eta_h[:g] <= b_h[:g] * (1 + slack_h)).flatten().flip(0)
    rows: dict[tuple[str, int], torch.Tensor] = {}
    n_eval = 0
    done = {M: 0 for M in cand}
    while any(M not in tau for M in cand):
        live = [M for M in cand if M not in tau]
        for M in live:
            if done[M] >= cand[M].numel():
                raise RuntimeError("block verification: no stop position (eta > h everywhere)")
        # The next ``chunk`` positions (descending) of the union of the live candidates.
        pending = torch.unique(torch.cat([cand[M][done[M]:] for M in live]))
        idx = pending.flip(0)[:chunk]
        lo = int(idx[-1])
        idx_d = idx.to(dev)
        P, Q = p_rows(idx_d), q_rows(idx_d)
        n_eval += idx.numel()
        where = {int(t): k for k, t in enumerate(idx.tolist())}
        for M in live:
            c = cand[M][done[M]:]
            c = c[c >= lo]
            done[M] += c.numel()
            if not c.numel():
                continue
            sel = torch.tensor([where[int(t)] for t in c.tolist()], device=dev)
            c_d = c.to(dev)
            b = weights[M]

            def mass(i, bi, sel=sel):
                return residual_masses(P[sel], Q[sel], lse_p[i], lse_q[i], bi)

            h = stop_probability(null_root_residual(mass, b, M)(c_d), b[c_d])
            hit = torch.nonzero((eta[c_d] <= h).cpu()).flatten()
            if hit.numel():
                tau[M] = int(c[hit[0]])
        for k, t in enumerate(idx.tolist()):  # keep the stop rows for the emissions
            if any(tau.get(M) == t for M in live):
                rows[("p", t)], rows[("q", t)] = P[k], Q[k]

    def row(kind: str, t: int) -> torch.Tensor:
        if (kind, t) not in rows:
            at = torch.tensor([t], device=dev)
            rows[(kind, t)] = (p_rows if kind == "p" else q_rows)(at)[0]
        return rows[(kind, t)]

    out: dict[float, tuple[int, int, float | None]] = {}
    for M in envelopes:
        t, b = tau[M], weights[M]
        p_row = row("p", t)
        q = None if t == g else _probs(row("q", t), lse_q[t])
        m = M if t == 0 and t < g else 1.0
        dist = emit_distribution(_probs(p_row, lse_p[t]), q, b[t], m)
        y = emitted_token(sample_by_inversion(dist, u), p_row.shape[-1])
        out[M] = (t, y, None if y < 0 else float(p_row[y].float() - lse_p[t]))
    return out, n_eval


def block_verify_logits_lazy(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    draft: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[int, int, torch.Tensor]:
    """``block_verify_logits`` with the lazy scan (pathwise identical results).

    Skips the residual pass entirely on whole-accepted drafts (eta_g <= b_g).
    """
    g = draft.numel()
    dev = p_logits.device
    if p_logits.stride(-1) != 1:
        p_logits = p_logits.contiguous()
    if q_logits.stride(-1) != 1:
        q_logits = q_logits.contiguous()
    lse_p = torch.logsumexp(p_logits.float(), -1)
    lse_q = torch.logsumexp(q_logits.float(), -1) if g else lse_p[:0]
    ar = torch.arange(g, device=dev)
    lpx = p_logits[ar, draft].float() - lse_p[:g]
    lqx = q_logits[ar, draft].float() - lse_q
    tau, y, logp, _ = block_verify_stats_lazy(
        lpx,
        lqx,
        lse_p,
        lse_q,
        lambda idx: p_logits[idx],
        lambda idx: q_logits[idx],
        generator,
    )
    return tau, y, logp
