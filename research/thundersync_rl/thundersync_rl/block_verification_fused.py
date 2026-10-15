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

import math

import torch
import triton
import triton.language as tl

from thundersync_rl.prof import timed
from thundersync_rl.block_verification import (
    acceptance_weights,
    band_exit,
    check_emission,
    check_null_m,
    check_stop_probabilities,
    check_verification_inputs,
    zero_mass_below,
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
    check_emission(dist, "residual" if q_row is not None else "bonus")
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
    check_verification_inputs(lpx, lqx, lse_p, lse_q)
    b = acceptance_weights(lpx - lqx)  # [g+1]
    r = residual_masses(p_logits, q_logits, lse_p[:g], lse_q, b[:g])
    h = stop_probabilities(r, b)
    check_stop_probabilities(h, "block_verify_logits")
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
    band: float | None = None,
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

    ``band``: banded IS (``block_verification``): the target is q at positions
    1..T, so the log-ratio there is 0 and residual masses of rows i < T are
    exactly 0 (b_i <= 1); those rows are never read. Pathwise identical to
    ``block_verify`` with the same ``band``.
    """
    g = lpx.numel()
    check_null_m(null_m, g)
    check_verification_inputs(lpx, lqx, lse_p, lse_q)
    dev = lse_p.device
    log_ratio = lpx - lqx
    t_exit = 0  # rows below this have target q
    if band is not None:
        t = band_exit(log_ratio, band)
        t_exit = g if t is None else t
        log_ratio = torch.cat([torch.zeros_like(log_ratio[:t_exit]), log_ratio[t_exit:]])
    b = acceptance_weights(log_ratio, null_m)

    def mass(idx, bi):
        return residual_masses(p_rows(idx), q_rows(idx), lse_p[idx], lse_q[idx], bi)

    if t_exit:
        mass = zero_mass_below(t_exit, mass)

    eta = torch.rand(g + 1, generator=generator, dtype=torch.float64, device=dev)
    tau, n_eval = lazy_stop_index(
        b, eta, null_root_residual(mass, b, null_m), lazy_slack(lse_p[:g])
    )
    at = torch.tensor([tau], device=dev)
    q_row = None if tau == g else q_rows(at)[0]
    lse_q_tau = None if tau == g else lse_q[tau]
    if tau < t_exit:
        # Banded IS: the target row is q's (only the ⊥-BV root reaches here).
        p_row, lse_p_tau = q_row, lse_q_tau
    else:
        p_row, lse_p_tau = p_rows(at)[0], lse_p[tau]
    m = null_m if tau == 0 and tau < g else 1.0
    y = _emit(p_row, q_row, lse_p_tau, lse_q_tau, b[tau], generator, m)
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
    check_verification_inputs(lpx, lqx, lse_p, lse_q)
    dev = lse_p.device
    weights: dict[float, torch.Tensor] = {}
    tau: dict[float, int] = {}
    cand: dict[float, torch.Tensor] = {}  # descending candidate positions per M
    with timed("scan_setup"):
        eta = torch.rand(g + 1, generator=generator, dtype=torch.float64, device=dev)
        u = float(torch.rand(1, generator=generator, dtype=torch.float64, device=dev))
        slack = lazy_slack(lse_p[:g])
        slack_h = slack.cpu() if isinstance(slack, torch.Tensor) else slack
        eta_h = eta.cpu()
        for M in envelopes:
            check_null_m(M, g)
        # Every envelope's weights in one pass (bitwise ``acceptance_weights``: the
        # same fp64 cumsum, minus log M after the root, exp(S - cummax S) per row).
        s_cum = torch.cat(
            [torch.zeros(1, dtype=torch.float64, device=dev), torch.cumsum((lpx - lqx).double(), 0)]
        )
        S = s_cum.unsqueeze(0).repeat(len(envelopes), 1)
        for k, M in enumerate(envelopes):
            if M != 1.0:
                S[k, 1:] -= math.log(M)
        B = (S - torch.cummax(S, 1).values).exp()
        B_h = B.cpu()
        for k, M in enumerate(envelopes):
            weights[M], b_h = B[k], B_h[k]
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
        with timed("p_rows"):
            P = p_rows(idx_d)
        with timed("q_rows"):
            Q = q_rows(idx_d)
        n_eval += idx.numel()
        # Chunk-local row of each position: positions are descending, so a
        # searchsorted on the ascending reversal gives it without a Python dict.
        asc = idx.flip(0)
        with timed("residual"):
            # Every live envelope's candidates in this chunk, one residual launch and
            # one host sync for all of them (the kernel is per row, so each r is
            # bitwise its own launch's; the root adjustment as ``null_root_residual``).
            parts = []
            for M in live:
                c = cand[M][done[M]:]
                c = c[c >= lo]
                done[M] += c.numel()
                if c.numel():
                    parts.append((M, c))
            if parts:
                cs = torch.cat([c for _, c in parts])
                sel = (idx.numel() - 1 - torch.searchsorted(asc, cs)).to(dev)
                c_d = cs.to(dev)
                bi = torch.cat([weights[M][c.to(dev)] for M, c in parts])
                root = c_d == 0
                inv = torch.cat(
                    [torch.full((c.numel(),), M, dtype=torch.float64, device=dev) for M, c in parts]
                )
                nonone = inv != 1.0
                b_res = torch.where(root & nonone, bi / inv, bi)
                r = residual_masses(P[sel], Q[sel], lse_p[c_d], lse_q[c_d], b_res)
                r = torch.where(root & nonone, r + (1.0 - 1.0 / inv), r)
                h_c = stop_probability(r, bi)
                check_stop_probabilities(h_c, "grid scan")
                hit = (eta[c_d] <= h_c).cpu()
                a = 0
                for M, c in parts:
                    hm = torch.nonzero(hit[a : a + c.numel()]).flatten()
                    a += c.numel()
                    if hm.numel():
                        tau[M] = int(c[hm[0]])
        for M in live:  # keep the stop rows for the emissions
            t = tau.get(M)
            if t is not None and t >= lo and ("p", t) not in rows:
                k = idx.numel() - 1 - int(torch.searchsorted(asc, torch.tensor(t)))
                rows[("p", t)], rows[("q", t)] = P[k], Q[k]

    def row(kind: str, t: int) -> torch.Tensor:
        if (kind, t) not in rows:
            at = torch.tensor([t], device=dev)
            rows[(kind, t)] = (p_rows if kind == "p" else q_rows)(at)[0]
        return rows[(kind, t)]

    out: dict[float, tuple[int, int, float | None]] = {}
    with timed("emission"):
        for M in envelopes:
            t, b = tau[M], weights[M]
            p_row = row("p", t)
            q = None if t == g else _probs(row("q", t), lse_q[t])
            m = M if t == 0 and t < g else 1.0
            dist = emit_distribution(_probs(p_row, lse_p[t]), q, b[t], m)
            check_emission(dist, f"M={M} tau={t} of g={g}")
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
