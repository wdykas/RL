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
"""Packed-transfer ⊥-BV grid scan (block_verification_fused.block_verify_stats_lazy_grid) vs the per-value-read
implementation it replaced (c44e8e43, copied below as ``_grid_ref``): same outcomes, rows evaluated and generator
state, bit for bit, on CUDA; the same errors; and the host-sync count it was built to cut."""

import math
import warnings

import pytest
import torch

from thundersync_rl.block_verification import (
    NonFiniteVerification,
    acceptance_weights,  # noqa: F401  (reference namespace)
    check_emission,
    check_null_m,
    check_stop_probabilities,
    check_verification_inputs,
    emit_distribution,
    emitted_token,
    lazy_slack,
    stop_probability,
)
from thundersync_rl.block_verification_fused import (
    _d2h,
    _h2d,
    _law_bug,
    _probs,
    block_verify_stats_lazy_grid,
    residual_masses,
    sample_by_inversion,
)
from thundersync_rl.prof import timed

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


def _grid_ref(
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
            if m != 1.0 and _law_bug() == "null_m_1":
                m = 1.0  # debug negative control: root residual without the ⊥ envelope (no BOT column)
            dist = emit_distribution(_probs(p_row, lse_p[t]), q, b[t], m)
            if q is not None and m == 1.0 and _law_bug() == "residual_from_p":
                dist = _probs(p_row, lse_p[t])  # debug negative control (see ``_emit``)
            check_emission(dist, f"M={M} tau={t} of g={g}")
            y = emitted_token(sample_by_inversion(dist, u), p_row.shape[-1])
            out[M] = (t, y, None if y < 0 else float(p_row[y].float() - lse_p[t]))
    return out, n_eval


def _case(seed, g, V, dev, accept_all=False, p_zero=0, sharp=0.6):
    rng = torch.Generator().manual_seed(seed)
    p = torch.log_softmax(torch.randn(g + 1, V, generator=rng) * 2, -1)
    q = torch.log_softmax(p[:g] + sharp * torch.randn(g, V, generator=rng), -1)
    if accept_all:
        q = p[:g].clone()
    draft = torch.multinomial(q.exp(), 1, generator=rng)[:, 0]
    ar = torch.arange(g)
    lpx, lqx = p[ar, draft].clone(), q[ar, draft].clone()
    for j in range(p_zero):
        lpx[(7 * j + 3) % g] = float("-inf")  # exact top-p: the drafted token is outside the nucleus
    return p.to(dev), q.to(dev), lpx.to(dev), lqx.to(dev)


def _run(fn, p, q, lpx, lqx, seed, envs, chunk):
    g = lpx.numel()
    dev = p.device
    z = torch.zeros(g + 1, device=dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    out, n_eval = fn(lpx, lqx, z, z[:g], lambda i: p[i.to(dev)], lambda i: q[i.to(dev)], gen, envs, chunk=chunk)
    return out, n_eval, gen.get_state()


CASES = [
    dict(g=1, V=5), dict(g=2, V=9), dict(g=40, V=7), dict(g=300, V=33), dict(g=2000, V=129),
    dict(g=64, V=17, accept_all=True), dict(g=50, V=11, p_zero=2), dict(g=500, V=65, sharp=3.0),
]


@cuda
@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("chunk", [1, 4, 512])
def test_packed_grid_scan_is_bitwise_the_per_read_scan(case, seed, chunk):
    p, q, lpx, lqx = _case(seed, dev="cuda", **CASES[case])
    envs = (1.0, 1.25, 1.5, 2.0, 3.0)
    got = _run(block_verify_stats_lazy_grid, p, q, lpx, lqx, 1000 + seed, envs, chunk)
    want = _run(_grid_ref, p, q, lpx, lqx, 1000 + seed, envs, chunk)
    assert got[0] == want[0]  # every envelope's (tau, y, log p(y)): ints and fp32-exact floats
    assert got[1] == want[1]
    assert torch.equal(got[2], want[2])  # the same draws were taken from the generator


@cuda
def test_packed_grid_scan_bot_and_bonus_are_reached():
    """Non-vacuous: the cases above include BOT emissions (root residual at M > 1) and whole-draft bonuses."""
    seen = set()
    for case in range(len(CASES)):
        for seed in range(5):
            p, q, lpx, lqx = _case(seed, dev="cuda", **CASES[case])
            out, _, _ = _run(block_verify_stats_lazy_grid, p, q, lpx, lqx, 1000 + seed, (1.0, 1.5, 3.0), 4)
            g = lpx.numel()
            for t, y, lp in out.values():
                seen.add("bot" if y < 0 else ("bonus" if t == g else "residual"))
    assert {"bot", "bonus", "residual"} <= seen, seen


@cuda
def test_packed_grid_scan_cpu_equals_cuda_reference_paths():
    """The CPU path (tests) is the same code without pinned copies."""
    p, q, lpx, lqx = _case(3, 60, 13, "cpu")
    envs = (1.0, 1.5)
    assert _run(block_verify_stats_lazy_grid, p, q, lpx, lqx, 7, envs, 8)[:2] == _run(_grid_ref, p, q, lpx, lqx, 7, envs, 8)[:2]


@cuda
def test_errors_are_the_reference_errors():
    p, q, lpx, lqx = _case(1, 20, 9, "cuda")
    bad = lqx.clone()
    bad[3] = float("-inf")
    for fn in (block_verify_stats_lazy_grid, _grid_ref):
        with pytest.raises(NonFiniteVerification, match="engine log q"):
            _run(fn, p, q, lpx, bad, 1, (1.0, 1.5), 4)
    nanp = lpx.clone()
    nanp[2] = float("nan")
    for fn in (block_verify_stats_lazy_grid, _grid_ref):
        with pytest.raises(NonFiniteVerification, match="learner log p"):
            _run(fn, p, q, nanp, lqx, 1, (1.0, 1.5), 4)
    pn = p.clone()
    pn[20] = float("nan")  # bonus row only: caught by the emission check
    for fn in (block_verify_stats_lazy_grid, _grid_ref):
        with pytest.raises(NonFiniteVerification, match="emission"):
            _run(fn, pn, p[:20].clone(), lpx, lpx.clone(), 1, (1.0,), 4)


def _syncs(fn, *a, **k):
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            fn(*a, **k)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    return sum("synchroniz" in str(x.message) for x in w)


@cuda
def test_host_syncs_are_cut():
    p, q, lpx, lqx = _case(0, 2000, 129, "cuda", sharp=3.0)
    envs = (1.0, 1.25, 1.5, 2.0, 3.0)
    new = _syncs(_run, block_verify_stats_lazy_grid, p, q, lpx, lqx, 5, envs, 512)
    old = _syncs(_run, _grid_ref, p, q, lpx, lqx, 5, envs, 512)
    print(f"[devscan] host syncs per row: packed {new} vs per-read {old}")
    assert new < old


@cuda
@pytest.mark.parametrize("n", [1, 7, 512, 8191, 8192, 70000])
def test_finite_selection_sum_is_the_full_sum(n):
    """Worker row pack: with every token finite, d[isfinite(d)] is d itself, so its device sums are d's (bitwise)."""
    d = torch.randn(n, dtype=torch.float64, device="cuda") * 1e-3
    s = d[torch.isfinite(d)]
    assert torch.equal(s.sum(), d.sum()) and torch.equal((s * s).sum(), (d * d).sum())


@cuda
def test_transfer_helpers_preserve_values():
    x = torch.randn(1000, dtype=torch.float32)
    y = _h2d(x, torch.device("cuda"))
    assert y.is_cuda and torch.equal(y.cpu(), x)
    parts = [torch.tensor([True], device="cuda"), torch.arange(5, device="cuda"), y[:7], y.double()[:3]]
    got = _d2h(parts)
    assert [g.tolist() for g in got] == [[1.0], [0.0, 1.0, 2.0, 3.0, 4.0], [float(v) for v in x[:7]], x.double()[:3].tolist()]


@cuda
def test_flag_off_is_the_per_read_scan(monkeypatch):
    """THUNDERSYNC_DEVSCAN=0 (A/B control and fallback) runs the replaced implementation: same results, same syncs."""
    p, q, lpx, lqx = _case(2, 500, 65, "cuda", sharp=3.0)
    envs = (1.0, 1.5, 3.0)
    on = _run(block_verify_stats_lazy_grid, p, q, lpx, lqx, 9, envs, 512)
    monkeypatch.setenv("THUNDERSYNC_DEVSCAN", "0")
    off = _run(block_verify_stats_lazy_grid, p, q, lpx, lqx, 9, envs, 512)
    ref = _run(_grid_ref, p, q, lpx, lqx, 9, envs, 512)
    assert on[:2] == off[:2] == ref[:2] and torch.equal(off[2], ref[2])
    n_off = _syncs(_run, block_verify_stats_lazy_grid, p, q, lpx, lqx, 9, envs, 512)
    n_ref = _syncs(_run, _grid_ref, p, q, lpx, lqx, 9, envs, 512)
    monkeypatch.setenv("THUNDERSYNC_DEVSCAN", "1")
    n_on = _syncs(_run, block_verify_stats_lazy_grid, p, q, lpx, lqx, 9, envs, 512)
    print(f"[devscan] host syncs per row: on {n_on} off {n_off} reference {n_ref}")
    assert n_off == n_ref and n_on < n_off
