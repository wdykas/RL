"""53-bit engine sampler (keyed_sampling.wide_sample_logits / _neg_log_exp53): tail law, batch invariance, cost.

GPU only. The far-tail law is tested on a two-token race: a token with relative probability r = p_u / p_top wins
with probability r / (1 + r). Counting wins over N keyed draws (a Triton kernel evaluating the sampler's own noise
function) gives a binomial estimate; the old fp32 / 24-bit noises are evaluated exactly (24-bit grid) or
empirically (FlashInfer's kernel, the engine's production sampler before ts-exact-harden).
"""

import math
import os
import sys
import time

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU test")

if torch.cuda.is_available():
    import triton
    import triton.language as tl

    from thundersync_rl.keyed_sampling import _mix32, _neg_log_exp53, wide_sample_logits

    @triton.jit
    def _race_kernel(out_ptr, log_r, n_per, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        wins = tl.zeros([BLOCK], tl.int64)
        for it in range(0, n_per, BLOCK):
            idx = (pid * n_per + it + tl.arange(0, BLOCK)).to(tl.uint32)
            key = _mix32(idx ^ 0x1234567)
            g0 = _neg_log_exp53(tl.zeros([BLOCK], tl.uint32), key)
            g1 = _neg_log_exp53(tl.full([BLOCK], 1, tl.uint32), key)
            wins += (log_r + g1 > g0).to(tl.int64)
        tl.store(out_ptr + pid, tl.sum(wins, axis=0))


def _race_wins(r: float, programs: int = 4096, n_per: int = 1 << 20) -> tuple[int, int]:
    out = torch.zeros(programs, dtype=torch.int64, device="cuda")
    _race_kernel[(programs,)](out, math.log(r), n_per, BLOCK=1024)
    return int(out.sum()), programs * n_per


def _old_keyed_win_prob(r: float) -> float:
    """Exact win probability of the pre-fix keyed sampler (U on the 2^24 midpoint grid, fp64 math)."""
    import numpy as np

    n = 1 << 24
    u = (np.arange(n, dtype=np.float64) + 0.5) / n
    g = -np.log(-np.log(u))  # sorted ascending
    # win iff log r + G1 > G0: P = mean over G1 of P(G0 < log r + G1)
    return float(np.searchsorted(g, math.log(r) + g, side="left").sum() / n / n)


@pytest.mark.parametrize("r", [1e-3, 1e-6, 1e-8])
def test_tail_law_two_token_race(r):
    wins, n = _race_wins(r)
    p = r / (1 + r)
    sd = math.sqrt(n * p * (1 - p))
    old = _old_keyed_win_prob(r)
    print(f"[wide sampler] r={r:.0e} N={n:.2e} wins={wins} expected={n * p:.1f} (sd {sd:.1f}) "
          f"old 24-bit keyed exact={old:.3e} (true {p:.3e})")
    assert abs(wins - n * p) < 5 * sd + 1


def test_unreachable_before_reachable_now():
    # r = 1e-9 is below the old 24-bit keyed range (exp(-20.18) = 1.7e-9): never sampled before.
    assert _old_keyed_win_prob(1e-9) == 0.0
    wins, n = _race_wins(1e-9, programs=8192, n_per=1 << 21)
    exp = n * 1e-9
    print(f"[wide sampler] r=1e-9 N={n:.2e} wins={wins} expected={exp:.1f}")
    assert abs(wins - exp) < 5 * math.sqrt(exp) + 1


def test_flashinfer_tail_floor():
    flashinfer = pytest.importorskip("flashinfer")
    r, rows, reps = 1e-7, 1 << 22, 256
    logits = torch.tensor([0.0, math.log(r)], device="cuda").repeat(rows, 1).contiguous()
    gen = torch.Generator(device="cuda").manual_seed(0)
    wins = 0
    for _ in range(reps):
        y = flashinfer.sampling.sampling_from_logits(logits, deterministic=True, generator=gen)
        wins += int((y == 1).sum())
    n = rows * reps
    print(f"[flashinfer] r={r:.0e} N={n:.2e} wins={wins} expected={n * r:.1f}")
    # Recorded, not asserted: the production sampler's tail before the fix.


def test_batch_invariant_and_reproducible():
    torch.manual_seed(0)
    V = 131072
    logits = torch.randn(64, V, device="cuda") * 3
    rid = torch.arange(1000, 1064, device="cuda", dtype=torch.int64)
    pos = torch.randint(10, 9000, (64,), device="cuda", dtype=torch.int64)
    temp = torch.ones(64, device="cuda")
    a = wide_sample_logits(logits, rid, pos, temp, salt=42)
    b = wide_sample_logits(logits, rid, pos, temp, salt=42)
    assert torch.equal(a, b)
    perm = torch.randperm(64, device="cuda")
    c = wide_sample_logits(logits[perm], rid[perm], pos[perm], temp[perm], salt=42)
    assert torch.equal(a[perm], c)
    d = wide_sample_logits(logits[:5], rid[:5], pos[:5], temp[:5], salt=42)
    assert torch.equal(a[:5], d)
    assert not torch.equal(a, wide_sample_logits(logits, rid, pos, temp, salt=43))


def test_row_law_chi_square():
    # One row of V=256 sampled under many keys: frequencies match softmax (bulk of the law).
    V, n = 256, 1 << 20
    z = torch.randn(V, device="cuda", dtype=torch.float32) * 2
    logits = z.repeat(n, 1)
    rid = torch.arange(n, device="cuda", dtype=torch.int64)
    y = wide_sample_logits(logits, rid, torch.zeros(n, device="cuda", dtype=torch.int64),
                           torch.ones(n, device="cuda"), salt=7)
    cnt = torch.bincount(y, minlength=V).double()
    p = torch.softmax(z.double(), -1) * n
    keep = p > 5
    chi2 = float(((cnt - p)[keep] ** 2 / p[keep]).sum())
    df = int(keep.sum()) - 1
    print(f"[wide sampler] row law chi2={chi2:.1f} df={df}")
    assert chi2 < df + 6 * math.sqrt(2 * df)


def test_cost_per_token():
    V, rows = 131072, 256
    logits = torch.randn(rows, V, device="cuda")
    rid = torch.arange(rows, device="cuda", dtype=torch.int64)
    pos = torch.zeros(rows, device="cuda", dtype=torch.int64)
    temp = torch.ones(rows, device="cuda")
    for _ in range(3):
        wide_sample_logits(logits, rid, pos, temp, salt=1)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        wide_sample_logits(logits, rid, pos, temp, salt=1)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t) / 20 * 1e6
    line = f"[wide sampler] V={V} rows={rows}: {us:.0f} us per decode step ({us / rows:.2f} us per token)"
    try:
        import flashinfer

        gen = torch.Generator(device="cuda").manual_seed(0)
        for _ in range(3):
            flashinfer.sampling.sampling_from_logits(logits, deterministic=True, generator=gen)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(20):
            flashinfer.sampling.sampling_from_logits(logits, deterministic=True, generator=gen)
        torch.cuda.synchronize()
        line += f"; flashinfer {(time.perf_counter() - t) / 20 * 1e6:.0f} us"
    except ImportError:
        pass
    print(line)
