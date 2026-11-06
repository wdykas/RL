"""ts-lawfix: 64-bit noise key of the 53-bit engine sampler (keyed_sampling.wide_sample_logits(key64=True)).

The 32-bit key mix32(...(salt, request id, position)) coincides for ~0.5 (request, position) pairs per 8 x 8k group;
two such tokens shared their Gumbel noise (same logits -> same token, always). With key64 the pair needs both words
to coincide. Tests: colliding 32-bit pairs found by a host replica of the key derivation draw identical tokens with
key64=False and independent tokens with key64=True; key64 row law (chi-square) and far tail (two-token race);
key64=False is the previous kernel (bitwise check against the frozen tree: research/lawfix/key64_bitwise.py).
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU test")

M32 = 0xFFFFFFFF


def _mix32(x):
    x ^= x >> 16
    x = (x * 0x7FEB352D) & M32
    x ^= x >> 15
    x = (x * 0x846CA68B) & M32
    return x ^ (x >> 16)


def key32(salt, rid, pos):
    """Host replica of _wide_gumbel_argmax_kernel's 32-bit key."""
    lo, hi = rid & M32, (rid >> 32) & M32
    s_lo, s_hi = salt & 0x7FFFFFFF, (salt >> 31) & 0x7FFFFFFF
    k = _mix32(lo ^ s_lo)
    k = _mix32(k ^ hi ^ s_hi)
    return _mix32(k ^ (pos & M32) ^ 0x5BD1E995)


def _np_mix32(x):
    import numpy as np

    x = x.astype(np.uint64)
    x ^= x >> np.uint64(16)
    x = (x * np.uint64(0x7FEB352D)) & np.uint64(M32)
    x ^= x >> np.uint64(15)
    x = (x * np.uint64(0x846CA68B)) & np.uint64(M32)
    return x ^ (x >> np.uint64(16))


def _colliding_pairs(salt, n_rid=1 << 17, pos_bits=6):
    """(request, position) pairs with equal 32-bit keys. key = mix32(k(rid) ^ pos ^ c) and mix32 is a
    bijection, so (rid, pos) and (rid', pos') collide iff k(rid) ^ k(rid') = pos ^ pos': rare for a given
    pair of requests, but then at EVERY position pos with pos ^ d in range (whole rows share noise, permuted)."""
    import numpy as np

    s_lo, s_hi = salt & 0x7FFFFFFF, (salt >> 31) & 0x7FFFFFFF
    rid = np.arange(n_rid, dtype=np.uint64)
    k = _np_mix32(_np_mix32(rid ^ np.uint64(s_lo)) ^ np.uint64(s_hi))  # hi word of rid = 0
    top = k >> np.uint64(pos_bits)
    order = np.argsort(top, kind="stable")
    pairs = []
    for a, b in zip(order[:-1], order[1:]):
        if top[a] == top[b]:
            d = int(k[a] ^ k[b])
            for pos in range(1 << pos_bits):
                pairs.append(((int(rid[a]), pos), (int(rid[b]), pos ^ d)))
    for (r1, p1), (r2, p2) in pairs[:50]:
        assert key32(salt, r1, p1) == key32(salt, r2, p2)
    return pairs[:4096]


if torch.cuda.is_available():
    import triton
    import triton.language as tl

    from thundersync_rl.keyed_sampling import _mix32 as _tmix32
    from thundersync_rl.keyed_sampling import _neg_log_exp53_kw, wide_sample_logits

    @triton.jit
    def _race_kernel64(out_ptr, log_r, n_per, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        wins = tl.zeros([BLOCK], tl.int64)
        for it in range(0, n_per, BLOCK):
            idx = (pid * n_per + it + tl.arange(0, BLOCK)).to(tl.uint32)
            key = _tmix32(idx ^ 0x1234567)
            key2 = _tmix32(idx ^ 0x7654321)
            g0 = _neg_log_exp53_kw(tl.zeros([BLOCK], tl.uint32), key, key2, True)
            g1 = _neg_log_exp53_kw(tl.full([BLOCK], 1, tl.uint32), key, key2, True)
            wins += (log_r + g1 > g0).to(tl.int64)
        tl.store(out_ptr + pid, tl.sum(wins, axis=0))


def test_host_key_replica_matches_kernel_collisions():
    salt = (0x1234ABCD << 32) | 0x5678
    pairs = _colliding_pairs(salt)
    assert len(pairs) >= 10, len(pairs)
    a = torch.tensor([p[0] for p in pairs] + [p[1] for p in pairs], device="cuda", dtype=torch.int64)
    rid, pos = a[:, 0].contiguous(), a[:, 1].contiguous()
    n = len(pairs)
    torch.manual_seed(0)
    z = torch.randn(n, 131072, device="cuda") * 0.5  # flat rows: chance agreement ~1e-5
    logits = torch.cat([z, z])
    temp = torch.ones(2 * n, device="cuda")
    y32 = wide_sample_logits(logits, rid, pos, temp, salt, key64=False)
    y64 = wide_sample_logits(logits, rid, pos, temp, salt, key64=True)
    same32 = int((y32[:n] == y32[n:]).sum())
    same64 = int((y64[:n] == y64[n:]).sum())
    print(f"[key64] {n} colliding 32-bit (request, position) pairs: identical tokens key32 {same32}/{n}, key64 {same64}/{n}")
    assert same32 == n  # the 32-bit key: the two tokens shared their noise
    assert same64 <= 1


def test_key64_row_law_chi_square():
    V, n = 256, 1 << 20
    z = torch.randn(V, device="cuda", dtype=torch.float32) * 2
    logits = z.repeat(n, 1)
    rid = torch.arange(n, device="cuda", dtype=torch.int64)
    y = wide_sample_logits(logits, rid, torch.zeros(n, device="cuda", dtype=torch.int64),
                           torch.ones(n, device="cuda"), salt=7, key64=True)
    cnt = torch.bincount(y, minlength=V).double()
    p = torch.softmax(z.double(), -1) * n
    keep = p > 5
    chi2 = float(((cnt - p)[keep] ** 2 / p[keep]).sum())
    df = int(keep.sum()) - 1
    print(f"[key64] row law chi2={chi2:.1f} df={df}")
    assert chi2 < df + 6 * math.sqrt(2 * df)


def test_key64_positions_and_requests_law():
    # Many (request, position) keys of ONE row law: the token counts follow softmax (keys vary in both words).
    V = 64
    z = torch.randn(V, device="cuda") * 1.5
    rid = torch.arange(1024, device="cuda", dtype=torch.int64).repeat_interleave(1024)
    pos = torch.arange(1024, device="cuda", dtype=torch.int64).repeat(1024)
    y = wide_sample_logits(z.repeat(rid.numel(), 1), rid, pos, torch.ones(rid.numel(), device="cuda"), salt=99,
                           key64=True)
    cnt = torch.bincount(y, minlength=V).double()
    p = torch.softmax(z.double(), -1) * rid.numel()
    keep = p > 5
    chi2 = float(((cnt - p)[keep] ** 2 / p[keep]).sum())
    df = int(keep.sum()) - 1
    print(f"[key64] (request, position) grid law chi2={chi2:.1f} df={df}")
    assert chi2 < df + 6 * math.sqrt(2 * df)


@pytest.mark.parametrize("r", [1e-3, 1e-7])
def test_key64_tail_two_token_race(r):
    programs, n_per = 4096, 1 << 20
    out = torch.zeros(programs, dtype=torch.int64, device="cuda")
    _race_kernel64[(programs,)](out, math.log(r), n_per, BLOCK=1024)
    wins, n = int(out.sum()), programs * n_per
    p = r / (1 + r)
    sd = math.sqrt(n * p * (1 - p))
    print(f"[key64] race r={r:.0e} N={n:.2e} wins={wins} expected={n * p:.1f} (sd {sd:.1f})")
    assert abs(wins - n * p) < 5 * sd + 1


def test_key64_batch_invariant_and_differs_from_key32():
    torch.manual_seed(1)
    logits = torch.randn(64, 131072, device="cuda") * 3
    rid = torch.arange(1000, 1064, device="cuda", dtype=torch.int64)
    pos = torch.randint(10, 9000, (64,), device="cuda", dtype=torch.int64)
    temp = torch.ones(64, device="cuda")
    a = wide_sample_logits(logits, rid, pos, temp, salt=42)
    perm = torch.randperm(64, device="cuda")
    assert torch.equal(a[perm], wide_sample_logits(logits[perm], rid[perm], pos[perm], temp[perm], salt=42))
    assert not torch.equal(a, wide_sample_logits(logits, rid, pos, temp, salt=42, key64=False))
