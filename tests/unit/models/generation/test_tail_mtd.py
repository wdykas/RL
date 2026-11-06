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
"""Multi-token tail decode (tail_mtd): the width rule, reference packing (CPU), the device proposal rule == the
Python rule replayed in research/tail_decode/mtd_replay.py, and the sample-and-match acceptance (GPU)."""

import random

import numpy as np
import pytest
import torch

from nemo_rl.models.generation.megatron import tail_mtd as tm

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_width_rule():
    # no acceptance -> no extra rows; good acceptance -> wide; monotone in alpha
    assert tm.width(0.0, 100.0, 4) == 0
    assert tm.width(0.9, 1.0, 4) == 4
    ws = [tm.width(a, 1.0, 4) for a in (0.0, 0.05, 0.2, 0.5, 0.8, 0.95)]
    assert ws == sorted(ws) and ws[0] == 0 and ws[-1] == 4
    # the prior proposes rows for a new tail; ~50 failed rows stop them; a predictable tail keeps them
    assert tm.width(tm.PRIOR_ACC, tm.PRIOR_TRIALS, 4) >= 1
    assert tm.width(tm.PRIOR_ACC + 0.0, tm.PRIOR_TRIALS + 60.0, 4) == 0
    assert tm.width(tm.PRIOR_ACC + 30.0, tm.PRIOR_TRIALS + 40.0, 4) == 4


def test_width_break_even_filter():
    """plan's vectorized candidate filter: width > 0 exactly when the estimate is above BETA (no tail the rule would
    give rows to is skipped), for every K."""
    rng = random.Random(5)
    for _ in range(20000):
        n = rng.choice([1.0, 2.0, 10.0, 51.0, 1000.0]) * rng.random() + 1.0
        a = n * rng.choice([rng.random() * 0.03, rng.random()])
        for K in (1, 4, 7):
            w = tm.width(a, n, K)
            assert (w > 0) == (a > tm.BETA * (1.0 - 1e-9) * n) or abs(a / n - tm.BETA) < 1e-9, (a, n, K, w)


def test_pack_refs():
    r = tm.pack_refs([[5, 6, 7], [], [9]])
    assert r.dtype == np.int32 and r.tolist() == [5, 6, 7, -1, -1, 9, -1]
    assert tm.pack_refs([]).size == 0


def _py_proposal(H, refs, K, chunk, bud, nmax=4):
    """mtd_replay's rule: longest suffix match n (<= nmax) of the context in the references (never the context
    itself); ties: references in order, then the latest position; the k following tokens of that reference; k <= K,
    chunk room, budget."""
    p = len(H)
    k = max(0, min(K, chunk - 2 - (p - 1) % chunk, bud - 1 - p))
    best = None
    for rank, src in enumerate(refs):
        for n in range(min(nmax, p), 0, -1):
            key = H[p - n:]
            hit = [e for e in range(n, len(src)) if src[e - n:e] == key]
            if hit:
                if best is None or n > best[0]:
                    best = (n, rank, max(hit))
                break
    if best is None or k == 0:
        return []
    n, rank, e = best
    return list(refs[rank][e:e + k])


def _run_device(H_list, refs_list, K, chunk, buds, cut_refs=False):
    T = len(H_list)
    S = 4096
    dev = torch.device("cuda")
    H = torch.zeros(T, S, dtype=torch.int32, device=dev)
    ref_cap = 4096
    R = torch.full((T, ref_cap), -1, dtype=torch.int32, device=dev)
    RK = torch.full((T, ref_cap), 15, dtype=torch.int8, device=dev)
    RLEN = torch.zeros(T, dtype=torch.int32, device=dev)
    P = torch.zeros(T, dtype=torch.int32, device=dev)
    BUD = torch.tensor(buds, dtype=torch.int32, device=dev)
    for t, (h, refs) in enumerate(zip(H_list, refs_list)):
        H[t, : len(h)] = torch.tensor(h, dtype=torch.int32)
        P[t] = len(h)
        r = tm.pack_refs(refs)
        if cut_refs:  # references cut at ref_cap (no closing separator); stale tokens of an earlier tail after them
            r = r[:-1]
            R[t, r.size: r.size + 16] = 7
        if r.size == 0:
            r = np.full(1, -1, dtype=np.int32)
        rk = np.minimum(1 + np.concatenate([[0], np.cumsum(r[:-1] == -1)]), 14).astype(np.int8)
        R[t, : r.size] = torch.from_numpy(r)
        RK[t, : r.size] = torch.from_numpy(rk)
        RLEN[t] = r.size
    lay = torch.full((8, T), -1, dtype=torch.int32, device=dev)
    n_rows = T  # one MCore row per tail, extras after them
    x = n_rows
    for t in range(T):
        lay[0, t], lay[1, t], lay[2, t], lay[3, t], lay[4, t], lay[6, t] = t, t, K, x, t, 0
        x += K
    lay[5, 0] = n_rows
    best = torch.full((T,), -1, dtype=torch.int64, device=dev)
    BLOCK = 1024
    nb = (ref_cap + BLOCK - 1) // BLOCK
    tm._lookup_kernel[(T, nb)](H, R, RK, RLEN, P, lay[0], lay[2], best, H.stride(0), R.stride(0),
                               BLOCK=BLOCK, NMAXC=tm.NMAX)
    rows = n_rows + T * K + 8
    i64 = lambda: torch.zeros(rows, dtype=torch.int64, device=dev)  # noqa: E731
    i32 = lambda: torch.zeros(rows, dtype=torch.int32, device=dev)  # noqa: E731
    in_ids, pos_ids, blk = i64(), i64(), i64()
    loc, pir, rix, ql, kvl, mbi, lix, off, sid = (i32() for _ in range(9))
    btab = torch.arange(rows * 32, dtype=torch.int32, device=dev).view(rows, 32)
    chain = torch.full((rows, 8), -1, dtype=torch.int32, device=dev)
    prop = torch.full((T, 8), -1, dtype=torch.int32, device=dev)
    kk = torch.zeros(T, dtype=torch.int32, device=dev)
    pout = torch.zeros(T, dtype=torch.int32, device=dev)
    sids = torch.full((T, 8), -1, dtype=torch.int32, device=dev)
    tm._expand_kernel[(T,)](
        H, R, RLEN, P, BUD, best, lay, prop, kk, pout, sids, in_ids, pos_ids, blk, loc, pir, rix, ql, kvl, btab, mbi, lix,
        off, chain, sid, H.stride(0), R.stride(0), lay.stride(0), prop.stride(0), btab.stride(0), chain.stride(0),
        sids.stride(0), 32, 0, 1000, n_rows, K=K, CHUNK=chunk, BS=256, KPOW=8, BTCOLS=32, LOOK_MAX=8, ALWAYS=False,
    )
    out = dict(prop=prop.cpu(), kk=kk.cpu(), in_ids=in_ids.cpu(), pos=pos_ids.cpu(), kvl=kvl.cpu(), ql=ql.cpu(),
               off=off.cpu(), chain=chain.cpu(), mbi=mbi.cpu(), lay=lay, H=H, P=P, BUD=BUD, sids=sids, R=R)
    return out


@cuda
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_device_proposal_equals_python_rule(seed):
    rng = random.Random(seed)
    K, chunk = 4, 128
    H_list, refs_list, buds = [], [], []
    for t in range(12):
        vocab = rng.choice([5, 20, 1000])  # small vocab: many matches; large: few
        p = rng.randrange(5, 600)
        h = [rng.randrange(vocab) for _ in range(p)]
        refs = []
        for _ in range(rng.randrange(0, 4)):
            r = [rng.randrange(vocab) for _ in range(rng.randrange(0, 300))]
            if r and rng.random() < 0.5:  # plant the context's suffix in a reference
                n = rng.randrange(1, 5)
                at = rng.randrange(0, len(r))
                r[at:at] = h[-n:]
            refs.append(r)
        H_list.append(h)
        refs_list.append(refs)
        buds.append(p + rng.choice([2, 3, 50, 4000]))
    out = _run_device(H_list, refs_list, K, chunk, buds)
    for t in range(12):
        exp = _py_proposal(H_list[t], refs_list[t], K, chunk, buds[t])
        kk = int(out["kk"][t])
        got = out["prop"][t, :kk].tolist()
        assert got == exp, (t, got, exp)
        p = len(H_list[t])
        # row 0: the device's last token / position; extra rows live up to kk, masked after
        assert int(out["in_ids"][t]) == H_list[t][-1] and int(out["pos"][t]) == p - 1 and int(out["kvl"][t]) == p
        x0 = 12 + t * K
        for i in range(1, K + 1):
            e = x0 + i - 1
            live = i <= kk
            assert int(out["ql"][e]) == 1  # masked rows keep their query token (varlen alignment)
            assert int(out["off"][e]) == (i if live else 0)
            if live:
                assert int(out["in_ids"][e]) == exp[i - 1] and int(out["pos"][e]) == p - 1 + i
                assert int(out["kvl"][e]) == p + i and int(out["mbi"][e]) == t
                assert out["chain"][e, :i].tolist() == [t] + list(range(x0, x0 + i - 1))


@cuda
def test_no_own_context_source():
    """The context itself is never a source: a tail whose context repeats (the suffix recurs in it) but whose
    references do not contain the suffix gets no proposal; the same suffix in a reference proposes what follows it."""
    h = [1, 2, 3, 4, 5] * 20
    out = _run_device([h, h], [[[9, 9, 9]], [[8, 4, 5, 6, 7, 6, 3]]], 4, 128, [len(h) + 100] * 2)
    assert int(out["kk"][0]) == 0
    assert out["prop"][1, : int(out["kk"][1])].tolist() == [6, 7, 6, 3]


@cuda
def test_proposal_stops_at_reference_end():
    """A reference cut at ref_cap has no closing separator: the proposal stops at the reference's length and never
    reads the stale tokens after it."""
    h = list(range(100, 140))
    out = _run_device([h], [[[50, h[-2], h[-1], 60, 61]]], 4, 128, [len(h) + 100], cut_refs=True)
    assert out["prop"][0, : int(out["kk"][0])].tolist() == [60, 61]


@cuda
@pytest.mark.parametrize("case", ["all", "none", "partial", "eos", "budget"])
def test_accept_sample_and_match(case):
    """a = 1 + leading proposals equal to the sampled tokens; cut after EOS and at the budget; H / P / done."""
    K, chunk = 4, 128
    h = list(range(100, 140))
    p = len(h)
    prop = [7, 8, 9, 10]
    s_rows = {"all": [7, 8, 9, 10, 11], "none": [5, 8, 9, 10, 11], "partial": [7, 8, 4, 10, 11],
              "eos": [7, 2, 9, 10, 11], "budget": [7, 8, 9, 10, 11]}[case]
    bud = p + (3 if case == "budget" else 1000)
    out = _run_device([h], [[h[-1:] + prop]], K, chunk, [bud])
    dev = torch.device("cuda")
    lay = out["lay"]
    kk = int(out["kk"][0])
    exp_kk = min(K, bud - 1 - p)
    assert kk == exp_kk
    s0 = torch.tensor([s_rows[0]], dtype=torch.int64, device=dev)
    s_ext = torch.tensor(s_rows[1:] + [0] * 4, dtype=torch.int64, device=dev)
    EOS = torch.tensor([2], dtype=torch.int32, device=dev)
    DONE = torch.zeros(1, dtype=torch.int32, device=dev)
    row_now = torch.zeros(1, dtype=torch.int32, device=dev)
    pin = torch.tensor([p], dtype=torch.int32, device=dev)
    acc = torch.zeros(1, dtype=torch.int32, device=dev)
    tok = torch.zeros(1, 8, dtype=torch.int64, device=dev)
    slots = torch.zeros(1, dtype=torch.int32, device=dev)
    sids = torch.zeros(1, 8, dtype=torch.int32, device=dev)
    cur = torch.zeros(1, dtype=torch.int32, device=dev)
    kk_t = torch.tensor([kk], dtype=torch.int32, device=dev)
    prop_t = torch.full((1, 8), -1, dtype=torch.int32, device=dev)
    prop_t[0, :kk] = torch.tensor(prop[:kk], dtype=torch.int32)
    tm._accept_kernel[(1,)](
        out["H"], out["P"], out["BUD"], EOS, DONE, lay, row_now, s0, s_ext, prop_t, kk_t, pin, out["sids"],
        acc, tok, slots, sids, cur, out["H"].stride(0), lay.stride(0), prop_t.stride(0), tok.stride(0),
        out["sids"].stride(0), K=K, KPOW=8, CHUNK=chunk,
    )
    a = int(acc[0])
    exp_a = {"all": 5, "none": 1, "partial": 3, "eos": 2, "budget": 3}[case]
    assert a == exp_a, (case, a)
    assert tok[0, :a].tolist() == s_rows[:a]
    assert int(out["P"][0]) == p + a
    assert out["H"][0, p: p + a].tolist() == s_rows[:a]
    assert int(DONE[0]) == (1 if case in ("eos", "budget") else 0)
    assert int(cur[0]) == (p - 1) % chunk
