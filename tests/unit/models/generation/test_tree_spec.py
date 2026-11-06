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
"""tree_spec (exact tree speculation): host width rule (CPU), device candidates + trie == the Python rule, the
accept walk == sample-and-match along the keyed draws (incl. EOS / budget / the accept_topk negative), and the
tree_kv copies (partial block, per-layer gather, commit) on a KV buffer."""

import numpy as np
import pytest
import torch

from nemo_rl.models.generation.megatron import tree_spec as ts

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------- host width rule (CPU)


def test_water_fill_critical_first_and_break_even():
    rates = np.array([0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005] + [0.001] * 56)
    # one long tail and short ones: nodes go to the long tail until its marginal node is below the cost
    w = ts.water_fill([8000.0, 1000.0, 500.0], rates, [64, 64, 64], 64, beta=0.004)
    assert w[0] > 0 and w[1] == 0 and w[2] == 0
    cum = 1 + np.cumsum(rates)
    k = w[0]
    assert rates[k - 1] / cum[k - 1] > 0.004 and rates[k] / cum[k] <= 0.004  # stopped at break-even
    # equal tails share; the budget caps the total
    w = ts.water_fill([4000.0, 4000.0], rates, [64, 64], 3, beta=0.0)
    assert sum(w) == 3 and abs(w[0] - w[1]) <= 1
    # nothing worth a node
    assert ts.water_fill([8000.0], np.full(64, 0.001), [64], 64, beta=0.004) == [0]
    # the critical tail at its cap: no node elsewhere
    assert ts.water_fill([8000.0, 10.0], rates, [2, 64], 64, beta=0.0) == [2, 0]


def test_plan_widths_minimizes_phase_time_with_ties():
    rates = np.array([0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005] + [0.001] * 56)
    cum = np.concatenate([[1.0], 1 + np.cumsum(rates)])
    # one critical tail: nodes only there
    w = ts.plan_widths([8000.0, 1000.0, 500.0], rates, [64, 64, 64], 64, beta=0.004)
    assert w[0] > 0 and w[1] == 0 and w[2] == 0
    # its phase time T (1 + beta N) is at the grid optimum: neither one node fewer nor one more is better
    f = lambda k: 8000.0 / cum[k] * (1 + 0.004 * k)  # noqa: E731
    assert f(w[0]) <= f(w[0] - 1) * 1.02 and f(w[0]) <= f(w[0] + 1) * 1.02
    # three tied tails: all get nodes (a lower phase time needs every one of them)
    w = ts.plan_widths([8000.0, 8000.0, 8000.0], rates, [64, 64, 64], 64, beta=0.004)
    assert min(w) > 0 and max(w) - min(w) <= 1
    # ties make nodes dearer: fewer per tail than a lone critical tail gets
    assert max(w) <= ts.plan_widths([8000.0], rates, [64], 64, beta=0.004)[0]
    # the budget binds; nothing worth a node; nothing left to decode
    assert sum(ts.plan_widths([8000.0] * 4, rates, [64] * 4, 6, beta=0.0)) <= 6
    assert ts.plan_widths([8000.0], np.full(64, 1e-5), [64], 64, beta=0.004) == [0]
    assert ts.plan_widths([0.0, 0.0], rates, [8, 8], 16) == [0, 0]


def test_rates_prior_chains_unobserved_indices():
    t = ts.TreeSpec.__new__(ts.TreeSpec)
    t.accA = np.zeros(ts.WMAX)
    t.accN = np.zeros(ts.WMAX)
    r = ts.TreeSpec.rates(t)
    assert np.allclose(r, ts.PRIOR0)  # no evidence: the prior everywhere
    t.accN[:4] = 100.0
    t.accA[:4] = [60.0, 30.0, 10.0, 0.0]
    r = ts.TreeSpec.rates(t)
    assert r[0] == pytest.approx((ts.PRIOR0 + 60) / 101) and r[3] < 0.01 and r[4] == pytest.approx(r[3])


def test_eviction_releases_tail_state_and_readmission_registers_fresh():
    """Review T1: an evicted request leaves like a finished one (its tid freed, not tracked, unseen), so its
    re-admission's prefill registers it again with fresh tables."""
    t = ts.TreeSpec.__new__(ts.TreeSpec)
    t.t_max, t.W, t.D, t.S = 4, 8, 4, 1000
    t.tails, t._seen, t._sorted = {}, set(), None
    t.free_tids = [3, 2, 1, 0]
    t._rid_of = np.full(4, -1, dtype=np.int64)
    t._trk = np.zeros(4, dtype=bool)
    t._wlast = np.zeros(4, dtype=np.int64)
    t.stats = {"registered": 0}
    t.ctx = type("C", (), {"max_sequence_length": 900})()
    t._epoch, t.mtp, t.mtp_len = 0, None, torch.zeros(5, dtype=torch.int32)

    class _SP:
        num_tokens_to_generate, num_tokens_total, termination_id = 100, None, 2
        ts_mtd_refs = ts.pack_refs([[5, 6, 7]]).tobytes()

    req = type("R", (), {"sampling_params": _SP(), "prompt_tokens": [1, 2, 3]})()
    for rid in (11, 12):
        t._seen.add(rid)
        tail = t.maybe_register(rid, req)
        tail.tracked = True
        t._trk[tail.tid] = True
    tid11 = t.tails[11].tid
    t.on_post_process(np.asarray([], dtype=np.int64), np.asarray([11]))  # 11 evicted
    assert 11 not in t.tails and 11 not in t._seen and not t._trk[tid11] and t._rid_of[tid11] == -1
    assert tid11 in t.free_tids and 12 in t.tails
    again = t.maybe_register(11, req)  # re-admitted: a fresh tail
    assert again is not None and not again.tracked and not again.refs_loaded
    t.on_post_process(np.asarray([12]), None)  # 12 finished
    assert 12 not in t.tails


def _spec(neg=""):
    """A TreeSpec with only the host state registration / epochs / liveness read."""
    t = ts.TreeSpec.__new__(ts.TreeSpec)
    t.t_max, t.W, t.D, t.S, t.neg = 4, 8, 4, 1000, neg
    t.tails, t._seen, t._sorted = {}, set(), None
    t.free_tids = [3, 2, 1, 0]
    t._rid_of = np.full(4, -1, dtype=np.int64)
    t._trk = np.zeros(4, dtype=bool)
    t._wlast = np.zeros(4, dtype=np.int64)
    t._mslot_host = np.full(4, -1, dtype=np.int64)
    t.stats = {"registered": 0, "epochs": 0, "epoch_released": 0, "epoch_kept": 0, "cut_dropped": 0, "tails_full": 0}
    t.ctx = type("C", (), {"max_sequence_length": 900})()
    t._epoch, t.mtp, t.mtp_len, t.mtp_free = 0, None, torch.zeros(5, dtype=torch.int32), []
    t.engine = type("E", (), {"get_request": staticmethod(lambda rid: None), "_weight_epoch": 0})()
    return t


class _SP:
    num_tokens_to_generate, num_tokens_total, termination_id = 100, None, 2
    ts_mtd_refs = ts.pack_refs([[5, 6, 7]]).tobytes()


_REQ = type("R", (), {"sampling_params": _SP(), "prompt_tokens": [1, 2, 3]})()


def _ctx(ids):
    return type("Ctx", (), {"request_ids": torch.tensor(ids, dtype=torch.int32), "total_request_count": len(ids)})()


def test_new_weight_epoch_keeps_resident_tails_and_releases_rebuilt_requests():
    """Lag K: at the first layout of a new engine weight epoch a resident request's tail (decode row or paused) keeps
    its version-free state (tracked, same tail object: no host re-sync) with its key id unresolved and its MTP slot
    (old-weights K / V) dropped; a request MCore rebuilt (re-added for a prefill by a RECOMPUTE resume, or not resident
    yet) is released and unseen, so its prefill registers it again. NEGATIVE CONTROL tree_stale_version keeps every
    old tail."""
    t = _spec()
    for rid in (11, 12, 13, 14):
        t._seen.add(rid)
        t.maybe_register(rid, _REQ).kid = 77
    for rid in (11, 12, 14):
        t.tails[rid].tracked = True
    t.tails[11].mslot, t._mslot_host[t.tails[11].tid] = 5, 5
    t.mtp_len[t.tails[11].tid] = 3
    keep11, keep12 = t.tails[11], t.tails[12]
    # context: 12 paused (index 0), 11 decode, 14 re-added for a prefill; 13 not resident (waiting to be re-added)
    rids = np.asarray([11, 14], dtype=np.int64)
    t._new_epoch(_ctx([12, 11, 14]), rids, 1, 1, epoch=1)
    assert t._epoch == 1 and t.stats["epochs"] == 1
    assert t.stats["epoch_kept"] == 2 and t.stats["epoch_released"] == 2
    assert t.tails[11] is keep11 and t.tails[12] is keep12 and keep11.tracked and keep12.tracked
    assert keep11.kid == -1 and keep12.kid == -1
    assert keep11.mslot == -1 and t.mtp_free == [5] and int(t.mtp_len.sum()) == 0
    assert set(t.tails) == {11, 12} and t._seen == {11, 12}
    again = t.maybe_register(14, _REQ)  # the re-added request's prefill: a fresh tail
    assert again is not None and not again.tracked and again.kid == -1
    n = _spec("tree_stale_version")
    for rid in (11, 14):
        n.maybe_register(rid, _REQ)
    n._new_epoch(_ctx([11, 14]), rids, 1, 0, epoch=1)
    assert n._epoch == 1 and set(n.tails) == {11, 14}


def test_layout_commits_only_for_its_registered_tail():
    """A layout's tail commits only while it is the request's registered tail object: a forward laid out before a
    resume and sampled after it commits for a kept tail (it ran on the old weights entirely, as its root row did), not
    for a released / re-registered one."""
    t = _spec()
    tail = t.maybe_register(11, _REQ)
    lay = ts._Forward(slot=0, tails=[(tail, 0, 4, 0)], n_rows=1, E=4, epoch=0)
    t.engine._weight_epoch = 1
    assert t._live(lay, tail)  # straddling forward, kept tail
    t._new_epoch(_ctx([11]), np.asarray([11]), 0, 0, epoch=1)  # 11 is a prefill row now: rebuilt, released
    assert not t._live(lay, tail)
    again = t.maybe_register(11, _REQ)
    assert not t._live(lay, tail) and t._live(ts._Forward(slot=1, tails=[(again, 0, 4, 0)], n_rows=1, E=4, epoch=1),
                                              again)


class _MainEngine:
    """Megatron-LM main's shape: finished rows applied by the controller's finalize callback through
    _process_request_outputs (keywords), survivors by post_process_requests (positional, through the same method)."""

    def __init__(self):
        self.seen = []

    def _process_request_outputs(self, request_ids, finished_request_ids, sample, *a, **kw):
        self.seen.append(sample.tolist())

    def post_process_requests(self, request_ids, finished_request_ids, evict_request_ids, step_time, sample,
                              accepted_tokens, log_probs, *a, **kw):
        self._process_request_outputs(request_ids, finished_request_ids, sample)


class _OldEngine:
    def __init__(self):
        self.seen = []

    def post_process_requests(self, request_ids, finished_request_ids, evict_request_ids, step_time, sample,
                              accepted_tokens, log_probs, *a, **kw):
        self.seen.append(sample.tolist())


@pytest.mark.parametrize("main", [True, False])
def test_tails_tokens_reach_finished_and_surviving_rows(main):
    """Regression (main-pin in-loop: a finished tail's log-probs outnumbered its tokens by a - 1): the tails'
    accepted tokens replace the sample wherever MCore applies it, the finalize path of finished rows included."""
    t = _spec()
    t.on_post_process = lambda fin, ev: None
    eng = _MainEngine() if main else _OldEngine()
    ts.install_sample_substitution(eng, t)
    ids, sample = torch.tensor([11, 12]), torch.tensor([5, 9])
    t._res = {11: ([5, 6, 7], [0.1, 0.2], True, 10)}
    if main:  # the finalize callback (keywords) for finished row 11, then post-processing of the survivors
        eng._process_request_outputs(request_ids=ids[:1], finished_request_ids=ids[:1], sample=sample[:1])
        assert eng.seen[-1] == [[5, 6, 7]]
    eng.post_process_requests(ids, ids[:1], None, 0.0, sample, None, None)
    assert eng.seen[-1] == [[5, 6, 7], 9] and t._res is None
    t._res = {}
    eng.post_process_requests(ids, ids[:0], None, 0.0, sample, None, None)
    assert eng.seen[-1] == [5, 9]  # no multi-token tail: MCore's own sample object


def test_wake_cut_rows_commit_no_tree_tokens():
    """Rows MCore finishes without the step's token (stop-word path: a wake cut) are dropped from the resolve: no list
    sample, no log-probs, no key registered."""
    t = _spec()
    t._res = {11: ([5, 6, 7], [0.1, 0.2], False, 10), 12: ([8], [], False, 20)}
    t._drop_cut_rows({11, 99})
    assert set(t._res) == {12} and t.stats["cut_dropped"] == 1
    t._drop_cut_rows(None)
    t._drop_cut_rows(set())
    assert set(t._res) == {12}


def test_tail_ids_exhausted_is_counted():
    t = _spec()
    for rid in range(10, 15):
        t.maybe_register(rid, _REQ)
    assert len(t.tails) == 4 and t.stats["tails_full"] == 1


def test_set_width_cap():
    """set_width_cap clamps to [0, W] on a live tail and reports unknown requests."""
    t = ts._Tail(rid=5, tid=0, plen=3, budget_end=100, eos=-1, refs=None, w_cap=8, d_cap=4)
    spec = object.__new__(ts.TreeSpec)
    spec.tails, spec.W = {5: t}, 16
    assert spec.set_width_cap(5, 0) and t.w_cap == 0
    assert spec.set_width_cap(5, 99) and t.w_cap == 16
    assert not spec.set_width_cap(6, 3)


def _stub_spec(n_tids=4):
    spec = object.__new__(ts.TreeSpec)
    spec.tails, spec.W, spec.D, spec.S = {}, 16, 4, 4096 + 8 * ts.LOOK_MAX + 8
    spec.free_tids = list(range(n_tids - 1, -1, -1))
    spec._rid_of = np.full(n_tids, -1, dtype=np.int64)
    spec._trk = np.zeros(n_tids, dtype=bool)
    spec._wlast = np.zeros(n_tids, dtype=np.int64)
    spec._sorted, spec._seen, spec.stats = None, set(), {"registered": 0}
    spec.ctx = type("C", (), {"max_sequence_length": 4096})()
    return spec


def test_register_tail_for_a_decoding_request():
    """register_tail makes an already-decoding request a tail (packed refs, caps clamped); unknown / repeated
    requests and a full tail table are refused."""
    sp = type("SP", (), {"num_tokens_to_generate": 100, "num_tokens_total": None, "termination_id": 2})()
    req = type("R", (), {"sampling_params": sp, "prompt_tokens": [5] * 30})()

    class Eng:
        def get_request(self, rid):
            if rid != 7:
                raise KeyError(rid)
            return req

    spec = _stub_spec(n_tids=1)
    spec.engine = Eng()
    assert spec.register_tail(7, [[1, 2, 3], [4, 5]], w=99, d=2)
    t = spec.tails[7]
    assert t.w_cap == 16 and t.d_cap == 2 and t.budget_end == 130 and t.eos == 2 and not t.own
    assert np.array_equal(np.frombuffer(t.refs, dtype=np.int32), ts.pack_refs([[1, 2, 3], [4, 5]]))
    assert 7 in spec._seen and spec._rid_of[t.tid] == 7 and not spec._trk[t.tid]
    assert not spec.register_tail(7, [[1]], w=4)  # already a tail
    assert not spec.register_tail(8, [[1]], w=4)  # unknown request
    spec.engine = type("E2", (), {"get_request": lambda self, rid: req})()
    assert not spec.register_tail(9, [[1]], w=4)  # no free tail id


def test_epoch_map_counts_blocks_registered_under_an_older_epoch():
    """install_epoch_map: blocks keep the epoch they were registered under until re-allocated."""

    class Alloc:
        def register_kv_block_hashes(self, block_ids, block_hashes, parent_hashes=None):
            return list(block_ids)

        def allocate_memory_blocks(self, n):
            return torch.arange(10, 10 + n)

    eng = type("E", (), {})()
    eng._weight_epoch = 0
    eng.context = type("C", (), {"kv_block_allocator": Alloc()})()
    spec = object.__new__(ts.TreeSpec)
    spec._blk_epoch = {}
    ts.install_epoch_map(eng, spec)
    eng.context.kv_block_allocator.register_kv_block_hashes([3, 4, 10], [1, 2, 3])
    eng._weight_epoch = 1
    eng.context.kv_block_allocator.register_kv_block_hashes([5], [9])
    assert spec._blk_epoch == {3: 0, 4: 0, 10: 0, 5: 1}
    eng.context.kv_block_allocator.allocate_memory_blocks(1)  # block 10 re-allocated: its old epoch is gone
    assert 10 not in spec._blk_epoch and spec._blk_epoch[3] == 0


def test_mtp_depth_rule():
    """The MTP draft depth stops at the first depth whose marginal node is not worth one depth's price."""
    r = np.array([0.71, 0.17, 0.025, 0.001, 0.0])
    assert ts.mtp_depth(r, 0.03, 4) == 2  # 0.71 / 1 > .03, 0.17 / 1.71 = 0.099 > .03, 0.025 / 1.88 = 0.013 < .03
    assert ts.mtp_depth(r, 0.005, 4) == 3
    assert ts.mtp_depth(r, 0.0, 4) == 4  # every depth up to kmax gains something
    assert ts.mtp_depth(r, 0.0, 2) == 2
    assert ts.mtp_depth(r, 0.8, 4) == 0
    assert ts.mtp_depth(np.zeros(4), 0.0, 4) == 0


# ---------------------------------------------------------------------------- Python reference of the device rule


def py_tree(h, refs, p, w, dmax, nmax=4, cmax=32, mtp=()):
    """Candidates (n desc, reference position asc) of the suffix match (n <= nmax) of h[:p] in refs, first cmax,
    inserted best first into a trie of <= w nodes and depth <= dmax. Returns [(token, parent, depth)]."""
    rl = len(refs)
    cands = []
    for e in range(1, rl):
        if refs[e] < 0:
            continue
        n = 0
        for d in range(nmax):
            kp, pos = p - 1 - d, e - 1 - d
            if kp < 0 or pos < 0 or h[kp] < 0 or refs[pos] != h[kp]:
                break
            n += 1
        if n:
            cands.append((-n, e))
    cands.sort()
    nodes = []
    cur = -1
    for d, x in enumerate(mtp, start=1):  # candidate 0: the MTP chain
        if d > dmax or x < 0 or len(nodes) >= w:
            break
        nodes.append((x, cur, d))
        cur = len(nodes) - 1
    for _negn, e in cands[:cmax]:
        cur = -1
        for d in range(1, ts.LOOK_MAX):
            pos = e + d - 1
            x = refs[pos] if pos < rl else -1
            if d > dmax or x < 0:
                break
            hit = [i for i, (tk, pr, _d) in enumerate(nodes) if pr == cur and tk == x]
            if hit:
                cur = hit[0]
            elif len(nodes) < w:
                nodes.append((x, cur, d))
                cur = len(nodes) - 1
            else:
                break
    return nodes


def _device_trees(H, R, RLEN, P, BUD, lay, D, chunk=128, bs=256, mtp=None, own=()):
    T = lay.shape[1]
    nb = -(-max(R.shape[1], H.shape[1] if own else 0) // ts.BLOCK)  # own-history tails: blocks cover H too
    dev = H.device
    i32 = dict(dtype=torch.int32, device=dev)
    OWN = torch.zeros(T, **i32)
    for j in own:
        OWN[j] = 1
    bcnt = torch.zeros(T, nb, ts.NMAX, **i32)
    cand = torch.zeros(T, ts.CMAX, **i32)
    ncand = torch.zeros(T, **i32)
    ntok, npar, ndep = (torch.zeros(T, ts.WMAX, **i32) for _ in range(3))
    nn, pout = torch.zeros(T, **i32), torch.zeros(T, **i32)
    import triton

    ts._count_kernel[(T, nb)](H, R, RLEN, P, OWN, lay, bcnt, H.stride(0), R.stride(0), lay.stride(0), nb,
                              BLOCKC=ts.BLOCK, NMAXC=ts.NMAX)
    ts._cand_kernel[(T, nb)](H, R, RLEN, P, OWN, lay, bcnt, cand, ncand, H.stride(0), R.stride(0), lay.stride(0), nb,
                             BLOCKC=ts.BLOCK, NMAXC=ts.NMAX, NBPOW=triton.next_power_of_2(nb), CMAXC=ts.CMAX)
    mtok = torch.full((T, ts.KMTP), -1, **i32)
    mlen = torch.zeros(T, **i32)
    for j, chain in (mtp or {}).items():
        mtok[j, : len(chain)] = torch.tensor(chain, dtype=torch.int32)
        mlen[j] = len(chain)
    nm = torch.zeros(T, dtype=torch.int64, device=dev)
    ts._build_kernel[(T,)](H, R, RLEN, P, OWN, BUD, lay, cand, ncand, ntok, npar, ndep, nn, pout, mtok, mlen, nm,
                           H.stride(0), R.stride(0), lay.stride(0), mtok.stride(0), D, CHUNK=chunk, BS=bs,
                           WPOW=ts.WMAX, CMAXC=ts.CMAX, LOOK=ts.LOOK_MAX, ALWAYS=False, KM=ts.KMTP)
    _device_trees.nm = nm.cpu()
    return ntok.cpu(), npar.cpu(), ndep.cpu(), nn.cpu(), pout.cpu()


@cuda
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("vocab", [6, 40])
def test_device_tree_equals_python_rule(seed, vocab):
    rng = np.random.default_rng(seed)
    T, S, RC = 6, 400, 3000
    H = torch.zeros(T, S, dtype=torch.int32)
    R = torch.full((T, RC), -1, dtype=torch.int32)
    RLEN = torch.zeros(T, dtype=torch.int32)
    P = torch.zeros(T, dtype=torch.int32)
    BUD = torch.full((T,), 10_000, dtype=torch.int32)
    lay = torch.full((8, T), -1, dtype=torch.int32)
    cases = []
    for j in range(T):
        p = int(rng.integers(5, S - 20))
        h = rng.integers(0, vocab, p).astype(np.int32)
        refs = []
        for _ in range(int(rng.integers(1, 5))):
            refs += list(rng.integers(0, vocab, int(rng.integers(50, 600))).astype(np.int32)) + [-1]
        refs = np.asarray(refs[:RC], dtype=np.int32)
        w = int(rng.choice([1, 4, 16, 64]))
        H[j, :p] = torch.from_numpy(h)
        R[j, : refs.size] = torch.from_numpy(refs)
        RLEN[j] = refs.size
        P[j] = p
        lay[0, j], lay[2, j], lay[6, j] = j, w, 0
        cases.append((h, refs, p, w))
    BUD[2] = int(P[2]) + 3  # budget room caps the depth: dmax = bud - 1 - p = 2
    dev = torch.device("cuda")
    mtp = {j: [int(x) for x in rng.integers(0, vocab, int(rng.integers(1, ts.KMTP + 1)))] for j in range(0, T, 2)}
    own = (1, 4)  # src bit1 (XS-SD): candidates from the tail's own history h[:p] (tail 4 also has an MTP chain)
    for D in (4, 7):
        out = _device_trees(H.to(dev), R.to(dev), RLEN.to(dev), P.to(dev), BUD.to(dev), lay.to(dev), D, mtp=mtp,
                            own=own)
        ntok, npar, ndep, nn, pout = out
        for j, (h, refs, p, w) in enumerate(cases):
            dmax = min(D, 128 - 2 - (p - 1) % 128, 256 - 1 - (p - 1) % 256, int(BUD[j]) - 1 - p, ts.LOOK_MAX - 1)
            src = list(h[:p]) if j in own else list(refs)
            ref = py_tree(list(h), src, p, w, dmax, mtp=mtp.get(j, ()))
            got = [(int(ntok[j, i]), int(npar[j, i]), int(ndep[j, i])) for i in range(int(nn[j]))]
            assert got == ref, (j, D, got[:6], ref[:6])
            assert int(pout[j]) == p
            # MTP nodes are the first insertion indices: as many as the chain contributed (the Python rule's prefix)
            n_mtp = 0
            for x, (tk, pr, dd) in zip(mtp.get(j, ()), ref):
                if tk != x or dd != n_mtp + 1:
                    break
                n_mtp += 1
            assert int(_device_trees.nm[j]) == n_mtp, (j, D, int(_device_trees.nm[j]), n_mtp)


# ---------------------------------------------------------------------------- accept walk


def py_accept(nodes, s_root, s_node, eos, bud, p, alt_root=None, alt_node=None):
    """Sample-and-match along the keyed draws: emitted = draws on the path; cut after EOS; <= bud - p tokens."""
    em, cur, s_cur, a_cur, path = [s_root], -1, s_root, alt_root, []
    for _d in range(1, ts.LOOK_MAX):
        hit = [i for i, (tk, pr, _dd) in enumerate(nodes) if pr == cur and (tk == s_cur or (alt_root is not None and tk == a_cur))]
        if not hit:
            break
        i = max(hit)
        path.append(i)
        em[-1] = nodes[i][0]
        em.append(s_node[i])
        cur, s_cur = i, s_node[i]
        a_cur = alt_node[i] if alt_node is not None else None
    a = len(em)
    for k, t in enumerate(em):
        if t == eos:
            a = min(a, k + 1)
            break
    a = max(1, min(a, bud - p))
    return em[:a], path[: a - 1]


@cuda
@pytest.mark.parametrize("seed", list(range(6)))
@pytest.mark.parametrize("topk", [False, True])
@pytest.mark.parametrize("root_only", [False, True])
def test_accept_walk_equals_reference(seed, topk, root_only):
    """The device walk == the Python rule; root_only (a forward sampled after an engine resume): a = 1, the root's draw
    only, no path, the slot kept for the Mamba cursor commit."""
    rng = np.random.default_rng(seed)
    T, W = 5, ts.WMAX
    n_rows = 7
    dev = torch.device("cuda")
    i32 = dict(dtype=torch.int32)
    H = torch.zeros(T, 600, **i32)
    P = torch.zeros(T, **i32)
    BUD = torch.full((T,), 10_000, **i32)
    EOS = torch.full((T,), 3, **i32)
    DONE = torch.zeros(T, **i32)
    lay = torch.full((8, T), -1, **i32)
    ntok, npar = torch.full((T, W), -1, **i32), torch.full((T, W), -2, **i32)
    nn = torch.zeros(T, **i32)
    s_ext = torch.full((64,), -9, dtype=torch.int64)
    alt_ext = torch.full((64,), -1, dtype=torch.int64)
    s0 = torch.zeros(16, dtype=torch.int64)
    alt0 = torch.full((16,), -1, dtype=torch.int64)
    row_now = torch.full((T,), -1, **i32)
    pin = torch.zeros(T, **i32)
    x = n_rows
    expect = []
    for j in range(T):
        p = int(rng.integers(10, 400))
        nodes = []
        k = int(rng.integers(1, 12))
        for i in range(k):
            par = int(rng.integers(-1, i)) if i else -1
            dd = 1 if par < 0 else nodes[par][2] + 1
            if dd > 6:
                par, dd = -1, 1
            nodes.append((int(rng.integers(0, 5)), par, dd))
        sr = int(rng.integers(0, 5))
        sn = [int(v) for v in rng.integers(0, 5, k)]
        ar = int(rng.integers(0, 5)) if topk else None
        an = [int(v) for v in rng.integers(0, 5, k)] if topk else None
        if j == 3:
            BUD[j] = p + 2  # budget cut
        lay[0, j], lay[1, j], lay[2, j], lay[3, j], lay[4, j] = j, j, k, x, 10 + j
        P[j] = p
        pin[j] = p
        row_now[j] = j
        s0[j] = sr
        if topk:
            alt0[j] = ar
        for i, (tk, pr, dd) in enumerate(nodes):
            ntok[j, i], npar[j, i] = tk, pr
            s_ext[x - n_rows + i] = sn[i]
            if topk:
                alt_ext[x - n_rows + i] = an[i]
        nn[j] = k
        expect.append((nodes, sr, sn, p, ar, an, x))
        x += k
    priv = torch.arange(1000, 1064, **i32)
    lay[5, 0] = n_rows
    out = {k: v.to(dev) for k, v in dict(H=H, P=P, BUD=BUD, EOS=EOS, DONE=DONE, lay=lay, ntok=ntok, npar=npar, nn=nn,
                                          s_ext=s_ext, alt_ext=alt_ext, s0=s0, alt0=alt0, row_now=row_now, pin=pin,
                                          priv=priv).items()}
    acc = torch.zeros(T, dtype=torch.int32, device=dev)
    tok = torch.zeros(T, 8, dtype=torch.int64, device=dev)
    slots = torch.zeros(T, dtype=torch.int32, device=dev)
    cur = torch.zeros(T, dtype=torch.int32, device=dev)
    pidx = torch.zeros(T, 8, dtype=torch.int32, device=dev)
    pblk = torch.zeros(T, 8, dtype=torch.int32, device=dev)
    rblk = torch.zeros(T, dtype=torch.int32, device=dev)
    none_i = torch.full((64,), -1, dtype=torch.int64, device=dev)
    none_f = torch.zeros(64, dtype=torch.float32, device=dev)
    ts._accept_kernel[(T,)](
        out["H"], out["P"], out["BUD"], out["EOS"], out["DONE"], out["lay"], out["row_now"], out["s0"], out["s_ext"],
        out["alt0"], out["alt_ext"], none_i, none_i, none_f, none_f, none_i, none_i,
        out["ntok"], out["npar"], out["nn"], out["pin"], out["priv"],
        torch.full((1,), int(root_only), dtype=torch.int32, device=dev),
        acc, tok, slots, cur, pidx, pblk, rblk, torch.zeros(T, dtype=torch.int64, device=dev), out["H"].stride(0),
        out["lay"].stride(0), tok.stride(0),
        pidx.stride(0), CHUNK=128, BS=256, WPOW=W, LOOK=ts.LOOK_MAX, KPOW=8, TOPK=topk)
    for j, (nodes, sr, sn, p, ar, an, x0) in enumerate(expect):
        if root_only:
            assert int(acc[j]) == 1 and tok[j, 0].item() == sr and pidx[j].tolist() == [-1] * 8
            assert int(out["P"][j]) == p + 1 and int(out["H"][j, p]) == sr
            assert int(slots[j]) == 10 + j  # nodes ran: the cursor commit must run (a = 1: cursor w0 + 1)
            continue
        em, path = py_accept(nodes, sr, sn, 3, int(BUD[j]), p, ar, an)
        a = int(acc[j])
        assert a == len(em), (j, a, em)
        assert tok[j, :a].tolist() == em
        assert pidx[j, : a - 1].tolist() == [x0 - n_rows + i for i in path]
        assert pblk[j, : a - 1].tolist() == [1000 + x0 - n_rows + i for i in path]
        assert int(out["P"][j]) == p + a and out["H"][j, p: p + a].tolist() == em
        assert int(cur[j]) == (p - 1) % 128


# ---------------------------------------------------------------------------- tree_kv copies


@cuda
def test_tree_kv_partial_gather_and_commit():
    from nemo_rl.models.generation.megatron import tree_kv

    g = torch.Generator(device="cuda").manual_seed(0)
    L, NB, BS, HH, DD = 3, 12, 16, 2, 8
    buf = torch.randn(2, L, NB, BS, HH, DD, device="cuda", generator=g, dtype=torch.float32).to(torch.bfloat16)
    s = tree_kv._strides(buf)
    real, p = 2, 16 * 0 + 11  # root at position p - 1 = 10 of block `real`
    priv = [7, 8, 9]  # nodes: a (depth 1), b (depth 1), c (child of a, depth 2)
    part = torch.full((4, 3), -1, dtype=torch.int32, device="cuda")
    for i in range(3):
        part[i] = torch.tensor([real, priv[i], (p - 1) % BS])
    tree_kv._partial_copy_kernel[(4, L)](buf, part, *s, BLOCK=64)
    for i in range(3):
        assert torch.equal(buf[:, :, priv[i], : (p - 1) % BS], buf[:, :, real, : (p - 1) % BS])
    gath = torch.full((4, tree_kv.LOOK_MAX, 3), -1, dtype=torch.int32, device="cuda")
    for i in range(3):
        gath[i, 0] = torch.tensor([real, priv[i], (p - 1) % BS])  # root
    gath[2, 1] = torch.tensor([priv[0], priv[2], p % BS])  # c's ancestor a at depth 1
    for lyr in range(L):
        tree_kv.layer_gather(buf, gath, lyr)
    for i in range(3):
        assert torch.equal(buf[:, :, priv[i], (p - 1) % BS], buf[:, :, real, (p - 1) % BS])
    assert torch.equal(buf[:, :, priv[2], p % BS], buf[:, :, priv[0], p % BS])
    # commit: path a -> c accepted (a = 3 tokens): positions p, p + 1 <- a's and c's own K / V
    want_p, want_p1 = buf[:, :, priv[0], p % BS].clone(), buf[:, :, priv[2], (p + 1) % BS].clone()
    other = buf[:, :, real, (p + 2) % BS].clone()
    path_blk = torch.full((1, 8), -1, dtype=torch.int32, device="cuda")
    path_blk[0, :2] = torch.tensor([priv[0], priv[2]])
    ctx = type("C", (), {"memory_buffer": buf, "block_size_tokens": BS})()
    tree_kv.commit_copy(ctx, torch.tensor([real], dtype=torch.int32, device="cuda"),
                        torch.tensor([3], dtype=torch.int32, device="cuda"),
                        torch.tensor([p], dtype=torch.int32, device="cuda"), path_blk)
    assert torch.equal(buf[:, :, real, p % BS], want_p) and torch.equal(buf[:, :, real, (p + 1) % BS], want_p1)
    assert torch.equal(buf[:, :, real, (p + 2) % BS], other)


@cuda
def test_tree_kv_copies_large_pool_int64_offsets():
    """Regression (node-row log-probs off on full Nano at buffer_gb 30): with a KV pool where
    layer * layer_stride passes 2^31 elements, the per-layer gather (and the partial / commit copies) must still hit
    the right block of the right layer (int64 offsets), and write nothing else."""
    from nemo_rl.models.generation.megatron import tree_kv

    free = torch.cuda.mem_get_info()[0]
    L, BS, HH, DD = 6, 256, 2, 128
    NB = 6600  # layer 5 x NB x 256 x 2 x 128 > 2^31
    need = 2 * L * NB * BS * HH * DD * 2
    if free < need + (2 << 30):
        pytest.skip("needs ~11 GB of free GPU memory")
    buf = torch.zeros(2, L, NB, BS, HH, DD, device="cuda", dtype=torch.bfloat16)
    assert (L - 1) * buf.stride(1) >= 2**31
    s = tree_kv._strides(buf)
    real, priv, p = NB - 3, NB - 1, 11  # high block ids (as the allocator hands out on a large pool)
    for lyr in range(L):
        buf[:, lyr, real] = float(lyr + 1)
    part = torch.full((1, 3), -1, dtype=torch.int32, device="cuda")
    part[0] = torch.tensor([real, priv, (p - 1) % BS])
    tree_kv._partial_copy_kernel[(1, L)](buf, part, *s, BLOCK=1024)
    gath = torch.full((1, tree_kv.LOOK_MAX, 3), -1, dtype=torch.int32, device="cuda")
    gath[0, 0] = torch.tensor([real, priv, (p - 1) % BS])
    for lyr in range(L):
        tree_kv.layer_gather(buf, gath, lyr)
    for lyr in range(L):
        assert torch.equal(buf[:, lyr, priv, : p], buf[:, lyr, real, : p]), lyr
        assert (buf[:, lyr, priv, p:] == 0).all(), lyr
    assert int(torch.count_nonzero(buf[:, :, : real])) == 0  # nothing written below (a wrapped offset lands there)
    path_blk = torch.full((1, 8), -1, dtype=torch.int32, device="cuda")
    path_blk[0, 0] = priv
    buf[:, :, priv, p % BS] = -7.0
    ctx = type("C", (), {"memory_buffer": buf, "block_size_tokens": BS})()
    tree_kv.commit_copy(ctx, torch.tensor([real], dtype=torch.int32, device="cuda"),
                        torch.tensor([2], dtype=torch.int32, device="cuda"),
                        torch.tensor([p], dtype=torch.int32, device="cuda"), path_blk)
    assert (buf[:, :, real, p % BS] == -7.0).all()
    assert int(torch.count_nonzero(buf[:, :, : real])) == 0
    del buf


@cuda
@pytest.mark.parametrize("seed", list(range(4)))
def test_accept_walk_draft_prob_negative(seed):
    """Negative tree_accept_by_draft_prob: the first child is taken when its row's accept indicator is set, else the
    row's residual token is emitted and the walk stops (exact in law, not keyed-pathwise: the gates must catch it)."""
    rng = np.random.default_rng(100 + seed)
    W, T, n_rows = ts.WMAX, 4, 5
    dev = torch.device("cuda")
    i32 = dict(dtype=torch.int32)
    H, P = torch.zeros(T, 600, **i32), torch.zeros(T, **i32)
    BUD, EOS, DONE = torch.full((T,), 10_000, **i32), torch.full((T,), -5, **i32), torch.zeros(T, **i32)
    lay, row_now, pin = torch.full((8, T), -1, **i32), torch.full((T,), -1, **i32), torch.zeros(T, **i32)
    ntok, npar, nn = torch.full((T, W), -1, **i32), torch.full((T, W), -2, **i32), torch.zeros(T, **i32)
    s0, s_ext = torch.zeros(16, dtype=torch.int64), torch.full((64,), -9, dtype=torch.int64)
    q0, q_ext = torch.zeros(16), torch.zeros(64)
    r0, r_ext = torch.full((16,), -1, dtype=torch.int64), torch.full((64,), -1, dtype=torch.int64)
    x, exp = n_rows, []
    for j in range(T):
        k = int(rng.integers(1, 9))
        nodes = []
        for i in range(k):
            par = int(rng.integers(-1, i)) if i else -1
            nodes.append((int(rng.integers(0, 6)), par))
        p = int(rng.integers(10, 300))
        lay[0, j], lay[1, j], lay[2, j], lay[3, j], lay[4, j] = j, j, k, x, 3 + j
        P[j] = pin[j] = p
        row_now[j] = j
        s0[j] = int(rng.integers(0, 6))
        q0[j] = float(rng.integers(0, 2))
        r0[j] = int(rng.integers(100, 106))
        for i, (tk, pr) in enumerate(nodes):
            ntok[j, i], npar[j, i] = tk, pr
            s_ext[x - n_rows + i] = int(rng.integers(0, 6))
            q_ext[x - n_rows + i] = float(rng.integers(0, 2))
            r_ext[x - n_rows + i] = int(rng.integers(100, 106))
        nn[j] = k
        # reference walk
        em, cur, qc, rc = [int(s0[j])], -1, float(q0[j]), int(r0[j])
        for _d in range(1, ts.LOOK_MAX):
            kids = [i for i, (_t, pr) in enumerate(nodes) if pr == cur]
            if not kids:
                break
            c = min(kids)
            if qc <= 0:
                em[-1] = rc
                break
            e = x - n_rows + c
            em[-1] = nodes[c][0]
            em.append(int(s_ext[e]))
            cur, qc, rc = c, float(q_ext[e]), int(r_ext[e])
        exp.append((p, em))
        x += k
    lay[5, 0] = n_rows
    t = {k_: v.to(dev) for k_, v in dict(H=H, P=P, BUD=BUD, EOS=EOS, DONE=DONE, lay=lay, row_now=row_now, pin=pin,
                                         ntok=ntok, npar=npar, nn=nn, s0=s0, s_ext=s_ext, q0=q0, q_ext=q_ext,
                                         r0=r0, r_ext=r_ext).items()}
    none_i = torch.full((64,), -1, dtype=torch.int64, device=dev)
    acc = torch.zeros(T, dtype=torch.int32, device=dev)
    tok = torch.zeros(T, 8, dtype=torch.int64, device=dev)
    z = [torch.zeros(T, dtype=torch.int32, device=dev) for _ in range(3)]
    pidx, pblk = (torch.zeros(T, 8, dtype=torch.int32, device=dev) for _ in range(2))
    priv = torch.arange(64, dtype=torch.int32, device=dev)
    ts._accept_kernel[(T,)](
        t["H"], t["P"], t["BUD"], t["EOS"], t["DONE"], t["lay"], t["row_now"], t["s0"], t["s_ext"], none_i, none_i,
        none_i, none_i, t["q0"], t["q_ext"], t["r0"], t["r_ext"], t["ntok"], t["npar"], t["nn"], t["pin"], priv,
        torch.zeros(1, dtype=torch.int32, device=dev), acc, tok, z[0], z[1], pidx, pblk, z[2], torch.zeros(T, dtype=torch.int64, device=dev), t["H"].stride(0), t["lay"].stride(0), tok.stride(0), pidx.stride(0),
        CHUNK=128, BS=256, WPOW=W, LOOK=ts.LOOK_MAX, KPOW=8, TOPK=False, ACCEPT_ALL=False, DRAFTPROB=True)
    for j, (p, em) in enumerate(exp):
        assert tok[j, : int(acc[j])].tolist() == em, (j, tok[j].tolist(), em)
