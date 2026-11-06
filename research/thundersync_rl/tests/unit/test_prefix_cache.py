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
"""Learner prefix-state cache: bookkeeping (CPU).

The bitwise property (cached forward == whole-row forward) needs GPU kernels:
tests/functional/check_prefix_cache.py (Nano L8 / L16, rejections, sticky anchors).
"""

from collections import OrderedDict

import pytest
import torch

from thundersync_rl.prefix_cache import PrefixCacheDirectory, lpt_owners

GIB = 2**30


# ---- driver: placement and membership ------------------------------------------------
def _directory(max_gb=1.0, kv=1024, state=2**20, chunk=128, **kw):
    d = PrefixCacheDirectory(gpu_gb=0.5, max_gb=max_gb, **kw)
    d.configure(
        [
            {"kv_bytes_per_token": kv, "state_bytes_per_boundary": state, "chunk": chunk, "max_keep": 4},
            {"kv_bytes_per_token": kv // 2, "state_bytes_per_boundary": 2 * state, "chunk": chunk, "max_keep": 4},
        ]
    )
    return d


def test_lpt_owners_fixed_rows_stay_and_count_as_load():
    owners = lpt_owners([10, 10, 10, 1], 2, fixed={3: 1, 0: 0})
    assert owners[3] == 1 and owners[0] == 0
    # rank 0 holds 10, rank 1 holds 1: the next longest goes to rank 1
    assert owners[1] == 1 and owners[2] == 0
    assert lpt_owners([5, 7, 3], 2) == lpt_owners([5, 7, 3], 2, {})


def test_configure_takes_largest_per_rank_cost():
    d = _directory()
    assert d.kv_bytes == 1024 and d.state_bytes == 2 * 2**20 and d.chunk == 128
    with pytest.raises(ValueError):
        PrefixCacheDirectory().configure(
            [{"kv_bytes_per_token": 1, "state_bytes_per_boundary": 1, "chunk": c} for c in (64, 128)]
        )


def test_continued_rows_keep_their_rank():
    d = _directory()
    keys = [d.new_key() for _ in range(4)]
    owners, sent = d.place(keys, [1000, 900, 800, 700], [872, 772, 672, 572], ranks=2)
    assert sent == keys
    # next round: same rows, lengths grew; owners are sticky whatever the balance says
    owners2, _ = d.place(keys[::-1], [828, 928, 1028, 1128], [700, 800, 900, 1000], ranks=2)
    assert owners2 == [owners[3], owners[2], owners[1], owners[0]]
    # load of a cached row is what it forwards: from its last boundary before the piece
    assert d.forwarded(keys[0], 1128, 1000) == 1128 - 896
    assert d.forwarded("new", 1128, 1000) == 1128


def test_budget_releases_least_recently_verified_rows():
    d = _directory(max_gb=0.01, policy="lru")  # ~10 MiB per rank
    per_row = d.entry_bytes(2000, 1872)
    assert per_row == 2000 * 1024 + 2 * 2 * 2**20
    keys = [d.new_key() for _ in range(3)]
    for k in keys:  # one rank: rows land on rank 0 one after the other
        d.place([k], [2000], [1872], ranks=1)
    # 3 x ~6 MiB > 10 MiB: the oldest rows were released, the newest kept
    rel = d.take_release()
    assert rel and rel[0] == keys[0] and keys[2] not in rel
    assert d.rank_bytes()[0] <= d.max_bytes
    assert d.take_release() == []
    # a row larger than the whole budget is not cached
    big = d.new_key()
    _, sent = d.place([big], [20000], [19872], ranks=1)
    assert sent == [None] and big not in d.owner


def _cyclic_hits(d, rows, rounds, ranks=1, n0=1000, c=128):
    """Continuation rounds visit every row in turn; hits = placements of a row already cached."""
    keys = [d.new_key() for _ in range(rows)]
    hits = 0
    for t in range(rounds):
        for k in keys:
            n = n0 + t * c
            was = k in d.owner
            _, sent = d.place([k], [n + c], [n], ranks=ranks)
            hits += int(was and sent[0] is not None)
    return hits / (rows * (rounds - 1))


def test_lru_thrashes_under_cyclic_rounds_and_pinning_does_not():
    """The 128k failure mode: working set above the budget."""
    kw = dict(kv=1024, state=2**20, row_cap_tokens=4000)
    lru = _directory(max_gb=0.03, policy="lru", **kw)
    pin = _directory(max_gb=0.03, policy="pin", **kw)
    per_row = pin.row_bytes()
    assert per_row == pin.entry_bytes(4000, 1) + 2 * 2**20
    fits = int(pin.max_bytes // per_row)
    assert 0 < fits < 10
    assert _cyclic_hits(lru, 10, 6) == 0.0  # every access misses
    frac = _cyclic_hits(pin, 10, 6)
    assert frac == pytest.approx(fits / 10)
    c = pin.take_counts()
    assert c["admitted"] == fits and c["evicted"] == 0 and c["denied"] == (10 - fits) * 6
    assert c["eligible"] == fits * 5
    assert lru.take_counts()["evicted"] > 0


def test_auto_budget_is_rows_per_rank_times_row_bytes_capped():
    d = _directory(max_gb=1.0, kv=1024, state=2**20, row_cap_tokens=4000)
    d.plan_step(rows=10, ranks=4)  # 3 rows per rank
    assert d.need_bytes == 3 * d.row_bytes() and d.max_bytes == d.need_bytes
    d.plan_step(rows=10_000, ranks=4)
    assert d.max_bytes == 1 * GIB and d.need_bytes > d.max_bytes
    f = _directory(max_gb=1.0, size="fixed", row_cap_tokens=4000)
    f.plan_step(rows=10, ranks=4)
    assert f.max_bytes == 1 * GIB
    with pytest.raises(ValueError):
        _directory(policy="mru")


def test_release_and_reset():
    d = _directory()
    k = d.new_key()
    d.place([k], [500], [372], ranks=2)
    d.release([k, None, "unknown"])
    assert d.take_release() == [k] and k not in d.owner
    d.place([k], [500], [372], ranks=2)
    d.reset()
    assert d.owner == {} and d.take_release() == []


# ---- learner: paged pools and boundary bookkeeping (no kernels) -------------------------
def _cache(chunk=4, max_keep=4, page=8, n_pages=64, n_slots=32, max_len=256):
    """A PrefixStateCache on CPU pools: one attention layer (1 head x 1 dim) and one Mamba layer."""
    from nemo_rl.models.megatron.prefix_state_cache import PrefixStateCache, _Pools

    c = PrefixStateCache.__new__(PrefixStateCache)
    c.chunk, c.max_keep, c.gpu_bytes, c.max_len = chunk, max_keep, 1 << 40, max_len
    c.entries = OrderedDict()
    c.weights_version = None
    c.attention_layers, c.mamba_layers = [1], [2]
    c.stats = {k: 0 for k in ("hits", "rows", "tokens_skipped", "tokens_forwarded", "v0_captured",
                              "v0_hits", "v0_trimmed", "v0_trim_dropped", "pool_fallback")}
    c.negctl, c.graphs, c.last_from_v0 = 0.0, True, []
    c.pools = _Pools({1: (1, 1, torch.float32)}, {2: (1, 1, torch.float32, 1, 1, 1)}, page, n_pages,
                     n_slots, -(-(max_len + page) // page), max_keep, torch.device("cpu"))
    return c


def _fwd(c, keys, rows, starts, floors=None, pins=None):
    """``cache.forward`` with the hooks emulated through the same device metadata.

    Before the writes it checks what a real forward reads: every cached position [0, start)
    of a row holds that row's token as K (page table), and the input state slot holds the
    state of boundary ``start`` (the zero slot for 0). Writes: K = token id at every
    forwarded position (``kv_write_index``), and each kept boundary's state = the boundary.
    """
    from nemo_rl.models.megatron.prefix_state_cache import TRASH_SLOT, _ACTIVE, kv_write_index

    with torch.no_grad(), c.forward(keys, rows, starts, pins, floors):
        call = _ACTIVE[0]
        meta, p = call.meta, c.pools
        s = max(r.length for r in call.rows)
        for b, r in enumerate(call.rows):
            assert int(meta["start"][b]) == r.start and int(meta["length"][b]) == r.length
            for pos in range(r.start):
                pg = int(meta["page_table"][b, pos // p.page])
                assert p.k[1][pg, pos % p.page, 0, 0] == float(r.tokens[pos]), (b, pos)
            assert float(p.ssm[2][int(meta["slot_in"][b])].item()) == float(r.start)
        valid, pg, off = kv_write_index(meta, s, p.page, p.maxp)
        tok = torch.zeros(len(call.rows), s)
        for b, r in enumerate(call.rows):
            tok[b, : r.length] = r.tokens[r.start :].float()
        p.k[1].index_put_((pg.reshape(-1), off.reshape(-1)), tok.reshape(-1, 1, 1))
        for b, r in enumerate(call.rows):
            for j in range(p.K):
                slot = int(meta["keep_slot"][b, j])
                if slot != TRASH_SLOT:
                    p.ssm[2][slot] = float(r.start + int(meta["keep_off"][b, j]))
    _check_ownership(c)


def _check_ownership(c):
    """Every page / slot is free or owned by exactly one resident entry (no leak, no sharing)."""
    from nemo_rl.models.megatron.prefix_state_cache import TRASH_PAGE, TRASH_SLOT

    p = c.pools
    owned_p = [pg for e in c.entries.values() if e.host is None for pg in e.pages]
    owned_s = [s for e in c.entries.values() if e.host is None for s in e.states.values()]
    assert len(owned_p) == len(set(owned_p)) and len(owned_s) == len(set(owned_s))
    assert sorted(owned_p + p.free_pages) == list(range(TRASH_PAGE + 1, p.n_pages))
    assert sorted(owned_s + p.free_slots) == list(range(TRASH_SLOT + 1, p.n_slots))
    for e in c.entries.values():
        if e.host is None:
            assert len(e.pages) == -(-e.kv_len // p.page)
            for b, slot in e.states.items():
                assert float(p.ssm[2][slot].item()) == float(b)


def test_keep_boundaries():
    c = _cache(chunk=4, max_keep=4)
    assert c.keep(0, 3) == []  # only boundary 0: nothing worth keeping
    assert c.keep(0, 9) == [0, 4, 8]  # boundary 9 itself is never usable
    assert c.keep(8, 12) == [8]
    # long pieces: start, last and evenly spaced in between
    assert c.keep(0, 64) == [0, 20, 40, 60]


def test_start_uses_matching_boundary_and_kv_is_the_prefix():
    c = _cache(chunk=4)
    row = torch.arange(100, 111)  # n = 11
    _fwd(c, ["a"], [row], [0])
    e = c.entries["a"]
    assert sorted(e.states) == [4, 8] and e.kv_len == 8 and len(e.pages) == 1
    nxt = torch.cat([row, torch.arange(200, 205)])
    assert c.start("a", nxt, 11) == 8
    rej = torch.cat([row[:6], torch.tensor([7]), torch.arange(300, 305)])
    assert c.start("a", rej, 7) == 4
    assert c.start("a", nxt, 4) == 0
    assert c.start("missing", nxt, 11) == 0
    assert c.start(None, nxt, 11) == 0


def test_continue_reuses_pages_and_the_start_slot():
    c = _cache(chunk=4, page=8)
    row = torch.arange(100, 111)
    _fwd(c, ["a"], [row], [0])
    slot8, page0 = c.entries["a"].states[8], c.entries["a"].pages[0]
    nxt = torch.cat([row, torch.arange(200, 207)])  # n = 18
    _fwd(c, ["a"], [nxt], [8])  # reads [0, 8) through the page table (checked in _fwd)
    e = c.entries["a"]
    assert sorted(e.states) == [8, 12, 16] and e.kv_len == 16 and e.states[8] == slot8
    assert e.pages[0] == page0 and len(e.pages) == 2
    assert torch.equal(e.tokens, nxt[:16])
    # a rejection inside the piece: back to boundary 12, positions >= 12 rewritten
    rej = torch.cat([nxt[:13], torch.tensor([9]), torch.arange(400, 410)])
    s = c.start("a", rej, 14)
    assert s == 12
    _fwd(c, ["a"], [rej], [s])
    assert sorted(c.entries["a"].states) == [12, 16, 20]


def test_pin_retains_the_floor_boundary_across_pieces():
    c = _cache(chunk=4)
    row = torch.arange(100, 111)
    _fwd(c, ["a"], [row], [0])  # boundaries 4, 8
    s4 = c.entries["a"].states[4]
    nxt = torch.cat([row, torch.arange(200, 207)])  # n = 18
    _fwd(c, ["a"], [nxt], [8], pins=[6])  # keeps 8, 12, 16 and the pinned 4
    assert sorted(c.entries["a"].states) == [4, 8, 12, 16] and c.entries["a"].states[4] == s4
    nxt2 = torch.cat([nxt, torch.arange(300, 306)])  # n = 24
    _fwd(c, ["a"], [nxt2], [16], pins=[6])
    assert sorted(c.entries["a"].states) == [4, 16, 20]
    assert c.start("a", nxt2[:20], 7) == 4
    _fwd(c, ["a"], [nxt2[:20]], [4])  # the scan from the pinned floor reads [0, 4) intact
    _fwd(c, ["a"], [nxt2], [16])
    assert sorted(c.entries["a"].states) == [16, 20]
    assert c.retain(None, 8, [8], 6) == [] and c.retain(c.entries["a"], 0, [0, 4], 6) == []


def test_weights_version_and_release_return_everything():
    c = _cache(chunk=4)
    _fwd(c, ["a", "b"], [torch.arange(10), torch.arange(20, 37)], [0, 0])
    c.set_weights_version(1)
    assert not c.entries
    _check_ownership(c)
    _fwd(c, ["a"], [torch.arange(10)], [0])
    c.set_weights_version(1)
    assert "a" in c.entries
    c.release(["a", "zz"])
    assert not c.entries
    _check_ownership(c)


def test_uncached_rows_use_pages_only_for_the_forward():
    c = _cache(chunk=4)
    free = len(c.pools.free_pages)
    _fwd(c, [None, "a"], [torch.arange(30), torch.arange(50, 61)], [0, 0])
    assert list(c.entries) == ["a"] and len(c.pools.free_pages) == free - 1


def test_spill_and_reload_keep_contents_under_a_small_pool():
    """Rows beyond the pools go to host memory and come back into other pages / slots."""
    c = _cache(chunk=4, page=8, n_pages=8, n_slots=10)
    gen = torch.Generator().manual_seed(1)
    rows = {f"r{i}": torch.randint(5, 1000, (20,), generator=gen) for i in range(4)}
    for k, r in rows.items():
        _fwd(c, [k], [r], [0])
    assert any(e.host is not None for e in c.entries.values())
    for _ in range(3):
        for k in list(rows):
            nxt = torch.cat([rows[k], torch.randint(5, 1000, (4,), generator=gen)])
            s = c.start(k, nxt, rows[k].numel())
            _fwd(c, [k], [nxt], [s])  # reads back the spilled prefix (checked in _fwd)
            rows[k] = nxt
    assert c.stats.get("host_evictions", 0) > 0 and c.stats.get("host_loads", 0) > 0


def test_rows_that_do_not_fit_fall_back_to_uncached():
    c = _cache(chunk=4, page=8, n_pages=5, n_slots=6)
    _fwd(c, ["a"], [torch.arange(30)], [0])  # 4 pages of 4 usable
    big = torch.arange(100, 140)  # 5 pages: never fits next to anything
    with pytest.raises(RuntimeError, match="does not fit"):
        _fwd(c, [None], [big], [0])
    _check_ownership(c)


def test_fits_counts_pages_entry_slots_and_new_boundaries():
    c = _cache(chunk=4, page=8, n_pages=12, n_slots=10)
    _fwd(c, ["a"], [torch.arange(30)], [0])  # boundaries 4..28 kept (max_keep 4) -> 3 slots
    assert c.row_need("a", 40, 28) == (5, len(c.entries["a"].states) + 2)  # keep(28, 40) = 28, 32, 36
    assert c.row_need(None, 40, 0) == (5, 0)
    assert c.row_need("new", 40, 0, 10) == (5, 4)
    assert c.fits([("a", 40, 28, None)]) and not c.fits([("a", 40, 28, None), ("b", 40, 0, None)])


def test_random_rounds_never_leak_or_share():
    """Continuation rounds with rejections, releases, re-forwards and spills: ownership holds
    after every forward (``_check_ownership``) and every read sees its own row (``_fwd``)."""
    c = _cache(chunk=4, page=8, n_pages=24, n_slots=24)
    gen = torch.Generator().manual_seed(7)
    rows = {f"r{i}": torch.randint(5, 1000, (int(torch.randint(5, 30, (1,), generator=gen)),), generator=gen)
            for i in range(6)}
    for k, r in rows.items():
        _fwd(c, [k], [r], [0])
    for _ in range(30):
        ks = [k for k in rows if torch.rand(1, generator=gen).item() < 0.6][:3] or [next(iter(rows))]
        nxt, starts = [], []
        for k in ks:
            r = rows[k]
            if torch.rand(1, generator=gen).item() < 0.3 and r.numel() > 3:
                cut = int(torch.randint(1, r.numel(), (1,), generator=gen))
                r = torch.cat([r[:cut], torch.tensor([3])])
            n = torch.cat([r, torch.randint(5, 1000, (int(torch.randint(1, 12, (1,), generator=gen)),), generator=gen)])
            n = n[: c.max_len - 8]
            nxt.append(n)
            starts.append(c.start(k, n, r.numel()))
            rows[k] = n
        _fwd(c, ks, nxt, starts)
        if torch.rand(1, generator=gen).item() < 0.2:
            k = next(iter(rows))
            c.release([k])
            rows[k] = rows[k][:5]
            _fwd(c, [k], [rows[k]], [0])


# ---- anchors of a segment ---------------------------------------------------------------
def test_fetch_anchors_from_offset_matches_slice_of_full():
    from thundersync_rl.routes import RouteRefs, fetch_anchors

    from types import SimpleNamespace

    torch.manual_seed(0)
    full = torch.randint(0, 8, (40, 3, 2), dtype=torch.int16)
    full[5] = -1
    blocks = [
        SimpleNamespace(lo=lo, hi=min(lo + 16, 40), layers=[1, 3, 5], ref=full[lo : lo + 16])
        for lo in range(0, 40, 16)
    ]
    refs = SimpleNamespace(blocks=blocks, length=37)
    get = lambda x: x  # noqa: E731
    whole = fetch_anchors(refs, [3, 5], 2, 48, get)
    for lo in (0, 7, 16, 30):
        seg = fetch_anchors(refs, [3, 5], 2, 48 - lo, get, lo=lo)
        assert torch.equal(seg, whole[lo:]), lo
    assert RouteRefs is not None


def test_graph_key_is_the_eligibility_hook():
    """learner_graphs asks the cache (``graph_key``): () without a cache forward, None when the
    active cache forward is not graph-safe, else a key with the pool addresses / sizes."""
    from nemo_rl.models.megatron import prefix_state_cache as psc

    assert psc.graph_key() == ()
    c = _cache(chunk=4)
    with torch.no_grad(), c.forward(["a"], [torch.arange(10)], [0]):
        k1 = psc.graph_key()
        assert k1[0] == "prefix" and k1[1:] == c.pools.ptr_key()
        c.graphs = False
        assert psc.graph_key() is None
    assert psc.graph_key() == ()


def test_pools_follow_the_activation_dtype_until_state_is_cached():
    c = _cache(chunk=4)
    c._retype_kv(torch.float16)
    c._retype_conv(torch.float16)
    assert c.pools.k[1].dtype == torch.float16 and c.pools.conv[2].dtype == torch.float16
    assert c.pools.ssm[2].dtype == torch.float32  # SSM states stay fp32
    c._retype_kv(torch.float32)
    c._retype_conv(torch.float32)
    _fwd(c, ["a"], [torch.arange(10)], [0])
    with pytest.raises(RuntimeError, match="with state cached"):
        c._retype_conv(torch.float16)
    with pytest.raises(RuntimeError, match="with state cached"):
        c._retype_kv(torch.float16)
