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
    """The 128k failure mode (agent_carried_walk.md): working set above the budget."""
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


# ---- learner: boundary bookkeeping (no kernels) ---------------------------------------
def _cache(chunk=4, max_keep=4):
    from nemo_rl.models.megatron.prefix_state_cache import PrefixStateCache

    c = PrefixStateCache.__new__(PrefixStateCache)
    c.chunk, c.max_keep, c.gpu_bytes = chunk, max_keep, 1 << 40
    c.entries = OrderedDict()
    c.weights_version = None
    c.attention_layers, c.mamba_layers = [1], [2]
    c.stats = {"hits": 0, "rows": 0, "tokens_skipped": 0, "tokens_forwarded": 0}
    return c


def test_keep_boundaries():
    c = _cache(chunk=4, max_keep=4)
    assert c.keep(0, 3) == []  # only boundary 0: nothing worth keeping
    assert c.keep(0, 9) == [0, 4, 8]  # boundary 9 itself is never usable
    assert c.keep(8, 12) == [8]
    # long pieces: start, last and evenly spaced in between
    assert c.keep(0, 64) == [0, 20, 40, 60]


def _commit(c, key, tokens, start, kv_dim=2, pin=None):
    """What a forward of [start, n) leaves behind (the hooks' pending tensors)."""
    from nemo_rl.models.megatron.prefix_state_cache import _Row

    n = tokens.numel()
    keep = c.keep(start, n)
    entry = c.entries.get(key) if start else None
    r = _Row(key, tokens, start, entry, keep, c.retain(c.entries.get(key), start, keep, pin))
    if r.keep and r.new_kv_len > r.start:
        m = r.new_kv_len - start
        k = tokens[start : start + m].float().view(m, 1, 1).expand(m, 1, kv_dim).contiguous()
        r.pending_kv[1] = (k, -k)
    for s in r.keep:
        if s != start:
            r.pending_states[s] = {2: (torch.full((3, 2), float(s)), torch.full((2, 2), float(s)))}
    c._commit(r)


def test_start_uses_matching_boundary_and_kv_is_the_prefix():
    c = _cache(chunk=4)
    row = torch.arange(100, 111)  # n = 11
    _commit(c, "a", row, 0)
    e = c.entries["a"]
    assert sorted(e.states) == [4, 8] and e.kv_len == 8
    assert torch.equal(e.kv[1][0][:8, 0, 0], row[:8].float())
    # accepted piece: prompt = 11 (logits from position 10) -> boundary 8
    nxt = torch.cat([row, torch.arange(200, 205)])
    assert c.start("a", nxt, 11) == 8
    # rejection inside the earlier piece at position 6: tokens differ from 6 on -> boundary 4
    rej = torch.cat([row[:6], torch.tensor([7]), torch.arange(300, 305)])
    assert c.start("a", rej, 7) == 4
    # prompt short of the boundary: logits needed from position 3 -> 0
    assert c.start("a", nxt, 4) == 0
    assert c.start("missing", nxt, 11) == 0
    assert c.start(None, nxt, 11) == 0


def test_commit_continues_from_cached_kv_and_states():
    c = _cache(chunk=4)
    row = torch.arange(100, 111)
    _commit(c, "a", row, 0)
    old_state_8 = c.entries["a"].states[8]
    nxt = torch.cat([row, torch.arange(200, 207)])  # n = 18
    _commit(c, "a", nxt, 8)
    e = c.entries["a"]
    assert sorted(e.states) == [8, 12, 16] and e.kv_len == 16
    assert e.states[8] is old_state_8  # the start boundary carries over
    assert torch.equal(e.kv[1][0][:16, 0, 0], nxt[:16].float())
    assert torch.equal(e.tokens, nxt[:16])


def test_pin_retains_the_floor_boundary_across_pieces():
    """Carried walk (DESIGN 14t): a pinned older boundary survives later pieces, so a
    rejection scan can re-forward the block from its floor's piece."""
    c = _cache(chunk=4)
    row = torch.arange(100, 111)
    _commit(c, "a", row, 0)  # boundaries 4, 8
    s4 = c.entries["a"].states[4]
    nxt = torch.cat([row, torch.arange(200, 207)])  # n = 18
    _commit(c, "a", nxt, 8, pin=6)  # keeps 8, 12, 16 and the pinned 4
    assert sorted(c.entries["a"].states) == [4, 8, 12, 16]
    assert c.entries["a"].states[4] is s4
    nxt2 = torch.cat([nxt, torch.arange(300, 306)])  # n = 24
    _commit(c, "a", nxt2, 16, pin=6)
    assert sorted(c.entries["a"].states) == [4, 16, 20]
    # a scan row whose prompt ends at 7 (floor piece) starts from boundary 4
    assert c.start("a", nxt2[:20], 7) == 4
    # without a pin the old boundary is dropped
    _commit(c, "a", nxt2, 16)
    assert sorted(c.entries["a"].states) == [16, 20]
    assert c.retain(None, 8, [8], 6) == [] and c.retain(c.entries["a"], 0, [0, 4], 6) == []


def test_weights_version_and_release_drop_entries():
    c = _cache(chunk=4)
    _commit(c, "a", torch.arange(10), 0)
    _commit(c, "b", torch.arange(10), 0)
    c.set_weights_version(1)
    assert not c.entries
    _commit(c, "a", torch.arange(10), 0)
    c.set_weights_version(1)
    assert "a" in c.entries
    c.release(["a", "zz"])
    assert not c.entries


def test_grow_amortizes_and_preserves_prefix():
    from nemo_rl.models.megatron.prefix_state_cache import _grow

    buf = _grow(None, torch.ones(300, 1), 0, 300)
    cap = buf.shape[0]
    assert cap >= 300
    buf2 = _grow(buf, torch.full((10, 1), 2.0), 300, 310)
    assert buf2.data_ptr() == buf.data_ptr() or cap < 310
    assert torch.equal(buf2[:300], torch.ones(300, 1)) and torch.equal(buf2[300:310], torch.full((10, 1), 2.0))
    big = _grow(buf2, torch.full((cap, 1), 3.0), 310, 310 + cap)
    assert big.shape[0] >= 310 + cap and torch.equal(big[:300], torch.ones(300, 1))


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
