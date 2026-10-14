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
"""q transport P0/P1: blocks own their bytes, sealing while drafting, verifier prefetch.

* P0: a block put from a row slice of a long draft pickles to its own size, not
  the draft's (``own_block``; ``put_hidden`` and ``put_routes``).
* P1: ``SealingHiddenLog`` hands out exactly the rows ``HiddenLog`` would
  (bitwise), for any pop/take boundaries (block re-splits), eviction rewrites,
  missing rows and closes, with every object either handed out or freed.
* P1: ``HiddenPrefetch`` + ``HiddenRows`` read the same bytes as on-demand gets,
  holding at most two batches.
"""

import pickle
import threading

import torch
from thundersync_rl.engine_q import (
    HiddenLog,
    HiddenPrefetch,
    HiddenRefs,
    HiddenRows,
    SealingHiddenLog,
    own_block,
    put_hidden,
)
from thundersync_rl.routes import put_routes


class _Store:
    """Fake object store; refs are ints; tracks live objects."""

    def __init__(self):
        self.objs, self._n = {}, 0
        self.lock = threading.Lock()
        self.gets = 0

    def put(self, t):
        # Store what pickling would ship (whole storage of the tensor given).
        assert t.untyped_storage().nbytes() == t.numel() * t.element_size(), "put of a view"
        with self.lock:
            self._n += 1
            self.objs[self._n] = pickle.loads(pickle.dumps(t))
            return self._n

    def get(self, r):
        with self.lock:
            self.gets += 1
            return self.objs[r]

    def free(self, refs):
        with self.lock:
            for r in refs:
                self.objs.pop(r)


def test_pickled_block_size_is_block_bytes():
    h = torch.randn(4096, 64)
    sizes = []
    put_hidden(h, put=lambda b: sizes.append((len(pickle.dumps(b)), b.numel() * 4)))
    assert len(sizes) == 16
    for pk, nb in sizes:
        assert nb <= pk <= nb + 2048, (pk, nb)  # was 4096 * 64 * 4 per block
    routes = torch.randint(0, 64, (5000, 3, 6), dtype=torch.int16)
    rs = []
    put_routes(routes, lambda b: rs.append((len(pickle.dumps(b)), b.numel() * 2)), block_positions=1024)
    for pk, nb in rs:
        assert nb <= pk <= nb + 2048, (pk, nb)
    # own_block keeps tensors that already own their bytes.
    t = torch.randn(3, 4)
    assert own_block(t) is t and own_block(h[:3]) is not h[:3]


def _record_stream(log, rows, keys, steps, rng, holes=()):
    """Decode-like recording: each forward logs one row per live key (random order/rows)."""
    for t in range(steps):
        ks = [k for k in keys if t < rows[k].shape[0] and (k, t) not in holes]
        perm = torch.randperm(len(ks), generator=rng).tolist()
        ks = [ks[i] for i in perm]
        if not ks:
            continue
        hid = torch.stack([rows[k][t] for k in ks])
        log.record([(j, k, t) for j, k in enumerate(ks)], hid)


def _cat(store, blocks):
    return torch.cat([store.objs[r] for r, _, _ in blocks]) if blocks else torch.empty(0, 0)


def test_sealing_log_is_bitwise_hiddenlog_for_any_boundaries():
    rng = torch.Generator().manual_seed(0)
    d, B = 16, 8
    lens = {"a": 70, "b": 33, "c": 8, "d": 5, "e": 40}
    rows = {k: torch.randn(n, d, generator=rng) for k, n in lens.items()}
    store = _Store()
    log = SealingHiddenLog(chunk_rows=16, block_rows=B, put=store.put, get=store.get, free=store.free)
    ref = HiddenLog(chunk_rows=16)
    keys = list(lens)
    # Hole at e:20 -> e's longest prefix is 20 rows.
    holes = {("e", 20)}
    _record_stream(log, rows, keys, 30, torch.Generator().manual_seed(1), holes)
    _record_stream(ref, rows, keys, 30, torch.Generator().manual_seed(1), holes)
    # Running draft "a": pieces with boundaries inside blocks (re-splits).
    got_a, taken_a = [], []
    for lo, n in ((0, 13), (13, 3), (16, 9)):
        blocks = log.take_refs("a", lo, n)
        taken_a.append(blocks)
        assert sum(r for _, r, _ in blocks) == n
        got_a.append(_cat(store, blocks))
        assert torch.equal(got_a[-1], ref.take("a", lo, n))
    # Eviction recompute rewrites a sealed row of b (bitwise the same): dropped.
    log.record([(0, "b", 2)], rows["b"][2:3])
    _record_stream(log, rows, keys, 80, torch.Generator().manual_seed(2), holes)
    _record_stream(ref, rows, keys, 80, torch.Generator().manual_seed(2), holes)
    assert log.sealed_bytes > 0 and log.resplits > 0
    # Pop the rest; the deadline puts only partial tails.
    tail0 = log.tail_put_bytes
    out = {k: log.pop_refs(k, lens[k]) for k in ("b", "c", "d", "e")}
    assert log.tail_put_bytes - tail0 < sum(lens.values()) * d * 4
    for k in ("b", "c", "d", "e"):
        exp = ref.pop(k, lens[k])
        assert torch.equal(_cat(store, out[k]), exp), k
        for r, n, nb in out[k]:
            t = store.objs[r]
            assert t.shape[0] == n and nb == t.untyped_storage().nbytes()
    assert out["e"] and sum(n for _, n, _ in out["e"]) == 20
    rest = log.take_refs("a", 25, 45)
    assert torch.equal(_cat(store, rest), ref.take("a", 25, 45))
    log.close("a")
    # Every object left in the store was handed out (none leaked by sealing,
    # re-splits or closes).
    handed = {r for bl in list(out.values()) + [rest] + taken_a for r, _, _ in bl}
    assert set(store.objs) == handed
    assert log.live_rows() == 0
    log.stop()


def test_sealing_close_frees_unhanded_blocks():
    store = _Store()
    log = SealingHiddenLog(chunk_rows=64, block_rows=4, put=store.put, get=store.get, free=store.free)
    h = torch.randn(20, 3)
    log.record([(t, "x", t) for t in range(20)], h)
    blocks = log.take_refs("x", 0, 6)  # 1 sealed block + a re-split of the next
    assert torch.equal(_cat(store, blocks), h[:6])
    handed = {r for r, _, _ in blocks}
    log.close("x")
    assert set(store.objs) == handed  # sealed, never handed out -> freed
    assert log.pop_refs("x", 3) is None
    log.stop()


def test_prefetch_reads_the_same_bytes_and_is_bounded():
    store = _Store()
    hs = [torch.randn(n, 8) for n in (300, 50, 513, 7)]
    refs = [HiddenRefs.from_blocks(put_hidden(h, put=store.put)) for h in hs]
    pf = HiddenPrefetch(get=store.get, max_bytes=1 << 30, max_batches=2)
    assert pf.submit(refs[:2]) and pf.submit(refs[2:3])
    assert not pf.submit(refs[3:])  # two batches held
    for h, r in zip(hs[:3], refs[:3]):
        rows = HiddenRows(r, get=store.get, prefetched=pf.futures)
        idx = [0, h.shape[0] - 1, h.shape[0] // 2]
        assert torch.equal(rows(idx), h[idx])
        assert rows.fetched_real_bytes == rows.fetched_bytes
    pf.release()
    assert pf.submit(refs[3:])
    pf.close()
    assert pf.futures == {}
