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
"""Engine-sourced q for block verification under batch invariance.

With batch-invariant kernels the inference engine's decode forward is bitwise
the training forward, so the true sampling law q of a drafted token is the
engine's own distribution:

* log q(x_i) of every drafted token is the engine's reported log-prob
  (``generated_log_probs``, already streamed with the draft);
* a full q row, needed only where the lazy scan computes a residual mass and
  at the residual sample, is ``log_softmax(W_head^(v) h_i)`` with h_i the
  engine's final hidden state (LM-head input, after the final norm) at that
  step and W_head^(v) the LM-head weight of the drafting version v.

So the verifier needs no scorer copy and no whole-model weight stash: the
engine logs h_i per drafted token (d x 2 bytes, ``HiddenLog``) and the learner
keeps only the LM head per pending version (V x d x 2 bytes). p comes from the
training model's own bf16 forward at theta_{k+1}.

Tensor parallelism (training TP = inference TP, as batch invariance requires):
the LM head is vocab-parallel, so each rank stashes only its shard
(V/TP x d), computes its logits shard of a row, and the full row is the
shards all-gathered in rank order followed by one fp32 log-softmax: what the
engine does (``ColumnParallelLinear`` with gathered output, then the
controller's ``log_softmax(logits.float())``) and what batch-invariant
training does (``_GatheredVocabLogprob``). All-gathers are copies, so the row
is bitwise the TP=1 row of the same logits.

This module holds the parts that need no Megatron: the engine-side hidden
store and per-step row mapping, the gathered log-prob rows, and the per-draft
verification rule.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np
import torch

from nemo_rl.models.megatron import exact_nucleus

# Sampling-params attribute that tags a draft request for hidden-state logging.
HIDDEN_KEY_ATTR = "ts_hidden_key"
# Key of the per-call transport statistics in ``pop_draft_hidden`` /
# ``take_draft_hidden`` results (not a draft key).
HIDDEN_STATS_KEY = "__ts_hidden_stats__"
# Sampling-params attribute of a request that continues an anchored rollout at
# theta_k (anchor horizon): its key in ``register_engine_anchors``.
ANCHOR_KEY_ATTR = "ts_anchor_key"


def hidden_owner(key: str, tp: int) -> int:
    """Tensor-parallel rank that logs draft ``key``'s hidden rows (and nucleus ids): crc32 of the key mod TP.

    Deterministic across processes (Python's ``hash`` of a str is salted per process), so every TP rank of an
    engine agrees on the owner without communicating.
    """
    import zlib

    return zlib.crc32(key.encode()) % int(tp)




class ClosedKeys(dict):
    """Closed log keys -> the log generation they were closed in (long runs).

    A closed key's later rows are dropped (an aborted stream may still finish its engine step). Keys used to be
    kept forever (one string per draft / continuation stream of the run on every engine rank); ``prune(gen)``
    forgets keys closed before generation ``gen``. The driver advances the generation once per step
    (``prune_closed_logs``) and prunes with a lag of steps, long after any late row of an aborted stream.
    """

    gen = 0

    def add(self, key: str) -> None:
        self.setdefault(key, self.gen)

    def prune(self, before: int) -> int:
        old = [k for k, g in self.items() if g < before]
        for k in old:
            del self[k]
        return len(old)


class HiddenLog:
    """Final hidden states of tagged draft requests, by (key, generated index).

    Engine side. ``record`` copies the selected rows of a forward's LM-head
    input to (pinned, when CUDA is available) host memory without blocking
    the engine; ``pop`` returns rows 0..n-1 of a draft, or None if any is
    missing. Memory: d x (head-input dtype) bytes per drafted token (4 d under
    the fp32 LM head of the fp16-GEMM recipe, 2 d otherwise), released as
    drafts are popped. Thread-safe (the engine records on its loop thread).
    """

    def __init__(self, chunk_rows: int = 8192):
        self.chunk_rows = chunk_rows
        self._lock = threading.Lock()
        self._rows: dict[str, dict[int, tuple[list, int]]] = {}
        self._closed = ClosedKeys()
        self._cur: Optional[list] = None  # [buffer, used, live rows, event]
        self.recorded = 0

    def _alloc(self, k: int, like: torch.Tensor) -> tuple[list, int]:
        cur = self._cur
        if cur is None or cur[1] + k > cur[0].shape[0]:
            n = max(self.chunk_rows, k)
            buf = torch.empty(
                (n, like.shape[-1]),
                dtype=like.dtype,
                pin_memory=like.is_cuda,
            )
            cur = self._cur = [buf, 0, 0, None]
        off = cur[1]
        cur[1] += k
        return cur, off

    def record(self, entries: list[tuple[int, str, int]], hidden: torch.Tensor) -> None:
        """Store ``hidden[row]`` as token ``t`` of draft ``key`` for each entry."""
        with self._lock:
            entries = [e for e in entries if e[1] not in self._closed]
            if not entries:
                return
            idx = torch.tensor([r for r, _, _ in entries])
            if hidden.is_cuda:
                # Pinned and non-blocking: a pageable copy would make the host
                # wait for the forward that produced ``hidden``.
                idx = idx.pin_memory().to(hidden.device, non_blocking=True)
            sel = hidden.detach().index_select(0, idx)
            chunk, off = self._alloc(len(entries), sel)
            chunk[0][off : off + len(entries)].copy_(sel, non_blocking=sel.is_cuda)
            if sel.is_cuda:
                ev = torch.cuda.Event()
                ev.record()
                chunk[3] = ev
            for j, (_, key, t) in enumerate(entries):
                slots = self._rows.setdefault(key, {})
                old = slots.get(t)
                if old is not None:
                    old[0][2] -= 1
                slots[t] = (chunk, off + j)
                chunk[2] += 1
            self.recorded += len(entries)

    def pop(self, key: str, n: int) -> Optional[torch.Tensor]:
        """Rows 0..m-1 of draft ``key`` [m, d] (CPU): the longest logged prefix (m <= n).

        Closes the key: rows the engine records for it later are dropped.
        """
        with self._lock:
            self._closed.add(key)
            slots = self._rows.pop(key, {})
            for chunk, _ in slots.values():
                chunk[2] -= 1
            # The longest contiguous prefix of rows: a row the engine did not
            # log (or the source check rejected) truncates the draft there.
            m = 0
            while m < n and m in slots:
                m += 1
            if m == 0:
                return None if n else torch.empty(0, 0)
            for ev in {id(slots[t][0]): slots[t][0][3] for t in range(m)}.values():
                if ev is not None:
                    ev.synchronize()
            return torch.stack([slots[t][0][0][slots[t][1]] for t in range(m)])

    def take(self, key: str, lo: int, n: int) -> Optional[torch.Tensor]:
        """Rows lo..lo+m-1 of a running draft [m, d] (the longest logged run, m <= n).

        The rows leave the log; the key stays open (later rows keep being
        logged), so a draft can be taken in pieces while it is decoded.
        """
        with self._lock:
            slots = self._rows.get(key, {})
            m = 0
            while m < n and lo + m in slots:
                m += 1
            if m == 0:
                return None if n else torch.empty(0, 0)
            taken = [slots.pop(t) for t in range(lo, lo + m)]
            for chunk, _ in taken:
                chunk[2] -= 1
            for ev in {id(c[0]): c[3] for c, _ in taken}.values():
                if ev is not None:
                    ev.synchronize()
            return torch.stack([c[0][i] for c, i in taken])

    def close(self, key: str) -> None:
        """Drop a draft's rows and the rows the engine logs for it later."""
        self.pop(key, 0)

    def live_rows(self) -> int:
        with self._lock:
            return sum(len(s) for s in self._rows.values())

    def prune_closed(self, lag: int = 2) -> int:
        """Advance the close generation; forget keys closed ``lag`` or more generations ago."""
        with self._lock:
            self._closed.gen += 1
            return self._closed.prune(self._closed.gen - int(lag) + 1)  # closed at gen <= now - lag

    def bookkeeping(self) -> dict[str, float]:
        """Sizes for the long-run leak probe (leak_probe.py)."""
        with self._lock:
            return {
                "open_keys": float(len(self._rows)),
                "closed_keys": float(len(self._closed)),
                "rows": float(sum(len(v) for v in self._rows.values())),
            }


# Rows per object-store block; also the sealing granularity of
# ``SealingHiddenLog`` (defined here so both use the same value).
HIDDEN_BLOCK_ROWS = int(os.environ.get("THUNDERSYNC_HIDDEN_BLOCK_ROWS", "256"))


def stack_rows(rows: list) -> torch.Tensor:
    """``torch.stack([c[0][i] for c, i in rows])`` for logged rows [(chunk, idx)], with one view per chunk run.

    A decode step logs one row per draft into the current chunk, so a draft's block of B rows is a few runs of
    equally spaced rows, one run per chunk. Each run becomes one strided view and a single ``torch.cat`` copies
    them (outside the GIL): a few Python objects and one GIL release per block instead of B row views (an
    unevenly spaced run, e.g. an eviction rewrite, keeps per-row views). Same bytes, dtype and shape as the stack
    (a fresh contiguous [len(rows), d] tensor that owns its bytes).
    """
    views = []
    a, n = 0, len(rows)
    while a < n:
        c, i0 = rows[a]
        buf = c[0]
        b = a + 1
        if b < n and rows[b][0] is c and rows[b][1] > i0:
            step = rows[b][1] - i0
            while b < n and rows[b][0] is c and rows[b][1] == i0 + (b - a) * step:
                b += 1
            views.append(buf[i0 : rows[b - 1][1] + 1 : step])
        else:
            views.append(buf[i0 : i0 + 1])
        a = b
    return torch.cat(views)


# Hidden-block payload in the object store (P3). ``ray.put`` of a torch tensor pickles it through cloudpickle and
# torch's storage serializer, a byte copy under the GIL on the engine rank's seal thread. A numpy array is pickled
# with protocol-5 out-of-band buffers instead: Ray copies its bytes into the object store without the GIL. The
# payload is (tag, torch dtype name, integer view of the same bytes); ``hidden_block`` turns it back into a tensor
# with exactly those bytes (16-bit floats travel as int16, so bf16 needs no numpy dtype). Plain tensors (older
# blocks, test stores) pass through unchanged.
_HIDDEN_PAYLOAD = "ts-hidden-v1"
_INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def hidden_payload(b: torch.Tensor) -> tuple:
    """``b`` (CPU, contiguous, owns its bytes) as an object-store payload of the same bytes, no copy."""
    view = b.view(_INT_VIEW[b.element_size()]) if b.is_floating_point() or b.is_complex() else b
    return (_HIDDEN_PAYLOAD, str(b.dtype).removeprefix("torch."), view.numpy())


def hidden_block(obj: Any) -> torch.Tensor:
    """A hidden block read from the object store (``hidden_payload`` or a tensor) as a tensor that owns its bytes."""
    if isinstance(obj, torch.Tensor):
        return obj
    tag, dtype, arr = obj
    if tag != _HIDDEN_PAYLOAD:
        raise ValueError(f"not a hidden-block payload: {tag!r}")
    # The array returned by ray.get is a read-only view of the object store; copy it (numpy copies without the
    # GIL), as deserializing a torch tensor did, so the object can be freed while the rows are in use.
    t = torch.from_numpy(arr.copy())
    want = getattr(torch, dtype)
    return t if t.dtype == want else t.view(want)


def put_hidden_block(b: torch.Tensor, put: Optional[Callable[[Any], Any]] = None) -> Any:
    """Put one block: through ``put`` as a tensor (tests, custom stores) or with ``ray.put`` as a payload."""
    if put is not None:
        return put(b)
    import ray

    return ray.put(hidden_payload(b))


_NO_ROW = -1
_CHUNK_SHIFT = 32  # row ref = chunk sequence number << 32 | row index in the chunk


class SealingHiddenLog(HiddenLog):
    """``HiddenLog`` that puts full blocks into the object store while drafting.

    Each time the rows [s, s + B) of an open draft are all logged (s = the
    draft's seal frontier, B = ``block_rows``), a background thread stacks
    them (a fresh tensor that owns its bytes), ``ray.put``s it and drops the
    pinned rows. ``pop_refs`` / ``take_refs`` (the ``as_refs`` transport) then
    hand out the sealed blocks' handles and put only the rows not sealed yet,
    so the deadline cost is O(drafts) rather than O(drafted tokens).

    Bit-identity: a sealed block holds exactly the logged rows (``torch.stack``
    copies bytes), so the learner reads the same bytes as with ``pop`` +
    ``put_hidden``. A row the engine rewrites after its block was sealed
    (recompute after an eviction) is bitwise the same under batch invariance
    and is dropped. A take that ends inside a sealed block re-splits that block
    (one local ``get`` and two puts of < B rows), so every handle handed out
    covers whole objects and the driver's free-after-verify stays per object.

    Vectorized (P4): the logged rows of each open draft live in one row of a
    dense window ``_win`` [open drafts, W] of row refs (positions base..base+W-1
    of the draft; -1 = not logged), so ``record`` is a few numpy operations per
    decode step, not Python per row: key ids, the frontier filter, one gather
    and copy of the rows, one scatter into the window, and the block-complete
    test (the block's last row first, then the whole block, only for drafts
    whose last row is logged). A complete block leaves the window with its
    seal job, so the frontier moves when the job is queued, as before.
    Draft key ids are never reused (closed ids stay closed; ``prune_closed``
    forgets the key strings).

    Bounded: seal jobs are small (key, start, row refs) notices in a queue of
    at most ``max_pending`` entries; when it is full a block is simply sealed
    later (``record`` never blocks the engine). Pinned rows are dropped as soon
    as their block is put, so host memory holds each row once (pinned or in
    the object store), as without sealing.
    """

    def __init__(
        self,
        chunk_rows: int = 8192,
        block_rows: int = HIDDEN_BLOCK_ROWS,
        put: Optional[Callable[[Any], Any]] = None,
        get: Optional[Callable[[Any], Any]] = None,
        free: Optional[Callable[[list], None]] = None,
        max_pending: int = 1024,
    ):
        import queue

        super().__init__(chunk_rows)
        self.block_rows = block_rows
        self._put = put
        self._getf = get
        self._freef = free
        # key -> [(ref, lo, rows, nbytes)] sealed blocks, contiguous from the
        # key's hand-out frontier; key -> seal jobs in flight.
        self._sealed: dict[str, list[tuple[Any, int, int, int]]] = {}
        self._pending: dict[str, int] = {}
        self._pending_rows = 0
        # Draft key ids and their window rows.
        self._kid: dict[str, int] = {}
        self._next_kid = 0
        self._wr_of_kid = np.full(1024, -1, dtype=np.int64)
        self._kclosed = np.zeros(1024, dtype=bool)
        self._close_gen: dict[str, int] = {}
        self._gen = 0
        W = 2 * block_rows
        self._win = np.full((64, W), _NO_ROW, dtype=np.int64)
        self._base = np.zeros(64, dtype=np.int64)  # position of window column 0
        self._front = np.zeros(64, dtype=np.int64)  # seal frontier (rows below: sealed / handed out)
        self._wr_key: list[Optional[str]] = [None] * 64
        self._free_wr = list(range(63, -1, -1))
        # Pinned chunks by sequence number, with their live row counts.
        self._chunks: dict[int, list] = {}
        self._chunk_live: dict[int, int] = {}
        self._seq = -1
        self._cond = threading.Condition(self._lock)
        self._jobs: "queue.Queue" = queue.Queue(maxsize=max_pending)
        self.sealed_bytes = 0
        self.seal_put_s = 0.0
        self.tail_put_bytes = 0
        self.resplits = 0
        self._thread = threading.Thread(
            target=self._run, name="ts-hidden-seal", daemon=True
        )
        self._thread.start()

    # -- object store helpers ------------------------------------------------
    def _do_put(self, b: torch.Tensor) -> Any:
        return put_hidden_block(b, self._put)

    def _do_get(self, ref: Any) -> torch.Tensor:
        if self._getf is not None:
            return hidden_block(self._getf(ref))
        import ray

        return hidden_block(ray.get(ref))

    def _do_free(self, refs: list) -> None:
        if not refs:
            return
        (self._freef or ray_free)(refs)

    # -- key ids / window rows (lock held) ---------------------------------------
    def _key_id(self, key: str) -> int:
        kid = self._kid.get(key)
        if kid is None:
            kid = self._kid[key] = self._next_kid
            self._next_kid += 1
            if kid >= self._wr_of_kid.shape[0]:
                n = max(2 * self._wr_of_kid.shape[0], kid + 1)
                self._wr_of_kid = np.concatenate(
                    [self._wr_of_kid, np.full(n - self._wr_of_kid.shape[0], -1, dtype=np.int64)]
                )
                self._kclosed = np.concatenate(
                    [self._kclosed, np.zeros(n - self._kclosed.shape[0], dtype=bool)]
                )
        return kid

    def _window_row(self, kid: int, key: str) -> int:
        """Window row of an open key (allocated on first use); -1 for a closed key."""
        w = int(self._wr_of_kid[kid])
        if w >= 0 or self._kclosed[kid]:
            return w
        if not self._free_wr:
            n = self._win.shape[0]
            self._win = np.concatenate([self._win, np.full_like(self._win, _NO_ROW)])
            self._base = np.concatenate([self._base, np.zeros(n, dtype=np.int64)])
            self._front = np.concatenate([self._front, np.zeros(n, dtype=np.int64)])
            self._wr_key += [None] * n
            self._free_wr = list(range(2 * n - 1, n - 1, -1))
        w = self._free_wr.pop()
        self._win[w] = _NO_ROW
        self._base[w] = self._front[w] = 0
        self._wr_key[w] = key
        self._wr_of_kid[kid] = w
        return w

    def _grow_window(self, need: int) -> None:
        W = self._win.shape[1]
        if need <= W:
            return
        n = max(2 * W, need)
        win = np.full((self._win.shape[0], n), _NO_ROW, dtype=np.int64)
        win[:, :W] = self._win
        self._win = win

    def _drop_refs(self, refs: np.ndarray) -> None:
        """Rows leaving the log (sealed, handed out, closed, rewritten): release their chunks."""
        refs = refs[refs >= 0]
        if not refs.size:
            return
        seqs, counts = np.unique(refs >> _CHUNK_SHIFT, return_counts=True)
        for s, c in zip(seqs.tolist(), counts.tolist()):
            left = self._chunk_live[s] - c
            if left <= 0 and s != self._seq:
                del self._chunk_live[s], self._chunks[s]
            else:
                self._chunk_live[s] = left

    def _rows_of(self, refs: np.ndarray) -> list:
        """[(chunk, idx)] of row refs (``stack_rows`` input)."""
        ch = self._chunks
        return [(ch[r >> _CHUNK_SHIFT], r & 0xFFFFFFFF) for r in refs.tolist()]

    def _rebase(self, w: int) -> None:
        """Move window row ``w`` so its column 0 is the first logged row or the frontier."""
        row = self._win[w]
        rel = int(self._front[w] - self._base[w])
        filled = np.flatnonzero(row[: min(rel, row.shape[0])] >= 0)
        d = int(filled[0]) if filled.size else rel
        if d <= 0:
            return
        W = row.shape[0]
        if d >= W:
            row[:] = _NO_ROW
        else:
            row[: W - d] = row[d:]
            row[W - d :] = _NO_ROW
        self._base[w] += d

    # -- engine side -----------------------------------------------------------
    def record(self, entries: list[tuple[int, str, int]], hidden: torch.Tensor) -> None:
        if entries:
            rows, keys, ts = zip(*entries)
            self.record_rows(rows, keys, ts, hidden)

    def record_rows(self, rows, keys, ts, hidden: torch.Tensor) -> None:
        """Store ``hidden[rows[j]]`` as token ``ts[j]`` of draft ``keys[j]`` (one decode step's rows)."""
        n = len(rows)
        if n == 0:
            return
        with self._lock:
            get = self._kid.get
            ids = [get(k) for k in keys]
            if None in ids:  # first row of a draft: assign its id
                ids = [self._key_id(k) if i is None else i for k, i in zip(keys, ids)]
            kids = np.array(ids, dtype=np.int64)
            wr = self._wr_of_kid[kids]
            new = (wr < 0) & ~self._kclosed[kids]
            if new.any():
                for j in np.flatnonzero(new).tolist():
                    self._window_row(int(kids[j]), keys[j])
                wr = self._wr_of_kid[kids]
            ts = np.asarray(ts, dtype=np.int64)
            rows = np.asarray(rows, dtype=np.int64)
            # Closed keys and rows below the frontier (sealed or handed out; an eviction recompute rewrites the
            # same t, bitwise the same under batch invariance) are dropped.
            keep = (wr >= 0) & (ts >= self._front[np.maximum(wr, 0)])
            if not keep.all():
                rows, wr, ts = rows[keep], wr[keep], ts[keep]
            k = int(rows.shape[0])
            if k == 0:
                return
            off = ts - self._base[wr]
            self._grow_window(int(off.max()) + 1)
            idx = torch.from_numpy(rows)
            if hidden.is_cuda:
                # Pinned and non-blocking: a pageable copy would make the host wait for the forward.
                idx = idx.pin_memory().to(hidden.device, non_blocking=True)
            sel = hidden.detach().index_select(0, idx)
            chunk, start = self._alloc(k, sel)
            if chunk is not self._chunks.get(self._seq):
                if self._chunk_live.get(self._seq) == 0:  # the retired chunk has no live rows left
                    del self._chunk_live[self._seq], self._chunks[self._seq]
                self._seq += 1
                self._chunks[self._seq] = chunk
                self._chunk_live[self._seq] = 0
            chunk[0][start : start + k].copy_(sel, non_blocking=sel.is_cuda)
            if sel.is_cuda:
                ev = torch.cuda.Event()
                ev.record()
                chunk[3] = ev
            refs = (self._seq << _CHUNK_SHIFT) + start + np.arange(k, dtype=np.int64)
            old = self._win[wr, off]
            self._win[wr, off] = refs
            self._chunk_live[self._seq] += k
            # Rewritten rows (an eviction recompute of a row still in the window) release their old refs; of a
            # duplicate (key, t) within this step one write wins and the other is released (its old ref once).
            lost = self._win[wr, off] != refs
            self._drop_refs(old[~lost])
            self._drop_refs(refs[lost])
            self.recorded += k
            self._schedule_rows(wr)

    def _block_complete(self, w: np.ndarray) -> np.ndarray:
        """Window rows of ``w`` whose block at the frontier is fully logged (last row tested first)."""
        B = self.block_rows
        rel = self._front[w] - self._base[w]
        last = rel + B - 1
        inwin = last < self._win.shape[1]
        w, rel, last = w[inwin], rel[inwin], last[inwin]
        hit = self._win[w, last] >= 0
        w, rel = w[hit], rel[hit]
        if not w.size:
            return w
        cols = rel[:, None] + np.arange(B)[None, :]
        return w[(self._win[w[:, None], cols] >= 0).all(1)]

    def _schedule_rows(self, wr: np.ndarray) -> None:
        """Queue seal jobs for every complete block at the frontier of these window rows (lock held)."""
        import queue

        cand = self._block_complete(np.unique(wr))
        B = self.block_rows
        while cand.size:
            for w in cand.tolist():
                key = self._wr_key[w]
                s = int(self._front[w])
                rel = s - int(self._base[w])
                refs = self._win[w, rel : rel + B].copy()
                try:
                    self._jobs.put_nowait((key, self._kid[key], s, refs))
                except queue.Full:
                    return  # sealed on a later record
                self._win[w, rel : rel + B] = _NO_ROW
                self._front[w] = s + B
                self._pending[key] = self._pending.get(key, 0) + 1
                self._pending_rows += B
                self._rebase(w)
            cand = self._block_complete(cand)

    def _schedule(self, key: str) -> None:
        kid = self._kid.get(key)
        if kid is not None and self._wr_of_kid[kid] >= 0:
            self._schedule_rows(np.array([self._wr_of_kid[kid]], dtype=np.int64))

    def seal_frontier(self, key: str) -> int:
        """Rows of ``key`` below this position are sealed or handed out."""
        with self._lock:
            kid = self._kid.get(key)
            w = -1 if kid is None else int(self._wr_of_kid[kid])
            return 0 if w < 0 else int(self._front[w])


    def _run(self) -> None:
        import time as _time

        while True:
            job = self._jobs.get()
            if job is None:
                return
            key, kid, s, refs = job
            try:
                with self._lock:
                    rows = self._rows_of(refs)
                    ok = not self._kclosed[kid]
                if ok:
                    for ev in {id(c[0]): c[3] for c, _ in rows}.values():
                        if ev is not None:
                            ev.synchronize()
                    blk = stack_rows(rows)
                    t0 = _time.perf_counter()
                    ref = self._do_put(blk)
                    dt = _time.perf_counter() - t0
                    nb = blk.numel() * blk.element_size()
                    stale = None
                    with self._lock:
                        if self._kclosed[kid]:
                            stale = ref  # closed meanwhile
                        else:
                            self._sealed.setdefault(key, []).append((ref, s, self.block_rows, nb))
                            self.sealed_bytes += nb
                            self.seal_put_s += dt
                    if stale is not None:
                        self._do_free([stale])
            finally:
                with self._lock:
                    self._drop_refs(refs)
                    self._pending_rows -= int(refs.shape[0])
                    self._pending[key] = self._pending.get(key, 1) - 1
                    if self._pending[key] <= 0:
                        self._pending.pop(key, None)
                    self._cond.notify_all()

    def _wait_key(self, key: str) -> None:
        """Wait until no seal job of ``key`` is in flight (lock held)."""
        while self._pending.get(key, 0) > 0:
            self._cond.wait()

    # -- hand-out ----------------------------------------------------------------
    def _put_rows(self, rows: list) -> list[tuple[Any, int, int]]:
        """Put pinned rows [(chunk, idx)] as fresh blocks of <= B rows."""
        for ev in {id(c[0]): c[3] for c, _ in rows}.values():
            if ev is not None:
                ev.synchronize()
        out = []
        for a in range(0, len(rows), self.block_rows):
            blk = stack_rows(rows[a : a + self.block_rows])
            nb = blk.numel() * blk.element_size()
            out.append((self._do_put(blk), int(blk.shape[0]), nb))
            self.tail_put_bytes += nb
        return out

    def _take_locked(self, key: str, lo: int, n: int) -> list[tuple[Any, int, int]]:
        """Blocks covering rows lo..lo+m-1 (longest logged run, m <= n); lock held."""
        sealed = self._sealed.get(key, [])
        kid = self._kid.get(key)
        if kid is None and not sealed:
            return []  # never logged here (another tensor-parallel rank logs this draft)
        w = self._window_row(self._key_id(key), key)
        out: list[tuple[Any, int, int]] = []
        cur, end = lo, lo + n
        while cur < end:
            blk = next((b for b in sealed if b[1] <= cur < b[1] + b[2]), None)
            if blk is not None:
                ref, b_lo, b_rows, nb = blk
                if b_lo == cur and b_lo + b_rows <= end:
                    out.append((ref, b_rows, nb))
                    sealed.remove(blk)
                    cur += b_rows
                    continue
                # The take starts or ends inside this block: re-split it so
                # every handle covers whole objects.
                h = self._do_get(ref)
                sealed.remove(blk)
                self._do_free([ref])
                self.resplits += 1
                parts = [(b_lo, cur), (cur, min(end, b_lo + b_rows)), (min(end, b_lo + b_rows), b_lo + b_rows)]
                for a, b in parts:
                    if b <= a:
                        continue
                    piece = h[a - b_lo : b - b_lo].clone()
                    r = self._do_put(piece)
                    pnb = piece.numel() * piece.element_size()
                    if a == cur:
                        out.append((r, b - a, pnb))
                    else:
                        sealed.append((r, a, b - a, pnb))
                cur = min(end, b_lo + b_rows)
                continue
            if w < 0:
                break
            a = cur - int(self._base[w])
            W = self._win.shape[1]
            if a < 0 or a >= W:
                break
            # Logged rows from cur, up to the next sealed block (none lies in the window: blocks are sealed from
            # the window, and rows below the frontier are never logged again).
            stop = min([end] + [b[1] for b in sealed if b[1] > cur])
            seg = self._win[w, a : min(W, a + stop - cur)]
            miss = np.flatnonzero(seg < 0)
            m = int(miss[0]) if miss.size else int(seg.shape[0])
            if m == 0:
                break  # a row the engine did not log: the draft is cut here
            refs = seg[:m].copy()
            seg[:m] = _NO_ROW
            out.extend(self._put_rows(self._rows_of(refs)))
            self._drop_refs(refs)
            cur += m
        sealed.sort(key=lambda b: b[1])
        if w >= 0:
            # Rows below the new frontier are handed out; seal from there on.
            self._front[w] = max(int(self._front[w]), cur)
            self._rebase(w)
        return out

    def take_refs(self, key: str, lo: int, n: int) -> Optional[list[tuple[Any, int, int]]]:
        """``take`` as object-store blocks [(ref, rows, bytes)]; the key stays open."""
        with self._lock:
            self._wait_key(key)
            out = self._take_locked(key, lo, n)
            self._schedule(key)
        if not out:
            return None if n else []
        return out

    def pop_refs(self, key: str, n: int) -> Optional[list[tuple[Any, int, int]]]:
        """``pop`` as object-store blocks [(ref, rows, bytes)]; closes the key."""
        with self._lock:
            self._wait_key(key)
            out = self._take_locked(key, 0, n)
            left = self._close_locked(key)
        self._do_free(left)
        if not out:
            return None if n else []
        return out

    def _close_locked(self, key: str) -> list:
        kid = self._key_id(key)
        self._close_gen.setdefault(key, self._gen)
        self._kclosed[kid] = True
        w = int(self._wr_of_kid[kid])
        if w >= 0:
            self._drop_refs(self._win[w].copy())
            self._win[w] = _NO_ROW
            self._wr_key[w] = None
            self._wr_of_kid[kid] = -1
            self._free_wr.append(w)
        return [b[0] for b in self._sealed.pop(key, [])]

    def close(self, key: str) -> None:
        with self._lock:
            self._wait_key(key)
            left = self._close_locked(key)
        self._do_free(left)

    def pop(self, key: str, n: int) -> Optional[torch.Tensor]:
        """Tensor rows (driver transport / tests): sealed blocks are fetched back."""
        blocks = self.pop_refs(key, n)
        if blocks is None:
            return None
        if not blocks:
            return torch.empty(0, 0)
        h = torch.cat([self._do_get(r) for r, _, _ in blocks])
        self._do_free([r for r, _, _ in blocks])
        return h

    def take(self, key: str, lo: int, n: int) -> Optional[torch.Tensor]:
        blocks = self.take_refs(key, lo, n)
        if blocks is None:
            return None
        if not blocks:
            return torch.empty(0, 0)
        h = torch.cat([self._do_get(r) for r, _, _ in blocks])
        self._do_free([r for r, _, _ in blocks])
        return h

    def _open_rows(self) -> int:
        used = [w for w, k in enumerate(self._wr_key) if k is not None]
        return int((self._win[used] >= 0).sum()) if used else 0

    def live_rows(self) -> int:
        with self._lock:
            return (
                self._open_rows()
                + self._pending_rows
                + sum(b[2] for bl in self._sealed.values() for b in bl)
            )

    def prune_closed(self, lag: int = 2) -> int:
        """Advance the close generation; forget the strings of keys closed ``lag`` or more generations ago.

        Their ids stay closed (ids are never reused), so rows still arriving for them are dropped.
        """
        with self._lock:
            self._gen += 1
            old = [k for k, g in self._close_gen.items() if g <= self._gen - int(lag)]
            for k in old:
                del self._close_gen[k]
                self._kid.pop(k, None)
            return len(old)

    def bookkeeping(self) -> dict[str, float]:
        """Sizes for the long-run leak probe (leak_probe.py)."""
        with self._lock:
            return {
                "open_keys": float(sum(1 for k in self._wr_key if k is not None)),
                "closed_keys": float(len(self._close_gen)),
                "rows": float(self._open_rows()),
                "chunks": float(len(self._chunks)),
            }

    def stop(self) -> None:
        self._jobs.put(None)
        self._thread.join()


# ---------------------------------------------------------------------------
# Moving hidden states: generation worker -> object store -> verifying learner
# ---------------------------------------------------------------------------
#
# A draft's hidden states are d x 2 bytes per drafted token, but only drafts
# the lazy scan does not accept whole ever read a q row, and those read only
# the rows the scan visits (top candidates down to tau). So the generation
# worker puts each draft's rows of one drafting round into the Ray object
# store once, in blocks of ``HIDDEN_BLOCK_ROWS`` rows, and only the handles
# travel engine -> driver -> learner. The learner fetches a block the first
# time its lazy scan asks for a row in it (``HiddenRows``): fully accepted
# drafts fetch nothing, and the data moves from the generation node's object
# store to the learner's node without passing through the driver. The driver
# frees the objects (``free_hidden``) once the draft is verified or dropped.


@dataclass
class HiddenSeg:
    """Handle to one block of a draft's hidden states in the object store.

    ``ref``: ObjectRef of a [rows, d] tensor (put by the generation worker
    that drafted these rows; it owns the object), None once released. The
    driver holds only these handles, never the rows.
    """

    ref: Any
    rows: int
    nbytes: int

    def release(self) -> None:
        """Drop this handle's reference (``free_hidden`` also deletes the object)."""
        self.ref = None


@dataclass
class HiddenRefs:
    """A draft's hidden states as object-store block handles, in row order.

    Draft tuple index 7 under ``hidden_transport="object_store"``; a [g, d]
    tensor under ``"driver"``. A cohort draft chains the blocks of its
    drafting rounds; the handles are shared with the cohort dict, so releasing
    them clears the draft's references everywhere on the driver.
    """

    segs: list[HiddenSeg] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return sum(s.rows for s in self.segs)

    @property
    def nbytes(self) -> int:
        return sum(s.nbytes for s in self.segs)

    def release(self) -> None:
        for s in self.segs:
            s.release()

    @classmethod
    def from_blocks(cls, blocks: list[tuple]) -> "HiddenRefs":
        """From ``put_hidden``'s [(ref, rows, bytes)]."""
        return cls([HiddenSeg(*b) for b in blocks])


def ray_free(refs: list) -> None:
    """Delete objects from every object store now (Ray internal API).

    Called by the driver once the learner calls that could read them have
    returned, so a lingering Python reference anywhere (driver, owner,
    learner) cannot keep a step's hidden states alive.
    """
    from ray._private import internal_api

    internal_api.free(refs, local_only=False)


def free_hidden(hiddens, free: Optional[Callable[[list], None]] = None) -> int:
    """Release the handles of ``HiddenRefs`` in ``hiddens`` and free their objects.

    Other entries (tensors, None) are ignored. Returns the bytes released.
    """
    refs, nbytes = [], 0
    for h in hiddens:
        if not isinstance(h, HiddenRefs):
            continue
        for s in h.segs:
            if s.ref is not None:
                refs.append(s.ref)
                nbytes += s.nbytes
                s.release()
    if refs:
        (free or ray_free)(refs)
    return nbytes


# Rows per object-store object. The learner fetches whole blocks, so this
# bounds what a rejected draft pulls beyond the rows its lazy scan visits
# (at most one partial block per visited run); 256 rows are 2 MB at d = 4096
# and 3.7 MB at d = 7168 (bf16), well above Ray's inline-object size. Each
# block is one ray.put, so larger blocks cut per-put overhead at the step
# deadline (THUNDERSYNC_HIDDEN_BLOCK_ROWS overrides it for measurements).


def put_hidden(
    h: torch.Tensor,
    put: Optional[Callable[[Any], Any]] = None,
    block_rows: int = HIDDEN_BLOCK_ROWS,
) -> list[tuple]:
    """Generation-worker side: put rows [n, d] in the object store once.

    One object per ``block_rows`` rows. Returns [(ref, rows, bytes)] per
    block, the only thing sent back to the driver (``HiddenRefs.from_blocks``).
    """
    out = []
    for a in range(0, int(h.shape[0]), block_rows):
        b = own_block(h[a : a + block_rows])
        out.append((put_hidden_block(b, put), int(b.shape[0]), b.numel() * b.element_size()))
    return out


def own_block(b: torch.Tensor) -> torch.Tensor:
    """``b`` as a tensor that owns exactly its own bytes.

    Pickling a torch tensor (``ray.put``) ships its whole storage, and a row
    slice of a larger tensor is a view of that storage (``.contiguous()`` is a
    no-op on it): putting the 256-row blocks of an n-row draft as views would
    ship n rows per block, n^2/256 rows per draft. Clone unless ``b`` already
    owns exactly its bytes.
    """
    if (
        b.is_contiguous()
        and b.storage_offset() == 0
        and b.untyped_storage().nbytes() == b.numel() * b.element_size()
    ):
        return b
    return b.clone()


def cat_hidden(parts: list) -> Any:
    """A draft's hidden states from its per-segment parts.

    Tensors (``hidden_transport="driver"``) are concatenated, skipping empty
    segments; ``HiddenRefs`` are chained (no data is touched).
    """
    if parts and isinstance(parts[0], HiddenRefs):
        return HiddenRefs([s for p in parts for s in p.segs if s.rows])
    parts = [h for h in parts if h.numel()]
    return torch.cat(parts) if parts else torch.empty(0, 0)


def hidden_num_rows(h: Any) -> int:
    """Rows of draft tuple index 7 (tensor, ``HiddenRefs`` or ``HiddenRecompute``)."""
    if isinstance(h, torch.Tensor):
        return int(h.shape[0])
    return int(h.rows)




class HiddenRows:
    """Learner side: rows of one draft's hidden states, fetched on first use.

    ``src`` is a [g, d] tensor (driver transport: already here) or a
    ``HiddenRefs``. ``__call__(positions)`` returns rows [k, d] (CPU). With
    refs, each block that holds a requested position is fetched once
    (``get``, ``ray.get`` by default) and cached for this draft; untouched
    blocks are never fetched. ``fetched_bytes`` counts what was pulled.
    Tensor-parallel ranks of a replica run the same lazy scan, so they ask
    for the same positions in the same order and fetch the same objects
    (immutable, so every rank sees identical bytes; ranks on one node share
    the node's copy in the object store).
    """

    def __init__(
        self,
        src: Any,
        get: Optional[Callable[[Any], Any]] = None,
        prefetched: Optional[dict] = None,
    ):
        self.src = src
        self.fetched_bytes = 0
        # Storage bytes actually deserialized (== fetched_bytes for blocks
        # that own their bytes; see ``own_block``) and wall time spent
        # waiting for blocks inside the scan.
        self.fetched_real_bytes = 0
        self.fetch_s = 0.0
        self._get = get
        # ref -> concurrent.futures.Future of the block, from ``HiddenPrefetch``.
        self._prefetched = prefetched if prefetched is not None else {}
        if isinstance(src, HiddenRefs):
            self._starts = []
            a = 0
            for s in src.segs:
                self._starts.append(a)
                a += s.rows
            self._cache: dict[int, torch.Tensor] = {}

    def _block(self, k: int) -> torch.Tensor:
        h = self._cache.get(k)
        if h is None:
            seg = self.src.segs[k]
            if seg.ref is None:
                raise RuntimeError("hidden states of this draft were already released")
            import time as _time

            t0 = _time.perf_counter()
            fut = self._prefetched.get(_ref_key(seg.ref))
            if fut is not None and not fut.cancelled():
                h = fut.result()
            else:
                get = self._get
                if get is None:
                    import ray

                    get = ray.get
                h = hidden_block(get(seg.ref))
            self.fetch_s += _time.perf_counter() - t0
            self.fetched_real_bytes += h.untyped_storage().nbytes()
            if h.shape[0] != seg.rows:
                raise RuntimeError(
                    f"hidden block has {h.shape[0]} rows, expected {seg.rows}"
                )
            self._cache[k] = h
            self.fetched_bytes += seg.nbytes
        return h

    def rows_at(self, positions) -> torch.Tensor:
        """``self(positions)`` for an int64 array of positions, vectorized per block (DS2): the same rows, in order,
        fetched through the same ``_block`` (same accounting)."""
        pos = np.ascontiguousarray(positions, dtype=np.int64)
        if not isinstance(self.src, HiddenRefs):
            return self.src[torch.from_numpy(pos)]
        starts = np.asarray(self._starts, dtype=np.int64)
        blk = np.searchsorted(starts, pos, side="right") - 1
        out = None
        for k in np.unique(blk):  # ascending block order; every touched block is fetched once, as in __call__
            js = np.nonzero(blk == k)[0]
            b = self._block(int(k))
            if out is None:
                out = b.new_empty((pos.size,) + tuple(b.shape[1:]))
            out[torch.from_numpy(js)] = b[torch.from_numpy(pos[js] - starts[k])]
        if out is None:
            return torch.stack([])  # no positions: as __call__ (raises)
        return out

    def rows_at_device(self, positions, device) -> torch.Tensor:
        """``rows_at(positions)`` on ``device`` (THUNDERSYNC_HIDDEN_DEVICE_GATHER, DS2's q rows): every touched block
        is copied to the device once per draft into a [g, d] device buffer, and the rows are indexed there. The same
        bytes in the same order as ``rows_at(...).to(device)``, without the host gather and the pinned staging copy
        of every chunk (Ultra-L27: ~25-40 ms of host time per 2048-position chunk, FLOOR_ULTRA §6.5). Blocks go
        through the same ``_block`` (same fetch accounting). Per rank: at most g x d x itemsize bytes (one draft's
        hidden states; 268 MB at g 8192, d 8192, fp32), released with this object."""
        pos = np.ascontiguousarray(positions, dtype=np.int64)
        if not isinstance(self.src, HiddenRefs):
            return self.src[torch.from_numpy(pos.copy())].to(device)
        cap = float(os.environ.get("THUNDERSYNC_HIDDEN_DEVICE_GATHER_MAX_GB", "2") or 2) * 2**30
        if self.src.nbytes > cap:
            # Long rows (e.g. 32k+ contexts): the [g, d] buffer would exceed the cap; the host path (same bytes).
            return self.rows_at(pos).to(device)
        starts = np.asarray(self._starts, dtype=np.int64)
        blk = np.searchsorted(starts, pos, side="right") - 1
        buf = self.__dict__.get("_dev_buf")
        filled = self.__dict__.setdefault("_dev_filled", set())
        for k in np.unique(blk):
            k = int(k)
            if k in filled:
                continue
            b = self._block(k)
            if buf is None:
                buf = torch.empty((self.src.rows,) + tuple(b.shape[1:]), dtype=b.dtype, device=device)
                self._dev_buf = buf
            buf[int(starts[k]) : int(starts[k]) + b.shape[0]].copy_(b)
            filled.add(k)
        if buf is None:
            return torch.stack([])  # no positions: as __call__ (raises)
        return buf[torch.from_numpy(pos.copy()).to(device)]  # copy: a reversed view has a negative stride

    def __call__(self, positions: list[int]) -> torch.Tensor:
        if not isinstance(self.src, HiddenRefs):
            return self.src[positions]
        import bisect

        out = []
        for t in positions:
            k = bisect.bisect_right(self._starts, t) - 1
            out.append(self._block(k)[t - self._starts[k]])
        return torch.stack(out)


def _ref_key(ref: Any) -> Any:
    """Hashable identity of an object-store handle (ObjectRef id, else the object)."""
    b = getattr(ref, "binary", None)
    return b() if callable(b) else ref


class HiddenPrefetch:
    """Verifier side: fetch the blocks of upcoming drafts in a background thread.

    ``submit(hiddens)`` starts fetching every block of one verification batch
    (``HiddenRefs`` of its drafts); ``futures`` maps a block's handle to its
    Future, which ``HiddenRows`` waits on instead of a blocking ``ray.get``.
    Bounded: at most ``max_batches`` batches are held, and a batch is only
    submitted while the held bytes stay under ``max_bytes`` (the first batch
    always is: one batch is what an on-demand scan may cache anyway); a batch
    that is not submitted is fetched on demand as before. ``release`` drops the
    oldest batch's blocks. Moves bytes only: the scan reads the same blocks.
    """

    def __init__(
        self,
        get: Optional[Callable[[Any], Any]] = None,
        max_bytes: int = 8 << 30,
        max_batches: int = 2,
    ):
        from concurrent.futures import ThreadPoolExecutor

        self._get = get
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="ts-hidden-prefetch")
        self.max_bytes = max_bytes
        self.max_batches = max_batches
        self.futures: dict[Any, Any] = {}
        self._held: list[tuple[list, int]] = []
        self.submitted_bytes = 0
        self.skipped_batches = 0

    def held_bytes(self) -> int:
        return sum(nb for _, nb in self._held)

    def submit(self, hiddens: list) -> bool:
        segs = [
            seg
            for h in hiddens
            if isinstance(h, HiddenRefs)
            for seg in h.segs
            if seg.ref is not None
        ]
        nb = sum(seg.nbytes for seg in segs)
        if self._held and (
            len(self._held) >= self.max_batches or self.held_bytes() + nb > self.max_bytes
        ):
            self.skipped_batches += 1
            return False
        get = self._get
        if get is None:
            import ray

            get = ray.get
        keys = []
        for seg in segs:
            k = _ref_key(seg.ref)
            if k not in self.futures:
                self.futures[k] = self._pool.submit(lambda r: hidden_block(get(r)), seg.ref)
                keys.append(k)
        self._held.append((keys, nb))
        self.submitted_bytes += nb
        return True

    def release(self) -> None:
        """Drop the oldest held batch's blocks (cancel any not yet fetched)."""
        if not self._held:
            return
        keys, _ = self._held.pop(0)
        for k in keys:
            f = self.futures.pop(k, None)
            if f is not None:
                f.cancel()

    def close(self) -> None:
        while self._held:
            self.release()
        self._pool.shutdown(wait=True)


# Rows per full-vocabulary log-softmax when log-probs of many positions are
# gathered at once (bounds the [rows, V] fp32 temporaries, as training's
# gathered TP log-probs do).
GATHER_CHUNK_ROWS = 4096

VocabGather = Callable[[torch.Tensor], torch.Tensor]


def no_gather(x: torch.Tensor) -> torch.Tensor:
    """The vocab gather of tensor-parallel size 1."""
    return x


class TPVocabGather:
    """Rank-order all-gather of vocab shards over ``group`` (a ``VocabGather`` that also names its group,
    so the canonical-block log-probs can work on the local shard: ``GatheredLogprobRows.token_logprobs``)."""

    def __init__(self, group) -> None:
        self.group = group

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        from nemo_rl.models.megatron.zero_train_gen_mismatch import _all_gather_vocab

        return _all_gather_vocab(x, self.group)


def gathered_log_softmax(
    local: torch.Tensor,
    gather: VocabGather,
    vocab_limit: int,
    anchors: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Full-vocabulary fp32 log-prob rows of vocab-sharded logits [k, V/TP].

    ``gather`` concatenates the tensor-parallel ranks' shards along the last
    dimension in rank (= vocabulary) order (``no_gather`` at TP=1): the
    engine's logits all-gather. Then the engine's ``raw_logprobs`` path: fp32
    cast and one full-row ``log_softmax``. A collective at TP>1: every rank
    of the group must call it with the same row count.
    """
    z = gather(local.contiguous())
    from nemo_rl.models.megatron import vocab_lse

    if vocab_lse.enabled():
        # Canonical-block LSE (vocab_lse.py): fp32(z) - LSE, the definition the trainer and the
        # engine share under NEMO_RL_VOCAB_LSE=canonical_block. Exact top-p is refused there.
        if anchors is not None or exact_nucleus.active():
            raise RuntimeError("NEMO_RL_VOCAB_LSE=canonical_block does not compose with exact top-p yet")
        return vocab_lse.logprob_rows(z, vocab_limit)
    # Exact top-p (exact_nucleus.py): the masked log_softmax the engine samples
    # from and reports; plain log_softmax when off. ``anchors`` [k, C]: the
    # sticky-nucleus anchors of these rows (None: self).
    return exact_nucleus.policy_log_softmax(z[..., :vocab_limit], anchors)


class GatheredLogprobRows:
    """Log-prob rows of one draft's vocab-sharded logits, gathered on demand.

    ``local`` [n, V/TP]: this rank's logits shard per position. Rows are
    computed by ``gathered_log_softmax`` only where asked (lazy scan) and,
    for the drafted tokens' log-probs, ``GATHER_CHUNK_ROWS`` rows at a time.
    Under batch-invariant kernels a row's log-softmax does not depend on how
    many rows run together, so both give the bits of training's gathered
    log-probs. Memory: the local shard plus one chunk of full rows, instead of
    [n, V] fp32. Every call is a collective at TP>1.
    """

    def __init__(
        self,
        local: torch.Tensor,
        gather: VocabGather,
        vocab_limit: int,
        chunk_rows: int = GATHER_CHUNK_ROWS,
    ):
        self.local = local
        self.gather = gather
        self.vocab_limit = vocab_limit
        self.chunk_rows = chunk_rows
        # Sticky-nucleus anchors per row [n, C] (None: self), set by the verifier.
        self.anchors: Optional[torch.Tensor] = None

    def _anchors(self, idx_or_slice) -> Optional[torch.Tensor]:
        if self.anchors is None:
            return None
        if isinstance(idx_or_slice, torch.Tensor):
            idx_or_slice = idx_or_slice.to(self.anchors.device)
        return self.anchors[idx_or_slice].to(self.local.device)

    def __call__(self, idx: torch.Tensor) -> torch.Tensor:
        return gathered_log_softmax(
            self.local[idx.to(self.local.device)],
            self.gather,
            self.vocab_limit,
            self._anchors(idx),
        )

    def token_logprobs(self, tokens: torch.Tensor) -> torch.Tensor:
        """[k] fp32 log-prob of ``tokens[j]`` at row j, for rows 0..k-1."""
        from nemo_rl.models.megatron import vocab_lse

        if vocab_lse.enabled() and self.anchors is None and (
            self.gather is no_gather
            or (isinstance(self.gather, TPVocabGather) and int(self.local.shape[-1]) % vocab_lse.BLOCK == 0)
        ):
            # Sharded only when the shard is whole canonical blocks (lse_sharded's precondition); any other
            # shard width takes the gathered row, the same definition (fp32(z) - LSE of the full row).
            return self._token_logprobs_sharded(tokens)
        k = tokens.numel()
        out = []
        for a in range(0, k, self.chunk_rows):
            b = min(k, a + self.chunk_rows)
            lp = gathered_log_softmax(
                self.local[a:b], self.gather, self.vocab_limit, self._anchors(slice(a, b))
            )
            out.append(lp.gather(-1, tokens[a:b].view(-1, 1).to(lp.device))[:, 0])
            del lp
        if not out:
            return torch.zeros(0, dtype=torch.float32, device=self.local.device)
        return torch.cat(out)

    def _token_logprobs_sharded(self, tokens: torch.Tensor) -> torch.Tensor:
        """Canonical-block log-probs of the drafted tokens from this rank's vocab shard (vocab_lse.py): the
        LSE from the shard's block partials plus an all-gather of 8 bytes per block per row, the token's
        logit on its owner rank, and an all-reduce in which every other rank adds +0. Bitwise the
        gathered-row value ``fp32(z_x) - LSE`` (TP-invariant partials), without gathering [k, V] rows."""
        from nemo_rl.models.megatron import vocab_lse

        k = tokens.numel()
        local = self.local[:k]
        group = getattr(self.gather, "group", None)
        world = 1 if group is None else torch.distributed.get_world_size(group)
        width = int(local.shape[-1])
        offset = 0 if group is None else torch.distributed.get_rank(group) * width
        lse = vocab_lse.lse_sharded(local, offset, self.vocab_limit, group)
        tok = tokens.to(local.device).view(-1).long()
        own = (tok >= offset) & (tok < offset + width)
        zx = local.gather(-1, (tok - offset).clamp(0, width - 1).view(-1, 1))[:, 0].float()
        val = torch.where(own, zx - lse, torch.zeros_like(lse))
        if world > 1:
            torch.distributed.all_reduce(val, group=group)
        return val


def head_stash_bytes(
    vocab: int, hidden: int, tp: int, versions: int = 1, elem_bytes: int = 2
) -> int:
    """GPU bytes of the LM-head stash on one tensor-parallel rank.

    Each rank keeps its vocab-parallel shard per pending version: ceil(V/TP)
    x d elements (Megatron pads V to a multiple of TP, so V/TP exactly).
    """
    return versions * (-(-vocab // tp)) * hidden * elem_bytes




class DeferredLogprobRows(GatheredLogprobRows):
    """Log-prob rows of one row's fp32 head computed from its head input in chunks.

    ``x`` [n, h]: the LM head's input at positions plen - 1 .. n - 1 of a
    verification forward (``deferred_head.verification_logits``); ``weight`` the
    fp32 head [V, h]. A row is ``log_softmax(fp32_gemm(x[i], W)[:vocab_limit])``,
    the materialized path's ops on the same row (row-invariant GEMM, row-wise
    log-softmax), so values are bitwise those of whole-batch logits; at most
    ``chunk_rows`` (default ``deferred_head.CHUNK_TOKENS`` = 4096) rows of
    [V] fp32 exist at a time. Memory per verification forward: the head input
    (tokens x h) instead of tokens x V x 4 B (68 GiB at 34 x 4096 x 131k).
    """

    def __init__(self, x: torch.Tensor, weight: torch.Tensor, vocab_limit: int, chunk_rows: Optional[int] = None):
        from nemo_rl.models.megatron.deferred_head import CHUNK_TOKENS

        self.local = x
        self.weight = weight
        self.gather = None
        self.vocab_limit = vocab_limit
        self.chunk_rows = int(chunk_rows or CHUNK_TOKENS)
        self.anchors = None

    def _rows(self, x: torch.Tensor, anchors: Optional[torch.Tensor] = None) -> torch.Tensor:
        from nemo_rl.models.megatron import fp16_gemm

        logits = fp16_gemm.fp32_gemm(x.to(torch.float32), self.weight)
        return exact_nucleus.policy_log_softmax(logits[..., : self.vocab_limit], anchors)

    def __call__(self, idx: torch.Tensor) -> torch.Tensor:
        idx = idx.to(self.local.device)
        out = [
            self._rows(
                self.local[idx[a : a + self.chunk_rows]],
                self._anchors(idx[a : a + self.chunk_rows]),
            )
            for a in range(0, idx.numel(), self.chunk_rows)
        ]
        if not out:
            return torch.zeros(0, self.vocab_limit, dtype=torch.float32, device=self.local.device)
        return torch.cat(out) if len(out) > 1 else out[0]

    def token_logprobs(self, tokens: torch.Tensor) -> torch.Tensor:
        k = tokens.numel()
        out = []
        for a in range(0, k, self.chunk_rows):
            b = min(k, a + self.chunk_rows)
            lp = self._rows(self.local[a:b], self._anchors(slice(a, b)))
            out.append(lp.gather(-1, tokens[a:b].view(-1, 1).to(lp.device))[:, 0])
            del lp
        if not out:
            return torch.zeros(0, dtype=torch.float32, device=self.local.device)
        return torch.cat(out)


def verify_engine_q(
    p_logprobs: Union[torch.Tensor, GatheredLogprobRows],
    draft: torch.Tensor,
    engine_logq: torch.Tensor,
    q_logprob_rows: Callable[[torch.Tensor], torch.Tensor],
    generator: Optional[torch.Generator] = None,
    null_m: float = 1.0,
    band: Optional[float] = None,
) -> tuple[int, int, torch.Tensor, int]:
    """Block-verify one draft with p from training and q from the engine.

    ``p_logprobs``: [g+1, V] fp32 log-probs of the training model at
    theta_{k+1} (log-softmax rows, as training computes them), or a
    ``GatheredLogprobRows`` over its vocab-sharded logits (tensor parallelism:
    rows gathered on demand, the same values).
    ``engine_logq``: [g] fp32 log q(x_i), the engine's log-probs of the drafted
    tokens. ``q_logprob_rows(idx)``: [k, V] fp32 q log-prob rows at positions
    ``idx`` (reconstructed from the engine's hidden states), called only where
    the lazy scan needs a residual mass and for the residual sample.

    Same rule and generator calls as ``block_verify_logits`` (one ``rand(g+1)``,
    one ``multinomial``), with log-prob rows standing in for logits (their
    logsumexp is 0). Returns (tau, y, logp, residual masses computed); logp are
    entries of ``p_logprobs``, i.e. training's own log-probs of the kept tokens.
    ``null_m`` > 1: null-augmented block verification (y = ``NULL_TOKEN``: BOT,
    the draft yields no output). ``band``: banded IS (``block_verification``).
    """
    from thundersync_rl.block_verification_fused import block_verify_stats_lazy

    g = draft.numel()
    if isinstance(p_logprobs, GatheredLogprobRows):
        dev = p_logprobs.local.device
        lpx = p_logprobs.token_logprobs(draft.to(dev))
        p_rows = p_logprobs
    else:
        dev = p_logprobs.device
        lpx = p_logprobs[torch.arange(g, device=dev), draft]

        def p_rows(idx):
            return p_logprobs[idx]

    lqx = engine_logq.to(device=dev, dtype=torch.float32)
    zeros = torch.zeros(g + 1, dtype=torch.float32, device=dev)
    return block_verify_stats_lazy(
        lpx, lqx, zeros, zeros[:g], p_rows, q_logprob_rows, generator, null_m, band
    )


def verify_engine_q_grid(
    p_logprobs: Union[torch.Tensor, GatheredLogprobRows],
    draft: torch.Tensor,
    engine_logq: torch.Tensor,
    q_logprob_rows: Callable[[torch.Tensor], torch.Tensor],
    generator: Optional[torch.Generator],
    envelopes: tuple[float, ...],
) -> tuple[dict[float, tuple[int, int, Optional[float]]], torch.Tensor, int]:
    """``verify_engine_q`` at every ⊥-BV envelope in ``envelopes``, one set of draws.

    Returns ({M: (tau, y, log p(y) or None for BOT)}, log p of the drafted
    tokens [g], residual masses computed); see
    ``block_verification_fused.block_verify_stats_lazy_grid``.
    """
    from thundersync_rl.block_verification_fused import block_verify_stats_lazy_grid

    g = draft.numel()
    if isinstance(p_logprobs, GatheredLogprobRows):
        dev = p_logprobs.local.device
        lpx = p_logprobs.token_logprobs(draft.to(dev))
        p_rows = p_logprobs
    else:
        dev = p_logprobs.device
        lpx = p_logprobs[torch.arange(g, device=dev), draft]

        def p_rows(idx):
            return p_logprobs[idx]

    lqx = engine_logq.to(device=dev, dtype=torch.float32)
    zeros = torch.zeros(g + 1, dtype=torch.float32, device=dev)
    out, n_eval = block_verify_stats_lazy_grid(
        lpx, lqx, zeros, zeros[:g], p_rows, q_logprob_rows, generator, envelopes
    )
    return out, lpx, n_eval


def _p_stats(p_logprobs, draft):
    g = draft.numel()
    if isinstance(p_logprobs, GatheredLogprobRows):
        dev = p_logprobs.local.device
        return dev, p_logprobs.token_logprobs(draft.to(dev)), p_logprobs
    dev = p_logprobs.device

    def p_rows(idx):
        return p_logprobs[idx]

    return dev, p_logprobs[torch.arange(g, device=dev), draft.to(dev)], p_rows


def verify_engine_q_walk(
    p_logprobs: Union[torch.Tensor, GatheredLogprobRows],
    draft: torch.Tensor,
    engine_logq: torch.Tensor,
    q_logprob_rows: Callable[[torch.Tensor], torch.Tensor],
    generator: Optional[torch.Generator],
    walk: dict,
) -> dict:
    """One continuation piece of a carried block (``carried_walk.carried_piece``).

    Placement-agnostic: the inputs are those of ``verify_engine_q`` plus the
    driver's walk state {"log_b", "floor", "n", "final", "rule"}. q rows are read
    only when a decision is due and fails (the scan of this piece).
    """
    from thundersync_rl.carried_walk import DecisionRule, carried_piece

    g = draft.numel()
    dev, lpx, p_rows = _p_stats(p_logprobs, draft)
    lqx = engine_logq.to(device=dev, dtype=torch.float32)
    zeros = torch.zeros(g + 1, dtype=torch.float32, device=dev)
    out = carried_piece(
        lpx, lqx, zeros, zeros[:g], p_rows, q_logprob_rows, generator,
        walk, DecisionRule.of(walk["rule"]), bool(walk["final"]),
    )
    lp = lpx.tolist()
    out["token_logprobs"] = lp
    out["log_ratio"] = (lpx.float() - lqx).tolist()
    if out["decision"] == "reject":
        acc = out["tau"] - int(walk["n"])
        out.update(accepted=acc, next=out["next"], logprobs=lp[:acc] + [out["next_logp"]],
                   draft_logq=engine_logq[:acc].tolist())
    return out


def verify_engine_q_scan(
    p_logprobs: Union[torch.Tensor, GatheredLogprobRows],
    draft: torch.Tensor,
    engine_logq: torch.Tensor,
    q_logprob_rows: Callable[[torch.Tensor], torch.Tensor],
    generator: Optional[torch.Generator],
    scan: dict,
) -> dict:
    """Rejection scan of a carried block (``carried_walk.carried_scan``).

    ``draft``: block positions [pos0, stop). The walk uses the driver's carried
    log-ratios ``scan["log_ratio"]`` (the values the pieces computed); this
    forward's own log p of the same tokens must equal them bitwise (counted in
    "walk_ratio_mismatch": under batch invariance and the prefix-state cache, 0).
    """
    from thundersync_rl.carried_walk import carried_scan

    g = draft.numel()
    dev, lpx, p_rows = _p_stats(p_logprobs, draft)
    lqx = engine_logq.to(device=dev, dtype=torch.float32)
    mine = (lpx.float() - lqx).tolist()
    carried = [float(x) for x in scan["log_ratio"]]
    zeros = torch.zeros(g + 1, dtype=torch.float32, device=dev)
    out = carried_scan(
        carried, float(scan["log_b0"]), int(scan["pos0"]), int(scan["floor"]),
        int(scan["stop"]), zeros, zeros[:g], p_rows, q_logprob_rows, generator,
    )
    out["walk_ratio_mismatch"] = sum(
        a != b for a, b in zip(mine, carried)
    )
    acc = out["tau"] - int(scan["pos0"])
    lp = lpx.tolist()
    out.update(accepted=acc, logprobs=lp[:acc] + [out["next_logp"]],
               draft_logq=engine_logq[:acc].tolist(), token_logprobs=lp)
    return out
