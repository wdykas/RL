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

import torch

# Sampling-params attribute that tags a draft request for hidden-state logging.
HIDDEN_KEY_ATTR = "ts_hidden_key"
# Key of the per-call transport statistics in ``pop_draft_hidden`` /
# ``take_draft_hidden`` results (not a draft key).
HIDDEN_STATS_KEY = "__ts_hidden_stats__"
# Sampling-params attribute of a request that continues an anchored rollout at
# theta_k (anchor horizon): its key in ``register_engine_anchors``.
ANCHOR_KEY_ATTR = "ts_anchor_key"


def step_rows(
    request_ids: list[int],
    kv_offsets: list[int],
    query_lengths: list[int],
    lookup: Callable[[int], Optional[tuple[str, int]]],
) -> list[tuple[int, str, int]]:
    """Rows of one engine forward that sample a tagged request's next token.

    The forward produces one LM-head row per active request (request order),
    the row of its last input token, from which that request's next token is
    sampled. ``lookup(request_id)`` gives (draft key, prompt length) for
    tagged requests, else None. Returns (row, key, t): row ``row`` produced
    generated token ``t`` of draft ``key``, where t = (position of the last
    input token) + 1 - prompt length. Rows with t < 0 (an intermediate prefill
    chunk, which samples nothing) are skipped. Recomputation after an engine
    eviction rewrites the same t, so a draft's rows are keyed by position, not
    by arrival order.
    """
    out = []
    for row, (rid, kv, ql) in enumerate(zip(request_ids, kv_offsets, query_lengths)):
        info = lookup(int(rid))
        if info is None:
            continue
        key, plen = info
        t = int(kv) + int(ql) - plen
        if t >= 0:
            out.append((row, key, t))
    return out


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
        self._closed: set[str] = set()
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


# Rows per object-store block; also the sealing granularity of
# ``SealingHiddenLog`` (defined here so both use the same value).
HIDDEN_BLOCK_ROWS = int(os.environ.get("THUNDERSYNC_HIDDEN_BLOCK_ROWS", "256"))


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

    Bounded: seal jobs are tiny (key, start) notices in a queue of at most
    ``max_pending`` entries; when it is full a block is simply sealed later
    (``record`` never blocks the engine). Pinned rows are dropped as soon as
    their block is put, so host memory holds each row once (pinned or in the
    object store), as without sealing.
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
        # key's hand-out frontier; key -> seal frontier; key -> jobs in flight.
        self._sealed: dict[str, list[tuple[Any, int, int, int]]] = {}
        self._next_seal: dict[str, int] = {}
        self._pending: dict[str, int] = {}
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
        if self._put is not None:
            return self._put(b)
        import ray

        return ray.put(b)

    def _do_get(self, ref: Any) -> torch.Tensor:
        if self._getf is not None:
            return self._getf(ref)
        import ray

        return ray.get(ref)

    def _do_free(self, refs: list) -> None:
        if not refs:
            return
        (self._freef or ray_free)(refs)

    # -- engine side -----------------------------------------------------------
    def record(self, entries: list[tuple[int, str, int]], hidden: torch.Tensor) -> None:
        with self._lock:
            # Rows already sealed (an eviction recompute rewrites the same t,
            # bitwise the same under batch invariance) are dropped.
            entries = [
                e for e in entries if e[2] >= self._next_seal.get(e[1], 0)
            ]
        if not entries:
            return
        super().record(entries, hidden)
        with self._lock:
            for key in {k for _, k, _ in entries}:
                self._schedule(key)

    def _schedule(self, key: str) -> None:
        """Queue seal jobs for every complete block at the key's frontier (lock held)."""
        import queue

        if key in self._closed:
            return
        slots = self._rows.get(key)
        if not slots:
            return
        s = self._next_seal.get(key, 0)
        B = self.block_rows
        while all(t in slots for t in range(s, s + B)):
            try:
                self._jobs.put_nowait((key, s))
            except queue.Full:
                return  # sealed on a later record
            self._pending[key] = self._pending.get(key, 0) + 1
            s += B
            self._next_seal[key] = s

    def _run(self) -> None:
        import time as _time

        while True:
            job = self._jobs.get()
            if job is None:
                return
            key, s = job
            try:
                with self._lock:
                    slots = self._rows.get(key, {})
                    rows = [slots.get(t) for t in range(s, s + self.block_rows)]
                    ok = key not in self._closed and all(r is not None for r in rows)
                if ok:
                    for ev in {id(c[0]): c[3] for c, _ in rows}.values():
                        if ev is not None:
                            ev.synchronize()
                    blk = torch.stack([c[0][i] for c, i in rows])
                    t0 = _time.perf_counter()
                    ref = self._do_put(blk)
                    dt = _time.perf_counter() - t0
                    nb = blk.numel() * blk.element_size()
                    stale = None
                    with self._lock:
                        slots = self._rows.get(key, {})
                        if key in self._closed or any(
                            slots.get(t) is not r
                            for t, r in zip(range(s, s + self.block_rows), rows)
                        ):
                            stale = ref  # closed or taken meanwhile
                        else:
                            for t in range(s, s + self.block_rows):
                                c, _ = slots.pop(t)
                                c[2] -= 1
                            self._sealed.setdefault(key, []).append(
                                (ref, s, self.block_rows, nb)
                            )
                            self.sealed_bytes += nb
                            self.seal_put_s += dt
                    if stale is not None:
                        self._do_free([stale])
            finally:
                with self._lock:
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
            blk = torch.stack([c[0][i] for c, i in rows[a : a + self.block_rows]])
            nb = blk.numel() * blk.element_size()
            out.append((self._do_put(blk), int(blk.shape[0]), nb))
            self.tail_put_bytes += nb
        return out

    def _take_locked(self, key: str, lo: int, n: int) -> list[tuple[Any, int, int]]:
        """Blocks covering rows lo..lo+m-1 (longest logged run, m <= n); lock held."""
        sealed = self._sealed.get(key, [])
        slots = self._rows.get(key, {})
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
            run = []
            while cur + len(run) < end and (cur + len(run)) in slots and not any(
                b[1] == cur + len(run) for b in sealed
            ):
                run.append(slots.pop(cur + len(run)))
            if not run:
                break  # a row the engine did not log: the draft is cut here
            for c, _ in run:
                c[2] -= 1
            out.extend(self._put_rows(run))
            cur += len(run)
        sealed.sort(key=lambda b: b[1])
        # Rows below the new frontier are handed out; seal from there on.
        self._next_seal[key] = max(self._next_seal.get(key, 0), cur)
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
        self._closed.add(key)
        for c, _ in self._rows.pop(key, {}).values():
            c[2] -= 1
        left = [b[0] for b in self._sealed.pop(key, [])]
        self._next_seal.pop(key, None)
        return left

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

    def live_rows(self) -> int:
        with self._lock:
            return sum(len(s) for s in self._rows.values()) + sum(
                b[2] for bl in self._sealed.values() for b in bl
            )

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
    if put is None:
        import ray

        put = ray.put
    out = []
    for a in range(0, int(h.shape[0]), block_rows):
        b = own_block(h[a : a + block_rows])
        out.append((put(b), int(b.shape[0]), b.numel() * b.element_size()))
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
    """Rows of draft tuple index 7 (tensor or ``HiddenRefs``)."""
    return h.rows if isinstance(h, HiddenRefs) else int(h.shape[0])


def hidden_nbytes(h: Any) -> int:
    if h is None:
        return 0
    if isinstance(h, HiddenRefs):
        return h.nbytes
    return h.numel() * h.element_size()


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
                h = get(seg.ref)
            self.fetch_s += _time.perf_counter() - t0
            self.fetched_real_bytes += h.untyped_storage().nbytes()
            if h.shape[0] != seg.rows:
                raise RuntimeError(
                    f"hidden block has {h.shape[0]} rows, expected {seg.rows}"
                )
            self._cache[k] = h
            self.fetched_bytes += seg.nbytes
        return h

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
                self.futures[k] = self._pool.submit(get, seg.ref)
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


def gathered_log_softmax(
    local: torch.Tensor, gather: VocabGather, vocab_limit: int
) -> torch.Tensor:
    """Full-vocabulary fp32 log-prob rows of vocab-sharded logits [k, V/TP].

    ``gather`` concatenates the tensor-parallel ranks' shards along the last
    dimension in rank (= vocabulary) order (``no_gather`` at TP=1): the
    engine's logits all-gather. Then the engine's ``raw_logprobs`` path: fp32
    cast and one full-row ``log_softmax``. A collective at TP>1: every rank
    of the group must call it with the same row count.
    """
    z = gather(local.contiguous())
    return torch.log_softmax(z[..., :vocab_limit].float(), -1)


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

    def __call__(self, idx: torch.Tensor) -> torch.Tensor:
        return gathered_log_softmax(
            self.local[idx.to(self.local.device)], self.gather, self.vocab_limit
        )

    def token_logprobs(self, tokens: torch.Tensor) -> torch.Tensor:
        """[k] fp32 log-prob of ``tokens[j]`` at row j, for rows 0..k-1."""
        k = tokens.numel()
        out = []
        for a in range(0, k, self.chunk_rows):
            b = min(k, a + self.chunk_rows)
            lp = gathered_log_softmax(self.local[a:b], self.gather, self.vocab_limit)
            out.append(lp.gather(-1, tokens[a:b].view(-1, 1).to(lp.device))[:, 0])
            del lp
        if not out:
            return torch.zeros(0, dtype=torch.float32, device=self.local.device)
        return torch.cat(out)


def head_stash_bytes(
    vocab: int, hidden: int, tp: int, versions: int = 1, elem_bytes: int = 2
) -> int:
    """GPU bytes of the LM-head stash on one tensor-parallel rank.

    Each rank keeps its vocab-parallel shard per pending version: ceil(V/TP)
    x d elements (Megatron pads V to a multiple of TP, so V/TP exactly).
    """
    return versions * (-(-vocab // tp)) * hidden * elem_bytes


def residual_rows_needed(n_eval: int, tau: int, g: int) -> int:
    """Q rows a lazy verification computed: residual masses plus the emit row."""
    return n_eval + (tau < g)


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
