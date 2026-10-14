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
"""Drafter routes for sticky MoE routing (anchors), engine log to learner.

Sticky routing (``nemo_rl/models/megatron/sticky_routing.py``) defines the
policy of step n+1 as F(theta_{n+1}; A) where the anchor A at a position is
the plain theta_n forward's top-k expert set on the actual prefix (positions
up to a content-independent cut c) or "self" (beyond c, rows never drafted).
The engine at theta_n records the plain top-k of every token it processes
(MCore RouterReplay RECORD into a static buffer, captured into the CUDA
graphs) and this module keeps the rows of tagged draft requests, keyed by
absolute token position (``RouteLog``), so that the learner can replay them.

Position convention: the anchor of position t is the route of the token AT
position t (the input token of the forward step that samples token t+1). The
log-prob of token t+1 depends on anchors 0..t only, so a row
[prompt (P tokens), draft (g tokens)] needs positions 0..P+g-2 for the
drafted tokens and position P+g-1 only for a bonus token after the draft.

Dense anchor tensors are [positions, num_moe_layers, topk] int16 with -1 for
"no anchor" (self). That is also the layout of NeMo RL's ``routed_experts``
training field per row (``router_replay.py``), restricted to a pipeline
stage's layers on the learner.

Transport: the engine rank that logged a draft puts its dense routes in the
object store in position blocks (``put_routes``); the driver holds only block
handles and the logged-prefix length (``RouteRefs``); each learner rank
fetches the blocks of its own pipeline stage's layers (``fetch_anchors``).
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch

NO_ANCHOR = -1
ROUTE_DTYPE = torch.int16
# Positions per object-store block of a draft's routes. The learner fetches
# whole blocks, so this bounds what it pulls beyond a truncated draft's kept
# positions; 4096 positions are 7 MB at 40 MoE layers x top-22 (int16).
ROUTE_BLOCK_POSITIONS = 4096


class RouteLog:
    """Engine side: routes of tagged requests by (key, absolute position).

    ``record(rows, sel, kid, pos)`` runs once per engine forward: ``rows``
    [n, L_local, K] is the forward's route buffer (device), ``sel`` the rows of
    tagged tokens, ``kid`` / ``pos`` their key ids (``key_id``) and absolute
    positions (numpy arrays). It gathers the tagged rows on the device (the
    index reaches the device from pinned memory without blocking), enqueues
    one asynchronous copy into a pinned staging ring and records an event: no
    host-device synchronization and no Python per token or per request, so it
    neither stalls the forward it follows nor breaks asynchronous scheduling.
    Staged rows are appended to a host store (one vectorized copy per
    forward) once their copy has completed: at later records when the event
    has fired, and at ``pop``.

    ``pop(keys)`` returns, per key with rows logged since its last pop, (lo,
    run_end, routes [n, L_local, K] int16 of positions lo..lo+n-1, -1 where
    unlogged), where positions lo..run_end-1 are all logged: one vectorized
    pass over the store plus one dense scatter per key. ``close=True`` closes
    the keys (later rows are dropped); ``close=False`` keeps them open, so a
    running draft's routes can be taken in pieces. A position recorded twice
    (an engine eviction recomputes it) keeps one of its rows: under batch
    invariance they are bitwise equal.
    """

    def __init__(self, ring_rows: int = 65536, drain_every: int = 64):
        self.ring_rows = ring_rows
        self.drain_every = drain_every
        self._lock = threading.Lock()
        self._ids: dict[str, int] = {}
        self._closed = np.zeros(0, dtype=bool)  # by key id
        self._ring: Optional[torch.Tensor] = None
        self._head = 0
        self._pending: deque = deque()  # (start, m, kid, pos, event)
        # Host store of drained rows (grows by doubling; compacted at pop).
        self._rows: Optional[np.ndarray] = None
        self._kid = np.zeros(0, dtype=np.int64)
        self._pos = np.zeros(0, dtype=np.int64)
        self._used = 0
        self.recorded = 0

    def key_id(self, key: str) -> int:
        with self._lock:
            return self._key_id(key)

    def _key_id(self, key: str) -> int:
        kid = self._ids.get(key)
        if kid is None:
            kid = self._ids[key] = len(self._ids)
            if kid >= self._closed.shape[0]:
                grown = np.zeros(max(1024, 2 * self._closed.shape[0]), dtype=bool)
                grown[: self._closed.shape[0]] = self._closed
                self._closed = grown
        return kid

    def _stage(self, m: int, like: torch.Tensor) -> int:
        """Start row of m free staging rows (drains what the reuse overwrites)."""
        shape = tuple(like.shape[1:])
        if (
            self._ring is None
            or tuple(self._ring.shape[1:]) != shape
            or m > self._ring.shape[0]
        ):
            self._drain(force=True)
            self._ring = torch.empty(
                (max(self.ring_rows, m), *shape),
                dtype=ROUTE_DTYPE,
                pin_memory=like.is_cuda,
            )
            self._head = 0
        if self._head + m > self._ring.shape[0]:
            self._head = 0
        lo, hi = self._head, self._head + m
        # Rows still staged in [lo, hi) must be drained first. Their forwards
        # ran a whole ring ago, so the wait is normally zero.
        while self._pending and any(
            s < hi and lo < s + n for s, n, *_ in self._pending
        ):
            self._move(self._pending.popleft())
        self._head = hi
        return lo

    def record(
        self, rows: torch.Tensor, sel: np.ndarray, kid: np.ndarray, pos: np.ndarray
    ) -> None:
        m = int(sel.shape[0])
        if m == 0:
            return
        with self._lock:
            start = self._stage(m, rows)
            idx = torch.from_numpy(np.ascontiguousarray(sel, dtype=np.int64))
            if rows.is_cuda:
                idx = idx.pin_memory().to(rows.device, non_blocking=True)
            picked = rows.index_select(0, idx).to(ROUTE_DTYPE)
            self._ring[start : start + m].copy_(picked, non_blocking=rows.is_cuda)
            event = None
            if rows.is_cuda:
                event = torch.cuda.Event()
                event.record()
            self._pending.append(
                (
                    start,
                    m,
                    np.asarray(kid, dtype=np.int64),
                    np.asarray(pos, dtype=np.int64),
                    event,
                )
            )
            self.recorded += m
            if len(self._pending) >= self.drain_every:
                self._drain(force=False)

    def _drain(self, force: bool) -> None:
        while self._pending:
            ev = self._pending[0][4]
            if not force and ev is not None and not ev.query():
                return
            self._move(self._pending.popleft())

    def _move(self, item) -> None:
        """Append one staged forward's rows to the store (closed keys dropped)."""
        start, m, kid, pos, event = item
        if event is not None:
            event.synchronize()
        keep = ~self._closed[kid]
        rows = self._ring[start : start + m].numpy()[keep]
        kid, pos = kid[keep], pos[keep]
        k = int(kid.shape[0])
        if k == 0:
            return
        if self._rows is None or self._used + k > self._rows.shape[0]:
            cap = max(
                self._used + k,
                2 * (0 if self._rows is None else self._rows.shape[0]),
                4096,
            )
            grown = np.empty((cap, *rows.shape[1:]), dtype=np.int16)
            kg, pg = np.empty(cap, dtype=np.int64), np.empty(cap, dtype=np.int64)
            if self._rows is not None:
                grown[: self._used] = self._rows[: self._used]
                kg[: self._used], pg[: self._used] = (
                    self._kid[: self._used],
                    self._pos[: self._used],
                )
            self._rows, self._kid, self._pos = grown, kg, pg
        u = self._used
        self._rows[u : u + k] = rows
        self._kid[u : u + k], self._pos[u : u + k] = kid, pos
        self._used = u + k

    def pop(
        self, keys: Sequence[str], close: bool = True
    ) -> dict[str, tuple[int, int, torch.Tensor]]:
        with self._lock:
            ids = np.array([self._key_id(k) for k in keys], dtype=np.int64)
            self._drain(force=True)
            if close:
                self._closed[ids] = True
            if self._used == 0 or ids.size == 0:
                return {}
            u = self._used
            want = np.zeros(self._closed.shape[0] + 1, dtype=bool)
            want[ids] = True
            sel = np.flatnonzero(want[self._kid[:u]])  # taken rows have kid -1
            out: dict[str, tuple[int, int, torch.Tensor]] = {}
            if sel.size:
                order = sel[np.argsort(self._kid[sel], kind="stable")]
                kid_sorted = self._kid[order]
                name = {int(i): k for k, i in zip(keys, ids)}
                for grp in np.split(order, np.flatnonzero(np.diff(kid_sorted)) + 1):
                    pos = self._pos[grp]
                    lo, n = int(pos.min()), int(pos.max()) + 1
                    dense = np.full(
                        (n - lo, *self._rows.shape[1:]), NO_ANCHOR, dtype=np.int16
                    )
                    logged = np.zeros(n - lo, dtype=bool)
                    # Duplicate positions carry bitwise-equal routes: any wins.
                    dense[pos - lo] = self._rows[grp]
                    logged[pos - lo] = True
                    missing = np.flatnonzero(~logged)
                    run_end = lo + (int(missing[0]) if missing.size else n - lo)
                    out[name[int(self._kid[grp[0]])]] = (
                        lo,
                        run_end,
                        torch.from_numpy(dense),
                    )
                self._kid[sel] = -1  # taken
            # Compact once taken rows and closed keys hold half of the store.
            kid = self._kid[:u]
            live = (kid >= 0) & ~self._closed[np.maximum(kid, 0)]
            if 2 * int(live.sum()) <= u:
                keep = np.flatnonzero(live)
                k = int(keep.size)
                self._rows[:k] = self._rows[keep]
                self._kid[:k], self._pos[:k] = self._kid[keep], self._pos[keep]
                self._used = k
            return out

    def stored_rows(self) -> int:
        with self._lock:
            return self._used


class SlotTags:
    """Key id of each engine context slot's request (-1: untagged).

    ``refresh(request_ids, fresh)`` is called once per forward with the
    context's per-slot request ids and a mask of slots whose request may be
    new (in prefill); it looks up only slots whose request id changed or
    that are fresh, so its Python work is per new request, not per token.
    ``lookup(rid)`` returns the draft key of a request, or None.
    """

    def __init__(
        self, lookup: Callable[[int], Optional[str]], key_id: Callable[[str], int]
    ):
        self.lookup = lookup
        self.key_id = key_id
        self.rid = np.full(0, -1, dtype=np.int64)
        self.kid = np.full(0, -1, dtype=np.int64)

    def refresh(self, request_ids: np.ndarray, fresh: np.ndarray) -> np.ndarray:
        n = int(request_ids.shape[0])
        if n > self.rid.shape[0]:
            grow = n - self.rid.shape[0]
            self.rid = np.concatenate([self.rid, np.full(grow, -1, dtype=np.int64)])
            self.kid = np.concatenate([self.kid, np.full(grow, -1, dtype=np.int64)])
        rid = request_ids.astype(np.int64)
        for slot in np.flatnonzero((rid != self.rid[:n]) | fresh):
            key = self.lookup(int(rid[slot])) if rid[slot] >= 0 else None
            self.kid[slot] = -1 if key is None else self.key_id(key)
        self.rid[:n] = rid
        return self.kid[:n]


def step_route_selection(
    token_slots: np.ndarray, token_pos: np.ndarray, slot_kid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rows, key ids, positions) of the tagged tokens of one forward.

    ``token_slots[j]`` / ``token_pos[j]``: context slot and absolute position
    of the step's token j (unpadded tokens); ``slot_kid[slot]``: key id of
    tagged slots, -1 otherwise. Vectorized (no Python per token).
    """
    kid = slot_kid[token_slots]
    sel = np.flatnonzero(kid >= 0)
    return sel, kid[sel], token_pos[sel]


# ---- Transport: route blocks in the object store, handles on the driver ----


@dataclass
class RouteBlock:
    """Object-store handle of one engine rank's routes of a draft.

    ``ref``: ObjectRef of an int16 tensor [rows, len(layers), K] holding the
    routes of positions lo, lo+1, ... (-1 where unlogged); only positions
    below ``hi`` belong to the draft (a truncated draft keeps the block with a
    smaller ``hi``). ``layers``: the global MoE layer numbers of its columns
    (one pipeline stage's layers).
    """

    ref: Any
    layers: tuple[int, ...]
    lo: int
    hi: int
    nbytes: int


@dataclass
class RouteRefs:
    """A draft's anchors as object-store block handles (sticky routing).

    Draft tuple index 8 and ``Plan.anchors``. Positions 0..length-1 are the
    row's anchored positions (later ones: self); positions 0..complete-1 are
    logged at every MoE layer. The driver only holds these handles and two
    integers; each learner rank fetches the blocks of its own layers
    (``fetch_anchors``). Blocks later in the list take precedence where they
    are logged (a continuation round's log over an earlier one: equal where
    both are set, see ``splice_refs``).
    """

    blocks: list[RouteBlock] = field(default_factory=list)
    length: int = 0
    complete: int = 0
    # Logged prefix per MoE layer number (``merge_route_refs``).
    prefix: dict[int, int] = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        return sum(b.nbytes for b in self.blocks)

    def truncate(self, n: int) -> "RouteRefs":
        """The anchors of the first n positions (block handles shared)."""
        n = min(n, self.length)
        return RouteRefs(
            [replace(b, hi=min(b.hi, n)) for b in self.blocks if b.lo < n],
            n,
            min(self.complete, n),
            {ln: min(p, n) for ln, p in self.prefix.items()},
        )


def put_routes(
    rows: torch.Tensor,
    put: Callable[[Any], Any],
    block_positions: int = ROUTE_BLOCK_POSITIONS,
    start: int = 0,
) -> list[tuple[Any, int, int, int]]:
    """Engine side: one draft's dense routes [n, L, K] into the object store.

    ``rows[j]`` is position ``start + j``. One object per ``block_positions``
    positions. Returns [(ref, lo, hi, bytes)] per block (absolute positions),
    the only thing sent to the driver.
    """
    out = []
    for a in range(0, int(rows.shape[0]), block_positions):
        b = rows[a : a + block_positions].contiguous()
        lo = start + a
        out.append((put(b), lo, lo + int(b.shape[0]), b.numel() * b.element_size()))
    return out


def merge_route_refs(
    parts: Sequence[tuple[Sequence[int], int, int, Sequence[tuple]]],
    length: int,
    all_layers: Sequence[int],
    base: Optional["RouteRefs"] = None,
) -> RouteRefs:
    """A draft's ``RouteRefs`` from the parts every engine rank returned.

    ``parts``: (layer numbers, lo, run_end, [(ref, lo, hi, bytes)]) per
    engine rank that logged the draft (pipeline stages hold different
    layers), where positions lo..run_end-1 are logged. ``base``: routes of
    the same draft taken earlier (a running draft taken in pieces); the new
    blocks extend it. Positions are complete where every layer of
    ``all_layers`` is logged: the minimum over stages of the logged prefix.
    """
    prefix = dict(base.prefix) if base is not None else {}
    blocks = list(base.blocks) if base is not None else []
    for layers, lo, run_end, blks in parts:
        layers = tuple(int(x) for x in layers)
        for ln in layers:
            if lo <= prefix.get(ln, 0):
                prefix[ln] = max(prefix.get(ln, 0), int(run_end))
        blocks += [
            RouteBlock(ref, layers, int(a), min(int(b), length), int(nb))
            for ref, a, b, nb in blks
            if a < length
        ]
    complete = min((prefix.get(int(ln), 0) for ln in all_layers), default=0)
    return RouteRefs(blocks, length, min(complete, length), prefix)


def splice_refs(prior: Optional[RouteRefs], new: RouteRefs) -> RouteRefs:
    """Anchors of a longer row: ``new`` where logged, else ``prior``.

    Both are plain-theta_n routes of the same prefix wherever both are set
    (bitwise equal under batch invariance), so which one wins is immaterial.
    ``prior`` must already be cut to the shared prefix. When ``new`` covers
    all of ``prior``, ``prior``'s blocks are dropped (nothing to fetch twice).
    """
    if prior is None or new.complete >= prior.length:
        return new
    prefix = {ln: max(p, prior.prefix.get(ln, 0)) for ln, p in new.prefix.items()}
    return RouteRefs(
        prior.blocks + new.blocks,
        new.length,
        max(new.complete, prior.complete),
        prefix,
    )


def fetch_anchors(
    refs: Optional[RouteRefs],
    layers: Sequence[int],
    topk: Optional[int],
    width: int,
    get: Callable[[Any], Any],
) -> torch.Tensor:
    """Learner side: dense anchors [width, len(layers), K] of the given layers.

    Fetches only blocks that hold some of ``layers`` (this rank's pipeline
    stage) and some position below ``min(width, refs.length)``. Unlogged
    entries and positions past the anchored length are -1 (self). ``topk``
    None: taken from the first fetched block (1 if none is fetched).
    """
    col = {int(ln): j for j, ln in enumerate(layers)}
    n_max = 0 if refs is None else min(width, refs.length)
    out = None if topk is None else _no_anchors(width, len(layers), topk)
    for b in refs.blocks if refs is not None else ():
        hi = min(b.hi, n_max)
        src = [j for j, ln in enumerate(b.layers) if ln in col]
        if hi <= b.lo or not src:
            continue
        rows = get(b.ref)[: hi - b.lo][:, src].to(ROUTE_DTYPE)
        if out is None:
            out = _no_anchors(width, len(layers), int(rows.shape[-1]))
        dst = torch.tensor([col[b.layers[j]] for j in src], dtype=torch.long)
        idx = rows.ge(0).all(-1).all(-1).nonzero().flatten()
        out[(b.lo + idx).unsqueeze(1), dst.unsqueeze(0)] = rows[idx]
    return _no_anchors(width, len(layers), 1) if out is None else out


def _no_anchors(n: int, layers: int, topk: int) -> torch.Tensor:
    return torch.full((n, layers, topk), NO_ANCHOR, dtype=ROUTE_DTYPE)


def anchored_draft_length(complete: int, prompt_len: int, g: int) -> int:
    """Longest m <= g whose drafted tokens all have their anchors.

    Token i of the draft (0-based) is sampled from the forward of position
    P+i-1, which depends on anchors 0..P+i-1. So m tokens need positions
    0..P+m-2 complete, i.e. P+m-1 <= complete. (The bonus after m tokens
    would need P+m-1 too; it is never sampled from an unanchored row: see
    ``SpeculativeGeneration``.)
    """
    return max(0, min(g, complete - prompt_len + 1)) if complete >= prompt_len else 0


def position_complete(anchors: torch.Tensor) -> torch.Tensor:
    """[N] bool: positions whose anchor is set at every layer."""
    return anchors.ge(0).all(-1).all(-1)


def compare_anchors(anchors: torch.Tensor, ref: torch.Tensor) -> tuple[int, int]:
    """(positions checked, positions equal) of ``anchors`` against ``ref``.

    Positions where ``anchors`` is set at every layer are checked; each must
    be set in ``ref`` with the same expert set at every layer (order within
    the top-k ignored). Used to check logged anchors against an independent
    engine forward of the same row.
    """
    pos, same = _compare(anchors, ref)
    return int(pos.numel()), int(same.all(-1).sum())


def anchor_mismatches(anchors: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """[m, 2] (position, layer column) where ``compare_anchors`` finds a difference.

    Ordered by position, then layer: row 0 is the first mismatch.
    """
    pos, same = _compare(anchors, ref)
    bad = (~same).nonzero()
    return torch.stack([pos[bad[:, 0]], bad[:, 1]], 1) if bad.numel() else bad


def _compare(anchors: torch.Tensor, ref: torch.Tensor):
    """(checked positions [n], [n, L] bool: same expert set as ``ref``)."""
    pos = position_complete(anchors).nonzero().flatten()
    inside = pos < ref.shape[0]
    a = anchors[pos].sort(-1).values
    r = torch.full_like(a, NO_ANCHOR)
    r[inside] = ref[pos[inside]].to(a.dtype).sort(-1).values
    return pos, (a == r).all(-1)
