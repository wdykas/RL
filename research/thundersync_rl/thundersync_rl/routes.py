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

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

import numpy as np
import torch

from thundersync_rl.engine_q import own_block

NO_ANCHOR = -1
ROUTE_DTYPE = torch.int16
# Sticky nucleus (exact top-p, nemo_rl/models/megatron/exact_nucleus.py): the
# drafter's nucleus ids per sampled row ride the route transport as a pseudo
# "layer" with this number. Its blocks are kept in ``RouteRefs.nucleus`` (never
# dropped by ``splice_refs``: a continuation's request re-logs routes from
# position 0 but nucleus ids only from its own last prompt position) and are
# fetched with ``fetch_nucleus``. Row at absolute position P = the nucleus of
# the logits row of input position P (it samples token P + 1). Encoding: id =
# hi * 2^15 + lo, two non-negative int16 per id; padding (and "self", a
# nucleus over the cap) = NUCLEUS_PAD in both halves, so a logged row is
# all >= 0 and an unlogged one -1 (self).
NUCLEUS_LAYER = -7
NUCLEUS_PAD = 0x7FFF
# Positions per object-store block of a draft's routes. The learner fetches
# whole blocks, so this bounds what it pulls beyond a truncated draft's kept
# positions; 4096 positions are 7 MB at 40 MoE layers x top-22 (int16).
ROUTE_BLOCK_POSITIONS = 4096




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
    # Sticky nucleus: the drafter's nucleus-id blocks (``NUCLEUS_LAYER``), in
    # logging order (later blocks win where logged; equal where both are).
    nucleus: list[RouteBlock] = field(default_factory=list)

    @property
    def nbytes(self) -> int:
        return sum(b.nbytes for b in self.blocks) + sum(b.nbytes for b in self.nucleus)

    def truncate(self, n: int) -> "RouteRefs":
        """The anchors of the first n positions (block handles shared)."""
        n = min(n, self.length)
        return RouteRefs(
            [replace(b, hi=min(b.hi, n)) for b in self.blocks if b.lo < n],
            n,
            min(self.complete, n),
            {ln: min(p, n) for ln, p in self.prefix.items()},
            [replace(b, hi=min(b.hi, n)) for b in self.nucleus if b.lo < n],
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
        # A row slice is a view: pickling it would ship the whole draft (own_block).
        b = own_block(rows[a : a + block_positions])
        lo = start + a
        out.append((put(b), lo, lo + int(b.shape[0]), b.numel() * b.element_size()))
    return out


# Sparse route block (a sequence-parallel TP rank's share of a draft): (tag, positions [m] int64, routes [m, L, K]).
_SPARSE_ROUTES = "ts-route-sparse-v1"
















def encode_nucleus(ids: torch.Tensor) -> torch.Tensor:
    """[n, C] int32 nucleus ids (-1 = padding) -> [n, 1, 2C] int16 route rows."""
    pad = ids < 0
    hi = torch.where(pad, NUCLEUS_PAD, ids >> 15)
    lo = torch.where(pad, NUCLEUS_PAD, ids & 0x7FFF)
    return torch.stack([hi, lo], -1).reshape(ids.shape[0], 1, -1).to(ROUTE_DTYPE)


def decode_nucleus(rows: torch.Tensor) -> torch.Tensor:
    """[n, 1, 2C] int16 -> [n, C] int32 ids, -1 for padding and unlogged rows."""
    r = rows.reshape(rows.shape[0], -1, 2).to(torch.int32)
    hi, lo = r[..., 0], r[..., 1]
    bad = (hi < 0) | (lo < 0) | (hi == NUCLEUS_PAD)  # real ids have hi < 2^15 - 1
    return torch.where(bad, torch.full_like(hi, -1), hi * 32768 + lo)


def fetch_nucleus(
    refs: Optional[RouteRefs],
    width: int,
    cap: int,
    get: Callable[[Any], Any],
    lo: int = 0,
) -> torch.Tensor:
    """Learner side: drafter nucleus ids [width, cap] int32 of positions [lo, lo + width).

    -1 everywhere a position has no logged row (self: the plain nucleus). Later
    blocks override earlier ones where logged (bitwise equal where both are).
    """
    out = torch.full((width, 1, 2 * cap), -1, dtype=ROUTE_DTYPE)
    n_max = 0 if refs is None else min(lo + width, refs.length)
    for b in refs.nucleus if refs is not None else ():
        a, hi = max(b.lo, lo), min(b.hi, n_max)
        if hi <= a:
            continue
        rows = get(b.ref)[a - b.lo : hi - b.lo].to(ROUTE_DTYPE)
        k = min(int(rows.shape[-1]), 2 * cap)
        idx = rows.ge(0).all(-1).all(-1).nonzero().flatten()
        out[a - lo + idx, :, :k] = rows[idx][..., :k]
        if k < 2 * cap:
            out[a - lo + idx, :, k:] = NUCLEUS_PAD
    return decode_nucleus(out)














# ---- Engine anchor apply (anchor horizon), engine side ----


