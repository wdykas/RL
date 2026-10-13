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
training field per row (``router_replay.py``).
"""

from __future__ import annotations

import threading
from typing import Any, Optional, Sequence

import torch

NO_ANCHOR = -1
ROUTE_DTYPE = torch.int16


class RouteLog:
    """Engine side: routes of tagged requests by (key, absolute position).

    ``record(entries, rows)``: ``rows`` [n, L_local, K] is this forward's
    route buffer (GPU, its first n local token rows); each entry
    ``(row, key, pos)`` stores ``rows[row]`` as position ``pos`` of ``key``.
    The copy to pinned host memory is asynchronous (one copy per forward,
    ordered on the current stream before the next forward overwrites the
    buffer). ``pop(key)`` returns (positions [m] int64, routes [m, L_local, K])
    sorted by position, waiting for the copies it needs; the key is then
    closed (later rows are dropped). A position recorded twice (an engine
    eviction recomputes it) keeps the latest row: under batch invariance it
    is bitwise the same route.
    """

    def __init__(self, chunk_rows: int = 65536):
        self.chunk_rows = chunk_rows
        self._lock = threading.Lock()
        self._rows: dict[str, dict[int, tuple[list, int]]] = {}
        self._closed: set[str] = set()
        self._cur: Optional[list] = None  # [buffer, used, live, event]
        self.recorded = 0

    def _alloc(self, k: int, like: torch.Tensor) -> tuple[list, int]:
        cur = self._cur
        if cur is None or cur[1] + k > cur[0].shape[0] or cur[0].shape[1:] != like.shape[1:]:
            n = max(self.chunk_rows, k)
            buf = torch.empty(
                (n, *like.shape[1:]), dtype=ROUTE_DTYPE, pin_memory=like.is_cuda
            )
            cur = self._cur = [buf, 0, 0, None]
        off = cur[1]
        cur[1] += k
        return cur, off

    def record(self, entries: list[tuple[int, str, int]], rows: torch.Tensor) -> None:
        with self._lock:
            entries = [e for e in entries if e[1] not in self._closed]
            if not entries:
                return
            idx = torch.tensor([r for r, _, _ in entries], device=rows.device)
            sel = rows.index_select(0, idx).to(ROUTE_DTYPE)
            chunk, off = self._alloc(len(entries), sel)
            chunk[0][off : off + len(entries)].copy_(sel, non_blocking=sel.is_cuda)
            if sel.is_cuda:
                ev = torch.cuda.Event()
                ev.record()
                chunk[3] = ev
            for j, (_, key, pos) in enumerate(entries):
                slots = self._rows.setdefault(key, {})
                old = slots.get(pos)
                if old is not None:
                    old[0][2] -= 1
                slots[pos] = (chunk, off + j)
                chunk[2] += 1
            self.recorded += len(entries)

    def pop(self, key: str) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        with self._lock:
            self._closed.add(key)
            slots = self._rows.pop(key, {})
            if not slots:
                return None
            for chunk, _ in slots.values():
                chunk[2] -= 1
            for ev in {id(c[0]): c[0][3] for c in slots.values()}.values():
                if ev is not None:
                    ev.synchronize()
            pos = sorted(slots)
            rows = torch.stack([slots[p][0][0][slots[p][1]] for p in pos])
            return torch.tensor(pos, dtype=torch.long), rows.clone()

    def live_rows(self) -> int:
        with self._lock:
            return sum(len(s) for s in self._rows.values())


def step_route_entries(
    token_request_idx: Sequence[int],
    token_pos: Sequence[int],
    request_key: dict[int, str],
    local_start: int,
    local_count: int,
) -> list[tuple[int, str, int]]:
    """(local row, key, absolute position) of the tagged tokens of one forward.

    ``token_request_idx[j]`` / ``token_pos[j]``: context slot and absolute
    position of the step's token j (unpadded tokens only);
    ``request_key[slot]``: draft key of tagged slots. This rank's router saw
    tokens ``local_start .. local_start + local_count - 1`` (sequence
    parallelism shards the padded token axis contiguously across TP ranks;
    without it, all tokens).
    """
    out = []
    hi = min(len(token_request_idx), local_start + local_count)
    for j in range(max(local_start, 0), hi):
        key = request_key.get(int(token_request_idx[j]))
        if key is not None:
            out.append((j - local_start, key, int(token_pos[j])))
    return out


def merge_route_parts(
    parts: Sequence[tuple[Sequence[int], torch.Tensor, torch.Tensor]],
    num_positions: int,
    layer_numbers: Optional[Sequence[int]] = None,
) -> tuple[torch.Tensor, list[int]]:
    """Dense anchors [num_positions, L, K] from per-rank route logs of one draft.

    ``parts``: (layer numbers, positions [m], routes [m, len(layers), K]) per
    engine rank that logged this draft (pipeline stages hold different
    layers; sequence-parallel TP ranks hold different positions). Columns are
    the union of layer numbers in ascending order (``layer_numbers`` fixes
    them). Missing entries are ``NO_ANCHOR``. Returns (anchors, layers).
    """
    if layer_numbers is None:
        layer_numbers = sorted({int(x) for p in parts for x in p[0]})
    col = {ln: j for j, ln in enumerate(layer_numbers)}
    k = next((int(p[2].shape[-1]) for p in parts if p[2].numel()), 1)
    out = torch.full(
        (num_positions, len(layer_numbers), k), NO_ANCHOR, dtype=ROUTE_DTYPE
    )
    for layers, pos, routes in parts:
        if routes.numel() == 0:
            continue
        keep = pos < num_positions
        pos, routes = pos[keep], routes[keep]
        cols = torch.tensor([col[int(x)] for x in layers], dtype=torch.long)
        out[pos.unsqueeze(1), cols.unsqueeze(0)] = routes.to(ROUTE_DTYPE)
    return out, list(layer_numbers)


def position_complete(anchors: torch.Tensor) -> torch.Tensor:
    """[N] bool: positions whose anchor is set at every layer."""
    return anchors.ge(0).all(-1).all(-1)


def anchored_draft_length(anchors: torch.Tensor, prompt_len: int, g: int) -> int:
    """Longest m <= g whose drafted tokens all have their anchors.

    Token i of the draft (0-based) is sampled from the forward of position
    P+i-1, which depends on anchors 0..P+i-1. So m tokens need positions
    0..P+m-2 complete. (The bonus after m tokens would need P+m-1 too; it is
    never sampled from an unanchored row: see ``SpeculativeGeneration``.)
    """
    ok = position_complete(anchors)
    n_ok = int(ok.numel()) if bool(ok.all()) else int((~ok).nonzero()[0, 0])
    # positions 0..n_ok-1 complete: m tokens need P+m-1 <= n_ok.
    return max(0, min(g, n_ok - prompt_len + 1)) if n_ok >= prompt_len else 0


def splice(anchors: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
    """Anchors of a longer row: ``new`` where set, else the earlier ``anchors``.

    Both are plain-theta_n routes of the same prefix wherever both are set
    (bitwise equal under batch invariance), so which one wins is immaterial.
    """
    n = max(anchors.shape[0], new.shape[0])
    out = torch.full((n, *new.shape[1:]), NO_ANCHOR, dtype=ROUTE_DTYPE)
    out[: anchors.shape[0]] = anchors
    seg = new.ge(0).all(-1).all(-1)
    idx = seg.nonzero().flatten()
    out[idx] = new[idx]
    return out


def no_anchor_rows(n: int, layers: int, topk: int) -> torch.Tensor:
    return torch.full((n, layers, topk), NO_ANCHOR, dtype=ROUTE_DTYPE)


def route_bytes(anchors: Optional[Any]) -> int:
    if anchors is None:
        return 0
    return anchors.numel() * anchors.element_size()
