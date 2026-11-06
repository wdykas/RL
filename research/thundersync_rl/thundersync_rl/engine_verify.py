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
"""Engine prefill capture for engine-side verification (B': ``thundersync.verify_engine_share``, colocated).

Under batch invariance the engine's prefill of a row is bitwise the training forward of that row, so an engine
holding theta_k can supply p of drafted tokens:

* the drafted row (prompt + draft) is submitted as an engine request that generates one (discarded) token, tagged
  ``VERIFY_KEY_ATTR`` / ``VERIFY_PLEN_ATTR`` (worker ``run_verify_requests``);
* during its prefill the engine's LM-head input rows of every token are copied into a static buffer (captured with
  the CUDA graphs), and the rows of positions plen-1 .. n-1 of tagged requests are kept per key (``CaptureStore``):
  g+1 rows of d head-input elements per draft;
* ``nolanes_engine.nl_ev_take`` reduces them on the engine to canonical-block LSE partials and drafted-token
  logits, and the learner decides the row from those (``ds2.EngineShardedRows``).

This module holds the parts that need neither Megatron nor Ray.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

# Sampling-params attributes of an engine verification request: its key and
# the draft's prompt length (rows of positions plen-1 .. n-1 are kept).
VERIFY_KEY_ATTR = "ts_verify_key"
VERIFY_PLEN_ATTR = "ts_verify_plen"


class CaptureStore:
    """Head-input rows of tagged verification requests, by key (engine side).

    ``begin(key, lo, hi, d, dtype, device)`` allocates [hi - lo, d] for
    positions lo .. hi-1; ``put(key, positions, rows)`` writes them
    (positions as numpy, rows on device; a position written twice -- a
    re-prefill after an eviction -- is overwritten with the same bits under
    batch invariance); ``complete(key)`` says every position arrived;
    ``pop(key)`` hands the rows out. Memory: (hi - lo) x d x head-input bytes
    per request in flight on this rank.
    """

    def __init__(self):
        self._rows: dict[str, torch.Tensor] = {}
        self._lo: dict[str, int] = {}
        self._seen: dict[str, np.ndarray] = {}

    def __contains__(self, key: str) -> bool:
        return key in self._rows

    def begin(self, key: str, lo: int, hi: int, d: int, dtype, device) -> None:
        if key in self._rows:
            return
        n = max(0, int(hi) - int(lo))
        self._rows[key] = torch.empty((n, int(d)), dtype=dtype, device=device)
        self._lo[key] = int(lo)
        self._seen[key] = np.zeros(n, dtype=bool)

    def put(self, key: str, positions: np.ndarray, rows: torch.Tensor) -> None:
        buf = self._rows.get(key)
        if buf is None:
            return
        rel = np.asarray(positions, dtype=np.int64) - self._lo[key]
        ok = (rel >= 0) & (rel < buf.shape[0])
        if not ok.all():
            rel = rel[ok]
            rows = rows[torch.from_numpy(np.flatnonzero(ok)).to(rows.device)]
        if rel.size == 0:
            return
        idx = torch.from_numpy(rel)
        if buf.is_cuda:
            idx = idx.pin_memory().to(buf.device, non_blocking=True)
        buf.index_copy_(0, idx, rows.to(buf.dtype))
        self._seen[key][rel] = True

    def complete(self, key: str) -> bool:
        seen = self._seen.get(key)
        return seen is not None and bool(seen.all())

    def keys(self) -> list[str]:
        return list(self._rows)

    def pop(self, key: str) -> Optional[torch.Tensor]:
        self._lo.pop(key, None)
        seen = self._seen.pop(key, None)
        rows = self._rows.pop(key, None)
        if rows is None or seen is None or not seen.all():
            return None
        return rows

    def drop(self, keys) -> None:
        for k in keys:
            self._rows.pop(k, None)
            self._lo.pop(k, None)
            self._seen.pop(k, None)

    def nbytes(self) -> int:
        return sum(r.numel() * r.element_size() for r in self._rows.values())


def capture_selection(
    token_slots: np.ndarray,
    token_pos: np.ndarray,
    slot_kid: np.ndarray,
    kid_lo: np.ndarray,
    kid_hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(token rows, key ids, positions) of one forward's tokens to capture.

    ``token_slots[j]`` / ``token_pos[j]``: context slot and absolute position
    of the step's token j; ``slot_kid``: key id per slot (-1 untagged);
    ``kid_lo`` / ``kid_hi``: kept positions [lo, hi) per key id. Vectorized.
    """
    kid = slot_kid[token_slots]
    tagged = kid >= 0
    k = np.where(tagged, kid, 0)
    keep = tagged & (token_pos >= kid_lo[k]) & (token_pos < kid_hi[k])
    sel = np.flatnonzero(keep)
    return sel, kid[sel], token_pos[sel]


