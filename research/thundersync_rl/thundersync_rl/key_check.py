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
"""Always-on noise-key consistency of the keyed engine samplers (wide / exact_int).

Each sampled token's noise is keyed by (request, position). A key must be drawn once, and a request's positions
must advance by exactly one per decode step it samples in. F11 (wide sampler key positions from the request, not
gpu_view): under async scheduling the gpu_view lengths were not compacted with the rows, so about 1e-4 of the draws
used another request's position, some of them a key the request had already used (reused noise).

``KeyContinuity.observe(rids, pos)`` runs once per sampler call on the host arrays the sampler already holds. It
checks that the call's request ids are distinct and that every request also sampled in the previous call is now
at its previous position + 1. Requests absent from the previous call (new, or paused for a step) start fresh, so
only the previous call's (request -> position) map is kept: O(running requests) host work and memory, no device
sync. Raises ``SamplerKeyError`` on a violation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class SamplerKeyError(RuntimeError):
    """A keyed sampler drew a (request, position) noise key out of sequence (reused or skipped)."""


class KeyContinuity:
    def __init__(self, name: str = "sampler") -> None:
        self.name = name
        self.prev: dict[int, int] = {}
        self.calls = 0
        self.rows_checked = 0
        self.violations = 0

    def check(self, rids: np.ndarray, pos: np.ndarray) -> list[str]:
        """Violations of this call (request ids / positions as host int arrays; position < 0: unknown, skipped)."""
        bad: list[str] = []
        cur: dict[int, int] = {}
        for rid, p in zip(rids.tolist(), pos.tolist()):
            if rid in cur:
                bad.append(f"request {rid} sampled twice in one call (positions {cur[rid]}, {p})")
                continue
            cur[rid] = p
            if p < 0:
                continue
            self.rows_checked += 1
            q = self.prev.get(rid)
            if q is not None and q >= 0 and p != q + 1:
                bad.append(f"request {rid} position {q} -> {p} (expected {q + 1})")
        self.prev = cur
        self.calls += 1
        self.violations += len(bad)
        return bad

    def observe(self, rids: np.ndarray, pos: Optional[np.ndarray]) -> None:
        if pos is None:
            return
        bad = self.check(rids, pos)
        if bad:
            raise SamplerKeyError(
                f"{self.name} noise keys out of sequence ({len(bad)} rows of {len(rids)}): {bad[:5]} -- a "
                f"(request, position) key reused or skipped (key positions must be len(prompt) + len(generated))"
            )

    def stats(self) -> dict[str, int]:
        return {"key_check_calls": self.calls, "key_check_rows": self.rows_checked,
                "key_check_violations": self.violations}
