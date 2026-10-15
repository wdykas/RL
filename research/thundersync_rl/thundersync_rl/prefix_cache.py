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
"""Driver side of the learner prefix-state cache (DESIGN.md section 14o).

The learner ranks hold the state (``nemo_rl/models/megatron/prefix_state_cache.py``); the
driver decides membership and placement, so every rank of a replica (TP x PP) receives the
same lists and picks the same start positions without any communication among ranks:

- placement: a continued row is verified on the data-parallel rank that holds its state
  (sticky owner); new rows go longest-first onto the least-loaded rank, by the tokens each
  row will actually forward (placement never enters a decision, ``lpt_owners``);
- memory: each rank's cached rows cost at most ``max_gb`` (positions x K/V bytes + kept
  boundaries x Mamba state bytes, the largest per-rank costs the workers report); the least
  recently verified rows of an over-budget rank are released (they forward whole next time:
  slower, same bits). Device residency within that is the worker's (``gpu_gb``);
- lifetime: a row's state is released when its rollout is complete; everything is dropped
  when the learner's weights change (the worker clears itself on a new weights version, and
  the directory resets at each step).
"""

from __future__ import annotations

import itertools
from collections import OrderedDict
from typing import Optional


def lpt_owners(
    lengths: list[int], ranks: int, fixed: Optional[dict[int, int]] = None
) -> list[int]:
    """Rank of each row: ``fixed`` rows stay; the rest longest first onto the least-loaded rank.

    Placement only: which rank forwards a row does not enter any decision.
    """
    load = [0] * ranks
    owner = [0] * len(lengths)
    fixed = fixed or {}
    for i, r in fixed.items():
        owner[i] = r
        load[r] += lengths[i]
    for i in sorted(
        (i for i in range(len(lengths)) if i not in fixed), key=lambda i: (-lengths[i], i)
    ):
        r = min(range(ranks), key=lambda r: (load[r], r))
        owner[i] = r
        load[r] += lengths[i]
    return owner


class PrefixCacheDirectory:
    """Membership and placement of rows in the learner ranks' prefix-state caches."""

    def __init__(self, gpu_gb: float = 8.0, max_gb: float = 32.0):
        if max_gb <= 0 or gpu_gb < 0:
            raise ValueError("learner_prefix_cache: need max_gb > 0 and gpu_gb >= 0")
        self.gpu_gb = float(gpu_gb)
        self.max_bytes = int(max_gb * 2**30)
        self.kv_bytes = self.state_bytes = 0
        self.chunk = 128
        self.max_keep = 4
        self.configured = False
        self._ids = itertools.count()
        self.reset()

    def configure(self, costs: list[dict[str, int]]) -> None:
        """Per-rank costs reported by the learner workers; the largest bounds every rank."""
        self.kv_bytes = max(int(c["kv_bytes_per_token"]) for c in costs)
        self.state_bytes = max(int(c["state_bytes_per_boundary"]) for c in costs)
        chunks = {int(c["chunk"]) for c in costs}
        if len(chunks) != 1:
            raise ValueError(f"learner ranks disagree on the chunk size: {chunks}")
        self.chunk = chunks.pop()
        self.max_keep = max(int(c.get("max_keep", 4)) for c in costs)
        self.configured = True

    def reset(self) -> None:
        """Forget every row (the learner dropped them: new weights)."""
        self.owner: dict[str, int] = {}
        self.cost: dict[str, int] = {}
        self.lru: dict[int, OrderedDict] = {}
        self._release: list[str] = []

    def new_key(self) -> str:
        return f"p{next(self._ids)}"

    def entry_bytes(self, length: int, prompt_len: int) -> int:
        """Upper bound of a row's cached bytes after verifying [prompt_len, length)."""
        start = max(0, (prompt_len - 1) // self.chunk * self.chunk)
        boundaries = min(self.max_keep, (length - 1) // self.chunk - start // self.chunk + 1)
        return length * self.kv_bytes + boundaries * self.state_bytes

    def forwarded(self, key: Optional[str], length: int, prompt_len: int) -> int:
        """Tokens a row is expected to forward (placement load)."""
        if key in self.owner:
            return length - (prompt_len - 1) // self.chunk * self.chunk
        return length

    def place(
        self, keys: list[str], lengths: list[int], prompt_lens: list[int], ranks: int
    ) -> tuple[list[int], list[Optional[str]]]:
        """(owner rank, key to send or None) of each row of one verification call."""
        fixed = {i: self.owner[k] for i, k in enumerate(keys) if k in self.owner}
        load = [self.forwarded(k, n, p) for k, n, p in zip(keys, lengths, prompt_lens)]
        owners = lpt_owners(load, ranks, fixed)
        sent: list[Optional[str]] = []
        for k, n, p, r in zip(keys, lengths, prompt_lens, owners):
            lru = self.lru.setdefault(r, OrderedDict())
            lru.pop(k, None)
            need = self.entry_bytes(n, p)
            used = sum(self.cost[x] for x in lru)
            while lru and used + need > self.max_bytes:
                old, _ = lru.popitem(last=False)  # least recently verified
                used -= self.cost.pop(old)
                self.owner.pop(old, None)
                self._release.append(old)
            if need > self.max_bytes:
                self._forget(k)
                sent.append(None)
                continue
            lru[k] = True
            self.cost[k] = need
            self.owner[k] = r
            sent.append(k)
        return owners, sent

    def release(self, keys) -> None:
        """Rows whose rollouts are complete (released by the next call)."""
        for k in keys:
            if k in self.owner or k in self.cost:
                self._forget(k)
                self._release.append(k)

    def _forget(self, key: str) -> None:
        r = self.owner.pop(key, None)
        self.cost.pop(key, None)
        if r is not None:
            self.lru.get(r, OrderedDict()).pop(key, None)

    def take_release(self) -> list[str]:
        out, self._release = self._release, []
        return out

    def rank_bytes(self) -> dict[int, int]:
        return {r: sum(self.cost[k] for k in lru) for r, lru in self.lru.items()}
