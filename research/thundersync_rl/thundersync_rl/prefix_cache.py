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
"""Driver side of the learner prefix-state cache (DESIGN.md section 14p).

The learner ranks hold the state (``nemo_rl/models/megatron/prefix_state_cache.py``); the
driver decides membership and placement, so every rank of a replica (TP x PP) receives the
same lists and picks the same start positions without any communication among ranks:

- placement: a continued row is verified on the data-parallel rank that holds its state
  (sticky owner); new rows go longest-first onto the least-loaded rank, by the tokens each
  row will actually forward (placement never enters a decision, ``lpt_owners``);
- memory: each rank's cached rows cost at most the budget (positions x K/V bytes + kept
  boundaries x Mamba state bytes, the largest per-rank costs the workers report). ``size``
  "auto" sets the budget to what this step's continued rows need at the row cap
  (rows per rank x one row's bytes at ``row_cap_tokens``), at most ``max_gb``; "fixed" uses
  ``max_gb``. ``policy`` "pin" (default) admits a row only if its reservation (its bytes at
  the row cap) fits and never evicts: continuation rounds visit every row cyclically, and
  least-recently-used eviction with a working set above the budget misses on *every* access
  (measured in the MC of agent_carried_walk.md; it silently degrades to O(L^2 / c)), while
  pinning keeps budget / need of the rows hitting. "lru" is the former policy. Rows not
  cached forward whole (slower, same bits). Device residency is the worker's (``gpu_gb``);
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

    def __init__(
        self,
        gpu_gb: float = 8.0,
        max_gb: float = 32.0,
        size: str = "auto",
        policy: str = "pin",
        row_cap_tokens: Optional[int] = None,
        min_hit_frac: float = 0.0,
    ):
        if max_gb <= 0 or gpu_gb < 0:
            raise ValueError("learner_prefix_cache: need max_gb > 0 and gpu_gb >= 0")
        if size not in ("auto", "fixed") or policy not in ("pin", "lru"):
            raise ValueError(f"learner_prefix_cache: size {size!r} / policy {policy!r}")
        self.gpu_gb = float(gpu_gb)
        self.cap_bytes = int(max_gb * 2**30)
        self.max_bytes = self.cap_bytes
        self.size, self.policy = size, policy
        self.row_cap_tokens = row_cap_tokens
        self.min_hit_frac = float(min_hit_frac)
        self.need_bytes = 0
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
        self.seen: set[str] = set()  # keys forwarded under the cache at least once
        self.counts = dict(admitted=0, denied=0, evicted=0, eligible=0)

    def row_bytes(self, prompt_len: int = 1) -> int:
        """One row's bytes at the row cap (its reservation under "pin")."""
        n = int(self.row_cap_tokens or 0)
        return self.entry_bytes(n, prompt_len) + (self.state_bytes if self.policy == "pin" else 0)

    def plan_step(self, rows: int, ranks: int) -> None:
        """Budget for a step's continuation rounds: ``rows`` continued rows over ``ranks``."""
        per_rank = -(-rows // max(ranks, 1))
        self.need_bytes = per_rank * self.row_bytes() if self.row_cap_tokens else 0
        if self.size == "auto" and self.need_bytes:
            self.max_bytes = min(self.cap_bytes, max(self.need_bytes, 1))
        else:
            self.max_bytes = self.cap_bytes

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
        if self.policy == "pin":
            return owners, self._place_pinned(keys, lengths, prompt_lens, owners)
        sent: list[Optional[str]] = []
        for k, n, p, r in zip(keys, lengths, prompt_lens, owners):
            # eligible: forwarded before (an evicted row still counts: its miss is thrash)
            self.counts["eligible"] += int(k in self.seen)
            lru = self.lru.setdefault(r, OrderedDict())
            lru.pop(k, None)
            need = self.entry_bytes(n, p)
            used = sum(self.cost[x] for x in lru)
            while lru and used + need > self.max_bytes:
                old, _ = lru.popitem(last=False)  # least recently verified
                used -= self.cost.pop(old)
                self.owner.pop(old, None)
                self._release.append(old)
                self.counts["evicted"] += 1
            if need > self.max_bytes:
                self._forget(k)
                sent.append(None)
                continue
            lru[k] = True
            self.cost[k] = need
            self.owner[k] = r
            self.seen.add(k)
            sent.append(k)
        return owners, sent

    def _place_pinned(self, keys, lengths, prompt_lens, owners) -> list[Optional[str]]:
        """Admission without eviction: a row reserves its bytes at the row cap once
        (without a row cap: its current bytes, updated as it grows)."""
        sent: list[Optional[str]] = []
        for k, n, p, r in zip(keys, lengths, prompt_lens, owners):
            if k in self.owner:
                self.counts["eligible"] += int(k in self.seen)
                if not self.row_cap_tokens:
                    self.cost[k] = max(self.cost[k], self.entry_bytes(n, p) + self.state_bytes)
                self.seen.add(k)
                sent.append(k)
                continue
            lru = self.lru.setdefault(r, OrderedDict())
            need = (
                self.row_bytes(p)
                if self.row_cap_tokens
                else self.entry_bytes(n, p) + self.state_bytes
            )
            used = sum(self.cost[x] for x in lru)
            if used + need > self.max_bytes:
                self.counts["denied"] += 1
                sent.append(None)
                continue
            self.counts["admitted"] += 1
            lru[k] = True
            self.cost[k] = need
            self.owner[k] = r
            self.seen.add(k)
            sent.append(k)
        return sent

    def take_counts(self) -> dict[str, float]:
        """This step's admission counters and budget (metrics), then zero the counters."""
        out = {k: float(v) for k, v in self.counts.items()}
        out["need_gb_per_rank"] = self.need_bytes / 2**30
        out["budget_gb_per_rank"] = self.max_bytes / 2**30
        out["max_rank_gb"] = max(self.rank_bytes().values(), default=0) / 2**30
        self.counts = dict(admitted=0, denied=0, evicted=0, eligible=0)
        return out

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
