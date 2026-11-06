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
"""q-row recompute broker (``hidden_transport=recompute`` on the learner-verify path).

Learner ranks ask this small actor for the drafter's head-input rows of a draft; it submits the prefill to the
engine's rank 0 (``run_verify_requests``), gathers the captured rows from every engine rank as object refs
(``engine_verify_take_refs``) and hands the ref back. Learner and engine workers never receive each other's actor
handles: deserializing a ``ThunderSyncMegatronPolicyWorker`` handle inside another worker of that class re-pickles
the class by value and breaks its ``super`` calls (the first local run of this build died that way).

One prefill per key: concurrent requests for the same key (tensor-parallel learner ranks of one verifier) wait for the
same prefill; the ref is dropped here after ``consumers`` takes (each caller holds its own reference).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional


class QRecomputeBroker:
    def __init__(self, rank0: Any, workers: list[Any]):
        self.rank0 = rank0
        self.workers = list(workers)
        self.version: Optional[int] = None
        self.refs: dict[str, Any] = {}
        self.pending: dict[str, asyncio.Future] = {}
        self.taken: dict[str, int] = {}
        self.stats = {"requests": 0, "prefill_tokens": 0, "s": 0.0}

    async def set_version(self, version: Optional[int]) -> None:
        """The weight version the engine holds now (the driver sets it after every refit / anchor stage)."""
        self.version = None if version is None else int(version)

    async def q_recompute(
        self, keys: list[str], rows: list[list[int]], prompt_lens: list[int], expect_version: int, consumers: int = 1
    ) -> dict[str, Any]:
        if self.version is None or int(expect_version) != self.version:
            raise RuntimeError(
                f"q recompute (R2): the engine holds version {self.version}, the drafts were drafted on "
                f"{expect_version}"
            )
        new = [j for j, k in enumerate(keys) if k not in self.refs and k not in self.pending]
        if new:
            sub = [keys[j] for j in new]

            async def submit():
                t0 = time.perf_counter()
                vkeys = [f"q|{k}" for k in sub]
                await self.rank0.run_verify_requests.remote(
                    [rows[j] for j in new], vkeys, [int(prompt_lens[j]) for j in new]
                )
                got: dict[str, Any] = {}
                for r in await asyncio.gather(*[w.engine_verify_take_refs.remote(vkeys) for w in self.workers]):
                    got.update(r)
                for k, vk in zip(sub, vkeys):
                    if vk not in got:
                        raise RuntimeError(f"q recompute: no engine rank captured rows for {k}")
                    self.refs[k] = got[vk]
                self.stats["requests"] += len(sub)
                self.stats["prefill_tokens"] += int(sum(len(rows[j]) for j in new))
                self.stats["s"] += time.perf_counter() - t0

            fut = asyncio.ensure_future(submit())
            for k in sub:
                self.pending[k] = fut
            try:
                await fut
            finally:
                for k in sub:
                    self.pending.pop(k, None)
        waits = {id(self.pending[k]): self.pending[k] for k in keys if k in self.pending}
        if waits:
            await asyncio.gather(*waits.values())
        out = {}
        for k in keys:
            out[k] = self.refs[k]
            self.taken[k] = self.taken.get(k, 0) + 1
            if self.taken[k] >= int(consumers):
                self.refs.pop(k, None)
                self.taken.pop(k, None)
        return out

    async def pop_stats(self) -> dict[str, float]:
        out = {**{k: float(v) for k, v in self.stats.items()}, "held_refs": float(len(self.refs))}
        self.stats = {"requests": 0, "prefill_tokens": 0, "s": 0.0}
        return out


def make_broker(rank0: Any, workers: list[Any]) -> Any:
    """The broker actor (async; no GPU, no CPU reservation) for one engine."""
    import ray

    return ray.remote(num_cpus=0)(QRecomputeBroker).remote(rank0, workers)
