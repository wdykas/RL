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
"""Exact cross-iteration speculative rollouts.

While iteration k runs, the inference engine (holding theta_k) drafts the
rollouts of iteration k+1's prompts with keyed sampling (keyed_sampling.py).
When iteration k+1 starts, the learner (holding theta_{k+1}) re-derives every
draft position under theta_{k+1} with the same keys and keeps the agreeing
prefix plus the first token it emits differently; the inference engine then
decodes only the remainder with the same keys. Each token is therefore exactly
the token theta_{k+1} would emit for that key, so the samples are exactly
on-policy no matter what produced the drafts.

``SpeculativeGeneration`` wraps the generation interface so the rollout code is
unchanged: a request for a prompt that has a verified plan returns immediately
(whole draft accepted) or submits only its continuation.
"""

from __future__ import annotations

import asyncio
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@dataclass
class Plan:
    """How one rollout of the current iteration will be produced."""

    seed: int
    prefix: list[int] = field(default_factory=list)  # tokens already fixed
    prefix_logprobs: list[float] = field(default_factory=list)
    complete: bool = False


def _key(ids: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(t) for t in ids.tolist())


class SpeculativeGeneration:
    def __init__(
        self,
        base,
        learner_policy,
        *,
        vocab_limit: int,
        head_k: int,
        draft_budget: int,
        max_new_tokens: int,
        pad_token_id: int,
        draft_variants: int = 1,
        variant_eps: float = 0.0,
    ):
        self.base = base
        self.learner_policy = learner_policy
        self.vocab_limit = vocab_limit
        self.head_k = head_k
        self.draft_budget = draft_budget
        self.max_new_tokens = max_new_tokens
        self.pad_token_id = pad_token_id
        self.draft_variants = draft_variants
        self.variant_eps = variant_eps
        workers = base._policy.worker_group
        eods = ray.get(
            workers.run_all_workers_single_data(
                "install_keyed_sampler", vocab_limit=vocab_limit, head_k=head_k
            )
        )
        self.eod = next(e for e in eods if e is not None)
        self._seeds = itertools.count(1)
        self.plans: dict[tuple[int, ...], list[Plan]] = defaultdict(list)
        self.stats: dict[str, float] = {}

    # -- drafting (iteration k, for iteration k+1's prompts) --------------------

    async def draft(self, prompts: list[torch.Tensor]) -> list[list[tuple[torch.Tensor, int, list[int]]]]:
        """Decode up to ``draft_budget`` tokens per prompt with fresh keys.

        Each prompt gets ``draft_variants`` drafts sharing its seed: variant 0 is an
        exact keyed sample, the others add keyed jitter so they split at near-tie
        races. Any draft is valid input to verification, so the longest verified
        prefix among them can be kept without biasing the samples.
        """
        m = self.draft_variants
        seeds = [next(self._seeds) for _ in prompts]
        rows = [(i, v) for i in range(len(prompts)) for v in range(m)]
        width = max(p.numel() for p in prompts)
        ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        for r, (i, _) in enumerate(rows):
            ids[r, : prompts[i].numel()] = prompts[i]
        data = BatchedDataDict(
            {
                "input_ids": ids,
                "input_lengths": torch.tensor([prompts[i].numel() for i, _ in rows]),
                "noise_seed": torch.tensor([seeds[i] for i, _ in rows]),
                "max_new_tokens": torch.full((len(rows),), self.draft_budget),
                "draft_variant": torch.tensor([v for _, v in rows]),
                "variant_eps": torch.tensor(
                    [self.variant_eps if v else 0.0 for _, v in rows], dtype=torch.float64
                ),
            }
        )
        out: list[list[Any]] = [[None] * m for _ in prompts]
        async for r, res in self.base.generate_async(data):
            i, v = rows[r]
            plen = prompts[i].numel()
            glen = int(res["generation_lengths"][0])
            out[i][v] = (prompts[i], seeds[i], res["output_ids"][0, plen : plen + glen].tolist())
        return out

    async def self_test(self, prompts: list[torch.Tensor]) -> dict[str, Any]:
        """Same weights on both sides: keys must make drafts reproducible and verified."""
        a = [v[0] for v in await self.draft(prompts)]
        # Re-draft with the SAME seeds: identical tokens prove keyed sampling is active.
        start = a[0][1]
        self._seeds = itertools.count(start)
        b = [v[0] for v in await self.draft(prompts)]
        same = [x[2] == y[2] for x, y in zip(a, b)]
        out: dict[str, Any] = {"reproducible": same}
        for shift in (-1, 0, 1):
            rows = [torch.cat([p, torch.tensor(d, dtype=torch.long)]) for p, _, d in a]
            res = ray.get(
                self.learner_policy.worker_group.run_all_workers_single_data(
                    "verify_drafts",
                    rows=rows,
                    prompt_lens=[p.numel() for p, _, _ in a],
                    seeds=[s for _, s, _ in a],
                    vocab_limit=self.vocab_limit,
                    head_k=self.head_k,
                    position_shift=shift,
                )
            )
            by_row = dict(x for rr in res for x in rr)
            out[f"accepted_shift{shift}"] = [
                (by_row[i]["accepted"], len(a[i][2])) for i in range(len(a))
            ]
        return out

    # -- verification (start of iteration k+1, learner holds theta_{k+1}) -------

    def verify(self, drafts: list[list[tuple[torch.Tensor, int, list[int]]]]) -> None:
        flat = [(t, d) for t, variants in enumerate(drafts) for d in variants]
        rows = [torch.cat([p, torch.tensor(d, dtype=torch.long)]) for _, (p, _, d) in flat]
        res = ray.get(
            self.learner_policy.worker_group.run_all_workers_single_data(
                "verify_drafts",
                rows=rows,
                prompt_lens=[p.numel() for _, (p, _, _) in flat],
                seeds=[s for _, (_, s, _) in flat],
                vocab_limit=self.vocab_limit,
                head_k=self.head_k,
            )
        )
        by_row = dict(x for rank_res in res for x in rank_res)
        # Every verified prefix is a prefix of the same keyed target sequence, so
        # the longest one per trajectory is kept.
        best: dict[int, tuple[tuple[torch.Tensor, int, list[int]], dict[str, Any]]] = {}
        variant_wins = 0
        for r, (t, d) in enumerate(flat):
            if t not in best or by_row[r]["accepted"] > best[t][1]["accepted"]:
                if t in best:
                    variant_wins += 1
                best[t] = (d, by_row[r])
        self.plans.clear()
        kept = total = full = 0
        for i in range(len(drafts)):
            (prompt, seed, draft), r = best[i]
            n_acc = r["accepted"]
            stopped = len(draft) < self.draft_budget  # the draft itself terminated
            if n_acc == len(draft) and stopped:
                plan = Plan(seed, draft, r["logprobs"][:n_acc], complete=True)
                full += 1
            else:
                prefix = draft[:n_acc] + [r["next"]]
                done = r["next"] == self.eod or len(prefix) >= self.max_new_tokens
                plan = Plan(seed, prefix, r["logprobs"], complete=done)
            kept += n_acc
            total += len(draft)
            self.plans[_key(prompt)].append(plan)
        self.stats = {
            "spec/draft_tokens": float(total),
            "spec/accepted_frac": kept / max(total, 1),
            "spec/full_accept": full / max(len(drafts), 1),
            "spec/variant_wins": float(variant_wins),
        }

    # -- GenerationInterface used by the rollout ----------------------------------

    async def generate_async(
        self, data: BatchedDataDict, greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict], None]:
        assert not greedy
        tasks = [
            asyncio.create_task(self._one(i, data.get_batch(i, 1))) for i in range(data.size)
        ]
        for fut in asyncio.as_completed(tasks):
            yield await fut

    async def _one(self, index: int, datum: BatchedDataDict):
        plen = int(datum["input_lengths"][0])
        prompt = datum["input_ids"][0, :plen]
        queue = self.plans.get(_key(prompt))
        plan = queue.pop(0) if queue else Plan(seed=next(self._seeds))
        gen = list(plan.prefix)
        lps = list(plan.prefix_logprobs)
        if not plan.complete:
            ids = torch.cat([prompt, torch.tensor(plan.prefix, dtype=torch.long)])
            cont = BatchedDataDict(
                {
                    "input_ids": ids.view(1, -1),
                    "input_lengths": torch.tensor([ids.numel()]),
                    "noise_seed": torch.tensor([plan.seed]),
                    "max_new_tokens": torch.tensor([self.max_new_tokens - len(gen)]),
                }
            )
            if "stop_strings" in datum:
                cont["stop_strings"] = datum["stop_strings"]
            async for _, res in self.base.generate_async(cont):
                glen = int(res["generation_lengths"][0])
                c0 = ids.numel()
                gen += res["output_ids"][0, c0 : c0 + glen].tolist()
                lps += res["logprobs"][0, c0 : c0 + glen].tolist()
        n = plen + len(gen)
        out = BatchedDataDict(
            {
                "output_ids": torch.cat([prompt, torch.tensor(gen, dtype=torch.long)]).view(1, -1),
                "logprobs": torch.cat([torch.zeros(plen), torch.tensor(lps, dtype=torch.float)]).view(1, -1),
                "generation_lengths": torch.tensor([len(gen)]),
                "unpadded_sequence_lengths": torch.tensor([n]),
                "gen_leader_worker_idx": [0],
            }
        )
        return index, out

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)
