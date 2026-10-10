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
import os
import time
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


@dataclass
class CohortDrafts:
    """Drafts advanced during one iteration (for scoring at its theta)."""

    live: list[dict[str, Any]]
    prev_lens: list[int]


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
        verify_mode: str = "keyed",
        verify_precision: str = "model",
        q_storage: str = "full",
        verify_batch_tokens: int = 16384,
        longest_first: bool = False,
        first_chunk_groups: int = 0,
        group_aligned: bool = False,
        verify_on: str = "learner",
        pool_weights: tuple[int, ...] | None = None,
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
        self.stream_interval = 16
        self.verify_mode = verify_mode
        self.verify_precision = verify_precision
        # "full": store each draft's full-vocab q at theta_k (O(tokens x vocab)).
        # "stash": stash theta_k's weights and recompute q per batch at
        # verification (O(params + batch x vocab)); scales to large models.
        self.q_storage = q_storage
        self.verify_batch_tokens = verify_batch_tokens
        self.longest_first = longest_first
        self.first_chunk_groups = first_chunk_groups
        self.group_aligned = group_aligned
        # Where stash + block verification run: "learner" or "inference" (the
        # generation workers hold theta_k until the refit and theta_{k+1} after
        # it, and sit idle while rollouts wait for verification).
        groups = {
            "learner": [learner_policy.worker_group],
            "inference": [base._policy.worker_group],
            # Split rows across both pools: all GPUs are idle at verification.
            "both": [learner_policy.worker_group, base._policy.worker_group],
        }[verify_on]
        self.verifiers = groups
        self.verifier = groups[0]
        # Chunk assignment pattern, e.g. weights (1, 2) -> [learner, inf, inf].
        w = list(pool_weights) if pool_weights else [1] * len(groups)
        self._chunk_pools = [g for g, k in zip(groups, w) for _ in range(k)]
        self.current_step = 0
        if verify_mode == "block":
            # Randomized block verification: drafts use the engine's own sampler,
            # which draws from the full (padded) vocabulary, so p and q must too.
            self.draft_variants = 1
            self.vocab_limit = 1 << 30
        workers = base._policy.worker_group
        eods = ray.get(
            workers.run_all_workers_single_data(
                "install_keyed_sampler", vocab_limit=vocab_limit, head_k=head_k
            )
            if verify_mode == "keyed"
            else workers.run_all_workers_single_data("termination_id")
        )
        self.eod = next(e for e in eods if e is not None)
        self._seeds = itertools.count(1)
        self._verify_seeds = itertools.count(1)
        self.plans: dict[tuple[int, ...], list[Plan]] = defaultdict(list)
        self.stats: dict[str, float] = {}
        # Multi-iteration drafts: target step -> drafts for that step's rollouts.
        # A draft keeps growing across iterations (resumed from its prefix under
        # whatever weights the engine holds); the learner stores, per position,
        # the q of the weights that drafted it.
        self.cohorts: dict[int, list[dict[str, Any]]] = {}

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
        fields = {
            "input_ids": ids,
            "input_lengths": torch.tensor([prompts[i].numel() for i, _ in rows]),
            "max_new_tokens": torch.full((len(rows),), self.draft_budget),
        }
        if self.verify_mode == "keyed":
            fields["noise_seed"] = torch.tensor([seeds[i] for i, _ in rows])
            fields["draft_variant"] = torch.tensor([v for _, v in rows])
            fields["variant_eps"] = torch.tensor(
                [self.variant_eps if v else 0.0 for _, v in rows], dtype=torch.float64
            )
        else:
            seeds = [-1] * len(prompts)
        data = BatchedDataDict(fields)
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

    async def draft_streaming(self, prompts: list[torch.Tensor]):
        """Start drafts on the engine without waiting; return a collector.

        The collector, called at the deadline (right before the optimizer step),
        aborts unfinished drafts and returns every draft's tokens so far. Any
        prefix of a draft is a valid draft for verification.
        """
        rank0 = self.base._policy.worker_group.workers[0]
        ray.get(
            rank0.start_drafts.remote(
                [p.tolist() for p in prompts], self.draft_budget, self.stream_interval
            )
        )

        step = self.current_step

        def collect():
            res = ray.get(rank0.collect_drafts.remote())
            return [
                [(prompts[i], -1, toks, done, f"{step + 1}:{i}", [(step, 0, len(toks))])]
                for i, (toks, done) in enumerate(res)
            ]

        return collect

    async def draft_cohorts(self, step: int, new: dict[int, list[torch.Tensor]]):
        """Resume unfinished drafts of future steps and start new cohorts.

        Returns a collector to call at this iteration's deadline; it appends the
        streamed tokens and returns a ``CohortDrafts`` for ``score_prev``.
        """
        for target, prompts in new.items():
            if target not in self.cohorts:
                self.cohorts[target] = [
                    {"prompt": p, "tokens": [], "finished": False, "key": f"{target}:{j}", "segments": []}
                    for j, p in enumerate(prompts)
                ]
        live = [
            d
            for target in sorted(self.cohorts)
            if target > step
            for d in self.cohorts[target]
            if not d["finished"]
        ]
        rank0 = self.base._policy.worker_group.workers[0]
        if live:
            ray.get(
                rank0.start_drafts.remote(
                    [torch.cat([d["prompt"], torch.tensor(d["tokens"], dtype=torch.long)]).tolist()
                     for d in live],
                    [self.max_new_tokens - len(d["tokens"]) for d in live],
                    self.stream_interval,
                )
            )

        def collect():
            res = ray.get(rank0.collect_drafts.remote()) if live else []
            prev = [len(d["tokens"]) for d in live]
            for d, (toks, done) in zip(live, res):
                start = len(d["tokens"])
                d["tokens"].extend(toks)
                d["finished"] = done
                d["segments"].append((step, start, len(d["tokens"])))
            return CohortDrafts(live, prev)

        return collect

    def verify_cohort(self, step: int) -> bool:
        """Block-verify every draft of ``step``'s cohort (learner at theta_step)."""
        cohort = self.cohorts.pop(step, None)
        if cohort is None:
            return False
        drafts = [
            [(d["prompt"], -1, d["tokens"], d["finished"], d["key"], d["segments"])]
            for d in cohort
        ]
        self.verify(drafts)
        return True

    # -- verification (start of iteration k+1, learner holds theta_{k+1}) -------

    @staticmethod
    def _flat_rows(drafts):
        flat = [(t, d) for t, variants in enumerate(drafts) for d in variants]
        rows = [torch.cat([d[0], torch.tensor(d[2], dtype=torch.long)]) for _, d in flat]
        return flat, rows

    def score_prev(self, drafts) -> None:
        """Block mode, learner still at theta_k: store each draft's full log q."""
        if self.verify_mode != "block" or drafts is None:
            return
        if self.q_storage == "stash":
            # Learner still at theta_k: stash its weights; keep only versions that
            # pending drafts were drafted with.
            keep = {self.current_step}
            for cohort in self.cohorts.values():
                for d in cohort:
                    keep.update(v for v, _, _ in d["segments"])
            ray.get(
                [
                    ref
                    for g in self.verifiers
                    for ref in g.run_all_workers_single_data(
                        "stash_weights",
                        version=self.current_step,
                        keep=sorted(keep),
                        precision="fp32" if self.verify_precision in ("fp32", "tf32") else "model",
                    )
                ]
            )
            return
        if isinstance(drafts, CohortDrafts):
            if not drafts.live:
                return
            ray.get(
                self.learner_policy.worker_group.run_all_workers_single_data(
                    "score_drafts_q",
                    rows=[
                        torch.cat([d["prompt"], torch.tensor(d["tokens"], dtype=torch.long)])
                        for d in drafts.live
                    ],
                    prompt_lens=[d["prompt"].numel() for d in drafts.live],
                    vocab_limit=self.vocab_limit,
                    precision=self.verify_precision,
                    keys=[d["key"] for d in drafts.live],
                    from_lens=drafts.prev_lens,
                )
            )
            return
        flat, rows = self._flat_rows(drafts)
        ray.get(
            self.learner_policy.worker_group.run_all_workers_single_data(
                "score_drafts_q",
                rows=rows,
                prompt_lens=[d[0].numel() for _, d in flat],
                vocab_limit=self.vocab_limit,
                precision=self.verify_precision,
            )
        )

    def verify(self, drafts: list[list[tuple[torch.Tensor, int, list[int]]]]) -> None:
        flat, rows = self._flat_rows(drafts)
        if self.verify_mode == "block" and self.q_storage == "stash":
            # Contiguous row slices per verifier pool, verified concurrently.
            k = len(self.verifiers)
            cuts = [round(j * len(flat) / k) for j in range(k + 1)]
            seed = next(self._verify_seeds)
            futs = []
            for g, a, b in zip(self.verifiers, cuts, cuts[1:]):
                futs.append((a, g.run_all_workers_single_data(
                    "verify_drafts_block_stashed",
                    rows=rows[a:b],
                    prompt_lens=[d[0].numel() for _, d in flat[a:b]],
                    segments=[d[5] for _, d in flat[a:b]],
                    vocab_limit=self.vocab_limit,
                    seed=seed * 7 + a,
                    keys=[d[4] for _, d in flat[a:b]],
                    precision="fp32" if self.verify_precision in ("fp32", "tf32") else "model",
                    batch_tokens=self.verify_batch_tokens,
                )))
            res = [
                [(a + j, r) for j, r in rank_res]
                for a, refs in futs
                for rank_res in ray.get(refs)
            ]
        elif self.verify_mode == "block":
            res = ray.get(
                self.learner_policy.worker_group.run_all_workers_single_data(
                    "verify_drafts_block",
                    rows=rows,
                    prompt_lens=[d[0].numel() for _, d in flat],
                    vocab_limit=self.vocab_limit,
                    seed=next(self._verify_seeds),
                    precision=self.verify_precision,
                    keys=[d[4] for _, d in flat] if len(flat[0][1]) > 4 else None,
                )
            )
        else:
            res = ray.get(
                self.learner_policy.worker_group.run_all_workers_single_data(
                    "verify_drafts",
                    rows=rows,
                    prompt_lens=[d[0].numel() for _, d in flat],
                    seeds=[d[1] for _, d in flat],
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
            d, r = best[i]
            plan = self._plan(d, r)
            full += plan.complete and r["accepted"] == len(d[2])
            kept += r["accepted"]
            total += len(d[2])
            self.plans[_key(d[0])].append(plan)
        self.stats = {
            "spec/draft_tokens": float(total),
            "spec/accepted_frac": kept / max(total, 1),
            "spec/full_accept": full / max(len(drafts), 1),
            "spec/variant_wins": float(variant_wins),
        }

    def _plan(self, d, r) -> Plan:
        """Turn one draft and its verification result into a rollout plan."""
        prompt, seed, draft = d[0], d[1], d[2]
        n_acc = r["accepted"]
        # The draft terminated on its own (EOS/stop) rather than at the budget
        # or the deadline: a fully accepted draft is then a finished rollout.
        finished = d[3] if len(d) > 3 else True
        # Cohort/streamed drafts carry the full budget: finished = EOS or the cap.
        stopped = finished and (len(d) > 4 or len(draft) < self.draft_budget)
        if n_acc == len(draft) and stopped:
            return Plan(seed, draft, r["logprobs"][:n_acc], complete=True)
        prefix = draft[:n_acc] + [r["next"]]
        done = r["next"] == self.eod or len(prefix) >= self.max_new_tokens
        return Plan(seed, prefix, r["logprobs"], complete=done)

    async def verify_chunked(self, drafts, chunks: int) -> None:
        """Run ``_verify_chunked``; on failure wake every waiting rollout so the
        error surfaces instead of the rollouts waiting forever for plans."""
        self._verify_error = None
        try:
            await self._verify_chunked(drafts, chunks)
        except BaseException as e:
            self._verify_error = e
            cv = getattr(self, "_plan_cv", None)
            if cv is not None:
                async with cv:
                    cv.notify_all()
            raise

    async def _debug_compare_pools(self, rows) -> None:
        """THUNDERSYNC_SPEC_CHECK: score the same rows on every verifier pool,
        concurrently with live rollouts, and report the largest difference."""
        outs = []
        for g in self.verifiers:
            refs = g.run_all_workers_single_data("scorer_token_logprobs", rows=rows)
            res = await asyncio.gather(*[asyncio.wrap_future(r.future()) for r in refs])
            outs.append([x for x in res if x is not None][0])
        sums = []
        for g in self.verifiers:
            refs = g.run_all_workers_single_data("model_param_checksum")
            sums.append(await asyncio.gather(*[asyncio.wrap_future(r.future()) for r in refs]))
        diffs = [float((a - b).abs().max()) for a, b in zip(outs[0], outs[-1])]
        print(f"[spec check] pools={len(outs)} max|p diff| per row={[round(d, 6) for d in diffs]} param sums={sums}", flush=True)

    def _group_chunks(self, flat, rows, chunks: int):
        """Order rows group by group and cut chunks at group boundaries.

        The learner trains on complete groups, so chunks never split a group.
        An optional first chunk holds the ``first_chunk_groups`` cheapest fully
        drafted groups (fast to verify, no decoding left); the other groups follow
        longest first (likely stragglers' continuations start early), split into
        chunks of equal token counts.
        """
        groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for j, (_, d) in enumerate(flat):
            groups[_key(d[0])].append(j)
        cost = {k: sum(rows[j].numel() for j in js) for k, js in groups.items()}
        # Groups whose drafts all finished complete as soon as they verify;
        # unfinished (deadline-truncated) drafts still need decoding.
        finished = {
            k: all(len(flat[j][1]) <= 3 or flat[j][1][3] for j in js) for k, js in groups.items()
        }
        by_cost = sorted(groups, key=lambda k: (not finished[k], cost[k]))
        first = by_cost[: self.first_chunk_groups]
        rest = sorted(by_cost[self.first_chunk_groups :], key=lambda k: -max(
            len(flat[j][1][2]) for j in groups[k]))
        order = [j for k in first + rest for j in groups[k]]
        bounds = [0] + ([sum(len(groups[k]) for k in first)] if first else [])
        rest_chunks = chunks - len(bounds) + 1
        total = sum(cost[k] for k in rest)
        acc, pos = 0, bounds[-1]
        for k in rest:
            acc += cost[k]
            pos += len(groups[k])
            done = len(bounds) - (2 if first else 1)  # rest chunks closed so far
            if done + 1 < rest_chunks and acc >= total * (done + 1) / rest_chunks:
                bounds.append(pos)
        if bounds[-1] != len(order):
            bounds.append(len(order))
        return [flat[j] for j in order], [rows[j] for j in order], bounds

    async def _verify_chunked(self, drafts, chunks: int) -> None:
        """Stash-mode verification in ``chunks`` pieces, publishing plans as each
        chunk finishes so rollouts start without waiting for the whole batch.

        Chunks keep the drafts' order (no prioritization); every chunk call is
        queued on the learner at once and processed in order.
        """
        flat, rows = self._flat_rows(drafts)
        if os.environ.get("THUNDERSYNC_SPEC_CHECK") and len(self.verifiers) > 1:
            await self._debug_compare_pools(rows[:4])
        if self.longest_first:
            # Longest drafts first: their continuations (the likely critical path)
            # start earliest. Verification order does not change any outcome.
            order = sorted(range(len(flat)), key=lambda j: -len(flat[j][1][2]))
            flat = [flat[j] for j in order]
            rows = [rows[j] for j in order]
        self.plans.clear()
        self._pending = defaultdict(int)
        for _, d in flat:
            self._pending[_key(d[0])] += 1
        self._plan_cv = asyncio.Condition()
        n = len(flat)
        bounds = [round(c * n / chunks) for c in range(chunks + 1)]
        if (self.group_aligned or self.first_chunk_groups) and chunks > 1:
            flat, rows, bounds = self._group_chunks(flat, rows, chunks)
        calls = []
        for c, (a, b) in enumerate(zip(bounds, bounds[1:])):
            if b <= a:
                continue
            # Weighted round-robin of chunks over the verifier pools.
            group = self._chunk_pools[c % len(self._chunk_pools)]
            refs = group.run_all_workers_single_data(
                "verify_drafts_block_stashed",
                rows=rows[a:b],
                prompt_lens=[d[0].numel() for _, d in flat[a:b]],
                segments=[d[5] for _, d in flat[a:b]],
                vocab_limit=self.vocab_limit,
                seed=next(self._verify_seeds),
                keys=[d[4] for _, d in flat[a:b]],
                batch_tokens=self.verify_batch_tokens,
                precision="fp32" if self.verify_precision in ("fp32", "tf32") else "model",
            )
            calls.append((a, refs))
        kept = total = full = 0

        async def fetch(a, refs):
            return a, refs, await asyncio.gather(
                *[asyncio.wrap_future(r.future()) for r in refs]
            )

        for fut in asyncio.as_completed([fetch(a, refs) for a, refs in calls]):
            a, refs, res = await fut
            by_row = dict(x for rank_res in res for x in rank_res)
            ends = [c[0] for c in calls[1:]] + [len(flat)]
            chunk_rows = ends[[c[0] for c in calls].index(a)] - a
            if len(by_row) != chunk_rows:
                raise RuntimeError(
                    f"verification returned {len(by_row)} of {chunk_rows} rows "
                    f"(per-rank counts {[len(x) for x in res]}): row ownership mismatch"
                )
            log_path = os.environ.get("THUNDERSYNC_REJECT_LOG")
            if log_path:
                import json

                with open(log_path, "a") as f:
                    for j, r in by_row.items():
                        d = flat[a + j][1]
                        f.write(json.dumps({
                            "step": self.current_step, "accepted": r["accepted"],
                            "draft_len": len(d[2]), "finished": bool(d[3]) if len(d) > 3 else True,
                            "segments": d[5] if len(d) > 5 else None,
                        }) + "\n")
            async with self._plan_cv:
                for j, r in by_row.items():
                    d = flat[a + j][1]
                    plan = self._plan(d, r)
                    full += plan.complete and r["accepted"] == len(d[2])
                    kept += r["accepted"]
                    total += len(d[2])
                    self.plans[_key(d[0])].append(plan)
                    self._pending[_key(d[0])] -= 1
                self._plan_cv.notify_all()
        self.stats = {
            "spec/draft_tokens": float(total),
            "spec/accepted_frac": kept / max(total, 1),
            "spec/full_accept": full / max(n, 1),
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
        key = _key(prompt)
        cv = getattr(self, "_plan_cv", None)
        if cv is not None:
            # Chunked verification: wait until this trajectory's plan is published.
            async with cv:
                while not self.plans.get(key) and self._pending.get(key, 0) > 0:
                    if getattr(self, "_verify_error", None) is not None:
                        raise RuntimeError("draft verification failed") from self._verify_error
                    await cv.wait()
        queue = self.plans.get(key)
        plan = queue.pop(0) if queue else Plan(
            seed=next(self._seeds) if self.verify_mode == "keyed" else -1
        )
        gen = list(plan.prefix)
        lps = list(plan.prefix_logprobs)
        if not plan.complete:
            ids = torch.cat([prompt, torch.tensor(plan.prefix, dtype=torch.long)])
            cont = BatchedDataDict(
                {
                    "input_ids": ids.view(1, -1),
                    "input_lengths": torch.tensor([ids.numel()]),
                    "max_new_tokens": torch.tensor([self.max_new_tokens - len(gen)]),
                }
            )
            if plan.seed >= 0:
                cont["noise_seed"] = torch.tensor([plan.seed])
            if "stop_strings" in datum:
                cont["stop_strings"] = datum["stop_strings"]
            t0 = time.perf_counter()
            async for _, res in self.base.generate_async(cont):
                glen = int(res["generation_lengths"][0])
                c0 = ids.numel()
                gen += res["output_ids"][0, c0 : c0 + glen].tolist()
                lps += res["logprobs"][0, c0 : c0 + glen].tolist()
            log_path = os.environ.get("THUNDERSYNC_REJECT_LOG")
            if log_path:
                import json

                with open(log_path, "a") as f:
                    f.write(json.dumps({
                        "cont_step": self.current_step, "prefix": len(plan.prefix),
                        "cont_tokens": len(gen) - len(plan.prefix),
                        "t_start": t0, "secs": time.perf_counter() - t0,
                    }) + "\n")
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
