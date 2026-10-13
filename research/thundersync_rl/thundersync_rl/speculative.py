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
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from thundersync_rl.engine_q import (
    HiddenRefs,
    cat_hidden,
    free_hidden,
    hidden_nbytes,
    hidden_num_rows,
)
from thundersync_rl.fused_train import FusedRef
from thundersync_rl.routes import (
    anchored_draft_length,
    merge_route_parts,
    route_bytes,
    splice,
)


@dataclass
class Plan:
    """How one rollout of the current iteration will be produced."""

    seed: int
    prefix: list[int] = field(default_factory=list)  # tokens already fixed
    prefix_logprobs: list[float] = field(default_factory=list)
    complete: bool = False
    # Fused verification + training: the learner kept the verification
    # forward of this (complete, fully accepted) draft for training.
    fused: Optional[FusedRef] = None
    # Tail lookup: the rejected draft's tokens after the rejection point, a
    # reference for exact lookup speculative decoding of the continuation.
    dead_suffix: list[int] = field(default_factory=list)
    # Sticky MoE routing: the drafter's routes for prompt + prefix positions
    # ([positions, L, K] int16, -1 = self); the training forward of this
    # rollout replays them (routes.py).
    anchors: Optional[torch.Tensor] = field(default=None, compare=False)


@dataclass
class CohortDrafts:
    """Drafts advanced during one iteration (their tokens were drafted at its theta)."""

    live: list[dict[str, Any]]


def _key(ids: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(t) for t in ids.tolist())


def _drafted(res) -> tuple[list[int], bool, list[float] | None]:
    """(tokens, finished, engine log-probs or None) of one ``collect_drafts`` entry."""
    toks, done, *rest = res
    lps = rest[0] if rest else None
    return toks, done, (lps if lps is not None and len(lps) == len(toks) else None)


def cohort_draft(d: dict[str, Any]) -> tuple:
    """Draft tuple of a cohort draft, as ``verify``/``verify_chunked`` take them.

    (prompt, seed, tokens, finished, key, segments, engine log-probs or None,
    engine hidden states or None: a [len(tokens), d] tensor or, with
    ``hidden_transport="object_store"``, ``HiddenRefs`` handles)
    """
    hidden = d.get("hidden")
    return (
        d["prompt"],
        -1,
        d["tokens"],
        d["finished"],
        d["key"],
        d["segments"],
        d.get("logprobs"),
        None if hidden is None else cat_hidden(hidden),
    )


def _truncate_to_rows(drafted, hidden):
    """Cut a drafted segment to the rows the engine logged for it.

    ``hidden`` holds the longest logged prefix of rows (see ``HiddenLog.pop``).
    Any prefix of a draft is a valid draft, so a segment with fewer rows than
    tokens is truncated there and no longer counts as finished. Exact as long
    as whether a row is logged does not depend on the draft's token values
    (rows are dropped by engine bookkeeping, not by content).
    """
    toks, done, lps = drafted
    if hidden is None:
        return drafted
    m = hidden_num_rows(hidden)
    if m >= len(toks):
        return drafted
    return toks[:m], False, (None if lps is None else lps[:m])


def engine_q_ready(d: tuple) -> bool:
    """Whether a draft carries what engine-sourced verification needs.

    The engine's log-prob of every token (index 6) and its hidden state at
    every position (index 7). Drafts without them are dropped before
    verification (their rollouts decode from scratch): exact, since whether
    the engine logged a draft does not depend on its tokens.
    """
    n = len(d[2])
    lps = d[6] if len(d) > 6 else None
    hid = d[7] if len(d) > 7 else None
    return (
        lps is not None
        and len(lps) == n
        and hid is not None
        and (n == 0 or hidden_num_rows(hid) >= n)
    )


def behavior_logprobs(
    logp: list[float],
    n_acc: int,
    engine_logq: list[float] | None,
    draft_logq: list[float] | None,
) -> list[float]:
    """Generation log-probs reported for a verified draft's kept tokens.

    ``logp``: log p (verifier, theta_{k+1}) of the kept tokens; the first
    ``n_acc`` are accepted draft tokens, any further one the verifier's own
    bonus/residual token. An accepted draft token was sampled by the engine from
    q' (bf16 engine law) and accepted against q (the scorer's recompute of the
    drafting weights): its output law is p + K^T (q' - q) ~= p * q'/q, so its
    behavior log-prob is log p + log q' - log q. The verifier's own token keeps
    log p: it is drawn by the verifier (bonus: exactly from p; residual: from
    norm(max(b p - q, 0)), which involves q, never q'), so there is no q'/q factor
    to attribute to it. Without the engine's or the verifier's log q (keyed mode,
    non-streamed drafts, or an engine that did not report log-probs) log p is
    reported unchanged.
    """
    if (
        engine_logq is None
        or draft_logq is None
        or len(draft_logq) < n_acc
        or len(engine_logq) < n_acc
    ):
        return list(logp)
    out = list(logp)
    for j in range(n_acc):
        if engine_logq[j] == draft_logq[j]:
            # q is the engine's law (engine-sourced verification): no
            # correction, and log p is reported bitwise.
            continue
        v = logp[j] + engine_logq[j] - draft_logq[j]
        if math.isfinite(v):
            out[j] = v
    return out


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
        draft_variants: int,
        variant_eps: float,
        verify_mode: str,
        verify_precision: str,
        verify_batch_tokens: int,
        longest_first: bool,
        tail_lookup: bool = False,
        tail_lookup_max_refs: int = 8,
        verify_on: str,
        pool_weights: tuple[int, ...] | None,
        fused_verify_train: bool,
        fused_max_retained_tokens: int,
        verify_lazy: bool = False,
        verify_source: str = "scorer",
        hidden_transport: str = "object_store",
        sticky_delta: Optional[float] = None,
        null_bv: bool = False,
        null_bv_policy: Optional[dict[str, Any]] = None,
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
        self.verify_lazy = verify_lazy
        # "scorer": p and q from the scorer copy and whole-model weight stashes.
        # "engine" (batch invariance): q from the engine's own log-probs and
        # logged hidden states with an LM-head-only stash, p from the training
        # model (engine_q.py).
        self.verify_source = verify_source
        # How the engine's hidden states reach the learner (engine source):
        # "object_store": the generation worker puts each draft's rows into the
        # Ray object store (blocks of HIDDEN_BLOCK_ROWS) and the driver passes
        # only handles; the learner fetches a block when its lazy scan first
        # needs a q row in it (fully accepted drafts fetch nothing). "driver": the rows travel
        # generation worker -> driver -> every learner worker (A/B baseline).
        if hidden_transport not in ("object_store", "driver"):
            raise ValueError(f"unknown hidden_transport {hidden_transport!r}")
        self.hidden_transport = hidden_transport
        # Deletes freed hidden-state objects (None: ray_free; tests replace it).
        self._free_refs = None
        self._hidden_nonce = itertools.count()
        self.verify_batch_tokens = verify_batch_tokens
        self.tail_lookup = tail_lookup
        self.tail_lookup_max_refs = tail_lookup_max_refs
        self.longest_first = longest_first
        # Fused verification + training (fused_train.py): p comes from the
        # training model with autograd, and the forwards of drafts that become
        # finished rollouts are kept (up to this many tokens per learner rank)
        # for their training step.
        self.fused_verify_train = fused_verify_train
        self.fused_max_retained_tokens = fused_max_retained_tokens
        # Complete plans with a retained forward, handed to the rollout that
        # consumed them: prompt key -> [(generated tokens, ref)].
        self._fused_claims: dict[
            tuple[int, ...], list[tuple[tuple[int, ...], FusedRef]]
        ] = defaultdict(list)
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
        if tail_lookup:
            ray.get(workers.run_all_workers_single_data("install_lookup_references"))
        if verify_source == "engine":
            ok = ray.get(
                workers.run_all_workers_single_data("install_draft_hidden_log")
            )
            if not any(x is True for x in ok) or any(x is False for x in ok):
                print(
                    "[thundersync] verify_source=engine: the engine cannot log "
                    f"hidden states ({ok}); falling back to verify_source=scorer",
                    flush=True,
                )
                self.verify_source = "scorer"
        # Sticky MoE routing (routes.py, nemo_rl sticky_routing.py): drafts
        # carry the drafter's routes (draft tuple index 8) and verification /
        # training replay them; continuations up to the rollout cap are drafted
        # by the same (previous) weights and verified (``verify_sticky``).
        self.sticky_delta = sticky_delta
        self._anchor_claims: dict[
            tuple[int, ...], list[tuple[tuple[int, ...], torch.Tensor]]
        ] = defaultdict(list)
        if sticky_delta is not None:
            if self.verify_source != "engine" or verify_mode != "block":
                raise ValueError(
                    "sticky routing needs verify_mode='block' with "
                    "verify_source='engine' (the engine logs the drafter's routes "
                    f"with its hidden states; got {verify_mode!r}/{self.verify_source!r})"
                )
            ok = ray.get(
                workers.run_all_workers_single_data("install_draft_route_log")
            )
            if not any(x is True for x in ok) or any(x is False for x in ok):
                raise RuntimeError(
                    "sticky routing: the engine cannot log the drafter's routes "
                    f"({ok}); is policy.megatron_cfg.sticky_routing_delta set on "
                    "the generation workers (RouterReplay RECORD)?"
                )
        # Pooled null-augmented block verification (null_bv.py): the engine
        # drafts K G drafts per prompt; verification fills the G slots of each
        # prompt with the first non-BOT outcomes in drafting order. Target
        # step -> {prompt key: slots} of the drafts made for that step.
        self.null_bv = null_bv
        self.null_bv_policy = dict(null_bv_policy or {})
        self._null_slots: dict[int, dict[tuple[int, ...], int]] = {}
        if null_bv and (self.verify_source != "engine" or verify_mode != "block"):
            raise ValueError(
                "null_bv needs verify_mode='block' with verify_source='engine' "
                f"(got {verify_mode!r}/{self.verify_source!r}; the scorer path is "
                "not built yet)"
            )
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

    async def draft(
        self, prompts: list[torch.Tensor]
    ) -> list[list[tuple[torch.Tensor, int, list[int]]]]:
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
            "max_new_tokens": torch.full(
                (len(rows),), min(self.draft_budget, self.max_new_tokens)
            ),
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
            out[i][v] = (
                prompts[i],
                seeds[i],
                res["output_ids"][0, plen : plen + glen].tolist(),
            )
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
                    batch_tokens=self.verify_batch_tokens,
                    position_shift=shift,
                )
            )
            by_row = dict(x for rr in res for x in rr)
            out[f"accepted_shift{shift}"] = [
                (by_row[i]["accepted"], len(a[i][2])) for i in range(len(a))
            ]
        return out

    async def draft_streaming(
        self,
        prompts: list[torch.Tensor],
        null_slots: Optional[dict[tuple[int, ...], int]] = None,
    ):
        """Start drafts on the engine without waiting; return a collector.

        The collector, called at the deadline (right before the optimizer step),
        aborts unfinished drafts and returns every draft's tokens so far. Any
        prefix of a draft is a valid draft for verification.

        ``null_slots`` (⊥-BV): rollouts the next step will request per prompt
        key; ``prompts`` then holds K G drafts per GRPO group, and their
        order here is the fixed verification order of each prompt's pool.
        """
        rank0 = self.base._policy.worker_group.workers[0]
        budget = min(self.draft_budget, self.max_new_tokens)
        step = self.current_step
        if null_slots is not None:
            for t in [t for t in self._null_slots if t <= step]:
                del self._null_slots[t]  # never verified (no drafts reached it)
            self._null_slots[step + 1] = dict(null_slots)
        hkeys = self._hidden_keys([f"{step + 1}:{i}" for i in range(len(prompts))])
        ray.get(
            rank0.start_drafts.remote(
                [p.tolist() for p in prompts],
                budget,
                self.stream_interval,
                **({} if hkeys is None else {"hidden_keys": hkeys}),
            )
        )

        def collect():
            res = [_drafted(x) for x in self._collect_drafts(rank0)]
            res, hid, anchors = self._pop_logs(
                hkeys, [p.numel() for p in prompts], res
            )
            return [
                [
                    (
                        prompts[i],
                        -1,
                        toks,
                        # Stopping at a draft budget below max_new_tokens is not
                        # the end of the rollout (EOS exactly at the budget is).
                        done
                        and (
                            len(toks) < budget
                            or budget == self.max_new_tokens
                            or (bool(toks) and toks[-1] == self.eod)
                        ),
                        f"{step + 1}:{i}",
                        [(step, 0, len(toks))],
                        lps,
                        hid[i],
                        *(() if self.sticky_delta is None else (anchors[i],)),
                    )
                ]
                for i, (toks, done, lps) in enumerate(res)
            ]

        return collect

    def _collect_drafts(self, rank0) -> list:
        """Snapshot and abort the drafts, content from the engines' request records.

        The engines (one per generation data-parallel rank) are asked for the
        record contents of the unfinished draft requests they hold; rank 0, which
        holds the streams, then aborts them and uses those contents instead of
        its streamed frames (which skip tokens after an eviction). Tokens decoded
        between the two calls are discarded: the cut is a time, not a function
        of the tokens, and any prefix of a draft is a valid draft.
        """
        ids = ray.get(rank0.draft_request_ids.remote())
        records: dict[int, tuple] = {}
        if ids:
            for r in ray.get(
                self.base._policy.worker_group.run_all_workers_single_data(
                    "draft_records", request_ids=sorted(ids.values())
                )
            ):
                records.update(r)
        return ray.get(rank0.collect_drafts.remote(records=records))

    def _hidden_keys(self, keys: list[str]) -> list[str] | None:
        """Engine hidden-log keys of one drafting round (None: not logging)."""
        if self.verify_source != "engine":
            return None
        nonce = next(self._hidden_nonce)
        return [f"{nonce}|{k}" for k in keys]

    def _pop_hidden(self, hkeys: list[str] | None, lengths: list[int]) -> list[Any]:
        """Each draft's logged engine hidden states, or None where missing.

        ``hidden_transport="object_store"``: ``HiddenRefs`` block handles
        (the generation worker put the rows into the object store; the
        driver never receives them). ``"driver"``: [n, d] CPU tensors.
        """
        if hkeys is None:
            return [None] * len(lengths)
        refs = self.hidden_transport == "object_store"
        merged: dict[str, Any] = {}
        for r in ray.get(
            self.base._policy.worker_group.run_all_workers_single_data(
                "pop_draft_hidden",
                keys=hkeys,
                lengths=lengths,
                **({"as_refs": True} if refs else {}),
            )
        ):
            merged.update(r)
        out: list[Any] = []
        for k, n in zip(hkeys, lengths):
            if n == 0:
                out.append(HiddenRefs() if refs else torch.empty(0, 0))
            elif k not in merged:
                out.append(None)
            elif refs:
                out.append(HiddenRefs.from_blocks(merged[k]))
            else:
                out.append(merged[k])
        self._count_hidden("spec/hidden_bytes_logged", sum(map(hidden_nbytes, out)))
        return out

    def _truncate(self, drafted, hidden):
        """``_truncate_to_rows``, counting truncated drafts / tokens in the stats."""
        out = _truncate_to_rows(drafted, hidden)
        cut = len(drafted[0]) - len(out[0])
        if cut:
            for name, v in (
                ("spec/hidden_truncated_drafts", 1.0),
                ("spec/hidden_truncated_tokens", float(cut)),
            ):
                self.stats[name] = self.stats.get(name, 0.0) + v
        return out

    def _pop_routes(
        self, hkeys: list[str] | None, prompt_lens: list[int], lengths: list[int]
    ) -> list[Optional[torch.Tensor]]:
        """Each draft's dense anchors [P + g, L, K] (sticky routing), None if unlogged.

        Every engine rank that routed the draft's tokens contributes its part
        (pipeline stages: their layers; sequence-parallel TP ranks: their
        positions); missing entries stay -1 and truncate the draft
        (``_truncate_to_anchors``).
        """
        n = len(lengths)
        if self.sticky_delta is None or hkeys is None:
            return [None] * n
        parts: dict[str, list] = defaultdict(list)
        for r in ray.get(
            self.base._policy.worker_group.run_all_workers_single_data(
                "pop_draft_routes", keys=hkeys
            )
        ):
            for k, v in r.items():
                parts[k].append(v)
        out: list[Optional[torch.Tensor]] = []
        for k, plen, g in zip(hkeys, prompt_lens, lengths):
            if k not in parts:
                out.append(None)
                continue
            out.append(merge_route_parts(parts[k], plen + g)[0])
        self.stats["spec/route_bytes_logged"] = self.stats.get(
            "spec/route_bytes_logged", 0.0
        ) + float(sum(map(route_bytes, out)))
        return out

    def _pop_logs(self, hkeys, prompt_lens: list[int], res: list, prior=None):
        """A drafting round's engine logs: (drafts, hidden states, anchors).

        Routes first: each draft is cut to its anchored length
        (``_truncate_to_anchors``; ``prior``: earlier anchors of the same
        rows, spliced under the new log). Hidden states are then popped for
        the kept tokens only and the draft cut to the logged rows
        (``_truncate``), so its hidden rows always match its token count, as
        verification requires.
        """
        anchors = self._pop_routes(hkeys, prompt_lens, [len(x[0]) for x in res])
        if prior is not None:
            anchors = [
                a if a is None else splice(b, a) for a, b in zip(anchors, prior)
            ]
        res = [
            self._truncate_to_anchors(x, a, plen)
            for x, a, plen in zip(res, anchors, prompt_lens)
        ]
        hid = self._pop_hidden(hkeys, [len(x[0]) for x in res])
        res = [self._truncate(x, h) for x, h in zip(res, hid)]
        return res, hid, anchors

    def _truncate_to_anchors(self, drafted, anchors, prompt_len: int):
        """Cut a draft to the tokens whose anchors were all logged (sticky routing).

        Any prefix of a draft is a valid draft, and whether a route was logged
        is engine bookkeeping, not a function of the tokens, so this is exact.
        """
        if self.sticky_delta is None or anchors is None:
            return drafted
        toks, done, lps = drafted
        m = anchored_draft_length(anchors, prompt_len, len(toks))
        if m >= len(toks):
            return drafted
        for name, v in (
            ("spec/route_truncated_drafts", 1.0),
            ("spec/route_truncated_tokens", float(len(toks) - m)),
        ):
            self.stats[name] = self.stats.get(name, 0.0) + v
        return toks[:m], False, (None if lps is None else lps[:m])

    def _count_hidden(self, name: str, nbytes: int) -> None:
        if self.verify_source == "engine":
            self.stats[name] = self.stats.get(name, 0.0) + float(nbytes)

    def _pending_hidden_bytes(self) -> int:
        """Bytes of hidden states that pending cohort drafts hold."""
        segs = {}
        for cohort in self.cohorts.values():
            for d in cohort:
                for p in d.get("hidden") or []:
                    for seg in p.segs if isinstance(p, HiddenRefs) else ():
                        if seg.ref is not None:
                            segs[id(seg)] = seg.nbytes
                    if isinstance(p, torch.Tensor):
                        segs[id(p)] = hidden_nbytes(p)
        return sum(segs.values())

    def _release_hidden(self, drafts) -> None:
        """Free these drafts' hidden states (verified or discarded).

        Drops the driver's handles and deletes the objects from the object
        stores at once (``free_hidden``), so no reference held elsewhere can
        keep them across steps. Only called after every learner call that
        could read them has returned.
        """
        self._free_hidden([d[7] if len(d) > 7 else None for d in drafts])

    def _free_hidden(self, hiddens) -> None:
        n = free_hidden(hiddens, self._free_refs)
        self._count_hidden("spec/hidden_bytes_freed", n)

    async def draft_cohorts(self, step: int, new: dict[int, list[torch.Tensor]]):
        """Resume unfinished drafts of future steps and start new cohorts.

        Returns a collector to call at this iteration's deadline; it appends the
        streamed tokens and returns a ``CohortDrafts`` for ``score_prev``.
        """
        for target, prompts in new.items():
            if target not in self.cohorts:
                self.cohorts[target] = [
                    {
                        "prompt": p,
                        "tokens": [],
                        "finished": False,
                        "key": f"{target}:{j}",
                        "segments": [],
                        # Engine log-probs of "tokens" (None once a segment
                        # came without them).
                        "logprobs": [],
                        # Engine hidden states per segment (engine-sourced
                        # q; None once a segment came without them).
                        "hidden": [],
                    }
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
        hkeys = self._hidden_keys([d["key"] for d in live]) if live else None
        if live:
            ray.get(
                rank0.start_drafts.remote(
                    [
                        torch.cat(
                            [d["prompt"], torch.tensor(d["tokens"], dtype=torch.long)]
                        ).tolist()
                        for d in live
                    ],
                    [self.max_new_tokens - len(d["tokens"]) for d in live],
                    self.stream_interval,
                    **({} if hkeys is None else {"hidden_keys": hkeys}),
                )
            )

        def collect():
            res = self._collect_drafts(rank0) if live else []
            hid = self._pop_hidden(hkeys, [len(_drafted(x)[0]) for x in res])
            for d, x, h in zip(live, res, hid):
                toks, done, lps = self._truncate(_drafted(x), h)
                if d.get("hidden") is not None and h is not None:
                    d["hidden"].append(h)
                else:
                    # Unverifiable with engine q from now on: free what it has.
                    self._free_hidden((d.get("hidden") or []) + [h])
                    d["hidden"] = None
                start = len(d["tokens"])
                d["tokens"].extend(toks)
                if d.get("logprobs") is not None and lps is not None:
                    d["logprobs"].extend(lps)
                else:
                    d["logprobs"] = None
                d["finished"] = done
                d["segments"].append((step, start, len(d["tokens"])))
            if hkeys is not None:
                # Host memory the pending drafts' hidden states occupy (object
                # stores of the generation nodes, or the driver's).
                self.stats["spec/hidden_bytes_pending"] = float(
                    self._pending_hidden_bytes()
                )
            return CohortDrafts(live)

        return collect

    def verify_cohort(self, step: int) -> bool:
        """Block-verify every draft of ``step``'s cohort (learner at theta_step)."""
        cohort = self.cohorts.pop(step, None)
        if cohort is None:
            return False
        self.verify([[cohort_draft(d)] for d in cohort])
        return True

    # -- verification (start of iteration k+1, learner holds theta_{k+1}) -------

    @staticmethod
    def _flat_rows(drafts):
        flat = [(t, d) for t, variants in enumerate(drafts) for d in variants]
        rows = [
            torch.cat([d[0], torch.tensor(d[2], dtype=torch.long)]) for _, d in flat
        ]
        return flat, rows

    def score_prev(self, drafts) -> None:
        """Block mode, verifiers still at theta_k: stash theta_k's weights.

        q of every token drafted at theta_k is recomputed from this stash at
        verification. Only versions that pending drafts were drafted with are kept.
        """
        if self.verify_mode != "block" or drafts is None:
            return
        keep = {self.current_step}
        for cohort in self.cohorts.values():
            for d in cohort:
                keep.update(v for v, _, _ in d["segments"])
        if self.verify_source == "engine":
            # Only the LM head: full q rows are rebuilt from the engine's
            # hidden states, never by a forward of the stashed version.
            method, kw = "stash_head", {}
        else:
            method, kw = "stash_weights", {"precision": self.verify_precision}
        ray.get(
            [
                ref
                for g in self.verifiers
                for ref in g.run_all_workers_single_data(
                    method, version=self.current_step, keep=sorted(keep), **kw
                )
            ]
        )

    def _engine_q_filter(self, drafts):
        """Engine-sourced q: drop drafts the engine did not fully log (counted)."""
        if self.verify_source != "engine":
            return drafts, 0
        def ready(d):
            if not engine_q_ready(d):
                return False
            # Sticky routing: p is defined only with the drafter's routes.
            return self.sticky_delta is None or (len(d) > 8 and d[8] is not None)

        kept = [[d for d in variants if ready(d)] for variants in drafts]
        self._release_hidden(
            d for variants in drafts for d in variants if not ready(d)
        )
        dropped = sum(len(v) for v in drafts) - sum(len(v) for v in kept)
        return [v for v in kept if v], dropped

    def _null_prepare(self, drafts):
        """⊥-BV for this step's drafts: (drafts, per-row group names, slots by
        group, empty drafts dropped), or None when the drafts are not pooled.

        Empty drafts (cut before their first token) are dropped: BOT lives at
        the first drafted position. Whether a draft is empty is a deadline
        (timing) fact, not a function of its tokens, as for any dropped draft.
        """
        slots_by_key = self._null_slots.pop(self.current_step, None)
        if not self.null_bv or slots_by_key is None:
            return None
        kept, empty = [], []
        for variants in drafts:
            v = [d for d in variants if len(d[2]) > 0]
            empty += [d for d in variants if len(d[2]) == 0]
            if v:
                kept.append(v)
        self._release_hidden(empty)
        names: dict[tuple[int, ...], str] = {}
        for variants in kept:
            for d in variants:
                names.setdefault(_key(d[0]), f"{self.current_step}:{len(names)}")
        slots = {names[k]: int(slots_by_key.get(k, 0)) for k in names}
        missing = [n for n, s in slots.items() if s <= 0]
        if missing:
            raise RuntimeError(f"null_bv: drafts for prompts without slots {missing}")
        self._null_names = names
        self._null_all_slots = slots_by_key
        return kept, names, slots, len(empty)

    def _verify_method(self, flat) -> tuple[str, dict[str, Any]]:
        """Worker method and its extra arguments for block verification."""
        if self.verify_source == "engine":
            hidden = [d[7] for _, d in flat]
            if self.hidden_transport == "driver":
                # Every row's hidden states pass through the driver.
                self._count_hidden(
                    "spec/hidden_bytes_fetched", sum(map(hidden_nbytes, hidden))
                )
            extra = {"engine_logq": [d[6] for _, d in flat], "hidden": hidden}
            if self.sticky_delta is not None:
                extra["anchors"] = [d[8] for _, d in flat]
            return "verify_drafts_block_engine", extra
        return "verify_drafts_block_stashed", {}

    def _verified_hidden(self, flat, by_row: dict[int, Any], a: int = 0) -> None:
        """Count fetched hidden bytes of verified rows a + j and free their handles."""
        if self.verify_source != "engine":
            return
        if self.hidden_transport == "object_store":
            self._count_hidden(
                "spec/hidden_bytes_fetched",
                sum(r.get("hidden_bytes_fetched", 0) for r in by_row.values()),
            )
        self._release_hidden(flat[a + j][1] for j in by_row)

    def verify(self, drafts: list[list[tuple[torch.Tensor, int, list[int]]]]) -> None:
        if self.fused_verify_train:
            raise NotImplementedError(
                "fused_verify_train runs through chunked verification only"
            )
        drafts, dropped = self._engine_q_filter(drafts)
        null = self._null_prepare(drafts)
        if null is not None:
            drafts, names, slots, n_empty = null
            flat, rows = self._flat_rows(drafts)
            self._record_siblings(flat)
            groups = [names[_key(d[0])] for _, d in flat]
            by_row = self._verify_block(
                flat,
                rows,
                null={
                    "null_groups": groups,
                    "null_slots": slots,
                    "null_policy": self.null_bv_policy,
                },
            )
            self._finish_verify_null(flat, by_row, groups, slots, dropped, n_empty)
            return
        flat, rows = self._flat_rows(drafts)
        self._record_siblings(flat)
        if self.verify_mode == "block":
            by_row = self._verify_block(flat, rows)
        else:
            res = ray.get(
                self.learner_policy.worker_group.run_all_workers_single_data(
                    "verify_drafts",
                    rows=rows,
                    prompt_lens=[d[0].numel() for _, d in flat],
                    seeds=[d[1] for _, d in flat],
                    vocab_limit=self.vocab_limit,
                    head_k=self.head_k,
                    batch_tokens=self.verify_batch_tokens,
                )
            )
            by_row = dict(x for rank_res in res for x in rank_res)
        self._finish_verify(drafts, flat, by_row, dropped)

    def _verify_block(self, flat, rows, null=None) -> dict[int, dict[str, Any]]:
        """Block-verify ``flat`` drafts on every verifier pool; results by row.

        ``null``: ⊥-BV arguments for the worker (``null_groups`` per row,
        ``null_slots``, ``null_policy``); a prompt's pool must be decided by
        one worker, so it needs a single verifier pool.
        """
        if null is not None and len(self.verifiers) != 1:
            raise RuntimeError("null_bv needs a single verifier pool (verify_on=learner)")
        if True:
            # Contiguous row slices per verifier pool, verified concurrently.
            k = len(self.verifiers)
            cuts = [round(j * len(flat) / k) for j in range(k + 1)]
            futs = []
            for g, a, b in zip(self.verifiers, cuts, cuts[1:]):
                method, extra = self._verify_method(flat[a:b])
                if null is not None:
                    extra = {
                        **extra,
                        "null_groups": null["null_groups"][a:b],
                        "null_slots": null["null_slots"],
                        "null_policy": null["null_policy"],
                    }
                futs.append(
                    (
                        a,
                        g.run_all_workers_single_data(
                            method,
                            **extra,
                            rows=rows[a:b],
                            prompt_lens=[d[0].numel() for _, d in flat[a:b]],
                            segments=[d[5] for _, d in flat[a:b]],
                            vocab_limit=self.vocab_limit,
                            # A fresh counter value per call: verification
                            # randomness is never reused across calls or steps
                            # (``seed * 7 + a`` collided between consecutive
                            # calls, e.g. (s, a=7) and (s+1, a=0), so a later
                            # decision could reuse a stream that shaped the
                            # update producing the policy it verifies against).
                            seed=next(self._verify_seeds),
                            keys=[d[4] for _, d in flat[a:b]],
                            precision=self.verify_precision,
                            batch_tokens=self.verify_batch_tokens,
                            lazy=self.verify_lazy,
                        ),
                    )
                )
            res = [
                [(a + j, r) for j, r in rank_res]
                for a, refs in futs
                for rank_res in ray.get(refs)
            ]
        by_row = dict(x for rank_res in res for x in rank_res)
        self._verified_hidden(flat, by_row)
        return by_row

    def _finish_verify(self, drafts, flat, by_row, dropped) -> None:
        """Plans and acceptance statistics from verification results."""
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
        # Draft age (versions between the drafting weights and the verifying
        # ones), token-weighted: drafted and kept tokens per age bucket.
        age_tok: dict[int, int] = {}
        age_kept: dict[int, int] = {}
        for i in range(len(drafts)):
            d, r = best[i]
            plan = self._plan(d, r)
            full += plan.complete and r["accepted"] == len(d[2])
            kept += r["accepted"]
            total += len(d[2])
            self._count_ages(d, r["accepted"], age_tok, age_kept)
            self.plans[_key(d[0])].append(plan)
        self.stats.update(
            {
                "spec/draft_tokens": float(total),
                "spec/accepted_frac": kept / max(total, 1),
                "spec/full_accept": full / max(len(drafts), 1),
                "spec/variant_wins": float(variant_wins),
            }
        )
        self._age_stats(age_tok, age_kept)
        if self.verify_source == "engine":
            self.stats["spec/engine_q_dropped"] = float(dropped)

    def _finish_verify_null(self, flat, by_row, groups, slots, dropped, n_empty):
        """Plans and ⊥-BV statistics: each prompt's outputs in drafting order.

        Slots left empty (drafts ran out) get no plan: those rollouts decode
        fresh from the current policy, an exact p-sample.
        """
        self.plans.clear()
        kept = total = full = 0
        n_out = n_bot = n_unused = n_tail = 0
        m_sum = 0.0
        outs_by_group: dict[str, int] = defaultdict(int)
        age_tok: dict[int, int] = {}
        age_kept: dict[int, int] = {}
        decided_len: list[int] = []
        rejected: list[tuple[int, int]] = []  # (draft length, tau) of tail decodes
        for r, (_, d) in enumerate(flat):
            res = by_row[r]
            state = res.get("null")
            if state is None:
                raise RuntimeError(f"null_bv: row {r} was verified without its group")
            if state == "unused":
                n_unused += 1
                continue
            m_sum += float(res["null_m"])
            g = len(d[2])
            decided_len.append(g)
            if state == "bot":
                n_bot += 1
                continue
            n_out += 1
            outs_by_group[groups[r]] += 1
            plan = self._plan(d, res)
            full += plan.complete and res["accepted"] == g
            kept += res["accepted"]
            total += g
            if res["accepted"] < g:
                n_tail += 1
                rejected.append((g, res["accepted"]))
            self._count_ages(d, res["accepted"], age_tok, age_kept)
            self.plans[_key(d[0])].append(plan)
        # Earliest rejection among the longest decided drafts (top decile by
        # length): the step waits on these continuations.
        n_dec = len(decided_len)
        if n_dec:
            cut = sorted(decided_len, reverse=True)[max(1, n_dec // 10) - 1]
            long_rej = [(g, t) for g, t in rejected if g >= cut]
        else:
            cut, long_rej = 0, []
        fresh = sum(max(0, s - outs_by_group[grp]) for grp, s in slots.items())
        # Prompts with slots but no verifiable draft at all decode fresh too.
        named = set(getattr(self, "_null_names", {}))
        fresh += sum(
            s for k, s in getattr(self, "_null_all_slots", {}).items() if k not in named
        )
        n_slots = sum(getattr(self, "_null_all_slots", {}).values())
        decided = n_out + n_bot
        self.stats.update(
            {
                "spec/draft_tokens": float(total),
                "spec/accepted_frac": kept / max(total, 1),
                "spec/full_accept": full / max(n_out, 1),
                "spec/null_drafts": float(len(flat)),
                "spec/null_slots": float(n_slots),
                "spec/null_outputs": float(n_out),
                "spec/null_bot_frac": n_bot / max(decided, 1),
                "spec/null_tail_frac": n_tail / max(n_out, 1),
                "spec/null_fresh_slots": float(fresh),
                "spec/null_unused_drafts": float(n_unused),
                "spec/null_empty_dropped": float(n_empty),
                "spec/null_drafts_per_output": decided / max(n_out, 1),
                "spec/null_mean_m": m_sum / max(decided, 1),
                "spec/null_longest_len_cut": float(cut),
                "spec/null_longest_rejections": float(len(long_rej)),
                "spec/null_longest_min_reject_pos": float(
                    min((t for _, t in long_rej), default=-1)
                ),
                "spec/null_longest_min_reject_frac": min(
                    (t / max(g, 1) for g, t in long_rej), default=1.0
                ),
            }
        )
        self._age_stats(age_tok, age_kept)
        if self.verify_source == "engine":
            self.stats["spec/engine_q_dropped"] = float(dropped)

    def _stopped(self, d) -> bool:
        """Whether draft ``d``, if fully accepted, is a finished rollout."""
        # The draft terminated on its own (EOS/stop) rather than at the budget
        # or the deadline: a fully accepted draft is then a finished rollout.
        finished = d[3] if len(d) > 3 else True
        # Cohort/streamed drafts carry the full budget: finished = EOS or the cap.
        stopped = finished and (len(d) > 4 or len(d[2]) < self.draft_budget)
        # A draft at the length cap is a finished rollout: nothing follows it.
        return stopped or len(d[2]) >= self.max_new_tokens

    def _plan(self, d, r) -> Plan:
        """Turn one draft and its verification result into a rollout plan."""
        prompt, seed, draft = d[0], d[1], d[2]
        n_acc = r["accepted"]
        lps = behavior_logprobs(
            r["logprobs"],
            n_acc,
            d[6] if len(d) > 6 else None,
            r.get("draft_logq"),
        )
        complete = n_acc == len(draft) and self._stopped(d)
        fused = r.get("fused")
        if fused is not None and not complete:
            # The worker keeps a graph only for drafts flagged as finished and
            # fully accepted; anything else means driver and worker disagree.
            raise RuntimeError(
                f"learner kept a fused forward for draft {fused['key']!r}, "
                f"which is not a finished rollout ({n_acc} of {len(draft)} "
                "tokens accepted)"
            )
        anchors = None
        if self.sticky_delta is not None:
            # Training replays the anchors of prompt + kept tokens only: entries
            # past the kept prefix are routes of rejected draft tokens.
            anchors = d[8][: prompt.numel() + n_acc]
            if n_acc == len(draft) and not complete:
                # The bonus row's anchor (the last drafted token's route) was
                # not logged: keep the whole draft, and the next token comes
                # from a continuation round (``verify_sticky``), an exact
                # sampler of p given this prefix like the bonus draw itself.
                return Plan(seed, list(draft), lps[:n_acc], anchors=anchors)
        if complete:
            return Plan(
                seed,
                draft,
                lps[:n_acc],
                complete=True,
                anchors=anchors,
                fused=None
                if fused is None
                else FusedRef(
                    key=fused["key"],
                    dp_rank=fused["dp_rank"],
                    num_tokens=prompt.numel() + len(draft),
                ),
            )
        prefix = draft[:n_acc] + [r["next"]]
        done = r["next"] == self.eod or len(prefix) >= self.max_new_tokens
        dead = list(draft[n_acc + 1 :]) if getattr(self, "tail_lookup", False) else []
        return Plan(
            seed, prefix, lps, complete=done, dead_suffix=dead, anchors=anchors
        )

    def _record_siblings(self, flat) -> None:
        """Draft tokens of every prompt verified this step (tail-lookup references)."""
        self._sibling_drafts: dict[tuple[int, ...], list[list[int]]] = {}
        if not self.tail_lookup:
            return
        for _, d in flat:
            self._sibling_drafts.setdefault(_key(d[0]), []).append(list(d[2]))

    def _lookup_references(self, key, plan: Plan) -> list[list[int]]:
        """References for the continuation's exact lookup decoding.

        The rejected draft's own tail first (the continuation often
        re-converges with it), then the drafts of the same prompt (its GRPO
        group). The engine also searches the request's own context. These only
        propose tokens; the engine's accept rule keeps the output an exact
        sample of the current policy whatever they contain.
        """
        refs = [plan.dead_suffix] if plan.dead_suffix else []
        for sib in getattr(self, "_sibling_drafts", {}).get(key, []):
            if len(refs) >= self.tail_lookup_max_refs:
                break
            refs.append(sib)
        return refs

    def _count_ages(self, d, accepted: int, age_tok: dict, age_kept: dict) -> None:
        """Add draft ``d``'s drafted and kept tokens per age (versions behind; 4 = 4+)."""
        for v, a, b in d[5] if len(d) > 5 and d[5] else ():
            age = min(max(self.current_step - v, 0), 4)
            age_tok[age] = age_tok.get(age, 0) + (b - a)
            age_kept[age] = age_kept.get(age, 0) + max(0, min(b, accepted) - a)

    def _age_stats(self, age_tok: dict, age_kept: dict) -> None:
        """spec/age{a}_token_frac and spec/age{a}_accepted_frac per draft age."""
        n_aged = sum(age_tok.values())
        for age, n in age_tok.items():
            label = f"{age}+" if age == 4 else str(age)
            self.stats[f"spec/age{label}_token_frac"] = n / max(n_aged, 1)
            self.stats[f"spec/age{label}_accepted_frac"] = age_kept[age] / max(n, 1)

    async def verify_chunked(self, drafts, chunks: int) -> None:
        """Run ``_verify_chunked``; on failure, wake every waiting rollout.

        The error then surfaces instead of the rollouts waiting forever for plans.
        """
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
        """Score the same rows on every verifier pool and report the largest difference.

        Debug check (THUNDERSYNC_SPEC_CHECK), run concurrently with live rollouts.
        """
        outs = []
        for g in self.verifiers:
            refs = g.run_all_workers_single_data("scorer_token_logprobs", rows=rows)
            res = await asyncio.gather(*[asyncio.wrap_future(r.future()) for r in refs])
            outs.append([x for x in res if x is not None][0])
        sums = []
        for g in self.verifiers:
            refs = g.run_all_workers_single_data("model_param_checksum")
            sums.append(
                await asyncio.gather(*[asyncio.wrap_future(r.future()) for r in refs])
            )
        diffs = [float((a - b).abs().max()) for a, b in zip(outs[0], outs[-1])]
        print(
            f"[spec check] pools={len(outs)} max|p diff| per row={[round(d, 6) for d in diffs]} param sums={sums}",
            flush=True,
        )

    async def _verify_chunked(self, drafts, chunks: int) -> None:
        """Stash-mode verification in ``chunks`` pieces, publishing plans per chunk.

        Rollouts start as soon as their chunk is verified instead of waiting for
        the whole batch. Rows are ordered longest draft first with
        ``longest_first``; chunks go round-robin over the verifier pools, and
        every chunk call is queued at once (each pool runs its calls in order).
        """
        if self.null_bv and self.current_step in self._null_slots:
            # Rollouts run concurrently with chunked verification and would
            # take an empty plan (fresh decode) before their pool is decided.
            raise RuntimeError(
                "null_bv: chunked/overlapped verification of ⊥-BV pools is not "
                "built yet (the loop calls verify() before the rollouts)"
            )
        drafts, dropped = self._engine_q_filter(drafts)
        flat, rows = self._flat_rows(drafts)
        self._record_siblings(flat)
        if os.environ.get("THUNDERSYNC_SPEC_CHECK") and len(self.verifiers) > 1:
            await self._debug_compare_pools(rows[:4])
        if self.longest_first:
            # Longest drafts first: their continuations (the likely critical path)
            # start earliest. Verification order does not change any outcome.
            order = sorted(range(len(flat)), key=lambda j: -len(flat[j][1][2]))
            flat = [flat[j] for j in order]
            rows = [rows[j] for j in order]
        self.plans.clear()
        self._fused_claims.clear()
        self._pending = defaultdict(int)
        for _, d in flat:
            self._pending[_key(d[0])] += 1
        self._plan_cv = asyncio.Condition()
        n = len(flat)
        bounds = [round(c * n / chunks) for c in range(chunks + 1)]
        calls = []
        for c, (a, b) in enumerate(zip(bounds, bounds[1:])):
            if b <= a:
                continue
            # Weighted round-robin of chunks over the verifier pools.
            group = self._chunk_pools[c % len(self._chunk_pools)]
            kwargs = dict(
                rows=rows[a:b],
                prompt_lens=[d[0].numel() for _, d in flat[a:b]],
                segments=[d[5] for _, d in flat[a:b]],
                vocab_limit=self.vocab_limit,
                seed=next(self._verify_seeds),
                keys=[d[4] for _, d in flat[a:b]],
                batch_tokens=self.verify_batch_tokens,
                precision=self.verify_precision,
                lazy=self.verify_lazy,
            )
            if self.fused_verify_train:
                refs = group.run_all_workers_single_data(
                    "verify_drafts_block_fused",
                    **kwargs,
                    retain=[self._stopped(d) for _, d in flat[a:b]],
                    max_retained_tokens=self.fused_max_retained_tokens,
                )
            else:
                method, extra = self._verify_method(flat[a:b])
                refs = group.run_all_workers_single_data(method, **kwargs, **extra)
            calls.append((a, refs))
        kept = total = full = fused = 0
        age_tok: dict[int, int] = {}
        age_kept: dict[int, int] = {}

        async def fetch(a, refs):
            return (
                a,
                refs,
                await asyncio.gather(*[asyncio.wrap_future(r.future()) for r in refs]),
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
            self._verified_hidden(flat, by_row, a)
            log_path = os.environ.get("THUNDERSYNC_REJECT_LOG")
            if log_path:
                import json

                with open(log_path, "a") as f:
                    for j, r in by_row.items():
                        d = flat[a + j][1]
                        f.write(
                            json.dumps(
                                {
                                    "step": self.current_step,
                                    "accepted": r["accepted"],
                                    "draft_len": len(d[2]),
                                    "finished": bool(d[3]) if len(d) > 3 else True,
                                    "segments": d[5] if len(d) > 5 else None,
                                }
                            )
                            + "\n"
                        )
            async with self._plan_cv:
                for j, r in by_row.items():
                    d = flat[a + j][1]
                    plan = self._plan(d, r)
                    full += plan.complete and r["accepted"] == len(d[2])
                    fused += plan.fused is not None
                    kept += r["accepted"]
                    total += len(d[2])
                    self._count_ages(d, r["accepted"], age_tok, age_kept)
                    self.plans[_key(d[0])].append(plan)
                    self._pending[_key(d[0])] -= 1
                self._plan_cv.notify_all()
        self.stats.update(
            {
                "spec/draft_tokens": float(total),
                "spec/accepted_frac": kept / max(total, 1),
                "spec/full_accept": full / max(n, 1),
            }
        )
        self._age_stats(age_tok, age_kept)
        if self.fused_verify_train:
            self.stats["fused/retained"] = float(fused)
        if self.verify_source == "engine":
            self.stats["spec/engine_q_dropped"] = float(dropped)

    # -- sticky routing: verification and theta_n continuations --------------------

    async def verify_sticky(self, drafts, max_rounds: int = 1000) -> None:
        """Verify drafts and complete every drafted rollout (sticky MoE routing).

        Called at the start of iteration k with the learner at theta_k and the
        engine still at theta_{k-1} (the refit waits until this returns). The
        target is p = F(theta_k; A): anchors A = plain theta_{k-1} routes on the
        rollout's own prefix at every position up to the rollout cap (the cut
        c = prompt + max_new_tokens, content-independent), "self" for rows
        never drafted. Positions after a rejection need theta_{k-1} routes on
        the new prefix, so the engine keeps theta_{k-1} and continues each
        incomplete plan in rounds: it drafts from prompt + kept prefix to the
        end (EOS or the cap) with its plain forward, logging routes, hidden
        states and log-probs as for any draft, and the learner block-verifies
        that draft against the same p (anchors: the earlier ones spliced with
        the new route log). Every round is exact block verification of a q
        draft against the fixed target p, so the final rollout is an exact
        sample of p. Rounds end when every plan is complete; the expected
        number is small when acceptance is high.
        """
        self._plan_cv = None
        self.verify(drafts)
        rounds = cont_tokens = cont_kept = retries = 0
        while True:
            pending = [
                (key, plan)
                for key, queue in self.plans.items()
                for plan in queue
                if not plan.complete
            ]
            if not pending:
                break
            if rounds >= max_rounds:
                raise RuntimeError(
                    f"sticky routing: {len(pending)} rollouts still incomplete after "
                    f"{rounds} continuation rounds"
                )
            rounds += 1
            n_tok, n_kept, n_retry = await self._sticky_round(pending, rounds)
            cont_tokens += n_tok
            cont_kept += n_kept
            retries += n_retry
        self.stats.update(
            {
                "spec/sticky_rounds": float(rounds),
                "spec/sticky_cont_draft_tokens": float(cont_tokens),
                "spec/sticky_cont_accepted_frac": cont_kept / max(cont_tokens, 1),
                "spec/sticky_cont_retries": float(retries),
            }
        )

    async def _sticky_round(self, pending, rnd: int) -> tuple[int, int, int]:
        """One continuation round: engine (theta_{k-1}) drafts, learner verifies."""
        rank0 = self.base._policy.worker_group.workers[0]
        prompts = [
            torch.cat(
                [torch.tensor(key, dtype=torch.long), torch.tensor(plan.prefix, dtype=torch.long)]
            )
            for key, plan in pending
        ]
        budgets = [self.max_new_tokens - len(plan.prefix) for _, plan in pending]
        hkeys = self._hidden_keys([f"c{rnd}:{j}" for j in range(len(pending))])
        ray.get(
            rank0.start_drafts.remote(
                [p.tolist() for p in prompts],
                budgets,
                self.stream_interval,
                hidden_keys=hkeys,
            )
        )
        # Run to completion (EOS or the cap): there is no deadline here.
        while not all(ray.get(rank0.drafts_done.remote())):
            await asyncio.sleep(0.02)
        res = [_drafted(x) for x in self._collect_drafts(rank0)]
        res, hid, routes = self._pop_logs(
            hkeys,
            [p.numel() for p in prompts],
            res,
            prior=[plan.anchors for _, plan in pending],
        )
        version = self.current_step - 1
        flat, rows, owners, retry = [], [], [], 0
        for j, ((key, plan), p, (toks, done, lps), h, a) in enumerate(
            zip(pending, prompts, res, hid, routes)
        ):
            d = (
                p,
                -1,
                toks,
                done,
                f"c{rnd}:{j}",
                [(version, 0, len(toks))],
                lps,
                h,
                a,
            )
            if not toks or not engine_q_ready(d) or a is None:
                # Nothing verifiable this round (logging is engine bookkeeping,
                # independent of the tokens): retry from the same prefix.
                self._release_hidden([d])
                retry += 1
                continue
            flat.append((len(owners), d))
            rows.append(torch.cat([p, torch.tensor(toks, dtype=torch.long)]))
            owners.append(plan)
        if not flat:
            return 0, 0, retry
        by_row = self._verify_block(flat, rows)
        n_tok = n_kept = 0
        for r, ((_, d), plan) in enumerate(zip(flat, owners)):
            out = by_row[r]
            p, toks, done = d[0], d[2], d[3]
            g, n_acc = len(toks), out["accepted"]
            lps = behavior_logprobs(out["logprobs"], n_acc, d[6], out.get("draft_logq"))
            n_tok += g
            n_kept += n_acc
            if n_acc == g:
                plan.prefix += list(toks)
                plan.prefix_logprobs += lps[:g]
                plan.anchors = d[8][: p.numel() + g]
                plan.complete = bool(done) or len(plan.prefix) >= self.max_new_tokens
            else:
                y = out["next"]
                plan.prefix += list(toks[:n_acc]) + [y]
                plan.prefix_logprobs += lps[: n_acc + 1]
                plan.anchors = d[8][: p.numel() + n_acc]
                plan.complete = y == self.eod or len(plan.prefix) >= self.max_new_tokens
        return n_tok, n_kept, retry

    def claim_anchors(
        self, prompt: torch.Tensor, token_ids: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Anchors of a finished speculative rollout (sticky routing), or None.

        ``prompt``: the generation prompt; ``token_ids``: the trajectory. None
        for rollouts that were never drafted: anchor = self, plain routing.
        """
        entries = self._anchor_claims.get(_key(prompt))
        if not entries:
            return None
        p = prompt.numel()
        for j, (gen, anchors) in enumerate(entries):
            if (
                token_ids.numel() >= p + len(gen)
                and _key(token_ids[p : p + len(gen)]) == gen
            ):
                del entries[j]
                return anchors
        return None

    # -- GenerationInterface used by the rollout ----------------------------------

    async def generate_async(
        self, data: BatchedDataDict, greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict], None]:
        assert not greedy
        tasks = [
            asyncio.create_task(self._one(i, data.get_batch(i, 1)))
            for i in range(data.size)
        ]
        for fut in asyncio.as_completed(tasks):
            yield await fut

    async def _keyed_replay_check(self, prompt, seed, gen, datum) -> None:
        """Debug: a keyed speculative rollout must equal plain keyed decoding.

        With batch invariance and the same random stream (noise_seed, keyed by
        position), speculation reproduces exactly the rollout ordinary decoding
        produces at the current weights: decode the prompt from scratch with the
        same seed and compare token for token (``spec/keyed_replay_*``).
        """
        data = BatchedDataDict(
            {
                "input_ids": prompt.view(1, -1),
                "input_lengths": torch.tensor([prompt.numel()]),
                "max_new_tokens": torch.tensor([self.max_new_tokens]),
                "noise_seed": torch.tensor([seed]),
            }
        )
        if "stop_strings" in datum:
            data["stop_strings"] = datum["stop_strings"]
        ref: list[int] = []
        async for _, res in self.base.generate_async(data):
            glen = int(res["generation_lengths"][0])
            c0 = prompt.numel()
            ref += res["output_ids"][0, c0 : c0 + glen].tolist()
        same = ref == gen
        self.stats["spec/keyed_replay_checked"] = (
            self.stats.get("spec/keyed_replay_checked", 0.0) + 1
        )
        self.stats["spec/keyed_replay_match"] = self.stats.get(
            "spec/keyed_replay_match", 0.0
        ) + float(same)
        if not same:
            first = next(
                (i for i, (x, y) in enumerate(zip(ref, gen)) if x != y),
                min(len(ref), len(gen)),
            )
            print(
                f"[keyed replay] MISMATCH seed={seed} first_diff={first} "
                f"len spec={len(gen)} plain={len(ref)}",
                flush=True,
            )

    async def _one(self, index: int, datum: BatchedDataDict):
        if any(datum.get("stop_strings") or []):
            # Drafts run without stop strings, so a plan could pass one.
            raise ValueError(
                "thundersync.speculate does not support per-sample stop_strings"
            )
        plen = int(datum["input_lengths"][0])
        prompt = datum["input_ids"][0, :plen]
        key = _key(prompt)
        cv = getattr(self, "_plan_cv", None)
        if cv is not None:
            # Chunked verification: wait until this trajectory's plan is published.
            async with cv:
                while not self.plans.get(key) and self._pending.get(key, 0) > 0:
                    if getattr(self, "_verify_error", None) is not None:
                        raise RuntimeError(
                            "draft verification failed"
                        ) from self._verify_error
                    await cv.wait()
        queue = self.plans.get(key)
        plan = (
            queue.pop(0)
            if queue
            else Plan(seed=next(self._seeds) if self.verify_mode == "keyed" else -1)
        )
        gen = list(plan.prefix)
        lps = list(plan.prefix_logprobs)
        if plan.fused is not None:
            self._fused_claims[key].append((tuple(gen), plan.fused))
        if self.sticky_delta is not None and plan.anchors is not None:
            if not plan.complete:
                # Its continuation would be decoded plainly by the refit
                # engine, which is not the sticky policy the prefix was
                # verified against.
                raise RuntimeError(
                    "sticky routing: a drafted rollout reached the rollout loop "
                    "incomplete (verify_sticky must complete every plan)"
                )
            self._anchor_claims[key].append((tuple(gen), plan.anchors))
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
            if self.tail_lookup:
                refs = self._lookup_references(key, plan)
                if refs:
                    cont["lookup_references"] = [refs]
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
                    f.write(
                        json.dumps(
                            {
                                "cont_step": self.current_step,
                                "prefix": len(plan.prefix),
                                "cont_tokens": len(gen) - len(plan.prefix),
                                "t_start": t0,
                                "secs": time.perf_counter() - t0,
                            }
                        )
                        + "\n"
                    )
        if (
            os.environ.get("THUNDERSYNC_KEYED_REPLAY")
            and self.verify_mode == "keyed"
            and plan.seed >= 0
            and plan.prefix
        ):
            await self._keyed_replay_check(prompt, plan.seed, gen, datum)
        n = plen + len(gen)
        out = BatchedDataDict(
            {
                "output_ids": torch.cat(
                    [prompt, torch.tensor(gen, dtype=torch.long)]
                ).view(1, -1),
                "logprobs": torch.cat(
                    [torch.zeros(plen), torch.tensor(lps, dtype=torch.float)]
                ).view(1, -1),
                "generation_lengths": torch.tensor([len(gen)]),
                "unpadded_sequence_lengths": torch.tensor([n]),
                "gen_leader_worker_idx": [0],
            }
        )
        return index, out

    def claim_fused(
        self, prompt: torch.Tensor, token_ids: torch.Tensor
    ) -> Optional[FusedRef]:
        """The retained verification forward a finished rollout can train from.

        ``prompt``: the rollout's generation prompt; ``token_ids``: its whole
        trajectory (prompt, generated tokens, environment reply). Returns (and
        removes) a ref whose verified row is a prefix of the trajectory, or
        None. Equal trajectories may share a ref: their forwards are the same.
        """
        entries = self._fused_claims.get(_key(prompt))
        if not entries:
            return None
        p = prompt.numel()
        for j, (gen, ref) in enumerate(entries):
            if (
                token_ids.numel() >= p + len(gen)
                and _key(token_ids[p : p + len(gen)]) == gen
            ):
                del entries[j]
                return ref
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)
