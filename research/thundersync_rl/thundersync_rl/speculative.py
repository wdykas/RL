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
import contextlib
import itertools
import math
import os
import time
from collections import Counter, defaultdict
import dataclasses
from dataclasses import dataclass, field, replace
from typing import Any, AsyncGenerator, Callable, Optional

import numpy as np
import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from thundersync_rl.engine_q import (
    HIDDEN_STATS_KEY,
    HiddenRefs,
    cat_hidden,
    free_hidden,
    hidden_nbytes,
    hidden_num_rows,
    ray_free,
)
from thundersync_rl.fused_train import FusedRef
from thundersync_rl.multi_turn import TurnEnvironment, TurnRecord, pair_rows, replay
from thundersync_rl.null_bv import NullGroup, NullPolicy, row_envelopes
from thundersync_rl.prefix_cache import PrefixCacheDirectory, lpt_owners
from thundersync_rl import vit
from thundersync_rl.tail_trace import TailTrace
from thundersync_rl.routes import (
    RouteRefs,
    anchor_mismatches,
    anchored_draft_length,
    compare_anchors,
    fetch_anchors,
    merge_route_refs,
    splice_refs,
)


# Per-draft transport results of engine-q verification (worker -> driver).
HIDDEN_FETCH_KEYS = ("hidden_bytes_fetched", "hidden_real_bytes_fetched", "hidden_fetch_s")


def _null_result(res: dict[str, Any], m: Optional[float]) -> dict[str, Any]:
    """A ⊥-BV row's result at envelope ``m`` from its grid (None: unused)."""
    base = {k: res.get(k, 0) for k in HIDDEN_FETCH_KEYS}
    if m is None:
        return {**base, "null": "unused", "accepted": 0, "next": -1, "logprobs": []}
    tau, y, lp_y = res["null_grid"][m]
    return {
        **base,
        "accepted": tau,
        "next": y,
        "logprobs": res["token_logprobs"][:tau] + ([] if y < 0 else [lp_y]),
        "draft_logq": res["draft_logq"][:tau],
        "null": "out" if y >= 0 else "bot",
        "null_m": m,
        "log_wstar": res.get("null_log_wstar", 0.0),
    }


def data_parallel_size(group) -> int:
    """Model data-parallel degree of a worker pool (the ranks rows are placed on).

    Not ``RayWorkerGroup.dp_size``, which counts tied-worker groups: a
    Megatron pool has one per worker, so tensor/pipeline-parallel ranks of
    the same replica would be counted as replicas.
    """
    ann = getattr(group, "sharding_annotations", None)
    return 1 if ann is None else int(ann.get_axis_size("data_parallel"))


class PrefixClaims:
    """Values keyed by (prompt, generated tokens), claimed by a trajectory.

    A trajectory claims (and removes) the first stored value of its prompt
    whose generated tokens are a prefix of what follows the prompt. ``claim``
    hashes every prefix of the trajectory at once (a random-weight linear
    hash, one vectorized pass) and checks each stored entry of the prompt
    against its prefix hash in O(1), confirming a hit by comparing tokens,
    instead of building a key of every entry's length (O(G) passes over up
    to 64k tokens per claim).
    """

    def __init__(self):
        self._by_prompt: dict[tuple[int, ...], list[tuple[int, int, np.ndarray, Any]]] = (
            defaultdict(list)
        )
        self._w = np.zeros(0, dtype=np.uint64)

    def _weights(self, n: int) -> np.ndarray:
        if n > self._w.shape[0]:
            rng = np.random.default_rng(self._w.shape[0] + 1)
            more = rng.integers(1, 2**63, size=max(n, 2 * self._w.shape[0]) - self._w.shape[0], dtype=np.uint64)
            self._w = np.concatenate([self._w, more])
        return self._w[:n]

    def _prefix_hashes(self, toks: np.ndarray) -> np.ndarray:
        """h[n] = hash of toks[:n] for n = 0..len (uint64, wraps)."""
        with np.errstate(over="ignore"):
            terms = self._weights(toks.shape[0]) * (toks.astype(np.uint64) + np.uint64(1))
            return np.concatenate([np.zeros(1, dtype=np.uint64), np.cumsum(terms, dtype=np.uint64)])

    def add(self, prompt_key: tuple[int, ...], gen: list[int], value: Any) -> None:
        toks = np.asarray(gen, dtype=np.int64)
        h = int(self._prefix_hashes(toks)[-1])
        self._by_prompt[prompt_key].append((toks.shape[0], h, toks, value))

    def claim(self, prompt: torch.Tensor, token_ids: torch.Tensor) -> Optional[Any]:
        entries = self._by_prompt.get(_key(prompt))
        if not entries:
            return None
        rest = token_ids[prompt.numel() :].cpu().numpy().astype(np.int64)
        pref = self._prefix_hashes(rest)
        for j, (n, h, toks, value) in enumerate(entries):
            if n <= rest.shape[0] and int(pref[n]) == h and np.array_equal(rest[:n], toks):
                del entries[j]
                return value
        return None

    def clear(self) -> None:
        self._by_prompt.clear()


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
    # (object-store handles, ``RouteRefs``); the training forward of this
    # rollout replays them (routes.py).
    anchors: Optional[RouteRefs] = field(default=None, compare=False)
    # Sticky MoE routing: the engine stream that drafted this whole-accepted,
    # unfinished prefix and still decodes it (``_continue_sticky`` goes on
    # with it instead of starting a new one).
    stream: Optional[dict[str, Any]] = field(default=None, compare=False)
    # (drafted tokens, accepted tokens) of the draft verification turned into
    # this plan; None for plans not made from a draft (``TailTrace``).
    verified: Optional[tuple[int, int]] = field(default=None, compare=False)
    # Needed a continuation after verification (a tail row: rejected inside
    # its draft, or a whole-accepted draft cut before EOS / the cap).
    tail: bool = field(default=False, compare=False)
    # Learner prefix-state cache: the key the learner caches this rollout's
    # verification state under (continuation pieces; DESIGN 14p).
    prefix_key: Optional[str] = field(default=None, compare=False)
    # Log of the row's importance (loss) weight, 0 for exact samples of p.
    # Opt-in tools only: drop-IS (log max(1, W*/M), needs null_bv) or banded
    # IS (log w at the band exit, excludes null_bv).
    log_weight: float = field(default=0.0, compare=False)
    # Anchor horizon: the key of this plan's anchors on the generation workers
    # (``register_engine_anchors``); the refit engine continues the rollout
    # applying them below the horizon.
    engine_key: Optional[str] = None
    # Multi-turn rollouts under sticky routing (``multi_turn.py``): ``prefix``
    # then holds the whole row after the prompt (turns and observations), the
    # current turn starts at ``prefix[turn_start]``, ``turns`` records every
    # finished turn with the environment's answer, ``row`` is the rollout-loop
    # row the plan serves.
    turn_start: int = 0
    turns: list[TurnRecord] = field(default_factory=list, compare=False)
    row: Optional[int] = None
    # Verify-in-train (``vit.py``): a placeholder of a VIT group, filled once its
    # training forward decides (or its fallback verification does).
    vit: bool = field(default=False, compare=False)


@dataclass(frozen=True)
class RowInfo:
    """Verification facts of one rollout the trainer may use (``claim_row``)."""

    tail: bool
    log_weight: float


@dataclass
class CohortDrafts:
    """Drafts advanced during one iteration (their tokens were drafted at its theta)."""

    live: list[dict[str, Any]]


def _key(ids: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(t) for t in ids.tolist())


def _drafted(res) -> tuple[list[int], bool, list[float] | None]:
    """(tokens, finished, engine log-probs or None) of one ``collect_drafts`` entry."""
    toks, done, *rest = res
    # Token ids arrive as one int array (typed transport); the plan keeps ints.
    toks = toks.tolist() if isinstance(toks, np.ndarray) else list(toks)
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
        # Python floats (engine log q may arrive as a float32 array): compare
        # and correct in double, as with the former per-token float lists.
        eq, dq = float(engine_logq[j]), float(draft_logq[j])
        if eq == dq:
            # q is the engine's law (engine-sourced verification): no
            # correction, and log p is reported bitwise.
            continue
        v = logp[j] + eq - dq
        if math.isfinite(v):
            out[j] = v
    return out


# Ordered learner call numbers (``SpeculativeGeneration.learner_seq``), shared by
# every SpeculativeGeneration of the driver (one per engine lane).
_LEARNER_SEQ = itertools.count()


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
        null_bv_drop_is: bool = False,
        null_bv_spare_prompts: int = 0,
        sticky_chunk_tokens: int = 2048,
        sticky_chunk_growth: float = 0.0,
        draft_until_complete: bool = False,
        draft_weights: str = "current",
        max_seq_len: Optional[int] = None,
        anchor_horizon: Optional[int] = None,
        banded_is_band: Optional[float] = None,
        learner_prefix_cache: Optional[dict[str, float]] = None,
    ):
        self.base = base
        self.learner_policy = learner_policy
        self.vocab_limit = vocab_limit
        self.head_k = head_k
        self.draft_budget = draft_budget
        self.max_new_tokens = max_new_tokens
        # Total-length cap of a rollout (prompt + generated), as the engine
        # enforces it (``max_total_sequence_length``); None: none.
        self.max_seq_len = max_seq_len
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
        # consumed them (prompt, generated tokens -> ref).
        self._fused_claims = PrefixClaims()
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
        # Weights the engine drafts with (DESIGN.md 14f): "current" = theta_k;
        # "momentum" = theta_hat_{k+1}, staged by stage_draft_weights and loaded
        # by load_staged_draft_weights before the drafts start. Only q changes; engine-sourced q keeps verification exact.
        if draft_weights not in ("current", "momentum"):
            raise ValueError(f"unknown draft_weights {draft_weights!r}")
        if draft_weights != "current" and (
            verify_mode != "block" or verify_source != "engine" or verify_on != "learner"
        ):
            raise ValueError(
                "draft_weights=momentum needs verify_mode=block, verify_source=engine "
                "and verify_on=learner (q is the engine's law, its LM head stashed on "
                "the learner while the draft weights are loaded)"
            )
        self.draft_weights = draft_weights
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
                workers.run_all_workers_single_data(
                    "install_draft_hidden_log",
                    seal_blocks=hidden_transport == "object_store",
                )
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
        # Continuations are verified every this many decoded tokens while the
        # engine keeps decoding them (``_continue_sticky``).
        self.sticky_chunk_tokens = sticky_chunk_tokens
        # Piece size grows with the row's generated position (``_piece_tokens``).
        self.sticky_chunk_growth = float(sticky_chunk_growth)
        # Learner prefix-state cache (DESIGN 14p): continuation pieces are
        # forwarded from the row's cached state on the learner rank that holds
        # it ({"gpu_gb", "max_gb"} per learner rank; None: off).
        self.prefix_cache = (
            None
            if learner_prefix_cache is None
            else PrefixCacheDirectory(**learner_prefix_cache)
        )
        self._prefix_step: Optional[int] = None
        # Measurement only: the deadline waits for every draft to finish.
        self.draft_until_complete = draft_until_complete
        self._cont_ids = itertools.count()
        self._anchor_claims = PrefixClaims()
        self._row_claims = PrefixClaims()
        # Tail rows per prompt key of the current step's plans (``tail_count``).
        self._tail_counts: dict[tuple[int, ...], int] = {}
        # ⊥-BV prompt drop: prompt keys whose pools were dropped this step.
        self.null_dropped_keys: set[tuple[int, ...]] = set()
        # Anchor horizon H (sticky routing; None: anchors up to the cap).
        # Positions >= c = P + min(H, cap) are self: theta_{k-1} work per
        # drafted rollout ends once its anchors below c are logged, and the
        # refit engine continues it, applying the anchors (``_horizon``).
        self.anchor_horizon = anchor_horizon
        self._anchor_keys = itertools.count()
        # Multi-turn rollouts (sticky routing, scheme A): the environment the
        # driver steps at every verified turn end, and the drafted plans
        # paired with rollout-loop rows (``bind_rows``/``claim_row_plan``).
        self.turn_env: Optional[TurnEnvironment] = None
        self._row_prompts: list[torch.Tensor] = []
        self._row_plans: dict[int, Plan] = {}
        # Route blocks by the step whose training replays them (freed by
        # ``release_routes`` once that step has trained).
        self._route_blocks: dict[int, list] = defaultdict(list)
        self._route_layers: list[int] = []
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
            # Logging ranks return their MoE layer numbers (pipeline stages:
            # their own layers); the union is every MoE layer of the model.
            self._route_layers = sorted(
                {int(ln) for x in ok if isinstance(x, list) for ln in x}
            )
            if not self._route_layers or any(x is False for x in ok):
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
        # Built once: its DP table is memoised across steps.
        self._null_policy = NullPolicy(**self.null_bv_policy) if null_bv else None
        # Drop-IS (opt-in, unbiased gradient, not exact samples): outputs that
        # need a tail decode are discarded like BOT; kept rows are weighted.
        self.null_bv_drop_is = null_bv_drop_is
        # Prompt drop: the step trains on the first (prompts - spares) prompts
        # whose pools were not dropped (``null_selected_keys``, drafting order).
        self.null_bv_spare_prompts = null_bv_spare_prompts
        self.null_selected_keys: Optional[list[tuple[int, ...]]] = None
        self._null_slots: dict[int, dict[tuple[int, ...], int]] = {}
        if null_bv and (self.verify_source != "engine" or verify_mode != "block"):
            raise ValueError(
                "null_bv needs verify_mode='block' with verify_source='engine' "
                f"(got {verify_mode!r}/{self.verify_source!r}; the scorer path is "
                "not built yet)"
            )
        # Banded importance sampling (block_verification.py): drafts are
        # verified against q inside the band and p after it; each rollout then
        # carries a loss weight w (``claim_log_weight``). Only a draft's first
        # verification is banded: continuations (sticky rounds) are plain BV,
        # exact samplers of p as the weight requires.
        self.banded_is_band = banded_is_band
        self._weight_claims = PrefixClaims()
        self._band_rows: list[tuple[float, bool]] = []
        if banded_is_band is not None and (
            self.verify_source != "engine" or verify_mode != "block"
        ):
            raise ValueError(
                "banded_is_band needs verify_mode='block' with verify_source='engine' "
                f"(got {verify_mode!r}/{self.verify_source!r})"
            )
        self._seeds = itertools.count(1)
        self._verify_seeds = itertools.count(1)
        # Verify-in-train (``vit.py``, opt-in; set by the training loop).
        self.verify_in_train = False
        self.vit_group_size = 0
        self._vit_groups: list = []
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
        if self.sticky_delta is not None:
            # Drafts are not aborted at the deadline (``_handout_running``):
            # their streams go on toward the cap as continuations.
            budget = self.max_new_tokens
        step = self.current_step
        if null_slots is not None:
            for t in [t for t in self._null_slots if t <= step]:
                del self._null_slots[t]  # never verified (no drafts reached it)
            self._null_slots[step + 1] = dict(null_slots)
        hkeys = self._hidden_keys([f"{step + 1}:{i}" for i in range(len(prompts))])
        # ⊥-BV: the primaries (one per output slot) come first and must all
        # run at once; the extras may queue behind them.
        self._check_draft_capacity(
            rank0, sum(null_slots.values()) if null_slots else len(prompts)
        )
        t_start = time.perf_counter()
        ray.get(
            rank0.start_drafts.remote(
                [p.tolist() for p in prompts],
                budget,
                self.stream_interval,
                **({} if hkeys is None else {"hidden_keys": hkeys}),
            )
        )

        def collect(defer: bool = False):
            """The drafts at this deadline.

            ``defer`` (sticky routing): take the cut (each draft's length)
            now and return a callable that hands the drafts out (routes,
            hidden rows) later, e.g. while another lane refits: the cut is
            then a single snapshot call.
            """
            plens = [p.numel() for p in prompts]
            if self.sticky_delta is not None:
                t_cut = time.perf_counter()
                if self.draft_until_complete:
                    idx = list(range(len(prompts)))
                    while not all(d for _, d in ray.get(rank0.draft_lengths.remote(idx))):
                        time.sleep(0.05)
                self.stats["tail/next_draft_complete_wait_s"] = time.perf_counter() - t_cut
                self.stats["tail/next_draft_window_s"] = time.perf_counter() - t_start
                t0 = time.perf_counter()
                peeked = [_drafted(x) for x in self._peek_streams(rank0, list(range(len(prompts))))]
                self.stats["cut/peek_s"] = time.perf_counter() - t0
                handout = lambda: self._handout_running(  # noqa: E731
                    peeked, hkeys, plens, budget, target=step + 1
                )
                return (lambda: build(*handout())) if defer else build(*handout())
            res = [_drafted(x) for x in self._collect_drafts(rank0)]
            res, hid, anchors = self._pop_logs(hkeys, plens, res, target=step + 1)
            return build(res, hid, anchors, [None] * len(prompts))

        def build(res, hid, anchors, streams):
            plens = [p.numel() for p in prompts]
            if self.sticky_delta is not None:
                self.stats["tail/next_drafted_tokens"] = float(sum(len(x[0]) for x in res))
                if os.environ.get("THUNDERSYNC_DEBUG_DRAFT_LENS"):
                    # Debug: drafted length per draft, in drafting (= admission) order.
                    print(f"[draft lens] target={step + 1} {[len(x[0]) for x in res]}", flush=True)
                self.stats["tail/next_drafts"] = float(len(res))
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
                        *(() if self.sticky_delta is None else (anchors[i], streams[i])),
                    )
                ]
                for i, (toks, done, lps) in enumerate(res)
            ]

        collect.deferrable = self.sticky_delta is not None
        return collect

    def _check_draft_capacity(self, rank0, n: int) -> None:
        """Fail fast when the engine cannot run a cohort's ``n`` primary drafts at once.

        Drafts beyond the engine's concurrent request capacity start only when
        earlier ones finish (a whole draft's decode time later), so they are
        short or empty at the cut and their rows decode as long tails or
        fresh. ⊥-BV extras are submitted after the primaries and may queue. Raise the capacity with
        ``policy.generation.mcore_generation_config.max_requests`` (hybrid
        models: the Mamba region is then sized for that many requests, per
        engine rank max_requests x Mamba state bytes per request) or fewer
        drafts per cohort.
        """
        cap = self.__dict__.get("_draft_capacity")
        if cap is None:
            query = getattr(rank0, "engine_request_capacity", None)  # absent on test stubs
            cap = self._draft_capacity = int(ray.get(query.remote())) if query else 0
        if cap and n > cap:
            raise RuntimeError(
                f"{n} primary drafts per cohort but the engine runs {cap} requests at once "
                "(max_requests x engine DP): set "
                "policy.generation.mcore_generation_config.max_requests >= "
                f"{n} / engine DP, or draft fewer rows per cohort"
            )

    def _peek_streams(self, rank0, idx: list[int]) -> list:
        """(tokens, finished, engine log-probs) of running drafts, left running.

        Unfinished drafts read the engines' request records (as
        ``_collect_drafts``: streamed frames skip tokens after an eviction).
        """
        t0 = time.perf_counter()
        ids = ray.get(rank0.draft_request_ids.remote(idx))
        refs, rids = self._record_refs(ids)
        out = ray.get(rank0.peek_drafts.remote(idx, record_refs=refs, request_ids=rids))
        # Driver seconds spent fetching draft snapshots (typed transport).
        self.stats["driver/peek_s"] = self.stats.get("driver/peek_s", 0.0) + time.perf_counter() - t0
        return out

    def _record_refs(self, ids: dict[int, int]) -> tuple[list, list[int]]:
        """The other engine ranks' ``draft_records`` as object refs (for rank 0).

        The records go to rank 0, which reads its own directly, without passing
        through the driver; rank 0 is not sent a ``draft_records`` call (its
        ``peek_drafts`` waits on these refs while it runs, so a call queued
        behind it on rank 0 would never start).
        """
        if not ids:
            return [], []
        rids = sorted(ids.values())
        others = self.base._policy.worker_group.workers[1:]
        return [w.draft_records.remote(request_ids=rids) for w in others], rids

    def _handout_running(self, res, hkeys, prompt_lens, budget: int, target: int):
        """Sticky routing: hand out the drafts cut at ``res`` (``_peek_streams``), streams left running.

        Each draft is cut to its logged routes and hidden rows; its stream
        (``_continue_sticky``) keeps decoding at theta_{k-1}, which the
        engine holds until every drafted rollout is complete, so a
        whole-accepted draft continues without a new prefill. Returns
        (drafts, hidden, anchors, stream handles).
        """
        n = len(prompt_lens)
        t1 = time.perf_counter()
        routes = self._pop_routes(
            hkeys, prompt_lens, [budget] * n, target=target, close=False
        )
        t2 = time.perf_counter()
        cut = [
            self._truncate_to_anchors(x, a, plen)
            for x, a, plen in zip(res, routes, prompt_lens)
        ]
        hid = self._take_hidden(hkeys, [0] * n, [len(x[0]) for x in cut])
        t3 = time.perf_counter()
        # Where the deadline cut goes (driver wall seconds).
        self.stats["cut/routes_s"] = t2 - t1
        self.stats["cut/hidden_s"] = t3 - t2
        res = [self._truncate(x, h) for x, h in zip(cut, hid)]
        anchors, streams = [], []
        for i, ((toks, _, _), a, plen) in enumerate(zip(res, routes, prompt_lens)):
            anchors.append(
                None if a is None else self._cut(a.truncate(plen + len(toks)), plen)
            )
            streams.append(
                dict(idx=i, hkey=hkeys[i], length=plen + budget, taken=len(toks), routes=a)
            )
        return res, hid, anchors, streams

    def _collect_drafts(self, rank0) -> list:
        """Snapshot and abort the drafts, content from the engines' request records.

        The engines (one per generation data-parallel rank) are asked for the
        record contents of the unfinished draft requests they hold; rank 0, which
        holds the streams, then aborts them and uses those contents instead of
        its streamed frames (which skip tokens after an eviction). Tokens decoded
        between the two calls are discarded: the cut is a time, not a function
        of the tokens, and any prefix of a draft is a valid draft.
        """
        t0 = time.perf_counter()
        ids = ray.get(rank0.draft_request_ids.remote())
        refs, rids = self._record_refs(ids)
        out = ray.get(rank0.collect_drafts.remote(record_refs=refs, request_ids=rids))
        self.stats["driver/collect_s"] = self.stats.get("driver/collect_s", 0.0) + time.perf_counter() - t0
        return out

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
        t_call, sts = time.perf_counter(), []
        for r in ray.get(
            self.base._policy.worker_group.run_all_workers_single_data(
                "pop_draft_hidden",
                keys=hkeys,
                lengths=lengths,
                **({"as_refs": True} if refs else {}),
            )
        ):
            sts.append(self._hidden_call_stats(r))
            merged.update(r)
        self._count_hidden("spec/hidden_handout_wall_s", time.perf_counter() - t_call)
        for name, k in (("spec/hidden_put_s", "call_s"), ("spec/hidden_seal_put_s", "seal_put_s")):
            self._count_hidden(name, max([st.get(k, 0.0) for st in sts] or [0.0]))
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

    def _take_hidden(
        self, hkeys: list[str], starts: list[int], lengths: list[int]
    ) -> list[Any]:
        """Hidden rows start..start+m-1 of running drafts (``_pop_hidden`` for pieces)."""
        refs = self.hidden_transport == "object_store"
        merged: dict[str, Any] = {}
        t_call, sts = time.perf_counter(), []
        for r in ray.get(
            self.base._policy.worker_group.run_all_workers_single_data(
                "take_draft_hidden",
                keys=hkeys,
                starts=starts,
                lengths=lengths,
                **({"as_refs": True} if refs else {}),
            )
        ):
            sts.append(self._hidden_call_stats(r))
            merged.update(r)
        self._count_hidden("spec/hidden_handout_wall_s", time.perf_counter() - t_call)
        for name, k in (("spec/hidden_put_s", "call_s"), ("spec/hidden_seal_put_s", "seal_put_s")):
            self._count_hidden(name, max([st.get(k, 0.0) for st in sts] or [0.0]))
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
        self,
        hkeys: list[str] | None,
        prompt_lens: list[int],
        lengths: list[int],
        target: int,
        close: bool = True,
        bases: Optional[list[Optional[RouteRefs]]] = None,
    ) -> list[Optional[RouteRefs]]:
        """Each draft's anchors over [P + g] positions (sticky routing), None if unlogged.

        Every engine rank that routed the draft's tokens contributes its part
        (pipeline stages: their layers) as object-store blocks; the driver
        keeps only the handles (``RouteRefs``). Positions past the logged
        prefix truncate the draft (``_truncate_to_anchors``). ``target``: the
        step whose training replays these anchors (``release_routes`` frees
        them after it). ``close=False`` with ``bases``: drafts still running,
        whose routes taken earlier (``bases[i]``) the new blocks extend.
        """
        n = len(lengths)
        if self.sticky_delta is None or hkeys is None:
            return [None] * n
        bases = bases or [None] * n
        parts: dict[str, list] = defaultdict(list)
        new_bytes = 0
        for r in ray.get(
            self.base._policy.worker_group.run_all_workers_single_data(
                "pop_draft_routes", keys=hkeys, close=close
            )
        ):
            for k, v in r.items():
                parts[k].append(v)
                self._route_blocks[target] += [b[0] for b in v[3]]
                new_bytes += sum(b[3] for b in v[3])
        out: list[Optional[RouteRefs]] = []
        for k, plen, g, base in zip(hkeys, prompt_lens, lengths, bases):
            if k not in parts:
                out.append(None if base is None else replace(base, length=plen + g))
                continue
            out.append(
                merge_route_refs(parts[k], plen + g, self._route_layers, base=base)
            )
        self.stats["spec/route_bytes_logged"] = (
            self.stats.get("spec/route_bytes_logged", 0.0) + new_bytes
        )
        return out

    def release_routes(self, step: int) -> None:
        """Free the route blocks of every step <= ``step`` (trained by now)."""
        refs = []
        for s in [s for s in self._route_blocks if s <= step]:
            refs += self._route_blocks.pop(s)
        self._anchor_claims.clear()
        if refs:
            (self._free_refs or ray_free)(refs)

    def _pop_logs(
        self, hkeys, prompt_lens: list[int], res: list, target: int, prior=None
    ):
        """A drafting round's engine logs: (drafts, hidden states, anchors).

        Routes first: each draft is cut to its anchored length
        (``_truncate_to_anchors``; ``prior``: earlier anchors of the same
        rows, spliced under the new log). Hidden states are then popped for
        the kept tokens only and the draft cut to the logged rows
        (``_truncate``), so its hidden rows always match its token count, as
        verification requires.
        """
        anchors = self._pop_routes(
            hkeys, prompt_lens, [len(x[0]) for x in res], target
        )
        if prior is not None:
            anchors = [
                a if a is None else splice_refs(b, a) for a, b in zip(anchors, prior)
            ]
        res = [
            self._truncate_to_anchors(x, a, plen)
            for x, a, plen in zip(res, anchors, prompt_lens)
        ]
        anchors = [self._cut(a, plen) for a, plen in zip(anchors, prompt_lens)]
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
        m = anchored_draft_length(
            anchors.complete, prompt_len, len(toks), self._horizon(prompt_len)
        )
        if m >= len(toks):
            return drafted
        for name, v in (
            ("spec/route_truncated_drafts", 1.0),
            ("spec/route_truncated_tokens", float(len(toks) - m)),
        ):
            self.stats[name] = self.stats.get(name, 0.0) + v
        return toks[:m], False, (None if lps is None else lps[:m])

    def _hidden_call_stats(self, r: dict) -> dict:
        """Fold one worker's ``HIDDEN_STATS_KEY`` entry into the step stats.

        Times are per worker (they run in parallel): the step stat is the sum
        over calls of the slowest worker's time.
        """
        st = r.pop(HIDDEN_STATS_KEY, None)
        if not st:
            return {}
        self._count_hidden("spec/hidden_put_real_bytes", st.get("put_bytes", 0))
        self._count_hidden("spec/hidden_sealed_bytes", st.get("sealed_bytes", 0))
        self._count_hidden("spec/hidden_resplits", st.get("resplits", 0))
        return st

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
                    d["logprobs"].extend(np.asarray(lps).tolist())
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
        With momentum draft weights the drafting head was stashed by
        ``stage_draft_weights`` (theta_k is not what drafted).
        """
        if self.verify_mode != "block" or drafts is None:
            return
        if self.draft_weights != "current":
            return
        self._stash_drafting_version()

    def _stash_drafting_version(self, keep_previous: bool = False) -> None:
        """Stash the verifiers' live weights as version ``current_step``.

        ``keep_previous``: also keep version ``current_step - 1`` (its drafts
        are not verified yet).
        """
        keep = {self.current_step}
        if keep_previous:
            keep.add(self.current_step - 1)
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

    def plans_cover(self, prompts: list[torch.Tensor], pending_ok: bool = False) -> bool:
        """Whether every rollout of ``prompts`` will be served by a complete plan.

        Then no rollout of this step decodes on the engine (``_one`` returns the
        plan), so the engine need not hold theta_k for the step.
        ``pending_ok``: plans the continuation rounds are still completing
        count (``begin_sticky`` before ``complete_tails``). Not with an anchor
        horizon: a continued plan may stop at the horizon and then needs the
        refit engine at theta_k.
        """
        if self._row_plans:
            # Multi-turn: drafted plans are paired with rows (``bind_rows``).
            return len(self._row_plans) == len(prompts) and all(
                plan.complete for plan in self._row_plans.values()
            )
        need = Counter(_key(p) for p in prompts)
        return all(
            all(
                plan.complete
                # Plans the continuation rounds are completing (no horizon).
                or (
                    pending_ok
                    and self.anchor_horizon is None
                    and id(plan) in self.__dict__.get("_tail_plans", ())
                )
                for plan in self.plans.get(key, [])[:n]
            )
            and len(self.plans.get(key, [])) >= n
            for key, n in need.items()
        )

    def stage_draft_weights(
        self, refit: Callable[[], Any], stage_on_engine: bool = True
    ) -> None:
        """Stage the draft weights of this step's drafts on the engine.

        ``draft_weights=momentum``: theta_hat_{k+1} needs only the optimizer
        state after step k-1's update (theta_k, m_k, v_k), so it is staged at
        the top of step k, while the learner and the engine are idle between
        the previous step's verification and this step's refit: the learner
        writes theta_hat_{k+1} into its parameters (owned main shards, chunked),
        the LM head of those weights is stashed as drafting version
        ``current_step`` (q rows at verification come from it and the engine's
        hidden states), ``refit`` streams the parameters to the engine through
        the normal refit path, every engine rank copies its live weights into
        its host staging buffer while the learner restores theta_k bitwise.
        The caller then refits theta_k for this step's rollouts;
        ``load_staged_draft_weights`` puts the staged weights back once those
        rollouts are done, without the learner. ``stage_on_engine=False`` (no
        rollout of this step decodes on the engine, ``plans_cover``): no host
        copy and no theta_k refit; the engine drafts with the refit weights
        right away. The engine keeps them until the next refit: drafting, the
        route check and the sticky continuation rounds all run on them.
        """
        group = self.learner_policy.worker_group
        engine = self.base._policy.worker_group
        t0 = time.perf_counter()
        loads = ray.get(
            group.run_all_workers_single_data("load_draft_weights", kind=self.draft_weights)
        )
        stage_refs = None
        try:
            t1 = time.perf_counter()
            # Drafts of version k-1 are still verified during this step's
            # rollouts (chunked verification): keep their head.
            self._stash_drafting_version(keep_previous=True)
            t2 = time.perf_counter()
            refit()
            t3 = time.perf_counter()
            if stage_on_engine:
                stage_refs = engine.run_all_workers_single_data("stage_engine_weights")
        finally:
            # The learner's restore overlaps the engine's device -> host copy.
            restores = ray.get(group.run_all_workers_single_data("restore_training_weights"))
            stages = ray.get(stage_refs) if stage_refs is not None else []
        t4 = time.perf_counter()
        self._staged_version = self.current_step if stage_on_engine else None
        self._staged_checksums = [x.get("checksum") for x in stages]
        changed = sum(x["draft/changed_elems"] for x in loads)
        total = sum(x["draft/shard_elems"] for x in loads)
        stats = {
            "draft/load_s": t1 - t0,
            "draft/load_compute_s": max(x["draft/compute_s"] for x in loads),
            "draft/load_gather_s": max(x["draft/gather_s"] for x in loads),
            "draft/stash_s": t2 - t1,
            "draft/refit_s": t3 - t2,
            "draft/stage_restore_s": t4 - t3,
            "draft/engine_stage_s": max((x["stage_s"] for x in stages), default=0.0),
            "draft/engine_stage_gb": max((x["stage_bytes"] for x in stages), default=0.0)
            / 2**30,
            "draft/direct": float(not stage_on_engine),
            "draft/restore_worker_s": max(x["draft/restore_s"] for x in restores),
            "draft/total_s": t4 - t0,
            "draft/peak_extra_gb": max(x["draft/peak_extra_bytes"] for x in loads) / 2**30,
            "draft/changed_frac": changed / max(total, 1),
        }
        self.stats.update(stats)
        print(
            f"[draft weights] staged {self.draft_weights} step={self.current_step} "
            f"changed={int(changed)}/{int(total)} "
            + " ".join(f"{k}={v:.4g}" for k, v in stats.items()),
            flush=True,
        )

    def load_staged_draft_weights(self) -> None:
        """Engine: replace the live weights by the staged draft weights.

        Called once this step's rollouts (sampled at theta_k) are done; each
        engine rank copies its own host staging buffer into its parameters
        (no learner, no collective). With ``THUNDERSYNC_MISMATCH_CHECK`` the
        loaded weights' checksum must equal the staged one.
        """
        if getattr(self, "_staged_version", None) != self.current_step:
            raise RuntimeError(
                f"no draft weights staged for step {self.current_step} "
                f"(staged: {getattr(self, '_staged_version', None)})"
            )
        t0 = time.perf_counter()
        loads = ray.get(
            self.base._policy.worker_group.run_all_workers_single_data(
                "load_engine_weights"
            )
        )
        sums = [x.get("checksum") for x in loads]
        if sums != self._staged_checksums:
            raise RuntimeError(
                f"draft weights: engine checksums after load {sums} != staged "
                f"{self._staged_checksums}"
            )
        self._staged_version = None
        self.stats["draft/engine_load_s"] = time.perf_counter() - t0
        self.stats["draft/engine_load_worker_s"] = max(x["load_s"] for x in loads)

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
            for k in HIDDEN_FETCH_KEYS:
                self._count_hidden(
                    "spec/" + k, sum(r.get(k, 0) for r in by_row.values())
                )
        self._release_hidden(flat[a + j][1] for j in by_row)

    def verify(self, drafts: list[list[tuple[torch.Tensor, int, list[int]]]]) -> None:
        if self.fused_verify_train:
            raise NotImplementedError(
                "fused_verify_train runs through chunked verification only"
            )
        drafts, dropped = self._engine_q_filter(drafts)
        self.null_dropped_keys, self._null_dropped = set(), set()
        self.null_selected_keys = None
        null = self._null_prepare(drafts)
        self._vit_groups = []
        if null is not None:
            drafts, names, slots, n_empty = null
            if self.verify_in_train:
                drafts = self._vit_split(drafts, names, slots)
                vit_names = {names[vg.key] for vg in self._vit_groups}
                slots = {n: v for n, v in slots.items() if n not in vit_names}
            flat, rows = self._flat_rows(drafts)
            self._record_siblings(flat)
            groups = [names[_key(d[0])] for _, d in flat]
            by_row = self._verify_null(flat, rows, groups, slots) if flat else {}
            self._finish_verify_null(flat, by_row, groups, slots, dropped, n_empty)
            self._vit_place()
            self._count_tails()
            return
        # verify_in_train needs pooled ⊥-BV drafts (one pool per prompt); without
        # pools this step every draft is verified here (validated: null_bv).
        flat, rows = self._flat_rows(drafts)
        self._record_siblings(flat)
        if not flat:
            by_row = {}
        elif self.verify_mode == "block":
            by_row = self._verify_block(flat, rows, band=self.banded_is_band)
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
        self._vit_place()
        self._count_tails()

    # -- verify-in-train (``vit.py``) -------------------------------------------

    def _vit_split(self, drafts, names, slots):
        """Take the pools whose first G drafts are all finished out of verify0.

        A pool qualifies when it serves exactly G rows (one prompt), its first G
        drafts (drafting order) are non-empty finished rollouts if whole-accepted
        (``_stopped``) with engine log q and anchors. Which pools qualify is a
        timing/length fact; the outcome of each pool is the regular
        ``decide_pools`` outcome either way (``vit.py``).
        """
        if names is None:
            raise RuntimeError("verify_in_train needs pooled ⊥-BV drafts")
        G = int(self.vit_group_size)
        base = next(self._verify_seeds)
        keep = []
        for v in drafts:
            key = _key(v[0][0])
            n_slots = int(slots[names[key]]) if names is not None else len(v)
            head = v[:G]
            ok = (
                G > 0
                and n_slots == G
                and len(v) >= G
                and all(
                    len(d[2]) > 0
                    and self._stopped(d)
                    and len(d) > 8
                    and d[6] is not None
                    and d[8] is not None
                    for d in head
                )
            )
            if not ok:
                keep.append(v)
                continue
            n = len(v)
            ms = [
                float(self._null_policy.m(n - j, G - j)) if names is not None else 1.0
                for j in range(G)
            ]
            self._vit_groups.append(
                vit.VitGroup(
                    key=key,
                    drafts=list(v),
                    ms=ms,
                    seeds=[vit.row_seed(base, d[4]) for d in v],
                )
            )
        self._vit_seed_base = base
        self._vit_null = names is not None
        self.stats["vit/candidate_groups"] = float(len(self._vit_groups))
        self.stats["vit/candidate_rows"] = float(G * len(self._vit_groups))
        return keep

    def _vit_place(self) -> None:
        """Placeholder plans the VIT groups' rollouts wait on (``_await_plan``)."""
        for vg in self._vit_groups:
            vg.placeholders = [Plan(seed=-1, vit=True) for _ in range(self.vit_group_size)]
            self.plans[vg.key] = list(vg.placeholders)

    def take_vit_groups(self) -> list:
        """This step's VIT groups (once); creates the step's completion event."""
        groups, self._vit_groups = self._vit_groups, []
        self._vit_done = asyncio.Event()
        self._vit_tail_pending = []
        self._vit_error = None
        if not groups:
            self._vit_done.set()
        return groups

    async def _vit_fill(self, ph: Plan, plan: Plan) -> None:
        for f in dataclasses.fields(Plan):
            if f.name != "vit":
                setattr(ph, f.name, getattr(plan, f.name))

    async def vit_pass(self, vg, lpx_by_row: dict[int, list[float]]) -> None:
        """The group's training forward accepted its first G drafts whole: serve them."""
        G = self.vit_group_size
        for j in range(G):
            d = vg.drafts[j]
            g = len(d[2])
            lpx = lpx_by_row[j]
            if len(lpx) != g:
                raise RuntimeError(f"verify_in_train: {len(lpx)} log-probs for a {g}-token draft")
            r = {
                "accepted": g,
                "next": -1,
                "logprobs": list(lpx),
                "draft_logq": [float(x) for x in d[6][:g]],
            }
            plan = self._plan(d, r)
            if not plan.complete:
                raise RuntimeError("verify_in_train: an accepted candidate is not a finished rollout")
            plan.verified = (g, g)
            plan.tail = False
            await self._vit_fill(vg.placeholders[j], plan)
        self._release_hidden(vg.drafts)
        vg.passed = True
        await self._notify_tails()

    async def vit_fallback(self, groups: list) -> list:
        """Regular verification of failed VIT pools, with the rows' own streams.

        Returns the pending (tail) plans; fills every placeholder.
        """
        if not groups:
            return []
        G = self.vit_group_size
        flat, rows, envs = [], [], []
        for t, vg in enumerate(groups):
            n = len(vg.drafts)
            for j, d in enumerate(vg.drafts):
                flat.append((t, d))
                rows.append(torch.cat([d[0], torch.tensor(d[2], dtype=torch.long)]))
                if self._vit_null:
                    envs.append(row_envelopes(self._null_policy, n, G, j))
        futs = self._submit_verify_block(
            flat,
            rows,
            null_envelopes=envs if self._vit_null else None,
            ordered=True,
            row_seed_base=self._vit_seed_base,
        )
        done = []
        for a, refs in futs:
            done.append((a, await asyncio.to_thread(ray.get, refs)))
        if getattr(self, "learner_ordered", False):
            self._learner_inflight.difference_update(r for _, refs in futs for r in refs)
        by_row = self._verify_block_results(flat, done)
        pending = []
        at = 0
        for vg in groups:
            n = len(vg.drafts)
            plans = []
            state = NullGroup(n, G, self._null_policy) if self._vit_null else None
            for j, d in enumerate(vg.drafts):
                r = by_row[at + j]
                if state is not None:
                    m = state.next_m()
                    if m is None:
                        continue
                    tau, y, _ = r["null_grid"][m]
                    state.record(y >= 0)
                    res = _null_result(r, m)
                    if res["null"] == "bot":
                        continue
                else:
                    if j >= G:
                        break
                    res = r
                plan = self._plan(d, res)
                plan.verified = (len(d[2]), res["accepted"])
                plan.stream = None  # its draft stream was stopped at begin_sticky
                plan.tail = not plan.complete
                plans.append(plan)
            at += n
            if len(plans) != G:
                raise RuntimeError(
                    f"verify_in_train: fallback pool yielded {len(plans)} outputs for {G} rows "
                    "(a fresh decode would need theta_k on the engine)"
                )
            for ph, plan in zip(vg.placeholders, plans):
                await self._vit_fill(ph, plan)
                if not ph.complete:
                    pending.append((vg.key, ph))
            vg.passed = False
        self._release_hidden(d for vg in groups for d in vg.drafts)
        await self._notify_tails()
        return pending

    def vit_finish(self, pending: list, error: Optional[BaseException] = None) -> None:
        self._vit_tail_pending = pending
        if error is not None:
            self._vit_error = error
            self._tail_error = error
        ev = getattr(self, "_vit_done", None)
        if ev is not None:
            ev.set()

    async def vit_complete_tails(self) -> None:
        """Before the refit: wait for the VIT decisions, then the fallback pools' tails."""
        ev = getattr(self, "_vit_done", None)
        if ev is None:
            return
        await ev.wait()
        if self._vit_error is not None:
            raise RuntimeError("verify_in_train failed") from self._vit_error
        pending, self._vit_tail_pending = self._vit_tail_pending, []
        if pending:
            await self.complete_tails(pending)

    def _select_prompts(self) -> None:
        """Prompt drop: keep the first (prompts - spares) undropped prompts in
        drafting order (dropped ones fill a shortfall and decode fresh); plans
        of the other prompts are discarded so their tails are never decoded."""
        order = list(self._null_all_slots)  # drafting (= batch) order
        train = len(order) - self.null_bv_spare_prompts
        dropped = self.null_dropped_keys
        kept = [k for k in order if k not in dropped][:train]
        kept += [k for k in order if k in dropped][: train - len(kept)]
        self.null_selected_keys = kept
        for key in [k for k in self.plans if k not in set(kept)]:
            del self.plans[key]

    def _count_tails(self) -> None:
        """Mark plans that need a continuation and count them per prompt."""
        self._tail_counts = {}
        for key, queue in self.plans.items():
            for plan in queue:
                plan.tail = not plan.complete
            self._tail_counts[key] = sum(plan.tail for plan in queue)

    def tail_count(self, prompt: torch.Tensor) -> int:
        """Rollouts of this prompt whose plan needed a continuation (tail rows)."""
        return self._tail_counts.get(_key(prompt), 0)

    def claim_row(self, prompt: torch.Tensor, token_ids: torch.Tensor) -> Optional[RowInfo]:
        """Verification facts of a finished speculative rollout (None: never drafted)."""
        return self._row_claims.claim(prompt, token_ids)

    def _verify_block(
        self,
        flat,
        rows,
        null_envelopes: Optional[list[tuple[float, ...]]] = None,
        band: Optional[float] = None,
    ) -> dict[int, dict[str, Any]]:
        """Block-verify ``flat`` drafts on every verifier pool; results by row.

        Engine-sourced verification places rows on a pool's data-parallel
        ranks by tokens (``lpt_owners``). ``null_envelopes`` (⊥-BV): every row
        is verified at each of its envelopes (``null_envelopes[i]`` for row i);
        the driver decides (``_decide_null``).
        ``band``: banded IS (only a draft's first verification; never with
        ``null_envelopes``).
        """
        futs = self._submit_verify_block(flat, rows, null_envelopes, band=band)
        return self._verify_block_results(flat, [(a, ray.get(refs)) for a, refs in futs])

    async def _verify_block_async(
        self, flat, rows, prefix_keys: Optional[list[str]] = None
    ) -> dict[int, dict[str, Any]]:
        """``_verify_block`` without blocking the driver's event loop.

        While streaming over the tails it is an ordered learner call
        (``ordered_call``): learner actors are asynchronous Ray actors, which
        may start queued calls in different orders on different ranks; the
        numbered calls run in the same order everywhere.
        """
        futs = self._submit_verify_block(flat, rows, ordered=True, prefix_keys=prefix_keys)
        done = []
        for a, refs in futs:
            # A worker thread waits on the results; the event loop keeps going.
            done.append((a, await asyncio.to_thread(ray.get, refs)))
        if getattr(self, "learner_ordered", False):
            self._learner_inflight.difference_update(r for _, refs in futs for r in refs)
        return self._verify_block_results(flat, done)

    def _submit_verify_block(
        self,
        flat,
        rows,
        null_envelopes=None,
        ordered=False,
        band=None,
        prefix_keys=None,
        row_seed_base=None,
    ) -> list:
        """Send ``flat``'s block verification to the verifier pools: [(first row, refs)].

        ``ordered``: as numbered learner calls while streaming over the tails.
        ``prefix_keys`` (learner prefix-state cache on): the rows' cache keys;
        a row goes to the rank that holds its state (``PrefixCacheDirectory``).
        """
        # Contiguous row slices per verifier pool, verified concurrently.
        k = len(self.verifiers)
        cuts = [round(j * len(flat) / k) for j in range(k + 1)]
        futs = []
        for g, a, b in zip(self.verifiers, cuts, cuts[1:]):
            method, extra = self._verify_method(flat[a:b])
            if method == "verify_drafts_block_engine":
                # Rows placed on the pool's data-parallel ranks by tokens.
                lengths = [r.numel() for r in rows[a:b]]
                ranks = data_parallel_size(g)
                if prefix_keys is not None and self.prefix_cache is not None:
                    plens = [d[0].numel() for _, d in flat[a:b]]
                    loads = [
                        self.prefix_cache.forwarded(k, n, p)
                        for k, n, p in zip(prefix_keys[a:b], lengths, plens)
                    ]
                    owners, sent = self.prefix_cache.place(
                        prefix_keys[a:b], lengths, plens, ranks
                    )
                    extra = {
                        **extra,
                        "prefix_keys": sent,
                        "prefix_release": self.prefix_cache.take_release(),
                        "prefix_gpu_gb": self.prefix_cache.gpu_gb,
                    }
                    lengths = loads
                else:
                    owners = lpt_owners(lengths, ranks)
                extra = {**extra, "row_ranks": owners}
                self._count_balance(lengths, owners, ranks)
            if null_envelopes is not None:
                extra = {**extra, "null_envelopes": [list(e) for e in null_envelopes[a:b]]}
            if band is not None:
                extra = {**extra, "band": band}
            if row_seed_base is not None:
                if method != "verify_drafts_block_engine":
                    raise RuntimeError("verify_in_train needs engine-sourced q")
                extra = {**extra, "row_seed_base": int(row_seed_base)}
            seq = self.learner_seq() if ordered else None
            call = (method,) if seq is None else ("ordered_call",)
            if seq is not None:
                extra = {**extra, "seq": seq, "method": method}
            futs.append(
                (
                    a,
                    g.run_all_workers_single_data(
                        *call,
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
            if seq is not None:
                self.learner_track(futs[-1][1])
        return futs

    def _verify_block_results(self, flat, done) -> dict[int, dict[str, Any]]:
        """Results by row from [(first row, per-rank results)] of ``_submit_verify_block``."""
        by_row = dict(
            (a + j, r) for a, ranks in done for rank_res in ranks for j, r in rank_res
        )
        missing = sorted(set(range(len(flat))) - by_row.keys())
        if missing:
            raise RuntimeError(f"block verification returned no result for rows {missing}")
        self._verified_hidden(flat, by_row)
        return by_row

    def _count_balance(self, lengths: list[int], owners: list[int], ranks: int) -> None:
        """``spec/verify_load_max_over_mean``: busiest rank's tokens over the mean."""
        load = [0] * ranks
        for n, r in zip(lengths, owners):
            load[r] += n
        if sum(load):
            self.stats["spec/verify_load_max_over_mean"] = max(load) * ranks / sum(load)

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
        pairs: list[tuple[int, int]] = []
        for i in range(len(drafts)):
            d, r = best[i]
            plan = self._plan(d, r)
            plan.verified = (len(d[2]), r["accepted"])
            pairs.append((len(d[2]), r["accepted"]))
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
        self._band_stats()
        self._longest_stats(pairs)
        if self.verify_source == "engine":
            self.stats["spec/engine_q_dropped"] = float(dropped)

    def _verify_null(
        self, flat, rows, groups: list[str], slots: dict[str, int]
    ) -> dict[int, dict[str, Any]]:
        """⊥-BV of the pools in waves: forward only rows that can still be used.

        Each wave forwards, per prompt with slots left, its next
        ceil(s * M) rows in drafting order (s slots left, M the envelope the
        counts give now: the expected number of drafts that fill s slots).
        Every forwarded row is verified at every envelope it can be decided
        at (``row_envelopes``: a function of its index in the pool and the
        counts, not of any content) with one set of draws (``null_grid``);
        the driver then walks each prompt's rows in
        drafting order with the count process (``NullGroup``) and takes each
        row's outcome at the M it gets. Which rows are forwarded depends on
        counts only, so the decisions are those of the one-shot process
        (``decide_pools``); rows after the slots fill are "unused", and those
        never forwarded cost nothing (their hidden states are freed).
        """
        policy = self._null_policy
        queue: dict[str, list[int]] = defaultdict(list)
        for i, g in enumerate(groups):
            queue[g].append(i)
        row_env: dict[int, tuple[float, ...]] = {}
        for g, q in queue.items():
            for j, i in enumerate(q):
                row_env[i] = row_envelopes(policy, len(q), int(slots[g]), j)
        state = {g: NullGroup(len(q), int(slots[g]), policy) for g, q in queue.items()}
        out: dict[int, dict[str, Any]] = {}
        waves = 0
        while True:
            batch = []
            for g, q in queue.items():
                m = state[g].next_m()
                if m is None or not q:
                    continue
                k = min(len(q), math.ceil(state[g].s * m))
                batch += q[:k]
                del q[:k]
            if not batch:
                break
            waves += 1
            res = self._verify_block(
                [flat[i] for i in batch],
                [rows[i] for i in batch],
                null_envelopes=[row_env[i] for i in batch],
            )
            for j in sorted(range(len(batch)), key=lambda j: batch[j]):
                i, r = batch[j], res[j]
                m = state[groups[i]].next_m()
                tail_drop = False
                if m is not None:
                    tau, y, _ = r["null_grid"][m]
                    output = y >= 0
                    # Drop-IS: an output rejected inside its draft is dropped.
                    tail_drop = self.null_bv_drop_is and output and tau < len(flat[i][1][2])
                    state[groups[i]].record(output, tail_dropped=tail_drop)
                out[i] = _null_result(r, m)
                if tail_drop:
                    out[i]["null"] = "tail_dropped"
        for st in state.values():
            st.next_m()  # a pool left short after its last draft is dropped too
        self._null_dropped = {g for g, st in state.items() if st.dropped}
        skipped = [i for q in queue.values() for i in q]
        self._release_hidden(flat[i][1] for i in skipped)
        for i in skipped:
            out[i] = _null_result({}, None)
        self.stats["spec/null_waves"] = float(waves)
        self.stats["spec/null_envelopes_per_row"] = sum(map(len, row_env.values())) / max(len(row_env), 1)
        self.stats["spec/null_skipped_drafts"] = float(len(skipped))
        return out

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
        out_pairs: list[tuple[int, int]] = []
        rejected: list[tuple[int, int]] = []  # (draft length, tau) of tail decodes
        dropped_groups = getattr(self, "_null_dropped", set())
        n_tail_drops = 0
        weights: list[float] = []
        for r, (_, d) in enumerate(flat):
            res = by_row[r]
            state = res.get("null")
            if state is None:
                raise RuntimeError(f"null_bv: row {r} was verified without its group")
            if state == "unused" or groups[r] in dropped_groups:
                # Unused, or its prompt was dropped (count-only): no plan.
                n_unused += state == "unused"
                continue
            if state == "tail_dropped":
                n_tail_drops += 1
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
            plan.verified = (g, res["accepted"])
            if self.null_bv_drop_is:
                # max(1, W*/M): the kept row's law is p P(no tail | y) / P(no tail).
                plan.log_weight = max(0.0, float(res["log_wstar"]) - math.log(res["null_m"]))
                weights.append(math.exp(plan.log_weight))
            full += plan.complete and res["accepted"] == g
            kept += res["accepted"]
            total += g
            out_pairs.append((g, res["accepted"]))
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
        fresh = sum(
            max(0, s - outs_by_group[grp])
            for grp, s in slots.items()
            if grp not in dropped_groups
        )
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
        if self._null_policy.prompt_drop:
            inv = {name: key for key, name in self._null_names.items()}
            self.null_dropped_keys = {inv[g] for g in dropped_groups}
            self._select_prompts()
            self.stats["spec/null_dropped_prompts"] = float(len(dropped_groups))
            self.stats["spec/null_drop_rate"] = len(dropped_groups) / max(len(slots), 1)
        if self.null_bv_drop_is:
            sw, sw2 = sum(weights), sum(w * w for w in weights)
            self.stats.update(
                {
                    "spec/drop_is_tail_drops": float(n_tail_drops),
                    "spec/drop_is_ess": sw * sw / (len(weights) * sw2) if weights else 1.0,
                    "spec/drop_is_weighted_frac": sum(w != 1.0 for w in weights)
                    / max(len(weights), 1),
                    "spec/drop_is_max_weight": max(weights, default=1.0),
                }
            )
        self._age_stats(age_tok, age_kept)
        self._band_stats()
        self._longest_stats(out_pairs)
        if self.verify_source == "engine":
            self.stats["spec/engine_q_dropped"] = float(dropped)

    def _cap(self, prompt_len: int) -> int:
        """Most tokens a rollout of this prompt can generate (the length cap).

        ``max_new_tokens``, or fewer where prompt + generated would pass the
        engine's total-length cap: the engine stops there, so a prefix at
        this cap is a finished rollout (a continuation of it would generate
        nothing). It depends on the prompt length only, never on content.
        """
        if self.max_seq_len is None:
            return self.max_new_tokens
        return max(0, min(self.max_new_tokens, self.max_seq_len - prompt_len))

    def _horizon(self, prompt_len: int) -> Optional[int]:
        """Anchor horizon position c = P + min(H, cap) (None: no horizon).

        Positions < c carry the plain theta_{k-1} routes (drafted rollouts),
        positions >= c are self. A function of the prompt length only.
        """
        if self.anchor_horizon is None:
            return None
        # The rollout's cap: max_new_tokens bounds one turn only when the
        # rollout has several (multi-turn: the total-length cap bounds it).
        cap = (
            self._cap(prompt_len)
            if self.turn_env is None or self.max_seq_len is None
            else max(0, self.max_seq_len - prompt_len)
        )
        return prompt_len + min(int(self.anchor_horizon), cap)

    def _cut(self, anchors: Optional[RouteRefs], prompt_len: int) -> Optional[RouteRefs]:
        """``anchors`` restricted to positions below the horizon (self beyond)."""
        c = self._horizon(prompt_len)
        return anchors if anchors is None or c is None else anchors.truncate(c)

    def _horizon_done(self, plan: Plan, prompt_len: int) -> bool:
        """Whether a plan's theta_{k-1} part is done: its anchors below c are logged.

        The rest of the rollout is sampled from p_k directly by the refit
        engine, which applies those anchors.
        """
        c = self._horizon(prompt_len)
        return (
            c is not None
            and plan.anchors is not None
            and plan.anchors.complete >= c
        )

    def _band_stats(self) -> None:
        """Banded-IS statistics over this verification's outputs.

        ``spec/band_ess``: (sum w)^2 / (n sum w^2), the weights' effective
        sample fraction (1/ess = gradient-variance inflation from the weights);
        ``spec/band_exit_frac``: outputs whose prefix left the band (only those
        can be rejected).
        """
        rows, self._band_rows = self._band_rows, []
        if self.banded_is_band is None or not rows:
            return
        w = [math.exp(lw) for lw, _ in rows]
        self.stats.update(
            {
                "spec/band_outputs": float(len(w)),
                "spec/band_exit_frac": sum(e for _, e in rows) / len(w),
                "spec/band_ess": sum(w) ** 2 / (len(w) * sum(x * x for x in w)),
                "spec/band_w_min": min(w),
                "spec/band_w_max": max(w),
            }
        )

    def _stopped(self, d) -> bool:
        """Whether draft ``d``, if fully accepted, is a finished rollout."""
        # The draft terminated on its own (EOS/stop) rather than at the budget
        # or the deadline: a fully accepted draft is then a finished rollout.
        finished = d[3] if len(d) > 3 else True
        # Cohort/streamed drafts carry the full budget: finished = EOS or the cap.
        stopped = finished and (len(d) > 4 or len(d[2]) < self.draft_budget)
        # A draft at the length cap is a finished rollout: nothing follows it.
        return stopped or len(d[2]) >= self._cap(d[0].numel())

    def _plan(self, d, r) -> Plan:
        """Turn one draft and its verification result into a rollout plan."""
        plan = self._plan_tokens(d, r)
        if "band_log_weight" in r:
            plan.log_weight = float(r["band_log_weight"])
            self._band_rows.append((plan.log_weight, bool(r["band_exited"])))
        return plan

    def _plan_tokens(self, d, r) -> Plan:
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
            anchors = d[8].truncate(prompt.numel() + n_acc)
            if n_acc == len(draft) and not complete:
                # The bonus row's anchor (the last drafted token's route) was
                # not logged: keep the whole draft, and the next token comes
                # from a continuation round (``verify_sticky``), an exact
                # sampler of p given this prefix like the bonus draw itself.
                stream = d[9] if len(d) > 9 else None
                return Plan(seed, list(draft), lps[:n_acc], anchors=anchors, stream=stream)
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
        done = r["next"] == self.eod or len(prefix) >= self._cap(prompt.numel())
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

    def _longest_stats(self, pairs: list[tuple[int, int]]) -> None:
        """Earliest rejection among the longest verified drafts (``spec/longest_*``).

        ``pairs`` = (draft length, accepted tokens) of every verified draft. The
        longest drafts (top decile by length, at least one) are the step's likely
        stragglers: a rejection at position t leaves (length - t) tokens to
        decode after verification, so the step waits on the earliest one
        (research note agent_short_acceptance.md, falsifier 1).
        """
        if not pairs:
            return
        lens = sorted((g for g, _ in pairs), reverse=True)
        cut = lens[max(1, len(lens) // 10) - 1]
        top = [(g, t) for g, t in pairs if g >= cut]
        rej = [(g, t) for g, t in top if t < g]
        self.stats.update(
            {
                "spec/longest_len_cut": float(cut),
                "spec/longest_drafts": float(len(top)),
                "spec/longest_rejections": float(len(rej)),
                "spec/longest_min_reject_pos": float(min((t for _, t in rej), default=-1)),
                "spec/longest_min_reject_frac": min((t / max(g, 1) for g, t in rej), default=1.0),
                "spec/longest_max_remaining": float(max((g - t for g, t in rej), default=0)),
                "spec/longest_accepted_frac": sum(t for _, t in top) / max(sum(g for g, _ in top), 1),
            }
        )

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
                if self.banded_is_band is not None:
                    extra["band"] = self.banded_is_band
                refs = group.run_all_workers_single_data(method, **kwargs, **extra)
            calls.append((a, refs))
        kept = total = full = fused = 0
        age_tok: dict[int, int] = {}
        pairs: list[tuple[int, int]] = []
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
                    pairs.append((len(d[2]), r["accepted"]))
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
        self._band_stats()
        self._longest_stats(pairs)
        if self.fused_verify_train:
            self.stats["fused/retained"] = float(fused)
        if self.verify_source == "engine":
            self.stats["spec/engine_q_dropped"] = float(dropped)

    # -- sticky routing: verification and theta_n continuations --------------------

    async def verify_sticky(self, drafts) -> None:
        """Verify drafts and complete every drafted rollout (sticky MoE routing).

        Called at the start of iteration k with the learner at theta_k and the
        engine still at the drafting weights D = theta_{k-1} (or theta_hat_k with
        momentum draft weights; theta_{k-1} below stands for D) until the refit
        after this returns. The
        target is p = F(theta_k; A): anchors A = plain theta_{k-1} routes on the
        rollout's own prefix at every position up to the rollout cap (the cut
        c = prompt + max_new_tokens, content-independent), "self" for rows
        never drafted. Positions after a rejection need theta_{k-1} routes on
        the new prefix, so the engine keeps theta_{k-1} and continues every
        incomplete plan (``_continue_sticky``): it decodes from prompt + kept
        prefix toward the end (EOS or the cap) with its plain forward, logging
        routes, hidden states and log-probs as for any draft, and the learner
        block-verifies each continuation in pieces while it is decoded. Every
        piece is exact block verification of a q draft against the fixed
        target p, so the final rollout is an exact sample of p.
        """
        pending = await self.begin_sticky(drafts)
        # Also with nothing pending (every draft whole-accepted and complete):
        # the step then reports zero continuation rounds instead of no stats.
        await self._finish_tails(pending)

    async def _finish_tails(self, pending) -> None:
        """The continuation rounds of ``pending``, then the step's tail timeline stats."""
        trace = self.__dict__.pop("_tail_trace", None)
        ended = self.__dict__.pop("_sticky_ended", ())
        plans = self.__dict__.pop("_sticky_plans", ())
        await self._continue_sticky(pending, trace, ended)
        if trace is not None:
            self.stats.update(trace.summary())
        if self.turn_env is not None and os.environ.get("THUNDERSYNC_ENGINE_Q_CHECK") == "1":
            await self._route_check_plans()
        self._register_horizon_plans(plans)

    async def begin_sticky(self, drafts) -> list:
        """``verify_sticky`` up to its continuation rounds; returns the pending plans.

        Verifies the drafts (and, as a debug check, their routes) and stops
        every draft stream an unfinished whole-accepted plan does not go on
        with. ``complete_tails`` then runs the continuation rounds; rollouts
        of complete plans may be served and trained meanwhile (their prefix,
        log-probs and anchors are final).
        """
        self._plan_cv = None
        trace = TailTrace()
        if os.environ.get("THUNDERSYNC_ENGINE_Q_CHECK") == "1":
            t_check = time.perf_counter()
            await self._route_check(drafts)
            trace.mark("route_check")
            # Debug re-prefill inside time/verify: reported so it can be subtracted.
            self.stats["time/route_check"] = time.perf_counter() - t_check
        streams = [d[9] for v in drafts for d in v if len(d) > 9 and d[9] is not None]
        self.verify(drafts)
        trace.mark("verify0")
        for key, queue in self.plans.items():
            for plan in queue:
                if plan.verified is not None:
                    trace.verified(key, plan, *plan.verified)
        ended = []
        if self.turn_env is not None:
            # Multi-turn: each drafted plan serves one rollout-loop row (the
            # environment is stepped with that row's state). A plan whose
            # first turn ended in verification goes to the environment.
            self._row_plans = pair_rows(self._row_prompts, self.plans)
            left = sum(len(q) for q in self.plans.values())
            if left:
                raise RuntimeError(
                    f"multi-turn: {left} verified plans have no rollout row "
                    "(drafted prompts differ from the step's rows)"
                )
            for i, plan in self._row_plans.items():
                plan.row = i
                if plan.complete:
                    plan.complete = False
                    if not self._horizon_done(plan, len(self._row_prompts[i])):
                        # (Past the horizon the refit engine's turn loop,
                        # ``finish_row``, steps the environment.)
                        ended.append((_key(self._row_prompts[i]), plan))
            plans = [(_key(self._row_prompts[i]), p) for i, p in self._row_plans.items()]
        else:
            plans = [(key, plan) for key, queue in self.plans.items() for plan in queue]
        waiting = {id(plan) for _, plan in ended}  # on the environment
        pending = [
            (key, plan)
            for key, plan in plans
            if not plan.complete
            and not plan.vit
            and id(plan) not in waiting
            and not self._horizon_done(plan, len(key))
        ]
        # The drafts' streams still decode: whole-accepted unfinished plans go
        # on with theirs, every other stream stops.
        kept = {id(plan.stream) for _, plan in pending if plan.stream is not None}
        self._stop_streams([x for x in streams if id(x) not in kept])
        self._tail_plans = {id(plan) for _, plan in pending}
        # Verify-in-train placeholders: their rollouts wait for the training forward.
        self._tail_plans |= {id(plan) for _, plan in plans if plan.vit}
        # Rank-0 tail timeline of this step (TailTrace), finished by the
        # continuation rounds (``_continue_sticky``).
        self._tail_trace = trace
        # Multi-turn: plans whose turn ended in verification (to the
        # environment first); anchor horizon: every plan of the step.
        self._sticky_ended, self._sticky_plans = ended, plans
        return pending

    # -- streaming training over the tails (one engine, sticky) ------------------

    def begin_streaming_tails(self) -> None:
        """Rollouts of this step start before its tails are complete.

        Called inside the step's event loop before its rollouts. A rollout
        whose plan is incomplete waits for ``complete_tails`` to finish it; a
        rollout with no plan (decoded at theta_k) waits for ``weights_ready``.
        Training of complete groups therefore streams over the other rows'
        continuation rounds: both are learner-side at theta_k, and a complete
        plan's prefix, log-probs and anchors never change again.
        """
        self._tail_cv = asyncio.Condition()
        self._tail_error: Optional[BaseException] = None
        self._weights_ready = asyncio.Event()
        # Learner calls in flight are numbered and run in that order on every
        # rank (``learner_seq`` / ``ordered_call``); ``learner_drain`` waits
        # for all of them before an unordered call (the refit).
        self.learner_ordered = True
        self._learner_inflight: set = set()
        self.learner_wait = {"drain": 0.0}

    def end_streaming_tails(self) -> None:
        for k, v in getattr(self, "learner_wait", {}).items():
            self.stats[f"learner/{k}_wait_s"] = v
        self._tail_cv = None
        self._weights_ready = None
        self.learner_ordered = False

    def learner_seq(self) -> Optional[int]:
        """The next ordered-call number while streaming over the tails, else None.

        Persistent across steps: the workers count every ordered call they run.
        """
        if not self.__dict__.get("learner_ordered", False):
            return None
        # One counter per process: the learner workers count every ordered call,
        # whichever SpeculativeGeneration (engine lane) sent it.
        return next(_LEARNER_SEQ)

    def learner_track(self, refs) -> None:
        """Ordered learner calls in flight (``learner_drain`` waits for them)."""
        self._learner_inflight.update(refs)

    async def learner_drain(self) -> None:
        """Wait until every ordered learner call sent so far has finished."""
        t0 = time.perf_counter()
        pending = list(self._learner_inflight)
        if pending:
            await asyncio.to_thread(ray.wait, pending, num_returns=len(pending))
        self._learner_inflight.difference_update(pending)
        self.learner_wait["drain"] += time.perf_counter() - t0

    async def _notify_tails(self) -> None:
        cv = getattr(self, "_tail_cv", None)
        if cv is not None:
            async with cv:
                cv.notify_all()

    async def complete_tails(self, pending) -> None:
        """The continuation rounds of ``pending`` (``begin_sticky``), waking waiters."""
        try:
            await self._finish_tails(pending)
        except BaseException as exc:
            self._tail_error = exc
            raise
        finally:
            await self._notify_tails()

    def weights_ready(self, error: Optional[BaseException] = None) -> None:
        """The engine holds theta_k: rollouts without a plan may decode.

        ``error``: the tails or the refit failed; waiting rollouts raise.
        """
        if error is not None:
            self._tail_error = error
        ev = getattr(self, "_weights_ready", None)
        if ev is not None:
            ev.set()

    async def _await_plan(self, plan: Plan) -> None:
        cv = getattr(self, "_tail_cv", None)
        if cv is None or plan.complete:
            return
        async with cv:
            # Done when complete, or (anchor horizon) handed to the refit engine,
            # which continues it once it holds theta_k.
            while not (plan.complete or plan.engine_key is not None):
                if self._tail_error is not None:
                    raise RuntimeError("sticky continuation failed") from self._tail_error
                await cv.wait()

    def _register_horizon_plans(self, plans) -> None:
        """Hand the anchors of plans the refit engine continues to the engine.

        Anchor horizon: every incomplete plan now has its anchors below c
        (``_horizon_done``); ``_one`` continues it at theta_k with the request
        tagged by ``plan.engine_key``. All generation workers get the handles
        (the coordinator picks the rank that runs a request; the rank that
        admits it fetches its own stage's blocks); no route data passes
        through the driver.
        """
        if self.anchor_horizon is None:
            return
        # Engine anchor-apply counters so far (cumulative, per rank; summed).
        totals: dict[str, float] = defaultdict(float)
        for st in ray.get(
            self.base._policy.worker_group.run_all_workers_single_data(
                "engine_anchor_stats"
            )
        ):
            for k, v in (st or {}).items():
                totals[k] += v
        for k, v in totals.items():
            self.stats[f"spec/anchor_{k}"] = v
        entries = {}
        for key, plan in plans:
            if plan.complete:
                continue
            if not self._horizon_done(plan, len(key)):
                raise RuntimeError(
                    "sticky routing: a drafted rollout left the theta_{k-1} "
                    "rounds without its anchors below the horizon"
                )
            plan.engine_key = f"h{next(self._anchor_keys)}"
            entries[plan.engine_key] = self._cut(plan.anchors, len(key))
        ray.get(
            self.base._policy.worker_group.run_all_workers_single_data(
                "register_engine_anchors", entries=entries
            )
        )
        self.stats["spec/horizon_plans"] = float(len(entries))

    def bind_rows(self, rows: list[dict[str, Any]]) -> None:
        """This step's rollout rows (multi-turn): initial sample states, in row order.

        Called before ``verify_sticky``; the drafted plans are paired with
        these rows there (``pair_rows``) and claimed by ``claim_row_plan``.
        """
        if self.turn_env is None:
            raise RuntimeError("bind_rows needs a turn environment (multi-turn rollouts)")
        self.turn_env.bind(rows)
        self._row_prompts = [
            torch.cat([m["token_ids"] for m in r["message_log"]]) for r in rows
        ]
        self._row_plans = {}

    def claim_row_plan(self, row: int) -> Optional[Plan]:
        """The finished drafted plan of rollout-loop row ``row`` (multi-turn), or None."""
        plan = self._row_plans.pop(row, None)
        if plan is not None and not plan.complete and plan.engine_key is None:
            raise RuntimeError(
                f"sticky routing: row {row}'s drafted multi-turn rollout reached the "
                "rollout loop unfinished (verify_sticky must complete every turn)"
            )
        return plan

    async def replay_row(
        self, row: int
    ) -> Optional[tuple[dict, dict, Optional[RouteRefs]]]:
        """Row ``row``'s drafted multi-turn rollout, or None if undrafted.

        (final sample state, sample metrics, anchors), rebuilt from the
        recorded turns (the environment is not stepped again). Anchor
        horizon: a plan handed to the refit engine is finished first
        (``_finish_turns``).
        """
        plan = self.claim_row_plan(row)
        if plan is None:
            return None
        if not plan.complete:
            await self._finish_turns(row, plan)
        state, metrics = replay(
            self.turn_env, row, plan.prefix, plan.prefix_logprobs, plan.turns
        )
        return state, metrics, plan.anchors

    async def _finish_turns(self, row: int, plan: Plan) -> None:
        """Anchor horizon: finish a drafted multi-turn rollout on the refit engine.

        The plan's anchors below c are complete (``_horizon_done``), so every
        later token is sampled from p_k by the engine at theta_k with the
        request tagged (``plan.engine_key``): each turn's prefill (prompt,
        earlier turns, observations) re-applies the anchors below c by
        position. Turn ends and environment steps follow the same rules as
        the turns before the horizon (``TurnEnvironment``).
        """
        key = _key(self._row_prompts[row])
        prompt = self._row_prompts[row]
        while True:
            n_turn = len(plan.prefix) - plan.turn_start
            if n_turn > 0 and self._turn_ended(key, plan, plan.prefix[-1]):
                rec = await self.turn_env.step(
                    row, len(key), list(plan.prefix), plan.turns, plan.turn_start
                )
                plan.turns.append(rec)
                if self.turn_env.rollout_ends(plan.turns):
                    break
                plan.prefix += rec.obs
                plan.prefix_logprobs += [0.0] * len(rec.obs)
                plan.turn_start = len(plan.prefix)
                continue
            ids = torch.cat([prompt, torch.tensor(plan.prefix, dtype=torch.long)])
            cont = BatchedDataDict(
                {
                    "input_ids": ids.view(1, -1),
                    "input_lengths": torch.tensor([ids.numel()]),
                    "max_new_tokens": torch.tensor(
                        [self._turn_cap(key, plan) - n_turn]
                    ),
                    "ts_anchor_key": [plan.engine_key],
                }
            )
            new = 0
            async for _, res in self.base.generate_async(cont):
                glen = int(res["generation_lengths"][0])
                c0 = ids.numel()
                plan.prefix += res["output_ids"][0, c0 : c0 + glen].tolist()
                plan.prefix_logprobs += res["logprobs"][0, c0 : c0 + glen].tolist()
                new += glen
            if new == 0 and not self._turn_ended(key, plan, plan.prefix[-1] if plan.prefix else -1):
                raise RuntimeError(
                    f"row {row}: the engine generated nothing before the turn ended"
                )
        plan.complete = True
        self.stats["spec/horizon_engine_turns"] = self.stats.get(
            "spec/horizon_engine_turns", 0.0
        ) + float(len(plan.turns))

    def _turn_cap(self, key, plan: Plan) -> int:
        """Most tokens the plan's current turn can generate (per-turn length cap)."""
        return self._cap(len(key) + plan.turn_start)

    def _turn_ended(self, key, plan: Plan, last: int) -> bool:
        """Whether the verified prefix ends the current turn (``last``: its last token)."""
        return last == self.eod or len(plan.prefix) - plan.turn_start >= self._turn_cap(
            key, plan
        )

    def _stop_streams(self, streams) -> None:
        """Abort engine draft streams and close their logs (rank 0 + every engine)."""
        if not streams:
            return
        rank0 = self.base._policy.worker_group.workers[0]
        ray.get(rank0.abort_drafts.remote([x["idx"] for x in streams]))
        ray.get(
            self.base._policy.worker_group.run_all_workers_single_data(
                "close_draft_logs", keys=[x["hkey"] for x in streams]
            )
        )

    async def _route_check(self, drafts) -> None:
        """Debug: the drafts' logged anchors against a fresh engine prefill.

        ``THUNDERSYNC_ENGINE_Q_CHECK=1``, sticky routing. The engine still
        holds the drafting weights here, so re-routing prompt + draft in one
        prefill (another batch layout and code path than the decode steps
        that logged most anchors) must give the same expert set at every
        anchored position; a mis-mapped log (wrong slot or position) would
        not. Counted in ``spec/route_check_tokens`` / ``spec/route_check_exact``.
        """
        ds = [d for v in drafts for d in v if len(d) > 8 and d[8] is not None and d[2]]
        # THUNDERSYNC_ROUTE_CHECK_ROWS=n: check a seeded random sample of n drafts
        # (bounded cost at scale; unset: every draft).
        sample = int(os.environ.get("THUNDERSYNC_ROUTE_CHECK_ROWS", "0") or 0)
        if 0 < sample < len(ds):
            import random

            pick = sorted(random.Random(self.current_step).sample(range(len(ds)), sample))
            ds = [ds[j] for j in pick]
        await self._route_check_rows(
            [(d[0], d[2], d[8], f"key={d[4]} segments={d[5]}") for d in ds],
            "route_check",
        )

    async def _route_check_plans(self) -> None:
        """Debug: finished multi-turn plans' anchors against a fresh engine prefill.

        ``THUNDERSYNC_ENGINE_Q_CHECK=1``, multi-turn sticky routing, after
        ``_continue_sticky`` (the engine still holds the drafting weights).
        Covers the observation positions, whose anchors come from the
        continuation streams' prefills (``spec/route_check_mt_*``;
        ``spec/route_check_mt_obs_positions``: anchored observation tokens).
        """
        items, obs = [], 0
        for i, plan in sorted(self._row_plans.items()):
            if plan.anchors is None or not plan.prefix:
                continue
            prompt = self._row_prompts[i]
            n = plan.anchors.length - prompt.numel()
            items.append((prompt, plan.prefix[:n], plan.anchors, f"row={i} turns={len(plan.turns)}"))
            obs += sum(
                max(0, min(t.gen_end + len(t.obs), n) - t.gen_end)
                for t in plan.turns
            )
        self.stats["spec/route_check_mt_obs_positions"] = float(obs)
        await self._route_check_rows(items, "route_check_mt")

    async def _route_check_rows(self, items, name: str) -> None:
        """Re-route (prompt, tokens) rows in one engine prefill; compare to their anchors.

        ``items``: (prompt, tokens, ``RouteRefs``, label) per row. Counted in
        ``spec/{name}_tokens`` / ``spec/{name}_exact``.
        """
        if not items:
            return
        rank0 = self.base._policy.worker_group.workers[0]
        rows = [torch.cat([p, torch.tensor(t, dtype=torch.long)]) for p, t, _, _ in items]
        hkeys = self._hidden_keys([f"{name}:{j}" for j in range(len(items))])
        # Next to the drafts' streams, which keep running (``_handout_running``).
        idx = ray.get(
            rank0.add_drafts.remote(
                [r.tolist() for r in rows], [1] * len(rows), self.stream_interval,
                hidden_keys=hkeys,
            )
        )
        while not all(done for _, done in ray.get(rank0.draft_lengths.remote(idx))):
            await asyncio.sleep(0.02)
        check = -1  # route blocks of this check only: freed below
        refs = self._pop_routes(
            hkeys, [r.numel() for r in rows], [0] * len(rows), target=check
        )
        self._pop_hidden(hkeys, [0] * len(rows))
        tok = exact = 0
        for (p, toks, anchors, label), ref in zip(items, refs):
            n = p.numel() + len(toks)
            mine = fetch_anchors(anchors, self._route_layers, None, n, ray.get)
            other = fetch_anchors(ref, self._route_layers, mine.shape[-1], n, ray.get)
            nt, ne = compare_anchors(mine, other)
            tok, exact = tok + nt, exact + ne
            if ne < nt:
                # First differing (position, layer) of this draft.
                bad = anchor_mismatches(mine, other)
                pos, col = bad[0].tolist()
                print(
                    f"[{name}] mismatch {label} prompt={p.numel()} "
                    f"len={n} pos={pos} layer={self._route_layers[col]} "
                    f"positions={nt - ne} (pos, layer column)={bad[:8].tolist()} "
                    f"logged={mine[pos, col].tolist()} "
                    f"prefill={other[pos, col].tolist()} "
                    f"blocks={[(b.lo, b.hi) for b in anchors.blocks]}",
                    flush=True,
                )
        (self._free_refs or ray_free)(self._route_blocks.pop(check, []))
        self.stats[f"spec/{name}_tokens"] = float(tok)
        self.stats[f"spec/{name}_exact"] = float(exact)
        print(
            f"[{name.replace('_', ' ')}] rows={len(items)} tok={tok} exact={exact}",
            flush=True,
        )

    def _piece_tokens(self, generated: int) -> int:
        """Continuation piece size at generated position ``generated`` of a row.

        ``max(sticky_chunk_tokens, growth * generated)``: a function of the
        position only (never of the tokens), so the cut stays exogenous and
        verification exact. Each piece is verified by a forward of the whole row
        (prompt + prefix + piece), so a fixed size costs O(L^2 / chunk) learner
        work per continued row; growing pieces need O(log L) of them
        (growth 1.0: <= 7 pieces per 8k row at chunk 128), at the price of
        finding a rejection up to growth * position tokens later.
        """
        return max(1, int(self.sticky_chunk_tokens), int(self.sticky_chunk_growth * generated))

    async def _continue_sticky(
        self,
        pending,
        trace: Optional[TailTrace] = None,
        ended=(),
        max_retries: int = 100,
    ) -> None:
        """Complete incomplete plans at theta_{k-1}, verifying while the engine decodes.

        One engine stream per plan decodes from prompt + kept prefix toward
        EOS or the cap. Whenever a stream has ``sticky_chunk_tokens`` new
        tokens (a fixed count: the cut is exogenous; a finished stream: all
        that is left), that piece is a q draft given the plan's current
        prefix, and the ready pieces of all streams are block-verified
        together while every stream keeps decoding. Whole acceptance extends
        the prefix and the same stream goes on (its later tokens are q
        samples given the extended prefix, a valid draft; no bonus is needed);
        a rejection appends the residual token, aborts the stream and starts a
        new one from the new prefix. A rejection therefore wastes at most the
        tokens the stream decoded past the piece, and verification overlaps
        decoding instead of alternating with it. Anchors of a piece are the
        stream's own route log (its prefill re-routes prompt + prefix);
        pieces are cut to their logged routes and hidden rows, which is
        engine bookkeeping, independent of the tokens.

        Multi-turn (``turn_env``): a verified prefix that ends a turn (EOS or
        the per-turn cap) goes to the environment (``ended``: plans whose
        turn ended before this call); unless the rollout ends there, its
        observation is appended and a new stream decodes the next turn from
        prompt + turns + observation. That stream's prefill routes the
        observation positions: their anchors. Environment steps run
        concurrently with the streams of other plans.

        Rounds do not block each other: a stream whose piece is being verified
        keeps decoding on the engine and gets its next piece only after the
        result (pieces of one plan are verified in order); pieces of other
        streams are cut and sent meanwhile. ``trace``: the step's TailTrace.
        """
        rank0 = self.base._policy.worker_group.workers[0]
        version = self.current_step - 1
        stats = dict(rounds=0, streams=0, tokens=0, kept=0, retries=0, turns=0)
        pieces_per_plan: dict[int, int] = defaultdict(int)
        stats.update(prefix_rows=0, prefix_hits=0, prefix_skipped=0)
        self._prefix_cache_begin()
        live: list[dict[str, Any]] = []
        retries: dict[int, int] = defaultdict(int)
        turns: dict[asyncio.Task, tuple] = {}

        def end_turn(key, plan: Plan) -> None:
            if self.turn_env is None:
                plan.complete = True
                return
            task = asyncio.create_task(
                self.turn_env.step(
                    plan.row, len(key), list(plan.prefix), plan.turns, plan.turn_start
                )
            )
            turns[task] = (key, plan)
            stats["turns"] += 1

        def next_turns() -> None:
            """Apply finished environment steps; start the next turns' streams."""
            nxt = []
            for task in [t for t in turns if t.done()]:
                key, plan = turns.pop(task)
                rec = task.result()
                plan.turns.append(rec)
                if self.turn_env.rollout_ends(plan.turns):
                    plan.complete = True
                    if trace is not None:
                        trace.completed(plan)
                    continue
                # New lists: a complete plan's prefix may be its draft's token list.
                plan.prefix = plan.prefix + rec.obs
                plan.prefix_logprobs = plan.prefix_logprobs + [0.0] * len(rec.obs)
                plan.turn_start = len(plan.prefix)
                nxt.append((key, plan))
            start(nxt)

        def start(items) -> None:
            if not items:
                return
            prompts = [
                torch.cat(
                    [
                        torch.tensor(key, dtype=torch.long),
                        torch.tensor(plan.prefix, dtype=torch.long),
                    ]
                )
                for key, plan in items
            ]
            budgets = [
                self._turn_cap(key, plan) - (len(plan.prefix) - plan.turn_start)
                if self.anchor_horizon is None
                # Anchor horizon: H + 1 tokens suffice (position c - 1 is routed
                # by the forward that samples token H); the refit engine does
                # the rest. At least one: a re-prefill logs every prefix position.
                else max(
                    1,
                    min(
                        self._turn_cap(key, plan) - (len(plan.prefix) - plan.turn_start),
                        int(self.anchor_horizon) + 1 - len(plan.prefix),
                    ),
                )
                for key, plan in items
            ]
            hkeys = self._hidden_keys([f"c{next(self._cont_ids)}" for _ in items])
            idx = ray.get(
                rank0.add_drafts.remote(
                    [p.tolist() for p in prompts],
                    budgets,
                    self.stream_interval,
                    hidden_keys=hkeys,
                )
            )
            for (key, plan), i, hk, p in zip(items, idx, hkeys, prompts):
                live.append(
                    dict(
                        key=key,
                        plan=plan,
                        idx=i,
                        hkey=hk,
                        length=len(key) + plan.turn_start + self._turn_cap(key, plan),
                        taken=0,
                        routes=None,
                    )
                )
            stats["streams"] += len(items)

        def stop(streams) -> None:
            self._stop_streams(streams)
            for s in streams:
                live.remove(s)

        # Plans whose draft stream still decodes go on with it (no new prefill).
        for key, plan in pending:
            if plan.stream is not None:
                live.append(dict(plan.stream, key=key, plan=plan))
                plan.stream = None
        start([(key, plan) for key, plan in pending if not any(s["plan"] is plan for s in live)])
        for key, plan in ended:
            end_turn(key, plan)
        inflight: dict[asyncio.Task, tuple[list, list, float]] = {}
        busy: set[int] = set()

        def pieces(cand):
            """(flat drafts, rows, owner streams) of the ready pieces of ``cand``."""
            idx = [s["idx"] for s in cand]
            snaps = self._peek_streams(rank0, idx)
            ready = []
            for s, (toks, done, lps) in zip(cand, snaps):
                toks = toks.tolist() if isinstance(toks, np.ndarray) else list(toks)
                a = s["taken"]
                s["decoded"] = len(toks)  # stream length when its piece is taken
                chunk = self._piece_tokens(len(s["plan"].prefix))
                n = len(toks) - a if done else ((len(toks) - a) // chunk) * chunk
                if n > 0 or done:
                    # (stream, piece, stream finished, piece log-probs)
                    ready.append(
                        (
                            s,
                            toks[a : a + n],
                            done,
                            None if lps is None else lps[a : a + n],
                        )
                    )
            if not ready:
                return [], [], []
            routes = self._pop_routes(
                [s["hkey"] for s, *_ in ready],
                [s["length"] for s, *_ in ready],
                [0] * len(ready),
                target=self.current_step,
                close=False,
                bases=[s["routes"] for s, *_ in ready],
            )
            # Pieces are cut to the anchored length, then to the logged hidden
            # rows (engine bookkeeping, independent of the tokens).
            cut = []
            for (s, toks, done, lps), a in zip(ready, routes):
                s["routes"] = a
                plen = len(s["key"]) + len(s["plan"].prefix)
                m = (
                    0
                    if a is None
                    else anchored_draft_length(
                        a.complete, plen, len(toks), self._horizon(len(s["key"]))
                    )
                )
                cut.append((s, toks, done, lps, plen, m))
            some = [c for c in cut if c[5] > 0]
            taken = dict(
                zip(
                    (id(c[0]) for c in some),
                    self._take_hidden(
                        [c[0]["hkey"] for c in some],
                        [c[0]["taken"] for c in some],
                        [c[5] for c in some],
                    )
                    if some
                    else [],
                )
            )
            hid = [taken.get(id(c[0])) for c in cut]
            flat, rows, owners, stale = [], [], [], []
            for (s, toks, done, lps, plen, m), h in zip(cut, hid):
                m = min(m, 0 if h is None else hidden_num_rows(h))
                prompt = torch.cat(
                    [
                        torch.tensor(s["key"], dtype=torch.long),
                        torch.tensor(s["plan"].prefix, dtype=torch.long),
                    ]
                )
                d = (
                    prompt,
                    -1,
                    toks[:m],
                    done and m == len(toks),
                    s["hkey"],
                    [(version, 0, m)],
                    None if lps is None else lps[:m],
                    h,
                    None
                    if s["routes"] is None
                    else self._cut(s["routes"].truncate(plen + m), len(s["key"])),
                )
                if m > 0 and engine_q_ready(d):
                    flat.append((len(flat), d))
                    rows.append(
                        torch.cat([prompt, torch.tensor(d[2], dtype=torch.long)])
                    )
                    owners.append(s)
                    continue
                self._release_hidden([d])
                if done:
                    # Finished, but its remaining tokens cannot be verified (a
                    # logging gap); nothing more will be logged: start again
                    # from the prefix. A running stream just waits.
                    stale.append(s)
            if stale:
                for s in stale:
                    retries[id(s["plan"])] += 1
                    if retries[id(s["plan"])] > max_retries:
                        raise RuntimeError(
                            "sticky routing: a continuation was not logged "
                            f"{max_retries} times in a row"
                        )
                stats["retries"] += len(stale)
                stop(stale)
                start([(s["key"], s["plan"]) for s in stale])
            return flat, rows, owners

        async def apply(flat, owners, by_row, t_verify) -> None:
            """Extend or cut each owner's plan by its piece's verification result."""
            done_streams, restart = [], []
            n_rejects = 0
            for r, ((_, d), s) in enumerate(zip(flat, owners)):
                out = by_row[r]
                plan, p, toks, done = s["plan"], d[0], d[2], d[3]
                pieces_per_plan[id(plan)] += 1
                g, n_acc = len(toks), out["accepted"]
                lps = behavior_logprobs(
                    out["logprobs"], n_acc, d[6], out.get("draft_logq")
                )
                stats["tokens"] += g
                stats["kept"] += n_acc
                if "prefix_start" in out:
                    stats["prefix_rows"] += 1
                    stats["prefix_hits"] += int(out["prefix_start"] > 0)
                    stats["prefix_skipped"] += out["prefix_start"]
                start_pos = s["taken"]
                if n_acc == g:
                    plan.prefix += list(toks)
                    plan.prefix_logprobs += lps[:g]
                    plan.anchors = d[8].truncate(p.numel() + g)
                    s["taken"] += g
                    if self._horizon_done(plan, len(s["key"])):
                        # Anchor horizon: the refit engine continues this
                        # rollout (incl. a turn end here: ``finish_row``).
                        done_streams.append(s)
                    elif (bool(done) and self.anchor_horizon is None) or self._turn_ended(
                        s["key"], plan, plan.prefix[-1]
                    ):
                        done_streams.append(s)
                        end_turn(s["key"], plan)
                else:
                    n_rejects += 1
                    y = out["next"]
                    plan.prefix += list(toks[:n_acc]) + [y]
                    plan.prefix_logprobs += lps[: n_acc + 1]
                    plan.anchors = d[8].truncate(p.numel() + n_acc)
                    done_streams.append(s)
                    if self._horizon_done(plan, len(s["key"])):
                        pass
                    elif self._turn_ended(s["key"], plan, y):
                        end_turn(s["key"], plan)
                    else:
                        restart.append(s)
                if trace is not None:
                    trace.piece(plan, start_pos, g, n_acc, s["decoded"])
                    if plan.complete:
                        trace.completed(plan)
            if trace is not None:
                trace.round(len(live), len(flat), t_verify, n_rejects)
            if self.prefix_cache is not None:
                # Complete rollouts need no more pieces: free their learner state.
                self.prefix_cache.release(
                    s["plan"].prefix_key for s in owners if s["plan"].complete
                )
            if done_streams:
                stop(done_streams)
                # Rollouts waiting on these plans can be served and trained.
                await self._notify_tails()
            if restart:
                start([(s["key"], s["plan"]) for s in restart])

        try:
            while live or inflight or turns:
                progressed = False
                if turns:
                    n_turns = len(turns)
                    next_turns()
                    if len(turns) < n_turns:
                        progressed = True
                        # Rollouts that ended in the environment can be served.
                        await self._notify_tails()
                for task in [t for t in inflight if t.done()]:
                    flat, owners, t_sent = inflight.pop(task)
                    for s in owners:
                        busy.discard(id(s))
                    # Verification latency of the round (overlaps decoding).
                    await apply(flat, owners, task.result(), time.perf_counter() - t_sent)
                    progressed = True
                idle = [s for s in live if id(s) not in busy]
                if idle:
                    status = ray.get(rank0.draft_lengths.remote([s["idx"] for s in idle]))
                    cand = [
                        s
                        for s, (n, done) in zip(idle, status)
                        if done or n - s["taken"] >= self._piece_tokens(len(s["plan"].prefix))
                    ]
                    flat, rows, owners = pieces(cand) if cand else ([], [], [])
                    if flat:
                        stats["rounds"] += 1
                        task = asyncio.create_task(
                            self._verify_block_async(
                                flat, rows, self._prefix_keys([s["plan"] for s in owners])
                            )
                        )
                        inflight[task] = (flat, owners, time.perf_counter())
                        busy.update(id(s) for s in owners)
                        progressed = True
                if not progressed:
                    if inflight:
                        await asyncio.wait(
                            list(inflight), timeout=0.02, return_when=asyncio.FIRST_COMPLETED
                        )
                    else:
                        await asyncio.sleep(0.02)
        finally:
            for task in inflight:
                task.cancel()
            if self.prefix_cache is not None:
                self.prefix_cache.release(plan.prefix_key for _, plan in pending)
        if stats["prefix_rows"]:
            self.stats.update(
                {
                    "spec/prefix_cache_hit_frac": stats["prefix_hits"] / stats["prefix_rows"],
                    "spec/prefix_cache_tokens_skipped": float(stats["prefix_skipped"]),
                }
            )
        self.stats.update(
            {
                "spec/sticky_rounds": float(stats["rounds"]),
                # Verified continuation pieces per continued row (learner forwards of
                # the whole row each): mean and max.
                "spec/sticky_pieces_per_row": (
                    sum(pieces_per_plan.values()) / max(len(pieces_per_plan), 1)
                ),
                "spec/sticky_pieces_per_row_max": float(max(pieces_per_plan.values(), default=0)),
                "spec/sticky_streams": float(stats["streams"]),
                "spec/sticky_cont_draft_tokens": float(stats["tokens"]),
                "spec/sticky_cont_accepted_frac": stats["kept"]
                / max(stats["tokens"], 1),
                "spec/sticky_cont_retries": float(stats["retries"]),
                "spec/sticky_env_turns": float(stats["turns"]),
            }
        )
        if self.turn_env is not None:
            # Environment latency now sits before the refit (DESIGN 14m).
            self.stats["spec/sticky_env_s"] = self.turn_env.env_s
            self.turn_env.env_s = 0.0

    def _prefix_cache_begin(self) -> None:
        """Learner prefix-state cache: costs on first use; empty at each new step.

        The learner drops its cached states when its weights change (every
        optimizer step), so the directory starts empty for each step's rounds.
        """
        if self.prefix_cache is None:
            return
        if not self.prefix_cache.configured:
            costs = ray.get(
                self.verifiers[0].run_all_workers_single_data("prefix_state_cache_costs")
            )
            self.prefix_cache.configure(costs)
        if self._prefix_step != self.current_step:
            self.prefix_cache.reset()
            self._prefix_step = self.current_step

    def _prefix_keys(self, plans) -> Optional[list[str]]:
        """Cache keys of the plans whose pieces are verified (None: cache off)."""
        if self.prefix_cache is None:
            return None
        for plan in plans:
            if plan.prefix_key is None:
                plan.prefix_key = self.prefix_cache.new_key()
        return [plan.prefix_key for plan in plans]

    def claim_anchors(
        self, prompt: torch.Tensor, token_ids: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Anchors of a finished speculative rollout (sticky routing), or None.

        ``prompt``: the generation prompt; ``token_ids``: the trajectory. None
        for rollouts that were never drafted: anchor = self, plain routing.
        """
        return self._anchor_claims.claim(prompt, token_ids)

    def claim_log_weight(self, prompt: torch.Tensor, token_ids: torch.Tensor) -> float:
        """Banded IS: log loss weight of a finished rollout (0 if it has none).

        ``prompt``: the generation prompt; ``token_ids``: the trajectory. Rows
        with equal verified prefixes are interchangeable here: their weights
        are a function of that prefix (w = exp(S_T), T inside the prefix).
        """
        log_w = self._weight_claims.claim(prompt, token_ids)
        return 0.0 if log_w is None else float(log_w)

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
        if getattr(self, "_tail_cv", None) is not None:
            # Streaming over the tails: a drafted rollout is served once its
            # continuation rounds complete it; one never drafted decodes at
            # theta_k once the engine holds it.
            if id(plan) in getattr(self, "_tail_plans", ()):
                await self._await_plan(plan)
            if not plan.complete:
                # Never drafted, or (anchor horizon) continued at theta_k.
                await self._weights_ready.wait()
                if self._tail_error is not None:
                    raise RuntimeError("sticky tails or refit failed") from self._tail_error
        gen = list(plan.prefix)
        lps = list(plan.prefix_logprobs)
        if plan.fused is not None:
            self._fused_claims.add(key, gen, plan.fused)
        if self.banded_is_band is not None and plan.log_weight != 0.0:
            # Registered when the plan is taken (complete under sticky
            # routing): the weight is final at the first verification (every
            # later token is from p); the claim matches the trajectory by its
            # verified prefix.
            self._weight_claims.add(key, gen, plan.log_weight)
        if self.sticky_delta is not None and plan.anchors is not None:
            if not plan.complete and plan.engine_key is None:
                # Its continuation would be decoded plainly by the refit
                # engine, which is not the sticky policy the prefix was
                # verified against.
                raise RuntimeError(
                    "sticky routing: a drafted rollout reached the rollout loop "
                    "incomplete (verify_sticky must complete every plan)"
                )
            self._anchor_claims.add(key, gen, plan.anchors)
        if plan.verified is not None:
            self._row_claims.add(key, gen, RowInfo(plan.tail, plan.log_weight))
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
            if plan.engine_key is not None:
                # Anchor horizon: the engine applies the plan's anchors below c.
                cont["ts_anchor_key"] = [plan.engine_key]
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
        return self._fused_claims.claim(prompt, token_ids)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)
