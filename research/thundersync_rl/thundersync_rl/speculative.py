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

import array
import asyncio
import itertools
import math
import os
import threading
import time
import traceback
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

import numpy as np
import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from thundersync_rl.engine_q import (
    free_hidden,
    hidden_num_rows,
)
from thundersync_rl.null_bv import LogRatioCalibration, NullGroup, NullPolicy, row_envelopes, select_pools
from thundersync_rl import vit
from thundersync_rl.run_state import SeedStream
from thundersync_rl.routes import (
    RouteRefs,
)
from thundersync_rl.envflag import env_flag


# Per-draft transport results of engine-q verification (worker -> driver).
HIDDEN_FETCH_KEYS = ("hidden_bytes_fetched", "hidden_real_bytes_fetched", "hidden_fetch_s", "q_recomputed",
                     "ds2_rows", "fwd_group_rows", "ds2_dg_checked",  # engagement / tripwire counters (rows)
                     "ds2_vw_checked", "ds2_vw_mismatch")  # THUNDERSYNC_VWALL_CHECK: rows re-run on the old path
# Smallest continuation piece the run cap (``SpeculativeGeneration._piece_tokens``) cuts to: the production
# fixed chunk (thundersync.yaml sticky_continuation_chunk_tokens 128) is never capped.
STICKY_MIN_PIECE = 128


def lpt_owners(lengths: list[int], ranks: int, fixed: Optional[dict[int, int]] = None) -> list[int]:
    """Rank of each row: ``fixed`` rows stay; the rest longest first onto the least-loaded rank.

    Placement only: which rank forwards a row does not enter any decision. (Moved from the async branch's
    prefix_cache.py, which the colocated branch does not carry.)
    """
    load = [0] * ranks
    owner = [0] * len(lengths)
    fixed = fixed or {}
    for i, r in fixed.items():
        owner[i] = r
        load[r] += lengths[i]
    for i in sorted((i for i in range(len(lengths)) if i not in fixed), key=lambda i: (-lengths[i], i)):
        r = min(range(ranks), key=lambda r: (load[r], r))
        owner[i] = r
        load[r] += lengths[i]
    return owner


def _null_result(res: dict[str, Any], m: Optional[float]) -> dict[str, Any]:
    """A ⊥-BV row's result at envelope ``m`` from its grid (None: unused)."""
    base = {k: res.get(k, 0) for k in HIDDEN_FETCH_KEYS}
    base["lr_moments"] = res.get("null_lr_moments", (0, 0.0, 0.0))
    # Per-segment (draft version) and per-position moments (drift_log.py).
    base["lr_seg_moments"] = res.get("lr_seg_moments", ())
    base["lr_pos_moments"] = res.get("lr_pos_moments", ())
    # Exact top-p: drafted tokens outside the target nucleus (log p = -inf).
    base["p_zero_tokens"] = int(res.get("p_zero_tokens", 0))
    if res.get("null_empty"):
        base["null_empty"] = True  # an empty draft decided by its keyed coin (no plan: a fresh decode)
    if m is None:
        return {**base, "null": "unused", "accepted": 0, "next": -1, "logprobs": []}
    tau, y, lp_y = res["null_grid"][m]
    if "null_grid_pit" in res:
        # The emission-law PIT at the envelope this row is decided at (law_check, THUNDERSYNC_LAW_CHECK).
        from thundersync_rl import law_check

        law_check.record_null_emission(res["null_grid_pit"].get(m))
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


def null_wave_mult() -> float:
    """``THUNDERSYNC_NULL_WAVE_MULT`` (F3 lazy pool walk; unset = today's waves): a ⊥-BV wave forwards
    ceil(s * min(M, mult)) rows (s slots left, M the envelope the counts give). mult = 1: exactly the s rows the
    slots can still take (fewest unused verified drafts, more waves). A constant: wave sizes stay functions of the
    counts only, so the decisions are those of the one-shot process (``_verify_null``'s argument)."""
    v = os.environ.get("THUNDERSYNC_NULL_WAVE_MULT", "").strip()
    if not v:
        return math.inf
    x = float(v)
    if not x >= 1.0:
        raise ValueError(f"THUNDERSYNC_NULL_WAVE_MULT must be >= 1 (got {v!r})")
    return x


def null_wave_size(s: int, m: float) -> int:
    """Rows of the next ⊥-BV wave of a pool with s slots left at envelope m (before the queue bound)."""
    return math.ceil(s * min(m, null_wave_mult()))


def row_seeds_enabled() -> bool:
    """``THUNDERSYNC_ROW_SEEDS=1`` (F3): every verify0 row draws its uniforms from its own generator, seeded by
    (step, the row's canonical identity) only, so its decision does not depend on which call, rank or position it
    is verified in (verification order and placement move latency only)."""
    from thundersync_rl.envflag import env_flag

    return env_flag("THUNDERSYNC_ROW_SEEDS")


class _NullPoolRun:
    """⊥-BV waves of one prompt's pool (``_verify_null`` / streaming verify0).

    ``rows``: the pool's flat row indices in drafting order. ``next_wave``
    gives the next ceil(s * M) rows to forward (s slots left, M the envelope
    the counts give now; none once the pool is decided); ``decide`` takes one
    forwarded row's grid result at the M the count process gives it, rows in
    drafting order; ``close`` returns the rows never forwarded (unused).
    Everything that selects rows or envelopes is a function of counts, so a
    pool decides the same way whatever call its waves ride in.
    """

    def __init__(self, rows: list[int], slots: int, policy: NullPolicy, one_wave: bool = False):
        self.one_wave = one_wave  # forward every draft of the pool in its first wave (nolanes: one call per chunk)
        self.queue = list(rows)
        self.state = NullGroup(len(rows), slots, policy)
        self.env = {i: row_envelopes(policy, len(rows), slots, j) for j, i in enumerate(rows)}
        self.pos = {i: j for j, i in enumerate(rows)}  # drafting-order position in the pool (empty-draft coin key)
        self.out: dict[int, dict[str, Any]] = {}
        self.inflight = 0
        self.waves = 0

    def next_wave(self) -> list[int]:
        if self.inflight:
            return []
        m = self.state.next_m()
        if m is None or not self.queue:
            return []
        k = len(self.queue) if self.one_wave else min(len(self.queue), null_wave_size(self.state.s, m))
        batch, self.queue = self.queue[:k], self.queue[k:]
        self.inflight = len(batch)  # decided before the next wave
        self.waves += 1
        return batch

    @property
    def decided(self) -> bool:
        """No row left to forward or decide (``next_wave`` is empty, none in flight)."""
        return not self.inflight and (not self.queue or self.state.next_m() is None)

    def decide(self, i: int, r: dict[str, Any], drop_is: bool, drafted: int) -> None:
        self.inflight -= 1
        m = self.state.next_m()
        tail_drop = False
        if m is not None:
            tau, y, _ = r["null_grid"][m]
            output = y >= 0
            # Drop-IS: an output rejected inside its draft is dropped.
            tail_drop = drop_is and output and tau < drafted
            self.state.record(output, tail_dropped=tail_drop)
        self.out[i] = _null_result(r, m)
        if tail_drop:
            self.out[i]["null"] = "tail_dropped"

    def close(self) -> list[int]:
        self.state.next_m()  # a pool left short after its last draft is dropped too
        skipped, self.queue = self.queue, []
        return skipped


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


def _prefill_check_rows_cap() -> int:
    """Rows per step of the engine prefill check (THUNDERSYNC_ENGINE_PREFILL_CHECK_ROWS)."""
    return int(os.environ.get("THUNDERSYNC_ENGINE_PREFILL_CHECK_ROWS", "8"))


@dataclass
class Plan:
    """How one rollout of the current iteration will be produced."""

    seed: int
    prefix: list[int] = field(default_factory=list)  # tokens already fixed
    prefix_logprobs: list[float] = field(default_factory=list)
    complete: bool = False
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
    # verification state under (continuation pieces).
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
    turns: list[Any] = field(default_factory=list, compare=False)
    row: Optional[int] = None
    # Verify-in-train (``vit.py``): a placeholder of a VIT group, filled once its
    # training forward decides (or its fallback verification does).
    vit: bool = field(default=False, compare=False)
    # Anchor epochs (aep.py): the weight version whose plain routes anchor this row and whose engine drafted
    # it (the draft's version tag); None for rows the engine sampled at the target version.
    anchor_version: Optional[int] = field(default=None, compare=False)
    # Anchor epochs: a slot no draft filled (or a forced-fresh row), completed as a continuation from the
    # empty prefix on the anchor engine.
    aep_fill: bool = field(default=False, compare=False)
    # Sticky continuation pieces of this plan rejected so far this step (``_continue_sticky``): with the
    # decided prefix length, the history that caps the plan's next piece sizes (``_piece_tokens``).
    cont_rejects: int = field(default=0, compare=False)


@dataclass(frozen=True)
class RowInfo:
    """Verification facts of one rollout the trainer may use (``claim_row``)."""

    tail: bool
    log_weight: float




def wide_fresh_salt(seed: int) -> int:
    """Salt of the wide sampler's stable fresh-row keys (worker.install_wide_sampler): a function of grpo.seed
    only (not of the lane or the resume segment), so a resumed run draws the uninterrupted run's fresh rows."""
    import zlib as _zlib

    return _zlib.crc32(f"wide/{int(seed)}/fresh".encode()) | ((int(seed) & 0x7FFFFFFF) << 32)


def _key(ids: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(t) for t in ids.tolist())


def empty_draft_uniform(seed: int, stream: str, step: int, pool_key: tuple[int, ...], j: int) -> float:
    """⊥-BV coin of an empty (or unlogged) draft: a uniform in [0, 1) from ("bv-empty", seed, stream, step, the pool's
    prompt key, the draft's position j in its pool's drafting order) only; never from any draft's tokens or timing."""
    import hashlib

    h = hashlib.sha256()
    h.update(f"bv-empty/{int(seed)}/{stream}/{int(step)}/{int(j)}/".encode())
    h.update(array.array("q", [int(t) for t in pool_key]).tobytes())
    return (int.from_bytes(h.digest()[:8], "little") >> 11) * 2.0**-53


def empty_draft_result(u: float, envelopes) -> dict[str, Any]:
    """⊥-BV of a zero-length draft against p/M (+) (1 - 1/M)⊥ at its root: ⊥ with probability 1 - 1/M, else an output
    whose every token is a fresh p decode (no prefix, no residual token: the slot decodes fresh at theta_k). The grid
    gives (tau 0, y 0 = output / -1 = ⊥, log p 0.0) at every envelope the row can be decided at; ``null_empty`` marks
    the row so no plan carries the placeholder y."""
    grid = {float(m): (0, 0 if u < 1.0 / float(m) else -1, 0.0) for m in envelopes}
    return {"null_grid": grid, "token_logprobs": [], "draft_logq": [], "null_empty": True}


def _debug_forced_fresh(key: tuple[int, ...]) -> bool:
    """Debug (THUNDERSYNC_DEBUG_FRESH_MOD=m): prompts whose token sum is 0 mod m ignore
    their drafts and decode fresh on the engine at the current weights, so every step
    has engine-sampled rows for the exactness checks (drafted rows carry the learner's
    own log-probs). The choice depends on the prompt only, so outputs stay exact.
    """
    m = int(os.environ.get("THUNDERSYNC_DEBUG_FRESH_MOD", "0") or 0)
    return m > 0 and sum(key) % m == 0






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


# Ordered call numbers (``SpeculativeGeneration.learner_seq``): one counter per
# worker group, kept on the group object, so every SpeculativeGeneration of the
# driver (one per engine lane) numbers the shared learner group's calls from one
# sequence, and each group's ranks (``ordered_call``) see 0, 1, 2, ... with no
# gap (a call numbered from another group's counter would leave one forever).
def _group_seq(group) -> int:
    counter = group.__dict__.get("_ts_ordered_seq")
    if counter is None:
        counter = group.__dict__.setdefault("_ts_ordered_seq", itertools.count())
    return next(counter)

@ray.remote(num_cpus=0)
class ReplyMailbox:  # pragma: no cover - exercised through Ray
    """Results of posted learner calls (``verify_post``), delivered off the learner actors' reply path.

    Each learner rank posts ``[ref]`` (its call's results, ``ray.put``) under the call's key when the call ends;
    ``take(key, n)`` returns the n refs ordered by rank once all n ranks posted. Asynchronous and tiny: it never
    waits on a learner's event loop."""

    def __init__(self) -> None:
        self._got: dict[str, dict[int, Any]] = {}
        self._cv = asyncio.Condition()

    async def post(self, key: str, rank: int, ref_list: list) -> None:
        async with self._cv:
            self._got.setdefault(key, {})[int(rank)] = ref_list[0]
            self._cv.notify_all()

    async def take(self, key: str, n: int) -> list:
        async with self._cv:
            while len(self._got.get(key, ())) < n:
                await self._cv.wait()
            got = self._got.pop(key)
        return [got[r] for r in sorted(got)]


_REPLY_MAILBOX: list = []


def reply_mailbox() -> Any:
    """The driver's ReplyMailbox (made once)."""
    if not _REPLY_MAILBOX:
        _REPLY_MAILBOX.append(ReplyMailbox.remote())
    return _REPLY_MAILBOX[0]


def await_posted_replies(refs: list, mailbox: Any, key: str) -> list:
    """The per-rank results of a posted call: from the mailbox as soon as every rank posted; a rank whose call
    raised surfaces its error through the call's own ref (the mailbox never gets that rank's post)."""
    take = mailbox.take.remote(key, len(refs))
    pending = list(refs)
    while True:
        ready, _ = ray.wait([take] + pending, num_returns=1)
        if take in ready:
            break
        for r in ready:
            ray.get(r)  # None on success; raises the rank's exception
            pending.remove(r)
    return ray.get(ray.get(take))


@dataclass
class EngineVerifier:
    """The engine that verifies this step's drafts (``verify_on="engine"``).

    ``group``: its generation worker group (holds the current weights);
    ``q_lane``: [rank-0 actor, worker actors] of the lane that drafted the
    cohort and still holds its drafting weights (``hidden_transport=
    "recompute"``: q rows are rebuilt there), or None (q from logged rows).
    """

    group: Any
    q_lane: Optional[list[Any]] = None
    # Calls the first verification of the step must wait for (LM-head stash
    # and import on the verifier lane), sent at the top of the step.
    wait_refs: list = field(default_factory=list)

    @property
    def workers(self) -> list[Any]:
        return list(self.group.workers)


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
        draft_admission: str = "group",
        sticky_chunk_tokens: int = 2048,
        sticky_chunk_growth: float = 0.0,
        sticky_run_factor: float = 4.0,
        sticky_decision: str = "every",
        sticky_decision_dip: float = 0.1,
        draft_until_complete: bool = False,
        window_pace: Optional[dict[str, float]] = None,
        draft_weights: str = "current",
        max_seq_len: Optional[int] = None,
        anchor_horizon: Optional[int] = None,
        banded_is_band: Optional[float] = None,
        learner_prefix_cache: Optional[dict[str, float]] = None,
        learner_prefix_cache_verify0: bool = False,
        seed: int = 0,
        seed_stream: str = "0",
        engine_sampler: str = "engine",
        engine_max_len: Optional[int] = None,
        sampler_segment: int = 0,
        wide_key64: bool = True,
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
        # Engine tokens per streamed partial frame of a draft. The frames feed only rank 0's readiness counts
        # (``draft_lengths``: when a continuation piece is long enough to cut) and the finished flag (final frame,
        # which carries the merged record); every cut reads its contents from the engines' request records
        # (``draft_records`` -> ``peek_drafts`` / ``collect_drafts``) and its hidden rows / routes from the logs'
        # frontiers, never from partial frames. 64 (was 16): a quarter of the per-step partial building on the
        # engine and frame handling on rank 0's loop; a piece is noticed up to 63 tokens later (pieces are
        # thousands of tokens).
        self.stream_interval = 64
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
        if fused_verify_train or fused_max_retained_tokens:
            raise ValueError("fused_verify_train: fused verification + training is not on the colocated branch")
        self.fused_verify_train = False
        # Where stash + block verification run: "learner" or "inference" (the
        # generation workers hold theta_k until the refit and theta_{k+1} after
        # it, and sit idle while rollouts wait for verification).
        # verify_on="engine" (engine_verify.py): drafts and continuation pieces
        # are verified by an engine holding the current weights
        # (``engine_verifier``, set per step); the learner pool stays for the
        # learner fallback (rows longer than the engine's length cap) and the
        # THUNDERSYNC_ENGINE_VERIFY_CHECK comparison.
        if verify_on != "learner" or verify_source != "engine" or verify_mode != "block" or draft_weights != "current":
            raise ValueError(
                "the colocated branch verifies blocks on the learner against engine-sourced q of drafts sampled at "
                "the current weights (verify_on=learner, verify_source=engine, verify_mode=block, draft_weights=current)"
            )
        self.verify_on = verify_on
        self.engine_verifier: Optional[EngineVerifier] = None
        self.engine_max_len = engine_max_len
        self._ev_ids = itertools.count()
        groups = {
            "learner": [learner_policy.worker_group],
            "engine": [learner_policy.worker_group],
            "inference": [base._policy.worker_group],
            # Split rows across both pools: all GPUs are idle at verification.
            "both": [learner_policy.worker_group, base._policy.worker_group],
        }[verify_on]
        self.verifiers = groups
        self.verifier = groups[0]
        # Weights the engine drafts with: "current" = theta_k;
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
        if engine_sampler not in ("wide", "engine", "exact_int"):
            raise ValueError(f"unknown engine_sampler {engine_sampler!r}")
        self.engine_sampler = engine_sampler if verify_mode != "keyed" else "keyed"
        # Run segment of the keyed engine samplers (ts-nano-scale): keys are
        # (salt, engine request id, token index) and engine request ids restart at 0 in a
        # new process, so a resumed run (segment = its first step > 0) must not reuse the
        # salt of the run it continues: the same Gumbel / integer words would drive the
        # resumed rollouts. Segment 0 keeps the old salt (fresh runs unchanged, bitwise).
        seg = f"/seg{int(sampler_segment)}" if int(sampler_segment) else ""
        self.sampler_segment = int(sampler_segment)
        if self.engine_sampler == "wide":
            # Before install_draft_hidden_log below (it wraps the kernel installed now).
            import zlib as _zlib

            salt = _zlib.crc32(f"wide/{seed}/{seed_stream}{seg}".encode()) | (
                (int(seed) & 0x7FFFFFFF) << 32
            )
            ok = ray.get(
                workers.run_all_workers_single_data(
                    "install_wide_sampler", salt=salt, key64=bool(wide_key64), fresh_salt=wide_fresh_salt(seed)
                )
            )
            if not any(x is not None for x in ok):
                raise RuntimeError("engine_sampler=wide: no engine rank installed the sampler")
        if tail_lookup:
            ray.get(workers.run_all_workers_single_data("install_lookup_references"))
        ok = ray.get(
            workers.run_all_workers_single_data(
                "install_draft_hidden_log",
                seal_blocks=hidden_transport == "object_store",
            )
        )
        if not any(x is True for x in ok) or any(x is False for x in ok):
            # Was a print + silent switch to verify_source=scorer (another q
            # source and cost than the validated config; also hit when the
            # engine was not initialized yet). The config asked for engine q.
            raise RuntimeError(
                "verify_source=engine: the engine cannot log hidden states "
                f"(install_draft_hidden_log per rank: {ok}); set "
                "thundersync.verify_source=scorer explicitly to run the scorer path"
            )
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
        # After a continuation rejection, pieces are capped at this many mean accepted runs of the plan
        # (``_piece_tokens``; 0: no cap).
        self.sticky_run_factor = float(sticky_run_factor)
        # Carried-walk decisions: None keeps every piece a block.
        from thundersync_rl.carried_walk import DecisionRule

        rule = DecisionRule(sticky_decision, float(sticky_decision_dip))
        self.cw_rule = None if rule.mode == "every" else rule
        # Learner prefix-state cache: continuation pieces are
        # forwarded from the row's cached state on the learner rank that holds
        # it ({"gpu_gb", "max_gb"} per learner rank; None: off).
        if learner_prefix_cache is not None:
            raise ValueError("learner_prefix_cache: the learner prefix-state cache is not on the colocated branch")
        self.prefix_cache = None
        self._prefix_step: Optional[int] = None
        # verify0 capture: verify0's whole-row forwards keep the rows' states
        # so a continued row's first piece hits the cache. id(draft) -> (key, draft) of the
        # captures not settled yet; key -> the plan made from that draft.
        self.prefix_cache_verify0 = bool(learner_prefix_cache_verify0) and self.prefix_cache is not None
        self._v0_cap: dict[int, tuple[str, Any]] = {}
        self._v0_plan: dict[str, Plan] = {}
        self._v0_step_keys: list[str] = []
        # Measurement only: the deadline waits for every draft to finish.
        self.draft_until_complete = draft_until_complete
        # Drafting-window pacing (window_pace.py): the cut holds a target
        # window fitted from earlier cuts (sticky streams only).
        if window_pace is not None:
            raise ValueError("window_pace: drafting-window pacing (lanes) is not on the colocated branch")
        self._pacer = None
        self._cont_ids = itertools.count()
        # Long runs: rank 0's draft-table entries per owning cohort, {owner step: {table gen:
        # [indices]}}, released once the owner is trained (``release_consumed``). ``_draft_gen`` mirrors rank 0's
        # table generation (one per ``start_drafts`` this wrapper sends).
        self._stream_owner: dict[int, dict[int, list[int]]] = {}
        self._draft_gen = 0
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
        self.turn_env: Optional[Any] = None
        self._row_prompts: list[torch.Tensor] = []
        self._row_plans: dict[int, Plan] = {}
        # Route blocks by the step whose training replays them (freed by
        # ``release_routes`` once that step has trained).
        self._route_blocks: dict[int, list] = defaultdict(list)
        self._route_layers: list[int] = []
        # Pooled null-augmented block verification (null_bv.py): the engine
        # drafts K G drafts per prompt; verification fills the G slots of each
        # prompt with the first non-BOT outcomes in drafting order. Target
        # step -> {prompt key: slots} of the drafts made for that step.
        self.null_bv = null_bv
        self.null_bv_policy = dict(null_bv_policy or {})
        calibrate = self.null_bv_policy.pop("calibrate", False)
        decay = self.null_bv_policy.pop("calibration_decay", 0.5)
        # Built once: its DP table is memoised across steps (rebuilt per step
        # when calibrated: ``_verify_null``).
        self._null_policy = NullPolicy(**self.null_bv_policy) if null_bv else None
        # Calibrated ⊥-BV policy: per-token Var log p/q from earlier steps'
        # verified drafts only (null_bv.LogRatioCalibration: exactness).
        self._null_calib = (
            LogRatioCalibration(self.null_bv_policy.get("logratio_var", 8e-7), decay)
            if null_bv and calibrate
            else None
        )
        # Drop-IS (opt-in, unbiased gradient, not exact samples): outputs that
        # need a tail decode are discarded like BOT; kept rows are weighted.
        self.null_bv_drop_is = null_bv_drop_is
        # Prompt drop: the step trains on the first (prompts - spares) prompts
        # whose pools were not dropped (``null_selected_keys``, drafting order).
        self.null_bv_spare_prompts = null_bv_spare_prompts
        # Engine admission order of ⊥-BV drafts (``next_draft_prompts``).
        self.draft_admission = draft_admission
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
        # Verification randomness (eta, emissions) and keyed draft seeds:
        # hash(grpo.seed, stream, step, counter), never reused across calls,
        # steps, lanes or a resume (the counters are checkpointed:
        # ``seed_state`` / ``load_seed_state``).
        self.sentinel = None  # sentinel.WeightSentinel (set by the training loop)
        self._seeds = SeedStream(seed, f"draft/{seed_stream}", lambda: self.current_step)
        self._empty_coin_seed = (int(seed), str(seed_stream))
        self._verify_seeds = SeedStream(
            seed, f"verify/{seed_stream}", lambda: self.current_step
        )
        # Verify-in-train (``vit.py``, opt-in; set by the training loop).
        self.verify_in_train = False
        self.vit_group_size = 0
        self._vit_groups: list = []
        self.vit_mbs = 0
        self.vit_dp = 1
        self.vit_min_pass_prob = 0.0
        self.vit_checkpoint_bytes = float("inf")
        self.vit_bytes_per_token = 0.0
        self.vit_predictor = vit.PassPredictor()
        self.plans: dict[tuple[int, ...], list[Plan]] = defaultdict(list)
        self.stats: dict[str, float] = {}
        # Multi-iteration drafts: target step -> drafts for that step's rollouts.
        # A draft keeps growing across iterations (resumed from its prefix under
        # whatever weights the engine holds); the learner stores, per position,
        # the q of the weights that drafted it.
        self.cohorts: dict[int, list[dict[str, Any]]] = {}

    # -- drafting (iteration k, for iteration k+1's prompts) --------------------











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






    def _count_hidden(self, name: str, nbytes: int) -> None:
        self.stats[name] = self.stats.get(name, 0.0) + float(nbytes)


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



    # -- verification (start of iteration k+1, learner holds theta_{k+1}) -------

    @staticmethod
    def _flat_rows(drafts):
        flat = [(t, d) for t, variants in enumerate(drafts) for d in variants]
        rows = [
            torch.cat([d[0], torch.tensor(d[2], dtype=torch.long)]) for _, d in flat
        ]
        return flat, rows





    # Past-only policy state saved at a checkpoint: plain data attributes only.
    _POLICY_STATE = {
        "_null_calib": ("value", "last_step", "verify_row_s", "token_s", "by_age", "age_step"),
        "_la_policy": ("mode", "last_forecast", "switches"),
        "vit_predictor": ("steps", "current"),
        "_pacer": ("full_window_s",),
    }

    def policy_state(self) -> dict[str, Any]:
        """Past-only choices' state (⊥-BV log-ratio calibration, VIT pass predictor, window pacer).

        Exactness never depends on it (each is a function of earlier steps only); saved so a
        resumed run makes the same choices as the uninterrupted one. Shared objects (lanes share one calibration)
        are saved per wrapper and loaded into the same shared object (identical values).
        """
        import copy

        out = {}
        for name, fields in self._POLICY_STATE.items():
            obj = self.__dict__.get(name)
            if obj is not None:
                out[name] = {f: copy.deepcopy(getattr(obj, f, None)) for f in fields}
        return out






    def end_step(self) -> int:
        """Drop the plans left over by this step (verified against its policy only).

        Plans are otherwise replaced at the next verification; a step without
        drafts (before a checkpoint / validation boundary) has none, and a later
        row with the same prompt tokens (duplicate prompts, the next epoch) must
        never claim a plan verified against an older version. Returns the count.
        """
        n = sum(len(q) for q in self.plans.values())
        self.plans.clear()
        # ⊥-BV prompt drop: the next step's selection comes from its own pools
        # (none when it has no drafts: its first prompts are trained).
        self.null_selected_keys = None
        return n



    def _engine_q_filter(self, drafts):
        """Engine-sourced q: drop drafts the engine did not fully log (counted)."""
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

        Empty drafts (cut before their first token, or unlogged) stay in their
        pools and are walked at their drafting index: ⊥-BV of a zero-length
        draft is ⊥ with probability 1 - 1/M_j, else a fresh p decode, drawn
        from a keyed coin (``empty_draft_uniform``), never forwarded. Dropping
        them instead made a pool's draft count (and prompt drop) a function of
        slot timing, which other drafts' contents move.
        """
        slots_by_key = self._null_slots.pop(self.current_step, None)
        if not self.null_bv or slots_by_key is None:
            return None
        kept, empty = [], []
        for variants in drafts:
            empty += [d for d in variants if len(d[2]) == 0]
            if variants:
                kept.append(list(variants))
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
        hidden = [d[7] for _, d in flat]
        extra = {"engine_logq": [d[6] for _, d in flat], "hidden": hidden}
        return "verify_drafts_block_engine", extra

    def _verified_hidden(self, flat, by_row: dict[int, Any], a: int = 0) -> None:
        """Count fetched hidden bytes of verified rows a + j and free their handles."""
        if self.hidden_transport == "object_store":
            for k in HIDDEN_FETCH_KEYS:
                self._count_hidden(
                    "spec/" + k, sum(r.get(k, 0) for r in by_row.values())
                )
        self._release_hidden(flat[a + j][1] for j in by_row)

    def _v0_batch_tokens(self) -> Optional[int]:
        """Padded tokens per learner forward of verify0 (``thundersync.verify0_batch_tokens``).

        verify0 (the step-top block verification of every draft, one-shot or
        streamed) forwards whole 8k drafts; at 16384 tokens most
        go one per forward and the learner forward is launch-bound (~0.4 s per
        call at full Nano), so verify0 packs more rows per forward. Continuation
        pieces keep ``verify_batch_tokens``. None: ``verify_batch_tokens``.
        """
        v = self.__dict__.get("verify0_batch_tokens")
        return int(v) if v else None

    def _v0_kw(self) -> dict[str, int]:
        """``batch_tokens`` for a verify0 call, only when the knob is set (default calls unchanged)."""
        v = self._v0_batch_tokens()
        return {} if v is None else {"batch_tokens": v}

    # -- EDV (edv.py, DESIGN §22.9): prepare, η seeds, submit a subset, re-walk one pool -------------------------






    def verify_post(self, drafts: list[list[tuple[torch.Tensor, int, list[int]]]]) -> dict[str, Any]:
        """Post the first ⊥-BV wave of these pooled drafts to the learner and return at once (the handle for
        ``verify_collect``). Several posts may be outstanding: the learner runs the calls in posting order (Ray
        actor order) back to back, with no driver round trip between them. Row seeds (F3) make every decision a
        function of the row alone, so the call a row rides in moves latency only. Anything but learner-verified
        pooled drafts falls back to ``verify`` at collect time."""
        t_v0 = time.perf_counter()
        drafts, dropped = self._engine_q_filter(drafts)
        null = self._null_prepare(drafts)
        if null is None:
            raise RuntimeError("verify_post: the drafts have no ⊥-BV slots for this step (set _null_slots first)")
        drafts, names, slots, n_empty = null
        flat, rows = self._flat_rows(drafts)
        self._record_siblings(flat)
        groups = [names[_key(d[0])] for _, d in flat]
        self._update_null_calibration()
        pools = self._null_pools(flat, groups, slots)
        batch = self._null_wave(pools, names, self._null_all_slots)
        h = dict(flat=flat, rows=rows, groups=groups, slots=slots, dropped=dropped, n_empty=n_empty, pools=pools,
                 batch=batch, names=names, all_slots=self._null_all_slots, t_post=time.perf_counter(), futs=None)
        fwd = [i for i in batch if len(flat[i][1][2]) > 0]  # empty drafts: decided by their coin at collect
        if fwd:
            h["futs"] = self._submit_verify_block(
                [flat[i] for i in fwd], [rows[i] for i in fwd],
                null_envelopes=[pools[groups[i]].env[i] for i in fwd], posted=True, **self._v0_kw(),
                **self._v0_cap_kw(),
            )
        self._vprof_add("prepare_s", time.perf_counter() - t_v0)
        self._vprof_add("submit_s", time.perf_counter() - h["t_post"])
        return h

    def verify_collect(self, h: dict[str, Any]) -> None:
        """Finish a ``verify_post``: the first wave's results, any later waves (blocking, as ``_verify_null``), the
        plans and statistics. Collect handles in posting order."""
        if "sync" in h:
            self.verify(h["sync"])
            return
        self.null_dropped_keys, self._null_dropped = set(), set()
        self.null_selected_keys = None
        self._vit_groups = []
        flat, rows, groups, pools = h["flat"], h["rows"], h["groups"], h["pools"]
        self._null_names, self._null_all_slots = h["names"], h["all_slots"]
        calib = self.__dict__.get("_null_calib")
        t_verify, n_forwarded, waves = h["t_post"], 0, 0
        batch, futs = h["batch"], h["futs"]
        edv_done = h.get("edv_done") or {}
        while batch:
            waves += 1
            # EDV: only the submitted rows (``sub``) were forwarded; the deferred ones carry their predicted results.
            sub = [i for i in batch if i not in edv_done] if (waves == 1 and h.get("sub") is not None) else batch
            sub = [i for i in sub if len(flat[i][1][2]) > 0]  # empty drafts: their keyed coins, never forwarded
            n_forwarded += len(sub)
            tw = time.perf_counter()
            if futs is not None:
                got = [
                    (a, ray.get(refs)) if not isinstance(a, tuple) else (a[0], await_posted_replies(refs, reply_mailbox(), a[1]))
                    for a, refs in futs
                ]
                t2 = time.perf_counter()
                res = self._verify_block_results([flat[i] for i in sub], got) if sub else []
                self._vprof_add("get_s", t2 - tw)
                self._vprof_add("results_s", time.perf_counter() - t2)
                self._vprof_add("calls", 1.0)
            else:
                res = self._verify_block(
                    [flat[i] for i in sub], [rows[i] for i in sub],
                    null_envelopes=[pools[groups[i]].env[i] for i in sub], **self._v0_kw(), **self._v0_cap_kw(),
                ) if sub else []
            ta = time.perf_counter()
            got_rows = {i: res[j] for j, i in enumerate(sub)}
            got_rows.update(self._null_empty_results(pools, groups, flat, [i for i in batch if not len(flat[i][1][2])]))
            if waves == 1:
                got_rows.update({i: r for i, r in edv_done.items() if i in batch})
                h["res"] = dict(got_rows)
            self._null_apply(pools, groups, flat, got_rows)
            w = min(waves, 3)
            self._vprof_add(f"wave{w}_s", ta - tw)
            self._vprof_add(f"wave{w}_rows", len(batch))
            self._vprof_add("apply_s", time.perf_counter() - ta)
            batch, futs = self._null_wave(pools, h["names"], h["all_slots"]), None
        by_row = self._null_close(pools, flat)
        self.stats["spec/null_waves"] = float(waves)
        if calib is not None:
            calib.update_verify(time.perf_counter() - t_verify, n_forwarded)
        t_v2 = time.perf_counter()
        self._finish_verify_null(flat, by_row, groups, h["slots"], h["dropped"], h["n_empty"])
        self._v0_settle()
        self._vit_place()
        self._count_tails()
        self._vprof_add("finish_s", time.perf_counter() - t_v2)

    # -- verify-in-train (``vit.py``) -------------------------------------------





    def _vit_place(self) -> None:
        """Placeholder plans the VIT groups' rollouts wait on (``_await_plan``)."""
        for vg in self._vit_groups:
            vg.placeholders = [Plan(seed=-1, vit=True) for _ in range(self.vit_group_size)]
            self.plans[vg.key] = list(vg.placeholders)







    def _null_selection(self, all_slots) -> tuple[dict, int]:
        """(slots per prompt, trained slots) of the prompt-drop selection, shared by ``_select_prompts`` and the lazy
        waves (``_null_wave``): the pools' slots with ``null_bv_spare_slots`` spare slots (the nolanes stage sets
        S x G: a duplicated prompt's pool holds all its groups), else one unit per prompt with
        ``null_bv_spare_prompts`` spare prompts."""
        spare_slots = self.__dict__.get("null_bv_spare_slots")
        if spare_slots is None:
            units = dict.fromkeys(all_slots, 1)
            return units, len(units) - int(self.null_bv_spare_prompts)
        slots = {k: int(v) for k, v in all_slots.items()}
        return slots, sum(slots.values()) - int(spare_slots)

    def _select_prompts(self) -> None:
        """Prompt drop: keep the undropped prompts in drafting order while they fit the trained slots (dropped ones
        fill a shortfall and decode fresh: ``null_bv.select_pools``); plans of the other prompts are discarded so
        their tails are never decoded."""
        slots, train = self._null_selection(self._null_all_slots)
        kept = select_pools(list(self._null_all_slots), slots, train, self.null_dropped_keys)
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





    def _verify_block(
        self,
        flat,
        rows,
        null_envelopes: Optional[list[tuple[float, ...]]] = None,
        band: Optional[float] = None,
        batch_tokens: Optional[int] = None,
        capture: bool = False,
    ) -> dict[int, dict[str, Any]]:
        """Block-verify ``flat`` drafts on every verifier pool; results by row.

        ``batch_tokens``: padded tokens per learner forward of this call
        (default ``verify_batch_tokens``; verify0 passes ``_v0_kw()``).

        Engine-sourced verification places rows on a pool's data-parallel
        ranks by tokens (``lpt_owners``). ``null_envelopes`` (⊥-BV): every row
        is verified at each of its envelopes (``null_envelopes[i]`` for row i);
        the driver decides (``_decide_null``).
        ``band``: banded IS (only a draft's first verification; never with
        ``null_envelopes``).
        """
        t0 = time.perf_counter()
        futs = self._submit_verify_block(
            flat, rows, null_envelopes, band=band, batch_tokens=batch_tokens, capture=capture
        )
        t1 = time.perf_counter()
        got = [(a, ray.get(refs)) for a, refs in futs]
        t2 = time.perf_counter()
        out = self._verify_block_results(flat, got)
        self._vprof_add("submit_s", t1 - t0)
        self._vprof_add("get_s", t2 - t1)
        self._vprof_add("results_s", time.perf_counter() - t2)
        self._vprof_add("calls", 1.0)
        return out

    def _vprof_add(self, key: str, v: float) -> None:
        """Driver-side verify timing (logging only): ``vprof/<key>`` summed over the step's calls."""
        k = "vprof/" + key
        self.stats[k] = self.stats.get(k, 0.0) + float(v)

    def _vprof_ranks(self, done) -> None:
        """Fold the ranks' call timing (``vprof_s`` on each rank's first result) into ``vprof/``: per call the
        slowest and fastest rank's call seconds (the call waits for the slowest), forwards, EP dummies, fetch waits."""
        per = []
        for _, ranks in done:
            for rank_res in ranks:
                for _, r in rank_res:
                    vp = r.pop("vprof_s", None)
                    if vp is not None:
                        per.append(vp)
        if not per:
            return
        # Per-call timeline (wall clock): the ranks' call entry / exit (nolanes.call_idle).
        tl = self.__dict__.setdefault("_vp_calls", [])
        tl.append({int(p.get("rank", -1)): (p.get("t_in", 0.0), p.get("t_out", 0.0)) for p in per})
        tot = [p.get("total", 0.0) for p in per]
        self._vprof_add("rank_total_max", max(tot))
        self._vprof_add("rank_total_min", min(tot))
        for key in ("forward_host", "v_setup", "ep_dummy_host", "hidden_fetch_s", "n_fwd", "n_dummy", "fwd_tokens"):
            self._vprof_add(f"rank_{key}_max", max(p.get(key, 0.0) for p in per))



    def _call_seed(self, kind: str, drafts) -> int:
        """Verification seed of one call (``_call_seed_ident``)."""
        return self._call_seed_ident(kind, drafts)[0]

    def _call_seed_ident(self, kind: str, drafts) -> tuple[int, list]:
        """Verification seed of one call, by canonical identity (not arrival).

        Block verification (verify0, ⊥-BV waves, streamed verify0 calls) and
        continuation-piece verification submit calls from different tasks and
        threads, so drawing ``next(self._verify_seeds)`` gave each call a seed
        that depended on which caller came first. The seed is now a hash of
        (grpo.seed, stream, step, kind, the call's rows), each row named by its
        key and its verification ordinal in this step (0 for its first call,
        1 for its next, ...). A row's calls are sequential (one piece in flight
        per stream; waves one after another), so its ordinals do not depend on
        timing, and every (row, ordinal) occurs in exactly one call: no two
        calls of a step share a seed (fresh randomness, as before). Rows without
        a key are named by (prompt length, crc32 of their tokens).
        Cost: O(rows) driver work per call; nothing on the ranks.
        """
        step = int(self.current_step)
        with self.__dict__.setdefault("_ord_lock", threading.Lock()):
            ords = self.__dict__.get("_row_ords")
            if ords is None or ords[0] != step:
                ords = self._row_ords = (step, {})
            table = ords[1]
            ident = []
            for d in drafts:
                rid = d[4] if len(d) > 4 and d[4] is not None else (
                    "crc",
                    int(d[0].numel()),
                    zlib.crc32(np.asarray(d[2], dtype=np.int64).tobytes()),
                )
                n = table.get(rid, 0)
                table[rid] = n + 1
                ident.append((rid, n))
        return self._verify_seeds.keyed(kind, tuple(ident)), ident

    def _call_seed_and_rows(self, kind: str, drafts) -> tuple[int, list[int]]:
        """``_call_seed`` plus per-row seeds (THUNDERSYNC_ROW_SEEDS): each row's seed is a function of (grpo.seed,
        step, the row's identity and verification ordinal) only, distinct per (row, ordinal) within a step, and
        independent of the call's other rows and of the row's position in it."""
        seed, ident = self._call_seed_ident(kind, drafts)
        return seed, [self._verify_seeds.keyed("v0row", rid, n) for rid, n in ident]

    def _submit_verify_block(
        self,
        flat,
        rows,
        null_envelopes=None,
        posted: bool = False,
        ordered=False,
        band=None,
        prefix_keys=None,
        row_seed_base=None,
        walks=None,
        walk_scans=None,
        pins=None,
        extra_kw: Optional[dict[str, Any]] = None,
        batch_tokens: Optional[int] = None,
        capture: bool = False,
    ) -> list:
        """Send ``flat``'s block verification to the verifier pools: [(first row, refs)].

        ``ordered``: as numbered learner calls while streaming over the tails.
        ``prefix_keys`` (learner prefix-state cache on): the rows' cache keys;
        a row goes to the rank that holds its state (``PrefixCacheDirectory``).
        ``batch_tokens``: padded tokens per learner forward (default
        ``verify_batch_tokens``). Packing only: each rank verifies the same rows
        in the same order with the same random stream for any value, and
        batch-invariant kernels give every row the same logits.
        ``capture`` (verify0 calls): with ``learner_prefix_cache_verify0`` the rows are
        forwarded under fresh cache keys (``_v0_capture_keys``).
        """
        bt = int(batch_tokens) if batch_tokens else self.verify_batch_tokens
        capture = capture and self._v0_capture_on() and row_seed_base is None
        # Contiguous row slices per verifier pool, verified concurrently.
        k = len(self.verifiers)
        cuts = [round(j * len(flat) / k) for j in range(k + 1)]
        futs = []
        for g, a, b in zip(self.verifiers, cuts, cuts[1:]):
            method, extra = self._verify_method(flat[a:b])
            want_rows = (
                row_seeds_enabled() and null_envelopes is not None and walks is None
                and walk_scans is None and row_seed_base is None and method == "verify_drafts_block_engine"
            )
            if want_rows:
                call_seed, row_seeds = self._call_seed_and_rows("block", [d for _, d in flat[a:b]])
            else:
                call_seed, row_seeds = self._call_seed("block", [d for _, d in flat[a:b]]), None
            if method == "verify_drafts_block_engine":
                # Rows placed on the pool's data-parallel ranks by tokens.
                lengths = [r.numel() for r in rows[a:b]]
                ranks = data_parallel_size(g)
                owners = lpt_owners(lengths, ranks)
                extra = {**extra, "row_ranks": owners}
                self._count_balance(lengths, owners, ranks)
            if null_envelopes is not None:
                extra = {**extra, "null_envelopes": [list(e) for e in null_envelopes[a:b]]}
            if walks is not None:
                extra = {**extra, "walks": walks[a:b]}
            if walk_scans is not None:
                extra = {**extra, "walk_scans": walk_scans[a:b]}
            if band is not None:
                extra = {**extra, "band": band}
            if row_seed_base is not None:
                if method != "verify_drafts_block_engine":
                    raise RuntimeError("verify_in_train needs engine-sourced q")
                extra = {**extra, "row_seed_base": int(row_seed_base)}
            if row_seeds is not None:
                extra = {**extra, "row_seeds": [int(x) for x in row_seeds]}
            if extra_kw:
                extra = {**extra, **extra_kw}
            reply_key = None
            if posted and not ordered and method == "verify_drafts_block_engine":
                # Posted ahead (verify_post): results return through the mailbox, not the actor reply (see
                # ReplyMailbox / the worker's reply_to).
                self._reply_seq = self.__dict__.get("_reply_seq", 0) + 1
                reply_key = f"v{os.getpid()}:{self._reply_seq}"
                extra = {**extra, "reply_to": (reply_mailbox(), reply_key)}
            seq = self.learner_seq(g) if ordered else None
            call = (method,) if seq is None else ("ordered_call",)
            if seq is not None:
                extra = {**extra, "seq": seq, "method": method}
            futs.append(
                (
                    a if reply_key is None else (a, reply_key),
                    g.run_all_workers_single_data(
                        *call,
                        **extra,
                        rows=rows[a:b],
                        prompt_lens=[d[0].numel() for _, d in flat[a:b]],
                        segments=[d[5] for _, d in flat[a:b]],
                        vocab_limit=self.vocab_limit,
                        # Fresh per call and step, by the call's canonical
                        # identity (``_call_seed``; ``seed * 7 + a`` once
                        # collided between consecutive calls, and a shared
                        # counter depended on which caller arrived first).
                        seed=call_seed,
                        keys=[d[4] for _, d in flat[a:b]],
                        precision=self.verify_precision,
                        batch_tokens=bt,
                        lazy=self.verify_lazy,
                    ),
                )
            )
            if seq is not None:
                self.learner_track(futs[-1][1])
        return futs







    def _verify_block_results(self, flat, done, keep_hidden: bool = False) -> dict[int, dict[str, Any]]:
        """Results by row from [(first row, per-rank results)] of ``_submit_verify_block``."""
        self._vprof_ranks(done)
        by_row = dict(
            (a + j, r) for a, ranks in done for rank_res in ranks for j, r in rank_res
        )
        missing = sorted(set(range(len(flat))) - by_row.keys())
        if missing:
            raise RuntimeError(f"block verification returned no result for rows {missing}")
        for r in by_row.values():
            cb = r.get("prefix_cache_bytes")
            if cb is not None:
                # Largest learner rank's prefix-state cache (device / pinned host) this step.
                for name, v in zip(("gpu", "host", "pool"), cb):
                    k = f"spec/prefix_cache_rank_{name}_gb_max"
                    self.stats[k] = max(self.stats.get(k, 0.0), float(v) / 2**30)
            pgr = r.get("prefix_cache_graph")
            if pgr is not None:
                # Cached forwards by graph outcome on the busiest-reporting rank (process totals).
                for name, v in zip(("calls", "replays", "captures", "eager"), pgr):
                    k = f"spec/prefix_cache_graph_{name}"
                    self.stats[k] = max(self.stats.get(k, 0.0), float(v))
        if keep_hidden:
            # Carried-walk pieces: the block's hidden rows stay until its decision.
            self._verified_hidden(flat, {})
            for k in HIDDEN_FETCH_KEYS:
                self._count_hidden("spec/" + k, sum(r.get(k, 0) for r in by_row.values()))
        else:
            self._verified_hidden(flat, by_row)
        return by_row

    def _count_balance(self, lengths: list[int], owners: list[int], ranks: int) -> None:
        """``spec/verify_load_max_over_mean``: busiest rank's tokens over the mean.

        Over the step's verification calls (each call waits for its busiest
        rank): ranks * sum(busiest) / sum(total). It used to be overwritten by
        every call, so the step reported its last call -- typically one small
        piece on one rank, i.e. exactly ``ranks`` (8.0 in every full-Nano ab /
        x2 history) -- which hid verify0's balance.
        ``spec/verify_load_first_max_over_mean``: the step's first call (verify0).
        """
        load = [0] * ranks
        for n, r in zip(lengths, owners):
            load[r] += n
        if sum(load):
            s = self.stats
            s["spec/verify_load_busiest_tokens"] = s.get("spec/verify_load_busiest_tokens", 0.0) + max(load)
            s["spec/verify_load_tokens"] = s.get("spec/verify_load_tokens", 0.0) + sum(load)
            s["spec/verify_load_max_over_mean"] = (
                ranks * s["spec/verify_load_busiest_tokens"] / s["spec/verify_load_tokens"]
            )
            s.setdefault("spec/verify_load_first_max_over_mean", max(load) * ranks / sum(load))




    def _draft_age(self) -> int:
        """Versions between this step's verifier and the weights that drafted its cohort (schedule only)."""
        version = self.__dict__.get("cohort_draft_version")
        if version is None:
            version = self.current_step - 1
        return int(self.current_step) - int(version)

    def _update_null_calibration(self) -> None:
        """This step's calibrated ⊥-BV policy (earlier steps' rows only); idempotent."""
        calib = self.__dict__.get("_null_calib")  # absent on test stubs
        if calib is not None:
            # This step's v: earlier steps' rows only (fixed before any of
            # this step's drafts is verified).
            v = calib.estimate(self._draft_age(), self.current_step)
            cv = calib.verify_cost(int(self.null_bv_policy.get("ref_len", self.max_new_tokens)))
            if (v, cv) != (self._null_policy.logratio_var, self._null_policy.verify_cost):
                self._null_policy = NullPolicy(
                    **{**self.null_bv_policy, "logratio_var": v, "verify_cost": cv}
                )
            self.stats["spec/null_v_used"] = v
            self.stats["spec/null_verify_cost_used"] = cv



    def _null_pools(self, flat, groups: list[str], slots: dict[str, int]) -> dict[str, "_NullPoolRun"]:
        """One wave runner per prompt pool (first-row order); rows in drafting order."""
        queue: dict[str, list[int]] = defaultdict(list)
        for i, g in enumerate(groups):
            queue[g].append(i)
        one = bool(self.__dict__.get("null_one_wave", False))
        return {g: _NullPoolRun(q, int(slots[g]), self._null_policy, one_wave=one) for g, q in queue.items()}

    def _null_wave(self, pools: dict[str, "_NullPoolRun"], names, all_slots) -> list[int]:
        """The next ⊥-BV wave: every active pool's next rows (``_NullPoolRun.next_wave``), pools in drafting order.

        Without spare prompts every pool is active. Under prompt drop with spares, the selection
        (``_select_prompts`` / ``null_bv.select_pools``) walks the prompts in drafting order (``all_slots``) and keeps
        each undropped one whose slots still fit the trained slots. A pool is active iff it fits in that walk with
        every earlier pool still deciding counted as undropped; ``used`` only grows along the walk, so a pool that
        does not fit is never kept, by the undropped pass or by the shortfall fill, whatever its own outcome. A spare
        pool's rows are therefore forwarded only once an earlier pool drops. Activation is a function of the drop
        counts, slot counts and the fixed order alone, never of content, and a pool's decisions do not depend on the
        wave or call its rows ride in (row seeds), so the selected prompts and their outputs are those of verifying
        every pool (MODEL.md v2 §12); pools never activated cost nothing (their rows are closed unused).
        """
        spares = int(self.__dict__.get("null_bv_spare_prompts", 0) or 0)
        if spares <= 0 or not self._null_policy.prompt_drop or names is None or all_slots is None:
            return [i for run in pools.values() for i in run.next_wave()]
        slots, train = self._null_selection(all_slots)
        batch, used = [], 0
        for key in all_slots:
            if used + slots[key] > train:
                continue
            group = names.get(key)
            if group is not None:  # a prompt without drafts is never dropped (its slots decode fresh)
                run = pools[group]
                batch += run.next_wave()
                if run.state.dropped:
                    continue
            used += slots[key]
        return batch

    def _null_empty_results(self, pools, groups, flat, idx: list[int]) -> dict[int, dict[str, Any]]:
        """Grid results of empty drafts (no token: cut before their first, or unlogged): each its keyed coin at every
        envelope its position can be decided at, so the pool walk decides it in drafting order like any draft."""
        seed, stream = self.__dict__.get("_empty_coin_seed", (0, "0"))
        out = {}
        for i in idx:
            run = pools[groups[i]]
            u = empty_draft_uniform(seed, stream, self.current_step, _key(flat[i][1][0]), run.pos[i])
            out[i] = empty_draft_result(u, run.env[i])
        if idx:
            # running total over the run (the per-verify count is spec/null_empty_walked)
            self.stats["spec/null_empty_walked_total"] = (
                self.stats.get("spec/null_empty_walked_total", 0.0) + float(len(idx))
            )
        return out

    def _null_apply(self, pools, groups, flat, results: dict[int, dict[str, Any]]) -> None:
        """Decide the rows of ``results`` (row -> grid result), each pool in drafting order."""
        for i in sorted(results):
            pools[groups[i]].decide(
                i, results[i], self.null_bv_drop_is,
                len(flat[i][1][2]) if self.null_bv_drop_is else 0,
            )

    def _null_close(self, pools, flat) -> dict[int, dict[str, Any]]:
        """Finish every pool: rows never forwarded are unused (hidden freed); stats."""
        out: dict[int, dict[str, Any]] = {}
        skipped: list[int] = []
        for run in pools.values():
            out.update(run.out)
            skipped += run.close()
        self._null_dropped = {g for g, run in pools.items() if run.state.dropped}
        self._release_hidden(flat[i][1] for i in skipped)
        for i in skipped:
            out[i] = _null_result({}, None)
        envs = [len(e) for run in pools.values() for e in run.env.values()]
        self.stats["spec/null_envelopes_per_row"] = sum(envs) / max(len(envs), 1)
        self.stats["spec/null_skipped_drafts"] = float(len(skipped))
        return out

    def _null_plan(self, d, res) -> Plan:
        """The plan of a ⊥-BV output row (``_finish_verify_null``)."""
        plan = self._plan(d, res)
        plan.verified = (len(d[2]), res["accepted"])
        if self.null_bv_drop_is:
            # max(1, W*/M): the kept row's law is p P(no tail | y) / P(no tail).
            plan.log_weight = max(0.0, float(res["log_wstar"]) - math.log(res["null_m"]))
        return plan


    def _finish_verify_null(
        self, flat, by_row, groups, slots, dropped, n_empty, published=None, clear=True, publish=True
    ):
        """Plans and ⊥-BV statistics: each prompt's outputs in drafting order.

        Slots left empty (drafts ran out) get no plan: those rollouts decode
        fresh from the current policy, an exact p-sample. published
        (streaming verify0): row -> the plan already published for it
        (_null_plan); only the statistics are computed. clear /
        publish: as _finish_verify.
        """
        if clear and published is None:
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
        # Per group: every output's plan complete at verify0 (fully drafted),
        # and also every draft accepted whole (trainable from its verify forward).
        grp_done: dict[str, bool] = {}
        grp_whole: dict[str, bool] = {}
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
            if res.get("null_empty"):
                continue  # an empty draft's output is a fresh decode: no plan, its slot decodes fresh at theta_k
            if published is not None:
                plan = published[r]
            elif publish:
                plan = self._null_plan(d, res)
            else:
                # Statistics only (pipelined engine verification published per block).
                plan = self._plan_tokens(d, res)
                plan.verified = (g, res["accepted"])
                if self.null_bv_drop_is:
                    plan.log_weight = max(
                        0.0, float(res["log_wstar"]) - math.log(res["null_m"])
                    )
            if self.null_bv_drop_is:
                weights.append(math.exp(plan.log_weight))
            # Complete at verify0 (streamed plans may have finished their tails since).
            done = plan.complete if published is None else not plan.tail
            full += done and res["accepted"] == g
            grp_done[groups[r]] = grp_done.get(groups[r], True) and done
            grp_whole[groups[r]] = (
                grp_whole.get(groups[r], True) and done and res["accepted"] == g
            )
            kept += res["accepted"]
            total += g
            out_pairs.append((g, res["accepted"]))
            if res["accepted"] < g:
                n_tail += 1
                rejected.append((g, res["accepted"]))
            self._count_ages(d, res["accepted"], age_tok, age_kept)
            if publish and published is None:
                self.plans[_key(d[0])].append(plan)
        self._v0_pairs = out_pairs
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
        live = [grp for grp in slots if grp not in dropped_groups]
        filled = {grp for grp in live if outs_by_group[grp] >= slots[grp]}
        n_live = max(len(live), 1)
        self.stats["spec/null_live_groups"] = float(len(live))
        self.stats["spec/null_groups_fully_drafted"] = (
            sum(grp_done.get(grp, False) for grp in filled) / n_live
        )
        self.stats["spec/null_groups_whole_accepted"] = (
            sum(grp_whole.get(grp, False) for grp in filled) / n_live
        )
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
                "spec/null_empty_walked": float(n_empty),  # empty drafts of this verify, each walked by its coin
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
        # Per-token log p/q moments of every row verified this step (decided or
        # not): the calibration of LATER steps' policies, and a logged signal.
        moments = [tuple(by_row[r].get("lr_moments", (0, 0.0, 0.0))) for r in range(len(flat))]
        self.stats["spec/p_zero_tokens"] = self.stats.get("spec/p_zero_tokens", 0.0) + float(
            sum(int(by_row[r].get("p_zero_tokens", 0)) for r in range(len(flat)))
        )
        n_lr = sum(m[0] for m in moments)
        if n_lr >= 2:
            mean = sum(m[1] for m in moments) / n_lr
            self.stats["spec/logratio_var_step"] = max(
                sum(m[2] for m in moments) / n_lr - mean * mean, 0.0
            )
            self.stats["spec/logratio_mean_step"] = mean
        # Per draft age (versions behind; 4 = 4+) and per position bin (drift_log.py).
        from thundersync_rl import drift_log

        self.stats.update(
            drift_log.age_moment_stats((by_row[r] for r in range(len(flat))), self.current_step)
        )
        calib = self.__dict__.get("_null_calib")  # absent on test stubs
        if calib is not None:
            calib.update(moments, self._draft_age(), self.current_step)
            self.stats["spec/null_v_next"] = calib.estimate()
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
        cap = self.__dict__.get("_v0_cap", {}).get(id(d))
        if cap is not None and cap[1] is d:
            # verify0 capture: the plan's pieces continue from the state cached under it.
            plan.prefix_key = cap[0]
            self._v0_plan[cap[0]] = plan
        if os.environ.get("THUNDERSYNC_TAIL_DUMP"):
            # Debug: the draft's tokens for the tail dump (TailTrace).
            plan.draft_tokens = [int(x) for x in d[2]]
        if "band_log_weight" in r:
            plan.log_weight = float(r["band_log_weight"])
            self._band_rows.append((plan.log_weight, bool(r["band_exited"])))
        return plan

    def _plan_tokens(self, d, r) -> Plan:
        plan = self._plan_tokens_inner(d, r)
        segs = d[5] if len(d) > 5 and d[5] else ()
        if segs:
            # The version that drafted the row (anchor epochs: its anchor version; aep.py R1).
            plan.anchor_version = int(segs[0][0])
        return plan

    def _plan_tokens_inner(self, d, r) -> Plan:
        prompt, seed, draft = d[0], d[1], d[2]
        n_acc = r["accepted"]
        lps = behavior_logprobs(
            r["logprobs"],
            n_acc,
            d[6] if len(d) > 6 else None,
            r.get("draft_logq"),
        )
        complete = n_acc == len(draft) and self._stopped(d)
        anchors = None
        if complete:
            return Plan(
                seed,
                draft,
                lps[:n_acc],
                complete=True,
                anchors=anchors,
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
        (measured).
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

    def _log_verify_error(self) -> None:
        """Print the verification error's full traceback once per error.

        The error reaches the rollouts as "draft verification failed" (one per
        waiting sample) and the learner as an incomplete canonical order; this
        is the one place its cause is printed.
        """
        err = getattr(self, "_verify_error", None)
        if err is None or getattr(self, "_verify_error_logged", None) is err:
            return
        self._verify_error_logged = err
        print(
            "[thundersync] draft verification failed:\n"
            + "".join(traceback.format_exception(err)),
            flush=True,
        )




    # -- sticky routing: verification and theta_n continuations --------------------





    # -- risk-first streaming verify0 ------------------------------





    # -- premise check: engine prefill p == learner p (THUNDERSYNC_ENGINE_PREFILL_CHECK) --



    @staticmethod
    def _prefill_check_record(rows: list, prompt: torch.Tensor, plan) -> None:
        """Append a complete verified plan as an engine prefill check candidate.

        The check compares the engine's p of these tokens with the learner's
        (prefix_logprobs) once the engine holds theta_k. Rows with a rejection
        or a continuation (residual and continuation tokens) are preferred at
        check time.
        """
        if plan.verified is None or not plan.complete:
            return
        rows.append(
            (prompt, list(plan.prefix), list(plan.prefix_logprobs), plan.anchors,
             bool(plan.tail or plan.verified[1] < plan.verified[0]))
        )


    # -- pipelined engine verification (verify_on=engine, sticky, streaming) ------



    # -- streaming training over the tails (one engine, sticky) ------------------





    def learner_seq(self, group: Any = None) -> Optional[int]:
        """The next ordered-call number of ``group`` (default: the learner) while ordered, else None.

        Persistent across steps: the workers count every ordered call they run.
        ``group``: the worker group the call goes to (a verifier pool); each
        group has its own sequence (``_group_seq``).
        """
        if not self.__dict__.get("learner_ordered", False):
            return None
        return _group_seq(group if group is not None else self.learner_policy.worker_group)

    def learner_track(self, refs) -> None:
        """Ordered learner calls in flight (``learner_drain`` waits for them)."""
        self._learner_inflight.update(refs)






    # False once the step's refit left the engine off theta_k (a cohort lane
    # that took no fresh refit, or ``draft_now``): set by the driver's refit.
    fresh_at_current: bool = True

    def _check_fresh_weights(self, key) -> None:
        """A rollout about to decode at theta_k: the engine must hold theta_k.

        The driver skips the theta_k refit when it predicts that no rollout
        decodes on the engine (lanes: ``lane_fresh``; one engine: ``draft_now``).
        A wrong prediction used to sample such rows from the drafting weights
        silently (v0s-xlanes, 6fc1ae9c: forced-fresh rows, max|dlogp| 1.66).
        """
        if not self.fresh_at_current:
            raise RuntimeError(
                f"a rollout of prompt {key[:8]!r}... decodes at theta_k, but this step's "
                "refit left the engine on the drafting weights (the driver predicted "
                "no fresh row)"
            )

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














    # ---- carried walk ------------------------------------------------
    _cw_retained_max = 0










    def _v0_capture_on(self) -> bool:
        """verify0 capture is on for this call.

        Off (verify0 forwards uncached, as before) under multi-turn rollouts (a plan
        complete at verify0 may continue with the environment's turn) and the anchor
        horizon (the refit engine continues rollouts past it); and for engine-side
        verification (no learner forward).
        """
        return (
            bool(self.__dict__.get("prefix_cache_verify0", False))
            and self.prefix_cache is not None
            and self.turn_env is None
            and self.anchor_horizon is None
            and self.verify_on == "learner"
            and self.verify_source == "engine"
        )

    def _v0_cap_kw(self) -> dict[str, bool]:
        """``capture`` for a verify0 call, only when the capture is on (default calls unchanged)."""
        return {"capture": True} if self._v0_capture_on() else {}


    def _v0_settle(self, drafts=None) -> None:
        """verify0 decided ``drafts`` (None: every capture not settled yet).

        A captured row whose plan continues keeps its state, cut on the learner ranks to
        the boundary its first piece starts from (bound = prompt + kept prefix - 1); every
        other capture (complete rollouts, ⊥ / unused / dropped pool rows, losing variants)
        is released. Memory only: ``PrefixStateCache.start`` picks the same boundary by
        content either way, and nothing here feeds a decision.
        """
        if not self.__dict__.get("_v0_cap"):
            return
        ids = (
            list(self._v0_cap)
            if drafts is None
            else [id(d) for d in drafts if id(d) in self._v0_cap and self._v0_cap[id(d)][1] is d]
        )
        for i in ids:
            key, d = self._v0_cap.pop(i)
            plan = self._v0_plan.pop(key, None)
            if plan is not None and plan.prefix_key == key and not plan.complete and not plan.vit:
                self.prefix_cache.settle(key, d[0].numel() + len(plan.prefix) - 1)
            else:
                if plan is not None and plan.prefix_key == key:
                    plan.prefix_key = None
                self.prefix_cache.settle(key, None)




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
        forced_fresh = _debug_forced_fresh(key)
        if forced_fresh:
            self.stats["debug/forced_fresh_rows"] = (
                self.stats.get("debug/forced_fresh_rows", 0.0) + 1.0
            )
        if forced_fresh and self.__dict__.get("aep") is not None:
            # Anchor epochs: forced-fresh rows are served by their empty-prefix continuations (``_aep_fill``).
            forced_fresh = False
        cv = getattr(self, "_plan_cv", None)
        if cv is not None and not forced_fresh:
            # Chunked verification: wait until this trajectory's plan is published.
            async with cv:
                while not self.plans.get(key) and self._pending.get(key, 0) > 0:
                    if getattr(self, "_verify_error", None) is not None:
                        self._log_verify_error()
                        raise RuntimeError(
                            "draft verification failed"
                        ) from self._verify_error
                    await cv.wait()
        queue = None if forced_fresh else self.plans.get(key)
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
                self._check_fresh_weights(key)
        aep = self.__dict__.get("aep")
        if (
            aep is not None
            and not plan.complete
            and not aep.engine_holds_target(self.current_step)
        ):
            # Anchor epochs: the engine holds the anchor, never theta_k (R2).
            raise RuntimeError(
                f"anchor_epoch: a rollout of step {self.current_step} would decode on the anchor engine "
                f"(theta_{aep.engine_version}) without verification"
            )
        gen = list(plan.prefix)
        lps = list(plan.prefix_logprobs)
        pc = getattr(self, "_prefill_check_rows", None)
        if self.sticky_delta is not None and pc is not None and len(pc) < 4 * _prefill_check_rows_cap():
            self._prefill_check_record(pc, prompt, plan)
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
                    "max_new_tokens": torch.tensor([max(0, self._cap(plen) - len(gen))]),
                }
            )
            if plan.seed >= 0:
                cont["noise_seed"] = torch.tensor([plan.seed])
            elif plan.verified is None and not plan.prefix and "noise_seed" in datum:
                # Never drafted: decoded fresh at theta_k under the row's stable key (the rollout loop's
                # hash(seed, "fresh", step, row, turn); worker.install_wide_sampler). Drafts and the continuations
                # of drafted rows keep their engine request-id keys.
                cont["noise_seed"] = datum["noise_seed"]
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
            t_end = time.perf_counter()
            n_cont = len(gen) - len(plan.prefix)
            t_step0 = self.__dict__.get("step_t0")
            if t_step0 is not None:
                # Critical continued row of the step (one engine, θ-only routing: continuations decode at theta_k on
                # the engine that also drafts the next cohort): its end, length and decode rate.
                st = self.stats
                end = t_end - t_step0
                st["tail/one_cont_rows"] = st.get("tail/one_cont_rows", 0.0) + 1
                if end > st.get("tail/one_slowest_end_s", -1.0):
                    st["tail/one_slowest_end_s"] = end
                    st["tail/one_slowest_start_s"] = t0 - t_step0
                    st["tail/one_slowest_cont_tokens"] = float(n_cont)
                    st["tail/one_slowest_prefix"] = float(len(plan.prefix))
                    st["tail/one_slowest_rejected"] = float(
                        plan.verified is not None and plan.verified[1] < plan.verified[0]
                    )
                    st["tail/one_slowest_ms_per_tok"] = 1e3 * (t_end - t0) / max(1, n_cont)
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
        if self.sticky_delta is None and pc is not None and len(pc) < 4 * _prefill_check_rows_cap():
            # θ-only routing (one engine): record the finished row, continuation and fresh tokens included (their
            # logged p is the engine's decode p at theta_k; drafted and residual tokens carry the learner's
            # verification p), so the end-of-step engine prefill check covers every token kind.
            pc.append(
                (prompt, list(gen), list(lps), None,
                 bool(not plan.complete or (plan.verified is not None and plan.verified[1] < plan.verified[0])))
            )
        if (
            env_flag("THUNDERSYNC_KEYED_REPLAY")
            and self.verify_mode == "keyed"
            and plan.seed >= 0
            and plan.prefix
        ):
            await self._keyed_replay_check(prompt, plan.seed, gen, datum)
        # Path coverage of the exactness checks: which paths
        # produced this step's rows (machine-checked by run_rules expectations).
        st = self.stats
        st["exact/rows"] = st.get("exact/rows", 0.0) + 1
        if plan.verified is None:
            st["exact/rows_undrafted"] = st.get("exact/rows_undrafted", 0.0) + 1
        elif plan.verified[1] < plan.verified[0]:
            st["exact/rows_rejected"] = st.get("exact/rows_rejected", 0.0) + 1
        if not plan.complete:
            st["exact/rows_engine_cont"] = st.get("exact/rows_engine_cont", 0.0) + 1
            st["exact/engine_tokens"] = st.get("exact/engine_tokens", 0.0) + len(gen) - len(plan.prefix)
        if gen and gen[-1] == self.eod:
            st["exact/rows_eos"] = st.get("exact/rows_eos", 0.0) + 1
        if len(gen) < self._cap(plen):
            st["exact/rows_short"] = st.get("exact/rows_short", 0.0) + 1
        if len(gen) > self._cap(plen):
            # Every single-turn row passes here (fresh, complete plan, continued
            # plan): a row longer than the cap is not a sample of the capped policy.
            raise RuntimeError(
                f"row exceeds the generation cap: {len(gen)} > {self._cap(plen)} tokens "
                f"(prompt {plen}, kept prefix {len(plan.prefix)}, complete={plan.complete})"
            )
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


    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)
