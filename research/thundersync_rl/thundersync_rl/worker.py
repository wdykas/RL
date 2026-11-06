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
"""Megatron policy worker with gradient-streaming (ThunderSyncRL) methods.

Builds on the worker's split train-step API (``begin_train_step`` /
``train_microbatch`` / ``finish_train_step``), which already accumulates
un-normalized gradients locally under ``model.no_sync()`` and defers the 1/N
normalization, the DP reduction, clipping and the optimizer step to a single
barrier. On top of that we use the DDP ``grad_data`` buffers as scratch: every
streamed chunk's backward lands there and is immediately moved into a
per-(group, reward) accumulator, so its advantage can be applied once the
group closes. At the barrier, the folded batch gradient is written back into
the grad buffers and the regular ``finish_train_step`` runs unchanged.
"""

import asyncio
import functools
import inspect
import itertools
import os
import time
import zlib
from contextlib import nullcontext
from typing import Any, Optional

import numpy as np
import ray
import torch

from nemo_rl.algorithms.grad_streaming import GradStreamingSpec, StreamBucket
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.data.multimodal_utils import attach_media_token_validity_mask
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron import train as megatron_train
from nemo_rl.models.megatron.data import get_microbatch_iterator
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)
from thundersync_rl import engine_prof
from thundersync_rl.draft_weights import (
    DraftSwap,
    EngineStage,
    check_no_expert_bias_updates,
    load_draft,
    restore_training,
)
from thundersync_rl import mem_debug
from thundersync_rl import verify_yield, vit, vit_check
from thundersync_rl.nolanes_engine import NolanesEngineMixin
from thundersync_rl.fused_train import (
    FusedGraph,
    FusedGraphStore,
    FusedOutcome,
    make_fused_model_forward,
    substitute_model_forward,
)
from thundersync_rl.envflag import env_flag
from thundersync_rl.debug_hooks import verify_no_bi as _debug_verify_no_bi

WORKER_FQN = "thundersync_rl.worker.ThunderSyncMegatronPolicyWorker"

# THUNDERSYNC_V0_NO_INTERLEAVE check (defined in verify_yield: posted calls run it without importing this module).
_check_v0_no_interleave = verify_yield.check_v0_no_interleave

# Learner-side state of one engine lane's refit communicators (cohort lanes): the
# packed collective's ``model_update_group`` and the native MCore (m2n) refit's
# process group, copy service and destination rank offset (refit_transport mcore).
_LANE_REFIT_ATTRS = (
    "model_update_group",
    "refit_pg",
    "refit_copy_service",
    "refit_dst_rank_offset",
)


def _split_to_fit(batch: list, fits) -> list[list]:
    """Consecutive parts of ``batch`` (order kept) each satisfying ``fits`` (greedy)."""
    out: list[list] = []
    for i in batch:
        if out and fits(out[-1] + [i]):
            out[-1].append(i)
        else:
            if not fits([i]):
                raise RuntimeError("prefix-state cache: one row does not fit the pools")
            out.append([i])
    return out


def _prefix_check_on() -> bool:
    """``THUNDERSYNC_PREFIX_CACHE_CHECK=1``: each prefix-state forward is checked
    against a whole-row forward (one extra forward per batch, matched by the
    expert-parallel dummy forwards of ranks without rows)."""
    return os.environ.get("THUNDERSYNC_PREFIX_CACHE_CHECK") == "1"


def _model_in_eval(method):
    """Run ``method`` with ``self.model`` in eval mode.

    Learner: restore the previous mode afterwards, since verification runs while
    a streamed train step is open and leaving eval mode on would silently disable
    dropout for the rest of the step. Generation workers: leave it in eval mode;
    the inference engine runs on the same module concurrently and requires it.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "dynamic_inference_engine", None) is not None:
            self.model.eval()
            return method(self, *args, **kwargs)
        was_training = self.model.training
        self.model.eval()
        try:
            return method(self, *args, **kwargs)
        finally:
            self.model.train(was_training)

    return wrapper


def _step_routes(replays, n: int) -> Optional[torch.Tensor]:
    """Routes [n, L_local, K] of the engine forward that just ran (sticky routing).

    Read where each router recorded them this step (``_install_route_record``):
    the static buffer MCore bound for a CUDA-graph step, else the eager
    forward's ``recorded_topk_idx`` (on a graph replay no Python runs, so that
    attribute is stale and the bound buffer is the only source). None if a
    layer recorded nothing: those positions stay unanchored and truncate their
    drafts (engine bookkeeping, not token content).
    """
    cols = []
    for r in replays:
        src = r.static_buffer if r.static_buffer is not None else r.recorded_topk_idx
        if src is None or int(src.shape[0]) < n:
            return None
        cols.append(src[:n])
    return torch.stack(cols, 1)


def _drop_mcore_route_bookkeeping(controller) -> None:
    """Switch off MCore's own routing-replay bookkeeping on an engine controller.

    Sticky routing turns on ``moe_enable_routing_replay`` so that the routers
    record (``_install_route_record``); MCore then also runs, every legacy
    step, ``_router_record_bookkeeping``: a synchronous ``.cpu()`` of the
    step's [tokens, L, K] routes (a TP all-gather under sequence parallelism),
    scattered into per-KV-block host storage, from which every finished
    request's ``routing_indices`` is rebuilt and sent back in its reply. The
    route log replaces all of it, so it returns None (nothing stored, nothing
    rebuilt). Upstream candidate: an MCore option to record routes without
    this bookkeeping.
    """
    controller._router_record_bookkeeping = lambda: None


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """The innermost module under DDP / Float16Module wrappers."""
    while hasattr(model, "module"):
        model = model.module
    return model


def _plain_floats(x: Any) -> list[float] | None:
    """Engine log-probs (list, tensor or a serialized ("tensor", data) pair) as floats."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.float().tolist()
    if isinstance(x, (list, tuple)) and len(x) == 2 and x[0] == "tensor":
        from megatron.core.inference.inference_request import deserialize_tensor

        return deserialize_tensor(x[1]).float().tolist()
    return [float(v) for v in x]


def group_pipeline_calls(
    items: list[dict[str, Any]], *, merge: bool
) -> list[list[int]]:
    """Group a dispatch's items into pipeline schedule calls (lists of indices).

    Under pipeline parallelism each ``forward_backward_func`` call pays a
    fill/drain bubble of ``P - 1`` micro-batch slots, so a rank's items run as
    few calls as their gradient destinations allow:

    - all final items (``group=None``: advantages in the loss) share one call;
      their gradient accumulates in the grad buffers (or is captured into the
      batch accumulator once, after the call);
    - the items of one open-group bucket ``(group, reward)`` share one call,
      captured into that bucket once;
    - an item with a ``fused_key`` (retained verification forward, PP = 1
      only) runs alone.

    Calls are ordered by their first item, and items keep their order within
    a call, so micro-batches run in dispatch order per destination. Only the
    order in which gradients are summed changes (fp32 rounding).
    ``merge=False`` returns one call per item (the previous behavior).
    """
    if not merge:
        return [[i] for i in range(len(items))]
    calls: list[list[int]] = []
    by_key: dict[Any, list[int]] = {}
    for i, item in enumerate(items):
        if item.get("fused_key") is not None:
            calls.append([i])
            continue
        key = (item["group"], None if item["group"] is None else item["reward"])
        if key not in by_key:
            by_key[key] = []
            calls.append(by_key[key])
        by_key[key].append(i)
    return calls


def count_only_chunk(state: dict[str, Any], item: dict[str, Any]) -> None:
    """Add a skipped chunk's valid sequences/tokens to the step's normalization.

    The same sums ``_prepare_train_piece`` adds before its forward, so the
    loss normalizer (and the gradient) equals training the chunk: every one
    of its rows has advantage 0, i.e. exactly zero gradient (the loss and
    model are validated advantage-weighted, ``_require_advantage_weighted_gradients``).
    Raises if a row carries a nonzero advantage (never drop gradient silently).
    """
    data = item["data"]
    if item.get("group") is not None or item.get("fused_key") is not None:
        raise RuntimeError("count-only chunks must be final, unfused chunks")
    sample_mask = data["sample_mask"]
    token_mask = data["token_mask"][:, 1:] * sample_mask.unsqueeze(-1)
    if bool((data["advantages"][:, 1:] * token_mask).ne(0).any()):
        raise RuntimeError(
            "count-only chunk has a row with a nonzero advantage on a valid token "
            "(planner / worker disagree on zero-advantage rows)"
        )
    state["num_chunks"] += 1
    state["local_valid_seqs"] = state["local_valid_seqs"] + torch.sum(sample_mask).to(torch.float64)
    state["local_valid_toks"] = state["local_valid_toks"] + torch.sum(token_mask).to(torch.float64)


def apply_draft_frame(
    frame: dict[str, Any],
    tokens: list[int],
    logprobs: list[list[float] | None],
    i: int,
) -> bool:
    """Apply one streaming frame of draft ``i``; return True on the final frame.

    ``tokens`` is draft i's token list (updated in place). ``logprobs[i]`` holds
    the engine's log-prob of each token, or None once a frame did not carry one
    per new token (the driver then reports the scorer's log p, see
    ``SpeculativeGeneration._plan``).
    """
    if "partial" in frame:
        part = frame["partial"]
        new = list(part["new_tokens"])
        tokens.extend(new)
        lps = _plain_floats(part.get("new_log_probs"))
        if logprobs[i] is not None:
            if lps is not None and len(lps) == len(new):
                logprobs[i].extend(lps)
            else:
                logprobs[i] = None
        return False
    if "final" in frame:
        final = frame["final"]
        get = (
            final.get if isinstance(final, dict) else lambda k: getattr(final, k, None)
        )
        tokens[:] = list(get("generated_tokens"))
        lps = _plain_floats(get("generated_log_probs"))
        logprobs[i] = lps if lps is not None and len(lps) == len(tokens) else None
        return True
    return False


def typed_tokens(t) -> np.ndarray:
    """Token ids as one int64 array (no per-token Python objects in transport)."""
    return np.asarray(t, dtype=np.int64)


def typed_logq(lp) -> np.ndarray | None:
    """Engine log-probs as one float array, bitwise.

    The engine computes them in fp32, so float32 holds them exactly; any value
    that does not round-trip (never expected) keeps the whole array float64.
    """
    if lp is None:
        return None
    a = np.asarray(lp, dtype=np.float64)
    a32 = a.astype(np.float32)
    return a32 if np.array_equal(a32.astype(np.float64), a, equal_nan=True) else a


def snapshot_drafts(
    tokens: list[list[int]],
    logprobs: list[list[float] | None],
    done: list[bool],
) -> list[tuple[np.ndarray, bool, np.ndarray | None]]:
    """(tokens, finished, engine logprobs or None) per draft, copied as typed arrays."""
    out = []
    for t, lp, d in zip(tokens, logprobs, done):
        ok = lp is not None and len(lp) == len(t)
        out.append((typed_tokens(t), d, typed_logq(lp) if ok else None))
    return out


def _merge_record_refs(records, record_refs, worker=None, request_ids=None) -> dict:
    """``records``, this rank's own ``draft_records(request_ids)`` and the others' (refs)."""
    out = dict(records or {})
    if worker is not None and request_ids:
        out.update(worker.draft_records(request_ids))
    for r in ray.get(list(record_refs or [])):
        out.update(r)
    return out


def _call_log(worker: Any, name: str) -> None:
    """Debug (``THUNDERSYNC_CALL_LOG``): which learner call this rank starts, in order.

    Every learner rank must start the same calls in the same order (their
    collectives pair up); comparing the ranks' sequences finds a desync.
    """
    if env_flag("THUNDERSYNC_CALL_LOG"):
        print(
            f"[call log] rank={getattr(worker, 'rank', '?')} {name} t={time.time():.6f}",
            flush=True,
        )


if engine_prof.enabled():
    # Debug: engine step timing in every process that runs a dynamic engine.
    engine_prof.install()


def snapshot_anchor_weights(optimizer: Any, draft_kind: Optional[str]) -> None:
    """Keep the next step's sticky-rounding anchor weights (no-op when it is off).

    Called right before the optimizer step of step k, with theta_k, m_k, v_k
    still in place. The anchor forward is the drafter's forward: theta_k
    (``draft_kind`` None: the engine drafts the next step's rollouts with the
    refit theta_k), or theta_hat_{k+1} for draft weights ``draft_kind`` staged
    at the top of this step. theta_hat_{k+1} is a deterministic function of
    (theta_k, m_k, v_k), which the optimizer step has not changed yet, so it is
    recomputed here bitwise as staged (``load_draft``), kept, and theta_k is
    restored; the store stays one copy of the rank's 16-bit shard.
    """
    from nemo_rl.models.megatron import sticky_rounding

    if not sticky_rounding.enabled():
        return
    if draft_kind is None:
        sticky_rounding.snapshot()
        return
    swap = load_draft(optimizer, draft_kind)
    try:
        sticky_rounding.snapshot()
    finally:
        restore_training(swap)


def _emit_sampled(key: Any, every: int) -> bool:
    """Emission-law sample: a function of the row key only (THUNDERSYNC_EMISSION_LAW_EVERY)."""
    if every <= 1:
        return True
    import zlib

    return zlib.crc32(f"emit/{key}".encode()) % every == 0


def learner_expandable_segments() -> bool:
    """Whether the learner's allocator uses expandable segments (default on).

    ``THUNDERSYNC_LEARNER_EXPANDABLE_SEGMENTS=0`` keeps the stock segments.
    """
    return os.environ.get("THUNDERSYNC_LEARNER_EXPANDABLE_SEGMENTS", "1") != "0"


_BASE_INIT_SIGNATURE = inspect.signature(MegatronPolicyWorkerImpl.__init__)


def _binds_init_optimizer(args: tuple, kwargs: dict) -> bool:
    """Whether a MegatronPolicyWorkerImpl constructor call builds an optimizer (the learner)."""
    bound = _BASE_INIT_SIGNATURE.bind_partial(None, *args, **kwargs)
    bound.apply_defaults()
    return bool(bound.arguments["init_optimizer"])


def _binds_colocated_generation(args: tuple, kwargs: dict) -> bool:
    """Whether a MegatronPolicyWorkerImpl constructor call's config has colocated generation (one process holds the
    learner and the engine)."""
    bound = _BASE_INIT_SIGNATURE.bind_partial(None, *args, **kwargs)
    cfg = bound.arguments.get("config") or {}
    return bool(((cfg.get("generation") or {}).get("colocated") or {}).get("enabled", False))


def enable_learner_expandable_segments() -> bool:
    """Learner process only: new CUDA allocator segments are expandable.

    The learner's in-loop memory interleaves long-lived allocations (Adam
    state, created at the first optimizer step; prefix-cache entries; head
    stashes) with transient ones of varying size (micro-batches of different
    lengths, verification forwards, the deferred head's chunk buffers). With
    fixed segments, free space stranded inside partly used segments cannot
    serve a larger request: full-Nano 8k learners died with 8.5-9.5 GiB
    "reserved but unallocated" next to a 1.2-2 GiB request
. An expandable segment maps physical pages
    into one virtual range per stream; on an allocation miss the allocator
    unmaps free pages anywhere in it and maps them where the request fits,
    so stranded free memory is reusable. Values are untouched (allocation
    placement only), so every forward / gradient is bitwise unchanged.

    Called right before the checkpoint load (the learner's pre-load hook) and
    again after construction (no-op repeat; covers a run that loads nothing).
    Static memory (weights, grad buffers, the fp32 master shard) was allocated
    before this call in fixed segments that are never freed; everything
    allocated afterwards, including a resumed run's loaded Adam state and the
    load's read buffers, uses expandable segments. Engines (no optimizer) keep
    the stock allocator: their CUDA graphs and refit receive paths are not
    changed. The learner sends weights through NCCL (no CUDA-IPC handles), so
    the stock worker's warning about slower IPC weight transfer does not apply.
    """
    if not learner_expandable_segments() or not torch.cuda.is_available():
        return False
    setter = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
    if setter is None:  # torch < 2.10
        setter = torch.cuda.memory._set_allocator_settings
    setter("expandable_segments:True")
    # NCCL graph buffer registration pins the virtual addresses a captured collective reads and writes
    # (peers map them for zero-copy transfers); an expandable segment later unmaps free pages anywhere
    # and maps others in their place (empty_cache, allocation misses), so a learner-graph replay's
    # NCCL all-to-all (the ordered TP >= 3 reduce-scatter, fp16_gemm.ordered_reduce_scatter_rows) read
    # and wrote through stale mappings: Super-L8 TP4 x EP4 loop step 0 learner log-probs garbage (63 of
    # 64 graph-replayed rows wrong, max |d| 20.7), step 1 illegal memory access (scu-loop3); reproduced
    # by research/scaleup/sticky_lg_probe.py --expandable 1 --churn_gb 8, fixed by this (ALL BITWISE).
    # Unregistered, NCCL stages through its own buffers as in eager collectives: no extra memory per
    # rank. Read by NCCL when a capture registers buffers (after this call), so setting it here is in
    # time; an explicit NCCL_GRAPH_REGISTER in the environment is kept.
    if "NCCL_GRAPH_REGISTER" not in os.environ:
        os.environ["NCCL_GRAPH_REGISTER"] = "0"
    return True


@torch.no_grad()
def _restash(
    stash: dict[int, Any], version: int, keep: list[int], srcs: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Copies of ``srcs`` for ``stash[version]``, written into the buffers of a version being dropped when one is.

    Drops every version not in ``keep`` (and not ``version``) from ``stash`` first, then reuses the first dropped
    version's tensors (same shapes / dtypes) instead of cloning next to them: no extra full copy per stash and no
    allocator churn (the old order cloned the new version while the dropped one was still alive). Values are the
    same bits as ``clone()``.
    """
    dropped = [stash.pop(v) for v in list(stash) if v not in keep and v != version]
    old = stash.pop(version, None)
    reuse = old if old is not None else (dropped[0] if dropped else None)
    if isinstance(reuse, torch.Tensor):
        reuse = [reuse]
    del dropped, old
    if reuse is not None and len(reuse) == len(srcs) and all(
        b.shape == s.shape and b.dtype == s.dtype and b.device == s.device for b, s in zip(reuse, srcs)
    ):
        for b, s in zip(reuse, srcs):
            b.copy_(s)
        return list(reuse)
    del reuse
    return [s.detach().clone() for s in srcs]


_FRESH_KEY_BIT = 1 << 62


class _RequestKeys:
    """Engine request id -> the request's ``noise_seed`` (-1: none), cached for the wide sampler.

    A dense window of int64 slots indexed by request id - base (ids grow by one per request; a call only asks
    for running requests, so the window spans the ids in flight). A miss looks the request up once
    (engine.get_request); the window slides past ids below the smallest id asked for, so the cache stays bounded
    by the in-flight span (ids behind the window, e.g. a request paused for long, are looked up uncached).
    """

    UNKNOWN = -2

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.base = 0
        self.slots = np.empty(0, dtype=np.int64)

    def _fetch(self, rid: int) -> int:
        try:
            req = self.engine.get_request(int(rid))
        except Exception:  # noqa: BLE001 - an id the engine no longer holds: no key
            return -1
        sp = getattr(req, "sampling_params", None)
        s = getattr(sp, "noise_seed", None)
        return int(s) if s is not None and int(s) >= 0 else -1

    def lookup(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        out = np.full(ids.shape, -1, dtype=np.int64)
        if ids.size == 0:
            return out
        lo, hi = int(ids.min()), int(ids.max())
        if lo - self.base > self.slots.size // 2:
            cut = min(lo - self.base, self.slots.size)
            self.slots = self.slots[cut:].copy()
            self.base += cut
        if self.slots.size == 0:
            self.base = lo
        need = hi - self.base + 1
        if need > self.slots.size:
            grow = max(need - self.slots.size, self.slots.size, 1024)
            self.slots = np.concatenate([self.slots, np.full(grow, self.UNKNOWN, dtype=np.int64)])
        inside = ids >= self.base
        idx = ids[inside] - self.base
        vals = self.slots[idx]
        for j in np.nonzero(vals == self.UNKNOWN)[0]:
            vals[j] = self.slots[idx[j]] = self._fetch(int(idx[j]) + self.base)
        out[inside] = vals
        for j in np.nonzero(~inside)[0]:
            out[j] = self._fetch(int(ids[j]))
        return out


class KeyedBatchError(RuntimeError):
    """A request whose sampling parameters the keyed sampler cannot serve reached a keyed engine."""


def _key_check_positions(context: Any, ids: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """``pos`` with the partial prefill chunk's row set to -1 (unknown: the key checks skip it).

    With chunked prefill, MCore samples the row of a request whose prompt is not done yet
    (``context.chunked_prefill_request_id``: more prompt tokens follow this step) and discards the draw: the next
    chunk's first prompt token replaces it. Its key position len(prompt) is drawn again, and kept, by the request's
    final chunk, so the discarded draw is not a key reuse. The draw itself is unchanged (same noise as unchunked).
    """
    cid = int(getattr(context, "chunked_prefill_request_id", -1))
    if cid < 0:
        return pos
    hit = ids == cid
    if not hit.any():
        return pos
    out = pos.copy()
    out[hit] = -1
    return out


def _request_positions(engine: Any, ids: np.ndarray, lens: Optional["_RequestLengths"] = None) -> np.ndarray:
    """Index of the token each engine request samples now: len(prompt_tokens) + len(generated_tokens) of its latest
    record (an eviction checkpoint moves generated tokens into the prompt: the sum is unchanged); -1 if unknown.

    The wide sampler's noise key position. ``gpu_view.request_kv_length_offsets + request_query_lengths`` is not
    usable under async scheduling: when requests finish, the controller compacts the pending logits, temperatures
    and the host request ids into survivor order but not those two gpu_view arrays, so for one step every row after
    a finished one read another request's position (measured: 10 of 524,288 draws at Nano-L8 8k, 2 of them
    an earlier (request, position) key of the same request = reused noise; timing dependent, so a resumed step did
    not reproduce). Host cost: one dict lookup per running request per decode step; with ``lens`` (an engine with a
    ``requests`` dict) the prompt lengths come from the ``_RequestLengths`` cache (same values).
    """
    out = np.full(ids.shape, -1, dtype=np.int64)
    if lens is not None and isinstance(getattr(engine, "requests", None), dict):
        lens.begin()
        for j, rid in enumerate(ids.tolist()):
            c = lens.get(engine, rid)
            if c is not None:
                out[j] = c[2] + len(c[0].generated_tokens)
        return out
    for j, rid in enumerate(ids.tolist()):
        try:
            req = engine.get_request(int(rid))
            out[j] = len(req.prompt_tokens) + len(req.generated_tokens)
        except Exception:  # noqa: BLE001 - not an engine request (tests) / unknown id: the gpu_view position
            pass
    return out


def _install_one_tolist_log_probs(controller: Any) -> None:
    """MCore's async-schedule log-prob materialization with one ``tolist()`` per step (was one per request).

    ``TextGenerationController._materialize_async_sched_log_probs`` turns the step's CPU log-prob view into a list
    per request with ``view[slice].tolist()`` per request: a tensor slice + tolist per row, ~4-5 us each (1.9 ms per
    decode step at 384 rows on the Super lane engine's TP0, py-spy 10-26). Without MTP accepted counts and top-n
    log-probs (every ThunderSync engine: logprobs_mode raw), the rows are the consecutive row_counts slices of the
    view: one ``tolist()`` of the view, then list slices. Same Python floats (tolist converts each element the same
    way), same structure; any other case keeps MCore's code. Installed per controller (instance attribute shadows
    the staticmethod); upstream candidate.
    """
    orig = getattr(controller, "_materialize_async_sched_log_probs", None)
    if orig is None or getattr(orig, "_ts_one_tolist", False):
        return

    def materialize(transfer, accepted_counts_cpu=None):
        if transfer is None or accepted_counts_cpu is not None or any(transfer.top_n_counts):
            return orig(transfer, accepted_counts_cpu)
        vals = transfer.selected_log_probs_cpu_view.tolist()
        log_probs = []
        off = 0
        for row_count, _, _ in zip(transfer.row_counts, transfer.top_n_counts, transfer.skip_prompt_log_probs):
            log_probs.append(vals[off : off + row_count])
            off += row_count
        return log_probs, None

    materialize._ts_one_tolist = True
    controller._materialize_async_sched_log_probs = materialize


class _RequestLengths:
    """Engine request id -> (latest record, its prompt tensor, len, first record's prompt tensor, len, hidden key).

    The ts engine's per-decode-step host loops (the wide sampler's key positions, the hidden-state log's row
    indices) read, for every running request, len(prompt_tokens) of its latest record and of its first record: a
    torch ``Tensor.__len__`` each (~1 us under the torch-function check), three per row per step, plus the
    draft key. Prompt tensors do not change length in place: an eviction checkpoint appends a new record, and
    MCore only re-assigns ``prompt_tokens`` (``.cpu()``, a terminal stop-word trim). An entry is reused only while
    the request's latest record, its prompt tensor and the first record's prompt tensor are the same objects, so
    every length (and the draft key, read from that record's sampling params) equals the uncached one; an id the
    engine reuses belongs to a new request object, so it misses. ``begin()`` starts a call: entries not looked up since the previous
    ``begin()`` are dropped (bounded by the running requests).
    """

    __slots__ = ("_old", "_new")

    def __init__(self) -> None:
        self._old: dict = {}
        self._new: dict = {}

    def begin(self) -> None:
        if self._new:
            self._old, self._new = self._new, {}

    def get(self, engine: Any, rid: int):
        """(cur, cur prompt, len, first prompt, len, hidden key or None) of request ``rid``; None if unknown."""
        entry = engine.requests.get(rid)
        if entry is None:
            return None
        rec = entry.record
        cur = rec[-1]
        first = rec[0]
        c = self._old.get(rid) or self._new.get(rid)
        if c is None or c[0] is not cur or c[1] is not cur.prompt_tokens or c[3] is not first.prompt_tokens:
            from thundersync_rl.engine_q import HIDDEN_KEY_ATTR

            c = (
                cur,
                cur.prompt_tokens,
                len(cur.prompt_tokens),
                first.prompt_tokens,
                len(first.prompt_tokens),
                getattr(cur.sampling_params, HIDDEN_KEY_ATTR, None),
            )
        self._new[rid] = c
        return c


def allocator_stats() -> dict[str, float]:
    """Allocator peaks since the last ``reset_peak_memory_stats`` (GiB), cheap.

    ``frag_peak``: peak bytes free inside partly used fixed segments (what they
    strand; free pages of expandable segments are not counted: they are
    reusable); ``nontorch``: device memory in use that this process's
    allocator does not hold (CUDA context, NCCL buffers, kernels; includes other
    processes on the GPU); ``alloc_retries``: allocator misses that had to free
    cached blocks and retry (cumulative).
    """
    st = torch.cuda.memory_stats()
    free, total = torch.cuda.mem_get_info()
    g = 2.0**30
    return {
        "learner_mem_reserved_peak_gb": st.get("reserved_bytes.all.peak", 0) / g,
        "learner_mem_frag_peak_gb": st.get("inactive_split_bytes.all.peak", 0) / g,
        "learner_mem_nontorch_gb": (total - free - torch.cuda.memory_reserved()) / g,
        "learner_mem_alloc_retries": float(st.get("num_alloc_retries", 0)),
        # Driver-level allocator churn (cumulative cudaMalloc / cudaFree, or segment map / unmap under expandable
        # segments): flat after warm-up = the caching allocator reached steady state on its own.
        "learner_mem_device_allocs": float(st.get("num_device_alloc", 0)),
        "learner_mem_device_frees": float(st.get("num_device_free", 0)),
    }


def _moe_capacity_reruns() -> int:
    """Learner calls rerun on Megatron's MoE dispatcher after a device-dispatch capacity overflow (cumulative)."""
    from nemo_rl.models.megatron import learner_graphs

    return int(learner_graphs.stats().get("capacity_rerun", 0))


def _min_sampling_temperature() -> float:
    """The engine sampler's temperature clamp (FlashInferSampling)."""
    try:
        from megatron.core.inference.sampling_params import MIN_SAMPLING_TEMPERATURE

        return float(MIN_SAMPLING_TEMPERATURE)
    except Exception:  # noqa: BLE001 - older Megatron: no clamp constant there
        return 1e-5


def _lg_feature(name):
    """Tag the learner forwards of a worker method for learner_graphs' coverage counters
    (``feat_<name>_graph`` / ``feat_<name>_eager``; nested tags win)."""
    import functools

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            from nemo_rl.models.megatron import learner_graphs as _lg

            with _lg.feature(name):
                return fn(*a, **k)

        return wrapper

    return deco


from thundersync_rl.engine_slots import ACTOR_MAX_CONCURRENCY as _ACTOR_MAX_CONCURRENCY  # noqa: E402


def _sentinel_join(worker: Any, what: str) -> None:
    """Join the weight-sentinel digests in flight before a parameter write (no-op when none is)."""
    if worker.__dict__.get("_sentinel_async") is not None:
        from thundersync_rl import sentinel_async

        sentinel_async.join_before_write(worker, what)


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker"),
    # Above the driver's engine row slots + control calls (engine_slots.py): Ray's async-actor default (1000) queued
    # continuations behind 1008 drafts at Super-L22 wake_verify.
    max_concurrency=_ACTOR_MAX_CONCURRENCY,
)  # pragma: no cover
class ThunderSyncMegatronPolicyWorker(
    NolanesEngineMixin, verify_yield.YieldWorkerMixin, MegatronPolicyWorkerImpl
):
    """Driver-friendly wrappers over the core split API's streaming mode."""

    # The streaming planner skips zero-advantage rows under EP itself (count-only
    # chunks, equal live forwards per rank): no stock "ignored with EP" warning.
    _planner_skips_zero_advantage = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # The learner's allocator mode must be set before the checkpoint load, not after the constructor:
        # a resumed learner loads its Adam state there (an uninterrupted one creates it lazily at the first
        # optimizer step, after the constructor, in expandable segments). Loaded into stock segments, every
        # state tensor took a segment of its own (19.03 MiB -> 20 MiB, the < 1 MiB tail never split off:
        # +0.70 GiB allocated at L8) and the load's freed read buffers stayed cached as fixed segments no
        # training activation fits (+17 GiB reserved at L8), so every resumed process ran with less headroom
        # than the uninterrupted run (full-Nano resume OOM at step 44). Weights, grad buffers and the fp32
        # master shard are allocated before the hook, in stock segments, as before.
        # Colocated generation (research/colo/MODEL.md v2 §11): the learner process is also the engine, so it keeps
        # the stock allocator, as stock colocated does. Its state is RESIDENT (the colo validators require
        # policy.offload_optimizer_for_refit=false; offload_before_refit below moves nothing). Expandable segments are
        # off here because the colocated cycle still swaps the training parameters around the dedicated inference
        # model every step, and through expandable segments that churn remaps pages under training: Super-L8
        # policy_training 7.0 s (stock allocator) vs 30 s, and the speculative arm still OOMed with them,
        # so they were not its fix.
        colocated = _binds_colocated_generation(args, kwargs)
        if _binds_init_optimizer(args, kwargs) and not colocated:
            prior = getattr(self, "_pre_load_checkpoint_hook", None)

            def _pre_load(state: Any, model: Any) -> None:
                if prior is not None:
                    prior(state, model)
                enable_learner_expandable_segments()

            self._pre_load_checkpoint_hook = _pre_load
        super().__init__(*args, **kwargs)
        from nemo_rl.models.megatron import sticky_rounding

        # Sticky rounding (opt-in): every policy forward
        # (verification, training, log-probs) runs the theta_n anchor forward
        # in lockstep; theta_n is this rank's parameter shard on host, kept by
        # ``finish_train_step`` before each optimizer step.
        sticky_rounding.configure(
            sticky_rounding.sticky_rounding_enabled(self.cfg), self.model
        )
        # Bumped whenever the learner's parameters change (optimizer step, draft
        # weight swap): state cached from a forward (learner prefix-state cache)
        # is valid only for the version it was computed at.
        self._weights_version = 0
        if getattr(self, "optimizer", None) is not None:
            step = self.optimizer.step

            def step_and_bump(*args: Any, **kwargs: Any) -> Any:
                # A weight-sentinel digest in flight reads these parameters (sentinel_async).
                _sentinel_join(self, "optimizer step")
                snap = self.__dict__.get("_la_snap")
                if snap is not None:
                    # Lookahead drafts: the weights captured after the last step (or at lookahead_init) become the
                    # previous version before this step writes the next ones.
                    snap.commit()
                try:
                    return step(*args, **kwargs)
                finally:
                    self._weights_version += 1
                    if snap is not None:
                        self._la_step += 1
                        snap.capture(self._refit_named())

            self.optimizer.step = step_and_bump
            # Learner only (engines have no optimizer). Already set by the pre-load hook when a checkpoint
            # (pretrained or resume) was loaded; idempotent.
            self._expandable_segments = False if colocated else enable_learner_expandable_segments()
        self._prefix_cache = None

    def stream_begin_step(
        self,
        loss_fn: LossFunction,
        *,
        gbs: int,
        mbs: int,
        storage_device: str,
        max_buckets_per_group: int,
        max_open_groups: int,
        merge_pipeline_calls: bool = True,
    ) -> None:
        """Open an optimizer step whose gradient arrives in streamed chunks.

        ``merge_pipeline_calls``: run all chunks of a dispatch that share a
        gradient destination as one pipeline schedule call (see
        :func:`group_pipeline_calls`); ``False`` runs one call per chunk.
        """
        # Retained verification forwards of this step (fused_verify_train).
        self._fused_graphs = FusedGraphStore()
        self.begin_train_step(
            loss_fn=loss_fn,
            gbs=gbs,
            mbs=mbs,
            grad_streaming=GradStreamingSpec(
                storage_device=storage_device,
                max_buckets_per_group=max_buckets_per_group,
                max_open_groups=max_open_groups,
            ),
        )
        self._stream_busy_s = 0.0
        self._stream_merge_calls = merge_pipeline_calls
        # Rows / tokens / digest this rank received this step (streaming.row_audit).
        self._rows_received = [0, 0, 0]
        # Received valid tokens this rank took back out of the loss denominator (verify-in-train rejections;
        # run_checks.step_counts_merge).
        self._toks_uncounted = 0
        self._rows_uncounted = 0

    def _with_routes(self, data: BatchedDataDict) -> BatchedDataDict:
        """Give a batch ``routed_experts`` when sticky routing is on.

        Rows the driver gave anchors carry ``route_refs`` (``RouteRefs`` per
        row, None for undrafted rows); they are fetched for this pipeline
        stage's layers only (``_anchor_batch``). Rows without anchors get -1
        everywhere: anchor = self, i.e. plain routing, which is the sticky
        policy of undrafted rows.
        """
        if "input_ids" not in data:
            return data
        refs = data.pop("route_refs", None) if "route_refs" in data else None
        b, width = data["input_ids"].shape[:2]
        routed = self._anchor_batch(refs, b, width)
        if routed is not None:
            data["routed_experts"] = routed
        nuc = self._nucleus_batch(refs, b, width)
        if nuc is not None:
            data["nucleus_anchors"] = nuc
        return data

    def _nucleus_batch(self, refs, batch: int, width: int) -> torch.Tensor | None:
        """Sticky nucleus: ``nucleus_anchors`` [batch, width, cap] int32 (-1 = self).

        Index = logits row = absolute input position (the row that samples the
        next token), as the engine logged it. Only the last pipeline stage
        computes log-probs; rows without route refs (undrafted) are self.
        """
        from nemo_rl.models.megatron import exact_nucleus

        if not exact_nucleus.sticky():
            return None
        from megatron.core import parallel_state as ps

        if ps.get_pipeline_model_parallel_world_size() > 1 and not ps.is_pipeline_last_stage(
            ignore_virtual=True
        ):
            return None
        from thundersync_rl.routes import fetch_nucleus

        cap = exact_nucleus.anchor_cap()
        out = torch.full((batch, width, cap), -1, dtype=torch.int32)
        for i, a in enumerate(refs or []):
            if a is not None and i < batch:
                out[i] = fetch_nucleus(a, width, cap, ray.get)
        return out

    def train_microbatches(self, datas, *args, **kwargs):
        datas = [self._with_routes(d) for d in datas]
        shapes = [tuple(d["input_ids"].shape) for d in datas if "input_ids" in d]
        if shapes:  # largest micro-batch (rows x padded length) this step: the activation term of the peak
            self._step_max_mb_tokens = max(
                getattr(self, "_step_max_mb_tokens", 0), max(int(s[0]) * int(s[-1]) for s in shapes)
            )
        with mem_debug.capture(self.rank, f"train_microbatches {shapes}"):
            return super().train_microbatches(datas, *args, **kwargs)

    # -- EDV (edv.py, DESIGN §22.9): the capture of a group whose deferred row was not accepted whole is dropped -----

    def nl_row_etas(self, seeds: list[int], lengths: list[int]) -> list[float]:
        """Each row's final coin η_g: the last value of its verifier's first draw rand(g + 1) from its row seed
        (``vit.eta_last``, the generator call ``ds2.grid_scan_sharded`` makes first), on this rank's device (CUDA
        Philox is device-specific, so the driver cannot draw it). 8 B per row, no state."""
        from thundersync_rl import vit

        dev = torch.device("cuda", torch.cuda.current_device())
        with torch.no_grad():
            vals = [vit.eta_last(int(s), int(g), dev) for s, g in zip(seeds, lengths)]
            return torch.stack(vals).cpu().tolist() if vals else []

    def nl_edv_arm(self, on: bool) -> None:
        """Arm (every rank alike) the per-call surprise agreement before each exact-accumulator capture."""
        from thundersync_rl import edv

        self._edv_armed = bool(on)
        self._edv_log = []
        edv.take()

    def _capture_may_discard(self) -> bool:
        return bool(getattr(self, "_edv_armed", False))

    def _pre_capture_discard(self, state) -> bool:
        """Agree over every learner rank on the call's surprises (one all-reduce of 64 int32), log them, and drop
        the capture when any deferred row of the group failed. Ranks holding no deferred row contribute zeros."""
        from thundersync_rl import edv

        dec = edv.take()
        mask = edv.surprise_mask(dec)
        bits = torch.tensor([(mask >> s) & 1 for s in range(64)], dtype=torch.int32, device="cuda")
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(bits, op=torch.distributed.ReduceOp.MAX)
        agreed = sum(int(b) << s for s, b in enumerate(bits.tolist()))
        self._edv_log.append({"mask": agreed, "decisions": dec})
        return agreed != 0

    def nl_edv_take(self) -> list:
        """The calls' surprise masks (agreed) and this rank's decisions (slot, b_g, η_g, accepted) since the last take."""
        out, self._edv_log = list(getattr(self, "_edv_log", [])), []
        return out

    @_lg_feature("logprob")
    def get_logprobs(self, *, data, **kwargs):
        return super().get_logprobs(data=self._with_routes(data), **kwargs)

    def _run_train_pieces(self, state, pieces, variable_shapes=False):
        """Pipeline call; under ``verify_yield`` its micro-batch boundaries may run
        posted continuation verifications (``verify_yield.py``)."""
        if getattr(self, "_yield_ok", False):
            import dataclasses

            pieces = [
                dataclasses.replace(
                    p,
                    data_iterator=verify_yield.boundary_iter(
                        p.data_iterator, self._yield_point, self._yield_first
                    ),
                )
                for p in pieces
            ]
        return super()._run_train_pieces(state, pieces, variable_shapes)

    def finish_train_step(self) -> dict[str, Any]:
        """Finish the train step; optionally perturb expert gradients (test only).

        ``THUNDERSYNC_DEBUG_EXPERT_GRAD_NOISE=<scale>`` adds seeded noise to the MoE
        expert gradients before each optimizer step, so expert weights change even when
        every reward is 0 (random tiny models). Exercises refit into the engine's expert
        weights: any stale engine weight shows up in the mismatch check.
        """
        self._maybe_master_jitter()
        # Anchor weights of the next step's policy: the weights its drafts come from.
        snapshot_anchor_weights(self.optimizer, getattr(self, "_staged_draft_kind", None))
        self._staged_draft_kind = None
        # Drift logging (drift_log.py): read-only probe around the optimizer step.
        probe = self._drift_probe_or_none()
        if probe is None:
            return self._finish_train_step_maybe_noisy()
        step = self.optimizer.step

        def probed_step(*args, **kwargs):
            probe.pre(self.optimizer)
            result = step(*args, **kwargs)
            probe.post(self.optimizer, result)
            return result

        self.optimizer.step = probed_step
        try:
            metrics = self._finish_train_step_maybe_noisy()
        finally:
            self.optimizer.step = step
        metrics["drift"] = dict(probe.last)
        return metrics

    def _maybe_master_jitter(self) -> None:
        """THUNDERSYNC_MASTER_JITTER=1 (research): spread the fp32 masters' sub-ulp residuals once, before the first
        optimizer step of a fresh run (master_jitter.py). Stored bf16 weights are unchanged (checked)."""
        from thundersync_rl import master_jitter

        if not master_jitter.enabled() or getattr(self, "_master_jitter_done", False):
            return
        self._master_jitter_done = True
        from thundersync_rl.draft_weights import _dist_optimizers

        opts = _dist_optimizers(self.optimizer)
        if any(len(o.optimizer.state) for o in opts):
            print("[master_jitter] optimizer state present (resumed run): not applied", flush=True)
            return
        rank = torch.distributed.get_rank()
        # Seed by element identity (parameter name + the shard's start in the grad buffer), NOT by rank: parameters
        # replicated over TP (norms, biases, ...) keep one master copy per TP rank, and those copies must get identical
        # offsets, or their bf16 values diverge after the first optimizer step (Super TP2 exactness failure, 10-27).
        chunks = self.model if isinstance(self.model, list) else [self.model]
        name_of = {id(p): n for c in chunks for n, p in c.named_parameters()}
        acc = torch.zeros(3, dtype=torch.float64, device="cuda")
        with torch.no_grad():
            for o in opts:
                for grp, mgrp in zip(o.shard_fp32_from_float16_groups, o.model_float16_groups):
                    for m, mp in zip(grp, mgrp):
                        start = o._get_model_param_range_map(mp)["gbuf_world_in_bucket"].start
                        n, bad, phi = master_jitter.jitter_(m, master_jitter.seed_of(name_of[id(mp)], start))
                        acc += torch.tensor([n, bad, phi], dtype=torch.float64, device="cuda")
        torch.distributed.all_reduce(acc)
        if rank == 0:
            print(f"[master_jitter] applied: {int(acc[0])} elements jittered, {int(acc[1])} reverted (bf16 would change), "
                  f"phi = 4 mean|r|/ulp = {float(acc[2] / acc[0].clamp_min(1)):.4f}", flush=True)

    def _drift_probe_or_none(self):
        """The learner's DriftProbe (THUNDERSYNC_DRIFT_LOG, default on), None when off or unsupported."""
        from thundersync_rl import drift_log

        if not drift_log.enabled() or getattr(self, "optimizer", None) is None:
            return None
        probe = getattr(self, "_drift_probe", None)
        if probe is None:
            from thundersync_rl.draft_weights import _dist_optimizers

            try:
                _dist_optimizers(self.optimizer)  # same layouts as momentum drafts; same verdict on every rank
            except NotImplementedError as exc:
                print(f"[drift_log] off: {exc}", flush=True)
                self._drift_probe = False
                return None
            probe = self._drift_probe = drift_log.DriftProbe()
            if self.rank == 0:
                print(f"[drift_log] optimizer-shard probe on (stride {drift_log.stride()})", flush=True)
        return probe or None

    def _finish_train_step_maybe_noisy(self) -> dict[str, Any]:
        scale = os.environ.get("THUNDERSYNC_DEBUG_EXPERT_GRAD_NOISE")
        if not scale:
            return super().finish_train_step()
        step = self.optimizer.step
        n_done = getattr(self, "_debug_noise_steps", 0)

        def noisy_step(*args, **kwargs):
            # Same seed and parameter order on every rank: replicas of an expert agree.
            g = torch.Generator(device="cuda").manual_seed(1234 + n_done)
            chunks = self.model if isinstance(self.model, list) else [self.model]
            for chunk in chunks:
                for name, p in chunk.named_parameters():
                    if (
                        ".experts." in name
                        and getattr(p, "main_grad", None) is not None
                    ):
                        p.main_grad.add_(
                            torch.randn(
                                p.main_grad.shape,
                                generator=g,
                                device=p.main_grad.device,
                            )
                            * float(scale)
                        )
            return step(*args, **kwargs)

        self.optimizer.step = noisy_step
        try:
            return super().finish_train_step()
        finally:
            self.optimizer.step = step
            self._debug_noise_steps = n_done + 1

    def load_draft_weights(self, kind: str, steps: int = 1) -> dict[str, float]:
        """Replace the model parameters by draft weights ``kind`` (collective).

        ``"momentum"``: theta_hat_{k+1} = AdamW step of the optimizer state with
        gradient 0 (``draft_weights.py``). For the refit that feeds the engine
        the next step's drafting weights; no forward may run until
        ``restore_training_weights``. The fp16 expert-weight cache is dropped
        here and at the restore: it is keyed by (param id, data_ptr), which the
        in-place swap keeps, so a forward that ran while the draft weights were
        loaded would have left theta_hat casts that hit again after the restore
        (silently drafting-weight experts in verification and training). A few
        linears are recast on the next forward.
        """
        _call_log(self, "load_draft_weights")
        if getattr(self, "_draft_swap", None) is not None:
            raise RuntimeError("draft weights already loaded")
        _sentinel_join(self, "load_draft_weights")
        check_no_expert_bias_updates(self.model)
        self._weights_version += 1
        # steps = h > 1: Psi_h (anchor epochs); 1: theta_hat_{k+1} as before.
        self._draft_swap: Optional[DraftSwap] = load_draft(
            self.optimizer, kind, **({} if int(steps) == 1 else {"steps": int(steps)})
        )
        # The next step's drafts come from these weights (finish_train_step).
        self._staged_draft_kind = kind
        self._log_mem(f"load_draft_weights {kind}")
        return dict(self._draft_swap.stats)

    def restore_training_weights(self) -> dict[str, float]:
        """Put the training weights theta_k back, bitwise (after ``load_draft_weights``)."""
        _call_log(self, "restore_training_weights")
        swap, self._draft_swap = self._draft_swap, None
        if swap is None:
            raise RuntimeError("no draft weights loaded")
        _sentinel_join(self, "restore_training_weights")
        self._weights_version += 1
        return restore_training(swap)

    def stage_engine_weights(self) -> dict[str, float]:
        """Engine rank: keep the live (draft) weights in host staging (``EngineStage``).

        Called right after the draft weights were refit into this engine, before
        the refit of the step's rollout weights; ``load_engine_weights`` puts
        them back once the rollouts are done.
        """
        if getattr(self, "_engine_stage", None) is None:
            self._engine_stage = EngineStage()
        return self._engine_stage.stage(
            self.model, checksum=env_flag("THUNDERSYNC_MISMATCH_CHECK")
        )

    def load_engine_weights(self) -> dict[str, float]:
        """Engine rank: replace the live weights by the staged draft weights."""
        stage = getattr(self, "_engine_stage", None)
        if stage is None:
            raise RuntimeError("load_engine_weights before stage_engine_weights")
        _sentinel_join(self, "load_engine_weights")
        return stage.load(
            self.model, checksum=env_flag("THUNDERSYNC_MISMATCH_CHECK")
        )

    def _layout_groups(self) -> dict[str, Any]:
        """Global ranks of this rank's TP x CP, ETP x EP and EP groups (cached; None when the group is trivial)."""
        cached = self.__dict__.get("_layout_groups_cache")
        if cached is not None:
            return cached
        from megatron.core import parallel_state as ps

        def ranks(group) -> Optional[tuple[int, ...]]:
            r = tuple(int(x) for x in torch.distributed.get_process_group_ranks(group))
            return r if len(r) > 1 else None

        ep = ps.get_expert_model_parallel_world_size() or 1
        out = {
            "tp_cp": ranks(ps.get_tensor_and_context_parallel_group()),
            "etp_ep": ranks(ps.get_expert_tensor_and_model_parallel_group()),
            "ep": ranks(ps.get_expert_model_parallel_group()) if ep > 1 else None,
        }
        self._layout_groups_cache = out
        return out

    def device_total_gb(self) -> float:
        """This rank's device memory (GiB), for the startup micro-batch memory budget (mem_budget)."""
        return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**30

    def layout_ids(self) -> dict[str, Any]:
        """Startup layout report (``run_checks.layout_violations``): this rank's groups, data-parallel rank and
        verify-yield domain. Host-side group lookups only."""
        from megatron.core import parallel_state as ps

        return {
            "rank": int(torch.distributed.get_rank()),
            "dp": int(ps.get_data_parallel_rank()),
            **self._layout_groups(),
            "yield_domain": self._yield_domain_key(),
        }

    # -- lookahead drafts (X5, lookahead_draft.py) ------------------------------------------------------------------
    def _refit_named(self) -> dict[str, torch.Tensor]:
        """The native MCore refit's source tensors of this rank (``get_refit_tensor_dict`` of the unwrapped core)."""
        from megatron.core.resharding.refit import _unwrap_model_cores
        from megatron.core.resharding.utils import get_refit_tensor_dict

        src_core, _, _ = _unwrap_model_cores(self.model, None)
        return get_refit_tensor_dict(src_core)

    def lookahead_init(self, alpha: Any, load_dir: Optional[str] = None, start_step: int = 0) -> dict[str, Any]:
        """Learner: keep theta_{k-1} for lookahead drafts (restored from ``load_dir`` on a resume)."""
        from thundersync_rl.lookahead_draft import PrevSnapshot

        cfg = getattr(self, "cfg", {}) or {}
        ddp = ((cfg.get("megatron_cfg") or {}).get("distributed_data_parallel_config") or {})
        if ddp.get("overlap_param_gather"):
            raise RuntimeError("lookahead drafts need overlap_param_gather=false (theta_k complete at the step end)")
        self._la_alphas = tuple(float(a) for a in (alpha if isinstance(alpha, (list, tuple)) else [alpha]))
        self._la_step = int(start_step)  # optimizer steps taken: the live weights are theta_{_la_step}
        self._la_snap = PrevSnapshot()
        restored = 0
        if load_dir:
            path = os.path.join(load_dir, f"rank{torch.distributed.get_rank()}.pt")
            if os.path.exists(path):
                state = torch.load(path, map_location="cpu", weights_only=True)
                saved = state.get("alpha", list(self._la_alphas))
                saved = tuple(float(x) for x in (saved if isinstance(saved, (list, tuple)) else [saved]))
                if saved != self._la_alphas:
                    raise RuntimeError(f"lookahead resume: alpha {self._la_alphas} != saved {saved}")
                self._la_snap.load_state(state["prev"], self._refit_named())
                restored = len(state["prev"])
        # The live weights (theta_0, or theta_k on a resume) become the previous version at the next optimizer step.
        named = self._refit_named()
        self._la_snap.capture(named)
        host = 2 * sum(t.numel() * t.element_size() for t in named.values() if t.is_floating_point())
        return {"rank": int(torch.distributed.get_rank()), "restored": restored, "host_gib": host / 2**30}

    def lookahead_save(self, path: str) -> int:
        """Learner: save theta_{k-1} (pinned host) with the checkpoint, for a bitwise resume of theta~."""
        snap = self.__dict__.get("_la_snap")
        if snap is None:
            return 0
        os.makedirs(path, exist_ok=True)
        state = snap.state()
        torch.save({"alpha": list(self._la_alphas), "step": self._la_step, "prev": state},
                   os.path.join(path, f"rank{torch.distributed.get_rank()}.pt"))
        return len(state)

    def _la_extrapolated(self, t: torch.Tensor) -> torch.Tensor:
        """theta~ of one live tensor (itself when no previous version is kept: before the first step)."""
        from thundersync_rl.lookahead_draft import extrapolate

        snap = self.__dict__.get("_la_snap")
        prev = snap.prev_for(t) if snap is not None else None
        if prev is None:
            return t
        return extrapolate(t.detach(), prev, self._la_alpha)

    @property
    def _la_alpha(self) -> float:
        """alpha of the live version theta_{_la_step} (lookahead_draft.alpha_at)."""
        from thundersync_rl.lookahead_draft import alpha_at

        return alpha_at(self._la_alphas, self._la_step)

    def swap_weights_via_reshard(self, is_source: bool) -> bool:
        """Native MCore refit; a lookahead drafter refit (``select_refit_group(lookahead=True)``) sends theta~."""
        la = self.__dict__.pop("_la_send_next", False)
        if not (is_source and la):
            return super().swap_weights_via_reshard(is_source)
        from megatron.core.resharding.copy_services.nccl_m2n_copy_service import NCCLM2NCopyService
        from megatron.core.resharding.refit import swap_model_weights

        from thundersync_rl.lookahead_draft import make_send_transform

        if self.refit_execution_batch_bytes is None:
            raise RuntimeError("Native MCore refit was not initialized.")
        if isinstance(self.refit_copy_service, NCCLM2NCopyService):
            raise RuntimeError("lookahead drafts: the NCCL M2N copy service takes no source-side transform")
        snap = self.__dict__.get("_la_snap")
        if snap is None:
            raise RuntimeError("lookahead refit requested but lookahead_init was not called")
        tf = make_send_transform(snap, self._la_alpha)
        swap_model_weights(
            self.model,
            None,
            refit_method=self.refit_copy_service,
            group=self.refit_pg,
            src_rank_offset=0,
            dst_rank_offset=self.refit_dst_rank_offset,
            transform=tf,
            execution_batch_bytes=self.refit_execution_batch_bytes,
        )
        self._la_last_sent = tf.sent
        if torch.distributed.get_rank() == 0:
            print(f"[lookahead] drafter refit: sent theta~ for {tf.sent} slices (alpha {self._la_alpha})", flush=True)
        return True

    @torch.no_grad()
    def sentinel_weight_digest(
        self,
        dtypes: Optional[dict[str, str]] = None,
        learner: bool = False,
        lookahead: bool = False,
    ) -> tuple[int, dict, dict]:
        """(tensor-parallel rank(s), {canonical name: digest}, coverage) of this rank now.

        The rank key is ``(dense TP rank, expert TP rank)`` once model parallelism is
        initialized (``sentinel.shard_rank`` picks the one a tensor is sharded by).

        Engine and learner ranks both run it (``sentinel.WeightSentinel``).
        Engine: the generation model's parameters (fp16-held GEMM / router / head
        weights in their fp16 storage), ``expert_bias`` buffers, and the tensors
        the expert kernels actually read (concatenated / fp16 expert buffers,
        ``<param>#buf``). Learner (``learner=True``): the same names, each cast
        to the engine's storage dtype (``dtypes``), compute aliases digested from
        their source parameter; ``expert_bias`` must equal its first digest on
        this rank.
        """
        self._assert_no_refit("sentinel_weight_digest")
        return self._sentinel_digest_now(dtypes, learner, lookahead=lookahead)

    def sentinel_digest_begin(
        self,
        token: int,
        dtypes: Optional[dict[str, str]] = None,
        learner: bool = True,
        lookahead: bool = False,
    ) -> bool:
        """Start ``sentinel_weight_digest(dtypes, learner)`` off the training / decode path
        (``sentinel_async``: side stream + thread, ordered after the work queued now; every weight write - optimizer
        step, draft load / restore, refit, engine stage load - joins it first). ``sentinel_digest_result(token)``
        returns it."""
        self._assert_no_refit("sentinel_digest_begin")
        from thundersync_rl import sentinel_async

        sentinel_async.begin(
            self,
            int(token),
            lambda: self._sentinel_digest_now(
                dtypes, learner, chunk=sentinel_async.ASYNC_CHUNK, lookahead=lookahead
            ),
        )
        return True

    async def sentinel_digest_result(self, token: int) -> tuple[Any, float, float]:
        """Learner: (digest as ``sentinel_weight_digest`` returns it, digest time s, parameter-write wait s)."""
        from thundersync_rl import sentinel_async

        return await sentinel_async.result(self, int(token))

    @torch.no_grad()
    def _sentinel_digest_now(
        self,
        dtypes: Optional[dict[str, str]] = None,
        learner: bool = False,
        chunk: Optional[int] = None,
        lookahead: bool = False,
    ) -> tuple[Any, dict, dict]:
        from megatron.core import parallel_state as ps

        from thundersync_rl.sentinel import (
            WeightSentinelError,
            canonical_named_tensors,
            compute_tensors,
            digest_many,
            digest_named,
        )

        ep = ps.get_expert_model_parallel_rank() if ps.model_parallel_is_initialized() else 0
        tp = ps.get_tensor_model_parallel_rank() if ps.model_parallel_is_initialized() else 0
        if ps.model_parallel_is_initialized():
            # Expert tensors are sharded by the expert-TP rank, not the dense TP rank: with
            # ETP 1 the dense TP rank holding expert e depends on the EP size, so engine and
            # learner at different EP sizes would key one expert differently (not compared).
            tp = (tp, ps.get_expert_tensor_parallel_rank())
        model = self.model
        if not learner and getattr(self, "_gen_model", None) is not None:
            model = self._gen_model()  # the engine's (inference-layout) model
        named = list(canonical_named_tensors(model, ep_rank=ep))
        if not learner and os.environ.get("THUNDERSYNC_DEBUG_SENTINEL_FLIP", "") != "":
            # Negative control (sentinel.debug_flip_byte): one flipped weight byte on engine rank (TP 0, EP 0).
            from thundersync_rl.sentinel import debug_flip_byte

            n = self.__dict__["_debug_flip_calls"] = self.__dict__.get("_debug_flip_calls", 0) + 1
            if tp == (0, 0) or tp == 0:
                if ep == 0 and (hit := debug_flip_byte(named, n)) is not None:
                    print(f"[debug sentinel flip] engine digest {n}: flipped one byte of {hit}", flush=True)
        if learner and lookahead:
            # The drafting lane holds theta~ (lookahead drafts): digest theta~ of each tensor, a group of tensors
            # at a time (transient device memory bounded by the group, not the model).
            out, group, size = {}, [], 0
            for name, t in named:
                group.append((name, self._la_extrapolated(t)))
                size += t.numel() * t.element_size()
                if size >= (1 << 30):
                    out.update(digest_named(group, dtypes, chunk=chunk))
                    group, size = [], 0
            if group:
                out.update(digest_named(group, dtypes, chunk=chunk))
        else:
            out = digest_named(named, dtypes, chunk=chunk)
        cov = {
            "params": sum(1 for n, _ in named if not n.endswith("expert_bias")),
            "fp16_params": sum(1 for _, t in named if t.dtype == torch.float16),
            "expert_bias": sum(1 for n, _ in named if n.endswith("expert_bias")),
        }
        if not learner:
            comp = list(compute_tensors(model, ep_rank=ep))
            out.update(digest_many((key, t, None) for key, t, _ in comp))
            cov["compute_buffers"] = len(comp)
            cov["compute_buffers_aliased"] = sum(1 for *_, a in comp if a)
        if learner:
            bias = {k: v for k, v in out.items() if k.endswith("expert_bias")}
            ref = self.__dict__.setdefault("_sentinel_expert_bias", dict(bias))
            moved = sorted(k for k in bias if ref.get(k) != bias[k])
            if moved:
                raise WeightSentinelError(
                    f"weight sentinel: learner rank {self.rank} expert_bias changed "
                    f"during training: {moved[:10]} (bias updates are not supported)"
                )
        return tp, out, cov

    @torch.no_grad()
    def sentinel_head_stash_check(self, version: int) -> bool:
        """Learner: the LM head stashed as ``version`` is bitwise the live head now.

        Called while the drafting weights are loaded (right after the stash and
        the refit that the weight sentinel compared with the engine), so the
        stash q rows are rebuilt with equals the engine's drafting head.
        """
        from thundersync_rl.sentinel import WeightSentinelError

        head = self._lm_head()
        if head is None:
            return True
        live = head[1]
        stash = getattr(self, "_head_stash", {}).get(version)
        if stash is None or stash.shape != live.shape or not torch.equal(stash, live.detach()):
            raise WeightSentinelError(
                f"weight sentinel: learner rank {self.rank} LM-head stash of version "
                f"{version} differs from the drafting head the engine was refit with"
            )
        return True

    @torch.no_grad()
    def resume_state_digest(self) -> dict[str, Any]:
        """Debug (resume tests): bitwise digests of this rank's training state.

        Parameters, optimizer state (fp32 masters, Adam moments, step and lr of
        every group) and the momentum draft theta_hat it implies (computed, not
        written). Equal at a checkpoint save and right after loading it.
        """
        import hashlib

        from megatron.core import parallel_state as ps

        from thundersync_rl.draft_weights import CHUNK_ELEMS, _adam_state, _draft_chunks, _shards
        from thundersync_rl.sentinel import canonical_named_tensors, tensor_digest

        def fold(items) -> str:
            h = hashlib.blake2b(digest_size=8)
            for x in items:
                h.update(repr(x).encode())
            return h.hexdigest()

        ep = ps.get_expert_model_parallel_rank()
        params = fold(
            (n, tensor_digest(t)) for n, t in sorted(
                canonical_named_tensors(self.model, ep_rank=ep), key=lambda x: x[0]
            )
        )
        out: dict[str, Any] = {"rank": self.rank, "params": params}
        la_snap = self.__dict__.get("_la_snap")
        if la_snap is not None:
            # Lookahead drafts: theta_{k-1} (committed) defines theta~ of the next drafter refits.
            out["lookahead"] = {"alpha": list(self._la_alphas), "step": self._la_step, "prev": la_snap.digest()}
        try:
            shards = list(_shards(self.optimizer))
        except NotImplementedError as exc:
            out["optimizer"] = f"n/a ({exc})"
            return out
        opt, draft, groups = [], [], []
        for sh in shards:
            m, v = _adam_state(sh.inner, sh.key)
            main = sh.main if sh.main is not None else sh.key
            opt.append((tensor_digest(main), tensor_digest(m), tensor_digest(v)))
            for _, d in _draft_chunks(sh, CHUNK_ELEMS):
                draft.append(tensor_digest(d))
            g = sh.group
            groups.append((int(g.get("step", 0)), float(g["lr"]), float(g["weight_decay"])))
        out["optimizer"] = fold(opt)
        out["theta_hat"] = fold(draft)
        out["groups"] = fold(groups)
        out["first_group"] = groups[0] if groups else None
        return out

    @torch.no_grad()
    def set_check_env(self, env: dict[str, Optional[str]]) -> bool:
        """Sampled exactness checks (checks_sched.py, THUNDERSYNC_CHECK_EVERY): set / unset the
        check switches in this worker process; they are read at call time."""
        from thundersync_rl.checks_sched import CHECK_VARS, apply_env

        apply_env({k: v for k, v in env.items() if k in CHECK_VARS})
        return True

    def engine_weight_checksum(self) -> float:
        """Engine rank (debug): checksum of the weights this rank holds now.

        Cohort lanes read it when a cohort is pinned and when it completes
        (``LaneStepOps.fingerprint``); one pass over the rank's parameters.
        """
        return EngineStage._checksum(EngineStage._params(self.model))

    async def ordered_call(self, seq: int, method: str, **kwargs) -> Any:
        """Run learner method ``method`` as this rank's ``seq``-th ordered call.

        The learner workers are asynchronous Ray actors (they have async
        methods), which may start queued calls in a different order on each
        rank. The driver numbers every learner call it sends while several are
        in flight (training dispatches, verification forwards) and each rank
        runs them strictly in that order: call ``seq`` waits until calls
        0..seq-1 have run here. The method itself is synchronous and runs on
        this actor's event loop, so calls never overlap. Every rank therefore
        runs the same sequence, and their collectives pair up, by construction.
        """
        cv = getattr(self, "_ordered_cv", None)
        if cv is None:
            cv = self._ordered_cv = asyncio.Condition()
            self._ordered_next = 0
        async with cv:
            while self._ordered_next != seq:
                if seq < self._ordered_next:
                    raise RuntimeError(f"ordered learner call {seq} arrived after {self._ordered_next}")
                await cv.wait()
        failed = getattr(self, "_ordered_failed", None)
        if failed is not None:
            # Poisoned: an earlier ordered call raised on this rank (e.g. CUDA
            # OOM inside a training forward). Its peers may still be blocked
            # in that call's collectives (EP all-to-all / all-gather, TP, PP);
            # running this call would issue new collectives on the same
            # communicators that can never pair up (ts_por dec-on/dec-off,
            # i3-*, i4-all-mom: rank 0 entered verify's EP lockstep
            # all_gather while ranks 1-3 waited in the backward's MoE
            # dispatch -> 600 s NCCL watchdog). Fail without running it.
            self._ordered_next += 1
            async with cv:
                cv.notify_all()
            raise RuntimeError(
                f"learner rank {getattr(self, 'rank', '?')} is poisoned by failed "
                f"ordered call {failed[0]} ({failed[1]}); not running call {seq} ({method})"
            )
        _check_v0_no_interleave(self, method, kwargs)
        t0 = time.perf_counter()
        # verify_yield: the running call (a training dispatch's boundaries may
        # run posted verifications numbered after it).
        y0 = verify_yield.YieldWorkerMixin._yield_begin_call(self, seq)
        try:
            return getattr(self, method)(**kwargs)
        except BaseException as exc:
            self._ordered_failed = (f"{seq}:{method}", f"{type(exc).__name__}: {exc}"[:500])
            import traceback

            # ordered_call failures were never printed on the worker (only the
            # driver's RayTaskError, if it ever read the ref): log them here.
            print(
                f"[ordered_call FAILED] rank={getattr(self, 'rank', '?')} seq={seq} "
                f"method={method}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                flush=True,
            )
            raise
        finally:
            # Learner seconds per method (``ordered_digest``): verification vs training.
            busy = self.__dict__.setdefault("_ordered_busy", {})
            inline = verify_yield.YieldWorkerMixin._yield_end_call(self, y0)
            busy[method] = busy.get(method, 0.0) + time.perf_counter() - t0 - inline
            if inline:
                busy["verify_inlined"] = busy.get("verify_inlined", 0.0) + inline
            self._ordered_next += 1
            # What this rank ran, in order (``ordered_digest``).
            self._ordered_crc = zlib.crc32(
                f"{seq}:{method};".encode(), getattr(self, "_ordered_crc", 0)
            )
            async with cv:
                cv.notify_all()

    def ordered_digest(self) -> tuple[int, int]:
        """(ordered calls run, crc32 of their (number, method) sequence) on this rank."""
        return getattr(self, "_ordered_next", 0), getattr(self, "_ordered_crc", 0)

    def ordered_busy(self) -> dict[str, float]:
        """Seconds this rank spent in each ordered method since the last call (reset)."""
        busy, self._ordered_busy = self.__dict__.get("_ordered_busy", {}), {}
        return busy

    def fp16_range_stats(self) -> dict[str, float]:
        """Learner rank: fp16 range monitor since the last call (silent_bug_sweep.md).

        The first call turns the activation counter on (fp16_gemm.range_count at
        every learner fp16 cast; device-side, no host sync). Returns the elements
        above fp16's max that the casts turned into inf, and the parameters'
        amax (one multi-tensor norm over this rank's weights): weights above
        65504 become inf in the engine's fp16 copy.
        """
        from nemo_rl.models.megatron import fp16_gemm

        from nemo_rl.models.megatron import deferred_head

        fp16_gemm.range_monitor(True)
        out = {f"fp16/act_{k}": v for k, v in fp16_gemm.range_take().items()}
        for k, v in deferred_head.materialized_take().items():
            out[f"learner/head_materialized_{k}"] = float(v)
        params = [p for p in self.model.parameters() if p.is_floating_point() and p.numel()]
        if params:
            with torch.no_grad():
                amax = torch.stack(torch._foreach_norm(params, float("inf"))).float()
            out["fp16/weight_amax"] = float(amax.max().item())
            out["fp16/weights_over_max"] = float((amax > fp16_gemm.FP16_MAX).sum().item())
        return out

    def engine_step_stats(self) -> dict[str, float]:
        """Engine rank (debug, ``THUNDERSYNC_ENGINE_PROF``): step timing since the last call."""
        from thundersync_rl import engine_prof

        return engine_prof.summary()

    # -- refit-in-flight guard (research/ray_order_audit.md, "latent: mid-refit window") --
    # The refit transfers below are ``async def`` and run off the actor's event loop
    # (``await_off_loop``), so the actor serves other calls meanwhile. Calls that read the
    # live weights (head stash, engine verification, weight digests) must never run then:
    # today the driver's schedule guarantees it; this turns a regression into an error.
    def _refit_guard_enter(self) -> None:
        _sentinel_join(self, "refit")  # an async sentinel digest reads the weights the refit writes (engine)
        self.__dict__["_ts_refit_in_flight"] = self.__dict__.get("_ts_refit_in_flight", 0) + 1

    def _refit_guard_exit(self) -> None:
        self.__dict__["_ts_refit_in_flight"] = self.__dict__.get("_ts_refit_in_flight", 1) - 1

    def _assert_no_refit(self, what: str) -> None:
        if self.__dict__.get("_ts_refit_in_flight", 0) > 0:
            raise RuntimeError(
                f"{what} called while a weight refit is in flight on this worker "
                "(it would read half-updated weights; research/ray_order_audit.md)"
            )

    def offload_before_refit(self, *args, **kwargs):
        _sentinel_join(self, "offload_before_refit")
        if getattr(self, "is_generation_colocated", False) and not getattr(self, "offload_optimizer_for_refit", True):
            # Resident colocated learner: optimizer state and grad buffers stay on the device across the engine phase
            # (the engine's KV / state buffer is sized by rule beside them, research/colo/jobs/engine_buf.py).
            self.finalize_async_save()
            return None
        return super().offload_before_refit(*args, **kwargs)

    # Colocated transitions (research/colo/MODEL.md v2 §10): the per-step learner moves around the engine phase,
    # timed on rank 0 (one line per call; the stock loop's training_prep / weight_sync timers cover several of them).
    def _colo_timed(self, what: str, fn, *args, **kwargs):
        if not getattr(self, "is_generation_colocated", False) or getattr(self, "rank", 0) != 0:
            return fn(*args, **kwargs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        print(f"[colo transition] {what} {time.perf_counter() - t0:.3f}s alloc={torch.cuda.memory_allocated() / 2**30:.1f}GiB",
              flush=True)
        return out

    def move_optimizer(self, device: str):
        return self._colo_timed(f"move_optimizer->{device}", super().move_optimizer, device)

    def move_model(self, model, device: str, *args, **kwargs):
        return self._colo_timed(f"move_model->{device} {kwargs}", super().move_model, model, device, *args, **kwargs)

    def prepare_for_training(self, *args, **kwargs):
        return self._colo_timed("prepare_for_training", super().prepare_for_training, *args, **kwargs)

    def finish_train_step_presharded(self) -> dict[str, Any]:
        """The optimizer step, plus the router-freeze fingerprint of this rank's routers after it (user rule 11-02;
        router_freeze.py): the stage requires it unchanged from the first step on every rank."""
        out = super().finish_train_step_presharded()
        from thundersync_rl.router_freeze import router_fingerprint

        out["nl_router_fp"] = router_fingerprint(self.model)
        return out

    def offload_after_refit(self, *args, **kwargs):
        _sentinel_join(self, "offload_after_refit")
        return super().offload_after_refit(*args, **kwargs)

    def load_checkpoint(self, *args, **kwargs):
        _sentinel_join(self, "load_checkpoint")
        return super().load_checkpoint(*args, **kwargs)

    async def update_weights_from_collective(self, *args, **kwargs):
        self._refit_guard_enter()
        try:
            return await super().update_weights_from_collective(*args, **kwargs)
        finally:
            self._refit_guard_exit()

    async def nccl_reshard_refit(self, *args, **kwargs):
        self._refit_guard_enter()
        try:
            return await super().nccl_reshard_refit(*args, **kwargs)
        finally:
            self._refit_guard_exit()

    def keep_refit_group(self, lane: int) -> None:
        """Learner rank: file the live refit communicators under engine lane ``lane``.

        Cohort lanes (lanes.py) give every engine lane its own refit
        communicators (learner ranks + that lane's ranks). ``init_collective``
        releases the previous ``model_update_group``, so the live state is moved
        out first; the next lane's init then builds new ones. The native MCore
        refit plan is the same for every lane (same group-rank layout), but each
        lane's engine ranks build it collectively at their init:
        ``clear_refit_plan_cache`` makes the learner join that build.
        """
        groups = self.__dict__.setdefault("_lane_refit_groups", {})
        state = {a: getattr(self, a, None) for a in _LANE_REFIT_ATTRS}
        if state["model_update_group"] is None and state["refit_pg"] is None:
            raise RuntimeError("keep_refit_group: no refit communicator to keep")
        groups[lane] = state
        self.model_update_group = None

    def select_refit_group(self, lane: int, lookahead: bool = False) -> None:
        """Learner rank: the next refit goes to engine lane ``lane`` only (``lookahead``: it sends theta~, a
        drafter-only lane; lookahead_draft.py)."""
        for attr, value in self._lane_refit_groups[lane].items():
            setattr(self, attr, value)
        if lookahead:
            if self.__dict__.get("_la_snap") is None:
                raise RuntimeError("select_refit_group(lookahead=True) before lookahead_init")
            self._la_send_next = True
        else:
            self.__dict__.pop("_la_send_next", None)

    def clear_refit_plan_cache(self) -> None:
        """Learner rank: drop MCore's cached refit plans (before a new lane's native init)."""
        from megatron.core.resharding.refit import clear_plan_cache

        clear_plan_cache()

    @torch.no_grad()
    def engine_weight_digest(self) -> dict[str, list]:
        """Engine rank (debug): bitwise digest of every parameter and buffer it holds.

        Per tensor (named_parameters, then named_buffers, sorted by name) a
        position-weighted sum of its raw bits (``refit_m2n.tensor_digest``): any
        bit difference between two refits of the same weights changes it, with
        probability ~1 - 2^-61. Includes derived caches (e.g. the Mamba
        ``-exp(A_log)`` cache) and the fp16 expert buffers.
        """
        from thundersync_rl.refit_m2n import model_digest

        names, digests, dtypes = model_digest(self.model)
        return {"names": names, "digests": digests, "dtypes": dtypes}

    @torch.no_grad()
    def engine_poison_refit_weights(self) -> int:
        """Engine rank (debug): overwrite every refit tensor with a sentinel.

        The refit crosscheck refits the same weights twice; poisoning in between
        makes a tensor the second transport skipped show up as a digest mismatch
        instead of passing on the first transport's values.
        """
        _sentinel_join(self, "engine_poison_refit_weights")
        from thundersync_rl.refit_m2n import poison_refit_tensors

        return poison_refit_tensors(self.model)

    def _log_mem(self, where: str) -> None:
        if env_flag("THUNDERSYNC_MEM_LOG"):
            print(
                f"[thundersync mem] rank={self.rank} {where}: "
                f"alloc={torch.cuda.memory_allocated() / 2**30:.1f}G "
                f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}G",
                flush=True,
            )
            torch.cuda.reset_peak_memory_stats()

    @_lg_feature("train")
    def stream_accumulate(
        self,
        items: list[dict[str, Any]],
        closes: list[tuple[Any, dict[float, float]]],
    ) -> dict[str, Any]:
        """Backprop streamed chunks, then fold any groups that closed.

        Each item is ``{"data": BatchedDataDict, "group": gid | None,
        "reward": float | None, "fused_key": str | None}``; ``group=None`` items
        carry final advantages. An item with a ``fused_key`` is one row whose
        model forward is the retained verification forward of that key
        (fused_train.py); everything else about its training is unchanged.

        Items with the same gradient destination (all final items; the items
        of one (group, reward) bucket) run as one pipeline schedule call with
        all their micro-batches (:func:`group_pipeline_calls`), so under
        pipeline parallelism a dispatch pays one fill/drain bubble per
        destination instead of one per chunk.
        """
        _call_log(self, "stream_accumulate")
        t0 = time.perf_counter()
        fused_trained = fused_tokens = fused_fallback = 0
        vit_results: list[dict[str, Any]] = []
        state = self._train_step_state
        from thundersync_rl.streaming import row_audit_items

        got = row_audit_items(items, env_flag("THUNDERSYNC_ROW_AUDIT"))
        acc = getattr(self, "_rows_received", [0, 0, 0])
        self._rows_received = [acc[0] + got[0], acc[1] + got[1], (acc[2] + got[2]) & 0xFFFFFFFF]
        if env_flag("THUNDERSYNC_ROW_AUDIT"):
            print(f"[row audit] rank={self.rank} rows={got[0]} tokens={got[1]} digest={got[2]}", flush=True)
        # Count-only chunks (planner: all-zero advantage under expert
        # parallelism, StreamPlanner.count_only): normalization counts only, no
        # forward. Every rank skips exactly the chunks the planner marked, so
        # forward counts stay equal across the EP all-to-all.
        skipped = [it for it in items if it.get("count_only")]
        if skipped:
            items = [it for it in items if not it.get("count_only")]
            for it in skipped:
                count_only_chunk(state, it)
        calls = group_pipeline_calls(
            items, merge=getattr(self, "_stream_merge_calls", True)
        )
        prof = os.environ.get("THUNDERSYNC_LEARNER_PROF")
        tprof = os.environ.get("THUNDERSYNC_LEARNER_TORCHPROF")
        for call in calls:
            call_items = [items[i] for i in call]
            first = call_items[0]
            bucket = (
                None
                if first["group"] is None
                else StreamBucket(group=first["group"], reward=first["reward"])
            )
            # Chunks smaller than a micro-batch run as one micro-batch of exactly
            # their rows (no dummy rows); larger chunks arrive padded to a
            # multiple of the configured size and use it. Keeping the configured
            # size bounds memory and matches the synchronous trainer's numerics
            # (Megatron fp32 gradients shift ~1e-4 with micro-batch size).
            step_mbs = state["mbs"]
            mbs_per_item = [min(it["data"].size, step_mbs) for it in call_items]
            if prof:
                torch.cuda.synchronize()
                t_call = time.perf_counter()
            for it in call_items:
                adv0 = it["data"].get("advantages")
                if adv0 is None or bool((adv0 != 0).any()):
                    self._n_items = getattr(self, "_n_items", 0) + 1
            fused_key = first.get("fused_key")
            assert fused_key is None or len(call_items) == 1
            graph = None if fused_key is None else self._fused_graphs.pop(fused_key)
            graph_tokens = 0 if graph is None else graph.num_tokens
            # A key without a graph (over the budget, or dropped) trains normally.
            outcome = FusedOutcome(
                used=False, fallback=fused_key is not None and graph is None
            )
            substitute = (
                nullcontext()
                if graph is None
                else substitute_model_forward(
                    megatron_train,
                    make_fused_model_forward(
                        graph,
                        megatron_train.model_forward,
                        outcome,
                        check=env_flag("THUNDERSYNC_FUSED_CHECK"),
                    ),
                )
            )
            del graph  # only the substitute forward holds it now
            if first.get("vit"):
                fn = self._vit_spread_call if first.get("vit_spread") else self._vit_train_call
                vit_results.extend(fn(call_items, mbs_per_item, tprof))
                continue
            try:
                with substitute:
                    if self._vit_check_due(bucket, fused_key):
                        vit_check.run_check(
                            self, call_items, mbs_per_item, bucket, tprof, self._vit_calls
                        )
                    else:
                        # verify_yield: plain training calls only (a fused call's
                        # substitute forward would serve the verification too).
                        self._yield_ok = (
                            fused_key is None
                            and getattr(self, "_yield_mailbox", None) is not None
                            and getattr(self, "_yield_domain", None) is not None
                            and getattr(self, "_ordered_running", None) is not None
                        )
                        try:
                            self._stream_train_call(call_items, mbs_per_item, bucket, tprof)
                        finally:
                            self._yield_ok = False
            finally:
                # Frees the graph's activations also when no forward ran (a
                # zero-advantage row skipped by the worker).
                del substitute
            fused_trained += outcome.used
            fused_tokens += graph_tokens if outcome.used else 0
            fused_fallback += outcome.fallback
            if prof:
                torch.cuda.synchronize()
                rows = sum(it["data"].size for it in call_items)
                nz = 0
                for it in call_items:
                    adv = it["data"].get("advantages")
                    nz += int((adv.abs().sum(-1) > 0).sum()) if adv is not None else 0
                shapes = [tuple(it["data"]["input_ids"].shape) for it in call_items]
                print(
                    f"[learner prof] rank={self.rank} chunks={len(call_items)} "
                    f"rows={rows} nonzero={nz} shapes={shapes} "
                    f"secs={time.perf_counter() - t_call:.4f}",
                    flush=True,
                )
        if closes:
            self.close_stream_groups(closes)
        torch.cuda.synchronize()
        self._log_mem("stream_accumulate")
        self._stream_busy_s += time.perf_counter() - t0
        acc = self._train_step_state["grad_stream"]
        return {
            "num_items": len(items),
            "num_skipped_items": len(skipped),
            "num_calls": len(calls),
            "open_buffers": acc.num_open_buffers(),
            "fused_trained": fused_trained,
            "fused_tokens": fused_tokens,
            "fused_fallback": fused_fallback,
            "vit": vit_results,
        }

    def _vit_train_call(
        self,
        call_items: list[dict[str, Any]],
        mbs_per_item: list[int],
        tprof: str | None,
    ) -> list[dict[str, Any]]:
        """Verify-in-train (``vit.py``): train each item (one group's rows, one
        micro-batch) and decide its rows' block verification from that forward.

        A failed micro-batch's loss is zeroed inside the loss (its backward still
        runs: every rank keeps the same schedule) and its normalization counts and
        metrics are taken back, so it leaves the step exactly as if never trained.
        Zero-advantage rows are trained too: their forward is the verification.
        Returns [{"group", "pass", "rows"}] per item in call order.
        """
        from megatron.core import parallel_state as ps

        if (ps.get_pipeline_model_parallel_world_size() or 1) > 1:
            raise RuntimeError("verify_in_train: pipeline parallelism is not built")
        state = self._train_step_state
        if any(int(it["data"].size) != int(m) for it, m in zip(call_items, mbs_per_item)):
            raise RuntimeError("verify_in_train: every item must be exactly one micro-batch")
        dev = torch.device("cuda", torch.cuda.current_device())
        counts = []
        for it in call_items:
            d = it["data"]
            d["vit_eta"] = torch.stack(
                [
                    vit.eta_last(int(d["vit_seed"][b]), int(d["vit_len"][b]), dev)
                    if float(d["sample_mask"][b]) != 0.0
                    else torch.ones((), dtype=torch.float64, device=dev)
                    for b in range(d.size)
                ]
            ).cpu()
            counts.append(vit.item_counts(d))
        n_metrics = len(state["all_mb_metrics"])
        n_losses = len(state["mb_losses"])
        skip = state["skip_zero_advantage_rows"]
        state["skip_zero_advantage_rows"] = False
        results: list[dict[str, Any]] = []
        try:
            with vit.decision_hook(results):
                self._stream_train_call(call_items, mbs_per_item, None, tprof)
        finally:
            state["skip_zero_advantage_rows"] = skip
        if len(results) != len(call_items):
            raise RuntimeError(
                f"verify_in_train: {len(results)} decisions for {len(call_items)} micro-batches"
            )
        failed = [k for k, r in enumerate(results) if not r["pass"]]
        self._count_uncounted(call_items[k] for k in failed)
        for k in failed:
            seqs, toks = counts[k]
            for name, v in (("local_valid_seqs", seqs), ("local_valid_toks", toks)):
                state[name] = state[name] - v.to(state[name].device)
            for name, v in (("trained_valid_seqs", seqs), ("trained_valid_toks", toks)):
                state[name] = state[name] - v.to(state[name].device)
        new = state["all_mb_metrics"][n_metrics:]
        if failed and len(new) == len(call_items):
            kept = [m for k, m in enumerate(new) if k not in set(failed)]
            del state["all_mb_metrics"][n_metrics:]
            state["all_mb_metrics"].extend(kept)
            del state["mb_losses"][n_losses:]
            state["mb_losses"].extend(m["loss"] for m in kept if "loss" in m)
        return [
            {"group": it.get("vit_group", -1), **r} for it, r in zip(call_items, results)
        ]

    def _vit_spread_call(
        self,
        call_items: list[dict[str, Any]],
        mbs_per_item: list[int],
        tprof: str | None,
    ) -> list[dict[str, Any]]:
        """Verify-in-train for ONE group spread over ranks and micro-batches (``vit.py``).

        Every rank holds some of the group's rows as micro-batches with only that
        group's rows (fillers: fully masked). Phase 1 runs all of them forward
        (training mode, autograd on, no backward) through the training path and
        keeps each micro-batch's decoder output with its graph and the routes it
        used; the loss hook reads each row's decision. The ranks then agree (MIN
        all-reduce over the data-parallel group). Pass: the regular training call
        runs over the same items with each forward replaced by its retained
        decoder output (LM head, loss and backward run as always; activation
        recompute replays the retained routes). Fail: nothing is backpropagated
        and nothing is counted; the forwards served as the verification.
        Memory per rank: the retained graphs, i.e. under full recompute the
        layer inputs of this rank's rows of the group.
        """
        from megatron.core import parallel_state as ps

        from nemo_rl.models.megatron.router_replay import _router_replay_instances_for_model
        from nemo_rl.models.policy.workers.megatron_policy_worker import (
            _should_use_router_replay,
        )

        if (ps.get_pipeline_model_parallel_world_size() or 1) > 1:
            raise RuntimeError("verify_in_train: pipeline parallelism is not built")
        state = self._train_step_state
        dev = torch.device("cuda", torch.cuda.current_device())
        for it in call_items:
            d = it["data"]
            d["vit_eta"] = torch.stack(
                [
                    vit.eta_last(int(d["vit_seed"][b]), int(d["vit_len"][b]), dev)
                    if float(d["sample_mask"][b]) != 0.0
                    else torch.ones((), dtype=torch.float64, device=dev)
                    for b in range(d.size)
                ]
            ).cpu()
        # Phase 1 on copies: ``_with_routes`` consumes the route handles, which the
        # training call needs again.
        datas = [self._with_routes(it["data"]) for it in vit_check.copy_items(call_items)]
        datas = [d["data"] if isinstance(d, dict) and "data" in d else d for d in datas]
        decoder = _unwrap(self.model).decoder
        instances = [inst for inst, _ in _router_replay_instances_for_model(self.model)]
        captured: list[Any] = []
        retained: list[vit.RetainedSlot] = []
        results: list[dict[str, Any]] = []
        original = megatron_train.model_forward

        def phase1_forward(**kwargs):
            out = original(**kwargs)
            routes = [
                inst.replay_backward_list[-1] if inst.replay_backward_list else None
                for inst in instances
            ]
            ids = kwargs["input_ids_cp_sharded"].detach().cpu()
            retained.append(
                vit.RetainedSlot(ids=ids, output=captured[-1], routes=routes, num_tokens=ids.numel())
            )
            return out

        t0 = time.perf_counter()
        mem0 = torch.cuda.memory_allocated()
        hook = decoder.register_forward_hook(lambda _m, _a, out: captured.append(out))
        try:
            pieces = []
            for data, mbs in zip(datas, mbs_per_item):
                attach_media_token_validity_mask(data, self.media_placeholder_token_id)
                iterator, n_mb, mb_size, _, padded = get_microbatch_iterator(
                    data,
                    self.cfg,
                    mbs,
                    straggler_timer=self.mcore_state.straggler_timer,
                    delegate_pack_to_model=self.delegate_pack_to_model,
                    delegate_mtp_loss_mask_to_model=self.delegate_mtp_loss_mask_to_model,
                    model_slices_context_parallel_inputs=self.model_slices_context_parallel_inputs,
                    mtp_enabled=self.mtp_enabled,
                )
                rr = _should_use_router_replay(
                    enabled=self._router_replay_enabled, data=data, stage="train", require=True
                )
                pieces.append((iterator, int(n_mb), int(mb_size), int(padded), bool(rr)))
            if any(p[1] != 1 for p in pieces):
                raise RuntimeError("verify_in_train: every item must be exactly one micro-batch")
            if len({p[4] for p in pieces}) > 1:
                raise RuntimeError("verify_in_train: items disagree on router replay")
            n = len(pieces)
            post = megatron_train.LossPostProcessor(
                loss_fn=state["loss_fn"],
                cfg=self.cfg,
                num_microbatches=n,
                sampling_params=self.sampling_params,
                draft_model=self.draft_model,
                defer_draft_normalization=True,
                teacher_output_layer_weight_by_index=self._opd_full_teacher_lm_heads,
            )
            one = torch.tensor(1.0, device="cuda")
            with (
                torch.enable_grad(),
                vit.decision_hook(results),
                substitute_model_forward(megatron_train, phase1_forward),
            ):
                megatron_train.megatron_forward_backward(
                    model=self.model,
                    data_iterator=itertools.chain.from_iterable(p[0] for p in pieces),
                    num_microbatches=n,
                    seq_length=max(p[3] for p in pieces),
                    mbs=max(p[2] for p in pieces),
                    post_processing_fn=post,
                    forward_only=True,
                    defer_fp32_logits=self.defer_fp32_logits,
                    global_valid_seqs=one,
                    global_valid_toks=one,
                    sampling_params=self.sampling_params,
                    straggler_timer=self.mcore_state.straggler_timer,
                    model_slices_context_parallel_inputs=self.model_slices_context_parallel_inputs,
                    draft_model=self.draft_model,
                    enable_hidden_capture=False,
                    use_fused_linear_logprobs=self.cfg["megatron_cfg"].get(
                        "use_fused_linear_logprobs", False
                    ),
                    use_router_replay=pieces[0][4],
                    router_replay_train=True,
                )
        finally:
            hook.remove()
        del captured
        torch.cuda.synchronize()
        retained_bytes = torch.cuda.memory_allocated() - mem0
        n_tok = sum(r.num_tokens for r in retained)
        if len(results) != len(call_items) or len(retained) != len(call_items):
            raise RuntimeError(
                f"verify_in_train: {len(results)} decisions / {len(retained)} forwards "
                f"for {len(call_items)} micro-batches"
            )
        flag = torch.tensor([1 if all(r["pass"] for r in results) else 0], dtype=torch.int32, device="cuda")
        torch.distributed.all_reduce(
            flag, op=torch.distributed.ReduceOp.MIN, group=ps.get_data_parallel_group()
        )
        keep = bool(flag.item())
        t1 = time.perf_counter()
        if keep:
            slots = iter(retained)

            def phase2_forward(**kwargs):
                return vit.retained_forward(next(slots), original, decoder, instances, kwargs)

            skip = state["skip_zero_advantage_rows"]
            state["skip_zero_advantage_rows"] = False
            # Debug (THUNDERSYNC_VIT_SPREAD_CHECK=1): the retained-forward training
            # equals the regular training of the same micro-batches (bitwise under
            # deterministic_training); the regular result is kept.
            check = os.environ.get("THUNDERSYNC_VIT_SPREAD_CHECK") == "1"
            if check:
                rerun = vit_check.copy_items(call_items)
                snap = vit_check.snapshot(self)
            try:
                with substitute_model_forward(megatron_train, phase2_forward):
                    self._stream_train_call(call_items, mbs_per_item, None, tprof)
                if check:
                    g_vit = vit_check.grads_to_host(self)
                    vit_check.restore(self, snap)
                    self._stream_train_call(rerun, mbs_per_item, None, tprof)
                    diff = vit_check.compare(self, g_vit)
                    del g_vit, snap
                    print(
                        f"[vit spread check] rank={self.rank} micro_batches={len(call_items)} "
                        f"grad_diff={diff[0]}/{diff[1]:.3e}/{diff[2]:.3e}/{diff[3]}/{diff[4]}",
                        flush=True,
                    )
            finally:
                state["skip_zero_advantage_rows"] = skip
        del retained
        if not keep:
            self._count_uncounted(call_items)  # never trained: not in the loss denominator
        if env_flag("THUNDERSYNC_SPEC_PROF"):
            print(
                f"[vit spread] rank={self.rank} micro_batches={len(call_items)} kept={keep} "
                f"retained_bytes_per_token={retained_bytes / max(n_tok, 1):.0f} "
                f"phase1_s={t1 - t0:.3f} phase2_s={time.perf_counter() - t1:.3f}",
                flush=True,
            )
        return [
            {
                "group": it.get("vit_group", -1),
                "pass": keep,
                "rows": r["rows"],
                "retained_bytes_per_token": retained_bytes / max(n_tok, 1),
            }
            for it, r in zip(call_items, results)
        ]

    def _count_uncounted(self, items) -> None:
        """Received tokens of ``items`` that leave the loss denominator (``row_audit`` counting)."""
        from thundersync_rl.streaming import row_audit

        for it in items:
            rows, toks, _ = row_audit(it["data"])
            self._rows_uncounted = getattr(self, "_rows_uncounted", 0) + rows
            self._toks_uncounted = getattr(self, "_toks_uncounted", 0) + toks

    def _vit_check_due(self, bucket, fused_key) -> bool:
        """Debug (``THUNDERSYNC_VIT_CHECK=1``): check the first final-advantage calls of each step."""
        if not vit_check.enabled() or bucket is not None or fused_key is not None:
            return False
        key = id(self._train_step_state)
        if getattr(self, "_vit_step_key", None) != key:
            self._vit_step_key, self._vit_calls = key, 0
        if self._vit_calls >= vit_check.max_calls():
            return False
        self._vit_calls += 1
        return True

    def _stream_train_call(
        self,
        call_items: list[dict[str, Any]],
        mbs_per_item: list[int],
        bucket: StreamBucket | None,
        tprof: str | None,
    ) -> None:
        """``train_microbatches`` on one call's items (optionally under the profiler)."""
        datas = [it["data"] for it in call_items]
        if (
            tprof
            and getattr(self, "_n_items", 0) >= int(tprof)
            and not getattr(self, "_tprof_done", False)
        ):
            self._tprof_done = True
            from torch.profiler import ProfilerActivity, profile

            tp = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            # The profiler's trace metadata lists every process group's backend
            # config; groups created with backend options unset (Megatron's
            # gloo / sub-groups) make torch raise "value was None" there
            # (dr-x-lanes). Only the key_averages table is printed here, no
            # trace is exported, so skip that metadata.
            tp._get_distributed_info = lambda: None
            with tp:
                self.train_microbatches(
                    datas, stream_bucket=bucket, micro_batch_sizes=mbs_per_item
                )
                torch.cuda.synchronize()
            print(
                f"[learner torchprof] shapes="
                f"{[tuple(d['input_ids'].shape) for d in datas]}\n"
                + tp.key_averages().table(
                    sort_by="cpu_time_total",
                    row_limit=30,
                    max_name_column_width=60,
                ),
                flush=True,
            )
        else:
            self.train_microbatches(
                datas, stream_bucket=bucket, micro_batch_sizes=mbs_per_item
            )

    def stream_finish_step(self) -> dict[str, Any]:
        """Barrier: write back, normalize, DP-reduce, clip, step."""
        self._v0ni_trained = False
        _call_log(self, "stream_finish_step")
        peak = self._train_step_state["grad_stream"].peak_open_buffers
        # Retained verification graphs never outlive the weights they were
        # taken at; any left (their rollout trained the regular way) go now.
        unclaimed = self._fused_graphs.clear()
        # Learner prefix-state cache pools (eager learner): every entry is stale after this step's update; free the
        # pools for the optimizer step / checkpoint save and re-allocate them on the next verification.
        pc = getattr(self, "_prefix_cache", None)
        pool_dropped = bool(pc is not None and pc.drop_pools())
        # Memory of this step's dispatches (per rank). No runtime allocator management here: the caching allocator
        # (expandable segments, set once at init) keeps what the step reserved and the next step reuses it. The
        # per-step empty_cache that used to run here (above THUNDERSYNC_EMPTY_CACHE_FRAC of the device) is gone: it
        # never fired at the device-fitting layouts (alloc-ab ec97 == no flush at Nano, 0 retries) and at Super-L22
        # it only returned blocks the next step re-reserved (5-6 s per step, reserved 150 GiB both before and after).
        mem = {
            "learner_mem_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
            "learner_mem_reserved_gb": torch.cuda.memory_reserved() / 2**30,
            "learner_prefix_pool_dropped": float(pool_dropped),
            **allocator_stats(),
            "learner_moe_capacity_reruns": float(_moe_capacity_reruns()),
        }
        # Headroom of this step's peak on this device: total - (allocator reserved peak + memory outside PyTorch
        # now: CUDA contexts, NCCL / symmetric buffers, other processes). StreamingLearner.finish takes the min over
        # ranks (the peak keys the max), so the step line shows the tightest rank, not rank 0.
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        mem["learner_mem_headroom_gb"] = (
            total / 2**30 - mem["learner_mem_reserved_peak_gb"] - mem["learner_mem_nontorch_gb"]
        )
        mem["learner_max_mb_tokens"] = float(getattr(self, "_step_max_mb_tokens", 0))
        self._step_max_mb_tokens = 0
        torch.cuda.reset_peak_memory_stats()
        if env_flag("THUNDERSYNC_MEM_LOG"):
            print(f"[thundersync mem] rank={self.rank} step dispatches: {mem}", flush=True)
        st = self._train_step_state
        # Local loss-denominator counts before the DP reduction (one 2-scalar device->host copy).
        local_seqs, local_toks = torch.stack([st["local_valid_seqs"], st["local_valid_toks"]]).tolist()
        results = self.finish_train_step()
        results.update(mem)
        results["stream_peak_open_buffers"] = peak
        results["stream_learner_busy_s"] = self._stream_busy_s
        results["stream_fused_unclaimed"] = unclaimed
        from nemo_rl.models.megatron import learner_graphs as _lg

        line = _lg.summary_line()
        if line:
            # Step-end totals (the call-count-triggered line stops at 100 / 500-multiples;
            # run_rules reads each rank's last line).
            print(line, flush=True)
        from megatron.core import parallel_state as ps

        # One rank per data-parallel replica owns its micro-batch counts
        # (train_lp_check_merge; TP / PP / CP ranks repeat the same metrics).
        results["stream_dp_count_owner"] = bool(
            ps.get_tensor_model_parallel_rank() == 0
            and ps.get_pipeline_model_parallel_rank() == 0
            and ps.get_context_parallel_rank() == 0
        )
        results["stream_rows_received"] = list(getattr(self, "_rows_received", [0, 0, 0]))
        groups = self._layout_groups()
        if groups["ep"] is not None:
            # Cumulative EP lockstep forwards: equal within every EP and TP x CP group (run_checks).
            results["stream_lockstep"] = (
                int(getattr(self, "_lockstep_forwards", 0)),
                min(groups["ep"]),
                None if groups["tp_cp"] is None else min(groups["tp_cp"]),
            )
        # Run rule 2 (run_checks.step_counts_merge): loss denominators vs the tokens dealt; peers must agree.
        counts = results.pop("step_counts", None)
        if counts is not None:
            results["stream_step_counts"] = {
                "dp": int(ps.get_data_parallel_rank()),
                "local_seqs": float(local_seqs),
                "local_toks": float(local_toks),
                "num_microbatches": int(counts["num_microbatches"]),
                "global_seqs": float(counts["global_valid_seqs"]),
                "global_toks": float(counts["global_valid_toks"]),
                "trained_toks": float(counts["trained_valid_toks"]),
                "uncounted_toks": float(getattr(self, "_toks_uncounted", 0)),
                "uncounted_rows": float(getattr(self, "_rows_uncounted", 0)),
            }
        return results

    def stream_abort_step(self) -> None:
        """Abort the open streamed step and drop its retained verification graphs."""
        self._fused_graphs = FusedGraphStore()
        self.abort_train_step()

    @torch.no_grad()
    def grad_digest(self) -> tuple[float, int]:
        """Debug: (float64 sum, crc32 of the bytes) of this rank's gradient buffers now.

        Bitwise fingerprint of the accumulated gradient before the step's
        reductions (``THUNDERSYNC_DEBUG_ORDER_REPLAY``).
        """
        total, crc = 0.0, 0
        for b in self.model.buffers + self.model.expert_parallel_buffers:
            g = b.grad_data.detach()
            total += float(g.double().sum())
            crc = zlib.crc32(g.contiguous().view(torch.uint8).cpu().numpy().tobytes(), crc)
        return total, crc

    def grad_param_digests(self) -> dict[str, int]:
        """Debug: crc32 of each parameter's accumulated gradient (``main_grad``) now.

        Names the parameters whose gradient bits differ between two
        ``THUNDERSYNC_DEBUG_ORDER_REPLAY=params`` replays.
        """
        out = {}
        for name, p in self.model.named_parameters():
            g = getattr(p, "main_grad", None)
            if g is not None:
                raw = g.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                out[name] = zlib.crc32(raw)
        return out

    def grad_param_snapshot(self, tag: str) -> None:
        """Debug: keep a CPU copy of every parameter's accumulated gradient under ``tag``."""
        snaps = self.__dict__.setdefault("_grad_snapshots", {})
        snaps[tag] = {
            name: p.main_grad.detach().cpu().clone()
            for name, p in self.model.named_parameters()
            if getattr(p, "main_grad", None) is not None
        }

    def grad_param_compare(self, a: str, b: str, drop: bool = True) -> list[tuple]:
        """Debug: (name, differing elements, max |diff|, signed zeros only) per differing parameter."""
        snaps = self.__dict__.get("_grad_snapshots", {})
        out = []
        for name, ga in snaps[a].items():
            gb = snaps[b][name]
            ne = ga.view(torch.int32) != gb.view(torch.int32)
            n = int(ne.sum())
            if n:
                d = (ga.float() - gb.float()).abs()
                out.append((name, n, ga.numel(), float(d.max()), bool((d[ne] == 0).all())))
        if drop:
            snaps.clear()
        return out

    def get_flat_grads(self) -> torch.Tensor:
        """Return this rank's gradient buffers on CPU (for equivalence tests).

        Valid after a step with DP=1, where the buffers hold the full
        normalized gradient until the next step zeroes them.
        """
        buffers = self.model.buffers + self.model.expert_parallel_buffers
        return torch.cat([b.grad_data.detach().float().cpu() for b in buffers])

    def get_flat_params(self) -> dict[str, torch.Tensor]:
        """Return this rank's parameters on CPU (for equivalence tests)."""
        return {
            name: p.detach().float().cpu().clone()
            for name, p in self.model.named_parameters()
        }

    # ---- Exact cross-iteration speculative rollouts (keyed sampling) ----

    def install_lookup_references(self) -> None:
        """Generation workers: forward per-request ``lookup_references`` to the engine.

        Requests whose data carries ``lookup_references`` (a list of token
        sequences per row) get them on their SamplingParams, for the engine's
        exact lookup speculative decoding (``speculative_method="lookup"``).
        """
        if getattr(self, "_lookup_refs_installed", False):
            return
        inner = self._prepare_data_for_generation

        def prepare(data, greedy=False):
            prompts, mm, sps = inner(data, greedy)
            if "lookup_references" in data:
                for i, sp in enumerate(sps):
                    sp.lookup_references = [
                        list(r) for r in data["lookup_references"][i]
                    ]
            return prompts, mm, sps

        self._prepare_data_for_generation = prepare
        self._lookup_refs_installed = True

    def install_wide_sampler(
        self, salt: int, key64: bool = True, fresh_salt: int | None = None
    ) -> dict[str, Any] | None:
        """Generation workers: the engine's sampler -> the 53-bit keyed Gumbel race.

        Replaces the engine's FlashInfer / torch sampler (fp32 uniforms: no token below
        ~6e-8 x p_top is ever drawn, redteam_1016 §6) by ``wide_sample_logits``: each
        row's noise is keyed by (``salt``, engine request id, index of the sampled
        token), fp64 race over the full (padded) vocabulary at the request's
        temperature, so the law is softmax_fp64(fp32 logits / T) to ~1e-16 relative
        and a row's draw does not depend on the batch. Device tensors only, no host
        sync. Requests with top-k / top-p filters, and the speculative-decoding
        path (``token_to_request_index``), keep the engine's own kernel (counted in
        ``wide_sampler_stats``). Install before ``install_draft_hidden_log`` (it
        wraps the kernel installed at that time). Returns the stats dict, None
        without an engine. ``key64`` (``thundersync.wide_sampler_key64``, default True since
        ts-lawfix): two 32-bit noise-key words per (request, position) (keyed_sampling).

        ``fresh_salt`` (resume reproducibility): a request whose sampling params carry ``noise_seed`` >= 0 (routed
        from the generation data's ``noise_seed`` field; the ThunderSync loop sets it on every rollout row it decodes
        fresh: hash(grpo.seed, "fresh", step, row, turn)) is keyed by (fresh_salt, noise_seed | 2**62, position)
        instead of (salt, engine request id, position). Engine request ids are a per-process counter in arrival
        order, so they differ between runs and across a resume; the stable key makes a fresh row's tokens a function
        of (seed, step, row, the engine's logits) only. ``fresh_salt`` does not depend on the lane or the resume
        segment; the 2**62 bit keeps the two key spaces apart. None: request ids only (as before).
        """
        from thundersync_rl.keyed_sampling import wide_sample_logits

        engine = getattr(self, "dynamic_inference_engine", None)
        if engine is None:
            return None
        sampling = engine.controller._sampling
        if getattr(sampling, "_wide_installed", False):
            return dict(sampling._wide_stats)
        if getattr(sampling, "_keyed_installed", False):
            raise RuntimeError("install_wide_sampler: the keyed sampler is installed")
        # Exact top-p (_install_nucleus_sampler, at engine build): the nucleus wrapper
        # masks the sampled rows in place and must stay outside the race, else the race
        # would sample the untruncated row (ts-nano-scale composition fix): the race
        # becomes the nucleus wrapper's inner kernel.
        under_nucleus = bool(getattr(sampling, "_nucleus_installed", False))
        orig_kernel = sampling._nucleus_inner if under_nucleus else sampling.sample_kernel
        stats = {"calls": 0, "fallback_calls": 0, "s_calls": 0, "stable_rows": 0, "gpu_view_pos_rows": 0}
        fresh_keys = _RequestKeys(engine) if fresh_salt is not None else None
        lens = self._request_lengths()
        from thundersync_rl.key_check import KeyContinuity

        # Always on: every (request, position) noise key is drawn once, positions contiguous per request (raises).
        key_check = KeyContinuity("wide sampler")
        # Multi-token tail decode (tail_mtd) draws its extra rows with this salt and advances these checks at commit.
        self._wide_key_check, self._wide_salt, self._wide_key64 = key_check, int(salt), bool(key64)
        if fresh_salt is not None:
            self._route_noise_seed()
        # Debug (THUNDERSYNC_DEBUG_LAW_BUG=temp:<f>): sample at temperature f x T while every
        # logged log-prob stays at T -- a deliberately WRONG engine law, the negative control
        # of the law check (the value checks all pass under it).
        bug = os.environ.get("THUNDERSYNC_DEBUG_LAW_BUG", "")
        temp_bug = float(bug.split(":")[1]) if bug.startswith("temp:") else 1.0
        # Keyed sampler S (kernel agent, keyed_sampling_s): bitwise the wide sampler's draws, faster; the wide kernel
        # (wide_sample_logits) stays for rows keyed per row (row_salts), key32, and as S's test reference.
        from thundersync_rl import keyed_sampling_s as s_mod

        s_ws = [None]

        def wide_kernel(
            logits,
            n,
            context,
            *,
            gather_indices=None,
            token_to_request_index=None,
            output=None,
            no_top_k=True,
            no_top_p=True,
            **kw,
        ):
            if not (no_top_k and no_top_p):
                # One request's top-k / top-p would switch the WHOLE batch to the unkeyed kernel: batch composition
                # must never change another row's sampling (BUGS.md, harness 11-06). Keyed mode refuses it.
                raise KeyedBatchError(
                    "wide sampler: a top-k / top-p request reached the keyed sampler; it would move every row of "
                    "the batch to the unkeyed kernel (keyed mode accepts top_k None / 0 and top_p 1 only)")
            if token_to_request_index is not None:
                stats["fallback_calls"] += 1
                return orig_kernel(
                    logits,
                    n,
                    context,
                    gather_indices=gather_indices,
                    token_to_request_index=token_to_request_index,
                    output=output,
                    no_top_k=no_top_k,
                    no_top_p=no_top_p,
                    **kw,
                )
            stats["calls"] += 1
            gv = context.gpu_view
            lo = context.paused_request_count
            # request_ids is the context's pinned host array: a fresh pinned copy, then
            # an async H2D (the context rewrites its array next step).
            host_rids = context.request_ids[lo : lo + n].to(torch.int64)
            hpos = _request_positions(engine, host_rids.numpy(), lens)
            cpos = _key_check_positions(context, host_rids.numpy(), hpos)
            key_check.observe(host_rids.numpy(), cpos)
            stats.update(key_check.stats())
            # The per-row host arrays (key ids, key positions, per-row salts) reach the device in ONE pinned
            # [k, n] int64 block and one async H2D (was one fresh pinned buffer + H2D each): same values.
            h_rids = host_rids.numpy()
            if self.__dict__.get("_nl") is not None:
                # nolanes (nolanes_engine.py): the noise-key registry (E4: each (key, position) drawn once).
                h_rids = self._nl_key_ids(engine, h_rids)
                self._nl_observe_keys(h_rids, cpos)
            known = hpos >= 0
            if fresh_salt is not None:
                keys = fresh_keys.lookup(h_rids)
                stable = keys >= 0
            else:
                stable = None
            fresh = stable is not None and bool(stable.any())
            parts = [np.where(stable, keys | _FRESH_KEY_BIT, h_rids) if fresh else h_rids]
            any_known = bool(known.any())
            if any_known:
                parts.append(hpos)
            if fresh:
                parts.append(np.where(stable, int(fresh_salt), int(salt)).astype(np.int64))
                stats["stable_rows"] += int(stable.sum())
            dev = torch.from_numpy(np.stack(parts)).pin_memory().to(logits.device, non_blocking=True)
            rids = dev[0]
            row_salts = dev[-1] if fresh else None
            # Index of the token being sampled = the request's length after this step, from the request itself
            # (_request_positions: the gpu_view lengths are not compacted with the rows under async scheduling);
            # the gpu_view position only for a row whose request is unknown.
            if any_known and bool(known.all()):
                pos = dev[1]
            else:
                stats["gpu_view_pos_rows"] += int((~known).sum())
                pos = gv.request_kv_length_offsets[:n].long() + gv.request_query_lengths[:n].long()
                if any_known:
                    pos = torch.where(dev[1] >= 0, dev[1], pos)
            temp = gv.temperature[:n].clamp(min=_min_sampling_temperature())
            if temp_bug != 1.0:
                temp = temp * temp_bug
            rows = logits[gather_indices[:n]] if gather_indices is not None else logits[:n]
            if row_salts is None and key64:
                # Keyed sampler S (blocked two-kernel race), bitwise the same draws as
                # wide_sample_logits(key64=True) (test_keyed_sampling_s.py).
                stats["s_calls"] += 1  # non-vacuity (R35 3(h); nl_summary requires > 0)
                need = s_mod.workspace_bytes(n, rows.shape[1])
                if s_ws[0] is None or s_ws[0].numel() < need:
                    s_ws[0] = torch.empty(max(need, 1), dtype=torch.uint8, device=rows.device)
                out = torch.empty(n, dtype=torch.int64, device=rows.device)
                s_mod.wide_sample(rows if rows.stride(-1) == 1 else rows.contiguous(), rids, pos, temp, salt,
                                  out=out, workspace=s_ws[0])
            else:
                out = wide_sample_logits(rows, rids, pos, temp, salt, key64=key64, row_salts=row_salts)
            if output is None:
                return out
            output.copy_(out)
            return output

        if under_nucleus:
            sampling._nucleus_inner = wide_kernel
        else:
            sampling.sample_kernel = wide_kernel
        sampling._wide_installed = True
        sampling._wide_stats = stats
        stats["key64"] = bool(key64)
        print(f"[wide sampler] rank={getattr(self, 'rank', '?')} 53-bit keyed Gumbel race installed (salt={salt}) "
              f"key={'64' if key64 else '32'}-bit"
              + ("" if fresh_salt is None else f" fresh rows keyed by (seed, step, row) (fresh_salt={fresh_salt})"),
              flush=True)
        return dict(stats)

    def _request_lengths(self) -> "_RequestLengths":
        """This worker's request-length cache, shared by the sampler key positions and the hidden-state log."""
        lens = getattr(self, "_req_lens", None)
        if lens is None:
            lens = self._req_lens = _RequestLengths()
        return lens

    def _route_noise_seed(self) -> None:
        """Generation data field ``noise_seed`` -> SamplingParams.noise_seed (it travels to the engine with the
        serialized sampling params). Idempotent; the keyed sampler routes it itself."""
        if getattr(self, "_noise_seed_routed", False) or getattr(
            getattr(getattr(getattr(self, "dynamic_inference_engine", None), "controller", None), "_sampling", None),
            "_keyed_installed", False,
        ):
            return
        orig_prepare = self._prepare_data_for_generation

        def prepare(data, greedy=False):
            prompts, mm, sps = orig_prepare(data, greedy)
            if data is not None and "noise_seed" in data:
                for i, sp in enumerate(sps):
                    sp.noise_seed = int(data["noise_seed"][i])
            return prompts, mm, sps

        self._prepare_data_for_generation = prepare
        self._noise_seed_routed = True

    def install_exact_int_sampler(self, salt: int) -> dict[str, Any] | None:
        """Generation workers: the engine's sampler and logged log-probs -> the exact integer law.

        ``engine_sampler: exact_int`` (nemo_rl/models/megatron/exact_int.py).
        The policy is pi(v) = w_v / W, w = floor(exp(float64(l)) 2^52), l = the fp32
        log_softmax row (batch-invariant). (1) Sampling: ``exact_int.sample_logprob_rows``
        draws EXACTLY from w / W (keyed by (salt, engine request id, index of the sampled
        token): rejection to a uniform integer r < W, integer prefix sums), one Triton
        program per row, no host sync. (2) Logged log-probs: the context's
        ``calculate_log_probs_tensors`` returns fp32(log(w_x) - log(W)) of every selected
        token (prompt and generated), the value the learner trains on. Top-k / top-p
        requests and in-engine speculative decoding are refused (validated off).
        Install before ``install_draft_hidden_log`` (it wraps this kernel).
        """
        from nemo_rl.models.megatron import exact_int as xi

        engine = getattr(self, "dynamic_inference_engine", None)
        ctx = getattr(self, "inference_context", None)
        if engine is None or ctx is None:
            return None
        sampling = engine.controller._sampling
        if getattr(sampling, "_exact_int_installed", False):
            return self.exact_int_sampler_stats()
        if getattr(sampling, "_keyed_installed", False) or getattr(sampling, "_wide_installed", False):
            raise RuntimeError("install_exact_int_sampler: another sampler is installed")
        stats: dict[str, Any] = {"calls": 0, "rows": 0, "logprob_calls": 0}
        bad = torch.zeros(1, dtype=torch.int64, device="cuda")
        from thundersync_rl.key_check import KeyContinuity

        # Always on: every (request, position) noise key is drawn once, positions contiguous per request (raises).
        key_check = KeyContinuity("exact_int sampler")
        lens = self._request_lengths()
        # Debug negative control (THUNDERSYNC_DEBUG_LAW_BUG=temp:<f>): sample the integer law
        # of log_softmax(z / f) while the logged law stays the T = 1 row's.
        bug = os.environ.get("THUNDERSYNC_DEBUG_LAW_BUG", "")
        temp_bug = float(bug.split(":")[1]) if bug.startswith("temp:") else 1.0

        def exact_kernel(
            logits, n, context, *, gather_indices=None, token_to_request_index=None,
            output=None, no_top_k=True, no_top_p=True, **kw,
        ):
            if token_to_request_index is not None or not (no_top_k and no_top_p):
                raise RuntimeError(
                    "engine_sampler=exact_int: top-k / top-p requests or in-engine speculative "
                    "decoding reached the exact sampler (validated off)"
                )
            stats["calls"] += 1
            stats["rows"] += int(n)
            gv = context.gpu_view
            lo = context.paused_request_count
            host_rids = context.request_ids[lo : lo + n].to(torch.int64)
            rids = host_rids.pin_memory().to(logits.device, non_blocking=True)
            # Request-derived positions (_request_positions; the gpu_view lengths are not compacted with the rows
            # under async scheduling).
            hpos = _request_positions(engine, host_rids.numpy(), lens)
            key_check.observe(host_rids.numpy(), _key_check_positions(context, host_rids.numpy(), hpos))
            stats.update(key_check.stats())
            pos = gv.request_kv_length_offsets[:n].long() + gv.request_query_lengths[:n].long()
            if (hpos >= 0).any():
                hp = torch.from_numpy(hpos).pin_memory().to(logits.device, non_blocking=True)
                pos = torch.where(hp >= 0, hp, pos)
            rows = logits[gather_indices[:n]] if gather_indices is not None else logits[:n]
            z = rows.float()
            if temp_bug != 1.0:
                z = z / temp_bug
            tok, W, _ = xi.sample_logprob_rows(torch.log_softmax(z, -1), rids, pos, salt)
            bad.add_((W <= 0).sum())
            if output is None:
                return tok
            output.copy_(tok)
            return output

        orig_lp = ctx.calculate_log_probs_tensors

        def calculate_log_probs_tensors(logits, new_tokens, *args, **kwargs):
            selected, log_probs = orig_lp(logits, new_tokens, *args, **kwargs)
            stats["logprob_calls"] += 1
            n = selected.shape[0]
            lam = xi.int_logprob_selected(log_probs[:n], selected)
            return lam.to(selected.dtype), log_probs

        sampling.sample_kernel = exact_kernel
        ctx.calculate_log_probs_tensors = calculate_log_probs_tensors
        sampling._exact_int_installed = True
        sampling._exact_int_stats = stats
        sampling._exact_int_bad = bad
        print(
            f"[exact_int sampler] rank={getattr(self, 'rank', '?')} integer law (2^52 weights) "
            f"installed: exact keyed sampling + log(w)-log(W) logged log-probs (salt={salt})",
            flush=True,
        )
        return self.exact_int_sampler_stats()

    def exact_int_sampler_stats(self) -> dict[str, Any] | None:
        engine = getattr(self, "dynamic_inference_engine", None)
        sampling = getattr(getattr(engine, "controller", None), "_sampling", None)
        st = getattr(sampling, "_exact_int_stats", None)
        if st is None:
            return None
        out = dict(st)
        out["invalid_rows"] = int(sampling._exact_int_bad.item())
        if out["invalid_rows"]:
            raise RuntimeError(f"exact_int sampler: {out['invalid_rows']} rows with no integer mass")
        return out

    def wide_sampler_stats(self) -> dict[str, Any] | None:
        engine = getattr(self, "dynamic_inference_engine", None)
        st = getattr(getattr(getattr(engine, "controller", None), "_sampling", None), "_wide_stats", None)
        return None if st is None else dict(st)

    def install_keyed_sampler(self, vocab_limit: int, head_k: int) -> int | None:
        """Generation workers: sample requests that carry ``noise_seed`` with keyed noise.

        The token at absolute position L of a request with seed s is
        ``keyed_sample(p, s, L)``, so a continuation request (prompt + accepted
        prefix) reproduces the uninterrupted decode exactly. Rows without a seed
        use the engine's own sampler. Also routes per-row ``noise_seed`` and
        ``max_new_tokens`` fields of the generation data into SamplingParams.
        """
        from thundersync_rl.keyed_sampling import keyed_sample_logits

        engine = getattr(self, "dynamic_inference_engine", None)
        if engine is None:
            return None
        eod = int(self.megatron_tokenizer.eod)
        sampling = engine.controller._sampling
        if getattr(sampling, "_keyed_installed", False):
            return eod
        orig_kernel = sampling.sample_kernel

        def keyed_kernel(
            logits,
            n,
            context,
            *,
            gather_indices=None,
            token_to_request_index=None,
            output=None,
            **kw,
        ):
            if token_to_request_index is not None:
                return orig_kernel(
                    logits,
                    n,
                    context,
                    gather_indices=gather_indices,
                    token_to_request_index=token_to_request_index,
                    output=output,
                    **kw,
                )
            lo, hi = context.paused_request_count, context.total_request_count
            # Key = (seed, index of the token being sampled). The context's
            # sequence length counts the current step's input token during
            # decode, so take the index from the request itself.
            seeds, positions, jitter = [], [], []
            for rid in context.request_ids[lo:hi].tolist()[:n]:
                req = engine.get_request(rid)
                sp = req.sampling_params
                seeds.append(int(getattr(sp, "noise_seed", -1)))
                positions.append(len(req.prompt_tokens) + len(req.generated_tokens))
                jitter.append(
                    (
                        float(getattr(sp, "draft_variant", 0)),
                        float(getattr(sp, "variant_eps", 0.0)),
                    )
                )
            if all(s < 0 for s in seeds):
                return orig_kernel(
                    logits,
                    n,
                    context,
                    gather_indices=gather_indices,
                    output=output,
                    **kw,
                )
            if all(s >= 0 for s in seeds):
                out = (
                    output
                    if output is not None
                    else torch.empty(n, device=logits.device, dtype=torch.int64)
                )
            else:
                out = orig_kernel(
                    logits,
                    n,
                    context,
                    gather_indices=gather_indices,
                    output=output,
                    **kw,
                )
            rows = (
                logits[gather_indices[:n]] if gather_indices is not None else logits[:n]
            )
            # Pinned host tensors + non-blocking copies: no per-step stream sync,
            # so the engine's async scheduling overlap is preserved.
            seed_c = torch.tensor(seeds, dtype=torch.long).pin_memory()
            pos_c = torch.tensor(positions, dtype=torch.long).pin_memory()
            jit_c = torch.tensor(jitter, dtype=torch.float64).pin_memory()
            if all(s >= 0 for s in seeds):
                out.copy_(keyed_sample_logits(rows, seed_c, pos_c, vocab_limit, jit_c))
            else:
                idx = [j for j, s in enumerate(seeds) if s >= 0]
                idx_t = (
                    torch.tensor(idx, dtype=torch.long)
                    .pin_memory()
                    .to(rows.device, non_blocking=True)
                )
                out[idx_t] = keyed_sample_logits(
                    rows[idx_t], seed_c[idx], pos_c[idx], vocab_limit, jit_c[idx]
                ).to(out.dtype)
            return out

        sampling.sample_kernel = keyed_kernel
        sampling._keyed_installed = True

        orig_prepare = self._prepare_data_for_generation

        def prepare(data, greedy=False):
            prompts, mm, sps = orig_prepare(data, greedy)
            if "noise_seed" in data:
                for i, sp in enumerate(sps):
                    sp.noise_seed = int(data["noise_seed"][i])
            if "max_new_tokens" in data:
                for i, sp in enumerate(sps):
                    sp.num_tokens_to_generate = int(data["max_new_tokens"][i])
            if "draft_variant" in data:
                for i, sp in enumerate(sps):
                    sp.draft_variant = int(data["draft_variant"][i])
                    sp.variant_eps = float(data["variant_eps"][i])
            return prompts, mm, sps

        self._prepare_data_for_generation = prepare
        return eod

    @torch.no_grad()
    @_model_in_eval
    @_lg_feature("verify")
    def verify_drafts(
        self,
        rows: list[torch.Tensor],
        prompt_lens: list[int],
        seeds: list[int],
        vocab_limit: int,
        head_k: int,
        batch_tokens: int,
        position_shift: int = 0,
    ) -> list[tuple[int, dict[str, Any]]]:
        """Learner: keep each draft's longest prefix the current policy would emit.

        ``rows[i]`` = prompt + draft tokens. Position j (predicting token j+1) emits
        ``keyed_sample(p_theta(. | row[:j+1]), seed, j+1)``; the draft is kept up to
        the first disagreement, where the emitted token replaces it. If the whole
        draft agrees, the next token after it is emitted too. Rows are split across
        DP ranks and batched right-padded (causal attention). Returns
        [(row_index, {"accepted", "next", "logprobs"})] for this rank's rows.
        """
        from thundersync_rl.keyed_sampling import keyed_sample_logits

        from megatron.core import parallel_state as ps

        # By data-parallel rank: the tensor-parallel ranks of a replica share rows.
        dp, dp_rank = ps.get_data_parallel_world_size(), ps.get_data_parallel_rank()
        mine = sorted(
            (i for i in range(len(rows)) if i % dp == dp_rank),
            key=lambda i: rows[i].numel(),
        )
        groups = []
        b = 0
        while b < len(mine):
            e = b + 1
            while e < len(mine) and rows[mine[e]].numel() * (e + 1 - b) <= batch_tokens:
                e += 1
            groups.append(mine[b:e])
            b = e
        from nemo_rl.models.megatron import learner_graphs

        # Expert parallelism: every forward is an all-to-all over the EP group.
        lockstep = self._expert_parallel_lockstep([[] for _ in groups])
        for _ in range(len(groups), len(groups) if lockstep is None else len(lockstep)):
            n = self.cfg["make_sequence_length_divisible_by"]
            learner_graphs.guarded_forward(
                lambda: self.model(
                    input_ids=torch.zeros((1, n), dtype=torch.long, device="cuda"),
                    position_ids=torch.arange(n, device="cuda")[None],
                    attention_mask=None,
                ),
                self.model,
            )
        results = []
        for group in groups:
            s = max(rows[i].numel() for i in group)
            # Pad like training (make_sequence_length_divisible_by): batch-invariant
            # kernels are invariant within a token-count shape class only.
            m = self.cfg["make_sequence_length_divisible_by"]
            s = -(-s // m) * m
            ids = torch.zeros((len(group), s), dtype=torch.long)
            for r, i in enumerate(group):
                ids[r, : rows[i].numel()] = rows[i]
            ids = ids.cuda()
            pos = torch.arange(s, device="cuda").expand(len(group), -1)
            logits = learner_graphs.guarded_forward(
                lambda: self.model(input_ids=ids, position_ids=pos, attention_mask=None), self.model
            )
            for r, i in enumerate(group):
                n, plen = rows[i].numel(), prompt_lens[i]
                lp = torch.log_softmax(
                    logits[r, plen - 1 : n, :vocab_limit].float(), -1
                )
                positions = torch.arange(plen, n + 1, device="cuda") + position_shift
                emitted = keyed_sample_logits(
                    lp, torch.full_like(positions, seeds[i]), positions
                )
                draft = ids[r, plen:n]
                agree = emitted[: draft.numel()] == draft
                n_acc = int(agree.long().cumprod(0).sum())
                kept = torch.cat([draft[:n_acc], emitted[n_acc : n_acc + 1]])
                # Sampling uses the vocab_limit slice (as the engine's keyed sampler
                # does); the reported log-probs are over the full vocabulary, as
                # training computes them (bitwise equal under batch invariance).
                lps = (
                    torch.log_softmax(logits[r, plen - 1 : plen + n_acc].float(), -1)
                    .gather(-1, kept.unsqueeze(-1))[:, 0]
                    .tolist()
                )
                results.append(
                    (
                        i,
                        {
                            "accepted": n_acc,
                            "next": int(emitted[n_acc]),
                            "logprobs": lps,
                        },
                    )
                )
            del logits
        return results

    # ---- Verification scorer: fp32 Megatron copy of the policy ----

    def _fp32_scorer(self):
        """Megatron copy of the policy with fp32 params/activations.

        Built from a shallow copy of the training model provider (same
        architecture, layer spec and process groups; fp32 dtypes; no DDP/optimizer) and refreshed by copying
        the training model's own bf16-valued parameters. Scoring p and q with it
        removes the bf16 activation-rounding noise that dominates the p/q ratio.
        """
        import copy

        t0 = time.perf_counter()
        if getattr(self, "_mfp32", None) is None:
            from megatron.core import parallel_state as ps

            if ps.get_context_parallel_world_size() > 1:
                # NeMo-RL runs MCore context parallelism only with sequence
                # packing (THD); the scorer would have to score packed, CP-sharded
                # batches through that path, which also needs packing support in
                # the streaming trainer.
                raise NotImplementedError(
                    "speculative verification scorer does not support context "
                    "parallelism (requires packed THD scoring)"
                )
            # Shallow copy: only top-level dtype/recompute fields change; process
            # groups (not copyable) are shared with the training model.
            provider = copy.copy(self.megatron_cfg.model)
            # "fp32" (default): fp32 params/activations, removes bf16 rounding
            # noise from p/q (~99% kept). "model": keep the training dtypes
            # (half the memory, bf16 speed, ~91% kept).
            if getattr(self, "_scorer_precision", "fp32") == "fp32":
                # Batch-invariant kernels are BF16-only; an fp32 scorer is built
                # outside batch-invariant mode (its higher precision, not
                # bitwise parity with training, is what it is for). TE attention
                # settings are process-wide, so they stay as configured; its
                # attention is replaced by SDPA below.
                provider.batch_invariant_mode = False
                provider.params_dtype = torch.float32
                provider.pipeline_dtype = torch.float32
                provider.bf16 = False
                provider.fp16 = False
            # On inference workers the provider builds MCore's inference-optimized
            # layers (engine-owned buffers): sharing them corrupted the engine's
            # weights/logits. The scorer always uses the standard TE layers.
            if getattr(provider, "transformer_impl", None) == "inference_optimized":
                from megatron.bridge.models.gpt_provider import (
                    GPTModelProvider,
                    default_layer_spec,
                )

                provider.transformer_impl = "transformer_engine"
                if isinstance(provider, GPTModelProvider):
                    provider.transformer_layer_spec = default_layer_spec
            # The scorer never runs under CUDA graphs; in an inference process a
            # graph-enabled copy would join the engine's graph registry.
            provider.cuda_graph_impl = "none"
            if hasattr(provider, "inference_cuda_graph_scope"):
                from megatron.core.transformer.enums import InferenceCudaGraphScope

                provider.inference_cuda_graph_scope = InferenceCudaGraphScope.none
            provider.recompute_granularity = None
            provider.recompute_method = None
            provider.recompute_num_layers = None
            if provider.params_dtype == torch.float32:
                # The fused bias-SwiGLU kernel is ~4x off the memory bound in
                # fp32; the unfused path is replaced by a Triton SwiGLU below.
                # (A training-dtype scorer keeps the training op sequence, which
                # batch-invariant kernels need to reproduce training numerics.)
                provider.bias_activation_fusion = False
            provider.finalize()
            self._mfp32 = provider.provide().cuda().eval()
            # RotaryEmbedding.forward is a process-wide lru_cache(32). In an
            # inference process the engine's CUDA graphs read tables from that
            # cache, so the scorer's own RoPE calls must not evict them.
            # (Megatron-LM #8358 graph-owned capture constants: forward is no lru_cache, no __wrapped__, nothing to do.)
            rope = getattr(self._mfp32, "rotary_pos_emb", None)
            raw = getattr(type(rope).forward, "__wrapped__", None) if rope is not None else None
            if raw is not None and hasattr(rope.forward, "cache_parameters"):
                import types

                rope.forward = types.MethodType(raw, rope)
            if provider.params_dtype == torch.float32:
                # Megatron/TE fall back to unfused fp32 attention and GLU paths.
                from thundersync_rl.scorer_kernels import (
                    use_fused_swiglu,
                    use_sdpa_attention,
                )

                attn_bf16 = os.environ.get("THUNDERSYNC_SCORER_ATTN_BF16") == "1"
                n_attn = use_sdpa_attention(
                    self._mfp32, provider, torch.bfloat16 if attn_bf16 else None
                )
                n_mlp = use_fused_swiglu(self._mfp32)
            else:
                n_attn = n_mlp = 0
            from thundersync_rl.scorer_kernels import use_selective_output

            selective = use_selective_output(self._mfp32)
            if self.rank == 0:
                print(
                    f"[thundersync scorer] dtype={provider.params_dtype} "
                    f"sdpa_attention={n_attn} fused_swiglu={n_mlp} "
                    f"selective_lm_head={selective}",
                    flush=True,
                )
            src_names = [n for n, _ in self.model.named_parameters()]
            self._mfp32_map = []
            src = dict(self.model.named_parameters())
            for name, p in self._mfp32.named_parameters():
                match = [n for n in src_names if n == name or n.endswith("." + name)]
                assert len(match) == 1, (name, match[:3])
                self._mfp32_map.append((p, src[match[0]]))
        with torch.no_grad():
            for dst, src in self._mfp32_map:
                dst.copy_(src.detach().to(dst.dtype))
        torch.cuda.synchronize()
        self._prof["export"] = self._prof.get("export", 0.0) + time.perf_counter() - t0
        return self._mfp32

    def termination_id(self) -> int | None:
        tok = getattr(self, "megatron_tokenizer", None)
        return None if tok is None else int(tok.eod)

    # ---- Deadline-truncated streaming drafts (generation rank 0) ----

    def engine_request_capacity(self) -> int:
        """Requests the engine fleet decodes at once: this engine's ``max_requests`` x engine DP.

        A draft beyond it waits for a running one to finish, so it starts only
        after a whole draft's decode time (``SpeculativeGeneration.draft_streaming``
        rejects that).
        """
        from megatron.core import parallel_state as ps

        ctx = getattr(self, "inference_context", None)
        if ctx is None:
            return 0
        return int(ctx.max_requests) * ps.get_data_parallel_world_size()

    def _own_client_loop_wanted(self) -> bool:
        """Rank 0's inference client (draft streams, their frames, aborts, fresh-row requests) on its own event
        loop (``THUNDERSYNC_CLIENT_LOOP=0``: on the engine loop, as before). Global rank 0 is a TP rank 0 of a lane
        engine, the rank its decode loop waits on; the client's coroutines no longer interleave with its engine
        steps. One client, same socket and request-id counter: requests and their ids are unchanged."""
        return os.environ.get("THUNDERSYNC_CLIENT_LOOP", "1") != "0"

    def start_drafts(
        self,
        prompts: list[list[int]],
        max_new_tokens: int | list[int],
        stream_interval: int,
        hidden_keys: list[str] | None = None,
    ) -> bool:
        """Submit draft requests as streams and return immediately.

        Tokens and the engine's log-probs of them accumulate in
        ``self._draft_tokens`` / ``self._draft_logprobs`` as partial frames
        arrive; ``collect_drafts`` aborts whatever is unfinished at the deadline.
        Any prefix of a draft is a valid draft, so truncation is harmless.

        ``hidden_keys[i]`` (engine-sourced q): tag draft i so the engine logs
        its final hidden states under that key (``install_draft_hidden_log``,
        fetched with ``pop_draft_hidden``).
        """


        if self.inference_client is None or self._inference_loop is None:
            return False
        # Fresh buffers per call, captured by the coroutines below: streams aborted
        # at an earlier deadline may still deliver late frames, which must not
        # land in this round's drafts.
        # Releasable tables (draft_table.py; long runs): entries of consumed cohorts are freed by
        # ``release_drafts`` instead of living until the next ``start_drafts`` (anchor epochs: one per epoch).
        from thundersync_rl.draft_table import DraftTable

        (
            self._draft_tokens,
            self._draft_logprobs,
            self._draft_done,
            self._draft_streams,
        ) = (DraftTable(list), DraftTable(lambda: None), DraftTable(lambda: True), {})
        self._draft_cancelled: set[int] = set()
        # Table generation: ``release_drafts`` ignores indices of an earlier table.
        self._draft_gen = int(getattr(self, "_draft_gen", 0)) + 1
        self.add_drafts(prompts, max_new_tokens, stream_interval, hidden_keys)
        return self._draft_gen

    def add_drafts(
        self,
        prompts: list[list[int]],
        max_new_tokens: int | list[int],
        stream_interval: int,
        hidden_keys: list[str] | None = None,
    ) -> list[int]:
        """Submit more draft streams next to those of the last ``start_drafts``.

        Returns their indices in the draft lists (``drafts_done``,
        ``peek_drafts``, ``abort_drafts``, ``collect_drafts``).
        """
        import asyncio

        from thundersync_rl.engine_q import HIDDEN_KEY_ATTR

        tokens, logprobs, done, streams = (
            self._draft_tokens,
            self._draft_logprobs,
            self._draft_done,
            self._draft_streams,
        )
        base = len(tokens)
        tokens += [[] for _ in prompts]
        logprobs += [[] for _ in prompts]
        done += [False] * len(prompts)

        async def run_one(i, j, prompt):
            # return_log_probs=True: the engine reports log q'(x) of each sampled
            # token (temperature 1, no top-k/top-p under speculation, so raw and
            # processed log-probs coincide).
            sp = self._build_sampling_params(greedy=False, stop_words=None)
            sp.num_tokens_to_generate = (
                max_new_tokens[j]
                if isinstance(max_new_tokens, list)
                else max_new_tokens
            )
            sp.streaming_interval = stream_interval
            if hidden_keys is not None:
                # Travels with the serialized sampling params to the engine.
                sp.add_attributes({HIDDEN_KEY_ATTR: hidden_keys[j]})
            stream = self.inference_client.add_request_streaming(prompt, sp)
            streams[i] = stream
            if i in cancelled:  # aborted before its stream existed
                self.inference_client.abort_request(stream.request_id)
            async for frame in stream:
                done[i] = apply_draft_frame(frame, tokens[i], logprobs, i) or done[i]

        cancelled = self._draft_cancelled
        for j, p in enumerate(prompts):
            asyncio.run_coroutine_threadsafe(
                run_one(base + j, j, list(p)), self._client_loop()
            )
        return list(range(base, base + len(prompts)))

    def release_drafts(self, gen: int, indices: list[int]) -> dict[str, int]:
        """Free rank 0's entries of finished / aborted drafts ``indices`` of table ``gen`` (long runs).

        Called by the driver once the cohort owning them is trained (``SpeculativeGeneration.release_consumed``).
        A still-running stream is kept (counted in "running"; the driver retries it). Indices of an older table
        (``start_drafts`` replaced it) are already gone. Returns counts (draft_table.release_entries) and the
        live entries left.
        """
        import asyncio

        from thundersync_rl.draft_table import release_entries

        if getattr(self, "_draft_tokens", None) is None or int(gen) != int(getattr(self, "_draft_gen", -1)):
            return {"released": 0, "running": 0, "gone": len(indices), "tokens": 0, "live": 0}
        done, streams, cancelled = self._draft_done, self._draft_streams, self._draft_cancelled

        def running(i: int) -> bool:
            return i < len(done) and not done[i] and i in streams and i not in cancelled

        async def run():
            # On the loop thread: no frame is half-applied while entries are dropped.
            out = release_entries(
                {
                    "tokens": self._draft_tokens,
                    "logprobs": self._draft_logprobs,
                    "done": done,
                    "streams": streams,
                    "cancelled": cancelled,
                },
                indices,
                running,
            )
            out["live"] = self._draft_tokens.live() if hasattr(self._draft_tokens, "live") else len(self._draft_tokens)
            return out

        if self._inference_loop is None:
            return asyncio.run(run())
        return asyncio.run_coroutine_threadsafe(run(), self._client_loop()).result()

    def prune_closed_logs(self, lag: int = 2) -> dict[str, int]:
        """Engine ranks, once per step (long runs): forget the keys of draft logs closed ``lag`` steps ago.

        The hidden-state log, the route log and the sticky-nucleus route log keep every closed key so late rows of
        an aborted stream are dropped; without pruning they grow by one key per stream of the run.
        """
        out = {}
        for name in ("_hidden_log", "_route_log", "_nucleus_rlog"):
            log = getattr(self, name, None)
            fn = getattr(log, "prune_closed", None)
            if fn is not None:
                out[name.strip("_")] = int(fn(lag))
        cap = getattr(self, "_verify_cap", None)
        if cap is not None and "reset_keys" in cap and not cap.get("inflight"):
            # Verification-capture key table (one key per q-recompute draft / prefill-check row): forgotten once per
            # step when no verification request is in flight (lr-aep-x: +216 keys/step without it).
            out["verify_cap_keys"] = len(cap.get("key_table") or ())
            cap["reset_keys"]()
        return out

    def longrun_snapshot(self) -> dict[str, float]:
        """Memory and bookkeeping sizes of this worker for the long-run leak probe (leak_probe.py).

        O(1)-ish per call (dict lengths, allocator counters, /proc): no GPU sync beyond the allocator query.
        """
        from thundersync_rl.leak_probe import process_stats

        out = dict(process_stats())
        tok = getattr(self, "_draft_tokens", None)
        if tok is not None:
            out["rank0_draft_entries"] = float(len(tok))
            out["rank0_draft_live"] = float(tok.live() if hasattr(tok, "live") else len(tok))
            out["rank0_draft_streams"] = float(len(getattr(self, "_draft_streams", None) or {}))
            out["rank0_draft_cancelled"] = float(len(getattr(self, "_draft_cancelled", None) or ()))
            if hasattr(tok, "_d"):
                out["rank0_draft_tokens_held"] = float(sum(len(t) for t in tok._d.values()))
        for name, pre in (("_hidden_log", "hidden_log"), ("_route_log", "route_log"), ("_nucleus_rlog", "nucleus_log")):
            log = getattr(self, name, None)
            fn = getattr(log, "bookkeeping", None)
            if fn is not None:
                out.update({f"{pre}_{k}": float(v) for k, v in fn().items()})
        cap = getattr(self, "_verify_cap", None)
        if cap is not None:
            with cap["lock"]:
                out["verify_cap_keys"] = float(len(getattr(cap["store"], "_rows", {}) or {}))
                out["verify_cap_events"] = float(len(cap["events"]))
            kt = cap.get("key_table")
            if kt is not None:
                out["verify_cap_key_table"] = float(len(kt))
        out["engine_anchor_refs"] = float(len(getattr(self, "_engine_anchor_refs", None) or {}))
        stash = getattr(self, "_head_stash", None)
        if stash is not None:
            out["head_stash_versions"] = float(len(stash))
            out["head_stash_gb"] = float(self.head_stash_bytes()) / 2**30
        ws = getattr(self, "_weight_stash", None)
        if ws is not None:
            out["weight_stash_versions"] = float(len(ws))
        out["grad_snapshots"] = float(len(self.__dict__.get("_grad_snapshots", {}) or {}))
        pc = getattr(self, "_prefix_cache", None)
        if pc is not None:
            for k in ("gpu_bytes", "host_bytes", "entries"):
                v = getattr(pc, k, None)
                v = v() if callable(v) else v
                if isinstance(v, (int, float)):
                    out[f"prefix_cache_{k}"] = float(v)
        return out

    def draft_token_total(self) -> tuple[int, int]:
        """(streamed draft tokens so far, unfinished drafts) on generation rank 0 (bench)."""
        tokens = getattr(self, "_draft_tokens", None) or []
        done = getattr(self, "_draft_done", None) or []
        return sum(len(t) for t in tokens), sum(1 for d in done if not d)

    def draft_lengths(self, indices: list[int]) -> list[tuple[int, bool]]:
        """(streamed token count, finished) of the listed drafts (rank 0).

        Counts from streamed frames: a readiness signal only (after an
        eviction frames may skip tokens); contents come from ``peek_drafts``.
        """
        tokens, done = self._draft_tokens, self._draft_done
        return [(len(tokens[i]), bool(done[i])) for i in indices]

    def count_unfinished_drafts(self, n: int) -> int:
        """How many of drafts 0..n-1 are still running (rank 0; O(1) reply).

        A count only (``window_pace.WindowPacer.hold`` releases the cut at 0).
        """
        done = self._draft_done
        return sum(1 for i in range(n) if not done[i])

    def peek_drafts(
        self,
        indices: list[int],
        records: dict[int, tuple] | None = None,
        record_refs: list | None = None,
        request_ids: list[int] | None = None,
    ) -> list[tuple[np.ndarray, bool, np.ndarray | None]]:
        """(tokens, finished, engine logprobs) of the listed drafts, left running.

        Unfinished drafts read ``records`` (the engines' request records,
        ``draft_records``) when given, as ``collect_drafts`` does; finished ones
        their final frame. ``record_refs``: the other engine ranks'
        ``draft_records`` results as object refs, fetched here, and
        ``request_ids`` this rank's own records to read (the records never pass
        through the driver).
        """
        import asyncio

        records = _merge_record_refs(records, record_refs, self, request_ids)

        tokens, logprobs, done, streams = (
            self._draft_tokens,
            self._draft_logprobs,
            self._draft_done,
            self._draft_streams,
        )

        async def snapshot():
            out = []
            for i in indices:
                rec = None
                if not done[i] and i in streams:
                    rec = (records or {}).get(int(streams[i].request_id))
                    if rec is None:
                        # Finished in the engine after its record was read: its
                        # final frame (the merged record) is in flight.
                        await self._await_final_frame(i, done, tokens, "peek")
                if rec is None:
                    out += snapshot_drafts([tokens[i]], [logprobs[i]], [done[i]])
                    continue
                toks, lps = rec
                ok = lps is not None and len(lps) == len(toks)
                out.append((typed_tokens(toks), False, typed_logq(lps) if ok else None))
            return out

        return asyncio.run_coroutine_threadsafe(
            snapshot(), self._client_loop()
        ).result()

    def abort_drafts(self, indices: list[int]) -> None:
        """Abort the listed drafts that are still running (rank 0)."""
        import asyncio

        done, streams = self._draft_done, self._draft_streams

        async def abort():
            for i in indices:
                self._draft_cancelled.add(i)
                if not done[i] and i in streams:
                    self.inference_client.abort_request(streams[i].request_id)

        asyncio.run_coroutine_threadsafe(abort(), self._client_loop()).result()

    def draft_request_ids(self, indices: list[int] | None = None) -> dict[int, int]:
        """Engine request id of each unfinished draft (rank 0 holds the streams).

        ``indices``: only these drafts (default: all).
        """
        done = getattr(self, "_draft_done", None) or []
        streams = getattr(self, "_draft_streams", None) or {}
        # run_one (on the inference loop thread) inserts streams[i] while this
        # actor thread reads: iterate an atomic snapshot (aep1e-x8k died with
        # "dictionary changed size during iteration" here).
        return {
            i: int(st.request_id)
            for i, st in tuple(streams.items())
            if not done[i] and (indices is None or i in indices)
        }

    def draft_records(self, request_ids: list[int]) -> dict[int, tuple]:
        """(tokens, engine log-probs) of the listed requests this engine holds.

        Read from the engine's request records (``record.merge()``, as its final
        reply does), not from streamed partial frames: after an eviction the
        engine resumes a request as a new record whose generated tokens restart
        at 0 while its partial-frame counter does not, so the frames silently
        skip tokens (Megatron DynamicInferenceEngine._try_send_streaming_partials).
        """
        import asyncio

        engine = getattr(self, "dynamic_inference_engine", None)
        loop = getattr(self, "_inference_loop", None)
        if engine is None or loop is None:
            return {}

        async def read():
            out = {}
            for rid in request_ids:
                entry = engine.requests.get(rid)
                if entry is not None:
                    merged = entry.record.merge()
                    out[rid] = (
                        typed_tokens(merged.generated_tokens),
                        typed_logq(_plain_floats(merged.generated_log_probs)),
                    )
            return out

        return asyncio.run_coroutine_threadsafe(read(), loop).result()

    async def _await_final_frame(self, i: int, done, tokens, where: str) -> None:
        """Wait (on the engine loop) for draft ``i``'s final frame; raise on timeout.

        Called only for an unfinished stream whose request has no engine record
        (finished in the engine; final reply in flight). An empty stream needs
        no wait (no frame arrived, nothing can be skipped). Counted in
        ``draft_collect_stats``.
        """
        import asyncio

        st = self.__dict__.setdefault(
            "_collect_stats", {"unrecorded": 0, "unrecorded_empty": 0, "waited_final": 0, "wait_s": 0.0}
        )
        st["unrecorded"] += 1
        if not tokens[i]:
            st["unrecorded_empty"] += 1
            return
        timeout = float(os.environ.get("THUNDERSYNC_COLLECT_FINAL_TIMEOUT_S", "30"))
        t0 = time.perf_counter()
        while not done[i]:
            if time.perf_counter() - t0 > timeout:
                raise RuntimeError(
                    f"{where}_drafts: draft {i} has no engine record and its final "
                    f"frame did not arrive within {timeout:.0f} s; its streamed frames "
                    f"({len(tokens[i])} tokens) may skip tokens after an eviction, so "
                    "they are not used (THUNDERSYNC_COLLECT_FINAL_TIMEOUT_S)"
                )
            await asyncio.sleep(0.002)
        st["waited_final"] += 1
        st["wait_s"] += time.perf_counter() - t0

    def draft_collect_stats(self) -> dict[str, float]:
        """Rank 0: record-fallback counters since the last call (then reset)."""
        st = self.__dict__.pop("_collect_stats", None) or {}
        return {k: float(v) for k, v in st.items()}

    def collect_drafts(
        self,
        records: dict[int, tuple] | None = None,
        record_refs: list | None = None,
        request_ids: list[int] | None = None,
    ) -> list[tuple[np.ndarray, bool, np.ndarray | None]]:
        """Abort unfinished drafts; return (tokens, finished, engine logprobs) per draft.

        The log-probs are None when the engine did not report one per token.
        ``record_refs``, ``request_ids``: as in ``peek_drafts``.
        """
        import asyncio

        records = _merge_record_refs(records, record_refs, self, request_ids)

        tokens, logprobs, done, streams = (
            self._draft_tokens,
            self._draft_logprobs,
            self._draft_done,
            self._draft_streams,
        )

        async def abort_and_snapshot():
            # Snapshot on the loop thread so no frame is half-applied.
            stats = {"from_record": 0, "fallback": 0, "differed": 0}
            # Phase 1 (may yield to the loop): streams without an engine record
            # finished (or failed) in the engine after the records were read
            # (``engine.requests`` drops a request only then). Their streamed
            # frames may skip tokens after an eviction, so they are never used:
            # wait for the final frame, which carries the merged record (raises
            # on timeout). Done before any content is replaced below, so no late
            # frame can land on a draft after its record content was taken.
            for i, stream in list(streams.items()):
                if not done[i] and (records or {}).get(int(stream.request_id)) is None:
                    stats["fallback"] += 1
                    await self._await_final_frame(i, done, tokens, "collect")
            # Phase 2 (no await): record contents, then abort.
            for i, stream in list(streams.items()):
                if done[i]:
                    continue
                rec = (records or {}).get(int(stream.request_id))
                if rec is None:
                    # Unrecorded and still running: it had no record and no frame
                    # at the cut (not admitted yet). Frames that arrived during
                    # phase 1 are after the cut: the draft is the empty prefix.
                    del tokens[i][:]
                    if logprobs[i] is not None:
                        logprobs[i] = []
                else:
                    toks, lps = rec
                    toks = [int(t) for t in toks]
                    lps = None if lps is None else [float(v) for v in lps]
                    stats["from_record"] += 1
                    stats["differed"] += int(toks != tokens[i])
                    tokens[i][:] = toks
                    if logprobs[i] is not None or lps is not None:
                        logprobs[i] = (
                            lps
                            if lps is not None and len(lps) == len(tokens[i])
                            else None
                        )
                self.inference_client.abort_request(stream.request_id)
            if env_flag("THUNDERSYNC_HIDDEN_DEBUG"):
                print(f"[collect dbg] {stats}", flush=True)
            return snapshot_drafts(tokens, logprobs, done)

        return asyncio.run_coroutine_threadsafe(
            abort_and_snapshot(), self._client_loop()
        ).result()

    # ---- Memory-scalable block verification: stash weights, not q ----
    # q is recomputed at verification time from a bf16 stash of the weights that
    # drafted each position (2 bytes/param per model-parallel shard per stashed
    # version), so memory is O(params + one batch x vocab) instead of
    # O(all draft tokens x vocab).

    @torch.no_grad()
    def stash_weights(self, version: int, keep: list[int], precision: str) -> int:
        """Stash the current training parameters as ``version``; drop others not in ``keep``."""
        self._prof = {}
        self._scorer_precision = precision
        self._fp32_scorer()  # builds the scorer and its parameter map once
        if not hasattr(self, "_weight_stash"):
            self._weight_stash: dict[int, list[torch.Tensor]] = {}
        self._weight_stash[version] = _restash(
            self._weight_stash, version, keep, [src for _, src in self._mfp32_map]
        )
        self._log_mem(f"stash_weights v={version} n={len(self._weight_stash)}")
        return len(self._weight_stash)

    @torch.no_grad()
    def _load_scorer(self, version: int | None) -> None:
        """Load stashed ``version`` (None: current training weights) into the scorer."""
        t0 = time.perf_counter()
        srcs = (
            [src for _, src in self._mfp32_map]
            if version is None
            else self._weight_stash[version]
        )
        for (dst, _), src in zip(self._mfp32_map, srcs):
            dst.copy_(src)
        torch.cuda.synchronize()
        self._prof["load"] = self._prof.get("load", 0.0) + time.perf_counter() - t0

    @torch.no_grad()
    @_model_in_eval
    @_lg_feature("verify_block")
    def verify_drafts_block_stashed(
        self,
        rows: list[torch.Tensor],
        prompt_lens: list[int],
        segments: list[list[tuple[int, int, int]]],
        vocab_limit: int,
        seed: int,
        keys: list[str],
        batch_tokens: int,
        precision: str,
        lazy: bool = False,
    ) -> list[tuple[int, dict[str, Any]]]:
        """Block-verify drafts whose q is recomputed per batch from stashed weights.

        ``segments[i]``: [(version, start, end)] - draft tokens [start, end) of row
        i were drafted by stashed weights ``version``. p is the current weights.
        Returns [(row_index, {"accepted", "next", "logprobs", "draft_logq"})]:
        "logprobs" = log p of the kept tokens (accepted prefix + next),
        "draft_logq" = log q of the accepted prefix.

        ``lazy``: lazy block verification (residual masses only where the
        backward scan needs them; pathwise identical to the eager rule given the
        same logits). Logit-free when the scorer supports position selection:
        keep final hidden states [T, d] instead of [T, V] logits, compute the
        logsumexps and draft-token logits in row chunks, and recompute full rows
        only at visited positions (``THUNDERSYNC_LAZY_LOGITS=1`` keeps the logits
        and only makes the scan lazy).
        """
        from thundersync_rl.block_verification_fused import (
            block_verify_logits,
            block_verify_logits_lazy,
        )
        from thundersync_rl.scorer_kernels import supports_selection

        self._prof = {}
        t_all = time.perf_counter()
        self._scorer_precision = precision
        scorer = self._fp32_scorer()
        from megatron.core import parallel_state as ps

        logit_free = (
            lazy
            and supports_selection(scorer)
            and os.environ.get("THUNDERSYNC_LAZY_LOGITS") != "1"
        )
        verify_fn = block_verify_logits_lazy if lazy else block_verify_logits

        # Same stream on every TP rank of a replica: identical decisions.
        gen = torch.Generator(device="cuda").manual_seed(
            seed * 1_000_003 + ps.get_data_parallel_rank()
        )
        results = []
        batches = list(self._iter_row_id_batches(rows, batch_tokens, keys))
        plans = [
            sorted({v for i in group for v, a, b in segments[i] if b > a})
            for group, _ in batches
        ]
        lockstep = self._expert_parallel_lockstep(plans)
        for j in range(len(batches) if lockstep is None else len(lockstep)):
            if j >= len(batches):
                # Expert parallelism: out of rows, join the other ranks' forwards.
                for v in lockstep[j]:
                    self._load_scorer(v)
                    self._dummy_score(scorer)
                self._load_scorer(None)
                self._dummy_score(scorer)
                continue
            group, ids = batches[j]
            pos = torch.arange(ids.shape[1], device="cuda").expand(len(group), -1)
            versions = plans[j] if lockstep is None else lockstep[j]
            if logit_free:
                results.extend(
                    self._verify_batch_logit_free(
                        scorer,
                        group,
                        ids,
                        pos,
                        versions,
                        rows,
                        prompt_lens,
                        segments,
                        vocab_limit,
                        gen,
                    )
                )
                continue
            lq_rows = self._stashed_q_logits(
                scorer,
                group,
                ids,
                pos,
                prompt_lens,
                segments,
                vocab_limit,
                lockstep_versions=None if lockstep is None else lockstep[j],
            )
            self._load_scorer(None)
            spans = [(prompt_lens[i] - 1, rows[i].numel()) for i in group]
            logits = self._score(
                scorer,
                ids,
                pos,
                select=(
                    torch.cat([torch.arange(a, e) for a, e in spans]).cuda(),
                    torch.cat(
                        [torch.full((e - a,), r) for r, (a, e) in enumerate(spans)]
                    ).cuda(),
                ),
            )
            if logits is None:
                continue
            off = 0
            for r, i in enumerate(group):
                n, plen = rows[i].numel(), prompt_lens[i]
                g = n - plen
                p_logits = logits[off : off + g + 1, :vocab_limit]
                off += g + 1
                parts = [t for _, t in sorted(lq_rows.pop(i), key=lambda x: x[0])]
                q_logits = torch.cat(parts) if parts else p_logits[:0]
                assert q_logits.shape[0] == g, (q_logits.shape, g)
                draft = ids[r, plen:n]
                # Raw logits in, no [g, V] log-softmax / fp64 temporaries.
                tau, y, logp = verify_fn(p_logits, q_logits, draft, gen)
                # log q of the accepted draft tokens (same vocab slice as the
                # acceptance ratios) for the behavior log-probs (speculative._plan).
                qa = q_logits[:tau].float()
                logq = (
                    qa.gather(-1, draft[:tau].view(-1, 1))[:, 0]
                    - torch.logsumexp(qa, -1)
                    if tau
                    else qa.new_empty(0)
                )
                if self._batch_invariant:
                    # Report log-probs exactly as training computes them (fp32
                    # log-softmax + gather), so they match the trainer bitwise.
                    kept = torch.cat([draft[:tau], torch.tensor([y], device="cuda")])
                    logp = torch.log_softmax(p_logits[: tau + 1].float(), -1).gather(
                        -1, kept.unsqueeze(-1)
                    )[:, 0]
                results.append(
                    (
                        i,
                        {
                            "accepted": tau,
                            "next": y,
                            "logprobs": logp.tolist(),
                            # log q of the accepted draft tokens (same vocab
                            # slice as verification), for the behavior
                            # log-probs log p + log q' - log q (speculative._plan).
                            "draft_logq": logq.tolist(),
                        },
                    )
                )
            del logits
        self._log_mem("verify_drafts_block_stashed")
        if env_flag("THUNDERSYNC_SPEC_PROF"):
            pool = (
                "gen"
                if getattr(self, "dynamic_inference_engine", None) is not None
                else "learner"
            )
            ntok = sum(rows[i].numel() for i in range(len(rows)))
            print(
                f"[spec prof] verify_stashed pool={pool} rank={self.rank} rows={len(rows)} tok={ntok} lazy={int(lazy)} logit_free={int(logit_free)} total={time.perf_counter() - t_all:.3f} {self._prof}",
                flush=True,
            )
        if lazy and os.environ.get("THUNDERSYNC_LAZY_CHECK") == "1":
            # Debug: rerun the eager logits path with the same seed and compare.
            ref = dict(
                self.verify_drafts_block_stashed(
                    rows,
                    prompt_lens,
                    segments,
                    vocab_limit,
                    seed,
                    keys,
                    batch_tokens,
                    precision,
                    lazy=False,
                )
            )
            diff = sum(
                (o["accepted"], o["next"]) != (ref[i]["accepted"], ref[i]["next"])
                for i, o in results
            )
            dlp = max(
                (
                    max(abs(a - b) for a, b in zip(o["logprobs"], ref[i]["logprobs"]))
                    for i, o in results
                    if o["accepted"] == ref[i]["accepted"]
                ),
                default=0.0,
            )
            print(
                f"[lazy check] rank={self.rank} rows={len(results)} logit_free="
                f"{int(logit_free)} decision_mismatches={diff} max|dlogp|={dlp:.3e}",
                flush=True,
            )
        return results

    @torch.no_grad()
    def _stashed_q_logits(
        self,
        scorer,
        group: list[int],
        ids: torch.Tensor,
        pos: torch.Tensor,
        prompt_lens: list[int],
        segments: list[list[tuple[int, int, int]]],
        vocab_limit: int,
        lockstep_versions: list[int] | None = None,
    ) -> dict[int, list[tuple[int, torch.Tensor]]]:
        """Q logits of a batch's draft positions, from the weights that drafted them.

        Returns row -> [(segment start, [segment length, V] logits)]; empty off
        the last pipeline stage. ``lockstep_versions``: the expert-parallel
        group's common versions for this batch slot (see
        ``_expert_parallel_lockstep``); versions this rank has no rows for run a
        dummy forward.
        """
        lq_rows: dict[int, list[tuple[int, torch.Tensor]]] = {i: [] for i in group}
        versions = lockstep_versions
        if versions is None:
            versions = sorted({v for i in group for v, _, _ in segments[i]})
        for v in versions:
            # q of version v is needed only at its segments' positions; causal
            # attention lets each row stop at its last such position.
            segs = [
                (r, i, a, b)
                for r, i in enumerate(group)
                for sv, a, b in segments[i]
                if sv == v and b > a
            ]
            if not segs:
                if lockstep_versions is not None:
                    # Expert parallelism: another rank needs version v here.
                    self._load_scorer(v)
                    self._dummy_score(scorer)
                continue
            sub_rows = sorted({r for r, _, _, _ in segs})
            local = {r: k for k, r in enumerate(sub_rows)}
            width = max(prompt_lens[i] - 1 + b for _, i, _, b in segs)
            s_idx = torch.cat(
                [
                    torch.arange(prompt_lens[i] - 1 + a, prompt_lens[i] - 1 + b)
                    for _, i, a, b in segs
                ]
            )
            b_idx = torch.cat([torch.full((b - a,), local[r]) for r, _, a, b in segs])
            self._load_scorer(v)
            logits = self._score(
                scorer,
                ids[sub_rows, :width],
                pos[: len(sub_rows), :width],
                select=(s_idx.cuda(), b_idx.cuda()),
            )
            if logits is None:  # not the last pipeline stage
                continue
            off = 0
            for _, i, a, b in segs:
                lq_rows[i].append((a, logits[off : off + b - a, :vocab_limit]))
                off += b - a
            del logits
        return lq_rows

    def _verify_batch_logit_free(
        self,
        scorer,
        group,
        ids,
        pos,
        versions,
        rows,
        prompt_lens,
        segments,
        vocab_limit,
        gen,
    ) -> list[tuple[int, dict[str, Any]]]:
        """One batch of ``verify_drafts_block_stashed`` without [T, V] logits.

        Same forwards, version order and generator calls as the logits path; q
        and p come as final hidden states, reduced to (logsumexp, draft-token
        logit) per position in row chunks (``ChunkedHead``). The lazy scan then
        recomputes full rows only where it needs r_i, plus the row it samples
        from. q rows of version v are recomputed with the head weight of stash v
        (the values the scorer held during that forward).
        """
        from thundersync_rl.block_verification_fused import block_verify_stats_lazy
        from thundersync_rl.logit_free import ChunkedHead

        chunk = int(os.environ.get("THUNDERSYNC_LAZY_CHUNK", "512"))
        # Row id -> [(start, head, base, length, lse, picked)] per q segment.
        q_parts: dict[int, list[tuple]] = {i: [] for i in group}
        heads = []
        for v in versions:
            segs = [
                (r, i, a, b)
                for r, i in enumerate(group)
                for sv, a, b in segments[i]
                if sv == v and b > a
            ]
            if not segs:
                # Expert parallelism: another rank needs version v here.
                self._load_scorer(v)
                self._dummy_score(scorer)
                continue
            sub_rows = sorted({r for r, _, _, _ in segs})
            local = {r: k for k, r in enumerate(sub_rows)}
            width = max(prompt_lens[i] - 1 + b for _, i, _, b in segs)
            s_idx = torch.cat(
                [
                    torch.arange(prompt_lens[i] - 1 + a, prompt_lens[i] - 1 + b)
                    for _, i, a, b in segs
                ]
            )
            b_idx = torch.cat([torch.full((b - a,), local[r]) for r, _, a, b in segs])
            self._load_scorer(v)
            out = self._score_hidden(
                scorer,
                ids[sub_rows, :width],
                pos[: len(sub_rows), :width],
                (s_idx.cuda(), b_idx.cuda()),
                vocab_limit,
            )
            if out is None:  # not the last pipeline stage
                continue
            hidden, logits_fn, w_live = out
            tokens = torch.cat(
                [
                    rows[i][prompt_lens[i] + a : prompt_lens[i] + b]
                    for _, i, a, b in segs
                ]
            ).cuda()
            head = ChunkedHead(hidden, logits_fn, lambda w=w_live: w, chunk)
            lse, picked = head.stats(tokens)
            # Rows recomputed later: the scorer then holds other weights.
            head.weight = self._stashed_head_weight(v, w_live)
            heads.append(head)
            off = 0
            for _, i, a, b in segs:
                n = b - a
                q_parts[i].append(
                    (a, head, off, n, lse[off : off + n], picked[off : off + n])
                )
                off += n
        self._load_scorer(None)
        spans = [(prompt_lens[i] - 1, rows[i].numel()) for i in group]
        out = self._score_hidden(
            scorer,
            ids,
            pos,
            (
                torch.cat([torch.arange(a, e) for a, e in spans]).cuda(),
                torch.cat(
                    [torch.full((e - a,), r) for r, (a, e) in enumerate(spans)]
                ).cuda(),
            ),
            vocab_limit,
        )
        if out is None:
            return []
        hidden, logits_fn, w_live = out
        tokens, pins, off = [], [], 0
        for i in group:
            draft = rows[i][prompt_lens[i] :]
            tokens += [draft, torch.zeros(1, dtype=draft.dtype)]  # bonus: no token
            off += draft.numel() + 1
            pins.append(off - 1)
        head_p = ChunkedHead(hidden, logits_fn, lambda: w_live, chunk)
        heads.append(head_p)
        lse_p_all, picked_p_all = head_p.stats(torch.cat(tokens).cuda(), pins)
        results = []
        off = 0
        empty = torch.empty(0, dtype=torch.float32, device="cuda")
        for i in group:
            g = rows[i].numel() - prompt_lens[i]
            lse_p = lse_p_all[off : off + g + 1]
            lpx = picked_p_all[off : off + g] - lse_p[:g]
            parts = sorted(q_parts.pop(i), key=lambda x: x[0])
            lse_q = torch.cat([pt[4] for pt in parts]) if parts else empty
            lqx = (torch.cat([pt[5] for pt in parts]) if parts else empty) - lse_q
            assert lqx.numel() == g, (lqx.numel(), g)

            def q_rows(idx, parts=parts):
                out = []
                for t in idx.tolist():
                    a, head, base, _, _, _ = next(
                        pt for pt in parts if pt[0] <= t < pt[0] + pt[3]
                    )
                    out.append(head.row(base + t - a))
                return torch.stack(out)

            tau, y, logp, n_eval = block_verify_stats_lazy(
                lpx,
                lqx,
                lse_p,
                lse_q,
                lambda idx, off=off: head_p.rows(idx + off),
                q_rows,
                gen,
            )
            self._prof["lazy_rows"] = self._prof.get("lazy_rows", 0) + n_eval
            results.append(
                (
                    i,
                    {
                        "accepted": tau,
                        "next": y,
                        "logprobs": logp.tolist(),
                        "draft_logq": lqx[:tau].tolist(),
                    },
                )
            )
            off += g + 1
        for head in heads:
            head.release()
        return results

    def _stashed_head_weight(self, version, w_live):
        """Callable giving the LM head weight of stash ``version`` (scorer dtype).

        Bitwise the values ``_load_scorer(version)`` copied into ``w_live``.
        """
        k = next(j for j, (dst, _) in enumerate(self._mfp32_map) if dst is w_live)
        return lambda: self._weight_stash[version][k].to(w_live.dtype)

    def _score_hidden(self, scorer, ids, pos, select, vocab_limit):
        """Final hidden states at ``select`` plus a function computing their logits.

        Returns (hidden [n, h], logits_fn(h, weight) -> fp32 [m, vocab_limit],
        head weight) on the last pipeline stage, else None. ``logits_fn`` is the
        forward's LM head (with its logit scaling), all-gathered over the
        vocabulary under tensor parallelism one row chunk at a time.
        """
        from megatron.core import parallel_state as ps
        from thundersync_rl.scorer_kernels import capture_head_input

        with capture_head_input(scorer, select) as cap:
            out = self._score_full(scorer, ids, pos)
        if out is None:
            return None
        ((hidden, weight, rgo),) = cap
        layer = scorer.output_layer
        w_live = layer.weight if weight is None else weight
        tp = ps.get_tensor_model_parallel_world_size()
        gathered = layer.gather_output if rgo is None else rgo
        scale = getattr(scorer, "_scale_logits", None)

        def logits_fn(h, w):
            z, _ = layer._ts_orig_forward(
                h.unsqueeze(1), weight=w, runtime_gather_output=rgo
            )
            if scale is not None:
                z = scale(z)
            z = z[:, 0].float()
            if tp > 1 and not gathered:
                # Vocab-parallel -> full vocab, one chunk of rows at a time.
                parts = [torch.empty_like(z) for _ in range(tp)]
                torch.distributed.all_gather(
                    parts, z.contiguous(), group=ps.get_tensor_model_parallel_group()
                )
                z = torch.cat(parts, dim=-1)
            return z[:, :vocab_limit]

        return hidden, logits_fn, w_live

    # ---- Engine-sourced q under batch invariance (verify_source="engine") ----
    # q of a drafted token is the engine's own log-prob; full q rows come from
    # the engine's logged final hidden states and an LM-head-only stash per
    # pending version (engine_q.py). p is the training model's own forward.

    def _initialize_inference_engine(self, mcore_generation_config: dict) -> None:
        """Build the inference engine with the draft hidden-state copy installed.

        The engine captures its CUDA graphs in its constructor; the LM-head copy
        pre-hook must exist by then on every rank (the captures run TP/EP
        collectives, so all ranks capture the same graphs), or graph replays
        skip it. The constructor is wrapped for the duration of the build.
        """
        from megatron.core.inference.engines import dynamic_engine as de

        orig = de.DynamicInferenceEngine
        de.DynamicInferenceEngine = self._engine_class(orig)
        try:
            super()._initialize_inference_engine(mcore_generation_config)
        finally:
            de.DynamicInferenceEngine = orig
        from thundersync_rl.sprsd import SALT_ENV

        salt = os.environ.get(SALT_ENV, "").strip()
        if salt and getattr(self, "dynamic_inference_engine", None) is not None:
            # thundersync.mode=sprsd (SPR-SD: the stock loop has no ThunderSync stage to install them): the keyed
            # sampler and XS-SD's tree (+ MTP head) on this engine, as colo's stage setup does for XS-SD
            self.install_wide_sampler(salt=int(salt), key64=True)
            info = self.install_tree_xssd()
            print(f"[sprsd] rank={getattr(self, 'rank', '?')} keyed sampler + tree {info}", flush=True)

    def _engine_class(self, orig: type) -> type:
        """``orig`` (MCore's DynamicInferenceEngine) with the draft logs armed.

        The hidden-state copy and the route record are installed before the
        constructor captures its CUDA graphs. With the route record, MCore's
        own routing-replay bookkeeping is dropped and async scheduling is
        allowed (``_drop_mcore_route_bookkeeping``).
        """
        worker = self

        class _Engine(orig):  # type: ignore[misc, valid-type]
            def __init__(self, controller, context, *args, **kwargs):
                worker._install_hidden_copy(context)
                worker._install_route_record(context)
                if getattr(worker, "_route_record", None) is not None:
                    _drop_mcore_route_bookkeeping(controller)
                    # Before graph capture: the routers read the buffer.
                    worker._install_anchor_apply(context)
                # Engine-side verification: every forward's head-input rows are
                # copied before the last-token selection (captured in the graphs).
                worker._install_verify_capture(context)
                super().__init__(controller, context, *args, **kwargs)
                _install_one_tolist_log_probs(controller)
                # Exact top-p: innermost sampler wrapper (installed before the
                # draft logs wrap the kernel), so they see the raw logits.
                worker._install_nucleus_sampler(controller)
                if getattr(worker, "_route_log_state", None) is not None:
                    # An engine rebuilt after the route log was installed.
                    worker._wrap_route_log(controller, context)
                if getattr(worker, "_anchor_apply", None) is not None:
                    worker._wrap_anchor_apply(self, controller, context)
                if getattr(worker, "_verify_cap", None) is not None:
                    worker._wrap_verify_capture(self, controller, context)

            def _validate_async_sched_support_for_config(self) -> None:
                if getattr(worker, "_route_record", None) is None:
                    return super()._validate_async_sched_support_for_config()
                # MCore rejects routing replay under async scheduling because of
                # its own per-step bookkeeping (a synchronous host copy and
                # per-KV-block storage, rebuilt per request at completion),
                # which is dropped here. Our route log reads each forward's
                # routes right behind it on its stream, from the token layout
                # at launch, so async scheduling's later compaction does not
                # affect it. The remaining checks still apply.
                cfg = self.controller.inference_wrapped_model.model.config
                cfg.moe_enable_routing_replay = False
                try:
                    super()._validate_async_sched_support_for_config()
                finally:
                    cfg.moe_enable_routing_replay = True

        return _Engine

    def _build_sampling_params(self, greedy, stop_words, **kwargs):
        """MCore sampling params; under the exact top-p policy Megatron's own
        top-p is off (top_p 0): the nucleus is masked by ``_install_nucleus_sampler``."""
        sp = super()._build_sampling_params(greedy, stop_words, **kwargs)
        from nemo_rl.models.megatron import exact_nucleus

        if exact_nucleus.active() and not greedy:
            sp.top_p = 0.0
        return sp

    def _install_nucleus_sampler(self, controller) -> None:
        """Exact top-p on the engine (``exact_nucleus.py``; no-op when off).

        Wraps the sampling kernel: the rows it samples (``logits[gather_indices
        [:n]]``) are replaced in place by their nucleus-masked copy (-inf outside
        the canonical nucleus of the row's fp32 log_softmax), then the original
        kernel samples them. The in-place write lands in the controller's logits
        buffer, so the selected-token log-prob the controller computes next
        (``raw_logprobs``: ``log_softmax(logits.float())``) is the truncated
        policy's log-prob, the value the learner computes from the same raw row.
        Megatron's own top-p is off (``_build_sampling_params``). Rows of
        requests that sample nothing (prefill chunks) are never in
        ``gather_indices[:n]``. No host sync (sort / cumsum / masked_fill only).
        Sticky nucleus: the engine samples the plain nucleus (drafts, tails and
        fresh rollouts are all "self"); the drafter's nucleus ids are logged by
        ``install_nucleus_log``.
        """
        from nemo_rl.models.megatron import exact_nucleus

        if not exact_nucleus.active():
            return
        sampling = controller._sampling
        if getattr(sampling, "_nucleus_installed", False):
            return
        inner = sampling.sample_kernel
        cap = exact_nucleus.anchor_cap()
        worker = self

        def nucleus_kernel(logits, n, context, *, gather_indices=None, token_to_request_index=None, **kw):
            if token_to_request_index is not None:
                raise RuntimeError(
                    "exact top-p: in-engine speculative decoding is not supported"
                )
            if n > 0:
                idx = (
                    gather_indices[:n]
                    if gather_indices is not None
                    else torch.arange(n, device=logits.device)
                )
                hook = getattr(worker, "_nucleus_ids_hook", None)
                ids = exact_nucleus.mask_sampled_rows_(
                    logits, idx, cap if hook is not None else 0
                )
                if hook is not None:
                    # Sticky nucleus: the drafter's nucleus ids of the sampled
                    # rows, logged per tagged request (install_nucleus_log).
                    hook(ids, n, context)
            # The inner kernel is read at call time: install_wide_sampler (53-bit
            # keyed race, installed later by the driver) replaces it, so the nucleus
            # mask stays the OUTER wrapper and the race samples the masked rows.
            return sampling._nucleus_inner(
                logits,
                n,
                context,
                gather_indices=gather_indices,
                token_to_request_index=token_to_request_index,
                **kw,
            )

        sampling._nucleus_inner = inner
        sampling.sample_kernel = nucleus_kernel
        sampling._nucleus_installed = True
        print(f"[nucleus] engine sampler: {exact_nucleus.describe()}", flush=True)

    def _install_hidden_copy(self, context) -> None:
        """Copy each forward's LM-head input rows into a static buffer.

        One [max_requests, d] buffer per engine (bf16: 14 MB at d=7168 and
        1024 requests), written by a pre-hook on the output layer; the copy is
        captured into the CUDA graphs. Only the last pipeline stage has an
        output layer. Rows are read by ``install_draft_hidden_log``'s sampler.
        """
        model = _unwrap(self._gen_model())
        layer = getattr(model, "output_layer", None)
        if layer is None or getattr(self, "_hidden_copy", None) is not None:
            return
        try:
            from megatron.core.inference.utils import InferenceMode
        except ImportError:  # pragma: no cover
            InferenceMode = None
        materialized = bool(context.config.materialize_only_last_token_logits)
        # Tied embeddings: the output layer holds no weight of its own.
        weight = layer.weight
        if weight is None:
            weight = model.shared_embedding_or_output_weight()
        from nemo_rl.models.megatron.fp16_gemm import head_input_dtype

        # The head input dtype: fp16 under forward_gemm_precision=fp16, where the
        # head GEMM rounds its fp32 input to fp16 first (the copy rounds the same
        # way, so q recomputed from these rows is exact); else the weight dtype.
        static = torch.empty(
            (int(context.max_requests), weight.shape[1]),
            dtype=head_input_dtype(weight.dtype),
            device=weight.device,
        )
        filled = [False]

        def hook(module, args, kwargs):
            if InferenceMode is not None and not InferenceMode.is_active():
                return  # not an engine forward
            h = args[0] if args else kwargs["input_"]
            rows = h.reshape(-1, h.shape[-1])
            if not materialized:
                if torch.cuda.is_current_stream_capturing():
                    return  # per-token rows: only copied on eager forwards
                rows = context.last_token_logits(rows.unsqueeze(0))
            k = min(rows.shape[0], static.shape[0])
            static[:k].copy_(rows[:k])
            filled[0] = True

        layer.register_forward_pre_hook(hook, with_kwargs=True)
        self._hidden_copy = {
            "static": static,
            "filled": filled,
            "weight": weight,
            "materialized": materialized,
        }

    # ---- Sticky MoE routing: the drafter's routes (anchors), engine side ----

    def _install_route_record(self, context) -> None:
        """Arm MCore ``RouterReplay`` RECORD on the engine's MoE layers.

        Sticky routing only (``policy.megatron_cfg.sticky_routing_delta``).
        The model config has ``moe_enable_routing_replay`` (sticky routing sets
        it), so MCore's dynamic context binds each router's record buffer per
        step: its ``routing_indices_buffer`` on CUDA-graph steps (the buffer
        the graphs are captured with), none on eager steps, where the router
        keeps the forward's top-k in ``recorded_topk_idx``. The route log
        reads them from there (``_step_routes``). Armed here, before the
        engine constructor captures its CUDA graphs, because recording is
        restricted to inference forwards (``sticky_routing``'s wrapper) and
        that flag must be set when the graphs are captured.
        """
        from megatron.core.transformer.moe.router_replay import RouterReplayAction

        from nemo_rl.models.megatron import sticky_routing
        from nemo_rl.models.megatron.router_replay import (
            _router_replay_instances_for_model,
        )

        if sticky_routing.get_delta() is None:
            return
        if not getattr(context, "moe_enable_routing_replay", False):
            raise RuntimeError(
                "sticky routing: the inference context does not record routes "
                "(model config needs moe_enable_routing_replay)"
            )
        sticky_routing.install()
        insts = sorted(
            _router_replay_instances_for_model(self._gen_model()), key=lambda x: x[1]
        )
        if not insts:
            return
        rec = getattr(self, "_route_record", None)
        if rec is not None and [id(r) for r in rec["replays"]] == [
            id(r) for r, _ in insts
        ]:
            return
        # First build, or a rebuilt model whose routers are new instances:
        # (re-)arm them; the route log reads ``_route_record`` per forward.
        from megatron.core.utils import get_model_config

        cfg = get_model_config(self._gen_model())
        for replay, _ in insts:
            replay._nrl_record_inference_only = True
            replay.set_router_replay_action(RouterReplayAction.RECORD)
        self._route_record = {
            "replays": [replay for replay, _ in insts],
            "layers": [ln for _, ln in insts],
            "sequence_parallel": bool(getattr(cfg, "sequence_parallel", False)),
        }

    def install_draft_route_log(self) -> list[int] | None | bool:
        """Log the routes of tagged draft requests (prompt and drafted tokens).

        Generation workers, sticky routing. Wraps the controller's
        ``_dynamic_step_forward_logits``, which runs in Python for every engine
        forward (eager or CUDA-graph replay, legacy or async scheduling). Before
        the forward it selects the step's tagged tokens from the context's CPU
        bookkeeping, which at launch time describes exactly this forward (slot
        and absolute position of every token): numpy over the token axis, a
        draft-key lookup only for slots whose request is new
        (``routes.SlotTags``). After the forward it enqueues the gather and the
        asynchronous copy of those route rows to pinned host memory on the
        forward's stream, before the next forward can overwrite the buffer
        (``routes.RouteLog``): no host-device synchronization. Tags are the
        hidden-state log's draft keys (``HIDDEN_KEY_ATTR``).

        Pipeline stages log their own layers. Tensor parallelism: under
        sequence parallelism each TP rank routed its contiguous token shard and
        logs the tagged tokens of that shard (no collective; ``pop_draft_routes``
        returns its logged positions with sparse blocks and the driver merges the
        ranks' parts, ``routes.merge_tp_route_parts``).
        ``THUNDERSYNC_ROUTE_TP_SPLIT=0``: TP rank 0 logs after an all-gather of
        the shards (one collective of the step's [tokens, L, K] routes on steps
        with tagged tokens), as before. Without sequence parallelism every TP
        rank holds every route and TP rank 0 logs. Returns the logged MoE layer
        numbers when installed, None on ranks with nothing to log, False when
        unsupported.
        """
        from megatron.core import parallel_state as ps
        from thundersync_rl.routes import RouteLog

        engine = getattr(self, "dynamic_inference_engine", None)
        ctx = getattr(self, "inference_context", None)
        rec = getattr(self, "_route_record", None)
        if engine is None or ctx is None or rec is None:
            return False
        tp = ps.get_tensor_model_parallel_world_size()
        sp = rec["sequence_parallel"] and tp > 1
        split = sp and os.environ.get("THUNDERSYNC_ROUTE_TP_SPLIT", "1") != "0"
        logs = ps.get_tensor_model_parallel_rank() == 0 or split
        if not logs and not sp:
            return None
        if getattr(self, "_route_log_state", None) is None:
            # Unsplit: sequence-parallel TP ranks > 0 only take part in the
            # all-gather; their log never records.
            self._route_log = RouteLog()
            self._route_log_state = {"controller": None, "records": logs, "split": split}
        self._wrap_route_log(engine.controller, ctx)
        return list(rec["layers"]) if logs else None

    def _wrap_route_log(self, controller, ctx) -> None:
        """Wrap ``controller``'s forward with the route log (once per controller).

        Called by ``install_draft_route_log`` and again for every engine built
        later (``_engine_class``): a rebuilt engine has a new controller and
        context, and the log (with its open keys) carries over. The routers
        are read from ``_route_record`` at every forward, so re-armed
        instances are picked up.
        """
        from megatron.core import parallel_state as ps
        from thundersync_rl.engine_q import HIDDEN_KEY_ATTR
        from thundersync_rl.routes import SlotTags, step_route_selection

        state = self._route_log_state
        if state["controller"] is controller:
            return
        state["controller"] = controller
        log = self._route_log
        tp = ps.get_tensor_model_parallel_world_size()
        sp = self._route_record["sequence_parallel"] and tp > 1
        split = bool(state.get("split"))
        tp_rank = ps.get_tensor_model_parallel_rank()
        orig = controller._dynamic_step_forward_logits

        def lookup(rid):
            # Not cached by id: the engine may reuse request ids
            # (``SlotTags`` looks up again whenever a slot's request is new).
            entry = self.dynamic_inference_engine.requests.get(rid)
            if entry is None:
                return None
            return getattr(entry.record[-1].sampling_params, HIDDEN_KEY_ATTR, None)

        tags = SlotTags(lookup, log.key_id)
        records = state["records"]

        def select():
            n = int(ctx.active_token_count)
            if n <= 0:
                return None
            e = int(ctx.total_request_count)
            slot_kid = tags.refresh(
                ctx.request_ids[:e].numpy(),
                ctx.request_in_prefill_status_tensor[:e].numpy() != 0,
            )
            if not (slot_kid >= 0).any():
                return None
            sel = step_route_selection(
                ctx.token_to_request_idx[:n].numpy(),
                ctx.token_to_pos_ids[:n].numpy(),
                slot_kid,
            )
            return sel if sel[0].size else None

        def step_rows(n_rows: int):
            replays = self._route_record["replays"]
            if not sp:
                return _step_routes(replays, n_rows)
            chunk = int(ctx.padded_active_token_count) // tp
            local = _step_routes(replays, chunk)
            if local is None:
                return None
            full = local.new_empty((chunk * tp, *local.shape[1:]))
            torch.distributed.all_gather_into_tensor(
                full, local.contiguous(), group=ps.get_tensor_model_parallel_group()
            )
            return full

        orig_init = controller._dynamic_step_context_init
        dummy = [False]

        def context_init(*args, **kwargs):
            # An expert-parallel dummy step (MCore ``dummy_forward``, idle EP
            # ranks) sets only request and token counts: request ids, slots
            # and positions are still the last real step's, so its padding
            # rows would be logged as that step's tokens.
            dummy[0] = bool(kwargs.get("is_dummy_forward", False))
            return orig_init(*args, **kwargs)

        def shard_rows(sel):
            # This rank's sequence-parallel shard of the step's padded tokens: rows [r * chunk, (r + 1) * chunk)
            # of the gathered routes, i.e. its local route rows.
            chunk = int(ctx.padded_active_token_count) // tp
            rows, kid, pos = sel
            m = (rows >= tp_rank * chunk) & (rows < (tp_rank + 1) * chunk)
            if not m.any():
                return None, None
            local = _step_routes(self._route_record["replays"], chunk)
            if local is None:
                return None, None
            return local, (rows[m] - tp_rank * chunk, kid[m], pos[m])

        def forward_logits(*args, **kwargs):
            sel = None if dummy[0] else select()
            out = orig(*args, **kwargs)
            if sel is not None and split:
                rows, mine = shard_rows(sel)
                if rows is not None:
                    log.record(rows, *mine)
            elif sel is not None:
                rows = step_rows(int(sel[0][-1]) + 1)
                if rows is not None and records:
                    log.record(rows, *sel)
            return out

        controller._dynamic_step_context_init = context_init
        controller._dynamic_step_forward_logits = forward_logits

    # ---- Anchor horizon: sticky anchors applied by the engine (theta_k) ----

    def _install_anchor_apply(self, context) -> None:
        """Attach the per-token anchor buffer to the engine's routers.

        Sticky routing with ``policy.megatron_cfg.sticky_anchor_horizon``: a
        rollout drafted at theta_{k-1} whose anchors are complete below its
        horizon continues on the refit engine, which must route positions
        below the horizon with the sticky rule against those anchors.
        ``routes.EngineAnchors`` keeps one static [max_tokens, L_local, K]
        buffer filled per forward (``_wrap_anchor_apply``); every router reads
        its layer's column inside its CUDA graphs
        (``sticky_routing.engine_step_anchors``), so this runs before the
        engine constructor captures them. Every engine rank with MoE layers
        applies (pipeline stages: their layers; TP ranks: all, each routes
        its own tokens).
        """
        from megatron.core import parallel_state as ps
        from megatron.core.utils import get_model_config

        from nemo_rl.models.megatron import sticky_routing
        from thundersync_rl.routes import EngineAnchors

        if not sticky_routing.engine_anchor_apply():
            # Neither an anchor horizon nor engine-side verification.
            return
        rec = self._route_record
        cfg = get_model_config(self._gen_model())
        state = getattr(self, "_anchor_apply", None)
        if state is None or state["anchors"].buffer.shape[0] < int(context.max_tokens):
            anchors = EngineAnchors(
                int(context.max_tokens),
                len(rec["replays"]),
                int(cfg.moe_router_topk),
                torch.device("cuda", torch.cuda.current_device()),
                self._fetch_engine_anchors,
            )
            state = self._anchor_apply = {
                "anchors": anchors,
                "controller": None,
                "requests": {},  # anchor key -> engine request id
                "steps": 0,
            }
        rank = (
            ps.get_tensor_model_parallel_rank()
            if rec["sequence_parallel"] and ps.get_tensor_model_parallel_world_size() > 1
            else 0
        )
        buf = state["anchors"].buffer
        for j, replay in enumerate(rec["replays"]):
            setattr(replay, sticky_routing.APPLY_ATTR, (buf[:, j, :], rank))

    def register_engine_anchors(self, entries: dict[str, Any]) -> int:
        """{anchor key: RouteRefs} of rollouts the engine continues at theta_k.

        Generation workers (all of them: the coordinator picks the data-parallel
        rank that runs a request). Only handles are stored; the rank that
        admits a request fetches its own stage's blocks (``_fetch_engine_anchors``).
        Replaces the previous step's entries (their rollouts have finished).
        """
        self._engine_anchor_refs = dict(entries)
        return len(entries)

    def _fetch_engine_anchors(self, key: str):
        """Dense anchors [c, L_local, K] of one registered request (this stage)."""
        from thundersync_rl.routes import fetch_anchors

        refs = (getattr(self, "_engine_anchor_refs", None) or {}).get(key)
        if refs is None:
            # Only tagged requests reach here (SlotTags looks up ANCHOR_KEY_ATTR):
            # without its anchors the request would be routed plain, i.e. its p
            # is not the sticky p the verification / continuation assumes, and
            # nothing downstream notices (silent_bug_sweep.md).
            st = self._anchor_apply
            st["missing"] = st.get("missing", 0) + 1
            msg = (
                f"engine anchors: request tagged {key!r} has no registered anchors "
                "(never registered, or dropped while the request was still queued); "
                "it would be routed plain"
            )
            if os.environ.get("THUNDERSYNC_ALLOW_MISSING_ANCHORS") != "1":
                raise RuntimeError(msg + " (THUNDERSYNC_ALLOW_MISSING_ANCHORS=1 to count and continue)")
            if st["missing"] <= 5:
                print(f"[thundersync] WARNING {msg}", flush=True)
            return None
        t0 = time.perf_counter()
        rows = fetch_anchors(
            refs,
            self._route_record["layers"],
            int(self._anchor_apply["anchors"].buffer.shape[-1]),
            int(refs.length),
            ray.get,
        )
        st = self._anchor_apply
        st["fetch_s"] = st.get("fetch_s", 0.0) + time.perf_counter() - t0
        return rows

    def _wrap_anchor_apply(self, engine, controller, ctx) -> None:
        """Fill the anchor buffer before every engine forward (once per controller).

        Wraps ``_dynamic_step_forward_logits`` (runs in Python for every
        forward, eager or graph replay, legacy or async scheduling, and for
        expert-parallel dummy steps). Before the launch, from the context's CPU
        bookkeeping of this forward: anchor id per slot (``routes.SlotTags``:
        a lookup only for slots whose request is new; the first lookup of a
        tagged request loads its anchors into the pool), then per token
        (numpy) and ``EngineAnchors.fill``. A dummy step (MCore
        ``dummy_forward``: request ids and positions are the last real
        step's) makes every row self. Requests that left the engine release
        their pool pages every 64 forwards.
        """
        from thundersync_rl.engine_q import ANCHOR_KEY_ATTR
        from thundersync_rl.routes import SlotTags

        state = self._anchor_apply
        if state["controller"] is controller:
            return
        state["controller"] = controller
        anchors, active = state["anchors"], state["requests"]

        def lookup(rid):
            entry = engine.requests.get(rid)
            if entry is None:
                return None
            key = getattr(entry.record[-1].sampling_params, ANCHOR_KEY_ATTR, None)
            if key is not None:
                active[key] = rid
            return key

        tags = SlotTags(lookup, anchors.anchor_id)
        # Debug control for the end-to-end check: never apply (the engine then
        # samples plain routes on anchored prefixes; the mismatch check must
        # fail where the rule changes a route).
        skip = os.environ.get("THUNDERSYNC_DEBUG_NO_ANCHOR_APPLY") == "1"

        def sweep():
            for key in [k for k, rid in active.items() if rid not in engine.requests]:
                anchors.release(key)
                del active[key]

        orig_init = controller._dynamic_step_context_init
        orig = controller._dynamic_step_forward_logits
        dummy = [False]

        def context_init(*args, **kwargs):
            dummy[0] = bool(kwargs.get("is_dummy_forward", False))
            return orig_init(*args, **kwargs)

        def forward_logits(*args, **kwargs):
            if dummy[0] or skip:
                anchors.clear()
            else:
                n = int(ctx.active_token_count)
                e = int(ctx.total_request_count)
                if n > 0 and e > 0:
                    slot_aid = tags.refresh(
                        ctx.request_ids[:e].numpy(),
                        ctx.request_in_prefill_status_tensor[:e].numpy() != 0,
                    )
                    anchors.fill(
                        slot_aid[ctx.token_to_request_idx[:n].numpy()],
                        ctx.token_to_pos_ids[:n].numpy(),
                    )
                else:
                    anchors.clear()
            state["steps"] += 1
            if active and state["steps"] % 64 == 0:
                sweep()
            return orig(*args, **kwargs)

        controller._dynamic_step_context_init = context_init
        controller._dynamic_step_forward_logits = forward_logits

    def engine_anchor_stats(self) -> dict[str, float]:
        """Anchor-apply counters of this engine rank (0s when not installed)."""
        st = getattr(self, "_anchor_apply", None)
        if st is None:
            return {}
        a = st["anchors"]
        return {
            "applied_tokens": float(a.applied_tokens),
            "loaded_requests": float(a.loaded_requests),
            "pool_bytes": float(a.pool_bytes),
            "buffer_bytes": float(a.buffer.numel() * a.buffer.element_size()),
            "fetch_s": float(st.get("fetch_s", 0.0)),
            "missing": float(st.get("missing", 0)),
        }

    def install_tree_xssd(self, fresh_salt: int | None = None) -> dict | None:
        """Generation workers, colo mode with thundersync.tree_xssd (env THUNDERSYNC_TAIL_TREE = W:D:rows:1): install
        tree_spec on the engine with own-history tails (src bit1) for every row, keyed by the wide sampler (key id =
        engine request id, as one-token decode). Needs the wide sampler first. None without an engine."""
        import numpy as np

        engine = getattr(self, "dynamic_inference_engine", None)
        if engine is None:
            return None
        spec = os.environ.get("THUNDERSYNC_TAIL_TREE", "").strip()
        if not spec:
            raise RuntimeError("install_tree_xssd: THUNDERSYNC_TAIL_TREE not set (colo tree_xssd)")
        if getattr(self, "_wide_key_check", None) is None:
            raise RuntimeError("install_tree_xssd: install the wide sampler first")
        w, d, rows, _ref = (int(x) for x in spec.split(":"))
        from nemo_rl.models.generation.megatron import tree_kv, tree_spec
        from nemo_rl.models.megatron import mamba_bi_lookahead

        if mamba_bi_lookahead.tree_capacity() == 0 or tree_kv.capacity() == 0:
            raise RuntimeError("install_tree_xssd: node rows were not installed before the engine was built")
        key_ids = None
        if fresh_salt is not None:
            # stable keys (colo stable_keys): every fresh row is keyed by (fresh_salt, noise_seed | 2**62, position) in
            # the wide sampler; node rows draw with the same key id and salt, so the tree emits one-token decode's rows
            fk = _RequestKeys(engine)

            def key_ids(rids):
                keys = fk.lookup(np.asarray(rids, dtype=np.int64))
                if (keys < 0).any():
                    raise RuntimeError("install_tree_xssd: stable keys on, but a tree row has no noise_seed")
                return keys | _FRESH_KEY_BIT

            self._wide_salt = int(fresh_salt)  # node rows' salt (the tree sampler reads it); plain rows: per-row salt
        ts = tree_spec.install(
            engine, W=w, D=d, t_max=int(engine.context.max_requests), ref_cap=1, node_cap=rows,
            sample_rows=self._nl_tree_sample, key_check=self._wide_key_check, keyreg=None, key_ids=key_ids,
            scores_rows=self._nl_tree_scores, uniform_rows=self._nl_tree_uniform, own_history=True,
        )
        self._xssd_tree_attr = np.asarray([w, d, 2], dtype=np.int32).tobytes()  # src bit1: own history
        mtp = os.environ.get("THUNDERSYNC_TAIL_TREE_MTP", "").strip()
        if mtp:  # colo tree_xssd_mtp: the MTP chain as every tree row's first candidate
            self._nl_tree = ts
            self._nl_install_tree_mtp(mtp)
        return {"W": w, "D": d, "rows": ts.node_cap, "t_max": ts.t_max, "mtp": bool(mtp)}

    def _prepare_data_for_generation(self, data, greedy: bool = False):
        """Tag requests that continue an anchored rollout (anchor horizon).

        ``data["ts_anchor_key"][i]`` (optional): the request's key in
        ``register_engine_anchors``; it travels with the serialized sampling
        params to the engine rank that runs the request.
        """
        prompts, media, sps = super()._prepare_data_for_generation(data, greedy)
        if data is not None and "max_new_tokens" in data:
            # Per-request generation budget (a continuation after a rejection asks
            # for cap - kept prefix). The stock worker builds every request's
            # SamplingParams from the config's max_new_tokens, so without this a
            # non-keyed continuation decoded a whole fresh cap on top of its prefix
            # (pr-x8k-off: rows of 10.1k tokens at an 8k cap; redteam_1016 §9).
            for i, sp in enumerate(sps):
                sp.num_tokens_to_generate = int(data["max_new_tokens"][i])
        # nolanes: a hidden-log key on every SC request (nolanes_engine.py).
        self._nl_request_attrs(data, sps)
        # XS-SD on tree_spec (colo tree_xssd): every request is a tree tail with own-history candidates.
        tree_attr = self.__dict__.get("_xssd_tree_attr")
        if tree_attr is not None:
            from nemo_rl.models.generation.megatron.tree_spec import TREE_ATTR

            for sp in sps:
                sp.add_attributes({TREE_ATTR: tree_attr})
        keys = data.get("ts_anchor_key") if data is not None else None
        if keys:
            from thundersync_rl.engine_q import ANCHOR_KEY_ATTR

            for sp, key in zip(sps, keys):
                if key:
                    sp.add_attributes({ANCHOR_KEY_ATTR: key})
        return prompts, media, sps

    def pop_draft_routes(self, keys: list[str], close: bool = True) -> dict[str, tuple]:
        """{key: (layer numbers, lo, run_end, [(ref, lo, hi, bytes)])}.

        Routes this rank logged for the drafts since their last pop, as
        object-store blocks of dense routes (``routes.put_routes``; this rank
        owns the objects, the driver frees them); positions lo..run_end-1 are
        all logged. ``close``: drop rows logged for these keys later (a
        finished or aborted draft); False for a draft still running.
        """
        from thundersync_rl.routes import logged_mask, put_route_rows, put_routes

        log = getattr(self, "_route_log", None)
        out: dict[str, Any] = {}
        if log is not None and (getattr(self, "_route_log_state", None) or {}).get("split"):
            # This TP rank's token shard: sparse blocks of its logged positions + the logged mask, merged with the
            # other TP ranks' parts by the driver (``routes.merge_tp_route_parts``).
            layers = self._route_record["layers"]
            out = {
                k: (layers, lo, run_end, put_route_rows(pos, rows, ray.put), logged_mask(pos, lo))
                for k, (lo, run_end, pos, rows) in log.pop(keys, close=close, sparse=True).items()
            }
        elif log is not None:
            layers = self._route_record["layers"]
            out = {
                k: (layers, lo, run_end, put_routes(rows, ray.put, start=lo))
                for k, (lo, run_end, rows) in log.pop(keys, close=close).items()
            }
        nlog = getattr(self, "_nucleus_rlog", None)
        if nlog is not None:
            # Sticky nucleus ids (this rank logs the hidden rows): a second part.
            from thundersync_rl.routes import NUCLEUS_LAYER

            for k, (lo, run_end, rows) in nlog.pop(keys, close=close).items():
                part = ((NUCLEUS_LAYER,), lo, run_end, put_routes(rows, ray.put, start=lo))
                out[k] = [out[k], part] if k in out else [part]
        return out

    def drafts_done(self) -> list[bool]:
        """Finished flag per draft of the last ``start_drafts`` call (rank 0)."""
        return list(getattr(self, "_draft_done", None) or [])

    def install_draft_hidden_log(self, seal_blocks: bool = False) -> bool | None:
        """Log the engine's LM-head input for tagged draft requests.

        ``seal_blocks`` (object-store transport): a ``SealingHiddenLog`` that
        puts each full ``HIDDEN_BLOCK_ROWS`` block while drafting, so the
        deadline puts only partial tails (``THUNDERSYNC_HIDDEN_SEAL=0``
        disables it for A/B).

        Generation workers. A forward pre-hook on the engine model's output
        layer copies, per forward, the row of each tagged request's last input
        token (the hidden state its next token is sampled from) to host memory,
        keyed by (draft key, generated index). Returns True when installed,
        None on ranks that hold no rows to log (not the last pipeline stage,
        tensor-parallel rank > 0), False when unsupported (no engine, MTP
        speculative decoding inside the engine, sequence-parallel head input).

        Tensor parallelism: with ``materialize_only_last_token_logits`` the
        engine all-gathers the sequence-parallel hidden states and selects each
        request's last row before the output layer (with its sequence
        parallelism switched off), so the hook's input is the full set of rows,
        replicated on every TP rank. Each TP rank logs the drafts it owns
        (``hidden_owner``: crc32 of the draft key mod TP), so a draft's rows are
        all on one rank and the per-row work is split TP ways; the driver's
        hand-out asks every rank and merges by key (``THUNDERSYNC_HIDDEN_TP_SPLIT=0``:
        TP rank 0 logs every draft, as before). Without it the head input is
        sequence-sharded; that layout is unsupported (False).

        Pipeline parallelism: the output layer exists only on the engine's last
        stage, which computes the logits (the controller then broadcasts them
        to the other stages for sampling), so the hook is installed there; the
        dynamic engine's requests, with their sampling params, reach every
        model-parallel rank, so the last stage sees the tags. Other stages
        return None. Expert parallelism: each engine replica logs the drafts
        it runs; an expert-parallel dummy step has no requests and logs
        nothing.
        """
        from megatron.core import parallel_state as ps
        from thundersync_rl.engine_q import HIDDEN_KEY_ATTR, HiddenLog, hidden_owner

        from nemo_rl.models.megatron.fp16_gemm import head_matmul

        engine = getattr(self, "dynamic_inference_engine", None)
        ctx = getattr(self, "inference_context", None)
        if engine is None or ctx is None:
            return False
        if getattr(self, "_hidden_log", None) is not None:
            return True
        if not ps.is_pipeline_last_stage(ignore_virtual=True):
            return None
        tp_rank = ps.get_tensor_model_parallel_rank()
        try:
            tp = ps.get_tensor_model_parallel_world_size()
        except AssertionError:  # model parallelism not initialized (unit tests): one rank
            tp = 1
        split = tp > 1 and os.environ.get("THUNDERSYNC_HIDDEN_TP_SPLIT", "1") != "0"
        if tp_rank != 0 and not split:
            return None
        if getattr(ctx, "num_speculative_tokens", 0):
            return False
        materialized = bool(ctx.config.materialize_only_last_token_logits)
        layer = _unwrap(self._gen_model()).output_layer
        if getattr(layer, "sequence_parallel", False) and not materialized:
            return False
        if seal_blocks and os.environ.get("THUNDERSYNC_HIDDEN_SEAL", "1") != "0":
            from thundersync_rl.engine_q import SealingHiddenLog

            log = SealingHiddenLog()
        else:
            log = HiddenLog()

        # The LM-head input rows are copied into a static buffer by a pre-hook
        # installed on every rank when the engine was built, before it captured
        # its CUDA graphs (``_install_hidden_copy``), so the copy replays with
        # every graph. Here only the logging (sampler side) is added.
        copy = getattr(self, "_hidden_copy", None)
        if copy is None:
            return False
        static, filled, weight = copy["static"], copy["filled"], copy["weight"]
        dbg = {"kernel": 0, "rows": 0, "rejected": 0}
        debug = env_flag("THUNDERSYNC_HIDDEN_DEBUG")

        sampling = engine.controller._sampling
        inner = sampling.sample_kernel
        # Its own length cache: under the TP split this loop looks up only this rank's rows, and a cache shared
        # with the wide sampler (all rows, ``begin()`` every step) would drop the sampler's other entries each step.
        lens = self._hidden_lens = _RequestLengths()
        from thundersync_rl.routes import SlotTags

        def slot_key(rid):
            entry = self.dynamic_inference_engine.requests.get(rid)
            return None if entry is None else getattr(entry.record[-1].sampling_params, HIDDEN_KEY_ATTR, None)

        # Per context slot: 1 = a draft this rank logs, 0 = another TP rank's draft, -1 = untagged. Looked up only
        # when a slot's request changes (``SlotTags``), so the row loop below runs over this rank's rows only.
        owned_slots = SlotTags(slot_key, lambda key: int(not split or hidden_owner(key, tp) == tp_rank))

        def logging_kernel(logits, n, context, **kw):
            dbg["kernel"] += 1
            # With materialized last-token rows every forward (eager or graph
            # replay) refills the buffer; per-token rows are only copied on
            # eager forwards, flagged from Python.
            entries = []
            abspos: dict = {}
            if materialized or filled[0]:
                filled[0] = False
                engine_now = self.dynamic_inference_engine
                a, e = context.paused_request_count, context.total_request_count
                # Absolute positions only feed the sticky-nucleus log.
                want_abs = getattr(self, "_nucleus_rlog", None) is not None
                lens.begin()
                rids = context.request_ids[:e].numpy()
                pf = getattr(context, "request_in_prefill_status_tensor", None)
                # Slots in prefill may hold a new request under a reused id: looked up again.
                tag = owned_slots.refresh(rids, np.ones(e, dtype=bool) if pf is None else pf[:e].numpy() != 0)
                for row in (np.flatnonzero(tag[a:e] == 1)).tolist():
                    rid = int(rids[a + row])
                    # (latest record, its prompt, len, first prompt, len, key): cached prompt lengths
                    # (_RequestLengths), the same values as len() of the request's tensors.
                    c = lens.get(engine_now, rid)
                    if c is None or c[5] is None:
                        continue
                    # Index of the token this row samples, relative to the
                    # request's original prompt: after an eviction the engine
                    # resumes the request as a new record whose prompt includes
                    # the earlier tokens and whose generated tokens restart at 0.
                    t = c[2] + len(c[0].generated_tokens) - c[4]
                    entries.append((row, c[5], t))
                    if want_abs:
                        # Absolute input position of the row (sticky nucleus log).
                        abspos[(row, c[5])] = c[4] + t - 1
            # Probe reference read BEFORE the inner kernel: exact top-p masks the sampled rows in
            # place (-inf outside the nucleus), so a post-sampling read rejected nearly every
            # row under CHECKS=1 and every top-p draft was dropped (ns-topp / tp-*: vacuous).
            probe_ref = None
            # Read per call, not at install: sampled checks (THUNDERSYNC_CHECK_EVERY) switch it off on the
            # non-check steps (set_check_env). Captured once, it stayed on for the whole run, a GPU->CPU sync on
            # every decode step (long200: 1-4-row decode 9.2-10 ms vs 6.4-7.1 isolated).
            probe = os.environ.get("THUNDERSYNC_ENGINE_Q_CHECK") == "1"
            if probe and entries and logits is not None and torch.is_tensor(logits):
                rows_t = torch.tensor([r for r, _, _ in entries], device=static.device)
                lg = logits.reshape(-1, logits.shape[-1])
                lo = 0
                if weight.shape[0] < lg.shape[-1]:  # vocab-parallel shard (TP > 1)
                    from megatron.core import parallel_state as ps_

                    lo = ps_.get_tensor_model_parallel_rank() * weight.shape[0]
                cols = torch.linspace(0, weight.shape[0] - 1, 16, device=weight.device).long()
                probe_ref = (rows_t, cols, lg[rows_t][:, lo + cols].float())
            out = inner(logits, n, context, **kw)
            if not entries:
                return out
            # Source check (THUNDERSYNC_ENGINE_Q_CHECK=1 only: its .tolist() is
            # a GPU->CPU sync per decode step): a logged row must reproduce
            # 16 probe logits of the row the engine sampled from. Rows follow
            # the controller's async-schedule compaction (compact() below), so
            # the check should never reject; rejected rows are not recorded and
            # their drafts are truncated there (HiddenLog.pop returns the
            # longest logged prefix).
            if probe_ref is not None:
                # Probe columns of this rank's vocabulary shard (any TP, any
                # sampled token): a stale row disagrees on all of them.
                rows_t, cols, ref = probe_ref
                mine = head_matmul(static[rows_t], weight[cols])
                ok = ((mine - ref).abs().amax(-1) <= 1.0).tolist()
                if not all(ok):
                    dbg["rejected"] += len(entries) - sum(ok)
                    entries = [en for en, k_ in zip(entries, ok) if k_]
            if entries:
                dbg["rows"] += len(entries)
                rows_, keys_, ts_ = zip(*entries)
                hid = static[: max(rows_) + 1]
                if hasattr(log, "record_rows"):
                    log.record_rows(rows_, keys_, ts_, hid)  # vectorized (SealingHiddenLog)
                else:
                    log.record(entries, hid)
                # nolanes: the weight version of the forward these rows were sampled from (E1).
                self._nl_record_tags(keys_, ts_)
                nlog = getattr(self, "_nucleus_rlog", None)
                ids = getattr(self, "_nucleus_last_ids", None)
                if nlog is not None and ids is not None:
                    # Sticky nucleus: the drafter's nucleus ids of the same rows
                    # as the hidden rows (so a draft cut to its logged hidden
                    # rows has every nucleus row), keyed by absolute position.
                    from thundersync_rl.routes import encode_nucleus

                    self._nucleus_last_ids = None
                    nlog.record(
                        encode_nucleus(ids),
                        np.array([r for r, _, _ in entries], dtype=np.int64),
                        np.array([nlog.key_id(k) for _, k, _ in entries], dtype=np.int64),
                        np.array([abspos[(r, k)] for r, k, _ in entries], dtype=np.int64),
                    )
            if debug and dbg["kernel"] % 500 == 1:
                print(f"[hidden dbg] {dbg}", flush=True)
            return out

        sampling.sample_kernel = logging_kernel
        from nemo_rl.models.megatron import exact_nucleus

        if exact_nucleus.sticky():
            # Sticky nucleus: the nucleus sampler hands each sampled row's
            # drafter-nucleus ids to this log (RouteLog keyed by absolute
            # position; transported as NUCLEUS_LAYER route blocks).
            from thundersync_rl.routes import RouteLog

            self._nucleus_rlog = RouteLog()

            def ids_hook(ids, n, context):
                self._nucleus_last_ids = ids

            self._nucleus_ids_hook = ids_hook

        # Under async scheduling the controller launches the next forward
        # before finished requests are resolved, then compacts the surviving
        # rows and reorders the pending logits to match
        # (``_compact_async_sched_logits``). Apply the same permutation to the
        # hidden-copy buffer, on the same stream, so row i of the buffer stays
        # the hidden state of logits row i.
        controller = engine.controller
        orig_compact = controller._compact_async_sched_logits

        def compact(survivor_idxs, *args, **kwargs):
            n = survivor_idxs.numel()
            if n and not torch.equal(
                survivor_idxs, torch.arange(n, device=survivor_idxs.device)
            ):
                static[:n].copy_(static[survivor_idxs.to(static.device)])
            return orig_compact(survivor_idxs, *args, **kwargs)

        controller._compact_async_sched_logits = compact
        self._hidden_log = log
        return True

    def pop_draft_hidden(
        self, keys: list[str], lengths: list[int], as_refs: bool = False
    ) -> dict[str, Any]:
        """Logged hidden states [n, d] (CPU) of the drafts this engine ran.

        Drafts another engine ran, or with a missing row, are left out. Every
        key is closed: rows of later engine steps (an aborted stream finishing
        its step) are discarded.

        ``as_refs``: put each draft's rows into the Ray object store (one
        object per ``HIDDEN_BLOCK_ROWS`` rows) and return [(ObjectRef, rows,
        bytes)] per block instead of the tensor, so the caller only handles references
        and the verifying learner fetches the rows directly from this node,
        only if its lazy scan needs them. This worker owns the objects but
        keeps no reference: the driver frees them after verification.
        """
        log = getattr(self, "_hidden_log", None)
        if log is None:
            return {}
        return self._hand_out_hidden(log, keys, [0] * len(keys), lengths, as_refs, True)

    def take_draft_hidden(
        self,
        keys: list[str],
        starts: list[int],
        lengths: list[int],
        as_refs: bool = False,
    ) -> dict[str, Any]:
        """Hidden rows start..start+m-1 of running drafts (m <= length), keys left open.

        As ``pop_draft_hidden`` (same transport), for drafts verified in
        pieces while the engine keeps decoding them.
        """
        log = getattr(self, "_hidden_log", None)
        if log is None:
            return {}
        return self._hand_out_hidden(log, keys, starts, lengths, as_refs, False)

    def _hand_out_hidden(self, log, keys, starts, lengths, as_refs: bool, close: bool):
        """Shared body of ``pop_draft_hidden`` (close) / ``take_draft_hidden``.

        Also returns ``HIDDEN_STATS_KEY``: this call's wall time and the bytes it
        put (storage bytes, i.e. what the object store really holds), plus the
        sealing log's cumulative counters.
        """
        from thundersync_rl.engine_q import HIDDEN_STATS_KEY, SealingHiddenLog, put_hidden

        t0 = time.perf_counter()
        sealing = as_refs and isinstance(log, SealingHiddenLog)
        out: dict[str, Any] = {}
        put_bytes = 0
        tail0 = log.tail_put_bytes if sealing else 0
        for k, a, n in zip(keys, starts, lengths):
            if sealing:
                blocks = log.pop_refs(k, n) if close else log.take_refs(k, a, n)
                if blocks is not None and n > 0:
                    out[k] = blocks
                continue
            h = log.pop(k, n) if close else log.take(k, a, n)
            if h is not None and n > 0:
                if as_refs:
                    out[k] = put_hidden(h)
                    put_bytes += sum(b[2] for b in out[k])
                else:
                    out[k] = h
        if sealing:
            put_bytes = log.tail_put_bytes - tail0
        stats = {"call_s": time.perf_counter() - t0, "put_bytes": put_bytes}
        if sealing:
            # Background sealing since the previous hand-out (deltas).
            cum = {
                "sealed_bytes": log.sealed_bytes,
                "seal_put_s": log.seal_put_s,
                "resplits": log.resplits,
            }
            last = getattr(self, "_hidden_seal_last", {})
            stats.update({k: v - last.get(k, 0) for k, v in cum.items()})
            self._hidden_seal_last = cum
        out[HIDDEN_STATS_KEY] = stats
        return out

    def close_draft_logs(self, keys: list[str]) -> None:
        """Close the hidden-state and route logs of finished or aborted drafts."""
        hid = getattr(self, "_hidden_log", None)
        for k in keys if hid is not None else ():
            hid.close(k)
        log = getattr(self, "_route_log", None)
        if log is not None:
            log.pop(keys, close=True)

    # ---- Engine-side verification (thundersync.verify_on=engine; engine_verify.py) ----
    # The engine holding theta_k prefills drafted rows (one discarded sampled
    # token each), keeps their LM-head input rows and decides block
    # verification itself; p never leaves the engine rank that computed it.

    def _install_verify_capture(self, context) -> None:
        """Copy every engine forward's head-input rows before the last-token selection.

        ``policy.megatron_cfg.engine_verify``. With
        ``materialize_only_last_token_logits`` the model selects each request's
        last row (``context.last_token_logits``) before the output layer; a
        wrapper on that method first copies all [padded tokens, d] rows into a
        static buffer (head-input dtype, as ``_install_hidden_copy`` rounds
        them), so the copy is captured into the CUDA graphs and replays with
        them. ``_wrap_verify_capture`` then keeps the rows of tagged
        verification requests. Last pipeline stage only (the output layer);
        every TP rank (the rows are gathered over sequence parallelism there).
        Per rank: (max_tokens + 1024) x d x head-input bytes (Nano L8 fp32
        head: 17408 x 2688 x 4 B = 187 MB) plus one row copy per forward.
        """
        if not ((getattr(self, "cfg", None) or {}).get("megatron_cfg") or {}).get("engine_verify"):
            return
        model = _unwrap(self._gen_model())
        layer = getattr(model, "output_layer", None)
        if layer is None:
            return
        if not bool(context.config.materialize_only_last_token_logits):
            raise RuntimeError(
                "engine-side verification needs "
                "mcore_generation_config.materialize_only_last_token_logits=true"
            )
        try:
            from megatron.core.inference.utils import InferenceMode
        except ImportError:  # pragma: no cover
            InferenceMode = None
        from nemo_rl.models.megatron.fp16_gemm import head_input_dtype
        from thundersync_rl.engine_verify import CaptureStore

        weight = layer.weight
        if weight is None:
            weight = model.shared_embedding_or_output_weight()
        d = int(weight.shape[1])
        static = torch.empty(
            (int(context.max_tokens) + 1024, d),
            dtype=head_input_dtype(weight.dtype),
            device=weight.device,
        )
        overflow = [0]
        orig = context.last_token_logits

        def last_token_logits(x):
            if (
                x.dim() == 3
                and x.shape[-1] == d
                and (InferenceMode is None or InferenceMode.is_active())
            ):
                rows = x.reshape(-1, d)
                if rows.shape[0] <= static.shape[0]:
                    static[: rows.shape[0]].copy_(rows)
                else:
                    overflow[0] += 1
            return orig(x)

        context.last_token_logits = last_token_logits
        import threading

        self._verify_cap = {
            "static": static,
            "d": d,
            "store": CaptureStore(),
            "lock": threading.Lock(),
            "events": {},
            "overflow": overflow,
            "controller": None,
            "captured_rows": 0,
        }

    def _wrap_verify_capture(self, engine, controller, ctx) -> None:
        """Keep tagged verification requests' rows after every engine forward.

        Wraps ``_dynamic_step_forward_logits`` (every forward, eager or graph
        replay, as ``_wrap_anchor_apply``): before the launch the step's token
        layout (CPU bookkeeping) selects the tokens of tagged requests at
        positions plen-1 .. n-1; after it, those rows of the static buffer are
        copied into the request's ``CaptureStore`` entry on the same stream (the
        next forward is launched only after this returns). Python work per new
        request (``SlotTags``) and numpy per token; no device sync.
        """
        from thundersync_rl.engine_verify import (
            VERIFY_KEY_ATTR,
            VERIFY_PLEN_ATTR,
            capture_selection,
        )
        from thundersync_rl.routes import SlotTags

        cap = self._verify_cap
        if cap["controller"] is controller:
            return
        cap["controller"] = controller
        store, static, d = cap["store"], cap["static"], cap["d"]
        ids: dict[str, int] = {}
        names: list[str] = []
        bounds = {"lo": np.zeros(0, dtype=np.int64), "hi": np.zeros(0, dtype=np.int64)}

        def key_id(key: str) -> int:
            kid = ids.get(key)
            if kid is None:
                kid = ids[key] = len(names)
                names.append(key)
            return kid

        def lookup(rid):
            entry = engine.requests.get(rid)
            if entry is None:
                return None
            sp = entry.record[-1].sampling_params
            key = getattr(sp, VERIFY_KEY_ATTR, None)
            if key is None:
                return None
            plen = int(getattr(sp, VERIFY_PLEN_ATTR))
            n = len(entry.record[0].prompt_tokens)
            kid = key_id(key)
            if kid >= bounds["lo"].shape[0]:
                grow = max(256, kid + 1 - bounds["lo"].shape[0])
                bounds["lo"] = np.concatenate([bounds["lo"], np.zeros(grow, dtype=np.int64)])
                bounds["hi"] = np.concatenate([bounds["hi"], np.zeros(grow, dtype=np.int64)])
            bounds["lo"][kid], bounds["hi"][kid] = plen - 1, n
            with cap["lock"]:
                store.begin(key, plen - 1, n, d, static.dtype, static.device)
            return key

        tags = SlotTags(lookup, key_id)
        orig_init = controller._dynamic_step_context_init
        orig = controller._dynamic_step_forward_logits
        dummy = [False]

        def reset_keys() -> None:
            # Long runs: the key table grows by one entry per verification request (q recompute:
            # one per recomputed draft). Called only with no verification request in flight (run_verify_requests).
            ids.clear()
            names.clear()
            bounds["lo"] = np.zeros(0, dtype=np.int64)
            bounds["hi"] = np.zeros(0, dtype=np.int64)
            tags.rid = np.full(0, -1, dtype=np.int64)
            tags.kid = np.full(0, -1, dtype=np.int64)

        cap["reset_keys"] = reset_keys
        cap["key_table"] = ids

        def context_init(*args, **kwargs):
            dummy[0] = bool(kwargs.get("is_dummy_forward", False))
            return orig_init(*args, **kwargs)

        def forward_logits(*args, **kwargs):
            sel = None
            if not dummy[0]:
                n = int(ctx.active_token_count)
                e = int(ctx.total_request_count)
                if n > 0 and e > 0:
                    slot_kid = tags.refresh(
                        ctx.request_ids[:e].numpy(),
                        ctx.request_in_prefill_status_tensor[:e].numpy() != 0,
                    )
                    if (slot_kid >= 0).any():
                        sel, kids, pos = capture_selection(
                            ctx.token_to_request_idx[:n].numpy(),
                            ctx.token_to_pos_ids[:n].numpy(),
                            slot_kid,
                            bounds["lo"],
                            bounds["hi"],
                        )
            out = orig(*args, **kwargs)
            if sel is not None and sel.size:
                idx = torch.from_numpy(sel).pin_memory().to(static.device, non_blocking=True)
                rows = static.index_select(0, idx)
                with cap["lock"]:
                    for kid in np.unique(kids):
                        m = np.flatnonzero(kids == kid)
                        sub = rows.index_select(
                            0, torch.from_numpy(m).pin_memory().to(rows.device, non_blocking=True)
                        )
                        store.put(names[int(kid)], pos[m], sub)
                        ev = torch.cuda.Event()
                        ev.record()
                        cap["events"][names[int(kid)]] = ev
                    cap["captured_rows"] += int(sel.size)
            return out

        controller._dynamic_step_context_init = context_init
        controller._dynamic_step_forward_logits = forward_logits

        # Verification requests go to the front of the waiting queue: the
        # learner waits on them, while queued drafts (⊥-BV extras beyond the
        # request capacity) only lose drafting time. Admission order changes
        # timing only (drafts are cut by time; a verification is a function of
        # its row). A request still needs a free slot: engine verification needs
        # max_requests headroom of about one verification block per rank.
        orig_sched = engine.schedule_waiting_requests

        def is_verify(rid) -> bool:
            entry = engine.requests.get(rid)
            if entry is None:
                return False
            return getattr(entry.record[-1].sampling_params, VERIFY_KEY_ATTR, None) is not None

        def schedule_waiting_requests(*args, **kwargs):
            w = engine.waiting_request_ids
            if len(w) > 1 and int(getattr(ctx, "chunked_prefill_request_id", -1)) < 0:
                ids = list(w)
                ver = [r for r in ids if is_verify(r)]
                if ver and ids[: len(ver)] != ver:
                    vs = set(ver)
                    w.clear()
                    w.extend(ver + [r for r in ids if r not in vs])
            return orig_sched(*args, **kwargs)

        engine.schedule_waiting_requests = schedule_waiting_requests

    async def run_verify_requests(
        self,
        rows: list[Any],
        keys: list[str],
        prompt_lens: list[int],
        anchor_keys: Optional[list[Optional[str]]] = None,
    ) -> dict[str, float]:
        """Generation rank 0: prefill ``rows`` on this engine; return once all finished.

        Each row is a request generating one token (discarded), tagged with
        its verification key and draft prompt length (and, with sticky
        routing, its anchor key: the engine applies the draft's anchors in the
        prefill). The coordinator places requests on any data-parallel rank;
        that rank keeps the rows (``engine_verify_ready``).
        """
        self._assert_no_refit("run_verify_requests")
        from thundersync_rl.engine_q import ANCHOR_KEY_ATTR
        from thundersync_rl.engine_verify import VERIFY_KEY_ATTR, VERIFY_PLEN_ATTR

        if self.inference_client is None or self._inference_loop is None:
            raise RuntimeError("run_verify_requests: no inference engine on this worker")
        t0 = time.perf_counter()
        anchor_keys = anchor_keys or [None] * len(rows)
        cap = getattr(self, "_verify_cap", None)
        if cap is not None:
            cap["inflight"] = cap.get("inflight", 0) + len(rows)

        async def one(row, key, plen, akey):
            sp = self._build_sampling_params(greedy=False, stop_words=None)
            sp.num_tokens_to_generate = 1
            attrs = {VERIFY_KEY_ATTR: key, VERIFY_PLEN_ATTR: int(plen)}
            if akey:
                attrs[ANCHOR_KEY_ATTR] = akey
            sp.add_attributes(attrs)
            await self.inference_client.add_request([int(t) for t in row], sp)

        async def run():
            await asyncio.gather(
                *(one(r, k, p, a) for r, k, p, a in zip(rows, keys, prompt_lens, anchor_keys))
            )

        try:
            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(run(), self._client_loop()))
        finally:
            if cap is not None:
                cap["inflight"] -= len(rows)
                if not cap["inflight"] and len(cap.get("key_table") or ()) > 4096 and "reset_keys" in cap:
                    # Long runs: forget verification keys (one per q-recompute draft) once none is in flight.
                    cap["reset_keys"]()
        return {"requests": float(len(rows)), "s": time.perf_counter() - t0}

    # ---- q-row recompute on the anchor engine (hidden_transport=recompute, verify_on=learner) ----
    def engine_verify_take_refs(self, keys: list[str]) -> dict[str, Any]:
        """Captured rows of the listed keys this rank holds, put into the object store (no driver hop)."""
        import ray

        rows = self.engine_verify_take(keys)
        return {k: ray.put(v) for k, v in rows.items()}

    def set_q_engine(self, broker: Any) -> bool:
        """Learner ranks: the q-recompute broker of the anchor engine (q_proxy.py; hidden_transport=recompute)."""
        self._q_engine = broker
        return True

    def engine_verify_ready(self, keys: list[str]) -> list[str]:
        """The listed verification keys whose rows this rank holds completely."""
        cap = getattr(self, "_verify_cap", None)
        if cap is None:
            return []
        with cap["lock"]:
            return [k for k in keys if cap["store"].complete(k)]

    def _verify_pop(self, key: str) -> Optional[torch.Tensor]:
        cap = self._verify_cap
        with cap["lock"]:
            ev = cap["events"].pop(key, None)
            rows = cap["store"].pop(key)
        if ev is not None:
            ev.synchronize()
        return rows

    def engine_verify_take(self, keys: list[str]) -> dict[str, torch.Tensor]:
        """Captured rows [n - plen + 1, d] (CPU) of the listed keys this rank holds.

        q recompute (``hidden_transport="recompute"``): the drafting lane
        prefilled the drafts at the drafting weights; row j is the head input
        the drafter sampled token j from.
        """
        cap = getattr(self, "_verify_cap", None)
        if cap is None:
            return {}
        out = {}
        for k in keys:
            with cap["lock"]:
                ok = cap["store"].complete(k)
            if ok:
                rows = self._verify_pop(k)
                if rows is not None:
                    out[k] = rows.cpu()
        return out

    def engine_verify_drop(self, keys: list[str]) -> None:
        """Forget captured rows of these keys (a failed or abandoned call)."""
        cap = getattr(self, "_verify_cap", None)
        if cap is None:
            return
        with cap["lock"]:
            cap["store"].drop(keys)
            for k in keys:
                cap["events"].pop(k, None)

    def add_engine_anchors(self, entries: dict[str, Any]) -> int:
        """{anchor key: RouteRefs} the engine applies to tagged verification prefills.

        Added to (not replacing) the registered entries; ``drop_engine_anchors``
        removes them after the call. Only handles are stored; the rank that
        admits a request fetches its own stage's blocks.
        """
        refs = dict(getattr(self, "_engine_anchor_refs", None) or {})
        refs.update(entries)
        self._engine_anchor_refs = refs
        return len(refs)

    def drop_engine_anchors(self, keys: list[str]) -> None:
        refs = getattr(self, "_engine_anchor_refs", None) or {}
        for k in keys:
            refs.pop(k, None)

    def head_stash_cpu(self, version: int, after: Any = None) -> Optional[torch.Tensor]:
        """This rank's stashed LM-head shard of ``version`` on the host.

        None only on ranks without the head (pipeline stages before the last).
        ``after``: the stash call's ref, so Ray runs this export after it
        (asyncio actors do not keep submission order).
        """
        w = getattr(self, "_head_stash", {}).get(version)
        if w is None:
            if self._lm_head() is not None:
                raise RuntimeError(
                    f"head_stash_cpu: no LM-head stash of version {version} "
                    f"(have {sorted(getattr(self, '_head_stash', {}))})"
                )
            return None
        return w.detach().cpu()

    def import_head_stash(
        self, version: int, weight: Optional[torch.Tensor], keep: list[int], after: Any = None
    ) -> int:
        """Install another lane's stashed head shard as ``version`` (same TP/PP rank).

        Engine-side verification on lanes: the verifying lane needs the LM head
        of the version that drafted the cohort (q rows). Per rank: V/TP x d x
        2 B per kept version. ``after``: a ref this import must follow (Ray
        data dependency; unused).
        """
        head = self._lm_head()
        if head is None:
            return 0
        if weight is None:
            raise RuntimeError(f"import_head_stash: no LM-head shard of version {version}")
        _, live, _ = head
        if not hasattr(self, "_head_stash"):
            self._head_stash = {}
        self._head_stash[version] = weight.to(device=live.device, dtype=live.dtype)
        for v in list(self._head_stash):
            if v not in keep and v != version:
                del self._head_stash[v]
        return len(self._head_stash)

    def _engine_head_logprobs_fn(self, weight: torch.Tensor, vocab_limit: int):
        """(head GEMM fn [k, d] -> fp32 logits shard, head-input dtype) for ``weight``.

        ``fp16_gemm.head_matmul`` is the model's LM-head GEMM (fp16 operands /
        exact hi-lo fp32 head of the recipe), called directly rather than
        through the output layer: the engine's own pre-hooks fire on any
        output-layer call while the engine is active (process-wide flag).
        """
        from nemo_rl.models.megatron.fp16_gemm import head_input_dtype, head_matmul

        head = self._lm_head()
        scale = head[2] if head is not None else None
        dtype = head_input_dtype(weight.dtype)

        def fn(h: torch.Tensor) -> torch.Tensor:
            z = head_matmul(h.to(device=weight.device, dtype=dtype), weight)
            if scale is not None:
                z = scale(z)
            return z

        return fn

    async def engine_verify_decide(self, **kwargs) -> list[tuple[str, dict[str, Any]]]:
        """Decide block verification of the items whose rows this rank captured.

        Runs in a thread (the actor keeps serving the driver's draft and log
        calls meanwhile). See ``_engine_verify_decide``.
        """
        self._assert_no_refit("engine_verify_decide")
        dev = torch.cuda.current_device()
        if getattr(self, "_ev_stream", None) is None:
            # Own (non-blocking) stream: the decision's host syncs wait for its
            # own kernels only, not for the engine's decode steps queued on the
            # engine's stream.
            self._ev_stream = torch.cuda.Stream(device=dev)

        def run():
            torch.cuda.set_device(dev)
            with torch.cuda.stream(self._ev_stream):
                out = self._engine_verify_decide(**kwargs)
                self._ev_stream.synchronize()
            return out

        return await asyncio.to_thread(run)

    @torch.no_grad()
    def _engine_verify_decide(
        self,
        items: list[dict[str, Any]],
        seed: int,
        vocab_limit: int,
        q_lane: Optional[list[Any]] = None,
        chunk_rows: int = 512,
        check: bool = False,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Block verification on the engine: p from this rank's captured rows.

        ``items[i]``: {"key", "tokens" (draft), "logq" (engine log q of the
        draft tokens), "hidden" (the drafter's head inputs: tensor,
        ``HiddenRefs`` or ``HiddenRecompute``), "segments", optional
        "envelopes" (⊥-BV grid), "band", "plen", "row"}. Items whose rows this
        rank does not hold are skipped (another rank decides them).

        p: ``HeadLogprobRows`` over the captured rows with the live LM head
        (theta_k), ``chunk_rows`` rows per head GEMM. q rows: the stashed head
        of the drafting version over the drafter's head inputs; with
        ``HiddenRecompute`` the rows are rebuilt by a prefill on the drafting
        lane ``q_lane`` = [rank-0 actor, every worker actor] (two-pass
        decision, ``verify_rows_two_pass``). Every row has its own random
        stream (``row_seed(seed, key)``). The rule is ``verify_engine_q`` /
        ``verify_engine_q_grid``, as on the learner; results have the learner
        method's keys. ``check``: also "p_token_logprobs" (log p of every
        drafted token) and q rows rebuilt and compared with the engine's log q.

        Tensor parallel size 1 only (validated): at TP>1 the vocab gather
        would be a collective outside the engine's own step order.
        """
        from megatron.core import parallel_state as ps
        from thundersync_rl.engine_q import HiddenRows
        from thundersync_rl.engine_verify import (
            HeadLogprobRows,
            HiddenRecompute,
            NeedQ,
            decide_row,
            decide_whole_rows,
            verify_rows_two_pass,
        )

        if ps.get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError("engine-side verification at TP>1 is not built yet")
        cap = getattr(self, "_verify_cap", None)
        head = self._lm_head()
        if cap is None or head is None:
            return []
        t0 = time.perf_counter()
        live = head[1]
        p_head = self._engine_head_logprobs_fn(live, vocab_limit)
        mine: list[dict[str, Any]] = []
        prows: dict[str, torch.Tensor] = {}
        for it in sorted(items, key=lambda x: x["key"]):
            with cap["lock"]:
                ok = cap["store"].complete(it["key"])
            if not ok:
                continue
            rows = self._verify_pop(it["key"])
            if rows is None:
                continue
            prows[it["key"]] = rows
            mine.append(it)
        dev = live.device
        stash_fns: dict[int, Any] = {}

        def q_fn(version: int):
            fn = stash_fns.get(version)
            if fn is None:
                w = getattr(self, "_head_stash", {}).get(version)
                if w is None:
                    raise RuntimeError(
                        f"engine verify: no LM-head stash of drafting version {version} "
                        f"(have {sorted(getattr(self, '_head_stash', {}))})"
                    )
                fn = stash_fns[version] = self._engine_head_logprobs_fn(w, vocab_limit)
            return fn

        from thundersync_rl.engine_q import gathered_log_softmax, no_gather

        def q_rows_from(src, segments, g):
            version = [0] * g
            for v, a, b in segments:
                version[a:b] = [v] * (b - a)
            h = HiddenRows(src) if not isinstance(src, torch.Tensor) else None

            def q_rows(idx):
                idx_l = idx.tolist()
                out: list[Any] = [None] * len(idx_l)
                by_v: dict[int, list[int]] = {}
                for j, t in enumerate(idx_l):
                    by_v.setdefault(version[t], []).append(j)
                for v, js in by_v.items():
                    pos = [idx_l[j] for j in js]
                    hv = src[pos] if h is None else h(pos)
                    lp = gathered_log_softmax(q_fn(v)(hv), no_gather, vocab_limit)
                    for k, j in enumerate(js):
                        out[j] = lp[k]
                return torch.stack(out)

            return q_rows

        def own_q(it):
            src = it["hidden"]
            if isinstance(src, HiddenRecompute):

                def raise_need(idx):
                    raise NeedQ()

                return raise_need
            return q_rows_from(src, it["segments"], len(it["tokens"]))

        stats = {"q_rows": 0, "check_tok": 0, "check_exact": 0, "check_max": 0.0, "recompute_s": 0.0}
        # Phase 1 of every row needs log p of all its drafted tokens: one batched
        # head pass over every row (few large GEMMs instead of per-row launches).
        from thundersync_rl.engine_verify import batched_token_logprobs

        t_lp = time.perf_counter()
        # Every row's draft ids and log q in one host-to-device copy each (the
        # same values as one torch.tensor per row).
        lens = [len(it["tokens"]) for it in mine]
        keys_m = [it["key"] for it in mine]
        if mine:
            tok_all = torch.from_numpy(
                np.concatenate([np.asarray(it["tokens"], dtype=np.int64).reshape(-1) for it in mine])
            ).to(dev)
            lq_all = torch.from_numpy(
                np.concatenate([np.asarray(it["logq"], dtype=np.float32).reshape(-1) for it in mine])
            ).to(dev)
            drafts_t = dict(zip(keys_m, torch.split(tok_all, lens)))
            lq_t = dict(zip(keys_m, torch.split(lq_all, lens)))
        else:
            drafts_t, lq_t = {}, {}
        lpx_all = dict(
            zip(
                [it["key"] for it in mine],
                batched_token_logprobs(
                    [prows[it["key"]] for it in mine],
                    [drafts_t[it["key"]] for it in mine],
                    p_head,
                    vocab_limit,
                    int(os.environ.get("THUNDERSYNC_EV_CHUNK_ROWS", "2048")),
                ),
            )
        )
        t_lp = time.perf_counter() - t_lp

        def decide_one(it, q_rows):
            key = it["key"]
            g = len(it["tokens"])
            p_lp = HeadLogprobRows(prows[key], p_head, vocab_limit, chunk_rows)
            p_lp.cached_token_logprobs = lpx_all[key]
            assert prows[key].shape[0] == g + 1, (prows[key].shape, g)
            draft = drafts_t[key]
            lq = lq_t[key]
            q_rows = q_rows if q_rows is not None else own_q(it)
            env = it.get("envelopes")
            res, lpx, n_eval = decide_row(
                p_lp, draft, lq, q_rows, seed, key,
                tuple(env) if env is not None else None, it.get("band"),
            )
            stats["q_rows"] += n_eval
            if check:
                if lpx is None:
                    lpx = p_lp.token_logprobs(draft)
                res["p_token_logprobs"] = lpx.tolist()
                if g:
                    rec = q_rows(torch.arange(g)).gather(-1, draft.view(-1, 1))[:, 0]
                    dq = (rec - lq).abs()
                    stats["check_tok"] += g
                    stats["check_exact"] += int((dq == 0).sum())
                    stats["check_max"] = max(stats["check_max"], float(dq.max()))
            res["hidden_bytes_fetched"] = 0
            return res

        def recompute_q(need):
            t_rq = time.perf_counter()
            try:
                return _recompute_q(need)
            finally:
                stats["recompute_s"] += time.perf_counter() - t_rq

        def _recompute_q(need):
            if q_lane is None:
                raise RuntimeError("engine verify: q rows need recompute but no drafting lane given")
            rank0, workers = q_lane[0], q_lane[1]
            qkeys = [f"q|{it['key']}" for it in need]
            rows = [
                np.asarray(list(it["prompt"]) + list(it["tokens"]), dtype=np.int64) for it in need
            ]
            ray.get(
                rank0.run_verify_requests.remote(
                    rows, qkeys, [int(it["plen"]) for it in need]
                )
            )
            got: dict[str, torch.Tensor] = {}
            for r in ray.get([w.engine_verify_take.remote(qkeys) for w in workers]):
                got.update(r)
            out = {}
            for it, qk in zip(need, qkeys):
                h = got.get(qk)
                if h is None:
                    raise RuntimeError(f"engine verify: q recompute returned no rows for {qk}")
                g = len(it["tokens"])
                out[it["key"]] = q_rows_from(h[:g].to(dev), it["segments"], g)
            return out

        fast: dict[str, dict[str, Any]] = {}
        t_fast = 0.0
        rowwise = os.environ.get("THUNDERSYNC_EV_DECIDE") == "rowwise"
        if not check and not rowwise and mine:
            # Rows accepted whole (most of them) in one batched pass; the others
            # row by row below .
            t_fast = time.perf_counter()
            fast, _ = decide_whole_rows(
                [
                    {
                        "key": it["key"],
                        "hidden": prows[it["key"]],
                        "draft": drafts_t[it["key"]],
                        "lpx": lpx_all[it["key"]],
                        "lq": lq_t[it["key"]],
                        "envelopes": it.get("envelopes"),
                        "band": it.get("band"),
                    }
                    for it in mine
                ],
                p_head,
                vocab_limit,
                seed,
                int(os.environ.get("THUNDERSYNC_EV_CHUNK_ROWS", "2048")),
            )
            t_fast = time.perf_counter() - t_fast
            if fast and os.environ.get("THUNDERSYNC_EV_DECIDE_CHECK") == "1":
                # Debug: the batched decisions must equal the per-row ones bitwise.
                n_eq = 0
                for it in mine:
                    r = fast.get(it["key"])
                    if r is None:
                        continue
                    ref = dict(decide_one(it, None))
                    ref.pop("hidden_bytes_fetched", None)
                    if ref != r:
                        raise RuntimeError(
                            f"engine verify: batched decision != per-row decision for {it['key']}"
                        )
                    n_eq += 1
                print(
                    f"[ev decide check] rank={self.rank} batched={len(fast)} equal={n_eq} "
                    f"rows={len(mine)}",
                    flush=True,
                )
            for r in fast.values():
                r["hidden_bytes_fetched"] = 0
        rest = [it for it in mine if it["key"] not in fast]
        if check:
            # Every row's q is rebuilt and compared: recompute them all up front.
            recompute = [it for it in mine if isinstance(it["hidden"], HiddenRecompute)]
            qs = recompute_q(recompute) if recompute else {}
            results = [(it["key"], decide_one(it, qs.get(it["key"]))) for it in mine]
            n_recomputed = len(recompute)
        else:
            results, n_recomputed = verify_rows_two_pass(rest, decide_one, recompute_q)
            results = [(it["key"], fast[it["key"]]) for it in mine if it["key"] in fast] + results
        for _, r in results:
            r["engine_rank"] = int(self.rank)
        if results:
            results[0][1]["_engine_stats"] = {
                "decide_s": time.perf_counter() - t0,
                "decide_p_tokens_s": t_lp,
                "decide_batched_s": t_fast,
                "decide_batched_rows": float(len(fast)),
                "decide_recompute_s": stats["recompute_s"],
                "rows": float(len(results)),
                "q_rows": float(stats["q_rows"]),
                "q_recomputed_rows": float(n_recomputed),
                "captured_rows": float(cap["captured_rows"]),
                "capture_overflow": float(cap["overflow"][0]),
            }
        if check and stats["check_tok"]:
            print(
                f"[engine verify q check] rank={self.rank} tok={stats['check_tok']} "
                f"exact={stats['check_exact']} max|dlogq|={stats['check_max']:.3e}",
                flush=True,
            )
        return results

    def parallel_ranks(self) -> dict[str, int]:
        """(data, tensor, pipeline) parallel ranks of this worker and its stage role."""
        from megatron.core import parallel_state as ps

        return {
            "dp": int(ps.get_data_parallel_rank()),
            "tp": int(ps.get_tensor_model_parallel_rank()),
            "pp": int(ps.get_pipeline_model_parallel_rank()),
            "last": bool(
                ps.get_pipeline_model_parallel_world_size() == 1
                or ps.is_pipeline_last_stage(ignore_virtual=True)
            ),
        }

    def live_head_cpu(self) -> Optional[torch.Tensor]:
        """This rank's live LM-head shard on the host (None off the last stage)."""
        head = self._lm_head()
        return None if head is None else head[1].detach().cpu()

    async def engine_prefill_logprobs(self, **kwargs) -> dict[str, list[float]]:
        dev = torch.cuda.current_device()

        def run():
            torch.cuda.set_device(dev)
            return self._engine_prefill_logprobs(**kwargs)

        return await asyncio.to_thread(run)

    @torch.no_grad()
    def _engine_prefill_logprobs(
        self,
        items: list[dict[str, Any]],
        vocab_limit: int,
        head_shards: Optional[list[torch.Tensor]] = None,
        chunk_rows: int = 512,
    ) -> dict[str, list[float]]:
        """log p of each item's tokens from this rank's captured prefill rows.

        The premise check of engine-side verification
        (THUNDERSYNC_ENGINE_PREFILL_CHECK): no decision, only p.
        ``head_shards`` (tensor parallelism): every TP rank's live head shard in
        rank order; this rank (TP rank 0 of its replica) runs one head GEMM per
        shard with the shard's own shape and concatenates the logits, as the
        engine's gathered output does, so no collective runs outside the engine.
        Other TP ranks drop their rows and return {}.
        """
        from thundersync_rl.engine_verify import HeadLogprobRows

        from nemo_rl.models.megatron import exact_int as _xi

        from nemo_rl.models.megatron import exact_nucleus

        sticky_nuc = exact_nucleus.sticky()
        engine = getattr(self, "dynamic_inference_engine", None)
        exact_law = _xi.active() or bool(
            getattr(getattr(getattr(engine, "controller", None), "_sampling", None), "_exact_int_installed", False)
        )
        cap = getattr(self, "_verify_cap", None)
        head = self._lm_head()
        if cap is None or head is None:
            return {}
        live = head[1]
        if head_shards is None:
            p_head = self._engine_head_logprobs_fn(live, vocab_limit)
        else:
            head_shards = [ray.get(x) if isinstance(x, ray.ObjectRef) else x for x in head_shards]
            fns = [
                self._engine_head_logprobs_fn(w.to(device=live.device, dtype=live.dtype), vocab_limit)
                for w in head_shards
            ]

            def p_head(h):
                return torch.cat([f(h) for f in fns], -1)

        out = {}
        for it in items:
            with cap["lock"]:
                ok = cap["store"].complete(it["key"])
            if not ok:
                continue
            rows = self._verify_pop(it["key"])
            if rows is None:
                continue
            p_lp = HeadLogprobRows(rows, p_head, vocab_limit, chunk_rows)
            toks = torch.tensor(it["tokens"], dtype=torch.long, device=live.device)
            if sticky_nuc and it.get("plen") is not None:
                # Sticky nucleus (ts-nano-scale): the target policy is F(theta_k; A) with A the
                # drafter's nucleus ids on the row's route refs (as the learner's verification,
                # fetch_nucleus); without them the engine's plain-nucleus p differs wherever the
                # sticky set was kept, and the check could not run under top-p (ns-topp).
                from thundersync_rl.engine_q import gathered_log_softmax, no_gather
                from thundersync_rl.routes import fetch_nucleus

                refs = (getattr(self, "_engine_anchor_refs", None) or {}).get(it["key"])
                anc = fetch_nucleus(
                    refs, int(toks.numel()), exact_nucleus.anchor_cap(), ray.get, lo=int(it["plen"]) - 1
                )
                vals = []
                for a in range(0, int(toks.numel()), chunk_rows):
                    b = min(int(toks.numel()), a + chunk_rows)
                    lp = gathered_log_softmax(
                        p_head(rows[a:b]), no_gather, vocab_limit, anc[a:b].to(live.device)
                    )
                    vals.append(lp.gather(-1, toks[a:b].view(-1, 1))[:, 0])
                out[it["key"]] = (torch.cat(vals) if vals else toks.new_zeros(0).float()).tolist()
                continue
            if exact_law:
                # Exact integer law: the logged / trained value lambda.
                from thundersync_rl.exact_bv import p_token_lambda

                out[it["key"]] = p_token_lambda(p_lp, toks).tolist()
            else:
                out[it["key"]] = p_lp.token_logprobs(toks).tolist()
        return out

    def _lm_head(self):
        """(output layer, live head weight, logit scaling or None) of the training model.

        None off the last pipeline stage: Megatron builds the output layer only
        there (under tied embeddings the last stage keeps its own copy of the
        embedding weight as the output weight, synchronized by the embedding
        group, which ``shared_embedding_or_output_weight`` returns).
        """
        # Engine workers: the model the inference engine runs.
        gpt = _unwrap(
            self._gen_model() if getattr(self, "is_refit_destination", False) else self.model
        )
        layer = getattr(gpt, "output_layer", None)
        if layer is None or not getattr(gpt, "post_process", True):
            return None
        weight = (
            gpt.shared_embedding_or_output_weight()
            if getattr(gpt, "share_embeddings_and_output_weights", False)
            else layer.weight
        )
        return layer, weight, getattr(gpt, "_scale_logits", None)

    @torch.no_grad()
    def stash_head(self, version: int, keep: list[int], lookahead: bool = False) -> int:
        """Stash only the LM-head weight as ``version``.

        The live weight is this rank's vocab-parallel shard (V/TP x d; under
        tied embeddings the embedding's shard, same vocabulary split), so each
        tensor-parallel rank keeps 1/TP of the head per pending version.
        Pipeline parallelism: only the last stage holds the head and verifies;
        other stages stash nothing and return 0.
        """
        self._assert_no_refit("stash_head")
        head = self._lm_head()
        if head is None:
            return 0
        _, weight, _ = head
        if lookahead:
            # Version ``version`` drafted with theta~ (lookahead drafts): q rows are rebuilt with theta~'s head.
            weight = self._la_extrapolated(weight)
        if not hasattr(self, "_head_stash"):
            self._head_stash: dict[int, torch.Tensor] = {}
        self._head_stash[version] = _restash(self._head_stash, version, keep, [weight])[0]
        self._log_mem(
            f"stash_head v={version} n={len(self._head_stash)} "
            f"bytes={self.head_stash_bytes()}"
        )
        return len(self._head_stash)

    def head_stash_bytes(self) -> int:
        """Bytes of the LM-head stashes on this GPU (all pending versions)."""
        return sum(
            w.numel() * w.element_size()
            for w in getattr(self, "_head_stash", {}).values()
        )

    def _tp_vocab_gather(self):
        """Rank-order all-gather of vocab shards over the tensor-parallel group.

        The concatenation the engine's gathered output layer and
        batch-invariant training (``_GatheredVocabLogprob``) perform; identity
        at TP=1.
        """
        from megatron.core import parallel_state as ps
        from thundersync_rl.engine_q import TPVocabGather, no_gather

        if ps.get_tensor_model_parallel_world_size() == 1:
            return no_gather
        return TPVocabGather(ps.get_tensor_model_parallel_group())

    def _head_local(self, hidden: torch.Tensor, version: int) -> torch.Tensor:
        """This rank's vocab shard of the q logits [n, V/TP] of ``hidden`` [n, d] under head ``version``: the
        output layer with the stashed weight, sequence parallelism off, local logits, logit scaling (the rows
        ``_head_logprobs`` gathers). No collective."""
        layer, _, scale = self._lm_head()
        from nemo_rl.models.megatron.fp16_gemm import head_input_dtype

        from thundersync_rl.block_verification_fused import _h2d

        w = self._head_stash[version]
        # Host rows: cast where ``Tensor.to(device, dtype)`` casts a host-to-device copy (on the host), then a
        # pinned asynchronous copy (a pageable one waits for the stream). Same bytes on the device.
        from thundersync_rl.block_verification_fused import devscan_enabled

        if devscan_enabled():
            h = _h2d(hidden.to(dtype=head_input_dtype(w.dtype)), w.device)
        else:
            h = hidden.to(device=w.device, dtype=head_input_dtype(w.dtype))
        sp = getattr(layer, "sequence_parallel", False)
        if sp:
            # The rows are full (not sequence-sharded), as in the engine.
            layer.sequence_parallel = False
        try:
            z, _ = layer(h.unsqueeze(1), weight=w, runtime_gather_output=False)
        finally:
            if sp:
                layer.sequence_parallel = True
        if scale is not None:
            z = scale(z)
        return z[:, 0]

    def _ds2_grid(self, p_lp, h, draft, lq, segments, vocab_limit, generator, envelopes):
        """⊥-BV grid scan of one draft on this rank's vocabulary shard (``ds2.grid_scan_sharded``): returns
        (outcomes per envelope, log p of the drafted tokens [g], positions evaluated), as ``verify_engine_q_grid``."""
        import numpy as np

        from nemo_rl.models.megatron import vocab_lse
        from thundersync_rl import ds2

        from thundersync_rl.prof import timed

        group = p_lp.gather.group
        width = int(p_lp.local.shape[-1])
        offset = torch.distributed.get_rank(group) * width
        with timed("ds2/rows"):
            rows = ds2.ShardedRows(p_lp.local, vocab_limit, group, offset)
            dev = rows.zp.device
            lpx = rows.token_logprobs(draft.to(dev))
            lqx = lq.to(device=dev, dtype=torch.float32)
        vers = ds2.version_array(segments, draft.numel())
        # THUNDERSYNC_HIDDEN_DEVICE_GATHER (default off): q rows' hidden states gathered on the device (same bytes).
        dev_gather = env_flag("THUNDERSYNC_HIDDEN_DEVICE_GATHER") and hasattr(h, "rows_at_device")
        # tripwire under the engine q check (CHECKS): every 32nd DS2 row compares its first chunk with the host path
        n_ds2 = self.__dict__.get("_dg_rows", 0) + 1
        self._dg_rows = n_ds2
        dg_check = dev_gather and os.environ.get("THUNDERSYNC_ENGINE_Q_CHECK") == "1" and n_ds2 % 32 == 1
        dg_checked: list = []

        def q_local(idx: torch.Tensor):
            pos = idx.numpy().astype(np.int64)
            vs = vers[pos]
            z = None
            for v in np.unique(vs):
                js = np.nonzero(vs == v)[0]
                if dev_gather:
                    with timed("ds2/q_local/fetch"):
                        rows_v = h.rows_at_device(pos[js], dev)
                    if dg_check and not dg_checked:
                        # CHECKS tripwire: one chunk per checked row == the host path's rows, bitwise
                        if not torch.equal(rows_v.cpu(), h.rows_at(pos[js])):
                            raise RuntimeError("THUNDERSYNC_HIDDEN_DEVICE_GATHER: device rows differ from the host gather")
                        dg_checked.append(1)
                        self._dg_checked_row = 1  # counter spec/ds2_dg_checked (comparisons that ran)
                else:
                    with timed("ds2/q_local/fetch"):
                        rows_v = h.rows_at(pos[js])
                with timed("ds2/q_local/head"):
                    zv = self._head_local(rows_v, int(v))
                if js.size == pos.size:
                    z = zv
                    break
                if z is None:
                    z = zv.new_empty((pos.size,) + tuple(zv.shape[1:]))
                z.index_copy_(0, torch.from_numpy(js).to(zv.device), zv)
            z = z.contiguous()
            with timed("ds2/q_local/lse"):
                return z, vocab_lse.lse_sharded(z, offset, vocab_limit, group)

        vw_check = env_flag("THUNDERSYNC_VWALL_CHECK")  # dedup is unconditional: every DS2 row has a reference
        g_state = generator.get_state() if vw_check else None
        out, n_eval = ds2.grid_scan_sharded(rows, lpx, lqx, q_local, generator, envelopes)
        if vw_check:
            # Gate check (THUNDERSYNC_VWALL_CHECK=1): the same row on the old path (per-envelope emissions, host-gathered
            # q rows: no device gather) from the same generator state must give the same outcomes at
            # every envelope, the same positions evaluated and the same final generator state, bitwise. Every TP rank
            # runs it (the same collectives); a mismatch raises after counting.
            def q_local_ref(idx: torch.Tensor):
                pos = idx.numpy().astype(np.int64)
                vs = vers[pos]
                z = None
                for v in np.unique(vs):
                    js = np.nonzero(vs == v)[0]
                    zv = self._head_local(h.rows_at(pos[js]), int(v))
                    if js.size == pos.size:
                        z = zv
                        break
                    if z is None:
                        z = zv.new_empty((pos.size,) + tuple(zv.shape[1:]))
                    z.index_copy_(0, torch.from_numpy(js).to(zv.device), zv)
                z = z.contiguous()
                return z, vocab_lse.lse_sharded(z, offset, vocab_limit, group)

            same = ds2.vwall_check(rows, lpx, lqx, q_local_ref, g_state, envelopes, out, n_eval, generator.get_state())
            self._vw_checked_row = 1
            if not same:
                self._vw_mismatch_row = 1
                raise RuntimeError(
                    f"THUNDERSYNC_VWALL_CHECK: DS2 row differs from the old path: new={out} n={n_eval}"
                )
        return out, lpx, n_eval

    def _head_logprobs(
        self, hidden: torch.Tensor, version: int, vocab_limit: int
    ) -> torch.Tensor:
        """Q log-prob rows [n, V] (fp32) of ``hidden`` [n, d] under head ``version``.

        The training model's own output layer with the stashed weight (this
        rank's vocab shard at TP>1), run as the engine runs it on its gathered
        last-token rows: sequence parallelism off, local logits, logit scaling,
        then the shards all-gathered in rank order and one fp32 log-softmax
        (``gathered_log_softmax``). Under batch invariance the GEMM gives each
        row's logits independently of the row count, so a row equals the
        engine's row for the same hidden state. A collective at TP>1: every
        rank of the group calls it with the same rows (the lazy scan's
        decisions are identical on all of them).
        """
        from thundersync_rl.engine_q import gathered_log_softmax

        return gathered_log_softmax(
            self._head_local(hidden, version), self._tp_vocab_gather(), vocab_limit
        )

    @torch.no_grad()
    @_model_in_eval
    @_lg_feature("verify_block")
    @_debug_verify_no_bi
    def verify_drafts_block_engine(
        self,
        rows: list[torch.Tensor],
        prompt_lens: list[int],
        segments: list[list[tuple[int, int, int]]],
        engine_logq: list[list[float]],
        hidden: list[Any],
        vocab_limit: int,
        seed: int,
        keys: list[str],
        batch_tokens: int = 0,
        anchors: list[Any] | None = None,
        null_envelopes: list[list[float]] | None = None,
        row_ranks: list[int] | None = None,
        band: float | None = None,
        prefix_keys: list[str | None] | None = None,
        prefix_release: list[str] | None = None,
        prefix_gpu_gb: float = 8.0,
        row_seed_base: int | None = None,
        walks: list[dict[str, Any] | None] | None = None,
        walk_scans: list[dict[str, Any] | None] | None = None,
        prefix_pins: list[int | None] | None = None,
        return_token_logprobs: bool = False,
        nucleus_anchors: list[Any] | None = None,
        prefix_trim: list[tuple[str, int]] | None = None,
        prefix_capture: bool = False,
        row_seeds: list[int] | None = None,
        reply_to: tuple[Any, str] | None = None,
        **_: Any,
    ) -> list[tuple[int, dict[str, Any]]] | None:
        """Block verification with q from the engine and p from the training model.

        ``reply_to`` = (mailbox actor, key) (posted calls, ``spec.verify_post``): the results are ``ray.put`` and
        their ref posted to the mailbox the moment the call ends, and the call returns None. This actor is an async
        Ray actor, and the reply of a synchronous method can wait behind the bodies of later queued calls (measured:
        a posted chunk's results reached the driver only after the last posted call, 9 s late); the mailbox path does
        not depend on this actor's event loop.

        Exact top-p (``exact_nucleus.py``): p and q rows are the nucleus-masked
        log-softmax rows (the policy the engine samples from); a drafted token
        outside the target nucleus has log p = -inf and is rejected with
        probability 1; the residual max(b p - q, 0) and the bonus row live on
        the target nucleus. ``nucleus_anchors[i]`` (sticky nucleus): [g, C]
        int32 drafter nucleus ids of row i's drafted positions (-1 padded; an
        all -1 row is "self"); p row j then follows the sticky rule with
        anchor j (the bonus row g is "self").

        ``row_seed_base`` (verify-in-train fallback, ``vit.py``): row i draws its
        uniforms from its own generator seeded ``vit.row_seed(row_seed_base,
        keys[i])`` instead of the call's per-rank stream, so its decision is
        the one the training forward already made with the same stream.

        ``row_seeds`` (THUNDERSYNC_ROW_SEEDS, verify0 ⊥-BV grid rows): row i draws
        its uniforms from its own generator seeded ``row_seeds[i]`` (the driver's
        per-row seed: step + row identity), so its decision does not depend on the
        call, the rank or its position in either.

        Carried-walk continuation pieces (``carried_walk.py``):
        ``walks[i]`` = {"log_b", "floor", "n", "final", "rule"}: row i's draft is a
        piece of a carried block; the walk advances and a decision is taken only
        when ``rule`` says so (result "walk", "decision", "token_logprobs"; with
        "reject" also "accepted"/"next"/"logprobs" relative to the piece).
        ``walk_scans[i]`` = {"log_ratio", "log_b0", "pos0", "floor", "stop"}: the
        rejection scan of a carried block below its deciding piece; row i's draft is
        block positions [pos0, stop). ``prefix_pins[i]``: keep the cached boundary
        at or below this row position (the block's floor) for such a scan.

        ``return_token_logprobs`` (THUNDERSYNC_ENGINE_VERIFY_CHECK): every result
        also carries "p_token_logprobs", log p of every drafted token.

        Learner prefix-state cache (``prefix_keys``): row i is
        cached under ``prefix_keys[i]`` (None: not cached) and forwarded only
        from its largest cached chunk boundary <= prompt_lens[i] - 1 whose
        tokens match; its p logits are bitwise those of the whole-row forward
        (``prefix_state_cache.py``). ``prefix_release``: keys to drop first.
        The driver decides membership (the same lists reach every rank of a
        replica); ``prefix_gpu_gb`` bounds this rank's device residency
        (colder rows wait in pinned host memory). Each result then carries
        "prefix_start" (positions not forwarded) and "prefix_from_v0" (its start
        came from a verify0 capture); the rank's first result also carries
        "prefix_cache_bytes" [device, host] of this rank's cache.
        ``prefix_capture`` (verify0 capture): the keyed rows are this
        step's first verification of whole drafts; their state is kept from the
        prompt's chunk boundary on (any continuation starts at or after it).
        ``prefix_trim`` [(key, bound)]: verify0 captures the driver has decided,
        cut to the largest boundary <= bound (memory only; applied before this
        call's rows, like ``prefix_release``).

        ``row_ranks[i]``: the data-parallel rank that verifies row i (the
        driver balances tokens, ``speculative.lpt_owners``); default: a stable
        hash of the key.

        Pooled ⊥-BV (``null_bv.py``): with ``null_envelopes`` row i is
        verified at each envelope M of ``null_envelopes[i]`` (the ones its pool
        index can be decided at, ``row_envelopes``) with one set of random draws
        (``verify_engine_q_grid``) and reports ``"null_grid"`` = {M: (tau, y,
        log p(y) or None)} plus ``"token_logprobs"`` (log p of the drafted
        tokens up to the largest tau) and ``"draft_logq"`` (the same span);
        the driver decides which M each row is taken at (``decide_pools``).

        ``band``: banded importance sampling (``block_verification``); every
        output row then reports ``"band_log_weight"`` (log w = S_T, the row's
        loss weight) and ``"band_exited"`` (its prefix left the band).

        ``anchors[i]`` (sticky MoE routing): the drafter's routes of row i,
        [positions, L, K] int16 (-1: self); p is then F(theta; anchors), the
        sticky policy the rollouts are sampled from.

        Requires batch-invariant kernels (the engine's decode is the training
        forward) and the same parallel layout as the engine (TP, PP, EP; batch
        invariance enforces it). Per draft i:
        p = log-softmax of the training model's forward of ``rows[i]`` at the
        current weights (a training-shaped forward, as
        ``verify_drafts_block_fused`` runs it, without autograd; at TP>1 its
        vocab-parallel logits are all-gathered in rank order before the
        log-softmax, as ``_GatheredVocabLogprob`` does); log q(x_j) =
        ``engine_logq[i][j]``; q rows only where the lazy scan needs them, from
        ``hidden[i]`` (the engine's final hidden states: a [g, d] tensor, or
        ``HiddenRefs`` object-store block handles) and the LM-head stash
        of the version that drafted position j (``segments``). With handles,
        a block is fetched (``ray.get``, directly from the generation
        worker's node) only when the lazy scan first needs a q row in it:
        phase 1 (p, b, the whole-draft test) reads only log q of the drafted
        tokens, so a fully accepted draft fetches nothing.
        No scorer copy and no whole-model stash. Returns the same per-row dicts
        as ``verify_drafts_block_stashed``: "logprobs" are training's own
        log-probs of the kept tokens and "draft_logq" the engine's, so the
        behavior-log-prob correction is exactly zero; plus
        "hidden_bytes_fetched" (bytes this draft pulled from the object store).

        Tensor parallelism: every TP rank of a replica verifies the same rows
        (``_my_rows`` hashes by data-parallel rank) with the same generator
        seed and bitwise the same gathered p and q rows, so the lazy scans
        take the same decisions and call the row gathers in lockstep.

        Pipeline parallelism: every stage of a replica runs the same rows'
        forwards through the forward-only pipeline schedule
        (``_training_forward``); only the last stage holds logits (and the LM
        head and its stashes), so it alone runs the lazy scans and returns the
        results. Other stages return [] (the driver merges all ranks' rows).

        Expert parallelism: each training forward runs all-to-alls over the
        expert-parallel group, which spans data-parallel replicas with
        different row counts. The group agrees on the largest count
        (``_expert_parallel_lockstep`` with empty version plans: q needs no
        forward here) and short ranks pad with one-row dummy forwards
        (``_dummy_training_forward``, output discarded). The q rows (LM head
        only) and the lazy scan's vocab gathers involve no experts and only
        the tensor-parallel group, i.e. ranks of one replica that verify the
        same rows in the same order, so they need no lockstep across replicas.

        ``THUNDERSYNC_ENGINE_Q_CHECK=1``: also rebuild every q row of each draft
        and print how many drafted tokens' log q match the engine bitwise and
        the largest difference (expected: all, 0); with anchors, also the
        sticky rule's counters over these forwards (``[sticky check]``: token
        layers, fraction that kept the anchor, fraction whose experts differ
        from plain top-k).
        """
        _call_log(self, "verify_drafts_block_engine")
        from megatron.core import parallel_state as ps
        from thundersync_rl.block_verification import banded_log_weight, nonfinite_context
        from nemo_rl.models.megatron.deferred_head import (
            DeferredHeadLogits,
            verification_logits,
        )
        from thundersync_rl.engine_q import (
            DeferredLogprobRows,
            GatheredLogprobRows,
            HiddenPrefetch,
            HiddenRefs,
            HiddenRows,
            hidden_num_rows,
            verify_engine_q,
            verify_engine_q_grid,
        )

        from nemo_rl.models.megatron import exact_nucleus, sticky_rounding, sticky_routing
        from nemo_rl.models.megatron import exact_int as xi_law
        from thundersync_rl import ds2 as _ds2
        from thundersync_rl import exact_bv
        from thundersync_rl.block_verification_fused import _d2h, _h2d, devscan_enabled
        from thundersync_rl.prof import pop as prof_pop
        from thundersync_rl.prof import timed

        # Exact integer law (engine_sampler=exact_int): exact verification.
        exact_law = xi_law.active()
        if exact_law:
            n_calls = self.__dict__.get("_exact_bv_calls", 0) + 1
            self._exact_bv_calls = n_calls
            if n_calls % 20 == 1:
                print(f"[exact_bv] rank={getattr(self, 'rank', '?')} calls={n_calls} {exact_bv.counts()}", flush=True)
        prof_pop()  # phases of this call only
        torch_prof = self._torch_profiler(null_envelopes is not None)

        if not self.cfg["megatron_cfg"].get("batch_invariant_mode"):
            raise RuntimeError(
                "verify_source='engine' requires policy.megatron_cfg.batch_invariant_mode"
            )
        self._check_dropless_moe()
        tp = ps.get_tensor_model_parallel_world_size()
        last_stage = (
            ps.get_pipeline_model_parallel_world_size() == 1
            or ps.is_pipeline_last_stage(ignore_virtual=True)
        )
        gather = self._tp_vocab_gather() if last_stage else None
        self._prof = {}
        t_all = time.perf_counter()
        # Call-level timing of this rank (logging only; host seconds, no device sync): attached to the rank's first
        # result as "vprof_s" and folded into the driver's vprof/ stats (_verify_block_results).
        vp: dict[str, float] = {"t_in": time.time(), "rank": float(getattr(self, "rank", -1))}
        gen = None  # created on the logits' device (last stage only)
        check = os.environ.get("THUNDERSYNC_ENGINE_Q_CHECK") == "1"
        law_on = os.environ.get("THUNDERSYNC_LAW_CHECK") == "1"
        # THUNDERSYNC_EMISSION_LAW_EVERY=n (production subset): the emission-law PIT
        # only for rows whose key hashes to 0 mod n (a content-independent sample of decisions).
        emit_every = max(1, int(os.environ.get("THUNDERSYNC_EMISSION_LAW_EVERY", "1") or 1))
        from thundersync_rl import drift_log as _drift_log

        drift_on = _drift_log.enabled()
        drift_row_moments = _drift_log.row_moments
        chk_max, chk_tok, chk_exact = 0.0, 0, 0
        results = []
        n_rows = 0
        if row_ranks is not None:
            dp_rank = ps.get_data_parallel_rank()
            mine = sorted(
                (i for i in range(len(rows)) if row_ranks[i] == dp_rank),
                key=lambda i: rows[i].numel(),
            )
        else:
            mine = self._my_rows(rows, keys)
        nuc_anchors = None
        if (
            nucleus_anchors is None
            and anchors is not None
            and exact_nucleus.sticky()
            and last_stage
        ):
            # Sticky nucleus: the drafter's nucleus ids ride the route refs
            # (NUCLEUS_LAYER blocks); p row j of row i is input position
            # prompt_lens[i] - 1 + j, the rows that sample its drafted tokens.
            from thundersync_rl.routes import fetch_nucleus

            cap = exact_nucleus.anchor_cap()
            nuc_anchors = {}
            for i in mine:
                if anchors[i] is not None:
                    g_i = rows[i].numel() - prompt_lens[i]
                    nuc_anchors[i] = fetch_nucleus(
                        anchors[i], g_i + 1, cap, ray.get, lo=prompt_lens[i] - 1
                    )
            if check:
                exact_nucleus.start_stats()
        if nucleus_anchors is not None and exact_nucleus.active():
            nuc_anchors = {}
            for i in range(len(rows)):
                a = nucleus_anchors[i]
                if a is None:
                    continue
                a = torch.as_tensor(a, dtype=torch.int32)
                g_i = rows[i].numel() - prompt_lens[i]
                full = torch.full((g_i + 1, a.shape[-1]), -1, dtype=torch.int32)
                full[: min(g_i, a.shape[0])] = a[:g_i]
                nuc_anchors[i] = full
            if check:
                exact_nucleus.start_stats()
        cache = None
        starts = dict.fromkeys(mine, 0)
        from_v0: dict[int, bool] = {}
        if prefix_keys is not None:
            cache = self._prefix_state_cache(prefix_gpu_gb)
            cache.set_weights_version(self._weights_version)
            cache.release(prefix_release)
            for k_trim, bound in prefix_trim or ():
                cache.trim(k_trim, int(bound))
            cache.negctl = float(os.environ.get("THUNDERSYNC_PREFIX_CACHE_NEGCTL", "0") or 0)
            starts = {
                i: cache.start(prefix_keys[i], rows[i], prompt_lens[i]) for i in mine
            }
        # Several drafts per training-shaped forward (padded as training pads a
        # micro-batch): batch-invariant kernels give each row the same logits
        # as a one-row forward, and per-call overhead dominates small rows.
        batches = self._row_batches([rows[i][starts[i] :] for i in mine], batch_tokens)
        batches = [[mine[j] for j in b] for b in batches]
        if cache is not None:
            # A cached forward holds every row's pages / state slots at once: split calls that
            # would not fit the pools (same rows and order; packing only).
            batches = [
                part
                for b in batches
                for part in _split_to_fit(
                    b,
                    lambda idx: cache.fits(
                        [
                            (prefix_keys[i], rows[i].numel(), starts[i],
                             prompt_lens[i] if prefix_capture else None)
                            for i in idx
                        ]
                    )
                    and self._prefix_check_fits([rows[i].numel() for i in idx]),
                )
            ]
        with timed("v_lockstep"):
            lockstep = self._expert_parallel_lockstep([[] for _ in batches])
        n_dummy = 0 if lockstep is None else len(lockstep) - len(batches)
        # THUNDERSYNC_STICKY_CHECK=0: skip the sticky-rule counters of the engine q check; collecting them
        # makes every verification forward ineligible for learner CUDA graphs (learner_graphs._replay_state),
        # so CHECKS=1 runs could never exercise replayed no-grad graphs.
        sticky_check = (
            check and anchors is not None and os.environ.get("THUNDERSYNC_STICKY_CHECK", "1") == "1"
        )
        if sticky_check:
            sticky_routing.start_stats()
            if sticky_rounding.enabled():
                sticky_rounding.start_stats()

        # Verifier prefetch (object-store transport, last stage only): fetch
        # the next forward batch's hidden blocks in a background thread while
        # the current batch runs, at most two batches / THUNDERSYNC_HIDDEN_
        # PREFETCH_GB held. Default ("auto") under null-BV, whose lazy scans
        # read most rows (a ⊥ outcome scans to the root); plain BV reads few
        # rows of few drafts, so it keeps fetching on demand. Bytes only: the
        # scans read the same blocks.
        pf_mode = os.environ.get("THUNDERSYNC_HIDDEN_PREFETCH", "auto")
        use_pf = last_stage and any(isinstance(h, HiddenRefs) for h in hidden) and (
            pf_mode in ("1", "all") or (pf_mode == "auto" and null_envelopes is not None)
        )
        pf = (
            HiddenPrefetch(
                max_bytes=int(
                    float(os.environ.get("THUNDERSYNC_HIDDEN_PREFETCH_GB", "8")) * 2**30
                ),
                # Every batch of the call may be fetched ahead (the row tails can run after the call's last forward,
                # §21.11): the window is bounded by bytes only (max_bytes).
                max_batches=max(2, len(batches)),
            )
            if use_pf
            else None
        )
        pf_held: list[int] = []

        def prefetch(b: int) -> None:
            if pf is not None and b < len(batches) and b not in pf_held:
                if pf.submit([hidden[i] for i in batches[b]]):
                    pf_held.append(b)

        reforward: dict[int, tuple[float, int]] = {}

        # Anchor prefetch: the next batch's sticky-routing anchors (object-store gets of its route blocks + the
        # dense [B, S, L, K] scatter, ~86 ms per 2-row batch at Super-L22, all on the host before the forward's
        # first launch) are built in a background thread while the current batch runs on the GPU. Same call and
        # inputs as _training_forward_rows makes itself: the same anchors (THUNDERSYNC_ANCHOR_PREFETCH=0: off).
        anchor_pool = None
        if anchors is not None and os.environ.get("THUNDERSYNC_ANCHOR_PREFETCH", "1") != "0":
            # Also under the prefix-state cache: the anchors of a cached forward depend only on
            # its rows, starts (fixed above) and row padding (_prefix_forward_lists), not on the
            # cache's state, so they are built ahead the same way (identical routed tensor).
            from concurrent.futures import ThreadPoolExecutor

            anchor_pool = ThreadPoolExecutor(1, thread_name_prefix="ts-anchor-prefetch")
        anchor_futs: dict[int, Any] = {}

        def anchors_ahead(b: int) -> None:
            if anchor_pool is not None and b < len(batches) and b not in anchor_futs:
                rows_b = [rows[i] for i in batches[b]]
                if cache is not None:
                    segs, _, _, f_starts, _, _, f_anchors = self._prefix_forward_lists(
                        [None] * len(rows_b), rows_b, [starts[i] for i in batches[b]],
                        [anchors[i] for i in batches[b]],
                    )
                    anchor_futs[b] = anchor_pool.submit(
                        self._anchor_batch, f_anchors, len(segs), self._padded_width(segs), f_starts
                    )
                    return
                anchor_futs[b] = anchor_pool.submit(
                    self._anchor_batch,
                    [anchors[i] for i in batches[b]],
                    len(rows_b),
                    self._padded_width(rows_b),
                )

        # Forward grouping (THUNDERSYNC_VERIFY_FWD_GROUP=G, default 1): run up to G of this call's expert-parallel
        # lockstep forwards back to back, then the row tails of their rows (same rows, same order, same per-row
        # computations; only the host interleaving changes). Each lockstep forward starts when the slowest of the EP
        # peers arrives, and the host row tail before it (scan, emissions, D2H) varies by row, so with G = 1 every
        # forward waits for the max over ranks of the preceding tail (Ultra-L27 UFN nsys: corr -0.96 between a rank's
        # host gap and its forward-opening barrier wait). Grouping pays that max once per group of G tails.
        # Per-rank cost: the held verification logits of up to G batches, capped by THUNDERSYNC_VERIFY_FWD_GROUP_GB
        # (default 8 GiB; tokens x vocab/TP x 4 B per row, 0.55 GiB per 8.4k-token row at TP8 and V 131k).
        fwd_group = max(1, int(os.environ.get("THUNDERSYNC_VERIFY_FWD_GROUP", "1") or 1))
        fwd_group_bytes = float(os.environ.get("THUNDERSYNC_VERIFY_FWD_GROUP_GB", "8") or 8) * 2**30

        def _prefetch_window(bi):
            if pf is not None:
                while pf_held and pf_held[0] < bi:
                    pf.release()  # every row of that batch is verified
                    pf_held.pop(0)
                prefetch(bi)
                prefetch(bi + 1)

        grp_now = [1]  # forwards in the group whose tails are running (counter spec/fwd_group_rows)

        def row_logits():
            if fwd_group == 1:
                yield from _row_logits_batches()
                return
            pending: list = []  # (batch index, [(row, payload)]), forwards done, tails not yet run
            held = 0.0
            for bi, items, nbytes in _row_logits_batches(grouped=True):
                pending.append((bi, items))
                held += nbytes
                # Deferred head (head inputs held, not logits): the row tails run after the call's last forward, so
                # the EP lockstep forwards never wait on a peer's host tails (DESIGN §21.10); held bytes still cap.
                deferred_b = any(isinstance(x[1], tuple) and x[1] and x[1][0] == "deferred" for x in items)
                if (len(pending) >= fwd_group and not deferred_b) or held >= fwd_group_bytes or bi == len(batches) - 1:
                    grp_now[0] = len(pending)
                    for bj, its in pending:
                        yield from its
                        if pf is not None:  # every row of batch bj is verified: drop its blocks
                            while pf_held and pf_held[0] <= bj:
                                pf.release()
                                pf_held.pop(0)
                            # Refill the window with the next batches not yet fetched: with the tails after the
                            # call's forwards, the forward loop has ended and no longer prefetches.
                            for b_next in range(bj + 1, len(batches)):
                                if b_next not in pf_held:
                                    prefetch(b_next)
                                    break
                    pending, held = [], 0.0

        def _row_logits_batches(grouped=False):
            for bi, batch in enumerate(batches):
                if not grouped:
                    _prefetch_window(bi)
                elif pf is not None:
                    # Grouped: fetch this batch's hidden blocks while the group's forwards run (the tails read
                    # them after the group's last forward); not released before its tails ran.
                    prefetch(bi)
                anchors_ahead(bi)
                routed_b = anchor_futs.pop(bi).result() if bi in anchor_futs else None
                anchors_ahead(bi + 1)
                t0 = time.perf_counter()
                if "v_setup" not in vp:
                    vp["v_setup"] = t0 - t_all  # call entry -> first forward launch (host)
                from nemo_rl.models.megatron import bi_context_parallel as _cpx

                with timed("v_forward"), verification_logits(), _cpx.gather_outputs():  # CP: whole rows on every CP rank
                    if cache is not None:
                        lb = self._prefix_cached_forward(
                            cache,
                            [prefix_keys[i] for i in batch],
                            [rows[i] for i in batch],
                            [starts[i] for i in batch],
                            None if anchors is None else [anchors[i] for i in batch],
                            [prompt_lens[i] for i in batch],
                            None if prefix_pins is None else [prefix_pins[i] for i in batch],
                            [prompt_lens[i] if prefix_capture else None for i in batch],
                            routed=routed_b,
                        )
                        for k_b, i_b in enumerate(batch):
                            from_v0[i_b] = cache.last_from_v0[k_b]
                    elif anchors is not None:
                        lb = self._training_forward_rows(
                            [rows[i] for i in batch],
                            anchors=[anchors[i] for i in batch],
                            routed=routed_b,
                        )
                    elif len(batch) == 1:
                        lb = self._training_forward(rows[batch[0]])
                    else:
                        lb = self._training_forward_rows([rows[i] for i in batch])
                dt = time.perf_counter() - t0
                self._prof["forward"] = self._prof.get("forward", 0.0) + dt
                vp["n_fwd"] = vp.get("n_fwd", 0) + 1
                _m = int(self.cfg.get("make_sequence_length_divisible_by", 1) or 1)
                vp["fwd_tokens"] = vp.get("fwd_tokens", 0) + len(batch) * -(-max(rows[i].numel() for i in batch) // _m) * _m
                # Piece-prefix re-forward attribution (reopen trigger of the learner prefix-state cache,
                # D-2026-10-19-coord-c83): this batch's forward seconds, split over its rows by forwarded
                # tokens; a row's prefix share = positions [start, prompt_len - 1) it recomputes before its
                # piece's first logit row. Token-proportional (attention is superlinear in length: an
                # underestimate for long prefixes).
                fwd_tok = [rows[i].numel() - starts[i] for i in batch]
                tot = max(1, sum(fwd_tok))
                for k, i in enumerate(batch):
                    pre = max(0, prompt_lens[i] - 1 - starts[i])
                    reforward[i] = (dt * fwd_tok[k] / tot * pre / max(1, fwd_tok[k]), pre)
                if not getattr(self, "_verify_head_logged", False):
                    self._verify_head_logged = True
                    print(
                        f"[verify head] rank={self.rank} chunked="
                        f"{isinstance(lb, DeferredHeadLogits)} (False: whole-batch logits)",
                        flush=True,
                    )
                if isinstance(lb, DeferredHeadLogits):
                    # Chunked verification head: per row its head input only.
                    items = [(i, ("deferred", lb.row_input(k), lb.head_weight)) for k, i in enumerate(batch)]
                else:
                    items = [(i, (None if lb is None else lb[k : k + 1])) for k, i in enumerate(batch)]
                if lb is None:
                    nbytes = 0
                elif isinstance(lb, DeferredHeadLogits):
                    nbytes = lb._x.numel() * lb._x.element_size()  # the held head input
                else:
                    nbytes = lb.numel() * lb.element_size()
                del lb
                if grouped:
                    yield bi, items, nbytes
                else:
                    yield from items
                    del items

        def _timed_tails(it):
            # Per-row tail time (THUNDERSYNC_SPEC_PROF: device-synchronized): the code after ``yield`` runs when the
            # loop asks for the next row, i.e. after the row's body (``continue`` included), before the next forward.
            for item in it:
                with timed("v_row_tail"):
                    yield item

        for i, logits in _timed_tails(row_logits()):
            n, plen = rows[i].numel(), prompt_lens[i]
            g = n - plen
            if logits is None:  # not the last pipeline stage
                continue
            # Logits of positions [starts[i], n) (prefix-state cache; else 0).
            off = starts[i]
            plen, n = plen - off, n - off
            dev = logits[1].device if isinstance(logits, tuple) else logits.device
            if gen is None:
                gen = torch.Generator(device=dev).manual_seed(
                    seed * 1_000_003 + ps.get_data_parallel_rank()
                )
            deferred = isinstance(logits, tuple)
            if deferred:
                # Never [tokens, V] for a batch: rows in <= 4096-row chunks from
                # the head input, bitwise the materialized rows (DeferredLogprobRows).
                _, x_rows, w_head = logits
                if tp == 1:
                    p_lp = DeferredLogprobRows(x_rows[plen - 1 : n], w_head, vocab_limit)
                else:
                    # This rank's vocabulary shard of the row's logits from its head input: the forward's own head
                    # GEMM on the row (row-invariant fp32_gemm: bitwise the batch's shard rows), then as below.
                    from nemo_rl.models.megatron import fp16_gemm as _f16

                    p_lp = GatheredLogprobRows(_f16.fp32_gemm(x_rows[plen - 1 : n], w_head), gather, vocab_limit)
            elif tp == 1:
                # Log-probs exactly as training computes them (fp32 log-softmax;
                # exact top-p: of the nucleus-masked rows).
                with timed("p_log_softmax"):
                    p_lp = exact_nucleus.policy_log_softmax(
                        logits[0, plen - 1 : n, :vocab_limit],
                        None
                        if nuc_anchors is None or i not in nuc_anchors
                        else nuc_anchors[i].to(logits.device),
                    )
            else:
                # Vocab-parallel logits: rows gathered on demand, as batch-
                # invariant training's gathered log-softmax computes them.
                p_lp = GatheredLogprobRows(logits[0, plen - 1 : n], gather, vocab_limit)
            if nuc_anchors is not None and not isinstance(p_lp, torch.Tensor):
                p_lp.anchors = nuc_anchors.get(i)
            del logits
            # Pinned, asynchronous host-to-device copies (a pageable copy waits for the stream).
            devscan = devscan_enabled()
            if devscan:
                draft = _h2d(rows[i][plen + off :], dev)
                lq_h = torch.tensor(engine_logq[i], dtype=torch.float32)
                lq = _h2d(lq_h, dev)
            else:
                draft = rows[i][plen + off :].to(dev)
                lq = torch.tensor(engine_logq[i], dtype=torch.float32, device=dev)
            assert lq.numel() == g, (lq.numel(), g)
            assert hidden_num_rows(hidden[i]) == g, (hidden_num_rows(hidden[i]), g)
            from thundersync_rl.engine_verify import HiddenRecompute

            q_engine = getattr(self, "_q_engine", None)
            if isinstance(hidden[i], HiddenRecompute) and q_engine is not None:
                # Learner-verify path with hidden_transport=recompute (anchor epochs): nothing was
                # logged while drafting; the q rows are rebuilt on first use by a prefill of prompt + draft on the
                # engine that still holds the drafting version (the anchor, R2). Bitwise the logged rows under
                # batch invariance: CHECKS compare every rebuilt row's q of the drafted token with the engine's
                # logged log q (the engine q check below).
                from thundersync_rl.engine_verify import RecomputedRows

                qvs = sorted({int(v) for v, a, b in segments[i] if b > a})
                if len(qvs) != 1:
                    raise RuntimeError(f"q recompute: draft {keys[i]} spans versions {qvs} (one engine version only)")
                qkey = f"{seed}|{keys[i]}|{prompt_lens[i]}|{g}"

                def q_fetch(i=i, qkey=qkey, qv=qvs[0]):
                    ref = ray.get(
                        q_engine.q_recompute.remote(
                            [qkey], [rows[i].tolist()], [int(prompt_lens[i])], qv, consumers=tp
                        )
                    )[qkey]
                    return ray.get(ref)

                h = RecomputedRows(q_fetch, g)
            elif isinstance(hidden[i], HiddenRecompute):
                # Engine-side verification's check (hidden_transport="recompute":
                # no q rows here): p of the drafted tokens only, no decision.
                if not return_token_logprobs:
                    raise RuntimeError(
                        "learner verification of a draft without logged hidden "
                        "states (hidden_transport=recompute)"
                    )
                if exact_law:
                    lpx = exact_bv.p_token_lambda(p_lp, draft)
                else:
                    lpx = (
                        p_lp.token_logprobs(draft)
                        if isinstance(p_lp, GatheredLogprobRows)
                        else p_lp[torch.arange(g, device=dev), draft]
                    )
                results.append((i, {"p_token_logprobs": lpx.tolist(), "check_only": True}))
                del p_lp
                continue
            # Phase 2 only: rows (per segment) fetched when the scan asks.
            if not isinstance(hidden[i], HiddenRecompute):
                h = HiddenRows(hidden[i], prefetched=None if pf is None else pf.futures)
            version = [0] * g
            for v, a, b in segments[i]:
                version[a:b] = [v] * (b - a)

            def q_rows(idx, h=h, version=version):
                idx_l = idx.tolist()
                out: list[Any] = [None] * len(idx_l)
                by_v: dict[int, list[int]] = {}
                for j, t in enumerate(idx_l):
                    by_v.setdefault(version[t], []).append(j)
                for v, js in by_v.items():
                    with timed("q_rows/hidden_fetch"):
                        hv = h([idx_l[j] for j in js])
                    with timed("q_rows/head_gemm"):
                        lp = self._head_logprobs(hv, v, vocab_limit)
                    for k, j in enumerate(js):
                        out[j] = lp[k]
                return torch.stack(out)

            if row_seeds is not None:
                assert row_seed_base is None and walks is None and walk_scans is None, "row_seeds: verify0 grid rows only"
                row_gen = vit.row_generator(int(row_seeds[i]), dev)
            elif row_seed_base is not None:
                row_gen = vit.row_generator(vit.row_seed(row_seed_base, keys[i]), dev)
            else:
                row_gen = gen
            walk = None if walks is None else walks[i]
            scan = None if walk_scans is None else walk_scans[i]
            ds2_used = False  # this row's scan ran on the vocabulary shard (counter spec/ds2_rows)
            edv_row = None  # EDV shadow record of a DS2 row (ds2.take_last_row)
            with nonfinite_context(
                rank=self.rank, row=i, prompt_len=prompt_lens[i], g=g,
                seed=seed, segments=segments[i],
                weights_version=getattr(self, "_weights_version", None),
            ):
                if walk is not None or scan is not None:
                    from thundersync_rl.engine_q import verify_engine_q_scan, verify_engine_q_walk

                    assert null_envelopes is None and band is None, "carried walk: plain BV only"
                    assert row_seed_base is None, "carried walk: not a verify-in-train fallback call"
                    if walk is not None:
                        res = verify_engine_q_walk(p_lp, draft, lq, q_rows, gen, walk)
                    else:
                        res = verify_engine_q_scan(p_lp, draft, lq, q_rows, gen, scan)
                    n_rows += res.get("rows_evaluated", 0)
                elif null_envelopes is not None:
                    # Banded IS is rejected with null_bv by validation (§14e).
                    assert band is None, "banded IS does not compose with pooled null_bv"
                    with timed("grid_scan_total"):
                        if exact_law:
                            # Exact integer laws: lpx = lambda_p (fp32).
                            grid, lpx, n_eval = exact_bv.verify_exact_grid(
                                p_lp, q_rows, draft, lq, row_gen, tuple(null_envelopes[i])
                            )
                        elif _ds2.eligible(
                            p_lp, h, exact_law, nuc_anchors is not None or exact_nucleus.active()
                        ):
                            # DS2 (THUNDERSYNC_DS2=1): the same scan on the vocabulary shard (ds2.py).
                            ds2_used = True
                            grid, lpx, n_eval = self._ds2_grid(
                                p_lp, h, draft, lq, segments[i], vocab_limit, row_gen,
                                tuple(null_envelopes[i]),
                            )
                            edv_row = _ds2.take_last_row()  # EDV shadow record (log only, host values)
                        else:
                            grid, lpx, n_eval = verify_engine_q_grid(
                                p_lp, draft, lq, q_rows, row_gen, tuple(null_envelopes[i])
                            )
                    tau = max(t for t, _, _ in grid.values())
                    if os.environ.get("THUNDERSYNC_NL_DEBUG_ROWS") == "1":
                        self._nl_debug_rows(grid, p_lp, q_rows, draft, lq, ds2_used)
                    if devscan:
                        # log W* = max_t log prod_{u<=t} p/q over the draft (drop-IS weights).
                        d_lr = lpx.double() - lq.to(lpx.device).double()
                        lr = torch.cumsum(d_lr, 0)
                        # Exact top-p: a drafted token outside the target nucleus has
                        # log p = -inf (rejected with probability 1); the moments
                        # (the ⊥-BV calibration) are those of the finite tokens.
                        # One packed device-to-host copy for the row's values (W*, moments, log p of
                        # the drafted tokens); with every token finite the moments are the device
                        # sums of d_lr itself (the same tensor the finite selection would copy).
                        w_h, npz_h, s1_h, s2_h, lpx_h = _d2h(
                            [
                                lr.max().view(1) if lr.numel() else lr.new_zeros(1),
                                (~torch.isfinite(d_lr)).sum().view(1),
                                d_lr.sum().view(1),
                                (d_lr * d_lr).sum().view(1),
                                lpx,
                            ]
                        )
                        n_pzero = int(npz_h[0])
                        # Per-age / per-position moments (drift_log.py): logging only. Host copy of
                        # d_lr: the same fp64 differences (fp32 log p and log q are exact in fp64).
                        seg_m, pos_m = (
                            drift_row_moments(lpx_h - lq_h.double(), segments[i]) if drift_on else ((), ())
                        )
                        if n_pzero:
                            d_f = d_lr[torch.isfinite(d_lr)]
                            moments = (int(d_f.numel()), float(d_f.sum()), float((d_f * d_f).sum()))
                        else:
                            moments = (int(d_lr.numel()), float(s1_h[0]), float(s2_h[0]))
                        w_star = float(w_h[0]) if lr.numel() else 0.0
                        tok_lp, d_lq = lpx_h[:tau].tolist(), lq_h[:tau].tolist()
                    else:
                        # log W* = max_t log prod_{u<=t} p/q over the draft (drop-IS weights).
                        lr = torch.cumsum(lpx.double() - lq.to(lpx.device).double(), 0)
                        d_lr = lpx.double() - lq.to(lpx.device).double()
                        d_fin = torch.isfinite(d_lr)
                        n_pzero = int((~d_fin).sum())
                        seg_m, pos_m = drift_row_moments(d_lr, segments[i]) if drift_on else ((), ())
                        d_lr = d_lr[d_fin]
                        w_star = float(lr.max()) if lr.numel() else 0.0
                        moments = (int(d_lr.numel()), float(d_lr.sum()), float((d_lr * d_lr).sum()))
                        tok_lp, d_lq = lpx[:tau].tolist(), lq[:tau].tolist()
                    res = {
                        "null_grid": grid,
                        "null_log_wstar": w_star,
                        # Per-token log p/q moments over the draft (the ⊥-BV
                        # policy's calibration of later steps, null_bv.py).
                        "null_lr_moments": moments,
                        "token_logprobs": tok_lp,
                        "draft_logq": d_lq,
                        "p_zero_tokens": n_pzero,
                        "lr_seg_moments": seg_m,
                        "lr_pos_moments": pos_m,
                    }
                    if ds2_used and edv_row is not None:
                        # EDV shadow (DESIGN §22.5): eta_g and b_g per envelope for the driver's jsonl. Read only.
                        res["edv_shadow"] = {**edv_row, "key": keys[i]}
                    from thundersync_rl import acc_dump

                    if acc_dump.picked(keys[i]):
                        # Research dump (acceptance decay study, off by default): read-only.
                        acc_dump.dump_row(
                            self.rank, getattr(self, "_weights_version", None), keys[i], segments[i],
                            p_lp, q_rows, draft, lq, lpx, grid, row=rows[i], prompt_len=prompt_lens[i],
                        )
                    if law_on and _emit_sampled(keys[i], emit_every):
                        # ⊥-BV emission law: PIT of the token / BOT emitted at every envelope M under
                        # the ⊥-BV law given (draft, tau_M, M), recomputed from the definition; the
                        # driver keeps the PIT of the M the pool's counts decide the row at.
                        res["null_grid_pit"] = self._null_emission_pits(
                            p_lp, draft, lq, q_rows, grid, seed, keys[i]
                        )
                    n_rows += n_eval
                elif exact_law:
                    assert band is None, "banded IS is refused with exact_int"
                    tau, y, logp, n_eval = exact_bv.verify_exact(
                        p_lp, q_rows, draft, lq, row_gen, 1.0
                    )
                    res = {
                        "accepted": tau,
                        "next": y,
                        "logprobs": logp.tolist(),
                        "draft_logq": lq[:tau].tolist(),
                    }
                    if law_on and y >= 0 and _emit_sampled(keys[i], emit_every):
                        # Emission-law check (ts-exact-harden 14as.8) on the exact-Bernoulli path: the
                        # reference law from the log-prob rows equals w/W to 2^-52 relative.
                        self._emission_law(p_lp, draft, lq, q_rows, tau, int(y), seed, keys[i])
                    n_rows += n_eval + (tau < g)
                else:
                    tau, y, logp, n_eval = verify_engine_q(
                        p_lp, draft, lq, q_rows, row_gen, 1.0, band
                    )
                    res = {
                        "accepted": tau,
                        "next": y,
                        "logprobs": logp.tolist(),
                        "draft_logq": lq[:tau].tolist(),
                    }
                    if law_on and band is None and y >= 0 and _emit_sampled(keys[i], emit_every):
                        # Emission law check (law_check.reference_emission): the emitted token
                        # vs the BV law given (draft, tau) recomputed from the definition; its own
                        # uniform (the verifier's generator stream is untouched).
                        self._emission_law(p_lp, draft, lq, q_rows, tau, int(y), seed, keys[i])
                    if band is not None:
                        # logp[:tau] are the log p of the kept draft tokens the scan used.
                        log_w, exited = banded_log_weight(logp[:tau] - lq[:tau], band)
                        res["band_log_weight"] = log_w
                        res["band_exited"] = exited
                    n_rows += n_eval + (tau < g)
            if return_token_logprobs:
                # Every drafted token's log p (check; also walk / scan / grid rows).
                if exact_law:
                    lpx = exact_bv.p_token_lambda(p_lp, draft)
                else:
                    lpx = (
                        p_lp.token_logprobs(draft)
                        if isinstance(p_lp, GatheredLogprobRows)
                        else p_lp[torch.arange(g, device=dev), draft]
                    )
                res["p_token_logprobs"] = lpx.tolist()
            # What the scan needed (not the check): nominal and storage bytes,
            # and the time the scan waited for blocks.
            fetched = h.fetched_bytes
            fetch_info = {
                "hidden_real_bytes_fetched": h.fetched_real_bytes,
                "hidden_fetch_s": h.fetch_s,
                "q_recomputed": int(bool(getattr(h, "recomputed", False))),
                # engagement counters (logging only): DS2 scan; forward ran in a group of >= 2 (FWD_GROUP)
                "ds2_rows": int(ds2_used),
                "fwd_group_rows": int(grp_now[0] > 1),
                "ds2_dg_checked": int(self.__dict__.pop("_dg_checked_row", 0)),
                "ds2_vw_checked": int(self.__dict__.pop("_vw_checked_row", 0)),
                "ds2_vw_mismatch": int(self.__dict__.pop("_vw_mismatch_row", 0)),
            }
            self._prof["hidden_fetch_s"] = self._prof.get("hidden_fetch_s", 0.0) + h.fetch_s
            if check:
                for a in range(0, g, 512):
                    idx = torch.arange(a, min(g, a + 512))
                    if exact_law:
                        # Logged law check: the engine's lambda_q == the q rows' lambda.
                        rec = xi_law.int_logprob(q_rows(idx), draft[idx.to(dev)]).float()
                    else:
                        rec = q_rows(idx).gather(-1, draft[idx.to(dev)].view(-1, 1))[:, 0]
                    d = (rec - lq[a : a + idx.numel()]).abs()
                    chk_max = max(chk_max, float(d.max()))
                    chk_exact += int((d == 0).sum())
                    chk_tok += idx.numel()
                    bad = (d > 0).nonzero().flatten()
                    if bad.numel() and env_flag("THUNDERSYNC_HIDDEN_DEBUG"):
                        pos = (bad + a).tolist()[:8]
                        print(
                            f"[engine q check] inexact positions {pos} of g={g} "
                            f"d={[round(float(d[b]), 2) for b in bad[:8]]} "
                            f"rec={[round(float(rec[b]), 2) for b in bad[:4]]} "
                            f"eng={[round(float(lq[a + int(b)]), 2) for b in bad[:4]]} "
                            f"eng_prev={[round(float(lq[a + int(b) - 1]), 2) if a + int(b) > 0 else None for b in bad[:4]]} "
                            f"eng_next={[round(float(lq[a + int(b) + 1]), 2) if a + int(b) + 1 < g else None for b in bad[:4]]} "
                            f"versions_at={[version[t] for t in pos]} segs={segments[i]}",
                            flush=True,
                        )
            if cache is not None:
                res["prefix_start"] = off
                res["prefix_from_v0"] = bool(from_v0.get(i, False))
                if not results:
                    b_now = cache.bytes()
                    res["prefix_cache_bytes"] = [b_now["gpu"], b_now["host"], b_now["pool"]]
                    pgs = self.__dict__.get("_prefix_graph_stats")
                    if pgs:
                        res["prefix_cache_graph"] = [pgs["calls"], pgs["replays"], pgs["captures"], pgs["eager"]]
            if i in reforward:
                res["prefix_reforward_s"], res["prefix_reforward_tokens"] = reforward[i]
            results.append((i, {**res, "hidden_bytes_fetched": fetched, **fetch_info}))
            del p_lp, h
        if pf is not None:
            self._prof["hidden_prefetched_bytes"] = pf.submitted_bytes
            self._prof["hidden_prefetch_skipped"] = pf.skipped_batches
            pf.close()
        if anchor_pool is not None:
            anchor_pool.shutdown(wait=True, cancel_futures=True)
        # The prefix-cache check runs a second (whole-row) forward per batch.
        per_batch = 2 if cache is not None and _prefix_check_on() else 1
        t_dummy = time.perf_counter()
        with timed("v_ep_dummy"):
            for _ in range(n_dummy * per_batch):
                # Expert parallelism: out of rows, join the other replicas' forwards.
                self._dummy_training_forward()
        vp["ep_dummy_host"] = time.perf_counter() - t_dummy
        self._prof["ep_dummy_forwards"] = n_dummy
        self._prof["q_rows"] = n_rows
        if _ds2.COUNTS["rows"] != getattr(self, "_ds2_rows_logged", 0):
            self._ds2_rows_logged = _ds2.COUNTS["rows"]
            print(f"[ds2] rank={self.rank} " + " ".join(f"{k}={v}" for k, v in _ds2.COUNTS.items()), flush=True)
        if cache is not None:
            self._prof["prefix_tokens_skipped"] = sum(starts.values())
            self._prof["prefix_tokens_forwarded"] = sum(
                rows[i].numel() - starts[i] for i in mine
            )
            self._prof.update(
                {f"prefix_cache_{k}_bytes": v for k, v in cache.bytes().items() if k != "rows"}
            )
        self._prof.update(prof_pop())
        self._log_mem("verify_drafts_block_engine")
        if check and last_stage:
            print(
                f"[engine q check] rank={self.rank} rows={len(results)} tok={chk_tok} "
                f"exact={chk_exact} max|dlogq|={chk_max:.3e}",
                flush=True,
            )
        if check and nuc_anchors is not None:
            ns = exact_nucleus.pop_stats()
            print(
                f"[nucleus check] rank={self.rank} rows={int(ns.get('rows', 0))} "
                f"anchored={int(ns.get('anchored', 0))} kept={int(ns.get('kept', 0))} "
                f"differs_from_plain={int(ns.get('differs', 0))}",
                flush=True,
            )
        if sticky_check:
            st = sticky_routing.pop_stats()
            print(
                f"[sticky check] rank={self.rank} token_layers={int(st['token_layers'])} "
                f"anchor_kept={st['anchor_kept']:.6f} "
                f"differs_from_plain={st['differs_from_plain']:.6f}",
                flush=True,
            )
            if sticky_rounding.enabled():
                sround = sticky_rounding.pop_stats()
                print(
                    f"[sround check] rank={self.rank} elements={int(sround['elements'])} "
                    f"changed={int(sround['changed'])} "
                    f"changed_frac={sround['changed_frac']:.6f}",
                    flush=True,
                )
        if torch_prof is not None:
            torch_prof.__exit__(None, None, None)
            path = f"{os.environ['THUNDERSYNC_TORCH_PROF']}.rank{self.rank}.txt"
            with open(path, "w") as f:
                for key in ("self_cuda_time_total", "self_cpu_time_total"):
                    f.write(torch_prof.key_averages().table(sort_by=key, row_limit=40) + "\n")
            print(f"[spec prof] torch profile -> {path}", flush=True)
        if env_flag("THUNDERSYNC_SPEC_PROF"):
            ntok = sum(rows[i].numel() for i, _ in results)
            print(
                f"[spec prof] verify_engine rank={self.rank} rows={len(results)} "
                f"tok={ntok} total={time.perf_counter() - t_all:.3f} "
                f"batches={len(batches)} batch_tokens={batch_tokens} "
                f"null={int(null_envelopes is not None)} {self._prof}",
                flush=True,
            )
        if results:
            vp["total"] = time.perf_counter() - t_all
            vp["t_out"] = time.time()
            vp["forward_host"] = self._prof.get("forward", 0.0)
            vp["n_dummy"] = n_dummy
            vp["hidden_fetch_s"] = self._prof.get("hidden_fetch_s", 0.0)
            results[0][1]["vprof_s"] = vp
        if reply_to is not None:
            mailbox, key = reply_to
            mailbox.post.remote(key, int(getattr(self, "rank", 0)), [ray.put(results)])
            return None
        return results

    def _nl_debug_rows(self, grid, p_lp, q_rows, draft, lq, ds2_used) -> None:
        """Debug (THUNDERSYNC_NL_DEBUG_ROWS=1): at a rejected row's stop position, how the full p and q rows the
        residual is computed from differ (a correct verifier at equal weights has p == q rows bitwise)."""
        from thundersync_rl.engine_q import GatheredLogprobRows

        g = draft.numel()
        ts = sorted({int(t) for t, _, _ in grid.values() if 0 < int(t) < g})
        if not ts:
            return
        t = ts[0]
        lpx_all = (p_lp.token_logprobs(draft.to(lq.device)) if isinstance(p_lp, GatheredLogprobRows)
                   else p_lp[torch.arange(g, device=lq.device), draft.to(lq.device)])
        dl = (lpx_all.double() - lq.double())
        print(f"[nl debug rows] grid={ {float(m): (int(a), int(y)) for m, (a, y, _) in grid.items()} } "
              f"max|lp-lq| over draft={float(dl.abs().max()):.3e} nonzero={int((dl != 0).sum())}", flush=True)
        dev = lq.device
        at = torch.tensor([t], device=dev)
        P = p_lp(at)[0] if isinstance(p_lp, GatheredLogprobRows) else p_lp[t]
        Q = q_rows(at.cpu() if not callable(getattr(q_rows, "device", None)) else at)[0]
        P, Q = P.double().to(dev), Q.double().to(dev)
        n = min(P.numel(), Q.numel())
        pp, qq = P[:n].exp(), Q[:n].exp()
        d = (pp - qq).abs()
        j = int(d.argmax())
        x = int(draft[t])
        print(f"[nl debug rows] rank={getattr(self, 'rank', '?')} ds2={int(ds2_used)} g={g} tau={t} nP={P.numel()} nQ={Q.numel()} "
              f"sumP={float(P.exp().sum()):.6f} sumQ={float(Q.exp().sum()):.6f} max|p-q|={float(d.max()):.3e} at={j} "
              f"p={float(pp[j]):.3e} q={float(qq[j]):.3e} draft_tok={x} lP={float(P[x]):.6f} lQ={float(Q[x]):.6f} "
              f"lq_engine={float(lq[t]):.6f} n_diff={int((d > 1e-6).sum())}", flush=True)

    def _emission_law(self, p_lp, draft, lq, q_rows, tau: int, y: int, seed: int, key: str) -> None:
        """Accumulate the emission-law PIT of one plain-BV decision (THUNDERSYNC_LAW_CHECK)."""
        from thundersync_rl import law_check
        from thundersync_rl.engine_q import GatheredLogprobRows

        g = draft.numel()
        dev = lq.device
        if isinstance(p_lp, GatheredLogprobRows):
            lpx = p_lp.token_logprobs(draft.to(dev))
            lp_row = p_lp(torch.tensor([tau], device=dev))[0]
        else:
            lpx = p_lp[torch.arange(g, device=dev), draft]
            lp_row = p_lp[tau]
        lq_row = None if tau == g else q_rows(torch.tensor([tau]))[0]
        dist = law_check.reference_emission(lp_row, lq_row, lpx, lq, tau)
        # 64-bit hash uniform (ts-lawfix): crc32 of structured keys is not an independent uniform.
        u0 = law_check.hash_uniform("emit", seed, key, tau)
        u, t = law_check.emission_pit(dist, y, u0)
        acc = getattr(self, "_emission_acc", None)
        if acc is None:
            acc = self._emission_acc = {"residual": [], "bonus": []}
        acc["residual" if tau < g else "bonus"].append((u, t))

    def _null_emission_pits(self, p_lp, draft, lq, q_rows, grid, seed: int, key: str) -> dict:
        from thundersync_rl import law_check
        from thundersync_rl.engine_q import GatheredLogprobRows

        g = draft.numel()
        dev = lq.device
        if isinstance(p_lp, GatheredLogprobRows):
            lpx = p_lp.token_logprobs(draft.to(dev))
            prow = lambda t: p_lp(torch.tensor([t], device=dev))[0]  # noqa: E731
        else:
            lpx = p_lp[torch.arange(g, device=dev), draft]
            prow = lambda t: p_lp[t]  # noqa: E731
        qcache: dict[int, torch.Tensor] = {}
        out = {}
        for m, (tau, y, _) in grid.items():
            if tau < g and tau not in qcache:
                qcache[tau] = q_rows(torch.tensor([tau]))[0]
            dist = law_check.reference_null_emission(
                prow(tau), qcache.get(tau), lpx, lq, tau, float(m)
            )
            idx = int(y) if int(y) >= 0 else dist.numel() - 1  # BOT = the last column
            u0 = law_check.hash_uniform("emit_null", seed, key, m)
            out[m] = law_check.emission_pit(dist, idx, u0)
        return out

    def pop_emission_law(self) -> dict[str, Any] | None:
        """This rank's emission-law PIT stats since the last pop ({kind: [N_STATS]}); None on
        ranks that do not own the counts (TP rank > 0 repeat rank 0's decisions; non-last PP stages)."""
        from megatron.core import parallel_state as ps
        from thundersync_rl import law_check

        acc, self._emission_acc = getattr(self, "_emission_acc", None), None
        last = ps.get_pipeline_model_parallel_world_size() == 1 or ps.is_pipeline_last_stage(ignore_virtual=True)
        if not last or ps.get_tensor_model_parallel_rank() != 0:
            return None
        out = {}
        for kind in ("residual", "bonus"):
            pts = (acc or {}).get(kind) or []
            u = torch.tensor([a for a, _ in pts], dtype=torch.float64)
            t = torch.tensor([b for _, b in pts], dtype=torch.float64)
            out[kind] = law_check.accumulate(u, t).tolist()
        return out

    def _torch_profiler(self, null: bool):
        """``THUNDERSYNC_TORCH_PROF=<path prefix>``: profile this rank's second ⊥-BV
        verification call (the first one compiles and warms caches); None otherwise."""
        if not null or not os.environ.get("THUNDERSYNC_TORCH_PROF"):
            return None
        self._null_verify_calls = getattr(self, "_null_verify_calls", 0) + 1
        if self._null_verify_calls != 2:
            return None
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        )
        # Same as the learner torch profile: the trace metadata lists every process group's backend config and
        # groups without one make torch 2.11 raise "value was None" at __enter__ (s2fix2 vprof-tp2, Super-L8 TP2xEP2).
        # Only key_averages tables are written here.
        prof._get_distributed_info = lambda: None
        prof.__enter__()
        return prof

    def _check_dropless_moe(self) -> None:
        """Raise if MoE layers drop tokens (expert capacity factor set).

        With a capacity factor, whether a token reaches its experts depends on
        every other token routed to them in the same forward (other rows, other
        expert-parallel ranks' rows, lockstep dummies), so p of a row is not a
        function of the row and its weights: verification against it is not
        exact. Dropless MoE (capacity factor None, Megatron's default) only.
        """
        from megatron.core.utils import get_model_config

        try:
            config = get_model_config(self.model)
        except (AttributeError, RuntimeError):  # no Megatron config (fakes)
            return
        if getattr(config, "num_moe_experts", None) and (
            getattr(config, "moe_expert_capacity_factor", None) is not None
        ):
            raise RuntimeError(
                "verify_source='engine' requires dropless MoE "
                "(moe_expert_capacity_factor=None): token dropping makes a "
                "row's p depend on the other tokens in its forward"
            )

    # ---- Fused verification + training (batch-invariant kernels; fused_train.py) ----

    def _check_fused_verify_supported(self, state: dict[str, Any]) -> None:
        """Raise unless the verification forward can stand in for training's."""
        from megatron.core import parallel_state as ps

        problems = []
        if state["grad_stream"] is None:
            problems.append("the train step was opened without gradient streaming")
        if not self.cfg["megatron_cfg"].get("batch_invariant_mode"):
            problems.append("batch_invariant_mode is off")
        if state["mtp_enabled"]:
            problems.append("multi-token prediction needs the loss mask in the forward")
        if self.draft_model is not None:
            problems.append(
                "draft-model training captures hidden states in the forward"
            )
        if self.cfg["megatron_cfg"].get("use_fused_linear_logprobs"):
            problems.append("fused linear log-probs produce no logits")
        if ps.get_pipeline_model_parallel_world_size() != 1:
            problems.append("pipeline parallelism (the graph spans stages)")
        if ps.get_tensor_model_parallel_world_size() != 1:
            problems.append("tensor parallelism (training logits are vocab-sharded)")
        if ps.get_context_parallel_world_size() != 1:
            problems.append("context parallelism")
        if problems:
            raise RuntimeError(
                "fused verification + training unsupported: " + "; ".join(problems)
            )

    def _prefix_state_cache(self, gpu_gb: float):
        """This rank's learner prefix-state cache (created on first use; hooks the model)."""
        from nemo_rl.models.megatron.prefix_state_cache import PrefixStateCache

        if self._prefix_cache is None:
            from nemo_rl.models.megatron import sticky_rounding

            if sticky_rounding.enabled():
                raise NotImplementedError(
                    "learner_prefix_cache with sticky_rounding: the theta_n anchor "
                    "forward's cached state is not built yet"
                )
            self._prefix_cache = PrefixStateCache(
                self.model,
                int(gpu_gb * 2**30),
                # Page-table width: the longest row (prompt + generation) plus one page.
                max_len=int(self.cfg.get("max_total_sequence_length") or 0) or None,
            )
        self._prefix_cache.gpu_bytes = int(gpu_gb * 2**30)
        return self._prefix_cache

    def prefix_state_cache_costs(self) -> dict[str, int]:
        """This rank's cache bytes per cached position and per kept chunk boundary."""
        return self._prefix_state_cache(8.0).costs()

    def _prefix_forward_lists(self, keys, rows, starts, anchors, pins=None, floors=None):
        """(segs, keys, rows, starts, pins, floors, anchors) of a prefix-state forward, padded.

        CUDA graphs: the row count is padded to a power of two with one-token dummy rows (not
        cached, outputs ignored), so piece calls share few graph shapes. Shared by the forward
        and the verifier's anchor prefetch, so both build the same anchors.
        """
        segs = [r[s:] for r, s in zip(rows, starts)]
        pad = self._prefix_row_pad(len(rows), max(x.numel() for x in segs))
        f_keys, f_rows, f_starts = list(keys), list(rows), list(starts)
        f_pins = None if pins is None else list(pins)
        f_floors = None if floors is None else list(floors)
        f_anchors = None if anchors is None else list(anchors)
        for _ in range(pad):
            f_keys.append(None)
            f_rows.append(rows[0][:1])
            f_starts.append(0)
            segs.append(rows[0][:1])
            for lst in (f_pins, f_floors, f_anchors):
                if lst is not None:
                    lst.append(None)
        return segs, f_keys, f_rows, f_starts, f_pins, f_floors, f_anchors

    def _prefix_cached_forward(
        self, cache, keys, rows, starts, anchors, prompt_lens, pins=None, floors=None, routed=None
    ) -> torch.Tensor | None:
        """Logits [B, S, V] of rows[b][starts[b]:] continued from the prefix-state cache.

        ``routed``: the anchors built ahead by the verifier's prefetch (``_anchor_batch`` of
        ``_prefix_forward_lists``' padded rows); None builds them here.
        ``THUNDERSYNC_PREFIX_CACHE_CHECK=1``: also run the whole rows and count
        the rows whose logits at positions [prompt - 1, n) are bitwise equal
        (expected: all).
        """
        segs, f_keys, f_rows, f_starts, f_pins, f_floors, f_anchors = self._prefix_forward_lists(
            keys, rows, starts, anchors, pins, floors
        )
        from nemo_rl.models.megatron import learner_graphs

        st0 = learner_graphs.stats() if learner_graphs.enabled() else {}
        with cache.forward(f_keys, f_rows, f_starts, f_pins, f_floors):
            lb = self._training_forward_rows(
                segs, anchors=f_anchors, anchor_starts=f_starts, routed=routed
            )
        st1 = learner_graphs.stats() if learner_graphs.enabled() else {}
        # Cached forwards by graph outcome: captured / replayed / eager.
        pg = self.__dict__.setdefault(
            "_prefix_graph_stats", {"calls": 0, "replays": 0, "captures": 0, "eager": 0}
        )
        pg["calls"] += 1
        if st1.get("captured", 0) > st0.get("captured", 0):
            pg["captures"] += 1
        elif st1.get("replay_fwd", 0) > st0.get("replay_fwd", 0):
            pg["replays"] += 1
        else:
            pg["eager"] += 1
        from_v0 = list(cache.last_from_v0)
        if _prefix_check_on():
            full = self._training_forward_rows(rows, anchors=anchors)
            if lb is not None:
                st = self.__dict__.setdefault(
                    "_prefix_check",
                    {"rows": 0, "exact": 0, "hits": 0, "positions": 0, "v0_hits": 0,
                     "v0_exact": 0, "captures": 0, "captures_exact": 0},
                )
                from nemo_rl.models.megatron.deferred_head import DeferredHeadLogits

                for b, (r, s, p) in enumerate(zip(rows, starts, prompt_lens)):
                    n = r.numel()
                    if isinstance(lb, DeferredHeadLogits) and isinstance(full, DeferredHeadLogits):
                        # Chunked verification head: equal head inputs (and the same
                        # weight) give equal logits rows (row-invariant fp32_gemm).
                        same = torch.equal(
                            lb.row_input(b)[p - 1 - s : n - s], full.row_input(b)[p - 1 : n]
                        )
                    else:
                        same = torch.equal(lb[b, p - 1 - s : n - s], full[b, p - 1 : n])
                    st["rows"] += 1
                    st["exact"] += int(same)
                    st["hits"] += int(s > 0)
                    st["positions"] += n - p + 1
                    # First pieces started from a verify0 capture, and the verify0 capture
                    # forwards themselves (their logits decide verify0: must equal too).
                    st["v0_hits"] += int(from_v0[b])
                    st["v0_exact"] += int(from_v0[b] and same)
                    cap = floors is not None and floors[b] is not None and keys[b] is not None
                    st["captures"] += int(cap)
                    st["captures_exact"] += int(cap and same)
                verdict = "PASS" if st["exact"] == st["rows"] else "FAIL"
                print(f"[prefix cache check] {verdict} rank={self.rank} {st}", flush=True)
            del full
        return lb

    def _prefix_check_fits(self, lengths: list[int]) -> bool:
        """THUNDERSYNC_PREFIX_CACHE_CHECK under device MoE dispatch: the check's whole-row forward of
        a batch must fit the dispatch token cap (one check forward per batch keeps the EP lockstep)."""
        from nemo_rl.models.megatron import device_dispatch

        if not (_prefix_check_on() and device_dispatch.enabled()):
            return True
        m = self.cfg.get("make_sequence_length_divisible_by", 1)
        return len(lengths) * (-(-max(lengths) // m) * m) <= device_dispatch.token_cap()

    def _prefix_row_pad(self, b: int, longest: int) -> int:
        """Dummy rows that bring a graphed prefix-state forward to a power-of-two row count.

        Only under learner CUDA graphs with device MoE dispatch (the graphed path), and only
        while B x the padded length stays within the dispatch token cap.
        """
        from nemo_rl.models.megatron import device_dispatch, learner_graphs

        if not (learner_graphs.enabled() and device_dispatch.enabled()) or b < 1:
            return 0
        target = 1 << (b - 1).bit_length()
        m = self.cfg.get("make_sequence_length_divisible_by", 1)
        s_pad = learner_graphs.bucket_length(-(-longest // m) * m)
        if target * s_pad > device_dispatch.token_cap():
            return 0
        return target - b

    def _row_batches(
        self, rows: list[torch.Tensor], batch_tokens: int
    ) -> list[list[int]]:
        """Consecutive groups of ``rows`` whose padded micro-batch fits ``batch_tokens``.

        A group costs len(group) x its longest row padded to
        ``make_sequence_length_divisible_by``; a row longer than the budget
        forms its own group. ``batch_tokens <= 0`` gives one row per group.
        """
        m = self.cfg.get("make_sequence_length_divisible_by", 1)
        out: list[list[int]] = []
        longest = 0
        for j, row in enumerate(rows):
            padded = -(-row.numel() // m) * m
            if (
                out
                and batch_tokens > 0
                and (len(out[-1]) + 1) * max(longest, padded) <= batch_tokens
            ):
                out[-1].append(j)
                longest = max(longest, padded)
            else:
                out.append([j])
                longest = padded
        return out

    def _training_forward(
        self, row: torch.Tensor, anchors: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """Logits [1, S, V] of one row, computed exactly as a training forward."""
        if anchors is None:
            return self._training_forward_rows([row])
        return self._training_forward_rows([row], anchors=[anchors])

    def _anchor_batch(
        self,
        anchors: list[Any] | None,
        batch: int,
        width: int,
        starts: list[int] | None = None,
    ) -> torch.Tensor | None:
        """``routed_experts`` [batch, width, L_stage, K] for a sticky-routing forward.

        None when sticky routing is off. ``anchors[b]``: row b's ``RouteRefs``
        (or None). Only this pipeline stage's MoE layers are materialized, and
        only their blocks are fetched from the object store. Rows without
        anchors (and padding) get -1: anchor = self, i.e. plain routing.
        ``starts[b]``: row b's batch row holds positions [starts[b], ...) of the
        rollout (prefix-state forwards), so its anchors start there too.
        """
        from nemo_rl.models.megatron import sticky_routing

        if sticky_routing.get_delta() is None or not self._router_replay_enabled:
            return None
        from thundersync_rl.routes import ROUTE_DTYPE, fetch_anchors

        layers, k = self._stage_route_layers()
        out = torch.full((batch, width, len(layers), k), -1, dtype=ROUTE_DTYPE)
        for b, a in enumerate(anchors or []):
            if a is not None:
                lo = 0 if starts is None else starts[b]
                out[b] = fetch_anchors(a, layers, k, width, ray.get, lo=lo)
        return out

    def _padded_width(self, rows: list[torch.Tensor]) -> int:
        """Micro-batch width of ``rows`` in ``_training_forward_rows`` (``make_sequence_length_divisible_by``)."""
        m = self.cfg["make_sequence_length_divisible_by"]
        return -(-max(row.numel() for row in rows) // m) * m

    def _stage_route_layers(self) -> tuple[list[int], int]:
        """(MoE layer numbers of this pipeline stage, router top-k)."""
        layers = getattr(self, "_stage_layers", None)
        if layers is None:
            from megatron.core.utils import get_model_config

            from nemo_rl.models.megatron.router_replay import (
                _router_replay_instances_for_model,
            )

            nums = sorted(
                ln for _, ln in _router_replay_instances_for_model(self.model)
            )
            k = int(get_model_config(self.model).moe_router_topk)
            layers = self._stage_layers = (nums, k)
        return layers

    def _training_forward_rows(
        self,
        rows: list[torch.Tensor],
        anchors: list[Any] | None = None,
        anchor_starts: list[int] | None = None,
        routed: Any = None,
    ) -> torch.Tensor | None:
        """Logits [B, S, V] of ``rows`` in one micro-batch, exactly as a training forward.

        ``routed``: this call's ``_anchor_batch(anchors, len(rows), self._padded_width(rows), anchor_starts)``
        computed ahead (the verifier's anchor prefetch); None computes it here.

        ``anchor_starts[b]``: rows[b] holds rollout positions [anchor_starts[b], ...)
        (a prefix-state forward); its anchors are taken from there.

        Same padding (``make_sequence_length_divisible_by``), micro-batch
        processing and ``model_forward`` call as ``train_microbatch``; autograd
        follows the caller's grad mode.

        Pipeline parallelism: the same forward through Megatron's pipeline
        schedule, forward-only (``_pipeline_forward_only``), as training runs
        each micro-batch. Every stage must call it for the same row; the last
        stage returns the logits (vocab-parallel at TP>1), the others None.
        """
        from megatron.core import parallel_state as ps

        lens = [row.numel() for row in rows]
        ids = torch.zeros((len(rows), self._padded_width(rows)), dtype=torch.long)
        for b, row in enumerate(rows):
            ids[b, : lens[b]] = row
        data = BatchedDataDict({"input_ids": ids, "input_lengths": torch.tensor(lens)})
        if routed is None:
            routed = self._anchor_batch(anchors, len(rows), ids.shape[1], anchor_starts)
        if routed is not None:
            # Sticky routing: the forward that defines p replays the anchors
            # through the training path's RouterReplay (REPLAY_FORWARD).
            data["routed_experts"] = routed
        attach_media_token_validity_mask(data, self.media_placeholder_token_id)
        iterator, _, _, _, padded_seq_length = get_microbatch_iterator(
            data,
            self.cfg,
            len(rows),
            straggler_timer=self.mcore_state.straggler_timer,
            delegate_pack_to_model=self.delegate_pack_to_model,
            delegate_mtp_loss_mask_to_model=self.delegate_mtp_loss_mask_to_model,
            model_slices_context_parallel_inputs=self.model_slices_context_parallel_inputs,
            mtp_enabled=self.mtp_enabled,
        )
        mb = next(iterator)

        def forward(model):
            if routed is None:
                return _model_forward(model)
            from nemo_rl.models.megatron.router_replay import (
                clear_router_replay,
                set_router_replay_forward,
            )

            set_router_replay_forward(model, mb.routed_experts_cp_sharded)
            try:
                return _model_forward(model)
            finally:
                clear_router_replay(model)

        def _model_forward(model):
            return megatron_train.model_forward(
                model=model,
                data_dict=mb.data_dict,
                input_ids_cp_sharded=mb.input_ids_cp_sharded,
                position_ids=mb.position_ids,
                attention_mask=mb.attention_mask,
                packed_seq_params=mb.packed_seq_params,
                defer_fp32_logits=self.defer_fp32_logits,
                mtp_loss_mask=mb.mtp_loss_mask,
                padding_mask=mb.padding_mask,
                straggler_timer=self.mcore_state.straggler_timer,
                use_fused_linear_logprobs=False,
                media_token_validity_mask=mb.media_token_validity_mask,
                model_slices_context_parallel_inputs=self.model_slices_context_parallel_inputs,
            )

        if ps.get_pipeline_model_parallel_world_size() == 1:
            return forward(self.model)
        return self._pipeline_forward_only(forward, padded_seq_length, len(rows))

    def _pipeline_forward_only(
        self, forward, seq_length: int, micro_batch_size: int = 1
    ) -> torch.Tensor | None:
        """``forward(model)`` of one micro-batch through the pipeline schedule.

        Megatron's ``get_forward_backward_func`` with ``forward_only=True`` and
        ``collect_non_loss_data=True`` (one micro-batch of size 1, sequence
        length ``seq_length`` as training passes it): each stage runs its layers
        on the activations received from the previous stage, and the last stage
        collects the output (logits). Returns them on the last stage, None on
        the others. No autograd: a graph across stages would need the backward
        schedule. The DDP ``no_sync`` hook is bypassed for the call (the
        forward-only schedule enters it without leaving it, and verification
        must not change gradient-sync state mid-step).
        """
        from megatron.core import parallel_state as ps
        from megatron.core.pipeline_parallel import get_forward_backward_func
        from megatron.core.utils import get_model_config

        if torch.is_grad_enabled():
            raise NotImplementedError(
                "autograd through a pipeline-parallel verification forward is "
                "not implemented (it needs the backward schedule)"
            )

        def step(data_iterator, model):
            next(data_iterator)
            return forward(model), lambda out, non_loss_data=True: out.detach()

        config = get_model_config(self.model)
        no_sync = config.no_sync_func
        config.no_sync_func = None
        try:
            out = get_forward_backward_func()(
                forward_step_func=step,
                data_iterator=iter([None]),
                model=self.model,
                num_microbatches=1,
                seq_length=seq_length,
                micro_batch_size=micro_batch_size,
                decoder_seq_length=seq_length,
                forward_only=True,
                collect_non_loss_data=True,
            )
        finally:
            config.no_sync_func = no_sync
        if not ps.is_pipeline_last_stage(ignore_virtual=True):
            return None
        return out[0]

    def _dummy_training_forward(self) -> None:
        """One-row training forward whose output is discarded (EP lockstep filler).

        A single token padded to ``make_sequence_length_divisible_by`` (a
        multiple of TP, so sequence parallelism splits it evenly), no autograd,
        through the pipeline schedule at PP>1: it joins the expert-parallel
        group's all-to-alls like a real row.
        """
        self._training_forward(torch.zeros(1, dtype=torch.long))

    @_lg_feature("verify_fused")
    def verify_drafts_block_fused(
        self,
        rows: list[torch.Tensor],
        prompt_lens: list[int],
        segments: list[list[tuple[int, int, int]]],
        vocab_limit: int,
        seed: int,
        keys: list[str],
        batch_tokens: int,
        precision: str,
        retain: list[bool],
        max_retained_tokens: int,
        lazy: bool = False,
    ) -> list[tuple[int, dict[str, Any]]]:
        """Block verification with p from the training model; keep training forwards.

        As ``verify_drafts_block_stashed`` (q from the stashed weights), but p
        is the training model's forward of each row at the current weights,
        run like a training forward inside the open train step. For a row with
        ``retain[i]`` (the driver would turn its full acceptance into a
        finished rollout) the forward runs with autograd; if verification
        accepts the whole draft and the rank's retained graphs stay within
        ``max_retained_tokens``, the graph is kept for the row's training call
        and the result carries ``"fused": {"key", "dp_rank"}``. Other graphs
        are freed right away.
        """
        from megatron.core import parallel_state as ps

        # Triton kernel: imported where it runs (GPU workers only).
        from thundersync_rl.block_verification_fused import (
            block_verify_logits,
            block_verify_logits_lazy,
        )

        # ``lazy``: same decisions from the lazy scan (no residual pass on
        # whole-accepted drafts).
        verify_fn = block_verify_logits_lazy if lazy else block_verify_logits

        state = self._assert_step_open()
        self._check_fused_verify_supported(state)
        self._prof = {}
        t_all = time.perf_counter()
        self._scorer_precision = precision
        scorer = self._fp32_scorer()
        self.model.train()  # the mode training runs in (the step is open)
        dp_rank = ps.get_data_parallel_rank()
        gen = torch.Generator(device="cuda").manual_seed(seed * 1_000_003 + dp_rank)
        results = []
        for group, ids in self._iter_row_id_batches(rows, batch_tokens, keys):
            pos = torch.arange(ids.shape[1], device="cuda").expand(len(group), -1)
            lq_rows = self._stashed_q_logits(
                scorer, group, ids, pos, prompt_lens, segments, vocab_limit
            )
            for r, i in enumerate(group):
                n, plen = rows[i].numel(), prompt_lens[i]
                g = n - plen
                keep = retain[i] and self._fused_graphs.can_retain(
                    n, max_retained_tokens
                )
                t0 = time.perf_counter()
                with torch.set_grad_enabled(keep):
                    logits = self._training_forward(rows[i])
                self._prof["forward"] = (
                    self._prof.get("forward", 0.0) + time.perf_counter() - t0
                )
                with torch.no_grad():
                    p_logits = logits[0, plen - 1 : n, :vocab_limit].detach().float()
                    parts = [t for _, t in sorted(lq_rows.pop(i), key=lambda x: x[0])]
                    q_logits = torch.cat(parts) if parts else p_logits[:0]
                    assert q_logits.shape[0] == g, (q_logits.shape, g)
                    draft = ids[r, plen:n]
                    tau, y, _ = verify_fn(p_logits, q_logits, draft, gen)
                    # Log-probs exactly as training computes them (fp32
                    # log-softmax + gather): generation log-probs of the plan
                    # equal the trainer's bitwise.
                    kept = torch.cat([draft[:tau], torch.tensor([y], device="cuda")])
                    logp = torch.log_softmax(p_logits[: tau + 1], -1).gather(
                        -1, kept.unsqueeze(-1)
                    )[:, 0]
                res: dict[str, Any] = {
                    "accepted": tau,
                    "next": y,
                    "logprobs": logp.tolist(),
                }
                if keep and tau == g:
                    self._fused_graphs.put(
                        keys[i],
                        FusedGraph(
                            ids=rows[i].detach().cpu().clone(), logits=logits[:, :n]
                        ),
                    )
                    res["fused"] = {"key": keys[i], "dp_rank": dp_rank}
                del logits, p_logits, q_logits
                results.append((i, res))
        self._log_mem("verify_drafts_block_fused")
        if env_flag("THUNDERSYNC_SPEC_PROF"):
            ntok = sum(rows[i].numel() for i in range(len(rows)))
            print(
                f"[spec prof] verify_fused rank={self.rank} rows={len(rows)} tok={ntok} "
                f"retained={len(self._fused_graphs)} "
                f"retained_tok={self._fused_graphs.retained_tokens} "
                f"total={time.perf_counter() - t_all:.3f} {self._prof}",
                flush=True,
            )
        return results

    @property
    def _batch_invariant(self) -> bool:
        """Whether the scorer runs batch-invariant kernels (bf16 scorer under BI).

        An fp32 scorer is built outside batch-invariant mode, so training-parity
        measures (full-row scoring, training's log-softmax) do not apply to it.
        """
        return bool(self.cfg["megatron_cfg"].get("batch_invariant_mode")) and (
            getattr(self, "_scorer_precision", "fp32") != "fp32"
        )

    def _score(self, scorer, ids, pos, select=None):
        """Scorer logits: [b, s, V], or [n, V] at ``select = (seq_idx, batch_idx)``."""
        from megatron.core import parallel_state as ps
        from thundersync_rl.scorer_kernels import select_positions, supports_selection

        if select is None:
            return self._score_full(scorer, ids, pos)
        # Batch-invariant GEMMs are invariant within a token-count shape class
        # only: score full rows, as training does, to reproduce its numerics.
        if not supports_selection(scorer) or self._batch_invariant:
            logits = self._score_full(scorer, ids, pos)
            return None if logits is None else logits[select[1], select[0]]
        with select_positions(scorer, select):
            logits = self._score_full(scorer, ids, pos)
        if logits is None:  # not the last pipeline stage
            return None
        if (
            env_flag("THUNDERSYNC_SELECT_CHECK")
            and ps.get_pipeline_model_parallel_world_size() == 1
        ):
            # Debug: selected logits vs the full forward (PP=1 only: a second
            # pipeline schedule on the last stage alone would deadlock).
            full = self._score_full(scorer, ids, pos)[select[1], select[0]]
            diff = torch.log_softmax(logits[0].float(), -1) - torch.log_softmax(
                full.float(), -1
            )
            print(
                f"[select check] n={full.shape[0]} max|dlogp|={diff.abs().max().item():.3e}",
                flush=True,
            )
        return logits[0]

    def _score_full(self, scorer, ids, pos):
        """Scorer forward under the MoE device dispatch's capacity guard (learner_graphs.guarded_forward)."""
        from nemo_rl.models.megatron import learner_graphs

        return learner_graphs.guarded_forward(lambda: self._score_full_unguarded(scorer, ids, pos), scorer)

    def _score_full_unguarded(self, scorer, ids, pos):
        from megatron.core import parallel_state as ps

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        seq_len = ids.shape[1]
        m = self.cfg["make_sequence_length_divisible_by"]
        padded = (
            self._batch_invariant
            and ps.get_tensor_model_parallel_world_size() > 1
            and seq_len % m
        )
        if padded:
            # Batch-invariant scorer at TP>1: sequence parallelism splits the
            # sequence over TP ranks, and each rank's norm row count must stay
            # in training's shape class, so pad like training. Causal attention
            # leaves the logits of the real positions unchanged (sliced below).
            padded = -(-seq_len // m) * m
            ids = torch.nn.functional.pad(ids, (0, padded - seq_len))
            pos = torch.arange(padded, device=ids.device).expand(ids.shape[0], -1)
        # On generation workers Megatron's global InferenceMode is active while the
        # engine runs; it requires gathered TP logits from any forward.
        gather_kw = {}
        try:
            from megatron.core.inference.utils import InferenceMode

            if InferenceMode.is_active():
                gather_kw["runtime_gather_output"] = True
        except ImportError:
            pass
        if (
            env_flag("THUNDERSYNC_SPEC_TORCHPROF")
            and not getattr(self, "_profiled", False)
            and ps.get_pipeline_model_parallel_world_size() == 1
        ):
            # Debug: profile one extra forward (PP=1 only: outside the schedule).
            self._profiled = True
            from torch.profiler import ProfilerActivity, profile

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
            ) as prof:
                scorer(
                    input_ids=ids, position_ids=pos, attention_mask=None, **gather_kw
                )
                torch.cuda.synchronize()
            print(
                f"[spec torchprof] ids={tuple(ids.shape)}\n"
                + prof.key_averages(group_by_input_shape=True).table(
                    sort_by="self_cuda_time_total",
                    row_limit=25,
                    max_name_column_width=40,
                    max_shapes_column_width=90,
                ),
                flush=True,
            )
        if ps.get_pipeline_model_parallel_world_size() == 1:
            logits = scorer(
                input_ids=ids, position_ids=pos, attention_mask=None, **gather_kw
            ).float()
        else:
            # Run the pipeline schedule forward-only; logits exist on the last stage.
            from megatron.core.pipeline_parallel import get_forward_backward_func

            captured: list[torch.Tensor] = []

            def fwd_step(data_iterator, model):
                b_ids, b_pos = next(data_iterator)
                out = model(input_ids=b_ids, position_ids=b_pos, attention_mask=None)

                def collect(output_tensor):
                    captured.append(output_tensor.detach().float())
                    return torch.zeros((), device=output_tensor.device), {}

                return out, collect

            get_forward_backward_func()(
                forward_step_func=fwd_step,
                data_iterator=iter([(ids, pos)]),
                model=[scorer],
                num_microbatches=1,
                seq_length=ids.shape[1],
                micro_batch_size=ids.shape[0],
                forward_only=True,
            )
            if not ps.is_pipeline_last_stage(ignore_virtual=True):
                torch.cuda.synchronize()
                self._prof["forward"] = (
                    self._prof.get("forward", 0.0) + time.perf_counter() - t0
                )
                return None
            logits = captured[0]
        tp = ps.get_tensor_model_parallel_world_size()
        if tp > 1 and not gather_kw:
            # Vocab-parallel logits -> full vocab for this batch only (memory is
            # bounded by batch_tokens, not by the number of draft tokens).
            parts = [torch.empty_like(logits) for _ in range(tp)]
            torch.distributed.all_gather(
                parts, logits.contiguous(), group=ps.get_tensor_model_parallel_group()
            )
            logits = torch.cat(parts, dim=-1)
        if padded:
            # Only full-row forwards are padded (batch invariance never selects
            # positions); selected logits are [1, n_selected, V] and kept whole.
            logits = logits[:, :seq_len]
        torch.cuda.synchronize()
        self._prof["forward"] = (
            self._prof.get("forward", 0.0) + time.perf_counter() - t0
        )
        return logits

    def _expert_parallel_lockstep(
        self, plans: list[list[int]]
    ) -> list[list[int]] | None:
        """Common forward schedule of the expert-parallel group, or None without EP.

        ``plans[j]``: stash versions batch j of this rank scores q with (p follows).
        Every scorer forward runs an all-to-all over the EP group, and each rank
        holds only its own experts, so all EP ranks must run the same number of
        forwards with the same weights loaded: batch j scores, in order, the union
        of the ranks' versions for j, and ranks without rows there run a dummy.
        """
        from megatron.core import parallel_state as ps

        if ps.get_expert_model_parallel_world_size() <= 1:
            return None
        gathered: list[Any] = [None] * ps.get_expert_model_parallel_world_size()
        torch.distributed.all_gather_object(
            gathered, plans, group=ps.get_expert_model_parallel_group()
        )
        n = max(len(p) for p in gathered)
        # Forwards this rank scheduled through the lockstep (cumulative; run_checks.lockstep_violations).
        self._lockstep_forwards = getattr(self, "_lockstep_forwards", 0) + n
        return [
            sorted({v for p in gathered if j < len(p) for v in p[j]}) for j in range(n)
        ]

    def _dummy_score(self, scorer) -> None:
        """One-row scorer forward whose output is discarded (EP lockstep filler)."""
        n = self.cfg["make_sequence_length_divisible_by"]
        ids = torch.zeros((1, n), dtype=torch.long, device="cuda")
        pos = torch.arange(n, device="cuda")[None]
        zero = torch.zeros(1, dtype=torch.long, device="cuda")
        self._score(scorer, ids, pos, select=(zero, zero))

    def _my_rows(self, rows, keys) -> list[int]:
        """Indices of the rows this data-parallel rank owns, shortest first.

        Ownership is a stable hash of the draft key, so the tensor-parallel
        ranks of one replica process the same rows together.
        """
        import zlib

        from megatron.core import parallel_state as ps

        dp, dp_rank = ps.get_data_parallel_world_size(), ps.get_data_parallel_rank()
        return sorted(
            (
                i
                for i in range(len(rows))
                if zlib.crc32(keys[i].encode()) % dp == dp_rank
            ),
            key=lambda i: rows[i].numel(),
        )

    def _iter_row_id_batches(self, rows, batch_tokens, keys):
        """This rank's rows, length-sorted, as padded token-id batches.

        Rows are owned by data-parallel rank (stable hash of the key), so the
        tensor-parallel ranks of one replica process the same rows together.
        """
        mine = self._my_rows(rows, keys)
        b = 0
        while b < len(mine):
            e = b + 1
            while e < len(mine) and rows[mine[e]].numel() * (e + 1 - b) <= batch_tokens:
                e += 1
            group = mine[b:e]
            b = e
            s = max(rows[i].numel() for i in group)
            ids = torch.zeros((len(group), s), dtype=torch.long)
            for r, i in enumerate(group):
                ids[r, : rows[i].numel()] = rows[i]
            yield group, ids.cuda()

    @torch.no_grad()
    def scorer_token_logprobs(
        self, rows: list[torch.Tensor]
    ) -> list[torch.Tensor] | None:
        """Debug/validation: next-token logprobs of ``rows`` under the fp32 scorer.

        Uses the current weights; returned on last-pipeline-stage DP-rank-0 workers.
        """
        from megatron.core import parallel_state as ps

        self._prof = {}
        scorer = self._fp32_scorer()
        out = []
        for row in rows:
            ids = row.view(1, -1).cuda()
            pos = torch.arange(ids.shape[1], device="cuda").view(1, -1)
            logits = self._score(scorer, ids, pos)
            if logits is None:
                continue
            lp = torch.log_softmax(logits[0, :-1].float(), -1)
            out.append(lp.gather(-1, ids[0, 1:, None]).squeeze(-1).cpu())
        last = ps.is_pipeline_last_stage(ignore_virtual=True)
        return (
            out
            if (
                last
                and ps.get_data_parallel_rank() == 0
                and ps.get_tensor_model_parallel_rank() == 0
            )
            else None
        )

    @torch.no_grad()
    def model_param_checksum(self) -> float:
        """Debug: sum of the (refit target) model parameters, to compare pools."""
        return float(sum(p.detach().double().sum() for p in self.model.parameters()))

    def debug_refit_export_names(self) -> dict[str, Any]:
        """Debug: names (and a checksum) this rank would send at refit."""
        names, total = [], 0.0
        for name, t in self._iter_params_with_optional_kv_scales(kv_scales=None):
            names.append(name)
            total += float(t.detach().double().sum())
        layers = sorted(
            {int(n.split("layers.")[1].split(".")[0]) for n in names if "layers." in n}
        )
        return {
            "rank": self.rank,
            "n": len(names),
            "layers": (layers[:3], layers[-3:], len(layers)),
            "sum": total,
        }
