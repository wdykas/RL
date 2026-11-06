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
import os
import time
import zlib
from typing import Any, Optional

import numpy as np
import ray
import torch

from nemo_rl.data.multimodal_utils import attach_media_token_validity_mask
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron import train as megatron_train
from nemo_rl.models.megatron.data import get_microbatch_iterator
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)
from thundersync_rl import vit
from thundersync_rl.nolanes_engine import KEY_RID_ATTR, NolanesEngineMixin
from thundersync_rl.xspr_engine import XsprEngineMixin
from thundersync_rl.envflag import env_flag
from thundersync_rl.debug_hooks import verify_no_bi as _debug_verify_no_bi

WORKER_FQN = "thundersync_rl.worker.ThunderSyncMegatronPolicyWorker"

# Learner-side state of one engine lane's refit communicators (cohort lanes): the
# packed collective's ``model_update_group`` and the native MCore (m2n) refit's
# process group, copy service and destination rank offset (refit_transport mcore).
_LANE_REFIT_ATTRS = (
    "model_update_group",
    "refit_pg",
    "refit_copy_service",
    "refit_dst_rank_offset",
)




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
    if not isinstance(cfg, dict):
        return False
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


class ChunkIdGuard:
    """CHECKS guard for the partial-chunk skip (_hidden_log_rows / _key_check_positions): the live
    ``chunked_prefill_request_id`` read at sampling time must equal the id MCore later reports as consumed by that
    forward (``consumed_chunked_prefill_request_id`` of ``_process_request_outputs``, the engine's pre-step snapshot).
    ``seen`` records the value each sampling call read; ``check`` compares the latest one with the consumed id once
    per processed step and raises on a difference."""

    def __init__(self) -> None:
        self.last: Optional[int] = None
        self.checked = 0

    def seen(self, cid: int) -> None:
        self.last = int(cid)

    def check(self, consumed: int) -> None:
        if self.last is None:
            return
        live, self.last = self.last, None
        self.checked += 1
        if self.checked % 200 == 1:
            print(f"[chunk id guard] checked={self.checked} (live id == consumed id)", flush=True)
        if live != int(consumed):
            raise RuntimeError(
                f"[chunk id guard] the sampling hook read chunked_prefill_request_id {live}, MCore consumed "
                f"{int(consumed)}: a partial-chunk row would be mis-attributed (hidden log / key checks)"
            )


def install_chunk_id_guard(engine: Any) -> "ChunkIdGuard":
    """Wrap ``engine._process_request_outputs`` with the guard's check (idempotent per engine)."""
    guard = getattr(engine, "_ts_chunk_id_guard", None)
    if guard is not None:
        return guard
    import inspect

    guard = ChunkIdGuard()
    orig = engine._process_request_outputs
    name = "consumed_chunked_prefill_request_id"
    params = list(inspect.signature(orig).parameters)
    if name not in params:
        raise RuntimeError(f"[chunk id guard] {type(engine).__name__}._process_request_outputs has no {name}")
    pos = params.index(name)

    def _process_request_outputs(*args, **kwargs):
        # The consumed id, passed by keyword or by position (never silently absent).
        guard.check(kwargs[name] if name in kwargs else args[pos])
        return orig(*args, **kwargs)

    engine._process_request_outputs = _process_request_outputs
    engine._ts_chunk_id_guard = guard
    return guard


def _hidden_log_rows(context: Any, tag: np.ndarray, rids: np.ndarray, a: int, e: int) -> list[int]:
    """Active rows (relative to ``a``) whose hidden state and version tag the draft hidden log records: this rank's
    tagged rows (``tag == 1``) except a partial prefill chunk's row. MCore samples that row and discards the draw (more
    prompt tokens follow, ``context.chunked_prefill_request_id``); recording it would put a row at the draft's index 0
    that only the final chunk's record overwrites (_key_check_positions skips the same row).

    Reading the live ``chunked_prefill_request_id`` at sampling time gives the sampled forward's value: the engine
    schedules (the only writer) before a forward, and under async scheduling the controller samples the pending
    forward before it admits the next one (no-overlap path: sample -> update -> schedule -> forward); an overlap step
    admits nothing (``_should_run_async_sched_overlap`` is False while the queue head can be scheduled)."""
    rows = np.flatnonzero(tag[a:e] == 1)
    cid = int(getattr(context, "chunked_prefill_request_id", -1))
    if cid >= 0 and rows.size:
        rows = rows[rids[a + rows] != cid]
    return rows.tolist()


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


def _request_keys(engine: Any, ids: np.ndarray, lens: "_RequestLengths") -> np.ndarray:
    """Noise-key id of each engine request: its ``KEY_RID_ATTR`` (exact SPR's row key, the same for every request
    that carries the row: decode, check and continuation); -1 where a request has none. Reads the cache entries
    ``_request_positions`` filled for the same call."""
    out = np.full(ids.shape, -1, dtype=np.int64)
    for j, rid in enumerate(ids.tolist()):
        c = lens.get(engine, rid)
        if c is not None and c[6] is not None:
            out[j] = int(c[6])
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
        """(cur, cur prompt, len, first prompt, len, hidden key or None, noise-key override or None) of request
        ``rid``; None if unknown."""
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
                getattr(cur.sampling_params, KEY_RID_ATTR, None),
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


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker"),
    # Above the driver's engine row slots + control calls (engine_slots.py): Ray's async-actor default (1000) queued
    # continuations behind 1008 drafts at Super-L22 wake_verify.
    max_concurrency=_ACTOR_MAX_CONCURRENCY,
)  # pragma: no cover
class ThunderSyncMegatronPolicyWorker(XsprEngineMixin, NolanesEngineMixin, MegatronPolicyWorkerImpl):
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
        # policy.offload_optimizer_for_refit=false; offload_before_refit below keeps the grad buffers where
        # megatron_cfg.resident_learner is set). Expandable segments are
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
        # Bumped whenever the learner's parameters change (optimizer step, draft
        # weight swap): state cached from a forward (learner prefix-state cache)
        # is valid only for the version it was computed at.
        self._weights_version = 0
        if getattr(self, "optimizer", None) is not None:
            step = self.optimizer.step

            def step_and_bump(*args: Any, **kwargs: Any) -> Any:
                try:
                    return step(*args, **kwargs)
                finally:
                    self._weights_version += 1

            self.optimizer.step = step_and_bump
            # Learner only (engines have no optimizer). Already set by the pre-load hook when a checkpoint
            # (pretrained or resume) was loaded; idempotent.
            self._expandable_segments = False if colocated else enable_learner_expandable_segments()
        self._prefix_cache = None


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
        shapes = [tuple(d["input_ids"].shape) for d in datas if "input_ids" in d]
        if shapes:  # largest micro-batch (rows x padded length) this step: the activation term of the peak
            self._step_max_mb_tokens = max(
                getattr(self, "_step_max_mb_tokens", 0), max(int(s[0]) * int(s[-1]) for s in shapes)
            )
        return super().train_microbatches(datas, *args, **kwargs)

    @_lg_feature("logprob")
    def get_logprobs(self, *, data, **kwargs):
        return super().get_logprobs(data=data, **kwargs)


    def finish_train_step(self) -> dict[str, Any]:
        """Finish the train step; optionally perturb expert gradients (test only).

        ``THUNDERSYNC_DEBUG_EXPERT_GRAD_NOISE=<scale>`` adds seeded noise to the MoE
        expert gradients before each optimizer step, so expert weights change even when
        every reward is 0 (random tiny models). Exercises refit into the engine's expert
        weights: any stale engine weight shows up in the mismatch check.
        """
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
        t0 = time.perf_counter()
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
            busy[method] = busy.get(method, 0.0) + time.perf_counter() - t0
            self._ordered_next += 1
            # What this rank ran, in order (``ordered_digest``).
            self._ordered_crc = zlib.crc32(
                f"{seq}:{method};".encode(), getattr(self, "_ordered_crc", 0)
            )
            async with cv:
                cv.notify_all()


    # -- refit-in-flight guard (research/ray_order_audit.md, "latent: mid-refit window") --
    # The refit transfers below are ``async def`` and run off the actor's event loop
    # (``await_off_loop``), so the actor serves other calls meanwhile. Calls that read the
    # live weights (head stash, engine verification, weight digests) must never run then:
    # today the driver's schedule guarantees it; this turns a regression into an error.
    def _refit_guard_enter(self) -> None:
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
        mcfg = (getattr(self, "cfg", None) or {}).get("megatron_cfg") or {}
        if getattr(self, "is_generation_colocated", False) and mcfg.get("resident_learner", False):
            # Resident colocated learner (policy.megatron_cfg.resident_learner, set by the modes whose stages read the
            # learner after the stand-down: nolanes verify, colo): grad buffers stay on the device across the engine
            # phase. Every other mode (xspr, isspr) offloads as stock / SPR-BI does.
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


    def _log_mem(self, where: str) -> None:
        if env_flag("THUNDERSYNC_MEM_LOG"):
            print(
                f"[thundersync mem] rank={self.rank} {where}: "
                f"alloc={torch.cuda.memory_allocated() / 2**30:.1f}G "
                f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}G",
                flush=True,
            )
            torch.cuda.reset_peak_memory_stats()


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
            if token_to_request_index is not None or not (no_top_k and no_top_p):
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
            if self._xspr_on():
                # Exact SPR: every row is keyed by its row key (xspr_engine.py), never by the engine request id.
                h_rids = _request_keys(engine, h_rids, lens)
                if (h_rids < 0).any():
                    raise RuntimeError("[xspr] a request without a row key reached the keyed sampler")
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

    def termination_id(self) -> int | None:
        tok = getattr(self, "megatron_tokenizer", None)
        return None if tok is None else int(tok.eod)

    # ---- Deadline-truncated streaming drafts (generation rank 0) ----


    def _own_client_loop_wanted(self) -> bool:
        """Rank 0's inference client (draft streams, their frames, aborts, fresh-row requests) on its own event
        loop (``THUNDERSYNC_CLIENT_LOOP=0``: on the engine loop, as before). Global rank 0 is a TP rank 0 of a lane
        engine, the rank its decode loop waits on; the client's coroutines no longer interleave with its engine
        steps. One client, same socket and request-id counter: requests and their ids are unchanged."""
        return os.environ.get("THUNDERSYNC_CLIENT_LOOP", "1") != "0"







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
                # Engine-side check capture (exact / IS-SPR check requests, xspr_engine): every forward's head-input
                # rows are copied before the last-token selection (captured in the graphs).
                worker._install_verify_capture(context)
                super().__init__(controller, context, *args, **kwargs)
                _install_one_tolist_log_probs(controller)
                # Exact top-p: innermost sampler wrapper (installed before the
                # draft logs wrap the kernel), so they see the raw logits.
                worker._install_nucleus_sampler(controller)


        return _Engine

    # ---- Engine head-input capture (policy.megatron_cfg.engine_verify; engine_verify.py) ----
    # Exact / IS-SPR check requests (xspr_engine.XsprEngineMixin) read the captured head-input rows of a check's
    # prefill from the static buffer.

    def _install_verify_capture(self, context) -> None:
        """Copy every engine forward's head-input rows before the last-token selection.

        ``policy.megatron_cfg.engine_verify``. With
        ``materialize_only_last_token_logits`` the model selects each request's
        last row (``context.last_token_logits``) before the output layer; a
        wrapper on that method first copies all [padded tokens, d] rows into a
        static buffer (head-input dtype, as ``_install_hidden_copy`` rounds
        them), so the copy is captured into the CUDA graphs and replays with
        them. ``xspr_engine`` reads the rows of its check requests from the
        buffer after each forward. Last pipeline stage only (the output layer);
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
        from nemo_rl.models.megatron.memory_saver import engine_scratch_region

        # Engine scratch (released during colocated training with the TMS=1 overlay, like the dk row workspace):
        # rewritten by every forward before it is read, so no contents survive a pause.
        with engine_scratch_region():
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
                elif torch.cuda.is_current_stream_capturing():
                    # A graph captured without the copy would replay without it and never count an overflow.
                    raise RuntimeError(f"verify capture: a CUDA graph of {rows.shape[0]} rows exceeds the "
                                       f"{static.shape[0]}-row head-input buffer")
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

    def _build_sampling_params(self, greedy, stop_words, **kwargs):
        """MCore sampling params; under the exact top-p policy Megatron's own
        top-p is off (top_p 0): the nucleus is masked by ``_install_nucleus_sampler``."""
        sp = super()._build_sampling_params(greedy, stop_words, **kwargs)

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


    # ---- Anchor horizon: sticky anchors applied by the engine (theta_k) ----






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
        # exact SPR: the row key / check of every xspr request (xspr_engine.py).
        self._xspr_request_attrs(data, sps)
        keys = data.get("ts_anchor_key") if data is not None else None
        if keys:
            from thundersync_rl.engine_q import ANCHOR_KEY_ATTR

            for sp, key in zip(sps, keys):
                if key:
                    sp.add_attributes({ANCHOR_KEY_ATTR: key})
        return prompts, media, sps



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

        # CHECKS: the live mid-prefill request id read here must be the one MCore reports consumed for the forward.
        chunk_guard = (
            install_chunk_id_guard(self.dynamic_inference_engine)
            if os.environ.get("THUNDERSYNC_NL_CHECKS") == "1"
            and hasattr(self.dynamic_inference_engine, "_process_request_outputs")
            else None
        )

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
                if chunk_guard is not None:
                    chunk_guard.seen(int(getattr(context, "chunked_prefill_request_id", -1)))
                for row in _hidden_log_rows(context, tag, rids, a, e):
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


        # Under async scheduling the controller launches the next forward
        # before finished requests are resolved, then compacts the surviving
        # rows and reorders the pending logits to match
        # (``_compact_async_sched_forward``; ``_compact_async_sched_logits`` before
        # Megatron-LM main). Apply the same permutation to the hidden-copy buffer,
        # on the same stream, so row i of the buffer stays the hidden state of
        # logits row i.
        controller = engine.controller
        compact_name = (
            "_compact_async_sched_forward"
            if hasattr(controller, "_compact_async_sched_forward")
            else "_compact_async_sched_logits"
        )
        orig_compact = getattr(controller, compact_name)

        def compact(survivor_idxs, *args, **kwargs):
            n = survivor_idxs.numel()
            if n and not torch.equal(
                survivor_idxs, torch.arange(n, device=survivor_idxs.device)
            ):
                static[:n].copy_(static[survivor_idxs.to(static.device)])
            return orig_compact(survivor_idxs, *args, **kwargs)

        setattr(controller, compact_name, compact)
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


    # ---- Engine-side verification (thundersync.verify_on=engine; engine_verify.py) ----
    # The engine holding theta_k prefills drafted rows (one discarded sampled
    # token each), keeps their LM-head input rows and decides block
    # verification itself; p never leaves the engine rank that computed it.


    # ---- q-row recompute on the anchor engine (hidden_transport=recompute, verify_on=learner) ----





    def engine_verify_drop(self, keys: list[str]) -> None:
        """Forget captured rows of these keys (a failed or abandoned call)."""
        cap = getattr(self, "_verify_cap", None)
        if cap is None:
            return
        with cap["lock"]:
            cap["store"].drop(keys)
            for k in keys:
                cap["events"].pop(k, None)






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

        from nemo_rl.models.megatron import exact_nucleus
        from nemo_rl.models.megatron import exact_int as xi_law
        from thundersync_rl import ds2 as _ds2
        from thundersync_rl.block_verification_fused import _d2h, _h2d, devscan_enabled
        from thundersync_rl.prof import pop as prof_pop
        from thundersync_rl.prof import timed

        if xi_law.active():
            raise RuntimeError("the exact integer law (engine_sampler=exact_int, exact_bv) is not on the colocated branch")
        exact_law = False
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
        cache = None
        starts = dict.fromkeys(mine, 0)
        # Several drafts per training-shaped forward (padded as training pads a
        # micro-batch): batch-invariant kernels give each row the same logits
        # as a one-row forward, and per-call overhead dominates small rows.
        batches = self._row_batches([rows[i][starts[i] :] for i in mine], batch_tokens)
        batches = [[mine[j] for j in b] for b in batches]
        with timed("v_lockstep"):
            lockstep = self._expert_parallel_lockstep([[] for _ in batches])
        n_dummy = 0 if lockstep is None else len(lockstep) - len(batches)

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
                t0 = time.perf_counter()
                if "v_setup" not in vp:
                    vp["v_setup"] = t0 - t_all  # call entry -> first forward launch (host)
                from nemo_rl.models.megatron import bi_context_parallel as _cpx

                with timed("v_forward"), verification_logits(), _cpx.gather_outputs():  # CP: whole rows on every CP rank
                    if len(batch) == 1:
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
            # Phase 2 only: rows (per segment) fetched when the scan asks.
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
                        if _ds2.eligible(
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

                    if law_on and _emit_sampled(keys[i], emit_every):
                        # ⊥-BV emission law: PIT of the token / BOT emitted at every envelope M under
                        # the ⊥-BV law given (draft, tau_M, M), recomputed from the definition; the
                        # driver keeps the PIT of the M the pool's counts decide the row at.
                        res["null_grid_pit"] = self._null_emission_pits(
                            p_lp, draft, lq, q_rows, grid, seed, keys[i]
                        )
                    n_rows += n_eval
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
            if i in reforward:
                res["prefix_reforward_s"], res["prefix_reforward_tokens"] = reforward[i]
            results.append((i, {**res, "hidden_bytes_fetched": fetched, **fetch_info}))
            del p_lp, h
        if pf is not None:
            self._prof["hidden_prefetched_bytes"] = pf.submitted_bytes
            self._prof["hidden_prefetch_skipped"] = pf.skipped_batches
            pf.close()
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

    def _padded_width(self, rows: list[torch.Tensor]) -> int:
        """Micro-batch width of ``rows`` in ``_training_forward_rows`` (``make_sequence_length_divisible_by``)."""
        m = self.cfg["make_sequence_length_divisible_by"]
        return -(-max(row.numel() for row in rows) // m) * m


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


    @property
    def _batch_invariant(self) -> bool:
        """Whether the scorer runs batch-invariant kernels (bf16 scorer under BI).

        An fp32 scorer is built outside batch-invariant mode, so training-parity
        measures (full-row scoring, training's log-softmax) do not apply to it.
        """
        return bool(self.cfg["megatron_cfg"].get("batch_invariant_mode")) and (
            getattr(self, "_scorer_precision", "fp32") != "fp32"
        )

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



