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

import atexit
import os
import warnings
from contextlib import nullcontext
from typing import Any, ContextManager, Sequence

try:
    from torch_memory_saver import (  # pyrefly: ignore[import-error]
        torch_memory_saver,
    )

    torch_memory_saver.hook_mode = "torch"

    HAVE_TORCH_MEMORY_SAVER = True
except ImportError:
    HAVE_TORCH_MEMORY_SAVER = False

# The torch_memory_saver release the overlay pins (the research notes, summarized in docs/colo/); the teardown hook below
# uses its region flag call directly.
TMS_PINNED_VERSION = "0.0.10"


def tms_version_problem() -> str:
    """'' when the importable torch_memory_saver is the pinned release, else the problem (also '' without it)."""
    if not HAVE_TORCH_MEMORY_SAVER:
        return ""
    from importlib.metadata import PackageNotFoundError, version

    try:
        v = version("torch_memory_saver")
    except PackageNotFoundError:
        v = "unknown"
    if v != TMS_PINNED_VERSION:
        return f"torch_memory_saver {v} is not the pinned {TMS_PINNED_VERSION} (the TMS=1 overlay)"
    return ""


# torch_memory_saver region tag for the colocated inference model's weights.
_INFERENCE_MODEL_OFFLOAD_TAG = "nemo_rl_megatron_inference_model"


def reshard_rewrites_weights() -> bool:
    """Whether every colocated wake rewrites the inference model's refit tensors (all parameters and persistent
    buffers: MCore ``named_refit_tensors``) with the training weights. False only under the negative control
    ``NEMO_RL_DEBUG_COLOC_SKIP_RESHARD=1``, whose engine keeps its first weights."""
    return os.environ.get("NEMO_RL_DEBUG_COLOC_SKIP_RESHARD") != "1"


def inference_model_alloc_region(cpu_backup: bool = True) -> ContextManager[None]:
    """Allocation region to build the colocated inference model under.

    Returns a torch_memory_saver region, or a null context. ``cpu_backup``: the region's contents are copied to
    CPU at every pause and back at every resume. Not needed when every wake reshards fresh weights into the model
    (``reshard_rewrites_weights``): the refit tensors are rewritten after the resume anyway, and the region's other
    tensors (``account_inference_region``) are kept by ``pause_inference_weights``.
    """
    if HAVE_TORCH_MEMORY_SAVER:
        _install_teardown()
        return torch_memory_saver.region(
            tag=_INFERENCE_MODEL_OFFLOAD_TAG, enable_cpu_backup=cpu_backup
        )
    warnings.warn(
        "torch_memory_saver is unavailable; the colocated inference model will stay "
        "GPU-resident alongside the training model (higher peak memory). Install "
        "torch_memory_saver to enable inference-weight offload.",
        stacklevel=2,
    )
    return nullcontext()


def active_blocks(device: int) -> dict[int, int]:
    """{address: requested bytes} of the caching allocator's live blocks on ``device`` (every pool)."""
    import torch

    out = {}
    for seg in torch.cuda.memory_snapshot():
        if seg.get("device") != device:
            continue
        addr = seg["address"]
        for b in seg["blocks"]:
            if b["state"] == "active_allocated":
                out[addr] = b.get("requested_size", b["size"])
            addr += b["size"]
    return out


def _reachable_tensors(module: Any) -> list:
    """Every CUDA tensor the module tree holds: parameters, buffers and tensor attributes (also inside a list, tuple
    or dict attribute)."""
    import torch

    seen, out = set(), []

    def add(t):
        if isinstance(t, torch.Tensor) and t.is_cuda and id(t) not in seen:
            seen.add(id(t))
            out.append(t)

    for m in module.modules():
        for t in m._parameters.values():
            add(t)
        for t in m._buffers.values():
            add(t)
        for k, v in vars(m).items():
            if k in ("_parameters", "_buffers", "_modules"):
                continue
            if isinstance(v, (list, tuple)):
                for x in v:
                    add(x)
            elif isinstance(v, dict):
                for x in v.values():
                    add(x)
            else:
                add(v)
    return out


def account_inference_region(module: Any, build_blocks: dict[int, int]) -> tuple[list, int]:
    """The tensors the no-backup offload must keep, and the region's bytes.

    ``build_blocks``: the allocator blocks the build created (``active_blocks`` after minus before), i.e. the region.
    Every block must hold a tensor reachable from the model; the refit tensors (MCore ``named_refit_tensors``) are
    rewritten by every wake's reshard, every other reachable tensor in the region is kept (copied to CPU at the
    pause). A block holding no reachable tensor would come back as garbage after a pause: refused, listed.
    """
    from megatron.core.resharding.utils import named_refit_tensors

    starts = sorted(build_blocks)
    import bisect

    def block_of(ptr: int):
        i = bisect.bisect_right(starts, ptr) - 1
        if i >= 0 and ptr < starts[i] + max(build_blocks[starts[i]], 1):
            return starts[i]
        return None

    refit_blocks = {block_of(t.data_ptr()) for _, t in named_refit_tensors(module)}
    covered, keep = set(), []
    for t in _reachable_tensors(module):
        b = block_of(t.untyped_storage().data_ptr())
        if b is None:
            continue
        covered.add(b)
        if b not in refit_blocks:
            keep.append(t)
    unknown = [(a, n) for a, n in build_blocks.items() if a not in covered]
    if unknown:
        raise RuntimeError(
            f"inference model region: {len(unknown)} allocations ({sum(n for _, n in unknown) / 2**20:.1f} MiB; "
            f"sizes {[n for _, n in unknown[:8]]}) hold no tensor reachable from the model's parameters, buffers or "
            "tensor attributes: released without a CPU backup they would come back as garbage"
        )
    return keep, sum(build_blocks.values())


def consolidate_into_arena(module: Any, build_blocks: dict[int, int]) -> tuple[int, list]:
    """Move every tensor the module tree holds in ``build_blocks`` (the allocator blocks its build created, in the
    default pool) into one allocation made in the current region.

    torch_memory_saver maps / unmaps each of its allocations with its own driver calls, ~2.6 ms per allocation inside a
    colocated worker (cuMemCreate + cuMemMap + cuMemSetAccess at resume, cuMemUnmap + cuMemRelease at pause; nsys,
   ): the inference model built inside the region came to 484 pool segments, 1.27 s per resume and 0.33 s per
    pause, while the engine scratch (3 allocations, larger) resumed in 15 ms. Built outside the region and moved
    here, the weights are one allocation. Each storage is copied into a 512 B aligned slice of the arena and every
    tensor on it re-pointed (``.data``) at the same offset, size and stride: values, dtypes, views and Parameter
    objects are unchanged. The original storages go back to the default caching allocator.

    Returns (arena bytes, the moved tensors the no-backup offload must keep: those on a storage no refit tensor --
    MCore ``named_refit_tensors``, rewritten by every wake's reshard -- lives on, build blocks no reachable tensor
    holds: (count, bytes), left resident in the default pool). Refused: a move after the engine captured CUDA graphs
    (they hold the weights' addresses), and a moved storage something outside the module tree still holds (its
    block stays allocated after the move: twice the memory, and a holder reading stale weights after a reshard).
    """
    import bisect

    import torch
    from megatron.core.resharding.utils import invalidate_refit_tensor_cache, named_refit_tensors
    from megatron.core.transformer.cuda_graphs import _CudagraphGlobalRecord

    if getattr(_CudagraphGlobalRecord, "cudagraph_inference_record", None):
        raise RuntimeError("consolidate_into_arena after inference CUDA graphs were captured: they hold the old addresses")
    starts = sorted(build_blocks)

    def in_region(ptr: int) -> bool:
        i = bisect.bisect_right(starts, ptr) - 1
        return i >= 0 and ptr < starts[i] + max(build_blocks[starts[i]], 1)

    groups: dict[int, list] = {}
    for t in _reachable_tensors(module):
        sp = t.untyped_storage().data_ptr()
        if in_region(sp):
            groups.setdefault(sp, []).append(t)
    moved_blocks = {starts[bisect.bisect_right(starts, sp) - 1] for sp in groups}
    unmoved = [b for b in starts if b not in moved_blocks]
    unmoved_info = (len(unmoved), sum(build_blocks[b] for b in unmoved))
    if not groups:
        return 0, [], unmoved_info
    refit_storages = {t.untyped_storage().data_ptr() for _, t in named_refit_tensors(module)}
    keep = [t for sp, ts in groups.items() if sp not in refit_storages for t in ts]
    layout, total = [], 0
    for sp, ts in groups.items():
        nbytes = ts[0].untyped_storage().nbytes()
        layout.append((total, nbytes, ts))
        total += -(-nbytes // 512) * 512
    device = next(iter(groups.values()))[0].device
    arena = torch.empty(total, dtype=torch.uint8, device=device)
    storage = arena.untyped_storage()
    with torch.no_grad():
        for off, nbytes, ts in layout:
            src = torch.empty(0, dtype=torch.uint8, device=device).set_(ts[0].untyped_storage())
            arena[off : off + nbytes].copy_(src)
            for t in ts:
                item = t.element_size()
                new = torch.empty(0, dtype=t.dtype, device=device).set_(
                    storage, off // item + t.storage_offset(), t.size(), t.stride()
                )
                t.data = new
    del src
    torch.cuda.synchronize()
    invalidate_refit_tensor_cache(module)  # the refit planner re-reads the re-pointed tensors
    held = [a for a in active_blocks(device.index) if a in moved_blocks]
    if held:
        raise RuntimeError(
            f"consolidate_into_arena: {len(held)} moved storages ({sum(build_blocks[a] for a in held) / 2**20:.1f} MiB) "
            "are still allocated after the move: something outside the module tree holds them"
        )
    return total, keep, unmoved_info


# Inference-weight region of this process: expected bytes, first pause checked; last pause / resume phase seconds.
_WEIGHTS = {"bytes": 0, "checked": False, "paused": False}
WEIGHT_OFFLOAD_STATS: dict = {}


def set_inference_region_bytes(nbytes: int) -> None:
    _WEIGHTS["bytes"] = int(nbytes)


def _released_or_raise(what: str, free_before: int, expected: int) -> None:
    import torch

    freed = torch.cuda.mem_get_info()[0] - free_before
    if expected and freed < 0.9 * expected:
        raise RuntimeError(
            f"torch_memory_saver is active but pausing {what} released {freed / 2**30:.2f} of "
            f"{expected / 2**30:.2f} GiB on device {torch.cuda.current_device()}: TMS requested but not functional"
        )


def pause_inference_weights(keep: Sequence[Any] = ()) -> list:
    """Release the colocated inference weights (no-op without torch_memory_saver).

    ``keep`` (a region built without CPU backup): tensors whose contents must survive, copied to CPU first. Returns
    what ``resume_inference_weights`` restores. The first pause of the process checks that the region's memory was
    released. Phase times in ``WEIGHT_OFFLOAD_STATS``.
    """
    if not HAVE_TORCH_MEMORY_SAVER:
        return []
    import time

    import torch

    t0 = time.perf_counter()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    kept = [(t, t.to("cpu", copy=True)) for t in keep]
    t2 = time.perf_counter()
    free0 = torch.cuda.mem_get_info()[0] if not _WEIGHTS["checked"] else 0
    torch_memory_saver.pause(_INFERENCE_MODEL_OFFLOAD_TAG)
    tc = time.perf_counter()
    _WEIGHTS["paused"] = True
    if not _WEIGHTS["checked"]:
        _released_or_raise("the inference weights", free0, _WEIGHTS["bytes"])
        _WEIGHTS["checked"] = True
    t3 = time.perf_counter()
    WEIGHT_OFFLOAD_STATS.update(
        pause_sync_s=t1 - t0,
        keep_copy_s=t2 - t1,
        pause_s=t3 - t2,
        pause_call_s=tc - t2,
        keep_n=len(kept),
        keep_bytes=sum(c.numel() * c.element_size() for _, c in kept),
    )
    return kept


def resume_inference_weights(kept: Sequence[tuple[Any, Any]] = ()) -> None:
    """Map the inference weights back at their GPU addresses and restore ``kept`` (no-op without
    torch_memory_saver). Without CPU backup the refit tensors hold garbage until the next reshard."""
    if HAVE_TORCH_MEMORY_SAVER:
        import time

        import torch

        tw = time.perf_counter()
        torch.cuda.synchronize()  # work queued before the wake (training) is not the resume's
        t0 = time.perf_counter()
        torch_memory_saver.resume(_INFERENCE_MODEL_OFFLOAD_TAG)
        _WEIGHTS["paused"] = False
        tc = time.perf_counter()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for t, cpu in kept:
            t.copy_(cpu)
        torch.cuda.synchronize()
        WEIGHT_OFFLOAD_STATS.update(
            resume_wait_s=t0 - tw,
            resume_s=t1 - t0,
            resume_call_s=tc - t0,
            keep_restore_s=time.perf_counter() - t1,
        )


# torch_memory_saver region tag for the colocated engine's scratch (buffers whose contents never outlive one forward,
# e.g. fp16_moe_dk's expert row workspace): released during training without a CPU backup, re-mapped at the same
# virtual addresses (the engine's CUDA graphs captured them) before the engine resumes.
_ENGINE_SCRATCH_TAG = "nemo_rl_engine_scratch"
_ENGINE_SCRATCH = {"used": False, "paused": False}


def engine_scratch_region() -> ContextManager[None]:
    """Allocation region for engine scratch (a pausable torch_memory_saver region, else a null context)."""
    if HAVE_TORCH_MEMORY_SAVER:
        _install_teardown()
        _ENGINE_SCRATCH["used"] = True
        return torch_memory_saver.region(tag=_ENGINE_SCRATCH_TAG, enable_cpu_backup=False)
    return nullcontext()


def check_engine_scratch_released(expected: int) -> None:
    """Pause and resume the engine scratch once (before any graph captures it; no contents to keep) and refuse a
    torch_memory_saver that does not return its physical memory."""
    import torch

    torch.cuda.synchronize()
    free0 = torch.cuda.mem_get_info()[0]
    torch_memory_saver.pause(_ENGINE_SCRATCH_TAG)
    try:
        _released_or_raise("the engine scratch", free0, expected)
    finally:
        torch_memory_saver.resume(_ENGINE_SCRATCH_TAG)


def pause_engine_scratch() -> None:
    """Release the engine scratch's physical memory (contents dropped; no-op without a region)."""
    if _ENGINE_SCRATCH["used"] and not _ENGINE_SCRATCH["paused"]:
        import torch

        torch.cuda.synchronize()  # no engine kernel may still use it when it is unmapped
        torch_memory_saver.pause(_ENGINE_SCRATCH_TAG)
        _ENGINE_SCRATCH["paused"] = True


def resume_engine_scratch() -> None:
    """Map the engine scratch back at its addresses (no-op unless paused)."""
    if _ENGINE_SCRATCH["paused"]:
        torch_memory_saver.resume(_ENGINE_SCRATCH_TAG)
        _ENGINE_SCRATCH["paused"] = False


# Process teardown. torch_memory_saver (torch hook mode) exits the process with status 1 when one of its allocations
# is freed outside a region ("only support interesting region"), which interpreter shutdown does to every region
# tensor still alive. At exit, map back what we paused (a paused allocation must not be freed unmapped), drop our
# references, and leave a region active for the rest of the shutdown so the remaining frees (the inference model,
# Megatron's KV region) are legal. The region flag is per thread and set on the thread running the atexit hooks,
# the main thread, which also runs module finalization. The worker's other threads (the inference and client event
# loops) are daemon threads: once finalization starts CPython never runs them again, so no region tensor is freed
# from them during shutdown.
_TEARDOWN = {"installed": False}


def _teardown() -> None:
    if _ENGINE_SCRATCH["paused"]:
        resume_engine_scratch()
    if _WEIGHTS["paused"]:
        torch_memory_saver.resume(_INFERENCE_MODEL_OFFLOAD_TAG)
        _WEIGHTS["paused"] = False
    try:
        from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext

        torch_memory_saver.resume(DynamicInferenceContext.TMS_TAG)
    except Exception:  # noqa: BLE001 - no Megatron context / tag never paused
        pass
    # From here on every free counts as inside a region, for the rest of the shutdown (module and static
    # destructors included): set the library's per-thread region flag directly (torch_memory_saver 0.0.10, pinned by
    # the TMS overlay; a region() context manager would be closed again when its generator is collected).
    torch_memory_saver._ensure_initialized()
    torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(True)
    from nemo_rl.models.megatron import fp16_moe_dk

    fp16_moe_dk.release_row_workspaces()


def _install_teardown() -> None:
    problem = tms_version_problem()
    if problem:
        raise RuntimeError(problem)
    if not _TEARDOWN["installed"]:
        _TEARDOWN["installed"] = True
        atexit.register(_teardown)
