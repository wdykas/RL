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
    _WEIGHTS["paused"] = True
    if not _WEIGHTS["checked"]:
        _released_or_raise("the inference weights", free0, _WEIGHTS["bytes"])
        _WEIGHTS["checked"] = True
    t3 = time.perf_counter()
    WEIGHT_OFFLOAD_STATS.update(
        pause_sync_s=t1 - t0,
        keep_copy_s=t2 - t1,
        pause_s=t3 - t2,
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
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for t, cpu in kept:
            t.copy_(cpu)
        torch.cuda.synchronize()
        WEIGHT_OFFLOAD_STATS.update(
            resume_wait_s=t0 - tw, resume_s=t1 - t0, keep_restore_s=time.perf_counter() - t1
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
# Megatron's KV region) are legal.
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
    if not _TEARDOWN["installed"]:
        _TEARDOWN["installed"] = True
        atexit.register(_teardown)
