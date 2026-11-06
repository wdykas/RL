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
"""Engine KV / state offload proportional to the live state, not to the buffer's capacity.

With ``kv_cache_management_mode: offload`` and torch_memory_saver, Megatron-Core allocates the engine's KV cache
and Mamba state buffers in a torch_memory_saver region with a CPU backup: every suspend copies the whole buffer to
the host and every resume copies it back (Nano 8k, 20 GiB per rank: 0.7-1.4 s per suspend, 0.27 s per resume;
colocated step +2.4 s against KV persist, research/colo RESULTS.md). Only the state of the requests still resident
at the suspend has to survive (paused / in-flight rows, and the prefix cache's blocks):

- KV: every block not in the allocator's free stack (``block_bag[:pool_avail]``): the requests' blocks, the dummy
  block and the prefix cache's cached blocks;
- Mamba conv / SSM (and speculative intermediate) states: every slot not in the free-slot stack.

``install()`` gives the context's region no CPU backup and, around Megatron's own pause / resume of it, copies
exactly those blocks and slots to the host and back (bitwise). Every other tensor allocated in the region is copied
whole; a region allocation that no context tensor holds is refused at init. Per rank per suspend:
live blocks x (block bytes over every KV plane) + live slots x (state bytes per slot); 0 when no request is
resident (a colocated step that finished its rollouts), the whole buffer only when it is full.
"""

from __future__ import annotations

import contextlib
import weakref
from typing import Any

import torch

_STATE: dict = {"installed": False, "ctx": None, "blocks": {}, "saved": None, "stats": {}}
# Components that hold KV blocks / Mamba slots outside the requests' tables (e.g. a reserved private block pool):
# name -> callable returning the ids they hold now. They count as owners in the suspend's live-set check.
_BLOCK_OWNERS: dict = {}
_SLOT_OWNERS: dict = {}


def register_block_owner(name: str, held_ids) -> None:
    """Register ``held_ids() -> iterable of KV block ids`` held outside the requests' block tables."""
    _BLOCK_OWNERS[name] = held_ids


def register_slot_owner(name: str, held_ids) -> None:
    """Register ``held_ids() -> iterable of Mamba state slots`` held outside the requests' slot map."""
    _SLOT_OWNERS[name] = held_ids


def _block_hashes(alloc: Any):
    """The prefix cache's per-block hashes (-1: not cached), or None without a prefix cache. Megatron-LM main keeps
    them on ``alloc.pc_state`` (PrefixCacheBlockState), the colo pin on the allocator itself."""
    if hasattr(alloc, "pc_state"):
        return None if alloc.pc_state is None else alloc.pc_state.block_hashes
    return getattr(alloc, "block_hashes", None)


def _live_index(ctx: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """(live KV block ids, live Mamba slot ids) as device int64 tensors: the complements of the free stacks.

    Megatron's free pools are stacks (pop: ``avail -= n``, push: ``bag[avail:avail + n] = released``); releases come
    in any order, so the entries above the free count are NOT the allocated set (a push overwrites positions whose
    old ids are still allocated). Live = every id not in the free part ``bag[:avail]``: the requests' blocks (active
    and paused), the dummy block, and the prefix cache's cached blocks (outside the free stack until evicted).
    Checked against the owners: the live blocks must be exactly the resident requests' block tables, the prefix
    cache's cached blocks and the dummy block, and the live slots exactly the resident requests' Mamba slots.
    """
    dev = torch.cuda.current_device()
    alloc = ctx.kv_block_allocator
    free_ids = alloc.block_bag[: alloc.pool_avail].to(torch.int64)
    is_free = torch.zeros(alloc.pool_size, dtype=torch.bool)
    is_free[free_ids] = True
    blocks = torch.nonzero(~is_free).flatten()
    slots = torch.empty(0, dtype=torch.int64)
    md = getattr(ctx, "mamba_metadata", None)
    slot_free = None
    if md is not None:
        slot_free = torch.zeros(md.max_requests, dtype=torch.bool)
        slot_free[md.mamba_state_free_slots[: md.mamba_state_free_slot_count].to(torch.int64)] = True
        slots = torch.nonzero(~slot_free).flatten()
    # Owners: every resident (active or paused) request's block table and Mamba slot, the prefix cache's cached
    # blocks, the dummy block and the registered holders (register_block_owner / register_slot_owner). They must be
    # exactly the complement of the free stacks; anything else is refused.
    n = int(getattr(ctx, "total_request_count", 0) or 0)
    owned = torch.zeros(alloc.pool_size, dtype=torch.bool)
    owned[alloc.pool_size - 1] = True  # dummy block
    table = getattr(ctx, "request_to_kv_block_ids", None)
    if n and table is not None:
        used = table[:n].flatten().to("cpu", torch.int64)
        owned[used[used >= 0]] = True
    hashes = _block_hashes(alloc)
    if hashes is not None:
        owned |= hashes.to("cpu") != -1
    for held in _BLOCK_OWNERS.values():
        ids = torch.as_tensor(list(held()), dtype=torch.int64)
        owned[ids] = True
    if not torch.equal(owned, ~is_free):
        raise RuntimeError(
            f"kv_live_offload: KV blocks owned by requests / prefix cache / dummy ({int(owned.sum())}) are not the "
            f"allocator's non-free blocks ({int((~is_free).sum())}); {int((owned & is_free).sum())} owned blocks free, "
            f"{int((~owned & ~is_free).sum())} allocated blocks unowned"
        )
    if slot_free is not None:
        slot_owned = torch.zeros(md.max_requests, dtype=torch.bool)
        if n:
            req_slots = md.request_to_mamba_state_idx[:n].to("cpu", torch.int64)
            slot_owned[req_slots[req_slots >= 0]] = True
        for held in _SLOT_OWNERS.values():
            slot_owned[torch.as_tensor(list(held()), dtype=torch.int64)] = True
        if not torch.equal(slot_owned, ~slot_free):
            raise RuntimeError(
                f"kv_live_offload: Mamba slots owned by requests ({int(slot_owned.sum())}) are not the non-free slots "
                f"({int((~slot_free).sum())})"
            )
    return blocks.to(dev), slots.to(dev)


def _region_tensors(ctx: Any) -> list[tuple[str, torch.Tensor, str]]:
    """(name, tensor, how) for every context tensor allocated in the region: 'kv' (block dim), 'slot' (slot dim),
    'whole'. Refuses a region allocation no context tensor holds."""
    import bisect

    blocks = _STATE["blocks"]
    starts = sorted(blocks)

    def block_of(ptr: int):
        i = bisect.bisect_right(starts, ptr) - 1
        return starts[i] if i >= 0 and ptr < starts[i] + max(blocks[starts[i]], 1) else None

    # The context's tensors and those of its Megatron inference helper objects (e.g. mamba_metadata).
    owners = [("", ctx)] + [
        (f"{k}.", v)
        for k, v in vars(ctx).items()
        if hasattr(v, "__dict__") and type(v).__module__.startswith("megatron.core.inference")
    ]
    attrs = [(p + n, t) for p, o in owners for n, t in list(vars(o).items())]
    out, covered = [], set()
    for name, t in attrs:
        if not isinstance(t, torch.Tensor) or not t.is_cuda:
            continue
        b = block_of(t.untyped_storage().data_ptr())
        if b is None or b in covered:
            continue
        covered.add(b)
        if name in ("memory_buffer", "_hp_kv_inv"):
            how = "kv"
        elif name.startswith("mamba_") and name.endswith("_states"):
            how = "slot"
        else:
            how = "whole"
        out.append((name, t, how))
    unknown = [(a, n) for a, n in blocks.items() if a not in covered]
    if unknown:
        raise RuntimeError(
            f"engine KV / state region: {len(unknown)} allocations ({sum(n for _, n in unknown) / 2**20:.1f} MiB) "
            "hold no context tensor: their contents would not survive a suspend without a CPU backup"
        )
    return out


def _kv_dim(ctx: Any, name: str) -> int:
    # memory_buffer: (2 | planes, layers, blocks, ...) or MLA latent (layers, blocks, ...); _hp_kv_inv: (2, layers,
    # blocks, block_tokens).
    return 1 if (name == "memory_buffer" and getattr(ctx, "cache_mla_latent", False)) else 2


# Negative control (gates only): the suspend drops the first KV block of the first resident request from the copy, so
# that request reads garbage there after the wake and the engine / learner sentinel must fail.
DROP_ONE_ENV = "NEMO_RL_DEBUG_KV_LIVE_DROP_ONE"


def _owner_digest(ctx: Any, regions: list, blocks: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Per live block / slot, a 64-bit digest (bit-pattern sum) over every region tensor holding it: computed from the
    device state itself, independent of what the suspend chose to copy."""
    parts = []
    for name, t, how in regions:
        if how == "whole":
            continue
        dim, idx = (_kv_dim(ctx, name), blocks) if how == "kv" else (1, slots)
        if idx.numel() == 0:
            continue
        rows = t.index_select(dim, idx).movedim(dim, 0).contiguous().reshape(idx.numel(), -1)
        word = torch.int32 if rows.shape[1] * rows.element_size() % 4 == 0 else torch.uint8
        parts.append(rows.view(word).sum(1, dtype=torch.int64))
    return torch.cat(parts).cpu() if parts else torch.empty(0, dtype=torch.int64)


def save_live(ctx: Any) -> int:
    """Copy the resident state of the region's tensors to the host; returns the bytes copied. Under the exactness
    checks (THUNDERSYNC_NL_CHECKS=1) a digest of every live block / slot is taken from the device first and compared
    after the resume (restore_live): the resident requests' state must survive the suspend, whatever was copied."""
    import os

    torch.cuda.synchronize()
    blocks, slots = _live_index(ctx)
    if os.environ.get("THUNDERSYNC_NL_CHECKS") == "1":
        _STATE["digest"] = (blocks, slots, _owner_digest(ctx, _region_tensors(ctx), blocks, slots))
    table = getattr(ctx, "request_to_kv_block_ids", None)
    if os.environ.get(DROP_ONE_ENV) == "1" and table is not None and int(getattr(ctx, "total_request_count", 0) or 0):
        victim = int(table[0, 0])
        blocks = blocks[blocks != victim]
    saved, nbytes = [], 0
    for name, t, how in _region_tensors(ctx):
        if how == "kv":
            dim, idx = _kv_dim(ctx, name), blocks
        elif how == "slot":
            dim, idx = 1, slots
        else:
            dim, idx = None, None
        part = t if dim is None else t.index_select(dim, idx)
        cpu = part.to("cpu", copy=True)
        nbytes += cpu.numel() * cpu.element_size()
        saved.append((t, dim, idx, cpu))
    _STATE["saved"] = saved
    _STATE["stats"].update(saved_bytes=nbytes, live_blocks=int(blocks.numel()), live_slots=int(slots.numel()))
    return nbytes


def restore_live(ctx: Any) -> None:
    """Write the saved resident state back into the (re-mapped) region tensors. Under the exactness checks
    (THUNDERSYNC_NL_CHECKS=1) the restored live state is read back and compared bitwise with what was saved."""
    import os

    saved = _STATE["saved"] or []
    for t, dim, idx, cpu in saved:
        if dim is None:
            t.copy_(cpu)
        else:
            t.index_copy_(dim, idx, cpu.to(t.device))
    torch.cuda.synchronize()
    if os.environ.get("THUNDERSYNC_NL_CHECKS") == "1":
        # Failures are recorded here (this runs inside Megatron's resume on the engine loop thread, where raising
        # would stall the run) and raised by raise_if_check_failed on the worker's wake path.
        for t, dim, idx, cpu in saved:
            back = (t if dim is None else t.index_select(dim, idx)).cpu()
            if not torch.equal(back.view(torch.uint8), cpu.view(torch.uint8)):
                _STATE["check_error"] = "the restored live KV / state differs from what the suspend saved"
        digest = _STATE.pop("digest", None)
        if digest is not None:
            blocks, slots, before = digest
            after = _owner_digest(ctx, _region_tensors(ctx), blocks, slots)
            if not torch.equal(before, after):
                _STATE["check_error"] = (
                    f"{int((before != after).sum())} live KV blocks / Mamba slots of resident requests changed across "
                    "the suspend"
                )
            _STATE["stats"]["checked_blocks"] = int(blocks.numel())
            _STATE["stats"]["checked_slots"] = int(slots.numel())
    _STATE["saved"] = None


class _SaverProxy:
    """Stands in for torch_memory_saver inside Megatron's dynamic_context module: the context's region gets no CPU
    backup (its allocator blocks recorded), and its pause / resume save / restore only the live state."""

    def __init__(self, real: Any, tag: str):
        self._real = real
        self._tag = tag

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    @contextlib.contextmanager
    def region(self, tag: str, enable_cpu_backup: bool = False, **kwargs):
        if tag != self._tag:
            with self._real.region(tag=tag, enable_cpu_backup=enable_cpu_backup, **kwargs):
                yield
            return
        from nemo_rl.models.megatron.memory_saver import active_blocks

        dev = torch.cuda.current_device()
        torch.cuda.synchronize()
        before = active_blocks(dev)
        with self._real.region(tag=tag, enable_cpu_backup=False, **kwargs):
            yield
        torch.cuda.synchronize()
        _STATE["blocks"].update({a: n for a, n in active_blocks(dev).items() if a not in before})

    # pause / resume run inside Megatron's suspend / resume on the engine loop thread, where an exception stalls the
    # run: a failed check is recorded (raise_if_check_failed raises it on the worker's sleep / wake path) and a suspend
    # whose live set could not be established is not paused (the state stays resident, nothing is lost).
    def pause(self, tag=None):
        ctx = _STATE["ctx"]() if _STATE["ctx"] is not None else None
        if tag == self._tag and ctx is not None:
            try:
                save_live(ctx)
            except Exception as e:  # noqa: BLE001 - reported on the worker's sleep path
                _STATE["check_error"] = str(e)
                _STATE["skipped_pause"] = True
                return
        self._real.pause(tag)

    def resume(self, tag=None):
        if tag == self._tag and _STATE.pop("skipped_pause", False):
            return  # never paused: nothing to map back or restore
        self._real.resume(tag)
        ctx = _STATE["ctx"]() if _STATE["ctx"] is not None else None
        if tag == self._tag and ctx is not None and _STATE["saved"] is not None:
            restore_live(ctx)


def install() -> None:
    """Route Megatron's KV / state region of every DynamicInferenceContext built afterwards through _SaverProxy
    (needs torch_memory_saver; used with kv_cache_management_mode offload)."""
    if _STATE["installed"]:
        return
    from megatron.core.inference.contexts import dynamic_context as dc

    if not getattr(dc, "HAVE_TORCH_MEMORY_SAVER", False):
        raise RuntimeError("kv_live_offload needs torch_memory_saver (the TMS=1 overlay)")
    dc.torch_memory_saver = _SaverProxy(dc.torch_memory_saver, dc.DynamicInferenceContext.TMS_TAG)
    orig_init = dc.DynamicInferenceContext.__init__

    def __init__(self, *args, **kwargs):
        _STATE["blocks"] = {}
        _STATE["ctx"] = weakref.ref(self)
        orig_init(self, *args, **kwargs)
        if getattr(self, "_uses_torch_memory_saver", False):
            _region_tensors(self)  # refuse an unaccounted region allocation now, not at the first suspend

    dc.DynamicInferenceContext.__init__ = __init__
    _STATE["installed"] = True


def raise_if_check_failed() -> None:
    """Raise the last suspend's (live set / region accounting) or resume's (live-state check) failure, if any: called
    on the worker's sleep and wake paths so it reaches the driver instead of stalling the engine loop."""
    err = _STATE.pop("check_error", None)
    if err is not None:
        raise RuntimeError(f"kv_live_offload: {err}")


def last_stats() -> dict:
    """Bytes and live blocks / slots of the last suspend (colocated transition log)."""
    return dict(_STATE["stats"])
