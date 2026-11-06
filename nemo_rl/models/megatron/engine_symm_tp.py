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
"""One-shot ordered TP collectives for the inference engine (symmetric memory, NVLink).

``NEMO_RL_ENGINE_SYMM_TP=1`` (default off; engine processes only, ``install`` from
``MegatronGenerationWorker._prepare_engine_numerics``): the sequence-parallel row collectives of the fp16-GEMM forward
(``fp16_gemm._reduce_scatter_rows`` / ``_gather_rows``) run as one Triton kernel each instead of NCCL (TP = 2: ring
reduce-scatter; TP >= 3: all-to-all + rank-order fold) and NCCL all-gather:

* reduce-scatter (fp32 partials): every rank stores the TP - 1 output slices it does not own into the owners'
  symmetric staging (NVLink stores), one block-level barrier, then folds ``((p_0 + p_1) + p_2) + ...`` in rank order
  in fp32 (``add.rn``), own partial read locally. That is exactly ``fp16_gemm._fold_rank_order`` (TP >= 3) and the
  single commutative IEEE addition NCCL does at TP = 2, so every output element has today's bits.
* all-gather (any dtype): every rank stores its rows into every peer's staging, one barrier. Under CUDA-graph
  capture the gathered tensor IS this rank's staging (a view, no copy pass); eagerly the staged rows are copied to a
  fresh output. A copy: bitwise by construction.
* one barrier per collective, no trailing barrier: the staging is double-buffered per kind (RS / AG, alternating
  regions call by call, the parity a function of the call's position since the forward started, so every rank and
  every CUDA-graph replay use the same region for the same call). A rank writes region p of its peers again only two
  calls of that kind later, i.e. after passing the barrier of the call in between, which a peer reaches only after its
  stream finished reading region p (and consuming an all-gather view of it). The first collective of each model
  forward (``forward_start``, a forward pre-hook on the engine model) and every eager collective also wait at a
  leading barrier, so a forward's first calls never race the previous forward's last ones whatever their parity.

The block-level barriers (``symm_mem_sync``, Megatron's NVLS collectives' barrier) reset their signal pad, so the
kernels are CUDA-graph capturable with static pointers. Grid size is a function of the message size only (identical
on every rank). Messages above the staging capacity fall back to the stock path (same bits; speed-only threshold):
decode steps fit, large prefill graphs fall back.

Per rank: one symmetric buffer of ``NEMO_RL_ENGINE_SYMM_TP_MB`` (default 64) MiB per TP group (four regions: RS
and AG, two each), allocated before graph capture. Scales with TP <= 8 (NVLink domain; one peer pointer per rank); TP > 8 or groups without symmetric
memory keep the stock path.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

_MAX_WS = 8
_NEG = os.environ.get("NEMO_RL_ENGINE_SYMM_TP_NEG", "0") == "1"
_MAX_BLOCKS = 64
_STATE: dict[str, Any] = {"group": None, "hdl": None, "bytes": 0, "views": {}, "calls": 0, "fallback": 0, "lead": 0,
                          "parity": {"rs": 0, "ag": 0}, "first": True, "prev_eager": True, "hooked": False}
_REGIONS = {"rs": 0, "ag": 2}  # region index of parity 0 (parity 1 = +1); 4 equal regions

try:
    import triton
    import triton.language as tl

    from megatron.core.inference.communication.torch_symm_triton.barrier import symm_mem_sync
    from megatron.core.inference.communication.torch_symm_triton.utils import sync_threads

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


def enabled_by_env() -> bool:
    return os.environ.get("NEMO_RL_ENGINE_SYMM_TP", "0").strip().lower() not in ("", "0", "false", "no", "off")


def install(group: Any) -> bool:
    """Allocate the symmetric staging for ``group`` (the engine's TP group). Call before CUDA-graph capture."""
    if not HAVE_TRITON or group is None:
        return False
    ws = torch.distributed.get_world_size(group)
    if not 2 <= ws <= _MAX_WS:
        return False
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

    mb = int(os.environ.get("NEMO_RL_ENGINE_SYMM_TP_MB", "64"))
    key = f"nemo_rl_engine_tp:{group.group_name}"
    buf = SymmetricMemoryManager.get_buffer(key, process_group=group, size_mb=mb)
    hdl = buf.symm_mem_hdl
    if hdl is None or hdl.world_size != ws:
        print(f"[engine_symm_tp] symmetric memory unavailable ({buf.init_failure_reason}); stock collectives", flush=True)
        return False
    _STATE.update(group=group, hdl=hdl, bytes=buf.symm_buffer.numel(), views={})
    if torch.distributed.get_rank() == 0:
        print(f"[engine_symm_tp] one-shot ordered TP reduce-scatter / all-gather on TP{ws} ({mb} MiB staging/rank)",
              flush=True)
    return True


def active_for(group: Any) -> bool:
    if _STATE["hdl"] is None or group is None:
        return False
    return group is _STATE["group"] or getattr(group, "group_name", None) == _STATE["group"].group_name


def stats() -> dict[str, int]:
    return {"calls": _STATE["calls"], "fallback": _STATE["fallback"], "lead": _STATE["lead"]}


def _region_bytes() -> int:
    return (_STATE["bytes"] // 4) // 16 * 16


def _views(numel: int, dtype: torch.dtype, region: int = 0) -> list[torch.Tensor]:
    key = (numel, dtype, region)
    v = _STATE["views"].get(key)
    if v is None:
        hdl = _STATE["hdl"]
        off = region * _region_bytes() // torch.empty((), dtype=dtype).element_size()
        v = [hdl.get_buffer(r, (numel,), dtype, off) for r in range(hdl.world_size)]
        _STATE["views"][key] = v
    return v


def forward_start(*_args, **_kwargs) -> None:
    """A model forward begins: parities restart at 0 and the next collective waits at a leading barrier."""
    _STATE["first"] = True
    _STATE["parity"] = {"rs": 0, "ag": 0}


def register_forward_start(model: Any) -> bool:
    """Install ``forward_start`` as a forward pre-hook on the engine model (call before CUDA-graph capture)."""
    if model is None or not hasattr(model, "register_forward_pre_hook"):
        return False
    model.register_forward_pre_hook(forward_start)
    _STATE["hooked"] = True
    return True


def _schedule(kind: str) -> tuple[int, bool, bool]:
    """(region, lead barrier?, capturing?) of the next collective of ``kind``."""
    capturing = torch.cuda.is_current_stream_capturing()
    lead = (not capturing) or _STATE["first"] or _STATE["prev_eager"] or not _STATE["hooked"]
    p = _STATE["parity"][kind]
    _STATE["parity"][kind] = 1 - p
    _STATE["first"] = False
    _STATE["prev_eager"] = not capturing
    _STATE["lead"] += int(lead)
    return _REGIONS[kind] + p, lead, capturing


def _max_blocks(ws: int) -> int:
    """symm_mem_sync gives each (block, peer) one uint32 slot of the signal pad: blocks <= pad / (4 ws)."""
    try:
        import torch.distributed._symmetric_memory as symm_mem

        pad = int(symm_mem.get_signal_pad_size())
    except (ImportError, AttributeError, RuntimeError):
        pad = 2048
    return max(1, min(_MAX_BLOCKS, pad // (4 * ws)))


def _grid(n: int, block: int, ws: int) -> int:
    return max(1, min(triton.cdiv(n, block), _max_blocks(ws)))


if HAVE_TRITON:

    @triton.jit
    def _peer(d: tl.constexpr, p0, p1, p2, p3, p4, p5, p6, p7):
        if d == 0:
            return p0
        elif d == 1:
            return p1
        elif d == 2:
            return p2
        elif d == 3:
            return p3
        elif d == 4:
            return p4
        elif d == 5:
            return p5
        elif d == 6:
            return p6
        else:
            return p7

    @triton.jit
    def _rs_push_fold_kernel(src, s0, s1, s2, s3, s4, s5, s6, s7, out, signal_pad_ptrs, n,
                             RANK: tl.constexpr, WS: tl.constexpr, BLOCK: tl.constexpr, LEAD: tl.constexpr):
        """src: this rank's partial of all WS output slices (slice d at d * n). Staging of rank r: [WS][n]."""
        if LEAD:  # every rank finished the previous collectives' reads of this staging
            symm_mem_sync(signal_pad_ptrs, None, RANK, WS, hasPreviousMemAccess=True, hasSubsequentMemAccess=True)
            sync_threads()
        pid = tl.program_id(0)
        step = tl.num_programs(0).to(tl.int64) * BLOCK
        first = pid.to(tl.int64) * BLOCK
        n64 = first * 0 + n  # int64 slice offsets d * n: the staging holds WS * n elements (may pass 2^31)
        lane = tl.arange(0, BLOCK)
        for start in range(first, n, step):
            offs = start + lane
            mask = offs < n
            for d in tl.static_range(WS):
                if d != RANK:
                    v = tl.load(src + d * n64 + offs, mask=mask)
                    tl.store(_peer(d, s0, s1, s2, s3, s4, s5, s6, s7) + RANK * n64 + offs, v, mask=mask)
        sync_threads()
        symm_mem_sync(signal_pad_ptrs, None, RANK, WS, hasPreviousMemAccess=True, hasSubsequentMemAccess=True)
        sync_threads()
        mine = _peer(RANK, s0, s1, s2, s3, s4, s5, s6, s7)
        for start in range(first, n, step):
            offs = start + lane
            mask = offs < n
            if RANK == 0:
                acc = tl.load(src + offs, mask=mask)
            else:
                acc = tl.load(mine + offs, mask=mask)
            for r in tl.static_range(1, WS):
                if r == RANK:
                    acc = acc + tl.load(src + RANK * n64 + offs, mask=mask)
                else:
                    acc = acc + tl.load(mine + r * n64 + offs, mask=mask)
            tl.store(out + offs, acc, mask=mask)

    @triton.jit
    def _ag_push_kernel(src, s0, s1, s2, s3, s4, s5, s6, s7, out, signal_pad_ptrs, n,
                        RANK: tl.constexpr, WS: tl.constexpr, BLOCK: tl.constexpr, LEAD: tl.constexpr,
                        COPY_OUT: tl.constexpr):
        """src: this rank's n elements; out: [WS][n] in rank order (COPY_OUT; else the output is this rank's
        staging). Staging of rank r: [WS][n]."""
        if LEAD:
            symm_mem_sync(signal_pad_ptrs, None, RANK, WS, hasPreviousMemAccess=True, hasSubsequentMemAccess=True)
            sync_threads()
        pid = tl.program_id(0)
        step = tl.num_programs(0).to(tl.int64) * BLOCK
        first = pid.to(tl.int64) * BLOCK
        n64 = first * 0 + n  # int64 slice offsets d * n: the staging holds WS * n elements (may pass 2^31)
        lane = tl.arange(0, BLOCK)
        for start in range(first, n, step):
            offs = start + lane
            mask = offs < n
            v = tl.load(src + offs, mask=mask)
            for d in tl.static_range(WS):
                tl.store(_peer(d, s0, s1, s2, s3, s4, s5, s6, s7) + RANK * n64 + offs, v, mask=mask)
        sync_threads()
        symm_mem_sync(signal_pad_ptrs, None, RANK, WS, hasPreviousMemAccess=True, hasSubsequentMemAccess=True)
        if COPY_OUT:
            sync_threads()
            mine = _peer(RANK, s0, s1, s2, s3, s4, s5, s6, s7)
            for start in range(first, n, step):
                offs = start + lane
                mask = offs < n
                for r in tl.static_range(WS):
                    tl.store(out + r * n64 + offs, tl.load(mine + r * n64 + offs, mask=mask), mask=mask)


def _launch(kernel, src: torch.Tensor, out: torch.Tensor, n: int, staging_dtype: torch.dtype, region: int,
            **flags) -> None:
    hdl = _STATE["hdl"]
    ws = hdl.world_size
    views = _views(ws * n, staging_dtype, region)
    ptrs = list(views) + [views[0]] * (_MAX_WS - ws)
    vec = max(1, 16 // src.element_size())
    block = 256 * vec  # 8 warps, one 16-byte vector per thread per iteration
    kernel[(_grid(n, block, ws),)](src, *ptrs, out, hdl.signal_pad_ptrs_dev, n, RANK=hdl.rank, WS=ws, BLOCK=block,
                                    num_warps=8, **flags)


def reduce_scatter_rows(x: torch.Tensor, group: Any) -> Optional[torch.Tensor]:
    """Ordered fp32 reduce-scatter along dim 0, or None (caller takes the stock path; same bits)."""
    ws = _STATE["hdl"].world_size
    if (x.dtype != torch.float32 or not x.is_cuda or x.shape[0] % ws != 0
            or x.numel() * x.element_size() > _region_bytes()):
        _STATE["fallback"] += 1
        _STATE["prev_eager"] = True  # the next symmetric collective waits at a leading barrier
        return None
    x = x.contiguous()
    out = torch.empty((x.shape[0] // ws, *x.shape[1:]), dtype=x.dtype, device=x.device)
    n = out.numel()
    if n:
        region, lead, _ = _schedule("rs")
        _launch(_rs_push_fold_kernel, x, out, n, x.dtype, region, LEAD=lead)
        if _NEG:
            # Negative control (NEMO_RL_ENGINE_SYMM_TP_NEG=1, tests only): one element off by one ulp-ish.
            out.view(-1)[:1].mul_(1.0000002)
    _STATE["calls"] += 1
    return out


def gather_rows(x: torch.Tensor, group: Any, view_ok: bool = False) -> Optional[torch.Tensor]:
    """All-gather along dim 0 (rank order), or None (stock path). ``view_ok``: the caller consumes the result before
    the next collective (the GEMM right after the gather), so under capture it may be the staging itself."""
    ws = _STATE["hdl"].world_size
    if not x.is_cuda or x.numel() * ws * x.element_size() > _region_bytes():
        _STATE["fallback"] += 1
        _STATE["prev_eager"] = True
        return None
    x = x.contiguous()
    n = x.numel()
    if n == 0:
        _STATE["calls"] += 1
        return torch.empty((ws * x.shape[0], *x.shape[1:]), dtype=x.dtype, device=x.device)
    region, lead, capturing = _schedule("ag")
    view = capturing and view_ok
    if view:
        # The gathered rows are this rank's staging region (read by this layer's GEMM before the barrier of the next
        # collective of either kind; rewritten two all-gathers later).
        out = _views(ws * n, x.dtype, region)[_STATE["hdl"].rank].view(ws * x.shape[0], *x.shape[1:])
    else:
        out = torch.empty((ws * x.shape[0], *x.shape[1:]), dtype=x.dtype, device=x.device)
    _launch(_ag_push_kernel, x, out, n, x.dtype, region, LEAD=lead, COPY_OUT=not view)
    _STATE["calls"] += 1
    return out
