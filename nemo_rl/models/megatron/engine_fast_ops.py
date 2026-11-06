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
"""Engine decode fast paths that keep today's bits (research/engine_latency).

``NEMO_RL_ENGINE_FAST_LOGSOFTMAX=1`` (default off; engine processes): the batch-invariant log-softmax that Megatron
registers for ``aten::_log_softmax`` (``batch_invariant_kernels.log_softmax``: ONE program per row, three sequential
passes over the row in 1024-column blocks) runs as three multi-program kernels over (row, block):

  K1 (row, b): block max  m_b = max(maximum(x_b, -inf))
  K2 (row, b): M = max over the block maxima (read in block order); s_b = sum(where(mask, exp(x_b - M), 0)) with the
               original's block shape (1024 columns, 4 warps), so each block sum has the original's reduction tree
  K3 (row, b): S = ((0 + s_0) + s_1) + ... in block order (the original's ``sum_exp +=`` fold), L = log(S),
               out_b = x_b - M - L

The max is exact in any order (fp32 max; the tests include rows whose maximum is a mixed +0 / -0 set and NaNs), the
block sums are the original's per-iteration ``tl.sum`` and the fold is the original's order, so every output element
equals the original kernel's. One program per row reaches ~13 GB/s at V = 131072 (119 us per decode step at 1 row,
outside the CUDA graph, on the sampling critical path); the three kernels fill the GPU. Workspace: two fp32
[rows, ceil(V / 1024)] tensors per call (512 B per row each at V 131072), transient.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

BLOCK = 1024  # the original kernel's BLOCK_SIZE (its reduction tree depends on it)

_STATE = {"installed": False, "orig": None, "calls": 0}


def fast_logsoftmax_by_env() -> bool:
    return os.environ.get("NEMO_RL_ENGINE_FAST_LOGSOFTMAX", "0").strip().lower() not in ("", "0", "false", "no", "off")


@triton.jit
def _blockmax_kernel(input_ptr, bmax_ptr, input_row_stride, n_cols, nblk, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offset = b * BLOCK_SIZE
    col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
    mask = col_idx < n_cols
    vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))
    tl.store(bmax_ptr + row_idx * nblk + b, tl.max(tl.maximum(vals, -float("inf"))))


@triton.jit
def _blocksum_kernel(input_ptr, bmax_ptr, bsum_ptr, input_row_stride, n_cols, nblk, BLOCK_SIZE: tl.constexpr,
                     NB_POW2: tl.constexpr):
    row_idx = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    bs = tl.arange(0, NB_POW2)
    max_val = tl.max(tl.load(bmax_ptr + row_idx * nblk + bs, mask=bs < nblk, other=-float("inf")))
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offset = b * BLOCK_SIZE
    col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
    mask = col_idx < n_cols
    vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
    exp_vals = tl.exp(vals - max_val)
    tl.store(bsum_ptr + row_idx * nblk + b, tl.sum(tl.where(mask, exp_vals, 0.0)))


@triton.jit
def _output_kernel(input_ptr, output_ptr, bmax_ptr, bsum_ptr, input_row_stride, output_row_stride, n_cols, nblk,
                   BLOCK_SIZE: tl.constexpr, NB_POW2: tl.constexpr, NB: tl.constexpr):
    row_idx = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    bs = tl.arange(0, NB_POW2)
    max_val = tl.max(tl.load(bmax_ptr + row_idx * nblk + bs, mask=bs < nblk, other=-float("inf")))
    # The original's fold order ((0 + s_0) + s_1) + ...; unrolled (NB = nblk, constexpr) so the NB loads are issued
    # independently and only the adds are serial.
    sum_exp = 0.0
    for j in tl.static_range(NB):
        sum_exp += tl.load(bsum_ptr + row_idx * NB + j)
    log_sum_exp = tl.log(sum_exp)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    col_offset = b * BLOCK_SIZE
    col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
    mask = col_idx < n_cols
    vals = tl.load(row_start_ptr + col_idx, mask=mask)
    output = vals - max_val - log_sum_exp
    tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Drop-in for ``batch_invariant_kernels.log_softmax`` (same signature, same bits)."""
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError("This implementation only supports log_softmax along the last dimension")
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1]).contiguous()
    n_rows, n_cols = input_2d.shape
    output = torch.empty_like(input_2d)
    if n_rows == 0 or n_cols == 0 or input_2d.dtype != torch.float32:
        # The original handles these; keep its exact code path.
        return _STATE["orig"](input, dim=dim)
    nblk = triton.cdiv(n_cols, BLOCK)
    nb2 = max(triton.next_power_of_2(nblk), 2)
    bmax = torch.empty((n_rows, nblk), dtype=torch.float32, device=input.device)
    bsum = torch.empty((n_rows, nblk), dtype=torch.float32, device=input.device)
    grid = (n_rows, nblk)
    _blockmax_kernel[grid](input_2d, bmax, input_2d.stride(0), n_cols, nblk, BLOCK_SIZE=BLOCK)
    _blocksum_kernel[grid](input_2d, bmax, bsum, input_2d.stride(0), n_cols, nblk, BLOCK_SIZE=BLOCK, NB_POW2=nb2)
    _output_kernel[grid](input_2d, output, bmax, bsum, input_2d.stride(0), output.stride(0), n_cols, nblk,
                         BLOCK_SIZE=BLOCK, NB_POW2=nb2, NB=nblk)
    _STATE["calls"] += 1
    return output.reshape(original_shape)


def install_fast_logsoftmax() -> bool:
    """Point Megatron's batch-invariant ``aten::_log_softmax`` implementation at ``log_softmax`` (this process)."""
    from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik

    if _STATE["installed"]:
        return True
    _STATE["orig"] = bik.log_softmax
    if torch.cuda.is_available():
        # Install-time self-check (R27 review): one row against the original, bitwise; a Triton / compiler change
        # that alters either reduction tree is fatal here, not a silent engine / learner mismatch later.
        g = torch.Generator(device="cuda").manual_seed(1234)
        row = torch.randn(2, 131072 + 17, device="cuda", generator=g) * 6.0
        want, got = _STATE["orig"](row, dim=-1), log_softmax(row, dim=-1)
        if not torch.equal(want, got):
            raise RuntimeError(
                "NEMO_RL_ENGINE_FAST_LOGSOFTMAX: install self-check failed (fast log-softmax != the original, max |d| "
                f"{float((want - got).abs().max()):.3g})"
            )
    bik.log_softmax = log_softmax
    _STATE["installed"] = True
    return True
