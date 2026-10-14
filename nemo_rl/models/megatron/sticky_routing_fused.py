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
"""One-kernel sticky rule (``sticky_routing.sticky_select``) for engine forwards.

The rule is a pure selection: comparisons, gathers and one fp32 subtraction
(``max_out - delta``), so a fused kernel returns bitwise what the reference
torch composition returns (unit-tested against it, ties and invalid anchor
rows included). The engine applies it in every MoE layer of every forward
when the anchor horizon is on; as ~15 small kernels it cost ~55 us per layer
at decode sizes on GB200, as one kernel ~one launch.

Per token row (one program): valid anchors (all ids >= 0, pairwise
distinct), min of the anchors' scores, max of the other experts' scores,
keep = valid and min_in >= max_out - delta; a kept anchor set that differs
from the plain set is written in descending score order with ties in anchor
order (the order of a stable descending sort), else (not kept, or kept and
equal to the plain set) the plain top-k in its own order.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sticky_kernel(
    scores_ptr,
    s_stride,
    anchors_ptr,
    a_stride,
    pvals_ptr,
    pv_stride,
    pidx_ptr,
    pi_stride,
    out_vals_ptr,
    out_idx_ptr,
    keep_ptr,
    E,
    K,
    delta,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    ks = tl.arange(0, BLOCK_K)
    kmask = ks < K
    a = tl.load(anchors_ptr + row * a_stride + ks, mask=kmask, other=0).to(tl.int64)
    # valid: every id >= 0 and no id repeated
    neg = tl.sum(((a < 0) & kmask).to(tl.int32), axis=0)
    same = (a[:, None] == a[None, :]) & kmask[:, None] & kmask[None, :]
    same = same & (ks[:, None] != ks[None, :])
    dup = tl.sum(tl.sum(same.to(tl.int32), axis=1), axis=0)
    valid = (neg == 0) & (dup == 0)
    a_safe = tl.where(a < 0, 0, a)
    s_in = tl.load(scores_ptr + row * s_stride + a_safe, mask=kmask, other=float("-inf"))
    min_in = tl.min(tl.where(kmask, s_in, float("inf")), axis=0)
    es = tl.arange(0, BLOCK_E)
    emask = es < E
    s = tl.load(scores_ptr + row * s_stride + es, mask=emask, other=float("-inf"))
    is_anchor = tl.sum(((es[:, None] == a_safe[None, :]) & kmask[None, :]).to(tl.int32), axis=1) > 0
    max_out = tl.max(tl.where(is_anchor | ~emask, float("-inf"), s), axis=0)
    keep = valid & (min_in >= max_out - delta)
    # rank of anchor i in a stable descending sort of s_in
    gt = (s_in[None, :] > s_in[:, None]) & kmask[None, :]
    eq_before = (s_in[None, :] == s_in[:, None]) & (ks[None, :] < ks[:, None]) & kmask[None, :]
    rank = tl.sum((gt | eq_before).to(tl.int32), axis=1)
    pv = tl.load(pvals_ptr + row * pv_stride + ks, mask=kmask, other=0.0)
    pi = tl.load(pidx_ptr + row * pi_stride + ks, mask=kmask, other=0).to(tl.int64)
    # anchor set == plain set (both K distinct ids when valid): every anchor is a plain id
    hit = (a[:, None] == pi[None, :]) & kmask[:, None] & kmask[None, :]
    in_plain = tl.sum(hit.to(tl.int32), axis=1) > 0
    same_set = tl.sum((in_plain & kmask).to(tl.int32), axis=0) == K
    reorder = keep & ~same_set
    # plain, or the kept anchors at their ranks (a permutation of 0..K-1)
    tl.store(out_vals_ptr + row * K + ks, pv, mask=kmask & ~reorder)
    tl.store(out_idx_ptr + row * K + ks, pi, mask=kmask & ~reorder)
    tl.store(out_vals_ptr + row * K + rank, s_in, mask=kmask & reorder)
    tl.store(out_idx_ptr + row * K + rank, a, mask=kmask & reorder)
    tl.store(keep_ptr + row, keep.to(tl.int8))


def sticky_select_fused(
    scores: torch.Tensor,
    anchors: torch.Tensor,
    plain_vals: torch.Tensor,
    plain_idx: torch.Tensor,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``sticky_routing.sticky_select`` in one kernel (CUDA tensors)."""
    T, E = scores.shape
    K = anchors.shape[1]
    vals = torch.empty((T, K), dtype=scores.dtype, device=scores.device)
    idx = torch.empty((T, K), dtype=torch.long, device=scores.device)
    keep = torch.empty((T,), dtype=torch.int8, device=scores.device)
    if T > 0:
        _sticky_kernel[(T,)](
            scores,
            scores.stride(0),
            anchors,
            anchors.stride(0),
            plain_vals,
            plain_vals.stride(0),
            plain_idx,
            plain_idx.stride(0),
            vals,
            idx,
            keep,
            E,
            K,
            float(delta),
            BLOCK_E=triton.next_power_of_2(E),
            BLOCK_K=max(2, triton.next_power_of_2(K)),
        )
    return vals, idx, keep.bool()
