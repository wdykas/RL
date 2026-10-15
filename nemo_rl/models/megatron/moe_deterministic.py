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
"""Deterministic backward for MoE token permutation (training).

Megatron-Core's ``moe_utils.permute`` gathers each token once per routed
expert (``tokens.index_select(0, sorted_indices)``, with repeated indices). Its
autograd backward is ``index_add_`` into the token gradient, which adds a
token's top-k copies with atomics in a run-dependent order: the MoE layer's
input gradient (and everything below it) then differs bit for bit between
identical steps, although every forward and every expert weight gradient is
deterministic (measured on Nemotron-3-Nano L8: the first decoder layer whose
input gradient differs across repeats is an MoE layer, its output gradient
equal).

``GatherRows`` keeps the forward (the same gather, same bits) and sums each
token's copies in a fixed order (by permuted position) in fp32 in backward.
The values change by rounding only; the training gradient becomes
reproducible. Per rank: one [tokens, top-k] int64 index table and one
[tokens, top-k, hidden] gather of the incoming gradient (the size of the
permuted activations) during backward.
"""

from __future__ import annotations

import torch


class GatherRows(torch.autograd.Function):
    """``x.index_select(0, idx)`` whose backward sums duplicates in a fixed order."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:  # noqa: D102
        ctx.save_for_backward(idx)
        ctx.n_rows = x.shape[0]
        return x.index_select(0, idx)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # noqa: D102
        (idx,) = ctx.saved_tensors
        return segment_sum_rows(grad, idx, ctx.n_rows), None


def segment_sum_rows(grad: torch.Tensor, idx: torch.Tensor, n_rows: int) -> torch.Tensor:
    """out[t] = sum of grad[j] over j with idx[j] == t, in increasing j, accumulated in fp32."""
    if idx.numel() == 0:
        return grad.new_zeros((n_rows,) + tuple(grad.shape[1:]))
    sorted_tok, order = torch.sort(idx, stable=True)  # copies of a token in increasing position
    counts = torch.bincount(sorted_tok, minlength=n_rows)
    k = int(counts.max())
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(idx.numel(), device=idx.device) - starts[sorted_tok]
    table = torch.full((n_rows, k), -1, dtype=torch.long, device=idx.device)
    table[sorted_tok, rank] = order
    valid = table >= 0
    rows = grad.reshape(grad.shape[0], -1).float()
    gathered = rows.index_select(0, table.clamp_min(0).reshape(-1)).view(n_rows, k, -1)
    gathered = gathered * valid.unsqueeze(-1)
    out = gathered[:, 0]
    for j in range(1, k):  # fixed left-to-right order
        out = out + gathered[:, j]
    return out.to(grad.dtype).view((n_rows,) + tuple(grad.shape[1:]))


def install() -> None:
    """Route Megatron-Core's MoE ``permute`` token gather through ``GatherRows`` (idempotent)."""
    import megatron.core.transformer.moe.moe_utils as moe_utils

    orig = moe_utils.permute
    if getattr(orig, "_ts_deterministic", False):
        return

    def permute(tokens, routing_map, *args, **kwargs):
        out = orig(tokens, routing_map, *args, **kwargs)
        if not (torch.is_grad_enabled() and tokens.requires_grad):
            return out
        if not isinstance(out, tuple) or len(out) < 3 or not isinstance(out[2], torch.Tensor):
            return out
        # out[0] == tokens.index_select(0, sorted_indices); recomputed with a
        # deterministic backward (same forward bits).
        return (GatherRows.apply(tokens, out[2]),) + tuple(out[1:])

    permute._ts_deterministic = True
    moe_utils.permute = permute
    try:
        import megatron.core.transformer.moe.token_dispatcher as token_dispatcher

        if hasattr(token_dispatcher, "permute"):
            token_dispatcher.permute = permute
    except ImportError:
        pass
