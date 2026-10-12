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
"""Batch-invariant tensor-parallel sum reductions at TP>=3.

A 2-rank sum is one commutative IEEE addition, so NCCL's TP reduce-scatter and
all-reduce give a token the same bits at every token count at TP=2. At TP>=3 the
order of a token's partial sums under NCCL's default and Ring algorithms depends on
which rank owns the token, i.e. on the token count (measured with
research/thundersync_rl/scripts/bi_tp/nccl_tp_invariance.py), which breaks batch
invariance and train/generation parity.

The ordered reduction sends each rank's partial rows to their owner with one
all-to-all (the same bytes per rank as a ring reduce-scatter), then sums the TP
partials in ascending rank order in fp32 and rounds once. A token's bits depend
only on its partials, never on the token count or the rank that owns it. An
all-reduce is an ordered reduce-scatter of the flattened tensor followed by an
all-gather.

``install()`` reroutes every floating-point SUM reduce-scatter / all-reduce on
the tensor-parallel or expert-tensor-parallel group of size >= 3 (Transformer
Engine and Megatron-Core both call ``torch.distributed`` for these; MCore's
mappings module keeps its own alias). Groups of size <= 2 and every other group
keep NCCL. The training and generation workers must both install it (they do,
from ``enable_batch_invariant_kernels``), so their reductions share numerics.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist

from nemo_rl.models.megatron.batch_invariant import batch_invariant_token_multiple

_ORIG: dict[str, Any] = {}


class _DoneWork:
    """Handle for an ordered reduction run with ``async_op=True`` (already done)."""

    def wait(self, timeout: Any = None) -> bool:
        return True

    def is_completed(self) -> bool:
        return True

    def is_success(self) -> bool:
        return True


def _ordered_groups() -> list:
    """TP and expert-TP groups of size >= 3 (empty before Megatron initializes them)."""
    from megatron.core import parallel_state

    groups = []
    for get in (
        parallel_state.get_tensor_model_parallel_group,
        parallel_state.get_expert_tensor_parallel_group,
    ):
        group = get(check_initialized=False)
        if group is not None and dist.get_world_size(group) >= 3:
            groups.append(group)
    return groups


# id(group) -> (group, whether it spans the same ranks as a TP / expert-TP
# group). Ranks, not identity: a model may hold its own handle to the TP ranks.
# Holding the group keeps its id from being reused.
_MATCH: dict[int, tuple[Any, bool]] = {}
_REPORTED: set[tuple[int, ...]] = set()


def _is_sum(op: Any) -> bool:
    return op is None or op == dist.ReduceOp.SUM


def _use_ordered(tensor: torch.Tensor, op: Any, group: Any) -> bool:
    if group is None or not tensor.is_floating_point() or not _is_sum(op):
        return False
    cached = _MATCH.get(id(group))
    if cached is not None and cached[0] is group:
        return cached[1]
    tp_ranks = [dist.get_process_group_ranks(g) for g in _ordered_groups()]
    if not tp_ranks:
        return False  # not initialized yet: decide again on a later call
    ranks = dist.get_process_group_ranks(group)
    match = ranks in tp_ranks
    _MATCH[id(group)] = (group, match)
    if match and tuple(ranks) not in _REPORTED:
        _REPORTED.add(tuple(ranks))
        print(
            f"[ordered_tp_reduce] ordered SUM reductions on ranks {ranks}",
            flush=True,
        )
    return match


def ordered_reduce_scatter(
    output: torch.Tensor, input_: torch.Tensor, group: Any
) -> None:
    """``output`` = rank-order fp32 sum of every rank's chunk ``rank`` of ``input_``."""
    world = dist.get_world_size(group)
    if input_.numel() != output.numel() * world:
        raise ValueError(
            f"ordered reduce-scatter: input numel {input_.numel()} != "
            f"{world} x output numel {output.numel()}"
        )
    recv = torch.empty_like(input_)
    _ORIG["all_to_all_single"](recv, input_.contiguous(), group=group)
    parts = recv.view(world, -1)
    acc = parts[0].float()
    for r in range(1, world):  # ascending rank order
        acc.add_(parts[r])
    output.view(-1).copy_(acc)


def ordered_all_reduce(tensor: torch.Tensor, group: Any) -> None:
    """In-place rank-order fp32 sum over ``group``."""
    world = dist.get_world_size(group)
    flat = tensor.reshape(-1)
    chunk = math.ceil(flat.numel() / world)
    padded = flat.new_zeros(chunk * world)
    padded[: flat.numel()].copy_(flat)
    local = flat.new_empty(chunk)
    ordered_reduce_scatter(local, padded, group)
    _ORIG["all_gather_into_tensor"](padded, local, group=group)
    tensor.copy_(padded[: flat.numel()].view_as(tensor))


def _reduce_scatter_tensor(
    output, input, op=dist.ReduceOp.SUM, group=None, async_op=False
):  # noqa: A002
    if not _use_ordered(input, op, group):
        return _ORIG["reduce_scatter_tensor"](
            output, input, op=op, group=group, async_op=async_op
        )
    ordered_reduce_scatter(output, input, group)
    return _DoneWork() if async_op else None


def _reduce_scatter_base(
    output, input, op=dist.ReduceOp.SUM, group=None, async_op=False
):  # noqa: A002
    if not _use_ordered(input, op, group):
        return _ORIG["_reduce_scatter_base"](
            output, input, op=op, group=group, async_op=async_op
        )
    ordered_reduce_scatter(output, input, group)
    return _DoneWork() if async_op else None


def _all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
    if not _use_ordered(tensor, op, group):
        return _ORIG["all_reduce"](tensor, op=op, group=group, async_op=async_op)
    ordered_all_reduce(tensor, group)
    return _DoneWork() if async_op else None


def install() -> None:
    """Route TP>=3 sum reductions through the ordered implementation (idempotent)."""
    if _ORIG:
        return
    for name in (
        "reduce_scatter_tensor",
        "_reduce_scatter_base",
        "all_reduce",
        "all_to_all_single",
        "all_gather_into_tensor",
    ):
        if hasattr(dist, name):
            _ORIG[name] = getattr(dist, name)
    dist.reduce_scatter_tensor = _reduce_scatter_tensor
    dist.all_reduce = _all_reduce
    if "_reduce_scatter_base" in _ORIG:
        dist._reduce_scatter_base = _reduce_scatter_base

    from megatron.core.tensor_parallel import mappings

    if mappings.dist_reduce_scatter_func is _ORIG.get("reduce_scatter_tensor"):
        mappings.dist_reduce_scatter_func = _reduce_scatter_tensor
    elif mappings.dist_reduce_scatter_func is _ORIG.get("_reduce_scatter_base"):
        mappings.dist_reduce_scatter_func = _reduce_scatter_base


def configure_inference_token_rounder(tp_size: int) -> None:
    """Raise MCore's batch-invariant bucket multiple to lcm(64, 32*TP).

    Each rank's norm under sequence parallelism sees tokens/TP rows, which must
    stay a multiple of 32; 64 suffices only up to TP=2.
    """
    from megatron.core.inference import batch_dimensions_utils
    from megatron.core.inference.contexts import dynamic_context

    rounder = batch_invariant_token_multiple(tp_size)
    batch_dimensions_utils.TOKEN_ROUNDER = rounder
    dynamic_context.DynamicInferenceContext.TOKEN_ROUNDER = rounder
