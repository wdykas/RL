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
"""TP-sharded latent MoE projections under the fp16 recipe (opt-in).

Nemotron Super / Ultra project every MoE layer's input to a latent (``fc1_latent_proj``,
hidden -> latent) and the combined expert output back (``fc2_latent_proj``, latent ->
hidden). Megatron builds both as ``parallel_mode="duplicated"`` linears: every TP rank holds
the whole weight (Ultra: 1.61B parameters -> +9 GB per learner rank with fp32 grads and
distributed Adam at DP x CP 32, +2.8 GB per engine GPU per weight version).

``shard_model`` keeps 1/TP of each weight per TP rank (rows = output features, dim 0) and
marks the module column-parallel for Megatron-Bridge (``tensor_model_parallel``,
``partition_dim = 0``: HF import scatters the rows, export / refit gathers them) and for
distributed checkpoints (``sharded_state_dict`` with a TP axis). The forward computes the
same numbers as the duplicated layer, bit for bit, in one of two ways:

- ``weight`` (learner, and any call with many rows): all-gather the weight shards (a copy)
  and run the recipe's fp16 GEMM on the full weight -- the duplicated layer's own call.
  Backward: each rank's wgrad of the full weight (bf16-rounded, as Megatron's DDP hook
  receives it) is reduce-scattered in fp32 over TP straight into the shard's ``main_grad``
  (the TP sum the duplicated layer gets from ``sequence_parallel`` grad all-reduce).
- ``rows`` (engine decode, few rows): all-gather the fp16 input rows over TP, multiply by
  the local weight shard ([T, N/TP] fp32), and all-to-all the column blocks back to each
  rank's rows ([T/TP, N]). Every output element is the same fp16 x fp16 -> fp32 dot
  product over the whole K; ``probe`` checks that the GEMM on a column block gives the full
  GEMM's bits (it is the batch-invariant cuBLAS GEMM without split-K; the property used is
  the column analogue of the row invariance the whole recipe relies on).

Per-rank cost (TP ranks t, rows per rank r, hidden d, latent l, per MoE layer):
memory 2 d l (2/t) bytes of bf16 / fp16 weight instead of 2 d l (2) -- plus, on the learner,
fp32 main grad and optimizer state on 1/t of the elements; transient full weight of one
layer (2 d l x 2 bytes) during its forward / backward. ``weight`` mode moves the weight:
all-gather 2 d l (t-1)/t x 2 bytes per forward (and per recompute), reduce-scatter of the
fp32 wgrad 2 d l (t-1)/t x 4 bytes per backward. ``rows`` mode moves activations:
all-gather of r t (d + l) fp16 rows and all-to-all of r t (l + d) fp32 values per forward.
Ultra (d 8192, l 2048, t 8, 48 layers): weight all-gather 3.2 GB per forward (~3.5 ms at
~0.9 TB/s), so the engine uses ``rows`` below ``NEMO_RL_LATENT_SHARD_ROWS_MAX`` rows per rank
(default 1024: a decode step moves < 0.1 GB per layer instead of 67 MB of weights).
"""

from __future__ import annotations

import os
from typing import Any

import torch

_STATE: dict[str, Any] = {"enabled": False}


def enabled() -> bool:
    return bool(_STATE["enabled"]) or os.environ.get("NEMO_RL_LATENT_TP_SHARD") == "1"


def install(on: bool = True) -> None:
    _STATE["enabled"] = bool(on)


def _rows_max() -> int:
    return int(os.environ.get("NEMO_RL_LATENT_SHARD_ROWS_MAX", "1024"))


def _mode() -> str:
    """auto | weight | rows (NEMO_RL_LATENT_SHARD_MODE; probes force one)."""
    return os.environ.get("NEMO_RL_LATENT_SHARD_MODE", "auto")


def _make_sharded_state_dict(module: Any):
    def sharded_state_dict(prefix="", sharded_offsets=(), metadata=None):
        from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

        state_dict = module.state_dict(prefix="", keep_vars=True)
        dp_cp = (metadata or {}).get("dp_cp_group")
        if dp_cp is None:
            from megatron.core import parallel_state as ps

            dp_cp = ps.get_data_parallel_group(with_context_parallel=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {"weight": 0},
            sharded_offsets,
            tp_group=module._ts_latent_shard["group"],
            dp_cp_group=dp_cp,
        )

    return sharded_state_dict


def shard_linear(lin: Any, group: Any) -> bool:
    """Replace a duplicated linear's weight by this TP rank's row block (in place)."""
    if getattr(lin, "_ts_latent_shard", None) is not None:
        return False
    tp = torch.distributed.get_world_size(group)
    if tp == 1:
        return False
    rank = torch.distributed.get_rank(group)
    w = lin.weight
    out = int(w.shape[0])
    if out % tp:
        raise ValueError(f"latent projection rows {out} not divisible by TP {tp}")
    if getattr(lin, "bias", None) is not None and lin.bias.numel():
        raise NotImplementedError("latent TP shard: biased latent projections are not built")
    rows = out // tp
    shard = torch.nn.Parameter(
        w.detach()[rank * rows : (rank + 1) * rows].clone(), requires_grad=w.requires_grad
    )
    for k, v in w.__dict__.items():
        if not k.startswith("_"):
            setattr(shard, k, v)
    # Gradients are complete per shard after the in-backward reduce-scatter: no TP all-reduce
    # (finalize_model_grads' sequence-parallel pass), TP-sharded for grad norm / optimizer.
    shard.sequence_parallel = False
    shard.tensor_model_parallel = True
    shard.partition_dim = 0
    shard.partition_stride = 1
    shard.allreduce = True
    del lin.weight
    lin.register_parameter("weight", shard)
    lin.tensor_model_parallel = True  # Megatron-Bridge AutoMapping: column-parallel (dim 0)
    lin.partition_dim = 0
    lin._ts_latent_shard = {"group": group, "out": out, "tp": tp, "rank": rank}
    lin.sharded_state_dict = _make_sharded_state_dict(lin)
    return True


def shard_model(model: Any) -> int:
    """Shard every MoE layer's latent projections over its TP group. Returns the count."""
    from megatron.core.transformer.moe.moe_layer import MoELayer

    chunks = model if isinstance(model, (list, tuple)) else [model]
    n = 0
    for chunk in chunks:
        for m in chunk.modules():
            if not isinstance(m, MoELayer) or not getattr(m.config, "moe_latent_size", None):
                continue
            group = getattr(m, "tp_group", None)
            if group is None:
                from megatron.core import parallel_state as ps

                group = ps.get_tensor_model_parallel_group()
            for name in ("fc1_latent_proj", "fc2_latent_proj"):
                lin = getattr(m, name, None)
                if lin is not None and shard_linear(lin, group):
                    n += 1
    return n


def pre_wrap_hook(model: list) -> list:
    """Megatron-Bridge ``pre_wrap_hook``: shard before DDP sizes its buffers."""
    if enabled():
        n = shard_model(model)
        if n and torch.distributed.get_rank() == 0:
            print(f"[latent_tp_shard] {n} latent projections sharded over TP", flush=True)
    return model


# ---------------------------------------------------------------------------
# Forward paths.
# ---------------------------------------------------------------------------


def _gather_weight(shard: torch.Tensor, group: Any) -> torch.Tensor:
    from nemo_rl.models.megatron.fp16_gemm import _gather_rows_dim0

    return _gather_rows_dim0(shard.detach(), group)


class _ShardedLinear(torch.autograd.Function):
    """``Fp16Linear`` (no TP comm) on the all-gathered weight; the full weight is never saved.

    Forward: x -> fp16 (the recipe's site cast), fp32 GEMM with the gathered weight.
    Backward (bf16 operands, fp32 accumulation, as ``Fp16Linear.backward``): the weight is
    gathered again; dgrad on the full weight; the full wgrad is rounded to the weight dtype
    (what Megatron DDP's hook would add for a duplicated weight), summed over TP in fp32
    (``ordered_reduce_scatter_rows``) and added into the shard's ``main_grad``.
    """

    @staticmethod
    def forward(ctx, x, shard, group, site):  # noqa: D102
        from nemo_rl.models.megatron import fp16_gemm as fg

        x16 = fg.cast_fp16_counted(x, site)
        out = fg.fp16_gemm(x16, _gather_weight(shard, group))
        ctx.save_for_backward(x16, shard)
        ctx.group = group
        ctx.x_dtype = x.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):  # noqa: D102
        from nemo_rl.models.megatron import fp16_gemm as fg

        x16, shard = ctx.saved_tensors
        w = _gather_weight(shard, ctx.group)
        n = w.shape[0]
        g2 = grad_out.reshape(-1, n).to(torch.bfloat16).contiguous()
        dgrad = None
        if ctx.needs_input_grad[0]:
            dgrad = fg._mm(g2, w.to(torch.bfloat16), torch.float32)
            dgrad = dgrad.view(*x16.shape[:-1], w.shape[1]).to(ctx.x_dtype)
        del w
        wgrad = None
        if ctx.needs_input_grad[1]:
            x2 = x16.reshape(-1, x16.shape[-1]).to(torch.bfloat16)
            full = fg._mm(g2.t(), x2, torch.float32).to(shard.dtype).float()
            g = fg.ordered_reduce_scatter_rows(full, ctx.group)
            mg = getattr(shard, "main_grad", None)
            if mg is not None and mg.shape == g.shape and hasattr(shard, "grad_added_to_main_grad"):
                mg.add_(g)
                shard.grad_added_to_main_grad = True
                wgrad = torch.empty_like(shard)  # Megatron's dummy-gradient contract
            else:
                wgrad = g.to(shard.dtype)
        return dgrad, wgrad, None, None


def _rows_forward(x16: torch.Tensor, shard: torch.Tensor, info: dict) -> torch.Tensor:
    """[T/TP, K] fp16 rows -> [T/TP, N] fp32 via the local column block (no autograd)."""
    from nemo_rl.models.megatron.fp16_gemm import _gather_rows_dim0, fp16_gemm

    tp, group = info["tp"], info["group"]
    k = x16.shape[-1]
    lead = x16.shape[:-1]
    x2 = x16.reshape(-1, k).contiguous()
    rows = x2.shape[0]
    xa = _gather_rows_dim0(x2, group)  # [T, K]
    part = fp16_gemm(xa, shard)  # [T, N/TP] fp32: this rank's column block for every row
    nb = part.shape[1]
    send = part.view(tp, rows, nb).contiguous()  # row block j -> rank j
    recv = torch.empty_like(send)
    torch.distributed.all_to_all_single(recv, send, group=group)
    # recv[i] = rank i's column block of my rows -> [rows, TP * nb]
    out = recv.permute(1, 0, 2).reshape(rows, tp * nb)
    return out.view(*lead, tp * nb)


def linear_forward(module: Any, x: torch.Tensor) -> torch.Tensor:
    """fp32 output of a sharded latent projection (the duplicated layer's bits)."""
    from nemo_rl.models.megatron import fp16_gemm as fg

    info = module._ts_latent_shard
    shard = module.weight
    tracked = torch.is_grad_enabled() and (x.requires_grad or shard.requires_grad)
    mode = _mode()
    rows = x.numel() // x.shape[-1]
    if mode == "rows" or (mode == "auto" and not tracked and rows <= _rows_max()):
        if tracked:
            raise RuntimeError("latent TP shard: rows mode has no backward")
        return _rows_forward(fg.cast_fp16_counted(x, module), shard, info)
    if tracked:
        return _ShardedLinear.apply(x, shard, info["group"], module)
    return fg.fp16_gemm(fg.cast_fp16_counted(x, module), _gather_weight(shard, info["group"]))


def patch_bridge_mapping() -> None:
    """Megatron-Bridge AutoMapping: parallelism of the latent projections (idempotent).

    - a sharded projection (``_ts_latent_shard``) is column-parallel (rows split over TP);
    - the engine's ``InferenceLinear`` (inference_optimized latent projections, duplicated) is
      not in Bridge's module-type registry: at TP > 1 loading or exporting it raised "Cannot
      determine parallelism type for module 'InferenceLinear'" (Super-L8 engine at TP 2,
      research/scaleup/pp-super-tp2ep2.log); it is replicated, like the training TELinear.
    """
    try:
        from megatron.bridge.models.conversion.param_mapping import AutoMapping
    except ImportError:  # pragma: no cover
        return
    if getattr(AutoMapping, "_ts_latent_patched", False):
        return
    orig = AutoMapping._detect_parallelism_type

    def _detect_parallelism_type(self, module):
        if getattr(module, "_ts_latent_shard", None) is not None:
            return "column"
        if type(module).__name__ == "InferenceLinear" and getattr(module, "parallel_mode", None) in (
            None,
            "duplicated",
        ):
            return "replicated"
        return orig(self, module)

    AutoMapping._detect_parallelism_type = _detect_parallelism_type
    AutoMapping._ts_latent_patched = True
