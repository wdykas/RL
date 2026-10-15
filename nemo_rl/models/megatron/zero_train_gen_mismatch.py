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

"""Zero train/generation KL (batch-invariant Megatron): validate the recipe, then enable.

The recipe sets every required value; this module only checks them. Only BF16
(no FP8) is supported.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field, fields
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

import torch
from packaging.version import Version

from nemo_rl.algorithms.logits_sampling_utils import (
    TrainingSamplingParams,
    need_top_k_or_top_p_filtering,
)
from nemo_rl.models.generation.megatron.config import merged_inference_megatron_cfg
from nemo_rl.models.megatron import bi_mean, mamba_ssd_parity, moe_deterministic
from nemo_rl.models.megatron.batch_invariant import batch_invariant_token_multiple

if TYPE_CHECKING:
    from nemo_rl.models.policy import PolicyConfig

# TransformerConfig fields the installed Megatron-Core must define for this mode.
MEGATRON_CORE_REQUIRED_CONFIG_FIELDS = (
    "batch_invariant_mode",
    "batch_invariant_backend",
    "batch_invariant_collective",
    "flash_attention_version",
)

TRANSFORMER_ENGINE_MIN_VERSION = Version("2.18")
FLASH_ATTN_MIN_VERSION = Version("2.8.1")
CUTEDSL_MIN_VERSION = Version("4.6.0.dev0")

_ZERO_KL_MEGATRON_REQUIRED: dict[str, Any] = {
    "batch_invariant_mode": True,
    "moe_permute_fusion": False,
    "attention_backend": "flash",
    "flash_attention_version": 4,
    "batch_invariant_backend": "te_native",
    "batch_invariant_collective": "ordered",
}

_ZERO_KL_GENERATION_REQUIRED: dict[str, Any] = {
    "enable_chunked_prefill": False,
}

# Workspace-free cuBLAS (fixed reduction order) for the te_native backend.
_TE_NATIVE_ENV_VARS = {
    "CUBLASLT_WORKSPACE_SIZE": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":0:0",
}


@dataclass
class ZeroTrainGenValidation:
    """Collected validation outcome."""

    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_invalid(self, header: str) -> None:
        if not self.violations:
            return
        bullet = "\n".join(f"  - {msg}" for msg in self.violations)
        raise ValueError(f"{header}\n{bullet}")

    def emit_warnings(self) -> None:
        for msg in self.warnings:
            warnings.warn(msg, UserWarning, stacklevel=3)


def validate_zero_train_gen_mismatch(
    config: PolicyConfig,
    *,
    check_packages: bool = True,
    check_platform: bool = True,
) -> ZeroTrainGenValidation:
    """Run all zero-KL gates; return violations and non-fatal warnings."""
    out = ZeroTrainGenValidation()
    if not config["megatron_cfg"].get("zero_train_gen_mismatch"):
        return out

    _validate_required_values(config, out)
    _validate_backend(config, out)
    _validate_platform(out, check_device=check_platform)
    if check_packages:
        _validate_packages(out)
    _validate_precision(config, out)
    return out


def validate_batch_invariant_mode(config: PolicyConfig) -> ZeroTrainGenValidation:
    """Checks for ``batch_invariant_mode=True`` without enabling MCore kernels."""
    out = ZeroTrainGenValidation()
    megatron_cfg = config["megatron_cfg"]
    if not megatron_cfg.get("batch_invariant_mode"):
        return out

    required_fields = (
        "batch_invariant_backend",
        "batch_invariant_collective",
        "flash_attention_version",
    )
    missing_fields = [field for field in required_fields if field not in megatron_cfg]
    if missing_fields:
        out.violations.append(
            "batch_invariant_mode=True requires policy.megatron_cfg fields: "
            f"{', '.join(missing_fields)}."
        )

    # Any TP, provided generation uses the identical parallel layout (checked
    # with the matching fields below). TP>=3 parity rests on NCCL's
    # reduce-scatter giving a token the same bits at every batch size; measure
    # it with research/thundersync_rl/scripts/bi_tp/nccl_tp_invariance.py
    # (research/thundersync_rl/DESIGN.md section 14).
    _validate_batch_invariant_tensor_parallel(config, out)
    if megatron_cfg.get("context_parallel_size") != 1:
        out.violations.append(
            "batch_invariant_mode=True currently requires training context "
            "parallel size 1."
        )
    if config["sequence_packing"]["enabled"]:
        out.violations.append(
            "batch_invariant_mode=True requires sequence_packing.enabled=False: "
            "packing changes microbatch composition and the packed log-prob path "
            "does not share generation's fused log-softmax."
        )
    token_multiple = batch_invariant_token_multiple(
        megatron_cfg["tensor_model_parallel_size"]
    )
    if megatron_cfg["tensor_model_parallel_size"] > 1:
        # Under sequence parallelism each rank's RMSNorm sees tokens/TP rows,
        # which must stay a multiple of 32 (TE native norm bit class):
        # lcm(64, 32*TP), i.e. 64 at TP=2.
        token_multiple = math.lcm(
            token_multiple, 32 * megatron_cfg["tensor_model_parallel_size"]
        )
    if config["make_sequence_length_divisible_by"] % token_multiple:
        out.violations.append(
            "batch_invariant_mode=True requires policy.make_sequence_length_divisible_by "
            f"to be a multiple of {token_multiple} to align with MCore generation "
            f"buckets (got {config['make_sequence_length_divisible_by']})."
        )
    dynamic_batching = config["dynamic_batching"]
    if (
        dynamic_batching["enabled"]
        and dynamic_batching["sequence_length_round"] % token_multiple
    ):
        out.violations.append(
            "batch_invariant_mode=True requires "
            f"policy.dynamic_batching.sequence_length_round to be a multiple of "
            f"{token_multiple} (got {dynamic_batching['sequence_length_round']})."
        )
    if megatron_cfg.get("batch_invariant_backend") == "te_native":
        env_vars = megatron_cfg.get("env_vars") or {}
        # Opt-in equivalent of CUBLAS_WORKSPACE_CONFIG=:0:0 (run_ab_arm.sh CUBLAS_WS_SHIM=1):
        # the LD_PRELOAD shim thundersync_work/research/callcost/cublas_ws_shim gives every cuBLAS
        # handle a 0-byte workspace (same cuBLAS state, no split-K) without the variable, which
        # PyTorch 2.11 re-parses with a fresh std::regex on every cuBLAS call (~35 us host).
        shim = "cublas_ws_shim" in str(env_vars.get("LD_PRELOAD") or "") and str(
            env_vars.get("CUBLAS_WORKSPACE_FIXED_BYTES")
        ) == "0"
        for key, value in _TE_NATIVE_ENV_VARS.items():
            if key == "CUBLAS_WORKSPACE_CONFIG" and shim and env_vars.get(key) is None:
                continue
            if env_vars.get(key) != value:
                out.violations.append(
                    f"batch_invariant_backend='te_native' requires "
                    f"policy.megatron_cfg.env_vars.{key}={value!r} "
                    f"(got {env_vars.get(key)!r})."
                )
    if megatron_cfg.get("use_fused_linear_logprobs"):
        out.violations.append(
            "batch_invariant_mode=True is incompatible with "
            "use_fused_linear_logprobs=True because generation parity requires "
            "the shared float-log_softmax-gather path."
        )
    if megatron_cfg.get("attention_backend") != "flash":
        out.violations.append(
            "batch_invariant_mode=True requires "
            "policy.megatron_cfg.attention_backend='flash'."
        )
    fa_ver = megatron_cfg.get("flash_attention_version")
    if fa_ver not in (3, 4):
        out.violations.append(
            "batch_invariant_mode=True requires "
            "policy.megatron_cfg.flash_attention_version to be 3 or 4."
        )

    generation_cfg = config.get("generation")
    if generation_cfg is None or generation_cfg.get("backend") != "megatron":
        out.violations.append(
            "batch_invariant_mode=True requires policy.generation.backend='megatron'."
        )
    else:
        try:
            inference_cfg = merged_inference_megatron_cfg(config)
        except ValueError as exc:
            if str(exc) not in out.violations:  # the TP>1 checks may report it
                out.violations.append(str(exc))
            return out
        if (
            inference_cfg.get("transformer_impl") == "inference_optimized"
            and config.get("precision") != "bfloat16"
        ):
            out.violations.append(
                "batch_invariant_mode=True with "
                "transformer_impl='inference_optimized' requires "
                "policy.precision='bfloat16'."
            )
        if inference_cfg.get("logprobs_mode") != "raw_logprobs":
            out.violations.append(
                "batch_invariant_mode=True requires "
                "policy.generation.mcore_generation_config.logprobs_mode='raw_logprobs' "
                f"(got {inference_cfg.get('logprobs_mode')!r})."
            )
        sampling = TrainingSamplingParams(
            top_k=generation_cfg["top_k"],
            top_p=generation_cfg["top_p"],
            temperature=generation_cfg["temperature"],
        )
        if sampling.temperature != 1.0 or need_top_k_or_top_p_filtering(sampling):
            out.violations.append(
                "batch_invariant_mode=True scores raw logprobs, so it requires "
                "policy.generation.temperature=1.0, top_p=1.0, top_k=null "
                f"(got temperature={sampling.temperature}, top_p={sampling.top_p}, "
                f"top_k={sampling.top_k})."
            )
        # Batch invariance needs the identical parallel layout on both sides:
        # a different TP changes the row-parallel GEMMs' K split, and EP/expert
        # TP change the collectives and expert GEMM shapes. Pipeline parallelism
        # only moves activations between stages (point to point, no reduction),
        # so training and inference may use different PP (e.g. inference PP=1,
        # where Megatron inference at PP>1 under BI was measured to generate
        # garbage: reward 0.05, every rollout at the length cap).
        matching_fields = (
            "tensor_model_parallel_size",
            "expert_model_parallel_size",
            "expert_tensor_parallel_size",
            "context_parallel_size",
            "batch_invariant_mode",
            "batch_invariant_backend",
            "batch_invariant_collective",
            "attention_backend",
            "flash_attention_version",
            "fp32_residual_connection",
            "forward_gemm_precision",
            "lm_head_precision",
            "down_proj_prescale_log2",
            "moe_ep_invariant_combine",
        )
        if megatron_cfg.get("moe_ep_invariant_combine", False) or (
            os.environ.get("THUNDERSYNC_ALLOW_EP_MISMATCH") == "1"
        ):
            # The expert-parallel size may differ between training and generation
            # when the MoE combine is EP-invariant (ep_invariant_combine.py: exact
            # fixed-point limbs, same bits at every EP). Measured
            # (thundersync_work/runs_1node/epinv-lanes-ep, Nano L8, learner EP2,
            # engine lanes EP1, forced fresh rows every step): 4/4 steps mismatch 0;
            # control without the combine (epinv-lanes-ep-off) max |dlogp| 0.66-0.79.
            # Engine EP1/2/4 greedy decode dumps bitwise identical (epinv-checks).
            # THUNDERSYNC_ALLOW_EP_MISMATCH=1 without the combine: debug only
            # (inexact; the old combine is EP-dependent).
            matching_fields = tuple(f for f in matching_fields if f != "expert_model_parallel_size")
        mismatched_fields = [
            field
            for field in matching_fields
            if _resolved(inference_cfg, field) != _resolved(megatron_cfg, field)
        ]
        if mismatched_fields:
            out.violations.append(
                "Training and generation must use the same Megatron settings for "
                f"batch invariance: {', '.join(mismatched_fields)}."
            )

    return out


def _resolved(cfg: dict[str, Any], field: str) -> Any:
    """Config value as Megatron resolves it (an unset expert TP means TP)."""
    value = cfg.get(field)
    if field == "expert_tensor_parallel_size" and value is None:
        return cfg.get("tensor_model_parallel_size")
    return value


def _validate_batch_invariant_tensor_parallel(
    config: PolicyConfig, out: ZeroTrainGenValidation
) -> None:
    """Training and generation must run the same tensor-parallel collectives.

    Both sides reduce-scatter row-parallel outputs and all-gather norm outputs
    with NCCL under sequence parallelism (``inference_optimized`` layers require
    it at TP>1). Fused TP communication and TE userbuffer overlap swap those
    collectives for other kernels. Layout equality (TP, PP, EP, expert TP)
    between the two sides is checked with the other matching fields.
    """
    megatron_cfg = config["megatron_cfg"]
    tp_size = megatron_cfg.get("tensor_model_parallel_size")
    tensor_parallel = isinstance(tp_size, int) and tp_size > 1
    prefix = f"batch_invariant_mode=True with tensor_model_parallel_size={tp_size}"
    if tensor_parallel and not megatron_cfg.get("sequence_parallel"):
        out.violations.append(
            f"{prefix} requires policy.megatron_cfg.sequence_parallel=true "
            "(generation's inference_optimized layers run sequence parallel; "
            "training must use the same reduce-scatter/all-gather path and norm "
            "row counts)."
        )
    sides = [("policy.megatron_cfg", megatron_cfg)]
    generation_cfg = config.get("generation")
    if generation_cfg is not None and generation_cfg.get("backend") == "megatron":
        try:
            inference_cfg = merged_inference_megatron_cfg(config)
        except ValueError as exc:
            # e.g. inference_optimized without sequence parallelism.
            out.violations.append(str(exc))
            inference_cfg = None
        if inference_cfg is not None:
            sides.append(("policy.generation.mcore_generation_config", inference_cfg))
            if tensor_parallel and not inference_cfg.get("sequence_parallel"):
                out.violations.append(
                    f"{prefix} requires policy.generation.mcore_generation_config."
                    "sequence_parallel=true (or unset, inheriting training's)."
                )
    for label, cfg in sides:
        for key in ("inference_fuse_tp_communication", "tp_comm_overlap"):
            if cfg.get(key):
                out.violations.append(
                    f"{prefix} requires {label}.{key}=false: it replaces the NCCL "
                    "reduce-scatter/all-gather that training and generation share "
                    "(fused TP inference also reads a stale symmetric buffer "
                    "under batch-invariant mode)."
                )


def allow_installed_flash_attn_4() -> None:
    """Enable MCore's FA4 path for the pinned ``flash-attn-4`` b19.

    MCore gates ``HAVE_FA4`` on ``>=4.0.0b20``, which needs a newer
    ``apache-tvm-ffi`` than we pin. The call signature is identical.
    """
    # Deferred: Megatron-Core exists only in worker venvs.
    import megatron.core.transformer.attention as mcore_attention

    if getattr(mcore_attention, "HAVE_FA4", False):
        return
    if _first_package_version(("flash-attn-4", "flash_attn_4")) is None:
        return
    # Deferred: optional dependency, checked above.
    from flash_attn.cute import flash_attn_varlen_func

    mcore_attention.flash_attn4_varlen_func = flash_attn_varlen_func
    mcore_attention.HAVE_FA4 = True


# Upper bound on gathered tokens (batch x sequence chunk) per full-vocabulary
# log-softmax when the caller passes no chunk size: bounds the [tokens, V]
# fp32 temporaries of the TP>1 gathered path (~2.4 GB at V=151936).
_GATHERED_LOGPROB_CHUNK_TOKENS = 4096


def _seq_chunks(seq_len: int, chunk: int):
    for start in range(0, seq_len, chunk):
        yield start, min(seq_len, start + chunk)


def _gathered_chunk_size(logits: torch.Tensor, chunk_size: int | None) -> int:
    if chunk_size is not None:
        return chunk_size
    batch = max(1, int(logits.shape[0]))
    return max(1, _GATHERED_LOGPROB_CHUNK_TOKENS // batch)


def _gathered_target_logprobs(
    gather_chunk: Any, target: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """Selected-token log-probs from full-vocabulary chunks.

    ``gather_chunk(start, end)`` returns the full-vocabulary logits
    ``[B, end - start, V]`` of a sequence slice. Each chunk takes exactly the
    TP=1 path of ``model_utils._tp_target_logprobs`` (fp32 cast, one
    full-row ``log_softmax``, gather), which is what Megatron inference runs
    after its vocab all-gather; under batch-invariant kernels the row-wise
    ``log_softmax`` does not depend on the chunking.
    """
    out = []
    for start, end in _seq_chunks(int(target.shape[1]), chunk_size):
        full = gather_chunk(start, end).to(torch.float32)
        log_probs = torch.nn.functional.log_softmax(full, dim=-1)
        out.append(log_probs.gather(-1, target[:, start:end, None]).squeeze(-1))
        del full, log_probs
    return (torch.cat(out, dim=1) if len(out) > 1 else out[0]).contiguous()


def _gathered_logprob_grad(
    gather_chunk: Any,
    target: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    chunk_size: int,
    grad_input: torch.Tensor,
) -> torch.Tensor:
    """Gradient of the selected log-probs w.r.t. this rank's vocab shard.

    d logp(t) / d z_v = 1[v == t] - softmax(z)_v, so the local slice is
    ``g * (onehot - p)[..., start:end]``; written as ``g - p * g`` at the
    target (as ``log_softmax``'s own backward forms it) and ``-(p * g)``
    elsewhere. Fills and returns ``grad_input`` ``[B, S, V_local]``.
    """
    for start, end in _seq_chunks(int(target.shape[1]), chunk_size):
        full = gather_chunk(start, end).to(torch.float32)
        probs = torch.nn.functional.log_softmax(full, dim=-1)[
            ..., vocab_start_index:vocab_end_index
        ].exp()
        del full
        g = grad_output[:, start:end].to(torch.float32)
        chunk_grad = probs.mul_(g.unsqueeze(-1)).neg_()
        chunk_target = target[:, start:end]
        local = (chunk_target >= vocab_start_index) & (chunk_target < vocab_end_index)
        index = (chunk_target - vocab_start_index).clamp_(
            0, vocab_end_index - vocab_start_index - 1
        )
        chosen = chunk_grad.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        chunk_grad.scatter_(
            -1,
            index.unsqueeze(-1),
            torch.where(local, g + chosen, chosen).unsqueeze(-1),
        )
        grad_input[:, start:end].copy_(chunk_grad)
        del probs, chunk_grad
    return grad_input


def _all_gather_vocab(
    local_chunk: torch.Tensor, group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """Concatenate the TP ranks' vocab shards in rank (= vocabulary) order."""
    local_chunk = local_chunk.contiguous()
    parts = [
        torch.empty_like(local_chunk)
        for _ in range(torch.distributed.get_world_size(group))
    ]
    torch.distributed.all_gather(parts, local_chunk, group=group)
    return torch.cat(parts, dim=-1)


class _GatheredVocabLogprob(torch.autograd.Function):
    """TP>1 log-probs computed exactly as Megatron inference computes them.

    Training's ``DistributedLogprob`` normalizes with MAX/SUM all-reduces of
    partial statistics; inference all-gathers the logits and runs one
    full-vocabulary ``log_softmax``. This function gathers the vocab shards
    per sequence chunk and runs the TP=1 path, so training and inference
    log-probs agree bitwise. Only the local bf16/fp32 shard is saved; backward
    re-gathers per chunk and returns the gradient of the local shard.
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
        chunk_size: int | None,
        group: torch.distributed.ProcessGroup,
        inference_only: bool = False,
    ) -> torch.Tensor:
        local_vocab = int(vocab_parallel_logits.shape[-1])
        if vocab_end_index - vocab_start_index != local_vocab or (
            vocab_start_index != torch.distributed.get_rank(group) * local_vocab
        ):
            raise ValueError(
                "batch-invariant TP log-probs expect an even vocab split in rank "
                f"order (rank {torch.distributed.get_rank(group)}, local vocab "
                f"{local_vocab}, range [{vocab_start_index}, {vocab_end_index}))."
            )
        chunk = _gathered_chunk_size(vocab_parallel_logits, chunk_size)

        def gather_chunk(start: int, end: int) -> torch.Tensor:
            return _all_gather_vocab(vocab_parallel_logits[:, start:end], group)

        log_probs = _gathered_target_logprobs(gather_chunk, target, chunk)
        if not inference_only:
            ctx.save_for_backward(vocab_parallel_logits, target)
            ctx.vocab_range = (vocab_start_index, vocab_end_index)
            ctx.chunk = chunk
            ctx.group = group
        return log_probs

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        vocab_parallel_logits, target = ctx.saved_tensors
        group = ctx.group

        def gather_chunk(start: int, end: int) -> torch.Tensor:
            return _all_gather_vocab(vocab_parallel_logits[:, start:end], group)

        grad_input = _gathered_logprob_grad(
            gather_chunk,
            target,
            grad_outputs[0],
            vocab_start_index=ctx.vocab_range[0],
            vocab_end_index=ctx.vocab_range[1],
            chunk_size=ctx.chunk,
            grad_input=torch.zeros_like(vocab_parallel_logits),
        )
        return grad_input, None, None, None, None, None, None


class _LocalVocabLogprob(torch.autograd.Function):
    """TP=1 log-probs that save only the logits (backward recomputes per chunk).

    Forward is ``model_utils._tp_target_logprobs``'s TP=1 path chunk by chunk
    (fp32 cast, row-wise ``log_softmax``, gather), so the log-probs are bitwise
    the same. Autograd through that path would also keep every chunk's
    ``log_softmax`` output: one more [B, S, V] fp32 buffer (9.3 GiB for 16384
    tokens x 151936 vocab). Here backward forms ``g * (onehot - softmax)`` one
    sequence chunk at a time, as ``_GatheredVocabLogprob`` does under TP>1.
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        logits: torch.Tensor,
        target: torch.Tensor,
        chunk_size: int | None,
    ) -> torch.Tensor:
        chunk = _gathered_chunk_size(logits, chunk_size)

        def chunk_of(start: int, end: int) -> torch.Tensor:
            return logits[:, start:end]

        log_probs = _gathered_target_logprobs(chunk_of, target, chunk)
        ctx.save_for_backward(logits, target)
        ctx.chunk = chunk
        return log_probs

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple:
        logits, target = ctx.saved_tensors

        def chunk_of(start: int, end: int) -> torch.Tensor:
            return logits[:, start:end]

        grad = _gathered_logprob_grad(
            chunk_of,
            target,
            grad_outputs[0],
            vocab_start_index=0,
            vocab_end_index=int(logits.shape[-1]),
            chunk_size=ctx.chunk,
            grad_input=torch.zeros_like(logits),
        )
        return grad, None, None


class _BatchInvariantLogprob:
    """Log-probs computed the way Megatron Inference computes them (bitwise).

    Replaces ``model_utils.DistributedLogprob`` / ``ChunkedDistributedLogprob``
    under batch-invariant mode. TP=1: the local full-vocabulary log-softmax.
    TP>1: gather the vocab shards (chunked) and run the same log-softmax, as
    inference does after its logits all-gather (``_GatheredVocabLogprob``).
    """

    def __init__(self, original: Any, *, chunked: bool) -> None:
        self._original = original
        self._chunked = chunked

    def apply(self, *args: Any) -> Any:
        # Deferred: model_utils is heavy and only needed once the shim is installed.
        from nemo_rl.distributed import model_utils

        if self._chunked:
            logits, target, start, end, chunk_size, group, inference_only = args
        else:
            logits, target, start, end, group, inference_only = args
            chunk_size = None
        if group is None or torch.distributed.get_world_size(group) == 1:
            from nemo_rl.models.megatron.deferred_head import DeferredHeadLogits

            if isinstance(logits, DeferredHeadLogits):
                # The training head's logits, never materialized for the whole
                # micro-batch (deferred_head.py); the same chunking as below.
                if inference_only or not torch.is_grad_enabled():
                    logits = logits.materialize()
                else:
                    return logits.target_logprobs(
                        target, _gathered_chunk_size(logits, chunk_size)
                    )
            if not inference_only and torch.is_grad_enabled() and logits.requires_grad:
                return _LocalVocabLogprob.apply(logits, target, chunk_size)
            return model_utils._tp_target_logprobs(
                logits,
                target,
                vocab_start_index=start,
                vocab_end_index=end,
                tp_group=None,
                chunk_size=chunk_size,
                inference_only=inference_only,
            )
        return _GatheredVocabLogprob.apply(
            logits, target, start, end, chunk_size, group, inference_only
        )


def _use_inference_log_softmax() -> None:
    # Deferred: model_utils is heavy and only needed once the shim is installed.
    from nemo_rl.distributed import model_utils

    for name, chunked in (
        ("DistributedLogprob", False),
        ("ChunkedDistributedLogprob", True),
    ):
        current = getattr(model_utils, name)
        if not isinstance(current, _BatchInvariantLogprob):
            setattr(model_utils, name, _BatchInvariantLogprob(current, chunked=chunked))


def enable_batch_invariant_kernels(config: PolicyConfig) -> None:
    """Pin TE FA support and call MCore ``enable_batch_invariant_mode``."""
    megatron_cfg = config["megatron_cfg"]
    allow_installed_flash_attn_4()
    collective = megatron_cfg["batch_invariant_collective"]

    # Deferred: Megatron-Core exists only in worker venvs.
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        assert_te_supports_batch_invariant_attention,
    )
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        enable_batch_invariant_mode as enable_mcore_batch_invariant_mode,
    )

    assert_te_supports_batch_invariant_attention()
    # Full reductions (x.mean()) must be means, not sums, under the registered mean.dim.
    bi_mean.install()
    enable_mcore_batch_invariant_mode(
        backend=megatron_cfg["batch_invariant_backend"], collective=collective
    )
    # Opt-in: bitwise-reproducible training (policy.megatron_cfg.deterministic_training).
    # Batch invariance (train == generation logprobs) does not need it.
    deterministic = bool(megatron_cfg.get("deterministic_training", False))
    # Mamba: train on the engine's SSD kernels, pinned the same way in every process
    # (+ fixed-order conv weight/bias grads and mamba_ssm deterministic mode if deterministic).
    mamba_ssd_parity.install(deterministic=deterministic)
    if deterministic:
        # MoE: reproducible token-gather backward (no atomics over a token's top-k copies).
        moe_deterministic.install()
        # Attention: Transformer Engine picks the deterministic flash-attention
        # backward when this is 0 at module construction (this runs before the
        # model is built). With the MoE gather above, repeating a micro-batch's
        # fwd+bwd gives bitwise equal gradients on Nemotron-3-Nano L8; measured
        # cost: none within noise (0.40-0.42 s vs 0.42-0.43 s per fwd+bwd).
        os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
        # Megatron's deterministic mode (Bridge _validate_and_apply_deterministic_mode):
        # torch's deterministic implementations of every op that has one (index_add /
        # scatter / embedding backward without atomics, ...). warn_only: the batch-
        # invariant recipe runs cuBLAS with CUBLAS_WORKSPACE_CONFIG=:0:0 (no workspace,
        # hence no split-K: deterministic), which torch's allowlist (:4096:8, :16:8)
        # does not recognize; ops with no deterministic implementation are named in
        # warnings instead of raising. No fill of uninitialized memory (Megatron's
        # default in deterministic mode; it costs a write per allocation).
        import torch

        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.utils.deterministic.fill_uninitialized_memory = False
        if os.environ.get("NCCL_ALGO") not in ("Tree", "Ring", "CollnetDirect", "CollnetChain", "^NVLS"):
            print(
                "[zero_train_gen_mismatch] deterministic_training: NCCL_ALGO is "
                f"{os.environ.get('NCCL_ALGO')!r}; Megatron's deterministic mode requires one of "
                "Tree/Ring/CollnetDirect/CollnetChain/^NVLS (NVLS reductions are not "
                "reproducible); set policy.megatron_cfg.env_vars.NCCL_ALGO",
                flush=True,
            )
    _use_inference_log_softmax()
    # Training LM head (fp32 head, TP=1): logits formed chunk by chunk inside the
    # log-prob op, never for a whole micro-batch (deferred_head.py). Forward
    # log-probs are bitwise the same; per-rank head memory no longer grows with
    # tokens per micro-batch.
    from nemo_rl.models.megatron import fp16_gemm

    fp16_gemm._STATE["deferred_head"] = bool(megatron_cfg.get("deferred_lm_head", True))
    print(
        "[zero_train_gen_mismatch] batch-invariant kernels enabled: "
        f"backend={megatron_cfg['batch_invariant_backend']} "
        f"collective={collective} "
        f"deterministic_training={deterministic} "
        f"deferred_lm_head={fp16_gemm._STATE['deferred_head']} "
        f"flash_attention_version={megatron_cfg['flash_attention_version']} "
        f"CUTE_DSL_LIBS={os.environ.get('CUTE_DSL_LIBS')}",
        flush=True,
    )


def validate_zero_train_gen_kl(
    config: PolicyConfig, *, check_environment: bool
) -> None:
    """Raise if the zero-KL / batch-invariant config is invalid.

    ``check_environment=False`` runs only the config checks, so the driver can
    call it before Ray allocates GPUs; workers also check the installed
    packages and the GPU.
    """
    if "megatron_cfg" not in config:  # DTensor / Automodel policies
        return
    result = validate_zero_train_gen_mismatch(
        config,
        check_packages=check_environment,
        check_platform=check_environment,
    )
    result.emit_warnings()
    result.raise_if_invalid(
        "policy.megatron_cfg.zero_train_gen_mismatch=true failed validation:"
    )

    bi_result = validate_batch_invariant_mode(config)
    bi_result.emit_warnings()
    bi_result.raise_if_invalid("batch_invariant_mode=True failed validation:")


def _validate_required_values(
    config: PolicyConfig, out: ZeroTrainGenValidation
) -> None:
    """The recipe must set every value the preset needs; nothing is rewritten."""
    megatron_cfg = config["megatron_cfg"]
    for key, value in _ZERO_KL_MEGATRON_REQUIRED.items():
        if megatron_cfg.get(key) != value:
            out.violations.append(
                f"policy.megatron_cfg.{key} must be {value!r} "
                f"(got {megatron_cfg.get(key)!r})."
            )

    generation = config.get("generation")
    if generation is None or generation.get("backend") != "megatron":
        return  # reported by _validate_backend
    mcore_generation_config = generation["mcore_generation_config"]
    for key, value in _ZERO_KL_GENERATION_REQUIRED.items():
        if mcore_generation_config.get(key) != value:
            out.violations.append(
                f"policy.generation.mcore_generation_config.{key} must be "
                f"{value!r} (got {mcore_generation_config.get(key)!r})."
            )


def _validate_backend(config: PolicyConfig, out: ZeroTrainGenValidation) -> None:
    generation = config.get("generation")
    if generation is None or generation.get("backend") != "megatron":
        out.violations.append(
            "policy.generation.backend must be 'megatron' "
            f"(got {generation.get('backend') if generation else None!r})."
        )
        return

    # Colocated generation runs a dedicated inference-layout model (the inference
    # transformer_impl below differs from training's) refreshed by MCore's in-place
    # reshard on every wake.
    if (
        generation["colocated"]["enabled"]
        and generation.get("refit_transport") != "mcore"
    ):
        out.violations.append(
            "zero_train_gen_mismatch with colocated generation requires "
            "policy.generation.refit_transport='mcore' (in-place reshard into the "
            "dedicated inference model)."
        )

    inference_cfg = merged_inference_megatron_cfg(config)
    impl = inference_cfg.get("transformer_impl")
    if impl != "inference_optimized":
        out.violations.append(
            f"generation transformer_impl must be 'inference_optimized' (got {impl!r})."
        )


def _validate_platform(out: ZeroTrainGenValidation, *, check_device: bool) -> None:
    """Blackwell + FA4 only (mcore extra: flash-attn-4, CuteDSL). Hopper/FA3 not supported."""
    if not check_device:
        return

    if not torch.cuda.is_available():
        return
    major, _minor = torch.cuda.get_device_capability()

    if major == 9:
        out.violations.append(
            "zero_train_gen_mismatch does not support Hopper (SM90); use "
            "Blackwell (SM100+) with flash_attention_version=4."
        )
    elif major < 10:
        out.violations.append(
            f"zero_train_gen_mismatch requires Blackwell (SM100+); got sm_{major}x."
        )


def _validate_megatron_core_features(out: ZeroTrainGenValidation) -> None:
    try:
        # Deferred: Megatron-Core exists only in worker venvs.
        from megatron.core.transformer.transformer_config import TransformerConfig
    except ImportError:
        out.violations.append(
            "Megatron-Core is not importable (required for zero_train_gen_mismatch)."
        )
        return
    defined = {f.name for f in fields(TransformerConfig)}
    missing = [f for f in MEGATRON_CORE_REQUIRED_CONFIG_FIELDS if f not in defined]
    if missing:
        out.violations.append(
            "Installed Megatron-Core lacks TransformerConfig fields required for "
            f"zero_train_gen_mismatch: {', '.join(missing)}."
        )


def _package_version(dist_name: str) -> Version | None:
    try:
        return Version(version(dist_name))
    except PackageNotFoundError:
        return None


def _first_package_version(dist_names: tuple[str, ...]) -> Version | None:
    for dist_name in dist_names:
        found = _package_version(dist_name)
        if found is not None:
            return found
    return None


def _validate_packages(out: ZeroTrainGenValidation) -> None:
    _validate_megatron_core_features(out)

    te_ver = _package_version("transformer_engine")
    if te_ver is None:
        out.violations.append(
            "transformer_engine is not installed (required for zero_train_gen_mismatch)."
        )
    elif te_ver < TRANSFORMER_ENGINE_MIN_VERSION:
        out.violations.append(
            f"transformer_engine>={TRANSFORMER_ENGINE_MIN_VERSION} required "
            f"(got {te_ver})."
        )

    fa = _package_version("flash_attn")
    if fa is None or fa < FLASH_ATTN_MIN_VERSION:
        out.violations.append(
            f"flash_attn>={FLASH_ATTN_MIN_VERSION} required (got {fa})."
        )

    fa4 = _first_package_version(("flash-attn-4", "flash_attn_4"))
    if fa4 is None:
        out.violations.append("flash-attn-4 is not installed (mcore extra).")
    cutedsl = _package_version("nvidia-cutlass-dsl")
    if cutedsl is None or cutedsl < CUTEDSL_MIN_VERSION:
        out.violations.append(
            f"nvidia-cutlass-dsl>={CUTEDSL_MIN_VERSION} required (got {cutedsl})."
        )


def validate_zero_train_gen_model_provider(
    config: PolicyConfig, model_cfg: Any
) -> None:
    """Reject unsupported architectures using the resolved Bridge provider.

    Hybrid (Mamba) layer patterns are allowed: MCore's batch-invariant Mamba
    decode (buffered chunk replay) is used with dynamic inference; measured with
    THUNDERSYNC_MISMATCH_CHECK in the ThunderSync loop.
    """
    if not config["megatron_cfg"].get("zero_train_gen_mismatch"):
        return

    unsupported_fields = [
        field_name
        for field_name in (
            "multi_latent_attention",
            "experimental_attention_variant",
        )
        if getattr(model_cfg, field_name, None)
    ]
    if unsupported_fields:
        raise ValueError(
            "zero_train_gen_mismatch does not support the resolved model "
            f"architecture fields: {', '.join(unsupported_fields)}."
        )


def _validate_precision(config: PolicyConfig, out: ZeroTrainGenValidation) -> None:
    """Only BF16 without FP8 is supported, on both the train and generation side."""
    if config.get("precision") != "bfloat16":
        out.violations.append(
            "zero_train_gen_mismatch requires policy.precision='bfloat16' "
            f"(got {config.get('precision')!r})."
        )

    sides = [("policy.megatron_cfg", config["megatron_cfg"])]
    generation = config.get("generation")
    if generation is not None and generation.get("backend") == "megatron":
        sides.append(("generation", merged_inference_megatron_cfg(config)))
    for label, cfg in sides:
        if cfg.get("fp8_cfg") and cfg["fp8_cfg"].get("enabled"):
            out.violations.append(
                f"zero_train_gen_mismatch does not support FP8 ({label}.fp8_cfg.enabled=true)."
            )
