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

import asyncio
import concurrent.futures
from contextlib import nullcontext
import gc
import importlib
import logging
import os
import threading
import time
import warnings
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, AsyncGenerator, Mapping, Optional

import requests
import torch
from megatron.core.inference.config import (
    AsyncScheduleMode,
    CudaGraphSizingDistribution,
    InferenceConfig,
    KVCacheManagementMode,
    MambaInferenceStateConfig,
    PrefixCachingCoordinatorPolicy,
    PrefixCachingEvictionPolicy,
)
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
from megatron.core.inference.quantization.utils import (
    quantize_params_to_mxfp8,
    resolve_mxfp8_backend,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import set_decode_expert_padding
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_inference_spec,
)
from megatron.core.resharding.copy_services.gloo_copy_service import GlooCopyService
from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService
from megatron.core.resharding.copy_services.nccl_m2n_copy_service import (
    NCCLM2NCopyService,
)
from megatron.core.resharding.refit import (
    prepare_swap_model_weights,
    swap_model_weights,
)
from megatron.core.resharding.transforms import MXFP8ReshardTransform
from megatron.core.resharding.utils import get_refit_tensor_dict
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.cuda_graph_config import (
    normalize_inference_cuda_graph_scope,
)
from megatron.core.transformer.cuda_graphs import (
    CudaGraphManager,
    _CudagraphGlobalRecord,
)
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.utils import (
    set_model_config_attribute,
    toggle_cuda_graphs,
)
from megatron.core.utils import unwrap_model
from torch.distributed.distributed_c10d import (
    PrefixStore,
    ProcessGroup,
    ProcessGroupGloo,
    _world,
)

from nemo_rl.data.captured_media import MediaColumnSpec
from nemo_rl.data.multimodal_utils import CACHED_VIDEO_FRAME_MANIFEST_MAGIC
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.held_port import receive_held_socket
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.megatron.config import (
    MultimodalPromptConfigOverrides,
    resolve_refit_execution_batch_bytes,
)
from nemo_rl.models.generation.megatron.progress import (
    GenerationProgressMonitor,
    with_generated_token_counter,
)
from nemo_rl.models.generation.megatron.utils import (
    build_image_preprocessing_config,
    build_prompt_and_multimodal_data,
    build_video_preprocessing_config,
    log_gpu_memory,
    resolve_mamba_conv_states_dtype,
    resolve_torch_dtype,
    sample_vision_tensors,
)
from nemo_rl.models.megatron.memory_saver import (
    HAVE_TORCH_MEMORY_SAVER,
    WEIGHT_OFFLOAD_STATS,
    pause_engine_scratch,
    pause_inference_weights,
    resume_engine_scratch,
    resume_inference_weights,
)
from nemo_rl.models.megatron.zero_train_gen_mismatch import (
    allow_installed_flash_attn_4,
)
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_consumer
from nemo_rl.weight_sync.nccl_reshard_utils import (
    _INDIVIDUAL_EXPERT_RE,
    _STR_TO_DTYPE,
    HFToLocalParamMap,
    LocalParamSpec,
    RefitCtx,
    is_nccl_reshard_param,
    restore_refit_info_placements,
)


class Fp16HeldWeightReceive(MXFP8ReshardTransform):
    """Native MCore refit into engine weights held in fp16 (receiver-side cast).

    Under the fp16 GEMM recipe the engine holds its dense GEMM weights
    (``fp16_gemm.hold_fp16_weights``) and expert weights (``fp16_moe``) in fp16,
    while the learner's parameters, and therefore the plan's wire dtype, are bf16
    (the plan is built at ``init_collective``, before the engine converts). fp16
    and bf16 have the same width, so a direct receive into the fp16 storage would
    reinterpret the bits silently. Every tensor whose current dtype is fp16 is
    received into a bf16 buffer instead and written with ``copy_``: the same
    bf16 -> fp16 rounding the packed collective refit's ``copy_`` applies, so the
    engine ends with bitwise the same weights. Other tensors keep the direct
    receive. Receiver-only: the sender transmits bf16 with no transform.

    Subclasses ``MXFP8ReshardTransform`` only because ``NCCLM2NCopyService``
    accepts that type for receiver-side conversion; no MXFP8 state is used.
    """

    def __init__(self, core: torch.nn.Module):
        # No super().__init__: it validates MXFP8 persistent buffers.
        self.core = core
        self.convertible_params: set[str] = set()
        self.persistent_buffers: dict = {}
        self.buffer_key_prefix = ""
        self.convert_on_send = False
        self.backend = None
        self._pending_1d: dict = {}

    @property
    def tensors(self) -> dict[str, torch.Tensor]:
        # The refit executor's own name -> tensor map of the destination model
        # (cached on the module; rebuilt when MCore replaces a buffer).
        return get_refit_tensor_dict(self.core)

    def should_transform(self, param_name: str) -> bool:
        from nemo_rl.models.megatron.fp16_moe import FP16_EXPERT_ATTR

        tensor = self.tensors.get(param_name)
        # fp16 expert parameters (fp16_expert_params) are fp16 on both sides when the plan is
        # built: the wire is fp16 and the receive is direct.
        return (
            tensor is not None
            and tensor.dtype == torch.float16
            and not getattr(tensor, FP16_EXPERT_ATTR, False)
        )

    def prepare_send(self, param_name, src_slice, src_param):
        raise RuntimeError("Fp16HeldWeightReceive is a receiver-side transform")

    def prepare_recv(self, param_name, dst_slice):
        dst = self.tensors[param_name].data[dst_slice]
        return [torch.empty(dst.shape, dtype=torch.bfloat16, device=dst.device)]

    def finalize_recv(self, param_name, dst_slice, recv_buffers):
        dst = self.tensors[param_name].data[dst_slice]
        with torch.inference_mode() if dst.is_inference() else torch.no_grad():
            dst.copy_(recv_buffers[0])


class Fp16HeldLocalReshard(Fp16HeldWeightReceive):
    """``Fp16HeldWeightReceive`` for the colocated wake reshard (sender and receiver in one process).

    The colocated dedicated inference model holds its dense GEMM weights in fp16 (``_prepare_engine_numerics``)
    from its first engine initialization on, so every later wake reshards bf16 training parameters into fp16
    storage. MCore asks the transform about both sides of every op by parameter name, so the SEND of a tensor
    the receiver holds in fp16 also lands here: it sends the training parameter's bf16 slice unchanged (what the
    untransformed send does), and the receive stages bf16 and writes it with ``copy_`` (the RN cast the per-call
    cast applies). Port of ts-syncuni c7f370c2 (su-l8b: the receiver-only class raised on the second wake).
    """

    def prepare_send(self, param_name, src_slice, src_param):
        return [src_param.data[src_slice]]


def _refit_core(model: Any) -> torch.nn.Module:
    """The module MCore's refit walks (``refit._unwrap_model_cores``)."""
    return unwrap_model(model[0] if isinstance(model, (list, tuple)) else model)


def _inference_optimized_transformer_layer_spec(config: Any) -> Any:
    """Build the generic GPT layer spec backed by MCore inference linears."""
    return get_gpt_layer_with_inference_spec(
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        qk_l2_norm=config.qk_l2_norm,
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=getattr(
            config, "moe_use_legacy_grouped_gemm", False
        ),
    )


def _pin_rotary_tables(lang_module: Any) -> None:
    """Memoize ``lang_module.rotary_pos_emb.forward`` per module, without eviction.

    ``RotaryEmbedding.forward`` is an ``lru_cache(maxsize=32)`` shared by every
    instance in the process. Inference CUDA graphs bake in the address of the
    cached RoPE table, so an eviction (any other model's RoPE calls) or a
    ``cache_clear()`` frees memory the graphs keep reading, and later
    allocations overwrite it (wrong logits, eventually NaN). The per-instance
    memo keeps every table alive as long as the module; it has no
    ``cache_parameters``, so the ``cache_clear()`` sites below skip it.
    """
    import types

    rope = getattr(lang_module, "rotary_pos_emb", None)
    forward = getattr(rope, "forward", None)
    if rope is None or not hasattr(forward, "cache_parameters"):
        return
    # Megatron-LM #8358 (graph-owned capture constants): forward is no longer an lru_cache; it keeps a
    # class-level cache and hands the tables a graph captures to that graph, which keeps them alive until
    # it is destroyed. ``cache_parameters`` / ``cache_clear`` remain as compatibility attributes only, and
    # there is nothing to pin.
    compute = getattr(type(rope).forward, "__wrapped__", None)
    if compute is None:
        return
    tables: dict[tuple, Any] = {}

    def pinned_forward(self, max_seq_len, offset=0, packed_seq=False, cp_group=None):
        key = (max_seq_len, offset, packed_seq, cp_group)
        if key not in tables:
            tables[key] = compute(self, max_seq_len, offset, packed_seq, cp_group)
        return tables[key]

    rope.forward = types.MethodType(pinned_forward, rope)


def _configure_inference_optimized_layer_spec(model_provider: Any) -> None:
    """Select MCore inference linears for a Bridge generic-GPT provider.

    Only the generic path needs this. Model-specific providers (DeepSeek,
    MiniMax, GLM, ...) install ``get_gpt_decoder_block_spec``, which already
    dispatches ``transformer_impl='inference_optimized'`` itself and — unlike
    the uniform spec built here — gives dense and MoE layers *different* specs.
    Overwriting those would rebuild a ``moe_layer_freq`` model's dense layers
    as MoE layers, so leave any non-default spec alone.

    Non-GPT providers are left alone too. Hybrid/Mamba stacks (Nemotron-H via
    ``HybridModelProvider``) are not ``GPTModelProvider`` subclasses and resolve
    ``inference_optimized`` through their own stack spec, so this is a no-op for
    them rather than an error.
    """
    # Bridge imports ModelOpt plugins that can re-enter MCore while this module
    # is still initializing, so keep this cycle-sensitive provider import local.
    from megatron.bridge.models.gpt_provider import GPTModelProvider, default_layer_spec

    if not isinstance(model_provider, GPTModelProvider):
        return
    if model_provider.transformer_layer_spec is not default_layer_spec:
        return
    model_provider.transformer_layer_spec = _inference_optimized_transformer_layer_spec


def _resolve_mxfp8_refit_backend(model_config: Any) -> str:
    """Resolve the MXFP8 storage required by the grouped-GEMM backend."""
    return resolve_mxfp8_backend(model_config.inference_grouped_gemm_backend)


@dataclass
class _MegatronRefitTask:
    """A Bridge import task and its mutable inference destination."""

    conversion_task: Any
    destination: Any
    target_id: int

    @property
    def param_name(self) -> str:
        return self.conversion_task.param_name

    @property
    def cache_key(self) -> tuple[int | None, str]:
        """Stable identity across Bridge task rebuilds."""
        return (
            self.conversion_task.vp_stage,
            self.conversion_task.global_param_name,
        )

    @property
    def dependencies(self) -> tuple[str, ...]:
        return self.conversion_task.hf_param_names

    @property
    def expected_shape(self) -> torch.Size:
        return torch.Size(self.destination.shape)

    @property
    def is_mxfp8(self) -> bool:
        return isinstance(self.destination, MXFP8Tensor)

    @property
    def is_quantized(self) -> bool:
        # MCore owns detection of TE tensor and grouped-parameter storage.
        from megatron.core.fp8_utils import (
            is_float8tensor,
            is_grouped_tensor_with_quantized_storage,
        )

        return (
            self.is_mxfp8
            or is_float8tensor(self.destination)
            or is_grouped_tensor_with_quantized_storage(self.destination)
        )


@dataclass
class _MegatronBulkRefitPiece:
    """One HF-local shard and the Megatron destination region it populates."""

    task: _MegatronRefitTask
    spec: Any
    shape: torch.Size
    device: torch.device
    destination: torch.Tensor | None


def _http_sampling_defaults(generation_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Sampling defaults for HTTP requests that omit temperature, top_p or top_k.

    Training recomputes log probabilities with the configured generation sampling
    parameters. Without explicit server defaults, an omitted request field falls
    back to the server's hardcoded value or to the model's generation_config.json,
    so the request would sample off-policy.
    """
    top_p = generation_cfg["top_p"]
    top_k = generation_cfg["top_k"]
    return {
        "default_temperature": float(generation_cfg["temperature"]),
        "default_top_p": 1.0 if top_p is None else float(top_p),
        "default_top_k": 0 if top_k is None or top_k < 0 else int(top_k),
    }


def _resolve_coordinator_policy(
    mcore_generation_config,
) -> PrefixCachingCoordinatorPolicy:
    """Effective DP-coordinator routing policy for this generation config.

    Resolved in one place so the engine and the HTTP frontends cannot disagree:
    if the engine routes on prefix affinity but the frontends skip hashing, the
    coordinator silently sees no hashes and degrades to load balancing.
    """
    if not mcore_generation_config["enable_prefix_caching"]:
        return PrefixCachingCoordinatorPolicy.LOAD_BALANCED
    if "prefix_caching_coordinator_policy" not in mcore_generation_config:
        return InferenceConfig.prefix_caching_coordinator_policy
    return PrefixCachingCoordinatorPolicy(
        mcore_generation_config["prefix_caching_coordinator_policy"]
    )


def _apply_optional_inference_config_kwargs(
    inference_config_kwargs: dict[str, Any],
    mcore_generation_config: dict[str, Any],
) -> None:
    """Forward configured optional MCore inference settings with validated types."""
    enum_fields = {
        "cuda_graph_sizing_distribution": CudaGraphSizingDistribution,
        "async_sched_mode": AsyncScheduleMode,
        "prefix_caching_eviction_policy": PrefixCachingEvictionPolicy,
    }
    cast_fields = {
        "cuda_graph_max_tokens": int,
        # Prefill/mixed CUDA graphs up to max_tokens (exponential sizes) instead of
        # cuda_graph_max_tokens: removes the eager prefill-step floor (engine-side
        # verification's small piece prefills, restart prefills).
        "cuda_graph_all_prefills": bool,
        "cuda_graph_mixed_prefill_count": int,
        "vision_embedding_cache_max_bytes": int,
        "allow_stale_multimodal_embeddings": bool,
        "prefix_cache_ttl_seconds": float,
        "prefix_caching_routing_alpha": float,
        # Hybrid models: share of buffer_size_gb for Mamba states (per-request SSM /
        # conv state + batch-invariant replay buffers). The default (0.1) admits only
        # ~8 requests per rank at full Nemotron-3-Nano with a 10 GB buffer.
        "mamba_memory_ratio": float,
        "logging_step_interval": int,
        # Lookup (n-gram) speculative decoding; forwarded only when set, so
        # Megatron versions without it are unaffected.
        "speculative_method": str,
        "lookup_ngram_max": int,
        "lookup_ngram_min": int,
    }
    for key, enum_type in enum_fields.items():
        if key in mcore_generation_config:
            inference_config_kwargs[key] = enum_type(mcore_generation_config[key])
    for key, cast_type in cast_fields.items():
        if key in mcore_generation_config:
            inference_config_kwargs[key] = cast_type(mcore_generation_config[key])
    if "prefix_caching_mamba_gb" in mcore_generation_config:
        inference_config_kwargs["prefix_caching_mamba_gb"] = mcore_generation_config[
            "prefix_caching_mamba_gb"
        ]


def _apply_inference_cuda_graph_scope(
    engine_model: MegatronModule, mcore_generation_config: dict[str, Any]
) -> None:
    """Apply the optional CUDA graph scope to the engine model."""
    if "inference_cuda_graph_scope" in mcore_generation_config:
        engine_model.config.inference_cuda_graph_scope = InferenceCudaGraphScope[
            mcore_generation_config["inference_cuda_graph_scope"]
        ]


if TYPE_CHECKING:
    from nemo_rl.data_plane.interfaces import DataPlaneConfig


# How often a caller waiting on generation checks that the engine loop is still alive
# (source: rl-determinism 90e78222). A dead engine loop never resolves pending requests.
_ENGINE_LIVENESS_POLL_S = 5.0


class MegatronGenerationMixin:
    """Engine lifecycle, coordinator, HTTP server, and finish-generation machinery.

    The host class must provide:

     - model: the megatron module.
     - cfg: policy config (TypedDict).
     - rank: global rank (used for logging).
     - tokenizer: HF tokenizer.
     - megatron_tokenizer: tokenizer for inference.
     - processor: optional multimodal processor.
     - is_generation_colocated: Whether colocated or distributed.
     - _reserved_http_server_port: driver-reserved server port, or None.
    """

    # Colocated-reshard hosts assign the dedicated inference-layout model here
    # (see MegatronPolicyWorkerImpl._build_colocated_inference_model).
    processor: Optional[Any] = None
    inference_model = None
    _colocated_reshard_plan = None
    _image_preprocessing_config: Optional[Any] = None

    def _gen_model(self) -> MegatronModule:
        """The model the inference engine wraps.

        Returns the dedicated inference-layout model when one exists (colocated
        reshard), otherwise the shared training model.
        """
        return self.inference_model if self.inference_model is not None else self.model

    def _init_inference_engine_state(self) -> None:
        """Reset all inference-engine attributes to their uninitialized state."""
        self.dynamic_inference_engine = None
        self.inference_client = None
        self.inference_context = None
        self.inference_wrapped_model = None
        self.base_url = None
        self._inference_engine_initialized = False
        self._inference_engine_asleep = (
            True  # Start paused since we begin with training
        )
        self._inference_loop = None
        self._inference_thread = None
        self._token_capture_enabled = False
        self._request_payload_stager = None
        self._request_prompt_preparer = None

    def _get_megatron_inference_wrapper_cls(self) -> Optional[type]:
        """Resolve the configured Megatron inference wrapper, if any.

        Returns:
            The wrapper class, or None when no wrapper is configured.
        """
        class_path = self.cfg["generation"]["mcore_generation_config"].get(
            "megatron_inference_wrapper"
        )
        if class_path is None:
            return None
        # Resolved once per worker: this is called per sample during generation.
        cached = getattr(self, "_megatron_inference_wrapper_cls", None)
        if cached is not None:
            return cached
        module_name, _, class_name = class_path.rpartition(".")
        if not module_name:
            raise ValueError(
                "megatron_inference_wrapper must be a fully qualified class name, "
                f"got {class_path!r}."
            )
        try:
            wrapper_cls = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Could not resolve megatron_inference_wrapper {class_path!r} "
                f"(from policy.generation.mcore_generation_config): {e}"
            ) from e
        self._megatron_inference_wrapper_cls = wrapper_cls
        return wrapper_cls

    @staticmethod
    def _wrapper_supports_modality(
        inference_wrapper_cls: Optional[type], modality: str
    ) -> bool:
        """Whether the configured inference wrapper advertises `modality` support."""
        return bool(
            inference_wrapper_cls is not None
            and getattr(inference_wrapper_cls, f"supports_{modality}", False)
        )

    def _inference_model_and_media_parts(self, inference_wrapper_cls=None):
        """Return the language model and its optional multimodal parent."""
        model = unwrap_model(self._gen_model())
        if isinstance(model, (list, tuple)):
            if len(model) != 1:
                raise NotImplementedError("Virtual pipeline models are not supported.")
            model = model[0]
        if inference_wrapper_cls is None:
            inference_wrapper_cls = self._get_megatron_inference_wrapper_cls()
        if not any(
            self._wrapper_supports_modality(inference_wrapper_cls, modality)
            for modality in ("image", "video", "audio")
        ):
            return model, None
        return model.language_model, model

    def _vision_pixel_dtype(self) -> torch.dtype:
        """The dtype the vision encoder casts pixels to before encoding them.

        Read from the encoder's own parameters rather than the policy dtype:
        a config may keep the vision tower in fp32 under a bf16 language model.
        """
        _, media_model = self._inference_model_and_media_parts()
        vision_model = getattr(media_model, "vision_model", None)
        if vision_model is None:
            raise RuntimeError(
                "Megatron media capture requires the vision encoder on the "
                "coordinator's model-parallel stage"
            )
        return next(vision_model.parameters()).dtype

    def _build_image_preprocessing_config(self, generation_config: dict[str, Any]):
        """Build raw-image preprocessing settings."""
        inference_wrapper_cls = self._get_megatron_inference_wrapper_cls()
        if not self._wrapper_supports_modality(inference_wrapper_cls, "image"):
            return None
        processor = self.processor
        if processor is None:
            raise ValueError(
                "Megatron multimodal generation requires the policy processor."
            )
        # Omit absent keys entirely so MCore's own dataclass defaults apply;
        # passing None would override them (these fields are `bool`, not `Optional`).
        image_kwargs: dict[str, Any] = {}
        if "image_dynamic_resolution" in generation_config:
            image_kwargs["dynamic_resolution"] = bool(
                generation_config["image_dynamic_resolution"]
            )
        if "vision_model_type" in generation_config:
            image_kwargs["vision_model_type"] = str(
                generation_config["vision_model_type"]
            )
        if "image_dynamic_resolution_model_length" in generation_config:
            image_kwargs["dynamic_resolution_model_length"] = int(
                generation_config["image_dynamic_resolution_model_length"]
            )
        if "image_dynamic_resolution_rounding_mode" in generation_config:
            image_kwargs["dynamic_resolution_rounding_mode"] = str(
                generation_config["image_dynamic_resolution_rounding_mode"]
            )
        if "image_dynamic_resolution_resize_mode" in generation_config:
            image_kwargs["dynamic_resolution_resize_mode"] = str(
                generation_config["image_dynamic_resolution_resize_mode"]
            )
        return build_image_preprocessing_config(
            processor.image_processor,
            **image_kwargs,
        )

    def _setup_colocated_cuda_graph_managers(self) -> None:
        """Create inference CUDA-graph managers for shared-model colocated generation.

        Colocated policies build the shared training model without CUDA graphs.
        But CUDA graphs configs MUST be in-place at initialization.

        Must run at worker init: colocated reshard is detected via the pending reshard plan
        (its dedicated inference model is built later, with graphs enabled);
        the plan is consumed on the first generation cycle.
        """
        generation_cfg = self.cfg.get("generation")
        if (
            generation_cfg is None
            or generation_cfg.get("backend") != "megatron"
            or not self.is_generation_colocated
            or self._colocated_reshard_plan is not None
        ):
            return
        mcore_generation_config = generation_cfg["mcore_generation_config"]
        cuda_graph_impl = mcore_generation_config["cuda_graph_impl"]
        if cuda_graph_impl == "none":
            return
        if cuda_graph_impl != "local":
            raise ValueError(
                "Colocated Megatron generation supports only cuda_graph_impl "
                f"'none' or 'local' for inference CUDA graphs, got '{cuda_graph_impl}'. "
                "'transformer_engine' and 'full_iteration' are training-only capture modes."
            )

        lang_module = unwrap_model(self.model)
        # A model built with graphs enabled already owns managers.
        if not any(
            hasattr(module, "cudagraph_manager") for module in lang_module.modules()
        ):
            scope = normalize_inference_cuda_graph_scope(
                mcore_generation_config.get("inference_cuda_graph_scope"),
                cuda_graph_impl,
            )
            # Need the correct configs.
            set_model_config_attribute(lang_module, "cuda_graph_impl", cuda_graph_impl)
            set_model_config_attribute(lang_module, "inference_cuda_graph_scope", scope)
            set_model_config_attribute(lang_module, "cuda_graph_modules", [])

            # Need to recurse the configs' effects down into modules.
            # Megatron-LM has no API for attaching inference CUDA-graph managers to
            # an already-built model (reported upstream), so this mirrors its
            # construction-time setup.
            # TODO: switch to the upstream API once it exists.
            for module in lang_module.modules():
                if not isinstance(module, GraphableMegatronModule):
                    continue
                if hasattr(module, "create_mcore_cudagraph_manager"):
                    module.create_mcore_cudagraph_manager(module.config)
                else:
                    module.cudagraph_manager = CudaGraphManager(module.config)

            # Handle MTP as well.
            # TODO: this path only becomes testable once #3331 merges.
            if getattr(lang_module, "mtp_process", False):
                if hasattr(lang_module, "_setup_mtp_cuda_graphs") and not hasattr(
                    lang_module, "_mtp_cudagraph_manager"
                ):
                    lang_module._setup_mtp_cuda_graphs()
                assert hasattr(lang_module, "_mtp_cudagraph_manager"), (
                    f"cuda_graph_impl='{cuda_graph_impl}', but no MTP graph manager was created."
                )

            assert any(
                hasattr(module, "cudagraph_manager") for module in lang_module.modules()
            ), (
                f"cuda_graph_impl='{cuda_graph_impl}' is set for colocated Megatron "
                "generation, but no CUDA-graph manager could be created for this model."
            )

            # When the model-level manager owns block-scope graphs,
            # construction deletes the decoder's fallback manager.
            decoder = getattr(lang_module, "decoder", None)
            if (
                hasattr(lang_module, "cudagraph_manager")
                and decoder is not None
                and hasattr(decoder, "cudagraph_manager")
            ):
                del decoder.cudagraph_manager

        # Detach for training; this caches the managers built above.
        toggle_cuda_graphs(lang_module, set_to="none")

    def get_inference_cuda_graph_capture_count(self) -> int:
        """Inference CUDA graphs captured in this worker process (0 = eager decode)."""
        return len(_CudagraphGlobalRecord.cudagraph_inference_record)

    def _initialize_inference_engine(self, mcore_generation_config: dict) -> None:
        """Initialize the persistent inference engine and client."""
        # TODO: Switch to standardized Megatron API.
        if self._inference_engine_initialized:
            return

        from megatron.core.inference.config import MultimodalPromptConfig
        from megatron.core.inference.contexts.dynamic_context import (
            DynamicInferenceContext,
        )
        from megatron.core.inference.engines.dynamic_engine import (
            DynamicInferenceEngine,
        )
        from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
            GPTInferenceWrapper,
        )
        from megatron.core.inference.text_generation_controllers.text_generation_controller import (
            TextGenerationController,
        )
        from megatron.core.utils import get_attr_wrapped_model

        inference_wrapper_cls = self._get_megatron_inference_wrapper_cls()
        inference_model, media_model = self._inference_model_and_media_parts(
            inference_wrapper_cls
        )
        engine_model = media_model if media_model is not None else self._gen_model()
        pg_collection = get_attr_wrapped_model(self._gen_model(), "pg_collection")
        model_config = inference_model.config

        buffer_size_gb = mcore_generation_config["buffer_size_gb"]
        num_cuda_graphs = mcore_generation_config["num_cuda_graphs"]
        block_size_tokens = mcore_generation_config["block_size_tokens"]
        enable_chunked_prefill = mcore_generation_config["enable_chunked_prefill"]
        use_cuda_graphs_for_non_decode_steps = mcore_generation_config[
            "use_cuda_graphs_for_non_decode_steps"
        ]
        max_tokens = mcore_generation_config["max_tokens"]

        # The value may be overwritten by `recompute_kv_cache_after_weight_updates`.
        kv_cache_management_mode = mcore_generation_config["kv_cache_management_mode"]
        needs_static_kv_pointers = kv_cache_management_mode != "persist"

        materialize_only_last_token_logits = mcore_generation_config[
            "materialize_only_last_token_logits"
        ]
        num_speculative_tokens = mcore_generation_config["num_speculative_tokens"]
        max_requests = mcore_generation_config.get("max_requests")

        mamba_inference_state_config = MambaInferenceStateConfig.from_model(
            inference_model
        )
        is_hybrid_model = mamba_inference_state_config is not None
        if is_hybrid_model:
            if (
                mcore_generation_config.get("mamba_inference_ssm_states_dtype")
                is not None
            ):
                mamba_inference_state_config.ssm_states_dtype = resolve_torch_dtype(
                    mcore_generation_config["mamba_inference_ssm_states_dtype"]
                )
            conv_states_dtype = resolve_mamba_conv_states_dtype(
                mcore_generation_config, self.cfg["megatron_cfg"]
            )
            if conv_states_dtype is not None:
                mamba_inference_state_config.conv_states_dtype = conv_states_dtype

        # flashinfer's fused-RoPE kernel only dispatches fp16/bf16 q/k.
        use_flashinfer_fused_rope = model_config.params_dtype in (
            torch.float16,
            torch.bfloat16,
        )
        # Batch-invariant mode needs the policy's RoPE bits, and MCore rejects
        # an explicit True there.
        if getattr(model_config, "batch_invariant_mode", False):
            if getattr(model_config, "inference_fuse_tp_communication", False):
                # Under batch invariance the fused reduce-scatter falls back to
                # NCCL, but the next layer still skips its norm + all-gather and
                # reads the (stale) symmetric buffer.
                raise ValueError(
                    "batch_invariant_mode=True requires "
                    "inference_fuse_tp_communication=False on the generation model."
                )
            use_flashinfer_fused_rope = False
            allow_installed_flash_attn_4()
            # Without flashinfer RoPE the engine reads its RoPE table from
            # RotaryEmbedding.forward's process-wide lru_cache; CUDA graphs
            # capture that tensor's address, so it must never be evicted/freed.
            _pin_rotary_tables(unwrap_model(self._gen_model()))

        image_preprocessing_config = self._build_image_preprocessing_config(
            mcore_generation_config
        )
        self._image_preprocessing_config = image_preprocessing_config
        video_preprocessing_config = build_video_preprocessing_config(
            image_preprocessing_config,
            mcore_generation_config,
            frame_manifest_magic=CACHED_VIDEO_FRAME_MANIFEST_MAGIC,
        )
        multimodal_prompt_config = None
        prompt_config_overrides = mcore_generation_config.get(
            "multimodal_prompt_config"
        )
        if prompt_config_overrides is not None:
            if inference_wrapper_cls is None:
                raise ValueError(
                    "multimodal_prompt_config requires megatron_inference_wrapper."
                )
            wrapper_prompt_defaults = getattr(
                inference_wrapper_cls, "multimodal_prompt_config", None
            )
            if wrapper_prompt_defaults is None:
                raise ValueError(
                    f"{inference_wrapper_cls.__name__} does not define a multimodal "
                    "prompt contract to override."
                )
            prompt_config_overrides = MultimodalPromptConfigOverrides.model_validate(
                prompt_config_overrides
            )
            multimodal_prompt_config = MultimodalPromptConfig.from_dict(
                prompt_config_overrides.model_dump(exclude_unset=True),
                defaults=wrapper_prompt_defaults,
            )

        inference_config_kwargs: dict[str, Any] = {
            "block_size_tokens": block_size_tokens,
            "buffer_size_gb": buffer_size_gb,
            "num_cuda_graphs": num_cuda_graphs,
            "max_tokens": max_tokens,
            "max_sequence_length": mcore_generation_config["max_model_len"],
            "kv_cache_management_mode": KVCacheManagementMode(kv_cache_management_mode),
            "static_kv_memory_pointers": needs_static_kv_pointers,
            "use_cuda_graphs_for_non_decode_steps": use_cuda_graphs_for_non_decode_steps,
            "use_flashinfer_fused_rope": use_flashinfer_fused_rope,
            "sampling_backend": "flashinfer",
            "use_synchronous_zmq_collectives": True,
            "materialize_only_last_token_logits": materialize_only_last_token_logits,
            "enable_chunked_prefill": enable_chunked_prefill,
            "enable_prefix_caching": mcore_generation_config["enable_prefix_caching"],
            "prefix_caching_coordinator_policy": _resolve_coordinator_policy(
                mcore_generation_config
            ),
            "pg_collection": pg_collection,
            "mamba_inference_state_config": mamba_inference_state_config,
            # Reserve more KV-cache space when speculative decoding is enabled.
            # With max_requests set, MCore sizes the Mamba region for exactly that
            # many requests and gives the rest to KV (the fixed ratio would cap
            # the request count at ratio * buffer / state bytes instead).
            "mamba_memory_ratio": (
                0.1 + 0.1 * num_speculative_tokens
                if is_hybrid_model and max_requests is None
                else None
            ),
            "num_speculative_tokens": num_speculative_tokens,
            "logprobs_mode": mcore_generation_config["logprobs_mode"],
            "max_requests": max_requests,
            "image_preprocessing_config": image_preprocessing_config,
            "video_preprocessing_config": video_preprocessing_config,
            "multimodal_prompt_config": multimodal_prompt_config,
        }
        _apply_optional_inference_config_kwargs(
            inference_config_kwargs, mcore_generation_config
        )

        inference_config = InferenceConfig(**inference_config_kwargs)

        _apply_inference_cuda_graph_scope(engine_model, mcore_generation_config)

        self._prepare_engine_numerics(engine_model)
        if kv_cache_management_mode == "offload" and HAVE_TORCH_MEMORY_SAVER:
            # Suspend / resume copy the resident KV blocks and Mamba slots, not the whole buffer.
            from nemo_rl.models.megatron import kv_live_offload

            kv_live_offload.install()
        self.inference_context = DynamicInferenceContext(
            engine_model.config, inference_config
        )
        # fp16_moe_dk's expert row workspace at the gathered buffer's worst case, before the graphs capture it
        # (engine scratch: released while a colocated learner trains, with torch_memory_saver).
        from nemo_rl.models.megatron import fp16_moe_dk

        ws_bytes, ws_chunk = fp16_moe_dk.reserve_engine_row_workspace(
            engine_model, self.inference_context, colocated=bool(self.is_generation_colocated)
        )
        if ws_bytes and self.rank == 0:
            print(
                f"[Rank 0] engine expert row workspace {ws_bytes / 2**30:.2f} GiB, fc2 {ws_chunk} columns / pass"
                f" (released during training: {HAVE_TORCH_MEMORY_SAVER})"
            )
        if media_model is None:
            self.inference_wrapped_model = GPTInferenceWrapper(
                engine_model, self.inference_context
            )
        else:
            if inference_wrapper_cls is None:
                raise ValueError(
                    "Multimodal inference requires megatron_inference_wrapper."
                )
            self.inference_wrapped_model = inference_wrapper_cls(
                engine_model, self.inference_context
            )
        text_generation_controller = TextGenerationController(
            inference_wrapped_model=self.inference_wrapped_model,
            tokenizer=self.megatron_tokenizer,
        )
        self.dynamic_inference_engine = with_generated_token_counter(
            DynamicInferenceEngine
        )(text_generation_controller, self.inference_context)
        self._start_generation_progress_monitor(mcore_generation_config)

        self._inference_engine_initialized = True
        self._inference_engine_asleep = True
        print(f"[Rank {self.rank}] Initialized persistent inference engine")

    def _prepare_engine_numerics(self, engine_model: Any) -> None:
        """Engine-only numerics setup, before the context is sized and graphs captured.

        - fp16 GEMM forward: hold the dense GEMM weights in fp16 (``hold_fp16_weights``)
          on a dedicated inference model whose refits write through ``copy_``: the
          packed collective (``refit_transport: null``) and the native MCore refit
          (``refit_transport: mcore``), which receives every fp16-held tensor through
          ``Fp16HeldWeightReceive`` (bf16 on the wire, ``copy_`` into the fp16 storage).
          The nccl_reshard M-to-N refit receives raw bytes in the sender's dtype, so it
          keeps the per-call cast.
        - batch-invariant Mamba: decode without the z replay buffer through the batch-invariant
          decode step (``mamba_bi_fused_decode``) and count the replay buffers in the inference
          memory budget (``mamba_bi_decode``).
        """
        from nemo_rl.models.megatron import fp16_gemm

        refit_transport = self.cfg["generation"].get("refit_transport")
        # Colocated generation with a dedicated inference model (its transformer_impl or layout differs from
        # training's, e.g. the BI engine's inference_optimized): hold those weights in fp16 too. Its wake reshard
        # writes through Fp16HeldLocalReshard (bf16 slice, copy_ = the RN cast). Without the hold every decode step
        # re-casts every dense GEMM weight inside the captured graphs (ts-syncuni: 5.68 vs 1.34 ms per decode step at
        # full Nano EP4). A reshardless colocated engine shares the training model and keeps the per-call cast.
        dedicated_colocated = self.is_generation_colocated and self.inference_model is not None
        if (
            fp16_gemm.enabled()
            and (
                (self.is_refit_destination and not self.is_generation_colocated)
                or dedicated_colocated
            )
            and refit_transport in (None, "mcore")
        ):
            n = fp16_gemm.hold_fp16_weights(self._gen_model())
            if self.rank == 0:
                print(
                    f"[fp16_gemm] engine holds {n} dense GEMM weights in fp16",
                    flush=True,
                )
        if getattr(engine_model.config, "batch_invariant_mode", False):
            from nemo_rl.models.megatron import mamba_bi_decode

            mamba_bi_decode.install()
            # Tree speculation (thundersync.tail_tree, tree_spec): the node-row state (tree_kv: scratch conv and
            # replay rows in the Mamba layers, the private-block KV gather), before the context is sized and the
            # graphs captured.
            tree = os.environ.get("THUNDERSYNC_TAIL_TREE", "").strip()
            if tree:
                from nemo_rl.models.generation.megatron import tree_kv

                tree_kv.install(int(tree.split(":")[2]))
                # MTP draft source (thundersync.tail_tree_mtp): the decoder output of every engine forward, captured
                # into a fixed buffer by a hook on the final norm (inside the graphs once they are captured).
                if os.environ.get("THUNDERSYNC_TAIL_TREE_MTP", "").strip():
                    from nemo_rl.models.generation.megatron import tree_mtp

                    gcfg = self.cfg["generation"]["mcore_generation_config"]
                    tree_mtp.install_hidden_capture(
                        engine_model, int(gcfg["max_tokens"]), int(engine_model.config.hidden_size)
                    )

    def _start_generation_progress_monitor(self, mcore_generation_config: dict) -> None:
        """Progress log + stall watchdog over this rank's engine counters (progress.py)."""
        interval = mcore_generation_config.get("progress_log_interval_s")
        if interval is None:
            return
        rank = torch.distributed.get_rank()
        stall = mcore_generation_config.get("stall_timeout_s")
        self._generation_progress = GenerationProgressMonitor(
            self.dynamic_inference_engine,
            rank,
            interval_s=float(interval),
            stall_timeout_s=float(stall) if stall is not None else 0.0,
            log=rank == 0
            or bool(mcore_generation_config.get("progress_log_all_ranks")),
        ).start()

    async def _start_inference_coordinator(self):
        """Start the inference coordinator and engine loop."""
        self.coordinator_addr = await self.dynamic_inference_engine.start_listening_to_data_parallel_coordinator(
            inference_coordinator_port=None,
            launch_inference_coordinator=True,
        )
        if torch.distributed.get_rank() == 0:
            from megatron.core.inference.inference_client import InferenceClient

            self.inference_client = InferenceClient(
                inference_coordinator_address=self.coordinator_addr, deserialize=True
            )
            result = self.inference_client.start()
            if result is not None:
                await result
            if self._own_client_loop_wanted():
                await self._move_client_to_own_loop()

        self._inference_engine_asleep = False

    def _own_client_loop_wanted(self) -> bool:
        """Run rank 0's inference client on its own event-loop thread (subclasses opt in; default: no)."""
        return False

    def _client_loop(self):
        """The event loop that owns the inference client's socket (its own loop, else the engine loop)."""
        return getattr(self, "_inference_client_loop", None) or self._inference_loop

    async def _move_client_to_own_loop(self) -> None:
        """Move the client's listener (and with it every client call) off the engine loop.

        The engine loop then runs only the engine. The client keeps its socket and request-id counter (no new
        connection), so requests and their ids are unchanged. ZMQ sockets are not thread safe: every send and the
        listener run on the client loop (``_client_loop``; ``_client_send`` for the engine-control signals).
        """
        local_rank = int(os.environ["LOCAL_RANK"])
        box: dict = {}

        def run_loop():
            torch.cuda.set_device(local_rank)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            box["loop"] = loop
            loop.run_forever()

        self._inference_client_thread = threading.Thread(target=run_loop, name="inference-client", daemon=True)
        self._inference_client_thread.start()
        while "loop" not in box:
            await asyncio.sleep(0.001)
        client = self.inference_client
        client.listener_task.cancel()
        try:
            await client.listener_task
        except asyncio.CancelledError:
            pass

        async def listen():
            client._loop = asyncio.get_running_loop()
            client.listener_task = client._loop.create_task(client._recv_task())

        await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(listen(), box["loop"]))
        self._inference_client_loop = box["loop"]

    async def _client_send(self, fn) -> None:
        """Call ``fn()`` (a client send) on the client loop, from a coroutine on the engine loop."""
        loop = self._client_loop()
        if loop is self._inference_loop:
            fn()
            return

        async def call():
            fn()

        await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(call(), loop))

    def _sleep(self) -> None:
        """Pause + suspend the engine. No-op if already asleep."""
        if self._inference_engine_asleep:
            return
        future = asyncio.run_coroutine_threadsafe(
            self._sleep_engine(), self._inference_loop
        )
        future.result()
        from nemo_rl.models.megatron import kv_live_offload

        kv_live_offload.raise_if_check_failed()
        pause_engine_scratch()
        torch.distributed.barrier()
        self._inference_engine_asleep = True
        print(f"[Rank {self.rank}] paused inference engine")

        st = kv_live_offload.last_stats()
        if self.rank == 0 and st:
            print(
                f"[colo kv] suspend saved {st['saved_bytes'] / 2**20:.1f} MiB: {st['live_blocks']} KV blocks, "
                f"{st['live_slots']} Mamba slots"
                + (
                    f" | last resume: {st['checked_blocks']} blocks / {st['checked_slots']} slots unchanged"
                    if "checked_blocks" in st
                    else ""
                ),
                flush=True,
            )

    async def _sleep_engine(self):
        if torch.distributed.get_rank() == 0:
            await self._client_send(self.inference_client.pause_engines)
        await self.dynamic_inference_engine.wait_until(EngineState.PAUSED)

        if torch.distributed.get_rank() == 0:
            await self._client_send(self.inference_client.suspend_engines)
        await self.dynamic_inference_engine.wait_until(EngineState.SUSPENDED)

    def _wake(self) -> None:
        """Resume + unpause the engine. No-op if already awake."""
        if not self._inference_engine_asleep:
            return
        resume_engine_scratch()
        # Every rank is ready before any engine resumes. After the resume an engine steps its resumed requests at once
        # (EP collectives on its GPU); a default-group barrier issued then can precede those collectives on one rank
        # and follow them on another, and NCCL collectives issued in different orders on different ranks deadlock
        # (Nano 8k SPR-BI: 8 ranks in this barrier, one engine in its first step's collective).
        torch.distributed.barrier()
        future = asyncio.run_coroutine_threadsafe(
            self._wake_engine(), self._inference_loop
        )
        future.result()
        from nemo_rl.models.megatron import kv_live_offload

        kv_live_offload.raise_if_check_failed()
        self._inference_engine_asleep = False
        print(f"[Rank {self.rank}] resumed inference engine")

    async def _wake_engine(self):
        if torch.distributed.get_rank() == 0:
            await self._client_send(self.inference_client.resume_engines)
        await self.dynamic_inference_engine.wait_until(EngineState.RESUMED)

        if torch.distributed.get_rank() == 0:
            await self._client_send(self.inference_client.unpause_engines)
        await self.dynamic_inference_engine.wait_until(EngineState.RUNNING)

    def _start_inference_loop_thread(self):
        """Start a background thread with a persistent event loop for inference."""
        # CUDA current_device is per-thread.
        # The worker's __init__ thread called set_device(LOCAL_RANK), and this thread must match.
        local_rank = int(os.environ["LOCAL_RANK"])

        def run_loop():
            torch.cuda.set_device(local_rank)
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
            self._inference_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._inference_loop)
            self._inference_loop.run_forever()

        self._inference_thread = threading.Thread(target=run_loop, daemon=True)
        self._inference_thread.start()
        while self._inference_loop is None:
            time.sleep(0.001)

    def _setup_openai_api_server(self) -> str:
        """Start the OpenAI-compatible HTTP server on this worker."""
        import random

        from megatron.core.inference.text_generation_server.dynamic_text_gen_server.text_generation_server import (
            start_text_gen_server,
        )

        from nemo_rl.distributed.virtual_cluster import (
            _get_free_port_local,
            _get_node_ip_local,
        )

        gen_cfg = self.cfg["generation"]["mcore_generation_config"]
        coordinator_policy = _resolve_coordinator_policy(gen_cfg)

        ip = _get_node_ip_local()
        # Setup Megatron inference front-end ports for service.
        # If ports are reserved (such as for Gym), use those.
        reserved_port = self._reserved_http_server_port
        if reserved_port is not None:
            # Defer socket handoff until immediately before server startup.
            # Holding this listener across model initialization can fork
            # into a persistent child process, which then receives Gym
            # traffic despite never accepting HTTP requests.
            reserved_socket = receive_held_socket(reserved_port)
            server_port = reserved_socket.getsockname()[1]
        else:
            # Seed port assignment by rank to ensure that inference servers
            # do not listen to the same port and parallelize tokenization,
            # preprocessing, and hashing on multiple CPUs.
            reserved_socket = None
            server_port = _get_free_port_local(
                rng=random.Random(torch.distributed.get_rank())
            )

        server_kwargs: dict[str, Any] = _http_sampling_defaults(self.cfg["generation"])
        if "http_server_num_replicas" in gen_cfg:
            server_kwargs["num_replicas"] = int(gen_cfg["http_server_num_replicas"])

        start_text_gen_server(
            coordinator_addr=self.coordinator_addr,
            tokenizer=self.megatron_tokenizer,
            rank=torch.distributed.get_rank(),
            server_port=server_port,
            parsers=gen_cfg["parsers"],
            verbose=False,
            sock=reserved_socket,
            multimodal_prompt_config=self.inference_wrapped_model.multimodal_prompt_config,
            # The frontend hashes prompts so the coordinator does not have to.
            # Whether it bothers follows from the routing policy, so pass that
            # rather than encoding the decision here; the block size is only the
            # granularity and must match the engine's.
            block_size_tokens=gen_cfg["block_size_tokens"],
            prefix_caching_coordinator_policy=coordinator_policy,
            **server_kwargs,
        )

        base_url = f"http://{ip}:{server_port}/v1"
        max_wait_time = 300
        start_time = time.time()
        with requests.Session() as session:
            while True:
                if time.time() - start_time > max_wait_time:
                    raise TimeoutError(
                        f"[Megatron HTTP] Rank {self.rank} OpenAI server failed "
                        f"to start within {max_wait_time}s"
                    )
                try:
                    response = session.get(f"{base_url}/health", timeout=10)
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
        return base_url

    def _run_async_coordinator_start(self):
        """Start the coordinator and engine loop in the background thread."""
        if self._inference_loop is None:
            self._start_inference_loop_thread()

        future = asyncio.run_coroutine_threadsafe(
            self._start_inference_coordinator(), self._inference_loop
        )
        # _start_inference_coordinator awaits RUNNING, so future.result() only returns once
        # this rank's engine is fully warmed up. Cross-rank sync is handled by Ray's actor
        # group semantics (the caller waits for all workers' prepare_for_generation).
        future.result()
        print(f"[Rank {torch.distributed.get_rank()}] Coordinator started")

        if (
            self.cfg["generation"]["mcore_generation_config"]["expose_http_server"]
            and self.dynamic_inference_engine.is_mp_coordinator
        ):
            print(f"[Rank {torch.distributed.get_rank()}] Starting HTTP Server")
            self.base_url = self._setup_openai_api_server()
        else:
            print(f"[Rank {torch.distributed.get_rank()}] HTTP Server not started")
            self.base_url = None

    def finish_generation(self, *, release_gpu: bool = True) -> None:
        """Wind down a generation cycle.

        Args:
            release_gpu: the caller needs the GPUs for itself (a training
                step or a checkpoint save), so even a colocated engine must
                fully stand down. Pass False between generation phases (e.g.
                after validation) to let a colocated engine keep serving;
                tearing it down there would discard KV/prefix caches and CUDA
                graphs for no reason.
                For non-colocated engines this body reduces to a rotary-cache clear
                (their engine pause happens in suspend_for_refit).
        """
        if self.is_generation_colocated and not release_gpu:
            return
        print(f"[Rank {self.rank}] finishing generation", flush=True)

        # Report how many steps the async scheduler processed. This includes
        # steps that ran a non-overlapped async ordering because they could
        # not overlap -- the legacy path never increments this counter.
        if (
            self.cfg["generation"]["mcore_generation_config"].get("async_sched_mode")
            == "async"
            and self._inference_engine_initialized
        ):
            print(
                f"[Rank {self.rank}] mcore async scheduling steps (cumul): "
                f"{self.inference_context.async_sched_step_count}",
                flush=True,
            )

        log_gpu_memory("finish_generation START")

        inference_model, media_model = self._inference_model_and_media_parts()
        lang_module = unwrap_model(inference_model)
        graph_module = unwrap_model(
            media_model if media_model is not None else inference_model
        )

        if self.is_generation_colocated:
            if self._inference_engine_initialized and not self._inference_engine_asleep:
                self._sleep()
            cuda_graph_impl = self.cfg["generation"]["mcore_generation_config"][
                "cuda_graph_impl"
            ]
            if cuda_graph_impl != "none":
                # Restore the same full model tree cached during worker setup.
                toggle_cuda_graphs(graph_module, set_to="none")
                # Need to turn off padding before training.
                # Gains nightly MoE coverage once #2884 and #3570 merge.
                set_decode_expert_padding(lang_module, set_to=False)

        rotary_module = getattr(lang_module, "rotary_pos_emb", None)
        if rotary_module is not None and hasattr(
            rotary_module.forward, "cache_parameters"
        ):
            rotary_module.forward.cache_clear()

        if self.is_generation_colocated:
            # Offload the inference weights to CPU.
            if self.inference_model is not None:
                self._offload_inference_model()
            gc.collect()
            torch.cuda.empty_cache()

        log_gpu_memory("finish_generation END")

    def prepare_for_generation(self, tags=None, **kwargs) -> None:
        """Enter inference mode and start (or wake) the inference engine.

        Idempotent wake: a plain call (no tags) on an already-serving engine returns immediately.
        Refit-protocol calls (tags) are never skipped.

        Called in both colocated and non-colocated setups.
        Even in non-colocated mode, Megatron's engine has to be intentionally paused before a refit
        (and its weights are not detachable), so we have to switch modes around every refit.
        """
        if (
            tags is None
            and self._inference_engine_initialized
            and not self._inference_engine_asleep
        ):
            print(
                f"[Rank {self.rank}] prepare_for_generation: engine already "
                "serving, skipping",
                flush=True,
            )
            return
        log_gpu_memory("prepare_for_generation START")
        mcore_generation_config = self.cfg["generation"]["mcore_generation_config"]

        # Colocated reshard: build the dedicated inference-layout model on the first cycle.
        if self._colocated_reshard_plan is not None:
            self._build_colocated_inference_model(self.cfg)

        if self.is_generation_colocated and self.inference_model is None:
            self.model = self.move_model(
                self.model, "cuda", move_params=True, move_grads=False
            )
            # Because DP inference requests are asynchronously scheduled per rank, pre-forward hooks that trigger DP collectives (such as an overlapped param gather after optimizer steps) will stall or hang.
            # Instead, synchronously gather all model compute weights from the sharded model state here, and deactivate all pre-forward hooks.
            # Incompatible with FSDP2 or Megatron-FSDP for inference.
            if (
                self.should_disable_forward_pre_hook
                and self._forward_pre_hook_enabled()
            ):
                self._disable_forward_pre_hook_until_next_train_step(param_sync=True)

        # Colocated reshard (hosts without a dedicated inference model skip it).
        if self.inference_model is not None:
            self._reshard_into_inference_model()

        inference_model, media_model = self._inference_model_and_media_parts()
        # `flash_decode` selects Megatron Inference's deprecated static-batching decode path,
        # which would cause an assertion error if taken.
        inference_model.config.flash_decode = False
        lang_module = unwrap_model(inference_model)
        graph_module = unwrap_model(
            media_model if media_model is not None else inference_model
        )
        lang_module.eval()

        rotary_module = getattr(lang_module, "rotary_pos_emb", None)
        if rotary_module is not None and hasattr(
            rotary_module.forward, "cache_parameters"
        ):
            rotary_module.forward.cache_clear()

        cuda_graph_impl = mcore_generation_config["cuda_graph_impl"]
        if cuda_graph_impl != "none":
            # Use the same root object as _setup_colocated_cuda_graph_managers;
            # CUDA-graph manager caches are keyed by model identity.
            toggle_cuda_graphs(graph_module, set_to=cuda_graph_impl)

        # tags=["weights"] means we are inside refit_policy_generation between
        # suspend_for_refit and the weight transfer — the engine was intentionally
        # paused and waking it now would race NVSHMEM init / weight transfer against
        # CUDA-graph replay, corrupting TE FP8 state. The subsequent
        # prepare_for_generation(tags=["kv_cache"]) is what actually wakes it.
        if tags is None or "weights" not in tags:
            if not self._inference_engine_initialized:
                self._initialize_inference_engine(mcore_generation_config)
                self._run_async_coordinator_start()
            else:
                self._wake()

        log_gpu_memory("prepare_for_generation END")

    def report_dp_openai_server_base_url(self) -> Optional[str]:
        """Return this worker's OpenAI server base URL (None if not the leader)."""
        return self.base_url

    def setup_token_capture(
        self,
        dp_cfg: "DataPlaneConfig",
        staging_partition: str,
        *,
        capture_media: bool = False,
    ) -> MediaColumnSpec | None:
        """Install canonical TQ capture on each MInf model-parallel leader.

        ``capture_media`` builds the sink/source against the media-enabled
        staging schema so the stager can hand the engine's media tensors to
        TQ beside each call's tokens. Returns the media column spec the
        coordinator pinned the staging column to (``None`` for text-only
        capture and on followers, which host no hooks).
        """
        engine = self.dynamic_inference_engine
        if engine is None:
            raise RuntimeError(
                "Megatron token capture requires an initialized inference engine"
            )
        if capture_media and self._image_preprocessing_config is None:
            raise ValueError(
                "Megatron media capture requires an image-capable inference wrapper "
                "(mcore_generation_config.megatron_inference_wrapper)"
            )
        missing = [
            name
            for name in ("payload_stager", "prompt_preparer")
            if not hasattr(engine, name)
        ]
        if missing:
            raise RuntimeError(
                "Megatron token capture requires MInf RequestPayloadStager, request "
                f"metadata, and prompt preparation support; missing {', '.join(missing)}"
            )
        self._token_capture_enabled = True
        if not engine.is_mp_coordinator:
            return None

        from nemo_rl.data_plane import build_data_plane_client
        from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
        from nemo_rl.models.generation.megatron.token_capture import (
            TQMegatronPromptPreparer,
            TQMegatronTokenStager,
        )

        # Own no segment, as vLLM capture does, so a save never waits on a
        # generation-only worker. A trainer in this process attached first and
        # keeps its segment: TQ reuses a process's existing client.
        dp_client = build_data_plane_client(dp_cfg, bootstrap=False, segment_size=0)
        # MInf emits float32 pixels and the trainer casts them to the vision
        # encoder's weight dtype before encoding, so the stager casts first and
        # the media column is pinned to that dtype (the rule vLLM follows).
        pixel_dtype = self._vision_pixel_dtype() if capture_media else None
        prompt_preparer = TQMegatronPromptPreparer(
            TQTokenSource(
                dp_client,
                staging_partition=staging_partition,
                capture_media=capture_media,
            )
        )
        engine.prompt_preparer = prompt_preparer
        self._request_prompt_preparer = prompt_preparer
        stager = TQMegatronTokenStager(
            TQTokenSink(
                dp_client,
                staging_partition=staging_partition,
                capture_media=capture_media,
                media_pixel_dtype=pixel_dtype,
            )
        )
        engine.payload_stager = stager
        self._request_payload_stager = stager
        if not capture_media:
            return None
        assert pixel_dtype is not None
        return MediaColumnSpec(
            pixel_dtype=pixel_dtype,
            patch_size=int(self._image_preprocessing_config.patch_dim),
        )

    def set_rollout_weight_version(self, version: int) -> None:
        """Stamp subsequent MInf requests with the trainer weight version."""
        if type(version) is not int or version < 0:
            raise ValueError(
                f"rollout weight version must be a non-negative int, got {version!r}"
            )
        if not self._token_capture_enabled:
            raise RuntimeError("Megatron token capture is not initialized")
        if torch.distributed.get_rank() != 0:
            return
        if self.inference_client is None:
            raise RuntimeError("Megatron token capture is not initialized")

        # The client's socket is also read by its listener task on the inference
        # loop thread, and ZMQ sockets are not thread safe: send from that loop.
        async def _send_epoch() -> None:
            self.inference_client.set_generation_epoch(version)

        asyncio.run_coroutine_threadsafe(_send_epoch(), self._client_loop()).result()

    def _build_sampling_params(
        self,
        greedy: bool,
        stop_words: Optional[list[str]],
        *,
        return_prompt_tokens: bool = False,
    ) -> SamplingParams:
        """Build mcore SamplingParams for a single request."""
        top_k_cfg = self.cfg["generation"]["top_k"]
        top_k_val = 1 if greedy else (int(top_k_cfg) if top_k_cfg is not None else 0)

        top_p_cfg = self.cfg["generation"]["top_p"]
        top_p_val = (
            0.0 if greedy else (float(top_p_cfg) if top_p_cfg is not None else 0.0)
        )

        return SamplingParams(
            temperature=self.cfg["generation"]["temperature"] if not greedy else 0,
            top_k=top_k_val,
            top_p=top_p_val,
            skip_prompt_log_probs=True,
            return_log_probs=True,
            num_tokens_to_generate=self.cfg["generation"]["max_new_tokens"],
            termination_id=(
                -1
                if self.cfg["generation"].get("ignore_eos", False)
                else self.megatron_tokenizer.eod
            ),
            stop_words=stop_words,
            return_prompt_tokens=return_prompt_tokens,
            detokenize_stop_sequence=True,
        )

    def _merge_stop_strings(
        self, batch_stop_strings: Optional[list[Optional[list[str]]]]
    ) -> Optional[list[str]]:
        """Union the config's stop_strings with the given per-sample stop strings."""
        stop_set: set[str] = set()
        if self.cfg["generation"]["stop_strings"]:
            stop_set.update(self.cfg["generation"]["stop_strings"])
        if batch_stop_strings is not None:
            for sample_ss in batch_stop_strings:
                if sample_ss:
                    stop_set.update(sample_ss)
        # Sorted: set order depends on PYTHONHASHSEED, so a plain list differs between processes.
        return sorted(stop_set) if stop_set else None

    def _sample_vision_tensors(self, data, index: int):
        """Return one sample's vision tensors from RL PackedTensors."""
        return sample_vision_tensors(data, index)

    def _build_prompt_and_multimodal_data(self, data, index: int):
        """Build one pre-expanded token prompt and optional MCore media payload."""
        return build_prompt_and_multimodal_data(
            data,
            index,
            sample_tensors=self._sample_vision_tensors,
            supports_modality=lambda modality: self._wrapper_supports_modality(
                self._get_megatron_inference_wrapper_cls(),
                modality,
            ),
        )

    def _prepare_data_for_generation(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> tuple[list[list[int]], list[Optional[Any]], list[SamplingParams]]:
        """Build prompts, optional media payloads, and sampling parameters."""
        if data is not None:
            assert isinstance(data, BatchedDataDict), (
                f"data must be a BatchedDataDict, got type: {type(data)}"
            )
            is_right_padded, error_msg = verify_right_padding(
                data, pad_value=self.tokenizer.pad_token_id
            )
            if not is_right_padded:
                warnings.warn(
                    f"Input to Megatron Generation worker is not properly right-padded: {error_msg}"
                )

        batch_stop_strings = data.get("stop_strings", [])
        prompts: list[list[int]] = []
        multi_modal_data_list: list[Optional[Any]] = []
        sampling_params = []
        for i in range(data.size):
            prompt, multi_modal_data = self._build_prompt_and_multimodal_data(data, i)
            sample_stop_strings = (
                batch_stop_strings[i] if i < len(batch_stop_strings) else None
            )
            stop_words = self._merge_stop_strings(
                [sample_stop_strings] if sample_stop_strings else None
            )
            prompts.append(prompt)
            multi_modal_data_list.append(multi_modal_data)
            sampling_params.append(
                self._build_sampling_params(
                    greedy,
                    stop_words,
                    return_prompt_tokens=multi_modal_data is not None,
                )
            )

        return prompts, multi_modal_data_list, sampling_params

    def _parse_result_to_batched_data_dict(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        result: list,
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Pack DynamicInferenceRequest results into a GenerationOutputSpec batch."""
        input_lengths = data["input_lengths"]
        input_ids = data["input_ids"]
        batch_size = input_ids.size(0)
        max_gen_seq_len = max(len(x.generated_tokens) for x in result)
        padded_input_length = input_ids.size(1)

        expected_prompt_lengths = [int(length) for length in input_lengths.tolist()]
        inference_prompt_lengths = [
            len(x.prompt_tokens)
            if getattr(x, "prompt_tokens", None) is not None
            else expected_prompt_lengths[i]
            for i, x in enumerate(result)
        ]
        if any(getattr(x, "prompt_tokens", None) is not None for x in result):
            if inference_prompt_lengths != expected_prompt_lengths:
                raise RuntimeError(
                    "Megatron inference prompt lengths do not match the training "
                    "processor input lengths: "
                    f"inference={inference_prompt_lengths}, "
                    f"training={expected_prompt_lengths}."
                )

        max_seq_len = padded_input_length + max_gen_seq_len
        output_ids_padded = torch.full(
            (batch_size, max_seq_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )

        logprobs_padded = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.float,
            device=input_ids.device,
        )

        generation_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        unpadded_sequence_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        for i in range(batch_size):
            # Take the prompt from the request we submitted rather than from the
            # engine's reply. Multimodal requests do set
            # SamplingParams.return_prompt_tokens, but only so the echoed tokens
            # can be length-checked below; the padded output is still built from
            # the prompt we already hold rather than shipped back over ZMQ.
            prompt_len = input_lengths[i].item()
            generated_tokens = result[i].generated_tokens
            seq_len = prompt_len + len(generated_tokens)
            output_ids_padded[i, :prompt_len] = input_ids[i, :prompt_len]
            output_ids_padded[i, prompt_len:seq_len] = torch.tensor(
                generated_tokens, dtype=torch.long, device=input_ids.device
            )
            generation_lengths[i] = len(generated_tokens)
            unpadded_sequence_lengths[i] = seq_len
            gen_logprobs = result[i].generated_log_probs
            logprobs_padded[i, prompt_len : prompt_len + len(gen_logprobs)] = (
                torch.tensor(
                    gen_logprobs,
                    dtype=torch.float,
                    device=input_ids.device,
                )
            )

        out_dict = {
            "output_ids": output_ids_padded,
            "logprobs": logprobs_padded,
            "generation_lengths": generation_lengths,
            "unpadded_sequence_lengths": unpadded_sequence_lengths,
        }

        return BatchedDataDict.from_batches([out_dict]).to("cpu")

    @wrap_with_nvtx_name("megatron_policy_worker/generate")
    def generate(
        self, *, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Synchronous batched generation via the mcore data-parallel coordinator.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors
            greedy: Whether to use greedy decoding instead of sampling

        Returns:
            BatchedDataDict conforming to GenerationOutputSpec:
                - output_ids: input + generated token IDs with proper padding
                - logprobs: Log probabilities for tokens
                - generation_lengths: Lengths of each response
                - unpadded_sequence_lengths: Lengths of each input + generated sequence
        """
        prompts, multi_modal_data_list, sampling_params = (
            self._prepare_data_for_generation(data, greedy)
        )
        if self._inference_loop is None:
            raise RuntimeError(
                "Inference loop not initialized. Call prepare_for_generation() first."
            )
        future = asyncio.run_coroutine_threadsafe(
            self._generate_with_persistent_engine(
                prompts,
                multi_modal_data_list,
                sampling_params,
            ),
            self._client_loop(),
        )
        while True:
            try:
                result = future.result(timeout=_ENGINE_LIVENESS_POLL_S)
            except concurrent.futures.TimeoutError:
                self._raise_if_inference_engine_dead(future)
            else:
                break

        return self._parse_result_to_batched_data_dict(data, result)

    def _raise_if_inference_engine_dead(self, future: concurrent.futures.Future) -> None:
        """Fail a pending request instead of waiting forever on a dead engine loop.

        The MCore engine loop logs a fatal step error (e.g. a CUDA fault during graph replay) and exits without
        resolving in-flight request futures, so a caller blocked on them would hang. Checked: the engine loop thread,
        rank 0's inference-client thread (when it has its own loop) and the engine's loop task.
        """
        reason = None
        thread = getattr(self, "_inference_thread", None)
        if thread is not None and not thread.is_alive():
            reason = "the inference event-loop thread exited"
        client_thread = getattr(self, "_inference_client_thread", None)
        if reason is None and client_thread is not None and not client_thread.is_alive():
            reason = "the inference-client event-loop thread exited"
        engine_task = getattr(getattr(self, "dynamic_inference_engine", None), "engine_loop_task", None)
        if reason is None and engine_task is not None and engine_task.done():
            reason = "the Megatron inference engine loop exited"
            if not engine_task.cancelled() and engine_task.exception() is not None:
                reason += f" ({engine_task.exception()!r})"
        if reason is not None:
            future.cancel()
            raise RuntimeError(
                f"[Rank {getattr(self, 'rank', '?')}] generation request cannot complete: {reason}. "
                "See the worker log for the engine traceback."
            )

    async def generate_async(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Streaming generation: yield `(index, batch)` tuples as they complete.

        Args:
            data: BatchedDataDict with input_ids and input_lengths
            greedy: Whether to use greedy decoding instead of sampling

        Yields:
            Tuple of (original_index, BatchedDataDict conforming to GenerationOutputSpec for the single sequence)
        """
        if self._inference_loop is None:
            raise RuntimeError(
                "Inference loop not initialized. Call prepare_for_generation() first."
            )

        async def _generate_single_item(
            index: int,
        ) -> tuple[int, BatchedDataDict[GenerationOutputSpec]]:
            datum = data.get_batch(index, 1)
            prompts, multi_modal_data_list, sampling_params = (
                self._prepare_data_for_generation(datum, greedy)
            )
            future = asyncio.run_coroutine_threadsafe(
                self._generate_with_persistent_engine(
                    prompts,
                    multi_modal_data_list,
                    sampling_params,
                ),
                self._client_loop(),
            )
            wrapped = asyncio.wrap_future(future)
            while True:
                try:
                    result = await asyncio.wait_for(
                        asyncio.shield(wrapped), timeout=_ENGINE_LIVENESS_POLL_S
                    )
                except asyncio.TimeoutError:
                    self._raise_if_inference_engine_dead(future)
                except asyncio.CancelledError:
                    # shield keeps the poll from cancelling the request; a cancelled caller still must.
                    wrapped.cancel()
                    raise
                else:
                    break
            output = self._parse_result_to_batched_data_dict(datum, result)
            return (index, output)

        tasks = [
            asyncio.create_task(_generate_single_item(i)) for i in range(data.size)
        ]
        try:
            for result in asyncio.as_completed(tasks):
                yield await result
        finally:
            # Closing the stream or failing one item must cancel its siblings.
            # Cancellation propagates through wrap_future to the inference loop.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _generate_with_persistent_engine(
        self,
        prompts: list[list[int]],
        multi_modal_data_list: list[Optional[Any]],
        sampling_params: list[SamplingParams],
    ) -> list:
        """Submit requests through the persistent inference client (rank 0 only)."""
        from megatron.core.inference.inference_request import DynamicInferenceRequest

        dist_rank = torch.distributed.get_rank()
        assert dist_rank == 0, (
            "Only rank 0 creates a client to communicate with the coordinator"
        )

        print(f"[Rank {dist_rank}] Submitting {len(prompts)} requests to coordinator")

        pending_requests = []
        try:
            for prompt, multi_modal_data, request_sampling_params in zip(
                prompts, multi_modal_data_list, sampling_params, strict=True
            ):
                pending_requests.append(
                    self.inference_client.add_request_with_id(
                        prompt,
                        request_sampling_params,
                        multi_modal_data=multi_modal_data,
                    )
                )

            results: list[DynamicInferenceRequest] = await asyncio.gather(
                *(future for _, future in pending_requests)
            )
        except BaseException:
            # CancelledError is a BaseException. Cancelling only the local
            # futures leaves requests running in Megatron's coordinator.
            for request_id, future in pending_requests:
                try:
                    self.inference_client.abort_request(request_id)
                except Exception:
                    logging.exception("Failed to abort Megatron request %s", request_id)
                finally:
                    future.cancel()
            await asyncio.gather(
                *(future for _, future in pending_requests), return_exceptions=True
            )
            raise
        print(f"[Rank {dist_rank}] Completed {len(results)} requests")
        return results


class MegatronGenerationRefitMixin:
    """Refit collective, weight transfer, and engine suspend/resume around refits."""

    def _init_generation_refit_state(self) -> None:
        """Reset every refit attribute to its unprepared state.

        Declaring these up front keeps ``prepare_refit_info`` readiness a value
        check rather than an ``hasattr`` probe, and makes the full set visible
        to anyone adding a ``__getstate__``.
        """
        # Bridge import plan, populated by prepare_refit_info.
        self._generation_refit_state_dict_info = None
        self._generation_refit_tasks = None
        self._generation_refit_model_chunks = None
        self._generation_refit_dependency_counts = None
        self._generation_mxfp8_destinations: (
            dict[tuple[int | None, str], MXFP8Tensor] | None
        ) = None
        # MXFP8 conversion removes the original nn.Parameters, so Bridge cannot
        # rediscover those tasks through named_parameters() during a later shard
        # recovery. Preserve the complete ordered task plan once destinations are
        # made persistent.
        self._generation_mxfp8_refit_tasks: list[_MegatronRefitTask] | None = None
        # Per-refit progress, reset at the start of each transfer.
        self._generation_refit_task_index = 0
        self._generation_refit_remaining_dependencies = {}
        self._generation_refit_pending_weights = {}
        self._generation_refit_pending_streams = {}
        # nccl_reshard (M-to-N) transport state. The source and destination
        # roles use the same public state names and RefitBuilderInterface.
        self._generation_nccl_reshard_groups = None
        self.nccl_reshard_refit_info = None
        self.hf_to_local_param_map = None
        self._generation_m2n_pending = None
        # Native MCore copy service, built by init_collective_mcore_generation.
        self.refit_copy_service = None
        self.refit_execution_batch_bytes: int | None = None
        # Refit TCPStore server pre-bound by reserve_refit_store (rank 0 only):
        # (port, store), consumed by the next init_collective_mcore_generation.
        self._prebound_refit_store: tuple[int, Any] | None = None

    def reserve_refit_store(
        self,
        world_size: int,
        port_range_low: int,
        port_range_high: int,
        preferred_port: int | None = None,
        max_attempts: int = 64,
    ) -> tuple[str, int]:
        """Bind the refit rendezvous TCPStore server here (global rank 0) and keep it.

        Picking a port on the driver and binding it later on rank 0 is a
        check-then-use race, and before this the bound port was not even the
        probed one (``port + 1``): any listener already on ``port + 1`` (the
        learner world's MASTER_PORT, another engine lane's store) made rank 0
        die with EADDRINUSE. Here the server is bound first, in the master
        port band, retrying on a bound port; the caller broadcasts the
        returned (ip, port) to every other rank. Cost: one listening socket on
        one rank, O(attempts) binds once per communicator, independent of
        world size.

        Returns:
            (node ip of this rank, bound port).
        """
        from nemo_rl.distributed.virtual_cluster import (
            _get_node_ip_local,
            refit_store_port_candidates,
        )

        if torch.distributed.get_rank() != 0:
            raise RuntimeError(
                "reserve_refit_store must run on the store master (global rank 0)"
            )
        candidates = refit_store_port_candidates(
            port_range_low, port_range_high, preferred_port, max_attempts
        )
        errors = []
        for port in candidates:
            try:
                store = torch.distributed.TCPStore(
                    host_name="0.0.0.0",
                    port=port,
                    world_size=world_size,
                    is_master=True,
                    wait_for_workers=False,
                )
            except (torch.distributed.DistNetworkError, RuntimeError) as e:
                errors.append(f"{port}: {type(e).__name__}")
                continue
            self._prebound_refit_store = (port, store)
            ip = _get_node_ip_local()
            print(
                f"  refit: native store bound on rank 0 at {ip}:{port} "
                f"(attempt {len(errors) + 1})",
                flush=True,
            )
            return ip, port
        raise RuntimeError(
            f"Could not bind the refit TCPStore in [{port_range_low}, "
            f"{port_range_high}) after {len(candidates)} attempts: {errors[-8:]}"
        )

    def init_collective_mcore_generation(
        self,
        ip: str,
        port: int,
        world_size: int,
        rank_offset: int,
        refit_execution_batch_bytes: int | None,
        refit_backend: str,
    ) -> None:
        """Initialize the refit collective for non-colocated weight transfer.

        Args:
            ip: IP address for the process group rendezvous.
            port: Port for the process group rendezvous.
            world_size: Total world size (train + inference workers).
            rank_offset: Offset for this side's ranks (`train_world_size` for inference).
            refit_execution_batch_bytes: Configured native MCore staging limit.
                None uses NeMo-RL's dynamic collective-refit default.
            refit_backend: Copy-service backend ("gloo", "nccl", or
                "nccl_m2n"; "nvshmem" is currently broken, see the issue
                below).
        """
        if refit_backend == "nvshmem":
            warnings.warn(
                'refit_backend="nvshmem" is currently broken; prefer "nccl" or '
                '"gloo". See https://github.com/NVIDIA-NeMo/RL/issues/3646',
                stacklevel=2,
            )
        execution_batch_bytes = resolve_refit_execution_batch_bytes(
            refit_execution_batch_bytes
        )

        local_rank = torch.distributed.get_rank()
        global_rank = local_rank + rank_offset

        # Bind/connect exactly `port`: the port the caller probed (or that
        # reserve_refit_store already bound here). Nothing else in this path
        # binds `port`; the old `port + 1` was neither probed nor reserved and
        # collided with other listeners (EADDRINUSE, cohort lanes).
        prebound = getattr(self, "_prebound_refit_store", None)
        self._prebound_refit_store = None
        if global_rank == 0 and prebound is not None and prebound[0] == port:
            store = prebound[1]
        else:
            store = torch.distributed.TCPStore(
                host_name=ip,
                port=port,
                world_size=world_size,
                is_master=(global_rank == 0),
            )

        group_name = "refit"
        pg_prefix_store = PrefixStore(f"{group_name}/", store)

        # Training and inference workers run in separate torch.distributed worlds.
        # The public APIs (new_group, init_process_group) assume all ranks belong to one world;
        # new_group validates ranks against the default PG, and init_process_group can only
        # be called once. We construct the PG manually using the same internal pattern as
        # _new_process_group_helper, skipping the single-world assumptions.
        pg = ProcessGroup(pg_prefix_store, global_rank, world_size)
        gloo_store = PrefixStore("cpu/", pg_prefix_store)
        gloo_backend = ProcessGroupGloo(gloo_store, global_rank, world_size)
        gloo_backend._set_sequence_number_for_group()
        pg._register_backend(
            torch.device("cpu"),
            ProcessGroup.BackendType.GLOO,
            gloo_backend,
        )
        pg._set_default_backend(ProcessGroup.BackendType.GLOO)

        # The NCCL copy service moves the actual weight bytes with CUDA-tensor P2P
        # (`torch.distributed.batch_isend_irecv`), which needs an NCCL backend
        # registered for the cuda device on this cross-world PG. GLOO stays the
        # default backend so the object collectives in `prepare_swap_model_weights`
        # (all_gather_object / broadcast_object_list) keep using CPU tensors.
        # NCCLM2NCopyService also uses this backend to coordinate topology and
        # communicator IDs before native M-to-N calls.
        if refit_backend in ("nccl", "nccl_m2n"):
            from torch.distributed.distributed_c10d import ProcessGroupNCCL

            # Ensure the NCCL communicator binds to this rank's own GPU.
            torch.cuda.set_device(torch.cuda.current_device())
            nccl_store = PrefixStore("cuda/", pg_prefix_store)
            nccl_options = ProcessGroupNCCL.Options()
            nccl_backend = ProcessGroupNCCL(
                nccl_store, global_rank, world_size, nccl_options
            )
            nccl_backend._set_sequence_number_for_group()
            # Create the group-wide NCCL communicator now, on every rank.
            nccl_backend.eager_connect_single_device(
                torch.device("cuda", torch.cuda.current_device())
            )
            pg._register_backend(
                torch.device("cuda"),
                ProcessGroup.BackendType.NCCL,
                nccl_backend,
            )

        pg._set_group_name(group_name)

        self.refit_pg = pg
        self.refit_execution_batch_bytes = execution_batch_bytes

        # Register in torch.distributed's global state so that high-level ops
        # (all_gather_object, broadcast_object_list) work with this PG.
        _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
        _world.pg_map[pg] = ("gloo", pg_prefix_store)
        _world.pg_names[pg] = group_name

        if refit_backend == "nvshmem":
            # Deferred: importing NVSHMEMCopyService loads the optional nvshmem bindings.
            from megatron.core.resharding.copy_services.nvshmem_copy_service import (
                NVSHMEMCopyService,
            )

            self.refit_copy_service = NVSHMEMCopyService(group=self.refit_pg)
        elif refit_backend == "nccl_m2n":
            self.refit_copy_service = NCCLM2NCopyService(group=self.refit_pg)
        elif refit_backend == "nccl":
            self.refit_copy_service = NCCLCopyService(group=self.refit_pg)
        else:
            self.refit_copy_service = GlooCopyService(group=self.refit_pg)

        is_source = rank_offset == 0
        # Cache for later refit calls (swap_weights_via_reshard).
        self.refit_dst_rank_offset = (
            torch.distributed.get_world_size() if is_source else rank_offset
        )

        # Build and cache the reshard plan (and any MXFP8 transforms) collectively.
        # All participating ranks (training + generation) call this simultaneously.
        prepare_swap_model_weights(
            src_model=self.model if is_source else None,
            target_model=None if is_source else self.model,
            group=self.refit_pg,
            src_rank_offset=0,
            dst_rank_offset=self.refit_dst_rank_offset,
            execution_batch_bytes=self.refit_execution_batch_bytes,
        )

    def preinit_nvshmem_collective(self) -> None:
        """Initialize NVShmem collectively before any weight transfer.

        Must be called on ALL participating ranks (training + inference) simultaneously,
        outside CUDA graph capture. Lazy initialization during graph recording or replay
        can corrupt CUDA graph state.
        """
        if self.refit_copy_service is None:
            return
        if not hasattr(self.refit_copy_service, "_ensure_initialized"):
            return
        self.refit_copy_service._ensure_initialized()

    def swap_weights_via_reshard(self, is_source: bool) -> bool:
        """Transfer weights using Megatron's `swap_model_weights` API.

        Args:
            is_source: True for training workers (senders), False for inference workers (receivers).

        Returns:
            True on success.
        """
        src_model = self.model if is_source else None
        dst_model = None if is_source else self.model
        if self.refit_execution_batch_bytes is None:
            raise RuntimeError("Native MCore refit was not initialized.")

        transform = None
        if not is_source:
            from nemo_rl.models.megatron import fp16_gemm

            if fp16_gemm.enabled():
                # The engine holds dense GEMM weights (hold_fp16_weights) and its
                # expert weights (fp16_moe) in fp16 while the plan's wire dtype is
                # the learner's bf16; see Fp16HeldWeightReceive.
                transform = getattr(self, "_fp16_refit_receive", None)
                if transform is None:
                    transform = self._fp16_refit_receive = Fp16HeldWeightReceive(
                        _refit_core(self.model)
                    )
        # Destination: engine tensors allocated lazily during generation (the fp16
        # expert buffers) are inference tensors, which in-place writes may only
        # touch under inference mode (as in _reshard_into_inference_model).
        with torch.inference_mode() if dst_model is not None else nullcontext():
            swap_model_weights(
                src_model,
                dst_model,
                refit_method=self.refit_copy_service,
                group=self.refit_pg,
                src_rank_offset=0,
                dst_rank_offset=self.refit_dst_rank_offset,
                transform=transform,
                execution_batch_bytes=self.refit_execution_batch_bytes,
            )

        return True

    def init_nccl_reshard_comm_groups_generation(
        self,
        *,
        pp_ips: list[str],
        pp_ports: list[int],
        pp_size: int,
        train_ranks_per_stage: int,
        sub_world_size: int,
        rank_prefix: Optional[int] = None,
    ) -> None:
        """Join every training PP stage's M-to-N communicator as a gen rank."""
        # Keep this local because the module imports optional NCCL4Py bindings.
        from nemo_rl.distributed.refit_watchdog import RELEASE_GRACE_S, release_within
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        generation_rank = self.rank if rank_prefix is None else rank_prefix
        rank_in_group = train_ranks_per_stage + generation_rank
        torch.cuda.empty_cache()
        stale_groups = getattr(self, "_generation_nccl_reshard_groups", None) or {}
        self._generation_nccl_reshard_groups = {}
        for stage in range(pp_size):
            group = StatelessProcessGroup(
                master_address=pp_ips[stage],
                port=pp_ports[stage],
                rank=rank_in_group,
                world_size=sub_world_size,
            )
            group.init_nccl_communicator(device=torch.cuda.current_device())
            self._generation_nccl_reshard_groups[stage] = group

        for previous in stale_groups.values():
            release_within(
                previous.abort, RELEASE_GRACE_S, "a previous reshard bulk communicator"
            )

    def _commit_megatron_bulk_refit_piece(
        self, piece: _MegatronBulkRefitPiece, tensor: torch.Tensor
    ) -> None:
        """Commit one received HF-local weight to its inference destination."""
        if tensor.shape != piece.shape:
            raise ValueError(
                f"Shape mismatch for Megatron bulk refit weight {piece.spec.name!r} "
                f"of {piece.task.param_name!r}: expected {tuple(piece.shape)}, "
                f"got {tuple(tensor.shape)}."
            )
        if not piece.task.is_quantized:
            assert piece.destination is not None
            piece.spec.select(piece.destination).copy_(tensor)
            return

        pending = self._generation_m2n_pending.setdefault(piece.task.target_id, {})
        pending[piece.spec.name] = tensor
        required = {
            spec.name for spec in piece.task.conversion_task.local_hf_param_specs()
        }
        if required.issubset(pending):
            logical_weight = piece.task.conversion_task.combine_local_hf_weights(
                pending
            )
            if piece.task.is_mxfp8:
                self._write_generation_refit_weight(piece.task, logical_weight)
            else:
                if logical_weight.shape != piece.task.expected_shape:
                    raise ValueError(
                        f"Shape mismatch for Megatron parameter {piece.task.param_name!r}: "
                        f"expected {tuple(piece.task.expected_shape)}, "
                        f"got {tuple(logical_weight.shape)}."
                    )
                # Only bulk hooks use MCore's TE/grouped quantized writer.
                # Keep this optional API out of the packed collective/misc path.
                from megatron.core.fp8_utils import copy_tensor_to_quantized_param

                copy_tensor_to_quantized_param(piece.task.destination, logical_weight)
            del self._generation_m2n_pending[piece.task.target_id]

    def _build_destination_hf_to_local_param_map(
        self,
        refit_info: dict[str, Any],
        tasks: list[_MegatronRefitTask],
    ) -> HFToLocalParamMap:
        """Map canonical HF FFN shards onto local Megatron destinations."""
        pieces: dict[str, _MegatronBulkRefitPiece] = {}

        def add_piece(task: _MegatronRefitTask, spec: Any) -> None:
            if spec.name in pieces:
                raise RuntimeError(
                    f"Duplicate Megatron M-to-N target for {spec.name!r}."
                )
            destination = None if task.is_quantized else task.destination
            pieces[spec.name] = _MegatronBulkRefitPiece(
                task=task,
                spec=spec,
                shape=spec.selected_shape(task.expected_shape),
                device=task.destination.device,
                destination=destination,
            )

        for task in tasks:
            for spec in task.conversion_task.local_hf_param_specs():
                if is_nccl_reshard_param(spec.name):
                    add_piece(task, spec)

        expert_pieces: dict[
            tuple[str, str], list[tuple[int, _MegatronBulkRefitPiece]]
        ] = {}
        for name, piece in pieces.items():
            match = _INDIVIDUAL_EXPERT_RE.match(name)
            if match:
                expert_pieces.setdefault((match.group(1), match.group(3)), []).append(
                    (int(match.group(2)), piece)
                )

        def grouped_spec(
            grouped: tuple[_MegatronBulkRefitPiece, ...],
            wire_dtype: torch.dtype,
        ) -> LocalParamSpec:
            first = grouped[0]

            def pre(_base: Any) -> RefitCtx:
                return RefitCtx(
                    buf=torch.empty(
                        (len(grouped), *first.shape),
                        dtype=wire_dtype,
                        device=first.device,
                    )
                )

            def post(ctx: RefitCtx) -> None:
                for received, piece in zip(ctx.buf.unbind(0), grouped, strict=True):
                    self._commit_megatron_bulk_refit_piece(piece, received)

            return LocalParamSpec(base=None, pre=pre, post=post, wire_dtype=wire_dtype)

        def staged_spec(
            piece: _MegatronBulkRefitPiece, wire_dtype: torch.dtype
        ) -> LocalParamSpec:
            def pre(_base: Any) -> RefitCtx:
                return RefitCtx(
                    buf=torch.empty(piece.shape, dtype=wire_dtype, device=piece.device)
                )

            def post(ctx: RefitCtx) -> None:
                self._commit_megatron_bulk_refit_piece(piece, ctx.buf)

            return LocalParamSpec(base=None, pre=pre, post=post, wire_dtype=wire_dtype)

        def direct_spec(
            piece: _MegatronBulkRefitPiece, wire_dtype: torch.dtype
        ) -> LocalParamSpec:
            assert piece.destination is not None

            def pre(base: torch.Tensor) -> RefitCtx:
                return RefitCtx(buf=piece.spec.select(base))

            return LocalParamSpec(
                base=piece.destination, pre=pre, wire_dtype=wire_dtype
            )

        specs = {}
        for layer_name in refit_info["layer_names"]:
            for param_info in refit_info["per_layer_params"][layer_name]:
                name = param_info["name"]
                wire_dtype_value = param_info.get("dtype")
                wire_dtype = _STR_TO_DTYPE.get(str(wire_dtype_value))
                if wire_dtype is None or wire_dtype not in (
                    torch.bfloat16,
                    torch.float16,
                    torch.float32,
                ):
                    raise ValueError(
                        f"Unsupported Megatron bulk refit wire dtype "
                        f"{wire_dtype_value!r} for {name!r}."
                    )
                grouped_proj = param_info.get("grouped_expert_proj")
                if grouped_proj is not None:
                    prefix = name.rsplit(f".{grouped_proj}.weight", 1)[0]
                    grouped = tuple(
                        piece
                        for _, piece in sorted(
                            expert_pieces.get((prefix, grouped_proj), [])
                        )
                    )
                    if not grouped:
                        raise ValueError(
                            f"No local Megatron experts map to M-to-N weight {name!r}."
                        )
                    specs[name] = grouped_spec(grouped, wire_dtype)
                    continue

                piece = pieces.get(name)
                if piece is None:
                    raise ValueError(
                        f"No local Megatron destination maps to M-to-N weight {name!r}."
                    )
                specs[name] = (
                    staged_spec(piece, wire_dtype)
                    if piece.task.is_quantized or piece.destination.dtype != wire_dtype
                    else direct_spec(piece, wire_dtype)
                )

        return HFToLocalParamMap(specs=specs)

    def _prepare_mxfp8_refit(self, tasks: list[_MegatronRefitTask]) -> None:
        """Install persistent MCore MXFP8 destinations before engine initialization."""
        if self._generation_mxfp8_destinations is not None:
            for task in tasks:
                destination = self._generation_mxfp8_destinations.get(task.cache_key)
                if destination is not None:
                    task.destination = destination
                    task.target_id = id(destination)
            return

        model_chunks = (
            self.model if isinstance(self.model, (list, tuple)) else [self.model]
        )
        cores = [unwrap_model(model) for model in model_chunks]
        mxfp8_cores = [
            core
            for core in cores
            if getattr(core.config, "transformer_impl", None) == "inference_optimized"
            and getattr(core.config, "fp8_recipe", None) == "mxfp8"
        ]
        if not mxfp8_cores:
            self._generation_mxfp8_destinations = {}
            return
        if self._inference_engine_initialized:
            raise RuntimeError(
                "MXFP8 refit buffers must be prepared before inference-engine "
                "initialization; construct MegatronGeneration with skip_weight_load=True."
            )

        destination_by_id: dict[int, MXFP8Tensor] = {}
        for core in mxfp8_cores:
            decoder = core.decoder if hasattr(core, "decoder") else core
            param_name_by_id = {
                id(param): name for name, param in decoder.named_parameters()
            }
            persistent_buffers = quantize_params_to_mxfp8(
                decoder, backend=_resolve_mxfp8_refit_backend(core.config)
            )
            destination_by_id.update(
                {
                    param_id: persistent_buffers[name]
                    for param_id, name in param_name_by_id.items()
                    if name in persistent_buffers
                }
            )

        destinations: dict[tuple[int | None, str], MXFP8Tensor] = {}
        for task in tasks:
            if task.target_id in destination_by_id:
                task.destination = destination_by_id[task.target_id]
                task.target_id = id(task.destination)
                destinations[task.cache_key] = task.destination
        self._generation_mxfp8_destinations = destinations
        self._generation_mxfp8_refit_tasks = list(tasks)

    def _refresh_flashinfer_mxfp8_weights(self) -> None:
        """Refresh derived FlashInfer expert storage after canonical weights change."""
        model_chunks = (
            self.model if isinstance(self.model, (list, tuple)) else [self.model]
        )
        refreshed = False
        for model in model_chunks:
            core = unwrap_model(model)
            for module in core.modules():
                refresh = getattr(module, "refresh_flashinfer_mxfp8_weights", None)
                if refresh is not None:
                    refreshed = bool(refresh()) or refreshed
        if refreshed:
            # Repacking is asynchronous. Finish before another stream replays
            # CUDA graphs that read the derived expert buffers.
            torch.cuda.synchronize()

    def _build_generation_refit_tasks(
        self,
    ) -> tuple[list[torch.nn.Module], list[_MegatronRefitTask]]:
        """Resolve Bridge import tasks against the persistent inference model."""
        model_chunks = (
            list(self.model) if isinstance(self.model, (list, tuple)) else [self.model]
        )
        if self._generation_mxfp8_refit_tasks is not None:
            return model_chunks, list(self._generation_mxfp8_refit_tasks)

        conversion_tasks = self.megatron_bridge.get_conversion_tasks(model_chunks)
        tasks: list[_MegatronRefitTask] = []
        for conversion_task in conversion_tasks:
            if conversion_task is None or conversion_task.megatron_module is None:
                continue
            destination = conversion_task.param_weight
            if destination is None:
                raise RuntimeError(
                    f"Bridge task {conversion_task.param_name!r} has no destination."
                )
            tasks.append(
                _MegatronRefitTask(
                    conversion_task=replace(conversion_task, param_weight=None),
                    destination=destination,
                    target_id=id(destination),
                )
            )

        if not tasks:
            raise RuntimeError("Megatron Bridge produced no local refit tasks.")
        return model_chunks, tasks

    def _install_generation_refit_plan(
        self,
        *,
        model_chunks: list[torch.nn.Module],
        tasks: list[_MegatronRefitTask],
        state_dict_info: dict[str, Any],
    ) -> None:
        """Install the packed-broadcast subset of a Bridge receive plan."""
        if not state_dict_info:
            raise ValueError(
                "Megatron packed refit requires non-empty state_dict_info."
            )

        required_names = {name for task in tasks for name in task.dependencies}
        missing_names = sorted(required_names.difference(state_dict_info))
        if missing_names:
            preview = ", ".join(missing_names[:20])
            raise ValueError(
                "Megatron refit metadata is missing Bridge inputs: "
                f"{preview}{' ...' if len(missing_names) > 20 else ''}"
            )

        self._generation_refit_model_chunks = model_chunks
        self._generation_refit_tasks = tasks
        self._generation_refit_dependency_counts = Counter(
            name for task in tasks for name in task.dependencies
        )
        self._generation_refit_state_dict_info = state_dict_info

    @torch.no_grad()
    def _prepare_destination_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Build the local HF-to-Megatron receive plan before the first refit."""
        model_chunks, tasks = self._build_generation_refit_tasks()
        self._prepare_mxfp8_refit(tasks)
        self._install_generation_refit_plan(
            model_chunks=model_chunks,
            tasks=tasks,
            state_dict_info=state_dict_info,
        )

    @torch.no_grad()
    def _prepare_destination_nccl_reshard_refit_info(
        self, refit_info: dict[str, Any]
    ) -> None:
        """Prepare Megatron as the destination of NCCL M-to-N reshard refit."""
        refit_info = restore_refit_info_placements(refit_info)
        model_chunks, tasks = self._build_generation_refit_tasks()
        self._prepare_mxfp8_refit(tasks)
        bulk_map = self.build_hf_to_local_param_map(refit_info, destination_tasks=tasks)

        misc_meta = refit_info.get("misc_meta", {})
        misc_state_dict_info = {
            name: (tuple(meta["shape"]), _STR_TO_DTYPE[meta["dtype"]])
            for name, meta in misc_meta.items()
        }
        misc_tasks = [
            task
            for task in tasks
            if set(task.dependencies).issubset(misc_state_dict_info)
        ]
        self._install_generation_refit_plan(
            model_chunks=model_chunks,
            tasks=misc_tasks,
            state_dict_info=misc_state_dict_info,
        )
        self.nccl_reshard_refit_info = refit_info
        self.hf_to_local_param_map = bulk_map

    @torch.no_grad()
    def _destination_nccl_reshard_refit(
        self, refit_timeout_s: Optional[float] = None
    ) -> bool:
        """Receive bulk FFN shards via M-to-N, then import packed misc weights."""
        from nemo_rl.distributed.refit_watchdog import sync_stream_within

        # Keep this transport import local: xferdtensor probes the optional
        # nccl.m2n extension at import time.
        from nemo_rl.weight_sync.xferdtensor import DTensorRef, xferdtensor

        self._generation_m2n_pending: dict[int, dict[str, torch.Tensor]] = {}

        def receive_one(param_info: dict[str, Any], group: Any, stream: Any) -> None:
            spec = self.hf_to_local_param_map.get(param_info["name"])
            if spec is None:
                raise RuntimeError(
                    f"Megatron M-to-N refit has no destination for {param_info['name']!r}."
                )
            ctx = spec.materialize()
            destination = DTensorRef(ctx.buf, param_info["global_shape"])
            xferdtensor(
                None,
                param_info["src_mesh_info"],
                param_info["src_placements"],
                destination,
                param_info["dst_mesh_info"],
                param_info["dst_placements"],
                group,
                stream,
            )
            if spec.post is not None:
                spec.post(ctx)

        stage_params: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        refit_info = self.nccl_reshard_refit_info
        for layer_name in refit_info["layer_names"]:
            for param_info in refit_info["per_layer_params"][layer_name]:
                stage_params.setdefault(param_info.get("pp_stage", 0), []).append(
                    param_info
                )

        num_streams = max(
            1,
            min(int(os.environ.get("NRL_REFIT_NUM_STREAMS", "2")), len(stage_params)),
        )
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        events: dict[int, torch.cuda.Event] = {}
        try:
            for index, (stage, params) in enumerate(stage_params.items()):
                previous_index = index - num_streams
                if previous_index in events:
                    events[previous_index].synchronize()
                stage_stream = streams[index % num_streams]
                with torch.cuda.stream(stage_stream):
                    group = self._generation_nccl_reshard_groups[stage]
                    for param_info in params:
                        receive_one(param_info, group, stage_stream)
                    event = torch.cuda.Event()
                    event.record()
                    events[index] = event

            current_stream = torch.cuda.current_stream()
            for event in events.values():
                current_stream.wait_event(event)
            sync_stream_within(
                current_stream,
                refit_timeout_s,
                "the Megatron destination bulk parameter transfer",
            )
            if self._generation_m2n_pending:
                raise RuntimeError(
                    "Megatron M-to-N refit ended with incomplete fused quantized weights: "
                    f"{sorted(self._generation_m2n_pending)}"
                )
            torch.cuda.empty_cache()
            result = self._update_destination_weights_from_collective(
                refit_timeout_s=refit_timeout_s,
                refresh_mxfp8=False,
            )
            self._refresh_flashinfer_mxfp8_weights()
            return result
        finally:
            self._generation_m2n_pending.clear()

    def _write_generation_refit_weight(
        self, task: _MegatronRefitTask, converted_weight: torch.Tensor
    ) -> None:
        """Copy one converted weight into its persistent inference destination.

        ``copy_`` rather than rebinding: an MXFP8 destination quantizes in place
        and keeps its storage address, which is what already-captured CUDA
        graphs point at.

        Args:
            task: The Bridge import task owning the destination tensor.
            converted_weight: The assembled weight in the destination's layout.

        Raises:
            ValueError: If the converted weight does not match the destination
                shape, which means the refit plan and the model disagree.
        """
        if converted_weight.shape != task.expected_shape:
            raise ValueError(
                f"Shape mismatch for Megatron parameter {task.param_name!r}: "
                f"expected {tuple(task.expected_shape)}, got {tuple(converted_weight.shape)}."
            )

        task.destination.copy_(converted_weight)

    def _load_generation_refit_batch(
        self, weights: list[tuple[str, torch.Tensor]]
    ) -> None:
        """Import every Bridge task whose HF inputs have now all arrived.

        Called once per ``packed_broadcast_consumer`` batch. Weights arrive in
        HF order, but a task may fuse several of them (gate/up, stacked
        experts), so each tensor is buffered until its task's dependency count
        reaches zero, then the task is converted and written in task order.
        Buffered tensors record their producing stream so the conversion can
        wait on it before reading across streams.

        Args:
            weights: ``(hf_name, tensor)`` pairs unpacked from this batch.
        """
        batch_stream = (
            torch.cuda.current_stream() if weights and weights[0][1].is_cuda else None
        )
        for name, tensor in weights:
            if self._generation_refit_remaining_dependencies.get(name, 0) > 0:
                self._generation_refit_pending_weights[name] = tensor
                if batch_stream is not None:
                    self._generation_refit_pending_streams[name] = batch_stream

        tasks = self._generation_refit_tasks
        while self._generation_refit_task_index < len(tasks):
            task = tasks[self._generation_refit_task_index]
            if any(
                name not in self._generation_refit_pending_weights
                for name in task.dependencies
            ):
                break

            # packed_broadcast_consumer alternates CUDA streams. A compound Bridge
            # mapping can retain one input across a batch boundary, so make the
            # current batch wait for the stream that received each retained input.
            current_stream = (
                torch.cuda.current_stream() if batch_stream is not None else None
            )
            for name in task.dependencies:
                source_stream = self._generation_refit_pending_streams.get(name)
                if current_stream is not None and source_stream is not None:
                    current_stream.wait_stream(source_stream)

            converted = next(
                iter(
                    self.megatron_bridge.stream_weights_hf_to_megatron(
                        self._generation_refit_model_chunks,
                        conversion_tasks=[task.conversion_task],
                        hf_state_dict=self._generation_refit_pending_weights,
                    )
                ),
                None,
            )
            if converted is None:
                raise RuntimeError(
                    f"Bridge produced no local value for {task.param_name!r}."
                )
            self._write_generation_refit_weight(task, converted.weight)

            for name in task.dependencies:
                self._generation_refit_remaining_dependencies[name] -= 1
                if self._generation_refit_remaining_dependencies[name] == 0:
                    del self._generation_refit_pending_weights[name]
                    self._generation_refit_pending_streams.pop(name, None)
            self._generation_refit_task_index += 1

    @torch.no_grad()
    def _update_destination_weights_from_collective(
        self,
        refit_timeout_s: Optional[float] = None,
        *,
        refresh_mxfp8: bool = True,
    ) -> bool:
        """Receive packed HF tensors and import them into the Megatron model."""
        from nemo_rl.distributed.refit_watchdog import sync_stream_within

        if self._generation_refit_state_dict_info is None:
            raise RuntimeError(
                "Megatron refit metadata is not prepared. Call prepare_refit_info first."
            )

        self._generation_refit_task_index = 0
        self._generation_refit_pending_weights: dict[str, torch.Tensor] = {}
        self._generation_refit_pending_streams: dict[str, torch.cuda.Stream] = {}
        self._generation_refit_remaining_dependencies = Counter(
            self._generation_refit_dependency_counts
        )
        try:
            packed_broadcast_consumer(
                iterator=iter(self._generation_refit_state_dict_info.items()),
                group=self.model_update_group,
                src=0,
                post_unpack_func=self._load_generation_refit_batch,
            )
            if self._generation_refit_task_index != len(self._generation_refit_tasks):
                task = self._generation_refit_tasks[self._generation_refit_task_index]
                missing = [
                    name
                    for name in task.dependencies
                    if name not in self._generation_refit_pending_weights
                ]
                raise RuntimeError(
                    f"Megatron refit ended before {task.param_name!r}; "
                    f"missing inputs: {missing}."
                )

            self.megatron_bridge.finalize_hf_import(self._generation_refit_model_chunks)
            if refresh_mxfp8:
                self._refresh_flashinfer_mxfp8_weights()
            sync_stream_within(
                torch.cuda.current_stream(),
                refit_timeout_s,
                "the Megatron destination packed parameter transfer",
            )
            return True
        finally:
            self._generation_refit_pending_weights.clear()
            self._generation_refit_pending_streams.clear()

    def _onload_inference_model(self) -> None:
        """Restore the colocated inference weights to GPU before resharding / generation."""
        if not self._inference_model_offloaded:
            return
        resume_inference_weights(getattr(self, "_inference_model_kept", ()))
        self._inference_model_kept = []
        self._inference_model_offloaded = False

    def _offload_inference_model(self) -> None:
        """Offload the colocated inference weights to CPU while training runs."""
        if (
            self.inference_model is None
            or self._inference_model_offloaded
            or not HAVE_TORCH_MEMORY_SAVER
        ):
            return
        self._inference_model_kept = pause_inference_weights(
            getattr(self.inference_model, "_nrl_offload_keep", ())
        )
        self._inference_model_offloaded = True

    def _reshard_into_inference_model(self) -> None:
        """Reshard current training weights into the colocated inference-layout model."""
        inference_model = self.inference_model
        if inference_model is None:
            return
        configured_batch_bytes = self.cfg["generation"]["mcore_generation_config"][
            "refit_execution_batch_bytes"
        ]
        # Preserve MCore's single-submission colocated path unless batching is
        # explicitly requested. Non-colocated native refit resolves null to a
        # bounded default when its cross-worker collective is initialized.
        execution_batch_bytes = (
            resolve_refit_execution_batch_bytes(configured_batch_bytes)
            if configured_batch_bytes is not None
            else None
        )

        # Bring the inference weights back to GPU.
        self._onload_inference_model()
        self.model = self.move_model(
            self.model, "cuda", move_params=True, move_grads=False
        )
        # TODO: Optimize away the full synchronization.
        torch.cuda.synchronize()

        # The swap reads the training params as its source;
        # under overlap_param_gather they stay stale after the optimizer step until gathered.
        if self.should_disable_forward_pre_hook and self._forward_pre_hook_enabled():
            self._disable_forward_pre_hook_until_next_train_step(param_sync=True)

        # Build + cache the same-rank reshard plan once, before the first CUDA-graph capture.
        if not self._swap_weights_plan_prepared:
            prepare_swap_model_weights(
                src_model=self.model,
                target_model=inference_model,
                group=None,
                src_rank_offset=0,
                dst_rank_offset=0,
                execution_batch_bytes=execution_batch_bytes,
            )
            self._swap_weights_plan_prepared = True

        # Dense GEMM weights the engine holds in fp16 (_prepare_engine_numerics, from the second wake on) are
        # sent as the bf16 training slice and written with copy_ (Fp16HeldLocalReshard); a dtype-strict receive
        # would reinterpret the bf16 bits. Tensors that are not fp16-held keep the direct same-rank copy.
        from nemo_rl.models.megatron import fp16_gemm

        transform = None
        if fp16_gemm.enabled():
            transform = getattr(self, "_fp16_colocated_reshard", None)
            if transform is None:
                transform = self._fp16_colocated_reshard = Fp16HeldLocalReshard(
                    _refit_core(inference_model)
                )
        n_wakes = getattr(self, "_colocated_reshards", 0)
        self._colocated_reshards = n_wakes + 1
        if n_wakes >= 1 and os.environ.get("NEMO_RL_DEBUG_COLOC_SKIP_RESHARD") == "1":
            # Negative control (ThunderSync colo X1 gate, refused without CHECKS by the colo validator): the engine
            # keeps its first weights, so the train-lp sentinel and the law check must fail from step 1 on.
            print(f"[colocated-reshard] DEBUG skip reshard (wake {n_wakes})", flush=True)
        else:
            # The inference model only runs under inference mode, where tensors it
            # allocates lazily (e.g. the fp16 GEMM weight caches of the zero-mismatch
            # recipe) become inference tensors; outside inference mode MCore's in-place
            # copy into them raises. Write the new weights under inference mode too.
            torch.cuda.synchronize()
            t_swap = time.perf_counter()
            with torch.inference_mode():
                swap_model_weights(
                    self.model,
                    inference_model,
                    refit_method=self.cfg["generation"]["mcore_generation_config"][
                        "refit_backend"
                    ],
                    group=None,
                    src_rank_offset=0,
                    dst_rank_offset=0,
                    transform=transform,
                    execution_batch_bytes=execution_batch_bytes,
                )
            torch.cuda.synchronize()
            if self.rank == 0:
                st = WEIGHT_OFFLOAD_STATS
                print(
                    f"[colo weights] wake {n_wakes}: reshard {time.perf_counter() - t_swap:.3f}s"
                    + (
                        f" | wait {st.get('resume_wait_s', 0):.3f}s resume {st.get('resume_s', 0):.3f}s"
                        f" (call {st.get('resume_call_s', 0):.3f}s)"
                        f" keep restore {st.get('keep_restore_s', 0):.3f}s"
                        f" | last pause: sync {st.get('pause_sync_s', 0):.3f}s keep copy {st.get('keep_copy_s', 0):.3f}s"
                        f" pause {st.get('pause_s', 0):.3f}s (call {st.get('pause_call_s', 0):.3f}s)"
                        f" keep {st.get('keep_n', 0)} buffers"
                        f" {st.get('keep_bytes', 0) / 2**20:.1f} MiB"
                        if st
                        else ""
                    ),
                    flush=True,
                )
        # Offload training model.
        self.model = self.move_model(
            self.model, "cpu", move_params=True, move_grads=False
        )
        # TODO: Optimize away the full synchronization.
        torch.cuda.synchronize()

    def suspend_for_refit(self) -> None:
        """Pause+suspend the inference engine before a weight refit."""
        if not self._inference_engine_initialized:
            return
        self._sleep()

    def resume_after_refit(self) -> None:
        """Resume+unpause the inference engine after a weight refit."""
        if not self._inference_engine_initialized:
            return
        self._wake()
