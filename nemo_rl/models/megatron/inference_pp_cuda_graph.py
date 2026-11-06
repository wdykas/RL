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
"""Block-scope inference CUDA graphs on non-first pipeline stages.

With ``inference_cuda_graph_scope=block`` Megatron graphs the whole
``GPTModel`` forward, and its CUDA graph manager refreshes only the call's
tensor arguments before each replay. A non-first pipeline stage gets its input
another way: the inference wrapper calls ``set_input_tensor`` with the tensor
received from the previous stage, ``GPTModel`` passes ``decoder_input=None``,
and ``TransformerBlock.forward`` reads ``self.input_tensor``. The graph
therefore captures the address of the receive buffer that existed at capture
time, and every replay reads that stale buffer instead of the activations just
received. Measured on Qwen3-1.7B with inference PP=2: ~45-nat log-prob gap to
training and reward ~0; with layer-scope graphs or eager mode the same run is
exact. ``HybridModel`` / ``HybridStack`` have the same structure.

``install()`` routes the stage input through the graph: on the graphed path of
a non-first stage the received tensor is passed as ``decoder_input`` (so the
manager copies it into the graph's static input on replay), and the decoder
reads the ``hidden_states`` it is given instead of its ``input_tensor``.
"""

from __future__ import annotations

from typing import Any

_INSTALLED = False


def install() -> None:
    """Patch the block-graphed model classes so their graphs read the stage input.

    Covers ``GPTModel`` / ``TransformerBlock`` and ``HybridModel`` / ``HybridStack``,
    the two model classes with block-scope inference graphs.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    allow_attention_free_stages()
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.transformer_block import TransformerBlock

    _patch(GPTModel, TransformerBlock)
    try:
        from megatron.core.models.hybrid.hybrid_block import HybridStack
        from megatron.core.models.hybrid.hybrid_model import HybridModel
    except ImportError:
        pass
    else:
        _patch(HybridModel, HybridStack)
    _INSTALLED = True


def _patch(model_cls: type, block_cls: type) -> None:
    from megatron.core.utils import WrappedTensor

    orig_call = model_cls.__call__
    orig_block_forward = block_cls.forward

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        decoder = getattr(self, "decoder", None)
        if (
            self.pre_process
            or decoder is None
            or decoder.input_tensor is None
            or kwargs.get("decoder_input") is not None
            or not self._should_call_local_cudagraph(*args, **kwargs)
        ):
            return orig_call(self, *args, **kwargs)
        kwargs["decoder_input"] = decoder.input_tensor
        decoder._graph_stage_input = True
        try:
            return orig_call(self, *args, **kwargs)
        finally:
            decoder._graph_stage_input = False

    def block_forward(self, *args: Any, **kwargs: Any) -> Any:
        if not getattr(self, "_graph_stage_input", False):
            return orig_block_forward(self, *args, **kwargs)
        # Inside the graphed model forward: hidden_states is the graph's static input.
        hidden_states = (
            kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        )
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()
            if "hidden_states" in kwargs:
                kwargs["hidden_states"] = hidden_states
            else:
                args = (hidden_states, *args[1:])
        received = self.input_tensor
        self.input_tensor = hidden_states
        try:
            return orig_block_forward(self, *args, **kwargs)
        finally:
            self.input_tensor = received

    model_cls.__call__ = __call__
    block_cls.forward = block_forward


_PHANTOM = -1  # layer-map key of the stand-in attention layer of an attention-free pipeline stage


def allow_attention_free_stages() -> None:
    """Engine PP > 1 on hybrid models: a pipeline stage with no attention layer (idempotent).

    ``DynamicInferenceContext`` sizes its paged KV blocks per attention layer of the stage and raises
    "with no attention is not supported" when a stage has none (e.g. Nemotron-H L8 at PP 2: MEME | M*EM). The
    block bookkeeping (token -> block tables, request admission) is still needed on that stage: every stage runs
    the same schedule, and the block count is already the MIN over the PP group. Fix: a stand-in attention entry
    (layer-map key -1, used by no layer) sizes blocks as one layer -- the stage's count then never binds the MIN --
    and the KV buffer is allocated with zero layers (no memory).
    """
    from megatron.core.inference.contexts import dynamic_context as dc

    if getattr(dc, "_ts_attention_free", False):
        return
    orig_maps = dc.get_layer_maps_from_layer_type_list

    def get_layer_maps_from_layer_type_list(layer_type_list):
        maps = orig_maps(layer_type_list)
        att = dc.Symbols.ATTENTION
        dsa = getattr(dc.Symbols, "DS_ATTENTION", None)
        if not maps.get(att) and not (dsa is not None and maps.get(dsa)):
            maps[att] = {_PHANTOM: 0}
        return maps

    dc.get_layer_maps_from_layer_type_list = get_layer_maps_from_layer_type_list
    cls = dc.DynamicInferenceContext
    orig_alloc = cls._allocate_memory_buffer

    def _allocate_memory_buffer(self):
        if _PHANTOM in getattr(self, "layer_map", {}):
            n = self.num_attention_layers
            self.num_attention_layers = 0  # no K / V storage on an attention-free stage
            try:
                return orig_alloc(self)
            finally:
                self.num_attention_layers = n
        return orig_alloc(self)

    cls._allocate_memory_buffer = _allocate_memory_buffer
    dc._ts_attention_free = True
