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
exact.

``install()`` routes the stage input through the graph: on the graphed path of
a non-first stage the received tensor is passed as ``decoder_input`` (so the
manager copies it into the graph's static input on replay), and the decoder
reads the ``hidden_states`` it is given instead of its ``input_tensor``.
"""

from __future__ import annotations

from typing import Any

_INSTALLED = False


def install() -> None:
    """Patch ``GPTModel`` / ``TransformerBlock`` so block-scope graphs read the stage input."""
    global _INSTALLED
    if _INSTALLED:
        return
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.transformer_block import TransformerBlock
    from megatron.core.utils import WrappedTensor

    orig_call = GPTModel.__call__
    orig_block_forward = TransformerBlock.forward

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
        # Inside the graphed GPTModel forward: hidden_states is the graph's static input.
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

    GPTModel.__call__ = __call__
    TransformerBlock.forward = block_forward
    _INSTALLED = True
