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
import pytest
import torch

from nemo_rl.models.megatron import inference_pp_cuda_graph as pg

gpt_model = pytest.importorskip("megatron.core.models.gpt.gpt_model")
transformer_block = pytest.importorskip("megatron.core.transformer.transformer_block")
hybrid_model = pytest.importorskip("megatron.core.models.hybrid.hybrid_model")
hybrid_block = pytest.importorskip("megatron.core.models.hybrid.hybrid_block")
PAIRS = {
    "gpt": (gpt_model.GPTModel, transformer_block.TransformerBlock),
    "hybrid": (hybrid_model.HybridModel, hybrid_block.HybridStack),
}


@pytest.fixture(params=sorted(PAIRS))
def patched(request, monkeypatch):
    """Install the fix over stubs of Megatron's block-scope graph: the manager runs
    the model forward once at capture on static copies of the call's tensor
    arguments; afterwards it only copies new argument values into those buffers and
    replays, i.e. returns the captured output tensor, which aliases whatever memory
    the forward read at capture. The stub model forward hands decoder_input to the
    decoder like GPTModel; the stub decoder reads input_tensor on a non-first stage
    like TransformerBlock."""
    static = {}
    Model, Block = PAIRS[request.param]

    def fake_model_call(self, *args, **kwargs):
        if not self._should_call_local_cudagraph(*args, **kwargs):
            return type(self).forward(self, **kwargs)
        x = kwargs.get("decoder_input")
        if x is not None:
            buf = static.setdefault("decoder_input", torch.empty_like(x))
            buf.copy_(x)
            kwargs["decoder_input"] = buf
        if "output" not in static:  # capture
            static["output"] = type(self).forward(self, **kwargs)
        return static["output"].clone()  # replay

    def fake_model_forward(self, decoder_input=None, **kwargs):
        return self.decoder.forward(hidden_states=decoder_input, attention_mask=None)

    def fake_block_forward(self, hidden_states=None, **kwargs):
        return hidden_states if self.pre_process else self.input_tensor

    # Stub every class install() patches, so monkeypatch restores all of them.
    for model_cls, block_cls in PAIRS.values():
        monkeypatch.setattr(model_cls, "__call__", fake_model_call)
        monkeypatch.setattr(model_cls, "forward", fake_model_forward)
        monkeypatch.setattr(
            model_cls,
            "_should_call_local_cudagraph",
            lambda self, *a, **k: self._graphed,
        )
        monkeypatch.setattr(block_cls, "forward", fake_block_forward)
    monkeypatch.setattr(pg, "_INSTALLED", False)
    pg.install()
    return static, Model, Block


def _model(classes, pre_process, graphed):
    Model, Block = classes
    decoder = object.__new__(Block)
    decoder.__dict__.update(pre_process=pre_process, input_tensor=None)
    model = object.__new__(Model)
    model.__dict__.update(pre_process=pre_process, _graphed=graphed)
    model.__dict__["decoder"] = decoder  # bypass nn.Module.__setattr__ (no __init__)
    return model


def test_graphed_non_first_stage_reads_each_received_tensor(patched):
    static, *classes = patched
    model = _model(classes, pre_process=False, graphed=True)
    for step in range(3):
        received = torch.full((4, 1, 8), float(step))
        model.decoder.input_tensor = received  # set_input_tensor(recv_buffer)
        out = model(input_ids=None, position_ids=None, attention_mask=None)
        torch.testing.assert_close(out, received, rtol=0, atol=0)
        assert model.decoder.input_tensor is received  # restored after the call
    assert "decoder_input" in static  # the stage input is a graph input


def test_eager_and_first_stage_paths_unchanged(patched):
    _, *classes = patched
    eager = _model(classes, pre_process=False, graphed=False)
    eager.decoder.input_tensor = torch.ones(2, 1, 8)
    out = eager(input_ids=None, position_ids=None, attention_mask=None)
    assert out is eager.decoder.input_tensor
    first = _model(classes, pre_process=True, graphed=True)
    x = torch.randn(2, 1, 8)
    out = first(input_ids=None, position_ids=None, attention_mask=None, decoder_input=x)
    torch.testing.assert_close(out, x)
