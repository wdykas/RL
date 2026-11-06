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
"""GPU test: after a training forward + backward of a recomputed MoE layer, memory returns to its pre-call level.

nemo_rl/models/megatron/moe_dispatcher_refs.py. A Megatron MoE layer (AlltoAll dispatcher, EP 1) with the MoE
activation recompute: the backward re-runs the forward under autograd, and the dispatcher's kept ``probs`` held the
recompute's graph (its detached input and that input's gradient, two [T, d] fp32 tensors) until the next forward.
With the patch the call leaves nothing behind; without it (negative control) the two tensors stay.
"""

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

T, D = 4096, 1024


@pytest.fixture
def moe_layer(tmp_path):
    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.spec_utils import get_submodules

    created = False
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
        created = True
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        hidden_size=D,
        ffn_hidden_size=2 * D,
        moe_ffn_hidden_size=D,
        num_layers=1,
        num_attention_heads=8,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_token_dispatcher_type="alltoall",
        moe_router_dtype="fp32",
        recompute_granularity="selective",
        recompute_modules=["moe"],
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        add_bias_linear=False,
        bias_activation_fusion=False,
        gated_linear_unit=False,
    )
    submodules = get_submodules(get_gpt_layer_local_submodules(num_experts=8, moe_grouped_gemm=False).mlp)
    layer = MoELayer(config, submodules).cuda().train()
    yield layer
    parallel_state.destroy_model_parallel()
    if created:
        dist.destroy_process_group()


def _left_after_call(layer) -> int:
    """Bytes still allocated after forward + backward, beyond the parameters' gradients."""
    x = torch.randn(T, 1, D, device="cuda", requires_grad=True)
    layer.zero_grad(set_to_none=False)
    out, _ = layer(x)
    out.sum().backward()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    del out
    x.grad = None
    return base


def test_recomputed_moe_leaves_no_graph_after_the_call(moe_layer, monkeypatch):
    from megatron.core.transformer.moe import token_dispatcher as td

    from nemo_rl.models.megatron import moe_dispatcher_refs

    assert moe_layer.moe_layer_recompute
    cls = td.MoEAlltoAllTokenDispatcher
    patched = getattr(cls.combine_postprocess, "__wrapped__", None) is not None
    if not patched:
        moe_dispatcher_refs.install()
    plain = cls.combine_postprocess.__wrapped__

    def measure():
        _left_after_call(moe_layer)  # warm-up: parameter grads exist
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        _left_after_call(moe_layer)
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() - before, moe_layer.token_dispatcher.probs

    # Negative control: the stock combine keeps probs (with the recompute's graph) after the call.
    monkeypatch.setattr(cls, "combine_postprocess", plain)
    moe_layer.token_dispatcher.probs = None
    _left_after_call(moe_layer)
    torch.cuda.synchronize()
    stock_floor = torch.cuda.memory_allocated()
    probs = moe_layer.token_dispatcher.probs
    assert probs is not None and probs.grad_fn is not None
    moe_layer.token_dispatcher.probs = None
    del probs
    torch.cuda.synchronize()
    freed = stock_floor - torch.cuda.memory_allocated()
    # The recompute's detached input and its grad: 2 x T x D fp32.
    assert freed >= 2 * T * D * 4

    # Fix: nothing kept.
    monkeypatch.undo()
    delta, probs = measure()
    assert probs is None
    assert delta == 0
