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
"""GPU test: activation recompute of the batch-invariant Mamba mixer (mamba_ssd_parity).

With ``install(recompute=True)`` a training ``MambaMixer.forward`` checkpoints
in-projection + conv + SSD scan + gated norm. Checked on a Nemotron-3-Nano-shaped mixer
(hidden 2688, 64 heads x 64, 8 groups, state 128) with fp32 activations (the fp16-GEMM
recipe's dtype): the forward output is bitwise the one without recompute, gradients agree
to fp32 rounding (the scan backward's atomics need not be reproducible), the activation
memory held between forward and backward drops, and the backward's re-run of the forward
adds no fp16 range-monitor counts.
"""

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

ROWS, SEQLEN = 2, 1024


@pytest.fixture
def nano_mixer(tmp_path):
    from megatron.core import parallel_state
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.mamba_mixer import MambaMixer
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig

    from nemo_rl.models.megatron import mamba_ssd_parity

    created = False
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}"
        )
        created = True
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        hidden_size=2688,
        num_layers=1,
        num_attention_heads=1,
        mamba_num_heads=64,
        mamba_head_dim=64,
        mamba_num_groups=8,
        mamba_state_dim=128,
        use_cpu_initialization=True,
    )
    mixer = MambaMixer(
        config,
        hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
        config.hidden_size,
        layer_number=1,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"]),
    )
    config.batch_invariant_mode = True
    mixer = mixer.to(device="cuda", dtype=torch.float32).train()
    yield mixer
    mamba_ssd_parity.install(recompute=False)
    parallel_state.destroy_model_parallel()
    if created:
        dist.destroy_process_group()


def _run(mixer, recompute: bool):
    """(output, input grad, param grads, bytes held between forward and backward)."""
    from nemo_rl.models.megatron import mamba_ssd_parity

    mamba_ssd_parity.install(recompute=recompute)
    mixer.zero_grad(set_to_none=True)
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(SEQLEN, ROWS, mixer.config.hidden_size, device="cuda", generator=gen)
    x.requires_grad_(True)
    dy = torch.randn(SEQLEN, ROWS, mixer.config.hidden_size, device="cuda", generator=gen)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    out, _ = mixer(x)
    torch.cuda.synchronize()
    held = torch.cuda.memory_allocated() - before - out.numel() * out.element_size()
    out.backward(dy)
    torch.cuda.synchronize()
    grads = {n: p.grad.detach().clone() for n, p in mixer.named_parameters() if p.grad is not None}
    return out.detach().clone(), x.grad.detach().clone(), grads, held


def test_recompute_forward_bitwise_grads_close_memory_lower(nano_mixer):
    out0, dx0, g0, held0 = _run(nano_mixer, recompute=False)
    out1, dx1, g1, held1 = _run(nano_mixer, recompute=True)
    # Forward: the same kernels on the same inputs, bitwise.
    assert torch.equal(out0, out1)
    # Gradients: the backward's reductions (scan atomics) need not repeat bit for bit.
    assert set(g0) == set(g1) and len(g0) > 0
    for name, ref in [("input", dx0)] + sorted(g0.items()):
        got = dx1 if name == "input" else g1[name]
        err = (got - ref).abs().max().item()
        scale = ref.abs().max().item()
        assert err <= 1e-5 * max(scale, 1e-30), f"{name}: max |diff| {err:.3e} vs max |ref| {scale:.3e}"
    tokens = ROWS * SEQLEN
    kib0, kib1 = held0 / tokens / 1024, held1 / tokens / 1024
    print(f"[mamba recompute] held per token: off {kib0:.1f} KiB, on {kib1:.1f} KiB")
    # Without recompute the mixer keeps its fp32 conv input, x / z / B / C / dt copies and
    # the scan output (~80 KiB / token at these shapes) plus the in-projection's own saved
    # tensors; with it, only what the out-projection (outside the checkpoint) saves: its
    # input, the gated-norm output (16 KiB / token in fp32, d_inner 4096).
    assert kib0 > 75
    assert kib1 < 17


def test_recompute_reruns_without_range_counts(monkeypatch):
    from torch.utils.checkpoint import checkpoint

    from nemo_rl.models.megatron import fp16_gemm, mamba_ssd_parity

    monkeypatch.setitem(fp16_gemm._RANGE, "on", True)
    seen = []

    def fn(x):
        seen.append(fp16_gemm._RANGE["on"])
        return x.sin() * 2

    x = torch.randn(64, device="cuda", requires_grad=True)
    checkpoint(fn, x, use_reentrant=False, context_fn=mamba_ssd_parity._checkpoint_contexts).sum().backward()
    assert seen == [True, False]  # forward counts, the backward's re-run does not
    assert fp16_gemm._RANGE["on"] is True


def test_no_grad_and_inference_paths_are_not_checkpointed(nano_mixer):
    from nemo_rl.models.megatron import mamba_ssd_parity

    calls = []
    real = torch.utils.checkpoint.checkpoint

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    torch.utils.checkpoint.checkpoint = spy
    try:
        mamba_ssd_parity.install(recompute=True)
        x = torch.randn(256, 1, nano_mixer.config.hidden_size, device="cuda")
        with torch.no_grad():
            y_ng, _ = nano_mixer(x)
        assert calls == []
        y_g, _ = nano_mixer(x.requires_grad_(True))
        assert calls == [1]
        assert torch.equal(y_ng, y_g.detach())
    finally:
        torch.utils.checkpoint.checkpoint = real
