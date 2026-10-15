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
"""GPU test: batch-invariant Mamba training == the Megatron engine's prefill and decode.

nemo_rl/models/megatron/mamba_ssd_parity.py. A MambaMixer's training forward (with the
patch installed) against the same mixer's dynamic-engine prefill of a varlen batch and
batch-invariant decode of the rest, bitwise, on bf16 inputs and on the fp32 inputs the
fp16-GEMM recipe produces; and the same check failing without the patch when the engine's
cumsum runs another tiling than mamba_ssm's.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

SEQLEN = 300
PROMPT_LENS = (77, 128, 200, 256)


@pytest.fixture
def mixer_factory(tmp_path):
    from megatron.core import parallel_state
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.mamba_mixer import MambaMixer
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig

    created = False
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=0,
            world_size=1,
            init_method=f"file://{tmp_path / 'pg'}",
        )
        created = True
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)

    def make(dtype):
        config = TransformerConfig(
            hidden_size=256,
            num_layers=1,
            num_attention_heads=1,
            use_cpu_initialization=True,
        )
        mixer = MambaMixer(
            config,
            hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
            config.hidden_size,
            layer_number=1,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["tp", "cp"]
            ),
        )
        # The mixer only reads the flag; the config checks it adds concern other layers.
        config.batch_invariant_mode = True
        return mixer.to(device="cuda", dtype=dtype)

    yield make
    parallel_state.destroy_model_parallel()
    if created:
        dist.destroy_process_group()


def _train_and_generate(mixer, zxBCdt):
    """(training output, engine prefill + decode output), both (SEQLEN, batch, d_inner)."""
    device, dtype = zxBCdt.device, zxBCdt.dtype
    batch = len(PROMPT_LENS)
    y_train = mixer._ssm_training(zxBCdt)

    conv_state = torch.zeros(
        batch, mixer.conv1d_weight.shape[0], mixer.d_conv, device=device, dtype=dtype
    )
    ssm_state = torch.zeros(
        batch, mixer.nheads_local_tp, mixer.headdim, mixer.d_state, device=device
    )
    lens = torch.tensor(PROMPT_LENS, device=device)
    cu_seqlens = torch.cat([lens.new_zeros(1), lens.cumsum(0)]).to(torch.int32)
    slots = torch.arange(batch, dtype=torch.int32, device=device)
    context = SimpleNamespace(
        mamba_slot_allocator=None,
        mamba_metadata=SimpleNamespace(
            seq_idx=torch.repeat_interleave(slots, lens).unsqueeze(0),
            cu_seqlens=cu_seqlens,
            batch_indices_prefill=slots,
            intermediate_chunk_indices=None,
            intermediate_abs_positions=None,
            intermediate_real_count=None,
            cu_chunk_seqlens=None,
            last_chunk_indices=None,
            seq_idx_for_varlen=None,
            conv_seq_idx=None,
            conv_seq_start=None,
        ),
    )
    prompts = torch.cat([zxBCdt[:n, i] for i, n in enumerate(PROMPT_LENS)]).unsqueeze(1)
    y_prefill = mixer.ssm_prefill(prompts, conv_state, ssm_state, context)
    y_gen = torch.zeros_like(y_train)
    for i, n in enumerate(PROMPT_LENS):
        y_gen[:n, i] = y_prefill[cu_seqlens[i] : cu_seqlens[i + 1], 0]
    for step in range(SEQLEN - min(PROMPT_LENS)):
        live = [i for i, n in enumerate(PROMPT_LENS) if n + step < SEQLEN]
        tokens = torch.stack(
            [zxBCdt[PROMPT_LENS[i] + step, i] for i in live]
        ).unsqueeze(1)
        y_step = mixer.ssm_decode(
            tokens,
            conv_state,
            ssm_state,
            batch_indices=torch.tensor(live, dtype=torch.int32, device=device),
        )
        for row, i in enumerate(live):
            y_gen[PROMPT_LENS[i] + step, i] = y_step[row, 0]
    return y_train, y_gen


def _inputs(mixer, dtype):
    generator = torch.Generator(device="cuda").manual_seed(0)
    proj_dim = mixer.in_proj.weight.shape[0]
    zxBCdt = torch.randn(
        SEQLEN, len(PROMPT_LENS), proj_dim, device="cuda", generator=generator
    )
    return zxBCdt.to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_training_matches_engine_prefill_and_decode(mixer_factory, dtype):
    from nemo_rl.models.megatron import mamba_ssd_parity

    mamba_ssd_parity.install()
    mixer = mixer_factory(dtype)
    zxBCdt = _inputs(mixer, dtype).requires_grad_()
    y_train, y_gen = _train_and_generate(mixer, zxBCdt)
    assert torch.equal(y_train, y_gen)
    # The backward (mamba_ssm's) runs and reaches the input and the SSM parameters.
    y_train.float().pow(2).sum().backward()
    assert zxBCdt.grad is not None and torch.isfinite(zxBCdt.grad).all()
    assert mixer.A_log.grad is not None and mixer.D.grad is not None


def test_mamba_ssm_training_on_another_tiling_is_caught(mixer_factory):
    """Control: the unpatched training forward (mamba_ssm) with a cumsum head tiling the
    engine does not use differs from the engine, so the check above is not vacuous."""
    import triton
    from mamba_ssm.ops.triton import ssd_chunk_state as mamba_ssm_chunk_state

    from nemo_rl.models.megatron import mamba_ssd_parity

    mamba_ssd_parity.install()
    mixer = mixer_factory(torch.bfloat16)
    original = mamba_ssd_parity._ORIGINAL_SSM_TRAINING[0]
    kernel = mamba_ssm_chunk_state._chunk_cumsum_fwd_kernel
    saved = kernel.configs
    kernel.configs = [triton.Config({"BLOCK_SIZE_H": 1})]
    kernel.cache.clear()
    try:
        zxBCdt = _inputs(mixer, torch.bfloat16)
        _, y_gen = _train_and_generate(mixer, zxBCdt)
        y_old = original(mixer, zxBCdt)
    finally:
        kernel.configs = saved
        kernel.cache.clear()
    assert not torch.equal(y_old, y_gen)


@pytest.mark.parametrize("activation", ["silu", None])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_conv_weight_grads_are_reproducible(activation, dtype):
    """The training conv: kernel forward bits, close gradients, bitwise-repeatable weight grads."""
    from causal_conv1d import causal_conv1d_fn

    from nemo_rl.models.megatron.mamba_ssd_parity import _ConvFixedOrderGradFn

    torch.manual_seed(0)
    rows, channels, length, width = 8, 512, 1024, 4
    x0 = torch.randn(rows, channels, length, device="cuda", dtype=dtype)
    w0 = (0.3 * torch.randn(channels, width, device="cuda")).to(dtype)
    b0 = (0.1 * torch.randn(channels, device="cuda")).to(dtype)
    dout = torch.randn(rows, channels, length, device="cuda", dtype=dtype)

    def run(fn):
        x, w, b = (t.clone().requires_grad_() for t in (x0, w0, b0))
        y = fn(x, w, b)
        y.backward(dout)
        return y.detach(), x.grad, w.grad, b.grad

    ref = run(lambda x, w, b: causal_conv1d_fn(x=x, weight=w, bias=b, activation=activation))
    got = run(lambda x, w, b: _ConvFixedOrderGradFn.apply(x, w, b, activation))
    assert torch.equal(got[0], ref[0])  # forward: the kernel's bits
    assert torch.equal(got[1], ref[1])  # input gradient: the kernel's
    for g, r in zip(got[2:], ref[2:]):  # weight / bias: the same sums, rounded per row
        scale = float(r.float().abs().max())
        torch.testing.assert_close(g.float(), r.float(), rtol=0, atol=2e-3 * scale)
    for _ in range(3):
        again = run(lambda x, w, b: _ConvFixedOrderGradFn.apply(x, w, b, activation))
        assert all(torch.equal(a, g) for a, g in zip(again, got))
