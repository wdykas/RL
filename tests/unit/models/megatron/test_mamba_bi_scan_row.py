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
"""Crossing-row chunk boundary of the batch-invariant decode step: bitwise MCore's chunk state + state passing."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _pin():
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        _pin_mamba_autotuners,
    )

    from nemo_rl.models.megatron import mamba_ssd_parity

    _pin_mamba_autotuners()
    mamba_ssd_parity.pin_engine_kernels()


def _inputs(R, nh, hd, ng, ds, seed, scale):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*s):
        return torch.randn(*s, device="cuda", generator=g) * scale

    return dict(
        x=rnd(R * 128, nh, hd),
        dt=rnd(R * 128, nh) - 2.0,
        B=rnd(R * 128, ng, ds),
        C=rnd(R * 128, ng, ds),
        # Per-head decay from long memory (|A| 1e-4) to fast (|A| 2): the boundary update
        # exp(dA) * S + new is then non-trivial (with |A| ~ 1, exp(dA) underflows to 0 over a chunk).
        A=-torch.exp(torch.empty(nh, device="cuda").uniform_(-9.2, 0.7, generator=g)),
        state=torch.randn(R, nh, hd, ds, device="cuda", generator=g) * 0.1,
    )


# MCore's own pin of the chunk-state tile (4 warps: tcgen05 on Blackwell) instead of the parity table's.
_MCORE_STATE_TILE = ({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, 4, 4)


def compare_boundary(seed, scale):
    """Crossing rows (1 in 5, plus a padding lane): the boundary kernel's states == MCore's tile pair, bits."""
    from megatron.core.ssm.ops.mamba2.ssd_chunk_state import _chunk_cumsum_fwd, _chunk_state_fwd
    from megatron.core.ssm.ops.mamba2.ssd_state_passing import _state_passing_fwd

    from nemo_rl.models.megatron import mamba_bi_scan_row as sr

    R, nh, hd, ng, ds = 256, 16, 64, 2, 128
    d = _inputs(R, nh, hd, ng, ds, seed, scale)
    if seed == 2:
        d["x"][::7] = 0.0
        d["x"][1::11] = -0.0
        d["state"][..., ::5] = 0.0
    g = torch.Generator(device="cuda").manual_seed(100 + seed)
    dt_bias = torch.randn(nh, device="cuda", generator=g) * 0.1
    slots = torch.randperm(R, device="cuda", generator=g).to(torch.int32)
    slots[3] = -1
    pos = torch.arange(R, device="cuda") % 128
    pos = torch.where(torch.arange(R, device="cuda") % 5 == 0, 127, pos)
    rows = torch.where(slots >= 0, pos, -1).to(torch.int32)
    flags = (rows == 127).to(torch.int32)
    starts = slots.clamp_min(0) * 128
    dA, dt = _chunk_cumsum_fwd(d["dt"], d["A"], 128, None, dt_bias=dt_bias, dt_softplus=True,
                               chunk_starts=starts, target_rows=rows)
    ref = d["state"].clone()
    states = _chunk_state_fwd(d["B"], d["x"], dt, dA, None, states_in_fp32=True, chunk_flags=flags,
                              chunk_starts=starts)
    _state_passing_fwd(states.flatten(-2), dA, None, initial_states=ref.flatten(-2), seq_idx=slots,
                       dst_states=ref.flatten(-2), dst_indices=slots, dst_flags=flags)
    new = d["state"].clone()
    sr.chunk_state_pass_rows(d["x"], d["B"], dt, dA, starts, flags, slots, new, sr.pinned_state_config())
    assert torch.isfinite(ref).all()
    assert not torch.equal(ref, d["state"])  # some rows crossed
    assert torch.equal(ref.view(torch.int32), new.view(torch.int32))


@cuda
@pytest.mark.parametrize("seed,scale", [(0, 1.0), (1, 1e-3), (2, 30.0)])
@pytest.mark.parametrize("state_tile", ["parity", "mcore"])
def test_boundary_matches_mcore_chunk_state_and_state_passing(seed, scale, state_tile):
    import triton
    from megatron.core.ssm.ops.mamba2 import ssd_chunk_state

    _pin()
    kernel = ssd_chunk_state._chunk_state_fwd_kernel
    saved = kernel.configs
    if state_tile == "mcore":
        kw, warps, stages = _MCORE_STATE_TILE
        kernel.configs = [triton.Config(kw, num_warps=warps, num_stages=stages)]
        kernel.cache.clear()
    try:
        compare_boundary(seed, scale)
    finally:
        kernel.configs = saved
        kernel.cache.clear()
