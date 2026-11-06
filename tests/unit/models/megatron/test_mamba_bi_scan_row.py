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
"""Target-row Mamba decode chunk scan: bitwise MCore's tile kernel (and so prefill/training)."""

import pytest
import torch

from nemo_rl.models.megatron import mamba_bi_decode as mbd

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


@cuda
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("scale", [1.0, 1e-3, 30.0])
@pytest.mark.parametrize("d_hdim", [False, True])
@pytest.mark.parametrize("state_tile", ["parity", "mcore"])
def test_row_scan_matches_tile_scan_every_position(seed, scale, d_hdim, state_tile):
    """Every target row 0..127, many crossings, shuffled slots, a padding lane, signed zeros.

    Row CB, row scan and the crossing-row boundary kernel vs MCore's tile pipeline: outputs
    and the committed SSM states, bit for bit.
    """
    import triton
    from megatron.core.ssm.ops.mamba2 import batch_invariant_decode as bid
    from megatron.core.ssm.ops.mamba2 import ssd_chunk_state

    from nemo_rl.models.megatron import mamba_bi_scan_row as sr

    _pin()
    assert sr.pinned_config_matches()
    kernel = ssd_chunk_state._chunk_state_fwd_kernel
    saved = kernel.configs
    if state_tile == "mcore":
        kw, warps, stages = _MCORE_STATE_TILE
        kernel.configs = [triton.Config(kw, num_warps=warps, num_stages=stages)]
        kernel.cache.clear()
    try:
        _compare_pipelines(seed, scale, d_hdim)
    finally:
        kernel.configs = saved
        kernel.cache.clear()


def _compare_pipelines(seed, scale, d_hdim):
    from megatron.core.ssm.ops.mamba2 import batch_invariant_decode as bid

    from nemo_rl.models.megatron import mamba_bi_scan_row as sr

    assert sr.pinned_state_config() is not None and sr.pinned_bmm_config() is not None
    R, nh, hd, ng, ds = 256, 16, 64, 2, 128
    d = _inputs(R, nh, hd, ng, ds, seed, scale)
    if seed == 2:
        d["x"][::7] = 0.0
        d["x"][1::11] = -0.0
        d["state"][..., ::5] = 0.0
    g = torch.Generator(device="cuda").manual_seed(100 + seed)
    D = torch.randn((nh, hd) if d_hdim else (nh,), device="cuda", generator=g)
    dt_bias = torch.randn(nh, device="cuda", generator=g) * 0.1
    slots = torch.randperm(R, device="cuda", generator=g).to(torch.int32)
    slots[3] = -1
    pos = torch.arange(R, device="cuda") % 128
    pos = torch.where(torch.arange(R, device="cuda") % 5 == 0, 127, pos)  # ~1 in 5 rows crosses
    rows = torch.where(slots >= 0, pos, -1).to(torch.int32)
    flags = (rows == 127).to(torch.int32)
    starts = slots.clamp_min(0) * 128
    res = []
    for row_kernel in (False, True):
        out = torch.zeros(R, nh, hd, device="cuda")
        st = d["state"].clone()
        kw = dict(chunk_starts=starts, slots=slots, target_rows=rows, chunk_flags=flags, initial_states=st,
                  out=out, D=D, dt_bias=dt_bias, dt_softplus=True)
        if row_kernel:
            sr.decode_rows(d["x"], d["dt"], d["A"], d["B"], d["C"], 128, **kw)
        else:
            bid._mamba_chunk_scan_decode_rows(d["x"], None, d["dt"], d["A"], d["B"], d["C"], 128, **kw)
        res.append((out, st))
    (o_ref, s_ref), (o_row, s_row) = res
    assert torch.isfinite(o_ref).all()
    assert torch.equal(o_ref.view(torch.int32), o_row.view(torch.int32))  # bits, incl. signed zeros
    assert torch.equal(s_ref.view(torch.int32), s_row.view(torch.int32))
    assert (o_row[3] == 0).all()  # padding lane untouched


@cuda
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("slots,prefill", [((0, 3, 1), (5, 127, 130)), ((7, 2, 5, 0, 9), (128, 1, 255, 64, 300))])
@pytest.mark.parametrize("mixed_decay", [False, True])
def test_decode_with_row_scan_is_bitwise_identical(graph, slots, prefill, mixed_decay):
    """300 decode steps (>= 2 chunk crossings per slot) with the row scan vs MCore's tile scan and z path."""
    from tests.unit.models.megatron.test_mamba_bi_decode import _decode

    _pin()
    kw = dict(steps=300, slots=slots, prefill=prefill, graph=graph, mixed_decay=mixed_decay)
    try:
        mbd._ROW_SCAN["on"] = False
        out_tile, st_tile = _decode(True, fused=True, **kw)
        out_ref, st_ref = _decode(False, **kw)  # MCore's z-buffer path
        mbd._ROW_SCAN["on"] = True
        out_row, st_row = _decode(True, fused=True, **kw)
    finally:
        mbd._ROW_SCAN["on"] = True
        mbd._FUSED_PREP["on"] = True
    assert torch.isfinite(out_row[:, : len(slots)]).all()
    assert torch.equal(out_tile, out_row) and torch.equal(st_tile, st_row)
    assert torch.equal(out_ref, out_row) and torch.equal(st_ref, st_row)


@cuda
def test_row_scan_declines_other_pinned_tiles():
    """A chunk-scan config other than the mirrored one keeps MCore's kernel."""
    import triton
    from megatron.core.ssm.ops.mamba2 import ssd_chunk_scan

    from nemo_rl.models.megatron import mamba_bi_scan_row as sr

    _pin()
    kernel = ssd_chunk_scan._chunk_scan_fwd_kernel
    saved = kernel.configs
    try:
        kernel.configs = [triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_warps=2)]
        assert not sr.pinned_config_matches()
    finally:
        kernel.configs = saved
    assert sr.pinned_config_matches()
