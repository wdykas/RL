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
"""fp16 range monitor counted inside the fp16 cast: same counts, same fp16 bits as the separate pass."""

import pytest
import torch

from nemo_rl.models.megatron import fp16_gemm as G
from nemo_rl.models.megatron import fp16_moe as fm

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture
def monitor():
    G.range_monitor(True)
    G.range_take()
    yield
    G.range_take()
    G.range_monitor(False)
    G._RANGE["fused"] = True


def _hard_values(n, dtype, dev, seed=0):
    """Random values spanning fp16's range edges, plus inf / nan / signed zero / subnormals."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, generator=g) * 3e4
    edges = torch.tensor(
        [65504.0, 65505.0, 65519.0, 65519.99, 65520.0, 65536.0, 1e9, 6e-8, 3e-8, 1e-40, 0.0, -0.0,
         float("inf"), float("-inf"), float("nan"), -65504.0, -65520.0, 2049.0, 2051.0]
    )
    idx = torch.randint(0, n, (len(edges) * 4,), generator=g)
    x[idx] = edges.repeat(4) * torch.where(torch.rand(len(idx), generator=g) < 0.5, 1.0, -1.0)
    return x.to(dtype).to(dev)


def _unfused(x, site=None):
    G._RANGE["fused"] = False
    try:
        G.range_take()
        y = G.cast_fp16_counted(x, site)
        return y, G.range_take()
    finally:
        G._RANGE["fused"] = True


def _fused(x, site=None):
    G.range_take()
    y = G.cast_fp16_counted(x, site)
    return y, G.range_take()


def _bits(y):
    return y.contiguous().view(torch.int16)


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1,), (4095,), (4097,), (3001, 7), (64, 3, 1856), (2, 1 << 20)])
def test_fused_cast_counts_and_bits_equal_reference(monitor, dtype, shape):
    n = 1
    for d in shape:
        n *= d
    x = _hard_values(n, dtype, "cuda").view(shape)
    assert G.range_fusable(x)
    y_ref, c_ref = _unfused(x)
    y, c = _fused(x)
    assert c == c_ref
    assert c["checked"] == float(n)
    ref = int(torch.count_nonzero(x.float().abs() > G.FP16_MAX))
    assert c["overflow"] == float(ref)
    assert y.dtype == torch.float16 and y.shape == x.shape and y.is_contiguous()
    assert torch.equal(_bits(y), _bits(x.to(torch.float16)))
    assert torch.equal(_bits(y), _bits(y_ref))


@cuda
def test_fused_cast_every_fp32_bit_pattern_near_the_rounding_edges(monitor):
    # Every fp32 bit pattern in [65000, 66000] and [0, 2^-13] and their negatives: fp16 rounding and the
    # overflow threshold at the exact boundaries.
    lo = torch.tensor([65000.0, 0.0]).view(torch.int32)
    hi = torch.tensor([66000.0, 2.0**-13]).view(torch.int32)
    parts = [torch.arange(int(a), int(b) + 1, dtype=torch.int32) for a, b in zip(lo, hi)]
    x = torch.cat(parts).view(torch.float32)
    x = torch.cat([x, -x]).cuda()
    y_ref, c_ref = _unfused(x)
    y, c = _fused(x)
    assert c == c_ref and torch.equal(_bits(y), _bits(y_ref))


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_non_contiguous_takes_the_unfused_path(monitor, dtype):
    x = _hard_values(3001 * 7, dtype, "cuda").view(3001, 7).t()
    assert not G.range_fusable(x)
    y_ref, c_ref = _unfused(x)
    y, c = _fused(x)
    assert c == c_ref and torch.equal(_bits(y), _bits(y_ref))
    assert torch.equal(_bits(y), _bits(x.to(torch.float16)))


@cuda
def test_fp16_input_and_monitor_off(monitor):
    x16 = torch.randn(100, device="cuda", dtype=torch.float16)
    y, c = _fused(x16)
    assert y.data_ptr() == x16.data_ptr() and c == {"overflow": 0.0, "checked": 0.0}
    G.range_monitor(False)
    x = torch.full((10,), 1e6, device="cuda")
    assert not G.range_fusable(x)
    y, c = _fused(x)
    assert c["overflow"] == 0.0 and torch.equal(_bits(y), _bits(x.to(torch.float16)))


@cuda
def test_graph_capture_casts_without_counting(monitor):
    x = _hard_values(8192, torch.float32, "cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        G.cast_fp16_counted(x)  # warm-up (Triton compile) outside capture
    torch.cuda.current_stream().wait_stream(s)
    G.range_take()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = G.cast_fp16_counted(x)
    g.replay()
    torch.cuda.synchronize()
    assert G.range_take() == {"overflow": 0.0, "checked": 0.0}
    assert torch.equal(_bits(y), _bits(x.to(torch.float16)))


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("live_rows", [0, 1, 777, 1000])
def test_live_rows_cast_counts_and_bits(monitor, dtype, live_rows):
    rows, h = 1000, 1856
    x = _hard_values(rows * h, dtype, "cuda").view(rows, h)
    x[live_rows:] = 0  # a zero tail: the unfused count over every row equals the live-row count
    live = torch.tensor(live_rows, device="cuda")
    G._RANGE["fused"] = False
    G.range_take()
    ref = fm._cast_live_fp16_counted(x, live)
    c_ref = G.range_take()
    G._RANGE["fused"] = True
    y = fm._cast_live_fp16_counted(x, live)
    c = G.range_take()
    assert c == c_ref
    assert c["overflow"] == float(torch.count_nonzero(x[:live_rows].float().abs() > G.FP16_MAX))
    assert torch.equal(_bits(y[:live_rows]), _bits(ref[:live_rows]))
    assert torch.equal(_bits(y[:live_rows]), _bits(x[:live_rows].to(torch.float16)))
    # Rows past live (uninitialized in a capacity buffer) are neither cast nor counted by the fused pass.
    x[live_rows:] = 1e6
    G.range_take()
    fm._cast_live_fp16_counted(x, live)
    assert G.range_take() == c


def _experts(e, n, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(n, k, generator=g) * 0.05).to(torch.bfloat16).cuda() for _ in range(e)]


@cuda
@pytest.mark.parametrize("device_counts", [False, True])
def test_grouped_linear_counts_its_input_in_the_cast(monitor, device_counts):
    splits = [37, 0, 129, 64]
    k, n = 256, 128
    x = _hard_values(sum(splits) * k, torch.float32, "cuda").view(-1, k)
    x = torch.where(x.isfinite(), x.clamp(-7e4, 7e4), x).nan_to_num(0.0)  # finite GEMM outputs
    ws = _experts(len(splits), n, k)
    m = torch.tensor(splits, device="cuda") if device_counts else splits
    expect = float(torch.count_nonzero(x.abs() > G.FP16_MAX))

    def run(fused):
        G._RANGE["fused"] = fused
        G.range_take()
        if fused:
            fm._HOOKS["count_in"] = True
            try:
                y = fm.grouped_linear(x, m, ws)
            finally:
                fm._HOOKS.pop("count_in", None)
        else:  # the previous call sequence: count, then the GEMM's own cast
            G.range_count(x)
            y = fm.grouped_linear(x, m, ws)
        return y, G.range_take()

    y_ref, c_ref = run(False)
    y, c = run(True)
    G._RANGE["fused"] = True
    assert c == c_ref and c["overflow"] == expect
    live = sum(splits)
    assert torch.equal(y[:live].view(torch.int32), y_ref[:live].view(torch.int32))
    assert "count_in" not in fm._HOOKS


@cuda
def test_fp16_linear_forward_bits_and_counts(monitor):
    x = _hard_values(96 * 512, torch.bfloat16, "cuda").view(96, 512)
    x = torch.where(x.float().abs() > 1e5, torch.zeros_like(x), x)
    w = (torch.randn(384, 512) * 0.02).to(torch.bfloat16).cuda()

    def run(fused):
        G._RANGE["fused"] = fused
        G.range_take()
        with torch.no_grad():
            y = G.fp16_linear(x, w)
        return y, G.range_take()

    y_ref, c_ref = run(False)
    y, c = run(True)
    assert c == c_ref and c["overflow"] > 0
    assert torch.equal(y.view(torch.int32), y_ref.view(torch.int32))
