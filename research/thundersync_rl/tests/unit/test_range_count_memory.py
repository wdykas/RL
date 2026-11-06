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
"""fp16 range monitor: exact counts with bounded temporaries.

final50b resume: learner rank 0 OOMed inside the MoE recompute allocating count_nonzero's int64 copy of the
range monitor's mask (3.40 GiB for the R_cap x 1856 fp32 routed-expert activation); the old count held 2.25x its
fp32 input in temporaries on top of the training peak.
"""

import pytest
import torch

from nemo_rl.models.megatron import fp16_gemm as G

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _count(x):
    G.range_monitor(True)
    try:
        G.range_take()
        G.range_count(x)
        return G.range_take()
    finally:
        G.range_monitor(False)


@pytest.mark.parametrize("dev", ["cpu", pytest.param("cuda", marks=cuda)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_counts_match_reference(dev, dtype, monkeypatch):
    monkeypatch.setattr(G, "_RANGE_CHUNK_ELEMS", 1000)  # several chunks on the fallback path
    torch.manual_seed(0)
    x = (torch.randn(3001, 7, device=dev) * 3e4).to(dtype)
    x[5, 3] = float("inf")
    x[9, 1] = float("-inf")
    x[11, 2] = float("nan")  # not counted (as the reference)
    ref = int(torch.count_nonzero(x.float().abs() > G.FP16_MAX))
    assert ref > 10
    assert _count(x) == {"overflow": float(ref), "checked": float(x.numel())}
    xt = x.t()  # non-contiguous: chunked along rows
    assert _count(xt) == {"overflow": float(ref), "checked": float(x.numel())}


@cuda
def test_transient_memory_is_bounded():
    x = torch.randn(1 << 15, 1856, device="cuda")  # 232 MiB fp32
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    _count(x)
    torch.cuda.synchronize()
    transient = torch.cuda.max_memory_allocated() - base
    assert transient < 1 << 20, transient  # the old count: ~2.25 x 232 MiB
