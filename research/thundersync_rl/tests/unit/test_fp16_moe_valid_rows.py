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
"""Engine expert activations skip CUDA-graph padding rows and stay bitwise the
training (eager) values on the rows they compute."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("rows,valid,f", [(256, 0, 1856), (256, 37, 1856), (512, 512, 1856), (64, 13, 3000)])
def test_relu2_valid_rows_bitwise(rows, valid, f):
    from nemo_rl.models.megatron import fp16_moe

    g = torch.Generator(device="cuda").manual_seed(rows + valid)
    h = torch.randn(rows, f, device="cuda", generator=g) * 40  # some squares overflow fp16
    h[0, :6] = torch.tensor([-0.0, 0.0, 1e-30, -1e-30, 300.0, -3.0])
    p = torch.rand(rows, device="cuda", generator=g)
    eager = fp16_moe._relu2_eager(h, p).to(torch.float16)
    n = torch.tensor([valid], dtype=torch.int32, device="cuda")
    out = fp16_moe.relu2_probs_fwd(h, p, torch.float16, valid_rows=n)
    assert torch.equal(out[:valid].view(torch.int16), eager[:valid].view(torch.int16))
    # The training path (no valid_rows) is the eager ops.
    assert torch.equal(fp16_moe.relu2_probs_fwd(h, p, torch.float32), fp16_moe._relu2_eager(h, p))


@pytest.mark.parametrize("valid", [0, 5, 128])
def test_swiglu_valid_rows_match_full(valid):
    from nemo_rl.models.megatron import fp16_moe

    g = torch.Generator(device="cuda").manual_seed(valid)
    h = torch.randn(128, 2 * 768, device="cuda", generator=g) * 4
    p = torch.rand(128, device="cuda", generator=g)
    full = fp16_moe.swiglu_probs_fwd(h, p, torch.float16)
    n = torch.tensor([valid], dtype=torch.int64, device="cuda")
    part = fp16_moe.swiglu_probs_fwd(h, p, torch.float16, valid_rows=n)
    assert torch.equal(part[:valid].view(torch.int16), full[:valid].view(torch.int16))


def test_relu2_valid_rows_under_cuda_graph():
    from nemo_rl.models.megatron import fp16_moe

    h = torch.randn(1024, 1856, device="cuda")
    p = torch.rand(1024, device="cuda")
    n = torch.tensor([100], dtype=torch.int32, device="cuda")
    fp16_moe.relu2_probs_fwd(h, p, torch.float16, valid_rows=n)  # compile outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fp16_moe.relu2_probs_fwd(h, p, torch.float16, valid_rows=n)
    for valid in (3, 700):
        n.fill_(valid)
        graph.replay()
        eager = fp16_moe._relu2_eager(h, p).to(torch.float16)
        assert torch.equal(out[:valid].view(torch.int16), eager[:valid].view(torch.int16))
