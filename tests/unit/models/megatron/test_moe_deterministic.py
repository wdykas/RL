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

from nemo_rl.models.megatron.moe_deterministic import GatherRows, segment_sum_rows

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_gather_rows_forward_and_backward(device):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(50, 16, generator=g).to(device).requires_grad_()
    idx = torch.randint(0, 50, (300,), generator=g).to(device)  # repeated rows, some absent
    y = GatherRows.apply(x, idx)
    assert torch.equal(y, x.detach().index_select(0, idx))  # same forward bits
    go = torch.randn(300, 16, generator=g).to(device)
    y.backward(go)
    ref = torch.zeros(50, 16, dtype=torch.float64, device=device).index_add_(0, idx, go.double())
    assert torch.allclose(x.grad.double(), ref, atol=1e-5)
    # Reproducible: the same inputs give the same bits.
    again = segment_sum_rows(go, idx, 50)
    assert torch.equal(again, x.grad)


def test_segment_sum_order_is_by_position():
    go = torch.tensor([[1e8], [1.0], [-1e8]], dtype=torch.float32)
    idx = torch.tensor([0, 0, 0])
    # ((1e8 + 1) - 1e8) in fp32, left to right by position
    assert segment_sum_rows(go, idx, 1).item() == (torch.tensor(1e8) + 1.0 - 1e8).item()
