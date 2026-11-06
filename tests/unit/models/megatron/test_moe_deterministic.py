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


@pytest.mark.parametrize("device", DEVICES)
def test_segment_sum_static_topk_matches_host_width(device):
    # Dropless top-k: every token has exactly k copies; the static width (no host
    # read) must give the same bits as the width read from the counts.
    torch.manual_seed(0)
    n, k, h = 37, 6, 5
    idx = torch.stack([torch.randperm(32)[:k] for _ in range(n)])  # expert ids per token
    routing = torch.zeros(n, 32, dtype=torch.bool)
    routing[torch.arange(n).unsqueeze(1), idx] = True
    tok = routing.T.contiguous().nonzero()[:, 1].to(device)  # expert-major gather index
    g = torch.randn(tok.numel(), h, device=device)
    a = segment_sum_rows(g, tok, n)
    b = segment_sum_rows(g, tok, n, k)
    assert torch.equal(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_segment_sum_sparse_padding_rows(device):
    # Cross-fade sparse dispatch: the static T x k argsort permute keeps rows past the real count (padding: token ids
    # of False entries, so a token can appear more than k times there). With ``live`` they are left out; the real
    # copies are summed in position order, bitwise equal to the fixed-width sum whose extra slots carry zero gradients.
    torch.manual_seed(1)
    n, e, k, h = 23, 16, 6, 4
    full = torch.stack([torch.randperm(e)[:k] for _ in range(n)])
    routing_full = torch.zeros(n, e, dtype=torch.bool)
    routing_full[torch.arange(n).unsqueeze(1), full] = True
    drop = torch.rand(n, e) < 0.3
    routing = routing_full & ~drop  # 0..k real slots per token
    flat = routing.T.contiguous().reshape(-1).argsort(descending=True, stable=True)[: n * k]
    idx = (flat % n).to(device)
    live = routing.sum().to(device)
    g = torch.randn(n * k, h, device=device)
    g[int(live):] = float("nan")  # padding rows: never summed
    out = segment_sum_rows(g, idx, n, k, live)
    ref = torch.zeros(n, h, dtype=torch.float32, device=device)
    for j in range(int(live)):  # left to right by position, fp32
        ref[idx[j]] = ref[idx[j]] + g[j].float()
    assert torch.equal(out, ref.to(g.dtype))
    # fixed width: same tokens, the dropped slots present with zero gradient -> same bits
    flat_f = routing_full.T.contiguous().reshape(-1).argsort(descending=True, stable=True)[: n * k]
    idx_f = (flat_f % n).to(device)
    real = routing.T.contiguous().reshape(-1)[flat_f].to(device)
    g_f = torch.zeros(n * k, h, device=device)
    # the j-th real row of the sparse permutation is the j-th real row of the fixed one (both expert-major, stable)
    g_f[real] = g[: int(live)]
    assert torch.equal(segment_sum_rows(g_f, idx_f, n, k), out)
    # the padding rows really repeat tokens more than k times (non-vacuous)
    cnt = torch.bincount(idx.cpu(), minlength=n)
    assert int(cnt.max()) > k or int(live) == n * k
