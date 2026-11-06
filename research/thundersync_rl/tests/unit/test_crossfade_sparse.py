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
"""Cross-fade learner sparse dispatch (crossfade_routing_sparse): padding rows of the static T x topk permutation.

CPU: the inverse permutation map with padding rows, device dispatch route tables with padding rows, and the
end-to-end combine (permute -> per-expert op -> batch-invariant unpermute): sparse == fixed width, bitwise, forward and
input gradient."""

import pytest
import torch

from nemo_rl.models.megatron import crossfade_routing as cf


@pytest.fixture
def sparse_on():
    cf.install()
    old = cf.get_state()
    cf._STATE.update(m=0.01, k=2, extra=2, sparse=True)
    yield
    cf._STATE.update(old)


def _maps(T=7, E=8, C=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, E, generator=g)
    b = torch.zeros(E)
    p, idx = cf.crossfade_route(x, b, k=2, extra=C - 2, m=0.05)
    p[:, 2:] = torch.where(torch.rand(T, C - 2, generator=g) < 0.5, torch.zeros_like(p[:, 2:]), p[:, 2:])
    probs = torch.zeros(T, E).scatter(1, idx, p)
    fixed = torch.zeros(T, E, dtype=torch.int).scatter(1, idx, 1).bool()
    return probs, fixed, fixed & (probs != 0), C


def test_inverse_map_with_padding_rows(sparse_on):
    from megatron.core.transformer.moe import moe_utils

    probs, fixed, sparse, C = _maps()
    T = probs.shape[0]
    tok = torch.randn(T, 3)
    _, _, si, inv, _ = moe_utils.permute(tok, sparse, probs=probs, num_out_tokens=T * C,
                                         return_batch_invariant_inverse_map=True)
    rows, experts = inv[0], inv[1]
    flat = sparse.T.reshape(-1).argsort(descending=True, stable=True)[: T * C]
    n_real = int(sparse.sum())
    for t in range(T):
        used = [(int(r), int(e)) for r, e in zip(rows[t], experts[t]) if r >= 0]
        assert sorted(e for _, e in used) == sparse[t].nonzero().flatten().tolist()
        for r, e in used:
            assert r < n_real and int(si[r]) == t and int(flat[r]) // T == e
        assert all(int(r) == -1 for r in rows[t][len(used):])
    # no padding row is referenced
    assert rows[rows >= 0].max() < n_real


def test_route_tables_ignore_padding_rows():
    from nemo_rl.models.megatron.device_dispatch import route_tables

    g = torch.Generator().manual_seed(1)
    C = torch.randint(0, 5, (2, 8), generator=g)
    me = 0
    total = int(C[me].sum())
    ref = route_tables(C, me, total, 64)
    pad = route_tables(C, me, total + 9, 64)
    assert torch.equal(pad["fwd_rank"][:total], ref["fwd_rank"]) and torch.equal(pad["fwd_pos"][:total], ref["fwd_pos"])
    for k in ("back_rank", "back_pos", "live", "tokens_per_expert"):
        assert torch.equal(pad[k], ref[k])


class _Garbage(torch.autograd.Function):
    """Rows at and past ``n``: NaN in the forward output and in the backward gradient (unwritten GEMM rows)."""

    @staticmethod
    def forward(ctx, y, n):
        ctx.n = n
        y = y.clone()
        y[n:] = float("nan")
        return y

    @staticmethod
    def backward(ctx, g):
        g = g.clone()
        g[ctx.n:] = float("nan")
        return g, None


@pytest.mark.parametrize("zero_tail", [True, False])
def test_sparse_combine_equals_fixed_width_bitwise(sparse_on, zero_tail):
    """permute -> grouped-GEMM stand-in (NaN in padding rows, fwd and bwd) -> batch-invariant unpermute:
    sparse == fixed width bitwise (output, input grad, router-logit grad through the rule) with the local-path zero-tail; the control
    without it is corrupted."""
    from megatron.core.transformer.moe import batch_invariant, moe_utils

    from nemo_rl.models.megatron.device_dispatch import _ZeroTailGrad

    T, E, C = 16, 8, 4
    logits0 = 3 * torch.randn(T, E, generator=torch.Generator().manual_seed(3))
    W = torch.randn(E, 5, 5, generator=torch.Generator().manual_seed(4))

    def run(sparse_mode, zt):
        cf._STATE["sparse"] = sparse_mode
        x = torch.randn(T, 5, generator=torch.Generator().manual_seed(5)).requires_grad_(True)
        logits = logits0.clone().requires_grad_(True)
        p, idx = cf.crossfade_route(logits, torch.zeros(E), k=2, extra=C - 2, m=0.05)
        probs = torch.zeros(T, E).scatter(1, idx, p)
        rmap = torch.zeros(T, E, dtype=torch.bool).scatter(1, idx, True)
        if sparse_mode:
            rmap = rmap & (probs != 0)
        px, pp, si, inv, _ = moe_utils.permute(x, rmap, probs=probs, num_out_tokens=T * C,
                                               return_batch_invariant_inverse_map=True)
        n_real = int(rmap.sum())
        if zt:
            px, pp = _ZeroTailGrad.apply(px, pp, rmap.sum())
        flat = rmap.T.reshape(-1).argsort(descending=True, stable=True)[: T * C]
        ex = torch.div(flat, T, rounding_mode="floor")
        y = _Garbage.apply(torch.einsum("rh,rhk->rk", px, W[ex]) * pp.unsqueeze(1), n_real)
        out = batch_invariant.unpermute(y, x.shape, probs=None, num_experts=E, inverse_map=inv)
        gx, gl = torch.autograd.grad((out * torch.arange(1.0, 6.0)).sum(), (x, logits))
        return out.detach(), gx, gl, n_real

    ref = run(False, False)
    got = run(True, zero_tail)
    assert got[3] < ref[3] == T * C  # padding rows actually present
    assert torch.equal(ref[0], got[0])  # forward: padding rows never read
    if zero_tail:
        assert torch.equal(ref[1], got[1]) and torch.equal(ref[2], got[2])
    else:
        assert torch.isnan(got[1]).any() or torch.isnan(got[2]).any()
