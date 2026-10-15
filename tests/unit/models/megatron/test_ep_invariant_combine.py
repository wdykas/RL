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
"""EP-invariant MoE combine (ep_invariant_combine.py): same bits at EP 1 / 2 / 4 / 8.

EP is emulated on one GPU: the engine runs ``engine_experts`` once per EP rank on that
rank's expert slice (every gathered token) and the cross-rank sum of the limb partials
is formed in several orders; the learner sums the AlltoAll rows (cuBLAS expert GEMMs)
through ``fused_unpermute``. The real NVLS limb reduce-scatter is checked on GPUs by
thundersync_work/research/ep_invariant/nvls_limb_rs.py (torchrun).
"""

import os

os.environ.setdefault("CUBLASLT_WORKSPACE_SIZE", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":0:0")

from fractions import Fraction  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _bits(t):
    return t.contiguous().view(torch.int32)


@pytest.fixture
def eic():
    from nemo_rl.models.megatron import ep_invariant_combine as m

    m.set_enabled(True)
    yield m
    m.set_enabled(False)


def _fp16_products(n, h, gen, spread=20):
    """fp32 values that are multiples of 2^-48 with a wide exponent range (like fp16 GEMM outputs)."""
    a = (torch.randn(n, h, device="cuda", generator=gen) * 4).to(torch.float16).float()
    b = torch.exp2(torch.randint(-spread, 15, (n, h), device="cuda", generator=gen).float())
    b = b.to(torch.float16).float()  # powers of two in fp16 range (subnormals included)
    return a * b  # exact in fp32: fp16 x fp16


def test_limbs_are_exact_and_correctly_rounded(eic):
    gen = torch.Generator(device="cuda").manual_seed(0)
    k, h = 6, 512
    v = _fp16_products(k, h, gen)
    v[0, :8] = torch.tensor([2.0**40, -(2.0**40), 3.0, -0.0, 0.0, 65504.0**2, 2.0**-48, -(2.0**-48)])
    v[1, :8] = torch.tensor([-(2.0**40), 2.0**40 + 1.0, 2.0**-40, -0.0, -0.0, 1.0, 2.0**-48, -(2.0**-47)])
    v[:, 3] = -0.0  # all -0.0
    v[:, 4] = 0.0
    v[0, 4] = -0.0  # mixed signed zeros
    limbs = eic.to_limbs(v)
    hi, lo = limbs[:, :h], limbs[:, h:]
    # Each value is exactly hi + lo * 2^-48 with integer limbs.
    assert torch.equal(hi, hi.floor()) and torch.equal(lo, lo.round())
    assert torch.equal((hi + lo * 2.0**-48), v.double())
    out = eic.finalize_limbs(limbs.sum(0, keepdim=True))[0]
    vals = v.double().cpu().tolist()
    for j in range(h):  # exact rational sum, correctly rounded to fp32 (ties to even via fp64 exact)
        exact = sum(Fraction(vals[i][j]) for i in range(k))
        ref = _round_fp32(exact)
        assert out[j].item() == ref or (ref == 0 and out[j].item() == 0), (j, out[j].item(), ref)
    # Exact zero sums are +0.0 (as an fp64 sum started at +0.0).
    assert not torch.signbit(out[3]) and not torch.signbit(out[4])


def _round_fp32(x: Fraction) -> float:
    """Correctly rounded (nearest-even) fp32 value of an exact rational."""
    if x == 0:
        return 0.0
    import struct

    f = float(x)  # nearest fp64
    c = struct.unpack("f", struct.pack("f", f))[0]  # fp64 -> fp32 nearest (double rounding check below)
    # Fix double rounding: pick the nearest of the fp32 neighbours of c by exact comparison.
    import numpy as np

    cands = [np.nextafter(np.float32(c), np.float32(-np.inf)), np.float32(c), np.nextafter(np.float32(c), np.float32(np.inf))]
    best = min(cands, key=lambda y: (abs(Fraction(float(y)) - x), int(np.float32(y).view(np.int32)) & 1))
    return float(best)


def test_any_grouping_and_order_gives_same_bits(eic):
    gen = torch.Generator(device="cuda").manual_seed(1)
    k, t, h = 8, 64, 1024
    v = _fp16_products(k * t, h, gen).view(t, k, h)
    v[::5, 1] = -v[::5, 0]  # exact cancellations
    limbs = eic.to_limbs(v.reshape(-1, h)).view(t, k, 2 * h)
    ref = eic.finalize_limbs(limbs.sum(1))
    for trial in range(6):
        perm = torch.randperm(k, generator=torch.Generator().manual_seed(trial))
        cut = 1 + trial % (k - 1)
        g1 = limbs[:, perm[:cut]].sum(1)
        g2 = limbs[:, perm[cut:]].sum(1)
        for acc in (g1 + g2, g2 + g1):
            assert torch.equal(_bits(eic.finalize_limbs(acc)), _bits(ref))


def test_non_finite_rows(eic):
    v = torch.tensor(
        [[float("inf"), float("inf"), float("nan"), 1.0], [1.0, float("-inf"), 2.0, 3.0]],
        device="cuda",
    )
    out = eic.finalize_limbs(eic.to_limbs(v).sum(0, keepdim=True))[0].tolist()
    assert out[0] == float("inf") and out[1] != out[1] and out[2] != out[2] and out[3] == 4.0


def test_check_exact_range(eic):
    assert eic.check_exact_range(6, 1856, 0) == []  # Nemotron-3-Nano
    assert eic.check_exact_range(8, 2048, 3) == []  # 1T-class (k=8)
    assert eic.check_exact_range(22, 2688, 0) == []
    assert eic.check_exact_range(64, 1856, 0)  # lo sums would exceed 2^53
    assert eic.check_exact_range(32, 1 << 16, 8)  # hi sums could round


def _moe_case(e, topk, h, f, t, gen, activation):
    fn = 2 * f if activation == "swiglu" else f
    w1 = (torch.randn(e, fn, h, device="cuda", generator=gen) * 0.02).to(torch.float16)
    w2 = (torch.randn(e, h, f, device="cuda", generator=gen) * 0.02).to(torch.float16)
    x32 = torch.randn(t, h, device="cuda", generator=gen)
    scores = torch.randn(t, e, device="cuda", generator=gen)
    top = scores.topk(topk, dim=1)
    probs = torch.softmax(top.values, dim=1).float()
    return w1, w2, x32, top.indices.to(torch.int64), probs


def _engine(fm, x16, probs, routing, w1, w2, ep, activation, down_scale, order):
    e = w1.shape[0]
    el = e // ep
    t = x16.shape[0]
    vt = torch.tensor([t], dtype=torch.int32, device="cuda")
    parts = [
        fm.engine_experts(
            x16, probs, routing, vt, w1[r * el : (r + 1) * el].contiguous(),
            w2[r * el : (r + 1) * el].contiguous(), r * el, ep, activation,
            down_scale=down_scale, limbs=True,
        )[:t]
        for r in range(ep)
    ]
    if ep == 1:
        return parts[0]
    from nemo_rl.models.megatron import ep_invariant_combine as eic

    acc = torch.zeros_like(parts[0])
    for r in (order if order is not None else range(ep)):
        acc = acc + parts[r]
    return eic.finalize_limbs(acc)


def _learner(fm, x32, probs, routing, w1, w2, activation, down_scale, ep):
    from megatron.core.transformer.moe.moe_utils import permute

    from nemo_rl.models.megatron import bi_unpermute

    t, e = x32.shape[0], w1.shape[0]
    rm = torch.zeros(t, e, dtype=torch.bool, device="cuda")
    rm.scatter_(1, routing, True)
    pr = torch.zeros(t, e, device="cuda")
    pr.scatter_(1, routing, probs)
    xp, pp, _, inv, _ = permute(
        x32, rm, probs=pr, num_out_tokens=t * routing.shape[1],
        return_batch_invariant_inverse_map=True,
    )
    counts = rm.sum(0).tolist()
    pp = pp.reshape(-1)
    if down_scale:
        pp = pp * 2.0**-down_scale
    out, s = [], 0
    for i, c in enumerate(counts):  # the training path's cuBLAS expert GEMMs
        hx = torch.mm(xp[s : s + c].to(torch.float16), w1[i].t(), out_dtype=torch.float32)
        act = (fm.relu2_probs if activation == "relu2" else fm.swiglu_probs)(hx, pp[s : s + c])  # training ops
        y = torch.mm(act.to(torch.float16), w2[i].t(), out_dtype=torch.float32)
        out.append(y * 2.0**down_scale if down_scale else y)
        s += c
    rows = torch.cat(out).requires_grad_()
    res = bi_unpermute.fused_unpermute(
        rows, torch.Size((t, x32.shape[1])), probs=None, num_experts=e, inverse_map=inv,
        ep_size=ep, cross_rank_fp64=False,
    )
    return res, rows, inv


@pytest.mark.parametrize("activation", ["relu2", "swiglu"])
@pytest.mark.parametrize("down_scale", [0, 2])
def test_engine_and_learner_bitwise_at_every_ep(eic, activation, down_scale):
    from nemo_rl.models.megatron import fp16_moe as fm

    gen = torch.Generator(device="cuda").manual_seed(3 + down_scale)
    e, topk, h, f, t = 64, 6, 1024, 512, 192
    w1, w2, x32, routing, probs = _moe_case(e, topk, h, f, t, gen, activation)
    x16 = x32.to(torch.float16)
    ref = _engine(fm, x16, probs, routing, w1, w2, 1, activation, down_scale, None)
    for ep in (2, 4, 8):
        for order in (None, list(reversed(range(ep)))):
            out = _engine(fm, x16, probs, routing, w1, w2, ep, activation, down_scale, order)
            assert torch.equal(_bits(out), _bits(ref)), (ep, order)
    for ep in (1, 2, 4, 8):  # the learner's ep_size no longer changes anything
        lrn, _, _ = _learner(fm, x32, probs, routing, w1, w2, activation, down_scale, ep)
        assert torch.equal(_bits(lrn.detach()), _bits(ref)), ep


def test_learner_backward_is_token_gradient(eic):
    from nemo_rl.models.megatron import fp16_moe as fm

    gen = torch.Generator(device="cuda").manual_seed(5)
    e, topk, h, f, t = 32, 4, 256, 128, 50
    w1, w2, x32, routing, probs = _moe_case(e, topk, h, f, t, gen, "relu2")
    out, rows, inv = _learner(fm, x32, probs, routing, w1, w2, "relu2", 0, 2)
    g = torch.randn(t, h, device="cuda", generator=gen)
    (dx,) = torch.autograd.grad(out, rows, g)
    ref = torch.zeros_like(rows)
    r = inv[0]
    for j in range(topk):
        ok = r[:, j] >= 0
        ref[r[ok, j]] = g[ok]
    assert torch.equal(dx, ref)


def test_engine_limbs_under_cuda_graph_replay(eic):
    from nemo_rl.models.megatron import fp16_moe as fm

    gen = torch.Generator(device="cuda").manual_seed(9)
    e, topk, h, f, t, ep = 32, 6, 512, 256, 256, 4
    w1, w2, x32, routing, probs = _moe_case(e, topk, h, f, t, gen, "relu2")
    x16 = x32.to(torch.float16)
    el = e // ep
    vt = torch.tensor([16], dtype=torch.int32, device="cuda")
    args = (x16, probs, routing, vt, w1[:el].contiguous(), w2[:el].contiguous(), 0, ep, "relu2")
    fm.engine_experts(*args, tokens_hint=16, limbs=True)  # compile outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fm.engine_experts(*args, tokens_hint=16, limbs=True)
    for valid in (3, 16, 200):
        vt.fill_(valid)
        graph.replay()
        eager = fm.engine_experts(*args, tokens_hint=None, limbs=True)
        assert torch.equal(out[:valid], eager[:valid]), valid
