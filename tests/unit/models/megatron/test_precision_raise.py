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
"""Targeted precision raises (nemo_rl/models/megatron/precision_raise.py).

Config / presets / tagging on CPU; on GPU, engine == learner bitwise for the hi/lo dense sites
(batch invariance of ``Fp16Linear`` at FP32 input precision, fused vs unfused split) and for the
routed-expert ``routed_fc2_hilo`` / ``combine_payload_fp16`` numerics (learner
``training_experts_forward`` + BI unpermute vs engine ``engine_experts``, EP 1/2/4 emulated).
"""

import os

os.environ.setdefault("CUBLASLT_WORKSPACE_SIZE", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":0:0")

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from nemo_rl.models.megatron import precision_raise as pr  # noqa: E402

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture
def raised(monkeypatch):
    """Set the process's raised sites for one test (no Megatron patching)."""

    def set_(sites=(), c16=False):
        monkeypatch.setitem(pr._STATE, "sites", frozenset(sites))
        monkeypatch.setitem(pr._STATE, "combine16", c16)

    return set_


def test_presets_nest_and_never_raise_routed_fc1():
    p = pr.PRESETS
    assert set(p["32k"]) == {"ssd_scan_x3", "attn_qkv_hilo", "attn_core_qp", "attn_core_kv", "mamba_inout_hilo"}
    assert set(p["32k_qp"]) == set(p["32k"]) - {"attn_core_kv"}
    assert set(p["128k_qp"]) == set(p["128k"]) - {"attn_core_kv"}
    assert pr.resolve({"raise_sites": ["attn_core_fp32"]})[0] == {"attn_core_qp", "attn_core_kv"}
    with pytest.raises(ValueError):
        pr.resolve({"raise_sites": ["attn_core_kv"]})
    assert set(p["128k"]) == set(p["32k"]) | {"shared_expert_hilo"}
    assert set(p["128k_fc2"]) == set(p["128k"]) | {"routed_fc2_hilo"}
    assert all(set(v) <= set(pr.SITES) for v in p.values())
    assert not any("fc1" in s for s in pr.SITES)


def test_resolve_and_canonical():
    assert pr.resolve(None) == (frozenset(), False)
    assert pr.resolve({"preset": "32k"})[0] == frozenset(pr.PRESETS["32k"])
    s, c16 = pr.resolve({"preset": "32k", "raise_sites": ["shared_expert_hilo"], "combine_payload_fp16": True})
    assert s == frozenset(pr.PRESETS["128k"]) and c16
    # Same resolved set, different spelling: equal for the learner == engine check.
    assert pr.canonical({"preset": "128k"}) == pr.canonical(
        {"raise_sites": list(reversed(pr.PRESETS["128k"]))}
    )
    assert pr.canonical(None) == pr.canonical({"preset": "none"})
    with pytest.raises(ValueError):
        pr.resolve({"preset": "64k"})
    with pytest.raises(ValueError):
        pr.resolve({"raise_sites": ["routed_fc1_hilo"]})


def test_validate_config():
    base = {"forward_gemm_precision": "fp16", "precision_raise": {"preset": "32k"}}
    assert pr.validate_config(base) == []
    assert pr.validate_config({"precision_raise": {"preset": "32k"}})  # needs the fp16 forward
    assert pr.validate_config({**base, "precision_raise": {"preset": "x"}})
    inf_same = {**base, "precision_raise": {"raise_sites": list(pr.PRESETS["32k"])}}
    assert pr.validate_config(base, inf_same) == []
    inf_diff = {**base, "precision_raise": {"preset": "128k"}}
    assert pr.validate_config(base, inf_diff)


def test_zero_train_gen_mismatch_compares_resolved_sites():
    from nemo_rl.models.megatron.zero_train_gen_mismatch import _resolved

    a = _resolved({"precision_raise": {"preset": "128k"}}, "precision_raise")
    b = _resolved({"precision_raise": {"raise_sites": list(pr.PRESETS["128k"])}}, "precision_raise")
    assert a == b
    assert _resolved({}, "precision_raise") == _resolved({"precision_raise": None}, "precision_raise")
    assert a != _resolved({"precision_raise": {"preset": "32k"}}, "precision_raise")


def test_init_patch_tags_named_linears(monkeypatch):
    class Lin:
        pass

    class Block:
        def __init__(self):
            self.in_proj, self.out_proj, self.other = Lin(), Lin(), Lin()

    monkeypatch.setitem(pr._STATE, "orig", {})
    orig = Block.__init__
    try:
        pr._patch_init(Block, ("in_proj", "out_proj", "missing"), "t")
        b = Block()
        assert pr.hilo_input(b.in_proj) and pr.hilo_input(b.out_proj)
        assert not pr.hilo_input(b.other)
    finally:
        Block.__init__ = orig


def test_te_linear_forward_uses_hilo_for_tagged_module(monkeypatch):
    """A tagged linear runs fp32_gemm (hi/lo); an untagged one the fp16 GEMM (CPU reference math)."""
    from nemo_rl.models.megatron import fp16_gemm as G

    torch.manual_seed(0)
    w = (torch.randn(24, 40) * 0.1).to(torch.bfloat16)
    x = torch.randn(5, 40) * 3
    mod = SimpleNamespace(weight=w, parallel_mode=None, use_bias=False, bias=None)
    plain, _ = G._te_linear_forward(mod, x, norm=False)
    mod._ts_hilo_input = True
    hilo, _ = G._te_linear_forward(mod, x, norm=False)
    assert torch.equal(plain, G.fp16_gemm(x, w))
    assert torch.equal(hilo, G.fp32_gemm(x, w))
    ref = x.double() @ w.double().t()
    assert (hilo.double() - ref).abs().max() < (plain.double() - ref).abs().max() / 100


# ---------------------------------------------------------------------------
# GPU: engine == learner bitwise.
# ---------------------------------------------------------------------------


def _bits(t):
    return t.contiguous().view(torch.int32)


@needs_cuda
@pytest.mark.parametrize("k,n", [(2688, 4096), (4096, 2688), (2688, 3712), (1856, 2688)])
def test_hilo_linear_rows_are_batch_invariant(k, n):
    """The raised dense GEMM (Nano in_proj/out_proj/qkv/shared-expert shapes): a row's bits do not
    depend on M (decode rows 1..256 vs a learner micro-batch > one 2048-row chunk), on the fused vs
    unfused split, or on autograd (learner forward) vs no_grad (engine)."""
    from nemo_rl.models.megatron import fp16_gemm as G

    gen = torch.Generator(device="cuda").manual_seed(k + n)
    w = (torch.randn(n, k, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    x = torch.randn(4100, k, device="cuda", generator=gen) * torch.exp(
        torch.randn(1, k, device="cuda", generator=gen)
    )
    xg = x.clone().requires_grad_()
    wg = w.clone().requires_grad_()
    ref = G.fp16_linear(xg, wg, precision=G.FP32_HEAD)  # learner (grad mode, chunked)
    with torch.no_grad():
        w16 = w.to(torch.float16)  # engine weights held in fp16 (hold_fp16_weights)
        for m in (1, 3, 16, 64, 255, 256):
            for start in (0, 37, 4100 - m):
                eng = G.fp16_linear(x[start : start + m], w16, precision=G.FP32_HEAD)
                assert torch.equal(_bits(eng), _bits(ref[start : start + m].detach())), (m, start)
        unfused = G.fp32_gemm(x[:300].cpu().cuda(), w)  # chunked path, maybe fused kernels
        x32 = x[:300]
        hi = x32.to(torch.float16)
        lo = ((x32 - hi.float()) * 2048.0).to(torch.float16)
        both = torch.mm(torch.cat([hi, lo]), w.to(torch.float16).t(), out_dtype=torch.float32)
        manual = both[:300] + both[300:] / 2048.0
        assert torch.equal(_bits(unfused), _bits(manual))
    # backward unchanged in kind: bf16 dgrad / wgrad of the fp32 input
    (ref.sum()).backward()
    assert xg.grad is not None and wg.grad is not None and torch.isfinite(xg.grad).all()


def _moe_case(e, topk, h, f, t, gen, activation):
    fn = 2 * f if activation == "swiglu" else f
    # bf16 weights (the learner's), held by the engine as exact fp16 copies
    w1 = (torch.randn(e, fn, h, device="cuda", generator=gen) * 0.02).to(torch.bfloat16).to(torch.float16)
    w2 = (torch.randn(e, h, f, device="cuda", generator=gen) * 0.02).to(torch.bfloat16).to(torch.float16)
    x32 = torch.randn(t, h, device="cuda", generator=gen)
    scores = torch.randn(t, e, device="cuda", generator=gen)
    top = scores.topk(topk, dim=1)
    probs = torch.softmax(top.values, dim=1).float()
    return w1, w2, x32, top.indices.to(torch.int64), probs


def _module(w1, w2, activation):
    import torch.nn.functional as F

    act = SimpleNamespace(__name__="squared_relu") if activation == "relu2" else F.silu
    cfg = SimpleNamespace(activation_func=act, gated_linear_unit=activation == "swiglu")

    def lin(w):
        return SimpleNamespace(num_gemms=w.shape[0], **{f"weight{i}": w[i] for i in range(w.shape[0])})

    return SimpleNamespace(linear_fc1=lin(w1), linear_fc2=lin(w2), config=cfg)


def _learner(fm, x32, probs, routing, w1, w2, activation, device_counts):
    """The training path: MCore permute -> training_experts_forward -> BI fp64 unpermute."""
    from megatron.core.transformer.moe.moe_utils import permute

    from nemo_rl.models.megatron import bi_unpermute

    t, e = x32.shape[0], w1.shape[0]
    rm = torch.zeros(t, e, dtype=torch.bool, device="cuda")
    rm.scatter_(1, routing, True)
    prb = torch.zeros(t, e, device="cuda")
    prb.scatter_(1, routing, probs)
    xp, pp, _, inv, _ = permute(
        x32, rm, probs=prb, num_out_tokens=t * routing.shape[1],
        return_batch_invariant_inverse_map=True,
    )
    counts = rm.sum(0)
    tpe = counts.to(torch.int32) if device_counts else counts.cpu()
    xp = xp.detach().requires_grad_()
    mod = _module(w1.to(torch.bfloat16), w2.to(torch.bfloat16), activation)
    y, _ = fm.training_experts_forward(mod, xp, tpe, pp)
    res = bi_unpermute.fused_unpermute(
        y, torch.Size((t, x32.shape[1])), probs=None, num_experts=e, inverse_map=inv,
        ep_size=1, cross_rank_fp64=False,
    )
    return res, xp


def _engine(fm, x16, probs, routing, w1, w2, ep, activation, limbs, **kw):
    from nemo_rl.models.megatron import ep_invariant_combine as eic

    e = w1.shape[0]
    el = e // ep
    t = x16.shape[0]
    vt = torch.tensor([t], dtype=torch.int32, device="cuda")
    parts = [
        fm.engine_experts(
            x16, probs, routing, vt, w1[r * el : (r + 1) * el].contiguous(),
            w2[r * el : (r + 1) * el].contiguous(), r * el, ep, activation, limbs=limbs, **kw,
        )[:t]
        for r in range(ep)
    ]
    if ep == 1:
        return parts[0]
    acc = torch.zeros_like(parts[0])
    for r in range(ep):
        acc = acc + parts[r]
    return eic.finalize_limbs(acc)


@needs_cuda
@pytest.mark.parametrize("activation", ["relu2", "swiglu"])
@pytest.mark.parametrize("fc2_hilo,c16", [(True, False), (False, True), (True, True), (False, False)])
@pytest.mark.parametrize("gemm", ["cublas", "segment"])
def test_routed_fc2_hilo_and_payload_engine_equals_learner(monkeypatch, raised, activation, fc2_hilo, c16, gemm):
    from nemo_rl.models.megatron import ep_invariant_combine as eic
    from nemo_rl.models.megatron import fp16_moe as fm

    monkeypatch.setenv("NEMO_RL_FP16_MOE_TRAIN_GEMM", gemm)
    raised((pr.ROUTED_FC2_HILO,) if fc2_hilo else (), c16)
    gen = torch.Generator(device="cuda").manual_seed(11)
    e, topk, h, f, t = 32, 6, 512, 256, 96
    w1, w2, x32, routing, probs = _moe_case(e, topk, h, f, t, gen, activation)
    x16 = x32.to(torch.float16)
    fm.invalidate_weight_cache()
    kw = {"fc2_hilo": fc2_hilo, "payload_fp16": c16}
    for device_counts in (False, True):
        lrn, xp = _learner(fm, x32.to(torch.float16).float(), probs, routing, w1, w2, activation, device_counts)
        ref = _engine(fm, x16, probs, routing, w1, w2, 1, activation, False, **kw)
        assert torch.equal(_bits(lrn.detach()), _bits(ref)), device_counts
    # EP-invariant limbs: same bits at EP 1/2/4 with the raise on
    eic.set_enabled(True)
    try:
        ref1 = _engine(fm, x16, probs, routing, w1, w2, 1, activation, True, **kw)
        for ep in (2, 4):
            out = _engine(fm, x16, probs, routing, w1, w2, ep, activation, True, **kw)
            assert torch.equal(_bits(out), _bits(ref1)), ep
    finally:
        eic.set_enabled(False)
    # backward still flows (fc2 hi/lo keeps today's fc2 backward)
    lrn.sum().backward()
    assert xp.grad is not None and torch.isfinite(xp.grad).all()


@needs_cuda
def test_routed_fc2_hilo_is_closer_to_fp32(raised):
    from nemo_rl.models.megatron import fp16_moe as fm

    gen = torch.Generator(device="cuda").manual_seed(2)
    e, topk, h, f, t = 16, 4, 256, 128, 64
    w1, w2, x32, routing, probs = _moe_case(e, topk, h, f, t, gen, "relu2")
    x16 = x32.to(torch.float16)
    base = _engine(fm, x16, probs, routing, w1, w2, 1, "relu2", False)
    hilo = _engine(fm, x16, probs, routing, w1, w2, 1, "relu2", False, fc2_hilo=True)
    # fp64 reference of the same model (fc1 input fp16 as in the recipe)
    out = torch.zeros(t, h, dtype=torch.float64, device="cuda")
    for j in range(topk):
        ei = routing[:, j]
        hx = torch.einsum("tk,tnk->tn", x16.double(), w1[ei].double())
        a = torch.relu(hx).pow(2) * probs[:, j : j + 1].double()
        out += torch.einsum("tf,thf->th", a, w2[ei].double())
    eb = (base.double() - out).abs().max().item()
    eh = (hilo.double() - out).abs().max().item()
    assert eh < eb / 20, (eb, eh)


@needs_cuda
def test_engine_fc2_hilo_under_cuda_graph_replay():
    from nemo_rl.models.megatron import fp16_moe as fm

    gen = torch.Generator(device="cuda").manual_seed(9)
    e, topk, h, f, t = 32, 6, 512, 256, 256
    w1, w2, x32, routing, probs = _moe_case(e, topk, h, f, t, gen, "relu2")
    x16 = x32.to(torch.float16)
    vt = torch.tensor([16], dtype=torch.int32, device="cuda")
    args = (x16, probs, routing, vt, w1, w2, 0, 1, "relu2")
    kw = {"tokens_hint": 16, "fc2_hilo": True, "payload_fp16": True}
    fm.engine_experts(*args, **kw)  # compile outside capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fm.engine_experts(*args, **kw)
    g.replay()
    torch.cuda.synchronize()
    eager = fm.engine_experts(*args, tokens_hint=None, fc2_hilo=True, payload_fp16=True)
    assert torch.equal(_bits(out[:16]), _bits(eager[:16]))
