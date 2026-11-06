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
"""Cross-fade MoE routing (nemo_rl crossfade_routing.py): the rule against a per-token statement, the m -> 0
limit, batch invariance, the gradient, the MCore patch (sparse training path == dense engine path) and the
config plumbing."""

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.megatron import crossfade_routing as cf

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _reference_row(x, b, k, extra, m, sf):
    """One token, fp64, Python: the rule as written in BUILD_DESIGN.md §1."""
    E = len(x)
    s = [1.0 / (1.0 + torch.exp(torch.tensor(-xi, dtype=torch.float64)).item()) for xi in x]
    t = [s[e] + b[e] for e in range(E)]
    C = min(E, k + extra)
    idx = sorted(range(E), key=lambda e: (-t[e], e))[:C]
    v = [t[e] for e in idx]
    tau = (v[k - 1] + v[k]) * 0.5
    c = [min(max((vj - tau) / m + 0.5, 0.0), 1.0) for vj in v]
    w = [s[e] * cj for e, cj in zip(idx, c)]
    W = sum(w)
    return idx, [wj / W * (sf or 1.0) for wj in w]


@pytest.mark.parametrize("k,extra,E,m,sf", [(1, 2, 8, 0.05, None), (6, 4, 128, 0.01, 2.5), (22, 4, 512, 0.003, 5.0)])
def test_rule_matches_per_token_reference(k, extra, E, m, sf):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, E, generator=g) * 0.5
    b = torch.randn(E, generator=g) * 0.01
    p, idx = cf.crossfade_route(x, b, k, extra, m, sf)
    assert p.dtype == torch.float32 and p.shape == (64, min(E, k + extra))
    for r in range(64):
        ridx, rp = _reference_row(x[r].tolist(), b.tolist(), k, extra, m, sf)
        assert idx[r].tolist() == ridx
        # fp32 selection scores vs fp64: the ramp slope 1/m amplifies a score rounding of ~1e-8 to ~1e-5 in c
        torch.testing.assert_close(p[r].double(), torch.tensor(rp, dtype=torch.float64), rtol=1e-4, atol=5e-8 / m)


def test_ties_go_to_the_lower_expert_id():
    x = torch.zeros(3, 16)
    b = torch.zeros(16)
    _, idx = cf.crossfade_route(x, b, k=4, extra=4, m=0.01)
    assert idx.tolist() == [list(range(8))] * 3


def test_stock_experts_always_kept_and_band_bounds():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(512, 128, generator=g)
    b = torch.randn(128, generator=g) * 0.01
    k, extra = 6, 4
    p, idx = cf.crossfade_route(x, b, k, extra, m=0.01)
    t = torch.sigmoid(x) + b
    stock = torch.sort(-t, dim=1, stable=True).indices[:, :k]
    assert torch.equal(idx[:, :k], stock)
    assert (p[:, :k] > 0).all()
    torch.testing.assert_close(p.sum(1), torch.ones(512), rtol=1e-6, atol=1e-6)


def test_m_to_zero_is_stock_topk():
    """m -> 0: every token without a k / k+1 tie routes exactly as MCore's sigmoid + bias top-k."""
    from megatron.core.transformer.moe import moe_utils

    orig = getattr(moe_utils.topk_routing_with_score_function, "__wrapped__", moe_utils.topk_routing_with_score_function)
    g = torch.Generator().manual_seed(2)
    x = torch.randn(256, 128, generator=g)
    b = torch.randn(128, generator=g) * 0.01
    k = 6
    p, idx = cf.crossfade_route(x, b, k, 4, m=1e-12, scaling_factor=2.5)
    sp, sidx = orig(x, k, score_function="sigmoid", expert_bias=b, scaling_factor=2.5, dense_output=True)
    assert (p[:, k:] == 0).all()
    order = torch.argsort(sidx, dim=1)
    assert torch.equal(idx[:, :k].sort(1).values, sidx.sort(1).values)
    torch.testing.assert_close(
        p[:, :k].gather(1, torch.argsort(idx[:, :k], 1)), sp.gather(1, order), rtol=1e-6, atol=1e-7
    )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda)])
def test_rule_is_batch_invariant(device):
    g = torch.Generator().manual_seed(3)
    x = torch.randn(300, 512, generator=g).to(device)
    b = (torch.randn(512, generator=g) * 0.01).to(device)
    full_p, full_i = cf.crossfade_route(x, b, 22, 4, 0.003, 5.0)
    for rows in (slice(0, 1), slice(7, 8), slice(5, 37), slice(299, 300)):
        p, i = cf.crossfade_route(x[rows].clone(), b, 22, 4, 0.003, 5.0)
        assert torch.equal(p, full_p[rows]) and torch.equal(i, full_i[rows])
    perm = torch.randperm(300, generator=g).to(device)
    p, i = cf.crossfade_route(x[perm], b, 22, 4, 0.003, 5.0)
    assert torch.equal(p, full_p[perm]) and torch.equal(i, full_i[perm])


def test_gradient_matches_finite_differences_inside_the_band():
    """d(sum_j a_j p_j)/dx through s, the ramp and tau, against fp64 central differences of the reference."""
    g = torch.Generator().manual_seed(4)
    k, extra, m = 3, 3, 0.05
    E = 12
    x = (torch.randn(1, E, generator=g) * 0.3).requires_grad_(True)
    b = torch.zeros(E)
    a = torch.randn(k + extra, generator=g)
    p, idx = cf.crossfade_route(x, b, k, extra, m)
    (p[0] * a).sum().backward()
    grad = x.grad[0].double()

    def f(xr):
        ridx, rp = _reference_row(xr, [0.0] * E, k, extra, m, None)
        assert ridx == idx[0].tolist()  # the step stays inside the same candidate order
        return sum(aj * pj for aj, pj in zip(a.tolist(), rp))

    h = 1e-5
    x0 = x.detach()[0].double().tolist()
    fd = []
    for e in range(E):
        xp, xm = list(x0), list(x0)
        xp[e] += h
        xm[e] -= h
        fd.append((f(xp) - f(xm)) / (2 * h))
    torch.testing.assert_close(grad, torch.tensor(fd, dtype=torch.float64), rtol=2e-3, atol=2e-4)
    assert (grad[[e for e in range(E) if e not in idx[0].tolist()]] == 0).all()


@pytest.fixture
def cf_on():
    cf.install()
    old = cf.get_state()
    cf._STATE.update(m=0.01, k=6, extra=4)
    yield
    cf._STATE.update(old)


def test_patch_sparse_training_path_equals_dense_engine_path(cf_on):
    from megatron.core.transformer.moe import router as mcore_router

    g = torch.Generator().manual_seed(5)
    x = torch.randn(96, 128, generator=g)
    b = torch.randn(128, generator=g) * 0.01
    fn = mcore_router.topk_routing_with_score_function
    assert getattr(fn, cf._PATCH_ATTR, False)
    dp, di = fn(x, 10, score_function="sigmoid", expert_bias=b, scaling_factor=2.5, dense_output=True)
    sp, smap = fn(x, 10, score_function="sigmoid", expert_bias=b, scaling_factor=2.5)
    assert smap.sum(1).eq(10).all()  # fixed-width dispatch: C candidates per token
    assert torch.equal(sp.gather(1, di), dp)
    assert torch.equal(smap.int().gather(1, di), torch.ones_like(di, dtype=torch.int32))
    ref_p, ref_i = cf.crossfade_route(x, b, 6, 4, 0.01, 2.5)
    assert torch.equal(di, ref_i) and torch.equal(dp, ref_p.type_as(x))
    # the engine's MoE kernels read the dense [T, topk] tensors by raw pointer
    assert dp.is_contiguous() and di.is_contiguous()


def test_patch_rejects_an_unwidened_router(cf_on):
    from megatron.core.transformer.moe import router as mcore_router

    with pytest.raises(RuntimeError, match="not widened"):
        mcore_router.topk_routing_with_score_function(
            torch.randn(4, 128), 6, score_function="sigmoid", expert_bias=torch.zeros(128)
        )
    with pytest.raises(NotImplementedError):
        mcore_router.topk_routing_with_score_function(torch.randn(4, 128), 10, score_function="softmax")


def test_patch_off_is_megatron():
    cf.install()
    old = cf.get_state()
    cf._STATE.update(m=None, k=None, extra=None)
    try:
        from megatron.core.transformer.moe import router as mcore_router

        fn = mcore_router.topk_routing_with_score_function
        x, b = torch.randn(16, 32), torch.zeros(32)
        p1, i1 = fn(x, 4, score_function="sigmoid", expert_bias=b, dense_output=True)
        p2, i2 = fn.__wrapped__(x, 4, score_function="sigmoid", expert_bias=b, dense_output=True)
        assert torch.equal(p1, p2) and torch.equal(i1, i2)
    finally:
        cf._STATE.update(old)


def test_configure_widens_topk_once_and_checks_conflicts():
    old = cf.get_state()
    try:
        cfg = {"megatron_cfg": {"crossfade_routing_m": 0.003, "crossfade_routing_extra": 4}}
        mc = SimpleNamespace(moe_router_topk=22, moe_router_fusion=False)
        assert cf.configure(cfg, mc) == 0.003
        assert mc.moe_router_topk == 26 and cf.get_state()["k"] == 22
        cf.configure(cfg, mc)  # the inference provider snapshot of the same config: no second widening
        assert mc.moe_router_topk == 26
        with pytest.raises(ValueError, match="moe_router_fusion"):
            cf.configure(cfg, SimpleNamespace(moe_router_topk=22, moe_router_fusion=True))
        with pytest.raises(ValueError, match="configured twice"):
            cf.configure({"megatron_cfg": {"crossfade_routing_m": 0.01}}, SimpleNamespace(moe_router_topk=22))
        assert cf.configure({"megatron_cfg": {}}, SimpleNamespace(moe_router_topk=22)) is None
    finally:
        cf._STATE.update(old)


def _master(**ts):
    base = dict(
        crossfade_routing_m=0.003, crossfade_routing_extra=4, sticky_routing=False, sticky_routing_delta=0.03,
        cohort_lanes=1, sticky_rounding=False,
    )
    base.update(ts)
    return SimpleNamespace(thundersync=SimpleNamespace(**base), policy={"megatron_cfg": {}})


def test_validator():
    from thundersync_rl.grpo_loop import validate_crossfade_routing

    validate_crossfade_routing(_master())
    validate_crossfade_routing(_master(crossfade_routing_m=None, sticky_routing=True, cohort_lanes=2))
    for bad, msg in (
        (dict(sticky_routing=True), "sticky_routing must be false"),
        (dict(cohort_lanes=2), "cohort_lanes must be 1"),
        (dict(crossfade_routing_m=0.0), "must be > 0"),
        (dict(crossfade_routing_extra=0), "extra must be >= 1"),
    ):
        with pytest.raises(ValueError, match=msg):
            validate_crossfade_routing(_master(**bad))


@cuda
@pytest.mark.parametrize("E,k,extra,m,sf", [(128, 6, 4, 0.01, 2.5), (512, 22, 4, 0.003, 5.0), (512, 22, 8, 0.03, 5.0)])
@pytest.mark.parametrize("kind", ["randn", "ties", "band"])
def test_fused_kernel_is_bitwise_the_reference(monkeypatch, E, k, extra, m, sf, kind):
    """The fused CUDA kernel (kernel task E) gives the reference's bits: gates and candidate ids, every row."""
    g = torch.Generator(device="cuda").manual_seed(7)
    T = 1024
    x = torch.randn(T, E, generator=g, device="cuda") * 2.0
    b = torch.randn(E, generator=g, device="cuda") * 0.01
    if kind == "ties":
        x, b = torch.round(x * 4) / 4, torch.zeros_like(b)
    elif kind == "band":
        x = torch.randn(T, E, generator=g, device="cuda") * 1e-3
    pf, i_f = cf.crossfade_route(x, b, k, extra, m, sf)
    monkeypatch.setenv("THUNDERSYNC_CF_FUSED", "0")
    pr, ir = cf.crossfade_route(x, b, k, extra, m, sf)
    assert pf.is_contiguous() and i_f.is_contiguous() and i_f.dtype == torch.int64
    assert torch.equal(i_f, ir)
    assert torch.equal(pf.view(torch.int32), pr.view(torch.int32))


@cuda
def test_fused_backward_matches_the_reference_gradient(monkeypatch):
    g = torch.Generator(device="cuda").manual_seed(8)
    x = (torch.randn(256, 512, generator=g, device="cuda") * 1e-3).requires_grad_(True)
    b = torch.zeros(512, device="cuda")
    gp = torch.randn(256, 26, generator=g, device="cuda")
    p, _ = cf.crossfade_route(x, b, 22, 4, 0.003, 5.0)
    (gf,) = torch.autograd.grad((p * gp).sum(), x)
    monkeypatch.setenv("THUNDERSYNC_CF_FUSED", "0")
    p2, _ = cf.crossfade_route(x, b, 22, 4, 0.003, 5.0)
    (gr,) = torch.autograd.grad((p2 * gp).sum(), x)
    rel = (gf - gr).norm(dim=1) / (gr.norm(dim=1) + 1e-12)
    assert float(rel.max()) < 1e-4


def test_draft_at_step_top_validator():
    from thundersync_rl.grpo_loop import validate_draft_at_step_top

    def m(**ts):
        base = dict(draft_at_step_top=True, sticky_routing=False, sticky_routing_delta=0.03, cohort_lanes=1,
                    draft_weights="current", draft_after_verify=False)
        base.update(ts)
        return SimpleNamespace(thundersync=SimpleNamespace(**base))

    validate_draft_at_step_top(m())
    validate_draft_at_step_top(m(draft_at_step_top=False, sticky_routing=True, cohort_lanes=2))
    for bad in (dict(sticky_routing=True), dict(cohort_lanes=2), dict(draft_weights="momentum"),
                dict(draft_after_verify=True)):
        with pytest.raises(ValueError):
            validate_draft_at_step_top(m(**bad))


@pytest.fixture
def ramp_stopgrad():
    old = cf._STATE.get("ramp_stopgrad", False)
    cf._STATE["ramp_stopgrad"] = True
    yield
    cf._STATE["ramp_stopgrad"] = old


def _stock_topk_gates(x, b, k, sf):
    """Megatron's sigmoid top-k gates: s[top-k of s + b] normalized, times sf (no gradient through the selection)."""
    s = torch.sigmoid(x.float())
    _, idx = torch.topk((s + b).detach(), k, dim=1)
    w = torch.gather(s, 1, idx)
    return w / w.sum(1, keepdim=True) * (sf or 1.0), idx


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda)])
def test_ramp_stopgrad_forward_is_bitwise_unchanged(device, monkeypatch):
    g = torch.Generator(device=device).manual_seed(11)
    x = torch.randn(512, 128, generator=g, device=device) * 1e-2
    b = torch.randn(128, generator=g, device=device) * 1e-3
    if device == "cpu":
        monkeypatch.setenv("THUNDERSYNC_CF_FUSED", "0")
    p0, i0 = cf.crossfade_route(x, b, 6, 4, 0.01, 2.5)
    cf._STATE["ramp_stopgrad"] = True
    try:
        p1, i1 = cf.crossfade_route(x, b, 6, 4, 0.01, 2.5)
    finally:
        cf._STATE["ramp_stopgrad"] = False
    assert torch.equal(i0, i1) and torch.equal(p0.view(torch.int32), p1.view(torch.int32))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda)])
def test_ramp_stopgrad_gradient_is_stock_at_m_to_zero_and_differs_with_the_ramp(device, monkeypatch, ramp_stopgrad):
    """m -> 0: every candidate is fully in or out, so the stop-grad gradient equals stock top-k's; at m .01 with many
    rows in the band the full ramp gradient differs (non-vacuous control)."""
    if device == "cpu":
        monkeypatch.setenv("THUNDERSYNC_CF_FUSED", "0")
    g = torch.Generator(device=device).manual_seed(12)
    x = (torch.randn(256, 128, generator=g, device=device) * 2.0).requires_grad_(True)
    b = torch.zeros(128, device=device)
    gp = torch.randn(256, 6, generator=g, device=device)
    p, _ = cf.crossfade_route(x, b, 6, 4, 1e-9, 2.5)
    (g_sg,) = torch.autograd.grad((p[:, :6] * gp).sum(), x)
    ps, _ = _stock_topk_gates(x, b, 6, 2.5)
    (g_st,) = torch.autograd.grad((ps * gp).sum(), x)
    rel = (g_sg - g_st).norm() / g_st.norm()
    assert float(rel) < 1e-5, float(rel)
    # control: in-band inputs, full ramp gradient vs stop-grad gradient
    xb = (torch.randn(256, 128, generator=g, device=device) * 1e-3).requires_grad_(True)
    gp2 = torch.randn(256, 10, generator=g, device=device)
    p1, _ = cf.crossfade_route(xb, b, 6, 4, 0.01, 2.5)
    (g1,) = torch.autograd.grad((p1 * gp2).sum(), xb)
    cf._STATE["ramp_stopgrad"] = False
    p2, _ = cf.crossfade_route(xb, b, 6, 4, 0.01, 2.5)
    (g2,) = torch.autograd.grad((p2 * gp2).sum(), xb)
    cf._STATE["ramp_stopgrad"] = True
    assert float((g1 - g2).norm() / g2.norm()) > 0.1
    assert float(g2.norm() / g1.norm()) > 1.5  # the ramp term (slope 1/m) dominates in the band


@cuda
def test_ramp_stopgrad_fused_backward_equals_reference_stopgrad(monkeypatch, ramp_stopgrad):
    g = torch.Generator(device="cuda").manual_seed(13)
    x = (torch.randn(256, 512, generator=g, device="cuda") * 1e-3).requires_grad_(True)
    b = torch.zeros(512, device="cuda")
    gp = torch.randn(256, 26, generator=g, device="cuda")
    p, _ = cf.crossfade_route(x, b, 22, 4, 0.003, 5.0)
    (gf,) = torch.autograd.grad((p * gp).sum(), x)
    monkeypatch.setenv("THUNDERSYNC_CF_FUSED", "0")
    p2, _ = cf.crossfade_route(x, b, 22, 4, 0.003, 5.0)
    (gr,) = torch.autograd.grad((p2 * gp).sum(), x)
    assert float(((gf - gr).norm(dim=1) / (gr.norm(dim=1) + 1e-12)).max()) < 1e-5


def test_ramp_stopgrad_config_plumbing():
    assert cf.crossfade_ramp_stopgrad({"megatron_cfg": {"crossfade_routing_ramp_stopgrad": True}})
    assert not cf.crossfade_ramp_stopgrad({"megatron_cfg": {}})
