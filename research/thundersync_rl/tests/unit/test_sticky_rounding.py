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
"""Sticky fp16 rounding (nemo_rl sticky_rounding.py): the rule r(x; a), its
edge cases, the research emulation's semantics, the anchor stream's site
bookkeeping, keyed expert anchors and theta_n's host store."""

import math

import numpy as np
import pytest
import torch

from nemo_rl.models.megatron import sticky_rounding as sr

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
F16 = torch.float16
INF = float("inf")
NAN = float("nan")


def _reference(x: float, a: float) -> float:
    """Scalar statement of the rule (numpy fp16 neighbours, float64 compare)."""
    rtn = float(np.float16(x))
    if math.isnan(a) or math.isnan(x):
        return rtn
    a16 = np.float16(a)
    up = float(np.nextafter(a16, np.float16(INF)))
    dn = float(np.nextafter(a16, np.float16(-INF)))
    return float(a16) if dn < x < up else rtn


def _r(x, a):
    return sr.sticky_round16(
        torch.tensor(x, dtype=torch.float32), torch.tensor(a, dtype=F16)
    )


def _same(t: torch.Tensor, ref: list[float]) -> bool:
    """Bitwise for numbers (signed zeros included), NaN == NaN."""
    r = torch.tensor(ref, dtype=F16)
    t = t.reshape(r.shape)
    return bool(torch.equal(t.view(torch.int16), r.view(torch.int16))) or bool(
        (torch.isnan(t) & torch.isnan(r)).all()
    )


def test_matches_scalar_reference_random():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(20000, generator=g) * torch.exp2(torch.randint(-20, 12, (20000,), generator=g).float())
    # anchors near x: RTN, either neighbour, two ulps away, or far
    x16 = x.to(F16)
    up = torch.nextafter(x16, torch.full_like(x16, INF))
    dn = torch.nextafter(x16, torch.full_like(x16, -INF))
    choice = torch.randint(0, 5, x.shape, generator=g)
    a = torch.where(choice == 0, x16, torch.where(choice == 1, up, torch.where(choice == 2, dn, torch.where(choice == 3, torch.nextafter(up, torch.full_like(up, INF)), (-x).to(F16)))))
    out = sr.sticky_round16(x, a)
    ref = [_reference(float(xi), float(ai)) for xi, ai in zip(x.tolist(), a.tolist())]
    assert _same(out, ref)
    # the anchor changes the rounding for some elements and stays within one ulp
    assert (out != x16).any()
    finite = torch.isfinite(out) & torch.isfinite(x16) & (x16.abs() < 60000)
    # one ulp of the binade on either side of x (the larger at a binade edge)
    ulp = torch.maximum(up.float() - x16.float(), x16.float() - dn.float())
    assert ((out.float() - x)[finite].abs() <= ulp[finite]).all()


@pytest.mark.parametrize(
    "x,a,expected",
    [
        (1.0, 1.0009765625, 1.0),  # exactly representable x: r = x for any a
        (1.0, 0.99951171875, 1.0),
        (1.0 + 2.0**-12, 1.0009765625, 1.0009765625),  # below midpoint, anchor up
        (1.0 + 3 * 2.0**-12, 1.0, 1.0),  # above midpoint, anchor down
        (1.0 + 2.0**-10, 1.0, 1.0009765625),  # x is a's next16: not strictly inside
        # binade boundary at 2: ulp below 2 is 2^-10, above is 2^-9
        (2.0 - 2.0**-12, 1.9990234375, 1.9990234375),
        (2.0 + 0.9 * 2.0**-9, 2.0, 2.0),  # RTN would give 2 + 2^-9
        (2.0 - 0.9 * 2.0**-10, 2.0, 2.0),
        (2.0 + 0.9 * 2.0**-9, 1.9990234375, 2.001953125),  # anchor not a neighbour
        # subnormals (fp16 min subnormal 2^-24)
        (0.7 * 2.0**-24, 0.0, 0.0),
        (0.3 * 2.0**-24, 2.0**-24, 2.0**-24),
        (1.5 * 2.0**-24, 2.0**-24, 2.0**-24),
        (2.0**-14 + 0.4 * 2.0**-24, 2.0**-14 + 2.0**-24, 2.0**-14 + 2.0**-24),
        # sign changes
        (-0.3 * 2.0**-24, 0.0, 0.0),  # anchor +0 kept for a tiny negative x
        (0.3 * 2.0**-24, -(2.0**-24), 0.0),  # anchor on the other side of 0: RTN
        (-1.0001, 1.0, -1.0),
        # inf / nan / overflow
        (NAN, 1.0, NAN),
        (1.0004, NAN, 1.0),  # NaN anchor = self (RTN)
        (INF, 65504.0, INF),
        (-INF, -65504.0, -INF),
        (65500.0, INF, 65504.0),  # inf's prev16 is 65504 = x's RTN... x < 65504 < inf
        (70000.0, INF, INF),
        (70000.0, 65504.0, 65504.0),  # reference semantics: next16(65504) = inf
        (65519.0, 65472.0, 65504.0),
    ],
)
def test_edge_cases(x, a, expected):
    out = _r(x, a)
    assert _same(out, [expected]), (x, a, float(out), expected)
    assert _same(out, [_reference(x, a)])


def test_zero_sign_is_part_of_the_rule():
    # a = -0: its neighbours are -2^-24 and +2^-24, so a tiny positive x takes -0
    out = _r(0.3 * 2.0**-24, -0.0)
    assert out.item() == 0.0 and math.copysign(1.0, out.item()) < 0


def test_fp64_and_bf16_inputs():
    x = torch.tensor([1.0 + 2.0**-12, 3.0], dtype=torch.float64)
    a = torch.tensor([1.0009765625, 3.0], dtype=F16)
    assert sr.sticky_round16(x, a).tolist() == [1.0009765625, 3.0]
    xb = torch.tensor([1.0078125], dtype=torch.bfloat16)  # exactly representable in fp16
    assert sr.sticky_round16(xb, torch.tensor([1.0087890625], dtype=F16)).item() == 1.0078125


def test_anchor_must_be_fp16():
    with pytest.raises(TypeError):
        sr.sticky_round16(torch.ones(2), torch.ones(2))


def test_emulation_semantics():
    """research/sticky/sticky_round.py's apply step, restated: bitwise the same."""
    g = torch.Generator().manual_seed(1)
    x_n = torch.randn(4096, generator=g)
    x_n1 = x_n + 1e-4 * torch.randn(4096, generator=g)
    a = x_n.to(F16)  # the drafter's recorded value

    def emulation(x, a):
        r = x.to(F16)
        xf = x.detach().float()
        up = torch.nextafter(a, torch.tensor(INF, dtype=a.dtype)).float()
        dn = torch.nextafter(a, torch.tensor(-INF, dtype=a.dtype)).float()
        use = (dn < xf) & (xf < up)
        return torch.where(use, a, r)

    out = sr.sticky_round16(x_n1, a)
    assert torch.equal(out.view(torch.int16), emulation(x_n1, a).view(torch.int16))
    assert (out != x_n1.to(F16)).float().mean() > 0.05


def test_sticky_cast_is_straight_through():
    x = torch.randn(64, requires_grad=True)
    a = (x.detach() + 1e-4).to(F16)
    y = sr.StickyCast.apply(x, a)
    assert y.dtype == torch.float32
    assert torch.equal(y, sr.sticky_round16(x.detach(), a).float())
    g = torch.randn(64)
    y.backward(g)
    assert torch.equal(x.grad, g)


def _stream(mask, tp_rank=0, tp_size=1):
    return sr.AnchorStream(None, mask, tp_rank, tp_size)


def test_site_cast_record_apply_and_masks():
    S, B, D = 4, 2, 8
    mask = torch.tensor([True, False] * S)  # [S*B], s-major: batch row 0 anchored
    s = _stream(mask)
    x_n = torch.randn(S, B, D)
    x_n1 = x_n + 3e-4 * torch.randn(S, B, D)
    sr._STATE["stream"] = s
    try:
        s.mode = sr.RECORD
        assert torch.equal(sr.site_cast("lin", x_n), x_n.to(F16))
        s.mode = sr.APPLY
        sr.start_stats()
        out = sr.site_cast("lin", x_n1)
        st = sr.pop_stats()
        s.end_layer()
    finally:
        sr._STATE["stream"] = None
    want = sr.sticky_round16(x_n1, x_n.to(F16))
    assert torch.equal(out[:, 0], want[:, 0])  # anchored row
    assert torch.equal(out[:, 1], x_n1[:, 1].to(F16))  # self row: RTN
    assert st["elements"] == S * B * D and st["changed"] > 0


def test_mask_sequence_parallel_shard_and_mismatch():
    S, B = 8, 2
    mask = torch.arange(S * B) % 3 == 0
    s = _stream(mask, tp_rank=1, tp_size=2)
    full = s.mask_like(torch.zeros(S, B, 4, 3))
    assert full.shape == (S, B, 1, 1) and torch.equal(full.reshape(-1), mask)
    shard = s.mask_like(torch.zeros(S // 2, B, 4))
    assert torch.equal(shard.reshape(-1), mask[S * B // 2 :])
    with pytest.raises(RuntimeError):
        s.mask_like(torch.zeros(3, B, 4))


def test_unapplied_or_missing_anchor_raises():
    s = _stream(torch.ones(2, dtype=torch.bool))
    s.record("a", torch.ones(2, 1, 1), torch.ones(2, 1, 1, dtype=F16))
    with pytest.raises(RuntimeError):
        s.end_layer()
    with pytest.raises(RuntimeError):
        s.pop("b", torch.Size([2, 1, 1]))


def test_keyed_expert_anchors_follow_tokens_not_rows():
    """fc2 anchors are matched by (expert, source rank, token), whatever the row order."""
    T, E, H = 6, 3, 5
    g = torch.Generator().manual_seed(2)
    a_n = torch.randn(T, E, H, generator=g)  # the anchor forward's value of (t, e)
    a_n1 = a_n + 2e-4 * torch.randn(T, E, H, generator=g)
    routes_n = [[0, 1], [1, 2], [0, 2], [0, 1], [1, 2], [0, 2]]
    routes_n1 = [[0, 1], [1, 2], [0, 1], [0, 1], [1, 2], [0, 2]]  # token 2: 2 -> 1
    anchored = [1, 1, 1, 1, 0, 1]  # token 4 not anchored

    def rows(routes):
        out_keys, out_vals, tpe = [], [], []
        for e in range(E):
            toks = [t for t in range(T) if e in routes[t]]
            tpe.append(len(toks))
            for t in toks:
                out_keys.append([0.0, float(t), float(anchored[t]), 0.0])
        return torch.tensor(out_keys), tpe

    s = _stream(torch.ones(T, dtype=torch.bool))
    sr._STATE["stream"] = s
    try:
        k_n, tpe_n = rows(routes_n)
        vals_n = torch.stack([a_n[int(k[1]), e] for e, n in enumerate(tpe_n) for k in k_n[sum(tpe_n[:e]) : sum(tpe_n[:e]) + n]])
        s.mode = sr.RECORD
        s.pending_keys = (k_n, tpe_n)
        sr.experts_fc2_input("experts", vals_n, tpe_n)
        k_1, tpe_1 = rows(routes_n1)
        pairs = [(int(k[1]), e) for e, n in enumerate(tpe_1) for k in k_1[sum(tpe_1[:e]) : sum(tpe_1[:e]) + n]]
        vals_1 = torch.stack([a_n1[t, e] for t, e in pairs])
        s.mode = sr.APPLY
        s.pending_keys = (k_1, tpe_1)
        out = sr.experts_fc2_input("experts", vals_1, tpe_1)
        s.end_layer()
    finally:
        sr._STATE["stream"] = None
    for row, (t, e) in enumerate(pairs):
        if anchored[t] and e in routes_n[t]:
            want = sr.sticky_round16(a_n1[t, e], a_n[t, e].to(F16))
        else:
            want = a_n1[t, e].to(F16)
        assert torch.equal(out[row].to(F16), want), (t, e)


def test_theta_store_keeps_the_snapshot():
    lin = torch.nn.Linear(4, 3)
    emb = torch.nn.Embedding(5, 4)
    store = sr.ThetaStore([emb, lin])
    assert not store.ready and store.fetch(0, torch.device("cpu")) == ({}, None)
    store.snapshot()
    old = lin.weight.detach().clone()
    with torch.no_grad():
        lin.weight.add_(1.0)
    got = store.fetch(1, torch.device("cpu"))[0]
    assert torch.equal(got["weight"], old) and set(got) == {"weight", "bias"}
    assert store.host_bytes() == sum(p.numel() * 4 for p in [*emb.parameters(), *lin.parameters()])


def test_stream_prefetch_keeps_at_most_two_units():
    units = [torch.nn.Linear(2, 2) for _ in range(4)]
    store = sr.ThetaStore(units)
    store.snapshot()
    s = sr.AnchorStream(store, torch.ones(1, dtype=torch.bool))
    for i, u in enumerate(units):
        s.params(u, torch.device("cpu"))
        assert sorted(s._buf) == [j for j in (i, i + 1) if j < len(units)]


def test_policy_forward_requires_a_snapshot_for_anchored_tokens():
    sr._STATE["store"] = None
    with pytest.raises(RuntimeError):
        with sr.policy_forward(None, torch.tensor([True])):
            pass
    with sr.policy_forward(None, torch.tensor([False])) as s:
        assert sr.active_stream() is s
    assert sr.active_stream() is None


@cuda
def test_rule_on_gpu_matches_cpu():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(1 << 16, generator=g)
    a = (x + 5e-4 * torch.randn(x.shape, generator=g)).to(F16)
    assert torch.equal(sr.sticky_round16(x.cuda(), a.cuda()).cpu().view(torch.int16), sr.sticky_round16(x, a).view(torch.int16))


def test_config_flag():
    assert not sr.sticky_rounding_enabled({})
    assert not sr.sticky_rounding_enabled({"megatron_cfg": {"sticky_rounding": False}})
    assert sr.sticky_rounding_enabled({"megatron_cfg": {"sticky_rounding": True}})


def _master(megatron=None, **ts):
    from types import SimpleNamespace

    from thundersync_rl.grpo_loop import ThunderSyncConfig

    base = dict(
        speculate=True,
        verify_mode="block",
        verify_source="engine",
        verify_on="learner",
        draft_lookahead=1,
        sticky_routing=True,
        sticky_rounding=True,
    )
    base.update(ts)
    mcfg = {"forward_gemm_precision": "fp16", "expert_model_parallel_size": 2}
    mcfg.update(megatron or {})
    return SimpleNamespace(
        thundersync=ThunderSyncConfig(**base),
        grpo=SimpleNamespace(max_rollout_turns=1),
        policy={
            "megatron_cfg": mcfg,
            "generation": {"mcore_generation_config": {"enable_prefix_caching": False}},
        },
    )


def test_grpo_config_default_off_and_handoff():
    from thundersync_rl.grpo_loop import (
        ThunderSyncConfig,
        configure_sticky_policy,
        validate_sticky_rounding,
    )

    assert ThunderSyncConfig().sticky_rounding is False
    m = _master()
    validate_sticky_rounding(m)
    configure_sticky_policy(m)
    assert m.policy["megatron_cfg"]["sticky_rounding"] is True
    assert sr.sticky_rounding_enabled(m.policy)
    m = _master(sticky_rounding=False, sticky_routing=False)
    validate_sticky_rounding(m)
    configure_sticky_policy(m)
    assert m.policy["megatron_cfg"]["sticky_rounding"] is False


@pytest.mark.parametrize(
    "ts,megatron,match",
    [
        ({"sticky_routing": False}, {}, "sticky_routing must be on"),
        ({}, {"forward_gemm_precision": "bf16"}, "forward_gemm_precision"),
        ({}, {"pipeline_model_parallel_size": 2}, "pipeline_model_parallel_size"),
        ({}, {"context_parallel_size": 2}, "context_parallel_size"),
        ({}, {"activation_checkpointing": True}, "activation_checkpointing"),
        ({}, {"moe_token_dispatcher_type": "flex"}, "dispatcher"),
        ({}, {"moe_shared_expert_overlap": True}, "shared_expert_overlap"),
        ({"fused_verify_train": True}, {}, "fused_verify_train"),
    ],
)
def test_grpo_config_rejects(ts, megatron, match):
    from thundersync_rl.grpo_loop import validate_sticky_rounding

    with pytest.raises(ValueError, match=match):
        validate_sticky_rounding(_master(megatron, **ts))
