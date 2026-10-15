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
"""Run-readiness guards: weight sentinel, non-finite verification, draft collection
without frame fallback, multi-turn row prefixes, expert-bias + momentum."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
import torch

from thundersync_rl.block_verification import NonFiniteVerification, block_verify
from thundersync_rl.block_verification_fused import (
    block_verify_logits,
    block_verify_stats_lazy,
    block_verify_stats_lazy_grid,
)
from thundersync_rl.multi_turn import PrefixGuardGeneration, RowPrefixError, check_row_extends
from thundersync_rl.sentinel import (
    WeightSentinel,
    WeightSentinelError,
    canonical_named_tensors,
    compare,
    digest_named,
    engine_dtypes,
    tensor_digest,
)

# ---------------------------------------------------------------------------
# Weight sentinel (issue 7)
# ---------------------------------------------------------------------------


def test_digest_is_bitwise_and_position_sensitive():
    t = torch.randn(1000, dtype=torch.bfloat16)
    d = tensor_digest(t)
    assert d == tensor_digest(t.clone())
    u = t.clone()
    u.view(torch.int16)[17] ^= 1  # one ulp
    assert tensor_digest(u) != d
    assert tensor_digest(t.flip(0)) != d  # same multiset, other order
    # The learner casts to the engine dtype: bf16 -> fp16 equals the engine's copy.
    eng = t.to(torch.float16)
    assert tensor_digest(t, torch.float16) == tensor_digest(eng)


class _Layer(torch.nn.Module):
    def __init__(self, number, n_local, ep_rank):
        super().__init__()
        self.layer_number = number
        self.mlp = torch.nn.Module()
        self.mlp.experts = torch.nn.Module()
        self.mlp.experts.num_local_experts = n_local
        for j in range(n_local):
            setattr(self.mlp.experts, f"x{j}", None)
        self.mlp.experts.linear_fc1 = torch.nn.Module()
        for j in range(n_local):
            self.mlp.experts.linear_fc1.register_parameter(
                f"weight{j}", torch.nn.Parameter(torch.full((2,), float(100 * number + ep_rank * n_local + j)))
            )
        self.mlp.router = torch.nn.Module()
        self.mlp.router.register_buffer("expert_bias", torch.zeros(4))
        self.norm = torch.nn.LayerNorm(2)


class _Model(torch.nn.Module):
    def __init__(self, first_layer, n_layers, n_local, ep_rank):
        super().__init__()
        self.module = torch.nn.Module()  # a wrapper prefix, stripped
        self.module.decoder = torch.nn.Module()
        self.module.decoder.layers = torch.nn.ModuleList(
            [_Layer(first_layer + i + 1, n_local, ep_rank) for i in range(n_layers)]
        )


def test_canonical_names_are_layout_independent():
    # Learner: EP 1, one stage of 4 layers; engine: EP 2, two stages of 2 layers.
    learner = dict(canonical_named_tensors(_Model(0, 4, 4, 0), ep_rank=0))
    engine = {}
    for pp_first in (0, 2):
        for ep in (0, 1):
            engine.update(canonical_named_tensors(_Model(pp_first, 2, 2, ep), ep_rank=ep))
    assert "decoder.layers.3.mlp.experts.linear_fc1.weight3" in learner
    assert "decoder.layers.2.mlp.router.expert_bias" in learner
    assert set(engine) == set(learner)
    for k in learner:
        assert torch.equal(engine[k], learner[k]), k


def _side(model, ep=0, tp=0, dtypes=None):
    return (tp, digest_named(canonical_named_tensors(model, ep_rank=ep), dtypes))


def test_compare_names_mismatching_tensors_and_checks_replicas():
    m = _Model(0, 2, 2, 0)
    eng = [_side(m), _side(m)]  # two data-parallel replicas
    cov = compare(eng, [_side(_Model(0, 2, 2, 0))], "t")
    assert cov["sentinel/tensors"] > 0 and cov["sentinel/expert_bias"] == 2
    bad = _Model(0, 2, 2, 0)
    with torch.no_grad():
        bad.module.decoder.layers[1].mlp.experts.linear_fc1.weight1[0] += 1
    with pytest.raises(WeightSentinelError, match=r"decoder\.layers\.1\.mlp\.experts\.linear_fc1\.weight1"):
        compare(eng, [_side(bad)], "theta_3")
    with pytest.raises(WeightSentinelError, match="replicas disagree"):
        compare([_side(m), _side(bad)], [_side(m)], "t")
    with pytest.raises(WeightSentinelError, match="vacuous"):
        compare([(0, {"a": ("float32", 1, 0, 0)})], [(0, {"b": ("float32", 1, 0, 0)})], "t")


def test_sentinel_casts_learner_to_engine_dtype():
    lrn = _Model(0, 1, 2, 0).to(torch.bfloat16)
    eng = _Model(0, 1, 2, 0).to(torch.bfloat16)
    # The engine holds its GEMM weights in fp16 (hold_fp16_weights).
    for j in range(2):
        w = getattr(eng.module.decoder.layers[0].mlp.experts.linear_fc1, f"weight{j}")
        w.data = w.data.to(torch.float16)
    e = [_side(eng)]
    compare(e, [_side(lrn, dtypes=engine_dtypes(e))], "fp16")
    with pytest.raises(WeightSentinelError):
        compare(e, [_side(lrn)], "no cast")  # dtype is part of the digest


def test_weight_sentinel_driver_check_and_reference(monkeypatch):
    m = _Model(0, 1, 2, 0)
    s = WeightSentinel(policy=None)
    monkeypatch.setattr(s, "engine_digest", lambda gen: [_side(gen)])
    monkeypatch.setattr(s, "learner_digest", lambda dtypes, **kw: [_side(m, dtypes=dtypes)])
    ref = s.check(m, "theta_hat_1")
    s.check_engine_against(m, ref, "reload")
    stats = s.pop_stats()
    assert stats["sentinel/checks"] == 2 and stats["sentinel/tensors"] > 0
    moved = _Model(0, 1, 2, 0)
    with torch.no_grad():
        moved.module.decoder.layers[0].norm.weight[0] = 3.0
    with pytest.raises(WeightSentinelError, match="norm.weight"):
        s.check_engine_against(moved, ref, "reload")


def test_worker_digest_flags_expert_bias_change(monkeypatch):
    import megatron.core.parallel_state as ps

    from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker

    monkeypatch.setattr(ps, "model_parallel_is_initialized", lambda: False)
    w = object.__new__(ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)
    w.model, w.rank = _Model(0, 1, 2, 0), 0
    tp, d, cov = w.sentinel_weight_digest(learner=True)
    assert tp == 0 and any(k.endswith("expert_bias") for k in d) and cov["expert_bias"] == 1
    w.sentinel_weight_digest(learner=True)  # unchanged: fine
    w.model.module.decoder.layers[0].mlp.router.expert_bias += 1
    with pytest.raises(WeightSentinelError, match="expert_bias changed"):
        w.sentinel_weight_digest(learner=True)


class _Experts(torch.nn.Module):
    """A grouped-MLP experts module holding fp16 [E, N, K] buffers (fp16_moe layout)."""

    def __init__(self, n=3, alias=True, dtype=torch.bfloat16):
        super().__init__()
        self.num_local_experts = n
        for lin in ("linear_fc1", "linear_fc2"):
            m = torch.nn.Module()
            m.num_gemms = n
            for i in range(n):
                m.register_parameter(f"weight{i}", torch.nn.Parameter(torch.randn(4, 2).to(dtype)))
            setattr(self, lin, m)
        if alias:
            from nemo_rl.models.megatron.fp16_moe import _ensure_fp16_expert_weights

            _ensure_fp16_expert_weights(self)


def _expert_model(**kw):
    m = torch.nn.Module()
    m.decoder = torch.nn.Module()
    m.decoder.layers = torch.nn.ModuleList([torch.nn.Module()])
    m.decoder.layers[0].layer_number = 1
    m.decoder.layers[0].mlp = torch.nn.Module()
    m.decoder.layers[0].mlp.experts = _Experts(**kw)
    return m


def _engine_side(model):
    from thundersync_rl.sentinel import compute_tensors

    tp, d = _side(model)
    for key, t, _ in compute_tensors(model):
        d[key] = tensor_digest(t)
    return tp, d


def test_sentinel_checks_the_fp16_expert_buffers_the_kernels_read():
    torch.manual_seed(0)
    learner = _expert_model(alias=False)
    engine = _expert_model(alias=False)
    engine.load_state_dict(learner.state_dict())
    from nemo_rl.models.megatron.fp16_moe import _ensure_fp16_expert_weights

    _ensure_fp16_expert_weights(engine.decoder.layers[0].mlp.experts)
    e = [_engine_side(engine)]
    keys = [k for k in e[0][1] if k.endswith("#buf")]
    assert len(keys) == 6 and "decoder.layers.0.mlp.experts.linear_fc2.weight2#buf" in keys
    cov = compare(e, [_side(learner, dtypes=engine_dtypes(e))], "fp16 experts")
    assert cov["sentinel/tensors"] == 12  # 6 params + 6 compute buffers
    # The old raw-byte refit: bf16 bits written into the fp16 buffer. No error at
    # write time, wrong values; the sentinel names the buffer (and its parameter view).
    buf = engine.decoder.layers[0].mlp.experts._fc1_weight
    raw = learner.decoder.layers[0].mlp.experts.linear_fc1.weight1.detach()
    buf[1].view(torch.int16).copy_(raw.view(torch.int16))
    with pytest.raises(WeightSentinelError, match=r"linear_fc1\.weight1#buf"):
        compare([_engine_side(engine)], [_side(learner, dtypes=engine_dtypes(e))], "raw bytes")


def test_sentinel_catches_a_buffer_that_no_longer_aliases_its_parameter():
    torch.manual_seed(1)
    learner = _expert_model(alias=False)
    engine = _expert_model(alias=False)
    engine.load_state_dict(learner.state_dict())
    exp = engine.decoder.layers[0].mlp.experts
    from nemo_rl.models.megatron.fp16_moe import _ensure_fp16_expert_weights

    _ensure_fp16_expert_weights(exp)
    exp.linear_fc2.weight0.data = exp.linear_fc2.weight0.data.clone()  # view re-pointed
    with torch.no_grad():
        exp.linear_fc2.weight0.add_(1)  # a refit lands in the parameter only
        learner.decoder.layers[0].mlp.experts.linear_fc2.weight0.add_(1)
    e = [_engine_side(engine)]
    from thundersync_rl.sentinel import compute_tensors

    assert sum(1 for *_, a in compute_tensors(engine) if not a) == 1
    with pytest.raises(WeightSentinelError, match=r"linear_fc2\.weight0#buf"):
        compare(e, [_side(learner, dtypes=engine_dtypes(e))], "stale buffer")


def test_learner_fp16_expert_cache_check(monkeypatch):
    from nemo_rl.models.megatron import fp16_moe
    from thundersync_rl.sentinel import check_expert_weight_cache

    m = _expert_model(alias=False)
    lin = m.decoder.layers[0].mlp.experts.linear_fc1
    monkeypatch.setattr(fp16_moe, "_HOOKS", {**fp16_moe._HOOKS, "cache": True})
    fp16_moe._WEIGHT_CACHE.clear()
    fp16_moe._stacked_fp16(tuple(fp16_moe._expert_weights(lin)))
    assert check_expert_weight_cache(m) == 3
    with torch.no_grad():
        lin.weight2.add_(1)  # in place, same pointer, no invalidate_weight_cache()
    with pytest.raises(WeightSentinelError, match="stale cache"):
        check_expert_weight_cache(m)
    fp16_moe.invalidate_weight_cache()
    assert check_expert_weight_cache(m) == 0


def test_coverage_report_lists_checked_and_unchecked():
    from thundersync_rl.sentinel import coverage_report

    text = coverage_report(
        "initial", {"sentinel/tensors": 10, "sentinel/elems": 100, "sentinel/expert_bias": 2,
                    "sentinel/compute_buffers": 4},
        {"compute_buffers": 4, "compute_buffers_aliased": 4, "fp16_params": 3},
        {"learner_fp16_expert_cache": 5}, ["e.x"], [],
    )
    assert "4 expert compute buffers" in text and "not checked" in text and "hi/lo" in text


# ---------------------------------------------------------------------------
# Expert-bias updates with momentum drafts (issue 9)
# ---------------------------------------------------------------------------


def test_momentum_refuses_expert_bias_updates():
    from thundersync_rl.draft_weights import check_no_expert_bias_updates
    from thundersync_rl.grpo_loop import validate_draft_weights

    m = torch.nn.Module()
    m.router = torch.nn.Module()
    m.router.register_buffer("expert_bias", torch.zeros(2))
    m.router.config = SimpleNamespace(moe_router_bias_update_rate=1e-3, moe_router_enable_expert_bias=True)
    with pytest.raises(NotImplementedError, match="expert_bias"):
        check_no_expert_bias_updates(m)
    m.router.config.moe_router_bias_update_rate = 0.0
    check_no_expert_bias_updates(m)

    ts = SimpleNamespace(
        draft_weights="momentum", speculate=True, verify_mode="block", verify_source="engine",
        verify_on="learner", draft_lookahead=1, draft_after_verify=False, fused_verify_train=False,
    )
    mc = SimpleNamespace(
        thundersync=ts,
        policy={"megatron_cfg": {"optimizer": {"use_distributed_optimizer": True}, "moe_router_bias_update_rate": 1e-3}},
    )
    with pytest.raises(ValueError, match="expert_bias"):
        validate_draft_weights(mc)
    mc.policy["megatron_cfg"]["moe_router_bias_update_rate"] = 0.0
    validate_draft_weights(mc)


# ---------------------------------------------------------------------------
# Non-finite verification inputs (issue 6)
# ---------------------------------------------------------------------------


def _rows(g, V=16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    lp = torch.log_softmax(torch.randn(g + 1, V, generator=gen), -1)
    lq = torch.log_softmax(torch.randn(g, V, generator=gen), -1)
    draft = torch.randint(0, V, (g,), generator=gen)
    return lp, lq, draft


def _stats(lp, lq, draft):
    g = draft.numel()
    ar = torch.arange(g)
    return lp[ar, draft].clone(), lq[ar, draft].clone()


def test_q_zero_draft_token_is_an_error_not_a_silent_cut():
    lp, lq, draft = _rows(6)
    lpx, lqx = _stats(lp, lq, draft)
    lqx[3] = float("-inf")  # the engine's logged q of the drafted token is 0
    z = torch.zeros(7)
    with pytest.raises(NonFiniteVerification, match="engine log q .* positions \\[3\\]"):
        block_verify_stats_lazy(lpx, lqx, z, z[:6], lambda i: lp[i], lambda i: lq[i], torch.Generator())
    with pytest.raises(NonFiniteVerification):
        block_verify_stats_lazy_grid(
            lpx, lqx, z, z[:6], lambda i: lp[i], lambda i: lq[i], torch.Generator(), (1.0, 1.5)
        )
    lq2 = lq.clone()
    lq2[3, draft[3]] = float("-inf")
    with pytest.raises(NonFiniteVerification):
        block_verify(lp, lq2, draft, torch.Generator())


def test_nan_learner_logits_are_an_error_not_a_device_assert():
    lp, lq, draft = _rows(5)
    p_logits = lp.clone()
    p_logits[5, :] = float("nan")  # the bonus row
    with pytest.raises(NonFiniteVerification, match="logsumexp"):
        block_verify_logits(p_logits, lq, draft, torch.Generator())
    lpx, lqx = _stats(lp, lq, draft)
    lpx[1] = float("nan")
    z = torch.zeros(6)
    with pytest.raises(NonFiniteVerification, match="learner log p"):
        block_verify_stats_lazy(lpx, lqx, z, z[:5], lambda i: lp[i], lambda i: lq[i], torch.Generator())
    # NaN only in the bonus row of log-prob rows (engine path, lse = 0): the
    # emission check catches it before multinomial.
    lpn = lp.clone()
    lpn[5] = float("nan")
    lpx, lqx = _stats(lp, lq, draft)
    lqx[:] = lpx  # accept everything -> bonus emission
    with pytest.raises(NonFiniteVerification, match="emission"):
        block_verify_stats_lazy(lpx, lqx, z, z[:5], lambda i: lpn[i], lambda i: lq[i], torch.Generator())


def test_p_zero_draft_token_is_a_legitimate_rejection():
    lp, lq, draft = _rows(4, seed=3)
    lpx, lqx = _stats(lp, lq, draft)
    lp2 = lp.clone()
    lp2[2, draft[2]] = float("-inf")
    lp2[2] = torch.log_softmax(lp2[2], -1)
    lpx2, _ = _stats(lp2, lq, draft)
    z = torch.zeros(5)
    for s in range(20):
        tau, y, logp, _ = block_verify_stats_lazy(
            lpx2, lqx, z, z[:4], lambda i: lp2[i], lambda i: lq[i], torch.Generator().manual_seed(s)
        )
        assert tau <= 2  # never keeps a p = 0 token
        assert torch.isfinite(logp).all()


def test_finite_inputs_unchanged_by_the_checks():
    # Same decisions as before the checks (the checks draw no randomness).
    lp, lq, draft = _rows(8, seed=5)
    out = [block_verify(lp, lq, draft, torch.Generator().manual_seed(s)) for s in range(30)]
    lpx, lqx = _stats(lp, lq, draft)
    z = torch.zeros(9)
    lazy = [
        block_verify_stats_lazy(lpx, lqx, z, z[:8], lambda i: lp[i], lambda i: lq[i], torch.Generator().manual_seed(s))[:2]
        for s in range(30)
    ]
    assert out == lazy


# ---------------------------------------------------------------------------
# collect_drafts never uses streamed frames for an unrecorded draft (issue 8)
# ---------------------------------------------------------------------------


class _Stream:
    def __init__(self, rid):
        self.request_id = rid


def _collect_worker():
    from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker

    w = object.__new__(ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    w._inference_loop = loop
    w.inference_client = SimpleNamespace(aborted=[], abort_request=lambda rid: w.inference_client.aborted.append(rid))
    w.dynamic_inference_engine = None
    return w, loop


def test_unrecorded_draft_waits_for_its_final_frame(monkeypatch):
    import thundersync_rl.worker as worker_mod

    monkeypatch.setattr(worker_mod.ray, "get", lambda refs: [])
    w, loop = _collect_worker()
    # Draft 0: recorded (frames skipped a token after an eviction; the record is right).
    # Draft 1: finished in the engine after the records were read: frames are
    #          [7, 9] (a skipped token), the final frame carries [7, 8, 9].
    # Draft 2: not admitted yet (no record, no frame): empty.
    w._draft_tokens = [[1, 3], [7, 9], []]
    w._draft_logprobs = [[-0.1, -0.3], [-0.7, -0.9], []]
    w._draft_done = [False, False, False]
    w._draft_streams = {0: _Stream(10), 1: _Stream(11), 2: _Stream(12)}

    def final_frame():
        w._draft_tokens[1][:] = [7, 8, 9]
        w._draft_logprobs[1] = [-0.7, -0.8, -0.9]
        w._draft_done[1] = True

    loop.call_soon_threadsafe(loop.call_later, 0.3, final_frame)
    out = w.collect_drafts(records={10: ([1, 2, 3], [-0.1, -0.2, -0.3])})
    assert out[0][0].tolist() == [1, 2, 3]
    assert out[1][0].tolist() == [7, 8, 9] and out[1][1]
    assert out[2][0].tolist() == []
    assert sorted(w.inference_client.aborted) == [10, 12]  # the finished one is not aborted
    st = w.draft_collect_stats()
    assert st["unrecorded"] == 2 and st["waited_final"] == 1 and st["unrecorded_empty"] == 1


def test_unrecorded_running_draft_is_an_error(monkeypatch):
    import thundersync_rl.worker as worker_mod

    monkeypatch.setattr(worker_mod.ray, "get", lambda refs: [])
    monkeypatch.setenv("THUNDERSYNC_COLLECT_FINAL_TIMEOUT_S", "0.1")
    w, _ = _collect_worker()
    w._draft_tokens, w._draft_logprobs = [[4, 5]], [[-0.4, -0.5]]
    w._draft_done, w._draft_streams = [False], {0: _Stream(1)}
    with pytest.raises(RuntimeError, match="final frame did not arrive"):
        w.collect_drafts(records={})


# ---------------------------------------------------------------------------
# Multi-turn rows are prefix extensions (coordinator: longctx finding 7/14)
# ---------------------------------------------------------------------------


class _Gen:
    def __init__(self, outs):
        self.outs = list(outs)

    async def generate_async(self, data, greedy=False):
        n = int(data["input_lengths"][0])
        tail = torch.tensor(self.outs.pop(0))
        yield 0, {
            "output_ids": torch.cat([data["input_ids"][0, :n], tail]).view(1, -1),
            "generation_lengths": torch.tensor([tail.numel()]),
        }


def _call(gen, ids):
    async def go():
        data = {"input_ids": torch.tensor([ids]), "input_lengths": torch.tensor([len(ids)])}
        return [r async for r in gen.generate_async(data)]

    return asyncio.run(go())


def test_prefix_guard_accepts_incremental_turns_and_rejects_rerendered_history():
    g = PrefixGuardGeneration(_Gen([[5, 6], [9]]), row=3)
    _call(g, [1, 2])  # turn 1: prompt [1, 2] -> generated [5, 6]
    _call(g, [1, 2, 5, 6, 7, 8])  # turn 2: + observation [7, 8]
    g.final_check([{"token_ids": torch.tensor([1, 2, 5, 6, 7, 8, 9])}])
    h = PrefixGuardGeneration(_Gen([[5, 6], [9]]), row=4)
    _call(h, [1, 2])
    with pytest.raises(RowPrefixError, match="row 4 turn 1"):
        _call(h, [1, 2, 6, 7, 8])  # history re-rendered: generated tokens changed
    # The rollout loop swallows per-turn generation errors: final_check re-raises.
    with pytest.raises(RowPrefixError):
        h.final_check([{"token_ids": torch.tensor([1, 2, 5, 6])}])
    with pytest.raises(RowPrefixError, match="first difference at position 1"):
        check_row_extends(torch.tensor([1, 2, 3]), torch.tensor([1, 9, 3, 4]), "x")


def test_template_that_truncates_history_needs_the_flag():
    from thundersync_rl.grpo_loop import check_multi_turn_template

    tok = SimpleNamespace(chat_template="{% if truncate_history_thinking %}x{% endif %}")
    mc = SimpleNamespace(grpo=SimpleNamespace(max_rollout_turns=3), policy={"tokenizer": {}})
    with pytest.raises(ValueError, match="truncate_history_thinking=false"):
        check_multi_turn_template(mc, tok)
    mc.policy["tokenizer"]["chat_template_kwargs"] = {"truncate_history_thinking": False}
    check_multi_turn_template(mc, tok)
    mc.grpo.max_rollout_turns = 1
    mc.policy["tokenizer"] = {}
    check_multi_turn_template(mc, tok)  # single turn: no history is rendered


def test_deterministic_training_pins_mamba_ssm_backward_autotuners():
    # Imported before deterministic mode is set (as in the workers): every candidate kept,
    # so each process would autotune its own tile -> run-to-run gradient bits differ.
    import sys

    pytest.importorskip("mamba_ssm")
    from mamba_ssm.ops.triton import ssd_combined  # noqa: F401
    from mamba_ssm.utils.determinism import set_deterministic_mode
    from triton.runtime.autotuner import Autotuner

    from nemo_rl.models.megatron.mamba_ssd_parity import pin_mamba_ssm_autotuners

    def multi():
        return [
            n for m, mod in list(sys.modules.items()) if m.startswith("mamba_ssm.ops") and mod
            for n, a in vars(mod).items() if isinstance(a, Autotuner) and len(a.configs) > 1
        ]

    set_deterministic_mode(True)
    try:
        before = multi()
        n = pin_mamba_ssm_autotuners()
        assert n == len(before) and multi() == []
    finally:
        set_deterministic_mode(None)
