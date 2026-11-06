"""Guards that replaced silent fallbacks (research/silent_bug_sweep.md)."""

import torch

from nemo_rl.models.megatron.prefix_state_cache import _position_embedding_type


class _Wrap(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.module = inner


def test_rope_guard_unwraps_every_level():
    model = torch.nn.Linear(2, 2)
    model.position_embedding_type = "rope"
    assert _position_embedding_type(_Wrap(_Wrap(model))) == "rope"  # DDP(Float16Module(model))
    plain = torch.nn.Linear(2, 2)
    plain.position_embedding_type = "none"
    assert _position_embedding_type(_Wrap(_Wrap(plain))) == "none"
    assert _position_embedding_type(_Wrap(torch.nn.Linear(2, 2))) is None


def test_fp16_range_counter_counts_overflow_and_skips_fp16():
    from nemo_rl.models.megatron import fp16_gemm as G

    G.range_monitor(True)
    try:
        G.range_take()
        G.range_count(torch.tensor([1.0, 7e4, -1e5, 3.0]))
        G.range_count(torch.tensor([1.0], dtype=torch.float16))  # already fp16: not a cast site
        assert G.range_take() == {"overflow": 2.0, "checked": 4.0}
        assert G.range_take() == {"overflow": 0.0, "checked": 0.0}
    finally:
        G.range_monitor(False)
    G.range_count(torch.tensor([1e6]))  # off: not counted
    assert G.range_take()["overflow"] == 0.0


def test_precision_guards_reject_silent_divergence():
    import pytest

    from thundersync_rl.grpo_loop import validate_precision_guards

    ok = {"megatron_cfg": {"forward_gemm_precision": "fp16", "batch_invariant_mode": True},
          "generation": {"refit_transport": "mcore", "mcore_generation_config": {}}}
    validate_precision_guards(ok)
    for bad in (
        {"megatron_cfg": {**ok["megatron_cfg"], "moe_shared_expert_overlap": True}},
        {"megatron_cfg": {**ok["megatron_cfg"], "use_grouped_gemm_for_shared_expert": True}},
        {"generation": {"refit_transport": "nccl_reshard", "mcore_generation_config": {}}},
        {"generation": {"refit_transport": "mcore",
                        "mcore_generation_config": {"mamba_inference_ssm_states_dtype": "bfloat16"}}},
    ):
        with pytest.raises(ValueError, match="fp16 GEMM recipe"):
            validate_precision_guards({**ok, **bad})
    # fp32 overrides and configs without the recipe pass.
    validate_precision_guards({**ok, "generation": {"mcore_generation_config": {
        "mamba_inference_ssm_states_dtype": "float32"}}})
    validate_precision_guards({"megatron_cfg": {"moe_shared_expert_overlap": True}})


def test_deferred_head_materialization_counted_and_strict(monkeypatch):
    import pytest

    from nemo_rl.models.megatron import deferred_head as D

    D.materialized_take()
    D._note_materialized(torch.ops.aten.add.Tensor, "function", True)
    D._note_materialized(torch.ops.aten.add.Tensor, "dispatch", True, below_autograd=True)
    assert D.materialized_take() == {"function": 1, "dispatch": 1, "dispatch_grad": 1}
    monkeypatch.setenv("THUNDERSYNC_STRICT_DEFERRED_HEAD", "1")
    with pytest.raises(RuntimeError, match="no gradient"):
        D._note_materialized(torch.ops.aten.add.Tensor, "dispatch", True, below_autograd=True)
    D._note_materialized(torch.ops.aten.add.Tensor, "dispatch", False, below_autograd=True)  # no grad: ok


def test_draft_capacity_zero_is_not_cached(monkeypatch):
    from types import SimpleNamespace

    from thundersync_rl import speculative as S

    monkeypatch.setattr(S.ray, "get", lambda x: x)
    answers = iter([0, 64])
    rank0 = SimpleNamespace(engine_request_capacity=SimpleNamespace(remote=lambda: next(answers)))
    s = object.__new__(S.SpeculativeGeneration)
    s._check_draft_capacity(rank0, 100)  # engine not up: 0, guard skipped, not cached
    import pytest

    with pytest.raises(RuntimeError, match="primary drafts per cohort"):
        s._check_draft_capacity(rank0, 100)  # asked again: 64 < 100


def test_verify_yield_rejects_pipeline_parallel():
    import pytest
    from types import SimpleNamespace

    from thundersync_rl.grpo_loop import ThunderSyncConfig, validate_verify_yield

    cfg = SimpleNamespace(
        thundersync=ThunderSyncConfig(verify_yield=True, sticky_routing=True),
        policy={"megatron_cfg": {"pipeline_model_parallel_size": 2}},
    )
    with pytest.raises(ValueError, match="pipeline_model_parallel_size > 1"):
        validate_verify_yield(cfg)


def test_crosscheck_without_native_refit_raises(monkeypatch):
    import pytest
    from types import SimpleNamespace

    from thundersync_rl import refit_m2n

    monkeypatch.setenv("THUNDERSYNC_REFIT_CROSSCHECK", "1")
    with pytest.raises(RuntimeError, match="nothing to crosscheck"):
        refit_m2n.install_crosscheck(None, SimpleNamespace(uses_native_refit=False), None, None)
