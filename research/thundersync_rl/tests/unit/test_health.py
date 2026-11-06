"""Per-step health line (health.py): each silent-waste pattern raises its flag."""

from thundersync_rl.health import annotate, step_health


def _flags(step, m, prev=None, mbs=None, expect=None):
    return [f.split("=")[0].split("(")[0] for f in step_health(step, m, prev, mbs, expect)[0]]


def test_clean_ts_step_is_ok():
    m = {
        "lanes/refits": 3.0, "lanes/fresh_refit": 0.0, "spec/engine_q_dropped": 0.0,
        "spec/null_fresh_slots": 0.0, "learner/filler_frac": 0.0, "learner/dispatches": 17.0,
        "learner/real_microbatches": 136.0, "train/nonzero_adv_rows": 272.0,
        "sentinel/time_s": 1.5, "spec/full_accept": 0.7, "time/iteration": 150.0,
    }
    line = annotate(2, m, {"lanes/refits": 2.0}, mbs=2)
    assert line.startswith("[health step 2] OK"), line
    assert m["health/warn"] == 0.0 and m["health/flags"] == ""


def test_dropped_drafts_decoding_fresh_flagged():
    # bh-cc step 2: 55 drafts never admitted -> dropped -> 34 fresh rows, extra lane refit.
    m = {"lanes/refits": 4.0, "lanes/fresh_refit": 1.0, "spec/engine_q_dropped": 55.0,
         "spec/null_fresh_slots": 34.0, "spec/full_accept": 0.27}
    f = _flags(2, m, {"lanes/refits": 2.0})
    assert {"q_dropped", "fresh_rows", "lane_refits+2"} <= set(f), f


def test_lane_fresh_refit_without_fresh_rows_flagged():
    assert "lane_fresh_refit" in _flags(3, {"lanes/fresh_refit": 1.0, "lanes/refits": 4.0})
    # Step 0 always refits the cohort lane fresh: not a flag.
    assert _flags(0, {"lanes/fresh_refit": 1.0, "lanes/refits": 1.0}) == []


def test_async1_split_dispatch_flagged():
    m = {"learner/dispatches": 2.0, "timeline/dispatches": [[0, 14, 8, 1, 0], [14, 120, 504, 1, 0]]}
    assert _flags(1, m) == ["dispatches"]
    assert _flags(0, m) == []  # step 0 of async1 has no stale rollouts yet


def test_ts_arm_many_dispatches_not_flagged_but_fragments_reported():
    disp = [[0, 2, 32], [2, 4, 32], [4, 5, 8]]
    m = {"learner/dispatches": 3.0, "timeline/dispatches": disp, "tail/end_s": 100.0}
    flags, info = step_health(1, m)
    assert flags == [] and "disp=3,frag=1" in info


def test_filler_and_mbs_cap_flagged():
    m = {"learner/filler_frac": 0.13, "learner/filler_microbatches": 24.0,
         "learner/real_microbatches": 160.0, "learner/skipped_microbatches": 100.0,
         "train/nonzero_adv_rows": 170.0, "train/rows": 512.0}
    assert {"filler_frac", "mbs_eff"} <= set(_flags(1, m, mbs=2))
    # MBS 1: no MBS-efficiency check.
    assert "mbs_eff" not in _flags(1, m, mbs=1)


def test_sentinel_drain_prefix_cache_plans():
    m = {"sentinel/time_s": 3.1, "learner/drain_wait_s": 12.0, "spec/prefix_cache_eligible": 100.0,
         "spec/prefix_cache_denied": 20.0, "spec/prefix_cache_evicted": 1.0, "spec/plans_dropped": 2.0,
         "spec/route_truncated_drafts": 1.0}
    assert set(_flags(1, m)) == {
        "sentinel", "drain_wait", "pc_denied", "pc_evicted", "plans_dropped", "route_truncated"
    }


def test_missing_and_nan_metrics_are_ignored():
    assert step_health(1, {"learner/filler_frac": float("nan"), "health/flags": "x"}) == ([], [])


def test_mbs_eff_uses_all_rows_without_skip():
    # No zero-advantage skip: every trained row is live (dr-ab: 512 rows in 256 mb = MBS 2).
    m = {"learner/real_microbatches": 256.0, "learner/skipped_microbatches": 0.0,
         "train/nonzero_adv_rows": 200.0, "train/rows": 512.0}
    assert "mbs_eff" not in _flags(1, m, mbs=2)


def test_unknown_thundersync_keys_rejected(monkeypatch):
    from types import SimpleNamespace

    import pytest

    from thundersync_rl.grpo_loop import ThunderSyncConfig, validate_known_keys

    validate_known_keys(SimpleNamespace(thundersync=ThunderSyncConfig(verify_yield=True)))
    bad = SimpleNamespace(thundersync=ThunderSyncConfig(verify_yeild=True, remainder_lookup=True))
    with pytest.raises(ValueError, match=r"\['remainder_lookup', 'verify_yeild'\]"):
        validate_known_keys(bad)
    monkeypatch.setenv("THUNDERSYNC_ALLOW_UNKNOWN_KEYS", "1")
    validate_known_keys(bad)  # warning only


def test_enabled_features_that_did_nothing_and_fp16_range():
    m = {"learner/yield_inlined": 0.0, "engine/anchor_missing": 2.0, "fp16/act_overflow": 7.0,
         "fp16/weight_amax": 7.0e4, "learner/head_materialized_dispatch_grad": 1.0}
    f = set(_flags(1, m, expect={"verify_yield": True}))
    assert f == {"yield_inlined", "anchor_missing", "fp16_overflow", "fp16_weight_amax",
                 "head_nograd_materialized"}
    assert "yield_inlined" not in _flags(1, m, expect={})  # verify_yield off: no expectation


def test_rows_per_live_mb_metric():
    m = {"learner/real_microbatches": 128.0, "learner/skipped_microbatches": 10.0,
         "train/nonzero_adv_rows": 168.0}
    annotate(3, m, mbs=2)
    assert abs(m["learner/rows_per_live_mb"] - 168 / 128) < 1e-9
