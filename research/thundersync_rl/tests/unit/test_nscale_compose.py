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
"""ts-nano-scale integration: compositions of the merged opt-in features.

1. Exact top-p (ts-topp) + the 53-bit engine sampler (ts-exact-harden, default): the nucleus
   wrapper is installed at engine build, the race later by the driver. Before the fix the race
   replaced the whole kernel chain and sampled the UNtruncated row (the in-place mask never ran).
2. exact_int + exact_top_p: refused until the composition is wired.
"""

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.megatron import exact_nucleus
from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _worker(sampling):
    w = object.__new__(ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)
    w.dynamic_inference_engine = SimpleNamespace(controller=SimpleNamespace(_sampling=sampling))
    w.rank = 0
    return w


@pytest.fixture
def nucleus_on():
    exact_nucleus.configure(0.9)
    yield
    exact_nucleus.configure(None)


def test_wide_sampler_goes_inside_the_nucleus_wrapper(nucleus_on):
    seen = []

    def engine_kernel(logits, n, context, **kw):
        seen.append("engine")
        return torch.zeros(n, dtype=torch.int64)

    sampling = SimpleNamespace(sample_kernel=engine_kernel)
    w = _worker(sampling)
    w._install_nucleus_sampler(w.dynamic_inference_engine.controller)
    nucleus_kernel = sampling.sample_kernel
    assert sampling._nucleus_inner is engine_kernel
    assert w.install_wide_sampler(salt=7) is not None
    # The mask stays the outer kernel; the race replaced the engine kernel inside it.
    assert sampling.sample_kernel is nucleus_kernel
    assert sampling._nucleus_inner is not engine_kernel
    assert sampling._wide_installed


def test_wide_sampler_without_top_p_replaces_the_kernel():
    exact_nucleus.configure(None)

    def engine_kernel(logits, n, context, **kw):
        return torch.zeros(n, dtype=torch.int64)

    sampling = SimpleNamespace(sample_kernel=engine_kernel)
    w = _worker(sampling)
    w._install_nucleus_sampler(w.dynamic_inference_engine.controller)  # no-op when off
    assert sampling.sample_kernel is engine_kernel
    w.install_wide_sampler(salt=7)
    assert sampling.sample_kernel is not engine_kernel
    assert not getattr(sampling, "_nucleus_installed", False)


@cuda
def test_wide_race_samples_only_nucleus_tokens(nucleus_on):
    """Real chain on the GPU: nucleus mask (in place) -> 53-bit race; every draw in N."""
    torch.manual_seed(0)
    n, V = 64, 4096
    dev = "cuda"
    logits = (torch.randn(n, V, device=dev) * 3.0).float()
    ref = logits.clone()

    def engine_kernel(*a, **k):  # must not be reached (no top-k/top-p requests)
        raise AssertionError("fallback kernel used")

    sampling = SimpleNamespace(sample_kernel=engine_kernel)
    w = _worker(sampling)
    w._install_nucleus_sampler(w.dynamic_inference_engine.controller)
    w.install_wide_sampler(salt=11)
    lp = torch.log_softmax(ref, dim=-1)
    keep = exact_nucleus.plain_mask(exact_nucleus.mass_weights(lp), 0.9)
    hits = 0
    for step in range(16):
        lg = ref.clone()
        ctx = SimpleNamespace(
            gpu_view=SimpleNamespace(
                request_kv_length_offsets=torch.full((n,), step, device=dev, dtype=torch.int32),
                request_query_lengths=torch.ones(n, device=dev, dtype=torch.int32),
                temperature=torch.ones(n, device=dev),
            ),
            paused_request_count=0,
            request_ids=torch.arange(n, dtype=torch.int64),
        )
        tok = sampling.sample_kernel(
            lg, n, ctx, gather_indices=torch.arange(n, device=dev), no_top_k=True, no_top_p=True
        )
        assert bool(keep.gather(1, tok.view(-1, 1)).all()), "a token outside the nucleus was drawn"
        hits += n
    assert sampling._wide_stats["calls"] == 16 and sampling._wide_stats["fallback_calls"] == 0
    assert hits == 16 * n


def _salts(segment, sampler):
    import test_speculative_driver as drv

    got = []
    name = "install_wide_sampler" if sampler == "wide" else "install_exact_int_sampler"
    drv._spec(
        {},
        {name: lambda salt, **kw: got.append(salt) or {}},  # wide: also key64 (ts-lawfix)
        engine_sampler=sampler,
        seed=42,
        seed_stream="1",
        sampler_segment=segment,
    )
    return got


@pytest.mark.parametrize("sampler", ["wide", "exact_int"])
def test_resumed_run_does_not_reuse_sampler_keys(monkeypatch, sampler):
    """Engine request ids restart at 0 in a resumed process: its sampler salt must differ
    from the saving run's (else the same keyed words drive the resumed rollouts), and a
    fresh run keeps the pre-nscale salt bitwise."""
    import zlib

    import thundersync_rl.speculative as spec_mod

    monkeypatch.setattr(spec_mod.ray, "get", lambda x: x)
    s0, s5, s9 = _salts(0, sampler), _salts(5, sampler), _salts(9, sampler)
    old = zlib.crc32(f"{sampler}/42/1".encode()) | ((42 & 0x7FFFFFFF) << 32)
    assert s0 == [old]
    assert len({s0[0], s5[0], s9[0]}) == 3


# --- sampled checks (checks_sched.py) ----------------------------------------------------------------------------


def test_check_schedule_full_on_one_step_in_k_and_restores_env():
    from thundersync_rl import checks_sched as cs

    env = {
        "THUNDERSYNC_CHECK_EVERY": "10",
        "THUNDERSYNC_MISMATCH_CHECK": "1",
        "THUNDERSYNC_ENGINE_Q_CHECK": "1",
        "THUNDERSYNC_LAW_CHECK": "1",
        "THUNDERSYNC_TRAIN_LP_CHECK": "1",
        "THUNDERSYNC_CHECK_OFF_ENV": "THUNDERSYNC_TRAIN_LP_CHECK=1,THUNDERSYNC_LAW_CHECK=1,"
        "THUNDERSYNC_MISMATCH_CHECK=1,THUNDERSYNC_MISMATCH_CHECK_ROWS=16",
    }
    sched = cs.CheckSchedule.from_env(env)
    assert sched.every == 10 and sched.phase == 5
    assert [s for s in range(30) if sched.is_full(s)] == [5, 15, 25]
    seen = []
    target = dict(env)
    import os as _os

    saved = {k: _os.environ.get(k) for k in cs.CHECK_VARS}
    try:
        for step in range(7):
            full = sched.apply(step, [seen.append])
            on = {k: _os.environ.get(k) for k in cs.CHECK_VARS}
            if full:
                assert on["THUNDERSYNC_ENGINE_Q_CHECK"] == "1" and on["THUNDERSYNC_MISMATCH_CHECK_ROWS"] is None
            else:
                assert on["THUNDERSYNC_ENGINE_Q_CHECK"] is None and on["THUNDERSYNC_MISMATCH_CHECK_ROWS"] == "16"
                assert on["THUNDERSYNC_TRAIN_LP_CHECK"] == "1"
        # Workers are told only on a switch: steps 0 (off), 5 (full), 6 (off).
        assert len(seen) == 3
    finally:
        cs.apply_env(saved)
    assert cs.CheckSchedule.from_env({"THUNDERSYNC_CHECK_EVERY": "1"}) is None
    with pytest.raises(ValueError):
        cs.parse_env_list("NOT_A_CHECK=1")
    _ = target


# Super-L22 8+8 ts step 2 (runs_ab ...-prod-par-tp-pc0-pool, sticky begin_sticky path): verify0 at the top
# of the step (time/verify, tail/verify0_s inside it), the deferred draft handout, then the step body;
# the refit, the cut and the finish run inside the body (values rounded to ms; body = rollout + post).
_PC0_STEP2 = {
    "time/iteration": 148.442, "time/step_total": 100.588, "time/rollout": 59.570,
    "time/post_rollout_to_step": 41.018, "time/verify": 39.596, "time/draft_handout_wait": 7.990,
    "tail/verify0_s": 36.906, "time/refit": 1.836, "time/draft_cut": 0.817, "time/draft_start": 0.222,
    "time/finish_step": 2.571, "time/mismatch_check": 0.897,
}


def _parts(d):
    return (d["decomp/verify0_s"] + d["decomp/rollout_rest_s"] + d["decomp/post_train_s"]
            + d["decomp/checks_s"] + d["decomp/outside_s"])


def test_step_decomposition_names_the_step_top_phases():
    from thundersync_rl.checks_sched import STEP_TOP_KEYS, step_decomposition

    m = dict(_PC0_STEP2)
    d = step_decomposition(m)
    assert abs(_parts(d) - m["time/iteration"]) < 1e-9
    # Outside the body = verify0 + handout + 0.26 s; the old split named refit / cut / start / finish
    # (body or offsets) and missed verify + handout: unattributed 42.4 s, attributed_frac 0.71.
    assert abs(d["decomp/step_top_s"] - (39.596 + 7.990)) < 1e-9
    assert abs(d["decomp/unattributed_s"] - (148.442 - 100.588 - 39.596 - 7.990)) < 1e-9
    assert d["decomp/unattributed_s"] < 0.3 and d["decomp/attributed_frac"] > 0.99
    # verify0 ran at the top of the step: not carved out of the rollout window.
    assert d["decomp/verify0_s"] == 0.0 and d["decomp/rollout_rest_s"] == m["time/rollout"]
    # Body phases never reduce the unattributed time.
    for k in ("time/refit", "time/draft_cut", "time/draft_start", "time/finish_step"):
        assert step_decomposition({**m, k: 30.0})["decomp/unattributed_s"] == d["decomp/unattributed_s"]
    # A step-top phase not timed (here the handout) is unattributed, not hidden.
    d2 = step_decomposition({**m, "time/draft_handout_wait": 0.0})
    assert abs(d2["decomp/unattributed_s"] - d["decomp/unattributed_s"] - 7.990) < 1e-9
    # The caller names a refit that ran at the step top (non-sticky path) / the lagged first rollouts.
    d3 = step_decomposition({**m, "time/verify": 37.76}, STEP_TOP_KEYS + ("time/refit",))
    assert abs(d3["decomp/unattributed_s"] - d["decomp/unattributed_s"]) < 1e-9
    d4 = step_decomposition({**m, "time/verify": 30.0, "time/first_rollout": 9.596})
    assert abs(d4["decomp/unattributed_s"] - d["decomp/unattributed_s"]) < 1e-9
    assert step_decomposition({}) == {}


def test_step_decomposition_streaming_verify0_and_lanes():
    from thundersync_rl.checks_sched import step_decomposition

    # Streaming verify0: its calls run inside the body (rollout window); time/verify is the schedule only.
    m = {
        "time/iteration": 107.2, "time/step_total": 103.6, "time/rollout": 71.7,
        "time/post_rollout_to_step": 31.9, "tail/verify0_s": 50.3, "time/verify": 0.4,
        "time/draft_handout_wait": 3.1, "time/finish_step": 0.4, "time/mismatch_check": 1.5,
    }
    d = step_decomposition(m, verify0_in_body=True)
    assert abs(_parts(d) - 107.2) < 1e-9
    assert d["decomp/verify0_s"] == 50.3 and abs(d["decomp/rollout_rest_s"] - 21.4) < 1e-9
    assert abs(d["decomp/unattributed_s"] - 0.1) < 1e-9
    # Named step-top time beyond the outside time is clipped (no negative unattributed).
    d = step_decomposition({**m, "time/verify": 9.0}, verify0_in_body=True)
    assert d["decomp/step_top_s"] == d["decomp/outside_s"] and d["decomp/unattributed_s"] == 0.0
    # Lanes: the cohort verification and the refits run outside the body; keys are counted once.
    m = {"time/iteration": 100.0, "time/step_total": 60.0, "time/rollout": 30.0,
         "time/post_rollout_to_step": 30.0, "time/verify": 35.0, "time/refit": 4.0}
    d = step_decomposition(m, ("time/verify", "time/refit", "time/verify"))
    assert abs(d["decomp/unattributed_s"] - 1.0) < 1e-9 and abs(_parts(d) - 100.0) < 1e-9


def test_emission_sample_is_a_function_of_the_key_only():
    from thundersync_rl.worker import _emit_sampled

    keys = [f"p{i}/s{j}" for i in range(50) for j in range(8)]
    a = [_emit_sampled(k, 4) for k in keys]
    assert a == [_emit_sampled(k, 4) for k in keys]
    assert 0.15 < sum(a) / len(a) < 0.35
    assert all(_emit_sampled(k, 1) for k in keys)


def test_aep_refuses_untested_merged_compositions():
    import thundersync_rl.grpo_loop as gl

    src = open(gl.__file__).read()
    i = src.index("def validate_anchor_epoch")
    body = src[i : src.index("\ndef ", i + 10)]
    assert "exact_top_p / sticky_nucleus with anchor_epoch is untested" in body
    assert "engine_sampler=exact_int with anchor_epoch is untested" in body


def test_plain_engine_sampler_wide_for_baselines(monkeypatch):
    """Fair A/B: async1 / sync install the same 53-bit race (ported from ts-syncuni), with a
    per-segment salt on resume; the baseline arm configs default to it."""
    import os
    from types import SimpleNamespace

    import yaml

    import thundersync_rl.grpo_loop as gl

    got = []
    gen = SimpleNamespace(_policy=SimpleNamespace(worker_group=SimpleNamespace(
        run_all_workers_single_data=lambda m, salt, **kw: got.append((m, salt, kw)) or [{}])))
    monkeypatch.setattr(gl.ray, "get", lambda x: x)
    cfg = SimpleNamespace(grpo=SimpleNamespace(seed=42))
    gl.install_plain_wide_sampler(gen, cfg)
    gl.install_plain_wide_sampler(gen, cfg, segment=6)
    assert got[0][0] == "install_wide_sampler" and got[0][1] != got[1][1]
    assert got[0][2] == {"key64": True}  # ts-lawfix default: 64-bit noise key
    gl.install_plain_wide_sampler(gen, SimpleNamespace(grpo=cfg.grpo, thundersync=SimpleNamespace(wide_sampler_key64=False)))
    assert got[2][2] == {"key64": False}
    assert gl.ThunderSyncConfig().plain_engine_sampler == "engine"
    here = os.path.join(os.path.dirname(gl.__file__), "..", "configs", "ab")
    for arm in ("ab_async1.yaml", "ab_sync.yaml"):
        assert yaml.safe_load(open(os.path.join(here, arm)))["thundersync"]["plain_engine_sampler"] == "wide"


def test_checkpoints_refuse_async_save_with_learner_expandable_segments(monkeypatch):
    """ns-resume: the async-save worker's CUDA IPC fails on expandable-segment memory (pidfd_getfd)."""
    from test_run_state import _master

    from thundersync_rl.grpo_loop import validate_run_control

    m = _master()
    m.policy = {"megatron_cfg": {"checkpoint": {"async_save": True}}}
    monkeypatch.delenv("THUNDERSYNC_LEARNER_EXPANDABLE_SEGMENTS", raising=False)
    with pytest.raises(ValueError, match="async_save must be false"):
        validate_run_control(m)
    m.policy["megatron_cfg"]["checkpoint"]["async_save"] = False
    validate_run_control(m)
    m.policy["megatron_cfg"]["checkpoint"]["async_save"] = True
    monkeypatch.setenv("THUNDERSYNC_LEARNER_EXPANDABLE_SEGMENTS", "0")
    validate_run_control(m)
