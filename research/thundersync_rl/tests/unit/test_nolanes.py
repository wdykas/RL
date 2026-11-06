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
"""nolanes mode (nolanes.py, nolanes_engine.py): the config validator (every refusal), version tags across a
refit (q rebuilt from the tag's head is bitwise the logged q; a wrong / shifted tag is not), the noise-key
registry (E4), output rows (E7 negative), provenance (E6/E9)."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from thundersync_rl import nolanes
from thundersync_rl.nolanes_engine import KeyRegistry, NolanesEngineMixin, VersionTagLog


def _valid():
    return {
        "thundersync": {"mode": "nolanes", "k_drafts": 1.5},
        "grpo": {"num_prompts_per_step": 16, "max_rollout_turns": 1, "use_dynamic_sampling": False},
        "async_rl": {
            "sampler": {"name": "in_order", "max_lookahead_versions": 2},
            "recompute_kv_cache_after_weight_updates": False,
            "min_groups_for_streaming_train": 16,
            "rollout_failure": {
                "max_skipped_prompts": 0, "max_consecutive_dropped_prompts": 0,
                "max_data_attempts_per_prompt": 1, "max_infra_attempts_per_prompt": 1,
            },
        },
        "policy": {
            "train_micro_batch_size": 2,
            "max_total_sequence_length": 4096,
            "megatron_cfg": {
                "enabled": True,
                "forward_gemm_precision": "fp16",
                "fp32_residual_connection": True,
                "moe_router_dtype": "fp32",
                "defer_fp32_logits": True,
                "batch_invariant_mode": True,
                "moe_router_fusion": False,
                "freeze_moe_router": True,
                "moe_router_bias_update_rate": 0.0,
            },
            "generation": {
                "backend": "megatron",
                "colocated": {"enabled": False},
                "top_k": None,
                "top_p": 1.0,
                "mcore_generation_config": {
                    "kv_cache_management_mode": "persist", "enable_prefix_caching": False, "async_sched_mode": "legacy",
                },
            },
            "sequence_packing": {"enabled": False},
        },
    }


def test_valid_config_has_no_problems():
    assert nolanes.config_problems(_valid()) == []
    cfg = nolanes.validate_config(_valid())
    assert cfg.k_drafts == 1.5 and cfg.crossfade_routing_m == 0.03 and len(cfg.crossfade_routing_m_profile) == 3


@pytest.mark.parametrize(
    "path,value,needle",
    [
        ("policy.generation.mcore_generation_config.kv_cache_management_mode", "recompute", "persist"),
        ("async_rl.recompute_kv_cache_after_weight_updates", True, "persist refit"),
        ("async_rl.sampler.name", "windowed", "in_order"),
        ("async_rl.min_groups_for_streaming_train", 4, "R2"),
        ("async_rl.rollout_failure.max_skipped_prompts", 1, "MEDIUM-4"),
        ("grpo.max_rollout_turns", 2, "multi-turn"),
        ("grpo.use_dynamic_sampling", True, "E6"),
        ("policy.megatron_cfg.forward_gemm_precision", "bf16", "V8b"),
        ("policy.megatron_cfg.batch_invariant_mode", False, "V8b"),
        ("policy.megatron_cfg.moe_router_fusion", True, "cross-fade"),
        ("policy.generation.top_p", 0.9, "wide sampler"),
        ("thundersync.verify_split_x", 0.5, "not built yet"),
        ("thundersync.k_drafts", 0.5, "k_drafts"),
        ("policy.generation.temperature", 0.7, "temperature"),
        ("policy.generation.stop_strings", ["</answer>"], "stop strings"),
        ("async_rl.rollout_failure.max_data_attempts_per_prompt", 2, "retried"),
        ("async_rl.rollout_failure.max_infra_attempts_per_prompt", 3, "retried"),
    ],
)
def test_validator_refuses(path, value, needle):
    c = copy.deepcopy(_valid())
    cur = c
    *head, last = path.split(".")
    for p in head:
        cur = cur[p]
    cur[last] = value
    problems = nolanes.config_problems(c)
    assert any(needle in p for p in problems), problems
    with pytest.raises(ValueError):
        nolanes.validate_config(c)


def test_validator_refuses_sparse_without_device_dispatch():
    c = _valid()
    c["thundersync"]["crossfade_routing_sparse"] = True
    assert any("DEVICE_DISPATCH" in p for p in nolanes.config_problems(c))
    c["policy"]["megatron_cfg"]["env_vars"] = {"NEMO_RL_MOE_DEVICE_DISPATCH": "1"}
    assert nolanes.config_problems(c) == []


def test_validator_refuses_r29_paths(monkeypatch):
    for patch, needle in [
        ({"ppo": {"x": 1}}, "ppo"),
        ({"checkpointing": {"enabled": True}}, "checkpointing"),
        ({"token_capture": {"enabled": True}}, "token capture"),
        ({"env": {"should_use_nemo_gym": True}}, "NeMo-Gym"),
    ]:
        c = _valid()
        c.update(patch)
        assert any(needle in p for p in nolanes.config_problems(c)), (patch, nolanes.config_problems(c))
    c = _valid()
    c["async_rl"]["generation_router"] = {"enabled": True}
    assert any("generation_router" in p for p in nolanes.config_problems(c))
    c = _valid()
    c["async_rl"]["rollout_failure"]["native"] = {"env_timeout_s": 30.0}
    assert any("deadline" in p for p in nolanes.config_problems(c))
    c = _valid()
    c["policy"]["megatron_cfg"]["tensor_model_parallel_size"] = 4
    c["policy"]["megatron_cfg"]["env_vars"] = {"NEMO_RL_TP_REDUCE": "stock"}
    assert any("TP_REDUCE" in p for p in nolanes.config_problems(c))
    monkeypatch.setenv("THUNDERSYNC_NL_NEG", "wrong_tag")
    monkeypatch.delenv("THUNDERSYNC_NL_CHECKS", raising=False)
    assert any("THUNDERSYNC_NL_CHECKS" in p for p in nolanes.config_problems(_valid()))
    monkeypatch.setenv("THUNDERSYNC_NL_CHECKS", "1")
    assert nolanes.config_problems(_valid()) == []


def test_early_reject_tail():
    # 60 rejections over 64k scanned positions, 192 rows: ~4.6 expected (x3) early; 5 observed is fine, 30 is not.
    assert nolanes.early_reject_tail(60, 64000, 192, 5) > 0.1
    assert nolanes.early_reject_tail(60, 64000, 192, 30) < 1e-4
    assert nolanes.early_reject_tail(0, 64000, 192, 0) == 1.0
    # The R30 signature: many early rejections at near-zero hazard.
    assert nolanes.early_reject_tail(40, 120000, 128, 12) < 1e-4


def test_expert_bias_update_is_refused_not_warned():
    """User rule 11-02: a nonzero expert-bias rate is a hard error (router_frozen_problems), no longer a warning."""
    c = _valid()
    c["policy"]["megatron_cfg"]["moe_router_bias_update_rate"] = 1e-3
    assert nolanes.config_warnings(c) == []
    assert any("moe_router_bias_update_rate must be 0" in p for p in nolanes.config_problems(c))


def test_validator_refuses_router_replay_and_unknown_keys():
    c = _valid()
    c["policy"]["router_replay"] = {"enabled": True}
    assert any("E2" in p for p in nolanes.config_problems(c))
    c = _valid()
    c["thundersync"]["cohort_lanes"] = 2
    assert any("unknown keys" in p for p in nolanes.config_problems(c))


def test_negative_names(monkeypatch):
    monkeypatch.setenv("THUNDERSYNC_NL_NEG", "key_reuse")
    assert nolanes.negative() == "key_reuse"
    monkeypatch.setenv("THUNDERSYNC_NL_NEG", "nonsense")
    with pytest.raises(ValueError):
        nolanes.negative()


# -- version tags ------------------------------------------------------------------------------------------------


class _FakeEngineWorker(NolanesEngineMixin):
    """The mixin's version logic without an engine: forward / sample / refit by hand."""

    def __init__(self, neg=""):
        self._nl = {
            "version": 3, "fwd_version": 3, "refits": 0, "tags": VersionTagLog(), "keyreg": KeyRegistry(),
            "neg": neg, "neg_tokens": 0, "steps_since_refit": 0, "tag_mismatch": 0,
        }

    def forward(self):
        self._nl["fwd_version"] = self._nl["version"]
        self._nl["steps_since_refit"] += 1

    def refit(self):
        # suspend_for_refit's version step (the stock suspend is not under test here)
        self._nl["version"] += 1
        self._nl["refits"] += 1
        self._nl["steps_since_refit"] = 0


def test_version_segments_and_shifts():
    assert nolanes.version_segments([3, 3, 4, 4, 4, 5]) == [(3, 0, 2), (4, 2, 5), (5, 5, 6)]
    segs = nolanes.version_segments([3, 3, 4, 4])
    assert nolanes.shift_segments(segs, "wrong_tag", 3) == [(3, 0, 4)]
    assert nolanes.shift_segments(segs, "tag_off_by_one", 3) == [(3, 0, 3), (4, 3, 4)]
    assert nolanes.shift_segments(segs, "", 3) == segs


def _drafted_row(worker, heads, h, n_before, n_after, gen):
    """Sample a row of n_before tokens at the resident version, refit, n_after more; returns (tokens, logq)."""
    toks, lq = [], []
    for t in range(n_before + n_after):
        if t == n_before:
            worker.refit()
        worker.forward()
        v = worker._nl_tag_version()
        # One row at a time on both sides (the batch-invariant forward's property).
        logits = (h[t : t + 1] @ heads[worker._nl["version"]].T).double()[0]
        lp = torch.log_softmax(logits, -1)
        x = int(torch.multinomial(lp.exp(), 1, generator=gen))
        toks.append(x)
        lq.append(float(lp[x]))
        worker._nl["tags"].record(["k"], [t], v)
    return toks, lq


def _rebuild(heads, h, toks, segs):
    out = []
    for v, a, b in segs:
        for j in range(a, b):
            lp = torch.log_softmax((h[j : j + 1] @ heads[v].T).double()[0], -1)
            out.append(float(lp[toks[j]]))
    return out


def test_q_rebuilt_from_tagged_head_is_bitwise_across_a_refit():
    gen = torch.Generator().manual_seed(0)
    heads = {3: torch.randn(11, 5, generator=gen), 4: torch.randn(11, 5, generator=gen)}
    h = torch.randn(12, 5, generator=gen)
    w = _FakeEngineWorker()
    toks, lq = _drafted_row(w, heads, h, 5, 7, gen)
    vers = w._nl["tags"].pop("k", 12)
    segs = nolanes.version_segments(vers)
    assert segs == [(3, 0, 5), (4, 5, 12)]  # a mixed-version row
    assert _rebuild(heads, h, toks, segs) == lq  # bitwise
    # E1 negatives: the dispatch version for every token, or the refit boundary shifted by one, break it.
    assert _rebuild(heads, h, toks, nolanes.shift_segments(segs, "wrong_tag", 3)) != lq
    assert _rebuild(heads, h, toks, nolanes.shift_segments(segs, "tag_off_by_one", 3)) != lq


def test_engine_side_tag_off_by_one_negative_and_torn_forward():
    gen = torch.Generator().manual_seed(1)
    heads = {3: torch.randn(11, 5, generator=gen), 4: torch.randn(11, 5, generator=gen)}
    h = torch.randn(10, 5, generator=gen)
    w = _FakeEngineWorker(neg="tag_off_by_one")
    toks, lq = _drafted_row(w, heads, h, 4, 6, gen)
    segs = nolanes.version_segments(w._nl["tags"].pop("k", 10))
    assert w._nl["neg_tokens"] > 0
    assert _rebuild(heads, h, toks, segs) != lq
    # Async scheduling: a forward launched before the refit and sampled after it keeps the forward's version.
    w2 = _FakeEngineWorker()
    w2.forward()
    w2.refit()
    assert w2._nl_tag_version() == 3 and w2._nl["straddle_samples"] == 1
    w2.refit()  # two versions apart: impossible, refused
    with pytest.raises(RuntimeError, match="sampled logits of a version-3 forward"):
        w2._nl_tag_version()


def test_version_tag_log_prefix_and_gaps():
    log = VersionTagLog()
    log.record(["a", "a", "b"], [0, 1, 0], 7)
    log.record(["a"], [3], 8)  # token 2 never logged
    assert log.pop("a", 10) == [7, 7]  # the longest tagged prefix
    assert log.pop("a", 10) == []  # closed
    assert log.pop("b", 1) == [7]


# -- keys (E4) ---------------------------------------------------------------------------------------------------


def test_key_registry_refuses_reuse():
    reg = KeyRegistry()
    assert reg.observe(np.array([5, 6]), np.array([10, 20])) == []
    assert reg.observe(np.array([5, 6]), np.array([11, 21])) == []
    # A continuation keyed by request 5 at a position its draft already drew: reuse.
    bad = reg.observe(np.array([5]), np.array([11]))
    assert bad and reg.violations == 1
    # A new request id at any position is fresh; unknown positions are skipped.
    assert reg.observe(np.array([9, 5]), np.array([11, -1])) == []


def test_key_reuse_raises_in_the_mixin():
    from thundersync_rl.key_check import SamplerKeyError

    w = _FakeEngineWorker()
    w._nl_observe_keys(np.array([1]), np.array([4]))
    with pytest.raises(SamplerKeyError, match="E4"):
        w._nl_observe_keys(np.array([1]), np.array([3]))


def test_request_attrs_tag_every_request_but_continuations():
    from thundersync_rl.engine_q import HIDDEN_KEY_ATTR
    from thundersync_rl.nolanes_engine import KEY_RID_ATTR

    import itertools

    w = _FakeEngineWorker()
    w._nl["counter"] = itertools.count()

    class SP:
        def __init__(self):
            self.attrs = {}

        def add_attributes(self, d):
            self.attrs.update(d)

    sps = [SP(), SP()]
    w._nl_request_attrs({"nl_no_log": [0, 1], "nl_key_rid": [-1, 42]}, sps)
    assert sps[0].attrs == {HIDDEN_KEY_ATTR: "nl0"}
    assert sps[1].attrs == {KEY_RID_ATTR: 42}
    sps = [SP()]
    w._nl_request_attrs(None, sps)
    assert sps[0].attrs == {HIDDEN_KEY_ATTR: "nl1"}


# -- outputs, E7, provenance -------------------------------------------------------------------------------------


def _draft(i, toks, lq):
    return nolanes.DraftRow("g", i, [1, 2], toks, lq, f"nl{i}", 100 + i, True)


def test_outputs_of_group_and_engine_q_negative():
    d0, d1 = _draft(0, [5, 6, 7], [-0.1, -0.2, -0.3]), _draft(1, [5, 8, 9], [-0.1, -0.4, -0.5])
    whole = SimpleNamespace(prefix=[5, 6, 7], prefix_logprobs=[-0.11, -0.21, -0.31], complete=True)
    cont = SimpleNamespace(prefix=[5, 3], prefix_logprobs=[-0.12, -2.0], complete=False)
    out = nolanes.outputs_of_group([d0, d1], [whole, cont], 3)
    assert [o.kind for o in out] == ["whole", "continued", "fresh"]
    assert out[0].prefix_logprobs == [-0.11, -0.21, -0.31]  # verified p, not q (E7)
    n = nolanes.keep_engine_q(out)
    # Negative: accepted tokens (draft prefix) take q; the residual token (3 != 8) keeps p.
    assert n == 4 and out[0].prefix_logprobs == [-0.1, -0.2, -0.3] and out[1].prefix_logprobs == [-0.1, -2.0]
    bad = SimpleNamespace(prefix=[5], prefix_logprobs=[], complete=False)
    with pytest.raises(RuntimeError, match="log-probs"):
        nolanes.outputs_of_group([d0], [bad], 1)
    with pytest.raises(RuntimeError, match="slots"):
        nolanes.outputs_of_group([d0, d1], [whole, cont], 1)


def test_continuation_budget():
    assert nolanes.continuation_budget(10, 3, 100, None) == 97
    assert nolanes.continuation_budget(10, 3, 100, 50) == 37
    assert nolanes.continuation_budget(60, 3, 100, 50) == 0


def test_provenance_check():
    tags = [{nolanes.VERIFIED_TAG: 4}, {nolanes.VERIFIED_TAG: 4}]
    assert nolanes.check_provenance(tags, 4, "t") == 2
    with pytest.raises(nolanes.ProvenanceError, match="E6/E9"):
        nolanes.check_provenance(tags + [{"weight_version": 3}], 4, "t")
    with pytest.raises(nolanes.ProvenanceError):
        nolanes.check_provenance(tags, 5, "t")


def test_crossfade_layer_bands_nano_profile():
    from nemo_rl.models.megatron import crossfade_routing as cf

    cfg = nolanes.NolanesConfig()
    layers = list(range(1, 47, 2))  # 23 MoE layers
    bands = cf.layer_bands(cfg.crossfade_routing_m, [tuple(r) for r in cfg.crossfade_routing_m_profile], layers)
    scales = [round(bands[ln] / 0.03, 6) for ln in layers]
    assert scales == [0.3] * 6 + [1.0] * 12 + [0.5] * 5
    assert cf.moe_layer_numbers(SimpleNamespace(hybrid_override_pattern="MEMEMEM*")) == [2, 4, 6]
    with pytest.raises(ValueError):
        cf.layer_bands(0.03, [(0.0, 0.5, 1.0)], layers)


def test_stream_chunks_contiguous_balanced_and_complete():
    keys = list("abcdefghij")
    sizes = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    assert nolanes.stream_chunks(keys, sizes, 1) == [keys]
    out = nolanes.stream_chunks(keys, sizes, 4)
    assert [k for run in out for k in run] == keys and len(out) == 4
    assert all(len(run) in (2, 3) for run in out)
    # never more runs than keys; zero-size keys stay in order
    assert [k for run in nolanes.stream_chunks(keys[:2], [0, 5], 8) for k in run] == keys[:2]
    assert len(nolanes.stream_chunks(keys[:2], [3, 5], 8)) <= 2
    # a function of order and counts only
    assert nolanes.stream_chunks(keys, sizes, 3) == nolanes.stream_chunks(list(keys), list(sizes), 3)
    with pytest.raises(ValueError):
        nolanes.stream_chunks(keys, sizes[:3], 2)


def test_merge_call_stats_sums_counts_and_weights_ratios():
    a = {"spec/null_drafts": 30.0, "spec/null_outputs": 24.0, "spec/null_waves": 3.0, "spec/null_bot_frac": 0.1}
    b = {"spec/null_drafts": 10.0, "spec/null_outputs": 8.0, "spec/null_waves": 2.0, "spec/null_bot_frac": 0.5}
    m = nolanes.merge_call_stats([a, b])
    assert m["spec/null_drafts"] == 40.0 and m["spec/null_outputs"] == 32.0 and m["spec/null_waves"] == 3.0
    assert abs(m["spec/null_bot_frac"] - 0.2) < 1e-12
    assert nolanes.merge_call_stats([a]) == a


def test_merge_call_stats_group_fractions_weighted_by_live_groups():
    """R36: per-call fractions of live groups stay fractions (they were summed, exceeding 1 with several calls)."""
    a = {"spec/null_drafts": 30.0, "spec/null_live_groups": 3.0, "spec/null_groups_fully_drafted": 1.0,
         "spec/null_groups_whole_accepted": 2 / 3}
    b = {"spec/null_drafts": 10.0, "spec/null_live_groups": 1.0, "spec/null_groups_fully_drafted": 0.0,
         "spec/null_groups_whole_accepted": 0.0}
    m = nolanes.merge_call_stats([a, b])
    assert m["spec/null_live_groups"] == 4.0
    assert abs(m["spec/null_groups_fully_drafted"] - 0.75) < 1e-12
    assert abs(m["spec/null_groups_whole_accepted"] - 0.5) < 1e-12


def test_validator_stream_chunks():
    c = _valid()
    c["thundersync"]["verify_stream_chunks"] = 4
    assert nolanes.config_problems(c) == []
    c["thundersync"]["verify_stream_chunks"] = 0
    assert any("verify_stream_chunks" in p for p in nolanes.config_problems(c))
    c["thundersync"]["verify_stream_chunks"] = 4
    c["thundersync"]["row_seeds"] = False
    assert any("row_seeds" in p for p in nolanes.config_problems(c))


def _overlap():
    c = _valid()
    c["policy"]["megatron_cfg"]["distributed_data_parallel_config"] = {"grad_reduce_in_fp32": True}
    c["policy"]["megatron_cfg"]["moe_router_bias_update_rate"] = 0.0
    return c


def test_r32_validator_items(monkeypatch):
    monkeypatch.delenv("THUNDERSYNC_NL_NEG", raising=False)
    monkeypatch.delenv("THUNDERSYNC_NL_CHECKS", raising=False)
    # (6) every debug / negative knob refused without CHECKS, in the process env or the workers' env_vars
    for name in sorted(nolanes.DEBUG_ENV_NAMES) + ["THUNDERSYNC_NL_NEG", "THUNDERSYNC_NL_DEBUG_STEP_SLEEP_MS",
                                                   "THUNDERSYNC_DEBUG_LAW_BUG", "THUNDERSYNC_DEBUG_SENTINEL_FLIP"]:
        c = _overlap()
        c["policy"]["megatron_cfg"]["env_vars"] = {name: "1"}
        assert any("THUNDERSYNC_NL_CHECKS" in p for p in nolanes.config_problems(c)), name
    # driver-owned knobs (none on the colocated branch) belong to configure_driver
    for name in nolanes.DRIVER_OWNED_ENV:
        c = _overlap()
        c["policy"]["megatron_cfg"]["env_vars"] = {name: "1"}
        assert any("configure_driver" in p for p in nolanes.config_problems(c)), name
        monkeypatch.setenv(name, "1")
        assert any("configure_driver" in p for p in nolanes.config_problems(_overlap())), name
        monkeypatch.delenv(name)


def test_removed_verify_order_key_fails_to_load():
    # R34: risky_first was removed; a config that still sets thundersync.verify_order must be refused, not ignored.
    for v in ("fixed", "risky_first"):
        c = _valid()
        c["thundersync"]["verify_order"] = v
        with pytest.raises(ValueError, match="unknown keys.*verify_order"):
            nolanes.NolanesConfig.from_config(c)
        assert any("verify_order" in p for p in nolanes.config_problems(c))


def test_nolanes_admission_reads_the_installed_version():
    """ACCEPT_DECAY §2 (unconditional in nolanes): the SC's admission version is the stage's, i.e. the engine's
    installed version, not the trainer bump; stock keeps the trainer version."""
    from types import SimpleNamespace

    import thundersync_rl.nolanes_stage as ns
    from nemo_rl.algorithms import single_controller as sc_mod
    from nemo_rl.experience.rollout_manager import RolloutManager

    klass = getattr(sc_mod.SingleControllerActor, "__ray_actor_class__", None)
    if klass is None:
        klass = sc_mod.SingleControllerActor.__ray_metadata__.modified_class
    fn = klass._admission_version
    st = ns.VerifyStage.__new__(ns.VerifyStage)
    st.admit_hold_version = None
    rm = SimpleNamespace(weight_version=lambda: 6)
    assert fn(SimpleNamespace(_nl=st, _trainer_version=7, _rollout_manager=rm)) == 6  # installed, not the bump
    assert fn(SimpleNamespace(_nl=None, _trainer_version=7, _rollout_manager=rm)) == 7  # stock
    assert "weight_version" in vars(RolloutManager)


def test_every_admission_site_reads_the_admission_version():
    """R35 2(g): every sampler admission call in single_controller.py goes through _admission_version, so a new site
    cannot silently bypass admit_after_refit (or the X1 admission hold) by reading the trainer version."""
    import re

    from nemo_rl.algorithms import single_controller as sc_mod

    src = open(sc_mod.__file__).read()
    assert "trainer_version_fn=lambda: self._trainer_version" not in src
    sites = re.findall(r"trainer_version_fn=([^,)\s]+)", src)
    assert len(sites) >= 4, sites  # 4 admission sites today (initial, refill, pump, resume)
    assert set(sites) == {"self._admission_version"}, sites


def test_x1_admission_hold_lasts_until_the_optimizer_step():
    """X1 gate tool (THUNDERSYNC_NL_DEBUG_ADMIT_AFTER_TAILS): during step k (hold k-1) admissions wait until the
    optimizer step bumps the trainer version; no hold (production) leaves the installed version unchanged."""
    import thundersync_rl.nolanes_stage as ns

    st = ns.VerifyStage.__new__(ns.VerifyStage)
    st.admit_hold_version = None
    assert st.admission_version(6, 7) == 6
    st.admit_hold_version = 5  # mid-step k=6 (engine 6, trainer 6)
    assert st.admission_version(6, 6) == 5
    assert st.admission_version(6, 7) == 6  # optimizer step done: released
    assert "THUNDERSYNC_NL_DEBUG_ADMIT_AFTER_TAILS".startswith(nolanes.DEBUG_ENV_PREFIXES)


def test_stage_run_defines_its_locals_before_use():
    """Regression (X1: UnboundLocalError at step 0): every local of VerifyStage.run that the draft loop reads
    (mixed_drafts) is bound before its first use in the source."""
    import inspect

    from thundersync_rl.nolanes_stage import VerifyStage

    src = inspect.getsource(VerifyStage.run)
    for name in ("mixed_drafts", "metas"):
        first_def = src.find(f"{name}: ")
        uses = [i for i in (src.find(f"{name}."), src.find(f"in {name}"), src.find(f"{name}[")) if i >= 0]
        assert first_def >= 0 and uses and first_def < min(uses), name


def test_colo_stage_checks_read_only_colo_config_fields():
    """Regression (slim gate: XS with CHECKS=1 raised AttributeError 'ColoConfig' object has no attribute
    'variable_group'): the VerifyStage methods ColoStage runs read only ColoConfig fields as plain attributes; any
    other thundersync key is read with getattr and a default."""
    import dataclasses
    import inspect
    import re

    from thundersync_rl.colo import ColoConfig
    from thundersync_rl.colo_stage import ColoStage
    from thundersync_rl.nolanes_stage import VerifyStage

    fields = {f.name for f in dataclasses.fields(ColoConfig)}
    # Every VerifyStage method ColoStage inherits without overriding (its own calls and the controller's
    # ``self._nl.X`` calls, e.g. defers_post_step_wake / check_router_frozen, all resolve to these).
    shared = [n for n, f in vars(VerifyStage).items()
              if callable(f) and not n.startswith("__") and n not in vars(ColoStage)]
    assert {"_check_batch", "defers_post_step_wake"} <= set(shared)
    for name in shared:
        src = inspect.getsource(getattr(VerifyStage, name))
        used = set(re.findall(r"\bself\.cfg\.(\w+)", src))
        assert used <= fields, (name, sorted(used - fields))


def _segments_reference(versions):
    segs = []
    for i, v in enumerate(versions):
        if segs and segs[-1][0] == v and segs[-1][2] == i:
            segs[-1] = (v, segs[-1][1], i + 1)
        else:
            segs.append((int(v), i, i + 1))
    return segs


def test_version_segments_matches_the_reference_loop():
    """The vectorized version_segments returns exactly the runs of the per-token loop it replaced (python ints)."""
    import numpy as np

    rng = np.random.default_rng(0)
    cases = [[], [7], [3, 3, 3], [3, 4, 3], list(range(5))]
    for _ in range(200):
        n = int(rng.integers(0, 300))
        cases.append(np.repeat(rng.integers(0, 4, size=n // 7 + 1), rng.integers(1, 20, size=n // 7 + 1)).tolist())
    for c in cases:
        ref = _segments_reference(c)
        for arg in (c, np.asarray(c, dtype=np.int64), np.asarray(c, dtype=np.int32)):
            got = nolanes.version_segments(arg)
            assert got == ref and all(type(x) is int for s in got for x in s), (c, got, ref)


def test_tag_pop_matches_the_reference_loop():
    """VersionTagLog.pop (vectorized) = the longest tagged prefix of the first n tags, as the per-token loop did."""
    import numpy as np

    rng = np.random.default_rng(1)
    for _ in range(100):
        log = VersionTagLog()
        n_tok = int(rng.integers(0, 60))
        ts = rng.permutation(n_tok)[: int(rng.integers(0, n_tok + 1))].tolist() if n_tok else []
        for t in ts:
            log.record(["k"], [t], int(rng.integers(0, 3)))
        raw = list(log._tags.get("k", []))
        n = int(rng.integers(0, 70))
        ref = []
        for v in raw[:n]:
            if v < 0:
                break
            ref.append(int(v))
        got = log.pop("k", n)
        assert got == ref and all(type(x) is int for x in got)
    assert VersionTagLog().pop("missing", 5) == []


def test_nl_base_does_not_start_the_http_front_end():
    """DESIGN §16.9: the OpenAI HTTP server (unused by nolanes) cost ~80 s of the initial weight sync."""
    import os

    import yaml

    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "configs", "nl", "nl_base.yaml")
    gen = yaml.safe_load(open(here))["policy"]["generation"]["mcore_generation_config"]
    assert gen["expose_http_server"] is False


def test_validator_refuses_a_trainable_router_or_bias_update():
    """User rule 11-02 (hard error): router gate weights frozen and expert-bias update rate exactly 0."""
    assert not nolanes.router_frozen_problems(_valid())
    assert nolanes.router_frozen_line(_valid()) == "router: frozen (weights+bias)"
    for key, val, frag in (("freeze_moe_router", False, "freeze_moe_router"),
                           ("moe_router_bias_update_rate", 1e-3, "moe_router_bias_update_rate"),
                           ("moe_router_bias_update_rate", None, "moe_router_bias_update_rate")):
        c = _valid()
        if val is None:
            c["policy"]["megatron_cfg"].pop(key)
        else:
            c["policy"]["megatron_cfg"][key] = val
        assert any(frag in p for p in nolanes.config_problems(c)), (key, val)
        with pytest.raises(ValueError, match="router not frozen"):
            nolanes.router_frozen_line(c)
    import os

    import yaml

    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "configs", "nl", "nl_base.yaml")
    m = yaml.safe_load(open(here))["policy"]["megatron_cfg"]
    assert m["freeze_moe_router"] is True and m["moe_router_bias_update_rate"] == 0.0




def test_dry_preflight_derives_the_layout_fields_before_the_bi_validators():
    """Regression (Ultra): the DRY preflight ran the batch-invariance validators on the
    raw config, without the EP-invariant combine configure_driver derives, so learner EP 32 vs engine EP 16 failed
    "must use the same expert_model_parallel_size". The preflight must derive the same fields first."""
    import pathlib

    src = (pathlib.Path(nolanes.__file__).parents[1] / "run_nolanes_sc.py").read_text()
    assert 0 <= src.index("derive_layout_fields(c)") < src.index("validate_zero_train_gen_kl(pol")
    for lep, gep, want in ((8, 8, False), (32, 16, True), (2, None, False)):
        c = {"policy": {"megatron_cfg": {"expert_model_parallel_size": lep},
                        "generation": {"mcore_generation_config": {} if gep is None else
                                       {"expert_model_parallel_size": gep}}}}
        nolanes.derive_layout_fields(c)
        assert c["policy"]["megatron_cfg"]["moe_ep_invariant_combine"] is want


def test_configure_driver_derives_the_combine_and_verify_tokens(monkeypatch):
    """Structural review L15 / L17: configure_driver turns the EP-invariant combine on exactly when engine EP !=
    learner EP, and resolves verify_batch_tokens by the rule (MBS x max seq)."""
    import sys
    import types

    reg = types.ModuleType("nemo_rl.distributed.ray_actor_environment_registry")
    reg.ACTOR_ENVIRONMENT_REGISTRY = {}
    vc = types.ModuleType("nemo_rl.distributed.virtual_cluster")
    vc.PY_EXECUTABLES = types.SimpleNamespace(MCORE="mcore")
    monkeypatch.setitem(sys.modules, "nemo_rl.distributed.ray_actor_environment_registry", reg)
    monkeypatch.setitem(sys.modules, "nemo_rl.distributed.virtual_cluster", vc)
    monkeypatch.delenv("THUNDERSYNC_NL_NEG", raising=False)
    for lep, gep, want in ((8, 8, False), (32, 16, True)):
        c = _overlap()
        c["policy"]["megatron_cfg"]["expert_model_parallel_size"] = lep
        c["policy"]["generation"]["mcore_generation_config"]["expert_model_parallel_size"] = gep
        nolanes.configure_driver(c)
        assert c["policy"]["megatron_cfg"]["moe_ep_invariant_combine"] is want
        assert c["thundersync"]["verify_batch_tokens"] == 2 * 4096


# -- wake_verify (colocated verify at the wake: drafts cut at the stand-down) -----------------------------------------


def test_wake_verify_needs_a_colocated_engine():
    c = copy.deepcopy(_valid())
    c["thundersync"]["wake_verify"] = True
    assert any("wake_verify needs a colocated engine" in p for p in nolanes.config_problems(c))


def test_drafts_of_record_marks_cut_drafts_unfinished():
    def comp(toks, ref):
        return SimpleNamespace(message_log=[{"role": "assistant", "token_ids": torch.tensor(toks),
                                             "generation_logprobs": torch.zeros(len(toks)), nolanes.DRAFT_REF: ref}])

    rec = SimpleNamespace(completions=[comp([5, 6], {"key": "nl0", "rid": 1}),
                                       comp([5, 7, 8], {"key": "nl1", "rid": 2, "cut": 1}),
                                       comp([9], {"key": "nl2", "rid": 3, "cut": 0})])
    rows = nolanes.drafts_of_record("g", [{"token_ids": torch.tensor([1, 2])}], rec)
    assert [r.finished for r in rows] == [True, False, True]
    assert rows[1].tokens == [5, 7, 8] and rows[1].prompt == [1, 2]


def test_mark_cut_marks_drafts_only_and_once():
    from thundersync_rl.engine_q import HIDDEN_KEY_ATTR
    from thundersync_rl.nolanes_engine import CUT_ATTR, mark_cut

    class SP:
        def __init__(self, **kw):
            self.__dict__.update(kw)
            self.num_tokens_to_generate = 512

        def add_attributes(self, d):
            self.__dict__.update(d)

    class Rec:
        def __init__(self, sp):
            self.requests = [SimpleNamespace(sampling_params=sp)]

        def __getitem__(self, i):
            return self.requests[i]

    sps = {1: SP(**{HIDDEN_KEY_ATTR: "nl0"}), 2: SP(**{HIDDEN_KEY_ATTR: "nl1"}), 3: SP(), 4: SP(**{HIDDEN_KEY_ATTR: "nl2"})}
    eng = SimpleNamespace(requests={k: SimpleNamespace(record=Rec(v)) for k, v in sps.items()},
                          waiting_request_ids=[4], stop_word_finished_request_ids=set())
    assert mark_cut(eng) == {"cut": 2, "waiting": 1, "marked": 0}
    assert eng.stop_word_finished_request_ids == {1, 2}  # resident drafts: finished at the next step, no new token
    assert sps[4].num_tokens_to_generate == 1  # waiting draft: one token, then finished
    assert getattr(sps[3], CUT_ATTR, 0) == 0 and 3 not in eng.stop_word_finished_request_ids  # continuation untouched
    assert all(getattr(sps[k], CUT_ATTR) == 1 for k in (1, 2, 4))
    assert mark_cut(eng) == {"cut": 0, "waiting": 0, "marked": 3}  # a second round marks nothing twice


def test_parse_result_hands_back_the_cut_flag():
    from thundersync_rl.engine_q import HIDDEN_KEY_ATTR
    from thundersync_rl.nolanes_engine import CUT_ATTR

    class Base:
        def _parse_result_to_batched_data_dict(self, data, result):
            return {}

    class W(NolanesEngineMixin, Base):
        pass

    w = W()
    w._nl = {}
    res = [SimpleNamespace(request_id=7, sampling_params=SimpleNamespace(**{HIDDEN_KEY_ATTR: "nl3", CUT_ATTR: 1})),
           SimpleNamespace(request_id=8, sampling_params=SimpleNamespace(**{HIDDEN_KEY_ATTR: "nl4"}))]
    out = w._parse_result_to_batched_data_dict(None, res)
    assert out["ts_draft"] == [{"key": "nl3", "rid": 7, "cut": 1}, {"key": "nl4", "rid": 8, "cut": 0}]


def test_variable_group_validator():
    c = copy.deepcopy(_valid())
    c["thundersync"]["variable_group"] = True
    assert any("variable_group needs a constant envelope" in p for p in nolanes.config_problems(c))
    c["thundersync"]["null_bv_fixed_m"] = 1.4
    c["thundersync"]["k_drafts"] = 1.0
    assert not any("variable_group" in p for p in nolanes.config_problems(c))
    c["thundersync"]["k_drafts"] = 1.75
    assert any("variable_group: k_drafts must be 1" in p for p in nolanes.config_problems(c))


def test_verify_batch_tokens_rule_is_the_micro_batch_capped_by_t_cap():
    """Verify budget (ts-colo e46914cd): the training micro-batch MBS x max_total_sequence_length, capped by T_cap
    under MoE device dispatch (a budget of T_cap overflowed the receive capacity under cross-fade routing); an
    explicit value wins."""
    c = copy.deepcopy(_valid())
    pol = c["policy"]
    mb = pol["train_micro_batch_size"] * pol["max_total_sequence_length"]
    envs = pol.setdefault("megatron_cfg", {}).setdefault("env_vars", {})
    cfg = nolanes.NolanesConfig.from_config(c)
    assert nolanes.effective_verify_batch_tokens(c, cfg) == mb
    envs.update(NEMO_RL_MOE_DEVICE_DISPATCH="1", NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS=str(3 * mb))
    assert nolanes.effective_verify_batch_tokens(c, cfg) == mb
    envs["NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS"] = str(mb // 2)
    assert nolanes.effective_verify_batch_tokens(c, cfg) == mb // 2
    c["thundersync"]["verify_batch_tokens"] = 1234
    assert nolanes.effective_verify_batch_tokens(c, nolanes.NolanesConfig.from_config(c)) == 1234


def _colocated_wake_verify(nps=16, inflight=None, buffered=None, lag=1):
    c = copy.deepcopy(_valid())
    c["policy"]["generation"]["colocated"]["enabled"] = True
    c["policy"]["offload_optimizer_for_refit"] = False
    c["thundersync"]["wake_verify"] = True
    c["grpo"]["num_prompts_per_step"] = nps
    c["async_rl"]["sampler"]["max_lookahead_versions"] = lag
    if inflight is not None:
        c["async_rl"]["max_inflight_prompts"] = inflight
    if buffered is not None:
        c["async_rl"]["max_buffered_rollouts"] = buffered
    return c


def test_wake_verify_needs_the_whole_lookahead_batch_in_flight():
    """Review F1: with the post-step wake deferred, a lookahead group not in flight at the cut would wait for a wake
    only the next stage gives (hang): lag 1 and inflight / buffered >= 2 x num_prompts_per_step are required."""
    ok = nolanes.config_problems(_colocated_wake_verify(inflight=32, buffered=32))
    assert not any("wake_verify needs" in p for p in ok), ok
    for kw, needle in ((dict(inflight=31, buffered=32), "max_inflight_prompts"),
                       (dict(inflight=32, buffered=16), "max_buffered_rollouts"),
                       (dict(inflight=32, buffered=32, lag=2), "max_lookahead_versions"),
                       (dict(inflight=32, buffered=32, lag=0), "max_lookahead_versions"),
                       (dict(inflight=None, buffered=32), "max_inflight_prompts")):
        probs = nolanes.config_problems(_colocated_wake_verify(**kw))
        assert any("wake_verify needs" in p and needle in p for p in probs), (kw, probs)


def test_colo_stage_defers_no_post_step_wake():
    """Regression (Ultra XS): ColoStage's ColoConfig has no wake_verify field; defers_post_step_wake must read
    it with a default instead of raising AttributeError after step 0."""
    from thundersync_rl.colo_stage import ColoStage

    st = ColoStage.__new__(ColoStage)
    st.cfg = SimpleNamespace()
    st.colocated = True
    assert st.defers_post_step_wake() is False
    st.cfg = SimpleNamespace(wake_verify=True)
    assert st.defers_post_step_wake() is True


def test_post_save_wake_is_skipped_when_the_stage_defers_it():
    """Review B1: on a checkpoint-save step under wake_verify, the bare post-save wake would open the dispatch gate with
    the rollout weight version still k; the stage's wake installs theta_{k+1} and stamps the version first."""
    import inspect

    from nemo_rl.algorithms import single_controller

    src = inspect.getsource(single_controller)
    i = src.index("The save onloaded model+optimizer")
    block = src[i : i + 1500]
    assert "if nl_defer_wake:" in block
    i_defer, i_else = block.index("if nl_defer_wake:"), block.index("else:")
    assert "finalize_async_save" in block[i_defer:i_else] and "offload_after_refit" not in block[i_defer:i_else]
    assert i_else < block.index("self._rollout_permitted.set()")


def test_a_draft_cut_before_its_first_token_parses_as_empty():
    from thundersync_rl.nolanes_engine import normalize_empty_results

    rs = [SimpleNamespace(generated_tokens=[], generated_log_probs=None),
          SimpleNamespace(generated_tokens=[4, 5], generated_log_probs=[-0.1, -0.2])]
    assert normalize_empty_results(rs) == 1
    assert rs[0].generated_log_probs == [] and rs[1].generated_log_probs == [-0.1, -0.2]


def test_prefix_caching_needs_a_weight_scoped_prefix_cache(monkeypatch):
    """E8: prefix caching is refused unless the engine scopes the prefix cache by weight version."""
    c = _valid()
    c["policy"]["generation"]["mcore_generation_config"]["enable_prefix_caching"] = True
    monkeypatch.setattr(nolanes, "mcore_prefix_cache_weight_scoped", lambda: False)
    assert any("E8" in p for p in nolanes.config_problems(c))
    monkeypatch.setattr(nolanes, "mcore_prefix_cache_weight_scoped", lambda: True)
    assert not any("E8" in p for p in nolanes.config_problems(c))


def test_pinned_megatron_core_scopes_the_prefix_cache_by_weight_version():
    pytest.importorskip("megatron.core.inference.engines.dynamic_engine")
    assert nolanes.mcore_prefix_cache_weight_scoped()


def test_chunk_id_guard_raises_when_the_live_id_differs_from_the_consumed_one():
    from types import SimpleNamespace

    import pytest

    worker = pytest.importorskip("thundersync_rl.worker")
    def proc(request_ids, finished, consumed_chunked_prefill_request_id, top_n=None):
        return "ok"

    eng = SimpleNamespace(_process_request_outputs=proc)
    g = worker.install_chunk_id_guard(eng)
    assert worker.install_chunk_id_guard(eng) is g  # idempotent
    g.seen(7)
    assert eng._process_request_outputs(None, None, consumed_chunked_prefill_request_id=7) == "ok"
    eng._process_request_outputs(None, None, 3)  # nothing seen since: no check
    g.seen(-1)
    with pytest.raises(RuntimeError, match="chunk id guard"):
        eng._process_request_outputs(None, None, 5)  # positional: still checked
    assert g.checked == 2
    with pytest.raises(RuntimeError, match="has no consumed_chunked_prefill_request_id"):
        worker.install_chunk_id_guard(SimpleNamespace(_process_request_outputs=lambda a, b: None))


def test_kv_offload_allowed_with_torch_memory_saver():
    """KV offload (released to a CPU backup during training, restored at its addresses) is valid exactly when
    torch_memory_saver is importable (the TMS=1 overlay); persist always; recompute never."""
    import importlib.util

    have_tms = importlib.util.find_spec("torch_memory_saver") is not None
    assert nolanes.kv_mode_problems({"kv_cache_management_mode": "persist"}) == []
    off = nolanes.kv_mode_problems({"kv_cache_management_mode": "offload"})
    assert (off == []) if have_tms else ("torch_memory_saver" in off[0])
    assert "persist or offload" in nolanes.kv_mode_problems({"kv_cache_management_mode": "recompute"})[0]


def test_offload_refuses_an_unpinned_torch_memory_saver():
    """Review batch 25 (E3a): the teardown hook uses the pinned torch_memory_saver's region flag call; any other
    release fails fast in the validator."""
    import importlib.util

    from nemo_rl.models.megatron import memory_saver as ms

    if importlib.util.find_spec("torch_memory_saver") is None:
        pytest.skip("torch_memory_saver (see docs/colo/HANDOFF.md, overlays) not on PYTHONPATH")
    assert ms.tms_version_problem() == ""
    pinned = ms.TMS_PINNED_VERSION
    try:
        ms.TMS_PINNED_VERSION = "0.0.0"
        off = nolanes.kv_mode_problems({"kv_cache_management_mode": "offload"})
        assert off and "not the pinned" in off[0], off
    finally:
        ms.TMS_PINNED_VERSION = pinned


def test_merge_call_stats_sums_empty_walked_and_keeps_the_run_total():
    a = {"spec/null_drafts": 10.0, "spec/null_empty_walked": 2.0, "spec/null_empty_walked_total": 7.0}
    b = {"spec/null_drafts": 30.0, "spec/null_empty_walked": 1.0, "spec/null_empty_walked_total": 8.0}
    m = nolanes.merge_call_stats([a, b])
    assert m["spec/null_empty_walked"] == 3.0  # a count over both calls' pools
    assert m["spec/null_empty_walked_total"] == 8.0  # the cumulative counter after the last call


def test_kv_live_drop_one_negative_needs_checks(monkeypatch):
    """The kv_live_offload negative control (kv_live_offload.DROP_ONE_ENV) is refused without THUNDERSYNC_NL_CHECKS,
    set in the process env or in the workers' env_vars."""
    from nemo_rl.models.megatron.kv_live_offload import DROP_ONE_ENV

    monkeypatch.delenv("THUNDERSYNC_NL_CHECKS", raising=False)
    monkeypatch.delenv(DROP_ONE_ENV, raising=False)
    assert DROP_ONE_ENV in nolanes.DEBUG_ENV_NAMES
    c = copy.deepcopy(_valid())
    assert not any(DROP_ONE_ENV in p for p in nolanes.config_problems(c))
    c["policy"]["megatron_cfg"].setdefault("env_vars", {})[DROP_ONE_ENV] = "1"
    assert any(DROP_ONE_ENV in p and "THUNDERSYNC_NL_CHECKS" in p for p in nolanes.config_problems(c))
    c = copy.deepcopy(_valid())
    monkeypatch.setenv(DROP_ONE_ENV, "1")
    assert any(DROP_ONE_ENV in p and "THUNDERSYNC_NL_CHECKS" in p for p in nolanes.config_problems(c))
    monkeypatch.setenv("THUNDERSYNC_NL_CHECKS", "1")
    assert not any(DROP_ONE_ENV in p for p in nolanes.config_problems(c))
