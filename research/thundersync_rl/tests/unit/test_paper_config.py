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
"""configs/paper/*.yaml resolve to the env-var gate configuration, and thundersync.driver_env.

The env-var route is the resolved launch of the Super-L22 gate (data/run_ab_arm/gate3_super_l22_{ts,async1}.golden:
the training argv and exported variables run_ab_arm.sh produced for go_gate3.sh SE=1 SA=1 V0=1 t1 / a8). Removing
the layout and per-run arguments from that argv, applying the rest to the arm's ab config and adding the two
settings decided after the gate (adaptive draft age D6, NCCL_BUFFSIZE) must give the paper config exactly; its
driver_env must equal the gate's exported configuration variables.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thundersync_rl import driver_env

HERE = Path(__file__).resolve().parent
CFG = HERE.parents[1] / "configs"
DATA = HERE / "data" / "run_ab_arm"

# Launcher-owned arguments (layout, model, lengths, batch, steps, seed, output paths), not part of an arm's config.
LAYOUT = (
    "logger.log_dir=", "policy.model_name=", "grpo.max_num_steps=", "grpo.num_prompts_per_step=",
    "grpo.num_generations_per_prompt=", "policy.train_global_batch_size=", "policy.max_total_sequence_length=",
    "data.max_input_seq_length=", "policy.generation.max_new_tokens=", "cluster.", "policy.generation.colocated.",
    "policy.megatron_cfg.tensor_model_parallel_size=", "policy.megatron_cfg.pipeline_model_parallel_size=",
    "policy.megatron_cfg.expert_model_parallel_size=", "policy.megatron_cfg.sequence_parallel=",
    "++policy.generation.mcore_generation_config.tensor_model_parallel_size=",
    "++policy.generation.mcore_generation_config.pipeline_model_parallel_size=",
    "++policy.generation.mcore_generation_config.expert_model_parallel_size=",
    "++policy.megatron_cfg.moe_ep_invariant_combine=", "grpo.seed=",
)
# Exported variables that are harness / measurement-setup, not configuration (truncated-model parity reward, mismatch
# row report, interpreter selection, TF32 / Ray log settings of the launcher).
NOT_CONFIG_ENV = (
    "THUNDERSYNC_DEBUG_PARITY_REWARD", "THUNDERSYNC_MISMATCH_ROW_REPORT", "THUNDERSYNC_RUN_DIR",
    "NEMO_RL_PY_EXECUTABLES_SYSTEM", "NVIDIA_TF32_OVERRIDE", "RAY_DEDUP_LOGS",
)
AFTER_GATE = ["++policy.megatron_cfg.env_vars.NCCL_BUFFSIZE='1048576'"]
VL = ["++policy.megatron_cfg.env_vars.NEMO_RL_VOCAB_LSE='canonical_block'"]
# arm: (paper file, golden, ab config of the env route, extra settings decided after the gate, extra driver env)
ARMS = {
    "ts": ("ts.yaml", "gate3_super_l22_ts", "ab_ts_scale.yaml",
           ["++thundersync.cohort_lane_lookahead_adaptive=true"] + AFTER_GATE, {}),
    "a1": ("a1.yaml", "gate3_super_l22_async1", "ab_async1.yaml", VL + AFTER_GATE,
           {"NEMO_RL_VOCAB_LSE": "canonical_block"}),
    "a1l": ("a1l.yaml", "gate3_super_l22_async1", "ab_async1l.yaml", VL + AFTER_GATE,
            {"NEMO_RL_VOCAB_LSE": "canonical_block"}),
    "ak": ("ak.yaml", "gate3_super_l22_async1", "ab_async2.yaml", VL + AFTER_GATE,
           {"NEMO_RL_VOCAB_LSE": "canonical_block"}),
}


def _golden(name):
    args, env = [], {}
    for line in (DATA / f"{name}.golden").read_text().splitlines():
        kind, _, rest = line.partition(" ")
        if kind == "ARG":
            args.append(rest)
        elif kind == "ENV":
            k, _, v = rest.partition("=")
            env[k] = v
    # argv: python <entry> --config <cfg> --history-json <path> <overrides...>
    return args[6:], env


def _compose(path, overrides=()):
    from nemo_rl.utils.config import load_config, parse_hydra_overrides, register_omegaconf_resolvers
    from omegaconf import OmegaConf

    register_omegaconf_resolvers()
    c = load_config(str(path))
    if overrides:
        c = parse_hydra_overrides(c, list(overrides))
    return OmegaConf.to_container(c, resolve=True)


def env_route(arm):
    _, golden, ab, extra, _ = ARMS[arm]
    args, env = _golden(golden)
    keep = [a for a in args if not a.startswith(LAYOUT)]
    cfg = _compose(CFG / "ab" / ab, keep + extra)
    cfg.setdefault("thundersync", {}).pop("driver_env", None)
    return cfg, {k: v for k, v in env.items() if k not in NOT_CONFIG_ENV}


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_paper_config_equals_the_env_gate_config(arm):
    pytest.importorskip("omegaconf")
    paper = _compose(CFG / "paper" / ARMS[arm][0])
    denv = paper["thundersync"].pop("driver_env")
    cfg, env = env_route(arm)
    assert paper == cfg
    assert denv == {**env, **ARMS[arm][4]}


def test_paper_ts_has_the_architecture_on():
    pytest.importorskip("omegaconf")
    t = _compose(CFG / "paper" / "ts.yaml")["thundersync"]
    assert t["cohort_lanes"] == 2 and t["cohort_lane_lookahead"] == 2 and t["cohort_lane_lookahead_adaptive"]
    assert t["null_bv"] and t["null_bv_calibrate"] and t["engine_sampler"] == "wide"
    assert t["verify0_stream"] and t["learner_prefix_cache_verify0"]
    for k in ("THUNDERSYNC_SENTINEL_EVERY", "THUNDERSYNC_SENTINEL_ASYNC", "THUNDERSYNC_EARLY_HANDOUT",
              "THUNDERSYNC_V0_HOLD_TRAINING", "THUNDERSYNC_V0_NO_INTERLEAVE"):
        assert t["driver_env"][k] == "1", k
    # research hooks stay off in the paper config
    flat = repr(t["driver_env"])
    for k in ("LOOKAHEAD_DRAFT_ALPHA", "MASTER_JITTER", "ACC_DUMP", "THUNDERSYNC_DS2", "DEBUG"):
        assert k not in flat, k


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_paper_config_validates(arm):
    pytest.importorskip("omegaconf")
    from thundersync_rl.grpo_loop import ThunderSyncMasterConfig

    c = ThunderSyncMasterConfig(**_compose(CFG / "paper" / ARMS[arm][0]))
    assert c.policy["megatron_cfg"]["env_vars"]["NCCL_BUFFSIZE"] == "1048576"


def test_a_changed_setting_is_detected():
    """Negative control: the env route with one gate setting changed no longer equals the paper config."""
    pytest.importorskip("omegaconf")
    paper = _compose(CFG / "paper" / "ts.yaml")
    paper["thundersync"].pop("driver_env")
    args, _ = _golden("gate3_super_l22_ts")
    keep = [a for a in args if not a.startswith(LAYOUT)]
    keep = [a.replace("null_bv_drafts_per_output=1.75", "null_bv_drafts_per_output=2.0") for a in keep]
    cfg = _compose(CFG / "ab" / "ab_ts_scale.yaml", keep + ARMS["ts"][3])
    assert paper != cfg


def test_driver_env_apply_and_conflicts():
    env = {"A": "1"}
    assert driver_env.apply({"THUNDERSYNC_X": 1, "NEMO_RL_Y": "z"}, env) == {"THUNDERSYNC_X": "1", "NEMO_RL_Y": "z"}
    assert env == {"A": "1", "THUNDERSYNC_X": "1", "NEMO_RL_Y": "z"}
    driver_env.apply({"THUNDERSYNC_X": "1"}, env)  # same value: accepted
    with pytest.raises(ValueError, match="THUNDERSYNC_X"):
        driver_env.apply({"THUNDERSYNC_X": "0"}, env)
    assert env["THUNDERSYNC_X"] == "1"
    assert driver_env.apply({}, env) == {} and driver_env.apply(None, env) == {}


def test_driver_env_effective_filters_harness_variables():
    env = {"THUNDERSYNC_B": "1", "NEMO_RL_A": "x", "THUNDERSYNC_RUN_DIR": "/r", "NEMO_RL_SNAPSHOT_PINS": "/p", "PATH": "/b"}
    assert driver_env.effective(env) == {"NEMO_RL_A": "x", "THUNDERSYNC_B": "1"}
    assert list(driver_env.effective(env)) == ["NEMO_RL_A", "THUNDERSYNC_B"]
