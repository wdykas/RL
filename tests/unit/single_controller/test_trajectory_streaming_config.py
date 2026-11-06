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
"""Config validation for async_rl.trajectory_streaming."""

from __future__ import annotations

from typing import Any, Callable

import pytest

from nemo_rl.algorithms.advantage_estimator import AdvEstimatorConfig
from nemo_rl.algorithms.async_utils.staleness_sampler import InOrderSamplerConfig
from nemo_rl.algorithms.grpo import GRPOConfig
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.single_controller_utils import AsyncRLConfig, MasterConfig
from nemo_rl.algorithms.single_controller_utils.config import (
    TokenCaptureConfig,
    TrajectoryStreamingConfig,
    _validate_trajectory_streaming,
)


def _config(**overrides: Callable[[MasterConfig], Any]) -> MasterConfig:
    mc = MasterConfig.model_construct(
        async_rl=AsyncRLConfig(
            sampler=InOrderSamplerConfig(max_lookahead_versions=0),
            trajectory_streaming=TrajectoryStreamingConfig(),
        ),
        policy={
            "train_micro_batch_size": 4,
            "sequence_packing": {"enabled": True},
            "dynamic_batching": {"enabled": False},
            "megatron_cfg": {
                "enabled": True,
                "mtp_num_layers": None,
                "expert_model_parallel_size": 1,
                "moe_router_load_balancing_type": "none",
                "distributed_data_parallel_config": {"grad_reduce_in_fp32": True},
            },
        },
        grpo=GRPOConfig.model_construct(
            seq_logprob_error_threshold=None,
            adv_estimator=AdvEstimatorConfig(name="grpo"),
        ),
        loss_fn=ClippedPGLossConfig(reference_policy_kl_penalty=0.0),
        env={},
        token_capture=TokenCaptureConfig(),
    )
    for name, apply in overrides.items():
        apply(mc)
    return mc


def test_accepts_supported_config():
    _validate_trajectory_streaming(_config())


@pytest.mark.parametrize("name", ["gdpo", "reinforce_plus_plus", "opd"])
def test_rejects_estimators_that_are_not_per_group_reward_functions(name):
    mc = _config(est=lambda mc: setattr(mc.grpo.adv_estimator, "name", name))
    with pytest.raises(ValueError, match="adv_estimator.name='grpo'"):
        _validate_trajectory_streaming(mc)


def test_rejects_positive_example_nll():
    mc = _config(nll=lambda mc: setattr(mc.loss_fn, "positive_example_nll_weight", 0.1))
    with pytest.raises(ValueError, match="positive_example_nll_weight=0"):
        _validate_trajectory_streaming(mc)


@pytest.mark.parametrize(
    "apply",
    [
        lambda mc: mc.env.update(should_use_nemo_gym=True),
        lambda mc: setattr(mc.token_capture, "enabled", True),
    ],
    ids=["nemo_gym", "token_capture"],
)
def test_rejects_rollout_paths_without_per_trajectory_publishing(apply):
    with pytest.raises(ValueError, match="native rollout path"):
        _validate_trajectory_streaming(_config(path=apply))


def test_fixed_size_microbatches_require_mbs_one():
    def fixed(mc):
        mc.policy["sequence_packing"]["enabled"] = False

    with pytest.raises(ValueError, match="train_micro_batch_size=1"):
        _validate_trajectory_streaming(_config(fixed=fixed))

    def fixed_mbs1(mc):
        fixed(mc)
        mc.policy["train_micro_batch_size"] = 1

    _validate_trajectory_streaming(_config(fixed=fixed_mbs1))

    def dynamic(mc):
        fixed(mc)
        mc.policy["dynamic_batching"]["enabled"] = True

    _validate_trajectory_streaming(_config(dynamic=dynamic))


def test_rejects_expert_parallelism():
    def ep2(mc):
        mc.policy["megatron_cfg"]["expert_model_parallel_size"] = 2

    with pytest.raises(ValueError, match="expert_model_parallel_size=1"):
        _validate_trajectory_streaming(_config(ep=ep2))
