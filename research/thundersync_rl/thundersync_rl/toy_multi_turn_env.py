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
"""A stochastic multi-turn test environment (multi-turn exactness runs).

Each step answers the last assistant message with an observation of random
length and content, a reward that depends on the message and on the
environment's own randomness, and terminates at random or after
``max_turns`` steps. The environment is not deterministic (its RNG is never
reseeded per call), so a rollout loop that stepped it twice for the same
turn would see a different answer: speculative multi-turn rollouts must
record and replay its answers (``multi_turn.py``). The turn count travels in
the metadata (``toy_turn``) next to the dataset's ``extra_env_info``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import ray
import torch

from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

_WORDS = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "omega",
    "seven",
    "blue",
    "tool",
    "result",
    "ok",
)


@ray.remote(max_restarts=0, max_task_retries=0)  # pragma: no cover - Ray actor
class ToyMultiTurnEnvironment(EnvironmentInterface):
    def __init__(self, max_turns: int = 3, stop_prob: float = 0.25, seed: int = 0):
        self.max_turns = max_turns
        self.stop_prob = stop_prob
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def step(self, message_log_batch: list[list[dict[str, Any]]], metadata: list[Any]):
        observations, out_meta, rewards, terminated = [], [], [], []
        for log, meta in zip(message_log_batch, metadata):
            self.calls += 1
            meta = dict(meta or {})
            turn = int(meta.get("toy_turn", 0)) + 1
            last = log[-1]["content"] if log else ""
            done = turn >= self.max_turns or self.rng.random() < self.stop_prob
            n = int(self.rng.integers(2, 12))
            words = " ".join(self.rng.choice(_WORDS, size=n))
            observations.append(
                {"role": "environment", "content": f"\nObservation {turn}: {words}\n"}
            )
            rewards.append(float(len(last) % 7) / 7.0 + float(self.rng.random()) * 0.1)
            terminated.append(done)
            meta["toy_turn"] = turn
            out_meta.append(meta)
        return EnvironmentReturn(
            observations=observations,
            metadata=out_meta,
            next_stop_strings=[None] * len(observations),
            rewards=torch.tensor(rewards),
            terminateds=torch.tensor(terminated),
            answers=[None] * len(observations),
        )

    def global_post_process_and_metrics(self, batch):
        return batch, {"toy_env_calls": float(self.calls)}

    def shutdown(self) -> None:
        pass
