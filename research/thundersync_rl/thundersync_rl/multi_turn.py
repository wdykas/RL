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
"""Multi-turn rollouts of drafted plans under sticky routing (scheme A).

Every turn of a drafted rollout is sampled at the drafting weights
theta_{k-1} before the refit (``SpeculativeGeneration._continue_sticky``):
when a VERIFIED prefix ends a turn, the driver steps the environment on it
(``TurnEnvironment.step``; unverified draft tokens never reach the
environment or its tools), appends the observation and continues the rollout
from prompt + turns + observation. The rollout loop later rebuilds the
trajectory from the recorded turns (``replay``) instead of calling the
environment again: environments need not be deterministic, and tools may
have side effects.

The turn semantics are those of ``run_sample_multi_turn_rollout`` (the path
undrafted rollouts take): a turn ends at EOS or at the per-turn length cap,
the environment is stepped after every turn, an observation is truncated to
the total-length cap (which ends the rollout), and the rollout ends when the
environment terminates it or after ``max_turns`` turns.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Union

import torch

from nemo_rl.data.llm_message_utils import get_keys_from_message_log


@dataclass
class TurnRecord:
    """One finished turn of a drafted rollout and the environment's answer.

    ``gen_start``/``gen_end``: the turn's generated tokens in ``Plan.prefix``;
    ``input_length``: the turn's context length (prompt + earlier turns and
    observations). ``obs``: the observation's token ids (already truncated to
    the total-length cap).
    """

    gen_start: int
    gen_end: int
    input_length: int
    obs: list[int]
    obs_message: dict[str, Any]
    reward: Union[float, dict[str, float]]
    terminated: bool
    truncated: bool
    metadata: Any

    @property
    def last(self) -> bool:
        return self.terminated or self.truncated


class TurnEnvironment:
    """Steps the environment for drafted rollouts on the driver.

    ``rows``: the initial sample state of each rollout of the step (row order
    of the rollout loop: ``message_log``, ``extra_env_info``, ``task_name``,
    ``stop_strings``). One ``step`` per verified turn end; the token lists of
    one turn are all that crosses the driver (no per-token tensors).
    """

    def __init__(self, tokenizer, task_to_env, max_seq_len: int, max_turns: int):
        if max_turns < 2:
            raise ValueError(f"max_turns must be >= 2 (got {max_turns})")
        self.tokenizer = tokenizer
        self.task_to_env = task_to_env
        self.max_seq_len = int(max_seq_len)
        self.max_turns = int(max_turns)
        self.rows: list[dict[str, Any]] = []
        self.steps = 0
        # Wall time inside environment steps (summed over concurrent steps).
        self.env_s = 0.0

    def bind(self, rows: list[dict[str, Any]]) -> None:
        for i, row in enumerate(rows):
            if any(row.get("stop_strings") or []):
                raise ValueError(
                    f"row {i}: thundersync.speculate does not support per-sample "
                    "stop_strings (drafts run without them)"
                )
        self.rows = rows

    def message_log(
        self, row: int, prefix: list[int], turns: list[TurnRecord], logprobs=None
    ) -> list[dict[str, Any]]:
        """The row's message log: initial messages, then each turn and observation."""
        log = copy.deepcopy(self.rows[row]["message_log"])
        for t in turns:
            log.append(self._assistant(prefix[t.gen_start : t.gen_end], logprobs, t))
            log.append(
                dict(t.obs_message, token_ids=torch.tensor(t.obs, dtype=torch.long))
            )
        return log

    def _assistant(self, gen: list[int], logprobs, t: TurnRecord) -> dict[str, Any]:
        ids = torch.tensor(gen, dtype=torch.long)
        msg = {
            "role": "assistant",
            "content": self.tokenizer.batch_decode([ids], skip_special_tokens=True)[0],
            "token_ids": ids,
        }
        if logprobs is not None:
            msg["generation_logprobs"] = torch.tensor(
                logprobs[t.gen_start : t.gen_end], dtype=torch.float
            )
        return msg

    def env_info(self, row: int, turns: list[TurnRecord]) -> Any:
        """``extra_env_info`` in effect after ``turns`` (updated by non-final turns)."""
        info = self.rows[row]["extra_env_info"]
        for t in turns:
            if not t.last and t.metadata is not None:
                info = t.metadata
        return info

    async def step(
        self,
        row: int,
        prompt_len: int,
        prefix: list[int],
        turns: list[TurnRecord],
        turn_start: int,
    ) -> TurnRecord:
        """Step the environment on the turn ``prefix[turn_start:]`` of row ``row``."""
        record = TurnRecord(
            gen_start=turn_start,
            gen_end=len(prefix),
            input_length=prompt_len + turn_start,
            obs=[],
            obs_message={},
            reward=0.0,
            terminated=False,
            truncated=False,
            metadata=None,
        )
        log = self.message_log(row, prefix, turns)
        log.append(self._assistant(prefix[turn_start:], None, record))
        task = self.rows[row]["task_name"]
        if task not in self.task_to_env:
            raise ValueError(f"No environment found for task type: {task}")
        # One environment call per verified turn end, awaited on the driver's
        # event loop (no thread per call: thousands of rollouts may wait on
        # slow tools at once). Same request as ``calculate_rewards`` makes.
        t0 = time.perf_counter()
        out = await self.task_to_env[task].step.remote(
            [get_keys_from_message_log(log, ["role", "content"])],
            [copy.deepcopy(self.env_info(row, turns))],
        )
        self.env_s += time.perf_counter() - t0
        self.steps += 1
        if isinstance(out.rewards, dict):
            record.reward = {k: float(r[0]) for k, r in out.rewards.items()}
        else:
            record.reward = float(out.rewards[0])
        record.terminated = bool(out.terminateds[0])
        content = out.observations[0]["content"]
        obs = self.tokenizer(
            content, return_tensors="pt", add_special_tokens=False
        ).input_ids[0]
        gen = record.gen_end - record.gen_start
        if record.input_length + gen + len(obs) >= self.max_seq_len:
            obs = obs[: max(0, self.max_seq_len - record.input_length - gen)]
            record.truncated = True
        record.obs = [int(t) for t in obs.tolist()]
        record.obs_message = {"role": out.observations[0]["role"], "content": content}
        if not record.last:
            if out.next_stop_strings is not None and out.next_stop_strings[0] is not None:
                raise ValueError(
                    "thundersync.speculate does not support environment stop strings "
                    "(drafts and continuations run without them)"
                )
            record.metadata = out.metadata[0]
        return record

    def rollout_ends(self, turns: list[TurnRecord]) -> bool:
        return bool(turns) and (turns[-1].last or len(turns) >= self.max_turns)


def replay(
    env: TurnEnvironment,
    row: int,
    prefix: list[int],
    prefix_logprobs: list[float],
    turns: list[TurnRecord],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(final sample state, sample metrics) of a drafted multi-turn rollout.

    The same structure ``run_sample_multi_turn_rollout`` returns, rebuilt from
    the turns recorded while the driver sampled them (the environment is not
    called again).
    """
    if not env.rollout_ends(turns):
        raise RuntimeError(
            f"row {row}: a drafted multi-turn rollout reached the rollout loop "
            "unfinished (verify_sticky must complete every turn)"
        )
    init = env.rows[row]
    total = 0.0
    named: dict[str, float] = {}
    for t in turns:
        if isinstance(t.reward, dict):
            for k, v in t.reward.items():
                named[k] = named.get(k, 0.0) + v
            total += sum(t.reward.values())
        else:
            total += t.reward
    gen_tokens = [t.gen_end - t.gen_start for t in turns]
    env_tokens = sum(len(t.obs) for t in turns)
    state = {
        "message_log": env.message_log(row, prefix, turns, prefix_logprobs),
        "extra_env_info": env.env_info(row, turns),
        "task_name": init["task_name"],
        "total_reward": torch.tensor(total),
        "stop_strings": init.get("stop_strings"),
        "idx": init.get("idx", row),
    }
    for k, v in named.items():
        state[k] = torch.tensor(v)
    metrics = {
        "turn_count": len(turns),
        "total_tokens": sum(gen_tokens) + env_tokens,
        "assistant_tokens": sum(gen_tokens),
        "env_tokens": env_tokens,
        "terminated": turns[-1].terminated,
        "truncated": turns[-1].truncated,
        "max_turns_reached": len(turns) >= env.max_turns,
        "total_reward": total,
        "turn_gen_tokens": gen_tokens,
        "turn_input_tokens": [t.input_length for t in turns],
        "turn_total_tokens": [t.input_length + g for t, g in zip(turns, gen_tokens)],
        "max_gen_tokens_per_turn": max(gen_tokens),
        "per_worker_token_counts": {0: sum(gen_tokens)},
    }
    return state, metrics


def pair_rows(
    rows: list[torch.Tensor], plans: dict[tuple[int, ...], list[Any]]
) -> dict[int, Any]:
    """Row index -> plan: rows with the same prompt take its plans in queue order.

    The order of a prompt's plans does not depend on their tokens (drafting
    order), so which row gets which plan is content-independent. Paired plans
    leave ``plans``; rows without one decode undrafted (plain multi-turn).
    """
    out: dict[int, Any] = {}
    for i, p in enumerate(rows):
        queue = plans.get(tuple(int(t) for t in p.tolist()))
        if queue:
            out[i] = queue.pop(0)
    return out


__all__ = ["TurnEnvironment", "TurnRecord", "pair_rows", "replay"]
