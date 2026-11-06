"""Parity reward for the stock nrl_k arms on truncated models (Super-L22, Ultra-L27).

The ts / lanes-k runs on truncated models set THUNDERSYNC_DEBUG_PARITY_REWARD=1: the reward of a row is the crc32
parity of its tokens, so groups have nonzero advantages although the truncated model never solves a task. The stock
NeMo-RL loop has no such switch. This environment is NeMo-RL's MathEnvironment (same verify workers, same call
pattern and cost) with the reward replaced by crc32(assistant text) & 1. It only matches the reward *statistics*
(Bernoulli(1/2) per row, independent within a group) of the ts runs, not the exact values (ts hashes token ids,
prompt included). Nothing else in the stock loop changes. Full Nano uses the real DAPO reward on both sides.
"""

import zlib

import ray
import torch

from nemo_rl.environments.math_environment import MathEnvironment

_Base = MathEnvironment.__ray_actor_class__  # the undecorated class behind @ray.remote


@ray.remote(max_restarts=-1, max_task_retries=-1, max_concurrency=1000)  # pragma: no cover
class ParityMathEnvironment(_Base):
    def step(self, message_log_batch, metadata, return_extracted_answer=False):
        ret = super().step(message_log_batch, metadata, return_extracted_answer)
        texts = [
            "".join(str(m["content"]) for m in conv if m["role"] == "assistant")
            for conv in message_log_batch
        ]
        rewards = torch.tensor(
            [float(zlib.crc32(t.encode("utf-8", "surrogatepass")) & 1) for t in texts],
            dtype=ret.rewards.dtype,
        ).cpu()
        return ret._replace(rewards=rewards)  # EnvironmentReturn is a NamedTuple


FQN = "nrl_parity_env.ParityMathEnvironment"


def register() -> None:
    """Register env name ``math_parity`` (driver side, before setup creates the environments)."""
    from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
    from nemo_rl.environments.utils import ENV_REGISTRY, register_env

    ACTOR_ENVIRONMENT_REGISTRY[FQN] = PY_EXECUTABLES.SYSTEM
    if "math_parity" not in ENV_REGISTRY:
        register_env("math_parity", FQN)
