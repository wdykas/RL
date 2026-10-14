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
"""ThunderSyncRL GRPO driver (Megatron training + non-colocated Megatron inference)."""

import argparse
import json
import os
import pprint
import sys
import traceback

import ray
from omegaconf import OmegaConf
from thundersync_rl.grpo_loop import (
    ThunderSyncMasterConfig,
    configure_sticky_policy,
    thundersync_grpo_train,
)
from thundersync_rl.worker import WORKER_FQN

from nemo_rl.algorithms.grpo import setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES, init_ray
from nemo_rl.environments.utils import shutdown_environments
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

ACTOR_ENVIRONMENT_REGISTRY[WORKER_FQN] = PY_EXECUTABLES.MCORE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--history-json", type=str, default=None)
    args, overrides = parser.parse_known_args()

    register_omegaconf_resolvers()
    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    config = ThunderSyncMasterConfig(**OmegaConf.to_container(config, resolve=True))
    config.policy["worker_extension_cls_fqn"] = WORKER_FQN
    # Sticky MoE routing: the Megatron workers read the threshold from the
    # policy config when they build the model (RouterReplay instances).
    configure_sticky_policy(config)
    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    pprint.pprint(config)

    init_ray()
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer
    )
    dataset, val_dataset, task_to_env, _ = setup_response_data(
        tokenizer, config.data, config.env
    )
    (
        policy,
        policy_generation,
        _nemo_gym,
        _clusters,
        dataloader,
        _val_dataloader,
        loss_fn,
        logger,
        _checkpointer,
        _grpo_state,
        master_config,
        _teachers,
        _aliases,
    ) = setup(config, tokenizer, dataset, val_dataset)

    def write_history(history) -> None:
        # Rewritten after every step (atomic rename): a run cut short (timeout,
        # crash) still leaves every completed step's metrics for the checkers.
        if not args.history_json:
            return
        path = os.path.abspath(args.history_json)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w") as f:
            json.dump(history, f, indent=1)
        os.replace(path + ".tmp", path)

    try:
        history = thundersync_grpo_train(
            policy=policy,
            policy_generation=policy_generation,
            dataloader=dataloader,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            task_to_env=task_to_env,
            master_config=master_config,
            logger=logger,
            on_step=write_history,
        )
    except BaseException:
        # A failed learner or engine rank can leave its peers blocked in a
        # collective (EP all-to-all, PP send/recv) while holding their GPU memory,
        # and Ray's normal teardown waits on them. Exit the driver at once instead:
        # Ray then kills every worker actor this job owns, which frees the GPUs and
        # ends the job.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        shutdown_environments(task_to_env)
    write_history(history)
    # Release the workers and placement groups while Ray is still connected.
    # Left to garbage collection they are shut down from __del__ during
    # interpreter finalization, after Ray's atexit teardown: the Ray calls there
    # reconnect a new driver to the cluster and crash in CoreWorkerProcess
    # (exit 1 after a completed run).
    for obj in (policy_generation, policy, *(_clusters or ())):
        shutdown = getattr(obj, "shutdown", None)
        if shutdown is not None:
            try:
                shutdown()
            except Exception as exc:  # teardown only: report, keep going
                print(f"[thundersync] shutdown of {type(obj).__name__} failed: {exc}")
    ray.shutdown()


if __name__ == "__main__":
    main()
