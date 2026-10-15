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
    check_multi_turn_template,
    configure_sticky_policy,
    thundersync_grpo_train,
)
from thundersync_rl.lanes import build_engine_lanes
from thundersync_rl.run_state import check_schedule_unchanged, load_state, resolve_resume
from thundersync_rl.refit_m2n import crosscheck_enabled, install_crosscheck
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
    parser.add_argument(
        "--toy-multi-turn-env",
        type=int,
        default=None,
        metavar="MAX_TURNS",
        help="Test runs: replace every task's environment with the stochastic "
        "multi-turn toy environment (thundersync_rl/toy_multi_turn_env.py) ending "
        "rollouts after at most MAX_TURNS turns.",
    )
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

    # Resume is explicit: setup() below loads the latest checkpoint_dir/step_*
    # whenever one exists; refuse that unless checkpointing is enabled and it
    # is a ThunderSync checkpoint (run_state.resolve_resume).
    ckpt_cfg = config.checkpointing
    if not isinstance(ckpt_cfg, dict):
        ckpt_cfg = dict(ckpt_cfg)
    resume_path = resolve_resume(ckpt_cfg)
    start_step = int(load_state(resume_path)["next_step"]) if resume_path else 0
    if resume_path:
        check_schedule_unchanged(
            resume_path, config.grpo.max_num_steps, config.grpo.max_num_epochs
        )
        print(f"[thundersync] resuming from {resume_path} (next step {start_step})", flush=True)

    init_ray()
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    check_multi_turn_template(config, tokenizer)
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer
    )
    dataset, val_dataset, task_to_env, val_task_to_env = setup_response_data(
        tokenizer, config.data, config.env
    )
    if args.toy_multi_turn_env is not None:
        from thundersync_rl.toy_multi_turn_env import ToyMultiTurnEnvironment

        shutdown_environments(task_to_env)
        toy = ToyMultiTurnEnvironment.remote(max_turns=args.toy_multi_turn_env)
        task_to_env = {name: toy for name in task_to_env}
    (
        policy,
        policy_generation,
        _nemo_gym,
        clusters,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
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

    if crosscheck_enabled():
        # Debug: every refit also runs the packed broadcast and compares the
        # engine weights bitwise (refit_m2n.py; refit_transport mcore only).
        install_crosscheck(policy, policy_generation, *clusters)
    lanes = None
    if master_config.thundersync.cohort_lanes > 1:
        # Cohort lanes (DESIGN.md section 14j): setup built lane 0 on
        # policy.generation.colocated.resources; one more engine of that shape.
        train_cluster, inference_cluster = clusters
        lanes = build_engine_lanes(
            policy,
            policy_generation,
            train_cluster,
            inference_cluster,
            master_config.policy,
            tokenizer,
            num_lanes=master_config.thundersync.cohort_lanes,
            start_step=start_step,
        )

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
            lanes=lanes,
            checkpointer=checkpointer,
            grpo_save_state=grpo_state,
            val_dataloader=val_dataloader,
            val_task_to_env=val_task_to_env,
            resume_path=str(resume_path) if resume_path else None,
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
    extra_lanes = [lane.generation for lane in (lanes or [])[1:]]
    for obj in (*extra_lanes, policy_generation, policy, *(clusters or ())):
        shutdown = getattr(obj, "shutdown", None)
        if shutdown is not None:
            try:
                shutdown()
            except Exception as exc:  # teardown only: report, keep going
                print(f"[thundersync] shutdown of {type(obj).__name__} failed: {exc}")
    ray.shutdown()


if __name__ == "__main__":
    main()
