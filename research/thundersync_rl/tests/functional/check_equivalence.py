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
"""GPU check: streamed GRPO updates == batch-synchronous GRPO updates.

Mirrors the paper's Appendix G replay: two Megatron policies built from the
same checkpoint on disjoint GPUs consume identical logical batches. Policy A
runs the stock ``Policy.train`` with final advantages. Policy B receives the
same trajectories in a random arrival order through the streaming planner and
accumulator. We report the relative parameter gap

    Delta_k = ||theta_k^stream - theta_k^sync|| / ||theta_k^sync - theta_0||

after every update, and fail if it exceeds ``--tol``.
"""

import argparse
import random

import ray
import torch
from omegaconf import OmegaConf
from thundersync_rl.grpo_loop import ThunderSyncMasterConfig, make_group_advantage_fn
from thundersync_rl.streaming import (
    StreamingLearner,
    StreamPlanner,
    Trajectory,
    pad_batch_to_multiple,
)
from thundersync_rl.worker import WORKER_FQN

from nemo_rl.algorithms.loss import ClippedPGLossFn
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import (
    PY_EXECUTABLES,
    RayVirtualCluster,
    init_ray,
)
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)

ACTOR_ENVIRONMENT_REGISTRY[WORKER_FQN] = PY_EXECUTABLES.MCORE


def make_batch(num_groups, group_size, seq_len, vocab, seed):
    g = torch.Generator().manual_seed(seed)
    n = num_groups * group_size
    input_ids = torch.randint(0, vocab, (n, seq_len), generator=g)
    prompt_len = torch.randint(8, seq_len // 4, (n,), generator=g)
    lengths = torch.randint(seq_len // 2, seq_len + 1, (n,), generator=g)
    pos = torch.arange(seq_len).unsqueeze(0)
    token_mask = (
        (pos >= prompt_len.unsqueeze(1)) & (pos < lengths.unsqueeze(1))
    ).float()
    rewards = (torch.rand(n, generator=g) < 0.4).float().tolist()
    zeros = torch.zeros(n, seq_len)
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": lengths,
            "token_mask": token_mask,
            "sample_mask": torch.ones(n),
            "generation_logprobs": zeros.clone(),
            "prev_logprobs": zeros.clone(),
            "reference_policy_logprobs": zeros.clone(),
        }
    )
    return data, rewards


def rows(data: BatchedDataDict, idx: list[int]) -> BatchedDataDict:
    return BatchedDataDict({k: v[idx].clone() for k, v in data.items()})


def get_params(policy: Policy) -> dict[str, torch.Tensor]:
    return ray.get(policy.worker_group.run_all_workers_single_data("get_flat_params"))[
        0
    ]


def get_grads(policy: Policy) -> torch.Tensor:
    return ray.get(policy.worker_group.run_all_workers_single_data("get_flat_grads"))[0]


def flat_diff_norm(a, b):
    return torch.sqrt(sum(((a[k] - b[k]) ** 2).sum() for k in a)).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--gpus-per-policy", type=int, default=1)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--num-groups", type=int, default=4)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument(
        "--mode",
        choices=["stream", "permuted_sync", "mbs_sync"],
        default="stream",
        help="permuted_sync: control run where policy B does sync train() on a "
        "row-permuted batch, i.e. a batching perturbation without streaming",
    )
    p.add_argument(
        "--simulate-retry",
        action="store_true",
        help="stream group 0's trajectories, then discard them as a retried "
        "rollout would, before the normal stream; the update must not change",
    )
    p.add_argument(
        "--positive-advantages",
        action="store_true",
        help="use advantage 1 for every row (no sign cancellation in the gradient)",
    )
    args, overrides = p.parse_known_args()

    register_omegaconf_resolvers()
    cfg = load_config(args.config)
    if overrides:
        cfg = parse_hydra_overrides(cfg, overrides)
    cfg = ThunderSyncMasterConfig(**OmegaConf.to_container(cfg, resolve=True))
    pcfg = cfg.policy
    n = args.num_groups * args.group_size
    G = args.group_size
    pcfg["train_global_batch_size"] = n
    pcfg["megatron_cfg"]["train_iters"] = args.steps  # normally set by grpo.setup()
    cfg.grpo.num_generations_per_prompt = G
    tokenizer = get_tokenizer(pcfg["tokenizer"])
    mbs = pcfg["train_micro_batch_size"]

    init_ray()
    policies = []
    for name in ("a", "b"):
        cluster = RayVirtualCluster(
            bundle_ct_per_node_list=[args.gpus_per_policy],
            use_gpus=True,
            num_gpus_per_node=args.gpus_per_policy,
            max_colocated_worker_groups=1,
            name=f"equiv_{name}",
        )
        policies.append(
            Policy(
                cluster=cluster,
                config=pcfg,
                tokenizer=tokenizer,
                name_prefix=f"equiv_{name}",
                init_reference_model=False,
                worker_extension_cls_fqn=WORKER_FQN,
            )
        )
    policy_a, policy_b = policies
    learner = StreamingLearner(policy_b)
    loss_fn = ClippedPGLossFn(cfg.loss_fn)
    adv_fn = make_group_advantage_fn(cfg)
    ts = cfg.thundersync
    compare_grads = learner.dp_size == 1

    def run_streamed(data, rewards, rng):
        planner = StreamPlanner(
            adv_fn,
            dp_size=learner.dp_size,
            max_chunk_trajectories=ts.max_chunk_trajectories,
            group_only=False,
            max_open_groups=ts.max_open_groups,
        )
        for gi in range(args.num_groups):
            planner.register_group(gi, G)
        learner.begin(
            loss_fn,
            gbs=n,
            mbs=mbs,
            storage_device=ts.storage_device,
            max_buckets_per_group=ts.max_buckets_per_group,
            max_open_groups=ts.max_open_groups,
        )

        def dispatch():
            d = planner.next_dispatch()
            per_rank = []
            for chunks in d.per_rank:
                items = []
                for c in chunks:
                    idx = [t.index for t in c.trajectories]
                    cd = rows(data, idx)
                    a = c.advantages if c.group is None else [1.0] * len(idx)
                    cd["advantages"] = (
                        torch.tensor(a)
                        .unsqueeze(-1)
                        .expand_as(cd["token_mask"])
                        .clone()
                    )
                    items.append(
                        {
                            "data": pad_batch_to_multiple(cd, mbs),
                            "group": c.group,
                            "reward": c.reward,
                        }
                    )
                per_rank.append(items)
            ray.get(learner.submit(per_rank, d.closes))

        if args.simulate_retry:
            idx = list(range(G))
            per_rank = [[] for _ in range(learner.dp_size)]
            for r in sorted({rewards[i] for i in idx}):
                rows_r = [i for i in idx if rewards[i] == r]
                cd = rows(data, rows_r)
                cd["advantages"] = torch.ones_like(cd["token_mask"])
                per_rank[0].append(
                    {"data": pad_batch_to_multiple(cd, mbs), "group": 0, "reward": r}
                )
            ray.get(learner.submit(per_rank, []))
            ray.get(
                learner.worker_group.run_all_workers_single_data(
                    "stream_discard", groups=[0]
                )
            )

        order = list(range(n))
        rng.shuffle(order)
        for i in order:
            planner.add(
                Trajectory(
                    group=i // G,
                    index=i,
                    reward=rewards[i],
                    payload=None,
                    num_tokens=int(data["input_lengths"][i]),
                )
            )
            if rng.random() < 0.3:
                dispatch()
        while planner.has_work():
            dispatch()
        assert planner.all_done()
        return learner.finish()

    theta0 = get_params(policy_a)
    assert flat_diff_norm(theta0, get_params(policy_b)) == 0.0
    vocab = min(tokenizer.vocab_size, 32000)
    rng = random.Random(0)
    worst = {"grad": 0.0, "delta": 0.0}
    for step in range(args.steps):
        data, rewards = make_batch(args.num_groups, G, args.seq_len, vocab, seed=step)
        advs = []
        for gi in range(args.num_groups):
            advs += adv_fn(rewards[gi * G : (gi + 1) * G])
        sync_data = BatchedDataDict(dict(data))
        sync_data["advantages"] = (
            torch.tensor(advs).unsqueeze(-1).expand_as(data["token_mask"]).clone()
        )

        if args.positive_advantages:
            sync_data["advantages"] = torch.ones_like(sync_data["advantages"])
        res_a = policy_a.train(sync_data, loss_fn)
        if args.mode == "mbs_sync":
            # Same data, same order: only the micro-batch size differs.
            res_b = policy_b.train(sync_data, loss_fn, mbs=1)
        elif args.mode == "permuted_sync":
            perm = list(range(n))
            rng.shuffle(perm)
            res_b = policy_b.train(rows(sync_data, perm), loss_fn)
        else:
            res_b = run_streamed(data, rewards, rng)

        if args.mode == "mbs_sync" and step == 0:
            na = ray.get(
                policy_a.worker_group.run_all_workers_single_data("get_named_grads")
            )[0]
            nb = ray.get(
                policy_b.worker_group.run_all_workers_single_data("get_named_grads")
            )[0]
            rows_ = sorted(
                (
                    (((na[k] - nb[k]).norm() / na[k].norm().clamp_min(1e-30)).item(), k)
                    for k in na
                ),
                reverse=True,
            )
            print("per-parameter grad rel err (worst 12):", flush=True)
            for e, k in rows_[:12]:
                print(f"  {e:.3e}  {k}", flush=True)
            print(f"  median {rows_[len(rows_) // 2][0]:.3e}", flush=True)
            for e, k in rows_:
                if any(
                    t in k
                    for t in (
                        "embedding",
                        "output_layer",
                        "final_layernorm",
                        "layers.0.",
                        "layers.23.",
                    )
                ):
                    print(f"  [probe] {e:.3e}  {k}", flush=True)
        grad_err = float("nan")
        if compare_grads:
            ga, gb = get_grads(policy_a), get_grads(policy_b)
            grad_err = ((ga - gb).norm() / ga.norm()).item()
            worst["grad"] = max(worst["grad"], grad_err)
        ta, tb = get_params(policy_a), get_params(policy_b)
        delta = flat_diff_norm(ta, tb) / max(flat_diff_norm(ta, theta0), 1e-30)
        worst["delta"] = max(worst["delta"], delta)
        print(
            f"[equiv step {step}] mode={args.mode} "
            f"grad_norm a={float(torch.as_tensor(res_a['grad_norm']).mean()):.6e} "
            f"b={float(torch.as_tensor(res_b['grad_norm']).mean()):.6e} "
            f"grad_rel_err={grad_err:.3e} Delta={delta:.3e} "
            f"peak_open_buffers={res_b.get('stream_peak_open_buffers')}",
            flush=True,
        )
    passed = worst["grad"] <= args.tol if compare_grads else worst["delta"] <= args.tol
    status = "PASS" if passed else "FAIL"
    print(
        f"EQUIVALENCE {status} ({args.mode}): max grad rel err = {worst['grad']:.3e}, "
        f"max Delta_k = {worst['delta']:.3e} (tol {args.tol})",
        flush=True,
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
