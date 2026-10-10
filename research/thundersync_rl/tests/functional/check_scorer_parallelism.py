"""Compare fp32-scorer token logprobs with NeMo-RL get_logprobs (same weights, PP/TP-aware)."""

import sys

import ray
import torch

sys.argv = [
    sys.argv[0],
    "--config",
    "research/thundersync_rl/configs/grpo_math_1.5b_thundersync_megatron.yaml",
] + sys.argv[1:]
sys.path.insert(0, "research/thundersync_rl")
from omegaconf import OmegaConf
from thundersync_rl.grpo_loop import ThunderSyncMasterConfig
from thundersync_rl.worker import WORKER_FQN

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
register_omegaconf_resolvers()
cfg = load_config(sys.argv[2])
cfg = parse_hydra_overrides(cfg, sys.argv[3:]) if len(sys.argv) > 3 else cfg
pcfg = ThunderSyncMasterConfig(**OmegaConf.to_container(cfg, resolve=True)).policy
pcfg["train_global_batch_size"] = 2
pcfg["megatron_cfg"]["train_iters"] = 1
init_ray()
cluster = RayVirtualCluster(
    bundle_ct_per_node_list=[2],
    use_gpus=True,
    num_gpus_per_node=2,
    max_colocated_worker_groups=1,
    name="pp_check",
)
tok = get_tokenizer(pcfg["tokenizer"])
policy = Policy(
    cluster=cluster,
    config=pcfg,
    tokenizer=tok,
    name_prefix="ppc",
    init_reference_model=False,
    worker_extension_cls_fqn=WORKER_FQN,
)
g = torch.Generator().manual_seed(0)
rows = [torch.randint(100, 30000, (n,), generator=g) for n in (96, 160, 64, 128)]
width = max(r.numel() for r in rows)
ids = torch.zeros(len(rows), width, dtype=torch.long)
for i, r in enumerate(rows):
    ids[i, : r.numel()] = r
data = BatchedDataDict(
    {"input_ids": ids, "input_lengths": torch.tensor([r.numel() for r in rows])}
)
ref = policy.get_logprobs(data)["logprobs"]
res = [
    x
    for x in ray.get(
        policy.worker_group.run_all_workers_single_data(
            "scorer_token_logprobs", rows=rows
        )
    )
    if x is not None
][0]
for i, r in enumerate(rows):
    a = ref[i, 1 : r.numel()].float()
    b = res[i]
    print(
        f"row {i}: max|diff|={float((a - b).abs().max()):.4f} mean|diff|={float((a - b).abs().mean()):.5f} mean lp={float(b.mean()):.3f}"
    )
