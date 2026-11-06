import sys

import ray

sys.path.insert(0, "research/thundersync_rl")
from omegaconf import OmegaConf
from thundersync_rl.grpo_loop import ThunderSyncMasterConfig
from thundersync_rl.worker import WORKER_FQN

from nemo_rl.algorithms.utils import get_tokenizer
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
cfg = parse_hydra_overrides(
    load_config(
        "research/thundersync_rl/configs/grpo_math_1.5b_thundersync_megatron.yaml"
    ),
    sys.argv[1:],
)
pcfg = ThunderSyncMasterConfig(**OmegaConf.to_container(cfg, resolve=True)).policy
pcfg["megatron_cfg"]["train_iters"] = 1
init_ray()
cluster = RayVirtualCluster(
    bundle_ct_per_node_list=[2],
    use_gpus=True,
    num_gpus_per_node=2,
    max_colocated_worker_groups=1,
    name="refit_check",
)
policy = Policy(
    cluster=cluster,
    config=pcfg,
    tokenizer=get_tokenizer(pcfg["tokenizer"]),
    name_prefix="rc",
    init_reference_model=False,
    worker_extension_cls_fqn=WORKER_FQN,
)
for r in ray.get(
    policy.worker_group.run_all_workers_single_data("debug_refit_export_names")
):
    print("EXPORT", r)
