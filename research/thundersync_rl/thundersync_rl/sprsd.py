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
"""SPR-SD (``thundersync.mode=sprsd``): the stock colocated partial-rollout loop (SPR-BI) with XS-SD's drafter.

The fair column for XS-SD (research/colo/results/17-...): SPR-BI's loop, sampler gate and lag are untouched (no
ThunderSync stage: ``SingleController._init_nolanes`` returns None for this mode). Only the engine changes, exactly as
in XS-SD: the 53-bit keyed sampler (same law as the stock sampler) and tree_spec speculation on every row
(own-history candidates, optionally the MTP head as candidate 0), installed by each generation worker right after it
builds its engine (``ThunderSyncMegatronPolicyWorker._initialize_inference_engine``). Tree speculation emits the
keyed one-token decode's tokens, so the rows have the law of SPR-BI's rows; the only difference is speed.

Per rank: the tree's node rows / private KV / Mamba scratch (``tree_xssd`` W:D:rows) and the MTP head's weights + KV
slots, as XS-SD; no collectives beyond the engine's own.
"""

from __future__ import annotations

import os
import zlib
from typing import Any

MODE = "sprsd"
# Generation-worker env: the keyed sampler's salt; its presence makes the worker install sampler + tree at engine build.
SALT_ENV = "THUNDERSYNC_SPRSD_SALT"
# thundersync keys an SPR-SD arm may carry: the drafter, and the cross-fade band (F3: part of the policy definition, the
# same in every arm; the launcher writes it to the thundersync block and to policy.megatron_cfg). Any stale-row stage
# key (wake_verify, k_drafts, spare_prompts, variable_group, ...) is refused: SPR-SD verifies nothing.
BAND_KEYS = {"crossfade_routing_m", "crossfade_routing_m_profile", "crossfade_routing_extra",
             "crossfade_routing_ramp_stopgrad", "crossfade_routing_sparse", "crossfade_routing_sparse_engine",
             "crossfade_off_reason"}
ALLOWED_KEYS = {"mode", "tree_xssd", "tree_xssd_mtp"} | BAND_KEYS


def _get(d: Any, path: str, default: Any = None) -> Any:
    from thundersync_rl import colo

    return colo._get(d, path, default)


def config_problems(config: Any) -> list[str]:
    from thundersync_rl import nolanes

    p: list[str] = []
    ts = _get(config, "thundersync") or {}
    pol = _get(config, "policy") or {}
    gen = _get(pol, "generation") or {}
    mcfg = _get(pol, "megatron_cfg") or {}
    mcore = _get(gen, "mcore_generation_config") or {}
    if _get(gen, "backend") != "megatron":
        p.append("policy.generation.backend must be megatron")
    if not _get(mcfg, "batch_invariant_mode", False):
        p.append("sprsd is SPR-BI + drafter: policy.megatron_cfg.batch_invariant_mode must be true (tree_spec exactness)")
    spec = _get(ts, "tree_xssd")
    try:
        w, d, rows = (int(x) for x in str(spec).split(":"))
        if w < 1 or d < 1 or rows < 1:
            raise ValueError
    except ValueError:
        p.append(f"thundersync.tree_xssd must be W:D:rows (got {spec!r})")
    xm = _get(ts, "tree_xssd_mtp")
    if xm:  # as colo: path:slots
        path, _, slots = str(xm).rpartition(":")
        if not path or not os.path.isfile(path):
            p.append(f"thundersync.tree_xssd_mtp: no head file {path!r} (format path:slots)")
        if not slots.isdigit() or not 1 <= int(slots) <= 1024:
            p.append("thundersync.tree_xssd_mtp slots must be in 1..1024 (format path:slots)")
        if int(_get(mcfg, "tensor_model_parallel_size", 1) or 1) != 1:
            p.append("thundersync.tree_xssd_mtp needs tensor_model_parallel_size 1 (full-vocab output layer)")
    if _get(mcore, "enable_prefix_caching", False) and not nolanes.mcore_prefix_cache_weight_scoped():
        p.append("prefix caching needs a Megatron-Core whose prefix cache is scoped by weight version")
    # One band: band keys in the thundersync block must equal policy.megatron_cfg's (the stock loop applies only the
    # latter; a different value there would be a second band that silently does not apply)
    for k in sorted(BAND_KEYS & set(ts)):
        if k != "crossfade_off_reason" and ts.get(k) != mcfg.get(k):
            p.append(f"thundersync.{k}={ts.get(k)!r} differs from policy.megatron_cfg.{k}={mcfg.get(k)!r} (sprsd routes "
                     "with policy.megatron_cfg's band only)")
    unknown = sorted(set(ts) - ALLOWED_KEYS)
    if unknown:
        p.append(f"thundersync keys not used by sprsd: {unknown}")
    p += nolanes.router_frozen_problems(config)
    return p


def configure_driver(config: dict) -> None:
    """Driver side, before setup_single_controller: validate; ThunderSync worker class; the engine-side env."""
    from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES

    from thundersync_rl import nolanes

    problems = config_problems(config)
    if problems:
        raise ValueError("thundersync.mode=sprsd: " + "; ".join(problems) + ".")
    ts = config["thundersync"]
    nolanes.derive_layout_fields(config)
    ACTOR_ENVIRONMENT_REGISTRY[nolanes.WORKER_FQN] = PY_EXECUTABLES.MCORE
    config["policy"]["worker_extension_cls_fqn"] = nolanes.WORKER_FQN
    mcfg = config["policy"]["megatron_cfg"]
    seed = int(_get(config, "grpo.seed", 0) or 0)
    salt = zlib.crc32(f"wide/{seed}/sprsd".encode()) | ((seed & 0x7FFFFFFF) << 32)
    w, d, rows = (int(x) for x in str(ts["tree_xssd"]).split(":"))
    # the stock loop prints no per-step [tree] line: rank 0's tree_spec prints its own every THUNDERSYNC_TREE_LOG_S s
    env = {SALT_ENV: str(salt), "THUNDERSYNC_TAIL_TREE": f"{w}:{d}:{rows}:1",
           "THUNDERSYNC_TREE_LOG_S": os.environ.get("THUNDERSYNC_TREE_LOG_S", "60")}
    if ts.get("tree_xssd_mtp"):
        env["THUNDERSYNC_TAIL_TREE_MTP"] = str(ts["tree_xssd_mtp"])  # path:slots, as colo
    mcfg["env_vars"] = dict(mcfg.get("env_vars") or {}, **env)
    os.environ.update(env)
    print(f"[sprsd] stock loop (lag {int(_get(config, 'async_rl.sampler.max_lookahead_versions', 0) or 0)}) + keyed "
          f"sampler + tree_spec {w}:{d}:{rows}" + (f" + MTP {ts['tree_xssd_mtp']}" if ts.get("tree_xssd_mtp") else ""),
          flush=True)
