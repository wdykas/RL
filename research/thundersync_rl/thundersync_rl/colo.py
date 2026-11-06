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
"""ThunderSync colocated mode (``thundersync.mode=colo``; docs/colo/THEORY.md).

One GPU fleet generates and trains: NeMo-RL's SingleController loop with a colocated Megatron engine (the BI fp16
engine, a dedicated inference model resharded from the learner at every wake), lag 0. Every token of a step's batch
is sampled at theta_k by an engine whose log-probs equal the learner's bitwise, so the batch IS an exact theta_k
sample: there is nothing to verify (MODEL.md §0, arm XS).

What this mode adds to the stock colocated loop:
- the 53-bit keyed engine sampler (``install_wide_sampler``, the nolanes engine sampler: exact sampling law);
- always on: the train-lp sentinel (every trained token's training log-prob == its generation log-prob, bitwise)
  and the router-frozen check (user rule 11-02);
- CHECKS=1 (``THUNDERSYNC_NL_CHECKS``): a learner forward of every output token (mismatch: bitwise), the randomized
  PIT law test of every output token and of every row's first token (thundersync_rl/law_check.py);
- the negative control ``NEMO_RL_DEBUG_COLOC_SKIP_RESHARD=1`` (engine keeps theta_0): sentinel and law must fail.

Exactness conditions of the mode (validator): colocated Megatron generation, batch-invariant numerics on learner and
engine (zero_train_gen_mismatch), router weights + bias frozen, persist KV (or offload with the TMS=1
torch_memory_saver overlay), lag 0 (every row of batch k is admitted after the wake that installs theta_k).

Prefix caching and chunked prefill are allowed on a Megatron-Core whose prefix cache is scoped by weight version: it
salts the cache with a weight epoch bumped at every engine resume (the wake that installs theta_k), so a row of batch k
can only match blocks computed under theta_k, and its batch-invariant engine keeps prefill chunks and restored Mamba
states on SSM chunk boundaries (bitwise the unsplit prefill). The wide sampler skips the discarded draw of a partial
prefill chunk (worker.install_wide_sampler).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any, Literal, Optional

MODE = "colo"
SKIP_RESHARD_ENV = "NEMO_RL_DEBUG_COLOC_SKIP_RESHARD"
# Process-wide engine levers that raise in a learner process (megatron_worker.install_engine_latency_levers).
ENGINE_ONLY_ENV = ("NEMO_RL_ENGINE_SYMM_TP", "NEMO_RL_ENGINE_FAST_LOGSOFTMAX")


def _get(d: Any, path: str, default: Any = None) -> Any:
    cur = d
    for part in path.split("."):
        if cur is None:
            return default
        cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
    return default if cur is None else cur


@dataclass
class ColoConfig:
    """``thundersync:`` block of an SC config in colo mode (every key optional except ``mode``)."""

    mode: Literal["colo"] = MODE
    seed_stream: str = "colo"
    wide_sampler_key64: bool = True
    # XS-SD on tree_spec (research/colo/NEXT.md "XS-SD on the single path"): "W:D:rows" opts EVERY fresh row into the
    # engine's exact tree speculation with candidates from the row's own history (ts_tree src bit1). Same keys as
    # one-token decode, so the rows are XS's token for token. None: plain XS.
    tree_xssd: str | None = None
    # MTP draft source of the tree (tree_mtp, "path:slots": a trained MCore MTP head, K / V slots for that many
    # concurrent rows; TP 1). The chain is the trie's first candidate on every tree row. None: own history only.
    tree_xssd_mtp: str | None = None
    # T1 tools (as on the xssd branch): stable_keys = noise keys from (seed, step, prompt idx, sample index), so two
    # runs draw the same rows; row_dump = directory for each step's rows (tokens + generation log-prob bits).
    stable_keys: bool = False
    # report_only: failed CHECKS and the train-lp sentinel are printed but do not stop the run (diagnostics only, e.g.
    # a multi-step T1 of a path with a known defect).
    report_only: bool = False
    row_dump: str | None = None

    @classmethod
    def from_config(cls, config: Any) -> "ColoConfig":
        block = _get(config, "thundersync") or {}
        block = dict(block) if not isinstance(block, dict) else block
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(block) - known)
        if unknown:
            raise ValueError(f"thundersync (mode=colo): unknown keys {unknown} (known: {sorted(known)})")
        out = cls(**{k: v for k, v in block.items() if k in known})
        if out.mode != MODE:
            raise ValueError(f"thundersync.mode must be {MODE!r} (got {out.mode!r})")
        return out


def enabled(config: Any) -> bool:
    return _get(config, "thundersync.mode") == MODE


def config_problems(config: Any) -> list[str]:
    from thundersync_rl import nolanes

    p: list[str] = []
    pol = _get(config, "policy") or {}
    gen = _get(pol, "generation") or {}
    mcfg = _get(pol, "megatron_cfg") or {}
    mcore = _get(gen, "mcore_generation_config") or {}
    if _get(gen, "backend") != "megatron":
        p.append("policy.generation.backend must be megatron")
    if not _get(gen, "colocated.enabled", False):
        p.append("policy.generation.colocated.enabled must be true (colo: one fleet generates and trains)")
    if not _get(mcfg, "batch_invariant_mode", False) or not _get(mcfg, "zero_train_gen_mismatch", False):
        p.append("colo needs batch-invariant numerics (policy.megatron_cfg.batch_invariant_mode and "
                 "zero_train_gen_mismatch): the batch is exact only if engine log-probs == learner log-probs")
    p += nolanes.kv_mode_problems(mcore)
    # Prefix caching: only on a Megatron-Core whose prefix cache is scoped by weight version (every wake that installs
    # theta_k bumps the epoch salted into the block hashes, so no block computed under theta_{k-1} is ever matched).
    if _get(mcore, "enable_prefix_caching", False) and not nolanes.mcore_prefix_cache_weight_scoped():
        p.append("prefix caching needs a Megatron-Core whose prefix cache is scoped by weight version (the installed "
                 "one is not: a cached block from theta_{k-1} would be read at theta_k)")
    if nolanes.negative() == "tree_stale_cache" and not (
            _get(config, "thundersync.tree_xssd") and _get(mcore, "enable_prefix_caching", False)):
        p.append("THUNDERSYNC_NL_NEG=tree_stale_cache needs tree_xssd and enable_prefix_caching (else it is vacuous)")
    xm = _get(config, "thundersync.tree_xssd_mtp")
    if xm:
        path, _, slots = str(xm).rpartition(":")
        if not _get(config, "thundersync.tree_xssd"):
            p.append("thundersync.tree_xssd_mtp needs thundersync.tree_xssd")
        if not path or not os.path.isfile(path):
            p.append(f"thundersync.tree_xssd_mtp: no head file {path!r} (format path:slots)")
        if not slots.isdigit() or not 1 <= int(slots) <= 1024:
            p.append("thundersync.tree_xssd_mtp slots must be in 1..1024 (format path:slots)")
        if int(_get(mcfg, "tensor_model_parallel_size", 1) or 1) != 1:
            p.append("thundersync.tree_xssd_mtp needs tensor_model_parallel_size 1 (full-vocab output layer)")
    if _get(pol, "offload_optimizer_for_refit", True):
        p.append("policy.offload_optimizer_for_refit must be false (resident colocated learner: MODEL.md v2 §11)")
    lag = _get(config, "async_rl.sampler.max_lookahead_versions", 0)
    if _get(config, "async_rl.sampler.name") != "in_order" or int(lag or 0) != 0:
        p.append("colo sync needs async_rl.sampler in_order with max_lookahead_versions 0 (every row admitted after "
                 "the wake that installs theta_k)")
    env = _get(mcfg, "env_vars") or {}
    for k in ENGINE_ONLY_ENV:
        if str(env.get(k, "0")) == "1":
            p.append(f"policy.megatron_cfg.env_vars.{k} is an engine-process lever (raises in a colocated learner)")
    if str(env.get(SKIP_RESHARD_ENV, os.environ.get(SKIP_RESHARD_ENV, "0"))) == "1" and not nolanes.checks_on():
        p.append(f"{SKIP_RESHARD_ENV} is a negative control: refused without CHECKS")
    p += nolanes.router_frozen_problems(config)
    return p


def validate_config(config: Any) -> ColoConfig:
    problems = config_problems(config)
    if problems:
        raise ValueError("thundersync.mode=colo: " + "; ".join(problems) + ".")
    return ColoConfig.from_config(config)


def configure_driver(config: dict) -> ColoConfig:
    """Driver side, before setup_single_controller: validate and select the ThunderSync worker class (keyed sampler,
    router fingerprints, law check)."""
    from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES

    from thundersync_rl import nolanes

    cfg = validate_config(config)
    nolanes.derive_layout_fields(config)
    ACTOR_ENVIRONMENT_REGISTRY[nolanes.WORKER_FQN] = PY_EXECUTABLES.MCORE
    config["policy"]["worker_extension_cls_fqn"] = nolanes.WORKER_FQN
    mcfg = config["policy"]["megatron_cfg"]
    # Plain top-k routing: with one weight version per step there is nothing to verify, so theta-only routing needs
    # no cross-fade band (MODEL.md §1, XS).
    mcfg["sticky_routing_delta"] = None
    mcfg["crossfade_routing_m"] = None
    # Always-on sentinel: the loss reports every trained token's training log-prob against its generation log-prob.
    config.setdefault("loss_fn", {})["train_lp_check"] = True
    if os.environ.get(SKIP_RESHARD_ENV) == "1":
        mcfg["env_vars"] = dict(mcfg.get("env_vars") or {}, **{SKIP_RESHARD_ENV: "1"})
    if cfg.tree_xssd:
        w, d, rows = (int(x) for x in str(cfg.tree_xssd).split(":"))
        # the engine-side node rows (Mamba scratch rows, KV gather) are sized from this before the context exists;
        # ref tokens 1: own-history tails carry no references
        mcfg["env_vars"] = dict(mcfg.get("env_vars") or {}, THUNDERSYNC_TAIL_TREE=f"{w}:{d}:{rows}:1")
        os.environ["THUNDERSYNC_TAIL_TREE"] = f"{w}:{d}:{rows}:1"
        if cfg.tree_xssd_mtp:
            # the decoder-output capture is installed on the engine model before the engine is built (megatron_worker)
            mcfg["env_vars"]["THUNDERSYNC_TAIL_TREE_MTP"] = str(cfg.tree_xssd_mtp)
            os.environ["THUNDERSYNC_TAIL_TREE_MTP"] = str(cfg.tree_xssd_mtp)
    neg = nolanes.negative()
    print(f"[colo] mode=colo sync (lag 0) engine sampler=wide key64={int(cfg.wide_sampler_key64)} "
          f"checks={int(nolanes.checks_on())}" + (f" NEGATIVE CONTROL {neg}" if neg else "")
          + (" NEGATIVE CONTROL skip_reshard" if os.environ.get(SKIP_RESHARD_ENV) == "1" else ""), flush=True)
    return cfg
