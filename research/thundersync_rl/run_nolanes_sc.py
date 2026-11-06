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
"""Driver for the nolanes (NL2) arms: NeMo-RL's SingleController async GRPO entrypoint, run as is.

Runs ``$RL_ROOT/examples/run_grpo_single_controller.py`` with the given argv (config: configs/nl/nl_base.yaml + CLI;
the entrypoint's own hook calls thundersync_rl.nolanes.configure_driver when thundersync.mode == nolanes).
Driver-side additions only (same shape as the research notes, summarized in docs/colo/):
  - NL_PARITY_REWARD=1 registers env ``math_parity`` (thundersync_rl/nl_parity_env.py; the truncated-model reward);
  - NL_EFFECTIVE_CONFIG=<path> writes the fully resolved config (after the CLI overrides, before MasterConfig) as YAML;
  - NL_DRY=1 preflight, no Ray / GPUs: config compose + CLI overrides + resolve, outdated-config check, MasterConfig,
    the SC guards (async_grpo null, data plane on), sampler buffer capacity, KV-mode / recompute agreement, the BI
    (zero-KL + batch-invariant) and fp16-GEMM config validators, thundersync_rl.nolanes.validate_config, and a print of
    the thundersync block. Prints "[nl] DRY OK ..." on success, exits non-zero on any failure.
"""

import os
import runpy
import sys

TAG = "[nl]"


def _register_parity() -> None:
    from thundersync_rl import nl_parity_env

    nl_parity_env.register()
    print(f"{TAG} env math_parity registered (crc32 parity reward of the assistant text)", flush=True)


def _patch_effective_config_dump(out: str) -> None:
    import nemo_rl.utils.config as cfg_mod
    from omegaconf import OmegaConf

    inner = cfg_mod.parse_hydra_overrides

    def parse_and_dump(config, overrides):
        merged = inner(config, overrides)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            f.write(OmegaConf.to_yaml(merged, resolve=True))
        print(f"{TAG} effective config -> {out}", flush=True)
        return merged

    cfg_mod.parse_hydra_overrides = parse_and_dump


def _dry(argv: list[str]) -> None:
    import argparse

    import nemo_rl.utils.config as cfg_mod
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.async_utils.staleness_sampler import required_buffer_capacity_for_config
    from nemo_rl.algorithms.single_controller_utils import MasterConfig
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
    from nemo_rl.utils.outdated_config_checks import check_outdated_config

    register_omegaconf_resolvers()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    a, ov = ap.parse_known_args(argv)
    # cfg_mod.parse_hydra_overrides is the dumping wrapper when NL_EFFECTIVE_CONFIG is set.
    c = cfg_mod.parse_hydra_overrides(load_config(a.config), ov) if ov else load_config(a.config)
    c = OmegaConf.to_container(c, resolve=True)
    check_outdated_config(c)
    mc = MasterConfig(**c)
    assert mc.grpo.async_grpo is None, "grpo.async_grpo must be null on the SC path"
    assert mc.data_plane["enabled"] if isinstance(mc.data_plane, dict) else mc.data_plane.enabled, (
        "data_plane.enabled must be true on the SC path")
    s = mc.async_rl.sampler
    need = required_buffer_capacity_for_config(
        s, mc.grpo.num_prompts_per_step, min_groups_for_streaming_train=mc.async_rl.min_groups_for_streaming_train)
    assert need is None or mc.async_rl.max_buffered_rollouts >= need, (
        f"async_rl.max_buffered_rollouts={mc.async_rl.max_buffered_rollouts} < sampler capacity {need}")
    pol = c["policy"]
    gen = pol["generation"]
    kvm = gen["mcore_generation_config"]["kv_cache_management_mode"]
    # megatron_generation.py refuses a mismatch at engine setup; catch it here.
    assert mc.async_rl.recompute_kv_cache_after_weight_updates == (kvm == "recompute"), (
        "async_rl.recompute_kv_cache_after_weight_updates must equal (kv_cache_management_mode == recompute)")
    spares = int((c.get("thundersync") or {}).get("spare_prompts", 0) or 0)
    assert pol["train_global_batch_size"] == (c["grpo"]["num_prompts_per_step"] - spares) * c["grpo"]["num_generations_per_prompt"], (
        "SC needs one optimizer step per RL step: train_global_batch_size == (prompts - spare prompts) x G")

    # Batch-invariant numerics (the validators the learner / engine workers run at startup; config-only here).
    try:
        from nemo_rl.models.generation.megatron.config import merged_inference_megatron_cfg
        from nemo_rl.models.megatron import fp16_gemm
        from nemo_rl.models.megatron.zero_train_gen_mismatch import validate_zero_train_gen_kl

        from thundersync_rl.nolanes import derive_layout_fields

        derive_layout_fields(c)  # as configure_driver does before the workers validate (Ultra: EP 32 vs 16)
        validate_zero_train_gen_kl(pol, check_environment=False)
        errs = fp16_gemm.validate_config(pol["megatron_cfg"], merged_inference_megatron_cfg(pol))
        assert not errs, "forward_gemm_precision: " + "; ".join(errs)
        print(f"{TAG} DRY BI validators OK (zero_train_gen_mismatch, batch_invariant_mode, fp16_gemm)", flush=True)
    except ImportError as e:
        print(f"{TAG} DRY note: BI validators not importable here ({e}); skipped", flush=True)

    m = pol["megatron_cfg"]
    mg = gen["mcore_generation_config"]
    print(f"{TAG} DRY numerics forward_gemm_precision={m.get('forward_gemm_precision')} "
          f"fp32_residual_connection={m.get('fp32_residual_connection')} defer_fp32_logits={m.get('defer_fp32_logits')} "
          f"moe_router_dtype={m.get('moe_router_dtype')} batch_invariant_mode={m.get('batch_invariant_mode')} "
          f"backend={m.get('batch_invariant_backend')} collective={m.get('batch_invariant_collective')} "
          f"attn={m.get('attention_backend')}/FA{m.get('flash_attention_version')} "
          f"TP={m.get('tensor_model_parallel_size')} EP={m.get('expert_model_parallel_size')} "
          f"genTP={mg.get('tensor_model_parallel_size')} genEP={mg.get('expert_model_parallel_size')} "
          f"packing={pol['sequence_packing']['enabled']} seqdiv={pol['make_sequence_length_divisible_by']}", flush=True)
    print(f"{TAG} DRY engine KV mode={kvm} recompute_after_refit={mc.async_rl.recompute_kv_cache_after_weight_updates} "
          f"logprobs_mode={mg.get('logprobs_mode')} chunked_prefill={mg.get('enable_chunked_prefill')} "
          f"prefix_caching={mg.get('enable_prefix_caching')} graphs={mg.get('num_cuda_graphs')} "
          f"all_prefills={mg.get('cuda_graph_all_prefills')} max_requests={mg.get('max_requests')} "
          f"buffer_gb={mg.get('buffer_size_gb')} max_tokens={mg.get('max_tokens')}", flush=True)
    print(f"{TAG} DRY loss force_on_policy_ratio={c['loss_fn'].get('force_on_policy_ratio')} "
          f"seq_logprob_error_threshold={c['grpo'].get('seq_logprob_error_threshold')} "
          f"IS={c['loss_fn'].get('use_importance_sampling_correction')}", flush=True)
    print(f"{TAG} DRY thundersync {c.get('thundersync')}", flush=True)
    from thundersync_rl import nolanes as _nl_mod

    print(f"{TAG} DRY {_nl_mod.router_frozen_line(c)} (freeze_moe_router={m.get('freeze_moe_router')} "
          f"moe_router_bias_update_rate={m.get('moe_router_bias_update_rate')})", flush=True)

    try:
        from thundersync_rl import nolanes
    except ImportError as e:
        print(f"{TAG} DRY note: thundersync_rl.nolanes not importable ({e}); nolanes validator skipped", flush=True)
    else:
        if (c.get("thundersync") or {}).get("mode") == "colo":
            from thundersync_rl import colo

            ccfg = colo.validate_config(c)
            print(f"{TAG} DRY colo.validate_config OK: {ccfg}", flush=True)
        elif not hasattr(nolanes, "validate_config"):
            print(f"{TAG} DRY note: thundersync_rl.nolanes has no validate_config yet; skipped", flush=True)
        else:
            ncfg = nolanes.validate_config(c)
            print(f"{TAG} DRY nolanes.validate_config OK: {ncfg}", flush=True)

    print(f"{TAG} DRY OK sampler={s.name} lookahead={getattr(s, 'max_lookahead_versions', None)} "
          f"buffer={mc.async_rl.max_buffered_rollouts} need={need} inflight={mc.async_rl.max_inflight_prompts} "
          f"min_groups={mc.async_rl.min_groups_for_streaming_train} env={c['data']['default']['env_name']} "
          f"gbs={pol['train_global_batch_size']} P*G={c['grpo']['num_prompts_per_step'] * c['grpo']['num_generations_per_prompt']} "
          f"mode={(c.get('thundersync') or {}).get('mode')}", flush=True)


def main() -> None:
    root = os.environ["RL_ROOT"]
    script = os.path.join(root, "examples", "run_grpo_single_controller.py")
    if os.environ.get("NL_PARITY_REWARD", "0") == "1":
        _register_parity()
    out = os.environ.get("NL_EFFECTIVE_CONFIG")
    if out:
        _patch_effective_config_dump(out)
    if os.environ.get("NL_DRY", "0") == "1":
        _dry(sys.argv[1:])
        return
    sys.argv = [script, *sys.argv[1:]]
    # run_grpo_single_controller imports parse_hydra_overrides by name when runpy executes it, so the patched module
    # attribute (effective-config dump) is the one it binds.
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
