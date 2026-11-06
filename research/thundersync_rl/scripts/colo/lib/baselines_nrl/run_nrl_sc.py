"""Driver for the nrl_k arms: NeMo-RL's stock SingleController async GRPO entrypoint, unchanged.

Runs ``$RL_ROOT/examples/run_grpo_single_controller.py`` with the given argv. Two driver-side additions, neither of
which touches the training loop, the sampler, refit or numerics:
  - NRL_PARITY_REWARD=1 registers env ``math_parity`` (nrl_parity_env.py; the truncated-model reward of the ts runs);
  - NRL_EFFECTIVE_CONFIG=<path> writes the fully resolved config (after the CLI overrides, before MasterConfig) as
    YAML, for the diff against the stock recipe (diff_vs_recipe.py).
Per-step timing / lag come from the loop's own prints (parse_nrl.py reads them from run.log).
"""

import os
import runpy
import sys


def _require_frozen_router(c) -> None:
    """Hard rule (user, 11-02): router gate weights and router expert bias frozen in every RL run. Refuse otherwise."""
    import frozen_router

    ok, why = frozen_router.config_frozen(c)
    if not ok:
        print(f"[nrl] REFUSED: router not frozen: {why}", flush=True)
        sys.exit(4)
    print(f"[nrl] {why}", flush=True)


def main() -> None:
    root = os.environ["RL_ROOT"]
    script = os.path.join(root, "examples", "run_grpo_single_controller.py")
    if os.environ.get("NRL_PARITY_REWARD", "0") == "1":
        import nrl_parity_env

        nrl_parity_env.register()
        print("[nrl] env math_parity registered (crc32 parity reward of the assistant text)", flush=True)
    out = os.environ.get("NRL_EFFECTIVE_CONFIG")
    if out:
        import nemo_rl.utils.config as cfg_mod
        from omegaconf import OmegaConf

        inner = cfg_mod.parse_hydra_overrides

        def parse_and_dump(config, overrides):
            merged = inner(config, overrides)
            with open(out, "w") as f:
                f.write(OmegaConf.to_yaml(merged, resolve=True))
            print(f"[nrl] effective config -> {out}", flush=True)
            _require_frozen_router(OmegaConf.to_container(merged, resolve=True))
            return merged

        cfg_mod.parse_hydra_overrides = parse_and_dump
    if os.environ.get("NRL_DRY", "0") == "1":
        # Preflight: the entrypoint's own config path (load, CLI overrides, resolve, outdated-config check,
        # MasterConfig validation, the legacy-async / data-plane guards) plus the sampler / buffer capacity check,
        # then exit before Ray starts.
        import argparse

        from omegaconf import OmegaConf

        from nemo_rl.algorithms.async_utils.staleness_sampler import required_buffer_capacity_for_config
        from nemo_rl.algorithms.single_controller_utils import MasterConfig
        from nemo_rl.utils.config import load_config, parse_hydra_overrides, register_omegaconf_resolvers
        from nemo_rl.utils.outdated_config_checks import check_outdated_config

        register_omegaconf_resolvers()
        ap = argparse.ArgumentParser()
        ap.add_argument("--config")
        a, ov = ap.parse_known_args(sys.argv[1:])
        c = cfg_mod.parse_hydra_overrides(load_config(a.config), ov) if out else parse_hydra_overrides(load_config(a.config), ov)
        c = OmegaConf.to_container(c, resolve=True)
        check_outdated_config(c)
        _require_frozen_router(c)
        mc = MasterConfig(**c)
        assert mc.grpo.async_grpo is None, "grpo.async_grpo must be null on the SC path"
        assert mc.data_plane["enabled"] if isinstance(mc.data_plane, dict) else mc.data_plane.enabled
        s = mc.async_rl.sampler
        need = required_buffer_capacity_for_config(
            s, mc.grpo.num_prompts_per_step, min_groups_for_streaming_train=mc.async_rl.min_groups_for_streaming_train)
        assert need is None or mc.async_rl.max_buffered_rollouts >= need
        kvm = c["policy"]["generation"]["mcore_generation_config"]["kv_cache_management_mode"]
        # megatron_generation.py refuses a mismatch at engine setup; catch it here.
        assert mc.async_rl.recompute_kv_cache_after_weight_updates == (kvm == "recompute"), (
            "async_rl.recompute_kv_cache_after_weight_updates must equal (kv_cache_management_mode == recompute)")
        print(f"[nrl] DRY KV mode={kvm} recompute_after_refit={mc.async_rl.recompute_kv_cache_after_weight_updates} "
              f"force_on_policy_ratio={c['loss_fn'].get('force_on_policy_ratio')} "
              f"seq_logprob_error_threshold={c['grpo'].get('seq_logprob_error_threshold')} "
              f"IS={c['loss_fn'].get('use_importance_sampling_correction')}", flush=True)
        print(f"[nrl] DRY OK sampler={s.name} lookahead={getattr(s, 'max_lookahead_versions', None)} "
              f"buffer={mc.async_rl.max_buffered_rollouts} need={need} inflight={mc.async_rl.max_inflight_prompts} "
              f"min_groups={mc.async_rl.min_groups_for_streaming_train} env={c['data']['default']['env_name']} "
              f"gbs={c['policy']['train_global_batch_size']} P*G={c['grpo']['num_prompts_per_step'] * c['grpo']['num_generations_per_prompt']}",
              flush=True)
        assert need is None or mc.async_rl.max_buffered_rollouts >= need
        return
    sys.argv = [script, *sys.argv[1:]]
    # run_grpo_single_controller imports parse_hydra_overrides by name at module load: patch the module attribute
    # before runpy executes it (runpy re-imports nothing; the `from ... import` binds the patched function).
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
