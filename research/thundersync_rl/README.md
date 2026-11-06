# ThunderSync: exact on-policy speculative GRPO

ThunderSync runs GRPO with rollouts that are **exactly on-policy** at every optimizer step, and keeps the step
time close to one-step-stale asynchronous GRPO. While the learner trains step k, the inference engine drafts
step k+1's rollouts with the current weights. Once theta_{k+1} exists, block verification turns each draft into
an exact sample of pi_{theta_{k+1}}: an accepted prefix plus a corrected continuation. The rollouts the learner
trains on are therefore distributed exactly as fresh synchronous rollouts would be. There is no importance
weighting and no staleness.

The guarantee rests on three pieces:

1. **Bitwise train == engine numerics.** A batch-invariant fp16-GEMM recipe (`nemo_rl/models/megatron/`) makes
   the Megatron inference engine and the Megatron learner compute bitwise-identical log-probs for the same
   weights and tokens. This holds at any batch composition and under TP / EP / PP / CP layouts. Any mismatch is
   a bug, and the in-loop checks report it.
2. **Exact sampling.** 53-bit keyed engine sampling, exact block verification (eager and fused), and pooled
   null-augmented block verification (⊥-BV) with a fixed group size.
3. **Sticky MoE routing.** A draft's expert routes are replayed (anchored) where they remain admissible, so
   continuations stay verifiable on large MoEs.

Gradients only need to be accurate. Neither run-to-run nor cross-layout bitwise backward is required.

## Layout

| area | where |
|---|---|
| Batch-invariant kernels and recipe (runtime patches of Megatron / TE / FA4; each names the upstream PR that retires it) | `nemo_rl/models/megatron/` (`batch_invariant`, `bi_*`, `fp16_*`, `fp32_residual`, `deferred_head`, `fused_*_ops`, `mamba_*`, `ep_invariant_combine`, `exact_nucleus`, `exact_int`) |
| Scale-up | `latent_tp_shard`, `bi_context_parallel`, `inference_pp_cuda_graph`, `prefix_state_cache_cp` |
| Learner | `device_dispatch` (host-sync-free MoE dispatch), `sticky_routing*`, `router_replay`, `learner_graphs` / `graph_capture` (opt-in), `prefix_state_cache`, `precision_raise` (opt-in) |
| Recipe wiring | `zero_train_gen_mismatch.py`, `setup.py`, `train.py`, `nemo_rl/models/policy/workers/megatron_policy_worker.py`, `nemo_rl/models/generation/megatron/` |
| Gradient streaming | `nemo_rl/algorithms/grad_streaming.py` (accumulates micro-batches that arrive in any order, exactly) |
| ThunderSync loop | `research/thundersync_rl/thundersync_rl/`: `grpo_loop.py` (config, validators, loop), `speculative.py` (draft / verify / continue driver), `worker.py` (learner worker), `streaming.py` |
| Exact verification core | `block_verification*.py`, `null_bv.py`, `keyed_sampling.py`, `sampling64.py`, `exact_bv.py`, `engine_q.py`, `routes.py`, `draft_weights.py`, `draft_table.py` |
| Checks and long runs | `law_check.py`, `checks_sched.py`, `health.py`, `run_state.py` (checkpoint / resume), `sentinel.py` (weight digests), `leak_probe.py` |
| Opt-in levers | `lanes.py`, `la2.py`, `aep.py`, `refit_m2n.py`, `verify_yield.py`, `verify0_stream.py`, `vit.py`, `fused_train.py`, `window_pace.py`, `prefix_cache.py`, `carried_walk.py`, `multi_turn.py`, `engine_verify.py`, `logit_free.py`, `scorer_kernels.py` |

## Configuration

`configs/thundersync.yaml` is the canonical configuration. It states every `thundersync.*` key, plus the
batch-invariant recipe, engine and learner keys, all with their production values. Keys whose code default
differs from production are marked. The file is generated from the production arm, so do not hand-edit it.

The production arm is `configs/ab/ab_ts_scale.yaml`. It enables:

- sticky routing (delta 0.03);
- pooled ⊥-BV (2.0 drafts per output, calibrated, `fit` admission);
- two cohort lanes with lookahead 2;
- engine-side q with verification on the learner;
- an eager learner with device MoE dispatch, and engine CUDA graphs on.

Baselines on the same learner path are `ab_sync.yaml` (synchronous) and `ab_async1.yaml` (one step stale,
IS-corrected). Startup validators reject unsupported combinations and unknown keys.

Per-run settings that depend on the sequence caps:

- `train_micro_batch_tokens` and the device-dispatch token cap scale as MBS x (prompt cap + generation cap).
- `verify0_batch_tokens` equals the dispatch cap.
- `draft_budget` equals `max_new_tokens`.

## Running

```bash
python research/thundersync_rl/run_thundersync_grpo.py --config research/thundersync_rl/configs/thundersync.yaml \
    policy.model_name=<model> <layout overrides: TP / PP / EP, cluster, generation resources>
```

The engine and the learner run on separate GPUs (`policy.generation.colocated.enabled=false`).

## Exactness checks

The checks are switched on through `policy.megatron_cfg.env_vars` and run in the loop:

| variable | check |
|---|---|
| `THUNDERSYNC_MISMATCH_CHECK` | One learner forward on the rollout must reproduce the engine's stored log-probs bitwise. `mismatch/max_abs_logprob_diff` must be 0. |
| `THUNDERSYNC_TRAIN_LP_CHECK` | The training forward's log-probs must equal the rollout's. |
| `THUNDERSYNC_ENGINE_PREFILL_CHECK` | The engine's prefill log-probs must equal its decode log-probs. |
| `THUNDERSYNC_LAW_CHECK` | Randomized PIT law test of the emitted tokens, plus the ⊥-BV emission-law test (`THUNDERSYNC_EMISSION_LAW_EVERY`). |
| `THUNDERSYNC_CHECK_EVERY=k` | Runs the full set every k-th step and the cheap subset on the other steps (long runs). |

A run is exact when every checked step reports 0 mismatch and the law tests pass. Negative controls, such as a
deliberately wrong temperature, must fail.

## Tests

```bash
# CPU (no GPU needed)
pytest research/thundersync_rl/tests/unit tests/unit/models/megatron tests/unit/algorithms/test_grad_streaming.py
# GPU functional checks (streamed == synchronous update, prefix cache, parallel parity)
bash research/thundersync_rl/tests/functional/equivalence.sh
```

## Upstreaming

Megatron changes live here as runtime patches until their upstream PR is merged and the Megatron-LM pin
contains it. The pin bump that picks up a PR deletes the matching patch in the same commit.

- Merged upstream: NVIDIA/Megatron-LM#8242 (batch-invariant mean) and #8214 (batch-invariant Mamba prefix
  caching).
- Open: #8221 (z-free batch-invariant Mamba decode), #8261 (EP-invariant combine), #8366 (hybrid pipeline
  stages without attention) and #8347 (MoE recompute combine; learner speed only, not carried here).
