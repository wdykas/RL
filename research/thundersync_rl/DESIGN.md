# ThunderSyncRL with cross-iteration speculation: design

Zero-staleness GRPO that runs faster than an asynchronous pipeline. Every gradient is
taken at the exact weights that generated its data, yet the generation engine never
waits for the optimizer step. NeMo-RL, Megatron training + non-colocated Megatron
inference, branch `thundersync-rl`.

| | seconds per iteration |
|---|---|
| Zero staleness **with speculation** (100 steps) | **3.02** (median 2.70, p90 4.28) |
| Zero staleness without speculation (100 steps) | 6.13 (median 5.86) |
| Async GRPO, lag 4, same settings | 4.26 (median 2.40, p90 7.75) |

Qwen2.5-Math-1.5B-Instruct, 1 GB200 node: 2 Megatron inference GPUs + 2 Megatron
training GPUs, 16 prompts x 8 samples, up to 4096 tokens. 99.5% of drafted tokens are
kept after exact verification. Reward matches the no-speculation baseline within
run-to-run noise.

## 1. The problem

Synchronous on-policy GRPO alternates two phases. The engine samples a batch at weights
θ_k; the learner computes the update; the engine gets θ_{k+1} and starts again. Each
phase leaves the other's GPUs idle, and the rollout phase ends with a long tail where a
few long trajectories decode alone on an almost empty engine.

Asynchronous RL removes the idle time by letting the engine run ahead on older weights.
The price is staleness: the learner trains on trajectories sampled up to L versions ago
and needs importance corrections or clipping to stay stable.

This work removes both kinds of idle time without giving up exactness, in two layers:

- **Gradient streaming** (the ThunderSyncRL paper) moves the learner's work into the
  rollout tail: each trajectory is backpropagated as soon as it finishes, at θ_k.
- **Cross-iteration speculation** (new here) moves the engine's work for iteration k+1
  into iteration k: it drafts the next batch under the current weights and verifies those
  drafts exactly once θ_{k+1} exists.

## 2. Layer 1: gradient streaming

A GRPO advantage depends on the whole group, so a trajectory's gradient seemingly cannot
be used before its group finishes. But the update is linear in the per-trajectory score
gradients H_i = ∇ log π_θk(y_i) (masked, summed over tokens):

```
Δ = Σ_i a_i · H_i
B[g, r] = Σ_{i ∈ group g, r_i = r} H_i        accumulated while group g is open
Δ = Σ_g Σ_r a_g(r) · B[g, r]                  folded in when g closes
```

Every group-wise estimator gives trajectories with equal rewards in the same group the
same advantage, so per-(group, reward) buckets are exact for any estimator, including
NeMo-RL's default leave-one-out baseline with leave-one-out std (not affine in the
reward). With binary rewards this is two buffers per open group. A group with more than
`max_buckets_per_group` distinct rewards collapses to the paper's affine
(Σ r_i H_i, Σ H_i) form, and the accumulator raises an error if the estimator is not
affine.

Normalization, the data-parallel reduction, clipping, the single optimizer step and the
refit happen once per batch. On Qwen2.5-0.5B in fp32 the streamed gradient differs from
`Policy.train` by a relative 4.6–7.4e-6, the same order as re-batching the synchronous
update.

Scheduling:

- **Final-first.** Rows of closed groups (final advantages known) always go first. Idle
  learner time streams one open group's rows into buckets.
- **Length-sorted chunks.** Final rows from different groups share micro-batches, sorted
  by length to limit padding, balanced over data-parallel ranks by tokens.
- **Zero-advantage chunks.** With `loss_fn.skip_zero_advantage_rows`, zero-advantage rows
  are chunked separately; the worker counts their tokens for normalization and drops the
  whole chunk without a forward pass. Learner time 2.04 → 1.55 s per step.
- **Expert parallelism.** With `expert_model_parallel_size > 1` every forward/backward
  runs an all-to-all over the EP group, which spans data-parallel ranks, so all ranks must
  run the same number of micro-batches per dispatch. The planner then pads short ranks
  with one-row padding chunks (fully masked: no gradient, no normalization count), and the
  worker ignores `skip_zero_advantage_rows` (zero-advantage rows run a forward).

## 3. Layer 2: cross-iteration speculation

Zero staleness forces a drain at every update: iteration k+1 cannot sample until θ_{k+1}
exists. While iteration k runs, the engine drafts the rollouts of the next iterations
under the weights it has. After the step, each draft is verified against θ_{k+1} with a
rule whose output is distributed exactly as π_θ{k+1}. Only rejected or unfinished
remainders are decoded fresh.

**Why it can work.** One optimizer step moves the policy very little. The per-token
rejection rate between π_k and π_{k+1} measured in bf16 is about 1.7e-3, and it is the
same for drafts made one to four versions earlier: bf16 activation rounding dominates, not
policy change. In fp32 the true one-step drift is about 3e-4 per token. Most drafted
tokens are acceptable, provided verification is precise enough to see the real drift
through the numerical noise.

Lifecycle of one iteration:

1. **Draft.** From the start of iteration k, the Megatron inference engine streams drafts
   for the prompts of iterations k+1 … k+`draft_lookahead` (`start_drafts`). Draft
   requests share the engine with iteration k's own rollouts and fill the capacity its
   tail leaves idle. New cohorts ramp in at most two per step (`new_cohort_targets`).
2. **Cut at the deadline.** Just before the optimizer step, unfinished drafts are aborted
   and snapshotted (`collect_drafts`). Any prefix of a draft is a valid draft; a cut draft
   resumes from its prefix in the next iteration, under the newer weights.
3. **Record provenance.** Each draft carries segments `(version, start, end)` saying
   which weights drafted which positions: verification needs each token's true proposal
   distribution.
4. **Stash weights.** Still at θ_k, every verifier keeps a bf16 copy of the weights
   (`stash_weights`), one per version that drafted pending tokens.
5. **Verify.** At iteration k+1, verifiers recompute q from the stashes and p from
   θ_{k+1}, block-verify each draft, and publish a plan per trajectory: the kept prefix
   plus one bonus or residual token.
6. **Continue.** The rollout code is unchanged. `SpeculativeGeneration` wraps the
   generation interface: a fully accepted, finished draft returns immediately; anything
   else is continued by the engine from its verified prefix.

## 4. Block verification

Token-by-token speculative sampling rejects at the first position where the noise in p/q
happens to fall low. Block verification (Sun et al. 2024) looks at the whole draft, so
positive and negative noise in the ratio partly cancel along the block. For a draft
x_1..x_g from q, target p, and per-position distributions p_i, q_i:

```
b_0 = 1,   b_i = min(1, b_{i-1} · p(x_i)/q(x_i))
         → log b_i = S_i − max_{j≤i} S_j,   S_i = Σ_{j≤i} log(p/q)(x_j)
r_i = Σ_v max(b_i · p_i(v) − q_i(v), 0)               (i < g)
h_i = r_i / (r_i + 1 − b_i)   (i < g),     h_g = b_g
τ   = max{ i : η_i ≤ h_i },   η_i ~ U(0,1) independent
keep x_1..x_τ, then emit  y ~ p_g                          if τ = g  (bonus token)
                          y ~ norm(max(b_τ p_τ − q_τ, 0))  otherwise (residual)
```

The output (x_1..x_τ, y) is distributed exactly as an autoregressive sample from p.
`tests/unit/test_block_verification.py` checks this by exhaustive enumeration (total
variation < 1e-10).

| Verification rule | Draft tokens kept |
|---|---|
| Keyed Gumbel, token level (exactly on-policy alternative) | 51–60% |
| Block verification, bf16 scorer | 87–91% |
| Block verification, fp32 LM head | 91% |
| **Block verification, fp32 Megatron scorer with TF32 GEMMs** | **98–99.5%** |

**Fused implementation.** `block_verify_logits` works from raw logits: one `logsumexp`
per row of p and q, the draft-token ratios and b on length-g vectors, and one Triton pass
per position for the residual masses r_i. Only the residual row actually sampled from is
materialized. It shares `acceptance_weights`, `stop_probabilities` and
`emit_distribution` with the reference and rounds the same way (fp32 log-probs,
exponentiated in fp64); it is draw-for-draw identical to the reference in tests.

**Keyed sampling (alternative mode).** `verify_mode=keyed` makes every token a
Gumbel-max sample whose noise is a hash of (seed, absolute position, token id), computed
in a Triton kernel inside the engine's sampler. Two policies emit the same token wherever
their Gumbel races agree, so a draft is checked position by position with no q at all.
Exactly on-policy, but only 51–60% of tokens are kept, so block verification is the
default.

## 5. The verification scorer

Acceptance depends on log p − log q for weights one to three versions apart, which is
smaller than bf16 rounding noise. The verifier scores with a second Megatron `GPTModel`
built from a shallow copy of the training provider with fp32 parameters and activations,
refreshed by copying the training weights (about 7 ms at 1.5B). With
`NVIDIA_TF32_OVERRIDE=1` its GEMMs run on TF32 tensor cores. No Hugging Face model is
used anywhere.

**Making the fp32 forward fast** (`scorer_kernels.py`, `worker.py`). About 85% of
verification time is scorer forward passes, and Megatron/TE fall back to slow paths in
fp32:

- **Attention through torch SDPA.** Replaces plain causal softmax attention only (no
  sliding window, no attention sinks), using each module's own softmax scale. About 2x
  faster per layer than the unfused fp32 path and about 1000x closer to exact (max error
  1.7e-6 vs 2e-3 with TF32 score GEMMs).
- **Triton SwiGLU** for dense `MLP` modules only (MoE experts keep their own forward):
  one pass over the fc1 output instead of strided `silu`, `+ offset` and `mul`; 3.8x
  faster, relative error 4e-7.
- **Selective scoring** (`select_positions`). Each version's q forward stops at the last
  position that version drafted (causal attention makes the truncation exact), and the
  LM head runs only on positions verification needs; the p forward skips prompt
  positions. Logits are bitwise identical to the full forward
  (`THUNDERSYNC_SELECT_CHECK=1`).

Attention needs fp32-level accuracy: running only attention in bf16 inside the fp32 scorer
drops tokens kept from 99.2% to 96.4% and fully accepted drafts from 98.5% to 94.7%,
slowing iterations from 3.31 to 4.12 s. A Triton flash-attention kernel with `tf32x3`
dots matched fp32 accuracy but was 2x slower than SDPA.

**q storage scales with parameters, not tokens.** Storing each draft's full-vocabulary q
costs O(draft tokens x vocabulary), about 150k floats per token. Instead the verifiers keep
one bf16 weight copy per pending version (2 bytes per parameter per model-parallel shard)
and recompute q per batch: memory O(parameters + one batch x vocabulary).

## 6. Overlapped verification pipeline

At the start of iteration k+1 every GPU is briefly free. Verification uses all of them and
overlaps with the rollouts:

- **Both pools verify** (`verify_on=both`). Rows are split into `verify_chunks` chunks
  assigned round-robin to the learner and inference worker groups. Rows are owned by
  data-parallel rank through a stable hash of the draft key; sampling is seeded per
  data-parallel rank, so tensor-parallel partners make identical decisions.
- **Plans publish per chunk** (`verify_overlap`). Rollouts wait on a condition variable
  for their own plan, not the whole batch. Verification errors wake every waiting rollout
  and surface instead of hanging it.
- **Longest drafts first** (`verify_longest_first`), so the continuations most likely to
  become stragglers start earliest.
- **Training starts as soon as groups complete**, while later chunks still verify.

Typical iteration where every draft is accepted (about 2.3 s, learner-bound):

```
time (s)        0.0       0.5       1.0       1.5       2.0   2.3
learner GPUs    [verify ][====== train complete groups ======][step/refit]
inference GPUs  [verify  ][/////// draft iterations k+2..k+4 ////////////]
rollouts           [remainders]
```

When one draft is rejected early, its remainder decodes serially and extends the
rollout lane by 1–3 s.

## 7. Exactness

**Exact.** Gradient streaming produces the synchronous update up to fp32 summation order.
Block verification returns exact samples of p given each draft's true proposal
distribution q. Every gradient is taken at the weights that define the sampling target:
no staleness, no importance weights.

**Approximation.** Drafts are sampled by the bf16 inference engine, but q is recomputed
by the fp32 scorer. The two differ by numerical noise at equal weights, the same kind of
train/inference mismatch every RL pipeline has (per-token probability ratio about 1.0035
at equal weights). Batch-invariant kernels (NeMo-RL PR 3208) would make q bitwise the
sampling distribution; keyed mode avoids the issue at lower acceptance.

**Reward check over 100 steps.** Speculation vs a second no-speculation baseline: reward
difference −0.0016 ± 0.0037 (paired by step). The two baselines differ from each other by
−0.0080 ± 0.0036, so run-to-run noise is larger than any effect of speculation.

## 8. Results

Qwen2.5-Math-1.5B-Instruct, 2 inference + 2 training GPUs, 128 rollouts per step, group
streaming. Mean seconds per iteration after warm-up; 30-step runs unless marked.

| Configuration | Mean | Median | p90 | Tokens kept | Drafts whole |
|---|---|---|---|---|---|
| Zero staleness, no speculation (100 steps) | 6.13 | 5.86 | 7.88 | – | – |
| Speculation, lookahead 3, overlapped verification on both pools (40 steps) | 3.97 | 3.84 | 4.59 | 99.1% | 98.6% |
| + fused scorer kernels (SDPA, Triton SwiGLU) | 3.55 | 3.35 | 4.42 | 99.2% | 98.8% |
| + selective scoring | 3.31 | 3.07 | 4.02 | 99.2% | 98.5% |
| + zero-advantage rows in their own chunks | 3.03 | 2.79 | 4.23 | 99.4% | 98.5% |
| **Same, 100 steps** | **3.02** | **2.70** | **4.28** | **99.5%** | **98.4%** |
| Async GRPO, lag 4, same settings (single controller) | 4.26 | 2.40 | 7.75 | – | – |

Async has a lower median but a much worse tail: with zero-advantage skip its learner is
fast, so it becomes generation-bound and stalls 2–13 s on some steps.

Tried and not adopted:

| Change | Result |
|---|---|
| bf16 attention in the fp32 scorer | 4.12 s; tokens kept 96.4% |
| Triton flash attention with tf32x3 dots | 2x slower than fp32 SDPA |
| Verify only on inference GPUs | 3.78 s; competes with drafting |
| Small first verification chunk; group-aligned chunks; 1:2 and 1:3 pool weights | 3.2–3.9 s; neutral or worse |
| Delay drafting until few rollouts remain in flight (0 or 2) | 6.30 s, 3.74 s; unfinished drafts |
| Pause drafts while continuations decode (always, or only in the tail) | 5.73 s, 4.14 s; drafting time is worth more |
| Micro-batch 16, token-budgeted chunks, fused linear-logprob loss | No change in learner time |
| 3 inference + 1 training GPUs | 3.32 s per 120 rollouts (no speculation 5.53 s): single learner GPU becomes the bottleneck |

## 9. Where the remaining time goes

- **Steps where every draft is accepted take about 2.0–2.3 s**, learner-bound: about
  0.6 s until the first verified groups, training, and a 0.2 s step and refit.
- **About one draft per step (of 128) is rejected.** Rejections occur at about 1.5e-5
  per token, the same for tokens drafted one or three versions earlier, so lookahead costs
  no acceptance. They reflect the true one-step policy change.
- **A rejection costs its remainder:** median about 450 tokens, decoded serially at about
  3 ms per token. That straggler tail is most of the gap between the 2.3 s floor and the
  3.0 s mean.
- **Generation is near saturation:** inference GPUs about 83% busy keeping up with
  drafting (about 115k drafted tokens per step); learner GPUs about 30%.
- **The learner is CPU-bound:** about 130 ms of CPU for 70 ms of GPU per training
  micro-batch. `policy.make_sequence_length_divisible_by=128` stops cuDNN fused attention
  from building a new plan for every padded length: 270 → 130 ms CPU per micro-batch,
  learner 1.48 → 1.19 s per step. Iteration time did not move because generation is the
  bottleneck.

## 10. When speculation pays

Speculation converts idle generation capacity into next-iteration rollouts; it cannot
create capacity. Its gain is bounded by the gap between the zero-staleness baseline and
an asynchronous pipeline.

| Setting | Zero staleness | Async lag 4 | Speculative |
|---|---|---|---|
| Qwen2.5-Math-1.5B, 2 + 2 GPUs (tail-bound) | 6.13 s | 4.26 s | **3.02 s** |
| Qwen3-4B, 2 + 2 GPUs (generation saturated) | ≈42.5 s | – | ≈43–47 s |
| Qwen3-4B, 3 inference + 1 training | 31.3 s | 31.3 s | ≈37 s |

At Qwen3-4B nearly every rollout runs to the 4096-token cap: the engine is
throughput-bound for the whole iteration and gradient streaming already hides the learner,
so zero staleness equals async and speculation only adds verification work. Use
speculation when the rollout phase ends in a latency-bound tail on mostly idle GPUs.

## 11. Scaling status

| Piece | Status |
|---|---|
| Drafting, streaming, abort | Megatron inference client and coordinator; inherits inference TP/PP/EP/DP |
| q storage | bf16 weight stash per pending version; memory O(parameters + batch x vocabulary) |
| Scorer memory | fp32 adds 4 bytes per parameter per shard; `verify_precision=model` keeps training dtypes (about 90% kept vs 99%) |
| Tensor parallelism | Vocab-parallel logits gathered per batch; validated at TP=2 (99.9% kept) |
| Pipeline parallelism | Forward-only pipeline schedule, last stage verifies; scorer log-probs at PP=2 bitwise equal PP=1. Inference must be pinned to PP=1 (`mcore_generation_config.pipeline_model_parallel_size=1`) |
| Context parallelism | Not implemented; the scorer raises. Needs packed (THD) scoring |
| MoE / expert parallelism | Provider copy carries the EP layout; fused kernels skip MoE modules; untested end to end |
| Large models on few learner GPUs | At 4B with one learner GPU: verify on the inference pool, or `verify_batch_tokens=4096` and the bf16 scorer to fit |

## 12. Configuration

Best 1.5B configuration, on top of the ThunderSync GRPO config:

```
NVIDIA_TF32_OVERRIDE=1
++thundersync.speculate=true
++thundersync.verify_mode=block
++thundersync.verify_precision=fp32
++thundersync.draft_budget=4096
++thundersync.draft_lookahead=3
++thundersync.draft_start_frac=0.0
++thundersync.verify_on=both
++thundersync.verify_chunks=4
++thundersync.verify_overlap=true
++thundersync.verify_longest_first=true
loss_fn.skip_zero_advantage_rows=true
policy.megatron_cfg.fp32_lm_head=true
policy.make_sequence_length_divisible_by=128
```

| Key | Meaning |
|---|---|
| `draft_lookahead` | Future iterations drafted concurrently; unfinished drafts resume across iterations |
| `draft_start_frac` | Fraction of this iteration's rollouts finished before drafting starts (0: from the start) |
| `verify_precision` | `fp32` scorer (TF32 GEMMs under the env flag) or `model` (training dtypes) |
| `verify_on` | `learner`, `inference` or `both` worker groups run verification |
| `verify_chunks` | Verification pieces; plans publish as each finishes |
| `verify_batch_tokens` | Tokens per scorer forward batch (bounds logits memory), default 16384 |
| `draft_budget` | Maximum new tokens per draft |

Debug switches: `THUNDERSYNC_SPEC_PROF` (verification timing), `THUNDERSYNC_SPEC_CHECK`
(cross-pool scorer check), `THUNDERSYNC_SELECT_CHECK` (selective vs full logits, PP=1),
`THUNDERSYNC_SPEC_TORCHPROF` (scorer profile, PP=1), `THUNDERSYNC_REJECT_LOG` (per-draft
outcomes and continuation timing), `THUNDERSYNC_LEARNER_PROF` and
`THUNDERSYNC_LEARNER_TORCHPROF` (training chunks), `THUNDERSYNC_SCORER_ATTN_BF16`
(experiment: bf16 attention in the scorer), `THUNDERSYNC_SPEC_SELFTEST` (keyed mode:
re-draft with the same seeds and verify on unchanged weights before training),
`THUNDERSYNC_MISMATCH_CHECK` (rollout vs learner log-probs before each optimizer step).

## 13. Code map

| File | Role |
|---|---|
| `thundersync_rl/grpo_loop.py` | GRPO loop: per-sample rollouts, learner loop, drafter and verification tasks, config, cohort ramp |
| `thundersync_rl/streaming.py` | StreamPlanner: arrivals to dispatches, bucket vs final chunks, zero-advantage chunks, DP balancing |
| `thundersync_rl/speculative.py` | SpeculativeGeneration: cohorts and segments, stash calls, chunked verification, plans, continuations |
| `thundersync_rl/worker.py` | Megatron worker extension: streaming train calls, scorer build and refresh, weight stashes, stashed block verification, draft streams |
| `thundersync_rl/block_verification.py` | Reference block verification, shared acceptance helpers, exact output law |
| `thundersync_rl/block_verification_fused.py` | Block verification from raw logits with a Triton residual-mass kernel |
| `thundersync_rl/scorer_kernels.py` | SDPA core attention, Triton SwiGLU, selective LM head |
| `thundersync_rl/keyed_sampling.py` | Position-keyed Gumbel-max sampling kernel (keyed mode) |
| `thundersync_rl/fused_train.py` | Fused verification + training (BI, experimental): retained graphs, logit stitching, forward substitution |
| `nemo_rl/algorithms/grad_streaming.py` | Core: per-(group, reward) gradient accumulators, final-first selection (shared with the single controller) |

Tests: 119 research unit tests (block verification by enumeration, fused verification
draw-for-draw, scorer kernels on GPU, streaming equivalence, speculative driver plans,
cohorts and ramp, chunked pools, error surfacing, fused verification + training: routing,
claims, config checks and gradient equality on a tiny causal model), 7 core zero-advantage tests, and GPU
functional checks for scorer and refit parallelism.

## 14. Batch invariance (NeMo-RL PR 3208)

Branch `thundersync-rl-bi` applies PR 3208 (`policy.megatron_cfg.zero_train_gen_mismatch`):
Megatron training and Megatron inference run the same batch-invariant kernel stack, so
generation log-probs and training log-probs of the same tokens are bitwise equal.

**How the kernels get batch invariance** (`te_native` backend): native cuBLASLt GEMMs with
a starved workspace (no split-K, so a fixed K-reduction order at every row count),
batch-invariant `aten::_log_softmax` / `mean`, TE RMSNorm kept native under 64-multiple
token alignment, FlashAttention 4 with `num_splits=1`. Requirements enforced by the PR's
validator: BF16, TP=1, CP=1, no packing, generation `transformer_impl=inference_optimized`
(so no QKV-bias models such as Qwen2.5), `make_sequence_length_divisible_by` a multiple of 64.

**Environment.** TE 2.18 (built from source against torch 2.11), flash-attn-4 4.0.0b19, CuTe
DSL 4.6.2 (including `nvidia-cutlass-dsl-libs-core`, the Python frontend), quack-kernels 0.6.4
and apache-tvm-ffi 0.1.11, as a PYTHONPATH overlay (`thundersync_work/build_bi_overlay.sh`).

**Measured with ThunderSync** (`THUNDERSYNC_MISMATCH_CHECK=1`: before the optimizer step the
learner recomputes the batch's log-probs and compares them with the rollout log-probs):

| Model | Mismatch |
|---|---|
| Qwen3-1.7B, BI | 0 on every token (4 steps) |
| Nemotron-Nano-9B-v2 (hybrid Mamba), BI, validator's hybrid exclusion relaxed | 0 on every token (3 steps) |

The PR's exclusion of hybrid Mamba models is an untested-scope limit: MCore's
batch-invariant Mamba decode (buffered chunk replay) already makes them exact.

**Speculation under BI.**

- *Scorer precision.* BI kernels are BF16-only (`TransformerConfig` asserts it), so an fp32
  scorer is built with `batch_invariant_mode=False` on its own copy, keeping the process-wide
  TE attention settings (its attention is replaced by SDPA anyway). A bf16 scorer would match
  the policy bitwise once it scores like training (training op sequence, rows padded to the
  training multiple, full-row LM head, training's fp32 log-softmax: 92-97% of tokens bitwise
  equal so far), but its rounding jitter between consecutive weight versions is part of the
  policy, so exact verification against it rejects long drafts: 33-44% of draft tokens kept on
  Qwen3-1.7B's ~3k-token outputs versus 94-98% with the fp32 scorer. The fp32 scorer is the
  default; rollouts then follow the fp32 approximation of the policy, as without BI.
- *Where the scorer runs.* On the learner it is stable but does not scale under BI: BI
  requires TP=1, so each learner GPU holds the full fp32 copy (4 B/param) plus a bf16 stash
  per pending version on top of training state (Qwen3-8B ran out of memory on 2 and 3 learner
  GPUs). On the inference GPUs (`verify_on=inference`, the recommended BI setting) it first
  broke the engine: acceptance decayed step by step (97% -> ~40%) and the engine eventually
  sampled from NaN logits (token id 155647 outside the 151936 vocab).
- *Root cause and fix.* Under BI, NeMo-RL turns flashinfer RoPE off, so the engine reads its
  RoPE table from `RotaryEmbedding.forward`, an `lru_cache(maxsize=32)` shared by every
  instance in the process, and the engine's inference CUDA graphs capture that table's
  address. The scorer's own RoPE calls evicted the engine's entry, and the generation
  worker's `cache_clear()` (prepare/finish generation) freed it; the scorer's activations
  and weight stashes then reused the memory the graphs kept reading. Fixes:
  `megatron_worker._pin_rotary_tables` memoizes the engine's RoPE tables per module with no
  eviction (BI path; unit-tested in `tests/unit/models/generation/test_megatron_rope_pin.py`),
  and the scorer's RoPE forward bypasses the shared cache. The scorer copy also builds
  standard Transformer Engine layers without CUDA graphs, so both verifier pools compute
  identical p and the scorer never joins the engine's graph state. Result (Qwen3-1.7B, BI,
  verification on inference GPUs, 10 steps): 92-99% of draft tokens kept with no decay; the
  train/generation log-prob gap stays at the fp32-scorer level (mean 0.011) on accepted
  tokens and is zero on engine-decoded ones. Qwen3-8B, BI, 1 node (2 inference + 2 training
  GPUs, verification on inference, 4 steps): 93-99% kept, 91-95% of drafts fully accepted,
  about 50 s per step at ~3.5k mean generated tokens.
- The same eviction hazard applies to any process that captures inference CUDA graphs with
  the non-flashinfer RoPE path and creates other RoPE models or calls `cache_clear()`
  (upstream: MCore could keep the dynamic-inference table in `rotary_pos_emb_cache`, as its
  flashinfer path does).

### BI-enabled optimizations (opt-in)

Bitwise train/generation parity removes work that exists only to bridge numerics. All
of these are off by default.

**Fused verification + training** (`thundersync.fused_verify_train`, `fused_train.py`).
*Experimental, opt-in; exact on GPU, not faster yet (A/B below).* With BI, the
learner's bf16 training forward is the engine's numerics, so p for verification can come
from the training model itself at θ_{k+1}, and that p forward is exactly the forward that
training of iteration k+1 needs for the same trajectory. Fused mode keeps it:

1. q is recomputed from the weight stashes as before (bf16 scorer, no grad).
2. p is a forward of the training model, inside the already open train step, shaped like a
   training micro-batch (one row, padded to `make_sequence_length_divisible_by`, same
   `model_forward` call). Rows whose full acceptance would make a finished rollout (EOS or
   length cap) run with autograd; the others without.
3. A draft that is fully accepted and finished keeps its graph on the verifying data-parallel
   rank, up to `fused_max_retained_tokens` per rank (`FusedGraphStore`); every other graph is
   freed at once. Its plan is complete, so the rollout returns immediately and the
   environment rewards it through the normal rollout path; no extra reward hook is needed.
4. The driver matches the finished trajectory to its retained forward (`claim_fused`), and
   the planner sends it as a one-row chunk pinned to that rank, into its (group, reward)
   bucket while the group is open or as a final row once it closed, exactly as any other
   trajectory. The worker runs the regular `train_microbatch`, with only `model_forward`
   replaced by the retained logits: same loss, token and sequence counts, bucket capture and
   finalization. The training row is the verified row plus the environment's reply; causal
   attention makes the verified logits its prefix, and the reply positions, masked out of the
   loss, are filled with constants that carry no gradient. Any mismatch (other tokens, an
   unmasked token after the verified row, a missing graph) falls back to the real forward.
5. Retained graphs never outlive the step: the barrier drops unclaimed ones before the
   optimizer step.

The gradient equals the unfused path's up to fp32 summation order (per-row losses are
additive, and BI makes a row's forward independent of its micro-batch). Each fused
trajectory saves one full training forward of its row (`fused/trajectories`,
`fused/forward_tokens_saved`; `fused/retained`, `fused/claimed`, `fused/fallbacks`,
`fused/unclaimed` track the rest). It pays where bf16 verification keeps most drafts
whole: short outputs, where block verification against the training-dtype policy keeps
about 87-91% of tokens. On long outputs bf16 acceptance is low (33-44% of draft tokens on
Qwen3-1.7B's ~3k-token outputs), few drafts finish whole, and the fp32 scorer
(not fusable: its p is not the training forward) is the better choice. Requirements,
enforced by `validate_fused_verify_train` and on the worker: `speculate`,
`verify_mode=block`, `verify_on=learner`, `verify_precision=model`,
`batch_invariant_mode`, PP=1 (and BI's TP=1, CP=1), generation temperature 1, no fused
linear log-probs, no MTP or draft-model training. `THUNDERSYNC_FUSED_CHECK=1` also runs the
real forward for each fused row and prints the largest logit difference over the verified
positions (expected 0).

GPU A/B (Qwen3-1.7B, BI, 2 + 2 GPUs, 2k-token cap, learner bf16 block verification, same
seed, 6 steps, `thundersync_work/fused_ab.sh`): every fused forward matched a fresh forward
bitwise (`THUNDERSYNC_FUSED_CHECK`: max |logit diff| 0), and the train/generation mismatch
check read 0 on every token in every step, whereas unfused bf16 verification left 1-4% of
accepted tokens inexact (its scorer copy is not the training forward). 5-13 trajectories per
step were fused (10-27k forward tokens saved, no fallbacks), but steady-state steps took
13.4-14.4 s against 10.5-11.0 s unfused: rollouts wait about 3 s longer for verification
forwards that run with autograd, which costs more than the training forwards they save at
this size. Acceptance was the same either way (41-52% of draft tokens) and well below the
fp32 scorer's 92-99%, so fused mode is a net loss here. It is worth retesting where
training forwards are expensive relative to verification (large models, learner-bound runs).

**Keyed verification under BI** (`verify_mode=keyed`). With bitwise parity the learner's
logits for a draft are the ones the engine sampled from, so keyed verification needs no
scorer copy and no weight stashes: p from the current training weights is all it reads, and
the check is token-level (keep the agreeing prefix, then the learner's own token). It stays
exactly on-policy, at the lower token-level acceptance of section 4.

**No importance-sampling correction.** Generation and training log-probs are bitwise equal
under BI, so the importance weights are exactly 1 and
`loss_fn.use_importance_sampling_correction` can be disabled. In fused mode the plan's
log-probs are computed with training's own log-softmax from the training forward, so this
holds for fully accepted speculative tokens too.

## 15. Open work

- **Stragglers.** About one rejected draft per step decodes its remainder at about 3 ms
  per token. Only faster single-sequence decoding helps, such as engine-level speculative
  decoding with an MTP or EAGLE head.
- **Generation capacity.** At 1.5B generation is near saturation on 2 + 2 GPUs, and a
  3 + 1 split makes the learner the bottleneck; faster engine decoding is the lever.
- **Training CPU overhead.** About 130 ms CPU vs 70 ms GPU per micro-batch; training CUDA
  graphs would need static or bucketed shapes.
- **Batch invariance at scale.** Run the BI speculation presets (inference-side verification)
  on 2-node 8B/14B and Nemotron-Nano on the cluster (1-node 8B passes); validate the opt-in
  BI optimizations on GPU.
- **Scaling gaps.** Context parallelism in the scorer, MoE validation, zero-advantage
  skip with sequence packing.
- **Upstream.** Default `make_sequence_length_divisible_by` to 64–128 for Megatron
  training.
- **Fused verification + training on GPU.** Validate `fused_verify_train` end to end
  (`THUNDERSYNC_FUSED_CHECK=1`, gradient against the unfused path) and measure the saved
  learner time on short-output tasks (section 14).
