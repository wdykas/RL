# ThunderSyncRL: zero-staleness GRPO via gradient streaming

Implementation of [ThunderSyncRL: Lossless Acceleration of Agentic Reinforcement Learning](https://arxiv.org/abs/2610.05935)
on NeMo RL with the **Megatron training backend** and the **non-colocated Megatron inference backend**.

Full design write-up (algorithm, verification math, scorer, results): [`DESIGN.md`](DESIGN.md)
(rendered page: [`docs/design.html`](docs/design.html)).

## Idea

In synchronous GRPO, the learner idles until the slowest rollout of the batch
finishes. Async RL hides that idle time but trains on stale trajectories.
ThunderSyncRL keeps the update *exactly* synchronous (every gradient is taken
at the same `theta_k` that generated the batch) and instead moves backward work
into the rollout tail: each trajectory is backpropagated as soon as its reward
arrives, without waiting for the rest of its group.

The obstacle is that GRPO's advantage `a_i` depends on the whole group. But the
update `sum_i a_i H_i` (with `H_i` the gradient of trajectory `i`'s masked log-prob
score at `theta_k`) is linear in the `H_i`, and every GRPO-style estimator gives
trajectories with equal rewards in the same group the same advantage. So we
accumulate

```
B[g, r] = sum_{i in group g, r_i = r} H_i
```

while the group is open, and when it closes fold `sum_r a_g(r) * B[g, r]` into the
batch gradient. For binary rewards this is two buffers per open group, the same
as the paper's `G1 = sum r_i H_i`, `G2 = sum H_i`. Unlike the paper's affine form it
is exact for **any** group-wise estimator, including NeMo RL's default
leave-one-out baseline with leave-one-out std, which is *not* affine in the
reward. When a group sees more than `max_buckets_per_group` distinct rewards,
its buckets collapse to the paper's `(G1, G2)` form. That is exact for affine
estimators (shared mean/std), and the accumulator raises an error, rather than
silently corrupting the update, if the estimator is not affine.

Normalization (`1/N_tokens` or `1/N_seqs`), the DP reduction, grad clipping, the
single optimizer step and the refit all happen once, at the batch barrier.

## How it maps onto NeMo RL

| Piece | Where |
|---|---|
| Learner step lifecycle: zero grads, local accumulation under `no_sync`, deferred 1/N, single DP reduce, `optimizer.step()` | existing `MegatronPolicyWorker.begin_train_step` / `train_microbatch` / `finish_train_step` |
| Moving each chunk's gradient out of the DDP `grad_data` buffers into per-(group, reward) accumulators; write-back at the barrier | `nemo_rl/algorithms/grad_streaming.py`, `thundersync_rl/worker.py` (worker extension via `policy.worker_extension_cls_fqn`) |
| Arrival → dispatch planning (bucket chunks vs. final-advantage chunks, group closes, DP token balancing) | `thundersync_rl/streaming.py` |
| Rollouts streamed per sample from the Megatron inference engine; learner loop; refit | `thundersync_rl/grpo_loop.py` |
| Advantages | NeMo RL's own `_create_advantage_estimator` + `_clip_grpo_advantages`, applied per closed group |

No core `nemo_rl/` files are modified.

## Running

```bash
# 4 GPUs: 2 inference (Megatron, non-colocated) + 2 training (Megatron)
uv run research/thundersync_rl/run_thundersync_grpo.py \
    --config research/thundersync_rl/configs/grpo_math_1.5b_thundersync_megatron.yaml

# Group-level streaming / batch-synchronous baseline through the same code path
uv run research/thundersync_rl/run_thundersync_grpo.py \
    --config research/thundersync_rl/configs/grpo_math_1.5b_thundersync_megatron.yaml \
    thundersync.granularity=group   # or: batch
```

Run these from `research/thundersync_rl` with `PYTHONPATH=$PWD:${PYTHONPATH:-}`, or
use the functional scripts below, which set this up.

### Config (`thundersync:` block)

| key | meaning |
|---|---|
| `granularity` | `trajectory`: ThunderSyncRL with final-first scheduling (closed-group work first; idle learner time streams one open group's trajectories). `group`: start a group's backward when the whole group has finished. `batch`: synchronous baseline. All three produce the identical update. |
| `storage_device` | `cuda`, or `cpu` (pinned host memory, as in the paper) for the per-group accumulators. |
| `max_buckets_per_group` | distinct rewards per open group before collapsing to `(G1, G2)`. |
| `max_open_groups` | cap on groups holding gradient buckets at once. Accumulator memory is about `(1 + max_buckets_per_group * max_open_groups)` fp32 copies of the per-rank gradient; the worker checks this fits at step start and fails fast if not. |
| `max_chunk_trajectories` | cap on trajectories per backward chunk. |

Single-controller equivalent: `async_rl.trajectory_streaming` (`storage_device`,
`max_buckets_per_group`, `max_open_groups`, `publish_coalesce_s`) with the
in-order sampler at `max_lookahead_versions=0`.

Zero staleness removes policy lag, not the numerical mismatch between the
inference engine and the trainer at the same weights. Measured: a per-token
probability ratio of about 1.0035 and a KL of about 1e-4. Removing it bitwise
would take batch-invariant kernels in both engines. To correct for it instead,
set `loss_fn.use_importance_sampling_correction=true` (optionally
`truncated_importance_sampling_ratio`, `truncated_importance_sampling_type`).
The weights exp(log pi_train - log pi_infer) are detached per-token constants,
so streaming stays exact, and they use the training forward's log-probs, so no
extra forward pass is needed.

For zero-staleness runs, set `loss_fn.force_on_policy_ratio=true`. With one
optimizer step per batch, every gradient is taken at the same theta_k that
generated the data, so a separate `prev_logprobs` forward pass would return
exactly the current log-probs (ratio 1, clipping inactive).

### Requirements, validated at startup

- `loss_fn.force_on_policy_ratio=true` (on-policy, one step per batch) and `loss_fn.reference_policy_kl_penalty=0`. Importance-sampling correction (`use_importance_sampling_correction`, truncated or not) is optional and supported.
- `policy.generation.colocated.enabled=false`: rollout and learner need separate GPUs.
- `distributed_data_parallel_config.grad_reduce_in_fp32=true`.
- Sequence packing and dynamic batching are off. Streamed chunks are small; packing support is follow-up work.

### Memory

Each open group with streamed trajectories holds `#distinct rewards` fp32 copies
of the per-rank gradient (2 for binary rewards), plus one batch accumulator.
Groups whose trajectories are all ready at dispatch time skip the accumulators
and are added directly with their final advantages. For large models, set
`storage_device: cpu`.

## Tests

```bash
cd research/thundersync_rl
# CPU: streamed == sum_i a_i H_i under NeMo RL's estimators, random arrival orders, DP=1..3
uv run --group test pytest tests/unit
# GPU (2): Megatron sync train() vs streamed, parameter gap Delta_k after each update (paper App. G)
bash tests/functional/equivalence.sh
# GPU (2): end-to-end 2 GRPO steps, Megatron train + non-colocated Megatron inference
bash tests/functional/thundersync_grpo.sh
```

## Results (1x GB200 node, 4 GPUs)

**Equivalence** (`tests/functional/check_equivalence.py`, Qwen2.5-0.5B, fp32 with
TF32 disabled, 4 groups x 8, random arrival order, 3 updates):

| comparison | grad relative error | parameter gap Delta_k |
|---|---|---|
| streamed vs. sync `Policy.train` | 4.6e-6 to 7.4e-6 | 1.0e-4 to 1.3e-4 |
| control: sync vs. sync on a row-permuted batch | 0.9e-6 to 6.5e-6 | 3.6e-5 to 6.2e-5 |

Streaming deviates from the synchronous update by the same order as merely
re-batching it: fp32 summation-order noise. With TF32 left on, both are about 1e-2.

**Speed** (Qwen2.5-Math-1.5B-Instruct, 2 Megatron inference GPUs + 2 Megatron
training GPUs (DP=2), 16 prompts x 8, up to 2048 tokens; mean of steps 1-5):

| `empty_unused_memory_level` | mode | iteration | rollout | last rollout to step done | learner busy |
|---|---|---|---|---|---|
| 0 (config default) | `granularity=trajectory` | **8.53 s** | 7.77 s | **0.57 s** | 6.07 s |
| 0 | `granularity=batch` (sync) | 10.29 s | 6.87 s | 3.22 s | 3.08 s |
| 1 | `granularity=trajectory` | 8.93 s | 7.15 s | 1.55 s | 7.63 s |
| 1 | `granularity=batch` (sync) | 11.58 s | 6.99 s | 4.37 s | 4.32 s |

With level 0, the time after the last rollout drops from 3.22 s to 0.57 s
(the DP reduce, the optimizer step and the refit), so an iteration is close to
the rollout time. Iteration speedup is 1.21x to 1.30x. Rollout time varies by
about 1 s between runs because sampled lengths differ, so the tail column is
the cleaner comparison. The learner does about twice the backward work when
streaming: each of the many small chunks pays fixed per-call costs. A
micro-batch of 1 made this worse (10.3 s learner busy), so the fix is larger,
packed chunks rather than less padding.

## When does trajectory granularity beat group granularity?

`thundersync.granularity` selects when backward work may start: `trajectory`
(as soon as a reward arrives, ThunderSyncRL), `group` (when the trajectory's
whole group has finished) or `batch` (synchronous baseline). All three produce
the identical update. Only the overlap differs.

**Model.** The step ends at T (the last rollout's arrival) plus a tail: the
backward work still owed at T, plus the fixed barrier (DP reduce, optimizer
step, refit). With group release, the last group's G trajectories all start
after T. With trajectory release, only the last trajectory does. So the most
trajectory granularity can save is about (G-1) times the backward time of a
late trajectory. Against that, every streamed bucket chunk is an extra, small
Megatron micro-batch.

**Measurements** (Qwen2.5-Math-1.5B, Megatron training + Megatron inference, 16x8, up to 4096 tokens):

| assumption | measured |
|---|---|
| siblings finish independently, so groups done = F(t)^8 | false: intra-class correlation of arrival times within a group is about 0.72. When 50% of trajectories are done, 29% of groups are done (the independent model predicts 0.4%) |
| same learner cost per token | false: fixed cost about 0.09-0.16 s per bucket chunk. A micro-batch is CPU-launch bound at this size (torch profiler: 114 ms CPU vs 21.5 ms GPU for 1x1024 tokens) |

Replaying the measured arrivals with the fitted learner cost model
(`thundersync_work/analyze_timeline.py`) gives these tails after the last
arrival:

| | trajectory release (measured cost) | trajectory release (zero per-bucket overhead) | group release |
|---|---|---|---|
| 2 inference + 2 train GPUs | 0.31 s | 0.19 s | 0.36 s |
| 3 inference + 1 train GPU | 1.07 s | 0.21 s | 0.37 s |

End-to-end iteration time (mean of steps 1-7):

| setup | batch | group | trajectory |
|---|---|---|---|
| 2 inference + 2 train GPUs | 10.55 s | 8.08 s | 7.98 s |
| 3 inference + 1 train GPU | 12.26 s | **7.89 s** | 8.32 s |

**Larger model** (Qwen3-4B, long reasoning traces, up to 4096 tokens, 2+2 GPUs).
Groups close later here (ICC 0.54; 10% of groups are done when 50% of
trajectories are done), and the per-bucket overhead is negligible next to the
per-row GPU work (fitted c of about 0), so trajectory release should win.

| mode | iteration | rollout | tail | learner busy |
|---|---|---|---|---|
| batch (sync) | 55.77 s | 48.84 s | 6.59 s | 6.5 s |
| group | 51.04 s | 48.87 s | 1.84 s | 10.9 s |
| trajectory, host buckets, `max_open_groups=2` | 50.67 s | 48.28 s | 2.08 s | 22.3 s |
| trajectory, host buckets, `max_open_groups=6` | **50.26 s** | 48.48 s | **1.44 s** | 33.8 s |

The optimizer-step barrier itself is only 0.07 s (`time/finish_step`), so the
tails are backward work. Trajectory mode did not cut its tail because of the
open-group cap. A 4B model's fp32 gradient is 15 GB per copy, so GPU-resident
buckets do not fit next to its activations, and memory forced
`max_open_groups=2`. Long reasoning keeps about 16 groups open at once, so 14 of
them behaved like group mode: the dispatches spanning the last arrival were
24-32 final rows with no buckets. With host buckets, the cap is bounded by
host RAM, and the worker checks that it fits. Raising the cap to 6 cut the tail
to 1.44 s (22% below group). In the steps where at most 6 groups were still
open, only the final trajectory remained after the last arrival (tail 0.4-0.5 s).
A cap large enough for every open group would need DP-sharded or
lower-precision buckets.

**Conclusion.** Zero-staleness streaming beats synchronous training at every
scale tested: 1.3-1.55x at 1.5B and 1.11x at 4B. At 4B the rollout dominates
the step (49 of 51 s), which caps what any learner-side overlap can save. Which
granularity wins follows from three measurable quantities:

1. **Within-group correlation of completion times.** High correlation (ICC
   0.72 at 1.5B) means groups close nearly as early as their trajectories, so
   group release already starts most work early.
2. **Per-bucket overhead relative to per-row backward GPU time.** About 90 ms
   of CPU dispatch per Megatron micro-batch dominates at 1.5B and is negligible
   at 4B with long rows.
3. **Open-group capacity.** Buckets cost about 2 fp32 gradient copies per open
   group. If the cap is far below the number of concurrently open groups,
   trajectory mode degenerates to group mode.

Small models, or high correlation: use `granularity: group`. Large models with
long, variable trajectories and enough accumulator memory (host buckets, a
large `max_open_groups`): use `granularity: trajectory`.
`analyze_timeline.py` measures all three from a short run.

## Core integration (SingleController)

The streaming pieces also live in core NeMo RL and run through the
SingleController GRPO trainer (Megatron training, non-colocated Megatron
inference, in-order sampler with `max_lookahead_versions=0`), enabled by
`async_rl.trajectory_streaming`:

- `nemo_rl/algorithms/grad_streaming.py`: the accumulator.
- `MegatronPolicyWorker` split train-step API: `begin_train_step(grad_streaming=)`,
  `train_microbatch(stream_bucket=)`, `close_stream_groups`, `discard_stream_groups`.
- Per-trajectory publishing: coalesced `TQReplayBuffer.commit_trajectories`, then `seal_group`.
- Train-pump trajectory branch with final-first selection, the open-group cap, and
  retry discards.
- Rollout snapshots are deferred while a trajectory step is open.

Exactness tests: `tests/unit/single_controller/test_trajectory_streaming_pump.py`
(drives the real pump helpers against a real replay buffer, including retries
and the cap).

## Not yet implemented

- On-policy distillation streaming (the paper's per-turn OPD variant).
- DP-sharded accumulators (reduce-scatter on capture): would divide accumulator memory by the DP size.
- Lower per-micro-batch CPU overhead in the Megatron training path (CUDA graphs or larger packed
  micro-batches). This is what limits trajectory granularity on small models.

## Cross-iteration speculative rollouts (`thundersync.speculate`)

Zero staleness forces a drain at every update: iteration k+1 cannot sample until
theta_{k+1} exists. We draft iteration k+1's rollouts *during* iteration k under
theta_k and verify them exactly once theta_{k+1} is available.

Key measurements (Qwen2.5-Math-1.5B, bf16):

* One update moves the policy less than bf16 rounding noise: per-token rejection
  pi_k -> pi_{k+1} is ~1.7e-3 and identical for drafts made 1-4 iterations earlier
  (lag-flat), and identical between engines at equal weights. In fp32 the true
  one-step drift is ~3e-4. Forecasting theta_{k+1} (momentum, exact Adam state)
  does not help at bf16; precision does (fp32 LM head: 1.7e-3 -> 1.1e-3).
* Token-level verification keeps ~60% of draft tokens; block verification (Sun
  et al. 2024) ~87-91%, because the noise has random sign and cancels across a
  block.

Design (`speculative.py`, worker methods in `worker.py`):

1. Once `draft_start_frac` of iteration k's rollouts finished, the engine streams
   drafts for the next `draft_lookahead` iterations' prompts (`start_drafts`);
   they are aborted at the deadline right before the optimizer step
   (`collect_drafts`) - any prefix is a valid draft - and resume from their
   prefixes under the newer weights in the following iteration. Each draft
   records which weight version drafted which positions.
2. Still at theta_k, the verifiers stash theta_k's weights (`stash_weights`,
   bf16, one copy per version that drafted pending tokens).
3. At iteration k+1, the verifiers recompute q per batch from the stashes and p
   from theta_{k+1}, block-verify each draft (`verify_drafts_block_stashed`) in
   chunks overlapped with the rollouts, and publish each trajectory's plan
   (kept prefix + bonus/residual token); the engine decodes only the remainders.
   The rollout code is unchanged (`SpeculativeGeneration` wraps the generation
   interface).

Exactness: block verification returns exact samples of p given the drafts' true
q. q is computed with trainer numerics while drafts are sampled by the inference
engine, so samples are exact up to the train/inference numerical mismatch (the
same envelope ordinary RL already has); with batch-invariant kernels (NeMo-RL PR
3208) q is bitwise the sampling distribution. `verify_mode=keyed` is an exactly
on-policy alternative (position-keyed Gumbel-max sampling, Triton kernel
`keyed_sampling.py`) with token-level acceptance.

Results (2 gen + 2 train GPUs, 6 steps, group streaming):

| config | iteration s | rollout s | draft tokens kept |
|---|---|---|---|
| baseline | 7.65 | 7.02 | - |
| keyed verification | 6.92 | 5.65 | 51% |
| block verification | 6.28 | 3.40 | 87% |
| + zero-adv skip + streaming drafts | 5.48 | 4.03 | 86% |
| + fp32 LM head | 4.45 | 3.10 | 91% |
| + Megatron fp32 scorer, TF32 GEMMs (`verify_precision=fp32`, `NVIDIA_TF32_OVERRIDE=1`) | 4.70-4.98 | 2.8-3.2 | 98-99% |
| + verification split across learner and inference GPUs (`verify_on=both`) | 4.49 | - | 99.3% |
| + multi-iteration drafts (`draft_lookahead=2`) | 4.57 (median 4.07) | - | 99.5% (97.5% drafts whole) |
| + verification overlapped with rollouts on both pools (`verify_overlap`, 4 chunks, longest first), lookahead 2 - 22 steps | **4.21 (median 3.73)** | - | 99.3% (97.8% drafts whole) |
| + lookahead 3, drafting from iteration start - 40 steps | 3.97 (median 3.84, p90 4.59) | - | 99.1% (98.6% drafts whole) |
| + fused fp32 scorer kernels (SDPA attention, Triton SwiGLU) - 30 steps | 3.55 (median 3.35) | - | 99.2% |
| + scoring only needed positions (per-version forwards stop at the segment end, LM head on segment positions only) | 3.31 (median 3.07) | - | 99.2% |
| + zero-advantage rows in chunks of their own (learner 2.04 -> 1.55 s) - 30 steps | **3.03 (median 2.79)** | - | 99.4% (98.5% drafts whole) |
| same, **100 steps** (zero-staleness baseline without speculation, same settings: 6.13, median 5.86) | **3.02 (median 2.70, p90 4.28)** | - | 99.5% (98.4% drafts whole) |
| reference: regular async, lag 4 (single controller), same settings (zero-adv skip, fp32 head) | 4.26 (median 2.40, p90 7.75; generation-bound, bursty) | - | - |

Scorer: block verification scores p and q with a second Megatron GPTModel
(training provider copy with fp32 dtypes, refreshed by copying the training
parameters) - bf16 activation rounding, not policy change, caused nearly all
rejections. Its fp32 forward replaces Megatron's unfused fp32 paths with torch
SDPA attention and a Triton SwiGLU (`scorer_kernels.py`), and the LM head runs
only on positions whose logits verification needs (bitwise identical logits).
Attention needs fp32-level accuracy: bf16 attention inside the fp32 scorer
drops acceptance from 99.2% to 96.4% and slows iterations by 25%.
Block verification itself runs from raw logits (`block_verification_fused.py`,
draw-for-draw identical to the reference, no [g, V] temporaries).

Where the remaining time goes (1.5B, 2+2): a step whose drafts are all
accepted takes ~2.3 s (learner-bound: ~0.6 s until the first verified groups,
1.55 s training, 0.2 s refit). About one draft per step (of 128) is rejected:
rejections occur at ~1.5e-5 per token, independent of how many versions old
the drafting weights are (lookahead is free), so they reflect the one-step
policy change. The rejected trajectory's remainder (median ~450 tokens) then
decodes serially at ~3 ms/token, which is the straggler tail on top of the
2.3 s. Pausing future-cohort drafts while such continuations decode does not
help (drafting time is worth more than the contention it causes).

Tests: `tests/unit/test_block_verification.py` checks by exact enumeration that
block verification returns the target distribution (to 1e-10).

### Scaling to large models

| piece | status |
|---|---|
| drafting / streaming / abort | Megatron inference client + coordinator: inherits inference TP/PP/EP/DP |
| q for verification | bf16 stash of the drafting weights (2 B/param per model-parallel shard per pending version), q recomputed per batch at verification - memory O(params + batch x vocab), not O(draft tokens x vocab) |
| scorer | second Megatron GPTModel from the training provider (same TP/PP/EP sharding); fp32 adds 4 B/param per shard; `verify_precision=model` keeps training dtypes (2 B/param, bf16 speed, ~90% vs ~98% tokens kept) |
| TP | vocab-parallel logits gathered per batch; rows owned by DP rank; DP-seeded sampling so TP partners agree - validated TP=2 (99.9% kept) |
| PP | forward-only Megatron pipeline schedule; last stage verifies - validated: scorer logprobs at PP=2 bitwise equal PP=1 (`tests/functional/check_scorer_parallelism.py`) |
| CP | not yet: scorer raises NotImplementedError for context parallelism |
| EP / MoE | provider copy carries the EP layout; untested (fp32 grouped GEMM support to check) |
| compute | verification = 2 forwards over draft tokens (q, p); fp32/TF32 scorer costs more than a bf16 forward - for learner-bound runs use `verify_precision=model` or put scoring on idle capacity |
| engine capacity | speculation fills idle generation capacity; at Qwen3-4B on 2 saturated inference GPUs steady state only matches the baseline (~43 s) and lookahead-3 warm-up is slow - needs spare generation capacity |
| memory | per pending draft version a bf16 stash (2 B/param) + the fp32 scorer (4 B/param): at 4B this OOMs the learner with `verify_on=both`; use `verify_on=inference` (or fewer versions / bf16 scorer) |
| zero-advantage skip | split train API, TP/PP-consistent; rejects sequence packing/dynamic batching (driver-side row dropping needed for those) |

PP end-to-end: training PP=2 with inference pinned to PP=1
(`++policy.generation.mcore_generation_config.pipeline_model_parallel_size=1`) runs
speculation normally (98.7-99.8% draft tokens kept, rewards as usual). Without the override
the dedicated inference model inherits the training PP=2 and generates garbage (Megatron
inference with PP>1 in this setup); the refit export itself is correct at PP=2
(`tests/functional/check_refit_export_parallelism.py`: same 28 layers and checksum as PP=1).

### When speculation pays

Speculation converts idle generation capacity (the latency-bound tail and drain of a
zero-staleness iteration) into next-iteration rollouts. It cannot create capacity:

| setting | baseline | speculative (steady state) |
|---|---|---|
| Qwen2.5-Math-1.5B, 2+2 GPUs (tail-bound), 100 steps | 6.13 s | 3.02 s (-51%; async lag 4: 4.26 s) |
| Qwen3-4B, 2+2 GPUs (generation saturated) | ~42.5 s | ~43-47 s |
| Qwen3-4B, 3 gen + 1 train (generation throughput-bound, learner mostly idle) | 31.3 s (async lag 4: 31.3 s) | ~37 s; learner-side verification 41-64 s |

Speculation can at most close the gap between the zero-staleness baseline and
an async pipeline, and at Qwen3-4B (3+1) that gap is zero: nearly every rollout
runs to the length cap, the engine is throughput-bound all iteration, and
gradient streaming already hides the learner.

Use it when the rollout phase is dominated by a few long stragglers on mostly idle GPUs;
disable it when the decode batch stays large for the whole iteration. Delaying or pausing
drafting to free decode capacity (for example until few rollouts remain in flight, or
while continuations decode) was tried and does not help at 1.5B: drafting time is worth
more there than the decode capacity it shares.
