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

Accumulator memory. With buckets (`max_open_groups > 0`) the accumulator keeps a batch
accumulator `batch_acc` (one fp32 copy of the rank's grad buffers, always on the GPU)
plus the open buckets, and the grad buffers are scratch space between chunks. Group
granularity (`max_open_groups=0`) has no buckets: every chunk carries final advantages,
so its gradient stays in the grad buffers and accumulates there as in non-streamed
training. No `batch_acc`, no per-chunk copy, `finalize` is a no-op, and the memory check
needs 0 bytes (−30.5 GiB/GPU for Qwen3-8B on 2 GPUs). It is bitwise equal to the
`batch_acc` form when every chunk runs one micro-batch (`acc + (0 + g)` = `acc + g`).
A chunk of several micro-batches is now summed micro-batch by micro-batch
(`(acc + g1) + g2`, the order of non-streamed training) instead of chunk-sum first
(`acc + (g1 + g2)`), so the two forms differ only in fp32 summation order.

Scheduling:

- **Final-first.** Rows of closed groups (final advantages known) always go first. Idle
  learner time streams one open group's rows into buckets.
- **Length-sorted chunks.** Final rows from different groups share micro-batches, sorted
  by length to limit padding, balanced over data-parallel ranks by tokens. With
  zero-advantage skipping (and no EP) the balance counts only live (nonzero-advantage)
  chunks' tokens, total tokens breaking ties: a skipped chunk costs no forward. This
  changes only which rank runs a chunk (fp32 order).
- **Zero-advantage chunks.** With `loss_fn.skip_zero_advantage_rows`, zero-advantage rows
  are chunked separately; the worker counts their tokens for normalization and drops the
  whole chunk without a forward pass. Learner time 2.04 → 1.55 s per step. A rank whose
  first chunk of a step is all zero-advantage no longer runs a forced forward/backward
  to report metrics: it reports one synthesized all-masked metric entry (zero sums, ±inf
  extrema) from the metric keys seen earlier on that worker. Only the worker's very first
  such chunk still runs. Gradients are unchanged (the forced pass added exact zeros);
  diagnostic metrics now average over the rows that actually ran.
- **Expert parallelism.** With `expert_model_parallel_size > 1` every forward/backward
  runs an all-to-all over the EP group, which spans data-parallel ranks, so all ranks must
  run the same number of micro-batches per dispatch. The planner then pads short ranks
  with one-row padding chunks (fully masked: no gradient, no normalization count), and the
  worker ignores `skip_zero_advantage_rows` (zero-advantage rows run a forward).
  Open-group buckets are not exact under EP: a bucket is scaled by its group's advantage
  afterwards, assuming the rank's gradient came only from its own rows, but expert weights
  also get gradient from tokens other EP ranks route to them. `validate_config` therefore
  requires `granularity=group` (or `max_open_groups=0`) with EP; final chunks carry their
  advantages in the loss and are exact. GPU check (random 4-layer, 8-expert Qwen3-MoE,
  EP=2 x DP=2 per policy, fp32, uneven per-rank chunk counts): streamed and synchronous
  updates match (Delta_k <= 6.6e-5 over 3 steps, same as a row-permuted synchronous control);
  with open-group buckets the expert weights diverged (Delta_k 0.24).
  Verification under EP: every scorer forward is an all-to-all over the EP group and each
  rank holds only its own experts, so the learner's verifiers run in lockstep
  (`_expert_parallel_lockstep`): the ranks exchange their per-batch stash versions once,
  each batch slot scores the union of versions in the same order, and a rank without rows
  for a slot or version runs a one-row dummy forward; keyed verification pads its batch
  count the same way. Inference-side verification is rejected under inference EP: the engine
  steps its EP ranks together, so a worker blocked in a scorer forward deadlocks against its
  peers' engine all-to-all (observed). End to end on the tiny MoE (EP=2 learner and
  inference, `verify_on=learner`): block and keyed speculation run without hangs.

### Streaming under pipeline parallelism

Branch `thundersync-streaming-pp`, for ~500B MoE training at PP = 4..16. Unit-tested on
CPU with a fake schedule; the GPU equivalence check at PP=2 is written but not run yet.

**Before.** The worker ran every chunk as its own `train_microbatch`, i.e. its own
`forward_backward_func` call with `m = ceil(rows / mbs)` micro-batches (a chunk of at
most `mbs` rows is one micro-batch of exactly its rows). A rank given `k` chunks in a
dispatch ran `k` pipeline schedules. A 1F1B call of `m` micro-batches takes `m + P - 1`
micro-batch slots, so it idles `(P - 1) / (m + P - 1)` and the dispatch idles
`k (P - 1)` slots out of `M + k (P - 1)` (`M` = the rank's micro-batches). With equal
chunks the bubble depends only on `m`, not on `k`: `max_chunk_trajectories=8`, mbs=1
gives 27% (PP4), 47% (PP8), 65% (PP16); mbs=4 (m = 2) gives 60%, 78%, 88%. Each EP padding
chunk was a separate one-micro-batch call ((P-1)/P idle). Under EP x PP the per-call
micro-batch counts could also differ between EP peers (the planner equalized only the
total), so peers could run different forward/backward sequences against the same
all-to-all. That case was never run.

**After: one call per dispatch and gradient destination.** `stream_accumulate` groups a
dispatch's items with `group_pipeline_calls` and runs each group with the new core
method `MegatronPolicyWorkerImpl.train_microbatches(datas, stream_bucket,
micro_batch_sizes)`. All final chunks (group granularity: the whole dispatch) form one
call. Their gradient accumulates in place in the grad buffers (or is captured into
`batch_acc` once). The chunks of one open-group bucket `(group, reward)` form one call,
captured once into that bucket. A fused chunk (PP=1 only) runs alone. Calls are ordered
by first item, and items keep dispatch order within a call. `train_microbatch(data)` is
now `train_microbatches([data])`.

- *Per slice, unchanged:* normalization counts (`local_valid_*` over all rows),
  zero-advantage skipping (`_drop_zero_advantage_rows` with the slice's own micro-batch
  size), the per-chunk micro-batch size `min(rows, mbs)`, MTP masks and media masks.
  `num_chunks` still counts slices. New: `num_pipeline_calls`. A call's iterator chains
  the slices' micro-batch iterators. `LossPostProcessor` gets the call's total `M`, which
  cancels mcore's `1/M` loss scaling exactly as before. One change: an all-zero slice
  is dropped without a forward whenever another slice of the same call is live (that
  call reports the step's metrics). Before, this happened only after the worker's first
  metrics.
- *Shapes.* Pipeline stages receive into buffers of the call's fixed
  `(seq, mbs, hidden)` unless `config.variable_seq_lengths`. When PP > 1 and the slices'
  `(micro-batch size, padded length)` differ, `train_microbatches` sets
  `variable_seq_lengths=True` for that call only. Every send then exchanges a 3-int
  shape first: one small extra P2P per send, which is negligible against MoE micro-batch
  times but not measured. This keeps each micro-batch's computation identical to the
  per-chunk path with no padding. Megatron's `TransformerConfig` rejects
  `variable_seq_lengths` with the `allgather` MoE token dispatcher, so in that case the
  slices fall back to one call each, with a warning. With uniform shapes or PP=1 nothing
  changes. The variable-shape path warns "untested end to end" until the GPU check
  passes.
- *Exactness.* The same micro-batches run in the same order. The gradient differs from
  the per-chunk path only in fp32 rounding: the order of `batch_acc` sums in bucket mode,
  and `M/M` vs `m/m` loss scaling. CPU tests check this against per-chunk calls and an
  fp64 reference.
- *EP.* EP requires group granularity, so a rank's whole dispatch is one call. The
  planner's padding makes every rank's `M` equal, so EP peers run identical 1F1B
  sequences. The test checks one call per rank with equal counts.
- *Not covered:* interleaved (virtual) PP. Megatron's interleaved schedule needs at least
  PP micro-batches per call, and the remainder group must be 0 or at least PP. A merged
  call meets this more often than per-chunk calls, but tail dispatches can still violate
  it (Megatron raises). Padding micro-batches for VPP is not implemented.

**Planner: `thundersync.min_pipeline_microbatches`** (default `null` = 2·PP when PP > 1,
else 0). While rollouts run (`has_work(flush=False)`), rows of closed groups are held
until they make at least `min_pipeline_microbatches x DP` micro-batches. They are counted
as the worker will run them: chunks of `max_chunk_trajectories`, `ceil(rows/mbs)` each,
and skipped all-zero chunks count 0. Bucket work and pending closes are not held, so
idle time still streams open groups. Once the rollouts are done, `learner_loop` flushes
(`flush=True`), so the tail never waits for the threshold.
`thundersync.merge_pipeline_calls=false` restores per-chunk calls (A/B and control).

**Bubble: analytic** (`pipeline_bubble_fraction`, `k` equal chunks of `m` micro-batches
per rank, per-chunk calls → one call):

| PP | k=4, m=8 | k=8, m=8 | k=8, m=2 (mbs 4) | held m = 2P | held m = 4P |
|---|---|---|---|---|---|
| 4 | 27.3% → 8.6% | 27.3% → 4.5% | 60.0% → 15.8% | 27.3% | 15.8% |
| 8 | 46.7% → 17.9% | 46.7% → 9.9% | 77.8% → 30.4% | 30.4% | 17.9% |
| 16 | 65.2% → 31.9% | 65.2% → 19.0% | 88.2% → 48.4% | 31.9% | 19.0% |

**Step time: modeled** (`scripts/pp_streaming_sim.py`: real planner, 512 rollouts,
G=16, DP=8, mbs=1, chunk 8, lognormal arrivals with `spread` = arrival scale relative to
the learner's work, 8 seeds; step time in micro-batch slots):

| spread (regime) | PP | per-chunk | merged | merged + hold 2P | merged + hold 4P |
|---|---|---|---|---|---|
| 0.2 (learner-bound) | 4 | 126.4 | 112.1 | 110.0 | 115.0 |
| 0.2 | 8 | 164.9 | 123.0 | 122.8 | 120.8 |
| 0.2 | 16 | 247.7 | 147.4 | 134.7 | 156.4 |
| 0.5 | 8 | 226.0 | 213.6 | 217.0 | 232.6 |
| 0.5 | 16 | 285.5 | 228.1 | 240.6 | 272.6 |
| 0.8 | 16 | 356.2 | 339.7 | 356.7 | 388.7 |
| 1.5 (rollout-bound) | 8 | 596.0 | 596.0 | 607.7 | 619.7 |

Merging never hurts in this model, and it cuts learner-bound steps by 11% (PP4), 25%
(PP8) and 40% (PP16): bubbles that sit on the critical path shrink. When rollouts
dominate, small dispatches fall in idle time and their bubbles cost nothing. The hold
lowers the bubble fraction everywhere, but held rows run after the last arrival. The
model shows it helping only when the learner is the bottleneck at PP16 (−9%), and
lengthening the tail elsewhere (up to +5% at 2P, +9% at 4P). The default 2·PP is the
requested setting; this model says to A/B it against `min_pipeline_microbatches=0` on
the real workload before relying on it.

**GPU check (written, not run):** `tests/functional/equivalence_pp.sh` runs
`check_equivalence.py` with Qwen2.5-0.5B, fp32, PP=2, TP=1, 2+2 GPUs,
`thundersync.max_open_groups=0`, mbs 4, `--trim-seq` (chunks cut to their longest row
rounded up to 16, so one call mixes row counts and sequence lengths), and the default
hold (4 micro-batches). It runs once merged and once with `merge_pipeline_calls=false`
as the control, and gathers parameters and gradients from both stages. It passes when
the gradient relative error is ≤ 1e-4 vs `Policy.train`.

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

**Lazy verification** (`verify_lazy=true`, off by default until the GPU A/B).
`lazy_stop_index` uses two facts: h_g = b_g, and h_i ≤ b_i (because r_i ≤ b_i). It
draws the same η = `rand(g+1)` as the eager rule, then:

- **Whole draft.** If η_g ≤ b_g, τ = g and no r_i is computed. b needs only
  log p(x_i) and log q(x_i).
- **Backward scan.** Otherwise it scans i = g−1, g−2, …, skipping any position with
  η_i > b_i(1 + slack_i). Those positions can never satisfy η_i ≤ h_i.
- **Candidates.** The remaining positions are evaluated from the top, in batches of
  4, 8, …. Each uses the eager r_i (the same Triton program per row, so a row's r is
  bitwise the same in any batch) and the eager h_i formula. The first hit is τ.
- **Emission.** The emitted token is sampled from the row at τ exactly as before.

The generator calls (one `rand`, one `multinomial`), their order and every comparison
outcome are unchanged. So (τ, y), the log-probs and the generator state afterwards are
bitwise those of `block_verify_logits` given the same logits. The scan neither stops
early at b_i = 1 nor trusts h_i ≤ b_i exactly, because both can fail in floating point:
h = r/((r+1)−b) with tiny r, and Σp > 1 by the fp32 logsumexp rounding. The slack
slack_i = 1e-4 + 2^-20·|lse_p,i| covers that rounding; an adversarial test at
|logits| up to 3000 measured an excess under 2^-20·|lse|.

Tests compare lazy against eager on random, exhaustive and fp32/bf16 GPU cases, checking
equal (τ, y), bitwise logp and equal RNG state. Measured on a shared GPU, the per-draft
cost is about the same, since the two logsumexps dominate. The gain is skipping the
residual pass on the about 98% of drafts that are whole-accepted, and making the
logit-free inputs below possible.

**Logit-free inputs** (`verify_lazy=true` when the scorer supports position selection;
`THUNDERSYNC_LAZY_LOGITS=1` keeps logits and makes only the scan lazy).

- **Capture.** `capture_head_input` stores the LM head's input, the final hidden states
  [T, d], at the selected positions instead of logits.
- **Statistics.** `ChunkedHead` computes per-position (logsumexp, draft-token logit) in
  fixed row chunks of `THUNDERSYNC_LAZY_CHUNK` rows (default 512), using the forward's
  own head call. Under TP it all-gathers one chunk at a time.
- **Pinned rows.** It keeps only the bonus row of every draft.
- **Recompute.** Rows the scan visits are recomputed as part of their own chunk, so
  each is bitwise the row its statistics came from. q rows of version v use stash v's
  head weight.
- **Memory.** Per scorer batch, about 2·T·V·4 B of p+q logits (about 19 GB at 16k
  tokens and V = 152k) becomes T·d·4 B of hidden states per forward, plus one
  [512, V] chunk.
- **What is unchanged.** LM-head FLOPs are the same, and the vocab-parallel
  all-gather is still done, now per chunk.
- **Identity.** Decisions are pathwise those of the eager rule applied to these
  chunk-computed logits. They match the logits path exactly in the unit test (exact
  arithmetic). On real models the chunk GEMM and logsumexp shapes differ from the
  full-batch ones, so expect last-ulp differences and an extremely rare flipped
  decision, not bitwise equality.

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
at equal weights). The output law is linear in the engine's q': out = p + Kᵀ(q' − q),
and with ~98% acceptance K ≈ I, so an accepted draft token x is distributed as about
p(x)·q'(x)/q(x), not p(x).

**Behavior log-probs.** Generation log-probs feed the importance-sampling correction
(`loss_fn.use_importance_sampling_correction`, TIS), so they must describe that law.
Every draft carries the engine's own log q'(x) of each sampled token (`start_drafts`
reads `new_log_probs` from the partial frames and `generated_log_probs` from the final
one; cohort drafts append each resumed segment's log-probs). Block verification returns
log q(x) of each accepted token (`draft_logq`: the scorer's recomputed q from the
stashed weights, same vocabulary slice as the acceptance ratios), and `_plan` reports
(`behavior_logprobs`):

| Token | Reported generation log-prob |
|---|---|
| accepted draft token | log p + log q' − log q |
| verifier's bonus token (whole draft accepted) | log p: sampled exactly from p |
| verifier's residual token (after a rejection) | log p: sampled from norm(max(b·p − q, 0)), which never involves q'; the q'-dependence of the BV output law has no per-token closed form there, and these tokens are ~1–2% of drafts |
| engine continuation | the engine's log-prob, as in normal generation |

If a draft has no engine log-probs (keyed mode, non-streamed `draft`, or a segment whose
frames lacked them; one missing segment disables it for the whole cohort draft) or the
verifier returned no `draft_logq`, log p is reported as before. The engine's log-probs
are of the sampling distribution itself: speculation requires temperature 1 and no
top-k/top-p, so raw and processed log-probs coincide. With this, TIS sees the full
train-vs-behavior gap instead of only its train-vs-p_fp32 half (G2 in
`thundersync_work/research/estimator_theory.md`). GPU check (1.5B, 30 steps,
`THUNDERSYNC_MISMATCH_CHECK`): the corrected log-probs are applied (tokens bitwise equal
to training's log-probs 18% → 35%), while the mean distance to training's log-probs is
unchanged within noise (3.0e-3 vs 3.5e-3; KL 7.6e-5 vs 8.1e-5) — expected, since the check
measures train-vs-reported, not reported-vs-true-law. Batch-invariant kernels with
training-precision verification (`verify_precision=model`) make q bitwise the sampling
distribution at lower acceptance (section 14); keyed mode avoids the issue too.

**Configuration guards.** Verification compares raw logits and drafts ignore stop
strings, so `validate_config` rejects speculation with temperature ≠ 1, top-p < 1, top-k,
or stop strings. Streaming computes advantages per group as each group closes, so it
requires `grpo.adv_estimator.name=grpo` and no `seq_logprob_error_in_loss`. A streamed
draft stopped by `draft_budget` below `max_new_tokens` is not a finished rollout.
Inference-side verification
requires inference TP = PP = 1 (the scorer and the running engine would share
communicators); keyed verification requires learner TP = PP = 1. Streaming rejects reward
penalties and per-sample advantage overrides, and MoE settings that are not
advantage-weighted or depend on micro-batch composition (load-balancing or z-loss, expert-
bias updates, capacity token dropping). Generation workers stay in eval mode after
verification (the engine shares the module); the learner restores its training mode.

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
| Same, rerun in the lookahead session (100 steps) | 3.01 | 2.55 | 5.19 | 99.4% | 97.4% |
| **Lookahead 4 (100 steps)** | **2.86** | **2.54** | **4.18** | **99.4%** | **98.5%** |
| Lookahead 5 (100 steps) | 2.98 | 2.53 | 4.35 | 99.2% | 98.7% |
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
| Start drafting at the start of the iteration instead of at the first finished rollout | 3.17 vs 3.01 s (100 steps); drafts compete with the iteration's own rollouts |
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

### 10.1 Exact acceptance under batch invariance (bf16)

With BI and engine-sourced q, verification compares the true bf16 laws of π_n and π_{n+1}.
Measured in the real system (Qwen3-1.7B): 24–52% of draft tokens kept (control run: mean 45%, 15–25% of drafts whole), vs 99.4% with the
earlier (inexact) fp32 scorer. With that acceptance, engine-q speculation on a
generation-saturated run (MAXLEN 4096) took a mean 92.8 s/step (81–120 s; 13 of 30 steps completed before a timeout) vs 29.5 s without speculation
(`thundersync_work/runs_perf`).

Offline ablation (Qwen2.5-Math-1.5B, steady-state masters, one AdamW step, 2048-token
drafts; full table in `thundersync_work/research/ACCEPTANCE_LEDGER.md`): bf16 weights with
fp32 compute keep 96.6% (whole drafts 95.3%), all-bf16 62–68%, so the weights (and their 17x
amplified bf16 update) are not the cause; rounding activations to bf16 is. No single site
dominates: fp32 residual stream 73.9%; + fp32 logits 78.3%; + fp32 norms/softmax/LM head 82.0%;
+ every GEMM emitting fp32 86.8%; + fp32 inputs to attention-out / MLP-down 90.1%. LM head
alone in fp32: 69.7%. Hysteresis rounding of the weights (prototyped, removed):
+4 pts only. Next: the same placements in Megatron (training and inference identically, BI
preserved), starting with `fp32_residual_connection` (now passed through; BI requires it to
match on both sides).

## 11. Scaling status

| Piece | Status |
|---|---|
| Drafting, streaming, abort | Megatron inference client and coordinator; inherits inference TP/PP/EP/DP |
| q storage | bf16 weight stash per pending version; memory O(parameters + batch x vocabulary) |
| Scorer memory | fp32 adds 4 bytes per parameter per shard; `verify_precision=model` keeps training dtypes (about 90% kept vs 99%); under BI, `verify_source=engine` needs no scorer and only an LM-head stash per version (section 14) |
| Tensor parallelism | Vocab-parallel logits gathered per batch; validated at TP=2 (99.9% kept) |
| Pipeline parallelism (training) | Streamed chunks of a dispatch run as one pipeline call per gradient destination (one per dispatch in group mode); `min_pipeline_microbatches` holds rows of closed groups to fill the pipeline (section 2). CPU-tested; PP=2 GPU check written, not run. Interleaved PP not implemented for streaming |
| Pipeline parallelism (verification) | Forward-only pipeline schedule, last stage verifies; scorer log-probs at PP=2 bitwise equal PP=1. Inference must be pinned to PP=1 (`mcore_generation_config.pipeline_model_parallel_size=1`) for inference-side scorers; under BI `verify_source=engine` runs PP>1 on both sides (measured exact at training PP=2 / inference PP=2, section 14b) |
| Context parallelism | Not implemented; the scorer raises. Needs packed (THD) scoring |
| MoE / expert parallelism | Streaming exact on GPU with EP=2 (group granularity; tiny random MoE); verification on the learner in EP lockstep (inference-side verification rejected under EP); under BI `verify_source=engine` at EP>1 pads training forwards to the EP group's count (unit-tested, section 14); no real-model run yet |
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
| `verify_lazy` | Lazy, logit-free block verification (section 4); default `false` until GPU-validated |
| `verify_source` | `scorer` (default) or `engine`: under BI, q from the engine's log-probs and logged hidden states with an LM-head stash, p from the training model (section 14) |
| `verify_on` | `learner`, `inference` or `both` worker groups run verification |
| `verify_chunks` | Verification pieces; plans publish as each finishes |
| `verify_batch_tokens` | Tokens per scorer forward batch (bounds logits memory), default 16384 |
| `draft_budget` | Maximum new tokens per draft |
| `merge_pipeline_calls` | One pipeline schedule call per dispatch and gradient destination (default `true`; section 2) |
| `sticky_routing` / `sticky_routing_delta` | MoE: sticky routing anchored to the drafter (section 14c); off by default, δ 0.03 router-logit units |
| `null_bv` / `null_bv_drafts_per_output` / `null_bv_logratio_var` / `null_bv_fixed_m` | Pooled null-augmented block verification (section 14d): K·G drafts per group, first G non-⊥ outputs in drafting order; off by default, K 1.5, DP envelope at v 8e-7 unless a fixed M is given |
| `sticky_continuation_chunk_tokens` | Sticky routing: continuations are verified every this many decoded tokens while they decode (2048) |
| `min_pipeline_microbatches` | Hold closed-group rows until this many micro-batches per learner rank are queued; flushed when rollouts end (`null`: 2·PP if PP > 1, else 0) |

Debug switches: `THUNDERSYNC_SPEC_PROF` (verification timing), `THUNDERSYNC_SPEC_CHECK`
(cross-pool scorer check), `THUNDERSYNC_SELECT_CHECK` (selective vs full logits, PP=1),
`THUNDERSYNC_SPEC_TORCHPROF` (scorer profile, PP=1), `THUNDERSYNC_REJECT_LOG` (per-draft
outcomes and continuation timing), `THUNDERSYNC_LEARNER_PROF` and
`THUNDERSYNC_LEARNER_TORCHPROF` (training chunks), `THUNDERSYNC_SCORER_ATTN_BF16`
(experiment: bf16 attention in the scorer), `THUNDERSYNC_SPEC_SELFTEST` (keyed mode:
re-draft with the same seeds and verify on unchanged weights before training),
`THUNDERSYNC_MISMATCH_CHECK` (rollout vs learner log-probs before each optimizer step),
`THUNDERSYNC_LAZY_LOGITS` (with `verify_lazy`: keep logits, lazy scan only),
`THUNDERSYNC_LAZY_CHUNK` (logit-free row chunk, default 512), `THUNDERSYNC_LAZY_CHECK` (with
`verify_lazy`: rerun the eager logits path on the same seed and print decision mismatches),
`THUNDERSYNC_ENGINE_Q_CHECK` (with `verify_source=engine`: rebuild every q row from the logged
hidden states and print how many drafted tokens' log q match the engine bitwise).

## 13. Code map

| File | Role |
|---|---|
| `thundersync_rl/grpo_loop.py` | GRPO loop: per-sample rollouts, learner loop, drafter and verification tasks, config, cohort ramp |
| `thundersync_rl/streaming.py` | StreamPlanner: arrivals to dispatches, bucket vs final chunks, zero-advantage chunks, DP balancing, pipeline hold (`min_pipeline_microbatches`), bubble model |
| `scripts/pp_streaming_sim.py` | Step-time model of streaming under PP (real planner, no GPU) |
| `thundersync_rl/speculative.py` | SpeculativeGeneration: cohorts and segments, stash calls, chunked verification, plans, continuations |
| `thundersync_rl/worker.py` | Megatron worker extension: streaming train calls, scorer build and refresh, weight stashes, stashed block verification, draft streams |
| `thundersync_rl/block_verification.py` | Reference block verification, shared acceptance helpers, exact output law |
| `thundersync_rl/block_verification_fused.py` | Block verification from raw logits with a Triton residual-mass kernel; lazy variant from per-position statistics |
| `thundersync_rl/null_bv.py` | Pooled ⊥-BV: envelope policy (DP over drafts/slots left, Brownian tail-decode model), per-prompt slot state (section 14d) |
| `thundersync_rl/verify0_stream.py` | Risk-first streaming verify0: past-only risk model, call schedule over whole prompts, growing canonical order, tail feed (section 14z) |
| `thundersync_rl/lanes.py` | Cohort lanes: lane roles, the cohort pinning invariant, lane timeline, lane builder and per-lane refit (section 14j) |
| `thundersync_rl/routes.py` | Sticky routing: engine route log, object-store route blocks (`RouteRefs`), per-stage fetch (section 14c); rule in `nemo_rl/models/megatron/sticky_routing.py` |
| `thundersync_rl/engine_q.py` | Engine-sourced q under BI: engine hidden-state log, per-step row mapping, verification from engine log q + rows on demand |
| `thundersync_rl/logit_free.py` | `ChunkedHead`: LM-head statistics from hidden states in fixed row chunks, rows recomputed on demand |
| `thundersync_rl/scorer_kernels.py` | SDPA core attention, Triton SwiGLU, selective LM head |
| `thundersync_rl/keyed_sampling.py` | Position-keyed Gumbel-max sampling kernel (keyed mode) |
| `thundersync_rl/fused_train.py` | Fused verification + training (BI, experimental): retained graphs, logit stitching, forward substitution |
| `nemo_rl/algorithms/grad_streaming.py` | Core: per-(group, reward) gradient accumulators, final-first selection (shared with the single controller) |

Tests: 252 research unit tests (block verification by enumeration, fused verification
draw-for-draw, scorer kernels on GPU, streaming equivalence, speculative driver plans,
cohorts and ramp, chunked pools, error surfacing, fused verification + training: routing,
claims, config checks and gradient equality on a tiny causal model; engine-sourced q: hidden
log and engine hook on fakes, rule vs the eager rule, worker path on GPU with bitwise log q
reconstruction, head-GEMM row invariance at Qwen3 shapes, driver plumbing and config checks), 7 core zero-advantage tests, and GPU
functional checks for scorer and refit parallelism.

## 14. Batch invariance (NeMo-RL PR 3208)

Branch `thundersync-rl-bi` applies PR 3208 (`policy.megatron_cfg.zero_train_gen_mismatch`):
Megatron training and Megatron inference run the same batch-invariant kernel stack, so
generation log-probs and training log-probs of the same tokens are bitwise equal.

**How the kernels get batch invariance** (`te_native` backend): native cuBLASLt GEMMs with
a starved workspace (no split-K, so a fixed K-reduction order at every row count),
batch-invariant `aten::_log_softmax` / `mean`, TE RMSNorm kept native under 64-multiple
token alignment, FlashAttention 4 with `num_splits=1`. Requirements enforced by the PR's
validator: BF16, CP=1, no packing, generation `transformer_impl=inference_optimized`
(so no QKV-bias models such as Qwen2.5), `make_sequence_length_divisible_by` a multiple of 64,
and **identical parallelism in training and inference** (next paragraph). The PR shipped with
a TP=1 gate; branch `thundersync-bi-tp2` replaces it.

**Parallelism under BI: any TP, same layout on both sides** (`zero_train_gen_mismatch.py`;
design notes in `thundersync_work/research/bi_tensor_parallel.md`). Rules, checked by
`validate_batch_invariant_mode`:

- training TP, EP and expert TP equal the merged generation config's (PP may differ) (an unset expert TP
  resolves to TP; generation pins expert TP to 1, so training needs 1 too). A different TP
  changes the K split of the row-parallel GEMMs, which no kernel choice undoes;
- at TP>1, `sequence_parallel=true` on both sides (`inference_optimized` layers require it;
  training must run the same reduce-scatter / all-gather and the same per-rank norm row
  counts), and `make_sequence_length_divisible_by` (and the dynamic-batching round) a
  multiple of lcm(64, 32*TP), so each rank's tokens/TP norm rows stay a multiple of 32
  (64 at TP=2, 128 at TP=4, 256 at TP=8);
- `inference_fuse_tp_communication=false` and `tp_comm_overlap=false` (at any TP; the
  generation worker also refuses fused TP communication at build time). Under BI the fused
  reduce-scatter falls back to NCCL, but the next layer would still skip its norm + all-gather
  and read a stale symmetric buffer.

Log-probs at TP>1: training's `DistributedLogprob` normalized with MAX/SUM all-reduces of
partial statistics, while inference all-gathers the logits and runs one full-vocabulary
`log_softmax`. Under BI, `DistributedLogprob` / `ChunkedDistributedLogprob` are replaced by
`_BatchInvariantLogprob`: TP=1 keeps the local path; TP>1 runs `_GatheredVocabLogprob`, which
all-gathers the vocab shards per sequence chunk (at most 4096 tokens per chunk unless
`logprob_chunk_size` is set), runs the TP=1 path on them (bitwise inference's), saves only the
local shard, and in backward re-gathers per chunk and returns g * (onehot - softmax) on the
local vocab slice. Unit tests (CPU, TP ranks as threads): forward bitwise equal to the
full-vocabulary log-softmax at TP 2/3/4; backward equal to the full-vocabulary gradient on each
shard (to 1 ulp; only the forward must be bitwise).

Collectives: a 2-rank reduction is one commutative IEEE addition, so TP=2 reduce-scatter and
all-reduce give a token the same bits at every batch size whatever NCCL algorithm runs. At
TP>=3 the order of a token's partial sums may depend on which rank owns the token, i.e. on
the token count. `scripts/bi_tp/nccl_tp_invariance.py` measures it (TP 2/3/4, reduce-scatter
and all-reduce, plus an ordered all-to-all + fixed-rank-order fp32 sum as the reference fix).
Measured end to end (2026-10-12, `tests/functional/check_bi_tp_logprobs.py`): the
batch-invariant training forward of Qwen3-1.7B at TP=4 (sequence parallel), stock NCCL, gives
bitwise identical log-probs for 16 sequences whether each sits in one micro-batch layout or
another (different micro-batch, row, padded length and companions): 0/16 differ. The earlier
collective-level probe (synthetic token counts) does not reflect what the model runs, and no
ordered-reduction workaround is kept. Remaining check: training TP=4 against the engine at TP=4
(2 nodes, `tp_parity_submit.sh`).

ThunderSync under BI at TP>1: the learner's bf16 (`verify_precision=model`) scorer
all-gathers its vocab-parallel logits before the log-softmax (same bits as training's
gathered path) and pads its rows to `make_sequence_length_divisible_by` (sequence
parallelism needs a multiple of TP; the norm row class needs training's multiple); fused
verification + training still requires TP=1 (it reads p from the training forward's logits,
which are vocab-sharded at TP>1); engine-sourced q works at any TP (below, "Engine-sourced q
at TP>1"); keyed verification still requires learner TP=PP=1 (it calls
the training model directly); scorers on the inference workers still require
`verify_on=learner` at inference TP>1 (shared engine communicators). The scorer's padding
applies to full-row forwards only and its output is sliced back to the real length only when
it padded (the logit-free path's selected-position logits, `[1, n_selected, V]`, are never
padded, since batch invariance always scores full rows).

**Environment.** TE 2.18 (built from source against torch 2.11), flash-attn-4 4.0.0b19, CuTe
DSL 4.6.2 (including `nvidia-cutlass-dsl-libs-core`, the Python frontend), quack-kernels 0.6.4
and apache-tvm-ffi 0.1.11, as a PYTHONPATH overlay (`thundersync_work/build_bi_overlay.sh`).

**Measured with ThunderSync** (`THUNDERSYNC_MISMATCH_CHECK=1`: before the optimizer step the
learner recomputes the batch's log-probs and compares them with the rollout log-probs):

| Model | Mismatch |
|---|---|
| Qwen3-1.7B, BI | 0 on every token (4 steps) |
| Qwen3-1.7B, BI, TP=2 (train and inference) | 0 on every token (4 steps) |
| Qwen3-1.7B, BI, training PP=2, inference PP=1 | 0 on every token (2 steps) |
| Qwen3-1.7B, BI, training PP=2, inference PP=2 | 0 on every token (4 steps, fp16 GEMM forward) after the block-scope CUDA graph fix (section 14b); before it ~43-nat gap |
| Nemotron-Nano-9B-v2 (hybrid Mamba), BI, validator's hybrid exclusion relaxed | 0 on every token (3 steps) |

The PR's exclusion of hybrid Mamba models is an untested-scope limit: MCore's
batch-invariant Mamba decode (buffered chunk replay) already makes them exact.

**Mamba SSD kernels: train == engine by construction** (`nemo_rl/models/megatron/mamba_ssd_parity.py`,
installed by `enable_batch_invariant_kernels`). The Nano-9B row above held because the two
libraries' Triton autotuning happened to pick the same tiles: training ran mamba_ssm's fused
split-conv1d-scan, the engine runs MCore's copy of the SSD kernels, and with each library
pinned on its own (MAMBA_DETERMINISTIC=1) 2-3% of the outputs differ [measured, one Nano
layer; thundersync_work/research/agent_nemotron_hybrid.md section 6]. The dA cumsum head
tiling changes the bits at bf16 and fp32, the chunk-scan warp count at fp32 [measured, GB200].
In BI mode the learner's `MambaMixer._ssm_training` now runs the conv, then the engine's
varlen SSD forward (gate inside the scan, fp32 chunk states) with mamba_ssm's backward, and
the engine's SSD forward kernels and varlen conv are pinned to constant configs in every
process (`PINNED_CONFIGS`, built explicitly so an import-time candidate list cannot change
them). Same change upstream: Megatron-LM branch `fix/mamba-ssd-train-inference-parity`
(draft PR text: thundersync_work/research/pr_mamba_parity.md); this Megatron-Core cannot
store the pre-gate output, so the backward recomputes it (one extra SSD forward per layer).
Cost of the pinned configs vs Triton autotune on the SSD forward: -10..+17% (bf16), +4..+12%
(fp32) at Nano/Super/Ultra-scale per-rank shapes [measured, GB200; PR text]. Test:
`tests/unit/models/megatron/test_mamba_ssd_parity.py` (mixer training == engine prefill +
decode bitwise, bf16 and fp32 inputs; control: unpatched training on another cumsum tiling
differs).

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
- *Where the scorer runs.* On the learner it is stable but does not scale under BI at TP=1
  (the only BI layout before `thundersync-bi-tp2`): each learner GPU holds the full fp32 copy (4 B/param) plus a bf16 stash
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
`batch_invariant_mode`, TP=1 and PP=1 (and BI's CP=1), generation temperature 1, no fused
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
training forwards are expensive relative to verification (large models, learner-bound runs). On
Qwen3-8B (1 node, 2 + 2 GPUs) the default settings run out of memory on the learner (weight
stashes, retained graphs and training state on TP=1 GPUs); it fits with
`draft_lookahead=1 verify_batch_tokens=4096 fused_max_retained_tokens=8192`, keeping 66-68%
of draft tokens, but only about one trajectory per step finishes whole and gets fused.

**Engine-sourced q** (`thundersync.verify_source=engine`, `engine_q.py`). *Opt-in. Exact end
to end on GPU at TP=1 and at training PP=2 / inference PP=1 (Qwen3-1.7B, BI, 4 steps each: every
`[engine q check]` exact, train/generation mismatch 0, nothing dropped, 35-48% kept); at TP=2
the train/generation mismatch is 0 and 7 of 9 checks were exact before the probe-column source
check (below) was added. Earlier at TP=1: every `[engine q check]` exact, train/generation
mismatch 0, no dropped drafts, 40% of draft tokens kept — the exact-bf16 rate); TP 2, EP 2
and PP 2 are unit-tested (fake process groups), GPU runs in progress.*

Getting the engine-side capture right on a real engine took four fixes, each found on GPU:
(1) decode steps replay CUDA graphs, which skip Python hooks, so the LM-head pre-hook only
copies rows into a static buffer, and the copy must be registered *before* the graphs are
recaptured; (2) rows are keyed by token index relative to the request's original prompt
(an evicted request resumes as a new record whose generated tokens restart at 0); (3) in
~2% of decode graph replays one row of the buffer is stale — every logged row is checked at
the source against the logit of the token the engine sampled from it (one dot product per
row) and stale rows are not recorded; the draft is truncated at its first missing row (any
prefix of a draft is a valid draft); (4) **Megatron's streamed partial frames skip tokens
after an eviction** (`DynamicInferenceEngine._try_send_streaming_partials` keeps an emitted-
length counter per request id that survives the record reset), so unfinished drafts
assembled from frames were not the sequences the engine generated — in every verification
mode, not only engine-q. Draft content is now read from each engine's merged request record
(`record.merge()`, as the engine's own final reply does) when the drafts are collected.

The
default verifier needs an fp32 copy of the model (4 B/param) plus a bf16 whole-model stash
per pending version (2 B/param each): about 82 GB on each learner GPU at 8B with three
pending versions at TP=1, impossible at 70B. Under BI neither is needed. The engine's decode
is bitwise the training forward, so the drafts' true law q is the engine's own
distribution, and p can be the training model's forward at θ_{k+1}:

- **log q(x_i)** of every drafted token is the engine's own log-prob, already streamed with
  the draft (G2, draft tuple index 6).
- **p** is the training model's bf16 forward of the row, run like a training micro-batch
  (`_training_forward`, no autograd, eval mode), turned into log-probs with training's fp32
  log-softmax.
- **Lazy scan.** Block verification needs only b_i = min(1, b_{i−1}·p(x_i)/q(x_i)), which
  uses only log p(x_i) and log q(x_i), plus full q rows where the lazy scan computes r_i
  (η_i ≤ b_i(1 + slack), visited from the top) and the residual row at τ. Whole-accepted
  drafts (η_g ≤ b_g) need no q row at all.
- **Rows on demand from hidden states.** Every engine forward runs a pre-hook on the
  engine model's output layer (`install_draft_hidden_log`) that copies the head input rows
  into a static buffer (captured into CUDA-graph decode, which skips Python hooks); a
  wrapper around the engine's sampler, which runs in Python after every forward, logs
  them. For each draft request (tagged
  through a sampling-params attribute) it copies the row of its last input token to pinned
  host memory: the final hidden state after the final norm, from which that token was
  sampled (`HiddenLog`, d × 2 B per drafted token). Rows are keyed by (draft, generated
  index t = last position + 1 − prompt length), so an intermediate prefill chunk records
  nothing and an engine eviction's re-prefill overwrites the same t. At the deadline the
  engine that ran each draft puts its rows into the Ray object store and the draft carries
  only handles (tuple index 7, `HiddenRefs`; see "Hidden-state data movement" below). The
  learner keeps only the LM-head weight per pending version (`stash_head`, V × d × 2 B) and
  rebuilds a q row as `log_softmax(output_layer(h, weight=W_head^(v)))`. It uses the
  training model's own output layer and logit scaling, and the version v that drafted that
  position (segments).
- **Why the rows are exact.** The hidden states are bitwise the training forward's under
  BI. The head GEMM under BI's cuBLAS settings (no split-K) gives a row the same logits at
  any row count. `test_head_rows_are_batch_invariant_under_bi_cublas_settings` checks this
  for the Qwen3-1.7B and Qwen3-8B head shapes at 1-600 rows (bitwise). In a run,
  `THUNDERSYNC_ENGINE_Q_CHECK=1` rebuilds every q row and prints how many drafted tokens'
  log q equal the engine's bitwise (expected: all, max |dlogq| = 0). If they ever differ,
  b and the rows would come from slightly different q's: an ulp-level inconsistency of the
  same kind as the scorer's today. The fallback would then be to rebuild log q(x_i) from
  the rows too, at one head GEMM per drafted token.
- **Decisions.** `verify_engine_q` feeds log-prob rows (logsumexp 0) to
  `block_verify_stats_lazy`. It makes the same `rand(g+1)` and `multinomial` calls as
  `block_verify_logits`, and unit tests check the same decisions and generator state as the
  eager rule on the same rows.
- **Reported log-probs.** The plan reports training's own log p of the kept tokens (entries
  of the training log-softmax rows) and `draft_logq` = the engine's log q. G2's correction
  log q' − log q is therefore exactly zero; `behavior_logprobs` skips equal pairs, so log p
  is reported bitwise. With BI's 0-mismatch engine log-probs, `THUNDERSYNC_MISMATCH_CHECK`
  should read 0 on every token. Importance-sampling correction is unnecessary in this mode.
- **Exactness.** q is the drafts' true sampling law, p is the training policy at θ_{k+1},
  and the rule is block verification. The rollouts are therefore exact samples of the
  policy being trained, with no fp32 approximation. Acceptance is that of exact bf16
  speculation, the same as `verify_precision=model` with a training-exact scorer (about 50%
  on long outputs; see the offline measurement below).
- **Fallbacks.** If the engine cannot log hidden states, `SpeculativeGeneration` reverts to
  `verify_source=scorer` with a printed warning. That happens with no engine, MTP
  speculative decoding inside the engine, or a sequence-sharded head input
  (`materialize_only_last_token_logits=false` with sequence parallelism; the validator
  rejects it at TP>1). A single draft
  without full engine log-probs or hidden rows is dropped before verification and its
  rollout decodes from scratch (`spec/engine_q_dropped`). Dropping is exact because
  whether a draft was logged does not depend on its tokens. A hook that never fires (for
  example under a whole-model CUDA graph, which replays without Python) shows up as every
  draft dropped. No path re-runs a forward of a stashed version.

**Engine-sourced q at TP>1** (branch `thundersync-bi-scalable`: `thundersync-bi-engine-q`
plus `thundersync-bi-tp2`). BI requires the same TP, PP, EP and sequence parallelism in
training and generation, so the learner's vocab-parallel LM-head shard on TP rank r is the
engine's shard on its rank r (same vocabulary split, rank order; tied embeddings use the
embedding's shard, same split). How each part scales:

- *Head stash.* `stash_head` clones the live head weight, which at TP>1 is already the rank's
  shard: V/TP x d per pending version on each GPU (`head_stash_bytes`, logged with
  `THUNDERSYNC_MEM_LOG=1`).
- *Engine hidden states.* With `materialize_only_last_token_logits=true` (the default, and
  required at TP>1 by the validator), MCore's `GPTModel._postprocess` all-gathers the
  sequence-parallel hidden states, keeps each request's last row and calls the output layer
  with its sequence parallelism switched off. The pre-hook therefore sees the full rows,
  identical on every TP rank, and TP rank 0 alone logs them: one device-to-host copy per
  forward and no extra collective (other ranks return None from `install_draft_hidden_log`).
- *How the engine computes q at TP>1.* `ColumnParallelLinear` computes the local logits
  shard `h W_r^T` (`[n, V/TP]`), all-gathers it along the vocabulary (NCCL
  `all_gather_into_tensor` then a rank-order concatenation, or the NVLS multimem gather,
  both pure copies), applies the logit scaling, and the controller's `raw_logprobs` path
  runs `log_softmax(logits.float())` over the full padded vocabulary.
- *q rows on demand* (`_head_logprobs`, `gathered_log_softmax`). Each rank runs the training
  model's output layer on the logged rows with its stashed shard (sequence parallelism
  switched off for the call, `runtime_gather_output=False`), applies the logit scaling,
  all-gathers the shards in rank order (`_all_gather_vocab`, the same helper as
  `_GatheredVocabLogprob`) and runs the same fp32 `log_softmax`. Per element this is the
  engine's computation: the local GEMM has the same shard shape, and under BI's cuBLAS
  settings a row's logits do not depend on the row count.
- *p at TP>1.* The training forward's logits are vocab-parallel. `GatheredLogprobRows` keeps
  the local shard (`[g+1, V/TP]`), computes the drafted tokens' log p in chunks of up to
  4096 rows (gather, fp32 log-softmax, pick), and gathers full rows only where the lazy scan
  asks for them. This is `_GatheredVocabLogprob`'s forward, so log p is bitwise what
  training uses. The full `[g+1, V]` fp32 matrix is never materialized (TP=1 keeps the dense
  path).
- *Identical decisions on every TP rank.* All ranks of a replica receive the same rows,
  hidden states and engine log-probs. They select the same rows (`_my_rows` hashes by DP
  rank), seed the generator with `seed * 1_000_003 + DP rank`, and see bitwise identical
  gathered p and q rows. Their lazy scans therefore make the same `rand`/`multinomial` calls
  and issue the same gathers in the same order, and every rank returns the same plan. A
  divergence would deadlock the gathers; the unit tests assert lockstep.
- *Tests* (`test_engine_q.py`; TP ranks are threads sharing a fake rank-order all_gather):
  q rows from sharded heads equal the full-vocabulary row bitwise (TP 1/2/4, bf16 and fp32);
  gathered p (chunked and on demand) equals the dense log-softmax; the engine rule on every
  fake rank matches the TP=1 rule on dense rows (decisions, log p, generator state,
  identical gather sequences); the worker's `verify_drafts_block_engine` at TP 2/4 returns
  exactly the TP=1 worker's results, with every rank's `[engine q check]` exact (GPU, a few
  seconds); `stash_head` keeps 1/TP of the head per rank; the hidden-log install rules per TP
  rank; and the validator accepts any TP with the same layout and rejects a train/generation
  TP mismatch and non-materialized logits at TP>1.
- *Not covered:* CP (1 under BI anyway) and virtual (interleaved) pipeline stages (not
  implemented; the validator says so).

**Engine-sourced q under EP and PP** (branch `thundersync-scalable-ep-pp`, for large MoE
models). Status: *unit-tested* (CPU, fake groups); GPU launchers written, not yet run.
BI requires the same PP and EP on both sides, so the layouts match as for TP.

- *Expert parallelism.* p comes from the training model at the current weights only, and q
  rows only from the dense LM head (no experts). So the one collective that spans
  data-parallel replicas is the expert all-to-all inside each training forward, and the
  only requirement is an equal number of training forwards per verification call across
  the EP group (an unequal count hangs the all-to-all, as seen for the scorer path, section
  2). The EP ranks agree on the largest per-rank row count (`_expert_parallel_lockstep` with
  empty version plans: q needs no forward) and short ranks run one-token dummy forwards
  after their real rows (`_dummy_training_forward`: no autograd, eval mode, padded to
  `make_sequence_length_divisible_by`, a multiple of TP, so sequence parallelism splits it
  evenly; output discarded). Logged as `ep_dummy_forwards` in `[spec prof]`.
- *Lazy-scan gathers under EP.* The q-row and p-row vocab gathers use the dense TP group,
  whose ranks belong to one replica (same rows, same seed, same decisions, same order), so
  their collective sequences match without cross-replica lockstep. With EP and TP groups
  overlapping (MCore carves EP out of DP x TP), the interleaving is still consistent: every
  rank runs [forward (EP all-to-all), its replica's gathers] per row, then its dummies.
  Tested with EP=2 x TP=2 (x PP=2) thread ranks and barriers for the all-to-all.
- *Dropless MoE only.* With an expert capacity factor, whether a token reaches its experts
  depends on every other token routed to them in that forward (other replicas' rows, the
  dummies), so p of a row is not a function of the row and weights: provably inexact, so
  the worker raises (`_check_dropless_moe`). Megatron's default is dropless.
- *Pipeline parallelism (learner).* `_training_forward` runs the row through Megatron's
  pipeline schedule (`get_forward_backward_func`, `forward_only=True`,
  `collect_non_loss_data=True`, one micro-batch of one row, `seq_length` = padded length, as
  training passes it), each stage calling the same `model_forward`. The DDP `no_sync` hook
  is bypassed for the call (the forward-only schedule enters it and never leaves it). The
  last stage receives the logits (vocab-parallel at TP>1) and runs the lazy scan, q rows and
  `THUNDERSYNC_ENGINE_Q_CHECK`; it alone holds the LM head, so `stash_head` stashes nothing
  on other stages. Results come back from the last stage only: other stages return [] and
  the driver already merges every rank's (row, result) pairs, as for the scorer under PP.
  All stages of a replica run the same rows in the same order (the schedule's send/receive
  needs it); stage 0's next forward overlaps the last stage's scan. Autograd through the
  schedule (fused mode) is not implemented.
- *Pipeline parallelism (engine).* The output layer exists only on the engine's last stage,
  which computes the logits and broadcasts them to the other stages for sampling. The
  hidden-state pre-hook and sampler wrapper (`install_draft_hidden_log`) are installed there
  (other stages return None); the dynamic engine broadcasts each request with its
  serialized sampling params to every model-parallel rank, so the last stage sees the draft
  tags. Measured exact at inference PP=2 (section 14b, `engine_q_pp.sh` with `GEN_PP=2`).
- *Tests* (`test_engine_q_ep_pp.py`, CPU; ranks are threads, process groups fakes, the
  expert all-to-all and the pipeline send/receive modeled as barriers per forward): for
  (TP, PP, EP) in {(1,1,2), (2,1,2), (1,2,1), (2,2,1), (1,2,2), (2,2,2)} with replicas owning
  9 and 3 rows, every rank of an EP group runs the same forward count (dummies one padded
  token, after the real rows), last stages return exactly the TP=PP=EP=1 results (decisions,
  log p, log q), other stages [], stages of a replica run identical row sequences, and
  vocab gathers occur only on last stages, identically per TP rank; a control without the
  lockstep breaks the fake all-to-all. `_training_forward` at PP=2 calls the schedule
  forward-only with one micro-batch of the padded length, returns the PP=1 logits on the
  last stage and None elsewhere, restores `no_sync_func`, and refuses autograd;
  `stash_head` is a no-op off the last stage; the hidden log and sampler wrapper install on
  the engine's last stage only; capacity-factor MoE is rejected; the validator accepts
  PP/EP>1 with matching generation layout (warning "untested end to end") and rejects a
  train/generation PP or EP mismatch and virtual pipeline stages.

Memory, per learner GPU, three pending versions, ~1.5M drafted tokens per step (128 rollouts
x 3 cohorts x 4k):

| | scorer mode (fp32 scorer + bf16 stashes) | engine mode (LM-head stashes) |
|---|---|---|
| Qwen3-8B (d = 4096, V = 151,936), TP=1 | 33 GB + 3 x 16 GB = 82 GB | 3 x 1.24 GB = 3.7 GB |
| Qwen3-8B, TP=2 | 16.5 GB + 3 x 8 GB = 40.5 GB | 3 x 0.62 GB = 1.9 GB |
| 70B (Llama-3, d = 8192, V = 128,256), TP=1 | 282 GB + 3 x 141 GB (does not fit) | 3 x 2.1 GB = 6.3 GB |
| 70B, TP=8 (V padded to 129,024) | 35 GB + 3 x 17.6 GB = 88 GB | 3 x 0.26 GB = 0.79 GB |
| Hidden states (host memory, not GPU; logged once per replica, on TP rank 0) | none | 8 KB/token at 8B (12.6 GB per 1.5M tokens), 16 KB/token at 70B (25 GB) |

The head stash is V/TP x d x 2 B per version per GPU: 2.1 GB / 8 ≈ 0.26 GB at 70B and TP=8.
Transient GPU memory per verified row at TP=1 is the training forward's logits plus fp32
log-probs, (g+1) x V x 4 B each (about 2.5 GB each for a 4k-token draft at V = 152k), and a
few q rows. At TP>1 it is the local logits shard, (g+1) x V/TP x 4 B, plus one chunk of at
most 4096 gathered fp32 rows. Only the hidden-state rows the scan visits go to the GPU (on
every TP rank).

**Hidden-state data movement** (`thundersync.hidden_transport`, default `object_store`;
`HiddenRefs`/`HiddenRows`/`free_hidden` in `engine_q.py`). A step logs d x 2 B per drafted
token (8 KB at 8B, 14 KB at d = 7168), but only drafts the lazy scan does not accept whole
read any q row, and those read only the rows the scan visits (the candidates
η_i ≤ b_i(1 + slack) from the top down to τ, plus τ). So the rows never pass through the
driver:

1. *Put once, where they were produced.* At the deadline `pop_draft_hidden(as_refs=True)`
   on the generation worker that ran the draft (TP rank 0) puts its rows into the Ray object
   store in blocks of `HIDDEN_BLOCK_ROWS` = 256 rows (2 MB at d = 4096, 3.7 MB at
   d = 7168) and returns `[(ObjectRef, rows, bytes)]`. The driver wraps them in
   `HiddenRefs`; a cohort draft chains the blocks of its drafting rounds (no copy). The
   driver holds handles only (unit-tested: no tensor at index 7).
2. *Phase 1 without rows.* The learner computes p, log p(x_i), b_i and the whole-draft test
   η_g ≤ b_g from the training forward and the engine's log q (shipped with the draft).
   A fully accepted draft never touches its handles.
3. *Phase 2 on demand.* `HiddenRows` fetches a block (`ray.get`, generation node →
   learner node through the object store, never via the driver) the first time the scan
   asks for a row in it, and caches it for that draft. All TP ranks of a replica run the
   same scan, so they fetch the same objects in the same order (immutable, hence identical
   bytes; ranks on one node share the node's copy). Unit-tested at TP=2 with thread ranks.
   No extra driver round trip: phase 2 happens inside the same verification call, in the
   scan's `q_rows` callback.
4. *Freed after verification.* When a verification chunk returns, the driver releases its
   rows' handles and deletes their objects (`free_hidden` → Ray's `internal_api.free`), as
   it does for drafts dropped before verification and for a cohort draft that loses a
   segment. Unverified cohort drafts keep their blocks across steps, as their tokens. An
   explicit free rather than reference counting alone: in one local-Ray probe (fake
   generation/learner actors, before this design was final) the owner still counted a
   local reference after the driver had dropped its own, cause not identified; the explicit
   free makes the lifetime independent of stray Python references. Untested on a real
   cluster; `internal_api.free` is a Ray-internal API.

Bytes per step through the driver: O(drafted tokens) small fields (token ids and the
engine log q, today about 8 + 9 B per token as int64 rows and pickled Python floats; float32
log q would make the latter 4 B), plus a ~100 B handle per 256-row block. Hidden rows
move O(non-whole drafts × visited blocks × 256 × d × 2 B) directly generation → learner.
Metrics: `spec/hidden_bytes_logged` (put this step), `spec/hidden_bytes_fetched` (pulled
by learners, per replica, as reported by each verified draft), `spec/hidden_bytes_freed`,
`spec/hidden_bytes_pending` (held by unverified cohort drafts after this step's drafting:
the host-memory high-water mark of the object stores, on the generation nodes). Estimates,
~1.5M drafted tokens per step (128 rollouts x 3 cohorts x 4k, ~384 drafts), not measured:

| | logged | driver path (`hidden_transport=driver`) | object store path, fetched |
|---|---|---|---|
| 8B (d = 4096) | 12.3 GB | 12.3 GB in to the driver, and 12.3 GB out to every learner node (every learner worker receives every draft's rows) | 5% non-whole drafts, 2 blocks each: ~40 MB (bound: those drafts' full rows, ~0.6 GB); bf16-exact acceptance, ~54% non-whole: ~0.4 GB (bound ~6.6 GB) |
| 500B MoE (d = 7168) | 21.5 GB | 21.5 GB in, 21.5 GB out per learner node | 5%: ~70 MB (bound ~1.1 GB); 54%: ~0.75 GB (bound ~11.6 GB) |

At larger batches the logged bytes grow linearly (e.g. 1024 rollouts x 3 cohorts x 8k
tokens at d = 7168: ~360 GB per step, all of which the driver path would funnel through one
process) while the fetched bytes follow the non-whole drafts. "Visited blocks" is an
estimate: the scan visits few positions near the top and τ, but its candidates can be
spread over the draft; `spec/hidden_bytes_fetched` measures it. Object-store capacity on
the generation nodes must hold `spec/hidden_bytes_pending` plus the step's logged bytes;
Ray spills to disk beyond that (untested here). The `driver` transport stays selectable for
A/B: it moves more bytes but makes no on-demand `ray.get` inside the scan, whose latency per
rejected draft is not measured yet.

Requirements, enforced by `validate_engine_verify_source` and on the worker: `speculate`,
`verify_mode=block`, `verify_on=learner`, `batch_invariant_mode`, generation TP and EP
equal to training's (PP may differ) (the BI validator also checks the full layout), no virtual pipeline
stages (not implemented), dropless MoE (worker), at TP>1
`materialize_only_last_token_logits=true`, no `fused_verify_train`, Megatron generation with
`num_speculative_tokens=0` and `logprobs_mode=raw_logprobs`. Possible extension: combining
with fused mode, whose p forward is the same training forward.

GPU validation to run (BI flags from `thundersync_work/bi_flags.txt`):
`TP=2 MODEL=Qwen/Qwen3-1.7B bash research/thundersync_rl/scripts/bi_tp/engine_q_tp.sh` (2
inference + 2 training GPUs on one node; `MODEL=Qwen/Qwen3-8B` next; `TP=1` for the A/B
baseline). It sets `THUNDERSYNC_ENGINE_Q_CHECK=1 THUNDERSYNC_MISMATCH_CHECK=1`. Pass: the
mismatch check reads 0 on every token of every step, every `[engine q check]` line has
`exact == tok` and `max|dlogq|=0`, and there is no fallback to the scorer. Also expect
`spec/engine_q_dropped` near 0 and the same kept fraction as `verify_precision=model`.
EP and PP (same pass criteria; `engine_q_tp.sh` now takes `TP`, `PP`, `EP`):
`bash research/thundersync_rl/scripts/bi_tp/engine_q_moe.sh` (tiny random Qwen3-MoE,
`thundersync_work/models/tiny-qwen3-moe`, EP=2 on 2 + 2 GPUs; also expect nonzero
`ep_dummy_forwards` on some steps and no hang) and
`bash research/thundersync_rl/scripts/bi_tp/engine_q_pp.sh` (Qwen3-1.7B, PP=2 on 2 + 2 GPUs).

For the hidden-state transport: run the same script with
`ARGS="++thundersync.hidden_transport=driver"` and with the default (same seed); plans (kept fraction,
mismatch, `[engine q check]`) must agree, `spec/hidden_bytes_fetched` should be a small
fraction of `spec/hidden_bytes_logged` under `object_store`, `spec/hidden_bytes_freed`
should track logged bytes with a lag of the pending cohorts, and `ray memory --stats-only`
(or the dashboard's object store usage) on the generation nodes should stay flat across
steps.

**Keyed verification under BI** (`verify_mode=keyed`). With bitwise parity the learner's
logits for a draft are the ones the engine sampled from, so keyed verification needs no
scorer copy and no weight stashes: p from the current training weights is all it reads, and
the check is token-level (keep the agreeing prefix, then the learner's own token). It stays
exactly on-policy, at the lower token-level acceptance of section 4. GPU check (Qwen3-1.7B, BI, 2 + 2
GPUs, `thundersync_work/keyed_bi.sh`): the self-test re-drafts with the same seeds and keeps
100% of every draft on unchanged weights, and the train/generation mismatch check reads 0 on
every token of every step once the plan reports full-vocabulary log-probs (sampling keeps the
`vocab_limit` slice the engine's keyed sampler uses). Acceptance across a weight update is
10-13% of draft tokens, against 92-99% for block verification with the fp32 scorer, so keyed
mode trades nearly all of the speculation gain for removing the scorer and the stashes; it is
the memory-light option, not the fast one. On Qwen3-8B (1 node, 2 + 2 GPUs) it fits on the
learner, keeps 7-10% of draft tokens and takes 66-100 s per step against about 50 s for
block verification on the inference GPUs.

**No importance-sampling correction.** Generation and training log-probs are bitwise equal
under BI for engine-decoded tokens, so their importance weights are exactly 1. That makes
`loss_fn.use_importance_sampling_correction` unnecessary without speculation, and with
keyed or fused verification (both report training's own log-probs for accepted tokens;
measured mismatch 0). With the default fp32-scorer block verification, accepted tokens
carry the fp32 scorer's log-probs (mean |dlogp| about 0.011 against training), so keep
the correction on there.

**Precision of an exact policy (offline measurement).** With q the exact sampling law and
p the same numeric function at the next weights (Qwen2.5-Math-1.5B, one Adam step at lr
1e-6, 128 MATH-500 drafts up to 3072 tokens; `thundersync_work/fp16_acceptance.py`):
bf16 keeps 57.0% of draft tokens (46.3% of drafts whole), fp16 90.0% (85.7%), fp32 96.0%
(94.5%); per-token TV 3.3e-3 / 6.6e-4 / 2.4e-4. bf16's rounding noise between consecutive
versions is the policy's own, so exact bf16 speculation is inherently ~50%; an fp16
(batch-invariant) policy would allow exact speculation without the fp32 scorer or stashes.

## 14b. fp16 GEMM forward (`forward_gemm_precision: fp16`, "V8b")

Exact acceptance is limited by forward rounding noise, not by the verification rule. The
fp16 GEMM forward (`nemo_rl/models/megatron/fp16_gemm.py`) runs the same functions in
training and in the Megatron engine: every linear casts its input and its bf16 weight to
fp16 (exact for |w| in [6.1e-5, 65504]) and emits fp32 from one cuBLAS GEMM (batch
invariant at every M tested, workspace 0); fp32 residual stream (`fp32_residual_connection`,
required); RMSNorm in fp32 on the fp32 residual (batch-invariant row mean, our own backward);
SwiGLU on fp32; q/k/v rounded to fp16 after the q/k norms, so RoPE, FlashAttention (fp16
probabilities), its fp16 output into `linear_proj`, and the engine KV cache are fp16; LM head
`lm_head_precision: fp32` (default): the fp32 final-norm output times the head weight with
every product exact (two fp16 GEMMs on a hi/lo split of the input; IEEE fp32 cuBLAS is not
row-invariant at M = 1-8 and `NVIDIA_TF32_OVERRIDE=1` forces TF32). Weights, checkpoints and
optimizer are unchanged. Backward runs dgrad/wgrad on bf16 casts; FlashAttention's fp16
backward gets its incoming gradient scaled by 2^e (amax to 2^0) and unscaled on q/k/v.
TP/SP: norm outputs all-gathered in fp16, row-parallel outputs reduce-scattered in fp32 (2x
today's bytes). PP: `pipeline_dtype: float32` required (stages exchange the fp32 residual).
The engine-q hidden log holds the head input in the head dtype (fp32: 2x bf16 bytes).

Measured (Qwen3-1.7B, 2048-token drafts, 4 steps, checks on; control: 39-52% kept, 15-25%
whole): every step mismatch 0 / exact_token_frac 1, engine-q checks exact, 0 dropped,
0 truncated.

| layout | kept (steps 1-3) | whole (steps 1-3) |
|---|---|---|
| TP1 | 97.7 / 97.9 / 95.0 | 95.3 / 95.3 / 93.8 |
| TP2 (SP) | 96.2 / 96.1 / 92.8 | 93.0 / 94.5 / 89.8 |
| train PP2 / engine PP1 | 96.0 / 95.6 / 90.9 | 93.8 / 93.0 / 88.3 |
| control bf16, PP2 | 46.2 / 43.0 / 41.8 | 30.5 / 18.0 / 26.6 |

Bug found on the way: MCore `BatchInvariantRMSNormFn.backward` multiplies its second dx
term by the norm weight once more (exact only for w = 1); with Qwen3's q/k norm weights the
gradient norm was ~100x too large (RL grad_norm 15-28, then NaN).

**MoE** (`nemo_rl/models/megatron/fp16_moe.py`). Expert GEMMs form fp16 x fp16 products and
accumulate K serially in fp32, emitted in fp32, so a row's result depends only on the row and
its expert (batch invariant). Two implementations give identical bits (measured at 500B-MoE
expert shapes, 1-1024 rows per expert): the engine's `InferenceGroupedMLP` runs a Triton grouped
GEMM with both operands fed through TMA descriptors (rows grouped by expert and padded to the
tile's BLOCK_M, CUDA-graph safe; tiles tuned per rows-per-expert bucket) on fp16 expert weights:
the engine's contiguous per-expert weight buffers are built in fp16 instead of bf16 (same bytes;
refit's `copy_` into each parameter lands there, bf16 -> fp16 exact in range), and training's
`TEGroupedMLP` runs one cuBLAS GEMM per expert without split-K on cached fp16 weight copies
(invalidated after every optimizer step), about 4x faster than the Triton kernel at training
shapes. Speeds at hidden 7168 / expert FFN 2048 (8 local experts), engine kernel vs the (inexact) bf16
cuBLAS grouped GEMM: fc1 76 vs 110 us and fc2 47 vs 58 us at 4 rows per expert, fc1 135 vs 107 us
and fc2 85 vs 58 us at 256 rows. SwiGLU-with-probs in fp32 (same eager ops on both
sides). Router logits fp32 through the exact fp32 GEMM. Combine: the engine sums a token's
local experts in fp64 in ascending expert order and rounds the rank partial to fp32 (as
Megatron's batch-invariant training unpermute), then MCore's `ordered_reduce_scatter_v`
(fp32, rank order; the bf16 cast after it is dropped). The engine all-gathers fp16 token
rows bit-cast through MCore's bf16 NVLS buffers (a copy); training dispatches fp32 rows.
Backward per expert on bf16 casts. Tiny Qwen3-MoE (8 experts, top-2): grad norm 0.177897
vs fp32 reference 0.177899 (bf16 baseline 0.178224). EP=2, 4 steps, checks on: mismatch 0,
engine-q exact, 0 dropped; the random tiny model gets reward 0, so no weight changes and
both this run and the bf16 control keep 100% (acceptance is not measurable on it). Training cuBLAS
+ engine TMA kernel at EP=2, 4 steps: mismatch 0, engine-q exact, 0 dropped; with
`THUNDERSYNC_DEBUG_EXPERT_GRAD_NOISE=1` (seeded noise on expert gradients, so expert weights change
every step even at reward 0) still mismatch 0 with 95-97% kept, i.e. refit into the fp16 engine
expert weights is exact.
Qwen3-30B-A3B at EP=2 on 2 + 2 GPUs does not fit (about 15B parameters per training GPU
with bf16 weights, fp32 grads, masters and Adam state: ~270 GB).

Engine PP>1: Megatron's inference wrapper casts last-stage logits to params_dtype and the
controller broadcasts them in params_dtype; under the fp16 GEMM forward both are patched to fp32
(`_patch_inference_pp_logits`). Separately, Megatron's block-scope inference CUDA graphs (the
default, `inference_cuda_graph_scope: block`, which graph the whole `GPTModel` forward) replayed a
stale input on non-first stages: the graph manager refreshes only the call's tensor arguments,
but a non-first stage gets its input through `set_input_tensor` (`decoder_input=None`,
`TransformerBlock` reads `self.input_tensor`), so every replay read the receive buffer captured at
warmup. Qwen3-1.7B at inference PP=2 decoded garbage (~45-nat gap, reward ~0.02); graphs off or
layer scope were exact. `nemo_rl/models/megatron/inference_pp_cuda_graph.py` (installed when the
engine's PP > 1) passes the received tensor as `decoder_input` on the graphed path so it becomes a
graph input. Measured, training PP=2 / inference PP=2, block graphs, 4 steps: mismatch 0, engine-q
exact, 0 dropped, kept 96.7 / 97.2 / 96.5%. Same code on Megatron main d715782 (upstream bug).
`HybridModel` / `HybridStack` have the same block-graph path and are patched the same way; measured exact on Nemotron-Nano-9B-v2 (bf16 + BI, training PP=2 / inference PP=2, 2 steps). Known gaps: MoE with
shared experts, expert TP, latent MoE (rejected at config time); the NCCL inference dispatcher at
EP>=3 is untested (the default NVLS dispatcher is used); TP=4 training forward measured
batch-invariant (0/16 sequences differ), TP=4 engine not yet run.

**Engine weight casts (engine package A).** Every recipe GEMM rounds its weight to fp16, and at decode the cast costs
4-8x the GEMM itself (Super-120B at TP=1: ~15 ms/step of dense casts plus 2.6 ms of LM-head casts, measured per shape in
research/agent_speedup_engine.md). The dedicated engine model (`is_refit_destination`, `refit_transport: null`) therefore
holds its dense GEMM weights in fp16 (`fp16_gemm.hold_fp16_weights`, called by `_prepare_engine_numerics` before the
context is sized and the CUDA graphs are captured): each patched linear's, the untied LM head's and the router's
`.data` becomes its fp16 copy. That is the same bytes and no second copy, and the per-call cast becomes a no-op. Nothing
is cached, so there is nothing to invalidate: refit `copy_` rounds bf16 -> fp16 once per refit, and momentum
`EngineStage` stage/load round-trips the fp16 storage. Embeddings, norms, tied heads and Mamba conv/A/D stay bf16. The
native MCore refit (`refit_transport: mcore`, the m2n refit of section 14q) receives every fp16-held tensor through
`Fp16HeldWeightReceive` (bf16 on the wire, `copy_` into the fp16 storage), so the engine holds fp16 under it too. The
nccl_reshard M-to-N refit writes raw sender-dtype bytes, so with it the engine keeps the per-call cast. The fp32 head (and router) GEMM runs once on stacked [hi; lo] rows on both sides
(bitwise the two-GEMM result: the BI GEMM is row-invariant).

**Hybrid engine memory (engine package A).** Batch-invariant Mamba decode (MCore) replays the training chunk scan from
per-slot buffers of the current chunk's x, z, dt, B, C (fp32 under the recipe). `mamba_bi_decode` (installed by
`_prepare_engine_numerics` when the engine is batch-invariant) does two things:
- It drops the z buffer. The gate is row-local and only the target row is stored, so the scan runs without z, and one
  kernel applies the scan's gate expression to the target row with the current token's z. That is bitwise the same,
  and saves chunk x d_inner x 4 B per slot per Mamba layer (Nano-30B 46 MiB/slot, +28% sequences/GPU at 8k; Super
  160 MiB, +37%) [derived].
- It counts the remaining replay buffers in `DynamicInferenceContext`'s Mamba share (MCore counts only conv + SSM
  state), keeping the KV share, so `max_requests` sized from memory cannot overcommit. Per rank, no communication.

The same z change inside the chunk-scan kernel is the Megatron-LM draft PR NVIDIA/Megatron-LM#8221
(`fix/mamba-bi-decode-drop-z-buffer`).

**Measured (one GPU, async scheduler, CUDA graphs; thundersync_work ledger X-engA).** Engine outputs (tokens and
logprobs) are bitwise identical to the per-call-cast engine with the z buffer, across two refits, a momentum
stage/load and prompts straddling the Mamba chunk, on Qwen3-30B-A3B L16 and Nemotron-3-Nano L8/L16. Only GEMM linears
are held in fp16: the patched TE RMSNorm reads its weight in fp32, and an fp16 copy changed logprobs by up to 0.19.
Decode ms/step, before -> after: Nano L16 5.77 -> 2.88 at 1 row and 8k context, 10.49 -> 7.45 at 128 rows and 8k; Qwen3
L16 7.84 -> 5.48 and 15.86 -> 13.49. The gain is in the intercept: the per-call weight casts (Nano L16 4.5 -> 1.6 ms of
elementwise kernels per step). The slope per row is unchanged except for Mamba, where it drops 9% because there are no
z reads (1.44 -> 1.31 us per row per layer at 8k).
A rerun on an otherwise idle cluster node reproduces these numbers to within 1%. Fitting tau = tau0 + beta * rows
(1 to 128 rows per GPU) gives, for Nano L16 at 8k, tau0 6.30 -> 3.42 ms and beta 0.0323 -> 0.0315 ms/row; at 32k,
tau0 6.67 -> 3.80 ms and beta 0.0371 -> 0.0363 ms/row.

**fp16 range.** Max |GEMM input| over 16 MATH prompts x 512 greedy tokens (HF bf16, decode
included; `thundersync_work/fp16_range_probe.py`): Qwen3-8B 10752 (6.1x headroom to 65504),
Qwen3-14B 7296 (9x), Qwen3-30B-A3B 484 (135x, expert down_proj). The dense down-projection input
(the SwiGLU output) is the only site near the limit; every other site is under 260. Opt-in
`down_proj_prescale_log2: s` multiplies the down-projection input (dense fc2 and the MoE expert
fc2) by 2^-s before the fp16 cast and the fp32 output by 2^s, with wgrad scaled to match. That
is exact apart from fp16 subnormals and widens the range by 2^s. Default 0; probe a new dense model
before running it and set s=2-4 if headroom is below ~4x. The setting must match in training and
generation (checked by the mismatch validator).

**Mamba layer input (2026-10-15, ts-mamba-fp32-in 6931e160; default on).** Stock `MambaLayer._prepare_mixer_input`
casts the layer input to `params_dtype` before `in_proj`'s fused pre-norm, so with bf16 params the fp32 residual was
rounded to bf16 at every Mamba layer, against this recipe (fp32 norm on the fp32 residual, as attention / MoE
layers do). `_patch_mamba_layer_input` feeds the fp32 residual to the patched fused norm + fp16 GEMM; training and
engine both run it, so batch invariance holds (m32-x1k / m32-x8k EXACT, 4 steps mismatch 0). Measured effect on
draft acceptance: Nano L8 harness per-token log-ratio variance v 2.5e-6 -> 5.9e-7 (thundersync_work
research/acceptance_offline_vs_inloop.md: the cast was the x~4 part of the in-loop vs offline gap). Per rank: no
memory or communication change (one fewer cast). `policy.megatron_cfg.mamba_fp32_prenorm_input=false` restores the
stock cast for control arms only; it must match between training and generation (validator field list).

## 14c. Sticky MoE routing anchored to the drafter (`thundersync.sticky_routing`)

*Opt-in, off by default (`sticky_routing: false`; `sticky_routing_delta: 0.03` when on).
Unit-tested (CPU + GPU); end to end exact (`scripts/bi_tp/sticky_moe.sh`, 4 GPUs, EP=2, 4 steps,
2026-10-13): mismatch 0 on every step for the tiny MoE and real Qwen3-30B-A3B layers (L4), with the
non-vacuity checks (weights change, sticky changes routes, engine routes == re-prefill routes).* Design and measurements:
`thundersync_work/research/agent_first_principles.md` section 3, `ACCEPTANCE_LEDGER.md`
section T; implementation notes: `thundersync_work/research/agent_sticky_impl.md`.

**Policy.** For MoE models, most of the per-token log p/q noise between consecutive versions
is top-k route flips at near ties. The policy of step k is redefined as
p_k = F(θ_k; A): at each token and MoE layer, with router scores s (θ_k) and anchor set A,
keep A if min_{e∈A} s_e ≥ max_{e∉A} s_e − δ, else plain top-k
(`nemo_rl/models/megatron/sticky_routing.py`; δ in router-logit units for softmax routers,
default 0.03, which measured the same as 0.1 and pure replay on Qwen3-30B-A3B). Kept anchors
are ordered by the current scores; an anchor set equal to the plain set is returned in the
plain top-k's own order, so F(θ; plain_topk(θ)) is bitwise the plain forward also where
MCore's top-k is unsorted (no-grad forwards without BI) or has exact ties (the normalization
sums over k in index order; a reordered set changed probability bits before this rule).
An anchor row with a negative or repeated id means "self" (plain top-k).

**Sigmoid routers with an expert bias (Nemotron-H, DeepSeek-V3 class).** MCore selects on
sigmoid(logits) + expert_bias and gathers the unbiased sigmoid at the selected experts, then
normalizes over top-k and scales. The rule runs on the biased selection scores (δ in
sigmoid-score units), MCore computes the probabilities itself. Group-limited top-k: the plain
set is MCore's group-limited one and the keep test compares against every expert outside the
anchor set (conservative). The expert bias is part of θ (frozen under NeMo-RL's GRPO default
`moe_router_bias_update_rate: 0`; if updated between steps it is simply part of θ_k). The fused
TE router (`moe_router_fusion`) bypasses RouterReplay and is rejected with sticky routing.
Unit test `tests/unit/test_sticky_hybrid.py` (Nano router: 128 experts, top-6, scale 2.5; with
and without groups; sparse and dense paths; sorted/unsorted; ties): anchors = plain top-k is
bitwise plain. **Hybrid models** (HybridModel pattern strings): MoE layers are the 'E'
positions of the main pattern (`router_replay._global_moe_layer_numbers`; before, moe_layer_freq
was assumed and Nano counted 52 MoE layers instead of 23, so learner anchors and engine route
logs disagreed in shape). Non-gated squared-ReLU experts run on the fp16 expert path
(`fp16_moe.relu2_probs`, same eager fp32 function in training and engine); shared experts are
dense TE MLPs on the fp16 GEMM path. Under the fp16 GEMM forward the engine's Mamba conv
state defaults to fp32 (`resolve_mamba_conv_states_dtype`; in_proj emits fp32 and MCore casts
the engine's conv input to the conv-state dtype, params_dtype by default, which rounds it).
Hybrids use the same route record, transport and async-scheduling path as MoE models (no
hybrid-specific scheduler requirement; per-rank route cost = that of section 14c with L = number
of 'E' layers of the stage: Nano 23 x 6, Super 40 x 22 int16 per token). 4-GPU check
`scripts/bi_tp/nano_hybrid_fp16.sh` (Nano L8 MEMEM*EM, EP=2, expert-grad noise, 4 steps,
SCHED=async default | legacy), with the Mamba SSD parity port active: PASS with sticky on and off
under async and legacy scheduling (2026-10-14): train == engine mismatch 0 and exact_token_frac 1
on every step, engine q exact, weights change every step, every sticky step's anchors equal a
fresh engine prefill ([route check]) and the sticky rule changes routes against plain top-k
(thundersync_work/research/agent_hybrid_integration.md). With the integrated recipe (branch ts-hybrid-integ:
sticky routing + momentum draft weights + pooled ⊥-BV, async scheduling) the same check PASSES (2026-10-14): sticky +
momentum + ⊥-BV and its STICKY=0 control both give mismatch 0 and exact_token_frac 1 on all 4 steps, engine q exact,
3 ⊥-BV steps (128 output slots from 192 drafts, no fresh slots), momentum drafts staged every step; with sticky every
step's route check is exact (196608 / 57384 / 50028 tokens) and deadline-cut drafts are completed by 8–11 Mamba
continuation rounds (thundersync_work/research/agent_nano_integ.md).

**Full Nemotron-3-Nano (52 layers, 31.6B) per-rank layout** [derived from the HF config; TP=1, PP=1].
Routed experts 29.4B, everything else 2.2B. Learner rank (N GPUs, EP=E, DistributedOptimizer, fp32 masters
outside TE): experts (2 B bf16 + 4 B fp32 main grad + 2 B fp16 expert cache) × 29.4B/E + 12 B × 29.4B/N
(master + Adam over expert DP), dense 6 B × 2.2B + 12 B × 2.2B/N; momentum drafts add ≤ 0.42 GB scratch.
EP=8, N=8: ≈ 90 GB; EP=4, N=8: ≈ 119 GB; EP=4, N=4: ≈ 167 GB (no activation headroom); EP=2, N=2: ≈ 320 GB.
Engine rank (same TP/EP): 58.8 GB/E experts + 4.4 GB dense; per request slot SSM state 46 MiB + BI Mamba
replay buffers 116 MiB (not yet in MCore's state accounting) + KV 6 KiB/token. Smallest layout without optimizer
offload: EP=8 on 8 learner + 8 engine GPUs (`thundersync_work/sbatch/submit_nano.sh`). Measured on one 4-GPU node
(2026-10-14): EP=2 2+2 with the optimizer offloaded to host builds the engine (~76 GB/GPU incl. CUDA graphs) but the
learner runs out of memory (~188 GB) in its first forward at micro batch 8 × 2048 tokens; micro batch 1 untested. Scaling: every term is
per-rank and divides by EP (experts) or is replicated per engine replica (Mamba/attention state, ∝ slots); none
grows with the number of nodes.

**Anchor schedule (exogenous).** A at position t = the plain θ_{k−1} forward's top-k on the
rollout's own prefix, for every position up to the rollout cap c = prompt + min(max_new_tokens,
max_total_sequence_length − prompt) (the engine stops there; a plan at c is complete:
`SpeculativeGeneration._cap`; content-independent); "self" for rollouts never drafted (or whose draft was dropped for
missing logs, which is bookkeeping, not content). No recursion: anchors always come from a
plain forward. Drafts are single-version (`draft_lookahead=1`), so every drafted position's
anchor is exactly the drafter's logged route.

**Engine (drafter) side** (`worker.py`, `routes.py`). With sticky routing every MoE router's
MCore `RouterReplay` is in RECORD, armed before the engine captures its CUDA graphs, recording
only in inference forwards. The model's `moe_enable_routing_replay` makes MCore's dynamic
context bind each router's record buffer per step: its `routing_indices_buffer` on CUDA-graph
steps (the copy is part of the captured router), none on eager steps (the top-k stays in
`recorded_topk_idx`); the route log reads each forward's routes from that binding
(`_step_routes`). MCore's own routing-replay bookkeeping is dropped (`_drop_mcore_route_bookkeeping`:
its synchronous host copy of every step's routes, per-KV-block host storage and per-request
`routing_indices` in replies), which also lets the engine run with async scheduling: MCore
rejects routing replay under async scheduling only for that bookkeeping, and the engine
subclass (`_engine_class`) passes its check when the route record is installed. No MCore
source change; an MCore option to record routes without the bookkeeping would replace the
override. Measured engine decode step (L16 Qwen3-30B-A3B, one GPU, every token logged): async
with the route log 17.27 ms at 512 concurrent decodes vs 17.25 ms without routing replay
(6.89 vs 6.90 ms at 1); legacy with MCore's bookkeeping 20.28 ms (8.16 ms at 1)
(`thundersync_work/research/agent_scaling_fixes.md`). `install_draft_route_log` wraps the controller's
`_dynamic_step_forward_logits` (Python on every forward, graphed or eager): before the launch
it selects the step's tagged tokens from the context's CPU bookkeeping (slot and absolute
position of each token, i.e. the forward being launched) with numpy, looking up draft keys
only for slots whose request is new (`SlotTags`); after the launch it enqueues the gather of
those route rows on the device (index from pinned memory, non-blocking) and one asynchronous
copy to a pinned staging ring (`RouteLog`): no host-device sync and no Python per token
(0.2–0.4 ms host time per forward vs 3.6–5.5 ms before, which waited for the forward;
`thundersync_work/research/agent_scaling_fixes.md`). Staged rows are appended to a host store
once their copy completed; rows are keyed by draft key and absolute position, so eviction
re-prefills write the same positions. Expert-parallel dummy steps (an idle engine runs MCore
`dummy_forward`, whose context sets only request and token counts: request ids, slots and
positions are still the last real step's) select no tokens; the wrapper reads each forward's
`is_dummy_forward` from `_dynamic_step_context_init`. (Logging them overwrote the last position
of the last request to finish and position 0 of the first request of the last prefill step with
padding routes: `[route check]` 4 positions off at EP=2, 2 at PP=2; reproduced on one GPU with two
`dummy_forward` steps, exact with the fix: `thundersync_work/research/agent_route_regression.md`.) A rebuilt engine's forward
is wrapped into the same log and new router instances are re-armed. Prompt positions are logged by the draft's own prefill
(prefix caching must be off). Pipeline stages log their layers; under sequence parallelism the
TP shards are all-gathered (only on steps with tagged tokens) and TP rank 0 logs.
**Transport** (`RouteRefs`): `pop_draft_routes` puts each draft's dense routes in the Ray object
store in blocks of 4096 positions and returns only (layers, logged range, block handles); the
driver keeps the handles (draft tuple index 8, `Plan.anchors`) and the logged prefix length,
and each learner rank fetches only the blocks of its own pipeline stage's layers
(`fetch_anchors`; `routed_experts` in the stage-local layout). Nothing per token passes through
the driver; the driver frees a step's blocks once the step has trained (`release_routes`).
Per learner rank and step: rows(DP rank) × S × (L/PP) × K × 2 B (fetching only the
sequence-parallel shard is not built yet). Drafts are cut to the tokens whose anchors are all
logged (`anchored_draft_length`: g tokens need positions 0..P+g−2), before their hidden states
are popped, so hidden rows always match the kept tokens (`_pop_logs`).

**Learner side.** Every forward that defines p_k applies the rule with the same anchors:
the verification forward (`_training_forward_rows(..., anchors=...)`, through NeMo RL's
`set_router_replay_forward`, also under PP and in EP lockstep dummies), the training forward
of the rollout (the training batch carries the plans' `route_refs`; the worker's
`_with_routes` fetches them into `routed_experts` [B, S, L_stage, K] and gives all-self rows to
rows and batches without anchors), and the mismatch check's `get_logprobs`. The sticky wrapper around
`RouterReplay.get_replay_topk` stores the effective selection for REPLAY_BACKWARD, so
activation recompute replays exactly the forward's routes. Top-k has zero derivative a.e., so
the gradient is the ordinary backward with routes fixed (as R3 router replay).

**Continuations at θ_{k−1}** (`SpeculativeGeneration.verify_sticky`). Positions after a
rejection still need θ_{k−1} routes on the new prefix. With sticky routing, at the start of
iteration k the driver verifies the drafts while the engine still holds θ_{k−1} and completes
every drafted rollout before the refit (`_continue_sticky`): each incomplete plan gets an
engine stream decoding with the plain θ_{k−1} forward from prompt + kept prefix toward EOS or
the cap (logging routes, hidden states and log-probs as for any draft). Every
`sticky_continuation_chunk_tokens` (2048) decoded tokens, or at the stream's end, the new
piece is block-verified against the same p_k, together with the ready pieces of all other
streams, while every stream keeps decoding. Whole acceptance extends the prefix and the same
stream goes on: its later tokens are q samples given the extended prefix, the same law as a
restarted stream, so no bonus is needed. A rejection appends the residual token, aborts the
stream and starts a new one from the new prefix, so it wastes at most the tokens decoded past
the piece (before: the whole tail to EOS), and verification overlaps decoding instead of
alternating with it. Pieces are cut at fixed token counts (exogenous) and to their logged
routes and hidden rows (bookkeeping). Deadline drafts are not aborted under sticky routing
(`_collect_running`): their streams run toward the cap, the deadline snapshot is verified as
the first piece, and a whole-accepted unfinished draft goes on with its own stream (no new
prefill of prompt + draft; the engine keeps decoding it through the optimizer step, at
θ_{k−1}, which it holds until every drafted rollout is complete); every other draft's stream
stops after the first verification. A whole-accepted unfinished draft emits no bonus token
(the bonus row's anchor, the route of the last drafted token, is not logged at the deadline);
it needs a continuation anyway because every drafted rollout must be completed at θ_{k−1}, and
that continuation is now its own running stream. Then the engine refits
to θ_k and the iteration's rollouts start (drafted ones return their completed plans; the
rest decode plainly at θ_k = F(θ_k; self)). Exactness of the whole scheme (drafts cut at
exogenous times, block verification, no bonus, continuation rounds) is checked by exact
enumeration on a toy routed language model (`test_routes.py`, TV < 1e-12), with a control
showing that anchoring continuations to self after a rejection (anchors depending on τ)
is biased.

**Requirements** (`validate_sticky_routing`): speculate, verify_mode=block,
verify_source=engine (BI), draft_lookahead=1, no tail_lookup, no
engine prefix caching, no R3 router replay, an MoE model (the engine must report a route
log, else the driver raises).

**Checks** (`scripts/bi_tp/sticky_moe.sh`, `THUNDERSYNC_ENGINE_Q_CHECK=1`
`THUNDERSYNC_MISMATCH_CHECK=1`). Mismatch 0 compares the learner with itself under the same
anchors, so the launcher also asserts that the checks are not vacuous: drafts are accepted
every step after the first; the weights change every step (`mismatch/param_checksum`;
expert-gradient noise on every model, since zero-reward runs do not update); the anchors of
every draft equal a fresh engine prefill of the same rows at the drafting weights
(`[route check]`, `spec/route_check_exact == spec/route_check_tokens > 0`: catches a route
log mapped to the wrong slot or position); and the rule changed routes against plain top-k
(`[sticky check]` `differs_from_plain > 0`). These prove numerics; the output law of the
scheme rests on the enumeration tests (`test_routes.py`).

**Cost and open issues.** No extra learner FLOPs beyond the verification forwards of
continuations. The engine refit waits for the continuation rounds, so rollouts no
longer overlap verification (keeping θ_{k−1} available after the refit, e.g. as a sparse
weight delta toggled per chunk, would restore the overlap; not built yet). Multi-version drafts
(`draft_lookahead>1`) are not built yet. Multi-turn rollouts: section 14m; a horizon c below the
rollout cap with engine anchor apply: section 14i.

## 14d. Pooled null-augmented block verification (`thundersync.null_bv`)

*Opt-in, off by default. Unit-tested (exact enumeration, CPU + GPU worker path); end to end
exact (`scripts/bi_tp/null_bv_moe.sh`, tiny MoE, K=1.5, 2026-10-13): mismatch 0 on every step,
1.48 drafts per output, ⊥ fraction 0.32, no fresh slots.*
Scheme and measurements: `thundersync_work/research/agent_multi_draft.md`; implementation notes:
`thundersync_work/research/agent_null_bv_impl.md`.

**Why.** The step waits on the earliest rejection among the ~40 longest drafts. Any exact
single-draft verifier has P(reject before t) ≥ TV(prefix_t), and block verification puts ~30% of
its rejections in the first 10% of a draft (`agent_multi_draft.md` §2.4). ⊥-BV lets a draft be
*discarded at the root* instead: with K ≈ 1.5–2 drafts per output, outputs needing a tail decode
drop from .12 to 8e-4 (M = 1.5) at 32k on real Qwen3-30B-A3B L16 rows, and none of the measured
ones rejects in the first half of the draft [measured offline, `agent_multi_draft.md`].

**Rule.** Draft j of a prompt is block-verified (section 4) against the augmented target

    p̂ = p/M ⊕ (1 − 1/M)·⊥        (⊥ = "no output", only possible at the first position; q(⊥) = 0)

In section 4's notation: the first log-ratio gets −log M, so b_i = w_i / max(M, max_{u≤i} w_u)
(w = prefix ratio p/q); r_i (i ≥ 1) is unchanged in form; the root residual is
(p_1/M − q_1)₊ plus the ⊥ atom 1 − 1/M (so h_0 = 1). Outcomes: τ = g (whole draft), τ ≥ 1 with a
residual token (prefix + token + a p tail, as today), τ = 0 with a vocabulary token (rare: a
fresh-tail output), or ⊥ (nothing sampled, draft discarded). Since block verification is exact for
any pair of autoregressive laws, the outcome is ⊥ with probability exactly 1 − 1/M and otherwise
an exact p-sample [derived]. M = 1 is block verification bit for bit (`null_m` defaults to 1;
the existing tests pass unchanged).

**Pooling (fixed G).** The engine drafts N = max(G, round(K·G)) drafts per GRPO group
(`null_bv_drafts_per_output` = K). Each prompt's drafts are decided in their drafting order (a
fixed index order, chosen before verification); M_j = policy(drafts left d, slots left s) depends
on counts only; the first G non-⊥ outputs fill the group's G slots; once the slots are full the
remaining drafts are not decided; slots still empty when the drafts run out decode fresh from p.
Exactness [derived; enumerated]: given the past, a decided draft's non-⊥ output is p-distributed,
independent of the past, with success probability 1/M_j; M_j and the stopping rule are functions
of the past counts, so the successes in index order are iid p (optional sampling) and the G slots
are G iid p-samples. Forbidden (and not done): reordering drafts or choosing M_j by content
(length, reward, acceptance). Group size G is unchanged (variable G is not built: it changes the
GRPO recipe and needs the user's approval).

**Envelope policy** (`null_bv.py`). A DP over (d, s) minimising the expected critical-path cost
Σ P(tail decode)·E[tail/cap] + one full fresh decode per unfilled slot, with the per-success
tail-decode probability from the Brownian model of log w (drift a²/2, variance a² = L·v, L = the
rollout cap, v = `null_bv_logratio_var`, default 8e-7; validated against real rows in
`agent_multi_draft.md` §2.3). M = 1 whenever d ≤ s. At v = 8e-7, cap 32k, G = 16, N = 24 it starts
at M = 1.3 and falls to 1.1–1.0 as slots fill [derived]. `null_bv_fixed_m` replaces the DP by a
constant (still 1 when d ≤ s).

**Implementation.**
- Samplers (`block_verification.py`, `block_verification_fused.py`, `engine_q.verify_engine_q`):
  a `null_m` argument; the emitted id `NULL_TOKEN` (−1) is ⊥. Same generator calls (one
  `rand(g+1)`, one `multinomial`, over V + 1 entries at a ⊥-capable root residual); eager, lazy and
  engine-q paths are draw-for-draw identical (tested). Empty drafts are rejected by the sampler
  and dropped from pools by the driver (⊥ lives at the first drafted position; emptiness is a
  deadline fact, not a function of the tokens).
- Placement and decisions are decoupled (scale). The driver places verification rows on the
  learner's data-parallel ranks by tokens (`lpt_owners`, longest first onto the least-loaded
  rank; `spec/verify_load_max_over_mean`), not by prompt: with all of a prompt's drafts on one
  rank, the busiest rank carried 1.43× the mean at DP 8, 1.93× at 16, 2.62× at 32 and 4.16× at 64
  in the review's simulation [derived], LPT ≈ 1.00×. The worker
  (`verify_drafts_block_engine(null_envelopes, row_ranks)`) verifies each of its rows at every
  envelope the policy can pick (`envelopes`: 1, and the grid or the fixed M) with one set of draws
  per row (`verify_engine_q_grid`: one `rand(g+1)` and one uniform u; per M the lazy scan with
  M's weights and the emission by inverting M's emission CDF at u) and returns
  {M: (τ, y, log p(y))}. The driver then walks each prompt's rows in drafting order with the
  count process (`decide_pools`, `NullGroup`) and takes each row's outcome at the M it gets;
  rows after the slots fill are "unused". **Exactness is unchanged**: a row's draws are
  independent of every other row, the M used for it is a function of earlier rows' outcomes
  only, and only one of its outcomes is ever used, so how outcomes at different M are coupled
  is irrelevant (the optional-sampling argument above). Tested by exact enumeration of the pooled
  process over comonotonically coupled outcomes at all M with `decide_pools` (fixed M and d/s
  policies, TV < 1e-9), a control that picks a row's M from its own outcomes (TV > 1e-3), a
  chi-square of the grid sampler's marginal at each M, and the worker's grid results against the
  reference sampler on GPU. Cost: the forward is shared; per row up to |envelopes| (≤ 13) lazy
  scans and emissions, each scan re-using the cached rows.
- Waves (`_verify_null`): each wave forwards, per prompt with slots left, its next ⌈s·M⌉ rows in
  drafting order (s slots left, M the current envelope: the expected number of drafts that fill
  s slots); decisions run between waves. Rows past the filled slots are never forwarded (their
  hidden states are freed; `spec/null_skipped_drafts`, `spec/null_waves`). Which rows are
  forwarded depends on counts only, so the decisions equal the one-shot process (tested against
  `decide_pools` on random outcomes).
- Driver (`speculative.py`): `draft_streaming(prompts, null_slots)` records the slots of the next
  step (the drafting order is the verification order); `verify` builds the pools and plans only
  for outputs (in drafting order); rollouts without a plan decode fresh (exact). Sticky routing
  composes unchanged: anchors stay per draft, and outputs with a rejection are completed by the
  θ_{k−1} continuation rounds of section 14c.
- Loop (`grpo_loop.py`): K·G drafts per group; ⊥-BV verification runs before the rollouts, in
  waves (publishing each prompt's plans as soon as its pool is decided, overlapped with the
  rollouts, is not built yet).
- Requirements (`validate_null_bv`): speculate, verify_mode=block, verify_source=engine,
  verify_on=learner, draft_lookahead=1, no fused_verify_train, no tail_lookup.

**Metrics** (per step): `spec/null_bot_frac` (⊥ / decided drafts), `spec/null_tail_frac`
(outputs with a rejection, i.e. a tail decode / outputs), `spec/null_longest_min_reject_pos` and
`_frac` (earliest rejection among the longest decile of decided drafts; −1 / 1.0 if none),
`spec/null_drafts_per_output` (decided drafts / outputs), `spec/null_fresh_slots`,
`spec/null_unused_drafts`, `spec/null_mean_m`, `spec/null_empty_dropped`, plus `spec/null_slots`,
`spec/null_outputs`, `spec/null_drafts`; `spec/accepted_frac` and `spec/full_accept` are over
outputs.

**Tests** (`tests/unit/test_null_bv.py`, 25): per-draft law = p/M ⊕ (1−1/M)⊥ by exact enumeration
(V = 3, L = 3, TV < 1e-12); joint law of the G = 2 slots from N = 5 drafts = p⊗p by exact
enumeration for the DP, a fixed M = 2.5 and a count-adaptive d/s policy (TV < 1e-9), with a
content-dependent-M control biased (TV > 1e-3); Monte Carlo through the worker's sampler and
`NullGroup` (chi-square z = +0.2 on 63 dof, 12k groups) against the outcome-selection control
(z = +164); eager/lazy/engine-q identity with ⊥; DP policy and the Brownian model against the
report's numbers; config rejections; driver pools/plans/metrics/fresh slots; worker decisions on
GPU against a reference loop [measured: all pass; full research suite 404 passed, the 7
`test_streaming_pipeline.py` import errors are pre-existing].

**Costs and open items.** K× engine draft decode and K× verification forwards (unused rows are
forwarded too); no extra training work. Not built yet: chunked publication of pools, skipping
forwards of unused drafts, multi-version (cohort) pools, the scorer path, variable G (needs
approval). Not measured yet: acceptance/step time on a real MoE with K drafts (engine capacity is
the price: `agent_multi_draft.md` §3).

### 14d.0 When to use ⊥-BV (K > 1) and prompt drop (agent_tail.md, 2026-10-14)

Measured on Nano L8 (complete drafts, 5 steps per arm):
- The rejection tail after verify0 is 8.5 s at K = 1 and 0.00 s at K = 1.5 / 2 / 3; exact at every K.
- But verify0 grew 1.5 → 13–17 s. ⊥ and near-⊥ drafts scan back to the root, and q rows were rebuilt per envelope in
  small batches.
- The shared-row scan (rows built once per chunk of 512 positions, across envelopes) is the fix. Re-measure before
  using K > 1 for speed.

Use K > 1 only when **both** hold:
1. **The step is train-bound with near-full draft coverage**, i.e. the engine's idle window covers L·τ(K·B) before
   θ_{k+1} exists (lanes, or a learner slower than generation).
   - With one engine under sticky routing at Nano 8k/16k, generation binds, the best coverage is ≈ 0, and K changes
     nothing.
2. **The draft-side costs fit**:
   - per-token draft τ ×1.2–1.7 (K 1.5–3, B 128 on 4 GPUs; ×1.3–2.4 at EP8, B 1024);
   - engine slots ×K (Nano 210/258 MiB per row at 8k/16k; Ultra ~1.6 GB per row at 64k binds first);
   - learner verify forward ×(drafts per output, 1.2–1.6).

Then:
- K 1.5 at 8k, 2 at 16k (v ≈ 1.1–1.5e-6) brings the rejection tail under 5% of a generation (Brownian model,
  B 128, G 16).
- Prompt drop with constant M (`null_bv_fixed_m` 1.75–2, K 2–2.5, a few spares) removes the residual tails that the
  DP's forced M = 1 leaves at pool exhaustion; one envelope also means one scan per draft.
- Defaults stay K = 1 (null_bv off) otherwise.

### 14d.1 Opt-in tail tools (off by default; `thundersync_work/research/agent_tail.md`)

The defaults are unchanged: fixed G, the DP envelopes, and the standard GRPO estimator.

- **Prompt drop** (`null_bv_prompt_drop`, `null_bv_spare_prompts`; needs `null_bv_fixed_m`).
  - Rule: M is the constant `null_bv_fixed_m` at every count; the forced M = 1 is never used. A
    pool that can no longer fill its G slots (drafts left < slots left) is dropped.
  - Why the kept groups are exact: the drop depends on success counts only. Given the past, each
    draft succeeds with probability 1/M for every prompt and every (p, q), so P(drop) is the same
    for every prompt and the kept groups are G iid p-samples.
  - Enumerated through `decide_pools`:
    - kept-group TV < 1e-9;
    - P(drop) equals the Binomial count law for two (p, q) pairs;
    - a content-based drop (control) has TV > 1e-2 (`tests/unit/test_tail_tools.py`).
  - Batch selection: the dataloader batch carries `null_bv_spare_prompts` extra prompts. The step
    trains on the first (batch − spares) undropped prompts in batch order
    (`SpeculativeGeneration._select_prompts`). Plans of the other prompts are discarded before any
    tail decode. If too few prompts survive, dropped prompts fill the shortfall and decode fresh.
  - This **shifts the prompt mix** through the spare order, like DAPO's dynamic-sampling filter.
    It never shifts it through the samples' contents.
  - Metrics: `spec/null_dropped_prompts`, `spec/null_drop_rate`.
- **Drop-IS** (`null_bv_drop_is`). It gives an unbiased gradient, not exact samples, so it is
  opt-in only, like banded IS.
  - Rule: a ⊥-BV output rejected inside its draft is discarded like ⊥: it fills no slot and gets
    no tail decode. Each kept row's advantage is multiplied by max(1, W*/M), where W* = max_t of
    the prefix ratio p/q (the worker reports `null_log_wstar`).
  - Why it is unbiased: q(x)·P(whole | x)·max(1, W*/M) = p(x)/M exactly, so the weighted kept
    rows follow p up to a common constant. Enumerated per draft and for the pooled process; the
    unweighted control has TV > 1e-3.
  - Combined with prompt drop, no row needs a tail from a root decision at fixed G.
  - Continuation pieces of deadline-cut drafts are still plain BV.
  - Metrics: `spec/drop_is_ess`, `spec/drop_is_weighted_frac`, `spec/drop_is_max_weight`,
    `spec/drop_is_tail_drops`.
- **Tail-excluded baseline** (`tail_excluded_baseline`; needs group or batch granularity). This
  is an estimator change.
  - Rule: each row's baseline (and std) is computed over the group's *other* rows that needed no
    continuation after verification, or over all other rows if there are none. A group's
    tail-free rows can then train before its tails return (`StreamPlanner`,
    `register_group(tails=...)`).
  - Why it is unbiased: the baseline is a function of the other rows' (value, flag) pairs only.
    E[A_i ∇log p(y_i)] is therefore exactly the policy gradient for the mean baseline, and a
    positive multiple of it with the std.
  - Pitfall: the flag depends on y (P(tail | y) is a function of W*(y)), so the rule must not
    change with the row's own flag. GRPO's self-inclusive mean over the tail-free subset is
    biased.
  - Exact enumeration: that variant fails, and so does dropping the tail rows. The rule above
    holds to 1e-12.
- **Guard**: `NullPolicy.m` accepts only the integer counts (d, s). An envelope chosen from any
  pool-wide content statistic, such as the mean length of all other drafts, is biased
  (z = +208, `agent_tail_exact_schemes.md` §3.4).

**Calibrated envelope policy (`thundersync.null_bv_calibrate`, opt-in; ts-lanes-stream).** The DP's
tail model needs the per-token Var log p/q v. A constant v is wrong in both directions: at Nano L8
1k, v = 8e-7 makes tails look free at M ≥ 1.1, so the DP picks M ≈ 1.1 (⊥ on ~9% of drafts). Yet
about one output per step (1/128) needs a tail of 400–500 tokens. That tail is the slowest row,
and with streaming it sets the step [M, ord-p-l2-m128 steps 3–4]. During real post-training, v
also drifts as the policy sharpens. The calibration measures v on every verified row:
- each worker returns (tokens, Σ log p/q, Σ (log p/q)²) per row, 3 numbers, O(rows) per rank;
- the driver keeps a moving average over steps (`null_bv_calibration_decay`);
- the policy table is rebuilt at the step's first verification, from earlier steps only.
Exactness: within a step v is fixed before any of the step's drafts is verified (a function of the
past, like a hyperparameter), so M_j stays a function of the counts (d, s) alone. The optional-
sampling argument above holds unchanged. Estimating v from the step's own rows would be forbidden
(content-dependent M). Lanes share one calibration (v and the verify cost belong to the drafter and verifier, not to a lane); steps are verified one after another, so each step still sees earlier steps only.
`null_bv_tail_weight` scales the tail cost against a fresh decode (a constant).
Verification is not free either. Each verified draft costs `verify_cost` fresh decodes, so a BOT
draft costs a verification too. `verify_cost` is measured the same past-only way: earlier steps'
verification seconds per verified draft, divided by a full decode (earlier drafts' per-token
latency at the cut times the cap). It therefore adapts to wherever verification runs (learner,
engine, fused with training). This matters because fixed M = 1.3 at L8 was slower: verify0 grew
from 1.5 to 2.1–2.3 s while a rejection still occurred [M, cut-p-l2-M13]. Logged:
`spec/null_v_used`, `spec/null_verify_cost_used`, `spec/logratio_var_step`,
`spec/logratio_mean_step`, `spec/null_v_next`.
Unit-tested (past-only update, decay, counts-only M). Validation queued: order-replay exactness run
and perf vs `cut-p-l2-M13` (fixed M = 1.3).

## 14e. Banded importance sampling (`thundersync.banded_is_band`)

*Opt-in, off by default (unset = every row is an exact sample of the current policy). Unit-tested
on CPU (exact enumeration, sampler identity, driver/loss plumbing); the GPU worker test and any
end-to-end run are not run yet.* Analysis, measurements and what it gives up:
`thundersync_work/research/agent_draft_reuse.md` §1.

**What changes semantically.** With the band on, rows are **exact in expectation, not in law**:
the trainer gets (Y, w) with E[w·f(Y)] = E_p f(Y) for every f, so the policy gradient stays
unbiased, but the rows themselves follow m = p/w, which is close to q (TV(m, p) ≈ TV(q, p) at
sequence level, ≈ 0.4·√(L·v) ≈ 9% at v = 1.5e-6, L = 32k) [derived; enumerated]. Unweighted
statistics of the rows (mean reward, lengths, pass rates, entropy) are statistics of m, not p;
`reward_is` is the weighted (unbiased) mean reward.

**Rule.** For a draft with prefix log-ratio S_t = Σ_{s≤t} log p(x_s)/q(x_s), let T be the first t
with |S_t| ≥ b (T = g if none). Block verification (section 4) runs against the target p̃ = q at
positions 1..T and p after T (rows past the draft are p); p̃ depends on the prefix only, so the
result is exact for p̃ [derived]. Since b_i = 1 and r_i = 0 below T, every output keeps the draft
through T (τ ≥ T) and the whole prefix inside the band is accepted with probability 1. The
continuation after a rejection is ordinary p decoding, and the row's loss weight is w = exp(S_T)
(exp(S_g) without exit), so |log w| < b + one token's |log p/q|. Lemma (agent_first_principles §2.1)
+ optional stopping give E[w f(Y)] = E_p f [enumerated to 1e-12]. b → 0 is plain BV; b = ∞ is
whole-sequence IS (rejected by validation: unbounded weights).

**GRPO advantages.** The weight is per row; the group's other rows are drawn from m, not p. Exact
enumeration of E[Σ_i w_i A_i ∇log p(y_i)] against the exactly-on-policy expectation
(`research/draft_reuse/grpo_banded_enum.log`):
- RLOO / mean-centred advantages with a constant normaliser: identical (rel. err 1e-15). A_i =
  R_i − b(Y_{−i}) and E[w_i ∇log p(y_i)] = 0, so the baseline term vanishes whatever law the other
  rows follow.
- Leave-one-out std normalisation (`grpo.normalize_rewards`, NeMo-RL's default): A_i =
  a(Y_{−i})·R_i + c(Y_{−i}), so the per-prompt gradient is E_m[a]·∇J_x instead of E_p[a]·∇J_x:
  **the same direction per prompt**, a prompt scale off by 0.1–0.7% at TV(q,p) 1.5–15% in the toy.
- Token-level loss (1/N_tok over the batch): the normaliser depends on the other rows' lengths;
  toy group of 3: rel. err 0.5–2.5%, cosine ≥ 0.99995. In a batch of ~1000 rows N_tok
  concentrates and the effect becomes a global scale of size |E_m len − E_p len|/E_p len [derived].
- The group's product of weights Π_j w_j reproduces the exact expectation for every estimator
  (1e-16), at ESS^G variance (not built: ESS 0.95 → 0.44 at G = 16).
- For scale: exact GRPO itself differs from ∇J by more (token mean: cos 0.994; std: ×1.2).
- `force_on_policy_ratio`: required and unchanged; w multiplies the advantage (a detached per-row
  constant), never the PPO ratio. Folding w into prev_logprobs would let the ratio clip act on
  it: biased, up to 46% rel. err in the toy (control row of the log).

**Cost.** One draft per output (no extra decode), and gradient variance × 1/ESS, ESS =
E[w]²/E[w²] ≈ e^{−L·v} when nothing exits. Brownian model (`research/draft_reuse/banded_sim.log`,
20k drafts per cell at the cap): v = 1.5e-6, 32k, b = 1: tail decodes 0/20000 (BV: 16%; ⊥-BV
M = 1.5: 6.9e-3, M = 2: 1.5e-4 with 1.5–2× drafts), ESS 0.95; 64k: 2.6e-4 with ESS 0.90 (⊥-BV
M = 2: 4e-3). b = 0.5 at 32k: 3.5e-3, ESS 0.952. Rejections, when they happen, are after T, i.e. late.

**Composition with ⊥-BV.** Mathematically valid (enumerated with per-group ⊥-BV: P(output) = 1/M
exactly, weighted law p) but wasteful: inside the band the ⊥-BV acceptance is 1/M instead of 1, so ⊥
only protects the post-exit walk. Use one or the other: banded IS (K = 1, ESS cost) or ⊥-BV (K > 1,
exact rows). The pooled ⊥-BV of section 14d (`null_envelopes` grid, driver-side decisions) is not
banded, so `banded_is_band` with `null_bv: true` is rejected by validation (not built).

**Composition with sticky routing, sticky rounding and momentum draft weights.** Banded IS only
changes the target of a draft's *first* verification (`verify` → `_verify_block(band=...)`), whatever
p is (the sticky policy F(θ_k; A), with or without sticky rounding) and whatever q is (the engine's
logged law at θ_{k−1} or the momentum-predicted θ̂_k). Sticky continuation rounds
(`_continue_sticky`) call `_verify_block` without a band: plain BV, exact samplers of p given the
prefix, which is what the weight w = exp(S_T) requires. The weight is attached to the `Plan`, survives
the in-place continuation of the plan, and is registered (`PrefixClaims`, O(1) per entry) when the
completed plan is taken by the rollout loop; the trainer claims it by the full trajectory.

**Code.** `block_verification.band_exit / banded_log_weight / banded_target / zero_mass_below`;
`band` argument of `block_verify(_lazy)`, `output_distribution_given_draft`,
`block_verify_stats_lazy`, `verify_engine_q`, `worker.verify_drafts_block_engine` (rows report
`band_log_weight`, `band_exited`); `speculative.Plan.log_weight`, `claim_log_weight`; only a
draft's first verification is banded (sticky continuation rounds are plain BV, exact samplers of p
as the weight requires); `grpo_loop.build_chunk_data` multiplies the advantage by
`payload["is_weight"]` (open-group buckets stay exact: the bucket is Σ w_i ∇log p_i, scaled
later). Metrics: `spec/band_ess`, `spec/band_exit_frac`, `spec/band_w_min/max`,
`spec/band_outputs`, `reward_is`. Validation: finite b > 0, speculate, block + engine source, no
fused verify-train.

**Tests** (`tests/unit/test_banded_is.py`, 22): weighted law = p by exact enumeration for
b ∈ {0.15, 0.5, 1.5, ∞} × M ∈ {1, 1.7} (w a function of the output, τ ≥ T, P(⊥) = 1 − 1/M;
unweighted law visibly ≠ p); GRPO group expectations by enumeration (RLOO exact; LOO-std same
direction, other scale; product weights exact); eager / lazy / engine-q pathwise identical with
the band (incl. generator state); config, loss weight, driver plans/claims/stats, continuation
rounds unbanded; worker on GPU (skipped on CPU) [measured: CPU suite 358 passed].
## 14f. Momentum-predicted draft weights (`thundersync.draft_weights: momentum`)

*Opt-in, default `current`. Unit-tested (formula, Megatron DistributedOptimizer on 1 and 2 ranks
with TP / EP / DP sharding, precision-aware optimizer (TE FusedAdam fp32 masters and bf16 + int16
remainders, 1 rank), engine staging round trip, driver order, config); end to end exact on the tiny MoE with sticky
routing at EP=2, with and without ⊥-BV (`sticky_moe.sh`, `null_bv_moe.sh`, 2026-10-13) and at TP=2 EP=2 with sequence parallelism; L4 runs queued.*
Implementation notes: `thundersync_work/research/agent_momentum_draft.md`; offline numbers:
`ACCEPTANCE_LEDGER.md` section T (per-token Var log p/q of a θ_n draft vs a momentum draft:
~10–15x lower with momentum on Qwen3-30B-A3B layers).

**What changes.** The engine drafts the rollouts of step k+1 with

    θ̂_{k+1} = round_dtype( θ_k − lr·( m̂/(√v̂ + ε) + wd·θ_k ) ),  m̂ = β1 m_k/(1−β1^t), v̂ = β2 v_k/(1−β2^t)

i.e. the AdamW step of the optimizer state (θ_k, m_k, v_k) with gradient 0, t = steps taken + 1,
lr / wd / β / ε of each parameter's group (lr = the scheduler value for the coming update; NeMo RL
steps the scheduler after the optimizer), rounded to the parameter dtype. The expression and its
operation order are the research harness's `draft_master_fn("mom")`
(`research/sticky/adam_draft_gpu.py`), so the draft is bitwise the one the offline numbers were
measured with (`tests/unit/test_draft_weights.py`). θ̂ is a function of the optimizer state
before step k's gradient, so it is available as soon as step k starts.

**Exactness: the drafter is only a different proposal q.** Block verification (section 4) and
⊥-BV (14d) are exact for any proposal law q, provided q is the law the tokens were actually sampled
from and is known to the verifier. With engine-sourced q (section 14) both hold by construction:
log q of each drafted token is the engine's own log-prob of the forward that sampled it, and full q
rows are rebuilt from the engine's logged final hidden states with the drafting LM head. The target
p = π_{k+1} = F(θ_{k+1}; A) is untouched (θ_{k+1} is the real optimizer step). Every place that
assumed "the drafter holds θ_k" was checked:

| Place | Assumption before | With momentum draft weights |
|---|---|---|
| Engine-q rows (lazy residuals, residual sample) | LM-head stash of version k = θ_k's head, taken in `score_prev` at the end of step k | the head is stashed by `stage_draft_weights` (top of step k) while θ̂ is loaded on the learner, as version k (version k−1 is kept: its drafts are still verified during step k's rollouts); `score_prev` does not overwrite it. `[engine q check]` (logged log q == hidden × stashed head) fails if the two ever differ |
| Scorer q (`verify_source=scorer`) | recomputes q from a θ_k weight stash | rejected by validation (would be the wrong q) |
| Sticky anchors (14c) | A = plain drafter routes = plain θ_{k−1} | A = plain θ̂_k routes. R1: θ̂_k depends only on the state before g_k, never on θ_k's update outcome or on the draft. R2: anchors are the drafter's routes on the rollout's own prefix; R3: the learner applies the logged anchors, as before |
| Sticky continuation rounds | engine still at the drafting weights when `verify_sticky` runs; continuations draft + anchor with them | unchanged: the engine keeps θ̂ from the post-rollout load until the top-of-loop refits, which run after `verify_sticky`; the route check (`[route check]`) re-prefills on θ̂ and must match the logged anchors |
| Rollouts of step k (sampled from p_k = θ_k) | engine at θ_k during the step | θ̂ is staged in engine host memory at the top of the step and θ_k refit over it; the engine loads θ̂ (local host → device copy into the same parameter storage) only after the step's last rollout returns, then drafts. Drafted rollouts' plans were all published before (each rollout waits for its plan); chunked verification uses logged hidden states and stashed heads, not the engine's live weights |
| Tail decodes after a rejection (non-sticky) | on the refit engine (θ_{k+1}) | unchanged |
| Refit ordering | one refit per step (θ_{k+1} at the top of the loop) | two, both at the top of step k: θ̂_{k+1} (then staged to host by every engine rank), then θ_k |
| Learner training forwards during the swap | n/a | the learner's parameters hold θ̂ only inside `stage_draft_weights` at the top of the step, before `_run_one_step` starts any training dispatch (no lock: nothing else runs on the learner then); restore is the optimizer's own main → model copy + all-gather, bitwise θ_k (tested); grads/main params are never touched; the fp16 expert-weight cache keeps θ_k's casts (valid again after restore; no forward runs while swapped) |
| Prefix caching | — | engine prefix-cache hits on KV computed by another version change q, but q is still the engine's logged law (the hidden state the head is applied to is the one actually computed), so exactness holds; sticky routing requires prefix caching off anyway |
| Lookahead > 1 (cohorts) | drafts span versions | rejected (a two-step momentum draft is not built yet) |
| Mismatch check | learner vs itself under the same anchors | unchanged (p never depended on the drafter) |

**Implementation.**
- `draft_weights.py`: `momentum_draft_master` (the formula); `load_draft(optimizer, "momentum")`
  walks every DistributedOptimizer of the (Chained) optimizer and, on each rank, only the main-
  parameter shards that rank owns (dense and expert optimizers, any TP / PP / EP / DP layout: the
  shards are already partitioned by Megatron), computes the draft in chunks of `CHUNK_ELEMS`
  (16M) elements, rounds it into that shard's slice of the DDP parameter buffer (what the
  optimizer's main → model copy does after a step), then all-gathers the buffers over the data-
  parallel groups (`start_param_sync(force_sync=True)`, the post-step gather). fp32 model
  parameters, whose main shard *is* the buffer slice, are saved before and restored after.
  `restore_training` = restore those + `quantize_and_sync_model_params_from_main_params()`.
- Worker: `load_draft_weights(kind)`, `restore_training_weights()`.
- Precision-aware optimizer (`use_precision_aware_optimizer=true`, the NeMo-RL default): the inner
  optimizer's parameters are the model-parameter shards (views of the buffer) and TE FusedAdam holds
  the masters: fp32, scaled fp16, or (Megatron's default `store_param_remainders`) the bf16 parameter
  plus int16 low bits, reconstructed per chunk by `master_from_remainders` (high half = param bits −
  [remainder < 0], TE's round-up undone). TE writes the parameter inside its step, so there is no
  main → model copy to redo: the owned slice is saved before the draft is written and copied back
  (2 B per owned element), then the all-gather.
- Worker: `load_draft_weights(kind)`, `restore_training_weights()` (learner);
  `stage_engine_weights()`, `load_engine_weights()` (engine, `draft_weights.EngineStage`: one
  pinned host copy of the rank's parameters, allocated once; copies into the same parameter storage,
  so CUDA graphs stay valid; checksums under `THUNDERSYNC_MISMATCH_CHECK`).
- Driver: `SpeculativeGeneration.stage_draft_weights(refit)`: load → stash LM head (version k,
  keeping k−1) → `refit` (the existing `refit_policy_generation`, unchanged transport) → every
  engine rank stages its weights to host while the learner restores θ_k (also on failure).
  `load_staged_draft_weights()`: engine-only load, refuses a stage of another step, checks the
  checksums. Metrics `draft/*`.
- Loop: at the top of step k (after `verify_sticky`, before the θ_k refit and before any
  dispatch) `stage_draft_weights`, then the θ_k refit (also at step 0). The drafter task starts
  when the step's last rollout returns (instead of `draft_start_frac`), runs
  `load_staged_draft_weights` in a thread and drafts; it does not wait for the learner (no learner
  lock). Validation (`validate_draft_weights`): speculate, block, engine q, verify_on=learner,
  lookahead 1, no draft_after_verify, no fused_verify_train, distributed optimizer, Adam (TE
  FusedAdam under the precision-aware optimizer).

**Per-rank cost** (P parameters, learner world W = TP·PP·DP for dense / EP·ETP·PP·DP_exp for
experts; each rank owns ≈ P/W main elements, P/(TP·PP·EP-share) model elements) [derived]:
- compute: ≈ 10 elementwise flops per owned element (≈ P/W), memory-bound: reads master, m, v
  (12 B) and writes the 2 B param → ≈ 14·P/W bytes of HBM traffic, twice-ish counting the
  restore copy (6·P/W B). Measured on one GB200 GPU (`thundersync_work/research/momdraft/bench_draft.py`,
  unfused torch ops, 16M-element chunks): 15 ms per 1e9 owned elements, 0.42 GB extra
  [measured]. A 500B model on 512 learner GPUs owns ≈ 1e9 elements per rank → ≈ 15 ms
  [derived].
- memory: chunk temporaries ≈ 24 B × min(CHUNK_ELEMS, largest owned shard) ≤ 0.42 GB [measured]
  (independent of model size), plus a copy of the rank's fp32 *model* parameters (norm weights;
  ≈ 0 for a bf16 model), plus the existing LM-head stash (V/TP × d × 2 B, two versions). Under the
  precision-aware optimizer also the owned bf16 slice (2·P/W B; ≈ 2 GB at 1e9 owned elements) and,
  only for bf16/fp16-stored moments, their fp32 view per shard. No copy of the model, master or
  moments on the learner; nothing is gathered except the parameters themselves.
- engine: one pinned host copy of the rank's parameters (P_engine_local × 2 B; e.g. ≈ 16 GB per
  rank for a 1 TB bf16 model over 64-way engine model parallelism, ≈ 128 GB host per 8-GPU node),
  one device → host and one host → device copy of them per step (≈ 0.13 s per 16 GB at the
  125 GB/s pinned H2D measured on this node [derived]), no collective.
- communication: two extra parameter all-gathers over each DP group (θ̂, then θ_k), each equal to
  the gather Megatron already does after every optimizer step (2·P_local·(DP−1)/DP bytes per rank,
  P_local = model elements on the rank); plus one extra engine refit per step through the
  existing sharded refit path (model bytes per engine rank; with the sparse-delta refit of
  `agent_system_structure.md` this becomes ~0.03 B/param).
- critical path: the stage runs at the top of the step while learner and engine are idle anyway
  (between `verify_sticky` and the θ_k refit): + load + stash + one refit + engine D2H ∥ restore
  per step (measured below). It adds no synchronization beyond the refit that already exists; the
  post-rollout load is engine-local (one Ray fan-out to the engine ranks, no collective, no learner),
  so nothing grows with the learner's world size.
- drafting window: one weight set per engine, so θ̂ drafts cannot share the engine with the
  step's own θ_k rollouts; drafting runs from the last rollout to the optimizer step (the
  host → device copy, ~0.1 s, is the only delay) instead of from `draft_start_frac`. Drafting while
  the rollout tail decodes needs a second weight version on the engine (a θ̂ replica, or
  per-request weight versions via a sparse delta): not built yet.
- no fresh rows (built, sticky routing): after `verify_sticky` every drafted rollout is complete;
  when every rollout of step k is served by a complete plan (`plans_cover`; measured
  `null_fresh_slots` is 0 in recent runs), no rollout decodes on the engine, so the θ_k refit and the
  host stage are skipped: the stage's refit leaves θ̂_{k+1} live and drafting starts at the step's
  first rollout (`draft/direct` = 1). Exactness: rollouts are the plans either way; only the drafts'
  start time changes, and it depends on step k's outcomes, which precede drafting (a draft's cut
  points only need to be independent of that draft's own tokens). Saves one refit per such step.
- remaining waits (not built yet; critical-path scout #6): (1) with fresh rows, drafting still starts
  only after the step's *last* rollout, so one long fresh decode delays all next-step drafting by up to
  a full decode (the learner-lock wait is gone). Design: at `draft_start_frac` the engine loads θ̂_{k+1} and starts
  drafting; every step-k rollout still decoding continues on the engine as a q = θ̂_{k+1} draft that
  the learner block-verifies in pieces against p_k (the streamed-continuation machinery of 14c:
  rejection → residual sample → restart from the new prefix). Exact for any q (section 4); such rows
  are never drafted rows ("self" anchors under sticky routing), so their target is plain θ_k. Cost:
  learner verification forwards of those tail tokens (it trains them anyway) and a rejection rate set
  by θ̂_{k+1} vs θ_k (one optimizer step apart). (2) The stage is on the critical path at the top of
  the step. Design: stage during the continuation phase instead — the engine receives θ̂ into its host
  staging (the receive writes host buffers, not the live θ̂_k the continuations decode with) and the
  learner computes θ̂ into a separate bucket buffer between verification pieces (so its parameters stay
  θ_k for verification), or a sparse θ̂ − θ_k delta (~1% of elements change per step at L4 [measured
  below]) applied on the engine. Per-rank cost unchanged except one bucket of learner scratch.

**Measurements.**
- End to end exact [measured, 2026-10-13 20:36]: `DRAFT_WEIGHTS=momentum scripts/bi_tp/sticky_moe.sh`
  (tiny MoE, EP=2, sticky δ 0.03, expert-gradient noise, 4 steps, precision-aware optimizer off,
  `draft_start_frac=1`): `[engine_q] PASS (4/4 steps)`, mismatch 0 every step, engine q exact,
  route check exact (39368/39368, 25945/25945, 24712/24712 tokens: the anchors logged on θ̂ equal a
  fresh θ̂ prefill), sticky rule active, weights change every step, draft refit every drafting step
  with `draft/changed_frac` 0.09–0.13% of elements from step 1 (0 at step 0: no momentum yet).
- TP=2 EP=2 (SP) sticky + momentum (`TP=2 DRAFT_WEIGHTS=momentum sticky_moe.sh`): `[engine_q] PASS (4/4)`,
  `[sticky] OK`, mismatch 0 [measured].
- ⊥-BV (14d) + sticky + momentum (`DRAFT_WEIGHTS=momentum scripts/bi_tp/null_bv_moe.sh`, K 1.5, M 2):
  `[engine_q] PASS (4/4)`, `[null_bv] OK`, mismatch 0; accepted .971/.996/.985, ⊥ fraction .32–.33, tail-decode
  fraction .016–.031, 1.48 drafts per output, no fresh slots [measured].
- Draft-refit cost on this toy (2+2 GPUs): load 22 ms, refit 138 ms (same as the regular refit,
  `time/refit` 124 ms), restore 4 ms, peak extra learner memory 0.41 GB [measured]. Draft compute
  alone: 15 ms per 1e9 owned elements, 0.42 GB (`bench_draft.py`) [measured].
- Acceptance, paired (same script, `draft_start_frac=1` in both arms): `spec/accepted_frac` steps 1–3
  θ_k .936/.965/.967 vs momentum .967/.954/.980 [measured]; no consistent difference on this toy (3 steps,
  different draft lengths, drafter/target gap dominated by injected random expert-gradient noise that momentum
  cannot predict). The acceptance evidence is ledger T on real layers with real Adam; a real-model paired run
  (L4 arms queued) is next. ⊥-BV paired: θ_k accepted .976 / ⊥ .326 / tail .039 vs momentum .984 / .328 / .026.
- Drafting window [measured]: momentum arms drafted 2–3.5x fewer tokens than the θ_k arms (⊥-BV 42k vs 150k):
  the drafter first waits for the learner lock, i.e. for the in-flight learner dispatch, before the refit.
  Fixed (2026-10-14): θ̂_{k+1} is staged at the top of step k and loaded engine-locally after the rollouts,
  or refit directly when no rollout decodes fresh (see Implementation). Paired after the fix (ts-integration,
  tiny MoE EP=2, sticky, async scheduling, 4 steps, both arms `draft_start_frac=1`) [measured]: both PASS
  (engine q exact, mismatch 0, route check exact every step); draft tokens per step current 119096 / 30080 /
  28273 vs momentum 119096 / 26932 / 33354 (equal within noise; before: 2–3.5x fewer). Momentum steps 1–2
  took the direct path (stage .16 s: load .02 + refit .135; no θ_k refit); step 0 the staged path (engine
  device → host stage .11 s for 1.39 GB per rank).

**Not built yet.** Megatron-FSDP, fp8/fp4 param gather, a two-step draft for lookahead > 1, the
stage off the critical path (host-staging receive during the rollouts, or a sparse delta), sticky
rounding together with momentum drafts (its θ_n anchor store would have to hold θ̂).

## 14g. Sticky fp16 rounding toward a θ_{k−1} anchor forward (`thundersync.sticky_rounding`)

*Opt-in, off by default (`sticky_rounding: false`); not part of the safe default recipe, intended for later
convergence runs. Requires sticky routing (14c). Unit-tested (CPU); learner lockstep checked on a real Megatron
MoE at TP=1, TP=2 (sequence parallel) and EP=2: bitwise equal to the research emulation for logits and every
gradient. End-to-end 4-GPU exactness run (`scripts/bi_tp/sticky_rounding_moe.sh`): queued, not run yet.*
Design note: `thundersync_work/research/STICKY_ROUNDING.md`; implementation notes and measurements:
`thundersync_work/research/agent_sticky_rounding_impl.md`; code: `nemo_rl/models/megatron/sticky_rounding.py`.

**Policy.** The fp16-GEMM recipe (14b) rounds fp32 activations to fp16 at fixed sites: every linear-layer input
(attention QKV and output projections, dense MLP fc1/fc2, shared experts), the q/k/v casts after the q/k norms
(the KV cache), and the expert GEMM inputs (fc1: the token's MoE input; fc2: SwiGLU output × routing prob).
The policy of step k replaces round-to-nearest at every site by

    r(x; a) = a          if prev16(a) < x < next16(a)
            = RTN16(x)   otherwise

with a = the value the anchor forward (plain θ_{k−1}, same tokens, same numerics) produced at that site. Edge
semantics (unit-tested): r is the identity on fp16-representable x; a NaN anchor means "self" (RTN); signed zeros
follow the rule (a = −0 holds a tiny positive x); next16(65504) = inf, so an anchor 65504 holds any finite x above
it (as the emulation). The error stays below one fp16 ulp for x inside the fp16 range. The LM head has no fp16
site under the default fp32 head; with `lm_head_precision: fp16` its input is rounded to nearest (anchor = self).

**Anchor schedule.** Exactly sticky routing's (14c): a token is rounding-anchored iff it has a valid route anchor
at every local MoE layer (drafted rollouts up to the cap c), else self (RTN = the plain recipe). Expert sites are
matched by (expert, token): the fc1 anchor is per token (the anchor forward's MoE input), the fc2 anchor of
(t, e) is the anchor forward's value if θ_{k−1} routed t to e, else self. So anchors depend only on θ_{k−1}, the
rollout's prefix and the content-independent schedule (R1, R2); they never depend on θ_k, on the draft's future
tokens, on τ or on verification noise.

**Exactness.** (R1) p_k(·|prefix) = F(θ_k; anchors(prefix)) is a fixed function of the prefix for given θ_{k−1},
θ_k; its gradient is backprop through F with anchors as constants and a straight-through cast (as for RTN, whose
true derivative is also 0 a.e.). (R2) Anchored positions are fixed before any token is sampled. (R3) Every
forward that computes p_k on anchored tokens runs on the learner and computes the same F: the verification
forward, the training forward and the log-prob forward all run the lockstep below with the same θ_{k−1} snapshot,
and the anchor forward is the batch-invariant training forward, so a token's anchors do not depend on the batch
it is verified or trained in, nor on later tokens (causal). **The engine computes nothing new**: drafts are
sampled from q = the plain θ_{k−1} engine forward (the drafter is the anchor forward); after a rejection the
replacement token is sampled by the learner from the residual of p_k; continuations of drafted rollouts are
θ_{k−1} drafting rounds verified by the learner (sticky routing's continuation mechanism, 14c), so the engine
never samples an anchored position from π_k; rollouts never drafted are decoded on the refit engine at θ_k with
self anchors = RTN = the engine's plain recipe. Hence block verification against p_k is exact, as in 14c.
Checked bitwise on GPU (single replica): lockstep == the research emulation (call-order record/apply, all three
site kinds) for logits and all gradients; all-self mask == plain forward (logits and gradients); in a
half-anchored batch the self row equals the plain forward and the anchored row equals the full lockstep.

**Learner implementation.**
- θ_{k−1} store (`ThetaStore`): in `finish_train_step`, right before the optimizer step, every rank copies its own
  local 16-bit parameters (its TP/PP/EP shard as the model holds them; with the distributed optimizer these are
  the gathered model params of the shard, not optimizer state) into pinned host buffers (async D2H on a side
  stream, waited before the step). No gathers, no cross-rank traffic.
- Lockstep (`_gpt_forward`, `_embedding_forward`, `_layer_forward`): a GPT forward on the learner with armed
  sticky-routing replay targets (exactly the policy forwards) opens an anchor stream. The embedding runs at θ_k
  and at θ_{k−1} (the anchor residual stream); each transformer layer first runs at θ_{k−1} on the anchor stream
  (`torch.func.functional_call` with the streamed weights, `no_grad`, recording fp16 values per site), then at
  θ_k on the live stream applying r(x; a) per site. The θ_{k−1} weights of unit i+1 are copied H2D on a side
  stream while unit i runs; at most two units are resident.
- Sites: `fp16_gemm` calls a site hook at each fp16 cast (`Fp16Linear` input keyed by module, q/k/v keyed by
  (attention, index)); MoE: the dispatched copy of the MoE input is rounded in token space before dispatch (the
  router keeps the unrounded fp32 input), and a 4-column fp32 key (source rank, token index, anchored) is
  dispatched with the layer's own routing map through the same dispatcher, so fc2 anchors are matched by key at
  the expert rank under any EP layout (`moe_preprocess`, `experts_fc2_input`). Expert weight fp16 caching is
  bypassed for the transient θ_{k−1} weights.
- Same sticky routes: the anchor pass replays the same anchors A (F(θ_{k−1}; A) = A = the plain θ_{k−1} routes
  under BI), so the anchor forward's expert dispatch is the drafter's; the policy pass dispatches F(θ_k; A).
- Not built yet (rejected by `validate_sticky_rounding`): pipeline parallelism, context parallelism, activation
  recompute, fused verify+train, flex/DeepEP dispatchers, shared-expert overlap, hybrid shortcut-MoE blocks and
  hyper-connection layers (`lockstep_units` raises for any decoder layer class without a lockstep).
- Hybrid models (Nemotron-H: `HybridModel` / `HybridStack`): `HybridModel.forward` opens the policy forward like
  `GPTModel.forward`, and `MambaLayer.forward` runs the same two-pass lockstep as `TransformerLayer` (attention, MoE
  and MLP layers of the hybrid stack are `TransformerLayer`s). A Mamba layer's rounding sites are its in_proj /
  out_proj inputs (`Fp16Linear`); conv1d and the SSD scan run in fp32 under the recipe and have no fp16 site. Before
  this, a hybrid model with `sticky_rounding: true` took θ_{k−1} snapshots but never opened a policy forward (only
  `GPTModel` was patched): the policy was silently the plain recipe. The Mamba mixer reads conv1d / dt_bias / A_log /
  D through its `MambaContextParallel` helper, which holds the Parameter objects; the anchor pass rebinds such
  captured parameters to the streamed θ_{k−1} tensors (`_captured_params_swapped`). Checked on GPU (Nano L8, BI, 1
  GPU): lockstep logits == research emulation bitwise, all-self == plain, half-anchored rows bitwise; gradients are
  not bitwise-checkable on the hybrid (the plain Mamba SSD backward is non-deterministic). 4-GPU E2E: queued.
  Measured v cut (L8 harness, momentum drafts): ×.42 / .23 / .14 / .16 at Adam steps 2 / 4 / 8 / 16
  (thundersync_work/research/agent_raise_acceptance.md).

**Scaling (per rank; P_r = the rank's 16-bit parameter shard in bytes, U_max = its largest lockstep unit,
T = tokens of a micro-batch on the rank, d = hidden size, k = top-k, n_mb = policy-forward micro-batches per
step).** [derived unless marked]
- Host memory: P_r (pinned), = the rank's 16-bit model shard; nothing else. Ultra-scale example: 500B params over
  64 model-parallel ranks ≈ 16 GB/rank (GB200 Grace: ~480 GB LPDDR per 4 GPUs → fits next to the optimizer's
  host-resident state only if that budget is planned; the node-level host budget must include it).
- GPU memory: θ_{k−1} weights of one unit during its anchor pass; the next unit's H2D copy is issued when the
  anchor pass ends and overlaps the policy pass, so ≤ 1·U_max is resident outside that overlap (≤ 2·U_max at the
  boundary) + one layer's anchors (fp16, ≈ the
  layer's fp16 site inputs for T tokens: for an MoE layer ≈ T·(d + d_qkv + d_attn_out) + T·k·(d_ffn_expert)
  elements × 2 B, plus 16 B keys per routed row) + the anchor residual stream (T·d·4 B) + the anchor layer's
  inference working set (freed before the policy pass of the layer). Measured on real Qwen3-30B-A3B layers (L4,
  1 GPU, T = 4096 / 8192 tokens, U_max = 1.19 GB): peak allocated +32 / +64 MB (0.03 / 0.05 U_max) for a no-grad
  verification forward and +0 for forward+backward, because the peaks (LM-head logits, backward) are elsewhere [M].
- H2D traffic: P_r per policy-forward micro-batch, one copy per unit (every micro-batch walks all layers), i.e. n_mb·P_r per step;
  hidden behind compute when t_fwd(T) ≥ U_max / B_H2D per layer, B_H2D ≈ 400 GB/s per direction on NVLink-C2C
  [hypothesis: to measure]. This scales with n_mb: the levers are (a) larger policy micro-batches (one stream per
  micro-batch), (b) a sparse θ_k − θ_{k−1} delta held on GPU and applied per layer (bytes ∝ fraction of 16-bit
  weights changed per step; to measure at real lr), (c) processing several micro-batches per streamed layer
  (needs schedule support). Until measured at scale this is the main cost risk, not a blocker.
- Compute: one extra forward over the policy-forward tokens: +100% on forward-only verification passes, ≈ +33% of
  fwd+bwd on training passes (activation-free: inference-mode) [derived]. Measured (L4, 1 GPU, B = 4 × 1024):
  GPU time on the compute stream of a no-grad (verification) forward 27.5 → 54.7 ms (2.0x, as derived); the θ_{k−1}
  stream is 5 H2D copies (one flat pinned buffer per unit, expert weights held as fp16) = 44.6 ms on the copy
  stream (≈120 GB/s, overlapped); wall time in this 4-layer harness is CPU-launch bound (plain 0.12 s wall for
  27 ms of GPU work): fwd+bwd 0.42 → 0.60 s, no-grad forward 0.12 → 0.30 s [M].
- Communication: the anchor pass repeats the layer's TP collectives and EP all-to-alls (forward only), and each
  MoE layer adds two key dispatches per pass of 16 B per routed row (≈ 4/d of a hidden dispatch) with the same
  tokens-per-expert synchronization MCore's dispatch already performs. No global synchronization is added;
  all collectives are within the groups the layer already uses, in the same order on every rank (EP lockstep
  dummies run the anchor pass too).
- PP (design, not built yet): the anchor residual stream travels with the stage activation (the stage output
  becomes the pair (h, h̄); h̄ carries no gradient), i.e. 2x forward p2p bytes and the same schedule; interleaved
  virtual stages index the store per model chunk. No extra synchronization.
- Hybrid (Mamba-2) layers (design, not built yet): their linear sites (in_proj / out_proj inputs) are Fp16Linear
  sites already; the layer wrapper applies to Mamba layers unchanged (the training forward scans the whole
  sequence, no state cache); the SSD scan and conv have no fp16 cast in the recipe, so no extra sites
  (agent_nemotron_hybrid.md for the scan's TF32 numerics).

**Interaction.** Sticky routing: the anchor pass already computes the plain θ_{k−1} routes on the learner layer by
layer, so it can also supply sticky-routing anchors without route transport (the policy pass of layer l needs only
layer l's anchor routes, available right before it) — a candidate replacement for driver/object-store route
transport; requires the drafter's routes to equal the learner's plain θ_{k−1} routes bitwise, which BI and the
route check already assert. ⊥-BV: none (target π_k per draft). Momentum draft weights (separate branch): the
anchor forward stays the plain θ_{k−1} forward (the policy definition), independent of the drafter's weights.

**Measured value (emulation, ledger T).** Qwen3-30B-A3B 16 layers, steady-state AdamW: sticky rounding on top of
sticky routing lowers Var_q(log p/q) 1.3x for θ_{k−1} drafts and 3–10x for momentum / oracle drafts (all three
site kinds needed); ~20% of activation elements are rounded toward the anchor at steady state.

## 14h. Learner memory at scale (2026-10-14 multi-node OOMs)

**What failed.** The first multi-node runs of ts-integration (Qwen3-30B-A3B, 48 layers, TP1 PP1 EP8 and PP2 x EP2,
micro-batch 8 x 2048 = 16384 tokens per rank) died with CUDA OOM inside the *training* forward of step 0
(`stream_accumulate` -> `train_microbatches` -> MoE experts), and the PP run then hung on idle GPUs. Full
Nemotron-3-Nano at EP=2 (single node, optimizer offload, micro-batch 1) also OOMed in its first training forward.

**Root cause [M] (thundersync_work/research/agent_oom.md, allocator snapshots).**
- No activation recompute: a training micro-batch kept every layer's activations, 166 KB per token per Qwen3 layer:
  MoE rows 95 KB (k = 8 rows per token x 11.9 KB: fp32 fc1 output saved by SwiGLU, fp16 fc1 / fc2 inputs) + dense
  71 KB. Per rank: S + L·T·(a_d + k·ρ·a_r), ρ = the rank's received-row imbalance (1.19 / 1.55 measured on EP8 ranks
  7 / 2 from the OOM allocation sizes; the router is frozen, so ρ is whatever the data gives). With S = 38 GB (step 0:
  bf16 params + fp32 grads + fp16 expert cache), the most loaded rank needs 3.0-3.6 GB per layer and runs out after
  40-48 layers: the observed ~145 GB of transient memory, OOM inside `routed_experts_compute`.
- The fp32-input LM head (`fp32_gemm`, hi/lo split) held 3 logits-sized fp32 buffers at once (3 x 9.3 GiB), and the
  TP=1 batch-invariant log-probs kept the log-softmax output for backward (a 4th).
- Nano EP=2: static. 155.9 GiB was allocated before the first training call (bf16 params + fp32 grads + the
  DistributedOptimizer fp32 master shard, which optimizer offload leaves on GPU), plus the persistent fp16 expert
  cache (2 B per local expert parameter, 29.4 GB). EP=8 is the Nano layout.
- Hang: on a learner exception the driver called `stream_abort_step` on every rank with an unbounded `ray.get`;
  ranks blocked in a collective with the failed rank (PP send/recv, EP all-to-all) never answered.

**Fixes (all bitwise: mismatch 0 and engine-q exact on Qwen3 L4/L16 and Nano L8, EP=2; param checksums identical
with and without recompute).**
- Activation recompute in the learner (`engine_q_tp.sh` ACKPT): `selective` (default at PP=1; Megatron selective
  recompute of the MoE block - router, dispatch, experts, combine - which then keeps only its fp32 input; routes
  replay exactly through R3 REPLAY_BACKWARD and the sticky wrapper) or `full` (default at PP>1; every layer keeps only
  its fp32 [T, hidden] input). Sticky rounding still requires `false` (its anchor forward is not recomputed yet).
- `fp32_gemm` on CUDA: row-chunked (4096 rows), in place: the output is the only [M, N] buffer.
- TP=1 BI log-probs (`_LocalVocabLogprob`): save only the logits; backward recomputes the softmax per chunk.
- fp16 expert weights for the training GEMMs are cast just in time per layer and the 4 most recent expert linears
  (two MoE layers) are kept, buffers reused: 2 x 0.3 GiB at Nano EP=8 instead of 7.35 GB (29.4 GB at EP=2). Cost
  [M] +0.33 ms per MoE layer forward at Nano EP=8 shapes (+15% of the expert forward GEMMs; +1.2 ms at EP=2).
- `StreamingLearner.abort` waits at most 60 s; the driver exits (`os._exit(1)`) on any training exception, so Ray
  kills the job's actors and the job ends (validated: both OOM reproductions exited within ~1 min).

**Per-rank learner budget (T = tokens per micro-batch, L = layers on the rank, H hidden, V vocab, n = micro-batches
in flight on a PP stage (1 at PP=1, up to PP under 1F1B)).**
peak = S + n·L·T·c + (one recomputed layer: T·(a_d + k·ρ·a_r)) + loss (2·T·V/TP·4 B + chunk temporaries), where
S = (2 + 4) B per local parameter + optimizer state (precision-aware FusedAdam: 10 B per shard element, allocated at
the first step; plain DistributedOptimizer: 12 B) + the two-layer fp16 expert buffers, and c = 4H B (full) or
a_d + 4H B (selective; a_d measured 71 KB Qwen3, ~85 KB per Nano Mamba layer). Measured [M] L16 EP=2, T = 16384:
transient 72.6 GB (none) -> 47 GB (selective) -> 26 GB (full) for Qwen3; 48.7 -> 37.2 GB for Nano (selective).
- Qwen3-30B-A3B EP=8, T = 16384, L = 48, after step 1 [D]: S = 69.5 GB (31 GB at step 0); selective peak ~166 GB,
  full ~104 GB (184 GB GPU) -> selective.
- Qwen3-30B-A3B PP=2 x EP=2 [D]: S = 125 GB; selective ~193 GB (2 micro-batches in flight) does not fit, full
  ~159 GB -> full (submit scripts default ACKPT=full at PP>1).
- Nemotron-3-Nano EP=8 (52 layers: 23 Mamba, 23 MoE, 6 attention; plain DistributedOptimizer) [D]: S = 83 GB after
  step 1; selective ~153 GB, full ~114 GB -> selective fits.
- Ultra [H: ~470B expert + 30B dense params, hidden ~8k, ~100 layers, V 131k; e.g. TP=4 (SP) PP=4 EP=64]: S ~45 GB
  per rank; full recompute keeps 25 layers x T x 8 KB / TP x 4 in flight = 13 GB at T = 16k (54 GB for one 64k
  sequence), selective ~106 GB at T = 16k -> full at PP>1. Every term is per micro-batch tokens and per rank shard;
  none grows with the global batch, and EP enters only through ρ of one recomputed layer.

**Learner batch-invariant unpermute: fused (`policy.megatron_cfg.fused_bi_unpermute`, default on).** The NVLS expert
path (`NVLSAllGatherVDispatcher`, AllGather-V + ordered ReduceScatter-V) is the engine's only: built for
`transformer_impl=inference_optimized`, selected under `InferenceMode`, no backward, symmetric buffers of EP x tokens
per rank (EP=64, 16k tokens, hidden 8k: ~51 GB). The learner (training and its no-grad forwards) runs the AlltoAll
dispatcher, whose Megatron `moe/batch_invariant.unpermute` did EP x top-k eager [T, H] index_select / fp64 passes
forward and built EP x k dense [T·k, H] gradients backward. `nemo_rl/models/megatron/bi_unpermute.py` replaces it
(patching `moe_utils.batch_invariant_unpermute`): per token the K slots are visited in (EP rank, slot) order, fp64
per-rank partials are rounded to fp32 and added in rank order (fp64 + one rounding for `multimem`) in one Triton
kernel; backward writes g + 0.0 to each slot's row. Same operations on the same values: skipped +0.0 terms never
change a sum started at +0.0, and the eager backward is +0.0 + g per row. O(T·k·H), independent of EP.
- Bitwise [M]: output and input gradient equal Megatron's (int32 bit patterns, signed zeros, cancellations, skewed
  routing) at EP 1/2/4/8/64, top-k 1/2/6/8/22, hidden 2048-8192, ordered and multimem (23 GPU tests); E2E Nano L8
  (sticky + momentum + ⊥-BV; STICKY=0) and Qwen3 L4 (sticky + momentum; ⊥-BV) EP=2: mismatch 0, engine-q exact.
- Per MoE layer forward + backward, 16384 tokens per rank [M, one GB200, EP emulated]: Nano EP2 21.2 -> 0.72 ms,
  Nano EP8 85.8 -> 0.68 ms, Qwen3 EP8 96.2 -> 0.69 ms; Ultra-like EP64 x top-22 x hidden 8192 at 2048 tokens
  1774 -> 0.87 ms (linear in tokens: ~14 s -> ~7 ms at 16384). Peak extra memory 2.6 -> 1.15 GiB (Nano), 3.0 -> 1.44
  GiB (Ultra-like, 2048 tokens): the output, its input gradient and [T, k] int32 maps. Per 16k-token micro-batch this
  removes ~2.0 s (Nano EP8, 23 MoE layers) / ~4.6 s (Qwen3 EP8, 48 layers) of learner time, more with MoE recompute.

## 14j. Cohort lanes: two weight versions in the engine fleet (`thundersync.cohort_lanes: 2`)

*Opt-in tool, not the default (`cohort_lanes: 1` = one engine; the plan of record is one engine with
section 14k's scheduling). Built and unit-tested (CPU, fake engines:
`tests/unit/test_lanes.py`); 4-GPU end-to-end results below. Notes, measurements and resume
section: `thundersync_work/research/agent_lanes.md`.*

**Why.** With sticky routing a cohort's continuation streams must decode on the version that
drafted it (section 14c), and a single-version engine holds that version until every
continuation of cohort k is verified. Drafting of cohort k+1 therefore starts only after
cohort k's tails, and the drafts get only the learner's training time as their window
(`grpo_loop.py`, sticky branch of the main loop). Two lanes remove that coupling: cohort k+1
is drafted on the other lane from t_k, the top of step k. Window bound ≈
(L_max·τ + T_verify + g)/2 instead of ≈ L_max·τ [derived, `agent_fp_async_floor.md` §1, §5];
reconciled model: lanes ≈ 1.7–1.9x faster than today's single-version plan of record and
0.88–0.97 of a non-exact two-cohort async-1 on the same fleet [derived,
`agent_fp_reconcile.md` §2–3, `fp_reconcile/model.py`].

**Construction (option 1: lanes are disjoint engine replicas).** A lane is one
MegatronGeneration (its own MCore DP coordinator) over its own GPUs, one or more full TP/PP/EP
replicas, holding one weight version at a time (`lanes.py`). Lane 0 is the engine `grpo.setup`
builds on `policy.generation.colocated.resources`; `build_engine_lanes` adds lane 1 with the
same placement shape and config (hence the same batch-invariant forward). Each lane has its own
refit communicator: the learner keeps one process group per lane (`keep_refit_group`) and selects
the lane's group before a refit (`select_refit_group`, `lanes.refit_lane`), through the same
bucketed packed-collective refit as one engine. Each lane has its own `SpeculativeGeneration`
(so its own draft-stream registry, route log, hidden log, ⊥-BV slot table, plans and route
blocks). Step k (`CohortLanes.step`, `grpo_loop._train_on_lanes`):

1. the drafting lane (k+1 mod 2) takes the drafter D_{k+1} of cohort k+1 (θ_k, or θ̂_{k+1}
   with momentum: learner load, refit of this lane only, learner restore; no host stage),
   is pinned to cohort k+1 and starts drafting it;
2. the cohort lane (k mod 2), still at D_k, verifies cohort k's drafts and decodes + verifies
   every continuation stream (`verify_sticky`, unchanged) — concurrently with step 1's drafts
   on the other GPUs — and is unpinned when every rollout of cohort k is complete;
3. only if some rollout of step k has no complete plan (step 0; a draft dropped for missing
   logs) the cohort lane refits θ_k and decodes it there;
4. rollouts are served from the cohort lane's plans, the learner trains, cohort k+1's drafts are
   cut at the usual deadline (their streams keep running on their lane, section 14c).
Roles alternate. Without fresh rows exactly one lane is refit per step (each lane every two
steps), as with today's direct momentum path.

**Option 2 (every engine rank holds two weight copies and two request pools) — not chosen.**
Per-rank costs, W = engine weight bytes, r = GPUs per replica, s = state per request slot
[derived]:

| per engine rank | one engine (today) | option 1: lanes | option 2: two copies |
|---|---|---|---|
| weights | W/r | W/r | 2W/r |
| request-slot state (KV, Mamba SSM, BI replay) | s per slot | s per slot (same slots per GPU) | s per slot, two pools in the HBM left after 2W/r |
| refit bytes per step | W-stream on every rank | W-stream on the drafting lane's ranks only (each rank every 2 steps) | W-stream on every rank, received while the other copy decodes |
| refit stall | whole fleet paused | drafting lane only; cohort lane keeps decoding tails | none, but receive contends with decode |
| tail decode τ | — | unchanged | mixed-version batches re-read a second weight set per layer: +20–25% τ in the launch-bound tail (fp_async_floor §3) |
| collectives across lanes | — | none | — |
| minimum fleet | 1 replica | 2 replicas | 1 replica |
| load balance | — | static half per role; idle when tails are short (measured, below) | dynamic |

- Nemotron-3-Nano (31.6B; engine weights 58.8 GB experts / EP + 4.4 GB dense per rank, section
  14c): 8 engine GPUs as 2 lanes × EP=4: 19.1 GB/rank; option 2 at EP=8 on the same 8 GPUs:
  2 × 11.8 = 23.5 GB/rank. Lanes at EP=8 (16 engine GPUs): 11.8 GB/rank. Slots per GPU and
  their 162 MiB + 6 KiB/token of state are unchanged.
- Ultra-class (~1 TB fp16 engine weights, 16-GPU replicas, 186 GB HBM) [H shapes]: lanes keep
  62.5 GB/GPU of weights; option 2 needs 125 GB/GPU, leaving ~55 GB for state (≈ 0.86–1.26 GB per
  row: ~45–60 rows/GPU instead of ~100–120). At R = 128 engine GPUs each lane has 4 replicas.
- Momentum drafts with lanes need no engine host stage (section 14f's `EngineStage`: a pinned host
  copy of each engine rank's parameters, ~12 GB/rank at Nano EP=8): the drafting lane decodes
  nothing at θ_k, so θ̂_{k+1} is refit into it directly.

**Exactness (unchanged argument; only placement and start times change).** Section 14c's
scheme is exact given R1 (anchors are plain top-k of the drafting version D_k on each rollout's
own prefix, D_k fixed before θ_k's gradient exists), R2 (continuations after a rejection, and
the undrafted remainder, decode with D_k's plain forward and log D_k's routes) and R3 (the
learner verifies and trains with the logged anchors and logged q). Lanes keep all three:
- *One version per cohort.* Cohort k is pinned to lane k mod 2 from the start of its drafts
  (the lane was refit to D_k right before, `CohortLanes.pin` records the version) to the end of
  its last continuation (`unpin`). `CohortLanes.refit` raises on a pinned lane and `check`
  raises before completion if the lane no longer holds the cohort's version. So every draft
  token, every continuation token and every logged route of cohort k comes from D_k, as with one
  engine (which guarantees it by refitting only after `verify_sticky`).
- *No cross-lane leakage.* Each lane's `SpeculativeGeneration` sends requests only to its own
  engine (rank 0 of its DP coordinator) and pops routes / hidden rows / log-probs only from its
  own workers; cohort k's draft tuples, streams and route blocks exist only in lane k mod 2's
  object. The other lane never receives a cohort-k request: continuation streams of cohort k
  never run on lane-(k+1) weights.
- *R1 for D_{k+1}.* θ_k and θ̂_{k+1} = AdamW(θ_k, m_k, v_k; g = 0) exist at t_k and do not depend
  on θ_{k+1}. Drafting earlier does not change what the drafts are proposals of; they are
  verified against p_{k+1} with their logged q (any proposal law fixed before θ_{k+1} is valid),
  cut at the same deadline (before the optimizer step).
- *Fresh rows.* Rows without a complete plan decode at θ_k (p_k = F(θ_k; self)) on the cohort
  lane after it is unpinned and refit, as today.
- *Learner state.* Verification of cohort k needs the LM head of D_k (engine-sourced q). The
  momentum stage of D_{k+1} now runs before cohort k's verification; it stashes head version k
  and keeps k−1 (`stash_head(keep=[k−1, k])`, unchanged code), and restores θ_k bitwise before
  verification starts. With current drafts θ_k's head is stashed at the end of step k, after
  cohort k is verified.
- *⊥-BV per lane.* A prompt's K·G drafts are all drafted on one lane in a fixed order, its slot
  table lives in that lane's object; pooled verification is unchanged.
Tested on fake engines (`test_lanes.py`): with current and momentum drafts and fresh-row steps,
every cohort is drafted, continued and served on lane t mod 2 at its drafting version; the next
cohort's drafts start before the current cohort completes; a refit of the cohort lane before its
cohort completes raises; a cohort completing on a lane that changed weights or never drafted it
raises.

**Scaling.** Every per-rank cost above is per lane-local GPU and independent of the learner
size; nothing per token passes through the driver (routes, hidden rows and log-probs stay in each
lane's object-store blocks, section 14c); no collective spans lanes. The refit is the existing
transport with one communicator per lane: the packed collective, or the m2n refit (section 14q,
`refit_transport: mcore`, one native refit group per lane); the per-lane selection is a pointer swap
on each learner rank. Not built yet: per-lane selection for `refit_transport: nccl_reshard` (the
learner's per-PP-stage communicators would be kept per lane the same way), a multi-node lane launcher (each lane = whole
nodes of replicas; `build_engine_lanes` already takes multi-node placement shapes), lane
rebalancing (static halves now; `lanes/idle_frac` is the metric), lookahead ≥ 2 (a+1 lanes, the
version map of `agent_fp_async_floor.md` §2), and lanes without sticky routing (validated off).

**Metrics** (per step, seconds from the top of the step): `lanes/draft_start_s` (cohort k+1's drafts
start), `lanes/tail_end_s` (cohort k's last continuation verified), `lanes/overlap_s` = tail_end −
draft_start (> 0: drafting overlapped the tails; one engine is ≤ 0), `lanes/cohort_lane_idle_s`,
`lanes/drafting_lane_idle_s`, `lanes/idle_frac` (idle lane-seconds / (2 × step time)),
`lanes/fresh_refit`, `lanes/refits`. One-engine runs report `time/draft_start` (drafter start from the
top of the rollouts) for the same comparison.

**4-GPU end to end** (`scripts/bi_tp/sticky_moe.sh LANES=2`, 2 training GPUs + two one-GPU lanes,
EP=1; reference LANES=1 = one engine with DP=2 on the same two GPUs; tiny Qwen3-MoE d2048, BI, engine-sourced
q, sticky routing, expert-grad noise, 16 prompts x 8, 1024 tokens, 4 steps, one run per arm, 2026-10-14) [measured]:

| arm | mismatch / exact | engine q | route check s1/s2/s3 | accepted s1/s2/s3 | cohort lane | fingerprint held / moved | overlap s1/s2 | draft tok s2/s3 | iteration s1/s2/s3 |
|---|---|---|---|---|---|---|---|---|---|
| lanes, current | 0 / 1, 4 steps | 5 lines, 0 inexact | all exact (131072, 131072, 129893) | .903/.896/.920 | k mod 2 | every cohort / s2, s3 | +5.0 / +7.6 s | 117k / 117k | 7.12 / 9.65 / 5.42 s |
| one engine, current | 0 / 1 | 6 lines, 0 inexact | exact | .906/1.0/1.0 | — | — | −0.31 / −0.82 s | 128 / 128 | 7.52 / 9.38 / 8.60 s |
| lanes, momentum | 0 / 1 | 6 lines, 0 inexact | all exact (131072 each) | .903/.901/.933 | k mod 2 | every cohort / s2, s3 | +6.3 / +7.1 s | 117k / 118k | 8.58 / 9.26 / 8.01 s |
| one engine, momentum | 0 / 1 | 5 lines, 0 inexact | exact | .897/.998/.986 | — | — | −0.34 / −2.10 s | 3.8k / 2.1k | 5.50 / 9.18 / 9.06 s |

- *Each cohort on its version, read from the engine.* With `THUNDERSYNC_MISMATCH_CHECK` every lane's
  engine ranks checksum their own weights when a cohort is pinned and again when it completes
  (`engine_weight_checksum`, `LaneStepOps.fingerprint`); a difference raises. Both lane arms: held for
  every cohort, and different from the previous cohort's at steps 2 and 3 (`lanes/fingerprint_moved`).
  With current drafts the engine fingerprint of cohort k times the 2 learner replicas equals,
  bit for bit, the learner's checksum of θ_{k−1} (cohorts 2 and 3; `research/lanes/summarize.py`):
  every draft, continuation and logged route of cohort k ran on θ_{k−1}, the version its anchors
  name. Non-vacuity as in section 14c: weights change every step, steps ≥ 1 accept drafts, the
  sticky rule changed routes (max differs_from_plain .0054 / .0066).
- *Overlap.* With lanes cohort k+1's drafts start 0.13–0.17 s into step k and run during cohort k's
  5–7 s of tails (overlap +5.0…+7.6 s). One engine starts them 0.3–2.1 s after the tails end, inside
  the training window only: cohorts 2 and 3 got 128 (current) / 2–4k (momentum) drafted tokens and
  decoded the rest as continuation streams (85k–129k tokens) versus 117k drafted + 9–14k continuation
  tokens with lanes.
- *Step time: no difference measured on this setup.* Mean of steps 1–3: lanes 7.40 vs 8.50 s (current),
  8.62 vs 7.91 s (momentum); one run each, step 3 of a lane run drafts nothing, so these are not
  steady-state numbers. On this model the step is bound by the cohort's tail continuations (5–7 s for
  9–14k tokens on one lane), which lanes do not shorten; the window the lanes free (drafting in
  parallel with the tails instead of after them) is what the derived 1.7–1.9x rests on and needs a
  model whose drafting time is comparable to the tails. Open: 8-step repeats and Nemotron-3-Nano L8
  arms (cluster queue: `lanes-l2c-s8`, `lanes-l1c-s8`, `lanes-n2m-L8`, `lanes-n1m-L8`); why the lane's
  tail phase (6.5–7.3 s, momentum) is longer than one engine's (4.1–6.1 s) on similar continuation
  counts is not explained yet (one GPU per lane vs DP=2 for the tails is one candidate).
- *8-step repeat and Nemotron-3-Nano L8 (single-node cluster jobs, same layout, RL-lanes 633119f2;
  all PASS: mismatch 0, engine q exact, route check exact, sticky rule active).* Tiny MoE, current,
  8 steps: mean iteration steps 1–7 lanes 8.66 s vs one engine 8.65 s; lanes drafted 115–119k tokens
  per cohort vs 128–6.3k for one engine (cohorts 2–6), overlap +5.3…+7.5 s vs −0.3…−0.4 s; the
  engine fingerprint × 2 equals the learner's θ_{k−1} checksum for cohorts 2–7. Nano L8 (EP=1,
  momentum, fp16 recipe, 4 steps): lanes 6.78 / 18.16 / 11.58 s vs one engine 6.78 / 18.47 / 15.87 s
  (mean steps 1–3 12.17 vs 13.71 s); tails (time/verify) lanes 3.41 / 9.82 / 9.25 s vs one engine
  3.84 / 15.39 / 13.13 s; one engine drafted 766 / 1329 tokens for cohorts 2/3 (the rest decoded as
  122k / 121k continuation tokens), lanes 55k / 117k. EP=1 skips zero-advantage training (learner
  busy ≈ 0.1 s), so these runs do not exercise training overlap. Split lanes stay an opt-in tool; the
  measured critical path (`thundersync_work/research/agent_critical_path_measured.md`) is the tail
  and the serial post-tail chain, which lanes do not shorten; the one-engine work continues in
  section 14k (branch ts-one-engine).

## 14k. One engine under sticky routing: training streams over the tails

*Default with sticky routing (`thundersync.stream_training_over_tails: true`,
`thundersync.canonical_group_order: true`). Branch ts-one-engine; notes and measurements:
`thundersync_work/research/agent_one_engine.md`; critical-path measurements that motivated it:
`thundersync_work/research/agent_critical_path_measured.md`.*

**Why one engine needs two versions anyway.** Under sticky routing a cohort's continuation streams
(the tails) decode on its drafting version D_k (section 14c, R2): their anchors are D_k's plain
routes. Drafts of cohort k+1 need D_{k+1}. So one engine holding one version serializes tails →
refit → next drafts for any draft weights; a single-version engine with drafts overlapping the tails
exists only without sticky routing (where it already runs). What one engine can overlap is the
*learner*: before this section, `verify_sticky` (all tails) finished before `_run_one_step`, so
training (2.1–2.5 s at Nano L8 EP=2) waited for the last tail [measured, critical_path_measured §3].

**Streaming over the tails (exact).** Step k:
1. `begin_sticky`: verify cohort k's drafts at θ_k (learner, engine-sourced q), stop the streams no
   pending plan continues; the plans of rows still to complete are the step's *tails*.
2. `_run_one_step` starts at once. Its pre-rollout task runs `complete_tails` (the continuation
   rounds on D_k), then `finish_generation`, the momentum stage (if any) and the θ_k refit, then
   `weights_ready`. A rollout with a complete plan is served immediately; a rollout whose plan is a
   tail waits until the rounds complete it; a rollout never drafted (no plan) waits for
   `weights_ready` and decodes at θ_k. Next-cohort drafts wait for the refit too
   (`drafts_after_pre_rollout`).
3. The streaming learner trains groups as they close, i.e. while other rows' tails still run.
Exactness: training of step k and verification of the tail pieces are both learner-side at θ_k; θ
changes only at `finish`; a complete plan's prefix, log-probs and anchors never change again
(`claim_anchors`). The engine still holds one version at a time (D_k until the tails end, then θ_k).
Learner calls of training and piece verification interleave on the actors in submission order (as
the non-sticky chunked verification already does). The momentum stage loads θ̂ into the learner's
parameters between training micro-batches: Ray runs the learner's calls in submission order, so no
training forward sees θ̂ (load, refit and restore are issued from the driver's event loop as one
synchronous sequence).

**Tail-last canonical group order.** After verify0 the groups are ordered (ready now) < (pending
tails) < (never drafted), by group index within each class, and the planner dispatches whole groups
one at a time in that order (`StreamPlanner.set_order`), rows sorted by trajectory index. The order
depends only on the verification results, so micro-batch composition, DP rank placement and the
order gradients add into the fp32 accumulator are functions of the data, not of arrival times
(arrival order changes fp32 summation order, hence the bits of θ_{k+1}). The exposed time after the
last tail is the training of the tail groups still held (canonical order holds a tail group that
finished early behind an earlier-index tail group still decoding) [unit test: identical dispatches
under three arrival orders; GPU reproducibility runs queued]. Per-rank cost: none (no extra buffer;
U/V buffers to remove the last group's training are agent_tail_exact_schemes §3.1(b), not built).

**Training determinism (the kernels under the canonical order) [M].** A canonical order only fixes
the gradient bits if every backward kernel is reproducible. With
`policy.megatron_cfg.deterministic_training=true` (opt-in, default false; 14n), `enable_batch_invariant_kernels`
installs all four. Forward bits are unchanged and gradients change by rounding only:
- MoE token gather (`moe_deterministic.GatherRows`): Megatron's permute backward `index_add_`s a token's
  top-k copies with atomics; now a fixed-order fp32 segment sum. Per rank: a [tokens, top-k] index
  table and one gather of the permuted-activation gradient in backward.
- TE attention: `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` (set in that mode) selects the deterministic
  flash-attention backward. Cost not measurable at Nano L8.
- Mamba conv1d (`mamba_ssd_parity._ConvFixedOrderGradFn`): causal_conv1d adds the rows' weight / bias
  gradients with atomics (one row is reproducible); now the kernel's backward per row, fp32 partials
  summed in row order. Per rank: `rows` kernel launches per Mamba layer per micro-batch (~+0.9 ms per
  layer at 8 x 2048 tokens, Nano channels), no extra activation memory.
- mamba_ssm chunk-scan backward: its deterministic mode (A_log / dt_bias partials reduced in order,
  no measurable cost at Nano-L8 shapes).
Validation: in-loop order replay (`THUNDERSYNC_DEBUG_ORDER_REPLAY=params`: each step's trajectories
retrained in two other arrival orders and the same order twice, per-parameter gradient crc32 and
dispatch signatures vs the real step), Nano L8 EP2, parity reward: canonical == real bitwise on both
learner ranks, steps 1–5 of 6; the arrival-order control differs every step (ts-lanes-stream
cb90f824, local; exactness PASS 6/6). Before the conv / mamba_ssm fixes: same-order replays differed
in the conv1d weight / bias gradients (8 of 435 tensors), then in one dt_bias element.

**Continuation rounds (exact, faster).** `sticky_continuation_chunk_tokens` 2048 → 128 (256 at first; 128 per the tail agent's Nano L8
measurement, 20.9 → 16.7 s per step; below the
1024-token caps of the test runs; 2048 > cap meant every piece was a whole remainder, so a rejection
was found only after the stream had decoded all of it). Rounds no longer block the driver:
`_verify_block_async` sends a piece's verification and the loop keeps cutting other streams' pieces,
serving rollouts and dispatching training; one piece per stream is in flight, so a plan's pieces are
verified in order (same decisions as before; each is block verification of a q draft against the
fixed target).

**Perf mode.** `PERF=1` in `engine_q_tp.sh` / `sticky_moe.sh` turns the debug checks off (mismatch
check, route check, engine-q and sticky checks, spec/mem profiling): timing runs; the checker prints
"SKIPPED (PERF mode, checks off)". Exactness runs keep them on.

**Measured, Nemotron-3-Nano L8, EP=2, 2+2 GPUs, PERF=1, 6 steps, one run per arm** (`time/iteration`,
steps 2–5; `time/step_total` is only `_run_one_step` and excludes the tails when STREAM=0):

| arm | iteration s (s2–s5) | mean | tails / V | train busy | after last rollout | drafted tok/cohort |
|---|---|---|---|---|---|---|
| baseline (STREAM=0, chunk 2048), current | 22.9 / 22.2 / 23.2 / 23.3 | 22.9 | V 18.1–19.4 | 2.4 (serial) | 2.5–3.1 | 24–33k |
| streaming only (93195ee6), current | 18.7 / 19.0 / 15.2 / 23.8 | 19.1 | tails 12.4–22.2 | 2.9–4.2 | 0.4–4.8 | 2.4–39k |
| streaming + rounds (29acacc8), current | 17.7 / 16.7 / 19.5 / 16.5 | 17.6 | tails 13.6–16.9 | 5.1–6.8 | 0.4–1.4 | 2.9–10k |
| baseline, momentum | 23.9 / 22.1 / 20.7 / 23.3 | 22.5 | V 16.2–20.0 | 2.2–2.7 | 2.3–3.3 | 23–29k |
| streaming + rounds, momentum | 17.8 / 17.4 / 17.5 / 15.9 | 17.1 | tails 14.2–14.8 | 3.2–5.9 | 0.4–1.2 | 2.7–8.6k |

−23% (current) / −24% (momentum) per step. Training now overlaps the tails (its busy time grows:
it shares the learner with piece verification). The drafting window shrinks to what follows the
tails + refit (drafts 3–10k tokens instead of 24–33k), so more of the next cohort is decoded as
continuation: the one-engine trade under sticky's two versions. Exactness (checks on, 29acacc8):
Nano L8 current PASS 4/4 (mismatch 0, engine q exact, route check exact, sticky rule active).

**Learner call order (exact by construction).** Learner workers are asynchronous Ray actors (they
have async methods), which may start queued calls in different orders on different ranks. With
training dispatches, piece verifications and the momentum stage queued at once, ranks paired
collectives of different calls: Nano L8 momentum + streaming hung (rank 0 in a verification's EP
all_gather_object, rank 1 in a MoE forward's all-gather), reproduced on the tiny MoE. While
streaming, every training dispatch and piece verification is a numbered learner call
(`SpeculativeGeneration.learner_seq`) and each rank runs them strictly in number order (worker
`ordered_call`), several in flight; unordered calls (refit, momentum stage) first drain the ordered
ones. After every streamed step the driver compares each rank's crc32 of its (number, method)
sequence and raises on a difference. A global driver-side lock (one call batch in flight) was the
interim fix; step time with it was the same within noise (11.8–12.0 s vs 12.1–12.2 s ordered).

**In-loop decode (profiled).** At EP=2 the engine's decode step cost 11–14 ms at every row count
(EP=1: 3.9–6.6 ms): each MoE layer ran ReLU², the probability multiply and the fp16 cast over the
worst-case padded CUDA-graph buffer (every EP rank's tokens × top-k, ~102k rows × 1856 at Nano
shapes), ~2.2 ms per layer per step [torch profiler]. `fp16_moe.relu2_probs_fwd` /
`swiglu_probs_fwd` now take the permuted-row end (`offsets[-1]`, a device scalar) in the engine
and skip padding rows (never read: the grouped GEMM stops there); bitwise the eager values (GPU
tests incl. CUDA-graph replay). After: 5.1–5.5 ms (1–4 rows), 6.4–6.9 ms (17–64 rows). Another
~1.7 ms/step is the LM head's per-step fp32 hi/lo copies, which engine package A's stacked head
removes (not on this branch). Per rank: no memory change; the fix matters more with depth (full
Nano: 23 MoE layers).

**Measured after both (Nano L8, EP=2, PERF, 6 steps, `time/iteration` steps 2–5, one run each):**
current 12.1 s, momentum 12.2 s (from 22.9 / 22.5 baseline and 17.6 / 17.1 before the decode fix).
Steps 3–5 of the current-draft run: verify0 0.7–0.9 s; tails + refit + serving 5.6–6.7 s
(continuation tokens 47–65k at 17–20 rounds, drafts 54–72k tokens); after the last rollout 3.5–4.5 s
(the remaining training of the tail groups and draft collection 0.5–0.6 s); learner busy 5.8–6.1 s
per step. The learner (training + piece verification at chunk 128) is now the longest resource on
the step; the tail-last order leaves the tail groups' training exposed after the tails.
Determinism: every L8 run so far had zero rewards, so all data gradients were zero and identical
parameter checksums across arrival orders prove nothing; non-vacuous pairs
(`THUNDERSYNC_DEBUG_PARITY_REWARD`) are queued.
## 14m. Multi-turn rollouts under sticky routing, scheme A (`grpo.max_rollout_turns > 1`)

*ts-nano-perf (2026-10-14):* multi-turn and the anchor horizon (14i) run with
`thundersync.stream_training_over_tails=false` (14k): streaming over their tails is not built yet (a
streamed rollout waits for its plan to complete in the continuation rounds; multi-turn plans complete
through environment steps in `replay_row`, horizon plans on the refit engine). The validator rejects
the combination and the multi-turn / horizon launchers set `STREAM=0`. The continuation rounds
themselves are the shared non-blocking loop (chunk 128) with the environment steps folded in.

*Built on branch `ts-multiturn-a` (from ts-integration 29064e90); on whenever sticky routing is on
and `grpo.max_rollout_turns > 1`. Unit-tested on a fake engine; GPU end to end: see the status line
at the end of this section.* Design brief and exactness proof by enumeration:
`thundersync_work/research/agent_multiturn_sticky.md` (tests on RL-sticky b8212df1,
`test_sticky_multi_turn_all_turns_at_drafter_weights_is_exact`, TV < 1e-12, prefix-dependent
stochastic environment, 3 turns; control `..._later_turns_plain_at_new_weights_are_biased`);
implementation notes: `thundersync_work/research/agent_multiturn_a.md`.

**Why.** A token of turn 2 depends on the hidden states of every earlier position, i.e. on
F(θ_k; A) there. Decoding turn 2 on the refit engine (θ_k, plain routing) re-routes turn 1
without its anchors: the result is neither p_k nor any policy the learner trains on (control
test: biased).

**Scheme A: every turn of a drafted rollout at the drafting weights, before the refit**
(`multi_turn.py`, `SpeculativeGeneration.verify_sticky` / `_continue_sticky`).
Turn 1 is drafted and block-verified as before. When a VERIFIED prefix ends a turn (EOS, or
the per-turn cap `_cap(prompt + earlier turns and observations)`, the same budget the plain
multi-turn path gets), the driver steps the environment on it (`TurnEnvironment.step`: the same
`env.step.remote` request `calculate_rewards` makes on the row's message log, awaited on the
driver's event loop, so other rollouts' streams and environment steps keep going). Unverified draft tokens never reach the environment or its tools. Unless the rollout
ends there (terminated, observation truncated at the total-length cap, or `max_rollout_turns`
turns), the observation is appended to the plan and a new θ_{k−1} stream decodes the next
turn from prompt + turns + observation; its pieces are verified exactly like continuations
(chunks of `sticky_continuation_chunk_tokens`, residual token on a rejection, restart from
the new prefix). The plan's `prefix` is the whole row after the prompt (turn tokens and
observation tokens; observation log-probs are placeholders, masked from the loss);
`Plan.turns` records each turn's span and the environment's answer.
- Anchors of observation positions come from the continuation stream's own prefill (plain
  θ_{k−1} on the actual prefix: the definition of A). Each new stream re-logs every position
  of its row (prefix caching is off), so the plan's `RouteRefs` after its last verified piece
  cover the whole row up to the last generated token; the final observation follows every
  generated token and needs no anchor.
- Rows: each drafted plan is paired with one rollout-loop row (`bind_rows` before
  `verify_sticky`; rows with the same prompt take its plans in queue order, which does not
  depend on the tokens), so the environment sees that row's `extra_env_info`/`task_name`.
- Rollout loop (`grpo_loop._run_one_step`): a row with a finished drafted plan is rebuilt from
  the recorded turns (`multi_turn.replay`: same message log, rewards, metadata and metrics as
  `run_sample_multi_turn_rollout`, checked against it in `test_multi_turn_sticky.py`); the
  environment is NOT stepped again (environments need not be deterministic; tools may have side
  effects). Training anchors come with the plan. Rows without a plan (never drafted, dropped
  drafts, ⊥-BV fresh slots) run `run_sample_multi_turn_rollout` unchanged at θ_k ("self"
  everywhere, F(θ; self) = plain).
- Exactness [proof by enumeration, brief]: each piece is a q draft given the current prefix,
  block-verified against the fixed target p = F(θ_k; A); an environment step is a function of the
  verified prefix and its own randomness. No new decision rule. ⊥-BV composes: its output
  decisions are made on turn-1 drafts; every later token is conditional sampling given a
  verified prefix [derived].
- Not supported (raise): environment `next_stop_strings` and per-sample `stop_strings` (drafts
  and continuations run without stop strings; same as single-turn speculation).

**Per-rank / per-step costs at scale [derived].**
- Driver: one environment step per verified turn end (the same count as plain multi-turn) and one
  stream start per later turn; per turn only token lists cross the driver (the turn's tokens, the
  observation's ids). Routes stay object-store handles (`RouteRefs`), never fetched by the driver.
  Environment steps are awaited Ray calls (no thread per call, unlike the plain path's
  `asyncio.to_thread(calculate_rewards)`, which caps concurrent steps at the default thread pool):
  thousands of rollouts can wait on slow tools at once while every other rollout's stream decodes.
- Engine (θ_{k−1}): each later turn is a new request whose prefill re-reads prompt + earlier turns
  + observation: O(turns × S) prefill tokens per rollout without prefix caching, plus its route
  log (L_local × K int16 per position on the logging rank, e.g. Ultra-like L=60/PP, K=22: 2.6 kB
  per position per stage, all positions re-logged per turn). Step 4 of the brief (prefix
  caching: prefill O(observation) per turn, only new positions logged) is not built yet; prefix
  caching stays off under sticky routing. What it needs [C, D]: the route log records absolute
  positions (`token_to_pos_ids`), so a cache hit would log from the first uncached position, and
  `merge_route_refs(base=plan.anchors)` already splices such a log over the plan's anchors (same
  θ_{k−1}, same prefix: bitwise equal under BI); the next-turn and restart streams would pass
  `plan.anchors` as their base. Open: prefix caching is engine-global, so drafts of one GRPO group
  would hit each other's prompt blocks and need a shared prompt-anchor base; the route checks'
  prefills must bypass the cache (else they compare a log with itself); cache entries must never
  cross a weight version. All untested on GPU.
- Learner: unchanged per token: anchors for every row position (observations included) fetched
  per pipeline stage from the object store; one verification forward per piece (as single-turn
  continuations).
- Critical path: the refit waits until every turn of every drafted rollout is done, so
  ENVIRONMENT LATENCY (tool calls) now blocks the refit. Correct but slow for agentic tasks.
  Fixes: θ_{k−1} retention after the refit (engine keeps θ_{k−1} as a second weight version or
  sparse delta, continuations run on it while new rollouts run at θ_k), or scheme B (anchor
  horizon H with sticky replay in engine prefill: θ_{k−1} work bounded by H tokens; another
  branch, `ts-multiturn-b`).

**Checks** (`scripts/bi_tp/sticky_multiturn_moe.sh`: `sticky_moe.sh` with the stochastic toy
environment `toy_multi_turn_env.py`, `--toy-multi-turn-env TURNS`, small per-turn budget).
Everything `sticky_moe.sh` asserts (mismatch 0 over every assistant token of every turn, engine q
exact, drafts' route check exact, weights change, drafts accepted, sticky rule active) plus, every
step after the first: drafted multi-turn rollouts replayed (`spec/replayed_multi_turn_rows`), the
driver stepped the environment (`spec/sticky_env_turns`), some rollout ran > 1 turn
(`max_turns`), and the finished plans' anchors equal a fresh engine prefill of their full rows
(`[route check mt]`, `spec/route_check_mt_exact == spec/route_check_mt_tokens > 0`) with anchored
observation positions among them (`spec/route_check_mt_obs_positions > 0`).

*GPU status (2026-10-14, 4 GPUs, EP=2, 4 steps, 3 turns, 128 tokens/turn):* PASS on the tiny MoE,
real Qwen3-30B-A3B layers (L4), ⊥-BV (K=1.5) and momentum draft weights: mismatch 0 over every
assistant token of every turn, `[route check mt]` exact (51–55k positions per step, 2.0–2.5k of them
observation positions), 128/128 rollouts per step replayed, environment stepped once per turn.
Turns ended at the per-turn cap almost always (truncated models rarely emit EOS): EOS turn ends are
unit-tested only. Environment latency on the refit path not measured (instant toy environment).

*Hybrid Nemotron-3-Nano (branch ts-nano-multiturn = ts-hybrid-integ + ts-multiturn-ab, 2026-10-14):* L8
(MEMEM*EM, sigmoid + expert-bias router, fp16 GEMM recipe, EP=2, async, 4 steps): PASS, mismatch 0 every step,
engine q 8/8 exact, `[route check mt]` exact 53114/52003/55061 positions incl. 2194/1985/2381 observation
positions, 128/128 rows replayed, 277–305 environment steps per step. Every later turn is a new engine request whose
prefill rebuilds the Mamba conv/SSM state from prompt + turns + observations (no state truncation anywhere), and
route layers are the MoE router instances (hybrid numbering from the route log); nothing in the multi-turn code
is layer-type specific. With the anchor horizon (H = 160, section 14i): also PASS (below).

## 14i. Sticky routing anchor horizon with engine anchor apply (`thundersync.sticky_anchor_horizon`)

*Opt-in (unset: today's behaviour, anchors up to the rollout cap). Requires sticky routing (14c).
Unit-tested (`tests/unit/test_anchor_horizon.py`); end-to-end exactness: see the status line at the end of this
section. Notes: `thundersync_work/research/agent_multiturn_b.md`; the scheme's output law was proven by enumeration
(`test_sticky_multi_turn_horizon_with_engine_replay_is_exact`, RL-sticky b8212df1 / RL-nullbv 7e3fbdb0, TV < 1e-12,
observations on both sides of the horizon).*

**Policy.** p_k = F(θ_k; A_H): A = the plain θ_{k−1} routes on the rollout's own prefix at positions
t < c = P + min(H, cap) (P prompt tokens; content-independent), self at t ≥ c and for rollouts never drafted.
With H unset c is the cap (14c). Everything else in 14c (the rule, δ, block verification, no bonus) is unchanged.

**What the horizon buys.** Under 14c every drafted rollout is completed at θ_{k−1} before the refit (continuation
rounds to EOS or the cap; for multi-turn rollouts every turn, environment latency included). With a horizon the
θ_{k−1} part of a drafted rollout ends as soon as its anchors below c are logged: at most H + 1 continuation tokens
per rollout (position c − 1 is routed by the forward that samples token H). The refit engine then samples the rest
of the rollout from p_k directly, which needs only the anchors below c, re-applied in its forwards.

**Driver** (`speculative.py`). `_horizon(P)` = c; draft and continuation anchors are cut to c (`_cut`:
`RouteRefs.truncate(c)`, so the learner's verification and training forwards see self beyond c); a drafted token
needs logged anchors only below c (`anchored_draft_length(..., horizon=c)`), so drafts longer than H are verified
against p_k without θ_{k−1} routes past c. A plan is done at θ_{k−1} when complete or when its anchors are complete
below c (`_horizon_done`); continuation streams decode at most H + 1 tokens; a stream that stops at that budget is
not a finished rollout. After the continuation rounds (`verify_sticky`) every incomplete plan gets an anchor key and
all generation workers receive {key: RouteRefs} (handles only, `register_engine_anchors`; the coordinator picks
the data-parallel rank that runs a request, so every rank holds the small handle table). `_one` continues such a
plan on the refit engine with the request tagged (`ts_anchor_key` sampling-params attribute, like the draft
hidden-log tag). Training claims the plan's anchors (length c) as before.

**Engine anchor apply** (`worker.py` `_install_anchor_apply` / `_wrap_anchor_apply`, `routes.EngineAnchors`,
`sticky_routing.py`). Every engine rank with MoE layers (all TP ranks, each pipeline stage for its layers):
- *Static buffer* [max_tokens, L_local, K] int16, attached to the routers before the engine captures its CUDA
  graphs. In RECORD (engine) mode the sticky wrapper records the plain top-k as before (the drafter's route log is
  unchanged) and then applies `sticky_select` (the learner's function, same arithmetic) against
  `buffer[r·T : (r+1)·T]` (T the forward's token count; r the TP rank under sequence parallelism, else 0): a fixed
  slice per graph, so graph replays read what the step wrote. Rows of -1 are self (bitwise plain top-k), so
  untagged requests and drafts are routed exactly as before.
- *Per-request pool*: the rank that admits a tagged request fetches its own stage's anchor blocks
  (`fetch_anchors`, object store) when the request first appears in the context, into pages of 256 positions of a
  device pool (grown by doubling); pages are freed once the request has left the engine (sweep every 64 forwards).
- *Per forward* (wrapper of the controller's `_dynamic_step_forward_logits`, runs before each launch, eager or
  graph, legacy or async scheduling): anchor id per slot from the context's CPU bookkeeping (`SlotTags`: Python
  only for slots whose request is new), then numpy over the step's tokens (anchor id, absolute position; rows
  below the request's c), one pinned non-blocking index copy, a fill of last step's rows with -1 and one gather
  from the pool, all enqueued on the forward's stream. Prefill, chunked prefill and eviction re-prefill re-apply
  anchors by absolute position; decode beyond c routes plainly. No host-device synchronization, no Python per
  token, no driver traffic per token.
- *Expert-parallel dummy steps* (MCore `dummy_forward` keeps the last real step's ids/positions): every row is
  made self (`EngineAnchors.clear`), like the route log's dummy-step guard (14c).
- MCore: no source change (the RouterReplay wrapper and the controller-method wrapper are ours). Upstream
  candidate: a per-token "replay with rule" hook in MCore's RouterReplay for inference (record + apply).

**Per-rank costs at scale** [D unless marked]. Notation: L_local MoE layers on the stage, K top-k, H horizon, P
prompt, R anchored requests resident on an engine rank.
- Pool memory: R × (P + H) × L_local × K × 2 B (pages of 256 positions). Ultra-like (60 MoE layers, K = 22,
  PP = 8 → L_local ≈ 7.5, H = 4096, P small): ≈ 1.4 MB per request per stage; 256 resident anchored requests
  ≈ 350 MB per rank. Only rollouts that were drafted, continued past c and are still running hold pages.
- Static buffer: max_tokens × L_local × K × 2 B (16384 tokens: 5.4 MB at L_local 7.5, K 22).
- Transport: each anchored request's stage blocks are fetched once per engine rank that runs it (TP ranks fetch
  the same bytes; no driver gathers: the driver sends handles only). Per step on the device: 16 B of index per
  anchored token H2D + a gather of L_local × K × 2 B per anchored token.
- Compute: the rule per MoE layer and token, O(E) elementwise + a K-sort, inside the router; measured overhead and
  the fill's host/device time: see the status line. Without a horizon nothing is installed (no change to engine
  graphs).
- θ_{k−1} work: ≤ H + 1 continuation tokens per drafted rollout (vs to EOS/cap); the refit no longer waits for
  long continuations (and, with multi-turn, for environment latency past the horizon).

**Multi-turn.** Turns below c run at θ_{k−1} exactly as in scheme A (14m, branch ts-multiturn-a: driver turn loop,
observation anchors from the continuation's own prefill); once a row's anchors are complete below c, the rest of
the rollout (later turns included) can be decoded by the refit engine with the request tagged, since each later
turn's prefill re-applies the anchors below c by position. Built on branch ts-multiturn-ab (scheme A merged with
this branch): `_horizon` is clipped by the total-length cap for multi-turn rollouts (max_new_tokens bounds one turn
there); a turn that ends at or past the horizon is not sent to the environment at θ_{k−1}; the rollout loop finishes
a handed-over row with `_finish_turns` (tagged engine requests per turn, one environment step per turn end, the same
turn rules as below the horizon) and then rebuilds it with scheme A's `replay`.

**Rule kernel.** The engine applies the rule in every MoE layer of every forward once the horizon is on, so it is
one Triton kernel (`sticky_routing_fused.py`; a pure selection, bitwise equal to `sticky_select`: GPU unit test with
exact ties, invalid anchor rows, unsorted plain top-k and anchor sets equal to the plain set). The kernel follows the
rule's output order exactly: a kept anchor set equal to the plain set is returned in the plain top-k's order, a
different kept set in descending score order (on the hybrid branch the torch rule gained the plain-order case; the
kernel ported from ts-integration still sorted such rows, a bitwise mismatch caught by the GPU unit test and fixed). Measured [M] (GB200, CUDA graph, `thundersync_work/research/multiturn_b/
bench_anchor_apply.py`): torch composition +53–82 µs per layer (Qwen3-30B-A3B shapes, 48 layers: +2.6 ms per decode
forward), fused +2–8 µs per layer at T = 64–512 tokens (+0.10–0.20 ms per forward over 48 layers) and +14 µs at
T = 8192 (+0.69 ms); Ultra-like (E 512, K 22, 15 layers per stage): +0.12 ms per decode forward. Fill: 6 µs host
and device per forward without anchored tokens, 0.30 ms host + 0.30 ms device for an 8192-token fully anchored
prefill; loading one request (P + H = 5120 positions) 0.5–1.7 ms.

**Status.** [M] 4 GPUs, tiny MoE EP=2, async scheduling, 4 steps (`scripts/bi_tp/sticky_horizon_moe.sh`, H = 64):
mismatch 0 and exact_token_frac 1 every step (the horizon continuations' engine log-probs equal the learner's
training forward with the same anchors), route checks exact, 41–97 plans per step handed to the refit engine,
15821 tokens routed with applied anchors. Control `NOAPPLY=1` (engine never applies): max |Δlog p| 0.30–0.44,
exact_token_frac 0.80–0.84, i.e. the anchors below the horizon change the policy and the apply path is what
reproduces it. Same check with the fused rule: PASS; also PASS at engine/learner PP = 2 (EP = 1; each stage applies
its own layers' anchors, 42366 anchored tokens by step 3) and TP = 2 with sequence parallelism (EP = 2; each TP
rank reads its token shard of the buffer, 37020 anchored tokens), mismatch 0 and route check exact every step. Multi-turn (branch ts-multiturn-ab = ts-multiturn-a +
this branch, `_finish_turns`; toy stochastic environment, 3 turns × 128 tokens, H = 160, 4 steps): PASS, mismatch 0
every step over every turn, route check over full rows exact incl. 1118–1382 anchored observation positions per
step, 128 environment steps per step below the horizon (θ_{k−1}) and 234–280 turns finished on the refit engine
beyond it; control without apply: max |Δlog p| 0.53–0.66. Runs and numbers: `thundersync_work/research/
agent_multiturn_b.md`.

*Nemotron-3-Nano L8 on ts-nano-multiturn* (hybrid, fp16 recipe, EP=2, async, 4 steps) [M]: single-turn H = 64
PASS (mismatch 0, engine q 6/6 exact, 19/117/122 plans to the refit engine, 23204 anchored tokens by step 3);
multi-turn H = 160 PASS (mismatch 0 over every turn, `[route check mt]` exact 32552/33680/33320 positions incl.
1252/1118/1382 observation positions, 128 environment steps per step below the horizon, 261/234/280 turns finished
on the refit engine, 85033 anchored tokens). No-apply controls on Nano: queued, not yet measured.

**Nemotron-H (Nano/Super/Ultra) routers.** The rule acts on the scores MCore selects on (`compute_topk`'s input):
sigmoid + expert bias for Nemotron-H, after which MCore gathers the sigmoid scores of the effective experts; engine
apply and learner replay go through the same call (unit test with E = 128, top-6, expert bias, scaling factor:
bitwise equal, rule active). Hybrid layer numbering: anchors exist per MoE router instance only (route layers =
MoE layer numbers); Mamba and attention layers carry none. MCore's fused router (`moe_router_fusion`) bypasses
RouterReplay and is not supported (as for route recording).

## 14n. ts-nano-perf: one Nemotron-3-Nano branch for the at-scale A/B (2026-10-14)

*Branch ts-nano-perf (../RL-nano-perf), code 2341c723. Merge order and conflict decisions are in the merge commits.
Notes: thundersync_work/research/agent_nano_perf.md.*

**Contents.**
- nano-scale3-launch (unpermute + scale3 fixes)
- engine package A
- one-engine d9c6f55f (streaming over tails, ordered learner calls, chunk 128, canonical order, decode activation fix
  4259748d)
- ts-tail 1aabf28c (batched ⊥-BV rebuild; opt-in tail tools)
- multi-turn A/B (needs `stream_training_over_tails=false`, 14m)
- the A/B kit (rollout_lag=1 same-learner async-1, stock arms, progress watchdog)

**Validation [M].**
- Nano L8 EP2, parity reward, expert-grad noise: exactness PASS 4/4 on all 6 runs.
  - Arms: ⊥-BV + momentum, momentum only, ⊥-BV + current, d1, d2, PERF.
- GPU unit tests 307 pass; CPU 769 pass.
- The d1/d2 checksums differ from step 0. This is engine sampling (one RNG stream per engine step, 14k), not
  accumulation, so paired runs do not test determinism. The in-loop order replay (one-engine cc83f7bf) does.

**Same-tree A/B, Nano L8 [M].**
- Setup: EP2 (2+2 GPUs), 16 prompts x 8, 1k cap, MBS 8 / LPBS 4. Net s/step at steps 2-5 (checks subtracted).

| arm | s/step |
|---|---|
| sync | 10.3 |
| async1 | 7.55 |
| async1_nrl | 7.9 |
| sync_coloc | 19.6 |
| ts_por | 11.0 |
| ts_nonull | 12.2 |

- ThunderSync is ≈ 0.69 of async-1 here.
- Where the ts_por step goes:
  - Learner training busy 5.4-5.9 s vs 2.7 s for the same rows on sync / async1 (same worker). Canonical group order
    dispatches one group at a time: 16 dispatches of ~0.35 s. The fixed cost per dispatch, not the compute, doubles
    the training time.
  - Learner piece verification 0.6-1.5 s.
  - verify0 1.1-2.4 s, serial at the top of the step, before any rollout; route check excluded.
  - Tails 1.0-4.8 s, overlapped.
- The ts step is learner-bound: busy 5.5-8.3 s plus verify0. async1 is generation-bound: 6.9 s next-batch decode,
  learner 2.7 s hidden.
- Next levers:
  - fewer, larger canonical dispatches (fixed group blocks keep determinism);
  - fused verify+train (ts-fused-vt);
  - q-transport P0/P1 (the hidden/route put bug inflates verify and tails);
  - verify0 overlap.

**Full-depth decode [M] (thundersync_work/research/nano_perf/ep_decode_bench.py, full Nano, one node EP4).**
- Engine config as the EP8 runs: max_tokens 16384, graphs to 512 tokens, NVLS dispatch.
- ts-nano-perf decode: **27.9 ms/step at 1 row/rank, 29.0 ms at 8 rows**, graphs on.
- Per-kind device time (graphs off, 10 steps): MoE kernels 23.7 of 28.9 ms. Each one is a *launch grid sized for the
  padded worst-case buffer* (max_tokens x EP x top-k ≈ 393k rows at EP4, 786k at EP8), with early exit past the live
  rows:
  - `_relu2_probs_kernel` 9.6 ms (0.42 ms x 23);
  - `_grouped_gemm_fp16w_kernel` 10.5 ms (0.23 ms x 46);
  - `_unpermute_fp64_kernel` 2.4 ms (grid over global_max tokens);
  - `_permute_tokens_kernel` 1.0 ms.
  - Everything else: dense GEMMs 1.3, elementwise 3.0, Mamba 0.9, NVLS AGV + ordered RSV 0.4-1.3, attention 0.1 ms.
- The pre-fix engine also ran eager relu, square, x prob and cast over all padded rows. That was ~4x the bounded
  kernel's grid cost and covers the ~210 ms/step measured at EP8 (runs_nano_scale, RL-nano-scale2).
  - Package A (≈12-15 ms at full depth) was also missing there.
  - CUDA graphs were used in both (35 graphs, 64-512 tokens).
- **Next engine lever:** make the MoE kernels' grids independent of the worst case.
  - Option 1: persistent grid (SM count) looping to `offsets[-1]` / valid tokens.
  - Option 2: grid per CUDA-graph bucket (bucket tokens x EP x top-k).
  - Per-row compute is unchanged, so this is bitwise and CUDA-graph safe; the grid is per rank and independent of EP.
  - [D] ~23 -> ~1-2 ms, so full Nano ≈ 6-8 ms/step at EP4. EP8 doubles today's padded grids (≈ 50 ms/step at EP8
    today [D]), so EP8 gains more.
  - Also: `select_engine_config` picks the grouped-GEMM tile from the padded row count.

**Deterministic training (opt-in, `policy.megatron_cfg.deterministic_training`, default false).**

- *What the default gives.* Exactness needs a 0 logprob difference between the engine and the training forward. That is
  batch invariance, which `enable_batch_invariant_kernels` installs by default (forward behaviour only).
- *What the mode adds.* Bitwise-reproducible gradients: the same rows give the same update. It installs four fixes, all
  backward-only, so forward values and exactness are unchanged:
  1. fixed-order MoE token-gather backward (`moe_deterministic.GatherRows`);
  2. Transformer Engine deterministic attention backward (`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`);
  3. causal_conv1d weight/bias gradients per row, summed in row order (`mamba_ssd_parity._ConvFixedOrderGradFn`);
  4. mamba_ssm deterministic chunk-scan backward.
- *Cost (Nano L8).*
  - The conv per-row gradient is ~+0.9 ms per Mamba layer per micro-batch: one kernel launch per row.
  - The other three are within noise (0.40-0.42 vs 0.42-0.43 s per fwd+bwd).
  - Per rank, the cost is proportional to rows x Mamba layers per micro-batch, independent of EP / node count.
- *Coupling.* The in-loop order replay (`THUNDERSYNC_DEBUG_ORDER_REPLAY`) compares gradient digests of the same rows in
  other orders, so it requires this mode. ThunderSync's `validate_config` rejects the replay without it.
- *Measured.* With the full recipe the replay matched bitwise on Nano L8 steps 1-5 (one-engine agent).

**Frozen for the full-Nano A/B (2026-10-15): ../RL-nano-perf-scale @ 5b3530de.**

*Validation [M].*

| run | tip | what | result |
|---|---|---|---|
| np-v2-x1k | 6ed6cb24 | 1k exactness + 59 GPU unit tests | EXACT 4/4 |
| np-dh2-lanes-x | 9ff5bd21 | lanes exactness | EXACT |
| np-dh3-8k | a6006fb0 | Nano L8 at 8k | EXACT 3/3, no learner OOM |
| np-v-8k-pc | 9a96457d | Nano L8 at 8k, prefix cache on | perf (below) |

*Learner memory at 8k (np-dh3-8k).* The fixes in this tip:
- deferred LM head (logits never materialized for a whole micro-batch);
- micro-batches bounded by tokens (`thundersync.train_micro_batch_tokens`);
- cached memory released before the optimizer step, only under pressure.

Measured: peak 81 GB allocated, 144-149 GB reserved, i.e. fragmentation.

*First 8k piece-verify number (np-v-8k-pc, step 2).* With the prefix cache, 10.7 s, against 9.4 s of training; cache
hits 76 %. A step whose drafts were cut at ~1 token decodes the whole cohort as tails (59 s), so the drafting window
at 8k is the next lever.

*Engine-vs-learner EP (np-v2-lanes-ep).* Rows the engine samples plainly differ under an EP mismatch, by up to 0.89 in
log-prob. The old MoE combine is EP-dependent. With `moe_ep_invariant_combine` (section 14q) the engine EP may
differ from the learner's: epinv-lanes-ep, the same layout, is exact.

*A/B layout.* Every arm uses learner EP4 x DP2 on 8 GPUs plus 8 engine GPUs:
- one EP4 engine at engine DP 2, or
- 2 lanes x EP4.

All arms set:
- `max_requests` 128 per engine rank (the primaries of an EP4 lane);
- `buffer_size_gb` 48;
- prefix cache on at >= 4k on the ts arms;
- no debug reward or noise (`submit_ab.sh` refuses them);
- CHECKS=0.

**Integration round 2 (2026-10-15, ts-nano-perf on eb351bc0).** Merged, in order (conflict choices are in the
merge commit messages):
- ts-lanes-stream e64345d9: token-bounded canonical blocks, deferred draft cut + serve yield, calibrated ⊥-BV with
  measured verify cost (one calibrator shared by the lanes), group draft admission (default).
  `ab_base` turns `thundersync.null_bv_calibrate` on: the fixed prior v is ~4.5x below the measured Nano v
  (cal-x-l2: 3.5-3.8e-6); cal-x-l2 and adm2-x-l2-grp-cal are EXACT.
- ts-run-ready 04c4b855 (14s): checkpoints/resume, epochs, validation, seeds, NaN guards, always-on weight sentinel,
  multi-turn prefix and collect_drafts guards, leftover-plan fix. Guards and sentinel are validated (rr-gpu-unit,
  rr-sentinel-lanes-ep). 04c4b855 adds pinned mamba_ssm backward autotune configs (a cross-process
  nondeterminism source), torch deterministic algorithms under deterministic_training, a kill-at-step env
  (`THUNDERSYNC_DEBUG_STOP_AT`) and a resume precheck of the run length. Exact resume is NOT yet shown:
  rr-resume-nano failed (LR schedule sized from a different max_num_steps; sync arm not reproducible before the
  resume; comparator bug); the fixed test rr-resume-nano2 is pending.
- ts-m2n-refit e31b781a (14q). m2n-xc-one / m2n-xc-lanes: crosscheck PASS on every refit (theta and theta-hat),
  m2n-x-one / m2n-x-lanes EXACT 6/6. So `ab_base` sets `refit_transport: mcore` + `refit_backend: nccl` for every
  arm on the ThunderSync loop (sync, async1, ts_*), so refit cost is equal across them; async1_nrl (stock loop) keeps
  its transport.
- ts-verify-in-train (14r), the 4c4ae593 tree plus the candidate-selection fix 1fe2cf7a (VIT never engaged before
  it; vit-l2's "VIT on" perf arm therefore measured nothing). Opt-in, default off.
- ts-carried-walk 0997895f (DESIGN section renumbered 14r -> 14t): continuation decision rules opt-in (default
  `every`); prefix-cache `pin` policy and `auto` size are the defaults (LRU thrashes under cyclic rounds); the
  chunked verification head is always on (fp16 recipe, TP 1): cw-x-4k hit a 68 GB OOM in a verification forward
  that materialized [rows x tokens x V] fp32 logits; now every verification forward (verify0, pieces, walk, scans)
  takes log p from at most 4096 rows at a time, bitwise the materialized path. **Scale gap:** at TP > 1 the
  vocab-parallel verification (`GatheredLogprobRows`) still keeps full-length [n, V/TP] logit shards per row.
- ts-learner-callcost 45bbb962, opt-in: training expert GEMMs default to per-expert cuBLAS;
  `NEMO_RL_FP16_MOE_TRAIN_GEMM=auto|segment` (e.g. via `policy.megatron_cfg.env_vars`) selects the bitwise segment
  kernels (full-Nano EP4 fwd+bwd 1.6x at 1k, ~2 % at 8k, GPU kernel time ~2x cuBLAS's).
- ts-ep-invariant a036bb7c (14u): EP-invariant MoE combine. The engine-vs-learner EP equality rule is lifted when it
  is on (both validators); `run_ab_arm.sh` turns it on when GEN_EP != EP.

*Round-2 validation [M] (Nano L8, parity reward, no expert-grad noise; runs_1node/i2-*, code bd792ea1).*

| run | what | result |
|---|---|---|
| i2-ut | GPU unit tests, tip vs base eb351bc0 | tip 1769 passed, 3 failed + 10 errors, all also failing at base (base: 5 failed incl. 2 hidden-transport tests that run-ready fixed) |
| i2-x1k | ts_por one engine EP2, 1k, m2n refit + crosscheck | EXACT 4/4, crosscheck 5/5 PASS, non-vacuous |
| i2-xlanes | learner EP2, 2 lanes EP1 (EP-invariant combine), forced-fresh rows, m2n crosscheck | EXACT 4/4, crosscheck 7/7 PASS |
| i2-x8k | ts_por EP2 at 8k, prefix cache (pin/auto), chunked verify head | EXACT 4/4, no OOM; continuation pieces verified: 511k / 504k tokens in 46 / 61 rounds (steps 2-3) |
| i2-p | equal GPUs (4): lanes vs async1 at 1k, steps 2-5 | lanes 8.44 / 7.51 / 8.77 / 6.92 s (mean 7.91); async1 7.11 / 4.96 / 5.02 / 4.70 (mean 5.45): lanes = 0.69 of async1 |

At 8k on one engine the drafting window is again the limit: steps 2-3 cut the drafts at 519 / 857 tokens and decode
~0.5M continuation tokens (60-69 s steps). Exact resume: not shown (i2-resume / rr-resume-nano2 pending).

**Frozen: ../RL-nano-perf-scale2 @ 8f99ebc9, fork branch ts-nano-perf-scale2.** 8f99ebc9 = the validated bd792ea1 plus
the ep-invariant branch's validator wording (same rule), DESIGN and two unit tests.

*After the freeze (on ts-nano-perf, not in the snapshot):* ts-learner-callcost be160af1 (TMA segment kernel, per-pass
choice by cost; one MoE layer 32 experts x 8k fwd+bwd 23.3 -> 7.7 ms) merged opt-in, cuBLAS default on forward, dgrad
and wgrad until cc-full-ep4-new2 confirms the 8k gain. Opt-in `CUBLAS_WS_SHIM=1` (run_ab_arm.sh / submit_ab.sh /
env.sh): an LD_PRELOAD shim gives every cuBLAS handle a 0-byte workspace instead of CUBLAS_WORKSPACE_CONFIG=:0:0
(PyTorch 2.11 re-parses that variable with std::regex per cuBLAS call; 58 -> 24 us host per mm, same bits;
upstream fix pytorch#177912, backport #200066 open). The te_native env check accepts the shim as equivalent. Not a
default until the model-level fingerprint checks pass.

*A/B layout after round 2: unchanged, learner EP4 x DP2 on every arm (8 GPUs) + 8 engine GPUs* (ts_por: 2 lanes x
EP4; sync / async1: one EP4 engine at engine DP 2), 16 GPUs = 4 nodes per arm. Engine EP = learner EP, so the
EP-invariant combine stays off (the old combine is exact at equal EP and cheaper). Alternative considered: learner
EP8 + 2 x EP4 lanes with the combine on. It would cost the lanes ~5 % engine decode (full Nano EP4 7.5 -> 7.9 ms/step)
and ~16 ms per 4k-token prefill step, and the one-engine baselines would then run a different engine layout (EP8)
from the lanes. No measurement shows learner EP8 faster than EP4 x DP2 at equal GPUs (same FLOPs per rank; EP8
saves per-rank expert memory, DP2 adds a gradient all-reduce), so the known 5 % cost decides it. Revisit if the
learner at EP4 runs short of memory at full Nano 8k (report learner/mem_peak_gb from the first run).

## 14o. Engine MoE kernels on live-row grids (2026-10-14)

*Branch ts-moe-livegrid (../RL-livegrid) from ts-nano-perf. Code: `nemo_rl/models/megatron/fp16_moe.py`
(`engine_experts`, `permute_live`, persistent kernels). Notes: thundersync_work/research/agent_moe_livegrid.md.*

**Problem [M, 14n].** The NVLS AllGather-V dispatcher gathers into a fixed buffer of global_max = max_tokens x EP
rows (16384 x EP), the CUDA-graph-safe worst case. Every engine expert kernel launched a grid over that buffer and
exited past the live rows: the activation one program per permuted row (393k at EP4, 786k at EP8), the grouped GEMM
one per BLOCK_M tile, unpermute one per gathered token, permute a fixed grid looping to max_pairs, plus an 8 MB
inverse-map fill per layer. The grouped-GEMM tile was picked from the padded row count (BM = 128 for a few rows per
expert). Full Nano EP4 decode: 23.7 of 28.9 ms per engine step in these kernels; the cost grows linearly with EP.

**Design.**
- Persistent grids. Each kernel loops over its live work, read on device: GEMM M tiles below `offsets[-1]`,
  activation rows below `offsets[-1]`, unpermute and permute tokens below `valid_tokens`. The launch shape is static
  (CUDA-graph safe); the work is proportional to the live rows.
- Grid = min(padded work, host hint, 8 programs per SM). The hint is the dispatcher's host estimate
  (`_host_valid_tokens_estimate` = this rank's bucket or eager token count x EP); permuted rows are bounded by
  tokens x min(topk, E_local) + min(E_local, pairs) x (BM - 1). The hint never decides which rows are computed:
  NVLS does not sync bucket sizes across EP ranks, so the gathered count can exceed it, and the loops cover that.
- `permute_live`: MCore's count (persistent variant) and aligned-offset kernels, then one persistent kernel over the
  valid tokens that copies each local (token, expert) pair to an atomically claimed row of its segment and writes
  the token's whole inverse-map row (-1 where it does not route). Same contract and buffer shapes as
  `permute_tokens(..., return_batch_invariant_inverse_map=True)`. Row order inside a segment was already arbitrary
  (atomics); outputs do not depend on it.
- Tile from the bucket: `select_engine_config(hint x topk / (E_local x EP))`, the rule EP = 1 already applied
  (there the buffer is the bucket).
- The down-projection prescale (14b) moved into the activation (prob x 2^-s) and unpermute (y x 2^s) kernels: the
  same fp32 multiplies, on live rows only.

**Why bitwise identical.**
- Per-row arithmetic is unchanged: each GEMM tile runs the same K loop (exact fp16 products, K in order into one fp32
  accumulator); the activation and the fp64 expert-order sum run the same ops per element. The grid only decides
  which program computes a row.
- Tile choice: a row's bits do not depend on BLOCK_M/N/K (14b). EP = 1 has always picked the tile per bucket and
  passed every engine-q exactness run against the cuBLAS learner. The new unit test checks every engine tile, each
  with three grid sizes (including one program looping over all tiles), against cuBLAS at Nano fc1 (1856 x 2688)
  and fc2 (2688 x 1856) shapes, with zero, -0, subnormal and 60000 rows [T].
- `engine_experts` vs the pre-change path (MCore permute, padded tile, non-persistent GEMM, full-buffer activation):
  bitwise on every live token for relu2 and swiglu, prescale 0 and 3, EP 1 / 4 / 8, and a hint below the live count.
  Under CUDA-graph replay, with live rows changing from 5 to 300 against a 64-token capture hint, it is also
  bitwise [T].

**Scaling.** Per-rank work is O(live gathered tokens x topk) for permute and O(local live rows) for GEMM, activation
and unpermute. The fixed overhead is at most 8 x SMs early-exit programs per kernel, independent of EP and
max_tokens; the old path scaled with max_tokens x EP. The buffers keep the worst-case size (memory, not time).
Gathering every token to every rank stays the AllGather-V dispatcher's design.

**Results [M].** One Nano MoE layer (single GPU, EP-rank buffer emulated, CUDA-graph replay), x 23 MoE layers
(`thundersync_work/research/moe_livegrid/moe_layer_bench.py`):

| tokens/rank (bucket) | EP4 old | EP4 new | EP8 old | EP8 new |
|---|---|---|---|---|
| 1 (64) | 22.3 | 2.1 | 44.2 | 1.9 |
| 8 (64) | 23.8 | 3.4 | 44.7 | 2.5 |
| 32 (64) | 24.2 | 3.7 | 44.9 | 2.8 |
| 128 (128) | 24.3 | 4.2 | 45.1 | 3.0 |
| 1024 (eager) | 27.8 | 8.5 | 48.9 | 7.5 |
| 4096 (eager) | 43.9 | 23.0 | 65.9 | 22.8 |

ms per engine step. The EP4 old column reproduces the full-model profile (23.7 ms, 14n).

Full Nemotron-3-Nano, engine-only decode bench (`thundersync_work/research/moe_livegrid/livegrid_bench.sh`, one
4-GPU node; old = ts-nano-perf, new = bacfde29) [M]:

| | old | new |
|---|---|---|
| EP4, graphs on, N = 1 / 8 / 32 / 128 per rank (C 512), ms/step | 27.9 / 29.0 / 29.0 / 29.0 | 7.6 / 8.5 / 8.5 / 8.5 |
| EP4 device ms/step at N = 8 (graphs off): MoE kernels + expert activation | 14.0 + 9.7 | 2.2 + 0.1 |
| EP2, graphs on (dump points), ms/step | 17.8-19.0 | 7.3-8.5 |

- Bitwise: greedy tokens and raw log-probs, old vs new, every rank. EP4 and EP2, each with graphs on and graphs off,
  over 5 points (prefill buckets from 48 x 8 to 512 tokens, decode at 1-8 requests per rank): 23 requests x 48 tokens
  per rank, 0 token and 0 log-prob bit mismatches.
- Nano L8 engine-q exactness, EP2: PASS 4/4 for ⊥-BV + momentum and for ⊥-BV + current. GPU unit suite: 307 + 31
  pass (lg-val-bacfde29).
- EP8 [D]: the old one-layer cost doubles with the buffer (EP8 - EP4 = +21 ms/step over 23 layers) and the new
  one does not grow (-0.9). Estimate: ~50 ms/step old, ~8-9 ms/step new; the NVLS collectives add a little at 8
  ranks.
- What remains at EP4 decode (new, 8.5 ms): grouped GEMM 1.7 ms, which is the expert weight reads of the touched
  experts; dense GEMM 1.3; elementwise/copies 3.2; Mamba 0.9; NVLS collectives ~0.5.

**Upstream candidate (MCore, not opened).** `permute_tokens` / `_permute_tokens_kernel` loop to `max_pairs`, and
`_init_permutation_map_kernel` / the inverse-map `torch.full` cover the whole buffer. Bounding them by `valid_pairs`
/ `valid_tokens` is the same CUDA-graph-safe change, and it is independent of this recipe.

## 14p. Learner prefix-state cache for exact piece verification (`thundersync.learner_prefix_cache`)

*Branch ts-learner-prefix-cache (../RL-prefixcache). Notes: thundersync_work/research/agent_prefix_cache.md.
Problem statement: thundersync_work/research/piece_verify_scaling.md.*

**Problem [code].** `_continue_sticky` verifies every continuation piece (128 tokens) as the row prompt + prefix +
piece, and `verify_drafts_block_engine` forwards the whole row (`_training_forward_rows`). A continued row of length L
costs sum_j F(P + j c): O(L^2 / c), ~32x one forward at 8k and ~128x at 32k.

**Mechanism.** Per continued row, each learner rank keeps the state of its own layers at chunk boundaries s (multiples
of the Mamba chunk size from position 0):
- attention K/V of positions [0, s) as fed to core attention (this rank's KV heads);
- Mamba conv inputs (xBC) of [s - d_conv + 1, s) and the fp32 SSM state at s (this rank's heads).

The next piece's row is forwarded only from the largest cached boundary s <= prompt - 1 whose tokens [0, s) equal the
row's (longest common prefix, so a rejected piece falls back to an earlier boundary). At most chunk - 1 positions
before the piece are recomputed; per row the work is O(L) total. After the forward the rank keeps the start boundary
and those in (start, n - 1] (at most `max_keep` = 4, evenly spaced, for long pieces).

**Why the logits are bitwise the whole-row forward's (batch invariance).**
- Attention: FA4 varlen over the new queries and cached + new keys, called through TE's own FA4 entry point with TE's
  causal window and num_splits (anchored split-KV when `bi_split_kv` is installed, num_splits=1 otherwise). Key
  blocks are aligned to absolute position 0 in both forwards, so each query row reduces the same blocks in the same
  order.
- Mamba (`mamba_ssd_parity.ssm_prefix_forward`): the conv of every output position reads the same d_conv inputs; the
  chunked SSD scan restarted at a chunk boundary from that boundary's fp32 state runs the same per-chunk kernels and
  the same state recurrence `s_c = exp(dA_c) s_{c-1} + S_c` as the full scan (the identity the engine's BI prefill
  with prefix caching relies on). Pinned kernel configs as in training.
- Everything else is per token: row-invariant GEMMs (BI), norms, embedding, dropless MoE; sticky anchors are taken
  per position from the segment start (`fetch_anchors(lo=...)`).

**Who decides what (no new collectives).** The driver (`thundersync_rl/prefix_cache.py`, `PrefixCacheDirectory`)
decides membership and placement and sends the same lists to every rank of a replica, so all TP/PP ranks pick the
same starts (same shapes) without talking to each other:
- placement: a continued row is verified on the DP rank holding its state; new rows LPT by tokens they will forward;
- budget (ts-carried-walk): per rank at most `max_gb`; `learner_prefix_cache_size: auto` (default) sizes it to the
  step's continued rows per rank x one row's bytes at the row cap (positions x K/V bytes + kept boundaries x state
  bytes, the largest per-rank costs the workers report). `learner_prefix_cache_policy: pin` (default) admits a row
  only if its reservation at the row cap fits and never evicts; the former least-recently-verified eviction
  (`lru`) misses on every access once the working set exceeds the budget, because continuation rounds visit rows
  cyclically (unit test `test_lru_thrashes_under_cyclic_rounds_and_pinning_does_not`): at 128k that silently fell
  back to O(L^2 / c). Uncached rows forward whole (slower, same bits). Metrics `spec/prefix_cache_eligible_hit_frac`
  (hits / pieces of rows forwarded before), `_admitted`, `_denied`, `_evicted`, `_need_gb_per_rank`,
  `_budget_gb_per_rank`, `_max_rank_gb`; a `[prefix cache] WARNING` line below 0.9 or on any denial/eviction;
  `learner_prefix_cache_min_hit_frac` > 0 raises below it;
- lifetime: released when the rollout is complete (piggybacked on the next verify call), at the end of the step's
  continuation rounds, and dropped by the worker whenever its weights version changes (optimizer step, draft
  weight swap);
- residency: within the budget the worker keeps `gpu_gb` on device; colder rows move to pinned host memory (LRU) and
  come back when used. Residency never changes shapes or bits.

EP: rows per rank already differ under EP; the lockstep counts forward calls, which the cache does not change. PP: each
stage caches its own layers; the pipeline's sequence length is the segment length, identical on all stages.

**Interactions.** Training forwards (streamed over tails, canonical or block dispatch, activation recompute) run the
whole rows with the hooks inactive (they only act inside `PrefixStateCache.forward`). verify0 (the drafts' first
verification) forwards whole rows and caches nothing (a row's first piece then forwards whole once). Multi-turn: the
cached prefix stays valid across turns (observations extend the row). Not built yet: RoPE models (absolute position
offsets), context parallelism, sticky rounding's theta_n anchor forward, verify_on=inference/both, caching verify0
(needs the accepted length on every PP stage).

**Per-rank memory [D].** Entry = n x KV/token + (kept boundaries, typically 2) x state.
| | KV per token | state per boundary | per 32k row | per 128k row |
|---|---|---|---|---|
| Nano (6 attn, 23 Mamba; TP1, EP8 does not shard them) | 6 KiB | 46.8 MiB | 0.28 GB | 0.84 GB |
| Ultra [H] per TP4 x PP4 group (12 KB/token, 0.40 GB state) | 12 KB | 0.40 GB | 1.2 GB / group = 0.08 GB per GPU | 2.4 GB / group = 0.15 GB per GPU |
At 16 continued rows per learner rank: Nano 4.5 GB (32k) / 13 GB (128k) per rank (of which `gpu_gb` on device);
Ultra 1.2 / 2.4 GB per GPU. KV heads are replicated, not sharded, when TP exceeds the KV head count.

**Validation [M]** (thundersync_work/research/prefix_cache/*.log; check script tests/functional/check_prefix_cache.py:
continuation rounds of 128-token pieces with random rejections, the cached forward of each round's rows in one batch
with mixed starts vs the whole rows, logits compared with torch.equal).
- Bitwise, 0 mismatches, max |d| 0 everywhere: Nano L8 quick (15 pieces, 4 rejections, 12 hits, host eviction);
  L8 sticky anchors, 6 rows to 4k (201 pieces, 57 rejections, 195 hits; sticky kept the anchor on 17.9% of
  token-layers); L8 plain, rejection prob 0.5; L16 sticky; L8 TP2 (sequence parallel) sticky; L8 EP2 sticky.
- In-loop (local Nano L8 EP2 ts_por, 1k cap, parity reward, expert-grad noise, CHECKS=1 and
  THUNDERSYNC_PREFIX_CACHE_CHECK=1; runs_prefixcache/pcl-x-1k-c): exactness EXACT 4/4 steps (mismatch 0, engine q
  exact), 123/123 cached piece forwards bitwise equal to the whole-row forward. At a 1k cap most continued rows need
  one piece, whose first forward is whole (verify0 is not cached): 3 hits. A first run hung: the check's extra forward
  was not matched by EP dummy forwards on a rank without rows (fixed 0472a8b3).
- Piece forward time, one row, prefix P + 128-token piece, cached start = last boundary (1 GPU, median of 5):

  | P | L8 full | L8 cached | L16 full | L16 cached |
  |---|---|---|---|---|
  | 1k | 98.5 ms | 96.3 | 200.4 | 189.3 |
  | 2k | 100.4 | 96.7 | 206.1 | 191.1 |
  | 4k | 101.1 | 94.7 | 206.7 | 189.2 |
  | 8k | 108.3 | 93.9 | 217.4 | 189.2 |

  At L8/L16 and <= 8k a one-row forward is dominated by per-call cost (~90 ms per 8 layers on this node, flat in
  length: research/determinism/fwdlen.json, 85 ms at 1k, 98 ms at 8k, 122 ms at 16k for L8). The cache removes the
  length-dependent part: cached time is flat in P, full grows 10 ms (L8) / 17 ms (L16) from 1k to 8k. Measured gain
  at this scale: x1.0-1.2.
- Projection [D] to full Nano (52 layers ~ 3.3 x L16) and 32k: length-dependent work of a whole-row forward
  ~ 2.4 us/token per 8 layers (fwdlen 8k->16k) -> ~16 us/token full depth, plus attention growing with L; a 32k
  piece forward ~0.5 s of token work vs ~4 ms for a 256-token cached segment. Per continued 32k row: 256 pieces x
  ~0.25 s (average 16k) ~ 60 s of learner work without the cache, ~1-2 s with it. The per-call cost is shared by all
  rows batched in one verification call and is not changed by the cache; it needs its own fix (profile first).

## 14q. m2n refit: every learner rank sends its own shards (`refit_transport: mcore`, 2026-10-15)

**Why.** The default refit (`refit_transport: null`) exports the whole model in HF layout and broadcasts it from
learner rank 0 to every engine rank (`packed_broadcast_producer src=0`). Every engine rank receives the full model P and
one process sends it, per refit, and a ThunderSync step refits twice (θ̂_{k+1} for drafting, θ_k when some row decodes
fresh). Measured at full Nano (P = 63 GB, EP8): 1.32–1.51 s per θ refit and 0.76 s per θ̂ refit, ≈ 45 GB/s
(`runs_nano_scale/nano-full-*`). The cost grows with P: ≈ 5 s per refit at Super, ≈ 24 s at Ultra, ≈ 44 s at 1T.

**What.** `refit_transport: mcore` with `mcore_generation_config.refit_backend: nccl` (or `nccl_m2n`) uses Megatron
Core's native refit (`megatron.core.resharding`). The plan is built once per lane and cached. It works on Megatron names
for every tensor, including Mamba, attention, embeddings, the head, the experts and the router `expert_bias`. There is no HF
export and no rank-0 path. Each engine rank receives exactly its own shards. Each shard's source is balanced over the
learner's DP replicas (`select_src_metadata_balanced`), and each learner rank sends straight from its live parameters.
ThunderSync keeps engine TP and EP equal to the learner's (section 14j), so each tensor's plan is a whole-shard copy from
one learner rank. With PP, an engine rank receives from each stage that owns its layers. `nccl` is batched NCCL
point-to-point (`NCCLCopyService`). `nccl_m2n` is native NCCL M2N (`NCCLM2NCopyService`), which needs nccl-extensions'
`nccl.m2n`. The nemo-rl-fa4b image does not have it (`ModuleNotFoundError: nccl.m2n`), so runs use `nccl` for now.

**Bitwise contract.** After an m2n refit the engine must hold bitwise the weights the packed broadcast writes, for θ and
θ̂. Three pieces make that hold:
- **fp16-held weights.** The engine holds its dense GEMM weights, untied head and router (`hold_fp16_weights`) and its
  expert buffers (`fp16_moe`) in fp16, but the plan's wire dtype is the learner's bf16. The plan is built at
  `init_collective`, before engine initialization converts the weights. fp16 and bf16 have the same width, so a direct
  receive would write bf16 bits into fp16 storage silently. `Fp16HeldWeightReceive` (`megatron_worker.py`) is a
  receive-side transform. Each tensor whose current dtype is fp16 is received into a bf16 buffer, then written with
  `copy_`, which is the broadcast path's bf16 → fp16 rounding. The transform subclasses `MXFP8ReshardTransform` only
  because the M2N service accepts that type. Before this change the expert buffers were already fp16 under `mcore`, so
  that combination wrote raw bf16 bits into them.
- **Inference tensors.** The destination swap runs under `torch.inference_mode()`, as the colocated reshard does: engine
  tensors allocated during generation are inference tensors.
- **Derived caches.** MCore's `refresh_module_caches` refreshes derived caches such as the Mamba `-exp(A_log)` cache,
  as the bridge import does.

**Lanes.** Each lane gets its own native refit group: the learner ranks plus that lane's ranks, with no idle ranks, as
M2N requires. `keep_refit_group` / `select_refit_group` file and restore `model_update_group`, `refit_pg`, the copy
service and the rank offset. The plan's cache key is the same for every lane, so the learner clears MCore's plan cache
before each new lane's init (`clear_refit_plan_cache`). Otherwise it would skip that lane's collective plan build and
the lane's engine ranks would wait forever. The plan itself is identical across lanes (same group-rank layout).

**θ̂.** θ̂ goes through the same path. `load_draft_weights` writes θ̂ into the learner's parameter buffer: each
distributed-optimizer shard owner computes its slice, then a DP all-gather runs, as after an optimizer step. No rank
gathers the model. The m2n refit then sends each rank's TP/EP/PP-local θ̂ shards. Not built yet: a direct push from the
distributed-optimizer shards, which would skip the load all-gather and the restore (`draft/load_gather_s` 1.30 s at full
Nano [M]). It needs a plan whose sources are DO-bucket ranges rather than whole TP-local parameters. Engine fan-in would
be DP × PP sources per tensor, and per-rank bytes would be unchanged.

**Per-rank cost per refit.** R is the bytes one engine GPU holds per weight version. P is the whole model. Bandwidths
are [H]: NVLink-domain P2P ≈ 400 GB/s per GPU, inter-rack IB ≈ 45 GB/s per GPU (the measured broadcast rate). The m2n
figures exclude fixed software overhead. nccl_reshard's measured 2.32 s at Ultra-v3 (1.1 TB, TP8xEP32xPP2 → TP8xDP8)
bounds it from above.

| model | P (bf16) | engine replica | R per engine GPU | broadcast per refit (P / 45 GB/s) | m2n per refit, NVLink / IB (R / BW) | learner send per rank |
|---|---|---|---|---|---|---|
| Nano | 63 GB | EP8 (8 GPUs) | 10.9 GB [M `draft/engine_stage_gb`] | 1.32–1.51 s [M] | 0.03 s / 0.24 s | ≈ R (8 learner ranks) |
| Super | 240 GB | TP4×EP8 (8) [H] | ≈ 32 GB [H: 8% dense] | ≈ 5.3 s | 0.08 s / 0.7 s | P / L |
| Ultra | 1.1 TB | TP8×EP64 (64) | ≈ 17 GB [D, future_scale_issues §2.2] | ≈ 24 s | – / 0.38 s | 1.1 TB / 512 ≈ 2.1 GB |
| 1T | ≈ 2 TB | TP8×EP64 | ≈ 39 GB [D] | ≈ 44 s | – / 0.87 s | ≈ 2–4 GB |

Other per-rank costs:
- **Engine.** Transient bf16 staging for the fp16-held tensors of one execution batch, ≤ `refit_execution_batch_bytes`
  (default 2% of GPU memory, ≤ 5 GiB).
- **Learner.** Nothing extra: it sends views of its live parameters.
- **Plan build.** Once per lane, an `all_gather_object` of every rank's tensor metadata: O(ranks × local tensors)
  entries on every rank. At Ultra that is ~0.6k ranks × ~1.5k tensors [D]. It is one-time, not per refit.
- **Nothing per refit through one process.** The engine's fan-in per tensor is learner PP (one stage owns a layer).

**Crosscheck (debug).** With `THUNDERSYNC_REFIT_CROSSCHECK=1` (`refit_m2n.py`), every refit runs these steps:
1. The packed broadcast, as the reference.
2. A bitwise digest of every engine parameter and buffer, including the fp16 expert buffers and the Mamba cache. The
   digest is a position-weighted sum of the raw bits modulo two primes.
3. The engine overwrites every refit tensor with NaN or -7.
4. The m2n refit.
5. The digest again, compared per rank and per tensor.

Each refit prints one line, `[m2n-crosscheck] PASS|FAIL refit=<n> ... broadcast_s m2n_s`. The poison step means a
tensor the m2n refit skipped cannot pass by keeping the broadcast's values. The run continues on the m2n weights, so the
same run's exactness checks (mismatch 0, engine-q, route check) also test them. With `=strict` the run raises on the
first FAIL.

**Status.** Unit tests (`tests/unit/test_refit_m2n.py`):
- the digest changes on any one-bit change;
- through MCore's real executor, the fp16 receive equals the broadcast's `copy_` bitwise, including fp16 subnormals and
  overflow;
- a direct receive does not equal it;
- lane state is filed and restored per lane.

GPU runs are queued (Nano L8, one engine EP2 and lanes 2×EP1, crosscheck plus exactness, with a same-tree broadcast
control). The measured refit seconds go in this section when they report.

## 14r. Verify-in-train (VIT): design B, exactness, debug check (2026-10-15; not built beyond the check)

*Notes and numbers: `thundersync_work/research/agent_verify_in_train.md`. Branch ts-verify-in-train (../RL-vit).*

**Idea.** For a group whose G drafts are all finished at t_{n+1}, the block-verification forward at theta_{n+1} is the
group's canonical training forward: decisions are read from the training forward's `next_token_logprobs` (same
function of (p, q, u) as `verify_drafts`), then the backward runs. Verification of those rows costs no forward.

**Design B (chosen).** Only groups whose drafts are all finished AND whole-accepted contribute gradient from the fused
call; for them the draft rewards are the final rewards (no provisional advantage). A micro-batch in which the forward
finds a rejection is masked like a padding row (sample_mask = token_mask = advantages = 0; the backward still runs on
every rank, keeping EP/PP lockstep) and its group trains later in its canonical (tail-last) position. With
group-aligned micro-batches and deterministic kernels the accumulated gradient equals canonical training bitwise.
**Design A (provisional advantages + linear correction over all accepted rows) is rejected**: exact only in exact
arithmetic (fp two-term sum; the replacement row trains in another micro-batch), and it saves <= 0.03 s/step at L8.

**Measured coverage (Nano L8 1k, steps 2-5).** Fully-finished groups 0.48 (ord-p-l2-m128) / 0.69 (lanes2r-l2), 0 with
one lane; fusable drafted-token share 0.50 / 0.80; P(rejection | finished group) 0.03-0.04. Group-major admission
(untested) would raise m128's 0.48 toward 0.89.

**Debug check `THUNDERSYNC_VIT_CHECK=1`** (`vit_check.py`; first `THUNDERSYNC_VIT_CHECK_CALLS` final calls per step):
regular call twice (control), training-forward p vs generation_logprobs (verifier/engine p) bitwise, masked-row
content independence (exact zero), row-absent and correction fp differences (information); the regular result is
what trains. Bitwise needs `policy.megatron_cfg.deterministic_training=true` (14n). Cluster line vit-x; result: see the
agent note.

**Decision (2026-10-15).** Plan of record: hybrid placement (VIT for fully-drafted all-accepted groups, engine-side
verification inside the continuation prefill for the rest), linear correction dropped. VIT rows are exact without
`deterministic_training` (decisions come from the training forward itself); determinism is only for bitwise tests.

**Placement.** VIT for fully-finished groups, engine-side verification (ts-engine-verify, not merged) for unfinished drafts (ideally inside the
continuation's own prefill). Per-rank cost of VIT: no memory beyond a training micro-batch; q (4 B/token) and the row
seed to the row's last PP stage; one [V/TP] head row per rejected position.

**Built (opt-in `thundersync.verify_in_train`, 647deaec).** Pooled ⊥-BV, sticky routing, streaming over the tails
with the canonical order, group granularity, PP 1. Pools whose first G drafts are finished skip verify0. Their draft
rewards come from the rollout function run on the draft text. Their training calls (one micro-batch per group, on
ranks by tokens in group order) are the step's first ordered learner calls. Each row's decision is
eta_g(row seed) <= b_g(M_j), computed in the loss from the training forward's log-probs, bitwise the verifier's rule.
A failed micro-batch's loss is multiplied by 0 (the backward runs on every rank), and its counts and metrics are
removed. Its pool is then verified the regular way with the same per-row streams, its tails run before the refit,
and it trains in its canonical position (after the groups ready at verify0). Passed groups leave the planner's order
(`mark_pretrained`). Their rollouts are served from the drafts, with the training forward's log-probs as generation
log-probs, so the in-loop mismatch check covers p == verifier bitwise. vit-x: training p == verifier p bitwise
(278k tokens). A masked ROW inside a live micro-batch is not bitwise under EP (routing changes the expert GEMM
shapes); VIT only ever masks whole micro-batches, which the check tests separately (wz/wm).

## 14s. Run readiness: checkpoints, resume, epochs, validation, always-on guards (branch ts-run-ready)

Before this the loop could not run a long job (`research/future_training_dynamics_issues.md` issues 1-4,
10-12, 14-15): no checkpoint, step counter hard-coded to 0, a stale NeMo-RL checkpoint loaded silently,
one epoch, no validation, verification seeds reused after a restart. Code: `run_state.py`,
`sentinel.py`, `grpo_loop.py` (`thundersync_grpo_train`), `run_thundersync_grpo.py`.

**Quiescent boundaries.** Checkpoints and validation run only at step boundaries where nothing
speculative is in flight: the step before a boundary drafts nothing past it (`Boundaries.draft_limit`
replaces `max_steps` in every drafting decision, lookahead > 1 included), so no draft, plan, anchor,
⊥-BV pool, lane pin or staged momentum weight crosses it. The step after a boundary decodes every rollout
fresh at θ_k (exact, as at step 0); an uninterrupted run with the same `save_period` does the same, so a
resumed run continues exactly as the saving run would have. Cost: one unspeculated step per checkpoint /
validation. Boundaries are `save_period` multiples, `val_period` multiples (from `val_start_at`), the
last step, and a timeout save (`checkpoint_must_save_by`) requested one step ahead. Leftover plans are
dropped at every step end (`SpeculativeGeneration.end_step`): a later row with the same prompt tokens
(duplicates, next epoch) must never claim a plan verified against an older version; ⊥-BV prompt selection
is reset with them. Scope of that hazard: plans are keyed by prompt tokens and are otherwise cleared
only when the next verification starts (`_finish_verify`, `_finish_verify_null`, `_verify_chunked`), so
a stale plan can be claimed only by a row of a step that runs no verification before its rollouts, with
the same prompt tokens. Before boundaries every step >= 1 verified first (drafts always exist; lanes
clear the cohort lane's wrapper at its `begin_sticky`; lookahead > 1 gives every step a cohort), so
the frozen snapshot RL-nano-perf-scale 5b3530de is not affected. With boundaries (no drafts for the
step after) it would be an exactness bug: DAPO-Math-17k has 1,791,700 rows but 17,398 distinct prompts
(up to 400 copies), so a 64-prompt batch shares a prompt with the previous one with probability ~0.2.

**What a checkpoint holds** (exact continuation):
- weights + optimizer + LR scheduler: NeMo-RL `policy.save_checkpoint`, Megatron distributed checkpoint
  (`fully_parallel_save`, every rank writes its own shards; no rank-0 gather; async save supported via
  `finalize_async_save`). The momentum draft θ̂ is a function of (θ, m, v, step, lr) only: nothing extra.
- `thundersync_state.pt` (driver, a few KB): next step; data position = (dataloader `state_dict` taken
  *before* the next untrained batch was fetched, epoch, prompt-drop re-batching carry) — the prefetch
  deque (`draft_lookahead + 1` batches) is rewound by construction; verification / draft seed counters
  per speculative wrapper (lane); lane versions; a config fingerprint (seed, batch shape, epochs,
  save_period, speculation knobs; a changed value refuses the resume). `train_dataloader.pt` is written
  too (same state, NeMo-RL format).
- Not saved (dropped at the boundary by construction): drafts, plans, anchors/routes, ⊥-BV pools,
  hidden-state blocks, the momentum host stage, learner prefix cache. Learner RNG: not used (no dropout).

**Resume is explicit** (`run_state.resolve_resume`, before `setup()`): NeMo-RL's setup loads the latest
`checkpoint_dir/step_*` whether or not checkpointing is enabled. Checkpointing disabled + a checkpoint
present = error; enabled + a checkpoint without `thundersync_state.pt` (not a ThunderSync boundary) =
error; enabled + ThunderSync checkpoint = resume at its step (lanes start at ("theta", step)).

**Epochs.** `PromptStream` iterates `grpo.max_num_epochs` passes of the StatefulDataLoader (new shuffle
per epoch), each batch tagged with its resume state; the prefetch deque spans epoch boundaries; the run
ends at `max_num_steps` or when the last epoch is exhausted. Unit test: resuming from every tagged
position (with and without re-batching) reproduces the remaining sequence.

**Validation** (`grpo.val_period`, `val_at_start`, `val_at_end`; NeMo-RL `validate`) at boundaries
only: the engine (non-lanes) or the next cohort lane is refit to θ_{k+1} (the next step then needs no
refit), checked by the weight sentinel, and the raw engine runs the eval: plain top-k routing (eval
requests never pass the speculative wrapper, carry no anchors, and the engine applies anchors only to
requests that carry them).

**Verification seeds** (`run_state.SeedStream`): seed = hash(grpo.seed, stream, step, counter); stream =
"verify/<lane>" or "draft/<lane>"; counters checkpointed. Two lanes no longer share η streams (both
counted from 1 before), two seeds no longer share η, a resume never reuses η.

**Always-on weight sentinel** (`sentinel.py`, `thundersync.weight_sentinel: true`; runs with CHECKS=0):
after every refit — initial weights at step 0 / resume, θ_k, θ̂_{k+1} (inside `stage_draft_weights`,
while the learner holds θ̂), drafter lane refits, validation refits — every engine rank digests its
weights and every learner rank digests the same tensors cast to the engine's storage dtype (fp16 GEMM
weights); per tensor two 64-bit sums over the raw bits (plain and position-weighted). Names are made
layout-independent (wrappers stripped, global layer numbers, global expert indices = EP rank × local
experts + local index), so engine and learner EP/PP may differ. Replicas within a side must agree;
across sides every common tensor must agree bitwise, else `WeightSentinelError` names the tensors. Also:
the momentum host stage reloaded = the θ̂ digest; the stashed LM head = the live θ̂ head (the head the
engine drafted with); `expert_bias` buffers included and must never change on the learner.
Compute tensors, not only parameters: the engine digests the storage its kernels read — the
concatenated / fp16 expert buffers `_fc1_weight` / `_fc2_weight` [E, N, K] per expert as
`<param>#buf` (compared with the learner's parameter cast to the buffer dtype; a buffer that stopped
aliasing its parameter, or a raw-byte refit that wrote bf16 bits into fp16 storage, is caught by
name) — and its fp16-held GEMM / router / LM-head weights in their fp16 storage. On the learner, the
fp16 expert-weight cache (`fp16_moe._WEIGHT_CACHE`, keyed by identity + pointer, not value) is checked
against the parameters at every θ check (not while θ̂ is loaded: it keeps θ_k's casts by design).
Not checked (printed with the coverage): per-call casts that are never stored (bf16 parameters cast
inside each GEMM when the engine does not hold fp16 weights, e.g. `refit_transport` mcore/m2n — a
deterministic function of a checked tensor), the hi/lo split of the fp32 head (activation side; no
cached weight copy exists on this tree), KV / Mamba state and CUDA-graph activation buffers. The
coverage list (pairs, compute buffers and how many alias their parameter, fp16-held parameters,
learner cache slices, engine-only / learner-only names) is printed at the first check and counters
are logged per step (`sentinel/*`); zero common tensors is an error (vacuous). Per rank: O(params/rank) reads, no gather, a dict of
(name -> 4 ints) to the driver per refit. Untested on GPU yet (queued).

**Non-finite verification** (`block_verification.check_verification_inputs`, all verify paths): a
drafted token with engine log q = -inf (q = 0: the logged q is not the sampling law), NaN/+inf learner log
p, a non-finite row logsumexp, a non-finite stop probability or an emission law without positive finite
mass raises `NonFiniteVerification` with the row, step seed, segments and weights version (counted,
printed). log p = -inf alone is legitimate (rejected with probability 1; counted). Before: the draft was
silently cut through a NaN, or `multinomial` hit a device assert.

**Draft collection never uses streamed frames** (`worker.collect_drafts` / `peek_drafts`): a running
draft without an engine record finished (or failed) in the engine after the records were read; its
frames may skip tokens after an eviction. The collector now waits for its final frame (the merged record)
before taking any record content (no late frame can land after), aborts nothing finished, treats an
empty unrecorded stream as the empty prefix (no record and no token at the cut), and raises if the final
frame does not come (`THUNDERSYNC_COLLECT_FINAL_TIMEOUT_S`, 30 s). Counted: `spec/collect_unrecorded`,
`spec/collect_waited_final`, `spec/collect_unrecorded_empty`.

**Expert-bias updates + momentum drafts** are refused (config validation and at `load_draft_weights`):
the momentum stage stages parameters only, not the `expert_bias` buffer (not built yet).

**Multi-turn rows are prefix extensions** (always on): `PrefixGuardGeneration` checks per turn that the
context starts with the previous turn's input + generated ids (and the final message log), the drafted
multi-turn replay checks prompt + prefix; a violation is `RowPrefixError` (re-raised after the rollout
loop, which swallows per-turn generation errors). Templates with `truncate_history_thinking` (Nemotron;
default True) are refused for multi-turn unless `chat_template_kwargs.truncate_history_thinking=false`
(set in `ab_base.yaml` and the multi-turn scripts).

**Run-to-run determinism** (needed only for bitwise resume tests; opt-in `deterministic_training`):
rr-resume-nano @cf3bc405 showed two runs with bitwise-identical rollouts (keyed fresh sampling) whose
gradients already differ at step 0, while in-process replays were bitwise equal. Cause found: mamba_ssm
filters its backward kernels' autotune candidates to one deterministic config only for modules imported
in deterministic mode; they are imported before `set_deterministic_mode`, so 42 kernels kept every
candidate and each process autotuned its own tile (cached within the process, so in-process replays
agreed). Now pinned at install (`mamba_ssd_parity.pin_mamba_ssm_autotuners`, mamba_ssm's own rule), plus
Megatron's deterministic mode (`torch.use_deterministic_algorithms(True)`, warn-only because the BI
recipe's `CUBLAS_WORKSPACE_CONFIG=:0:0` — no split-K, deterministic — is not on torch's allowlist;
`fill_uninitialized_memory=False`; NCCL_ALGO must be Ring/Tree, warned otherwise). Per-parameter
gradient digests before each step (`THUNDERSYNC_STEP_DIGEST`) locate any remaining source.
Resume must keep `grpo.max_num_steps` / `max_num_epochs` (Megatron's checkpointed LR/WD schedule is
sized from them; checked before setup). Kill simulation: `THUNDERSYNC_DEBUG_STOP_AT=<step>` exits the
driver (`os._exit`) right after that durable checkpoint.

**Not built yet** (validation refuses): checkpoints with `rollout_lag=1`, with
`data.use_multiple_dataloader`, with `ft_save_period`; checkpoints / validation on the non-streaming
lanes loop (`stream_training_over_tails=false`); validation with `rollout_lag=1`.

**Tests.** CPU: `tests/unit/test_run_state.py` (seeds, epochs, every-position data resume, boundaries,
explicit resume, fingerprint, run-control validation), `tests/unit/test_run_ready_guards.py` (sentinel
digests / names / dtype cast / replicas / expert_bias, non-finite inputs, collect without frame
fallback, prefix guard, template check, expert-bias guard). GPU kill-and-resume on Nano L8:
`thundersync_work/research/run_ready/resume_exact.sh` — (S) sync arm with reproducible rollouts
(`THUNDERSYNC_DEBUG_KEYED_FRESH`: keyed engine sampling, exact, independent of batch composition) and
deterministic training: 4 steps vs 2 + resume 2, per-step params / optimizer / rollout / learner log-prob
digests bitwise; (T) ts_por (sticky + momentum + ⊥-BV) with validation: boundary digest (params,
optimizer, θ̂, next prompts, seed counters) saved = loaded (enforced in-run with
`THUNDERSYNC_RESUME_DIGEST`), exactness checks after the resume, sentinel; (X) stale-checkpoint refusal.
A speculative run is not bitwise reproducible run to run (engine sampler and deadline cuts), so (T)
compares state, not trajectories.

## 14t. Carried-walk continuation verification (`thundersync.sticky_continuation_decision`)

*Opt-in (default `every` = section 14k unchanged). Branch ts-carried-walk (../RL-cwalk). Notes, MC and
measurements: `thundersync_work/research/agent_carried_walk.md`. Code: `carried_walk.py`, `engine_q.verify_engine_q_walk`
/ `verify_engine_q_scan`, `_continue_sticky` (driver), `PrefixStateCache` pins.*

**Problem [code].** Continuation pieces were verified as independent blocks: each piece restarts the walk at b = 1 and
ends with a terminal test, so a row of L tokens in pieces of c takes ~L/c terminal tests, each rejecting with
probability ≈ the walk's dip over that piece. Modeled [D, `research/future_longctx/piece_mc.log`, v = 1.1e-6]:
rejections per continued row 2.43 / 9.5 (32k / 128k) at c = 128, 0.36 / 0.72 with growth 1.0, 0.14 / 0.27 as one block.

**Rule.** The pieces of one engine stream form one *block*, decided only at *decision points*. Between them a piece
only advances the walk, log b_i = min(0, log b_{i-1} + log p(x_i) − log q(x_i)), and the *floor* f (last position with
b_f = 1). Per row the state is two scalars (log b, f). At a decision point T: eta_T ≤ b_T keeps the block (no bonus;
the stream goes on with a new block); otherwise tau = max{ i ∈ (f, T) : eta_i ≤ h_i } with fresh eta_i, or f if there
is none (h_f = 1), and y ~ norm(max(b_tau p_tau − q_tau, 0)). Decision rules (`DecisionRule`): `row` (only at the
stream's end: EOS, cap, turn end, anchor horizon), `dip` (also at a piece end where −log b ≥
`sticky_continuation_decision_dip`), `every` (each piece end: today's law). Forward pieces keep their sizes
(`sticky_continuation_chunk_tokens` / `_growth`); decisions are a separate schedule.

**Exactness.** (1) *Martingale lemma.* Forward block verification draws, at each position i not decided, a hit
(eta_i ≤ h_i) and the next token x_{i+1} ~ q_i. Let O_i be the output if the block were decided at i (keep with
prob b_i, else residual at the last hit). Given everything up to i, O_{i+1} has the law of O_i: the mass on
"prefix x_{≤i} then token y" is min(q(y), b_i p(y)) (kept) + h_i (1 − b_i + r_i) (b_i p(y) − q(y))_+ / r_i (hit at i)
= min(q, b p)(y) + (b p − q)_+(y) = b_i p(y), and the no-hit branch carries (1 − h_i)(1 − b_i + r_i) = 1 − b_i to
the earlier residual. So any a.s. finite stopping time of the past (tokens, the walk, positions, EOS, engine timing,
other rows, and even the forward-drawn hits) yields an exact p sample; deciding at T = L is plain block
verification of the whole block, deciding at every piece end is today's rule. Only the decision's own terminal draw
and future tokens are off limits. (2) The implementation draws no hit before the decision (the rules do not read
them), then eta_T, then eta_{<T} top-down — the same joint law as drawing them forward. (3) Below the floor no
position can matter (h_f = 1), so the rejection scan covers (f, T) only.
Tests (`tests/unit/test_carried_walk.py`, CPU): exact enumeration of complete-row laws (V = 3, cap 4, pieces of 1 and
2, rules every / row / dip 0.05 / dip 0.3 / token-random / two hit-dependent rules: TV < 1e-12 vs p); control that
ignores a failed terminal draw half the time: TV > 1e-3; exact expected rejections: row < every, row ≤ dip ≤ every
(5 models); pathwise: `carried_piece` + `carried_scan` over random cuts give the one-block tau with the same draws
(60 cases); Monte Carlo of the implementation's whole continuation process (20k rows, chi-square z < 4 for row,
dip 0.1, every; non-vacuous: rejections, scans and undecided pieces all occur) and a control that forgets the walk
between pieces (z > 20). Driver: `test_carried_walk_driver.py`. GPU residual path: `test_carried_walk_gpu.py`.

**No exact incremental form of the one-block law with early final rejections** (sketch). Under the one-block rule
P(reject | x_{1..T}) = 1 − b_T depends on the whole draft, and P(tau ≤ g | F_g) = E[(1 − b_T) Π_{g<j<T}(1 − h_j) | F_g]
is an expectation over the q-continuation. A rule that finally rejects at g < T decides on an F_g-measurable event, so
its conditional rejection law given the full draft differs from the one-block law unless it could compute that
expectation (a sum over all continuations) and then continue on a q-law tilted by "tau > g", which a q stream does
not sample. What can be done exactly is the family above: one block at the end (fewest rejections, latest detection)
or decisions at stopping times (earlier detection, more rejections). Exact speculative forks would keep the one-block
law and buy latency with engine decode (a q stream from the tentative (tau, y), used only if the final decision lands
there): not built.

**MC [modeled, `research/carried_walk/cw_mc.py` → `cw_mc_focus.log`, 4000 rows at 32k / 1500 at 128k; ±3–5% on
rejections]** (Brownian walk, drift −v/2, the rejection at the block's last running max; per continued row; rounds =
learner piece calls; waste = tokens decoded past tau before the decision):

| schedule | rounds 32k / 128k | rejections, age 1 (v 1.1e-6) 32k / 128k | waste, age 1 | rejections, age 2 (v 3.3e-6) | waste, age 2 |
|---|---|---|---|---|---|
| every, growth 0.25 (today's law) | 24 / 30 | 0.56 / 1.20 | 1.0k / 8.5k | 0.97 / 2.13 | 1.7k / 14.7k |
| every, growth 1.0 (tip default) | 9 / 12 | 0.34 / 0.73 | 1.8k / 14.8k | 0.60 / 1.27 | 3.2k / 24.2k |
| carried, growth 0.25, dip 0.05 | 24 / 30 | 0.40 / 0.97 | 1.2k / 8.5k | 0.82 / 1.87 | 1.9k / 14.7k |
| carried, growth 0.25, dip 0.1 | 24 / 31 | 0.28 / 0.78 | 1.7k / 10.0k | 0.65 / 1.71 | 2.2k / 15.8k |
| carried, growth 0.25, dip 0.2 | 24 / 31 | 0.19 / 0.55 | 2.5k / 13.3k | 0.45 / 1.36 | 3.2k / 18.7k |
| carried, growth 0.25, row (one block) | 24 / 32 | 0.16 / 0.38 | 3.5k / 31.7k | 0.28 / 0.61 | 6.2k / 52.1k |

Reading: the decision schedule is a rejections-vs-detection-delay trade; carrying the walk moves the frontier. At
the same forward pieces, `dip 0.05` has 20–30% fewer rejections than deciding every piece at the same waste;
`dip 0.1`–`0.2` match or beat growth 1.0 on both rejections and waste (age 1: −17% / −26% rejections, −6% / −10%
waste) at 2.6x the learner calls (each a cached O(c) forward); one block halves rejections again and triples the
waste. Recommendation until v is measured at 16k–128k: forward growth 0.25 + `dip` 0.1 (0.2 when a rejection
costs more than ~5k decode tokens, e.g. a 128k restart prefill). Full grid (all rules, forward growth 0 / 0.125 /
0.25, Nano L8 v, 16k): `cw_mc_pieces.log` (400–1000 rows, noisier).

**Per-rank costs.**
- Driver: per undecided block, log-ratio and log q as fp32 arrays (8 B/token) plus the kept-token log-probs.
- Hidden rows of the q stream (object-store transport) are kept from the floor's piece to the decision (positions
  below are freed as the floor advances): ≤ (T − f + piece) × d × 4 B per row, Nano 10.75 KB/token (352 MB per 32k
  row worst case), 1T-class ~41 KB/token. `spec/cw_hidden_retained_bytes_max`. With the engine's recompute transport
  (14q, ts-engine-verify) nothing is retained; the scan re-prefills on the drafting lane.
- Learner: one pinned Mamba boundary per cached row (Nano 46.8 MiB; Ultra ~25 MB per GPU [H]) so the scan forwards
  from the floor's piece; scan forwards only on rejections (modeled scan tokens ≤ wasted decode above).
- Placement: `verify_engine_q_walk` / `verify_engine_q_scan` take the same inputs as `verify_engine_q`, so the engine
  path (ts-engine-verify, not merged) calls them unchanged; the walk state and decisions travel through the driver.
Metrics: `spec/cw_decisions`, `_whole`, `_rejects`, `_scans`, `_scan_tokens`, `_ratio_mismatch` (must be 0: the scan's
forward reproduces the pieces' log p bitwise), `_hidden_retained_bytes_max`.
**Chunked verification head (all engine-sourced verification, ts-carried-walk).** A verification forward
(verify0, continuation pieces, carried walk pieces and scans) used to materialize [rows x tokens x V] fp32 logits for
the whole forward batch: cw-x-4k OOMed allocating 68 GiB (≈ 34 x 4096 x 131k x 4 B) at step 3 [M,
runs_1node/cw-x-4k-row/run.log]. Now `verify_drafts_block_engine` runs its forwards under
`deferred_head.verification_logits()`: with the fp32 head at TP=1 the model returns the `DeferredHeadLogits` stand-in
(as training's deferred LM head), and each row's log-probs come from `engine_q.DeferredLogprobRows`, which forms
`log_softmax(fp32_gemm(x_rows, W)[:vocab_limit])` for at most 4096 rows at a time (drafted-token log-probs) or for the
rows a scan asks for. Same ops per row as the materialized path (row-invariant GEMM, row-wise log-softmax), so
bitwise the same values [T: test_carried_walk_gpu.py::test_deferred_logprob_rows_equal_materialized]. Per rank:
the head input (tokens x h x 4 B) plus one [4096, V] fp32 chunk (2 GiB at V 131k) instead of tokens x V x 4 B.
Not covered: TP > 1 (vocab-parallel `GatheredLogprobRows` still keeps [n, V/TP] logits per row), heads outside the
fp16-GEMM recipe (logits materialized as before; `[verify head] chunked=False` in the log). Engine placement (ts-engine-verify, not merged)
already chunks p rows (`HeadLogprobRows`, 512 rows).

**Debug** `THUNDERSYNC_DEBUG_CW_FORCE_REJECT=N`: the first N decisions of each plan fail their terminal test (only when
b < 1) so short exactness runs exercise the rejection scan; changes the law (not an exactness test of the rule),
leaves every bitwise check meaningful. `spec/cw_forced`.

Requirements (`validate_carried_walk`): sticky routing, verify_source=engine, verify_on=learner on this branch, no
banded IS. Not built: carrying the walk through verify0 (it decides; ⊥-BV pools need its outcomes), speculative forks.

## 14u. EP-invariant MoE combine (`policy.megatron_cfg.moe_ep_invariant_combine`, opt-in)

*Branch ts-ep-invariant (../RL-epinv). Code: `nemo_rl/models/megatron/ep_invariant_combine.py`. Notes:
thundersync_work/research/agent_ep_invariant.md.*

**Problem [M].** The batch-invariant combine sums a token's expert rows per EP rank in fp64 and rounds each rank
partial to fp32. The partials are then added in rank order. That makes the rounding points depend on EP.
- Megatron's BI training unpermute mirrors the same tree, so the learner and the engine agree only at equal EP.
- One Nano layer (EP emulated): EP2 or EP4 vs EP1 changes 98-100% of tokens (max 2.4e-7 per element). Engine EPn
  equals learner EPn exactly.
- Nothing else depends on EP: the expert rows are row-local GEMMs, and routing, padding, permutation order and
  shared experts are EP-free.

**Design.**
- Each expert row value v is fp32 and, under the fp16 GEMM forward, a multiple of 2^-48. It maps to two fp64
  integer limbs: `hi = floor(v)` and `lo = (v - hi) * 2^48`.
- A token's limb sums are integers below 2^53 (checked at setup from top-k, the fc2 K and the prescale), so every
  addition is exact. Any grouping or order (per local expert, per rank, any collective) gives the same sums.
- `finalize` returns the correctly rounded fp32 of the exact sum (TwoSum round-to-odd, then fp32).
- Engine (AllGather-V/RS-V):
  - The unpermute kernel writes the rank's [T, 2H] fp64 limb partial into the NVLS RSV symmetric memory, which
    is sized 4x at allocation.
  - `limbs_reduce_scatter_v` sums the peers' limbs and finalizes. The NCCL dispatcher reduce-scatters fp64
    limbs instead.
  - At EP 1 the kernel finalizes directly. All of this is CUDA-graph safe.
- Learner (AlltoAll): the combine already returns each token's k rows to its home rank, so the bytes do not
  change. The BI unpermute sums the k rows through the limbs, with no rank grouping. The backward is unchanged
  and deterministic.

**Cost.**
- Engine combine bytes and RSV memory go up 4x (16 B per element instead of 4 B). Nano at max_tokens 16384: 2.8
  GB RSV at EP4 [D].
- Returning the k rows instead would cost 4k B (Nano 24, Super 88, 1T-class 32), so limbs are the cheaper exact
  form for k > 4.
- Measured on one GB200 node:

  | measurement | old | new |
  |---|---|---|
  | full Nano engine decode, EP4, CUDA graphs, ms/step (N = 1 / 8 / 128 per rank) | 7.52 / 8.48 / 8.46 | 7.94 / 8.89 / 8.88 (+5%) |
  | combine (unpermute + RS) per layer, EP4, 128 tokens/rank | 27 us | 56 us |
  | combine per layer, EP4, 4096 tokens/rank (prefill) | 0.44 ms | 1.14 ms (x23 layers: +16 ms/step) |
  | learner unpermute fwd+bwd, 1k-16k tokens | 0.64-0.67 ms | 0.28-0.68 ms (no regression) |

- The limb reduce-scatter (fp64 loads) is the cost; it is not tuned yet.
- At Ultra/1T EP (32-64), the per-rank cost of the AllGather design grows with EP whatever the combine. There the
  engine needs an AlltoAll dispatcher: the combine returns k rows, and the home rank runs the learner's
  `unpermute_rows`. That dispatcher is not built yet.

**Validation [M].**
- Unit: 68 GPU tests pass.
- NVLS EP2 and EP4 real reduce-scatter: limb results are bitwise equal to EP1, while the old ordered and multimem
  results differ from EP1.
- Nano L8 engine (livegrid, fp16, CUDA graphs on and off): greedy decode dumps at EP2 and EP4 are bitwise EP1 on
  every rank with the new combine. The old combine differs in 46 and 92 log-prob sequences (max 0.28). New EP1
  equals old EP1 on these dumps.
- End to end, Nano L8, learner EP2, engine lanes EP1, `THUNDERSYNC_DEBUG_FRESH_MOD=3` (16-48 forced fresh,
  engine-sampled rows per step), real gradients:
  - epinv-lanes-ep: 4/4 steps mismatch 0, engine-q exact.
  - epinv-lanes-ep-off (control, old combine): fails every step (max |dlogp| 0.66 / 0.30 / 0.79 / 0.74).
  - epinv-ep2-ctl (equal EP, new combine): exact.
- The validators (`zero_train_gen_mismatch`, `validate_engine_verify_source`) accept different learner and
  engine EP when the combine is on.

**Default.** Opt-in: +5% engine decode at EP4 and +16 ms per 4k-token-per-rank prefill step. Turn it on for
layouts whose engine EP differs from the learner's (lanes, Ultra/1T).

## 14v. Drafting-window pacing (`thundersync.draft_window_pace`, opt-in, 2026-10-15)

**Problem (measured).** With sticky streams and cohort lanes, cohort k+1 drafts from the top of step k until the cut at
the end of step k's training, so its window is step k's length. When all drafts run at once (192 drafts, max_requests
>= 192) they progress together and finish at ~7.1-7.4 s on Nano L8 1k. A fully drafted cohort gives a fast step
(6.2-6.7 s window), which is just short of full: the next cohort is cut at ~87-89% (169.8k/174.8k of 196.6k tokens),
pays a whole continuation phase and is slow (10.2-10.3 s), which gives a long window and a full cohort again
(runs_1node/ev3-p-l2-b2-s05 steps 3-7: 6.16/10.33/6.72/10.20 s windows; ev2-p-l2-b2 and adm-p-l2-grp-m192 the same
1/0/1/0 pattern). Mean step ~8.8 s although a fully drafted step at the full window would be ~7.4 s.

**Damper.** `window_pace.WindowPacer` holds the cut until `t_start + W (1 + margin)`, `W` = fitted time for every draft to
reach the cap, and releases it as soon as no draft is running. It does not wait when that deadline is more than
`max_wait_frac * W` away (a partial wait cannot avoid the continuation phase; a slow step's window is already long).
`W` is fitted from earlier cuts only: a cut with running drafts extrapolates `elapsed + (cap - min len) * elapsed / max
len` (move up at once, late is expensive); a cut with all drafts done can only lower it (blend). One pacer per run,
shared by the lanes (W is a property of the drafter). `draft_window_pace_init_s` is an optional prior.

**Exactness.** The deadline is a function of earlier steps' timings and draft lengths. The early release depends on
the current cohort through one count and only when it is 0: every draft is then finished, finished drafts do not
change, so the handed-out cohort equals that of a cut at the past-only deadline. Any cut is a valid draft prefix set
for verification in any case; pacing only moves tokens between verify0 and continuation rounds.

**Scale.** Driver-side scalars; during a wait one O(1)-reply call per 20 ms poll to the drafting engine's rank 0
(`count_unfinished_drafts`, O(n drafts) host work), no per-rank state, independent of TP/PP/EP and model size.

**Validation** (queued, research/window_pace/val.sh, user submits): pace-x-l2 (exact, CHECKS=1, prior forces holds),
pace-p1 (same commit: lanes m192 control vs paced vs async1, 8 steps), pace-p2 (default concurrency control vs paced).

## 14w. Dealing dispatch rows over every learner DP rank (`thundersync.deal_rows`, default on)

*Branch ts-deal-rows (../RL-deal, off ts-nano-perf 9cf376a0). Code: `streaming.py` (`StreamPlanner(deal_rows=)`,
`num_units`, `cut_units`, `deal_chunks`, `microbatch_counts`), `grpo_loop.py` (`canonical_groups_per_block`, metrics).
Notes: thundersync_work/research/agent_deal_rows.md; analysis: research/learner_offload_design.md §1.3, §5.*

**Problem [M + code].** Canonical dispatch sends blocks of `canonical_block_groups` (4) groups. The planner cut the
dispatch's rows (length-sorted, zero / nonzero advantage apart) into chunks of `max_chunk_trajectories` (8) rows and
placed whole chunks on DP ranks by tokens. Under EP every rank must run the same number of forwards, so
`_pad_to_equal_microbatches` padded every rank to the busiest one with all-masked fillers.
- Full Nano (learner DP 8, MBS 1): a 32-row dispatch = 4 ranks × 8 real + 4 ranks × 8 filler micro-batches. Per step
  17 × 8 = 136 micro-batches per rank against 64 that carry rows (53% filler, ≈ 100 s of a ~200 s learner).
- async1's 8-row leftover dispatch: 1 rank real, 7 ranks filler (13.9 s).
- Every dispatch took ~11–13 s flat in rows and tokens (`timeline/dispatches`, x2-lanes and ab2 runs).
- Filler fraction ≈ 1 − rows / (DP · chunk rows) per dispatch: it grows with DP. At DP 2 (Nano L8) it is 0, which is
  why the L8 audits never saw it.

**Design.** `next_dispatch` with `deal_rows`:
1. The closed-group rows of the dispatch, sorted by tokens (stable on the canonical order), form one class (two —
   nonzero / zero advantage — only when the worker skips zero-advantage rows, i.e. without EP).
2. `cut_units` cuts each class into `num_units = min(rows, DP · ceil(ceil(rows / u) / DP))` consecutive units,
   u = min(MBS, `max_chunk_trajectories`): one micro-batch per unit. Rounding the unit count up to a multiple of DP
   never raises the per-rank micro-batch count; the units just hold fewer rows. A unit is length-contiguous, so it
   pads only to its own longest row (the worker pads a chunk's micro-batches to the chunk's longest row; old 8-row
   chunks padded every row to the longest of 8).
3. `deal_chunks` places fused (pinned) chunks first, then the units longest first on the least-loaded rank, load =
   (live micro-batches, live tokens, tokens), lowest rank on ties (a heap: O(log DP) per unit). With one-micro-batch
   units this is a round-robin by count, longest against shortest within each round.
4. `_pad_to_equal_microbatches` is unchanged; it now adds a filler only when rows cannot fill whole rounds of DP
   units: never when rows is a multiple of DP or rows ≥ DP · ceil(ceil(rows / u) / DP), fewer than DP otherwise.
- Result per dispatch: every rank runs ceil(ceil(rows / u) / DP) micro-batches. Full Nano 32-row block: 4 per rank,
  0 filler (was 8 per rank, 32 filler); async1 8-row leftover: 1 per rank; async1 504 rows: 63 per rank; sync 512:
  64 per rank.
- `canonical_groups_per_block` (deal_rows on): a block has at least ceil(DP / G) groups, so its rows reach every
  rank at any DP, and under PP at least ceil(min_pipeline_microbatches · DP · MBS / G) groups, so each rank's one
  pipeline call per dispatch has 2·PP micro-batches (the canonical order never holds rows back the way the
  arrival-order planner does). Both are 0 changes at Nano (DP 8 ≤ 4 · 8 rows, PP 1).
- `deal_rows: false` restores the whole-chunk layout (`balance_chunks`) for A/Bs. VIT spread groups
  (`vit.plan_group_layout`, opt-in) keep their own layout.

**Exactness.** Only which rank and which micro-batch backpropagate a row change, and how far a row is padded
(masked positions). That is fp32 summation order of the gradient (E2 in learner_offload_design §2), the same class as
an MBS change; fillers add exactly ±0. The sampling law never touches training. The layout is a function of the
dispatch's rows and the config (sort by tokens, stable on the canonical order; ties to the lowest rank), so canonical
dispatches are still bitwise independent of arrival order (test: `test_canonical_dispatch_independent_of_arrival_order`)
and every rank still receives the same numbered `ordered_call` sequence: the driver plans once, ranks get their shard.
Global loss normalization (valid sequences / tokens summed over DP at `finish`) does not depend on placement.

**Metrics.** Per step: `learner/real_microbatches`, `learner/filler_microbatches` (summed over ranks),
`learner/microbatches_per_rank` (what one rank runs: sum over dispatches of the per-rank maximum),
`learner/filler_frac`, `learner/filler_heavy_dispatches` (filler > 25% of a dispatch's forwards; the first one per
step is also printed). `timeline/dispatches` rows gain (real micro-batches, filler micro-batches, micro-batches per
rank, real tokens on the busiest rank) after the old five fields, so per-dispatch time can be regressed on real work.
Opt-in attribution: `run_ab_arm.sh LEARNER_PROF=1` (per-rank per-call seconds and shapes) and `LEARNER_TORCHPROF=N`
(one torch-profiler table) reach multi-node learner workers through `policy.megatron_cfg.env_vars`.

**Scale (per-rank costs).** Driver: O(R log R) sort + O(U log DP) placement per dispatch (R rows, U units ≤ R).
Wire: the same rows; one item per unit instead of per 8 rows (MBS 1), so more, smaller items per rank. Worker: one
`_prepare_train_piece` per item (host-side counts and iterator, small next to the ~1.2 s full-Nano 8k micro-batch;
measured per call by LEARNER_PROF, not yet per item). Memory: a unit is ≤ one micro-batch, the activation unit
already; less sequence padding per micro-batch. Nothing new per parameter.

**1T desk check** (TP8 PP8 EP64 on 512 GPUs: learner DP 8, G 16, MBS 1, `canonical_block_groups` 4).
- DP 8, 64-row blocks: 8 units per rank, 0 filler with deal_rows. The old layout had no filler only when both
  advantage classes chunked into ≤ 8 chunks in total; any split such as 37 nonzero + 27 zero (5 + 4 chunks) put two
  8-row chunks on one rank and padded the other seven to 16 (44% filler).
- PP 8: min_pipeline_microbatches = 16 → `canonical_groups_per_block` raises blocks to 8 groups (128 rows, 16
  micro-batches per rank per call): bubble 7 / 23 = 30% per call instead of 7 / 15 = 47% at 8 per rank. Half as
  many dispatches per step, so the streaming overlap is coarser. Untested on GPU (PP>1
  streaming under the canonical order is itself untested end to end).
- Large DP (e.g. DP 64 or 128 from smaller TP·PP): blocks grow to ceil(DP / G) groups; class-boundary blocks (the
  last block of the ready / VIT / tail / fresh classes) can still hold fewer rows than DP and pad fewer than DP
  fillers each, ≤ 4 such dispatches per step.
- EP all-to-all: equal per-rank forward counts are kept by `_pad_to_equal_microbatches` exactly as before.
- Not covered here: per-micro-batch host cost (callcost, d2), the PP bubble of the arrival-order planner, VIT spread.

**Validation.**
- CPU [M]: `tests/unit/test_deal_rows.py` (DP 8 EP MBS 1 G 8 blocks of 1/2/4: ceil(rows/8) per rank, 0 filler; old
  layout ≥ half filler; DP 2 (L8) 16 per rank both layouts; one dispatch for rows ∈ {1, 3, 8, 12, 20, 32, 63, 504} × DP ∈ {2, 8, 12, 64}
  × MBS ∈ {1, 2, 4, 8}: minimal per-rank count, fillers < DP; arrival-order invariance; gradient sum exact incl.
  skip-zero; `final_microbatches` mirrors dispatch; 1T / large-DP / PP block sizes). Full unit suite: 952 passed, 1
  failed = the base failure (`test_group_mode_runs_one_call_with_all_microbatches_in_order`, fails on 9cf376a0 too).
- Nano L8 regression smoke (DP 2, no fillers before either): queue dr-x, dr-xl @ e141d03e (the placement code of the
  tip; the later heap / PP changes give identical layouts at PP 1). Pending.
- Full Nano (multi-node, `thundersync_work/sbatch/go_deal.sh`, frozen ../RL-deal-scale): EP4 × DP 2 lanes 2k
  exactness (CHECKS=1, 4 steps) and the same-tree 8k perf trio (ts_por lanes, async1, sync; CHECKS=0, 6 steps, real
  reward). Expected [D, learner_offload_design §6]: per 32-row dispatch ~4 micro-batches per rank and ~5.5–6.5 s
  (was 11.3–13.2), ts_por training ≈ 96–110 s (was 193–209), async1 train ≈ 105–107 s (was 119–121). Pending.

## 14x. Zero-advantage skipping under expert parallelism (with `deal_rows`; 2026-10-15)

**Problem [M].** `loss_fn.skip_zero_advantage_rows` (ab_base: true) was forced off under EP: the worker
dropped rows by a per-rank data test, so ranks could run different forward counts and hang the EP
all-to-all. Every arm therefore trained all 512 rows per step although only 176–328 (34–64%) have a
nonzero advantage (`train/nonzero_adv_rows`, runs_ab/*-ab2; GRPO zero-advantage groups are whole groups).
At full Nano 8k that is ~1.65 s per skipped micro-batch-slot, i.e. 35–60 s of exactly-zero gradient work
per step per arm (sync training 105–108 s).

**Design.** The planner decides, not the worker:
- `skip_zero_advantage_chunks` is on under EP when `deal_rows` is on. Rows are dealt per class
  (nonzero / zero advantage, `_classes`), so a zero chunk never holds a live row.
- `StreamPlanner.count_only(c)`: final, unfused, non-padding chunk whose every advantage is 0.
  `_pad_to_equal_microbatches` equalizes the **live** forwards per rank (count-only chunks count 0) and
  gives every rank at least one forward per dispatch (metric schema; an all-zero dispatch costs one
  filler per rank).
- `_build_dispatch` ships `count_only` per item (EP only; without EP the worker keeps skipping by its own
  test as before). The worker (`worker.count_only_chunk`) adds the chunk's `sample_mask` / token sums to
  the step's normalization exactly as `_prepare_train_piece` would and runs no forward; it raises if any
  valid token carries a nonzero advantage (never drops gradient silently).

**Exactness.** The skipped rows' gradient is exactly zero: the worker already validates that every gradient
term is advantage-weighted (`_require_advantage_weighted_gradients`: no MoE aux / z-loss, no bias updates,
no capacity dropping, no MTP), and that validation runs whenever the loss asks for skipping, EP or not.
Their tokens stay in the token-level normalizer (float64 integer sums: order-free). What changes is the
placement of the remaining rows (fp32 summation order, as `deal_rows`). The sampling law is untouched.

**Per-rank cost / scale.** Planner O(rows / DP) per dispatch as before; the worker reads two masks per
skipped chunk. Saved per rank: (zero rows / DP) micro-batches per step at any DP / EP / PP (with PP the live
count per rank sets the pipeline call). 1T (DP 8, G 16): same rule, no new communication.

**Expected [D].** Full Nano 8k, 512 rows, ~50% zero: training ≈ 64 → ≈ 33 live micro-batches per rank,
sync ≈ 106 → ≈ 55–60 s, ThunderSync (after deal_rows ≈ 105 s) → ≈ 55–60 s, async1 → generation-bound
(~110 s). **Validation:** CPU tests `test_skip_zero_ep.py`; GPU: `sbatch/queue_1node.tsv` sz-* (L8, EP 2:
exactness + skipped counts), multi-node `sbatch/go_bughunt.sh`.
## 14y. Full-Nano bug hunt fixes (branch ts-bughunt, 2026-10-15; thundersync_work/research/fullnano_bug_hunt.md)

- **async1 split dispatch [M → fixed].** With `rollout_lag=1` every row is in hand at the top of the step,
  but the arrival loop yielded after each completed group, so the learner sent group 0 alone: async1-*-ab2
  ran 8 rows (10.6–14.2 s, 8 micro-batches per rank under EP) then 504 rows, 72 micro-batches per rank vs
  sync's 64 for the same rows (worker `[step] chunks=9 microbatches=72` vs `chunks=8 microbatches=64`).
  The yield is now skipped for stale rollouts: one dispatch per step, like sync. Not an exactness arm.
- **Weight sentinel cost [M → fixed].** `tensor_digest` synchronized the host once per tensor; ~6k tensors
  per full-Nano rank made each check ~2 s (`sentinel/time_s` 1.9–2.0 s, inside `time/refit` 2.5–2.6 s on
  async1 / sync's critical path). `digest_many` packs small tensors (int64 segment sums: the same sums mod
  2**64), chunks large ones as before and reads all sums with one transfer: bitwise-equal digests
  (CPU test), O(params / rank), one sync.
- **verify-load metric [code → fixed].** `spec/verify_load_max_over_mean` was overwritten by every
  verification call, so each step reported its last call (one piece on one rank) = DP = 8.0 in every
  full-Nano history. Now ranks·Σbusiest/Σtotal over the step, plus `verify_load_first_max_over_mean`
  (verify0).

## 14z. Risk-first streaming verify0 (`thundersync.verify0_stream`, opt-in; branch ts-verify0-stream, 2026-10-15)

*Notes, enumeration and replay: `thundersync_work/research/agent_verify0_stream.md`, `research/v0s/`.
Base: ts-bughunt-integ 5f835b57 (deal_rows 14w + zero-advantage skip 14x + 14y).*

**Why.** verify0 (block verification of cohort k's drafts at θ_k) runs serially in front of every sticky
step: full Nano 8k lanes (ab2) 40.6–45.4 s for 768 drafts / 3.4–3.6M drafted tokens, and every plan is
published at its end (`tail/rows_done_q50_s` = `tail/verify0_s`). So the slowest row's tail starts at
~41 s, and training of rows that verify0 accepted whole waits for the last draft's decision. With the
filler and zero-advantage fixes the chain (verify0 + tail decode) is the binding term (fullnano_bug_hunt §0 #4).

**What it does.**
1. `prepare_stream` (step top, synchronous): filters and pools the drafts exactly as `verify` does,
   then `schedule_chunks`: calls of **whole prompts**, sorted by (−risk score, drafting index), cut at
   ≥ `verify0_stream_first_chunk_tokens` (262144) forward tokens for the first call and
   ≥ `verify0_stream_chunk_tokens` (524288) for each later one. Rollouts of a drafted prompt wait for its
   decision (`_plan_cv`, as chunked non-sticky verification already did); prompts never drafted decode at
   θ_k after the tails, as before.
2. `run_stream` (inside the step, the pre-rollout task): a pipeline of `verify0_stream_window` (2)
   ordered learner calls. Call c + window is sent once call c's results are applied; it carries the
   ⊥-BV follow-up waves of the pools decided so far (`_NullPoolRun`, shared with `_verify_null`) and the
   next scheduled prompts, so follow-up waves never cost an extra call. Results are applied in call
   order. A prompt whose pool is decided is **published at once**: plans in drafting order, its tails
   (`plan.tail`) handed to the already-running continuation rounds (`_continue_sticky(feed=TailFeed)`),
   its other draft streams stopped, its rollouts woken.
3. **Training overlap.** `StreamOrder` grows the canonical order with the decisions: the ready groups
   of the prompts decided after call 1 (group index order), after call 2, ..., then (once every prompt
   is decided) the groups with tails, then the groups with a rollout never drafted; blocks of
   `canonical_block_groups` within each segment (or `canonical_block_tokens` segments). The planner
   follows the order as it grows (`StreamPlanner.begin_order / extend_order / close_order`), so ready
   groups train between verification calls, ordered learner calls as before.
4. **Risk = past only.** `Verify0Risk` (one object shared by the lanes): per draft-length bucket
   (<256, <1k, <2k, <4k, <8k, ≥8k), the past steps' whole-acceptance rate (shrunk to a constant-hazard
   prior e^{−2e−5·L}) times the past mean remainder after a verify0 rejection (final length − accepted,
   shrunk to L). A prompt's score is the sum over its drafts (at 8k most prompts have a capped draft,
   so a max ties everywhere). Updated once a step's tails are done: step k's schedule reads steps < k.
   `verify0_stream_order: drafting` keeps drafting order (the control).

**Exactness (argument).** Each pool's decision is a function of (its drafts, its rows' verification
draws, the count process); the draws of a call are a fresh seed per call (`next(_verify_seeds)`), iid
and independent of every draft. Streaming changes only (a) which call a row's draws come from and when,
(b) when a decided plan is published / its tail starts / its group trains. Rules that keep both
law-neutral, all enforced in code:
- the schedule (call membership and order) is a function of counts (draft lengths, drafts per prompt,
  tokens) and earlier steps only — never this step's tokens, log-ratios or decisions;
- a prompt's pool is never split across calls (its follow-up waves ride in later calls but its count
  process runs in drafting order inside the pool); drafts inside a pool keep drafting order;
- every drafted row is decided whatever the order: no deadline, budget or other order-dependent switch
  to another sampler (prompt drop, which selects prompts after all pools, is rejected by the validator);
- tails start at the decision but are the same exogenous-piece continuation on D_k (§14c / 14k), and
  their law does not depend on start time;
- the canonical training order is a function of the schedule and the decisions; it moves fp32 summation
  order only, never which rows train.
Within these rules ordering by a content-derived count (draft length ends at EOS) is still exact: the
order permutes which fresh iid draws a pool gets, and nothing switches on it.

**Enumeration [T]** (`research/v0s/v0s_enum.py` → `v0s_enum.log`, 4 seeds; two prompts, N = 3/2 drafts,
G = 2/1 slots, ⊥-BV M ∈ {1, 1.5}, variable-length sequences so length is content; the STREAM arms run the
production `run_stream` / `schedule_chunks` / `_NullPoolRun` on every content combination):
| scheme | TV to p×p×p | gradient bias (|∇| 0.02–0.20) | per-token mismatch / engine-q |
|---|---|---|---|
| one-shot verify0 | ≤ 1.8e-15 | ≤ 2e-16 | 0 |
| streamed, risk-first by length, window 1 / 2 | ≤ 1.8e-15 (identical decisions) | ≤ 2e-16 | 0 |
| content order **inside a pool** (shortest first) | 0.090–0.139 | up to 0.091 | 0 |
| order-dependent switch (verify the first prompt only, by length) | 0.005–0.096 | up to 0.040 | 0 |
| same switch, past-only order | ≤ 2.8e-15 | ≤ 1e-16 | 0 |
The biased variants pass every in-loop check (each output carries p's own log-probs), so the
enumeration, not an E2E run, is the guard. Unit tests [T, `tests/unit/test_verify0_stream.py`, 33]:
streamed ⊥-BV decisions equal `_verify_null`'s for random grids, schedules and windows 1–3; call
composition independent of result timing; growing order == final order dispatches under any arrival
order; schedule reads counts only; driver end to end on the fake engine (streamed plans == `verify_sticky`'s).
CPU suite: 1071 pass / 0 fail (base 5f835b57: 1040 / 0).

**Costs (full Nano 8k, EP4 learner DP 8, lanes).**
- Per rank: no new GPU or host state. Driver: O(drafts) pools / plans (as today), O(groups) order,
  6-bucket risk table. Undecided prompts' draft streams keep decoding until their decision (today: until
  verify0 ends; with streaming the last decisions can come later, behind interleaved training).
  Prefix cache (14p): the per-step budget follows the tails admitted so far (same final "auto" size).
- Fixed per verify call [D, unmeasured]: at 8k a verify forward holds one row (padded rows × max length
  ≤ 16384), ~0.47 s per row-forward (ab2: 41.5 s / ~88 forwards per rank); a call waits for its busiest
  rank (≤ 1 row above the mean) plus a Ray / ordered-call round trip: ~0.2–0.5 s per call, ~8 calls per
  step (3.4M tokens / 524k) → +2–4 s learner time per step. (Training slots cost ~1.4 s each; verify
  forwards are cheaper per call.) `v0s/verify_service_s` vs the control's `time/verify` measures it.
- Ultra: calls stay EP-lockstep collectives of whole rows; their number is tokens / chunk tokens, the
  schedule is O(prompts log prompts) on the driver; nothing per rank grows.

**Expected effect [D, `research/v0s/replay_schedule.py` on ab2's measured tail rows; optimistic: training
interleave and engine contention not modelled].** Chain end (latest row done) vs measured, 0.4 s/call:
risk-first −14.4 s mean (0 … −36 s), drafting order −14.0 s, an oracle that knows the slowest tails −39.8 s.
So per-call publication brings ~half of verify0 off the chain on average; the past-only length/remainder
predictor is **no better than drafting order** on this run (at 8k most prompts look alike by length; the
strongest known predictors — drafter entropy, sibling drafts, ICC .75 — read this step's content and are
excluded by the rule above).

**Not built.** Prompt drop, verify-in-train, multi-turn, anchor horizon, tail-excluded baseline (validator).
Risk state is not checkpointed (a resumed run schedules differently; law unchanged). verify0 still starts
after the drafter stage at the step top (lanes: θ̂_{k+1} is staged through the learner, ~3 s). ⊥-BV
calibration's verify cost is the calls' service time, which includes interleaved training dispatches.

**Validation queued.** L8 single node (queue_1node, v0s-*): exactness one engine / lanes / ts_nonull /
8k with CHECKS=1, parity reward, forced fresh rows, zero-advantage groups; order replay. Full Nano 8k pair
(control vs streaming, same frozen tree ../RL-v0s-scale): `bash thundersync_work/sbatch/go_v0s.sh`.

**Lanes fix (branch ts-v0s-lanes-fix off ts-integ3, 2026-10-15).** v0s-xlanes (6fc1ae9c) failed with max|dlogp|
1.66 / 0.76 / 0.60 and exact_frac .875 / .688 / .625 on steps 1-3 [M runs_1node/v0s-xlanes]. Those fractions are
exactly 1 - forced-fresh rows / 128 (16 / 40 / 48 rows, `debug/forced_fresh_rows`), and `lanes/fresh_refit` was 0
on every one of those steps (bh-xlanes, same layout without streaming: 1, EXACT). Cause: with lanes the cohort
lane takes θ_k only if some row decodes fresh; the streamed path decided that from `StreamOrder.fresh` (groups
with fewer plans than rows), while `_one` also decodes fresh every row of a `THUNDERSYNC_DEBUG_FRESH_MOD` prompt
whatever its plans. Those prompts were drafted and fully planned, so `fresh` was empty, the lane kept θ̂_k (the
drafting weights) and the forced-fresh rows were sampled from θ̂_k and logged with its log-probs. Drafted rows
were exact throughout (route / engine-q checks exact). The one-engine path refits θ_k unconditionally, hence
v0s-x passed. Fix: `StreamOrder(fresh_keys=...)` (grpo_loop `stream_fresh_keys`, the same rule `_one` applies):
those groups are never ready at a decision and are listed in `fresh` (trained after the tail groups), so the
cohort lane refits. Guard: the driver's refit sets `spec.fresh_at_current` (lanes: `lane_fresh`; one engine:
`not draft_now`), and a rollout reaching the θ_k decode with it false raises instead of sampling from the drafting
weights. Production runs (no FRESH_MOD) had an empty `fresh_keys`, so their lanes path is unchanged by the fix;
the guard now checks the prediction on every step. The validator accepts cohort_lanes > 1 again. Cost: none per
rank (one set lookup per prompt on the driver). Tests: test_verify0_stream.py (fresh keys never ready; every
group `_one` decodes at θ_k is in `fresh`, seeds 0-3; guard raises). Rerun: queue_1node `v0sl-xlanes`
(+ `v0sl-x`, one engine, whose forced-fresh groups now train last).

## 14aa. Tail speed: verification yields into training, graphed restart prefill, lookup on real tail rows

*Branch ts-tail-speed (../RL-tail). Code: `thundersync_rl/verify_yield.py`, `worker.py` (`_run_train_pieces`,
`ordered_call`), `speculative.py` (`_submit_verify_block`, `_verify_block_async`), `tail_trace.py` (dump). Notes,
numbers and scripts: thundersync_work/research/agent_tail_speed.md, research/tail_speed/.*

**Where the post-refit tail goes [M, TailTrace timelines; tail_crit.py, round_wait.py].** One rejected row usually
sets the step. Its critical path is: rejection detected → restart (re-prefill of prompt + kept prefix) → decode a
piece → wait for the piece's verification → ... → decode the remainder → verify the last piece.
- Nano L8 1k, lanes (cut-p-l2*, ev3-*): tail after verify0 1.0–3.0 s per step; the slowest row has 1–2 restarts and
  its last segment decodes at ~3 ms/token, plus 0.1–1.3 s waiting for its last piece's verification. Piece-round
  latency p50 0.2–0.9 s, of which 20–75% is time behind a running training dispatch (~1.0–1.6 s each).
- Full Nano 8k EP8 (ts_nonull acc-momentum, 16 GPUs): tail 150–195 s of a 290–370 s step. The slowest row has 3–5
  rejections on its path; each continuation rejection is found 20–35 s after its token was decoded (2–4k wasted
  tokens at ~10 ms/token). Piece-round latency p50 ~21 s, ~half of it behind a training dispatch (~15 s each,
  17 dispatches per step); the rest is the verification forwards themselves queued behind each other (learner
  verify busy 45–95 s per step). The remainder decode itself runs at ~10 ms/token.
- Restart prefill: ≤ 5 restarts on the slowest row → ≤ 0.15 s (L8 eager, 25–31 ms each) and ≤ 0.8 s (full Nano
  eager, 125–170 ms each) per step; graphed prefill steps (`cuda_graph_all_prefills`, measured by the engine-verify
  agent: L8 27 → 7.6 ms, join +32 → +5.5 ms; full Nano 120 → 35 ms, join +148 → +32 ms at 1k) cut that ~4x.

**Verification yields into training (`thundersync.verify_yield`, opt-in).** While streaming over the tails every
learner call is numbered and runs in that order on every rank (14k ordered calls), so a piece verification sent
while a training dispatch runs waits for the whole dispatch. With `verify_yield` the driver posts each
continuation verification (pieces, carried-walk scans) to a `VerifyMailbox` actor and sends a small ordered call
(`posted_call`) to hold its place. A rank running a training dispatch asks the mailbox at every micro-batch
boundary whether the next ordered call is a posted verification; if so it runs it there (PP = 1: the boundary is
Megatron taking micro-batch i, after micro-batch i−1's backward; hook: a wrapper over the call's micro-batch
iterator) and posts the result. The ordered call then finds it done. The driver reads results from the mailbox.
- Agreement: ranks whose forwards share collectives decide together. Domain = the EP group (EP > 1; its ranks
  train in lockstep, equal micro-batch counts) or one DP replica (EP = 1; its TP/CP ranks hold the same rows).
  The mailbox makes one decision per (domain, dispatch, boundary, slot), atomically; every rank of the domain
  reads it. Ranks of different domains share no collective in a verification.
- Exactness: the dispatch runs the same micro-batches in the same order with the same loss scaling and the same
  gradient captures; the inlined call is a no-grad forward, and router-replay targets, RNG states and module modes
  are restored after it. Verification results do not depend on when they run (θ_k fixed within the step, explicit
  seeds, batch-invariant kernels). So gradients, decisions and tokens equal the ordered path's; only timing
  changes. Not inlined: fused-verify-train calls (their substitute forward), verify-in-train calls.
- Per-rank cost: one mailbox round trip per micro-batch boundary (~1 ms against 0.3–4 s micro-batches), the
  posted payload fetched once per rank (what the ordered call carried before), one result put. The mailbox holds
  the payload references of verifications in flight and the decisions of the running dispatch per domain
  (dropped when the domain moves on). One actor per learner; at thousands of ranks the boundary queries are one
  small RPC per rank per micro-batch.
- Wait bound: a verification waits at most one micro-batch (plus the posted verifications ahead of it) instead of
  a dispatch: Nano L8 1k (2 micro-batches per rank per dispatch) ~1.1 s → ≤ ~0.55 s; full Nano 8k EP8 (8
  micro-batches per rank per dispatch at mbs 1, deal-rows note DR) 11–15 s → ≤ ~1.5–2 s.
- Not built: PP > 1 (a 1F1B schedule has no boundary common to all stages; posted calls then run in order as
  before). Untested: TP > 1, CP > 1.

**Graphed restart prefill.** `mcore_generation_config.cuda_graph_all_prefills` (engine-verify 523cc454,
cherry-picked) graphs prefill and mixed steps up to `max_tokens`; every restart (verify0 rejection, piece
rejection, stale stream) pays the graphed step instead of the eager floor. Bitwise: the engine-verify agent's
graphed-prefill arms (evl-x-g, ev3-ex-l2-s05) are EXACT with engine-q checks; on the learner-verify path the
engine's q is checked per token by THUNDERSYNC_ENGINE_Q_CHECK (ty-x-* arms).

**Lookup speculation on real tail rows.** `THUNDERSYNC_TAIL_DUMP=<dir>` (debug) writes each step's drafted rows
(prompt, draft, final rollout, rejection positions and dead tokens). `research/tail_speed/lookup_replay.py` replays
the remainder lookup proposer (own context + dead tokens + group drafts; tail_remainder_spec `sim.py`) on the
remainders that were actually sampled and kept: with sample-and-match the forwards a lookup engine would need
follow exactly from those tokens. In-loop on Nano L8/L16 the remainder lookup engine path gave 1.00 tokens per
forward (remspec-nano4c: truncated models do not repeat text); real full-Nano text gave 1.74 offline
(agent_tail_remainder_spec.md). Results on dumped tail rows: agent_tail_speed.md.


## 14ab. Integration round 3: ts-integ3 (2026-10-15)

*Branch ts-integ3 (../RL-integ3) off ts-bughunt-integ 5f835b57 (= ts-nano-perf 9cf376a0 + deal_rows 14w +
zero-advantage skip 14x + bughunt 14y). Notes, decisions and validation: thundersync_work/research/agent_integ3.md.*

**Default vs opt-in, and why.**

| feature | state | evidence |
|---|---|---|
| Mamba fp32 pre-norm input (14b) | **default on** (`mamba_fp32_prenorm_input=false` = control) | m32-x1k / m32-x8k EXACT; m32-ut failures a subset of base; L8 harness v 2.5e-6 -> 5.9e-7 |
| learner micro-batch 2 rows, 20480-token cap (ab_base, every arm) | **default** | mbs2-full-base: 2x8192 1399 ms/row vs 1655 (1x8192), bench peak 112 GiB -> ~159 GB in-loop; 4x8192 OOM; mbs-l8x x4/xl4/xcap EXACT, grad-norm MBS1 vs 4 rel 8e-5 |
| verify_yield, tail dump, graphed-prefill knob (14aa) | opt-in | ty-x1k / ty-xl2 EXACT; L8 perf within noise; full-Nano go_tail yield arm (tyy) not run -> tail-redesign defaults (growth 0, graphed prefill) not adopted. Prefix cache at >= 4k was already run_ab_arm's default |
| streaming verify0 (14z) | opt-in | v0s-x / v0s-x8k / v0s-xnonull EXACT, order replay same_as_real; v0s-xlanes FAIL (max\|dlogp\| 1.66 / 0.76 / 0.60) = forced-fresh rows decoded on the cohort lane's drafting weights (no θ_k refit); fixed on ts-v0s-lanes-fix (14z "Lanes fix"), lanes re-allowed by the validator, rerun v0sl-xlanes pending; no full-Nano gain on record |
| window pacing (14v) | opt-in | pace-x EXACT; pace-p1 on 9.77 vs ctl 9.60 s, pace-p2 8.57 vs 8.01 s (no gain) |
| sticky rounding on hybrid (14g) | opt-in | sre2e-sr1-cur / sr1-mom / sr0-mom EXACT (c692af97); ~1.6x learner fwd+bwd; needs activation_checkpointing off |
| callcost kernels (`NEMO_RL_FP16_MOE_TRAIN_GEMM=auto`) | opt-in | at the chosen MBS 2: mbs2-full-cc 2835 ms vs base 2797 ms per 2x8192 micro-batch (shim 2803): not faster |
| engine verify (ts-engine-verify, prefetch / batched decisions) | **not merged** | every verify_on=engine arm at 486a4225 (ev4-*) and 3c013e3e (ev5-*) dies at step 1 ("canonical order incomplete ... 0 of 16 groups" after "draft verification failed"); branch is on lanes-stream 767917df and conflicts in 44 hunks with this tree's continuation code (carried walk vs pieces in flight). Merge after the crash is fixed and the branch is rebased on ts-integ3 |

**Fixes in this round.**
- *Verification seeds by canonical call identity.* Block verification (verify0, ⊥-BV waves, streamed verify0 calls)
  and continuation-piece verification drew `next(verify_seeds)` from different tasks/threads, so which call got which
  seed depended on arrival. `SpeculativeGeneration._call_seed` now hashes (grpo.seed, stream, step, kind, the call's
  rows as (row key, row verification ordinal in this step)). A row's calls are sequential (one piece in flight per
  stream; waves in order), so ordinals do not depend on timing; every (row, ordinal) is in exactly one call, so seeds
  stay fresh per call (exact in law as before). Rows without a key are named by (prompt length, crc32 of tokens).
  Cost: O(rows) driver hashing per call, nothing per rank. What remains timing-dependent: how many decoded tokens a
  piece takes (multiple chunks when available), so continuation pieces are still not bitwise reproducible run to run.
- *LEARNER_TORCHPROF.* The torch profiler's trace metadata enumerates every process group's backend config and
  raised on Megatron groups without one (dr-x-lanes). The hook only prints a key_averages table, so it now skips that
  metadata (`_get_distributed_info -> None`).
- *Section numbers.* Tail speed 14x -> 14aa (zero-advantage skip keeps 14x), window pacing 14v placed before 14w.

**Scale.** Nothing in this round adds per-rank state beyond the merged features' own (each section states it); the
seed change is driver-only; the micro-batch cap is a config function (all ranks and the planner agree) and bounds
activations by tokens (20480 per micro-batch), so longer caps fall back to 1 row automatically.

**Validation.** CPU suite tip vs base 5f835b57: thundersync 1145 pass / 7 fail vs 1077 / 7, same failures (GPU-only
test_streaming_pipeline); tests/unit/models/megatron 706 pass / 9 fail + 7 errors on both, same names. GPU single-node arms (research/integ3/val.sh, queue lines i3-*): ut, x1k, xlanes, x8k, xopt
(every opt-in at once), xsr / xsrl (sticky rounding, one engine / lanes), xm0 (Mamba fix off), resume, perf pair
p-lanes / p-async1. Full Nano: thundersync_work/sbatch/go_integ3.sh (`x`, `sr`, `mamba0`) on the frozen
../RL-integ3-scale (fork branch ts-integ3-scale). Results: agent_integ3.md and ledger I3.

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
- **MoE.** A real-model run; inference-side verification under EP (needs the engine paused
  or stepped in lockstep with the scorer); open-group buckets under EP.
- **Scaling gaps.** Context parallelism in the scorer, zero-advantage
  skip with sequence packing.
- **Streaming under PP on GPU.** Run `tests/functional/equivalence_pp.sh` (PP=2). Then
  measure learner busy time per step with `merge_pipeline_calls` on vs off and
  `min_pipeline_microbatches` 0 vs 2·PP at PP≥4 on a real model. Interleaved PP needs
  padding micro-batches up to Megatron's group constraint (not implemented).
- **Upstream.** Default `make_sequence_length_divisible_by` to 64–128 for Megatron
  training.
- **Engine-sourced q on GPU.** Run the `verify_source=engine` A/B against
  `verify_precision=model` on Qwen3-1.7B under BI (same kept fraction, engine q check 0,
  mismatch 0, few dropped drafts), then Qwen3-8B where scorer mode runs out of memory, then
  at TP=2 (`scripts/bi_tp/engine_q_tp.sh`) and TP=4 (2 nodes); EP=2 on the tiny MoE
  (`engine_q_moe.sh`) and PP=2 (`engine_q_pp.sh`), then a real MoE at EP x PP.
- **Sticky MoE routing on GPU** (section 14c). `scripts/bi_tp/sticky_moe.sh` on the tiny MoE
  (EP=2, expert-gradient noise) and on `qwen3-30b-a3b-L4`, A/B against `STICKY=0`; then the
  sparse-delta θ_{k−1} toggle so rollouts overlap verification again, and lookahead > 1.
- **Pooled ⊥-BV on GPU** (section 14d). `scripts/bi_tp/null_bv_moe.sh` (tiny MoE, EP=2, K = 1.5,
  M = 2) must read mismatch 0; then K/M sweeps on `qwen3-30b-a3b-L4` for ⊥ fraction, tail-decode
  fraction and earliest rejection among the longest drafts.
- **Fused verification + training on GPU.** Validate `fused_verify_train` end to end
  (`THUNDERSYNC_FUSED_CHECK=1`, gradient against the unfused path) and measure the saved
  learner time on short-output tasks (section 14).
