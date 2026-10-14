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

**fp16 range.** Max |GEMM input| over 16 MATH prompts x 512 greedy tokens (HF bf16, decode
included; `thundersync_work/fp16_range_probe.py`): Qwen3-8B 10752 (6.1x headroom to 65504),
Qwen3-14B 7296 (9x), Qwen3-30B-A3B 484 (135x, expert down_proj). The dense down-projection input
(the SwiGLU output) is the only site near the limit; every other site is under 260. Opt-in
`down_proj_prescale_log2: s` multiplies the down-projection input (dense fc2 and the MoE expert
fc2) by 2^-s before the fp16 cast and the fp32 output by 2^s, with wgrad scaled to match. That
is exact apart from fp16 subnormals and widens the range by 2^s. Default 0; probe a new dense model
before running it and set s=2-4 if headroom is below ~4x. The setting must match in training and
generation (checked by the mismatch validator).

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
are ordered by the current scores, so F(θ; plain_topk(θ)) is bitwise the plain forward.
An anchor row with a negative or repeated id means "self" (plain top-k).

**Anchor schedule (exogenous).** A at position t = the plain θ_{k−1} forward's top-k on the
rollout's own prefix, for every position up to the rollout cap c = prompt + max_new_tokens
(content-independent); "self" for rollouts never drafted (or whose draft was dropped for
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
re-prefills write the same positions. A rebuilt engine's forward is wrapped into the same log
and new router instances are re-armed. Prompt positions are logged by the draft's own prefill
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
verify_source=engine (BI), draft_lookahead=1, max_rollout_turns=1, no tail_lookup, no
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
(`draft_lookahead>1`), multi-turn rollouts (a later turn's prefill would need the anchors in
the engine: sticky replay in engine prefill, not built yet) and a horizon c below the rollout
cap (same engine feature) are not built yet.

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

## 14f. Momentum-predicted draft weights (`thundersync.draft_weights: momentum`)

*Opt-in, default `current`. Unit-tested (formula, Megatron DistributedOptimizer on 1 and 2 ranks
with TP / EP / DP sharding, driver order, config); end-to-end runs: see "Measurements".*
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
| Engine-q rows (lazy residuals, residual sample) | LM-head stash of version k = θ_k's head, taken in `score_prev` at the end of step k | the head is stashed by `refit_draft_weights` while θ̂ is loaded on the learner, as version k; `score_prev` does not overwrite it. `[engine q check]` (logged log q == hidden × stashed head) fails if the two ever differ |
| Scorer q (`verify_source=scorer`) | recomputes q from a θ_k weight stash | rejected by validation (would be the wrong q) |
| Sticky anchors (14c) | A = plain drafter routes = plain θ_{k−1} | A = plain θ̂_k routes. R1: θ̂_k depends only on the state before g_k, never on θ_k's update outcome or on the draft. R2: anchors are the drafter's routes on the rollout's own prefix; R3: the learner applies the logged anchors, as before |
| Sticky continuation rounds | engine still at the drafting weights when `verify_sticky` runs; continuations draft + anchor with them | unchanged: the engine keeps θ̂ from the draft refit until the top-of-loop refit, which runs after `verify_sticky`; the route check (`[route check]`) re-prefills on θ̂ and must match the logged anchors |
| Rollouts of step k (sampled from p_k = θ_k) | engine at θ_k during the step | drafting (and the θ̂ refit) starts only after the step's last rollout returns; drafted rollouts' plans were all published before (each rollout waits for its plan), chunked verification is awaited |
| Tail decodes after a rejection (non-sticky) | on the refit engine (θ_{k+1}) | unchanged |
| Refit ordering | one refit per step (θ_{k+1} at the top of the loop) | two: θ̂_{k+1} mid-step k, θ_{k+1} at the top of step k+1 |
| Learner training forwards during the swap | n/a | the learner's parameters hold θ̂ only inside one critical section (load → stash → refit → restore) guarded by the learner lock that every training dispatch holds; restore is the optimizer's own main → model copy + all-gather, bitwise θ_k (tested); grads/main params are never touched; the fp16 expert-weight cache keeps θ_k's casts (valid again after restore; no forward runs while swapped) |
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
- Driver: `SpeculativeGeneration.refit_draft_weights(refit)`: load → stash LM head (version k)
  → `refit` (the existing `refit_policy_generation`, unchanged transport) → restore (also on
  failure). Metrics `draft/*`.
- Loop: with `draft_weights=momentum`, the drafter task starts when the step's last rollout
  returns (instead of `draft_start_frac`), awaits the chunked verification task, takes the
  learner lock (held by every training dispatch) and runs the draft refit in a thread, then
  drafts. Validation (`validate_draft_weights`): speculate, block, engine q, verify_on=learner,
  lookahead 1, no draft_after_verify, no fused_verify_train, distributed optimizer, Adam, no
  precision-aware optimizer (its masters live inside TE FusedAdam: not built yet).

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
  ≈ 0 for a bf16 model), plus the existing LM-head stash (V/TP × d × 2 B). No copy of the model,
  master or moments; nothing is gathered except the parameters themselves.
- communication: two extra parameter all-gathers over each DP group (θ̂, then θ_k), each equal to
  the gather Megatron already does after every optimizer step (2·P_local·(DP−1)/DP bytes per rank,
  P_local = model elements on the rank); plus one extra engine refit per step through the
  existing sharded refit path (model bytes per engine rank; with the sparse-delta refit of
  `agent_system_structure.md` this becomes ~0.03 B/param).
- critical path: the draft refit runs after the step's last rollout, between learner
  dispatches; it delays the learner's remaining dispatches by load + stash + refit + restore
  (measured below) and the start of drafting by the same amount.
- drafting window: one weight set per engine, so θ̂ drafts cannot share the engine with the
  step's own θ_k rollouts; drafting runs from the last rollout to the optimizer step instead of
  from `draft_start_frac`. Drafting while the rollout tail decodes needs a second weight version
  on the engine (a θ̂ replica, or per-request weight versions via a sparse delta): not built yet.

**Measurements.** (filled below as runs complete)

**Not built yet.** Precision-aware optimizer (masters in TE FusedAdam), Megatron-FSDP,
fp8/fp4 param gather, a two-step draft for lookahead > 1, overlapping the draft refit with the
learner's tail (e.g. a sparse delta applied on the engine without the learner-side swap).

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
