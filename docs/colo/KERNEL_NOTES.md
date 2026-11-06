# Kernel and batch-invariant (BI) learner notes

These notes cover the kernel-level costs of exact (bitwise train/generate-consistent) RL on Nemotron-3 hybrid MoE
models (Nano, Super, Ultra), the BI learner vs the stock learner, and the open kernel asks. All hardware is GB200.
Labels: **[M]** measured, **[D]** derived from measured parts, **[H]** modeled. Routers (weights and bias) are frozen
in every run cited.

## Terms

- **Batch invariant (BI):** a row's output bits do not depend on which other rows share the batch, how many there are,
  or where the row sits. The inference engine and the trainer must produce bitwise identical numbers for the same
  token, so every kernel on the forward path is BI: fixed reduction order, no split-K or stream-K, and no
  shape-dependent tile choice that changes the K order.
- **Bitwise equal:** identical raw bytes, including NaN payloads and signed zeros. "Numerically close" is a failure.
- **Forward vs backward contract:**
  - The forward (engine decode, verify, and the learner's log-prob / training forward) must be BI and bitwise equal
    between engine and learner.
  - The backward (dgrad, wgrad) and gradient accumulation only need accuracy, not determinism or order invariance.
    They may use the fastest stock kernels (split-K, stream-K, cuBLAS / CUTLASS grouped GEMM).
- **CUDA-graph rule for engine kernels:** capturable, with static pointers, no host sync, no allocation inside, and a
  grid that depends only on static shapes. A kernel that replaces several must equal their composition bit for bit.

## 1. Kernel factors used to normalize BI arms

Regime: Nano, 8k context, 4 nodes (16 GPUs) colocated, TP1 EP8 DP16, MBS 1, 64 prompts x 8 samples, lag 1. The
comparison uses five same-job pairs of the stock pipeline and the same stock loop on the BI engine + BI learner, steps
3-10 [M].

| pair | stock train s | BI train s | f_train (BI/stock, per token) | stock gen s | BI gen s | f_gen (BI/stock, per generated token) | iteration stock / BI s |
|---|---|---|---|---|---|---|---|
| 1 | 26.41 | 24.41 | 0.930 | 28.59 | 35.60 | 1.253 | 57.47 / 62.23 |
| 2 | 26.40 | 24.28 | 0.924 | 28.36 | 34.34 | 1.218 | 60.38 / 60.86 |
| 3 | 25.28 | 24.15 | 0.959 | 28.70 | 33.85 | 1.184 | 59.53 / 60.38 |
| 4 | 28.11 | 24.93 | 0.892 | 28.55 | 33.83 | 1.192 | 64.08 / 63.37 |
| 5 | 26.03 | 24.64 | 0.948 | 28.26 | 33.80 | 1.198 | 60.76 / 62.93 |
| **mean** | | | **0.931 (sd 0.026)** | | | **1.209 (sd 0.028)** | ratio 1.026 |

- **f_gen = 1.209.** Generation (prefill + decode + wake, per generated token, in-loop) costs 21% more under BI. The
  factor also includes the engine-version difference between the two trees.
- **f_train ~ 1.** Training (forward + backward + optimizer) is 0.93x stock per token, so the BI learner is not slower
  in the colocated regime. The factor to apply is 1.0: dividing by 0.93 would charge BI for being faster. The
  difference is not only kernels, since the two trees run different Megatron-Core versions.
- Normalization rule for a BI arm: `s/step_norm = iteration - (gen + engine verify) * (1 - 1/f_gen)`, with
  f_gen = 1.209 and training unchanged. Stock arms are not normalized.
- Prefill and verify-forward factors are not measured separately; they sit inside f_gen.
- f_train and f_gen were not measured at lag 2 or 4. Per-token factors should carry over, but this is unverified.

Decode per step, same tree, kernels only (Nano EP8, context 4k / 8k, median ms/step, BI / stock) [M]:

| rows per rank | 1 | 4 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| BI / stock | 1.34-1.37 | 1.24-1.28 | 1.26-1.30 | 1.21-1.30 | 1.10-1.11 |

The in-loop f_gen of 1.21 is consistent with the mix of row counts over a step.

## 2. Nano non-colocated learner: the 51 s vs 35 s gap is layout and accumulator, not kernels

- **BI side, 51 s [M]:** non-colocated learner on 8 GPUs, TP1 EP8 MBS 1. It overlaps training with verification and
  sums gradients in an **int64 order-invariant accumulator**. Training in stage measured 50.7-51.5 s.
- **Stock side, 35 s [M]:** stock learner at EP4 MBS 2, policy training 34.6-35.7 s.
- The two sides differ in three ways at once: layout (EP8 MBS 1 vs EP4 MBS 2), gradient accumulator (int64 vs fp32
  grad buffers), and call structure (many per-group train calls vs one).
- At the same layout in the colocated loop (section 1), the BI learner trains at 0.93x stock per token. **The gap is
  not BI kernels.**
- **Decision:** the int64 accumulator gave order-invariant (bitwise reproducible) gradients. Gradients only need
  accuracy, not determinism, so it is replaced by in-place accumulation in the fp32 gradient buffers, as stock does.
  - int64 stays available as an opt-in for order-check and negative-control runs.
  - Per-rank memory becomes equal to stock at every layout.
  - When a group's verification fails late, the step's earlier calls are retrained exactly without that group, so
    nothing has to be subtracted from an accumulator.
- The colocated BI learner never carried this accumulator.

## 3. The EP4 OOM: int64 accumulator memory

- [M] At EP8, the accumulator covered 5.87e9 local gradient elements, **43.7 GiB per rank**. The per-step peak was
  139.7-142.0 GiB, with 145.6 GiB reserved.
- The accumulator costs 8 B per **local** gradient element. The Adam moments it displaces (parked to host during
  training) cost 8 B per **optimizer-shard** element, sharded over DP (dense) or expert-DP (experts). "Training-phase
  memory stays at stock" therefore holds only when the shard equals the local size.
- Extra bytes per rank over stock = `8*(E_loc + D_loc) - 8*(E_loc/EDP + D_loc/DP)` (E = expert parameters, D = dense).

| layout | E_loc / D_loc (1e9 params) | int64 accumulator | extra over stock |
|---|---|---|---|
| Nano EP8 (EDP 1, DP 8) | 3.67 / 2.20 | 47 GB (43.7 GiB [M]) | +15.4 GB |
| **Nano EP4 (EDP 2, DP 8)** | 7.34 / 2.20 | 76 GB | **+44.8 GB**: stock static 97.5 GiB becomes ~139 GiB, plus ~51 GiB transient, which exceeds HBM |
| Ultra TP8 EP32 PP2 (EDP 4, DP 32) | ~8.1 / ~2.1 | ~81 GB | ~+64 GB: does not fit |

With fp32 in-place accumulation (section 2), the extra is 0 at every layout.

## 4. Learner BI kernel costs

Per-step spans on learner rank 0, Super 22-layer cut, 4 nodes, steady steps (event-to-event spans, so they include GPU
idle inside the op) [M]:

| span | per step | calls |
|---|---|---|
| MoE expert GEMM forward (training + recompute + verify forwards) | 30-39 s | ~10.5k |
| MoE expert GEMM backward | 8.3-8.9 s | 1280 |
| Mamba SSD forward | ~3.5 s | ~4.6k |
| Mamba SSD backward | ~2.7 s | 640 |
| whole training step | ~80 s | - |

Moving the backward onto stock kernels gave no measurable gain under the production config (device MoE dispatch +
selective MoE recompute). **The BI expert forward is the largest learner term.**

BI segment-GEMM forward vs stock `torch._grouped_mm` (bf16, CUTLASS), 1 GPU, uneven rows per expert, us per call [M]:

| shape (local experts, K -> N) | rows/expert | BI fwd | stock fwd | BI / stock |
|---|---|---|---|---|
| Ultra EP64 fc1 (8, 2048 -> 5120) | 256 / 1024 / 2816 | 266 / 440 / 884 | 90 / 170 / 356 | 2.96 / 2.59 / 2.49 |
| Ultra EP64 fc2 (8, 5120 -> 2048) | 256 / 1024 / 2816 | 232 / 403 / 818 | 102 / 173 / 365 | 2.27 / 2.33 / 2.24 |
| Super-L22 EP8 fc1 (64, 1024 -> 2688) | 256 / 1024 | 516 / 1049 | 146 / 333 | 3.54 / 3.15 |
| Super-L22 EP8 fc2 (64, 2688 -> 1024) | 256 / 1024 | 459 / 881 | 139 / 331 | 3.30 / 2.66 |
| Nano EP4 fc1 (32, 2688 -> 1856) | 256 / 1024 / 2816 | 395 / 770 / 1745 | 151 / 306 / 671 | 2.62 / 2.52 / 2.60 |
| Nano EP4 fc2 (32, 1856 -> 2688) | 256 / 1024 / 2816 | 390 / 765 / 1714 | 136 / 306 / 658 | 2.87 / 2.50 / 2.60 |

- The BI forward is **2.2-3.5x stock**, worst with many local experts and few rows per expert.
- The floor is <= +1.2% over stock at large M (wave quantization only); the fixed K order costs nothing in principle.
  The gap is implementation: Triton without warp specialization or 2-CTA tcgen05 MMA reaches ~0.64 PF, against ~1.5 PF
  for CUTLASS grouped GEMM.
- BI Mamba SSD forward (pinned tiles, fp32) is 1.4-1.5x stock fp32. For example, Nano TP1 T8192 takes 934 us vs 643 us,
  and Ultra TP8 T8192 takes 739 us vs 498 us.
- Colocated Nano at MBS 2 runs out of memory because the BI Mamba training path keeps ~40 GiB per rank of fp32 scan
  intermediates that stock's fused kernels never keep (12.1 GiB SSD training, 28 GiB scan-and-gate). The current
  workaround is activation recompute of the BI mixer.

## 5. Open kernel asks

Each ask has the same bitwise rule: outputs (and every in-place update) equal today's kernel bits for every input, at
every row count 1-512 and every batch composition, and the kernel is CUDA-graph capturable. Today's production kernel
is the oracle.

### K1. Learner MoE expert forward (BI segment GEMM): top learner ask
- **Does:** grouped expert GEMM. Inputs: x fp16 [M, K] grouped by expert (unpadded); W bf16 [N, K] per expert at
  arbitrary addresses (int64 offset table), rounded to fp16 in registers; device counts int32 [E] (no host splits).
  Output: fp32 [M, N]. Rows past sum(counts) are neither read nor written.
- **Invariance contract:** each output row's bits are independent of M, of the tile and of the other rows. Fixed K
  order, one fp32 accumulator per output, no split-K / stream-K. The result must be bitwise equal to per-expert cuBLAS
  without split-K and to the engine's grouped fp16-weight GEMM, which must change in lockstep.
- **Cost today:** 2.2-3.5x stock CUTLASS grouped (table above), plus a separate fp32 -> fp16 row cast (~44 us per
  call at Ultra fc1).
- **Target:** <= ~1.1x stock across the shapes, priority Super EP8 and Ultra EP32/64 at 256-2816 rows per expert. Fuse
  the row cast.
- The backward has no order requirement: use stock kernels.

### K2. Learner Mamba SSD BI forward/backward with stock-sized activations
- **Does:** fused SSD chunk scan + gate for training.
- **Contract:** forward bitwise equal to the current BI forward, which is also the engine decode's reference. Backward
  needs accuracy only.
- **Cost today:** forward 1.4-1.5x stock fp32, plus ~40 GiB per rank of saved fp32 intermediates at Nano MBS 2, which
  forces recompute.
- **Target:** saved activations at stock's fused-kernel footprint and forward at <= stock fp32 time.

### K3. Engine BI Mamba-2 decode step
- **Does:** one decode token per row. To match the trainer's chunked scan bit for bit, decode keeps the current
  128-token chunk's inputs (x, dt, B, C, fp32) in per-slot replay buffers. Each step it re-runs the chunk-scan
  arithmetic for the one output row it needs. When a slot's chunk fills, it commits the chunk's end state into
  `ssm_state`.
- **Contract:** output rows, replay buffers, slot cursor and `ssm_state` are all bitwise equal. The bits are those of
  the training SSD tile on sm_100: `mma.sync` TF32 with raw fp32 operands and pinned tile K, `ex2.approx` after a
  multiply by log2e, and `div.full` sigmoid. Inactive graph lanes stay untouched but get a zeroed output row.
- **Cost today [M]:**
  - Super 22-layer cut, TP2, 144 rows: BI chain 120.8 us per layer, against 112.1 us for a fair stock with fp32 state.
    That is a +0.09 ms per decode iteration tax, and the BI scan alone already beats stock's scan.
  - Ultra per-GPU shape in the harness: 25.7 / 38.0 / 97.4 / 249 us at 1 / 8 / 64 / 256 rows, against 9.3 / 11.4 /
    31.9 / 97.3 us for stock fp32-state.
  - A fused candidate reaches 16.1 / 26.7 / 63.4 / 178 us.
- **Target:** <= 8 us per layer at 1-4 rows. At 144 rows: chain <= 70 us (scan <= 55); at 192 rows: <= 90 us. Speedup
  >= 1.25 at 64-256 rows on Ultra TP8 / Super TP4. Stretch: fold `causal_conv1d_update` and the gated RMSNorm into the
  same kernel.
- Known dead ends: a fully fused Triton kernel was 2x slower (occupancy), and a pipelined Triton scan was not bitwise
  at num_stages > 1. Promising: more independent MMA chains per SM, and persistent CTAs that prefetch the next
  (row, head).

### K4. Engine ordered TP row collectives (reduce-scatter / all-gather), TP 2-8
- **Does:** the sequence-parallel row collectives around every fp16-GEMM layer (22 RS + 22 AG per decode step at
  Super).
- **Contract:** each RS output element = ((p_0 + p_1) + p_2) + ... in fp32 round-to-nearest adds, with ranks in
  ascending order. There must be no unordered `multimem.ld_reduce` at TP >= 3. AG is a copy, so any copy is exact,
  including a `multimem.st` broadcast.
- **Cost:** previously 15-17 us per call against 6.6-9.6 us for stock NVLS (latency-bound, ~140 GB/s achieved). A
  one-barrier version has landed: -0.22 to -0.25 ms per decode iteration.
- **Target:** <= 8 us per call at 1-192 rows, every TP 2-8. Options: one-pass AG into a persistent symmetric buffer;
  double-buffered staging to drop the trailing barrier; fusing RS -> residual add -> RMSNorm (the norm's exact
  reduction order) -> AG.

### K5. Engine decode expert GEMM at few rows per expert
- **Contract:** the same fp16 x fp16 -> fp32 MMA chain over K, in today's order and tile-K (today's kernel equals cuBLAS
  without split-K at every shape).
- **Cost [M]:** 46 us per call at batch 1 (Ultra 27-layer cut, TP4 EP4, ~3.0 TB/s). At Super, 1 row: 11.3 us per call
  after the landed rewrite. At 160 rows the target (<= 105 us) is met at 99.7 us.
- **Target:** >= 6 TB/s at <= 2 rows per expert, i.e. <= 8 us per call at Super 1 row and <= 6 us at Ultra TP8 EP16.
  Approach: split each expert's output columns across CTAs (no split-K), stream the weights with TMA, keep the rows in
  registers.

### K6. Engine dense GEMMs at decode (M <= 64)
- **Contract:** bits equal to cuBLAS no-split-K: same K loop order and MMA instruction shape.
- **Cost:** 1.29 ms per step at Ultra-L27 TP4 batch 1, against a ~0.7 ms weight-byte floor.
- **Target:** a skinny tensor-core GEMM at >= 85% of HBM bandwidth that skips padded rows nobody reads.

### K7. Fused per-layer epilogues + programmatic dependent launch (PDL)
- **Does:** fuses residual add (fp32) -> RMSNorm (keeping the BI row-mean reduction layout) -> fp16 cast. Also fuses
  the MoE glue (router hi/lo split + combine, cross-fade router, count, permute; relu2 x prob into the fc1 epilogue)
  and the small Mamba glue kernels. PDL between consecutive kernels in the graph hides the ~0.77 us per kernel
  boundary.
- **Cost:** ~1.77 ms per step at Ultra-L27 batch 1: ~30 kernels per layer, 779 in-graph boundaries.
- **Target:** <= 12 kernels per layer, <= 0.4 ms per step.

### K8. Cross-fade routing: expert GEMM + EP dispatch that skip zero-weight band slots
- **Context:** cross-fade routing gives each token its top-k experts plus up to c band candidates, with a gate weight
  that ramps from 0 to 1 near the top-k boundary. The final routing uses m = 0.01 and a band cap of 8. Fixed-width
  dispatch pays cap/k (~+36%) in expert compute and all-to-all, but only 2.7 (Super) to 4.7 (Ultra) of the 8 band
  slots are live.
- **Contract:** a skipped slot gives exactly the bits of computing it with weight 0, and the combine sums the same
  nonzero terms in the same order. Static capacity T * (k + c) keeps graph shapes fixed. Results are batch invariant.
- **Target:** expert-path cost <= (k + live slots) / k of stock top-k, i.e. a routing cost of ~1.07-1.12x instead of
  ~1.36x. The learner already drops zero-weight slots; this ask is for the engine.
- **Also asked:** a per-layer band width m in the fused router (one fp32 m per MoE layer, same contract).

### K9. Keyed 53-bit ("wide") sampler: done
- **Does:** a Gumbel-max sample per row from softmax(logits / T), with noise hashed from (salt, request id, position,
  column). An fp32 pass selects candidates, then an exact fp64 race picks the winner; ties go to the lowest column.
- **Contract:** returned indices identical for every input, including -inf/NaN/+inf rows (all -inf -> column 0),
  `vocab_limit`, strided rows, bf16/fp16 logits, and per-row temperature.
- **Status:** a multi-program version is integrated: 202 -> 14 us device time at 1 row, bitwise.
- **Remaining:** a single capturable launch, <= 60 us at 160 rows and <= 10 us at 1 row.

### K10. BI log-softmax: mostly done
- A 3-kernel multi-program version is bitwise and takes 6.1 us device time at 1 row (was 120 us).
- Remaining: a single-launch version, to remove ~50 us of host launch time.

### K11. Row-tail vocab scan for block verification
- **Does:** after verify forwards, each verified row runs a vocab scan for the block walk (exact-nucleus / Gumbel-key
  comparison, residual normalizer, first-rejection search over the draft block), plus q-row reconstruction.
- **Cost:** 33 ms per row at Ultra, against a ~1.7 ms floor. Most of it is the q-head GEMM; the rest is host-serial
  code.
- **Target:** one fused, vocab-parallel (TP-sharded), batch-invariant kernel per batch of rows.

### K12. FA4 paged attention, TMA path: V rows past the sequence length leak
- **Problem:** with page size = tile_n = 128, the TMA KV path multiplies the last tile's V rows past `seqused_k` with
  P = 0. If those rows hold NaN or inf (uninitialized pool memory), the output row turns non-finite. The non-TMA path
  (page 256) is clean.
- **Ask:** zero V rows at or past `seqused_k` in shared memory before the PV MMA, so that the TMA path is bitwise equal
  to the non-TMA path on finite data.
- **Mitigation in use:** the engine zeroes newly acquired V pages.

### K13. Canonical-block vocab-parallel log-probs: done
- **Does:** token log-probs, residual masses and inverse-CDF draws from per-block (4096-column) log-sum-exp partials,
  combined in a fixed global order. No full-vocabulary all-gather is needed, and the result is bitwise identical
  across TP 1/2/4/8 and batch composition.
- **Status:** done and integrated, +3% over non-BI.

### K14. TP-invariant mode: open
- **Goal:** logits and log-probs bitwise identical across TP 1/2/4/8, so that engine and learner can use different TP.
- **Needs:** four TP-invariant reduction primitives plus a mode in Megatron-LM and NeMo-RL.

### Near floor already (no action)
Hi/lo router and LM-head GEMMs, canonical-block vocab LSE, EP-invariant limb combines, and the attention kernel.

## 6. Ranked bottlenecks

**Kernel level, engine decode step** (Super 22-layer cut, TP2 EP8, 144 rows per rank, ms per step) [M]:
- BI compute is at stock parity: 4.41 vs 4.37, because the BI expert GEMM now beats the stock fused MoE.
- The whole step is 5.95 BI vs 5.64 against a fair stock with fp32 SSM state, so the BI tax is +0.31 ms.
- That tax was mostly TP collectives (K4, +0.33), now cut by 0.22-0.25. Mamba decode accounts for +0.09 (K3).
- Ultra is tail-decode-bound, so every engine ms converts directly to step time.

**Kernel level, learner:**
1. BI MoE expert forward (K1): 2.2-3.5x stock on 30-39 s of span per step at Super. This is the largest learner
   term.
2. BI Mamba SSD forward and activation memory (K2): 1.4-1.5x stock, and forces recompute at Nano MBS 2.
3. Gradient accumulator (section 2-3): int64 cost ~2.3 s per step at Super plus the EP4 / Ultra OOM. Removed by fp32
   accumulation.

**System level** (Super 22-layer cut, 4 nodes, lag 1, s/step) [M]:

| # | item | s/step | note |
|---|---|---|---|
| 1 | training in stage | 45.3 | EP exchange ~10 s of it is common work (stock pays 9.5); cross-fade band ~4.4 |
| 2 | verify wall | 22.1 -> 14.7 | deferring ~half the drafts into the group's own training forward cut it |
| 3 | learner wait / serial learner work | ~5.6 | hand-out, prep, pre-verify RPCs |
| 4 | outside the stage | 1.9 | equal to stock's own |
| - | BI learner kernels vs stock (same layout) | ~0 | not a lever at Super |

- At Ultra (27-layer cut, 16 nodes), the tail decode chain binds (8.87 ms per token). Engine decode kernels (K3-K8) are
  the main lever there.
- At Super, the learner binds, and K1 / K2 plus call structure are the main levers.
