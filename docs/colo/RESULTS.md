# Colocated exact on-policy RL with speculation: verified results (Nov 2026)

This file holds the measured numbers behind the colocated work. Theory is in `docs/colo/THEORY.md`; the rules every run
follows are in `docs/colo/DECISIONS.md`. Every number below was measured on 16 GPUs (4 nodes) unless stated otherwise.

Status markers:
- **[provisional]**: the Nano numbers stay provisional until the stale main-pin unit tests on the production branch
  (about 130 fused-decode / decode / parity / wide-key tests) are confirmed as stale tests and not a code path the arms
  run. A test-only fix exists. On the newest test head the wake-order and parity suites pass. The decode suite (4
  failures) and the fused suite (16 failures) still need a re-test.
- **[degenerate regime]**: layer-truncated Super (L22) and Ultra (L27) emit noise text and run every row to the length
  cap (mean 8174 of 8192). They have no length tail for partial rollout to hide. Use them for plumbing, memory and
  scale only, never for headline ratios.
- **[speed only]**: measured on a tree that still had the colocated wake race (section 14). The timing is valid; the
  run is not exactness evidence.
- Commit shas, where they appear, refer to the pre-squash history. The public branches are squashed, so read
  "after the wake-barrier fix" and similar descriptions as the code state.

## 1. Definitions

### Arms

| arm | what it is | exact? |
|---|---|---|
| **a = SPR** | Stock NeMo-RL colocated partial rollout, in_order, lag K. In-flight requests pause at the stand-down and resume at the new weights. | no (prefixes from π_{n-1}) |
| **b = SPR-BI** | The stock SPR loop on our engine with batch-invariant (BI) numerics. Engine settings are identical to c's. | no |
| **c = ours** | One of the methods below, on the same engine and settings as b. | see method |
| XS | Exact colocated sync, lag 0, on the BI engine. This is the exact reference. | yes |
| XPR / XPR-S | Exact colocated partial rollout: carried prefixes are ⊥-BV-verified against θ_n. XPR-S adds spare prompts. | yes |
| wvs / wv | Verify at the wake: lookahead drafts cut at the stand-down are verified after training and continued inside the next phase (fixed G, spares, M 1.35). | yes |
| VG / VG-cap | Variable group size with wake verify. VG-cap caps rows in flight at SPR's count. | yes |
| IS-SPR | Stock pause/resume plus a keyed rescore at the wake, with importance weights. Group weights at K=1. | unbiased, not exact in law |
| IS-RLOO | IS-SPR with per-row weights, RLOO advantage (no std) and a constant length normaliser L_c. Used at K≥2; at K=1 it uses group weights. | unbiased up to the clip |
| EX-T | wake_verify plus exact tree speculation on the fork tails (shared `tree_spec` module). | yes |
| XS-SD | Exact sync plus keyed (tree) speculative decoding. | yes |
| BX-SD | Block verification at the wake plus a balanced fork lane. | yes |
| DX | Exact on-policy in the non-colocated async setting, with GRPO-Z. | yes |

### Ratios

- **c / b = method overhead.** Same tree, same engine, same numerics.
- **b / a = BI-numerics tax.** On the Megatron-LM main pins this ratio also includes the Megatron-LM version difference
  and the MoE routing implementation, because stock does not run on Megatron-LM main.
- **c / a = end to end.**

### Kernel-normalized column (kn)

The BI kernels are slower than stock, and the method should not be charged for that. The per-phase factors come from
five same-job SPR vs SPR-BI pairs at Nano 8k, 4 nodes, lag 1:
- **f_gen = 1.209** (sd 0.028): BI / stock seconds per generated token, measured in the loop (prefill + decode + wake).
- **f_train = 0.931** (sd 0.026) per trained token. It is applied as **1**: the BI learner is not slower, and dividing
  by 0.93 would charge BI for being faster.
- For a BI arm: `s/step_kn = s/step − (generation + engine verify) × (1 − 1/1.209)`. Stock arms are unchanged.
- For wake_verify arms the BI engine decode is the continuation phase. Their verify runs on the learner and is not
  scaled.
- Same-tree decode only (kernel, ms/step), BI / stock: 1.34-1.37 at 1 row/rank, 1.24-1.28 at 4, 1.26-1.30 at 16,
  1.21-1.30 at 32, 1.10-1.11 at 64.
- Not measured: f_gen and f_train at lag K = 2 / 4.

### Windows and steps

- **Steady steps.** Scoring starts at step 3 or later; the early steps include compile and warm-up. `score_pair.py`
  (in `research/thundersync_rl/scripts/colo/score/`) does the scoring.
- **Whole-cycle windows.** Stock-style arms (SPR, SPR-BI, IS-SPR) alternate long and short steps with period K + 1.
  A window must hold whole periods. Otherwise the mean is biased by up to ~3 % (the XS headline moved from 1.411 to
  1.392 when it was rescored). XS has no alternation, so its mean barely moves.
- **Same allocation, both orders.** The arms of a set run back to back in one allocation, ×2 with the arm order
  swapped, and are pooled.

### Checks carried by every reported run

- Routers frozen: router weights and expert bias, in every arm including stock. Checked in the effective config and by
  a per-step probe.
- Ours: the train-log-prob sentinel (learner training log-prob == engine sampling log-prob, bitwise, every token)
  reports `inexact = 0` on every step.
- `policy.offload_optimizer_for_refit=false` in every arm.
- `pair_check` (config diff: METHOD keys only between b and c) and `levers_audit` pass in preflight.

## 2. Headline: Nano real text, 8k, 4 nodes (TP1 EP8), lag 1 [provisional]

Regime: full Nemotron 3 Nano 30B-A3B, DAPO-Math with the real reward, 8k generation, P 64 × G 8, MBS 1, KV persist on
every arm, each arm at its best config. Stock runs without torch_memory_saver (TMS); our arms and SPR-BI run with TMS.
b and c use prefix caching and chunked prefill, 64 rows/rank, and a 13 GiB engine buffer. Routers are frozen
everywhere.

| c | K | code state | a SPR | b SPR-BI | c | c / b | b / a | c / a | kn c / b | kn c / a |
|---|---|---|---|---|---|---|---|---|---|---|
| XS (exact sync) | 1 | before the wake-barrier fix [speed only] | 56.63 | 60.62 | 84.37 | **1.39** | 1.07 | 1.49 | 1.36 | 1.31 |
| IS-SPR | 1 | after the wake-barrier fix; same job ×2, order swapped | 56.08 | 60.75 | 72.11 | **1.19** | 1.08 | 1.29 | **1.18** | 1.16 |

All values are s/step over whole-cycle windows (XS set: steps 3-10; IS-SPR set: steps 3-14).

Per-replica IS-SPR set:

| order | a SPR | b SPR-BI | IS-SPR | IS-SPR / b | b / a | IS-SPR / a | kn IS-SPR / b | kn b / a | kn IS-SPR / a |
|---|---|---|---|---|---|---|---|---|---|
| SPR-BI, IS-SPR, SPR | 56.31 | 61.32 | 72.50 | 1.182 | 1.089 | 1.287 | 1.173 | 0.985 | 1.156 |
| IS-SPR, SPR-BI, SPR | 55.84 | 60.17 | 71.71 | 1.192 | 1.078 | 1.284 | 1.187 | 0.973 | 1.156 |
| pooled | 56.08 | 60.75 | 72.11 | **1.187** | **1.083** | **1.286** | 1.180 | 0.979 | 1.156 |

- Kernel-normalized s/step: SPR-BI 55.47 / 54.35 and IS-SPR 65.07 / 64.54. The XS set normalized gives SPR-BI 54.73
  and XS 74.40: c / b 1.359, b / a 0.966, c / a 1.314.
- Per phase (same window), exposed generation: SPR 28.6 / 28.2, SPR-BI 33.9 / 33.7, IS-SPR 42.8 / 41.4.
- Per phase, policy training: SPR 25.4 / 25.2, SPR-BI 24.7 / 24.2, IS-SPR 27.0 / 27.5.
- IS-SPR's extra time is +8.3 s in generation (the keyed rescore at the wake) and +2.8 s in training (the cross-fade
  learner dispatch). Its step alternates 37 / 105 s and 54 / 89 s.
- No illegal memory access occurred in any of the 6 arms, and all 16 wakes ran in every arm. Routers were frozen
  16/16. The IS-SPR sentinel ran 16 steps with 0 inexact tokens. levers_audit passed on all 6 arms.
- Known config differences between a and b, which affect b / a only:
  - mamba_recompute is on for b only (costs b time);
  - the deferred-metrics and MoE device-dispatch learner env are on for b only (helps b's training: 24.5 vs 25.3 s).
- The XS headline must still be re-measured after the wake-barrier fix. Stock's best config is also being re-checked
  with the deferred-metrics lever (generic levers go to stock too).

The XS set before the whole-cycle correction read 1.411 / 1.078 / 1.520 (c / b, b / a, c / a). Those numbers are
superseded by the table above. The pre-registered prediction for XS / SPR-BI was 1.41 (band 1.35-1.47).

### Stock SPR, torch_memory_saver on vs off (same job, KV persist)

| order | TMS on | TMS off |
|---|---|---|
| on, off | 59.30 | 55.22 |
| off, on | 58.56 | 55.64 |
| pooled | **58.93** | **55.43** |

TMS costs stock +3.5 s/step, mostly in weight sync (3.4 vs 1.0 s). So stock runs TMS off (its best config), and ours
and SPR-BI run TMS on. A separate whole-cycle A/B gave on 60.39 vs off 56.63.

## 3. IS-SPR / IS-RLOO lag sweep, K = 1 / 2 / 4, vs SPR-BI at the same K [provisional]

Same tree for both arms of each pair, after the wake-barrier fix. Nano 8k, P 64 × G 8, 4 nodes (TP1 EP8 DP16), 12
steps, prefix caching + chunked prefill on, KV persist + TMS. Weights auto: group at K ≤ 1, row at K ≥ 2.
Each window is whole cycles.

| K | unit | IS-RLOO s/step | SPR-BI s/step | ratio | row ESS (min) | group ESS (min) | row Var(s) | group Var(s) | eps_hat | KL / token |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | group | 74.31 | 60.30 | **1.23** | 0.970 (0.929) | 0.803 (0.373) | 0.032 | 0.290 | 0.012 | 5.3e-6 |
| 2 | row | 81.38 | 56.41 | **1.44** | 0.968 (0.895) | 0.825 (0.607) | 0.063 | 0.920 | 4.9e-5 | 6.7e-6 |
| 4 | row | 81.75 | 58.47 | **1.40** | 0.969 (0.918) | 0.810 (0.530) | 0.032 | 0.225 | 0 | 2.3e-6 |

- The estimator is healthy at every K: row ESS ≈ 0.97, clip excess eps ≤ 1.2 %. Row weights beat group weights at every
  K (row Var(s) 3-15× lower).
- The kn column vs SPR-BI equals the raw ratio, because both arms run the same kernels.

Per phase (s/step):

| K | arm | total | exposed generation | policy training | weight sync |
|---|---|---|---|---|---|
| 1 | IS-RLOO | 74.31 | 42.09 | 27.33 | 1.96 |
| 1 | SPR-BI | 60.30 | 33.34 | 24.58 | 1.02 |
| 2 | IS-RLOO | 81.38 | 49.53 | 27.00 | 1.91 |
| 2 | SPR-BI | 56.41 | 29.49 | 24.38 | 1.00 |
| 4 | IS-RLOO | 81.75 | 47.56 | 28.05 | 2.26 |
| 4 | SPR-BI | 58.47 | 0.04 | 24.59 | 1.11 |

**Cost driver: the wake re-prefill.** At every wake, every row decoded before it is re-prefilled under θ_n. That is
2.1 / 2.6 / 2.9 M tokens per step at K = 1 / 2 / 4, and about 80 % of it belongs to rows that had already finished
(they need only their θ_n log-probs). The engine phase ends once P groups are ready. So when a phase is spent
re-prefilling a finished lookahead batch, the next batch is generated while the trainer waits. The step becomes
periodic: K=1 alternates ~100-115 / 40-57 s; K=4 runs four ~45 s steps, then one of 218-231 s. The log-prob pass itself
is only 1.1 / 1.7 / 2.4 s per rank per step; the rest is the re-prefill.

| K | predicted IS-RLOO / SPR-BI | measured | verdict |
|---|---|---|---|
| 1 | 1.07-1.13 | 1.23 | slower: the prediction did not count the re-prefill starving the buffer |
| 2 | 1.10-1.22 | 1.44 | slower, same cause |
| 4 | 1.15-1.40 | 1.40 | at the band edge |

The weight statistics were in or better than the predicted bands, except KL at K=1 (5.3e-6 vs 0.2-2.7e-6).

**Next, approved and pre-registered.** Take s_r from the learner's first BI training forward, which is bitwise equal to
the engine. Then re-prefill only rows that continue.

| K | now | predicted | ratio now | ratio predicted |
|---|---|---|---|---|
| 1 | 74.31 | 73.0-74.0 | 1.23 | 1.21-1.23 |
| 2 | 81.38 | 63-71 | 1.44 | 1.12-1.26 |
| 4 | 81.75 | 66-74 | 1.40 | 1.13-1.26 |

IS-RLOO toy enumeration (exact, CPU): at K = 1-4 the IS-RLOO gradient equals G∇J/L_c to 1e-12. A row + 1/l
normaliser biases by 0.003-0.015, and row + LOO std by 0.010-0.042. That is why std is dropped and L_c is constant.

### Earlier IS-SPR results (context)

- **Cross-fade routing is required.** The first IS-SPR and keyed exact-SPR runs had cross-fade off (cloned from the
  SPR-BI config). Without it Nano whole-row acceptance is 43-47 % (85-89 % with it), group ESS was 0.04-0.17 on half the
  steps, and KL was ~8e-5 per token. Those conclusions were retracted.
- With cross-fade on (Nano 8k, reduced engine: PC/CP off, MR 96): IS-SPR / SPR-BI = 1.07-1.12×, pooled 1.096×. On odd
  steps group ESS was 0.73-0.90, row ESS 0.95-0.98, and KL 0.2-2.7e-6 per token. These used the scorer's default
  (non-whole-cycle) window. Rescored over whole cycles they read ~1.17×, not 1.09×.
- With prefix caching and chunked prefill on the BI comparison engine (separate jobs, not same-job): 70.60 vs 62.61 =
  1.128×.
- Nano 16k TP2 EP8 (one replica, cross-fade off): IS-SPR / SPR-BI 1.048×, IS-SPR / SPR 1.38×.
- lr 0 control: rescore and generation log-probs are bitwise equal, s ≡ 0 on every step (up to 3.57 M tokens/step).
  No stale-state or numerics path enters s.

## 4. Exact speculative track at Nano 8k

### EX-T (tree speculation on the wake_verify fork tails)

| arm | s/step | / SPR-BI | kn / SPR-BI |
|---|---|---|---|
| SPR-BI | 66.49 | 1.00 | 1.00 (kn 60.48) |
| EX-T + MTP (r1_e3 head) | 91.20 | **1.372** | 1.368 |
| EX-T, references only (same tree) | 103.58 | 1.558 | 1.552 |
| EX-T + MTP (c3 head), control arm of the BX-SD pair, one order | 86.63 vs SPR-BI 67.86 | **1.28** | 1.29 |

Settings: same job, both orders, steps 2-9 (whole cycles), KV offload + TMS, buffer 20, MR 96, routers frozen, sentinel
0.
- Per phase, EX-T+MTP vs refs-only: continuation 44.75 vs 51.63, verify 6.03 vs 9.17, training 27.46 vs 27.90. MTP
  saves 12.4 s/step (−12 %).
- Prediction (registered before the run): 94 s (91-98). Measured 91.2: hit. c3 control: 85 (82-89) predicted, 86.63
  measured: hit.
- The first EX-T (refs only, earlier tree) was 101.28 s = 1.65× SPR-BI, against 1.84× for wvs0 (block verification at
  M 1, no spares, MTD 4).
- Exactness:
  - in-loop CHECKS on Nano 8k (16 ranks): mismatch exact every step on 3.3-3.5 M tokens, max |d| 0;
  - graphed tree stage with 0 fallbacks;
  - the 8 tree negatives all fail.
- Graphed tree-stage fixed overhead: 0.24 ms/step (target ≤ 0.35).

### XS-SD (exact sync + keyed speculative decode)

| arm (Nano 8k, same jobs, both orders) | s/step |
|---|---|
| SPR (stock) | 56.19 |
| SPR-BI | 59.86 |
| SPR-BI + the same K2 lookup drafter | 79.32 |
| XS-SD, K2 own-history lookup drafter | 118.79 |

- XS-SD / SPR-BI = 1.98 and XS-SD / (SPR-BI + same drafter) = 1.50. On Megatron's 1+K speculative path with lookup
  drafts, XS-SD is slower than plain XS.
- Slowest row: 1.16-1.20 tokens/forward. Iteration cost at width 3: 13.2 ms vs 6.84 ms.
- The latency bench attributed the extra cost: host critical path +2.40 ms, in-graph per-token glue +1.52 ms
  (later removed), sequential one-token Mamba replays +1.56 ms (later fused), token-proportional width cost ~+0.75 ms.
  Only the last one is intrinsic.
- After the fused W-token Mamba verify, the per-token verify cost is 28-33 µs per rank (target ≤ 34, met). The fixed
  speculative overhead is 1.65-1.95 ms per step on Megatron's path (target ≤ 0.35, missed). On `tree_spec` the fixed
  overhead is 0.29 ms (met).
- XS-SD + MTP (c3 head) on the main pin with prefix caching and chunked prefill:
  - T1 bitwise vs XS: 1024/1024 rows, 4 steps, buffers 16 / 20 / 30 / 48;
  - the tree_stale_mamba negative fails;
  - 2.06-2.13 tokens/iteration;
  - generation 19.5-20.7 s vs 25.5-25.9 s for plain XS (1.26×) on 2k rows.
- The fair 4-arm set with the c3 head (SPR / SPR-BI / SPR-SD / XS-SD) is pending. Prediction: 1.17-1.38× SPR-BI.

### BX-SD (block verification at the wake + balanced fork lane)

- Gates (1 node, Nano-L8): the positive is EXACT 5/5 and tree_accept_all fails. Lane placement audit: 0 misrouted rows.
- Pair 1 (fixed fork rule): 97.5 s = 1.49× SPR-BI against the EX-T+MTP control at 91.7 (1.40×). The prediction was 86
  (82-91): a MISS.
  - The rule "remaining > L/2" classed accepted-but-cut rows as forks on alternate steps and overloaded the lane.
  - On correctly placed steps the lane worked: continuation 36-39 s vs 44-45 s for the control.
- Pair 2 (choose_lane v2): a MISS again. The lane put ~108 rows on ranks with max_requests 96 and starved the tree
  of node room (6-8 vs 28-31 nodes per forward). Now fixed with a hard per-rank max_requests and room pricing.
- Capacity model at Nano 4 nodes, with the same max_requests as SPR-BI: the gain is capped at −9 % of the continuation
  (≈ −3.5 s/step) at best. A third pair is queued.
- Lag-K wake_verify is built; the powered K = 2 / 4 gates are pending.

### Draft head c3 (multi-depth Nano MTP drafter)

The head is a recurrent EAGLE-style MCore MTP layer (47.8 M params). It was trained with chain unrolling over depths
1-3 and full-vocab KL on Nano's own T=1 samples. Every engine run is EXACT: bitwise vs plain keyed decode, 0 graph
fallbacks.

| quantity | value |
|---|---|
| offline keyed acceptance, held-out, depth 1 / 2 / 3 / 4 | 0.750 / 0.702 / 0.689 / 0.674 (previous head r1_e3: 0.748 / 0.29 / 0.15) |
| in-engine per-depth conditional acceptance (forced depth 4) | **0.704 / 0.664 / 0.666 / 0.665** |
| slowest row, in-engine, production adaptive depth | **~1.9 tokens/forward** (1.905; the offline hardest row 1.83 at D3) |
| median row | 2.1-2.8 tokens/forward |
| draft cost | **0.28-0.35 ms per depth** (ctx 2k-8k, 16-64 live rows); KV append 0.06 ms/iteration |
| head KV | 513 MiB/rank at 64 slots × 8k |

The slowest rows are hard, capped, high-entropy rows (per-row α ~0.54 at depth 1). Raising them needs a better drafter
on high-entropy text, not more depth. The next round targets a slowest row ≥ 2.2.

### Acceptance audit: is Nano acceptance "too low"? No bug

| acceptance | expectation | measured |
|---|---|---|
| keyed, q := p | exactly 1 | 1.0000 (885k positions) |
| keyed, c3 depth 1, held-out | between (1−TV)/(1+TV) = 0.63 and 1−TV = 0.774; a broken (fresh-key) coupling would give Σpq = 0.606 | 0.751 (97 % of the coupling bound) |
| same, train rows | — | 0.788 (overfit gap 3.7 points, from 576 prompts) |
| one-step-stale policy, per token | 1 − 1.19·TV, TV 3.71e-4 → 0.99956 | 0.99956 |
| keyed token-wise recheck, whole rows | E exp(−1.19 Σ TV) = 0.092 | 0.120 (Jensen, expected sign) |
| BV, whole rows | E exp(S_L − max S) from the measured lp/lq → 0.892 | 85-89 % |
| lr 0 | 0 forks | 0 forks |

- The draft head's 0.75 is drafter quality on high-entropy text: held-out TV 0.23, KL 0.45 nats/token. 56 % of the loss
  is at H ≥ 1, and the head misses 7 % of near-deterministic tokens (a capacity signal).
- The ~10 % keyed-whole rate is the law exp(−Σ TV) over ~6.7k-token rows. Keyed forks are linear in L·TV. BV's
  whole-row rejection is a drawdown of a √L-scaled walk, which is why BV accepts 85-89 % whole.
- Super's 0.895 came from much easier text, so it is not a Nano target.
- Side check: on the non-BI harness two forwards of a row are not bitwise. With the BI recipe on, the learner forward is
  bitwise run to run (det_eq 1.0000).

### tree_spec production readiness

- The module is free of the old MTD code and runs on Megatron-LM main.
- Lag K (≤ 4) is harness-exact:
  - persist wakes and exact-SPR cuts at K = 1 / 2 / 4: EXACT;
  - a real weight change at each wake, bitwise vs plain decode with the same wakes: EXACT at K = 2 / 4;
  - tree_straddle_commit and tree_stale_version negatives fail.
- Unit tests: 323 + 108 pass. The engine test is EXACT at natural and forced width, graphed and eager. All 8 tree
  negatives fail.
- In-loop on the main pin (Nano 8k, 16 ranks, buffer 20): EXACT steps 0-3 (3.3-3.4 M tokens/step), 1.53 tokens/iteration.
- Three bugs were found and fixed on the way (see Known bugs).
- Open:
  - at buffer 30 the colocated loop OOMs at the TMS resume (140 GB resident). Tree state is < 0.5 GB/rank, so a config
    memory limit is suspected;
  - the port to the production branch is pending.

## 5. DX: exact in the non-colocated async setting

Stock NeMo-RL async reference, Nano 8k, 4 nodes, routers frozen:

| stock async lag | s/step |
|---|---|
| K1 | 56.2 |
| K2 | **40.5** (steps 3-12, 90 % CI 38.0-43.0; train 35.0, exposed generation 3.5) |
| K4 | 37.0 |

- GRPO-Z (the sibling-draft LOO-c baseline, train-on-arrival) is unbiased. Exact enumeration matches on-policy LOO-c
  within 2e-6. Negatives (own draft in the baseline; scale from current rows) fail as required. 16/16 unit tests pass.
- The 1-node gate is EXACT on law, mismatch, sentinel and trained-once checks. The drop-late-group negative fails.
- Engine-side ⊥-BV verify: bitwise (73,728 / 73,728 tokens exact, max |dlogp| 0).
- Fork-row latency per token under load: ~4.3 ms (fork rows on a lightly loaded group, refs only), ~3.8 ms with MTP,
  against the ≤ 4.2 ms needed. A group that keeps 64+ draft rows/rank sits at 1.2-1.5×.
- A 1-GPU (EP1) fork lane is rejected: t(16 rows) is 9.6 ms vs 6.9 on EP4.
- Revised projection at Nano 4 nodes: **1.10-1.30× async-2** (capacity-limited).
- Open: a lag-2 hang at step 3 (tree + MTP on the non-colocated stage), root cause unknown. Runs now abort on stalls.

## 6. F3: wider cross-fade band

Chosen Nano band **cfS30**: m = 0.03 × (0.3 / 1.0 / 0.5) on MoE layers 0-5 / 6-17 / 18-22, extra 4. It applies to every
arm, XS and SPR-BI included. Stock has no cross-fade code and stays plain top-k, so the method overhead stays measured
against SPR-BI.

| quantity | result |
|---|---|
| TV / token vs the previous band (m 0.01, extra 8) | **×0.89-0.95** in all three runs (predicted ×0.86-0.94) |
| step cost | **zero**: IS-SPR K=1 70.2 vs 71.9 s/step; learner −0.4 to −0.8 s |
| IS-SPR group ESS (same rows) | +0.00 to +0.055 |
| memory at K=2 | S30 runs clean; the previous band OOMed at the first wake (IS-SPR residency, since fixed) |
| gates | XS positive exact (mismatch 0 of 65536), engine-band ×2 negative caught; band rule batch-invariant (row slices and permutations bitwise) on CPU and CUDA |

The uniform m 0.03 variant costs +1.8 s/step with no TV gain over S30, so it is rejected. Other models keep m 0.01 /
extra 8 until they are measured the same way.

## 7. Super-L22, 4 nodes (TP2 EP8), lag 1 [degenerate regime: plumbing and scale only]

One allocation per row:

| arm | SPR | SPR-BI | ours | c / b | c / a |
|---|---|---|---|---|---|
| wvs (previous production approach) | 54.35 | 56.90 | 85.81 | **1.51** | 1.58 |
| XPR-S, stand-down verify | 54.27 | 57.91 | 103.93 | **1.80** | 1.92 |
| VG-cap, alternative | 54.04 | 59.09 | 85.66 | **1.45** | 1.59 |
| XS, exact sync reference (pooled) | 56.16 | 57.00 | 70.94 | **1.25** | 1.26 |

Other Super-L22 measurements (vs SPR only):
- XPR (no spares): 92.67 s = 1.71× SPR. Every step pays the tail floor: a 48.5 s tail phase plus an 11.1 s verify.
  Lag-1 drafts reject early on essentially every step (strict-alternation tail floor). Under strict alternation, plain
  XS (1.31× SPR) beats speculative XPR.
- XPR-S at M 1.25 with verify at the end of the phase: 1.90×. The rejected rows' re-decode becomes its own serial,
  latency-bound phase. Verify-at-the-wake was introduced to fix this.
- First wake_verify pair: 2.18×. Continued rows were starved by the next batch's drafts (12-17 ms/token).
- VG uncapped: 1.66×, because 154 rows/rank sit on the steep part of the BI decode curve. Capping rows in flight at
  SPR's count gives 1.51-1.52×.

## 8. Ultra-L27, 16 nodes (TP8 EP32, MBS 1 both arms) [degenerate regime]

| arm | s/step | split / note |
|---|---|---|
| SPR K1 | 65.40 | generation 31.9, training 30.3, weight sync 1.6; routers frozen 8/8 |
| XS | 102.51 → **1.57× SPR** | generation 67.8, training 31.4; sentinel 8/8, routers frozen 8/8 |
| XPR-S M 1.25, stand-down verify | 111.9 (1 step) | verify 33.9 s (936 rows = 1.83× trained rows; 23.7 s forward over 990k tokens/rank); generation excluding verify 31.7 (SPR 31.9) |
| VG-cap | 119.76 = 1.83× SPR | verify 16.7, continuation 50.1, cut / stand-down 9.6, training 33.5 |

- The first XS attempts OOMed: the expert-row buffers were sized TP× too large (the device-dispatch budget was read as
  tokens per rank). Fixed.
- Lazy ⊥-BV verify waves cut the Ultra verify from 34 s to 27 s per step (718-731 rows instead of 936).
- BI engine decode curve, colocated layout (ms/iteration):

  | rows/rank | 1 | 16 | 64 | 128 | 192 |
  |---|---|---|---|---|---|
  | ctx 4k | 4.81 | 5.47 | 5.82 | 6.67 | 7.99 |
  | ctx 8k | — | — | 6.05 | 6.92 | 11.05 |

  Decode is latency-bound: going from 64 to 128 rows/rank costs +15 %.
- int32 offsets: Ultra-L27 profiling pools pass one layer's 2^31 elements. They are safe today only through the engine's
  int64 block ids (section 15).
- The Ultra 2× gap is not closed. Ultra on the production branch with the fair 3-arm ladder is a next step.

## 9. Nano 16k (representative-regime attempt)

- At TP1 EP8 on 4 nodes no BI arm fits: training OOMs at 182-184 GiB. Stock TP1 is marginal.
- At TP2 EP8 (single replica each):

  | arm | s/step |
  |---|---|
  | SPR | 110.96 |
  | SPR-BI | 146.10 (b / a 1.317: the BI tax is larger at TP2) |
  | XS | 217.99 (c / b 1.49) |
  | wvs | 246.76 (c / b 1.69) |

- 34-36 % of rows hit the 16k cap. At 8k, 57-60 % of Nano rows hit the cap: Nano 8k is not a natural-EOS regime, and
  SPR hides only the ~40 % of rows that end early.
- 16k on the production branch is a next step.

## 10. Exactness gates

Every gate has a positive arm that must be exact and a negative control that must fail.

| method | positive | negative (must fail) |
|---|---|---|
| XS (Super-L8) | bitwise 4/4 | skip-reshard: fails (only 33 of 65,536 tokens bitwise exact; 61-65k of 65,536 inexact in the 4-GPU gate) |
| XPR (stress lr) | exact 6/6 (mismatch 0, law ok, sentinel 0) | carry_unverified: fails (law p 3.5e-243) |
| XPR-S (stress lr) | exact 6/6, 2 spare groups dropped per step | spare_by_logp: fails (first-token law p 9.3e-9 / 1.5e-10). spare_by_content was vacuous at stress lr (every row continues from τ < 128) |
| wake_verify / wvs | exact 6/6 | carry_unverified: fails (sentinel 65,536 / 65,536 inexact, law p 2.7e-89) |
| VG | exact 6/6 | vg_by_logp: fails (p 1.4e-9); carry_unverified: fails |
| AX (alternating exact SPR) | exact | **measured limit:** at these sizes the gate has no power for ax_drop_continued, ax_m_now or ax_m_current |
| keyed exact SPR (Super-L8) | exact at K 1 / 2, lr 0 to 1e-4, MAXLEN 512-2048 | skip_check, skip_finished, fresh_keys, keep_lp, new_only, no_wake_cut, state_prefix_end: fail |
| IS-SPR | audit exact 6/6, 128 rescores/step | isspr_stale_kv: caught (sentinel inexact on every token); isspr_stale_cache, isspr_no_prefill_cut: caught |
| IS-RLOO | unit, row unit every step at K 2, positive, lr 0: pass | negkv, negcache: caught |
| EX-T / tree_spec | in-loop CHECKS exact (3.3-3.6 M tokens/step) | tree_accept_all and the 8 tree negatives: fail |
| XS-SD | T1 vs XS bitwise 1024/1024 | no_ssm_rewind, no_conv_rewind, tree_stale_mamba: fail |
| BX-SD | exact 5/5 | tree_accept_all: fails |
| DX (GRPO-Z) | exact on law / mismatch / sentinel / trained-once | drop-late-group: fails |
| F3 band | XS exact under every band | engine band ×2: caught |
| live-only KV offload | positive unchanged | drop-one-block: fails |
| device-dispatch live-row sizing | forward and both grads bitwise vs the C-sized workspace; a forced overflow is flagged on every rank; compute-sanitizer clean | isspr_stale_kv: fails |

The production branch passes XS positive / negative, wvs positive / negative, and the device-dispatch units (15 passed).
The wake-order regression test passes 6/6 after the fix and fails both slow-rank cases before it.

## 11. BI learner: the 51 vs 35 s gap is layout/accumulator, not kernels

- In the colocated loop at the same layout (TP1 EP8, MBS 1), the BI learner trains at 0.93× stock per token.
- The non-colocated "51 vs 35 s" compares different setups:
  - BI side: EP8, MBS 1, with train_overlap and the int64 order-invariant gradient accumulator;
  - stock side: EP4, MBS 2.
  It is a layout plus accumulator difference, not BI kernels.
- The int64 accumulator is 8 B per local grad element (43.7 GiB/rank at Nano EP8). At EP4 it adds ~41.7 GiB over stock
  and OOMs; at Ultra it does not fit at all.
- Fix (approved in principle): accumulate in place in the fp32 grad buffers, as stock does. int64 stays opt-in for
  negative controls. This follows the rule that gradients need accuracy, not determinism.
- The colocated production branch has neither train_overlap nor the int64 accumulator.

## 12. Engine memory

- **KV offload penalizes stock.** With KV offload + TMS on every arm: SPR 61.60, SPR-BI 62.61, XS 88.30. With KV persist
  (no TMS): SPR 56.89, SPR-BI 62.67, XS 88.30. Offload costs stock ~+4.7 s/step and our arms ~0, so the headline uses KV
  persist on every arm (stock's best).
- **Live-only KV offload**: the suspend saves only live KV blocks and Mamba slots (192-2212 MiB per suspend instead of
  the whole region). Same-job A/B on SPR-BI: 62.72 → 59.27 s/step (−3.45). The positive gate is unchanged and the
  drop-one negative fails.
- **Shared engine row workspace.** The fp16 MoE engine path sized its fc1/fc2 outputs per call inside every capture
  pool (20.7 + 7.1 GiB at Nano 16k). It now uses one worst-case workspace reserved at init and released during training
  with TMS:

  | config | workspace per rank |
  |---|---|
  | Nano 8k TP1 | 15.5 GiB |
  | Nano 16k TP1 / TP2 | 27.85 / 13.94 GiB |
  | Super-L22 | 19.6 GiB |
  | Ultra-L27 | 20.3 GiB |

  Engine graph pools went from 21.9-28.1 GB reserved to 1.9 GB.
- **Learner device dispatch sized by live rows** under sparse cross-fade:

  | config | before | after |
  |---|---|---|
  | Nano 16k | 8.62 GiB | 4.92 GiB |
  | Ultra-L27 | 1.58 GiB | 1.30 GiB |

- **Resident learner state.** The ThunderSync worker kept grad buffers resident (~22 GB) in every mode, and IS-SPR
  OOMed at K = 2 / 4. They are now resident only in the modes that need them.
- **Allocator.** Expandable segments on the colocated learner more than doubled training time (48.9 vs 21.6 s), so they
  are off.

## 13. Negative results (ruled out by measurement)

| idea | result |
|---|---|
| **Ordered verification (F1) and the learner prefix check (F2)** on BV designs | Gain on EX-T ≈ 0 s (bound ≤ 0.5 s) against 7 s predicted. The BV cut is decided by the row's suffix: the critical row was never speculated as a hit in 0 of 10 steps. Prefix coverage is 4-21 %. |
| **Keyed exact SPR** (token-wise recheck at the wake) | 105.8 s = **1.80× SPR-BI** with cross-fade (1.83× without). Keyed matching is a token-wise coupling, so P(row forks) is linear in L: 417-473 of 512 rows fork and ~2.2-2.6 M tokens are re-decoded per step. |
| **Spare drafts** (wvs with 14 drafts/prompt, 8 spares) | Nano 8k: 116.7 s = 1.99× SPR-BI (2.10× SPR). Without spares (wvs0, M 1): 1.90×. The exposed tail is an early rejection (3-15 rows per step at τ < 16) forcing a full-length re-decode, not a volume cost. |
| **B' engine-prefill verify for EX-T** | 107.0 vs 100.3 s/step (unpaired). The engine prefill is 1.3× faster than the learner verify, not the 4× seen on the Super mixed bench, and the held learner decide cancels the gain. |
| **Carried π_{n−1} rows as drafts in XS-SD** | Derived from measured parts: keyed coupling forks 83-92 % of rows and re-decodes ~70 % of tokens. BV needs a carried window, which turns it into BX. Not built. |
| **Momentum-lookahead drafts** | CPU, real fp32 checkpoints: the 1-step update variance falls 2.7-5×, giving TV ×0.43-0.60 (derived). Value ≤ 0.03-0.08 at λ = 1 and 0 when the bulk sets the phase. Colocated it needs a second weight set. Dropped. |
| AX (alternating exact SPR, fixed batch) | 1.59-1.89× SPR-BI. Each continued row is a ~50 s serial tail, and the M rule drifted to 3.0. |
| XS-SD on Megatron's 1+K path with lookup drafts | 1.98× SPR-BI (section 4). |
| 1-GPU (EP1) fork lane | t(16 rows) 9.6 ms vs 6.9 ms on EP4. Rejected. |
| uniform cross-fade m 0.03 | +1.8 s/step, no TV gain over S30. |

## 14. The post-wake illegal memory access: root cause and fix

**Symptom.** A CUDA illegal memory access in engine decode, ~0.1-0.6 s after "dynamic engine resumed". It hit SPR-BI
and IS-SPR at lag 1 and 4, intermittently, with crashing ranks in same-node groups. It looked like a kernel bug and
blocked several lag-sweep and F3 runs.

**Root cause.** A cross-rank race in the colocated wake (`MegatronGenerationMixin._wake` in
`nemo_rl/models/generation/megatron/megatron_worker.py`).
- Rank 0's coordinator command (resume_engines, then unpause_engines) resumes **every** rank's engine.
- On the old code rank 0 sent it as soon as **its own** local wake finished. The default-group barrier came afterwards.
- A slower rank's engine thread therefore resumed and stepped while that rank's main thread was still onloading or
  resharding the inference weights, moving the model, or remapping engine scratch.
- With TMS that scratch was still unmapped, so the first expert forward faulted. cuda-gdb on two coredumps showed a warp
  MMU fault in the fp16 MoE count kernel at the base of the engine row workspace, the first allocation in the TMS
  scratch region.
- Without TMS the same race is silent: the engine forwards while the reshard is still writing that rank's weights.
- The same misordering also produced an NCCL deadlock: the resumed engine's EP collectives interleaved with the barrier.

**Causality.**
- Injecting a 3 s delay on one rank at the start of `_wake` produced the IMA deterministically on exactly that rank on
  the old code.
- With the same injection on the fixed code: 4/4 steps, 0 IMA, 64/64 resumes.
- The same-job IS-SPR sets after the fix: 6 arms × 16 wakes, 0 IMA.

**Fix.** Order the wake as pause → local wake → default-group barrier → rank 0's coordinator resume.
- Regression test: `tests/unit/models/generation/test_megatron_colocated_wake_order.py`. It needs no GPU. Two ranks
  run the real `_wake` / `_sleep` / `prepare_for_generation` against a fake coordinator, with one rank delayed.
- Results: 6/6 pass on the fixed code; both slow-rank cases fail on the old order.

**Upstream.** Upstream NeMo-RL main is affected (issue #2355: `_wake` issues the barrier after
`resume_engines` / `unpause_engines`). Stock SPR baselines run on stock plus this reliability-only fix.

## 15. Pool-size int32 offset audit

- **Tree bug (fixed).** `tree_kv._layer_gather_kernel` computed `layer * s_layer` in int32. Past 2^31 elements
  (Nano pools ≥ ~6.5k blocks, about buffer ≥ 25 GiB on 1 GPU) the product wrapped. Node rows then read stale K/V: tokens
  stayed identical, but log-probs were bitwise wrong from the first node token.
- Every tree exactness claim made before the fix holds only for pools under that size (buffer ≤ 20).
- The fix computes the offset on the host as a Python int, so Triton passes i64. The pool sweep (833-16,384 blocks)
  passes after the fix; every pool ≥ 6,977 blocks fails before it. A regression test covers a pool past 2^31.
- **Static audit of every kernel that indexes a pool:**

  | family | status |
  |---|---|
  | whole KV pool (tree_kv, live KV offload) | int64 |
  | one layer's KV slice (fused KV append, FA4 paged attention) | safe today because the engine's block ids are int64. Verified bitwise at 62,975 blocks (4.1e9 elements per layer) |
  | Mamba slot kernels in Megatron-LM (selective_state_update, causal_conv1d update/varlen, BI masked row update) | int32: latent at MR ≥ 4096 (Nano), 2048 (Super), 8192 (Ultra TP8) |
  | TP/EP symmetric staging (engine symmetric TP, EP-invariant combine reduce-scatter) | int32: latent at ≥ 262k local tokens (Ultra) |
  | MoE rows, verify/BV/rescore logits | already int64 |

- **pool-i64 fix.** Every latent int32 site is widened at the slot / row / block offset only. Megatron-LM tests
  (`test_pool_int64_offsets.py`) pass 4/4 on the fix and fail 4/4 on the old pin. Outputs are bitwise identical and
  timings within noise.
- A startup validator (`index_width.py`) refuses any pool wider than the loaded kernels index.
- No current headline arm reaches an overflow size: Nano 8k at buffer 13 is ≤ 0.27 of 2^31 per layer.

## 16. Known bugs

Each entry says whether upstream main (NVIDIA-NeMo/RL, Megatron-LM) is affected.

| bug | status | upstream main |
|---|---|---|
| Colocated wake race: rank 0 resumes all engines before slower ranks finish resharding / remapping scratch. Causes the IMA and NCCL deadlock (section 14). An earlier entry recorded the IMA as "open, not explained by barrier order"; the root-cause work above superseded it. | fixed, regression test | **yes** (NeMo-RL #2355) |
| tree_kv int32 layer-offset wrap past 2^31 elements: stale K/V in node rows, non-bitwise log-probs at large pools | fixed, regression test | no (our module) |
| Latent int32 sites in Megatron-LM Mamba slot kernels, fused KV append and BI-MoE ordered reduce-scatter | fixed on the pool-i64 branches with tests; not yet on the production pins | yes, latent (not hit at current sizes) |
| FA4 paged attention, TMA path (page 128): V rows past seqused_k in the last page are multiplied by P = 0, so NaN/inf garbage there makes the output non-finite. The cp.async path (page 256) zero-fills. | kernel feedback filed | Megatron-LM main is guarded (it zeroes newly acquired V pages); raw FA4 is not |
| Exact-SPR / IS-SPR check cache truncated `precomputed_block_hashes`, giving an IndexError in MCore's prefix registration on multi-block prompts | fixed (private hashes per block), regression test | no |
| IS-RLOO config: the flat `normalize_rewards: false` never reached the advantage estimator (it reads only `grpo.adv_estimator`), so std would have stayed on | fixed; validator reads only the block | no (our config/validator) |
| IS-SPR stage built from the thundersync block alone: lag and max_new_tokens resolved to 0 | fixed | no |
| ThunderSync worker kept learner grad buffers resident in every mode (OOM at the K = 2 / 4 wake) | fixed (resident only where the mode sets it) | no |
| `clipped_weights` mapped s = NaN to the clip value (Python min returns its first argument on NaN), hiding a non-finite log-prob | fixed (raises), row dumps for replay | no |
| One IS-SPR run hit a non-finite engine log-prob at step 6 (1 in ~36 cross-fade Nano steps) | **root cause open.** Ruled out: fp16 overflow at typical activations (46-55× headroom measured), the wake race, TMS/stale KV, the FA4 page tail. Guards now raise and dump the row | unknown |
| tree_spec on Megatron-LM main: finished rows are finalized inside the controller, so a tail finishing on a multi-token accept kept extended log-probs (3997 vs 3993) | fixed, regression test | no (MCore main is correct) |
| tree_spec lag K: the epoch re-sync ran on the no-overlap first step after a resume, one position ahead | fixed (keep resident tails), regression test | no |
| tree_spec: a forward laid out before a resume and sampled after it committed old-weight node tokens | fixed (root-only commit), negative control | no |
| tree_spec: node rows count against max_requests, so at MR = live rows the tree lays out no nodes | counters and warning added; arms use MR above live rows | no |
| tree_spec lazy MTP graph capture raced another engine thread's synchronize | fixed (thread-local capture mode) | no |
| BX-SD lane placement: misclassified forks, then exceeded max_requests | fixed twice; third pair pending | no |
| Stale main-pin unit tests (fused decode / decode / parity / wide key) on the production branch: they reference Megatron's pre-z-free Mamba path | test-only fix; decode suite 4 and fused suite 16 failures remain to re-test | no |
| SPR-BI lag-4 stall at step 3 with 205 rollouts in flight, after all engines paused cleanly (driver side) | not investigated | unknown |
| DX lag-2 hang at step 3 (non-colocated stage with tree + MTP) | open | unknown |
| tree loop at buffer 30: OOM at the TMS resume (140 GB resident) | open, config memory suspected | unknown |

## 17. Pending measurements

1. The XS / SPR-BI / SPR headline re-measured after the wake-barrier fix. Then lag K = 2 / 4 same-job sets for every
   method.
2. Stock vs stock + deferred metrics: adopt the faster stock config, then re-score b / a.
3. IS-RLOO with s_r from the learner forward (predictions in section 3).
4. The XS-SD + MTP (c3) fair 4-arm set (prediction 1.17-1.38× SPR-BI).
5. BX-SD pair 3, and the powered K = 2 / 4 wake_verify gates.
6. 16k Nano and Ultra on the production branch with the fair 3-arm ladder: preflight, `pair_check` + `levers_audit`, a
   1-node gate, then 4 nodes.
