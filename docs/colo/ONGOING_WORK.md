# Ongoing work: every in-flight stream

Start with [`HANDOFF.md`](HANDOFF.md) (production branch, environment, how to run, headline numbers). This file covers
what is still in flight. Every branch named here is on the wdykas/RL fork (NeMo-RL) or the wdykas/Megatron-LM fork,
and each is one squashed commit (or a short stack) with a descriptive message. Numbers are inlined; the full tables
are in [`RESULTS.md`](RESULTS.md).

## 0. Branch map

Two code lineages exist:
- **slim line**: `colo-handoff` (production) and the branches built on it. Megatron-LM / Bridge main pins, no
  Megatron patches. Run with the kit in `research/thundersync_rl/scripts/colo/`.
- **old-pin line**: `handoff/oldpin-base` (the earlier, larger research tree with tree speculation, MTP drafting and
  the non-colocated "nolanes" mode) and the branches built on it. It pairs with Megatron-LM `handoff/oldpin-pin`
  (= `6a3660905` + NVIDIA/Megatron-LM#8385 cherry-pick + its tests) and Megatron-Bridge `1f8873bb`. Point the kit at
  it with `COLO_RL=<old-pin checkout> COLO_MLM=<Megatron-LM handoff/oldpin-pin> COLO_MBRIDGE_TREE=<Bridge @ 1f8873bb>`
  (untested with this kit; the streams were run with equivalent launchers).
  The old-pin line's own launchers (`research/thundersync_rl/scripts/{ab,nl,colo}/`) expect a research workspace
  (`TS_WORK`: models, overlays, an environment script, run dirs) that is not published; prefer the colo kit.

| branch (RL fork) | on top of | stream |
|---|---|---|
| `colo-handoff` | upstream main `ce0cb739` | A. production |
| `handoff/wakeorder` | `colo-handoff` | A. wake-order fix (next production tip) |
| `handoff/slim-mainpin-tests` | `colo-handoff` | A. stale main-pin unit tests fix |
| `handoff/stock-baseline` | upstream main `ce0cb739` | baseline a (stock colocated loop + wake-barrier fix) |
| `handoff/isspr-dd` | `colo-handoff` | B. IS-SPR / IS-RLOO, current work |
| `handoff/isspr-pc` | `colo-handoff` | B. IS-SPR with prefix caching / chunked prefill (measured lag sweep) |
| `handoff/f3-band` | `colo-handoff` | H. wider cross-fade band |
| `handoff/xssd-slim2` | `colo-handoff` | D. XS-SD on the slim line |
| `handoff/tree-prod-slim` | `colo-handoff` | C. tree speculation, slim gate integration |
| `handoff/oldpin-base` | upstream main `ce0cb739` | common base of the old-pin line |
| `handoff/tree-spec` | `handoff/oldpin-base` | C. shared tree-speculation module, MTP head input contract v2 |
| `handoff/draft-head` | `handoff/tree-spec` | F. MTP draft head training / export / evaluation code |
| `handoff/tree-prod` | `handoff/oldpin-base` | C. tree speculation, production-ready (old pin) |
| `handoff/xssd-mainpin` | `handoff/oldpin-base` | D. XS-SD on Megatron-LM main |
| `handoff/ext-tree-xssd` | `handoff/oldpin-base` | D. XS-SD, old pin |
| `handoff/bxsd2` | `handoff/oldpin-base` | E. BX-SD fork lane + lag-K wake verify |
| `handoff/dx` | `handoff/oldpin-base` | G. DX, exact non-colocated async |
| `handoff/bilearn` | `handoff/oldpin-base` | I. BI learner in-place layout, kernel notes |
| `handoff/pool-i64` | `handoff/oldpin-base` | J. int64 pool offsets (NeMo-RL side) |

| branch (Megatron-LM fork) | on top of | what |
|---|---|---|
| `handoff/stock-pin` | `6a3660905` | stock baseline's Megatron-LM: + NVIDIA/Megatron-LM#8358 (graph-owned RoPE/YaRN capture constants) |
| `handoff/oldpin-pin` | `6a3660905` | old-pin line: + #8385 (BI MoE work bounded by live rows, int64 MXFP8 row indexing) + tests |
| `handoff/pool-i64` | main `a11c84aec` | int64 slot / block offsets in the pool-indexing kernels (upstream candidate) |

`colo-handoff` itself needs no Megatron-LM patches (main `a11c84aec`, Bridge main `9981c93`).

## 1. Rules every run follows

See [`DECISIONS.md`](DECISIONS.md). In short: routers (weights and expert bias) frozen in every arm; every arm at its
best config, same job, ×2 with the order swapped; whole-cycle windows; report c/b, b/a, c/a and the kernel-normalized
column (f_gen 1.209, f_train 1); exactness gates bitwise with a negative control that must fail; prediction written
before measuring.

## 2. Streams

### A. Production (`colo-handoff`)
- **State:** gated: XS and wvs positives exact, negatives fail; the live-row device-dispatch units pass. Same-job
  IS-SPR set at Nano 8k 4 nodes K=1: **1.19× SPR-BI** (kernel-normalized 1.18). The XS set (1.39× SPR-BI) predates
  the wake-barrier fix, speed only.
- **Next production tip:** `handoff/wakeorder` (engine wake ordering fix plus its unit test). Promote it once the
  wake-order unit gate passes.
- **Queued:** stock vs stock + deferred metrics A/B (adopt the faster stock config, re-score b/a); land the F3 band;
  re-measure XS / SPR-BI / SPR on the fixed tip; then the K = 2 / 4 sets.
- **Unit tests:** about 130 fused-decode / decode / parity / wide-key tests fail on the main pin and are believed
  stale (written against the old pin). `handoff/slim-mainpin-tests` fixes most: wake-order and parity suites pass,
  the decode suite has 4 failures and the fused suite 16 left. Until those are confirmed stale rather than a code path
  the arms run, the Nano numbers are provisional.

### B. IS-SPR / IS-RLOO, the near-on-policy arm (`handoff/isspr-dd`, `handoff/isspr-pc`)
- **What:** stock pause / resume plus a keyed rescore at the wake and importance weights; unbiased, not exact in law.
  Approved weighting: group weights at K=1, per-row + RLOO (no std, constant L_c) at K≥2.
- **Measured (`handoff/isspr-pc`):** same-job lag sweep K = 1 / 2 / 4 = 1.23 / 1.44 / 1.40× SPR-BI. The estimator is
  healthy: row ESS ≈ 0.97, ε ≤ 1.2%.
- **Cost driver:** the wake re-prefill of 2-3 M tokens per step, ~80% of them in finished rows.
- **In progress (`handoff/isspr-dd`):** take the rescore s_r from the BI training forward and re-prefill only
  continuation rows (learner-forward row weights are in); non-finite-row dumps on raise; `clipped_weights` raises on
  a non-finite s.
- **Open:** a non-finite log-prob seen once at step 6 of one run; root cause unknown. Reproduce with the dump on.

### C. Tree speculation (`handoff/tree-spec`, `handoff/tree-prod`, `handoff/tree-prod-slim`)
- **What:** a shared tree-speculation module: graphed tree stage, MTP draft source, prefix caching / chunked prefill
  through a version-keyed cache.
- **State:** `handoff/tree-prod` is MTD-free, on Megatron-LM main, exact in the harness for lag K ≤ 4, and passed
  review for the old pin. `handoff/tree-prod-slim` is the gate-only integration onto the slim line.
- **Fixed:** an int32 gather-offset overflow at large pools; key reuse in normal mode.
- **Exactness harness:** tree engine harness with 8 negative controls, all red.
- **Next:** port to the slim line; explain the OOM at engine buffer 30 GiB on Megatron-LM main (140 GB resident at
  the torch_memory_saver resume).

### D. XS-SD, exact sync + speculative decode (`handoff/xssd-slim2`, `handoff/xssd-mainpin`, `handoff/ext-tree-xssd`)
- **State:** exact at engine buffer 48 on the main pin (1024/1024 tokens), and the negative fails.
- **Running:** fair 4-arm SPR / SPR-BI / SPR-SD / XS-SD with the c3 draft head on the slim line.
- **Prediction:** 1.17-1.38× SPR-BI. `handoff/ext-tree-xssd` is the old-pin line.

### E. EX-T and BX-SD fork lane (`handoff/bxsd2`; EX-T lives in `handoff/tree-spec`)
- **EX-T + MTP:** 1.37× SPR-BI with draft head r1_e3, 1.28× with c3 (control arm); c3 pairs queued.
- **BX-SD:** v1 / v2 missed (97.5 s/step) because of two bugs, both fixed; at Nano 4 nodes the capacity model caps
  the gain at ~3.5 s/step; a third iteration is queued.
- **Lag-K wake verify:** built on `handoff/bxsd2`; the powered K = 2 / 4 gates are pending.

### F. Draft heads (`handoff/draft-head`)
- **c3 head:** multi-depth Nano MTP drafter, required by the MTP arms. In-engine per-depth acceptance 0.70 / 0.66 /
  0.67; slowest row ~1.9 tokens per forward; draft cost 0.28-0.35 ms per depth.
- **Acceptance audit:** no bug; c3 sits at 97% of the 1 − TV ceiling.
- **Round 2:** wide MLP, entropy-weighted loss, multi-layer features, many more prompts. Target: slowest row ≥ 2.2.
- **Checkpoint:** not in git (~96 MB); to be published as a release asset or HF model repo. Retrain with the code in
  `research/thundersync_rl/draft_head/` on `handoff/draft-head` (it needs the head input contract v2 of
  `handoff/tree-spec`).

### G. DX, exact in the non-colocated async setting (`handoff/dx`)
- **Built:** GRPO-Z (sibling-draft leave-one-out baseline, train on arrival; unbiased) and engine-side block
  verification (bitwise).
- **Stock async reference, Nano 8k, 4 nodes:** K=1 56.2 s, K=2 40.5 s, K=4 37.0 s per step.
- **Projection:** 1.10-1.30× async-2. The fork lane gives x ≈ 4.3 ms (3.8 with MTP) against the ≤ 4.2 ms needed.
- **Next:** the planned set-D runs; root-cause the nolanes lag-2 hang at step 3.

### H. F3, wider cross-fade band (`handoff/f3-band`)
- **Change:** cross-fade routing band cfS30 (margin 0.03 × [0.3 / 1 / 0.5], 4 extra experts) on every arm.
- **Effect:** routing TV ×0.89-0.95 at zero step cost.
- **Status:** reviewed, ready to land with the matching levers-audit update.

### I. Kernels, memory and the BI learner (`handoff/bilearn`)
- **Kernel factors:** f_gen 1.209 (BI generation per token), f_train ≈ 1.
- **The 51 vs 35 s learner gap at Nano:** layout and accumulator, not kernels. The int64 order-invariant accumulator is
  replaced by fp32 (gradients need accuracy, not determinism).
- **Notes:** `docs/colo/KERNEL_NOTES.md` on that branch: per-phase factors and the open kernel asks.

### J. Upstream candidates
- The colocated wake-barrier race (NeMo-RL #2355).
- Megatron-LM int32 latent offsets at large pools (`handoff/pool-i64` on the Megatron-LM fork; NeMo-RL side
  `handoff/pool-i64`).
- An FA4 page-128 V-tail NaN (external, reported upstream).
- Prefix-cache staleness across a weight refit.

## 3. Decisions in force

See [`DECISIONS.md`](DECISIONS.md). Ruled out by measurement: spare drafts, keyed exact SPR (1.8× SPR-BI), ordered
verification (F1 / F2), B' engine verify for EX-T, carried rows as drafts in XS-SD, momentum-lookahead drafts.

## 4. First day
1. Read `HANDOFF.md`, this file, `RESULTS.md`, then `THEORY.md`.
2. Build overlays, run `setup.sh`, then `submit.sh smoke`; it must end with PASS.
3. Run `submit.sh gate wvs` and `submit.sh gate xs`; positives exact, negatives fail.
4. Pick a stream, check out its branch, and follow its "Next" line. Write the prediction before you measure.
