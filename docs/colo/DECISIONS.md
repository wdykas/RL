# Colocated exact on-policy RL: decisions in force and run rules (Nov 2026)

These are the rules every run follows and the design decisions currently in force. Each comes with one line of
rationale. The numbers behind them are in `docs/colo/RESULTS.md`, the theory in `docs/colo/THEORY.md`.

The run kit lives in `research/thundersync_rl/scripts/colo/`:
- `submit.sh`, `setup.sh`, `env.sh`;
- `lib/presets.sh`: per-model shapes, bands and engine settings;
- `score/score_pair.py`: scoring;
- `lib/check_run.py`: the per-arm router / sentinel check. It fails an arm with exit 5.

## 1. Goal

- **Exact on-policy, correct, and at speed comparable to stock.** The target is production speculative exact colocated
  RL at ~1.2× stock colocated partial rollout, including at Ultra scale. Rationale: exactness is the method's value;
  it only matters if the cost over stock is small.
- **Speed, not convergence, for now.** Quality and convergence studies are parked. Rationale: an exact method has the
  on-policy law by construction, so speed and correctness are what is open.
- **Everything must scale to ~500B MoE multi-node (TP/PP/EP).** Every design states its per-rank costs. Rationale: a
  lever that only works at Nano is not a result.

## 2. Fair-comparison protocol

- **Three arms in one allocation:** a = stock SPR, b = SPR-BI (stock loop on our engine and numerics, engine settings
  identical to c), c = ours. Rationale: separates the method overhead (c / b) from the BI-numerics tax (b / a).
- **Same job, ×2, with the arm order swapped, then pooled.** Rationale: removes allocation and node variance (cross-job
  spread is several seconds per step).
- **Whole-cycle windows over steady steps** (step 3 or later; for period K + 1 alternation, whole periods only).
  Rationale: stock-style arms alternate long and short steps, and a partial cycle biases the mean.
- **Report c / b, b / a, c / a, plus a kernel-normalized column** (f_gen 1.209, f_train 1). Rationale: kernel speed is
  counted neither for nor against the method; BI kernel speed-ups are a separate stream.
- **The stock baseline runs its best config.** Example: stock runs without torch_memory_saver and with KV persist,
  because both measured faster for stock. Rationale: ratios against a weakened baseline are worthless.
- **Generic levers go to stock too; BI-only levers stay BI.** Rationale: a lever that would speed up stock is not part
  of the method's advantage.
- **Diff effective configs and justify every difference.** `pair_check` must show METHOD keys only between b and c, and
  every a-vs-b difference is listed with its effect. `levers_audit` runs in preflight. Rationale: hidden config drift
  has invalidated runs before (cross-fade off in early arms; KV offload penalizing stock).
- **Every arm carries the full approved lever set** (cross-fade band, engine memory levers, and so on). The levers
  audit enforces it. Rationale: an arm missing a lever measures the omission, not the method.
- **Representative regime for headlines.** Real text, long sequences, inference-bound, natural lengths. Layer-truncated
  Super/Ultra (noise text, every row at the cap, no tail) are for plumbing and scale only. Rationale: partial rollout
  only pays where there is a length tail to hide.
- **Lag K = 1..4 for every method.** Design, gate and report at each K, against SPR-BI and stock at the same K.
  Rationale: production async/partial-rollout systems run up to lag 4.
- **The non-colocated baseline is stock NeMo-RL async** (non-colocated, in_order sampler, lag 1 / 2 / 4 / 8). It is
  not our own lane implementation. Rationale: the comparison must be against what users actually run.
- **Max 4 nodes per job for development.** Ultra profiling is allowed at ≤ 32 nodes and ≤ 1 h. Rationale: cluster
  budget; pipeclean everything below 10 nodes first.

## 3. Exactness rules

- **Forward log-probs are bitwise equal between engine and learner (batch-invariant numerics).** Rationale: the
  accept/reject is exact only if p and q are computed bitwise identically.
- **The train-log-prob sentinel is on in every run**, without CHECKS: training log-prob == sampling log-prob on every
  token, every step, `inexact = 0`. Any non-zero value is a bug. Rationale: a free, always-on exactness witness.
- **Exactness gates have negative controls that must fail.** Gates are bitwise for tokens and log-probs, plus a law
  test. A gate whose negative passes, or is vacuous (nothing to bias), does not count. Rationale: a positive alone
  cannot distinguish "exact" from "the check has no power".
- **Gradients need accuracy, not determinism.** Bitwise determinism is required for the forward log-probs and the
  sampling, not for the backward. Rationale: the method's exactness is about the sample law; deterministic backward
  costs memory (the int64 accumulator) and speed for no exactness gain.
- **Routers frozen: router weights AND expert bias, in every run, baselines included.** Check the effective config and
  the per-step router/bias probe before trusting or submitting. Rationale: the method only needs to work with frozen
  routing, and an arm with live routing is not comparable.
- **Cross-fade routing on in every arm** that verifies or rescores, with the F3 band (Nano: cfS30) applied to XS and
  SPR-BI too. Rationale: without it Nano whole-row acceptance drops from 85-89 % to 43-47 %. Applying the band to every
  arm keeps the numerics comparable.

## 4. Engineering rules

- **Predict, then measure.** Write the per-phase prediction down before a run. A miss of more than ~10 % starts a bug
  hunt. Rationale: most large misses so far were bugs (lane placement, wake race, tail admission).
- **Structural changes only.** Every change must be principled, general across scales, and clean; no knob tuning or
  per-model hacks. Rationale: knob wins at Nano do not transfer to Ultra.
- **No allocator flushes or manual allocator control.** No `empty_cache` fractions, no `malloc_trim`. Remove peaks and
  churn at the source. Rationale: flushes hide memory bugs and cost time at scale.
- **CUDA graphs for the engine.** Inference is graphed, including speculative / tree stages (0 graph fallbacks is part
  of the gate). Learner graphs are optional. Rationale: decode is latency-bound and eager launches dominate small
  steps.
- **No small-model tuning.** Do not tune dev-model knobs (for example the KV buffer size) for speed. Rationale: focus on
  numerics and software readiness.
- **Every bug gets a regression test.** It also gets an entry recording whether upstream main is affected. Bugs that
  affect upstream get a minimal upstream fix. Rationale: stability; several bugs were rediscovered before this rule.
- **Keep the GPU-idle watchdog in every job and write nothing to home directories.** Rationale: the cluster reaps jobs
  whose GPUs look idle for ~30 min, and home quotas are small.
- **Report only verified results.** Mark provisional and speed-only numbers as such. Rationale: unverified numbers
  have had to be retracted before.

## 5. Design decisions in force

- **XS (exact colocated sync) is the exact reference.** Rationale: under strict colocated alternation it beat the
  speculative XPR (Super 1.31× vs 1.71× SPR), because lag-1 drafts reject early every step (tail floor).
- **The exact track continues as XS-SD (exact sync + keyed tree speculative decode) and BX (block verification at the
  wake).** Rationale: these attack the tail floor directly; the shared `tree_spec` module serves both.
- **IS-SPR is the near-on-policy production arm.** It uses the stock pause/resume plus a keyed rescore at the wake.
  - **Approved estimators:** group weights at K = 1; per-row weights + RLOO (no std) + constant L_c at K ≥ 2.
    Rationale: unbiased by exact enumeration; row weights have ESS ≈ 0.97 vs group ≈ 0.80; std and 1/l normalisers
    are biased under row weights.
- **Next IS-SPR step: take s_r from the learner's first BI training forward** and re-prefill only continuing rows.
  Rationale: ~80 % of the wake re-prefill belongs to rows that had already finished.
- **GRPO-Z is approved for DX** (sibling-draft LOO-c baseline, train-on-arrival). Rationale: unbiased (exact
  enumeration); it is bitwise stock GRPO when every draft is accepted.
- **DX engine verify runs as waves of the keyed check.** Rationale: within 0.02-0.03 T of position-major ordering with
  no scheduler change.
- **Fixed G.** Rationale: keeps the GRPO estimator and the batch shape stock.
- **The F3 band (cfS30) is adopted for Nano on every arm.** Rationale: TV ×0.89-0.95 at zero step cost.
- **No MTD and no tail MTD on the production branch.** Rationale: the tail is attacked by tree speculation and a
  learned MTP drafter instead.
- **MTP arms use the c3 draft head.** Rationale: per-depth acceptance 0.70 / 0.66 / 0.67, compared with 0.29 / 0.15 at
  depths 2 / 3 for the previous head.
- **KV persist for the headline; live-only KV offload where memory needs it.** Rationale: full KV offload costs stock
  +4.7 s/step and ours ~0, so it is unfair to stock.
- **The BI learner accumulates in place in fp32 grad buffers** (as stock does). int64 order-invariant accumulation is
  opt-in for negative controls only. Rationale: the int64 accumulator does not fit at EP4 or at Ultra, and gradients
  need accuracy, not determinism.
- **The wake runs pause → local wake → barrier → coordinator resume.** Rationale: the old order raced slower ranks and
  caused the post-wake IMA and an NCCL deadlock.
- **Kernel speed is never a blocker.** Faster BI kernels arrive later from main. Report kernel-normalized numbers.
  Rationale: the method's cost must be separable from kernel maturity.

## 6. Ruled out by measurement

Each of these was measured (or derived from measured parts) and dropped. Do not rebuild them without new evidence.

| idea | why it is out |
|---|---|
| Spare drafts (XPR-S / wvs spares) | Nano 8k wvs with spares 1.99× SPR-BI; the exposed tail is an early-rejection re-decode, which spares cannot hide |
| Keyed exact SPR (token-wise recheck at the wake) | 1.80× SPR-BI: keyed forks grow linearly in L, and most rows re-decode |
| Ordered verification (F1) and the learner prefix check (F2) on BV designs | ≈ 0 s gain vs 7 s predicted: the BV cut is decided by the suffix, and the critical row was found in 0 of 10 steps |
| B' engine-prefill verify for EX-T | net slower (107.0 vs 100.3 s/step): only 1.3× faster prefill, cancelled by the held learner decide |
| Carried π_{n−1} rows as drafts in XS-SD | keyed coupling forks 83-92 % of rows; BV needs a carry window, which is BX |
| Momentum-lookahead drafts | ≤ 0.03-0.08 ratio gain at best, needs a second engine weight set when colocated |
| 1-GPU (EP1) fork lane | slower per row than EP4 (9.6 vs 6.9 ms at 16 rows) |
| Uniform cross-fade m 0.03 | +1.8 s/step with no TV gain over the S30 band |
| Expandable segments on the colocated learner | training 48.9 vs 21.6 s |
