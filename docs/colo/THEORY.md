# Exact on-policy colocated RL with speculation: theory and method summary

This is the condensed theory behind the colocated arms in `research/thundersync_rl/` and the batch-invariant (BI)
model hooks in `nemo_rl/models/megatron/`. It explains what each method does, why it is exact (or only unbiased),
and what bounds its speed. Measured results are in `docs/colo/RESULTS.md`, decisions in `docs/colo/DECISIONS.md`,
the hand-off state in `docs/colo/HANDOFF.md`, and open work in `docs/colo/ONGOING_WORK.md`.

Labels used for numbers: [M] measured, [D] derived by a model script from measured inputs, [T] proof, [H] assumption
or projection.

Model shorthand: "Nano", "Super" and "Ultra" are the hybrid Mamba-MoE policies. "Super-L22" and "Ultra-L27" are
layer-truncated development versions that run on noise text, where every row hits the length cap.

Code comments cite the original research notes by name. Their content is condensed here: `MODEL.md` (the colocated
model, XPR, wake verify) maps to sections 1-2; `TAIL_FLOOR.md` (tail floor, spares, VG, exact SPR, IS-SPR) to sections
2-3; `FIRSTPRINCIPLES_A`..`_D` to sections 4-7; `EARLY_LATE.md` to section 8; `DESIGN.md §n` (the earlier non-colocated
ThunderSync design) is summarized in `research/thundersync_rl/README.md` and section 2 ("nolanes and DX").

---

## 1. Setting and notation

**GRPO step.** Step n trains a batch B_n of P = 64 prompts × G = 8 rows, with a new-token cap L (8192 unless stated
otherwise). θ_n is the policy that trains at step n, and π_n is its sampling law. θ_{n+1} = U(θ_n, B_n) is one
optimizer step per batch. The advantage is GRPO's leave-one-out (LOO) mean and std over the G − 1 siblings.

**Colocated loop.** The same GPUs alternate: a generation phase Φ_n with θ_n resident in the inference engine, then a
stand-down, training on B_n, then a refit/wake with θ_{n+1}. This is **strict alternation**: no GPU decodes and
trains at the same time.

**Stock partial rollout (SPR).** Stock NeMo-RL colocated, `in_order` sampler at lag K (`max_lookahead_versions = K`).
In-flight requests of batches n+1..n+K are paused in the engine at the stand-down (KV persisted) and resumed after
the wake. Its per-step budget is emergent: b ≈ L/(K+1) iterations at (K+1)·512 rows in flight. The phase length
alternates long (≈ L) and short (a few seconds) because Φ_{n+1} = L − Φ_n is the neutral period-2 mode of the
recursion.

**Why SPR is off-policy.** A row trained at step n carries tokens sampled under θ_{n−K..n−1}. After the wake it also
continues on stale KV/SSM state. On the BI engine the measured per-token gen-KL of SPR is 7-9e-5. That is ~35× the
pure version drift at Super. **Most of SPR's off-policyness is stale state, not stale weights.**

**Lag K.** The number of versions a row may straddle. Version divergence composes super-linearly:
v(age K)/v(age 1) = A(K) = K + 2 Σ_{d<K} (K − d) ρ_d. Measured A(2) = 2.6-3.1, so A(K) ≈ K^1.5 in variance and
≈ K^0.75 in TV.

**Exact (in law), the definition every exact arm meets.**
- Given the prompts and the past F_n, the trained rows of step n are i.i.d. π_n within and across groups.
- Which rows train is independent of their values.
- Engine numerics: the engine is **batch-invariant (BI)**. Engine log-probs equal learner log-probs bitwise
  (requirement **P1**: prefill = decode = learner forward, including chunk boundaries, prefix-cache hits and the
  Mamba scan vs step recurrence). The train-lp sentinel checks this every step.
- Gates: the sentinel (0 inexact tokens), a law check on every trained token (first-token cumulative check and
  all-token PIT under π_n), and a negative arm that must fail.

**Weaker targets.**
- **U (unbiased):** E[ĝ | F_n] equals the on-policy GRPO expectation. Says nothing about the law of the rows.
- **U′:** E[ĝ | F_n] = Σ_x c_x ∇J_x with c_x > 0: exact direction per prompt, perturbed prompt weights.
- With Adam and grad clipping, an unbiased gradient still gives a biased update at second order. Only E reproduces
  the training trajectory law of synchronous GRPO.

**Router frozen.** Every arm and every baseline runs with router weights and expert bias frozen. Cross-fade routing
(a continuous band around the top-k boundary) is on in every speculative arm. Hard top-k flips otherwise multiply
the per-token TV by 3.7-5× and v by 20-100×.

**Common tools.**
- **Keyed sampling.** y_t = argmax_v (log p_θ(v | y_<t) + g(key_row, t, v)), with Gumbel noise from a key unique per
  (run seed, step, prompt, sample). A row is a deterministic function f(θ, keys). By the Gumbel-max trick it is an
  exact π_θ sample when the keys are independent of θ.
- **Block verification (BV).** Verifies a draft x ~ q against p at the sequence level. It runs a likelihood-ratio walk
  over the draft, accepts a prefix up to a cut τ, and emits a residual token at τ. Continuing from τ+1 under p with
  **fresh keys** gives an exact p sample (martingale lemma). Among single-draft verifiers, BV maximizes the expected
  accepted length. It needs log q of every draft token and the full q row at the cut. The q row is rebuilt from a
  logged hidden state plus the stashed LM head of that version.
- **⊥-BV.** BV with a constant envelope M ≥ 1 and a null outcome ⊥ at the root. Its output is (1/M)·p ⊕ (1 − 1/M)·⊥:
  ⊥ happens with probability exactly 1 − 1/M, independent of the output value. This is what makes ⊥-based selection
  exact.
- **Keyed acceptance of speculative tokens.** A proposed token is accepted iff it equals the keyed argmax under θ_n,
  computed by a P1-bitwise verify forward. The accepted path *is* f(θ_n, keys) whatever the proposal source. So
  speculation changes speed, never the value.

---

## 2. The methods and how they relate

All arms run on stock NeMo-RL's colocated `in_order` loop unless stated otherwise.

### XS: exact sync (reference)
- Lag 0. Wake with θ_n, decode B_n from scratch on the BI engine, train, refit. No carry, verification or spares.
- **Exact:** every token is a θ_n sample, and engine lp = learner lp bitwise.
- Cost: one serial chain of L iterations at ~26-32 rows/rank (latency-bound). Measured 1.41-1.53× SPR-BI at Nano 8k.
  SPR-BI is SPR on the same BI engine, which isolates exactness from the BI kernel tax.

### XPR: carried prefixes as drafts
- SPR's carried prefixes are drafts. Each token logs its q: engine lp, head input and version tag. The training
  cohort's prefixes are BV-verified against π_n, and rejected rows continue from τ at π_n.
- **Exact** because a carried row is one causal proposal process whose q switches with the version. A cut at a step
  boundary is "the proposal switches to p", which is a stopping time.
- **New exactness condition:** the step at which a group trains must be decided *before* that step's outputs exist,
  e.g. by admission index (age). Stock "train what completed this step" selects on length, i.e. on τ, so it is not
  exact. Deferring a group because a row was continued is the same selection.
- XPR-E verifies once at the cohort's last step. XPR-V re-verifies every step against p_{n−1}. Both have the same
  step time because the tail is max-dominated.
- **Outcome:** in-loop, the carry collapses to complete lag-1 drafts, and the exposed tail makes XPR ≈ XS (§3).

### XPR-S: spare drafts
- A ⊥-BV pool at constant M with N drafts per prompt, plus S spare prompts (e.g. M 1.5, N 16, 72 admitted for 64
  trained).
- **Selection rule:** the train set is the first P prompts, in fixed admission order, whose pools fill G non-⊥ slots.
- **Why it is exact:**
  1. Within a pool, drafts are walked in index order. Whether to look at the next draft depends only on counts
     (drafts left, slots left). So the non-⊥ outputs, taken in order, are i.i.d. π_n (optional sampling).
  2. A pool is dropped iff its ⊥ count leaves fewer than G outputs. That is a function of the ⊥ pattern alone, whose
     law (Binomial, p = 1 − 1/M) is the same for every pool.
  3. P(outputs, ⊥ pattern) = P(⊥ pattern) × ⊗ π_n (chain rule). Given the selection, the kept outputs remain i.i.d.
  4. **Any rule that reads output content breaks it.** Examples: dropping continued rows, or ranking pools by
     continued rows. These are the negative arms.
- **Outcome:** ≈ 1.23× SPR modeled at Super, only 4 % better than XS, and fragile in v (Theorem S, §3).

### ⊥-BV
The verifier behind XPR-S, VG and wvs (`null_bv.py`). The ⊥ price is set by a quantile, not by the mean (Proposition Q,
§3). Root placement of ⊥ is optimal: late-⊥ placements give 2.4-43× more tails.

### Wake verify and wvs
- **Problem.** Verifying at the stand-down leaves rejections as a separate serial continuation phase. Measured:
  1.90× SPR.
- **Reorder.** At the stand-down, the drafts of batch k+1 still decoding are **cut** where they stand: all engine
  ranks pause through the coordinator, mark the same requests and unpause. So batch k+1 is complete at that
  stand-down.
- After training, the learner holds θ_{k+1} and verifies batch k+1 with the engine still stood down. Its wake is the
  refit. Rejected rows continue from τ, and accepted unfinished rows continue from their end, inside phase k+1, beside
  the drafting of batch k+2.
- One wake per step, no separate continuation phase.
- **Exact** because the cut length of a draft is a stopping time of its own sequence. The stand-down time depends on
  the previous batch and on engine timing, never on the draft's later tokens. BV of any stopping-time prefix is exact.
- **wvs** = wake verify with spares: the XPR-S ⊥-BV pool plus wake verify (fixed G, M 1.35, N = 14, 8 spare prompts).

### VG: variable-G ⊥-BV pool
- Admit ≈ 1.4 × 64 prompts × 8 drafts. ⊥-BV-verify **every** draft at a constant M = 1.4 (walk all, no stopping rule).
- A prompt's group is all of its non-⊥ outputs, of size Bin(8, 1/M). Groups of size < 2 are dropped. GRPO's LOO
  baseline is taken over the group's own size.
- **Exact:** group sizes and drops are functions of the ⊥ pattern only. What changes is the estimator's group size,
  not the sample law. It removes fixed G's per-prompt binomial margin (1.97 → 1.40 drafts per output).
- Modeled 1.18× SPR at Super with the engine-prefill verify.

### Exact SPR (xspr)
- Keep stock SPR pause/resume. At each wake, recompute every in-flight row's KV/SSM under θ_n by one prefill. During
  that prefill, re-run the keyed sampler at every generated position from 0, and take τ = the first mismatch.
- Truncate at τ and resume with the **same** key stream. The first new token is y_τ.
- **Exact:** after the final recheck each row equals the keyed θ_n decode from scratch (pathwise), so it is an exact
  π_n sample. θ_n is independent of batch-n keys under in-order admission and a BI engine.
- **Required amendments:**
  - **D1:** check every row of the batch, including lookahead rows that already finished under older weights.
  - **D2:** replace accepted-token log-probs with θ_n's.
  - **D3:** each wake rechecks all generated positions from 0.
  - **D4:** the same key stream after τ; keys never reused across epochs.
  - **D5:** no content-dependent drops. Evictions, retries or timeouts raise.
  - **P1:** bitwise, as above.
  - **P3:** Mamba state at τ needs a second prefill to τ, or state checkpoints.
- **Outcome:** the keyed token-wise check loses Σ_t δ_t, which is linear in L. With cross-fade on, 83-92 % of rows
  fork (τ median 1.1-1.8k). Exact SPR measured 1.80× SPR-BI. **The keyed check is the right continuation sampler,
  not the right wake check.**

### EX-T: exact SPR with fork-tail speculation
- Exact SPR, plus admission of the whole next batch at the wake, plus speculation on fork rows.
- Fork rows get a W-wide token tree from their own dead suffix, their siblings and a per-step suffix index, with
  keyed acceptance.
- Steady state Φ = max(L/2, L/s): the tail stops costing at s ≥ 2 tokens per iteration on fork rows.
- Exact: speculation only changes which forwards run.

### BX / BX-SD: block verification at the wake
- Balanced K = 1 SPR (whole next batch admitted at the wake, phases ≈ L/2).
- At the wake, BV-verify every carried row against θ_n inside the engine recompute prefill, which also rebuilds θ_n
  KV/SSM. Accepted rows resume. Rejected rows take the BV residual at τ, then decode under θ_n with **fresh** keys.
- Fork rows with τ < L/2 are accelerated (n-gram tree + MTP-style draft) or packed onto a lightly loaded EP group.
  BX-SD also runs the drafter on bulk rows.
- **Exact:**
  - (a) BV output + continuation completes to π_n. This needs the draft's law to be exactly the logged q: BI engine,
    no evictions.
  - (b) The continuation is f(θ_n, K′) with fresh K′.
  - (c) Speculation changes only which forwards run.
  - (d) In-order membership, no drops.
- **Negatives:** accept on top-k match; reuse the draft's old keys after a BV rejection (BV's decision reads X_{>τ},
  a function of those keys, so it is biased); skip the KV recompute.

### XS-SD: exact sync + keyed (tree) speculative decode
- XS unchanged. Each iteration a row may carry k ≤ K proposed tokens: an MTP/draft head run keyed with the same g,
  n-gram lookup over own text and siblings, or a token tree. One verify forward over 1 + k positions accepts while the
  proposal equals y_t and emits y_t at the first mismatch.
- **Exact (pathwise):** the emitted row equals the 1-token keyed decode for any proposal sequence. Proof by induction
  on positions, given that verify logits are bitwise the 1-token decode logits.
- **Requirements:**
  - R1: verify logits bitwise equal to the decode and learner logits. The open item is BI Mamba W-token verify through
    the replay buffer, with rewind as a counter reset.
  - R2: an identical keyed transform in decode and verify.
  - R3: no content-dependent drops.
- **Why keyed drafting matters (speed, not exactness):** a drafter sampled with the same Gumbel noise matches with
  probability ≥ (1 − TV)/(1 + TV), near the maximal-coupling bound 1 − TV. Independent sample-and-match gets only
  Σ p·q.
- Test T1: spec-on rows == spec-off rows bitwise, at K ∈ {0, 1, 2, 4, 8} and for every source.

### IS-SPR
- Stock SPR (lag K, pause/resume, no recheck, truncation or re-decode). At every wake, every row that ran under older
  weights is **rescored** under θ_n by one engine prefill. That recomputes KV/SSM, so every post-wake token is an exact
  θ_n sample, and it gives log p_n of every earlier token (keyed rescore).
- **Weights.** s_r = Σ_t (log p_n − log q) over the row's tokens; post-wake terms are 0 bitwise. Group weight
  W_g = min(C, e^{Σ_{r∈g} s_r}); row weight w_r = min(C, e^{s_r}). The weight scales the advantages (detached), and
  the loss uses a constant normalizer (a sum of row terms).
- **Lemma L.** If the cut is a stopping time of the row's own stream, the row's density is Π_{t<c} q_t · Π_{t≥c} p_t.
  So E_Q[W_g φ(Y_g)] = E_{P^⊗G}[φ] for **any** group statistic, including LOO-std GRPO. A cut that looks at the
  future length is not valid.
- **Which weight.**
  - **K = 1: group weight**, which is unbiased for the nonlinear GRPO statistic.
  - **K ≥ 2: row weight (RIS) with an RLOO / LOO-c advantage.** Group ESS collapses from K = 2 (0.20-0.57, clip mass
    4-26 %), while row ESS stays ≥ 0.73 to K = 4 early in training.
  - Row weights are exactly unbiased iff the advantage is linear in the siblings' rewards with a zero-mean multiplier
    (RLOO, or LOO mean with a past-only global scale; see Theorem R).
  - In colocated SPR a row's pause point depends on its siblings' lengths. That timing coupling leaves a cross term
    even for RLOO, sized by the unit tests.
- **Clip bias:** ‖bias‖ ≤ ε_C · Σ_g sup‖ĝ_g‖, with ε_C = E_Q[(W − C)⁺]. Logged without bias as ε̂_C = mean (e^s − C)⁺,
  together with Var(s) and ESS.
- **Guarantee:** U, not exact in law. Measured cost ≈ 1.08× SPR-BI at Nano 8k K = 1.
- Without the KV recompute, s² ≈ 8 (ESS ≈ 0). The recompute is what makes IS usable.

### nolanes and DX: non-colocated async
- **nolanes:** stock NeMo-RL async (non-colocated), plus a learner-side BV verify of the mixed-version drafts and
  continued-row tails. Measured 1.51× at Super-L22 and 1.71× at Ultra-L27 against stock async. The critical path is
  the learner verify wall followed by the tail chain.
- **DX:** stock async drafts at lag K_d ≥ 3, plus position-major θ_n verification on the inference pool, two-ended with
  the learner's fused forward. BV + fresh-key continuation runs on a fork lane with tree/MTP speculation, and the
  advantage is **GRPO-Z**, so rows train as they arrive.
- **GRPO-Z (Lemma Z):** A_i = (R(Y_i) − mean_{j≠i} R(X_j)) / σ_i, where the siblings' rewards are taken from their
  *drafts* X_j, which are known before θ_n exists.
  - Since the baseline depends only on X_{−i}, and rows are independent given F_n and θ_n, E[A_i ∇log π_n(Y_i)] =
    E[1/σ]·∇J_x. That is U′ with LOO-std, and **U (exact) with a past-only scale (LOO-c)**.
  - For accepted siblings Y_j = X_j, so it differs from stock only on rejected siblings.
  - Toy: E[ĝ]/∇J matches within Monte Carlo error for LOO-c (cos 1.00000).
  - It removes the group wait. Without it, 33-90 % of groups wait on a fork row.
- **Exact rows:** BV + fresh keys for any draft law q satisfying: (i) the token was sampled from q; (ii) q(x_t) and
  q_τ(·) are logged exactly; (iii) the draft randomness is independent of θ_n. Mixed-version drafts from in-flight
  refit are fine.

### How they relate

| method | rows | weights / estimator | what it pays | where it stands |
|---|---|---|---|---|
| XS | E | stock | full L chain at low batch | reference, ~1.4-1.5× SPR-BI |
| XPR / XPR-E / XPR-V | E | stock | verify + exposed tail ≈ L | ≈ XS |
| XPR-S, wvs | E | stock, fixed G | spares (Theorem S) + verify | ≈ 1.2-1.3× SPR |
| VG | E | LOO over a variable group | spares at the floor + verify | ≈ 1.18× SPR modeled |
| exact SPR | E | stock | keyed rework (Σδ) + tail | 1.80× measured |
| EX-T / BX / BX-SD | E | stock | wake check + fork depth L/s | 1.05-1.2× when s ≥ 2 on fork rows |
| XS-SD | E | stock | L/s chain | 1.03-1.07× SPR-BI with the module at target |
| IS-SPR | U | group (K = 1) / row (K ≥ 2) | rescore prefill, ESS | ≈ 1.08× SPR-BI |
| DX | E rows | GRPO-Z (U′ / U) | depth L·t_f/s_f off the learner path | 0.9-1.07× async-2 modeled |

---

## 3. Theorem T and the tail floor

### Theorem T (final-step tail)
- Take any exact BV-family learner with a causal train step, drafts at lag ≥ 1, and rows of length g.
- Then step n must produce π_n continuations of length g − τ_r for every continued row r of the training cohort. The
  latency is max_r (g − τ_r).
- With 512 rows and P(continue) = 9-22 %, some τ lands in the first 1-2 % of a row on almost every step
  (P(τ < g/8 | continued) ≈ 0.3). So the overflow is ≈ g iterations at small batch.
- The lookahead batches finish inside that phase, so the pool collapses to ~512 rows plus tails, i.e. **exact partial
  rollout with exposed tails ≈ exact sync**.

### Tail floor: Theorem F (single draft)
Suppose each output slot r has one pre-phase draft X_r, of any law and version mix, and Y_r ~ π_n comes from any exact
mechanism: any verifier, order, envelope or cross-row dependence. Then, for every j:
1. P(LCP_r < j) ≥ TV_j(r) := TV(p_{≤j}, q_{r,≤j}).
2. Row-independent mechanisms: P(∃ r: LCP_r < j) ≥ 1 − Π_r (1 − TV_j(r)).
3. Any joint mechanism: P(∃ r: LCP_r < j) ≥ TV(⊗ p_{≤j}, ⊗ q_{r,≤j}).
4. E[excess] ≥ Σ_{j≤J} P(∃ r: LCP_r < j).

The proof is the coupling inequality P(A ≠ B) ≥ TV on j-prefixes.

Numbers:
- At J = 4096 and v1 = 4e-6/token: E[excess] ≥ 0.997 J for row-independent schemes, ≥ 0.64 J for any joint coupling.
- With TV_j measured on real Nano text (cross-fade on), the optimal coupling still leaves 1.7-1.9 rows/step with
  LCP < 16, 6.1-6.9 with LCP < 128 and 22-25 with LCP < 4096. So a row needing ≥ L − 128 serial tokens exists on
  **99.8 % of steps under any coupling**.
- Plain BV sits 1.7-1.9× (model) or 2-7× (real text, j ≤ 128) above the floor.
- Re-verifying at intermediate phases does not help: the final verification sees a single age-≥1 draft again.

### Theorem S (price of spares)
- With N drafts per prompt for G outputs (k = N/G) and w_j = p_{≤j}/q_{≤j}:
  E[# outputs whose j-prefix matches no draft] ≥ G · sup_{t>k} P_p(w_j > t)(1 − k/t).
- Minimal drafts per output for ≤ 0.05 unmatched per step: k_min = 1.39 at K = 1, v1; 1.63 at 2v1; 2.07 at 4v1;
  3.43 at 10v1.
- Even the information-theoretic ideal is ≈ 1.10× SPR, because the spares decode inside the critical phase.
- **Lemma W:** no exact weights-only best-of-k selection exists without a residual branch. So implementable exact
  multi-draft schemes are BV-type (⊥-BV pools).

### Proposition Q (the ⊥ price is a quantile)
- For any exact single-draft mechanism with content-independent ⊥, c = P(¬⊥) = 1/M:
  P(¬⊥, LCP < j) ≥ c · E_p[(1 − M/w_j)⁺].
- With ln w_j ~ N(a²/2, a²), holding the per-row early-tail rate at ε needs ln M ≈ z_ε·a. So **⊥ ≈ 6 × TV at
  ε = 1e-4**.
- High acceptance sets the scale; exactness plus a bounded max-over-512-rows tail sets the multiplier.
- Real text makes it worse: early fork tokens carry heavy single-token ratios. Continued counts stayed flat (1-2 per
  step) from M 1.35 to 2.6 in-loop, where a Gaussian walk predicts a fast fall.

### Lemma R (post-rejection work is irreducible)
- Given a rejection at τ with Y_τ = y ≠ x_τ, the suffix has law π_n(·|y) whatever the coupling.
- Measured: 16-token runs of the dead suffix cover 4.4-5.3 % of the new suffix, against 3.4-4.2 % for an unrelated
  sibling. **The old suffix carries no more information than an independent sample.** Post-rejection work is
  (ℓ − τ)/s.

### Theorem D (depth)
- Model generation as rounds of θ_n evaluations on prefix sets. If a mechanism is exact for every θ_n consistent with
  the past, every output prefix must be evaluated, and each unguessed token costs a round. So
  **E[gen_n] ≥ t · E[max_r (ℓ_r − LCP_r)] / s**, with s the tokens accepted per forward.
- H(Y | θ_{n−1}, θ_n, keys) = 0. The gap is sequential evaluation depth, not information. The only levers are
  depth levers (s, t per round) or hardware placement.

### Corollary P (parity)
SPR fits two batches into one depth-L window by mixing two policies; exactness allows one policy per window. Parity
per step needs one of:
- **(A)** fork rows decoded at s ≥ 2 relative to the bulk (or at half the per-token latency): BX;
- **(B)** fork rows moved off the window onto hardware that does not train (a fork lane): FL;
- **(C)** 2B samples of one policy per window (XS at batch 2B), which changes the algorithm.

At lag K the condition generalizes to s ≥ K + 1 on fork rows.

### MTD on tails
MTD (lookup speculation) is exact. The floor vanishes iff tail tokens per iteration ≥ K + 1. Nano real text
measures 1.64, which cuts the K = 1 excess from ≈ 4.1k to ≈ 0.9k iterations. On noise text the effect is 0.

---

## 4. First principles A: exact sampling under strict alternation

- **Floor at s = 1:** gen ≥ L · t_tail + check = 8192 × 6.5 ms + 2.3-4.2 s, i.e. ≥ 55-57 s. That gives a step
  ≥ 83-85 s = **1.40-1.43× SPR-BI**. XS measures 87-90 s, so **XS is within 3-8 % of every exact
  strict-alternation scheme at s = 1**. All plumbing that does not raise s (spares, ⊥, couplings, schedules,
  re-verification) can win at most that.
- **BX, measured-anchored** (Nano 8k), slowest fork row, replay over full re-decodes of τ < 16 rows:
  - n-gram tree (W 64, sources limited to what exists at the wake): s = 1.58-1.60 → 1.19-1.23× SPR-BI;
  - online sources: 1.70-1.72;
  - n-gram W 8 + a 1-token MTP-style keyed draft (measured α by depth 0.895 / 0.856 / 0.809): **s = 2.13-2.16**,
    which closes the L/2 window (K 2: 2.84; K 3: 3.43).
- **The drafter is not the limit; the module is.**
  - On today's MCore speculative path (fixed overhead 2-2.7 ms per step, serial MTP draft 6-7 ms per depth), every
    exact design with a drafter is slower than plain XS (1.6-1.9×).
  - With an intrinsic W-token verify and a fused draft: BX 1.10-1.14×, XS-SD 1.04-1.10× at Nano 4 nodes.
  - **XS-SD is the simpler and faster exact route at small scale.**
- **Module targets** (Nano 8k headline, exact within 1.10× SPR-BI), in units of the 1-token decode step t1 ≈ 6.7 ms:
  - fixed speculative overhead f ≤ 0.05 t1 (≈ 0.35 ms);
  - draft per depth d ≤ 0.07 t1 (≈ 0.47 ms);
  - verify c ≤ 34 µs per extra token per rank (already met).
  - MCore today is 6-8× over on f and 13-25× over on d.
- **Which design wins depends on G, the number of lockstep EP groups.**
  - G = 2 (Nano 4 nodes): XS-SD.
  - G ≥ 4 (Super at EP8, Ultra): BX-SD with fork rows packed on one group.
    - Lever L1: draft only rows behind schedule. It reads only the row's own progress, so it is exact.
    - Lever L2: place fork rows on F of G groups. This is placement only, so it is exact.
- **Against SPR given the same drafter (SPR-SD, the fair baseline), exactness still costs 1.12-1.38×.** The only
  thing that beats Corollary P there is load asymmetry: fork rows on a lightly loaded group pay ≈ 1.3-1.5 t1 per
  iteration, against ≈ 2.0-2.2 t1 for bulk rows with K = 3 verify width.
  - Measured inputs: a bitwise-valid lane runs at λ = 0.69-0.74 of bulk latency. The wake check riding decode costs
    ×0.97-1.0, i.e. it does not hide.
  - The best Nano variant (one EP8 group carries the fork rows plus a lighter bulk share) gives 1.18-1.22× SPR-SD,
    against XS-SD's 1.27-1.28.
- **Momentum-lookahead drafts** (θ̃ = θ_n + the known part of the Adam step): exact as proposals, with 2.7-5× lower
  1-step update variance measured in parameter space. They do not remove the floor, and in colocated mode they need
  a second weight set. Dropped for colocated; free in DX.
- **Carried π_{n−1} rows as drafts inside XS-SD lose** (1.08-1.24×). They do not shorten the slowest row (Lemma R)
  and they add a batch of decoding rows.

## 5. First principles B: where exact sync's time goes

- XS at Nano 8k = 89.9 s: generation 62.5, train 25.3, other ~2.1 [M]. Generation is one serial chain of 8192
  forwards at 7.6 ms, latency-bound by kernel count (~3.5 ms of row-independent small-op floor). It uses ~27 % of the
  engine's 128-row throughput.
- **The whole gap to SPR-BI is half a chain:** (L/2) × 7.6 ms ≈ 31 s. SPR fills the engine's slack with *width* (rows
  of the next batch); exactness forbids that.
- **Exactness does not forbid filling the same slack with *depth*:** speculative tokens per row, verified under π_n.
  A verify of W = 3-4 tokens at 26 rows/rank costs only ~1.1-1.25× t(r).
- **Three exact levers, and only three:** s (tokens per forward on a fresh row), t (latency at low occupancy), and
  rows per DP rank (train / chain ratio).
- Strict alternation is not the binding constraint. 1 − 0.42^8 = 99.9 % of groups (8k) contain a capped row, so no
  group finishes early. **The binding constraints are on-policy causality and GRPO group baselines.**
- **Lookup alone is not enough.** n-gram/self/sibling lookup gives the median row 1.4-1.6× but the critical
  (slowest, least repetitive) row only 1.2-1.3×. The drafter must be prefix-conditioned (MTP head / trained draft
  head).
- XS-SD = gen/e + train + other: e = 1.8 gives 1.06×, e = 2.0 parity with SPR-BI. With t × 0.82 (fused decode ops),
  e = 1.8 gives **1.01× stock SPR**.
- **Rows per rank:** the absolute XS − SPR gap is (L/2)·t ≈ 24-31 s whatever the batch. So XS/SPR-BI falls 1.52 →
  1.31 → 1.15 → 1.08 at B = 512 → 4096. Exactness is cheap at production batch.
- **Honest bound:** with the same engine and drafter, the exact chain is L/s and SPR's is L/(2s). The price of
  exactness is (L/2)·t/s per step.
- XS-SD dominates the exact partial-rollout arms (EX-T phase max(L/2, L/s) + recompute + rework). **Build exactness
  into the decode, not the carry.**

## 6. First principles C: what on-policy needs at the gradient level

- **Theorem U (uniqueness):** from stale rows alone, the only estimator unbiased for every bounded statistic is IS.
  Its variance factor is 1 + χ²(P‖Q) ≥ e^{KL}.
- **Per-token or per-decision weights are biased** for a terminal reward: every token's term needs the full pre-wake
  product. Token-level on-policyness of post-wake tokens is exactly what lets the weight stop at the wake (Lemma L).
- **Theorem R (row weights).** Train Σ_i w_i f_i(Y_i; R_{−i}) with siblings unweighted. Then
  E_Q[w_i f_i] = E_{R_{−i}∼Q} E_{Y_i∼P}[f_i].
  - (a) RLOO / LOO mean with a constant normalizer: **exactly unbiased** for ∇J.
  - (b) Stock LOO-mean/LOO-std with binary reward: U′, Σ_x c(p_x^Q) ∇J_x. The prompt-scale error at |Δp| = 0.01 is
    ≤ 0.1 % for p ∈ [0.3, 0.7] and 2.4-5 % at p = 0.1/0.05/0.9.
  - (c) The per-row 1/ℓ normalizer adds an O(Δp) term.
- **GRPO-LOO-c:** LOO mean, a past-only global scale s_n, a constant length normalizer. RIS is then exactly unbiased
  for ∇J/s_n.
- **Coupled roulette (RC):** ĝ_g = h(Y^q) + (B_g/π_g)[f(Y^p) − h(Y^q)], where Y^p is the exact coupled group and
  B_g ~ Bern(π_g) is drawn from information fixed before the coin.
  - Unbiased for **any** statistic.
  - Added variance ≤ sup‖f‖²/π, with no likelihood ratio.
  - Cost is a *max* over selected re-decodes, so the variance-optimal rule is one shared coin per step for expensive
    groups.
  - π = 1 is exact SPR.
- **ESS model** with cross-fade on: row s ~ N(−n v/2, n v), so row ESS ≈ e^{−n·v}. RIS stays ≥ 0.8 while
  n·v_stale ≤ 0.22: v ≤ 3e-5 at 8k, ≤ 2e-5 at 16k, ≤ 7e-6 at 32k. Group weights multiply the exponent by G.
- **The early IS-SPR ESS collapse was a missing lever.** Those runs had cross-fade off. The fitted fork-event model
  implies a BV whole-row rejection of 0.24-0.57, which matches the measured no-cross-fade 0.53-0.57.
- **Staleness conservation:** at SPR speed a capped row has on average ≥ L − sL/2 pre-wake tokens. Schedules move
  them between steps but cannot remove them. A balanced schedule (constant n = L/2) minimizes the mean IS variance
  factor by Jensen. It matters only when L·v ≳ 0.3.
- **Tiers:** E (XS, exact SPR, EX-T, BX, XS-SD) is the reference; U (RIS + LOO-c, RC) is a separately named
  "on-policy-unbiased" tier, with logged ESS ≥ 0.8 and logged ε̂.

## 7. First principles D: exact vs lag-K async

- **Theorem 1 (exact cycle):**
  τ_E ≥ max{ T(1+ε) + o , a + (ℓ_crit − LCP_crit)·t_f/s_f + t_last + o }, where:
  - T is learner time per batch and o is refit + optimizer;
  - a is the time until θ_n is evaluable on the sampling hardware;
  - t_f/s_f is the fork-row latency per token.
  - LCP_crit < 128 on ≥ 99.8 % of steps (Theorem F with measured TV_j).
- **Theorem 2 (lag-K cycle):** τ_K ≥ max{T + o, decode work / inference throughput, L·t(r)/(K+1)}. The depth is
  amortized over K + 1 steps.
- **Corollary 1:** stock async is learner-bound for K ≥ 2 (exposed generation 0-3.6 s at Nano/Super/Ultra). So
  **ratio_K ≥ max(1 + ε, r), with r = (a + L t_f/s_f + o)/(T + o_a), independent of K for K ≥ 2**.
  - The gap is depth and nothing else: one L-long serial decode under the newest policy per step.
  - Width terms (wasted draft forwards, the verify prefill, re-decode) are second order: ε = 0.01-0.04 in replay.
  - At K = 1 async is generation-exposed, so an exact design drafting at K_d ≥ 3 can beat async-1.
  - **The x ≤ threshold condition:** exact matches async when x := t_f/s_f ≤ (T + o_a − a − o)/L. For example, Nano
    needs s_f ≥ 1.5-1.6 at t_f ≈ 6.5 ms (s_f 1 → 1.54; 1.3 → 1.19; 1.5 → 1.04; 2.15 → 1.01-1.04 of T).
- **Lemma V (position-major verification):** verify position P of all unverified rows by time a + P·v. A fork at τ
  finishes by a + τv + (ℓ − τ)x ≤ a + L·max(v, x).
  - If v ≤ x, discovery is free.
  - Replay: learner-fused-only verification gives 1.58-1.64 T; position-major/two-ended gives **1.01-1.04 T**.
  - **Verification order is first-order.** Late discovery, not verification cost, is what makes learner-side verify
    slow.
- **Corollary 2 (colocated):** under strict alternation the fork chain overlaps training only on a lane of share f.
  The exact surplus f·T/(1−f) + C (wake recompute) + εT does not shrink with K, while SPR-K's window ~L/(K+1) does.
  **The K penalty of colocated exact designs is placement, not exactness.**
- **Proposition N:** draft numerics set acceptance, not exactness. A non-BI engine's numeric mismatch (gen-KL 0.8-2e-3
  per token) is ~300-1000× the per-step drift, so every 8k row would be rejected and IS ESS ≈ e^{−13}. BI is a speed
  requirement for drafts.
- **Proposition S (selection):** Σ_{i∈S} f_i is unbiased for every f iff P(i ∈ S | Y_i, rest) does not depend on
  Y_i. The following are biased: first-B-to-finish (toy cos 0.90), accepted-only rows, deferring fork rows. No
  finite-sample Horvitz-Thompson repair exists; RC is the only unbiased fallback.
- **Proposition K:** lag costs volume, not depth. The critical remaining depth is 8021-8191 tokens at every divergence
  scale from half the age-1 drift to 7× it.
- **Proposition M:** a mechanism seeing M i.i.d. candidates and outputting an exact row or ⊥ has
  P(success) ≤ M / ess sup(p/q). Exact SIR therefore reduces to ⊥-BV or BV.
- **Ranking (modeled):**
  - DX: 0.90-1.07× async-2/4 at Nano with s_f ≥ 1.6 and a light lane. Revised on measured fork-row latency (6.2-7.6 ms
    loaded) to 1.10-1.30× unless fork rows run light-loaded with MTP.
  - RIS-async: ≈ 1.00-1.03 at every K (U).
  - CFL-Z (colocated: SPR-K carry + BV wake check + fork lane + GRPO-Z): 1.19-1.38× at Nano (lane 1/4) and 1.04-1.24×
    at Ultra (lane 1/16-1/32).

## 8. Early rejections and late verification

- **No part of early rejection is a bug.**
  - Observed first-fork hazard / (1.19 × TV estimated from lp − lq) = 0.93-1.01 in every position bin.
  - Kernel/numerics contribute 0: at lr 0, rescore = generation log-probs bitwise and there are 0 forks.
  - Hard routing flips were removed by cross-fade.
  - The rest is genuine change:
    - ~70-80 % is the base rate: 512 rows × ~13 forkable opening positions × the body TV;
    - ~20-33 % is a real position effect: opening-plan tokens move 1.44× more per token at lower entropy;
    - ~26 % of TV at every position is the cross-fade band coefficients moving with the update.
  - Typical early forks are opening-plan choices (" parse" → " find"). At the fork token |lp − lq| < 1e-2 in 30 of 35
    cases: the Gumbel margin was small, not the logit jump.
- **Early rejections cannot be removed exactly, and removing them would not shorten the phase.** The critical depth
  is 8170-8189 with all rows, 8044-8063 without τ < 128, and 7649-7679 without τ < 512. Dropping every fork before
  512 buys 6 %.
  - For the floor itself to vanish, TV would have to fall 30-35×, i.e. v ~10³× lower. Only an algorithm change could
    do that (lr, KL on openings), and none is approved.
  - Couplings, early-token key designs, "first K0 tokens fresh", shadow branches and fork prediction give no depth
    gain.
- **Late verification.** F1 (critical-path-ordered verification: verify the first J positions of every row first,
  dispatch forks immediately) and F2 (prefix check in the learner right after the optimizer step) are exact and
  free for keyed checks, because keyed decisions are prefix-local.
  - For BV, the cut is decided by the row's suffix. Measured: speculation on the prefix-only cut covers 4-21 % of
    τ < J rows (hit rate 0.5-0.7), and the critical row was covered on 0 of 10 steps.
  - **F1/F2 gain on BV designs ≈ 0.** Keep them for keyed designs only.
- **Lag K:**
  - Early-fork counts scale with √A(K): ×1.6-1.8 / 2.1-2.5 / 2.5-3.3 at K = 2 / 3 / 4.
  - Eager per-wake re-verification is free given the mandatory θ_n re-prefill. It keeps the critical-path forks at
    the K = 1 rate. **Use eager verification at every K.**
  - Every in-window exact design degrades with K, because the floor ≈ L·c/s is flat while SPR's window shrinks:
    XS-SD 1.14 → 1.45× SPR-BI at K = 4; EX-T + F1 + F2 1.57 → 2.0×. Only a fork lane is ~K-independent.
- **IS-SPR at lag K:**
  - Group weights fail from K = 2 (ESS ≤ 0.57, clip mass 4-26 %), and already at K = 1 at the acceptance-decay peak.
  - Row weights hold to K = 4 early in training (ESS ≥ 0.73, ε ≤ 0.5 %), and to K = 2 (async) / K = 3 (SPR carry) in
    the steady state.

## 9. Acceptance audit

Concern: "acceptance on Nano looks too low". Every acceptance number was predicted from first principles, then
measured. **Answer: no bug; both acceptances sit at their first-principles values.**

| acceptance | first-principles expectation | measured |
|---|---|---|
| keyed, q := p (same tensor) | 1 exactly | **1.0000** (885k positions) |
| draft head, depth 1, held-out, keyed | ≤ 1 − TV = 0.774 and ≥ (1−TV)/(1+TV) = 0.63; a broken (fresh-key) coupling gives Σpq = 0.606 | **0.751 = 97 % of the 1 − TV ceiling** |
| same, train rows | 1 − TV 0.810 | 0.788 (overfit gap 3.7 pts) |
| engine draft head, conditional on visited bases | = offline walk 0.704 | 0.704 |
| policy one step stale, per token, keyed | 1 − 1.19·TV with TV 3.71e-4 → 0.99956 | **0.99956** |
| keyed token-wise recheck, rows whole | E exp(−1.19 Σ_t TV_t) = 0.092 (Jensen-low) | 0.120 |
| BV, rows whole | E exp(S_L − max_k S_k) from measured lp/lq = 0.892 | 85-89 % |
| lr 0 | 0 forks | 0 forks, per-token 1.0000 |

- **Draft-head 0.75 is drafter quality on high-entropy text** (TV 0.23, KL 0.45 nats/token held-out), plus overfitting
  on a small prompt set.
  - 56 % of the loss sits at entropy ≥ 1 nat, where keyed already beats independent sampling 2-4×; that gain is the
    coupling working.
  - The head misses 7 % of near-deterministic tokens: a capacity signal for a 1-layer head.
  - Per-row corr(acceptance, mean entropy) = −0.80.
  - On Nano text even a perfect drafter gets only Σp² = 0.707 under independent sampling. Super's 0.895 comes from
    much easier text and a natively trained MTP head, so it is not a Nano target.
- **The ~10 % keyed-whole-row rate is the law exp(−Σ TV)** at TV 3.7e-4/token over ~6.7k-token rows (~2.9 expected
  forks per row). It is linear in L·TV. BV's whole-row rate is a drawdown of a √L-scaled walk, hence 85-89 %.
- A quoted "96 %" was a per-token BV number at M ≈ 1.11 over non-⊥ outputs, not a whole-row rate.
- **Side finding:** a harness that does not apply the BI recipe gives non-bitwise run-to-run forwards (keyed
  self-agreement 0.9948). With BI on, det_eq = 1.0000. Head acceptance does not depend on it.
- **Reporting rule:** always pair a keyed-whole-row rate with exp(−1.19 Σ TV), and a BV rate with M and its
  denominator. Report drafter acceptance next to 1 − TV and Σp².
- **Regression test** (CPU): q = p → keyed 1; (1−TV)/(1+TV) ≤ keyed ≤ 1−TV; keyed > Σpq; a fresh-key draft matches at
  Σpq (the broken-coupling signature); stale-policy mismatch/TV = O(1).

**Acceptance-decay mechanism.**
- Per-token divergence v between consecutive policies is not constant over training. On masters that start on the
  bf16 grid, the realized bf16 update is smaller than the master update. Once masters are off-grid (steady state),
  realized v rises 27-49× at matched step size. One long run went 2.6e-6 → 4.7e-5, peaked at 3.3e-4 and settled at
  ~2e-5.
- The openings carry the late-training change. Expect TV and early-fork counts ×3-7 later in training.
- This is genuine policy change (π_n is the bf16 model), not a bug. The structural lever (sticky rounding) is opt-in
  for convergence runs only.

---

## 10. Step-time model, and why predictions come before measurements

**The model.** Every arm is a sum of measured per-phase terms:

  step = decode (iterations × t(rows/rank)) + (re)prefill / verify + exposed tail + train + outside (refit,
  transitions)

Inputs:
- the BI decode curve t(r) (e.g. Nano 5.8 + 0.041 r ms; Ultra-L27 8.87 + 0.0146 r ms);
- verify/prefill cost per token per rank (engine prefill ~9-15 µs, learner forward 21-33 µs);
- train time from the stock split;
- acceptance from the BV row law, calibrated in-loop (v1 = 4e-6 at Super-L22, cross-checked by the 8.4-8.8 %
  continued rate).

Characteristic forms:
- SPR: gen ≈ L/(K+1) × t((K+1)·rows).
- XS: L × t(rows).
- Exact partial rollout: max(L/2, (L − τ_min)/s) × t + check.
- DX: max(T(1+ε)+o, δT + L t_f/s_f + o, τ_inf).

**Why predict first.**
1. **A prediction outside its band (±10 %) is a probable bug, not a surprise.** It triggers a bug hunt before any
   number is trusted. Examples where this paid off:
   - the IS-SPR ESS collapse traced to cross-fade being off;
   - the AX miss traced to heavy-tailed fork ratios (the Gaussian walk was wrong in the tail) plus two implementation
     inflations;
   - an XS training phase 4× too slow traced to expandable-segment allocator thrash in the colocated learner;
   - "acceptance too low" resolved as the law.
2. **It finds the fundamental bottleneck before code is written.** Theorems T, F, S, D and Corollary P were all
   derived before the corresponding builds, and they retired XPR, XPR-S, AX and exact SPR as speed paths without
   multi-node runs.
3. **It forces fair baselines.** Every ratio is stated against the strongest fair baseline at the same layout, budget
   and model: SPR-BI isolates exactness from the BI kernel tax, and SPR-SD isolates it from the drafter.
4. **It states the per-rank cost at scale** (TP/EP/PP, multi-node) for every design, so nothing is built that only
   works at 4 nodes.

**Registered falsifiers (examples).**
- XS ≥ 1.2× stock sync means the BI tax or the colocated transitions are larger than modeled.
- An exposed tail ≤ 5 s would mean the early-τ mass is absent in-loop.
- XS-SD K3 at Nano 8k must land at 60-63 s.
- Module microbench: f ≤ 0.35 ms, d ≤ 0.47 ms per depth, c ≤ 34 µs/token/rank.

## 11. Bottom line

- Under strict alternation, exactness in law and one draft per slot, the **tail floor is a theorem** (F, S, D),
  not an engineering gap. Spares, couplings, envelopes, schedules and re-verification cannot remove the
  ≈ L-token serial re-decode of an early-forking row, which exists on ≥ 99.8 % of steps.
- **The only exact levers are depth and placement:** tokens per forward on fork rows (keyed tree / MTP
  speculation), per-token latency at low occupancy, a fork lane or packed EP group, and, outside colocated mode,
  overlapping the fork chain with training (DX + GRPO-Z + position-major verify).
- **Simplest exact colocated route:** XS-SD (exact sync + keyed speculative decode). It needs a fast
  speculative-decode module (fused draft, intrinsic verify).
- **Relaxed tier:** IS-SPR / RIS (unbiased, ≈ 1.08×) with logged ESS and ε̂.
- Every exact arm is gated by bitwise P1, the sentinel, the law check and a failing negative.
