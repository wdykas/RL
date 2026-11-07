# DX: exact verification in the non-colocated async setting

DX is branch `handoff/dx`, built on `handoff/oldpin-base`. Megatron-LM pin: `handoff/oldpin-pin` on the wdykas/Megatron-LM fork.

What is on this branch:
- `thundersync.verify_on=engine`: θ_k's ⊥-BV verification of the drafts runs on the inference pool. It is bitwise
  equal to the learner, with canonical per-call seeds over fixed pool-order chunks. Each streamed chunk's slots are
  set at its own collect, which fixes the slot mix-up between chunks seen on a 4-node arm with
  `verify_stream_chunks > 1`.
- GRPO-Z train-on-arrival (`thundersync.grpo_z=loo_c`): the GRPO advantage uses the siblings' draft rewards as a
  leave-one-out baseline, so it stays unbiased. Rows train in units of exactly learner DP x MBS rows, in arrival
  order, so every DP / EP rank trains one row per unit.
- `thundersync.admit_after_tails`: fork priority by admission. The sampler admits the next lookahead batch only
  after the step's optimizer step. The admission gate depends on versions only, so exactness is unaffected.

Measured stock async reference (Nano 8k, 4 nodes): K=1 56.2 s, K=2 40.5 s, K=4 37.0 s per step. The projection for
DX is 1.10-1.30x async-2.

Known requirement: before tree / MTP arms run past step 6, rebase DX onto the tree_spec capture-race fix. That fix
makes lazy tree-stage CUDA-graph captures use `capture_error_mode="thread_local"`, so that an unsafe CUDA call from
another engine-process thread cannot invalidate a capture mid-flight. It is in `handoff/tree-spec` and
`handoff/bxsd2`; DX does not have it yet.

Open items:
- **Host-side hang at step 3 (culprit not identified).** The DX tip hung host-side at step 3 twice in BI-learner runs:
  the driver logs the train step but never issues the train call. The same arm ran 14 steps on the earlier DX state,
  which had GRPO-Z train-on-arrival but not the later commits. Those later commits are the suspects:
  - the GRPO-Z + early-deferred-verification relaunch;
  - the engine-verify FinalizePipeline drain;
  - two follow-ups.
- **Stall-abort coverage.** The async_rl stall-abort does not cover actor setup, so a hang during worker setup is not
  caught. One smoke of the DX tip hung there.
- The nolanes lag-2 hang at step 3 still needs a root cause. It may be the same issue as the host-side hang above.
