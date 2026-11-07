# Hand-off: exact on-policy colocated RL with speculation

This branch (`colo-handoff`) is upstream NeMo-RL main plus the colocated exact on-policy method, its batch-invariant
(BI) Megatron numerics, the exactness checks, a portable Slurm launch kit and these notes. It is the production line.
The in-flight research streams are separate branches on the same fork (`handoff/*`), listed in
[`ONGOING_WORK.md`](ONGOING_WORK.md).

| doc | what |
|---|---|
| this file | the method in one page, environment, how to run, fair-comparison protocol, gates |
| [`RESULTS.md`](RESULTS.md) | the verified numbers (tables inlined) and known bugs |
| [`DECISIONS.md`](DECISIONS.md) | the decisions and rules in force, with one-line rationale |
| [`THEORY.md`](THEORY.md) | the theory: methods, exactness conditions, tail floor, first-principles bounds, acceptance |
| [`ONGOING_WORK.md`](ONGOING_WORK.md) | every in-flight stream: branch, state, next step, results so far |

## 1. The method

GRPO is exactly on-policy only if every trained sample is drawn from the current weights π_n. Stock colocated partial
rollout (SPR: one GPU fleet both generates and trains; requests still decoding when training starts are paused and
resumed after the weight update) keeps requests running across the train step, so their prefixes come from π_{n-1}.
We keep that overlap and stay exact:

- The carried prefixes are treated as **speculative drafts**.
- After the step they are verified against π_n with a speculative-sampling accept/reject (block verification; the
  pooled, null-augmented variant is called ⊥-BV).
- The engine is **batch-invariant**: its log-probs are bitwise equal to the learner's for the same weights and tokens,
  at any batch composition, so accept/reject is exact.
- Accepted prefixes are exact π_n samples; rejected ones are resampled from π_n from the rejection point.

Arms in the code (`research/thundersync_rl/configs/colo/`):

| arm | config | what |
|---|---|---|
| `xs` | `colo.yaml` | exact colocated sync on the BI engine (lag 0). The exact reference. |
| `wvs` | `colo_wv.yaml` | production speculative arm: fixed G, P + S prompts admitted, 14 drafts per prompt at the constant envelope M 1.35, the first P prompts whose pools fill G slots are trained; drafts cut at the stand-down are verified when the next phase starts (wake verify) and continued ahead of new drafts. Selection depends only on ⊥ counts and fixed order, never on content, so every trained group is exact. |
| `xpr` / `xprs` | `colo_xpr.yaml` / `colo_xprs.yaml` | carried drafts verified at the stand-down, without / with spare prompts |
| `vg` | `colo_vg.yaml` | variable group size alternative |
| `isspr` | `colo_isspr.yaml` | IS-SPR: stock pause / resume plus a keyed rescore at the wake and importance weights (group at K=1, per-row + RLOO at K≥2). Unbiased, not exact in law. Currently the fastest near-on-policy arm. |
| `sprbi` | `colo_sprbi.yaml` | baseline b: the stock SPR loop on our engine and BI numerics, engine settings identical to the compared arm |
| `spr` / `cs` | stock tree | baseline a: stock SPR at lag K / stock colocated sync |

Every run of ours checks exactness online: a train-log-prob sentinel on every token every step (engine log-prob ==
learner log-prob, bitwise), and router weights plus expert bias frozen (checked from the effective config and a
per-step router fingerprint). The theory is in [`THEORY.md`](THEORY.md).

Code map:
- `research/thundersync_rl/thundersync_rl/`: the method (colocated stage `colo.py` / `colo_stage.py`, `nolanes*.py`,
  `null_bv.py` and `block_verification*.py` for ⊥-BV, `carried_walk.py`, `engine_slots.py`, `isspr*.py`, `xspr*.py`,
  `law_check.py` for the per-token law test, `router_freeze.py`, `keyed_sampling*.py` for the 53/64-bit keyed sampler).
- `nemo_rl/models/megatron/`: BI numerics (`zero_train_gen_mismatch.py` validates and installs them; `fp16_gemm.py`,
  `fp16_moe*.py`, `vocab_lse.py`, `mamba_bi_*`, `ep_invariant_combine.py`, `bi_unpermute.py`, `latent_tp_shard.py`),
  cross-fade routing (`crossfade_routing.py`, fused kernel in `cf_router_fused/`), engine memory (`memory_saver.py`,
  `kv_live_offload.py`), learner CUDA graphs (`learner_graphs.py`), device-side MoE dispatch (`device_dispatch.py`).
- `nemo_rl/utils/module_provenance.py`: every process checks at startup that it loaded the pinned Megatron-LM /
  Bridge / TE trees (root, git commit, content hash).
- `research/thundersync_rl/scripts/colo/`: the launch kit (below).
- Unit tests: `research/thundersync_rl/tests/unit/` and the touched `tests/unit/**`. Many need a GPU and skip on CPU.

## 2. Environment

Hardware used: GB200 nodes with 4 GPUs each (aarch64). The kit assumes 4 GPUs per node, Slurm with pyxis / enroot
containers, and a shared filesystem visible from every node.

Software pairing (all public):

| component | ours (`xs`, `wvs`, `isspr`, `sprbi`, ...) | stock baseline (`spr`, `cs`) |
|---|---|---|
| NeMo-RL | this branch | branch `handoff/stock-baseline` on the fork (stock colocated loop plus one wake-barrier ordering fix, reliability only) |
| Megatron-LM | NVIDIA/Megatron-LM main `a11c84aec`, no patches | `6a3660905` + cherry-pick of NVIDIA/Megatron-LM#8358 (branch `handoff/stock-pin` on the wdykas/Megatron-LM fork) |
| Megatron-Bridge | NVIDIA-NeMo/Megatron-Bridge main `9981c93` | `1f8873bb` (the submodule pin of this branch) |
| TransformerEngine | 2.18.0 (overlay) | same overlay (stock numerics: BI mode off) |
| flash-attn-4 / CuTe DSL | 4.0.0b19 / 4.6.2 (overlay) | same |
| torch_memory_saver | 0.0.10 (overlay, our arms) | off by default (stock's faster config) |

Container: a NeMo-RL image built from this repo's `docker/Dockerfile` (torch 2.11-era, Python 3.13, venv at
`/opt/nemo_rl_venv`). The overlays are built on top by `build_overlays.sh` rather than baked into the image.

Workspace layout the kit expects (override any path with the env vars in `env.sh`):

```
$COLO_WORKSPACE/
  RL/                      this branch (colo-handoff)
  RL-stock/                handoff/stock-baseline
  Megatron-LM-main/        NVIDIA/Megatron-LM @ a11c84aec
  Megatron-Bridge-main/    NVIDIA-NeMo/Megatron-Bridge @ 9981c93
  Megatron-LM-stock/       wdykas/Megatron-LM handoff/stock-pin
  Megatron-Bridge-stock/   NVIDIA-NeMo/Megatron-Bridge @ 1f8873bb
```

Models and data: full Nemotron-3 Nano (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`) and DAPO-Math-17k from the HF
hub; `setup.sh` downloads both. The Super / Ultra presets use layer-truncated checkpoints (Super-L8, Super-L22,
Ultra-L27) made with `scripts/colo/make_trunc_nemotron_h.py <hf dir> <N> <out>`. Truncated models emit noise text and
every row runs to the length cap, so they test plumbing and scale only.

## 3. How to run

All site-specific values are env vars; nothing in the kit names a cluster.

```bash
cd $COLO_WORKSPACE/RL
export COLO_OUT=/path/on/shared/fs/colo-out          # all outputs; never under $HOME
export COLO_IMAGE=/path/to/nemo-rl-image             # enroot image file or a registry image pyxis can pull
export COLO_CONTAINER_MOUNTS=/shared:/shared         # must cover the workspace, COLO_OUT and COLO_MODELS
export COLO_MODELS=/path/to/truncated-models         # nemotron3-super-L8 / -L22, nemotron3-ultra-L27
export COLO_ACCOUNT=<account> COLO_PARTITION=<partition>   # optional
K=research/thundersync_rl/scripts/colo
srun -N1 --container-image=$COLO_IMAGE --container-mounts=$COLO_CONTAINER_MOUNTS bash $K/build_overlays.sh   # once
bash $K/setup.sh                       # once per tree change: dirs, HF downloads, pins file
bash $K/submit.sh smoke                # 1 node, ~30 min; ends with "[smoke] PASS"
bash $K/submit.sh gate wvs             # exactness gate: positive exact, negative must fail
bash $K/submit.sh pair nano 1 sprbi,isspr,spr a   # fair 3-arm set, 4 nodes; submit again with the order swapped
python3 $K/score/score_pair.py $COLO_OUT/runs/colo-*-nano-a-*   # scorer (steady steps, whole-cycle windows)
```

`DRY=1 bash $K/submit.sh ...` prints the sbatch command instead of submitting. `COLO_SEGMENT=0` drops
`--segment=N` (which asks Slurm for one NVLink domain on GB200 NVL72 systems). Per-arm knobs (`STEPS`, `CHECKS=1`,
`MAX_REQ`, `ENGINE_BUF`, `XA='<hydra overrides>'`) are documented at the top of `lib/colo_arm.sh` and
`lib/presets.sh`; presets hold the exact knob sets of the verified runs per arm and model.

Every arm runs a config dry run first (`preflight.log`): compose, validate, capacity, and a hard refusal unless the
effective config has router weights and expert bias frozen. Each run ends with `lib/check_run.py` (sentinel, router
probes, negative-control verdicts). A GPU-idle watchdog (`lib/gpu_idle_watchdog.sh`) fails a job whose GPUs sit idle
too long, which also catches startup hangs; `GPU_IDLE_WATCHDOG=0` turns it off.

## 4. Fair-comparison protocol

Three arms in one allocation, each at its best config:
- a = stock SPR (stock tree, stock numerics);
- b = SPR-BI, the stock loop on our engine and BI numerics, engine settings identical to c;
- c = ours.

Read c / b as the method overhead, b / a as the BI-numerics tax, c / a as end to end. Rules:
- Same model, layout, prompts × G, max length, steps, seed in every arm; routers frozen in every arm, baselines included.
- Each set runs twice with the arm order swapped, in the same job, and all runs are scored together.
- Windows hold whole cycles: stock-style arms alternate long / short steps with period K+1; steady steps only (≥ 3).
- Report the kernel-normalized column too: BI arms' s/step minus (generation + verify) × (1 − 1/f_gen), with the
  measured BI generation-kernel factor f_gen = 1.209 and f_train ≈ 1. Kernel speed is not counted for or against the
  method.
- Stock runs its best config: generic levers (e.g. KV persist instead of offload, no torch_memory_saver) go to stock
  too; BI-only levers stay BI.
- Write the predicted per-phase split before measuring; a miss is treated as a probable bug.
- Lag: every method is designed and reported for K = 1..4.
- Headline numbers come from the representative regime (full Nano, real text, long sequences, inference-bound).

## 5. Gates

A gate is one positive run and one negative control on 1 node (Super-L8, TP2 EP4), `CHECKS=1`:
- positive: bitwise engine == learner log-prob mismatch 0 on every output token, the per-token law test passes, the
  sentinel reports inexact 0 every step, at a stress learning rate (1e-4) so the policy moves;
- negative: a deliberately broken variant that must FAIL the same checks (`carry_unverified` trains cut drafts
  unverified; `spare_by_logp` / `vg_by_logp` select on content; `isspr_stale_kv` resumes on stale state; XS skips the
  reshard so the engine keeps θ_0).

`bash $K/submit.sh gate <wvs|xprs|xpr|vg|isspr|xs> [K]`; verdicts are printed by `lib/check_run.py`. A change to the
method lands only with its gate green and its negative red.

## 6. Where things stand

- `colo-handoff` passes the unit, XS, IS-SPR and wvs gates (positive exact, negatives fail), with the F3 band.
- Nano real text, 8k, 4 nodes, K=1, same-job set run in both orders on this code, pooled: IS-SPR **1.11× SPR-BI**
  (kernel-normalized 1.10); XS 1.37× SPR-BI (kn 1.34); SPR-BI / SPR 1.13 (kn 1.02, the BI-numerics tax is almost all
  kernel speed); IS-SPR / SPR 1.24 (kn 1.12).
- IS-RLOO with learner row weights (`colo_isrloo.yaml`), same-job vs SPR-BI: K=1 1.09×, K=2 1.01× and K=4 0.85×
  without the wake recompute (baseline capped at 1000 engine rows by Ray's default actor concurrency; ~1.02× predicted
  with the cap lifted); row ESS ≥ 0.96. Earlier sweep: 1.23 / 1.44 / 1.40×.
- XS-SD (exact sync + keyed tree speculative decode, head_v2), fair same-job set: 1.19-1.20× SPR-BI, 1.23-1.24×
  SPR-SD (the drafter-equipped SPR baseline), 1.20-1.21× stock SPR kernel-normalized.
- Full tables, regimes and negative results: [`RESULTS.md`](RESULTS.md).

Top next steps:
1. Close the speculative gap at Nano: IS-SPR's cost is the wake re-prefill of 2-3 M tokens per step, ~80% of them in
   finished rows (fix in flight: `handoff/isspr-dd`); the exact track (XS-SD, BX-SD) next as it gates.
2. Re-measure XS on the post-fix tip (`handoff/wakeorder`) and run lag K = 2 / 4 sets.
3. 16k Nano and Ultra with the same fair 3-arm ladder (preflight, 1-node gate, then 4 nodes).

## 7. Artifacts not in git

- The c3 MTP draft head (`head_mcore.pt`, a few hundred MB) used by the MTP / tree-speculation arms is not committed.
  It would be published as a release asset or HF model repo next to the fork; until then it can be retrained with the
  draft-head code in `handoff/draft-head`.
- Truncated Super / Ultra checkpoints: regenerate with `make_trunc_nemotron_h.py`.
