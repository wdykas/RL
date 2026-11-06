# Nano multi-depth MTP draft head ("c3")

`c3` is a multi-depth MTP (multi-token prediction) drafter for Nemotron-3-Nano-30B-A3B. The MTP, tree-speculation
(tree_spec) and XS-SD arms use it as their draft source. It is a single Megatron-Core MTP layer, `"*-"`: one attention
layer followed by one dense MLP layer. It is applied recurrently, EAGLE-style, to draft depths 1..D. Speculation is
exact: the engine's keyed (Gumbel-max) sampler verifies every draft, so with or without the head the generated tokens
and log-probs match plain decoding bitwise. The head changes speed only.

## Architecture

- 10 tensors with `mtp.layers.0.*` names (see `head_names.txt`), 47.8M parameters, bf16. These are the names that
  Megatron-Core `HybridModel(mtp_num_layers=1, "*-")` and the engine's `tree_mtp.MTPHead` use. Swapping in the weights
  needs no code change.
- Inputs: `enorm(emb(x)) ++ hnorm(h)` -> `eh_proj` [2688 x 5376] -> attention (32 query heads, 2 KV groups, head dim
  128, no RoPE) -> squared-ReLU MLP (FFN 1856) -> `final_layernorm` -> Nano's frozen output layer -> q.
- Chaining contract (identical in training and engine):
  - depth 1 at base b: Nano's final-norm hidden h_b and emb(x_{b+1}). Its K/V go at position b.
  - depth k >= 2: the head's own post-`final_layernorm` output hf_{k-1} and emb(x_{b+k}). Its K/V go at b+k-1.
    Attention reads the true K/V at 0..b plus the head's chained K/V.
  - every depth: q_k predicts x_{b+k+1}. The target is p = Nano's softmax at b+k.

## Training recipe

1. **Data** (`prep_data.py`): Nano's own T=1 samples on DAPO-Math prompts, capped at 8192 tokens, taken from an RL
   row dump. Rows are deduplicated by (prompt, sample, token hash).
   - Train: RL steps 0-8, 11,241 rows / 576 prompts / 74.8M tokens.
   - Held out: the step-9 prompts, 1024 rows / 64 prompts / 6.86M tokens. The script asserts that train and held-out
     prompts do not overlap.
2. **Depth-1 warm start** (`mk_shim.py` + `train_head.py`).
   - `mk_shim.py` builds an HF-format Nano + MTP directory. It symlinks the Nano weights and adds the head to
     `mtp.safetensors`.
   - Megatron-Bridge loads that directory as a HybridModel with one MTP layer. The base model is frozen.
   - Loss: forward KL(p_Nano || q) over the full vocabulary. Output: `head_mcore.pt` (MCore names) and
     `mtp.safetensors` (HF names).
3. **Multi-depth chained training** (`train_chain.py`; this produces c3).
   - The chain is unrolled over depths 1-3 with full backprop through the head's own hidden ("training-time test").
   - Tokens are teacher-forced. Under keyed coupling, a chain reaches depth k only if every earlier draft equalled
     the realised token, so the teacher-forced token is exactly the one the engine feeds.
   - Loss: sum_k KL(p || q_k), full vocabulary, depth weights 1,1,1.
   - Nano is frozen, routers are frozen, and cross-fade routing uses m 0.01 (extra 8). Warm start from the depth-1
     head.
   - AdamW with fp32 master weights, lr 5e-4, warmup 30, cosine to 0.1x. 1953 steps x 16 rows (2.78 epochs), about
     2.7 h on one 4-GPU node (TP1, DP4).
   - The head is a plain-PyTorch copy of `tree_mtp.MTPHead`'s arithmetic. `--mode smoke` checks it against the engine
     code.
4. **Round-2 levers** (`train_v2.py`): `--ffn` (wider MLP, zero-initialized so it starts exactly at c3), `--feats`
   (extra Nano residual-stream layers fed to `eh_proj`; the engine needs an extra hidden capture for this),
   `--ent_gamma` (entropy-weighted loss), and `--tv_beta` (adds a TV term).

## Measured results

All engine runs were exact: tokens and log-probs bitwise equal to plain keyed decode, 0 mismatches, no CUDA-graph
fallbacks.

**Offline keyed acceptance, all 1024 held-out rows** (`eval_walk.py`). Keys are posterior Gumbel keys given Nano's
realised sample, which is the engine's coupling at T=1. α is the per-insertion conditional rate.

| depth | α pooled | per-row p10 | min row | depth-1-only head |
|---|---|---|---|---|
| 1 | 0.750 | 0.675 | 0.543 | 0.748 |
| 2 | 0.702 | 0.609 | 0.489 | 0.29 |
| 3 | 0.689 | 0.586 | 0.468 | 0.15 |
| 4 (not trained) | 0.674 | 0.568 | 0.450 | - |

Simulated chain walk (s = tokens per verify forward):
- D3: pooled 2.51, median row 2.58, slowest row 1.83.
- D4: pooled 2.68, median row 2.78, slowest row 1.87.

**In-engine, real long Nano rollouts.** Setup: 24 held-out prompts, T=1, EOS-terminated, cap 8192 (155k tokens), every
row an MTP tail, W16 D4.

| arm | verify steps (plain: 8190) | slowest row s | median row s | tokens / tail-iter |
|---|---|---|---|---|
| adaptive depth, 64 node rows | 4276 | 1.905 | 2.13 | 2.06 |
| adaptive depth, 128 node rows | 4239 | 1.922 | 2.26 | 2.16 |
| forced depth 4, 128 rows | 4238 | 1.922 | 2.76 | 2.67 |

- Per-depth conditional acceptance in the engine: **0.70 / 0.66 / 0.67** (0.704 / 0.664 / 0.666 / 0.665 at d1-d4).
  The offline walk on visited bases gives 0.704 / 0.666 / 0.661 / 0.651, so the engine matches it.
- d1 on visited bases (0.704) is below the all-bases 0.750. This is selection: after a rejection the next base sits in
  harder text.
- Engine head vs training arithmetic (`check_engine_head.py`): keyed drafts differ on 0.06-0.31% of (base, key) pairs
  at d1-d4, and acceptance agrees to 1e-4 (0.7430 vs 0.7429 at d1).
- The slowest row reaches about **1.9 tokens per forward**, with a median row of 2.1-2.8. Before this head, the
  depth-1-only head gave 1.60 tokens per iteration in-engine, and n-gram lookup drafting gave a slowest row of 1.14-1.20.

**Draft cost** (`mtp_bench2.py`, CUDA-graph replay, Nano dimensions; target <= 0.47 ms/depth):

| context | live rows | ms / depth |
|---|---|---|
| 2k | 16 | 0.275 |
| 2k | 64 | 0.304 |
| 8k | 16 | 0.305 |
| 8k | 64 | 0.345 |

Overall cost is **0.28-0.35 ms per depth**. The KV append adds 0.055-0.062 ms per iteration, and head KV takes 513 MiB
per rank at 64 slots x 8k.

**Acceptance audit** (`audit_a.py`, `test_acceptance_bounds.py`). This checks whether 0.75 is a bug. It is not:
- q := p gives keyed acceptance exactly 1.0000 (885k positions). The coupling is correct.
- c3 depth-1 keyed acceptance is 0.751 against a maximal-coupling bound of 1 - TV = 0.774. That is **97% of the 1-TV
  ceiling**. A draft with a fresh, uncoupled key would give sum p*q = 0.606.
- The target p is Nano's sampling distribution. Cross-entropy of the realised tokens is 0.695, against entropy 0.698.
- The remaining loss is drafter quality on high-entropy text (held-out TV 0.23, KL 0.45 nats/token). 56% of the loss
  sits at Nano entropy >= 1. Per-row acceptance correlates with mean entropy at -0.80, so the slowest rows are the
  high-entropy, 8k-capped rows (α 0.56-0.63).
- The head is overfit: keyed acceptance is 0.788 on train vs 0.751 held-out (576 prompts). It also misses 7% of
  near-deterministic tokens (H < 0.1), which is a capacity signal.
- With the BI recipe on (`--bi 1`), the learner forward is bitwise run to run, and the head's acceptance does not
  change (0.7499 with and without BI).

## Round-2 target

Target: **slowest row >= 2.2 tokens per forward** in-engine, up from about 1.9. The levers address the high-entropy
rows, not depth:
- a wider MLP (`--ffn 3712` / `7424`; cost from `mtp_bench2.py`);
- an entropy-weighted loss (`--ent_gamma`);
- multi-layer Nano features (`--feats`; needs the engine-side hidden capture);
- 10x or more training prompts, to close the 3.7-point train/held-out gap.

## How to run

`run_example.sh` drives every stage from environment variables. It assumes one node with 4 GPUs and this fork's
NeMo-RL + Megatron-Core + Megatron-Bridge environment.

```
export NANO_HF_DIR=<local HF snapshot of nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16>
export DRAFT_HEAD_DATA=<dir for train.pkl / heldout.pkl>
STAGE=prep   ROWDUMP=<row dump root> OUT=$DRAFT_HEAD_DATA bash run_example.sh
STAGE=shim   OUT=<w>/init                                    bash run_example.sh       # fresh head
STAGE=train1 SHIM=<w>/init/shim OUT=<w>/d1                   bash run_example.sh
STAGE=shim   HEAD_SAFETENSORS=<w>/d1/mtp.safetensors OUT=<w>/d1   bash run_example.sh  # trained shim
STAGE=chain  SHIM=<w>/d1/shim INIT=<w>/d1/head_mcore.pt OUT=<w>/c3 bash run_example.sh  # c3
STAGE=walk   SHIM=<w>/d1/shim HEAD=<w>/c3/head_mcore.pt OUT=<w>/walk bash run_example.sh
STAGE=audit  SHIM=<w>/d1/shim HEAD=<w>/c3/head_mcore.pt DATA_PKL=$DRAFT_HEAD_DATA/heldout.pkl OUT=<w>/audit bash run_example.sh
STAGE=check  SHIM=<w>/d1/shim HEAD=<w>/c3/head_mcore.pt OUT=<w>/check bash run_example.sh
STAGE=long   HEAD=<w>/c3/head_mcore.pt OUT=<w>/long bash run_example.sh
STAGE=bench  OUT=<w>/bench bash run_example.sh
STAGE=v2     SHIM=<w>/d1/shim INIT=<w>/c3/head_mcore.pt OUT=<w>/v2 bash run_example.sh --ffn 3712 --ent_gamma 1
```

To use the head in the engine, point the tree-speculation MTP source at the file: `thundersync.tail_tree_mtp=<path to
head_mcore.pt>`, TP 1.

Row-dump format for `prep_data.py`: `<root>/<run>/rows_step*.jsonl`. Each line is a JSON object with `step`, `idx`
(prompt index), `j` (sample index), `prompt` (token ids, or null if already given), and `tokens` (generated ids).

## Files

| file | role |
|---|---|
| `prep_data.py` | row dump -> `train.pkl` / `heldout.pkl` |
| `mk_shim.py` | HF-format Nano + MTP directory (symlinked Nano weights + `mtp.safetensors`) |
| `train_head.py` | depth-1 trainer/evaluator, Nano harness (`build`, `Runner`), data helpers |
| `train_chain.py` | multi-depth chained trainer (c3), keyed evaluation, smoke parity vs `tree_mtp.MTPHead` |
| `train_v2.py` | round-2 levers (wide MLP, extra features, entropy/TV-weighted loss) |
| `eval_walk.py` | held-out keyed per-depth acceptance + simulated chain walk (tokens/forward, slowest row) |
| `audit_a.py`, `test_acceptance_bounds.py` | acceptance audit vs first-principles bounds; CPU regression test (numpy) |
| `check_engine_head.py` | engine `tree_mtp.MTPHead` vs training arithmetic on whole rows |
| `tree_engine_long.py`, `dl_bench.py` | in-engine long rollouts, plain vs tree+MTP, bitwise check + per-row speed |
| `mk_engine_rows.py` | engine-generated rows -> `eval_walk` pickle (offline vs in-engine on the same text) |
| `mtp_bench2.py` | draft cost per depth (CUDA-graph replay), FFN / feature variants |
| `head_names.txt` | tensor names and shapes of the c3 checkpoint |
| `run_example.sh` | generic launcher (all inputs from env vars) |

## Checkpoint

The trained checkpoint (head_mcore.pt, ~96 MB) is not in git; it would be published as a release asset or a HF model
repo alongside the fork.
