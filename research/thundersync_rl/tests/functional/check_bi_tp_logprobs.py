# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Is the batch-invariant training forward batch invariant at this TP?

Loads one Megatron policy on ``--gpus`` GPUs (pass TP / BI flags as Hydra
overrides) and computes log-probs of the same sequences in two orders. Each
sequence then sits in a different micro-batch, at a different row, padded to a
different length, beside different sequences, so its rows' positions and the
token count per forward change. Under sequence parallelism a row's owner in
the tensor-parallel reduce-scatter depends on its position and the token count,
so this exercises exactly that dependence. Pass: every sequence's log-probs are
bitwise equal between the two layouts.

  python research/thundersync_rl/tests/functional/check_bi_tp_logprobs.py \\
      --config research/thundersync_rl/configs/grpo_math_1.5b_thundersync_megatron.yaml \\
      --gpus 4 policy.megatron_cfg.tensor_model_parallel_size=4 ... (BI flags)

Set NEMO_RL_ORDERED_TP_REDUCE=0 to run with NCCL's own TP reductions.
"""

from __future__ import annotations

import argparse
import sys

import torch
from omegaconf import OmegaConf

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def _batch(seqs: list[torch.Tensor], pad: int) -> BatchedDataDict:
    n = max(s.numel() for s in seqs)
    ids = torch.full((len(seqs), n), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : s.numel()] = s
    return BatchedDataDict(
        {"input_ids": ids, "input_lengths": torch.tensor([s.numel() for s in seqs])}
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--num-seqs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args, overrides = p.parse_known_args()

    register_omegaconf_resolvers()
    cfg = load_config(args.config)
    if overrides:
        cfg = parse_hydra_overrides(cfg, overrides)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    pcfg = cfg["policy"]
    pcfg["megatron_cfg"]["train_iters"] = 1  # normally set by grpo.setup()
    tokenizer = get_tokenizer(pcfg["tokenizer"])
    pad = tokenizer.pad_token_id or 0

    g = torch.Generator().manual_seed(args.seed)
    vocab = min(tokenizer.vocab_size, 150000)
    lengths = torch.randint(40, 1500, (args.num_seqs,), generator=g).tolist()
    seqs = [torch.randint(10, vocab, (n,), generator=g) for n in lengths]

    init_ray()
    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[args.gpus],
        use_gpus=True,
        num_gpus_per_node=args.gpus,
        max_colocated_worker_groups=1,
        name="bi_tp_logprobs",
    )
    policy = Policy(
        cluster=cluster,
        config=pcfg,
        tokenizer=tokenizer,
        name_prefix="bi_tp_logprobs",
        init_reference_model=False,
    )

    # Layout A: sequences in order. Layout B: a permutation, so each sequence sits
    # in another micro-batch, at another row, padded to another length, beside
    # other sequences (micro-batches must be full, so no one-row calls).
    perm = torch.randperm(len(seqs), generator=g).tolist()
    lp_a = policy.get_logprobs(_batch(seqs, pad))["logprobs"]
    lp_b = policy.get_logprobs(_batch([seqs[j] for j in perm], pad))["logprobs"]
    pos_b = {j: k for k, j in enumerate(perm)}
    worst, bad = 0.0, 0
    for i, s in enumerate(seqs):
        a = lp_a[i, : s.numel()]
        b = lp_b[pos_b[i], : s.numel()]
        d = (a.float() - b.float()).abs().max().item()
        worst = max(worst, d)
        bad += int(not torch.equal(a, b))
        print(
            f"[bi_tp] seq {i} len {s.numel()} row {i}->{pos_b[i]}: max|diff| {d:.3e}",
            flush=True,
        )
    verdict = "PASS" if bad == 0 else "FAIL"
    print(
        f"[bi_tp] {verdict}: {bad}/{len(seqs)} sequences differ between batch layouts; "
        f"max|diff| {worst:.3e}",
        flush=True,
    )
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
