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
"""torchrun --nproc-per-node=2: canonical-block sharded paths (trainer _GatheredVocabLogprob forward + backward,
verifier GatheredLogprobRows.token_logprobs) equal the gathered-row canonical path bit for bit at TP2."""

import os

import torch
import torch.distributed as dist


def main():
    assert os.environ.get("NEMO_RL_VOCAB_LSE") == "canonical_block"
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    from thundersync_rl.engine_q import GatheredLogprobRows, TPVocabGather

    from nemo_rl.models.megatron import vocab_lse
    from nemo_rl.models.megatron.zero_train_gen_mismatch import (
        _GatheredVocabLogprob,
        _gathered_logprob_grad,
        _gathered_target_logprobs,
    )

    group = dist.new_group(list(range(world)))
    V, B, S = 32768, 2, 700
    w = V // world
    bad = []
    for dtype in (torch.float32, torch.bfloat16):
        g = torch.Generator(device="cuda").manual_seed(11)
        full = (torch.randn(B, S, V, generator=g, device="cuda") * 3).to(dtype)
        target = torch.randint(0, V, (B, S), generator=g, device="cuda")
        gout = torch.randn(B, S, generator=g, device="cuda")
        local = full[..., rank * w:(rank + 1) * w].contiguous().requires_grad_(True)
        lp = _GatheredVocabLogprob.apply(local, target, rank * w, (rank + 1) * w, 128, group, False)
        want = _gathered_target_logprobs(lambda a, e: full[:, a:e], target, 128)
        if not torch.equal(lp.view(torch.int32), want.view(torch.int32)):
            bad.append(f"{dtype} trainer forward")
        (lp * gout).sum().backward()
        want_g = _gathered_logprob_grad(
            lambda a, e: full[:, a:e], target, gout, vocab_start_index=rank * w,
            vocab_end_index=(rank + 1) * w, chunk_size=128, grad_input=torch.zeros_like(local),
        )
        if not torch.equal(local.grad.float(), want_g.float()):
            bad.append(f"{dtype} trainer backward")
        rows = full[0]  # [S, V]: one draft's rows
        tok = target[0]
        p = GatheredLogprobRows(rows[:, rank * w:(rank + 1) * w], TPVocabGather(group), V)
        got = p.token_logprobs(tok)
        want_t = vocab_lse.logprob_rows(rows).gather(-1, tok[:, None])[:, 0]
        if not torch.equal(got.view(torch.int32), want_t.view(torch.int32)):
            bad.append(f"{dtype} verifier token log-probs")
    flag = torch.tensor([len(bad)], device="cuda")
    dist.all_reduce(flag)
    if rank == 0:
        print("bad:", bad, flush=True)
        if int(flag.item()) == 0:
            print("VOCAB_LSE MULTIRANK OK", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
