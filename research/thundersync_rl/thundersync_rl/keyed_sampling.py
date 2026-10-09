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
"""Position-keyed sampling for exact cross-iteration speculative rollouts.

Every token is a Gumbel-max sample whose noise is a deterministic function of
(trajectory seed, absolute position, token id):

    x = argmax_v  logit(v) + G(seed, position, v),   G = -log(-log U),

with U a hashed uniform. This is an exact sample from softmax(logit). Because the
noise depends only on the key, two policies emit the same token wherever their
races agree, so a draft produced under any weights can be checked position by
position: the emitted sequence is exactly what the target policy would produce
with the same keys, whatever produced the draft. Gumbel coupling matches the
optimal coupling at two-way forks, where nearly all disagreements occur.

One Triton kernel per call: each program reduces one row over the vocabulary
with the noise generated in registers, so there is no host sync and no
materialized noise.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _mix32(x):
    x = x ^ (x >> 16)
    x = x * 0x7FEB352D
    x = x ^ (x >> 15)
    x = x * 0x846CA68B
    return x ^ (x >> 16)


@triton.jit
def _uniform(cols, key):
    h = _mix32(cols * 0x9E3779B9 + key)
    return ((h >> 8).to(tl.float32) + 0.5) * (1.0 / 16777216.0)


@triton.jit
def _row_key(seed, pos, stream):
    lo = (seed & 0xFFFFFFFF).to(tl.uint32)
    hi = ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32)
    k = _mix32(lo ^ stream)
    k = _mix32(k ^ hi)
    return _mix32(k ^ (pos & 0xFFFFFFFF).to(tl.uint32))


@triton.jit
def _keyed_gumbel_argmax_kernel(
    logits_ptr, stride, seed_ptr, pos_ptr, variant_ptr, eps_ptr, out_ptr, vocab,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    seed = tl.load(seed_ptr + row)
    pos = tl.load(pos_ptr + row)
    key = _row_key(seed, pos, 0x632BE5AB)
    jkey = _row_key(seed + tl.load(variant_ptr + row) * 0x51ED27, pos, 0xC6577B56)
    eps = tl.load(eps_ptr + row)
    best = tl.full([BLOCK], float("-inf"), tl.float32)
    best_idx = tl.zeros([BLOCK], tl.int32)
    for off in range(0, vocab, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < vocab
        lg = tl.load(
            logits_ptr + row.to(tl.int64) * stride + cols, mask=mask, other=float("-inf")
        ).to(tl.float32)
        c = cols.to(tl.uint32)
        score = lg - tl.log(-tl.log(_uniform(c, key)))
        # Draft variants only (eps > 0): keyed logistic jitter on the logits.
        uj = _uniform(c, jkey)
        score = score + eps * (tl.log(uj) - tl.log(1.0 - uj))
        score = tl.where(mask, score, float("-inf"))
        better = score > best
        best = tl.where(better, score, best)
        best_idx = tl.where(better, cols, best_idx)
    m = tl.max(best, axis=0)
    idx = tl.min(tl.where(best == m, best_idx, 2147483647), axis=0)
    tl.store(out_ptr + row, idx.to(tl.int64))


def keyed_sample_logits(
    logits: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    vocab_limit: int | None = None,
    jitter: torch.Tensor | None = None,
) -> torch.Tensor:
    """Keyed Gumbel-max sample from softmax(``logits[:, :vocab_limit]``) per row.

    ``seeds``/``positions``: [n] int64 (position = index of the sampled token).
    ``jitter``: optional [n, 2] (draft variant id, eps); eps = 0 rows are exact
    samples, eps > 0 rows are perturbed drafts. Targets never pass jitter.
    """
    n, width = logits.shape
    vocab = width if vocab_limit is None else min(vocab_limit, width)
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    dev = logits.device
    seeds = seeds.long().to(dev, non_blocking=True).contiguous()
    positions = positions.long().to(dev, non_blocking=True).contiguous()
    if jitter is None:
        variants = torch.zeros(n, dtype=torch.int64, device=dev)
        eps = torch.zeros(n, dtype=torch.float32, device=dev)
    else:
        jitter = jitter.to(dev, non_blocking=True)
        variants = jitter[:, 0].long().contiguous()
        eps = jitter[:, 1].float().contiguous()
    out = torch.empty(n, dtype=torch.int64, device=dev)
    if n:
        _keyed_gumbel_argmax_kernel[(n,)](
            logits, logits.stride(0), seeds, positions, variants, eps, out, vocab,
            BLOCK=2048,
        )
    return out
