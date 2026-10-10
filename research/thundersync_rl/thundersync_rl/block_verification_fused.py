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
"""Block verification from raw logits without materializing [g, V] tensors.

Same algorithm and outputs as ``block_verification.block_verify`` (see there),
computed as: one logsumexp per row of p and q, the draft-token ratios and b on
[g] vectors, and one Triton pass per row for r_i = sum_v max(b_i p_i - q_i, 0).
Only the residual row that is actually sampled from is materialized.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _residual_mass_kernel(
    P, Q, lse_p, lse_q, b, out, sp, sq, vocab, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    lp0 = tl.load(lse_p + row)
    lq0 = tl.load(lse_q + row)
    bi = tl.load(b + row)
    acc = tl.zeros([BLOCK], tl.float64)
    for off in range(0, vocab, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < vocab
        p = tl.exp(
            tl.load(P + row.to(tl.int64) * sp + cols, mask=mask, other=float("-inf"))
            - lp0
        )
        q = tl.exp(
            tl.load(Q + row.to(tl.int64) * sq + cols, mask=mask, other=float("-inf"))
            - lq0
        )
        d = bi * p.to(tl.float64) - q.to(tl.float64)
        acc += tl.where(d > 0, d, 0.0)
    tl.store(out + row, tl.sum(acc, 0))


def block_verify_logits(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    draft: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[int, int, torch.Tensor]:
    """Sample (tau, y) from raw logits; also return log p of the kept tokens.

    ``p_logits``: [g+1, V] target logits, ``q_logits``: [g, V] draft logits (any
    float dtype; last dim contiguous), ``draft``: [g] token ids.
    Returns (tau, y, logp) with logp = log p(kept[i]) for kept = draft[:tau] + [y].
    """
    g = draft.numel()
    dev = p_logits.device
    if p_logits.stride(-1) != 1:
        p_logits = p_logits.contiguous()
    if q_logits.stride(-1) != 1:
        q_logits = q_logits.contiguous()
    lse_p = torch.logsumexp(p_logits.float(), -1)  # [g+1]
    lse_q = torch.logsumexp(q_logits.float(), -1) if g else lse_p[:0]
    ar = torch.arange(g, device=dev)
    lpx = p_logits[ar, draft].float() - lse_p[:g]
    lqx = q_logits[ar, draft].float() - lse_q
    s_cum = torch.cat(
        [
            torch.zeros(1, dtype=torch.float64, device=dev),
            torch.cumsum((lpx - lqx).double(), 0),
        ]
    )
    b = (s_cum - torch.cummax(s_cum, 0).values).exp()  # [g+1]
    r = torch.empty(g, dtype=torch.float64, device=dev)
    if g:
        _residual_mass_kernel[(g,)](
            p_logits,
            q_logits,
            lse_p.contiguous(),
            lse_q.contiguous(),
            b[:g].contiguous(),
            r,
            p_logits.stride(0),
            q_logits.stride(0),
            p_logits.shape[-1],
            BLOCK=2048,
        )
    denom = r + 1 - b[:g]
    h = torch.cat(
        [
            torch.where(denom > 0, r / denom.clamp(min=1e-300), torch.ones_like(denom)),
            b[g:],
        ]
    )
    eta = torch.rand(g + 1, generator=generator, dtype=torch.float64, device=dev)
    tau = int(torch.nonzero(eta <= h).flatten().max())
    p_row = (p_logits[tau].double() - lse_p[tau].double()).exp()
    if tau == g:
        dist = p_row
    else:
        q_row = (q_logits[tau].double() - lse_q[tau].double()).exp()
        dist = (b[tau] * p_row - q_row).clamp_(min=0)
        if not bool(dist.sum() > 0):  # p_tau == q_tau exactly: fall back to p
            dist = p_row
    y = int(torch.multinomial((dist / dist.sum()).float(), 1, generator=generator))
    kept = torch.cat([draft[:tau], torch.tensor([y], device=dev)])
    logp = p_logits[torch.arange(tau + 1, device=dev), kept].float() - lse_p[: tau + 1]
    return tau, y, logp
