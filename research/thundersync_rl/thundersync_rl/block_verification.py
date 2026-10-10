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
"""Block verification of one draft.

Sun et al. 2024, "Block Verification Accelerates Speculative Decoding".

Given a draft x_1..x_g sampled from q and the target p, keep x_1..x_tau and emit
one more token y, such that the result is distributed exactly as p:

    b_0 = 1,  b_i = min(1, b_{i-1} p(x_i)/q(x_i))
    r_i = sum_v max(b_i p_i(v) - q_i(v), 0)            (i < g)
    h_i = r_i / (r_i + 1 - b_i)   (i < g),   h_g = b_g
    tau = max{ i : eta_i <= h_i },  eta_i ~ U(0, 1) independent
    y ~ p_g                      if tau = g  (bonus token)
    y ~ max(b_tau p_tau - q_tau, 0) normalized   otherwise

with p_i = p(. | prefix, x_<=i) and q_i likewise.
"""

from __future__ import annotations

import torch


def acceptance_weights(log_ratio: torch.Tensor) -> torch.Tensor:
    """Acceptance weights b [g+1] (fp64) from log p(x_i) - log q(x_i) [g].

    Closed form of b_i = min(1, b_{i-1} p/q): log b_i = S_i - max_{j<=i} S_j.
    """
    s_cum = torch.cat(
        [
            torch.zeros(1, dtype=torch.float64, device=log_ratio.device),
            torch.cumsum(log_ratio.double(), 0),
        ]
    )
    return (s_cum - torch.cummax(s_cum, 0).values).exp()


def stop_probabilities(r: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Stop probabilities h [g+1] from the residual masses r [g] and b [g+1]."""
    denom = r + 1 - b[:-1]
    h = torch.where(denom > 0, r / denom.clamp(min=1e-300), torch.ones_like(denom))
    return torch.cat([h, b[-1:]])


def emit_distribution(
    p_row: torch.Tensor, q_row: torch.Tensor | None, b_tau: torch.Tensor
) -> torch.Tensor:
    """Unnormalized law of the emitted token: p (bonus, ``q_row=None``) or the residual.

    An empty residual happens only when p_tau == q_tau exactly (measure zero,
    e.g. equal weights); then fall back to p.
    """
    if q_row is None:
        return p_row
    dist = (b_tau * p_row - q_row).clamp(min=0)
    return dist if bool(dist.sum() > 0) else p_row


def block_weights(
    lp: torch.Tensor, lq: torch.Tensor, draft: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (h [g+1], b [g+1], residual [g, V]) for a draft of length g.

    ``lp``: [g+1, V] target log-probs at positions 0..g; ``lq``: [g, V] draft
    log-probs at positions 0..g-1; ``draft``: [g] token ids.
    """
    g = draft.numel()
    ar = torch.arange(g, device=lp.device)
    b = acceptance_weights(lp[ar, draft] - lq[ar, draft])
    resid = (b[:g, None] * lp[:g].double().exp() - lq.double().exp()).clamp_(min=0)
    return stop_probabilities(resid.sum(-1), b), b, resid


def _emit_dist(lp, lq, b, tau: int, g: int) -> torch.Tensor:
    """Distribution of the emitted token after keeping draft[:tau]."""
    q_row = None if tau == g else lq[tau].double().exp()
    return emit_distribution(lp[tau].double().exp(), q_row, b[tau])


def block_verify(
    lp: torch.Tensor,
    lq: torch.Tensor,
    draft: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[int, int]:
    """Sample (tau, y): keep draft[:tau], then emit y."""
    g = draft.numel()
    h, b, _ = block_weights(lp, lq, draft)
    eta = torch.rand(g + 1, generator=generator, dtype=torch.float64, device=lp.device)
    tau = int(torch.nonzero(eta <= h).flatten().max())
    dist = _emit_dist(lp, lq, b, tau, g)
    y = int(torch.multinomial((dist / dist.sum()).float(), 1, generator=generator))
    return tau, y


def output_distribution_given_draft(
    lp: torch.Tensor, lq: torch.Tensor, draft: torch.Tensor
) -> list[tuple[int, torch.Tensor]]:
    """Exact law of (tau, y) for a fixed draft: [(tau, P(tau) * P(y | tau))]."""
    g = draft.numel()
    h, b, _ = block_weights(lp, lq, draft)
    out = []
    for i in range(g + 1):
        p_tau = h[i] * torch.prod(1 - h[i + 1 :])
        dist = _emit_dist(lp, lq, b, i, g)
        out.append((i, p_tau * dist / dist.sum()))
    return out
