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

Lazy evaluation (``lazy_stop_index``): h_i <= b_i and P(tau = g) = h_g = b_g, so
with eta drawn up front tau = g is decided from b alone, and below g only
positions with eta_i <= b_i need r_i (an O(V) pass over p_i and q_i).
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


def stop_probability(r: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise h = r / (r + 1 - b) (1 where the denominator vanishes), i < g."""
    denom = r + 1 - b
    return torch.where(denom > 0, r / denom.clamp(min=1e-300), torch.ones_like(denom))


def stop_probabilities(r: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Stop probabilities h [g+1] from the residual masses r [g] and b [g+1]."""
    return torch.cat([stop_probability(r, b[:-1]), b[-1:]])


# Lazy scan: skip position i only when eta_i > b_i * (1 + slack_i). Exactly
# h_i <= b_i (r_i <= b_i). As computed, r_i <= b_i * sum_v p_i(v), and the
# computed p_i sums to 1 up to the fp32 rounding of log p_i = logit - lse (about
# |lse| * 2^-24 plus the entropy times 2^-24); h_i <= b_i (1 + slack_i) then holds
# for the computed h_i too, so a skipped position never has eta_i <= h_i.
# ``lazy_slack`` gives slack_i = LAZY_SLACK + 2^-20 |lse_p,i| (16 ulp of lse).
LAZY_SLACK = 1e-4


def lazy_slack(lse_p: torch.Tensor | None = None) -> torch.Tensor | float:
    """Per-position slack for ``lazy_stop_index`` from the fp32 logsumexps of p."""
    if lse_p is None:
        return LAZY_SLACK
    return LAZY_SLACK + lse_p.double().abs() * 2.0**-20


def lazy_stop_index(
    b: torch.Tensor,
    eta: torch.Tensor,
    residual_mass,
    slack: torch.Tensor | float = LAZY_SLACK,
    first: int = 4,
) -> tuple[int, int]:
    """Stop index tau = max{i : eta_i <= h_i}, evaluating r_i only where needed.

    ``b``, ``eta``: [g+1] fp64. ``residual_mass(idx)`` returns r at the positions
    ``idx`` (LongTensor [k], descending), bitwise equal to the eager r[idx].
    ``slack``: scalar or [g] (see ``lazy_slack``).
    Pathwise identical to ``max(nonzero(eta <= stop_probabilities(r, b)))``:

    * i = g: h_g = b_g, compared directly (whole draft accepted, no r at all);
    * i < g with eta_i > b_i (1 + slack) >= h_i: never a stop position, skipped;
    * the remaining candidates are evaluated from the top in batches of
      ``first``, 2 ``first``, ... with the eager formula; the first hit is tau.

    Returns (tau, number of r_i evaluated). Raises RuntimeError where the eager
    rule would take the max of an empty set.
    """
    g = b.numel() - 1
    b_h, eta_h = b.cpu(), eta.cpu()
    if bool(eta_h[g] <= b_h[g]):
        return g, 0
    if isinstance(slack, torch.Tensor):
        slack = slack.cpu()
    cand = torch.nonzero(eta_h[:g] <= b_h[:g] * (1 + slack)).flatten().flip(0)
    done, n = 0, first
    while done < cand.numel():
        idx = cand[done : done + n]
        idx_d = idx.to(b.device)
        h = stop_probability(residual_mass(idx_d), b[idx_d])
        hit = torch.nonzero((eta[idx_d] <= h).cpu()).flatten()
        done += idx.numel()
        if hit.numel():
            return int(idx[hit[0]]), done
        n *= 2
    raise RuntimeError("block verification: no stop position (eta > h everywhere)")


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


def block_verify_lazy(
    lp: torch.Tensor,
    lq: torch.Tensor,
    draft: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[int, int]:
    """``block_verify`` computing the residual masses r_i only where needed.

    Same generator calls in the same order (one ``rand(g+1)``, one
    ``multinomial``), so (tau, y) and the generator state afterwards are those
    of ``block_verify``.
    """
    g = draft.numel()
    ar = torch.arange(g, device=lp.device)
    b = acceptance_weights(lp[ar, draft] - lq[ar, draft])

    def residual_mass(idx):
        return (
            (b[idx, None] * lp[idx].double().exp() - lq[idx].double().exp())
            .clamp_(min=0)
            .sum(-1)
        )

    eta = torch.rand(g + 1, generator=generator, dtype=torch.float64, device=lp.device)
    tau, _ = lazy_stop_index(b, eta, residual_mass)
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
