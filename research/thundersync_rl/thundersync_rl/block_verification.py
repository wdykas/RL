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

Null-augmented block verification (⊥-BV, ``null_m`` = M >= 1): the draft is
verified against p_hat = p/M (+) (1 - 1/M) * BOT, where the "no output" symbol BOT
is possible only at the first position (q(BOT) = 0). Then log b_i gets -log M
from position 1 on (b_i = w_i / max(M, max_{u<=i} w_u), w = prefix p/q), the
residual at i >= 1 is unchanged in form, and the root residual (i = 0) is
(p_1/M - q_1)+ plus the BOT atom 1 - 1/M. Outcomes: an exact p-sample with
probability exactly 1/M, or BOT (``NULL_TOKEN``: the draft yields nothing, no
residual sample). M = 1 is plain block verification, bit for bit.
See ``null_bv.py`` for the pooled per-prompt process.

Banded importance sampling (``band`` = b > 0, opt-in): the verification target
is p~ = q at positions t <= T and p after, where T is the first position whose
prefix log-ratio S_t = sum_{s<=t} log p(x_s)/q(x_s) has |S_t| >= b (T = g when
the draft never leaves the band; the target beyond the draft is always p). p~
is a function of the prefix only, so block verification against it is exact
for p~; the in-band prefix is accepted with probability 1 and tau >= T for
every output. With the continuation drawn from p, the output Y and its weight
w = exp(S_T) satisfy E[w f(Y)] = E_p f(Y) for every f (the law of Y is p/w), so
rows are exact in expectation, not in law (DESIGN.md §14e). Composes with ⊥-BV:
P(output) is still exactly 1/M.

Lazy evaluation (``lazy_stop_index``): h_i <= b_i and P(tau = g) = h_g = b_g, so
with eta drawn up front tau = g is decided from b alone, and below g only
positions with eta_i <= b_i need r_i (an O(V) pass over p_i and q_i).
"""

from __future__ import annotations

import math

import torch

# Emitted token id meaning BOT ("no output"): the draft is discarded (⊥-BV).
NULL_TOKEN = -1


def acceptance_weights(log_ratio: torch.Tensor, null_m: float = 1.0) -> torch.Tensor:
    """Acceptance weights b [g+1] (fp64) from log p(x_i) - log q(x_i) [g].

    Closed form of b_i = min(1, b_{i-1} p/q): log b_i = S_i - max_{j<=i} S_j.
    ``null_m`` = M > 1 (⊥-BV): the first position's target is p/M, so S_i gets
    -log M for i >= 1 (b_i = w_i / max(M, max_{u<=i} w_u)).
    """
    s_cum = torch.cat(
        [
            torch.zeros(1, dtype=torch.float64, device=log_ratio.device),
            torch.cumsum(log_ratio.double(), 0),
        ]
    )
    if null_m != 1.0:
        s_cum[1:] -= math.log(null_m)
    return (s_cum - torch.cummax(s_cum, 0).values).exp()


def band_exit(log_ratio: torch.Tensor, band: float) -> int | None:
    """Banded IS exit position T (1-based): the first t with |S_t| >= band, or
    None when the prefix log-ratio never leaves the band.

    ``log_ratio``: [g] log p(x_t) - log q(x_t) (fp32, as verification computes
    it); S_t is its fp64 cumulative sum.
    """
    s = torch.cumsum(log_ratio.double(), 0)
    hit = torch.nonzero(s.abs() >= band).flatten()
    return int(hit[0]) + 1 if hit.numel() else None


def banded_log_weight(kept_log_ratio: torch.Tensor, band: float) -> tuple[float, bool]:
    """(log w, exited) of a banded-IS output from its kept draft tokens' log-ratios.

    Every output keeps the draft through T (tau >= T), so the kept prefix holds
    the exit, or the whole draft when it never left the band: log w = S_T, or
    S_g without an exit.
    """
    t = band_exit(kept_log_ratio, band)
    if t is None:
        return float(kept_log_ratio.double().sum()), False
    return float(torch.cumsum(kept_log_ratio.double(), 0)[t - 1]), True


def banded_target(
    lp: torch.Tensor, lq: torch.Tensor, draft: torch.Tensor, band: float
) -> torch.Tensor:
    """Target log-prob rows of banded IS: q's rows at positions 1..T, p's after.

    ``lp``: [g+1, V], ``lq``: [g, V]. Rows past the draft (the bonus row) stay p.
    """
    g = draft.numel()
    ar = torch.arange(g, device=lp.device)
    t = band_exit(lp[ar, draft] - lq[ar, draft], band)
    t = g if t is None else t
    return torch.cat([lq[:t].to(lp.dtype), lp[t:]])


def zero_mass_below(t_exit: int, residual_mass):
    """Banded IS: ``residual_mass(idx, b_idx)`` with rows idx < ``t_exit`` (target
    q, b <= 1, so sum (b q - q)+ = 0 exactly) set to 0 without reading them."""

    def f(idx: torch.Tensor, bi: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(idx.numel(), dtype=torch.float64, device=bi.device)
        sel = idx >= t_exit
        if bool(sel.any()):
            out[sel] = residual_mass(idx[sel], bi[sel])
        return out

    return f


def check_null_m(null_m: float, g: int) -> None:
    """⊥-BV needs M >= 1 and a non-empty draft (BOT lives at the first drafted position)."""
    if not null_m >= 1.0:
        raise ValueError(f"null_m must be >= 1 (got {null_m})")
    if null_m != 1.0 and g < 1:
        raise ValueError("⊥-BV (null_m > 1) needs a draft of at least one token")


def null_root_residual(
    residual_mass, b: torch.Tensor, null_m: float
):
    """Wrap ``residual_mass(idx, b_idx)`` with the ⊥-BV root residual.

    Row 0's target is p_1/M plus the BOT atom: r_0 = sum (b_0 p_1/M - q_1)+ + (1 - 1/M).
    ``residual_mass(idx, b_idx)`` computes sum_v (b_idx p - q)+ at rows ``idx``.
    """

    def f(idx: torch.Tensor) -> torch.Tensor:
        bi = b[idx]
        if null_m == 1.0:
            return residual_mass(idx, bi)
        root = idx == 0
        bi = torch.where(root, bi / null_m, bi)
        r = residual_mass(idx, bi)
        return torch.where(root, r + (1.0 - 1.0 / null_m), r)

    return f


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
    b_h, eta_h = torch.stack([b, eta]).cpu()  # one device sync
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
    p_row: torch.Tensor,
    q_row: torch.Tensor | None,
    b_tau: torch.Tensor,
    null_m: float = 1.0,
) -> torch.Tensor:
    """Unnormalized law of the emitted token: p (bonus, ``q_row=None``) or the residual.

    An empty residual happens only when p_tau == q_tau exactly (measure zero,
    e.g. equal weights); then fall back to p.

    ``null_m`` = M > 1: the root residual (tau = 0, only then is this passed)
    over vocabulary + BOT: [(b_0 p/M - q)+, 1 - 1/M]; the last entry is BOT.
    """
    if null_m != 1.0:
        assert q_row is not None, "⊥-BV: the root emission is always a residual"
        dist = (b_tau / null_m * p_row - q_row).clamp(min=0)
        return torch.cat([dist, dist.new_full((1,), 1.0 - 1.0 / null_m)])
    if q_row is None:
        return p_row
    dist = (b_tau * p_row - q_row).clamp(min=0)
    return dist if bool(dist.sum() > 0) else p_row


def emitted_token(index: int, vocab: int) -> int:
    """Token id of a sampled emission index (``NULL_TOKEN`` for the BOT entry)."""
    return NULL_TOKEN if index == vocab else index


def block_weights(
    lp: torch.Tensor,
    lq: torch.Tensor,
    draft: torch.Tensor,
    null_m: float = 1.0,
    band: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (h [g+1], b [g+1], residual masses r [g]) for a draft of length g.

    ``lp``: [g+1, V] target log-probs at positions 0..g; ``lq``: [g, V] draft
    log-probs at positions 0..g-1; ``draft``: [g] token ids. ``null_m``: ⊥-BV
    envelope (1 = plain block verification). ``band``: banded IS (None: off).
    """
    g = draft.numel()
    check_null_m(null_m, g)
    if band is not None:
        lp = banded_target(lp, lq, draft, band)
    ar = torch.arange(g, device=lp.device)
    b = acceptance_weights(lp[ar, draft] - lq[ar, draft], null_m)

    def mass(idx, bi):
        return (
            (bi[:, None] * lp[idx].double().exp() - lq[idx].double().exp())
            .clamp_(min=0)
            .sum(-1)
        )

    r = null_root_residual(mass, b, null_m)(ar)
    return stop_probabilities(r, b), b, r


def _emit_dist(lp, lq, b, tau: int, g: int, null_m: float = 1.0) -> torch.Tensor:
    """Distribution of the emitted token after keeping draft[:tau] (last entry BOT
    when the ⊥-BV root residual is emitted)."""
    q_row = None if tau == g else lq[tau].double().exp()
    m = null_m if tau == 0 and tau < g else 1.0
    return emit_distribution(lp[tau].double().exp(), q_row, b[tau], m)


def block_verify(
    lp: torch.Tensor,
    lq: torch.Tensor,
    draft: torch.Tensor,
    generator: torch.Generator | None = None,
    null_m: float = 1.0,
    band: float | None = None,
) -> tuple[int, int]:
    """Sample (tau, y): keep draft[:tau], then emit y (``NULL_TOKEN``: BOT, no output).

    ``band``: banded IS (see the module doc); the output's weight is
    ``banded_log_weight`` of its kept tokens.
    """
    g = draft.numel()
    if band is not None:
        lp = banded_target(lp, lq, draft, band)
    h, b, _ = block_weights(lp, lq, draft, null_m)
    eta = torch.rand(g + 1, generator=generator, dtype=torch.float64, device=lp.device)
    tau = int(torch.nonzero(eta <= h).flatten().max())
    dist = _emit_dist(lp, lq, b, tau, g, null_m)
    y = int(torch.multinomial((dist / dist.sum()).float(), 1, generator=generator))
    return tau, emitted_token(y, lp.shape[-1])


def block_verify_lazy(
    lp: torch.Tensor,
    lq: torch.Tensor,
    draft: torch.Tensor,
    generator: torch.Generator | None = None,
    null_m: float = 1.0,
    band: float | None = None,
) -> tuple[int, int]:
    """``block_verify`` computing the residual masses r_i only where needed.

    Same generator calls in the same order (one ``rand(g+1)``, one
    ``multinomial``), so (tau, y) and the generator state afterwards are those
    of ``block_verify``.
    """
    g = draft.numel()
    check_null_m(null_m, g)
    if band is not None:
        lp = banded_target(lp, lq, draft, band)
    ar = torch.arange(g, device=lp.device)
    b = acceptance_weights(lp[ar, draft] - lq[ar, draft], null_m)

    def mass(idx, bi):
        return (
            (bi[:, None] * lp[idx].double().exp() - lq[idx].double().exp())
            .clamp_(min=0)
            .sum(-1)
        )

    eta = torch.rand(g + 1, generator=generator, dtype=torch.float64, device=lp.device)
    tau, _ = lazy_stop_index(b, eta, null_root_residual(mass, b, null_m))
    dist = _emit_dist(lp, lq, b, tau, g, null_m)
    y = int(torch.multinomial((dist / dist.sum()).float(), 1, generator=generator))
    return tau, emitted_token(y, lp.shape[-1])


def output_distribution_given_draft(
    lp: torch.Tensor,
    lq: torch.Tensor,
    draft: torch.Tensor,
    null_m: float = 1.0,
    band: float | None = None,
) -> list[tuple[int, torch.Tensor]]:
    """Exact law of (tau, y) for a fixed draft: [(tau, P(tau) * P(y | tau))].

    With ⊥-BV (``null_m`` > 1) the tau = 0 entry has V + 1 columns, the last
    one being BOT. ``band``: banded IS (y is then drawn from the banded target;
    every non-BOT outcome has tau >= T).
    """
    g = draft.numel()
    if band is not None:
        lp = banded_target(lp, lq, draft, band)
    h, b, _ = block_weights(lp, lq, draft, null_m)
    out = []
    for i in range(g + 1):
        p_tau = h[i] * torch.prod(1 - h[i + 1 :])
        dist = _emit_dist(lp, lq, b, i, g, null_m)
        out.append((i, p_tau * dist / dist.sum()))
    return out
