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
"""In-loop statistical LAW test of the emitted tokens (THUNDERSYNC_LAW_CHECK; redteam_1016 §1).

The claim of exact on-policy rollouts: every emitted token y_t is a draw from the target
policy p(. | y_<t) (the learner's training forward at theta_k, with the row's anchors),
whatever produced it (accepted draft token, residual draw at a rejection, bonus token,
⊥-BV pool selection, engine decode of a fresh row or continuation). The bitwise checks
(mismatch, engine q, routes) compare log-prob VALUES; they cannot see a wrong sampling
LAW (a wrong residual, a truncated tail, a biased pool choice, a sampler temperature).

Randomized probability integral transform, vocabulary ordered by probability (most
probable first) so that systematic errors land at the same place for every position:

    u_t = P_p(X ranks above y_t) + U_t * P_p(X ties y_t),   U_t ~ U(0, 1) independent,
    "ranks above" = logit z_x > z_y; "ties" = z_x == z_y (y_t itself included).

Under the claim u_t ~ U(0, 1) given the prefix, so the u_t of a row are i.i.d. uniform and
so are those of independent rows. No sort: two masked sums over the vocabulary per token
(the tail complement t_t = 1 - u_t = P(z_x < z_y) + (1 - U_t) P(ties) is summed directly,
so the tail bins keep relative precision). Over-sampling probable tokens (low temperature,
a residual drawn from p or q instead of (p - q)+) pushes mass to small u; a truncated or
under-sampled tail empties the bins near u = 1.

Per rank the state is ``N_STATS`` float64 numbers (bin counts + three moments), summed
over ranks / micro-batches / steps: O(1) per rank at any scale. Per token the work is
O(V / TP) on the rank holding that vocabulary shard (one compare + exp + two sums, in
token chunks; 3 scalars all-reduced over TP per token), i.e. about one extra log-softmax
pass over the logits of the check forward.

Tests on the pooled stats (``law_test``): chi-square on the bins (bins with < 5 expected
counts merged), a binned Kolmogorov-Smirnov sup-distance at the bin edges (conservative),
z-test of mean(u) (= 1/2, var 1/12) and an exact test of sum(-log t) (Exp(1) per token under H0,
so the sum is Gamma(n, 1): tail-weighted). Combined p = min(1, 4 x min p) (Bonferroni). The tail
test uses the Gamma(n, 1) law itself, not a normal approximation of mean(-log t): the sum is
right-skewed (skewness 2 / sqrt(n)), and the normal approximation rejected a perfect sampler at
n = 3 with probability 2.2e-3 instead of 2.5e-5 (one emitted token at t ~ 1e-4 gave p = 5e-5 at
n = 3; exact p = 1.4e-2). Simulated under H0 (1.2M draws per n), the combined p < 1e-4 has
probability <= 1e-4 at every n in 1..200 (normal approximation: 2.2e-3 at n = 3). Power: the bins resolve
per-token law errors whose effect on the u histogram exceeds ~3/sqrt(n) (n ~ 1e6 tokens:
~3e-3 of mass moved); a tail law error of 1e-6 TV/token is NOT statistically detectable
in-loop at any feasible n (that is what the sampler unit tests are for).
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional

import torch

N_BULK = 64  # equal-width bins on [0, 63/64); the last 1/64 is split by tail thresholds
TAIL_T = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)  # bins on t = 1 - u: (1/64, 1e-2], ..., [0, 1e-6]
N_BINS = N_BULK + len(TAIL_T)
# counts, n, sum u, sum -log t, surprisal: sum (-log p(y) - H), sum Var(-log p(X));
# probability: sum (p(y) - sum p^2), sum Var(p(X))
N_STATS = N_BINS + 7


def hash_u64(*parts: Any) -> int:
    """64-bit hash of ``parts`` (blake2b; ts-lawfix). Replaces crc32 as the source of the
    checks' randomization: crc32 is affine over GF(2), so the uniforms of structured keys
    (consecutive row ids, a fixed tau) are not independent (chi-square sd 17-18 instead of 11.2
    over 64 bins at n = 330; research/lawfix/crc_uniforms.log) and 32-bit values collide."""
    import hashlib

    h = hashlib.blake2b("\x1f".join(str(p) for p in parts).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "little")


def hash_uniform(*parts: Any) -> float:
    """A U(0, 1) draw that is a function of ``parts`` only: (top 53 bits + 1/2) 2^-53."""
    return ((hash_u64(*parts) >> 11) + 0.5) * 2.0**-53


def pit_seed(seed: int, dp_rank: int, call: int, stream: str = "pit") -> int:
    """Seed of the PIT randomization uniforms of one log-prob micro-batch (63 bits).

    Every data-parallel replica runs its own micro-batches 0, 1, ... with the same shapes
    (logprob_batch_size rows padded to the batch width), so a seed of (seed, call) alone gave
    call c the SAME uniform grid on every replica: the near-deterministic tokens of D rows (u ~ U)
    then land in the same bin and the bin counts are overdispersed ~ 1 + (D - 1) f^2 (f = share
    of near-deterministic tokens): chi2 ~ 2 x df at full Nano (DP 8, entropy 0.7) for a perfect
    sampler. TP / CP ranks of one replica must keep the SAME uniforms
    (TP all-reduces the sums of one token; CP slices one grid), so the rank term is the
    data-parallel rank only."""
    return hash_u64(stream, int(seed), int(dp_rank), int(call)) & 0x7FFFFFFFFFFFFFFF


def bin_probs() -> list[float]:
    """H0 probability of each bin (sums to 1)."""
    p = [1.0 / N_BULK] * (N_BULK - 1)
    edges = [1.0 / N_BULK] + list(TAIL_T) + [0.0]
    p += [edges[i] - edges[i + 1] for i in range(len(edges) - 1)]
    return p


def bin_index(u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Bin of each (u, t = 1 - u) pair (float64 tensors)."""
    bulk = torch.clamp((u * N_BULK).floor().long(), 0, N_BULK - 1)
    idx = bulk.clone()
    in_tail = t <= 1.0 / N_BULK
    tail = torch.full_like(bulk, N_BULK - 1)
    for k, thr in enumerate(TAIL_T):
        tail = torch.where(t <= thr, torch.full_like(tail, N_BULK + k), tail)
    return torch.where(in_tail, tail, idx)


def accumulate(u: torch.Tensor, t: torch.Tensor, mom: Optional[torch.Tensor] = None) -> torch.Tensor:
    """[N_STATS] float64 stats of the given u / t values (and ``mom`` [4, N] from ``pit_rows``:
    -log p(y) - H, Var(-log p(X)), p(y) - sum p^2, Var(p(X)) per token)."""
    out = torch.zeros(N_STATS, dtype=torch.float64, device=u.device)
    if u.numel() == 0:
        return out
    out[:N_BINS] = torch.bincount(bin_index(u, t), minlength=N_BINS)[:N_BINS].double()
    out[N_BINS] = float(u.numel())
    out[N_BINS + 1] = u.sum()
    out[N_BINS + 2] = (-torch.log(t.clamp(min=1e-300))).sum()
    if mom is not None:
        out[N_BINS + 3 : N_BINS + 7] = mom.double().sum(-1)
    return out


def pit_rows(
    logits: torch.Tensor,
    target: torch.Tensor,
    uniforms: torch.Tensor,
    *,
    vocab_start: int = 0,
    tp_group: Any = None,
    temperature: float = 1.0,
    chunk: int = 256,
    moments: bool = False,
    row_index: Optional[torch.Tensor] = None,
    row_fn: Any = None,
    shadow_uniforms: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, ...]:
    """(u, t) per row of ``logits`` [N, V_local] (this rank's vocab shard) for ``target`` [N]
    (global token ids) with ``uniforms`` [N] in [0, 1). TP: shards [vocab_start, +V_local),
    one MAX and two SUM all-reduces of [chunk]-sized vectors per chunk (identical on every
    TP rank). float64 results.

    ``shadow_uniforms`` [N] (optional): a second randomization of the same tokens, returned as
    two more tensors (u', t') after the usual ones; the masked sums are shared, so it costs two
    fused multiply-adds per token (the legacy-seed shadow of ts-lawfix).
    """
    import torch.distributed as dist

    # ``row_index``: rows of ``logits`` to use (target / uniforms are per selected row), gathered per
    # chunk so no [n_selected, V] copy is made.
    n = logits.shape[0] if row_index is None else row_index.numel()
    vl = logits.shape[1]
    us, ts, ms, us2, ts2 = [], [], [], [], []
    tp = tp_group is not None and dist.is_initialized() and dist.get_world_size(tp_group) > 1
    for a in range(0, n, chunk):
        z = (logits[a : a + chunk] if row_index is None else logits[row_index[a : a + chunk]]).float()
        if temperature != 1.0:
            z = z / temperature
        if row_fn is not None:
            # The policy's own row transform (exact top-p: -inf outside the canonical / sticky
            # nucleus), so the PIT is taken under the law the engine samples and the learner
            # trains, not under the untruncated softmax. ``row_fn(z, rows)``: rows = indices
            # into ``logits`` (row_index-relative positions resolved here).
            rows = (
                torch.arange(a, min(a + chunk, n), device=z.device)
                if row_index is None
                else row_index[a : a + chunk]
            )
            z = row_fn(z, rows)
        y = target[a : a + chunk].long() - vocab_start
        own = (y >= 0) & (y < vl)
        zy = torch.where(
            own, z.gather(1, y.clamp(0, vl - 1).view(-1, 1))[:, 0], torch.full_like(z[:, 0], -math.inf)
        )
        m = z.max(-1).values
        if tp:
            dist.all_reduce(zy, op=dist.ReduceOp.MAX, group=tp_group)
            dist.all_reduce(m, op=dist.ReduceOp.MAX, group=tp_group)
        e = torch.exp(z - m[:, None])
        zyc = zy[:, None]
        above = torch.where(z > zyc, e, 0.0).sum(-1, dtype=torch.float64)
        ties = torch.where(z == zyc, e, 0.0).sum(-1, dtype=torch.float64)
        below = torch.where(z < zyc, e, 0.0).sum(-1, dtype=torch.float64)
        parts = [above, ties, below]
        if moments:
            # Unnormalized moments (shift m): S1 = sum e (z - m), S2 = sum e (z - m)^2,
            # Q2 = sum e^2, Q3 = sum e^3 (fp64 accumulation).
            # -inf entries (a truncated policy's masked tokens) have e = 0: give them zs = 0, not
            # -inf (0 x -inf = NaN in the moments).
            zs = torch.where(e > 0, z - m[:, None], torch.zeros_like(z)).double()
            ed = e.double()
            parts += [
                (ed * zs).sum(-1),
                (ed * zs * zs).sum(-1),
                (ed * ed).sum(-1),
                (ed * ed * ed).sum(-1),
            ]
        sums = torch.stack(parts)
        if tp:
            dist.all_reduce(sums, group=tp_group)
        above, ties, below = sums[0], sums[1], sums[2]
        tot = above + ties + below
        uu = uniforms[a : a + chunk].double()
        us.append((above + uu * ties) / tot)
        ts.append((below + (1.0 - uu) * ties) / tot)
        if shadow_uniforms is not None:
            uu2 = shadow_uniforms[a : a + chunk].double()
            us2.append((above + uu2 * ties) / tot)
            ts2.append((below + (1.0 - uu2) * ties) / tot)
        if moments:
            s1, s2, q2, q3 = sums[3] / tot, sums[4] / tot, sums[5] / tot**2, sums[6] / tot**3
            log_tot = torch.log(tot)
            lpy = (zy - m).double() - log_tot  # log p(y)
            # E[log p] = s1 - log tot; Var(log p) = s2 - s1^2
            ent_dev = -lpy + (s1 - log_tot)  # -log p(y) - H
            ent_var = (s2 - s1 * s1).clamp(min=0)
            py = lpy.exp()
            ms.append(torch.stack([ent_dev, ent_var, py - q2, (q3 - q2 * q2).clamp(min=0)]))
    if not us:
        z = torch.zeros(0, dtype=torch.float64, device=logits.device)
        out = (z, z, torch.zeros(4, 0, dtype=torch.float64, device=logits.device)) if moments else (z, z)
        return out + ((z, z) if shadow_uniforms is not None else ())
    out = (torch.cat(us), torch.cat(ts)) + ((torch.cat(ms, dim=1),) if moments else ())
    if shadow_uniforms is not None:
        out = out + (torch.cat(us2), torch.cat(ts2))
    return out


def enabled() -> bool:
    return os.environ.get("THUNDERSYNC_LAW_CHECK") == "1"


# ---------------------------------------------------------------- tests (CPU, pure Python)


def _chi2_sf(x: float, k: int) -> float:
    """P(Chi2_k > x): regularized upper incomplete gamma Q(k/2, x/2)."""
    if x <= 0:
        return 1.0
    a, z = 0.5 * k, 0.5 * x
    gln = math.lgamma(a)
    if z < a + 1.0:  # series for P, then Q = 1 - P
        term = s = 1.0 / a
        ap = a
        for _ in range(10000):
            ap += 1.0
            term *= z / ap
            s += term
            if abs(term) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(-z + a * math.log(z) - gln))
    # continued fraction for Q (Lentz)
    b = z + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return min(1.0, math.exp(-z + a * math.log(z) - gln) * h)


def _gamma_cdf_sf(x: float, k: float) -> tuple[float, float]:
    """(P(G <= x), P(G > x)) for G ~ Gamma(k, 1), each tail computed directly (no 1 - p cancellation
    in the small tail). Regularized incomplete gamma: series for P below k + 1, continued fraction for
    Q above; for k > 1e4 (skewness < 0.02) the Wilson-Hilferty normal form of (G / k)^(1/3)."""
    if x <= 0:
        return 0.0, 1.0
    if k > 1e4:
        z = ((x / k) ** (1.0 / 3.0) - (1.0 - 1.0 / (9.0 * k))) / math.sqrt(1.0 / (9.0 * k))
        return 0.5 * math.erfc(-z / math.sqrt(2.0)), 0.5 * math.erfc(z / math.sqrt(2.0))
    a, z = float(k), float(x)
    log_pref = -z + a * math.log(z) - math.lgamma(a)
    if z < a + 1.0:
        term = s = 1.0 / a
        ap = a
        for _ in range(100000):
            ap += 1.0
            term *= z / ap
            s += term
            if abs(term) < abs(s) * 1e-16:
                break
        p = min(1.0, s * math.exp(log_pref))
        return p, max(0.0, 1.0 - p)
    b = z + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-16:
            break
    q = min(1.0, math.exp(log_pref) * h)
    return max(0.0, 1.0 - q), q


def _gamma_two_sided(x: float, k: float) -> float:
    """Two-sided p of x under Gamma(k, 1): min(1, 2 x the smaller tail)."""
    lo, hi = _gamma_cdf_sf(x, k)
    return min(1.0, 2.0 * min(lo, hi))


def _ks_sf(lam: float) -> float:
    """Kolmogorov distribution P(K > lam)."""
    if lam < 0.2:
        return 1.0
    s = 0.0
    for j in range(1, 200):
        term = 2.0 * (-1) ** (j - 1) * math.exp(-2.0 * j * j * lam * lam)
        s += term
        if abs(term) < 1e-17:
            break
    return min(1.0, max(0.0, s))


def _norm_two_sided(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def law_test(stats: Any) -> dict[str, float]:
    """p-values of the pooled stats ([N_STATS] numbers) under H0 (u ~ U(0, 1) i.i.d.)."""
    st = [float(x) for x in (stats.tolist() if hasattr(stats, "tolist") else stats)]
    counts, n = st[:N_BINS], st[N_BINS]
    out: dict[str, float] = {"n": n}
    if n < 1:
        return {**out, "p": 1.0, "chi2": 0.0, "p_chi2": 1.0, "ks_d": 0.0, "p_ks": 1.0,
                "z_mean": 0.0, "p_mean": 1.0, "z_tail": 0.0, "p_tail": 1.0}
    probs = bin_probs()
    # chi-square, merging bins (in order) until each has >= 5 expected counts
    # The leftover bins after the last group (expected < 5 in total) join that last group: before
    # ts-lawfix they formed a group of their own, so e.g. 2 bonus tokens at t <= 1e-4 among n = 328
    # (expected 0.03) added 118 to the chi-square (a spurious emission-bonus "FAIL");
    # the tail z-test already weighs those bins correctly.
    groups: list[list[float]] = []
    acc_o = acc_e = 0.0
    for o, p in zip(counts, probs):
        acc_o += o
        acc_e += p * n
        if acc_e >= 5.0:
            groups.append([acc_o, acc_e])
            acc_o = acc_e = 0.0
    if acc_e > 0 or acc_o > 0:
        if groups:
            groups[-1][0] += acc_o
            groups[-1][1] += acc_e
        else:
            groups.append([acc_o, acc_e])
    chi2 = sum((o - e) ** 2 / max(e, 1e-300) for o, e in groups)
    df = max(len(groups) - 1, 1)
    out["chi2"], out["chi2_df"], out["p_chi2"] = chi2, float(df), _chi2_sf(chi2, df)
    # binned KS: sup over bin edges of |empirical CDF - edge|
    cum, cdf, d = 0.0, 0.0, 0.0
    for o, p in zip(counts, probs):
        cum += o
        cdf += p
        d = max(d, abs(cum / n - cdf))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    out["ks_d"], out["p_ks"] = d, _ks_sf(lam)
    zm = (st[N_BINS + 1] / n - 0.5) / math.sqrt(1.0 / (12.0 * n))
    zt = (st[N_BINS + 2] / n - 1.0) * math.sqrt(n)
    out["z_mean"], out["p_mean"] = zm, _norm_two_sided(zm)
    # z_tail is reported; p_tail is exact: sum(-log t) ~ Gamma(n, 1) under H0 (t ~ U(0, 1)).
    out["z_tail"], out["p_tail"] = zt, _gamma_two_sided(st[N_BINS + 2], n)
    ps = [out["p_chi2"], out["p_ks"], out["p_mean"], out["p_tail"]]
    if len(st) >= N_BINS + 7 and st[N_BINS + 4] > 0 and st[N_BINS + 6] > 0:
        # Surprisal and probability-of-the-emitted-token z-tests (exact per-token mean and
        # variance under H0, summed): most sensitive to sharpness errors (temperature,
        # residual / pool choices that favour probable or improbable tokens).
        zs = st[N_BINS + 3] / math.sqrt(st[N_BINS + 4])
        zp = st[N_BINS + 5] / math.sqrt(st[N_BINS + 6])
        out["z_surprisal"], out["p_surprisal"] = zs, _norm_two_sided(zs)
        out["z_prob"], out["p_prob"] = zp, _norm_two_sided(zp)
        ps += [out["p_surprisal"], out["p_prob"]]
    out["p"] = min(1.0, len(ps) * min(ps))
    return out


def alpha() -> float:
    """Per-step false-alarm level of the verdict (THUNDERSYNC_LAW_ALPHA, default 1e-4)."""
    return float(os.environ.get("THUNDERSYNC_LAW_ALPHA", "1e-4"))


def history_metrics(stats: Any, prefix: str = "law/") -> dict[str, Any]:
    """Step metrics: p-values, n, and the raw stats (list, for run-level pooling)."""
    res = law_test(stats)
    out = {f"{prefix}{k}": float(v) for k, v in res.items()}
    out[f"{prefix}stats"] = [float(x) for x in (stats.tolist() if hasattr(stats, "tolist") else stats)]
    out[f"{prefix}fail"] = float(res["n"] > 0 and res["p"] < alpha())
    return out


def format_line(step: int, stats: Any) -> str:
    r = law_test(stats)
    return (
        f"[law check] step={step} n={int(r['n'])} p={r['p']:.3e} chi2={r.get('chi2', 0):.1f}"
        f"/{int(r.get('chi2_df', 0))} p_chi2={r['p_chi2']:.3e} ks_d={r['ks_d']:.2e} p_ks={r['p_ks']:.3e} "
        f"z_mean={r['z_mean']:.2f} z_tail={r['z_tail']:.2f} "
        f"z_surprisal={r.get('z_surprisal', float('nan')):.2f} z_prob={r.get('z_prob', float('nan')):.2f} "
        f"{'FAIL' if r['n'] > 0 and r['p'] < alpha() else 'ok'}"
    )


def sample_exact_pit(
    probs: torch.Tensor, y: torch.Tensor, generator: Optional[torch.Generator] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU helper (tests / offline): (u, t) from probability rows [N, V] and tokens [N]."""
    U = torch.rand(y.numel(), dtype=torch.float64, generator=generator)
    return pit_rows(torch.log(probs.double()).float(), y, U, moments=True)


# ------------------------------------------------- emission law (verifier-side, independent reference)


def reference_emission(lp_row: torch.Tensor, lq_row: Optional[torch.Tensor], lpx: torch.Tensor,
                       lqx: torch.Tensor, tau: int) -> torch.Tensor:
    """Law of the token block verification emits after keeping draft[:tau], written from the
    definition independently of block_verification.py (no shared code):
    b_i = min(1, b_{i-1} p(x_i)/q(x_i)), b_0 = 1  (closed form exp(S_i - max_{j<=i} S_j)),
    residual at tau < g: (b_tau p - q)+ (p if it is empty); bonus at tau = g: p. fp64, unnormalized.
    """
    s = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum((lpx - lqx).double().cpu(), 0)])
    b = torch.exp(s[tau] - s[: tau + 1].max())
    p = lp_row.double().cpu().exp()
    if lq_row is None:
        return p
    r = (b * p - lq_row.double().cpu().exp()).clamp(min=0)
    return r if float(r.sum()) > 0 else p


def emission_pit(dist: torch.Tensor, y: int, uniform: float) -> tuple[float, float]:
    """Randomized PIT of y under ``dist`` (unnormalized, ranked by dist): (u, 1 - u)."""
    d = dist.double()
    tot = float(d.sum())
    dy = float(d[y])
    above = float(d[d > dy].sum())
    ties = float(d[d == dy].sum())
    below = float(d[d < dy].sum())
    return (above + uniform * ties) / tot, (below + (1.0 - uniform) * ties) / tot


def reference_null_emission(lp_row: torch.Tensor, lq_row: Optional[torch.Tensor], lpx: torch.Tensor,
                            lqx: torch.Tensor, tau: int, m: float) -> torch.Tensor:
    """⊥-BV (null-augmented block verification) emission law at envelope M given (draft, tau), from the
    definition, independent of block_verification.py: target of the first position p/M, so
    S_i = sum_{j<i} log p/q - log M for i >= 1, b_i = exp(S_i - max_{j<=i} S_j) (S_0 = 0);
    tau = g: p (bonus); tau = 0 < g: [(p/M - q)+, 1 - 1/M] (last entry = BOT, the draft yields no output);
    0 < tau < g: (b_tau p - q)+ (p if empty). fp64 unnormalized, length V (+1 with the BOT column at tau 0).
    """
    g = lpx.numel()
    p = lp_row.double().cpu().exp()
    if tau == g:
        return p
    q = lq_row.double().cpu().exp()
    if tau == 0:
        r = (p / m - q).clamp(min=0)
        return torch.cat([r, r.new_full((1,), 1.0 - 1.0 / m)])
    s = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum((lpx - lqx).double().cpu(), 0)])
    s[1:] -= math.log(m)
    b = torch.exp(s[tau] - s[: tau + 1].max())
    r = (b * p - q).clamp(min=0)
    return r if float(r.sum()) > 0 else p


# Driver-side collector of ⊥-BV emission PITs at the envelope each row was decided at (speculative._null_result).
_NULL_EMIT: list[tuple[float, float]] = []


def record_null_emission(pit) -> None:
    if pit is not None:
        _NULL_EMIT.append((float(pit[0]), float(pit[1])))


def pop_null_emission() -> Optional[torch.Tensor]:
    if not _NULL_EMIT:
        return None
    u = torch.tensor([a for a, _ in _NULL_EMIT], dtype=torch.float64)
    t = torch.tensor([b for _, b in _NULL_EMIT], dtype=torch.float64)
    _NULL_EMIT.clear()
    return accumulate(u, t)
