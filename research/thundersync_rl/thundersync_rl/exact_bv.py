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
"""Block verification that is exact BY CONSTRUCTION under integer laws (``engine_sampler: exact_int``).

p_i(v) = wp_i(v) / Wp_i and q_i(v) = wq_i(v) / Wq_i are the integer laws of the target and
drafter rows (``nemo_rl.models.megatron.exact_int``). Block verification (Sun et al. 2024,
``block_verification.py``) is a law-level identity for ANY pair of laws; with integer laws
every quantity it uses is a rational number:

    rho_j = p_j(x_j) / q_j(x_j) = wp_j(x_j) Wq_j / (wq_j(x_j) Wp_j)
    b_i   = P_i / max(M, max_{1<=j<=i} P_j),   P_i = prod_{j<i} rho_j  (M = 1: plain BV)
    r_i   = sum_v (b_i p_i(v) - q_i(v))+       (i < g)
    h_i   = r_i / (r_i + 1 - b_i)  (1 if the denominator is 0),  h_g = b_g
    tau   = max{i : eta_i <= h_i},  eta_i ~ U(0, 1) independent
    y ~ p_g (tau = g)  |  y ~ (b_tau p_tau - q_tau)+  (+ the BOT atom 1 - 1/M at the ⊥-BV root)

Here every random decision is EXACT:
  * eta_i <= a (a rational) is an exact Bernoulli(a): eta's binary expansion comes from keyed
    words (53 bits per block); a fast fp64 interval [a_lo, a_hi] with a certified error bound
    decides it unless eta's first block straddles the interval (probability ~ the bound,
    1e-12 or less with fp64 engine log q, ~1e-4 per long draft with fp32-logged log q); then
    a slow path computes a in exact integer interval arithmetic (``_Fixed``: 2^-F fixed point
    with directed rounding, F doubling) and extends eta block by block until decided
    (terminates with probability 1; the result is the exact Bernoulli).
  * the residual draw is exact rejection from an integer proposal G_v >= 2^60 (b_hi p_v - q_v)+
    (b_hi a certified upper bound on b): y ~ G exactly (integer sampler), accepted with
    probability 2^60 rho_y / G_y (exact Bernoulli). P(y) is proportional to rho_y exactly.
  * the bonus draw is the exact integer sampler on wp_g.

So the output law is EXACTLY p (enumerated with Fractions in test_exact_bv.py, given the
integer q law the drafts were sampled from), with no floating-point term anywhere in the law.
Floating point only decides how fast a decision is made, never which way.

Inputs mirror ``block_verify_stats_lazy`` / ``_grid``: p_rows(idx) / q_rows(idx) return fp32
log-softmax rows; ``lqx`` = the engine's logged lambda_q of the drafted tokens (fp32 or fp64;
used only by the fast path, never by the law). Randomness: one int64 key per call from the
caller's generator (one ``randint``), every uniform is a keyed word of (key, position, counter,
purpose), so the outcome does not depend on how lazily rows were evaluated.
"""

from __future__ import annotations

import math
import os
from fractions import Fraction
from typing import Callable, Optional, Union

import numpy as np
import torch

from nemo_rl.models.megatron import exact_int as xi

NULL_TOKEN = -1
K_PROP = 60  # proposal scale 2^60 (sum of proposal weights < 2^61)
WIDEN = 2.0**-49  # relative widening of fp64 quantities (>= 8 ulp)
RES_BATCH = 8  # residual candidates per rejection round
# Tests: skip the fp64 fast path (every decision through the exact slow path).
FORCE_SLOW = False
_COUNTS = {"decisions": 0, "slow": 0, "slow_blocks": 0, "residual_rounds": 0, "residual_draws": 0}


def counts() -> dict[str, int]:
    return dict(_COUNTS)


class NonFiniteVerification(RuntimeError):
    pass


def _law_bug() -> str:
    return os.environ.get("THUNDERSYNC_DEBUG_LAW_BUG", "").split(":")[0]


# --------------------------------------------------------------------------
# Row sources and integer statistics
# --------------------------------------------------------------------------


class Rows:
    """fp32 log-prob rows by position (a tensor [n, V] or a callable idx -> [k, V]),
    with cached integer statistics."""

    def __init__(self, rows: Union[torch.Tensor, Callable], chunk: int = 256):
        self._t = rows if isinstance(rows, torch.Tensor) else None
        self._f = None if self._t is not None else rows
        self.chunk = chunk
        self._w: dict[int, torch.Tensor] = {}
        self.evaluated = 0

    def rows(self, idx: torch.Tensor) -> torch.Tensor:
        self.evaluated += int(idx.numel())
        if self._t is not None:
            return self._t[idx.to(self._t.device)]
        return self._f(idx)

    def stats(self, idx: torch.Tensor, tokens: Optional[torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        Ws, wxs = [], []
        for a in range(0, idx.numel(), self.chunk):
            sub = idx[a : a + self.chunk]
            rows = self.rows(sub)
            W, wx = xi.row_stats(rows, None if tokens is None else tokens[a : a + self.chunk].to(rows.device))
            Ws.append(W)
            if wx is not None:
                wxs.append(wx)
        W = torch.cat(Ws) if Ws else torch.zeros(0, dtype=torch.int64)
        return W, (torch.cat(wxs) if wxs else (None if tokens is None else torch.zeros(0, dtype=torch.int64)))

    def weights(self, i: int, device=None) -> torch.Tensor:
        if i not in self._w:
            idx = torch.tensor([i])
            if self._t is not None:
                idx = idx.to(self._t.device)
            self._w[i] = xi.int_weights(self.rows(idx)[0].float())
        return self._w[i]

    def put_weights(self, i: int, w: torch.Tensor) -> None:
        self._w[i] = w


# --------------------------------------------------------------------------
# Exact interval arithmetic (fixed point 2^-F, directed rounding)
# --------------------------------------------------------------------------


def _cdiv(a: int, b: int) -> int:
    return -((-a) // b)


class _Exact:
    """Exact rationals of one draft: the products P_i as 2^-F fixed-point intervals."""

    def __init__(self, wpx: list[int], Wp: list[int], wqx: list[int], Wq: list[int], M: Fraction):
        self.wpx, self.Wp, self.wqx, self.Wq, self.M = wpx, Wp, wqx, Wq, M
        self._F = None

    def b_interval(self, i: int, F: int) -> tuple[Fraction, Fraction]:
        """[b_lo, b_hi] containing the exact b_i (directed rounding at 2^-F)."""
        one = 1 << F
        if i == 0:
            return Fraction(1), Fraction(1)
        Plo, Phi = one, one
        mx_lo = mx_hi = None
        Mlo = (self.M.numerator << F) // self.M.denominator
        Mhi = _cdiv(self.M.numerator << F, self.M.denominator)
        for j in range(i):
            num = self.wpx[j] * self.Wq[j]
            den = self.wqx[j] * self.Wp[j]
            if den == 0:
                raise NonFiniteVerification(f"q(x_{j}) = 0 under the q rows' integer law")
            Plo = (Plo * num) // den
            Phi = _cdiv(Phi * num, den)
            mx_lo = Plo if mx_lo is None else max(mx_lo, Plo)
            mx_hi = Phi if mx_hi is None else max(mx_hi, Phi)
        # b_i = P_i / max(M, max_{1<=j<=i} P_j)
        d_lo, d_hi = max(Mlo, mx_lo), max(Mhi, mx_hi)
        if Phi == 0:
            return Fraction(0), Fraction(0)
        b_lo = Fraction(Plo, d_hi) if d_hi > 0 else Fraction(0)
        b_hi = min(Fraction(1), Fraction(Phi, d_lo)) if d_lo > 0 else Fraction(1)
        return b_lo, b_hi


def _residual_mass_exact(b: Fraction, wp: torch.Tensor, wq: torch.Tensor, Wp: int, Wq: int) -> Fraction:
    """Exact r(b) = sum_v (b wp_v / Wp - wq_v / Wq)+ (set classified in fp64, near-ties exactly)."""
    if b == 0:
        return Fraction(0)
    bf = float(b)
    p = wp.double() / float(Wp)
    q = wq.double() / float(Wq)
    if bf == 0.0 or not math.isfinite(bf):
        amb = torch.ones_like(wp, dtype=torch.bool)
        sure_in = torch.zeros_like(amb)
    else:
        lhs = bf * p
        sure_in = (wp > 0) & (lhs > q * (1 + 2.0**-44))
        sure_out = (wp == 0) | (lhs * (1 + 2.0**-44) < q)
        amb = ~sure_in & ~sure_out
    A = int(torch.where(sure_in, wp, torch.zeros_like(wp)).sum())
    B = int(torch.where(sure_in, wq, torch.zeros_like(wq)).sum())
    if bool(amb.any()):
        ids = torch.nonzero(amb).flatten().tolist()
        wpl = wp[amb].tolist()
        wql = wq[amb].tolist()
        for _v, a_, c_ in zip(ids, wpl, wql):
            if b.numerator * a_ * Wq > b.denominator * c_ * Wp:
                A += a_
                B += c_
    return b * Fraction(A, Wp) - Fraction(B, Wq)


def _h_of(r: Fraction, b: Fraction) -> Fraction:
    den = r + 1 - b
    return Fraction(1) if den == 0 else r / den


def _eta_block_interval(words: Callable[[int], int], m: int) -> tuple[int, int]:
    """eta in [K / 2^(53 m), (K + 1) / 2^(53 m)) from the first m 53-bit blocks."""
    K = 0
    for k in range(m):
        K = (K << 53) | (words(k) >> 11)
    return K, 53 * m


def exact_bernoulli(
    words: Callable[[int], int],
    fast: tuple[float, float],
    exact: Callable[[int], tuple[Fraction, Fraction]],
) -> bool:
    """Exact Bernoulli(a): returns eta <= a, eta's 53-bit blocks from ``words(k)``.

    ``fast`` = certified fp64 bounds (a_lo, a_hi) with a_lo <= a <= a_hi; ``exact(F)`` returns
    Fraction bounds [lo, hi] of a at fixed-point precision 2^-F (exact as F grows). P(eta = a)
    = 0, so the loop ends with probability 1, and the answer is the event eta <= a itself.
    """
    _COUNTS["decisions"] += 1
    k0 = words(0) >> 11
    lo, hi = fast
    if FORCE_SLOW:
        lo, hi = -1.0, 2.0
    if (k0 + 1) * 2.0**-53 <= lo:  # exact in fp64: (k0 + 1) <= 2^53
        return True
    if k0 * 2.0**-53 >= hi:
        return False
    _COUNTS["slow"] += 1
    m, F = 1, 128
    while True:
        a_lo, a_hi = exact(F)
        K, bits = _eta_block_interval(words, m)
        e_lo, e_hi = Fraction(K, 1 << bits), Fraction(K + 1, 1 << bits)
        if e_hi <= a_lo:
            return True
        if e_lo >= a_hi:
            return False
        if Fraction(1, 1 << bits) > (a_hi - a_lo):
            m += 1
            _COUNTS["slow_blocks"] += 1
        else:
            F *= 2
        if F > 1 << 20:
            raise RuntimeError("exact_bernoulli: precision runaway")


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def _m_fraction(M: float) -> Fraction:
    return Fraction(*float(M).as_integer_ratio())


class _Verify:
    """State of one draft's exact verification (shared by every envelope M)."""

    def __init__(self, p_rows, q_rows, draft: torch.Tensor, lqx: Optional[torch.Tensor], key: int):
        self.P = p_rows if isinstance(p_rows, Rows) else Rows(p_rows)
        self.Q = q_rows if isinstance(q_rows, Rows) else Rows(q_rows)
        self.draft = draft
        g = self.g = int(draft.numel())
        self.key = int(key)
        dev = draft.device
        ar = torch.arange(g + 1, device=dev)
        tok = torch.cat([draft, draft.new_zeros(1)])
        Wp, wpx = self.P.stats(ar, tok)
        self.Wp_t, self.wpx_t = Wp, wpx[:g]
        if bool((Wp <= 0).any()):
            raise NonFiniteVerification("a target row has no integer mass (non-finite logits)")
        self.Wp = Wp.tolist()
        self.wpx = self.wpx_t.tolist()
        self.lam_p = xi.lam(self.wpx_t, Wp[:g]).to(dev)  # fp64, -inf where p(x) = 0
        if lqx is None:
            Wq, wqx = self.Q.stats(torch.arange(g, device=dev), draft)
            self._qx = (wqx.tolist(), Wq.tolist())
            self.lam_q = xi.lam(wqx, Wq).to(dev)
            eps_q = 2.0**-51
        else:
            self._qx = None
            self.lam_q = lqx.to(dev).double()
            eps_q = 2.0**-24 if lqx.dtype in (torch.float32, torch.bfloat16, torch.float16) else 2.0**-51
            if lqx.dtype in (torch.bfloat16, torch.float16):
                eps_q = 2.0**-7
        if g and not bool(torch.isfinite(self.lam_q).all()):
            raise NonFiniteVerification(
                "log q of a drafted token is not finite: the engine sampled a token of weight 0 "
                "under its logged integer law"
            )
        # Per-position error of d_j = lam_p_j - lam_q_j (lam_p is the fp64 formula on exact
        # integers; lam_q the logged value): |log w| + |log W| <= 2 x 37 nats at 2^52.
        lp_fin = torch.where(torch.isfinite(self.lam_p), self.lam_p, torch.zeros_like(self.lam_p))
        self.err = lp_fin.abs() * 2.0**-51 + self.lam_q.abs() * eps_q + 2.0**-44
        self.zero_from = None
        nz = torch.nonzero(self.wpx_t == 0).flatten()
        if nz.numel():
            self.zero_from = int(nz[0])  # p(x_j) = 0 exactly: b_i = 0 for i > j
        self._eta_k0 = None
        self._exact: dict[Fraction, _Exact] = {}

    # -- uniforms ---------------------------------------------------------
    def eta_words(self, i: int) -> Callable[[int], int]:
        return lambda k: xi.word_py(self.key, i, 0, xi.ETA, k)

    def eta_k0(self) -> np.ndarray:
        if self._eta_k0 is None:
            ar = np.arange(self.g + 1, dtype=np.int64)
            self._eta_k0 = xi.eta53(xi.words_np(self.key, ar, 0, xi.ETA, 0))
        return self._eta_k0

    # -- weights with certified bounds -----------------------------------
    def b_bounds(self, M: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(b_mid, b_lo, b_hi) [g+1] fp64 with b_lo <= b_i <= b_hi certified."""
        g = self.g
        dev = self.lam_p.device
        d = self.lam_p - self.lam_q
        zero = torch.zeros(1, dtype=torch.float64, device=dev)
        S = torch.cat([zero, torch.cumsum(torch.where(torch.isfinite(d), d, torch.zeros_like(d)), 0)])
        logM = math.log(M) if M != 1.0 else 0.0
        if M != 1.0:
            S[1:] -= logM
        cmS = torch.cummax(S, 0).values
        L = S - cmS
        E = torch.cat([zero, torch.cumsum(self.err, 0)])
        E = E + (torch.arange(g + 1, device=dev, dtype=torch.float64) + 1) * 2.0**-52 * torch.cummax(S.abs(), 0).values
        E = E + abs(logM) * 2.0**-51
        tot = 2 * E + L.abs() * 2.0**-51
        b = L.exp()
        b_lo = (L - tot).exp() * (1 - WIDEN)
        b_hi = torch.clamp((L + tot).exp() * (1 + WIDEN), max=1.0)
        if self.zero_from is not None:
            z = torch.arange(g + 1, device=dev) > self.zero_from
            b = torch.where(z, torch.zeros_like(b), b)
            b_lo = torch.where(z, torch.zeros_like(b_lo), b_lo)
            b_hi = torch.where(z, torch.zeros_like(b_hi), b_hi)
        b_hi[0] = b_lo[0] = b[0] = 1.0
        return b, b_lo, b_hi

    # -- exact slow path --------------------------------------------------
    def exact_ctx(self, M: float, upto: int) -> _Exact:
        """Exact integer data of positions < upto (q stats from rows when not yet known)."""
        if self._qx is None or len(self._qx[0]) < upto:
            idx = torch.arange(self.g, device=self.draft.device)
            Wq, wqx = self.Q.stats(idx, self.draft)
            self._qx = (wqx.tolist(), Wq.tolist())
            lam_q_rows = xi.lam(wqx, Wq).to(self.lam_q.device)
            # The logged lambda_q must be the rows' lambda (to the logged precision).
            bad = (lam_q_rows - self.lam_q).abs() > self.lam_q.abs() * 2.0**-23 + 2.0**-40
            if bool(bad.any()):
                raise NonFiniteVerification(
                    f"logged log q differs from the q rows' integer law at positions "
                    f"{torch.nonzero(bad).flatten()[:8].tolist()}"
                )
        Mf = _m_fraction(M)
        ex = self._exact.get(Mf)
        if ex is None:
            ex = self._exact[Mf] = _Exact(self.wpx, self.Wp, self._qx[0], self._qx[1], Mf)
        return ex

    def b_exact(self, M: float, i: int) -> Callable[[int], tuple[Fraction, Fraction]]:
        def f(F: int):
            return self.exact_ctx(M, i).b_interval(i, F)

        return f

    def h_exact(self, M: float, i: int) -> Callable[[int], tuple[Fraction, Fraction]]:
        def f(F: int):
            b_lo, b_hi = self.exact_ctx(M, i).b_interval(i, F)
            wp = self.P.weights(i)
            wq = self.Q.weights(i)
            Wp, Wq = self.Wp[i], int(self._qx[1][i]) if self._qx is not None else int(wq.sum())
            r_lo = _residual_mass_exact(b_lo, wp, wq, Wp, Wq)
            r_hi = _residual_mass_exact(b_hi, wp, wq, Wp, Wq)
            return _h_of(r_lo, b_lo), _h_of(r_hi, b_hi)

        return f

    # -- fast residual masses --------------------------------------------
    def h_bounds_vec(self, idx: np.ndarray, b, b_lo, b_hi) -> tuple[np.ndarray, np.ndarray]:
        """Certified fp64 (h_lo, h_hi) at the positions ``idx`` (one batched row fetch and one
        Triton residual pass per row; same bounds as ``h_bounds``). Rows are not cached."""
        dev = self.lam_p.device
        it = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=dev)
        P = self.P.rows(it).float()
        Q = self.Q.rows(it).float()
        rd = P.device
        itr = it.to(rd)
        bm = b.to(rd)[itr]
        bl = b_lo.to(rd)[itr]
        bh = b_hi.to(rd)[itr]
        _, r = xi.residual_mass(P, Q, self.Wp_t.to(rd)[itr], bm)
        V = P.shape[-1]
        del P, Q
        er = (V + 4) * 2.0**-51 * (bm + 1.0)
        r_lo = torch.clamp(r - (bm - bl) - er, min=0.0)
        r_hi = r + (bh - bm) + er
        den_lo = r_lo + 1 - bl
        h_lo = torch.where(den_lo <= 0, torch.zeros_like(r_lo), r_lo / torch.where(den_lo <= 0, torch.ones_like(den_lo), den_lo))
        den_hi = r_hi + 1 - bh
        h_hi = torch.where(
            den_hi <= 0, torch.ones_like(r_hi),
            torch.clamp(r_hi / torch.where(den_hi <= 0, torch.ones_like(den_hi), den_hi), max=1.0),
        )
        h_lo = h_lo * (1 - WIDEN)
        h_hi = torch.clamp(h_hi * (1 + WIDEN), max=1.0)
        out = torch.stack([h_lo, h_hi]).cpu().numpy()
        return out[0], out[1]

    def h_bounds(self, idx: list[int], b, b_lo, b_hi) -> list[tuple[float, float]]:
        out = []
        if not idx:
            return out
        dev = self.lam_p.device
        it = torch.tensor(idx, device=dev)
        P = self.P.rows(it)
        Q = self.Q.rows(it)
        for k, i in enumerate(idx):
            wp = xi.int_weights(P[k].float())
            wq = xi.int_weights(Q[k].float())
            self.P.put_weights(i, wp)
            self.Q.put_weights(i, wq)
            Wp, Wq = float(self.Wp[i]), float(wq.sum())
            bm, bl, bh = float(b[i]), float(b_lo[i]), float(b_hi[i])
            r = float(torch.clamp(bm * (wp.double() / Wp) - wq.double() / Wq, min=0).sum())
            V = wp.numel()
            er = (V + 4) * 2.0**-51 * (bm + 1.0)
            r_lo = max(0.0, r - (bm - bl) - er)
            r_hi = r + (bh - bm) + er
            den_lo = r_lo + 1 - bl
            h_lo = 0.0 if den_lo <= 0 else r_lo / den_lo
            den_hi = r_hi + 1 - bh
            h_hi = 1.0 if den_hi <= 0 else min(1.0, r_hi / den_hi)
            out.append((h_lo * (1 - WIDEN), min(1.0, h_hi * (1 + WIDEN))))
        return out

    # -- stop index -------------------------------------------------------
    def stop_index(self, M: float, batch: int = 16, max_chunk: int = 256) -> tuple[int, tuple]:
        """tau = max{i : eta_i <= h_i}: candidates from the top in chunks (16, 32, ... 256), the
        chunk's h bounds in one batched pass; positions whose eta's first block is surely above h_hi
        are decided False at once (exactly ``exact_bernoulli``'s first test), the rest in order
        through ``exact_bernoulli``. Outcome identical to the per-position scan (``stop_index_ref``)."""
        g = self.g
        b, b_lo, b_hi = self.b_bounds(M)
        bl_h, bh_h = b_lo.cpu().numpy(), b_hi.cpu().numpy()
        if exact_bernoulli(self.eta_words(g), (float(bl_h[g]), float(bh_h[g])), self.b_exact(M, g)):
            return g, (b, b_lo, b_hi)
        k0 = self.eta_k0()
        cand = np.nonzero(k0[:g] * 2.0**-53 < bh_h[:g])[0][::-1]
        done = 0
        n = batch
        while done < len(cand):
            chunk = cand[done : done + n]
            pos = chunk[chunk > 0]
            if pos.size:
                h_lo, h_hi = self.h_bounds_vec(pos, b, b_lo, b_hi)
                if FORCE_SLOW:
                    open_ = np.arange(pos.size)
                else:
                    open_ = np.nonzero(k0[pos] * 2.0**-53 < h_hi)[0]
                    _COUNTS["decisions"] += int(pos.size - open_.size)
                for j in open_.tolist():
                    i = int(pos[j])
                    if exact_bernoulli(self.eta_words(i), (float(h_lo[j]), float(h_hi[j])), self.h_exact(M, i)):
                        return i, (b, b_lo, b_hi)
            if chunk.size and int(chunk[-1]) == 0:
                return 0, (b, b_lo, b_hi)  # h_0 = 1 (b_0 = 1): eta_0 <= 1 always
            done += len(chunk)
            n = min(2 * n, max_chunk)
        return 0, (b, b_lo, b_hi)

    def stop_index_ref(self, M: float, batch: int = 4) -> tuple[int, tuple]:
        """The per-position scan (reference for tests; ``stop_index`` is the batched one)."""
        g = self.g
        b, b_lo, b_hi = self.b_bounds(M)
        bl_h, bh_h = b_lo.cpu().numpy(), b_hi.cpu().numpy()
        if exact_bernoulli(self.eta_words(g), (float(bl_h[g]), float(bh_h[g])), self.b_exact(M, g)):
            return g, (b, b_lo, b_hi)
        k0 = self.eta_k0()
        # eta_i <= h_i <= b_i: positions whose eta's first block is surely above b_hi never stop.
        cand = np.nonzero(k0[:g] * 2.0**-53 < bh_h[:g])[0][::-1].tolist()
        done = 0
        n = batch
        while done < len(cand):
            chunk = cand[done : done + n]
            hb = self.h_bounds([i for i in chunk if i > 0], b, b_lo, b_hi)
            hb_iter = iter(hb)
            for i in chunk:
                if i == 0:
                    return 0, (b, b_lo, b_hi)  # h_0 = 1 (b_0 = 1): eta_0 <= 1 always
                fast = next(hb_iter)
                if exact_bernoulli(self.eta_words(i), fast, self.h_exact(M, i)):
                    return i, (b, b_lo, b_hi)
            done += len(chunk)
            n *= 2
        return 0, (b, b_lo, b_hi)

    # -- emissions ----------------------------------------------------------
    def bonus(self) -> int:
        w = self.P.weights(self.g)
        return int(xi.sample_weights(w.view(1, -1), self.key, self.g, 0, xi.BONUS)[0])

    def residual(self, M: float, tau: int, bounds) -> int:
        """Exact y ~ (b_tau p_tau - q_tau)+ (+ BOT 1 - 1/M at the ⊥-BV root): rejection
        from the integer proposal G (>= 2^60 x the residual), exact acceptance."""
        b, b_lo, b_hi = bounds
        wp = self.P.weights(tau)
        wq = self.Q.weights(tau)
        Wp = self.Wp[tau]
        Wq = int(wq.sum())
        if _law_bug() == "residual_from_p":
            # Debug negative control: a deliberately WRONG residual law (p).
            return int(xi.sample_weights(wp.view(1, -1), self.key, tau, 0, xi.RESID)[0])
        root = tau == 0 and M != 1.0
        if root and _law_bug() == "null_m_1":
            # Debug negative control (as block_verification_fused): the ⊥-BV root residual
            # without the envelope -- (p - q)+, no BOT column.
            return self.residual(1.0, tau, bounds)
        Mf = _m_fraction(M)
        if root:
            be_lo = be_hi = None
            bl_f = float(Fraction(1) / Mf) * (1 - 2.0**-52)
            bh_f = float(Fraction(1) / Mf) * (1 + 2.0**-52)
        else:
            bl_f, bh_f = float(b_lo[tau]), float(b_hi[tau])
        p = wp.double() / float(Wp)
        q = wq.double() / float(Wq)
        U = bh_f * p - q
        U = U + WIDEN * (bh_f * p + q)
        G = torch.ceil(torch.clamp(U, min=0) * 2.0**K_PROP).to(torch.int64)
        if root:
            g_bot = _cdiv((Mf.numerator - Mf.denominator) << K_PROP, Mf.numerator)
            G = torch.cat([G, G.new_full((1,), g_bot)])
        if int(G.sum()) <= 0:
            # Empty residual: unreachable (b_tau = 1 and p_tau = q_tau force b_{tau+1} = 1, so
            # h_{tau+1} = 1 and tau is never the stop index); mirrors emit_distribution's p fallback.
            return int(xi.sample_weights(wp.view(1, -1), self.key, tau, 0, xi.RESID)[0])
        V = wp.numel()
        cum = torch.cumsum(G, 0)
        Gs = int(cum[-1])
        rnd = 0
        while True:
            _COUNTS["residual_rounds"] += 1
            j = np.arange(RES_BATCH, dtype=np.int64) + rnd * RES_BATCH
            # candidate y_j ~ G exactly (rejection to [0, Gs), prefix sums)
            ys = []
            for jj in j.tolist():
                r, _ = xi.uniform_below(Gs, lambda k, jj=jj: xi.word_py(self.key, tau, jj, xi.RESID, k))
                ys.append(r)
            yt = torch.searchsorted(cum, torch.tensor(ys, device=cum.device).view(-1, 1), right=True)[:, 0].tolist()
            for jj, y in zip(j.tolist(), yt):
                _COUNTS["residual_draws"] += 1
                Gy = int(G[y])
                words = lambda k, jj=jj: xi.word_py(self.key, tau, jj, xi.RESACC, k)
                if root and y == V:
                    a = Fraction((Mf.numerator - Mf.denominator) << K_PROP, Mf.numerator * Gy)
                    fast = (float(a) * (1 - 2.0**-50), min(1.0, float(a) * (1 + 2.0**-50)))
                    if exact_bernoulli(words, fast, lambda F, a=a: (a, a)):
                        return NULL_TOKEN
                    continue
                py, qy = float(p[y]), float(q[y])
                wid = WIDEN * (bh_f * py + qy)
                lo = max(0.0, bl_f * py - qy - wid)
                hi = max(0.0, bh_f * py - qy + wid)
                scale = 2.0**K_PROP / Gy
                fast = (lo * scale * (1 - 2.0**-50), min(1.0, hi * scale * (1 + 2.0**-50)))

                def exact(F, y=y, Gy=Gy):
                    if root:
                        bl = bh = Fraction(1) / Mf
                    else:
                        bl, bh = self.exact_ctx(M, tau).b_interval(tau, F)
                    pf = Fraction(int(wp[y]), Wp)
                    qf = Fraction(int(wq[y]), Wq)
                    s = Fraction(1 << K_PROP, Gy)
                    return max(Fraction(0), bl * pf - qf) * s, max(Fraction(0), bh * pf - qf) * s

                if exact_bernoulli(words, fast, exact):
                    return int(y)
            rnd += 1
            if rnd > 1 << 16:
                raise RuntimeError("exact residual: rejection did not terminate")

    def emit(self, M: float, tau: int, bounds) -> int:
        if tau == self.g:
            return self.bonus()
        return self.residual(M, tau, bounds)

    def lam_p_of(self, i: int, y: int) -> float:
        # The same device op as every other lambda (xi.lam), so the value is bitwise the learner's.
        w = self.P.weights(i)
        return float(xi.lam(w[y : y + 1], self.Wp_t[i : i + 1].to(w.device))[0])


def _key(generator: Optional[torch.Generator], device) -> int:
    dev = generator.device if generator is not None else device
    return int(torch.randint(0, 2**62, (1,), generator=generator, device=dev))


def verify_exact(
    p_rows,
    q_rows,
    draft: torch.Tensor,
    lqx: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    null_m: float = 1.0,
    key: Optional[int] = None,
) -> tuple[int, int, torch.Tensor, int]:
    """Exact block verification of one draft under integer laws.

    ``p_rows``: [g+1, V] fp32 log-prob rows or idx -> rows; ``q_rows``: [g, V] / idx -> rows;
    ``lqx``: [g] logged lambda_q (None: from the q rows). Returns (tau, y, logp, rows evaluated)
    with logp = fp32 lambda_p of draft[:tau] + [y] (y = ``NULL_TOKEN``: BOT, logp = draft[:tau]).
    """
    g = int(draft.numel())
    if null_m < 1.0 or (null_m != 1.0 and g < 1):
        raise ValueError("null_m must be >= 1 and ⊥-BV needs a draft")
    v = _Verify(p_rows, q_rows, draft, lqx, _key(generator, draft.device) if key is None else key)
    tau, bounds = v.stop_index(null_m)
    y = v.emit(null_m, tau, bounds)
    lp = v.lam_p[:tau].float()
    if y >= 0:
        lp = torch.cat([lp, torch.tensor([v.lam_p_of(tau, y)], dtype=torch.float64, device=lp.device).float()])
    return tau, y, lp, v.P.evaluated


def verify_exact_grid(
    p_rows,
    q_rows,
    draft: torch.Tensor,
    lqx: Optional[torch.Tensor],
    generator: Optional[torch.Generator],
    envelopes: tuple[float, ...],
    key: Optional[int] = None,
) -> tuple[dict[float, tuple[int, int, Optional[float]]], torch.Tensor, int]:
    """``verify_exact`` at every ⊥-BV envelope M, one key (shared eta and residual words).

    Returns ({M: (tau, y, fp32 lambda_p(y) or None for BOT)}, lambda_p of the drafted tokens
    (fp32 [g]), rows evaluated). Marginally in M each outcome is the exact M verifier.
    """
    v = _Verify(p_rows, q_rows, draft, lqx, _key(generator, draft.device) if key is None else key)
    out: dict[float, tuple[int, int, Optional[float]]] = {}
    for M in envelopes:
        if M < 1.0:
            raise ValueError("null_m must be >= 1")
        tau, bounds = v.stop_index(M)
        y = v.emit(M, tau, bounds)
        out[M] = (tau, y, None if y < 0 else float(torch.tensor(v.lam_p_of(tau, y), dtype=torch.float64).float()))
    return out, v.lam_p.float(), v.P.evaluated


def p_token_lambda(p_rows, draft: torch.Tensor) -> torch.Tensor:
    """fp32 lambda_p of the drafted tokens (rows 0..g-1)."""
    P = p_rows if isinstance(p_rows, Rows) else Rows(p_rows)
    g = int(draft.numel())
    W, wx = P.stats(torch.arange(g, device=draft.device), draft)
    return xi.lam(wx, W).float()


# --------------------------------------------------------------------------
# Exact laws (Fractions) for tests: the BV output law under integer laws
# --------------------------------------------------------------------------


def output_law_fraction(
    wp: list[list[int]], wq: list[list[int]], draft: list[int], M: Fraction = Fraction(1)
) -> dict[tuple[int, int], Fraction]:
    """Exact law of (tau, y) for a fixed draft (y = -1: BOT), every quantity a Fraction."""
    g = len(draft)
    p = [[Fraction(x, sum(r)) for x in r] for r in wp]
    q = [[Fraction(x, sum(r)) for x in r] for r in wq]
    P = [Fraction(1)]
    for j in range(g):
        px, qx = p[j][draft[j]], q[j][draft[j]]
        P.append(P[-1] * px / qx)
    b = [Fraction(1)]
    for i in range(1, g + 1):
        b.append(P[i] / max([M] + P[1 : i + 1]))
    V = len(wp[0])
    h = []
    res = []
    for i in range(g):
        bi = b[i] / M if (i == 0 and M != 1) else b[i]
        rv = [max(Fraction(0), bi * p[i][v] - q[i][v]) for v in range(V)]
        if i == 0 and M != 1:
            rv.append(1 - 1 / M)
        r = sum(rv)
        res.append(rv)
        h.append(_h_of(r, b[i]) if not (i == 0 and M != 1) else Fraction(1))
    h.append(b[g])
    out: dict[tuple[int, int], Fraction] = {}
    for i in range(g + 1):
        pt = h[i]
        for j in range(i + 1, g + 1):
            pt *= 1 - h[j]
        if pt == 0:
            continue
        if i == g:
            dist = p[g]
        else:
            dist = res[i]
            if sum(dist) == 0:
                dist = p[i]
        s = sum(dist)
        for v, x in enumerate(dist):
            if x:
                y = -1 if (v == V) else v
                out[(i, y)] = out.get((i, y), Fraction(0)) + pt * x / s
    return out
