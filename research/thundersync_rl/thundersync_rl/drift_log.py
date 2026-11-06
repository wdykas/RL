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
"""Drift logging for long runs (C60 drift phase-in). Read-only: no exactness impact.

Three groups of per-step metrics (theory/discussion/proposals/C60-drift-phasein.md §8):

1. **Per-age divergence moments** (driver, ⊥-BV verification = every production speculative path: lanes LA1/LA2,
   one engine, AEP). The worker returns, per verified row, the per-token log p/q moments of each draft segment
   (version, start, end) and of 4 absolute-position bins; the driver buckets segments by age = step - version
   (4 = 4+): ``spec/logratio_var_age{a}``, ``spec/logratio_mean_age{a}``, ``spec/tokens_age{a}``, and
   ``spec/logratio_var_pos{j}`` / ``spec/tokens_pos{j}`` (positions [0, 512), [512, 2048), [2048, 8192), 8192+).
   Cost: O(draft tokens) on the CPU of the verifying rank (one host copy of the row's log-ratio vector, which the
   pooled moments already read), O(rows) on the driver.

2. **Phase-in state** (learner, after the optimizer step, on this rank's optimizer shard), per frexp exponent bin E of
   the stored (16-bit) weight: ``n_E``, ``phi_E = 4 mean |theta_fp32 - theta_stored| / q_E`` (q_E = the storage ulp,
   2^(E-8) for bf16; 0 when every master sits on the grid, 1 when the sub-ulp residual is uniform) and
   ``changed_frac_E`` (stored value changed by this step).

3. **Adam update statistics** (the E2 harness's ``upd`` block, research/sticky_alt/adam_lag_alt.py, in lr units,
   incl. weight decay): rms / mean |u| with u = (theta_{t+1} - theta_t) / lr; rms of u - u_mom (u_mom = the
   zero-gradient AdamW step from the same state = ``draft_weights.momentum_draft_master``: what momentum misses);
   stored changed fraction, rms stored change, stored mismatch fraction vs the momentum draft; plus the pre-clip grad
   norm, clip factor and lr of the step.

Per-rank cost of 2 + 3 (N_r = elements of this rank's optimizer shard, ~N / DP): before the step, a fixed strided
sample of the shard (every ``THUNDERSYNC_DRIFT_STRIDE``-th element, default 64: theta, m, v -> 12 B x N_r / 64 of
transient GPU memory, freed after the step); after the step, one chunked elementwise pass over every
``THUNDERSYNC_DRIFT_PHI_STRIDE``-th shard element (default 8; master 4 B + stored 2 B read, 2^24-element chunks:
bounded temporaries, shared-memory bincount per exponent bin) for n_E / phi_E (n_E counts sampled elements), and the
pre-step sample for everything that needs theta_t (changed fraction, u, u_mom). Measured (GB200, fake shards,
research/nscale11/drift_probe_bench.log):  One all-reduce of 72 float64 over the learner world
(every shard element is owned by exactly one rank: sums over ranks count each model element once; parameters
duplicated over TP, e.g. latent projections, are counted TP times). No state across steps, nothing written to
parameters or optimizer state, no RNG. Off: ``THUNDERSYNC_DRIFT_LOG=0`` (default on).
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Iterable, Optional, Sequence

import torch

# Absolute token-position bins of the per-position moments (draft coordinates).
POS_EDGES = (512, 2048, 8192)
# Exponent bins: frexp exponent E of the stored weight (|w| in [2^(E-1), 2^E)), clamped to [E_LO, E_HI].
E_LO, E_HI = -14, 1
N_EBINS = E_HI - E_LO + 1
# Significant bits (incl. the implicit one) of the storage dtype: ulp = 2^(E - bits) in frexp's convention.
_SIG_BITS = {torch.bfloat16: 8, torch.float16: 11}


def enabled() -> bool:
    return os.environ.get("THUNDERSYNC_DRIFT_LOG", "1") != "0"


def stride() -> int:
    return max(1, int(os.environ.get("THUNDERSYNC_DRIFT_STRIDE", "64")))


def phi_stride() -> int:
    """Every k-th shard element enters the phase pass (n_E / phi_E); 1 = the whole shard."""
    return max(1, int(os.environ.get("THUNDERSYNC_DRIFT_PHI_STRIDE", "8")))


# ---------------------------------------------------------------------------
# 1. Per-age / per-position log p/q moments.
# ---------------------------------------------------------------------------


def _moments(x: torch.Tensor) -> tuple[int, float, float]:
    return int(x.numel()), float(x.sum()), float((x * x).sum())


def row_moments(
    d_lr: torch.Tensor, segments: Optional[Sequence[tuple[int, int, int]]]
) -> tuple[list[tuple[int, int, float, float]], list[tuple[int, float, float]]]:
    """(per-segment, per-position-bin) moments of one row's per-token log p/q ``d_lr`` (draft coordinates).

    Non-finite entries (exact top-p: log p = -inf) are excluded, as in the pooled moments. Segment entries are
    (version, n, sum, sum of squares); position entries (n, sum, sum of squares) for POS_EDGES' 4 bins.
    """
    x = d_lr.detach().double().cpu()
    fin = torch.isfinite(x)
    seg = []
    for v, a, b in segments or ():
        part = x[a:b][fin[a:b]]
        seg.append((int(v), *_moments(part)))
    pos = []
    edges = (0, *POS_EDGES, max(x.numel(), POS_EDGES[-1]))
    for lo, hi in zip(edges[:-1], edges[1:]):
        part = x[lo:hi][fin[lo:hi]]
        pos.append(_moments(part))
    return seg, pos


def _label(age: int) -> str:
    return f"{age}+" if age == 4 else str(age)


def age_moment_stats(rows: Iterable[dict], current_step: int) -> dict[str, float]:
    """spec/logratio_{var,mean}_age{a}, spec/tokens_age{a}, spec/logratio_var_pos{j}, spec/tokens_pos{j}.

    ``rows``: per-row results carrying ``lr_seg_moments`` / ``lr_pos_moments`` (``row_moments``). Age = steps
    between the drafting weights' version and the verifying step, clamped to [0, 4] (4 = 4+), as spec/age*_token_frac.
    """
    by_age: dict[int, list[float]] = {}
    by_pos: dict[int, list[float]] = {}
    for r in rows:
        for v, n, s, s2 in r.get("lr_seg_moments") or ():
            age = min(max(current_step - int(v), 0), 4)
            acc = by_age.setdefault(age, [0, 0.0, 0.0])
            acc[0] += n; acc[1] += s; acc[2] += s2
        for j, (n, s, s2) in enumerate(r.get("lr_pos_moments") or ()):
            acc = by_pos.setdefault(j, [0, 0.0, 0.0])
            acc[0] += n; acc[1] += s; acc[2] += s2
    out: dict[str, float] = {}
    for age, (n, s, s2) in sorted(by_age.items()):
        lab = _label(age)
        out[f"spec/tokens_age{lab}"] = float(n)
        if n >= 2:
            mean = s / n
            out[f"spec/logratio_mean_age{lab}"] = mean
            out[f"spec/logratio_var_age{lab}"] = max(s2 / n - mean * mean, 0.0)
    for j, (n, s, s2) in sorted(by_pos.items()):
        out[f"spec/tokens_pos{j}"] = float(n)
        if n >= 2:
            mean = s / n
            out[f"spec/logratio_var_pos{j}"] = max(s2 / n - mean * mean, 0.0)
    return out


# ---------------------------------------------------------------------------
# 2 + 3. Optimizer-shard phase-in and Adam update statistics.
# ---------------------------------------------------------------------------

# Slots of the reduced vector.
_U = ("n", "sum_u2", "sum_abs_u", "sum_resid2", "changed", "sum_su2", "mom_mismatch", "sum_sresid2")


def ebin(stored: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(bin index in [0, N_EBINS), frexp exponent) of stored values (fp32 view)."""
    e = torch.frexp(stored.float()).exponent
    return (e.clamp(E_LO, E_HI) - E_LO).long(), e


def ulp(e: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Storage ulp of values with frexp exponent ``e``: 2^(e - significant bits)."""
    return torch.ldexp(torch.ones_like(e, dtype=torch.float32), e - _SIG_BITS[dtype])


def phase_accumulate(master: torch.Tensor, stored: torch.Tensor, acc: torch.Tensor) -> None:
    """acc[0, b] += count, acc[1, b] += sum |master - stored| / q (per exponent bin b; exact zeros skipped).

    ``stored`` = the 16-bit value of the weight (its dtype gives q). Bitwise read-only.
    """
    s32 = stored.float()
    b, e = ebin(s32)
    # Exact zeros go to an extra bin N_EBINS (dropped). bincount with a few bins = shared-memory histogram on CUDA
    # (index_add_ into 16 float64 slots serialized on atomics: 1.3 s per 1e9 elements, research/nscale11 bench).
    b = torch.where(s32 != 0, b, N_EBINS)
    r = (master.float() - s32).abs() / ulp(e, stored.dtype)
    acc[0] += torch.bincount(b, minlength=N_EBINS + 1)[:N_EBINS].to(acc.dtype)
    # float64 weights: the per-bin sums do not depend on how the elements are grouped into calls.
    acc[1] += torch.bincount(b, weights=r.double(), minlength=N_EBINS + 1)[:N_EBINS].to(acc.dtype)


def update_accumulate(
    w0: torch.Tensor, w1: torch.Tensor, wmom: torch.Tensor, lr: float, dtype: torch.dtype,
    u_acc: torch.Tensor, chg_acc: torch.Tensor,
) -> None:
    """Adam-update sums (``_U`` slots) and per-bin changed counts of sampled elements.

    w0 = theta_t (master before the step), w1 = theta_{t+1}, wmom = the zero-gradient AdamW step from theta_t;
    stored values = rounding to ``dtype`` (what the optimizer's main -> model copy writes).
    """
    w0, w1, wmom = w0.double(), w1.double(), wmom.double()
    u = (w1 - w0) / lr
    um = (wmom - w0) / lr
    s0 = w0.to(dtype).double()
    s1 = w1.to(dtype).double()
    sm = wmom.to(dtype).double()
    changed = s1 != s0
    u_acc[0] += u.numel()
    u_acc[1] += (u * u).sum()
    u_acc[2] += u.abs().sum()
    u_acc[3] += ((u - um) ** 2).sum()
    u_acc[4] += changed.sum()
    u_acc[5] += (((s1 - s0) / lr) ** 2).sum()
    u_acc[6] += (s1 != sm).sum()
    u_acc[7] += (((s1 - sm) / lr) ** 2).sum()
    b, _ = ebin(s0)
    b = torch.where(s0 != 0, b, N_EBINS)
    chg_acc[0] += torch.bincount(b, minlength=N_EBINS + 1)[:N_EBINS].to(chg_acc.dtype)
    chg_acc[1] += torch.bincount(b, weights=changed.double(), minlength=N_EBINS + 1)[:N_EBINS].to(chg_acc.dtype)


def finalize(
    phase: torch.Tensor, chg: torch.Tensor, u_acc: torch.Tensor, *, grad_norm: Optional[float],
    clip_grad: float, lr: float, ok: bool,
) -> dict[str, float]:
    """Metric dict (``drift/...``) from the reduced sums."""
    out: dict[str, float] = {"drift/lr": float(lr), "drift/update_ok": float(ok)}
    if grad_norm is not None and math.isfinite(float(grad_norm)):
        gn = float(grad_norm)
        out["drift/grad_norm_preclip"] = gn
        out["drift/clip_factor"] = min(1.0, clip_grad / (gn + 1e-6)) if clip_grad > 0 else 1.0
    n = float(u_acc[0])
    if n > 0:
        out.update({
            "drift/rms_u": math.sqrt(float(u_acc[1]) / n),
            "drift/mean_abs_u": float(u_acc[2]) / n,
            "drift/rms_resid_mom": math.sqrt(float(u_acc[3]) / n),
            "drift/changed_frac": float(u_acc[4]) / n,
            "drift/rms_stored_u": math.sqrt(float(u_acc[5]) / n),
            "drift/stored_mom_mismatch_frac": float(u_acc[6]) / n,
            "drift/rms_stored_resid_mom": math.sqrt(float(u_acc[7]) / n),
            "drift/sampled_elements": n,
        })
    tot = float(phase[0].sum())
    if tot > 0:
        out["drift/elements"] = tot
        out["drift/phi"] = float(phase[1].sum()) * 4.0 / tot
    for b in range(N_EBINS):
        e = b + E_LO
        lab = f"E{e}" if E_LO < e < E_HI else (f"E<={e}" if e == E_LO else f"E>={e}")
        cnt = float(phase[0, b])
        if cnt > 0:
            out[f"drift/n_{lab}"] = cnt
            out[f"drift/phi_{lab}"] = 4.0 * float(phase[1, b]) / cnt
        if float(chg[0, b]) > 0:
            out[f"drift/changed_frac_{lab}"] = float(chg[1, b]) / float(chg[0, b])
    return out


class DriftProbe:
    """Wraps one optimizer step: ``pre(optimizer)``, the step, ``post(optimizer, step_result)``."""

    def __init__(self) -> None:
        self._sample: list[tuple[Any, tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict, slice]] = []
        self._lr = float("nan")
        self.last: dict[str, float] = {}

    @torch.no_grad()
    def pre(self, optimizer: Any) -> None:
        from thundersync_rl.draft_weights import _adam_state, _master_chunks, _shards

        self._sample = []
        k = stride()
        for sh in _shards(optimizer):
            n = sh.target.numel()
            if n == 0 or sh.target.dtype not in _SIG_BITS:
                continue
            sl = slice(0, n, k)
            master = _master_chunks(sh)
            m, v = _adam_state(sh.inner, sh.key)
            g = sh.group
            self._lr = float(g["lr"])
            hyper = {
                "lr": float(g["lr"]), "betas": tuple(g["betas"]), "eps": float(g["eps"]),
                "weight_decay": float(g["weight_decay"]), "step": int(g.get("step", 0)) + 1,
            }
            self._sample.append((sh, (master(sl).float().clone(), m[sl].float().clone(), v[sl].float().clone()),
                                 hyper, sl))

    @torch.no_grad()
    def post(self, optimizer: Any, step_result: Any, chunk: int = 1 << 24) -> dict[str, float]:
        from thundersync_rl.draft_weights import _master_chunks, _shards, momentum_draft_master

        t0 = time.perf_counter()
        ok, grad_norm = True, None
        if isinstance(step_result, tuple) and len(step_result) >= 2:
            ok, grad_norm = bool(step_result[0]), step_result[1]
        shards = [sh for sh in _shards(optimizer) if sh.target.numel() and sh.target.dtype in _SIG_BITS]
        if shards:
            dev = shards[0].target.device
        else:
            dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        phase = torch.zeros(2, N_EBINS, dtype=torch.float64, device=dev)
        chg = torch.zeros(2, N_EBINS, dtype=torch.float64, device=dev)
        u_acc = torch.zeros(len(_U), dtype=torch.float64, device=dev)
        # The accumulators run on concatenations of many shards' slices (up to ``chunk`` elements per call), not
        # per shard: each call ends in torch.bincount, which synchronizes with the host on CUDA, and a Megatron
        # distributed optimizer holds thousands of small shards (Super-L22: ~11.5k syncs, 137k launches,
        # ~2.2 s per step when called per shard). Counts are the same; float64 sums differ only in summation order.
        # Phase-in over the whole shard (theta_{t+1} vs its stored value), batched per stored dtype.
        k = phi_stride()
        pend: dict[torch.dtype, list] = {}

        def flush_phase(dt: torch.dtype) -> None:
            ms, ss = pend.pop(dt, ([], []))[:2]
            if ms:
                phase_accumulate(torch.cat(ms), torch.cat(ss), phase)

        for sh in shards:
            n = sh.target.numel()
            master = _master_chunks(sh)
            tgt = sh.target
            for start in range(0, n, chunk * k):
                sl = slice(start, min(start + chunk * k, n), k)
                ms, ss, cnt = pend.setdefault(tgt.dtype, ([], [], [0]))
                ms.append(master(sl).float())
                ss.append(tgt[sl])
                cnt[0] += ms[-1].numel()
                if cnt[0] >= chunk:
                    flush_phase(tgt.dtype)
        for dt in list(pend):
            flush_phase(dt)
        # Update statistics on the pre-step sample, batched per (hyper-parameters, stored dtype).
        groups: dict[tuple, list] = {}

        def flush_update(key: tuple) -> None:
            items = groups.pop(key, [])
            if not items:
                return
            h, dt = items[0][4], key[-1]
            w0, m0, v0, w1 = (torch.cat([it[j] for it in items]) for j in range(4))
            wmom = momentum_draft_master(
                w0, m0, v0, lr=h["lr"], beta1=h["betas"][0], beta2=h["betas"][1], eps=h["eps"],
                weight_decay=h["weight_decay"], step=h["step"],
            )
            lr = h["lr"] if h["lr"] > 0 else 1.0
            update_accumulate(w0, w1, wmom, lr, dt, u_acc, chg)

        for sh, (w0, m0, v0), h, sl in self._sample:
            key = (h["lr"], tuple(h["betas"]), h["eps"], h["weight_decay"], h["step"], sh.target.dtype)
            items = groups.setdefault(key, [])
            items.append((w0, m0, v0, _master_chunks(sh)(sl).float(), h))
            if sum(it[0].numel() for it in items) >= chunk:
                flush_update(key)
        for key in list(groups):
            flush_update(key)
        self._sample = []
        flat = torch.cat([phase.view(-1), chg.view(-1), u_acc])
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(flat)
        a, b = phase.numel(), chg.numel()
        phase = flat[:a].view(2, N_EBINS).cpu()
        chg = flat[a:a + b].view(2, N_EBINS).cpu()
        u_acc = flat[a + b:].cpu()
        clip = float(getattr(getattr(optimizer, "config", None), "clip_grad", 0.0) or 0.0)
        if grad_norm is not None and isinstance(grad_norm, torch.Tensor):
            grad_norm = float(grad_norm)
        out = finalize(phase, chg, u_acc, grad_norm=grad_norm, clip_grad=clip, lr=self._lr, ok=ok)
        out["drift/probe_s"] = time.perf_counter() - t0
        out["drift/phi_stride"] = float(k)
        self.last = out
        return out
