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
"""Order-invariant exact gradient accumulator (nolanes train overlap, R31 Option A variant).

A step's prompt groups may train in arrival order (as their continuation tails finish) only if the step gradient
is bitwise independent of that order. fp32 accumulation in the grad buffers is not (fp32 addition does not
associate), so after each group's backward its gradient (the grad buffers, used as scratch) is folded into one
int64 fixed-point accumulator and the buffers are zeroed:

    q = rint(x * 2^30)   (fp64: x * 2^30 is exact; the only rounding is of bits below 2^-30, the same in every order)

Integer addition is exact, associative and commutative, so the sum -- and the fp32 value ``finalize`` writes back
(sum * 2^-30 rounded once) -- is the same for every order of the groups. ``finalize`` runs before the step's 1/N
normalization, DP reduction, clipping and optimizer step, which then see one fixed fp32 gradient.

Range: a capture may hold |x| < 2^24 (an unnormalized group-gradient element of 1.7e7) and a step at most 511
captures, so |sum| < 511 * 2^54 < 2^63: no int64 overflow is possible. A larger or non-finite element raises at
finalize (counted per capture, order-free), as do more captures. Resolution 2^-30 (9.3e-10, unnormalized: the
step's 1/N then divides it by the token count): finer than fp32's own relative precision for every element above
~1e-2 unnormalized, and below Adam's eps after normalization for the rest.

Cost per learner rank: 8 B per element of the rank's local grad buffers (one int64; the grad buffers stay as
scratch), held only during the step's training phase (allocated at begin, dropped at finalize, before the optimizer
step); one fused pass per capture (Triton: read fp32, read-modify-write int64, zero fp32: 24 B per element, at ~90%
of HBM bandwidth) and one at finalize. No communication: each rank's rows of a group are a function of the group
(shard_meta_for_dp), so each rank's sum is order-invariant and the stock DP reduce sees fixed inputs.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Sequence

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

SHIFT = 30  # q = rint(x * 2^30)
MAX_ABS = float(2**24)  # per-capture element bound
MAX_CAPTURES = 511  # 511 * 2^24 * 2^30 < 2^63
CHUNK = 1 << 24  # elements per fp64 temporary (bounded scratch)


def to_fixed(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 / bf16 values -> int64 fixed point (x * 2^30, rounded) and a bad mask (non-finite or |x| >= 2^24)."""
    x64 = x.to(torch.float64)
    bad = ~torch.isfinite(x64) | (x64.abs() >= MAX_ABS)
    q = torch.round(torch.where(bad, torch.zeros_like(x64), x64) * float(2**SHIFT))
    return q.to(torch.int64), bad


if HAVE_TRITON:

    @triton.jit
    def _capture_kernel(x_ptr, acc_ptr, bad_ptr, n, BLOCK: tl.constexpr):
        # One pass: acc += rint(x * 2^30) (fp64, as to_fixed), x = 0, count bad elements.
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float64)
        bad = (tl.abs(x) >= 16777216.0) | (x != x)  # |x| >= 2^24 (incl. inf) or nan
        q = libdevice.rint(tl.where(bad, 0.0, x) * 1073741824.0).to(tl.int64)  # 2^30
        a = tl.load(acc_ptr + offs, mask=m, other=0)
        tl.store(acc_ptr + offs, a + q, mask=m)
        tl.store(x_ptr + offs, tl.zeros_like(x).to(x_ptr.dtype.element_ty), mask=m)
        nb = tl.sum((bad & m).to(tl.int64), axis=0)
        tl.atomic_add(bad_ptr, nb, mask=nb > 0)  # bad elements are rare: no contended atomic per program


def capture_into(x: torch.Tensor, acc: torch.Tensor, bad: torch.Tensor, chunk: int = CHUNK) -> None:
    """acc += to_fixed(x); bad += #bad; x.zero_(). One fused pass on CUDA (Triton), chunked torch otherwise;
    both compute rint(x * 2^30) in fp64 (round half to even), so the sums are identical."""
    flat = x.view(-1)
    if HAVE_TRITON and flat.is_cuda and flat.dtype == torch.float32 and os.environ.get(
        "NEMO_RL_EXACT_GRAD_ACCUM_TORCH", "0"
    ) != "1":
        n = flat.numel()
        # Bandwidth-bound elementwise pass (24 B per element): 1024 elements per program, 4 warps (8 per thread).
        # GB200, 1.07 G elements: 3.6 ms = 7.2 TB/s (~90% of HBM peak) vs 7.9 ms with 4096 / 4 warps; same sums
        # (research/nolanes2/steady/capture/bench.py). Per element, so it holds at every model size.
        BLOCK = 1024
        _capture_kernel[(triton.cdiv(n, BLOCK),)](flat, acc, bad, n, BLOCK=BLOCK, num_warps=4)
        return
    for a in range(0, flat.numel(), chunk):
        b = min(a + chunk, flat.numel())
        q, bd = to_fixed(flat[a:b])
        acc[a:b] += q
        bad += bd.sum()
    flat.zero_()


def from_fixed(q: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Fixed-point sums -> values (one rounding of the exact integer)."""
    return (q.to(torch.float64) * float(2.0**-SHIFT)).to(dtype)


class OrderInvariantGradAccumulator:
    """Folds the gradient tensors into an int64 fixed-point sum after each capture; ``finalize`` writes it back.

    ``grads``: the flat tensors one backward writes into (Megatron: the DDP ``grad_data`` buffers). ``reset`` at the
    start of a step, ``capture`` after each group's backward, ``finalize`` once, before the step's normalization /
    reduction / optimizer step; ``release`` drops the sums (the worker allocates them per step's training phase).

    ``check_elems`` (CHECKS runs): every capture also keeps the first ``check_elems`` elements of each tensor, and
    ``finalize`` re-sums them in other orders (reversed, two seeded permutations); ``order_check`` reports how many
    of the written elements would change under another arrival order (0 for this accumulator).
    """

    naive = False

    def __init__(
        self,
        grads: Sequence[torch.Tensor],
        chunk: int = CHUNK,
        check_elems: int = 0,
        acc_views: Optional[Sequence[torch.Tensor]] = None,
    ):
        self.grads = list(grads)
        self.chunk = int(chunk)
        self.check_elems = int(check_elems)
        if acc_views is not None:
            # Views of a persistent buffer the caller owns (no per-step allocation); zeroed by reset().
            if len(acc_views) != len(self.grads) or any(
                v.dtype != torch.int64 or v.numel() != g.numel() for v, g in zip(acc_views, self.grads)
            ):
                raise ValueError("acc_views must be int64 tensors with one element per gradient element")
            self.acc = list(acc_views)
        else:
            self.acc = [torch.zeros(g.numel(), dtype=torch.int64, device=g.device) for g in self.grads]
        self.replay_armed = False
        self.first_digest: Optional[int] = None
        self.replay: dict[str, float] = {}
        self.acc_slices = None
        dev = self.grads[0].device if self.grads else torch.device("cpu")
        self.bad = torch.zeros((), dtype=torch.int64, device=dev)
        self.captures = 0
        self.capture_s = 0.0
        self.samples: list[list[torch.Tensor]] = []
        self.order_check: dict[str, float] = {}

    @property
    def nbytes(self) -> int:
        return sum(a.numel() * 8 for a in self.acc)

    def _sync(self) -> None:
        if self.grads and self.grads[0].is_cuda:
            torch.cuda.synchronize(self.grads[0].device)

    def matches(self, grads: Sequence[torch.Tensor]) -> bool:
        return len(grads) == len(self.grads) and all(
            a.data_ptr() == b.data_ptr() and a.numel() == b.numel() for a, b in zip(grads, self.grads)
        )

    @torch.no_grad()
    def reset(self) -> None:
        for A in self.acc:
            A.zero_()
        for g in self.grads:  # the scratch starts the step empty (the step's first group lands on zeros)
            g.zero_()
        self.bad.zero_()
        self.captures = 0
        self.capture_s = 0.0
        self.samples = []
        self.order_check = {}
        self.replay_armed = False
        self.first_digest = None
        self.replay = {}
        self.acc_slices = None

    @torch.no_grad()
    def scratch_digest(self) -> int:
        """Order-free full digest of the scratch (sum of to_fixed over every element; CHECKS only, one torch pass)."""
        d = 0
        for g in self.grads:
            flat = g.view(-1)
            for a in range(0, flat.numel(), self.chunk):
                d += int(to_fixed(flat[a : a + self.chunk])[0].sum())
        return d

    @torch.no_grad()
    def replay_compare(self) -> None:
        """CHECKS: the scratch holds a re-run of the step's FIRST group (at the end of the window); compare it with
        that group's first contribution (sampled elements bitwise + full digest), then drop it (no add)."""
        self._sync()
        if self.check_elems > 0 and self.samples:
            first = self.samples[0]
            now = [g.view(-1)[: self.check_elems] for g in self.grads]
            mism = sum(int((a.view(torch.int32) != b.view(torch.int32)).sum()) for a, b in zip(first, now))
            num = sum(float(((a.double() - b.double()) ** 2).sum()) for a, b in zip(first, now))
            den = sum(float((a.double() ** 2).sum()) for a in first)
            self.replay = {
                "replay_checked": float(sum(t.numel() for t in now)),
                "replay_mismatch": float(mism),
                "replay_rel_l2": (num / den) ** 0.5 if den > 0 else 0.0,
                "replay_digest_equal": float(self.first_digest is not None and self.scratch_digest() == self.first_digest),
            }
        for g in self.grads:
            g.zero_()
        self.replay_armed = False

    @torch.no_grad()
    def discard(self) -> None:
        """Drop the gradient tensors' current values without adding them (ThunderSync EDV: a group whose deferred row
        was not accepted whole is retrained later). The fixed-point sum is untouched."""
        self._sync()
        for g in self.grads:
            g.zero_()
        self.discards = getattr(self, "discards", 0) + 1

    def _sample(self) -> None:
        if self.check_elems > 0:
            self.samples.append([g.view(-1)[: self.check_elems].clone() for g in self.grads])

    @torch.no_grad()
    def capture(self) -> None:
        """Add the gradient tensors' current values into the fixed-point sum, then zero them (they are scratch)."""
        self._sync()
        t0 = time.perf_counter()
        if self.check_elems > 0 and self.captures == 0:
            self.first_digest = self.scratch_digest()
        self._sample()
        for g, A in zip(self.grads, self.acc):
            capture_into(g, A, self.bad, self.chunk)
        self.captures += 1
        self._sync()
        self.capture_s += time.perf_counter() - t0

    @staticmethod
    def sum_in_order(samples: list[list[torch.Tensor]], order: Sequence[int], naive: bool) -> list[torch.Tensor]:
        """The written values of the sampled elements had the captures arrived in ``order``."""
        out = []
        for j in range(len(samples[0]) if samples else 0):
            if naive:
                acc = torch.zeros_like(samples[0][j])
                for i in order:
                    acc += samples[i][j]
                out.append(acc)
            else:
                A = torch.zeros(samples[0][j].numel(), dtype=torch.int64, device=samples[0][j].device)
                for i in order:
                    A += to_fixed(samples[i][j])[0]
                out.append(from_fixed(A, samples[0][j].dtype))
        return out

    @torch.no_grad()
    def _order_check(self) -> None:
        if not self.samples:
            return
        n = len(self.samples)
        written = [g.view(-1)[: self.check_elems] for g in self.grads]
        gen = torch.Generator().manual_seed(1234 + n)
        orders = [list(reversed(range(n)))] + [torch.randperm(n, generator=gen).tolist() for _ in range(2)]
        mismatch = 0
        for order in orders:
            alt = self.sum_in_order(self.samples, order, self.naive)
            mismatch = max(
                mismatch,
                sum(int((w.view(torch.int32) != a.view(torch.int32)).sum()) for w, a in zip(written, alt)),
            )
        # Power: would plain fp32 accumulation have changed bits between the arrival and the reversed order?
        fa = self.sum_in_order(self.samples, list(range(n)), True)
        fr = self.sum_in_order(self.samples, list(reversed(range(n))), True)
        naive_diff = sum(int((a.view(torch.int32) != b.view(torch.int32)).sum()) for a, b in zip(fa, fr))
        def rel(a, b):
            num = sum(float(((x.double() - y.double()) ** 2).sum()) for x, y in zip(a, b))
            den = sum(float((y.double() ** 2).sum()) for y in b)
            return (num / den) ** 0.5 if den > 0 else 0.0

        self.order_check = {
            "captures": float(n),
            "checked": float(sum(w.numel() for w in written)),
            "mismatch": float(mismatch),
            "naive_order_diff": float(naive_diff),
            # Accuracy vs the plain fp32 path (arrival order), against the fp32 order floor (arrival vs reversed).
            "grad_rel_l2_vs_fp32": rel(written, fa),
            "grad_rel_l2_fp32_order_floor": rel(fr, fa),
            "zero_here_nonzero_fp32": float(sum(int(((w == 0) & (a != 0)).sum()) for w, a in zip(written, fa))),
        }
        self.acc_slices = ([w.clone() for w in written], fa, fr)
        # the samples stay until release(): a CHECKS replay compares against samples[0]

    @torch.no_grad()
    def finalize(self) -> None:
        """Write the exact sum (one fixed fp32 value per element) into the gradient tensors."""
        nbad = int(self.bad.item())
        if nbad or self.captures > MAX_CAPTURES:
            raise RuntimeError(
                f"[exact_grad_accum] {nbad} gradient elements were non-finite or >= 2^24 in some capture, "
                f"{self.captures} captures (max {MAX_CAPTURES}): the step's gradient is not usable"
            )
        for g, A in zip(self.grads, self.acc):
            flat = g.view(-1)
            for a in range(0, flat.numel(), self.chunk):
                b = min(a + self.chunk, flat.numel())
                flat[a:b].copy_(from_fixed(A[a:b], flat.dtype))
        self._order_check()

    def adam_sim(self, inv_n: float, sim: dict, beta1: float, beta2: float, eps: float) -> dict:
        """Adam update on the sampled elements from three gradient tracks (this accumulator; plain fp32 in arrival
        order; plain fp32 reversed = the order floor), each with its own moments kept in ``sim`` across steps."""
        if self.acc_slices is None:
            return {}
        out = {}
        tracks = dict(zip(("fixed", "fp32", "fp32_rev"), self.acc_slices))
        sim["t"] = sim.get("t", 0) + 1
        t = sim["t"]
        upd = {}
        for name, gs in tracks.items():
            g = torch.cat([x.double().view(-1) for x in gs]) * inv_n
            m = sim.get(name + "_m", torch.zeros_like(g))
            v = sim.get(name + "_v", torch.zeros_like(g))
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g * g
            sim[name + "_m"], sim[name + "_v"] = m, v
            upd[name] = (m / (1 - beta1**t)) / ((v / (1 - beta2**t)).sqrt() + eps)

        def rel(a, b):
            d = float(torch.linalg.vector_norm(b))
            return float(torch.linalg.vector_norm(a - b)) / d if d > 0 else 0.0

        out["adam_rel_l2_vs_fp32"] = rel(upd["fixed"], upd["fp32"])
        out["adam_rel_l2_fp32_order_floor"] = rel(upd["fp32_rev"], upd["fp32"])
        return out

    def release(self) -> None:
        self.acc = []
        self.samples = []
        self.acc_slices = None

    def digest(self) -> Optional[torch.Tensor]:
        """Order-free digest of the current fixed-point state (int64 sum; for checks)."""
        if not self.acc:
            return None
        return sum(a.sum() for a in self.acc)


class NaiveArrivalOrderAccumulator(OrderInvariantGradAccumulator):
    """NEGATIVE CONTROL (THUNDERSYNC_NL_NEG=grad_order): the same capture / finalize protocol, but the step gradient
    is the plain fp32 sum in arrival order (a separate fp32 buffer), so its bits follow the arrival order."""

    naive = True

    def __init__(
        self,
        grads: Sequence[torch.Tensor],
        chunk: int = CHUNK,
        check_elems: int = 0,
        acc_views: Optional[Sequence[torch.Tensor]] = None,  # unused: the control keeps its own fp32 sums
    ):
        self.grads = list(grads)
        self.chunk = int(chunk)
        self.check_elems = int(check_elems)
        self.sums = [torch.zeros_like(g) for g in self.grads]
        self.acc = []
        dev = self.grads[0].device if self.grads else torch.device("cpu")
        self.bad = torch.zeros((), dtype=torch.int64, device=dev)
        self.captures = 0
        self.capture_s = 0.0
        self.replay_armed = False
        self.first_digest = None
        self.replay = {}
        self.acc_slices = None
        self.samples = []
        self.order_check = {}

    @property
    def nbytes(self) -> int:
        return sum(s.numel() * s.element_size() for s in self.sums)

    @torch.no_grad()
    def reset(self) -> None:
        for s in self.sums:
            s.zero_()
        for g in self.grads:
            g.zero_()
        self.bad.zero_()
        self.captures = 0
        self.capture_s = 0.0
        self.samples = []
        self.order_check = {}
        self.replay_armed = False
        self.first_digest = None
        self.replay = {}
        self.acc_slices = None

    @torch.no_grad()
    def capture(self) -> None:
        if self.check_elems > 0 and self.captures == 0:
            self.first_digest = self.scratch_digest()
        self._sample()
        for g, s in zip(self.grads, self.sums):
            s += g
            g.zero_()
        self.captures += 1

    @torch.no_grad()
    def finalize(self) -> None:
        for g, s in zip(self.grads, self.sums):
            g.copy_(s)
        self._order_check()

    def release(self) -> None:
        self.sums = []
        self.samples = []
        self.acc_slices = None
