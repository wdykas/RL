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
"""Carried-walk verification of continuation pieces (DESIGN.md section 14r).

A continued row is decoded by one q stream and verified in forward pieces. Today
every piece is its own block (the walk b restarts at 1 and the piece end is a
terminal test). Here the pieces of a stream form one *block* that is decided only
at *decision points*; between them a piece only advances the walk

    log b_i = min(0, log b_{i-1} + log p(x_i) - log q(x_i))      (b_i = min(1, b_{i-1} p/q))

and remembers the *floor* f = the last position with b_f = 1 (there h_f = 1, so
tau >= f). The per-row state is two scalars. At a decision point T:

* terminal test eta_T <= b_T: the whole block is kept (no bonus: the stream goes on,
  its later tokens are q samples given the extended prefix);
* else tau = max{ i in (f, T) : eta_i <= h_i } with fresh eta_i (none drawn before),
  or f when there is no hit; emit y ~ norm(max(b_tau p_tau - q_tau, 0)).

This is block verification of the whole block [0, T] (section 4) with the draws
taken in a different order (eta_T first, then eta_{<T}); positions below f never
matter (h_f = 1). So a row verified piece by piece with decisions only at its end has
exactly the one-block law, and every position's statistics are computed once.

Decision points may be any stopping time of the row's past (tokens, the walk,
positions, EOS, engine timing, other rows), but not of the block's own verifier
draws (none are drawn before the decision; ``test_carried_walk.py`` shows a rule
that peeks at them is biased). Rules here: ``row`` (only at the end of the stream:
the one-block law), ``dip`` (also wherever -log b >= delta at a piece end),
``every`` (every piece end: today's law). Exactness argument: DESIGN 14r.

No intermediate decision can be *final* and still give the one-block law: at a
point g < T the one-block rule's P(tau <= g | past) = E[prod_{g<j<=T}(1 - h_j) | past]
depends on the q-continuation's future; DESIGN 14r has the sketch.

Floating point: ``advance`` is the canonical recursion (sequential Python floats),
used by both the piece update and the rejection scan, so both see identical b.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import torch

from thundersync_rl.block_verification import (
    emit_distribution,
    lazy_slack,
    stop_probability,
)

RULES = ("every", "row", "dip")


@dataclass(frozen=True)
class DecisionRule:
    """When a carried block is decided (always at the end of its stream).

    ``every``: at every piece end (today's piece-as-block law); ``row``: only at the
    end of the stream (one block); ``dip``: also at a piece end where -log b >= dip.
    All are functions of the row's past only (exact for any choice).
    """

    mode: str = "every"
    dip: float = 0.2

    def __post_init__(self):
        if self.mode not in RULES:
            raise ValueError(f"carried walk rule {self.mode!r}: one of {RULES}")
        if self.mode == "dip" and not self.dip > 0:
            raise ValueError("carried walk dip rule needs dip > 0")

    def due(self, log_b: float, final: bool) -> bool:
        if final or self.mode == "every":
            return True
        if self.mode == "dip":
            return -log_b >= self.dip
        return False

    def as_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "dip": self.dip}

    @classmethod
    def of(cls, d: Any) -> "DecisionRule":
        if isinstance(d, DecisionRule):
            return d
        return cls(**d)


def advance(
    log_ratio: Sequence[float], log_b0: float, floor0: int, pos0: int
) -> tuple[list[float], int]:
    """Walk over one piece: (log b at positions pos0 .. pos0 + n, floor after).

    ``log_ratio``: fp32 log p(x) - log q(x) of the piece's n tokens. The first entry
    of the returned list is ``log_b0`` (position pos0). ``floor``: the last position
    with b == 1 exactly (ties count), starting from ``floor0``.
    """
    lb, floor = float(log_b0), int(floor0)
    out = [lb]
    for k, x in enumerate(log_ratio):
        lb = lb + float(x)
        if lb >= 0.0:
            lb = 0.0
            floor = pos0 + k + 1
        out.append(lb)
    return out, floor


@dataclass
class Walk:
    """A carried block's state (driver side; two scalars plus bookkeeping)."""

    log_b: float = 0.0  # log b at the current end of the block
    floor: int = 0  # block position of the last b == 1
    n: int = 0  # block tokens verified so far

    def as_dict(self) -> dict[str, Any]:
        return {"log_b": self.log_b, "floor": self.floor, "n": self.n}


def _hits_desc(
    lb: Sequence[float],
    lo: int,
    eta: torch.Tensor,
    slack,
    mass: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    first: int = 8,
) -> tuple[Optional[int], int]:
    """Highest local index j in (lo, len(eta)) with eta_j <= h_j, scanning lazily.

    ``lb[j]``: log b at local index j; ``mass(idx, b_idx)`` returns r at those
    indices. Skips j with eta_j > b_j (1 + slack_j) (h_j <= b_j: never a hit).
    Returns (j or None, rows evaluated).
    """
    g = eta.numel()
    b = torch.tensor([math.exp(x) for x in lb[:g]], dtype=torch.float64)
    eta_h = eta.detach().cpu().double()
    sl = slack.cpu() if isinstance(slack, torch.Tensor) else slack
    cand = torch.nonzero(eta_h <= b * (1 + sl)).flatten()
    cand = cand[cand > lo].flip(0)
    done, n = 0, first
    while done < cand.numel():
        idx = cand[done : done + n]
        r = mass(idx, b[idx]).detach().cpu().double()
        h = stop_probability(r, b[idx])
        hit = torch.nonzero(eta_h[idx] <= h).flatten()
        done += idx.numel()
        if hit.numel():
            return int(idx[hit[0]]), done
        n *= 2
    return None, done


def _rows_mass(p_rows, q_rows, lse_p, lse_q):
    """r(idx, b) = sum_v (b p - q)+ from on-demand raw logit rows (fp32 log-probs,
    exponentiated in fp64, as the reference)."""

    def mass(idx: torch.Tensor, bi: torch.Tensor) -> torch.Tensor:
        dev = lse_p.device
        i = idx.to(dev)
        P, Q = p_rows(i), q_rows(i)
        if P.is_cuda:
            from thundersync_rl.block_verification_fused import residual_masses

            return residual_masses(P, Q, lse_p[i], lse_q[i], bi.to(dev))
        p = (P.float() - lse_p[i, None]).double().exp()
        q = (Q.float() - lse_q[i, None]).double().exp()
        return (bi.to(dev)[:, None] * p - q).clamp_min(0).sum(-1)

    return mass


def _emit(p_row, q_row, lse_p_t, lse_q_t, b_t: float, generator) -> tuple[int, float]:
    """Residual sample at a rejection (multinomial on ``generator``) and its log p."""
    p = (p_row.float() - lse_p_t).double().exp()
    q = (q_row.float() - lse_q_t).double().exp()
    dist = emit_distribution(p, q, torch.tensor(b_t, dtype=torch.float64, device=p.device))
    y = int(torch.multinomial((dist / dist.sum()).float(), 1, generator=generator))
    return y, float(p_row[y].float() - lse_p_t)


def carried_piece(
    lpx: torch.Tensor,
    lqx: torch.Tensor,
    lse_p: torch.Tensor,
    lse_q: torch.Tensor,
    p_rows,
    q_rows,
    generator: Optional[torch.Generator],
    walk: dict[str, Any],
    rule: DecisionRule,
    final: bool,
) -> dict[str, Any]:
    """One piece of a carried block: advance the walk; decide if a decision is due.

    ``lpx``/``lqx``: [g] fp32 log p / log q of the piece's tokens; ``lse_p`` [g+1],
    ``lse_q`` [g]; ``p_rows(idx)`` / ``q_rows(idx)``: raw logits rows at local
    positions (row j predicts the piece's token j; rows are read only by a rejection
    scan). ``walk``: {"log_b", "floor", "n"} at the piece start (block positions);
    ``final``: the stream ended with this piece (EOS / cap / turn end / horizon).

    Returns {"walk": state after the piece, "decision": None | "whole" | "reject" |
    "scan", ...}: "reject" carries "tau" (block position) and "next", "next_logp";
    "scan" means the last hit lies before this piece: positions (floor, piece start)
    need a second call (``carried_scan``), with this piece already scanned (no hit).
    Generator calls when a decision is due: rand(1) (terminal), then on failure
    rand(g) (this piece's non-terminal positions) and one multinomial (emission).
    """
    g = lpx.numel()
    pos0, floor0 = int(walk["n"]), int(walk["floor"])
    lr = (lpx.float() - lqx.float().to(lpx.device)).cpu().tolist()
    lb, floor = advance(lr, float(walk["log_b"]), floor0, pos0)
    out: dict[str, Any] = {
        "walk": {"log_b": lb[-1], "floor": floor, "n": pos0 + g},
        "decision": None,
        "rows_evaluated": 0,
        "forced": False,
    }
    if not rule.due(lb[-1], final):
        return out
    dev = lse_p.device
    eta_t = torch.rand(1, generator=generator, dtype=torch.float64, device=dev)
    # Debug only (THUNDERSYNC_DEBUG_CW_FORCE_REJECT): fail the terminal test so the
    # rejection / scan path runs in short E2E checks. Changes the law (a plan with a
    # forced rejection is not a p sample); the bitwise checks stay meaningful.
    forced = bool(walk.get("force_reject")) and lb[-1] < 0.0
    if not forced and float(eta_t) <= math.exp(lb[-1]):
        out["decision"] = "whole"
        return out
    out["forced"] = forced
    eta = torch.rand(g, generator=generator, dtype=torch.float64, device=dev)
    mass = _rows_mass(p_rows, q_rows, lse_p, lse_q)
    j, n_eval = _hits_desc(lb, floor - pos0, eta, lazy_slack(lse_p[:g]), mass)
    out["rows_evaluated"] = n_eval
    if j is None and floor >= pos0:
        j = floor - pos0  # h = 1 at the floor
    if j is None:
        out["decision"] = "scan"
        return out
    at = torch.tensor([j], device=dev)
    y, lpy = _emit(p_rows(at)[0], q_rows(at)[0], lse_p[j], lse_q[j], math.exp(lb[j]), generator)
    out.update(decision="reject", tau=pos0 + j, next=y, next_logp=lpy)
    return out


def carried_scan(
    log_ratio: Sequence[float],
    log_b0: float,
    pos0: int,
    floor: int,
    stop: int,
    lse_p: torch.Tensor,
    lse_q: torch.Tensor,
    p_rows,
    q_rows,
    generator: Optional[torch.Generator],
) -> dict[str, Any]:
    """Rejection scan of a carried block below the deciding piece.

    The block's sub-draft starts at block position ``pos0`` (<= ``floor``) with walk
    value ``log_b0`` there; ``log_ratio``: the carried fp32 log-ratios of its tokens
    (the values the walk used); local row j is block position pos0 + j. Positions
    [stop, T) were scanned by ``carried_piece`` without a hit, so this scans
    (floor, stop) top down with fresh draws rand(stop - pos0), and takes the floor
    when there is no hit. Returns {"tau", "next", "next_logp", "rows_evaluated"}.
    """
    g = stop - pos0
    if len(log_ratio) != g or not pos0 <= floor < stop:
        raise ValueError(f"carried scan: {len(log_ratio)} ratios, pos0 {pos0}, floor {floor}, stop {stop}")
    lb, _ = advance(log_ratio, log_b0, pos0, pos0)
    # The carried values must reproduce the carried floor: b = 1 there, < 1 after.
    if lb[floor - pos0] != 0.0 or any(x == 0.0 for x in lb[floor - pos0 + 1 :]):
        raise RuntimeError(f"carried scan: walk does not reproduce floor {floor}")
    dev = lse_p.device
    eta = torch.rand(g, generator=generator, dtype=torch.float64, device=dev)
    mass = _rows_mass(p_rows, q_rows, lse_p, lse_q)
    j, n_eval = _hits_desc(lb, floor - pos0, eta, lazy_slack(lse_p[:g]), mass)
    if j is None:
        j = floor - pos0
    at = torch.tensor([j], device=dev)
    y, lpy = _emit(p_rows(at)[0], q_rows(at)[0], lse_p[j], lse_q[j], math.exp(lb[j]), generator)
    return {"tau": pos0 + j, "next": y, "next_logp": lpy, "rows_evaluated": n_eval}


def block_decision_reference(
    lp: torch.Tensor,
    lq: torch.Tensor,
    draft: torch.Tensor,
    eta_terminal: float,
    eta: torch.Tensor,
) -> int:
    """One-block tau from full log-prob rows with given draws (tests).

    tau = T if eta_terminal <= b_T, else max{ i < T : eta_i <= h_i } with b from the
    canonical recursion; the root has b = 1 (h = 1).
    """
    T = draft.numel()
    ar = torch.arange(T)
    lr = (lp[ar, draft].float() - lq[ar, draft].float()).tolist()
    lb, _ = advance(lr, 0.0, 0, 0)
    if eta_terminal <= math.exp(lb[-1]):
        return T
    for i in range(T - 1, -1, -1):
        bi = math.exp(lb[i])
        r = float(
            (bi * lp[i].double().exp() - lq[i].double().exp()).clamp_min(0).sum()
        )
        h = float(stop_probability(torch.tensor([r]), torch.tensor([bi]))[0])
        if float(eta[i]) <= h:
            return i
    raise RuntimeError("no stop position")
