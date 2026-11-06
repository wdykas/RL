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
"""DS2: the ⊥-BV grid scan on this rank's vocabulary shard (``THUNDERSYNC_DS2=1``; default off).

Why: at Super-L22 the verifier's per-row tail is host-bound. A draft whose outcome is decided at the ⊥ root (about a
quarter of them) scans almost every position. The scan evaluates a residual mass at each of them, in 512-position
chunks. Each chunk all-gathers the full fp32 p and q rows over TP, runs per-position Python and does one host sync.
That costs 80-160 ms per such row and ~80% of the row-tail time (research/verify0attr/README.md §26-§27).

What changes (same draws, same rule):
- p and q stay vocabulary-sharded. The residual masses are canonical-block masses (kernel task D, ``residual_partials``):
  every rank sums its own 4096-column blocks in fp64, the ranks all-gather 8 bytes per block, and every rank combines
  the blocks in global block order. No [k, V] row is gathered.
- The scan takes positions in chunks of ``FIRST_CHUNK`` then ``CHUNK`` (fewer host syncs). Evaluating more candidates
  than needed does not change tau: tau_M is the highest candidate with eta <= h, and h is a function of the position.
- The emission is drawn by task D's block-then-column inverse CDF, from the same masses. The extra ⊥ atom is the
  block after the last.

Exactness:
- The generator calls (eta = rand(g+1), u = rand(1)), the weights, the candidate filter and the stop rule are those of
  ``block_verification_fused._grid_packed``.
- Each summand of a residual mass is the same fp64 value as before: exp(fp64(fp32(z) - LSE)) with the canonical LSE.
  Only the order of the fp64 sums changes (fixed per 4096-column block, then the blocks in order), and so does the
  emission's inverse-CDF order (blocks, then columns in order).
- The law is the same; the outcome can differ from the gathered path only when eta or u lands within fp64 rounding of
  a stop or CDF boundary.
- The values do not depend on TP or the batch: task D's bitwise contract. Every rank of a TP group combines the same
  gathered blocks in the same order, so all of them take the same decisions.

The kernels are the frozen task-D reference kernels (``kernel_tasks/vocablse/vl_reference``, sha256-pinned in
``VL_ORACLE.sha256``) with one change: rows are read through an index array, so the p rows of the forward's logits and
a chunk's q rows are not copied per (envelope, position) pair. Per element the arithmetic is unchanged, and so is the
in-block reduction order: one program per (row, block), sequential 1024-column tiles. ``tests/unit/test_ds2.py``
checks every kernel bitwise against the frozen reference on materialized rows.
"""

from __future__ import annotations

import math
import os
from typing import Callable, Optional

import numpy as np
import torch
import triton
import triton.language as tl

from thundersync_rl.block_verification import (
    _COUNTS,
    NULL_TOKEN,
    check_emission,
    check_null_m,
    check_stop_probabilities,
    check_verification_inputs,
    lazy_slack,
    stop_probability,
)
from thundersync_rl.envflag import env_flag

BLOCK = 4096
_TILE = 1024  # the reference's in-block tile (part of its bits)
FIRST_CHUNK = 512  # positions in the first scanned chunk (as the gathered scan)
CHUNK = 2048  # positions in every later chunk (q logits [CHUNK, V/TP] per rank: 512 MiB fp32 at V 131k, TP 2)
ENV = "THUNDERSYNC_DS2"
# Per-process counters (logged by the worker after each verification call that used DS2): rows scanned, positions
# evaluated, and the emissions by kind (bonus = whole draft accepted, residual, bot = the ⊥ atom, root = a token at
# the ⊥ root).
COUNTS = {"rows": 0, "evals": 0, "chunks": 0, "emit_dedup": 0, "bonus": 0, "residual": 0, "root": 0, "bot": 0, "p_dtype": "", "q_dtype": ""}
# EDV shadow record of the last scanned row (DESIGN §22.5, log only): the final coin eta_g and b_g at every envelope,
# read from the host copies the scan already made (no device op, no draw, no change to any value). ``take_last_row``.
LAST_ROW: Optional[dict] = None
SHADOW = True  # always on; the unit test turns it off to show the scan is unchanged


def take_last_row() -> Optional[dict]:
    """The EDV shadow record of the row ``grid_scan_sharded`` scanned last on this process (then cleared)."""
    global LAST_ROW
    r, LAST_ROW = LAST_ROW, None
    return r


def enabled() -> bool:
    """``THUNDERSYNC_DS2`` (default off): the vocabulary-sharded grid scan of this module."""
    return env_flag(ENV)


@triton.jit
def _residual_rows_kernel(ZP, ZQ, IP, IQ, LP, LQ, Bw, OUT, sp, sq, voff, vlim, nb_local, HAS_Q: tl.constexpr,
                          BLOCK: tl.constexpr, TILE: tl.constexpr):
    # Task D ``_residual_kernel`` with the p / q rows read through IP / IQ (fp32 rows: see ``_layout``).
    row = tl.program_id(0)
    j = tl.program_id(1)
    ip = tl.load(IP + row).to(tl.int64)
    iq = tl.load(IQ + row).to(tl.int64)
    g0 = voff + j * BLOCK
    lse_p = tl.load(LP + row)
    lse_q = tl.load(LQ + row)
    b = tl.load(Bw + row)
    acc = tl.zeros([], tl.float64)
    for c0 in range(0, BLOCK, TILE):
        cols = c0 + tl.arange(0, TILE)
        ok = (g0 + cols) < vlim
        zp = tl.load(ZP + ip * sp + j * BLOCK + cols, mask=ok, other=float("-inf")).to(tl.float32)
        p = tl.exp((zp - lse_p).to(tl.float64))
        d = b * p
        if HAS_Q:
            zq = tl.load(ZQ + iq * sq + j * BLOCK + cols, mask=ok, other=float("-inf")).to(tl.float32)
            d = d - tl.exp((zq - lse_q).to(tl.float64))
        d = tl.where(ok & (d > 0), d, 0.0)
        acc += tl.sum(d, 0)
    tl.store(OUT + row.to(tl.int64) * nb_local + j, acc)


@triton.jit
def _mass_combine_kernel(Mz, OUT, nb):
    # Task D ``_mass_combine_kernel`` (unchanged).
    row = tl.program_id(0)
    t = tl.zeros([], tl.float64)
    for j in range(0, nb):
        t += tl.load(Mz + row.to(tl.int64) * nb + j)
    tl.store(OUT + row, t)


@triton.jit
def _sample_block_kernel(Mz, BOT, U, OB, OT, nb):
    # Task D ``_sample_block_kernel`` (unchanged).
    row = tl.program_id(0)
    base = Mz + row.to(tl.int64) * nb
    bot = tl.load(BOT + row)
    total = tl.zeros([], tl.float64)
    for j in range(0, nb):
        total += tl.load(base + j)
    total += bot
    t = tl.load(U + row) * total
    c = tl.zeros([], tl.float64)
    prev = tl.zeros([], tl.float64)
    found = tl.full([], -1, tl.int64)
    lastnz = tl.full([], -1, tl.int64)
    tin = tl.zeros([], tl.float64)
    for j in range(0, nb + 1):
        mj = tl.where(j < nb, tl.load(base + tl.minimum(j, nb - 1)), bot)
        c += mj
        hit = (found < 0) & (c > t)
        tin = tl.where(hit, t - prev, tin)
        found = tl.where(hit, j, found)
        lastnz = tl.where(mj > 0, j, lastnz)
        prev = c
    tin = tl.where(found < 0, float("inf"), tin)
    found = tl.where(found < 0, lastnz, found)
    tl.store(OB + row, found)
    tl.store(OT + row, tin)


@triton.jit
def _sample_column_rows_kernel(ZP, ZQ, IP, IQ, LP, LQ, Bw, BLK, TIN, OUT, sp, sq, voff, vlim, nb_local,
                               HAS_Q: tl.constexpr, BLOCK: tl.constexpr, TILE: tl.constexpr):
    # Task D ``_sample_column_kernel`` with the p / q rows read through IP / IQ.
    row = tl.program_id(0)
    ip = tl.load(IP + row).to(tl.int64)
    iq = tl.load(IQ + row).to(tl.int64)
    blk = tl.load(BLK + row)
    jl = blk - voff // BLOCK
    owned = (jl >= 0) & (jl < nb_local)
    jl = tl.where(owned, jl, 0)
    g0 = voff + jl * BLOCK
    lse_p = tl.load(LP + row)
    lse_q = tl.load(LQ + row)
    b = tl.load(Bw + row)
    tin = tl.load(TIN + row)
    carry = tl.zeros([], tl.float64)
    found = tl.full([], -1, tl.int64)
    lastnz = tl.full([], -1, tl.int64)
    for c0 in range(0, BLOCK, TILE):
        cols = c0 + tl.arange(0, TILE)
        ok = ((g0 + cols) < vlim) & owned
        zp = tl.load(ZP + ip * sp + jl * BLOCK + cols, mask=ok, other=float("-inf")).to(tl.float32)
        d = b * tl.exp((zp - lse_p).to(tl.float64))
        if HAS_Q:
            zq = tl.load(ZQ + iq * sq + jl * BLOCK + cols, mask=ok, other=float("-inf")).to(tl.float32)
            d = d - tl.exp((zq - lse_q).to(tl.float64))
        d = tl.where(ok & (d > 0), d, 0.0)
        cs = carry + tl.cumsum(d, 0)
        hit = cs > tin
        first = tl.min(tl.where(hit, cols, BLOCK), 0)
        found = tl.where((found < 0) & (first < BLOCK), (g0 + first).to(tl.int64), found)
        lnz = tl.max(tl.where(d > 0, cols, -1), 0)
        lastnz = tl.where(lnz >= 0, (g0 + lnz).to(tl.int64), lastnz)
        carry = carry + tl.sum(d, 0)
    found = tl.where(found < 0, lastnz, found)
    found = tl.where(owned, found, -1)
    tl.store(OUT + row, found)


def _rowmajor(z: torch.Tensor) -> torch.Tensor:
    if z.dim() != 2 or z.stride(-1) != 1:
        raise ValueError("DS2: logit rows must be 2-D with unit column stride")
    if z.shape[1] % BLOCK:
        raise ValueError(f"DS2: vocab shard width {z.shape[1]} is not a multiple of {BLOCK}")
    return z


def _layout(zp: torch.Tensor, ip: torch.Tensor, zq: Optional[torch.Tensor], iq: torch.Tensor):
    """The reference defines its values on fp32 rows (``_f32``: non-fp32 rows are cast first; reading 16-bit rows in
    place would change the load layout and with it the reduction tree). fp32 rows (the fp32 LM head's logits: the
    verifier's p and q) are read in place; other dtypes have their indexed rows copied to fp32 first.
    Returns (zp, ip, zq, iq)."""
    if zp.dtype == torch.float32 and (zq is None or zq.dtype == torch.float32):
        return zp, ip, zq, iq
    ar = torch.arange(int(ip.numel()), device=zp.device)
    zp2 = zp[_idx(ip, zp.device)].float()
    zq2 = None if zq is None else zq[_idx(iq, zq.device)].float()
    return zp2, ar, zq2, ar


def _idx(i: torch.Tensor, dev) -> torch.Tensor:
    return i.to(device=dev, dtype=torch.int64).contiguous()


def residual_block_masses(zp, ip, zq, iq, lse_p, lse_q, b, vocab_offset: int, vocab_limit: int) -> torch.Tensor:
    """[n, V_local // BLOCK] fp64 block masses of sum_v (b p - q)+ on this shard, row k from zp[ip[k]] / zq[iq[k]]
    (``zq`` None: q = 0, the masses of b p). Bitwise task D ``residual_partials`` of the materialized rows."""
    zp = _rowmajor(zp)
    dev = zp.device
    n = int(ip.numel())
    nb_local = zp.shape[1] // BLOCK
    out = torch.empty(n, nb_local, dtype=torch.float64, device=dev)
    if n:
        has_q = zq is not None
        zp, ip, zq, iq = _layout(zp, ip, _rowmajor(zq) if has_q else None, iq if has_q else ip)
        zq_ = zq if has_q else zp
        ip_d = _idx(ip, dev)
        iq_d = _idx(iq, dev) if has_q else ip_d
        _residual_rows_kernel[(n, nb_local)](
            zp, zq_, ip_d, iq_d, lse_p.float().contiguous(), lse_q.float().contiguous(), b.double().contiguous(),
            out, zp.stride(0), zq_.stride(0), vocab_offset, vocab_limit, nb_local, HAS_Q=has_q, BLOCK=BLOCK,
            TILE=_TILE,
        )
    return out


def gather_blocks(local: torch.Tensor, group) -> torch.Tensor:
    """[n, nb_local] per-rank block values -> [n, nb] in global block order (rank order = vocabulary order)."""
    if group is None or torch.distributed.get_world_size(group) == 1:
        return local
    world = torch.distributed.get_world_size(group)
    out = torch.empty((world,) + tuple(local.shape), dtype=local.dtype, device=local.device)
    torch.distributed.all_gather_into_tensor(out, local.contiguous(), group=group)
    return out.permute(1, 0, 2).reshape(local.shape[0], world * local.shape[1])


def mass_combine(masses: torch.Tensor) -> torch.Tensor:
    """[n, nb] fp64 -> [n] fp64, sequential over blocks in global order (task D ``mass_combine``)."""
    n, nb = masses.shape
    out = torch.empty(n, dtype=torch.float64, device=masses.device)
    if n:
        _mass_combine_kernel[(n,)](masses.contiguous(), out, nb)
    return out


def sample_block(masses: torch.Tensor, bot: torch.Tensor, u: torch.Tensor):
    """Task D ``sample_block``: (block index [n] int64, nb = the extra atom; in-block target [n] fp64)."""
    n, nb = masses.shape
    ob = torch.empty(n, dtype=torch.int64, device=masses.device)
    ot = torch.empty(n, dtype=torch.float64, device=masses.device)
    if n:
        _sample_block_kernel[(n,)](masses.contiguous(), bot.double().contiguous(), u.double().contiguous(), ob, ot, nb)
    return ob, ot


def sample_column(zp, ip, zq, iq, lse_p, lse_q, b, block, t_in, vocab_offset: int, vocab_limit: int) -> torch.Tensor:
    """Task D ``sample_column`` (rows through ip / iq): the global column inside ``block`` if this shard owns it,
    else -1."""
    zp = _rowmajor(zp)
    dev = zp.device
    n = int(ip.numel())
    out = torch.empty(n, dtype=torch.int64, device=dev)
    if n:
        has_q = zq is not None
        zp, ip, zq, iq = _layout(zp, ip, _rowmajor(zq) if has_q else None, iq if has_q else ip)
        zq_ = zq if has_q else zp
        ip_d = _idx(ip, dev)
        iq_d = _idx(iq, dev) if has_q else ip_d
        _sample_column_rows_kernel[(n,)](
            zp, zq_, ip_d, iq_d, lse_p.float().contiguous(), lse_q.float().contiguous(), b.double().contiguous(),
            block.contiguous(), t_in.double().contiguous(), out, zp.stride(0), zq_.stride(0), vocab_offset,
            vocab_limit, zp.shape[1] // BLOCK, HAS_Q=has_q, BLOCK=BLOCK, TILE=_TILE,
        )
    return out


def _law_bug() -> str:
    return os.environ.get("THUNDERSYNC_DEBUG_LAW_BUG", "").split(":")[0]


def _d2h(parts: list[torch.Tensor]) -> list[torch.Tensor]:
    from thundersync_rl.block_verification_fused import _d2h as d2h

    return d2h(parts)


def _h2d(t: torch.Tensor, dev) -> torch.Tensor:
    from thundersync_rl.block_verification_fused import _h2d as h2d

    return h2d(t, dev)


class ShardedRows:
    """One draft's p rows on this rank's vocabulary shard (DS2 input).

    ``zp`` [g+1, V/TP]: the verification forward's logits at positions plen-1 .. n-1 (this shard); ``lse_p`` [g+1]
    fp32 canonical LSE of each full row; ``group`` / ``offset``: the TP group and this shard's first column.
    """

    def __init__(self, zp: torch.Tensor, vocab_limit: int, group, offset: int):
        from nemo_rl.models.megatron import vocab_lse

        self.zp = _rowmajor(zp)
        self.vocab_limit = int(vocab_limit)
        self.group = group
        self.offset = int(offset)
        self.width = int(zp.shape[1])
        self.lse = vocab_lse.lse_sharded(self.zp, self.offset, self.vocab_limit, group)

    def token_logprobs(self, tokens: torch.Tensor) -> torch.Tensor:
        """[k] fp32 log p of tokens[j] at row j: the ops of ``GatheredLogprobRows._token_logprobs_sharded``."""
        k = tokens.numel()
        local = self.zp[:k]
        tok = tokens.to(local.device).view(-1).long()
        own = (tok >= self.offset) & (tok < self.offset + self.width)
        zx = local.gather(-1, (tok - self.offset).clamp(0, self.width - 1).view(-1, 1))[:, 0].float()
        lse = self.lse[:k]
        val = torch.where(own, zx - lse, torch.zeros_like(lse))
        if self.group is not None and torch.distributed.get_world_size(self.group) > 1:
            torch.distributed.all_reduce(val, group=self.group)
        return val


def grid_scan_sharded(
    rows: ShardedRows,
    lpx: torch.Tensor,
    lqx: torch.Tensor,
    q_local: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    generator: Optional[torch.Generator],
    envelopes: tuple[float, ...],
    first_chunk: int = FIRST_CHUNK,
    chunk: int = CHUNK,
) -> tuple[dict[float, tuple[int, int, Optional[float]]], int]:
    """``block_verification_fused.block_verify_stats_lazy_grid`` on the vocabulary shard (see the module doc).

    ``lpx`` [g] log p of the drafted tokens (``rows.token_logprobs``), ``lqx`` [g] the engine's log q.
    ``q_local(idx)`` (idx: host int64 positions): (q logits [k, V/TP] of this shard at those positions, their
    canonical LSE [k] fp32). It is a collective at TP > 1: every rank of the group asks for the same positions.
    Returns ({M: (tau, y, log p(y) or None for BOT)}, positions evaluated).
    """
    from thundersync_rl.prof import timed

    _t_setup = timed("ds2/setup")
    _t_setup.__enter__()
    g = lpx.numel()
    zp, lse_p_all = rows.zp, rows.lse
    dev = zp.device
    off, vlim, group = rows.offset, rows.vocab_limit, rows.group
    zeros = torch.zeros(g + 1, dtype=torch.float32, device=dev)
    lse_p0, lse_q0 = zeros, zeros[:g]  # the gathered scan's arguments: its rows are log-prob rows
    weights: dict[float, torch.Tensor] = {}
    tau: dict[float, int] = {}
    cand: dict[float, torch.Tensor] = {}
    # Setup: _grid_packed's, value for value (input check, draws, slack, weights; one packed host copy).
    parts = [lpx.float(), lqx.float().to(dev), lse_p0.float(), lse_q0.float()]
    flat_in = torch.cat([q.reshape(-1) for q in parts])
    p_zero = lpx == float("-inf")
    ok = torch.isfinite(flat_in)
    ok[: lpx.numel()] |= p_zero.reshape(-1)
    eta = torch.rand(g + 1, generator=generator, dtype=torch.float64, device=dev)
    u_d = torch.rand(1, generator=generator, dtype=torch.float64, device=dev)
    slack = lazy_slack(lse_p0[:g])
    for M in envelopes:
        check_null_m(M, g)
    s_cum = torch.cat([torch.zeros(1, dtype=torch.float64, device=dev), torch.cumsum((lpx - lqx).double(), 0)])
    S = s_cum.unsqueeze(0).repeat(len(envelopes), 1)
    for k, M in enumerate(envelopes):
        if M != 1.0:
            S[k, 1:] -= math.log(M)
    B = (S - torch.cummax(S, 1).values).exp()
    slack_t = slack if isinstance(slack, torch.Tensor) else torch.zeros(0, dtype=torch.float64, device=dev)
    ok_h, n0_h, eta_h, u_h, slack_hh, B_h = _d2h([ok.all().view(1), p_zero.sum().view(1), eta, u_d, slack_t, B])
    if not bool(ok_h[0]):
        check_verification_inputs(lpx, lqx, lse_p0, lse_q0)  # raises
    _COUNTS["drafts_checked"] += 1
    _COUNTS["p_zero_tokens"] += int(n0_h[0]) if g else 0
    _t_setup.__exit__(None, None, None)
    slack_h = slack_hh if isinstance(slack, torch.Tensor) else slack
    B_h = B_h.view(len(envelopes), g + 1)
    global LAST_ROW
    if SHADOW:
        LAST_ROW = {"g": int(g), "eta_g": float(eta_h[g]),
                    "b_g": {float(M): float(B_h[k, g]) for k, M in enumerate(envelopes)}}
    for k, M in enumerate(envelopes):
        weights[M], b_h = B[k], B_h[k]
        if bool(eta_h[g] <= b_h[g]):
            tau[M] = g
        else:
            cand[M] = torch.nonzero(eta_h[:g] <= b_h[:g] * (1 + slack_h)).flatten().flip(0)

    # Scan: the union of the live envelopes' candidates from the top, FIRST_CHUNK then CHUNK positions at a time.
    q_keep: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}  # stop position -> (q logits row [1, V/TP], lse [1])
    n_eval = 0
    done = {M: 0 for M in cand}
    size = first_chunk
    while any(M not in tau for M in cand):
        live = [M for M in cand if M not in tau]
        for M in live:
            if done[M] >= cand[M].numel():
                raise RuntimeError("block verification: no stop position (eta > h everywhere)")
        pending = torch.unique(torch.cat([cand[M][done[M]:] for M in live]))
        idx = pending.flip(0)[:size]
        size = chunk
        lo = int(idx[-1])
        with timed("ds2/q_local"):
            zq, lse_q = q_local(idx)
        COUNTS["chunks"] += 1
        n_eval += idx.numel()
        asc = idx.flip(0)
        pairs = []
        for M in live:
            c = cand[M][done[M]:]
            c = c[c >= lo]
            done[M] += c.numel()
            if c.numel():
                pairs.append((M, c))
        if pairs:
            with timed("ds2/mass"):
                cs = torch.cat([c for _, c in pairs])
                sel = _h2d(idx.numel() - 1 - torch.searchsorted(asc, cs), dev)
                c_d = _h2d(cs, dev)
                a0, bis = 0, []
                for M, c in pairs:
                    bis.append(weights[M][c_d[a0 : a0 + c.numel()]])
                    a0 += c.numel()
                bi = torch.cat(bis)
                root = c_d == 0
                inv = torch.cat([torch.full((c.numel(),), M, dtype=torch.float64, device=dev) for M, c in pairs])
                nonone = inv != 1.0
                b_res = torch.where(root & nonone, bi / inv, bi)
                m_loc = residual_block_masses(zp, c_d, zq, sel, lse_p_all[c_d], lse_q[sel], b_res, off, vlim)
                r = mass_combine(gather_blocks(m_loc, group))
                r = torch.where(root & nonone, r + (1.0 - 1.0 / inv), r)
                h_c = stop_probability(r, bi)
                fin_h, hit = _d2h([torch.isfinite(h_c).all().view(1), eta[c_d] <= h_c])
                if not bool(fin_h[0]):
                    check_stop_probabilities(h_c, "grid scan (DS2)")  # raises
                hit = hit.bool()
                a = 0
                for M, c in pairs:
                    hm = torch.nonzero(hit[a : a + c.numel()]).flatten()
                    a += c.numel()
                    if hm.numel():
                        tau[M] = int(c[hm[0]])
        for M in live:  # keep the q rows of the stop positions for the emissions
            t = tau.get(M)
            if t is not None and t >= lo and t not in q_keep:
                k = idx.numel() - 1 - int(torch.searchsorted(asc, torch.tensor(t)))
                q_keep[t] = (zq[k : k + 1].clone(), lse_q[k : k + 1].clone())
        del zq, lse_q

    # Emissions: every envelope's draw on the device; one host copy of (ok, y, log p(y)) for all of them.
    _t_emit = timed("ds2/emit")
    _t_emit.__enter__()
    bug = _law_bug()
    dev_parts: list[torch.Tensor] = []
    meta = []
    one = torch.ones(1, dtype=torch.float64, device=dev)
    # Emission dedup (unconditional; emit_dedup_enabled): the bonus emission (tau = g: a draw from p at row g with u) does not
    # depend on M, so envelopes accepted whole share one computation (the same kernels on the same inputs: the same
    # bits); the p masses are computed only where an emission reads them (a pure function: no draw, no state).
    dedup = emit_dedup_enabled()
    bonus_parts: Optional[list] = None
    for M in envelopes:
        t, b = tau[M], weights[M]
        if _reuse_bonus(dedup, t, g) and bonus_parts is not None:
            dev_parts += bonus_parts
            meta.append((M, t))
            COUNTS["emit_dedup"] += 1
            continue
        ip = torch.full((1,), t, dtype=torch.int64, device=dev)
        lp_t = lse_p_all[t : t + 1]
        p_masses = None
        if not (dedup and t == 0 and t != g and M != 1.0 and not bug):  # skipped only for a ⊥-root emission
            p_masses = gather_blocks(residual_block_masses(zp, ip, None, ip, lp_t, lp_t, one, off, vlim), group)
        if t == g:  # bonus row: p
            masses, bot, zq_t, lq_t, b_e, has_q = p_masses, 0.0, None, lp_t, one, False
        else:
            if t not in q_keep:
                zq1, lq1 = q_local(torch.tensor([t], dtype=torch.int64))
                q_keep[t] = (zq1, lq1)
            zq_t, lq_t = q_keep[t]
            m = M if t == 0 else 1.0
            if m != 1.0 and bug == "null_m_1":
                m = 1.0  # debug negative control: root residual without the ⊥ envelope (no BOT atom)
            b_e = (b[t] / m if m != 1.0 else b[t]).view(1)
            bot = 1.0 - 1.0 / m if m != 1.0 else 0.0
            has_q = True
            masses = gather_blocks(
                residual_block_masses(zp, ip, zq_t, torch.zeros(1, dtype=torch.int64, device=dev), lp_t, lq_t,
                                      b_e, off, vlim),
                group,
            )
            if m == 1.0 and bug == "residual_from_p":
                masses, has_q, b_e = p_masses, False, one  # debug negative control (see block_verification_fused._emit)
        tot = mass_combine(masses)
        bot_t = torch.full((1,), bot, dtype=torch.float64, device=dev)
        if has_q and bot == 0.0:
            # An empty residual (p_tau == q_tau exactly) falls back to p, as emit_distribution does; selected on the
            # device (both draws are taken, one is kept).
            empty = ~(tot > 0)
            blk_r, tin_r = sample_block(masses, bot_t, u_d)
            col_r = sample_column(zp, ip, zq_t, torch.zeros(1, dtype=torch.int64, device=dev), lp_t, lq_t, b_e,
                                  blk_r, tin_r, off, vlim)
            blk_p, tin_p = sample_block(p_masses, bot_t, u_d)
            col_p = sample_column(zp, ip, None, ip, lp_t, lp_t, one, blk_p, tin_p, off, vlim)
            blk = torch.where(empty, blk_p, blk_r)
            col = torch.where(empty, col_p, col_r)
            tot = torch.where(empty, mass_combine(p_masses), tot)
        else:
            blk, tin = sample_block(masses, bot_t, u_d)
            if has_q:
                col = sample_column(zp, ip, zq_t, torch.zeros(1, dtype=torch.int64, device=dev), lp_t, lq_t, b_e,
                                    blk, tin, off, vlim)
            else:
                col = sample_column(zp, ip, None, ip, lp_t, lp_t, b_e, blk, tin, off, vlim)
        if group is not None and torch.distributed.get_world_size(group) > 1:
            torch.distributed.all_reduce(col, op=torch.distributed.ReduceOp.MAX, group=group)
        nb = masses.shape[1]
        is_bot = blk == nb
        y = torch.where(is_bot, torch.full_like(col, NULL_TOKEN), col)
        # log p(y) on the owner rank (fp32(z_y) - LSE), +0 elsewhere, summed: the token log-prob's ops.
        yy = y.clamp(min=0)
        own = (yy >= off) & (yy < off + rows.width) & ~is_bot
        zy = zp[t : t + 1].gather(-1, (yy - off).clamp(0, rows.width - 1).view(1, 1))[:, 0].float()
        lp_y = torch.where(own, zy - lp_t, torch.zeros_like(lp_t))
        if group is not None and torch.distributed.get_world_size(group) > 1:
            torch.distributed.all_reduce(lp_y, group=group)
        total = tot + bot_t
        ok_e = torch.isfinite(total) & (total > 0) & (is_bot | (y >= 0))
        dev_parts += [ok_e.view(1), y.view(1), lp_y.view(1)]
        meta.append((M, t))
        if dedup and t == g:
            bonus_parts = dev_parts[-3:]
    got = _d2h(dev_parts)
    _t_emit.__exit__(None, None, None)
    out: dict[float, tuple[int, int, Optional[float]]] = {}
    for k, (M, t) in enumerate(meta):
        ok_e, yv, lpv = got[3 * k], got[3 * k + 1], got[3 * k + 2]
        if not bool(ok_e[0]):
            check_emission(torch.full((1,), float("nan")), f"M={M} tau={t} of g={g} (DS2)")  # raises
        y = int(yv[0])
        out[M] = (t, y, None if y < 0 else float(lpv[0]))
        COUNTS["bot" if y < 0 else ("bonus" if t == g else ("root" if t == 0 else "residual"))] += 1
    COUNTS["rows"] += 1
    COUNTS["evals"] += n_eval
    COUNTS["p_dtype"] = str(zp.dtype).replace("torch.", "")
    if q_keep:
        COUNTS["q_dtype"] = str(next(iter(q_keep.values()))[0].dtype).replace("torch.", "")
    return out, n_eval


def _reuse_bonus(dedup: bool, t: int, g: int) -> bool:
    """An envelope's emission may reuse the bonus emission: only when it is a bonus (tau = g) itself."""
    return dedup and t == g


def vwall_check(rows, lpx, lqx, q_local_ref, gen_state_before, envelopes, out, n_eval, gen_after) -> bool:
    """THUNDERSYNC_VWALL_CHECK's comparison: the same row on the old path (per-envelope emissions; ``q_local_ref``)
    from ``gen_state_before`` must give ``out`` at every envelope, ``n_eval`` positions and the final generator state
    ``gen_after`` (bitwise). Returns True when equal (a collective on every TP rank, like the scan)."""
    global _FORCE_DEDUP
    ref_gen = torch.Generator(device=lpx.device)
    ref_gen.set_state(gen_state_before)
    prev, _FORCE_DEDUP = _FORCE_DEDUP, False
    try:
        ref_out, ref_n = grid_scan_sharded(rows, lpx, lqx, q_local_ref, ref_gen, envelopes)
    finally:
        _FORCE_DEDUP = prev
    return ref_out == out and ref_n == n_eval and torch.equal(ref_gen.get_state(), gen_after)


_FORCE_DEDUP: Optional[bool] = None  # THUNDERSYNC_VWALL_CHECK's reference re-run forces the per-envelope path


def emit_dedup_enabled() -> bool:
    """Emission dedup (unconditional; same bits): one bonus emission per draft for every envelope accepted whole. Only
    THUNDERSYNC_VWALL_CHECK's reference re-run (and tests) turn it off, through ``_FORCE_DEDUP``."""
    return True if _FORCE_DEDUP is None else _FORCE_DEDUP


def eligible(p_lp, h, exact_law: bool, nucleus: bool) -> bool:
    """DS2 applies to the gathered TP path of the ⊥-BV grid: vocab-sharded logits with whole canonical blocks,
    canonical-block log-probs on, no exact integer law, no exact top-p, q rows from logged hidden states."""
    from nemo_rl.models.megatron import vocab_lse

    from thundersync_rl.engine_q import GatheredLogprobRows, HiddenRows, TPVocabGather

    return (
        enabled()
        and vocab_lse.enabled()
        and not exact_law
        and not nucleus
        and type(p_lp) is GatheredLogprobRows
        and p_lp.anchors is None
        and isinstance(p_lp.gather, TPVocabGather)
        and int(p_lp.local.shape[-1]) % BLOCK == 0
        and type(h) is HiddenRows
    )


def version_array(segments, g: int) -> np.ndarray:
    """[g] int64 drafting version per position (the worker's ``version`` list as an array)."""
    v = np.zeros(g, dtype=np.int64)
    for ver, a, b in segments:
        v[a:b] = ver
    return v
