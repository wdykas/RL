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
"""Multi-token tail decode (MTD): exact lookup speculation on opted-in engine requests (ThunderSync tails).

Design: the research notes, summarized in docs/colo/ Per engine iteration an opted-in request ("tail")
may carry up to K extra decode rows holding proposals: the tokens that followed the longest suffix match (n <= 4)
of its context in its references (the rejected draft from tau on, then the group's other drafts).
Every row is sampled by the engine's keyed sampler with the key one-token decode would use for that position, and
proposals are accepted while they equal the sampled tokens (sample-and-match). Row j <= a - 1 then has exactly
the prefix one-token decode has at that position, so the emitted tokens and log-probs are bitwise one-token
decode's: the speculation changes only the number of iterations.

Device-resolved (MCore async scheduling keeps overlapping forward N+1 with the host's resolve of N):
- device tail table (by tail id): committed length P, history H (prompt + generated), references R, budget, EOS,
  done; written at admission, updated by ``_accept_kernel`` after each sampling;
- host layout before each forward: per tail its row 0 entry, width k (from its resolved acceptance history,
  ``width``), first extra entry; extra rows sit after the step's request rows, inside the CUDA-graph padding (the
  graph is matched on N + E rows);
- ``_lookup_kernel`` + ``_expand_kernel`` after the bookkeeping H2D: proposals, and the rows' input ids, positions,
  KV slots / lengths, block-table rows, Mamba slots and lookahead offsets (row 0 too: the host's lengths lag the
  device's by up to one iteration);
- after MCore samples: the extra rows are drawn with the same keyed sampler, ``_accept_kernel`` finds a, appends
  to H, advances P, and the Mamba cursor / conv state commit runs (``mamba_bi_lookahead.commit``);
- at resolve the host appends the a tokens and their log-probs to the request, finishes on EOS / budget from the
  device result, and moves the noise-key checks by a - 1 (keys are checked at commit, not at draw).

Per rank: device table T_MAX x (max_seq + ref_cap) int32; <= E_cap extra decode rows per iteration on the ranks
that host tails; no communication (proposals are computed identically on every TP rank).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

# SamplingParams attribute (opts the request in): its reference tokens as int32 bytes, references separated by -1,
# in preference order (the rejected draft from tau on, then the group's other drafts); ``pack_refs`` builds it.
MTD_REFS_ATTR = "ts_mtd_refs"


def pack_refs(refs) -> np.ndarray:
    """[ref0, -1, ref1, -1, ...] int32 (driver side)."""
    parts = []
    for r in refs:
        parts.append(np.asarray(r, dtype=np.int32))
        parts.append(np.full(1, SEP, dtype=np.int32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)
NMAX = 4  # longest suffix matched
SEP = -1  # reference separator
# Width rule (``width``): expected tokens per unit cost, (1 - alpha^{k+1}) / (1 - alpha) / (1 + BETA k).
# BETA = marginal decode-row cost / iteration cost, measured on the Super-L22 engine (bench): 0.022 ms per
# row at 144-192 rows/rank (6.0 ms iteration) and ~0.05 ms per row at 1-16 rows (3.2 ms); 0.01 lies between.
BETA = 0.01
# Acceptance per allocated extra row, from the past only (proposals change; decisions stay exact):
# - pooled over the engine's tails (one engine = one model and task): mean (1/2 + A_g) / (1 + N_g), counts forgotten
#   per forward with a horizon of 2000 forwards (1 / (1 - POOL_FORGET));
# - per tail: (pooled mean + A_t) / (1 + N_t), its own counts forgotten over ~1000 iterations.
# Text no proposal can match (alpha < BETA) gets no rows: new tails start at the pooled mean, so after the first ~50
# failed rows of an engine nothing is proposed until the pooled evidence has decayed (a few probes per 2000 forwards).
PRIOR_ACC, PRIOR_TRIALS, FORGET, POOL_FORGET = 0.5, 1.0, 0.999, 0.9995


if HAVE_TRITON:

    @triton.jit
    def _lookup_kernel(H, R, RK, RLEN, P, lay_tid, lay_k, best, stride_h, stride_r,
                       BLOCK: tl.constexpr, NMAXC: tl.constexpr):
        """Program (layout entry j, reference block b): best (n, -reference rank, position) of the suffix match of
        tail j's last NMAXC tokens over its reference positions; atomic max into best[j]."""
        j = tl.program_id(0)
        b = tl.program_id(1)
        tid = tl.load(lay_tid + j)
        k = tl.load(lay_k + j)
        if (tid < 0) | (k <= 0):
            return
        tid64 = tid.to(tl.int64)
        p = tl.load(P + tid64)
        hrow = H + tid64 * stride_h
        rrow = R + tid64 * stride_r
        e = b * BLOCK + tl.arange(0, BLOCK)
        rl = tl.load(RLEN + tid64)
        valid = (e >= 1) & (e < rl)
        n = tl.zeros([BLOCK], dtype=tl.int64)
        run = valid
        for d in tl.static_range(NMAXC):
            kp = p - 1 - d  # key token (the context's last tokens, newest first)
            key = tl.load(hrow + kp, mask=kp >= 0, other=-7)
            pos = e - 1 - d
            v = tl.load(rrow + pos, mask=run & (pos >= 0), other=-5)
            m = run & (pos >= 0) & (v == key) & (key >= 0)
            n += m.to(tl.int64)
            run = m
        rank = tl.load(RK + tid64 * stride_r + e, mask=valid, other=15).to(tl.int64)
        fv = tl.load(rrow + e, mask=valid, other=-1)
        follow_ok = valid & (fv >= 0)
        score = tl.where(follow_ok & (n > 0), (n << 40) | ((15 - rank) << 32) | e.to(tl.int64), -1)
        tl.atomic_max(best + j, tl.max(score, axis=0))

    @triton.jit
    def _attach_kernel(H, P, lay, in_ids, stride_h, stride_lay):
        """Program j (attach entries, flag 2): the host uploaded the history through the forward before last; this
        forward's row-0 input is the token the last forward committed (an untracked tail's MCore view is the truth):
        H[P] <- it, P += 1. Runs before the lookup so the proposal key ends with it."""
        j = tl.program_id(0)
        tid = tl.load(lay + 0 * stride_lay + j)
        flag = tl.load(lay + 6 * stride_lay + j)
        if (tid < 0) | (flag != 2):
            return
        tid64 = tid.to(tl.int64)
        r0 = tl.load(lay + 1 * stride_lay + j).to(tl.int64)
        p = tl.load(P + tid64)
        tl.store(H + tid64 * stride_h + p, tl.load(in_ids + r0).to(tl.int32))
        tl.store(P + tid64, p + 1)

    @triton.jit
    def _expand_kernel(
        H, R, RLEN, P, BUD, best, lay, prop_out, kk_out, p_out, sid_out,
        in_ids, pos_ids, blk_idx, loc_pos, pos_in_req, req_idx, q_len, kv_len, btab, mamba_bi, logit_idx,
        look_off, look_chain, scratch_id,
        stride_h, stride_r, stride_lay, stride_prop, stride_bt, stride_chain, stride_sid,
        n_kv_cols, dummy_block, scratch_base, n_req_rows,
        K: tl.constexpr, CHUNK: tl.constexpr, BS: tl.constexpr, KPOW: tl.constexpr, BTCOLS: tl.constexpr,
        LOOK_MAX: tl.constexpr, ALWAYS: tl.constexpr,
    ):
        """Program j: layout entry j's proposal (from best[j]) and its rows: row 0's overrides, extra rows 1..k.
        ALWAYS (negative control mtd_accept_all only): without a match, propose the last token repeated."""
        j = tl.program_id(0)
        tid = tl.load(lay + 0 * stride_lay + j)
        if tid < 0:
            return
        r0 = tl.load(lay + 1 * stride_lay + j)
        k = tl.load(lay + 2 * stride_lay + j)
        x0 = tl.load(lay + 3 * stride_lay + j)
        slot = tl.load(lay + 4 * stride_lay + j)
        pf = tl.load(lay + 6 * stride_lay + j)
        tid64 = tid.to(tl.int64)
        p = tl.load(P + tid64)
        bud = tl.load(BUD + tid64)
        if pf == 1:
            # the tail's prefill forward: MCore's layout is right; acceptance appends its sampled first token
            tl.store(kk_out + j, 0)
            tl.store(p_out + j, p)
            return
        hrow = H + tid64 * stride_h
        # effective width: Mamba chunk room (no speculative row completes a chunk), budget, proposal length
        w0 = (p - 1) % CHUNK
        kk = tl.minimum(k, CHUNK - 2 - w0)
        kk = tl.minimum(kk, bud - 1 - p)
        kk = tl.maximum(kk, 0)
        sc = tl.load(best + j)
        has = sc >= 0
        e = (sc & 0xFFFFFFFF).to(tl.int64)  # score = n << 40 | (15 - rank) << 32 | reference position
        offs = tl.arange(0, KPOW)
        rl = tl.load(RLEN + tid64)
        d = tl.load(R + tid64 * stride_r + e + offs, mask=has & (offs < kk) & (e + offs < rl), other=-1)
        if ALWAYS:
            d = tl.where(has, d, tl.load(hrow + p - 1))
            has = sc > -2  # always true
        ok = has & (offs < kk) & (d >= 0)
        # proposal length: first invalid token cuts it (cumulative AND)
        okc = tl.cumsum((~ok).to(tl.int32), axis=0) == 0
        kk2 = tl.sum((okc & ok).to(tl.int32), axis=0)
        kk2 = tl.where(has, kk2, 0)
        tl.store(prop_out + j * stride_prop + offs, tl.where(offs < kk2, d, -1), mask=offs < K)
        tl.store(kk_out + j, kk2)
        tl.store(p_out + j, p)
        # ---- row 0 (entry r0): the device's position (the host's may lag) ----
        last = tl.load(hrow + p - 1)
        pos0 = p - 1
        r0_64 = r0.to(tl.int64)
        tl.store(in_ids + r0_64, last.to(tl.int64))
        tl.store(pos_ids + r0_64, pos0.to(tl.int64))
        b0 = tl.load(btab + r0_64 * stride_bt + pos0 // BS)
        tl.store(blk_idx + r0_64, b0.to(tl.int64))
        tl.store(loc_pos + r0_64, pos0 % BS)
        tl.store(pos_in_req + r0_64, pos0)
        tl.store(kv_len + r0_64, p)
        # ---- extra rows i = 1..k (entries x0 + i - 1); rows i > kk2 masked ----
        cols = tl.arange(0, BTCOLS)
        bt_row0 = tl.load(btab + r0_64 * stride_bt + cols, mask=cols < n_kv_cols, other=0)
        for i in tl.static_range(1, K + 1):
            if i <= k:
                ex = (x0 + i - 1).to(tl.int64)
                live = i <= kk2
                pos = p - 1 + i
                di = tl.sum(tl.where(offs == i - 1, d, 0), axis=0)
                bi_blk = tl.load(btab + r0_64 * stride_bt + pos // BS, mask=live, other=0)
                tl.store(in_ids + ex, tl.where(live, di, 0).to(tl.int64))
                tl.store(pos_ids + ex, tl.where(live, pos, 0).to(tl.int64))
                tl.store(blk_idx + ex, tl.where(live, bi_blk, dummy_block).to(tl.int64))
                tl.store(loc_pos + ex, tl.where(live, pos % BS, 0))
                tl.store(pos_in_req + ex, tl.where(live, pos, 0))
                tl.store(req_idx + ex, r0)
                # Every extra row keeps one query token (masked rows too): the varlen query offsets (cumulative
                # query lengths) must stay aligned with the token rows of the rows after it. A masked row attends
                # one key of the dummy block (finite, unused) and writes its K / V there.
                tl.store(q_len + ex, 1)
                tl.store(kv_len + ex, tl.where(live, p + i, 1))
                tl.store(btab + ex * stride_bt + cols, tl.where(live, bt_row0, dummy_block), mask=cols < n_kv_cols)
                tl.store(mamba_bi + ex, tl.where(live, slot, -1))
                tl.store(logit_idx + ex, ex.to(tl.int32))
                tl.store(look_off + ex, tl.where(live, i, 0))
                sid = scratch_base + (ex - n_req_rows)
                tl.store(scratch_id + ex, tl.where(live, sid, -1).to(tl.int32))
                tl.store(sid_out + j * stride_sid + (i - 1), tl.where(live, sid, -1).to(tl.int32))
                lc = tl.arange(0, LOOK_MAX)
                chain = tl.where(lc == 0, r0, x0 + lc - 1)
                tl.store(look_chain + ex * stride_chain + lc, tl.where(lc < i, chain, -1))

    @triton.jit
    def _accept_kernel(
        H, P, BUD, EOS, DONE, lay, row_now, s0, s_ext, prop, kk_in, p_in, sid_in,
        acc_out, tok_out, slots_out, sids_out, cur_out,
        stride_h, stride_lay, stride_prop, stride_tok, stride_sid,
        K: tl.constexpr, KPOW: tl.constexpr, CHUNK: tl.constexpr, ACCEPT_ALL: tl.constexpr = False,
    ):
        """Program j: sample-and-match acceptance of tail j's rows; H / P / done updated; Mamba commit inputs."""
        j = tl.program_id(0)
        tid = tl.load(lay + 0 * stride_lay + j)
        rn = tl.load(row_now + j)
        slot = tl.load(lay + 4 * stride_lay + j)
        x0 = tl.load(lay + 3 * stride_lay + j)
        if (tid < 0) | (rn < 0):
            tl.store(acc_out + j, 0)
            tl.store(slots_out + j, -1)
            return
        tid64 = tid.to(tl.int64)
        done = tl.load(DONE + tid64)
        p = tl.load(p_in + j)
        if done != 0:
            tl.store(acc_out + j, 0)
            tl.store(slots_out + j, -1)
            return
        kk = tl.load(kk_in + j)
        eos = tl.load(EOS + tid64)
        bud = tl.load(BUD + tid64)
        offs = tl.arange(0, KPOW)
        t0 = tl.load(s0 + rn).to(tl.int64)
        # s_i (i >= 1) = the extra row of entry x0 + i - 1, stored at s_ext[x0 + i - 1 - n_rows]
        n_rows = tl.load(lay + 5 * stride_lay)
        s_extra = tl.load(s_ext + (x0 - n_rows) + offs - 1, mask=(offs >= 1) & (offs <= kk), other=-9).to(tl.int64)
        s_all = tl.where(offs == 0, t0, s_extra)
        dprop = tl.load(prop + j * stride_prop + offs, mask=offs < kk, other=-11).to(tl.int64)
        # a = 1 + number of leading i with s_{i-1} == d_i (i = 1..kk), cut after EOS
        match = (offs < kk) & (s_all == dprop)  # s_i == d_{i+1}
        if ACCEPT_ALL:  # negative control mtd_accept_all: every proposal accepted whatever the draws
            match = offs < kk
        lead = tl.cumsum((~match).to(tl.int32), axis=0) == 0
        a = 1 + tl.sum((lead & match).to(tl.int32), axis=0)
        is_eos = (s_all == eos) & (offs < a)
        first_eos = tl.min(tl.where(is_eos, offs, KPOW + 1), axis=0)
        a = tl.where(first_eos < a, first_eos + 1, a)
        a = tl.minimum(a, bud - p)
        a = tl.maximum(a, 1)
        hrow = H + tid64 * stride_h
        tl.store(hrow + p + offs, s_all.to(tl.int32), mask=offs < a)
        tl.store(P + tid64, p + a)
        last = tl.sum(tl.where(offs == a - 1, s_all, 0), axis=0)
        fin = (last == eos) | (p + a >= bud)
        tl.store(DONE + tid64, fin.to(tl.int32))
        tl.store(acc_out + j, a)
        tl.store(tok_out + j * stride_tok + offs, s_all, mask=offs < a)
        tl.store(slots_out + j, tl.where(kk > 0, slot, -1))
        tl.store(cur_out + j, (p - 1) % CHUNK)
        tl.store(sids_out + j * stride_sid + offs, tl.load(sid_in + j * stride_sid + offs, mask=offs < K, other=-1),
                 mask=offs < K)


@dataclass
class _Tail:
    rid: int
    tid: int
    plen: int
    budget_end: int
    eos: int
    refs: list
    inited: bool = False
    acc: float = 0.0  # observed accepted extra rows / allocated-row trials (decayed); prior added in ``width``
    trials: float = 0.0
    tracked: bool = False  # device-resolved (extra rows now or results pending); else MCore manages the row alone
    refs_loaded: bool = False
    k_last: int = 0
    iters: int = 0
    tokens: int = 0
    extra_rows: int = 0


def width(acc: float, trials: float, kmax: int, beta: float = BETA) -> int:
    """k in 0..kmax maximizing expected emitted tokens per unit cost under a geometric acceptance model."""
    alpha = min(max(acc / max(trials, 1e-9), 0.0), 0.999)
    best_k, best = 0, 1.0
    for k in range(1, kmax + 1):
        v = (1.0 - alpha ** (k + 1)) / (1.0 - alpha) / (1.0 + beta * k)
        if v > best + 1e-12:
            best_k, best = k, v
    return best_k


@dataclass
class _Forward:
    """One forward's layout (host) and its device ring slot."""

    slot: int
    tails: list  # [(_Tail, row0 entry, k)]
    n_rows: int  # request rows of the forward (extra rows start here)
    E: int
    staging: Any = None


class TailMTD:
    """Engine-side runtime (one per engine). Install with ``install(engine, ...)`` after the engine is built."""

    RING = 4

    def __init__(self, engine, K: int, t_max: int, ref_cap: int, extra_cap: int, sample_rows, salt_info=None):
        from nemo_rl.models.megatron import mamba_bi_lookahead as la

        ctx = engine.context
        self.engine, self.ctx, self.ctrl = engine, ctx, engine.controller
        self.K, self.t_max, self.ref_cap = int(K), int(t_max), int(ref_cap)
        self.extra_cap = min(int(extra_cap), la.capacity())
        assert 1 <= self.K <= 7 and self.extra_cap > 0  # a <= K + 1 tokens fit the 8-wide result rows
        self.sample_rows = sample_rows  # (logits rows, rids np, positions np, temperature) -> int64 tokens
        dev = torch.device("cuda", torch.cuda.current_device())
        self.dev = dev
        S = int(ctx.max_sequence_length) + 8 * self.K + 8
        self.S = S
        i32 = dict(dtype=torch.int32, device=dev)
        self.H = torch.zeros(self.t_max, S, **i32)
        self.R = torch.full((self.t_max, self.ref_cap), SEP, **i32)
        self.RK = torch.full((self.t_max, self.ref_cap), 15, dtype=torch.int8, device=dev)
        self.RLEN = torch.zeros(self.t_max, **i32)
        self.P = torch.zeros(self.t_max, **i32)
        self.BUD = torch.zeros(self.t_max, **i32)
        self.EOS = torch.full((self.t_max,), -1, **i32)
        self.DONE = torch.ones(self.t_max, **i32)
        T = self.t_max
        self.ring = []
        for _ in range(self.RING):
            self.ring.append({
                "lay": torch.full((8, T), -1, **i32),
                "best": torch.empty(T, dtype=torch.int64, device=dev),
                "prop": torch.full((T, 8), -1, **i32),
                "kk": torch.zeros(T, **i32),
                "p": torch.zeros(T, **i32),
                "sid": torch.full((T, 8), -1, **i32),
                "row_now": torch.full((T,), -1, **i32),
                "s_ext": torch.zeros(max(self.extra_cap, 1), dtype=torch.int64, device=dev),
                "acc": torch.zeros(T, **i32),
                "tok": torch.zeros(T, 8, dtype=torch.int64, device=dev),
                "slots": torch.full((T,), -1, **i32),
                "sids": torch.full((T, 8), -1, **i32),
                "cur": torch.zeros(T, **i32),
                "lp_ext": torch.zeros(max(self.extra_cap, 1), dtype=torch.float32, device=dev),
                # acc, kk, done (3 T) + tokens (8 T) + extra log-probs (<= extra_cap)
                "h_out": torch.zeros(T * 11 + max(self.extra_cap, 1) + 8, dtype=torch.float64).pin_memory(),
                "h_lay": torch.zeros(8, T, dtype=torch.int32).pin_memory(),
                "h_rn": torch.zeros(T, dtype=torch.int32).pin_memory(),
                "ev": torch.cuda.Event(),
            })
        self.ring_next = 0
        self.tails: dict[int, _Tail] = {}  # rid -> tail
        self.free_tids = list(range(self.t_max - 1, -1, -1))
        # Per-tid mirrors of the tails' host state, so a forward's plan costs O(tails that may get rows), not
        # O(registered tails): request id (-1 free), decayed accepted / trial counts, tracked, width of the last forward.
        self._rid_of = np.full(self.t_max, -1, dtype=np.int64)
        self._acc = np.zeros(self.t_max, dtype=np.float64)
        self._tri = np.zeros(self.t_max, dtype=np.float64)
        self._trk = np.zeros(self.t_max, dtype=bool)
        self._klast = np.zeros(self.t_max, dtype=np.int64)
        self._sorted = None  # (sorted request ids, their tids), rebuilt when registration changes
        self.pending: list[_Forward] = []  # forwards launched, not yet sampled (FIFO)
        self.sampled: dict[int, dict] = {}  # rid -> result of the latest sampled forward (for the resolve)
        self.layout: Optional[_Forward] = None  # set by initialize_attention_state, consumed by the forward
        self.chunk = None
        self.stats = {"iters": 0, "tail_iters": 0, "tokens": 0, "extra_rows": 0, "accepted_extra": 0,
                      "widths": [0] * (self.K + 1), "registered": 0, "forwards_with_extras": 0}
        self._extra_now = 0  # E of the step being matched to a CUDA graph
        self._seen: set = set()  # request ids already checked for opting in (dropped when they finish)
        self.gA, self.gN = 0.0, 0.0  # pooled accepted / allocated extra rows (decayed)
        self.stats.update(attaches=0, detaches=0, tracked_iters=0)
        self._debug = [] if os.environ.get("THUNDERSYNC_TAIL_MTD_DEBUG") else None
        # Gate tools (X1): a fixed width for every tail (the truncated models' text gives alpha ~ 0, i.e. width 0),
        # and the negative control mtd_accept_all (proposals accepted without the sample-and-match test).
        fk = os.environ.get("THUNDERSYNC_TAIL_MTD_DEBUG_FORCE_K", "").strip()
        self.force_k = int(fk) if fk else None
        self.neg_accept_all = os.environ.get("THUNDERSYNC_NL_NEG", "").strip() == "mtd_accept_all"
        self.stats["proposed"] = 0
        self.stats["neg_tokens"] = 0
        self._key_check = None
        self._keyreg = None
        self._key_ids = None

    # ------------------------------------------------------------------ registration
    def maybe_register(self, rid: int, req) -> Optional[_Tail]:
        t = self.tails.get(rid)
        if t is not None:
            return t
        sp = req.sampling_params
        refs = getattr(sp, MTD_REFS_ATTR, None)
        if refs is None or not self.free_tids:
            return None
        plen = len(req.prompt_tokens)
        ntg = sp.num_tokens_to_generate if sp.num_tokens_to_generate is not None else (
            (sp.num_tokens_total or self.ctx.max_sequence_length) - plen)
        eos = sp.termination_id if sp.termination_id is not None else -1
        t = _Tail(rid=rid, tid=self.free_tids.pop(), plen=plen, budget_end=min(plen + int(ntg), self.S - 8 * self.K - 8),
                  eos=int(eos), refs=refs)
        self.tails[rid] = t
        self._rid_of[t.tid] = rid
        self._acc[t.tid] = self._tri[t.tid] = 0.0
        self._trk[t.tid] = False
        self._klast[t.tid] = 0
        self._sorted = None
        self.stats["registered"] += 1
        return t

    def _upload_refs(self, t: _Tail) -> None:
        """R <- the tail's references (once per tail)."""
        r = np.frombuffer(t.refs, dtype=np.int32).copy() if isinstance(t.refs, (bytes, bytearray)) else pack_refs(t.refs)
        r = r[: self.ref_cap]
        if r.size == 0:
            r = np.full(1, SEP, dtype=np.int32)
        # rank of each position: 1 + references before it (separators close a reference)
        rk = np.minimum(1 + np.concatenate([[0], np.cumsum(r[:-1] == SEP)]), 14).astype(np.int8)
        hr = torch.from_numpy(np.ascontiguousarray(r)).pin_memory()
        hk = torch.from_numpy(rk).pin_memory()
        self.R[t.tid, : hr.numel()].copy_(hr, non_blocking=True)
        self.RK[t.tid, : hk.numel()].copy_(hk, non_blocking=True)
        self.RLEN[t.tid].fill_(int(r.size))
        self._keep_refs = (hr, hk)
        t.refs_loaded = True

    def _attach_upload(self, t: _Tail, req) -> int:
        """Attach (the tail was MCore-managed, so MCore's view is the truth): H <- prompt + generated (through the
        forward before last), P, budget, EOS, done. The expand of this forward appends its row-0 input token (the last
        forward's commit). Returns the uploaded length."""
        if not t.refs_loaded:
            self._upload_refs(t)
        pt = req.prompt_tokens
        toks = (pt.tolist() if torch.is_tensor(pt) else list(pt)) + list(req.generated_tokens)
        h = torch.tensor(toks + [len(toks), t.budget_end, t.eos], dtype=torch.int32).pin_memory()
        hd = h.to(self.dev, non_blocking=True)
        n = len(toks)
        self.H[t.tid, :n].copy_(hd[:n])
        self.P[t.tid] = hd[n]
        self.BUD[t.tid] = hd[n + 1]
        self.EOS[t.tid] = hd[n + 2]
        self.DONE[t.tid].fill_(0)
        self._keep_h = h
        t.tracked = True
        self._trk[t.tid] = True
        self.stats["attaches"] += 1
        return n

    def _detach(self, ctx, t: _Tail, idx: int, row: int, req) -> None:
        """Hand the tail back to MCore. Its last forward had no extra rows (a = 1), so its committed length is the
        host's plus one and this forward's row-0 token sits at position p = len(prompt) + len(generated). MCore's
        bookkeeping (masked while tracked) is reset to that truth: KV offset / last block / block count (blocks
        allocated ahead are released) and this forward's token row."""
        bs = int(ctx.block_size_tokens)
        p = t.plen + len(req.generated_tokens)
        col = p // bs
        cnt = int(ctx.request_kv_block_counts[idx])
        if cnt > col + 1:
            extra = ctx.request_to_kv_block_ids[idx, col + 1: cnt].clone()
            ctx.kv_block_allocator.release_memory_blocks(extra)
            ctx.request_to_kv_block_ids[idx, col + 1: cnt] = -1
            ctx.request_kv_block_counts[idx] = col + 1
        blk = ctx.request_to_kv_block_ids[idx, col]
        ctx.request_kv_length_offsets[idx] = p
        ctx.request_last_kv_block_offset[idx] = p % bs
        ctx.request_last_kv_block_id[idx] = blk
        ctx.token_to_block_idx[row] = blk
        ctx.token_to_local_position_within_kv_block[row] = p % bs
        ctx.token_to_pos_ids[row] = p
        ctx.token_to_position_in_request[row] = p
        t.tracked = False
        self._trk[t.tid] = False
        self.stats["detaches"] += 1

    def release(self, rid: int) -> None:
        t = self.tails.pop(rid, None)
        if t is not None:
            t.tracked = False
            self._rid_of[t.tid] = -1
            self._trk[t.tid] = False
            self._klast[t.tid] = 0
            self._sorted = None
            self.free_tids.append(t.tid)

    def tracked_rids(self) -> np.ndarray:
        return self._rid_of[self._trk]

    def alpha(self, t: _Tail) -> tuple[float, float]:
        """(accepted, trials) pseudo-counts of tail t: the pooled mean as one prior trial plus its own counts."""
        mu = (PRIOR_ACC + self.gA) / (PRIOR_TRIALS + self.gN)
        return mu + t.acc, 1.0 + t.trials

    # ------------------------------------------------------------------ layout (host, before each forward)
    def plan(self, ctx, is_dummy: bool) -> int:
        """This forward's tracked tails and widths; returns E (extra rows). Called from initialize_attention_state.

        A tail is tracked (device-resolved) while it gets extra rows or results of extra rows are pending; otherwise
        MCore runs it as a plain row (no MTD work at all). Host work: O(prefill rows) + vectorized selection +
        O(tails above break-even or tracked)."""
        self.layout = None
        if is_dummy:
            return 0
        n = ctx.total_request_count - ctx.paused_request_count
        if n <= 0:
            return 0
        lo = ctx.paused_request_count
        rids = ctx.request_ids[lo: lo + n].numpy()
        n_dec = ctx.num_decode_requests
        for row in range(n_dec, n):  # registration at the prefill (opted-in requests carry references)
            rid = int(rids[row])
            if rid not in self.tails and rid not in self._seen:
                self._seen.add(rid)
                self.maybe_register(rid, self.engine.get_request(rid))
        if not self.tails:
            return 0
        if self._sorted is None:
            reg = np.nonzero(self._rid_of >= 0)[0]
            o = np.argsort(self._rid_of[reg], kind="stable")
            self._sorted = (self._rid_of[reg][o], reg[o])
        srt_rids, srt_tids = self._sorted
        dec = rids[:n_dec]
        pos = np.minimum(np.searchsorted(srt_rids, dec), srt_rids.size - 1)
        rows = np.nonzero(srt_rids[pos] == dec)[0]
        if rows.size == 0:
            return 0
        tids = srt_tids[pos[rows]]
        decode_only = ctx.num_prefill_requests == 0
        # Candidates: tails with device state (tracked) and, in decode-only forwards, untracked tails whose
        # acceptance estimate (past results only) is above break-even. width(alpha) > 0 iff alpha > BETA (k = 1 gains
        # (1 + alpha) / (1 + BETA); more rows gain less when alpha <= BETA), so every other tail gets width 0 and
        # stays a plain MCore row without any per-tail host work.
        live = self._trk[tids] | (self._klast[tids] > 0)
        if decode_only:
            if self.force_k is not None:
                live[:] = True
            else:
                mu = (PRIOR_ACC + self.gA) / (PRIOR_TRIALS + self.gN)
                live |= (mu + self._acc[tids]) > BETA * (1.0 - 1e-9) * (1.0 + self._tri[tids])
        self.stats["widths"][0] += int(rows.size - np.count_nonzero(live))
        rows = rows[live]
        if rows.size == 0:
            return 0
        cap = self.extra_cap
        room = min(int(ctx.max_requests), self._max_graph_rows(ctx)) - n
        cands = []
        for row in rows.tolist():
            t = self.tails[int(rids[row])]
            req = self.engine.get_request(t.rid)
            cands.append((t, row, req))
        # longest remaining budget first: the critical tail gets rows first
        cands.sort(key=lambda c: -(c[0].budget_end - (c[0].plen + len(c[2].generated_tokens))))
        tails, E, kprev = [], 0, []
        for t, row, req in cands:
            kp = t.k_last
            k = 0
            if decode_only:
                k = width(*self.alpha(t), self.K) if self.force_k is None else min(self.force_k, self.K)
                k = max(0, min(k, cap - E, room - E))
            flag = 0
            if t.tracked:
                if k == 0 and t.k_last == 0:
                    self._detach(ctx, t, lo + row, row, req)  # nothing pending: MCore from this forward on
            elif k > 0:
                self._attach_upload(t, req)
                flag = 2
            if t.tracked:
                tails.append((t, row, k, flag))
                kprev.append(kp)
                E += k
            t.k_last = k if t.tracked else 0
            self._klast[t.tid] = t.k_last
            self.stats["widths"][k] += 1
        if not tails:
            return 0
        self._ensure_kv(ctx, tails, lo, kprev)
        self.stats["tracked_iters"] += len(tails)
        self.layout = _Forward(slot=-1, tails=tails, n_rows=n, E=E)
        return E

    def _max_graph_rows(self, ctx) -> int:
        dims = getattr(ctx, "cuda_graph_batch_dimensions_list", None) or []
        dec = [int(d.token_count) for d in dims if int(getattr(d, "prefill_req_count", 0) or 0) == 0]
        return max(dec) if dec else int(ctx.max_requests)

    def _ensure_kv(self, ctx, tails, lo, kprev) -> None:
        """KV blocks for every position a tracked tail may write before the host sees its next result: the host
        knows the committed length through the forward before last (p_host); the last forward may add k_prev + 1
        and this one writes k + 1 rows."""
        bs = int(ctx.block_size_tokens)
        alloc = ctx.kv_block_allocator
        for (t, row, k, _pf), kp in zip(tails, kprev):
            req = self.engine.get_request(t.rid)
            p_host = t.plen + len(req.generated_tokens)
            need_pos = p_host + kp + k + 2
            idx = lo + row
            have = int(ctx.request_kv_block_counts[idx])
            need = need_pos // bs + 1
            if need > have:
                if need > ctx.request_to_kv_block_ids.shape[1]:
                    need = ctx.request_to_kv_block_ids.shape[1]
                    if need <= have:
                        continue
                ids = alloc.allocate_memory_blocks(need - have)
                if ids is None:
                    raise RuntimeError("[tail_mtd] KV pool exhausted for a tail's lookahead blocks")
                ctx.request_to_kv_block_ids[idx, have:need] = ids
                ctx.request_kv_block_counts[idx] = need

    def finish_layout(self, ctx) -> None:
        """After MCore matched the graph: cancel extra rows that do not fit the padded rows (eager steps)."""
        lay = self.layout
        if lay is None:
            return
        padded = int(ctx.padded_active_token_count)
        # Extra rows only in CUDA-graph decode steps: eager steps replace the logits by the survivors' at compaction.
        if lay.E > 0 and (lay.n_rows + lay.E > padded or not ctx.using_cuda_graph_this_step()):
            lay.tails = [(t, row, 0, pf) for t, row, _, pf in lay.tails]
            lay.E = 0
            for t, _row, _k, _pf in lay.tails:
                t.k_last = 0
                self._klast[t.tid] = 0

    # ------------------------------------------------------------------ forward (device, after the H2D)
    def expand(self, ctx) -> None:
        lay = self.layout
        self.layout = None
        self.pending = []  # the forward being launched is the only one whose logits are pending
        if lay is None:
            return
        from nemo_rl.models.megatron import mamba_bi_fused_decode as fd
        from nemo_rl.models.megatron import mamba_bi_lookahead as la

        r = self.ring[self.ring_next]
        lay.slot = self.ring_next
        self.ring_next = (self.ring_next + 1) % self.RING
        T = len(lay.tails)
        h = r["h_lay"]
        h.fill_(-1)
        x = lay.n_rows
        slots = ctx.mamba_metadata.request_to_mamba_state_idx
        lo = ctx.paused_request_count
        for j, (t, row, k, pf) in enumerate(lay.tails):
            h[0, j], h[1, j], h[2, j], h[3, j] = t.tid, row, k, x
            h[4, j] = int(slots[lo + row])
            h[6, j] = pf
            x += k
        h[5, 0] = lay.n_rows
        r["lay"].copy_(h, non_blocking=True)
        lay.staging = h
        r["best"].fill_(-1)
        gv = ctx.gpu_view
        if any(pf == 2 for *_x, pf in lay.tails):
            _attach_kernel[(T,)](self.H, self.P, r["lay"], gv.token_to_input_ids, self.H.stride(0), r["lay"].stride(0))
        BLOCK = 1024
        if lay.E > 0:
            _lookup_kernel[(T, triton.cdiv(self.ref_cap, BLOCK))](
                self.H, self.R, self.RK, self.RLEN, self.P, r["lay"][0], r["lay"][2], r["best"],
                self.H.stride(0), self.R.stride(0), BLOCK=BLOCK, NMAXC=NMAX)
        off, chain = fd.lookahead_buffers(int(ctx.max_requests), self.dev)
        sid = la.scratch_ids(int(ctx.max_requests), self.dev)
        bt = gv.mha_block_table
        _expand_kernel[(T,)](
            self.H, self.R, self.RLEN, self.P, self.BUD, r["best"], r["lay"], r["prop"], r["kk"], r["p"], r["sid"],
            gv.token_to_input_ids, gv.token_to_pos_ids, gv.token_to_block_idx,
            gv.token_to_local_position_within_kv_block, gv.token_to_position_in_request, gv.token_to_request_idx,
            gv.mha_query_lengths, gv.mha_kv_seq_lengths, bt, gv.mamba_batch_indices_decode, ctx.active_logit_idxs,
            off, chain, sid,
            self.H.stride(0), self.R.stride(0), r["lay"].stride(0), r["prop"].stride(0), bt.stride(0),
            chain.stride(0), r["sid"].stride(0),
            bt.shape[1], int(ctx.kv_block_allocator.dummy_block_idx), int(ctx.max_requests), lay.n_rows,
            K=self.K, CHUNK=self._chunk(), BS=int(ctx.block_size_tokens), KPOW=8,
            BTCOLS=triton.next_power_of_2(bt.shape[1]), LOOK_MAX=fd.LOOK_MAX, ALWAYS=self.neg_accept_all,
        )
        padded = int(ctx.padded_active_request_count)
        if lay.E > 0:
            q = gv.mha_query_lengths[:padded]
            torch.cumsum(q, 0, dtype=torch.int32, out=gv.mha_cu_query_seq_lengths[1: padded + 1])
            gv.real_token_count.fill_(lay.n_rows + lay.E)
        kv = gv.mha_kv_seq_lengths[:padded]
        torch.cumsum(kv, 0, dtype=torch.int32, out=gv.mha_cu_kv_seq_lengths[1: padded + 1])
        self.pending = [lay]
        self.stats["iters"] += 1
        self.stats["tail_iters"] += T
        self.stats["extra_rows"] += lay.E
        self.stats["forwards_with_extras"] += int(lay.E > 0)

    def after_forward(self, ctx) -> None:
        """Zero the lookahead offsets of the rows this forward used (every other forward sees offset 0)."""
        if not self.pending or self.pending[-1].E == 0:
            return
        from nemo_rl.models.megatron import mamba_bi_fused_decode as fd

        off, chain = fd.lookahead_buffers(int(ctx.max_requests), self.dev)
        lay = self.pending[-1]
        off[lay.n_rows: lay.n_rows + lay.E].zero_()

    def _chunk(self) -> int:
        if self.chunk is None:
            self.chunk = int(getattr(self.ctx, "mamba_chunk_size", 128) or 128)
        return self.chunk

    # ------------------------------------------------------------------ after MCore sampled (device)
    def after_sample(self, ctrl) -> None:
        if not self.pending:
            return
        lay = self.pending.pop(0)
        ctx = self.ctx
        r = self.ring[lay.slot]
        n = ctx.total_request_count - ctx.paused_request_count
        lo = ctx.paused_request_count
        cur = ctx.request_ids[lo: lo + n].numpy()
        lay_rids = np.asarray([t.rid for t, *_ in lay.tails], dtype=np.int64)
        hit = np.nonzero(np.isin(cur, lay_rids))[0]
        rid_row = {int(cur[i]): int(i) for i in hit}
        rn = r["h_rn"]
        rn.fill_(-1)
        ext_rows, ext_rids, ext_pos, ext_temp_rows = [], [], [], []
        x = lay.n_rows
        live = []
        for j, (t, row, k, _pf) in enumerate(lay.tails):
            now = rid_row.get(t.rid, -1)
            if t.rid in self.tails and now >= 0:
                rn[j] = now
                live.append((j, t, now, k, x))
                req = self.engine.get_request(t.rid)
                p_host = t.plen + len(req.generated_tokens)
                for i in range(1, k + 1):
                    ext_rows.append(x + i - 1)
                    ext_rids.append(t.rid)
                    ext_pos.append(p_host + i)
                    ext_temp_rows.append(now)
            x += k
        r["row_now"].copy_(rn, non_blocking=True)
        logits = ctrl._all_logits_cuda.squeeze(0)
        E = len(ext_rows)
        if E:
            rows = torch.tensor(ext_rows, dtype=torch.int64).pin_memory().to(self.dev, non_blocking=True)
            lg = logits.index_select(0, rows)
            temp = ctx.gpu_view.temperature.index_select(
                0, torch.tensor(ext_temp_rows, dtype=torch.int64).pin_memory().to(self.dev, non_blocking=True))
            s = self.sample_rows(lg, np.asarray(ext_rids, dtype=np.int64), np.asarray(ext_pos, dtype=np.int64), temp)
            # extra rows' samples at their position in the layout's extra block (masked rows: whatever)
            r["s_ext"][: lay.E].zero_()
            r["s_ext"].index_copy_(0, rows - lay.n_rows, s.to(torch.int64))
            sel, _ = ctx.calculate_log_probs_tensors(lg.unsqueeze(0), s.to(torch.int64), only_last_token_logits=True,
                                                     sampling=ctrl._sampling)
            r["lp_ext"].index_copy_(0, rows - lay.n_rows, sel.float())
        T = len(lay.tails)
        _accept_kernel[(T,)](
            self.H, self.P, self.BUD, self.EOS, self.DONE, r["lay"], r["row_now"], ctrl._sampled_tokens_cuda,
            r["s_ext"], r["prop"], r["kk"], r["p"], r["sid"], r["acc"], r["tok"], r["slots"], r["sids"], r["cur"],
            self.H.stride(0), r["lay"].stride(0), r["prop"].stride(0), r["tok"].stride(0), r["sid"].stride(0),
            K=self.K, KPOW=8, CHUNK=self._chunk(), ACCEPT_ALL=self.neg_accept_all,
        )
        if lay.E > 0:
            from nemo_rl.models.megatron import mamba_bi_lookahead as la

            la.commit(ctx.mamba_conv_states_full, r["slots"][:T], r["sids"][:T], r["acc"][:T], r["cur"][:T],
                      self._chunk())
        # results -> host (pinned, one copy): acc, kk, done, tokens, extra log-probs
        out = r["h_out"]
        T8 = T * 8
        pack = torch.cat([r["acc"][:T].double(), r["kk"][:T].double(), self.DONE.index_select(
            0, r["lay"][0][:T].clamp(min=0).long()).double(), r["tok"][:T].flatten().double(),
            r["lp_ext"][: max(lay.E, 1)].double()])
        assert pack.numel() <= out.numel(), (pack.numel(), out.numel())
        out[: pack.numel()].copy_(pack, non_blocking=True)
        r["ev"].record()
        lay.result = (r, T, T8, live)
        self._last_sampled = lay
        if self._debug is not None:  # measurement only (THUNDERSYNC_TAIL_MTD_DEBUG): proposals and every row's draw
            torch.cuda.synchronize()
            x = lay.n_rows
            for j, (t, row, k, _pf) in enumerate(lay.tails):
                if any(jj == j for jj, *_ in live):
                    kk = int(r["kk"][j])
                    self._debug.append(dict(
                        rid=t.rid, p=int(r["p"][j]), k=k, kk=kk, prop=r["prop"][j, :kk].tolist(),
                        s=[int(ctrl._sampled_tokens_cuda[rid_row[t.rid]])] + r["s_ext"][x - lay.n_rows: x - lay.n_rows + k].tolist(),
                        a=int(r["acc"][j])))
                x += k

    # ------------------------------------------------------------------ resolve (host)
    def resolve(self) -> dict:
        """rid -> (tokens [a], extra log-probs [a - 1], done) of the last sampled forward (waits on its copy)."""
        lay = getattr(self, "_last_sampled", None)
        self._last_sampled = None
        if lay is None:
            return {}
        r, T, T8, live = lay.result
        self.gA *= POOL_FORGET
        self.gN *= POOL_FORGET
        r["ev"].synchronize()
        out = r["h_out"].numpy()
        acc = out[:T].astype(np.int64)
        kk = out[T: 2 * T].astype(np.int64)
        done = out[2 * T: 3 * T].astype(np.int64)
        tok = out[3 * T: 3 * T + T8].astype(np.int64).reshape(T, 8)
        lp = out[3 * T + T8:]
        res = {}
        x_of = {}
        x = lay.n_rows
        for j, (t, row, k, _pf) in enumerate(lay.tails):
            x_of[j] = x
            x += k
        for j, t, now, k, _x in live:
            a = int(acc[j])
            if a <= 0:
                continue
            ext = [float(lp[(x_of[j] - lay.n_rows) + i - 1]) for i in range(1, a)]
            req = self.engine.get_request(t.rid)
            p_row0 = t.plen + len(req.generated_tokens)  # row 0's key position (generated not yet appended)
            res[t.rid] = (tok[j, :a].tolist(), ext, bool(done[j]), p_row0)
            # Acceptance per allocated extra row (geometric): of the k rows given, m = a - 1 were accepted and the
            # next one failed (a missing proposal is a failed row: the row was computed for nothing). With no rows
            # given (k = 0) there is no evidence: the estimate relaxes toward the prior, so the tail probes again.
            m, kq = a - 1, int(kk[j])
            t.acc *= FORGET
            t.trials *= FORGET
            if k > 0:
                t.acc += m
                t.trials += m + (1 if m < k else 0)
                self.gA += m
                self.gN += m + (1 if m < k else 0)
            self._acc[t.tid], self._tri[t.tid] = t.acc, t.trials
            t.iters += 1
            t.tokens += a
            self.stats["tokens"] += a
            self.stats["accepted_extra"] += a - 1
            self.stats["proposed"] += kq
            if self.neg_accept_all:
                self.stats["neg_tokens"] += a - 1
        return res

    def _commit_keys(self) -> None:
        """Noise keys are checked at commit: row 0's draw was observed by the sampler at position p; the a - 1
        committed extra positions p + 1 .. p + a - 1 go to the key registry, and the continuity check expects the
        next draw at p + a."""
        res = getattr(self, "_res", None) or {}
        kc, kr = self._key_check, self._keyreg
        for rid, (toks, _lp, _done, p0) in res.items():
            a = len(toks)
            if a <= 1:
                continue
            if kc is not None and rid in kc.prev and kc.prev[rid] >= 0:
                if kc.prev[rid] != p0:
                    raise RuntimeError(f"[tail_mtd] request {rid}: sampler saw position {kc.prev[rid]}, MTD {p0}")
                kc.prev[rid] = p0 + a - 1
            if kr is not None:
                kid = int(self._key_ids(np.asarray([rid], dtype=np.int64))[0]) if self._key_ids else rid
                bad = kr.observe(np.full(a - 1, kid, dtype=np.int64), np.arange(p0 + 1, p0 + a, dtype=np.int64))
                if bad:
                    from thundersync_rl.key_check import SamplerKeyError

                    raise SamplerKeyError(f"[tail_mtd] committed noise key reused: {bad[:3]}")

    def summary(self) -> dict:
        s = dict(self.stats)
        s["tokens_per_tail_iter"] = round(s["tokens"] / max(s["tail_iters"], 1), 4)
        return s


# ---------------------------------------------------------------------------- installation


def install(engine, *, K: int, t_max: int, ref_cap: int, sample_rows, key_check=None, keyreg=None,
            key_ids=None) -> TailMTD:
    """Hook a built DynamicInferenceEngine (idempotent per engine). ``sample_rows(logits, rids, positions, temp)``
    draws rows with the engine's keyed sampler (the worker's), ``key_check`` / ``keyreg`` are the sampler's
    noise-key checks (advanced at commit), ``key_ids`` maps request ids to noise-key ids."""
    from megatron.core.inference.batch_dimensions_utils import (
        CUDAGraphBatchDimensionBuilder,
        InferenceBatchDimensions,
    )

    from nemo_rl.models.megatron import mamba_bi_lookahead as la

    existing = getattr(engine, "_tail_mtd", None)
    if existing is not None:
        return existing
    mtd = TailMTD(engine, K, t_max, ref_cap, la.capacity(), sample_rows)
    mtd._key_check, mtd._keyreg, mtd._key_ids = key_check, keyreg, key_ids
    engine._tail_mtd = mtd
    ctx, ctrl = engine.context, engine.controller

    # graph matching on N + E rows (the extras sit in the decode graph's padding)
    if not getattr(CUDAGraphBatchDimensionBuilder, "_tail_mtd_wrapped", False):
        orig_match = CUDAGraphBatchDimensionBuilder.match_graph_config

        def match_graph_config(real_batch_dim, *a, **kw):
            e = _ACTIVE.get("extra", 0)
            if e > 0 and real_batch_dim.prefill_req_count == 0:
                real_batch_dim = InferenceBatchDimensions(
                    token_count=real_batch_dim.token_count + e, prefill_req_count=0,
                    decode_req_count=real_batch_dim.decode_req_count + e)
            return orig_match(real_batch_dim, *a, **kw)

        CUDAGraphBatchDimensionBuilder.match_graph_config = staticmethod(match_graph_config)
        CUDAGraphBatchDimensionBuilder._tail_mtd_wrapped = True

    orig_init = ctx.initialize_attention_state

    def initialize_attention_state(*a, **kw):
        dummy = bool(kw.get("is_expert_parallel_dummy_cuda_graph_step")) or kw.get("construct_graph_dimensions") is not None
        E = mtd.plan(ctx, dummy)
        _ACTIVE["extra"] = E
        try:
            out = orig_init(*a, **kw)
        finally:
            _ACTIVE["extra"] = 0
        mtd.finish_layout(ctx)
        return out

    ctx.initialize_attention_state = initialize_attention_state

    # MCore never allocates decode blocks for tails (``_ensure_kv`` does, ahead of the device's position)
    orig_next = ctx._next_decode_block_counts

    def _next_decode_block_counts(request_slice):
        c = orig_next(request_slice)
        tr = mtd.tracked_rids()
        if tr.size:
            mask = np.isin(ctx.request_ids[request_slice].numpy(), tr)
            if mask.any():
                c = c.clone()
                c[torch.from_numpy(mask)] = 0
        return c

    ctx._next_decode_block_counts = _next_decode_block_counts

    orig_fwd = ctrl._dynamic_step_forward_logits

    def _dynamic_step_forward_logits(*a, **kw):
        mtd.expand(ctx)
        out = orig_fwd(*a, **kw)
        mtd.after_forward(ctx)
        return out

    ctrl._dynamic_step_forward_logits = _dynamic_step_forward_logits

    orig_sample = ctrl._dynamic_step_sample_logits

    def _dynamic_step_sample_logits(*a, **kw):
        out = orig_sample(*a, **kw)
        mtd.after_sample(ctrl)
        return out

    ctrl._dynamic_step_sample_logits = _dynamic_step_sample_logits

    # discarded pending logits (lifecycle reset): drop their layouts
    logits_state = ctrl._async_sched_logits
    orig_clear = logits_state.clear

    def clear():
        mtd.pending.clear()
        return orig_clear()

    logits_state.clear = clear

    # resolve: finish from the device result; tokens and log-probs of every committed token
    orig_state = ctrl._build_async_sched_request_state

    def _build_async_sched_request_state(sampled_tokens_cpu, resolved_sequence_lengths):
        mtd._res = mtd.resolve()
        mtd._res_idx = {}
        ids, fin, mask = orig_state(sampled_tokens_cpu, resolved_sequence_lengths)
        if mtd._res:
            lo = ctx.paused_request_count
            n = ctx.total_request_count - lo
            cur = ctx.request_ids[lo: lo + n].numpy()
            hit = np.nonzero(np.isin(cur, np.fromiter(mtd._res.keys(), dtype=np.int64)))[0]
            mtd._res_idx = {int(i): int(cur[i]) for i in hit}  # active row -> request id (sampling order)
            changed = False
            for i, rid in mtd._res_idx.items():
                rr = mtd._res[rid]
                if rr[2] and int(mask[i]) != 0:
                    mask[i] = 0
                    changed = True
            if changed:
                idx = torch.nonzero(mask == 0, as_tuple=True)[0] + lo
                fin = ctx.request_ids[idx].clone()
            mtd._commit_keys()
        return ids, fin, mask

    ctrl._build_async_sched_request_state = _build_async_sched_request_state

    orig_mat = ctrl._materialize_async_sched_log_probs

    def _materialize_async_sched_log_probs(transfer, accepted_counts_cpu=None):
        lps, top = orig_mat(transfer, accepted_counts_cpu)
        res = getattr(mtd, "_res", None)
        if res and lps is not None:
            # rows are the active requests at sampling time (before compaction): the resolve's order
            for i, rid in getattr(mtd, "_res_idx", {}).items():
                rr = res[rid]
                if len(rr[0]) > 1 and i < len(lps):
                    lps[i] = list(lps[i]) + list(rr[1])
        return lps, top

    ctrl._materialize_async_sched_log_probs = _materialize_async_sched_log_probs

    orig_post = engine.post_process_requests

    def post_process_requests(request_ids, finished_request_ids, evict_request_ids, step_time, sample,
                              accepted_tokens, log_probs, *a, **kw):
        res = getattr(mtd, "_res", None)
        if res:
            toks = sample.tolist()
            new = []
            for rid, s in zip(request_ids.tolist(), toks):
                rr = res.get(int(rid))
                if rr is not None and len(rr[0]) > 1:
                    assert int(rr[0][0]) == int(s), ("[tail_mtd] row 0 sample mismatch", rid, rr[0][0], s)
                    new.append(list(rr[0]))
                else:
                    new.append(s)
            sample = _ListSample(new)
        out = orig_post(request_ids, finished_request_ids, evict_request_ids, step_time, sample, accepted_tokens,
                        log_probs, *a, **kw)
        for rid in finished_request_ids.tolist():
            mtd.release(int(rid))
            mtd._seen.discard(int(rid))
        mtd._res = None
        return out

    engine.post_process_requests = post_process_requests
    return mtd


_ACTIVE: dict = {}


class _ListSample:
    """``sample`` for post_process_requests with several tokens for some requests (tolist() -> ints / lists)."""

    def __init__(self, rows):
        self.rows = rows

    def tolist(self):
        return self.rows
