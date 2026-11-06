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
"""Exact tree speculation on opted-in engine requests (shared engine module; interface:
the research notes, summarized in docs/colo/). It generalizes tail_mtd's single proposal chain to a token tree.

Per engine iteration an opted-in request ("tail") gets up to W extra decode rows, one per tree node. The nodes are
the trie of the candidate continuations that follow the suffix matches (n = 4..1) of its context in its references:
the dead suffix, then the siblings. Candidates go best first: longer match, then reference order.

**Every node is a one-token decode row** at position P - 1 + depth of the same forward. It sees exactly the prefix
one-token decode sees once its ancestors are committed:
- Mamba: its ancestors' entries in ``look_chain``, their scratch replay rows in ``look_srow``, and its scratch conv
  window (mamba_bi_fused_decode / mamba_bi_lookahead);
- attention: a private KV block with the committed part of the last block, the root's and the ancestors' K / V
  gathered in every layer (tree_kv).
So a depth-d path is verified in ONE iteration, and its logits are bitwise one-token decode's.

**Sampling and acceptance.** Every node row is sampled by the engine's keyed sampler with the key one-token decode
uses for its position: (key id of the request, P + depth). Starting at the root, the walk follows the child whose
token equals the parent's draw: a = 1 + the depth reached, cut at EOS / budget. The emitted tokens are the draws
along the path, i.e. exactly one-token decode's (same keys, bitwise logits). Rejected nodes are never committed:
no SSM state is touched (depth <= chunk room). The accepted path's replay rows, conv state, cursor and K / V are
copied into the request's slot and block (``mamba_bi_lookahead.commit_tree``, ``tree_kv.commit_copy``).

**Device-resolved** (as tail_mtd): the attach, candidate, trie, layout, accept and commit kernels run on the forward's
stream with fixed grids and no host sync. The host lays out widths one iteration late and reconciles the accepted
tokens at the resolve.

**Width per tail** (host, past results only). r_i is the pooled, decayed rate at which the node inserted i-th lies on
the accepted path, so a W-node tree emits 1 + sum_{i<W} r_i tokens per iteration in expectation. Nodes are added
one at a time to the tail with the longest expected remaining time, R_t / s_t(W_t) (the critical path first). This
stops when that tail's marginal node gains less than BETA, the cost of one decode row relative to the iteration, or
when the per-rank node capacity is reached.

Per rank:
- device tail tables: T_MAX x (max_seq + ref_cap) int32;
- <= node_cap extra decode rows per iteration;
- node_cap scratch conv + replay rows per Mamba layer;
- node_cap private KV blocks.
No communication: candidates and trees are computed identically on every TP rank.
"""

from __future__ import annotations

import heapq
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

from nemo_rl.models.generation.megatron.tail_mtd import MTD_REFS_ATTR, SEP, _ListSample, pack_refs  # noqa: F401

# Optional per-request attr: int32 bytes [W, D, src_flags] (W = 0: plain decode). Without it the engine's W / D apply.
TREE_ATTR = "ts_tree"
NMAX = 4  # longest suffix matched
WMAX = 64  # nodes per tail per iteration (kernel width)
CMAX = 32  # candidate continuations per tail per iteration
LOOK_MAX = 8  # = mamba_bi_fused_decode.LOOK_MAX: depth <= 7
BLOCK = 1024  # reference positions per candidate program
KMTP = 4  # MTP draft depth (candidate 0 of the trie)
# Cost of one decode row relative to an iteration: Nano TP1 EP8 BI engine 0.03-0.045 ms per row at 26-78 rows/rank
# (bi_curve: 6.16 / 6.85 / 7.34 / 8.40 / 9.02 ms at 8 / 32 / 52 / 64 / 78 rows) against the in-loop ~10 ms
# iteration; FP_B M2: 0.6-0.9 ms per extra token at 26 rows/rank (0.03 ms / row).
BETA = 0.004
PRIOR0, FORGET = 0.5, 0.9995  # first node's prior acceptance; pooled counts forgotten over ~2000 tail iterations
NEGATIVES = ("tree_accept_all", "tree_accept_topk", "tree_accept_by_draft_prob", "tree_stale_mamba", "tree_key_off_by_one", "tree_key_reuse",
             "tree_fresh_keys",
             "tree_skip_kv_rollback", "tree_non_bi_verify", "tree_stale_cache")


if HAVE_TRITON:

    @triton.jit
    def _attach_kernel(H, P, lay, in_ids, stride_h, stride_lay):
        """Attach entries (flag 2): this forward's row-0 input is the token the last forward committed: H[P] <- it."""
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
    def _match_level(H, R, RLEN, P, OWN, tid64, b, stride_h, stride_r, BLOCKC: tl.constexpr, NMAXC: tl.constexpr):
        """Suffix-match length n(e) (0..NMAXC) of the tail's context at reference positions e of block b, for the
        positions whose following token R[e] is a token (not a separator / the end). An own-history tail (OWN[tid],
        src bit1: XS-SD) matches against its own history H[0 .. P): its live suffix never matches itself (e < P)."""
        p = tl.load(P + tid64)
        hrow = H + tid64 * stride_h
        rrow = R + tid64 * stride_r
        e = b * BLOCKC + tl.arange(0, BLOCKC)
        rl = tl.load(RLEN + tid64)
        if tl.load(OWN + tid64) != 0:
            rrow = hrow
            rl = p
        valid = (e >= 1) & (e < rl)
        n = tl.zeros([BLOCKC], dtype=tl.int32)
        run = valid
        for d in tl.static_range(NMAXC):
            kp = p - 1 - d
            key = tl.load(hrow + kp, mask=kp >= 0, other=-7)
            pos = e - 1 - d
            v = tl.load(rrow + pos, mask=run & (pos >= 0), other=-5)
            m = run & (pos >= 0) & (v == key) & (key >= 0)
            n += m.to(tl.int32)
            run = m
        fv = tl.load(rrow + e, mask=valid, other=-1)
        return tl.where(valid & (fv >= 0), n, 0), e

    @triton.jit
    def _count_kernel(H, R, RLEN, P, OWN, lay, bcnt, stride_h, stride_r, stride_lay, NB,
                      BLOCKC: tl.constexpr, NMAXC: tl.constexpr):
        """Program (tail j, block b): matches per level n = 1..NMAXC in the block -> bcnt[j, b, n - 1]."""
        j = tl.program_id(0)
        b = tl.program_id(1)
        tid = tl.load(lay + 0 * stride_lay + j)
        w = tl.load(lay + 2 * stride_lay + j)
        pf = tl.load(lay + 6 * stride_lay + j)
        base = bcnt + (j * NB + b) * NMAXC
        if (tid < 0) | (w <= 0) | (pf == 1):
            for lv in tl.static_range(NMAXC):
                tl.store(base + lv, 0)
            return
        n, _e = _match_level(H, R, RLEN, P, OWN, tid.to(tl.int64), b, stride_h, stride_r, BLOCKC, NMAXC)
        for lv in tl.static_range(NMAXC):
            tl.store(base + lv, tl.sum((n == lv + 1).to(tl.int32), axis=0))

    @triton.jit
    def _cand_kernel(H, R, RLEN, P, OWN, lay, bcnt, cand, ncand, stride_h, stride_r, stride_lay, NB,
                     BLOCKC: tl.constexpr, NMAXC: tl.constexpr, NBPOW: tl.constexpr, CMAXC: tl.constexpr):
        """Program (tail j, block b): candidate rank of each match = (level desc, position asc); the first CMAXC
        -> cand[j, rank] (reference positions; deterministic, identical on every TP rank)."""
        j = tl.program_id(0)
        b = tl.program_id(1)
        tid = tl.load(lay + 0 * stride_lay + j)
        w = tl.load(lay + 2 * stride_lay + j)
        pf = tl.load(lay + 6 * stride_lay + j)
        if (tid < 0) | (w <= 0) | (pf == 1):
            if b == 0:
                tl.store(ncand + j, 0)
            return
        bi = tl.arange(0, NBPOW)
        n, e = _match_level(H, R, RLEN, P, OWN, tid.to(tl.int64), b, stride_h, stride_r, BLOCKC, NMAXC)
        higher = 0
        total = 0
        for lvi in tl.static_range(NMAXC):
            lv = NMAXC - lvi  # NMAXC .. 1
            cnts = tl.load(bcnt + (j * NB + bi) * NMAXC + lv - 1, mask=bi < NB, other=0)
            before = tl.sum(tl.where(bi < b, cnts, 0), axis=0)
            m = n == lv
            rank = higher + before + tl.cumsum(m.to(tl.int32), axis=0) - 1
            tl.store(cand + j * CMAXC + rank, e.to(tl.int32), mask=m & (rank < CMAXC))
            higher += tl.sum(cnts, axis=0)
            total += tl.sum(cnts, axis=0)
        if b == 0:
            tl.store(ncand + j, tl.minimum(total, CMAXC))

    @triton.jit
    def _build_kernel(H, R, RLEN, P, OWN, BUD, lay, cand, ncand, ntok, npar, ndep, nn, p_out, mtok, mlen, msib, nm_out,
                      stride_h, stride_r, stride_lay, stride_m, D, CHUNK: tl.constexpr, BS: tl.constexpr,
                      WPOW: tl.constexpr, CMAXC: tl.constexpr, LOOK: tl.constexpr, ALWAYS: tl.constexpr,
                      KM: tl.constexpr):
        """Program j: the tail's trie of candidate continuations (best first) up to W nodes, depth <= min(D, chunk
        room, block room, budget). Node i: token ntok, parent npar (-1 = the root), depth ndep, in insertion order."""
        j = tl.program_id(0)
        tid = tl.load(lay + 0 * stride_lay + j)
        idx = tl.arange(0, WPOW)
        if tid < 0:
            tl.store(nn + j, 0)
            return
        tid64 = tid.to(tl.int64)
        p = tl.load(P + tid64)
        tl.store(p_out + j, p)
        w = tl.load(lay + 2 * stride_lay + j)
        pf = tl.load(lay + 6 * stride_lay + j)
        bud = tl.load(BUD + tid64)
        w0 = (p - 1) % CHUNK
        dmax = tl.minimum(D, CHUNK - 2 - w0)
        dmax = tl.minimum(dmax, BS - 1 - (p - 1) % BS)
        dmax = tl.minimum(dmax, bud - 1 - p)
        dmax = tl.minimum(dmax, LOOK - 1)
        nc = tl.load(ncand + j)
        if (w <= 0) | (pf == 1) | (dmax <= 0):
            nc = 0
        rl = tl.load(RLEN + tid64)
        rrow = R + tid64 * stride_r
        if tl.load(OWN + tid64) != 0:  # own-history tail: continuations read from H[0 .. P)
            rrow = H + tid64 * stride_h
            rl = p
        tok_v = tl.full([WPOW], -1, dtype=tl.int32)
        par_v = tl.full([WPOW], -2, dtype=tl.int32)
        dep_v = tl.zeros([WPOW], dtype=tl.int32)
        n = 0
        # candidate 0: the MTP head's drafted chain for this tail (tree_mtp; length 0 without a head)
        ml = tl.load(mlen + tid64)
        if (w > 0) & (pf != 1) & (dmax > 0) & (ml > 0):
            cur = -1
            for d in tl.static_range(1, KM + 1):
                x = tl.load(mtok + tid64 * stride_m + d - 1)
                ok = (d <= ml) & (d <= dmax) & (x >= 0) & (n < w)
                tok_v = tl.where(ok & (idx == n), x, tok_v)
                par_v = tl.where(ok & (idx == n), cur, par_v)
                dep_v = tl.where(ok & (idx == n), d, dep_v)
                cur = tl.where(ok, n, cur)
                n = tl.where(ok, n + 1, n)
        # the MTP depth-1 sibling (the depth-1 race's runner-up), right after the chain: a second child of the root
        sb = tl.load(msib + tid64)
        if (w > 0) & (pf != 1) & (dmax > 0) & (ml > 0) & (sb >= 0) & (n < w):
            first = tl.sum(tl.where((idx == 0) & (n > 0), tok_v, 0), axis=0)
            if (n == 0) | (first != sb):
                tok_v = tl.where(idx == n, sb, tok_v)
                par_v = tl.where(idx == n, -1, par_v)
                dep_v = tl.where(idx == n, 1, dep_v)
                n = n + 1
        tl.store(nm_out + j, n)  # MTP nodes = insertion indices 0 .. n - 1 (per-source acceptance counters)
        for c in range(0, nc):
            e = tl.load(cand + j * CMAXC + c)
            cur = -1
            alive = 1
            for d in tl.static_range(1, LOOK):
                pos = e + d - 1
                x = tl.load(rrow + pos, mask=pos < rl, other=-1)
                ok = (alive != 0) & (d <= dmax) & (x >= 0)
                hit = (par_v == cur) & (tok_v == x) & (idx < n)
                hidx = tl.max(tl.where(hit, idx, -1), axis=0)
                new = ok & (hidx < 0) & (n < w)
                tok_v = tl.where(new & (idx == n), x, tok_v)
                par_v = tl.where(new & (idx == n), cur, par_v)
                dep_v = tl.where(new & (idx == n), d, dep_v)
                nxt = tl.where(hidx >= 0, hidx, n)
                cur = tl.where(ok & ((hidx >= 0) | new), nxt, cur)
                alive = tl.where(ok & ((hidx >= 0) | new), 1, 0)
                n = tl.where(new, n + 1, n)
        if ALWAYS:
            # NEGATIVE CONTROL tree_accept_all only: without a candidate, a chain of the last token repeated
            if (n == 0) & (w > 0) & (pf != 1) & (dmax > 0):
                last = tl.load(H + tid64 * stride_h + p - 1)
                m = idx < tl.minimum(dmax, w)
                tok_v = tl.where(m, last, tok_v)
                par_v = tl.where(m, idx - 1, par_v)
                dep_v = tl.where(m, idx + 1, dep_v)
                n = tl.minimum(dmax, w)
        tl.store(ntok + j * WPOW + idx, tok_v)
        tl.store(npar + j * WPOW + idx, par_v)
        tl.store(ndep + j * WPOW + idx, dep_v)
        tl.store(nn + j, n)

    @triton.jit
    def _expand_kernel(
        H, lay, ntok, npar, ndep, nn, p_in, priv, rblk_out,
        in_ids, pos_ids, blk_idx, loc_pos, pos_in_req, req_idx, q_len, kv_len, btab, mamba_bi, logit_idx,
        look_off, look_chain, look_srow, scratch_id, kpos, gath, part,
        stride_h, stride_lay, stride_bt, stride_chain, stride_srow,
        n_kv_cols, dummy_block, scratch_base, key_shift,
        BS: tl.constexpr, WPOW: tl.constexpr, BTCOLS: tl.constexpr, LOOK: tl.constexpr,
    ):
        """Program (tail j, i): i = 0 the root row's device overrides; i >= 1 node i - 1's decode row (entry x0 + i - 1).
        Masked rows (allocated, no node): one query token on the dummy block, no Mamba slot."""
        j = tl.program_id(0)
        i = tl.program_id(1)
        tid = tl.load(lay + 0 * stride_lay + j)
        if tid < 0:
            return
        r0 = tl.load(lay + 1 * stride_lay + j)
        w = tl.load(lay + 2 * stride_lay + j)
        x0 = tl.load(lay + 3 * stride_lay + j)
        slot = tl.load(lay + 4 * stride_lay + j)
        pf = tl.load(lay + 6 * stride_lay + j)
        if pf == 1:
            return  # the tail's prefill forward: MCore's layout is right
        p = tl.load(p_in + j)
        r0_64 = r0.to(tl.int64)
        pos0 = p - 1
        c0 = pos0 // BS
        rblk = tl.load(btab + r0_64 * stride_bt + c0)
        if i == 0:
            tl.store(rblk_out + j, rblk)
            last = tl.load(H + tid.to(tl.int64) * stride_h + p - 1)
            tl.store(in_ids + r0_64, last.to(tl.int64))
            tl.store(pos_ids + r0_64, pos0.to(tl.int64))
            tl.store(blk_idx + r0_64, rblk.to(tl.int64))
            tl.store(loc_pos + r0_64, pos0 % BS)
            tl.store(pos_in_req + r0_64, pos0)
            tl.store(kv_len + r0_64, p)
            return
        k = i - 1
        if k >= w:
            return
        ex = (x0 + k).to(tl.int64)
        n_req_rows = tl.load(lay + 5 * stride_lay)  # the request rows of this forward (device: graph-replayable)
        sidx = ex - n_req_rows  # node row index of this forward: scratch rows, private block, key slot
        n = tl.load(nn + j)
        live = k < n
        cols = tl.arange(0, BTCOLS)
        lc = tl.arange(0, LOOK)
        tl.store(q_len + ex, 1)
        tl.store(req_idx + ex, r0)
        tl.store(logit_idx + ex, ex.to(tl.int32))
        if live:
            d = tl.load(ndep + j * WPOW + k)
            tok = tl.load(ntok + j * WPOW + k)
            pos = p - 1 + d
            pb = tl.load(priv + sidx)
            tl.store(in_ids + ex, tok.to(tl.int64))
            tl.store(pos_ids + ex, pos.to(tl.int64))
            tl.store(blk_idx + ex, pb.to(tl.int64))
            tl.store(loc_pos + ex, pos % BS)
            tl.store(pos_in_req + ex, pos)
            tl.store(kv_len + ex, p + d)
            bt_row0 = tl.load(btab + r0_64 * stride_bt + cols, mask=cols < n_kv_cols, other=0)
            tl.store(btab + ex * stride_bt + cols, tl.where(cols == c0, pb, bt_row0), mask=cols < n_kv_cols)
            tl.store(mamba_bi + ex, slot)
            tl.store(look_off + ex, d)
            tl.store(scratch_id + ex, (scratch_base + sidx).to(tl.int32))
            tl.store(kpos + sidx, (p + d + key_shift).to(tl.int64))
            # ancestors: walk the parents (depth d-1 .. 1); chain[0] = the root row
            chain = tl.where(lc == 0, r0, -1)
            srow = tl.where(lc == d - 1, sidx.to(tl.int32), -1)
            g_src = tl.where(lc == 0, rblk, -1)  # gather: the root's K / V at p - 1 from the request's block
            g_off = tl.where(lc == 0, pos0 % BS, 0)
            cur = tl.load(npar + j * WPOW + k)
            for s in tl.static_range(LOOK - 1):
                dd = d - 1 - s  # depth of `cur`
                ok = (cur >= 0) & (dd >= 1)
                ce = x0 + cur
                chain = tl.where(ok & (lc == dd), ce, chain)
                srow = tl.where(ok & (lc == dd - 1), (ce - n_req_rows).to(tl.int32), srow)
                apb = tl.load(priv + ce - n_req_rows, mask=ok, other=-1)
                g_src = tl.where(ok & (lc == dd), apb, g_src)
                g_off = tl.where(ok & (lc == dd), (pos0 + dd) % BS, g_off)
                cur = tl.where(ok, tl.load(npar + j * WPOW + tl.maximum(cur, 0), mask=ok, other=-1), cur)
            tl.store(look_chain + ex * stride_chain + lc, chain)
            tl.store(look_srow + ex * stride_srow + lc, srow)
            gb = gath + sidx * LOOK * 3 + lc * 3
            tl.store(gb + 0, g_src)
            tl.store(gb + 1, tl.where(g_src >= 0, pb, -1))
            tl.store(gb + 2, g_off)
            tl.store(part + sidx * 3 + 0, rblk)
            tl.store(part + sidx * 3 + 1, pb)
            tl.store(part + sidx * 3 + 2, pos0 % BS)
        else:
            tl.store(in_ids + ex, 0)
            tl.store(pos_ids + ex, 0)
            tl.store(blk_idx + ex, dummy_block)
            tl.store(loc_pos + ex, 0)
            tl.store(pos_in_req + ex, 0)
            tl.store(kv_len + ex, 1)
            tl.store(btab + ex * stride_bt + cols, tl.where(cols < n_kv_cols, dummy_block, 0), mask=cols < n_kv_cols)
            tl.store(mamba_bi + ex, -1)
            tl.store(look_off + ex, 0)
            tl.store(scratch_id + ex, -1)
            tl.store(kpos + sidx, 0)
            gb = gath + sidx * LOOK * 3 + lc * 3
            tl.store(gb + 0, tl.full([LOOK], -1, tl.int32))
            tl.store(part + sidx * 3 + 0, -1)

    @triton.jit
    def _accept_kernel(
        H, P, BUD, EOS, DONE, lay, row_now, s0, s_ext, alt_root, alt_ext, alt2_root, alt2_ext, pacc_root, pacc_ext,
        res_root, res_ext, ntok, npar, nn, p_in, priv,
        acc_out, tok_out, slots_out, cur_out, path_idx, path_blk, rblk, done_out,
        stride_h, stride_lay, stride_tok, stride_path,
        CHUNK: tl.constexpr, BS: tl.constexpr, WPOW: tl.constexpr, LOOK: tl.constexpr, KPOW: tl.constexpr,
        TOPK: tl.constexpr, ACCEPT_ALL: tl.constexpr = False, DRAFTPROB: tl.constexpr = False,
    ):
        """Program j: walk tail j's tree along the keyed draws (the child whose token == its parent's draw), emit the
        draws of the path, cut at EOS / budget; H / P / done; commit inputs (path scratch indices, private blocks)."""
        j = tl.program_id(0)
        tid = tl.load(lay + 0 * stride_lay + j)
        rn = tl.load(row_now + j)
        lc = tl.arange(0, LOOK)
        offs = tl.arange(0, KPOW)
        tl.store(path_idx + j * stride_path + lc, tl.full([LOOK], -1, tl.int32))
        tl.store(path_blk + j * stride_path + lc, tl.full([LOOK], -1, tl.int32))
        tl.store(done_out + j, 1)
        if (tid < 0) | (rn < 0):
            tl.store(acc_out + j, 0)
            tl.store(slots_out + j, -1)
            return
        tid64 = tid.to(tl.int64)
        if tl.load(DONE + tid64) != 0:
            tl.store(acc_out + j, 0)
            tl.store(slots_out + j, -1)
            return
        x0 = tl.load(lay + 3 * stride_lay + j)
        slot = tl.load(lay + 4 * stride_lay + j)
        n_rows = tl.load(lay + 5 * stride_lay)  # the request rows of this forward
        p = tl.load(p_in + j)
        n = tl.load(nn + j)
        eos = tl.load(EOS + tid64)
        bud = tl.load(BUD + tid64)
        idx = tl.arange(0, WPOW)
        tok_v = tl.load(ntok + j * WPOW + idx)
        par_v = tl.load(npar + j * WPOW + idx)
        s_cur = tl.load(s0 + rn).to(tl.int64)
        a_cur = tl.load(alt_root + rn).to(tl.int64)  # negative tree_accept_topk only: the row's runner-up tokens
        a2_cur = tl.load(alt2_root + rn).to(tl.int64)
        q_cur = tl.load(pacc_root + rn)  # negative tree_accept_by_draft_prob only: accept / residual of the row
        r_cur = tl.load(res_root + rn).to(tl.int64)
        stop = 0
        emitted = tl.where(offs == 0, s_cur, -1)
        cur = -1
        a = 1
        alive = 1
        for d in tl.static_range(1, LOOK):
            hit = (par_v == cur) & (tok_v.to(tl.int64) == s_cur) & (idx < n)
            if TOPK:
                tk64 = tok_v.to(tl.int64)
                hit = (par_v == cur) & ((tk64 == s_cur) | (tk64 == a_cur) | (tk64 == a2_cur)) & (idx < n)
            if ACCEPT_ALL:  # NEGATIVE CONTROL: any child accepted, whatever the draws
                hit = (par_v == cur) & (idx < n)
            hidx = tl.max(tl.where(hit, idx, -1), axis=0)
            if DRAFTPROB:
                # NEGATIVE CONTROL (exact in law, not keyed-pathwise): the first child c of cur is accepted with the
                # keyed probability q = p(c) at cur's row, else the residual race (c removed) is emitted and the walk stops
                first = tl.min(tl.where((par_v == cur) & (idx < n), idx, WPOW + 1), axis=0)
                hidx = tl.where((first <= WPOW) & (q_cur > 0.0), first, -1)
                stop = tl.where((alive != 0) & (first <= WPOW) & (q_cur <= 0.0), 1, stop)
                emitted = tl.where((alive != 0) & (first <= WPOW) & (q_cur <= 0.0) & (offs == d - 1), r_cur, emitted)
            go = (alive != 0) & (hidx >= 0)
            node = tl.maximum(hidx, 0)
            e_rel = x0 - n_rows + node
            ntk = tl.sum(tl.where(idx == node, tok_v, 0), axis=0).to(tl.int64)
            s_nxt = tl.load(s_ext + e_rel, mask=go, other=-9).to(tl.int64)
            a_nxt = tl.load(alt_ext + e_rel, mask=go, other=-9).to(tl.int64)
            a2_nxt = tl.load(alt2_ext + e_rel, mask=go, other=-9).to(tl.int64)
            q_nxt = tl.load(pacc_ext + e_rel, mask=go, other=0.0)
            r_nxt = tl.load(res_ext + e_rel, mask=go, other=-9).to(tl.int64)
            # the path's token at depth d is the node's (== its parent's draw unless the negative accepted a
            # runner-up); the draw of the node is the next token
            emitted = tl.where(go & (offs == d - 1), ntk, emitted)
            emitted = tl.where(go & (offs == d), s_nxt, emitted)
            tl.store(path_idx + j * stride_path + d - 1, e_rel.to(tl.int32), mask=go)
            tl.store(path_blk + j * stride_path + d - 1, tl.load(priv + e_rel, mask=go, other=-1), mask=go)
            s_cur = tl.where(go, s_nxt, s_cur)
            a_cur = tl.where(go, a_nxt, a_cur)
            a2_cur = tl.where(go, a2_nxt, a2_cur)
            q_cur = tl.where(go, q_nxt, q_cur)
            r_cur = tl.where(go, r_nxt, r_cur)
            cur = tl.where(go, node, cur)
            a = tl.where(go, a + 1, a)
            alive = tl.where(go & (stop == 0), 1, 0)
        is_eos = (emitted == eos) & (offs < a)
        first_eos = tl.min(tl.where(is_eos, offs, KPOW + 1), axis=0)
        a = tl.where(first_eos < a, first_eos + 1, a)
        a = tl.minimum(a, bud - p)
        a = tl.maximum(a, 1)
        hrow = H + tid64 * stride_h
        tl.store(hrow + p + offs, emitted.to(tl.int32), mask=offs < a)
        tl.store(P + tid64, p + a)
        last = tl.sum(tl.where(offs == a - 1, emitted, 0), axis=0)
        fin = (last == eos) | (p + a >= bud)
        tl.store(DONE + tid64, fin.to(tl.int32))
        tl.store(done_out + j, fin.to(tl.int64))
        tl.store(acc_out + j, a)
        tl.store(tok_out + j * stride_tok + offs, emitted, mask=offs < a)
        tl.store(slots_out + j, tl.where(n > 0, slot, -1))
        tl.store(cur_out + j, (p - 1) % CHUNK)


if HAVE_TRITON:

    @triton.jit
    def _rtc_kernel(lay, rtc, stride_lay, TP: tl.constexpr):
        """real_token_count <- request rows + node rows of this forward (when it has nodes): device-resolved, so the
        expand stage replays as one CUDA graph."""
        j = tl.arange(0, TP)
        tid = tl.load(lay + 0 * stride_lay + j)
        w = tl.load(lay + 2 * stride_lay + j)
        e = tl.sum(tl.where(tid >= 0, tl.maximum(w, 0), 0), axis=0)
        if e > 0:
            tl.store(rtc, tl.load(lay + 5 * stride_lay) + e)

    @triton.jit
    def _post_fwd_kernel(off, gath, part, lay, stride_lay, n_off, LOOK: tl.constexpr, GP: tl.constexpr):
        """Program e (node row e of the capacity): its Mamba offset back to 0, its KV gather / partial-copy entries
        cleared (every other forward sees offset 0 and an empty table)."""
        e = tl.program_id(0)
        n_rows = tl.load(lay + 5 * stride_lay)
        if (n_rows >= 0) & (n_rows + e < n_off):
            tl.store(off + n_rows + e, 0)
        lc = tl.arange(0, GP)
        tl.store(gath + e * LOOK * 3 + lc, tl.full([GP], -1, tl.int32), mask=lc < LOOK * 3)
        tl.store(part + e * 3 + tl.arange(0, 4), tl.full([4], -1, tl.int32), mask=tl.arange(0, 4) < 3)


@dataclass
class _Tail:
    rid: int
    tid: int
    plen: int
    budget_end: int
    eos: int
    refs: Any
    w_cap: int  # per-request W (ts_tree) or the engine's
    d_cap: int
    own: bool = False  # src bit1: candidates from the request's own history (XS-SD)
    kid: int = -1  # the sampler's key id of the request (cached at registration)
    mslot: int = -1  # its tree_mtp KV slot (-1: no MTP drafts)
    mstart: int = 0  # first position with head K / V (prefix-cache skipped tokens have no decoder output)
    prev_acc: int = 0  # tokens of its last resolved iteration (diagnostics)
    tracked: bool = False
    refs_loaded: bool = False
    w_last: int = 0
    iters: int = 0
    tokens: int = 0
    nodes: int = 0
    accepted_nodes: int = 0


@dataclass
class _Forward:
    slot: int
    tails: list  # [(_Tail, row0 entry, W, flag)]
    n_rows: int
    E: int
    staging: Any = None
    result: Any = None


def tree_tokens(widths_rates: np.ndarray, W: int) -> float:
    """Expected tokens per iteration of a W-node best-first tree: 1 + sum_{i<W} r_i."""
    return 1.0 + float(np.sum(widths_rates[:W]))


def plan_widths(rem: list, rates: np.ndarray, caps: list, budget: int, beta: float = BETA, grid: int = 96) -> list:
    """Nodes per tail minimizing the rank's phase time T (1 + beta N): the phase ends with the slowest tail, so a target
    T needs W_i(T) = the fewest nodes with R_i / s(W_i) <= T on every tail i (s(W) = 1 + sum_{i<W} r_i), and N(T) =
    sum_i W_i(T) node rows cost beta each on every iteration. T is searched on a geometric grid between the fastest
    reachable max_i R_i / s(cap_i) and the no-node max_i R_i. Ties (several tails near the max) are priced right: a
    lower T needs nodes on all of them. Pure; the host side of the width rule (past acceptance only)."""
    rem = np.asarray(rem, dtype=np.float64)
    if rem.size == 0 or float(rem.max()) <= 0:
        return [0] * int(rem.size)
    caps = np.minimum(np.asarray(caps, dtype=np.int64), len(rates))
    cum = np.concatenate([[1.0], 1.0 + np.cumsum(rates)])
    t_hi = float(rem.max())
    t_lo = float(np.max(rem / cum[caps]))
    if t_lo >= t_hi:
        return [0] * int(rem.size)
    ts_ = t_hi * (t_lo / t_hi) ** (np.arange(grid + 1) / grid)  # t_hi .. t_lo
    need = rem[None, :] / ts_[:, None]  # s each tail needs at each T
    w = np.searchsorted(cum, need * (1 - 1e-12), side="left")  # fewest nodes with s(W) >= need
    w = np.minimum(w, caps[None, :])
    n = w.sum(axis=1)
    j = ts_ * (1.0 + beta * n)
    j = np.where(n <= budget, j, np.inf)
    best = int(np.argmin(j))
    return [int(x) for x in w[best]]


def mtp_depth(rates: np.ndarray, price: float, kmax: int) -> int:
    """MTP draft depth: the longest prefix of depths whose marginal node saves more than it costs, r_k / s(k + 1) >
    price (price = one depth's time / the iteration period; s(k) = 1 + sum_{i<k} r_i). Pure."""
    cum = np.concatenate([[1.0], 1.0 + np.cumsum(rates)])
    k = 0
    while k < min(kmax, len(rates)) and rates[k] / cum[k + 1] > price:
        k += 1
    return k


def water_fill(rem: list, rates: np.ndarray, caps: list, budget: int, beta: float = BETA) -> list:
    """Nodes per tail: repeatedly one node to the tail with the longest expected remaining time R / s(W) (critical
    path first), while that tail's marginal node is worth its cost: r_W / s(W + 1) > beta (relative time saved >
    relative iteration cost). Pure; the host side of the width rule."""
    cum = np.concatenate([[1.0], 1.0 + np.cumsum(rates)])
    w = [0] * len(rem)
    heap = [(-r / cum[0], i) for i, r in enumerate(rem) if caps[i] > 0 and r > 0]
    heapq.heapify(heap)
    used = 0
    while heap and used < budget:
        _negt, i = heapq.heappop(heap)
        k = w[i]
        if k >= caps[i] or k >= len(rates):
            break  # the critical tail is at its width cap: nodes elsewhere do not shorten the critical path
        gain = rates[k] / cum[k + 1]
        if gain <= beta:
            break  # the critical tail gains less than a node costs: no other node is worth it either
        w[i] = k + 1
        used += 1
        heapq.heappush(heap, (-rem[i] / cum[k + 1], i))
    return w


class TreeSpec:
    """Engine-side runtime (one per engine). ``install(engine, ...)`` after the engine is built."""

    RING = 4

    def __init__(self, engine, W: int, D: int, t_max: int, ref_cap: int, node_cap: int, sample_rows, scores_rows=None,
                 uniform_rows=None, own_history: bool = False, top2_rows=None):
        from nemo_rl.models.generation.megatron import tree_kv
        from nemo_rl.models.megatron import mamba_bi_lookahead as la

        ctx = engine.context
        self.engine, self.ctx, self.ctrl = engine, ctx, engine.controller
        self.W, self.D = int(W), int(D)
        assert 1 <= self.W <= WMAX and 1 <= self.D <= LOOK_MAX - 1, (W, D)
        self.t_max, self.ref_cap = int(t_max), int(ref_cap)
        self.node_cap = min(int(node_cap), la.tree_capacity(), tree_kv.capacity())
        assert self.node_cap > 0, "tree_spec: install mamba_bi_lookahead(tree=True) and tree_kv before the engine"
        self.sample_rows = sample_rows
        # checks / negatives only: the full keyed race scores and a keyed uniform (thundersync_rl.keyed_scores)
        self.scores_rows, self.uniform_rows = scores_rows, uniform_rows
        self.top2_rows = top2_rows  # keyed runner-up of rows (the MTP depth-1 sibling; None: no sibling)
        dev = torch.device("cuda", torch.cuda.current_device())
        self.dev = dev
        S = int(ctx.max_sequence_length) + 8 * LOOK_MAX + 8
        self.S = S
        i32 = dict(dtype=torch.int32, device=dev)
        self.H = torch.zeros(self.t_max, S, **i32)
        self.R = torch.full((self.t_max, self.ref_cap), SEP, **i32)
        self.RLEN = torch.zeros(self.t_max, **i32)
        # src bit1 (own history, XS-SD): 1 for a tail whose candidates come from its own history H instead of R
        self.OWN = torch.zeros(self.t_max, **i32)
        self.P = torch.zeros(self.t_max, **i32)
        self.BUD = torch.zeros(self.t_max, **i32)
        self.EOS = torch.full((self.t_max,), -1, **i32)
        self.DONE = torch.ones(self.t_max, **i32)
        # MTP draft per tail id (tree_mtp): the chain drafted after the last accept, read by the next trie build
        self.mtp_tok = torch.full((self.t_max + 1, KMTP), -1, **i32)  # row t_max: the sink of rows without a tail
        self.mtp_len = torch.zeros(self.t_max + 1, **i32)
        self.mtp_sib = torch.full((self.t_max + 1,), -1, **i32)  # depth-1 sibling: the keyed race's runner-up
        self.mtp = None  # tree_mtp.MTPHead (attach_mtp)
        self._mtp_new_q: list = []  # per forward: tails registered in it (tail, token start, length), FIFO
        # candidate blocks cover the references, and the whole history when own-history tails are enabled
        self.NB = triton.cdiv(max(self.ref_cap, S if own_history else 0), BLOCK)
        T, E = self.t_max, self.node_cap
        self.ring = []
        for _ in range(self.RING):
            # the accept results the host reads: ONE int64 device buffer (one D2H per step) with views
            # acc | nn | done | p | tokens [T, 8] | path entries [T, 8] | MTP nodes
            ibuf = torch.zeros(T * 21, dtype=torch.int64, device=dev)
            self.ring.append({
                "lay": torch.full((8, T), -1, **i32),
                "bcnt": torch.zeros(T, self.NB, NMAX, **i32),
                "cand": torch.zeros(T, CMAX, **i32),
                "ncand": torch.zeros(T, **i32),
                "ntok": torch.zeros(T, WMAX, **i32),
                "npar": torch.zeros(T, WMAX, **i32),
                "ndep": torch.zeros(T, WMAX, **i32),
                "ibuf": ibuf,
                "acc": ibuf[0:T],
                "nn": ibuf[T: 2 * T],
                "done": ibuf[2 * T: 3 * T],
                "p": ibuf[3 * T: 4 * T],
                "tok": ibuf[4 * T: 12 * T].view(T, 8),
                "path_idx": ibuf[12 * T: 20 * T].view(T, LOOK_MAX),
                "nmtp": ibuf[20 * T: 21 * T],
                "kpos": torch.zeros(E, dtype=torch.int64, device=dev),
                # per node row: key id and tail index (staged with the layout, one H2D)
                "nk": torch.zeros(2, E, dtype=torch.int64, device=dev),
                "row_now": torch.full((T,), -1, **i32),
                "s_ext": torch.zeros(E, dtype=torch.int64, device=dev),
                "alt_ext": torch.full((E,), -1, dtype=torch.int64, device=dev),
                "alt2_ext": torch.full((E,), -1, dtype=torch.int64, device=dev),
                "pacc_ext": torch.zeros(E, dtype=torch.float32, device=dev),
                "res_ext": torch.full((E,), -1, dtype=torch.int64, device=dev),
                "lp_ext": torch.zeros(E, dtype=torch.float32, device=dev),
                "slots": torch.full((T,), -1, **i32),
                "cur": torch.zeros(T, **i32),
                "path_blk": torch.full((T, LOOK_MAX), -1, **i32),
                "rblk": torch.full((T,), -1, **i32),
                "h_out": torch.zeros(T * 21, dtype=torch.int64).pin_memory(),  # == ibuf
                "h_lp": torch.zeros(max(E, 1), dtype=torch.float32).pin_memory(),
                "h_lay": torch.zeros(8, T, dtype=torch.int32).pin_memory(),
                "h_rn": torch.zeros(T, dtype=torch.int32).pin_memory(),
                "h_rid": torch.zeros(E, dtype=torch.int64).pin_memory(),
                "h_nk": torch.zeros(2, E, dtype=torch.int64).pin_memory(),
                "h_tk": torch.zeros(T, dtype=torch.int64).pin_memory(),
                "tk": torch.zeros(T, dtype=torch.int64, device=dev),
                "ev": torch.cuda.Event(),
            })
        self.alt_none = torch.full((max(int(ctx.max_requests), 1) + E + 8,), -1, dtype=torch.int64, device=dev)
        self.f_none = torch.zeros(max(int(ctx.max_requests), 1) + E + 8, dtype=torch.float32, device=dev)
        self.arange_e = torch.arange(max(E, 1), dtype=torch.int64, device=dev)
        # CHECKS tool (THUNDERSYNC_TREE_DEBUG_RUNNERUP=1): per request, the runner-up of the keyed race at every
        # sampled position of every row (plain rows too): the harness's true runner-up draft source
        self.debug_ru = (os.environ.get("THUNDERSYNC_TREE_DEBUG_RUNNERUP", "") == "1"
                         and os.environ.get("THUNDERSYNC_NL_CHECKS", "") == "1")
        self.runner_ups: dict[int, dict[int, int]] = {}
        self.ring_next = 0
        self.tails: dict[int, _Tail] = {}
        self.free_tids = list(range(self.t_max - 1, -1, -1))
        self._rid_of = np.full(self.t_max, -1, dtype=np.int64)
        self._trk = np.zeros(self.t_max, dtype=bool)
        self._wlast = np.zeros(self.t_max, dtype=np.int64)
        self._mslot_host = np.full(self.t_max, -1, dtype=np.int64)
        self.mtp_free: list = []
        self._sorted = None
        self.pending: list[_Forward] = []
        self.layout: Optional[_Forward] = None
        self.chunk = None
        # pooled, decayed acceptance per insertion index (r_i) and trials
        self.accA = np.zeros(WMAX, dtype=np.float64)
        self.accN = np.zeros(WMAX, dtype=np.float64)
        self.stats = {"iters": 0, "tail_iters": 0, "tokens": 0, "nodes": 0, "accepted_nodes": 0, "registered": 0,
                      "forwards_with_nodes": 0, "attaches": 0, "detaches": 0, "widths": [0] * (WMAX + 1),
                      "neg_tokens": 0, "crit_tokens": 0, "crit_iters": 0, "pc_tail_shared_blocks": 0,
                      "pc_tail_stale_blocks": 0, "plan_no_room": 0, "layout_dropped_no_graph": 0,
                      "mtp_nodes": 0, "mtp_accepted": 0, "other_nodes": 0, "other_accepted": 0,
                      "mtp_d1_n_a1": 0, "mtp_d1_ok_a1": 0, "mtp_d1_n_a2": 0, "mtp_d1_ok_a2": 0}
        self._seen: set = set()
        self._blk_epoch: dict = {}  # prefix cache: block id -> weight epoch it was registered under (install_epoch_map)
        rank0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
        self._log_s = float(os.environ.get("THUNDERSYNC_TREE_LOG_S", "0") or 0) if rank0 else 0.0
        self._log_t, self._log_it, self._log_prev = None, 0, None
        fk = os.environ.get("THUNDERSYNC_TAIL_MTD_DEBUG_FORCE_K", "").strip()
        self.force_w = int(fk) if fk else None  # gate tool (CHECKS only): every tail gets this width
        neg = os.environ.get("THUNDERSYNC_NL_NEG", "").strip()
        self.neg = neg if neg in NEGATIVES else ""
        self.per_request = os.environ.get("THUNDERSYNC_NL_CHECKS", "") == "1"  # nl_tree_stats under CHECKS
        self.req_stats: dict[int, list] = {}
        self._key_check = None
        self._keyreg = None
        self._key_ids = None
        self.priv = tree_kv.PrivateBlocks(ctx, self.node_cap)
        self.use_graphs = os.environ.get("THUNDERSYNC_TREE_GRAPHS", "1") != "0"
        self.graph_pool = torch.cuda.graph_pool_handle()
        self.stats["graph_captures"] = 0
        self.stats["graph_fallbacks"] = 0
        self.kv_tab = tree_kv.tables(dev)

    # ------------------------------------------------------------------ registration
    def maybe_register(self, rid: int, req) -> Optional[_Tail]:
        t = self.tails.get(rid)
        if t is not None:
            return t
        sp = req.sampling_params
        refs = getattr(sp, MTD_REFS_ATTR, None)
        spec = getattr(sp, TREE_ATTR, None)
        w_cap, d_cap = self.W, self.D
        own = False
        if spec is not None:
            v = np.frombuffer(spec, dtype=np.int32) if isinstance(spec, (bytes, bytearray)) else np.asarray(spec)
            w_cap = min(int(v[0]), self.W)
            if v.size > 1 and int(v[1]) > 0:
                d_cap = min(int(v[1]), self.D)
            # src bit1 without refs: the request's own history is its reference (XS-SD). With refs the refs are
            # the source (one source per tail for now).
            own = v.size > 2 and (int(v[2]) & 2) != 0 and refs is None
        if (refs is None and not own) or w_cap <= 0:
            return None
        return self._new_tail(rid, req, refs, own, w_cap, d_cap)

    def register_tail(self, rid: int, refs: list, w: int, d: int = 0) -> bool:
        """Make an already-decoding request a tail from its next decode step (e.g. a verified row continued in place
        after its first rejection: refs = the rejected draft's suffix and its siblings' drafts). Its history is
        uploaded from the request record on attach, as for any tail; refs are proposals only. False if the request
        is unknown, already a tail, or no tail id is free."""
        rid = int(rid)
        if rid in self.tails:
            return False
        try:
            req = self.engine.get_request(rid)
        except KeyError:  # not (or no longer) in the engine
            return False
        if req is None or w <= 0:
            return False
        packed = pack_refs(refs).tobytes() if not isinstance(refs, (bytes, bytearray)) else refs
        t = self._new_tail(rid, req, packed, False, min(int(w), self.W), min(int(d), self.D) if d else self.D)
        self._seen.add(rid)
        return t is not None

    def _new_tail(self, rid: int, req, refs, own: bool, w_cap: int, d_cap: int) -> Optional[_Tail]:
        if not self.free_tids:
            return None
        sp = req.sampling_params
        plen = len(req.prompt_tokens)
        ntg = sp.num_tokens_to_generate if sp.num_tokens_to_generate is not None else (
            (sp.num_tokens_total or self.ctx.max_sequence_length) - plen)
        eos = sp.termination_id if sp.termination_id is not None else -1
        t = _Tail(rid=rid, tid=self.free_tids.pop(), plen=plen,
                  budget_end=min(plen + int(ntg), self.S - 8 * LOOK_MAX - 8), eos=int(eos), refs=refs, own=own,
                  w_cap=w_cap, d_cap=d_cap)
        self.tails[rid] = t
        self._rid_of[t.tid] = rid
        self._trk[t.tid] = False
        self._wlast[t.tid] = 0
        self._sorted = None
        self.stats["registered"] += 1
        return t

    def _upload_refs(self, t: _Tail) -> None:
        self.OWN[t.tid].fill_(1 if t.own else 0)  # (a reused tid must not keep the last tail's source)
        if t.own:
            if self.NB * BLOCK < self.S:
                raise RuntimeError("tree_spec: own-history tails need install(own_history=True) (candidate blocks)")
            self.RLEN[t.tid].fill_(0)
            t.refs_loaded = True
            return
        r = np.frombuffer(t.refs, dtype=np.int32).copy() if isinstance(t.refs, (bytes, bytearray)) else pack_refs(t.refs)
        r = r[: self.ref_cap]
        if r.size == 0:
            r = np.full(1, SEP, dtype=np.int32)
        hr = torch.from_numpy(np.ascontiguousarray(r)).pin_memory()
        self.R[t.tid, : hr.numel()].copy_(hr, non_blocking=True)
        self.RLEN[t.tid].fill_(int(r.size))
        self._keep_refs = hr
        t.refs_loaded = True

    def _attach_upload(self, t: _Tail, req) -> None:
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

    def _detach(self, ctx, t: _Tail, idx: int, row: int, req) -> None:
        """As tail_mtd: MCore's bookkeeping back to the truth (its last forward had no nodes)."""
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

    def _count_shared_blocks(self, ctx, idx: int, filled: int) -> None:
        """Prefix-cache non-vacuity: complete blocks of a new tail's prompt held by more than one request (matched
        from the cache, or matched by a later request) at its first prefill chunk."""
        alloc = ctx.kv_block_allocator
        refs = getattr(alloc, "block_ref_counts", None)
        if refs is None or not getattr(ctx, "enable_prefix_caching", False):
            return
        nb = int(filled) // int(ctx.block_size_tokens)
        if nb <= 0:
            return
        ids = ctx.request_to_kv_block_ids[idx, :nb].long().cpu()
        ids = ids[ids >= 0]
        self.stats["pc_tail_shared_blocks"] += int((refs[ids] > 1).sum())
        # strict cross-epoch evidence: blocks this tail uses that were registered under an older weight epoch
        # (production: never, the cache is salted per epoch; NEGATIVE tree_stale_cache: its stale hits)
        ep = int(getattr(self.engine, "_weight_epoch", 0))
        self.stats["pc_tail_stale_blocks"] += sum(1 for b in ids.tolist() if self._blk_epoch.get(b, ep) < ep)

    def set_width_cap(self, rid: int, w: int) -> bool:
        """Per-request node cap between iterations (host; e.g. a fork-lane rule turning a row's drafting on / off).
        The next plan() lays out <= w nodes for it; w = 0 lays out none (the tail detaches as with any zero width and
        decodes one token per iteration). Proposals only: exactness does not depend on it. False if not a tail."""
        t = self.tails.get(int(rid))
        if t is None:
            return False
        t.w_cap = max(0, min(int(w), self.W))
        return True

    def release(self, rid: int) -> None:
        t = self.tails.pop(rid, None)
        if t is not None:
            t.tracked = False
            self._rid_of[t.tid] = -1
            self._trk[t.tid] = False
            self._wlast[t.tid] = 0
            self._sorted = None
            self.free_tids.append(t.tid)
            if t.mslot >= 0:
                self.mtp_free.append(t.mslot)
                self._mslot_host[t.tid] = -1
                t.mslot = -1
            if self.mtp is not None:  # a reused tail id must not propose the last request's drafts
                self.mtp_len[t.tid] = 0
                self.mtp_sib[t.tid] = -1

    def on_post_process(self, finished, evicted) -> None:
        """Finished and evicted requests leave: their tail state is released. An evicted request is re-admitted from
        scratch (a new prefill), so it registers again at that prefill with fresh tables (review T1)."""
        gone = list(finished.tolist()) if finished is not None else []
        if evicted is not None:
            gone += evicted.tolist() if hasattr(evicted, "tolist") else list(evicted)
        for rid in gone:
            self.release(int(rid))
            self._seen.discard(int(rid))

    def tracked_rids(self) -> np.ndarray:
        return self._rid_of[self._trk]

    def rates(self) -> np.ndarray:
        """r_i (i < WMAX): pooled acceptance of the i-th inserted node; an unobserved index takes its predecessor's
        rate as its prior (one pseudo-trial)."""
        r = np.empty(WMAX)
        prev = PRIOR0
        for i in range(WMAX):
            r[i] = (prev + self.accA[i]) / (1.0 + self.accN[i])
            prev = r[i]
        return r

    # ------------------------------------------------------------------ layout (host, before each forward)
    def plan(self, ctx, is_dummy: bool) -> int:
        self.layout = None
        if is_dummy:
            return 0
        n = ctx.total_request_count - ctx.paused_request_count
        if n <= 0:
            return 0
        lo = ctx.paused_request_count
        rids = ctx.request_ids[lo: lo + n].numpy()
        n_dec = ctx.num_decode_requests
        tok0 = n_dec  # token row of the first prefill request (decode rows: one token each)
        new = []
        for row in range(n_dec, n):
            rid = int(rids[row])
            # this forward's prefill chunk of the request: q tokens at positions off .. off + q - 1 (chunked prefill:
            # several forwards; prefix caching: off starts past the skipped cached tokens)
            q = int(ctx.request_query_lengths[lo + row])
            off = int(ctx.request_kv_length_offsets[lo + row])
            if rid not in self.tails and rid not in self._seen:
                self._seen.add(rid)
                req = self.engine.get_request(rid)
                t = self.maybe_register(rid, req)
                if t is not None:
                    self._count_shared_blocks(ctx, lo + row, off + q)
                    if self.mtp is not None and self.mtp_free:
                        t.mslot = self.mtp_free.pop()
                        self._mslot_host[t.tid] = t.mslot
                        t.mstart = off
                        self.mtp.set_start(t.mslot, off)  # positions before off have no decoder output here
            t = self.tails.get(rid)
            if t is not None and t.mslot >= 0:
                new.append((t, tok0, q, off))
            tok0 += q
        if self.mtp is not None:
            self._mtp_new_q.append(new)  # consumed by this forward's after_sample (one per sampled forward)
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
        rates = self.rates()
        first_worth = rates[0] / (1.0 + rates[0]) > BETA
        live = self._trk[tids] | (self._wlast[tids] > 0) | (self._mslot_host[tids] >= 0)
        if decode_only and (first_worth or self.force_w is not None):
            live[:] = True
        self.stats["widths"][0] += int(rows.size - np.count_nonzero(live))
        rows = rows[live]
        if rows.size == 0:
            return 0
        # node rows are attention rows of the forward: they fit in the context's request capacity and the largest
        # decode graph next to the n request rows (an engine sized for exactly its admitted rows has no room)
        room = min(int(ctx.max_requests), self._max_graph_rows(ctx)) - n
        budget = max(0, min(self.node_cap, room))
        if decode_only and budget == 0:
            self.stats["plan_no_room"] += 1
            if self.stats["plan_no_room"] == 1:
                print(f"[tree_spec] WARNING: no room for node rows ({n} request rows, max_requests "
                      f"{int(ctx.max_requests)}, largest decode graph {self._max_graph_rows(ctx)}): node rows are "
                      "attention rows next to the requests; size max_requests >= live rows + node budget "
                      "([tree] plan_no_room counts these forwards)", flush=True)
        cands = []
        for row in rows.tolist():
            t = self.tails[int(rids[row])]
            req = self.engine.get_request(t.rid)
            cands.append((t, row, req))
        if decode_only:
            if self.force_w is not None:
                ws, used = [], 0
                for t, _row, _req in cands:
                    k = max(0, min(self.force_w, t.w_cap, budget - used))
                    ws.append(k)
                    used += k
            else:
                rem = [float(t.budget_end - (t.plen + len(req.generated_tokens))) for t, _r, req in cands]
                ws = plan_widths(rem, rates, [t.w_cap for t, _r, _q in cands], budget)
        else:
            ws = [0] * len(cands)
        tails, E, dprev = [], 0, []
        for (t, row, req), k in zip(cands, ws):
            flag = 0
            if t.tracked:
                if k == 0 and t.w_last == 0 and t.mslot < 0:
                    self._detach(ctx, t, lo + row, row, req)
            elif k > 0 or t.mslot >= 0:  # an MTP tail is device-resolved from its first decode (its KV follows P)
                self._attach_upload(t, req)
                flag = 2
            if t.tracked:
                tails.append((t, row, k, flag))
                dprev.append(t.d_cap if t.w_last > 0 else 0)
                E += k
            t.w_last = k if t.tracked else 0
            self._wlast[t.tid] = t.w_last
            self.stats["widths"][k] += 1
        if not tails:
            return 0
        self._ensure_kv(ctx, tails, lo, dprev)
        self.layout = _Forward(slot=-1, tails=tails, n_rows=n, E=E)
        return E

    def _max_graph_rows(self, ctx) -> int:
        dims = getattr(ctx, "cuda_graph_batch_dimensions_list", None) or []
        dec = [int(d.token_count) for d in dims if int(getattr(d, "prefill_req_count", 0) or 0) == 0]
        return max(dec) if dec else int(ctx.max_requests)

    def _ensure_kv(self, ctx, tails, lo, dprev) -> None:
        """Blocks of the request for every position its root may reach before the host sees the next result: the
        last forward may add up to dprev + 1 tokens, this one up to d_cap + 1 (nodes write private blocks)."""
        bs = int(ctx.block_size_tokens)
        alloc = ctx.kv_block_allocator
        for (t, row, k, _pf), dp in zip(tails, dprev):
            req = self.engine.get_request(t.rid)
            p_host = t.plen + len(req.generated_tokens)
            need_pos = p_host + dp + 1 + (t.d_cap + 1 if k > 0 else 1) + 1
            idx = lo + row
            have = int(ctx.request_kv_block_counts[idx])
            need = min(need_pos // bs + 1, ctx.request_to_kv_block_ids.shape[1])
            if need > have:
                ids = alloc.allocate_memory_blocks(need - have)
                if ids is None:
                    raise RuntimeError("[tree_spec] KV pool exhausted for a tail's blocks")
                ctx.request_to_kv_block_ids[idx, have:need] = ids
                ctx.request_kv_block_counts[idx] = need

    def finish_layout(self, ctx) -> None:
        lay = self.layout
        if lay is None:
            return
        padded = int(ctx.padded_active_token_count)
        if lay.E > 0 and (lay.n_rows + lay.E > padded or not ctx.using_cuda_graph_this_step()):
            self.stats["layout_dropped_no_graph"] += 1
            lay.tails = [(t, row, 0, pf) for t, row, _, pf in lay.tails]
            lay.E = 0
            for t, _row, _k, _pf in lay.tails:
                t.w_last = 0
                self._wlast[t.tid] = 0

    # ------------------------------------------------------------------ forward (device)
    def _chunk(self) -> int:
        if self.chunk is None:
            self.chunk = int(getattr(self.ctx, "mamba_chunk_size", 128) or 128)
        return self.chunk

    def expand(self, ctx) -> None:
        from nemo_rl.models.generation.megatron import tree_kv
        from nemo_rl.models.megatron import mamba_bi_fused_decode as fd
        from nemo_rl.models.megatron import mamba_bi_lookahead as la

        lay = self.layout
        self.layout = None
        self.pending = []
        if lay is None:
            tree_kv.clear(ctx)
            return
        r = self.ring[self.ring_next]
        lay.slot = self.ring_next
        self.ring_next = (self.ring_next + 1) % self.RING
        T = len(lay.tails)
        h = r["h_lay"]
        h.fill_(-1)
        hr = r["h_rid"]
        hnk = r["h_nk"]
        x = lay.n_rows
        slots = ctx.mamba_metadata.request_to_mamba_state_idx
        lo = ctx.paused_request_count
        for j, (t, row, k, pf) in enumerate(lay.tails):
            h[0, j], h[1, j], h[2, j], h[3, j] = t.tid, row, k, x
            h[4, j] = int(slots[lo + row])
            h[6, j] = pf
            h[7, j] = t.mslot
            if t.kid < 0:
                t.kid = int(self._key_ids(np.asarray([t.rid], dtype=np.int64))[0]) if self._key_ids else t.rid
            r["h_tk"][j] = t.kid
            if k:
                hr[x - lay.n_rows: x - lay.n_rows + k] = t.rid
                if t.kid < 0:
                    t.kid = int(self._key_ids(np.asarray([t.rid], dtype=np.int64))[0]) if self._key_ids else t.rid
                kid = t.kid
                hnk[0, x - lay.n_rows: x - lay.n_rows + k] = kid
                hnk[1, x - lay.n_rows: x - lay.n_rows + k] = j
            x += k
        h[5, 0] = lay.n_rows
        r["lay"].copy_(h, non_blocking=True)
        if self.mtp is not None:
            r["tk"].copy_(r["h_tk"], non_blocking=True)
        if lay.E:
            r["nk"][:, : lay.E].copy_(hnk[:, : lay.E], non_blocking=True)
        lay.staging = h
        if self._graphs_on():
            self._run_graph(r, "gA", lambda: self._expand_device(ctx, r, self.t_max, None), self._sig_expand(ctx))
        else:
            self._expand_device(ctx, r, T, int(ctx.padded_active_request_count))
        if lay.E > 0:
            self.kv_tab["dirty"] = True
        self.pending = [lay]
        self.stats["iters"] += 1
        self.stats["tail_iters"] += T
        self.stats["nodes"] += lay.E
        self.stats["forwards_with_nodes"] += int(lay.E > 0)

    # ------------------------------------------------------------------ MTP source (tree_mtp)
    def attach_mtp(self, head) -> None:
        """Use ``head`` (tree_mtp.MTPHead) as candidate 0: one KV slot per tail while slots last. The cost of one
        draft depth (t_max rows: the layer + the keyed sampler) is measured once here (the depth rule's price)."""
        self.mtp = head
        self.mtp_free = list(range(head.slots - 1, -1, -1))
        self.stats.update(mtp_tails=0, mtp_prefill_tokens=0, mtp_depths=[0] * (KMTP + 1))
        self._mtp_t_iter = None  # EMA of the engine iteration period (s), between after_sample calls
        self._mtp_t_last = None
        self.mtp_depth_s = self._calibrate_mtp()
        print(f"[tree_spec] MTP draft source: {head.slots} slots, one depth {1e3 * self.mtp_depth_s:.3f} ms "
              f"({self.t_max} rows)", flush=True)

    def _calibrate_mtp(self) -> float:
        """Seconds per draft depth: the head's row forward + the keyed sampler over t_max rows (sink slot rows: no
        cache is touched), timed after a warmup."""
        TM, dev = self.t_max, self.dev
        sl = torch.full((TM,), -1, dtype=torch.long, device=dev)
        pos = torch.zeros(TM, dtype=torch.long, device=dev)
        h = torch.zeros(TM, getattr(self.mtp, "in_width", self.mtp.emb.shape[1]), dtype=torch.bfloat16, device=dev)
        tok = torch.zeros(TM, dtype=torch.long, device=dev)
        kid = torch.zeros(TM, dtype=torch.int64, device=dev)
        temp = torch.ones(TM, dtype=torch.float32, device=dev)

        def one():
            _u, lg = self.mtp.forward_rows(sl, pos, h, tok)
            self.sample_rows(lg, kid, pos, temp)
        for _ in range(3):
            one()
        run = one
        if self._graphs_on():  # priced as it runs: the draft stage is graph-replayed (no launch overhead)
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                one()
            torch.cuda.current_stream().wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                one()
            run = g.replay
            run()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(10):
            run()
        e1.record()
        e1.synchronize()
        return e0.elapsed_time(e1) / 10 / 1e3

    def _mtp_depth(self) -> int:
        """Draft depth of this iteration: the longest prefix of depths whose marginal node is worth its price, the
        width rule's test with the MTP depth's own cost: r_k / s(k + 1) > c_depth / T_iter (past acceptance per
        insertion index, the MTP chain being the first candidate; T_iter the measured iteration period)."""
        import time

        now = time.perf_counter()
        if self._mtp_t_last is not None:
            dt = now - self._mtp_t_last
            self._mtp_t_iter = dt if self._mtp_t_iter is None else 0.9 * self._mtp_t_iter + 0.1 * dt
        self._mtp_t_last = now
        if self.force_w is not None or self._mtp_t_iter is None:
            k = KMTP
        else:
            k = mtp_depth(self.rates(), self.mtp_depth_s / max(self._mtp_t_iter, 1e-6), KMTP)
        self.stats["mtp_depths"][k] += 1
        return k

    def _sig_mtp(self, ctx):
        from nemo_rl.models.generation.megatron import tree_mtp

        return self._sig_common(ctx) + (ctx.gpu_view.temperature.data_ptr(), tree_mtp.hidden_buffer().data_ptr(),
                                        self.mtp.kc.data_ptr(), self.mtp_tok.data_ptr(), self.mtp_sib.data_ptr(),
                                        self.top2_rows is not None)

    def _mtp_prefill(self, ctx, ctrl) -> None:
        """Tails with a prefill chunk in this forward: the head's KV of the chunk's positions off .. off + q - 1
        (decoder outputs of the chunk's tokens; next tokens = the prompt shifted by one, and the token sampled from
        the request when the chunk ends its prompt). K / V only: no attention is needed to fill a cache."""
        from nemo_rl.models.generation.megatron import tree_mtp

        hid = tree_mtp.hidden_buffer()
        n = ctx.total_request_count - ctx.paused_request_count
        lo = ctx.paused_request_count
        cur = ctx.request_ids[lo: lo + n].numpy()
        slots, poss, hrows, toks, last_rows, rows_now = [], [], [], [], [], []
        new = self._mtp_new_q.pop(0) if self._mtp_new_q else []
        k = 0
        for t, tok0, q, off in new:
            hit = np.nonzero(cur == t.rid)[0]
            if t.rid not in self.tails or hit.size == 0 or t.mslot < 0 or q <= 0:
                continue
            req = self.engine.get_request(t.rid)
            pt = req.prompt_tokens
            prompt = pt.tolist() if torch.is_tensor(pt) else list(pt)
            plen = len(prompt)
            ends = off + q >= plen
            nxt = prompt[off + 1: off + q + 1]
            if ends:
                nxt = nxt[: q - 1] + [-1]  # the last position's next token is this forward's sample
                last_rows.append(k + q - 1)
                rows_now.append(int(hit[0]))
            slots.append(np.full(q, t.mslot))
            poss.append(np.arange(off, off + q))
            hrows.append(np.arange(tok0, tok0 + q))
            toks.append(np.asarray(nxt, dtype=np.int64))
            k += q
            if off == t.mstart:
                self.stats["mtp_tails"] += 1
            self.stats["mtp_prefill_tokens"] += q
        if not slots:
            return
        dev = self.dev
        sl = torch.from_numpy(np.concatenate(slots)).to(dev)
        ps = torch.from_numpy(np.concatenate(poss)).to(dev)
        hr = torch.from_numpy(np.concatenate(hrows)).to(dev).clamp(max=hid.shape[0] - 1)
        tk = torch.from_numpy(np.concatenate(toks)).to(dev)
        if last_rows:
            tk[torch.tensor(last_rows, device=dev)] = ctrl._sampled_tokens_cuda.index_select(
                0, torch.tensor(rows_now, device=dev)).to(torch.int64)
        self.mtp.append_kv(sl, ps, hid.index_select(0, hr), tk)

    def _mtp_device(self, ctx, r, n_rows_dev, depth_k: int) -> None:
        """After the accept (fixed shapes, graph-replayable per depth_k): per tail with a slot, the head's KV of its
        accepted positions P-1 .. P+a-2, then a depth_k draft chain from its last one: depth 1 drafts x_{P+a} with
        the key of P+a (the next root's draw), depth k chains the layer's output hidden and the drawn token. depth_k
        0: K / V only (the head's cache stays complete; no draft)."""
        from nemo_rl.models.generation.megatron import tree_mtp

        hid = tree_mtp.hidden_buffer()
        TM = self.t_max
        lay = r["lay"]
        tid = lay[0].long()
        mslot = lay[7].long()
        acc, p = r["acc"].long(), r["p"].long()
        ok = (tid >= 0) & (mslot >= 0) & (r["row_now"].long() >= 0) & (acc >= 1)
        k8 = torch.arange(8, device=self.dev)
        pos = p[:, None] - 1 + k8[None, :]
        prow = (n_rows_dev + r["path_idx"].long()).clamp(min=0)  # [TM, 8]: row of the node at depth k + 1
        hrow = torch.where(k8[None, :] == 0, lay[1].long().clamp(min=0)[:, None], prow.roll(1, dims=1))
        tok = r["tok"].long()
        app = ok[:, None] & (k8[None, :] < (acc - 1)[:, None])
        sl = torch.where(app, mslot[:, None], torch.full_like(pos, -1)).reshape(-1)
        hsel = hid.index_select(0, hrow.reshape(-1).clamp(max=hid.shape[0] - 1))
        self.mtp.append_kv(sl, pos.reshape(-1), hsel, tok.reshape(-1).clamp(min=0))
        # the last accepted position: full layer -> depth-1 draft
        a1 = (acc - 1).clamp(min=0)
        lpos = p - 1 + a1
        lh = hid.index_select(0, hrow.gather(1, a1[:, None])[:, 0].clamp(max=hid.shape[0] - 1))
        ltok = tok.gather(1, a1[:, None])[:, 0].clamp(min=0)
        lsl = torch.where(ok, mslot, torch.full_like(mslot, -1))
        temp = ctx.gpu_view.temperature.index_select(0, r["row_now"].long().clamp(min=0))
        drafts = []
        u, x = lh, ltok
        sib = None
        for depth in range(depth_k):
            u, logits = self.mtp.forward_rows(lsl, lpos + depth, u if depth == 0 else self.mtp.chain_input(u, lh), x)
            d = self.sample_rows(logits, r["tk"], p + acc + depth, temp).to(torch.int64)
            if depth == 0 and self.top2_rows is not None:
                sib = self.top2_rows(logits, r["tk"], p + acc, temp)  # same key and position as the depth-1 draw
            drafts.append(d)
            x = d
        live = ok & (r["done"] == 0)
        dst = torch.where(tid >= 0, tid, torch.full_like(tid, TM))
        dt = torch.full((tid.shape[0], KMTP), -1, dtype=torch.int32, device=self.dev)
        if depth_k:
            dt[:, :depth_k] = torch.stack(drafts, 1).to(torch.int32)
        self.mtp_tok.index_copy_(0, dst, torch.where(live[:, None], dt, torch.full_like(dt, -1)))
        self.mtp_len.index_copy_(0, dst, torch.where(live, torch.full_like(acc, depth_k),
                                                     torch.zeros_like(acc)).to(torch.int32))
        sv = sib.to(torch.int32) if sib is not None else torch.full_like(tid, -1, dtype=torch.int32)
        self.mtp_sib.index_copy_(0, dst, torch.where(live, sv, torch.full_like(sv, -1)))

    # ------------------------------------------------------------------ device stages (graph-replayable)
    def _graphs_on(self) -> bool:
        """The production path replays the tree's device stages as CUDA graphs (no per-step launches); negative
        controls and the CHECKS runner-up dump stay eager."""
        return self.use_graphs and not self.neg and not self.debug_ru

    def _run_graph(self, r, name, body, sig) -> None:
        """First use of a ring slot: eager (compiles the kernels); second: capture, then replay; a buffer that moved
        (its address is part of ``sig``) forces a recapture."""
        g = r.get(name)
        if g is not None and r.get(name + "_sig") == sig:
            g.replay()
            return
        if not r.get(name + "_warm"):
            body()
            r[name + "_warm"] = True
            return
        g = torch.cuda.CUDAGraph()
        try:
            # thread_local: other engine-process threads (e.g. a hidden-log seal thread's event.synchronize) may make
            # unsafe CUDA calls while a tree stage is captured lazily mid-run (a new ring slot / MTP depth); global
            # mode would invalidate the capture and fail their call (BX-SD)
            with torch.cuda.graph(g, pool=self.graph_pool, capture_error_mode="thread_local"):
                body()
        except Exception as ex:  # a stage that cannot be captured: eager from now on (correct, slower), loudly
            print(f"[tree_spec] CUDA graph capture of {name} failed ({type(ex).__name__}: {ex}); eager stages",
                  flush=True)
            self.use_graphs = False
            self.stats["graph_fallbacks"] += 1
            torch.cuda.synchronize()
            body()
            return
        r[name], r[name + "_sig"] = g, sig
        self.stats["graph_captures"] += 1
        g.replay()

    def _sig_common(self, ctx):
        """Every buffer both captured stages read or write that lives outside the ring slot: a moved one (any
        reallocation) forces a recapture."""
        from nemo_rl.models.generation.megatron import tree_kv
        from nemo_rl.models.megatron import mamba_bi_fused_decode as fd
        from nemo_rl.models.megatron import mamba_bi_lookahead as la

        mr = int(ctx.max_requests)
        off, chain = fd.lookahead_buffers(mr, self.dev)
        ws = getattr(getattr(self.sample_rows, "__self__", None), "__dict__", {}).get("_nl_mtd_ws")
        ptrs = [off.data_ptr(), chain.data_ptr(), fd.lookahead_rows(mr, self.dev).data_ptr(),
                la.scratch_ids(mr, self.dev).data_ptr(), self.kv_tab["gath"].data_ptr(), self.kv_tab["part"].data_ptr(),
                self.priv.ids.data_ptr(), -1 if ws is None else ws.data_ptr(), ws.numel() if ws is not None else 0]
        if la._STATE["nbuf"]:
            ptrs.append(la.num_buffered_ptrs(self.dev).data_ptr())
        if la._STATE["replay"]:
            ptrs.append(la.replay_ptrs(self.dev)[0].data_ptr())
        return tuple(ptrs)

    def _sig_expand(self, ctx):
        gv = ctx.gpu_view
        return self._sig_common(ctx) + (gv.token_to_input_ids.data_ptr(), gv.mha_block_table.data_ptr(), gv.mha_query_lengths.data_ptr(),
                gv.mha_kv_seq_lengths.data_ptr(), gv.mha_cu_query_seq_lengths.data_ptr(), ctx.active_logit_idxs.data_ptr(),
                gv.mamba_batch_indices_decode.data_ptr(), gv.real_token_count.data_ptr(), ctx.memory_buffer.data_ptr())

    def _expand_device(self, ctx, r, TG: int, padded) -> None:
        """attach, candidates, trie, node-row layout, cumulative lengths, token count, KV partial copy. ``TG`` = grid
        rows (the ring's capacity in graph mode: entries without a tail exit at once); ``padded`` None = the full
        request arrays (graph mode: a fixed shape; entries past the padded batch are never read)."""
        from nemo_rl.models.generation.megatron import tree_kv
        from nemo_rl.models.megatron import mamba_bi_fused_decode as fd
        from nemo_rl.models.megatron import mamba_bi_lookahead as la

        gv = ctx.gpu_view
        _attach_kernel[(TG,)](self.H, self.P, r["lay"], gv.token_to_input_ids, self.H.stride(0), r["lay"].stride(0))
        _count_kernel[(TG, self.NB)](self.H, self.R, self.RLEN, self.P, self.OWN, r["lay"], r["bcnt"], self.H.stride(0),
                                     self.R.stride(0), r["lay"].stride(0), self.NB, BLOCKC=BLOCK, NMAXC=NMAX)
        _cand_kernel[(TG, self.NB)](self.H, self.R, self.RLEN, self.P, self.OWN, r["lay"], r["bcnt"], r["cand"],
                                    r["ncand"], self.H.stride(0), self.R.stride(0), r["lay"].stride(0), self.NB,
                                    BLOCKC=BLOCK, NMAXC=NMAX, NBPOW=triton.next_power_of_2(self.NB), CMAXC=CMAX)
        _build_kernel[(TG,)](self.H, self.R, self.RLEN, self.P, self.OWN, self.BUD, r["lay"], r["cand"], r["ncand"],
                             r["ntok"], r["npar"], r["ndep"], r["nn"], r["p"], self.mtp_tok, self.mtp_len, self.mtp_sib,
                             r["nmtp"],
                             self.H.stride(0), self.R.stride(0), r["lay"].stride(0), self.mtp_tok.stride(0),
                             self.D if self.force_w is None else LOOK_MAX - 1,
                             CHUNK=self._chunk(), BS=int(ctx.block_size_tokens), WPOW=WMAX, CMAXC=CMAX,
                             LOOK=LOOK_MAX, ALWAYS=self.neg == "tree_accept_all", KM=KMTP)
        off, chain = fd.lookahead_buffers(int(ctx.max_requests), self.dev)
        srow = fd.lookahead_rows(int(ctx.max_requests), self.dev)
        sid = la.scratch_ids(int(ctx.max_requests), self.dev)
        bt = gv.mha_block_table
        _expand_kernel[(TG, WMAX + 1)](
            self.H, r["lay"], r["ntok"], r["npar"], r["ndep"], r["nn"], r["p"], self.priv.ids, r["rblk"],
            gv.token_to_input_ids, gv.token_to_pos_ids, gv.token_to_block_idx,
            gv.token_to_local_position_within_kv_block, gv.token_to_position_in_request, gv.token_to_request_idx,
            gv.mha_query_lengths, gv.mha_kv_seq_lengths, bt, gv.mamba_batch_indices_decode, ctx.active_logit_idxs,
            off, chain, srow, sid, r["kpos"], self.kv_tab["gath"], self.kv_tab["part"],
            self.H.stride(0), r["lay"].stride(0), bt.stride(0), chain.stride(0), srow.stride(0),
            bt.shape[1], int(ctx.kv_block_allocator.dummy_block_idx), int(ctx.max_requests),
            {"tree_key_off_by_one": 1, "tree_key_reuse": -1}.get(self.neg, 0),
            BS=int(ctx.block_size_tokens), WPOW=WMAX, BTCOLS=triton.next_power_of_2(bt.shape[1]), LOOK=LOOK_MAX,
        )
        L = padded if padded is not None else min(gv.mha_query_lengths.numel(), gv.mha_cu_query_seq_lengths.numel() - 1,
                                                  gv.mha_kv_seq_lengths.numel(), gv.mha_cu_kv_seq_lengths.numel() - 1)
        torch.cumsum(gv.mha_query_lengths[:L], 0, dtype=torch.int32, out=gv.mha_cu_query_seq_lengths[1: L + 1])
        _rtc_kernel[(1,)](r["lay"], gv.real_token_count, r["lay"].stride(0), TP=triton.next_power_of_2(self.t_max))
        tree_kv.partial_copy(ctx)
        torch.cumsum(gv.mha_kv_seq_lengths[:L], 0, dtype=torch.int32, out=gv.mha_cu_kv_seq_lengths[1: L + 1])

    def after_forward(self, ctx) -> None:
        """Node rows' Mamba offsets back to 0 and the KV gather tables cleared: one fixed-shape launch, stream-ordered
        after the forward."""
        from nemo_rl.models.generation.megatron import tree_kv
        from nemo_rl.models.megatron import mamba_bi_fused_decode as fd

        if not self.pending:
            return
        off, _chain = fd.lookahead_buffers(int(ctx.max_requests), self.dev)
        r = self.ring[self.pending[-1].slot]
        _post_fwd_kernel[(self.node_cap,)](off, self.kv_tab["gath"], self.kv_tab["part"], r["lay"], r["lay"].stride(0),
                                           off.numel(), LOOK=LOOK_MAX, GP=triton.next_power_of_2(3 * LOOK_MAX))
        self.kv_tab["dirty"] = False
        del tree_kv

    def _sig_sample(self, ctx, ctrl):
        return self._sig_common(ctx) + (ctx.gpu_view.temperature.data_ptr(), ctx.mamba_conv_states_full.data_ptr(), ctx.memory_buffer.data_ptr(),
                id(ctrl._sampling))

    def _sample_device(self, ctx, ctrl, r) -> None:
        """Every node row of the capacity drawn with its key (rows past this forward's nodes are never read), their
        log-probs, the accept walk, the commits and the results' D2H: fixed shapes, no host sync."""
        from nemo_rl.models.generation.megatron import tree_kv
        from nemo_rl.models.megatron import mamba_bi_lookahead as la

        TM, E = self.t_max, self.node_cap
        nk = r["nk"]
        temp = ctx.gpu_view.temperature.index_select(
            0, r["row_now"].index_select(0, nk[1]).clamp(min=0).long())
        s = self.sample_rows(r["lg_buf"], nk[0], r["kpos"], temp).to(torch.int64)
        r["s_ext"].copy_(s)
        sel, _ = ctx.calculate_log_probs_tensors(r["lg_buf"].unsqueeze(0), s, only_last_token_logits=True,
                                                 sampling=ctrl._sampling)
        r["lp_ext"].copy_(sel.float())
        _accept_kernel[(TM,)](
            self.H, self.P, self.BUD, self.EOS, self.DONE, r["lay"], r["row_now"], r["s0_buf"],
            r["s_ext"], self.alt_none, r["alt_ext"], self.alt_none, r["alt2_ext"], self.f_none, r["pacc_ext"],
            self.alt_none, r["res_ext"], r["ntok"], r["npar"], r["nn"], r["p"], self.priv.ids,
            r["acc"], r["tok"], r["slots"], r["cur"], r["path_idx"], r["path_blk"], r["rblk"], r["done"],
            self.H.stride(0), r["lay"].stride(0), r["tok"].stride(0), r["path_idx"].stride(0),
            CHUNK=self._chunk(), BS=int(ctx.block_size_tokens), WPOW=WMAX, LOOK=LOOK_MAX, KPOW=8,
            TOPK=False, ACCEPT_ALL=False, DRAFTPROB=False,
        )
        la.commit_tree(ctx.mamba_conv_states_full, r["slots"], r["path_idx"], r["acc"], r["cur"], self._chunk(),
                       int(ctx.max_requests))
        tree_kv.commit_copy(ctx, r["rblk"], r["acc"], r["p"], r["path_blk"])
        r["h_out"].copy_(r["ibuf"], non_blocking=True)
        r["h_lp"][:E].copy_(r["lp_ext"][:E], non_blocking=True)

    # ------------------------------------------------------------------ after MCore sampled (device)
    def _row_keys(self, rids: np.ndarray, extra: Optional[np.ndarray] = None):
        """Key ids (the sampler's) of request ids, as a device int64 tensor."""
        kid = self._key_ids(rids) if self._key_ids else rids
        return torch.from_numpy(np.asarray(kid, dtype=np.int64)).pin_memory().to(self.dev, non_blocking=True)

    def _root_positions(self, ctx, n):
        """Key position of every active row's draw this step: prompt length + generated tokens (host view)."""
        lo = ctx.paused_request_count
        rids = ctx.request_ids[lo: lo + n].numpy().astype(np.int64)
        pos = np.empty(n, dtype=np.int64)
        for i, rid in enumerate(rids.tolist()):
            req = self.engine.get_request(int(rid))
            pos[i] = len(req.prompt_tokens) + len(req.generated_tokens)
        return rids, torch.from_numpy(pos).pin_memory().to(self.dev, non_blocking=True), pos

    def _record_runner_ups(self, ctx, ctrl) -> None:
        n = ctx.total_request_count - ctx.paused_request_count
        if n <= 0 or self.scores_rows is None:
            return
        rids, pos_d, pos = self._root_positions(ctx, n)
        logits = ctrl._all_logits_cuda.squeeze(0)[:n]
        sc = self.scores_rows(logits, rids, pos_d, ctx.gpu_view.temperature[:n])
        draw = ctrl._sampled_tokens_cuda[:n].to(torch.int64)
        top2 = torch.topk(sc, 2, dim=-1).indices
        ru = torch.where(top2[:, 0] == draw, top2[:, 1], top2[:, 0]).cpu().numpy()
        for i, rid in enumerate(rids.tolist()):
            self.runner_ups.setdefault(int(rid), {})[int(pos[i])] = int(ru[i])

    def pop_runner_ups(self, rid: int) -> Optional[dict]:
        return self.runner_ups.pop(int(rid), None)

    def _neg_arrays(self, r, lay, ctx, ctrl, logits, n, rows_dev, lgl, rids_k, kpos_rows, temp_k):
        """Negative controls only (tree_accept_topk, tree_accept_by_draft_prob): per row (the step's active rows and
        the node rows) the keyed-race runner-up, and (draft-prob) the accept indicator u < p(c) of its first child c
        and the residual race without c. Returns the root-row arrays; the node-row ones go to the ring slot."""
        E = lay.E
        T = len(lay.tails)
        rids, pos_d, _pos = self._root_positions(ctx, n)
        temp0 = ctx.gpu_view.temperature[:n]
        logits0 = logits[:n]
        draw0 = ctrl._sampled_tokens_cuda[:n].to(torch.int64)
        raw2 = torch.topk(logits0.float(), 2, dim=-1).indices
        alt_root = self.alt_none.clone()
        alt_root[:n] = torch.where(raw2[:, 0] == draw0, raw2[:, 1], raw2[:, 0])
        alt2_root, pacc_root, res_root = self.alt_none.clone(), self.f_none.clone(), self.alt_none.clone()
        sc0 = self.scores_rows(logits0, rids, pos_d, temp0) if self.scores_rows is not None else None
        if sc0 is not None:
            t2 = torch.topk(sc0, 2, dim=-1).indices
            alt2_root[:n] = torch.where(t2[:, 0] == draw0, t2[:, 1], t2[:, 0])
        sc_k = None
        if rows_dev is not None and sc0 is not None:
            sc_k = self.scores_rows(lgl, rids_k, kpos_rows, temp_k)
            t2 = torch.topk(sc_k, 2, dim=-1).indices
            s_k = r["s_ext"].index_select(0, rows_dev)
            r["alt2_ext"].index_copy_(0, rows_dev, torch.where(t2[:, 0] == s_k, t2[:, 1], t2[:, 0]))
        if self.neg != "tree_accept_by_draft_prob" or sc0 is None or self.uniform_rows is None:
            return alt_root, alt2_root, pacc_root, res_root
        # first child of every parent (root = -1, node k) of every tail, on device
        W = WMAX
        idx = torch.arange(W, device=self.dev)
        npar = r["npar"][:T].long()
        valid = idx[None, :] < r["nn"][:T].long()[:, None]
        ntok = r["ntok"][:T].long()

        def first_child(mask):  # [..., W] -> first index or -1
            anyc = mask.any(-1)
            return torch.where(anyc, torch.argmax(mask.int(), dim=-1), torch.full_like(anyc, -1, dtype=torch.long))

        fc_root = first_child((npar == -1) & valid)  # [T]
        fc_node = first_child((npar[:, None, :] == idx[None, :, None]) & valid[:, None, :])  # [T, W]
        tok_root = torch.where(fc_root >= 0, ntok.gather(1, fc_root.clamp(min=0)[:, None])[:, 0], -1)
        tok_node = torch.where(fc_node >= 0, ntok.gather(1, fc_node.clamp(min=0)), -1)  # [T, W]

        def accept_and_residual(sc, lg, temp, kid, pos, c):
            p = torch.softmax(lg.float() / temp.float()[:, None], dim=-1)
            pc = torch.where(c >= 0, p.gather(1, c.clamp(min=0)[:, None])[:, 0], torch.zeros_like(p[:, 0]))
            u = self.uniform_rows(kid, pos)
            scm = sc.clone()
            scm.scatter_(1, c.clamp(min=0)[:, None], float("-inf"))
            res = torch.where(c >= 0, torch.argmax(scm, dim=-1), -1)
            return ((u < pc) & (c >= 0)).float(), res

        # root rows: the tails' rows now (row_now) carry their tree's first child
        rn = r["row_now"][:T].long()
        c0 = torch.full((n,), -1, dtype=torch.long, device=self.dev)
        ok = rn >= 0
        c0[rn[ok]] = tok_root[ok]
        kid0 = self._row_keys(rids)
        q0, res0 = accept_and_residual(sc0, logits0, temp0, kid0, pos_d, c0)
        pacc_root[:n], res_root[:n] = q0, res0
        if rows_dev is not None:
            # node rows: entry (x0 - n_rows) + k of tail j carries node k's first child
            x0 = (r["lay"][3][:T].long() - lay.n_rows)
            ent = (x0[:, None] + idx[None, :]).reshape(-1)
            tok_e = torch.full((E,), -1, dtype=torch.long, device=self.dev)
            keep = (valid.reshape(-1)) & (ent >= 0) & (ent < E)
            tok_e[ent[keep]] = tok_node.reshape(-1)[keep]
            ck = tok_e.index_select(0, rows_dev)
            qk, rk = accept_and_residual(sc_k, lgl, temp_k, self._row_keys(rids_k), kpos_rows, ck)
            r["pacc_ext"].index_copy_(0, rows_dev, qk)
            r["res_ext"].index_copy_(0, rows_dev, rk)
        return alt_root, alt2_root, pacc_root, res_root

    def after_sample(self, ctrl) -> None:
        if self.debug_ru:
            self._record_runner_ups(self.ctx, ctrl)
        if self.mtp is not None:
            self._mtp_prefill(self.ctx, ctrl)
        if not self.pending:
            return
        from nemo_rl.models.generation.megatron import tree_kv
        from nemo_rl.models.megatron import mamba_bi_lookahead as la

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
        live = []
        for j, (t, row, k, _pf) in enumerate(lay.tails):
            now = rid_row.get(t.rid, -1)
            if t.rid in self.tails and now >= 0:
                rn[j] = now
                live.append((j, t, now, k))
        r["row_now"].copy_(rn, non_blocking=True)
        logits = ctrl._all_logits_cuda.squeeze(0)
        E = lay.E
        if self._graphs_on():
            # production path: the node rows' logits and the roots' draws into fixed buffers, then one replay of
            # sample / log-prob / accept / commit / D2H
            if r.get("lg_buf") is None or r["lg_buf"].shape[1] != logits.shape[-1] or r["lg_buf"].dtype != logits.dtype:
                r["lg_buf"] = torch.zeros(self.node_cap, logits.shape[-1], dtype=logits.dtype, device=self.dev)
                r["s0_buf"] = torch.zeros(int(ctx.max_requests), dtype=torch.int64, device=self.dev)
                r["gB"] = None
                r["gB_warm"] = False
            if E:
                r["lg_buf"][:E].copy_(logits[lay.n_rows: lay.n_rows + E])
            ns = min(r["s0_buf"].numel(), ctrl._sampled_tokens_cuda.numel())
            r["s0_buf"][:ns].copy_(ctrl._sampled_tokens_cuda[:ns])
            self._run_graph(r, "gB", lambda: self._sample_device(ctx, ctrl, r), self._sig_sample(ctx, ctrl))
            r["ev"].record()
            if self.mtp is not None:  # after the results' event: the host never waits on the drafts
                k = self._mtp_depth()
                self._run_graph(r, f"gM{k}", lambda: self._mtp_device(ctx, r, r["lay"][5, 0].long(), k),
                                self._sig_mtp(ctx))
            lay.result = (r, len(lay.tails), len(lay.tails) * 8, live)
            self._last_sampled = lay
            return
        alt_root = self.alt_none
        # Node rows: all E rows of the layout, contiguous after the request rows, sampled together (a masked row's
        # draw is never read). Key ids and tail indices were staged with the layout (one H2D); key positions and the
        # temperature row (the tail's row now) are gathered on device: no per-row host work.
        rows_dev = lgl = temp = kpos_e = None
        rel = E > 0
        if E:
            lgl = logits[lay.n_rows: lay.n_rows + E]
            if self.neg == "tree_non_bi_verify":
                # NEGATIVE CONTROL: node rows drawn and scored from logits through another numeric path (rounded to
                # bf16): not one-token decode's bits
                lgl = lgl.to(torch.bfloat16).float()
            nk = r["nk"][:, :E]
            temp = ctx.gpu_view.temperature.index_select(
                0, r["row_now"].index_select(0, nk[1]).clamp(min=0).long())
            kid = nk[0] ^ (1 << 40) if self.neg == "tree_fresh_keys" else nk[0]  # NEGATIVE CONTROL: another key id
            kpos_e = r["kpos"][:E]
            s = self.sample_rows(lgl, kid, kpos_e, temp).to(torch.int64)
            r["s_ext"][:E].copy_(s)
            sel, _ = ctx.calculate_log_probs_tensors(lgl.unsqueeze(0), s, only_last_token_logits=True,
                                                     sampling=ctrl._sampling)
            r["lp_ext"][:E].copy_(sel.float())
            rows_dev = self.arange_e[:E]
            if self.neg == "tree_accept_topk":
                top2 = torch.topk(lgl.float(), 2, dim=-1).indices
                r["alt_ext"][:E].copy_(torch.where(top2[:, 0] == s, top2[:, 1], top2[:, 0]))
        rids_k = r["h_rid"][:E].numpy() if E else None
        alt2_root, pacc_root, res_root = self.alt_none, self.f_none, self.alt_none
        if self.neg in ("tree_accept_topk", "tree_accept_by_draft_prob") and E:
            r["alt2_ext"].fill_(-1)
            r["pacc_ext"].zero_()
            r["res_ext"].fill_(-1)
            have = bool(rel)
            alt_root, alt2_root, pacc_root, res_root = self._neg_arrays(
                r, lay, ctx, ctrl, logits, n, rows_dev if have else None, lgl if have else None,
                rids_k if have else None, kpos_e if have else None, temp if have else None)
            if self.neg == "tree_accept_by_draft_prob":
                alt_root = self.alt_none
        T = len(lay.tails)
        _accept_kernel[(T,)](
            self.H, self.P, self.BUD, self.EOS, self.DONE, r["lay"], r["row_now"], ctrl._sampled_tokens_cuda,
            r["s_ext"], alt_root, r["alt_ext"], alt2_root, r["alt2_ext"], pacc_root, r["pacc_ext"], res_root,
            r["res_ext"], r["ntok"], r["npar"], r["nn"], r["p"], self.priv.ids,
            r["acc"], r["tok"], r["slots"], r["cur"], r["path_idx"], r["path_blk"], r["rblk"], r["done"],
            self.H.stride(0), r["lay"].stride(0), r["tok"].stride(0), r["path_idx"].stride(0),
            CHUNK=self._chunk(), BS=int(ctx.block_size_tokens), WPOW=WMAX, LOOK=LOOK_MAX, KPOW=8,
            TOPK=self.neg == "tree_accept_topk", ACCEPT_ALL=self.neg == "tree_accept_all",
            DRAFTPROB=self.neg == "tree_accept_by_draft_prob",
        )
        if E > 0:
            la.commit_tree(ctx.mamba_conv_states_full, r["slots"][:T], r["path_idx"][:T], r["acc"][:T],
                           r["cur"][:T], self._chunk(), int(ctx.max_requests),
                           skip_replay=self.neg == "tree_stale_mamba")
            if self.neg != "tree_skip_kv_rollback":  # NEGATIVE CONTROL skips the accepted path's K / V copy
                tree_kv.commit_copy(ctx, r["rblk"][:T], r["acc"][:T], r["p"][:T], r["path_blk"][:T])
        r["h_out"].copy_(r["ibuf"], non_blocking=True)  # one D2H of every accept result
        if E:
            r["h_lp"][:E].copy_(r["lp_ext"][:E], non_blocking=True)
        T8 = T * 8
        r["ev"].record()
        if self.mtp is not None:
            self._mtp_device(ctx, r, r["lay"][5, 0].long(), self._mtp_depth())
        lay.result = (r, T, T8, live)
        self._last_sampled = lay

    def _x0s(self, lay):
        x = lay.n_rows
        out = []
        for _t, _row, k, _pf in lay.tails:
            if k:
                out.append((x, k))
            x += k
        return out

    # ------------------------------------------------------------------ resolve (host)
    def resolve(self) -> dict:
        lay = getattr(self, "_last_sampled", None)
        self._last_sampled = None
        if lay is None:
            return {}
        r, T, T8, live = lay.result
        self.accA *= FORGET
        self.accN *= FORGET
        r["ev"].synchronize()
        out = r["h_out"].numpy()
        TM = self.t_max
        acc, nn, done, p_dev = out[:T], out[TM: TM + T], out[2 * TM: 2 * TM + T], out[3 * TM: 3 * TM + T]
        tok = out[4 * TM: 12 * TM].reshape(TM, 8)[:T]
        path = out[12 * TM: 20 * TM].reshape(TM, 8)[:T]
        nmtp = out[20 * TM: 20 * TM + T]
        lp = r["h_lp"].numpy()
        res = {}
        x_of = {}
        x = lay.n_rows
        for j, (t, row, k, _pf) in enumerate(lay.tails):
            x_of[j] = x
            x += k
        # the rank's critical tail: the longest remaining budget among this forward's tails
        crit_rid, crit_rem = -1, -1
        for _j, t, _now, _k in live:
            rem = t.budget_end - t.plen - len(self.engine.get_request(t.rid).generated_tokens)
            if rem > crit_rem:
                crit_rid, crit_rem = t.rid, rem
        for j, t, now, k in live:
            a = int(acc[j])
            if a <= 0:
                continue
            ext = [float(lp[int(path[j, d])]) for d in range(a - 1)]
            req = self.engine.get_request(t.rid)
            p_row0 = t.plen + len(req.generated_tokens)
            if self.per_request and int(p_dev[j]) != p_row0:
                # CHECKS: the tree of this forward was laid out from a committed length the host has not resolved
                raise RuntimeError(f"[tree_spec] request {t.rid}: tree expanded at P={int(p_dev[j])}, host resolves "
                                   f"P={p_row0}")
            res[t.rid] = (tok[j, :a].tolist(), ext, bool(done[j]), p_row0)
            if k > 0:
                x0 = x_of[j] - lay.n_rows
                built = int(nn[j])
                self.accN[:k] += 1.0  # every allocated row is a trial (an unbuilt one is a row computed for nothing)
                nm = min(int(nmtp[j]), built)
                for d in range(a - 1):
                    ins = int(path[j, d]) - x0  # insertion index of the accepted node at depth d + 1
                    if 0 <= ins < WMAX:
                        self.accA[ins] += 1.0
                    self.stats["mtp_accepted" if 0 <= ins < nm else "other_accepted"] += 1
                self.stats["mtp_nodes"] += nm
                self.stats["other_nodes"] += built - nm
                if nm >= 1:  # depth-1 MTP acceptance split by the base: drafted after a root-only accept or deeper
                    b = "a1" if t.prev_acc <= 1 else "a2"
                    self.stats["mtp_d1_n_" + b] += 1
                    self.stats["mtp_d1_ok_" + b] += int(a >= 2 and int(path[j, 0]) - x0 == 0)
                t.nodes += built
            t.prev_acc = a
            t.iters += 1
            t.tokens += a
            t.accepted_nodes += a - 1
            self.stats["tokens"] += a
            self.stats["accepted_nodes"] += a - 1
            if t.rid == crit_rid:
                self.stats["crit_tokens"] += a
                self.stats["crit_iters"] += 1
            if self.neg:
                self.stats["neg_tokens"] += a - 1
            if self.per_request:
                st = self.req_stats.setdefault(t.rid, [0, 0, 0])
                st[0] += 1
                st[1] += a - 1
                st[2] += int(nn[j])
        self.maybe_log()
        return res

    def _commit_keys(self) -> None:
        """Noise keys are checked at commit (as tail_mtd): positions p0 + 1 .. p0 + a - 1 to the registry."""
        res = getattr(self, "_res", None) or {}
        kc, kr = self._key_check, self._keyreg
        for rid, (toks, _lp, _done, p0) in res.items():
            a = len(toks)
            if a <= 1:
                continue
            if kc is not None and rid in kc.prev and kc.prev[rid] >= 0:
                if kc.prev[rid] != p0:
                    raise RuntimeError(f"[tree_spec] request {rid}: sampler saw position {kc.prev[rid]}, tree {p0}")
                kc.prev[rid] = p0 + a - 1
            if kr is not None:
                kid = int(self._key_ids(np.asarray([rid], dtype=np.int64))[0]) if self._key_ids else rid
                bad = kr.observe(np.full(a - 1, kid, dtype=np.int64), np.arange(p0 + 1, p0 + a, dtype=np.int64))
                if bad:
                    from thundersync_rl.key_check import SamplerKeyError

                    raise SamplerKeyError(f"[tree_spec] committed noise key reused: {bad[:3]}")

    def maybe_log(self) -> None:
        """Worker-side [tree] line every THUNDERSYNC_TREE_LOG_S seconds on rank 0 (paths without a stage line, e.g. the
        stock SPR-SD loop): this rank's per-interval deltas plus the engine's ms per iteration."""
        import time

        iv = self._log_s
        if iv <= 0:
            return
        now = time.perf_counter()
        if self._log_t is None:
            self._log_t, self._log_it = now, self.stats["iters"]
            return
        if now - self._log_t < iv:
            return
        line, self._log_prev = stats_line(self.stats["iters"], [self.summary()], self._log_prev)
        it = self.stats["iters"] - self._log_it
        print(f"{line} ms/iter={1e3 * (now - self._log_t) / max(it, 1):.3f} (rank 0, last {now - self._log_t:.0f} s)",
              flush=True)
        self._log_t, self._log_it = now, self.stats["iters"]

    def pop_request_stats(self, rid: int) -> Optional[list]:
        return self.req_stats.pop(int(rid), None)

    def summary(self) -> dict:
        s = dict(self.stats)
        s["tokens_per_tail_iter"] = round(s["tokens"] / max(s["tail_iters"], 1), 4)
        s["crit_tok_per_iter"] = round(s["crit_tokens"] / max(s["crit_iters"], 1), 4)
        s["rates_head"] = [round(float(x), 4) for x in self.rates()[:8]]
        return s


# ---------------------------------------------------------------------------- installation


STAT_KEYS = ("tail_iters", "tokens", "nodes", "accepted_nodes", "forwards_with_nodes", "registered", "attaches",
             "neg_tokens", "crit_tokens", "crit_iters", "graph_captures", "graph_fallbacks", "mtp_nodes", "mtp_accepted",
             "other_nodes", "other_accepted", "mtp_tails", "pc_tail_shared_blocks", "pc_tail_stale_blocks",
             "plan_no_room", "layout_dropped_no_graph")


def stats_line(step: int, ms: list, prev: Optional[dict]) -> tuple:
    """The per-step [tree] line from every engine rank's summary() (counters are cumulative: the line shows this
    step's deltas): tokens per tail iteration overall and on the rank's critical (slowest) tail, accepted nodes, and
    per source (MTP chain vs references / own history) nodes and accepted nodes. Returns (line, totals)."""
    tot = {k: sum(int(m.get(k, 0)) for m in ms) for k in STAT_KEYS}
    prev = prev or {k: 0 for k in tot}
    d = {k: tot[k] - prev.get(k, 0) for k in tot}
    rates = ms[0].get("rates_head", []) if ms else []
    depths = [sum(int(m["mtp_depths"][i]) for m in ms if "mtp_depths" in m) for i in range(KMTP + 1)]
    line = (f"[tree] step={int(step)} ranks={len(ms)} tail_iters={d['tail_iters']} tokens={d['tokens']} "
            f"tok/iter={d['tokens'] / max(d['tail_iters'], 1):.4f} crit_tok/iter="
            f"{d['crit_tokens'] / max(d['crit_iters'], 1):.4f} crit_iters={d['crit_iters']} nodes={d['nodes']} "
            f"nodes/iter={d['nodes'] / max(d['forwards_with_nodes'], 1):.2f} accepted_nodes={d['accepted_nodes']} "
            f"mtp_nodes={d['mtp_nodes']} mtp_accepted={d['mtp_accepted']} other_nodes={d['other_nodes']} "
            f"other_accepted={d['other_accepted']} mtp_tails={d['mtp_tails']} mtp_depths={depths} "
            f"fwd_with_nodes={d['forwards_with_nodes']} registered={d['registered']} attaches={d['attaches']} "
            f"neg_tokens={d['neg_tokens']} pc_tail_shared_blocks={d['pc_tail_shared_blocks']} "
            f"pc_tail_stale_blocks={d['pc_tail_stale_blocks']} plan_no_room={d['plan_no_room']} "
            f"layout_dropped_no_graph={d['layout_dropped_no_graph']} "
            f"graph_captures={tot['graph_captures']} graph_fallbacks={tot['graph_fallbacks']} rates_rank0={rates}")
    return line, tot


def is_tree_request(sp) -> bool:
    """A request that would become a tree tail (references or a ts_tree spec on its sampling params)."""
    return getattr(sp, MTD_REFS_ATTR, None) is not None or getattr(sp, TREE_ATTR, None) is not None


def install_epoch_map(engine, ts: "TreeSpec") -> None:
    """Record the weight epoch each prefix-cache block is registered under (cleared when the block is re-allocated),
    so a tail can count blocks it uses from an older epoch ([tree] pc_tail_stale_blocks). Host-side, per
    registration call; no device work."""
    alloc = engine.context.kv_block_allocator
    orig_reg, orig_alloc = alloc.register_kv_block_hashes, alloc.allocate_memory_blocks

    def register_kv_block_hashes(block_ids, block_hashes, *a, **kw):
        out = orig_reg(block_ids, block_hashes, *a, **kw)
        ep = int(getattr(engine, "_weight_epoch", 0))
        for b in (out if out is not None else block_ids):
            ts._blk_epoch[int(b)] = ep
        return out

    def allocate_memory_blocks(num_blocks):
        ids = orig_alloc(num_blocks)
        if ids is not None:
            for b in ids.tolist():
                ts._blk_epoch.pop(int(b), None)
        return ids

    alloc.register_kv_block_hashes = register_kv_block_hashes
    alloc.allocate_memory_blocks = allocate_memory_blocks


def install_stale_cache_negative(engine) -> None:
    """NEGATIVE CONTROL tree_stale_cache: tree requests hash their prompt blocks with the PREVIOUS weight epoch's salt,
    so a continuation's prefill matches K / V cached under the old weights (MCore scopes the prefix cache by a weight
    epoch bumped at every resume). The mismatch / train-lp checks must fail; non-vacuous only with prefix caching on
    and old-epoch blocks still cached ([tree] pc_tail_shared_blocks)."""
    from megatron.core.inference.engines import dynamic_engine as de

    orig = engine._add_request

    def _add_request(request, *a, **kw):
        ep = int(getattr(engine, "_weight_epoch", 0))
        if ep > 0 and is_tree_request(request.sampling_params) and getattr(engine.context, "enable_prefix_caching",
                                                                           False):
            request.block_hash_salt = de._weight_scoped_salt(ep - 1, None)
            request.precomputed_block_hashes = []
            request._compute_block_hashes()
        return orig(request, *a, **kw)

    engine._add_request = _add_request


def install(engine, *, W: int, D: int, t_max: int, ref_cap: int, node_cap: int, sample_rows, key_check=None,
            keyreg=None, key_ids=None, scores_rows=None, uniform_rows=None, own_history: bool = False,
            top2_rows=None) -> TreeSpec:
    """Hook a built DynamicInferenceEngine (idempotent per engine), as tail_mtd.install. ``sample_rows(logits,
    key_rids np, positions (device int64), temperature)`` draws rows with the engine's keyed sampler."""
    from megatron.core.inference.batch_dimensions_utils import (
        CUDAGraphBatchDimensionBuilder,
        InferenceBatchDimensions,
    )

    existing = getattr(engine, "_tree_spec", None)
    if existing is not None:
        return existing
    ts = TreeSpec(engine, W, D, t_max, ref_cap, node_cap, sample_rows, scores_rows, uniform_rows,
                  own_history=own_history, top2_rows=top2_rows)
    ts._key_check, ts._keyreg, ts._key_ids = key_check, keyreg, key_ids
    engine._tree_spec = ts
    if getattr(engine.context, "enable_prefix_caching", False):
        install_epoch_map(engine, ts)
    if ts.neg == "tree_stale_cache":
        install_stale_cache_negative(engine)
    buf = engine.context.memory_buffer
    if buf.dim() == 6:  # the pool-size regime of the KV copies (int64 offsets past 2^31 elements; 15b97342)
        top = (buf.shape[1] - 1) * buf.stride(1)
        print(f"[tree_spec] KV pool {tuple(buf.shape)}: top layer offset {top} elements "
              f"({'past' if top >= 2**31 else 'under'} 2^31)", flush=True)
    ctx, ctrl = engine.context, engine.controller

    if not getattr(CUDAGraphBatchDimensionBuilder, "_tail_mtd_wrapped", False):
        from nemo_rl.models.generation.megatron import tail_mtd

        orig_match = CUDAGraphBatchDimensionBuilder.match_graph_config

        def match_graph_config(real_batch_dim, *a, **kw):
            e = tail_mtd._ACTIVE.get("extra", 0)
            if e > 0 and real_batch_dim.prefill_req_count == 0:
                real_batch_dim = InferenceBatchDimensions(
                    token_count=real_batch_dim.token_count + e, prefill_req_count=0,
                    decode_req_count=real_batch_dim.decode_req_count + e)
            return orig_match(real_batch_dim, *a, **kw)

        CUDAGraphBatchDimensionBuilder.match_graph_config = staticmethod(match_graph_config)
        CUDAGraphBatchDimensionBuilder._tail_mtd_wrapped = True

    from nemo_rl.models.generation.megatron import tail_mtd as _mtd

    orig_init = ctx.initialize_attention_state

    def initialize_attention_state(*a, **kw):
        dummy = bool(kw.get("is_expert_parallel_dummy_cuda_graph_step")) or kw.get("construct_graph_dimensions") is not None
        E = ts.plan(ctx, dummy)
        _mtd._ACTIVE["extra"] = E
        try:
            out = orig_init(*a, **kw)
        finally:
            _mtd._ACTIVE["extra"] = 0
        ts.finish_layout(ctx)
        return out

    ctx.initialize_attention_state = initialize_attention_state

    orig_next = ctx._next_decode_block_counts

    def _next_decode_block_counts(request_slice):
        c = orig_next(request_slice)
        tr = ts.tracked_rids()
        if tr.size:
            mask = np.isin(ctx.request_ids[request_slice].numpy(), tr)
            if mask.any():
                c = c.clone()
                c[torch.from_numpy(mask)] = 0
        return c

    ctx._next_decode_block_counts = _next_decode_block_counts

    orig_fwd = ctrl._dynamic_step_forward_logits

    def _dynamic_step_forward_logits(*a, **kw):
        ts.expand(ctx)
        out = orig_fwd(*a, **kw)
        ts.after_forward(ctx)
        return out

    ctrl._dynamic_step_forward_logits = _dynamic_step_forward_logits

    orig_sample = ctrl._dynamic_step_sample_logits

    def _dynamic_step_sample_logits(*a, **kw):
        out = orig_sample(*a, **kw)
        ts.after_sample(ctrl)
        return out

    ctrl._dynamic_step_sample_logits = _dynamic_step_sample_logits

    logits_state = ctrl._async_sched_logits
    orig_clear = logits_state.clear

    def clear():
        ts.pending.clear()
        return orig_clear()

    logits_state.clear = clear

    orig_state = ctrl._build_async_sched_request_state

    def _build_async_sched_request_state(sampled_tokens_cpu, resolved_sequence_lengths):
        ts._res = ts.resolve()
        ts._res_idx = {}
        ids, fin, mask = orig_state(sampled_tokens_cpu, resolved_sequence_lengths)
        if ts._res:
            lo = ctx.paused_request_count
            n = ctx.total_request_count - lo
            cur = ctx.request_ids[lo: lo + n].numpy()
            hit = np.nonzero(np.isin(cur, np.fromiter(ts._res.keys(), dtype=np.int64)))[0]
            ts._res_idx = {int(i): int(cur[i]) for i in hit}
            changed = False
            for i, rid in ts._res_idx.items():
                if ts._res[rid][2] and int(mask[i]) != 0:
                    mask[i] = 0
                    changed = True
            if changed:
                idx = torch.nonzero(mask == 0, as_tuple=True)[0] + lo
                fin = ctx.request_ids[idx].clone()
            ts._commit_keys()
        return ids, fin, mask

    ctrl._build_async_sched_request_state = _build_async_sched_request_state

    orig_mat = ctrl._materialize_async_sched_log_probs

    def _materialize_async_sched_log_probs(transfer, accepted_counts_cpu=None):
        lps, top = orig_mat(transfer, accepted_counts_cpu)
        res = getattr(ts, "_res", None)
        if res and lps is not None:
            for i, rid in getattr(ts, "_res_idx", {}).items():
                rr = res[rid]
                if len(rr[0]) > 1 and i < len(lps):
                    lps[i] = list(lps[i]) + list(rr[1])
        return lps, top

    ctrl._materialize_async_sched_log_probs = _materialize_async_sched_log_probs

    orig_post = engine.post_process_requests

    def post_process_requests(request_ids, finished_request_ids, evict_request_ids, step_time, sample,
                              accepted_tokens, log_probs, *a, **kw):
        res = getattr(ts, "_res", None)
        if res:
            toks = sample.tolist()
            new = []
            for rid, s in zip(request_ids.tolist(), toks):
                rr = res.get(int(rid))
                if rr is not None and len(rr[0]) > 1:
                    if ts.neg not in ("tree_accept_topk", "tree_accept_all", "tree_accept_by_draft_prob"):
                        assert int(rr[0][0]) == int(s), ("[tree_spec] row 0 sample mismatch", rid, rr[0][0], s)
                    new.append(list(rr[0]))
                else:
                    new.append(s)
            sample = _ListSample(new)
        out = orig_post(request_ids, finished_request_ids, evict_request_ids, step_time, sample, accepted_tokens,
                        log_probs, *a, **kw)
        ts.on_post_process(finished_request_ids, evict_request_ids)
        ts._res = None
        return out

    engine.post_process_requests = post_process_requests
    return ts
