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
"""Engine-side block verification (``thundersync.verify_on=engine``, DESIGN 14ac).

Under batch invariance the engine's prefill of a row is bitwise the training
forward of that row (the in-loop mismatch check already relies on engine decode
== training forward; anchored split-KV / num_splits=1 attention and BI Mamba make
prefill == decode). So an engine holding theta_{k} can compute p of drafted
tokens itself:

* the drafted row (prompt + draft) is submitted as an engine request that
  generates one (discarded) token, tagged ``VERIFY_KEY_ATTR`` (and, with
  sticky routing, the draft's anchors via the engine anchor-apply buffer);
* during its prefill the engine's LM-head input rows of every token are
  copied into a static buffer (captured with the CUDA graphs), and the rows of
  positions plen-1 .. n-1 of tagged requests are kept per key
  (``CaptureStore``): g+1 rows of d head-input elements per draft;
* the decision runs on the engine rank that holds the rows: p rows are the
  engine's own LM head applied to them in chunks of ``chunk_rows`` rows
  (``HeadLogprobRows``; never [g, V] at once), q rows come from the drafting
  version's LM head and the drafter's hidden states, and the lazy block
  verification rule (``engine_q.verify_engine_q`` / ``_grid``) is the same
  code the learner runs.

q-side hidden states (``hidden_transport="recompute"``, lanes only): drafts no
longer log their hidden states per token. When a draft's lazy scan needs q
rows (a rejection, a ⊥ outcome), the lane that drafted it -- still holding the
drafting weights D_k while the cohort's tails run -- prefills the draft again
and hands back its head-input rows (``NeedQ`` two-pass decision: phase 1 for
every row, one batched q prefill for the rows that need it, then those rows are
decided again with the same per-row random stream).

This module holds the parts that need neither Megatron nor Ray.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch

from thundersync_rl.engine_q import GatheredLogprobRows, gathered_log_softmax, no_gather

# Sampling-params attributes of an engine verification request: its key and
# the draft's prompt length (rows of positions plen-1 .. n-1 are kept).
VERIFY_KEY_ATTR = "ts_verify_key"
VERIFY_PLEN_ATTR = "ts_verify_plen"
# Rows per LM-head chunk on the engine (bounds the [rows, V] fp32 temporaries:
# 512 x 131072 x 4 B = 256 MiB at Nemotron vocabularies).
DEFAULT_CHUNK_ROWS = 512


class NeedQ(Exception):
    """Raised by a q-row source whose rows must first be recomputed (two-pass)."""


@dataclass
class HiddenRecompute:
    """Draft tuple index 7 under ``hidden_transport="recompute"``.

    No hidden rows were logged; ``rows`` is the draft's token count (every q
    row can be rebuilt by a prefill of the draft on the lane that drafted it).
    """

    rows: int
    nbytes: int = 0

    def release(self) -> None:
        pass


def row_seed(seed: int, key: str) -> int:
    """Per-row random stream: a fresh call seed and the row's key.

    Independent of which rank decides the row and in which order, so a row can
    be decided twice (``NeedQ``) with the same draws.
    """
    h = hashlib.blake2b(f"{seed}|{key}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") & ((1 << 63) - 1)


class HeadLogprobRows(GatheredLogprobRows):
    """Log-prob rows of a row's head inputs [n, d], the LM head applied on demand.

    ``head(h)`` returns this rank's logits shard [k, V/TP] of head inputs
    ``h`` [k, d] (fp32, exactly as the engine's LM head computes them; at TP=1
    the full row). Rows are computed only where asked (lazy scan) and, for the
    drafted tokens' log-probs, ``chunk_rows`` at a time, so at most
    chunk_rows x V fp32 values exist at once. Under batch-invariant kernels a
    row's logits and log-softmax do not depend on how many rows run together,
    so these are the bits of the training forward's log-probs.
    """

    def __init__(
        self,
        hidden: torch.Tensor,
        head: Callable[[torch.Tensor], torch.Tensor],
        vocab_limit: int,
        chunk_rows: int = DEFAULT_CHUNK_ROWS,
        gather=no_gather,
    ):
        super().__init__(hidden, gather, vocab_limit, chunk_rows)
        self.head = head
        # log p of the drafted tokens, when computed for many rows at once
        # (``batched_token_logprobs``): the same values, fewer GEMM launches.
        self.cached_token_logprobs: Optional[torch.Tensor] = None

    def __call__(self, idx: torch.Tensor) -> torch.Tensor:
        h = self.local[idx.to(self.local.device)]
        return gathered_log_softmax(self.head(h), self.gather, self.vocab_limit)

    def token_logprobs(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.cached_token_logprobs is not None:
            assert self.cached_token_logprobs.numel() == tokens.numel()
            return self.cached_token_logprobs
        k = tokens.numel()
        out = []
        for a in range(0, k, self.chunk_rows):
            b = min(k, a + self.chunk_rows)
            lp = gathered_log_softmax(
                self.head(self.local[a:b]), self.gather, self.vocab_limit
            )
            out.append(lp.gather(-1, tokens[a:b].view(-1, 1).to(lp.device))[:, 0])
            del lp
        if not out:
            return torch.zeros(0, dtype=torch.float32, device=self.local.device)
        return torch.cat(out)


class CaptureStore:
    """Head-input rows of tagged verification requests, by key (engine side).

    ``begin(key, lo, hi, d, dtype, device)`` allocates [hi - lo, d] for
    positions lo .. hi-1; ``put(key, positions, rows)`` writes them
    (positions as numpy, rows on device; a position written twice -- a
    re-prefill after an eviction -- is overwritten with the same bits under
    batch invariance); ``complete(key)`` says every position arrived;
    ``pop(key)`` hands the rows out. Memory: (hi - lo) x d x head-input bytes
    per request in flight on this rank.
    """

    def __init__(self):
        self._rows: dict[str, torch.Tensor] = {}
        self._lo: dict[str, int] = {}
        self._seen: dict[str, np.ndarray] = {}

    def __contains__(self, key: str) -> bool:
        return key in self._rows

    def begin(self, key: str, lo: int, hi: int, d: int, dtype, device) -> None:
        if key in self._rows:
            return
        n = max(0, int(hi) - int(lo))
        self._rows[key] = torch.empty((n, int(d)), dtype=dtype, device=device)
        self._lo[key] = int(lo)
        self._seen[key] = np.zeros(n, dtype=bool)

    def put(self, key: str, positions: np.ndarray, rows: torch.Tensor) -> None:
        buf = self._rows.get(key)
        if buf is None:
            return
        rel = np.asarray(positions, dtype=np.int64) - self._lo[key]
        ok = (rel >= 0) & (rel < buf.shape[0])
        if not ok.all():
            rel = rel[ok]
            rows = rows[torch.from_numpy(np.flatnonzero(ok)).to(rows.device)]
        if rel.size == 0:
            return
        idx = torch.from_numpy(rel)
        if buf.is_cuda:
            idx = idx.pin_memory().to(buf.device, non_blocking=True)
        buf.index_copy_(0, idx, rows.to(buf.dtype))
        self._seen[key][rel] = True

    def complete(self, key: str) -> bool:
        seen = self._seen.get(key)
        return seen is not None and bool(seen.all())

    def keys(self) -> list[str]:
        return list(self._rows)

    def pop(self, key: str) -> Optional[torch.Tensor]:
        self._lo.pop(key, None)
        seen = self._seen.pop(key, None)
        rows = self._rows.pop(key, None)
        if rows is None or seen is None or not seen.all():
            return None
        return rows

    def drop(self, keys) -> None:
        for k in keys:
            self._rows.pop(k, None)
            self._lo.pop(k, None)
            self._seen.pop(k, None)

    def nbytes(self) -> int:
        return sum(r.numel() * r.element_size() for r in self._rows.values())


def capture_selection(
    token_slots: np.ndarray,
    token_pos: np.ndarray,
    slot_kid: np.ndarray,
    kid_lo: np.ndarray,
    kid_hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(token rows, key ids, positions) of one forward's tokens to capture.

    ``token_slots[j]`` / ``token_pos[j]``: context slot and absolute position
    of the step's token j; ``slot_kid``: key id per slot (-1 untagged);
    ``kid_lo`` / ``kid_hi``: kept positions [lo, hi) per key id. Vectorized.
    """
    kid = slot_kid[token_slots]
    tagged = kid >= 0
    k = np.where(tagged, kid, 0)
    keep = tagged & (token_pos >= kid_lo[k]) & (token_pos < kid_hi[k])
    sel = np.flatnonzero(keep)
    return sel, kid[sel], token_pos[sel]


def verify_rows_two_pass(
    items: list[dict[str, Any]],
    decide_one: Callable[[dict[str, Any], Optional[Callable]], dict[str, Any]],
    recompute_q: Callable[[list[dict[str, Any]]], dict[str, Callable]],
) -> tuple[list[tuple[str, dict[str, Any]]], int]:
    """Decide every item; rows whose q source raises ``NeedQ`` are decided again.

    ``decide_one(item, q_rows_or_None)`` runs one row's verification with its
    own random stream (``row_seed``), so a second run reproduces the first
    run's draws exactly; with None it uses the item's own q source (which may
    raise ``NeedQ``). ``recompute_q(items)`` batches the q-row recomputation
    of the rows that needed it and returns {key: q_rows}. Returns
    ([(key, result)], rows recomputed).
    """
    done: list[tuple[str, dict[str, Any]]] = []
    need: list[dict[str, Any]] = []
    for it in items:
        try:
            done.append((it["key"], decide_one(it, None)))
        except NeedQ:
            need.append(it)
    if need:
        qs = recompute_q(need)
        for it in need:
            done.append((it["key"], decide_one(it, qs[it["key"]])))
    return done, len(need)


def batched_token_logprobs(
    hiddens: list[torch.Tensor],
    tokens: list[torch.Tensor],
    head: Callable[[torch.Tensor], torch.Tensor],
    vocab_limit: int,
    chunk_rows: int = 2048,
    gather=no_gather,
) -> list[torch.Tensor]:
    """log p(token) for many rows at once: rows j of ``hiddens[i]`` with ``tokens[i][j]``.

    All rows are concatenated and the head + log-softmax run on chunks of
    ``chunk_rows`` rows (row-invariant under batch invariance, so each value
    equals the per-row computation). Returns one [len(tokens[i])] tensor per row.
    """
    if not hiddens:
        return []
    h = torch.cat([x[: t.numel()] for x, t in zip(hiddens, tokens)])
    toks = torch.cat([t.to(h.device) for t in tokens])
    out = []
    for a in range(0, h.shape[0], chunk_rows):
        b = min(h.shape[0], a + chunk_rows)
        lp = gathered_log_softmax(head(h[a:b]), gather, vocab_limit)
        out.append(lp.gather(-1, toks[a:b].view(-1, 1))[:, 0])
        del lp
    flat = torch.cat(out) if out else torch.zeros(0, device=h.device)
    return list(torch.split(flat, [t.numel() for t in tokens]))


def decide_row(
    p_lp: "HeadLogprobRows",
    draft: torch.Tensor,
    lq: torch.Tensor,
    q_rows: Callable[[torch.Tensor], torch.Tensor],
    seed: int,
    key: str,
    envelopes: Optional[tuple[float, ...]] = None,
    band: Optional[float] = None,
) -> tuple[dict[str, Any], Optional[torch.Tensor], int]:
    """Block-verify one engine-verified row (the per-row decision).

    ``p_lp``: p rows of the row (theta_k, the engine's captured head inputs);
    ``draft`` [g] token ids and ``lq`` [g] fp32 engine log q, on p's device;
    ``q_rows(idx)``: q log-prob rows (may raise ``NeedQ``). Random stream
    ``row_seed(seed, key)``. ``envelopes``: the ⊥-BV grid
    (``verify_engine_q_grid``), else plain / banded BV (``verify_engine_q``).
    Returns (result with the learner method's keys, log p of every drafted
    token or None, residual masses computed).
    """
    from thundersync_rl.block_verification import banded_log_weight
    from thundersync_rl.engine_q import verify_engine_q, verify_engine_q_grid

    gen = torch.Generator(device=lq.device).manual_seed(row_seed(seed, key))
    if envelopes is not None:
        grid, lpx, n_eval = verify_engine_q_grid(p_lp, draft, lq, q_rows, gen, tuple(envelopes))
        tau = max(t for t, _, _ in grid.values())
        lr = torch.cumsum(lpx.double() - lq.double(), 0)
        res = {
            "null_grid": grid,
            "null_log_wstar": float(lr.max()) if lr.numel() else 0.0,
            "token_logprobs": lpx[:tau].tolist(),
            "draft_logq": lq[:tau].tolist(),
        }
        return res, lpx, n_eval
    tau, y, logp, n_eval = verify_engine_q(p_lp, draft, lq, q_rows, gen, 1.0, band)
    res = {
        "accepted": tau,
        "next": y,
        "logprobs": logp.tolist(),
        "draft_logq": lq[:tau].tolist(),
    }
    if band is not None:
        log_w, exited = banded_log_weight(logp[:tau] - lq[:tau], band)
        res["band_log_weight"] = log_w
        res["band_exited"] = exited
    return res, None, n_eval


def decide_whole_rows(
    rows: list[dict[str, Any]],
    p_head: Callable[[torch.Tensor], torch.Tensor],
    vocab_limit: int,
    seed: int,
    chunk_rows: int = 2048,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Batched decision of the rows whose drafts are accepted whole.

    ``rows[i]``: {"key", "hidden" (captured head inputs [g+1, d]), "draft" [g],
    "lpx" [g] (log p of the drafted tokens, ``batched_token_logprobs``),
    "lq" [g] fp32, optional "envelopes" (⊥-BV grid)}; banded rows and empty
    drafts are not taken (returned as not whole). Returns ({key: result} for
    the rows accepted whole at every envelope, [keys of the other rows]); the
    others are decided by ``decide_row`` (which restarts their random stream,
    so nothing drawn here is reused).

    Bitwise ``decide_row`` on the rows it decides: each row draws from its own
    stream ``row_seed(seed, key)`` exactly as the lazy scan does (eta [g+1],
    then u for the grid / the multinomial for plain BV), computes the same
    acceptance weights with the same ops, and the whole-draft test eta_g <= b_g
    is the lazy scan's first test. A whole draft never reads a q row (its
    emission is the bonus token from p_g), so no ``NeedQ``. The bonus rows
    p_g of all rows go through one head GEMM (row-invariant under batch
    invariance, the premise ``batched_token_logprobs`` already uses). Host
    syncs: two per call instead of several per row.
    """
    from thundersync_rl.block_verification import acceptance_weights, check_null_m
    from thundersync_rl.engine_q import no_gather

    slow: list[str] = []
    live: list[dict[str, Any]] = []
    flags: list[torch.Tensor] = []
    for r in rows:
        g = int(r["draft"].numel())
        if r.get("band") is not None or g < 1:
            slow.append(r["key"])
            continue
        lpx, lq = r["lpx"], r["lq"]
        dev = lq.device
        gen = torch.Generator(device=dev).manual_seed(row_seed(seed, r["key"]))
        env = r.get("envelopes")
        eta = torch.rand(g + 1, generator=gen, dtype=torch.float64, device=dev)
        st = {"r": r, "g": g, "gen": gen}
        if env is not None:
            env = tuple(env)
            for M in env:
                check_null_m(M, g)
            # verify_engine_q_grid -> block_verify_stats_lazy_grid, same ops.
            st["u"] = torch.rand(1, generator=gen, dtype=torch.float64, device=dev)
            lqx = lq.to(device=dev, dtype=torch.float32)
            s_cum = torch.cat(
                [
                    torch.zeros(1, dtype=torch.float64, device=dev),
                    torch.cumsum((lpx - lqx).double(), 0),
                ]
            )
            S = s_cum.unsqueeze(0).repeat(len(env), 1)
            for k, M in enumerate(env):
                if M != 1.0:
                    S[k, 1:] -= math.log(M)
            B = (S - torch.cummax(S, 1).values).exp()
            flags.append((eta[g] <= B[:, g]).all().view(1))
            st["env"] = env
            st["wstar"] = torch.cumsum(lpx.double() - lq.double(), 0).max().view(1)
        else:
            # verify_engine_q -> block_verify_stats_lazy (band None, M = 1).
            b = acceptance_weights(lpx - lq.to(device=dev, dtype=torch.float32), 1.0)
            flags.append((eta[g] <= b[g]).view(1))
            st["env"] = None
        live.append(st)
    if not live:
        return {}, slow
    whole_h = torch.cat([f.to(live[0]["r"]["lq"].device) for f in flags]).cpu().tolist()  # sync 1
    take = [st for st, w in zip(live, whole_h) if w]
    slow += [st["r"]["key"] for st, w in zip(live, whole_h) if not w]
    if not take:
        return {}, slow
    dev = take[0]["r"]["lq"].device
    H = torch.stack([st["r"]["hidden"][st["g"]] for st in take])
    ys, lys = [], []
    a = 0
    for c in range(0, H.shape[0], chunk_rows):
        P = gathered_log_softmax(p_head(H[c : c + chunk_rows]), no_gather, vocab_limit)
        for i in range(P.shape[0]):
            st = take[a + i]
            p_row = P[i]
            zero = torch.zeros((), dtype=torch.float32, device=p_row.device)  # lse_p[g]
            dist = (p_row.float() - zero).double().exp()  # _probs; bonus: dist = p
            if st["env"] is not None:
                # sample_by_inversion(dist, u)
                cdf = torch.cumsum(dist.double(), 0)
                y = torch.searchsorted(cdf, st["u"] * cdf[-1], right=True)
                y = y.clamp(max=dist.numel() - 1)
            else:
                # _emit: multinomial with the row's stream after eta
                y = torch.multinomial((dist / dist.sum()).float(), 1, generator=st["gen"])
            ys.append(y.view(1))
            lys.append((p_row[y].float() - zero).view(1))
        a += P.shape[0]
        del P
    y_h = torch.cat(ys).cpu().tolist()  # sync 2 (the transfers below follow it)
    ly_h = torch.cat(lys).cpu().tolist()
    lpx_h = torch.cat([st["r"]["lpx"] for st in take]).cpu()
    lq_h = torch.cat([st["r"]["lq"].to(dev) for st in take]).cpu()
    wl = [st for st in take if st["env"] is not None]
    w_h = torch.cat([st["wstar"].to(dev) for st in wl]).cpu().tolist() if wl else []
    out: dict[str, dict[str, Any]] = {}
    a = 0
    wi = 0
    for st, y, ly in zip(take, y_h, ly_h):
        g = st["g"]
        lpx_l = lpx_h[a : a + g].tolist()
        lq_l = lq_h[a : a + g].tolist()
        a += g
        if st["env"] is not None:
            res = {
                "null_grid": {M: (g, int(y), ly) for M in st["env"]},
                "null_log_wstar": w_h[wi],
                "token_logprobs": lpx_l,
                "draft_logq": lq_l,
            }
            wi += 1
        else:
            res = {
                "accepted": g,
                "next": int(y),
                "logprobs": lpx_l + [ly],
                "draft_logq": lq_l,
            }
        out[st["r"]["key"]] = res
    return out, slow
