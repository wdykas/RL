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
"""Research dump (acceptance decay study): per-token divergence of the verified drafts.

``THUNDERSYNC_ACC_DUMP=<dir>`` (off by default): for every ``THUNDERSYNC_ACC_DUMP_EVERY``-th verified ⊥-BV row
(by key hash), on the verifying learner rank, with the verifier's own p rows and the q rows rebuilt from the logged
hidden states (bitwise the engine's under batch invariance), write per draft position:

    lp_x, lq_x        log p / log q of the drafted token (recomputed from the full rows)
    H_p, H_q          entropies
    pmax_p, pmax_q    top-1 probabilities;  same_argmax
    tv                total variation TV(p, q)
    kl_qp             KL(q || p)
    m1_q, var_q       E_q[log p/q], Var_q[log p/q] (the conditional mean / variance of the logged walk increment)
    agree             P(Gumbel-max with shared noise picks the same token under p and q), on the top-K support
    miss              p+q mass outside the top-K support (/2)

Read-only: no RNG, nothing returned to the verifier. Cost: one head GEMM per drafted position of a dumped row plus
O(V) elementwise work and a top-K sort per position (chunked; bounded temporaries).
"""

from __future__ import annotations

import os
import zlib
from typing import Any

import torch

DIR = os.environ.get("THUNDERSYNC_ACC_DUMP") or None
EVERY = int(os.environ.get("THUNDERSYNC_ACC_DUMP_EVERY", "8"))
TOPK = int(os.environ.get("THUNDERSYNC_ACC_DUMP_TOPK", "4096"))
CHUNK = int(os.environ.get("THUNDERSYNC_ACC_DUMP_CHUNK", "128"))
NAMES = (
    "lp_x", "lq_x", "H_p", "H_q", "pmax_p", "pmax_q", "same_argmax", "tv", "kl_qp", "m1_q", "var_q", "agree", "miss",
)


def picked(key: Any) -> bool:
    return DIR is not None and zlib.crc32(str(key).encode()) % max(EVERY, 1) == 0


def _entropy(lp: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return -(p * torch.where(p > 0, lp, torch.zeros_like(lp))).sum(-1)


@torch.no_grad()
def chunk_stats(lp: torch.Tensor, lq: torch.Tensor, tok: torch.Tensor, topk: int = TOPK) -> torch.Tensor:
    """[k, len(NAMES)] statistics of k positions from full log-prob rows lp, lq [k, V]."""
    lp, lq = lp.double(), lq.double()
    p, q = lp.exp(), lq.exp()
    ok = (p > 0) & (q > 0)
    d = torch.where(ok, lp - lq, torch.zeros_like(lp))
    m1 = (q * d).sum(-1)
    var = (q * d * d).sum(-1) - m1 * m1
    kl_qp = -m1  # sum_v q (log q - log p) on the joint support
    tv = 0.5 * (p - q).abs().sum(-1)
    # Shared-noise Gumbel coupling: P(both argmax = v) = 1 / sum_u max(p_u/p_v, q_u/q_v). Sorting by r = log p/q,
    # the u with r_u >= r_v contribute p_u/p_v and the rest q_u/q_v.
    k = min(topk, p.shape[-1])
    top = (p + q).topk(k, dim=-1).indices
    ps, qs = p.gather(-1, top), q.gather(-1, top)
    r = ps.clamp_min(1e-300).log() - qs.clamp_min(1e-300).log()
    order = r.argsort(-1, descending=True)
    ps, qs = ps.gather(-1, order), qs.gather(-1, order)
    pc = ps.cumsum(-1)
    qt = qs.flip(-1).cumsum(-1).flip(-1) - qs
    inv = torch.zeros_like(ps)
    both = (ps > 0) & (qs > 0)
    inv[both] = 1.0 / (pc[both] / ps[both] + qt[both] / qs[both])
    agree = inv.sum(-1)
    miss = 1.0 - 0.5 * (ps.sum(-1) + qs.sum(-1))
    tcol = tok.view(-1, 1).to(lp.device)
    cols = [
        lp.gather(-1, tcol)[:, 0], lq.gather(-1, tcol)[:, 0], _entropy(lp, p), _entropy(lq, q),
        p.max(-1).values, q.max(-1).values, (lp.argmax(-1) == lq.argmax(-1)).double(),
        tv, kl_qp, m1, var, agree, miss,
    ]
    return torch.stack(cols, -1).float()


@torch.no_grad()
def dump_row(rank: int, version: Any, key: Any, segments: Any, p_lp: Any, q_rows: Any, draft: torch.Tensor,
             lq: torch.Tensor, lpx: torch.Tensor, grid: Any = None, row: Any = None, prompt_len: Any = None) -> None:
    g = int(draft.numel())
    dev = draft.device
    out = []
    for a in range(0, g, CHUNK):
        b = min(g, a + CHUNK)
        idx = torch.arange(a, b, device=dev)
        lpr = p_lp[a:b] if isinstance(p_lp, torch.Tensor) else p_lp(idx)
        lqr = q_rows(idx).to(lpr.device)
        out.append(chunk_stats(lpr, lqr, draft[a:b]).cpu())
        del lpr, lqr
    stats = torch.cat(out) if out else torch.zeros(0, len(NAMES))
    os.makedirs(DIR, exist_ok=True)
    name = f"v{version}_r{rank}_{zlib.crc32(str(key).encode()):08x}.pt"
    tmp = os.path.join(DIR, name + ".tmp")
    torch.save(
        dict(rank=rank, version=version, key=str(key), segments=[tuple(int(x) for x in s) for s in (segments or ())],
             g=g, draft=draft.int().cpu(), row=None if row is None else row.int().cpu(), prompt_len=None if prompt_len is None else int(prompt_len), lq_logged=lq.float().cpu(), lp_verify=lpx.float().cpu(),
             grid=repr(grid) if grid is not None else None,
             names=NAMES, stats=stats),
        tmp,
    )
    os.replace(tmp, os.path.join(DIR, name))
