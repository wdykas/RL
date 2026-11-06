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
"""EDV: η-deferred verification (research/nolanes2/DESIGN.md §22.5, §22.9; ``thundersync.edv_threshold``).

⊥-BV accepts a draft whole iff η_g <= b_g (block_verification.py), with η_g the last uniform of the row's first draw
(row seed, F3) and b_g = exp(S_g - max(log M, max_t S_t)), S the cumulative log p/q of the drafted tokens. η_g is
known before any forward. A finished draft with η_g <= e (e = 0.5) is accepted whole unless its log-ratio walk ends
>= |log e| below its running maximum (measured: no such row in 3816 at Super-L22, §22.10). EDV does not forward such
a draft before training: the pool walk takes it as whole, and the training forward of its group, whose log p are the
verifier's bitwise (BI; the train-lp sentinel), takes the decision itself (``decide_in_loss``):

* accepted (b_g >= η_g): the row was trained exactly as a verified whole row;
* a surprise: every rank skips the group's gradient capture (the fp32 buffers are zeroed, the step counters
  restored), the driver verifies the row the regular way, re-walks its pool, launches what the walk gives and
  retrains the group. The committed int64 sum never saw the discarded pass.

Every decision is the same function of (η, p, q, M) as without EDV: pathwise identical outcomes. Only where and
when the decision is computed changes, and that choice is a function of η_g (not draft content, E3).

Learner side (this module, worker process): the loss reads the per-row fields ``FIELDS``; for a deferred row the
micro-batch's ``generation_logprobs`` carry the engine's log q of the drafted tokens (the verified p is not known
yet), so the loss (1) decides from (log p of this forward, log q, η_g, M) and (2) replaces the row's generation
log-probs by this forward's log p, detached: the verified p of an accepted row, bitwise (E7). With
force_on_policy_ratio the loss value and gradient are then those of a verified whole row.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch

# Per-row training fields (float64 [B]): 1 for a deferred row; η_g; the envelope M_j; the row's slot in its group;
# its drafted-token count g (the loss checks it against the row's loss-token count).
FLAG, ETA, ENV, SLOT, LEN = "edv_flag", "edv_eta", "edv_m", "edv_slot", "edv_len"
FIELDS = (FLAG, ETA, ENV, SLOT, LEN)
NEG = "defer_no_decide"  # negative control: the loss accepts every deferred row (no surprise path)

# Decisions taken by the loss in this process since the last ``take``: (slot, b_g, η_g, accepted).
_PENDING: list[tuple[int, float, float, bool]] = []


def whole_weight(lpx: torch.Tensor, lqx: torch.Tensor, m: float) -> torch.Tensor:
    """b_g, the whole-draft acceptance weight, with the verifier's ops (``block_verification.acceptance_weights``,
    ds2's setup): fp64 cumsum of (lpx - lqx) after a 0, minus log M after the root, exp(S - cummax S) at g."""
    s = torch.cat([torch.zeros(1, dtype=torch.float64, device=lpx.device), torch.cumsum((lpx - lqx).double(), 0)])
    if m != 1.0:
        s[1:] -= math.log(m)
    return (s - torch.cummax(s, 0).values).exp()[-1]


def negative() -> bool:
    return os.environ.get("THUNDERSYNC_NL_NEG", "") == NEG


def decide_in_loss(curr_logprobs: torch.Tensor, generation_logprobs: torch.Tensor, token_mask: torch.Tensor,
                   sample_mask: torch.Tensor, data: Any) -> torch.Tensor:
    """Decide every deferred row of a training micro-batch and return the generation log-probs to use.

    ``curr_logprobs`` / ``generation_logprobs`` / ``token_mask``: [B, S-1] (shifted as in the loss); a deferred
    row's drafted tokens are its token_mask positions, in order, and its generation log-probs there are the engine's
    log q (fp32, the verifier's lqx bitwise). Returns ``generation_logprobs`` with each deferred row replaced by
    ``curr_logprobs.detach()``. Records (slot, b_g, η_g, accepted) per live deferred row.
    """
    flag = data[FLAG]
    if not bool((flag > 0).any()):
        return generation_logprobs
    lp = curr_logprobs.detach()
    out = generation_logprobs.clone()
    for b in range(lp.shape[0]):
        if float(flag[b]) <= 0:  # decided whatever its sample mask: its outcome decides the group's outputs
            continue
        pos = torch.nonzero(token_mask[b] > 0).flatten()
        if pos.numel() != int(data[LEN][b]):
            raise RuntimeError(f"[edv] deferred row of slot {int(data[SLOT][b])}: {pos.numel()} loss tokens for a "
                               f"draft of {int(data[LEN][b])} tokens")
        lpx = lp[b, pos].float()
        lqx = generation_logprobs[b, pos].float()
        bg = whole_weight(lpx, lqx, float(data[ENV][b]))
        eta = float(data[ETA][b])
        _PENDING.append((int(data[SLOT][b]), float(bg), eta, bool(eta <= float(bg))))
        out[b] = torch.where(token_mask[b] > 0, lp[b].to(out.dtype), out[b])
    return out


def take() -> list[tuple[int, float, float, bool]]:
    """The decisions recorded since the last take (and clear)."""
    out = list(_PENDING)
    _PENDING.clear()
    return out


def surprise_mask(decisions: list[tuple[int, float, float, bool]]) -> int:
    """Bit s set when slot s's deferred row was not accepted whole (none under the ``defer_no_decide`` negative)."""
    if negative():
        return 0
    m = 0
    for slot, _, _, ok in decisions:
        if not ok:
            m |= 1 << int(slot)
    return m


# -- driver side -------------------------------------------------------------------------------------------------


def eligible(finished: bool, eta: float, e: float) -> bool:
    """Defer a draft: it is complete (EOS or the cap: no bonus or continuation can follow whole acceptance) and
    η_g <= e."""
    return bool(finished) and float(eta) <= float(e)


def deferred_result(d: tuple, envelopes: tuple[float, ...]) -> dict[str, Any]:
    """The verification result a deferred draft is walked with: whole at every envelope (τ = g, an output), the
    engine's log q standing in for the not-yet-known p (replaced in the loss, ``decide_in_loss``). ``d``: the flat
    draft tuple (prompt, seed, tokens, finished, key, segments, logq, hidden)."""
    g = len(d[2])
    lq = [float(x) for x in d[6]]
    return {
        "null_grid": {float(m): (g, 0, 0.0) for m in envelopes},
        "null_log_wstar": 0.0,
        "null_lr_moments": (0, 0.0, 0.0),  # not a calibration sample (no p yet)
        "token_logprobs": lq,
        "draft_logq": lq,
        "p_zero_tokens": 0,
        "edv_deferred": True,
    }


def row_fields(n: int, deferred: dict[int, tuple[float, float, int]]) -> dict[str, torch.Tensor]:
    """Per-row EDV fields of a group of ``n`` rows: ``deferred`` maps slot -> (η_g, M, g)."""
    flag = torch.zeros(n, dtype=torch.float64)
    eta = torch.zeros(n, dtype=torch.float64)
    env = torch.ones(n, dtype=torch.float64)
    slot = torch.arange(n, dtype=torch.float64)
    ln = torch.zeros(n, dtype=torch.float64)
    for s, (e, m, g) in deferred.items():
        flag[s], eta[s], env[s], ln[s] = 1.0, float(e), float(m), float(g)
    return {FLAG: flag, ETA: eta, ENV: env, SLOT: slot, LEN: ln}
