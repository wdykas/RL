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
"""GRPO-Z: the GRPO advantage with the leave-one-out baseline taken from the siblings' DRAFT rewards
(docs/colo/THEORY.md §3.4, Lemma Z; design DX).

Setting. Row i of a group is an exact theta_n sample Y_i (accepted draft, or draft prefix + theta_n continuation). Each
slot also has a draft X_i (sampled from the drafting law q before theta_n exists) whose reward R(X_i) is known as soon
as the draft finished. The rows' randomness is independent across slots given (F_n, theta_n): drafts use independent
keys, continuations fresh keys, and theta_n does not depend on batch n's drafts (in-order admission).

    A_i = (R(Y_i) - b_i) / s_i,   b_i = mean_{j != i} R(X_j),

with s_i one of
- ``loo_std``: s_i = std_{j != i} R(X_j) (stock GRPO's leave-one-out std, Bessel-corrected, on the draft rewards; a
  group whose sibling draft rewards are all equal is not normalized, as stock). E[A_i grad log p(Y_i)] =
  E[1 / s_i] grad J_x: the exact per-prompt direction with a prompt weight c_x = E[1 / s_i] that depends on the
  drafting law (U' in FIRSTPRINCIPLES_D §1). Stock on-policy LOO-std has the same form with weight E[1 / std(Y_-i)].
- ``loo_c``: s_i = s_n, one scale per step that is a function of the PREVIOUS step's final rewards only (F_n
  measurable): E[A_i grad log p(Y_i)] = grad J_x / s_n, i.e. exactly the on-policy LOO-c gradient (U, exact).

Proof (both): b_i and s_i are functions of X_{-i} (and F_n) only, independent of Y_i given (F_n, theta_n); so
E[(R_i - b_i) / s_i * h_i] = E[1 / s_i] E[R_i h_i] - E[b_i / s_i] E[h_i] with h_i = grad log p(Y_i), E[h_i] = 0 (score
identity) and E[R_i h_i] = grad J_x.

When every row is an accepted draft (Y = X), ``loo_std`` is bitwise stock GRPO (``GRPOAdvantageEstimator`` with
use_leave_one_out_baseline and normalize_rewards): the same function on the same reward vector.

What breaks it (tests): the row's OWN draft reward in its baseline (b depends on X_i, which equals Y_i on acceptance),
or a scale computed from the current batch including the row.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import torch

MODES = ("off", "loo_std", "loo_c")
Mode = Literal["off", "loo_std", "loo_c"]
EPS = 1e-6  # stock GRPOAdvantageEstimator's normalization epsilon


def sibling_baseline_and_scale(
    group_ids: torch.Tensor, draft_rewards: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per row: (mean, Bessel std, trivial flag) of the OTHER rows' draft rewards in its group.

    Delegates to stock ``calculate_baseline_and_std_per_prompt`` with leave-one-out on the draft-reward vector: its
    row-i outputs read only rows j != i (baseline mask 1 - I, comparison mask excludes i), so they are functions of
    X_{-i} alone, and with draft == final rewards they are bitwise stock's.
    """
    from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt

    gid = group_ids.reshape(-1, 1) if group_ids.dim() == 1 else group_ids
    return calculate_baseline_and_std_per_prompt(
        gid, draft_rewards, torch.ones_like(draft_rewards), leave_one_out_baseline=True
    )


def advantages(
    group_ids: torch.Tensor,
    final_rewards: torch.Tensor,
    draft_rewards: torch.Tensor,
    mode: Mode,
    step_scale: Optional[float] = None,
) -> torch.Tensor:
    """GRPO-Z advantages [B] (one scalar per row; the caller expands to tokens).

    group_ids [B] (or [B, 1]): group equality keys; final_rewards [B]: R(Y_i); draft_rewards [B]: R(X_i) of the slot's
    draft (for an accepted whole row, X_i = Y_i). ``step_scale`` (loo_c): s_n > 0 from the previous step's rewards.
    """
    if mode not in ("loo_std", "loo_c"):
        raise ValueError(f"grpo_z mode {mode!r} (expected loo_std / loo_c)")
    if final_rewards.shape != draft_rewards.shape:
        raise ValueError(f"final {tuple(final_rewards.shape)} vs draft {tuple(draft_rewards.shape)} rewards")
    # float32, as the advantage stage reads rewards (stock's LOO helper is float32-only)
    final_rewards, draft_rewards = final_rewards.float(), draft_rewards.float()
    baseline, std, trivial = sibling_baseline_and_scale(group_ids, draft_rewards)
    adv = (final_rewards - baseline).unsqueeze(-1)
    if mode == "loo_std":
        norm = (std > 0) & ~trivial
        adv[norm] = adv[norm] / (std.unsqueeze(-1)[norm] + EPS)
    else:
        if step_scale is None or not float(step_scale) > 0.0:
            raise ValueError(f"grpo_z loo_c needs a positive step scale (got {step_scale!r})")
        adv = adv / float(step_scale)
    return adv.squeeze(-1)


def group_row_advantages(
    final_rewards: list[float],
    sources: list[Optional[int]],
    draft_rewards: list[float],
    mode: Mode,
    step_scale: Optional[float] = None,
) -> list[float]:
    """GRPO-Z advantages of one group's output rows, for any pool shape (nolanes: N >= G drafts per group).

    ``final_rewards[k]``: R(Y_k) of output slot k; ``sources[k]``: the index of the draft slot k came from (its draft
    tokens were accepted whole or continued after tau), or None for a fresh slot (no draft); ``draft_rewards[j]``:
    R(X_j) of every draft j of the group, in drafting order. Row k's baseline / scale are those of stock LOO at position
    s = sources[k] of the vector v = draft_rewards with v[s] := R(Y_k) (a fresh slot appends R(Y_k)): the mean / std of
    every OTHER draft's reward. Those drafts are independent of Y_k: under ⊥-BV (null_bv.py) a draft's non-⊥ output
    is p-distributed and independent of the past with success probability exactly 1/M_j, M_j a function of counts
    only, so which draft fills slot k does not depend on any draft's content, and Y_k is a function of its own source
    draft and fresh keys. A slot filled from another group's draft of the same prompt (a shared pool) or decoded fresh
    has no source here (all of the group's drafts form its baseline).
    With N = G and sources = 0..G-1 this is ``advantages`` on the group (bitwise; test).
    """
    out: list[float] = []
    gid = torch.zeros(len(draft_rewards) + 1, dtype=torch.long)
    for y, s in zip(final_rewards, sources):
        v = torch.tensor(list(draft_rewards), dtype=torch.float32)
        if s is None:
            v = torch.cat([v, torch.tensor([float(y)], dtype=torch.float32)])
            pos, g = len(draft_rewards), gid
        else:
            if not 0 <= int(s) < len(draft_rewards):
                raise ValueError(f"grpo_z: source draft {s} outside the group's {len(draft_rewards)} drafts")
            v[int(s)] = float(y)
            pos, g = int(s), gid[:-1]
        out.append(float(advantages(g, v, v, mode, step_scale)[pos]))
    return out


def next_step_scale(group_ids: torch.Tensor, final_rewards: torch.Tensor, floor: float = 1e-3) -> float:
    """s_{n+1} for loo_c: the mean over groups of the within-group std of step n's FINAL rewards (Bessel), floored.

    A function of step n's outputs only, hence F_{n+1}-measurable for step n+1 (independent of every row of step n+1).
    """
    gid = group_ids.reshape(-1)
    stds = []
    for g in torch.unique(gid):
        r = final_rewards[gid == g].double()
        if r.numel() > 1:
            stds.append(float(r.std(unbiased=True)))
    s = sum(stds) / len(stds) if stds else 1.0
    return max(s, float(floor))


def config_problems(config: Any) -> list[str]:
    """Reasons a ``thundersync.grpo_z`` setting is invalid (empty: valid). Pure function of the config dict."""

    def _get(d, path, default=None):
        cur = d
        for part in path.split("."):
            if cur is None:
                return default
            cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
        return default if cur is None else cur

    p: list[str] = []
    mode = _get(config, "thundersync.grpo_z", "off")
    if mode not in MODES:
        return [f"thundersync.grpo_z must be one of {MODES} (got {mode!r})"]
    if mode == "off":
        return p
    est = _get(config, "grpo.adv_estimator") or {}
    if _get(est, "name", "grpo") != "grpo":
        p.append("thundersync.grpo_z needs grpo.adv_estimator.name = grpo (it replaces the GRPO baseline)")
    if not _get(est, "use_leave_one_out_baseline", True):
        p.append("thundersync.grpo_z needs grpo.adv_estimator.use_leave_one_out_baseline (the baseline is LOO by "
                 "construction: a row's own draft reward never enters its baseline)")
    norm = bool(_get(est, "normalize_rewards", True))
    if mode == "loo_std" and not norm:
        p.append("thundersync.grpo_z = loo_std needs grpo.adv_estimator.normalize_rewards (use loo_c otherwise)")
    if int(_get(config, "grpo.num_generations_per_prompt", 0) or 0) < 2:
        p.append("thundersync.grpo_z needs grpo.num_generations_per_prompt >= 2 (siblings)")
    if _get(config, "grpo.use_dynamic_sampling", False):
        p.append("thundersync.grpo_z refuses dynamic sampling (it filters groups on the final rewards: selection on "
                 "content)")
    if _get(config, "grpo.reward_shaping.enabled", False):
        p.append("thundersync.grpo_z: reward shaping must be off (draft and final rewards must be the same function)")
    if _get(config, "thundersync.mode") != "nolanes":
        p.append("thundersync.grpo_z needs thundersync.mode = nolanes (verified rows with logged draft rewards)")
    return p
