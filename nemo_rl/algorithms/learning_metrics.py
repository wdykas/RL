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
"""Learning-vs-sharpening diagnostics for RL post-training.

These metrics compare the current policy with the initial (pre-RL) model on the same teacher-forced tokens.
They need only full next-token logits, so they are architecture-agnostic.

Headline metric, ``novel_mass``::

    novel_mass = E_tokens[ sum_{y not in N_p(pi_0)} pi_t(y) ]

``N_p(pi_0)`` is the initial model's top-p nucleus: the smallest token set holding ``p`` of its probability.

* Sharpening, or selecting among behaviours the initial model already ranked highly, keeps ``pi_t`` inside
  that nucleus. ``novel_mass`` then falls below its initial value ``novel_mass_init``.
* Moving probability onto continuations the initial model considered implausible makes it rise.

Supporting metrics:

* ``temperature_explained``: share of ``KL(pi_t || pi_0)`` explained by one global temperature.
  1.0 means pure temperature sharpening.
* ``top1_agree``: fraction of positions where the argmax is unchanged.
* ``fork_novel_mass``: ``novel_mass`` on the 5% of positions where the policy moved most.
* ``kl_to_init``, ``entropy`` and ``entropy_init``.
"""

from __future__ import annotations

import torch

TEMPERATURE_GRID = [round(0.3 + 0.05 * i, 2) for i in range(29)]  # 0.30 .. 1.70


def nucleus_mask(logp_init: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """Boolean mask [N, V] of the tokens inside the initial model's top-p nucleus."""
    probs, idx = logp_init.exp().sort(-1, descending=True)
    keep_sorted = (probs.cumsum(-1) - probs) < p  # include the token that crosses p
    return torch.zeros_like(keep_sorted).scatter(-1, idx, keep_sorted)


@torch.no_grad()
def learning_metrics(
    logits_now: torch.Tensor, logits_init: torch.Tensor, p: float = 0.9, chunk: int = 4096
) -> dict[str, float]:
    """Compare current-policy logits with initial-model logits on the same positions.

    Args:
        logits_now: [N, V] logits of the current policy.
        logits_init: [N, V] logits of the initial model on the same positions.
        p: nucleus mass that defines the initial model's support.
        chunk: number of positions processed at once (bounds peak memory).
    """
    assert logits_now.shape == logits_init.shape and logits_now.dim() == 2
    tot = {"kl": 0.0, "novel": 0.0, "novel0": 0.0, "top1": 0.0, "h": 0.0, "h0": 0.0}
    kl_t = torch.zeros(len(TEMPERATURE_GRID), dtype=torch.float64)
    per_kl, per_novel = [], []
    for s in range(0, logits_now.shape[0], chunk):
        z_now, z_init = logits_now[s : s + chunk].float(), logits_init[s : s + chunk].float()
        lp_now, lp_init = torch.log_softmax(z_now, -1), torch.log_softmax(z_init, -1)
        p_now = lp_now.exp()
        outside = ~nucleus_mask(lp_init, p)
        novel = (p_now * outside).sum(-1)
        kl = (p_now * (lp_now - lp_init)).sum(-1)
        tot["kl"] += kl.sum().item()
        tot["novel"] += novel.sum().item()
        tot["novel0"] += (lp_init.exp() * outside).sum(-1).sum().item()
        tot["top1"] += (lp_now.argmax(-1) == lp_init.argmax(-1)).float().sum().item()
        tot["h"] += -(p_now * lp_now).sum(-1).sum().item()
        tot["h0"] += -(lp_init.exp() * lp_init).sum(-1).sum().item()
        for j, t in enumerate(TEMPERATURE_GRID):
            kl_t[j] += (p_now * (lp_now - torch.log_softmax(z_init / t, -1))).sum().item()
        per_kl.append(kl)
        per_novel.append(novel)
    n = max(logits_now.shape[0], 1)
    kl_all, novel_all = torch.cat(per_kl), torch.cat(per_novel)
    fork = kl_all >= kl_all.quantile(0.95)
    j = int(kl_t.argmin())
    return {
        "novel_mass": tot["novel"] / n,
        "novel_mass_init": tot["novel0"] / n,
        "fork_novel_mass": novel_all[fork].mean().item(),
        "kl_to_init": tot["kl"] / n,
        "temperature_explained": 1.0 - kl_t[j].item() / max(tot["kl"], 1e-12),
        "best_temperature": TEMPERATURE_GRID[j],
        "top1_agree": tot["top1"] / n,
        "entropy": tot["h"] / n,
        "entropy_init": tot["h0"] / n,
    }
