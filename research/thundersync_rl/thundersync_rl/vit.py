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
"""Verify-in-train (VIT, ``thundersync.verify_in_train``).

A GRPO group whose G output drafts are all finished at the top of the step is
not verified by verify0. Its rows are trained at once, as one training
micro-batch per group (``micro-batches never mix groups``), and the block-
verification decision of every row is read from that training forward's own
log-probs inside the loss (``decision_hook``):

* every row whole-accepted (eta_g <= b_g at the row's envelope): the group is
  final (its draft rewards are its final rewards, so the advantages used are
  final) and its gradient stays; the rows' later rollouts are served from
  these drafts and are not trained again (``StreamPlanner.mark_pretrained``);
* otherwise the micro-batch's loss is multiplied by 0 (the backward still runs
  on every rank: expert/pipeline lockstep), its normalization counts and
  metrics are taken back, and the group's pool is verified the regular way
  (``SpeculativeGeneration.vit_fallback``) with the SAME per-row random streams
  (``row_seed``) and then trained canonically after its tails.

Exactness. Each row's uniforms come from its own generator seeded by
``row_seed(step base, draft key)``, in both the VIT decision and the fallback
verification, and the whole-acceptance test is bitwise the verifier's
(``whole_weight`` = ``acceptance_weights`` / the grid scan's b at index g).
The group's outcome is therefore always the regular outcome
``decide_pools(grid(eta, p))``: a pass is exactly the case where that outcome
is "the first G drafts, whole"; a fail recomputes it. VIT decides from the
training forward's p; the fallback verifier recomputes p and must agree with it
bitwise (checked in-loop by the mismatch check: the plans' log-probs of VIT rows
are the training forward's).
"""

from __future__ import annotations

import hashlib

import torch


# Data fields a VIT training item carries (per row; [B] or [B, S]).
VIT_FIELDS = ("vit_logq", "vit_start", "vit_len", "vit_seed", "vit_m", "vit_row")


def row_seed(base: int, key: str) -> int:
    """Seed of one draft's verification random stream (63-bit, content-free)."""
    h = hashlib.blake2b(f"{int(base)}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") & (2**63 - 1)


def row_generator(seed: int, device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))




















# -- groups larger than one micro-batch (forward first, then decide) --










class PassPredictor:
    """P(a group of finished drafts is accepted whole), from past steps' counts only.

    Per draft-length bucket (powers of two), whole-acceptance counts of the
    finished drafts verified in the last ``window`` steps (verify0, VIT and
    fallback outcomes). rate = (whole + 1) / (n + 1): 1 without data, so short
    rows start on verify-in-train. A group's pass probability is the product
    over its candidate drafts. Exactness does not depend on it (``vit.py``
    module doc): it only chooses where a group is verified.
    """

    def __init__(self, window: int = 4):
        self.window = max(1, int(window))
        self.steps: list[dict[int, list[int]]] = []
        self.current: dict[int, list[int]] = {}

    @staticmethod
    def bucket(length: int) -> int:
        return max(0, int(length) - 1).bit_length()

    def observe(self, length: int, whole: bool) -> None:
        c = self.current.setdefault(self.bucket(length), [0, 0])
        c[0] += int(bool(whole))
        c[1] += 1

    def end_step(self) -> None:
        self.steps.append(self.current)
        self.current = {}
        del self.steps[: -self.window]

    def rate(self, length: int) -> float:
        b = self.bucket(length)
        w = sum(s.get(b, [0, 0])[0] for s in self.steps)
        n = sum(s.get(b, [0, 0])[1] for s in self.steps)
        return (w + 1) / (n + 1)

