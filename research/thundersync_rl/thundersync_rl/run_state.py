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
"""Run state of the ThunderSync loop: seeds, data stream, step boundaries, resume.

Checkpoints are taken only at *quiescent* step boundaries: the step before a boundary drafts nothing for the steps after it, so
no draft, plan, anchor, ⊥-BV pool, lane pin or staged momentum weight crosses
the boundary, and the run state there is exactly

* the model and optimizer (Megatron distributed checkpoint, written shard by
  shard by every rank; the momentum draft theta_hat is a function of
  (theta, m, v, step, lr) and needs nothing else);
* the data stream position (the dataloader state before the next untrained
  batch, its epoch, and the prompt-drop re-batching carry);
* the step counter and the verification seed counters;
* the lanes' bookkeeping (no cohort may be pinned at a boundary).

The first step after a boundary decodes every rollout fresh at theta_k
(exact); the uninterrupted run does the same at that step, so a resumed run
continues exactly where the saving run would have. Validation runs only at
boundaries too (the engine holds no drafting weights there).
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Callable

import torch

STATE_FILE = "thundersync_state.pt"
STATE_FORMAT = 1
# Seeds feed torch.Generator.manual_seed(seed * 1_000_003 + rank) (< 2^63).
SEED_BITS = 40


def derive_seed(*parts: int | str) -> int:
    """A seed in [1, 2^40) from ``parts`` (stable across processes and runs)."""
    h = hashlib.blake2b(repr(tuple(parts)).encode(), digest_size=8).digest()
    return (int.from_bytes(h, "little") & ((1 << SEED_BITS) - 1)) or 1


class SeedStream:
    """Seeds hash(base seed, stream, step, counter); the counter is checkpointed.

    ``next(stream)`` returns a fresh seed for the current step
    (``step_fn()``). The counter never repeats within a process or across a
    resume (it is saved), the base is ``grpo.seed`` (two runs with different
    seeds draw different verification randomness) and ``stream`` separates
    users (e.g. cohort lanes, verification vs draft keys).
    """

    def __init__(self, base: int, stream: str, step_fn: Callable[[], int]) -> None:
        self.base = int(base)
        self.stream = str(stream)
        self.step_fn = step_fn
        self.counter = 0

    def __iter__(self) -> "SeedStream":
        return self

    def __next__(self) -> int:
        self.counter += 1
        return derive_seed(self.base, self.stream, int(self.step_fn()), self.counter)

    def keyed(self, *identity: Any) -> int:
        """Seed of the call named ``identity`` in the current step.

        A function of (base, stream, step, identity) only: unlike ``next``, it
        does not depend on the order in which concurrent callers arrive. The
        caller must make ``identity`` unique within a step (fresh randomness).
        """
        return derive_seed(self.base, self.stream, int(self.step_fn()), "id", *identity)

    def state_dict(self) -> dict[str, Any]:
        return {"base": self.base, "stream": self.stream, "counter": self.counter}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["base"] != self.base or state["stream"] != self.stream:
            raise ValueError(
                f"seed stream {self.stream!r} (base {self.base}) cannot load the "
                f"state of {state['stream']!r} (base {state['base']}): grpo.seed or "
                "the lane layout changed since the checkpoint"
            )
        self.counter = int(state["counter"])


# ---------------------------------------------------------------------------
# Data: epochs, re-batching, and a resumable position.
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# Step boundaries.
# ---------------------------------------------------------------------------










# ---------------------------------------------------------------------------
# Resume: explicit, never silent.
# ---------------------------------------------------------------------------

_STEP_DIR = re.compile(r"step_(\d+)")








# Settings whose change between save and resume would change the run's law or
# its data order; checked on resume (``config_fingerprint``).
FINGERPRINT_KEYS = (
    ("grpo", "seed"),
    ("grpo", "num_prompts_per_step"),
    ("grpo", "num_generations_per_prompt"),
    ("grpo", "max_num_epochs"),
    ("data", "shuffle"),
    ("checkpointing", "save_period"),
    ("thundersync", "speculate"),
    ("thundersync", "draft_weights"),
    ("thundersync", "cohort_lanes"),
    ("thundersync", "null_bv_prompt_drop"),
    ("thundersync", "null_bv_spare_prompts"),
    # Long runs: the epoch schedule and the rollout lag fix where a resume may start and what it holds.
    ("thundersync", "anchor_epoch"),
    ("thundersync", "anchor_start"),
    ("thundersync", "anchor_weights"),
    ("thundersync", "anchor_momentum_steps"),
    ("thundersync", "draft_lookahead"),
    ("thundersync", "rollout_lag"),
)






def save_state(path: str | os.PathLike, state: dict[str, Any]) -> None:
    state = {"format": STATE_FORMAT, **state}
    tmp = Path(path) / (STATE_FILE + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, Path(path) / STATE_FILE)


