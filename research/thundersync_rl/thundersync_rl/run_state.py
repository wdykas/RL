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

Checkpoints are taken only at *quiescent* step boundaries (DESIGN.md section
14s): the step before a boundary drafts nothing for the steps after it, so
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

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


@dataclass
class Tagged:
    """A batch and the stream state that yields it next (its resume point)."""

    batch: Any
    state: dict[str, Any]


def _concat(batches: list[Any]) -> Any:
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    return BatchedDataDict.from_batches(batches)


class PromptStream:
    """Batches of the dataloader over ``max_epochs`` epochs, each tagged with its resume state.

    ``rebatch_size``: re-chunk into batches of this many prompts in order
    (``thundersync.null_bv_prompt_drop`` spares), carrying partial batches
    across dataloader batches and epochs. ``state`` (from a ``Tagged.state``)
    resumes so the next batch yielded is the one that state was taken
    before. The dataloader must be a ``StatefulDataLoader`` (state_dict /
    load_state_dict) for resuming; epochs work with any iterable.
    """

    def __init__(
        self,
        dataloader: Any,
        max_epochs: int,
        rebatch_size: Optional[int] = None,
        state: Optional[dict[str, Any]] = None,
        concat: Callable[[list[Any]], Any] = _concat,
    ) -> None:
        self.dl = dataloader
        self.max_epochs = int(max_epochs)
        self.rebatch_size = rebatch_size
        self.concat = concat
        self.epoch = 0
        self.carry: list[Any] = []
        self._it: Optional[Iterator] = None
        if state is not None:
            self.epoch = int(state["epoch"])
            self.carry = list(state.get("carry") or [])
            if state.get("dl") is not None:
                self.dl.load_state_dict(state["dl"])
            # The loaded state is consumed by the next iter() (StatefulDataLoader).
            self._it = iter(self.dl) if self.epoch < self.max_epochs else None
        self.exhausted = self.epoch >= self.max_epochs

    def _dl_state(self) -> Optional[dict[str, Any]]:
        sd = getattr(self.dl, "state_dict", None)
        return sd() if sd is not None else None

    def state(self) -> dict[str, Any]:
        """Resume point: the next batch this stream would yield."""
        if self._it is None and not self.exhausted:
            self._it = iter(self.dl)
        return {"epoch": self.epoch, "dl": self._dl_state(), "carry": list(self.carry)}

    def _next_raw(self) -> Optional[Any]:
        while self.epoch < self.max_epochs:
            if self._it is None:
                self._it = iter(self.dl)
            b = next(self._it, None)
            if b is not None:
                return b
            self.epoch += 1  # next epoch: a fresh pass (new shuffle order)
            self._it = None
        self.exhausted = True
        return None

    def __iter__(self) -> "PromptStream":
        return self

    def __next__(self) -> Tagged:
        st = self.state()
        if self.rebatch_size is None:
            b = self._next_raw()
            if b is None:
                raise StopIteration
            return Tagged(b, st)
        size = self.rebatch_size
        n = sum(x.size for x in self.carry)
        while n < size:
            b = self._next_raw()
            if b is None:
                raise StopIteration  # a partial tail batch is dropped (drop_last)
            self.carry.append(b)
            n += b.size
        whole = self.concat(self.carry)
        out = whole.slice(0, size)
        self.carry = [whole.slice(size, n)] if n > size else []
        return Tagged(out, st)


# ---------------------------------------------------------------------------
# Step boundaries.
# ---------------------------------------------------------------------------


@dataclass
class Boundaries:
    """Which step ends are quiescent boundaries (checkpoint and/or validation).

    Step ``s`` (0-based) ends at boundary ``s + 1``. ``save_period`` /
    ``val_period`` <= 0 disable that kind. ``requested`` adds one-off
    boundaries (a timeout-triggered save, requested at least one step ahead
    so the step before it does not draft across).
    """

    max_steps: int
    save_period: int = 0
    val_period: int = 0
    val_start_at: int = 0
    requested: set[int] = field(default_factory=set)

    def saves_after(self, step: int) -> bool:
        b = step + 1
        return (self.save_period > 0 and b % self.save_period == 0) or b in self.requested or (
            self.save_period > 0 and b >= self.max_steps
        )

    def validates_after(self, step: int) -> bool:
        b = step + 1
        return self.val_period > 0 and b >= self.val_start_at and b % self.val_period == 0

    def is_boundary(self, step: int) -> bool:
        return self.saves_after(step) or self.validates_after(step)

    def draft_limit(self, step: int) -> int:
        """Drafts made during ``step`` may target steps < this (not across a boundary)."""
        b = step + 1  # boundaries are at b = s + 1 for the step s that ends there
        cands = [self.max_steps]
        if self.save_period > 0:
            cands.append(-(-b // self.save_period) * self.save_period)
        if self.val_period > 0:
            first = max(b, self.val_start_at)
            cands.append(-(-first // self.val_period) * self.val_period)
        cands += [r for r in self.requested if r >= b]
        return min(self.max_steps, min(cands))

    def may_draft_next(self, step: int) -> bool:
        return step + 1 < self.draft_limit(step)


# ---------------------------------------------------------------------------
# Resume: explicit, never silent.
# ---------------------------------------------------------------------------

_STEP_DIR = re.compile(r"step_(\d+)")


def latest_checkpoint(checkpoint_dir: str | os.PathLike) -> Optional[Path]:
    root = Path(checkpoint_dir)
    if not root.is_dir():
        return None
    steps = [
        (int(m.group(1)), p)
        for p in root.iterdir()
        if p.is_dir() and (m := _STEP_DIR.fullmatch(p.name))
    ]
    return max(steps)[1] if steps else None


def resolve_resume(ckpt_cfg: dict[str, Any]) -> Optional[Path]:
    """The checkpoint this run resumes from, or None; raises on an ambiguous state.

    NeMo-RL's ``setup()`` loads the latest ``checkpoint_dir/step_*`` (weights,
    optimizer, dataloader) whether or not checkpointing is enabled. A
    ThunderSync run therefore:

    * with ``checkpointing.enabled: false``: refuses to start if a checkpoint
      exists there (it would be loaded silently while the loop restarts at 0);
    * with checkpointing enabled: resumes from the latest checkpoint, which
      must be a ThunderSync checkpoint (``thundersync_state.pt``); a NeMo-RL
      GRPO checkpoint is refused (its state does not say where ThunderSync's
      boundary was).
    """
    path = latest_checkpoint(ckpt_cfg.get("checkpoint_dir", "results/grpo"))
    if path is None:
        return None
    if not ckpt_cfg.get("enabled", False):
        raise RuntimeError(
            f"found checkpoint {path} but checkpointing.enabled is false: NeMo-RL "
            "setup() would load its weights, optimizer and dataloader while the "
            "ThunderSync loop restarts at step 0. Use an empty checkpointing."
            "checkpoint_dir for a fresh run, or set checkpointing.enabled=true "
            "to resume from it."
        )
    if not (path / STATE_FILE).exists():
        raise RuntimeError(
            f"checkpoint {path} has no {STATE_FILE}: it was not written by the "
            "ThunderSync loop (exact resume needs its boundary state). Use another "
            "checkpointing.checkpoint_dir, or checkpointing.pretrained_checkpoint "
            "for a warm start."
        )
    return path


def check_schedule_unchanged(path: str | os.PathLike, max_num_steps: int, max_num_epochs: int) -> None:
    """Refuse a resume whose run length changed (before setup() loads the checkpoint).

    NeMo-RL sizes Megatron's LR / weight-decay schedule from min(max_num_steps,
    epochs x batches) (``train_iters``), and the checkpointed scheduler must match
    it (Megatron asserts deep inside the load otherwise). A stopped run resumes
    with the same grpo.max_num_steps and grpo.max_num_epochs.
    """
    import yaml

    cfg_path = Path(path) / "config.yaml"
    if not cfg_path.exists():
        return
    grpo = (yaml.safe_load(cfg_path.read_text()) or {}).get("grpo") or {}
    saved = (grpo.get("max_num_steps"), grpo.get("max_num_epochs"))
    if saved != (max_num_steps, max_num_epochs):
        raise RuntimeError(
            f"resume from {path}: grpo.max_num_steps / max_num_epochs were {saved}, now "
            f"{(max_num_steps, max_num_epochs)}; the checkpointed LR / weight-decay schedule "
            "was sized for the saved values (resume with the same run length)"
        )


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
)


def config_fingerprint(cfg: Any) -> dict[str, Any]:
    out = {}
    for sec, key in FINGERPRINT_KEYS:
        part = getattr(cfg, sec, None) if not isinstance(cfg, dict) else cfg.get(sec)
        if part is None:
            out[f"{sec}.{key}"] = None
            continue
        if not isinstance(part, dict):
            part = part.model_dump() if hasattr(part, "model_dump") else vars(part)
        out[f"{sec}.{key}"] = part.get(key)
    return out


def check_fingerprint(saved: dict[str, Any], now: dict[str, Any]) -> None:
    diff = {k: (saved.get(k), now.get(k)) for k in set(saved) | set(now) if saved.get(k) != now.get(k)}
    if diff:
        raise RuntimeError(
            "resume: settings changed since the checkpoint (saved, now): "
            + ", ".join(f"{k}={v}" for k, v in sorted(diff.items()))
        )


def save_state(path: str | os.PathLike, state: dict[str, Any]) -> None:
    state = {"format": STATE_FORMAT, **state}
    tmp = Path(path) / (STATE_FILE + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, Path(path) / STATE_FILE)


def load_state(path: str | os.PathLike) -> dict[str, Any]:
    state = torch.load(Path(path) / STATE_FILE, weights_only=False)
    if state.get("format") != STATE_FORMAT:
        raise RuntimeError(f"{path}/{STATE_FILE}: unknown format {state.get('format')}")
    return state
