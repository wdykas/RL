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
"""Verify-in-train (VIT, ``thundersync.verify_in_train``; DESIGN 14r).

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

import asyncio
import hashlib
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

# Data fields a VIT training item carries (per row; [B] or [B, S]).
VIT_FIELDS = ("vit_logq", "vit_start", "vit_len", "vit_seed", "vit_m", "vit_row")


def row_seed(base: int, key: str) -> int:
    """Seed of one draft's verification random stream (63-bit, content-free)."""
    h = hashlib.blake2b(f"{int(base)}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") & (2**63 - 1)


def row_generator(seed: int, device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


def eta_last(seed: int, g: int, device) -> torch.Tensor:
    """eta_g: the last of the verifier's first draw ``rand(g+1)`` (fp64) from the row's stream."""
    gen = row_generator(seed, device)
    return torch.rand(g + 1, generator=gen, dtype=torch.float64, device=device)[g]


def whole_weight(lpx: torch.Tensor, lqx: torch.Tensor, m: float) -> torch.Tensor:
    """b_g, the whole-draft acceptance weight, bitwise as the verifier computes it.

    ``lpx``/``lqx``: fp32 [g] log p / log q of the drafted tokens. Same ops as
    ``block_verification.acceptance_weights`` (plain) and the envelope rows of
    ``block_verify_stats_lazy_grid`` (⊥-BV): fp64 cumsum of (lpx - lqx), minus
    log M after the root, exp(S - cummax S).
    """
    s = torch.cat(
        [
            torch.zeros(1, dtype=torch.float64, device=lpx.device),
            torch.cumsum((lpx - lqx).double(), 0),
        ]
    )
    if m != 1.0:
        s[1:] -= math.log(m)
    return (s - torch.cummax(s, 0).values).exp()[-1]


def decide_rows(next_token_logprobs: torch.Tensor, data) -> tuple[bool, list[tuple[int, bool, list[float]]]]:
    """Whole-acceptance of every live row of one VIT micro-batch.

    Returns (all whole, [(row index in its group, whole, log p of the drafted
    tokens)]). Rows with sample_mask 0 (padding) are skipped.
    """
    lp = next_token_logprobs.detach()
    out = []
    ok = True
    sm = data["sample_mask"]
    for b in range(lp.shape[0]):
        if float(sm[b]) == 0.0:
            continue
        s, g = int(data["vit_start"][b]), int(data["vit_len"][b])
        lpx = lp[b, s - 1 : s - 1 + g].float()
        lqx = data["vit_logq"][b, s : s + g].to(device=lpx.device, dtype=torch.float32)
        bg = whole_weight(lpx, lqx, float(data["vit_m"][b]))
        eta = data["vit_eta"][b]
        whole = bool(eta.cpu() <= bg.cpu())
        ok &= whole
        out.append((int(data["vit_row"][b]), whole, lpx.tolist()))
    return ok, out


@contextmanager
def decision_hook(results: list):
    """Patch ClippedPGLossFn.__call__: decide each VIT micro-batch, zero its loss on a fail.

    One ``results`` entry per VIT micro-batch, in call order: {"pass", "rows"}.
    Non-VIT micro-batches (no ``vit_seed`` field) pass through untouched.
    """
    from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossFn

    orig = ClippedPGLossFn.__call__

    def patched(self, next_token_logprobs=None, data=None, *args, **kwargs):
        loss, metrics = orig(self, next_token_logprobs, data, *args, **kwargs)
        if data is not None and "vit_seed" in data and next_token_logprobs is not None:
            ok, rows = decide_rows(next_token_logprobs, data)
            results.append({"pass": ok, "rows": rows})
            if not ok:
                # The backward still runs (lockstep); every gradient it adds is exactly 0.
                loss = loss * 0.0
        return loss, metrics

    ClippedPGLossFn.__call__ = patched
    try:
        yield
    finally:
        ClippedPGLossFn.__call__ = orig


def item_counts(data) -> tuple[torch.Tensor, torch.Tensor]:
    """(valid sequences, valid tokens) of one item, as ``_prepare_train_piece`` counts them."""
    sample_mask = data["sample_mask"]
    seqs = torch.sum(sample_mask).to(torch.float64)
    toks = torch.sum(data["token_mask"][:, 1:] * sample_mask.unsqueeze(-1)).to(torch.float64)
    return seqs, toks


@dataclass
class VitGroup:
    """One prompt's pool taken by VIT at the top of a step."""

    key: tuple  # prompt key
    drafts: list  # the pool's drafts in drafting order (first G are the candidate outputs)
    ms: list  # envelope of each of the first G drafts if all earlier ones are outputs
    seeds: list  # row_seed of every pool draft
    placeholders: list = field(default_factory=list)  # the G plans rollouts wait on
    passed: Optional[bool] = None


class StubGeneration:
    """A generation backend that returns fixed token rows (VIT reward precompute).

    ``run_sample_multi_turn_rollout`` with this backend steps the environment on
    exactly the text the real rollout will be served (the draft), so the
    reward is the real rollout's reward (deterministic environment; checked
    when the real rollouts arrive, ``StreamPlanner.mark_pretrained``).
    """

    def __init__(self, cfg: Any, tokens: list[int], logprobs: list[float]):
        self.cfg = cfg
        self.tokens = list(tokens)
        self.logprobs = [float(x) for x in logprobs]

    async def generate_async(self, data: BatchedDataDict, greedy: bool = False):
        plen = int(data["input_lengths"][0])
        prompt = data["input_ids"][0, :plen]
        n = plen + len(self.tokens)
        out = BatchedDataDict(
            {
                "output_ids": torch.cat([prompt, torch.tensor(self.tokens, dtype=torch.long)]).view(1, -1),
                "logprobs": torch.cat(
                    [torch.zeros(plen), torch.tensor(self.logprobs, dtype=torch.float)]
                ).view(1, -1),
                "generation_lengths": torch.tensor([len(self.tokens)]),
                "unpadded_sequence_lengths": torch.tensor([n]),
                "gen_leader_worker_idx": [0],
            }
        )
        await asyncio.sleep(0)
        yield 0, out


def assistant_span(message_log: list[dict]) -> tuple[int, int]:
    """(start, length) of the single assistant message's tokens in the flattened row."""
    off, span = 0, None
    for m in message_log:
        n = int(m["token_ids"].numel())
        if m.get("role") == "assistant":
            if span is not None:
                raise RuntimeError("verify_in_train: more than one assistant turn")
            span = (off, n)
        off += n
    if span is None:
        raise RuntimeError("verify_in_train: no assistant turn")
    return span


def add_vit_fields(data, rows: list[dict]) -> None:
    """Attach the VIT per-row fields to a training batch built by ``build_chunk_data``.

    ``rows[b]``: {"start", "len", "logq" (list), "seed", "m", "row"} for row b of
    ``data`` (padding rows beyond len(rows) copy row 0 and are masked).
    """
    n, s = data["input_ids"].shape
    logq = torch.zeros(n, s, dtype=torch.float32)
    start = torch.zeros(n, dtype=torch.long)
    length = torch.zeros(n, dtype=torch.long)
    seed = torch.zeros(n, dtype=torch.long)
    m = torch.ones(n, dtype=torch.float64)
    row = torch.full((n,), -1, dtype=torch.long)
    for b in range(n):
        r = rows[b] if b < len(rows) else rows[0]
        a, g = int(r["start"]), int(r["len"])
        if a + g > s or len(r["logq"]) != g:
            raise RuntimeError(f"verify_in_train: row {b} span {a}+{g} does not fit {s}")
        logq[b, a : a + g] = torch.tensor(r["logq"], dtype=torch.float32)
        start[b], length[b], seed[b], m[b], row[b] = a, g, int(r["seed"]), float(r["m"]), int(r["row"])
    data["vit_logq"] = logq
    data["vit_start"] = start
    data["vit_len"] = length
    data["vit_seed"] = seed
    data["vit_m"] = m
    data["vit_row"] = row


# -- groups larger than one micro-batch (forward first, then decide; DESIGN 14r) --


@dataclass
class RetainedSlot:
    """One micro-batch's training forward kept between the decision and the backward.

    ``output``: the decoder's output with its autograd graph (under activation
    recompute the graph holds only the checkpointed layer inputs); ``routes``:
    per router-replay instance, the routes this forward used (the backward's
    recompute must replay them).
    """

    ids: torch.Tensor  # [b, S] model input ids (CPU)
    output: Any
    routes: list
    num_tokens: int


@contextmanager
def substitute_decoder_output(decoder: torch.nn.Module, output: Any):
    """Make ``decoder(...)`` return ``output`` (and run nothing) for one forward."""
    calls = []

    def forward(*args: Any, **kwargs: Any) -> Any:
        if calls:
            raise RuntimeError("verify_in_train: a retained decoder output was used twice")
        calls.append(1)
        return output

    decoder.forward = forward
    try:
        yield
    finally:
        del decoder.forward  # back to the class method


def retained_forward(slot: RetainedSlot, original, decoder, instances, kwargs) -> Any:
    """``model_forward`` from a retained slot: same ids, decoder output reused, routes restored.

    Runs inside ``forward_with_post_processing_fn`` after its
    ``set_router_replay_forward`` appended the anchors as this micro-batch's
    backward routes; those are replaced by the routes the retained forward
    actually used (the sticky rule may have kept plain routes).
    """
    ids = kwargs["input_ids_cp_sharded"]
    if tuple(ids.shape) != tuple(slot.ids.shape) or not torch.equal(ids.detach().cpu(), slot.ids):
        raise RuntimeError("verify_in_train: the micro-batch differs from the retained forward")
    for inst, r in zip(instances, slot.routes):
        if r is not None and inst.replay_backward_list:
            inst.replay_backward_list[-1] = r
    out, slot.output = slot.output, None
    with substitute_decoder_output(decoder, out):
        return original(**kwargs)


def plan_group_layout(lengths: list[int], dp_size: int, micro_batch_size: int):
    """One group's rows on (rank, micro-batch): [[row ids per micro-batch] per rank].

    Rows sorted by (length desc, id), dealt to ranks in snake order, cut into
    micro-batches of at most ``micro_batch_size`` rows; ranks with fewer
    micro-batches get filler slots ``None`` (a masked copy of the shortest row)
    so every rank runs the same count. A function of the lengths only.
    """
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    per_rank: list[list[int]] = [[] for _ in range(dp_size)]
    for j, i in enumerate(order):
        lap, k = divmod(j, dp_size)
        per_rank[k if lap % 2 == 0 else dp_size - 1 - k].append(i)
    slots = [
        [rows[a : a + micro_batch_size] for a in range(0, len(rows), micro_batch_size)]
        for rows in per_rank
    ]
    target = max(len(s) for s in slots)
    for s in slots:
        s.extend([None] * (target - len(s)))
    return slots


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

    def group_pass(self, lengths: list[int]) -> float:
        p = 1.0
        for n in lengths:
            p *= self.rate(n)
        return p
