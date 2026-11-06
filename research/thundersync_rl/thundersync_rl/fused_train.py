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
"""Fused verification + training under batch-invariant kernels (experimental).

With batch-invariant kernels the learner's bf16 training forward is bitwise the
engine's numerics, so when the learner verifies a draft with p from the
TRAINING model at theta_{k+1}, that p forward is exactly the forward training
needs for the trajectory. ``thundersync.fused_verify_train`` runs it with
autograd and, for drafts that verification turns into finished rollouts
(fully accepted and terminated), keeps the graph (``FusedGraphStore``). The
rollout returns at once, the environment rewards it as usual, and the
trajectory reaches the learner as a single-row chunk on the data-parallel rank
holding its graph (``FusedRef``). There the regular ``train_microbatch`` runs
unchanged except that the model forward is replaced by the retained logits
(``substitute_model_forward``): same loss, normalization counts, gradient
buckets and finalization, one forward fewer.

The training row is the verified row plus the environment's reply (masked out
of the loss). Causal attention makes the verified logits a prefix of the
training row's logits; the positions after it predict only masked tokens and
are filled with constants, which carry no gradient. Anything that does not
match (other tokens, unmasked positions after the verified row) falls back to
the real forward.

Everything here is CPU-importable so the bookkeeping is unit-tested without
Megatron or GPUs.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Iterator, Optional

import torch


@dataclass(frozen=True)
class FusedRef:
    """Driver-side handle on one retained verification forward."""

    key: str  # draft key; names the graph in the owner's FusedGraphStore
    dp_rank: int  # data-parallel rank holding the graph
    num_tokens: int  # tokens of the verified row (prompt + draft)


@dataclass
class FusedGraph:
    """Worker-side retained training-model forward of one verified row."""

    ids: torch.Tensor  # [n] token ids (prompt + draft), CPU
    logits: torch.Tensor  # [1, n, V] model output with its autograd graph

    @property
    def num_tokens(self) -> int:
        return self.ids.numel()


@dataclass
class FusedOutcome:
    """What a substituted forward did during one training call."""

    used: bool
    fallback: bool


class FusedGraphStore:
    """Retained graphs of one optimizer step, bounded by a token budget.

    Activation memory of a retained graph grows with its row length, so the
    budget is in tokens. Graphs must not outlive the step: ``clear`` before the
    optimizer step changes the parameters they were taken at.
    """

    def __init__(self) -> None:
        self._graphs: dict[str, FusedGraph] = {}
        self.retained_tokens = 0
        self.num_retained = 0  # graphs ever retained this step

    def can_retain(self, num_tokens: int, max_tokens: int) -> bool:
        return self.retained_tokens + num_tokens <= max_tokens

    def put(self, key: str, graph: FusedGraph) -> None:
        if key in self._graphs:
            raise KeyError(f"fused graph {key!r} already retained this step")
        self._graphs[key] = graph
        self.retained_tokens += graph.num_tokens
        self.num_retained += 1

    def pop(self, key: str) -> Optional[FusedGraph]:
        graph = self._graphs.pop(key, None)
        if graph is not None:
            self.retained_tokens -= graph.num_tokens
        return graph

    def clear(self) -> int:
        """Drop every graph (frees their activations); return how many were left."""
        left = len(self._graphs)
        self._graphs.clear()
        self.retained_tokens = 0
        return left

    def __len__(self) -> int:
        return len(self._graphs)


def stitch_fused_logits(
    graph: FusedGraph, input_ids: torch.Tensor, token_mask: torch.Tensor
) -> Optional[torch.Tensor]:
    """Logits for a training row from a retained forward, or None if it does not apply.

    ``input_ids`` / ``token_mask`` are the training micro-batch ([1, S]). The
    retained forward covers the first ``n`` positions (the verified row); it
    applies when the training row starts with exactly those tokens and every
    later token is masked out of the loss (position j predicts token j+1, so
    positions >= n-1 must predict only masked tokens). Positions >= n are filled
    with zeros: they enter the loss only through masked terms and carry no
    gradient.
    """
    n = graph.num_tokens
    if input_ids.dim() != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < n:
        return None
    if not torch.equal(input_ids[0, :n].cpu(), graph.ids):
        return None
    if bool(token_mask[0, n:].ne(0).any()):
        return None
    logits = graph.logits
    filler = logits.new_zeros((1, input_ids.shape[1] - n, logits.shape[-1]))
    # Always a fresh tensor: the loss path scales logits in place.
    return torch.cat([logits, filler], dim=1)


def make_fused_model_forward(
    graph: FusedGraph,
    original: Callable[..., torch.Tensor],
    outcome: FusedOutcome,
    *,
    check: bool,
) -> Callable[..., torch.Tensor]:
    """A drop-in for ``nemo_rl.models.megatron.train.model_forward``.

    The first call returns the retained logits when they apply (else runs
    ``original``); any further call (e.g. a rerun) runs ``original``.
    ``check`` (debug) also runs the real forward without grad and prints the
    largest logit difference over the verified positions (0 under batch
    invariance).
    """

    def forward(**kwargs: Any) -> torch.Tensor:
        if outcome.used or outcome.fallback:
            return original(**kwargs)
        logits = stitch_fused_logits(
            graph, kwargs["input_ids_cp_sharded"], kwargs["data_dict"]["token_mask"]
        )
        if logits is None:
            outcome.fallback = True
            return original(**kwargs)
        outcome.used = True
        if check:
            with torch.no_grad():
                ref = original(**kwargs)
            n = graph.num_tokens
            diff = (ref[:, :n].float() - logits[:, :n].detach().float()).abs().max()
            print(
                f"[fused check] tokens={n} max|logit diff|={float(diff):.3e}",
                flush=True,
            )
        return logits

    return forward


@contextlib.contextmanager
def substitute_model_forward(
    module: ModuleType, forward: Callable[..., torch.Tensor]
) -> Iterator[None]:
    """Temporarily replace ``module.model_forward``.

    The Megatron train path has no hook for precomputed logits;
    ``forward_with_post_processing_fn`` resolves ``model_forward`` from its
    module globals at call time, so swapping it there for one training call
    keeps every other step (loss, scaling, backward, gradient capture) the
    regular code.
    """
    original = module.model_forward
    module.model_forward = forward
    try:
        yield
    finally:
        module.model_forward = original
