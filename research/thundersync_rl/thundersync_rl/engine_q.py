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
"""Engine-sourced q for block verification under batch invariance.

With batch-invariant kernels the inference engine's decode forward is bitwise
the training forward, so the true sampling law q of a drafted token is the
engine's own distribution:

* log q(x_i) of every drafted token is the engine's reported log-prob
  (``generated_log_probs``, already streamed with the draft);
* a full q row, needed only where the lazy scan computes a residual mass and
  at the residual sample, is ``log_softmax(W_head^(v) h_i)`` with h_i the
  engine's final hidden state (LM-head input, after the final norm) at that
  step and W_head^(v) the LM-head weight of the drafting version v.

So the verifier needs no scorer copy and no whole-model weight stash: the
engine logs h_i per drafted token (d x 2 bytes, ``HiddenLog``) and the learner
keeps only the LM head per pending version (V x d x 2 bytes). p comes from the
training model's own bf16 forward at theta_{k+1}.

This module holds the parts that need no Megatron: the engine-side hidden
store and per-step row mapping, and the per-draft verification rule.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

import torch

# Sampling-params attribute that tags a draft request for hidden-state logging.
HIDDEN_KEY_ATTR = "ts_hidden_key"


def step_rows(
    request_ids: list[int],
    kv_offsets: list[int],
    query_lengths: list[int],
    lookup: Callable[[int], Optional[tuple[str, int]]],
) -> list[tuple[int, str, int]]:
    """Rows of one engine forward that sample a tagged request's next token.

    The forward produces one LM-head row per active request (request order),
    the row of its last input token, from which that request's next token is
    sampled. ``lookup(request_id)`` gives (draft key, prompt length) for
    tagged requests, else None. Returns (row, key, t): row ``row`` produced
    generated token ``t`` of draft ``key``, where t = (position of the last
    input token) + 1 - prompt length. Rows with t < 0 (an intermediate prefill
    chunk, which samples nothing) are skipped. Recomputation after an engine
    eviction rewrites the same t, so a draft's rows are keyed by position, not
    by arrival order.
    """
    out = []
    for row, (rid, kv, ql) in enumerate(zip(request_ids, kv_offsets, query_lengths)):
        info = lookup(int(rid))
        if info is None:
            continue
        key, plen = info
        t = int(kv) + int(ql) - plen
        if t >= 0:
            out.append((row, key, t))
    return out


class HiddenLog:
    """Final hidden states of tagged draft requests, by (key, generated index).

    Engine side. ``record`` copies the selected rows of a forward's LM-head
    input to (pinned, when CUDA is available) host memory without blocking
    the engine; ``pop`` returns rows 0..n-1 of a draft, or None if any is
    missing. Memory: d x 2 bytes (bf16) per drafted token, released as drafts
    are popped. Thread-safe (the engine records on its loop thread).
    """

    def __init__(self, chunk_rows: int = 8192):
        self.chunk_rows = chunk_rows
        self._lock = threading.Lock()
        self._rows: dict[str, dict[int, tuple[list, int]]] = {}
        self._closed: set[str] = set()
        self._cur: Optional[list] = None  # [buffer, used, live rows, event]
        self.recorded = 0

    def _alloc(self, k: int, like: torch.Tensor) -> tuple[list, int]:
        cur = self._cur
        if cur is None or cur[1] + k > cur[0].shape[0]:
            n = max(self.chunk_rows, k)
            buf = torch.empty(
                (n, like.shape[-1]),
                dtype=like.dtype,
                pin_memory=like.is_cuda,
            )
            cur = self._cur = [buf, 0, 0, None]
        off = cur[1]
        cur[1] += k
        return cur, off

    def record(self, entries: list[tuple[int, str, int]], hidden: torch.Tensor) -> None:
        """Store ``hidden[row]`` as token ``t`` of draft ``key`` for each entry."""
        with self._lock:
            entries = [e for e in entries if e[1] not in self._closed]
            if not entries:
                return
            idx = torch.tensor([r for r, _, _ in entries], device=hidden.device)
            sel = hidden.detach().index_select(0, idx)
            chunk, off = self._alloc(len(entries), sel)
            chunk[0][off : off + len(entries)].copy_(sel, non_blocking=sel.is_cuda)
            if sel.is_cuda:
                ev = torch.cuda.Event()
                ev.record()
                chunk[3] = ev
            for j, (_, key, t) in enumerate(entries):
                slots = self._rows.setdefault(key, {})
                old = slots.get(t)
                if old is not None:
                    old[0][2] -= 1
                slots[t] = (chunk, off + j)
                chunk[2] += 1
            self.recorded += len(entries)

    def pop(self, key: str, n: int) -> Optional[torch.Tensor]:
        """Rows 0..n-1 of draft ``key`` [n, d] (CPU), or None if any is missing.

        Closes the key: rows the engine records for it later are dropped.
        """
        with self._lock:
            self._closed.add(key)
            slots = self._rows.pop(key, {})
            for chunk, _ in slots.values():
                chunk[2] -= 1
            if any(t not in slots for t in range(n)):
                return None
            if n == 0:
                return torch.empty(0, 0)
            for ev in {id(c): c[3] for c, _ in slots.values()}.values():
                if ev is not None:
                    ev.synchronize()
            return torch.stack([slots[t][0][0][slots[t][1]] for t in range(n)])

    def live_rows(self) -> int:
        with self._lock:
            return sum(len(s) for s in self._rows.values())


def residual_rows_needed(n_eval: int, tau: int, g: int) -> int:
    """Q rows a lazy verification computed: residual masses plus the emit row."""
    return n_eval + (tau < g)


def verify_engine_q(
    p_logprobs: torch.Tensor,
    draft: torch.Tensor,
    engine_logq: torch.Tensor,
    q_logprob_rows: Callable[[torch.Tensor], torch.Tensor],
    generator: Optional[torch.Generator] = None,
) -> tuple[int, int, torch.Tensor, int]:
    """Block-verify one draft with p from training and q from the engine.

    ``p_logprobs``: [g+1, V] fp32 log-probs of the training model at
    theta_{k+1} (log-softmax rows, as training computes them).
    ``engine_logq``: [g] fp32 log q(x_i), the engine's log-probs of the drafted
    tokens. ``q_logprob_rows(idx)``: [k, V] fp32 q log-prob rows at positions
    ``idx`` (reconstructed from the engine's hidden states), called only where
    the lazy scan needs a residual mass and for the residual sample.

    Same rule and generator calls as ``block_verify_logits`` (one ``rand(g+1)``,
    one ``multinomial``), with log-prob rows standing in for logits (their
    logsumexp is 0). Returns (tau, y, logp, residual masses computed); logp are
    entries of ``p_logprobs``, i.e. training's own log-probs of the kept tokens.
    """
    from thundersync_rl.block_verification_fused import block_verify_stats_lazy

    g = draft.numel()
    dev = p_logprobs.device
    ar = torch.arange(g, device=dev)
    lpx = p_logprobs[ar, draft]
    lqx = engine_logq.to(device=dev, dtype=torch.float32)
    zeros = torch.zeros(g + 1, dtype=torch.float32, device=dev)
    return block_verify_stats_lazy(
        lpx,
        lqx,
        zeros,
        zeros[:g],
        lambda idx: p_logprobs[idx],
        q_logprob_rows,
        generator,
    )
