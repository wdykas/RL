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
"""Continuation verification yields into training at micro-batch boundaries (opt-in).

``thundersync.verify_yield`` (DESIGN.md section 14x). While streaming over the
tails, learner calls are numbered and every rank runs them in that order
(``ordered_call``). A piece verification sent while a training dispatch runs
gets the next number, so it waits for the whole dispatch (Nano L8 1k: ~1 s;
full Nano 8k EP8: ~15 s, measured in research/agent_tail_speed.md).

With ``verify_yield`` the driver posts each continuation verification to a
``VerifyMailbox`` (one Ray actor) and sends a small ordered call
(``posted_call``) in its place. A rank running a training dispatch asks the
mailbox at every micro-batch boundary (pipeline parallel size 1: after one
micro-batch's backward, before the next one's forward) whether the next
ordered call is a posted verification; if so it runs it there and posts the
result. The ordered call later finds it done and returns at once. The driver
reads results from the mailbox, whichever path produced them.

Exactness. The training dispatch runs the same micro-batches in the same
order with the same loss scaling: a verification is a no-grad forward that
leaves no state behind (router-replay targets, RNG states and module modes
are restored around it). Verification results do not depend on when they run
(fixed weights theta_k within the step, explicit seeds, batch-invariant
kernels). So the gradient, the decisions and the tokens are bitwise those of
the ordered path; only timing changes.

Agreement. Ranks whose forwards share collectives must run a verification at
the same boundary: the expert-parallel group (EP > 1: its ranks train in
lockstep, equal micro-batch counts) or one data-parallel replica's
tensor-parallel group (EP = 1). The mailbox decides once per
(domain, dispatch, boundary, slot), atomically (a single-threaded actor), and
every rank of the domain reads that decision. Ranks of different domains share
no collective in a verification, so their decisions may differ.

Per-rank cost: one mailbox round trip per micro-batch boundary (~1 ms; a
micro-batch is 0.3-4 s), the posted verification fetched once per rank (its
rows are what the ordered call carried), the result put once. The mailbox
holds posted payload references and per-boundary decisions of live dispatches
only (cleared when a result is read). Not built: pipeline parallel size > 1
(the 1F1B schedule has no common boundary across stages); there the posted
call runs in order as before.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any, Iterator, Optional

import ray
import torch


class Mailbox:
    """Posted continuation verifications, yield decisions and results (one per learner).

    Runs as the Ray actor ``VerifyMailbox`` (single-threaded: each decision is atomic).
    """

    def __init__(self) -> None:
        self.posted: dict[int, Any] = {}
        self.n_ranks: dict[int, int] = {}
        self.results: dict[int, dict[int, Any]] = defaultdict(dict)
        self.ready: dict[int, asyncio.Event] = {}
        # (domain, dispatch seq) -> verifications inlined in that dispatch
        self.inlined: dict[tuple, int] = defaultdict(int)
        self.decisions: dict[tuple, int] = {}
        # Latest dispatch a domain asked about. Once a rank of a domain is in a
        # later dispatch, every rank of it has passed all boundaries of the
        # earlier ones (the last micro-batch's collectives come after the last
        # boundary), so their decisions are dropped.
        self.current: dict[Any, int] = {}

    def post(self, seq: int, payload_box: list, n_ranks: int) -> None:
        self.posted[seq] = payload_box[0]
        self.n_ranks[seq] = int(n_ranks)
        ev = self.ready.setdefault(seq, asyncio.Event())
        if len(self.results.get(seq, ())) >= self.n_ranks[seq]:
            ev.set()

    def get(self, seq: int) -> list:
        return [self.posted[seq]]

    def decide(self, domain: Any, dispatch: int, boundary: int, slot: int) -> int:
        """The verification a domain runs at slot ``slot`` of a boundary, or -1.

        The next ordered call after dispatch ``dispatch`` and the ones already
        inlined there; eligible only if it is a posted verification.
        """
        key = (domain, dispatch, boundary, slot)
        if dispatch > self.current.get(domain, -1):
            self.current[domain] = dispatch
            for k in [k for k in self.decisions if k[0] == domain and k[1] < dispatch]:
                del self.decisions[k]
            for k in [k for k in self.inlined if k[0] == domain and k[1] < dispatch]:
                del self.inlined[k]
        if key in self.decisions:
            return self.decisions[key]
        k = (domain, dispatch)
        v = dispatch + 1 + self.inlined[k]
        d = v if v in self.posted else -1
        if d >= 0:
            self.inlined[k] += 1
        self.decisions[key] = d
        return d

    def put_result(self, seq: int, rank: int, value_box: list) -> None:
        self.results[seq][int(rank)] = value_box[0]
        ev = self.ready.setdefault(seq, asyncio.Event())
        if seq in self.n_ranks and len(self.results[seq]) >= self.n_ranks[seq]:
            ev.set()

    async def result(self, seq: int) -> list:
        """Every rank's [result reference, error] of ``seq`` (rank order), once all are in."""
        ev = self.ready.setdefault(seq, asyncio.Event())
        await ev.wait()
        out = self.results.pop(seq)
        self.posted.pop(seq, None)
        self.n_ranks.pop(seq, None)
        self.ready.pop(seq, None)
        return [out[r] for r in sorted(out)]


VerifyMailbox = ray.remote(num_cpus=0)(Mailbox)


def yield_domain() -> Optional[tuple]:
    """This learner rank's agreement domain, or None (pipeline parallel: not built).

    EP > 1: the expert-parallel group (the ranks that share an
    expert-data-parallel rank; they train in lockstep). EP = 1: one
    data-parallel replica (its tensor/context-parallel ranks hold the same rows).
    """
    from megatron.core import parallel_state as ps

    if (ps.get_pipeline_model_parallel_world_size() or 1) > 1:
        return None
    if (ps.get_expert_model_parallel_world_size() or 1) > 1:
        return ("ep", int(ps.get_expert_data_parallel_rank()))
    return ("dp", int(ps.get_data_parallel_rank()))


class ReplaySnapshot:
    """Router-replay state of every RouterReplay instance (restored after an inlined call)."""

    _FIELDS = (
        "target_topk_idx",
        "recorded_topk_idx",
        "replay_backward_list",
        "router_replay_action",
    )

    def __init__(self) -> None:
        try:
            from megatron.core.transformer.moe.router_replay import RouterReplay
        except ImportError:
            self.saved = []
            return
        self.saved = [
            (
                inst,
                {
                    f: (list(getattr(inst, f)) if f == "replay_backward_list" else getattr(inst, f))
                    for f in self._FIELDS
                    if hasattr(inst, f)
                },
            )
            for inst in RouterReplay.global_router_replay_instances
        ]

    def restore(self) -> None:
        for inst, fields in self.saved:
            for f, v in fields.items():
                setattr(inst, f, v)


class RngSnapshot:
    """CPU, CUDA and Megatron tracker RNG states (restored after an inlined call)."""

    def __init__(self) -> None:
        self.cpu = torch.get_rng_state()
        self.cuda = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        try:
            from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

            self.tracker = get_cuda_rng_tracker().get_states()
        except Exception:  # no tracker initialized
            self.tracker = None

    def restore(self) -> None:
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state(self.cuda)
        if self.tracker is not None:
            from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

            get_cuda_rng_tracker().set_states(self.tracker)


class YieldWorkerMixin:
    """Learner-worker side of ``verify_yield`` (mixed into the ThunderSync policy worker).

    Needs ``self.rank`` and ``self.model``; ``ordered_call`` brackets each call
    with ``_yield_begin_call`` / ``_yield_end_call``, and the training path wraps
    its micro-batch iterators with ``boundary_iter(..., self._yield_point, ...)``
    while ``_yield_ok``.
    """

    _yield_mailbox = None
    _yield_domain = None
    _yield_ok = False
    _ordered_running = None

    def _yield_begin_call(self, seq: int) -> float:
        self._ordered_running = seq
        self._yield_boundary = 0
        self._yield_first = [True]
        return (getattr(self, "_yield_stats", None) or {}).get("inline_s", 0.0)

    def _yield_end_call(self, inline_s0: float) -> float:
        """Seconds of verifications this call ran at its boundaries."""
        self._ordered_running = None
        return (getattr(self, "_yield_stats", None) or {}).get("inline_s", 0.0) - inline_s0

    def set_verify_mailbox(self, mailbox) -> bool:
        """Learner rank: the driver's VerifyMailbox (``thundersync.verify_yield``).

        Returns whether this rank yields at micro-batch boundaries (pipeline
        parallel size 1); otherwise posted calls run in order only.
        """
        self._yield_mailbox = mailbox
        self._yield_domain = self._yield_domain_key()
        self._yield_done: dict[int, bool] = {}
        self._yield_stats = {"inlined": 0, "inline_s": 0.0, "boundaries": 0, "poll_s": 0.0}
        return self._yield_domain is not None

    def _yield_domain_key(self) -> Optional[tuple]:
        return yield_domain()

    def posted_call(self, key: int) -> None:
        """Ordered call ``key`` of a posted verification: run it unless a boundary did."""
        if self._yield_done.pop(key, False):
            return None
        self._run_posted(key)
        return None

    def _run_posted(self, key: int) -> None:
        box = ray.get(self._yield_mailbox.get.remote(key))
        payload = ray.get(box[0])
        try:
            out = getattr(self, payload["method"])(**payload["kwargs"])
        except BaseException as exc:
            # The driver waits on the mailbox: report the failure there too.
            self._yield_mailbox.put_result.remote(key, self.rank, [[None, repr(exc)]])
            raise
        self._yield_mailbox.put_result.remote(key, self.rank, [[ray.put(out), None]])

    def yield_stats(self) -> dict[str, float]:
        """Inlined verifications on this rank since the last call (reset)."""
        st = getattr(self, "_yield_stats", None) or {}
        self._yield_stats = {"inlined": 0, "inline_s": 0.0, "boundaries": 0, "poll_s": 0.0}
        return st

    def _yield_point(self) -> None:
        """A micro-batch boundary of training dispatch ``_ordered_running``."""
        mb, dom, d = self._yield_mailbox, self._yield_domain, self._ordered_running
        b = self._yield_boundary
        self._yield_boundary += 1
        st = self._yield_stats
        st["boundaries"] += 1
        slot = 0
        while True:
            t0 = time.perf_counter()
            v = ray.get(mb.decide.remote(dom, d, b, slot))
            st["poll_s"] += time.perf_counter() - t0
            if v < 0:
                return
            slot += 1
            t0 = time.perf_counter()
            replay = ReplaySnapshot()
            rng = RngSnapshot()
            modes = [(m, m.training) for c in self._model_chunks() for m in c.modules()]
            grad = torch.is_grad_enabled()
            try:
                self._run_posted(v)
            finally:
                torch.set_grad_enabled(grad)
                for m, t in modes:
                    m.training = t
                rng.restore()
                replay.restore()
            self._yield_done[v] = True
            st["inlined"] += 1
            st["inline_s"] += time.perf_counter() - t0

    def _model_chunks(self) -> list:
        m = self.model
        return list(m) if isinstance(m, (list, tuple)) else [m]



def boundary_iter(it: Iterator[Any], hook, first: list) -> Iterator[Any]:
    """``it`` with ``hook()`` called before every item but the dispatch's first.

    ``first``: a one-element list shared by the dispatch's pipeline calls
    (True until the dispatch's first micro-batch is taken). Under pipeline
    parallel size 1 Megatron takes micro-batch i (``next``) only after
    micro-batch i-1's backward, so each call is a boundary with no live
    activations.
    """
    for x in it:
        if first[0]:
            first[0] = False
        else:
            hook()
        yield x
