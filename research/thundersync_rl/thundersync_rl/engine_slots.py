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
"""Engine row slots with continuations first (colocated nolanes; research/colo, wake_verify starvation 11-05).

Every generation request of the driver is one Ray call on engine rank 0 (``MegatronGeneration.generate_async``). Ray
runs at most ``max_concurrency`` calls of an async actor at once (1000 by default) and queues the rest FIFO. At
Super-L22 wake_verify one batch's drafts are 72 x 14 = 1008 calls: rank 0's in-flight count sat at exactly 1000 every
step, so a continuation launched after them waited in Ray's queue until drafts finished (~55 s), then
decoded: 12-17 ms per token against a 5.4 ms engine iteration.

The rule here (fixed, no controller):
- The driver holds ``capacity = max_requests x engine DP`` row slots (stock SPR: 128 rows / rank, 1024 at Super-L22).
  A generation call takes one slot before it is sent and returns it when its result is back, so the engine never
  holds more than ``max_requests`` rows per rank and nothing queues in Ray or in the engine's waiting list. Per rank
  this needs the coordinator's least-loaded routing (a rank only receives a row while it holds the fewest, hence at
  most ceil(capacity / DP) = max_requests); prefix-cache affinity routing could exceed it, so the stage requires
  the load_balanced coordinator policy whenever prefix caching is on (``require_least_loaded``).
- Engine rows sent outside generate_async (verify prefill requests) take their slots with ``hold(n)``: n slots at
  once, top priority, so they never wait in the engine's FIFO behind drafts.
- A free slot goes to a waiting continuation / fresh row (``nl_no_log``) before any waiting draft; FIFO within each
  class. Drafts are admitted only as slots free.
- At the wake that starts a phase, drafts are held (``hold_drafts``) until the step's continuations are queued
  (``open_drafts``): the continuations take their slots first, and the next batch's drafts fill what is left. Without
  the hold the drafts (released by the wake) took 1008 of 1024 slots first and decoded to the cap while 510
  continuations waited ~55 s on alternate steps.
- The engine's KV holds ``max_requests`` rows of ``max_sequence_length`` (``check_engine_reservation``): an admitted
  row is never paused or evicted.
- The rank-0 actor's call limit (``ACTOR_MAX_CONCURRENCY``) is above capacity + control calls, so the gate is the
  only admission point.

Exactness is unchanged: the gate reads only the request kind (a config bit set by the stage), never token content;
the keyed sampler draws each (request, position) from its key whatever step it rides in, and the BI forward makes a
row's logits independent of its batch.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import itertools
import time
from typing import Any, AsyncGenerator, Optional

# Calls a ThunderSync worker actor runs at once (ray.remote max_concurrency on ThunderSyncMegatronPolicyWorker).
ACTOR_MAX_CONCURRENCY = 4096
# Control calls (cut / count / stats / refit) that may run beside a full set of generation slots.
CONTROL_HEADROOM = 64

PRIO_CONT = 0  # continuation / fresh row: theta_k target decode of the step being trained
PRIO_DRAFT = 1  # draft of a later batch


class PrioritySlots:
    """``capacity`` slots; free slots go to the lowest (priority, arrival) waiter, a waiter of n slots atomically
    (head of line: a later waiter never jumps it). Single event loop."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"engine slots: capacity {capacity} < 1")
        self.capacity = int(capacity)
        self.used = 0
        # Drafts are admitted only while open (``hold_drafts``): the stage holds them from its wake until the step's
        # continuations are queued, so the continuations take the free slots before the next batch's drafts do.
        self.drafts_open = True
        # Stand-down cut (wake_verify, ``cut_waiting_drafts``): from the cut until the next wake a draft call is not sent;
        # it ends with zero tokens at the gate (``acquire`` returns False), so no draft reaches a stood-down engine.
        self.cutting = False
        self.draft_holding = 0  # draft calls granted a slot and not yet returned (sent / on an engine / result in transit)
        self.cut_at_gate = 0  # draft calls ended at the gate by the cut (this step)
        # Counted stand-down stop: every granted draft call carries an id (``nl_call_id``); the engine's rank 0 reports
        # the ids whose engine request finished (``nl_done_call_ids``). A granted call not yet reported is the only kind
        # that can still be on (or reach) an engine; one reported finished but not yet returned is a result in transit.
        self._call_ids = itertools.count(1)
        self.draft_calls: set[int] = set()  # granted, not yet returned
        self.engine_done: set[int] = set()  # of those, reported finished by the engine
        self.cont_arrivals = 0  # continuation / fresh / verify calls that reached the gate (held or granted)
        self._arrival_target: Optional[int] = None
        self._arrival_event: Optional[asyncio.Event] = None
        self._heap: list[tuple[int, int, int, asyncio.Future]] = []  # (prio, seq, n, future)
        self._live = 0  # waiters in the heap neither granted nor cancelled
        self._seq = itertools.count()
        self.stats = self._fresh_stats()

    @staticmethod
    def _fresh_stats() -> dict[str, float]:
        return {"peak": 0, "admitted_cont": 0, "admitted_draft": 0, "cont_wait_max_s": 0.0, "draft_wait_max_s": 0.0,
                "cont_waited": 0}

    def take_stats(self) -> dict[str, float]:
        s, self.stats = self.stats, self._fresh_stats()
        s["peak"] = max(s["peak"], self.used)
        s["in_use"] = self.used
        s["waiting"] = self._live
        return s

    def _grant(self, prio: int, n: int) -> None:
        self.used += n
        if prio != PRIO_CONT:
            self.draft_holding += n
        self.stats["peak"] = max(self.stats["peak"], self.used)
        self.stats["admitted_cont" if prio == PRIO_CONT else "admitted_draft"] += n

    def hold_drafts(self) -> None:
        """Wake: a new phase. Drafts wait until ``open_drafts``; the previous stand-down's cut mode ends."""
        self.drafts_open = False
        self.cutting = False

    def cut_waiting_drafts(self) -> int:
        """Stand-down (wake_verify cut): every draft call waiting for a slot ends now with zero tokens, unsent, and every
        draft call that reaches the gate until the next wake (``hold_drafts``) does too. Draft calls holding a slot are
        on the engines (or in transit) and are cut there. Returns the waiting drafts ended."""
        self.cutting = True
        n = 0
        keep = []
        while self._heap:
            item = heapq.heappop(self._heap)
            prio, _sq, k, fut = item
            if fut.done():
                continue
            if prio == PRIO_CONT:
                keep.append(item)
                continue
            self._live -= 1
            fut.set_result(False)
            n += 1
        for item in keep:
            heapq.heappush(self._heap, item)
        self.cut_at_gate += n
        self._dispatch()
        return n

    def open_drafts(self) -> None:
        self.drafts_open = True
        self._dispatch()

    def drain_drafts(self) -> int:
        """Stand-down (wake_verify cut): grant every waiting draft at once, over capacity, so the drafts already past
        the dispatch gate reach the engines now and are cut to <= 1 token (a waiting engine request finishes after its
        first token); without this each slot a cut frees admits one more unmarked draft, one cut round per capacity.
        The granted calls are sent after this returns, so the cut loop may still need a further round for the last
        arrivals (it re-cuts any unmarked draft). Rows per engine rank can exceed max_requests during the cut.
        Returns the drafts granted."""
        self.drafts_open = True
        n = 0
        keep = []
        while self._heap:
            item = heapq.heappop(self._heap)
            prio, _sq, k, fut = item
            if fut.done():
                continue
            if prio == PRIO_CONT:
                keep.append(item)
                continue
            self._live -= 1
            self._grant(prio, k)
            fut.set_result(None)
            n += k
        for item in keep:
            heapq.heappush(self._heap, item)
        self._dispatch()
        return n

    def _note_cont_arrival(self) -> None:
        self.cont_arrivals += 1
        if self._arrival_event is not None and self.cont_arrivals >= (self._arrival_target or 0):
            self._arrival_event.set()

    async def wait_cont_arrivals(self, target: int, timeout_s: float = 120.0) -> None:
        """Until ``target`` continuation calls have reached the gate (the stage's launched rows are all queued)."""
        if self.cont_arrivals >= target:
            return
        self._arrival_target, self._arrival_event = int(target), asyncio.Event()
        try:
            await asyncio.wait_for(self._arrival_event.wait(), timeout_s)
        except asyncio.TimeoutError:
            raise RuntimeError(f"engine slots: {self.cont_arrivals} of {target} continuation calls reached the gate "
                               f"after {timeout_s} s") from None
        finally:
            self._arrival_target, self._arrival_event = None, None

    async def acquire(self, prio: int, n: int = 1) -> bool:
        """Take ``n`` slots at once (n <= capacity). Returns False, holding nothing, for a draft ended by a stand-down
        cut (``cut_waiting_drafts``): the call must not be sent."""
        n = int(n)
        if not 1 <= n <= self.capacity:
            raise ValueError(f"engine slots: {n} slots asked of capacity {self.capacity}")
        if prio == PRIO_CONT:
            self._note_cont_arrival()
        elif self.cutting:
            self.cut_at_gate += 1
            return False
        if self.used + n <= self.capacity and (prio == PRIO_CONT or self.drafts_open) and self._live == 0:
            self._grant(prio, n)
            return True
        fut = asyncio.get_running_loop().create_future()
        t0 = time.perf_counter()
        heapq.heappush(self._heap, (int(prio), next(self._seq), n, fut))
        self._live += 1
        self._dispatch()
        try:
            await fut
        except asyncio.CancelledError:
            # Task.cancel() cancels the awaited future itself, so a waiter cancelled before its grant shows up here
            # with fut.cancelled() (not "not done"): it is uncounted here, once, either way.
            if fut.cancelled() or not fut.done():
                fut.cancel()
                self._live -= 1
                self._dispatch()  # a cancelled head may have blocked smaller waiters behind it
            elif fut.result() is not False:
                self.release(n, prio)  # the slots were handed over as the waiter was cancelled
            raise
        if fut.result() is False:
            return False  # ended by a stand-down cut while waiting: nothing held
        key = "cont_wait_max_s" if prio == PRIO_CONT else "draft_wait_max_s"
        self.stats[key] = max(self.stats[key], time.perf_counter() - t0)
        if prio == PRIO_CONT:
            self.stats["cont_waited"] += 1
        return True

    def _dispatch(self) -> None:
        while self._heap:
            prio, _s, n, fut = self._heap[0]
            if fut.done():  # cancelled waiter (already uncounted)
                heapq.heappop(self._heap)
                continue
            if prio != PRIO_CONT and not self.drafts_open:
                return  # the best waiter is a draft and drafts are held
            if self.used + n > self.capacity:
                return  # head of line waits for its n slots
            heapq.heappop(self._heap)
            self._live -= 1
            self._grant(prio, n)
            fut.set_result(None)

    def new_draft_call(self) -> int:
        cid = next(self._call_ids)
        self.draft_calls.add(cid)
        return cid

    def end_draft_call(self, cid: Optional[int]) -> None:
        if cid is not None:
            self.draft_calls.discard(cid)
            self.engine_done.discard(cid)

    def note_engine_done(self, ids) -> None:
        """Engine-finished call ids (rank 0's report); ids of calls already returned are ignored."""
        self.engine_done.update(i for i in ids if i in self.draft_calls)

    def drafts_not_engine_done(self) -> int:
        """Granted draft calls whose engine request has not been reported finished (may still be on / reach an engine)."""
        return len(self.draft_calls) - len(self.engine_done)

    def release(self, n: int = 1, prio: int = PRIO_CONT) -> None:
        if self.used < n or (prio != PRIO_CONT and self.draft_holding < n):
            raise RuntimeError("engine slots: release without a held slot")
        self.used -= n
        if prio != PRIO_CONT:
            self.draft_holding -= n
        self._dispatch()

    @contextlib.asynccontextmanager
    async def hold(self, n: int, prio: int = PRIO_CONT):
        """``async with slots.hold(n):`` n slots for a block of engine rows sent outside generate_async (e.g. verify
        prefill requests): top priority by default, released when the block ends."""
        if not await self.acquire(prio, n):
            raise RuntimeError("engine slots: hold() of draft-priority rows during a stand-down cut")
        try:
            yield
        finally:
            self.release(n, prio)


def row_priority(data: Any) -> int:
    """Continuation / fresh rows carry ``nl_no_log`` (the stage sets it; drafts never do). Single-turn only: a later
    turn of a multi-turn rollout carries no ``nl_no_log`` and would rank as a draft (nolanes requires
    max_rollout_turns 1)."""
    try:
        flags = data["nl_no_log"] if "nl_no_log" in data else None
    except (KeyError, TypeError):
        flags = None
    if flags is not None and any(bool(x) for x in flags):
        return PRIO_CONT
    return PRIO_DRAFT


def capacity_of(mc: dict, n_workers: int) -> tuple[int, int, int]:
    """(capacity, max_requests per engine rank, engine DP) by rule: max_requests x (workers / (TP x PP)), from the
    engine's resolved mcore_generation_config (``MegatronGeneration.cfg``)."""
    mr = mc.get("max_requests")
    if mr is None:
        raise ValueError("engine slots: policy.generation.mcore_generation_config.max_requests must be set")
    tp = int(mc.get("tensor_model_parallel_size", 1) or 1)
    pp = int(mc.get("pipeline_model_parallel_size", 1) or 1)
    dp = int(n_workers) // (tp * pp)
    if dp < 1 or dp * tp * pp != int(n_workers):
        raise ValueError(f"engine slots: {n_workers} workers is not a multiple of TP {tp} x PP {pp}")
    cap = int(mr) * dp
    if cap + CONTROL_HEADROOM > ACTOR_MAX_CONCURRENCY:
        raise ValueError(f"engine slots: capacity {cap} + {CONTROL_HEADROOM} control calls exceeds the worker actor's "
                         f"max_concurrency {ACTOR_MAX_CONCURRENCY}")
    return cap, int(mr), dp


def require_least_loaded(mc: dict) -> None:
    """The per-rank bound needs the coordinator's least-loaded ROUTING. Prefix caching itself may be on; its affinity
    routing may not (resolved as megatron_worker._resolve_coordinator_policy: caching off routes load-balanced; on,
    the configured prefix_caching_coordinator_policy, MCore default longest_prefix)."""
    if not mc.get("enable_prefix_caching", False):
        return
    policy = str(mc.get("prefix_caching_coordinator_policy", "longest_prefix"))
    policy = policy.split(".")[-1].lower()
    if policy != "load_balanced":
        raise ValueError(f"engine slots: prefix_caching_coordinator_policy {policy!r} routes on prefix affinity and could "
                         "put more than max_requests rows on one engine rank; set it to load_balanced")


def install(gens: list, capacity: int) -> PrioritySlots:
    """Gate ``generate_async`` of every distinct generation object in ``gens`` (instance attribute) with ONE set of
    ``capacity`` priority slots (the loop's handle and the rollout impl's handle may be different objects)."""
    uniq = list({id(g): g for g in gens if g is not None}.values())
    have = [g.__dict__.get("_ts_engine_slots") for g in uniq]
    if any(h is not None for h in have):
        raise RuntimeError("engine slots: already installed")
    slots = PrioritySlots(capacity)
    for gen in uniq:
        _wrap(gen, slots)
    return slots


def cut_result(data: Any) -> Any:
    """The generation output of a one-row draft call ended at the gate by a stand-down cut: the prompt and no generated
    token, in the engine parser's layout (megatron_worker._parse_result_to_batched_data_dict), marked cut, no hidden-log
    key."""
    import torch

    try:
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    except ImportError:  # unit harnesses without the telemetry overlay: the call data's own type below
        BatchedDataDict = type(data)

    ids = data["input_ids"]
    lens = data["input_lengths"]
    out = (type(data) if isinstance(data, BatchedDataDict) else dict)(
        {
            "output_ids": ids.detach().clone().cpu(),
            "logprobs": torch.zeros(tuple(ids.shape), dtype=torch.float),
            "generation_lengths": torch.zeros(int(ids.shape[0]), dtype=torch.long),
            "unpadded_sequence_lengths": lens.detach().clone().to(torch.long).cpu(),
        }
    )
    out["ts_draft"] = [{"key": None, "rid": -1, "cut": 1}]
    out["gen_leader_worker_idx"] = [0]
    return out


def _wrap(gen: Any, slots: PrioritySlots) -> None:
    orig = gen.generate_async

    async def generate_async(data: Any, greedy: bool = False) -> AsyncGenerator[Any, None]:
        # One row per call (the rollout path: one sample per _generate_response). A multi-row call would take its slots
        # one at a time (hold-and-wait): refused.
        n = int(getattr(data, "size", 1) or 1)
        if n != 1:
            raise RuntimeError(f"engine slots: a generation call of {n} rows (one row per call under the slot gate)")
        prio = row_priority(data)
        if not await slots.acquire(prio):
            # A draft ended by the stand-down cut before it was sent: zero tokens, no engine row, no hidden log (the
            # stage's empty-draft path: fixed G decodes the slot fresh, VG gives the draft its keyed ⊥ coin).
            yield 0, cut_result(data)
            return
        cid = None
        if prio != PRIO_CONT:
            cid = slots.new_draft_call()
            data["nl_call_id"] = [cid]  # the engine's rank 0 reports it when the request finishes (nl_done_call_ids)
        # The slot is returned when the call ends: with its result, on an error, or on a cancel (the engine's abort of
        # the request is sent by the cancelled call; the slot may be reused before the engine has dropped the row).
        try:
            async for item in orig(data, greedy=greedy):
                yield item
        finally:
            slots.end_draft_call(cid)
            slots.release(1, prio)

    gen.__dict__["_ts_engine_slots"] = slots
    gen.generate_async = generate_async


def check_engine_reservation(engine: Any, reserved_blocks: int = 0) -> dict[str, int]:
    """Engine rank: the KV pool holds every admitted row at full length (max_requests x max_kv_block_count blocks), so
    no row is ever paused or evicted for blocks. Raises otherwise (the buffer is sized by rule:
    research/colo/jobs/engine_buf.py). ``reserved_blocks``: blocks taken out of the pool for good (tree_spec's private
    node blocks, ``thundersync.tail_tree_rows``)."""
    ctx = engine.context
    alloc = ctx.kv_block_allocator
    usable = int(alloc.pool_size) - 1 - int(alloc.paused_limit) - int(reserved_blocks)
    need = int(ctx.max_requests) * int(ctx.max_kv_block_count)
    out = {"kv_blocks_usable": usable, "kv_blocks_needed": need, "max_requests": int(ctx.max_requests),
           "kv_blocks_reserved": int(reserved_blocks)}
    if usable < need:
        raise RuntimeError(f"[engine slots] KV pool {usable} blocks < max_requests {ctx.max_requests} x "
                           f"{ctx.max_kv_block_count} blocks/row = {need}: admitted rows could be paused; size the "
                           "engine buffer by rule (engine_buf.py) or lower max_requests")
    return out


def summary_line(step: int, cap: int, s: dict[str, float], extra: Optional[dict] = None) -> str:
    e = "" if not extra else " " + " ".join(f"{k}={v}" for k, v in extra.items())
    return (f"[engine slots] step={step} capacity={cap} peak={int(s['peak'])} in_use={int(s['in_use'])} "
            f"waiting={int(s['waiting'])} admitted_cont={int(s['admitted_cont'])} "
            f"admitted_draft={int(s['admitted_draft'])} cont_waited={int(s['cont_waited'])} "
            f"cont_wait_max_s={s['cont_wait_max_s']:.3f} draft_wait_max_s={s['draft_wait_max_s']:.2f}{e}")
