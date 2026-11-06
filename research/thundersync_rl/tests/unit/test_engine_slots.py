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
"""Engine row slots (engine_slots.py): capacity, continuations first, cancellation, capacity rule."""

import asyncio
import types

from thundersync_rl import engine_slots as es


class _Gen:
    def __init__(self):
        self.live = 0
        self.peak = 0
        self.order = []
        self.release = asyncio.Event()

    async def generate_async(self, data, greedy=False):
        self.live += 1
        self.peak = max(self.peak, self.live)
        self.order.append(data["tag"][0])
        await self.release.wait()
        self.live -= 1
        yield 0, data


def _data(tag, cont):
    d = {"tag": [tag]}
    if cont:
        d["nl_no_log"] = [1]
    return d


async def _drain(gen, data):
    async for _ in gen.generate_async(data):
        pass


def test_capacity_and_continuations_first():
    async def main():
        g = _Gen()
        slots = es.install([g, g], 4)
        tasks = [asyncio.ensure_future(_drain(g, _data(f"d{i}", False))) for i in range(10)]
        await asyncio.sleep(0.01)
        assert g.live == 4 and slots.used == 4
        tasks += [asyncio.ensure_future(_drain(g, _data(f"c{i}", True))) for i in range(3)]
        await asyncio.sleep(0.01)
        g.release.set()
        await asyncio.gather(*tasks)
        assert g.peak == 4 and slots.used == 0
        # the first 4 drafts ran, then every continuation before the remaining drafts (FIFO within a class)
        assert g.order[:4] == ["d0", "d1", "d2", "d3"]
        assert g.order[4:7] == ["c0", "c1", "c2"]
        assert g.order[7:] == [f"d{i}" for i in range(4, 10)]
        s = slots.take_stats()
        assert s["admitted_cont"] == 3 and s["admitted_draft"] == 10 and s["cont_waited"] == 3

    asyncio.run(main())


def test_cancelled_waiter_frees_nothing_and_is_skipped():
    async def main():
        g = _Gen()
        slots = es.install([g], 1)
        t1 = asyncio.ensure_future(_drain(g, _data("a", False)))
        await asyncio.sleep(0.01)
        t2 = asyncio.ensure_future(_drain(g, _data("b", True)))
        t3 = asyncio.ensure_future(_drain(g, _data("c", False)))
        await asyncio.sleep(0.01)
        t2.cancel()
        await asyncio.sleep(0.01)
        g.release.set()
        await asyncio.gather(t1, t3)
        assert g.order == ["a", "c"] and slots.used == 0

    asyncio.run(main())


def test_held_drafts_wait_for_the_queued_continuations():
    """The wake: drafts released first would take every slot; held until the continuations are queued, they get only
    the slots the continuations leave."""

    async def main():
        g = _Gen()
        slots = es.install([g], 4)
        slots.hold_drafts()
        tasks = [asyncio.ensure_future(_drain(g, _data(f"d{i}", False))) for i in range(4)]
        await asyncio.sleep(0.01)
        assert g.live == 0  # drafts held
        tasks += [asyncio.ensure_future(_drain(g, _data(f"c{i}", True))) for i in range(3)]
        await slots.wait_cont_arrivals(3, timeout_s=5)
        await asyncio.sleep(0.01)
        assert g.live == 3 and sorted(g.order) == ["c0", "c1", "c2"]
        slots.open_drafts()
        await asyncio.sleep(0.01)
        assert g.live == 4 and g.order[3] == "d0"  # one free slot left for drafts
        g.release.set()
        await asyncio.gather(*tasks)
        assert slots.used == 0 and g.order[4:] == ["d1", "d2", "d3"]

    asyncio.run(main())


def test_drain_grants_every_waiting_draft_at_the_stand_down():
    """Drafts > capacity, then the stand-down: drain_drafts sends every waiting draft at once (over capacity), so one
    cut round sees them all; a waiting continuation keeps its place."""

    async def main():
        g = _Gen()
        slots = es.install([g], 2)
        tasks = [asyncio.ensure_future(_drain(g, _data(f"d{i}", False))) for i in range(6)]
        await asyncio.sleep(0.01)
        assert g.live == 2
        assert slots.drain_drafts() == 4
        await asyncio.sleep(0.01)
        assert g.live == 6 and slots.used == 6  # over capacity until they finish (cut)
        late = asyncio.ensure_future(_drain(g, _data("c0", True)))
        await asyncio.sleep(0.01)
        assert g.live == 6  # waits for the over-capacity drafts to finish
        g.release.set()
        await asyncio.gather(*tasks, late)
        assert slots.used == 0 and g.order[-1] == "c0"

    asyncio.run(main())


def test_error_inside_the_call_releases_its_slot():
    class _Bad:
        async def generate_async(self, data, greedy=False):
            raise RuntimeError("engine died")
            yield  # pragma: no cover

    async def main():
        b = _Bad()
        slots = es.install([b], 1)
        for _ in range(3):
            try:
                await _drain(b, _data("x", True))
            except RuntimeError:
                pass
        assert slots.used == 0

    asyncio.run(main())


def test_multi_row_call_is_refused():
    async def main():
        g = _Gen()
        slots = es.install([g], 4)
        d = types.SimpleNamespace(size=2)
        try:
            await _drain(g, d)
        except RuntimeError:
            pass
        else:
            raise AssertionError("a 2-row call must be refused")
        assert slots.used == 0

    asyncio.run(main())


def test_held_slots_survive_a_stand_down_wake_cycle():
    """Slots held by rows in flight across a stand-down / wake (finish_generation / prepare_for_generation do not
    touch the gate): drafts held at the wake, continuations first, then the held drafts."""

    async def main():
        g = _Gen()
        slots = es.install([g], 3)
        t = [asyncio.ensure_future(_drain(g, _data(f"d{i}", False))) for i in range(2)]  # in flight at stand-down
        await asyncio.sleep(0.01)
        slots.hold_drafts()  # the wake
        t += [asyncio.ensure_future(_drain(g, _data(f"n{i}", False))) for i in range(2)]
        t += [asyncio.ensure_future(_drain(g, _data("c0", True)))]
        await slots.wait_cont_arrivals(1, timeout_s=5)
        await asyncio.sleep(0.01)
        assert g.order == ["d0", "d1", "c0"] and slots.used == 3
        slots.open_drafts()
        g.release.set()
        await asyncio.gather(*t)
        assert g.order[3:] == ["n0", "n1"] and slots.used == 0

    asyncio.run(main())


def test_atomic_n_slot_hold_has_top_priority_and_no_hold_and_wait():
    async def main():
        g = _Gen()
        slots = es.install([g], 4)
        t = [asyncio.ensure_future(_drain(g, _data(f"d{i}", False))) for i in range(3)]
        await asyncio.sleep(0.01)
        t += [asyncio.ensure_future(_drain(g, _data("d3", False)))]
        await asyncio.sleep(0.01)
        assert slots.used == 4
        got = []

        async def verify():
            async with slots.hold(3):
                got.append(slots.used)

        v = asyncio.ensure_future(verify())
        t += [asyncio.ensure_future(_drain(g, _data("d4", False)))]
        await asyncio.sleep(0.01)
        assert not got and slots.used == 4  # waits for 3 slots at once, holding none meanwhile
        g.release.set()
        await asyncio.gather(v, *t)
        assert got and got[0] >= 3 and slots.used == 0
        # the verify block ran before the later draft d4 was admitted
        assert g.order[-1] == "d4"

    asyncio.run(main())


def test_hold_released_when_the_wake_raises():
    """S1: drafts held at a wake that raises are released by the stage's finally (modelled here)."""

    async def main():
        g = _Gen()
        slots = es.install([g], 2)
        try:
            try:
                slots.hold_drafts()
                raise RuntimeError("refit aborted")
            finally:
                slots.open_drafts()
        except RuntimeError:
            pass
        g.release.set()
        await _drain(g, _data("d0", False))
        assert g.order == ["d0"] and slots.used == 0

    asyncio.run(main())


def test_capacity_rule():
    mc = {"max_requests": 128, "tensor_model_parallel_size": 2, "pipeline_model_parallel_size": 1}
    assert es.capacity_of(mc, 16) == (1024, 128, 8)
    mc["max_requests"] = 4096
    try:
        es.capacity_of(mc, 16)
    except ValueError:
        pass
    else:
        raise AssertionError("capacity above the actor's max_concurrency must raise")
    es.require_least_loaded({"enable_prefix_caching": False})
    es.require_least_loaded({"enable_prefix_caching": True, "prefix_caching_coordinator_policy": "load_balanced"})
    for pol in (None, "longest_prefix", "first_prefix_block"):
        cfg = {"enable_prefix_caching": True}
        if pol:
            cfg["prefix_caching_coordinator_policy"] = pol
        try:
            es.require_least_loaded(cfg)
        except ValueError:
            pass
        else:
            raise AssertionError(f"prefix-affinity routing {pol} must be refused")


def test_engine_reservation_check():
    alloc = types.SimpleNamespace(pool_size=1 + 128 * 40, paused_limit=0)
    eng = types.SimpleNamespace(context=types.SimpleNamespace(kv_block_allocator=alloc, max_requests=128,
                                                              max_kv_block_count=40))
    assert es.check_engine_reservation(eng)["kv_blocks_needed"] == 5120
    alloc.pool_size -= 1
    try:
        es.check_engine_reservation(eng)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a KV pool below max_requests full-length rows must raise")
    # tree_spec's private node blocks leave the pool for good (review T2): they count against the reservation
    alloc.pool_size += 1
    assert es.check_engine_reservation(eng, reserved_blocks=0)["kv_blocks_reserved"] == 0
    alloc.pool_size += 47
    assert es.check_engine_reservation(eng, reserved_blocks=47)["kv_blocks_usable"] == 5120
    try:
        es.check_engine_reservation(eng, reserved_blocks=48)
    except RuntimeError:
        pass
    else:
        raise AssertionError("reserved node blocks must count against the full-length reservation")


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("PASS", n)


# ---- stand-down cut at the gate (cut_waiting_drafts) and the counted stop (draft_holding) ----


def _tdata(tag, cont):
    import torch

    d = _data(tag, cont)
    d["input_ids"] = torch.tensor([[5, 6, 7, 0]])
    d["input_lengths"] = torch.tensor([3])
    return d


async def _collect(gen, data):
    return [item async for item in gen.generate_async(data)]


def test_cut_ends_waiting_drafts_unsent_and_counts_only_granted_drafts():
    """Drafts > capacity, then the stand-down cut: waiting drafts end at the gate with zero tokens (never sent), the
    granted ones are still held (counted by draft_holding until they return), a waiting continuation keeps its place,
    and no slot leaks."""

    async def main():
        g = _Gen()
        slots = es.install([g], 2)
        tasks = [asyncio.ensure_future(_collect(g, _tdata(f"d{i}", False))) for i in range(5)]
        await asyncio.sleep(0.01)
        assert g.live == 2 and slots.draft_holding == 2
        cont = asyncio.ensure_future(_collect(g, _tdata("c0", True)))
        await asyncio.sleep(0.01)
        assert slots.cut_waiting_drafts() == 3
        await asyncio.sleep(0.01)
        cut = [t.result() for t in tasks if t.done()]
        assert len(cut) == 3 and all(r[0][1]["ts_draft"][0]["cut"] == 1 for r in cut)
        assert all(int(r[0][1]["generation_lengths"][0]) == 0 for r in cut)
        assert g.order == ["d0", "d1"] and slots.draft_holding == 2  # the cut drafts were never sent
        g.release.set()
        await asyncio.gather(*tasks, cont)
        assert slots.draft_holding == 0 and slots.used == 0 and g.order[-1] == "c0"

    asyncio.run(main())


def test_a_draft_reaching_the_gate_after_the_cut_is_never_sent_until_the_next_wake():
    """The counted stop's premise: after the cut no ungranted draft reaches an engine, however late it arrives (here
    after 0.6 s of an empty engine), until the next wake's hold_drafts ends cut mode."""

    async def main():
        g = _Gen()
        g.release.set()
        slots = es.install([g], 4)
        slots.cut_waiting_drafts()
        await asyncio.sleep(0.6)
        out = await _collect(g, _tdata("late", False))
        assert g.order == [] and out[0][1]["ts_draft"][0]["key"] is None and slots.draft_holding == 0
        assert slots.cut_at_gate == 1
        c = await _collect(g, _tdata("c", True))  # continuations are never cut
        assert g.order == ["c"] and c[0][1]["tag"] == ["c"]
        slots.hold_drafts()
        slots.open_drafts()
        await _collect(g, _tdata("next", False))
        assert g.order == ["c", "next"]

    asyncio.run(main())


def test_cancelled_draft_waiter_and_cancelled_holder_leak_nothing():
    async def main():
        g = _Gen()
        slots = es.install([g], 1)
        holder = asyncio.ensure_future(_collect(g, _data("d0", False)))
        waiter = asyncio.ensure_future(_collect(g, _data("d1", False)))
        await asyncio.sleep(0.01)
        assert slots.draft_holding == 1
        waiter.cancel()
        holder.cancel()
        await asyncio.gather(holder, waiter, return_exceptions=True)
        assert slots.draft_holding == 0 and slots.used == 0 and slots._live == 0

    asyncio.run(main())


def test_granted_draft_calls_carry_ids_continuations_do_not_and_returns_clear_them():
    async def main():
        seen = []

        class G:
            async def generate_async(self, data, greedy=False):
                seen.append(data.get("nl_call_id"))
                yield 0, data

        g = G()
        slots = es.install([g], 4)
        await _drain(g, _data("d", False))
        await _drain(g, _data("c", True))
        assert seen[0] == [1] and seen[1] is None
        assert slots.draft_calls == set() and slots.engine_done == set()
        cid = slots.new_draft_call()
        slots.note_engine_done([cid, 999])  # unknown ids ignored
        assert slots.drafts_not_engine_done() == 0 and slots.engine_done == {cid}
        slots.end_draft_call(cid)
        assert slots.draft_calls == set() and slots.engine_done == set()

    asyncio.run(main())
