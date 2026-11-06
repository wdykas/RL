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
"""The lanes-k async driver (async_lanes.train_async_lanes) end to end on fakes (no GPU, no Ray, no torch).

The step function, the learner, the engine lanes, the refit and the rollouts are fakes that keep the contracts of
the real ones (grpo_loop._run_one_step with rollout_source / side_task; lanes.refit_lane; the engine samples at the
weights it holds). Checked over k = 1, 2, 3, 4:

- every row of cohort t is sampled at theta_{max(0, t - k)} on lane t % k (k = 1: the async1 schedule);
- a lane is refit only between learner dispatches (the gate) and never while it generates;
- the refit sends the live theta_t (the update is applied after the side task, at the end of step t);
- cohort t + k starts during step t (k cohorts in flight) and every step trains exactly its cohort's rows;
- a failed rollout fails the step (no hang), and the lane loop is stopped.
"""

from __future__ import annotations

import asyncio
import random
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest
import thundersync_rl
import thundersync_rl.lanes as lanes_mod
from thundersync_rl import async_lanes

G = 4


class _Batch:
    def __init__(self, idx, prompts=3):
        self.idx, self.size = idx, prompts

    def __getitem__(self, key):
        assert key == "message_log"
        return [[{"token_ids": [self.idx, j]}] for j in range(self.size)]

    def repeat_interleave(self, g):
        return SimpleNamespace(size=self.size * g, idx=self.idx)


class _Engine:
    """One engine lane: holds one weight version; samples report it."""

    def __init__(self, index):
        self.index = index
        self.version = 0
        self.active = 0
        self.lock = threading.Lock()
        self._policy = SimpleNamespace(worker_group=None)

    def prepare_for_generation(self):
        pass

    def finish_generation(self):
        pass


class _World:
    def __init__(self, k, n_steps, fail_at=None, seed=0):
        self.k, self.n = k, n_steps
        self.rng = random.Random(seed)
        self.live = 0  # the learner's live weights: theta_step during step `step`
        self.in_dispatch = False
        self.refits = []  # (lane, version, step)
        self.samples = {}  # (cohort, row) -> (version, lane)
        self.started_during = {}  # cohort -> the step running when it started
        self.step = -1
        self.trained = {}  # step -> sorted rows trained
        self.fail_at = fail_at
        self.saved = {}  # checkpoint path -> state

    # -- fakes -----------------------------------------------------------------------------------------------
    def refit_lane(self, policy, lane, lookahead=False):
        eng = lane.generation
        assert not self.in_dispatch, "refit during a learner dispatch"
        assert eng.active == 0, f"lane {lane.index} refit while generating"
        eng.version = self.live
        self.refits.append((lane.index, self.live, self.step))

    async def rollout(self, sample_idx, initial_sample_state, policy_generation, **kw):
        eng = policy_generation
        cohort = initial_sample_state["cohort"]
        with eng.lock:
            eng.active += 1
        v = eng.version
        try:
            await asyncio.sleep(self.rng.uniform(0.0, 0.004))
            assert eng.version == v, "engine weights moved under a generating row"
            if self.fail_at == (cohort, sample_idx):
                raise RuntimeError("injected rollout failure")
            self.samples[(cohort, sample_idx)] = (v, eng.index)
            return {"cohort": cohort, "version": v, "lane": eng.index}, {"assistant_tokens": 1}
        finally:
            with eng.lock:
                eng.active -= 1

    async def run_one_step(self, step, batch, rollout_source, side_task, side_hold_rows=None, **kw):
        """The contract of grpo_loop._run_one_step(rollout_source=, side_task=): rows stream in, every dispatch
        holds the gate, the side task ends before the optimizer step (which moves the live weights)."""
        self.step = step
        assert self.live == step
        # a lane refit follows (cohort step + k exists) <=> the learner holds dispatches over the last 1/8 of rows
        self.holds = getattr(self, "holds", {})
        self.holds[step] = side_hold_rows
        gate = asyncio.Lock()
        side = asyncio.create_task(side_task(gate))
        rows = []
        for fut in asyncio.as_completed(rollout_source()):
            i, state, metrics, anchors = await fut
            assert state["cohort"] == step and anchors is None
            rows.append(i)
            if len(rows) % G == 0:  # one dispatch per completed group
                async with gate:
                    self.in_dispatch = True
                    await asyncio.sleep(0.002)
                    self.in_dispatch = False
        side_res = await side
        self.trained.setdefault(step, sorted(rows))
        self.live += 1  # optimizer step: theta_{step+1}
        return {"_drafts": None, "_side": side_res, "reward": 0.0, "time/rollout": 0.0, "time/step_total": 0.0,
                "time/post_rollout_to_step": 0.0}


def _fake_modules(monkeypatch, world: _World):
    monkeypatch.setenv("THUNDERSYNC_LEAK_PROBE", "0")  # (on by default; it reads the real worker groups)
    for v in ("THUNDERSYNC_MISMATCH_CHECK", "THUNDERSYNC_CHECK_EVERY", "THUNDERSYNC_STEP_DIGEST"):
        monkeypatch.delenv(v, raising=False)
    gl = types.ModuleType("thundersync_rl.grpo_loop")
    gl.fp16_range_metrics = lambda policy, cfg: {}
    gl.make_group_advantage_fn = lambda cfg: None
    gl.check_learner_layout = lambda policy, vy: None
    gl.check_train_mem_cap = lambda policy, cfg: None
    gl.keyed_fresh_enabled = lambda cfg: False
    gl.install_keyed_fresh = lambda gen: None
    gl.install_plain_wide_sampler = lambda gen, cfg, segment=0, lane=0: None
    gl._run_workers = lambda policy, name: []
    gl._reforward_frac = lambda m, p: None
    gl._log_step = lambda *a, **k: None
    gl.KeyedFreshGeneration = None
    gl._initial_sample_state = lambda rep, i: {"cohort": rep.idx}

    async def run_one_step(**kw):
        return await world.run_one_step(**kw)

    gl._run_one_step = run_one_step
    gl._section = lambda cfg, name: dict(getattr(cfg, name, None) or {})
    monkeypatch.setitem(sys.modules, "thundersync_rl.grpo_loop", gl)
    monkeypatch.setattr(thundersync_rl, "grpo_loop", gl, raising=False)

    sent = types.ModuleType("thundersync_rl.sentinel")

    class WeightSentinel:
        def __init__(self, policy, enabled=True):
            self.checks = 0

        def check(self, gen, label, **kw):
            self.checks += 1

        def pop_stats(self):
            return {"sentinel/checks": float(self.checks)}

    sent.WeightSentinel = WeightSentinel
    monkeypatch.setitem(sys.modules, "thundersync_rl.sentinel", sent)
    st = types.ModuleType("thundersync_rl.streaming")
    st.StreamingLearner = lambda policy: SimpleNamespace(dp_size=1)
    monkeypatch.setitem(sys.modules, "thundersync_rl.streaming", st)
    rs = types.ModuleType("thundersync_rl.run_state")

    class PromptStream:
        def __init__(self, dl, max_epochs=1, **kw):
            self.it, self.epoch = iter(dl), 0

        def __iter__(self):
            return self

        def state(self):
            return {"epoch": self.epoch, "dl": None, "carry": []}

        def __next__(self):
            b = next(self.it)
            return SimpleNamespace(batch=b, state={"epoch": 0, "dl": None, "carry": [], "next_batch": b.idx})

    rs.PromptStream = PromptStream

    class Boundaries:
        def __init__(self, max_steps, save_period=0, **kw):
            self.max_steps, self.save_period = max_steps, save_period

        def saves_after(self, step):
            b = step + 1
            return self.save_period > 0 and (b % self.save_period == 0 or b >= self.max_steps)

    rs.Boundaries = Boundaries
    rs.check_fingerprint = lambda saved, now: None
    rs.config_fingerprint = lambda cfg: {}
    rs.load_state = lambda path: world.saved[path]
    rs.save_state = lambda path, state: world.saved.__setitem__(path, state)
    monkeypatch.setitem(sys.modules, "thundersync_rl.run_state", rs)
    ro = types.ModuleType("nemo_rl.experience.rollouts")
    ro.run_sample_multi_turn_rollout = world.rollout
    for name in ("nemo_rl", "nemo_rl.experience"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "nemo_rl.experience.rollouts", ro)
    monkeypatch.setattr(lanes_mod, "refit_lane", world.refit_lane)


def _cfg(k, n):
    return SimpleNamespace(
        thundersync=SimpleNamespace(rollout_lag=k, weight_sentinel=True, plain_engine_sampler="wide"),
        grpo=SimpleNamespace(max_num_steps=n, num_generations_per_prompt=G, max_num_epochs=1, seed=42,
                             max_rollout_turns=1),
        policy={"max_total_sequence_length": 64, "train_micro_batch_size": 1},
    )


def _run(monkeypatch, k, n, **kw):
    world = _World(k, n, **kw)
    _fake_modules(monkeypatch, world)
    lanes = [SimpleNamespace(index=i, generation=_Engine(i)) for i in range(k)]
    # the start() of a cohort reads the step the driver is in: record it through the ledger hook
    orig_start = async_lanes.AsyncLedger.start

    def start(self, lane, cohort, now=None):
        world.started_during[cohort] = world.step
        return orig_start(self, lane, cohort, now)

    monkeypatch.setattr(async_lanes.AsyncLedger, "start", start)
    hist = async_lanes.train_async_lanes(
        policy=SimpleNamespace(prepare_for_training=lambda: None),
        lanes=lanes,
        dataloader=[_Batch(i) for i in range(n + 3)],
        tokenizer=None,
        loss_fn=None,
        task_to_env={},
        master_config=_cfg(k, n),
    )
    return world, hist


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_versions_lanes_and_overlap(monkeypatch, k):
    n = 10
    world, hist = _run(monkeypatch, k, n)
    assert len(hist) == n
    rows = 3 * G
    for t in range(n):
        assert world.trained[t] == list(range(rows))
        want_v = max(0, t - k)
        assert {world.samples[(t, i)] for i in range(rows)} == {(want_v, t % k)}, t
        assert hist[t]["async/age"] == t - want_v and hist[t]["async/cohort_version"] == want_v
        assert hist[t]["async/lane"] == t % k
    # k = 1: the async1 schedule (cohort t at theta_{t-1}, cohort 0 at theta_0)
    if k == 1:
        assert [world.samples[(t, 0)][0] for t in range(n)] == [0] + list(range(n - 1))
    # cohort t + k starts during step t (lane t % k right after its cohort t), the first k at the top of step 0
    for c in range(n):
        assert world.started_during[c] == (-1 if c < k else c - k), (c, world.started_during)
    # refits: lane t % k to the live theta_t during step t (none at step 0: lanes hold theta_0), and none in the
    # last k steps (they start no cohort)
    assert world.refits == [(t % k, t, t) for t in range(1, n - k)]
    assert all(h.get("async/refit", 0.0) == (1.0 if 1 <= s < n - k else 0.0) for s, h in enumerate(hist))
    assert world.holds == {t: (max(1, rows // 8) if t + k < n else None) for t in range(n)}


def test_failed_rollout_fails_the_step(monkeypatch):
    with pytest.raises(RuntimeError, match="injected rollout failure"):
        _run(monkeypatch, 2, 6, fail_at=(3, 5))


def test_wrong_lane_count_refused(monkeypatch):
    world = _World(2, 4)
    _fake_modules(monkeypatch, world)
    with pytest.raises(ValueError, match="needs 2 engine lanes"):
        async_lanes.train_async_lanes(SimpleNamespace(), [SimpleNamespace(index=0, generation=_Engine(0))], [],
                                      None, None, {}, _cfg(2, 4))


def test_checks_refused(monkeypatch):
    world = _World(2, 4)
    _fake_modules(monkeypatch, world)
    monkeypatch.setenv("THUNDERSYNC_MISMATCH_CHECK", "1")
    lanes = [SimpleNamespace(index=i, generation=_Engine(i)) for i in range(2)]
    with pytest.raises(ValueError, match="CHECKS=0"):
        async_lanes.train_async_lanes(SimpleNamespace(), lanes, [], None, None, {}, _cfg(2, 4))


def test_lane_loop_runs_across_event_loops():
    loop = async_lanes.LaneLoop("t")
    try:
        async def slow():
            await asyncio.sleep(0.01)
            return 7

        fut = loop.submit(slow())
        asyncio.run(asyncio.sleep(0))  # a step's event loop ends; the lane loop keeps the coroutine

        async def wait():
            return await asyncio.wrap_future(fut)

        assert asyncio.run(wait()) == 7
    finally:
        loop.close()
    t0 = time.perf_counter()
    loop.close()
    assert time.perf_counter() - t0 < 1.0


class _Ckpt:
    def __init__(self):
        self.paths = []

    def finalize_pending(self):
        pass

    def init_tmp_checkpoint(self, step, info, cfg):
        self.paths.append(f"step_{step}")
        return f"step_{step}"

    def begin_finalization(self, path, wait_fn=None):
        pass

    def shutdown(self):
        pass


@pytest.mark.parametrize("k", [1, 2, 3])
def test_checkpoint_and_resume_refill(monkeypatch, k):
    """Saves at every save_period boundary without draining (the run is unchanged); a resume at boundary b trains
    batches b.. in order and refills the pipeline: cohorts b .. b + k - 1 at theta_b, then age k again."""
    n, period = 10, 4
    world = _World(k, n)
    _fake_modules(monkeypatch, world)
    lanes = [SimpleNamespace(index=i, generation=_Engine(i)) for i in range(k)]
    cfg = _cfg(k, n)
    cfg.checkpointing = {"enabled": True, "save_period": period}
    saved_steps = []
    pol = SimpleNamespace(
        prepare_for_training=lambda: None,
        save_checkpoint=lambda **kw: saved_steps.append(world.live),
        finalize_async_save=lambda: None,
    )
    ck = _Ckpt()
    data = [_Batch(i) for i in range(n + 3)]
    hist = async_lanes.train_async_lanes(pol, lanes, data, None, None, {}, cfg, checkpointer=ck)
    assert ck.paths == ["step_4", "step_8", "step_10"] and saved_steps == [4, 8, 10]  # learner at theta_b
    assert [h["async/age"] for h in hist] == [min(t, k) for t in range(n)]  # saving did not drain the pipeline
    st = world.saved["step_4"]
    assert st["next_step"] == 4 and st["async_lanes"]["k"] == k and st["data"]["next_batch"] == 4
    # resume from step_4 in a fresh world (learner restored at theta_4, lanes hold theta_4)
    w2 = _World(k, n)
    _fake_modules(monkeypatch, w2)
    w2.saved = world.saved
    w2.live = 4
    lanes2 = [SimpleNamespace(index=i, generation=_Engine(i)) for i in range(k)]
    for ln in lanes2:
        ln.generation.version = 4
    data2 = [_Batch(i) for i in range(4, n + 3)]  # the stream resumes at batch 4 (PromptStream state)
    hist2 = async_lanes.train_async_lanes(pol, lanes2, data2, None, None, {}, cfg, checkpointer=_Ckpt(),
                                          resume_path="step_4")
    assert [h["step"] for h in hist2] == list(range(4, n))
    assert [h["async/age"] for h in hist2] == [min(t - 4, k) for t in range(4, n)]
    assert all(w2.samples[(t, 0)] == (max(4, t - k), t % k) for t in range(4, n))
    assert [h["async/batch_digest"] for h in hist2] == [h["async/batch_digest"] for h in hist[4:]]


# ---------------------------------------------------------------------------------------------------------------
# Bounded refit hold (grpo_loop side_hold_rows; Super B1024 a1l idled ~35 s/step behind an unbounded hold)


def test_side_hold_limit_is_the_blocked_dispatch_time():
    assert async_lanes.side_hold_limit_s(32, 0.0, 0) is None  # nothing timed yet this step: unbounded
    assert async_lanes.side_hold_limit_s(32, 5.0, 0) is None
    assert async_lanes.side_hold_limit_s(32, 0.0, 64) is None
    # Super B1024 a1l: ~93 s of learner work for 1024 rows -> a 32-row capped dispatch is ~2.9 s.
    assert async_lanes.side_hold_limit_s(32, 93.0, 1024) == pytest.approx(2.90625)
    assert async_lanes.side_hold_limit_s(64, 10.0, 64) == pytest.approx(10.0)


def test_bounded_hold_returns_when_the_refit_is_ready_first():
    async def main():
        done = asyncio.Event()
        asyncio.get_running_loop().call_later(0.05, done.set)
        return await async_lanes.bounded_side_hold(done, 2.0)

    waited, expired = asyncio.run(main())
    assert not expired
    assert 0.03 <= waited < 1.0


def test_bounded_hold_expires_at_the_limit_on_a_long_tail():
    async def main():
        done = asyncio.Event()
        asyncio.get_running_loop().call_later(5.0, done.set)  # the cohort tail outlasts the dispatch
        out = await async_lanes.bounded_side_hold(done, 0.1)
        return out, done.is_set()

    (waited, expired), was_set = asyncio.run(main())
    assert expired and not was_set
    assert 0.08 <= waited < 1.0  # idled one dispatch's worth, not the tail


def test_unbounded_hold_waits_for_the_refit():
    async def main():
        done = asyncio.Event()
        asyncio.get_running_loop().call_later(0.1, done.set)
        return await async_lanes.bounded_side_hold(done, None)

    waited, expired = asyncio.run(main())
    assert not expired and waited >= 0.08


def test_bounded_hold_with_the_refit_already_done_does_not_wait():
    async def main():
        done = asyncio.Event()
        done.set()
        return await async_lanes.bounded_side_hold(done, 0.0)

    waited, expired = asyncio.run(main())
    assert not expired and waited < 0.05
