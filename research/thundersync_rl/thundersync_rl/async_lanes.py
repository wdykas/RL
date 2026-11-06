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
"""Lanes-k async GRPO baseline (``thundersync.async_lanes``; research/lanesk/README.md).

Fixed-age-k asynchronous GRPO on the ThunderSync learner path, for matched speed baselines against exact
on-policy speculation (the same engine split, data, seed, reward, learner, loss and IS correction as the a1 arm
``rollout_lag=1``; only the lag differs).

The engine fleet is k engine lanes (``lanes.build_engine_lanes``), each one engine holding one weight version.
Version schedule (theta_j = the learner's weights during step j; the optimizer step ends step j):

* step 0 starts cohorts 0 .. k-1 on lanes 0 .. k-1 at theta_0 (pipeline fill: cohort c < k has age c);
* lane j % k generates cohort j + k from theta_j: as soon as its cohort j is complete (every row generated) and
  theta_j exists (the top of step j), the lane is refit to theta_j and starts cohort j + k. The refit runs on the
  learner between two training dispatches of step j (``_run_one_step(side_task=...)``: the learner loop takes a
  gate around each dispatch), so theta_j is the live weights (the update is applied at the end of step j);
* step j trains cohort j (age k for j >= k), streaming its rows into training as they finish on the lane.

So k cohorts are in flight, one per lane; nothing waits except where the data forces it (step j cannot finish
before cohort j's last row; lane j % k cannot refit before its cohort j is complete). k = 1 is the async1
version schedule (cohort j at theta_{j-1}, cohort 0 at theta_0) on this path: the a1 equivalence gate.

Ages and IS: every row of cohort j >= k is sampled from theta_{j-k} and trained at theta_j (the step-j training
forward), with the same token IS weights exp(log pi_theta_j - generation log-prob) as async1.

This module holds the bookkeeping (``AsyncSchedule``, ``AsyncLedger``; no torch, no Ray: unit-tested) and the
driver (``train_async_lanes``, called by grpo_loop.thundersync_grpo_train).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------------------------------------------
# Refit hold (grpo_loop._run_one_step side_hold_rows)


def side_hold_limit_s(rows: int, busy_s: float, rows_done: int) -> Optional[float]:
    """Longest a lane-refit hold may keep the learner idle: the expected duration of the dispatch it blocks
    (``rows`` x the step's measured learner seconds per row so far).

    Holding pays only when the refit becomes ready sooner than that dispatch would finish. Otherwise the learner
    would sit idle through the cohort tail: Super B1024 a1l on 237dde26 idled ~35 s/step while its last 1/8 of rows
    (~45 s) finished. Returns None before any dispatch of the step has been timed; the hold is then unbounded, as
    before.
    """
    if rows_done <= 0 or busy_s <= 0.0:
        return None
    return max(0.0, float(rows) * float(busy_s) / float(rows_done))


async def bounded_side_hold(done: asyncio.Event, limit_s: Optional[float]) -> tuple[float, bool]:
    """Wait for ``done`` for at most ``limit_s`` seconds (None: no limit). Returns (seconds waited, expired)."""
    t0 = time.perf_counter()
    if done.is_set():
        return 0.0, False
    if limit_s is None:
        await done.wait()
        return time.perf_counter() - t0, False
    try:
        await asyncio.wait_for(done.wait(), timeout=limit_s)
    except asyncio.TimeoutError:
        return time.perf_counter() - t0, True
    return time.perf_counter() - t0, False


# ---------------------------------------------------------------------------------------------------------------
# Bookkeeping (pure)


@dataclass(frozen=True)
class AsyncSchedule:
    """The fixed-age-k version schedule of the lanes-k async baseline.

    ``k``: lanes = lag; ``max_steps``: steps of the run (cohort t is trained at step t); ``start``: the run's
    first step (theta_start is the version every lane holds at the start).
    """

    k: int
    max_steps: int
    start: int = 0

    def __post_init__(self):
        if int(self.k) < 1:
            raise ValueError(f"async lanes: k must be >= 1 (got {self.k})")

    def lane_of(self, cohort: int) -> int:
        """The lane that generates cohort ``cohort`` (and is refit after it)."""
        return cohort % self.k

    def initial_cohorts(self) -> list[int]:
        """Cohorts started at the top of the first step, all at theta_start (one per lane)."""
        return list(range(self.start, min(self.start + self.k, self.max_steps)))

    def version_of(self, cohort: int) -> int:
        """The learner step whose weights generate cohort ``cohort``."""
        return max(self.start, cohort - self.k)

    def age(self, cohort: int) -> int:
        """Steps between the generating weights and the trained-at weights (k in steady state)."""
        return cohort - self.version_of(cohort)

    def next_cohort(self, step: int) -> Optional[int]:
        """The cohort lane ``step % k`` starts during ``step`` once its cohort ``step`` is complete (None: past the
        run's end: the last k steps start nothing)."""
        t = step + self.k
        return t if t < self.max_steps else None


@dataclass
class LaneState:
    index: int
    version: int  # learner step whose weights the lane's engine holds
    cohort: Optional[int] = None  # cohort generating on the lane (None: free)
    t_free: Optional[float] = None  # when the lane's last cohort completed
    refits: int = 0
    idle_s: float = 0.0  # summed time between a cohort's completion and the next start


@dataclass
class AsyncLedger:
    """Runtime invariants of the schedule: a lane generates one cohort at a time, at the version the schedule
    names, and is never refit while it generates (a refit would move rows of a running cohort to other weights)."""

    schedule: AsyncSchedule
    lanes: list[LaneState] = field(default_factory=list)
    started: dict[int, tuple[int, int]] = field(default_factory=dict)  # cohort -> (lane, version)

    @classmethod
    def build(cls, schedule: AsyncSchedule) -> "AsyncLedger":
        return cls(schedule, [LaneState(i, schedule.start) for i in range(schedule.k)])

    def refit(self, lane: int, version: int) -> bool:
        """Lane ``lane`` takes theta_version; False if it already holds it (no refit needed)."""
        ln = self.lanes[lane]
        if ln.cohort is not None:
            raise RuntimeError(
                f"async lanes: lane {lane} still generates cohort {ln.cohort} at theta_{ln.version}; refitting it to "
                f"theta_{version} would move that cohort's rows to other weights"
            )
        if version < ln.version:
            raise RuntimeError(f"async lanes: lane {lane} would go back from theta_{ln.version} to theta_{version}")
        if version == ln.version:
            return False
        ln.version = version
        ln.refits += 1
        return True

    def start(self, lane: int, cohort: int, now: Optional[float] = None) -> int:
        """Cohort ``cohort`` starts on ``lane`` at the lane's version (checked against the schedule)."""
        ln = self.lanes[lane]
        s = self.schedule
        if cohort in self.started:
            raise RuntimeError(f"async lanes: cohort {cohort} already started")
        if lane != s.lane_of(cohort):
            raise RuntimeError(f"async lanes: cohort {cohort} belongs on lane {s.lane_of(cohort)}, not {lane}")
        if ln.cohort is not None:
            raise RuntimeError(f"async lanes: lane {lane} still generates cohort {ln.cohort}")
        if ln.version != s.version_of(cohort):
            raise RuntimeError(
                f"async lanes: cohort {cohort} must be generated at theta_{s.version_of(cohort)}, lane {lane} holds "
                f"theta_{ln.version}"
            )
        if now is not None and ln.t_free is not None:
            ln.idle_s += max(0.0, now - ln.t_free)
        ln.cohort = cohort
        self.started[cohort] = (lane, ln.version)
        return ln.version

    def finish(self, cohort: int, now: Optional[float] = None) -> None:
        """Every row of ``cohort`` is generated: its lane is free."""
        lane, _ = self.started[cohort]
        ln = self.lanes[lane]
        if ln.cohort != cohort:
            raise RuntimeError(f"async lanes: lane {lane} generates cohort {ln.cohort}, not {cohort}")
        ln.cohort = None
        ln.t_free = now

    def in_flight(self) -> list[int]:
        return sorted(ln.cohort for ln in self.lanes if ln.cohort is not None)

    def versions(self) -> list[int]:
        return [ln.version for ln in self.lanes]


# ---------------------------------------------------------------------------------------------------------------
# Generation on the lanes (one background event loop for every lane)


@dataclass
class CohortRun:
    """One cohort generating on a lane: a future per row (resolving to ``(i, state, metrics, None)``) and one for
    the whole cohort."""

    cohort: int
    lane: int
    version: int
    t_start: float
    rows: list[concurrent.futures.Future]
    done: Optional[concurrent.futures.Future] = None
    t_done: Optional[float] = None


class LaneLoop:
    """A background asyncio loop in its own thread: every lane's rollout coroutines run there, across steps (the
    driver's per-step event loops end with their step; a cohort generates for up to k steps). One thread for every
    lane, as the single-engine async1 path runs all rollouts on one loop (the tokenizer and the reward thread pool
    see the same concurrency); blocking reward calls run in ``asyncio.to_thread`` (rollouts.py)."""

    def __init__(self, name: str = "ts-async-lanes") -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name=name, daemon=True)
        self.thread.start()

    def submit(self, coro) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def close(self, timeout: float = 30.0) -> None:
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout)


async def generate_cohort(
    run: CohortRun,
    batch: Any,
    generation: Any,
    tokenizer: Any,
    task_to_env: Any,
    master_config: Any,
    on_done: Optional[Callable[[CohortRun], None]] = None,
) -> None:
    """Every rollout of ``batch`` (G per prompt) on ``generation`` at its current weights; row i resolves
    ``run.rows[i]`` when it finishes. The same per-row call as grpo_loop._generate_rollouts (async1), keyed by the
    step the rows are trained at (``run.cohort``) under THUNDERSYNC_DEBUG_KEYED_FRESH."""
    from nemo_rl.experience.rollouts import run_sample_multi_turn_rollout
    from thundersync_rl.grpo_loop import KeyedFreshGeneration, _initial_sample_state, keyed_fresh_enabled

    G = master_config.grpo.num_generations_per_prompt
    repeated = batch.repeat_interleave(G)
    keyed = keyed_fresh_enabled(master_config)
    failed: list[BaseException] = []

    async def one(i: int) -> None:
        try:
            gen = (
                KeyedFreshGeneration(generation, master_config.grpo.seed, run.cohort, i) if keyed else generation
            )
            state, metrics = await run_sample_multi_turn_rollout(
                sample_idx=i,
                initial_sample_state=_initial_sample_state(repeated, i),
                policy_generation=gen,
                tokenizer=tokenizer,
                task_to_env=task_to_env,
                max_seq_len=master_config.policy["max_total_sequence_length"],
                max_rollout_turns=master_config.grpo.max_rollout_turns,
                greedy=False,
            )
        except BaseException as exc:
            failed.append(exc)
            if not run.rows[i].done():
                run.rows[i].set_exception(exc)
            raise
        if not run.rows[i].done():  # (a failed sibling already failed the cohort's open rows)
            run.rows[i].set_result((i, state, metrics, None))

    try:
        await asyncio.gather(*(one(i) for i in range(repeated.size)))
    finally:
        run.t_done = time.perf_counter()
        for f in run.rows:  # a failed sibling: the others never resolve otherwise
            if not f.done():
                cause = f": {failed[0]!r}" if failed else ""
                f.set_exception(RuntimeError(f"async lanes: cohort {run.cohort} generation failed{cause}"))
    if on_done is not None:
        on_done(run)


# ---------------------------------------------------------------------------------------------------------------
# Driver


def train_async_lanes(
    policy: Any,
    lanes: list[Any],
    dataloader: Any,
    tokenizer: Any,
    loss_fn: Any,
    task_to_env: Any,
    master_config: Any,
    logger: Any = None,
    on_step: Optional[Callable[[list[dict[str, Any]]], None]] = None,
    checkpointer: Any = None,
    grpo_save_state: Any = None,
    resume_path: Optional[str] = None,
) -> list[dict[str, Any]]:
    """The lanes-k async loop (module docstring). ``lanes``: k engine lanes (``lanes.build_engine_lanes``; k = 1:
    the setup engine alone), each holding theta_start.

    Checkpoints (``checkpointing.enabled``, ``save_period``, ``checkpoint_must_save_by``): saved after step s at the
    boundary s + 1 (learner weights + optimizer, the data position of batch s + 1). The run itself does not drain: the
    cohorts s + 1 .. s + k keep generating on the lanes. A resume at boundary b refills the pipeline like step 0:
    cohorts b .. b + k - 1 at theta_b (ages 0 .. k - 1), then age k again. A baseline needs no bitwise trajectory
    across a resume (user 10-28); the resumed run trains the same batches in the same order."""
    import os

    from thundersync_rl.run_state import (
        Boundaries,
        check_fingerprint,
        config_fingerprint,
        load_state,
        save_state,
    )

    from thundersync_rl import grpo_loop as gl
    from thundersync_rl import leak_probe
    from thundersync_rl.checks_sched import STEP_TOP_KEYS, step_decomposition
    from thundersync_rl.envflag import env_flag
    from thundersync_rl.health import strict as health_strict
    from thundersync_rl.lanes import refit_lane
    from thundersync_rl.run_state import PromptStream
    from thundersync_rl.sentinel import WeightSentinel
    from thundersync_rl.streaming import StreamingLearner

    ts_cfg = master_config.thundersync
    grpo_cfg = master_config.grpo
    k = int(ts_cfg.rollout_lag)
    if len(lanes) != k:
        raise ValueError(f"async lanes: rollout_lag={k} needs {k} engine lanes, got {len(lanes)}")
    for name in ("THUNDERSYNC_CHECK_EVERY", "THUNDERSYNC_MISMATCH_CHECK"):
        if env_flag(name):
            # The a1 / a-k arms run without the exactness checks (stale by design); a check forward here would sit
            # on the critical path unaccounted for.
            raise ValueError(f"async lanes: {name} is not built on this path (the async arms run CHECKS=0)")
    gl.fp16_range_metrics(policy, master_config)  # turns the learner's fp16 range counter on (as the main loop)
    adv_fn = gl.make_group_advantage_fn(master_config)
    learner = StreamingLearner(policy)
    max_steps = int(grpo_cfg.max_num_steps)
    ckpt_cfg = gl._section(master_config, "checkpointing")
    ckpt_on = checkpointer is not None and bool(ckpt_cfg.get("enabled"))
    resume = load_state(resume_path) if resume_path is not None else None
    if resume is not None:
        if not ckpt_on:
            raise RuntimeError("resume_path given but checkpointing is not enabled")
        check_fingerprint(resume["fingerprint"], config_fingerprint(master_config))
        if (resume.get("async_lanes") or {}).get("k") not in (None, k):
            raise RuntimeError(f"async lanes: checkpoint written with k={resume['async_lanes']['k']}, this run k={k}")
    start_step = int(resume["next_step"]) if resume is not None else 0
    bounds = Boundaries(max_steps=max_steps, save_period=int(ckpt_cfg.get("save_period") or 0) if ckpt_on else 0)
    timeout = None
    if ckpt_on and ckpt_cfg.get("checkpoint_must_save_by"):
        from nemo_rl.utils.timer import TimeoutChecker

        timeout = TimeoutChecker(timeout=ckpt_cfg["checkpoint_must_save_by"], fit_last_save_time=True)
        timeout.start_iterations()
    sched = AsyncSchedule(k, max_steps, start=start_step)
    ledger = AsyncLedger.build(sched)
    sentinel = WeightSentinel(policy, enabled=ts_cfg.weight_sentinel)
    gl.check_learner_layout(policy, False)
    gl.check_train_mem_cap(policy, master_config)
    keyed = gl.keyed_fresh_enabled(master_config)
    for lane in lanes:
        lane.generation.prepare_for_generation()
        if keyed:
            gl.install_keyed_fresh(lane.generation)
        sentinel.check(lane.generation, f"initial weights (async lane {lane.index}, step 0)")
        if not keyed and getattr(ts_cfg, "plain_engine_sampler", "engine") == "wide":
            gl.install_plain_wide_sampler(lane.generation, master_config, segment=0, lane=lane.index)
    stream = PromptStream(
        dataloader, max_epochs=int(grpo_cfg.max_num_epochs or 1), state=resume["data"] if resume is not None else None
    )
    batches: dict[int, Any] = {}
    data_states: dict[int, Any] = {}  # cohort -> the stream state that yields its batch (the checkpoint's data position)
    next_c = [start_step]

    def fetch(upto: int) -> None:
        # Batches in data order (the same order as every arm): cohort c = the c-th batch of the stream.
        while next_c[0] <= upto:
            tb = next(stream, None)
            if tb is None:
                return
            batches[next_c[0]] = tb.batch
            data_states[next_c[0]] = tb.state
            next_c[0] += 1

    def batch_digest(b: Any) -> int:
        import zlib

        h = 0
        for i in range(b.size):
            ids = [int(t) for m in b["message_log"][i] for t in (
                m["token_ids"].tolist() if hasattr(m["token_ids"], "tolist") else m["token_ids"])]
            h = zlib.crc32(repr(ids).encode(), h)
        return h

    def save_boundary(next_step: int, final: bool) -> str:
        """Checkpoint at the boundary before ``next_step`` (learner state + data position; lanes keep running)."""
        t = time.perf_counter()
        if next_step not in data_states and not final:
            fetch(next_step)
        data_state = data_states.get(next_step) or stream.state()
        if grpo_save_state is not None:
            grpo_save_state.total_steps = next_step
            grpo_save_state.current_step = next_step
            grpo_save_state.current_epoch = int(data_state["epoch"])
            grpo_save_state.consumed_samples = next_step * int(grpo_cfg.num_prompts_per_step)
        info_d = vars(grpo_save_state) if grpo_save_state is not None else {"total_steps": next_step}
        checkpointer.finalize_pending()
        path = checkpointer.init_tmp_checkpoint(next_step, info_d, master_config)
        policy.save_checkpoint(
            weights_path=os.path.join(path, "policy", "weights"),
            optimizer_path=os.path.join(path, "policy", "optimizer"),
            tokenizer_path=os.path.join(path, "policy", "tokenizer"),
            is_final_checkpoint=final,
        )
        if data_state.get("dl") is not None:
            import torch

            torch.save(data_state["dl"], os.path.join(path, "train_dataloader.pt"))
        save_state(
            path,
            {
                "next_step": next_step,
                "data": data_state,
                "seeds": {},
                "policy_state": {},
                "pending_rollouts": None,
                "aep": None,
                "lanes": None,
                # lanes-k async: no cohort is persisted; the resume refills the pipeline at next_step.
                "async_lanes": {"k": k, "in_flight_dropped": sorted(c for c in runs if c >= next_step)},
                "fingerprint": config_fingerprint(master_config),
            },
        )
        checkpointer.begin_finalization(path, wait_fn=policy.finalize_async_save)
        print(f"[async lanes] checkpoint {path} (next step {next_step}) in {time.perf_counter() - t:.1f} s", flush=True)
        return path


    runner = LaneLoop()
    runs: dict[int, CohortRun] = {}
    lock = threading.Lock()  # ledger updates from the lane loop (completion) and the step's refit thread

    def on_done(run: CohortRun) -> None:
        with lock:
            ledger.finish(run.cohort, run.t_done)

    def start(cohort: int) -> None:
        lane = lanes[sched.lane_of(cohort)]
        G = grpo_cfg.num_generations_per_prompt
        now = time.perf_counter()
        with lock:
            version = ledger.start(lane.index, cohort, now)
        b = batches[cohort]
        run = CohortRun(
            cohort, lane.index, version, now, [concurrent.futures.Future() for _ in range(b.size * G)]
        )
        runs[cohort] = run
        run.done = runner.submit(
            generate_cohort(run, b, lane.generation, tokenizer, task_to_env, master_config, on_done)
        )

    history: list[dict[str, Any]] = []
    if resume is not None:
        print(f"[async lanes] resumed at step {start_step}: cohorts {sched.initial_cohorts()} refill the pipeline at "
              f"theta_{start_step}", flush=True)
    fetch(start_step + k - 1)
    for c in sched.initial_cohorts():
        if c in batches:
            start(c)  # every lane holds theta_start (setup's refit / the lane build / the resume): no refit
    step = start_step
    stopped = False
    try:
        while step < max_steps and step in runs:
            t0 = time.perf_counter()
            run = runs.pop(step)
            batch = batches.pop(step)
            nxt = sched.next_cohort(step)
            if nxt is not None:
                fetch(nxt)
                if nxt not in batches:
                    nxt = None  # data exhausted
            lane = lanes[sched.lane_of(step)]
            ready_at_top = run.done.done()
            in_flight = len(ledger.in_flight())
            info: dict[str, float] = {}

            def refit_and_start(lane=lane, step=step, nxt=nxt) -> None:
                # Learner gated (no dispatch in flight); the lane's cohort `step` is complete.
                t_r = time.perf_counter()
                lane.generation.finish_generation()
                with lock:
                    moved = ledger.refit(lane.index, step)
                if moved:
                    refit_lane(policy, lane)
                    sentinel.check(lane.generation, f"theta_{step} (async lane {lane.index})")
                info["async/refit_s"] = time.perf_counter() - t_r
                info["async/refit"] = float(moved)
                start(nxt)
                info["async/next_start_s"] = time.perf_counter() - t0

            async def side(gate: asyncio.Lock, run=run, nxt=nxt) -> dict[str, float]:
                await asyncio.wrap_future(run.done)  # raises if the cohort's generation failed
                info["async/cohort_done_s"] = max(0.0, (run.t_done or t0) - t0)
                if nxt is None:
                    return info
                t_g = time.perf_counter()
                async with gate:
                    info["async/gate_wait_s"] = time.perf_counter() - t_g
                    await asyncio.to_thread(refit_and_start)
                return info

            def source(run=run):
                return [asyncio.wrap_future(f) for f in run.rows]

            policy.prepare_for_training()
            metrics = asyncio.run(
                gl._run_one_step(
                    step=step,
                    batch=batch,
                    learner=learner,
                    policy_generation=lane.generation,
                    tokenizer=tokenizer,
                    task_to_env=task_to_env,
                    loss_fn=loss_fn,
                    master_config=master_config,
                    ts_cfg=ts_cfg,
                    adv_fn=adv_fn,
                    rollout_source=source,
                    side_task=side,
                    # Hold new training dispatches over the cohort's last 1/8 of rows when a lane refit follows (the
                    # refit then starts at completion: Super a1l / a2 lost up to 34 s per restart waiting on a dispatch).
                    side_hold_rows=(max(1, len(run.rows) // 8) if nxt is not None else None),
                    # ... and cap dispatches at 4 groups while that refit is pending (Super a2 waited 20-38 s behind
                    # one backlog dispatch on steps whose cohort was not ready at the top).
                    side_dispatch_rows=(4 * int(grpo_cfg.num_generations_per_prompt) if nxt is not None else None),
                )
            )
            metrics.pop("_drafts", None)
            metrics.pop("_side", None)
            gen_s = (run.t_done - run.t_start) if run.t_done is not None else float("nan")
            metrics.update(info)
            metrics.update(
                {
                    "async/k": float(k),
                    "async/age": float(sched.age(step)),
                    "async/cohort_version": float(run.version),
                    "async/lane": float(run.lane),
                    "async/in_flight_at_top": float(in_flight),
                    "async/cohort_ready_at_top": float(ready_at_top),
                    # Generation wall time of the cohort trained now (started k steps ago in steady state).
                    "async/cohort_gen_s": gen_s,
                    "async/lane_idle_s_total": float(sum(ln.idle_s for ln in ledger.lanes)),
                    "async/lane_refits_total": float(sum(ln.refits for ln in ledger.lanes)),
                    "time/refit": info.get("async/refit_s", 0.0),
                    "time/verify": 0.0,
                }
            )
            metrics["time/iteration"] = time.perf_counter() - t0
            if env_flag("THUNDERSYNC_STEP_DIGEST"):
                ranks = gl._run_workers(policy, "resume_state_digest")
                metrics["digest/params"] = "|".join(r["params"] for r in ranks)
                metrics["digest/optimizer"] = "|".join(str(r.get("optimizer")) for r in ranks)
            if leak_probe.enabled():
                groups = {"learner": policy.worker_group}
                groups.update({f"engine{ln.index}": ln.generation._policy.worker_group for ln in lanes})
                metrics.update(
                    leak_probe.collect(
                        groups, [], [os.environ.get("THUNDERSYNC_RUN_DIR")],
                        history_len=len(history) + 1,
                    )
                )
            metrics.update(sentinel.pop_stats())
            metrics.update(gl.fp16_range_metrics(policy, master_config))
            gl._reforward_frac(metrics, policy)
            metrics["step"] = step
            metrics["data/epoch"] = float(stream.epoch)
            metrics["async/batch_digest"] = float(batch_digest(batch))
            metrics["async/resumed_at"] = float(start_step)
            # The refit runs inside the step body here (between dispatches), never at its top.
            metrics.update(step_decomposition(metrics, STEP_TOP_KEYS))
            history.append(metrics)
            gl._log_step(
                step, metrics, logger, history, master_config.policy.get("train_micro_batch_size"),
                {"verify_yield": False},
            )
            if on_step is not None:
                on_step(history)
            if health_strict() and metrics.get("health/warn"):
                raise RuntimeError(f"THUNDERSYNC_HEALTH_STRICT: step {step} {metrics['health/flags']}")
            step += 1
            last = step >= max_steps or step not in runs
            stop_now = False
            if timeout is not None:
                timeout.mark_iteration()
                stop_now = bool(timeout.check_save()) and not last
            if ckpt_on and (bounds.saves_after(step - 1) or last or stop_now):
                save_boundary(step, final=last)
                if stop_now:
                    print(f"[async lanes] checkpoint_must_save_by due: stopping after the checkpoint before step {step}",
                          flush=True)
                    stopped = True
                    break
                stop_at = int(os.environ.get("THUNDERSYNC_DEBUG_STOP_AT", "0") or 0)
                if stop_at > 0 and step == stop_at and not last:
                    checkpointer.shutdown()
                    print(f"[async lanes] debug stop after step {step} (checkpoint durable)", flush=True)
                    os._exit(0)
        leftover = sorted(runs)
        if leftover and not stopped:
            raise RuntimeError(f"async lanes: cohorts {leftover} were started but never trained")
    finally:
        for r in runs.values():
            if r.done is not None:
                r.done.cancel()
        runner.close()
        if checkpointer is not None:
            checkpointer.shutdown()
    return history
