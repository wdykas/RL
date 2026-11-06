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
"""Long-run support: releasable rank-0 draft tables, consumed-cohort release, closed-log-key
pruning, epoch-aligned boundaries, past-only policy state, the leak probe and its verdict."""

from __future__ import annotations

import math

import numpy as np
import pytest
import thundersync_rl.speculative as spec_mod
import torch
from thundersync_rl.draft_table import DraftTable, release_entries
from thundersync_rl.engine_q import HiddenLog
from thundersync_rl.leak_probe import leak_verdict, reduce_snapshots, theil_sen
from thundersync_rl.routes import RouteLog
from thundersync_rl.run_state import Boundaries, check_fingerprint

# ---------------------------------------------------------------------------
# DraftTable


def test_draft_table_is_index_stable_and_releasable():
    tok, lp, done = DraftTable(list), DraftTable(lambda: None), DraftTable(lambda: True)
    tok += [[1, 2], [3], []]
    lp += [[-0.1, -0.2], [-0.3], []]
    done += [True, False, False]
    assert len(tok) == 3 and tok[0] == [1, 2] and tok[-1] == []
    tok[2].extend([7, 8])  # frames append in place
    assert tok[2] == [7, 8]
    out = release_entries({"tokens": tok, "logprobs": lp, "done": done, "streams": {0: "s0", 1: "s1"}},
                          [0, 1], running=lambda i: i == 1)
    assert out == {"released": 1, "running": 1, "gone": 0, "tokens": 2}
    assert tok.live() == 2 and len(tok) == 3  # indices stay stable
    assert tok[0] == [] and lp[0] is None and done[0] is True  # released: empty values
    tok[0].append(5)  # a late frame of a released stream lands in a throwaway
    tok[0] = [9]  # writes to released entries are dropped
    assert tok[0] == [] and tok.live() == 2
    assert tok[1] == [3]
    tok += [[4]]
    assert len(tok) == 4 and tok[3] == [4]
    assert [len(t) for t in tok] == [0, 1, 2, 1]
    with pytest.raises(IndexError):
        tok[4]


# ---------------------------------------------------------------------------
# release_consumed (driver) against a rank-0 stub with the real worker method body


class _Rank0:
    """Stand-in for generation rank 0: draft tables, start/add, release (worker.release_drafts logic)."""

    def __init__(self):
        self.gen = 0
        self.start([])

    def start(self, prompts):
        self.tok, self.lp, self.done = DraftTable(list), DraftTable(lambda: None), DraftTable(lambda: True)
        self.streams, self.cancelled = {}, set()
        self.gen += 1
        return self.add(prompts)

    def add(self, prompts):
        base = len(self.tok)
        self.tok += [list(p) for p in prompts]
        self.lp += [[0.0] * len(p) for p in prompts]
        self.done += [False] * len(prompts)
        for j in range(len(prompts)):
            self.streams[base + j] = f"st{base + j}"
        return list(range(base, base + len(prompts)))

    def release(self, gen, idx):
        if gen != self.gen:
            return {"released": 0, "running": 0, "gone": len(idx), "tokens": 0, "live": 0}
        r = release_entries(
            {"tokens": self.tok, "logprobs": self.lp, "done": self.done, "streams": self.streams,
             "cancelled": self.cancelled},
            idx,
            lambda i: not self.done[i] and i in self.streams and i not in self.cancelled,
        )
        r["live"] = self.tok.live()
        return r


class _Remote:
    def __init__(self, fn):
        self.remote = fn


def test_release_consumed_frees_trained_cohorts_only(monkeypatch):
    monkeypatch.setattr(spec_mod.ray, "get", lambda x: x)
    r0 = _Rank0()
    pruned = []
    workers = [type("W", (), {"release_drafts": _Remote(r0.release)})()]
    group = type("G", (), {"workers": workers,
                           "run_all_workers_single_data": lambda self, m, **kw: pruned.append(kw) or [{"hidden_log": 3}]})()
    s = spec_mod.SpeculativeGeneration.__new__(spec_mod.SpeculativeGeneration)
    s.base = type("B", (), {"_policy": type("P", (), {"worker_group": group})()})()
    s._stream_owner, s._draft_gen = {}, 1
    # cohorts 1..3 drafted (anchor epochs: add next to each other), continuations of cohort 1 at step 1
    for c in (1, 2, 3):
        s._note_streams(c, r0.add([[c] * 5] * 4))
    s._note_streams(1, r0.add([[9] * 3] * 2))
    for i in s._stream_owner[1][1]:
        r0.done[i] = True
    r0.done[s._stream_owner[1][1][0]] = False  # one stream of cohort 1 still running (kept, retried)
    st = s.release_consumed(1)
    assert st["mem/rank0_released"] == 5 and st["mem/rank0_running_kept"] == 1
    assert st["mem/closed_keys_pruned"] == 3 and pruned == [{"lag": 2}]
    assert sorted(s._stream_owner) == [1, 2, 3] and len(s._stream_owner[1][1]) == 6
    assert r0.tok.live() == 4 + 4 + 1  # cohorts 2, 3 untouched + the running stream
    r0.done[s._stream_owner[1][1][0]] = True
    st = s.release_consumed(1)
    assert st["mem/rank0_released"] == 1 and 1 not in s._stream_owner
    # an older table generation (rank 0 replaced its tables at an epoch switch): nothing to call
    s._draft_gen = 2
    r0.gen = 2
    st = s.release_consumed(3)
    assert st["mem/rank0_released"] == 0 and s._stream_owner == {}


# ---------------------------------------------------------------------------
# Closed-key pruning of the engine logs


def test_hidden_log_prunes_closed_keys_after_the_lag():
    log = HiddenLog(chunk_rows=16)
    h = torch.arange(12, dtype=torch.float32).view(4, 3)
    log.record([(0, "a", 0), (1, "a", 1), (2, "b", 0)], h)
    assert log.pop("a", 2).shape == (2, 3)
    log.close("b")
    log.record([(3, "a", 2)], h)  # a late row of a closed key: dropped
    assert log.bookkeeping()["rows"] == 0.0 and log.bookkeeping()["closed_keys"] == 2.0
    assert log.prune_closed(lag=2) == 0  # closed in gen 0, now gen 1
    assert log.prune_closed(lag=2) == 2  # gen 2: forgotten
    assert log.bookkeeping()["closed_keys"] == 0.0


def test_route_log_prunes_closed_key_strings_and_never_reuses_ids():
    log = RouteLog(ring_rows=64)
    ids = [log.key_id(k) for k in ("a", "b", "c")]
    assert ids == [0, 1, 2]
    log.pop(["a", "b"], close=True)
    assert log.prune_closed(lag=2) == 0
    assert log.prune_closed(lag=2) == 2
    assert log.bookkeeping()["keys"] == 1.0
    assert log.key_id("d") == 3  # ids are not reused (the closed flags of 0, 1 stay)
    assert bool(log._closed[0]) and bool(log._closed[1]) and not log._closed[3]


# ---------------------------------------------------------------------------
# Boundaries at epoch starts; fingerprint of older checkpoints


def test_boundaries_offset_puts_saves_at_epoch_starts():
    # anchor_start 2, epoch length 4: epoch starts 2, 6, 10, ...; save every 8 steps -> 2, 10, 18
    b = Boundaries(max_steps=30, save_period=8, offset=2)
    saves = [s + 1 for s in range(30) if b.saves_after(s)]
    assert saves == [2, 10, 18, 26, 30]
    assert b.draft_limit(3) == 10 and b.draft_limit(10) == 18
    plain = Boundaries(max_steps=10, save_period=4)
    assert [s + 1 for s in range(10) if plain.saves_after(s)] == [4, 8, 10]


def test_fingerprint_ignores_keys_an_older_checkpoint_lacks():
    check_fingerprint({"grpo.seed": 1}, {"grpo.seed": 1, "thundersync.anchor_epoch": None})
    with pytest.raises(RuntimeError):
        check_fingerprint({"grpo.seed": 1, "thundersync.anchor_epoch": 3}, {"grpo.seed": 1, "thundersync.anchor_epoch": 4})


# ---------------------------------------------------------------------------
# Past-only policy state


def test_policy_state_round_trip():
    from thundersync_rl.null_bv import LogRatioCalibration
    from thundersync_rl.vit import PassPredictor

    s = spec_mod.SpeculativeGeneration.__new__(spec_mod.SpeculativeGeneration)
    s._null_calib, s.vit_predictor, s._pacer = LogRatioCalibration(0.1), PassPredictor(3), None
    s._null_calib.update([(10, 1.0, 3.0)])
    s.vit_predictor.observe(100, True)
    s.vit_predictor.end_step()
    st = s.policy_state()
    t = spec_mod.SpeculativeGeneration.__new__(spec_mod.SpeculativeGeneration)
    t._null_calib, t.vit_predictor, t._pacer = LogRatioCalibration(0.1), PassPredictor(3), None
    t.load_policy_state(st)
    assert t._null_calib.estimate() == s._null_calib.estimate() and t.vit_predictor.rate(100) == s.vit_predictor.rate(100)
    st["vit_predictor"]["steps"][0][99] = [0, 99]  # a deep copy: the saved state is independent
    assert 99 not in s.vit_predictor.steps[0]


# ---------------------------------------------------------------------------
# Leak probe and verdict


def test_theil_sen_is_robust_to_spikes():
    xs = list(range(50))
    ys = [2.0 * x for x in xs]
    ys[10] = 1e6
    assert abs(theil_sen(xs, [float(y) for y in ys]) - 2.0) < 1e-9


def _hist(fn, n=60):
    return [{"step": i, **fn(i)} for i in range(n)]


def test_leak_verdict_flat_vs_growing():
    rng = np.random.default_rng(0)
    flat = _hist(lambda i: {"leak/engine_rss_gb_max": 30 + rng.normal() * 0.3 + (5 if i < 5 else 0),
                            "leak/engine_rank0_draft_live_max": 3000 + 50 * (i % 3),
                            "leak/run_dir_gb": 0.01 * i, "leak/history_len": i + 1})
    v = leak_verdict(flat)
    assert v["ok"], v["failed"]
    grow = _hist(lambda i: {"leak/engine_rss_gb_max": 30 + 0.05 * i,  # 25 GB over 500 steps
                            "leak/engine_rank0_draft_live_max": 3000 + 120 * i})
    v = leak_verdict(grow)
    assert not v["ok"] and set(v["failed"]) == {"leak/engine_rss_gb_max", "leak/engine_rank0_draft_live_max"}
    superlin = _hist(lambda i: {"leak/run_dir_gb": 0.001 * i * i})
    assert not leak_verdict(superlin)["ok"]
    assert leak_verdict([])["ok"] is False  # no data is not a pass


def test_reduce_snapshots():
    out = reduce_snapshots("engine", [{"rss_gb": 2.0, "rank0_draft_live": 10.0}, {"rss_gb": 3.0}])
    assert out["leak/engine_rss_gb_max"] == 3.0 and out["leak/engine_rss_gb_sum"] == 5.0
    assert out["leak/engine_rank0_draft_live_max"] == 10.0 and "leak/engine_rank0_draft_live_sum" not in out
    assert all(math.isfinite(v) for v in out.values())


def test_leak_verdict_judges_private_rss_not_mapped_plasma():
    """Total RSS that grows only through mapped shared pages (Ray plasma) is reported, not judged (final50)."""
    hist = _hist(lambda i: {"leak/engine0_rss_gb_max": 30 + 1.0 * i, "leak/engine0_rss_anon_gb_max": 20.0,
                            "leak/engine0_rss_shmem_gb_max": 10 + 1.0 * i})
    v = leak_verdict(hist)
    assert v["ok"], v["failed"]
    hist = _hist(lambda i: {"leak/engine0_rss_gb_max": 30 + 1.0 * i, "leak/engine0_rss_anon_gb_max": 20 + 1.0 * i})
    assert leak_verdict(hist)["failed"] == ["leak/engine0_rss_anon_gb_max"]
    # without the split (older trees) the total RSS is still judged
    assert not leak_verdict(_hist(lambda i: {"leak/engine0_rss_gb_max": 30 + 1.0 * i}))["ok"]


def test_leak_verdict_plateau_passes_persistent_growth_fails():
    """Step-like growth that stops before the final half is a plateau; growth still present in the final half (linear,
    or a step up inside the final half) fails (leak investigation 10-21: final50b learner RssAnon max)."""
    rng = np.random.default_rng(1)
    k = "leak/learner_rss_anon_gb_max"
    # final50b (ts arm, 10-20) learner RssAnon max per step: 7.07 -> 7.5 (step 12) -> 8.1 (step 18), flat after
    final50b = [6.349, 7.043, 7.071, 7.072, 7.058, 7.091, 7.093, 7.064, 7.065, 7.071, 7.107, 7.075, 7.53, 7.514, 7.514,
                7.549, 7.515, 7.195, 8.023, 8.02, 8.184, 8.1, 8.124, 8.012, 8.115, 8.122, 8.075, 8.16, 8.131, 8.129,
                8.136, 8.13, 8.153, 8.169, 8.169, 8.139, 8.171, 8.168, 8.144, 8.112, 8.218, 8.136, 8.171, 8.131, 8.161,
                8.186, 8.132, 8.208, 8.132, 8.194]
    plateau = _hist(lambda i: {k: final50b[i]}, n=50)
    v = leak_verdict(plateau)
    assert v["ok"], v["failed"]
    # the whole-window slope alone would have failed it (the over-projection the rule removes)
    row = [r for r in v["rows"] if r[0] == k][0]
    assert row[4] > row[5]
    # linear: fails in every window
    lin = _hist(lambda i: {k: 7.0 + 0.02 * i + rng.normal() * 0.03}, n=50)
    assert leak_verdict(lin)["failed"] == [k]
    # a staircase still stepping in the final half fails
    stair = _hist(lambda i: {k: 7.0 + 0.6 * (i // 8)}, n=50)
    assert leak_verdict(stair)["failed"] == [k]
    # the final-half rule only relaxes the whole-window test: a series the whole-window slope passes still passes
    # (a single late jump with few points after it is invisible to Theil-Sen either way: known limit, documented)
    late = _hist(lambda i: {k: 7.0 + 2.5 * (i >= 44)}, n=50)
    assert leak_verdict(late)["ok"]
    # a bimodal bounded gauge (leak-l8-big-live: engine1 route rows stored, x1000, cap 4.16M) passes
    bim = [0, 186, 2111, 151, 1511, 171, 1580, 587, 1938, 168, 1792, 182, 2107, 185, 1812, 183, 1894, 2131, 2131, 168,
           2098, 200, 2120, 137, 1473, 252, 1587, 1092, 2119, 204, 2119, 314, 2128, 235, 1632, 365, 1886, 2131, 2131, 0]
    assert leak_verdict(_hist(lambda i: {"leak/engine1_route_log_stored_rows_max": 1000.0 * bim[i]}, n=40))["ok"]
    # ... while the same gauge with a growing cap (its highs rise 2 % per step) fails
    grow = _hist(lambda i: {"leak/engine1_route_log_stored_rows_max": 1000.0 * bim[i] * (1 + 0.02 * i)}, n=40)
    assert not leak_verdict(grow)["ok"]
    # a noisy bounded gauge (0 or up to 32k rows at the sampling instant) passes
    gauge = _hist(lambda i: {"leak/engine1_hidden_log_rows_max": float(rng.integers(0, 32000) * (i % 3 != 0))}, n=40)
    assert leak_verdict(gauge)["ok"]


def test_leak_verdict_judges_live_memory_not_allocator_retention():
    """RssAnon that grows only through glibc's retained free pages (heap_free_gb) passes; live memory (rss_live_gb)
    that grows fails, whatever the allocator retains (leak investigation 10-21)."""
    rng = np.random.default_rng(2)

    def drv(live_slope, free_slope):
        def f(i):
            live = 3.0 + live_slope * i + rng.normal() * 0.02
            free = 0.3 + free_slope * i
            return {"leak/driver_rss_anon_gb": live + free, "leak/driver_heap_free_gb": free,
                    "leak/driver_rss_live_gb": live, "leak/driver_heap_inuse_gb": 0.5 + rng.normal() * 0.01}
        return _hist(f, n=50)

    retained = leak_verdict(drv(0.0, 0.03))  # +30 MB/step retained, live flat: RssAnon alone would fail
    assert retained["ok"], retained["failed"]
    assert not [r for r in retained["rows"] if r[0] == "leak/driver_rss_anon_gb"][0][4] <= 1.0  # it did grow
    leak = leak_verdict(drv(0.01, 0.0))  # +10 MB/step live: fails
    assert leak["failed"] == ["leak/driver_rss_live_gb"]
    both = leak_verdict(drv(0.01, 0.03))
    assert both["failed"] == ["leak/driver_rss_live_gb"]
    # without the live series (an older run / no glibc), RssAnon stays judged
    old = [{k: v for k, v in h.items() if "rss_live" not in k} for h in drv(0.0, 0.03)]
    assert leak_verdict(old)["failed"] == ["leak/driver_rss_anon_gb"]


_LIVE_SCRIPT = r"""
import json, sys
from thundersync_rl.leak_probe import process_stats
leak = sys.argv[1] == "leak"
held, hist, pins = [], [], []
process_stats()  # imports (torch) before the first step
for step in range(32):
    # transient buffers, larger every step (a new high-water mark), freed at the step end; a small long-lived object
    # allocated after them pins the heap top, so glibc keeps the freed pages resident (retention, live flat)
    tmp = [bytearray(32 * 1024) for _ in range(256 * (step + 1))]
    pins.append(bytearray(32 * 1024))
    del tmp
    if leak:  # the negative control: a deliberate per-step append (Python objects + malloc'd buffers, ~8 MB/step)
        held.append([str(i) * 4 for i in range(40000)])
        held.append([bytearray(32 * 1024) for _ in range(128)])
    hist.append({"step": step, **{f"leak/driver_{k}": v for k, v in process_stats().items() if k.endswith("_gb")}})
print(json.dumps(hist))
"""


@pytest.mark.parametrize("mode", ["retain", "leak"])
def test_leak_verdict_live_memory_real_process(mode):
    """Non-vacuous on a real process: allocator retention (RssAnon +8 MB/step, live flat) passes; the same process
    with a deliberate ~8 MB/step append fails on rss_live_gb."""
    import json
    import os
    import subprocess
    import sys

    from thundersync_rl import leak_probe

    if leak_probe._libc() is None:
        pytest.skip("not glibc (no mallinfo2)")
    env = dict(os.environ, MALLOC_ARENA_MAX="1", PYTHONPATH=os.pathsep.join(sys.path))
    env.pop("MALLOC_TRIM_THRESHOLD_", None)
    out = subprocess.run([sys.executable, "-c", _LIVE_SCRIPT, mode], capture_output=True, text=True, env=env,
                         check=True).stdout
    hist = json.loads(out.strip().splitlines()[-1])
    v = leak_verdict(hist)
    rows = {r[0]: r for r in v["rows"]}
    free = rows["leak/driver_heap_free_gb"]
    assert free[2] - free[1] > 0.03, free  # retention really happened in both arms (non-vacuous)
    if mode == "retain":
        assert v["ok"], (v["failed"], rows["leak/driver_rss_live_gb"])
        assert rows["leak/driver_rss_anon_gb"][4] > 0.5  # RssAnon itself grew (judged before: would fail)
    else:
        assert "leak/driver_rss_live_gb" in v["failed"], rows["leak/driver_rss_live_gb"]


def test_process_stats_has_rss_split():
    from thundersync_rl.leak_probe import process_stats

    st = process_stats()
    assert st["rss_anon_gb"] > 0 and "rss_shmem_gb" in st and st["rss_anon_gb"] <= st["rss_gb"] + 1e-6


def test_prefix_pools_dropped_only_without_learner_graphs(monkeypatch):
    from collections import OrderedDict

    from nemo_rl.models.megatron.prefix_state_cache import PrefixStateCache

    c = PrefixStateCache.__new__(PrefixStateCache)
    c.entries, c.pools = OrderedDict(), object()
    monkeypatch.setenv("NEMO_RL_LEARNER_GRAPHS", "1")
    assert c.drop_pools() is False and c.pools is not None  # graphs may hold the pool addresses
    monkeypatch.setenv("NEMO_RL_LEARNER_GRAPHS", "0")
    c.pools = None
    assert c.drop_pools() is False
    c.pools = type("P", (), {"free_pages": [], "free_slots": []})()
    assert c.drop_pools() is True and c.pools is None and not c.entries
