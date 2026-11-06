"""ENGINE_PROF reads every rank of every lane (it used to read lane 0, rank 0 only)."""

import threading
from types import SimpleNamespace

from thundersync_rl import engine_prof, grpo_loop


def _gen(per_rank):
    group = SimpleNamespace(run_all_workers_single_data=lambda name: per_rank)
    return SimpleNamespace(_policy=SimpleNamespace(worker_group=group))


def test_engine_prof_metrics_all_lanes_rank_max(monkeypatch):
    monkeypatch.setattr(grpo_loop.ray, "get", lambda x: x)
    lane0 = _gen([{"engine/decode_ms_rows1-1": 10.0}, {"engine/decode_ms_rows1-1": 12.0}])
    lane1 = _gen([{"engine/decode_ms_rows1-1": 7.0, "engine/steps": 5.0}, {}])
    out = grpo_loop.engine_prof_metrics(3, [("lane0", lane0), ("lane1", lane1)], cohort="lane1")
    assert out["engine/lane0/decode_ms_rows1-1"] == 12.0  # max over ranks, not rank 0
    assert out["engine/lane1/decode_ms_rows1-1"] == 7.0
    assert out["engine/decode_ms_rows1-1"] == 7.0 and out["engine/steps"] == 5.0  # cohort lane


def test_engine_prof_metrics_one_engine(monkeypatch):
    monkeypatch.setattr(grpo_loop.ray, "get", lambda x: x)
    out = grpo_loop.engine_prof_metrics(0, [("", _gen([{"engine/steps": 2.0}, {"engine/steps": 3.0}]))])
    assert out == {"engine/steps": 3.0}


def test_summary_swap_keeps_concurrent_appends():
    engine_prof._STEPS = []
    total = 200_000

    def writer():
        for _ in range(total):
            engine_prof._STEPS.append((0.0, 0.001, 1, 0))

    t = threading.Thread(target=writer)
    t.start()
    seen = 0
    while t.is_alive():
        seen += int(engine_prof.summary().get("engine/steps", 0.0))
    t.join()
    seen += int(engine_prof.summary().get("engine/steps", 0.0))
    # Copy-then-clear lost every append between the copy and the clear; the swap
    # can lose at most one per swap (an append that loaded the old list just before).
    assert total - seen <= 2
