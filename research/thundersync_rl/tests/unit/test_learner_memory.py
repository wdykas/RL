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
"""Learner allocator setup (integ5b): expandable segments on the learner only, allocator stats."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

worker = pytest.importorskip("thundersync_rl.worker")


def test_expandable_segments_flag_default_on(monkeypatch):
    monkeypatch.delenv("THUNDERSYNC_LEARNER_EXPANDABLE_SEGMENTS", raising=False)
    assert worker.learner_expandable_segments()
    monkeypatch.setenv("THUNDERSYNC_LEARNER_EXPANDABLE_SEGMENTS", "0")
    assert not worker.learner_expandable_segments()
    monkeypatch.setenv("THUNDERSYNC_LEARNER_EXPANDABLE_SEGMENTS", "1")
    assert worker.learner_expandable_segments()


def test_disabled_flag_leaves_allocator_alone(monkeypatch):
    monkeypatch.setenv("THUNDERSYNC_LEARNER_EXPANDABLE_SEGMENTS", "0")
    assert worker.enable_learner_expandable_segments() is False


def test_engine_worker_never_switches():
    """Only a worker with an optimizer (the learner) calls the switch: the call sits
    inside the optimizer branch of ``__init__`` (engines: init_optimizer=False)."""
    src = open(worker.__file__).read()
    init = src[src.index("    def __init__(self, *args: Any, **kwargs: Any) -> None:"):]
    init = init[: init.index("    def stream_begin_step")]
    branch = init[init.index('if getattr(self, "optimizer", None) is not None:'):]
    assert "enable_learner_expandable_segments()" in branch.split("self._prefix_cache = None")[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_switch_makes_later_segments_expandable_and_stats():
    code = textwrap.dedent(
        """
        import torch
        from thundersync_rl import worker
        a = torch.empty(2**28, dtype=torch.uint8, device="cuda")  # before: fixed segment
        assert worker.enable_learner_expandable_segments()
        b = torch.empty(2**28, dtype=torch.uint8, device="cuda")
        segs = {s["address"]: s.get("is_expandable") for s in torch.cuda.memory_snapshot()}
        assert any(v for v in segs.values()) and not all(v for v in segs.values()), segs
        st = worker.allocator_stats()
        assert set(st) == {"learner_mem_reserved_peak_gb", "learner_mem_frag_peak_gb",
                           "learner_mem_nontorch_gb", "learner_mem_alloc_retries",
                           "learner_mem_device_allocs", "learner_mem_device_frees"}
        assert st["learner_mem_reserved_peak_gb"] >= 0.5 and st["learner_mem_nontorch_gb"] >= 0
        print("ok")
        """
    )
    env = dict(os.environ)
    env.pop("THUNDERSYNC_LEARNER_EXPANDABLE_SEGMENTS", None)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr[-2000:]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_expandable_segments_turn_off_nccl_graph_registration():
    """Expandable segments remap pages under NCCL's graph-registered buffers (scu-loop3 TP4 x EP4
    illegal address): the switch also sets NCCL_GRAPH_REGISTER=0, unless it is set explicitly."""
    code = textwrap.dedent(
        """
        import os
        from thundersync_rl import worker
        explicit = os.environ.get("NCCL_GRAPH_REGISTER")
        assert worker.enable_learner_expandable_segments()
        assert os.environ["NCCL_GRAPH_REGISTER"] == (explicit or "0"), os.environ.get("NCCL_GRAPH_REGISTER")
        print("ok")
        """
    )
    for explicit in (None, "1"):
        env = dict(os.environ)
        env.pop("THUNDERSYNC_LEARNER_EXPANDABLE_SEGMENTS", None)
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        env.pop("NCCL_GRAPH_REGISTER", None)
        if explicit:
            env["NCCL_GRAPH_REGISTER"] = explicit
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
        assert out.returncode == 0 and "ok" in out.stdout, out.stderr[-2000:]


def test_disabled_flag_keeps_nccl_graph_registration(monkeypatch):
    monkeypatch.setenv("THUNDERSYNC_LEARNER_EXPANDABLE_SEGMENTS", "0")
    monkeypatch.delenv("NCCL_GRAPH_REGISTER", raising=False)
    assert worker.enable_learner_expandable_segments() is False
    assert "NCCL_GRAPH_REGISTER" not in os.environ


def test_binds_init_optimizer_positional_keyword_default():
    kw = {"worker_sharding_annotations": None}
    assert worker._binds_init_optimizer(("cfg", "tok"), kw)  # default True
    assert worker._binds_init_optimizer(("cfg", "tok", None, None, True), kw)
    assert not worker._binds_init_optimizer(("cfg", "tok", None, None, False), kw)
    assert not worker._binds_init_optimizer(("cfg", "tok"), {**kw, "init_optimizer": False})


def _construct_with_fake_base(monkeypatch, init_optimizer):
    """Run the ThunderSync constructor's prologue against a base whose __init__ only records
    whether expandable segments were switched on when the checkpoint load would run."""
    events = []
    monkeypatch.setattr(worker, "enable_learner_expandable_segments", lambda: events.append("enable") or True)

    def fake_base_init(self, *args, **kwargs):
        hook = getattr(self, "_pre_load_checkpoint_hook", None)
        if hook is not None:
            hook(None, None)  # setup_model_and_optimizer: hook, then load_checkpoint
        events.append("load_checkpoint")
        raise StopIteration  # stop before the rest of the real constructor

    monkeypatch.setattr(worker.MegatronPolicyWorkerImpl, "__init__", fake_base_init)
    cls = worker.ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class
    w = object.__new__(cls)
    with pytest.raises(StopIteration):
        cls.__init__(w, "cfg", "tok", init_optimizer=init_optimizer, worker_sharding_annotations=None)
    return events


def test_learner_switches_allocator_before_checkpoint_load(monkeypatch):
    """A resumed learner loads its Adam state in the checkpoint load: the expandable-segment switch
    must precede it (else stock segments: +0.70 GiB allocated / +17 GiB reserved at L8 resume)."""
    assert _construct_with_fake_base(monkeypatch, True) == ["enable", "load_checkpoint"]


def test_engine_has_no_pre_load_switch(monkeypatch):
    assert _construct_with_fake_base(monkeypatch, False) == ["load_checkpoint"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_loaded_state_packs_like_lazily_created_state():
    """The mechanism: per-tensor state allocated into an empty cache (checkpoint load) with stock
    segments pays 1 segment per 19.03 MiB tensor rounded to 20 MiB (tail < 1 MiB never split); after
    the switch the same allocations are exact."""
    code = textwrap.dedent(
        """
        import sys, torch
        from thundersync_rl import worker
        if sys.argv[1] == "1":
            assert worker.enable_learner_expandable_segments()
        n, size = 64, 19955712
        base = torch.cuda.memory_allocated()
        ts = [torch.empty(size, dtype=torch.uint8, device="cuda") for _ in range(n)]
        print("waste", torch.cuda.memory_allocated() - base - n * size)
        """
    )
    env = dict(os.environ)
    env.pop("THUNDERSYNC_LEARNER_EXPANDABLE_SEGMENTS", None)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    waste = {}
    for mode in ("0", "1"):
        out = subprocess.run([sys.executable, "-c", code, mode], env=env, capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr[-2000:]
        waste[mode] = int(out.stdout.split("waste")[-1])
    assert waste["0"] == 64 * (20 * 2**20 - 19955712), waste  # stock: the resumed-run allocation
    assert waste["1"] == 0, waste
