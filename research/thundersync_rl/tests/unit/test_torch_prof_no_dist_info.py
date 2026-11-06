# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Engine / verify torch profiles skip the trace's distributed metadata (torch 2.11 raises "value was None" at
__enter__ on process groups without a backend config)."""

import inspect

import torch

from thundersync_rl import worker


class _FakeProfile:
    entered: list = []

    def __init__(self, *args, **kwargs):
        pass

    def _get_distributed_info(self):
        raise RuntimeError("value was None")

    def __enter__(self):
        self.dist = self._get_distributed_info()  # what torch's trace metadata does
        _FakeProfile.entered.append(self)
        return self




def test_verify_torch_profiler_skips_distributed_info(monkeypatch):
    monkeypatch.setattr(torch.profiler, "profile", _FakeProfile)
    monkeypatch.setenv("THUNDERSYNC_TORCH_PROF", "/tmp/x")

    class _W:
        _null_verify_calls = 1

    # the class is a Ray actor class here: call the plain function under its method wrapper
    prof = inspect.unwrap(worker.ThunderSyncMegatronPolicyWorker._torch_profiler)(_W(), True)
    assert isinstance(prof, _FakeProfile) and prof.dist is None
