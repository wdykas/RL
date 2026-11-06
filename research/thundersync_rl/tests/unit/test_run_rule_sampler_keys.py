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
"""Always-on sampler noise-key consistency (key_check): every (request, position) key is drawn once and each
request's positions advance by one per decode step. Fires on the pre-F11 position source (gpu_view lengths not
compacted with the rows under async scheduling), passes on the request-derived one."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from thundersync_rl.key_check import KeyContinuity, SamplerKeyError


def _replay(source: str, seed: int = 0, n0: int = 16):
    """Async-scheduling replay (as test_wide_key_positions): requests finish at different steps, survivors are
    compacted into the finished rows, the gpu_view lengths are not. Feeds each step's (rids, positions) to a
    KeyContinuity; returns it (raises SamplerKeyError on the first bad step)."""
    rng = np.random.default_rng(seed)
    prompt = {r: int(rng.integers(50, 200)) for r in range(n0)}
    budget = {r: int(rng.integers(5, 40)) for r in range(n0)}
    gen = {r: 0 for r in range(n0)}
    order = list(range(n0))
    gv_len = [prompt[r] for r in order]
    kc = KeyContinuity("test")
    while order:
        n = len(order)
        if source == "request":
            pos = np.array([prompt[r] + gen[r] for r in order])
        else:  # pre-F11: gpu_view request_kv_length_offsets + request_query_lengths
            pos = np.array(gv_len[:n])
        kc.observe(np.array(order), pos)
        for r in order:
            gen[r] += 1
        survivors = [r for r in order if gen[r] < budget[r]]
        gv_len = [x + 1 for x in gv_len][: len(survivors)]
        order = survivors
    return kc, sum(budget.values())


def test_request_positions_pass():
    for seed in range(5):
        kc, total = _replay("request", seed)
        assert kc.violations == 0 and kc.rows_checked == total


def test_old_gpu_view_positions_fire():
    with pytest.raises(SamplerKeyError, match="noise keys out of sequence"):
        _replay("gpu_view")


def test_reused_skipped_and_duplicate_keys_fire():
    kc = KeyContinuity()
    assert kc.check(np.array([1, 2]), np.array([10, 20])) == []
    assert kc.check(np.array([2, 1]), np.array([21, 11])) == []  # row order may change
    assert kc.check(np.array([1, 2]), np.array([12, 22])) == []
    assert kc.check(np.array([1]), np.array([12]))  # reused key
    kc = KeyContinuity()
    kc.check(np.array([1]), np.array([10]))
    assert kc.check(np.array([1]), np.array([12]))  # skipped position
    assert KeyContinuity().check(np.array([3, 3]), np.array([5, 6]))  # one request twice in a call
    kc = KeyContinuity()
    kc.check(np.array([1, 2]), np.array([10, 20]))
    assert kc.check(np.array([2, 3]), np.array([21, 7])) == []  # request 1 finished / paused, 3 new
    assert kc.check(np.array([1, 3]), np.array([11, -1])) == []  # paused request back (fresh), unknown skipped


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_wide_kernel_raises_on_a_resampled_step():
    from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker

    class _Engine:
        reqs = {10: (100, 0), 11: (100, 5)}

        def get_request(self, rid):
            p, g = self.reqs[rid]
            return SimpleNamespace(prompt_tokens=[0] * p, generated_tokens=[0] * g, sampling_params=SimpleNamespace())

    eng = _Engine()
    sampling = SimpleNamespace(sample_kernel=lambda *a, **k: None)
    eng.controller = SimpleNamespace(_sampling=sampling)
    w = object.__new__(ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)
    w.dynamic_inference_engine, w.rank = eng, 0
    w.install_wide_sampler(salt=1)
    ctx = SimpleNamespace(
        gpu_view=SimpleNamespace(
            request_kv_length_offsets=torch.zeros(2, device="cuda", dtype=torch.int32),
            request_query_lengths=torch.ones(2, device="cuda", dtype=torch.int32),
            temperature=torch.ones(2, device="cuda"),
        ),
        paused_request_count=0,
        request_ids=torch.tensor([10, 11], dtype=torch.int32),
    )
    logits = torch.randn(2, 64, device="cuda")
    sampling.sample_kernel(logits, 2, ctx)
    eng.reqs = {10: (100, 1), 11: (100, 6)}
    sampling.sample_kernel(logits, 2, ctx)
    assert sampling._wide_stats["key_check_rows"] == 4 and sampling._wide_stats["key_check_violations"] == 0
    with pytest.raises(SamplerKeyError):
        sampling.sample_kernel(logits, 2, ctx)  # same positions again: reused noise keys
