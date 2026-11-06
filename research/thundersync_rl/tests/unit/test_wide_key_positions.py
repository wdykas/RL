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
"""Wide sampler noise-key positions come from the request, not from the gpu_view lengths.

Under async scheduling the controller compacts the pending logits / temperatures / host request ids into survivor
order when requests finish, but not gpu_view.request_kv_length_offsets / request_query_lengths: for one step the rows
after a finished request read another request's position (Nano-L8 8k: 10 of 524,288 draws, 2 of them a reused
(request, position) key; the resumed step's last tokens differed run to run).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker, _request_positions

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class _Engine:
    def __init__(self, reqs: dict[int, tuple[int, int]]):
        self.reqs = reqs  # id -> (prompt length, generated count)

    def get_request(self, rid):
        p, g = self.reqs[rid]
        return SimpleNamespace(
            prompt_tokens=torch.zeros(p, dtype=torch.long), generated_tokens=[0] * g, sampling_params=SimpleNamespace()
        )


def test_request_positions_and_unknown_ids():
    eng = _Engine({3: (100, 0), 4: (100, 7), 9: (2048, 8191)})
    assert _request_positions(eng, np.array([3, 4, 9, 5])).tolist() == [100, 107, 10239, -1]
    # An eviction checkpoint moves generated tokens into the prompt: same position.
    eng.reqs[4] = (107, 0)
    assert _request_positions(eng, np.array([4])).tolist() == [107]


@cuda
def test_positions_follow_the_request_when_gpu_view_lengths_are_stale():
    from thundersync_rl.keyed_sampling import wide_sample_logits

    torch.manual_seed(2)
    n, V = 6, 4096
    logits = torch.randn(n, V, device="cuda") * 3
    rids = [10, 11, 12, 13, 14, 15]
    true_pos = [140, 141, 8300, 8301, 8302, 8303]
    eng = _Engine({r: (100, p - 100) for r, p in zip(rids, true_pos)})
    sampling = SimpleNamespace(sample_kernel=lambda *a, **k: None)
    w = object.__new__(ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)
    eng.controller = SimpleNamespace(_sampling=sampling)
    w.dynamic_inference_engine = eng
    w.rank = 0
    w.install_wide_sampler(salt=77)
    # gpu_view lengths one row off after a compaction (row 2 reads a finished request's 8398).
    stale = [140, 141, 8398, 8300, 8301, 8302]
    ctx = SimpleNamespace(
        gpu_view=SimpleNamespace(
            request_kv_length_offsets=torch.tensor([p - 1 for p in stale], device="cuda", dtype=torch.int32),
            request_query_lengths=torch.ones(n, device="cuda", dtype=torch.int32),
            temperature=torch.ones(n, device="cuda"),
        ),
        paused_request_count=0,
        request_ids=torch.tensor(rids, dtype=torch.int32),
    )
    out = sampling.sample_kernel(logits.clone(), n, ctx, no_top_k=True, no_top_p=True)
    ref = wide_sample_logits(
        logits, torch.tensor(rids, device="cuda"), torch.tensor(true_pos, device="cuda"), torch.ones(n, device="cuda"), 77
    )
    assert torch.equal(out, ref)
    assert sampling._wide_stats["gpu_view_pos_rows"] == 0
    # Unknown requests (no engine record) keep the gpu_view position. Next decode step: the known requests are one
    # token longer (the always-on key check requires contiguous positions per request).
    eng.reqs.pop(15)
    for r in rids[:5]:
        p, g = eng.reqs[r]
        eng.reqs[r] = (p, g + 1)
    out = sampling.sample_kernel(logits.clone(), n, ctx, no_top_k=True, no_top_p=True)
    ref2 = wide_sample_logits(
        logits, torch.tensor(rids, device="cuda"), torch.tensor([p + 1 for p in true_pos[:5]] + [stale[5]], device="cuda"),
        torch.ones(n, device="cuda"), 77,
    )
    assert torch.equal(out, ref2) and sampling._wide_stats["gpu_view_pos_rows"] == 1


@cuda
def test_no_duplicate_keys_across_finishing_and_compacted_requests(monkeypatch):
    """Async-scheduling replay: requests finish at different steps, the survivors are compacted into the finished
    slots (rows, temperatures and host request ids move; the gpu_view lengths do not, as in the controller). Every
    (request, position) key the sampler draws must be new and each request's positions contiguous."""
    import thundersync_rl.keyed_sampling as ks

    seen = []
    real = ks.wide_sample_logits

    def spy(rows, rids, pos, temp, salt, **kw):
        seen.extend(zip(rids.tolist(), pos.tolist()))
        return real(rows, rids, pos, temp, salt, **kw)

    monkeypatch.setattr(ks, "wide_sample_logits", spy)
    rng = np.random.default_rng(0)
    n0, V = 16, 512
    prompt = {r: int(rng.integers(50, 200)) for r in range(n0)}
    budget = {r: int(rng.integers(5, 40)) for r in range(n0)}
    eng = _Engine({r: (prompt[r], 0) for r in range(n0)})
    sampling = SimpleNamespace(sample_kernel=lambda *a, **k: None)
    w = object.__new__(ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)
    eng.controller = SimpleNamespace(_sampling=sampling)
    w.dynamic_inference_engine = eng
    w.rank = 0
    w.install_wide_sampler(salt=5)
    order = list(range(n0))
    gv_len = [prompt[r] for r in order]  # gpu_view lengths in the forward's (pre-compaction) order
    while order:
        n = len(order)
        ctx = SimpleNamespace(
            gpu_view=SimpleNamespace(
                request_kv_length_offsets=torch.tensor([x - 1 for x in gv_len[:n]], device="cuda", dtype=torch.int32),
                request_query_lengths=torch.ones(n, device="cuda", dtype=torch.int32),
                temperature=torch.ones(n, device="cuda"),
            ),
            paused_request_count=0,
            request_ids=torch.tensor(order, dtype=torch.int32),
        )
        sampling.sample_kernel(torch.randn(n, V, device="cuda"), n, ctx, no_top_k=True, no_top_p=True)
        for r in order:
            p, g = eng.reqs[r]
            eng.reqs[r] = (p, g + 1)
        survivors = [r for r in order if eng.reqs[r][1] < budget[r]]
        # Stale gpu_view: the previous order's lengths + 1, not compacted with the survivors.
        gv_len = [x + 1 for x in gv_len][: len(survivors)]
        order = survivors
    assert len(seen) == len(set(seen)) == sum(budget.values())
    per = {}
    for rid, p in seen:
        per.setdefault(rid, []).append(p)
    for rid, ps in per.items():
        assert ps == list(range(prompt[rid], prompt[rid] + budget[rid])), rid
