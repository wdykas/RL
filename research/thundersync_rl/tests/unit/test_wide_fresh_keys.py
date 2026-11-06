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
"""Resume reproducibility of the wide (53-bit keyed) engine sampler's fresh rows.

The final50b resume drew different step-40 rollouts than the uninterrupted run (digest 2827081834 vs 2890557791)
although every model / optimizer digest matched: the sampler keyed each token by (salt, engine request id,
position), the request id is a per-process arrival-order counter, and a resumed segment also changes the salt. Rows
the loop decodes fresh (every row of the first step after a checkpoint) now carry a stable key hash(seed, "fresh",
step, row, turn) and a lane / segment independent salt.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from thundersync_rl.speculative import wide_fresh_salt
from thundersync_rl.worker import _FRESH_KEY_BIT, ThunderSyncMegatronPolicyWorker, _RequestKeys

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class _Engine:
    def __init__(self, seeds: dict[int, int]):
        self.seeds = seeds
        self.calls = 0

    def get_request(self, rid):
        self.calls += 1
        s = self.seeds.get(rid)
        return SimpleNamespace(sampling_params=SimpleNamespace(**({} if s is None else {"noise_seed": s})))


def test_request_keys_cache_window_and_misses():
    eng = _Engine({5: 77, 7: 0, 1000: 9})
    k = _RequestKeys(eng)
    assert k.lookup(np.array([5, 6, 7])).tolist() == [77, -1, 0]
    calls = eng.calls
    assert k.lookup(np.array([7, 5])).tolist() == [0, 77]
    assert eng.calls == calls  # cached
    # The window slides past small ids; an id behind it is still answered (uncached).
    assert k.lookup(np.array([1000])).tolist() == [9]
    assert k.base > 5 and k.slots.size < 4096
    assert k.lookup(np.array([5, 1000])).tolist() == [77, 9]


def test_fresh_salt_depends_on_seed_only():
    assert wide_fresh_salt(42) == wide_fresh_salt(42)
    assert wide_fresh_salt(42) != wide_fresh_salt(43)


@cuda
def test_row_salts_equal_to_scalar_salt_are_bitwise_the_scalar_call():
    from thundersync_rl.keyed_sampling import wide_sample_logits

    torch.manual_seed(0)
    n, V = 64, 4096
    lg = torch.randn(n, V, device="cuda") * 2
    rid = torch.arange(n, device="cuda", dtype=torch.int64) + 123
    pos = torch.full((n,), 17, device="cuda", dtype=torch.int64)
    t = torch.ones(n, device="cuda")
    salt = 0x1234_5678_9ABC
    a = wide_sample_logits(lg, rid, pos, t, salt)
    b = wide_sample_logits(lg, rid, pos, t, 999, row_salts=torch.full((n,), salt, device="cuda", dtype=torch.int64))
    assert torch.equal(a, b)
    c = wide_sample_logits(lg, rid, pos, t, salt, row_salts=torch.full((n,), salt + 1, device="cuda", dtype=torch.int64))
    assert not torch.equal(a, c)


def _worker(engine, sampling):
    w = object.__new__(ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)
    engine.controller = SimpleNamespace(_sampling=sampling)
    w.dynamic_inference_engine = engine
    w.rank = 0
    w._prepare_data_for_generation = lambda data, greedy=False: (None, None, [SimpleNamespace() for _ in range(2)])
    return w


def _ctx(rids, pos, dev="cuda"):
    n = len(rids)
    return SimpleNamespace(
        gpu_view=SimpleNamespace(
            request_kv_length_offsets=torch.tensor(pos, device=dev, dtype=torch.int32),
            request_query_lengths=torch.ones(n, device=dev, dtype=torch.int32),
            temperature=torch.ones(n, device=dev),
        ),
        paused_request_count=0,
        request_ids=torch.tensor(rids, dtype=torch.int64),
    )


@cuda
def test_fresh_rows_draw_by_stable_key_whatever_their_request_id_and_salt():
    """Two 'runs' (different request ids, different per-segment salts): rows carrying the same noise_seed draw the
    same tokens; rows without one keep the request-id key (bitwise the pre-fix kernel)."""
    from thundersync_rl.keyed_sampling import wide_sample_logits

    torch.manual_seed(1)
    n, V = 8, 4096
    logits = torch.randn(n, V, device="cuda") * 3
    pos = list(range(100, 100 + n))
    fs = wide_fresh_salt(42)
    toks = []
    for rid0, salt in ((0, 11), (5000, 22)):
        rids = list(range(rid0, rid0 + n))
        seeds = {r: 1000 + j for j, r in enumerate(rids) if j < n // 2}  # first half fresh-keyed
        sampling = SimpleNamespace(sample_kernel=lambda *a, **k: (_ for _ in ()).throw(AssertionError("fallback")))
        w = _worker(_Engine(seeds), sampling)
        w.install_wide_sampler(salt=salt, fresh_salt=fs)
        out = sampling.sample_kernel(logits.clone(), n, _ctx(rids, [p - 1 for p in pos]), no_top_k=True, no_top_p=True)
        toks.append(out)
        # Unkeyed half == the plain request-id kernel at this salt.
        ref = wide_sample_logits(
            logits, torch.tensor(rids, device="cuda"), torch.tensor(pos, device="cuda"), torch.ones(n, device="cuda"), salt
        )
        assert torch.equal(out[n // 2 :], ref[n // 2 :])
        assert sampling._wide_stats["stable_rows"] == n // 2
    assert torch.equal(toks[0][: n // 2], toks[1][: n // 2])
    # Stable rows use (fresh_salt, noise_seed | 2**62, position).
    keyed = wide_sample_logits(
        logits[: n // 2],
        torch.tensor([(1000 + j) | _FRESH_KEY_BIT for j in range(n // 2)], device="cuda"),
        torch.tensor(pos[: n // 2], device="cuda"),
        torch.ones(n // 2, device="cuda"),
        fs,
    )
    assert torch.equal(toks[0][: n // 2], keyed)


def test_noise_seed_is_routed_into_sampling_params():
    sampling = SimpleNamespace(sample_kernel=lambda *a, **k: None)
    w = _worker(_Engine({}), sampling)
    w.install_wide_sampler(salt=1, fresh_salt=2)
    _, _, sps = w._prepare_data_for_generation({"noise_seed": torch.tensor([5, 6])})
    assert [sp.noise_seed for sp in sps] == [5, 6]
    _, _, sps = w._prepare_data_for_generation({})
    assert not any(hasattr(sp, "noise_seed") for sp in sps)


def test_without_fresh_salt_noise_seed_is_not_routed():
    sampling = SimpleNamespace(sample_kernel=lambda *a, **k: None)
    w = _worker(_Engine({}), sampling)
    w.install_wide_sampler(salt=1)
    _, _, sps = w._prepare_data_for_generation({"noise_seed": torch.tensor([5, 6])})
    assert not any(hasattr(sp, "noise_seed") for sp in sps)
