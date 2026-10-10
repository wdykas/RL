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
"""Driver-side logic of cross-iteration speculative rollouts (no Ray, no GPUs)."""

import asyncio
from types import SimpleNamespace

import pytest
import thundersync_rl.speculative as spec_mod
import torch
from thundersync_rl.speculative import Plan, SpeculativeGeneration

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

EOD = 2


class _Remote:
    def __init__(self, fn):
        self.remote = fn


class _WorkerGroup:
    def __init__(self, handlers):
        self.handlers = handlers
        self.calls = []
        self.workers = [SimpleNamespace(**{k: _Remote(v) for k, v in handlers.items()})]

    def run_all_workers_single_data(self, method, **kwargs):
        self.calls.append((method, kwargs))
        return [self.handlers[method](**kwargs)]


class _Base:
    def __init__(self, handlers, gen_tail):
        self._policy = SimpleNamespace(worker_group=_WorkerGroup(handlers))
        self.gen_tail = gen_tail
        self.requests = []

    async def generate_async(self, data, greedy=False):
        self.requests.append(data)
        n = int(data["input_lengths"][0])
        ids = data["input_ids"][0, :n]
        tail = torch.tensor(self.gen_tail)
        yield (
            0,
            {
                "output_ids": torch.cat([ids, tail]).view(1, -1),
                "logprobs": torch.cat(
                    [torch.zeros(n), torch.full((len(tail),), -0.5)]
                ).view(1, -1),
                "generation_lengths": torch.tensor([len(tail)]),
            },
        )


@pytest.fixture(autouse=True)
def _no_ray(monkeypatch):
    monkeypatch.setattr(spec_mod.ray, "get", lambda x: x)


def _spec(learner_handlers, base_handlers=None, gen_tail=(9, EOD), **kw):
    handlers = {"termination_id": lambda: EOD, **(base_handlers or {})}
    learner = SimpleNamespace(worker_group=_WorkerGroup(learner_handlers))
    return SpeculativeGeneration(
        _Base(handlers, list(gen_tail)),
        learner,
        vocab_limit=100,
        head_k=8,
        draft_budget=4096,
        max_new_tokens=16,
        pad_token_id=0,
        verify_mode="block",
        q_storage="stash",
        **kw,
    )


def test_verify_builds_complete_and_continuation_plans():
    results = {0: (3, 0), 1: (1, 7), 2: (0, EOD)}  # row -> (accepted, next)

    def verify(rows, prompt_lens, segments, keys, **_):
        return [
            (i, {"accepted": a, "next": y, "logprobs": [-0.1] * (a + 1)})
            for i, (a, y) in results.items()
        ]

    s = _spec({"verify_drafts_block_stashed": verify})
    p = [torch.tensor([10 + i, 20 + i]) for i in range(3)]
    drafts = [
        [(p[0], -1, [4, 5, EOD], True, "1:0", [(0, 0, 3)])],  # fully accepted, finished
        [(p[1], -1, [4, 5, 6], False, "1:1", [(0, 0, 3)])],  # rejected after 1 token
        [(p[2], -1, [4], False, "1:2", [(0, 0, 1)])],  # first token replaced by EOS
    ]
    s.verify(drafts)
    plans = {k: v[0] for k, v in s.plans.items()}
    assert plans[(10, 20)] == Plan(-1, [4, 5, EOD], [-0.1] * 3, complete=True)
    assert plans[(11, 21)] == Plan(-1, [4, 7], [-0.1] * 2, complete=False)
    assert plans[(12, 22)] == Plan(-1, [EOD], [-0.1], complete=True)
    ((_, kw),) = s.learner_policy.worker_group.calls
    assert kw["keys"] == ["1:0", "1:1", "1:2"]
    assert kw["segments"][1] == [(0, 0, 3)]


def test_cohort_drafts_resume_and_track_versions():
    started, collected = [], [[([5, 6], False), ([7], True)], [([8], True)]]
    s = _spec(
        {"stash_weights": lambda **kw: 1},
        base_handlers={
            "start_drafts": lambda prompts, budgets, interval: started.append(
                (prompts, budgets)
            ),
            "collect_drafts": lambda: collected.pop(0),
        },
    )
    p0, p1 = torch.tensor([1, 1]), torch.tensor([3, 3])
    collect = asyncio.run(s.draft_cohorts(0, {2: [p0, p1]}))
    collect()
    assert started[0] == ([[1, 1], [3, 3]], [16, 16])
    d0, d1 = s.cohorts[2]
    assert d0["segments"] == [(0, 0, 2)] and d1["finished"]
    # Next iteration resumes only the unfinished draft, from its prefix, under step 1.
    collect = asyncio.run(s.draft_cohorts(1, {}))
    collect()
    assert started[1] == ([[1, 1, 5, 6]], [14])
    assert d0["tokens"] == [5, 6, 8] and d0["segments"] == [(0, 0, 2), (1, 2, 3)]


def test_rollout_request_continues_from_verified_prefix():
    s = _spec({})
    prompt = torch.tensor([30, 31, 32])
    s.plans[(30, 31, 32)] = [Plan(-1, [5, 6], [-0.1, -0.2], complete=False)]
    data = BatchedDataDict(
        {"input_ids": prompt.view(1, -1), "input_lengths": torch.tensor([3])}
    )
    out = [o for o in asyncio.run(_collect(s.generate_async(data)))]
    ((_, res),) = out
    assert res["output_ids"][0].tolist() == [30, 31, 32, 5, 6, 9, EOD]
    assert res["logprobs"][0].tolist() == pytest.approx(
        [0, 0, 0, -0.1, -0.2, -0.5, -0.5]
    )
    assert int(res["generation_lengths"][0]) == 4
    (req,) = s.base.requests
    assert req["input_ids"][0].tolist() == [30, 31, 32, 5, 6]
    assert int(req["max_new_tokens"][0]) == 16 - 2 and "noise_seed" not in req


def test_complete_plan_needs_no_engine_request():
    s = _spec({})
    s.plans[(1, 2)] = [Plan(-1, [3, EOD], [-0.3, -0.4], complete=True)]
    data = BatchedDataDict(
        {"input_ids": torch.tensor([[1, 2]]), "input_lengths": torch.tensor([2])}
    )
    ((_, res),) = asyncio.run(_collect(s.generate_async(data)))
    assert res["output_ids"][0].tolist() == [1, 2, 3, EOD] and not s.base.requests


async def _collect(agen):
    return [x async for x in agen]


class _Ref:
    """Stands in for a Ray ObjectRef: ``future()`` returns a resolved future."""

    def __init__(self, value=None, error=None):
        import concurrent.futures

        self._f = concurrent.futures.Future()
        if error is not None:
            self._f.set_exception(error)
        else:
            self._f.set_result(value)

    def future(self):
        return self._f


class _RefGroup(_WorkerGroup):
    def run_all_workers_single_data(self, method, **kwargs):
        self.calls.append((method, kwargs))
        try:
            return [_Ref(self.handlers[method](**kwargs))]
        except Exception as e:  # surfaces when awaited, like a failed Ray task
            return [_Ref(error=e)]


def _drafts(n):
    return [
        [(torch.tensor([50 + i]), -1, [4, EOD], True, f"1:{i}", [(0, 0, 2)])]
        for i in range(n)
    ]


def _accept_all(rows, prompt_lens, segments, keys, **_):
    return [
        (j, {"accepted": 2, "next": 0, "logprobs": [-0.1, -0.1]})
        for j in range(len(rows))
    ]


def test_chunked_verification_uses_all_pools_and_publishes_every_plan():
    s = _spec({})
    pools = [_RefGroup({"verify_drafts_block_stashed": _accept_all}) for _ in range(2)]
    s.verifiers = s._chunk_pools = pools
    asyncio.run(s.verify_chunked(_drafts(6), chunks=4))
    assert [len(p.calls) for p in pools] == [2, 2]
    assert sorted(k[0] for k in s.plans) == [50, 51, 52, 53, 54, 55]
    assert all(p[0].complete for p in s.plans.values())


def test_verification_failure_surfaces_instead_of_hanging():
    def boom(**_):
        raise ValueError("scorer failed")

    s = _spec({})
    s.verifiers = s._chunk_pools = [_RefGroup({"verify_drafts_block_stashed": boom})]
    data = BatchedDataDict(
        {"input_ids": torch.tensor([[50]]), "input_lengths": torch.tensor([1])}
    )

    async def scenario():
        verify = asyncio.create_task(s.verify_chunked(_drafts(1), chunks=1))
        rollout = asyncio.create_task(_collect(s.generate_async(data)))
        done, _ = await asyncio.wait({verify, rollout}, timeout=5)
        assert verify in done and rollout in done, (
            "rollout hung on a failed verification"
        )
        with pytest.raises(RuntimeError, match="verification failed"):
            rollout.result()

    asyncio.run(scenario())


def test_every_iteration_after_the_first_gets_a_cohort():
    from thundersync_rl.grpo_loop import new_cohort_targets

    for lookahead in (2, 3, 4):
        started: set[int] = set()
        for step in range(12):
            new = new_cohort_targets(step, lookahead, max_steps=12, existing=started)
            assert len(new) <= 2 and all(step < t < step + 1 + lookahead for t in new)
            started.update(new)
            started.discard(step)  # verified (popped) at its own iteration
            assert step + 1 >= 12 or step + 1 in started or step + 1 in new
        assert started.union(range(1, 12)) == set(range(1, 12))
    assert new_cohort_targets(0, 3, max_steps=12, existing=set()) == [1, 2]
    assert new_cohort_targets(1, 3, max_steps=12, existing={2}) == [3, 4]
    assert new_cohort_targets(10, 3, max_steps=12, existing=set()) == [11]
