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
        **{
            "draft_variants": 1,
            "variant_eps": 0.0,
            "verify_mode": "block",
            "verify_precision": "fp32",
            "verify_batch_tokens": 16384,
            "longest_first": False,
            "verify_on": "learner",
            "pool_weights": None,
            "fused_verify_train": False,
            "fused_max_retained_tokens": 16384,
            **kw,
        },
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
            "draft_request_ids": lambda: {},
            "collect_drafts": lambda records=None, record_refs=None, request_ids=None: collected.pop(0),
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


# -- G2: behavior log-probs of accepted draft tokens --------------------------


def test_behavior_logprobs_correct_accepted_tokens_only():
    from thundersync_rl.speculative import behavior_logprobs

    logp = [-1.0, -2.0, -3.0]  # 2 accepted draft tokens + the verifier's token
    out = behavior_logprobs(logp, 2, engine_logq=[-1.5, -2.5], draft_logq=[-1.25, -2.0])
    assert out == pytest.approx([-1.0 - 1.5 + 1.25, -2.0 - 2.5 + 2.0, -3.0])
    # Missing engine or verifier log q: log p unchanged (keyed / non-streamed).
    assert behavior_logprobs(logp, 2, None, [-1.0, -2.0]) == logp
    assert behavior_logprobs(logp, 2, [-1.0, -2.0], None) == logp
    # Non-finite correction (q underflow) falls back to log p for that token.
    out = behavior_logprobs(logp, 2, [-1.5, -2.5], [float("-inf"), -2.0])
    assert out == pytest.approx([-1.0, -2.5, -3.0])


def test_plans_report_p_times_engine_q_over_scorer_q():
    # row -> (accepted, next); verifier log p = -0.1 per kept token, log q = -0.3.
    results = {0: (3, 0), 1: (1, 7), 2: (2, 9)}

    def verify(rows, prompt_lens, segments, keys, **_):
        return [
            (
                i,
                {
                    "accepted": a,
                    "next": y,
                    "logprobs": [-0.1] * (a + 1),
                    "draft_logq": [-0.3] * a,
                },
            )
            for i, (a, y) in results.items()
        ]

    s = _spec({"verify_drafts_block_stashed": verify})
    p = [torch.tensor([10 + i, 20 + i]) for i in range(3)]
    eng = [-0.2, -0.4, -0.6]
    drafts = [
        [(p[0], -1, [4, 5, EOD], True, "1:0", [(0, 0, 3)], eng)],
        [(p[1], -1, [4, 5, 6], False, "1:1", [(0, 0, 3)], eng)],
        # No engine log-probs: the scorer's log p is kept.
        [(p[2], -1, [4, 5, 6], False, "1:2", [(0, 0, 3)], None)],
    ]
    s.verify(drafts)
    plans = {k: v[0] for k, v in s.plans.items()}
    full = plans[(10, 20)]
    assert full.complete and full.prefix == [4, 5, EOD]
    assert full.prefix_logprobs == pytest.approx([-0.1 + e + 0.3 for e in eng])
    cont = plans[(11, 21)]
    assert cont.prefix == [4, 7] and not cont.complete
    # Accepted token corrected; the verifier's residual token keeps log p.
    assert cont.prefix_logprobs == pytest.approx([-0.1 - 0.2 + 0.3, -0.1])
    assert plans[(12, 22)].prefix_logprobs == pytest.approx([-0.1] * 3)


def test_streamed_drafts_carry_engine_logprobs():
    s = _spec(
        {},
        base_handlers={
            "start_drafts": lambda prompts, budget, interval: None,
            "draft_request_ids": lambda: {},
            "collect_drafts": lambda records=None, record_refs=None, request_ids=None: [
                ([4, EOD], True, [-0.5, -0.25]),
                ([4, 5], False, [-0.5]),  # misaligned: dropped
                ([6], False),  # engine without log-probs
            ],
        },
    )
    prompts = [torch.tensor([1 + i]) for i in range(3)]
    collect = asyncio.run(s.draft_streaming(prompts))
    (a,), (b,), (c,) = collect()
    assert a[2] == [4, EOD] and a[3] and a[6] == [-0.5, -0.25]
    assert b[6] is None and c[6] is None and c[2] == [6]


def test_cohort_drafts_keep_engine_logprobs_across_resumptions():
    from thundersync_rl.speculative import cohort_draft

    collected = [
        [([5, 6], False, [-0.1, -0.2]), ([7], False, [-0.3])],
        [([8], True, [-0.4]), ([9], False)],  # second draft: segment without lps
        [([10], True, [-0.5])],
    ]
    s = _spec(
        {},
        base_handlers={
            "start_drafts": lambda prompts, budgets, interval: None,
            "draft_request_ids": lambda: {},
            "collect_drafts": lambda records=None, record_refs=None, request_ids=None: collected.pop(0),
        },
    )
    p0, p1 = torch.tensor([1, 1]), torch.tensor([3, 3])
    asyncio.run(s.draft_cohorts(0, {3: [p0, p1]}))()
    asyncio.run(s.draft_cohorts(1, {}))()
    d0, d1 = s.cohorts[3]
    assert d0["tokens"] == [5, 6, 8] and d0["logprobs"] == [-0.1, -0.2, -0.4]
    assert d0["segments"] == [(0, 0, 2), (1, 2, 3)]
    # Once a segment lacks log-probs the draft falls back to log p for good.
    asyncio.run(s.draft_cohorts(2, {}))()
    assert d1["tokens"] == [7, 9, 10] and d1["logprobs"] is None
    assert cohort_draft(d0)[6] == [-0.1, -0.2, -0.4]
    assert cohort_draft(d0)[:6] == (p0, -1, d0["tokens"], True, "3:0", d0["segments"])


def test_truncate_draft_to_logged_rows():
    import torch
    from thundersync_rl.speculative import _truncate_to_rows

    toks, lps = [5, 6, 7, 8], [-0.1, -0.2, -0.3, -0.4]
    full = torch.zeros(4, 2)
    assert _truncate_to_rows((toks, True, lps), full) == (toks, True, lps)
    assert _truncate_to_rows((toks, True, lps), None) == (toks, True, lps)
    # Fewer rows than tokens: the prefix is a valid draft, no longer finished.
    assert _truncate_to_rows((toks, True, lps), torch.zeros(2, 2)) == (
        [5, 6],
        False,
        [-0.1, -0.2],
    )


def test_verification_seeds_never_repeat_across_calls_and_steps():
    """Fresh randomness per verification call: a stream that shaped one update
    must never be reused for a later decision (pools, calls and steps)."""
    seeds = []

    def verify(rows, prompt_lens, segments, keys, seed, **_):
        seeds.append(seed)
        return [
            (j, {"accepted": 1, "next": 0, "logprobs": [-0.1, -0.1]})
            for j in range(len(rows))
        ]

    s = _spec({"verify_drafts_block_stashed": verify})
    for _ in range(5):  # steps, synchronous verification
        s.verify(_drafts(16))
    pools = [_RefGroup({"verify_drafts_block_stashed": verify}) for _ in range(2)]
    s.verifiers = s._chunk_pools = pools
    for _ in range(5):  # steps, chunked verification on two pools
        asyncio.run(s.verify_chunked(_drafts(16), chunks=4))
    assert len(seeds) == len(set(seeds)) > 10


def test_draft_age_stats_are_token_weighted_per_drafting_version():
    from thundersync_rl.speculative import SpeculativeGeneration

    s = object.__new__(SpeculativeGeneration)
    s.current_step, s.stats = 10, {}
    tok, kept = {}, {}
    # Tokens 0-99 drafted at version 9 (age 1), 100-299 at version 7 (age 3);
    # 150 tokens accepted: all of age 1, 50 of age 3.
    draft = (None, 0, [0] * 300, True, "k", [(9, 0, 100), (7, 100, 300)])
    s._count_ages(draft, 150, tok, kept)
    # A second draft at version 4 counts as age 4+; nothing accepted.
    s._count_ages((None, 0, [0] * 50, True, "k2", [(4, 0, 50)]), 0, tok, kept)
    s._age_stats(tok, kept)
    assert s.stats["spec/age1_token_frac"] == 100 / 350
    assert s.stats["spec/age1_accepted_frac"] == 1.0
    assert s.stats["spec/age3_accepted_frac"] == 50 / 200
    assert s.stats["spec/age4+_token_frac"] == 50 / 350
    assert s.stats["spec/age4+_accepted_frac"] == 0.0


def test_tail_lookup_references_dead_suffix_then_siblings():
    from thundersync_rl.speculative import Plan, SpeculativeGeneration, _key

    s = object.__new__(SpeculativeGeneration)
    s.tail_lookup, s.tail_lookup_max_refs = True, 3
    prompt = torch.tensor([1, 2, 3])
    flat = [
        (0, (prompt, 0, [5, 6, 7])),
        (1, (prompt, 0, [8, 9])),
        (2, (prompt, 0, [4])),
    ]
    s._record_siblings(flat)
    plan = Plan(seed=-1, prefix=[5, 6, 10], dead_suffix=[7])
    refs = s._lookup_references(_key(prompt), plan)
    # Dead suffix first, then the prompt's drafts, capped at tail_lookup_max_refs.
    assert refs == [[7], [5, 6, 7], [8, 9]]
    # Off: nothing recorded, no references.
    s.tail_lookup = False
    s._record_siblings(flat)
    assert s._lookup_references(_key(prompt), Plan(seed=-1)) == []
