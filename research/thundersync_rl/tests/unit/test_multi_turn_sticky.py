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
"""Multi-turn rollouts of drafted plans under sticky routing (scheme A):
the driver steps the environment at verified turn ends only, later turns are
theta_{k-1} continuations verified in pieces, and the rollout loop replays
the recorded turns (no Ray, no GPUs)."""

import asyncio
from types import SimpleNamespace

import pytest
import thundersync_rl.speculative as spec_mod
import torch
from test_speculative_driver import EOD, _spec as make_spec
from test_sticky_driver import _dense, _Engine
from thundersync_rl.multi_turn import TurnEnvironment, replay

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentReturn
from nemo_rl.experience.rollouts import run_sample_multi_turn_rollout


class _Ref:
    """A fake environment actor's result handle (hashable, like an ObjectRef)."""

    def __init__(self, value):
        self.value = value


def _get(x):
    if isinstance(x, list) and x and isinstance(x[0], _Ref):
        return [r.value for r in x]
    return x


@pytest.fixture(autouse=True)
def _no_ray(monkeypatch):
    monkeypatch.setattr(spec_mod.ray, "get", _get)


class _Tokenizer:
    """Text is the token ids joined by spaces (observations are "o<id> ...")."""

    pad_token_id = 0

    def batch_decode(self, ids, skip_special_tokens=True):
        return [" ".join(str(int(t)) for t in x if int(t) != EOD) for x in ids]

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        ids = [int(w[1:]) for w in text.split()]
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))


class _Env:
    """A prefix-dependent environment: observation and reward depend on the
    last assistant message; terminates once the assistant said "8"."""

    def __init__(self):
        self.calls = []

    @property
    def step(self):
        return SimpleNamespace(remote=self._step)

    def _step(self, messages, env_info):
        (log,) = messages
        last = log[-1]["content"]
        self.calls.append((last, dict(env_info[0])))
        done = "8" in last.split()
        obs = "o60" if done else "o50 o51"
        return _Ref(
            EnvironmentReturn(
                observations=[{"role": "environment", "content": obs}],
                metadata=[{"turn": env_info[0]["turn"] + 1}],
                next_stop_strings=[None],
                rewards=torch.tensor([1.0 if done else 0.5]),
                terminateds=torch.tensor([done]),
                answers=None,
            )
        )


def _rows(prompt):
    return [
        {
            "message_log": [{"role": "user", "content": "q", "token_ids": prompt}],
            "extra_env_info": {"turn": 0},
            "task_name": "toy",
            "stop_strings": None,
            "idx": 0,
        }
    ]


def test_turn_two_rejection_steps_env_once_per_verified_turn_end():
    engine = _Engine(
        [
            # Drafting round (theta_{k-1}): turn 1 of prompt [1, 2], finished.
            [([4, EOD], True, [-0.1] * 2, 3)],
            # Turn 2 stream from [1, 2, 4, EOD, 50, 51]: its prefill routes
            # every position (observation included); the last token unrouted.
            [([6, 7, EOD], True, [-0.2] * 3, 8)],
            # After the rejection inside turn 2: from [..., 6, 8].
            [([EOD], True, [-0.3], 8)],
        ]
    )
    calls = []
    results = [(2, 99), (1, 8), (1, 99)]  # whole; reject at 1 -> 8; whole

    def verify(rows, prompt_lens, engine_logq, anchors, **_):
        calls.append(
            ([r.tolist() for r in rows], prompt_lens, [a.length for a in anchors])
        )
        a, y = results.pop(0)
        return [
            (
                0,
                {
                    "accepted": a,
                    "next": y,
                    "logprobs": [-1.0] * (a + 1),
                    "draft_logq": engine_logq[0][:a],
                },
            )
        ]

    s = make_spec(
        {"verify_drafts_block_engine": verify},
        base_handlers=engine.handlers(),
        verify_source="engine",
        hidden_transport="driver",
        sticky_delta=0.03,
    )
    env = _Env()
    tok = _Tokenizer()
    s.turn_env = TurnEnvironment(tok, {"toy": env}, max_seq_len=64, max_turns=4)
    prompt = torch.tensor([1, 2])
    drafts = asyncio.run(s.draft_streaming([prompt]))()
    s.current_step = 1
    s.bind_rows(_rows(prompt))
    asyncio.run(s.verify_sticky(drafts))

    # Turn 2 and its continuation were verified against the full row, the
    # observation inside the verified "prompt".
    assert [c[0][0] for c in calls] == [
        [1, 2, 4, EOD],
        [1, 2, 4, EOD, 50, 51, 6, 7, EOD],
        [1, 2, 4, EOD, 50, 51, 6, 8, EOD],
    ]
    assert [c[1] for c in calls] == [[2], [6], [8]]
    # Streams: the draft, turn 2 from prompt + turn 1 + observation, the
    # restart after the rejection (budgets: per-turn cap 16).
    assert [p for p, _, _ in engine.started[1:]] == [
        [[1, 2, 4, EOD, 50, 51]],
        [[1, 2, 4, EOD, 50, 51, 6, 8]],
    ]
    assert [b for _, b, _ in engine.started[1:]] == [[16], [14]]
    # The environment saw verified turns only (never the rejected 7), once each.
    assert env.calls == [("4", {"turn": 0}), ("6 8", {"turn": 1})]
    assert s.turn_env.steps == 2 and s.stats["spec/sticky_env_turns"] == 2.0
    # Anchors cover the whole row up to the last verified token: the
    # observation positions' anchors come from the turn-2 streams' prefills.
    plan = s._row_plans[0]
    assert plan.complete and plan.prefix == [4, EOD, 50, 51, 6, 8, EOD]
    assert plan.anchors.length == 2 + 7
    a = _dense(plan.anchors)
    assert bool((a[:8] >= 1).all())
    assert [t.gen_start for t in plan.turns] == [0, 4]

    # The rollout loop replays the recorded turns (no environment call).
    state, metrics, anchors = asyncio.run(s.replay_row(0))
    assert len(env.calls) == 2 and anchors is plan.anchors
    log = state["message_log"]
    assert [m["role"] for m in log] == [
        "user",
        "assistant",
        "environment",
        "assistant",
        "environment",
    ]
    assert [m["token_ids"].tolist() for m in log] == [
        [1, 2],
        [4, EOD],
        [50, 51],
        [6, 8, EOD],
        [60],
    ]
    assert log[3]["generation_logprobs"].tolist() == [-1.0, -1.0, -1.0]
    assert float(state["total_reward"]) == 1.5
    assert metrics["turn_count"] == 2 and metrics["terminated"]
    assert metrics["assistant_tokens"] == 5 and metrics["env_tokens"] == 3
    assert state["extra_env_info"] == {"turn": 1}
    assert asyncio.run(s.replay_row(0)) is None  # claimed once


class _Gen:
    """Generation backend returning fixed turns (the plain multi-turn path)."""

    cfg = {"backend": "megatron"}

    def __init__(self, turns):
        self.turns = list(turns)

    async def generate_async(self, data, greedy=False):
        n = int(data["input_lengths"][0])
        gen = self.turns.pop(0)
        ids = torch.cat([data["input_ids"][0, :n], torch.tensor(gen)])
        yield (
            0,
            BatchedDataDict(
                {
                    "output_ids": ids.view(1, -1),
                    "logprobs": torch.cat(
                        [torch.zeros(n), torch.full((len(gen),), -1.0)]
                    ).view(1, -1),
                    "generation_lengths": torch.tensor([len(gen)]),
                    "unpadded_sequence_lengths": torch.tensor([ids.numel()]),
                }
            ),
        )


@pytest.mark.parametrize("max_seq_len", [64, 9])
def test_replay_matches_the_plain_multi_turn_rollout(max_seq_len):
    """Replay builds what ``run_sample_multi_turn_rollout`` builds for the same
    turns and environment answers (incl. observation truncation at the cap)."""
    prompt = torch.tensor([1, 2])
    rows = _rows(prompt)
    turns = [[4, EOD], [6, 8, EOD]]
    plain, plain_m = asyncio.run(
        run_sample_multi_turn_rollout(
            sample_idx=0,
            initial_sample_state=rows[0],
            policy_generation=_Gen(turns),
            tokenizer=_Tokenizer(),
            task_to_env={"toy": _Env()},
            max_seq_len=max_seq_len,
            max_rollout_turns=4,
        )
    )
    env = TurnEnvironment(
        _Tokenizer(), {"toy": _Env()}, max_seq_len=max_seq_len, max_turns=4
    )
    env.bind(rows)
    prefix, records = [], []
    for gen in turns[: plain_m["turn_count"]]:
        start = len(prefix)
        prefix += gen
        rec = asyncio.run(env.step(0, 2, prefix, records, start))
        records.append(rec)
        if env.rollout_ends(records):
            break
        prefix += rec.obs
    lps = [-1.0] * len(prefix)
    state, metrics = replay(env, 0, prefix, lps, records)
    assert [m["token_ids"].tolist() for m in state["message_log"]] == [
        m["token_ids"].tolist() for m in plain["message_log"]
    ]
    assert [m["content"] for m in state["message_log"]] == [
        m["content"] for m in plain["message_log"]
    ]
    assert float(state["total_reward"]) == float(plain["total_reward"])
    assert state["extra_env_info"] == plain["extra_env_info"]
    for k in (
        "turn_count",
        "total_tokens",
        "assistant_tokens",
        "env_tokens",
        "terminated",
        "truncated",
        "max_turns_reached",
        "turn_gen_tokens",
        "turn_input_tokens",
        "turn_total_tokens",
    ):
        assert metrics[k] == plain_m[k], k


def test_rows_without_a_plan_take_the_plain_path_and_unfinished_plans_fail_loud():
    from thundersync_rl.speculative import Plan

    s = make_spec(
        {},
        base_handlers={
            "install_draft_hidden_log": lambda: True,
            "install_draft_route_log": lambda: [1],
        },
        verify_source="engine",
        sticky_delta=0.03,
    )
    s.turn_env = TurnEnvironment(
        _Tokenizer(), {"toy": _Env()}, max_seq_len=64, max_turns=2
    )
    s.bind_rows(_rows(torch.tensor([1, 2])))
    assert asyncio.run(s.replay_row(0)) is None
    s._row_plans[0] = Plan(-1, [4], [-0.1], row=0)
    with pytest.raises(RuntimeError, match="unfinished"):
        asyncio.run(s.replay_row(0))


def test_horizon_hands_later_turns_to_the_refit_engine():
    """Anchor horizon (DESIGN 14i) with multi-turn rollouts. H = 4, prompt
    [1, 2] (c = 6): turn 1 [4, EOD] ends below the horizon and is answered by
    the environment at theta_{k-1}; the next stream needs only one token (its
    prefill routes the observation: positions 0..5 = all of c). The rest of
    turn 2 and later turns run on the refit engine with the anchor key; the
    environment is stepped once per turn end on either side of the horizon."""
    engine = _Engine(
        [
            [([4, EOD], True, [-0.1] * 2, 3)],  # turn 1, routes 0..2
            [([6], True, [-0.2], 6)],  # from [1, 2, 4, EOD, 50, 51]: 0..5
        ]
    )
    results = [(2, 99), (1, 99)]

    def verify(rows, prompt_lens, engine_logq, anchors, **_):
        a, y = results.pop(0)
        return [
            (0, {"accepted": a, "next": y, "logprobs": [-1.0] * (a + 1), "draft_logq": engine_logq[0][:a]})
        ]

    registered = []
    handlers = engine.handlers()
    handlers["register_engine_anchors"] = lambda entries: registered.append(entries) or len(entries)
    handlers["engine_anchor_stats"] = lambda: {}
    s = make_spec(
        {"verify_drafts_block_engine": verify},
        base_handlers=handlers,
        gen_tail=(8, EOD),
        verify_source="engine",
        hidden_transport="driver",
        sticky_delta=0.03,
        anchor_horizon=4,
    )
    env = _Env()
    s.turn_env = TurnEnvironment(_Tokenizer(), {"toy": env}, max_seq_len=64, max_turns=4)
    prompt = torch.tensor([1, 2])
    drafts = asyncio.run(s.draft_streaming([prompt]))()
    s.current_step = 1
    s.bind_rows(_rows(prompt))
    asyncio.run(s.verify_sticky(drafts))
    assert [p for p, _, _ in engine.started[1:]] == [[[1, 2, 4, EOD, 50, 51]]]
    assert [b for _, b, _ in engine.started[1:]] == [[1]]  # H + 1 - len(prefix)
    plan = s._row_plans[0]
    assert not plan.complete and plan.prefix == [4, EOD, 50, 51, 6]
    assert plan.anchors.complete == 6 and plan.anchors.length == 6
    ((key, refs),) = registered[0].items()
    assert key == plan.engine_key and refs.length == 6
    assert env.calls == [("4", {"turn": 0})]
    # Rollout loop: turn 2 finishes on the refit engine, the environment
    # answers it, and the rollout ends there.
    state, metrics, anchors = asyncio.run(s.replay_row(0))
    (req,) = s.base.requests
    assert req["ts_anchor_key"] == [plan.engine_key]
    assert req["input_ids"][0].tolist() == [1, 2, 4, EOD, 50, 51, 6]
    assert int(req["max_new_tokens"][0]) == 15
    assert env.calls == [("4", {"turn": 0}), ("6 8", {"turn": 1})]
    assert [m["token_ids"].tolist() for m in state["message_log"]] == [
        [1, 2], [4, EOD], [50, 51], [6, 8, EOD], [60],
    ]
    assert state["message_log"][3]["generation_logprobs"].tolist() == [-1.0, -0.5, -0.5]
    assert metrics["turn_count"] == 2 and anchors is plan.anchors
    assert s.stats["spec/horizon_engine_turns"] == 2.0
