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
"""Sticky continuation rounds at low acceptance stay linear in the row (no Ray, no GPUs).

research/liveness_lowacc: Super-L8, forced lr 3e-4, sticky_continuation_chunk_tokens 16384 (every piece the
whole remainder). A continuation rejected every ~37 tokens re-decoded and re-verified a ~7k-token remainder per
rejection; step 3's rollout never finished (the engine and learner stayed busy, nothing hung). The run cap
(``sticky_run_factor``) bounds a rejected plan's pieces by its mean accepted run.
"""

import asyncio

import pytest
import thundersync_rl.speculative as spec_mod
import torch
from test_speculative_driver import _spec as make_spec
from test_sticky_driver import _DecodingEngine

from thundersync_rl.speculative import STICKY_MIN_PIECE, Plan, SpeculativeGeneration

RESIDUAL = 99


@pytest.fixture(autouse=True)
def _no_ray(monkeypatch):
    monkeypatch.setattr(spec_mod.ray, "get", lambda x: x)


def _tok(g: int) -> int:
    """The drafter's token at generated position g (never EOD = 2, never the residual)."""
    return 10 + g % 7


class _PositionEngine(_DecodingEngine):
    """Continuation streams decode ``_tok(position)`` up to their budget, ``rate`` tokens per status poll.

    The drafter is a deterministic function of the position, so the verified rollout does not depend on how the
    rounds cut it into pieces; ``decoded()`` is the engine's decode work (tokens decoded by continuation streams).
    """

    def __init__(self, rounds, rate):
        super().__init__(rounds)
        self.rate = rate

    def _add(self, prompts, budget, hidden_keys, cont):
        if not cont:
            return super()._add(prompts, budget, hidden_keys, cont)
        self.started.append((prompts, budget, hidden_keys))
        base = len(self.streams)
        for p, b, k in zip(prompts, budget, hidden_keys or [None] * len(prompts)):
            g0 = len(p) - 1  # prompt [1]: generated position of the stream's first token
            toks = [_tok(g0 + i) for i in range(b)]
            self.streams.append(
                dict(tokens=toks, done=True, logq=[-0.2] * b, n_pos=0, key=k, routed=0, plen=len(p),
                     more=None, cont=True, peeked=False)
            )
        return list(range(base, len(self.streams)))

    def lengths(self, indices):
        out = []
        for i in indices:
            s = self.streams[i]
            if i not in self.aborted:
                s["visible"] = min(self._vis(s) + self.rate, len(self._full(s)))
            out.append((self._vis(s), self._vis(s) == len(self._full(s))))
        return out

    def decoded(self):
        return sum(s.get("visible", 0) for s in self.streams if s["cont"])


def _run(length, period, run_factor, chunk=16384, rate=64):
    """One row of ``length`` tokens whose drafts are rejected at every generated position = -1 mod ``period``."""
    engine = _PositionEngine(
        [[([_tok(0)], False, [-0.1], 2, [_tok(g) for g in range(1, length)])]], rate
    )
    calls = []

    def verify(rows, prompt_lens, engine_logq, anchors, **_):
        out = []
        for i, (r, plen) in enumerate(zip(rows, prompt_lens)):
            piece = r.tolist()[plen:]
            calls.append(len(piece))
            a, y = len(piece), RESIDUAL
            for j, t in enumerate(piece):
                if (plen - 1 + j) % period == period - 1 and t != RESIDUAL:
                    a = j
                    break
            out.append((i, {"accepted": a, "next": y, "logprobs": [-1.0] * (a + 1),
                            "draft_logq": engine_logq[i][:a]}))
        return out

    s = make_spec(
        {"verify_drafts_block_engine": verify},
        base_handlers=engine.handlers(),
        verify_source="engine",
        hidden_transport="driver",
        sticky_delta=0.03,
        sticky_chunk_tokens=chunk,
        sticky_run_factor=run_factor,
    )
    s.max_new_tokens = length
    drafts = asyncio.run(s.draft_streaming([torch.tensor([1])]))()
    s.current_step = 1
    asyncio.run(s.verify_sticky(drafts))
    (plan,) = s.plans[(1,)]
    return plan, engine.decoded(), calls, s.stats


def _expected(length, period):
    return [RESIDUAL if g % period == period - 1 else _tok(g) for g in range(length)]


def test_low_acceptance_continuation_work_is_linear_in_the_row():
    """Uncapped whole-remainder pieces re-decode the remainder per rejection (quadratic: the never-finishing
    step); the run cap keeps the decode and verified tokens O(L), and the rollout is the same."""
    length, period = 4096, 100
    plan0, dec0, calls0, st0 = _run(length, period, run_factor=0.0)
    plan1, dec1, calls1, st1 = _run(length, period, run_factor=4.0)
    want = _expected(length, period)
    assert plan0.complete and plan0.prefix == want
    assert plan1.complete and plan1.prefix == want
    rejects = length // period
    assert plan0.cont_rejects == plan1.cont_rejects == rejects
    # The bug: every rejection re-decodes and re-verifies the remainder, ~L^2 / (2 period).
    assert dec0 >= 15 * length and sum(calls0) >= 15 * length
    # Capped: pieces of <= max(128, 4 x period) tokens after the first rejection.
    assert dec1 <= 5 * length and sum(calls1) <= 5 * length
    assert max(calls1[2:]) <= max(STICKY_MIN_PIECE, 4 * period) + 1
    assert st1["spec/sticky_rounds"] <= rejects + length // (4 * period) + 4


def test_high_acceptance_pieces_are_unchanged():
    """A row whose continuation is never rejected is cut exactly as before (one whole-remainder piece)."""
    length = 2048
    plan0, dec0, calls0, _ = _run(length, 10**9, run_factor=0.0)
    plan1, dec1, calls1, _ = _run(length, 10**9, run_factor=4.0)
    assert plan0.prefix == plan1.prefix == [_tok(g) for g in range(length)]
    assert calls0 == calls1 and dec0 == dec1 and plan1.cont_rejects == 0


def test_piece_cap_is_a_function_of_decided_history():
    s = SpeculativeGeneration.__new__(SpeculativeGeneration)
    s.sticky_chunk_tokens, s.sticky_chunk_growth, s.sticky_run_factor = 16384, 0.0, 4.0
    plan = Plan(seed=0, prefix=[5] * 1000, verified=(4000, 999))
    # No continuation rejection yet: the position rule only (the first piece is never capped).
    assert s._piece_tokens(1000, plan) == 16384
    plan.cont_rejects = 1  # + the draft's rejection: 2 rejections over 1000 decided tokens
    assert s._piece_tokens(1000, plan) == 2000
    plan.cont_rejects = 39
    assert s._piece_tokens(1000, plan) == STICKY_MIN_PIECE  # ceil(4 x 1000 / 40) = 100 -> floor
    plan.verified = (4000, 4000)  # whole-accepted draft: only the continuation rejections count
    plan.cont_rejects = 10
    assert s._piece_tokens(1000, plan) == 400
    # Never above the position rule; production's fixed 128 chunk is never capped.
    s.sticky_chunk_tokens = 128
    assert s._piece_tokens(1000, plan) == 128
    s.sticky_chunk_tokens, s.sticky_run_factor = 16384, 0.0  # off: the old rule
    assert s._piece_tokens(1000, plan) == 16384
    assert s._piece_tokens(1000) == 16384
