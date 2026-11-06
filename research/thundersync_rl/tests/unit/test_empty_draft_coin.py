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
"""Empty drafts in the ⊥-BV pool walk (REVIEW batch 5, E1): an empty (cut-before-first-token / unlogged) draft is
walked at its drafting index with a keyed coin (⊥ with probability 1 - 1/M, else a fresh p decode). Exact
enumeration where emptiness correlates with a sibling's content: dropping empties biases the kept pools' outputs,
the keyed coin does not. Also: the coin's rate and keying, and a stand-down-cut draft through the real response
parser and drafts_of_record."""

from __future__ import annotations

import asyncio
import itertools
from types import SimpleNamespace

from thundersync_rl.null_bv import NullGroup, NullPolicy
from thundersync_rl.speculative import empty_draft_result, empty_draft_uniform

P = (0.6, 0.4)
Q = (0.3, 0.7)
M = 1.5
G, N = 2, 3


def _draft_outcomes(x):
    """Single-token ⊥-BV of draft token x ~ q against p/M (+) (1 - 1/M)⊥: {outcome: prob}, outcome a token or None (⊥)."""
    a = min(1.0, P[x] / (M * Q[x]))
    out = {x: a}
    res = [max(P[y] / M - Q[y], 0.0) for y in range(2)] + [1.0 - 1.0 / M]
    z = sum(res)
    for y in range(2):
        out[y] = out.get(y, 0.0) + (1 - a) * res[y] / z
    out[None] = (1 - a) * res[2] / z
    return out


def _empty_outcomes():
    return {0: P[0] / M, 1: P[1] / M, None: 1.0 - 1.0 / M}


def _kept_output_law(rule):
    """Exact law of the first G outputs of kept pools (prompt drop at constant M), draft 3 empty iff x1 == 0."""
    pol = NullPolicy(fixed_m=M, prompt_drop=True)
    law, kept_p = {}, 0.0
    for xs in itertools.product(range(2), repeat=N):
        px = Q[xs[0]] * Q[xs[1]] * Q[xs[2]]
        empty3 = xs[0] == 0  # a sibling's content moves the timing that empties draft 3
        dists = [_draft_outcomes(xs[0]), _draft_outcomes(xs[1])]
        if not empty3:
            dists.append(_draft_outcomes(xs[2]))
        elif rule == "coin":
            dists.append(_empty_outcomes())
        for outs in itertools.product(*[list(d.items()) for d in dists]):
            pr = px
            for _, p_ in outs:
                pr *= p_
            st = NullGroup(len(dists), G, pol)
            ys = []
            for y, _ in outs:
                if st.next_m() is None:
                    break
                st.record(y is not None)
                if y is not None:
                    ys.append(y)
            st.next_m()
            if st.dropped or len(ys) < G:
                continue
            kept_p += pr
            law[tuple(ys[:G])] = law.get(tuple(ys[:G]), 0.0) + pr
    return {k: v / kept_p for k, v in law.items()}


def _max_dev(law):
    return max(abs(law.get((a, b), 0.0) - P[a] * P[b]) for a in range(2) for b in range(2))


def test_dropping_empties_biases_kept_pools_and_the_keyed_coin_does_not():
    assert _max_dev(_kept_output_law("coin")) < 1e-12
    assert _max_dev(_kept_output_law("drop")) > 1e-3


def test_empty_coin_rate_and_keying():
    us = [empty_draft_uniform(42, "nl0", 7, (1, 2, 3), j) for j in range(20000)]
    outs = [empty_draft_result(u, (M,))["null_grid"][M][1] >= 0 for u in us]
    assert abs(sum(outs) / len(outs) - 1.0 / M) < 0.015
    assert us[:50] == [empty_draft_uniform(42, "nl0", 7, (1, 2, 3), j) for j in range(50)]
    assert us[:50] != [empty_draft_uniform(42, "nl0", 8, (1, 2, 3), j) for j in range(50)]
    assert us[:50] != [empty_draft_uniform(42, "nl0", 7, (1, 2, 4), j) for j in range(50)]
    r = empty_draft_result(0.5, (1.0, 1.5, 2.5))
    assert r["null_empty"] and r["null_grid"][1.0][1] == 0 and r["null_grid"][1.5][1] == 0 and r["null_grid"][2.5][1] == -1


def test_a_draft_cut_at_the_slot_gate_parses_to_an_unfinished_empty_draft():
    """T1: the stand-down cut's gate result through the real response parser (AsyncRolloutImpl._generate_response) and
    drafts_of_record."""
    import torch

    from nemo_rl.experience.rollout_manager import AsyncRolloutImpl
    from thundersync_rl import engine_slots, nolanes

    class _Gen:
        async def generate_async(self, data, greedy=False):
            raise AssertionError("a cut draft must not be sent")
            yield  # pragma: no cover

    gen = _Gen()
    slots = engine_slots.install([gen], 4)
    slots.cut_waiting_drafts()
    impl = SimpleNamespace(
        _tokenizer=SimpleNamespace(pad_token_id=0, decode=lambda ids, skip_special_tokens=True: ""),
        _policy_generation=gen, _timeouts=SimpleNamespace(generation_s=None), _deadline_registry=None,
    )
    prompt = [{"role": "user", "token_ids": torch.tensor([11, 12, 13])}]
    msg, _lens, _m = asyncio.run(AsyncRolloutImpl._generate_response(impl, prompt, None))
    assert msg["token_ids"].numel() == 0 and msg["generation_logprobs"].numel() == 0
    rec = SimpleNamespace(completions=[SimpleNamespace(message_log=prompt + [msg])])
    rows = nolanes.drafts_of_record("g", prompt, rec)
    assert len(rows) == 1 and rows[0].tokens == [] and not rows[0].finished and rows[0].key is None
