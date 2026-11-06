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
"""F3 (research/fundamental/DESIGN.md §4): per-row seeds, lazy ⊥-BV pool walk, q-side verify0 order.

- THUNDERSYNC_NULL_WAVE_MULT: wave sizes (counts only); with fixed per-row outcomes the decisions equal today's
  waves' for every multiplier, and multiplier 1 forwards fewer unused rows.
- THUNDERSYNC_ROW_SEEDS: per-row seeds are a function of (step, row identity, ordinal) only, independent of the
  call's other rows and their order; distinct across rows.
- verify0_stream_order=surprisal: needs per-row seeds; orders prompts by the q-side score.
The joint-law enumeration (content-ordered streaming + lazy waves through the production run_stream) is
research/v0s/f3_enum.py.
"""

import math
import random

import pytest
import torch

from thundersync_rl.null_bv import NullPolicy
from thundersync_rl.run_state import SeedStream
from thundersync_rl.speculative import (
    SpeculativeGeneration,
    _NullPoolRun,
    null_wave_mult,
    null_wave_size,
    row_seeds_enabled,
)
from thundersync_rl.verify0_stream import PromptCounts, Verify0Risk, schedule_chunks


def test_wave_mult_parsing(monkeypatch):
    monkeypatch.delenv("THUNDERSYNC_NULL_WAVE_MULT", raising=False)
    assert null_wave_mult() == math.inf and null_wave_size(8, 1.5) == 12
    monkeypatch.setenv("THUNDERSYNC_NULL_WAVE_MULT", "1")
    assert null_wave_size(8, 1.5) == 8 and null_wave_size(3, 1.0) == 3
    monkeypatch.setenv("THUNDERSYNC_NULL_WAVE_MULT", "1.25")
    assert null_wave_size(8, 1.5) == 10 and null_wave_size(8, 1.1) == 9
    monkeypatch.setenv("THUNDERSYNC_NULL_WAVE_MULT", "0.5")
    with pytest.raises(ValueError):
        null_wave_mult()


@pytest.mark.parametrize("seed", range(60))
def test_lazy_waves_same_outputs_in_drafting_order(monkeypatch, seed):
    """The rows the count process TAKES (successes in drafting order at their M) are identical for all multipliers."""
    policy = NullPolicy(fixed_m=1.5)
    rng = random.Random(1000 + seed)
    n, slots = rng.randint(1, 16), rng.randint(1, 8)
    u = {i: rng.random() for i in range(n)}

    def outcome(i, m):
        return (5, 1) if u[i] < 1.0 / m else (0, -1)

    taken = {}
    fwd = {}
    for mult in (None, "1"):
        if mult is None:
            monkeypatch.delenv("THUNDERSYNC_NULL_WAVE_MULT", raising=False)
        else:
            monkeypatch.setenv("THUNDERSYNC_NULL_WAVE_MULT", mult)
        run = _NullPoolRun(list(range(n)), slots, policy)
        nf = 0
        took = []
        while True:
            batch = run.next_wave()
            if not batch:
                break
            nf += len(batch)
            for i in batch:
                m = run.state.next_m()
                grid = {e: (*outcome(i, e), -1.0) for e in run.env[i]}
                res = {"null_grid": grid, "token_logprobs": [-1.0] * 10, "draft_logq": [-1.0] * 10}
                run.decide(i, res, False, 10)
                if m is not None and grid[m][1] >= 0:
                    took.append(i)
        run.close()
        taken[mult], fwd[mult] = took, nf
    assert taken[None] == taken["1"]
    assert fwd["1"] <= fwd[None]


def _spec_stub(step=3):
    s = object.__new__(SpeculativeGeneration)
    s.current_step = step
    s._verify_seeds = SeedStream(1234, "verify", lambda: s.current_step)
    return s


def _draft(key, n=5):
    return (torch.tensor([1, 2, 3]), -1, list(range(n)), True, key, [(0, 0, n)])


def test_row_seeds_independent_of_call_composition():
    a = _spec_stub()
    _, ra = a._call_seed_and_rows("block", [_draft("k1"), _draft("k2"), _draft("k3")])
    b = _spec_stub()
    _, rb = b._call_seed_and_rows("block", [_draft("k3"), _draft("k1")])
    c = _spec_stub()
    _, rc = c._call_seed_and_rows("block", [_draft("k2")])
    assert ra[0] == rb[1] and ra[2] == rb[0] and ra[1] == rc[0]  # same row -> same seed, any call, any position
    assert len(set(ra)) == 3
    # a second verification of the same row in the same step gets a fresh seed (ordinal 1)
    _, ra2 = a._call_seed_and_rows("block", [_draft("k1")])
    assert ra2[0] != ra[0]
    # another step: fresh
    d = _spec_stub(step=4)
    _, rd = d._call_seed_and_rows("block", [_draft("k1")])
    assert rd[0] != ra[0]


def test_call_seed_unchanged_by_the_refactor():
    a, b = _spec_stub(), _spec_stub()
    drafts = [_draft("k1"), _draft("k2")]
    assert a._call_seed("block", drafts) == b._call_seed_and_rows("block", drafts)[0]


def test_row_seeds_flag(monkeypatch):
    monkeypatch.delenv("THUNDERSYNC_ROW_SEEDS", raising=False)
    assert not row_seeds_enabled()
    monkeypatch.setenv("THUNDERSYNC_ROW_SEEDS", "1")
    assert row_seeds_enabled()


def test_surprisal_order_needs_row_seeds(monkeypatch):
    pcs = [
        PromptCounts(key="a", first=0, lengths=(10,), tokens=20, score=3.0),
        PromptCounts(key="b", first=1, lengths=(10,), tokens=20, score=9.0),
        PromptCounts(key="c", first=2, lengths=(10,), tokens=20, score=9.0),
    ]
    monkeypatch.delenv("THUNDERSYNC_ROW_SEEDS", raising=False)
    with pytest.raises(ValueError, match="ROW_SEEDS"):
        schedule_chunks(pcs, Verify0Risk(), 1, 1, "surprisal")
    monkeypatch.setenv("THUNDERSYNC_ROW_SEEDS", "1")
    chunks = schedule_chunks(pcs, Verify0Risk(), 1, 1, "surprisal")
    assert [c[0] for c in chunks] == ["b", "c", "a"]  # highest score first, drafting order on ties
    assert [c[0] for c in schedule_chunks(pcs, Verify0Risk(), 1, 1, "drafting")] == ["a", "b", "c"]


def test_per_row_generator_draws_do_not_depend_on_order():
    """The worker's per-row generator (vit.row_generator(seed)): a row's eta / u are the same whatever was drawn
    before it in the call."""
    from thundersync_rl import vit

    seeds = [11, 22, 33]
    first = {s: torch.rand(8, generator=vit.row_generator(s, "cpu"), dtype=torch.float64) for s in seeds}
    for order in ([0, 1, 2], [2, 0, 1]):
        for k in order:
            g = vit.row_generator(seeds[k], "cpu")
            assert torch.equal(torch.rand(8, generator=g, dtype=torch.float64), first[seeds[k]])
