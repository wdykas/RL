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
"""Lazy block verification is pathwise identical to the eager rule (CPU)."""

import itertools

import pytest
import torch
from thundersync_rl.block_verification import (
    acceptance_weights,
    block_verify,
    block_verify_lazy,
    lazy_stop_index,
    stop_probabilities,
)


def _case(gen, vocab, g, temp, noise):
    """Random target/draft log-probs; the draft is sampled from q."""
    p = torch.randn(g + 1, vocab, generator=gen, dtype=torch.float64) * temp
    q = p[:g] + torch.randn(g, vocab, generator=gen, dtype=torch.float64) * noise
    lp, lq = torch.log_softmax(p, -1), torch.log_softmax(q, -1)
    draft = (
        torch.multinomial(lq.exp(), 1, generator=gen).flatten()
        if g
        else torch.zeros(0, dtype=torch.long)
    )
    return lp, lq, draft


def _same_path(lp, lq, draft, seed):
    g1 = torch.Generator().manual_seed(seed)
    g2 = torch.Generator().manual_seed(seed)
    out = block_verify(lp, lq, draft, g1)
    assert block_verify_lazy(lp, lq, draft, g2) == out
    assert torch.equal(g1.get_state(), g2.get_state())
    return out


@pytest.mark.parametrize("noise", [0.0, 0.01, 0.3, 1.0, 3.0])
def test_lazy_matches_eager_randomized(noise):
    gen = torch.Generator().manual_seed(int(noise * 100))
    rejected = 0
    for trial in range(300):
        vocab = int(torch.randint(2, 9, (1,), generator=gen))
        g = int(torch.randint(0, 12, (1,), generator=gen))
        temp = float(torch.rand(1, generator=gen)) * 3
        lp, lq, draft = _case(gen, vocab, g, temp, noise)
        tau, _ = _same_path(lp, lq, draft, trial)
        rejected += tau < g
    if noise >= 0.3:
        assert rejected > 30  # the lazy backward scan is actually exercised


def test_lazy_matches_eager_float32_inputs():
    gen = torch.Generator().manual_seed(7)
    for trial in range(200):
        lp, lq, draft = _case(gen, 50, 16, 2.0, 0.5)
        _same_path(lp.float(), lq.float(), draft, trial)


@pytest.mark.parametrize(
    "vocab,g,temp,seed", [(3, 3, 1.0, 0), (2, 4, 2.0, 1), (4, 2, 0.3, 2)]
)
def test_lazy_matches_eager_every_draft(vocab, g, temp, seed):
    """Every draft of the exhaustive-enumeration cases, many generator seeds."""
    gen = torch.Generator().manual_seed(seed)
    for draft in itertools.product(range(vocab), repeat=g):
        lp = torch.log_softmax(
            torch.randn(g + 1, vocab, generator=gen, dtype=torch.float64) * temp, -1
        )
        lq = torch.log_softmax(
            torch.randn(g, vocab, generator=gen, dtype=torch.float64) * temp, -1
        )
        for s in range(20):
            _same_path(lp, lq, torch.tensor(draft), s)


def test_identical_models_accept_everything_lazily():
    gen = torch.Generator().manual_seed(0)
    lp = torch.log_softmax(torch.randn(6, 5, generator=gen, dtype=torch.float64), -1)
    draft = torch.tensor([1, 4, 0, 2, 3])
    out = block_verify_lazy(lp, lp[:5], draft, torch.Generator().manual_seed(1))
    assert out[0] == 5


def test_lazy_stop_index_matches_eager_rule():
    """Random b / eta / r (r <= b, including b = 1 and tiny r) against the max rule."""
    gen = torch.Generator().manual_seed(3)
    evaluated = total = 0
    for _ in range(5000):
        g = int(torch.randint(0, 40, (1,), generator=gen))
        scale = float(torch.rand(1, generator=gen))
        b = acceptance_weights(
            torch.randn(g, generator=gen, dtype=torch.float64) * scale
        )
        frac = torch.rand(g, generator=gen, dtype=torch.float64)
        frac[torch.rand(g, generator=gen) < 0.2] = 1.0  # r = b
        frac[torch.rand(g, generator=gen) < 0.1] = 1e-15  # h far below 1 even at b = 1
        r = b[:g] * frac
        eta = torch.rand(g + 1, generator=gen, dtype=torch.float64)
        hits = torch.nonzero(eta <= stop_probabilities(r, b)).flatten()
        calls = []

        def mass(idx, r=r, calls=calls):
            calls.append(idx.clone())
            return r[idx]

        if hits.numel() == 0:
            with pytest.raises(RuntimeError):
                lazy_stop_index(b, eta, mass)
            continue
        tau, n = lazy_stop_index(b, eta, mass)
        assert tau == int(hits.max())
        seen = torch.cat(calls) if calls else torch.zeros(0, dtype=torch.long)
        assert n == seen.numel()
        # Batches are evaluated from the top; only the last one reaches tau.
        assert all((c > tau).all() for c in calls[:-1])
        evaluated += n
        total += g
    assert evaluated < 0.5 * total
