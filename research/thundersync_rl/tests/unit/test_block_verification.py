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
"""Block verification returns exact samples of the target (exact enumeration)."""

import itertools
import math

import pytest
import torch
from thundersync_rl.block_verification import (
    block_verify,
    block_weights,
    output_distribution_given_draft,
)


def _random_lm(vocab: int, depth: int, gen: torch.Generator, temp: float):
    return {
        pre: torch.log_softmax(
            torch.randn(vocab, generator=gen, dtype=torch.float64) * temp, -1
        )
        for n in range(depth + 1)
        for pre in itertools.product(range(vocab), repeat=n)
    }


def _logprob(lm, seq):
    return sum(float(lm[tuple(seq[:i])][t]) for i, t in enumerate(seq))


@pytest.mark.parametrize(
    "vocab,g,temp,seed", [(3, 3, 1.0, 0), (2, 4, 2.0, 1), (4, 2, 0.3, 2)]
)
def test_block_verification_output_is_exactly_p(vocab, g, temp, seed):
    gen = torch.Generator().manual_seed(seed)
    P = _random_lm(vocab, g, gen, temp)
    Q = _random_lm(vocab, g, gen, temp)
    out = {s: 0.0 for s in itertools.product(range(vocab), repeat=g + 1)}
    for draft in itertools.product(range(vocab), repeat=g):
        q_draft = _logprob(Q, draft)
        lp = torch.stack([P[draft[:i]] for i in range(g + 1)])
        lq = torch.stack([Q[draft[:i]] for i in range(g)])
        for tau, py in output_distribution_given_draft(lp, lq, torch.tensor(draft)):
            for y in range(vocab):
                head = list(draft[:tau]) + [y]
                for rest in itertools.product(range(vocab), repeat=g + 1 - len(head)):
                    seq = tuple(head) + rest
                    tail = sum(
                        float(P[seq[: len(head) + j]][seq[len(head) + j]])
                        for j in range(len(rest))
                    )
                    out[seq] += math.exp(q_draft + tail) * float(py[y])
    tv = 0.5 * sum(abs(out[s] - math.exp(_logprob(P, s))) for s in out)
    assert tv < 1e-10


def test_identical_models_accept_everything():
    gen = torch.Generator().manual_seed(0)
    lp = torch.log_softmax(torch.randn(6, 5, generator=gen, dtype=torch.float64), -1)
    draft = torch.tensor([1, 4, 0, 2, 3])
    h, b, _ = block_weights(lp, lp[:5], draft)
    assert torch.allclose(b, torch.ones_like(b)) and float(h[-1]) == 1.0
    assert block_verify(lp, lp[:5], draft, torch.Generator().manual_seed(1))[0] == 5
