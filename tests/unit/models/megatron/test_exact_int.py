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
"""Exact integer categorical law (exact_int.py): combinatorial proofs on small cases."""

import random
from collections import Counter

import numpy as np
import pytest
import torch

from nemo_rl.models.megatron import exact_int as xi
from nemo_rl.models.megatron import exact_nucleus


def test_words_vectorized_equal_reference():
    rng = random.Random(0)
    for _ in range(200):
        salt = rng.getrandbits(64)
        a = rng.randrange(-(2**63), 2**63)
        b = rng.randrange(0, 2**40)
        k = rng.randrange(0, 50)
        ref = xi.word_py(salt, a, b, xi.ETA, k)
        got = int(xi.words_np(salt, np.array([a]), np.array([b]), xi.ETA, np.array([k]))[0])
        assert got == ref
    # broadcasting, signed salts
    w = xi.words_np(-5, np.arange(4), 7, xi.RESID, 0)
    assert [int(x) for x in w] == [xi.word_py(-5, i, 7, xi.RESID, 0) for i in range(4)]


@pytest.mark.parametrize("bits,W", [(8, 7), (10, 37), (10, 1000), (12, 4095), (12, 2049)])
def test_uniform_below_every_word(bits, W):
    """Every accepted word maps to each r in [0, W) the same number of times (no modulo
    bias): enumerate all 2^bits first words."""
    counts = Counter()
    rejected = 0
    for u in range(1 << bits):
        first = u << (64 - bits)

        def words(k, first=first):
            return first if k == 0 else 0  # second word: u = 0 < t always accepted

        r, used = xi.uniform_below(W, words, bits)
        if used == 1:
            counts[r] += 1
        else:
            rejected += 1
    per = ((1 << bits) - 1) // W
    assert set(counts) == set(range(W))
    assert all(c == per for c in counts.values())
    assert rejected == (1 << bits) - per * W


def test_prefix_selection_every_r():
    """Every r in [0, W) selects the token v with C_{v-1} <= r < C_v: token v gets exactly
    w_v of the W values (the sampled law is w / W exactly), python and torch paths alike."""
    rng = random.Random(1)
    for _ in range(20):
        V = rng.randrange(1, 9)
        w = [rng.randrange(0, 40) for _ in range(V)]
        if sum(w) == 0:
            w[rng.randrange(V)] = 3
        W = sum(w)
        got = Counter(xi.select_by_prefix(w, r) for r in range(W))
        assert dict(got) == {v: x for v, x in enumerate(w) if x}
        cum = torch.cumsum(torch.tensor(w, dtype=torch.int64), 0)
        sel = torch.searchsorted(cum, torch.arange(W).view(-1, 1), right=True)[:, 0]
        assert sel.tolist() == [xi.select_by_prefix(w, r) for r in range(W)]


def test_sample_weights_matches_reference():
    torch.manual_seed(0)
    w = torch.randint(0, 2**52, (64, 50), dtype=torch.int64)
    w[:, ::3] = 0
    a = np.arange(64) * 7 + 3
    b = np.arange(64) % 5
    got = xi.sample_weights(w, 1234, a, b, xi.TEST).tolist()
    ref = [
        xi.sample_weights_py(w[i].tolist(), lambda k, i=i: xi.word_py(1234, int(a[i]), int(b[i]), xi.TEST, k))
        for i in range(64)
    ]
    assert got == ref
    assert all(int(w[i, t]) > 0 for i, t in enumerate(got))


def test_weights_are_exact_nucleus_weights_and_shard_invariant():
    torch.manual_seed(0)
    z = torch.randn(9, 1000) * 6
    lp = torch.log_softmax(z, -1)
    w = xi.int_weights(lp)
    assert torch.equal(w, exact_nucleus.mass_weights(lp))
    W, wx = xi.row_stats(lp, torch.arange(9) * 11)
    assert torch.equal(W, w.sum(-1)) and torch.equal(wx, w[torch.arange(9), torch.arange(9) * 11])
    assert int(W.max()) < 2**53 and int(W.min()) > 2**51
    for tp in (2, 4, 8):
        shards = lp.view(9, tp, -1)
        Ws = sum(xi.row_stats(shards[:, s].contiguous())[0] for s in range(tp))
        assert torch.equal(Ws, W)  # integer sums: any split, any order
    # rows are independent of the batch (chunking / permutation)
    perm = torch.randperm(9)
    assert torch.equal(xi.row_stats(lp[perm])[0], W[perm])


def test_int_logprob_value():
    lp = torch.log_softmax(torch.tensor([[0.0, 1.0, -50.0, 2.0, float("-inf")]]), -1)
    W, wx = xi.row_stats(lp, torch.tensor([3]))
    lam = xi.int_logprob(lp, torch.tensor([[3]]))
    assert float(lam) == float(torch.log(wx.double()) - torch.log(W.double()))
    assert abs(float(lam) - float(lp[0, 3])) < 1e-6
    # weight-0 target (p < 2^-52: never sampled): the finite raw value is kept, and counted
    before = xi.stats()["zero_weight_targets"]
    tv = xi.training_value(lp[None], torch.tensor([[2]]))
    assert float(tv) == float(lp[0, 2]) and xi.stats()["zero_weight_targets"] == before + 1
    assert tv.dtype == torch.float32
    tv3 = xi.training_value(lp[None], torch.tensor([[3]]))
    assert float(tv3) == float(lam.float())


def test_sharded_sample_equals_gathered():
    torch.manual_seed(3)
    R, V = 12, 64
    lp = torch.log_softmax(torch.randn(R, V) * 4, -1)
    w = xi.int_weights(lp)
    salt = 99
    a = np.arange(R)

    def r_of_W(W):
        Wn = W.numpy()
        out = []
        for i in range(R):
            r, _ = xi.uniform_below(int(Wn[i]), lambda k, i=i: xi.word_py(salt, int(a[i]), 0, xi.ENGINE, k))
            out.append(r)
        return torch.tensor(out)

    ref = xi.sample_weights(w, salt, a, 0, xi.ENGINE)
    for world in (1, 2, 4, 8):
        shards = list(w.view(R, world, V // world).unbind(1))
        parts = []
        for rank in range(world):
            part = torch.zeros(R, world, dtype=torch.int64)
            part[:, rank] = shards[rank].sum(-1)
            parts.append(part)
        total = sum(parts)
        toks = []
        for rank in range(world):
            t, W = xi.sharded_sample(shards[rank].contiguous(), rank, world, r_of_W, lambda x: total, lambda x: x)
            toks.append(t)
        tok = torch.stack(toks).amax(0)
        assert torch.equal(tok, ref), world
        assert torch.equal(W, w.sum(-1))


def test_law_weights_top_p_is_nucleus_restriction():
    torch.manual_seed(5)
    lp = torch.log_softmax(torch.randn(6, 200) * 3, -1)
    w = xi.int_weights(lp)
    wn = xi.law_weights(lp, 0.9)
    keep = exact_nucleus.plain_mask(w, 0.9)
    assert torch.equal(wn, torch.where(keep, w, torch.zeros_like(w)))
    assert torch.equal(xi.law_weights(lp, None), w)
    # the restricted law reaches the nucleus mass threshold
    assert bool((wn.sum(-1) >= exact_nucleus.mass_threshold(w.sum(-1), 0.9)).all())


def test_engine_cpu_sampler_matches_reference():
    torch.manual_seed(7)
    lp = torch.log_softmax(torch.randn(10, 300) * 5, -1)
    a = torch.arange(10) + 1000
    b = torch.arange(10) * 3
    tok, W, wx = xi.sample_logprob_rows(lp, a, b, 42)
    ref = xi.sample_weights(xi.int_weights(lp), 42, a.numpy(), b.numpy(), xi.ENGINE)
    assert torch.equal(tok, ref)
    assert torch.equal(W, xi.int_weights(lp).sum(-1))
    assert torch.equal(wx, xi.int_weights(lp)[torch.arange(10), tok])


def test_learner_hook_value_is_lambda_and_gradient_is_straight_through():
    """Under exact_int the BI log-prob path returns fp32(lambda) (the logged value) and its
    gradient is d log_softmax, bitwise the plain path's (straight-through)."""
    from nemo_rl.models.megatron import zero_train_gen_mismatch as z

    torch.manual_seed(11)
    logits = (torch.randn(2, 7, 300) * 4).requires_grad_(True)
    target = torch.randint(0, 300, (2, 7))
    g_out = torch.randn(2, 7)
    try:
        xi.configure(False)
        plain = z._LocalVocabLogprob.apply(logits, target, 3)
        (gp,) = torch.autograd.grad((plain * g_out).sum(), logits)
        xi.configure(True)
        val = z._LocalVocabLogprob.apply(logits, target, 3)
        (gx,) = torch.autograd.grad((val * g_out).sum(), logits)
    finally:
        xi.configure(False)
    lp = torch.log_softmax(logits.detach().float(), -1)
    want = xi.int_logprob(lp, target).float()
    assert torch.equal(val.detach(), want)
    assert torch.equal(gx, gp)
    # the value differs from the fp32 log_softmax only by the integer law's definition
    assert float((val.detach() - plain.detach()).abs().max()) < 1e-5
