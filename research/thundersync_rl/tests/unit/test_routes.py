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
"""Drafter route log and anchors (routes.py), and exactness of the sticky
rollout scheme (block verification + theta_n continuation rounds) by exact
enumeration on a toy routed language model."""

import itertools
import math
from types import SimpleNamespace

import pytest
import torch
from thundersync_rl.block_verification import output_distribution_given_draft
from thundersync_rl.routes import (
    RouteLog,
    anchored_draft_length,
    merge_route_parts,
    splice,
    step_route_entries,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda)])
def test_route_log_keys_by_absolute_position(device):
    log = RouteLog(chunk_rows=4)
    L, K = 3, 2
    # step 1: prefill of draft a (positions 0..2) and an untagged request
    rows = torch.arange(5 * L * K, device=device).view(5, L, K)
    log.record([(0, "a", 0), (1, "a", 1), (2, "a", 2)], rows)
    # step 2: decode of a (position 3) at row 1; b at row 0
    rows2 = 100 + torch.arange(2 * L * K, device=device).view(2, L, K)
    log.record([(1, "a", 3), (0, "b", 7)], rows2)
    # an eviction recomputes position 1: overwrite
    rows3 = 200 + torch.arange(L * K, device=device).view(1, L, K)
    log.record([(0, "a", 1)], rows3)
    pos, r = log.pop("a")
    assert pos.tolist() == [0, 1, 2, 3]
    assert torch.equal(r[0], rows[0].cpu().to(r.dtype))
    assert torch.equal(r[1], rows3[0].cpu().to(r.dtype))
    assert torch.equal(r[3], rows2[1].cpu().to(r.dtype))
    log.record([(0, "a", 9)], rows)  # closed: dropped
    assert log.pop("a") is None
    assert log.pop("b")[0].tolist() == [7]
    assert log.live_rows() == 0


def test_step_entries_under_sequence_parallelism():
    slots = [4, 4, 4, 9, 2, 2]
    pos = [0, 1, 2, 30, 5, 6]
    keys = {4: "a", 2: "b"}
    full = step_route_entries(slots, pos, keys, 0, 6)
    assert full == [(0, "a", 0), (1, "a", 1), (2, "a", 2), (4, "b", 5), (5, "b", 6)]
    # padded token count 8 at TP=2: rank 0 routes tokens 0-3, rank 1 tokens 4-7
    r0 = step_route_entries(slots, pos, keys, 0, 4)
    r1 = step_route_entries(slots, pos, keys, 4, 4)
    assert r0 == full[:3] and r1 == [(0, "b", 5), (1, "b", 6)]


def test_merge_parts_from_pipeline_stages_and_sp_ranks():
    K = 2
    a = torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int16)  # layer 1, positions 0,1
    b = torch.tensor([[[5, 6], [7, 0]]], dtype=torch.int16)  # layers 3, 4, position 0
    c = torch.tensor([[[8, 9], [1, 3]]], dtype=torch.int16)  # layers 3, 4, position 1
    parts = [
        ([1], torch.tensor([0, 1]), a),
        ([3, 4], torch.tensor([0]), b),
        ([3, 4], torch.tensor([1]), c),
    ]
    out, layers = merge_route_parts(parts, 3)
    assert layers == [1, 3, 4] and out.shape == (3, 3, K)
    assert out[0].tolist() == [[1, 2], [5, 6], [7, 0]]
    assert out[1].tolist() == [[3, 4], [8, 9], [1, 3]]
    assert (out[2] == -1).all()


def test_anchored_length_and_splice():
    A = torch.full((7, 2, 1), -1, dtype=torch.int16)
    A[:5] = 1  # positions 0..4 logged
    # prompt 3, draft 4: tokens need positions up to P+m-2 -> m <= 5-3+1 = 3
    assert anchored_draft_length(A, 3, 4) == 3
    assert anchored_draft_length(A, 3, 2) == 2
    assert anchored_draft_length(A, 6, 4) == 0  # prompt not fully logged
    A[2, 1] = -1  # one layer missing at position 2
    assert anchored_draft_length(A, 2, 4) == 1
    B = torch.full((9, 2, 1), -1, dtype=torch.int16)
    B[4:8] = 3
    S = splice(A[:5], B)
    assert S.shape[0] == 9 and S[0, 0, 0] == 1 and S[4, 0, 0] == 3 and S[8, 0, 0] == -1
    assert S[2, 1, 0] == -1 and S[2, 0, 0] == 1


# --------------------------------------------------------------------------
# Exactness of the sticky rollout scheme (exact enumeration)
# --------------------------------------------------------------------------
#
# Toy MoE language model: tokens {0, 1, EOS=2}, rollout cap N. At every
# position the router picks one of E experts; the next-token law depends on
# the routes of all positions so far. Drafter q = plain theta_n. Target
# p = F(theta_{n+1}; A) with A = plain theta_n routes on the same prefix
# (sticky rule with threshold delta) at every position up to the cap. The
# driver scheme: draft from q, cut by an exogenous time (random length),
# block-verify against p; a whole-accepted unfinished draft emits no bonus
# (its last route is not logged) and continues; every continuation is a
# theta_n draft to the end, block-verified again. The output must be p.

V, EOS, E = 3, 2, 3


class _Toy:
    def __init__(self, seed, delta, cap):
        self.g = torch.Generator().manual_seed(seed)
        self.delta, self.cap = delta, cap
        self.scores = {}
        self.logits = {}

    def _score(self, ver, prefix):
        k = (ver, prefix)
        if k not in self.scores:
            base = self.scores.get((0, prefix))
            if base is None or ver == 0:
                s = torch.randn(E, generator=self.g, dtype=torch.float64) * 0.1
            else:  # small update: near ties flip
                s = base + 0.2 * torch.randn(E, generator=self.g, dtype=torch.float64)
            self.scores[k] = s
        return self.scores[k]

    def plain_routes(self, ver, seq):
        return tuple(int(self._score(ver, tuple(seq[: j + 1])).argmax()) for j in range(len(seq)))

    def sticky_routes(self, ver, seq, anchors):
        out = []
        for j, a in enumerate(anchors):
            s = self._score(ver, tuple(seq[: j + 1]))
            others = torch.cat([s[:a], s[a + 1 :]])
            out.append(a if s[a] >= others.max() - self.delta else int(s.argmax()))
        return tuple(out)

    def _lp(self, ver, seq, routes):
        k = (ver, tuple(seq), routes)
        if k not in self.logits:
            self.logits[k] = torch.log_softmax(
                torch.randn(V, generator=self.g, dtype=torch.float64), -1
            )
        return self.logits[k]

    def q(self, seq):  # drafter: plain theta_n (version 0)
        return self._lp(0, seq, self.plain_routes(0, seq))

    def p(self, seq, anchor_fn=None):  # target: sticky theta_{n+1} (version 1)
        anchors = (anchor_fn or (lambda s: self.plain_routes(0, s)))(seq)
        return self._lp(1, seq, self.sticky_routes(1, seq, anchors))


def _seq_logprob(fn, prompt, gen):
    return sum(float(fn(prompt + gen[:i])[t]) for i, t in enumerate(gen))


def _drafts(toy, prefix, prompt, max_len):
    """(draft, finished, q prob) for every complete q continuation of ``prefix``."""
    out = []

    def rec(d, lq):
        if (d and d[-1] == EOS) or len(prefix) + len(d) >= toy.cap or len(d) >= max_len:
            fin = (d and d[-1] == EOS) or len(prefix) + len(d) >= toy.cap
            out.append((tuple(d), bool(fin), math.exp(lq)))
            return
        row = toy.q(prompt + prefix + tuple(d))
        for t in range(V):
            rec(d + [t], lq + float(row[t]))

    rec([], 0.0)
    return out


def _finished(seq, cap):
    return (seq and seq[-1] == EOS) or len(seq) >= cap


def _rollout_law(toy, prompt, cut_law, p_fn):
    """Exact law of the rollout the sticky driver produces."""
    out: dict = {}

    def add(seq, w):
        out[seq] = out.get(seq, 0.0) + w

    def verify(prefix, draft, fin, w, rec):
        rows = prompt + prefix
        lp = torch.stack([p_fn(rows + draft[:i]) for i in range(len(draft) + 1)])
        lq = torch.stack([toy.q(rows + draft[:i]) for i in range(len(draft))])
        for tau, py in output_distribution_given_draft(lp, lq, torch.tensor(draft)):
            if tau == len(draft):
                pt = float(py.sum())
                if pt == 0:
                    continue
                if fin:
                    add(prefix + draft, w * pt)
                else:  # no bonus: continue from the whole draft
                    rec(prefix + draft, w * pt)
                continue
            for y in range(V):
                if float(py[y]) == 0:
                    continue
                new = prefix + draft[:tau] + (y,)
                if _finished(new, toy.cap):
                    add(new, w * float(py[y]))
                else:
                    rec(new, w * float(py[y]))

    def continuation(prefix, w):
        for d, fin, qd in _drafts(toy, prefix, prompt, 10**9):
            verify(prefix, d, fin, w * qd, continuation)

    for length, pl in cut_law:  # exogenous deadline: draft length cap
        for d, fin, qd in _drafts(toy, (), prompt, length):
            verify((), d, fin, pl * qd, continuation)
    return out


def _target_law(toy, prompt, p_fn):
    out = {}

    def rec(seq, lp):
        if _finished(seq, toy.cap):
            out[seq] = math.exp(lp)
            return
        row = p_fn(prompt + seq)
        for t in range(V):
            rec(seq + (t,), lp + float(row[t]))

    rec((), 0.0)
    return out


@pytest.mark.parametrize("seed,delta,cap", [(0, 0.1, 3), (1, 0.05, 4), (2, 1.0, 3)])
def test_sticky_rollout_scheme_is_exact(seed, delta, cap):
    toy = _Toy(seed, delta, cap)
    prompt = (0, 1)
    cut_law = [(1, 0.3), (2, 0.3), (cap, 0.4)]
    law = _rollout_law(toy, prompt, cut_law, toy.p)
    target = _target_law(toy, prompt, toy.p)
    tv = 0.5 * sum(abs(law.get(s, 0) - target.get(s, 0)) for s in set(law) | set(target))
    assert tv < 1e-12
    # the sticky target really differs from plain theta_{n+1}
    plain = _target_law(toy, prompt, lambda s: toy._lp(1, s, toy.plain_routes(1, s)))
    assert 0.5 * sum(abs(plain[s] - target.get(s, 0)) for s in plain) > 1e-3


def test_anchor_schedule_depending_on_tau_is_biased():
    """Control: anchoring continuation positions to self after a rejection
    (anchors that depend on where verification stopped, violating R2) does
    not sample the sticky target, nor the plain one."""
    toy = _Toy(0, 0.1, 3)
    prompt = (0, 1)
    # Continuation decoded from plain theta_{n+1} after the first rejection.
    plain_p = lambda s: toy._lp(1, s, toy.plain_routes(1, s))  # noqa: E731
    out: dict = {}

    def add(seq, w):
        out[seq] = out.get(seq, 0.0) + w

    def tail(prefix, w):
        if _finished(prefix, toy.cap):
            add(prefix, w)
            return
        row = plain_p(prompt + prefix)
        for t in range(V):
            tail(prefix + (t,), w * math.exp(float(row[t])))

    for d, fin, qd in _drafts(toy, (), prompt, 2):
        lp = torch.stack([toy.p(prompt + d[:i]) for i in range(len(d) + 1)])
        lq = torch.stack([toy.q(prompt + d[:i]) for i in range(len(d))])
        for tau, py in output_distribution_given_draft(lp, lq, torch.tensor(d)):
            for y in range(V):
                new = d[:tau] + (y,) if tau < len(d) or not fin else d
                w = qd * float(py[y])
                if tau == len(d) and fin:
                    add(d, w)
                else:
                    tail(new, w)
    target = _target_law(toy, prompt, toy.p)
    tv = 0.5 * sum(abs(out.get(s, 0) - target.get(s, 0)) for s in set(out) | set(target))
    assert tv > 1e-3
