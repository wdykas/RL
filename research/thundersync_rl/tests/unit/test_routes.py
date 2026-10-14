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

import math

import numpy as np
import pytest
import torch
from thundersync_rl.block_verification import output_distribution_given_draft
from thundersync_rl.routes import (
    RouteLog,
    SlotTags,
    anchored_draft_length,
    compare_anchors,
    fetch_anchors,
    merge_route_refs,
    put_routes,
    splice_refs,
    step_route_selection,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _sel(rows, keys, pos, log):
    """record() arguments for (row, key, position) entries."""
    return (
        np.array(rows, dtype=np.int64),
        np.array([log.key_id(k) for k in keys], dtype=np.int64),
        np.array(pos, dtype=np.int64),
    )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda)])
def test_route_log_keys_by_absolute_position(device):
    log = RouteLog(ring_rows=4, drain_every=2)
    L, K = 3, 2
    # step 1: prefill of draft a (positions 0..2) and an untagged request
    rows = torch.arange(5 * L * K, device=device).view(5, L, K)
    log.record(rows, *_sel([0, 1, 2], "aaa", [0, 1, 2], log))
    # step 2: decode of a (position 3) at row 1; b at row 0 (ring wraps)
    rows2 = 100 + torch.arange(2 * L * K, device=device).view(2, L, K)
    log.record(rows2, *_sel([1, 0], "ab", [3, 7], log))
    # an eviction recomputes position 1: overwrite
    rows3 = 200 + torch.arange(L * K, device=device).view(1, L, K)
    log.record(rows3, *_sel([0], "a", [1], log))
    lo, complete, r = log.pop(["a", "c"])["a"]
    assert (lo, complete) == (0, 4) and r.shape == (4, L, K) and r.dtype == torch.int16
    assert torch.equal(r[0], rows[0].cpu().to(r.dtype))
    assert r[1].tolist() in (rows[1].cpu().tolist(), rows3[0].cpu().tolist())
    assert torch.equal(r[3], rows2[1].cpu().to(r.dtype))
    log.record(rows, *_sel([0], "a", [9], log))  # closed: dropped
    assert log.pop(["a"]) == {}
    lo, complete, r = log.pop(["b"])["b"]
    assert (lo, complete) == (7, 8) and r.shape[0] == 1
    assert torch.equal(r[0], rows2[0].cpu().to(r.dtype))
    assert log.stored_rows() == 0  # compacted once every key closed


def test_route_log_many_keys_one_pop():
    """Decode-like forwards (one token for each of many drafts), popped at once."""
    log = RouteLog(ring_rows=1 << 12, drain_every=4)
    n, L, K, steps = 512, 4, 2, 6
    keys = [f"k{j}" for j in range(n)]
    for s in range(steps):
        rows = (torch.arange(n * L * K, dtype=torch.int32).view(n, L, K) + s) % 128
        log.record(rows, *_sel(range(n), keys, [s] * n, log))
    out = log.pop(keys[::2])
    assert sorted(out) == sorted(keys[::2])
    lo, c, r = out["k6"]
    assert (lo, c) == (0, steps) and r.shape == (steps, L, K)
    assert r[:, 0, 0].tolist() == [(6 * L * K + s) % 128 for s in range(steps)]
    assert len(log.pop(keys[1::2])) == n // 2 and log.stored_rows() == 0


def test_route_log_takes_a_running_draft_in_pieces():
    """close=False: rows logged since the last take, key left open; the
    pieces extend one RouteRefs (merge_route_refs with a base)."""
    log = RouteLog(ring_rows=64)
    L, K = 2, 1
    rows = torch.arange(10 * L * K, dtype=torch.int32).view(10, L, K)
    log.record(rows, *_sel(range(4), "aaaa", range(4), log))
    lo, end, r = log.pop(["a"], close=False)["a"]
    assert (lo, end) == (0, 4) and torch.equal(r, rows[:4].to(torch.int16))
    log.record(rows, *_sel([4, 5, 7], "aaa", [4, 5, 7], log))  # position 6 not logged
    lo, end, r = log.pop(["a"], close=False)["a"]
    assert (lo, end) == (4, 6) and r.shape[0] == 4 and bool((r[2] == -1).all())
    first = merge_route_refs(
        [([1, 2], 0, 4, put_routes(rows[:4].to(torch.int16), lambda t: t))], 20, [1, 2]
    )
    both = merge_route_refs(
        [([1, 2], lo, end, put_routes(r, lambda t: t, start=lo))],
        20,
        [1, 2],
        base=first,
    )
    assert both.complete == 6 and [(b.lo, b.hi) for b in both.blocks] == [
        (0, 4),
        (4, 8),
    ]
    out = fetch_anchors(both, [1, 2], K, 8, lambda t: t)
    assert torch.equal(out[:6], rows[:6].to(torch.int16)) and bool((out[6] == -1).all())
    log.pop(["a"])  # close
    log.record(rows, *_sel([6], "a", [6], log))
    assert log.pop(["a"], close=False) == {} and log.stored_rows() == 0


def test_slot_tags_look_up_only_new_requests():
    looked = []

    def lookup(rid):
        looked.append(rid)
        return {11: "a", 13: "b"}.get(rid)

    ids = {}
    tags = SlotTags(lookup, lambda k: ids.setdefault(k, len(ids)))
    kid = tags.refresh(np.array([11, 12, 13]), np.array([True, True, True]))
    assert kid.tolist() == [0, -1, 1] and looked == [11, 12, 13]
    # decode steps: nothing new, nothing looked up
    tags.refresh(np.array([11, 12, 13]), np.zeros(3, bool))
    assert looked == [11, 12, 13]
    # slot 1 gets a new request (fresh), slot 2 compacted onto slot 0
    kid = tags.refresh(np.array([13, 11]), np.array([False, True]))
    assert kid.tolist() == [1, 0] and looked[3:] == [13, 11]
    # a reused request id in prefill is looked up again
    kid = tags.refresh(np.array([13, 11]), np.array([False, True]))
    assert looked[-1] == 11


def test_step_selection():
    slots = np.array([4, 4, 4, 9, 2, 2])
    pos = np.array([0, 1, 2, 30, 5, 6])
    slot_kid = np.full(10, -1)
    slot_kid[4], slot_kid[2] = 0, 1
    sel, kid, p = step_route_selection(slots, pos, slot_kid)
    assert sel.tolist() == [0, 1, 2, 4, 5]
    assert kid.tolist() == [0, 0, 0, 1, 1] and p.tolist() == [0, 1, 2, 5, 6]


def _parts(dense_by_stage, block):
    """pop_draft_routes parts: (layers, logged prefix, blocks) per stage."""
    out = []
    for layers, dense, complete in dense_by_stage:
        out.append((layers, 0, complete, put_routes(dense, lambda t: t, block)))
    return out


def test_route_refs_from_pipeline_stages_fetch_per_stage():
    K = 2
    a = torch.arange(10 * 1 * K, dtype=torch.int16).view(10, 1, K)  # layer 1
    b = 100 + torch.arange(10 * 2 * K, dtype=torch.int16).view(10, 2, K)  # 3, 4
    b[8:] = -1  # stage 2 logged positions 0..7 only
    refs = merge_route_refs(
        _parts([([1], a, 10), ([3, 4], b, 8)], block=4), 9, [1, 3, 4]
    )
    assert refs.length == 9 and refs.complete == 8
    assert [(x.layers, x.lo, x.hi) for x in refs.blocks] == [
        ((1,), 0, 4),
        ((1,), 4, 8),
        ((1,), 8, 9),
        ((3, 4), 0, 4),
        ((3, 4), 4, 8),
        ((3, 4), 8, 9),
    ]
    fetched = []

    def get(t):
        fetched.append(t)
        return t

    # stage 2 of the learner: layers 3, 4 only; only its blocks are fetched
    out = fetch_anchors(refs, [3, 4], K, 12, get)
    assert out.shape == (12, 2, K)
    assert torch.equal(out[:8], b[:8]) and bool((out[8:] == -1).all())
    assert len(fetched) == 3 and all(t.shape[1] == 2 for t in fetched)
    # all layers (driver check) and a narrower width
    full = fetch_anchors(refs, [1, 3, 4], None, 6, lambda t: t)
    assert torch.equal(full[:, 0], a[:6, 0]) and torch.equal(full[:, 1:], b[:6])
    # a missing stage: nothing is complete
    assert merge_route_refs(_parts([([1], a, 10)], 4), 9, [1, 3, 4]).complete == 0


def test_truncate_and_splice_refs():
    K = 1
    old = torch.full((7, 1, K), 1, dtype=torch.int16)  # positions 0..6 (6: rejected)
    prior = merge_route_refs(_parts([([1], old, 7)], 4), 7, [1]).truncate(5)
    assert prior.length == 5 and prior.complete == 5
    assert [(x.lo, x.hi) for x in prior.blocks] == [(0, 4), (4, 5)]
    # the new round re-logs everything: prior's blocks are not kept
    new = torch.full((9, 1, K), 2, dtype=torch.int16)
    full_new = merge_route_refs(_parts([([1], new, 9)], 4), 9, [1])
    assert splice_refs(prior, full_new) is full_new
    # a gap in the new log at 0..2: prior fills it, rejected position 6 never leaks
    gap = new.clone()
    gap[:3] = -1
    s = splice_refs(prior, merge_route_refs(_parts([([1], gap, 0)], 4), 9, [1]))
    assert s.length == 9 and s.complete == 5
    out = fetch_anchors(s, [1], K, 9, lambda t: t)
    assert out[:, 0, 0].tolist() == [1, 1, 1, 2, 2, 2, 2, 2, 2]
    assert anchored_draft_length(s.complete, 3, 4) == 3


def test_anchored_length():
    # positions 0..4 logged; prompt 3, draft 4: m <= 5 - 3 + 1 = 3
    assert anchored_draft_length(5, 3, 4) == 3
    assert anchored_draft_length(5, 3, 2) == 2
    assert anchored_draft_length(5, 6, 4) == 0  # prompt not fully logged
    assert anchored_draft_length(2, 2, 4) == 1


def test_compare_anchors_against_a_reference_forward():
    A = torch.full((6, 2, 2), -1, dtype=torch.int16)
    A[:4] = torch.tensor([[1, 3], [0, 2]], dtype=torch.int16)  # positions 0..3 anchored
    ref = A[:5].clone().to(torch.int32)
    ref[1] = ref[1].flip(-1)  # same expert sets, other top-k order
    assert compare_anchors(A, ref) == (4, 4)
    ref[2, 1] = torch.tensor([0, 3])  # one layer routed elsewhere
    assert compare_anchors(A, ref) == (4, 3)
    assert compare_anchors(A, ref[:3]) == (4, 2)  # positions missing from ref
    assert compare_anchors(A, A[:0]) == (4, 0)
    assert compare_anchors(A[4:], ref) == (0, 0)  # nothing anchored


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
        return tuple(
            int(self._score(ver, tuple(seq[: j + 1])).argmax()) for j in range(len(seq))
        )

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
    tv = 0.5 * sum(
        abs(law.get(s, 0) - target.get(s, 0)) for s in set(law) | set(target)
    )
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
    tv = 0.5 * sum(
        abs(out.get(s, 0) - target.get(s, 0)) for s in set(out) | set(target)
    )
    assert tv > 1e-3
