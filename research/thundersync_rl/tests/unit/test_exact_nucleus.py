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
"""Exact top-p policy (nemo_rl/models/megatron/exact_nucleus.py): the canonical
nucleus, its sharded threshold search, the sticky-nucleus rule, gradients, and
exactness of speculative rollouts under a truncated policy by exact enumeration
(plain truncation and sticky nucleus, block verification and ⊥-BV)."""

import itertools
import math

import pytest
import torch
from thundersync_rl.block_verification import output_distribution_given_draft

from nemo_rl.models.megatron import exact_nucleus as xn

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _off():
    xn.configure(None)
    yield
    xn.configure(None)


def _brute(lp: torch.Tensor, p: float) -> torch.Tensor:
    """Definition, one row at a time in Python integers."""
    out = torch.zeros(lp.shape, dtype=torch.bool)
    for r in range(lp.shape[0]):
        w = [int(math.floor(math.exp(float(x)) * 2**52)) if x != float("-inf") else 0 for x in lp[r].double()]
        W = sum(w)
        T = math.ceil(float(W) * p)
        order = sorted(range(len(w)), key=lambda v: (-w[v], v))
        B = 0
        for v in order:
            out[r, v] = B < T
            B += w[v]
    return out


def _rows(n, V, seed, ties=False, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, V, generator=g) * scale
    if ties:
        z = (z * 2).round() / 2  # many exactly tied logits
    return z


@pytest.mark.parametrize("p", [0.95, 0.5, 0.999, 0.3])
@pytest.mark.parametrize("ties", [False, True])
def test_plain_mask_matches_definition(p, ties):
    z = _rows(16, 64, 0, ties)
    lp = torch.log_softmax(z, -1)
    m = xn.plain_mask(xn.mass_weights(lp), p)
    assert torch.equal(m, _brute(lp, p))
    # the nucleus reaches P (up to the 2^-52 quantization) and is minimal
    pr = lp.double().exp()
    mass = (pr * m).sum(-1)
    assert bool((mass >= p - 1e-9).all())
    smallest = torch.where(m, pr, torch.full_like(pr, 2.0)).min(-1).values
    assert bool((mass - smallest < p + 1e-9).all())
    assert bool(m[torch.arange(16), z.argmax(-1)].all())  # the top token is always kept


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("p", [0.95, 0.6])
def test_sharded_bisection_equals_gathered(tp, ties, p):
    V = 96
    z = _rows(12, V, 1 + tp, ties)
    w = xn.mass_weights(torch.log_softmax(z, -1))
    ref = xn.plain_mask(w, p)
    shards = list(w.chunk(tp, -1))

    def all_reduce_sum(xs):
        return sum(xs)

    # simulate the group: run every rank's program in lockstep through generators
    outs = [None] * tp
    import threading

    barrier = threading.Barrier(tp)
    slots: dict = {}
    lock = threading.Lock()

    def make_ops(rank):
        step = [0]

        def gather(x):
            k = step[0]
            step[0] += 1
            with lock:
                slots.setdefault(k, [None] * tp)[rank] = x.clone()
            barrier.wait()
            res = list(slots[k])
            barrier.wait()
            return res

        return (lambda x: sum(gather(x))), gather

    def run(rank):
        ar, ag = make_ops(rank)
        outs[rank] = xn.sharded_mask(shards[rank], p, rank * (V // tp), ar, ag, rank)

    th = [threading.Thread(target=run, args=(r,)) for r in range(tp)]
    [t.start() for t in th]
    [t.join() for t in th]
    assert torch.equal(torch.cat(outs, -1), ref)


def test_mask_is_row_invariant_and_block_invariant():
    xn.configure(0.9)
    z = _rows(37, 200, 3)
    full = xn.mask_logits_(z.clone(), block_rows=256)
    for blk in (1, 5, 16):
        assert torch.equal(xn.mask_logits_(z.clone(), block_rows=blk), full)
    for r in (0, 11, 36):
        assert torch.equal(xn.mask_logits_(z[r : r + 1].clone()), full[r : r + 1])
    lp = xn.policy_log_softmax(z)
    assert torch.equal(lp, torch.log_softmax(full, -1))
    assert bool(torch.isinf(lp).any()) and bool(torch.isfinite(lp.max(-1).values).all())


def test_policy_log_softmax_is_plain_when_off():
    z = _rows(4, 50, 4)
    assert torch.equal(xn.policy_log_softmax(z), torch.log_softmax(z, -1))
    assert torch.equal(xn.mask_chunk(z.clone()[None], 0, 4)[0], z)


def test_gradient_is_masked_softmax_gradient():
    xn.configure(0.8)
    z = _rows(6, 30, 5).requires_grad_(True)
    x = torch.log_softmax(z.detach(), -1).argmax(-1)  # in the nucleus
    zm = z.clone()
    m = xn.nucleus_mask(z.detach())
    zm = zm.masked_fill(~m, float("-inf"))
    lp = torch.log_softmax(zm, -1).gather(-1, x[:, None])[:, 0]
    lp.sum().backward()
    soft = torch.softmax(z.detach().masked_fill(~m, float("-inf")), -1)
    expect = torch.nn.functional.one_hot(x, 30).float() - soft
    assert torch.allclose(z.grad, expect, atol=1e-6)
    assert bool((z.grad[~m] == 0).all())


def test_sticky_self_anchor_is_plain_and_feasible_anchor_kept():
    p, d = 0.9, 0.02
    z = _rows(64, 40, 6, scale=2.0)
    w = xn.mass_weights(torch.log_softmax(z, -1))
    plain = xn.plain_mask(w, p)
    C = 40
    ids = torch.full((64, C), -1, dtype=torch.int32)
    for r in range(64):
        a = plain[r].nonzero().flatten()
        ids[r, : a.numel()] = a.to(torch.int32)
    assert torch.equal(xn.sticky_mask(w, ids, p, d), plain)
    assert torch.equal(xn.sticky_mask(w, torch.full((64, C), -1, dtype=torch.int32), p, d), plain)
    # a drifted target: the drafter's nucleus is kept when the drift is < delta
    z2 = z + 0.01 * _rows(64, 40, 7)
    w2 = xn.mass_weights(torch.log_softmax(z2, -1))
    st = xn.sticky_mask(w2, ids, p, d)
    a_mask, has = xn.anchor_set_mask(ids, 40)
    keep = xn.sticky_keep(w2, a_mask, has, p, d)
    assert bool(keep.float().mean() > 0.9)
    assert torch.equal(st[keep], a_mask[keep]) and torch.equal(st[~keep], xn.plain_mask(w2, p)[~keep])


@cuda
def test_cuda_matches_cpu_mask():
    z = _rows(64, 4096, 8, ties=True)
    lp = torch.log_softmax(z, -1)
    m_cpu = xn.plain_mask(xn.mass_weights(lp), 0.95)
    m_gpu = xn.plain_mask(xn.mass_weights(lp.cuda()), 0.95)
    assert torch.equal(m_gpu.cpu(), m_cpu)


# --------------------------------------------------------------------------
# Exact enumeration: speculative rollouts under a truncated policy
# --------------------------------------------------------------------------
#
# Toy LM: tokens {0, 1, 2, EOS = 3}, rollout cap. Drafter theta_n, target
# theta_{n+1} = theta_n + a small perturbation (nucleus boundaries flip).
# q = drafter's own plain nucleus law (what the engine samples). Target:
#   plain : p = theta_{n+1}'s plain nucleus law
#   sticky: p = F(theta_{n+1}; A), A = theta_n's plain nucleus on the same prefix.
# Scheme (as the sticky-routing driver): drafts from q cut at an exogenous
# length; block verification against p; a whole-accepted unfinished draft
# emits no bonus and continues; every continuation is a theta_n draft to the
# end, verified again. Output law must equal p's rollout law.

VT, EOS = 4, 3
TOPP = 0.8


class _NToy:
    def __init__(self, seed, eps, delta, cap):
        self.seed = seed
        self.eps, self.delta, self.cap = eps, delta, cap
        self.z = {}

    def logits(self, ver, prefix):
        """Deterministic per (seed, prefix): independent of evaluation order."""
        k = (ver, tuple(prefix))
        if k not in self.z:
            code = 1
            for t in prefix:
                code = code * (VT + 1) + t + 1
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + code)
            base = torch.randn(VT, generator=g, dtype=torch.float64) * 1.5
            self.z[(0, tuple(prefix))] = base
            self.z[(1, tuple(prefix))] = base + self.eps * torch.randn(
                VT, generator=g, dtype=torch.float64
            )
        return self.z[k]

    def _law(self, z, anchors=None):
        w = xn.mass_weights(torch.log_softmax(z, -1))
        a = None if anchors is None else anchors[None]
        m = xn.sticky_mask(w[None], a, TOPP, self.delta)[0]
        return torch.log_softmax(z.masked_fill(~m, float("-inf")), -1)

    def nucleus_ids(self, ver, prefix):
        w = xn.mass_weights(torch.log_softmax(self.logits(ver, prefix), -1))
        m = xn.plain_mask(w[None], TOPP)[0]
        ids = torch.full((VT,), -1, dtype=torch.int32)
        nz = m.nonzero().flatten()
        ids[: nz.numel()] = nz.to(torch.int32)
        return ids

    def q(self, seq):
        return self._law(self.logits(0, seq))

    def p_plain(self, seq):
        return self._law(self.logits(1, seq))

    def p_sticky(self, seq):
        return self._law(self.logits(1, seq), self.nucleus_ids(0, seq))


def _fin(seq, cap):
    return (len(seq) > 0 and seq[-1] == EOS) or len(seq) >= cap


def _drafts(toy, prompt, prefix, max_len):
    out = []

    def rec(d, lq):
        if lq == float("-inf"):
            return
        if _fin(prefix + tuple(d), toy.cap) or len(d) >= max_len:
            out.append((tuple(d), _fin(prefix + tuple(d), toy.cap), math.exp(lq)))
            return
        row = toy.q(prompt + prefix + tuple(d))
        for t in range(VT):
            rec(d + [t], lq + float(row[t]))

    rec([], 0.0)
    return out


def _scheme_law(toy, prompt, cut_law, p_fn, null_m=1.0):
    out: dict = {}
    bot = [0.0]

    def add(seq, w):
        out[seq] = out.get(seq, 0.0) + w

    def verify(prefix, draft, fin, w, m):
        rows = prompt + prefix
        lp = torch.stack([p_fn(rows + draft[:i]) for i in range(len(draft) + 1)])
        lq = torch.stack([toy.q(rows + draft[:i]) for i in range(len(draft))])
        for tau, py in output_distribution_given_draft(lp, lq, torch.tensor(draft, dtype=torch.long), null_m=m):
            if m > 1.0 and tau == 0 and py.numel() == VT + 1:
                bot[0] += w * float(py[-1])
                py = py[:-1]
            if tau == len(draft):
                pt = float(py.sum())
                if pt == 0:
                    continue
                if fin:
                    add(prefix + draft, w * pt)
                else:
                    cont(prefix + draft, w * pt)
                continue
            for y in range(VT):
                if float(py[y]) == 0:
                    continue
                new = prefix + draft[:tau] + (y,)
                if _fin(new, toy.cap):
                    add(new, w * float(py[y]))
                else:
                    cont(new, w * float(py[y]))

    def cont(prefix, w):
        for d, fin, qd in _drafts(toy, prompt, prefix, 10**9):
            verify(prefix, d, fin, w * qd, 1.0)

    for length, pl in cut_law:
        for d, fin, qd in _drafts(toy, prompt, (), length):
            verify((), d, fin, pl * qd, null_m)
    return out, bot[0]


def _target(toy, prompt, p_fn):
    out = {}

    def rec(seq, lp):
        if lp == float("-inf"):
            return
        if _fin(seq, toy.cap):
            out[seq] = math.exp(lp)
            return
        row = p_fn(prompt + seq)
        for t in range(VT):
            rec(seq + (t,), lp + float(row[t]))

    rec((), 0.0)
    return out


def _tv(a, b):
    return 0.5 * sum(abs(a.get(s, 0) - b.get(s, 0)) for s in set(a) | set(b))


@pytest.mark.parametrize("seed,eps,cap", [(1, 0.2, 3), (4, 0.1, 4), (3, 0.3, 3), (7, 0.2, 4)])
@pytest.mark.parametrize("rule", ["plain", "sticky"])
def test_truncated_speculation_is_exact(seed, eps, cap, rule):
    toy = _NToy(seed, eps, 0.1, cap)
    prompt = (0, 1)
    p_fn = toy.p_plain if rule == "plain" else toy.p_sticky
    cut_law = [(1, 0.3), (2, 0.3), (cap, 0.4)]
    law, _ = _scheme_law(toy, prompt, cut_law, p_fn)
    target = _target(toy, prompt, p_fn)
    assert _tv(law, target) < 1e-12
    if rule == "sticky":  # non-vacuous: the sticky target is not the plain one
        assert _tv(target, _target(toy, prompt, toy.p_plain)) > 1e-3
    # non-vacuous: supports differ (the drafter samples tokens the target excludes)
    flips = sum(
        int(not torch.equal(torch.isinf(toy.q(prompt + s)), torch.isinf(toy.p_plain(prompt + s))))
        for s in itertools.chain.from_iterable(
            itertools.product(range(VT), repeat=n) for n in range(cap)
        )
    )
    assert flips > 0


@pytest.mark.parametrize("m", [1.25, 2.0])
def test_truncated_null_bv_output_is_exact_given_not_bot(m):
    toy = _NToy(3, 0.3, 0.1, 3)
    prompt = (1,)
    law, bot = _scheme_law(toy, prompt, [(3, 1.0)], toy.p_plain, null_m=m)
    assert abs(bot - (1 - 1 / m)) < 1e-12
    cond = {s: v * m for s, v in law.items()}
    assert _tv(cond, _target(toy, prompt, toy.p_plain)) < 1e-12


@pytest.mark.parametrize("seed,eps,cut", [(1, 0.2, 1), (3, 0.3, 1), (4, 0.2, 2)])
def test_sticky_nucleus_differs_from_plain_and_tau_dependent_anchor_is_biased(seed, eps, cut):
    toy = _NToy(seed, eps, 0.1, 4)
    prompt = (0, 1)
    sticky = _target(toy, prompt, toy.p_sticky)
    plain = _target(toy, prompt, toy.p_plain)
    assert _tv(sticky, plain) > 1e-3
    # Control: continuation after a rejection sampled from the plain target (anchor
    # = self, i.e. depending on where verification stopped): not the sticky law.
    out: dict = {}

    def tail(prefix, w):
        if _fin(prefix, toy.cap):
            out[prefix] = out.get(prefix, 0.0) + w
            return
        row = toy.p_plain(prompt + prefix)
        for t in range(VT):
            if float(row[t]) > float("-inf"):
                tail(prefix + (t,), w * math.exp(float(row[t])))

    for d, fin, qd in _drafts(toy, prompt, (), cut):
        lp = torch.stack([toy.p_sticky(prompt + d[:i]) for i in range(len(d) + 1)])
        lq = torch.stack([toy.q(prompt + d[:i]) for i in range(len(d))])
        for tau, py in output_distribution_given_draft(lp, lq, torch.tensor(d, dtype=torch.long)):
            if tau == len(d):
                wgt = qd * float(py.sum())
                if wgt == 0:
                    continue
                if fin:
                    out[d] = out.get(d, 0.0) + wgt
                else:
                    tail(d, wgt)
                continue
            for y in range(VT):
                wgt = qd * float(py[y])
                if wgt:
                    tail(d[:tau] + (y,), wgt)
    assert _tv(out, sticky) > 1e-3


def test_sticky_reduces_support_flips_on_toy():
    """Acceptance: whole-draft acceptance of sticky >= plain on the toy (small drift)."""
    toy = _NToy(5, 0.05, 0.1, 4)
    prompt = (2,)

    def whole(p_fn):
        tot = 0.0
        for d, fin, qd in _drafts(toy, prompt, (), 4):
            lp = torch.stack([p_fn(prompt + d[:i]) for i in range(len(d) + 1)])
            lq = torch.stack([toy.q(prompt + d[:i]) for i in range(len(d))])
            dist = output_distribution_given_draft(lp, lq, torch.tensor(d, dtype=torch.long))
            tot += qd * float(dist[-1][1].sum())
        return tot

    assert whole(toy.p_sticky) >= whole(toy.p_plain) - 1e-12


# --------------------------------------------------------------------------
# Engine rows == learner chunks (the bitwise contract), gradients, anchors
# --------------------------------------------------------------------------


def _sample_targets(z, seed):
    g = torch.Generator().manual_seed(seed)
    m = xn.mask_logits_(z.clone())
    return torch.multinomial(torch.softmax(m.reshape(-1, z.shape[-1]), -1), 1, generator=g).view(z.shape[:-1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_engine_sampler_rows_equal_learner_chunks(dtype):
    from nemo_rl.models.megatron.zero_train_gen_mismatch import _gathered_target_logprobs

    xn.configure(0.9)
    B, S, V = 2, 9, 80
    z = _rows(B * S, V, 11).view(B, S, V).to(dtype)
    target = _sample_targets(z.float(), 0)
    # engine: the controller's logits buffer, rows masked in place by the sampler
    buf = z.reshape(B * S, V).clone()
    perm = torch.randperm(B * S, generator=torch.Generator().manual_seed(1))
    for k in range(0, B * S, 5):  # several decode steps, rows in any order
        xn.mask_sampled_rows_(buf, perm[k : k + 5])
    eng = torch.log_softmax(buf.float(), -1).gather(-1, target.view(-1, 1)).view(B, S)
    for chunk in (1, 4, S):
        lrn = _gathered_target_logprobs(lambda a, e: z[:, a:e], target, chunk)
        assert torch.equal(lrn, eng)
    assert bool((eng != torch.log_softmax(z.float(), -1).gather(-1, target[..., None])[..., 0]).any())


def test_learner_grad_helper_is_masked_softmax_grad():
    from nemo_rl.models.megatron.zero_train_gen_mismatch import _gathered_logprob_grad

    xn.configure(0.85)
    B, S, V = 2, 6, 50
    z = _rows(B * S, V, 12).view(B, S, V)
    target = _sample_targets(z, 2)
    go = torch.randn(B, S, generator=torch.Generator().manual_seed(3))
    got = _gathered_logprob_grad(
        lambda a, e: z[:, a:e], target, go, vocab_start_index=0, vocab_end_index=V,
        chunk_size=4, grad_input=torch.zeros_like(z),
    )
    zr = z.clone().requires_grad_(True)
    m = xn.nucleus_mask(z)
    lp = torch.log_softmax(zr.masked_fill(~m, float("-inf")), -1).gather(-1, target[..., None])[..., 0]
    (lp * go).sum().backward()
    assert torch.allclose(got, zr.grad, atol=1e-6) and bool((got[~m] == 0).all())


def test_learner_chunks_follow_anchor_context_with_offsets():
    from nemo_rl.models.megatron.zero_train_gen_mismatch import _gathered_target_logprobs

    xn.configure(0.8, sticky_delta=0.05)
    B, S, V, C = 1, 8, 30, 8
    z = _rows(S, V, 13, scale=1.0).view(B, S, V)
    zq = z + 0.05 * _rows(S, V, 14).view(B, S, V)  # drafter
    wq = xn.mass_weights(torch.log_softmax(zq, -1))
    anchors = torch.full((B, S, C), -1, dtype=torch.int32)
    for s_ in range(S):
        ids = xn.plain_mask(wq[0, s_][None], 0.8)[0].nonzero().flatten()[:C]
        anchors[0, s_, : ids.numel()] = ids.to(torch.int32)
    target = _sample_targets(z, 4)
    ref = torch.log_softmax(
        z.masked_fill(~xn.sticky_mask(xn.mass_weights(torch.log_softmax(z, -1)), anchors, 0.8, 0.05), float("-inf")), -1
    ).gather(-1, target[..., None])[..., 0]
    ref = ref.masked_fill(torch.isneginf(ref), 0.0)
    with xn.anchors_context(anchors):
        full = _gathered_target_logprobs(lambda a, e: z[:, a:e], target, 3)
        # a sub-range forward (deferred head backward chunks) with its absolute offset
        part = _gathered_target_logprobs(lambda a, e: z[:, 4 + a : 4 + e], target[:, 4:], 2, seq_offset=4)
    assert torch.equal(full, ref) and torch.equal(part, ref[:, 4:])


def test_engine_q_rows_with_anchors():
    from thundersync_rl.engine_q import GatheredLogprobRows, gathered_log_softmax, no_gather

    xn.configure(0.8, sticky_delta=0.05)
    n, V, C = 7, 40, 6
    z = _rows(n, V, 15, scale=1.0)
    anchors = torch.full((n, C), -1, dtype=torch.int32)
    anchors[:, 0] = z.argmax(-1).to(torch.int32)
    anchors[:, 1] = (z.argmax(-1) + 1).remainder(V).to(torch.int32)
    rows = GatheredLogprobRows(z, no_gather, 1 << 30, chunk_rows=3)
    rows.anchors = anchors
    full = gathered_log_softmax(z, no_gather, 1 << 30, anchors)
    idx = torch.tensor([5, 1, 3])
    assert torch.equal(rows(idx), full[idx])
    tok = full.argmax(-1)
    assert torch.equal(rows.token_logprobs(tok), full.gather(-1, tok[:, None])[:, 0])
    assert torch.equal(gathered_log_softmax(z, no_gather, 1 << 30), xn.policy_log_softmax(z))


def test_engine_anchor_ids_are_the_nucleus():
    xn.configure(0.9)
    z = _rows(10, 30, 16, scale=0.7)
    buf = z.clone()
    ids = xn.mask_sampled_rows_(buf, torch.arange(10), cap=8)
    m = xn.nucleus_mask(z)
    for r in range(10):
        size = int(m[r].sum())
        if size > 8:
            assert bool((ids[r] == -1).all())
        else:
            got = sorted(int(x) for x in ids[r] if x >= 0)
            assert got == m[r].nonzero().flatten().tolist()


def test_nucleus_route_blocks_roundtrip_and_splice():
    from thundersync_rl.routes import (
        NUCLEUS_LAYER,
        decode_nucleus,
        encode_nucleus,
        fetch_nucleus,
        merge_route_refs,
        put_routes,
        splice_refs,
    )

    C = 4
    ids = torch.tensor([[5, 131071, -1, -1], [-1, -1, -1, -1], [0, 1, 2, 3]], dtype=torch.int32)
    enc = encode_nucleus(ids)
    assert enc.dtype == torch.int16 and bool((enc >= 0).all())
    assert torch.equal(decode_nucleus(enc), ids)
    store = {}

    def put(t):
        store[len(store)] = t.clone()
        return len(store) - 1

    route = torch.zeros(6, 1, 2, dtype=torch.int16)  # MoE layer 1, positions 0..5
    # first round: prompt 3 tokens -> nucleus rows at input positions 2, 3, 4
    parts = [
        ((1,), 0, 6, put_routes(route, put)),
        ((NUCLEUS_LAYER,), 2, 5, put_routes(enc, put, start=2)),
    ]
    first = merge_route_refs(parts, 6, [1])
    assert first.complete == 6 and len(first.nucleus) == 1
    got = fetch_nucleus(first, 6, C, store.__getitem__)
    assert bool((got[:2] == -1).all()) and torch.equal(got[2:5], ids) and bool((got[5] == -1).all())
    # a continuation round re-logs routes from 0 and nucleus ids only from its
    # last prompt position (5): the prior round's nucleus rows must survive.
    prior = first.truncate(5)
    enc2 = encode_nucleus(torch.tensor([[7, 8, -1, -1]], dtype=torch.int32))
    new = merge_route_refs(
        [((1,), 0, 7, put_routes(torch.zeros(7, 1, 2, dtype=torch.int16), put)),
         ((NUCLEUS_LAYER,), 5, 6, put_routes(enc2, put, start=5))],
        7,
        [1],
    )
    s = splice_refs(prior, new)
    got = fetch_nucleus(s, 7, C, store.__getitem__)
    assert torch.equal(got[2:5], ids) and got[5].tolist() == [7, 8, -1, -1]
    # windowed fetch (verification: rows from prompt_len - 1)
    assert torch.equal(fetch_nucleus(s, 3, C, store.__getitem__, lo=3), got[3:6])


def test_restarted_stream_keeps_prefix_nucleus_rows():
    """Streamed continuation (ns2-topp FAIL): a stream restarted after a rejection
    builds its RouteRefs from its own logs and the driver replaces Plan.anchors with
    them; seeded with ``nucleus_base(plan.anchors)`` the kept prefix's nucleus rows
    survive (without the seed they fall back to self = plain nucleus at the learner)."""
    from thundersync_rl.routes import (
        NUCLEUS_LAYER,
        encode_nucleus,
        fetch_nucleus,
        merge_route_refs,
        nucleus_base,
        put_routes,
    )

    C = 4
    store = {}

    def put(t):
        store[len(store)] = t.clone()
        return len(store) - 1

    ids = torch.tensor([[5, 6, -1, -1], [1, -1, -1, -1], [0, 1, 2, 3], [9, -1, -1, -1]], dtype=torch.int32)
    # first stream: prompt 3, drafted 4 tokens -> nucleus rows at positions 2..5; reject at
    # draft index 2 (n_acc = 2): Plan.anchors = truncate(3 + 2) keeps rows 2, 3, 4
    first = merge_route_refs(
        [((1,), 0, 7, put_routes(torch.zeros(7, 1, 2, dtype=torch.int16), put)),
         ((NUCLEUS_LAYER,), 2, 6, put_routes(encode_nucleus(ids), put, start=2))],
        7, [1],
    )
    plan_anchors = first.truncate(5)
    assert nucleus_base(None) is None
    base = nucleus_base(plan_anchors)
    assert base.blocks == [] and base.complete == 0 and base.prefix == {}
    # restarted stream: prompt = 3 + 2 kept + residual = 6; re-logs routes from 0,
    # nucleus from its last prompt position 5
    enc2 = encode_nucleus(torch.tensor([[7, 8, -1, -1], [4, -1, -1, -1]], dtype=torch.int32))
    parts = [((1,), 0, 8, put_routes(torch.zeros(8, 1, 2, dtype=torch.int16), put)),
             ((NUCLEUS_LAYER,), 5, 7, put_routes(enc2, put, start=5))]
    for b, want_prefix in ((None, False), (base, True)):
        new = merge_route_refs(parts, 8, [1], base=b)
        assert new.complete == 8
        anchors = new.truncate(8)  # accepted piece: Plan.anchors = d[8].truncate(...)
        got = fetch_nucleus(anchors, 8, C, store.__getitem__)
        assert got[5].tolist() == [7, 8, -1, -1] and got[6].tolist() == [4, -1, -1, -1]
        if want_prefix:
            assert torch.equal(got[2:5], ids[:3])
        else:  # the bug: prefix rows lost (self)
            assert bool((got[2:5] == -1).all())


def test_config_validation_opt_in_only():
    from types import SimpleNamespace

    from thundersync_rl.grpo_loop import (
        ThunderSyncConfig,
        configure_sticky_policy,
        validate_exact_top_p,
        validate_speculation,
    )

    def master(**ts):
        policy = {
            "generation": {
                "temperature": 1.0, "top_p": 0.95, "top_k": None, "backend": "megatron",
                "stop_strings": None, "mcore_generation_config": {},
            },
            "megatron_cfg": {
                "batch_invariant_mode": True, "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
            },
        }
        kw = dict(
            speculate=True, verify_mode="block", verify_source="engine", verify_on="learner",
            sticky_routing=True,
        )
        kw.update(ts)
        cfg = ThunderSyncConfig(**kw)
        return SimpleNamespace(
            thundersync=cfg, policy=policy, loss_fn=SimpleNamespace(reference_policy_kl_penalty=0.0)
        )

    # Default: top_p < 1 is still refused (the refusal is lifted only by the opt-in).
    m = master()
    with pytest.raises(ValueError, match="top_p must be 1.0"):
        validate_speculation(m.thundersync, m.policy)
    validate_exact_top_p(m)  # off: nothing to check
    m = master(exact_top_p=True, sticky_nucleus=True)
    validate_exact_top_p(m)
    configure_sticky_policy(m)
    assert m.policy["megatron_cfg"]["exact_top_p"] == 0.95
    assert m.policy["megatron_cfg"]["sticky_nucleus_delta"] == 0.01
    with pytest.raises(ValueError, match="sticky_nucleus requires sticky_routing"):
        validate_exact_top_p(master(exact_top_p=True, sticky_nucleus=True, sticky_routing=False))
    with pytest.raises(ValueError, match="requires thundersync.exact_top_p"):
        validate_exact_top_p(master(sticky_nucleus=True))
    bad = master(exact_top_p=True)
    bad.policy["generation"]["top_k"] = 20
    with pytest.raises(ValueError, match="top_k must be unset"):
        validate_exact_top_p(bad)
    xn.configure_from_policy(m.policy)
    assert xn.active() and xn.sticky() and xn.top_p() == 0.95


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda)])
@pytest.mark.parametrize("noise", [0.05, 0.5])
@pytest.mark.parametrize("top_p", [0.9, 0.6])
def test_engine_rule_with_truncated_rows_matches_eager_rule(noise, top_p, device):
    """The lazy engine-q rule (``verify_engine_q``: lazy scan, Triton/torch residual
    masses, emission) on nucleus-masked rows (-inf outside N, tokens with p = 0)
    takes the eager rule's decisions and generator calls."""
    from thundersync_rl.block_verification_fused import block_verify_logits
    from thundersync_rl.engine_q import verify_engine_q

    xn.configure(top_p)
    torch.manual_seed(1)
    vocab, rejected, pzero = 40, 0, 0
    for trial in range(60):
        g = 1 + trial % 9
        p = (torch.randn(g + 1, vocab) * 2).to(device)
        q = p[:g] + (torch.randn(g, vocab) * noise).to(device)
        lp, lq = xn.policy_log_softmax(p), xn.policy_log_softmax(q)
        draft = torch.multinomial(lq.exp().cpu(), 1).flatten().to(device)
        engine_lq = lq[torch.arange(g, device=device), draft]
        g1 = torch.Generator(device=device).manual_seed(trial)
        g2 = torch.Generator(device=device).manual_seed(trial)
        tau, y, logp, _ = verify_engine_q(lp, draft, engine_lq, lambda idx: lq[idx], g1)
        ref = block_verify_logits(lp, lq, draft, g2)
        assert (tau, y) == ref[:2]
        assert torch.equal(g1.get_state(), g2.get_state())
        assert bool(torch.isfinite(lp[tau, y]))  # the emitted token is in the target nucleus
        rejected += tau < g
        pzero += int(torch.isinf(lp[torch.arange(g, device=device), draft]).sum())
    assert rejected > 0 and pzero > 0
