"""Carried-walk continuation verification (DESIGN 14r): exactness by enumeration, identity, MC.

Toy autoregressive models over V tokens (token 0 = EOS), cap N tokens after the prompt,
p and q random functions of the prefix. A continuation process decodes q, verifies in
pieces of c tokens and decides by a rule; we enumerate the law of the complete rows it
outputs and compare it with p's.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest
import torch

from thundersync_rl.block_verification import stop_probability
from thundersync_rl.carried_walk import (
    DecisionRule,
    advance,
    block_decision_reference,
    carried_piece,
    carried_scan,
)


class Toy:
    def __init__(self, V=3, N=4, seed=0, temp=1.0, close=0.0):
        g = torch.Generator().manual_seed(seed)
        self.V, self.N = V, N
        self.p_tab, self.q_tab = {}, {}
        for n in range(N):
            for pre in itertools.product(range(1, V), repeat=n):
                a = torch.randn(V, generator=g, dtype=torch.float64) * temp
                b = torch.randn(V, generator=g, dtype=torch.float64) * temp
                b = close * a + (1 - close) * b
                self.p_tab[pre] = torch.log_softmax(a, 0)
                self.q_tab[pre] = torch.log_softmax(b, 0)

    def _uniform(self):
        return torch.full((self.V,), -math.log(self.V), dtype=torch.float64)

    def lp(self, pre):
        # rows after a complete row (the unused bonus row) are uniform
        return self.p_tab.get(tuple(pre), self._uniform())

    def lq(self, pre):
        return self.q_tab.get(tuple(pre), self._uniform())

    def complete(self, seq):
        return (len(seq) > 0 and seq[-1] == 0) or len(seq) >= self.N

    def p_law(self):
        out = {}

        def rec(pre, pr):
            if self.complete(pre):
                out[tuple(pre)] = out.get(tuple(pre), 0.0) + pr
                return
            pv = self.lp(pre).double().exp()
            for v in range(self.V):
                rec(pre + [v], pr * float(pv[v]))

        rec([], 1.0)
        return out


def _h(toy, pre, lb):
    """Non-terminal stop probability at the position after ``pre`` with weight exp(lb)."""
    b = math.exp(lb)
    p = toy.lp(pre).double().exp()
    q = toy.lq(pre).double().exp()
    r = float((b * p - q).clamp_min(0).sum())
    return float(stop_probability(torch.tensor([r], dtype=torch.float64), torch.tensor([b], dtype=torch.float64))[0])


def _residual(toy, pre, lb):
    b = math.exp(lb)
    p = toy.lp(pre).double().exp()
    q = toy.lq(pre).double().exp()
    d = (b * p - q).clamp_min(0)
    return (d if float(d.sum()) > 0 else p) / float((d if float(d.sum()) > 0 else p).sum())


def enumerate_law(toy, c, rule):
    """Exact law of complete rows of the carried-walk process with decision ``rule``.

    rule(block_tokens, log_b, last_hit, piece_end, final) -> probability of deciding
    here (in [0, 1]; a randomized rule mixes over independent randomness). The
    process: from a prefix, a block starts (b = 1); at each position i >= 1 that is a
    piece end (every c tokens) or final, the rule may decide: terminal test b_i, else
    tau = last hit (hits at non-decided positions with prob h_i; the root is a hit).
    Returns (law, expected rejections per row).
    """
    law, rej = {}, [0.0]

    def add(seq, pr):
        law[tuple(seq)] = law.get(tuple(seq), 0.0) + pr

    def block(pre, pr):
        def step(x, lb, hit, pr):
            row = pre + x
            i = len(x)
            final = toy.complete(row) and i > 0
            piece_end = i > 0 and i % c == 0
            pd = 0.0
            if final:
                pd = 1.0
            elif piece_end:
                pd = rule(x, lb, hit, True, False)
            retry = getattr(rule, "retry", 0.0) if pd > 0 and not final else 0.0
            if pd > 0:
                b = math.exp(lb)
                w = pr * pd
                # whole
                if b > 0:
                    if final:
                        add(row, w * b)
                    else:
                        block(row, w * b)
                # control only: a failed terminal draw is ignored with prob ``retry``
                # (the walk goes on as if no decision was due)
                cont = w * (1 - b) * retry
                w_rej = w * (1 - b) * (1 - retry)
                if cont > 0:
                    _continue(x, lb, hit, cont)
                # reject at the last hit
                if b < 1 and w_rej > 0:
                    w = w_rej / (1 - b)
                    rej[0] += w * (1 - b)
                    t = hit
                    head = pre + x[:t]
                    res = _residual(toy, head, lbs[t])
                    for y in range(toy.V):
                        if float(res[y]) == 0:
                            continue
                        nxt = head + [y]
                        if toy.complete(nxt):
                            add(nxt, w * (1 - b) * float(res[y]))
                        else:
                            block(nxt, w * (1 - b) * float(res[y]))
            if pd < 1:
                _continue(x, lb, hit, pr * (1 - pd))

        def _continue(x, lb, hit, w):
                row = pre + x
                i = len(x)
                h = _h(toy, row, lb)
                qv = toy.lq(row).double().exp()
                lpv = toy.lp(row)
                for hit_here, ph in ((True, h), (False, 1 - h)):
                    if ph <= 0:
                        continue
                    for v in range(toy.V):
                        lr = float(lpv[v]) - float(toy.lq(row)[v])
                        nlb = min(0.0, lb + lr)
                        lbs.append(nlb)
                        step(x + [v], nlb, i if hit_here else hit, w * ph * float(qv[v]))
                        lbs.pop()

        lbs = [0.0]
        step([], 0.0, 0, pr)

    block([], 1.0)
    return law, rej[0]


def tv(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


EVERY = lambda x, lb, hit, pe, fin: 1.0
ROW = lambda x, lb, hit, pe, fin: 0.0


def DIP(d):
    return lambda x, lb, hit, pe, fin: 1.0 if -lb >= d else 0.0


def TOKEN_RANDOM(x, lb, hit, pe, fin):
    # randomized, depends on the tokens (independent randomness): exact by mixture
    return 0.3 + 0.4 * (x[-1] % 2)


def HIT_LAG(x, lb, hit, pe, fin):
    # decides when the forward-sampled hits left the last hit behind (exact:
    # the forward process is a martingale in law, DESIGN 14r)
    return 1.0 if hit < len(x) - 1 else 0.0


def HIT_ROOT(x, lb, hit, pe, fin):
    return 0.7 if hit == 0 else 0.1


class RETRY:
    """Control: decides at every piece end but ignores a failed terminal draw half
    the time (a rule that looks at the decision's own draw): biased."""

    retry = 0.5

    def __call__(self, x, lb, hit, pe, fin):
        return 1.0


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("c", [1, 2])
@pytest.mark.parametrize(
    "name,rule",
    [("every", EVERY), ("row", ROW), ("dip0.05", DIP(0.05)), ("dip0.3", DIP(0.3)), ("random", TOKEN_RANDOM),
     ("hit_lag", HIT_LAG), ("hit_root", HIT_ROOT)],
)
def test_law_is_p(seed, c, name, rule):
    toy = Toy(V=3, N=4, seed=seed)
    law, _ = enumerate_law(toy, c, rule)
    assert tv(law, toy.p_law()) < 1e-12, name


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_terminal_retry_is_biased(seed):
    toy = Toy(V=3, N=4, seed=seed)
    law, _ = enumerate_law(toy, 1, RETRY())
    assert tv(law, toy.p_law()) > 1e-3


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_one_block_rejects_least(seed):
    # Exact expected rejections per row: carrying the walk to the end of the row
    # never rejects more than deciding every piece (here strictly fewer).
    toy = Toy(V=3, N=5, seed=seed, temp=0.7, close=0.6)
    _, r_every = enumerate_law(toy, 1, EVERY)
    _, r_row = enumerate_law(toy, 1, ROW)
    _, r_dip = enumerate_law(toy, 1, DIP(0.2))
    assert r_row < r_every
    assert r_row <= r_dip + 1e-12 <= r_every + 2e-12


def test_advance_is_piecewise_consistent():
    rng = random.Random(0)
    lr = [rng.gauss(-1e-4, 3e-3) for _ in range(1000)]
    whole, f_whole = advance(lr, 0.0, 0, 0)
    lb, f, pos = 0.0, 0, 0
    got = [0.0]
    for a in range(0, 1000, 137):
        part, f = advance(lr[a : a + 137], lb, f, pos)
        got += part[1:]
        lb, pos = part[-1], pos + len(part) - 1
    assert got == whole and f == f_whole
    assert all(x <= 0.0 for x in whole)
    assert whole[f_whole] == 0.0 and all(x < 0 for x in whole[f_whole + 1 :])


def _rows(toy, prompt_pre, block):
    """Raw-logit rows (log-probs) for positions 0..len(block) of a block and lse 0."""
    P = torch.stack([toy.lp(prompt_pre + block[:i]) for i in range(len(block) + 1)])
    Q = torch.stack([toy.lq(prompt_pre + block[:i]) for i in range(len(block))])
    return P, Q


def _run_pieces(toy, block, cuts, rule, final, gen_seed):
    """Drive carried_piece over ``block`` cut at ``cuts``; scan if needed.

    Returns (decision dict, eta_T, eta of the decision piece, eta of the scan) by
    replaying the generators' draws.
    """
    walk = {"log_b": 0.0, "floor": 0, "n": 0}
    P, Q = _rows(toy, [], block)
    ar = torch.arange(len(block))
    lpx_all = P[ar, torch.tensor(block)]
    lqx_all = Q[ar, torch.tensor(block)]
    lr_all = (lpx_all - lqx_all).tolist()
    bounds = [0] + cuts + [len(block)]
    for k, (a, e) in enumerate(zip(bounds, bounds[1:])):
        last = k == len(bounds) - 2
        gen = torch.Generator().manual_seed(gen_seed + k)
        out = carried_piece(
            lpx_all[a:e],
            lqx_all[a:e],
            torch.zeros(e - a + 1),
            torch.zeros(e - a),
            lambda idx, a=a: P[a + idx],
            lambda idx, a=a: Q[a + idx],
            gen,
            walk,
            rule,
            final and last,
        )
        if out["decision"] is not None:
            rep = torch.Generator().manual_seed(gen_seed + k)
            eta_t = float(torch.rand(1, generator=rep, dtype=torch.float64))
            eta_piece = torch.rand(e - a, generator=rep, dtype=torch.float64)
            if out["decision"] != "scan":
                return out, a, e, eta_t, eta_piece, None
            fl = out["walk"]["floor"]
            # The sub-draft from the start of the piece holding the floor.
            s0 = max(x for x in bounds if x <= fl)
            lb_s0 = advance(lr_all[:s0], 0.0, 0, 0)[0][-1]
            gs = torch.Generator().manual_seed(gen_seed + 1000)
            sc = carried_scan(
                lr_all[s0:a],
                lb_s0,
                s0,
                fl,
                a,
                torch.zeros(a - s0 + 1),
                torch.zeros(a - s0),
                lambda idx: P[s0 + idx],
                lambda idx: Q[s0 + idx],
                gs,
            )
            rep = torch.Generator().manual_seed(gen_seed + 1000)
            eta_scan = torch.rand(a - s0, generator=rep, dtype=torch.float64)
            return {"decision": "reject", **sc, "scan": (s0, eta_scan)}, a, e, eta_t, eta_piece, eta_scan
        walk = out["walk"]
    raise AssertionError("no decision")


@pytest.mark.parametrize("trial", range(60))
def test_pieces_match_one_block_pathwise(trial):
    # carried_piece/carried_scan decide exactly as one-block verification of the whole
    # block with the same draws (cut anywhere, decision at the end).
    rng = random.Random(trial)
    V, L = 5, 24
    g = torch.Generator().manual_seed(trial)
    # A long random draft with random per-position p, q (no prefix dependence needed).
    P = torch.log_softmax(torch.randn(L + 1, V, generator=g) * 0.5, -1)
    Q = torch.log_softmax(P + torch.randn(L + 1, V, generator=g) * 0.3, -1)[:L]

    class T:
        pass

    tbl = T()
    block = [rng.randrange(V) for _ in range(L)]
    tbl.lp = lambda pre: P[len(pre)]
    tbl.lq = lambda pre: Q[len(pre)]
    cuts = sorted(rng.sample(range(1, L), rng.randrange(0, 6)))
    out, a, e, eta_t, eta_piece, eta_scan = _run_pieces(tbl, block, cuts, DecisionRule("row"), True, 7 * trial)
    eta = torch.ones(L, dtype=torch.float64)
    eta[a:e] = eta_piece
    if eta_scan is not None:
        s0 = out["scan"][0]
        eta[s0:a] = eta_scan
    ref = block_decision_reference(P, Q, torch.tensor(block), eta_t, eta)
    got = L if out["decision"] == "whole" else out["tau"]
    assert got == ref


def _simulate_row(toy, c, rule, gen, seen=None, reset_walk=False):
    """The implementation's continuation process on the toy (complete row)."""
    row: list[int] = []
    while not toy.complete(row):
        # a new stream from ``row``: decode q, verify in pieces of c with a carried walk
        walk = {"log_b": 0.0, "floor": 0, "n": 0}
        block: list[int] = []
        lr_hist: list[float] = []
        piece_starts = []
        decided = False
        while not decided:
            a = len(block)
            piece_starts.append(a)
            while True:
                pre = row + block
                qv = toy.lq(pre).double().exp()
                v = int(torch.multinomial(qv.float(), 1, generator=gen))
                block.append(v)
                if toy.complete(row + block) or len(block) - a == c:
                    break
            final = toy.complete(row + block)
            e = len(block)
            P = torch.stack([toy.lp(row + block[:i]) for i in range(a, e + 1)])
            Q = torch.stack([toy.lq(row + block[:i]) for i in range(a, e)])
            toks = torch.tensor(block[a:e])
            ar = torch.arange(e - a)
            lpx, lqx = P[ar, toks], Q[ar, toks]
            out = carried_piece(lpx, lqx, torch.zeros(e - a + 1), torch.zeros(e - a),
                                lambda idx, P=P: P[idx], lambda idx, Q=Q: Q[idx], gen, walk, rule, final)
            lr_hist += (lpx - lqx).tolist()
            if seen is not None:
                seen[out["decision"]] = seen.get(out["decision"], 0) + 1
            if out["decision"] is None:
                # control: forgetting the walk between pieces is biased
                walk = {"log_b": 0.0, "floor": len(block), "n": len(block)} if reset_walk else out["walk"]
                continue
            decided = True
            if out["decision"] == "whole":
                # A whole-accepted block extends the row; the stream's later tokens
                # are q samples given it, the same law as a new stream (next loop).
                row = row + block
                break
            if out["decision"] == "scan":
                fl = out["walk"]["floor"]
                s0 = max(x for x in piece_starts if x <= fl)
                lb_s0 = advance(lr_hist[:s0], 0.0, 0, 0)[0][-1]
                P2 = torch.stack([toy.lp(row + block[:i]) for i in range(s0, a + 1)])
                Q2 = torch.stack([toy.lq(row + block[:i]) for i in range(s0, a)])
                out = carried_scan(lr_hist[s0:a], lb_s0, s0, fl, a, torch.zeros(a - s0 + 1),
                                   torch.zeros(a - s0), lambda idx: P2[idx], lambda idx: Q2[idx], gen)
            row = row + block[: out["tau"]] + [out["next"]]
            break
    return tuple(row)


@pytest.mark.parametrize("rule", [DecisionRule("row"), DecisionRule("dip", 0.1), DecisionRule("every")])
def test_implementation_monte_carlo(rule):
    toy = Toy(V=3, N=4, seed=5, temp=1.0)
    law = toy.p_law()
    gen = torch.Generator().manual_seed(123)
    z, seen = _mc_z(toy, law, rule, gen)
    assert z < 4.0, z
    # non-vacuous: rejections in the deciding piece and scans below it both happen
    assert seen.get("reject", 0) > 100 and seen.get("whole", 0) > 100
    if rule.mode != "every":
        assert seen.get("scan", 0) > 20 and seen.get(None, 0) > 100


def _mc_z(toy, law, rule, gen, n=20000, **kw):
    counts: dict = {}
    seen: dict = {}
    for _ in range(n):
        k = _simulate_row(toy, 2, rule, gen, seen, **kw)
        counts[k] = counts.get(k, 0) + 1
    assert set(counts) <= set(law)
    chi2 = sum((counts.get(k, 0) - n * pk) ** 2 / (n * pk) for k, pk in law.items())
    dof = len(law) - 1
    return (chi2 - dof) / math.sqrt(2 * dof), seen


def test_implementation_monte_carlo_control():
    # Forgetting the walk between pieces (deciding only at the end) is biased.
    toy = Toy(V=3, N=4, seed=5, temp=1.0)
    z, _ = _mc_z(toy, toy.p_law(), DecisionRule("row"), torch.Generator().manual_seed(9), reset_walk=True)
    assert z > 20, z
