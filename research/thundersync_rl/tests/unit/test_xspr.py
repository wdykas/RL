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
"""Exact SPR (thundersync_rl/xspr.py, xspr_engine.py): the row loop against a toy keyed engine (T4: every row
equals the from-scratch keyed decode of its target version, whatever the pauses), the mutations (each changes a
row), the key ledger, the engine check (tau at a chunk boundary), the wake cut and the eviction guard."""

from __future__ import annotations

import asyncio
import hashlib
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from thundersync_rl import xspr
from thundersync_rl.xspr_engine import (CHECK_ATTR, NOCACHE_ATTR, CheckAccumulator, _CheckTags, check_cache_blocks,
                                         check_rows, cut_touched)

V, EOS, PROMPT = 6, 0, [3, 4, 5]


def _u(*parts) -> float:
    h = hashlib.blake2b("/".join(map(str, parts)).encode(), digest_size=8).digest()
    return (int.from_bytes(h, "little") + 0.5) / 2.0**64


def gumbel(key: int, pos: int) -> np.ndarray:
    return np.array([-math.log(-math.log(_u("g", key, pos, v))) for v in range(V)])


def drift_logits(v: int, ctx: list) -> np.ndarray:
    """Version-dependent toy logits: EOS unlikely early, likelier late; each version shifts every logit."""
    base = np.array([_u("b", tuple(ctx[-2:]), t) * 3.0 for t in range(V)])
    base[EOS] -= 4.0 - 0.4 * len(ctx)
    return base + v * np.array([_u("d", v, len(ctx), t) * 2.0 for t in range(V)])


def log_softmax(z: np.ndarray) -> np.ndarray:
    m = z.max()
    return z - (m + math.log(np.exp(z - m).sum()))


class World:
    """A toy keyed engine: token at absolute position p of row key k = argmax(logits_v(context) + g(k, p)), with v
    the version installed when it is decoded. ``wakes`` {tick: version}: at that decode tick the version moves, every
    in-flight request gets its generated count appended to its wake marks (xspr_engine.mark_wake) and, with ``cut``,
    every one that has decoded a token is cut (as the engine's wake cut)."""

    def __init__(self, logits=drift_logits, wakes=None, cap=12, cut=True):
        self.logits = logits
        self.cut = cut
        self.version = 0
        self.wakes = dict(wakes or {})
        self.tick = 0
        self.inflight: list = []
        self.cap = cap
        self.requests = 0

    async def step(self) -> None:
        await asyncio.sleep(0)
        self.tick += 1
        if self.tick in self.wakes:
            self.version = self.wakes[self.tick]
            for r in self.inflight:
                r.wake_at.append(r.gen)
                if r.touched and self.cut:
                    r.cut = True

    def sample(self, v: int, ctx: list, key: int) -> tuple[int, float]:
        z = self.logits(v, ctx)
        t = int(np.argmax(z + gumbel(key, len(ctx))))
        return t, float(np.float32(log_softmax(z)[t]))

    def reference(self, v: int, key: int) -> tuple[list, list]:
        ctx, toks, lps = list(PROMPT), [], []
        while len(toks) < self.cap:
            t, lp = self.sample(v, ctx, key)
            toks.append(t), lps.append(lp), ctx.append(t)
            if t == EOS:
                break
        return toks, lps


class FakeImpl:
    """``_generate_response`` of the toy engine (the decode / check requests ``ExactRows`` sends)."""

    def __init__(self, world: World):
        self.w = world
        self._tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens=True: "")

    async def _generate_response(self, message_log, stop_strings, native_generation_data=None, forced=None):
        w = self.w
        w.requests += 1
        prompt = [int(t) for t in message_log[0]["token_ids"]]
        prefix, plps = list(forced["prefix"]), list(forced["prefix_logprobs"])
        data = forced["generation_data"]
        key = int(data["xspr_key"])
        check = None
        req = SimpleNamespace(cut=False, touched=False, wake_at=[], gen=0)
        w.inflight.append(req)
        out, lps = list(prefix), list(plps)
        try:
            if data.get("xspr_check"):
                await w.step()  # the check's prefill: one forward at the version installed now
                req.touched = True
                v, n = w.version, int(data["xspr_check"])
                ys, clps = [], []
                for j in range(n):
                    t, _ = w.sample(v, prompt + prefix[:j], key)
                    ys.append(t)
                    clps.append(float(np.float32(log_softmax(w.logits(v, prompt + prefix[:j]))[prefix[j]])))
                tau = next((j for j in range(n) if ys[j] != prefix[j]), n)
                check = {"tau": tau, "lps": clps, "y": ys[tau] if tau < n else -1, "n": n}
                cont = int(data.get("xspr_continue") or 0)
                budget = int(data["max_new_tokens"]) if cont == 2 or (cont == 1 and tau == n) else 0
                if not data.get("xspr_continue"):
                    out.append(0), lps.append(0.0)  # the discarded token of a check-only request
            else:
                budget = int(data["max_new_tokens"])
            for _ in range(budget):
                if out and out[-1] == EOS and len(out) > len(prefix):
                    break
                await w.step()
                if req.cut:
                    break
                t, lp = w.sample(w.version, prompt + out, key)
                out.append(t), lps.append(lp)
                req.touched = True
                req.gen += 1
                if t == EOS:
                    break
        finally:
            w.inflight = [r for r in w.inflight if r is not req]  # by identity (namespaces compare by value)
        msg = self._msg(out, lps)
        msg["xspr"] = {"cut": int(req.cut), "check": check, "wake_at": list(req.wake_at)}
        return msg, torch.tensor([len(prompt)]), {}

    @staticmethod
    def _msg(tokens, lps):
        return {"role": "assistant", "content": "", "token_ids": torch.tensor(tokens, dtype=torch.long),
                "generation_logprobs": torch.tensor(lps, dtype=torch.float32)}


def run_rows(world: World, target: int, n_rows: int = 6, neg: str = "", start_version: int = 0,
             keep_forks: bool = False):
    """Rows of batch ``target`` dispatched at ``start_version``; returns [(row output, reference, cert)]."""
    from nemo_rl.experience.rollout_manager import ROLLOUT_TARGET_STEP

    world.version = start_version
    rows = xspr.ExactRows(seed=7, stream="t", max_new_tokens=world.cap, version_fn=lambda: world.version, neg=neg,
                          keep_forks=keep_forks)
    impl = FakeImpl(world)

    async def one(j):
        ROLLOUT_TARGET_STEP.set(target)
        msg, _, _ = await rows.respond(impl, [{"token_ids": PROMPT}], None, None, {"idx": 11}, j)
        return msg

    async def main():
        msgs = await asyncio.gather(*(one(j) for j in range(n_rows)))
        # Rows of batch `target` are complete only once theta_target is installed (wait_version holds them).
        return msgs

    async def drive():
        task = asyncio.ensure_future(main())
        while not task.done():
            await world.step()
            if world.tick > 10000:
                task.cancel()
                raise AssertionError("rows never finished")
        return task.result()

    msgs = asyncio.run(drive())
    out = []
    for j, m in enumerate(msgs):
        key = rows.key_of(target, 11, j)
        out.append(((m["token_ids"].tolist(), m["generation_logprobs"].tolist()), world.reference(target, key),
                    rows.pop_cert(target, 11, j)))
    return out


def _assert_exact(results, target):
    for (toks, lps), (rt, rl), cert in results:
        assert toks == rt, (toks, rt)
        assert np.array_equal(np.float32(lps), np.float32(rl))
        assert cert.version == target


def test_pure_rows_need_no_check():
    w = World(wakes={})
    res = run_rows(w, target=0)
    _assert_exact(res, 0)
    assert all(c.checks == 0 for _, _, c in res)


def test_rows_across_one_wake_equal_the_target_decode():
    # lag 1: dispatched at version 0 for batch 1, the wake to 1 lands mid-row (cut, check, resume).
    w = World(wakes={20: 1})
    res = run_rows(w, target=1)
    _assert_exact(res, 1)
    assert any(c.checks for _, _, c in res)
    assert any(c.redecoded for _, _, c in res), "the toy drift must force mismatches"


def test_rows_resumed_without_the_cut_are_still_exact():
    # the cut is an optimization: a row the engine resumed on its stale prefix is labelled with its submission
    # version and checked like a finished row
    w = World(wakes={120: 1}, cut=False)
    res = run_rows(w, target=1, n_rows=24)
    _assert_exact(res, 1)
    assert any(c.redecoded for _, _, c in res)


def test_rows_finished_before_the_wake_are_checked():
    # every row finishes under version 0, before the wake to its batch's version 1 (D1)
    w = World(wakes={400: 1})
    res = run_rows(w, target=1)
    _assert_exact(res, 1)
    assert all(c.checks >= 1 for _, _, c in res)


def test_multiple_wakes_lag_two():
    w = World(wakes={6: 1, 30: 2})
    res = run_rows(w, target=2, n_rows=8)
    _assert_exact(res, 2)
    assert max(c.checks for _, _, c in res) >= 2


def test_no_mismatch_resumes_where_it_paused():
    w = World(logits=lambda v, ctx: drift_logits(0, ctx), wakes={20: 1})
    res = run_rows(w, target=1)
    _assert_exact(res, 1)
    assert all(c.redecoded == 0 for _, _, c in res)


def _first_token_flip(v, ctx):
    z = drift_logits(0, ctx)
    if len(ctx) == len(PROMPT):
        z = z.copy()
        z[(1 + v) % V or 1] += 50.0  # the first token is forced, differently per version
    return z


def test_tau_zero():
    w = World(logits=_first_token_flip, wakes={200: 1})
    res = run_rows(w, target=1)
    _assert_exact(res, 1)
    assert all(c.mism and min(c.mism) == 0 for _, _, c in res)


def _last_token_flip(v, ctx, cap=12):
    z = drift_logits(0, ctx)
    z[EOS] = -1e9
    if len(ctx) == len(PROMPT) + cap - 1:
        z = z.copy()
        z[2 + v] += 50.0
    return z


def test_tau_last_and_cap():
    # rows run to the cap; only the last token differs between versions: tau = cap - 1, a one-token re-decode
    w = World(logits=_last_token_flip, wakes={400: 1})
    res = run_rows(w, target=1)
    _assert_exact(res, 1)
    for _, (rt, _), c in res:
        assert len(rt) == w.cap and c.mism == (w.cap - 1,) and c.redecoded == 1


def _eos_moves(v, ctx):
    z = drift_logits(0, ctx)
    z[EOS] = -1e9
    if len(ctx) == len(PROMPT) + 4 + 2 * v:
        z = z.copy()
        z[EOS] = 1e9  # EOS at position 4 under version 0, 6 under version 1
    return z


def test_tau_at_eos_later_and_earlier():
    later = run_rows(World(logits=_eos_moves, wakes={400: 1}), target=1)
    _assert_exact(later, 1)
    assert all(c.mism == (4,) and c.tokens == 7 for _, _, c in later)
    earlier = run_rows(World(logits=lambda v, ctx: _eos_moves(1 - v, ctx), wakes={400: 1}), target=1)
    _assert_exact(earlier, 1)
    assert all(c.tokens == 5 for _, _, c in earlier)


def test_row_paused_before_its_first_token():
    # the wake lands at the first tick: requests already admitted are cut with zero tokens if touched; none is
    w = World(wakes={1: 1})
    res = run_rows(w, target=1)
    _assert_exact(res, 1)


@pytest.mark.parametrize("neg", xspr.NEGATIVES)
def test_each_mutation_changes_some_row(neg):
    if neg == xspr.NEG_NEW_ONLY:
        # only bites at a row's second check (lag 2): ~5 tokens per row before the first wake, done before the second
        w = World(wakes={120: 1, 3000: 2})
        res = run_rows(w, target=2, n_rows=24, neg=neg)
    elif neg == xspr.NEG_NO_WAKE_CUT:
        w = World(wakes={120: 1}, cut=False)  # the engine resumes paused rows on their stale state
        res = run_rows(w, target=1, n_rows=24, neg=neg)
    else:
        res = run_rows(World(wakes={20: 1}), target=1, n_rows=8, neg=neg)
    wrong = sum(1 for (t, l), (rt, rl), _ in res if t != rt or not np.array_equal(np.float32(l), np.float32(rl)))
    assert wrong > 0, f"{neg}: no row differs from the target decode"


def test_mutations_all():
    for neg in xspr.NEGATIVES:
        test_each_mutation_changes_some_row(neg)


def test_key_ledger():
    led = xspr.KeyLedger()
    led.open(5)
    led.draw(5, 0, 10)
    with pytest.raises(RuntimeError, match="no redraw permit"):
        led.draw(5, 3, 8)
    led.permit(5, 3)
    led.draw(5, 3, 9)
    with pytest.raises(RuntimeError, match="no redraw permit"):
        led.draw(5, 4, 9)  # one permit, used
    with pytest.raises(RuntimeError, match="two rows"):
        led.open(5)
    with pytest.raises(RuntimeError, match="beyond"):
        led.permit(5, 20)


def test_row_keys_unique_per_step_prompt_sample():
    keys = {xspr.row_key(1, "s", step, p, j) for step in range(3) for p in range(20) for j in range(8)}
    assert len(keys) == 3 * 20 * 8
    assert all(0 <= k < 2**62 for k in keys)


def _toy_check(x, flip_at, chunk):
    torch.manual_seed(0)
    n, d = x.numel(), 4
    rows = torch.randn(n, d)
    w = torch.randn(V, d)
    logits = lambda h: h @ w.t()  # noqa: E731
    calls = []

    def sample(z, keys, pos, temp):
        calls.append(int(z.shape[0]))
        y = x[pos - 100].clone()  # matches x except where the test flips it
        y[(pos - 100) == flip_at] = (x[flip_at] + 1) % V if flip_at < n else 0
        return y

    tau, lps, y = check_rows(rows, x, 9, 100, 1.0, logits, sample, lambda z: torch.log_softmax(z, -1), chunk)
    ref = torch.log_softmax(rows @ w.t(), -1).gather(-1, x.view(-1, 1))[:, 0]
    return tau, lps, y, ref, calls


def test_check_tau_on_a_chunk_boundary():
    x = torch.tensor([1, 2, 3, 4, 5, 1, 2, 3, 4])
    for flip in (3, 4, 8, 9):  # last row of chunk 0, first of chunk 1, last row, none
        tau, lps, y, ref, calls = _toy_check(x, flip, chunk=4)
        assert tau == min(flip, 9)
        assert len(lps) == 9 and torch.equal(torch.tensor(lps), ref)
        assert y == ((int(x[flip]) + 1) % V if flip < 9 else -1)
        assert calls == [4, 4, 1]  # never more than chunk rows of logits at once


def test_check_empty_row():
    assert check_rows(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), 1, 0, 1.0, None, None, None, 4) == (0, [], -1)


def _entry(rid, key=None, check=None, records=1, prefilled=0):
    sp = SimpleNamespace(temperature=1.0)
    attrs = {}
    if key is not None:
        attrs["ts_nl_key_rid"] = key
    if check is not None:
        attrs[CHECK_ATTR] = check
    sp.__dict__.update(attrs)
    sp.add_attributes = lambda a: sp.__dict__.update(a)
    req = SimpleNamespace(sampling_params=sp, prompt_tokens=torch.arange(10), finished_chunk_token_count=prefilled)
    rec = SimpleNamespace(requests=[req] * records)
    rec.__getitem__ = None
    return SimpleNamespace(record=_Rec([req] * records))


class _Rec(list):
    @property
    def requests(self):
        return list(self)


def test_wake_cut_marks_only_resident_keyed_requests():
    eng = SimpleNamespace(
        requests={1: _entry(1, key=5), 2: _entry(2, key=6), 3: _entry(3), 4: _entry(4, key=7)},
        resume_request_ids=[2], waiting_request_ids=[], stop_word_finished_request_ids=set())
    assert cut_touched(eng) == 2
    assert eng.stop_word_finished_request_ids == {1, 4}
    assert getattr(eng.requests[1].record[-1].sampling_params, "ts_nl_cut") == 1
    assert not hasattr(eng.requests[2].record[-1].sampling_params, "ts_nl_cut")
    assert cut_touched(eng) == 0  # idempotent
    # chunked prefill: a waiting request part-way through its prefill has run forwards (cut); an unstarted one has not
    eng = SimpleNamespace(
        requests={1: _entry(1, key=5, prefilled=256), 2: _entry(2, key=6)}, resume_request_ids=[],
        waiting_request_ids=[1, 2], stop_word_finished_request_ids=set())
    st = {}
    assert cut_touched(eng, st) == 1 and eng.stop_word_finished_request_ids == {1} and st == {"midprefill": 1}
    # negative control isspr_no_prefill_cut: the mid-prefill request is skipped (the behaviour before 4389aae5)
    eng.requests = {1: _entry(1, key=5, prefilled=256)}
    eng.stop_word_finished_request_ids = set()
    assert cut_touched(eng, {}, prefill_cut=False) == 0


def test_a_prefill_runs_on_one_weight_epoch():
    from thundersync_rl.xspr_engine import prefill_one_epoch

    seen = {}
    prefill_one_epoch(seen, np.array([1, 2]), 3, lambda rid: False)
    prefill_one_epoch(seen, np.array([1]), 3, lambda rid: False)  # a later chunk, same weights
    prefill_one_epoch(seen, np.array([2]), 4, lambda rid: rid == 2)  # cut at the wake: void, allowed to finish
    with pytest.raises(RuntimeError, match="weight epochs 3 and 4"):
        prefill_one_epoch(seen, np.array([1]), 4, lambda rid: False)  # a chunked prefill the cut missed


def test_check_request_rows_and_eviction_guard():
    eng = SimpleNamespace(requests={7: _entry(7, key=3, check=4), 8: _entry(8, key=3, check=4, records=2)})
    tags = _CheckTags(eng)
    kid = tags._lookup(7)
    assert (tags.lo[kid], tags.hi[kid]) == (5, 9)  # rows at positions 5..8 predict prompt tokens 6..9
    assert tags.meta[kid]["first"] == 6 and tags.meta[kid]["x"].tolist() == [6, 7, 8, 9]
    with pytest.raises(RuntimeError, match="evicted"):
        tags._lookup(8)


def test_config_validator():
    from thundersync_rl import xspr as X

    good = {
        "thundersync": {"mode": "xspr"},
        "policy": {
            "offload_optimizer_for_refit": False,
            "generation": {"backend": "megatron", "colocated": {"enabled": True}, "top_k": None, "top_p": 1.0,
                           "stop_strings": None,
                           "mcore_generation_config": {"kv_cache_management_mode": "persist",
                                                       "enable_chunked_prefill": False, "enable_prefix_caching": False,
                                                       "logprobs_mode": "raw_logprobs",
                                                       "materialize_only_last_token_logits": True}},
            "megatron_cfg": {"batch_invariant_mode": True, "zero_train_gen_mismatch": True, "freeze_moe_router": True,
                             "moe_router_bias_update_rate": 0.0},
        },
        "async_rl": {"sampler": {"name": "in_order", "max_lookahead_versions": 1}},
        "grpo": {"max_rollout_turns": 1},
    }
    good["thundersync"].update(crossfade_routing_m=0.01, crossfade_routing_extra=8, crossfade_routing_sparse=False)
    problems = X.config_problems(good)
    assert not problems, problems
    import copy

    # the rescore layer carries prefix caching, chunked prefill and KV offload (the BI comparison engine)
    pc = copy.deepcopy(good)
    pc["policy"]["generation"]["mcore_generation_config"].update(
        enable_chunked_prefill=True, enable_prefix_caching=True, kv_cache_management_mode="offload", max_tokens=512)
    pc["policy"]["max_total_sequence_length"] = 10240
    assert not X.config_problems(pc), X.config_problems(pc)
    nocp = copy.deepcopy(pc)
    nocp["policy"]["generation"]["mcore_generation_config"]["enable_chunked_prefill"] = False
    assert any("max_tokens" in x for x in X.config_problems(nocp))  # one-forward checks need the whole row

    bad = copy.deepcopy(good)
    bad["policy"]["generation"]["mcore_generation_config"]["kv_cache_management_mode"] = "recompute"
    bad["policy"]["generation"]["top_p"] = 0.9
    bad["async_rl"]["sampler"]["name"] = "windowed"
    p = X.config_problems(bad)
    assert any("kv_cache_management_mode" in x for x in p) and any("top-p" in x for x in p)
    assert any("in_order" in x for x in p)


def test_forks_keep_tau_and_the_discarded_suffix():
    w = World(wakes={400: 1})
    res = run_rows(w, target=1, keep_forks=True)
    _assert_exact(res, 1)
    for (toks, _), _, c in res:
        assert [f[0] for f in c.forks] == list(c.mism) and len(c.flips) == len(c.mism)
        for (tau, dead, q, p), (lq_x, lp_x, lp_y, _) in zip(c.forks, c.flips):
            assert len(dead) == c.pre_check_len - tau and dead[0] != toks[tau]  # the old token at tau lost
            assert len(q) == len(p) == c.pre_check_len and (lq_x, lp_x) == (q[tau], p[tau])


def test_crossfade_required_unless_a_reason_is_given():
    from thundersync_rl import nolanes

    base = {"thundersync": {"mode": "xspr"}, "policy": {"megatron_cfg": {"env_vars": {"NEMO_RL_MOE_DEVICE_DISPATCH": "1"}}}}
    mcfg = base["policy"]["megatron_cfg"]
    off = nolanes.crossfade_problems(xspr.crossfade_view(base), mcfg)
    assert off and "cross-fade routing must be on" in off[0]
    base["thundersync"]["crossfade_off_reason"] = "dense model"
    assert nolanes.crossfade_problems(xspr.crossfade_view(base), mcfg) == []
    base["thundersync"] = {"mode": "isspr", "crossfade_routing_m": 0.01}  # any mode built on xspr: same rule
    assert nolanes.crossfade_problems(xspr.crossfade_view(base), mcfg) == []
    base["thundersync"]["crossfade_routing_m"] = -1.0
    assert any("> 0" in x for x in nolanes.crossfade_problems(xspr.crossfade_view(base), mcfg))
    on = {}
    nolanes.apply_crossfade(xspr.crossfade_view({"thundersync": {"crossfade_routing_m": 0.01}}), on)
    assert on["crossfade_routing_m"] == 0.01 and on["crossfade_routing_extra"] == 8 and on["crossfade_routing_sparse"]


def test_nolanes_refuses_crossfade_off_without_a_reason():
    from thundersync_rl import nolanes

    cfg = nolanes.NolanesConfig(crossfade_routing_m=None)
    assert nolanes.crossfade_problems(cfg, {}) and not nolanes.crossfade_problems(
        nolanes.NolanesConfig(crossfade_routing_m=None, crossfade_off_reason="ablation"), {})


def test_tau_histogram_bins_every_resumed_row_once():
    certs = [xspr.RowCert(version=1, key=i, tokens=10, mism=m) for i, m in enumerate([(3,), (20,), (500, 15), (), (5000,)])]
    st = xspr.tau_stats(certs)
    assert st["tau_hist"] == [2, 1, 0, 0, 1]  # a row is binned by the mismatch it resumed from (its last)


def test_nonfinite_sampled_logprob_guard_dumps_and_raises():
    import os
    import tempfile

    import thundersync_rl.worker as wk
    from thundersync_rl.xspr_engine import XsprEngineMixin, nonfinite_dump, raise_failure

    saved = wk._request_positions, wk._request_keys
    wk._request_positions = lambda engine, ids, lens: np.array([5, 7])
    wk._request_keys = lambda engine, ids, lens: np.array([11, 12])
    try:
        logits = torch.randn(1, 2, 6)
        logits[0, 1, 2] = float("inf")
        new_tokens = torch.tensor([0, 2])
        lp = torch.log_softmax(logits[0].float(), -1)[torch.arange(2), new_tokens]  # row 1: inf - inf = nan

        def calc(logits_, new_tokens_, *a, **k):
            return lp.clone(), None

        ctx = SimpleNamespace(paused_request_count=0, request_ids=torch.tensor([3, 4], dtype=torch.int32),
                              token_to_request_idx=torch.tensor([0, 1]), active_token_count=2,
                              calculate_log_probs_tensors=calc)
        engine = SimpleNamespace(context=ctx)
        worker = SimpleNamespace(_request_lengths=lambda: None, _verify_cap={"static": torch.randn(2, 4)}, rank=0,
                                 _xspr={})
        d = tempfile.mkdtemp()
        XsprEngineMixin._xspr_guard_logprobs(worker, engine, d)
        # the engine loop thread does not raise (it would stall the run): the failure is recorded and raised by the
        # worker's next result / stats / wake call
        engine.context.calculate_log_probs_tensors(logits, new_tokens)
        with pytest.raises(RuntimeError, match="non-finite"):
            raise_failure(worker._xspr)
        (f,) = os.listdir(d)
        assert f == "nonfinite_r0_rid4_pos7.pt"
        rec = torch.load(os.path.join(d, f))
        assert rec["key"] == 12 and rec["token"] == 2 and rec["inf"] == 1 and rec["logits"].shape == (6,)
        assert rec["head_input"].shape == (4,)
        # finite log-probs pass through untouched
        lp = torch.tensor([-1.0, -2.0])
        sel, _ = engine.context.calculate_log_probs_tensors(logits, new_tokens)
        assert torch.equal(sel, lp)
        with pytest.raises(RuntimeError):
            nonfinite_dump(worker, engine, logits, new_tokens, torch.tensor([float("-inf"), -1.0]), "")
    finally:
        wk._request_positions, wk._request_keys = saved


def test_segment_refuses_non_finite_log_probs():
    class Bad(FakeImpl):
        def __init__(self, world, which):
            super().__init__(world)
            self.which = which

        async def _generate_response(self, *a, **k):
            msg, in_len, m = await super()._generate_response(*a, **k)
            if self.which == "decoded":
                msg["generation_logprobs"][-1] = float("nan")
            else:
                msg["xspr"]["check"]["lps"][0] = float("-inf")
            return msg, in_len, m

    w = World(wakes={})
    rows = xspr.ExactRows(seed=7, stream="t", max_new_tokens=w.cap, version_fn=lambda: w.version)

    async def go(impl, check, prefix):
        return await rows._segment(impl, [{"token_ids": PROMPT}], None, 5, prefix, [0.0] * len(prefix), check=check)

    with pytest.raises(RuntimeError, match="non-finite decoded"):
        asyncio.run(go(Bad(w, "decoded"), "", []))
    with pytest.raises(RuntimeError, match="non-finite rescored log-prob -inf at token position 0 of row key 5"):
        asyncio.run(go(Bad(w, "rescored"), "only", [3, 4]))
    with pytest.raises(RuntimeError, match="position 7"):
        xspr.require_finite([0.0, float("inf")], "decoded", 1, 6)


def test_check_capture_overflow_raises():
    from thundersync_rl.xspr_engine import check_capture

    check_capture(3, 3)
    with pytest.raises(RuntimeError, match="overflowed"):
        check_capture(3, 4)

def test_chunked_check_equals_one_forward():
    # A check's rows split over prefill chunks (chunked prefill), tau at / around the chunk boundaries: the
    # accumulated (tau, log-probs, y) equals the one-forward result bitwise.
    x = torch.tensor([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1])
    n = x.numel()
    torch.manual_seed(0)
    rows = torch.randn(n, 4)
    w = torch.randn(V, 4)

    def run(lo, hi, flip):
        def sample(z, keys, pos, temp):
            y = x[pos - 100].clone()
            y[(pos - 100) == flip] = (x[flip] + 1) % V if flip < n else 0
            return y
        return check_rows(rows[lo:hi], x[lo:hi], 9, 100 + lo, 1.0, lambda h: h @ w.t(), sample,
                          lambda z: torch.log_softmax(z, -1), 3)

    for cuts in ([4, 8], [1, 2, 10], [5]):
        for flip in (0, 3, 4, 7, 8, 10, n):  # chunk-boundary rows, the first / last row, none
            one = run(0, n, flip)
            acc = CheckAccumulator(n)
            bounds = [0, *cuts, n]
            done = [acc.add(a, *run(a, b, flip)) for a, b in zip(bounds, bounds[1:])]
            assert done == [False] * (len(bounds) - 2) + [True] and acc.parts == len(bounds) - 1
            tau, lps, y = acc.result()
            assert (tau, y) == (one[0], one[2]) and torch.equal(torch.tensor(lps), torch.tensor(one[1]))
    acc = CheckAccumulator(4)
    with pytest.raises(RuntimeError, match="out of order"):
        acc.add(2, 2, [0.0, 0.0], -1)
    acc.add(0, 4, [0.0] * 4, -1)
    with pytest.raises(RuntimeError):
        acc.add(4, 1, [0.0], -1)


def test_check_cache_blocks():
    def shared(out, h):
        # one hash per complete block (MCore's registration chain needs them all); the private ones are unique
        assert len(out) == len(h) and len(set(out)) == len(out)
        return [x for x, y in zip(out, h) if x == y]

    h = list(range(1, 11))
    # block 4, prompt 10 + 6 candidates: rows at positions 9 .. 14 must be computed -> blocks [0,4), [4,8) shared
    assert shared(check_cache_blocks(h[:4], 16, 6, 4, False), h[:4]) == [1, 2]
    assert shared(check_cache_blocks(h[:3], 15, 6, 4, False), h[:3]) == [1, 2]  # first - 1 = 8: [8,12) private
    assert shared(check_cache_blocks(h[:3], 14, 6, 4, False), h[:3]) == [1]  # first - 1 = 7 lies in [4,8)
    assert check_cache_blocks(h[:3], 12, 0, 4, False) == [1, 2, 3]  # not a check: every block may match
    assert shared(check_cache_blocks(h[:4], 16, 6, 4, True), h[:4]) == []  # an independent reference: none
    a, b = check_cache_blocks(h[:4], 16, 6, 4, True), check_cache_blocks(h[:4], 16, 6, 4, True)
    assert not set(a) & set(b) and all(0 < x < 2**63 for x in a)  # never shared across requests; int64 range


def test_check_cache_rule_on_mcore_requests():
    from megatron.core.inference.engines import dynamic_engine as de
    from megatron.core.inference.inference_request import DynamicInferenceRequest
    from megatron.core.inference.sampling_params import SamplingParams

    from thundersync_rl import xspr_engine as xe

    eng = SimpleNamespace(_weight_epoch=3)
    saved = DynamicInferenceRequest._compute_block_hashes
    try:
        xe.install_check_cache_rule(eng, stale_cache=False)

        def req(n=0, nocache=False, salt=None):
            sp = SamplingParams()
            attrs = {}
            if n:
                attrs[CHECK_ATTR] = n
            if nocache:
                attrs[NOCACHE_ATTR] = 1
            sp.add_attributes(attrs)
            return DynamicInferenceRequest(request_id=1, prompt_tokens=torch.arange(40), sampling_params=sp,
                                           enable_prefix_caching=True, block_size_tokens=8, block_hash_salt=salt)

        salt = de._weight_scoped_salt(3, None)
        plain, chk = req(salt=salt), req(n=20, salt=salt)
        ph = plain.precomputed_block_hashes
        assert len(ph) == 5 and len(chk.precomputed_block_hashes) == 5
        keep = (40 - 20 - 1) // 8
        ch = chk.precomputed_block_hashes
        assert ch[:keep] == ph[:keep] and not set(ch[keep:]) & set(ph)
        assert not set(req(n=20, nocache=True, salt=salt).precomputed_block_hashes) & set(ph)
        chk.resalt_block_hashes(de._weight_scoped_salt(4, None))  # the resume re-salt keeps the rule
        assert len(chk.precomputed_block_hashes) == 5 and not set(chk.precomputed_block_hashes[keep:]) & set(ph)
        # negative control: a check request salted with the previous epoch shares the older version's blocks
        xe.install_check_cache_rule(eng, stale_cache=True)
        old = req(salt=de._weight_scoped_salt(2, None)).precomputed_block_hashes
        stale = req(n=20, salt=salt).precomputed_block_hashes
        assert stale[:keep] == old[:keep] and stale[:keep] != ph[:keep]
    finally:
        DynamicInferenceRequest._compute_block_hashes = saved
        xe._CACHE_RULE.update(engine=None, stale=False)



def test_check_cache_rule_multi_block_prompts():
    # Regression (IndexError in MCore's _register_range): prompts of several KV blocks, two check
    # requests on one prompt (siblings) share its leading blocks -- a cache hit on a partial prefix -- and every
    # request carries one hash per complete block, the index contract MCore's registration chain relies on.
    from megatron.core.inference.engines import dynamic_engine as de
    from megatron.core.inference.inference_request import DynamicInferenceRequest
    from megatron.core.inference.sampling_params import SamplingParams

    from thundersync_rl import xspr_engine as xe

    saved = DynamicInferenceRequest._compute_block_hashes
    try:
        xe.install_check_cache_rule(SimpleNamespace(_weight_epoch=1), stale_cache=False)
        prompt = torch.arange(1000, 1032)  # 4 blocks of 8

        def check(tail):
            sp = SamplingParams()
            sp.add_attributes({CHECK_ATTR: len(tail)})
            r = DynamicInferenceRequest(request_id=1, prompt_tokens=torch.cat([prompt, torch.tensor(tail)]),
                                        sampling_params=sp, enable_prefix_caching=True, block_size_tokens=8,
                                        block_hash_salt=de._weight_scoped_salt(1, None))
            return r.precomputed_block_hashes

        a, b = check(list(range(20))), check(list(range(50, 70)))
        complete = (32 + 20) // 8
        assert len(a) == len(b) == complete  # one hash per complete block
        for h in (a, b):  # MCore's registration reads hashes[k] and hashes[k - 1] for every complete block k
            assert all(h[k] and (k == 0 or h[k - 1]) for k in range(complete))
        keep = (32 + 20 - 20 - 1) // 8  # blocks ending before the first candidate's head-input row (position 31)
        assert keep == 3 and a[:keep] == b[:keep]  # the siblings' shared prompt prefix hits
        assert not set(a[keep:]) & set(b[keep:])  # the block holding position 31 and the candidates: private
        assert xe._CACHE_RULE["max_blocks"] == complete and xe._CACHE_RULE["max_shared"] == keep
    finally:
        DynamicInferenceRequest._compute_block_hashes = saved
        xe._CACHE_RULE.update(engine=None, stale=False, max_blocks=0, max_shared=0)


def test_engine_thread_failures_are_recorded_then_raised():
    from thundersync_rl.xspr_engine import raise_failure, record_failure

    xs = {}
    raise_failure(xs)  # nothing recorded: no-op
    raise_failure(None)
    record_failure(xs, RuntimeError("first"))
    record_failure(xs, RuntimeError("second"))  # the first failure is the one reported
    with pytest.raises(RuntimeError, match="^RuntimeError: first$"):
        raise_failure(xs)
    xs = {}
    record_failure(xs, IndexError("list index out of range"))  # any exception type, named in the report
    with pytest.raises(RuntimeError, match="^IndexError: list index out of range$"):
        raise_failure(xs)



def test_nonfinite_row_is_dumped_for_replay():
    import json
    import tempfile

    from nemo_rl.experience.rollout_manager import ROLLOUT_TARGET_STEP

    def nan_late(v, ctx):  # the law turns NaN once the context reaches 6 tokens
        z = drift_logits(v, ctx)
        return z * np.nan if len(ctx) >= 6 else z

    world = World(logits=nan_late)
    rows = xspr.ExactRows(seed=7, stream="t", max_new_tokens=world.cap, version_fn=lambda: world.version)
    rows.dump_dir = tempfile.mkdtemp()
    impl = FakeImpl(world)

    async def drive():
        ROLLOUT_TARGET_STEP.set(0)
        task = asyncio.ensure_future(rows.respond(impl, [{"token_ids": PROMPT}], None, None, {"idx": 11}, 0))
        while not task.done():
            await world.step()
        return task.result()

    try:
        asyncio.run(drive())
        raise AssertionError("a non-finite row must raise")
    except RuntimeError as e:
        assert "row dumped to" in str(e)
        path = str(e).split("row dumped to ")[-1].split(")")[0].strip()
    d = json.loads(open(path).read())
    assert d["prompt_token_ids"] == PROMPT and d["key"] == rows.key_of(0, 11, 0) and d["key"] in d["open_keys"]
    assert len(d["token_ids"]) == len(d["logprobs"]["generation"]) and not math.isfinite(d["logprobs"]["generation"][-1])
