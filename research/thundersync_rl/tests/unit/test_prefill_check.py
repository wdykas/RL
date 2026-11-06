"""Engine prefill check (engine p == learner p on sampled verified rows): bounded row selection with
rejected / continued rows first, and a one-ulp engine/learner disagreement is reported (ts-exact-harden)."""

import itertools
from types import SimpleNamespace

import pytest
import torch
import thundersync_rl.speculative as spec_mod


class _Remote:
    def __init__(self, fn):
        self.remote = fn


class _Worker:
    def __init__(self, engine_lp):
        self.engine_lp = engine_lp  # key order -> logprobs the "engine" computes
        self.items = None
        self.run_verify_requests = _Remote(lambda rows, keys, plens, akeys: None)
        self.parallel_ranks = _Remote(lambda: {"tp": 0, "dp": 0, "last": True})
        self.engine_prefill_logprobs = _Remote(self._lp)
        self.engine_verify_drop = _Remote(lambda keys: None)

    def _lp(self, items, vocab_limit, head_shards=None):
        self.items = items
        return {it["key"]: self.engine_lp(it["tokens"]) for it in items}


def _spec(worker):
    s = object.__new__(spec_mod.SpeculativeGeneration)
    s.base = SimpleNamespace(_policy=SimpleNamespace(worker_group=SimpleNamespace(workers=[worker])))
    s.engine_max_len = None
    s._ev_ids = itertools.count()
    s.sticky_delta = None
    s.vocab_limit = 100
    s.stats = {}
    return s


def _rows():
    # (prompt, tokens, learner log-probs, anchors, had a rejection / continuation)
    out = []
    for i in range(12):
        toks = [i + 1] * (50 + i)
        out.append((torch.tensor([1, 2, 3]), toks, [-0.5 - 0.01 * j for j in range(len(toks))], None, i % 4 == 0))
    return out


@pytest.fixture(autouse=True)
def _ray(monkeypatch):
    monkeypatch.setattr(spec_mod.ray, "get", lambda x: x)
    monkeypatch.setenv("THUNDERSYNC_ENGINE_PREFILL_CHECK_ROWS", "4")
    monkeypatch.setenv("THUNDERSYNC_ENGINE_PREFILL_CHECK_TOKENS", "100000")


def test_exact_engine_passes_and_rejected_rows_go_first():
    rows = _rows()
    by_tok = {tuple(r[1]): r[2] for r in rows}
    w = _Worker(lambda toks: by_tok[tuple(toks)])
    s = _spec(w)
    s._engine_prefill_check(rows)
    assert s.stats["spec/engine_prefill_check_rows"] == 4
    assert s.stats["spec/engine_prefill_check_tail_rows"] == 3  # rows 0, 4, 8 had rejections
    assert s.stats["spec/engine_prefill_check_exact"] == s.stats["spec/engine_prefill_check_tok"]
    assert s.stats["spec/engine_prefill_check_max_abs_dlogp"] == 0.0


def test_one_ulp_engine_disagreement_is_reported():
    rows = _rows()
    by_tok = {tuple(r[1]): r[2] for r in rows}

    def engine(toks):  # injected bug: the engine's p of one token is one fp32 ulp off
        lp = list(by_tok[tuple(toks)])
        lp[7] = float(torch.nextafter(torch.tensor(lp[7]), torch.tensor(0.0)))
        return lp

    s = _spec(_Worker(engine))
    s._engine_prefill_check(rows)
    assert s.stats["spec/engine_prefill_check_exact"] == s.stats["spec/engine_prefill_check_tok"] - 4
    assert s.stats["spec/engine_prefill_check_max_abs_dlogp"] > 0


def test_token_budget_bounds_the_check():
    import os

    os.environ["THUNDERSYNC_ENGINE_PREFILL_CHECK_TOKENS"] = "120"
    rows = _rows()
    by_tok = {tuple(r[1]): r[2] for r in rows}
    s = _spec(_Worker(lambda toks: by_tok[tuple(toks)]))
    s._engine_prefill_check(rows)
    assert s.stats["spec/engine_prefill_check_rows"] == 2  # 53 + 57 tokens fit, the third does not
