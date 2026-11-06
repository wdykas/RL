"""Engine prefill check rows when the check runs before the rollout loop served any row: a step whose drafts all
complete in their verification (no tails) has no served row at check time; the published, unserved complete plans
are the candidates. A step with no complete verified plan flags spec/engine_prefill_check_skipped."""

import asyncio
from collections import defaultdict

import torch
import thundersync_rl.speculative as spec_mod
from thundersync_rl.speculative import Plan


def _spec(plans, served=None):
    s = object.__new__(spec_mod.SpeculativeGeneration)
    s.stats = {}
    s.plans = defaultdict(list, plans)
    s._prefill_check_rows = served if served is not None else []
    s.checked = None

    def check(rows):
        s.checked = rows

    s._engine_prefill_check = check
    return s


def _plan(tokens, complete=True, verified=(4, 4), tail=False):
    p = Plan(seed=-1, prefix=list(tokens), prefix_logprobs=[-0.25] * len(tokens), complete=complete)
    p.verified, p.tail = verified, tail
    return p


def test_unserved_complete_plans_are_checked_when_none_was_served():
    plans = {(1, 2, 3): [_plan([7, 8, 9, 10]), _plan([5, 6], complete=False)], (4, 5): [_plan([11, 12], tail=True)],
             (6,): [_plan([13], verified=None)]}
    s = _spec(plans)
    asyncio.run(s.engine_prefill_check())
    rows = s.checked
    assert rows is not None and len(rows) == 2  # incomplete / unverified plans are not candidates
    by_prompt = {tuple(r[0].tolist()): r for r in rows}
    assert by_prompt[(1, 2, 3)][1] == [7, 8, 9, 10] and by_prompt[(1, 2, 3)][2] == [-0.25] * 4
    assert by_prompt[(4, 5)][4] is True and by_prompt[(1, 2, 3)][4] is False
    assert all(r[0].dtype == torch.long for r in rows)
    assert "spec/engine_prefill_check_skipped" not in s.stats
    assert s._prefill_check_rows is None


def test_served_rows_are_used_as_before():
    served = [(torch.tensor([9]), [1, 2], [-0.5, -0.5], None, False)]
    s = _spec({(1, 2, 3): [_plan([7, 8])]}, served=list(served))
    asyncio.run(s.engine_prefill_check())
    assert len(s.checked) == 1 and s.checked[0][1] == [1, 2]


def test_no_candidate_flags_skipped():
    s = _spec({(1,): [_plan([2], complete=False)]})
    asyncio.run(s.engine_prefill_check())
    assert s.checked is None
    assert s.stats["spec/engine_prefill_check_skipped"] == 1.0


def test_served_row_recording_is_unchanged():
    rows = []
    spec_mod.SpeculativeGeneration._prefill_check_record(rows, torch.tensor([1]), _plan([3, 4], verified=(2, 1)))
    spec_mod.SpeculativeGeneration._prefill_check_record(rows, torch.tensor([1]), _plan([3, 4], complete=False))
    assert len(rows) == 1 and rows[0][4] is True  # rejection inside the draft
