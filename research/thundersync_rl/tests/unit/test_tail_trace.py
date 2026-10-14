"""TailTrace: the rank-0 post-refit tail timeline (measurement only)."""

from dataclasses import dataclass, field

from thundersync_rl.tail_trace import TailTrace


@dataclass
class _Plan:
    prefix: list = field(default_factory=list)
    complete: bool = False


def test_rows_rejects_and_groups():
    tr = TailTrace(t0=0.0)
    tr.now = lambda: 1.0  # fixed clock
    whole = _Plan(prefix=[1] * 10, complete=True)  # verify0: whole-accepted finished draft
    rej = _Plan(prefix=[1] * 4)  # verify0: 3 of 8 kept + residual token
    tr.verified("a", whole, 10, 10)
    tr.verified("b", rej, 8, 3)
    tr.now = lambda: 2.0
    # Continuation piece of 6 tokens from stream offset 0, rejected after 2 (stream at 9 tokens).
    rej.prefix += [1, 1, 7]
    tr.piece(rej, 0, 6, 2, 9)
    tr.round(live=1, ready=1, verify_s=0.1, rejects=1)
    tr.now = lambda: 3.0
    rej.prefix += [1] * 5
    rej.complete = True
    tr.piece(rej, 0, 5, 5, 5)
    tr.completed(rej)
    out = tr.summary()
    assert out["tail/rows"] == 2 and out["tail/v0_rejected_rows"] == 1
    assert out["tail/cont_rejects"] == 1
    assert out["tail/end_s"] == 3.0
    assert out["tail/v0_wasted_tokens_max"] == 5  # 8 drafted - 3 kept
    assert out["tail/cont_wasted_tokens_mean"] == 7  # 9 decoded - (0 + 2)
    assert out["tail/slowest_v0_rejected"] == 1 and out["tail/slowest_final_len"] == 12
    rows = {r[0]: r for r in out["timeline/tail_rows"]}
    assert rows[0][-1] == 1.0 and rows[1][-1] == 3.0  # group 0 done at verify0
    assert [r[0] for r in out["timeline/tail_rejects"]] == [0, 1]
    assert out["timeline/tail_rejects"][1][1] == 6  # residual token position
    assert out["timeline/tail_rounds"] == [[2.0, 1, 1, 0.1, 1]]


def test_incomplete_rows_close_at_summary():
    tr = TailTrace(t0=0.0)
    tr.now = lambda: 1.0
    tr.verified("a", _Plan(prefix=[1] * 3), 3, 3)
    tr.now = lambda: 4.0
    out = tr.summary()
    assert out["tail/end_s"] == 4.0 and out["tail/group_done_q50_s"] == 4.0
