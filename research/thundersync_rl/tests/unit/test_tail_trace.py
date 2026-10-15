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


def test_tail_dump_records_rows_rejections_and_dead_tokens(tmp_path, monkeypatch):
    """THUNDERSYNC_TAIL_DUMP: prompt, draft, final rollout, rejection positions and dead tokens."""
    import torch
    from types import SimpleNamespace

    from thundersync_rl.tail_trace import TailTrace

    monkeypatch.setenv("THUNDERSYNC_TAIL_DUMP", str(tmp_path))
    tr = TailTrace()
    p = SimpleNamespace(prefix=[1, 2, 9], complete=False, draft_tokens=[1, 2, 3, 4, 5])
    q = SimpleNamespace(prefix=[7, 7], complete=True, draft_tokens=[7, 7])
    tr.verified((100, 101), p, 5, 2)  # rejected at 2: dead [3, 4, 5]
    tr.verified((100, 101), q, 2, 2)
    p.prefix += [6, 6, 8]  # piece [6, 6, 7, 7] verified from position 3, rejected at its 2
    tr.piece(p, 3, 4, 2, 7, toks=[6, 6, 7, 7])
    p.complete = True
    tr.completed(p)
    tr.summary()
    (f,) = list(tmp_path.iterdir())
    d = torch.load(f, weights_only=False)
    rows = {tuple(r["draft"].tolist()): r for r in d["rows"]}
    r = rows[(1, 2, 3, 4, 5)]
    assert r["key"].tolist() == [100, 101] and r["final"].tolist() == [1, 2, 9, 6, 6, 8]
    assert [(s, pos, dead.tolist()) for s, pos, dead in r["rejects"]] == [(0, 2, [3, 4, 5]), (1, 5, [7, 7])]
    assert rows[(7, 7)]["rejects"] == []
