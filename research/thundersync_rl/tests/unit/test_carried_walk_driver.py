"""Carried-walk continuation rounds in the speculative driver (DESIGN 14t; no Ray, no GPUs)."""

import array
import asyncio

import pytest
import thundersync_rl.speculative as spec_mod
import torch
from test_speculative_driver import EOD, _spec as make_spec
from test_sticky_driver import _DecodingEngine


@pytest.fixture(autouse=True)
def _no_ray(monkeypatch):
    monkeypatch.setattr(spec_mod.ray, "get", lambda x: x)


def test_carried_block_decides_at_stream_end_and_scans_below():
    """rule "row": pieces advance one walk; only the stream's last piece decides;
    a rejection whose last hit lies before the deciding piece is resolved by a scan
    of the block from the floor's piece; then a new stream continues the row."""
    engine = _DecodingEngine(
        [
            [([4], False, [-0.1], 2, [5, 6, 7, 8, 9, EOD])],
            [([EOD], True, [-0.3], 0)],  # new stream from [1, 4, 5, 6, 7, 30]
        ]
    )
    calls = []

    def walk_out(walk, g, decision, log_b, floor, **extra):
        return {
            "walk": {"log_b": log_b, "floor": floor, "n": walk["n"] + g},
            "decision": decision,
            "token_logprobs": [-1.0] * g,
            "log_ratio": [-0.05] * g,
            **extra,
        }

    script = [
        lambda w, g: walk_out(w, g, None, -0.1, 1),
        lambda w, g: walk_out(w, g, None, -0.3, 1),
        lambda w, g: walk_out(w, g, "scan", -0.5, 1),
        lambda w, g: walk_out(w, g, "whole", 0.0, 1),
    ]

    def verify(rows, prompt_lens, engine_logq, hidden, walks=None, walk_scans=None, **kw):
        calls.append(
            dict(
                rows=[r.tolist() for r in rows],
                plens=list(prompt_lens),
                walks=walks,
                scans=walk_scans,
                hidden_rows=[h.shape[0] for h in hidden],
            )
        )
        if walks is None and walk_scans is None:  # the drafts' first verification
            return [(0, {"accepted": 1, "next": 99, "logprobs": [-1.0, -1.0], "draft_logq": engine_logq[0][:1]})]
        if walk_scans is not None:
            sc = walk_scans[0]
            return [(0, {"tau": 3, "next": 30, "next_logp": -2.0, "accepted": 3 - sc["pos0"],
                         "walk_ratio_mismatch": 0})]
        g = rows[0].numel() - prompt_lens[0]
        return [(0, script.pop(0)(walks[0], g))]

    s = make_spec(
        {"verify_drafts_block_engine": verify},
        base_handlers=engine.handlers(),
        verify_source="engine",
        hidden_transport="driver",
        sticky_delta=0.03,
        sticky_chunk_tokens=2,
        sticky_decision="row",
    )
    drafts = asyncio.run(s.draft_streaming([torch.tensor([1])]))()
    s.current_step = 1
    asyncio.run(s.verify_sticky(drafts))
    v0, p1, p2, p3, scan, p4 = calls
    # pieces carry the pending block in their prompt and the walk state
    assert p1["rows"] == [[1, 4, 5, 6]] and p1["plens"] == [2]
    assert p2["rows"] == [[1, 4, 5, 6, 7, 8]] and p2["plens"] == [4]
    assert p3["rows"] == [[1, 4, 5, 6, 7, 8, 9, EOD]] and p3["plens"] == [6]
    assert p1["walks"][0]["n"] == 0 and p2["walks"][0] == {
        "log_b": -0.1, "floor": 1, "n": 2, "final": False, "rule": {"mode": "row", "dip": 0.1}
    }
    assert p3["walks"][0]["final"] is True and p3["walks"][0]["n"] == 4
    # the scan: block positions [0, 4) (the floor 1 lies in the first piece)
    sc = scan["scans"][0]
    assert (sc["pos0"], sc["floor"], sc["stop"]) == (0, 1, 4)
    assert sc["log_b0"] == 0.0 and sc["log_ratio"] == array.array("f", [-0.05] * 4).tolist()
    assert scan["rows"] == [[1, 4, 5, 6, 7, 8]] and scan["plens"] == [2]
    assert scan["hidden_rows"] == [4]
    # rejection at block position 3: keep [5, 6, 7] + 30; a new stream finishes the row
    assert [p for p, _, _ in engine.started[1:]] == [[[1, 4, 5, 6, 7, 30]]]
    assert p4["rows"] == [[1, 4, 5, 6, 7, 30, EOD]] and p4["walks"][0]["n"] == 0
    (plan,) = s.plans[(1,)]
    assert plan.complete and plan.prefix == [4, 5, 6, 7, 30, EOD]
    assert plan.prefix_logprobs[-3:] == [-1.0, -2.0, -1.0]
    assert plan.anchors.length == 1 + 6
    st = s.stats
    assert st["spec/cw_decisions"] == 2 and st["spec/cw_scans"] == 1
    assert st["spec/cw_rejects"] == 1 and st["spec/cw_whole"] == 1
    assert st["spec/cw_scan_tokens"] == 4 and st["spec/cw_ratio_mismatch"] == 0


def test_carried_whole_block_extends_and_stream_goes_on():
    """A non-final decision that keeps the block extends the prefix; the same
    stream goes on with a new block (walk reset)."""
    engine = _DecodingEngine([[([4], False, [-0.1], 2, [5, 6, 7, 8, 9, EOD])]])
    calls = []

    def verify(rows, prompt_lens, engine_logq, walks=None, walk_scans=None, **kw):
        calls.append((walks, list(prompt_lens)))
        if walks is None:
            return [(0, {"accepted": 1, "next": 99, "logprobs": [-1.0, -1.0], "draft_logq": engine_logq[0][:1]})]
        g = rows[0].numel() - prompt_lens[0]
        w = walks[0]
        dec = "whole" if (len(calls) == 3 or w["final"]) else None
        return [(0, {"walk": {"log_b": -0.2, "floor": 0, "n": w["n"] + g}, "decision": dec,
                     "token_logprobs": [-1.0] * g, "log_ratio": [-0.1] * g})]

    s = make_spec(
        {"verify_drafts_block_engine": verify},
        base_handlers=engine.handlers(),
        verify_source="engine",
        hidden_transport="driver",
        sticky_delta=0.03,
        sticky_chunk_tokens=2,
        sticky_decision="dip",
        sticky_decision_dip=0.15,
    )
    drafts = asyncio.run(s.draft_streaming([torch.tensor([1])]))()
    s.current_step = 1
    asyncio.run(s.verify_sticky(drafts))
    walks = [w for w, _ in calls[1:]]
    assert [w[0]["n"] for w in walks] == [0, 2, 0]  # block reset after the 2nd piece
    assert [p for _, p in calls[1:]] == [[2], [4], [6]]
    (plan,) = s.plans[(1,)]
    assert plan.complete and plan.prefix == [4, 5, 6, 7, 8, 9, EOD]
    assert len(engine.started) == 1  # no new stream
    assert s.stats["spec/cw_whole"] == 2 and s.stats["spec/cw_rejects"] == 0


def test_carried_walk_config_validation():
    from types import SimpleNamespace

    from thundersync_rl.carried_walk import DecisionRule
    from thundersync_rl.grpo_loop import validate_carried_walk

    with pytest.raises(ValueError):
        DecisionRule("bogus")
    with pytest.raises(ValueError):
        DecisionRule("dip", 0.0)

    def cfg(**kw):
        ts = dict(sticky_continuation_decision="row", sticky_continuation_decision_dip=0.1,
                  sticky_routing=True, verify_source="engine", verify_on="learner", banded_is_band=None)
        ts.update(kw)
        return SimpleNamespace(thundersync=SimpleNamespace(**ts))

    validate_carried_walk(cfg())
    validate_carried_walk(cfg(sticky_continuation_decision="every", sticky_routing=False))
    for bad in (dict(sticky_routing=False), dict(verify_on="both"), dict(banded_is_band=1.0)):
        with pytest.raises(ValueError):
            validate_carried_walk(cfg(**bad))
