"""engine_symm_tp double-buffer schedule (host side): which staging region and whether a leading barrier.

The kernels' race freedom rests on these rules (engine_symm_tp docstring): consecutive collectives of one kind alternate
regions; the first collective after a model forward starts, every eager collective, the first captured collective
after an eager one or a fallback, and every collective when no forward hook is installed wait at a leading barrier.
"""

import pytest
import torch

from nemo_rl.models.megatron import engine_symm_tp as est


@pytest.fixture
def state(monkeypatch):
    saved = {k: (dict(v) if isinstance(v, dict) else v) for k, v in est._STATE.items()}
    cap = {"on": True}
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: cap["on"])
    est._STATE.update(parity={"rs": 0, "ag": 0}, first=True, prev_eager=True, hooked=True, lead=0)
    yield cap
    est._STATE.clear()
    est._STATE.update(saved)


def test_captured_forward_alternates_and_leads_once(state):
    est.forward_start()
    seq = [est._schedule(k) for k in ("ag", "rs") * 4]
    regions = [r for r, _, _ in seq]
    assert regions == [2, 0, 3, 1, 2, 0, 3, 1]
    assert [lead for _, lead, _ in seq] == [True] + [False] * 7
    est.forward_start()  # the next forward (or graph) starts at parity 0 again, with a leading barrier
    assert est._schedule("ag") == (2, True, True)


def test_eager_calls_always_lead(state):
    state["on"] = False
    est.forward_start()
    assert all(est._schedule(k)[1] for k in ("rs", "ag", "rs", "ag"))
    state["on"] = True  # first captured call after an eager one leads too
    assert est._schedule("rs")[1] is True
    assert est._schedule("ag")[1] is False


def test_fallback_forces_next_lead(state):
    est.forward_start()
    est._schedule("rs")
    est._STATE["prev_eager"] = True  # what reduce_scatter_rows / gather_rows set when a message falls back
    assert est._schedule("ag")[1] is True


def test_no_forward_hook_every_call_leads(state):
    est._STATE["hooked"] = False
    est.forward_start()
    assert all(est._schedule(k)[1] for k in ("rs", "ag") * 3)
