"""Guards that replaced silent fallbacks (research/silent_bug_sweep.md)."""

import torch



class _Wrap(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.module = inner




def test_fp16_range_counter_counts_overflow_and_skips_fp16():
    from nemo_rl.models.megatron import fp16_gemm as G

    G.range_monitor(True)
    try:
        G.range_take()
        G.range_count(torch.tensor([1.0, 7e4, -1e5, 3.0]))
        G.range_count(torch.tensor([1.0], dtype=torch.float16))  # already fp16: not a cast site
        assert G.range_take() == {"overflow": 2.0, "checked": 4.0}
        assert G.range_take() == {"overflow": 0.0, "checked": 0.0}
    finally:
        G.range_monitor(False)
    G.range_count(torch.tensor([1e6]))  # off: not counted
    assert G.range_take()["overflow"] == 0.0




def test_deferred_head_materialization_counted_and_strict(monkeypatch):
    import pytest

    from nemo_rl.models.megatron import deferred_head as D

    D.materialized_take()
    D._note_materialized(torch.ops.aten.add.Tensor, "function", True)
    D._note_materialized(torch.ops.aten.add.Tensor, "dispatch", True, below_autograd=True)
    assert D.materialized_take() == {"function": 1, "dispatch": 1, "dispatch_grad": 1}
    monkeypatch.setenv("THUNDERSYNC_STRICT_DEFERRED_HEAD", "1")
    with pytest.raises(RuntimeError, match="no gradient"):
        D._note_materialized(torch.ops.aten.add.Tensor, "dispatch", True, below_autograd=True)
    D._note_materialized(torch.ops.aten.add.Tensor, "dispatch", False, below_autograd=True)  # no grad: ok


