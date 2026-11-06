"""Draft-weight swaps drop the fp16 expert-weight cache (silent_bug_sweep.md).

The cache is keyed by (param id, data_ptr); load_draft / restore_training write the
parameters in place, so stale casts taken during the swap would hit again afterwards.
"""

from types import SimpleNamespace

from nemo_rl.models.megatron import fp16_moe
from thundersync_rl import worker as W


def test_load_and_restore_drop_fp16_expert_cache(monkeypatch):
    monkeypatch.setattr(W, "load_draft", lambda opt, kind: SimpleNamespace(stats={}))
    monkeypatch.setattr(W, "restore_training", lambda swap: {"restored": 1.0})
    monkeypatch.setattr(W, "check_no_expert_bias_updates", lambda model: None)
    monkeypatch.setattr(W, "_call_log", lambda *a, **k: None)
    w = object.__new__(W.ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)
    w.model, w.optimizer, w._weights_version, w._draft_swap = None, None, 0, None
    w._log_mem = lambda *a, **k: None
    fp16_moe._WEIGHT_CACHE[("a",)] = ((1,), "theta_k cast")
    w.load_draft_weights("momentum")
    assert len(fp16_moe._WEIGHT_CACHE) == 0
    fp16_moe._WEIGHT_CACHE[("a",)] = ((1,), "theta_hat cast (forward during the swap)")
    assert w.restore_training_weights() == {"restored": 1.0}
    assert len(fp16_moe._WEIGHT_CACHE) == 0
