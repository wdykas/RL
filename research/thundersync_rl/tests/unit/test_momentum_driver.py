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
"""Momentum draft weights in the speculative driver and the config (no Ray, no GPUs)."""

from types import SimpleNamespace

import pytest
import thundersync_rl.speculative as spec_mod
import torch
from test_speculative_driver import _spec

ENGINE = {"install_draft_hidden_log": lambda: True}
LOAD = {
    "draft/compute_s": 0.1,
    "draft/gather_s": 0.2,
    "draft/peak_extra_bytes": 2.0**30,
    "draft/changed_elems": 3.0,
    "draft/shard_elems": 4.0,
}


@pytest.fixture(autouse=True)
def _no_ray(monkeypatch):
    monkeypatch.setattr(spec_mod.ray, "get", lambda x: x)


def _momentum_spec(order):
    return _spec(
        {
            "load_draft_weights": lambda kind: order.append(("load", kind)) or LOAD,
            "stash_head": lambda version, keep: order.append(("stash", version, keep)),
            "restore_training_weights": lambda: order.append(("restore",))
            or {"draft/restore_s": 0.05},
        },
        base_handlers=ENGINE,
        verify_source="engine",
        draft_weights="momentum",
    )


def test_refit_loads_stashes_refits_restores_in_order():
    order = []
    s = _momentum_spec(order)
    s.current_step = 5
    s.refit_draft_weights(lambda: order.append(("refit",)))
    # The drafting LM head is stashed while the draft weights are loaded, and the
    # engine refit reads them; theta_k comes back last.
    assert order == [
        ("load", "momentum"),
        ("stash", 5, [5]),
        ("refit",),
        ("restore",),
    ]
    assert s.stats["draft/changed_frac"] == 0.75
    assert s.stats["draft/peak_extra_gb"] == 1.0
    assert s.stats["draft/restore_worker_s"] == 0.05


def test_refit_failure_still_restores_training_weights():
    order = []
    s = _momentum_spec(order)

    def refit():
        raise RuntimeError("refit failed")

    with pytest.raises(RuntimeError, match="refit failed"):
        s.refit_draft_weights(refit)
    assert order[-1] == ("restore",)


def test_score_prev_keeps_the_momentum_stash():
    order = []
    s = _momentum_spec(order)
    drafts = [[(torch.tensor([1, 2]), -1, [3], True, "1:0", [(0, 0, 1)])]]
    s.score_prev(drafts)  # theta_k did not draft: no stash of the live head
    assert order == []


def test_current_draft_weights_stash_at_score_prev():
    order = []
    s = _spec(
        {"stash_head": lambda version, keep: order.append(("stash", version))},
        base_handlers=ENGINE,
        verify_source="engine",
    )
    assert s.draft_weights == "current"
    s.score_prev([[(torch.tensor([1, 2]), -1, [3], True, "1:0", [(0, 0, 1)])]])
    assert order == [("stash", 0)]


def test_momentum_needs_engine_q_on_the_learner():
    with pytest.raises(ValueError, match="verify_source=engine"):
        _spec({}, draft_weights="momentum")  # scorer q would come from theta_k
    with pytest.raises(ValueError, match="unknown draft_weights"):
        _spec({}, base_handlers=ENGINE, verify_source="engine", draft_weights="adam")


def _master(opt=None, **ts):
    from thundersync_rl.grpo_loop import ThunderSyncConfig

    base = dict(
        speculate=True,
        verify_mode="block",
        verify_source="engine",
        verify_on="learner",
        draft_lookahead=1,
        draft_weights="momentum",
    )
    base.update(ts)
    return SimpleNamespace(
        thundersync=ThunderSyncConfig(**base),
        policy={
            "megatron_cfg": {
                "optimizer": {
                    "optimizer": "adam",
                    "use_distributed_optimizer": True,
                    "use_precision_aware_optimizer": False,
                    **(opt or {}),
                }
            }
        },
    )


def test_config_default_current_and_momentum_valid():
    from thundersync_rl.grpo_loop import ThunderSyncConfig, validate_draft_weights

    assert ThunderSyncConfig().draft_weights == "current"
    validate_draft_weights(_master())
    validate_draft_weights(_master(draft_weights="current", speculate=False))


@pytest.mark.parametrize(
    "opt,ts,msg",
    [
        ({}, {"verify_source": "scorer"}, "verify_source"),
        ({}, {"verify_on": "both"}, "verify_on"),
        ({}, {"draft_lookahead": 2}, "draft_lookahead"),
        ({}, {"draft_after_verify": True}, "draft_after_verify"),
        ({}, {"fused_verify_train": True}, "fused_verify_train"),
        ({}, {"speculate": False}, "speculate"),
        ({"use_distributed_optimizer": False}, {}, "use_distributed_optimizer"),
        ({"use_precision_aware_optimizer": True}, {}, "precision_aware"),
        ({"optimizer": "sgd"}, {}, "'adam'"),
    ],
)
def test_config_rejections(opt, ts, msg):
    from thundersync_rl.grpo_loop import validate_draft_weights

    with pytest.raises(ValueError, match=msg):
        validate_draft_weights(_master(opt, **ts))
