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
"""colo XS-SD config rules (colo.config_problems): prefix caching only on a weight-scoped Megatron-Core prefix cache,
the tree_stale_cache negative needs tree_xssd + prefix caching."""

import copy

from thundersync_rl import colo, nolanes


def _base():
    return {
        "thundersync": {"mode": "colo"},
        "async_rl": {"sampler": {"name": "in_order", "max_lookahead_versions": 0}},
        "policy": {
            "offload_optimizer_for_refit": False,
            "megatron_cfg": {"batch_invariant_mode": True, "zero_train_gen_mismatch": True,
                             "tensor_model_parallel_size": 1},
            "generation": {
                "backend": "megatron",
                "colocated": {"enabled": True},
                "mcore_generation_config": {"kv_cache_management_mode": "persist", "enable_prefix_caching": False},
            },
        },
    }


def _has(problems, text):
    return any(text in p for p in problems)


def test_prefix_caching_needs_weight_scoped_cache(monkeypatch):
    c = _base()
    c["policy"]["generation"]["mcore_generation_config"]["enable_prefix_caching"] = True
    monkeypatch.setattr(nolanes, "mcore_prefix_cache_weight_scoped", lambda: False)
    assert _has(colo.config_problems(c), "scoped by weight version")
    monkeypatch.setattr(nolanes, "mcore_prefix_cache_weight_scoped", lambda: True)
    assert not _has(colo.config_problems(c), "scoped by weight version")


def test_tree_stale_cache_negative_not_vacuous(monkeypatch):
    monkeypatch.setenv("THUNDERSYNC_NL_NEG", "tree_stale_cache")
    monkeypatch.setattr(nolanes, "mcore_prefix_cache_weight_scoped", lambda: True)
    c = _base()
    c["thundersync"]["tree_xssd"] = "8:7:256"
    assert _has(colo.config_problems(c), "tree_stale_cache needs tree_xssd and enable_prefix_caching")
    c["policy"]["generation"]["mcore_generation_config"]["enable_prefix_caching"] = True
    assert not _has(colo.config_problems(c), "tree_stale_cache")


def test_sprsd_config_rules(tmp_path, monkeypatch):
    """SPR-SD (thundersync.mode=sprsd): the stock loop + XS-SD's drafter; BI numerics, a W:D:rows tree, head file / TP 1
    with the MTP source, scoped prefix cache, no other thundersync keys."""
    from thundersync_rl import sprsd

    monkeypatch.setattr(nolanes, "router_frozen_problems", lambda c: [])
    c = _base()
    c["thundersync"] = {"mode": "sprsd", "tree_xssd": "8:7:256"}
    assert sprsd.config_problems(c) == []
    bad = copy.deepcopy(c)
    bad["thundersync"]["tree_xssd"] = "8:7"
    assert _has(sprsd.config_problems(bad), "W:D:rows")
    for k, v in (("stable_keys", True), ("wake_verify", True), ("k_drafts", 1.5), ("spare_prompts", 8),
                 ("variable_group", True), ("crossfade_routing_m", 0.03)):
        bad = copy.deepcopy(c)
        bad["thundersync"][k] = v  # any stale-row stage key: SPR-SD is SPR-BI's loop + the drafter only
        assert _has(sprsd.config_problems(bad), "not used by sprsd"), k
    bad = copy.deepcopy(c)
    bad["policy"]["megatron_cfg"]["batch_invariant_mode"] = False
    assert _has(sprsd.config_problems(bad), "batch_invariant_mode")
    bad = copy.deepcopy(c)
    bad["thundersync"]["tree_xssd_mtp"] = str(tmp_path / "missing.pt") + ":64"
    assert _has(sprsd.config_problems(bad), "no head file")
    pc = copy.deepcopy(c)
    pc["policy"]["generation"]["mcore_generation_config"]["enable_prefix_caching"] = True
    monkeypatch.setattr(nolanes, "mcore_prefix_cache_weight_scoped", lambda: False)
    assert _has(sprsd.config_problems(pc), "scoped by weight version")
    monkeypatch.setattr(nolanes, "mcore_prefix_cache_weight_scoped", lambda: True)
    assert sprsd.config_problems(pc) == []
