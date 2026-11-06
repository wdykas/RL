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
"""Verify-wall knobs' validation (thundersync.verify_batch_tokens, THUNDERSYNC_VWALL_CHECK; DESIGN §15, R36)."""

from thundersync_rl import nolanes


def _valid_cfg():
    import importlib

    t = importlib.import_module("test_nolanes")
    return t._valid()


def test_verify_batch_tokens_validation():
    c = _valid_cfg()
    c["policy"]["train_micro_batch_size"] = 2
    c["policy"]["max_total_sequence_length"] = 10240
    c["thundersync"]["verify_batch_tokens"] = 20480
    assert nolanes.config_problems(c) == []
    c["thundersync"]["verify_batch_tokens"] = 20481  # over the micro-batch: refused only under CALL_CAP
    assert not any("verify_batch_tokens" in p for p in nolanes.config_problems(c))
    c["policy"].setdefault("megatron_cfg", {}).setdefault("env_vars", {})["NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP"] = "1"
    assert any("verify_batch_tokens" in p for p in nolanes.config_problems(c))
    del c["policy"]["megatron_cfg"]["env_vars"]["NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP"]
    c["thundersync"]["verify_batch_tokens"] = 0
    assert any("verify_batch_tokens" in p for p in nolanes.config_problems(c))


def test_verify_batch_tokens_fit_the_dispatch_capacity(monkeypatch):
    """R36 (c): the effective verify budget must fit NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS (knob or default)."""
    c = _valid_cfg()
    c["policy"].update(train_micro_batch_size=1, max_total_sequence_length=20480)
    envs = c["policy"].setdefault("megatron_cfg", {}).setdefault("env_vars", {})
    envs.update(NEMO_RL_MOE_DEVICE_DISPATCH="1", NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS="18432")
    c["thundersync"]["verify_batch_tokens"] = 20480  # within the micro-batch, over the dispatch capacity (Ultra shape)
    assert any("DISPATCH_TOKENS" in p for p in nolanes.config_problems(c))
    c["thundersync"]["verify_batch_tokens"] = None  # the rule: min(MBS x max seq, dispatch capacity) fits
    assert not any("DISPATCH_TOKENS" in p for p in nolanes.config_problems(c))
    assert nolanes.effective_verify_batch_tokens(c, nolanes.NolanesConfig.from_config(c)) == 18432


def test_verify_batch_tokens_rule():
    """Structural review L17: None derives MBS x max_total_sequence_length, capped by the dispatch capacity."""
    c = _valid_cfg()
    c["thundersync"]["verify_batch_tokens"] = None
    c["policy"].update(train_micro_batch_size=2, max_total_sequence_length=10240)
    f = lambda: nolanes.effective_verify_batch_tokens(c, nolanes.NolanesConfig.from_config(c))  # noqa: E731
    assert f() == 20480  # Super: MBS 2 x 10240
    c["policy"]["train_micro_batch_size"] = 1
    assert f() == 10240  # Ultra: MBS 1
    c["thundersync"]["verify_batch_tokens"] = 4096
    assert f() == 4096  # an explicit value wins
    c["thundersync"]["verify_batch_tokens"] = None
    c["policy"].pop("train_micro_batch_size")
    assert f() is None


def test_vwall_check_refused_without_checks(monkeypatch):
    """R36 (g): THUNDERSYNC_VWALL_CHECK doubles every DS2 row tail; it is a debug knob refused without CHECKS."""
    monkeypatch.delenv("THUNDERSYNC_NL_CHECKS", raising=False)
    monkeypatch.delenv("THUNDERSYNC_NL_NEG", raising=False)
    c = _valid_cfg()
    c["policy"].setdefault("megatron_cfg", {}).setdefault("env_vars", {})["THUNDERSYNC_VWALL_CHECK"] = "1"
    assert any("THUNDERSYNC_VWALL_CHECK" in p for p in nolanes.config_problems(c))
    monkeypatch.setenv("THUNDERSYNC_NL_CHECKS", "1")
    assert not any("THUNDERSYNC_VWALL_CHECK" in p for p in nolanes.config_problems(c))
