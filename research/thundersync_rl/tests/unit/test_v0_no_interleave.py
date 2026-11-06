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
"""THUNDERSYNC_V0_NO_INTERLEAVE: a learner rank refuses a verify0 (⊥-BV) call after a training dispatch of the same
step (verify0_stream with a window covering every call must still run all of verify0 before any training)."""

from types import SimpleNamespace

import pytest

from thundersync_rl.worker import _check_v0_no_interleave

V0 = {"null_envelopes": [[1.0, 1.5]]}
PIECE = {"null_envelopes": None}


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("THUNDERSYNC_V0_NO_INTERLEAVE", raising=False)
    w = SimpleNamespace()
    _check_v0_no_interleave(w, "stream_accumulate", {})
    _check_v0_no_interleave(w, "verify_drafts_block_engine", V0)  # no check, no raise


def test_verify0_then_training_passes_and_training_then_verify0_raises(monkeypatch):
    monkeypatch.setenv("THUNDERSYNC_V0_NO_INTERLEAVE", "1")
    w = SimpleNamespace(rank=0)
    for _ in range(3):
        _check_v0_no_interleave(w, "verify_drafts_block_engine", V0)
    _check_v0_no_interleave(w, "stream_accumulate", {})
    _check_v0_no_interleave(w, "verify_drafts_block_engine", PIECE)  # continuation pieces may interleave
    with pytest.raises(RuntimeError, match="verify0 interleave"):
        _check_v0_no_interleave(w, "verify_drafts_block_engine", V0)
    w._v0ni_trained = False  # stream_finish_step resets it
    _check_v0_no_interleave(w, "verify_drafts_block_engine", V0)
    assert w._v0ni_calls == 4
