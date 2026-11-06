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
"""THUNDERSYNC_DEBUG_VERIFY_NO_BI (negative control): the wrapped verification runs outside batch-invariant mode
only when the flag is on; off by default the call is untouched."""

import contextlib
import sys
import types

import pytest

from thundersync_rl import debug_hooks


@pytest.fixture
def fake_bi(monkeypatch):
    calls = []

    @contextlib.contextmanager
    def set_batch_invariant_mode(enabled=True, backend=None):
        calls.append(("enter", enabled))
        yield
        calls.append(("exit", enabled))

    name = "megatron.core.transformer.custom_layers.batch_invariant_kernels"
    mod = types.ModuleType(name)
    mod.set_batch_invariant_mode = set_batch_invariant_mode
    monkeypatch.setitem(sys.modules, name, mod)
    return calls


def test_off_by_default(monkeypatch, fake_bi):
    monkeypatch.delenv(debug_hooks.VERIFY_NO_BI_ENV, raising=False)
    f = debug_hooks.verify_no_bi(lambda x, y=1: (x, y))
    assert f(3, y=4) == (3, 4)
    monkeypatch.setenv(debug_hooks.VERIFY_NO_BI_ENV, "0")
    assert f(5) == (5, 1)
    assert fake_bi == []


def test_on_runs_outside_bi_mode(monkeypatch, fake_bi):
    monkeypatch.setenv(debug_hooks.VERIFY_NO_BI_ENV, "1")
    seen = []
    f = debug_hooks.verify_no_bi(lambda: seen.append(list(fake_bi)) or "r")
    assert f() == "r"
    assert seen == [[("enter", False)]]
    assert fake_bi == [("enter", False), ("exit", False)]


def test_wrapper_keeps_the_name():
    def verify_drafts_block_engine():
        return None

    assert debug_hooks.verify_no_bi(verify_drafts_block_engine).__name__ == "verify_drafts_block_engine"
