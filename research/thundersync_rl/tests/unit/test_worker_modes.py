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
"""Verification must not leave the policy model in eval mode mid train step."""

import pytest
import torch

from thundersync_rl.worker import _model_in_eval


class _Holder:
    def __init__(self, training: bool):
        self.model = torch.nn.Dropout(0.5)
        self.model.train(training)
        self.seen = None

    @_model_in_eval
    def verify(self, fail: bool = False):
        self.seen = self.model.training
        if fail:
            raise RuntimeError("boom")
        return "ok"


@pytest.mark.parametrize("training", [True, False])
def test_mode_restored(training):
    h = _Holder(training)
    assert h.verify() == "ok"
    assert h.seen is False
    assert h.model.training is training


def test_mode_restored_on_error():
    h = _Holder(True)
    with pytest.raises(RuntimeError):
        h.verify(fail=True)
    assert h.model.training is True
