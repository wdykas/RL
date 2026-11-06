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
"""Deferred ClippedPG metric scalars == per-scalar .item() (every ClippedPG unit test re-run).

Each ClippedPGLossFn call inside the imported tests is evaluated twice: as is, and
under ``deferred_metric_scalars`` + ``resolve_metric_scalars``; loss bits and every
metric value must match exactly.
"""

import math

import pytest
import torch

from nemo_rl.algorithms.loss import loss_functions as lf

from tests.unit.algorithms.test_loss_functions import (  # noqa: F401
    test_clipped_pg_loss_cispo,
    test_clipped_pg_loss_cispo_with_importance_sampling_correction,
    test_clipped_pg_loss_force_on_policy_ratio,
    test_clipped_pg_loss_icepop_importance_sampling,
    test_clipped_pg_loss_kl_penalty,
    test_clipped_pg_loss_masking,
    test_clipped_pg_loss_on_policy_kl_importance_sampling,
    test_clipped_pg_loss_on_policy_truncated_importance_sampling,
    test_clipped_pg_loss_ppo_clipping,
    test_clipped_pg_loss_reinforce_mode,
    test_clipped_pg_loss_reports_is_oob_ratio_tis,
    test_clipped_pg_loss_zero_mask,
)

_CALLS = [0]


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


@pytest.fixture(autouse=True)
def _compare_deferred(monkeypatch):
    orig = lf.ClippedPGLossFn.__call__

    def both(self, *args, **kwargs):
        loss, metrics = orig(self, *args, **kwargs)
        with lf.deferred_metric_scalars():
            loss2, metrics2 = orig(self, *args, **kwargs)
        assert any(isinstance(v, torch.Tensor) for v in metrics2.values())
        lf.resolve_metric_scalars([metrics2])
        assert torch.equal(loss.detach(), loss2.detach())
        assert metrics.keys() == metrics2.keys()
        bad = {k: (metrics[k], metrics2[k]) for k in metrics if not _same(metrics[k], metrics2[k])}
        assert not bad, bad
        _CALLS[0] += 1
        return loss, metrics

    monkeypatch.setattr(lf.ClippedPGLossFn, "__call__", both)
    yield


@pytest.mark.parametrize("dev", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_resolve_metric_scalars_dtypes(dev):
    d = [{"a": torch.tensor(1.5, device=dev), "b": torch.tensor(3, device=dev), "c": 2.0},
         {"a": torch.tensor(-0.25, dtype=torch.float64, device=dev)}]
    lf.resolve_metric_scalars(d)
    assert d == [{"a": 1.5, "b": 3, "c": 2.0}, {"a": -0.25}]
    assert isinstance(d[0]["b"], int)
    _CALLS[0] += 1
