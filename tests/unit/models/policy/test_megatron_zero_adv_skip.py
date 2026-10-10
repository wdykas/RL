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
"""loss_fn.skip_zero_advantage_rows on the Megatron split train API."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("megatron.bridge")

from nemo_rl.algorithms.loss.loss_functions import (  # noqa: E402
    ClippedPGLossConfig,
    ClippedPGLossFn,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict  # noqa: E402
from tests.unit.models.policy.test_megatron_split_state import (  # noqa: E402,F401
    _make_worker,
    mock_module_symbols,
)

pytestmark = pytest.mark.mcore


def _skip_loss() -> ClippedPGLossFn:
    return ClippedPGLossFn(
        ClippedPGLossConfig(
            reference_policy_kl_penalty=0.0,
            force_on_policy_ratio=True,
            skip_zero_advantage_rows=True,
        )
    )


def _worker():
    from nemo_rl.algorithms.loss.interfaces import LossType

    w = _make_worker(LossType.TOKEN_LEVEL)
    w.cfg["megatron_cfg"]["moe_router_load_balancing_type"] = "none"
    w.cfg["sequence_packing"] = {"enabled": False}
    w.cfg["dynamic_batching"] = {"enabled": False}
    return w


def _batch(live_rows: list[int], n: int = 8, toks: int = 256) -> BatchedDataDict:
    adv = torch.zeros(n, toks + 1)
    adv[live_rows] = 1.0
    return BatchedDataDict(
        {
            "sample_mask": torch.ones(n),
            "token_mask": torch.ones(n, toks + 1),
            "input_ids": torch.arange(n).unsqueeze(-1).expand(n, toks + 1).clone(),
            "advantages": adv,
        }
    )


def _trained_rows(mock_module_symbols) -> list[list[int]]:  # noqa: F811
    return [
        call.args[0]["input_ids"][:, 0].tolist()
        for call in mock_module_symbols["gmi"].call_args_list
    ]


def test_only_live_rows_are_trained_but_all_rows_are_counted(mock_module_symbols):  # noqa: F811
    w = _worker()
    w.begin_train_step(loss_fn=_skip_loss())
    w.train_microbatch(_batch([0, 2, 4, 7]))
    assert _trained_rows(mock_module_symbols) == [[0, 2, 4, 7]]
    state = w._train_step_state
    assert float(state["local_valid_toks"]) == pytest.approx(8 * 256)
    assert float(state["trained_valid_toks"]) == pytest.approx(4 * 256)


def test_kept_rows_are_topped_up_to_the_microbatch_size(mock_module_symbols):  # noqa: F811
    w = _worker()  # train_micro_batch_size = 4
    w.begin_train_step(loss_fn=_skip_loss())
    w.train_microbatch(_batch([1, 5, 6]))
    # 3 live rows + the first zero-advantage row, in batch order.
    assert _trained_rows(mock_module_symbols) == [[0, 1, 5, 6]]


def test_all_zero_chunk_runs_once_per_step_then_skips(mock_module_symbols):  # noqa: F811
    w = _worker()
    w.begin_train_step(loss_fn=_skip_loss())
    w.train_microbatch(_batch([]))  # first chunk: one microbatch keeps metrics
    w.train_microbatch(_batch([]))  # later all-zero chunks run nothing
    assert _trained_rows(mock_module_symbols) == [[0, 1, 2, 3]]
    assert float(w._train_step_state["local_valid_toks"]) == pytest.approx(16 * 256)


def test_without_flag_every_row_is_trained(mock_module_symbols):  # noqa: F811
    w = _worker()
    loss = ClippedPGLossFn(
        ClippedPGLossConfig(reference_policy_kl_penalty=0.0, force_on_policy_ratio=True)
    )
    w.begin_train_step(loss_fn=loss)
    w.train_microbatch(_batch([0]))
    assert _trained_rows(mock_module_symbols) == [list(range(8))]


@pytest.mark.parametrize(
    "override",
    [{"reference_policy_kl_penalty": 0.01}, {"positive_example_nll_weight": 0.1}],
)
def test_loss_rejects_non_advantage_weighted_terms(override):
    cfg = {
        "reference_policy_kl_penalty": 0.0,
        "skip_zero_advantage_rows": True,
        **override,
    }
    with pytest.raises(ValueError, match="advantage-weighted"):
        ClippedPGLossFn(ClippedPGLossConfig(**cfg))


def test_worker_rejects_sequence_packing(mock_module_symbols):  # noqa: F811
    w = _worker()
    w.cfg["sequence_packing"] = {"enabled": True}
    with pytest.raises(ValueError, match="fixed-size"):
        w.begin_train_step(loss_fn=_skip_loss())
