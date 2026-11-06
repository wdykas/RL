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
"""THUNDERSYNC_VERIFY_FWD_GROUP: a verify call's lockstep forwards run in groups before their row tails.

Same results bitwise for every group size (the forwards, the rows and the per-row computations are unchanged; only
the host interleaving of forwards and row tails moves), and the interleaving is the one documented: the forwards of a
group, then the tails of its rows in row order.
"""

import pytest
from nemo_rl.models.megatron import exact_nucleus
import torch
from test_engine_q import _engine_drafts, _engine_worker, _untimed

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _run(monkeypatch, group, group_gb=None, log=None):
    monkeypatch.setenv("THUNDERSYNC_VERIFY_FWD_GROUP", str(group))
    if group_gb is None:
        monkeypatch.delenv("THUNDERSYNC_VERIFY_FWD_GROUP_GB", raising=False)
    else:
        monkeypatch.setenv("THUNDERSYNC_VERIFY_FWD_GROUP_GB", str(group_gb))
    w, emb, heads = _engine_worker(monkeypatch)
    if log is not None:
        fwd_one, fwd_rows = w._training_forward, w._training_forward_rows

        def one(row):
            log.append("F")
            return fwd_one(row)

        def rows(rs):
            log.append("F")
            return fwd_rows(rs)

        w._training_forward, w._training_forward_rows = one, rows
        # The tp == 1 row tail starts with exact_nucleus.policy_log_softmax (imported inside the worker method,
        # looked up on the module at call time).
        tail = exact_nucleus.policy_log_softmax

        def logged_tail(*a, **k):
            log.append("T")
            return tail(*a, **k)

        monkeypatch.setattr(exact_nucleus, "policy_log_softmax", logged_tail)
    rows_, plens, segs, lqs, hids, keys = _engine_drafts(emb, heads, n=20)
    return _untimed(
        w.verify_drafts_block_engine(
            rows=rows_,
            prompt_lens=plens,
            segments=segs,
            engine_logq=lqs,
            hidden=hids,
            vocab_limit=1 << 30,
            seed=3,
            keys=keys,
            batch_tokens=24,  # several rows per forward and several forwards per call
        )
    )


@cuda
@pytest.mark.parametrize("group,group_gb", [(2, None), (3, None), (100, None), (100, 1e-9)])
def test_grouped_forwards_give_the_same_results_bitwise(monkeypatch, group, group_gb):
    ref = _run(monkeypatch, 1)
    got = _run(monkeypatch, group, group_gb)
    assert got == ref


@cuda
@pytest.mark.parametrize("group", [1, 3])
def test_grouped_forwards_run_before_their_tails(monkeypatch, group):
    log: list = []
    _run(monkeypatch, group, log=log)
    seq = "".join(log)
    assert seq.count("F") >= 4, seq  # the call has several forwards
    # Split into runs: each forward run has at most `group` forwards, and every tail run follows a forward run.
    runs = []
    for c in seq:
        if runs and runs[-1][0] == c:
            runs[-1][1] += 1
        else:
            runs.append([c, 1])
    assert runs[0][0] == "F"
    for c, n in runs:
        if c == "F":
            assert n <= group, seq
    if group > 1:
        assert any(c == "F" and n > 1 for c, n in runs), seq  # grouping is not vacuous


@cuda
def test_group_byte_cap_flushes_after_every_forward(monkeypatch):
    log: list = []
    _run(monkeypatch, 100, group_gb=1e-9, log=log)
    seq = "".join(log)
    assert "FF" not in seq, seq  # a cap below one forward's logits = ungrouped interleaving
