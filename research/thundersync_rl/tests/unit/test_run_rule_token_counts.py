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
"""Run rule 2 (layout audit): tokens dealt == the DP-reduced loss denominator (+ tokens taken back), and the
TP / PP / CP peers of each data-parallel replica agree on their counts."""

import copy

import torch

from thundersync_rl.run_checks import step_counts_merge
from thundersync_rl.streaming import row_audit


def _ranks(dp=2, tp=2, toks=(300, 200), seqs=(3, 2), n_mb=4, uncounted=(0, 0)):
    """Rank results of a TP x DP learner: rank = dp * tp + t (tp rank 0 owns the replica's counts)."""
    out = []
    g_toks, g_seqs = float(sum(toks)), float(sum(seqs))
    for d in range(dp):
        for t in range(tp):
            out.append(
                {
                    "stream_dp_count_owner": t == 0,
                    "stream_step_counts": {
                        "dp": d,
                        "local_seqs": float(seqs[d]),
                        "local_toks": float(toks[d]),
                        "num_microbatches": n_mb,
                        "global_seqs": g_seqs,
                        "global_toks": g_toks,
                        "trained_toks": g_toks,
                        "uncounted_toks": float(uncounted[d]),
                    },
                }
            )
    return out


def test_consistent_counts_pass():
    m = step_counts_merge(_ranks(), 500)
    assert m["violations"] == [] and m["global_valid_toks"] == 500 and m["tokens_uncounted"] == 0


def test_tokens_taken_back_by_vit_are_accounted():
    r = _ranks(toks=(250, 200), uncounted=(50, 0))
    assert step_counts_merge(r, 500)["violations"] == []
    assert step_counts_merge(_ranks(toks=(250, 200)), 500)["violations"]  # same deficit, not accounted


def test_tokens_dealt_but_not_in_the_denominator_fire():
    v = step_counts_merge(_ranks(), 640)["violations"]
    assert any("tokens dealt 640 != global_valid_toks 500" in x for x in v)


def test_dp_reduction_over_the_wrong_group_fires():
    # The all-reduce summed over TP peers too (or missed a replica): global != sum of the replicas' local counts.
    r = _ranks()
    for x in r:
        x["stream_step_counts"]["global_toks"] = 1000.0
    v = step_counts_merge(r, 1000)["violations"]
    assert any("sum of the DP replicas' local tokens 500" in x for x in v)


def test_peers_disagree_fire():
    r = _ranks()
    r[1]["stream_step_counts"]["num_microbatches"] = 5  # TP peer of replica 0 ran another micro-batch
    v = step_counts_merge(r, 500)["violations"]
    assert any("DP replica 0" in x for x in v)
    r = _ranks()
    r[3]["stream_step_counts"]["global_toks"] = 499.0
    v = step_counts_merge(r, 500)["violations"]
    assert any("disagree on (global_valid_seqs, global_valid_toks)" in x for x in v)


def test_trained_above_global_and_missing_rank_fire():
    r = _ranks()
    for x in r:
        x["stream_step_counts"]["trained_toks"] = 501.0
    assert any("trained_valid_toks 501" in x for x in step_counts_merge(r, 500)["violations"])
    r = _ranks()
    del r[2]["stream_step_counts"]
    assert any("sent no step counts" in x for x in step_counts_merge(r, 500)["violations"])
    assert step_counts_merge([{"x": 1}], 500) is None  # older workers: no data, no verdict


def test_row_audit_tokens_match_the_loss_count():
    # The driver's token count (row_audit) is the count _prepare_train_piece adds to local_valid_toks.
    g = torch.Generator().manual_seed(0)
    tm = (torch.rand(6, 9, generator=g) > 0.3).float()
    sm = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    data = {"sample_mask": sm, "token_mask": tm, "input_ids": torch.zeros(6, 9, dtype=torch.long)}
    local = float(torch.sum(tm[:, 1:] * sm.unsqueeze(-1)))
    assert row_audit(copy.deepcopy(data))[1] == local


def test_rows_taken_back_are_reported():
    r = _ranks(toks=(250, 200), uncounted=(50, 0))
    for x in r:
        x["stream_step_counts"]["uncounted_rows"] = 1.0
    # Owners only (one rank per replica): 2 replicas x 1 row.
    assert step_counts_merge(r, 500)["rows_uncounted"] == 2.0


def test_count_uncounted_uses_the_row_audit_counts():
    from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker

    w = object.__new__(ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)
    tm = torch.ones(3, 5)
    items = [{"data": {"sample_mask": torch.tensor([1.0, 1.0, 0.0]), "token_mask": tm}}]
    w._count_uncounted(items)
    w._count_uncounted(iter(items))
    assert (w._rows_uncounted, w._toks_uncounted) == (4, 16)
