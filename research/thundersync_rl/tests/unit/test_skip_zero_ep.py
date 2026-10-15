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
"""CPU tests: zero-advantage skipping under expert parallelism with dealt rows (DESIGN 14x).

The planner marks all-zero-advantage final chunks count-only and pads the remaining
(live) forwards to equal per-rank counts, so every EP rank runs the same number of
forwards; the worker counts a count-only chunk's tokens for normalization only.
"""

import pytest
import torch
from test_deal_rows import _per_rank_mbs, _run_canonical
from thundersync_rl.streaming import microbatch_counts
from thundersync_rl.worker import count_only_chunk


def _live_forwards(pl_skip, d, mbs):
    return [
        sum(c.num_microbatches(mbs) for c in cs if not (c.zero_advantage and not c.padding and c.fused is None) or not pl_skip)
        for cs in d.per_rank
    ]


@pytest.mark.parametrize("dp,G,groups,bg", [(8, 8, 64, 4), (8, 8, 64, 2), (8, 16, 32, 4), (64, 8, 64, 8), (2, 8, 16, 4)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_equal_live_forwards_and_count_only_marks(dp, G, groups, bg, seed):
    mbs = 1
    out = _run_canonical(dp, mbs, G, groups, bg, True, seed, skip_zero=True)
    rows_live = rows_zero = 0
    for d in out:
        live = [
            sum(c.num_microbatches(mbs) for c in cs if not (c.zero_advantage and not c.padding))
            for cs in d.per_rank
        ]
        assert len(set(live)) == 1 and live[0] >= 1, live  # equal forwards per rank, >= 1
        for cs in d.per_rank:
            for c in cs:
                if c.padding:
                    assert c.advantages == [0.0]
                elif c.zero_advantage:
                    rows_zero += len(c.trajectories)
                else:
                    assert any(a != 0.0 for a in c.advantages)  # live chunks never mix in zero rows
                    rows_live += len(c.trajectories)
        mb = microbatch_counts(d, mbs, lambda c: c.zero_advantage and not c.padding)
        assert mb["per_rank"] == live[0]
        # Fillers only top up the live count: fewer than DP per dispatch
        # (plus at most one per rank when every chunk of the dispatch is zero).
        assert mb["filler"] < dp or mb["real"] == 0
    assert rows_live + rows_zero == groups * G


def test_skipping_cuts_forwards_vs_training_every_row():
    dp, mbs, G, groups, bg = 8, 1, 8, 64, 4
    on = _run_canonical(dp, mbs, G, groups, bg, True, 3, skip_zero=True)
    off = _run_canonical(dp, mbs, G, groups, bg, True, 3, skip_zero=False)
    fwd_on = sum(max(_live_forwards(True, d, mbs)) for d in on)
    fwd_off = sum(max(_per_rank_mbs(d, mbs)) for d in off)
    assert fwd_off == 64  # 512 rows / DP 8, no filler (deal_rows)
    assert fwd_on < fwd_off


def test_all_zero_dispatch_runs_one_filler_per_rank():
    dp, mbs, G, groups = 8, 1, 8, 4
    out = _run_canonical(dp, mbs, G, groups, 4, True, 0, skip_zero=True, adv=lambda rs: [0.0] * len(rs))
    assert len(out) == 1
    d = out[0]
    for cs in d.per_rank:
        assert sum(1 for c in cs if c.padding) == 1
        assert all(c.zero_advantage for c in cs)


def _item(adv, sample=(1.0, 1.0), group=None):
    n, s = len(sample), 6
    return {
        "data": {
            "sample_mask": torch.tensor(sample),
            "token_mask": torch.ones(n, s),
            "advantages": torch.tensor(adv, dtype=torch.float32).unsqueeze(-1).expand(n, s).clone(),
        },
        "group": group,
        "fused_key": None,
        "count_only": True,
    }


def _state():
    z = lambda: torch.zeros((), dtype=torch.float64)
    return {"num_chunks": 0, "local_valid_seqs": z(), "local_valid_toks": z()}


def test_worker_counts_like_a_trained_chunk():
    st = _state()
    count_only_chunk(st, _item([0.0, 0.0], sample=(1.0, 0.0)))
    assert st["num_chunks"] == 1
    assert float(st["local_valid_seqs"]) == 1.0
    assert float(st["local_valid_toks"]) == 5.0  # token_mask[:, 1:] of the live row


def test_worker_rejects_nonzero_advantage_and_buckets():
    with pytest.raises(RuntimeError, match="nonzero advantage"):
        count_only_chunk(_state(), _item([0.0, 0.5]))
    with pytest.raises(RuntimeError, match="final, unfused"):
        count_only_chunk(_state(), _item([0.0, 0.0], group=(0, 1)))
