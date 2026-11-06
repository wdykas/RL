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
"""acc_dump (THUNDERSYNC_ACC_DUMP research hook): off by default; statistics; one file per dumped row."""

import os

import torch

from thundersync_rl import acc_dump


def test_off_by_default_picks_nothing(monkeypatch):
    monkeypatch.setattr(acc_dump, "DIR", None)
    assert not any(acc_dump.picked(k) for k in range(64))


def test_every_selects_a_fixed_key_subset(monkeypatch, tmp_path):
    monkeypatch.setattr(acc_dump, "DIR", str(tmp_path))
    monkeypatch.setattr(acc_dump, "EVERY", 4)
    picked = [k for k in range(400) if acc_dump.picked(k)]
    assert 50 < len(picked) < 150 and picked == [k for k in range(400) if acc_dump.picked(k)]


def test_identical_rows_give_zero_divergence_and_full_agreement():
    torch.manual_seed(0)
    lp = torch.log_softmax(torch.randn(5, 64, dtype=torch.float64), -1)
    tok = torch.randint(0, 64, (5,))
    s = acc_dump.chunk_stats(lp, lp.clone(), tok, topk=64)
    col = {n: s[:, i] for i, n in enumerate(acc_dump.NAMES)}
    assert torch.allclose(col["tv"], torch.zeros(5), atol=1e-6)
    assert torch.allclose(col["kl_qp"], torch.zeros(5), atol=1e-6)
    assert torch.allclose(col["agree"], torch.ones(5), atol=1e-5)
    assert torch.equal(col["same_argmax"], torch.ones(5))
    assert torch.allclose(col["lp_x"], lp.gather(-1, tok[:, None])[:, 0].float())


def test_dump_row_writes_one_file(monkeypatch, tmp_path):
    monkeypatch.setattr(acc_dump, "DIR", str(tmp_path))
    torch.manual_seed(1)
    p = torch.log_softmax(torch.randn(7, 32), -1)
    q = torch.log_softmax(torch.randn(7, 32), -1)
    draft = torch.randint(0, 32, (7,))
    acc_dump.dump_row(
        0, 3, "key-1", [(0, 7)], p, lambda idx: q[idx], draft,
        q.gather(-1, draft[:, None])[:, 0], p.gather(-1, draft[:, None])[:, 0],
    )
    files = os.listdir(tmp_path)
    assert len(files) == 1 and files[0].startswith("v3_r0_")
    rec = torch.load(os.path.join(tmp_path, files[0]))
    assert rec["g"] == 7 and rec["stats"].shape == (7, len(acc_dump.NAMES))
