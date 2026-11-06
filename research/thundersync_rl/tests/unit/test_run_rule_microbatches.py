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
"""Run rule 11 (layout audit): num_microbatches * mbs == data.size is a raise (eager, kept under python -O)."""

import subprocess
import sys

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _bdd(n):
    return BatchedDataDict({"input_ids": torch.arange(n * 3).view(n, 3), "sample_mask": torch.ones(n)})


def test_whole_microbatches_iterate_every_row():
    mbs = list(_bdd(8).make_microbatch_iterator(4))
    assert len(mbs) == 2 and sum(m.size for m in mbs) == 8


@pytest.mark.parametrize("n,mbs", [(7, 4), (3, 2), (4, 0)])
def test_partial_microbatch_raises_eagerly(n, mbs):
    with pytest.raises(ValueError, match="not a multiple of the provided microbatch size"):
        _bdd(n).make_microbatch_iterator(mbs)  # no next(): the check is not deferred to first use


def test_raise_survives_python_O():
    code = (
        "import torch\n"
        "from nemo_rl.distributed.batched_data_dict import BatchedDataDict\n"
        "d = BatchedDataDict({'x': torch.zeros(7, 2)})\n"
        "try:\n"
        "    list(d.make_microbatch_iterator(4))\n"
        "except ValueError:\n"
        "    raise SystemExit(3)\n"
    )
    assert subprocess.run([sys.executable, "-O", "-c", code]).returncode == 3
