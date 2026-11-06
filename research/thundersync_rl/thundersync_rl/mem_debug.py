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
"""Debug-only CUDA memory history around learner calls (``THUNDERSYNC_MEM_HISTORY=<dir>``).

Records the allocator history (``torch.cuda.memory._record_memory_history``) and
writes ``<dir>/rank<r>_peak.pickle`` whenever a wrapped call sets a new peak, and
``<dir>/rank<r>_oom.pickle`` when one raises a CUDA OOM. Inspect with
``scripts/mem_snapshot_report.py``. Off (zero cost) unless the variable is set.
"""

from __future__ import annotations

import contextlib
import os
from typing import Iterator

import torch

_STATE = {"started": False, "best": 0}


@contextlib.contextmanager
def capture(rank: int, where: str) -> Iterator[None]:
    """Record memory history over the block; dump on a new peak or an OOM."""
    out = os.environ.get("THUNDERSYNC_MEM_HISTORY")
    if not out:
        yield
        return
    os.makedirs(out, exist_ok=True)
    if not _STATE["started"]:
        torch.cuda.memory._record_memory_history(max_entries=400_000, stacks="python")
        _STATE["started"] = True
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    try:
        yield
    except torch.OutOfMemoryError:
        torch.cuda.memory._dump_snapshot(os.path.join(out, f"rank{rank}_oom.pickle"))
        raise
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    print(
        f"[thundersync memhist] rank={rank} {where}: base={base / 2**30:.2f}G "
        f"peak={peak / 2**30:.2f}G transient={(peak - base) / 2**30:.2f}G",
        flush=True,
    )
    if peak > _STATE["best"]:
        _STATE["best"] = peak
        torch.cuda.memory._dump_snapshot(os.path.join(out, f"rank{rank}_peak.pickle"))
