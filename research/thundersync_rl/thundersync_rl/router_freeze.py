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
"""The MoE router is always frozen (user rule 11-02): gate weights non-trainable and the expert bias never updated.

``router_fingerprint`` reads, on one learner rank, every MCore ``Router``'s gate ``weight``, ``bias`` (if any) and
``expert_bias`` buffer, and returns an integer fingerprint of their raw bits plus the names of any router Parameter
that is trainable. The learner computes it at the end of every optimizer step (it rides on finish_train_step's result,
no extra call) and the stage requires it equal to the first step's on every rank.

Fingerprint per tensor, viewed as [rows, cols] of its raw bit patterns (int16 / int32 / int64 by element size):
  - the int64 sum of every row's sum, weighted by (row + 1), and
  - the int64 sum of every column's sum, weighted by (col + 1).
A changed element changes one row sum and one column sum, so it changes both values unless the change cancels in both
weighted sums at once. Transient memory: rows + cols int64 per tensor (E + h, e.g. 512 + 8192 at Ultra), no copy of
the tensor; one read of the router bytes per step (h x E x 4 B per MoE layer, e.g. 16 MiB at Ultra).
"""

from __future__ import annotations

import zlib
from typing import Any, Iterable

import torch

_INT = {2: torch.int16, 4: torch.int32, 8: torch.int64}


def _tensor_fp(t: torch.Tensor) -> tuple[int, int]:
    v = t.detach()
    if v.dim() == 0:
        v = v.reshape(1)
    v = v.contiguous().view(_INT[v.element_size()])
    m = v.reshape(v.shape[0], -1) if v.dim() > 1 else v.reshape(1, -1)
    rows = m.sum(dim=1, dtype=torch.int64)
    cols = m.sum(dim=0, dtype=torch.int64)
    rw = torch.arange(1, rows.numel() + 1, device=rows.device, dtype=torch.int64)
    cw = torch.arange(1, cols.numel() + 1, device=cols.device, dtype=torch.int64)
    return int((rows * rw).sum()), int((cols * cw).sum())


def router_tensors(models: Any) -> tuple[list[tuple[str, torch.Tensor]], list[str]]:
    """(name, tensor) of every Router's weight / bias / expert_bias, and the names of trainable router Parameters."""
    from megatron.core.transformer.moe.router import Router

    models = models if isinstance(models, (list, tuple)) else [models]
    tensors: list[tuple[str, torch.Tensor]] = []
    trainable: list[str] = []
    for mi, mod in enumerate(models):
        for name, m in mod.named_modules():
            if not isinstance(m, Router):
                continue
            for pn in ("weight", "bias"):
                t = getattr(m, pn, None)
                if isinstance(t, torch.nn.Parameter):
                    tensors.append((f"{mi}.{name}.{pn}", t))
                    if t.requires_grad:
                        trainable.append(f"{mi}.{name}.{pn}")
            eb = getattr(m, "expert_bias", None)
            if torch.is_tensor(eb):
                tensors.append((f"{mi}.{name}.expert_bias", eb))
    return tensors, trainable


@torch.no_grad()
def router_fingerprint(models: Any) -> dict[str, Any]:
    tensors, trainable = router_tensors(models)
    fps = [_tensor_fp(t) for _, t in tensors]
    return {"tensors": len(tensors), "trainable": trainable, "fp": zlib.crc32(repr(fps).encode())}


def check_router_reports(reports: Iterable[dict], first: Any) -> tuple[bool, Any, int, list[str]]:
    """(ok, key, tensors, trainable) of one step's per-rank reports against the first step's key (None: this is it)."""
    reports = list(reports)
    key = [(int(r["tensors"]), int(r["fp"])) for r in reports]
    trainable = sorted({n for r in reports for n in r["trainable"]})
    n = sum(k[0] for k in key)
    ref = key if first is None else first
    return (key == ref and not trainable and n > 0), key, n, trainable
