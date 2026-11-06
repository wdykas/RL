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
"""Logit-free verification inputs: LM-head statistics from final hidden states.

Instead of [T, V] fp32 logits, the verifier keeps the final hidden states
[T, d] and computes per-position statistics (logsumexp and the logit of one
chosen token) one row chunk at a time. Full logits rows are recomputed only
where the lazy block-verification scan asks for them.

A row is only ever computed as part of its fixed chunk [c*C, (c+1)*C), through
the same head call, so a recomputed row is bitwise the row its statistics came
from (same GEMM shape and inputs; cuBLAS is deterministic for a fixed shape).
"""

from __future__ import annotations

from typing import Callable

import torch


class ChunkedHead:
    """Logits rows of ``hidden`` [n, d] under ``logits_fn(h, weight) -> [m, V]``.

    ``weight()`` returns the head weight; it may be swapped for a different
    tensor holding the same values (e.g. a stashed copy, once the scorer has
    loaded other weights) between ``stats`` and ``row``.
    """

    def __init__(
        self,
        hidden: torch.Tensor,
        logits_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        weight: Callable[[], torch.Tensor],
        chunk: int = 512,
    ):
        self.hidden = hidden
        self.logits_fn = logits_fn
        self.weight = weight
        self.chunk = chunk
        self.pinned: dict[int, torch.Tensor] = {}
        self._cached: tuple[int, torch.Tensor] | None = None
        self._w: torch.Tensor | None = None

    def __len__(self) -> int:
        return self.hidden.shape[0]

    def _logits(self, c: int) -> torch.Tensor:
        if self._cached is not None and self._cached[0] == c:
            return self._cached[1]
        if self._w is None:
            self._w = self.weight()
        a = c * self.chunk
        z = self.logits_fn(self.hidden[a : a + self.chunk], self._w).float()
        self._cached = (c, z)
        return z

    def stats(
        self, tokens: torch.Tensor, pin: list[int] | tuple[int, ...] = ()
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(logsumexp [n], logit of ``tokens[i]`` at row i [n]), both fp32.

        Rows listed in ``pin`` are kept (e.g. bonus positions, always sampled).
        """
        n = len(self)
        lse = torch.empty(n, dtype=torch.float32, device=self.hidden.device)
        picked = torch.empty_like(lse)
        pins = sorted(set(pin))
        for c in range((n + self.chunk - 1) // self.chunk):
            a, e = c * self.chunk, min(n, (c + 1) * self.chunk)
            z = self._logits(c)
            lse[a:e] = torch.logsumexp(z, -1)
            picked[a:e] = z.gather(1, tokens[a:e, None])[:, 0]
            for i in pins:
                if a <= i < e:
                    self.pinned[i] = z[i - a].clone()
        self.release()
        return lse, picked

    def row(self, i: int) -> torch.Tensor:
        """Logits row i [V] (fp32), recomputing its chunk unless pinned or cached."""
        if i in self.pinned:
            return self.pinned[i]
        return self._logits(i // self.chunk)[i % self.chunk]

    def rows(self, idx: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.row(i) for i in idx.tolist()])

    def release(self) -> None:
        """Drop the cached chunk and weight (pinned rows stay)."""
        self._cached = None
        self._w = None
