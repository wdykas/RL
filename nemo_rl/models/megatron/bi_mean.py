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
"""Full-reduction fix for Megatron-Core's batch-invariant ``aten::mean.dim`` override.

``enable_batch_invariant_mode`` registers ``mean_batch_invariant`` for ``aten::mean.dim``.
A full reduction (``x.mean()``, ``torch.mean(x)``, ``x.mean(dim=[])``) reaches it with an
empty ``dim``, and the pinned Megatron-Core divides the sum by the product over that empty
list (1): the forward value is the SUM. ``dim=None`` raises. Autograd is unaffected (the
``mean`` derivative divides by ``numel`` itself), but every value read from a full
``.mean()`` in a batch-invariant process is wrong.

``install()`` replaces the module's ``mean_batch_invariant`` with a wrapper that maps an
empty or ``None`` ``dim`` to every dimension, before ``enable_batch_invariant_mode``
registers it. This is the Megatron-LM change ``fix/batch-invariant-mean-full-reduction``
applied to the Megatron-Core this repo pins; it is a no-op on a Megatron-Core that already
has the fix.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

_MCORE_MEAN: Optional[Callable[..., Any]] = None


def mean_batch_invariant(input, dim, keepdim=False, dtype=None):
    """MCore ``mean_batch_invariant`` with an empty / ``None`` ``dim`` meaning all dims."""
    if dim is None or len(dim) == 0:
        dim = list(range(input.ndim))
    return _MCORE_MEAN(input, dim, keepdim=keepdim, dtype=dtype)


mean_batch_invariant._nemo_rl_full_reduction_fix = True  # type: ignore[attr-defined]


def mcore_full_mean_is_correct(fn: Callable[..., Any]) -> bool:
    """True if ``fn(x, [])`` is the mean of ``x`` (checked on CPU, no kernel launch)."""
    import torch

    return float(fn(torch.ones(4), [])) == 1.0


def install() -> None:
    """Patch MCore's ``mean_batch_invariant`` (call before ``enable_batch_invariant_mode``)."""
    global _MCORE_MEAN
    # Deferred: Megatron-Core exists only in worker venvs.
    from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik

    current = bik.mean_batch_invariant
    if getattr(current, "_nemo_rl_full_reduction_fix", False):
        return
    if mcore_full_mean_is_correct(current):
        return  # this Megatron-Core already has the fix
    if bik.is_batch_invariant_mode_enabled():
        raise RuntimeError(
            "bi_mean.install() must run before enable_batch_invariant_mode(): the "
            "registered aten::mean.dim kernel would keep the full-reduction bug."
        )
    _MCORE_MEAN = current
    bik.mean_batch_invariant = mean_batch_invariant
