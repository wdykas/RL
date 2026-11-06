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
"""MoE token dispatchers drop their routing-probability reference once the combine is done.

``MoEAlltoAllTokenDispatcher.dispatch_preprocess`` keeps ``self.probs`` (and the AllGather
dispatcher ``self.local_probs``) for the combine. Nothing reads it after
``combine_postprocess``, but the attribute lives until the layer's next forward. With the
MoE activation recompute (``recompute_modules: [moe]``) the backward re-runs the MoE forward
under autograd, so the kept probs carry a ``grad_fn`` whose graph ends at the recompute's
detached input (the pre-MLP norm output, fp32 [T, d]) and that input's ``.grad`` (another
[T, d]): two token-by-hidden fp32 tensors per MoE layer stay allocated after every training
call, until the next forward reaches that layer. At Nemotron-3-Nano (23 MoE layers, d 2688)
that is ~8 GiB at a 17k-token micro-batch: learner memory between calls (checkpoint save,
log-prob checks) that nothing uses. Measured at Nano-L8 (3 MoE layers): 1.02 GiB held after
the call, found through ``MoEAlltoAllTokenDispatcher.probs`` (ScatterBackward0).

``install()`` wraps ``combine_postprocess`` to clear the reference after the original runs.
Values are unchanged (the attribute is not read afterwards); stock NeMo-RL drops the same
attributes only before a colocated refit (``_clear_rope_and_moe_dispatcher_caches``).
"""

from __future__ import annotations

_PATCHED: list[bool] = []

# Dispatcher class name -> attributes set in dispatch_preprocess and not read after the combine.
_ATTRS = {
    "MoEAlltoAllTokenDispatcher": ("probs",),
    "MoEAllGatherTokenDispatcher": ("local_probs",),
}


def install() -> None:
    """Patch the dispatchers' ``combine_postprocess`` (idempotent)."""
    if _PATCHED:
        return
    from megatron.core.transformer.moe import token_dispatcher as td

    for name, attrs in _ATTRS.items():
        cls = getattr(td, name, None)
        if cls is None:
            continue
        orig = cls.combine_postprocess

        def combine_postprocess(self, *args, _orig=orig, _attrs=attrs, **kwargs):
            out = _orig(self, *args, **kwargs)
            for attr in _attrs:
                if getattr(self, attr, None) is not None:
                    setattr(self, attr, None)
            return out

        combine_postprocess.__wrapped__ = orig  # type: ignore[attr-defined]
        cls.combine_postprocess = combine_postprocess
    _PATCHED.append(True)
