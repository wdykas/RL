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
"""Mamba SSD chunk-scan dots at Triton ``tf32x3`` (``precision_raise`` site ``ssd_scan_x3``).

The chunk-scan forward (``acc = C · states``, ``acc += CB · x``) runs its fp32 ``tl.dot``s as
TF32 (10-bit operand mantissas) by default. tf32x3 splits each fp32 operand into a TF32 high
part and a TF32 residual and issues three MMAs (hi·hi + hi·lo + lo·hi), which reproduces the
IEEE result of the chunk-scan to the floor2 measurement (acceptance_floor2.md §1.3: this kernel
carries ~3/4 of the SSD share of the acceptance floor; bmm / chunk-state stay TF32).

``install()`` rewrites the source of every Triton kernel that issues the chunk-scan dots so each
``tl.dot`` call carries an explicit ``input_precision="tf32x3"``:

- Megatron's ``ssd_chunk_scan._chunk_scan_fwd_kernel`` (batch-invariant training through
  mamba_ssd_parity, engine prefill, MCore's buffered decode ``_mamba_chunk_scan_decode_rows``);
- mamba_ssm's ``_chunk_scan_fwd_kernel`` (the non-BI training path), when importable;
- ``mamba_bi_scan_row._chunk_scan_row_kernel`` (the engine's target-row decode replay).

Why a source rewrite and not ``triton.knobs.language.fp32_default``: Triton's in-memory kernel
cache key does not include that knob (a kernel compiled once under TF32 is silently reused), and
setting the knob writes ``TRITON_F32_DEFAULT`` into the environment, where restoring it to None
does not remove it (later TF32 compiles of other kernels would be keyed as tf32x3 on disk). The
rewritten source changes the kernel's own cache key (disk and memory), applies to every launch
path and thread, and leaves every other kernel untouched. Both the learner and the engine call
``install()`` before their first Mamba forward, so they compile the same kernels.

Per-rank cost: compute only (3 MMAs instead of 1 per scan product; no memory, no communication).
Heads shard over TP, so the cost per rank falls as 1/TP.
"""

from __future__ import annotations

import ast
from typing import Any

PRECISION = "tf32x3"
_STATE: dict[str, Any] = {"installed": False, "orig": []}


def _rewrite(src: str, precision: str) -> str:
    """``src`` with ``input_precision=precision`` added to every ``tl.dot(...)`` call."""
    tree = ast.parse(src)
    n = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "dot"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tl"
        ):
            kws = {k.arg for k in node.keywords}
            if "input_precision" in kws or "allow_tf32" in kws:
                raise ValueError("ssd_x3: a tl.dot already pins its precision")
            node.keywords.append(ast.keyword(arg="input_precision", value=ast.Constant(precision)))
            n += 1
    if n == 0:
        raise ValueError("ssd_x3: no tl.dot found")
    return ast.unparse(tree)


def _jit_fn(kernel: Any) -> Any:
    """The JITFunction behind an Autotuner / Heuristics wrapper (or the JITFunction itself)."""
    while hasattr(kernel, "fn") and not hasattr(kernel, "_unsafe_update_src"):
        kernel = kernel.fn
    return kernel


def _set_src(fn: Any, src: str) -> None:
    from collections import defaultdict

    fn._unsafe_update_src(src)
    # Drop compiled kernels (their in-memory key does not include the source).
    fn.device_caches = defaultdict(fn.create_binder)


def _targets() -> list:
    from megatron.core.ssm.ops.mamba2 import ssd_chunk_scan as mcore_scan

    from nemo_rl.models.megatron import mamba_bi_scan_row

    kernels = [mcore_scan._chunk_scan_fwd_kernel]
    if mamba_bi_scan_row.HAVE_TRITON:
        kernels.append(mamba_bi_scan_row._chunk_scan_row_kernel)
    try:
        from mamba_ssm.ops.triton import ssd_chunk_scan as pkg_scan

        kernels.append(pkg_scan._chunk_scan_fwd_kernel)
    except (ImportError, AttributeError):
        pass
    return [_jit_fn(k) for k in kernels]


def install(precision: str = PRECISION) -> None:
    """Chunk-scan dots at ``precision`` for this process (idempotent)."""
    if _STATE["installed"]:
        return
    for fn in _targets():
        _STATE["orig"].append((fn, fn.src))
        _set_src(fn, _rewrite(fn.src, precision))
    _STATE["installed"] = True


def uninstall() -> None:
    """Restore the TF32 kernels (tests)."""
    for fn, src in _STATE["orig"]:
        _set_src(fn, src)
    _STATE["orig"].clear()
    _STATE["installed"] = False


def installed() -> bool:
    return bool(_STATE["installed"])


def compiled_mma_counts(kernel: Any) -> list[dict[str, int]]:
    """Per compiled variant of ``kernel``: counts of TF32 MMA instructions in its PTX (evidence)."""
    import re

    fn = _jit_fn(kernel)
    out = []
    for caches in fn.device_caches.values():
        for k in caches[0].values():
            ptx = k.asm.get("ptx", "")
            out.append({
                "mma_tf32": len(re.findall(r"mma\.sync\.aligned\.\S*tf32", ptx)),
                "tcgen05_mma": len(re.findall(r"tcgen05\.mma", ptx)),
                "wgmma": len(re.findall(r"wgmma\.mma_async", ptx)),
                "cvt_rna_tf32": len(re.findall(r"cvt\.rna\.tf32", ptx)),
            })
    return out
