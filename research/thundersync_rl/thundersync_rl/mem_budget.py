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
"""Training micro-batch token cap validated against the learner's device memory.

The learner's peak allocation grows linearly in the tokens of its largest training
micro-batch (activations saved for backward, the MoE routed-token buffers):

    peak(T) = fixed_gb + T * bytes_per_token / 2**30

A step OOMs once ``peak(T) + nontorch_gb`` exceeds the device. The largest
micro-batch that fits with ``margin_gb`` to spare is therefore

    fit_tokens = floor((device_gb - nontorch_gb - margin_gb - fixed_gb) * 2**30 / bytes_per_token)

and the configured cap (rows per micro-batch x policy.max_total_sequence_length,
``grpo_loop.train_micro_batch_rows``) must not exceed it. ``fixed_gb``,
``bytes_per_token`` and ``nontorch_gb`` are measured for a model / layout
(``learner/mem_peak_gb`` against ``learner/max_mb_tokens``, ``learner/mem_nontorch_gb``
after the first checkpoint save); ``device_gb`` is read from the learner ranks at
startup (smallest rank). Off unless ``thundersync.train_mem_bytes_per_token`` is set.
"""

from __future__ import annotations

from typing import Any, Optional

GiB = float(2**30)
_STATE: dict[str, float] = {}


def fit_tokens(device_gb: float, nontorch_gb: float, fixed_gb: float, bytes_per_token: float, margin_gb: float) -> int:
    """Largest micro-batch (tokens) whose learner peak fits the device with ``margin_gb`` to spare."""
    room = (float(device_gb) - float(nontorch_gb) - float(margin_gb) - float(fixed_gb)) * GiB
    return max(0, int(room // float(bytes_per_token)))


def budget(ts_cfg: Any) -> Optional[dict[str, float]]:
    """The configured budget constants, or None when the check is off; ValueError if incomplete."""
    bpt = getattr(ts_cfg, "train_mem_bytes_per_token", None)
    if bpt is None:
        return None
    fixed = getattr(ts_cfg, "train_mem_fixed_gb", None)
    nontorch = getattr(ts_cfg, "train_mem_nontorch_gb", None)
    margin = float(getattr(ts_cfg, "train_mem_margin_gb", 5.0))
    if fixed is None or nontorch is None:
        raise ValueError(
            "thundersync.train_mem_bytes_per_token needs train_mem_fixed_gb and train_mem_nontorch_gb "
            "(measured: learner/mem_peak_gb vs learner/max_mb_tokens, learner/mem_nontorch_gb after a save)"
        )
    if float(bpt) <= 0 or margin < 0:
        raise ValueError(f"thundersync.train_mem_bytes_per_token must be > 0 and train_mem_margin_gb >= 0 "
                         f"(got {bpt}, {margin})")
    return {"bytes_per_token": float(bpt), "fixed_gb": float(fixed), "nontorch_gb": float(nontorch), "margin_gb": margin}


def check_cap(cap_tokens: int, max_row_tokens: int, device_gb: float, b: dict[str, float]) -> int:
    """Startup rule: ValueError when the micro-batch token cap does not fit; returns fit_tokens."""
    fit = fit_tokens(device_gb, b["nontorch_gb"], b["fixed_gb"], b["bytes_per_token"], b["margin_gb"])
    _STATE.update(device_gb=float(device_gb), **b)
    if int(cap_tokens) > fit:
        rows = fit // int(max_row_tokens)
        hint = (
            f"policy.train_micro_batch_size / thundersync.train_micro_batch_tokens giving <= {rows} rows "
            f"(<= {rows * int(max_row_tokens)} tokens)"
            if rows >= 1
            else "a smaller policy.max_total_sequence_length or a layout with less per-rank memory (one row does not fit)"
        )
        raise ValueError(
            f"training micro-batch cap {cap_tokens} tokens does not fit the learner: fit_tokens = "
            f"floor((device {device_gb:.2f} - nontorch {b['nontorch_gb']:.2f} - margin {b['margin_gb']:.2f} - fixed "
            f"{b['fixed_gb']:.2f} GiB) / {b['bytes_per_token'] / 2**20:.3f} MiB per token) = {fit}; use {hint}"
        )
    return fit


def measured_fit(peak_gb: float, mb_tokens: float, nontorch_gb: float) -> Optional[float]:
    """Per-step: fit_tokens from this step's own peak / largest micro-batch / non-torch memory (the
    configured slope); None before the startup check ran or without data."""
    if "device_gb" not in _STATE or not mb_tokens or peak_gb != peak_gb:
        return None
    bpt = _STATE["bytes_per_token"]
    fixed = float(peak_gb) - float(mb_tokens) * bpt / GiB
    return float(fit_tokens(_STATE["device_gb"], nontorch_gb, fixed, bpt, _STATE["margin_gb"]))
