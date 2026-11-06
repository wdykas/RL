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
"""Engine log-probs accumulated alongside streamed draft tokens (worker side)."""

from types import SimpleNamespace

import torch
import numpy as np
from thundersync_rl.worker import apply_draft_frame, snapshot_drafts


def _partial(toks, lps=None):
    part = {"request_id": 0, "new_tokens": toks}
    if lps is not None:
        part["new_log_probs"] = lps
    return {"partial": part}


def test_partial_frames_accumulate_logprobs_and_snapshot():
    tokens, logprobs, done = [[]], [[]], [False]
    assert not apply_draft_frame(_partial([4, 5], [-0.1, -0.2]), tokens[0], logprobs, 0)
    assert not apply_draft_frame(_partial([6], [-0.3]), tokens[0], logprobs, 0)
    snap = snapshot_drafts(tokens, logprobs, done)
    (toks, d, lps), = snap
    # Typed arrays; these log-probs are not fp32 values, so they stay float64 (bitwise).
    assert toks.tolist() == [4, 5, 6] and d is False and lps.dtype == np.float64
    assert lps.tolist() == [-0.1, -0.2, -0.3]
    lps[0] = 0.0  # a copy: later frames must not alias it
    assert logprobs[0] == [-0.1, -0.2, -0.3]


def test_partial_without_logprobs_drops_them():
    tokens, logprobs = [[]], [[]]
    apply_draft_frame(_partial([4], [-0.1]), tokens[0], logprobs, 0)
    apply_draft_frame(_partial([5]), tokens[0], logprobs, 0)
    apply_draft_frame(_partial([6], [-0.3]), tokens[0], logprobs, 0)
    assert tokens[0] == [4, 5, 6] and logprobs[0] is None
    (toks, d, lps), = snapshot_drafts(tokens, logprobs, [False])
    assert toks.tolist() == [4, 5, 6] and d is False and lps is None


def test_typed_logq_is_bitwise():
    from thundersync_rl.worker import typed_logq

    import torch

    fp32 = torch.randn(1000).float().tolist()  # engine log-probs: fp32 values as Python floats
    a = typed_logq(fp32)
    assert a.dtype == np.float32 and [float(v) for v in a] == fp32
    mixed = fp32[:10] + [0.1]  # 0.1 is not an fp32 value: the array stays float64
    b = typed_logq(mixed)
    assert b.dtype == np.float64 and b.tolist() == mixed


def test_final_frame_replaces_tokens_and_logprobs():
    tokens, logprobs = [[4]], [[-0.1]]
    final = {
        "generated_tokens": [4, 5, 2],
        "generated_log_probs": [-0.1, -0.2, -0.3],
    }
    assert apply_draft_frame({"final": final}, tokens[0], logprobs, 0)
    assert tokens[0] == [4, 5, 2] and logprobs[0] == [-0.1, -0.2, -0.3]
    # DynamicInferenceRequest-like final with a tensor of log-probs.
    req = SimpleNamespace(
        generated_tokens=[7, 2], generated_log_probs=torch.tensor([-0.5, -0.25])
    )
    tokens, logprobs = [[]], [[]]
    assert apply_draft_frame({"final": req}, tokens[0], logprobs, 0)
    assert logprobs[0] == [-0.5, -0.25]
    # Missing or misaligned final log-probs: None.
    tokens, logprobs = [[]], [[]]
    apply_draft_frame({"final": {"generated_tokens": [7, 2]}}, tokens[0], logprobs, 0)
    assert logprobs[0] is None
