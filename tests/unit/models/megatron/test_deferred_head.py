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
"""Deferred training LM head: the same log-probs as the materialized head, chunk by chunk."""

import pytest
import torch

fg = pytest.importorskip("nemo_rl.models.megatron.fp16_gemm")
zgm = pytest.importorskip("nemo_rl.models.megatron.zero_train_gen_mismatch")
from nemo_rl.models.megatron import deferred_head as dh  # noqa: E402


def _inputs(device, s=37, b=3, h=64, v=1000, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(s, b, h, generator=g) * 2).to(device)
    w = (torch.randn(v, h, generator=g) * 0.05).to(torch.bfloat16).to(device)
    target = torch.randint(0, v, (b, s), generator=g).to(device)
    gout = torch.randn(b, s, generator=g).to(device)
    return x, w, target, gout


def _materialized(x, w, target, gout, chunk):
    x = x.clone().requires_grad_(True)
    w = torch.nn.Parameter(w.clone())
    logits = fg.Fp16Linear.apply(x, w, fg.COMM_NONE, None, fg.FP32_HEAD)
    logits = logits.transpose(0, 1).contiguous().to(torch.float32)
    lp = zgm._LocalVocabLogprob.apply(logits, target, chunk)
    (lp * gout).sum().backward()
    return lp.detach(), x.grad, w.grad


def _deferred(x, w, target, gout, chunk):
    x = x.clone().requires_grad_(True)
    w = torch.nn.Parameter(w.clone())
    logits = dh.deferred_head_logits(x, w)
    # The ops between Megatron's output layer and the log-prob op.
    logits = logits.transpose(0, 1).contiguous().to(torch.float32)
    assert isinstance(logits, dh.DeferredHeadLogits)
    assert tuple(logits.shape) == (x.shape[1], x.shape[0], w.shape[0])
    op = zgm._BatchInvariantLogprob(None, chunked=True)
    lp = op.apply(logits, target, 0, w.shape[0], chunk, None, False)
    (lp * gout).sum().backward()
    return lp.detach(), x.grad, w.grad


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("chunk", [5, 16, 64])
def test_deferred_head_matches_materialized(device, chunk):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    args = _inputs(device)
    lp_ref, dx_ref, dw_ref = _materialized(*args, chunk)
    lp, dx, dw = _deferred(*args, chunk)
    if device == "cuda":
        # Row-invariant fp16 GEMM (batch-invariant backend): bitwise log-probs.
        assert torch.equal(lp, lp_ref)
    else:
        torch.testing.assert_close(lp, lp_ref, rtol=0, atol=1e-6)
    # Gradients: same math, wgrad summed over chunks (rounding only).
    torch.testing.assert_close(dx, dx_ref, rtol=1e-2, atol=1e-4)
    torch.testing.assert_close(dw.float(), dw_ref.float(), rtol=2e-2, atol=1e-3)


def test_views_and_fallback():
    x, w, target, _ = _inputs("cpu", s=11, b=2, h=16, v=50)
    full = fg.Fp16Linear.apply(x, w, fg.COMM_NONE, None, fg.FP32_HEAD).transpose(0, 1)
    d = dh.deferred_head_logits(x, w).transpose(0, 1)
    assert d.shape == full.shape and d.dtype == torch.float32 and d.dim() == 3
    # Sequence slices stay deferred and select the same positions.
    for view, ref in (
        (d.narrow(1, 3, 5), full.narrow(1, 3, 5)),
        (d[:, 2:9], full[:, 2:9]),
        (d[:, :, :], full),
    ):
        assert isinstance(view, dh.DeferredHeadLogits)
        torch.testing.assert_close(view.materialize(), ref, rtol=0, atol=0)
    # Any other op materializes through the tracked head.
    torch.testing.assert_close(d.sum(), full.sum())
    torch.testing.assert_close(d * 2, full * 2)
    with pytest.raises(RuntimeError, match="in-place"):
        d.div_(2.0)
