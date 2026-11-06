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
"""engine_fast_ops.log_softmax == Megatron's batch-invariant log_softmax, bitwise (GPU)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU test")


def _orig():
    from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik

    from nemo_rl.models.megatron import engine_fast_ops

    return engine_fast_ops._STATE["orig"] or bik.log_softmax


def _rows(kind, n, v, g):
    if kind == "randn":
        return torch.randn(n, v, generator=g, device="cuda") * 4
    if kind == "logits":  # LM-head-like: a few large entries
        x = torch.randn(n, v, generator=g, device="cuda") * 2
        idx = torch.randint(0, v, (n, 8), generator=g, device="cuda")
        x.scatter_(1, idx, 25.0 + torch.rand(n, 8, generator=g, device="cuda"))
        return x
    if kind == "special":
        x = torch.randn(n, v, generator=g, device="cuda") * 3
        m = torch.rand(n, v, generator=g, device="cuda")
        x[m < 0.2] = float("-inf")
        x[(m > 0.5) & (m < 0.50002)] = float("nan")
        if n > 1:
            x[1] = float("-inf")
        return x
    if kind == "zeros":  # the row max is a mix of +0 and -0 at varying positions
        x = -torch.rand(n, v, generator=g, device="cuda") * 30 - 1
        for r in range(n):
            pos = torch.randint(0, v, (6,), generator=g, device="cuda")
            x[r, pos[:3]] = 0.0
            x[r, pos[3:]] = -0.0
        return x
    if kind == "onehot":  # sum_exp == 1 exactly: log_sum_exp == 0 (signed-zero outputs visible)
        x = torch.full((n, v), -200.0, device="cuda")
        x[:, 0] = -0.0
        x[:, 1] = 0.0
        return x
    raise ValueError(kind)


@pytest.mark.parametrize("v", [131072, 131000, 152064, 2049, 1024, 100])
@pytest.mark.parametrize("n", [1, 7, 64])
@pytest.mark.parametrize("kind", ["randn", "logits", "special", "zeros", "onehot"])
def test_fast_log_softmax_bitwise(v, n, kind):
    from nemo_rl.models.megatron import engine_fast_ops

    engine_fast_ops.install_fast_logsoftmax()
    g = torch.Generator(device="cuda").manual_seed(v * 31 + n)
    x = _rows(kind, n, v, g)
    a = _orig()(x, dim=-1)
    b = engine_fast_ops.log_softmax(x, dim=-1)
    torch.cuda.synchronize()
    assert torch.equal(a.view(torch.int32), b.view(torch.int32)), (a[a != b][:5], b[a != b][:5])


def test_installed_override_used_by_aten():
    """With batch-invariant mode on, F.log_softmax reaches the fast kernel and keeps the bits."""
    from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik

    from nemo_rl.models.megatron import engine_fast_ops

    engine_fast_ops.install_fast_logsoftmax()
    x = torch.randn(3, 131072, device="cuda")
    calls = engine_fast_ops._STATE["calls"]
    with bik.set_batch_invariant_mode(True):
        y = torch.nn.functional.log_softmax(x, dim=-1)
    assert engine_fast_ops._STATE["calls"] == calls + 1
    assert torch.equal(y.view(torch.int32), _orig()(x).view(torch.int32))


def test_speed_report():
    from nemo_rl.models.megatron import engine_fast_ops

    engine_fast_ops.install_fast_logsoftmax()
    for n in (1, 16, 64, 256):
        x = torch.randn(n, 131072, device="cuda")
        res = {}
        for name, f in (("orig", _orig()), ("fast", engine_fast_ops.log_softmax)):
            for _ in range(3):
                f(x)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(30):
                f(x)
            e.record()
            torch.cuda.synchronize()
            res[name] = s.elapsed_time(e) / 30 * 1e3
            # device time: 30 calls captured in one CUDA graph (no host launch cost)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(30):
                    f(x)
            g.replay()
            s.record()
            g.replay()
            e.record()
            torch.cuda.synchronize()
            res[name + "_dev"] = s.elapsed_time(e) / 30 * 1e3
        print(f"[fast log_softmax] rows {n}: eager orig {res['orig']:.1f} us fast {res['fast']:.1f} us; "
              f"device (graphed) orig {res['orig_dev']:.1f} us fast {res['fast_dev']:.1f} us")
