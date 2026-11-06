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
"""Batch-invariant ``aten::mean.dim``: a full reduction is a mean (nemo_rl.models.megatron.bi_mean)."""

import pytest
import torch

bik = pytest.importorskip(
    "megatron.core.transformer.custom_layers.batch_invariant_kernels"
)

from nemo_rl.models.megatron import bi_mean  # noqa: E402


@pytest.fixture
def installed(monkeypatch):
    monkeypatch.setattr(bik, "mean_batch_invariant", bik.mean_batch_invariant)
    monkeypatch.setattr(bi_mean, "_MCORE_MEAN", bi_mean._MCORE_MEAN)
    bi_mean.install()
    return bik.mean_batch_invariant


@pytest.mark.parametrize("dim", [[], (), None])
@pytest.mark.parametrize("keepdim", [False, True])
def test_full_reduction_is_mean(installed, dim, keepdim):
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    out = installed(x, dim, keepdim=keepdim)
    torch.testing.assert_close(out, x.mean(dim=(0, 1, 2), keepdim=keepdim))


def test_scalar_input(installed):
    x = torch.tensor(3.5)
    torch.testing.assert_close(installed(x, []), x)


def test_multi_dim_unchanged(installed):
    x = torch.randn(3, 5, 7)
    torch.testing.assert_close(installed(x, [0, 2]), x.mean(dim=(0, 2)))


def test_install_is_idempotent(installed):
    bi_mean.install()
    assert bik.mean_batch_invariant is installed


def test_install_after_enable_raises(monkeypatch):
    monkeypatch.setattr(bik, "mean_batch_invariant", bik.mean_batch_invariant)
    if bi_mean.mcore_full_mean_is_correct(bik.mean_batch_invariant):
        pytest.skip("this Megatron-Core already has the fix")
    monkeypatch.setattr(bik, "is_batch_invariant_mode_enabled", lambda: True)
    with pytest.raises(RuntimeError, match="before enable_batch_invariant_mode"):
        bi_mean.install()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_registered_kernel_full_mean_and_grad(installed):
    """Through the dispatcher: x.mean() / torch.mean(x) under batch-invariant mode."""
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    ref = x.detach().sum() / x.numel()
    bik.enable_batch_invariant_mode(backend="triton")
    try:
        m = x.mean()
        m_fn = torch.mean(x)
        m_rows = x.mean(dim=-1)
        m.backward()
    finally:
        bik.disable_batch_invariant_mode()
    torch.testing.assert_close(m.detach(), ref)
    torch.testing.assert_close(m_fn.detach(), ref)
    torch.testing.assert_close(m_rows.detach(), x.detach().mean(dim=-1))
    torch.testing.assert_close(x.grad, torch.full_like(x, 1.0 / x.numel()))
