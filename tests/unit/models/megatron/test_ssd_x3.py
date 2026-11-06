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
"""ssd_x3 (precision_raise site ssd_scan_x3): chunk-scan dots at tf32x3 on every path.

Under the rewrite, training (mamba_ssd_parity) == engine prefill == MCore buffered decode,
bitwise (the batch-invariant decode step, which mirrors the TF32 chain, declines); the compiled
kernels issue 3x the TF32 MMAs; the result is closer to fp64 than TF32's.
"""

import pytest
import torch

from nemo_rl.models.megatron import ssd_x3
from tests.unit.models.megatron.test_mamba_ssd_parity import mixer_factory  # noqa: F401  (fixture)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_rewrite_pins_every_dot():
    src = (
        "def k(a, b, c):\n"
        "    acc = tl.dot(a, b) * 2\n"
        "    acc += tl.dot(c, b)\n"
        "    return acc\n"
    )
    out = ssd_x3._rewrite(src, "tf32x3")
    assert out.count("input_precision='tf32x3'") == 2
    with pytest.raises(ValueError):
        ssd_x3._rewrite("def k(a):\n    return a\n", "tf32x3")
    with pytest.raises(ValueError):
        ssd_x3._rewrite("def k(a, b):\n    return tl.dot(a, b, allow_tf32=True)\n", "tf32x3")


@pytest.fixture
def x3():
    ssd_x3.install()
    try:
        yield
    finally:
        ssd_x3.uninstall()


@cuda
def test_decode_step_declines_and_buffers_keep_z_under_x3(x3):
    """The batch-invariant decode step mirrors the TF32 chain: under tf32x3 it declines, so the engine's
    buffers keep z and MCore's (rewritten) tile path replays the scan."""
    from megatron.core.ssm.ops.mamba2 import batch_invariant_decode as bid

    from nemo_rl.models.megatron import mamba_bi_decode, mamba_bi_fused_decode
    from tests.unit.models.megatron.test_mamba_bi_scan_row import _pin

    _pin()
    mamba_bi_decode.install()
    dev = torch.device("cuda")
    assert not mamba_bi_fused_decode.supported_config(16, 64, 2, 128, 128, torch.float32, dev)
    buf = bid.BatchInvariantDecodeBuffers.allocate(4, 128, 16, 64, 2, 128, dev, torch.float32)
    assert buf.z is not None
    ssd_x3.uninstall()
    try:
        assert mamba_bi_fused_decode.supported_config(16, 64, 2, 128, 128, torch.float32, dev)
    finally:
        ssd_x3.install()


@cuda
def test_training_matches_engine_prefill_and_decode_x3(x3, mixer_factory):
    """mamba_ssd_parity training forward == engine prefill + decode at tf32x3 (fp32 recipe inputs)."""
    from tests.unit.models.megatron.test_mamba_ssd_parity import (
        test_training_matches_engine_prefill_and_decode,
    )

    test_training_matches_engine_prefill_and_decode(mixer_factory, torch.float32)


def _scan_inputs(seqlen=512, nh=8, hd=64, ng=1, ds=128, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*s):
        return torch.randn(*s, device="cuda", generator=g)

    return dict(
        x=rnd(1, seqlen, nh, hd),
        dt=rnd(1, seqlen, nh) - 2.0,
        A=-torch.exp(torch.empty(nh, device="cuda").uniform_(-4.0, 0.5, generator=g)),
        B=rnd(1, seqlen, ng, ds) * 0.3,
        C=rnd(1, seqlen, ng, ds) * 0.3,
        D=rnd(nh),
    )


def _reference(d):
    """fp64 sequential recurrence with softplus(dt)."""
    x, dt, A, B, C, D = (d[k].double() for k in ("x", "dt", "A", "B", "C", "D"))
    _, s, nh, hd = x.shape
    ds = B.shape[-1]
    dt = torch.nn.functional.softplus(dt)
    h = torch.zeros(nh, hd, ds, dtype=torch.float64, device=x.device)
    ys = []
    for t in range(s):
        a = torch.exp(dt[0, t] * A)  # [nh]
        bt = B[0, t, 0]  # ng = 1
        ct = C[0, t, 0]
        h = a[:, None, None] * h + (dt[0, t][:, None, None] * x[0, t][:, :, None]) * bt[None, None, :]
        ys.append(torch.einsum("hpn,n->hp", h, ct) + D[:, None] * x[0, t])
    return torch.stack(ys)[None]


def _engine(d):
    from nemo_rl.models.megatron import mamba_ssd_parity

    mamba_ssd_parity.pin_engine_kernels()
    return mamba_ssd_parity._engine_scan(
        d["x"], d["dt"], d["A"], d["B"], d["C"], 128, d["D"], None, None, True, (0.0, float("inf"))
    )


@cuda
def test_x3_is_closer_to_fp64_and_emits_three_mmas():
    from megatron.core.ssm.ops.mamba2 import ssd_chunk_scan

    d = _scan_inputs()
    ref = _reference(d)
    y_tf32 = _engine(d).double()
    mma_tf32 = ssd_x3.compiled_mma_counts(ssd_chunk_scan._chunk_scan_fwd_kernel)
    ssd_x3.install()
    try:
        y_x3 = _engine(d).double()
        mma_x3 = ssd_x3.compiled_mma_counts(ssd_chunk_scan._chunk_scan_fwd_kernel)
    finally:
        ssd_x3.uninstall()
    y_back = _engine(d)  # uninstall restores the TF32 kernel bit for bit
    assert torch.equal(y_back.double(), y_tf32)
    err_tf32 = ((y_tf32 - ref).norm() / ref.norm()).item()
    err_x3 = ((y_x3 - ref).norm() / ref.norm()).item()
    # bmm / chunk-state stay TF32, so the remaining error is theirs (measured 6.5e-4 vs 1.2e-3).
    assert err_x3 < 0.75 * err_tf32, (err_x3, err_tf32)
    assert len(mma_tf32) == 1 and len(mma_x3) == 1
    tf, x3 = mma_tf32[0], mma_x3[0]
    n_tf = tf["mma_tf32"] + tf["tcgen05_mma"] + tf["wgmma"]
    n_x3 = x3["mma_tf32"] + x3["tcgen05_mma"] + x3["wgmma"]
    assert n_tf > 0 and n_x3 == 3 * n_tf, (tf, x3)
