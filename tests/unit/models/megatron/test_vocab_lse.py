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
"""Canonical-block vocab log-probs (nemo_rl/models/megatron/vocab_lse.py, NEMO_RL_VOCAB_LSE=canonical_block):
accuracy, TP / batch invariance of the LSE, one definition across the trainer, verifier and engine call sites,
and (torchrun, 2 GPUs) the sharded trainer / verifier paths equal to the gathered-row path bit for bit."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.megatron import vocab_lse

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
V = 32768  # 8 canonical blocks


def _bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(-1).view(torch.uint8), b.contiguous().view(-1).view(torch.uint8)
    )


def _z(n, dtype=torch.float32, seed=0, width=V):
    g = torch.Generator(device="cuda").manual_seed(seed)
    z = torch.randn(n, width, generator=g, device="cuda") * 3
    z[:, :7] += 9.0
    return z.to(dtype)


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("vlim", [V, V - 100, 5000])
def test_lse_accuracy_tp_and_batch_invariance(dtype, vlim):
    z = _z(257, dtype)
    full = vocab_lse.lse_full(z, vlim)
    ref = torch.logsumexp(torch.where(torch.arange(V, device="cuda") < vlim, z.double(), float("-inf")), -1)
    assert ((full.double() - ref).abs() / ref.abs().clamp_min(1)).max() <= 2.0**-21
    for tp in (2, 4, 8):
        w = V // tp
        parts = torch.cat([vocab_lse.partials(z[:, r * w:(r + 1) * w], r * w, vlim) for r in range(tp)], 1)
        assert _bits_equal(vocab_lse.combine(parts), full)
    idx = torch.arange(256, -1, -5, device="cuda")
    assert _bits_equal(vocab_lse.lse_full(z[idx].contiguous(), vlim), full[idx])
    assert _bits_equal(vocab_lse.lse_full(z[3:4], vlim), full[3:4])


@cuda
def test_logprob_rows_is_z_minus_lse_and_normalized():
    z = _z(64)
    lp = vocab_lse.logprob_rows(z)
    assert _bits_equal(lp, z.float() - vocab_lse.lse_full(z).unsqueeze(-1))
    assert (lp.double().exp().sum(-1) - 1).abs().max() < 1e-5


@cuda
def test_edge_rows():
    z = torch.randn(4, V, device="cuda")
    z[0] = float("-inf")
    z[1, 5] = float("nan")
    z[2, 9] = float("inf")
    lse = vocab_lse.lse_full(z)
    assert lse[0].item() == float("-inf") and torch.isnan(lse[1]) and lse[2].item() == float("inf")
    assert torch.isfinite(lse[3])


@cuda
def test_call_sites_share_the_definition(monkeypatch):
    monkeypatch.delenv(vocab_lse.ENV, raising=False)  # the test sets the flag itself; starts from off
    from thundersync_rl.engine_q import gathered_log_softmax, no_gather

    from nemo_rl.models.megatron import exact_nucleus
    from nemo_rl.models.megatron.zero_train_gen_mismatch import _row_log_softmax

    z = _z(32)
    want_off = torch.log_softmax(z.float(), -1)
    assert _bits_equal(gathered_log_softmax(z, no_gather, V), want_off)  # flag off: unchanged
    monkeypatch.setenv(vocab_lse.ENV, "canonical_block")
    want = vocab_lse.logprob_rows(z)
    assert _bits_equal(gathered_log_softmax(z, no_gather, V), want)
    assert _bits_equal(exact_nucleus.policy_log_softmax(z), want)
    assert _bits_equal(_row_log_softmax(z.float()), want)


@cuda
def test_engine_raw_log_probs_switch(monkeypatch, capsys):
    monkeypatch.delenv(vocab_lse.ENV, raising=False)  # the test sets the flag itself; starts from off
    pytest.importorskip("megatron.core.inference.contexts.dynamic_context")
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext

    from nemo_rl.models.megatron.zero_train_gen_mismatch import _install_vocab_lse_engine

    z = _z(16)
    fake = SimpleNamespace(config=SimpleNamespace(logprobs_mode="raw_logprobs"))
    monkeypatch.setenv(vocab_lse.ENV, "canonical_block")
    _install_vocab_lse_engine()
    got = DynamicInferenceContext._processed_log_probs(fake, z, 16, None, None)
    assert _bits_equal(got, vocab_lse.logprob_rows(z))
    DynamicInferenceContext._processed_log_probs(fake, z, 16, None, None)
    out = capsys.readouterr().out
    assert out.count("engine raw log-probs first call") <= 1 and "role=engine" in out or getattr(
        DynamicInferenceContext._processed_log_probs, "_called", False
    )
    monkeypatch.delenv(vocab_lse.ENV)
    off = DynamicInferenceContext._processed_log_probs(fake, z, 16, None, None)
    assert _bits_equal(off, torch.nn.functional.log_softmax(z, dim=-1))


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="2 GPUs")
def test_sharded_paths_equal_gathered_rows_tp2():
    script = os.path.join(os.path.dirname(__file__), "vocab_lse_multirank.py")
    env = dict(os.environ, NEMO_RL_VOCAB_LSE="canonical_block")
    out = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2", script],
        env=env, capture_output=True, text=True, timeout=900,
    )
    assert "VOCAB_LSE MULTIRANK OK" in out.stdout, out.stdout[-3000:] + out.stderr[-3000:]


# --------------------------------------------------------------------------------------------------------------
# The fast kernels (one pass per block, row-vectorized combine, in-place bf16 reads, fused backward) must keep the
# bits of the reference kernels the definition was introduced with (kernel task D's vl_reference, copied here).
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _ref_partials_kernel(Z, OUT, stride_z, voff, vlim, nb_local, BLOCK: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    j = tl.program_id(1)
    base = Z + row.to(tl.int64) * stride_z + j * BLOCK
    g0 = voff + j * BLOCK
    m = tl.full([], float("-inf"), tl.float32)
    nan = tl.zeros([], tl.int32)
    pinf = tl.zeros([], tl.int32)
    for c0 in range(0, BLOCK, TILE):
        cols = c0 + tl.arange(0, TILE)
        ok = (g0 + cols) < vlim
        x = tl.load(base + cols, mask=ok, other=float("-inf")).to(tl.float32)
        nan += tl.sum((x != x).to(tl.int32), 0)
        pinf += tl.sum((x == float("inf")).to(tl.int32), 0)
        x = tl.where(x != x, float("-inf"), x)
        m = tl.maximum(m, tl.max(x, 0))
    s = tl.zeros([], tl.float32)
    for c0 in range(0, BLOCK, TILE):
        cols = c0 + tl.arange(0, TILE)
        ok = (g0 + cols) < vlim
        x = tl.load(base + cols, mask=ok, other=float("-inf")).to(tl.float32)
        live = (x > float("-inf")) & (x == x) & (m > float("-inf")) & (m < float("inf"))
        e = tl.where(live, tl.exp(tl.where(live, x - m, 0.0)), 0.0)
        s += tl.sum(e, 0)
    s = tl.where(pinf > 0, 1.0, s)
    m = tl.where(nan > 0, float("nan"), m)
    o = OUT + (row.to(tl.int64) * nb_local + j) * 2
    tl.store(o, m)
    tl.store(o + 1, s)


@triton.jit
def _ref_combine_kernel(P, OUT, nb):
    row = tl.program_id(0)
    base = P + row.to(tl.int64) * nb * 2
    M = tl.full([], float("-inf"), tl.float64)
    nan = tl.zeros([], tl.int32)
    for j in range(0, nb):
        mj = tl.load(base + 2 * j).to(tl.float64)
        nan += (mj != mj).to(tl.int32)
        M = tl.maximum(M, tl.where(mj != mj, float("-inf"), mj))
    S = tl.zeros([], tl.float64)
    for j in range(0, nb):
        mj = tl.load(base + 2 * j).to(tl.float64)
        sj = tl.load(base + 2 * j + 1).to(tl.float64)
        live = (mj > float("-inf")) & (mj == mj) & (M < float("inf"))
        S += tl.where(live, sj * tl.exp(tl.where(live, mj - M, 0.0)), 0.0)
    lse = tl.where(M == float("-inf"), float("-inf"), tl.where(M == float("inf"), float("inf"), M + tl.log(S)))
    lse = tl.where(nan > 0, float("nan"), lse)
    tl.store(OUT + row, lse.to(tl.float32))


def _ref_partials(z, off, vlim):
    z = z.float()
    n, w = z.shape
    vlim = min(vlim, off + w)  # as vocab_lse.partials passes it
    out = torch.empty(n, w // vocab_lse.BLOCK, 2, device=z.device)
    _ref_partials_kernel[(n, w // vocab_lse.BLOCK)](z, out, z.stride(0), off, vlim, w // vocab_lse.BLOCK,
                                                    BLOCK=vocab_lse.BLOCK, TILE=1024)
    return out


def _ref_combine(p):
    p = p.contiguous()
    out = torch.empty(p.shape[0], device=p.device)
    _ref_combine_kernel[(p.shape[0],)](p, out, p.shape[1])
    return out


def _special_rows(width):
    rows = [torch.randn(width, device="cuda") * 2 for _ in range(12)]
    rows[0][:] = float("-inf")
    rows[1][77] = float("nan")
    rows[2][width - 9] = float("inf")
    rows[3][5], rows[3][width // 2] = 1e38, -1e38
    rows[4][:] = 0.25
    rows[5][: width // 2] = float("-inf")
    rows[6] = -rows[6].abs() - 0.5
    rows[6][::1500] = 0.0
    rows[6][1::1700] = -0.0
    rows[7].zero_()
    rows[8] = -torch.zeros(width, device="cuda")
    rows[9] *= 20
    rows[10][3], rows[10][4] = float("nan"), float("inf")
    return torch.stack(rows)


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("vlim", [V, V - 100, V - 16, 5000, 1])
def test_fast_kernels_keep_reference_bits(dtype, vlim):
    z = torch.cat([_z(300, torch.float32, seed=4), _z(40, torch.float32, seed=5) * 7, _special_rows(V)]).to(dtype)
    want = _ref_partials(z, 0, vlim)
    assert _bits_equal(vocab_lse.partials(z, 0, vlim), want)
    for tp in (2, 8):
        w = V // tp
        for r in range(tp):
            assert _bits_equal(vocab_lse.partials(z[:, r * w:(r + 1) * w], r * w, vlim), _ref_partials(z[:, r * w:(r + 1) * w], r * w, vlim))
    wide = torch.zeros(z.shape[0], V + 512, device="cuda", dtype=dtype)
    wide[:, 256:256 + V] = z
    assert _bits_equal(vocab_lse.partials(wide[:, 256:256 + V], 0, vlim), want)
    odd = torch.zeros(z.shape[0], V + 17, device="cuda", dtype=dtype)
    odd[:, 3:3 + V] = z
    # misaligned views (odd base / row stride) give the bits of the same rows aligned (batch invariance)
    assert _bits_equal(vocab_lse.partials(odd[:, 3:3 + V], 0, vlim), want)
    assert _bits_equal(vocab_lse.combine(want), _ref_combine(want))


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_grad_equals_eager_chunk_loop(dtype):
    from nemo_rl.models.megatron.zero_train_gen_mismatch import _canonical_sharded_grad

    B, S, w, start = 2, 300, V // 2, V // 2
    local = _z(B * S, dtype, seed=9, width=w).view(B, S, w)
    target = torch.randint(0, V, (B, S), device="cuda")
    target[0, :50] = start + 7  # owned tokens on this shard
    gout = torch.randn(B, S, device="cuda")
    lse = vocab_lse.lse_full(local.float() + 1.0)
    want = _canonical_sharded_grad(local, target, gout, lse, (start, start + w), 64)
    got = vocab_lse.logprob_grad(local, target, gout, lse, start)
    assert got.dtype == local.dtype and _bits_equal(got, want)


def test_fused_grad_host_fallback_equals_eager_chunk_loop():
    """CPU tensors: logprob_grad takes the eager form (no Triton launch), bitwise the eager chunk loop."""
    from nemo_rl.models.megatron.zero_train_gen_mismatch import _canonical_sharded_grad

    torch.manual_seed(0)
    B, S, w, start = 2, 5, 64, 64
    for dtype in (torch.float32, torch.bfloat16):
        local = (torch.randn(B, S, w) * 3).to(dtype)
        target = torch.randint(0, 2 * w, (B, S))
        target[0, :2] = start + 3
        gout = torch.randn(B, S)
        lse = torch.logsumexp(local.float(), -1) + 1.0
        want = _canonical_sharded_grad(local, target, gout, lse, (start, start + w), S)
        got = vocab_lse.logprob_grad(local, target, gout, lse, start)
        assert got.dtype == local.dtype and _bits_equal(got, want)
