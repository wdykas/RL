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
"""Batch-invariant Mamba decode without the z buffer, and its memory accounting."""

import pytest
import torch

from nemo_rl.models.megatron import mamba_bi_decode as mbd

# Nemotron-3-Super per rank at TP=1: 128 heads x 64, 8 groups, d_state 128, chunk 128, 40 Mamba layers.
SUPER = dict(conv=(8192 + 2 * 8 * 128, 3), ssm=(128, 64, 128), chunk=128, layers=40)


def test_replay_bytes_super_shape():
    """fp32 replay buffers at Super width: 362 MiB/slot with z (+1.25 MiB output rows), 202 without."""
    kw = dict(chunk_size=SUPER["chunk"], elem_bytes=4, num_mamba_layers=SUPER["layers"])
    with_z = mbd.replay_bytes_per_request(
        SUPER["conv"], SUPER["ssm"], z_buffered=True, **kw
    )
    no_z = mbd.replay_bytes_per_request(
        SUPER["conv"], SUPER["ssm"], z_buffered=False, **kw
    )
    assert 362 < with_z / 2**20 < 364
    assert with_z - no_z == 40 * 128 * 8192 * 4  # chunk x d_inner x fp32 per layer


def test_budget_with_ratio_keeps_kv_and_fits_replay():
    total, r, p, q = 100 * 2**30, 0.1, 160 * 2**20, 202 * 2**20
    buf, paused, ratio = mbd.replay_adjusted_budget(
        total * 0.75, total * 0.25, r, None, p, q
    )
    new_total = buf + paused
    assert buf / paused == pytest.approx(3.0)
    assert (1 - ratio) * new_total == pytest.approx(
        (1 - r) * total
    )  # KV region unchanged
    requests = ratio * new_total / p  # what MCore sizes the Mamba region for
    assert requests == pytest.approx(r * total / (p + q))
    assert requests * (p + q) == pytest.approx(
        r * total
    )  # states + replay fit the region


def test_budget_with_max_requests_reserves_replay():
    total, n, q = 64 * 2**30, 100, 202 * 2**20
    buf, paused, ratio = mbd.replay_adjusted_budget(total, 0.0, None, n, 1, q)
    assert ratio is None and paused == 0.0
    assert buf == pytest.approx(total - n * q)
    with pytest.raises(ValueError, match="replay buffers"):
        mbd.replay_adjusted_budget(total, 0.0, None, 1000, 1, q)


def test_budget_noop_without_replay_and_unbuilt_branch():
    assert mbd.replay_adjusted_budget(10.0, 2.0, 0.1, None, 5, 0) == (10.0, 2.0, 0.1)
    with pytest.raises(NotImplementedError):
        mbd.replay_adjusted_budget(10.0, 0.0, None, None, 5, 7)


# ---------------------------------------------------------------------------
# GPU: decode outputs bitwise equal with and without the z buffer.
# ---------------------------------------------------------------------------

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _decode(z_free: bool, steps: int, slots, prefill, graph: bool, seed=0):
    """MCore's buffered decode (z buffer) or ours: per-step outputs and the final ssm_state."""
    from megatron.core.ssm.ops.mamba2 import batch_invariant_decode as bid

    nh, hd, ng, ds, chunk = 16, 64, 2, 128, 128
    R = max(slots) + 2
    g = torch.Generator(device="cuda").manual_seed(seed)
    mbd.install()  # idempotent; the reference arm calls MCore's originals
    cls = bid.BatchInvariantDecodeBuffers
    if z_free:
        alloc, seed_fn, scan = (
            cls.allocate,
            cls.seed,
            bid.batch_invariant_decode_buffered_scan,
        )
    else:
        alloc = mbd._ORIG["allocate"].__get__(None, cls)
        seed_fn, scan = mbd._ORIG["seed"], mbd._ORIG["scan"]
    buf = alloc(R, chunk, nh, hd, ng, ds, torch.device("cuda"), torch.float32)
    assert (buf.z is None) == z_free
    A = -torch.rand(nh, device="cuda", generator=g) - 0.5
    D = torch.randn(nh, device="cuda", generator=g)
    dtb = torch.randn(nh, device="cuda", generator=g) * 0.1
    state = torch.randn(R, nh, hd, ds, device="cuda", generator=g) * 0.1

    def rnd(*shape):
        return torch.randn(*shape, device="cuda", generator=g)

    # Seed each slot from a prefill tail (lengths straddle a chunk boundary).
    for slot, plen in zip(slots, prefill):
        seed_fn(
            buf,
            rnd(plen, nh, hd),
            rnd(plen, nh, hd),
            rnd(plen, nh),
            rnd(plen, ng, ds),
            rnd(plen, ng, ds),
            torch.tensor([0, plen], device="cuda", dtype=torch.int32),
            torch.tensor([slot], device="cuda", dtype=torch.int32),
        )
    # One padding lane (-1), as CUDA-graph padding does.
    bi = torch.tensor(list(slots) + [-1], device="cuda", dtype=torch.int32)
    n = bi.numel()
    ins = [
        torch.zeros(n, 1, nh, hd, device="cuda"),
        torch.zeros(n, 1, nh, hd, device="cuda"),
        torch.zeros(n, 1, nh, device="cuda"),
        torch.zeros(n, 1, ng, ds, device="cuda"),
        torch.zeros(n, 1, ng, ds, device="cuda"),
    ]

    def fill():
        for t in ins:
            t.copy_(torch.randn(t.shape, device="cuda", generator=g))

    outs = []
    if graph:
        fill()
        static_state = state.clone()
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):  # warm up the Triton kernels outside the graph
            scan(buf, *ins, A, D, dtb, bi, static_state.clone())
        torch.cuda.current_stream().wait_stream(side)
        # Capture runs once: capture on scratch copies, then restore buffers.
        snap = {k: v.clone() for k, v in vars(buf).items() if torch.is_tensor(v)}
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg):
            out = scan(buf, *ins, A, D, dtb, bi, state)
        for k, v in snap.items():
            getattr(buf, k).copy_(v)
        for _ in range(steps):
            fill()
            cg.replay()
            outs.append(out.clone())
    else:
        for _ in range(steps):
            fill()
            outs.append(scan(buf, *ins, A, D, dtb, bi, state).clone())
    return torch.stack(outs), state


@cuda
@pytest.mark.parametrize("graph", [False, True])
def test_decode_without_z_buffer_is_bitwise_identical(graph):
    """300 steps (each slot crosses >= 2 chunk boundaries), 3 slots + a padding lane."""
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        _pin_mamba_autotuners,
    )

    _pin_mamba_autotuners()  # as enable_batch_invariant_mode does in the engine
    kw = dict(steps=300, slots=(0, 3, 1), prefill=(5, 127, 130), graph=graph)
    out_ref, st_ref = _decode(False, **kw)
    out_new, st_new = _decode(True, **kw)
    assert torch.isfinite(out_ref[:, :3]).all()
    assert torch.equal(out_ref, out_new)
    assert torch.equal(st_ref, st_new)
