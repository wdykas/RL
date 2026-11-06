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
"""Engine MoE kernels on persistent live-row grids are bitwise the padded-grid path.

The reference is the path the engine ran before (MCore ``permute_tokens``, the tile
picked from the padded buffer, the non-persistent ``grouped_gemm`` kernel over the
whole buffer, the eager activation, an fp64 expert-order sum), and cuBLAS (the
learner's expert GEMM) for every engine tile.
"""

import os

os.environ.setdefault("CUBLASLT_WORKSPACE_SIZE", "0")  # cuBLAS without split-K (BI)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":0:0")

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _bits(t):
    return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int16)


def _segments(rows, counts, bm):
    """``rows`` laid out in BLOCK_M-aligned expert segments: x [M_pad, K], inclusive offsets, live rows."""
    offs, live, start = [], [], 0
    for c in counts:
        live += list(range(start, start + c))
        start += -(-c // bm) * bm
        offs.append(start)
    m_pad = max(start, bm) + 4 * bm  # buffer padding past offsets[-1]
    x = torch.full((m_pad, rows.shape[1]), float("nan"), dtype=torch.float16, device="cuda")
    x[live] = rows  # rows inside a segment's alignment padding stay NaN: computed, never read
    return x, torch.tensor(offs, dtype=torch.int32, device="cuda"), live


@pytest.mark.parametrize("n,k", [(1856, 2688), (2688, 1856)])  # Nemotron-3-Nano fc1, fc2
def test_grouped_gemm_every_tile_and_grid_matches_cublas(n, k):
    from nemo_rl.models.megatron import fp16_moe as fm

    gen = torch.Generator(device="cuda").manual_seed(n)
    e = 8
    w16 = (torch.randn(e, n, k, device="cuda", generator=gen) * 0.02).to(torch.float16)
    counts = [3, 0, 17, 1, 0, 40, 2, 9]
    rows = (torch.randn(sum(counts), k, device="cuda", generator=gen) * 3).to(torch.float16)
    rows[0] = 0.0
    rows[-1] = -0.0
    rows[10, : k // 2] = 6e-8  # fp16 subnormal
    rows[20] = 60000.0
    ref, s = [], 0  # cuBLAS per expert (the learner's expert GEMM)
    for i, c in enumerate(counts):
        ref.append(torch.mm(rows[s : s + c], w16[i].t(), out_dtype=torch.float32))
        s += c
    ref = torch.cat(ref)
    for _, cfg in fm._ENGINE_CONFIGS:
        x, offs, live = _segments(rows, counts, cfg[0])
        for hint in (None, 1, len(live)):  # 1: a single program loops over every tile
            out = fm.grouped_gemm_fp16w(x, w16, offs, cfg, rows_hint=hint)
            assert torch.equal(_bits(out[live]), _bits(ref)), (cfg, hint)


def _routing(t, valid, topk, n_experts, gen):
    r = torch.stack([torch.randperm(n_experts, device="cuda", generator=gen)[:topk] for _ in range(t)])
    r[valid - 3 : valid] = -1  # CUDA-graph padding tokens (masked routing)
    return r.to(torch.int64)


def _reference(x16, probs, routing, valid, w1, w2, start, ep, activation, down_scale):
    """The pre-livegrid engine path: padded tile, MCore permute, full-buffer kernels."""
    from megatron.core.inference.moe.permute import permute_tokens

    from nemo_rl.models.megatron import fp16_moe as fm

    e = w1.shape[0]
    topk = routing.shape[1]
    cfg = fm.select_engine_config(x16.shape[0] * topk / (e * ep))
    vt = torch.tensor([valid], dtype=torch.int32, device="cuda")
    xp, pp, _, offs, inv = permute_tokens(
        x16, probs, routing, start, e, vt, alignment=cfg[0],
        return_batch_invariant_inverse_map=True, row_alignment=cfg[0],
    )
    if down_scale:
        pp = pp * (2.0**-down_scale)
    h = fm.grouped_gemm(xp, None, offs, cfg, stacked=w1)
    if activation == "relu2":  # the engine kernel is bitwise these eager ops
        a = fm._relu2_eager(h, pp).to(torch.float16)
    else:  # one program per row over the whole buffer (the training forward's launch)
        a = fm.swiglu_probs_fwd(h, pp, torch.float16)
    y = fm.grouped_gemm(a, None, offs, cfg, stacked=w2)
    if down_scale:
        y = y * (2.0**down_scale)
    out = torch.zeros(valid, y.shape[1], dtype=torch.float64, device="cuda")
    for lid in range(e):  # ascending local expert, fp64
        rows = inv[:valid, lid]
        hit = rows >= 0
        out[hit] += y[rows[hit].long()].double()
    return out.float()


@pytest.mark.parametrize("activation", ["relu2", "swiglu"])
@pytest.mark.parametrize("down_scale", [0, 3])
@pytest.mark.parametrize("ep,valid,hint", [(4, 37, None), (4, 37, 40), (4, 200, 8), (1, 64, 64), (8, 512, None)])
def test_engine_experts_bitwise_padded_path(activation, down_scale, ep, valid, hint):
    from nemo_rl.models.megatron import fp16_moe as fm

    gen = torch.Generator(device="cuda").manual_seed(valid + ep)
    t, k, f, topk, e = 512, 512, 384, 6, 8
    start = e * (ep - 1)  # the last EP rank's experts
    fn = 2 * f if activation == "swiglu" else f
    w1 = (torch.randn(e, fn, k, device="cuda", generator=gen) * 0.05).to(torch.float16)
    w2 = (torch.randn(e, k, f, device="cuda", generator=gen) * 0.05).to(torch.float16)
    x16 = torch.randn(t, k, device="cuda", generator=gen).to(torch.float16)
    probs = torch.rand(t, topk, device="cuda", generator=gen)
    routing = _routing(t, valid, topk, e * ep, gen)
    ref = _reference(x16, probs, routing, valid, w1, w2, start, ep, activation, down_scale)
    vt = torch.tensor([valid], dtype=torch.int32, device="cuda")
    out = fm.engine_experts(x16, probs, routing, vt, w1, w2, start, ep, activation,
                            tokens_hint=hint, down_scale=down_scale)
    assert torch.equal(_bits(out[:valid]), _bits(ref))


def test_engine_experts_cuda_graph_replay_changing_live_rows():
    from nemo_rl.models.megatron import fp16_moe as fm

    gen = torch.Generator(device="cuda").manual_seed(7)
    t, k, f, topk, e, ep = 1024, 512, 384, 6, 8, 4
    w1 = (torch.randn(e, f, k, device="cuda", generator=gen) * 0.05).to(torch.float16)
    w2 = (torch.randn(e, k, f, device="cuda", generator=gen) * 0.05).to(torch.float16)
    x16 = torch.randn(t, k, device="cuda", generator=gen).to(torch.float16)
    probs = torch.rand(t, topk, device="cuda", generator=gen)
    routing = _routing(t, 64, topk, e * ep, gen)
    vt = torch.tensor([64], dtype=torch.int32, device="cuda")
    args = (x16, probs, routing, vt, w1, w2, 0, ep, "relu2")
    fm.engine_experts(*args, tokens_hint=64)  # compile outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fm.engine_experts(*args, tokens_hint=64)  # bucket of 64 gathered tokens
    for valid in (5, 64, 300):  # 300 > the bucket hint: the persistent loops cover it
        routing.copy_(_routing(t, valid, topk, e * ep, gen))
        vt.fill_(valid)
        graph.replay()
        ref = _reference(x16, probs, routing, valid, w1, w2, 0, ep, "relu2", 0)
        assert torch.equal(_bits(out[:valid]), _bits(ref)), valid
