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
"""Engine zero-weight-slot skip (crossfade_routing_sparse_engine): bitwise the fixed-width engine expert path.

A cross-fade slot outside the band has probability exactly 0. Mapping its expert id to -1 drops the (token, expert)
pair from the live-row permute; the fixed-width path computes the pair and adds exactly +0 to the token's fp64
combine. Both must give the same bits for every token (including tokens whose only local slots are zero-weight),
every EP rank's expert slice, limbs (EP-invariant combine) on and off, down-scale, and under CUDA-graph replay.
Also reports the live-row reduction and the kernel time of both paths (informational).
"""

import os

os.environ.setdefault("CUBLASLT_WORKSPACE_SIZE", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":0:0")

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _bits(t):
    return t.view(torch.int64) if t.dtype == torch.float64 else (
        t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int16))


def _cf_inputs(t, valid, k, c, n_experts, gen, zero_frac=0.6):
    """Cross-fade-like routing: k stock slots (nonzero) + c candidates, a fraction of them exactly zero."""
    topk = k + c
    r = torch.stack([torch.randperm(n_experts, device="cuda", generator=gen)[:topk] for _ in range(t)]).to(torch.int64)
    p = torch.rand(t, topk, device="cuda", generator=gen) + 0.05
    zero = torch.rand(t, c, device="cuda", generator=gen) < zero_frac
    p[:, k:] = torch.where(zero, torch.zeros_like(p[:, k:]), p[:, k:])
    p[0, k:] = 0.0  # a token with every candidate at zero weight
    p = p / p.sum(-1, keepdim=True)
    r[valid - 3: valid] = -1  # CUDA-graph padding tokens
    return p, r


def _skip(p, r):
    return torch.where(p != 0, r, torch.full_like(r, -1))


@pytest.mark.parametrize("activation", ["relu2", "swiglu"])
@pytest.mark.parametrize("limbs", [False, True])
@pytest.mark.parametrize("down_scale", [0, 3])
@pytest.mark.parametrize("ep,valid,hint", [(1, 64, 64), (4, 200, None), (4, 37, 8), (8, 512, None)])
def test_engine_skip_bitwise(activation, limbs, down_scale, ep, valid, hint):
    from nemo_rl.models.megatron import fp16_moe as fm

    gen = torch.Generator(device="cuda").manual_seed(valid + 31 * ep)
    t, kdim, f, k, c, e = 512, 512, 384, 6, 4, 16
    fn = 2 * f if activation == "swiglu" else f
    w1 = (torch.randn(e, fn, kdim, device="cuda", generator=gen) * 0.05).to(torch.float16)
    w2 = (torch.randn(e, kdim, f, device="cuda", generator=gen) * 0.05).to(torch.float16)
    x16 = torch.randn(t, kdim, device="cuda", generator=gen).to(torch.float16)
    p, r = _cf_inputs(t, valid, k, c, e * ep, gen)
    vt = torch.tensor([valid], dtype=torch.int32, device="cuda")
    for rank in range(ep):
        start = e * rank
        a = fm.engine_experts(x16, p, r, vt, w1, w2, start, ep, activation, tokens_hint=hint,
                              down_scale=down_scale, limbs=limbs)
        b = fm.engine_experts(x16, p, _skip(p, r), vt, w1, w2, start, ep, activation, tokens_hint=hint,
                              down_scale=down_scale, limbs=limbs)
        assert torch.equal(_bits(a[:valid]), _bits(b[:valid])), (rank, activation, limbs, down_scale)


def test_engine_skip_cuda_graph_replay():
    from nemo_rl.models.megatron import fp16_moe as fm

    gen = torch.Generator(device="cuda").manual_seed(11)
    t, kdim, f, k, c, e, ep = 1024, 512, 384, 6, 4, 8, 4
    w1 = (torch.randn(e, f, kdim, device="cuda", generator=gen) * 0.05).to(torch.float16)
    w2 = (torch.randn(e, kdim, f, device="cuda", generator=gen) * 0.05).to(torch.float16)
    x16 = torch.randn(t, kdim, device="cuda", generator=gen).to(torch.float16)
    p, r = _cf_inputs(t, 64, k, c, e * ep, gen)
    vt = torch.tensor([64], dtype=torch.int32, device="cuda")
    rs = _skip(p, r)
    fm.engine_experts(x16, p, rs, vt, w1, w2, 0, ep, "relu2", tokens_hint=64)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rs_g = torch.where(p != 0, r, torch.full_like(r, -1))
        out = fm.engine_experts(x16, p, rs_g, vt, w1, w2, 0, ep, "relu2", tokens_hint=64)
    for valid in (5, 64, 300):
        p2, r2 = _cf_inputs(t, valid, k, c, e * ep, gen)
        p.copy_(p2); r.copy_(r2); vt.fill_(valid)
        graph.replay()
        ref = fm.engine_experts(x16, p, r, vt, w1, w2, 0, ep, "relu2", tokens_hint=64)
        assert torch.equal(_bits(out[:valid]), _bits(ref[:valid])), valid


@pytest.mark.parametrize("shape", ["super_latent", "nano"])
def test_engine_skip_speed_report(shape):
    """Informational: live rows and kernel time, fixed width vs skip (k 22 + c 8, e 2.7 live extra: Super m .01)."""
    from nemo_rl.models.megatron import fp16_moe as fm

    gen = torch.Generator(device="cuda").manual_seed(5)
    if shape == "super_latent":
        t, kdim, f, k, c, e, ep, zf = 4096, 1024, 2688, 22, 8, 128, 4, 1 - 2.7 / 8
    else:
        t, kdim, f, k, c, e, ep, zf = 4096, 2688, 1856, 6, 4, 32, 4, 1 - 0.55 / 4
    w1 = (torch.randn(e, f, kdim, device="cuda", generator=gen) * 0.02).to(torch.float16)
    w2 = (torch.randn(e, kdim, f, device="cuda", generator=gen) * 0.02).to(torch.float16)
    x16 = torch.randn(t, kdim, device="cuda", generator=gen).to(torch.float16)
    p, r = _cf_inputs(t, t, k, c, e * ep, gen, zero_frac=zf)
    r[t - 3:] = torch.randint(0, e * ep, (3, k + c), device="cuda", generator=gen)
    vt = torch.tensor([t], dtype=torch.int32, device="cuda")
    rs = _skip(p, r)
    times = {}
    for name, rr in (("fixed", r), ("skip", rs)):
        for _ in range(3):
            fm.engine_experts(x16, p, rr, vt, w1, w2, 0, ep, "relu2")
        torch.cuda.synchronize()
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        ev[0].record()
        for _ in range(20):
            fm.engine_experts(x16, p, rr, vt, w1, w2, 0, ep, "relu2")
        ev[1].record(); torch.cuda.synchronize()
        times[name] = ev[0].elapsed_time(ev[1]) / 20
    local = lambda rr: int(((rr >= 0) & (rr < e)).sum())  # noqa: E731
    print(f"\n[engine skip {shape}] local pairs fixed {local(r)} skip {local(rs)} "
          f"({local(rs) / local(r):.3f}); ms fixed {times['fixed']:.3f} skip {times['skip']:.3f} "
          f"({times['skip'] / times['fixed']:.3f})")
