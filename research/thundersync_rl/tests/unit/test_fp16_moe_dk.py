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
"""The engine's relu2 expert path (fp16_moe_dk.py) gives the generic chain's bits.

- every dk tile (incl. the operand swap) == ``grouped_gemm_fp16w`` == per-expert cuBLAS (the learner's GEMM), at
  Super and Ultra expert shapes;
- the fused fc1 epilogue == ``relu2_probs_fwd`` on the fc1 output; the dk fc1 -> relu2 -> fc2 chain == the
  learner's ``grouped_linear`` -> eager relu2 -> ``grouped_linear``;
- the slot-map combine == ``unpermute_fp64`` for every combine mode;
- ``engine_experts`` (relu2: dk) == ``engine_experts_generic``, at every table tile and decode / prefill sizes; a
  token's output does not depend on the other tokens of the batch;
- CUDA-graph capture with the live token count changed between replays;
- relu2 calls dispatch to dk, swiglu and the fc2 hi/lo raise to the generic chain.
"""

import os

os.environ.setdefault("CUBLASLT_WORKSPACE_SIZE", "0")  # cuBLAS without split-K (BI)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":0:0")

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages, SWAP, programs/SM): the table's tiles plus both operand orders at
# every BLOCK_M.
_EXTRA = [
    (16, 64, 128, 4, 4, False, 4),
    (16, 32, 256, 4, 3, True, 8),
    (32, 64, 128, 4, 4, True, 2),
    (32, 128, 64, 4, 3, False, 4),
    (64, 64, 64, 4, 4, True, 2),
    (64, 128, 64, 4, 4, False, 1),
]


# Every relu2 expert shape (N, K) of the model configs the engine runs: fc1 [ffn, h_moe] and fc2 [h_moe, ffn],
# h_moe = moe_latent_size or hidden_size. Nano (hidden 2688, ffn 1856: K = 1856 is not a multiple of 128),
# Super (latent 1024, ffn 2688), Ultra (latent 2048, ffn 5120), synth-latent test model (latent 256, ffn 512).
EXPERT_SHAPES = {
    "nano": (2688, 1856),
    "super": (1024, 2688),
    "ultra": (2048, 5120),
    "synth": (256, 512),
}
_NK = [nk for h, f in EXPERT_SHAPES.values() for nk in ((f, h), (h, f))]


def _tiles(k=None):
    """The table's tiles plus _EXTRA (k unused: every tile runs at every K, ragged K tiles included)."""
    from nemo_rl.models.megatron import fp16_moe_dk as dk

    seen = []
    for _, c1, c2 in dk._DK_TABLE:
        for c in (c1, c2):
            if tuple(c) not in seen:
                seen.append(tuple(c))
    for c in _EXTRA:
        if c not in seen:
            seen.append(c)
    return seen


def _bits(t):
    return t.view({torch.float32: torch.int32, torch.float16: torch.int16, torch.float64: torch.int64}[t.dtype])


def _segments(rows, counts, bm):
    offs, live, start = [], [], 0
    for c in counts:
        live += list(range(start, start + c))
        start += -(-c // bm) * bm
        offs.append(start)
    m_pad = max(start, bm) + 2 * bm
    x = torch.full((m_pad, rows.shape[1]), float("nan"), dtype=rows.dtype, device="cuda")
    x[live] = rows  # alignment padding rows stay NaN: computed, never read
    return x, torch.tensor(offs, dtype=torch.int32, device="cuda"), torch.tensor(live, device="cuda")


@pytest.mark.parametrize("n,k", _NK)
def test_dk_gemm_every_tile_matches_engine_kernel_and_cublas(n, k):
    from nemo_rl.models.megatron import fp16_moe as fm
    from nemo_rl.models.megatron import fp16_moe_dk as dk

    gen = torch.Generator(device="cuda").manual_seed(n)
    e = 8
    w16 = (torch.randn(e, n, k, device="cuda", generator=gen) * 0.03).to(torch.float16)
    counts = [3, 0, 17, 1, 0, 40, 2, 9]
    rows = (torch.randn(sum(counts), k, device="cuda", generator=gen) * 3).to(torch.float16)
    rows[0] = 0.0
    rows[1, ::7] = 6e4  # large magnitudes (fp32 accumulation range)
    ref = torch.cat(
        [
            torch.mm(r, w16[i].t(), out_dtype=torch.float32)
            for i, r in enumerate(torch.split(rows, counts))
        ]
    )
    for c in _tiles(k):
        bm = c[0]
        x, offs, live = _segments(rows, counts, bm)
        y = dk.dk_grouped_gemm(x, w16, offs, c, rows_hint=sum(counts) + e * bm)[live]
        assert torch.equal(_bits(y), _bits(ref)), c
        y_eng = fm.grouped_gemm_fp16w(x, w16, offs, (bm, 64, 64, 4, 4))[live]
        assert torch.equal(_bits(y), _bits(y_eng)), c
        # a small grid hint loops more, never changes a value
        y1 = dk.dk_grouped_gemm(x, w16, offs, c, rows_hint=1)[live]
        assert torch.equal(_bits(y1), _bits(ref)), c


@pytest.mark.parametrize("model", list(EXPERT_SHAPES))
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("scale", [None, 2.0**-3])
def test_dk_fused_relu2_epilogue_matches_kernel_chain(out_dtype, scale, model):
    from nemo_rl.models.megatron import fp16_moe as fm
    from nemo_rl.models.megatron import fp16_moe_dk as dk

    gen = torch.Generator(device="cuda").manual_seed(7)
    k, n = EXPERT_SHAPES[model]
    e = 6
    w16 = (torch.randn(e, n, k, device="cuda", generator=gen) * 0.03).to(torch.float16)
    counts = [5, 1, 0, 33, 16, 2]
    rows = (torch.randn(sum(counts), k, device="cuda", generator=gen) * 2).to(torch.float16)
    for c in _tiles():
        bm = c[0]
        x, offs, live = _segments(rows, counts, bm)
        p = torch.rand(x.shape[0], device="cuda", generator=gen) * 0.3
        h = fm.grouped_gemm_fp16w(x, w16, offs, (bm, 64, 64, 4, 4))
        ref = fm.relu2_probs_fwd(h, p, out_dtype, valid_rows=offs[-1:], prob_scale=scale)[live]
        got = dk.dk_grouped_gemm(x, w16, offs, c, probs=p, out_dtype=out_dtype, prob_scale=scale)[live]
        assert torch.equal(_bits(got), _bits(ref)), c


@pytest.mark.parametrize("model", list(EXPERT_SHAPES))
def test_dk_chain_matches_learner_grouped_linear(model):
    """dk fc1 (+ fused relu2 x p) -> fc2 == the learner's te_gt fc1 -> eager relu2 -> fc2 (fp16 expert params)."""
    from nemo_rl.models.megatron import fp16_moe as fm
    from nemo_rl.models.megatron import fp16_moe_dk as dk

    gen = torch.Generator(device="cuda").manual_seed(3)
    h_dim, f_dim = EXPERT_SHAPES[model]
    e = 4
    w1 = [(torch.randn(f_dim, h_dim, device="cuda", generator=gen) * 0.03).to(torch.float16) for _ in range(e)]
    w2 = [(torch.randn(h_dim, f_dim, device="cuda", generator=gen) * 0.03).to(torch.float16) for _ in range(e)]
    counts = [7, 0, 21, 3]
    x = torch.randn(sum(counts), h_dim, device="cuda", generator=gen)
    p = torch.rand(sum(counts), device="cuda", generator=gen) * 0.3
    with torch.no_grad():
        h = fm.grouped_linear(x, counts, w1)
        a = fm._relu2_eager(h, p).to(torch.float16)  # the engine's fc2 operand (fp16 cast at fc2)
        y_learn = fm.grouped_linear(a.float(), counts, w2)
    w1s = torch.stack(w1).contiguous()
    w2s = torch.stack(w2).contiguous()
    for c in _tiles():
        bm = c[0]
        xs, offs, live = _segments(x.half(), counts, bm)
        ps = torch.zeros(xs.shape[0], device="cuda")
        ps[live] = p
        a_dk = dk.dk_grouped_gemm(xs, w1s, offs, c, probs=ps, out_dtype=torch.float16)
        y_dk = dk.dk_grouped_gemm(a_dk, w2s, offs, c)[live]
        assert torch.equal(_bits(a_dk[live]), _bits(a)), c
        assert torch.equal(_bits(y_dk), _bits(y_learn)), c


def _routing(t, e_local, ep, topk=30, real=22, seed=0):
    gen = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(e_local * ep, generator=gen)[:topk] for _ in range(t)])
    probs = torch.rand(t, topk, generator=gen) * 0.2 + 0.01
    probs[:, real:] = 0.0
    ids = torch.where(probs != 0, ids, torch.full_like(ids, -1))
    return ids.cuda(), probs.cuda()


@pytest.mark.parametrize("limbs,round16,scale", [(0, False, None), (1, False, None), (2, False, None), (0, True, 3)])
def test_dk_unpermute_matches(limbs, round16, scale):
    """The slot-map combine == unpermute_fp64."""
    from nemo_rl.models.megatron import fp16_moe as fm
    from nemo_rl.models.megatron import fp16_moe_dk as dk

    t, e, h = 50, 64, 1024
    ids, probs = _routing(t, e, 2)
    x = torch.randn(t, h, device="cuda").half()
    valid = torch.tensor([t - 3], dtype=torch.int32, device="cuda")
    xp, pp, offs, inv = fm.permute_live(x, probs, ids, 0, e, valid, 16)
    y = torch.randn(xp.shape[0], h, device="cuda") * 10
    y[::5] *= 1e-6  # mixed exponents: fp64 adds are order-sensitive in general
    kw = dict(tokens_hint=t, y_scale=None if scale is None else 2.0**scale, limbs=limbs, round16=round16)
    ref = fm.unpermute_fp64(y, inv, valid, **kw)
    n = t - 3
    # slot-map combine on the rows dk_route assigns (same expert outputs moved to its row order)
    src, pp2, offs2, slots = dk.dk_route(ids, probs, valid, 0, e, 16)
    y2 = torch.zeros(src.shape[0], h, device="cuda")
    lid = ids[:n] - 0
    local = (lid >= 0) & (lid < e)
    tok = torch.arange(n, device="cuda")[:, None].expand_as(lid)[local]
    y2[slots[:n][local].long()] = y[inv[tok, lid[local]].long()]
    got2 = dk.dk_unpermute_slots(y2, slots, ids, 0, valid, **kw)
    assert torch.equal(_bits(got2[:n]), _bits(ref[:n]))


@pytest.mark.parametrize("model", list(EXPERT_SHAPES))
@pytest.mark.parametrize("ep,limbs,skip0", [(4, False, False), (4, True, True), (8, False, True)])
def test_engine_experts_dk_matches_and_is_row_independent(ep, limbs, skip0, model):
    from nemo_rl.models.megatron import fp16_moe as fm
    from nemo_rl.models.megatron import fp16_moe_dk as dk

    e = 512 // ep // 4  # 4x fewer local experts than Super (test time); same per-expert shapes
    gen = torch.Generator(device="cuda").manual_seed(ep)
    h_dim, f_dim = EXPERT_SHAPES[model]
    w1 = (torch.randn(e, f_dim, h_dim, device="cuda", generator=gen) * 0.03).half()
    w2 = (torch.randn(e, h_dim, f_dim, device="cuda", generator=gen) * 0.03).half()
    t = 96
    ids, probs = _routing(t, e, ep, seed=ep)
    if skip0:  # zero-weight slots keep their ids: the kernels must drop them
        gen2 = torch.Generator().manual_seed(1)
        ids[:, 22:] = torch.stack([torch.randperm(e * ep, generator=gen2)[:8] for _ in range(t)]).cuda()
        ids[:, 22:] = torch.where(
            (ids[:, 22:, None] == ids[:, None, :22]).any(-1), torch.full_like(ids[:, 22:], -1), ids[:, 22:]
        )
    x = (torch.randn(t, h_dim, device="cuda", generator=gen) * 2).half()
    valid = torch.tensor([t], dtype=torch.int32, device="cuda")
    kw = dict(tokens_hint=t, limbs=limbs)
    ref_ids = torch.where(probs != 0, ids, torch.full_like(ids, -1)) if skip0 else ids
    ref = fm.engine_experts_generic(x, probs, ref_ids, valid, w1, w2, 0, ep, "relu2", **kw)
    ref2 = fm.engine_experts_generic(x, probs, ids, valid, w1, w2, 0, ep, "relu2", skip_zero_probs=skip0, **kw)
    assert torch.equal(_bits(ref2[:t]), _bits(ref[:t]))
    tiles = [(c1, c2) for _, c1, c2 in dk._DK_TABLE] + [(c, c) for c in _tiles()]
    for c1, c2 in tiles:
        dk.set_config_override(c1, c2)
        try:
            run = lambda xx, pp, ii, vv, **k: fm.engine_experts(  # noqa: E731
                xx, pp, ii, vv, w1, w2, 0, ep, "relu2", skip_zero_probs=skip0, **k
            )
            got = run(x, probs, ids, valid, **kw)
            assert torch.equal(_bits(got[:t]), _bits(ref[:t])), (c1, c2)
            # the first 9 tokens alone (another batch size and composition): same bits per token
            sub = torch.tensor([9], dtype=torch.int32, device="cuda")
            got9 = run(x[:9], probs[:9], ids[:9], sub, tokens_hint=9, limbs=limbs)
            assert torch.equal(_bits(got9[:9]), _bits(ref[:9])), (c1, c2)
            # tokens reversed: each token's row unchanged
            rev = torch.arange(t - 1, -1, -1, device="cuda")
            got_r = run(x[rev], probs[rev], ids[rev], valid, **kw)
            assert torch.equal(_bits(got_r[:t][rev]), _bits(ref[:t])), (c1, c2)
        finally:
            dk.set_config_override(None, None)


def test_dk_route_offsets_match_mcore_and_cover_every_pair():
    from nemo_rl.models.megatron import fp16_moe as fm
    from nemo_rl.models.megatron import fp16_moe_dk as dk

    t, e, ep = 300, 64, 8
    ids, probs = _routing(t, e, ep, seed=9)
    valid = torch.tensor([t - 7], dtype=torch.int32, device="cuda")
    x = torch.randn(t, 64, device="cuda").half()
    for bm in (16, 32, 64):
        for start in (0, e * 3):
            _, _, offs_ref, _ = fm.permute_live(x, probs, ids, start, e, valid, bm)
            src, pp, offs, slots = dk.dk_route(ids, probs, valid, start, e, bm)
            assert torch.equal(offs, offs_ref.to(offs.dtype))
            n = t - 7
            lid = ids[:n] - start
            local = (lid >= 0) & (lid < e)
            assert torch.equal(slots[:n] >= 0, local)
            rows = slots[:n][local].long()
            assert rows.unique().numel() == rows.numel()
            tok = torch.arange(n, device="cuda")[:, None].expand(n, ids.shape[1])[local]
            assert torch.equal(src[rows].long(), tok)
            assert torch.equal(pp[rows], probs[:n][local])
            # each row inside its expert's segment
            seg_end = offs.long()[lid[local]]
            seg_start = seg_end - torch.where(
                torch.bincount(lid[local], minlength=e) > 0,
                (torch.bincount(lid[local], minlength=e) + bm - 1) // bm * bm, 0)[lid[local]]
            assert bool(((rows >= seg_start) & (rows < seg_end)).all())
            used = int(offs[-1])
            assert int((src[:used] >= 0).sum()) == rows.numel()
    # many programs racing on the counters / the last-program prefix: same offsets every time
    big_ids, big_probs = _routing(4000, e, ep, seed=4)
    vbig = torch.tensor([4000], dtype=torch.int32, device="cuda")
    xb = torch.zeros(4000, 8, device="cuda").half()
    _, _, offs_ref, _ = fm.permute_live(xb, big_probs, big_ids, 0, e, vbig, 16)
    for _ in range(50):
        _, _, offs, _ = dk.dk_route(big_ids, big_probs, vbig, 0, e, 16)
        assert torch.equal(offs, offs_ref.to(offs.dtype))


@pytest.mark.parametrize("model", list(EXPERT_SHAPES))
def test_engine_experts_dk_cuda_graph_live_count_changes(model):
    from nemo_rl.models.megatron import fp16_moe as fm

    e, ep, t = 32, 4, 128
    gen = torch.Generator(device="cuda").manual_seed(11)
    h_dim, f_dim = EXPERT_SHAPES[model]
    w1 = (torch.randn(e, f_dim, h_dim, device="cuda", generator=gen) * 0.03).half()
    w2 = (torch.randn(e, h_dim, f_dim, device="cuda", generator=gen) * 0.03).half()
    ids, probs = _routing(t, e, ep, seed=5)
    x = (torch.randn(t, h_dim, device="cuda", generator=gen) * 2).half()
    valid = torch.tensor([t], dtype=torch.int32, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fm.engine_experts(x, probs, ids, valid, w1, w2, 0, ep, "relu2", tokens_hint=t)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fm.engine_experts(x, probs, ids, valid, w1, w2, 0, ep, "relu2", tokens_hint=t)
    for n in (t, 1, 37, 2, t):
        valid.fill_(n)
        g.replay()
        torch.cuda.synchronize()
        ref = fm.engine_experts_generic(x, probs, ids, valid, w1, w2, 0, ep, "relu2", tokens_hint=t)
        assert torch.equal(_bits(out[:n]), _bits(ref[:n])), n


def test_engine_experts_dispatch(monkeypatch):
    """relu2 -> dk (any size); swiglu and the fc2 hi/lo raise -> the generic chain."""
    from nemo_rl.models.megatron import fp16_moe as fm
    from nemo_rl.models.megatron import fp16_moe_dk as dk

    calls = []
    monkeypatch.setattr(dk, "engine_experts_dk", lambda *a, **k: calls.append("dk") or "dk")
    monkeypatch.setattr(fm, "engine_experts_generic", lambda *a, **k: calls.append("generic") or "generic")
    ids, probs = _routing(4, 8, 4)
    x, w1, w2 = torch.zeros(4, 32).half(), torch.zeros(8, 64, 32).half(), torch.zeros(8, 32, 64).half()
    valid = torch.tensor([4], dtype=torch.int32)
    assert fm.engine_experts(x, probs, ids, valid, w1, w2, 0, 4, "relu2") == "dk"
    assert fm.engine_experts(x, probs, ids, valid, w1, w2, 0, 4, "relu2", tokens_hint=1 << 20) == "dk"
    assert fm.engine_experts(x, probs, ids, valid, w1, w2, 0, 4, "relu2", fc2_hilo=True) == "generic"
    assert fm.engine_experts(x, probs, ids, valid, w1, w2, 0, 4, "swiglu") == "generic"
    assert calls == ["dk", "dk", "generic", "generic"]


@pytest.mark.parametrize("model", list(EXPERT_SHAPES))
@pytest.mark.parametrize("ep,rows_per_rank", [(8, 1280), (32, 640)])
def test_engine_experts_dk_prefill_sizes_match(ep, rows_per_rank, model):
    """Prefill-sized calls (the large-BLOCK_M table rows) == the generic chain."""
    from nemo_rl.models.megatron import fp16_moe as fm

    e = 512 // ep // 4
    gen = torch.Generator(device="cuda").manual_seed(ep + 1)
    h_dim, f_dim = EXPERT_SHAPES[model]
    w1 = (torch.randn(e, f_dim, h_dim, device="cuda", generator=gen) * 0.03).half()
    w2 = (torch.randn(e, h_dim, f_dim, device="cuda", generator=gen) * 0.03).half()
    t = rows_per_rank
    ids, probs = _routing(t, e, ep, seed=ep + 2)
    x = (torch.randn(t, h_dim, device="cuda", generator=gen) * 2).half()
    valid = torch.tensor([t], dtype=torch.int32, device="cuda")
    ref = fm.engine_experts_generic(x, probs, ids, valid, w1, w2, 0, ep, "relu2", tokens_hint=t)
    got = fm.engine_experts(x, probs, ids, valid, w1, w2, 0, ep, "relu2", tokens_hint=t)
    assert torch.equal(_bits(got[:t]), _bits(ref[:t]))


def _peak_delta(fn):
    """(result, peak bytes allocated above the starting point while ``fn`` runs); graph-pool blocks count."""
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    res = fn()
    torch.cuda.synchronize()
    return res, torch.cuda.max_memory_allocated() - base


@pytest.mark.parametrize("model", ["nano", "super"])
def test_engine_row_workspace_shared_and_worst_case(model):
    """One per-row workspace for every layer, graph and eager call, sized for the whole gathered buffer.

    The 11-05 Nano 16k engine-init OOM: every capture allocated the fc1 / fc2 outputs for the buffer's
    max_tokens x EP x min(top-k, local experts) rows in its private pool (a 20.69 GiB fc2 output). Bucket-sized
    rows are not a bound: under the NVLS dispatcher a rank replaying a small decode graph receives its EP peers'
    eager prefill tokens (an illegal memory access in the 9f18b05b gate). Checked here:
    - a capture allocates no per-row buffer (the workspace exists once);
    - a graph captured at a 64-token bucket replayed with every buffer token live == the eager generic chain;
    - eager calls after the captures == the generic chain.
    """
    from nemo_rl.models.megatron import fp16_moe as fm
    from nemo_rl.models.megatron import fp16_moe_dk as dk

    e, ep, topk, buf = 16, 8, 14, 4096  # Nano's 16 local experts at EP 8, cross-fade top-6+8
    gen = torch.Generator(device="cuda").manual_seed(21)
    h_dim, f_dim = EXPERT_SHAPES[model]
    w1 = (torch.randn(e, f_dim, h_dim, device="cuda", generator=gen) * 0.03).half()
    w2 = (torch.randn(e, h_dim, f_dim, device="cuda", generator=gen) * 0.03).half()
    ids, probs = _routing(buf, e, ep, topk=topk, real=6, seed=13)
    loc = torch.arange(buf, device="cuda") % e  # a local pair for (almost) every token
    ids[:, 0] = torch.where((ids[:, 1:] == loc[:, None]).any(1), ids[:, 0], loc)
    x = (torch.randn(buf, h_dim, device="cuda", generator=gen) * 2).half()
    valid = torch.tensor([64], dtype=torch.int32, device="cuda")
    for limbs in (False, True):
        out = torch.empty(buf, 2 * h_dim if limbs else h_dim, dtype=torch.float64 if limbs else torch.float32,
                          device="cuda")  # the engine passes its RSV rows (allocated once, outside graphs)
        run = lambda hint: fm.engine_experts(  # noqa: E731
            x, probs, ids, valid, w1, w2, 0, ep, "relu2", tokens_hint=hint, out=out, limbs=limbs
        )
        ref_of = lambda: fm.engine_experts_generic(  # noqa: E731
            x, probs, ids, valid, w1, w2, 0, ep, "relu2", limbs=limbs
        )
        run(64)  # eager: creates the workspace
        n_ws = len(dk._ROW_WS)
        graphs = []
        for bucket in (64, 512):
            valid.fill_(bucket)
            run(bucket)  # eager warmup at the bucket's tiles (Triton compiles outside the capture)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()

            def capture():
                with torch.cuda.graph(g):
                    run(bucket)

            _, grown = _peak_delta(capture)
            assert grown < 8 << 20, (bucket, grown)  # no per-row buffer inside the capture
            graphs.append(g)
        assert len(dk._ROW_WS) == n_ws
        for g in graphs:
            for n in (buf, 3000, 64, 1, 0):  # live tokens beyond the captured bucket: the NVLS mix
                valid.fill_(n)
                g.replay()
                torch.cuda.synchronize()
                got = out[:n].clone()
                assert torch.equal(_bits(got), _bits(ref_of()[:n])), (limbs, n)
        for n in (buf, 777):
            valid.fill_(n)
            run(n)
            torch.cuda.synchronize()
            assert torch.equal(_bits(out[:n]), _bits(ref_of()[:n])), (limbs, n)
        del graphs


def test_reserved_row_workspace_released_during_training_and_restored():
    """The engine reserves the workspace in the engine-scratch region (torch_memory_saver, the TMS=1 overlay): pausing
    it returns its physical memory (the colocated learner's training peak), resuming maps it back at the same
    addresses, and a graph captured before the pause replays to the same bits after the resume."""
    try:
        import torch_memory_saver  # noqa: F401
    except ImportError:
        pytest.skip("torch_memory_saver (see docs/colo/HANDOFF.md, overlays) not on PYTHONPATH")
    from nemo_rl.models.megatron import fp16_moe as fm
    from nemo_rl.models.megatron import fp16_moe_dk as dk
    from nemo_rl.models.megatron import memory_saver as ms

    e, ep, topk, buf = 16, 8, 14, 8192
    h_dim, f_dim = 2688, 1832  # an ffn of its own: a fresh workspace key for this test
    gen = torch.Generator(device="cuda").manual_seed(31)
    w1 = (torch.randn(e, f_dim, h_dim, device="cuda", generator=gen) * 0.03).half()
    w2 = (torch.randn(e, h_dim, f_dim, device="cuda", generator=gen) * 0.03).half()
    ids, probs = _routing(buf, e, ep, topk=topk, real=6, seed=17)
    x = (torch.randn(buf, h_dim, device="cuda", generator=gen) * 2).half()
    valid = torch.tensor([512], dtype=torch.int32, device="cuda")
    out = torch.empty(buf, h_dim, device="cuda")
    nbytes, chunk = dk.reserve_row_workspace(torch.device("cuda"), buf, topk, e, f_dim, h_dim, colocated=True)
    assert chunk == h_dim  # released during training: one fc2 pass
    assert nbytes == dk.workspace_rows(buf, topk, e) * (8 + 2 * f_dim + 4 * h_dim)
    ws_ptr = dk._ROW_WS[dk._ws_key(torch.device("cuda"), f_dim, h_dim)]["y"].data_ptr()
    run = lambda: fm.engine_experts(x, probs, ids, valid, w1, w2, 0, ep, "relu2", tokens_hint=512, out=out)  # noqa
    run()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    valid.fill_(buf)  # every token live (the NVLS worst case the reservation covers)
    g.replay()
    torch.cuda.synchronize()
    ref = out.clone()
    free0 = torch.cuda.mem_get_info()[0]
    ms.pause_engine_scratch()
    freed = torch.cuda.mem_get_info()[0] - free0
    assert freed >= 0.9 * nbytes, (freed, nbytes)
    out.zero_()
    ms.resume_engine_scratch()
    assert dk._ROW_WS[dk._ws_key(torch.device("cuda"), f_dim, h_dim)]["y"].data_ptr() == ws_ptr
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(_bits(out), _bits(ref))
    assert torch.equal(_bits(out), _bits(fm.engine_experts_generic(x, probs, ids, valid, w1, w2, 0, ep, "relu2")))
    del g


def test_fc2_passes_only_when_the_workspace_stays_resident():
    """Review E1 (batch 19): one fc2 pass when the workspace is released during training (torch_memory_saver) or the
    engine is not colocated; column passes (fc2 output <= fc1 output) when it would stay resident through a colocated
    learner's training. Both give the generic chain's bits. Review E5: a reserved workspace never grows."""
    from nemo_rl.models.megatron import fp16_moe as fm
    from nemo_rl.models.megatron import fp16_moe_dk as dk
    from nemo_rl.models.megatron import memory_saver as ms

    assert {m: dk.fc2_column_chunk(h, f) for m, (h, f) in EXPERT_SHAPES.items()} == {
        "nano": 896, "super": 1024, "ultra": 2048, "synth": 256}
    e, ep, topk, buf = 16, 8, 14, 1024
    h_dim, f_dim = 2688, 1848  # a workspace key of its own
    gen = torch.Generator(device="cuda").manual_seed(41)
    w1 = (torch.randn(e, f_dim, h_dim, device="cuda", generator=gen) * 0.03).half()
    w2 = (torch.randn(e, h_dim, f_dim, device="cuda", generator=gen) * 0.03).half()
    ids, probs = _routing(buf, e, ep, topk=topk, real=6, seed=19)
    x = (torch.randn(buf, h_dim, device="cuda", generator=gen) * 2).half()
    valid = torch.tensor([buf], dtype=torch.int32, device="cuda")
    ref = fm.engine_experts_generic(x, probs, ids, valid, w1, w2, 0, ep, "relu2", limbs=True)
    have = ms.HAVE_TORCH_MEMORY_SAVER
    try:
        for tms_on, colocated, want in ((False, True, 896), (False, False, h_dim), (have, True, h_dim if have else 896)):
            ms.HAVE_TORCH_MEMORY_SAVER = tms_on
            dk._ROW_WS.pop(dk._ws_key(torch.device("cuda"), f_dim, h_dim), None)
            _, chunk = dk.reserve_row_workspace(torch.device("cuda"), buf, topk, e, f_dim, h_dim, colocated=colocated)
            assert chunk == want, (tms_on, colocated, chunk)
            got = fm.engine_experts(x, probs, ids, valid, w1, w2, 0, ep, "relu2", limbs=True)
            assert torch.equal(_bits(got[:buf]), _bits(ref[:buf])), (tms_on, colocated)
    finally:
        ms.HAVE_TORCH_MEMORY_SAVER = have
    with pytest.raises(RuntimeError, match="reserved for"):  # E5: never grows past the reservation
        dk.row_workspace(torch.device("cuda"), dk.workspace_rows(2 * buf, topk, e), f_dim, h_dim)
