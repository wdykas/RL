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
"""Tests of the high-precision batch-invariant attention prototype.

GPU tests check bitwise equality of the training/prefill forward and engine
decode on a paged hp KV cache (Qwen3-30B-A3B attention shape: 32 q heads,
4 KV heads, head dim 128) and the error against an fp64 reference.
"""

import math

import pytest
import torch

from nemo_rl.models.megatron import hp_attention as hp

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
HQ, HKV, D = 32, 4, 128


# ---------------------------------------------------------------- CPU tests


def test_pow2_exact():
    e = torch.arange(-126, 128)
    assert torch.equal(hp.pow2(e), torch.ldexp(torch.ones(e.numel()), e))


@pytest.mark.parametrize("mag", [1e-6, 1e-3, 1.0, 37.0, 3e3])
def test_split_rows_precision_and_range(mag):
    torch.manual_seed(0)
    x = torch.randn(64, 4, 128) * mag
    hi, lo, inv = hp.split_rows(x, -40, 40)
    amax = hi.float().abs().amax(-1)
    assert ((amax >= 1) & (amax <= 2)).all()  # row amax prescaled into [1, 2]
    recon = (hi.double() + lo.double()) * inv.double().unsqueeze(-1)
    rel = ((recon - x.double()).abs() / x.double().abs().amax(-1, keepdim=True)).max()
    assert rel < 2**-23  # ~22 significant bits relative to the row amax
    # inverse scales are powers of two
    m, _ = torch.frexp(inv)
    assert torch.all(m == 0.5)


def test_split_rows_row_local():
    torch.manual_seed(1)
    x = torch.randn(10, 4, 128)
    full = hp.split_rows(x, -40, 40)
    part = hp.split_rows(x[3:4], -40, 40)
    for a, b in zip(full, part):
        assert torch.equal(a[3:4], b)


def test_build_tiles():
    q_seq = torch.tensor([0, 0, 0, 1, 1, 0])
    q_pos = torch.tensor([0, 1, 9, 5, 7, 8])
    ts, t0, tq = hp.build_tiles(q_seq, q_pos, 8)
    assert ts.tolist() == [0, 0, 1]
    assert t0.tolist() == [0, 8, 0]
    assert tq[0].tolist() == [0, 1, -1, -1, -1, -1, -1, -1]
    assert tq[1].tolist() == [5, 2, -1, -1, -1, -1, -1, -1]
    assert tq[2].tolist() == [-1, -1, -1, -1, -1, 3, -1, 4]


# ---------------------------------------------------------------- GPU tests


def _inputs(lens, seed=0, dev="cuda", hkv=HKV):
    """Qwen-like post-RoPE q/k (scores up to ~+-30, a sink key at position 0)
    and V rows spanning 1e-4 .. 50 in magnitude."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = sum(lens)
    q = torch.randn(t, HQ, D, generator=g) * 1.5
    k = torch.randn(t, hkv, D, generator=g) * 1.5
    v = torch.randn(t, hkv, D, generator=g)
    v = v * torch.exp(torch.randn(t, hkv, 1, generator=g) * 2.0).clamp(1e-4, 50)
    starts = [0] + list(torch.tensor(lens).cumsum(0)[:-1].tolist())
    sink = torch.randn(hkv, D, generator=g)
    for s in starts:  # attention-sink-like first key
        k[s] = sink * 3.0
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)
    return q.to(dev), k.to(dev), v.to(dev), cu.to(dev)


def _paged_cache(lens, k, v, cu, cfg, page_size, seed, nan_fill=True):
    """Cache with pages in a random order (engine-like), unused slots NaN."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    pages = [(n + page_size - 1) // page_size for n in lens]
    total = sum(pages) + 5
    perm = torch.randperm(total, generator=g).tolist()
    bt = torch.zeros(len(lens), max(pages), dtype=torch.int32)
    i = 0
    for b, p in enumerate(pages):
        bt[b, :p] = torch.tensor(perm[i : i + p])
        i += p
    cache = hp.HPKVCache(total, page_size, k.shape[1], D, cfg, k.device)
    if nan_fill:
        for t in (cache.kh, cache.kl, cache.vh, cache.vl, cache.k_inv, cache.v_inv):
            if t is not None:
                t.fill_(float("nan"))
    bt = bt.to(k.device)
    slots = []
    for b, n in enumerate(lens):
        pos = torch.arange(n, device=k.device)
        slots.append(bt[b, pos // page_size].long() * page_size + pos % page_size)
    cache.write(torch.cat(slots), k, v)
    return cache, bt


@cuda
@pytest.mark.parametrize("hkv", [4, 2])  # Qwen3-30B-A3B, Nemotron-3-Nano
@pytest.mark.parametrize("cfg_name", ["hp", "hp_lolo", "hp_1acc", "fp16", "kv_fp16"])
def test_decode_matches_prefill_bitwise(cfg_name, hkv):
    cfg = {
        "hp": hp.HPAttnConfig(),
        "hp_lolo": hp.HPAttnConfig(lo_lo=True),
        "hp_1acc": hp.HPAttnConfig(sep_acc=False),
        "fp16": hp.HPAttnConfig.fp16(),
        "kv_fp16": hp.HPAttnConfig(k_lo=False, v_lo=False),
    }[cfg_name]
    lens = [1, 9, 64, 65, 333, 1029, 2500]
    q, k, v, cu = _inputs(lens, seed=1, hkv=hkv)
    ref_out, ref_lse = hp.hp_attention_prefill(q, k, v, cu, cfg)
    scale = 1.0 / math.sqrt(D)
    qh, ql, qi = hp.split_q(q, cfg)
    g = torch.Generator(device="cpu").manual_seed(2)
    for page_size in (64, 128, 256):
        cache, bt = _paged_cache(lens, k, v, cu, cfg, page_size, seed=page_size)
        for trial in range(4):
            # a decode batch: a random subset of sequences, each at a random
            # position t (cache holds 0..t), 1..4 new tokens (spec verify).
            bs = torch.randperm(len(lens), generator=g)[: 1 + trial * 2].tolist()
            qi_list, seq_list, pos_list = [], [], []
            kv_lens = torch.zeros(len(lens), dtype=torch.int32)
            for b in bs:
                n_new = int(torch.randint(1, 5, (1,), generator=g))
                end = int(torch.randint(1, lens[b] + 1, (1,), generator=g))
                start = max(0, end - n_new)
                kv_lens[b] = end
                for t in range(start, end):
                    qi_list.append(int(cu[b]) + t)
                    seq_list.append(b)
                    pos_list.append(t)
            idx = torch.tensor(qi_list, device=q.device)
            q_seq = torch.tensor(seq_list, dtype=torch.int32, device=q.device)
            q_pos = torch.tensor(pos_list, dtype=torch.int32, device=q.device)
            for split in (False, True):
                for dec in (False, True):
                    out, lse = hp.hp_attention_cached(
                        qh[idx], ql[idx], qi[idx], q_seq, q_pos, cache, bt,
                        kv_lens.to(q.device), scale, split=split, decode_tiles=dec,
                    )  # fmt: skip
                    assert torch.equal(out, ref_out[idx]), (
                        page_size,
                        trial,
                        split,
                        dec,
                    )
                    assert torch.equal(lse, ref_lse[idx])


@cuda
def test_incremental_decode_token_by_token():
    """Engine-style: append one token's K/V, decode it, for every position."""
    cfg = hp.HPAttnConfig()
    lens = [150, 77, 3]
    q, k, v, cu = _inputs(lens, seed=3)
    ref, _ = hp.hp_attention_prefill(q, k, v, cu, cfg)
    page_size = 64
    pages = [(n + page_size - 1) // page_size for n in lens]
    perm = torch.randperm(sum(pages) + 5, generator=torch.Generator().manual_seed(5))
    cache = hp.HPKVCache(sum(pages) + 5, page_size, HKV, D, cfg, q.device)
    for t_ in (cache.kh, cache.kl, cache.vh, cache.vl):
        t_.fill_(float("nan"))
    bt = torch.zeros(len(lens), max(pages), dtype=torch.int32)
    i = 0
    for b, p in enumerate(pages):
        bt[b, :p] = perm[i : i + p]
        i += p
    bt = bt.to(q.device)
    qh, ql, qi = hp.split_q(q, cfg)
    kv_lens = torch.zeros(len(lens), dtype=torch.int32, device=q.device)
    outs = torch.full_like(ref, float("nan"))
    for t in range(max(lens)):
        active = [b for b in range(len(lens)) if t < lens[b]]
        rows = torch.tensor([int(cu[b]) + t for b in active], device=q.device)
        seqs = torch.tensor(active, dtype=torch.int32, device=q.device)
        slots = bt[seqs.long(), t // page_size].long() * page_size + t % page_size
        cache.write(slots, k[rows], v[rows])
        kv_lens[seqs.long()] = t + 1
        pos = torch.full_like(seqs, t)
        out, _ = hp.hp_attention_cached(
            qh[rows], ql[rows], qi[rows], seqs, pos, cache, bt, kv_lens,
            1.0 / math.sqrt(D), split=(t % 2 == 1), decode_tiles=(t % 3 != 0),
        )  # fmt: skip
        outs[rows] = out
    assert torch.equal(outs, ref)


@cuda
def test_prefill_batch_invariant():
    cfg = hp.HPAttnConfig()
    lens = [700, 129, 2048, 5]
    q, k, v, cu = _inputs(lens, seed=6)
    out, _ = hp.hp_attention_prefill(q, k, v, cu, cfg)
    for b in range(len(lens)):
        s, e = int(cu[b]), int(cu[b + 1])
        cu1 = torch.tensor([0, e - s], dtype=torch.int32, device=q.device)
        alone, _ = hp.hp_attention_prefill(q[s:e], k[s:e], v[s:e], cu1, cfg)
        assert torch.equal(alone, out[s:e])
    # a prefix of a sequence gives the prefix's outputs (causal, chunk-aligned)
    cu2 = torch.tensor([0, 1500], dtype=torch.int32, device=q.device)
    s = int(cu[2])
    pre, _ = hp.hp_attention_prefill(
        q[s : s + 1500], k[s : s + 1500], v[s : s + 1500], cu2, cfg
    )
    assert torch.equal(pre, out[s : s + 1500])


def _rel_err(out, ref):
    """max over rows of ||out - ref|| / ||ref|| (per token, per head)."""
    num = (out.double() - ref).norm(dim=-1)
    den = ref.norm(dim=-1).clamp_min(1e-300)
    return (num / den).max().item()


@cuda
def test_error_vs_fp64():
    lens = [1, 100, 1500]
    q, k, v, cu = _inputs(lens, seed=7)
    ref = hp.reference_attention(q, k, v, cu)
    err = {}
    for name, cfg in {
        "hp": hp.HPAttnConfig(),
        "fp16": hp.HPAttnConfig.fp16(),
    }.items():
        out, _ = hp.hp_attention_prefill(q, k, v, cu, cfg)
        err[name] = _rel_err(out, ref)
    assert err["hp"] < 1e-5, err
    assert err["fp16"] > 100 * err["hp"], err


@cuda
def test_backward_close_to_reference():
    lens = [300, 200]
    q, k, v, cu = _inputs(lens, seed=8)
    q, k, v = (t.requires_grad_() for t in (q, k, v))
    out = hp.hp_attention(q, k, v, cu, max(lens))
    w = torch.randn_like(out)
    (out * w).sum().backward()
    grads = [t.grad.clone() for t in (q, k, v)]
    qd, kd, vd = (t.detach().double().requires_grad_() for t in (q, k, v))
    ref = hp.reference_attention(qd, kd, vd, cu, dtype=torch.float64)
    (ref * w.double()).sum().backward()
    for got, want in zip(grads, (qd.grad, kd.grad, vd.grad)):
        rel = (got.double() - want).norm() / want.norm()
        assert rel < 2e-2, rel.item()


@cuda
@pytest.mark.parametrize("hkv", [4, 2, 32])
def test_fused_split_matches_reference(hkv):
    torch.manual_seed(9)
    x = torch.randn(257, hkv, D, device="cuda")
    x = x * torch.exp(torch.randn(257, hkv, 1, device="cuda") * 6)
    x[3, 0] = 0.0  # zero row
    x[4, 0] = 1e-42  # subnormal amax
    for emin, emax in ((-40, 40), (-40, 3)):
        ref = hp.split_rows(x, emin, emax)
        got = hp.split_rows_fused(x, emin, emax)
        for a, b in zip(ref, got):
            assert torch.equal(a, b)


# ------------------------------------------- sync-free layouts / Megatron wiring


def _tile_sets(tiles):
    ts, t0, tq = tiles
    out = set()
    for i in range(ts.numel()):
        q = tuple(tq[i].tolist())
        if any(x >= 0 for x in q):
            out.add((int(ts[i]), int(t0[i]), q))
    return out


@pytest.mark.parametrize("tm", [1, 2, 4])
def test_static_tiles_match_build_tiles(tm):
    # Engine-like step: decode rows (1 query at kv_len - 1), a prefill chunk continuing a
    # prefix, an empty padded request, then token-only padding rows.
    qlen = torch.tensor([1, 1, 7, 0, 3])
    kv = torch.tensor([10, 3, 12, 0, 3])
    cu = torch.cat([torch.zeros(1, dtype=torch.long), qlen.cumsum(0)])
    n = int(cu[-1]) + 5
    ts, t0, tq, seq, pos, valid = hp.static_tiles(cu, kv, n, tm)
    nv = int(cu[-1])
    assert valid.tolist() == [True] * nv + [False] * 5
    ref_seq = torch.repeat_interleave(torch.arange(5), qlen)
    ref_pos = torch.cat([torch.arange(k - q, k) for q, k in zip(qlen.tolist(), kv.tolist())])
    assert torch.equal(seq[:nv], ref_seq) and torch.equal(pos[:nv], ref_pos)
    assert _tile_sets((ts, t0, tq)) == _tile_sets(hp.build_tiles(ref_seq, ref_pos, tm))


def test_static_prefill_layout_pages():
    cu = torch.tensor([0, 70, 70, 200], dtype=torch.int32)
    slots, bt, lens, num_pages = hp.static_prefill_layout(cu, 210, 64)
    assert lens.tolist() == [70, 0, 130]
    assert num_pages == 210 // 64 + 3 + 1
    # every real token lands in its sequence's pages at its position; padding in the last page
    for b, (s, e) in enumerate([(0, 70), (70, 70), (70, 200)]):
        for t in range(s, e):
            p = t - s
            assert int(slots[t]) == int(bt[b, p // 64]) * 64 + p % 64
    assert (slots[200:] == (num_pages - 1) * 64).all()
    assert len(set(slots[:200].tolist())) == 200


@cuda
@pytest.mark.parametrize("mode", ["qp_kv", "qp"])
@pytest.mark.parametrize("hkv", [2, 4])
def test_learner_static_path_matches_prefill(hkv, mode):
    from nemo_rl.models.megatron import hp_attention_mcore as hpm

    lens = [1, 63, 64, 65, 700, 1500]
    q, k, v, cu = _inputs(lens, seed=11, hkv=hkv)
    cfg = hpm.config_for(True, mode == "qp_kv")
    scale = 1.0 / math.sqrt(D)
    ref, ref_lse = hp.hp_attention_prefill(q, k, v, cu, cfg, scale)
    # trailing padding rows (THD padded micro-batch): must not change real rows
    pad = 37
    qp, kp, vp = (torch.cat([t, torch.randn(pad, *t.shape[1:], device=t.device)]) for t in (q, k, v))
    out, lse = hpm._learner_forward(qp, kp, vp, cu, cfg, scale)
    n = q.shape[0]
    assert torch.equal(out[:n], ref) and torch.equal(lse[:n], ref_lse)


class _Ctx:
    """The engine-side state hp_attention_mcore reads from a DynamicInferenceContext."""


_MODES = {"qp_kv": (True, True), "qp": (True, False)}


@cuda
@pytest.mark.parametrize("mode", list(_MODES))
@pytest.mark.parametrize("hkv", [2, 4])
def test_engine_steps_match_learner(hkv, mode):
    """Engine steps through hp_attention_mcore.engine_attention == learner forward, bitwise.

    Paged cache in MCore's layout ([4, L, pool, block, Hkv, D] fp16 planes + [2, L, ...] fp32
    scales, block 256, random block order, NaN in unused slots, a dummy block for padding),
    written through the same slot arithmetic as ``_append_key_value_cache``; steps: a mixed
    prefill step (two prompts + token-only padding rows), then decode-only steps (with padded
    requests of length 0) at varying batch, split on and off, one step CUDA-graph captured.
    """
    from types import SimpleNamespace

    from nemo_rl.models.megatron import hp_attention_mcore as hpm

    torch.manual_seed(5)
    cfg = hpm.config_for(*_MODES[mode])
    hpm._STATE["cfg"] = cfg
    bs, L, layer = 256, 3, 1
    plen = [300, 41, 1]
    gen = 40
    lens = [p + gen for p in plen]
    q, k, v, cu = _inputs(lens, seed=21, hkv=hkv)
    scale = 1.0 / math.sqrt(D)
    ref, _ = hp.hp_attention_prefill(q, k, v, cu, cfg, scale)
    pool = 16
    ctx = _Ctx()
    ctx.block_size_tokens = bs
    ctx.memory_buffer = torch.full((2 if cfg.kv_fp16 else 4, L, pool, bs, hkv, D), float("nan"),
                                   dtype=torch.float16, device="cuda")
    ctx._hp_kv_inv = None if cfg.kv_fp16 else torch.full((2, L, pool, bs, hkv), float("nan"), device="cuda")
    dummy = pool - 1
    perm = torch.randperm(pool - 1).tolist()
    blocks = [perm[2 * i: 2 * i + 2] for i in range(len(lens))]
    bt_rows = torch.tensor(blocks, dtype=torch.int32, device="cuda")
    hpm._STATE["ctx"] = ctx
    module = SimpleNamespace(config=SimpleNamespace(softmax_scale=None, position_embedding_type="none"))
    starts = cu[:-1].tolist()

    def write(tokens):  # (seq, pos) list -> append through the engine slot arithmetic
        idx = torch.tensor([starts[s] + p for s, p in tokens], device="cuda")
        blk = torch.tensor([blocks[s][p // bs] for s, p in tokens], device="cuda")
        loc = torch.tensor([p % bs for _, p in tokens], device="cuda")
        slots = blk.long() * bs + loc.long()
        hpm._write_layer(ctx, layer, slots, k[idx], v[idx])
        return idx

    def step(rows, qlens, kvlens, pad_tokens, pad_reqs, decode, split_rows=32):
        hpm._STATE["split_rows"] = split_rows
        nreq = len(rows) + pad_reqs
        qcu = torch.tensor([0] + list(torch.tensor(qlens + [0] * pad_reqs).cumsum(0)), dtype=torch.int32,
                           device="cuda")
        kvl = torch.tensor(kvlens + [0] * pad_reqs, dtype=torch.int32, device="cuda")
        bt = torch.full((nreq, 2), dummy, dtype=torch.int32, device="cuda")
        bt[: len(rows)] = bt_rows[rows]
        toks = [(s, kv - ql + i) for s, ql, kv in zip(rows, qlens, kvlens) for i in range(ql)]
        idx = write(toks)
        qq = torch.cat([q[idx], torch.randn(pad_tokens, HQ, D, device="cuda")]).unsqueeze(1)
        kview = ctx.memory_buffer[0, layer]
        out = hpm.engine_attention(module, qq, kview, qcu, kvl, bt, decode, max_seqlen_k=512)
        assert out.shape == (qq.shape[0], 1, HQ, D) and out.dtype == torch.float32
        assert torch.equal(out[: len(toks), 0], ref[idx])
        assert (out[len(toks):] == 0).all()
        return qq, qcu, kvl, bt

    # prefill: whole prompts of seq 0 and 2, first 20 prompt tokens of seq 1 (chunked), padding
    step([0, 1, 2], [plen[0], 20, plen[2]], [plen[0], 20, plen[2]], pad_tokens=9, pad_reqs=0, decode=False)
    step([1], [plen[1] - 20], [plen[1]], pad_tokens=3, pad_reqs=2, decode=False)
    for i in range(gen):
        rows = [r for r in range(3) if (i + r) % 4 != 3]  # requests drop in and out of the batch
        kvs = [plen[r] + i + 1 for r in rows]
        if any(kv > lens[r] for r, kv in zip(rows, kvs)):
            continue
        for r in [r for r in range(3) if r not in rows]:  # keep every cache complete
            write([(r, plen[r] + i)])
        step(rows, [1] * len(rows), kvs, pad_tokens=i % 3, pad_reqs=i % 2, decode=True,
             split_rows=32 if i % 2 else 0)
    # CUDA graph: a decode step captured once (no host syncs allowed), replayed bitwise.
    rows = [0, 1, 2]
    kvs = [lens[r] for r in rows]
    hpm._STATE["split_rows"] = 32
    qcu = torch.arange(4, dtype=torch.int32, device="cuda")
    kvl = torch.tensor(kvs, dtype=torch.int32, device="cuda")
    idx = torch.tensor([starts[r] + kv - 1 for r, kv in zip(rows, kvs)], device="cuda")
    qq = q[idx].unsqueeze(1).clone()
    kview = ctx.memory_buffer[0, layer]
    eager = hpm.engine_attention(module, qq, kview, qcu, kvl, bt_rows, True, max_seqlen_k=512)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        hpm.engine_attention(module, qq, kview, qcu, kvl, bt_rows, True, max_seqlen_k=512)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        graphed = hpm.engine_attention(module, qq, kview, qcu, kvl, bt_rows, True, max_seqlen_k=512)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(graphed, eager) and torch.equal(eager[:, 0], ref[idx])


@cuda
def test_qp_mode_kv_is_plain_fp16():
    """attn_core_qp alone: the cache holds exactly fp16(k), fp16(v) (today's engine cache bytes)."""
    from nemo_rl.models.megatron import hp_attention_mcore as hpm

    cfg = hpm.config_for(True, False)
    k = torch.randn(50, 2, D, device="cuda") * 3
    v = torch.randn(50, 2, D, device="cuda")
    cache = hp.HPKVCache(1, 64, 2, D, cfg, "cuda")
    cache.write(torch.arange(50, device="cuda"), k, v)
    assert cache.kl is None and cache.vl is None and cache.k_inv is None
    assert torch.equal(cache.kh[0, :50], k.half()) and torch.equal(cache.vh[0, :50], v.half())
    # and the qp-only result differs from the full fp16 kernel (q / P hi/lo are live)
    q, kk, vv, cu = _inputs([300], seed=3, hkv=2)
    a, _ = hp.hp_attention_prefill(q, kk, vv, cu, cfg)
    b, _ = hp.hp_attention_prefill(q, kk, vv, cu, hp.HPAttnConfig.fp16())
    assert not torch.equal(a, b)


@cuda
@pytest.mark.parametrize("mode", ["qp_kv", "qp"])
@pytest.mark.parametrize("hkv", [2, 4])
def test_prefix_attention_matches_full_forward(hkv, mode, monkeypatch):
    """Learner prefix-state cache path (queries = the last rows of each sequence, keys = cached + new):
    bitwise the full-sequence learner forward's rows (precision_raise attn_core on prefix_state_cache)."""
    from nemo_rl.models.megatron import hp_attention_mcore as hpm

    lens = [65, 700, 1500, 9]
    starts = [64, 128, 1024, 0]  # cached prefix lengths (Mamba-chunk multiples or 0)
    q, k, v, cu = _inputs(lens, seed=5, hkv=hkv)
    cfg = hpm.config_for(True, mode == "qp_kv")
    monkeypatch.setitem(hpm._STATE, "cfg", cfg)
    scale = 1.0 / math.sqrt(D)
    with torch.no_grad():
        ref, _ = hpm._learner_forward(q, k, v, cu, cfg, scale)
        qs, cu_q = [], [0]
        for b, (n, s0) in enumerate(zip(lens, starts)):
            a = int(cu[b])
            qs.append(q[a + s0 : a + n])
            cu_q.append(cu_q[-1] + n - s0)
        cu_q = torch.tensor(cu_q, dtype=torch.int32, device=q.device)
        out = hpm.prefix_attention(torch.cat(qs), k, v, cu_q, cu, max(lens), scale)
    for b, (n, s0) in enumerate(zip(lens, starts)):
        a = int(cu[b])
        assert torch.equal(out[int(cu_q[b]) : int(cu_q[b + 1])], ref[a + s0 : a + n]), b
