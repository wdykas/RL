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
"""DS2 (thundersync_rl.ds2): the vocabulary-sharded ⊥-BV grid scan.

- Kernels: bitwise the frozen task-D reference kernels (copied below from kernel_tasks/vocablse/vl_reference, the
  file pinned in VL_ORACLE.sha256) on materialized rows, fp32 and bf16 logits, every vocab limit.
- Scan: the same outcomes (tau, y, log p(y)), the same generator draws, as the gathered scan
  (block_verification_fused.block_verify_stats_lazy_grid) on the same logits. It is independent of the chunk sizes.
  The cases reach BOT (⊥ root), residual and bonus emissions, multi-chunk scans and padded vocabularies.
- Fewer host syncs than the gathered scan on a row that scans to the root.
- Off by default; eligibility; HiddenRows.rows_at == HiddenRows.__call__.
TP 2 / 4 bitwise equality with TP 1 is ds2_tp.py (torchrun), run by test_ds2_tp below when >= 2 GPUs are visible.
"""

import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from thundersync_rl import ds2
from thundersync_rl.block_verification_fused import block_verify_stats_lazy_grid

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")

# ---- frozen task-D reference kernels (kernel_tasks/vocablse/vl_reference/__init__.py, verbatim) -------------------
_TILE = 1024


@triton.jit
def _ref_residual_kernel(ZP, ZQ, LP, LQ, Bw, OUT, sp, sq, voff, vlim, nb_local, HAS_Q: tl.constexpr,
                         BLOCK: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    j = tl.program_id(1)
    g0 = voff + j * BLOCK
    lse_p = tl.load(LP + row)
    lse_q = tl.load(LQ + row)
    b = tl.load(Bw + row)
    acc = tl.zeros([], tl.float64)
    for c0 in range(0, BLOCK, TILE):
        cols = c0 + tl.arange(0, TILE)
        ok = (g0 + cols) < vlim
        zp = tl.load(ZP + row.to(tl.int64) * sp + j * BLOCK + cols, mask=ok, other=float("-inf")).to(tl.float32)
        p = tl.exp((zp - lse_p).to(tl.float64))
        d = b * p
        if HAS_Q:
            zq = tl.load(ZQ + row.to(tl.int64) * sq + j * BLOCK + cols, mask=ok, other=float("-inf")).to(tl.float32)
            d = d - tl.exp((zq - lse_q).to(tl.float64))
        d = tl.where(ok & (d > 0), d, 0.0)
        acc += tl.sum(d, 0)
    tl.store(OUT + row.to(tl.int64) * nb_local + j, acc)


@triton.jit
def _ref_sample_column_kernel(ZP, ZQ, LP, LQ, Bw, BLK, TIN, OUT, sp, sq, voff, vlim, nb_local, HAS_Q: tl.constexpr,
                              BLOCK: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    blk = tl.load(BLK + row)
    jl = blk - voff // BLOCK
    owned = (jl >= 0) & (jl < nb_local)
    jl = tl.where(owned, jl, 0)
    g0 = voff + jl * BLOCK
    lse_p = tl.load(LP + row)
    lse_q = tl.load(LQ + row)
    b = tl.load(Bw + row)
    tin = tl.load(TIN + row)
    carry = tl.zeros([], tl.float64)
    found = tl.full([], -1, tl.int64)
    lastnz = tl.full([], -1, tl.int64)
    for c0 in range(0, BLOCK, TILE):
        cols = c0 + tl.arange(0, TILE)
        ok = ((g0 + cols) < vlim) & owned
        zp = tl.load(ZP + row.to(tl.int64) * sp + jl * BLOCK + cols, mask=ok, other=float("-inf")).to(tl.float32)
        d = b * tl.exp((zp - lse_p).to(tl.float64))
        if HAS_Q:
            zq = tl.load(ZQ + row.to(tl.int64) * sq + jl * BLOCK + cols, mask=ok, other=float("-inf")).to(tl.float32)
            d = d - tl.exp((zq - lse_q).to(tl.float64))
        d = tl.where(ok & (d > 0), d, 0.0)
        cs = carry + tl.cumsum(d, 0)
        hit = cs > tin
        first = tl.min(tl.where(hit, cols, BLOCK), 0)
        found = tl.where((found < 0) & (first < BLOCK), (g0 + first).to(tl.int64), found)
        lnz = tl.max(tl.where(d > 0, cols, -1), 0)
        lastnz = tl.where(lnz >= 0, (g0 + lnz).to(tl.int64), lastnz)
        carry = carry + tl.sum(d, 0)
    found = tl.where(found < 0, lastnz, found)
    found = tl.where(owned, found, -1)
    tl.store(OUT + row, found)


def _f32(z):
    return z if z is None or z.dtype == torch.float32 else z.float()


def ref_residual_partials(zp_local, zq_local, vocab_offset, vocab_limit, lse_p, lse_q, b):
    zp_local, zq_local = _f32(zp_local), _f32(zq_local)
    n, vl = zp_local.shape
    out = torch.empty(n, vl // ds2.BLOCK, dtype=torch.float64, device=zp_local.device)
    has_q = zq_local is not None
    zq = zq_local if has_q else zp_local
    _ref_residual_kernel[(n, vl // ds2.BLOCK)](zp_local, zq, lse_p, lse_q, b, out, zp_local.stride(0), zq.stride(0),
                                               vocab_offset, vocab_limit, vl // ds2.BLOCK, HAS_Q=has_q,
                                               BLOCK=ds2.BLOCK, TILE=_TILE)
    return out


def ref_sample_column(zp_local, zq_local, vocab_offset, vocab_limit, lse_p, lse_q, b, block, t_in):
    zp_local, zq_local = _f32(zp_local), _f32(zq_local)
    n, vl = zp_local.shape
    out = torch.empty(n, dtype=torch.int64, device=zp_local.device)
    has_q = zq_local is not None
    zq = zq_local if has_q else zp_local
    _ref_sample_column_kernel[(n,)](zp_local, zq, lse_p, lse_q, b, block, t_in, out, zp_local.stride(0),
                                    zq.stride(0), vocab_offset, vocab_limit, vl // ds2.BLOCK, HAS_Q=has_q,
                                    BLOCK=ds2.BLOCK, TILE=_TILE)
    return out


# ---- cases -------------------------------------------------------------------------------------------------------


def _lse(z, vlim):
    from nemo_rl.models.megatron import vocab_lse

    return vocab_lse.lse_full(z, vlim)


def _case(seed, g, nblk=2, vlim_cut=0, sharp=0.6, accept_all=False, dtype=torch.float32, scale=2.0):
    """p logits [g+1, V], q logits [g, V] (V = nblk * 4096; columns >= V - vlim_cut are padding), a draft from q."""
    V = nblk * ds2.BLOCK
    vlim = V - vlim_cut
    rng = torch.Generator().manual_seed(seed)
    zp = torch.randn(g + 1, V, generator=rng) * scale
    zq = zp[:g] + sharp * torch.randn(g, V, generator=rng)
    if accept_all:
        zq = zp[:g].clone()
    zp, zq = zp.to(dtype).cuda(), zq.to(dtype).cuda()
    lq_rows = zq[:, :vlim].float() - _lse(zq, vlim)[:, None]
    draft = torch.multinomial(lq_rows.exp(), 1, generator=torch.Generator(device="cuda").manual_seed(seed))[:, 0]
    return zp, zq, vlim, draft


def _gathered(zp, zq, vlim, draft, seed, envs, chunk=512):
    """The gathered scan on canonical log-prob rows (what the worker runs at TP > 1 without DS2)."""
    g = draft.numel()
    P = zp[:, :vlim].float() - _lse(zp, vlim)[:, None]
    Q = zq[:, :vlim].float() - _lse(zq, vlim)[:, None]
    ar = torch.arange(g, device="cuda")
    lpx, lqx = P[ar, draft], Q[ar, draft]
    z = torch.zeros(g + 1, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(seed)
    out, n_eval = block_verify_stats_lazy_grid(
        lpx, lqx, z, z[:g], lambda i: P[i.to("cuda")], lambda i: Q[i.to("cuda")], gen, envs, chunk=chunk
    )
    return out, lpx, lqx, gen.get_state()


def _sharded(zp, zq, vlim, draft, lqx, seed, envs, first=ds2.FIRST_CHUNK, chunk=ds2.CHUNK):
    rows = ds2.ShardedRows(zp, vlim, None, 0)
    lpx = rows.token_logprobs(draft)
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def q_local(idx):
        zz = zq[idx.to("cuda")].contiguous()
        return zz, _lse(zz, vlim)

    out, n_eval = ds2.grid_scan_sharded(rows, lpx, lqx, q_local, gen, envs, first_chunk=first, chunk=chunk)
    return out, lpx, gen.get_state(), n_eval


CASES = [
    dict(g=1), dict(g=2), dict(g=40), dict(g=300, sharp=0.05), dict(g=2000, sharp=0.02),
    dict(g=64, accept_all=True), dict(g=500, sharp=3.0), dict(g=200, nblk=3, vlim_cut=1000, sharp=0.05),
    dict(g=120, vlim_cut=4096 + 7, sharp=0.3), dict(g=150, dtype=torch.bfloat16, sharp=0.05),
]
ENVS = (1.0, 1.25, 1.5, 2.0, 3.0)


@cuda
@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("seed", range(4))
def test_sharded_scan_equals_gathered_scan(case, seed):
    zp, zq, vlim, draft = _case(seed, **CASES[case])
    want, lpx_w, lqx, gen_w = _gathered(zp, zq, vlim, draft, 100 + seed, ENVS)
    got, lpx_g, gen_g, _ = _sharded(zp, zq, vlim, draft, lqx, 100 + seed, ENVS)
    assert torch.equal(lpx_g, lpx_w)  # the drafted tokens' log p: bitwise
    assert got == want  # every envelope's (tau, y, log p(y))
    assert torch.equal(gen_g, gen_w)  # the same draws


@cuda
@pytest.mark.parametrize("chunks", [(1, 1), (4, 3), (512, 2048), (100000, 100000)])
def test_outcomes_do_not_depend_on_chunks(chunks):
    zp, zq, vlim, draft = _case(7, g=900, sharp=0.03)
    _, _, lqx, _ = _gathered(zp, zq, vlim, draft, 3, ENVS)
    base = _sharded(zp, zq, vlim, draft, lqx, 3, ENVS)
    got = _sharded(zp, zq, vlim, draft, lqx, 3, ENVS, *chunks)
    assert got[0] == base[0] and torch.equal(got[2], base[2])


@cuda
def test_cases_reach_bot_residual_bonus_and_multichunk():
    """Non-vacuous: BOT (⊥ root), residual and bonus emissions occur, and some rows scan several chunks."""
    seen, chunks = set(), 0
    for case in range(len(CASES)):
        for seed in range(4):
            zp, zq, vlim, draft = _case(seed, **CASES[case])
            _, _, lqx, _ = _gathered(zp, zq, vlim, draft, 100 + seed, ENVS)
            out, _, _, n_eval = _sharded(zp, zq, vlim, draft, lqx, 100 + seed, ENVS)
            g = draft.numel()
            chunks = max(chunks, n_eval)
            for t, y, _ in out.values():
                seen.add("bot" if y < 0 else ("bonus" if t == g else ("root" if t == 0 else "residual")))
    assert {"bot", "bonus", "residual"} <= seen, seen
    assert chunks > ds2.FIRST_CHUNK, chunks  # some row scanned past the first chunk


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("vlim_cut", [0, 1000, 4096 + 1])
@pytest.mark.parametrize("has_q", [True, False])
def test_kernels_are_the_task_d_reference(dtype, vlim_cut, has_q):
    zp, zq, vlim, _ = _case(11, g=37, nblk=3, vlim_cut=vlim_cut, dtype=dtype, sharp=0.2)
    n = 53
    gen = torch.Generator(device="cuda").manual_seed(5)
    ip = torch.randint(0, zp.shape[0], (n,), device="cuda", generator=gen)
    iq = torch.randint(0, zq.shape[0], (n,), device="cuda", generator=gen)
    lp = _lse(zp, vlim)[ip]
    lq = _lse(zq, vlim)[iq] if has_q else lp
    b = torch.rand(n, dtype=torch.float64, device="cuda", generator=gen)
    zq_ = zq if has_q else None
    for off, width in ((0, zp.shape[1]), (ds2.BLOCK, 2 * ds2.BLOCK), (0, ds2.BLOCK)):  # full rows and shards
        sp, sq = zp[:, off : off + width], (zq[:, off : off + width] if has_q else None)
        mine = ds2.residual_block_masses(sp, ip, sq, iq, lp, lq, b, off, vlim)
        ref = ref_residual_partials(sp[ip].contiguous(), None if sq is None else sq[iq].contiguous(), off, vlim,
                                    lp.contiguous(), lq.contiguous(), b)
        assert torch.equal(mine, ref)
    full = ds2.residual_block_masses(zp, ip, zq_, iq, lp, lq, b, 0, vlim)
    u = torch.rand(n, dtype=torch.float64, device="cuda", generator=gen)
    bot = torch.where(torch.arange(n, device="cuda") % 3 == 0, 0.25, 0.0).double()
    blk, tin = ds2.sample_block(full, bot, u)
    col = ds2.sample_column(zp, ip, zq_, iq, lp, lq, b, blk, tin, 0, vlim)
    rcol = ref_sample_column(zp[ip].contiguous(), None if zq_ is None else zq[iq].contiguous(), 0, vlim,
                             lp.contiguous(), lq.contiguous(), b, blk.contiguous(), tin.contiguous())
    assert torch.equal(col, rcol)
    # the draw is the inverse CDF of the fp64 masses, the extra atom included (task D item 4)
    t = u * (ds2.mass_combine(full) + bot)
    for k in range(n):
        if int(blk[k]) == full.shape[1]:
            assert float(t[k]) >= float(ds2.mass_combine(full)[k]) - 1e-12
            continue
        v = int(col[k])
        assert 0 <= v < vlim


@cuda
def test_errors_match_the_gathered_scan():
    from thundersync_rl.block_verification import NonFiniteVerification

    zp, zq, vlim, draft = _case(1, g=20)
    _, lpx, lqx, _ = _gathered(zp, zq, vlim, draft, 1, (1.0, 1.5))
    bad = lqx.clone()
    bad[3] = float("-inf")
    with pytest.raises(NonFiniteVerification, match="engine log q"):
        _sharded(zp, zq, vlim, draft, bad, 1, (1.0, 1.5))


def _syncs(fn, *a, **k):
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            fn(*a, **k)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    return sum("synchroniz" in str(x.message) for x in w)


@cuda
def test_fewer_host_syncs_on_a_row_that_scans_to_the_root():
    envs = (1.25, 1.5, 2.0)
    for seed in range(20):
        zp, zq, vlim, draft = _case(seed, g=4000, sharp=0.01)
        want, _, lqx, _ = _gathered(zp, zq, vlim, draft, 5, envs)
        if any(t == 0 for t, _, _ in want.values()):  # scans to the root
            break
    else:
        pytest.fail("no root-scanning row in 20 seeds")
    old = _syncs(_gathered, zp, zq, vlim, draft, 5, envs)
    new = _syncs(_sharded, zp, zq, vlim, draft, lqx, 5, envs)
    print(f"[ds2] host syncs on a root-scanning row (g=4000): sharded {new} vs gathered {old}")
    assert new < old


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(ds2.ENV, raising=False)
    assert not ds2.enabled()
    monkeypatch.setenv(ds2.ENV, "0")
    assert not ds2.enabled()
    monkeypatch.setenv(ds2.ENV, "1")
    assert ds2.enabled()


def test_eligibility(monkeypatch):
    from thundersync_rl.engine_q import DeferredLogprobRows, GatheredLogprobRows, HiddenRows, TPVocabGather, no_gather

    p_tp = GatheredLogprobRows(torch.zeros(3, 2 * ds2.BLOCK), TPVocabGather(None), 2 * ds2.BLOCK)
    h = HiddenRows(torch.zeros(2, 4))
    monkeypatch.setenv("NEMO_RL_VOCAB_LSE", "canonical_block")
    monkeypatch.setenv(ds2.ENV, "1")
    assert ds2.eligible(p_tp, h, False, False)
    assert not ds2.eligible(p_tp, h, True, False)  # exact integer law
    assert not ds2.eligible(p_tp, h, False, True)  # exact top-p
    assert not ds2.eligible(GatheredLogprobRows(torch.zeros(3, 8), no_gather, 8), h, False, False)  # TP 1
    assert not ds2.eligible(GatheredLogprobRows(torch.zeros(3, 4000), TPVocabGather(None), 4000), h, False, False)
    assert not ds2.eligible(DeferredLogprobRows(torch.zeros(3, 4), torch.zeros(8, 4), 8), h, False, False)
    assert not ds2.eligible(p_tp, object(), False, False)  # q rows not from logged hidden states
    monkeypatch.setenv(ds2.ENV, "0")
    assert not ds2.eligible(p_tp, h, False, False)
    monkeypatch.setenv(ds2.ENV, "1")
    monkeypatch.setenv("NEMO_RL_VOCAB_LSE", "off")
    assert not ds2.eligible(p_tp, h, False, False)


def test_hidden_rows_at_equals_call():
    from thundersync_rl.engine_q import HiddenRefs, HiddenRows, HiddenSeg

    rng = np.random.default_rng(0)
    blocks = [torch.randn(n, 6, dtype=torch.bfloat16) for n in (5, 1, 9, 4)]
    store = {f"r{k}": b for k, b in enumerate(blocks)}
    refs = HiddenRefs([HiddenSeg(ref=f"r{k}", rows=b.shape[0], nbytes=b.numel() * 2) for k, b in enumerate(blocks)])
    for _ in range(20):
        pos = rng.choice(19, size=int(rng.integers(1, 19)), replace=False)
        pos = np.sort(pos)[::-1] if rng.random() < 0.5 else pos
        a = HiddenRows(refs, get=lambda r: store[r])
        b = HiddenRows(refs, get=lambda r: store[r])
        assert torch.equal(a.rows_at(pos), b([int(x) for x in pos]))
        assert a.fetched_bytes == b.fetched_bytes
    dense = torch.randn(7, 3)
    assert torch.equal(HiddenRows(dense).rows_at(np.array([6, 0, 3])), dense[[6, 0, 3]])


@cuda
def test_hidden_rows_at_device_equals_rows_at():
    """THUNDERSYNC_HIDDEN_DEVICE_GATHER: the device gather returns rows_at's bytes, in order, on the device, with the
    same fetch accounting; repeated chunks reuse the copied blocks (fp32 and bf16 rows)."""
    from thundersync_rl.engine_q import HiddenRefs, HiddenRows, HiddenSeg

    rng = np.random.default_rng(1)
    for dt in (torch.float32, torch.bfloat16):
        blocks = [torch.randn(n, 6).to(dt) for n in (5, 1, 9, 4, 7)]
        store = {f"r{k}": b for k, b in enumerate(blocks)}
        refs = HiddenRefs([HiddenSeg(ref=f"r{k}", rows=b.shape[0], nbytes=b.numel() * b.element_size())
                           for k, b in enumerate(blocks)])
        a = HiddenRows(refs, get=lambda r: store[r])
        b = HiddenRows(refs, get=lambda r: store[r])
        for _ in range(12):  # chunks of one draft's scan
            pos = rng.choice(26, size=int(rng.integers(1, 26)), replace=False)
            pos = np.sort(pos)[::-1] if rng.random() < 0.5 else pos
            got = a.rows_at_device(pos, torch.device("cuda"))
            assert got.is_cuda and got.dtype == dt
            assert torch.equal(got.cpu(), b.rows_at(pos))
        assert a.fetched_bytes == b.fetched_bytes and a.fetched_real_bytes == b.fetched_real_bytes
    dense = torch.randn(7, 3)
    assert torch.equal(HiddenRows(dense).rows_at_device(np.array([6, 0, 3]), torch.device("cuda")).cpu(),
                       dense[[6, 0, 3]])


@cuda
def test_hidden_rows_at_device_cap_falls_back(monkeypatch):
    """Above THUNDERSYNC_HIDDEN_DEVICE_GATHER_MAX_GB the host path is used (no [g, d] buffer), same rows."""
    from thundersync_rl.engine_q import HiddenRefs, HiddenRows, HiddenSeg

    blocks = [torch.randn(n, 6) for n in (5, 9)]
    store = {f"r{k}": b for k, b in enumerate(blocks)}
    refs = HiddenRefs([HiddenSeg(ref=f"r{k}", rows=b.shape[0], nbytes=b.numel() * 4) for k, b in enumerate(blocks)])
    monkeypatch.setenv("THUNDERSYNC_HIDDEN_DEVICE_GATHER_MAX_GB", "1e-9")
    a = HiddenRows(refs, get=lambda r: store[r])
    pos = np.array([13, 2, 7])
    got = a.rows_at_device(pos, torch.device("cuda"))
    assert got.is_cuda and "_dev_buf" not in a.__dict__
    assert torch.equal(got.cpu(), HiddenRows(refs, get=lambda r: store[r]).rows_at(pos))


def test_version_array():
    segs = [(3, 0, 4), (5, 4, 4), (7, 4, 10)]
    v = [0] * 10
    for ver, a, b in segs:
        v[a:b] = [ver] * (b - a)
    assert ds2.version_array(segs, 10).tolist() == v


@cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="2 GPUs")
@pytest.mark.parametrize("tp", [2, 4])
def test_ds2_tp(tp):
    if torch.cuda.device_count() < tp:
        pytest.skip(f"{tp} GPUs")
    script = os.path.join(os.path.dirname(__file__), "ds2_tp.py")
    r = subprocess.run([sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={tp}", script],
                       capture_output=True, text=True, timeout=900)
    print(r.stdout[-4000:], r.stderr[-4000:])
    assert r.returncode == 0
    assert "DS2 TP PASS" in r.stdout


# -- vwall (11-01): emission dedup (unconditional) -------------------------------------------------------------------------


@cuda
@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("envs", [ENVS, (1.0, 1.25), (1.25,), (1.0,)])
def test_emit_dedup_equals_per_envelope_emissions(monkeypatch, case, seed, envs):
    """The deduplicated emissions are the per-envelope ones bitwise: every envelope's (tau, y, log p(y)), the drafted
    tokens' log p, the positions evaluated and the generator state."""
    zp, zq, vlim, draft = _case(seed, **CASES[case])
    _, _, lqx, _ = _gathered(zp, zq, vlim, draft, 100 + seed, envs)
    monkeypatch.setattr(ds2, "_FORCE_DEDUP", False)
    want = _sharded(zp, zq, vlim, draft, lqx, 100 + seed, envs)
    monkeypatch.setattr(ds2, "_FORCE_DEDUP", None)
    got = _sharded(zp, zq, vlim, draft, lqx, 100 + seed, envs)
    assert got[0] == want[0]
    assert torch.equal(got[1], want[1]) and torch.equal(got[2], want[2]) and got[3] == want[3]


@cuda
def test_emit_dedup_engages_and_skips_root_p_masses(monkeypatch):
    """Non-vacuous: rows accepted whole at several envelopes share their bonus emission; a ⊥-root emission at M != 1
    skips the unread p masses (one residual-mass launch fewer)."""
    monkeypatch.setattr(ds2, "_FORCE_DEDUP", None)
    zp, zq, vlim, draft = _case(0, g=64, accept_all=True)
    g = draft.numel()
    _, _, lqx, _ = _gathered(zp, zq, vlim, draft, 5, ENVS)
    for s in range(20):
        d0 = ds2.COUNTS["emit_dedup"]
        out, *_ = _sharded(zp, zq, vlim, draft, lqx, s, ENVS)
        n_bonus = sum(t == g for t, _, _ in out.values())
        assert ds2.COUNTS["emit_dedup"] - d0 == max(n_bonus - 1, 0)
        if n_bonus >= 2:
            break
    assert n_bonus >= 2
    # a ⊥-root row: count residual-mass launches with and without dedup
    calls = {"n": 0}
    real = ds2.residual_block_masses

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(ds2, "residual_block_masses", counting)
    found = False
    for seed in range(40):
        zp, zq, vlim, draft = _case(seed, g=300, sharp=0.05)
        _, _, lqx, _ = _gathered(zp, zq, vlim, draft, 100 + seed, (1.25,))
        monkeypatch.setattr(ds2, "_FORCE_DEDUP", False)
        calls["n"] = 0
        a = _sharded(zp, zq, vlim, draft, lqx, 100 + seed, (1.25,))
        n_off = calls["n"]
        monkeypatch.setattr(ds2, "_FORCE_DEDUP", None)
        calls["n"] = 0
        b = _sharded(zp, zq, vlim, draft, lqx, 100 + seed, (1.25,))
        assert a[0] == b[0]
        if a[0][1.25][0] == 0:
            assert calls["n"] == n_off - 1
            found = True
            break
    assert found, "no ⊥-root row among the seeds"


@cuda
def test_vwall_check_fires_on_a_wrong_bonus_reuse(monkeypatch):
    """R36 (f) mutant: dedup reusing the bonus emission for an envelope that was NOT accepted whole must be caught by
    THUNDERSYNC_VWALL_CHECK's comparison (ds2.vwall_check) against the per-envelope path; the correct dedup passes."""
    monkeypatch.setattr(ds2, "_FORCE_DEDUP", None)
    hits = 0
    for seed in range(40):
        zp, zq, vlim, draft = _case(seed, g=300, accept_all=True)  # M 1 always whole, M 1.25 whole w.p. 0.8
        _, _, lqx, _ = _gathered(zp, zq, vlim, draft, 100 + seed, (1.0, 1.25))
        rows = ds2.ShardedRows(zp, vlim, None, 0)
        lpx = rows.token_logprobs(draft)

        def q_local(idx):
            zz = zq[idx.to("cuda")].contiguous()
            return zz, _lse(zz, vlim)

        def run(mutant):
            if mutant:
                monkeypatch.setattr(ds2, "_reuse_bonus", lambda dedup, t, g: dedup)
            else:
                monkeypatch.setattr(ds2, "_reuse_bonus", lambda dedup, t, g: dedup and t == g)
            gen = torch.Generator(device="cuda").manual_seed(7 + seed)
            before = gen.get_state()
            out, n = ds2.grid_scan_sharded(rows, lpx, lqx, q_local, gen, (1.0, 1.25))
            return ds2.vwall_check(rows, lpx, lqx, q_local, before, (1.0, 1.25), out, n, gen.get_state()), out

        ok, out = run(False)
        assert ok
        g = draft.numel()
        # the mutant only differs when the first envelope is accepted whole and a later one is not
        if out[1.0][0] == g and out[1.25][0] != g:
            bad, _ = run(True)
            assert not bad
            hits += 1
    assert hits, "no row with a whole first envelope and a non-whole second one"
