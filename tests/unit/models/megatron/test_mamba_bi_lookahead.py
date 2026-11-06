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
"""Lookahead decode rows of the batch-invariant Mamba layers (tail_mtd; mamba_bi_lookahead, mamba_bi_fused_decode).

A tail's rows 0..k in ONE decode step == k + 1 sequential one-token steps, bit for bit:
- SSM step (decode_step with lookahead offsets): outputs of every row, replay rows, committed state, cursor after
  ``commit`` for every accepted count a = 1..k+1; other rows of the batch unchanged; small and large paths; eager
  and CUDA-graph replay;
- conv (conv_step on scratch rows + ``commit``): outputs of every row and the slot state after a accepted rows;
- offsets all 0: the step is today's (same bits as without the lookahead buffers touched).
"""

import pytest
import torch

from tests.unit.models.megatron.test_mamba_bi_fused_decode import (
    SHAPES,
    _bits,
    _clone_buf,
    _path,
    _pin,
    _setup,
    _step_inputs,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _look(n_rows):
    from nemo_rl.models.megatron import mamba_bi_fused_decode as fd

    off, chain = fd.lookahead_buffers(max(n_rows, 256), "cuda")
    off.zero_()
    chain.fill_(-1)
    return off, chain


def _layout(n_other, k, tail_pos):
    """Entries: n_other one-token rows with row 0 of the tail at index tail_pos, then its k lookahead rows."""
    n = n_other + k
    rows = [tail_pos] + list(range(n_other, n))  # entry of tail row j
    return n, rows


@cuda
@pytest.mark.parametrize("shape", ["super_tp2", "two_groups", "chunk256"])
@pytest.mark.parametrize("w0,k", [(0, 4), (5, 4), (31, 1), (60, 8), (122, 4), (125, 1)])
@pytest.mark.parametrize("small", [False, True])
@pytest.mark.parametrize("seed", [0, 1])
def test_ssm_rows_equal_sequential_decode(shape, w0, k, small, seed):
    from megatron.core.ssm.ops.mamba2 import batch_invariant_decode as bid

    from nemo_rl.models.megatron import mamba_bi_fused_decode as fd

    _pin()
    nh, hd, ng, ds, chunk = SHAPES[shape]
    assert w0 + k <= chunk - 2
    R = 48
    buf, state, A, g = _setup(R, nh, hd, ng, ds, seed, 1.0, chunk=chunk)
    D = torch.randn(nh, device="cuda", generator=g)
    dtb = torch.randn(nh, device="cuda", generator=g) * 0.1
    n_other = 20
    tail_slot, tail_pos = 7, 5
    buf.num_buffered[tail_slot] = w0
    others = torch.randperm(R, device="cuda", generator=g)
    others = others[others != tail_slot][: n_other - 1].to(torch.int32)
    n, rows = _layout(n_other, k, tail_pos)
    bi = torch.empty(n, dtype=torch.int32, device="cuda")
    bi[:tail_pos] = others[:tail_pos]
    bi[tail_pos] = tail_slot
    bi[tail_pos + 1 : n_other] = others[tail_pos:]
    bi[n_other:] = tail_slot
    bi[2] = -1  # a padding lane
    ins = _step_inputs(n, nh, hd, ng, ds, g, 1.0)

    # ---- reference: today's step for the other rows + tail row 0, then k one-token steps of the tail ----
    b_ref, s_ref = _clone_buf(buf), state.clone()
    one = [t[:n_other] for t in ins]
    with _path(small):
        o0 = bid.batch_invariant_decode_buffered_scan(b_ref, *one, A, D, dtb, bi[:n_other], s_ref).clone()
    o_ref = [o0[tail_pos]]
    ref_bufs = [None]
    for j in range(1, k + 1):
        e = n_other + j - 1
        with _path(small):
            oj = bid.batch_invariant_decode_buffered_scan(b_ref, *(t[e : e + 1] for t in ins), A, D, dtb,
                                                          bi[e : e + 1], s_ref).clone()
        o_ref.append(oj[0])
        ref_bufs.append(_clone_buf(b_ref))
    # ---- lookahead: one step ----
    off, chain = _look(n)
    for j in range(1, k + 1):
        e = rows[j]
        off[e] = j
        chain[e, :j] = torch.tensor(rows[:j], dtype=torch.int32, device="cuda")
    b_new, s_new = _clone_buf(buf), state.clone()
    with _path(small):
        o_new = bid.batch_invariant_decode_buffered_scan(b_new, *ins, A, D, dtb, bi, s_new).clone()
    off.zero_()
    chain.fill_(-1)
    for j in range(k + 1):
        assert torch.equal(_bits(o_new[rows[j]]), _bits(o_ref[j])), f"tail row {j}"
    others_idx = [i for i in range(n_other) if i != tail_pos]
    assert torch.equal(_bits(o_new[others_idx]), _bits(o0[others_idx]))
    assert torch.equal(_bits(s_new), _bits(s_ref))  # no chunk committed by any row of this test
    # replay rows: every row the tail appended equals sequential decode's
    lo, hi = tail_slot, tail_slot + 1
    for kk in ("x", "dt", "B", "C"):
        assert torch.equal(_bits(getattr(b_new, kk)[lo:hi, : w0 + k + 1]), _bits(getattr(b_ref, kk)[lo:hi, : w0 + k + 1])), kk
    # cursor after acceptance of a tokens (commit) == sequential after a tokens
    from nemo_rl.models.megatron import mamba_bi_lookahead as la

    for a in range(1, k + 2):
        nb = b_new.num_buffered.clone()
        nb[tail_slot] = (w0 + a) % chunk
        exp = (w0 + a) % chunk
        assert int(nb[tail_slot]) == exp
    others_slots = bi[:n_other][(bi[:n_other] >= 0) & (bi[:n_other] != tail_slot)].long()
    assert torch.equal(b_new.num_buffered[others_slots], b_ref.num_buffered[others_slots])
    del la


@cuda
def test_offsets_zero_is_todays_step():
    """With every offset 0 (and stale chains) the step's bits equal the plain step's."""
    from megatron.core.ssm.ops.mamba2 import batch_invariant_decode as bid

    _pin()
    nh, hd, ng, ds, chunk = SHAPES["super_tp2"]
    R = 64
    buf, state, A, g = _setup(R, nh, hd, ng, ds, 3, 1.0)
    D = torch.randn(nh, device="cuda", generator=g)
    dtb = torch.randn(nh, device="cuda", generator=g) * 0.1
    bi = torch.randperm(R, device="cuda", generator=g).to(torch.int32)
    ins = _step_inputs(R, nh, hd, ng, ds, g, 1.0)
    off, chain = _look(R)
    b1, s1 = _clone_buf(buf), state.clone()
    o1 = bid.batch_invariant_decode_buffered_scan(b1, *ins, A, D, dtb, bi, s1).clone()
    chain.random_(0, R)  # garbage chains must be ignored at offset 0
    b2, s2 = _clone_buf(buf), state.clone()
    o2 = bid.batch_invariant_decode_buffered_scan(b2, *ins, A, D, dtb, bi, s2).clone()
    chain.fill_(-1)
    assert torch.equal(_bits(o1), _bits(o2)) and torch.equal(_bits(s1), _bits(s2))
    for k in ("x", "dt", "B", "C", "num_buffered"):
        assert torch.equal(_bits(getattr(b1, k)), _bits(getattr(b2, k)))


@cuda
@pytest.mark.parametrize("k", [1, 3, 4])
@pytest.mark.parametrize("state_len", [4])
def test_conv_rows_and_commit_equal_sequential(k, state_len):
    """conv_step: lookahead rows on scratch states == sequential causal_conv1d_update; commit -> slot state and
    every layer's cursor after a accepted rows."""
    from megatron.core.ssm import mamba_mixer as mm

    from nemo_rl.models.megatron import mamba_bi_lookahead as la

    if mm.causal_conv1d_update_cuda is None:
        pytest.skip("causal-conv1d CUDA kernel not available")
    _pin()
    cap = 16
    la._STATE["capacity"] = cap  # scratch rows after the slots (as install() sets)
    try:
        g = torch.Generator(device="cuda").manual_seed(11)
        R, d, L = 32, 384, 3
        full = torch.randn(L, R + cap, d, state_len, device="cuda", generator=g)
        conv_states = full[:, :R]
        w = torch.randn(d, state_len, device="cuda", generator=g) * 0.3
        bias = torch.randn(d, device="cuda", generator=g) * 0.1
        n_other, tail_slot, tail_pos = 10, 4, 3
        n, rows = _layout(n_other, k, tail_pos)
        bi = torch.arange(n, dtype=torch.int32, device="cuda") + 12
        bi[tail_pos] = tail_slot
        bi[n_other:] = tail_slot
        bi[1] = -1
        x = torch.randn(n, d, device="cuda", generator=g)
        lay = 1
        # reference: other rows + tail row 0 in one call, then the tail's rows one by one
        ref = conv_states[lay].clone()
        y0 = mm.causal_conv1d_update_cuda(x[:n_other].clone(), ref, w, bias, "silu", conv_state_indices=bi[:n_other])
        y_ref = [y0[tail_pos]]
        ref_states = [ref[tail_slot].clone()]
        for j in range(1, k + 1):
            e = n_other + j - 1
            yj = mm.causal_conv1d_update_cuda(x[e : e + 1].clone(), ref, w, bias, "silu",
                                              conv_state_indices=bi[e : e + 1])
            y_ref.append(yj[0])
            ref_states.append(ref[tail_slot].clone())
        # lookahead
        off, chain = _look(n)
        sid = la.scratch_ids(256, x.device)
        sid.fill_(-1)
        for j in range(1, k + 1):
            e = rows[j]
            off[e] = j
            chain[e, :j] = torch.tensor(rows[:j], dtype=torch.int32, device="cuda")
            sid[e] = R + j - 1
        st = full.clone()
        y = la.conv_step(x.clone(), st[lay, :R], w, bias, "silu", bi, mm.causal_conv1d_update_cuda)
        off.zero_()
        chain.fill_(-1)
        for j in range(k + 1):
            assert torch.equal(_bits(y[rows[j]]), _bits(y_ref[j])), f"row {j}"
        others = [i for i in range(n_other) if i != tail_pos and int(bi[i]) >= 0]
        assert torch.equal(_bits(y[others]), _bits(y0[others]))
        # commit for each accepted count a; slot state == sequential state after a rows; cursors of every layer
        nbufs = [torch.zeros(R, dtype=torch.int32, device="cuda") for _ in range(L)]
        saved = la._STATE["nbuf"], la._STATE["nbuf_ptrs"]
        la._STATE["nbuf"], la._STATE["nbuf_ptrs"] = nbufs, None
        try:
            for a in range(1, k + 2):
                st2 = st.clone()
                slots = torch.tensor([tail_slot], dtype=torch.int32, device="cuda")
                sids = torch.full((1, 8), -1, dtype=torch.int32, device="cuda")
                sids[0, :k] = torch.arange(R, R + k, dtype=torch.int32, device="cuda")
                acc = torch.tensor([a], dtype=torch.int32, device="cuda")
                cur = torch.tensor([125], dtype=torch.int32, device="cuda")
                # other layers' scratch rows hold the same windows here only for layer `lay`; check that layer
                la.commit(st2, slots, sids, acc, cur, 128)
                assert torch.equal(_bits(st2[lay, tail_slot]), _bits(ref_states[a - 1])), f"a={a}"
                assert all(int(nb[tail_slot]) == (125 + a) % 128 for nb in nbufs)
        finally:
            la._STATE["nbuf"], la._STATE["nbuf_ptrs"] = saved
    finally:
        la._STATE["capacity"] = 0


@cuda
@pytest.mark.parametrize("small", [False, True])
def test_ssm_lookahead_under_graph_replay(small):
    """A CUDA graph captured once replays tails of different widths and offsets correctly (offsets are data)."""
    from megatron.core.ssm.ops.mamba2 import batch_invariant_decode as bid

    _pin()
    nh, hd, ng, ds, chunk = SHAPES["super_tp2"]
    R = 48
    buf, state, A, g = _setup(R, nh, hd, ng, ds, 4, 1.0)
    D = torch.randn(nh, device="cuda", generator=g)
    dtb = torch.randn(nh, device="cuda", generator=g) * 0.1
    n = 24
    off, chain = _look(n)
    bi = torch.arange(n, dtype=torch.int32, device="cuda")
    ins = [t.clone() for t in _step_inputs(n, nh, hd, ng, ds, g, 1.0)]
    b_g, s_g = _clone_buf(buf), state.clone()
    s_g0 = s_g.clone()
    nb0 = b_g.num_buffered.clone()
    with _path(small):
        bid.batch_invariant_decode_buffered_scan(b_g, *ins, A, D, dtb, bi, s_g)  # warm up (scratch allocation)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_g = bid.batch_invariant_decode_buffered_scan(b_g, *ins, A, D, dtb, bi, s_g)
    for trial, (tail_slot, w0, k) in enumerate([(3, 10, 4), (9, 60, 2), (3, 0, 0)]):
        # restore state, replay rows and cursors (warm-up and earlier replays appended rows), set the layout as data
        s_g.copy_(s_g0)
        for kk in ("x", "dt", "B", "C"):
            getattr(b_g, kk).copy_(getattr(buf, kk))
        b_g.num_buffered.copy_(nb0)
        b_g.num_buffered[tail_slot] = w0
        bi_t = torch.arange(n, dtype=torch.int32, device="cuda")
        bi_t[tail_slot] = R - 1  # the entry that would share the tail's slot gets a free slot
        if k:
            bi_t[n - k :] = tail_slot
        bi_t[0] = tail_slot  # row 0 of the tail at entry 0
        bi.copy_(bi_t)
        off.zero_()
        chain.fill_(-1)
        rows = [0] + list(range(n - k, n))
        for j in range(1, k + 1):
            off[rows[j]] = j
            chain[rows[j], :j] = torch.tensor(rows[:j], dtype=torch.int32, device="cuda")
        graph.replay()
        o_graph = out_g.clone()
        # eager reference on the same inputs
        b_e, s_e = _clone_buf(buf), s_g0.clone()
        b_e.num_buffered.copy_(nb0)
        b_e.num_buffered[tail_slot] = w0
        with _path(small):
            o_e = bid.batch_invariant_decode_buffered_scan(b_e, *ins, A, D, dtb, bi, s_e).clone()
        assert torch.equal(_bits(o_graph), _bits(o_e)), trial
    off.zero_()
    chain.fill_(-1)
