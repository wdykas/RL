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
"""Tree node rows of the batch-invariant Mamba layers (tree_spec; mamba_bi_fused_decode scratch replay rows,
tree_kv.commit_mamba), bitwise against sequential one-token decode.

A tail's root (its row 0) and a token tree of node rows in ONE decode step: every node's output equals one-token
decode of the root followed by the node's ancestors and the node, one at a time, bit for bit (siblings share a depth,
so they write scratch rows, never the slot's row). After acceptance of any root-to-node path, ``commit_mamba`` leaves
the slot's replay rows, cursor and conv state equal to sequential decode of that path. The negative
(skip_replay = tree_stale_mamba) must leave the slot's rows different. Small and large scan paths, two shapes,
several cursors.
"""

import contextlib
import copy
from types import SimpleNamespace

import pytest
import torch

from tests.unit.models.megatron.test_mamba_bi_fused_decode import (
    SHAPES,
    _bits,
    _path,
    _pin,
    _step_inputs,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# tree: parent of node i (-1 = root); nodes are entries after the batch rows, in this order
TREES = {
    "fork": [-1, -1, 0, 1, 1, 2],  # root -> {0, 1}; 0 -> 2; 1 -> {3, 4}; 2 -> 5
    "chain": [-1, 0, 1, 2],
    "wide": [-1, -1, -1, -1, 0, 0, 3],
}


def _depths(par):
    d = []
    for i, p in enumerate(par):
        d.append(1 if p < 0 else d[p] + 1)
    return d


def _path_to(par, i):
    out = []
    while i >= 0:
        out.append(i)
        i = par[i]
    return out[::-1]


def _setup(R, nh, hd, ng, ds, seed, scale, chunk=128):
    """Decode buffers as MCore allocates them on the installed pin (z-free on Megatron-LM main, where the dataclass has
    no z field), filled with random replay rows; slot states; A."""
    from megatron.core.ssm.ops.mamba2 import batch_invariant_decode as bid

    g = torch.Generator(device="cuda").manual_seed(seed)
    buf = bid.BatchInvariantDecodeBuffers.allocate(R, chunk, nh, hd, ng, ds, torch.device("cuda"), torch.float32)
    assert getattr(buf, "z", None) is None, "the decode step must serve this shape"
    for t in (buf.x, buf.dt, buf.B, buf.C):
        t.copy_(torch.randn(t.shape, device="cuda", generator=g) * scale)
    buf.dt.sub_(2.0)
    nb = torch.arange(R, device="cuda") % chunk
    nb = torch.where(torch.arange(R, device="cuda") % 5 == 0, chunk - 1, nb)
    buf.num_buffered.copy_(nb.to(torch.int32))
    state = torch.randn(R, nh, hd, ds, device="cuda", generator=g) * 0.1
    A = -torch.exp(torch.empty(nh, device="cuda").uniform_(-9.2, 0.7, generator=g))
    return buf, state, A, g


def _clone_buf(buf):
    """A copy of decode buffers with every tensor attribute cloned (any buffers class: with z before Megatron-LM #8221,
    MCore main's z-free dataclass, nemo_rl's)."""
    b = copy.copy(buf)
    for k, v in vars(buf).items():
        if torch.is_tensor(v):
            setattr(b, k, v.clone())
    return b


@contextlib.contextmanager
def _registered(bufs):
    """tree_kv's Mamba registry holding exactly ``bufs`` (layer order) for the block; restored after."""
    from nemo_rl.models.generation.megatron import tree_kv

    saved = dict(tree_kv._MAMBA)
    tree_kv._MAMBA.update(bufs=list(enumerate(bufs)), nbuf_ptrs=None, replay_ptrs=None)
    try:
        yield
    finally:
        tree_kv._MAMBA.clear()
        tree_kv._MAMBA.update(saved)


def _ctx(conv, slots):
    """The context fields commit_mamba reads: the full conv states and MCore's view of its slots."""
    return SimpleNamespace(tree_conv_states_full=conv, mamba_conv_states=conv[:, :slots])


def _scratch(buf, cap):
    R, chunk, nh, hd = buf.x.shape
    ng, ds = buf.B.shape[-2:]
    z = lambda *s: torch.zeros(*s, device="cuda", dtype=torch.float32)  # noqa: E731
    return (z(cap, nh, hd), z(cap, nh), z(cap, ng, ds), z(cap, ng, ds))


@cuda
@pytest.mark.parametrize("shape", ["super_tp2", "two_groups"])
@pytest.mark.parametrize("tree", list(TREES))
@pytest.mark.parametrize("w0", [0, 9, 100])
@pytest.mark.parametrize("small", [False, True])
def test_tree_rows_equal_sequential_decode_and_commit(shape, tree, w0, small):
    from megatron.core.ssm.ops.mamba2 import batch_invariant_decode as bid

    from nemo_rl.models.generation.megatron import tree_kv
    from nemo_rl.models.megatron import mamba_bi_fused_decode as fd

    _pin()
    nh, hd, ng, ds, chunk = SHAPES[shape]
    par = TREES[tree]
    dep = _depths(par)
    K = len(par)
    assert w0 + max(dep) <= chunk - 2
    R = 40
    buf, state, A, g = _setup(R, nh, hd, ng, ds, 7 + K + w0, 1.0, chunk=chunk)
    D = torch.randn(nh, device="cuda", generator=g)
    dtb = torch.randn(nh, device="cuda", generator=g) * 0.1
    n_other, tail_slot, tail_pos = 12, 5, 3
    buf.num_buffered[tail_slot] = w0
    others = torch.randperm(R, device="cuda", generator=g)
    others = others[others != tail_slot][: n_other].to(torch.int32)
    n = n_other + K
    bi = others.clone()
    bi = torch.cat([bi, torch.full((K,), tail_slot, dtype=torch.int32, device="cuda")])
    bi[tail_pos] = tail_slot
    bi[1] = -1  # a padding lane
    ins = _step_inputs(n, nh, hd, ng, ds, g, 1.0)
    node_e = [n_other + i for i in range(K)]

    # ---- reference per node: root step (with the batch), then the path's one-token steps ----
    b0, s0 = _clone_buf(buf), state.clone()
    with _path(small):
        o_all = bid.batch_invariant_decode_buffered_scan(b0, *(t[:n_other] for t in ins), A, D, dtb, bi[:n_other],
                                                         s0).clone()
    ref_out, ref_buf = {}, {}
    for i in range(K):
        b, s = _clone_buf(b0), s0.clone()
        for j in _path_to(par, i):
            e = node_e[j]
            with _path(small):
                o = bid.batch_invariant_decode_buffered_scan(b, *(t[e: e + 1] for t in ins), A, D, dtb,
                                                             bi[e: e + 1], s).clone()
        ref_out[i], ref_buf[i] = o[0], b

    # ---- tree: one step ----
    off, chain = fd.lookahead_buffers(256, "cuda")
    srow = fd.lookahead_rows(256, "cuda")
    off.zero_()
    chain.fill_(-1)
    srow.fill_(-1)
    for i in range(K):
        e = node_e[i]
        p = _path_to(par, i)  # root's child .. i
        off[e] = dep[i]
        chain[e, 0] = tail_pos
        for k, j in enumerate(p[:-1]):
            chain[e, k + 1] = node_e[j]
        for k, j in enumerate(p):
            srow[e, k] = j  # scratch row of node j = its index
    b1, s1 = _clone_buf(b0 if False else buf), state.clone()
    b1.scratch = _scratch(b1, 16)
    with _path(small):
        o_tree = bid.batch_invariant_decode_buffered_scan(b1, *ins, A, D, dtb, bi, s1).clone()
    off.zero_()
    chain.fill_(-1)
    srow.fill_(-1)
    # batch rows (incl. the root) unchanged; every node == its sequential reference
    keep = [i for i in range(n_other) if int(bi[i]) >= 0]
    assert torch.equal(_bits(o_tree[keep]), _bits(o_all[keep]))
    for i in range(K):
        assert torch.equal(_bits(o_tree[node_e[i]]), _bits(ref_out[i])), f"node {i}"
    assert torch.equal(_bits(s1), _bits(s0))  # no chunk committed by a node
    # the slot's rows above w0 untouched by the nodes (they wrote scratch rows)
    for k in ("x", "dt", "B", "C"):
        assert torch.equal(_bits(getattr(b1, k)[tail_slot, w0 + 1:]), _bits(getattr(buf, k)[tail_slot, w0 + 1:])), k

    # ---- root-only commit (a forward sampled after an engine resume: a = 1): the node rows moved the slot's write
    # cursor; the commit puts it back to w0 + 1 and the slot equals the root's one-token step (review MED-2) ----
    bc = _clone_buf(b1)
    bc.scratch = b1.scratch
    conv = torch.randn(1, R + 16, 8, 4, device="cuda")
    conv0 = conv.clone()
    with _registered([bc]):
        tree_kv.commit_mamba(_ctx(conv, R), torch.tensor([tail_slot], dtype=torch.int32, device="cuda"),
                             torch.full((1, 8), -1, dtype=torch.int32, device="cuda"),
                             torch.tensor([1], dtype=torch.int32, device="cuda"),
                             torch.tensor([w0], dtype=torch.int32, device="cuda"), chunk)
    assert int(bc.num_buffered[tail_slot]) == int(b0.num_buffered[tail_slot]) == (w0 + 1) % chunk
    for k in ("x", "dt", "B", "C"):
        assert torch.equal(_bits(getattr(bc, k)[tail_slot, : w0 + 1]), _bits(getattr(b0, k)[tail_slot, : w0 + 1])), k
    assert torch.equal(conv, conv0)  # a = 1: the root already updated the slot's conv state in place

    # ---- commit of every root-to-node path (and the negative) ----
    for i in range(K):
        p = _path_to(par, i)
        a = len(p) + 1  # root's token + the path's nodes' draws
        for skip in (False, True):
            bc = _clone_buf(b1)
            bc.scratch = b1.scratch
            conv = torch.zeros(1, R + 16, 8, 4, device="cuda")
            path = torch.full((1, 8), -1, dtype=torch.int32, device="cuda")
            path[0, : len(p)] = torch.tensor(p, dtype=torch.int32, device="cuda")
            with _registered([bc]):
                tree_kv.commit_mamba(_ctx(conv, R), torch.tensor([tail_slot], dtype=torch.int32, device="cuda"), path,
                                     torch.tensor([a], dtype=torch.int32, device="cuda"),
                                     torch.tensor([w0], dtype=torch.int32, device="cuda"), chunk, skip_replay=skip)
            assert int(bc.num_buffered[tail_slot]) == (w0 + a) % chunk
            rb = ref_buf[i]
            same = all(torch.equal(_bits(getattr(bc, k)[tail_slot, : w0 + a]),
                                   _bits(getattr(rb, k)[tail_slot, : w0 + a])) for k in ("x", "dt", "B", "C"))
            if not skip:
                assert same, f"path to {i}"
            else:
                assert not same, f"negative tree_stale_mamba left the slot right (path to {i})"


@cuda
@pytest.mark.parametrize("tree", list(TREES))
def test_tree_conv_windows_and_commit(tree):
    """conv_step on a tree: each node's scratch window is [slot state, x(root), x(ancestors)] (chain), so its
    output equals sequential causal_conv1d_update along its path; commit_mamba copies the depth a-1 node's state."""
    from megatron.core.ssm import mamba_mixer as mm

    from nemo_rl.models.generation.megatron import tree_kv
    from nemo_rl.models.megatron import mamba_bi_fused_decode as fd

    if mm.causal_conv1d_update_cuda is None:
        pytest.skip("causal-conv1d CUDA kernel not available")
    _pin()
    par = TREES[tree]
    dep = _depths(par)
    K = len(par)
    cap = 16
    g = torch.Generator(device="cuda").manual_seed(5)
    R, d, w = 24, 256, 4
    full = torch.randn(1, R + cap, d, w, device="cuda", generator=g)
    wt = torch.randn(d, w, device="cuda", generator=g) * 0.3
    bias = torch.randn(d, device="cuda", generator=g) * 0.1
    n_other, tail_slot, tail_pos = 6, 3, 2
    n = n_other + K
    bi = torch.arange(n, dtype=torch.int32, device="cuda") + 8
    bi[tail_pos] = tail_slot
    bi[n_other:] = tail_slot
    x = torch.randn(n, d, device="cuda", generator=g)
    node_e = [n_other + i for i in range(K)]
    # reference: root (with the batch), then each node's path one by one
    base = full[0, :R].clone()
    y0 = mm.causal_conv1d_update_cuda(x[:n_other].clone(), base, wt, bias, "silu", conv_state_indices=bi[:n_other])
    ref_y, ref_state = {}, {}
    for i in range(K):
        st = base.clone()
        for j in _path_to(par, i):
            e = node_e[j]
            yj = mm.causal_conv1d_update_cuda(x[e: e + 1].clone(), st, wt, bias, "silu",
                                              conv_state_indices=bi[e: e + 1])
        ref_y[i], ref_state[i] = yj[0], st[tail_slot].clone()
    off, chain = fd.lookahead_buffers(256, "cuda")
    off.zero_()
    chain.fill_(-1)
    sid = tree_kv.scratch_ids(256, x.device)  # a torch.device: the step buffers are keyed by its index
    sid.fill_(-1)
    for i in range(K):
        e = node_e[i]
        off[e] = dep[i]
        chain[e, 0] = tail_pos
        for k, j in enumerate(_path_to(par, i)[:-1]):
            chain[e, k + 1] = node_e[j]
        sid[e] = R + i
    stf = full.clone()
    y = tree_kv.conv_step(x.clone(), stf[0, :R], stf[0], wt, bias, "silu", bi, mm.causal_conv1d_update_cuda)
    off.zero_()
    chain.fill_(-1)
    # node-free rows (the batch, the root): bitwise MCore's batch-invariant decode conv on the slots, state included
    assert torch.equal(_bits(y[:n_other]), _bits(y0))
    assert torch.equal(_bits(stf[0, :R]), _bits(base))
    for i in range(K):
        assert torch.equal(_bits(y[node_e[i]]), _bits(ref_y[i])), f"node {i}"
    # commit: conv state <- the last path node's scratch row (replay rows: no buffers registered -> L = 0 guard)
    nb = torch.zeros(R, dtype=torch.int32, device="cuda")
    for i in range(K):
        p = _path_to(par, i)
        st2 = stf.clone()

        class _B:  # one registered layer with tiny replay buffers (copies not checked here)
            pass

        fake = _B()
        fake.x = torch.zeros(R, 128, 1, 4, device="cuda")
        fake.dt = torch.zeros(R, 128, 1, device="cuda")
        fake.B = torch.zeros(R, 128, 1, 4, device="cuda")
        fake.C = torch.zeros(R, 128, 1, 4, device="cuda")
        fake.scratch = (torch.zeros(cap, 1, 4, device="cuda"), torch.zeros(cap, 1, device="cuda"),
                        torch.zeros(cap, 1, 4, device="cuda"), torch.zeros(cap, 1, 4, device="cuda"))
        fake.num_buffered = nb
        path = torch.full((1, 8), -1, dtype=torch.int32, device="cuda")
        path[0, : len(p)] = torch.tensor(p, dtype=torch.int32, device="cuda")
        with _registered([fake]):
            tree_kv.commit_mamba(_ctx(st2, R), torch.tensor([tail_slot], dtype=torch.int32, device="cuda"), path,
                                 torch.tensor([len(p) + 1], dtype=torch.int32, device="cuda"),
                                 torch.tensor([7], dtype=torch.int32, device="cuda"), 128)
        assert torch.equal(_bits(st2[0, tail_slot]), _bits(ref_state[i])), f"commit path to {i}"
        assert int(nb[tail_slot]) == (7 + len(p) + 1) % 128
