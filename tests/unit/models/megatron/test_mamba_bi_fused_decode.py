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
"""The batch-invariant Mamba decode step (mamba_bi_fused_decode): MCore's training-tile bits, row-independent,
graph-capturable, on every shape it accepts.

- one step vs MCore's z-buffer tile pipeline (the training chunk-scan kernels) on a twin of the buffers: outputs,
  committed states, replay buffers and cursors; every target row 0..127, crossings, padding lanes, signed zeros,
  scales, fp32 / bf16 / fp16 D, both paths (CB and boundary inside the scan program or not), several shapes;
- 300 decode steps vs MCore's z-buffer path, eager and CUDA-graph replay;
- row independence: a row's bits do not depend on the batch (subsets, permutations, small vs large path);
- vs the learner: a MambaMixer's training forward through mamba_ssd_parity == engine prefill + this decode;
- the support rule: declines (z buffers, MCore's path) for unpinned or inexact tile blockings and other shapes.
"""

import contextlib

import pytest
import torch

from nemo_rl.models.megatron import mamba_bi_decode as mbd
from nemo_rl.models.megatron import mamba_bi_fused_decode as fd

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _pin():
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import _pin_mamba_autotuners

    from nemo_rl.models.megatron import mamba_ssd_parity

    _pin_mamba_autotuners()
    mamba_ssd_parity.pin_engine_kernels()
    mbd.install()  # installs the decode step


def _main_pin() -> bool:
    """Megatron-LM main (#8221): MCore allocates and seeds z-free decode buffers itself and its own buffered scan is the
    batch-invariant reference; there is no z-buffer path any more."""
    mbd.install()
    return bool(mbd._ORIG.get("mcore_z_free", False))


def _reference():
    """(allocate, seed, scan) of the reference decode: MCore's z-buffer path before #8221; Megatron-LM main's own z-free
    buffered decode on main (the function this step replaced: mamba_bi_fused_decode._ORIG["scan"])."""
    from megatron.core.ssm.ops.mamba2 import batch_invariant_decode as bid

    cls = bid.BatchInvariantDecodeBuffers
    alloc = mbd._ORIG["allocate"].__get__(None, cls)
    if _main_pin():
        return alloc, cls.seed, fd._ORIG["scan"]
    return alloc, mbd._ORIG["seed"], mbd._ORIG["scan"]


@contextlib.contextmanager
def _path(small):
    """Force the small (CB + boundary in the scan program) or the large path; None = the one-wave rule."""
    orig = fd.small_step
    if small is not None:
        fd.small_step = lambda *a, **k: small
    try:
        yield
    finally:
        fd.small_step = orig


def _bits(t):
    return t.contiguous().view(torch.int32)


def _step_inputs(n, nh, hd, ng, ds, g, scale):
    def rnd(*s):
        return torch.randn(*s, device="cuda", generator=g) * scale

    # x / B / C as strided views of one xBC row (the mixer's split), z / dt of zxBCdt.
    xbc = rnd(n, 1, nh * hd + 2 * ng * ds)
    x, B, C = torch.split(xbc, [nh * hd, ng * ds, ng * ds], dim=-1)
    zxdt = rnd(n, 1, nh * hd + nh)
    z, dt = torch.split(zxdt, [nh * hd, nh], dim=-1)
    return x.view(n, 1, nh, hd), z.view(n, 1, nh, hd), dt - 2.0, B.view(n, 1, ng, ds), C.view(n, 1, ng, ds)


def _setup(R, nh, hd, ng, ds, seed, scale, signed_zeros=False, cursor=None, chunk=128):
    from megatron.core.ssm.ops.mamba2 import batch_invariant_decode as bid

    g = torch.Generator(device="cuda").manual_seed(seed)
    buf = bid.BatchInvariantDecodeBuffers.allocate(R, chunk, nh, hd, ng, ds, torch.device("cuda"), torch.float32)
    assert getattr(buf, "z", None) is None, "the decode step must serve this shape"
    # on Megatron-LM main every buffer is z-free: the step serving the shape is proven by _serves, else the test would
    # compare MCore's scan with itself (review batch 51 V1)
    assert not _main_pin() or fd._serves(buf), "the decode step must serve this shape"
    for t in (buf.x, buf.dt, buf.B, buf.C):
        t.copy_(torch.randn(t.shape, device="cuda", generator=g) * scale)
    buf.dt.sub_(2.0)
    if signed_zeros:
        buf.x.view(-1)[::7] = 0.0
        buf.x.view(-1)[1::11] = -0.0
    if cursor is None:
        nb = torch.arange(R, device="cuda") % chunk  # every position
        nb = torch.where(torch.arange(R, device="cuda") % 5 == 0, chunk - 1, nb)  # ~1 in 5 crosses
    else:
        nb = torch.full((R,), cursor, device="cuda")
    buf.num_buffered.copy_(nb.to(torch.int32))
    state = torch.randn(R, nh, hd, ds, device="cuda", generator=g) * 0.1
    if signed_zeros:
        state[..., ::5] = 0.0
    A = -torch.exp(torch.empty(nh, device="cuda").uniform_(-9.2, 0.7, generator=g))
    return buf, state, A, g


def _clone_buf(buf):
    import copy

    b = copy.copy(buf)
    for k, v in vars(buf).items():
        if torch.is_tensor(v):
            setattr(b, k, v.clone())
    for k in ("bi_dt_out", "bi_dA_cumsum", "bi_cb_row"):
        if hasattr(b, k):
            delattr(b, k)
    return b


def _mcore_twin(buf):
    """The reference path's buffers (MCore's, with z before Megatron-LM #8221; MCore main's z-free ones on main) holding
    the same replay rows and cursors; z rows 0 (only the current token's z, written by the step, reaches an output)."""
    R, chunk, nh, hd = buf.x.shape
    ng, ds = buf.B.shape[-2:]
    twin = _reference()[0](R, chunk, nh, hd, ng, ds, buf.x.device, buf.x.dtype)
    assert _main_pin() or twin.z is not None
    for k in ("x", "dt", "B", "C", "num_buffered"):
        getattr(twin, k).copy_(getattr(buf, k))
    return twin


def _step(buf, state, ins, A, D, dtb, bi, small=None):
    """(MCore z-buffer path, the decode step) on copies: outputs, states, buffers."""
    from megatron.core.ssm.ops.mamba2 import batch_invariant_decode as bid

    twin, st_ref = _mcore_twin(buf), state.clone()
    out_ref = _reference()[2](twin, *ins, A, D, dtb, bi, st_ref).clone()
    b, st = _clone_buf(buf), state.clone()
    with _path(small):
        out = bid.batch_invariant_decode_buffered_scan(b, *ins, A, D, dtb, bi, st).clone()
    return (out_ref, st_ref, twin), (out, st, b)


SHAPES = {  # (nheads, head_dim, ngroups, d_state, chunk) per rank
    "super_tp2": (64, 64, 4, 128, 128),  # Super-L22 at TP2
    "two_groups": (16, 64, 2, 128, 128),
    "hd32_ds64": (16, 32, 2, 64, 128),  # other head_dim / d_state
    "chunk256": (16, 64, 2, 128, 256),  # another chunk size
}


@cuda
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("seed,scale", [(0, 1.0), (1, 1e-3), (2, 30.0)])
@pytest.mark.parametrize("d_hdim,d_dtype", [(False, torch.float32), (True, torch.float32), (False, torch.bfloat16),
                                             (True, torch.float16)])
@pytest.mark.parametrize("small", [False, True])
def test_step_matches_mcore_tile_pipeline(shape, seed, scale, d_hdim, d_dtype, small):
    """One step, every target row, crossings, shuffled slots, padding lanes: outputs, states, buffers bitwise."""
    _pin()
    nh, hd, ng, ds, chunk = SHAPES[shape]
    R = 160
    buf, state, A, g = _setup(R, nh, hd, ng, ds, seed, scale, signed_zeros=seed == 2, chunk=chunk)
    # D in its parameter dtype (the engine's mixer holds it in bf16; the kernels convert to fp32 on load)
    D = torch.randn((nh, hd) if d_hdim else (nh,), device="cuda", generator=g).to(d_dtype)
    dtb = torch.randn(nh, device="cuda", generator=g) * 0.1
    n = 150
    bi = torch.randperm(R, device="cuda", generator=g)[:n].to(torch.int32)
    bi[3] = -1
    bi[77] = -1
    ins = _step_inputs(n, nh, hd, ng, ds, g, scale)
    (o_ref, s_ref, b_ref), (o_new, s_new, b_new) = _step(buf, state, ins, A, D, dtb, bi, small)
    assert torch.isfinite(o_ref).all()
    assert not torch.equal(s_ref, state)  # some rows crossed
    assert torch.equal(_bits(o_ref), _bits(o_new))
    assert torch.equal(_bits(s_ref), _bits(s_new))
    for k in ("x", "dt", "B", "C", "num_buffered"):
        assert torch.equal(_bits(getattr(b_ref, k)), _bits(getattr(b_new, k))), k
    for k in ("target_rows", "chunk_flags"):
        assert torch.equal(getattr(b_ref, k)[:n], getattr(b_new, k)[:n]), k
    assert (o_new[3] == 0).all() and (o_new[77] == 0).all()


@cuda
@pytest.mark.parametrize("cursor", [0, 31, 32, 126, 127])
@pytest.mark.parametrize("small", [False, True])
def test_step_all_rows_at_one_cursor(cursor, small):
    """Synchronized cursors (equal prompt lengths): every row at the same position, incl. all crossing at once."""
    _pin()
    nh, hd, ng, ds, chunk = SHAPES["super_tp2"]
    R = 64
    buf, state, A, g = _setup(R, nh, hd, ng, ds, 5, 1.0, cursor=cursor)
    D = torch.randn(nh, device="cuda", generator=g)
    dtb = torch.randn(nh, device="cuda", generator=g) * 0.1
    bi = torch.randperm(R, device="cuda", generator=g).to(torch.int32)
    ins = _step_inputs(R, nh, hd, ng, ds, g, 1.0)
    (o_ref, s_ref, _), (o_new, s_new, _) = _step(buf, state, ins, A, D, dtb, bi, small)
    assert torch.equal(_bits(o_ref), _bits(o_new)) and torch.equal(_bits(s_ref), _bits(s_new))


@cuda
@pytest.mark.parametrize("seed", [0, 1])
def test_row_bits_independent_of_batch(seed):
    """Row r's output and state: same bits alone, in a subset, permuted, and next to extra rows.

    Under the one-wave rule a batch of a few rows takes the small path and a larger one the large path, so this
    also checks that the two give the same bits.
    """
    from megatron.core.ssm.ops.mamba2 import batch_invariant_decode as bid

    from nemo_rl.models.megatron import mamba_bi_scan_cuda

    _pin()
    nh, hd, ng, ds, chunk = SHAPES["super_tp2"]
    R = 96
    buf, state, A, g = _setup(R, nh, hd, ng, ds, seed, 1.0)
    D = torch.randn(nh, device="cuda", generator=g)
    dtb = torch.randn(nh, device="cuda", generator=g) * 0.1
    full = torch.randperm(R, device="cuda", generator=g).to(torch.int32)
    ins_full = _step_inputs(R, nh, hd, ng, ds, g, 1.0)
    scan = mamba_bi_scan_cuda.module(hd, ds, chunk, 32)

    def run(idx):
        b, st = _clone_buf(buf), state.clone()
        out = bid.batch_invariant_decode_buffered_scan(b, *(t[idx] for t in ins_full), A, D, dtb, full[idx], st)
        return out.clone(), st

    o_all, s_all = run(torch.arange(R, device="cuda"))
    smalls = []
    for idx in (
        torch.tensor([5], device="cuda"),
        torch.arange(0, R, 3, device="cuda"),
        torch.randperm(R, device="cuda", generator=g)[:40],
        torch.arange(R - 1, -1, -1, device="cuda"),
    ):
        smalls.append(fd.small_step(len(idx), nh, torch.device("cuda"), scan))
        o, s = run(idx)
        assert torch.equal(_bits(o), _bits(o_all[idx]))
        slots = full[idx].long()
        assert torch.equal(_bits(s[slots]), _bits(s_all[slots]))
    assert smalls[0] and not smalls[-1]  # both paths exercised


@cuda
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("slots,prefill", [((0, 3, 1), (5, 127, 130)), ((7, 2, 5, 0, 9), (128, 1, 255, 64, 300))])
@pytest.mark.parametrize("mixed_decay", [False, True])
@pytest.mark.parametrize("small", [False, True])
def test_300_steps_match_mcore_z_path(graph, slots, prefill, mixed_decay, small):
    """300 decode steps (>= 2 crossings per slot), eager and CUDA-graph replay: == MCore's z-buffer path."""
    from tests.unit.models.megatron.test_mamba_bi_decode import _decode

    _pin()
    kw = dict(steps=300, slots=slots, prefill=prefill, graph=graph, mixed_decay=mixed_decay)
    out_ref, st_ref = _decode(False, **kw)
    with _path(small):
        out_new, st_new = _decode(True, **kw)
    assert torch.isfinite(out_new[:, : len(slots)]).all()
    assert torch.equal(out_ref, out_new) and torch.equal(st_ref, st_new)


@cuda
@pytest.mark.parametrize("small", [False, True])
def test_graph_replay_equals_eager(small):
    """Capture one step (capture fails on any host sync); replay == eager == MCore."""
    from megatron.core.ssm.ops.mamba2 import batch_invariant_decode as bid

    _pin()
    nh, hd, ng, ds, _ = SHAPES["super_tp2"]
    R = 48
    buf, state, A, g = _setup(R, nh, hd, ng, ds, 3, 1.0)
    D = torch.randn(nh, device="cuda", generator=g)
    dtb = torch.randn(nh, device="cuda", generator=g) * 0.1
    bi = torch.randperm(R, device="cuda", generator=g).to(torch.int32)
    ins = _step_inputs(R, nh, hd, ng, ds, g, 1.0)
    (o_ref, s_ref, _), (o_eager, s_eager, _) = _step(buf, state, ins, A, D, dtb, bi, small)
    b, st = _clone_buf(buf), state.clone()
    snap = {k: v.clone() for k, v in vars(b).items() if torch.is_tensor(v)}
    st0 = st.clone()
    with _path(small):
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            bid.batch_invariant_decode_buffered_scan(_clone_buf(buf), *ins, A, D, dtb, bi, state.clone())
        torch.cuda.current_stream().wait_stream(side)
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg):
            out = bid.batch_invariant_decode_buffered_scan(b, *ins, A, D, dtb, bi, st)
    for k, v in snap.items():
        getattr(b, k).copy_(v)
    st.copy_(st0)
    cg.replay()
    torch.cuda.synchronize()
    assert torch.equal(_bits(out), _bits(o_eager)) and torch.equal(_bits(st), _bits(s_eager))
    assert torch.equal(_bits(out), _bits(o_ref)) and torch.equal(_bits(st), _bits(s_ref))


@cuda
def test_support_rule():
    """Unpinned SSD tiles, a chunk-scan tile whose K blocking adds masked K steps, or shapes the CUDA scan does not
    take: z buffers (MCore's path). The one-wave rule covers few rows and not many."""
    import triton
    from megatron.core.ssm.ops.mamba2 import batch_invariant_decode as bid
    from megatron.core.ssm.ops.mamba2 import ssd_chunk_scan

    from nemo_rl.models.megatron import mamba_bi_scan_cuda

    _pin()
    dev = torch.device("cuda")
    assert fd.supported_config(64, 64, 4, 128, 128, torch.float32, dev)
    kernel = ssd_chunk_scan._chunk_scan_fwd_kernel
    saved = kernel.configs
    try:
        kernel.configs = [triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}, num_warps=2)]
        assert not fd.supported_config(64, 64, 4, 128, 128, torch.float32, dev)  # K 64 does not divide M 32
        kernel.configs = saved + saved
        assert not fd.supported_config(64, 64, 4, 128, 128, torch.float32, dev)  # not pinned to one config
    finally:
        kernel.configs = saved
    assert not fd.supported_config(16, 40, 2, 128, 128, torch.float32, dev)  # head_dim 40
    assert not fd.supported_config(16, 64, 2, 96, 128, torch.float32, dev)  # d_state 96
    assert not fd.supported_config(64, 64, 4, 128, 128, torch.bfloat16, dev)  # 16-bit buffers keep z
    buf = bid.BatchInvariantDecodeBuffers.allocate(4, 128, 16, 40, 2, 128, dev, torch.float32)
    # a shape the step does not serve keeps MCore's path: z buffers before #8221, MCore main's own scan on main
    assert (not fd._serves(buf)) if _main_pin() else (buf.z is not None)
    scan = mamba_bi_scan_cuda.module(64, 128, 128, 32)
    assert fd.small_step(1, 64, dev, scan) and not fd.small_step(4096, 64, dev, scan)


# ---------------------------------------------------------------------------
# Learner parity: training forward (mamba_ssd_parity) == engine prefill + this decode.
# ---------------------------------------------------------------------------


@pytest.fixture
def grouped_mixer(tmp_path):
    """A MambaMixer with 16 heads in 2 groups, fp32."""
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.mamba_mixer import MambaMixer
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig

    created = False
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
        created = True
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        hidden_size=512,
        num_layers=1,
        num_attention_heads=1,
        use_cpu_initialization=True,
        mamba_num_heads=16,
        mamba_num_groups=2,
    )
    mixer = MambaMixer(
        config,
        hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
        config.hidden_size,
        layer_number=1,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"]),
    )
    config.batch_invariant_mode = True
    yield mixer.to(device="cuda", dtype=torch.float32)
    parallel_state.destroy_model_parallel()
    if created:
        dist.destroy_process_group()


@cuda
@pytest.mark.parametrize("small", [False, True])
def test_learner_training_forward_matches_decode(grouped_mixer, small):
    """The train-lp sentinel's equality at the mixer: training forward == prefill + decode, bitwise."""
    from tests.unit.models.megatron.test_mamba_ssd_parity import _inputs, _train_and_generate

    from nemo_rl.models.megatron import mamba_ssd_parity

    mamba_ssd_parity.install()
    _pin()
    zxBCdt = _inputs(grouped_mixer, torch.float32)
    with torch.no_grad(), _path(small):
        y_train, y_gen = _train_and_generate(grouped_mixer, zxBCdt)
        dec = grouped_mixer._get_batch_invariant_decoder()
        assert dec.buffers is not None and getattr(dec.buffers, "z", None) is None
    assert torch.equal(_bits(y_train), _bits(y_gen))
