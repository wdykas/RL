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
"""fp16 routed expert parameters (every MoE model under ``forward_gemm_precision: fp16``) and the te_gt learner forward.

Bits, not values: every comparison is on the integer view of the outputs.
GPU tests need SM100 + cuBLASLt 13.5.1 (fp16_moe.te_gt_supported).
"""

import inspect
import math

import pytest
import torch

from nemo_rl.models.megatron import fp16_moe as fm


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32 if t.element_size() == 4 else torch.int16)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _supported() -> bool:
    return torch.cuda.is_available() and fm.te_gt_supported()


needs_te_gt = pytest.mark.skipif(not _supported(), reason="needs SM100 + cuBLASLt 13.5.1 + TE")


def _master(e, n, k, dev, seed=0, special=False):
    """fp32 'master' expert weights; ``special``: values in fp16's subnormal range and exact zeros."""
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(e, n, k, device=dev, generator=g) * k**-0.5
    if special:
        w[:, ::7, :] *= 2.0**-17
        w[:, 1::11, ::5] = 0.0
    return w


def _params16(w32):
    """fp16(master) as E separately allocated tensors (parameters live at arbitrary addresses)."""
    return [w.to(torch.float16).clone() for w in w32.unbind(0)]


def _counts(e, m, seed, zero_frac=0.2):
    g = torch.Generator().manual_seed(seed)
    p = torch.distributions.Dirichlet(torch.full((e,), 2.0)).sample()
    keep = torch.rand(e, generator=g) >= zero_frac
    p = p * keep
    p = p / p.sum()
    c = (p * m).floor().long()
    c[int(torch.argmax(p))] += m - int(c.sum())
    return c


def _engine_ref(x16, w16, counts):
    """The engine kernel grouped_gemm_fp16w on BLOCK_M-padded segments, gathered back to x16's rows."""
    dev = x16.device
    m, k = x16.shape
    e = w16.shape[0]
    cfg = fm.select_engine_config(max(m, 1) / e)
    bm = cfg[0]
    cl = counts.tolist()
    pad = [(-(-c // bm)) * bm for c in cl]
    rows = max(bm, sum(pad))
    xp = torch.zeros(rows, k, dtype=torch.float16, device=dev)
    src = []
    d = 0
    for c, p in zip(cl, pad):
        src += list(range(d, d + c))
        d += p
    src = torch.tensor(src, dtype=torch.long, device=dev)
    xp[src] = x16[: len(src)]
    offs = torch.tensor(pad, dtype=torch.int64).cumsum(0).to(dev, torch.int32)
    return fm.grouped_gemm_fp16w(xp, w16, offs, cfg, rows)[src]


# Recorded shapes (MAIN_PRS_PERF §2, FEEDBACK_1031): (local experts, K, N, rows).
SHAPES = [
    ("lrn_super_ep8_fc1", 64, 1024, 2688, 64 * 256),
    ("lrn_super_ep8_fc2", 64, 2688, 1024, 64 * 256),
    ("lrn_ultra_ep32_fc1", 16, 2048, 5120, 16 * 1024),
    ("lrn_ultra_ep32_fc2", 16, 5120, 2048, 16 * 1024),
    ("lrn_ultra_ep64_fc2", 8, 5120, 2048, 8 * 1024),
    ("lrn_nano_ep4_fc1", 32, 2688, 1856, 32 * 256),
    ("lrn_super_l8_ep2_fc1", 256, 1024, 2688, 256 * 64),
    ("eng_super_ep4_fc2", 128, 2688, 1024, 2400),
    ("eng_super_ep8_fc1", 64, 1024, 2688, 1200),
    ("decode_1row", 128, 1024, 2688, 8),
]


@needs_te_gt
@pytest.mark.parametrize("name,e,k,n,m", SHAPES, ids=[s[0] for s in SHAPES])
def test_te_gt_on_fp16_params_bitwise_engine_and_row_invariant(name, e, k, n, m):
    """te_gt on the fp16 parameters (separate tensors) == the engine kernel on the same fp16 weights."""
    dev = torch.device("cuda")
    w32 = _master(e, n, k, dev, seed=e + k, special=True)
    ws = _params16(w32)
    w16 = torch.stack(ws)  # the engine's [E, N, K] buffer holds the same fp16 values
    counts = _counts(e, m, seed=m + e) if m >= e else torch.zeros(e, dtype=torch.long).index_fill_(
        0, torch.randperm(e, generator=torch.Generator().manual_seed(1))[:m], 1
    )
    x16 = torch.randn(m, k, device=dev).to(torch.float16)
    x16[:, ::13] = 0.0
    cnt = counts.to(dev, torch.int32)
    y = fm.te_gt_forward(x16, ws, cnt)
    assert _same(y, _engine_ref(x16, w16, counts)), name
    if m >= 2 * e:  # rows are batch independent: half of each expert's rows as their own batch
        half = counts // 2
        starts = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]])
        idx = torch.cat([torch.arange(s, s + h) for s, h in zip(starts.tolist(), half.tolist())]).to(dev)
        ys = fm.te_gt_forward(x16[idx].contiguous(), ws, half.to(dev, torch.int32))
        assert _same(ys, y[idx]), name


@needs_te_gt
@pytest.mark.parametrize("e", [64, 128, 200, 256])
def test_te_gt_capacity_tail_and_chunk_edges(e):
    """Rows past sum(counts) (a capacity buffer's tail, here NaN) are never read; chunk edges at 64 experts."""
    dev = torch.device("cuda")
    k, n = 256, 384
    ws = _params16(_master(e, n, k, dev, seed=3))
    counts = torch.zeros(e, dtype=torch.long)
    counts[0] = 5
    counts[63] = 17
    counts[64 % e] += 9
    counts[e - 1] += 33
    counts[e // 2] += 1
    live = int(counts.sum())
    x = torch.randn(live + 77, k, device=dev).to(torch.float16)
    x[live:] = float("nan")
    y = fm.te_gt_forward(x, ws, counts.to(dev, torch.int32))
    assert _same(y[:live], _engine_ref(x[:live].contiguous(), torch.stack(ws), counts))
    assert not torch.isnan(y[:live]).any()
    fm.te_gt_forward(x, ws, torch.zeros(e, dtype=torch.int32, device=dev))


@needs_te_gt
@pytest.mark.parametrize("e", [64, 256])
def test_te_gt_cuda_graph_replay_with_new_device_counts(e):
    dev = torch.device("cuda")
    k, n, cap = 512, 768, 4096
    ws = _params16(_master(e, n, k, dev, seed=5))
    w16 = torch.stack(ws)
    x = torch.zeros(cap, k, dtype=torch.float16, device=dev)
    cnt = torch.zeros(e, dtype=torch.int32, device=dev)
    cnt.copy_(_counts(e, 3000, seed=11).to(torch.int32))
    x.copy_(torch.randn(cap, k, device=dev).to(torch.float16))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fm.te_gt_forward(x, ws, cnt)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = fm.te_gt_forward(x, ws, cnt)
    for seed, m in ((12, 3000), (13, 4096), (14, 517)):
        c = _counts(e, m, seed=seed)
        cnt.copy_(c.to(torch.int32))
        x.copy_(torch.randn(cap, k, device=dev).to(torch.float16))
        g.replay()
        torch.cuda.synchronize()
        assert _same(y[:m], _engine_ref(x[:m].contiguous(), w16, c)), (seed, m)


def _ddp_leaf(w: torch.Tensor) -> torch.Tensor:
    """A leaf weight with Megatron DDP's attributes (fp32 main_grad): fused_learner_ops' accumulate target."""
    p = torch.nn.Parameter(w.detach().clone())
    p.main_grad = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
    p.grad_added_to_main_grad = False
    return p


@needs_te_gt
@pytest.mark.parametrize("device_counts", [True, False])
def test_grouped_linear_forward_dgrad_wgrad_bits(device_counts):
    """GroupedFp16Linear on fp16 expert params (device and host splits): forward == engine kernel; dx ==
    per-expert cuBLAS g @ bf16(W) cast to x's dtype; main_grad == float(bf16(g^T @ bf16(x16))), bit for bit."""
    from nemo_rl.models.megatron import fused_learner_ops as flo

    assert flo.enabled()
    dev = torch.device("cuda")
    e, k, n = 64, 256, 512
    ws = [_ddp_leaf(w) for w in _params16(_master(e, n, k, dev, seed=7))]
    counts = _counts(e, 3000, seed=8)
    live = int(counts.sum())
    x = torch.randn(3100, k, device=dev)
    gy = torch.randn(live, n, device=dev) * 1e-4
    xg = x.clone().requires_grad_()
    m_splits = counts.to(dev) if device_counts else counts.tolist()
    y = fm.grouped_linear(xg if device_counts else xg[:live], m_splits, ws)
    (y[:live] * gy).sum().backward()
    x16 = x[:live].to(torch.float16)
    assert _same(y[:live].detach(), _engine_ref(x16, torch.stack([w.detach() for w in ws]), counts))
    g = gy.to(torch.bfloat16)
    s0 = 0
    for w, c in zip(ws, counts.tolist()):
        if c:
            dx = torch.mm(g[s0 : s0 + c], w.detach().to(torch.bfloat16), out_dtype=torch.float32)
            assert _same(xg.grad[s0 : s0 + c], dx)
        wg = torch.mm(g[s0 : s0 + c].t(), x16[s0 : s0 + c].to(torch.bfloat16), out_dtype=torch.float32)
        assert _same(w.main_grad, wg.to(torch.bfloat16).float())
        assert w.grad_added_to_main_grad
        s0 += c
    assert sum(float(w.main_grad.abs().sum()) for w in ws) > 0  # non-vacuous


@needs_te_gt
def test_fp16_params_refuse_autograd_rounding_of_wgrad(monkeypatch):
    """Without the fused main_grad accumulate the fp16 path fails loudly (an fp16 .grad would underflow)."""
    from nemo_rl.models.megatron import fused_learner_ops as flo

    dev = torch.device("cuda")
    ws = [_ddp_leaf(w) for w in _params16(_master(8, 128, 256, dev, seed=4))]
    counts = _counts(8, 500, seed=4).to(dev)
    monkeypatch.setitem(flo.ENABLED, "on", False)
    y = fm.grouped_linear(torch.randn(500, 256, device=dev, requires_grad=True), counts, ws)
    with pytest.raises(RuntimeError, match="fused_learner_ops"):
        y.sum().backward()


# ---------------------------------------------------------------------------
# Non-finite guard (after every optimizer step).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dev", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_guard_reports_max_abs_and_fails_on_injected_inf_and_nan(dev):
    ws = [torch.randn(16, 32, device=dev).to(torch.float16) for _ in range(5)]
    ws[3][2, 7] = -1234.0
    assert fm.check_fp16_expert_params(ws) == 1234.0
    for bad in (float("inf"), float("-inf"), float("nan")):
        w = [t.clone() for t in ws]
        w[1][0, 0] = bad
        with pytest.raises(FloatingPointError, match="non-finite"):
            fm.check_fp16_expert_params(w)


def test_guard_runs_after_every_optimizer_step():
    """guard_after_optimizer_step(opt, params): a step that overflows a master weight past 65504 fails."""
    w32 = torch.randn(8, 8)
    p16 = w32.to(torch.float16)

    class Opt:
        def step(self):  # Megatron: master update, then model param = fp16(master) (copy_)
            p16.copy_(w32)
            return True, 0.0, 0

    opt = Opt()
    fm.guard_after_optimizer_step(opt, [p16])
    opt.step()
    assert math.isfinite(fm.fp16_expert_stats()["max_abs"])
    w32[0, 0] = 70000.0  # beyond fp16's range: fp16 rounds to inf
    with pytest.raises(FloatingPointError):
        opt.step()


def test_worker_installs_the_guard_with_the_fp16_expert_params():
    from nemo_rl.models.policy.workers import megatron_policy_worker as mpw

    src = inspect.getsource(mpw)
    assert "fp16_moe.guard_after_optimizer_step(self.optimizer, fp16_params)" in src
    assert "fp16_params = fp16_moe.fp16_expert_params(self.model)" in src


# ---------------------------------------------------------------------------
# Storage: only the routed experts become fp16, after the mixed-precision wrapper's bf16 cast.
# ---------------------------------------------------------------------------


def _fake_moe_model(e=4):
    from megatron.core.transformer.moe.experts import TEGroupedMLP

    class Linear(torch.nn.Module):
        def __init__(self, n, k):
            super().__init__()
            self.num_gemms = e
            for i in range(e):
                self.register_parameter(f"weight{i}", torch.nn.Parameter(torch.randn(n, k) * 1e-2))

    experts = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(experts)
    experts.linear_fc1 = Linear(12, 8)
    experts.linear_fc2 = Linear(8, 12)
    m = torch.nn.Module()
    m.router = torch.nn.Linear(8, e, bias=False)
    m.shared = torch.nn.Linear(8, 8, bias=False)  # a shared expert / dense linear: not routed
    m.experts = experts
    return m


def test_store_routed_experts_fp16_after_the_wrapper_cast():
    pytest.importorskip("megatron.core.transformer.moe.experts")
    model = _fake_moe_model()

    def bf16_wrapper(config, module):  # Float16Module: the whole module to bf16
        return module.bfloat16()

    ref = {n: p.detach().clone() for n, p in model.named_parameters()}
    wrapped = fm.fp16_routed_experts_wrapper(bf16_wrapper)(None, model)
    routed = fm.fp16_expert_params(wrapped)
    assert len(routed) == 8
    for name, p in wrapped.named_parameters():
        if ".linear_fc" in name:
            assert p.dtype == torch.float16 and getattr(p, fm.FP16_EXPERT_ATTR)
            # fp16(bf16(initial)): exact for every normal-range value
            assert _same(p.data, ref[name].to(torch.bfloat16).to(torch.float16))
        else:  # router, shared / dense linears: unchanged (bf16)
            assert p.dtype == torch.bfloat16 and not getattr(p, fm.FP16_EXPERT_ATTR, False)


def test_setup_stores_fp16_experts_under_the_fp16_recipe():
    from nemo_rl.models.megatron import setup

    src = inspect.getsource(setup.setup_model_and_optimizer)
    # unconditional under the fp16 GEMM recipe; fp32 grad buffers required for the fp16 experts
    assert 'policy_cfg["megatron_cfg"].get("forward_gemm_precision") == "fp16"' in src
    assert "fp16_moe.fp16_routed_experts_wrapper(" in src
    assert "not megatron_cfg.ddp.grad_reduce_in_fp32" in src
    assert "fp16_expert_params\")" not in src


# ---------------------------------------------------------------------------
# Refit: fp16 on the wire, received directly into the engine's fp16 expert parameter.
# ---------------------------------------------------------------------------


class _LocalService:
    supports_multiple_runs_per_plan = True
    requires_process_group_barrier = False

    def __init__(self):
        self.sends, self.recvs = {}, {}

    def set_plan(self, plan, *, transform=None):
        pass

    def set_model_roles(self, **kwargs):
        pass

    def execute_plan(self, *args, **kwargs):
        return False

    def submit_send(self, tensor, dest_rank, task_id=None):
        self.sends[task_id] = tensor.clone()

    def submit_recv(self, tensor, src_rank, task_id=None):
        self.recvs[task_id] = tensor

    def run(self):
        for task_id, buf in self.recvs.items():
            buf.view(torch.uint8).copy_(self.sends.pop(task_id).view(torch.uint8))  # raw bytes, as NCCL
        self.recvs.clear()


def _reshard_once(dst_tensor, src_tensor, transform):
    from megatron.core.resharding.execution import _execute_batch
    from megatron.core.resharding.utils import TransferOp

    full = (slice(None),)
    send = TransferOp(param_name="w", peer_rank=1, is_send=True, my_slice=full, peer_slice=full, task_id=0)
    recv = TransferOp(param_name="w", peer_rank=0, is_send=False, my_slice=full, peer_slice=full, task_id=0)
    service = _LocalService()
    _execute_batch([send], [], {"w": src_tensor}, {}, service, None, None, None)
    _execute_batch([], [recv], {}, {"w": dst_tensor}, service, None, transform, None)


def test_refit_carries_fp16_expert_params_bitwise():
    """Learner fp16 expert parameter -> engine fp16 expert parameter: fp16 on the wire, direct receive, same bits.
    Unmarked fp16 engine tensors (dense weights held in fp16, bf16 wire) still take Fp16HeldWeightReceive."""
    pytest.importorskip("megatron.core.resharding.execution")
    from nemo_rl.models.generation.megatron.megatron_worker import Fp16HeldWeightReceive

    g = torch.Generator().manual_seed(1)
    master = torch.randn(4096, generator=g) * 0.02
    master[:6] = torch.tensor([1e-6, -3e-7, 6e-8, 2e-5, 6.5e4, -1e-3])
    src = torch.nn.Parameter(master.to(torch.float16))  # learner: fp16(master)
    setattr(src, fm.FP16_EXPERT_ATTR, True)

    engine = torch.nn.Module()
    engine.w = torch.nn.Parameter(torch.full(master.shape, float("nan"), dtype=torch.float16))
    setattr(engine.w, fm.FP16_EXPERT_ATTR, True)
    engine.dense = torch.nn.Parameter(torch.zeros(4, dtype=torch.float16))  # held fp16, bf16 wire
    transform = Fp16HeldWeightReceive(engine)
    assert not transform.should_transform("w") and transform.should_transform("dense")
    _reshard_once(engine.w, src, transform)
    assert _same(engine.w.data, src.data)

    # Non-vacuous: the bf16-wire receive applied to fp16 bytes would corrupt them.
    class Always(Fp16HeldWeightReceive):
        def should_transform(self, name):
            return True

    bad = torch.nn.Module()
    bad.w = torch.nn.Parameter(torch.zeros(master.shape, dtype=torch.float16))
    _reshard_once(bad.w, src, Always(bad))
    assert not _same(bad.w.data, src.data)


DK_SHAPES = [  # engine decode / prefill shapes: (name, local experts, K, N, rows)
    ("eng_super_ep8_fc1_decode", 64, 1024, 2688, 64),
    ("eng_super_ep8_fc2_decode", 64, 2688, 1024, 64),
    ("eng_super_ep8_fc1", 64, 1024, 2688, 1760),
    ("eng_super_ep8_fc2", 64, 2688, 1024, 1760),
    ("eng_ultra_ep32_fc1", 16, 2048, 5120, 900),
    ("eng_ultra_ep32_fc2", 16, 5120, 2048, 900),
    ("eng_1row", 128, 1024, 2688, 1),
]


@needs_te_gt
@pytest.mark.parametrize("name,e,k,n,m", DK_SHAPES, ids=[s[0] for s in DK_SHAPES])
def test_dk1_engine_gemm_bitwise_te_gt_on_fp16_params(name, e, k, n, m):
    """dk1's engine expert GEMM (fp16_moe_dk.dk_grouped_gemm) on the engine's [E, N, K] fp16 buffer == the learner's
    te_gt on the same fp16 expert parameters, row for row, at both dk tile choices (fc1 / fc2 configs)."""
    from nemo_rl.models.megatron import fp16_moe_dk as dk

    dev = torch.device("cuda")
    ws = _params16(_master(e, n, k, dev, seed=e + n, special=True))
    w16 = torch.stack(ws)
    counts = _counts(e, m, seed=m + 3) if m >= e else torch.zeros(e, dtype=torch.long).index_fill_(
        0, torch.randperm(e, generator=torch.Generator().manual_seed(2))[:m], 1
    )
    x16 = torch.randn(m, k, device=dev).to(torch.float16)
    ref = fm.te_gt_forward(x16, ws, counts.to(dev, torch.int32))
    for cfg in dict.fromkeys(dk.select_dk_configs(max(m, 1) / e)):
        bm = cfg[0]
        cl = counts.tolist()
        pad = [(-(-c // bm)) * bm for c in cl]
        rows = max(bm, sum(pad))
        xp = torch.zeros(rows, k, dtype=torch.float16, device=dev)
        src, d = [], 0
        for c, p in zip(cl, pad):
            src += list(range(d, d + c))
            d += p
        src = torch.tensor(src, dtype=torch.long, device=dev)
        xp[src] = x16
        offs = torch.tensor(pad, dtype=torch.int64).cumsum(0).to(dev, torch.int32)
        y = dk.dk_grouped_gemm(xp, w16, offs, cfg, rows)
        assert _same(y[src], ref), (name, cfg)
