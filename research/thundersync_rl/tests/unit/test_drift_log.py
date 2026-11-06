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
"""Drift logging (drift_log.py): per-age log p/q moments, optimizer-shard phase-in, Adam update stats.

CPU tests: the moment bucketing against direct numpy, phi_E at its two anchors (on the grid = 0, uniform residual = 1),
the probe end to end around a reference AdamW step on fake optimizer shards (values vs direct computation, the probe
is read-only: optimizer state and weights bitwise as without it), the worker wiring (the step is wrapped and
restored, metrics carry the probe's dict), the driver wiring (_null_result passes the moments through).
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from thundersync_rl import draft_weights, drift_log


# --------------------------------------------------------------------------- 1. per-age moments


def test_row_moments_segments_positions_and_nonfinite():
    x = torch.arange(10000, dtype=torch.float64) * 1e-4 - 0.3
    x[5] = -math.inf  # exact top-p: drafted token outside the nucleus
    segs = [(7, 0, 300), (8, 300, 10000)]
    seg, pos = drift_log.row_moments(x, segs)
    xs = x.numpy()
    a = xs[0:300][np.isfinite(xs[0:300])]
    b = xs[300:]
    assert seg[0][0] == 7 and seg[0][1] == a.size and seg[0][2] == pytest.approx(a.sum()) \
        and seg[0][3] == pytest.approx((a * a).sum())
    assert seg[1][0] == 8 and seg[1][1] == b.size and seg[1][2] == pytest.approx(b.sum())
    edges = (0, 512, 2048, 8192, 10000)
    assert len(pos) == 4
    for j in range(4):
        part = xs[edges[j]:edges[j + 1]]
        part = part[np.isfinite(part)]
        assert pos[j][0] == part.size and pos[j][2] == pytest.approx((part * part).sum())
    assert sum(p[0] for p in pos) == sum(s[1] for s in seg) == 9999


def test_age_moment_stats_buckets_by_age_and_matches_numpy():
    rng = np.random.default_rng(0)
    xs = {a: rng.normal(0.01 * a, 0.1 * (a + 1), 400 + 50 * a) for a in range(6)}
    current = 10
    rows = []
    for a, x in xs.items():  # one row per age, version = current - a; ages 4 and 5 -> 4+
        rows.append({"lr_seg_moments": [(current - a, x.size, float(x.sum()), float((x * x).sum()))],
                     "lr_pos_moments": [(x.size, float(x.sum()), float((x * x).sum())), (0, 0.0, 0.0),
                                        (0, 0.0, 0.0), (0, 0.0, 0.0)]})
    out = drift_log.age_moment_stats(rows, current)
    for a in range(4):
        assert out[f"spec/tokens_age{a}"] == xs[a].size
        assert out[f"spec/logratio_var_age{a}"] == pytest.approx(xs[a].var(), rel=1e-9)
        assert out[f"spec/logratio_mean_age{a}"] == pytest.approx(xs[a].mean(), rel=1e-9)
    both = np.concatenate([xs[4], xs[5]])
    assert out["spec/tokens_age4+"] == both.size
    assert out["spec/logratio_var_age4+"] == pytest.approx(both.var(), rel=1e-9)
    allx = np.concatenate(list(xs.values()))
    assert out["spec/tokens_pos0"] == allx.size
    assert out["spec/logratio_var_pos0"] == pytest.approx(allx.var(), rel=1e-9)
    assert "spec/logratio_var_pos1" not in out  # empty bin: count only
    # A future version (age < 0 on a stale bookkeeping) clamps to age 0, as spec/age*_token_frac.
    assert "spec/tokens_age0" in drift_log.age_moment_stats(
        [{"lr_seg_moments": [(current + 1, 3, 0.0, 1.0)]}], current)


def test_null_result_passes_moments_through():
    from thundersync_rl.speculative import _null_result

    res = {"null_lr_moments": (3, 0.1, 0.2), "lr_seg_moments": [(4, 3, 0.1, 0.2)],
           "lr_pos_moments": [(3, 0.1, 0.2)] + [(0, 0.0, 0.0)] * 3, "null_grid": {}}
    out = _null_result(res, None)
    assert out["lr_seg_moments"] == res["lr_seg_moments"]
    assert out["lr_pos_moments"] == res["lr_pos_moments"]


# --------------------------------------------------------------------------- 2. phase-in


def _phi(master, dtype=torch.bfloat16):
    acc = torch.zeros(2, drift_log.N_EBINS, dtype=torch.float64)
    drift_log.phase_accumulate(master, master.to(dtype), acc)
    return acc


def test_phase_zero_on_grid_and_one_for_uniform_residual():
    g = torch.Generator().manual_seed(0)
    on_grid = (torch.randn(200000, generator=g) * 0.02).to(torch.bfloat16).float()
    # Powers of two have a half-size cell below them (the next binade's ulp): leave them out of the uniform case.
    on_grid = on_grid[torch.frexp(on_grid).mantissa.abs() != 0.5]
    acc = _phi(on_grid)
    assert float(acc[1].sum()) == 0.0 and float(acc[0].sum()) == on_grid.numel()
    # Uniform sub-ulp residual: master = grid value + U(-q/2, q/2) q -> 4 mean|r|/q -> 1.
    e = torch.frexp(on_grid).exponent
    q = drift_log.ulp(e, torch.bfloat16)
    u = torch.rand(on_grid.shape, generator=g) - 0.5
    master = on_grid + u * q * 0.999  # stay inside the cell (rounds back to the grid value)
    assert torch.equal(master.to(torch.bfloat16).float(), on_grid)
    acc = _phi(master)
    phi = 4 * acc[1] / acc[0].clamp_min(1)
    big = acc[0] > 5000
    assert big.sum() >= 3
    assert torch.allclose(phi[big], torch.full_like(phi[big], 0.999), atol=0.03)


def test_phase_bins_follow_the_exponent():
    vals = torch.tensor([0.75, 0.3, 0.02, 1e-6, 3.0, 0.0])  # E = 0, -1, -5, -19 (clamped), 2 (clamped), zero
    acc = _phi(vals)
    lo, hi = drift_log.E_LO, drift_log.E_HI
    expect = {0 - lo: 1, -1 - lo: 1, -5 - lo: 1, 0: 1, hi - lo: 1}
    got = {i: int(c) for i, c in enumerate(acc[0].tolist()) if c}
    assert got == expect  # the exact zero is skipped
    assert drift_log.ulp(torch.tensor([1]), torch.bfloat16).item() == 2.0 ** -7  # bf16 ulp at 1.0
    assert drift_log.ulp(torch.tensor([1]), torch.float16).item() == 2.0 ** -10


# --------------------------------------------------------------------------- 3. probe end to end


B1, B2, EPS, WD, LR = 0.9, 0.999, 1e-8, 0.01, 1e-3


class _FakeInner:
    def __init__(self):
        self.state = {}


class _FakeOpt:
    """One DistributedOptimizer-like layout: fp32 main shards + bf16 buffer slices; AdamW step in torch."""

    def __init__(self, n_shards=3, n=40000, seed=0, clip=1.0):
        g = torch.Generator().manual_seed(seed)
        self.inner = _FakeInner()
        self.group = {"lr": LR, "betas": (B1, B2), "eps": EPS, "weight_decay": WD, "step": 4}
        self.config = SimpleNamespace(clip_grad=clip)
        self.mains, self.targets = [], []
        for i in range(n_shards):
            w = torch.randn(n, generator=g) * (0.05 * (i + 1))
            self.mains.append(w)
            self.targets.append(w.to(torch.bfloat16))
            self.inner.state[w] = {"exp_avg": torch.randn(n, generator=g) * 1e-3,
                                   "exp_avg_sq": torch.rand(n, generator=g) * 1e-5 + 1e-8}
        self.grads = [torch.randn(n, generator=g) * 3e-3 for _ in range(n_shards)]

    def shards(self):
        for w, t in zip(self.mains, self.targets):
            yield draft_weights._Shard(inner=self.inner, group=self.group, key=w, target=t, aliased=False, main=w)

    def step(self):
        t = self.group["step"] + 1
        for w, tgt, gr in zip(self.mains, self.targets, self.grads):
            st = self.inner.state[w]
            st["exp_avg"].mul_(B1).add_(gr, alpha=1 - B1)
            st["exp_avg_sq"].mul_(B2).addcmul_(gr, gr, value=1 - B2)
            upd = (st["exp_avg"] / (1 - B1**t)) / ((st["exp_avg_sq"] / (1 - B2**t)).sqrt() + EPS)
            w.sub_(LR * (upd + WD * w))
            tgt.copy_(w.to(torch.bfloat16))
        self.group["step"] = t
        return True, 2.5, 0


@pytest.fixture
def fake_shards(monkeypatch):
    holder = {}
    monkeypatch.setattr(draft_weights, "_shards", lambda opt: opt.shards())
    return holder


def test_probe_values_and_read_only(fake_shards, monkeypatch):
    monkeypatch.setenv("THUNDERSYNC_DRIFT_STRIDE", "1")
    monkeypatch.setenv("THUNDERSYNC_DRIFT_PHI_STRIDE", "1")
    ref, opt = _FakeOpt(), _FakeOpt()
    w0 = [w.clone() for w in opt.mains]
    m0 = [opt.inner.state[w]["exp_avg"].clone() for w in opt.mains]
    v0 = [opt.inner.state[w]["exp_avg_sq"].clone() for w in opt.mains]
    probe = drift_log.DriftProbe()
    probe.pre(opt)
    res = opt.step()
    out = probe.post(opt, res)
    ref.step()
    # Read-only: weights, stored values and moments bitwise as the same step without the probe.
    for a, b in zip(opt.mains, ref.mains):
        assert torch.equal(a, b)
    for a, b in zip(opt.targets, ref.targets):
        assert torch.equal(a, b)
    for a, b in zip(opt.mains, ref.mains):
        assert torch.equal(opt.inner.state[a]["exp_avg"], ref.inner.state[b]["exp_avg"])
        assert torch.equal(opt.inner.state[a]["exp_avg_sq"], ref.inner.state[b]["exp_avg_sq"])
    # Values vs direct computation (float64).
    u = torch.cat([(w1.double() - w.double()) / LR for w, w1 in zip(w0, opt.mains)])
    mom = torch.cat([
        (draft_weights.momentum_draft_master(w, m, v, lr=LR, beta1=B1, beta2=B2, eps=EPS, weight_decay=WD, step=5)
         .double() - w.double()) / LR for w, m, v in zip(w0, m0, v0)])
    s0 = torch.cat([w.to(torch.bfloat16) for w in w0])
    s1 = torch.cat(opt.targets)
    assert out["drift/sampled_elements"] == u.numel()
    assert out["drift/rms_u"] == pytest.approx(u.pow(2).mean().sqrt().item(), rel=1e-9)
    assert out["drift/mean_abs_u"] == pytest.approx(u.abs().mean().item(), rel=1e-9)
    assert out["drift/rms_resid_mom"] == pytest.approx((u - mom).pow(2).mean().sqrt().item(), rel=1e-9)
    chg = (s1 != s0).double()
    assert out["drift/changed_frac"] == pytest.approx(chg.mean().item(), rel=1e-12)
    assert 0 < out["drift/changed_frac"] < 1
    assert out["drift/grad_norm_preclip"] == 2.5
    assert out["drift/clip_factor"] == pytest.approx(1.0 / (2.5 + 1e-6))
    assert out["drift/lr"] == LR and out["drift/update_ok"] == 1.0
    # phi over the shard after the step (masters vs their stored values).
    acc = torch.zeros(2, drift_log.N_EBINS, dtype=torch.float64)
    for w, t in zip(opt.mains, opt.targets):
        drift_log.phase_accumulate(w, t, acc)
    assert out["drift/phi"] == pytest.approx(4 * acc[1].sum().item() / acc[0].sum().item(), rel=1e-12)
    assert out["drift/elements"] == acc[0].sum().item()
    # Per-bin changed fractions add up to the total.
    tot = sum(out[k] * out[k.replace("changed_frac_", "n_")] for k in out if k.startswith("drift/changed_frac_E")
              and k.replace("changed_frac_", "n_") in out)
    assert tot > 0
    assert any(k.startswith("drift/phi_E") for k in out)


def test_probe_strided_sample_is_close_and_small(fake_shards, monkeypatch):
    monkeypatch.setenv("THUNDERSYNC_DRIFT_STRIDE", "16")
    full_opt, opt = _FakeOpt(n=160000), _FakeOpt(n=160000)
    p_full = drift_log.DriftProbe()
    monkeypatch.setenv("THUNDERSYNC_DRIFT_STRIDE", "1")
    p_full.pre(full_opt)
    full = p_full.post(full_opt, full_opt.step())
    monkeypatch.setenv("THUNDERSYNC_DRIFT_STRIDE", "16")
    p = drift_log.DriftProbe()
    p.pre(opt)
    assert sum(s[1][0].numel() for s in p._sample) == 3 * 10000  # 1/16 of the shards
    out = p.post(opt, opt.step())
    assert out["drift/sampled_elements"] == 3 * 10000
    assert out["drift/rms_u"] == pytest.approx(full["drift/rms_u"], rel=0.03)
    assert out["drift/changed_frac"] == pytest.approx(full["drift/changed_frac"], rel=0.1)
    assert out["drift/phi"] == full["drift/phi"]  # same phase sample (THUNDERSYNC_DRIFT_PHI_STRIDE) either way
    # The default phase sample (every 8th element) estimates the whole-shard phi closely.
    monkeypatch.setenv("THUNDERSYNC_DRIFT_PHI_STRIDE", "1")
    w_opt = _FakeOpt(n=160000)
    pw = drift_log.DriftProbe()
    pw.pre(w_opt)
    whole = pw.post(w_opt, w_opt.step())
    assert whole["drift/elements"] == 8 * full["drift/elements"]
    assert full["drift/phi"] == pytest.approx(whole["drift/phi"], rel=0.02)


def test_finalize_without_grad_norm_or_clip():
    z = torch.zeros(2, drift_log.N_EBINS, dtype=torch.float64)
    out = drift_log.finalize(z, z, torch.zeros(len(drift_log._U), dtype=torch.float64), grad_norm=None,
                             clip_grad=1.0, lr=1e-6, ok=True)
    assert "drift/clip_factor" not in out and out["drift/lr"] == 1e-6
    out = drift_log.finalize(z, z, torch.zeros(len(drift_log._U), dtype=torch.float64), grad_norm=0.05,
                             clip_grad=1.0, lr=1e-6, ok=True)
    assert out["drift/clip_factor"] == 1.0  # never clipped below the threshold


# --------------------------------------------------------------------------- 4. worker wiring


def test_worker_wraps_and_restores_the_optimizer_step(monkeypatch):
    from thundersync_rl import worker as worker_mod

    cls = worker_mod.ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class
    w = object.__new__(cls)
    w.rank = 0
    calls = []

    def step():
        calls.append("step")
        return True, 1.5, 0

    w.optimizer = SimpleNamespace(step=step)
    probe = SimpleNamespace(last={"drift/rms_u": 0.3})
    probe.pre = lambda opt: calls.append("pre")

    def post(opt, result):
        calls.append(("post", result))
        probe.last = {"drift/rms_u": 0.3, "drift/grad_norm_preclip": float(result[1])}

    probe.post = post
    w._drift_probe = probe
    monkeypatch.setattr(worker_mod, "snapshot_anchor_weights", lambda opt, kind: None)
    monkeypatch.setattr(cls, "_finish_train_step_maybe_noisy", lambda self: {"grad_norm": self.optimizer.step()[1]})
    metrics = w.finish_train_step()
    assert calls == ["pre", "step", ("post", (True, 1.5, 0))]
    assert w.optimizer.step is step  # restored
    assert metrics["drift"] == {"drift/rms_u": 0.3, "drift/grad_norm_preclip": 1.5}
    # Off: THUNDERSYNC_DRIFT_LOG=0 -> no probe, no "drift" key.
    monkeypatch.setenv("THUNDERSYNC_DRIFT_LOG", "0")
    calls.clear()
    metrics = w.finish_train_step()
    assert calls == ["step"] and "drift" not in metrics


# --------------------------------------------------------------------------- 5. run rule


def test_run_rule_drift_present():
    import importlib.util
    import os

    path = os.environ.get("TS_RUN_RULES") or os.path.join(
        os.path.dirname(__file__), *[".."] * 4, "..", "thundersync_work", "research", "ab", "run_rules.py")
    if not os.path.exists(path):
        pytest.skip("run_rules.py (thundersync_work) not next to this checkout")
    spec = importlib.util.spec_from_file_location("run_rules", path)
    rr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rr)
    good = {"drift/rms_u": 0.2, "drift/rms_resid_mom": 0.1, "drift/changed_frac": 0.01, "drift/phi": 0.3,
            "drift/grad_norm_preclip": 0.05, "drift/clip_factor": 1.0, "drift/lr": 1e-6,
            "spec/logratio_var_step": 1e-6, "spec/logratio_var_age1": 1e-6}
    r = {"hist": [dict(good), dict(good)], "log": ""}
    assert rr._metric(r, "drift.present") == 1.0
    bad = dict(good)
    del bad["spec/logratio_var_age1"]
    assert rr._metric({"hist": [good, bad], "log": ""}, "drift.present") == 0.0
    bad = dict(good)
    del bad["drift/phi"]
    assert rr._metric({"hist": [bad, good], "log": ""}, "drift.present") == 0.0


def test_probe_batches_shards_into_few_accumulator_calls(fake_shards, monkeypatch):
    # Thousands of small optimizer shards: the accumulators (each ends in a host-synchronizing bincount on CUDA)
    # run once per concatenated chunk, not once per shard; the metrics equal the per-shard reference.
    monkeypatch.setenv("THUNDERSYNC_DRIFT_STRIDE", "4")
    monkeypatch.setenv("THUNDERSYNC_DRIFT_PHI_STRIDE", "2")
    opt = _FakeOpt(n_shards=200, n=512)
    calls = {"phase": 0, "update": 0}
    real_phase, real_update = drift_log.phase_accumulate, drift_log.update_accumulate

    def phase(*a, **k):
        calls["phase"] += 1
        return real_phase(*a, **k)

    def update(*a, **k):
        calls["update"] += 1
        return real_update(*a, **k)

    monkeypatch.setattr(drift_log, "phase_accumulate", phase)
    monkeypatch.setattr(drift_log, "update_accumulate", update)
    w0 = [w.clone() for w in opt.mains]
    m0 = [opt.inner.state[w]["exp_avg"].clone() for w in opt.mains]
    v0 = [opt.inner.state[w]["exp_avg_sq"].clone() for w in opt.mains]
    probe = drift_log.DriftProbe()
    probe.pre(opt)
    out = probe.post(opt, opt.step())
    assert calls == {"phase": 1, "update": 1}, calls  # per-shard code: 200 + 200
    # Per-shard reference of the same sums.
    phase_ref = torch.zeros(2, drift_log.N_EBINS, dtype=torch.float64)
    chg_ref = torch.zeros(2, drift_log.N_EBINS, dtype=torch.float64)
    u_ref = torch.zeros(len(drift_log._U), dtype=torch.float64)
    for w, t, a, m, v in zip(opt.mains, opt.targets, w0, m0, v0):
        real_phase(w[::2], t[::2], phase_ref)
        sl = slice(0, w.numel(), 4)
        wm = draft_weights.momentum_draft_master(a[sl], m[sl], v[sl], lr=LR, beta1=B1, beta2=B2, eps=EPS,
                                                 weight_decay=WD, step=5)
        real_update(a[sl], w[sl], wm, LR, torch.bfloat16, u_ref, chg_ref)
    ref = drift_log.finalize(phase_ref, chg_ref, u_ref, grad_norm=2.5, clip_grad=1.0, lr=LR, ok=True)
    for key, val in ref.items():
        if key.startswith(("drift/n_", "drift/elements", "drift/sampled")):
            assert out[key] == val, key  # counts: exact
        else:
            assert out[key] == pytest.approx(val, rel=1e-12, abs=1e-300), key
