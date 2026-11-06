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
"""GPU: the exact integer sampler (Triton) == the reference, batch-invariant, and its
frequencies match w / W at huge N; engine-logged lambda == learner lambda bitwise."""

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nemo_rl.models.megatron import exact_int as xi  # noqa: E402
from nemo_rl.models.megatron import exact_nucleus  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _rows(n, V, seed=0, scale=6.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    z = torch.randn(n, V, device="cuda", generator=g) * scale
    z[:, :7] = torch.tensor([20.0, 5.0, 0.0, -10.0, -20.0, -30.0, -36.0], device="cuda")
    z[:, -3:] = float("-inf")
    return torch.log_softmax(z, -1)


def test_triton_weights_equal_torch_weights():
    lp = _rows(64, 131072)
    extra = torch.cat([-torch.rand(1 << 20, device="cuda") * 40, -torch.rand(1 << 16, device="cuda") * 1e-6,
                       torch.zeros(4, device="cuda"), torch.full((4,), float("-inf"), device="cuda")])
    for x in (lp, extra):
        tri = xi._int_weights_triton(x)
        ref = exact_nucleus.mass_weights(x)
        assert torch.equal(tri, ref), int((tri != ref).sum())
    cpu = exact_nucleus.mass_weights(extra.cpu())
    print(f"[exact_int gpu] cuda weights == cpu weights: {torch.equal(cpu, xi._int_weights_triton(extra).cpu())} "
          f"(differences {int((cpu != xi._int_weights_triton(extra).cpu()).sum())} of {extra.numel()})")
    W, wx = xi.row_stats(lp, torch.arange(64, device="cuda"))
    w = exact_nucleus.mass_weights(lp)
    assert torch.equal(W, w.sum(-1)) and torch.equal(wx, w[torch.arange(64), torch.arange(64)])


def test_words_device_equal_reference():
    a = torch.arange(-500, 500, device="cuda") * 7919
    b = torch.arange(1000, device="cuda") % 13
    k = torch.arange(1000, device="cuda") % 3
    salt = (1 << 63) + 12345
    got = xi.words_device(salt, a, b, xi.ETA, k).cpu().numpy().view(np.uint64)
    ref = xi.words_np(salt, a.cpu().numpy(), b.cpu().numpy(), xi.ETA, k.cpu().numpy())
    assert np.array_equal(got, ref)
    assert int(ref[0]) == xi.word_py(salt, int(a[0]), int(b[0]), xi.ETA, int(k[0]))


def test_sampler_equals_reference_and_batch_invariant():
    n, V = 96, 131072
    lp = _rows(n, V, seed=1, scale=3.0)
    a = torch.arange(n, device="cuda") * 1009 + 5
    b = torch.arange(n, device="cuda") + 4096
    salt = 0xDEADBEEF12345
    tok, W, wx = xi.sample_logprob_rows(lp, a, b, salt)
    w = exact_nucleus.mass_weights(lp)
    ref = xi.sample_weights(w, salt, a.cpu().numpy(), b.cpu().numpy(), xi.ENGINE)
    assert torch.equal(tok, ref)
    assert torch.equal(W, w.sum(-1)) and torch.equal(wx, w[torch.arange(n), tok])
    perm = torch.randperm(n, device="cuda")
    t2, _, _ = xi.sample_logprob_rows(lp[perm], a[perm], b[perm], salt)
    assert torch.equal(t2, tok[perm])
    t3, _, _ = xi.sample_logprob_rows(lp[5:17], a[5:17], b[5:17], salt)
    assert torch.equal(t3, tok[5:17])
    # strided rows (a view of a wider buffer)
    wide = torch.full((n, V + 64), float("-inf"), device="cuda")
    wide[:, :V] = lp
    t4, _, _ = xi.sample_logprob_rows(wide[:, :V], a, b, salt)
    assert torch.equal(t4, tok)


def test_engine_logged_lambda_equals_learner_lambda():
    """The engine's logged value (int_logprob_selected on the log_softmax rows and the
    selected l_x) == the learner chunk value (training_value) == lambda from the sampler's
    own (w_x, W), bitwise."""
    n, V = 64, 131072
    lp = _rows(n, V, seed=2)
    a = torch.arange(n, device="cuda")
    tok, W, wx = xi.sample_logprob_rows(lp, a, a * 0 + 7, 99)
    sel = lp[torch.arange(n), tok]
    engine = xi.int_logprob_selected(lp, sel).float()
    learner = xi.training_value(lp.view(4, 16, V), tok.view(4, 16)).view(-1)
    sampler = xi.lam(wx, W).float()
    assert torch.equal(engine, learner) and torch.equal(engine, sampler)


def _chi2_sf(stat, dof):
    from scipy.stats import chi2

    return float(chi2.sf(stat, dof))


def test_sampler_frequencies_huge_n():
    """2^31 keyed draws from one 16-token row whose integer law spans 1 .. 1e-10:
    chi-square over the bins with >= 5 expected, Poisson z for the deep tail."""
    V = 16
    p = torch.tensor([0.5, 0.25, 0.1, 0.05, 0.04, 0.03, 0.02, 0.005, 0.003, 0.001, 0.0005,
                      1e-4, 1e-6, 1e-8, 3e-9, 1e-10], dtype=torch.float64)
    lp = torch.log(p / p.sum()).float().cuda()
    w = exact_nucleus.mass_weights(lp.view(1, -1))[0].cpu()
    W = int(w.sum())
    law = w.double().numpy() / W
    chunk = 1 << 26
    total = 1 << 31
    counts = torch.zeros(V, dtype=torch.int64, device="cuda")
    rows = lp.view(1, V).expand(chunk, V)
    a = torch.arange(chunk, device="cuda")
    for c in range(total // chunk):
        tok, _, _ = xi.sample_logprob_rows(rows, a, torch.full_like(a, c), 2024)
        counts += torch.bincount(tok, minlength=V)
    obs = counts.cpu().double().numpy()
    exp = law * total
    big = exp >= 5
    stat = float(((obs[big] - exp[big]) ** 2 / exp[big]).sum())
    p_chi = _chi2_sf(stat, int(big.sum()) - 1)
    z_tail = [(float(obs[v]), float(exp[v]), (obs[v] - exp[v]) / math.sqrt(exp[v])) for v in range(11, V)]
    print(f"[exact_int gpu] N=2^31 chi2 p={p_chi:.3g} tail (obs, exp, z): "
          + ", ".join(f"({o:.0f}, {e:.1f}, {z:+.2f})" for o, e, z in z_tail))
    assert p_chi > 1e-4
    assert all(abs(z) < 5 for _, _, z in z_tail)


def test_pit_law_test_passes_exact_sampler_and_catches_wrong_temperature():
    """The in-loop law test (randomized PIT under the learner's p) on tokens drawn by the
    exact integer sampler: passes; the same rows sampled at T = 0.7 (logged T = 1): fails."""
    from thundersync_rl import law_check

    n, V = 200_000, 4096
    g = torch.Generator(device="cuda").manual_seed(5)
    z = torch.randn(n, V, device="cuda", generator=g) * 2.5
    lp = torch.log_softmax(z, -1)
    a = torch.arange(n, device="cuda")
    res = {}
    for name, rows in (("exact", lp), ("T0.7", torch.log_softmax(z / 0.7, -1))):
        tok, _, _ = xi.sample_logprob_rows(rows, a, a * 0 + 3, 77)
        u = torch.rand(n, dtype=torch.float64, device="cuda", generator=g)
        uu, tt = law_check.pit_rows(z, tok, u)[:2]
        res[name] = law_check.law_test(law_check.accumulate(uu, tt))["p"]
    print(f"[exact_int gpu] PIT law test p: exact {res['exact']:.3g}, T=0.7 sampled {res['T0.7']:.3g}")
    assert res["exact"] > 1e-4
    assert res["T0.7"] < 1e-12


def test_exact_bv_on_cuda_rows_matches_exact_law_and_cpu_outcomes():
    """exact_bv with CUDA rows (Triton weights): frequencies vs the Fraction law, and the
    same (tau, y) as on CPU rows for the same key whenever the integer weights agree."""
    from collections import Counter
    from fractions import Fraction

    from scipy.stats import chi2

    from thundersync_rl import exact_bv as xb

    rng = np.random.default_rng(3)
    g, Vc = 4, 6
    pw = rng.integers(1, 20, size=(g + 1, Vc)).astype(float)
    qw = rng.integers(1, 20, size=(g, Vc)).astype(float)
    P = torch.log(torch.tensor(pw / pw.sum(-1, keepdims=True))).float().cuda()
    Q = torch.log(torch.tensor(qw / qw.sum(-1, keepdims=True))).float().cuda()
    draft = torch.tensor([0, 2, 1, 3], device="cuda")
    for M in (1.0, 1.75):
        law = xb.output_law_fraction(
            xi.int_weights(P).tolist(), xi.int_weights(Q).tolist(), draft.tolist(),
            Fraction(*M.as_integer_ratio()),
        )
        n = 4000
        cnt = Counter()
        for k in range(n):
            tau, y, _, _ = xb.verify_exact(P, Q, draft, null_m=M, key=k * 7919 + 1)
            cnt[(tau, y)] += 1
        assert all(key in law for key in cnt)
        keys = list(law)
        e = np.array([float(law[x]) * n for x in keys])
        o = np.array([cnt.get(x, 0) for x in keys], dtype=float)
        big = e >= 5
        oo, ee = np.append(o[big], o[~big].sum()), np.append(e[big], e[~big].sum())
        keep = ee > 0
        p = float(chi2.sf((((oo - ee) ** 2) / np.where(keep, ee, 1))[keep].sum(), keep.sum() - 1))
        print(f"[exact_int gpu] exact_bv CUDA M={M} chi2 p={p:.3g}")
        assert p > 1e-4
    same_w = torch.equal(xi.int_weights(P).cpu(), exact_nucleus.mass_weights(P.cpu())) and torch.equal(
        xi.int_weights(Q).cpu(), exact_nucleus.mass_weights(Q.cpu())
    )
    if same_w:
        for k in range(100):
            a = xb.verify_exact(P, Q, draft, key=k)
            b = xb.verify_exact(P.cpu(), Q.cpu(), draft.cpu(), key=k)
            assert a[:2] == b[:2]


def test_residual_mass_triton_equals_torch():
    """exactint-final (10-20): the batched stop index's Triton residual pass == the torch formula."""
    P = _rows(48, 131072, seed=1)
    Q = torch.log_softmax(torch.log_softmax(P, -1) + 0.05 * torch.randn_like(P), -1)
    Wp, _ = xi.row_stats(P, None)
    b = torch.rand(48, device="cuda", dtype=torch.float64)
    b[:4] = 1.0
    Wq, r = xi._residual_triton(P, Q, Wp, b)
    wp, wq = exact_nucleus.mass_weights(P), exact_nucleus.mass_weights(Q)
    assert torch.equal(Wq, wq.sum(-1))
    ref = (b[:, None] * (wp.double() / Wp.double()[:, None]) - wq.double() / wq.sum(-1).double()[:, None]).clamp(min=0).sum(-1)
    assert float(((r - ref).abs() / (b + 1)).max()) <= (131072 + 4) * 2.0**-51


def test_exact_bv_grid_long_draft_time():
    """A full-Nano-sized draft (g 2048, V 131072, near on-policy q, envelopes 1 / 1.75 / 3): the
    batched stop index scans every position at M > 1 (stop at the root) in O(seconds), and its
    outcomes equal the per-position reference scan."""
    import time

    from thundersync_rl import exact_bv as xb

    g, V = 2048, 131072
    P = _rows(g + 1, V, seed=2, scale=2.0)
    Q = torch.log_softmax(P[:g] + 0.02 * torch.randn(g, V, device="cuda"), -1)
    draft = torch.multinomial(Q.exp(), 1, generator=torch.Generator(device="cuda").manual_seed(0))[:, 0]
    t0 = time.perf_counter()
    out, _, _ = xb.verify_exact_grid(P, Q, draft, None, None, (1.0, 1.75, 3.0), key=11)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"[exact_int gpu] exact_bv grid g={g} V={V}: {dt:.2f} s, outcomes {out}")
    for M, (tau, y, _) in out.items():
        v = xb._Verify(P, Q, draft, None, 11)
        assert v.stop_index_ref(M)[0] == tau
    assert dt < 30
