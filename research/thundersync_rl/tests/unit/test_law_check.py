"""Law test (law_check.py): calibrated under the exact verifier, catches injected law bugs.

Toy autoregressive target with prefix-independent per-position rows p_t, drafts from q_t,
the REAL block verifier (``block_verification.block_verify``), then the row completed by
exact sampling from p (as the loop completes every row), PIT of every emitted token.
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from thundersync_rl import block_verification as bv  # noqa: E402
from thundersync_rl import law_check  # noqa: E402


def _rows(n_rows, L=10, V=24, g=6, seed=0, temp_bug=None):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    zp = torch.randn(L, V, generator=gen, dtype=torch.float64) * 2.0
    zq = zp + 0.8 * torch.randn(L, V, generator=gen, dtype=torch.float64)
    lp = torch.log_softmax(zp, -1)
    lq = torch.log_softmax(zq, -1)
    stats = torch.zeros(law_check.N_STATS, dtype=torch.float64)
    rejections = 0
    for _ in range(n_rows):
        q_draw = lq.exp() if temp_bug is None else torch.softmax(zq / temp_bug, -1)
        draft = torch.multinomial(q_draw[:g].float(), 1, generator=gen)[:, 0]
        tau, y = bv.block_verify(lp[: g + 1].float(), lq[:g].float(), draft, gen)
        toks = draft[:tau].tolist() + [y]
        rejections += int(tau < g)
        while len(toks) < L:
            t = len(toks)
            toks.append(int(torch.multinomial(lp[t].exp().float(), 1, generator=gen)))
        y_t = torch.tensor(toks)
        U = torch.rand(L, dtype=torch.float64, generator=gen)
        u, t, mom = law_check.pit_rows(lp.float(), y_t, U, moments=True)
        stats += law_check.accumulate(u, t, mom)
    return stats, rejections


def test_pit_uniform_under_exact_verifier():
    stats, rej = _rows(3000, seed=1)
    assert rej > 300  # the residual path is exercised
    r = law_check.law_test(stats)
    assert r["n"] == 30000
    assert r["p"] > 1e-3, r


@pytest.mark.parametrize("bug", ["residual_from_p", "residual_from_q"])
def test_detects_wrong_residual_law(monkeypatch, bug):
    orig = bv._emit_dist

    def wrong_residual(lp, lq, b, tau, g, null_m=1.0):
        # BUG: the token at a rejection drawn from p (or from q) instead of the residual
        # (b p - q)+ / r. ~6% of the emitted tokens are residual draws here.
        if tau < g:
            return (lp if bug == "residual_from_p" else lq)[tau].double().exp()
        return orig(lp, lq, b, tau, g, null_m)

    monkeypatch.setattr(bv, "_emit_dist", wrong_residual)
    stats, rej = _rows(20000, seed=1)
    r = law_check.law_test(stats)
    print(f"[law test] injected {bug}:", law_check.format_line(0, stats))
    assert r["p"] < 1e-5, r


def test_detects_wrong_draft_law_reported_as_q():
    # The engine draws drafts at temperature 0.7 but logs log q at T=1: the BV output law
    # is no longer p (the old checks compare log-prob values only and pass).
    stats, _ = _rows(3000, seed=2, temp_bug=0.7)
    r = law_check.law_test(stats)
    print("[law test] injected sampler-temperature bug:", law_check.format_line(0, stats))
    assert r["p"] < 1e-6, r


def test_detects_truncated_tail():
    # Sampler that never emits tokens below 2% of the row max (a tail floor).
    gen = torch.Generator().manual_seed(3)
    V, n = 64, 40000
    z = torch.randn(n, V, generator=gen) * 2.5
    p = torch.softmax(z.double(), -1)
    keep = p >= 0.02 * p.max(-1, keepdim=True).values
    p_bad = torch.where(keep, p, 0.0)
    y = torch.multinomial(p_bad.float(), 1, generator=gen)[:, 0]
    U = torch.rand(n, dtype=torch.float64, generator=gen)
    u, t, mom = law_check.pit_rows(z, y, U, moments=True)
    r = law_check.law_test(law_check.accumulate(u, t, mom))
    assert r["p"] < 1e-6 and r["z_tail"] < 0, r
    # and the exact sampler passes on the same rows
    y0 = torch.multinomial(p.float(), 1, generator=gen)[:, 0]
    u0, t0, m0 = law_check.pit_rows(z, y0, U, moments=True)
    assert law_check.law_test(law_check.accumulate(u0, t0, m0))["p"] > 1e-3


def test_chi2_and_ks_tails():
    assert abs(law_check._chi2_sf(68.0, 68) - 0.4776) < 0.01  # median-ish
    assert law_check._chi2_sf(200.0, 68) < 1e-13
    assert abs(law_check._ks_sf(1.36) - 0.0494) < 2e-3
    assert abs(sum(law_check.bin_probs()) - 1.0) < 1e-12


def test_ties_and_tp_shards_equal_whole():
    # The vocab split over two "ranks" (no process group: sums by hand) equals the whole row.
    gen = torch.Generator().manual_seed(5)
    z = torch.randn(50, 40, generator=gen)
    z[:, 3] = z[:, 7]  # ties
    y = torch.randint(0, 40, (50,), generator=gen)
    U = torch.rand(50, dtype=torch.float64, generator=gen)
    u, t = law_check.pit_rows(z, y, U)
    assert torch.allclose(u + t, torch.ones_like(u), atol=1e-12)
    assert ((u >= 0) & (u <= 1)).all()


@pytest.mark.parametrize("bug", ["", "residual_from_p"])
def test_emission_law_check_with_engine_q_verifier(monkeypatch, bug):
    """Verifier-side emission law (worker._emission_law): PIT of each emitted token of the REAL plain-BV
    verifier (engine_q.verify_engine_q) under the BV law recomputed from the definition. Full power per residual
    draw: ~300 rejections suffice where the marginal test needs ~1e4 (test_detects_wrong_residual_law)."""
    import zlib

    from thundersync_rl.engine_q import verify_engine_q

    if bug:
        monkeypatch.setenv("THUNDERSYNC_DEBUG_LAW_BUG", bug)
    g, V, gen = 6, 24, torch.Generator().manual_seed(11)
    zp = torch.randn(g + 1, V, generator=gen, dtype=torch.float64) * 2.0
    zq = zp[:g] + 0.5 * torch.randn(g, V, generator=gen, dtype=torch.float64)
    lp = torch.log_softmax(zp, -1).float()
    lq_rows = torch.log_softmax(zq, -1).float()
    us, ts, n_res = [], [], 0
    for k in range(1500):
        draft = torch.multinomial(lq_rows.exp(), 1, generator=gen)[:, 0]
        lqx = lq_rows[torch.arange(g), draft]
        tau, y, _, _ = verify_engine_q(lp, draft, lqx, lambda idx: lq_rows[idx.long()], gen)
        dist = law_check.reference_emission(lp[tau], None if tau == g else lq_rows[tau], lp[torch.arange(g), draft], lqx, tau)
        u, t = law_check.emission_pit(dist, int(y), (zlib.crc32(f"{k}".encode()) + 0.5) / 2**32)
        if tau < g:
            us.append(u)
            ts.append(t)
    r = law_check.law_test(law_check.accumulate(torch.tensor(us, dtype=torch.float64), torch.tensor(ts, dtype=torch.float64)))
    print(f"[law test] emission check bug={bug or 'none'}: residual draws {len(us)} p={r['p']:.2e}")
    assert len(us) > 200
    if bug:
        assert r["p"] < 1e-6, r
    else:
        assert r["p"] > 1e-3, r


@pytest.mark.parametrize("bug", ["", "residual_from_p", "null_m_1"])
def test_null_bv_emission_law_check(monkeypatch, bug):
    """⊥-BV grid emissions (engine_q.verify_engine_q_grid, the REAL code) vs the ⊥-BV law at envelope M recomputed
    from the definition (law_check.reference_null_emission, BOT column included), M fixed before the draws as the
    pool counts fix it."""
    import zlib

    from thundersync_rl.engine_q import verify_engine_q_grid

    if bug:
        monkeypatch.setenv("THUNDERSYNC_DEBUG_LAW_BUG", bug)
    g, V, gen = 6, 24, torch.Generator().manual_seed(13)
    zp = torch.randn(g + 1, V, generator=gen, dtype=torch.float64) * 2.0
    zq = zp[:g] + 0.5 * torch.randn(g, V, generator=gen, dtype=torch.float64)
    lp = torch.log_softmax(zp, -1).float()
    lq_rows = torch.log_softmax(zq, -1).float()
    envelopes = (1.5, 2.0, 3.0)
    us, ts, kinds = [], [], {"bot": 0, "residual": 0, "bonus": 0}
    for k in range(1500):
        draft = torch.multinomial(lq_rows.exp(), 1, generator=gen)[:, 0]
        lqx = lq_rows[torch.arange(g), draft]
        grid, lpx, _ = verify_engine_q_grid(lp, draft, lqx, lambda idx: lq_rows[idx.long()], gen, envelopes)
        m = envelopes[k % 3]  # the pool's choice: fixed before this row's draws
        tau, y, _ = grid[m]
        dist = law_check.reference_null_emission(lp[tau], None if tau == g else lq_rows[tau], lpx, lqx, tau, m)
        kinds["bot" if y < 0 else ("bonus" if tau == g else "residual")] += 1
        u, t = law_check.emission_pit(dist, int(y) if y >= 0 else dist.numel() - 1, (zlib.crc32(f"{k}".encode()) + 0.5) / 2**32)
        us.append(u)
        ts.append(t)
    r = law_check.law_test(law_check.accumulate(torch.tensor(us, dtype=torch.float64), torch.tensor(ts, dtype=torch.float64)))
    print(f"[law test] null-BV emission check bug={bug or 'none'}: {kinds} p={r['p']:.2e}")
    assert kinds["residual"] > 100 and (kinds["bot"] > 50 or bug == "null_m_1")  # null_m_1 drops BOT
    if bug:
        assert r["p"] < 1e-6, r
    else:
        assert r["p"] > 1e-3, r


def test_truncated_top_p_policy_needs_the_nucleus_row_fn():
    """ns-topp (ts-nano-scale): samples of the exact top-p policy pass the law test only when the
    PIT is taken under the truncated row (row_fn = the canonical nucleus mask); under the
    untruncated softmax they fail (the in-loop failure); untruncated samples under the truncated
    row fail (the original merge bug: the race sampled outside the nucleus)."""
    from nemo_rl.models.megatron import exact_nucleus

    gen = torch.Generator().manual_seed(11)
    V, n, top_p = 64, 30000, 0.3
    z = torch.randn(n, V, generator=gen) * 2.0
    keep = exact_nucleus.nucleus_mask(z, top_p, None, 0.0)
    zt = z.masked_fill(~keep, float("-inf"))
    pt = torch.softmax(zt.double(), -1)
    y = torch.multinomial(pt.float(), 1, generator=gen)[:, 0]
    U = torch.rand(n, dtype=torch.float64, generator=gen)

    def row_fn(zz, rows):
        k = exact_nucleus.nucleus_mask(zz, top_p, None, 0.0)
        return zz.masked_fill(~k, float("-inf"))

    u, t, m = law_check.pit_rows(z, y, U, moments=True, row_fn=row_fn)
    r = law_check.law_test(law_check.accumulate(u, t, m))
    assert r["p"] > 1e-3 and math.isfinite(r["z_surprisal"]), r
    u, t, m = law_check.pit_rows(z, y, U, moments=True)  # untruncated p: the ns-topp failure
    assert law_check.law_test(law_check.accumulate(u, t, m))["p"] < 1e-6
    y_full = torch.multinomial(torch.softmax(z.double(), -1).float(), 1, generator=gen)[:, 0]
    u, t, m = law_check.pit_rows(z, y_full, U, moments=True, row_fn=row_fn)  # negative control
    assert law_check.law_test(law_check.accumulate(u, t, m))["p"] < 1e-6
