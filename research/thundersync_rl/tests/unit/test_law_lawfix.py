"""ts-lawfix: the law test's own randomization and chi-square.

1. PIT uniforms of the learner hook must differ across data-parallel replicas: with the pre-fix seed
   (seed, call) every replica's micro-batch c drew the same grid, and the near-deterministic tokens of
   D rows (u ~ U) shared a bin -> overdispersed counts, a perfect sampler FAILED at full Nano (DP 8).
2. chi-square: leftover tail bins (expected < 5) join the last group (the -chk-v11q step-3 emission
   bonus "FAIL" was 2 tokens in a lone group with expectation 0.03).
3. Emission-law uniforms: a 64-bit hash, not crc32 (affine: structured keys gave non-independent uniforms).
"""

import os
import statistics
import sys
import zlib

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from thundersync_rl import law_check  # noqa: E402


def _peaked_rows(gen, n, V=64, f_det=0.7):
    """Logits of n positions: a share f_det near-deterministic (p_top > 0.999), the rest flat-ish."""
    z = torch.randn(n, V, generator=gen, dtype=torch.float64)
    det = torch.rand(n, generator=gen) < f_det
    z[det, 0] += 12.0
    return z


def _replicas(D, calls, L, legacy, seed=5):
    gen = torch.Generator().manual_seed(seed)
    acc = torch.zeros(law_check.N_STATS, dtype=torch.float64)
    for rank in range(D):
        for c in range(calls):
            z = _peaked_rows(gen, L)
            y = torch.multinomial(torch.softmax(z, -1), 1, generator=gen)[:, 0]  # perfect sampler
            s = 42 * 1_000_003 + c if legacy else law_check.pit_seed(42, rank, c)
            U = torch.rand(L, generator=torch.Generator().manual_seed(s), dtype=torch.float64)
            u, t, m = law_check.pit_rows(z.float(), y, U, moments=True)
            acc += law_check.accumulate(u, t, m)
    return law_check.law_test(acc)


def test_shared_grid_across_replicas_fails_a_perfect_sampler_fixed_seed_passes():
    old = _replicas(D=8, calls=4, L=4000, legacy=True)
    new = _replicas(D=8, calls=4, L=4000, legacy=False)
    print(f"[lawfix] DP8 perfect sampler: legacy grid chi2={old['chi2']:.0f}/{int(old['chi2_df'])} p={old['p']:.1e}; "
          f"per-replica grid chi2={new['chi2']:.0f} p={new['p']:.1e}; surprisal z {old['z_surprisal']:.2f} (U-free)")
    assert old["p"] < 1e-6 and old["chi2"] > 2 * old["chi2_df"]
    assert new["p"] > 1e-3
    # The U-free statistics see the same tokens and agree that the sampler is right.
    assert abs(old["z_surprisal"]) < 4 and abs(old["z_prob"]) < 4


def test_single_replica_unaffected():
    assert _replicas(D=1, calls=16, L=4000, legacy=True)["p"] > 1e-3


def test_pit_seed_distinct_and_tp_cp_invariant():
    seeds = {law_check.pit_seed(42, r, c) for r in range(64) for c in range(256)}
    assert len(seeds) == 64 * 256
    assert law_check.pit_seed(42, 3, 7) == law_check.pit_seed(42, 3, 7)  # TP / CP ranks of one replica agree
    assert 0 <= law_check.pit_seed(1, 0, 0) < 2**63


def test_shadow_uniforms_share_the_sums():
    gen = torch.Generator().manual_seed(3)
    z = torch.randn(300, 40, generator=gen)
    y = torch.randint(0, 40, (300,), generator=gen)
    U1 = torch.rand(300, dtype=torch.float64, generator=gen)
    U2 = torch.rand(300, dtype=torch.float64, generator=gen)
    u1, t1, m1, u2, t2 = law_check.pit_rows(z, y, U1, moments=True, shadow_uniforms=U2)
    a2, b2 = law_check.pit_rows(z, y, U2)
    assert torch.equal(u2, a2) and torch.equal(t2, b2)
    a1, b1, _ = law_check.pit_rows(z, y, U1, moments=True)
    assert torch.equal(u1, a1) and torch.equal(t1, b1)


# -chk-v11q step 3 emission-law bonus counts (history.json law_emit_bonus/stats, bins 0..68).
_BONUS_STEP3 = [4, 7, 4, 4, 9, 5, 8, 8, 8, 4, 1, 3, 3, 4, 2, 7, 7, 6, 4, 6, 7, 4, 4, 5, 6, 1, 5, 4, 6, 4, 4, 4, 6, 5,
                9, 3, 2, 6, 4, 5, 6, 11, 4, 6, 4, 6, 8, 7, 1, 6, 4, 4, 5, 4, 5, 7, 2, 12, 5, 4, 3, 6, 4, 2, 2, 0, 2, 0, 0]


def test_chi2_leftover_tail_joins_last_group():
    st = _BONUS_STEP3 + [328.0, 164.35371548896336, 337.78492900046086]
    r = law_check.law_test(st)
    print(f"[lawfix] -chk-v11q step-3 bonus: chi2={r['chi2']:.1f}/{int(r['chi2_df'])} p={r['p']:.2e}")
    assert r["chi2"] < 100 and r["p"] > 1e-3
    # A real tail excess still fails: 30 of 328 tokens at t <= 1e-4 (expected 0.03).
    bad = list(_BONUS_STEP3)
    bad[66] = 30
    rb = law_check.law_test(bad + [sum(bad), 164.0, 400.0])
    assert rb["p"] < 1e-6, rb
    # Calibration at small n (every bin merges): p-values ~uniform-ish, few below 1e-3.
    gen = torch.Generator().manual_seed(9)
    ps = []
    for _ in range(300):
        u = torch.rand(300, dtype=torch.float64, generator=gen)
        ps.append(law_check.law_test(law_check.accumulate(u, 1 - u))["p"])
    assert sum(p < 1e-3 for p in ps) <= 2


def _chi2_64(us):
    c = [0] * 64
    for u in us:
        c[min(int(u * 64), 63)] += 1
    e = len(us) / 64
    return sum((x - e) ** 2 / e for x in c)


def test_hash_uniforms_independent_where_crc32_was_not():
    crc, h = [], []
    for trial in range(150):
        seed = 1_000_003 * trial + 17
        keys = [f"{trial % 7}|c{i}" for i in range(330)]
        crc.append(_chi2_64([(zlib.crc32(f"{seed}/{k}/128".encode()) + 0.5) / 2**32 for k in keys]))
        h.append(_chi2_64([law_check.hash_uniform("emit", seed, k, 128) for k in keys]))
    sd_crc, sd_h = statistics.pstdev(crc), statistics.pstdev(h)
    print(f"[lawfix] chi2(63) over 150 calls of 330 structured keys: crc32 mean {statistics.mean(crc):.1f} sd {sd_crc:.1f}; "
          f"blake2b mean {statistics.mean(h):.1f} sd {sd_h:.1f} (H0 63 / 11.2)")
    assert abs(statistics.mean(h) - 63) < 4 and 8.5 < sd_h < 14.5
    assert sd_crc > 14.5 or abs(statistics.mean(crc) - 63) > 6  # documents the crc32 defect
