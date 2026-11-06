"""Regression test for the acceptance audit (CPU, numpy): the keyed (shared-Gumbel) acceptance law the audit relies on.
  1. q == p -> keyed acceptance exactly 1 (posterior key given y, as eval_walk / audit_a / the engine sampler).
  2. (1 - TV)/(1 + TV) <= keyed <= 1 - TV, and keyed > independent (sum p q) for a non-trivial drafter.
  3. A draft drawn with a FRESH key (the broken-coupling bug) matches at sum p q, not at the keyed rate.
  4. Small perturbation (stale policy): keyed mismatch / TV in [1, 2] (measured 1.19 on Nano).
Run: python test_acceptance_bounds.py   (exit 0 = pass)
"""
import numpy as np

rng = np.random.default_rng(0)
V, T, K = 50, 4000, 8


def gumbel(shape):
    return -np.log(-np.log(rng.random(shape).clip(1e-300)))


def lsm(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def posterior_key(lp, y):
    """Gumbel key g given argmax(lp + g) == y (top-down construction used by eval_walk.py / audit_a.py)."""
    e = gumbel(lp.shape)
    z = gumbel((lp.shape[0], 1))
    u = -np.logaddexp(-z, -(lp + e))
    u[np.arange(lp.shape[0]), y] = z[:, 0]
    return u - lp


def rates(lp, lq):
    p, q = np.exp(lp), np.exp(lq)
    y = np.array([rng.choice(V, p=pi / pi.sum()) for pi in p])
    keyed = selfk = fresh = 0.0
    for _ in range(K):
        g = posterior_key(lp, y)
        assert (np.argmax(lp + g, -1) == y).all()
        keyed += (np.argmax(lq + g, -1) == y).mean() / K
        selfk += (np.argmax(lp + g, -1) == y).mean() / K
        fresh += (np.argmax(lq + gumbel(lp.shape), -1) == y).mean() / K
    tv = 0.5 * np.abs(p - q).sum(-1).mean()
    return keyed, selfk, fresh, tv, (p * q).sum(-1).mean()


lp = lsm(rng.normal(0, 2.0, (T, V)))
lq = lsm(lp + rng.normal(0, 0.8, (T, V)))
keyed, selfk, fresh, tv, pq = rates(lp, lq)
print(f"drafter: keyed {keyed:.4f} self {selfk:.4f} fresh-key {fresh:.4f} sum pq {pq:.4f} 1-TV {1 - tv:.4f}")
assert selfk == 1.0, "q == p must be accepted with probability 1 under the shared key"
assert (1 - tv) / (1 + tv) - 0.02 <= keyed <= 1 - tv + 0.02, "keyed outside the Gumbel-coupling bounds"
assert keyed > pq + 0.05, "keyed must beat independent sampling"
assert abs(fresh - pq) < 0.03, "a fresh-key draft must match at sum p q (this is what the broken-coupling bug looks like)"
lq2 = lsm(lp + rng.normal(0, 0.01, (T, V)))
k2, _, _, tv2, _ = rates(lp, lq2)
r = (1 - k2) / tv2
print(f"stale policy: keyed mismatch {1 - k2:.2e} TV {tv2:.2e} ratio {r:.2f}")
assert 0.8 <= r <= 2.2, "keyed mismatch / TV should be O(1) (1.19 measured on Nano)"
print("PASS")
