"""Acceptance audit, part A (GPU, 1 node x 4): is the draft head's keyed acceptance at its first-principles value?

Per depth-1 target position t (base b = t - 2, head input (h_b, x_{b+1}), q = head's q_1, p = Nano softmax at t-1):
  calib   : CE = mean -log p(y_t) vs H = mean entropy(p_t). y ~ Nano, so CE == H iff the p we compute is the sampling
            distribution (wrong layer / pre-norm / off-by-one would make CE > H).
  det     : a second independent Nano forward of the same row: fraction of positions with bitwise-equal log p, and the
            keyed acceptance of drafting with that second forward's p (q := p_2). Must be 1.0 (coupling sanity).
  self    : keyed acceptance with q := p (the same tensor). Must be exactly 1.0.
  keyed   : keyed acceptance of the head (posterior Gumbel key given y_t, the engine's coupling; K keys).
  indep   : acceptance with an INDEPENDENT key for the draft (prior Gumbel): what a broken key coupling would give;
            expectation E_p[q(y)] = sum p q.
  1-TV    : sum min(p, q): the maximal-coupling upper bound on any single-draft acceptance.
  sum p^2 : collision probability: independent acceptance of a PERFECT drafter (text-difficulty index).
  KL      : KL(p || q) nats/token; top1 = argmax q == argmax p.
Binned by entropy(p). Per row: keyed alpha and mean entropy (who are the slowest rows).
  torchrun --nproc-per-node 4 audit_a.py --shim <shim> --init <head.pt> --data <pkl> --tag <t> --out <dir> --limit 64
"""
import argparse
import os
import pickle
import sys
import zlib

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_chain as tc  # noqa: E402

RANK, WORLD, log = tc.RANK, tc.WORLD, tc.log
HB = [0.0, 0.1, 0.5, 1.0, 2.0, 1e9]
KEYS = ["n", "H", "CE", "det_eq", "det_keyed", "self_keyed", "keyed", "indep", "mintv", "pq", "p2", "kl", "top1"]


def enable_bi():
    """BI recipe of the RL run configs: BI kernels + fp32 residual + fp16 GEMM, and the
    provider fields, injected by wrapping AutoBridge.to_megatron_provider (train_head.build stays unchanged)."""
    from megatron.bridge import AutoBridge
    from megatron.core.transformer.enums import AttnBackend
    from nemo_rl.models.megatron import fp16_gemm, fp32_residual
    from nemo_rl.models.megatron.zero_train_gen_mismatch import enable_batch_invariant_kernels

    enable_batch_invariant_kernels({"megatron_cfg": {"batch_invariant_mode": True, "batch_invariant_backend": "te_native",
                                                     "batch_invariant_collective": "ordered",
                                                     "flash_attention_version": 4, "attention_backend": "flash"}})
    fp32_residual.install()
    if os.environ.get("AUDIT_FP16_GEMM") == "1":  # needs fp16 expert params (store_routed_experts_fp16); off by default
        fp16_gemm.install()
    from nemo_rl.models.megatron import bi_unpermute, moe_dispatcher_refs

    bi_unpermute.install()  # as megatron_policy_worker for batch_invariant_mode
    moe_dispatcher_refs.install()
    orig = AutoBridge.to_megatron_provider

    def to_provider(self, *a, **kw):
        prov = orig(self, *a, **kw)
        prov.fp32_residual_connection = True
        prov.attention_backend = AttnBackend.flash
        prov.flash_attention_version = 4
        prov.batch_invariant_mode = True
        prov.batch_invariant_backend = "te_native"
        prov.batch_invariant_collective = "ordered"
        return prov

    AutoBridge.to_megatron_provider = to_provider


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shim", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--keys", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--crossfade", type=int, default=1)
    ap.add_argument("--bi", type=int, default=1, help="learner forward on the BI recipe (as the RL run configs)")
    ns = ap.parse_args()
    os.makedirs(ns.out, exist_ok=True)
    if ns.bi:
        enable_bi()
    R, E, head = tc.nano(ns)
    log(f"[{ns.tag}] bi {ns.bi} provider batch_invariant_mode {getattr(R.m.config, 'batch_invariant_mode', None)} "
        f"backend {getattr(R.m.config, 'batch_invariant_backend', None)} fp32_residual "
        f"{getattr(R.m.config, 'fp32_residual_connection', None)} fa {getattr(R.m.config, 'flash_attention_version', None)}")
    d = tc.load(ns.data)
    rows = sorted(d["rows"], key=lambda r: (r["j"], r["idx"], r["run"]))[:ns.limit][RANK::WORLD]
    nb = len(HB) - 1
    S = {k: torch.zeros(nb, dtype=torch.float64, device="cuda") for k in KEYS}
    per_row = []
    Wf = R.Wf
    hb = torch.tensor(HB[1:-1], device="cuda")
    for r in rows:
        x, P = tc.row_tensor(d, r)
        N = x.numel()
        B0 = max(P - 1, 0)
        with torch.no_grad():
            h1 = tc.hidden(R, x)
            h2 = tc.hidden(R, x)  # independent second forward (determinism / q := p_2)
            outs = head.chain(h1, E, x, B0, 1)
            g = torch.Generator(device="cuda")
            rk = torch.zeros(2, dtype=torch.float64, device="cuda")
            rh = torch.zeros(1, dtype=torch.float64, device="cuda")
            for lo in range(B0 + 2, N, ns.chunk):
                hi = min(N, lo + ns.chunk)
                y = x[lo:hi]
                lp = torch.log_softmax(h1[lo - 1:hi - 1].float() @ Wf.t(), -1)
                lp2 = torch.log_softmax(h2[lo - 1:hi - 1].float() @ Wf.t(), -1)
                lq = torch.log_softmax(outs[0][lo - 2:hi - 2].float() @ Wf.t(), -1)  # base b = t - 2
                p, q = lp.exp(), lq.exp()
                H = -(p * lp).sum(-1)
                bi = torch.bucketize(H, hb)

                def add(k, v):
                    S[k].index_add_(0, bi, v.double())

                add("n", torch.ones_like(H))
                add("H", H)
                add("CE", -lp.gather(1, y[:, None])[:, 0])
                add("det_eq", (lp2 == lp).all(-1).float())
                add("mintv", torch.minimum(p, q).sum(-1))
                add("pq", (p * q).sum(-1))
                add("p2", (p * p).sum(-1))
                add("kl", (p * (lp - lq)).sum(-1))
                add("top1", (lq.argmax(-1) == lp.argmax(-1)).float())
                g.manual_seed((zlib.crc32(f"{r['idx']}-{r['j']}-{r['run']}-{lo}".encode()) * 2654435761) % (2**62))
                for _ in range(ns.keys):
                    e = tc.gumbel(lp.shape, g)
                    z = tc.gumbel((hi - lo, 1), g)
                    u = -torch.logaddexp(-z, -(lp + e))
                    u.scatter_(1, y[:, None], z)  # posterior perturbed scores: argmax(u) == y
                    gk = u - lp  # the Gumbel key itself
                    kd = (torch.argmax(lq + gk, -1) == y).float()
                    add("keyed", kd / ns.keys)
                    add("self_keyed", (torch.argmax(lp + gk, -1) == y).float() / ns.keys)
                    add("det_keyed", (torch.argmax(lp2 + gk, -1) == y).float() / ns.keys)
                    gi = tc.gumbel(lp.shape, g)  # independent prior key for the draft
                    add("indep", (torch.argmax(lq + gi, -1) == y).float() / ns.keys)
                    rk[0] += kd.sum().double()
                    rk[1] += kd.numel()
                    del e, u, gk, gi
                rh += H.sum().double()
                del lp, lp2, lq, p, q
            per_row.append(dict(idx=r["idx"], j=r["j"], L=N - P, alpha=(rk[0] / rk[1]).item(),
                                H=(rh / max(N - B0 - 2, 1)).item()))
            del h1, h2, outs
    for k in KEYS:
        dist.all_reduce(S[k])
    out = [None] * WORLD
    dist.all_gather_object(out, per_row)
    if RANK == 0:
        pr = [x_ for o in out for x_ in o]
        res = {k: S[k].tolist() for k in KEYS}
        pickle.dump(dict(bins=HB, stats=res, rows=pr), open(os.path.join(ns.out, f"audit_a_{ns.tag}.pkl"), "wb"))
        n = S["n"]
        tot = {k: (S[k].sum() / n.sum()).item() for k in KEYS if k != "n"}
        log(f"[{ns.tag}] rows {len(pr)} positions {int(n.sum().item())}")
        log(f"[{ns.tag}] POOLED " + " ".join(f"{k} {v:.4f}" for k, v in tot.items()))
        for i in range(nb):
            if n[i] > 0:
                log(f"[{ns.tag}] H in [{HB[i]}, {HB[i + 1]}): share {(n[i] / n.sum()).item():.3f} " +
                    " ".join(f"{k} {(S[k][i] / n[i]).item():.4f}" for k in KEYS if k != "n"))
        pr.sort(key=lambda z: z["alpha"])
        for z in pr[:5] + pr[-3:]:
            log(f"[{ns.tag}] row idx {z['idx']} j {z['j']} L {z['L']} keyed alpha {z['alpha']:.3f} mean H {z['H']:.3f}")
        import math
        a = [z["alpha"] for z in pr]
        h = [z["H"] for z in pr]
        ma, mh = sum(a) / len(a), sum(h) / len(h)
        cov = sum((u - ma) * (v - mh) for u, v in zip(a, h))
        cor = cov / math.sqrt(sum((u - ma) ** 2 for u in a) * sum((v - mh) ** 2 for v in h))
        log(f"[{ns.tag}] per-row corr(keyed alpha, mean entropy) {cor:.3f}")
    dist.barrier()
    log("done")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
