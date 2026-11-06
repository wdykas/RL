"""Draft-head2 held-out evaluation: keyed per-insertion acceptance by depth + simulated chain walk (tokens/forward).

Reuses train_chain.py (Head arithmetic == tree_mtp.MTPHead, parity-checked there) and its keyed rule:
per target position t, K posterior Gumbel keys given the realised y_t (exactly the engine's Gumbel-max coupling at T 1,
conditioned on the token Nano actually sampled), depth-k draft from base b = argmax(log q_k + u_t - log p_t).

Walk: for key k the engine at base b drafts depths 1..D, the verify accepts the leading run a of matching drafts and
commits a + 1 tokens (the bonus token is Nano's own sample); the next base is b + a + 1. steps = verify forwards over
the row's L generated tokens; s = L / steps. Reported per D in 1..Dmax (the chain truncated at D), per row mean over K
keys; "slowest row" = the row with the most verify steps (the critical row in a synchronous batch), plus min-s row.

  torchrun --nproc-per-node 4 eval_walk.py --shim <trained depth-1 shim> --init <head .pt> --out <dir> [--depth 3] [--keys 4]
"""
import argparse
import os
import pickle
import sys
import time
import zlib

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_chain as tc  # noqa: E402

RANK, WORLD, log = tc.RANK, tc.WORLD, tc.log


def walk(A, D, L):
    """A [Dmax, nb, K] bool keyed match per (depth, base offset, key). Returns (steps per key for chain depth D,
    reach[d] = number of visited bases whose chain reached depth d + 1, visits) -- reach / visits is the engine's
    rates_head (unconditional acceptance per insertion index over the bases the walk actually visits)."""
    Dm, nb, K = A.shape
    run = torch.cumprod(A[:D].int(), 0).sum(0).cpu()  # [nb, K] leading accepted run length at each base
    steps, reach, vis = [], [0] * D, 0
    split = [0, 0, 0, 0]  # depth-1 (trials, accepts) after a base iteration that accepted only the root | deeper
    for k in range(K):
        r = run[:, k].tolist()
        i, n, prev = 0, 0, -1
        while i < nb:
            for d in range(r[i]):
                reach[d] += 1
            if prev >= 0:
                g = 0 if prev == 0 else 2
                split[g] += 1
                split[g + 1] += int(r[i] > 0)
            prev = r[i]
            vis += 1
            i += r[i] + 1
            n += 1
        n += max(0, L - max(i, nb))  # the last <= Dmax tokens (no common base): one token per forward
        steps.append(n)
    return steps, reach, vis, split


def evaluate(ns, R, E, head):
    ho = tc.load(ns.heldout)
    allrows = sorted(ho["rows"], key=lambda r: (r["idx"], r["j"], r["run"]))
    if ns.limit:
        allrows = sorted(allrows, key=lambda r: (r["j"], r["idx"], r["run"]))[:ns.limit]
    rows = allrows[RANK::WORLD]
    K, C, Dp = ns.keys, ns.chunk, ns.depth
    Wf = R.Wf
    res, t0 = [], time.time()
    for ri, r in enumerate(rows):
        x, P = tc.row_tensor(ho, r)
        N = x.numel()
        B0 = max(P - 1, 0)
        nb = N - 1 - Dp - B0
        if nb <= 0:
            continue
        with torch.no_grad():
            hm_ = tc.hidden(R, x)
            outs = head.chain(hm_, E, x, B0, Dp)
            A = torch.zeros(Dp, nb, K, dtype=torch.bool, device="cuda")
            g = torch.Generator(device="cuda")
            for lo in range(B0 + 2, N, C):
                hi = min(N, lo + C)
                t = torch.arange(lo, hi, device="cuda")
                y = x[lo:hi]
                lp = torch.log_softmax(hm_[lo - 1:hi - 1].float() @ Wf.t(), -1)
                lqs = []
                for d in range(1, Dp + 1):
                    b = t - d - 1
                    sel = (b >= B0) & (b < B0 + nb)
                    if sel.any():
                        off = 0 if d == 1 else B0
                        rowsel = (b - off).clamp(0, outs[d - 1].shape[0] - 1)
                        lqs.append((torch.log_softmax(outs[d - 1][rowsel].float() @ Wf.t(), -1), b, sel))
                    else:
                        lqs.append((None, b, sel))
                g.manual_seed((zlib.crc32(f"{r['idx']}-{r['j']}-{r['run']}-{lo}".encode()) * 2654435761) % (2**62))
                for k in range(K):
                    e = tc.gumbel(lp.shape, g)
                    z = tc.gumbel((hi - lo, 1), g)
                    u = -torch.logaddexp(-z, -(lp + e))
                    u.scatter_(1, y[:, None], z)  # posterior key: argmax(lp + u) == y exactly
                    u = u - lp
                    for d, (lq, b, sel) in enumerate(lqs):
                        if lq is not None:
                            dd = torch.argmax(lq + u, -1)
                            A[d, (b - B0)[sel], k] = (dd == y)[sel]
                    del e, u
                del lp, lqs
            Cc = torch.cumprod(A.float(), 0)
            num = Cc.sum((1, 2)).double()
            den = torch.cat([torch.tensor([nb * K], device="cuda", dtype=torch.float64), num[:-1]])
        L = N - P
        wk = {D: walk(A, D, L) for D in range(1, Dp + 1)}
        steps = {D: v[0] for D, v in wk.items()}
        rec = dict(idx=r["idx"], j=r["j"], run=r["run"], P=P, L=L, nb=nb, num=num.tolist(), den=den.tolist(),
                   steps=steps, reach=wk[Dp][1], vis=wk[Dp][2], split=wk[Dp][3])
        res.append(rec)
        if ri % 64 == 0:
            al = [n_ / max(d_, 1e-9) for n_, d_ in zip(rec["num"], rec["den"])]
            log(f"row {ri}/{len(rows)} L {L} alpha " + " ".join(f"{a:.3f}" for a in al) + " s " +
                " ".join(f"{L * K / sum(steps[D]):.3f}" for D in steps) + f" {time.time() - t0:.0f}s")
        del hm_, outs, A
    out = [None] * WORLD
    dist.all_gather_object(out, res)
    if RANK == 0:
        allres = [x_ for o in out for x_ in o]
        pickle.dump(allres, open(os.path.join(ns.out, f"{ns.tag}.pkl"), "wb"))
        summarize(allres, ns.tag, Dp, K)
    dist.barrier()


def summarize(res, tag, Dp, K):
    def q(v, f):
        v = sorted(v)
        return v[min(len(v) - 1, int(f * len(v)))]

    log(f"[{tag}] rows {len(res)} tokens {sum(r['L'] for r in res)} keys {K}")
    for d in range(Dp):
        num = sum(r["num"][d] for r in res)
        den = sum(r["den"][d] for r in res)
        per = [r["num"][d] / r["den"][d] for r in res if r["den"][d] > 0]
        log(f"[{tag}] depth {d + 1}: per-insertion keyed alpha pooled {num / den:.4f} | per-row mean "
            f"{sum(per) / len(per):.4f} p10 {q(per, 0.10):.4f} min {min(per):.4f}")
    vis = sum(r["vis"] for r in res)
    rh = [sum(r["reach"][d] for r in res) / vis for d in range(Dp)]
    log(f"[{tag}] walk-visited bases (chain D {Dp}): unconditional rate per insertion index (engine rates_head) "
        + " ".join(f"{x:.4f}" for x in rh) + " | conditional " + " ".join(
            f"{rh[d] / (rh[d - 1] if d else 1.0):.4f}" for d in range(Dp)))
    sp = [sum(r["split"][i] for r in res) for i in range(4)]
    log(f"[{tag}] depth-1 split (walk, chain D {Dp}): after a=1 (root only) {sp[1] / max(sp[0], 1):.4f} of {sp[0]} | "
        f"after a>=2 {sp[3] / max(sp[2], 1):.4f} of {sp[2]}")
    for D in range(1, Dp + 1):
        s = [r["L"] * K / sum(r["steps"][D]) for r in res]
        st = [sum(r["steps"][D]) / K for r in res]
        slow = max(res, key=lambda r: sum(r["steps"][D]))
        pooled = sum(r["L"] for r in res) * K / sum(sum(r["steps"][D]) for r in res)
        log(f"[{tag}] chain D {D}: tokens/forward pooled {pooled:.3f} median-row {q(s, 0.5):.3f} p10 {q(s, 0.1):.3f} "
            f"min-row {min(s):.3f} | slowest row (most verify steps {max(st):.0f}): L {slow['L']} "
            f"s {slow['L'] * K / sum(slow['steps'][D]):.3f} (idx {slow['idx']} j {slow['j']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shim", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="walk")
    ap.add_argument("--heldout", default="", help="rows pickle (default $DRAFT_HEAD_DATA/heldout.pkl)")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--keys", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--crossfade", type=int, default=1)
    ns = ap.parse_args()
    ns.heldout = ns.heldout or tc.th.data_path("heldout")
    os.makedirs(ns.out, exist_ok=True)
    R, E, head = tc.nano(ns)
    evaluate(ns, R, E, head)
    log("done")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
