"""Engine-head parity on whole rows: the engine's tree_mtp.MTPHead (triton attention, bf16 weights, its forward_rows
chaining exactly as tree_spec._mtp_device does: depth k at position b + k - 1, input = previous depth's returned hf)
vs train_chain.Head (the training arithmetic), every base of every row, same posterior Gumbel keys.

Reports per depth: keyed conditional acceptance of each head, the fraction of (base, key) drafts that differ, and
max |hf diff|. If MTPHead's acceptance matches Head's, the head arithmetic in the engine is not the cause of any
engine-vs-offline gap (that would then sit in tree_spec's bookkeeping).

  torchrun --nproc-per-node 4 check_engine_head.py --shim <shim> --init <head .pt> --heldout <rows.pkl> --out <dir>
     [--tree_mtp <tree_mtp.py>] [--depth 4] [--keys 8] [--limit 0]
"""
import argparse
import importlib.util
import os
import sys
import zlib

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_chain as tc  # noqa: E402

RANK, WORLD, log = tc.RANK, tc.WORLD, tc.log


def eng_chain(mh, hm_, x, B0, D):
    """MTPHead chain from every base b in [B0, N-2]: one KV slot per base (true K/V of 0..b, then its own chain)."""
    N = x.numel()
    bases = torch.arange(B0, N - 1, device="cuda")
    nb = bases.numel()
    # true K/V of every position into slot 0, then broadcast to all slots
    mh.append_kv(torch.zeros(N - 1, dtype=torch.long, device="cuda"), torch.arange(N - 1, device="cuda"),
                 hm_[:N - 1], x[1:N])
    mh.kc[1:nb] = mh.kc[0:1]
    mh.vc[1:nb] = mh.vc[0:1]
    sl = torch.arange(nb, device="cuda")
    u, outs = hm_[bases], []
    for d in range(1, D + 1):
        tok = x[(bases + d).clamp(max=N - 1)]
        u, _lg = mh.forward_rows(sl, bases + d - 1, u, tok)
        outs.append(u.clone())
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shim", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--tree_mtp", default="", help="tree_mtp.py file (default: the installed nemo_rl module)")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--keys", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--crossfade", type=int, default=1)
    ns = ap.parse_args()
    os.makedirs(ns.out, exist_ok=True)
    R, E, head = tc.nano(ns)
    tm = tc.th.load_tree_mtp(ns.tree_mtp)
    ho = tc.load(ns.heldout)
    rows = sorted(ho["rows"], key=lambda r: (r["idx"], r["j"], r["run"]))
    if ns.limit:
        rows = rows[:ns.limit]
    rows = rows[RANK::WORLD]
    maxN = max(len(ho["prompts"][r["idx"]]) + len(r["tokens"]) for r in ho["rows"]) + 8
    maxB = max(len(r["tokens"]) for r in ho["rows"]) + 8
    mh = tm.MTPHead(ns.init, E, R.W, slots=maxB, max_seq=maxN, num_heads=head.nh, num_kv_heads=head.nkv,
                    head_dim=head.hd, eps=head.eps)
    Dp, K, C, Wf = ns.depth, ns.keys, ns.chunk, R.Wf
    st = torch.zeros(2, Dp, 2, dtype=torch.float64, device="cuda")  # [head, depth, (chain num, den)]
    diff = torch.zeros(Dp, 2, dtype=torch.float64, device="cuda")
    hmax = torch.zeros(Dp, device="cuda")
    for r in rows:
        x, P = tc.row_tensor(ho, r)
        N = x.numel()
        B0 = max(P - 1, 0)
        nb = N - 1 - Dp - B0
        if nb <= 0:
            continue
        with torch.no_grad():
            hm_ = tc.hidden(R, x)
            mine = head.chain(hm_, E, x, B0, Dp)
            mine = [mine[0][B0:]] + mine[1:]  # every depth indexed by base - B0
            eng = eng_chain(mh, hm_, x, B0, Dp)
            for d in range(Dp):
                hmax[d] = torch.maximum(hmax[d], (mine[d][:nb].float() - eng[d][:nb].float()).abs().max())
            A = torch.zeros(2, Dp, nb, K, dtype=torch.bool, device="cuda")
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
                    if not sel.any():
                        lqs.append(None)
                        continue
                    i = (b - B0).clamp(0, nb - 1)
                    lqs.append(([torch.log_softmax(o[d - 1][i].float() @ Wf.t(), -1) for o in (mine, eng)], b, sel))
                g.manual_seed((zlib.crc32(f"{r['idx']}-{r['j']}-{r['run']}-{lo}".encode()) * 2654435761) % (2**62))
                for k in range(K):
                    e = tc.gumbel(lp.shape, g)
                    z = tc.gumbel((hi - lo, 1), g)
                    u = -torch.logaddexp(-z, -(lp + e))
                    u.scatter_(1, y[:, None], z)
                    u = u - lp
                    for d, item in enumerate(lqs):
                        if item is None:
                            continue
                        (lq0, lq1), b, sel = item
                        d0, d1 = torch.argmax(lq0 + u, -1), torch.argmax(lq1 + u, -1)
                        A[0, d, (b - B0)[sel], k] = (d0 == y)[sel]
                        A[1, d, (b - B0)[sel], k] = (d1 == y)[sel]
                        diff[d, 0] += (d0 != d1)[sel].sum().double()
                        diff[d, 1] += sel.sum().double()
            for h_ in range(2):
                Cc = torch.cumprod(A[h_].float(), 0)
                num = Cc.sum((1, 2)).double()
                den = torch.cat([torch.tensor([nb * K], device="cuda", dtype=torch.float64), num[:-1]])
                st[h_, :, 0] += num
                st[h_, :, 1] += den
            del hm_, mine, eng, A
    dist.all_reduce(st)
    dist.all_reduce(diff)
    dist.all_reduce(hmax, op=dist.ReduceOp.MAX)
    for d in range(Dp):
        log(f"[parity] depth {d + 1}: keyed conditional alpha train-Head {st[0, d, 0] / st[0, d, 1]:.4f} | engine "
            f"MTPHead {st[1, d, 0] / st[1, d, 1]:.4f} | keyed drafts differing {diff[d, 0] / diff[d, 1]:.5f} "
            f"of {int(diff[d, 1])} | max|hf diff| {hmax[d]:.4f}")
    dist.barrier()
    log("done")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
