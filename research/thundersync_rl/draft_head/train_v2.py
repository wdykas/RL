"""Draft head v2 for Nano: a stronger drafter for the slow, high-entropy rows. Same chained contract as
train_chain.py (depth 1: Nano h_b + emb x_{b+1}; depth k >= 2: the head's own post-final-norm hf_{k-1}
+ emb x_{b+k}; K/V of depth k at b+k-1). Three structural levers:

  --ffn F        wider MLP (squared ReLU): fc1 [F, H], fc2 [H, F]. Init: c3's 1856 rows/cols copied, extra fc1 rows
                 random at c3's row scale, extra fc2 columns ZERO, so the v2 head starts exactly at c3.
                 tree_mtp.MTPHead reads the shapes from the weights: a drop-in for the engine (no code change).
  --feats L,..   extra Nano residual-stream features (outputs of decoder layers L, ...) at the base position, each
                 RMS-normed (own weight) and concatenated into eh_proj (extra columns ZERO-init). Depth k >= 2 reuses
                 the base position's features (fixed for the chain). Engine needs the extra hidden capture (not built).
  --ent_gamma g  per-position loss weight w = min((H(p_t) / Hbar)^g, wmax) (H the policy entropy, Hbar an EMA):
                 emphasis on the hard positions that make the slow rows slow.
  --tv_beta b    adds b * TV(p, q) = b * (1 - sum min(p, q)) to KL(p || q) (TV bounds the coupled acceptance).

Checkpoint: MCore/tree_mtp names; extra feature norms as mtp.layers.0.fnorm{i}.weight + meta "draft_head2.feats".
  torchrun --nproc-per-node 4 train_v2.py --mode {train,eval} --shim <shim> --init <head .pt> --out <dir> [...]
"""
import argparse
import math
import os
import sys
import time

import torch
import torch.distributed as dist

H2 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H2)
import eval_walk as ew  # noqa: E402
import train_chain as tc  # noqa: E402

RANK, WORLD, log, BF = tc.RANK, tc.WORLD, tc.log, tc.BF
FC1 = "mtp.layers.0.mtp_model_layer.layers.1.mlp.linear_fc1.weight"
FC2 = "mtp.layers.0.mtp_model_layer.layers.1.mlp.linear_fc2.weight"
EH = "mtp.layers.0.eh_proj.weight"


class Head2(tc.Head):
    def __init__(self, sd, R, nh, nkv, hd, eps, ffn=0, feats=()):
        sd = dict(sd)
        Hd = sd[EH].shape[0]
        old_feats = [int(v) for v in sd.get("draft_head2.feats", [])]
        if ffn and sd[FC1].shape[0] < ffn:  # widen: start == the given head
            f0 = sd[FC1].shape[0]
            g = torch.Generator().manual_seed(1)
            sc = sd[FC1].float().norm(dim=1).mean() / math.sqrt(Hd)
            extra = torch.randn(ffn - f0, Hd, generator=g) * sc
            sd[FC1] = torch.cat([sd[FC1].float(), extra], 0)
            sd[FC2] = torch.cat([sd[FC2].float(), torch.zeros(Hd, ffn - f0)], 1)
        feats = [int(v) for v in feats] if feats else old_feats
        assert feats[:len(old_feats)] == old_feats, (feats, old_feats)
        nf_have = len(old_feats)
        if len(feats) > nf_have:  # new feature columns: zero-init (start == the given head)
            sd[EH] = torch.cat([sd[EH].float(), torch.zeros(Hd, Hd * (len(feats) - nf_have))], 1)
        super().__init__({k: v for k, v in sd.items() if k in tc.KEYS}, nh, nkv, hd, eps)
        self.feats = feats
        for i in range(len(feats)):
            w = sd.get(f"mtp.layers.0.fnorm{i}.weight", torch.ones(Hd))
            setattr(self, f"fnorm{i}", torch.nn.Parameter(w.float().cuda()))
        self.R = R

    def inp(self, h, e, f=None):
        parts = [self.rms(e, self.enorm), self.rms(h, self.hnorm)]
        if self.feats:
            parts += [self.rms(f[i], getattr(self, f"fnorm{i}")) for i in range(len(self.feats))]
        u = self.mm(torch.cat(parts, -1), self.eh)
        x = self.mm(self.rms(u, self.ln_qkv), self.w_qkv).view(-1, self.nkv, self.gq + 2, self.hd)
        q = x[:, :, :self.gq].reshape(-1, self.nh, self.hd)
        return u, q, x[:, :, self.gq], x[:, :, self.gq + 1]

    def chain(self, h, E, x, B0, depth):
        """train_chain.Head.chain with the base position's features (self.R.feats of the last Nano forward)."""
        F = self.R.feats if self.feats else None
        N = x.numel()
        M = N - 1
        u1, q1, k1, v1 = self.inp(h[:M], E[x[1:]], [f[:M] for f in F] if F else None)
        o1 = self.attend(q1, [k1], [v1], M, 0)
        outs = [self.out(u1, o1)]
        ks, vs = [k1], [v1]
        base = torch.arange(B0, M, device=x.device)
        Fb = [f[B0:M] for f in F] if F else None
        for d in range(2, depth + 1):
            tok = x[(base + d).clamp(max=N - 1)]
            uk, qk, kk, vk = self.inp(outs[-1][B0:] if d == 2 else outs[-1], E[tok], Fb)
            ks.append(kk)
            vs.append(vk)
            ok = self.attend(qk, ks, vs, M, B0)
            outs.append(self.out(uk, ok))
        return outs

    def state(self):
        st = super().state()
        for i in range(len(self.feats)):
            st[f"mtp.layers.0.fnorm{i}.weight"] = getattr(self, f"fnorm{i}").detach().to(BF).cpu().contiguous()
        if self.feats:
            st["draft_head2.feats"] = torch.tensor(self.feats, dtype=torch.int32)
        return st


def build(ns):
    R, E, _h = tc.nano(ns)
    R.feats = []
    if ns.feats or ns.mode == "eval":
        sd0 = torch.load(ns.init, map_location="cpu")
        sd0 = sd0.get("state_dict", sd0)
        fl = [int(v) for v in ns.feats.split(",")] if ns.feats else [int(v) for v in sd0.get("draft_head2.feats", [])]
    else:
        fl, sd0 = [], None
    layers = R.m.decoder.layers
    cap = {}
    for li in fl:
        def hook(_m, _i, out, li=li):
            o = out[0] if isinstance(out, tuple) else out
            cap[li] = o.reshape(-1, o.shape[-1])
        layers[li].register_forward_hook(hook)

    def feats_after_forward():
        R.feats = [cap[li].detach().to(BF) for li in fl]

    sd = sd0 if sd0 is not None else torch.load(ns.init, map_location="cpu")
    sd = sd.get("state_dict", sd)
    m = R.m
    head = Head2(sd, R, m.config.num_attention_heads, m.config.num_query_groups, m.config.kv_channels,
                 getattr(m.config, "layernorm_epsilon", 1e-5), ffn=ns.ffn, feats=fl)
    log("v2 head: ffn", head.w_fc1.shape[0], "feats", fl, "eh", tuple(head.eh.shape), "params",
        sum(p.numel() for p in head.parameters()))
    orig_hidden = tc.hidden

    def hidden(R_, x):
        hm_ = orig_hidden(R_, x)
        if fl:
            feats_after_forward()
            assert R.feats[0].shape[0] == x.numel(), (R.feats[0].shape, x.numel())
        return hm_
    tc.hidden = hidden  # eval_walk / train loop both call tc.hidden right before head.chain
    return R, E, head


def grads(hf, hm_, Wf, d, B0, N, P, C, scale, stats, ns, hbar):
    """dL/dhf for depth d: w_t * [(q - p) + beta * q (g - <q, g>)], g = -1[q < p] (TV), w_t entropy weight."""
    off = 0 if d == 1 else B0
    lo, hi = max(P - 1, off), N - 2 - d + 1
    g = torch.zeros(hf.shape, dtype=torch.float32, device="cuda")
    for a in range(lo, hi, C):
        b = min(hi, a + C)
        with torch.no_grad():
            lp = torch.log_softmax(hm_[a + d:b + d].float() @ Wf.t(), -1)
            lq = torch.log_softmax(hf[a - off:b - off].detach().float() @ Wf.t(), -1)
            p, q = lp.exp(), lq.exp()
            ent = -(p * lp).sum(-1)
            stats[d - 1, 0] += (p * (lp - lq)).sum().double()
            stats[d - 1, 1] += torch.minimum(p, q).sum().double()
            stats[d - 1, 2] += (lq.argmax(-1) == lp.argmax(-1)).sum().double()
            stats[d - 1, 3] += b - a
            if d == 1:
                hbar[1] += ent.sum().double()
                hbar[2] += b - a
            gz = q - p
            if ns.tv_beta:
                gt = -(q < p).float()
                gz = gz + ns.tv_beta * q * (gt - (q * gt).sum(-1, keepdim=True))
            if ns.ent_gamma:
                w = (ent / hbar[0].float().clamp_min(1e-3)).clamp_min(1e-3).pow(ns.ent_gamma).clamp(max=ns.wmax)
                gz = gz * w[:, None]
            g[a - off:b - off] = (gz @ Wf) * scale
            del lp, lq, p, q, gz
    return g


def train(ns, R, E, head):
    tr = dict(prompts={}, rows=[])
    for path in ns.train.split(","):
        d_ = tc.load(path)
        tr["prompts"].update(d_["prompts"])
        tr["rows"] += d_["rows"]
    import random as _r
    _r.Random(0).shuffle(tr["rows"])
    rows = tr["rows"][RANK::WORLD]
    params = list(head.parameters())
    opt = torch.optim.AdamW(params, lr=ns.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    total = int((len(rows) * ns.epochs) // ns.accum)
    n = torch.tensor([total], device="cuda")
    dist.all_reduce(n, op=dist.ReduceOp.MIN)
    total = int(n.item())
    W = [float(v) for v in ns.depth_w.split(",")]
    hbar = torch.tensor([ns.hbar0, 0.0, 0.0], dtype=torch.float64, device="cuda")
    log(f"train rows/rank {len(rows)} steps {total} lr {ns.lr} depth {ns.depth} w {W} ffn {ns.ffn} feats {head.feats} "
        f"ent_gamma {ns.ent_gamma} wmax {ns.wmax} tv_beta {ns.tv_beta}")
    tfrac = [0.0]

    def lr_at(s):
        if s < ns.warmup:
            return ns.lr * (s + 1) / ns.warmup
        f = max((s - ns.warmup) / max(1, total - ns.warmup), tfrac[0])
        return ns.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(f, 1.0))))

    t0, ri, step = time.time(), 0, 0
    while step < total:
        stats = torch.zeros(ns.depth, 4, device="cuda", dtype=torch.float64)
        for _ in range(ns.accum):
            r = rows[ri % len(rows)]
            ri += 1
            x, P = tc.row_tensor(tr, r)
            N = x.numel()
            B0 = max(P - 1, 0)
            hm_ = tc.hidden(R, x)
            outs = head.chain(hm_, E, x, B0, ns.depth)
            gl = []
            for d in range(1, ns.depth + 1):
                nt = max(N - 2 - d + 1 - max(P - 1, 0 if d == 1 else B0), 1)
                gl.append(grads(outs[d - 1], hm_, R.Wf, d, B0, N, P, ns.chunk, W[d - 1] / (nt * ns.accum * WORLD),
                                stats, ns, hbar).to(BF))
            torch.autograd.backward(outs, gl)
            del outs, gl, hm_
        for p_ in params:
            if p_.grad is not None:
                dist.all_reduce(p_.grad)
        gn = torch.sqrt(sum((p_.grad.float() ** 2).sum() for p_ in params if p_.grad is not None))
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step)
        opt.step()
        opt.zero_grad(set_to_none=True)
        dist.all_reduce(stats)
        hs = hbar[1:].clone()
        dist.all_reduce(hs)
        hbar[0] = 0.98 * hbar[0] + 0.02 * (hs[0] / hs[1].clamp_min(1)) if step else hs[0] / hs[1].clamp_min(1)
        hbar[1:] = 0
        if step % ns.log_every == 0 or step == total - 1:
            s = " | ".join(f"d{d + 1} kl {stats[d, 0] / stats[d, 3]:.3f} 1-TV {stats[d, 1] / stats[d, 3]:.3f} "
                           f"top1 {stats[d, 2] / stats[d, 3]:.3f}" for d in range(ns.depth))
            log(f"step {step}/{total} {s} | gn {gn.item():.3f} lr {lr_at(step):.2e} Hbar {hbar[0].item():.3f} "
                f"{time.time() - t0:.0f}s")
        step += 1
        if ns.eval_every and step % ns.eval_every == 0:
            save(ns, head)
            quick(ns, R, E, head, f"quick{step}")
        flag = torch.tensor([(time.time() - t0) / ns.time_budget if ns.time_budget else 0.0], device="cuda")
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        tfrac[0] = flag.item()
        if tfrac[0] >= 1.0:
            log(f"time budget hit at step {step}")
            break
    save(ns, head)


def save(ns, head):
    if RANK == 0:
        st = head.state()
        torch.save(st, os.path.join(ns.out, "head.pt"))
        with open(os.path.join(ns.out, "head_names.txt"), "w") as f:
            for k, v in st.items():
                f.write(f"{k}\t{tuple(v.shape)}\t{v.dtype}\n")
        log("saved head.pt", sum(v.numel() for k, v in st.items() if k.startswith("mtp.")), "params")
    dist.barrier()


def quick(ns, R, E, head, tag):
    a = argparse.Namespace(**vars(ns))
    a.limit, a.tag, a.depth = ns.eval_quick, tag, 4
    with torch.no_grad():
        ew.evaluate(a, R, E, head)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval"], required=True)
    ap.add_argument("--shim", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout", default="", help="held-out pickle (default $DRAFT_HEAD_DATA/heldout.pkl)")
    ap.add_argument("--train", default="", help="comma-separated train pickles (default $DRAFT_HEAD_DATA/train.pkl)")
    ap.add_argument("--trainsample", default="", help="optional train-side rows pickle for a train-vs-held-out walk")
    ap.add_argument("--ffn", type=int, default=0)
    ap.add_argument("--feats", default="")
    ap.add_argument("--ent_gamma", type=float, default=0.0)
    ap.add_argument("--wmax", type=float, default=8.0)
    ap.add_argument("--hbar0", type=float, default=1.0)
    ap.add_argument("--tv_beta", type=float, default=0.0)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--depth_w", default="1,1,1")
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--keys", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--eval_every", type=int, default=400)
    ap.add_argument("--eval_quick", type=int, default=64)
    ap.add_argument("--time_budget", type=float, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="walk")
    ap.add_argument("--crossfade", type=int, default=1)
    ap.add_argument("--bi", type=int, default=0)
    ns = ap.parse_args()
    ns.heldout = ns.heldout or tc.th.data_path("heldout")
    ns.train = ns.train or tc.th.data_path("train")
    os.makedirs(ns.out, exist_ok=True)
    if ns.bi:  # BI recipe (acceptance audit's enable_bi): Nano forward bitwise with the engine's
        import audit_a

        audit_a.enable_bi()
        log("BI recipe enabled (audit_a.enable_bi)")
    R, E, head = build(ns)
    if ns.mode == "train":
        quick(ns, R, E, head, "quick0")
        train(ns, R, E, head)
        a = argparse.Namespace(**vars(ns))
        a.depth, a.tag = 4, "walk"
        with torch.no_grad():
            ew.evaluate(a, R, E, head)
            if ns.trainsample:  # train-side rows: train vs held-out walk (overfitting check)
                a.heldout, a.tag, a.limit = ns.trainsample, "trainwalk", 0
                ew.evaluate(a, R, E, head)
    else:
        a = argparse.Namespace(**vars(ns))
        a.depth = 4
        with torch.no_grad():
            ew.evaluate(a, R, E, head)
    dist.barrier()
    log("done")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
