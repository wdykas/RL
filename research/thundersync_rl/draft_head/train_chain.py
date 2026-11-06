"""XS-SD multi-depth drafter for Nemotron-3-Nano: the same 1-layer MCore MTP head ("*-", 10 tensors, 47.8M params,
tree_mtp.MTPHead names) trained UNROLLED over depths 1..D the way tree_spec chains it (the engine-side contract):

  depth 1 at base b : inputs (h_b = Nano decoder final_norm output, x_{b+1}); K/V written at position b
  depth k >= 2      : inputs (hf_{k-1} = previous depth's output AFTER the head's final_layernorm, x_{b+k});
                      K/V written at position b+k-1; attention reads true K/V at 0..b + its own chained K/V b+1..b+k-1
  every depth       : Nano's frozen output layer on hf_k -> q for x_{b+k+1}; target p = Nano softmax at b+k.

Tokens are teacher-forced (Nano's own T=1 samples): under keyed coupling depth k's draft only matters when depth k-1
was accepted, i.e. when its drafted token == the realised one, so the teacher-forced token IS the engine's input
on every chain that reaches depth k. The hidden is NOT teacher-forced: depth k >= 2 sees the head's own hf (full
backprop through the chain, EAGLE-3 "training-time test"). Loss = sum_k w_k * mean KL(p || q_k), full vocabulary.

The head here is a plain PyTorch re-implementation of tree_mtp.MTPHead's arithmetic (bf16 GEMMs, fp32 RMSNorm with
bf16 out, MCore grouped QKV, no RoPE, squared ReLU); --mode smoke checks it against tree_mtp.MTPHead.forward_rows
(the engine code) on a real row. Nano runs through th.build/th.Runner (train_head.py, the depth-1 trainer).

Eval (keyed, train_head.py's Gumbel rule): per target position t, K posterior Gumbel keys given the realised y_t;
the depth-k draft from base b = t-k-1 is d = argmax(log q_k + u_t - log p_t). Chain indicator C_k(b) = prod_{s<=k}
1[d_s == y_{b+s+1}] per key; per-insertion acceptance alpha_k = sum C_k / sum C_{k-1} (C_0 = 1), over bases
b in [P-1, N-1-D-... ] common to all depths. Also teacher-forced unconditional rate per depth.

  torchrun --nproc-per-node 4 train_chain.py --mode {smoke,train,eval} --shim <trained depth-1 shim> --init <depth-1 head_mcore.pt> --out <dir>
"""
import argparse
import importlib.util
import math
import os
import pickle
import sys
import time
import zlib

import torch
import torch.nn.functional as F

H2 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H2)
import train_head as th  # noqa: E402

import torch.distributed as dist  # noqa: E402

RANK, LOCAL, WORLD, log = th.RANK, th.LOCAL, th.WORLD, th.log
KEYS = ["mtp.layers.0.enorm.weight", "mtp.layers.0.hnorm.weight", "mtp.layers.0.eh_proj.weight",
        "mtp.layers.0.mtp_model_layer.layers.0.self_attention.linear_qkv.layer_norm_weight",
        "mtp.layers.0.mtp_model_layer.layers.0.self_attention.linear_qkv.weight",
        "mtp.layers.0.mtp_model_layer.layers.0.self_attention.linear_proj.weight",
        "mtp.layers.0.mtp_model_layer.layers.1.mlp.linear_fc1.layer_norm_weight",
        "mtp.layers.0.mtp_model_layer.layers.1.mlp.linear_fc1.weight",
        "mtp.layers.0.mtp_model_layer.layers.1.mlp.linear_fc2.weight",
        "mtp.layers.0.final_layernorm.weight"]
SHORT = ["enorm", "hnorm", "eh", "ln_qkv", "w_qkv", "w_proj", "ln_fc1", "w_fc1", "w_fc2", "final"]
BF = torch.bfloat16


class Head(torch.nn.Module):
    """tree_mtp.MTPHead arithmetic, parallel over bases, fp32 master params (bf16 compute)."""

    def __init__(self, sd, nh=32, nkv=2, hd=128, eps=1e-5):
        super().__init__()
        for k, s in zip(KEYS, SHORT):
            setattr(self, s, torch.nn.Parameter(sd[k].float().cuda()))
        self.nh, self.nkv, self.hd, self.gq, self.eps = nh, nkv, hd, nh // nkv, eps

    def rms(self, x, w):
        v = x.float()
        return (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.eps) * w.to(BF).float()).to(BF)

    def mm(self, x, w):
        return x @ w.to(BF).t()

    def inp(self, h, e):
        """u = eh_proj([enorm(e), hnorm(h)]), q/k/v (MCore grouped layout)."""
        u = self.mm(torch.cat([self.rms(e, self.enorm), self.rms(h, self.hnorm)], -1), self.eh)
        x = self.mm(self.rms(u, self.ln_qkv), self.w_qkv).view(-1, self.nkv, self.gq + 2, self.hd)
        q = x[:, :, :self.gq].reshape(-1, self.nh, self.hd)
        return u, q, x[:, :, self.gq], x[:, :, self.gq + 1]

    def out(self, u, o):
        u = u + self.mm(o.reshape(u.shape[0], -1), self.w_proj)
        z = torch.relu(self.mm(self.rms(u, self.ln_fc1), self.w_fc1))
        u = u + self.mm(z * z, self.w_fc2)
        return self.rms(u, self.final)

    def chain(self, h, E, x, B0, depth):
        """h [N, hid] Nano final-norm hidden (bf16), E = emb (frozen), x [N] tokens. Bases b in [0, N-2] for depth 1
        (its K/V are every committed position), [B0, N-2] for depths >= 2. Returns [hf_1 (len N-1), hf_2.., hf_D
        (len N-1-B0)], hf_k[i] = depth-k output hidden of base (i for k = 1, B0 + i otherwise)."""
        N = x.numel()
        M = N - 1
        u1, q1, k1, v1 = self.inp(h[:M], E[x[1:]])
        o1 = self.attend(q1, [k1], [v1], M, 0)
        outs = [self.out(u1, o1)]
        ks, vs = [k1], [v1]
        base = torch.arange(B0, M, device=x.device)
        for d in range(2, depth + 1):
            tok = x[(base + d).clamp(max=N - 1)]
            uk, qk, kk, vk = self.inp(outs[-1][B0:] if d == 2 else outs[-1], E[tok])
            ks.append(kk)
            vs.append(vk)
            ok = self.attend(qk, ks, vs, M, B0)
            outs.append(self.out(uk, ok))
        return outs

    def attend(self, q, ks, vs, M, B0):
        """Query i (base b = B0 + i, or i for depth 1) attends K1[0..b] and chained K_s[i] (s >= 2, same base)."""
        n = q.shape[0]
        b = torch.arange(n, device=q.device) + B0
        K = torch.cat(ks, 0)  # [M + (d-1) n, nkv, hd]
        V = torch.cat(vs, 0)
        j = torch.arange(M, device=q.device)
        mask = [j[None, :] <= b[:, None]]
        for _ in ks[1:]:
            mask.append(torch.eye(n, dtype=torch.bool, device=q.device))
        mask = torch.cat(mask, 1)
        Kx = K.repeat_interleave(self.gq, dim=1).transpose(0, 1)[None]  # [1, nh, L, hd]
        Vx = V.repeat_interleave(self.gq, dim=1).transpose(0, 1)[None]
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):  # never the math fallback ([heads, n, L] scores)
            o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], Kx, Vx, attn_mask=mask[None, None])
        return o[0].transpose(0, 1)  # [n, nh, hd]

    def state(self):
        return {k: getattr(self, s).detach().to(BF).cpu().contiguous() for k, s in zip(KEYS, SHORT)}


def nano(ns):
    bridge, model, m = th.build(ns.shim, ns.crossfade)
    for p in m.parameters():
        p.requires_grad_(False)
    R = th.Runner(model, m)
    E = m.embedding.word_embeddings.weight.detach()
    eps = getattr(m.config, "layernorm_epsilon", 1e-5)
    nh, nkv = m.config.num_attention_heads, m.config.num_query_groups
    hd = m.config.kv_channels
    log("nano cfg eps", eps, "heads", nh, nkv, hd, "emb", tuple(E.shape), E.dtype)
    sd = torch.load(ns.init, map_location="cpu")
    sd = sd.get("state_dict", sd)
    head = Head(sd, nh, nkv, hd, eps)
    return R, E, head


def hidden(R, x):
    with torch.no_grad():
        hm_, _ = R.forward(x, train=False)
    return hm_.to(BF)


def load(path):
    return pickle.load(open(path, "rb"))


def row_tensor(d, r):
    prompt = list(d["prompts"][r["idx"]])
    return torch.tensor(prompt + list(r["tokens"]), dtype=torch.long, device="cuda"), len(prompt)


def kl_grads(hf, hm_, Wf, d, B0, N, P, C, scale, stats):
    """KL(p || q) of depth d over its valid bases; returns dL/dhf (fp32, same rows as hf). Base b (row index b - off)
    predicts x_{b+d+1} with target p from h_{b+d}; valid for b >= P-1 and b+d+1 <= N-1."""
    off = 0 if d == 1 else B0
    lo, hi = max(P - 1, off), N - 2 - d + 1  # b in [lo, hi)
    g = torch.zeros(hf.shape, dtype=torch.float32, device="cuda")
    nt = max(hi - lo, 0)
    for a in range(lo, hi, C):
        b = min(hi, a + C)
        with torch.no_grad():
            lp = torch.log_softmax(hm_[a + d:b + d].float() @ Wf.t(), -1)
            lq = torch.log_softmax(hf[a - off:b - off].detach().float() @ Wf.t(), -1)
            p, q = lp.exp(), lq.exp()
            stats[d - 1, 0] += (p * (lp - lq)).sum().double()
            stats[d - 1, 1] += torch.minimum(p, q).sum().double()
            stats[d - 1, 2] += (lq.argmax(-1) == lp.argmax(-1)).sum().double()
            g[a - off:b - off] = ((q - p) @ Wf) * scale
            del lp, lq, p, q
    stats[d - 1, 3] += nt
    return g


def train(ns, R, E, head):
    tr = load(ns.train or th.data_path("train"))
    rows = [r for r in tr["rows"]][RANK::WORLD]
    params = list(head.parameters())
    opt = torch.optim.AdamW(params, lr=ns.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=ns.wd)
    total = int(min(ns.steps or 10**9, (len(rows) * ns.epochs) // ns.accum))
    n = torch.tensor([total], device="cuda")
    dist.all_reduce(n, op=dist.ReduceOp.MIN)
    total = int(n.item())
    W = [float(v) for v in ns.depth_w.split(",")]
    log(f"train rows/rank {len(rows)} steps {total} rows/step {ns.accum * WORLD} lr {ns.lr} depth {ns.depth} w {W}")

    def lr_at(s):
        if s < ns.warmup:
            return ns.lr * (s + 1) / ns.warmup
        f = (s - ns.warmup) / max(1, total - ns.warmup)
        if ns.time_budget:  # cosine over whichever runs out first: steps or the wall budget (rank-0 clock, broadcast)
            f = max(f, tfrac[0])
        return ns.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(f, 1.0))))

    tfrac = [0.0]
    t0, ri, step = time.time(), 0, 0
    Wf = R.Wf
    while step < total:
        stats = torch.zeros(ns.depth, 4, device="cuda", dtype=torch.float64)
        for _ in range(ns.accum):
            r = rows[ri % len(rows)]
            ri += 1
            x, P = row_tensor(tr, r)
            if ns.maxlen and x.numel() > ns.maxlen:
                x = x[:ns.maxlen]
            N = x.numel()
            B0 = max(P - 1, 0)
            hm_ = hidden(R, x)
            outs = head.chain(hm_, E, x, B0, ns.depth)
            grads = []
            for d in range(1, ns.depth + 1):
                nt = max(N - 2 - d + 1 - max(P - 1, 0 if d == 1 else B0), 1)
                grads.append(kl_grads(outs[d - 1], hm_, Wf, d, B0, N, P, ns.chunk,
                                      W[d - 1] / (nt * ns.accum * WORLD), stats).to(BF))
            torch.autograd.backward(outs, grads)
            del outs, grads, hm_
        for p_ in params:
            if p_.grad is not None:
                dist.all_reduce(p_.grad)
        gn = torch.sqrt(sum((p_.grad.float() ** 2).sum() for p_ in params if p_.grad is not None))
        torch.nn.utils.clip_grad_norm_(params, ns.clip)
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step)
        opt.step()
        opt.zero_grad(set_to_none=True)
        dist.all_reduce(stats)
        if step % ns.log_every == 0 or step == total - 1:
            s = " | ".join(f"d{d + 1} kl {stats[d, 0] / stats[d, 3]:.3f} 1-TV {stats[d, 1] / stats[d, 3]:.3f} "
                           f"top1 {stats[d, 2] / stats[d, 3]:.3f}" for d in range(ns.depth))
            log(f"step {step}/{total} {s} | gn {gn.item():.3f} lr {lr_at(step):.2e} rows {ri * WORLD} "
                f"{time.time() - t0:.0f}s")
        step += 1
        if ns.eval_every and step % ns.eval_every == 0 and step < total:
            save(ns, head, f"head_step{step}.pt" if ns.keep_mid else "head_mcore.pt")
            evaluate(ns, R, E, head, limit=ns.eval_quick, tag=f"quick{step}")
        flag = torch.tensor([(time.time() - t0) / ns.time_budget if ns.time_budget else 0.0], device="cuda")
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        tfrac[0] = flag.item()
        if tfrac[0] >= 1.0:
            log(f"time budget hit at step {step}")
            break
    save(ns, head, "head_mcore.pt")


def save(ns, head, name):
    if RANK == 0:
        os.makedirs(ns.out, exist_ok=True)
        st = head.state()
        torch.save(st, os.path.join(ns.out, name))
        with open(os.path.join(ns.out, "head_names.txt"), "w") as f:
            f.write("# tree_mtp.MTPHead / Megatron-Core names (HybridModel mtp_num_layers=1 '*-', TP1), bf16\n")
            for k, v in st.items():
                f.write(f"{k}\t{tuple(v.shape)}\t{v.dtype}\n")
        log("saved", name, sum(v.numel() for v in st.values()), "params")
    dist.barrier()


def gumbel(shape, g):
    return -torch.log(-torch.log(torch.rand(shape, device="cuda", generator=g).clamp_min(1e-30)))


def evaluate(ns, R, E, head, limit=0, tag="eval"):
    ho = load(ns.heldout)
    allrows = sorted(ho["rows"], key=lambda r: (r["idx"], r["j"], r["run"]))
    if limit:
        allrows = sorted(allrows, key=lambda r: (r["j"], r["idx"], r["run"]))[:limit]
    rows = allrows[RANK::WORLD]
    K, C, Dp = ns.keys, ns.chunk, ns.depth
    Wf = R.Wf
    res, t0 = [], time.time()
    for ri, r in enumerate(rows):
        x, P = row_tensor(ho, r)
        N = x.numel()
        B0 = max(P - 1, 0)
        with torch.no_grad():
            hm_ = hidden(R, x)
            outs = head.chain(hm_, E, x, B0, Dp)
            # common bases b in [P-1, N-2-Dp] (every depth's target x_{b+d+1} exists)
            nb = N - 1 - Dp - B0
            if nb <= 0:
                continue
            A = torch.zeros(Dp, nb, K, dtype=torch.bool, device="cuda")
            gr = torch.zeros(Dp, device="cuda", dtype=torch.float64)
            g = torch.Generator(device="cuda")
            # targets t = b + d + 1 cover [B0 + 2, N - 1]
            for lo in range(B0 + 2, N, C):
                hi = min(N, lo + C)
                t = torch.arange(lo, hi, device="cuda")
                y = x[lo:hi]
                lp = torch.log_softmax(hm_[lo - 1:hi - 1].float() @ Wf.t(), -1)
                lqs = []
                for d in range(1, Dp + 1):
                    b = t - d - 1
                    sel = (b >= B0) & (b < B0 + nb)
                    off = 0 if d == 1 else B0
                    lq = torch.full_like(lp[:1], 0.0).expand(hi - lo, -1)
                    if sel.any():
                        rowsel = (b - off).clamp(0, outs[d - 1].shape[0] - 1)
                        lq = torch.log_softmax(outs[d - 1][rowsel].float() @ Wf.t(), -1)
                    lqs.append((lq, b, sel))
                    gr[d - 1] += ((lq.argmax(-1) == y) & sel).sum().double()
                g.manual_seed((zlib.crc32(f"{r['idx']}-{r['j']}-{r['run']}-{lo}".encode()) * 2654435761) % (2**62))
                for k in range(K):
                    e = gumbel(lp.shape, g)
                    z = gumbel((hi - lo, 1), g)
                    u = -torch.logaddexp(-z, -(lp + e))
                    u.scatter_(1, y[:, None], z)
                    u = u - lp
                    for d, (lq, b, sel) in enumerate(lqs):
                        if sel.any():
                            dd = torch.argmax(lq + u, -1)
                            A[d, (b - B0)[sel], k] = (dd == y)[sel]
                    del e, u
                del lp, lqs
            Cc = torch.cumprod(A.float(), 0)  # chain indicator per depth
            num = Cc.sum((1, 2)).double()
            den = torch.cat([torch.tensor([nb * K], device="cuda", dtype=torch.float64), num[:-1]])
            tf = A.float().mean((1, 2)).double()
        rec = dict(idx=r["idx"], j=r["j"], run=r["run"], P=P, L=N - P, nb=nb, num=num.tolist(), den=den.tolist(),
                   tf=tf.tolist(), greedy=(gr / nb).tolist())
        res.append(rec)
        if ri % 32 == 0:
            al = [n_ / max(d_, 1e-9) for n_, d_ in zip(rec["num"], rec["den"])]
            log(f"[{tag}] row {ri}/{len(rows)} L {N - P} alpha " + " ".join(f"{a:.3f}" for a in al) +
                f" tf " + " ".join(f"{a:.3f}" for a in rec["tf"]) + f" {time.time() - t0:.0f}s")
        del hm_, outs
    out = [None] * WORLD
    dist.all_gather_object(out, res)
    if RANK == 0:
        allres = [x_ for o in out for x_ in o]
        os.makedirs(ns.out, exist_ok=True)
        pickle.dump(allres, open(os.path.join(ns.out, f"{tag}.pkl"), "wb"))
        summarize(allres, tag, Dp)
    dist.barrier()


def summarize(res, tag, Dp):
    def q(v, f):
        v = sorted(v)
        return v[min(len(v) - 1, int(f * len(v)))]

    for d in range(Dp):
        num = sum(r["num"][d] for r in res)
        den = sum(r["den"][d] for r in res)
        per = [r["num"][d] / max(r["den"][d], 1e-9) for r in res if r["den"][d] > 0]
        worst = min((r for r in res if r["den"][d] > 0), key=lambda r: r["num"][d] / r["den"][d])
        tfm = sum(r["tf"][d] * r["nb"] for r in res) / sum(r["nb"] for r in res)
        grm = sum(r["greedy"][d] * r["nb"] for r in res) / sum(r["nb"] for r in res)
        log(f"[{tag}] depth {d + 1}: per-insertion keyed alpha (pooled) {num / den:.4f} | per-row mean "
            f"{sum(per) / len(per):.4f} p10 {q(per, 0.10):.4f} min {worst['num'][d] / worst['den'][d]:.4f} "
            f"(idx {worst['idx']} j {worst['j']} L {worst['L']}) | teacher-forced keyed {tfm:.4f} greedy {grm:.4f} "
            f"| rows {len(per)}")


def smoke(ns, R, E, head):
    """(1) parity vs tree_mtp.MTPHead (engine code) on one held-out row: prefill true K/V 0..b, chain depths 1..D
    with teacher-forced tokens; (2) 2 train steps on short rows; (3) eval on 8 rows."""
    tm = th.load_tree_mtp(ns.tree_mtp)
    ho = load(ns.heldout)
    r = sorted(ho["rows"], key=lambda r: (r["idx"], r["j"]))[RANK]
    x, P = row_tensor(ho, r)
    x = x[:min(x.numel(), P + 600)]
    N = x.numel()
    B0 = max(P - 1, 0)
    hm_ = hidden(R, x)
    path = os.path.join(ns.out, "smoke_head.pt")
    if RANK == 0:
        torch.save(head.state(), path)
    dist.barrier()
    mh = tm.MTPHead(path, E, R.W, slots=2, max_seq=N + 8, num_heads=head.nh, num_kv_heads=head.nkv,
                    head_dim=head.hd, eps=head.eps)
    with torch.no_grad():
        outs = head.chain(hm_, E, x, B0, ns.depth)
        worst = [0.0] * ns.depth
        agree = [0, 0]
        for b in range(B0, N - 1 - ns.depth, 97):
            one = lambda v: torch.tensor([v], device="cuda")  # noqa: E731
            mh.kc.zero_()
            mh.vc.zero_()
            mh.append_kv(torch.zeros(b, dtype=torch.long, device="cuda"), torch.arange(b, device="cuda"),
                         hm_[:b], x[1:b + 1])
            u, tok = hm_[b:b + 1], x[b + 1:b + 2]
            for d in range(1, ns.depth + 1):
                u, lg = mh.forward_rows(one(0), one(b + d - 1), u, tok)
                mine = outs[d - 1][b - (0 if d == 1 else B0)]
                worst[d - 1] = max(worst[d - 1], (u[0].float() - mine.float()).abs().max().item())
                agree[0] += int(lg.argmax(-1)[0] == (mine.float() @ R.Wf.t()).argmax())
                agree[1] += 1
                tok = x[b + d + 1:b + d + 2]
        print(f"[smoke rank {RANK}] parity vs tree_mtp.MTPHead: max|hf diff| per depth {worst}; argmax agree "
              f"{agree[0]}/{agree[1]}", flush=True)
    # backend check + train-step timing on a long row
    tr = load(ns.train or th.data_path("train"))
    rr = sorted(tr["rows"], key=lambda r: -len(r["tokens"]))[RANK]
    x, P = row_tensor(tr, rr)
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.cuda.synchronize()
    t0 = time.time()
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        hm_ = hidden(R, x)
        torch.cuda.synchronize()
        t1 = time.time()
        outs = head.chain(hm_, E, x, max(P - 1, 0), ns.depth)
        st = torch.zeros(ns.depth, 4, device="cuda", dtype=torch.float64)
        gr = [kl_grads(outs[d - 1], hm_, R.Wf, d, max(P - 1, 0), x.numel(), P, ns.chunk, 1e-4, st).to(BF)
              for d in range(1, ns.depth + 1)]
        torch.autograd.backward(outs, gr)
    torch.cuda.synchronize()
    print(f"[smoke rank {RANK}] long row N {x.numel()}: nano fwd {t1 - t0:.2f}s head fwd+bwd {time.time() - t1:.2f}s "
          f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB; kl d1..: "
          f"{[round((st[d, 0] / st[d, 3]).item(), 3) for d in range(ns.depth)]}", flush=True)
    head.zero_grad(set_to_none=True)
    dist.barrier()
    evaluate(ns, R, E, head, limit=8, tag="smoke_eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "train", "eval"], required=True)
    ap.add_argument("--shim", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout", default="", help="held-out pickle (default $DRAFT_HEAD_DATA/heldout.pkl)")
    ap.add_argument("--train", default="", help="train pickle (default $DRAFT_HEAD_DATA/train.pkl)")
    ap.add_argument("--tree_mtp", default="", help="tree_mtp.py file (default: the installed nemo_rl module)")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--depth_w", default="1,1,1")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--maxlen", type=int, default=0)
    ap.add_argument("--keys", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=0)
    ap.add_argument("--eval_quick", type=int, default=64)
    ap.add_argument("--eval_limit", type=int, default=0)
    ap.add_argument("--keep_mid", type=int, default=0)
    ap.add_argument("--time_budget", type=float, default=0)
    ap.add_argument("--crossfade", type=int, default=1)
    ns = ap.parse_args()
    ns.heldout = ns.heldout or th.data_path("heldout")
    os.makedirs(ns.out, exist_ok=True)
    R, E, head = nano(ns)
    if ns.mode == "smoke":
        smoke(ns, R, E, head)
    elif ns.mode == "train":
        evaluate(ns, R, E, head, limit=ns.eval_quick, tag="quick0") if ns.eval_quick else None
        train(ns, R, E, head)
        evaluate(ns, R, E, head, limit=ns.eval_limit, tag="heldout")
    else:
        evaluate(ns, R, E, head, limit=ns.eval_limit, tag="heldout")
    dist.barrier()
    log("done")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
