"""XS-SD drafter (c) for Nemotron-3-Nano: train / evaluate a 1-layer Megatron-Core MTP head ("*-") on Nano's own text.

Model: Nano loaded through Megatron-Bridge from the shim dir (mk_shim.py: Nano weights by symlink + config
num_nextn_predict_layers 1, mtp_hybrid_override_pattern "*-" + mtp.safetensors). That is the stock HybridModel with
mtp_num_layers 1: MTP depth 1 at position i reads the decoder output h_i (AFTER Nano's final norm_f, the tensor the
LM head sees) and emb(x_{i+1}); enorm(emb) ++ hnorm(h) -> eh_proj -> attention layer "*" -> MLP layer "-" ->
mtp final_layernorm -> Nano's frozen output_layer -> logits for x_{i+2}.
Base frozen (requires_grad False; config.freeze_base_model_for_mtp runs the backbone under no_grad). TP1 EP1: every
rank holds the full model and its own rows (DP 4), MTP grads all-reduced by hand, fp32 AdamW master weights.

Loss: forward KL(p || q) per target position t (response tokens, t >= P+1): p = Nano softmax at t-1 (teacher, the
engine's sampling distribution at T 1), q = head softmax from base t-2. Full-vocab soft targets, chunked.

Eval (keyed acceptance, posterior-Gumbel rule): per target t, K Gumbel vectors g_t drawn from the posterior given the
realised y_t (argmax(log p_t + g_t) == y_t); keyed draft d = argmax(log q + g_t); alpha_t = mean_k 1[d == y_t].
Also: E_p[q] = sum_v p q (independent sample-and-match acceptance), 1-TV = sum min(p, q) (rejection-sampling
acceptance), greedy = 1[argmax q == y_t], top1 = 1[argmax q == argmax p].

  torchrun --nproc-per-node 4 train_head.py --mode train --shim <dir> --out <dir> [--steps N]
  torchrun --nproc-per-node 4 train_head.py --mode eval  --shim <dir with trained head> --out <dir>
"""
import argparse
import math
import os
import pickle
import sys
import time
import zlib

import torch
import torch.distributed as dist

RANK = int(os.environ.get("RANK", "0"))
LOCAL = int(os.environ.get("LOCAL_RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
H = os.path.dirname(os.path.abspath(__file__))


def log(*a):
    if RANK == 0:
        print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def build(shim, crossfade):
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    torch.cuda.set_device(LOCAL)
    dist.init_process_group("nccl")
    ps.initialize_model_parallel(1, 1, expert_model_parallel_size=1, expert_tensor_parallel_size=1)
    model_parallel_cuda_manual_seed(0)
    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(shim, torch_dtype=torch.bfloat16, trust_remote_code=True)
    prov = bridge.to_megatron_provider(load_weights=True)
    log("provider mtp:", prov.mtp_num_layers, prov.mtp_hybrid_override_pattern, "repeated", prov.mtp_use_repeated_layer,
        "ffn", prov.ffn_hidden_size, "act", getattr(prov.activation_func, "__name__", prov.activation_func))
    assert prov.mtp_num_layers == 1 and prov.mtp_hybrid_override_pattern == "*-"
    prov.tensor_model_parallel_size = 1
    prov.pipeline_model_parallel_size = 1
    prov.expert_model_parallel_size = 1
    prov.expert_tensor_parallel_size = 1
    prov.sequence_parallel = False
    prov.params_dtype = torch.bfloat16
    prov.bf16, prov.fp16 = True, False
    prov.gradient_accumulation_fusion = False
    prov.moe_router_dtype = "fp32"
    prov.moe_permute_fusion = False
    prov.moe_router_fusion = False
    prov.recompute_granularity = None
    prov.freeze_base_model_for_mtp = True
    if crossfade:
        from nemo_rl.models.megatron import crossfade_routing

        crossfade_routing.configure({"megatron_cfg": {"crossfade_routing_m": 0.01, "crossfade_routing_extra": 8}}, prov)
        log("cross-fade:", crossfade_routing.get_state())
    if getattr(prov, "moe_router_num_groups", None) == 1:
        prov.moe_router_num_groups = prov.moe_router_group_topk = None
    prov.finalize()
    model = prov.provide_distributed_model(wrap_with_ddp=False)[0]
    from megatron.core.utils import unwrap_model

    m = unwrap_model(model)
    assert m.mtp_process and len(m.mtp.layers) == 1
    for n_, p in m.named_parameters():
        p.requires_grad_(n_.startswith("mtp."))
    model.eval()
    log(f"model built; mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB; head params",
        sum(p.numel() for n_, p in m.named_parameters() if n_.startswith("mtp.")))
    return bridge, model, m


class Runner:
    def __init__(self, model, m):
        import megatron.core.models.hybrid.hybrid_model as hm

        self.model, self.m, self.cap = model, m, {}

        def _capture(hidden_states=None, main_hidden_states=None, **kw):
            self.cap["h"] = hidden_states
            return main_hidden_states

        hm.process_mtp_loss = _capture
        self.W = m.output_layer.weight.detach()  # [V, h] bf16, frozen (TP1: full vocab)
        self.Wf = self.W.float()

    def forward(self, x, train):
        """Returns main hidden [N, h] (no grad) and MTP hidden [N, h] (grad iff train)."""
        N = x.numel()
        pos = torch.arange(N, device="cuda").unsqueeze(0)
        self.cap.clear()
        self.m.training = train  # top-level flag only: backbone under no_grad, every submodule stays in eval
        with torch.set_grad_enabled(train):
            self.model(input_ids=x.unsqueeze(0), position_ids=pos, attention_mask=None,
                       output_processor=lambda hidden_states=None, **kw: hidden_states)
        self.m.training = False
        hs = self.cap.pop("h")
        assert hs.shape[0] == 2 * N, (hs.shape, N)
        return hs[:N, 0].detach(), hs[N:, 0]


def data_path(name):
    """<DRAFT_HEAD_DATA>/<name>.pkl: the directory prep_data.py wrote (train.pkl, heldout.pkl)."""
    d = os.environ.get("DRAFT_HEAD_DATA", "")
    if not d:
        raise SystemExit("DRAFT_HEAD_DATA is not set: point it at the directory written by prep_data.py "
                         "(train.pkl / heldout.pkl), or pass the pickle paths explicitly.")
    return os.path.join(d, f"{name}.pkl")


def load_tree_mtp(path=""):
    """The engine's MTP head module (nemo_rl.models.generation.megatron.tree_mtp); `path` overrides with a file."""
    if not path:
        import importlib

        return importlib.import_module("nemo_rl.models.generation.megatron.tree_mtp")
    import importlib.util

    spec = importlib.util.spec_from_file_location("tree_mtp", path)
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    return tm


def load_split(name):
    return pickle.load(open(data_path(name), "rb"))


def row_tensor(d, r):
    prompt = list(d["prompts"][r["idx"]])
    return torch.tensor(prompt + list(r["tokens"]), dtype=torch.long, device="cuda"), len(prompt)


def train(ns, bridge, model, m):
    R = Runner(model, m)
    tr = load_split("train")
    rows = tr["rows"][RANK::WORLD]
    params = [p for n_, p in m.named_parameters() if p.requires_grad]
    master = [p.detach().float().clone() for p in params]
    gbuf = [torch.zeros_like(mp) for mp in master]
    opt = torch.optim.AdamW(master, lr=ns.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=ns.wd)
    per_step = ns.accum
    total = min(ns.steps, (len(rows) * ns.epochs) // per_step) if ns.steps else (len(rows) * ns.epochs) // per_step
    n = torch.tensor([total], device="cuda")
    dist.all_reduce(n, op=dist.ReduceOp.MIN)
    total = int(n.item())
    log(f"train rows/rank {len(rows)}, steps {total}, rows/step {per_step * WORLD}, lr {ns.lr}")

    def lr_at(s):
        if s < ns.warmup:
            return ns.lr * (s + 1) / ns.warmup
        f = (s - ns.warmup) / max(1, total - ns.warmup)
        return ns.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * f)))

    C, t0, ri = ns.chunk, time.time(), 0
    for step in range(total):
        stats = torch.zeros(4, device="cuda", dtype=torch.float64)  # kl sum, positions, 1-TV sum, top1 sum
        for _ in range(per_step):
            r = rows[ri % len(rows)]
            ri += 1
            x, P = row_tensor(tr, r)
            N = x.numel()
            hm_, hq = R.forward(x, train=True)
            # targets t in [P+1, N-1]: p from main at t-1, q from mtp at t-2
            lo, hi = P - 1, N - 2  # mtp base index b = t-2 range
            nt = hi - lo
            gq = torch.zeros((N, hq.shape[-1]), dtype=torch.float32, device="cuda")
            for a in range(lo, hi, C):
                b = min(hi, a + C)
                with torch.no_grad():
                    lp = torch.log_softmax(hm_[a + 1:b + 1].float() @ R.Wf.t(), dim=-1)
                    lq = torch.log_softmax(hq[a:b].detach().float() @ R.Wf.t(), dim=-1)
                    p, q = lp.exp(), lq.exp()
                    stats[0] += (p * (lp - lq)).sum().double()
                    stats[2] += torch.minimum(p, q).sum().double()
                    stats[3] += (lq.argmax(-1) == lp.argmax(-1)).sum().double()
                    gq[a:b] = ((q - p) @ R.Wf) / (nt * per_step * WORLD)
                    del lp, lq, p, q
            stats[1] += nt
            torch.autograd.backward(hq, gq.to(hq.dtype))
            for g_, p_ in zip(gbuf, params):
                if p_.grad is not None:
                    g_.add_(p_.grad.float())
                    p_.grad = None
            del hm_, hq, gq
        for g_ in gbuf:
            dist.all_reduce(g_)
        gn = torch.sqrt(sum((g_ * g_).sum() for g_ in gbuf))
        clip = min(1.0, ns.clip / (gn.item() + 1e-6))
        for mp, g_ in zip(master, gbuf):
            mp.grad = g_ * clip
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step)
        opt.step()
        opt.zero_grad(set_to_none=True)
        for g_ in gbuf:
            g_.zero_()
        with torch.no_grad():
            for p_, mp in zip(params, master):
                p_.copy_(mp.to(p_.dtype))
        dist.all_reduce(stats)
        if step % ns.log_every == 0 or step == total - 1:
            log(f"step {step}/{total} kl {stats[0].item() / stats[1].item():.4f} 1-TV {stats[2].item() / stats[1].item():.4f} "
                f"top1 {stats[3].item() / stats[1].item():.4f} gn {gn.item():.3f} lr {lr_at(step):.2e} "
                f"rows {ri * WORLD} {time.time() - t0:.0f}s")
        if ns.eval_every and (step + 1) % ns.eval_every == 0 and step + 1 < total:
            evaluate(ns, R, limit=ns.eval_quick, tag=f"quick{step + 1}")
    save_head(ns, bridge, model, m)
    evaluate(ns, R, limit=ns.eval_quick, tag="final_quick")


def save_head(ns, bridge, model, m):
    from safetensors.torch import save_file

    os.makedirs(ns.out, exist_ok=True)
    mc = {k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()
          if k.startswith("mtp.") and "_extra_state" not in k and v is not None and torch.is_tensor(v)}
    hf = {}
    for name, w in bridge.export_hf_weights([model], cpu=False, show_progress=False):
        if name.startswith("mtp."):
            hf[name] = w.detach().to(torch.bfloat16).cpu().contiguous()
    if RANK == 0:
        torch.save(mc, os.path.join(ns.out, "head_mcore.pt"))
        save_file(hf, os.path.join(ns.out, "mtp.safetensors"))
        with open(os.path.join(ns.out, "head_names.txt"), "w") as f:
            f.write("# Megatron-Core names (HybridModel, mtp_num_layers=1, mtp_hybrid_override_pattern='*-', TP1)\n")
            for k, v in mc.items():
                f.write(f"{k}\t{tuple(v.shape)}\t{v.dtype}\n")
            f.write("# HF names (NemotronH flattened MTP; Megatron-Bridge NemotronHBridge mapping)\n")
            for k, v in hf.items():
                f.write(f"{k}\t{tuple(v.shape)}\t{v.dtype}\n")
        log("saved head:", len(mc), "mcore tensors,", len(hf), "hf tensors,",
            sum(v.numel() for v in hf.values()), "params ->", ns.out)
    dist.barrier()


def evaluate(ns, R, limit=0, tag="eval"):
    ho = load_split("heldout")
    allrows = sorted(ho["rows"], key=lambda r: (r["idx"], r["j"], r["run"]))
    if limit:
        # spread over prompts: sample j-major so a small limit still covers many prompts
        allrows = sorted(allrows, key=lambda r: (r["j"], r["idx"], r["run"]))[:limit]
    rows = allrows[RANK::WORLD]
    K, C = ns.keys, ns.chunk
    res = []
    t0 = time.time()
    for ri, r in enumerate(rows):
        x, P = row_tensor(ho, r)
        N = x.numel()
        with torch.no_grad():
            hm_, hq = R.forward(x, train=False)
            acc = torch.zeros(6, dtype=torch.float64, device="cuda")  # alpha, Epq, 1-TV, greedy, top1, n
            g = torch.Generator(device="cuda")
            for lo in range(P + 1, N, C):  # target positions t
                hi = min(N, lo + C)
                t = torch.arange(lo, hi, device="cuda")
                y = x[lo:hi]
                lp = torch.log_softmax(hm_[t - 1].float() @ R.Wf.t(), dim=-1)
                lq = torch.log_softmax(hq[t - 2].float() @ R.Wf.t(), dim=-1)
                g.manual_seed((zlib.crc32(f"{r['idx']}-{r['j']}-{r['run']}-{lo}".encode()) * 2654435761) % (2**62))
                if ri == 0 and lo == P + 1:  # metric sanity: the keyed rule must accept every token when q == p
                    g.manual_seed(12345)
                    e = -torch.log(-torch.log(torch.rand(lp.shape, device="cuda", generator=g).clamp_min(1e-30)))
                    z = -torch.log(-torch.log(torch.rand((hi - lo, 1), device="cuda", generator=g).clamp_min(1e-30)))
                    u = -torch.logaddexp(-z, -(lp + e))
                    u.scatter_(1, y[:, None], z)
                    ok = (torch.argmax(lp + (u - lp), dim=-1) == y).float().mean().item()
                    print(f"[sanity rank {RANK}] keyed self-acceptance (q == p) {ok:.4f}; "
                          f"head q(y) mean {lq.gather(1, y[:, None]).exp().mean().item():.3e} "
                          f"head max-prob mean {lq.max(-1).values.exp().mean().item():.3f}", flush=True)
                    assert ok == 1.0, ok
                    g.manual_seed((zlib.crc32(f"{r['idx']}-{r['j']}-{r['run']}-{lo}".encode()) * 2654435761) % (2**62))
                for _ in range(K):
                    e = -torch.log(-torch.log(torch.rand(lp.shape, device="cuda", generator=g).clamp_min(1e-30)))
                    z = -torch.log(-torch.log(torch.rand((hi - lo, 1), device="cuda", generator=g).clamp_min(1e-30)))
                    u = -torch.logaddexp(-z, -(lp + e))  # Gumbel(log p_v) truncated below Z
                    u.scatter_(1, y[:, None], z)  # realised token holds the max
                    d = torch.argmax(lq + (u - lp), dim=-1)
                    acc[0] += (d == y).sum().double() / K
                    del e, u
                p, q = lp.exp(), lq.exp()
                acc[1] += (p * q).sum().double()
                acc[2] += torch.minimum(p, q).sum().double()
                acc[3] += (lq.argmax(-1) == y).sum().double()
                acc[4] += (lq.argmax(-1) == lp.argmax(-1)).sum().double()
                acc[5] += hi - lo
                del lp, lq, p, q
        a = (acc[:5] / acc[5]).tolist()
        res.append(dict(idx=r["idx"], j=r["j"], run=r["run"], P=P, L=N - P, n=int(acc[5].item()),
                        alpha=a[0], epq=a[1], one_m_tv=a[2], greedy=a[3], top1=a[4]))
        if ri % 32 == 0:
            log(f"[{tag}] row {ri}/{len(rows)} L {N - P} alpha {a[0]:.3f} Epq {a[1]:.3f} 1-TV {a[2]:.3f} "
                f"greedy {a[3]:.3f} {time.time() - t0:.0f}s")
    out = [None] * WORLD
    dist.all_gather_object(out, res)
    if RANK == 0:
        allres = [x for o in out for x in o]
        os.makedirs(ns.out, exist_ok=True)
        pickle.dump(allres, open(os.path.join(ns.out, f"{tag}.pkl"), "wb"))
        summarize(allres, tag)
    dist.barrier()


def summarize(res, tag):
    def q(v, f):
        v = sorted(v)
        return v[min(len(v) - 1, int(f * len(v)))]

    n = sum(r["n"] for r in res)
    tw = {k: sum(r[k] * r["n"] for r in res) / n for k in ("alpha", "epq", "one_m_tv", "greedy", "top1")}
    al = [r["alpha"] for r in res]
    worst = min(res, key=lambda r: r["alpha"])
    log(f"[{tag}] rows {len(res)} positions {n} | token-weighted alpha {tw['alpha']:.4f} E_p[q] {tw['epq']:.4f} "
        f"1-TV {tw['one_m_tv']:.4f} greedy {tw['greedy']:.4f} top1(p) {tw['top1']:.4f} | per-row alpha mean "
        f"{sum(al) / len(al):.4f} p10 {q(al, 0.10):.4f} p5 {q(al, 0.05):.4f} min {worst['alpha']:.4f} "
        f"(idx {worst['idx']} j {worst['j']} L {worst['L']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval"], required=True)
    ap.add_argument("--shim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=4, help="rows per rank per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--keys", type=int, default=4, help="Gumbel draws per position in eval")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=0)
    ap.add_argument("--eval_quick", type=int, default=64)
    ap.add_argument("--eval_limit", type=int, default=0)
    ap.add_argument("--crossfade", type=int, default=1)
    ns = ap.parse_args()
    bridge, model, m = build(ns.shim, ns.crossfade)
    if ns.mode == "train":
        train(ns, bridge, model, m)
    else:
        evaluate(ns, Runner(model, m), limit=ns.eval_limit, tag="heldout")
    dist.barrier()
    log("done")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
