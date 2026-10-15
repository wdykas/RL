"""Bitwise check + timing of the learner prefix-state cache (nemo_rl/models/megatron/prefix_state_cache.py).

Builds the training model the way the learner does (Megatron provider from the HF checkpoint, batch-invariant
kernels, fp32 residual, fp16 GEMM forward recipe, fp32 router, FA4) and simulates continuation rounds: rows grow by
128-token pieces; a piece is rejected with probability --p_reject at a random position (the row keeps the accepted
part plus one replacement token). At every round, the logits of each row's piece positions [prompt - 1, n) from

  full:    the whole rows, padded, in one batch (what verify_drafts_block_engine runs today), and
  cached:  rows[i][start_i:] continued from the cache (one batch, mixed starts),

must be bitwise equal. With --sticky the same random sticky anchors are replayed in both (sliced to the segment).

  torchrun --nproc-per-node=<tp*ep> check_prefix_cache.py --model <nano-L8> [--tp 1] [--ep 1] [--rows 4]
      [--prompt 300] [--max_len 2048] [--rounds 40] [--sticky 1] [--out res.json]
  --bench 1024,2048,4096,8192 : time one piece verify forward (full row vs cached) at these prefix lengths instead.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("CUBLASLT_WORKSPACE_SIZE", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":0:0")
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

PIECE = 128


def log(*a):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def build(args):
    from nemo_rl.models.megatron import fp16_gemm, fp32_residual, sticky_routing
    from nemo_rl.models.megatron.zero_train_gen_mismatch import enable_batch_invariant_kernels

    enable_batch_invariant_kernels({"megatron_cfg": {
        "batch_invariant_mode": True, "batch_invariant_backend": "te_native",
        "batch_invariant_collective": "ordered", "flash_attention_version": 4, "attention_backend": "flash"}})
    fp32_residual.install()
    fp16_gemm.install()
    if args.sticky:
        sticky_routing.install()
        sticky_routing.set_delta(args.delta)
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29800 + os.getpid() % 500))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl", rank=rank, world_size=world)
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    ps.initialize_model_parallel(args.tp, 1, expert_model_parallel_size=args.ep)
    model_parallel_cuda_manual_seed(0)
    from megatron.bridge import AutoBridge
    from megatron.core.transformer.enums import AttnBackend

    prov = AutoBridge.from_hf_pretrained(args.model, torch_dtype=torch.bfloat16,
                                         trust_remote_code=True).to_megatron_provider(load_weights=True)
    prov.tensor_model_parallel_size = args.tp
    prov.pipeline_model_parallel_size = 1
    prov.expert_model_parallel_size = args.ep
    prov.expert_tensor_parallel_size = 1
    prov.sequence_parallel = args.tp > 1
    prov.params_dtype = torch.bfloat16
    prov.bf16, prov.fp16 = True, False
    prov.gradient_accumulation_fusion = False
    prov.fp32_residual_connection = True
    prov.moe_router_dtype = "fp32"
    prov.moe_permute_fusion = False
    prov.attention_backend = AttnBackend.flash
    prov.flash_attention_version = 4
    prov.batch_invariant_mode = True
    prov.batch_invariant_backend = "te_native"
    prov.batch_invariant_collective = "ordered"
    prov.bias_activation_fusion = False
    prov.moe_shared_expert_overlap = False
    prov.moe_enable_routing_replay = bool(args.sticky)
    if args.ep > 1:
        prov.moe_token_dispatcher_type = "alltoall"
    if getattr(prov, "moe_router_num_groups", None) == 1:
        prov.moe_router_num_groups = prov.moe_router_group_topk = None
    if hasattr(prov, "forward_gemm_precision"):
        prov.forward_gemm_precision = "fp16"
    prov.finalize()
    model = prov.provide_distributed_model(wrap_with_ddp=False)[0]
    model.eval()
    log("model built", f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params/rank")
    return model


def moe_dims(model):
    from megatron.core.utils import get_model_config

    from nemo_rl.models.megatron.router_replay import _global_moe_layer_numbers

    cfg = get_model_config(model)
    return len(_global_moe_layer_numbers(cfg)), int(cfg.moe_router_topk), int(cfg.num_moe_experts)


def forward(model, rows, anchors, mult, gather):
    """Logits [B, S, V] (fp32, vocab gathered over TP) of rows left-aligned and padded to ``mult``."""
    from nemo_rl.models.megatron.router_replay import clear_router_replay, set_router_replay_forward

    s = -(-max(r.numel() for r in rows) // mult) * mult
    ids = torch.zeros(len(rows), s, dtype=torch.long, device="cuda")
    for b, r in enumerate(rows):
        ids[b, : r.numel()] = r.cuda()
    pos = torch.arange(s, device="cuda").unsqueeze(0).expand(len(rows), -1).contiguous()
    if anchors is not None:
        a = torch.full((len(rows), s, *anchors[0].shape[1:]), -1, dtype=torch.long)
        for b, x in enumerate(anchors):
            a[b, : x.shape[0]] = x
        set_router_replay_forward(model, a.cuda())
    try:
        with torch.no_grad():
            out = model(input_ids=ids, position_ids=pos, attention_mask=None)
    finally:
        if anchors is not None:
            clear_router_replay(model)
    return gather(out)


def make_gather(tp):
    if tp == 1:
        return lambda x: x
    from megatron.core import parallel_state as ps

    def g(x):
        parts = [torch.empty_like(x) for _ in range(tp)]
        dist.all_gather(parts, x.contiguous(), group=ps.get_tensor_model_parallel_group())
        return torch.cat(parts, -1)

    return g


def random_anchors(n, layers, topk, experts, gen):
    """Per position: a random expert set (half the positions) or -1 (self)."""
    a = torch.full((n, layers, topk), -1, dtype=torch.long)
    m = torch.rand(n, generator=gen) < 0.5
    for i in m.nonzero().flatten().tolist():
        for ly in range(layers):
            a[i, ly] = torch.randperm(experts, generator=gen)[:topk]
    return a


def check(model, args):
    from nemo_rl.models.megatron.prefix_state_cache import PrefixStateCache

    cache = PrefixStateCache(model, gpu_bytes=int(args.gpu_gb * 2**30))
    gather = make_gather(args.tp)
    gen = torch.Generator().manual_seed(args.seed)  # same on every rank
    vocab = 30000
    rows = [torch.randint(100, vocab, (args.prompt + int(torch.randint(0, 200, (1,), generator=gen)),),
                          generator=gen) for _ in range(args.rows)]
    dims = moe_dims(model) if args.sticky else None
    anchors = ([random_anchors(args.max_len + PIECE, *dims, gen) for _ in rows] if args.sticky else None)
    if args.sticky:
        from nemo_rl.models.megatron import sticky_routing
        sticky_routing.start_stats()
    res = {"rounds": 0, "pieces": 0, "rejections": 0, "positions": 0, "mismatch": 0, "hits": 0,
           "max_abs_diff": 0.0, "starts": []}
    keys = [f"r{i}" for i in range(args.rows)]
    for rnd in range(args.rounds):
        live = [i for i in range(args.rows) if rows[i].numel() + PIECE <= args.max_len]
        if not live:
            break
        prompt_lens = [rows[i].numel() for i in live]
        piece = [torch.randint(100, vocab, (PIECE,), generator=gen) for _ in live]
        full_rows = [torch.cat([rows[i], p]) for i, p in zip(live, piece)]
        an = [anchors[i][: r.numel()] for i, r in zip(live, full_rows)] if anchors else None
        full = forward(model, full_rows, an, args.mult, gather)
        starts = [cache.start(keys[i], r, pl) for i, r, pl in zip(live, full_rows, prompt_lens)]
        seg = [r[s:] for r, s in zip(full_rows, starts)]
        an_seg = [a[s:] for a, s in zip(an, starts)] if an else None
        with cache.forward([keys[i] for i in live], full_rows, starts):
            inc = forward(model, seg, an_seg, args.mult, gather)
        for b, (i, r, pl, s) in enumerate(zip(live, full_rows, prompt_lens, starts)):
            n = r.numel()
            f = full[b, pl - 1 : n]
            c = inc[b, pl - 1 - s : n - s]
            same = torch.equal(f, c)
            res["pieces"] += 1
            res["positions"] += n - pl + 1
            res["hits"] += int(s > 0)
            res["starts"].append((pl, s))
            if not same:
                res["mismatch"] += 1
                d = float((f.float() - c.float()).abs().max())
                res["max_abs_diff"] = max(res["max_abs_diff"], d)
                log(f"MISMATCH round {rnd} row {i} prompt {pl} start {s} max|d| {d:.3e}")
            # verification outcome: accept, or reject at a random position (+ one replacement token)
            if torch.rand(1, generator=gen).item() < args.p_reject:
                k = int(torch.randint(0, PIECE, (1,), generator=gen))
                y = torch.randint(100, vocab, (1,), generator=gen)
                rows[i] = torch.cat([r[: pl + k], y])
                res["rejections"] += 1
            else:
                rows[i] = r
        res["rounds"] += 1
        if rnd % 5 == 0:
            log(f"round {rnd}: rows {len(live)} lens {[x.numel() for x in full_rows]} starts {starts} "
                f"mismatch so far {res['mismatch']}")
    if args.sticky:
        from nemo_rl.models.megatron import sticky_routing
        res["sticky"] = sticky_routing.pop_stats()
    res["cache_stats"] = cache.stats
    res["cache_bytes"] = cache.bytes()
    res.pop("starts") if len(res["starts"]) > 400 else None
    log("RESULT", json.dumps({k: v for k, v in res.items() if k != "starts"}))
    log("PASS" if res["mismatch"] == 0 and res["pieces"] > 0 else "FAIL")
    return res


def bench(model, args):
    """Piece-verify forward time vs prefix length: full row vs cached (start = last boundary)."""
    from nemo_rl.models.megatron.prefix_state_cache import PrefixStateCache

    cache = PrefixStateCache(model, gpu_bytes=int(args.gpu_gb * 2**30))
    gather = make_gather(args.tp)
    gen = torch.Generator().manual_seed(args.seed)
    out = {}
    for plen in (int(x) for x in args.bench.split(",")):
        prefix = torch.randint(100, 30000, (plen,), generator=gen)
        # Fill the cache through a forward of the prefix row (as the previous piece did).
        cache.clear()
        with cache.forward(["b"], [prefix], [0]):
            forward(model, [prefix], None, args.mult, gather)
        row = torch.cat([prefix, torch.randint(100, 30000, (PIECE,), generator=gen)])
        s = cache.start("b", row, plen)
        times = {"full": [], "cached": []}
        for _ in range(args.reps + 2):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            full = forward(model, [row], None, args.mult, gather)[0, plen - 1:]
            torch.cuda.synchronize()
            times["full"].append(time.perf_counter() - t0)
            del full
            t0 = time.perf_counter()
            with cache.forward(["b"], [row], [s]):
                inc = forward(model, [row[s:]], None, args.mult, gather)[0, plen - 1 - s:]
            torch.cuda.synchronize()
            times["cached"].append(time.perf_counter() - t0)
            del inc
        med = {k: sorted(v[2:])[len(v[2:]) // 2] for k, v in times.items()}
        out[plen] = {"start": s, "full_ms": 1e3 * med["full"], "cached_ms": 1e3 * med["cached"],
                     "speedup": med["full"] / med["cached"]}
        log(f"prefix {plen}: start {s} full {1e3 * med['full']:.1f} ms cached {1e3 * med['cached']:.1f} ms "
            f"x{med['full'] / med['cached']:.1f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--prompt", type=int, default=300)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--p_reject", type=float, default=0.3)
    ap.add_argument("--mult", type=int, default=64, help="make_sequence_length_divisible_by")
    ap.add_argument("--sticky", type=int, default=0)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--gpu_gb", type=float, default=0.05, help="small: exercises host eviction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", default="")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    model = build(args)
    res = bench(model, args) if args.bench else check(model, args)
    if args.out and (not dist.is_initialized() or dist.get_rank() == 0):
        json.dump({"args": vars(args), "result": res}, open(args.out, "w"), indent=1)
    dist.barrier()
    os._exit(0)


if __name__ == "__main__":
    main()
