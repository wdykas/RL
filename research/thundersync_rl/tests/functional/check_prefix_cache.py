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

  --mode v0 (verify0 capture): whole drafts (prompt + --g_min..--g_max tokens) are verified packed
  --v0_pack per forward, as verify0; the forward runs under the cache with fresh keys and the prompt's boundary
  as floor (capture). Checked: (1) the capture forward's logits [prompt - 1, n) == the plain whole-row forward's
  (verify0's decisions see the same bits); (2) decisions: finished (released) / rejected at a random tau (+ one
  replacement token) / cut (whole accepted); continuing rows are trimmed to their first piece's boundary;
  (3) every first piece (prefix + 128 tokens) continued from the verify0 state == the whole-row forward, bitwise;
  then --rounds more pieces with rejections. --control stale: perturb the weights after the capture without
  dropping the cache (a state kept across a weight change): every first-piece hit must MISMATCH, and
  set_weights_version must drop the entries (then equal again). --control negctl: captures committed scaled by
  (1 + 2^-6) (the in-loop THUNDERSYNC_PREFIX_CACHE_NEGCTL): every hit must mismatch.
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
    if args.graphs:
        # The learner's LGRAPHS path: device MoE dispatch (no host syncs) + learner CUDA graphs.
        os.environ["NEMO_RL_MOE_DEVICE_DISPATCH"] = "1"
        os.environ["NEMO_RL_LEARNER_GRAPHS"] = "1"
        # The uncached reference forwards all rows whole in one call (8 x ~3.4k tokens).
        os.environ.setdefault("NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS", "65536")
        from nemo_rl.models.megatron import device_dispatch

        device_dispatch.install()
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
    if args.graphs:
        from nemo_rl.models.megatron import learner_graphs

        learner_graphs.ensure_graph_safe_rng()
    log("model built", f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params/rank")
    return model


def moe_dims(model):
    from megatron.core.utils import get_model_config

    from nemo_rl.models.megatron.router_replay import _global_moe_layer_numbers

    cfg = get_model_config(model)
    return len(_global_moe_layer_numbers(cfg)), int(cfg.moe_router_topk), int(cfg.num_moe_experts)


def forward(model, rows, anchors, mult, gather, graphs=False):
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
            if graphs:
                from nemo_rl.models.megatron import learner_graphs

                out = learner_graphs.call(model, ids, pos, None)
            else:
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

    cache = PrefixStateCache(model, gpu_bytes=int(args.gpu_gb * 2**30), max_len=args.max_len + 2 * PIECE)
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
        with torch.no_grad(), cache.forward([keys[i] for i in live], full_rows, starts):
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


def _perturb(model, scale=1e-3, seed=1234):
    """Add small noise to the first Mamba in_proj and attention qkv weights (same on every rank)."""
    from megatron.core.ssm.mamba_mixer import MambaMixer
    from megatron.core.transformer.attention import Attention

    saved, done = [], set()
    g = torch.Generator(device="cuda").manual_seed(seed)
    for m in model.modules():
        if isinstance(m, MambaMixer) and "mamba" not in done:
            ws = [m.in_proj.weight]
            done.add("mamba")
        elif isinstance(m, Attention) and "attn" not in done:
            ws = [m.linear_qkv.weight]
            done.add("attn")
        else:
            continue
        for w in ws:
            saved.append((w, w.detach().clone()))
            noise = torch.randn(w.shape, generator=g, device=w.device, dtype=torch.float32)
            w.data.add_((noise * scale * w.float().std()).to(w.dtype))
    return saved


def check_v0(model, args):
    """verify0 capture: first pieces continued from verify0's state vs whole rows (bitwise).

    ``--graphs 1``: two caches run in lockstep on the same rows: "eager" (the model called
    directly) and "graph" (through ``learner_graphs.call``: first sighting eager, then
    captured, then replayed). Every cached forward of both must equal the uncached whole-row
    forward (eager, the reference) bitwise, and the graph cache must have replayed.
    """
    from nemo_rl.models.megatron.prefix_state_cache import PrefixStateCache

    max_len = args.g_max + args.prompt + 200 + (args.rounds + 4) * PIECE + 512
    names = ["eager", "graph"] if args.graphs else ["eager"]
    caches = {n: PrefixStateCache(model, gpu_bytes=int(args.gpu_gb * 2**30), max_len=max_len) for n in names}
    for c in caches.values():
        c.set_weights_version("w0")
        if args.control == "negctl":
            c.negctl = 2.0**-6
    gather = make_gather(args.tp)
    gen = torch.Generator().manual_seed(args.seed)
    vocab = 30000
    dims = moe_dims(model) if args.sticky else None
    prompts = [torch.randint(100, vocab, (args.prompt + int(torch.randint(0, 200, (1,), generator=gen)),),
                             generator=gen) for _ in range(args.rows)]
    drafts = [torch.randint(100, vocab, (int(torch.randint(args.g_min, args.g_max + 1, (1,), generator=gen)),),
                            generator=gen) for _ in range(args.rows)]
    rows = [torch.cat([p, d]) for p, d in zip(prompts, drafts)]
    cap_len = max(r.numel() for r in rows) + (args.rounds + 2) * PIECE
    anchors = [random_anchors(cap_len, *dims, gen) for _ in rows] if args.sticky else None
    keys = [f"v{i}" for i in range(args.rows)]
    res = {"v0_rows": 0, "v0_capture_mismatch": 0, "captured": 0, "finished": 0, "rejected": 0, "cut": 0,
           "first_pieces": 0, "first_hits": 0, "first_v0_hits": 0, "first_mismatch": 0, "first_skipped": 0,
           "first_recomputed_prefix": 0, "pieces": 0, "hits": 0, "mismatch": 0, "rejections": 0,
           "max_abs_diff": 0.0, "control": args.control, "graphs": int(args.graphs),
           "graph_vs_eager_mismatch": 0, "time_ms": {n: [] for n in names}, "capture_reserved_delta_mb": []}

    def lg_stats():
        if not args.graphs:
            return {}
        from nemo_rl.models.megatron import learner_graphs

        return dict(learner_graphs.stats())

    def cached(name, ks, b_rows, starts, an_seg, floors=None):
        """The cached forward of one cache; returns (logits, from_v0, ms, replayed)."""
        c = caches[name]
        seg = [r[s:] for r, s in zip(b_rows, starts)]
        st0 = lg_stats()
        torch.cuda.synchronize()
        r0 = torch.cuda.memory_reserved()
        t0 = time.perf_counter()
        with torch.no_grad(), c.forward(ks, b_rows, starts, floors=floors):
            out = forward(model, seg, an_seg, args.mult, gather, graphs=(name == "graph"))
        torch.cuda.synchronize()
        ms = 1e3 * (time.perf_counter() - t0)
        st1 = lg_stats()
        captured = st1.get("captured", 0) > st0.get("captured", 0)
        if captured:
            # The graph's private pool (+ any allocator growth of that call).
            res["capture_reserved_delta_mb"].append((torch.cuda.memory_reserved() - r0) >> 20)
        replayed = st1.get("replay_fwd", 0) > st0.get("replay_fwd", 0) and not captured
        return out, list(c.last_from_v0), ms, replayed

    def note_time(name, ms, replayed):
        res["time_ms"][name].append((round(ms, 2), bool(replayed)))

    mem0 = torch.cuda.memory_reserved()
    # ---- verify0: packed whole rows, captured ----
    order = sorted(range(args.rows), key=lambda i: -rows[i].numel())
    for a in range(0, args.rows, args.v0_pack):
        b_idx = order[a : a + args.v0_pack]
        b_rows = [rows[i] for i in b_idx]
        an = [anchors[i][: rows[i].numel()] for i in b_idx] if anchors else None
        full = forward(model, b_rows, an, args.mult, gather)
        outs = {}
        for n in names:
            outs[n], _, ms, rep = cached(n, [keys[i] for i in b_idx], b_rows, [0] * len(b_idx), an,
                                         floors=[prompts[i].numel() for i in b_idx])
            note_time(n, ms, rep)
        for b, i in enumerate(b_idx):
            pl, n_ = prompts[i].numel(), rows[i].numel()
            res["v0_rows"] += 1
            for n in names:
                if not torch.equal(full[b, pl - 1 : n_], outs[n][b, pl - 1 : n_]):
                    res["v0_capture_mismatch"] += 1
                    log(f"V0 CAPTURE MISMATCH ({n}) row {i}")
        del full, outs
    c0 = caches["eager"]
    res["captured"] = sum(k in c0.entries for k in keys)
    res["bytes_after_capture"] = c0.bytes()
    # ---- decisions (stand-ins for verify0's: a function of a separate stream) ----
    live, prefix_of = [], {}
    for i in range(args.rows):
        pl, g = prompts[i].numel(), drafts[i].numel()
        u = torch.rand(1, generator=gen).item()
        if u < args.p_finish:
            for c in caches.values():
                c.release([keys[i]])
            res["finished"] += 1
            continue
        if u < args.p_finish + args.p_reject:
            tau = int(torch.randint(0, g, (1,), generator=gen))
            y = torch.randint(100, vocab, (1,), generator=gen)
            prefix_of[i] = torch.cat([rows[i][: pl + tau], y])
            res["rejected"] += 1
        else:
            prefix_of[i] = rows[i]
            res["cut"] += 1
        for c in caches.values():
            c.trim(keys[i], prefix_of[i].numel() - 1)  # bound = first piece's prompt length - 1
        live.append(i)
    res["bytes_after_trim"] = c0.bytes()
    saved = _perturb(model) if args.control == "stale" else []
    # ---- first pieces, then more rounds ----
    cur = {i: prefix_of[i] for i in live}
    for rnd in range(args.rounds + 1):
        act = [i for i in live if cur[i].numel() + PIECE <= cap_len]
        if not act:
            break
        plens = [cur[i].numel() for i in act]
        full_rows = [torch.cat([cur[i], torch.randint(100, vocab, (PIECE,), generator=gen)]) for i in act]
        an = [anchors[i][: r.numel()] for i, r in zip(act, full_rows)] if anchors else None
        full = forward(model, full_rows, an, args.mult, gather)
        starts = {n: [caches[n].start(keys[i], r, pl) for i, r, pl in zip(act, full_rows, plens)] for n in names}
        if args.graphs:
            assert starts["graph"] == starts["eager"], (starts["graph"], starts["eager"])
        outs, fv0 = {}, {}
        for n in names:
            an_seg = [x[s:] for x, s in zip(an, starts[n])] if an else None
            outs[n], fv0[n], ms, rep = cached(n, [keys[i] for i in act], full_rows, starts[n], an_seg)
            note_time(n, ms, rep)
        st = starts["eager"]
        for b, (i, r, pl, s) in enumerate(zip(act, full_rows, plens, st)):
            n_ = r.numel()
            same = all(torch.equal(full[b, pl - 1 : n_], outs[n][b, pl - 1 - s : n_ - s]) for n in names)
            if args.graphs and not torch.equal(outs["eager"][b, pl - 1 - s : n_ - s], outs["graph"][b, pl - 1 - s : n_ - s]):
                res["graph_vs_eager_mismatch"] += 1
            if rnd == 0:
                res["first_pieces"] += 1
                res["first_hits"] += int(s > 0)
                res["first_v0_hits"] += int(fv0["eager"][b])
                res["first_mismatch"] += int(not same)
                res["first_skipped"] += s
                res["first_recomputed_prefix"] += pl - 1 - s
            else:
                res["pieces"] += 1
                res["hits"] += int(s > 0)
                res["mismatch"] += int(not same)
            if not same:
                d = max(float((full[b, pl - 1 : n_].float() - outs[n][b, pl - 1 - s : n_ - s].float()).abs().max())
                        for n in names)
                res["max_abs_diff"] = max(res["max_abs_diff"], d)
                if args.control == "none":
                    log(f"MISMATCH round {rnd} row {i} prompt {pl} start {s} v0 {fv0['eager'][b]} max|d| {d:.3e}")
            if rnd and torch.rand(1, generator=gen).item() < args.p_reject:
                k = int(torch.randint(0, PIECE, (1,), generator=gen))
                cur[i] = torch.cat([r[: pl + k], torch.randint(100, vocab, (1,), generator=gen)])
                res["rejections"] += 1
            else:
                cur[i] = r
        del full, outs
        if args.control != "none" and rnd >= (4 if args.graphs else 0):
            break  # controls: first pieces (+ rounds replayed from the graph under graphs)
    if args.control == "stale":
        # The worker's rule: a new weights version drops every entry -> no hit, equal again.
        act = live[: min(4, len(live))]
        full_rows = [torch.cat([prefix_of[i], torch.arange(200, 200 + PIECE)]) for i in act]
        an = [anchors[i][: r.numel()] for i, r in zip(act, full_rows)] if anchors else None
        full = forward(model, full_rows, an, args.mult, gather)
        res["stale_after_version_change_equal"] = True
        for n in names:
            caches[n].set_weights_version("w1")
            starts = [caches[n].start(keys[i], r, prefix_of[i].numel()) for i, r in zip(act, full_rows)]
            inc, _, _, _ = cached(n, [keys[i] for i in act], full_rows, starts, an)
            res["stale_after_version_change_starts"] = starts
            # real positions only (the hooked forward leaves padding positions zero)
            res["stale_after_version_change_equal"] &= all(
                torch.equal(full[b, : r.numel()], inc[b, : r.numel()]) for b, r in enumerate(full_rows)
            )
        for w, x in saved:
            w.data.copy_(x)
    res["cache_stats"] = {n: dict(c.stats) for n, c in caches.items()}
    res["cache_bytes"] = {n: c.bytes() for n, c in caches.items()}
    res["pool_bytes_per_cache"] = c0.pool_bytes()
    res["reserved_growth_bytes"] = torch.cuda.memory_reserved() - mem0
    if args.graphs:
        res["learner_graphs"] = lg_stats()
        # Graph-private pool memory (only the cached forwards go through learner_graphs here).
        segs = torch.cuda.memory._snapshot().get("segments", [])
        res["graph_pool_mb"] = sum(
            sg.get("total_size", 0) for sg in segs if tuple(sg.get("segment_pool_id", (0, 0))) != (0, 0)
        ) >> 20
        # Same calls: the eager cache's forward of each call the graph cache replayed.
        pairs = list(zip(res["time_ms"]["graph"], res["time_ms"]["eager"]))
        rep_t = [g for (g, r), _ in pairs if r]
        eag_t = [e for (_, r), (e, _) in pairs if r]
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None  # noqa: E731
        res["replayed_ms_median"], res["eager_ms_median_same_calls"] = med(rep_t), med(eag_t)
    log("RESULT", json.dumps(res))
    if args.control == "none":
        ok = (res["v0_capture_mismatch"] == 0 and res["first_mismatch"] == 0 and res["mismatch"] == 0
              and res["first_v0_hits"] > 0 and res["first_v0_hits"] == res["first_pieces"]
              and res["graph_vs_eager_mismatch"] == 0)
        if args.graphs:
            ok = ok and sum(r for _, r in res["time_ms"]["graph"]) > 0
    else:
        # The check must catch the bad state: every first piece from a verify0 capture mismatches.
        ok = (res["v0_capture_mismatch"] == 0 and res["first_v0_hits"] > 0
              and res["first_mismatch"] == res["first_v0_hits"])
        if args.control == "stale":
            ok = ok and res["stale_after_version_change_equal"] and not any(res["stale_after_version_change_starts"])
        if args.graphs:
            # ... and keeps failing in rounds replayed from the graph (the bad prefix stays in the pages).
            ok = ok and sum(r for _, r in res["time_ms"]["graph"]) > 0 and res["mismatch"] > 0
    log(("PASS" if ok else "FAIL") + f" (mode v0, control {args.control}, graphs {args.graphs})")
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
        with torch.no_grad(), cache.forward(["b"], [prefix], [0]):
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
            with torch.no_grad(), cache.forward(["b"], [row], [s]):
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
    ap.add_argument("--mode", default="pieces", choices=["pieces", "v0"])
    ap.add_argument("--g_min", type=int, default=300)
    ap.add_argument("--g_max", type=int, default=1500)
    ap.add_argument("--v0_pack", type=int, default=2)
    ap.add_argument("--p_finish", type=float, default=0.2)
    ap.add_argument("--control", default="none", choices=["none", "stale", "negctl"])
    ap.add_argument("--graphs", type=int, default=0, help="also run the cached forwards through learner CUDA graphs")
    args = ap.parse_args()
    model = build(args)
    if args.bench:
        res = bench(model, args)
    elif args.mode == "v0":
        res = check_v0(model, args)
    else:
        res = check(model, args)
    if args.out and (not dist.is_initialized() or dist.get_rank() == 0):
        json.dump({"args": vars(args), "result": res}, open(args.out, "w"), indent=1)
    dist.barrier()
    os._exit(0)


if __name__ == "__main__":
    main()
