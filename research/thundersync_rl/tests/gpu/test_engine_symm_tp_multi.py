"""engine_symm_tp (double-buffered, one barrier per collective) == the fp16-GEMM forward's stock collectives, bitwise.

torchrun --nproc-per-node 4 test_engine_symm_tp_multi.py --tp 2|4 [--bench 1]

Checks, per TP group:
- eager RS (fp32) / AG (fp16, fp32, bf16) at decode and prefill shapes (prefill above a region falls back): bitwise the
  stock path (NCCL ordered RS / all-gather);
- two CUDA graphs, each a forward of L layers of AG(view) -> consume -> RS with ragged shapes, captured after
  ``forward_start``; replayed in an irregular order (G1 G1 G2 G1 G2 G2 ...) with eager collectives in between and fresh
  data every replay; every output equals the eager stock recomputation (an overwrite race would corrupt a staging
  region the next replay reads);
- a sensitivity control at TP >= 3 (the reversed fold differs from the ordered one on random data);
- --bench: per-call latency in a graph of 20 x (AG view -> RS) at decode shapes, vs the stock NCCL path and, at TP2,
  vs MCore's NVLS multimem collectives (fp32 RS / fp16 AG; timing reference only).
Prints "SYMM_TP_MULTI PASS" on success.
"""
import argparse
import os
import time

import torch
import torch.distributed as dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--layers", type=int, default=11)
    a = ap.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    groups = [dist.new_group(list(range(i, i + a.tp))) for i in range(0, world, a.tp)]
    group = groups[rank // a.tp]
    tr = rank % a.tp
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import enable_batch_invariant_mode

    enable_batch_invariant_mode(backend="te_native", collective="ordered")
    from nemo_rl.models.megatron import engine_symm_tp as est
    from nemo_rl.models.megatron import fp16_gemm as fg

    assert fg.enable_engine_symm_tp(group), "symmetric memory unavailable"
    est._STATE["hooked"] = True  # the engine registers forward_start on its model; here the test calls it
    torch.set_grad_enabled(False)
    import inspect

    _view_kw = "view_ok" in inspect.signature(fg._gather_rows).parameters  # older trees (bench reference): no views
    _fs = getattr(est, "forward_start", lambda: None)

    def gather(x, view_ok=False):
        return fg._gather_rows(x, group, view_ok=view_ok) if _view_kw else fg._gather_rows(x, group)

    def stock_rs(x):
        fg._ENGINE_SYMM_TP["on"] = False
        try:
            return fg._reduce_scatter_rows(x, group)
        finally:
            fg._ENGINE_SYMM_TP["on"] = True

    def stock_ag(x):
        fg._ENGINE_SYMM_TP["on"] = False
        try:
            return fg._gather_rows(x, group)
        finally:
            fg._ENGINE_SYMM_TP["on"] = True

    g = torch.Generator(device="cuda").manual_seed(1234 + rank)
    bad = 0
    H = 4096

    def rnd(T, H_=H):
        return torch.randn(T, H_, generator=g, device="cuda") * (
            10.0 ** torch.randint(-3, 4, (T, 1), generator=g, device="cuda").float())

    # ---- eager ----
    for T, Hs in [(64, 2688), (128, 4096), (288, 4096), (64, 1024), (192, 3584), (16384, 4096)]:
        T = T // a.tp * a.tp
        x = rnd(T, Hs)
        bad += int(not torch.equal(stock_rs(x).view(torch.int32), fg._reduce_scatter_rows(x, group).view(torch.int32)))
        for dt in (torch.float16, torch.float32, torch.bfloat16):
            z = torch.randn(T // a.tp, Hs, generator=g, device="cuda").to(dt)
            bad += int(not torch.equal(stock_ag(z), gather(z, view_ok=True)))
    if a.tp >= 3:
        x = torch.randn(256, H, generator=g, device="cuda")
        parts = [torch.empty_like(x) for _ in range(a.tp)]
        dist.all_gather(parts, x, group=group)
        n = 256 // a.tp
        rev = parts[-1][tr * n:(tr + 1) * n].clone()
        for p in parts[-2::-1]:
            rev += p[tr * n:(tr + 1) * n]
        t = torch.tensor([int(not torch.equal(rev, fg._reduce_scatter_rows(x, group)))], device="cuda")
        dist.all_reduce(t, group=group)
        assert t.item() > 0, "sensitivity control: reversed order gave identical bits on every rank"

    # ---- graphs: forwards of L layers, AG(view) -> consume -> RS ----
    def layer_stack(xs, L, T):
        o = xs
        outs = []
        for l in range(L):
            ag = gather(o.half(), view_ok=True)  # [T, H] gathered, possibly the staging itself
            h = ag.float() * (1.0 + l / 64.0)  # the "GEMM" consuming the view before the next collective
            o = fg._reduce_scatter_rows(h, group)  # [T / tp, H]
            outs.append(o)
        return outs

    def ref_stack(xs, L):
        o = xs
        outs = []
        for l in range(L):
            o = stock_rs(stock_ag(o.half()).float() * (1.0 + l / 64.0))
            outs.append(o)
        return outs

    graphs = {}
    for name, T in (("G1", 64), ("G2", 144)):
        T = T // a.tp * a.tp
        xs = torch.zeros(T // a.tp, H, device="cuda")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                _fs()
                layer_stack(xs, a.layers, T)
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        _fs()
        with torch.cuda.graph(gr):
            outs = layer_stack(xs, a.layers, T)
        graphs[name] = (gr, xs, outs, T)
    order = ["G1", "G1", "G2", "G1", "G2", "G2", "E", "G1", "G2", "E", "E", "G2", "G1"] * 4
    for it, name in enumerate(order):
        if name == "E":  # eager collectives between replays (a prefill-like step)
            _fs()
            x = rnd(96 // a.tp * a.tp)
            bad += int(not torch.equal(stock_rs(x), fg._reduce_scatter_rows(x, group)))
            continue
        gr, xs, outs, T = graphs[name]
        xs.copy_(torch.randn(xs.shape, generator=g, device="cuda"))
        _fs()  # the engine's model forward pre-hook (host only; replays do not depend on it)
        gr.replay()
        ref = ref_stack(xs, a.layers)
        bad += sum(int(not torch.equal(o.view(torch.int32), r.view(torch.int32))) for o, r in zip(outs, ref))
    torch.cuda.synchronize()

    res = {}
    if a.bench:
        for T in (8, 144, 192):
            T = T // a.tp * a.tp
            xs = torch.zeros(T // a.tp, H, device="cuda")
            for name in ("stock", "symm"):
                fg._ENGINE_SYMM_TP["on"] = name == "symm"
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(2):
                        _fs()
                        layer_stack(xs, 20, T)
                torch.cuda.current_stream().wait_stream(s)
                gg = torch.cuda.CUDAGraph()
                _fs()
                with torch.cuda.graph(gg):
                    layer_stack(xs, 20, T)
                for _ in range(3):
                    gg.replay()
                torch.cuda.synchronize()
                dist.barrier(group=group)
                t0 = time.perf_counter()
                for _ in range(50):
                    gg.replay()
                torch.cuda.synchronize()
                res[(name, T)] = (time.perf_counter() - t0) / 50 / 20 * 1e6
            fg._ENGINE_SYMM_TP["on"] = True
            if a.tp == 2:
                try:
                    res[("nvls", T)] = nvls_pair_us(group, T, H)
                except Exception as exc:  # timing reference only
                    if rank == 0:
                        print(f"[symm_tp_multi] NVLS reference unavailable: {exc!r}", flush=True)
    t = torch.tensor([bad], device="cuda")
    dist.all_reduce(t)
    if rank == 0:
        for k, v in sorted(res.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            print(f"[symm_tp_multi] TP{a.tp} T={k[1]} {k[0]:5s}: RS+AG pair {v:.1f} us (incl. the x3 consume op)",
                  flush=True)
        print(f"[symm_tp_multi] TP{a.tp}: mismatches {t.item()} stats {est.stats()}", flush=True)
        print("SYMM_TP_MULTI PASS" if t.item() == 0 else "SYMM_TP_MULTI FAIL", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def nvls_pair_us(group, T, H):
    """Timing reference: MCore NVLS multimem RS (fp32, input in symmetric memory) + AG (fp16) per pair, in a graph."""
    from megatron.core.inference.communication.torch_symm_triton.collectives import (
        multimem_all_gather, multimem_reduce_scatter)
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

    hdl_rs = SymmetricMemoryManager.get_buffer("tdbench_rs", process_group=group, size_mb=32).symm_mem_hdl
    hdl_ag = SymmetricMemoryManager.get_buffer("tdbench_ag", process_group=group, size_mb=32).symm_mem_hdl
    ws = hdl_rs.world_size
    rs_in = hdl_rs.get_buffer(hdl_rs.rank, (T, H), torch.float32, 0)
    rs_out = torch.empty(T // ws, H, device="cuda")
    ag_out = hdl_ag.get_buffer(hdl_ag.rank, (T, H), torch.float16, 0)
    ag_in = torch.empty(T // ws, H, device="cuda", dtype=torch.float16)

    def pair():
        multimem_reduce_scatter(rs_out, rs_in, hdl_rs)
        ag_in.copy_(rs_out)
        multimem_all_gather(ag_out, ag_in, hdl_ag)
        rs_in.copy_(ag_out)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            pair()
    torch.cuda.current_stream().wait_stream(s)
    gg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gg):
        for _ in range(20):
            pair()
    for _ in range(3):
        gg.replay()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    t0 = time.perf_counter()
    for _ in range(50):
        gg.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 50 / 20 * 1e6


if __name__ == "__main__":
    main()
