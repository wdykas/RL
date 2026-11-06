"""[FFN / NF (extra eh feature columns) from env; prints a 2-layer estimate]
Microbench of the MTP draft source per tree iteration (tree_spec._mtp_device's work), Nano dimensions, random
weights, CUDA-graph replay (the production path): per depth = one tree_mtp.MTPHead.forward_rows over t_max = 64 rows
+ the keyed sampler S over the vocabulary; plus the KV append of the accepted positions (8 x 64 rows, once per
iteration). Target (lead, module acceptance): <= 0.47 ms per depth.
  python mtp_bench2.py [--live 16,64] [--ctx 2048,8192]
"""
import argparse
import os
import tempfile

import torch

from nemo_rl.models.generation.megatron import tree_mtp
from thundersync_rl import keyed_sampling_s as S

HID, NH, NKV, HD, V = 2688, 32, 2, 128, 131072
FFN = int(os.environ.get("FFN", "1856"))
NF = int(os.environ.get("NF", "0"))
TM, KMTP = 64, 4


def head(slots, max_seq):
    g = torch.Generator().manual_seed(0)
    r = lambda *s: (torch.randn(*s, generator=g) * 0.02).to(torch.bfloat16)  # noqa: E731
    p, a, f = "mtp.layers.0.", "mtp.layers.0.mtp_model_layer.layers.0.self_attention.", \
        "mtp.layers.0.mtp_model_layer.layers.1.mlp."
    one = lambda: torch.ones(HID, dtype=torch.bfloat16)  # noqa: E731
    sd = {p + "enorm.weight": one(), p + "hnorm.weight": one(), p + "eh_proj.weight": r(HID, 2 * HID),
          a + "linear_qkv.layer_norm_weight": one(), a + "linear_qkv.weight": r((NH + 2 * NKV) * HD, HID),
          a + "linear_proj.weight": r(HID, NH * HD), f + "linear_fc1.layer_norm_weight": one(),
          f + "linear_fc1.weight": r(FFN, HID), f + "linear_fc2.weight": r(HID, FFN), p + "final_layernorm.weight": one()}
    path = os.path.join(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")), "h.pt")
    torch.save(sd, path)
    emb = r(V, HID).cuda()
    out = r(V, HID).cuda()
    return tree_mtp.MTPHead(path, emb, out, slots=slots, max_seq=max_seq, num_heads=NH, num_kv_heads=NKV, head_dim=HD)


def timed_graph(fn, iters=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    gph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gph):
        fn()
    for _ in range(5):
        gph.replay()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    e0.record()
    for _ in range(iters):
        gph.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="16,64")
    ap.add_argument("--ctx", default="2048,8192")
    a = ap.parse_args()
    for ctx in map(int, a.ctx.split(",")):
        h = head(64, ctx + 16)
        h.kc.normal_(0, 1)
        h.vc.normal_(0, 1)
        for live in map(int, a.live.split(",")):
            sl = torch.full((TM,), -1, dtype=torch.long, device="cuda")
            sl[:live] = torch.arange(live, device="cuda")
            pos = torch.full((TM,), ctx - 8, dtype=torch.long, device="cuda")
            hid = torch.randn(TM, HID, device="cuda").to(torch.bfloat16)
            tok = torch.randint(0, V, (TM,), device="cuda")
            kid = torch.arange(TM, device="cuda", dtype=torch.int64)
            temp = torch.ones(TM, device="cuda")
            ws = torch.empty(S.workspace_bytes(TM, V), dtype=torch.uint8, device="cuda")
            outt = torch.empty(TM, dtype=torch.int64, device="cuda")
            sl8 = sl[:, None].expand(TM, 8).reshape(-1).contiguous()
            p8 = (pos[:, None] - 8 + torch.arange(8, device="cuda")).reshape(-1)
            h8 = hid.repeat_interleave(8, 0)
            t8 = tok.repeat_interleave(8)

            fbuf = torch.randn(TM, max(NF, 1) * HID, device="cuda").to(torch.bfloat16)
            fw = torch.ones(max(NF, 1) * HID, device="cuda").to(torch.bfloat16)
            ehx = (torch.randn(HID, max(NF, 1) * HID, device="cuda") * 0.02).to(torch.bfloat16)

            def feat():  # the extra feature columns of eh_proj (RMS of each feature + its GEMM), every depth
                if NF:
                    return tree_mtp._rms(fbuf.view(TM * NF, HID), fw[:HID], h.eps).view(TM, NF * HID) @ ehx.t()

            def depth():
                feat()
                u, lg = h.forward_rows(sl, pos, hid, tok)
                S.wide_sample(lg, kid, pos + 1, temp, 12345, out=outt, workspace=ws)

            def head_only():
                h.forward_rows(sl, pos, hid, tok)

            def append():
                h.append_kv(sl8, p8, h8, t8)

            def full():
                append()
                x, u = tok, hid
                for k in range(KMTP):
                    u, lg = h.forward_rows(sl, pos + k, u, x)
                    S.wide_sample(lg, kid, pos + 1 + k, temp, 12345, out=outt, workspace=ws)
                    x = outt

            u0 = h._u(hid, tok)
            q0, k0, v0 = h._qkv(u0)
            o0 = torch.empty_like(q0)
            sl32 = sl.to(torch.int32)
            ln = torch.where(sl >= 0, pos + 1, torch.zeros_like(pos)).to(torch.int32)
            lg0 = torch.empty(TM, V, dtype=torch.bfloat16, device="cuda")

            def p_u():
                h._u(hid, tok)

            def p_qkv():
                h._qkv(u0)

            def p_attn():
                h._attend(q0, sl, ln)

            def p_mlp():
                uu = u0 + o0.reshape(TM, -1) @ h.w_proj.t()
                z = torch.relu(tree_mtp._rms(uu, h.ln_fc1, h.eps) @ h.w_fc1.t())
                uu + (z * z) @ h.w_fc2.t()

            def p_lm():
                torch.matmul(tree_mtp._rms(u0, h.final, h.eps), h.out.t(), out=lg0)

            def p_s():
                S.wide_sample(lg0, kid, pos + 1, temp, 12345, out=outt, workspace=ws)

            parts = {n: timed_graph(f) for n, f in (("u", p_u), ("qkv", p_qkv), ("attn", p_attn), ("mlp", p_mlp),
                                                    ("lm", p_lm), ("S", p_s))}
            print(f"[mtpbench] parts ctx={ctx} live={live} " + " ".join(f"{k}={v:.3f}" for k, v in parts.items()),
                  flush=True)
            res = {n: timed_graph(f) for n, f in (("depth", depth), ("head_only", head_only), ("append", append),
                                                  ("iteration_K4", full))}
            print(f"[mtpbench] FFN={FFN} NF={NF} ctx={ctx} live={live} depth={res['depth']:.3f}ms two_layer_est="
                  f"{res['depth'] + res['head_only'] - parts['lm']:.3f}ms", flush=True)
            print(f"[mtpbench] ctx={ctx} live={live}/{TM} " + " ".join(f"{k}={v:.3f}ms" for k, v in res.items())
                  + f" per_depth(K4)={(res['iteration_K4'] - res['append']) / KMTP:.3f}ms", flush=True)
    print(f"[mtpbench] head KV per rank at slots=64 ctx=8192: {2 * 64 * 8208 * NKV * HD * 2 / 2**20:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
