"""Build an HF-format Nano + MTP checkpoint directory WITHOUT copying Nano weights.

<out>/ = symlinks to every file of the Nano HF snapshot, plus
  config.json                 Nano config + num_nextn_predict_layers 1, mtp_hybrid_override_pattern <pattern>
  mtp.safetensors             the draft head (HF names mtp.layers.{0,1}.*), ~100 MB bf16
  model.safetensors.index.json Nano index + the mtp.* keys -> mtp.safetensors
Megatron-Bridge (NemotronHBridge, flattened MTP mapping) then loads it as HybridModel(mtp_num_layers=1, "*-"),
the same way Super's native MTP checkpoint loads ("*E").

  python mk_shim.py --out <dir> [--head <trained mtp.safetensors>]   (no --head: fresh init)
"""
import argparse
import json
import os

import torch
from safetensors.torch import load_file, save_file

NANO = os.environ.get("NANO_HF_DIR", "")  # local HF snapshot dir of nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16


def init_head(cfg, seed=1234):
    """Fresh "*-" head: attention (inner 0) + dense squared-ReLU MLP (inner 1). Warm start: eh_proj = [0 | I] (+ small
    noise) and small output projections, so the untrained head ~ rmsnorm(h_i), i.e. it starts by predicting with Nano's
    own next-token direction instead of a random one. All norms = 1."""
    g = torch.Generator().manual_seed(seed)
    h, nh, nkv, hd, ffn = (cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"],
                           cfg["intermediate_size"])
    std = cfg.get("initializer_range", 0.02)
    out_std = 0.002

    def n(*s, sd=std):
        return (torch.randn(*s, generator=g) * sd).to(torch.bfloat16)

    eh = torch.randn(h, 2 * h, generator=g) * 0.002
    eh[:, h:] += torch.eye(h)  # input = [enorm(emb) | hnorm(h)]
    one = torch.ones(h, dtype=torch.bfloat16)
    return {
        "mtp.layers.0.enorm.weight": one.clone(),
        "mtp.layers.0.hnorm.weight": one.clone(),
        "mtp.layers.0.eh_proj.weight": eh.to(torch.bfloat16),
        "mtp.layers.0.norm.weight": one.clone(),
        "mtp.layers.0.mixer.q_proj.weight": n(nh * hd, h),
        "mtp.layers.0.mixer.k_proj.weight": n(nkv * hd, h),
        "mtp.layers.0.mixer.v_proj.weight": n(nkv * hd, h),
        "mtp.layers.0.mixer.o_proj.weight": n(h, nh * hd, sd=out_std),
        "mtp.layers.1.norm.weight": one.clone(),
        "mtp.layers.1.mixer.up_proj.weight": n(ffn, h),
        "mtp.layers.1.mixer.down_proj.weight": n(h, ffn, sd=out_std),
        "mtp.layers.1.final_layernorm.weight": one.clone(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--head", default="", help="trained head (HF-named mtp.* safetensors); empty = fresh init")
    ap.add_argument("--pattern", default="*-")
    ap.add_argument("--nano", default=NANO, help="Nano HF snapshot dir (default $NANO_HF_DIR)")
    ns = ap.parse_args()
    if not ns.nano or not os.path.isdir(ns.nano):
        raise SystemExit("pass --nano or set NANO_HF_DIR to a local HF snapshot of NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    os.makedirs(ns.out, exist_ok=True)
    for f in os.listdir(ns.nano):
        if f in ("config.json", "model.safetensors.index.json") or f.startswith("."):
            continue
        dst = os.path.join(ns.out, f)
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(os.path.join(ns.nano, f)), dst)
    cfg = json.load(open(os.path.join(ns.nano, "config.json")))
    cfg["num_nextn_predict_layers"] = 1
    cfg["mtp_hybrid_override_pattern"] = ns.pattern
    cfg["mtp_use_repeated_layer"] = True
    json.dump(cfg, open(os.path.join(ns.out, "config.json"), "w"), indent=2)
    if ns.head:
        head = load_file(ns.head)
    else:
        head = init_head(cfg)
    assert ns.pattern == "*-", "init_head/mapping below written for '*-'"
    save_file({k: v.contiguous() for k, v in head.items()}, os.path.join(ns.out, "mtp.safetensors"))
    idx = json.load(open(os.path.join(ns.nano, "model.safetensors.index.json")))
    for k in head:
        idx["weight_map"][k] = "mtp.safetensors"
    json.dump(idx, open(os.path.join(ns.out, "model.safetensors.index.json"), "w"), indent=2)
    print("shim", ns.out, "head tensors", {k: tuple(v.shape) for k, v in head.items()})


if __name__ == "__main__":
    main()
