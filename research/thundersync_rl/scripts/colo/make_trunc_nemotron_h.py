"""Truncate a Nemotron-H (Nemotron-3) HF checkpoint to its first N hybrid layers; the layer-pattern prefix is kept, so
the Mamba / attention / MoE interleave of the full model is preserved, and MTP layers are dropped.

  python make_trunc_nemotron_h.py <hf snapshot dir> N <out dir>

The colo kit's presets use Super-120B-A12B truncated to 8 and 22 layers (nemotron3-super-L8 / -L22) and Ultra truncated
to 27 layers (nemotron3-ultra-L27), under $COLO_MODELS. Truncated models emit noise text (every row runs to the length
cap), so they test plumbing and scale only; the headline regime is the full Nano model.
"""
import glob
import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

SNAP, N, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
os.makedirs(OUT, exist_ok=True)
T = {}
for f in sorted(glob.glob(os.path.join(SNAP, "*.safetensors"))):
    with safe_open(f, "pt") as h:
        for k in h.keys():
            if k.startswith("mtp") or ".mtp." in k:
                continue
            if k.startswith("backbone.layers.") and int(k.split(".")[2]) >= N:
                continue
            T[k] = h.get_tensor(k)
save_file(T, os.path.join(OUT, "model.safetensors"), metadata={"format": "pt"})
c = json.load(open(os.path.join(SNAP, "config.json")))
c["num_hidden_layers"] = N
c["hybrid_override_pattern"] = c["hybrid_override_pattern"][:N]
if "num_nextn_predict_layers" in c:
    c["num_nextn_predict_layers"] = 0
c.pop("mtp_hybrid_override_pattern", None)
json.dump(c, open(os.path.join(OUT, "config.json"), "w"), indent=1)
for f in os.listdir(SNAP):
    if f.endswith((".py", ".json", ".jinja")) and f not in ("config.json", "model.safetensors.index.json"):
        shutil.copy(os.path.join(SNAP, f), os.path.join(OUT, f))
print("wrote", OUT, len(T), "tensors, pattern", c["hybrid_override_pattern"])
