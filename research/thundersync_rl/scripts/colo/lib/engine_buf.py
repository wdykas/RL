"""Engine KV / state buffer by rule (colo lead 11-04; the colocated engine's buffer stays resident through training, so
it is sized to what the step can hold in flight, not tuned):

  buffer_GiB = ceil(1.25 x max_requests x per_request / 2^30)
  per_request = max_seq x KV bytes/token/rank            (attention layers x 2 x KV heads/TP x head_dim x 2 B)
              + Mamba layers x (SSM state + conv state)   (heads/TP x head_dim x d_state x 4 B + conv (k-1) x conv dim x 2 B)
              + BI Mamba decode replay buffers            (mamba_bi_decode: 25.5 MiB/request at Super-L22 TP2, measured;
                                                           scaled by Mamba layers x heads/TP x head_dim)
1.25 covers block-granular allocation (paged KV) and the per-request bookkeeping.
  python3 engine_buf.py <config.json> <TP> <max_requests> <max_seq>
"""
import json
import math
import sys

cfg = json.load(open(sys.argv[1]))
tp, nreq, mseq = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
pat = cfg["hybrid_override_pattern"]
n_att, n_mamba = pat.count("*"), pat.count("M")
kvh = max(1, cfg["num_key_value_heads"] // tp)
hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
kv_tok = n_att * 2 * kvh * hd * 2
mh = cfg["mamba_num_heads"] // tp
mhd = cfg["mamba_head_dim"]
dst = cfg["ssm_state_size"]
ssm = mh * mhd * dst * 4
conv_dim = mh * mhd + 2 * cfg["n_groups"] * dst // max(1, tp // 1)
conv = (cfg["conv_kernel"] - 1) * conv_dim * 2
replay = 25.5 * 2**20 * (n_mamba * mh * mhd) / (10 * 64 * 64)   # measured at Super-L22 TP2: 10 layers x 64 heads x 64
per_req = mseq * kv_tok + n_mamba * (ssm + conv) + replay
gib = math.ceil(1.25 * nreq * per_req / 2**30)
print(gib)
print(f"# att {n_att} mamba {n_mamba} kv/token/rank {kv_tok} B, per request {per_req / 2**20:.1f} MiB "
      f"(kv {mseq * kv_tok / 2**20:.1f}, ssm+conv {n_mamba * (ssm + conv) / 2**20:.1f}, replay {replay / 2**20:.1f}), "
      f"{nreq} requests -> {gib} GiB", file=sys.stderr)
