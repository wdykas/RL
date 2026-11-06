"""Engine-text rows for an apples-to-apples offline vs in-engine acceptance check: the plain arm's generations of
tree_engine_long.py --arm plain (--prompts real: prompt = last 400 tokens of the first n held-out prompts; request i < tails
is an MTP tail) -> eval_walk pickle {"prompts": {i: list}, "rows": [dict(idx=i, j=0, run="eng", tokens=list)]}.
  python mk_engine_rows.py <plain.rank0.pt> <out.pkl> [n=24] [tails=16]   (held-out prompts: $DRAFT_HEAD_DATA/heldout.pkl)"""
import os
import pickle
import sys

import torch

src, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 24
tails = int(sys.argv[4]) if len(sys.argv) > 4 else 16
if not os.environ.get("DRAFT_HEAD_DATA"):
    sys.exit("set DRAFT_HEAD_DATA to the prep_data.py output dir (heldout.pkl)")
ho = pickle.load(open(os.path.join(os.environ["DRAFT_HEAD_DATA"], "heldout.pkl"), "rb"))
real = [list(map(int, ho["prompts"][k]))[-400:] for k in sorted(ho["prompts"])[:n]]
done = torch.load(src)
d = dict(prompts={}, rows=[])
for i in range(tails):
    toks = done[7_000_000 + i][0].tolist()
    d["prompts"][i] = real[i % len(real)]
    d["rows"].append(dict(idx=i, j=0, run="eng", tokens=toks))
pickle.dump(d, open(out, "wb"), protocol=4)
print("[mkeng] rows", len(d["rows"]), "tokens", sum(len(r["tokens"]) for r in d["rows"]))
