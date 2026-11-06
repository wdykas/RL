"""Nano real-text rows (row dump of the policy's own exact samples, DAPO-Math, T 1, cap 8192) -> train / held-out pickles.

Input: --src DIR holding <run>/rows_step*.jsonl, one JSON object per line with keys
  step, idx (prompt index), j (sample index), prompt (token ids or null), tokens (generated token ids).
Output: --out DIR (default $DRAFT_HEAD_DATA) gets train.pkl and heldout.pkl.


Held out = every row of the step-9 prompts (64 prompts never seen in training; steps 0-8 train). Rows are deduplicated by
(prompt idx, sample j, token hash): runs share seeds, so some steps hold bitwise identical rows.
Output (system python3 readable): {"prompts": {idx: array('i')}, "rows": [dict(idx, j, step, run, tokens=array('i'))]}
"""
import argparse
import array
import glob
import json
import os
import pickle
import random
import zlib



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="row-dump root: <src>/<run>/rows_step*.jsonl")
    ap.add_argument("--out", default=os.environ.get("DRAFT_HEAD_DATA", ""), help="output dir (default $DRAFT_HEAD_DATA)")
    ap.add_argument("--held_step", type=int, default=9, help="rows of this RL step's prompts are held out")
    ns = ap.parse_args()
    if not ns.out:
        raise SystemExit("pass --out or set DRAFT_HEAD_DATA")
    R, O, HELD_STEP = ns.src, ns.out, ns.held_step
    os.makedirs(O, exist_ok=True)
    tr, ho = dict(prompts={}, rows=[]), dict(prompts={}, rows=[])
    seen = set()
    for f in sorted(glob.glob(f"{R}/*/rows_step*.jsonl")):
        run = os.path.basename(os.path.dirname(f))
        for line in open(f):
            d = json.loads(line)
            tgt = ho if d["step"] == HELD_STEP else tr
            if d["prompt"] is not None:
                tgt["prompts"].setdefault(d["idx"], array.array("i", d["prompt"]))
            key = (d["idx"], d["j"], zlib.crc32(array.array("i", d["tokens"]).tobytes()))
            if key in seen:
                continue
            seen.add(key)
            tgt["rows"].append(dict(idx=d["idx"], j=d["j"], step=d["step"], run=run,
                                    tokens=array.array("i", d["tokens"])))
    assert not (set(tr["prompts"]) & set(ho["prompts"]))
    random.Random(0).shuffle(tr["rows"])
    for name, s in (("train", tr), ("heldout", ho)):
        missing = [r for r in s["rows"] if r["idx"] not in s["prompts"]]
        assert not missing, (name, len(missing))
        ntok = sum(len(r["tokens"]) for r in s["rows"])
        print(name, "rows", len(s["rows"]), "prompts", len(s["prompts"]), "tokens", ntok, flush=True)
        with open(os.path.join(O, f"{name}.pkl"), "wb") as fh:
            pickle.dump(s, fh, protocol=4)


if __name__ == "__main__":
    main()
