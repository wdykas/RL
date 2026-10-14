#!/usr/bin/env python
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Who holds GPU memory at the peak of a recorded allocator history.

    python mem_snapshot_report.py rank0_peak.pickle [--top 25] [--depth 2]

Replays the trace (alloc / free_completed) of a ``torch.cuda.memory._dump_snapshot``
pickle (thundersync_rl.mem_debug), finds the peak of live bytes allocated inside the
trace window, and groups the blocks live at the peak by their innermost frames in
project code (nemo_rl / megatron / thundersync_rl / transformer_engine / flash_attn).
"""

import argparse
import collections
import pickle

KEEP = ("nemo_rl", "megatron", "thundersync_rl", "transformer_engine", "flash_attn")


def site(frames, depth):
    picked = [
        f"{f['filename'].split('/')[-1]}:{f['line']}:{f['name']}"
        for f in frames
        if any(k in f["filename"] for k in KEEP)
    ]
    return " <- ".join(picked[:depth]) or "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pickle")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--depth", type=int, default=2)
    a = ap.parse_args()
    snap = pickle.load(open(a.pickle, "rb"))
    trace = max(snap["device_traces"], key=len)
    live, cur, peak, peak_live = {}, 0, 0, {}
    for ev in trace:
        act = ev["action"]
        if act == "alloc":
            live[ev["addr"]] = ev
            cur += ev["size"]
            if cur > peak:
                peak, peak_live = cur, dict(live)
        elif act == "free_completed" and ev["addr"] in live:
            cur -= live.pop(ev["addr"])["size"]
    print(f"events={len(trace)} peak live (trace window)={peak / 2**30:.2f} GiB, "
          f"blocks={len(peak_live)}")
    groups = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    for ev in peak_live.values():
        g = groups[site(ev["frames"], a.depth)]
        g[0] += ev["size"]
        g[1] += 1
        g[2][ev["size"]] += 1
    for k, (size, n, sizes) in sorted(groups.items(), key=lambda kv: -kv[1][0])[: a.top]:
        common = ", ".join(f"{s / 2**20:.1f}MiBx{c}" for s, c in sizes.most_common(2))
        print(f"{size / 2**30:8.2f} GiB {n:6d} blk  [{common}]  {k}")


if __name__ == "__main__":
    main()
