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
"""Discrete-event model of one streamed step under pipeline parallelism.

Uses the real ``StreamPlanner`` and ``group_pipeline_calls`` (group
granularity, ``max_open_groups=0``). Rollouts finish at lognormal times
(``spread`` scales them relative to the learner's total work); the learner
takes a dispatch whenever it is idle and the planner has (non-held) work. A
pipeline call of ``m`` micro-batches costs ``m + PP - 1`` slots (1F1B, uniform
micro-batch cost); a dispatch costs its busiest rank's slots. Reports the
bubble fraction of learner busy time, calls per rank, the tail (last arrival
to learner done) and the step time, in micro-batch slots. No GPU, no Ray.

    python research/thundersync_rl/scripts/pp_streaming_sim.py
"""

import random

from thundersync_rl.streaming import StreamPlanner, Trajectory
from thundersync_rl.worker import group_pipeline_calls


def _loo(r):
    m = len(r)
    return [x - (sum(r) - x) / (m - 1) for x in r]


def simulate(
    pp, merge, min_mb, *, n=512, G=16, dp=8, mbs=1, chunk=8, seed=0, spread=0.8
):
    rng = random.Random(seed)
    work = n / dp  # micro-batches per rank per step (mbs=1)
    times = sorted((rng.lognormvariate(0, 0.6) * work * spread, i) for i in range(n))
    planner = StreamPlanner(
        _loo,
        dp_size=dp,
        max_chunk_trajectories=chunk,
        max_open_groups=0,
        micro_batch_size=mbs,
        equal_rank_microbatches=False,
        min_pipeline_microbatches=min_mb,
    )
    for g in range(n // G):
        planner.register_group(g, G)
    t = busy = bubble = 0.0
    calls_total = k = 0
    while True:
        while k < n and times[k][0] <= t:
            i = times[k][1]
            planner.add(
                Trajectory(
                    group=i // G,
                    index=i,
                    reward=float(rng.random() < 0.5),
                    payload=None,
                    num_tokens=rng.randint(100, 4000),
                )
            )
            k += 1
        done = k == n
        if planner.has_work(flush=done):
            d = planner.next_dispatch(flush=done)
            worst = (0, 0, 0)  # (slots, busy slots, calls) of the busiest rank
            for chunks in d.per_rank:
                items = [{"group": c.group, "reward": c.reward} for c in chunks]
                calls = group_pipeline_calls(items, merge=merge)
                ms = [sum(chunks[i].num_microbatches(mbs) for i in c) for c in calls]
                worst = max(worst, (sum(m + pp - 1 for m in ms), sum(ms), len(ms)))
            t += worst[0]
            busy += worst[0]
            bubble += worst[0] - worst[1]
            calls_total += worst[2]
        elif done:
            break
        else:
            t = times[k][0]
    assert planner.all_done()
    return dict(bubble=bubble / busy, calls=calls_total, tail=t - times[-1][0], step=t)


def main():
    for spread in (0.2, 0.5, 0.8, 1.5):
        for pp in (4, 8, 16):
            for name, merge, min_mb in (
                ("per-chunk", False, 0),
                ("merged", True, 0),
                ("merged+hold2P", True, 2 * pp),
                ("merged+hold4P", True, 4 * pp),
            ):
                rs = [
                    simulate(pp, merge, min_mb, seed=s, spread=spread) for s in range(8)
                ]
                avg = {key: sum(r[key] for r in rs) / len(rs) for key in rs[0]}
                print(
                    f"spread={spread} PP={pp:2d} {name:14s} "
                    f"bubble={avg['bubble'] * 100:5.1f}% calls/rank={avg['calls']:5.1f} "
                    f"tail={avg['tail']:6.1f} step={avg['step']:6.1f} slots",
                    flush=True,
                )


if __name__ == "__main__":
    main()
