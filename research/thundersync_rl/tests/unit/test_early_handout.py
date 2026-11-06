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
"""Early draft handout (grpo_loop ``_EARLY_HANDOUT_POOL``, THUNDERSYNC_EARLY_HANDOUT): the deferred cut's handout runs
on a worker thread right after the cut; the next step's top reads its result. The handout adds route blocks for a
later target while the step's end frees the trained step's blocks (``release_routes``) on the main thread."""

import threading
import time
from collections import defaultdict
from types import SimpleNamespace

from thundersync_rl import grpo_loop
from thundersync_rl.speculative import SpeculativeGeneration


def test_handout_future_returns_the_handout_and_raises_its_error():
    f = grpo_loop._EARLY_HANDOUT_POOL.submit(lambda: [("drafts", 3)]).result
    assert f() == [("drafts", 3)]

    def boom():
        raise ValueError("pop failed")

    g = grpo_loop._EARLY_HANDOUT_POOL.submit(boom).result
    try:
        g()
    except ValueError as e:
        assert "pop failed" in str(e)
    else:
        raise AssertionError("the handout's error must reach the step top")


def test_release_routes_tolerates_a_concurrent_handout():
    freed = []
    spec = SimpleNamespace(_route_blocks=defaultdict(list), _anchor_claims={}, _free_refs=freed.extend)
    stop = threading.Event()

    def handout():  # the early handout adds blocks for later targets meanwhile (bounded set of targets)
        t = 0
        while not stop.is_set():
            spec._route_blocks[10 + t % 50] += [("ref", t)]
            t += 1
            if t % 1000 == 0:
                time.sleep(0)

    th = threading.Thread(target=handout)
    th.start()
    try:
        for step in range(2000):
            spec._route_blocks[step % 5] += [("old", step)]
            SpeculativeGeneration.release_routes(spec, step % 5)
    finally:
        stop.set()
        th.join()
    assert all(k >= 10 for k in spec._route_blocks)  # only later targets remain
