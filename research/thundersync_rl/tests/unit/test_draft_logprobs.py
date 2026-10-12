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
"""Engine log-probs accumulated alongside streamed draft tokens (worker side)."""

from types import SimpleNamespace

import torch
from thundersync_rl.worker import (
    apply_draft_frame,
    apply_draft_records,
    engine_draft_records,
    snapshot_drafts,
)


def _partial(toks, lps=None):
    part = {"request_id": 0, "new_tokens": toks}
    if lps is not None:
        part["new_log_probs"] = lps
    return {"partial": part}


def test_partial_frames_accumulate_logprobs_and_snapshot():
    tokens, logprobs, done = [[]], [[]], [False]
    assert not apply_draft_frame(_partial([4, 5], [-0.1, -0.2]), tokens[0], logprobs, 0)
    assert not apply_draft_frame(_partial([6], [-0.3]), tokens[0], logprobs, 0)
    snap = snapshot_drafts(tokens, logprobs, done)
    assert snap == [([4, 5, 6], False, [-0.1, -0.2, -0.3])]
    snap[0][2].append(0.0)  # a copy: later frames must not alias it
    assert logprobs[0] == [-0.1, -0.2, -0.3]


def test_partial_without_logprobs_drops_them():
    tokens, logprobs = [[]], [[]]
    apply_draft_frame(_partial([4], [-0.1]), tokens[0], logprobs, 0)
    apply_draft_frame(_partial([5]), tokens[0], logprobs, 0)
    apply_draft_frame(_partial([6], [-0.3]), tokens[0], logprobs, 0)
    assert tokens[0] == [4, 5, 6] and logprobs[0] is None
    assert snapshot_drafts(tokens, logprobs, [False]) == [([4, 5, 6], False, None)]


def test_final_frame_replaces_tokens_and_logprobs():
    tokens, logprobs = [[4]], [[-0.1]]
    final = {
        "generated_tokens": [4, 5, 2],
        "generated_log_probs": [-0.1, -0.2, -0.3],
    }
    assert apply_draft_frame({"final": final}, tokens[0], logprobs, 0)
    assert tokens[0] == [4, 5, 2] and logprobs[0] == [-0.1, -0.2, -0.3]
    # DynamicInferenceRequest-like final with a tensor of log-probs.
    req = SimpleNamespace(
        generated_tokens=[7, 2], generated_log_probs=torch.tensor([-0.5, -0.25])
    )
    tokens, logprobs = [[]], [[]]
    assert apply_draft_frame({"final": req}, tokens[0], logprobs, 0)
    assert logprobs[0] == [-0.5, -0.25]
    # Missing or misaligned final log-probs: None.
    tokens, logprobs = [[]], [[]]
    apply_draft_frame({"final": {"generated_tokens": [7, 2]}}, tokens[0], logprobs, 0)
    assert logprobs[0] is None


class _Record:
    """Megatron request record: one request per (re)admission, merged on read."""

    def __init__(self, *reqs):
        self.requests = list(reqs)

    def merge(self):
        return SimpleNamespace(
            generated_tokens=[t for r in self.requests for t in r.generated_tokens],
            generated_log_probs=[
                v for r in self.requests for v in r.generated_log_probs
            ],
        )


def test_collect_uses_merged_engine_record_after_eviction():
    # Request 7 generated [4, 5] before an eviction and [6, 8, 9] after it. The
    # resumed record's generated tokens restart at 0 while the emitted-length
    # counter stays at 2, so the frames skipped its first two tokens.
    pre = SimpleNamespace(generated_tokens=[4, 5], generated_log_probs=[-0.1, -0.2])
    post = SimpleNamespace(
        generated_tokens=[6, 8, 9], generated_log_probs=[-0.3, -0.4, -0.5]
    )
    engine = SimpleNamespace(requests={7: SimpleNamespace(record=_Record(pre, post))})
    records = engine_draft_records(engine, [7, 99])  # 99: held by another engine
    assert records == {7: ([4, 5, 6, 8, 9], [-0.1, -0.2, -0.3, -0.4, -0.5])}

    # Frames delivered [4, 5] then [9]: 6 and 8 were skipped.
    tokens, logprobs, done = [[], [1, 2]], [[], [-0.6, -0.7]], [False, True]
    apply_draft_frame(_partial([4, 5], [-0.1, -0.2]), tokens[0], logprobs, 0)
    apply_draft_frame(_partial([9], [-0.5]), tokens[0], logprobs, 0)
    streams = {0: SimpleNamespace(request_id=7), 1: SimpleNamespace(request_id=8)}
    stats = apply_draft_records(tokens, logprobs, done, streams, records)
    assert stats == {"from_record": 1, "fallback": 0, "differed": 1}
    assert snapshot_drafts(tokens, logprobs, done) == [
        ([4, 5, 6, 8, 9], False, [-0.1, -0.2, -0.3, -0.4, -0.5]),
        ([1, 2], True, [-0.6, -0.7]),  # finished: final reply already merged
    ]
    # No record: frames kept.
    tokens2, lps2 = [[4, 5, 9]], [[-0.1, -0.2, -0.5]]
    stats = apply_draft_records(tokens2, lps2, [False], {0: streams[0]}, {})
    assert stats["fallback"] == 1 and tokens2 == [[4, 5, 9]]
