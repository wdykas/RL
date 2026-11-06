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
"""Chunked prefill: the key checks skip the discarded partial-prefill draw (from ts-colo 08e9f53e's
test_ab_async2_stock_config.py; that file's other tests cover pruned code)."""

from types import SimpleNamespace

import pytest


def test_key_checks_skip_the_discarded_partial_prefill_draw():
    """Chunked prefill: MCore samples (then discards) the row of a request whose prompt is not done; the key checks
    skip that row, so the final chunk's kept draw at the same position is not a reuse (was: chunked prefill refused
    with the keyed samplers, async2_stock Super-L22)."""
    import numpy as np
    import pytest
    from types import SimpleNamespace

    worker = pytest.importorskip("thundersync_rl.worker")
    from thundersync_rl.key_check import KeyContinuity

    kc = KeyContinuity("test")
    ids = np.array([7, 9], dtype=np.int64)
    # step 1: request 9 decodes at 20; request 7 prefills its first chunk (prompt length 300: position 300)
    pos = worker._key_check_positions(SimpleNamespace(chunked_prefill_request_id=7), ids, np.array([300, 20]))
    assert pos.tolist() == [-1, 20]
    kc.observe(ids, pos)
    # step 2: request 7's final chunk draws position 300 again, kept; request 9 continues
    pos = worker._key_check_positions(SimpleNamespace(chunked_prefill_request_id=-1), ids, np.array([300, 21]))
    assert pos.tolist() == [300, 21]
    kc.observe(ids, pos)
    kc.observe(ids, np.array([301, 22]))
    assert kc.violations == 0
    # a real reuse still raises
    with pytest.raises(Exception, match="out of sequence"):
        kc.observe(ids, np.array([301, 23]))


def test_hidden_log_skips_the_partial_prefill_chunk_row():
    import numpy as np
    import pytest
    from types import SimpleNamespace

    worker = pytest.importorskip("thundersync_rl.worker")
    # Rows a..e of the context: request 7 is mid chunked prefill (its draw is discarded): not logged.
    rids = np.array([5, 6, 7, 8])
    tag = np.array([1, 1, 1, 0])
    ctx = SimpleNamespace(chunked_prefill_request_id=7)
    assert worker._hidden_log_rows(ctx, tag, rids, 1, 4) == [0]
    assert worker._hidden_log_rows(SimpleNamespace(chunked_prefill_request_id=-1), tag, rids, 1, 4) == [0, 1]
    assert worker._hidden_log_rows(SimpleNamespace(), tag, rids, 0, 4) == [0, 1, 2]
