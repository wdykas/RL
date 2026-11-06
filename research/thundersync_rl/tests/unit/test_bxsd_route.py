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
"""CPU tests: BX-SD lane routing hook on Megatron's DP inference coordinator (thundersync_rl.bxsd_route).

The stub tests need no Megatron. ``test_real_handler_*`` drive MCore's own SUBMIT_REQUEST handler with a stub
coordinator (skipped where megatron.core / msgpack are not importable).
"""

import pickle

import pytest

from thundersync_rl import bxsd_route


class _Coord:
    def __init__(self, n):
        self.identity_to_rank_index = {bxsd_route.identity_of(r): r for r in range(n)}
        self.stock_calls = 0

    def get_best_data_parallel_rank(self, *_a, **_k):
        self.stock_calls += 1
        return bxsd_route.identity_of(0)


def _stock(coord, sender, metadata, bodies):
    return coord.get_best_data_parallel_rank([], media_cache_key=None)


def _md(sp):
    return [20, 7, sp, None]


def test_target_of():
    assert bxsd_route.target_of(_md({"ts_dp_rank": 5})) == 5
    assert bxsd_route.target_of(_md({"temperature": 1.0})) == -1
    assert bxsd_route.target_of(_md({"ts_dp_rank": None})) == -1
    assert bxsd_route.target_of([20]) == -1
    assert bxsd_route.target_of(_md("not a dict")) == -1


def test_routed_to_target_and_restored():
    c = _Coord(16)
    h = bxsd_route.routed(_stock)
    assert h(c, b"cli", _md({"ts_dp_rank": 11}), []) == bxsd_route.identity_of(11)
    assert "get_best_data_parallel_rank" not in c.__dict__  # the per-call shadow is removed
    assert h(c, b"cli", _md({}), []) == bxsd_route.identity_of(0) and c.stock_calls == 1


def test_unknown_target_falls_back_and_counts():
    c = _Coord(4)
    before = bxsd_route.STATS["misrouted"]
    h = bxsd_route.routed(_stock)
    assert h(c, b"cli", _md({"ts_dp_rank": 9}), []) == bxsd_route.identity_of(0)
    assert bxsd_route.STATS["misrouted"] == before + 1


def test_shadow_removed_on_exception():
    c = _Coord(4)

    def boom(*_a):
        raise ValueError("x")

    with pytest.raises(ValueError):
        bxsd_route.routed(boom)(c, b"cli", _md({"ts_dp_rank": 2}), [])
    assert "get_best_data_parallel_rank" not in c.__dict__


def test_entrypoint_pickles_by_reference():
    # The spawned coordinator receives the target by reference: it must be this module's function.
    f = pickle.loads(pickle.dumps(bxsd_route._entrypoint))
    assert f is bxsd_route._entrypoint


def _mcore():
    msgpack = pytest.importorskip("msgpack")
    pytest.importorskip("megatron.core.inference.data_parallel_inference_coordinator.handlers")
    from megatron.core.inference.data_parallel_inference_coordinator import handlers as H
    from megatron.core.inference.headers import Headers

    return msgpack, H, Headers


class _RealCoord:
    """The attributes MCore's handle_submit_request reads / writes (no prefix caching, no media)."""

    def __init__(self, n):
        import numpy as np

        self.known_clients = {b"cli"}
        self.next_request_id = 0
        self.request_id_to_client_id, self.request_id_to_client_request_id = {}, {}
        self.client_request_to_request_id, self.request_id_to_rank = {}, {}
        self.identities_of_data_parallel_ranks = [bxsd_route.identity_of(r) for r in range(n)]
        self.identity_to_rank_index = {bxsd_route.identity_of(r): r for r in range(n)}
        self._identities_list = list(self.identities_of_data_parallel_ranks)
        self._pending_counts = np.zeros(n, dtype=np.int32)
        self.enable_prefix_caching = False
        self.block_size_tokens = None
        self.prefix_caching_coordinator_policy = None
        self.vision_embedding_cache_enabled = False
        self.media_cache_coordinator_policy = None
        self.schedule_records = None
        self.sent = []

    def get_best_data_parallel_rank(self, request_hashes, media_cache_key=None):
        import numpy as np

        return self._identities_list[int(np.argmin(self._pending_counts))]

    def _send_to_engine(self, identity, frames):
        self.sent.append(identity)
        return True

    def _update_rank_hashes(self, *a):
        pass


def test_real_handler_routes_and_books():
    msgpack, H, Headers = _mcore()
    c = _RealCoord(16)
    h = bxsd_route.routed(H.HANDLERS[Headers.SUBMIT_REQUEST])
    bodies = [msgpack.packb([1, 2, 3]), msgpack.packb(None), None, None]
    for i, tgt in enumerate([9, 9, 3, -1, -1]):
        md = [Headers.SUBMIT_REQUEST.value, i, {"ts_dp_rank": tgt, "temperature": 1.0}, None]
        h(c, b"cli", md, bodies)
    assert c.sent[:3] == [bxsd_route.identity_of(r) for r in (9, 9, 3)]
    # untargeted requests take the stock least-loaded choice, which sees the routed rows' pending counts
    assert c.sent[3:] == [bxsd_route.identity_of(0), bxsd_route.identity_of(1)]
    assert int(c._pending_counts[9]) == 2 and int(c._pending_counts[3]) == 1
    assert c.request_id_to_rank[0] == bxsd_route.identity_of(9)


def test_patch_handlers_idempotent():
    _, H, Headers = _mcore()
    orig = H.HANDLERS[Headers.SUBMIT_REQUEST]
    try:
        bxsd_route.patch_handlers()
        once = H.HANDLERS[Headers.SUBMIT_REQUEST]
        bxsd_route.patch_handlers()
        assert H.HANDLERS[Headers.SUBMIT_REQUEST] is once and once.__wrapped__ is orig
    finally:
        H.HANDLERS[Headers.SUBMIT_REQUEST] = orig
