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
"""Learner device dispatch under sparse cross-fade routing: row arrays sized by live rows (stock k per token), not
by the C = k + extra candidate slots. CPU: sizing (Nano / Ultra per-rank rows), fixed-width routing unchanged, a
source routing more live rows than its return arrays raises the overflow flag on every rank. The 4-GPU exchange
(bitwise equal to the C-sized workspace, overflow memory safe) is test_dd_live_rows_gpu.py."""

import torch

from nemo_rl.models.megatron import crossfade_routing as cf
from nemo_rl.models.megatron import device_dispatch as dd


class _Sparse:
    def __init__(self, k=6, extra=8, sparse=True, m=0.01):
        self.new = dict(m=m, k=k, extra=extra, sparse=sparse)

    def __enter__(self):
        self.old = cf.get_state()
        cf._STATE.update(self.new)

    def __exit__(self, *a):
        cf._STATE.clear()
        cf._STATE.update(self.old)


def test_row_slots_are_the_stock_k_only_for_sparse_crossfade():
    with _Sparse(sparse=True):
        assert dd.row_slots(14) == 6
    with _Sparse(sparse=False):
        assert dd.row_slots(14) == 14  # fixed width: every candidate slot is a row
    with _Sparse(m=None):
        assert dd.row_slots(6) == 6


def test_receive_rows_by_live_rows_nano_and_ultra():
    # Nano 16k EP8: T_rank 20480, k 6, C 14, 128 experts.
    assert dd.receive_rows(8, 128, 14, 20480, 2.0) == 573440
    assert dd.receive_rows(8, 128, 14, 20480, 2.0, row_k=6) == 245760
    assert dd.receive_rows(8, 128, 6, 20480, 2.0) == 245760  # plain top-6: unchanged
    # Ultra-L27 TP8 EP32 (T_cap 18432 / TP8 sequence split): k 22, extra 8, 512 experts.
    assert dd.receive_rows(32, 512, 30, 2304, 2.0) == 138240
    assert dd.receive_rows(32, 512, 30, 2304, 2.0, row_k=22) == 101376
    # The hard worst case still caps it.
    assert dd.receive_rows(2, 4, 14, 100, 2.0, row_k=6) == 2 * 100 * 2


def _sparse_counts(world, experts, k, C, tokens, gen, all_live_rank=None):
    """[world, E] rows per source and expert: every token routes k + (0..C-k) distinct experts."""
    out = torch.zeros(world, experts, dtype=torch.int64)
    for s in range(world):
        for _ in range(tokens):
            n = C if s == all_live_rank else k + int(torch.randint(0, C - k + 1, (1,), generator=gen)) // 3
            e = torch.randperm(experts, generator=gen)[:n]
            out[s, e] += 1
    return out


def test_source_over_its_return_rows_overflows_on_every_rank():
    world, experts, k, C, T = 4, 32, 2, 6, 40
    gen = torch.Generator().manual_seed(3)
    r_cap = dd.receive_rows(world, experts, C, T, 2.0, row_k=k)
    s_cap = int(min(T * C, 2.0 * T * k))
    ok = _sparse_counts(world, experts, k, C, T, gen)
    assert int(ok.sum(1).max()) <= s_cap
    bad = _sparse_counts(world, experts, k, C, T, gen, all_live_rank=1)  # rank 1: every candidate live, 240 > 160
    assert int(bad[1].sum()) > s_cap
    for me in range(world):
        n_rows = T * C  # the static permutation (padding past the live rows)
        t_ok = dd.route_tables(ok, me, n_rows, r_cap, s_cap=s_cap)
        t_old = dd.route_tables(ok, me, n_rows, r_cap)
        assert not bool(t_ok["overflow"]) and not bool(t_old["overflow"])
        for key in ("fwd_rank", "fwd_pos", "back_rank", "back_pos", "live", "tokens_per_expert"):
            assert torch.equal(t_ok[key], t_old[key]), key  # the source bound changes no route
        t_bad = dd.route_tables(bad, me, n_rows, 10**9, s_cap=s_cap)
        assert bool(t_bad["overflow"]), me  # the same decision on every rank
        assert not bool(dd.route_tables(bad, me, n_rows, 10**9)["overflow"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
