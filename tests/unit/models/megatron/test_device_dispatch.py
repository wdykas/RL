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
"""CPU tests of the learner device dispatch's capacity and route math (device_dispatch.py).

The exchange's routes (``route_tables``) are checked against a reference AlltoAll +
chunk sort (owner rows: local expert major, source rank minor, source order within) at
small EP with top-k > local experts (the latent-MoE regime of Nemotron Super / Ultra),
and the receive capacity at Ultra's shape (EP 64, 512 experts, top-22) with simulated
routing counts.
"""

import types

import pytest
import torch

from nemo_rl.models.megatron import device_dispatch as dd


def _route(world, experts, topk, tokens, gen, skew=None):
    """Per source rank: [tokens, topk] distinct experts (uniform, or weighted by ``skew``)."""
    out = []
    for _ in range(world):
        if skew is None:
            scores = torch.rand(tokens, experts, generator=gen)
        else:
            # weighted sampling without replacement (Efraimidis-Spirakis keys u^(1/w))
            scores = torch.rand(tokens, experts, generator=gen).log() / skew
        out.append(scores.topk(topk, dim=1).indices)
    return out


def _counts(routes, experts):
    return torch.stack([torch.bincount(r.reshape(-1), minlength=experts) for r in routes]).to(torch.int64)


def test_receive_rows_formula():
    # Full Nemotron-3-Nano EP4 (top-6 <= 32 local experts): unchanged by the fix.
    assert dd.receive_rows(4, 128, 6, 20480, 2.0) == 2 * 20480 * 6 == 245760
    # Ultra EP64: 512 experts (8 local), top-22, T_cap 4096: c x the mean T_cap x top-k.
    assert dd.receive_rows(64, 512, 22, 4096, 2.0) == 2 * 4096 * 22 == 180224
    # The old bound c x T_cap x min(top-k, local experts) was below the mean (16 < 22 rows/token).
    old = 2 * 4096 * min(22, 512 // 64)
    assert old < 4096 * 22
    # Capped at the hard worst case (every peer sends every token's rows to one owner).
    assert dd.receive_rows(4, 16, 8, 100, 1e9) == 4 * 100 * min(8, 4)
    # c >= EP x min(k, ne) / k reaches the worst case exactly.
    assert dd.receive_rows(64, 512, 22, 4096, 64 * 8 / 22) == 64 * 4096 * 8


def test_rank_tokens_are_the_sequence_shard_of_a_call():
    """T_cap (NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS) is a call budget; a rank's MoE layer sees its sequence shard.

    Regression (11-05, Ultra-L27 XPR-S TP8 EP32 OOM in the verify forward): the workspace and every per-expert-row
    buffer were sized for T_cap tokens per rank, TP x the shard under sequence parallelism.
    """
    cfg = lambda tp, sp, cp=1: types.SimpleNamespace(  # noqa: E731
        tensor_model_parallel_size=tp, sequence_parallel=sp, context_parallel_size=cp
    )
    assert dd.token_split(cfg(8, True)) == 8
    assert dd.token_split(cfg(8, False)) == 1  # no sequence split: every TP rank holds the call
    assert dd.token_split(cfg(2, True, 2)) == 4
    assert dd.token_split(cfg(1, False, 4)) == 4
    assert dd.token_split(types.SimpleNamespace()) == 1
    assert dd.rank_tokens(18432, 8) == 2304
    assert dd.rank_tokens(10, 4) == 3  # ceil: a shard never exceeds it
    for t_call in range(1, 64):
        for split in (1, 2, 3, 8):
            if t_call % split == 0:  # sequence parallelism / CP pad the call to a multiple of the split
                assert t_call // split <= dd.rank_tokens(t_call, split)
    # Ultra-L27 XPR-S (TP8 + SP, EP32, 512 experts, routed slots 30 under the cross-fade band, latent 2048,
    # expert FFN 5120), T_cap 18432: 1,105,920 -> 138,240 receive rows; the rows x F fp32 expert activation
    # 21.09 -> 2.64 GiB.
    old = dd.receive_rows(32, 512, 30, 18432, 2.0)
    new = dd.receive_rows(32, 512, 30, dd.rank_tokens(18432, dd.token_split(cfg(8, True))), 2.0)
    assert (old, new) == (1105920, 138240)
    assert round(old * 5120 * 4 / 2**30, 2) == 21.09 and round(new * 5120 * 4 / 2**30, 2) == 2.64
    # Super-L22 (TP2 + SP, EP8), T_cap 20480: 1,228,800 -> 614,400.
    assert dd.receive_rows(8, 512, 30, dd.rank_tokens(20480, 2), 2.0) == 614400
    # The workspace key carries the split (models with different layouts never share a mis-sized workspace).
    d = types.SimpleNamespace(
        ep_group=types.SimpleNamespace(group_name="ep"), num_experts=512,
        config=types.SimpleNamespace(moe_router_topk=22, moe_latent_size=2048, hidden_size=8192,
                                     tensor_model_parallel_size=8, sequence_parallel=True, context_parallel_size=1),
    )
    assert dd._ws_key(d) == ("ep", 512, 22, 2048, 8, 22)  # rows/token: top-k (no sparse cross-fade)


def test_dispatch_width_is_latent_for_latent_moe():
    assert dd.dispatch_width(types.SimpleNamespace(hidden_size=8192, moe_latent_size=2048)) == 2048
    assert dd.dispatch_width(types.SimpleNamespace(hidden_size=2688, moe_latent_size=None)) == 2688
    assert dd.dispatch_width(types.SimpleNamespace(hidden_size=2688)) == 2688


def _reference_owner_rows(routes, counts, world, experts):
    """For each owner: list of (src, src_perm1_row) in AlltoAll + chunk-sort order."""
    ne = experts // world
    owners = []
    src_start = counts.cumsum(1) - counts
    for o in range(world):
        rows = []
        for e_local in range(ne):
            e = o * ne + e_local
            for s in range(world):
                for j in range(int(counts[s, e])):
                    rows.append((s, int(src_start[s, e]) + j))
        owners.append(rows)
    return owners


@pytest.mark.parametrize("world,experts,topk,tokens", [(2, 8, 6, 37), (4, 32, 12, 53), (8, 64, 22, 19)])
def test_route_tables_match_alltoall_order(world, experts, topk, tokens):
    gen = torch.Generator().manual_seed(world * 100 + topk)
    # Ranks hold different token counts (no shape agreement).
    routes = [r[: tokens + 3 * i] for i, r in enumerate(_route(world, experts, topk, tokens + 3 * world, gen))]
    C = _counts(routes, experts)
    ref = _reference_owner_rows(routes, C, world, experts)
    r_cap = max(len(x) for x in ref)  # exactly enough: no overflow
    received = [[None] * r_cap for _ in range(world)]
    for me in range(world):
        n = int(C[me].sum())
        t = dd.route_tables(C, me, n, r_cap)
        assert not bool(t["overflow"])
        for row in range(n):  # forward: my row -> (owner, owner row)
            received[int(t["fwd_rank"][row])][int(t["fwd_pos"][row])] = (me, row)
    for o in range(world):
        t = dd.route_tables(C, o, int(C[o].sum()), r_cap)
        live = int(t["live"])
        assert live == len(ref[o])
        assert received[o][:live] == ref[o]
        # backward / combine: received row -> (source, source row) is the inverse map
        back = [(int(t["back_rank"][r]), int(t["back_pos"][r])) for r in range(live)]
        assert back == ref[o]
        ne = experts // world
        assert t["tokens_per_expert"].tolist() == [int(C[:, o * ne + e].sum()) for e in range(ne)]


def test_overflow_is_flagged_on_every_rank_and_memory_safe():
    world, experts, topk = 4, 32, 12
    gen = torch.Generator().manual_seed(7)
    C = _counts(_route(world, experts, topk, 40, gen), experts)
    need = int(C.view(world, world, -1).sum(-1).sum(0).max())  # largest owner total
    for me in range(world):
        t = dd.route_tables(C, me, int(C[me].sum()), need - 1)
        assert bool(t["overflow"])  # same decision on every rank
        assert int(t["live"]) <= need - 1
        assert int(t["tokens_per_expert"].sum()) <= need - 1
        live = int(t["live"])  # rows past live are never pushed
        src_rows = C.sum(1)[t["back_rank"][:live]]
        assert bool((t["back_pos"][:live] < src_rows).all())


def _owner_totals(C, world):
    return C.view(world, world, -1).sum(-1).sum(0)  # [owner]


def test_ultra_ep64_capacity_overflow_is_not_a_per_call_event():
    """Ultra shape: EP 64, 512 experts (8 local), top-22, every rank at T_cap.

    Old bound (2 x T_cap x min(22, 8) = 16 T_cap): below the mean owner load 22 T_cap ->
    overflow on every call. New bound (2 x 22 T_cap): no overflow under uniform routing,
    and none under a moderately skewed router either (largest owner < 2x the mean);
    an overflow needs an owner above c x the mean.
    """
    world, experts, topk, t_cap = 64, 512, 22, 256
    gen = torch.Generator().manual_seed(0)
    old = 2 * t_cap * min(topk, experts // world)
    new = dd.receive_rows(world, experts, topk, t_cap, 2.0)
    mean = t_cap * topk
    for trial in range(3):
        C = _counts(_route(world, experts, topk, t_cap, gen), experts)
        tot = _owner_totals(C, world)
        assert int(tot.sum()) == world * mean
        assert int(tot.max()) > old  # every owner above the old bound, in fact
        assert int(tot.min()) > old
        assert int(tot.max()) <= new
        assert not bool(dd.route_tables(C, trial, int(C[trial].sum()), new)["overflow"])
    # Skewed router: expert popularity weights in [1, 3] (hot experts ~3x cold ones).
    w = 1 + 2 * torch.rand(experts, generator=gen)
    C = _counts(_route(world, experts, topk, t_cap, gen, skew=w), experts)
    tot = _owner_totals(C, world)
    assert int(tot.max()) / mean < 2.0
    assert not bool(dd.route_tables(C, 0, int(C[0].sum()), new)["overflow"])
    # Pathological: all tokens of every rank to owner 0's 8 experts (+ 14 elsewhere):
    # owner 0 gets 8 T_cap per source = 64 x 8 T_cap > 2 x 22 T_cap -> flagged (rerun on Megatron).
    C = torch.zeros(world, experts, dtype=torch.int64)
    C[:, :8] = t_cap
    C[:, 8:22] = t_cap
    assert bool(dd.route_tables(C, 5, int(C[5].sum()), new)["overflow"])


def test_call_cap_overflow_same_decision_on_every_rank(monkeypatch):
    # Per-call caps: each owner's cap travels with its counts; every rank computes the same
    # overflow flag from the [EP, E] table + caps (an owner over its own cap -> rerun).
    import torch

    monkeypatch.setenv("NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP", "1")
    monkeypatch.setenv("NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP_MIN", "1")
    W, E = 4, 8
    g = torch.Generator().manual_seed(0)
    C = torch.randint(0, 20, (W, E), generator=g).to(torch.int64)
    r_cap = 10_000
    caps = torch.tensor([dd.call_rows(r_cap, int(C[r].sum()), 2.0) for r in range(W)])
    flags = {bool(dd.route_tables(C, me, int(C[me].sum()), int(caps[me]), caps)["overflow"]) for me in range(W)}
    assert len(flags) == 1 and flags == {False}  # 2x own rows covers a balanced draw here
    caps2 = caps.clone()
    caps2[2] = 1  # owner 2 cannot hold what it receives
    flags = {bool(dd.route_tables(C, me, int(C[me].sum()), int(caps2[me]), caps2)["overflow"]) for me in range(W)}
    assert flags == {True}
    # Off: R_cap rows whatever the call.
    monkeypatch.setenv("NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP", "0")
    assert dd.call_rows(r_cap, 5, 2.0) == r_cap


def test_call_rows_floor_and_ceiling(monkeypatch):
    monkeypatch.setenv("NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP", "1")
    monkeypatch.delenv("NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP_MIN", raising=False)
    assert dd.call_rows(8000, 10, 2.0) == 1000  # floor R_cap / 8
    assert dd.call_rows(8000, 3000, 2.0) == 6000
    assert dd.call_rows(8000, 9000, 2.0) == 8000  # never above R_cap


def test_call_rows_nograd_floor(monkeypatch):
    """CALL_CAP_NOGRAD raises the floor of no-grad calls only (verification), never above R_cap; grad calls keep
    the common floor; 0 / unset: unchanged."""
    monkeypatch.setenv("NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP", "1")
    monkeypatch.delenv("NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP_MIN", raising=False)
    monkeypatch.delenv("NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP_NOGRAD", raising=False)
    with torch.no_grad():
        assert dd.call_rows(8000, 10, 2.0) == 1000
    monkeypatch.setenv("NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP_NOGRAD", "0.5")
    with torch.no_grad():
        assert dd.call_rows(8000, 10, 2.0) == 4000
        assert dd.call_rows(8000, 3000, 2.0) == 6000  # 2x own rows above the floor
        assert dd.call_rows(8000, 9000, 2.0) == 8000
    with torch.enable_grad():
        assert dd.call_rows(8000, 10, 2.0) == 1000  # training calls: common floor
    monkeypatch.setenv("NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP_NOGRAD", "2")
    with torch.no_grad():
        assert dd.call_rows(8000, 10, 2.0) == 8000
