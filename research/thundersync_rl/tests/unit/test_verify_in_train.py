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
"""CPU tests of verify-in-train (vit.py): the decision is bitwise the verifier's,
a pass is exactly the regular pool outcome "first G drafts whole", a failed
micro-batch leaves zero gradient, and the planner skips trained groups."""

import itertools
import math

import pytest
import torch

from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from test_null_bv import _model, _rows
from thundersync_rl import vit
from thundersync_rl.block_verification import acceptance_weights
from thundersync_rl.engine_q import verify_engine_q_grid
from thundersync_rl.null_bv import NullGroup, NullPolicy, decide_pools, row_envelopes
from thundersync_rl.streaming import StreamPlanner, Trajectory


@pytest.mark.parametrize("m", [1.0, 1.3, 2.0])
def test_whole_weight_is_the_verifiers(m):
    g = torch.Generator().manual_seed(0)
    for n in (1, 5, 40):
        lpx = -torch.rand(n, generator=g) * 3
        lqx = -torch.rand(n, generator=g) * 3
        assert torch.equal(vit.whole_weight(lpx, lqx, m), acceptance_weights(lpx - lqx, m)[-1])
        # The grid scan's envelope row (block_verify_stats_lazy_grid).
        s = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum((lpx - lqx).double(), 0)])
        S = s.unsqueeze(0).repeat(2, 1)
        if m != 1.0:
            S[1, 1:] -= math.log(m)
        B = (S - torch.cummax(S, 1).values).exp()
        assert torch.equal(vit.whole_weight(lpx, lqx, m), B[1 if m != 1.0 else 0, -1])


def test_vit_decision_equals_grid_whole_outcome():
    """eta_last(row seed) <= whole_weight  <=>  the grid scan with the row's own
    generator stops at tau = g at that envelope (same draws, same weights)."""
    V, L = 3, 4
    P, Q = _model(V, L, 3, conc=0.7), _model(V, L, 4, conc=0.7)
    Ms = (1.0, 1.5, 3.0)
    seen = set()
    for t, x in enumerate(itertools.product(range(V), repeat=L)):
        lp, lq = _rows(P, x, L + 1), _rows(Q, x, L)
        xt = torch.tensor(x)
        lqx = lq[torch.arange(L), xt]
        for k in range(3):
            seed = vit.row_seed(7, f"r{t}.{k}")
            grid, lpx, _ = verify_engine_q_grid(
                lp, xt, lqx, lambda i: lq[i], vit.row_generator(seed, "cpu"), Ms
            )
            eta = vit.eta_last(seed, L, "cpu")
            for M in Ms:
                whole = bool(eta <= vit.whole_weight(lpx, lqx, M))
                assert whole == (grid[M][0] == L), (x, M)
                seen.add(whole)
    assert seen == {True, False}  # both outcomes exercised


def test_pass_is_exactly_the_pool_outcome_first_g_whole():
    """Over many pools: when every candidate passes at its assumed envelope, the
    regular decide_pools walk takes rows 0..G-1 at those envelopes, whole."""
    V, L, N, G = 3, 3, 5, 3
    P, Q = _model(V, L, 11, conc=2.0), _model(V, L, 12, conc=2.0)
    pol = NullPolicy(fixed_m=None)
    gen = torch.Generator().manual_seed(1)
    n_pass = 0
    for trial in range(300):
        xs = [tuple(int(v) for v in torch.randint(0, V, (L,), generator=gen)) for _ in range(N)]
        seeds = [vit.row_seed(trial, f"d{j}") for j in range(N)]
        ms = [float(pol.m(N - j, G - j)) for j in range(G)]
        grids, lpxs, lqxs = [], [], []
        for j, x in enumerate(xs):
            lp, lq = _rows(P, x, L + 1), _rows(Q, x, L)
            xt = torch.tensor(x)
            lqx = lq[torch.arange(L), xt]
            grid, lpx, _ = verify_engine_q_grid(
                lp, xt, lqx, lambda i, lq=lq: lq[i], vit.row_generator(seeds[j], "cpu"),
                row_envelopes(pol, N, G, j),
            )
            grids.append(grid)
            lpxs.append(lpx)
            lqxs.append(lqx)
        passed = all(
            bool(vit.eta_last(seeds[j], L, "cpu") <= vit.whole_weight(lpxs[j], lqxs[j], ms[j]))
            for j in range(G)
        )
        outcomes = [{M: r[1] >= 0 for M, r in grid.items()} for grid in grids]
        chosen = decide_pools(["a"] * N, {"a": G}, outcomes, pol)
        first_g_whole = chosen[:G] == ms and all(
            grids[j][ms[j]][0] == L for j in range(G)
        ) and all(c is None for c in chosen[G:])
        assert passed == first_g_whole
        n_pass += passed
    assert 0 < n_pass < 300


def _vit_batch(fail_row: int | None):
    torch.manual_seed(0)
    b, s, g, start = 2, 8, 4, 3
    data = BatchedDataDict(
        {
            "input_ids": torch.randint(0, 50, (b, s)),
            "token_mask": torch.zeros(b, s),
            "sample_mask": torch.ones(b),
            "advantages": torch.tensor([[1.0] * s, [-1.0] * s]),
            "generation_logprobs": torch.zeros(b, s),
            "prev_logprobs": torch.zeros(b, s),
            "reference_policy_logprobs": torch.zeros(b, s),
        }
    )
    data["token_mask"][:, start : start + g] = 1
    lp = -torch.rand(b, s - 1) - 0.1
    rows = []
    for r in range(b):
        lpx = lp[r, start - 1 : start - 1 + g]
        # q above p -> b_g small (reject likely); q == p -> b_g = 1 (always whole)
        logq = (lpx + (3.0 if r == fail_row else 0.0)).tolist()
        rows.append({"start": start, "len": g, "logq": logq, "seed": 5 + r, "m": 1.0, "row": r})
    vit.add_vit_fields(data, rows)
    data["vit_eta"] = torch.stack([vit.eta_last(5 + r, g, "cpu") for r in range(b)])
    return data, lp


@pytest.mark.parametrize("fail_row", [None, 1])
def test_decision_hook_zeroes_failed_micro_batch(fail_row):
    data, lp = _vit_batch(fail_row)
    loss_fn = ClippedPGLossFn(ClippedPGLossConfig(reference_policy_kl_penalty=0.0, force_on_policy_ratio=True))
    lp = lp.clone().requires_grad_(True)
    results = []
    with vit.decision_hook(results):
        loss, _ = loss_fn(
            next_token_logprobs=lp, data=data,
            global_valid_seqs=torch.tensor(2.0), global_valid_toks=torch.tensor(8.0),
        )
    loss.backward()
    assert len(results) == 1
    r = results[0]
    assert [row for row, _, _ in r["rows"]] == [0, 1]
    assert r["rows"][0][1] is True  # q == p: b_g = 1 >= eta
    if fail_row is None:
        assert r["pass"] and lp.grad.abs().sum() > 0
    else:
        # exp(-12) acceptance: rejected for this seed
        assert not r["pass"]
        assert torch.equal(lp.grad, torch.zeros_like(lp.grad))
    assert r["rows"][0][2] == lp[0, 2:6].tolist()


def test_item_counts_match_prepare_train_piece():
    data, _ = _vit_batch(None)
    seqs, toks = vit.item_counts(data)
    assert float(seqs) == 2.0 and float(toks) == 8.0 and seqs.dtype == torch.float64


def _traj(p, i, r):
    return Trajectory(group=(0, p), index=i, reward=r, payload=None, num_tokens=10)


def test_planner_skips_pretrained_groups():
    pl = StreamPlanner(
        lambda rs: [x - sum(rs) / len(rs) for x in rs],
        dp_size=1, max_chunk_trajectories=8, max_open_groups=0, micro_batch_size=8,
        equal_rank_microbatches=False,
    )
    for p in range(3):
        pl.register_group((0, p), 2)
    pl.set_order([(0, 0), (0, 1), (0, 2)], [1, 2])
    pl.mark_pretrained((0, 1), [1.0, 0.0])
    assert pl._blocks == [[(0, 0)], [(0, 2)]]
    for p in range(3):
        for k in range(2):
            pl.add(_traj(p, 2 * p + k, float(k)))
    seen = []
    while pl.has_work():
        d = pl.next_dispatch()
        seen += [t.group for cs in d.per_rank for c in cs for t in c.trajectories]
    assert (0, 1) not in seen and sorted(set(seen)) == [(0, 0), (0, 2)]
    assert pl.all_done()


def test_planner_rejects_reward_mismatch():
    pl = StreamPlanner(
        lambda rs: [0.0] * len(rs), dp_size=1, max_chunk_trajectories=8,
        max_open_groups=0, micro_batch_size=8, equal_rank_microbatches=False,
    )
    pl.register_group((0, 0), 2)
    pl.mark_pretrained((0, 0), [1.0, 1.0])
    pl.add(_traj(0, 0, 1.0))
    with pytest.raises(RuntimeError, match="differ"):
        pl.add(_traj(0, 1, 0.0))


def test_assistant_span():
    log = [
        {"role": "user", "token_ids": torch.arange(5)},
        {"role": "assistant", "token_ids": torch.arange(3)},
        {"role": "environment", "token_ids": torch.arange(2)},
    ]
    assert vit.assistant_span(log) == (5, 3)


def test_plan_group_layout_spreads_one_group_and_pads():
    lay = vit.plan_group_layout([5, 9, 7, 3, 8], dp_size=2, micro_batch_size=2)
    rows = sorted(j for slots in lay for s in slots if s is not None for j in s)
    assert rows == [0, 1, 2, 3, 4]
    assert len(lay[0]) == len(lay[1])  # equal micro-batch counts (EP lockstep)
    assert any(s is None for slots in lay for s in slots)  # 5 rows on 2 ranks x 2 rows: a filler
    assert all(s is None or len(s) <= 2 for slots in lay for s in slots)
    assert lay == vit.plan_group_layout([5, 9, 7, 3, 8], 2, 2)  # a function of the lengths


def test_pass_predictor_uses_past_steps_only():
    pr = vit.PassPredictor(window=2)
    assert pr.group_pass([100, 3000]) == 1.0  # no data: start on VIT
    for _ in range(9):
        pr.observe(3000, False)
    pr.observe(3000, True)
    assert pr.rate(3000) == 1.0  # the current step does not count yet
    pr.end_step()
    assert pr.rate(3000) == (1 + 1) / (10 + 1)
    assert pr.rate(100) == 1.0  # other bucket
    pr.end_step()
    pr.end_step()  # window 2: the observations fall out
    assert pr.rate(3000) == 1.0


def test_retained_forward_reuses_output_and_restores_routes():
    class Dec(torch.nn.Module):
        def forward(self, x):
            raise AssertionError("the decoder must not run")

    class Inst:
        def __init__(self):
            self.replay_backward_list = [torch.tensor([9])]  # anchors appended by the forward

    dec, insts = Dec(), [Inst(), Inst()]
    out = torch.ones(2, requires_grad=True)
    slot = vit.RetainedSlot(
        ids=torch.tensor([[1, 2]]), output=out, routes=[torch.tensor([3]), None], num_tokens=2
    )
    got = vit.retained_forward(
        slot, lambda **kw: dec(None) * 2, dec, insts, {"input_ids_cp_sharded": torch.tensor([[1, 2]])}
    )
    assert torch.equal(got, out * 2) and got.requires_grad
    assert torch.equal(insts[0].replay_backward_list[-1], torch.tensor([3]))
    assert torch.equal(insts[1].replay_backward_list[-1], torch.tensor([9]))  # no route recorded
    assert "forward" not in dec.__dict__  # decoder restored
    with pytest.raises(RuntimeError, match="differs"):
        vit.retained_forward(
            vit.RetainedSlot(ids=torch.tensor([[1, 2]]), output=out, routes=[], num_tokens=2),
            lambda **kw: dec(None), dec, [], {"input_ids_cp_sharded": torch.tensor([[1, 3]])},
        )
