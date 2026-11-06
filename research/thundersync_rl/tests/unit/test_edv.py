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
"""EDV (edv.py, DESIGN §22.9): the loss-side decision is the verifier's, bitwise; the discard leaves the committed
int64 sum as if the group never trained; the finalize pipeline trains a resubmitted group once more; validator."""

import asyncio
import copy
import math

import pytest
import torch

from test_nolanes import _valid
from thundersync_rl import ds2, edv, nolanes
from thundersync_rl.block_verification import acceptance_weights
from thundersync_rl.finalize_pipeline import FinalizePipeline
from thundersync_rl.speculative import _null_result

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


@pytest.mark.parametrize("m", [1.0, 1.1, 1.5])
@pytest.mark.parametrize("seed", range(5))
def test_whole_weight_is_the_verifiers(m, seed):
    g = torch.Generator().manual_seed(seed)
    lpx = -torch.rand(300, generator=g).float() * 3
    lqx = lpx + 0.05 * torch.randn(300, generator=g).float()
    assert edv.whole_weight(lpx, lqx, m).item() == acceptance_weights(lpx - lqx, m)[-1].item()


def _batch(lpx_rows, lqx_rows, flags, etas, ms, S=40, start=5):
    B = len(lpx_rows)
    curr = torch.zeros(B, S)
    gen = torch.full((B, S), -7.0)
    tm = torch.zeros(B, S)
    lens = []
    for b, (lp, lq) in enumerate(zip(lpx_rows, lqx_rows)):
        n = lp.numel()
        curr[b, start:start + n], gen[b, start:start + n], tm[b, start:start + n] = lp, lq, 1.0
        lens.append(n)
    data = {"sample_mask": torch.ones(B), **edv.row_fields(B, {})}
    for b in range(B):
        if flags[b]:
            data[edv.FLAG][b], data[edv.ETA][b], data[edv.ENV][b], data[edv.LEN][b] = 1.0, etas[b], ms[b], lens[b]
    return curr, gen, tm, data


def test_decide_in_loss_records_and_replaces():
    edv.take()
    g = torch.Generator().manual_seed(3)
    lp0, lp1 = -torch.rand(20, generator=g), -torch.rand(25, generator=g)
    lq0, lq1 = lp0 + 0.01 * torch.randn(20, generator=g), lp1 + 0.5 * torch.randn(25, generator=g)
    curr, gen, tm, data = _batch([lp0, lp1], [lq0, lq1], [True, False], [0.3, 0.0], [1.1, 1.0])
    data["sample_mask"][0] = 0.0  # decided whatever its sample mask
    out = edv.decide_in_loss(curr, gen, tm, data["sample_mask"], data)
    dec = edv.take()
    assert len(dec) == 1
    slot, bg, eta, ok = dec[0]
    want = acceptance_weights(lp0.float() - lq0.float(), 1.1)[-1].item()
    assert slot == 0 and bg == want and eta == 0.3 and ok == (0.3 <= want)
    assert torch.equal(out[0][tm[0] > 0], curr[0][tm[0] > 0])  # deferred row: this forward's log p
    assert torch.equal(out[0][tm[0] == 0], gen[0][tm[0] == 0])
    assert torch.equal(out[1], gen[1])  # not deferred: untouched


def test_decide_in_loss_refuses_a_length_mismatch():
    lp = -torch.rand(10)
    curr, gen, tm, data = _batch([lp], [lp], [True], [0.2], [1.0])
    data[edv.LEN][0] = 11.0
    with pytest.raises(RuntimeError, match="loss tokens"):
        edv.decide_in_loss(curr, gen, tm, data["sample_mask"], data)
    edv.take()


def test_surprise_mask_and_the_negative(monkeypatch):
    dec = [(0, 0.9, 0.3, True), (3, 0.2, 0.4, False), (5, 0.1, 0.45, False)]
    assert edv.surprise_mask(dec) == (1 << 3) | (1 << 5)
    monkeypatch.setenv("THUNDERSYNC_NL_NEG", edv.NEG)
    assert edv.surprise_mask(dec) == 0


def test_threshold_is_fixed_and_the_controller_keys_are_gone(monkeypatch):
    # DESIGN §22.15: e is thundersync.edv_threshold every step (the X1 debug override aside); the removed controller
    # keys are refused as unknown, so a stale config fails at load instead of silently running a different e.
    from thundersync_rl import nolanes_stage
    monkeypatch.delenv("THUNDERSYNC_NL_DEBUG_EDV_E", raising=False)
    assert nolanes_stage.edv_e_of_step(0.5) == 0.5
    monkeypatch.setenv("THUNDERSYNC_NL_DEBUG_EDV_E", "0.95")
    assert nolanes_stage.edv_e_of_step(0.5) == 0.95
    for k, v in (("edv_controller", "cost"), ("edv_cost_ratio", 1e-3)):
        with pytest.raises(ValueError, match="unknown keys"):
            nolanes.NolanesConfig.from_config({"thundersync": {"mode": nolanes.MODE, k: v}})


def test_deferred_result_walks_as_a_whole_output():
    lq = [-0.5, -1.25, -2.0]
    d = (torch.tensor([1, 2]), -1, [7, 8, 9], True, "k", (), lq, None)
    res = edv.deferred_result(d, (1.0, 1.1))
    for m in (1.0, 1.1):
        r = _null_result(res, m)
        assert r["accepted"] == 3 and r["null"] == "out" and r["draft_logq"] == lq and r["logprobs"][:3] == lq
    assert res["null_lr_moments"] == (0, 0.0, 0.0)
    assert not edv.eligible(False, 0.1, 0.5) and edv.eligible(True, 0.5, 0.5) and not edv.eligible(True, 0.51, 0.5)


def test_pipeline_trains_a_resubmitted_group_once_more():
    async def main():
        trained = []

        async def prep(g, res):
            return res

        pipe = FinalizePipeline(prep)

        async def rows(v):
            await asyncio.sleep(0)
            return v

        pipe.submit("a", rows("a0"))
        pipe.submit("b", rows("b0"))

        async def train(batch):
            for g, out in batch:
                trained.append((g, out))
                if out == "a0":
                    pipe.resubmit("a", rows("a1"))

        await pipe.drain(2, train)
        return trained

    got = asyncio.run(main())
    assert sorted(got) == [("a", "a0"), ("a", "a1"), ("b", "b0")]


@cuda
def test_discard_leaves_the_committed_sum_bitwise():
    from nemo_rl.algorithms.exact_grad_accum import OrderInvariantGradAccumulator

    def run(seq):
        torch.manual_seed(0)
        grads = [torch.zeros(5000, device="cuda"), torch.zeros(777, device="cuda")]
        acc = OrderInvariantGradAccumulator(grads)
        acc.reset()
        for kind, scale in seq:
            for gr in grads:
                gr.copy_(torch.randn_like(gr) * scale)
            acc.discard() if kind == "drop" else acc.capture()
        acc.finalize()
        return [gr.clone() for gr in grads]

    with_drop = run([("keep", 1.0), ("drop", 3.0), ("keep", 2.0)])
    # the same draws, the dropped one never captured
    torch.manual_seed(0)
    g1 = [torch.randn(5000, device="cuda") * 1.0, torch.randn(777, device="cuda") * 1.0]
    _ = [torch.randn(5000, device="cuda"), torch.randn(777, device="cuda")]  # the dropped draw
    g3 = [torch.randn(5000, device="cuda") * 2.0, torch.randn(777, device="cuda") * 2.0]
    grads = [torch.zeros(5000, device="cuda"), torch.zeros(777, device="cuda")]
    acc = OrderInvariantGradAccumulator(grads)
    acc.reset()
    for src in (g1, g3):
        for gr, s in zip(grads, src):
            gr.copy_(s)
        acc.capture()
    acc.finalize()
    kept_only = [gr.clone() for gr in grads]
    assert all(torch.equal(a, b) for a, b in zip(with_drop, kept_only))


@cuda
def test_loss_side_b_g_equals_the_ds2_scan_bitwise():
    from test_ds2 import CASES, _case, _lse, _sharded

    for case in (2, 3, 4, 5):
        zp, zq, vlim, draft = _case(7, **CASES[case])
        lqx = (zq[:, :vlim].float() - _lse(zq, vlim)[:, None])[torch.arange(draft.numel(), device="cuda"), draft]
        envs = (1.0, 1.1, 1.5)
        out, lpx, _, _ = _sharded(zp, zq, vlim, draft, lqx, 11, envs)
        rec = ds2.take_last_row()
        for m in envs:
            assert edv.whole_weight(lpx, lqx, m).item() == rec["b_g"][m]


def _edv_cfg():
    c = copy.deepcopy(_valid())
    c["thundersync"]["edv_threshold"] = 0.5
    return c


@pytest.mark.parametrize(
    "mut,needle",
    [
        (lambda c: None, "train_overlap"),
        (lambda c: c["thundersync"].update(edv_threshold=1.5), "(0, 1)"),
        (lambda c: c.setdefault("loss_fn", {}).update(force_on_policy_ratio=False), "force_on_policy_ratio"),
        (lambda c: c["policy"]["sequence_packing"].update(enabled=True), "sequence_packing"),
        (lambda c: c.setdefault("loss_fn", {}).update(skip_zero_advantage_rows=True), "skip_zero_advantage_rows"),
    ],
)
def test_validator_edv(mut, needle):
    c = _edv_cfg()
    mut(c)
    assert any(needle in p for p in nolanes.config_problems(c)), nolanes.config_problems(c)


def test_validator_refuses_edv_debug_without_edv(monkeypatch):
    monkeypatch.setenv("THUNDERSYNC_NL_DEBUG_EDV_E", "0.95")
    assert any("THUNDERSYNC_NL_DEBUG_EDV_E" in p for p in nolanes.config_problems(_valid()))
