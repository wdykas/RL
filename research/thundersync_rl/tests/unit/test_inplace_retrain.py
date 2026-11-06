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
"""train_overlap with in-place accumulation (NEMO_RL_EXACT_GRAD_ACCUM=inplace): an EDV discard retrains the step's
earlier calls (MegatronPolicyWorker._inplace_retrain). After it the grad buffers hold EXACTLY the kept calls' sum
(bitwise what a run without the dropped call gives), the step counters / metric lists are those before the dropped
call, and the re-run calls are checked (hook) and not captured twice."""

import pytest
import torch

from nemo_rl.algorithms.exact_grad_accum import InPlaceGradAccumulator
from nemo_rl.models.policy.workers.megatron_policy_worker import MegatronPolicyWorkerImpl as W


class _Fake:
    """The pieces of the worker _inplace_retrain uses; a 'call' adds its deterministic gradient to the buffers and
    counts like the real loss does (valid tokens, one metrics entry)."""

    _restore_step_counters = staticmethod(W._restore_step_counters)
    _snapshot_step_counters = staticmethod(W._snapshot_step_counters)

    def __init__(self, grads):
        self.grads = grads
        self.hook_calls = 0
        self._yield_ok = True
        self.yield_seen = []

    def _live_rows(self, d):
        return torch.ones(1, dtype=torch.bool)

    def _prepare_train_piece(self, state, data, mbs, call_has_live=False):
        return data

    def _pipeline_call_groups(self, pieces):
        return [(pieces, False)]

    def _run_train_pieces(self, state, call, variable_shapes):
        self.yield_seen.append(self._yield_ok)
        for d in call:
            for g, x in zip(self.grads, d["g"]):
                g += x
            state["local_valid_toks"] += float(d["toks"])
            state["all_mb_metrics"].append({"loss": float(d["toks"])})

    def _inplace_retrain_call_done(self):
        self.hook_calls += 1


def _call(seed):
    gen = torch.Generator().manual_seed(seed)
    return {"g": [torch.randn(17, generator=gen) * 1e3, torch.randn(5, generator=gen) * 1e-3],
            "toks": 10 + seed}


def _run(fake, state, data):
    """The worker body's in-place path for one call: snapshot, run, capture + remember (no surprise)."""
    snap = fake._snapshot_step_counters(state)
    if "inplace_calls" not in state:
        state["inplace_step_snapshot"] = snap
        state["inplace_calls"] = []
    fake._run_train_pieces(state, [data], False)
    state["exact_accum"].capture()
    state["inplace_calls"].append(([data], None))


@pytest.mark.parametrize("drop_at", [0, 1, 3])
def test_retrain_drops_exactly_one_call(drop_at):
    calls = [_call(s) for s in range(4)]
    grads = [torch.zeros(17), torch.zeros(5)]
    fake = _Fake(grads)
    state = {"exact_accum": InPlaceGradAccumulator(grads), "local_valid_toks": torch.zeros(()),
             "all_mb_metrics": [], "skip_zero_advantage_rows": False, "mbs": 1}
    state["exact_accum"].reset()
    for i, c in enumerate(calls):
        if i == drop_at:  # this call's forward finds a surprise: it runs, then is dropped
            snap = fake._snapshot_step_counters(state)
            if "inplace_calls" not in state:
                state["inplace_step_snapshot"] = snap
                state["inplace_calls"] = []
            fake._run_train_pieces(state, [c], False)
            W._inplace_retrain(fake, state, snap)
        else:
            _run(fake, state, c)
    # reference: the same calls without the dropped one, in the same order, from zero
    ref = [torch.zeros(17), torch.zeros(5)]
    for i, c in enumerate(calls):
        if i != drop_at:
            for g, x in zip(ref, c["g"]):
                g += x
    assert all(torch.equal(a, b) for a, b in zip(grads, ref))  # bitwise
    kept = [c for i, c in enumerate(calls) if i != drop_at]
    assert float(state["local_valid_toks"]) == sum(c["toks"] for c in kept)
    assert [m["loss"] for m in state["all_mb_metrics"]] == [float(c["toks"]) for c in kept]
    ea = state["exact_accum"]
    assert ea.captures == len(kept) and ea.discards == 1 and ea.retrain_calls == drop_at
    assert fake.hook_calls == drop_at  # every re-run call went through the check hook
    assert fake._yield_ok is True  # restored
    # pass order: calls 0..drop_at-1, the dropped call, its re-run of 0..drop_at-1 (yield off), the remaining calls
    seen = fake.yield_seen
    assert seen[: drop_at + 1] == [True] * (drop_at + 1)
    assert seen[drop_at + 1 : 2 * drop_at + 1] == [False] * drop_at  # no posted work during a re-run
    assert seen[2 * drop_at + 1 :] == [True] * (len(calls) - drop_at - 1)


def test_retrain_refuses_open_draft_step():
    grads = [torch.zeros(3)]
    fake = _Fake(grads)
    state = {"exact_accum": InPlaceGradAccumulator(grads), "local_valid_toks": torch.zeros(()), "all_mb_metrics": [],
             "skip_zero_advantage_rows": False, "mbs": 1, "draft_step_state": object(),
             "inplace_calls": [], "inplace_step_snapshot": {}}
    with pytest.raises(RuntimeError, match="draft-head"):
        W._inplace_retrain(fake, state, {})
