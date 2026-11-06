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
"""``thundersync.verify0_batch_tokens``: verify0 packs more rows per learner forward.

Exactness argument tested here (the GPU half, bitwise logits of a row in any
batch, is batch invariance, checked in-loop by the mismatch check):
* packing is order preserving: the rows a rank verifies, and the order its
  per-call random stream is consumed in, do not depend on the budget;
* only verify0 calls (one-shot waves, streamed calls) get the verify0 budget;
  continuation pieces keep ``verify_batch_tokens``;
* plans are identical with and without the knob on the fake engine.
"""

import asyncio
import random
from types import SimpleNamespace

import pytest
import torch

from thundersync_rl import speculative as spec_mod
from thundersync_rl.block_verification import NULL_TOKEN
from thundersync_rl.verify0_stream import Verify0Risk
from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker


@pytest.fixture(autouse=True)
def _no_ray(monkeypatch):
    monkeypatch.setattr(spec_mod.ray, "get", lambda x: x)


def _batches(lengths, budget, m=128):
    fake = SimpleNamespace(cfg={"make_sequence_length_divisible_by": m})
    rows = [torch.zeros(n, dtype=torch.long) for n in lengths]
    cls = ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class
    return cls._row_batches(fake, rows, budget)


@pytest.mark.parametrize("seed", range(20))
def test_row_batches_preserve_order_for_any_budget(seed):
    rng = random.Random(seed)
    lengths = sorted(rng.randint(1, 8448) for _ in range(rng.randint(1, 120)))
    flat = None
    for budget in (0, 16384, 32768, 65536, 131072, 10**9):
        b = _batches(lengths, budget)
        order = [i for g in b for i in g]
        assert order == list(range(len(lengths)))  # same rows, same order: same random stream use
        for g in b:
            pad = -(-max(lengths[i] for i in g) // 128) * 128
            assert len(g) == 1 or len(g) * pad <= budget
        flat = order if flat is None else flat
        assert order == flat


def test_bigger_budget_fewer_forwards_at_8k():
    lengths = [5000] * 20 + [8400] * 80  # whole 8k drafts (padded 8448): one per 16384-token forward
    n16 = len(_batches(lengths, 16384))
    n64 = len(_batches(lengths, 65536))
    assert n16 == 7 + 80  # 3 x 5120 per forward, then one 8448 row per forward
    assert n64 == 14  # 12 x 5120, 8 x 5120, then 7 x 8448 per forward


def _sticky_spec(v0_batch):
    from test_speculative_driver import _spec as make_spec
    from test_sticky_driver import EOD, _Engine
    from test_verify0_stream import _engine_and_spec

    def rounds():
        return [
            [
                ([4, 5, 6], False, [-0.1] * 3, 4, [10, EOD]),
                ([7, EOD], True, [-0.2] * 2, 2),
                ([5], False, [-0.3], 0),
            ],
            [([11, EOD], True, [-0.5] * 2, 3)],
        ]

    outcome = {
        (1, 2, 4, 5, 6): (3, 99),
        (3, 7, EOD): (0, 9),
        (1, 2, 4, 5, 6, 10, EOD): (2, 99),
        (3, 9, 11, EOD): (1, EOD),
    }
    engine, s = _engine_and_spec(make_spec, _Engine, rounds(), outcome)
    s.verify0_batch_tokens = v0_batch
    return engine, s


def _verify_calls(s):
    out = []
    for name, kw in s.learner_policy.worker_group.calls:
        method = kw.get("method", name) if name == "ordered_call" else name
        if method == "verify_drafts_block_engine":
            out.append(kw["batch_tokens"])
    return out


def _plans(s):
    return {k: [(p.prefix, p.complete, p.anchors.length) for p in v] for k, v in s.plans.items()}


@pytest.mark.parametrize("stream", [False, True])
def test_only_verify0_calls_get_the_verify0_budget(stream):
    prompts = [torch.tensor([1, 2]), torch.tensor([3]), torch.tensor([8])]
    results = []
    for v0b in (None, 65536):
        engine, s = _sticky_spec(v0b)
        s.current_step = 0
        drafts = asyncio.run(s.draft_streaming(prompts))()
        s.current_step = 1
        if not stream:
            asyncio.run(s.verify_sticky(drafts))
            n_v0 = 1
        else:
            v0 = s.prepare_stream(drafts, Verify0Risk(), 1, 1, 2)

            class _Bag(set):
                def update(self, *a):
                    pass

                def difference_update(self, *a):
                    pass

            async def go():
                s.begin_streaming_tails()
                s._learner_inflight = _Bag()
                try:
                    await s.complete_stream(v0, lambda info: None)
                finally:
                    s.end_streaming_tails()

            asyncio.run(go())
            n_v0 = int(s.stats["v0s/calls"])
        calls = _verify_calls(s)
        assert len(calls) > n_v0  # verify0 + continuation pieces
        want0 = 16384 if v0b is None else v0b
        # verify0's calls come first (streamed: interleaved with pieces of decided
        # prompts, so count them instead of taking a prefix).
        assert calls.count(want0) >= n_v0
        if v0b is not None:
            assert calls.count(v0b) == n_v0
            assert all(c == 16384 for c in calls if c != v0b)
        results.append((_plans(s), len(engine.started)))
    assert results[0] == results[1]


def test_null_waves_use_the_verify0_budget():
    """⊥-BV verify0 (waves): every wave carries the verify0 budget; decisions unchanged."""
    from test_speculative_driver import EOD, _spec as make_spec

    N, G = 6, 2
    drafted = [([4 + j, EOD], True, [-0.1] * 2) for j in range(N)]
    out = []
    for v0b in (None, 49152):
        sent = []

        def verify(rows, prompt_lens, segments, keys, engine_logq, hidden, **kw):
            sent.append(kw["batch_tokens"])
            return [
                (
                    i,
                    {
                        # first three drafts ⊥ at every envelope: forces a second wave
                        "null_grid": {
                            M: (2, NULL_TOKEN, -1.0) if keys[i] in ("1:0", "1:1", "1:2") else (2, 0, -1.0)
                            for M in kw["null_envelopes"][i]
                        },
                        "token_logprobs": [-1.0, -1.0],
                        "draft_logq": engine_logq[i][:2],
                    },
                )
                for i in range(len(keys))
            ]

        engine = {
            "install_draft_hidden_log": lambda **_: True,
            "start_drafts": lambda prompts, budget, interval, hidden_keys=None: None,
            "draft_request_ids": lambda: {},
            "collect_drafts": lambda records=None, record_refs=None, request_ids=None: drafted,
            "pop_draft_hidden": lambda keys, lengths: {k: torch.ones(n, 4) for k, n in zip(keys, lengths) if n},
        }
        s = make_spec(
            {"verify_drafts_block_engine": verify, "stash_head": lambda **kw: 1},
            base_handlers=engine,
            verify_source="engine",
            hidden_transport="driver",
            null_bv=True,
            null_bv_policy={"fixed_m": 2.0},
        )
        s.verify0_batch_tokens = v0b
        s._free_hidden = lambda hs: None
        s.current_step = 0
        drafts = asyncio.run(s.draft_streaming([torch.tensor([1])] * N, null_slots={(1,): G}))()
        s.current_step = 1
        s.verify(drafts)
        assert s.stats["spec/null_waves"] >= 2.0
        assert sent == [16384 if v0b is None else v0b] * len(sent)
        out.append([p.prefix for p in s.plans[(1,)]])
    assert out[0] == out[1]


def test_config_validation():
    from test_sticky_driver import _master

    from thundersync_rl.grpo_loop import validate_verify0_batch_tokens

    validate_verify0_batch_tokens(_master())
    validate_verify0_batch_tokens(_master(verify0_batch_tokens=65536))
    with pytest.raises(ValueError, match="verify0_batch_tokens"):
        validate_verify0_batch_tokens(_master(verify0_batch_tokens=0))


def test_budget_must_fit_device_dispatch_cap():
    """LGRAPHS (device MoE dispatch, EP > 1): a verify forward above T_cap would raise in the exchange."""
    from test_sticky_driver import _master

    from thundersync_rl.grpo_loop import validate_verify0_batch_tokens

    def m(ep, envs, **ts):
        cfg = _master(**ts)
        cfg.policy["megatron_cfg"] = {"expert_model_parallel_size": ep, "env_vars": envs}
        return cfg

    dd = {"NEMO_RL_MOE_DEVICE_DISPATCH": "1", "NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS": "20480"}
    validate_verify0_batch_tokens(m(4, dd, verify0_batch_tokens=20480))
    validate_verify0_batch_tokens(m(4, dd))  # verify_batch_tokens 16384 <= 20480
    validate_verify0_batch_tokens(m(1, dd, verify0_batch_tokens=65536))  # EP 1: no workspace
    validate_verify0_batch_tokens(m(4, {}, verify0_batch_tokens=65536))  # Megatron's dispatcher
    with pytest.raises(ValueError, match="NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS=20480"):
        validate_verify0_batch_tokens(m(4, dd, verify0_batch_tokens=65536))
    with pytest.raises(ValueError, match="verify_batch_tokens"):
        validate_verify0_batch_tokens(m(4, dd, verify_batch_tokens=32768))
    validate_verify0_batch_tokens(m(4, dict(dd, NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS="65536"), verify0_batch_tokens=65536))


def test_budget_must_fit_training_micro_batch_with_call_cap():
    """NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP=1: a verify forward larger than the training micro-batch overflows the
    per-call receive cap (fixed floor R_cap / 8) and reruns on Megatron's dispatcher (Ultra-L27 MBS 1 with
    verify0_batch_tokens 20480: 90 of 95 verify forwards per rank ran twice). The validator refuses it."""
    from test_sticky_driver import _master

    from thundersync_rl.grpo_loop import validate_verify0_batch_tokens

    def m(envs, mbs=1, **ts):
        cfg = _master(**ts)
        cfg.policy["megatron_cfg"] = {"expert_model_parallel_size": 32, "env_vars": envs}
        cfg.policy["train_micro_batch_size"] = mbs
        cfg.policy["max_total_sequence_length"] = 10240
        return cfg

    dd = {"NEMO_RL_MOE_DEVICE_DISPATCH": "1", "NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS": "20480"}
    cap = dict(dd, NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP="1")
    # Without the per-call cap the receive buffers are R_cap rows: no new rule (paper Super config, MBS 2).
    validate_verify0_batch_tokens(m(dd, verify0_batch_tokens=20480))
    # The paper-tree Ultra overlay (MBS 1, CALL_CAP=1) with the file's verify0 budget: refused.
    with pytest.raises(ValueError, match="verify0_batch_tokens=20480 > the training micro-batch"):
        validate_verify0_batch_tokens(m(cap, verify0_batch_tokens=20480, verify_batch_tokens=10240))
    # verify_batch_tokens default 16384 also exceeds one 10240-token micro-batch (it is verify0's budget when unset).
    with pytest.raises(ValueError, match="> the training micro-batch"):
        validate_verify0_batch_tokens(m(cap))
    with pytest.raises(ValueError, match="verify_batch_tokens=16384"):
        validate_verify0_batch_tokens(m(cap, verify0_batch_tokens=10240))
    validate_verify0_batch_tokens(m(cap, verify0_batch_tokens=10240, verify_batch_tokens=10240))
    # MBS 2: two rows per training micro-batch, the 20480 budget fits.
    validate_verify0_batch_tokens(m(cap, mbs=2, verify0_batch_tokens=20480, verify_batch_tokens=16384))


def test_capacity_reruns_merge_as_max_over_ranks():
    from thundersync_rl.streaming import merge_rank_memory

    out = merge_rank_memory(
        [{"learner_moe_capacity_reruns": 3.0, "learner_mem_peak_gb": 1.0},
         {"learner_moe_capacity_reruns": 5.0, "learner_mem_peak_gb": 2.0}]
    )
    assert out["learner_moe_capacity_reruns"] == 5.0


def test_capacity_rerun_counter_reads_learner_graphs(monkeypatch):
    from nemo_rl.models.megatron import learner_graphs

    from thundersync_rl import worker

    monkeypatch.setattr(learner_graphs, "stats", lambda: {"capacity_rerun": 7, "replay_fwd": 3})
    assert worker._moe_capacity_reruns() == 7
    monkeypatch.setattr(learner_graphs, "stats", lambda: {})
    assert worker._moe_capacity_reruns() == 0


def test_call_cap_rule_reads_process_env(monkeypatch):
    """Reviewer finding (4): CALL_CAP set only in the process env (not megatron_cfg.env_vars) is still checked."""
    from test_sticky_driver import _master

    from thundersync_rl.grpo_loop import validate_verify0_batch_tokens

    cfg = _master(verify0_batch_tokens=20480, verify_batch_tokens=10240)
    cfg.policy["megatron_cfg"] = {
        "expert_model_parallel_size": 32,
        "env_vars": {"NEMO_RL_MOE_DEVICE_DISPATCH": "1", "NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS": "20480"},
    }
    cfg.policy["train_micro_batch_size"] = 1
    cfg.policy["max_total_sequence_length"] = 10240
    monkeypatch.delenv("NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP", raising=False)
    validate_verify0_batch_tokens(cfg)  # no cap anywhere: no rule
    monkeypatch.setenv("NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP", "1")
    with pytest.raises(ValueError, match="verify0_batch_tokens=20480 > the training micro-batch"):
        validate_verify0_batch_tokens(cfg)


def test_nograd_floor_admits_budgets_up_to_8f_micro_batches(monkeypatch):
    """CALL_CAP_NOGRAD = f: verify budgets up to max(1, 8 f) training micro-batches pass, larger ones are refused."""
    from test_sticky_driver import _master

    from thundersync_rl.grpo_loop import validate_verify0_batch_tokens

    monkeypatch.delenv("NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP_NOGRAD", raising=False)

    def m(ng, **ts):
        cfg = _master(**ts)
        envs = {"NEMO_RL_MOE_DEVICE_DISPATCH": "1", "NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS": "20480",
                "NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP": "1"}
        if ng is not None:
            envs["NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP_NOGRAD"] = ng
        cfg.policy["megatron_cfg"] = {"expert_model_parallel_size": 32, "env_vars": envs}
        cfg.policy["train_micro_batch_size"] = 1
        cfg.policy["max_total_sequence_length"] = 10240
        return cfg

    with pytest.raises(ValueError, match="> the training micro-batch"):
        validate_verify0_batch_tokens(m(None, verify0_batch_tokens=20480, verify_batch_tokens=10240))
    validate_verify0_batch_tokens(m("0.25", verify0_batch_tokens=20480, verify_batch_tokens=20480))
    with pytest.raises(ValueError, match="> the training micro-batch"):
        validate_verify0_batch_tokens(m("0.125", verify0_batch_tokens=20480, verify_batch_tokens=10240))
