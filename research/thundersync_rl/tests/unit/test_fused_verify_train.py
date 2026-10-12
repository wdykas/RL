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
"""CPU tests: fused verification + training (fused_train.py) trains every
trajectory exactly once, with the regular gradient and bookkeeping."""

import asyncio
import random
from types import SimpleNamespace

import pytest
import thundersync_rl.speculative as spec_mod
import torch
from thundersync_rl.fused_train import (
    FusedGraph,
    FusedGraphStore,
    FusedOutcome,
    FusedRef,
    make_fused_model_forward,
    stitch_fused_logits,
    substitute_model_forward,
)
from thundersync_rl.grpo_loop import ThunderSyncConfig, validate_fused_verify_train
from thundersync_rl.streaming import StreamPlanner, Trajectory

from nemo_rl.algorithms.grad_streaming import StreamingGroupAccumulator
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

# --------------------------------------------------------------------------
# Planner: routing of fused trajectories
# --------------------------------------------------------------------------


def _loo_adv(rewards):
    m = len(rewards)
    return [r - (sum(rewards) - r) / (m - 1) for r in rewards]


def _simulate_fused(seed, dp_size, max_open_groups, chunk=3):
    """Random arrivals, some trajectories fused; replay dispatches on accumulators."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    groups, G = 5, 6
    n = groups * G
    rewards = [float(rng.random() < 0.5) for _ in range(n)]
    tokens = [rng.randint(1, 50) for _ in range(n)]
    fused = {
        i: FusedRef(key=f"k{i}", dp_rank=rng.randrange(dp_size), num_tokens=tokens[i])
        for i in range(n)
        if rng.random() < 0.5
    }
    P = [11, 4]
    H = [[torch.randn(p, dtype=torch.float64) for p in P] for _ in range(n)]
    grads = [[torch.zeros(p, dtype=torch.float64) for p in P] for _ in range(dp_size)]
    accs = [
        StreamingGroupAccumulator(
            grads[r],
            storage_device="cpu",
            storage_dtype=torch.float64,
            max_buckets_per_group=2,
        )
        for r in range(dp_size)
    ]
    planner = StreamPlanner(
        _loo_adv,
        dp_size=dp_size,
        max_chunk_trajectories=chunk,
        max_open_groups=max_open_groups,
        micro_batch_size=1,
        equal_rank_microbatches=False,
    )
    for g in range(groups):
        planner.register_group(g, G)

    trained: dict[int, int] = {}  # index -> times backpropagated
    trained_tokens = [0] * groups
    fused_rows = 0

    def run_dispatch():
        nonlocal fused_rows
        d = planner.next_dispatch()
        for rank, chunks in enumerate(d.per_rank):
            for c in chunks:
                if c.fused is not None:
                    # One row, on the rank holding its graph, its own ref.
                    (t,) = c.trajectories
                    assert t.fused is c.fused and c.fused.dp_rank == rank
                    fused_rows += 1
                else:
                    assert all(t.fused is None for t in c.trajectories)
                advs = c.advantages if c.group is None else [1.0] * len(c.trajectories)
                for t, a in zip(c.trajectories, advs):
                    trained[t.index] = trained.get(t.index, 0) + 1
                    trained_tokens[t.group] += t.num_tokens
                    for buf, h in zip(grads[rank], H[t.index]):
                        buf.add_(h, alpha=a)
                accs[rank].capture(c.group, c.reward)
            for gid, adv in d.closes:
                accs[rank].close_group(gid, adv)

    order = list(range(n))
    rng.shuffle(order)
    for i in order:
        planner.add(
            Trajectory(
                group=i // G,
                index=i,
                reward=rewards[i],
                payload=None,
                num_tokens=tokens[i],
                fused=fused.get(i),
            )
        )
        if rng.random() < 0.5:
            run_dispatch()
    while planner.has_work():
        run_dispatch()
    assert planner.all_done()
    for a in accs:
        a.finalize()
    got = [sum(grads[r][k] for r in range(dp_size)) for k in range(len(P))]
    want = [torch.zeros(p, dtype=torch.float64) for p in P]
    for g in range(groups):
        idx = range(g * G, (g + 1) * G)
        for i, a in zip(idx, _loo_adv([rewards[i] for i in idx])):
            for w, h in zip(want, H[i]):
                w.add_(h, alpha=a)
    expected_tokens = [sum(tokens[g * G : (g + 1) * G]) for g in range(groups)]
    return got, want, trained, trained_tokens, expected_tokens, fused_rows, fused


@pytest.mark.parametrize("dp_size", [1, 2, 3])
@pytest.mark.parametrize("max_open_groups", [0, 2, 100])
@pytest.mark.parametrize("seed", range(4))
def test_fused_trajectories_trained_once_with_exact_update(
    dp_size, max_open_groups, seed
):
    got, want, trained, tok, want_tok, fused_rows, fused = _simulate_fused(
        seed, dp_size, max_open_groups
    )
    # Every trajectory is backpropagated exactly once (fused or regular) ...
    assert trained == {i: 1 for i in range(30)}
    # ... every fused one through its own pinned chunk ...
    assert fused_rows == len(fused)
    # ... the normalization counts see every token of every group once ...
    assert tok == want_tok
    # ... and the update is the synchronous one.
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-12, atol=1e-12)


def test_fused_chunk_pinned_to_missing_rank_raises():
    planner = StreamPlanner(
        _loo_adv,
        dp_size=2,
        max_chunk_trajectories=4,
        max_open_groups=0,
        micro_batch_size=1,
        equal_rank_microbatches=False,
    )
    planner.register_group(0, 2)
    for i in range(2):
        planner.add(
            Trajectory(
                group=0,
                index=i,
                reward=float(i),
                payload=None,
                num_tokens=3,
                fused=FusedRef(key=f"k{i}", dp_rank=2, num_tokens=3),
            )
        )
    with pytest.raises(ValueError, match="DP rank 2 of 2"):
        planner.next_dispatch()


# --------------------------------------------------------------------------
# Driver: plans, retain flags and claims
# --------------------------------------------------------------------------

EOD = 2


class _Ref:
    def __init__(self, value):
        import concurrent.futures

        self._f = concurrent.futures.Future()
        self._f.set_result(value)

    def future(self):
        return self._f


class _Group:
    def __init__(self, handlers):
        self.handlers = handlers
        self.calls = []

    def run_all_workers_single_data(self, method, **kwargs):
        self.calls.append((method, kwargs))
        return [_Ref(self.handlers[method](**kwargs))]


@pytest.fixture
def _no_ray(monkeypatch):
    monkeypatch.setattr(spec_mod.ray, "get", lambda x: x)


def _fused_spec(verify, max_retained_tokens=1 << 20):
    base_group = SimpleNamespace(run_all_workers_single_data=lambda method: [EOD])
    base = SimpleNamespace(_policy=SimpleNamespace(worker_group=base_group))
    learner = SimpleNamespace(worker_group=_Group({}))
    s = spec_mod.SpeculativeGeneration(
        base,
        learner,
        vocab_limit=100,
        head_k=8,
        draft_budget=4096,
        max_new_tokens=16,
        pad_token_id=0,
        draft_variants=1,
        variant_eps=0.0,
        verify_mode="block",
        verify_precision="model",
        verify_batch_tokens=16384,
        longest_first=False,
        verify_on="learner",
        pool_weights=None,
        fused_verify_train=True,
        fused_max_retained_tokens=max_retained_tokens,
    )
    pool = _Group({"verify_drafts_block_fused": verify})
    s.verifiers = s._chunk_pools = [pool]
    return s, pool


# row -> (accepted, next); drafts 0/1 finished, 2 unfinished.
_OUTCOMES = {0: (3, 0), 1: (1, 7), 2: (3, 9)}


def _fake_fused_worker(rows, prompt_lens, segments, keys, retain, **_):
    """Keeps a graph exactly where the real worker does: retain and fully accepted."""
    out = []
    for j, (a, y) in _OUTCOMES.items():
        r = {"accepted": a, "next": y, "logprobs": [-0.1] * (a + 1)}
        if retain[j] and a == rows[j].numel() - prompt_lens[j]:
            r["fused"] = {"key": keys[j], "dp_rank": 1}
        out.append((j, r))
    return out


def _three_drafts():
    p = [torch.tensor([10 + i, 20 + i]) for i in range(3)]
    return p, [
        [(p[0], -1, [4, 5, EOD], True, "1:0", [(0, 0, 3)])],  # accepted, finished
        [(p[1], -1, [4, 5, EOD], True, "1:1", [(0, 0, 3)])],  # rejected after 1
        [(p[2], -1, [4, 5, 6], False, "1:2", [(0, 0, 3)])],  # accepted, unfinished
    ]


async def _rollout(s, prompt):
    data = BatchedDataDict(
        {
            "input_ids": prompt.view(1, -1),
            "input_lengths": torch.tensor([prompt.numel()]),
        }
    )
    return [x async for x in s.generate_async(data)]


async def _engine_tail(data, greedy=False):
    n = int(data["input_lengths"][0])
    tail = torch.tensor([9, EOD])
    yield (
        0,
        {
            "output_ids": torch.cat([data["input_ids"][0, :n], tail]).view(1, -1),
            "logprobs": torch.zeros(1, n + 2),
            "generation_lengths": torch.tensor([2]),
        },
    )


def test_only_finished_fully_accepted_drafts_are_fused(_no_ray):
    s, pool = _fused_spec(_fake_fused_worker)
    s.base.generate_async = _engine_tail
    p, drafts = _three_drafts()
    asyncio.run(s.verify_chunked(drafts, chunks=1))
    ((method, kw),) = pool.calls
    assert method == "verify_drafts_block_fused"
    # Unfinished drafts are never candidates; finished ones are.
    assert kw["retain"] == [True, True, False]
    assert kw["precision"] == "model" and kw["max_retained_tokens"] == 1 << 20
    plans = {k[0]: v[0] for k, v in s.plans.items()}
    assert plans[10].complete and plans[10].fused == FusedRef("1:0", 1, 5)
    assert plans[11].fused is None and not plans[11].complete
    # Fully accepted but unfinished: a continuation, never fused.
    assert plans[12].fused is None and not plans[12].complete
    assert s.stats["fused/retained"] == 1.0

    for prompt in p:
        asyncio.run(_rollout(s, prompt))
    env_reply = torch.tensor([77, 78])
    full = torch.cat([p[0], torch.tensor([4, 5, EOD]), env_reply])
    # Wrong generated tokens: no claim.
    assert s.claim_fused(p[0], torch.cat([p[0], torch.tensor([4, 6, EOD])])) is None
    assert s.claim_fused(p[0], full) == FusedRef("1:0", 1, 5)
    assert s.claim_fused(p[0], full) is None  # claimed once
    # Continued rollouts carry no fused forward.
    for prompt in p[1:]:
        full = torch.cat([prompt, torch.tensor([4, 7, 9, EOD])])
        assert s.claim_fused(prompt, full) is None


def test_worker_fused_result_on_incomplete_plan_raises(_no_ray):
    def bad_worker(rows, prompt_lens, segments, keys, retain, **_):
        fused = {"key": keys[0], "dp_rank": 0}
        return [(0, {"accepted": 1, "next": 7, "logprobs": [-0.1] * 2, "fused": fused})]

    s, _ = _fused_spec(bad_worker)
    _, drafts = _three_drafts()
    with pytest.raises(RuntimeError, match="not a finished rollout"):
        asyncio.run(s.verify_chunked(drafts[:1], chunks=1))


def test_fused_mode_rejects_unchunked_verification(_no_ray):
    s, _ = _fused_spec(_fake_fused_worker)
    with pytest.raises(NotImplementedError):
        s.verify(_three_drafts()[1])


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------


def _policy(**megatron):
    return {
        "megatron_cfg": {
            "batch_invariant_mode": True,
            "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "use_fused_linear_logprobs": False,
            **megatron,
        },
        "generation": {"temperature": 1.0},
    }


def _ts(**kw):
    return ThunderSyncConfig(
        **{
            "speculate": True,
            "verify_mode": "block",
            "verify_on": "learner",
            "verify_precision": "model",
            "fused_verify_train": True,
            **kw,
        }
    )


def test_fused_config_valid():
    validate_fused_verify_train(_ts(), _policy())
    # Off: nothing is required.
    validate_fused_verify_train(
        ThunderSyncConfig(), _policy(batch_invariant_mode=False)
    )


@pytest.mark.parametrize(
    "ts_kw, policy, match",
    [
        ({"speculate": False}, _policy(), "speculate"),
        ({"verify_mode": "keyed"}, _policy(), "verify_mode"),
        ({"verify_on": "both"}, _policy(), "verify_on"),
        ({"verify_precision": "fp32"}, _policy(), "verify_precision"),
        ({"fused_max_retained_tokens": 0}, _policy(), "fused_max_retained_tokens"),
        ({"verify_lazy": True}, _policy(), "verify_lazy"),
        ({}, _policy(batch_invariant_mode=False), "batch_invariant_mode"),
        ({}, _policy(pipeline_model_parallel_size=2), "pipeline"),
        ({}, _policy(expert_model_parallel_size=2), "expert_model_parallel"),
        ({}, _policy(use_fused_linear_logprobs=True), "use_fused_linear_logprobs"),
        ({}, {**_policy(), "generation": {"temperature": 0.7}}, "temperature"),
    ],
)
def test_fused_config_rejects(ts_kw, policy, match):
    with pytest.raises(ValueError, match=match):
        validate_fused_verify_train(_ts(**ts_kw), policy)


# --------------------------------------------------------------------------
# Worker-level torch: the retained forward reproduces training's gradient
# --------------------------------------------------------------------------

V, D = 13, 8


class _TinyCausalLM(torch.nn.Module):
    """One causal self-attention layer + LM head (float64, CPU)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(V, D)
        self.layer = torch.nn.TransformerEncoderLayer(
            D, nhead=2, dim_feedforward=16, dropout=0.0, batch_first=True
        )
        self.head = torch.nn.Linear(D, V)
        self.double()

    def forward(self, ids):
        s = ids.shape[1]
        mask = torch.nn.Transformer.generate_square_subsequent_mask(s).double()
        return self.head(self.layer(self.emb(ids), src_mask=mask, is_causal=True))


def _model_forward(model):
    def forward(*, input_ids_cp_sharded, data_dict, **_):
        return model(input_ids_cp_sharded)

    return forward


def _train_step(module, ids, token_mask, adv):
    """Stand-in for train_microbatch: forward via module.model_forward, PG loss, backward."""
    logits = module.model_forward(
        input_ids_cp_sharded=ids, data_dict={"token_mask": token_mask}
    )
    logits.div_(1.0)  # the loss path scales logits in place (temperature)
    lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None])[..., 0]
    loss = -(adv * lp * token_mask[:, 1:]).sum()
    loss.backward()


def _grads(model):
    return [p.grad.clone() for p in model.parameters()]


def _rows():
    prompt, draft, env = [3, 4, 5], [6, 7, 1, 2], [9, 9, 10]
    row = torch.tensor(prompt + draft)
    seq = prompt + draft + env + [0, 0]  # training row: + env reply + padding
    ids = torch.tensor([seq])
    mask = torch.tensor([[0] * 3 + [1] * 4 + [0] * 5], dtype=torch.float64)
    return row, ids, mask


def _verification_graph(model, row):
    # Verification forward of prompt + draft, padded like training.
    padded = torch.cat([row, torch.zeros(5, dtype=torch.long)]).view(1, -1)
    logits = model(padded)
    return FusedGraph(ids=row.clone(), logits=logits[:, : row.numel()])


def test_fused_gradient_equals_regular_training_gradient():
    row, ids, mask = _rows()
    model = _TinyCausalLM()
    module = SimpleNamespace(model_forward=_model_forward(model))

    _train_step(module, ids, mask, adv=0.7)
    want = _grads(model)
    model.zero_grad()

    graph = _verification_graph(model, row)
    outcome = FusedOutcome(used=False, fallback=False)
    original = module.model_forward
    with substitute_model_forward(
        module, make_fused_model_forward(graph, original, outcome, check=False)
    ):
        _train_step(module, ids, mask, adv=0.7)
    assert module.model_forward is original
    assert outcome == FusedOutcome(used=True, fallback=False)
    for g, w in zip(_grads(model), want):
        torch.testing.assert_close(g, w, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("case", ["other_tokens", "unmasked_suffix", "short_row"])
def test_mismatched_rows_fall_back_to_the_real_forward(case):
    row, ids, mask = _rows()
    if case == "other_tokens":
        ids = ids.clone()
        ids[0, 5] = 11
    elif case == "unmasked_suffix":
        mask = mask.clone()
        mask[0, 8] = 1  # an environment token in the loss
    else:
        ids, mask = ids[:, :5], mask[:, :5]
    model = _TinyCausalLM()
    module = SimpleNamespace(model_forward=_model_forward(model))
    _train_step(module, ids, mask, adv=-1.3)
    want = _grads(model)
    model.zero_grad()

    graph = _verification_graph(model, row)
    assert stitch_fused_logits(graph, ids, mask) is None
    outcome = FusedOutcome(used=False, fallback=False)
    with substitute_model_forward(
        module,
        make_fused_model_forward(graph, module.model_forward, outcome, check=False),
    ):
        _train_step(module, ids, mask, adv=-1.3)
    assert outcome == FusedOutcome(used=False, fallback=True)
    for g, w in zip(_grads(model), want):
        torch.testing.assert_close(g, w, rtol=1e-10, atol=1e-12)


def test_retained_forward_is_used_once():
    row, ids, mask = _rows()
    model = _TinyCausalLM()
    calls = []

    def original(**kw):
        calls.append(1)
        return model(kw["input_ids_cp_sharded"])

    outcome = FusedOutcome(used=False, fallback=False)
    fwd = make_fused_model_forward(
        _verification_graph(model, row), original, outcome, check=True
    )
    fwd(input_ids_cp_sharded=ids, data_dict={"token_mask": mask})
    assert outcome.used and len(calls) == 1  # check=True: one reference forward
    fwd(input_ids_cp_sharded=ids, data_dict={"token_mask": mask})
    assert len(calls) == 2  # a second call (rerun) runs the real forward


def test_substitution_restored_on_error():
    module = SimpleNamespace(model_forward="orig")
    with pytest.raises(RuntimeError):
        with substitute_model_forward(module, "fused"):
            assert module.model_forward == "fused"
            raise RuntimeError
    assert module.model_forward == "orig"


def test_graph_store_budget_and_lifetime():
    store = FusedGraphStore()
    g = FusedGraph(ids=torch.arange(6), logits=torch.zeros(1, 6, V))
    assert store.can_retain(6, 10)
    store.put("a", g)
    assert not store.can_retain(6, 10) and store.can_retain(4, 10)
    with pytest.raises(KeyError):
        store.put("a", g)
    assert store.pop("missing") is None
    assert store.pop("a") is g and store.retained_tokens == 0
    store.put("b", g)
    store.put("c", g)
    assert store.clear() == 2 and len(store) == 0 and store.retained_tokens == 0
    assert store.num_retained == 3


def test_megatron_train_path_resolves_the_substituted_forward():
    """The real Megatron forward step calls the substituted ``model_forward``."""
    megatron_train = pytest.importorskip("nemo_rl.models.megatron.train")
    seen = {}

    def fused_forward(**kwargs):
        seen.update(kwargs)
        return torch.zeros(1, 4, V)

    ids = torch.tensor([[1, 2, 3, 4]])
    mb = SimpleNamespace(
        data_dict=BatchedDataDict({"input_ids": ids, "token_mask": torch.ones(1, 4)}),
        input_ids=ids,
        input_ids_cp_sharded=ids,
        attention_mask=None,
        position_ids=torch.arange(4).view(1, -1),
        packed_seq_params=None,
        cu_seqlens_padded=None,
        mtp_loss_mask=None,
        padding_mask=None,
        routed_experts_cp_sharded=None,
        original_seq_length=4,
        media_token_validity_mask=None,
    )
    with substitute_model_forward(megatron_train, fused_forward):
        # An unknown post-processor fails only after the model forward ran.
        with pytest.raises(TypeError, match="Unknown post-processing"):
            megatron_train.forward_with_post_processing_fn(
                iter([mb]), model=torch.nn.Identity(), post_processing_fn=object()
            )
    assert seen["input_ids_cp_sharded"] is ids
    assert seen["data_dict"]["token_mask"].shape == (1, 4)
