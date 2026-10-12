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
"""Engine-sourced q (verify_source="engine"): hidden log, rule, worker and driver."""

import asyncio
import os
import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace

import pytest
import thundersync_rl.speculative as spec_mod
import torch
from thundersync_rl.engine_q import (
    GatheredLogprobRows,
    HiddenLog,
    gathered_log_softmax,
    head_stash_bytes,
    no_gather,
    step_rows,
    verify_engine_q,
)
from thundersync_rl.grpo_loop import ThunderSyncConfig, validate_engine_verify_source

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# --------------------------------------------------------------------------
# Engine side: which forward row is which drafted token
# --------------------------------------------------------------------------


def test_step_rows_maps_tagged_requests_to_generated_index():
    info = {7: ("a", 5), 9: ("b", 3)}  # rid -> (key, prompt length); 8 untagged
    # Prefill of "a" (5 prompt tokens, nothing cached): its row samples token 0.
    # "b" decodes its token 4 (3 + 3 cached, one input token). rid 8 is a
    # regular rollout.
    out = step_rows([7, 8, 9], [0, 10, 6], [5, 1, 1], info.get)
    assert out == [(0, "a", 0), (2, "b", 4)]
    # An intermediate prefill chunk samples nothing; a re-prefill after an
    # eviction (prompt + 2 generated tokens) samples token 2 again.
    assert step_rows([7], [0], [3], info.get) == []
    assert step_rows([7], [0], [7], info.get) == [(0, "a", 2)]


def test_hidden_log_pop_returns_logged_prefix_and_closes_the_key():
    log = HiddenLog(chunk_rows=3)
    h0 = torch.arange(12.0).view(3, 4)
    log.record([(0, "a", 0), (2, "b", 0)], h0)
    log.record([(1, "a", 1)], h0 + 100)
    log.record([(0, "a", 2), (1, "b", 1)], h0 + 200)  # spills into a new chunk
    assert torch.equal(log.pop("a", 3), torch.stack([h0[0], h0[1] + 100, h0[0] + 200]))
    log.record([(0, "a", 3)], h0)  # late row of a closed key: dropped
    assert log.live_rows() == 2
    # Row 2 missing: the longest logged prefix (rows 0-1); the driver truncates.
    assert torch.equal(log.pop("b", 3), torch.stack([h0[2], h0[1] + 200]))
    log.record([(1, "d", 1)], h0)  # no row 0: nothing usable
    assert log.pop("d", 2) is None
    assert log.live_rows() == 0
    # A recomputed row (eviction) replaces the earlier one.
    log.record([(0, "c", 0)], h0)
    log.record([(1, "c", 0)], h0)
    assert torch.equal(log.pop("c", 1), h0[1:2])


def test_hidden_log_hook_on_a_fake_engine(monkeypatch):
    """install_draft_hidden_log: rows of tagged requests, keyed by position."""
    from megatron.core import parallel_state as ps
    from megatron.core.inference import utils as inf_utils
    from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker

    monkeypatch.setattr(ps, "is_pipeline_last_stage", lambda *a, **k: True)
    monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", lambda *a, **k: 0)
    monkeypatch.setattr(inf_utils.InferenceMode, "is_active", lambda *a, **k: True)

    def req(key, plen):
        sp = SimpleNamespace() if key is None else SimpleNamespace(ts_hidden_key=key)
        return SimpleNamespace(
            record=[
                SimpleNamespace(
                    sampling_params=sp, prompt_tokens=[0] * plen, generated_tokens=[]
                )
            ]
        )

    class Head(torch.nn.Module):
        sequence_parallel = False

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, 6))

        def forward(self, input_, weight=None, runtime_gather_output=None):
            return input_.sum(-1, keepdim=True), None

    model = SimpleNamespace(module=SimpleNamespace(output_layer=Head()))
    sampled = []
    sampling = SimpleNamespace(
        sample_kernel=lambda logits, n, context, **kw: sampled.append(n)
    )
    ctx = SimpleNamespace(
        max_requests=8,
        config=SimpleNamespace(materialize_only_last_token_logits=True),
        num_speculative_tokens=0,
        paused_request_count=0,
        total_request_count=2,
        request_ids=torch.tensor([1, 2]),
        request_kv_length_offsets=torch.tensor([0, 0]),
        request_query_lengths=torch.tensor([4, 2]),
    )
    w = object.__new__(ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)
    w.dynamic_inference_engine = SimpleNamespace(
        requests={1: req("d|0", 4), 2: req(None, 2)},
        controller=SimpleNamespace(_sampling=sampling),
    )
    w.inference_context = ctx
    w._gen_model = lambda: model
    w.inference_model = None
    w._install_hidden_copy(ctx)  # at engine construction, every rank
    assert w.install_draft_hidden_log() is True
    head = model.module.output_layer
    steps = []
    for s in range(3):  # prefill, then two decode steps
        h = torch.randn(2, 1, 6)
        steps.append(h[0, 0])
        head(h)
        # Rows are logged by the sampler that follows the forward (it also runs
        # after CUDA-graph replays, which skip Python hooks).
        sampling.sample_kernel(None, 2, ctx)
        for r in w.dynamic_inference_engine.requests.values():
            r.record[-1].generated_tokens.append(0)  # the engine's update phase
        ctx.request_kv_length_offsets += ctx.request_query_lengths
        ctx.request_query_lengths = torch.tensor([1, 1])
    out = w.pop_draft_hidden(["d|0", "x"], [3, 1])
    assert list(out) == ["d|0"]
    assert torch.equal(out["d|0"], torch.stack(steps))


@pytest.mark.parametrize(
    "tp_rank, sequence_parallel, materialized, expected",
    [
        (0, True, True, True),  # engine gathers SP rows before the head: log
        (1, True, True, None),  # replicated rows: TP rank 0 alone logs
        (0, True, False, False),  # sequence-sharded head input: unsupported
        (0, False, False, True),
    ],
)
def test_hidden_log_install_under_tensor_parallelism(
    monkeypatch, tp_rank, sequence_parallel, materialized, expected
):
    from megatron.core import parallel_state as ps
    from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker

    monkeypatch.setattr(ps, "is_pipeline_last_stage", lambda *a, **k: True)
    monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", lambda *a, **k: tp_rank)

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, 6))

        def forward(self, input_, weight=None, runtime_gather_output=None):
            return input_, None

    head = Head()
    head.sequence_parallel = sequence_parallel
    w = object.__new__(ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)
    w.dynamic_inference_engine = SimpleNamespace(
        requests={},
        controller=SimpleNamespace(
            _sampling=SimpleNamespace(sample_kernel=lambda *a, **k: None)
        ),
    )
    w.inference_context = SimpleNamespace(
        max_requests=8,
        config=SimpleNamespace(materialize_only_last_token_logits=materialized),
        num_speculative_tokens=0,
    )
    w._gen_model = lambda: SimpleNamespace(module=SimpleNamespace(output_layer=head))
    w._install_hidden_copy(w.inference_context)
    assert w.install_draft_hidden_log() is expected


# --------------------------------------------------------------------------
# The verification rule: engine log q + rows on demand == eager block rule
# --------------------------------------------------------------------------


@pytest.mark.parametrize("noise", [0.0, 0.05, 1.0])
def test_engine_rule_matches_eager_block_verification(noise):
    """Same decisions and generator calls as block_verify_logits on the same rows."""
    from thundersync_rl.block_verification_fused import block_verify_logits

    torch.manual_seed(0)
    vocab, rejected, rows_used = 50, 0, 0
    for trial in range(80):
        g = trial % 13
        p = torch.randn(g + 1, vocab) * 2
        q = p[:g] + torch.randn(g, vocab) * noise
        lp, lq = torch.log_softmax(p, -1), torch.log_softmax(q, -1)
        draft = torch.multinomial(lq.exp(), 1).flatten() if g else torch.zeros(0).long()
        engine_lq = lq[torch.arange(g), draft]
        asked = []

        def q_rows(idx, lq=lq):
            asked.extend(idx.tolist())
            return lq[idx]

        g1 = torch.Generator().manual_seed(trial)
        g2 = torch.Generator().manual_seed(trial)
        tau, y, logp, n_eval = verify_engine_q(lp, draft, engine_lq, q_rows, g1)
        ref = block_verify_logits(lp, lq, draft, g2)
        assert (tau, y) == ref[:2]
        # The eager rule renormalizes rows by their fp32 logsumexp (~1 ulp off
        # 0 for log-prob rows); the engine rule uses the rows as given.
        torch.testing.assert_close(logp, ref[2], rtol=0, atol=1e-6)
        assert torch.equal(g1.get_state(), g2.get_state())
        # log p of the kept tokens: training's own log-prob rows, bitwise.
        kept = torch.cat([draft[:tau], torch.tensor([y])])
        assert torch.equal(logp, lp[torch.arange(tau + 1), kept])
        # q rows only where the scan needed them, plus the residual row.
        assert len(asked) == n_eval + (tau < g)
        rejected += tau < g
        rows_used += len(asked)
    if noise >= 1.0:
        assert rejected > 20
    if noise == 0.0:
        assert rows_used == 0  # p == q: every draft whole, no q row


def test_behavior_logprobs_are_exactly_log_p_with_engine_q():
    from thundersync_rl.speculative import behavior_logprobs

    # Values whose (logp + e) - e would round in floating point.
    logp = [-1e-30, -3.3, -7.0]
    e = [-17.123456789, -0.1]
    assert behavior_logprobs(logp, 2, e, list(e)) == logp


# --------------------------------------------------------------------------
# Worker path on GPU with a fake training model / engine
# --------------------------------------------------------------------------

V, D, VERSIONS = 40, 16, 3  # V divisible by the fake TP sizes


class _Head(torch.nn.Module):
    """ColumnParallelLinear-like output layer: (input [s, b, d], weight) -> logits."""

    sequence_parallel = False

    def __init__(self, w):
        super().__init__()
        self.weight = torch.nn.Parameter(w.clone())

    def forward(self, input_, weight=None, runtime_gather_output=None):
        w = self.weight if weight is None else weight
        return _logits(input_, w), None


def _logits(h, w):
    """Row-wise head (no GEMM): a row's logits do not depend on the row count."""
    return (h.unsqueeze(-2) * w).sum(-1)


def _hidden(ids, emb):
    """Deterministic 'body': final hidden state of every prefix [n, d]."""
    return torch.tanh(torch.cumsum(emb[ids], 0) / 3)


def _engine_worker(monkeypatch):
    from megatron.core import parallel_state as ps
    from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker

    for name, val in [
        ("get_data_parallel_world_size", 1),
        ("get_data_parallel_rank", 0),
        ("get_tensor_model_parallel_world_size", 1),
        ("get_pipeline_model_parallel_world_size", 1),
    ]:
        monkeypatch.setattr(ps, name, lambda *a, _v=val, **k: _v)
    gen = torch.Generator().manual_seed(0)
    emb = torch.randn(V, D, generator=gen).cuda()
    heads = [torch.randn(V, D, generator=gen).cuda() * 2]
    for _ in range(VERSIONS - 1):
        heads.append(heads[-1] + torch.randn(V, D, generator=gen).cuda() * 0.3)
    w = object.__new__(ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)
    w.cfg = {"megatron_cfg": {"batch_invariant_mode": True}}
    w.rank = 0
    w.model = torch.nn.Module()
    w.model.module = torch.nn.Module()
    w.model.module.output_layer = _Head(heads[-1])  # current weights = last version
    w._head_stash = {v: heads[v].clone() for v in range(VERSIONS - 1)}

    def training_forward(row):
        h = _hidden(row.cuda(), emb)
        return w.model.module.output_layer(h.unsqueeze(1))[0].transpose(0, 1)

    w._training_forward = training_forward
    return w, emb, heads


def _engine_drafts(emb, heads, n=20, seed=1, device="cuda", key_names=None):
    """Drafts sampled 'by the engine' from versions 0/1, with its log q and hidden."""
    gen = torch.Generator(device=device).manual_seed(seed)
    rows, plens, segs, lqs, hids, keys = [], [], [], [], [], []
    for i in range(n):
        plen = 2 + i % 3
        g = [0, 1, 4, 9, 15][i % 5]
        cut = g // 2 if i % 2 else g
        ids = torch.randint(0, V, (plen,), device=device, generator=gen)
        lq, hs = [], []
        for t in range(g):
            v = 0 if t < cut else 1
            h = _hidden(ids, emb)[-1]
            # The engine's forward: a batch of 5 rows, this request's in row 3.
            batch = torch.randn(5, D, device=device, generator=gen)
            batch[3] = h
            lp = torch.log_softmax(_logits(batch, heads[v]).float(), -1)[3]
            x = torch.multinomial(lp.exp(), 1, generator=gen)
            lq.append(float(lp[x]))
            hs.append(h.cpu())
            ids = torch.cat([ids, x])
        rows.append(ids.cpu())
        plens.append(plen)
        segs.append([(0, 0, cut), (1, cut, g)])
        lqs.append(lq)
        hids.append(torch.stack(hs) if hs else torch.empty(0, 0))
        keys.append(f"k{i}" if key_names is None else key_names[i])
    return rows, plens, segs, lqs, hids, keys


@cuda
def test_engine_worker_matches_eager_rule_and_reports_training_logp(monkeypatch, capfd):
    from thundersync_rl.block_verification_fused import block_verify_logits

    monkeypatch.setenv("THUNDERSYNC_ENGINE_Q_CHECK", "1")
    w, emb, heads = _engine_worker(monkeypatch)
    rows, plens, segs, lqs, hids, keys = _engine_drafts(emb, heads)
    res = dict(
        w.verify_drafts_block_engine(
            rows=rows,
            prompt_lens=plens,
            segments=segs,
            engine_logq=lqs,
            hidden=hids,
            vocab_limit=1 << 30,
            seed=3,
            keys=keys,
            batch_tokens=64,
        )
    )
    out = capfd.readouterr().out
    n_tok = sum(len(x) for x in lqs)
    assert f"tok={n_tok} exact={n_tok} max|dlogq|=0.000e+00" in out
    # Reference: the eager rule on full rows, same per-rank generator stream
    # (rows are visited shortest first).
    gen = torch.Generator(device="cuda").manual_seed(3 * 1_000_003)
    rejected = 0
    for i in sorted(range(len(rows)), key=lambda i: rows[i].numel()):
        n, plen = rows[i].numel(), plens[i]
        h_all = _hidden(rows[i].cuda(), emb)
        lp = torch.log_softmax(_logits(h_all[plen - 1 : n], heads[-1]).float(), -1)
        vers = [v for v, a, b in segs[i] for _ in range(a, b)]
        lq = (
            torch.stack(
                [
                    torch.log_softmax(
                        _logits(h_all[plen - 1 + t], heads[v]).float(), -1
                    )
                    for t, v in enumerate(vers)
                ]
            )
            if vers
            else lp[:0]
        )
        draft = rows[i][plen:].cuda()
        tau, y, logp = block_verify_logits(lp, lq, draft, gen)
        r = res[i]
        assert (r["accepted"], r["next"]) == (tau, y)
        # Training's own log-probs (log-softmax + gather), bitwise.
        kept = torch.cat([draft[:tau], torch.tensor([y], device="cuda")])
        assert r["logprobs"] == lp[torch.arange(tau + 1), kept].tolist()
        torch.testing.assert_close(
            torch.tensor(r["logprobs"]), logp.cpu(), rtol=0, atol=1e-6
        )
        assert r["draft_logq"] == lqs[i][:tau]
        rejected += tau < n - plen
    assert rejected >= 2


@cuda
def test_head_rows_are_batch_invariant_under_bi_cublas_settings():
    """The LM-head GEMM gives a row the same logits at any row count.

    This is what makes q rows rebuilt from logged hidden states bitwise the
    engine's rows (the hidden states themselves are bitwise the training
    forward's under batch invariance). Run in a fresh process with the
    batch-invariant cuBLAS settings applied before CUDA initializes. Shapes
    of Qwen3-1.7B / Qwen3-8B heads (bf16).
    """
    code = textwrap.dedent(
        """
        import torch
        bad = []
        for d, v in [(2048, 151936), (4096, 151936)]:
            torch.manual_seed(0)
            w = (torch.randn(v, d, device="cuda") * 0.02).bfloat16()
            h = torch.randn(600, 1, d, device="cuda").bfloat16()
            ref = torch.nn.functional.linear(h, w)
            for m in [1, 2, 3, 7, 64, 300, 512]:
                for s in [0, 5]:
                    z = torch.nn.functional.linear(h[s : s + m], w)
                    if not torch.equal(z, ref[s : s + m]):
                        bad.append((d, m, s))
        print("BAD", bad)
        """
    )
    env = {
        **os.environ,
        "CUBLASLT_WORKSPACE_SIZE": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":0:0",
    }
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "BAD []" in out.stdout, out.stdout + out.stderr


# --------------------------------------------------------------------------
# Tensor parallelism: vocab-sharded heads, rows gathered in rank order
# (TP ranks are threads exchanging shards through a fake all_gather)
# --------------------------------------------------------------------------


class _ThreadTP:
    """TP ranks as threads; ``gather`` is a rank-order all-gather of the last dim.

    Every rank records the shapes it gathered, so lockstep (same collectives
    in the same order) can be asserted.
    """

    def __init__(self, tp):
        self.tp = tp
        self.slots = [None] * tp
        self.barrier = threading.Barrier(tp, timeout=60)
        self.calls = [[] for _ in range(tp)]

    def all_gather(self, rank, parts, x):
        self.calls[rank].append(tuple(x.shape))
        self.slots[rank] = x.clone()
        self.barrier.wait()
        for i, slot in enumerate(self.slots):
            parts[i].copy_(slot)
        self.barrier.wait()

    def gather(self, rank):
        def g(x):
            parts = [torch.empty_like(x) for _ in range(self.tp)]
            self.all_gather(rank, parts, x.contiguous())
            return torch.cat(parts, -1)

        return g

    def run(self, fn):
        """``fn(rank)`` on every rank (thread); returns the results."""
        out, errors = [None] * self.tp, []

        def body(r):
            try:
                out[r] = fn(r)
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                errors.append(exc)
                self.barrier.abort()

        threads = [threading.Thread(target=body, args=(r,)) for r in range(self.tp)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        return out


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_q_rows_from_sharded_heads_equal_the_full_vocab_row_bitwise(tp, dtype):
    """Each rank: logits of its head shard; gathered row == the TP=1 row."""
    torch.manual_seed(0)
    v, d, n = 48, 16, 7
    h = torch.randn(n, d).to(dtype)
    w = (torch.randn(v, d) * 2).to(dtype)
    full = torch.log_softmax(_logits(h, w).float(), -1)
    shards = w.chunk(tp, 0)
    grp = _ThreadTP(tp)

    def rank(r):
        assert shards[r].shape == (v // tp, d)  # stash: 1/TP of the head
        return gathered_log_softmax(_logits(h, shards[r]), grp.gather(r), 1 << 30)

    for row in grp.run(rank):
        assert torch.equal(row, full)
    # vocab_limit slices the gathered row, as at TP=1.
    out = grp.run(
        lambda r: gathered_log_softmax(_logits(h, shards[r]), grp.gather(r), 40)
    )
    assert torch.equal(out[0], torch.log_softmax(_logits(h, w)[:, :40].float(), -1))


@pytest.mark.parametrize("tp", [2, 4])
def test_gathered_p_rows_equal_dense_training_logprobs(tp):
    """Token log-probs (chunked) and rows on demand == the dense log-softmax."""
    torch.manual_seed(1)
    n, v = 11, 32
    logits = (torch.randn(n, v) * 3).to(torch.bfloat16)
    dense = torch.log_softmax(logits.float(), -1)
    tok = torch.randint(0, v, (n - 1,))
    grp = _ThreadTP(tp)

    def rank(r):
        rows = GatheredLogprobRows(
            logits.chunk(tp, -1)[r], grp.gather(r), 1 << 30, chunk_rows=3
        )
        return rows.token_logprobs(tok), rows(torch.tensor([4, 0, 10]))

    for lp, picked in grp.run(rank):
        assert torch.equal(lp, dense[torch.arange(n - 1), tok])
        assert torch.equal(picked, dense[[4, 0, 10]])
    # TP=1 through the same class.
    one = GatheredLogprobRows(logits, no_gather, 1 << 30, chunk_rows=4)
    assert torch.equal(one.token_logprobs(tok), dense[torch.arange(n - 1), tok])


@pytest.mark.parametrize("tp", [2, 4])
def test_engine_rule_decisions_identical_across_tp_ranks(tp):
    """Lazy scans on every rank: same decisions, generator state and gathers.

    p from sharded training logits, q rows from sharded head stashes and the
    engine's hidden states; all equal to the TP=1 rule on dense rows.
    """
    torch.manual_seed(2)
    v, d = 40, 8
    rejected = 0
    for trial in range(30):
        g = trial % 9
        hp = torch.randn(g + 1, d)
        w_p = torch.randn(v, d)
        w_q = w_p + torch.randn(v, d) * (0.05 if trial % 2 else 0.6)
        lp_dense = torch.log_softmax(_logits(hp, w_p).float(), -1)
        lq_dense = torch.log_softmax(_logits(hp[:g], w_q).float(), -1)
        draft = (
            torch.multinomial(lq_dense.exp(), 1).flatten()
            if g
            else torch.zeros(0).long()
        )
        engine_lq = lq_dense[torch.arange(g), draft]
        g_ref = torch.Generator().manual_seed(trial)
        ref = verify_engine_q(lp_dense, draft, engine_lq, lambda i: lq_dense[i], g_ref)
        rejected += ref[0] < g
        grp = _ThreadTP(tp)

        def rank(r, hp=hp, w_p=w_p, w_q=w_q, draft=draft, engine_lq=engine_lq):
            gather = grp.gather(r)
            p = GatheredLogprobRows(_logits(hp, w_p.chunk(tp)[r]), gather, 1 << 30)
            wq = w_q.chunk(tp)[r]

            def q_rows(idx):
                return gathered_log_softmax(_logits(hp[idx], wq), gather, 1 << 30)

            gen = torch.Generator().manual_seed(trial)  # same seed on every rank
            out = verify_engine_q(p, draft, engine_lq, q_rows, gen)
            return out, gen.get_state()

        for (tau, y, logp, n_eval), state in grp.run(rank):
            assert (tau, y, n_eval) == (ref[0], ref[1], ref[3])
            assert torch.equal(logp, ref[2])
            assert torch.equal(state, g_ref.get_state())
        assert all(c == grp.calls[0] for c in grp.calls)
    assert rejected >= 5


def test_head_stash_memory_divides_by_tp():
    gb = 1e9
    # Qwen3-8B head, bf16: 1.24 GB per version at TP=1.
    assert round(head_stash_bytes(151_936, 4096, 1) / gb, 2) == 1.24
    # 70B (Llama-3: V = 128,256, padded to 129,024 at TP=8; d = 8192).
    assert round(head_stash_bytes(128_256, 8192, 1) / gb, 2) == 2.10
    assert round(head_stash_bytes(129_024, 8192, 8) / gb, 2) == 0.26
    assert head_stash_bytes(129_024, 8192, 8, versions=3) == 3 * 16_128 * 8192 * 2


def test_stash_head_keeps_only_this_ranks_vocab_shard():
    """stash_head on each TP rank stores its V/TP x d shard (bytes / TP)."""
    from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker

    v, d, tp = 64, 8, 4
    w_full = torch.randn(v, d).to(torch.bfloat16)
    sizes = []
    for r in range(tp):
        w = object.__new__(
            ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class
        )
        w.rank = r
        w.model = torch.nn.Module()
        w.model.module = torch.nn.Module()
        w.model.module.output_layer = _Head(w_full.chunk(tp)[r])
        for ver in range(3):
            w.stash_head(version=ver, keep=[1, 2])
        assert sorted(w._head_stash) == [1, 2]
        assert torch.equal(w._head_stash[2], w_full.chunk(tp)[r])
        sizes.append(w.head_stash_bytes())
    assert sizes == [head_stash_bytes(v, d, tp, versions=2)] * tp
    assert sum(sizes) == head_stash_bytes(v, d, 1, versions=2)


def _tp_engine_workers(monkeypatch, tp):
    """``tp`` fake learner ranks (threads) of one replica, heads vocab-sharded."""
    from megatron.core import parallel_state as ps

    local = threading.local()
    grp = _ThreadTP(tp)
    w1, emb, heads = _engine_worker(monkeypatch)  # TP=1 reference worker
    monkeypatch.setattr(
        ps, "get_tensor_model_parallel_world_size", lambda *a, **k: local.tp
    )
    monkeypatch.setattr(ps, "get_tensor_model_parallel_group", lambda *a, **k: local)
    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda group=None: group.tp
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_gather",
        lambda parts, x, group=None: grp.all_gather(group.rank, parts, x),
    )
    workers = []
    for r in range(tp):
        w = object.__new__(type(w1))
        w.cfg, w.rank = w1.cfg, r
        w.model = torch.nn.Module()
        w.model.module = torch.nn.Module()
        w.model.module.output_layer = _Head(heads[-1].chunk(tp)[r])
        w._head_stash = {v: heads[v].chunk(tp)[r].clone() for v in range(VERSIONS - 1)}

        def training_forward(row, w=w):
            h = _hidden(row.cuda(), emb)
            return w.model.module.output_layer(h.unsqueeze(1))[0].transpose(0, 1)

        w._training_forward = training_forward
        workers.append(w)

    def run(r):
        local.tp, local.rank = tp, r
        return workers[r]

    return w1, run, emb, heads, grp, local


@cuda
@pytest.mark.parametrize("tp", [2, 4])
def test_engine_worker_at_tp_matches_tp1_bitwise(monkeypatch, capfd, tp):
    """verify_drafts_block_engine on TP thread ranks == the TP=1 worker."""
    monkeypatch.setenv("THUNDERSYNC_ENGINE_Q_CHECK", "1")
    w1, rank_worker, emb, heads, grp, local = _tp_engine_workers(monkeypatch, tp)
    rows, plens, segs, lqs, hids, keys = _engine_drafts(emb, heads)
    kw = dict(
        rows=rows,
        prompt_lens=plens,
        segments=segs,
        engine_logq=lqs,
        hidden=hids,
        vocab_limit=1 << 30,
        seed=3,
        keys=keys,
    )
    local.tp, local.rank = 1, 0
    ref = w1.verify_drafts_block_engine(**kw)
    assert sum(r["accepted"] < len(lqs[i]) for i, r in ref) >= 2
    res = grp.run(lambda r: rank_worker(r).verify_drafts_block_engine(**kw))
    for out in res:
        assert out == ref
    assert grp.calls[0] and all(c == grp.calls[0] for c in grp.calls)
    n_tok = sum(len(x) for x in lqs)
    # Every rank's rebuilt q rows match the engine's log q bitwise.
    out = capfd.readouterr().out
    assert out.count(f"tok={n_tok} exact={n_tok} max|dlogq|=0") == tp + 1


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


@pytest.fixture
def _no_ray(monkeypatch):
    monkeypatch.setattr(spec_mod.ray, "get", lambda x: x)


def _spec(learner, base, **kw):
    from test_speculative_driver import _spec as make

    return make(learner, base_handlers=base, verify_source="engine", **kw)


def test_engine_source_falls_back_when_the_engine_cannot_log(_no_ray, capsys):
    s = _spec({}, {"install_draft_hidden_log": lambda: False})
    assert s.verify_source == "scorer"
    assert "falling back" in capsys.readouterr().out
    s = _spec({}, {"install_draft_hidden_log": lambda: True})
    assert s.verify_source == "engine"


def test_engine_source_drafts_stash_and_verify_driver_transport(_no_ray):
    started, popped = [], []
    hid = {
        0: torch.ones(2, 4),  # draft 0: fully logged
        # draft 1: no hidden states (another engine lost them): dropped
    }

    def start(prompts, budget, interval, hidden_keys=None):
        started.append(hidden_keys)

    def pop(keys, lengths):
        popped.append((keys, lengths))
        return {k: hid[j] for j, k in enumerate(keys) if j in hid}

    def verify(rows, prompt_lens, segments, keys, engine_logq, hidden, **_):
        assert keys == ["1:0"] and engine_logq == [[-0.5, -0.25]]
        assert torch.equal(hidden[0], hid[0])
        return [
            (
                0,
                {
                    "accepted": 2,
                    "next": 0,
                    "logprobs": [-0.1, -0.2],
                    "draft_logq": [-0.5, -0.25],
                },
            )
        ]

    stashes = []
    s = _spec(
        {
            "verify_drafts_block_engine": verify,
            "stash_head": lambda **kw: stashes.append(kw) or 1,
        },
        {
            "install_draft_hidden_log": lambda: True,
            "start_drafts": start,
            "draft_request_ids": lambda: {},
            "collect_drafts": lambda records=None: [
                ([4, spec_mod_eod()], True, [-0.5, -0.25]),
                ([4, 5], False, [-0.5, -0.5]),
            ],
            "pop_draft_hidden": pop,
        },
        hidden_transport="driver",
    )
    prompts = [torch.tensor([1]), torch.tensor([2])]
    collect = asyncio.run(s.draft_streaming(prompts))
    drafts = collect()
    assert started[0] == ["0|1:0", "0|1:1"] and popped[0][1] == [2, 2]
    assert torch.equal(drafts[0][0][7], hid[0]) and drafts[1][0][7] is None
    s.score_prev(drafts)
    assert stashes == [{"version": 0, "keep": [0]}]
    s.verify(drafts)
    assert s.stats["spec/engine_q_dropped"] == 1.0
    (plan,) = s.plans[(1,)]
    assert plan.complete and plan.prefix_logprobs == [-0.1, -0.2]  # exactly log p
    assert (2,) not in s.plans  # dropped draft: its rollout decodes from scratch


def spec_mod_eod():
    from test_speculative_driver import EOD

    return EOD


def test_cohort_hidden_states_follow_segments_driver_transport(_no_ray):
    from thundersync_rl.speculative import cohort_draft

    collected = [[([5, 6], False, [-0.1, -0.2])], [([7], True, [-0.3])]]
    hids = [torch.zeros(2, 3), torch.ones(1, 3)]
    s = _spec(
        {},
        {
            "install_draft_hidden_log": lambda: True,
            "start_drafts": lambda prompts, budgets, interval, hidden_keys: None,
            "draft_request_ids": lambda: {},
            "collect_drafts": lambda records=None: collected.pop(0),
            "pop_draft_hidden": lambda keys, lengths: {keys[0]: hids.pop(0)},
        },
        hidden_transport="driver",
    )
    asyncio.run(s.draft_cohorts(0, {2: [torch.tensor([1, 1])]}))()
    asyncio.run(s.draft_cohorts(1, {}))()
    (d,) = s.cohorts[2]
    t = cohort_draft(d)
    assert t[6] == [-0.1, -0.2, -0.3]
    assert torch.equal(t[7], torch.cat([torch.zeros(2, 3), torch.ones(1, 3)]))
    assert d["segments"] == [(0, 0, 2), (1, 2, 3)]


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------


def _policy(**megatron):
    return {
        "megatron_cfg": {
            "batch_invariant_mode": True,
            "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": 1,
            **megatron,
        },
        "generation": {
            "temperature": 1.0,
            "backend": "megatron",
            "mcore_generation_config": {
                "num_speculative_tokens": 0,
                "logprobs_mode": "raw_logprobs",
            },
        },
    }


def _ts(**kw):
    return ThunderSyncConfig(
        **{
            "speculate": True,
            "verify_mode": "block",
            "verify_on": "learner",
            "verify_source": "engine",
            **kw,
        }
    )


def test_engine_source_config_valid():
    validate_engine_verify_source(_ts(), _policy())
    # Any TP with the same layout on both sides.
    validate_engine_verify_source(_ts(), _policy(tensor_model_parallel_size=2))
    p = _gen(tensor_model_parallel_size=4, materialize_only_last_token_logits=True)
    p["megatron_cfg"]["tensor_model_parallel_size"] = 4
    validate_engine_verify_source(_ts(), p)
    validate_engine_verify_source(
        ThunderSyncConfig(), _policy(batch_invariant_mode=False)
    )


def _gen(**mcore):
    p = _policy()
    p["generation"]["mcore_generation_config"].update(mcore)
    return p


def _tp_gen(tp, **mcore):
    p = _gen(**mcore)
    p["megatron_cfg"]["tensor_model_parallel_size"] = tp
    return p


@pytest.mark.parametrize(
    "ts_kw, policy, match",
    [
        ({"speculate": False}, _policy(), "speculate"),
        ({"verify_mode": "keyed"}, _policy(), "verify_mode"),
        ({"verify_on": "inference"}, _policy(), "verify_on"),
        ({"fused_verify_train": True}, _policy(), "fused_verify_train"),
        ({}, _policy(batch_invariant_mode=False), "batch_invariant_mode"),
        (
            {},
            _policy(virtual_pipeline_model_parallel_size=2),
            "virtual_pipeline_model_parallel_size > 1 is not implemented",
        ),
        (
            {},
            _gen(pipeline_model_parallel_size=2),
            r"pipeline_model_parallel_size \(2\) must equal",
        ),
        (
            {},
            _gen(expert_model_parallel_size=2),
            r"expert_model_parallel_size \(2\) must equal",
        ),
        ({}, _gen(num_speculative_tokens=2), "num_speculative_tokens"),
        ({}, _gen(logprobs_mode="processed_logprobs"), "logprobs_mode"),
        ({}, _gen(tensor_model_parallel_size=2), "tensor_model_parallel_size"),
        (
            {},
            _tp_gen(2, materialize_only_last_token_logits=False),
            "materialize_only_last_token_logits",
        ),
    ],
)
def test_engine_source_config_rejects(ts_kw, policy, match):
    with pytest.raises(ValueError, match=match):
        validate_engine_verify_source(_ts(**ts_kw), policy)
