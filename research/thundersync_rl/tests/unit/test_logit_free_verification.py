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
"""Logit-free lazy verification: chunked LM-head statistics and the worker path."""

import pytest
import torch
from thundersync_rl.logit_free import ChunkedHead


def _head(n=23, d=6, vocab=17, chunk=5, seed=0):
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(n, d, generator=gen)
    w = torch.randn(vocab, d, generator=gen)
    calls = []

    def logits_fn(h, weight):
        calls.append(h.shape[0])
        return h @ weight.t()

    return ChunkedHead(hidden, logits_fn, lambda: w, chunk), hidden, w, calls


def test_chunked_head_stats_and_rows():
    head, hidden, w, calls = _head()
    tokens = torch.randint(0, 17, (23,))
    lse, picked = head.stats(tokens, pin=[4, 22])
    assert calls == [5, 5, 5, 5, 3]  # fixed chunks, one pass
    full = torch.cat([hidden[a : a + 5] @ w.t() for a in range(0, 23, 5)])
    assert torch.equal(lse, torch.cat([torch.logsumexp(c, -1) for c in full.split(5)]))
    assert torch.equal(picked, full[torch.arange(23), tokens])
    # Rows: recomputed as part of their chunk (bitwise), pinned rows not at all.
    calls.clear()
    assert torch.equal(head.rows(torch.tensor([22, 4])), full[[22, 4]])
    assert calls == []
    assert torch.equal(head.rows(torch.tensor([13, 11, 7])), full[[13, 11, 7]])
    assert calls == [5, 5]  # chunk 2 once (cached), then chunk 1
    head.release()
    assert torch.equal(head.row(0), full[0])


def test_chunked_head_weight_swap_uses_new_tensor():
    head, hidden, w, _ = _head()
    head.stats(torch.zeros(23, dtype=torch.long))
    head.weight = lambda: w.clone()  # same values, different storage
    assert torch.equal(head.row(9), (hidden[5:10] @ w.t())[4])


# ---- worker path (GPU: the Triton residual kernel) ----

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
VOCAB, LIMIT, D = 40, 37, 8


class _Out(torch.nn.Module):
    """Stands in for Megatron's ColumnParallelLinear output layer."""

    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        self.sequence_parallel = False
        self.gather_output = False

    def forward(self, input_, weight=None, runtime_gather_output=None):
        w = self.weight if weight is None else weight
        return input_ @ w.t(), None


class _Scorer(torch.nn.Module):
    """Tiny causal LM with exactly representable logits (any GEMM shape agrees)."""

    def __init__(self, emb, out):
        super().__init__()
        self.emb = torch.nn.Parameter(emb)
        self.output_layer = _Out(out)

    def forward(self, input_ids, position_ids, attention_mask=None, **kw):
        x = self.emb[input_ids]  # [b, s, d]
        prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], 1)
        hidden = (x + 0.25 * prev).transpose(0, 1)  # [s, b, d]
        logits, _ = self.output_layer(hidden)
        return logits.transpose(0, 1).contiguous()


def _quantized(gen, *shape, step):
    return torch.randint(-2, 3, shape, generator=gen).float() * step


def _worker(monkeypatch):
    from megatron.core import parallel_state as ps
    from thundersync_rl.scorer_kernels import use_selective_output
    from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker

    for name, val in [
        ("get_data_parallel_world_size", 1),
        ("get_data_parallel_rank", 0),
        ("get_expert_model_parallel_world_size", 1),
        ("get_pipeline_model_parallel_world_size", 1),
        ("get_tensor_model_parallel_world_size", 1),
    ]:
        monkeypatch.setattr(ps, name, lambda *a, _v=val, **k: _v)
    gen = torch.Generator().manual_seed(0)
    versions = []
    emb = _quantized(gen, VOCAB, D, step=0.5)
    out = _quantized(gen, VOCAB, D, step=0.5)
    for _ in range(3):  # stashes 0, 1 and the current weights
        versions.append([emb.clone().cuda(), out.clone().cuda()])
        mask = torch.rand(VOCAB, D, generator=gen) < 0.15
        out = out + mask * _quantized(gen, VOCAB, D, step=0.5)
    scorer = _Scorer(*[t.clone() for t in versions[-1]]).cuda()
    assert use_selective_output(scorer)
    w = object.__new__(ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)
    w.model = torch.nn.Linear(1, 1)
    w.rank = 0
    w._fp32_scorer = lambda: scorer
    w._mfp32_map = [
        (scorer.emb, versions[-1][0]),
        (scorer.output_layer.weight, versions[-1][1]),
    ]
    w._weight_stash = {0: versions[0], 1: versions[1]}
    return w


def _inputs():
    gen = torch.Generator().manual_seed(1)
    rows, plens, segs, keys = [], [], [], []
    for i in range(24):
        plen = int(torch.randint(2, 6, (1,), generator=gen))
        g = [0, 1, 5, 12, 30][i % 5]
        rows.append(torch.randint(0, LIMIT, (plen + g,), generator=gen))
        plens.append(plen)
        cut = g // 2 if i % 3 else g
        segs.append([(0, 0, cut), (1, cut, g)])
        keys.append(f"k{i}")
    return rows, plens, segs, keys


@cuda
@pytest.mark.parametrize("chunk", ["4", "512"])
def test_logit_free_worker_matches_logits_path(monkeypatch, chunk):
    monkeypatch.setenv("THUNDERSYNC_LAZY_CHUNK", chunk)
    w = _worker(monkeypatch)
    rows, plens, segs, keys = _inputs()
    kw = dict(
        rows=rows,
        prompt_lens=plens,
        segments=segs,
        vocab_limit=LIMIT,
        keys=keys,
        batch_tokens=64,
        precision="fp32",
    )
    for seed in range(4):
        eager = dict(w.verify_drafts_block_stashed(seed=seed, **kw))
        lazy = dict(w.verify_drafts_block_stashed(seed=seed, lazy=True, **kw))
        assert w._prof.get("lazy_rows", 0) > 0  # the backward scan ran
        monkeypatch.setenv("THUNDERSYNC_LAZY_LOGITS", "1")
        lazy_logits = dict(w.verify_drafts_block_stashed(seed=seed, lazy=True, **kw))
        monkeypatch.delenv("THUNDERSYNC_LAZY_LOGITS")
        assert eager.keys() == lazy.keys() == set(range(len(rows)))
        n_rejected = 0
        for i, e in eager.items():
            assert (e["accepted"], e["next"]) == (
                lazy[i]["accepted"],
                lazy[i]["next"],
            )
            assert lazy_logits[i] == e  # same logits: bitwise
            torch.testing.assert_close(
                lazy[i]["logprobs"], e["logprobs"], rtol=1e-6, atol=1e-6
            )
            n_rejected += e["accepted"] < rows[i].numel() - plens[i]
        assert n_rejected > 3
