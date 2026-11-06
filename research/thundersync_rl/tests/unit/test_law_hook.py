"""The law-test hook of the learner's log-prob post-processor (LogprobsPostProcessor._law_stats): masked target
tokens only (the token at t is predicted by logits t-1), padding rows excluded, a correct sampler passes and
greedy (argmax) rows fail."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

import nemo_rl.models.megatron.train as mtrain  # noqa: E402
from thundersync_rl import law_check  # noqa: E402


def _run(monkeypatch, sampler):
    monkeypatch.setattr(mtrain, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(mtrain, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(mtrain, "get_context_parallel_group", lambda: None)
    g = torch.Generator().manual_seed(0)
    B, S, V = 6, 200, 50
    logits = torch.randn(B, S, V, generator=g) * 2
    ids = torch.zeros(B, S, dtype=torch.long)
    ids[:, 0] = 3
    for t in range(1, S):
        ids[:, t] = sampler(logits[:, t - 1], g)
    token_mask = torch.ones(B, S)
    token_mask[:, :20] = 0  # prompt
    sample_mask = torch.ones(B)
    sample_mask[-1] = 0  # a padding row
    data = {"input_ids": ids, "token_mask": token_mask, "sample_mask": sample_mask}
    pp = mtrain.LogprobsPostProcessor(
        cfg={"sequence_packing": {"enabled": False}}, law={"seed": 1, "acc": None}
    )
    pp._law_stats(logits, data, ids)
    return law_check.law_test(pp.law["acc"]), pp.law["acc"]


def test_hook_counts_masked_tokens_and_passes_exact_sampler(monkeypatch):
    r, acc = _run(monkeypatch, lambda z, g: torch.multinomial(torch.softmax(z, -1), 1, generator=g)[:, 0])
    assert r["n"] == 5 * (200 - 20)  # 5 real rows x generated tokens
    assert r["p"] > 1e-3


def test_hook_fails_a_wrong_sampler(monkeypatch):
    r, _ = _run(monkeypatch, lambda z, g: torch.multinomial(torch.softmax(z / 0.8, -1), 1, generator=g)[:, 0])
    assert r["p"] < 1e-6


def _hook_acc(monkeypatch, dp_rank, legacy=False):
    monkeypatch.setattr(mtrain, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(mtrain, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(mtrain, "get_context_parallel_group", lambda: None)
    monkeypatch.setattr(mtrain, "_law_dp_rank", lambda: dp_rank)
    g = torch.Generator().manual_seed(0)
    B, S, V = 2, 300, 30
    logits = torch.randn(B, S, V, generator=g)
    logits[..., 0] += 15.0  # near-deterministic rows: u ~ U, the PIT is the randomization itself
    ids = torch.zeros(B, S, dtype=torch.long)
    data = {"input_ids": ids, "token_mask": torch.ones(B, S), "sample_mask": torch.ones(B)}
    pp = mtrain.LogprobsPostProcessor(
        cfg={"sequence_packing": {"enabled": False}}, law={"seed": 1, "acc": None, "legacy": legacy}
    )
    pp._law_stats(logits, data, ids)
    return pp.law


def test_hook_pit_grid_differs_across_dp_replicas(monkeypatch):
    # ts-lawfix: the same micro-batch on two data-parallel replicas draws different PIT uniforms (u sums differ);
    # the legacy shadow grid is the same on both (the pre-fix behaviour, diagnostic only).
    a = _hook_acc(monkeypatch, 0, legacy=True)
    b = _hook_acc(monkeypatch, 1, legacy=True)
    nb = law_check.N_BINS
    assert float(a["acc"][nb + 1]) != float(b["acc"][nb + 1])
    assert float(a["acc_legacy"][nb + 1]) == float(b["acc_legacy"][nb + 1])
    assert torch.equal(a["acc"][: nb + 1], a["acc"][: nb + 1]) and float(a["acc"][nb]) == float(a["acc_legacy"][nb])
    assert torch.equal(_hook_acc(monkeypatch, 0)["acc"], a["acc"])  # reproducible per replica
