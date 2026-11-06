"""THUNDERSYNC_TRAIN_LP_CHECK: the training forward's log-probs vs the rollout log-probs, hidden by
force_on_policy_ratio (ratio := 1). Exact rows report 0; one perturbed token is caught."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn  # noqa: E402
from tests.unit.algorithms.test_loss_functions import _setup_clipped_pg_test_data  # noqa: E402


@pytest.mark.parametrize("bug", [0.0, 1e-6])
def test_training_forward_vs_rollout_logprobs(monkeypatch, bug):
    monkeypatch.setenv("THUNDERSYNC_TRAIN_LP_CHECK", "1")
    data, b, s, v = _setup_clipped_pg_test_data(device="cpu")
    loss_fn = ClippedPGLossFn(ClippedPGLossConfig(reference_policy_kl_penalty=0.0, force_on_policy_ratio=True))
    curr = torch.tensor([[-1.25, -0.5, -2.0]])
    data["advantages"][0, 1:] = torch.tensor([[1.0, -1.0, 2.0]])
    # The training forward's next-token log-probs (what prepare_loss_input hands the loss).
    loss_input = {"next_token_logprobs": curr.clone().requires_grad_(True)}
    # The rollout log-probs equal to them (exact run) ...
    data["generation_logprobs"][:, 1:] = curr
    # ... or with one token off by ``bug`` (e.g. a graph-replay / packing / capacity-rerun numerics bug).
    data["generation_logprobs"][0, 2] += bug
    loss, m = loss_fn(
        data=data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=torch.sum(data["sample_mask"].unsqueeze(-1) * data["token_mask"]),
        **loss_input,
    )
    assert m["probs_ratio"] == 1.0  # the ratio hides it
    assert m["train_gen_lp_tokens"] == 3
    if bug == 0.0:
        assert m["train_gen_lp_abs_diff_max"] == 0.0 and m["train_gen_lp_inexact_tokens"] == 0
    else:
        assert m["train_gen_lp_abs_diff_max"] > 0 and m["train_gen_lp_inexact_tokens"] == 1
