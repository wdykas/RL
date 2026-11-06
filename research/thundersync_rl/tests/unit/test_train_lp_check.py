"""loss_fn.train_lp_check (set by nolanes.configure_driver): the training forward's log-probs vs the rollout log-probs, hidden by
force_on_policy_ratio (ratio := 1). Exact rows report 0; one perturbed token is caught. The check runs on both ratio
paths: BUGS #15, the stock importance-sampled loss (nl_base, force_on_policy_ratio false) reported nothing and every
nl_base run died at step 0 with "train-lp sentinel missing"."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn  # noqa: E402
from tests.unit.algorithms.test_loss_functions import _setup_clipped_pg_test_data  # noqa: E402


@pytest.mark.parametrize("forced", [True, False], ids=["force_on_policy_ratio", "importance_sampled"])
@pytest.mark.parametrize("bug", [0.0, 1e-6])
def test_training_forward_vs_rollout_logprobs(monkeypatch, bug, forced):
    data, b, s, v = _setup_clipped_pg_test_data(device="cpu")
    loss_fn = ClippedPGLossFn(ClippedPGLossConfig(reference_policy_kl_penalty=0.0, force_on_policy_ratio=forced,
                                                  train_lp_check=True))
    assert loss_fn.train_lp_check
    assert "train_gen_lp_inexact_tokens" in loss_fn.metric_normalizations
    curr = torch.tensor([[-1.25, -0.5, -2.0]])
    data["advantages"][0, 1:] = torch.tensor([[1.0, -1.0, 2.0]])
    # The training forward's next-token log-probs (what prepare_loss_input hands the loss).
    loss_input = {"next_token_logprobs": curr.clone().requires_grad_(True)}
    # The rollout log-probs equal to them (exact run) ...
    data["generation_logprobs"][:, 1:] = curr
    # ... or with one token off by ``bug`` (e.g. a graph-replay / packing / capacity-rerun numerics bug).
    data["generation_logprobs"][0, 2] += bug
    if not forced:
        # Importance-sampled path (nl_base): prev_logprobs from the separate log-prob forward (equal on an exact run).
        data["prev_logprobs"][:, 1:] = curr
    loss, m = loss_fn(
        data=data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=torch.sum(data["sample_mask"].unsqueeze(-1) * data["token_mask"]),
        **loss_input,
    )
    if forced:
        assert m["probs_ratio"] == 1.0  # the ratio hides it
    assert m["train_gen_lp_tokens"] == 3
    if bug == 0.0:
        assert m["train_gen_lp_abs_diff_max"] == 0.0 and m["train_gen_lp_inexact_tokens"] == 0
    else:
        assert m["train_gen_lp_abs_diff_max"] > 0 and m["train_gen_lp_inexact_tokens"] == 1


def test_sentinel_off_by_default_and_not_an_env_switch(monkeypatch):
    monkeypatch.setenv("THUNDERSYNC_TRAIN_LP_CHECK", "1")  # the old env switch no longer turns it on
    for forced in (True, False):
        loss_fn = ClippedPGLossFn(ClippedPGLossConfig(reference_policy_kl_penalty=0.0, force_on_policy_ratio=forced))
        assert not loss_fn.train_lp_check
