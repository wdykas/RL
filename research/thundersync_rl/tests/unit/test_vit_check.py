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
"""CPU tests of the verify-in-train debug check helpers (vit_check.py) and of the
exactness facts it measures on GPU, on a toy causal model."""

import torch

from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from thundersync_rl import vit_check


def _data(b=3, s=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    tm = torch.ones(b, s)
    tm[:, :2] = 0  # prompt
    return BatchedDataDict(
        {
            "input_ids": torch.randint(0, 50, (b, s), generator=g),
            "input_lengths": torch.full((b,), s),
            "token_mask": tm,
            "sample_mask": torch.ones(b),
            "advantages": torch.tensor([[0.0] * s, [1.5] * s, [-0.5] * s]),
            "generation_logprobs": -torch.rand(b, s, generator=g),
            "prev_logprobs": torch.zeros(b, s),
            "reference_policy_logprobs": torch.zeros(b, s),
        }
    )


def test_pick_mask_drop_copy():
    items = [{"data": _data(), "group": None}]
    pos = vit_check.pick_row(items)
    assert pos == (0, 1)  # row 0 has advantage 0
    c = vit_check.copy_items(items)
    vit_check.mask_row(c, pos, replace=True)
    d = c[0]["data"]
    assert d["sample_mask"][1] == 0 and d["token_mask"][1].sum() == 0 and d["advantages"][1].abs().sum() == 0
    assert not torch.equal(d["input_ids"][1], items[0]["data"]["input_ids"][1])  # replaced
    assert items[0]["data"]["sample_mask"][1] == 1  # original untouched
    mbs = [8]
    c = vit_check.copy_items(items)
    assert vit_check.drop_row(c, pos, mbs) and c[0]["data"].size == 2 and mbs == [2]
    mbs = [2]  # slice of 3 rows runs as 2 micro-batches: refuse (forward count would change)
    assert not vit_check.drop_row(vit_check.copy_items(items), pos, mbs)


def test_capture_logprobs_reads_training_p():
    data = _data()
    loss_fn = ClippedPGLossFn(ClippedPGLossConfig(reference_policy_kl_penalty=0.0, force_on_policy_ratio=True))
    lp = data["generation_logprobs"][:, 1:].clone().requires_grad_(True)
    stats = dict(p_tokens=0.0, p_diff_tokens=0.0, p_max_abs=0.0, p_shape_skips=0.0)
    with vit_check.capture_logprobs(stats):
        loss, _ = loss_fn(
            next_token_logprobs=lp, data=data,
            global_valid_seqs=torch.tensor(3.0), global_valid_toks=torch.tensor(12.0),
        )
    assert stats["p_tokens"] == 12 and stats["p_diff_tokens"] == 0 and stats["p_max_abs"] == 0
    loss.backward()  # the patched call returns the real loss / graph
    assert lp.grad is not None
    lp2 = lp.detach().clone()
    lp2[1, 3] += 1e-3
    with vit_check.capture_logprobs(stats):
        loss_fn(next_token_logprobs=lp2, data=data, global_valid_seqs=torch.tensor(3.0), global_valid_toks=torch.tensor(12.0))
    assert stats["p_diff_tokens"] == 1
    assert ClippedPGLossFn.__call__.__name__ == "__call__"  # restored


class _Toy(torch.nn.Module):
    """Causal toy LM: embedding, causal mean mixer, MLP, head (rows independent)."""

    def __init__(self, v=50, d=16):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(v, d)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(d, 4 * d), torch.nn.GELU(), torch.nn.Linear(4 * d, d))
        self.head = torch.nn.Linear(d, v)

    def forward(self, ids):
        h = self.emb(ids)
        h = torch.cumsum(h, 1) / torch.arange(1, ids.shape[1] + 1).view(1, -1, 1)
        return self.head(h + self.mlp(h))


def _grads(model, data, adv_scale=1.0):
    model.zero_grad()
    logits = model(data["input_ids"])[:, :-1]
    lp = torch.log_softmax(logits, -1).gather(-1, data["input_ids"][:, 1:, None])[..., 0]
    m = data["token_mask"][:, 1:] * data["sample_mask"][:, None]
    # GRPO surrogate at ratio 1: -A * exp(lp - lp.detach()) (gradient -A grad lp)
    loss = -(data["advantages"][:, 1:] * adv_scale * torch.exp(lp - lp.detach()) * m).sum() / 12.0
    loss.backward()
    return [p.grad.clone() for p in model.parameters()]


def test_masked_row_contributes_exactly_zero_toy():
    model = _Toy()
    a = _data()
    vit_check.mask_row([{"data": a}], (0, 1), replace=False)
    b = vit_check.copy_items([{"data": a}])[0]["data"]
    vit_check.mask_row([{"data": b}], (0, 1), replace=True)
    ga, gb = _grads(model, a), _grads(model, b)
    assert all(torch.equal(x, y) for x, y in zip(ga, gb))  # content of a masked row never reaches the gradient


def test_linear_correction_exact_in_math_not_bitwise_contract():
    """g(a_hat) + g(a - a_hat) == g(a) up to fp rounding (here float64 as 'exact' reference)."""
    model = _Toy().double()
    d = _data()
    for k in ("advantages", "generation_logprobs", "token_mask", "sample_mask"):
        d[k] = d[k].double()
    g = _grads(model, d)
    g1 = _grads(model, d, 0.7)
    g2 = _grads(model, d, 1 - 0.7)
    for x, y, z in zip(g, g1, g2):
        assert torch.allclose(x, y + z, rtol=1e-12, atol=1e-15)
