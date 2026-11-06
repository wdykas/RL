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
"""Run-readiness guards: weight sentinel, non-finite verification, draft collection
without frame fallback, multi-turn row prefixes, expert-bias + momentum."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
import torch

from thundersync_rl.block_verification import NonFiniteVerification, block_verify
from thundersync_rl.block_verification_fused import (
    block_verify_logits,
    block_verify_stats_lazy,
    block_verify_stats_lazy_grid,
)

# ---------------------------------------------------------------------------
# Weight sentinel (issue 7)
# ---------------------------------------------------------------------------




class _Layer(torch.nn.Module):
    def __init__(self, number, n_local, ep_rank):
        super().__init__()
        self.layer_number = number
        self.mlp = torch.nn.Module()
        self.mlp.experts = torch.nn.Module()
        self.mlp.experts.num_local_experts = n_local
        for j in range(n_local):
            setattr(self.mlp.experts, f"x{j}", None)
        self.mlp.experts.linear_fc1 = torch.nn.Module()
        for j in range(n_local):
            self.mlp.experts.linear_fc1.register_parameter(
                f"weight{j}", torch.nn.Parameter(torch.full((2,), float(100 * number + ep_rank * n_local + j)))
            )
        self.mlp.router = torch.nn.Module()
        self.mlp.router.register_buffer("expert_bias", torch.zeros(4))
        self.norm = torch.nn.LayerNorm(2)


class _Model(torch.nn.Module):
    def __init__(self, first_layer, n_layers, n_local, ep_rank):
        super().__init__()
        self.module = torch.nn.Module()  # a wrapper prefix, stripped
        self.module.decoder = torch.nn.Module()
        self.module.decoder.layers = torch.nn.ModuleList(
            [_Layer(first_layer + i + 1, n_local, ep_rank) for i in range(n_layers)]
        )
















class _Experts(torch.nn.Module):
    """A grouped-MLP experts module holding fp16 [E, N, K] buffers (fp16_moe layout)."""

    def __init__(self, n=3, alias=True, dtype=torch.bfloat16):
        super().__init__()
        self.num_local_experts = n
        for lin in ("linear_fc1", "linear_fc2"):
            m = torch.nn.Module()
            m.num_gemms = n
            for i in range(n):
                m.register_parameter(f"weight{i}", torch.nn.Parameter(torch.randn(4, 2).to(dtype)))
            setattr(self, lin, m)
        if alias:
            from nemo_rl.models.megatron.fp16_moe import _ensure_fp16_expert_weights

            _ensure_fp16_expert_weights(self)


def _expert_model(**kw):
    m = torch.nn.Module()
    m.decoder = torch.nn.Module()
    m.decoder.layers = torch.nn.ModuleList([torch.nn.Module()])
    m.decoder.layers[0].layer_number = 1
    m.decoder.layers[0].mlp = torch.nn.Module()
    m.decoder.layers[0].mlp.experts = _Experts(**kw)
    return m










# ---------------------------------------------------------------------------
# Expert-bias updates with momentum drafts (issue 9)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Non-finite verification inputs (issue 6)
# ---------------------------------------------------------------------------


def _rows(g, V=16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    lp = torch.log_softmax(torch.randn(g + 1, V, generator=gen), -1)
    lq = torch.log_softmax(torch.randn(g, V, generator=gen), -1)
    draft = torch.randint(0, V, (g,), generator=gen)
    return lp, lq, draft


def _stats(lp, lq, draft):
    g = draft.numel()
    ar = torch.arange(g)
    return lp[ar, draft].clone(), lq[ar, draft].clone()


def test_q_zero_draft_token_is_an_error_not_a_silent_cut():
    lp, lq, draft = _rows(6)
    lpx, lqx = _stats(lp, lq, draft)
    lqx[3] = float("-inf")  # the engine's logged q of the drafted token is 0
    z = torch.zeros(7)
    with pytest.raises(NonFiniteVerification, match="engine log q .* positions \\[3\\]"):
        block_verify_stats_lazy(lpx, lqx, z, z[:6], lambda i: lp[i], lambda i: lq[i], torch.Generator())
    with pytest.raises(NonFiniteVerification):
        block_verify_stats_lazy_grid(
            lpx, lqx, z, z[:6], lambda i: lp[i], lambda i: lq[i], torch.Generator(), (1.0, 1.5)
        )
    lq2 = lq.clone()
    lq2[3, draft[3]] = float("-inf")
    with pytest.raises(NonFiniteVerification):
        block_verify(lp, lq2, draft, torch.Generator())


def test_nan_learner_logits_are_an_error_not_a_device_assert():
    lp, lq, draft = _rows(5)
    p_logits = lp.clone()
    p_logits[5, :] = float("nan")  # the bonus row
    with pytest.raises(NonFiniteVerification, match="logsumexp"):
        block_verify_logits(p_logits, lq, draft, torch.Generator())
    lpx, lqx = _stats(lp, lq, draft)
    lpx[1] = float("nan")
    z = torch.zeros(6)
    with pytest.raises(NonFiniteVerification, match="learner log p"):
        block_verify_stats_lazy(lpx, lqx, z, z[:5], lambda i: lp[i], lambda i: lq[i], torch.Generator())
    # NaN only in the bonus row of log-prob rows (engine path, lse = 0): the
    # emission check catches it before multinomial.
    lpn = lp.clone()
    lpn[5] = float("nan")
    lpx, lqx = _stats(lp, lq, draft)
    lqx[:] = lpx  # accept everything -> bonus emission
    with pytest.raises(NonFiniteVerification, match="emission"):
        block_verify_stats_lazy(lpx, lqx, z, z[:5], lambda i: lpn[i], lambda i: lq[i], torch.Generator())


def test_p_zero_draft_token_is_a_legitimate_rejection():
    lp, lq, draft = _rows(4, seed=3)
    lpx, lqx = _stats(lp, lq, draft)
    lp2 = lp.clone()
    lp2[2, draft[2]] = float("-inf")
    lp2[2] = torch.log_softmax(lp2[2], -1)
    lpx2, _ = _stats(lp2, lq, draft)
    z = torch.zeros(5)
    for s in range(20):
        tau, y, logp, _ = block_verify_stats_lazy(
            lpx2, lqx, z, z[:4], lambda i: lp2[i], lambda i: lq[i], torch.Generator().manual_seed(s)
        )
        assert tau <= 2  # never keeps a p = 0 token
        assert torch.isfinite(logp).all()


def test_finite_inputs_unchanged_by_the_checks():
    # Same decisions as before the checks (the checks draw no randomness).
    lp, lq, draft = _rows(8, seed=5)
    out = [block_verify(lp, lq, draft, torch.Generator().manual_seed(s)) for s in range(30)]
    lpx, lqx = _stats(lp, lq, draft)
    z = torch.zeros(9)
    lazy = [
        block_verify_stats_lazy(lpx, lqx, z, z[:8], lambda i: lp[i], lambda i: lq[i], torch.Generator().manual_seed(s))[:2]
        for s in range(30)
    ]
    assert out == lazy


# ---------------------------------------------------------------------------
# collect_drafts never uses streamed frames for an unrecorded draft (issue 8)
# ---------------------------------------------------------------------------


class _Stream:
    def __init__(self, rid):
        self.request_id = rid


def _collect_worker():
    from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker

    w = object.__new__(ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    w._inference_loop = loop
    w.inference_client = SimpleNamespace(aborted=[], abort_request=lambda rid: w.inference_client.aborted.append(rid))
    w.dynamic_inference_engine = None
    return w, loop


def test_unrecorded_running_draft_is_an_error(monkeypatch):
    import thundersync_rl.worker as worker_mod

    monkeypatch.setattr(worker_mod.ray, "get", lambda refs: [])
    monkeypatch.setenv("THUNDERSYNC_COLLECT_FINAL_TIMEOUT_S", "0.1")
    w, _ = _collect_worker()
    w._draft_tokens, w._draft_logprobs = [[4, 5]], [[-0.4, -0.5]]
    w._draft_done, w._draft_streams = [False], {0: _Stream(1)}
    with pytest.raises(RuntimeError, match="final frame did not arrive"):
        w.collect_drafts(records={})


# ---------------------------------------------------------------------------
# Multi-turn rows are prefix extensions (coordinator: longctx finding 7/14)
# ---------------------------------------------------------------------------


class _Gen:
    def __init__(self, outs):
        self.outs = list(outs)

    async def generate_async(self, data, greedy=False):
        n = int(data["input_lengths"][0])
        tail = torch.tensor(self.outs.pop(0))
        yield 0, {
            "output_ids": torch.cat([data["input_ids"][0, :n], tail]).view(1, -1),
            "generation_lengths": torch.tensor([tail.numel()]),
        }


def _call(gen, ids):
    async def go():
        data = {"input_ids": torch.tensor([ids]), "input_lengths": torch.tensor([len(ids)])}
        return [r async for r in gen.generate_async(data)]

    return asyncio.run(go())






def test_deterministic_training_pins_mamba_ssm_backward_autotuners():
    # Imported before deterministic mode is set (as in the workers): every candidate kept,
    # so each process would autotune its own tile -> run-to-run gradient bits differ.
    import sys

    pytest.importorskip("mamba_ssm")
    from mamba_ssm.ops.triton import ssd_combined  # noqa: F401
    from mamba_ssm.utils.determinism import set_deterministic_mode
    from triton.runtime.autotuner import Autotuner

    from nemo_rl.models.megatron.mamba_ssd_parity import pin_mamba_ssm_autotuners

    def multi():
        return [
            n for m, mod in list(sys.modules.items()) if m.startswith("mamba_ssm.ops") and mod
            for n, a in vars(mod).items() if isinstance(a, Autotuner) and len(a.configs) > 1
        ]

    set_deterministic_mode(True)
    try:
        before = multi()
        n = pin_mamba_ssm_autotuners()
        assert n == len(before) and multi() == []
    finally:
        set_deterministic_mode(None)
