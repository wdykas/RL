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
"""Batch-invariant Mamba decode on Megatron-LM main (#8221, z-free buffers allocated by MCore)."""

import types

import pytest
import torch

from nemo_rl.models.megatron import mamba_bi_fused_decode as fd


def _buffers(z: bool, nheads=8, headdim=4, ngroups=2, dstate=16, chunk=128):
    b = types.SimpleNamespace(
        x=torch.zeros(3, chunk, nheads, headdim),
        B=torch.zeros(3, chunk, ngroups, dstate),
    )
    if z:
        b.z = torch.zeros(3, chunk, nheads, headdim)
    return b


def test_step_serves_supported_z_free_buffers_only(monkeypatch):
    seen = []

    def supported(nheads, headdim, ngroups, dstate, chunk, dtype, device):
        seen.append((nheads, headdim, ngroups, dstate, chunk))
        return nheads == 8

    monkeypatch.setattr(fd, "supported_config", supported)
    monkeypatch.setattr(fd, "_SERVES", {})
    main_buf = _buffers(z=False)  # Megatron-LM main: no z attribute at all
    assert fd._serves(main_buf)
    assert seen == [(8, 4, 2, 16, 128)]
    assert fd._serves(main_buf) and len(seen) == 1  # cached per buffer object
    assert not fd._serves(_buffers(z=False, nheads=6))  # main's own decode for an unsupported layer
    with_z = _buffers(z=True)  # older Megatron-Core: a layer allocated with z keeps MCore's path
    assert not fd._serves(with_z)


def test_mcore_main_buffers_are_detected_z_free():
    bid = pytest.importorskip("megatron.core.ssm.ops.mamba2.batch_invariant_decode")
    from nemo_rl.models.megatron import mamba_bi_decode

    import dataclasses

    has_z = "z" in {f.name for f in dataclasses.fields(bid.BatchInvariantDecodeBuffers)}
    assert mamba_bi_decode.mcore_buffers_z_free() == (not has_z)


def test_crossfade_wrapper_forwards_megatron_main_routing_kwargs(monkeypatch):
    """Megatron-LM main calls topk_routing_with_score_function with topk_indices / qb_* /
    allow_batch_invariant_sigmoid_topk; with cross-fade off they reach MCore's routing unchanged."""
    from nemo_rl.models.megatron import crossfade_routing as cf

    got = {}

    def orig(logits, topk, **kw):
        got.update(kw)
        return "mcore"

    monkeypatch.setitem(cf._STATE, "m", None)
    wrapped = cf._wrap(orig)
    out = wrapped(torch.zeros(2, 4), 2, topk_indices=None, allow_batch_invariant_sigmoid_topk=False)
    assert out == "mcore"
    assert got["topk_indices"] is None and got["allow_batch_invariant_sigmoid_topk"] is False


def test_context_raw_log_softmax_routes_to_the_canonical_lse(monkeypatch):
    """Megatron-LM main's prefill / mixed steps call F.log_softmax inline in the dynamic context: under
    NEMO_RL_VOCAB_LSE those rows must use the canonical-block LSE (the trainer's), other calls stay torch's."""
    pytest.importorskip("megatron.core.inference.contexts.dynamic_context")
    from megatron.core.inference.contexts import dynamic_context

    from nemo_rl.models.megatron import vocab_lse
    from nemo_rl.models.megatron import zero_train_gen_mismatch as zgm

    monkeypatch.setattr(dynamic_context, "F", dynamic_context.F)  # restored after the test
    zgm._route_context_raw_log_softmax()
    zgm._route_context_raw_log_softmax()  # idempotent
    F = dynamic_context.F
    assert getattr(F, "_canonical_vocab_lse", False)
    monkeypatch.setattr(vocab_lse, "enabled", lambda: True)
    monkeypatch.setattr(vocab_lse, "logprob_rows", lambda z: torch.full_like(z, 7.0))
    rows = torch.randn(3, 11)
    assert torch.equal(F.log_softmax(rows, dim=-1), torch.full_like(rows, 7.0))
    # not a raw row log-softmax: torch's
    x3 = torch.randn(1, 3, 11)
    assert torch.equal(F.log_softmax(x3, dim=2), torch.nn.functional.log_softmax(x3, dim=2))
    assert F.relu is torch.nn.functional.relu
    monkeypatch.setattr(vocab_lse, "enabled", lambda: False)
    assert torch.equal(F.log_softmax(rows, dim=-1), torch.nn.functional.log_softmax(rows, dim=-1))
