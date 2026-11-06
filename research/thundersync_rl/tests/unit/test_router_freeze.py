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
"""Router freeze (user rule 11-02): the fingerprint of a real MCore Router changes with any bit of its weight, bias or
expert_bias, reports trainable router parameters, and the per-step report check compares with the first step."""

import pytest
import torch

from thundersync_rl.router_freeze import check_router_reports, router_fingerprint


def _router(bias: bool):
    from megatron.core.transformer.moe.router import Router

    class _R(Router):  # a real MCore Router subclass (isinstance Router); routing / forward unused here
        def __init__(self, bias):  # the tensors Router.__init__ creates, without its process-group setup
            torch.nn.Module.__init__(self)
            gen = torch.Generator().manual_seed(0)
            self.weight = torch.nn.Parameter(torch.randn(8, 16, generator=gen, dtype=torch.float32))
            self.bias = torch.nn.Parameter(torch.randn(8, generator=gen)) if bias else None
            self.register_buffer("expert_bias", torch.randn(8, generator=gen))

        def routing(self, logits):
            raise NotImplementedError

        def forward(self, input):
            raise NotImplementedError

    return _R(bias)


def _flip(t, i):
    with torch.no_grad():
        v = t.data.view(-1).view(torch.int32)
        v[i] ^= 1


def test_fingerprint_sees_every_router_tensor_bit():
    model = torch.nn.ModuleDict({"layer0": _router(True), "layer1": _router(False)})
    for p in model.parameters():
        p.requires_grad = False
    fp0 = router_fingerprint(model)
    assert fp0["tensors"] == 5 and fp0["trainable"] == []  # 2 weights, 1 bias, 2 expert_bias
    assert router_fingerprint(model) == fp0  # deterministic
    for t, i in ((model["layer0"].weight, 37), (model["layer1"].weight, 0), (model["layer0"].bias, 3),
                 (model["layer1"].expert_bias, 7)):
        _flip(t, i)
        assert router_fingerprint(model)["fp"] != fp0["fp"]
        _flip(t, i)
        assert router_fingerprint(model) == fp0
    # swapping two elements of a row (row sums unchanged) still changes the column-weighted sum
    w = model["layer0"].weight.data
    a, b = w[2, 1].clone(), w[2, 5].clone()
    w[2, 1], w[2, 5] = b, a
    assert router_fingerprint(model)["fp"] != fp0["fp"]
    w[2, 1], w[2, 5] = a, b
    model["layer1"].weight.requires_grad = True
    assert router_fingerprint(model)["trainable"] == ["0.layer1.weight"]


def test_report_check_against_the_first_step():
    rep = [{"tensors": 5, "trainable": [], "fp": 11}, {"tensors": 5, "trainable": [], "fp": 12}]
    ok, key, n, tr = check_router_reports(rep, None)
    assert ok and n == 10 and tr == []
    assert check_router_reports(rep, key)[0]
    changed = [dict(rep[0], fp=13), rep[1]]
    assert not check_router_reports(changed, key)[0]
    trainable = [dict(rep[0], trainable=["0.x.weight"]), rep[1]]
    assert not check_router_reports(trainable, key)[0]
    assert not check_router_reports([{"tensors": 0, "trainable": [], "fp": 0}], None)[0]  # no router: vacuous


def test_stage_check_reads_the_finish_result_and_raises(monkeypatch):
    import thundersync_rl.nolanes_stage as ns

    st = ns.VerifyStage.__new__(ns.VerifyStage)
    st._router_fp0 = None
    res = {"nl_router_fp": [{"tensors": 5, "trainable": [], "fp": 1}]}
    st.check_router_frozen(res, 0)
    st.check_router_frozen(res, 1)
    with pytest.raises(RuntimeError, match="router weights / bias changed"):
        st.check_router_frozen({"nl_router_fp": [{"tensors": 5, "trainable": [], "fp": 2}]}, 2)
    with pytest.raises(RuntimeError, match="no router fingerprint"):
        st.check_router_frozen({}, 3)
