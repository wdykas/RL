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
"""Hysteresis rounding of fp32 masters into bf16 weights."""

import torch

from nemo_rl.models.megatron.param_hysteresis import hysteresis_copy_


def test_tau_half_is_round_to_nearest():
    g = torch.Generator().manual_seed(0)
    main = torch.randn(10000, generator=g) * 0.02
    w = (main + torch.randn(10000, generator=g) * 1e-4).to(torch.bfloat16)
    hysteresis_copy_(w, main, 0.5)
    # Exactly at a midpoint the two rules may differ; random data never hits one.
    assert torch.equal(w, main.to(torch.bfloat16))


def test_tau_one_keeps_weight_until_a_full_ulp():
    up = 2.0**-7  # bf16 spacing in [1, 2)
    down = 2.0**-8  # spacing in [0.5, 1): below 1.0, and around 0.5 going up
    w = torch.tensor([1.0, 1.0, 1.0, 1.0, -0.5, 0.0], dtype=torch.bfloat16)
    main = torch.tensor(
        [
            1.0 + 0.6 * up,  # past the midpoint, under an ulp: kept
            1.0 + 1.0 * up,  # a full ulp: moves
            1.0 - 0.9 * down,  # toward zero across 2^0, under the smaller ulp: kept
            1.0 - 1.1 * down,  # past it: moves
            -0.5 - 0.6 * down,  # away from zero: kept
            1e-3,  # zero always rounds to nearest
        ]
    )
    hysteresis_copy_(w, main, 1.0)
    expected = torch.tensor([1.0, 1.0 + up, 1.0, (1.0 - 1.1 * down), -0.5, 1e-3]).to(
        torch.bfloat16
    )
    assert torch.equal(w, expected)


def test_random_walk_flips_far_less_than_round_to_nearest():
    g = torch.Generator().manual_seed(1)
    master = torch.randn(200000, generator=g) * 0.02
    rtn = master.to(torch.bfloat16)
    hyst = rtn.clone()
    flips_rtn = flips_hyst = 0
    for _ in range(20):  # noisy-sign Adam-like steps of 1e-6
        master += torch.randn(master.shape, generator=g).sign() * 1e-6
        new_rtn = master.to(torch.bfloat16)
        flips_rtn += (new_rtn != rtn).sum().item()
        rtn = new_rtn
        before = hyst.clone()
        hysteresis_copy_(hyst, master, 1.0)
        flips_hyst += (hyst != before).sum().item()
        # The weight never strays more than an ulp from its master.
        ulp = torch.ldexp(
            torch.full_like(master, 2.0**-7), torch.frexp(hyst.float()).exponent - 1
        )
        assert ((hyst.float() - master).abs() <= ulp).all()
    # Measured 4x fewer over 20 steps (tiny weights, ulp < step, flip under both).
    assert flips_hyst * 3 < flips_rtn
