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
import pytest
import torch

from nemo_rl.algorithms.learning_metrics import learning_metrics, nucleus_mask

N, V = 2000, 256


@pytest.fixture
def base():
    torch.manual_seed(0)
    return torch.randn(N, V) * 2


def test_nucleus_mask_covers_p(base):
    lp = torch.log_softmax(base, -1)
    mask = nucleus_mask(lp, 0.9)
    mass = (lp.exp() * mask).sum(-1)
    assert torch.all(mass >= 0.9 - 1e-6)
    # removing the least likely kept token drops below p
    assert mask.sum(-1).min() >= 1


def test_identity(base):
    m = learning_metrics(base, base)
    assert m["kl_to_init"] == pytest.approx(0.0, abs=1e-5)
    assert m["novel_mass"] == pytest.approx(m["novel_mass_init"], rel=1e-5)
    assert m["top1_agree"] == 1.0


def test_temperature_sharpening_reduces_novel_mass(base):
    m = learning_metrics(base / 0.6, base)
    assert m["novel_mass"] < m["novel_mass_init"]
    assert m["temperature_explained"] > 0.99
    assert m["best_temperature"] == pytest.approx(0.6)
    assert m["top1_agree"] == 1.0


def test_selection_within_support_reduces_novel_mass(base):
    # boost the initial model's 3rd-ranked token: argmax changes, but mass stays inside the initial support
    z = base.clone()
    third = base.topk(3, -1).indices[:, 2]
    z[torch.arange(N), third] += 4
    m = learning_metrics(z, base)
    assert m["novel_mass"] < m["novel_mass_init"]
    assert m["top1_agree"] < 0.1
    assert m["temperature_explained"] < 0.5  # a global temperature cannot explain selection


def test_new_behaviour_increases_novel_mass(base):
    # boost tokens the initial model considered implausible
    z = base.clone()
    unlikely = base.argsort(-1)[:, 20]
    z[torch.arange(N), unlikely] += 12
    m = learning_metrics(z, base)
    assert m["novel_mass"] > 5 * m["novel_mass_init"]
    assert m["fork_novel_mass"] >= m["novel_mass"]


def test_chunking_invariant(base):
    a = learning_metrics(base / 0.8, base, chunk=N)
    b = learning_metrics(base / 0.8, base, chunk=97)
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=1e-4, abs=1e-6)
