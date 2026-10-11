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
from thundersync_rl.block_verification import block_verify

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA (Triton)"
)


@pytest.mark.parametrize("noise", [0.01, 0.3])
def test_fused_matches_reference_draw_for_draw(noise):
    from thundersync_rl.block_verification_fused import block_verify_logits

    torch.manual_seed(0)
    vocab, g = 3000, 24
    for trial in range(50):
        p = torch.randn(g + 1, vocab, device="cuda") * 2
        q = p[:g] + torch.randn(g, vocab, device="cuda") * noise
        draft = torch.multinomial(torch.softmax(q, -1), 1).flatten()
        g1 = torch.Generator(device="cuda").manual_seed(trial)
        g2 = torch.Generator(device="cuda").manual_seed(trial)
        lp = torch.log_softmax(p, -1)
        ref = block_verify(lp, torch.log_softmax(q, -1), draft, g1)
        tau, y, logp = block_verify_logits(p, q, draft, g2)
        assert (tau, y) == ref
        kept = torch.cat([draft[:tau], torch.tensor([y], device="cuda")])
        torch.testing.assert_close(logp, lp[torch.arange(tau + 1, device="cuda"), kept])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("noise", [0.0, 0.003, 0.3, 2.0])
def test_lazy_logits_pathwise_identical(noise, dtype):
    """Lazy scan vs eager fused rule: same (tau, y), bitwise logp, same RNG state."""
    from thundersync_rl.block_verification_fused import (
        block_verify_logits,
        block_verify_logits_lazy,
    )

    torch.manual_seed(1)
    vocab = 3000
    rejected = 0
    for trial in range(60):
        g = trial % 40
        p = (torch.randn(g + 1, vocab, device="cuda") * 2).to(dtype)
        q = (p[:g].float() + torch.randn(g, vocab, device="cuda") * noise).to(dtype)
        draft = torch.multinomial(torch.softmax(q.float(), -1), 1).flatten()
        g1 = torch.Generator(device="cuda").manual_seed(trial)
        g2 = torch.Generator(device="cuda").manual_seed(trial)
        tau, y, logp = block_verify_logits(p, q, draft, g1)
        tau2, y2, logp2 = block_verify_logits_lazy(p, q, draft, g2)
        assert (tau, y) == (tau2, y2)
        assert torch.equal(logp, logp2)
        assert torch.equal(g1.get_state(), g2.get_state())
        rejected += tau < g
    if noise >= 0.3:
        assert rejected > 10


@pytest.mark.parametrize("scale", [1.0, 30.0, 300.0, 3000.0])
def test_residual_mass_within_lazy_slack(scale):
    """The computed r_i never exceeds b_i by more than the lazy slack.

    Adversarial: q is a point mass where p is smallest, so r_i ~ b_i * sum_v p_i(v).
    """
    from thundersync_rl.block_verification import lazy_slack
    from thundersync_rl.block_verification_fused import residual_masses

    torch.manual_seed(2)
    vocab = 152_064
    p = torch.randn(16, vocab, device="cuda") * scale
    q = torch.full_like(p, -1e4)
    q.scatter_(1, p.argmin(-1, keepdim=True), 0.0)
    lse_p, lse_q = torch.logsumexp(p, -1), torch.logsumexp(q, -1)
    b = torch.rand(16, device="cuda", dtype=torch.float64)
    b[0] = 1.0
    r = residual_masses(p, q, lse_p, lse_q, b)
    excess = r / b - 1
    assert (excess <= lazy_slack(lse_p) - 1e-4).all(), excess.max()
