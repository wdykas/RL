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

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (Triton)")


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
