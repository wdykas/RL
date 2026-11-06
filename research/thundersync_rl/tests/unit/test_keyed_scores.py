# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (see thundersync_rl/keyed_sampling.py).
"""keyed_scores (checks / negative controls only): the full keyed race of the wide sampler S. Its argmax is the
sampler's draw (S's fp64 race differs only on fp32 near-ties, absent from random logits), so its runner-up is the
runner-up of the SAME race; keyed_uniform is deterministic in (key id, position) and roughly uniform."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_scores_argmax_is_the_samplers_draw():
    from thundersync_rl import keyed_sampling_s as s_mod
    from thundersync_rl.keyed_scores import keyed_scores

    g = torch.Generator(device="cuda").manual_seed(0)
    n, v = 64, 50000
    logits = torch.randn(n, v, device="cuda", generator=g) * 3
    kid = torch.arange(1000, 1000 + n, dtype=torch.int64, device="cuda")
    pos = torch.arange(7, 7 + n, dtype=torch.int64, device="cuda") * 13
    temp = torch.ones(n, device="cuda")
    out = torch.empty(n, dtype=torch.int64, device="cuda")
    ws = torch.empty(s_mod.workspace_bytes(n, v), dtype=torch.uint8, device="cuda")
    s_mod.wide_sample(logits, kid, pos, temp, 12345, out=out, workspace=ws)
    sc = keyed_scores(logits, kid, pos, temp, 12345)
    assert torch.equal(sc.argmax(-1), out)
    # a different key or position changes the race
    sc2 = keyed_scores(logits, kid + 1, pos, temp, 12345)
    assert not torch.equal(sc2.argmax(-1), out)


@cuda
def test_keyed_uniform_deterministic_and_spread():
    from thundersync_rl.keyed_scores import keyed_uniform

    kid = torch.arange(0, 20000, dtype=torch.int64, device="cuda")
    pos = (kid * 7) % 1000
    u = keyed_uniform(kid, pos, 3)
    assert torch.equal(u, keyed_uniform(kid, pos, 3)) and not torch.equal(u, keyed_uniform(kid, pos, 4))
    assert float(u.min()) >= 0 and float(u.max()) < 1 and abs(float(u.mean()) - 0.5) < 0.02
