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


def _key_ids_with(records):
    """NolanesEngineMixin._nl_key_ids on a stub engine whose requests carry the given sampling-param attributes."""
    import numpy as np

    from thundersync_rl import nolanes_engine as ne

    class Entry:
        def __init__(self, attrs):
            sp = type("SP", (), {})()
            for k, v in attrs.items():
                setattr(sp, k, v)
            self.record = [type("Req", (), {"sampling_params": sp})()]

    eng = type("E", (), {"requests": {rid: Entry(a) for rid, a in records.items()}})()
    stub = type("W", (), {})()
    stub.__dict__["_nl"] = {"neg": "", "neg_tokens": 0}
    return ne.NolanesEngineMixin._nl_key_ids(stub, eng, np.asarray(sorted(records), dtype=np.int64)), stub


def test_key_rid_override_honoured_in_normal_mode():
    """A request carrying KEY_RID_ATTR draws under that key id in normal mode (not only under the key_reuse
    negative); requests without it keep their engine id; no negative tokens are counted."""
    from thundersync_rl import nolanes_engine as ne

    ids, stub = _key_ids_with({5: {}, 9: {ne.KEY_RID_ATTR: 5}, 11: {}})
    assert ids.tolist() == [5, 5, 11]
    assert stub.__dict__["_nl"]["neg_tokens"] == 0


@cuda
def test_continuation_under_key_rid_reproduces_the_original_draw():
    """A continuation (new engine id) with KEY_RID_ATTR = the original row's id draws exactly the original row's
    keyed draws at the same positions; the fresh-keys mutation (another key id) does not."""
    from thundersync_rl import keyed_sampling_s as s_mod
    from thundersync_rl import nolanes_engine as ne

    ids, _ = _key_ids_with({100: {}, 200: {ne.KEY_RID_ATTR: 100}})
    g = torch.Generator(device="cuda").manual_seed(1)
    n, v = 32, 40000
    logits = (torch.randn(1, v, device="cuda", generator=g) * 3).expand(n, v).contiguous()
    pos = torch.arange(50, 50 + n, dtype=torch.int64, device="cuda")
    temp = torch.ones(n, device="cuda")
    ws = torch.empty(s_mod.workspace_bytes(n, v), dtype=torch.uint8, device="cuda")

    def draws(kid):
        out = torch.empty(n, dtype=torch.int64, device="cuda")
        s_mod.wide_sample(logits, torch.full((n,), int(kid), dtype=torch.int64, device="cuda"), pos, temp, 777,
                          out=out, workspace=ws)
        return out

    orig, cont = draws(ids[0]), draws(ids[1])
    assert torch.equal(orig, cont)
    fresh = draws(int(ids[1]) ^ (1 << 40))  # tree_fresh_keys
    assert not torch.equal(orig, fresh)


@cuda
def test_keyed_top2_is_the_races_best_two():
    """keyed_top2 (proposals: tree_spec's MTP depth-1 sibling): out1 is S's draw and out2 the runner-up of the SAME
    keyed race (keyed_scores), on random logits (no fp32 near-ties)."""
    from thundersync_rl import keyed_sampling_s as s_mod
    from thundersync_rl.keyed_scores import keyed_scores

    g = torch.Generator(device="cuda").manual_seed(3)
    n, v = 48, 70000
    logits = torch.randn(n, v, device="cuda", generator=g) * 3
    kid = torch.arange(500, 500 + n, dtype=torch.int64, device="cuda")
    pos = torch.arange(3, 3 + n, dtype=torch.int64, device="cuda") * 11
    temp = torch.ones(n, device="cuda")
    s_out = torch.empty(n, dtype=torch.int64, device="cuda")
    s_mod.wide_sample(logits, kid, pos, temp, 99, out=s_out,
                      workspace=torch.empty(s_mod.workspace_bytes(n, v), dtype=torch.uint8, device="cuda"))
    o1, o2 = (torch.empty(n, dtype=torch.int64, device="cuda") for _ in range(2))
    s_mod.keyed_top2(logits, kid, pos, temp, 99, out1=o1, out2=o2,
                     workspace=torch.empty(s_mod.top2_workspace_bytes(n, v), dtype=torch.uint8, device="cuda"))
    assert torch.equal(o1, s_out)
    top = keyed_scores(logits, kid, pos, temp, 99).topk(2, dim=-1).indices
    assert torch.equal(o2, top[:, 1])
    # a row with a single finite logit has no runner-up
    one = torch.full((1, v), float("-inf"), device="cuda")
    one[0, 17] = 0.0
    s_mod.keyed_top2(one, kid[:1], pos[:1], temp[:1], 99, out1=o1[:1], out2=o2[:1],
                     workspace=torch.empty(s_mod.top2_workspace_bytes(1, v), dtype=torch.uint8, device="cuda"))
    assert int(o1[0]) == 17 and int(o2[0]) == -1
