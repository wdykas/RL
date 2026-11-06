"""index_width: the engine refuses pools whose largest offsets do not fit the loaded kernels' index width (CPU)."""

import types

import pytest
import torch

from thundersync_rl import index_width as iw

NANO = dict(block_tokens=256, attn_layers=6, kv_heads_local=2, head_dim=128, mamba_slots=128,
            ssm_slot_elems=64 * 64 * 128, conv_slot_elems=6144 * 3)
I32 = {"kv_layer": 31, "kv_pool": 31, "mamba_slot": 31, "moe_rows": 31}
I64 = {k: 63 for k in I32}


def test_bug_8679738_pool_is_refused_under_int32_layer_offsets():
    # full Nano at buffer_gb 30 (10390 blocks): the whole-pool offset passes 2^31 (the tree_kv bug), one layer does not
    p = iw.Plan(kv_blocks=10390, **NANO)
    assert [v.split(":")[0] for v in iw.violations(p, I32)] == ["kv_pool"]
    # an int32 kernel over the whole buffer is safe only while the buffer has < 2^31 elements: 2 x 6 x 2731 x 65536
    assert iw.violations(iw.Plan(kv_blocks=2730, **NANO), I32) == []
    assert iw.violations(iw.Plan(kv_blocks=2731, **NANO), I32)
    assert iw.violations(p, I64) == []


def test_per_layer_and_slot_and_row_thresholds():
    assert [v.split(":")[0] for v in iw.violations(iw.Plan(kv_blocks=32768, **NANO), {**I64, "kv_layer": 31})] == \
        ["kv_layer"]
    assert iw.violations(iw.Plan(kv_blocks=32767, **NANO), {**I64, "kv_layer": 31}) == []
    many = dict(NANO, mamba_slots=4096)
    assert [v.split(":")[0] for v in iw.violations(iw.Plan(kv_blocks=1, **many), {**I64, "mamba_slot": 31})] == \
        ["mamba_slot"]
    # Ultra (topk 22, F 5120): 19065 tokens per rank per forward fit int32 row offsets, 19066 do not
    ok = iw.Plan(kv_blocks=1, **NANO, moe_rows=19065 * 22, moe_row_elems=5120)
    moe = iw.Plan(kv_blocks=1, **NANO, moe_rows=19066 * 22, moe_row_elems=5120)
    assert iw.violations(ok, {**I64, "moe_rows": 31}) == []
    assert [v.split(":")[0] for v in iw.violations(moe, {**I64, "moe_rows": 31})] == ["moe_rows"]
    assert iw.violations(moe, I64) == []


def test_plan_from_context_and_check_engine(monkeypatch):
    ctx = types.SimpleNamespace(
        memory_buffer=torch.empty(2, 6, 10390, 256, 2, 128, dtype=torch.bfloat16, device="meta"),
        mamba_ssm_states=torch.empty(23, 128, 64, 64, 128, device="meta"),
        mamba_conv_states=torch.empty(23, 128, 6144, 3, device="meta"),
    )
    p = iw.plan_from_context(ctx)
    assert (p.kv_blocks, p.attn_layers, p.kv_heads_local, p.mamba_slots, p.ssm_slot_elems) == (10390, 6, 2, 128, 524288)
    eng = types.SimpleNamespace(context=ctx)
    monkeypatch.setattr(iw, "widths", lambda: dict(I64))
    out = iw.check_engine(eng)
    assert out["max_kv_pool"] == 2 * 6 * 10390 * 65536 and out["bits_kv_layer"] == 63
    monkeypatch.setattr(iw, "widths", lambda: dict(I32))
    with pytest.raises(RuntimeError, match="kv_pool"):
        iw.check_engine(eng)


def test_widths_probe_runs():
    w = iw.widths()
    assert set(w) == set(I32) and all(b in (31, 63) for b in w.values())
