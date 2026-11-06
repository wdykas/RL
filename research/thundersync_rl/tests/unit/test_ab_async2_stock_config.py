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
"""R-C7 fairness: arm async2_stock differs from arm async2 ONLY in batch invariance / the exact-numerics recipe.

Regression (Super-L22 , tree 237dde26): the stock arm also switched chunked prefill on. A partial
prefill chunk is sampled (MCore then replaces the draw with the next prompt token), so the always-on wide-sampler key
check saw request r at position len(prompt) twice and raised SamplerKeyError at step 1; the engine event loop exited.
Chunked prefill is not a batch-invariance setting, so it stays off on both arms (and every BI arm keeps it off).

Pure YAML (the configs' ``defaults`` chain, merged as nemo_rl.utils.config does for these files); no GPU, no omegaconf.
"""

from pathlib import Path

import yaml

AB = Path(__file__).resolve().parents[2] / "configs" / "ab"

# Every leaf where async2_stock may differ from async2, and why (README R-C7 / ab_async2_stock.yaml).
ALLOWED = {
    "policy.megatron_cfg.batch_invariant_mode": "BI off (the arm's point)",
    "policy.megatron_cfg.zero_train_gen_mismatch": "BI off",
    "policy.megatron_cfg.forward_gemm_precision": "exact-numerics recipe off: bf16 GEMMs",
    "policy.megatron_cfg.fp32_residual_connection": "exact-numerics recipe off: bf16 residual",
    "policy.megatron_cfg.moe_permute_fusion": "stock MoE permute fusion (BI requires it off)",
    "policy.megatron_cfg.flash_attention_version": "BI pins FA4; non-BI flash path auto-selects (FA4 not in image)",
    "policy.generation.mcore_generation_config.mamba_inference_ssm_states_dtype":
        "fp32 pin = the value BI forces (MCore InferenceConfig); equal in effect",
}


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    defaults = cfg.pop("defaults", None)
    if defaults is None:
        return cfg
    base: dict = {}
    for d in [defaults] if isinstance(defaults, str) else list(defaults):
        base = _merge(base, _load((path.parent / str(d)).resolve()))
    return _merge(base, cfg)


def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        key = f"{pre}{k}"
        if isinstance(v, dict) and v:
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def _arms():
    return _flat(_load(AB / "ab_async2.yaml")), _flat(_load(AB / "ab_async2_stock.yaml"))


def test_stock_differs_from_async2_only_in_bi():
    bi, stock = _arms()
    diff = sorted(k for k in set(bi) | set(stock) if bi.get(k, "<unset>") != stock.get(k, "<unset>"))
    extra = [f"{k}: async2={bi.get(k, '<unset>')!r} stock={stock.get(k, '<unset>')!r}" for k in diff if k not in ALLOWED]
    assert not extra, "async2_stock differs from async2 beyond BI / the exact recipe:\n" + "\n".join(extra)
    assert stock["policy.megatron_cfg.batch_invariant_mode"] is False
    assert bi["policy.megatron_cfg.batch_invariant_mode"] is True


def test_chunked_prefill_off_on_both_arms():
    bi, stock = _arms()
    key = "policy.generation.mcore_generation_config.enable_chunked_prefill"
    assert bi[key] is False and stock[key] is False


def test_no_ab_arm_enables_chunked_prefill():
    """The wide/exact_int sampler key check assumes one sample per (request, position): no A/B arm may chunk prefills."""
    key = "policy.generation.mcore_generation_config.enable_chunked_prefill"
    on = [p.name for p in sorted(AB.glob("ab_*.yaml")) if _flat(_load(p)).get(key) is True]
    assert not on, f"arms with chunked prefill on: {on}"


def test_stock_ssm_state_dtype_matches_bi_forced_fp32():
    _, stock = _arms()
    assert stock["policy.generation.mcore_generation_config.mamba_inference_ssm_states_dtype"] == "float32"


def test_key_checks_skip_the_discarded_partial_prefill_draw():
    """Chunked prefill: MCore samples (then discards) the row of a request whose prompt is not done; the key checks
    skip that row, so the final chunk's kept draw at the same position is not a reuse (was: chunked prefill refused
    with the keyed samplers, async2_stock Super-L22)."""
    import numpy as np
    import pytest
    from types import SimpleNamespace

    worker = pytest.importorskip("thundersync_rl.worker")
    from thundersync_rl.key_check import KeyContinuity

    kc = KeyContinuity("test")
    ids = np.array([7, 9], dtype=np.int64)
    # step 1: request 9 decodes at 20; request 7 prefills its first chunk (prompt length 300: position 300)
    pos = worker._key_check_positions(SimpleNamespace(chunked_prefill_request_id=7), ids, np.array([300, 20]))
    assert pos.tolist() == [-1, 20]
    kc.observe(ids, pos)
    # step 2: request 7's final chunk draws position 300 again, kept; request 9 continues
    pos = worker._key_check_positions(SimpleNamespace(chunked_prefill_request_id=-1), ids, np.array([300, 21]))
    assert pos.tolist() == [300, 21]
    kc.observe(ids, pos)
    kc.observe(ids, np.array([301, 22]))
    assert kc.violations == 0
    # a real reuse still raises
    with pytest.raises(Exception, match="out of sequence"):
        kc.observe(ids, np.array([301, 23]))
