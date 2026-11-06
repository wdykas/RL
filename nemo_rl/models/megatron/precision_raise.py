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
"""Targeted precision raises of the fp16 GEMM forward (``policy.megatron_cfg.precision_raise``).

Opt-in acceptance levers. Each raises the
precision of one class of rounding sites of the fp16-operand recipe; learner and engine call the
same functions, so engine == learner stays bitwise. None of them is needed for exactness.

Sites (``raise_sites``):

  ssd_scan_x3         Mamba SSD chunk-scan dots (C·states, CB·x) at Triton ``tf32x3`` instead of
                      TF32, in the training / prefill kernel and the engine's decode-row replay
                      kernel (``ssd_x3.py``)
  attn_qkv_hilo       attention linear_qkv input as an fp32 hi/lo split (``fp16_gemm.fp32_gemm``)
  attn_core_qp        q kept fp32 after the q/k norm and RoPE, near-fp32 batch-invariant attention
                      math (hi/lo q and P, fp32 softmax / O) on K and V rounded to fp16 as today
                      (the engine KV cache stays fp16), and the linear_proj input as hi/lo
                      (``hp_attention_mcore.py``)
  attn_core_kv        K and V kept as hi/lo in the attention (engine: hi/lo KV planes, 2x KV bytes);
                      needs attn_core_qp
  attn_core_fp32      alias: attn_core_qp + attn_core_kv (the offline "A.core" class)
  mamba_inout_hilo    Mamba in_proj (fused fp32 norm output) and out_proj (gated-norm output) inputs
                      as hi/lo
  shared_expert_hilo  shared-expert fc1 / fc2 inputs as hi/lo
  routed_fc2_hilo     routed-expert fc2 input (act(h) * prob) as hi/lo (2 fp16 expert GEMMs + an
                      fp32 combine; fc1 is never raised)

``combine_payload_fp16``: the routed-expert fc2 output (the combine payload) rounded to fp16 on
both sides before the fp64 combine (v 0.85-1.06x, n.s.; precision_recipe_v2 §1). Numerics only
here; the halved learner combine exchange is a separate byte cut.

Presets (``preset``): ``32k`` = ssd_scan_x3 + attn_qkv_hilo + attn_core_qp + attn_core_kv +
mamba_inout_hilo (full-Nano floor 9.7e-8 measured offline); ``128k`` = 32k + shared_expert_hilo
(3.5e-8); ``128k_fc2`` = 128k + routed_fc2_hilo (1.1e-8); ``32k_qp`` / ``128k_qp`` = the same
without attn_core_kv (fp16 KV cache). ``raise_sites`` adds to the preset.

Per-rank cost (no new communication except the hi/lo inputs' fp32 sequence-parallel all-gather:
2x the bytes of the fp16 gather at TP > 1 with sequence parallelism): hi/lo doubles the raised
GEMM's forward MMA work (one weight read: hi and lo rows are stacked), the fp32 GEMM input is saved
for backward (2x that activation's bytes); the backward stays bf16. attn_core_fp32 doubles the
engine's KV bytes per attention layer.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

SSD_SCAN_X3 = "ssd_scan_x3"
ATTN_QKV_HILO = "attn_qkv_hilo"
ATTN_CORE_QP = "attn_core_qp"
ATTN_CORE_KV = "attn_core_kv"
ATTN_CORE_FP32 = "attn_core_fp32"  # alias of qp + kv
MAMBA_INOUT_HILO = "mamba_inout_hilo"
SHARED_EXPERT_HILO = "shared_expert_hilo"
ROUTED_FC2_HILO = "routed_fc2_hilo"
SITES = (
    SSD_SCAN_X3,
    ATTN_QKV_HILO,
    ATTN_CORE_QP,
    ATTN_CORE_KV,
    MAMBA_INOUT_HILO,
    SHARED_EXPERT_HILO,
    ROUTED_FC2_HILO,
)
_ALIASES = {ATTN_CORE_FP32: (ATTN_CORE_QP, ATTN_CORE_KV)}
_P32K = (SSD_SCAN_X3, ATTN_QKV_HILO, ATTN_CORE_QP, ATTN_CORE_KV, MAMBA_INOUT_HILO)
PRESETS: dict[str, tuple[str, ...]] = {
    "none": (),
    "32k": _P32K,
    "128k": _P32K + (SHARED_EXPERT_HILO,),
    "128k_fc2": _P32K + (SHARED_EXPERT_HILO, ROUTED_FC2_HILO),
    # KV cache kept fp16 (attention core q/P raised only): the KV-bytes-free variants.
    "32k_qp": (SSD_SCAN_X3, ATTN_QKV_HILO, ATTN_CORE_QP, MAMBA_INOUT_HILO),
    "128k_qp": (SSD_SCAN_X3, ATTN_QKV_HILO, ATTN_CORE_QP, MAMBA_INOUT_HILO, SHARED_EXPERT_HILO),
}

_STATE: dict[str, Any] = {"sites": frozenset(), "combine16": False, "installed": False, "orig": {}}


def resolve(cfg: Optional[dict]) -> tuple[frozenset, bool]:
    """(raised sites, combine_payload_fp16) of a ``precision_raise`` config (None: nothing)."""
    if not cfg:
        return frozenset(), False
    preset = cfg.get("preset") or "none"
    if preset not in PRESETS:
        raise ValueError(f"precision_raise.preset must be one of {sorted(PRESETS)} (got {preset!r})")
    extra = []
    for x in cfg.get("raise_sites") or []:
        extra += list(_ALIASES.get(x, (x,)))
    bad = [s for s in extra if s not in SITES]
    if bad:
        raise ValueError(f"precision_raise.raise_sites: unknown {bad} (known: {list(SITES)} + {list(_ALIASES)})")
    out = frozenset(PRESETS[preset]) | frozenset(extra)
    if ATTN_CORE_KV in out and ATTN_CORE_QP not in out:
        raise ValueError("precision_raise: attn_core_kv needs attn_core_qp")
    return out, bool(cfg.get("combine_payload_fp16", False))


def canonical(cfg: Optional[dict]) -> tuple:
    """Comparable form (learner vs engine must agree): (sorted sites, combine16)."""
    sites, c16 = resolve(cfg)
    return tuple(sorted(sites)), c16


def validate_config(megatron_cfg: dict, inference_cfg: Optional[dict] = None) -> list[str]:
    cfg = megatron_cfg.get("precision_raise")
    errors = []
    try:
        sites, c16 = resolve(cfg)
    except ValueError as e:
        return [str(e)]
    if (sites or c16) and megatron_cfg.get("forward_gemm_precision") != "fp16":
        errors.append("policy.megatron_cfg.precision_raise requires forward_gemm_precision='fp16'.")
    if inference_cfg is not None and "precision_raise" in inference_cfg:
        if canonical(inference_cfg.get("precision_raise")) != canonical(cfg):
            errors.append("precision_raise must be the same in training and generation.")
    return errors


def sites() -> frozenset:
    return _STATE["sites"]


def on(site: str) -> bool:
    return site in _STATE["sites"]


def combine_payload_fp16() -> bool:
    return bool(_STATE["combine16"])


def hilo_input(module: Any) -> bool:
    """Whether this linear's GEMM input is raised to an fp32 hi/lo split (tagged at construction)."""
    return bool(getattr(module, "_ts_hilo_input", False))


def _tag(module: Any, names: Iterable[str]) -> None:
    for n in names:
        lin = getattr(module, n, None)
        if lin is not None:
            lin._ts_hilo_input = True


def _patch_init(cls: type, names: tuple[str, ...], key: str) -> None:
    orig = cls.__init__
    _STATE["orig"][(cls, key)] = orig

    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        _tag(self, names)

    cls.__init__ = __init__


def install(cfg: Optional[dict], megatron_cfg: Optional[dict] = None) -> tuple[frozenset, bool]:
    """Install the raises for this process (before the models are built; idempotent per config).

    Call after ``fp16_gemm.install`` in every training and generation worker.
    """
    sites_, c16 = resolve(cfg)
    if _STATE["installed"]:
        if (sites_, c16) != (_STATE["sites"], _STATE["combine16"]):
            raise RuntimeError("precision_raise.install called twice with different configs")
        return sites_, c16
    _STATE["sites"], _STATE["combine16"] = sites_, c16
    _STATE["installed"] = True
    if not sites_ and not c16:
        return sites_, c16
    from nemo_rl.models.megatron import fp16_gemm

    if not fp16_gemm.enabled():
        raise RuntimeError("precision_raise needs the fp16 GEMM forward (fp16_gemm.install first)")
    if MAMBA_INOUT_HILO in sites_:
        from megatron.core.ssm.mamba_mixer import MambaMixer

        _patch_init(MambaMixer, ("in_proj", "out_proj"), "mamba")
    if ATTN_QKV_HILO in sites_ or ATTN_CORE_QP in sites_:
        from megatron.core.transformer.attention import SelfAttention

        names = ((("linear_qkv",) if ATTN_QKV_HILO in sites_ else ())
                 + (("linear_proj",) if ATTN_CORE_QP in sites_ else ()))
        _patch_init(SelfAttention, names, "attn")
    if SHARED_EXPERT_HILO in sites_:
        from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

        _patch_init(SharedExpertMLP, ("linear_fc1", "linear_fc2"), "shared")
    if SSD_SCAN_X3 in sites_:
        from nemo_rl.models.megatron import ssd_x3

        ssd_x3.install()
    if ATTN_CORE_QP in sites_:
        from nemo_rl.models.megatron import hp_attention_mcore

        hp_attention_mcore.install(megatron_cfg or {}, qp=True, kv=ATTN_CORE_KV in sites_)
    print(f"[precision_raise] sites={sorted(sites_)} combine_payload_fp16={c16}", flush=True)
    return sites_, c16
