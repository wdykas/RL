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
"""The cross-fade band is part of the policy definition (F3, 11-06): every colo arm routes with the same band.

- XS (colo mode) carries the band from its thundersync block into policy.megatron_cfg (it used to force plain top-k);
- the stock SPR-BI arm sets it directly in policy.megatron_cfg (no thundersync block);
- every config under configs/colo resolves to the same band as colo.yaml;
- colo mode refuses a missing band without a stated crossfade_off_reason, like the other modes.
"""

import copy
import os

import pytest

from thundersync_rl import colo, nolanes

KEYS = ("crossfade_routing_m", "crossfade_routing_m_profile", "crossfade_routing_extra",
        "crossfade_routing_ramp_stopgrad", "crossfade_routing_sparse", "crossfade_routing_sparse_engine")
CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "colo")


def _load(name):
    from omegaconf import OmegaConf

    from nemo_rl.utils.config import load_config

    return OmegaConf.to_container(load_config(os.path.join(CFG_DIR, name)), resolve=False)


def _band(cfg):
    """The band an arm routes with: the thundersync block of a ThunderSync mode (apply_crossfade copies it into
    policy.megatron_cfg), else policy.megatron_cfg (stock arms)."""
    ts = cfg.get("thundersync") or {}
    src = ts if ts.get("mode") else (cfg.get("policy") or {}).get("megatron_cfg") or {}
    return {k: src.get(k) for k in KEYS}


def test_every_colo_config_routes_with_the_colo_band():
    ref = _band(_load("colo.yaml"))
    assert ref["crossfade_routing_m"] is not None and ref["crossfade_routing_m"] > 0
    names = sorted(f for f in os.listdir(CFG_DIR) if f.endswith(".yaml"))
    assert {"colo.yaml", "colo_sprbi.yaml", "colo_isspr.yaml"} <= set(names)
    for name in names:
        assert _band(_load(name)) == ref, name


def _xs_config():
    cfg = _load("colo.yaml")
    cfg["policy"]["megatron_cfg"].setdefault("env_vars", {})["NEMO_RL_MOE_DEVICE_DISPATCH"] = "1"
    return cfg


def test_colo_mode_applies_the_band_to_megatron_cfg():
    cfg = _xs_config()
    c = colo.ColoConfig.from_config(cfg)
    mcfg = {}
    nolanes.apply_crossfade(c, mcfg)
    assert {k: mcfg[k] for k in KEYS} == _band(cfg)
    # the stock arm's megatron_cfg band equals what apply_crossfade writes for XS (same policy definition)
    sprbi = _load("colo_sprbi.yaml")["policy"]["megatron_cfg"]
    assert {k: sprbi[k] for k in KEYS} == {k: mcfg[k] for k in KEYS}


def test_colo_mode_refuses_a_missing_band():
    cfg = _xs_config()
    off = copy.deepcopy(cfg)
    off["thundersync"]["crossfade_routing_m"] = None
    p = nolanes.crossfade_problems(colo.ColoConfig.from_config(off), off["policy"]["megatron_cfg"])
    assert any("cross-fade routing must be on" in s for s in p)
    off["thundersync"]["crossfade_off_reason"] = "debug"
    assert nolanes.crossfade_problems(colo.ColoConfig.from_config(off), off["policy"]["megatron_cfg"]) == []
    assert nolanes.crossfade_problems(colo.ColoConfig.from_config(cfg), cfg["policy"]["megatron_cfg"]) == []


def test_colo_config_rejects_unknown_thundersync_keys():
    cfg = _xs_config()
    cfg["thundersync"]["crossfade_routing_mm"] = 0.01
    with pytest.raises(ValueError, match="unknown keys"):
        colo.ColoConfig.from_config(cfg)


S30_PROFILE = [[0.0, 0.2608, 0.3], [0.2608, 0.7826, 1.0], [0.7826, 1.0, 0.5]]


def test_s30_profile_maps_the_nano_moe_layer_groups():
    """routing_crux cfS30 at Nano (23 MoE layers): m x 0.3 on MoE layers 0-5, x 1.0 on 6-17, x 0.5 on 18-22."""
    from nemo_rl.models.megatron.crossfade_routing import layer_bands

    layers = [3 * j + 2 for j in range(23)]  # any global layer numbers: only the MoE ordinal matters
    bands = [layer_bands(0.03, S30_PROFILE, layers)[ln] for ln in layers]
    assert bands == [0.03 * 0.3] * 6 + [0.03] * 12 + [0.03 * 0.5] * 5


def test_band_rule_is_batch_invariant_at_the_nano_shape(device="cpu"):
    """Every band of the F3 study (E 128, k 6; m .009 / .015 / .03 with extra 4, m .01 / .03 with extra 8): a row's
    gates and ids are bitwise the same whatever batch it is routed in (slices, permutations)."""
    import torch

    from nemo_rl.models.megatron import crossfade_routing as cf

    g = torch.Generator().manual_seed(11)
    x = torch.randn(257, 128, generator=g).to(device)
    b = (torch.randn(128, generator=g) * 0.01).to(device)
    perm = torch.randperm(257, generator=g).to(device)
    for m, extra in ((0.009, 4), (0.015, 4), (0.03, 4), (0.01, 8), (0.03, 8)):
        full_p, full_i = cf.crossfade_route(x, b, 6, extra, m, 2.5)
        for rows in (slice(0, 1), slice(9, 10), slice(3, 70), slice(256, 257)):
            p, i = cf.crossfade_route(x[rows].clone(), b, 6, extra, m, 2.5)
            assert torch.equal(p, full_p[rows]) and torch.equal(i, full_i[rows]), (m, extra, rows)
        p, i = cf.crossfade_route(x[perm], b, 6, extra, m, 2.5)
        assert torch.equal(p, full_p[perm]) and torch.equal(i, full_i[perm]), (m, extra)


def test_effective_config_dump_shows_the_applied_band(tmp_path=None):
    """REVIEW batch 49 condition 2: the effective-config dump is written at override parsing, before a mode's
    configure_driver copies its thundersync band into policy.megatron_cfg; append_applied_band rewrites the dump's
    megatron_cfg band with what was applied (other keys untouched)."""
    import tempfile

    import yaml

    import run_nolanes_sc as rns

    d = tmp_path or tempfile.mkdtemp()
    out = os.path.join(str(d), "effective_config.yaml")
    with open(out, "w") as f:
        yaml.safe_dump({"policy": {"megatron_cfg": {"crossfade_routing_m": 0.01, "other": 7}},
                        "thundersync": {"mode": "isspr", "crossfade_routing_m": 0.03}}, f)
    cfg = _xs_config()
    cfg["thundersync"].update(crossfade_routing_m=0.03, crossfade_routing_m_profile=S30_PROFILE,
                              crossfade_routing_extra=4)
    nolanes.apply_crossfade(colo.ColoConfig.from_config(cfg), cfg["policy"]["megatron_cfg"])
    rns.append_applied_band(out, cfg)
    with open(out) as f:
        dm = yaml.safe_load(f)["policy"]["megatron_cfg"]
    assert dm["other"] == 7
    assert dm["crossfade_routing_m"] == 0.03 and dm["crossfade_routing_extra"] == 4
    assert dm["crossfade_routing_m_profile"] == S30_PROFILE
    assert dm["crossfade_routing_sparse"] is True and dm["crossfade_routing_sparse_engine"] is True


def test_engine_band_scale_negative_is_refused_outside_checks():
    """REVIEW batch 49 note: THUNDERSYNC_DEBUG_CF_ENGINE_M_SCALE (engine band != learner band) is refused by
    crossfade_routing.configure unless CHECKS or the cf_engine_m negative is on, on every path (stock loop too)."""
    from types import SimpleNamespace

    from nemo_rl.models.megatron import crossfade_routing as cf

    cfg = {"megatron_cfg": {"crossfade_routing_m": 0.03, "crossfade_routing_extra": 4}}
    keys = ("THUNDERSYNC_DEBUG_CF_ENGINE_M_SCALE", "THUNDERSYNC_NL_CHECKS", "THUNDERSYNC_NL_NEG")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        os.environ["THUNDERSYNC_DEBUG_CF_ENGINE_M_SCALE"] = "2"
        with pytest.raises(ValueError, match="negative control"):
            cf.configure(cfg, SimpleNamespace(moe_router_topk=6))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
