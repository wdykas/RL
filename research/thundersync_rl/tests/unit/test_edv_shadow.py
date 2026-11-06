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
"""EDV shadow counter (DESIGN §22.5): log only.

1. The DS2 scan with the shadow record on equals the scan with it off: outcomes at every envelope, log p of the drafted
   tokens, positions evaluated and the generator's final state, bitwise.
2. The record's eta_g / b_g are the scan's own draws and weights (re-derived independently).
3. The driver's per-row result (the training rows' tokens and generation log-probs, i.e. every loss input the
   verification produces) is identical with and without the shadow field, and the jsonl lines are written.
"""

import json
import math

import pytest
import torch

from test_ds2 import CASES, _case, _lse, _sharded, cuda
from thundersync_rl import ds2, edv_shadow
from thundersync_rl.block_verification import acceptance_weights
from thundersync_rl.speculative import _null_result

ENVS = (1.0, 1.1, 1.5)


@cuda
@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("seed", range(3))
def test_scan_identical_with_shadow_on_and_off(case, seed, monkeypatch):
    zp, zq, vlim, draft = _case(seed, **CASES[case])
    lqx = (zq[:, :vlim].float() - _lse(zq, vlim)[:, None])[torch.arange(draft.numel(), device="cuda"), draft]
    monkeypatch.setattr(ds2, "SHADOW", False)
    ds2.LAST_ROW = None
    off = _sharded(zp, zq, vlim, draft, lqx, 200 + seed, ENVS)
    assert ds2.take_last_row() is None
    monkeypatch.setattr(ds2, "SHADOW", True)
    on = _sharded(zp, zq, vlim, draft, lqx, 200 + seed, ENVS)
    rec = ds2.take_last_row()
    assert rec is not None and ds2.take_last_row() is None
    out_off, lpx_off, gen_off, n_off = off
    out_on, lpx_on, gen_on, n_on = on
    assert out_on == out_off
    assert torch.equal(lpx_on, lpx_off)
    assert torch.equal(gen_on, gen_off)
    assert n_on == n_off
    # 2. the record is the scan's own eta_g and b_g (independent re-derivation)
    g = draft.numel()
    gen = torch.Generator(device="cuda").manual_seed(200 + seed)
    eta = torch.rand(g + 1, generator=gen, dtype=torch.float64, device="cuda")
    assert rec["g"] == g and rec["eta_g"] == float(eta[g])
    for M in ENVS:
        b = acceptance_weights(lpx_on - lqx, M)
        assert rec["b_g"][M] == pytest.approx(float(b[g]), rel=1e-12, abs=1e-300)
        tau, _, _ = out_on[M]
        assert (tau == g) == (rec["eta_g"] <= rec["b_g"][M])


def _res(g=6, tau=6, y=5, shadow=True):
    r = {
        "null_grid": {1.0: (tau, y, -0.5), 1.1: (2, -1, None)},
        "null_log_wstar": 0.01,
        "null_lr_moments": (g, -0.02, 0.003),
        "token_logprobs": [-1.0 - 0.1 * j for j in range(g)],
        "draft_logq": [-1.01 - 0.1 * j for j in range(g)],
        "p_zero_tokens": 0,
    }
    if shadow:
        r["edv_shadow"] = {"g": g, "eta_g": 0.3, "b_g": {1.0: 0.97, 1.1: 0.88}, "key": "7:abc:3"}
    return r


@pytest.mark.parametrize("m", [1.0, 1.1])
def test_driver_result_identical_and_jsonl_written(m, tmp_path, monkeypatch):
    monkeypatch.delenv("THUNDERSYNC_EDV_SHADOW_PATH", raising=False)
    path = edv_shadow.configure(str(tmp_path / "logs"))
    assert path == str(tmp_path / "edv_shadow.jsonl")
    assert edv_shadow.run_dir_of(str(tmp_path / "logs" / "exp_001")) == str(tmp_path)
    edv_shadow.set_step(4)
    with_shadow = _null_result(_res(), m)
    without = _null_result(_res(shadow=False), m)
    assert with_shadow == without
    edv_shadow.set_step(5)  # flushes step 4
    lines = [json.loads(x) for x in open(path)]
    assert len(lines) == 1
    rec = lines[0]
    assert rec["step"] == 4 and rec["key"] == "7:abc:3" and rec["m"] == m and rec["eta_g"] == 0.3
    if m == 1.0:
        assert rec["kind"] == "whole" and rec["whole"] and rec["tau"] == 6 and rec["b_g"] == 0.97
    else:
        assert rec["kind"] == "bot" and not rec["whole"] and rec["tau"] == 2 and rec["b_g"] == 0.88
    assert rec["lr_n"] == 6 and math.isclose(rec["lr_sum"], -0.02)


def test_unused_rows_and_missing_records_write_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("THUNDERSYNC_EDV_SHADOW_PATH", str(tmp_path / "x.jsonl"))
    edv_shadow.configure(None)
    edv_shadow.set_step(1)
    _null_result(_res(), None)
    _null_result(_res(shadow=False), 1.0)
    edv_shadow.flush()
    assert not (tmp_path / "x.jsonl").exists()


def test_stage_configures_from_the_controller_logger_at_first_step(tmp_path, monkeypatch):
    """The stage is built before the controller's Logger exists; the path is taken at the first step."""
    import types

    import thundersync_rl.nolanes_stage as ns

    monkeypatch.delenv("THUNDERSYNC_EDV_SHADOW_PATH", raising=False)
    monkeypatch.setattr(edv_shadow, "_PATH", None)
    st = ns.VerifyStage.__new__(ns.VerifyStage)
    st.sc = types.SimpleNamespace()
    with pytest.raises(RuntimeError, match="no run dir"):
        st._edv_configure()
    st.sc._logger = types.SimpleNamespace(base_log_dir=str(tmp_path / "logs" / "exp_001"))
    assert st._edv_configure() == str(tmp_path / "edv_shadow.jsonl")
