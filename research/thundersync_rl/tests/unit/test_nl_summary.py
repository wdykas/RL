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
"""X1 verdicts (scripts/nl/nl_summary.py) on fixture logs: a negative passes only on a real check failure (R37 (i));
sampler S requirements key off the effective config (R37 (ii)); the prod arm is exempt from mixed rows only."""

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("nl_summary", os.path.join(HERE, "..", "..", "scripts", "nl", "nl_summary.py"))
nls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nls)


def _run(tmp_path, name, lines, s_cfg="off"):
    d = tmp_path / name
    d.mkdir()
    (d / "run.log").write_text("\n".join(lines) + "\n")
    if s_cfg is not None:
        (d / "effective_config.yaml").write_text(
            f"policy:\n  megatron_cfg:\n    env_vars:\n      THUNDERSYNC_WIDE_SAMPLER_S: '{s_cfg}'\n")
    r = nls.parse(str(d))
    return r, nls.judge(r, name)


def _lines(mixed=384, eng=True, s_calls=10, sampler_s=True, extra=()):
    out = [f"[nolanes verify] step=0 rows=8 continued=0 mixed_version_rows={mixed} mixed_continued=0 neg_tokens=5",
           "[nolanes mismatch] step=0 tok=10 exact=10 max|d|=0",
           "[engine q check] step=0 tok=10 exact=10 max|dlogq|=0",
           "[law check] step=0 n=10 ok",
           "[router frozen] step=0 ranks=2 tensors=16 trainable=0 unchanged ok"]
    if eng:
        out.append(f"[nolanes engine] step=0 [{{'neg_tokens': 0, 'sampler_s': {sampler_s}, 's_calls': {s_calls}}}]")
    return out + list(extra)


def test_negative_with_only_a_vacuity_failure_is_not_a_passing_control(tmp_path):
    r, (ok, verdict) = _run(tmp_path, "x1-neg_engine_q_genlp-1", _lines(mixed=0))
    assert r["fails"] == ["vacuous: no mixed-version row"] and not ok and verdict.startswith("NEG FAIL")
    # the same negative with a real check failure passes
    r, (ok, _) = _run(tmp_path, "x1-neg_engine_q_genlp-2",
                      _lines(mixed=0, extra=["[nolanes mismatch] step=1 tok=10 exact=3 max|d|=0.1"]))
    assert ok


def test_sampler_s_ran_on_every_run(tmp_path):
    """Sampler S is unconditional: every run needs the engine stats line with s_calls > 0."""
    _, (ok, _) = _run(tmp_path, "x1-pos-1", _lines())
    assert ok
    r, (ok, _) = _run(tmp_path, "x1-pos-2", _lines(eng=False))
    assert not ok and any("no engine stats line" in f for f in r["fails"])
    r, (ok, _) = _run(tmp_path, "x1-pos-4", _lines(s_calls=0))
    assert not ok and any("s_calls == 0" in f for f in r["fails"])


def test_only_prod_and_sanity_are_exempt_from_mixed_rows(tmp_path):
    _, (ok, _) = _run(tmp_path, "x1-prod-1", _lines(mixed=0))
    assert ok
    _, (ok, _) = _run(tmp_path, "x1-pos-6", _lines(mixed=0))
    assert not ok


def test_missing_or_unreadable_effective_config_fails(tmp_path):
    """R38 (b): no effective_config.yaml, or one that does not parse, is a failure (not a silent skip of the S check)."""
    r, (ok, _) = _run(tmp_path, "x1-pos-7", _lines(), s_cfg=None)
    assert not ok and any("effective config missing" in f for f in r["fails"])
    d = tmp_path / "x1-pos-8"
    d.mkdir()
    (d / "run.log").write_text("\n".join(_lines()) + "\n")
    (d / "effective_config.yaml").write_text("")
    r = nls.parse(str(d))
    assert any("effective config missing / unreadable" in f for f in r["fails"])
    # a negative whose only failure is the missing config is not a passing control
    r, (ok, _) = _run(tmp_path, "x1-neg_wrong_tag-9", _lines(), s_cfg=None)
    assert not ok


def test_router_frozen_line_required_on_every_step(tmp_path):
    """User rule 11-02: every verify step needs a router-frozen 'unchanged ok' line; a changed router is a real failure."""
    base = _lines()
    no_router = [l for l in base if "[router frozen]" not in l]
    r, (ok, _) = _run(tmp_path, "x1-pos-10", no_router)
    assert not ok and any("router frozen: missing" in f for f in r["fails"])
    changed = no_router + ["[router frozen] step=0 ranks=2 tensors=16 trainable=1 CHANGED FAIL"]
    r, (ok, _) = _run(tmp_path, "x1-pos-11", changed)
    assert not ok and any(f.startswith("router changed") for f in r["fails"])
    # a negative whose only failure is the missing router line is not a passing control
    r, (ok, _) = _run(tmp_path, "x1-neg_wrong_tag-12", no_router)
    assert not ok
