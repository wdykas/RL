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
"""engine_sampler / plain_engine_sampler = exact_int: which sampler a run actually uses, the
learner's policy definition that follows it, and the validators (agent exactint-final, 10-20).

The learner trains lambda = log(w_x) - log(W) iff the engines of the run sample the integer law:
speculative arms -> engine_sampler, arms without speculation (sync / async1) -> plain_engine_sampler.
Before this, engine_sampler=exact_int on an async1 arm set the learner's integer law while the plain
engine still sampled with the wide / FlashInfer sampler (mismatch by construction).
"""

from types import SimpleNamespace

import pytest

import thundersync_rl.grpo_loop as gl


def _master(**ts):
    policy = {
        "generation": {
            "temperature": 1.0, "top_p": 1.0, "top_k": None, "backend": "megatron",
            "stop_strings": None, "mcore_generation_config": {},
        },
        "megatron_cfg": {
            "batch_invariant_mode": True, "tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1,
        },
    }
    kw = dict(speculate=True, verify_mode="block", verify_source="engine", verify_on="learner", sticky_routing=True)
    kw.update(ts)
    return SimpleNamespace(
        thundersync=gl.ThunderSyncConfig(**kw), policy=policy,
        loss_fn=SimpleNamespace(reference_policy_kl_penalty=0.0), grpo=SimpleNamespace(seed=42),
    )


@pytest.mark.parametrize(
    "ts, want",
    [
        (dict(engine_sampler="exact_int"), "exact_int"),
        (dict(engine_sampler="wide"), "wide"),
        (dict(engine_sampler="exact_int", verify_mode="keyed"), "keyed"),
        (dict(speculate=False, engine_sampler="exact_int", plain_engine_sampler="wide"), "wide"),
        (dict(speculate=False, engine_sampler="wide", plain_engine_sampler="exact_int"), "exact_int"),
        (dict(speculate=False, engine_sampler="exact_int", plain_engine_sampler="engine"), "engine"),
    ],
)
def test_effective_sampler_and_learner_law_follow_the_engines(ts, want):
    m = _master(**ts)
    assert gl.effective_engine_sampler(m.thundersync) == want
    gl.configure_sticky_policy(m)
    assert m.policy["megatron_cfg"]["exact_int_law"] is (want == "exact_int")


def test_validators_follow_the_effective_sampler(monkeypatch):
    monkeypatch.delenv("THUNDERSYNC_DEBUG_KEYED_FRESH", raising=False)
    gl.validate_exact_int(_master(engine_sampler="exact_int"))
    gl.validate_exact_int(_master(speculate=False, plain_engine_sampler="exact_int"))
    # A plain arm with engine_sampler=exact_int but a wide plain sampler: not an exact_int run.
    m = _master(speculate=False, engine_sampler="exact_int", plain_engine_sampler="wide")
    m.policy["generation"]["top_p"] = 0.9
    gl.validate_exact_int(m)
    # top-p < 1 with the integer law: refused, the message names the way out.
    for m in (_master(engine_sampler="exact_int"), _master(speculate=False, plain_engine_sampler="exact_int")):
        m.policy["generation"]["top_p"] = 0.95
        with pytest.raises(ValueError, match="engine_sampler=wide"):
            gl.validate_exact_int(m)
    m = _master(engine_sampler="exact_int", exact_top_p=True)
    m.policy["generation"]["top_p"] = 0.95
    with pytest.raises(ValueError, match="exact_top_p is not wired yet"):
        gl.validate_exact_top_p(m)
    m = _master(engine_sampler="wide", exact_top_p=True)
    m.policy["generation"]["top_p"] = 0.95
    gl.validate_exact_top_p(m)
    # Debug keyed fresh sampling replaces the plain sampler: refused with the integer law.
    monkeypatch.setenv("THUNDERSYNC_DEBUG_KEYED_FRESH", "1")
    with pytest.raises(ValueError, match="keyed fresh"):
        gl.validate_exact_int(_master(speculate=False, plain_engine_sampler="exact_int"))


def test_plain_exact_int_install_salts(monkeypatch):
    got = []
    gen = SimpleNamespace(_policy=SimpleNamespace(worker_group=SimpleNamespace(
        run_all_workers_single_data=lambda m, salt, **kw: got.append((m, salt, kw)) or [{}])))
    monkeypatch.setattr(gl.ray, "get", lambda x: x)
    cfg = SimpleNamespace(grpo=SimpleNamespace(seed=42))
    gl.install_plain_exact_int_sampler(gen, cfg)
    gl.install_plain_exact_int_sampler(gen, cfg, segment=6)
    assert got[0][0] == "install_exact_int_sampler" and got[0][1] != got[1][1]
    gen_none = SimpleNamespace(_policy=SimpleNamespace(worker_group=SimpleNamespace(
        run_all_workers_single_data=lambda m, salt, **kw: [None])))
    with pytest.raises(RuntimeError, match="no engine rank"):
        gl.install_plain_exact_int_sampler(gen_none, cfg)
