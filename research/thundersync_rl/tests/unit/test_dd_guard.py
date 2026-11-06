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
"""Red-team R2: forwards that bypass ``learner_graphs.call`` keep the MoE device dispatch's capacity guard.

Device dispatch is patched class-wide; an overflow drops rows and must rerun the forward on Megatron's
dispatcher (exact). Forced-overflow tests per path: ``guarded_forward`` itself, the scorer path
(``_score_full``, used by verify_drafts_block_stashed / _fused / scorer_token_logprobs / _dummy_score),
and verify_drafts' direct ``self.model`` forwards (real rows and EP lockstep fillers), plus a source scan
that no other direct model forward is left unguarded in worker.py.
"""

import ast
import inspect
import textwrap

import pytest
import torch

from nemo_rl.models.megatron import device_dispatch, learner_graphs


class _DD:
    """Fake device dispatch: enabled; overflows on the first ``n`` checks."""

    def __init__(self, monkeypatch, n=1):
        self.n, self.active, self.log = n, True, []
        monkeypatch.setattr(device_dispatch, "enabled", lambda: True)
        monkeypatch.setattr(device_dispatch, "set_active", self.set_active)
        monkeypatch.setattr(device_dispatch, "overflowed", self.overflowed)

    def set_active(self, flag):
        self.active = bool(flag)

    def overflowed(self, force=False):
        hit = self.n > 0
        self.n -= 1
        return hit


def _fn(dd):
    calls = []

    def fn():
        calls.append(dd.active)
        return torch.full((1,), float(len(calls)))

    return fn, calls


def test_no_overflow_runs_once(monkeypatch):
    dd = _DD(monkeypatch, n=0)
    fn, calls = _fn(dd)
    assert learner_graphs.guarded_forward(fn).item() == 1.0
    assert calls == [True] and dd.active


def test_overflow_reruns_on_megatron_dispatcher(monkeypatch):
    dd = _DD(monkeypatch, n=1)
    fn, calls = _fn(dd)
    before = learner_graphs.stats().get("capacity_rerun", 0)
    with torch.no_grad():
        out = learner_graphs.guarded_forward(fn)
    assert out.item() == 2.0  # the rerun's output, never the overflowed one
    assert calls == [True, False]  # second call on Megatron's dispatcher
    assert dd.active  # no grad: switched back at once
    assert learner_graphs.stats().get("capacity_rerun", 0) == before + 1


def test_overflow_with_grad_restores_counters_and_stays_off(monkeypatch):
    dd = _DD(monkeypatch, n=1)
    monkeypatch.setattr(learner_graphs, "_ep_size", lambda: 4)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.local_tokens_per_expert = torch.zeros(3)

    m = M()

    def fn():
        m.local_tokens_per_expert += 1  # routing counters accumulate per forward
        return torch.ones(1, requires_grad=True) * dd.active

    with torch.enable_grad():
        learner_graphs.guarded_forward(fn, m)
    assert torch.equal(m.local_tokens_per_expert, torch.ones(3))  # first pass rolled back
    assert not dd.active  # lasts through the backward; the next call() re-activates


def test_disabled_is_plain_call(monkeypatch):
    monkeypatch.setattr(device_dispatch, "enabled", lambda: False)
    monkeypatch.setattr(device_dispatch, "overflowed", lambda force=False: pytest.fail("checked while disabled"))
    assert learner_graphs.guarded_forward(lambda: 7) == 7


def _worker_cls():
    from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker

    return ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class


def test_scorer_path_reruns_on_overflow(monkeypatch):
    """_score_full (every scorer forward: stashed / fused verification, scorer_token_logprobs, dummies)."""
    dd = _DD(monkeypatch, n=1)
    seen = []

    class Fake:
        def _score_full_unguarded(self, scorer, ids, pos):
            seen.append(dd.active)
            return torch.full((1,), float(len(seen)))

    with torch.no_grad():
        out = _worker_cls()._score_full(Fake(), torch.nn.Linear(1, 1), None, None)
    assert seen == [True, False] and out.item() == 2.0


def _calls_in(func_src):
    tree = ast.parse(textwrap.dedent(func_src))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


def _is_direct_forward(call):
    f = call.func
    if isinstance(f, ast.Attribute) and f.attr == "model" and isinstance(f.value, ast.Name) and f.value.id == "self":
        return True  # self.model(...)
    return isinstance(f, ast.Name) and f.id in ("scorer", "model")  # scorer(...) / model(...)


def _inside_guard(func_src):
    """Direct forwards in ``func_src`` that are not inside a lambda passed to guarded_forward."""
    tree = ast.parse(textwrap.dedent(func_src))
    guarded = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "guarded_forward":
            for a in n.args:
                for m in ast.walk(a):
                    guarded.add(id(m))
    return [c for c in ast.walk(tree) if isinstance(c, ast.Call) and _is_direct_forward(c) and id(c) not in guarded]


def test_verify_drafts_forwards_are_guarded():
    src = inspect.getsource(_worker_cls().verify_drafts)
    direct = [c for c in _calls_in(src) if _is_direct_forward(c)]
    assert len(direct) == 2  # the lockstep filler and the real rows
    assert _inside_guard(src) == []


def test_no_unguarded_direct_forward_in_worker():
    """Every direct model / scorer forward in worker.py is guarded, or inside _score_full_unguarded
    (reached only through _score_full). _training_forward_rows & co. go through train.model_forward ->
    learner_graphs.call (guarded there)."""
    import thundersync_rl.worker as w

    tree = ast.parse(inspect.getsource(w))
    guarded = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "guarded_forward":
            for a in n.args:
                guarded.update(id(m) for m in ast.walk(a))
    enclosing = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for m in ast.walk(fn):
                enclosing.setdefault(id(m), set()).add(fn.name)
    bad = [
        (c.lineno, sorted(enclosing.get(id(c), ())))
        for c in ast.walk(tree)
        if isinstance(c, ast.Call) and _is_direct_forward(c) and id(c) not in guarded
        and "_score_full_unguarded" not in enclosing.get(id(c), ())
    ]
    assert bad == [], bad


def test_guard_scan_catches_a_broken_variant():
    """Negative control for the scan: an unguarded self.model(...) is reported."""
    broken = "def verify_drafts(self):\n    logits = self.model(input_ids=1, position_ids=2, attention_mask=None)\n"
    assert len(_inside_guard(broken)) == 1
    fixed = ("def verify_drafts(self):\n    logits = learner_graphs.guarded_forward("
             "lambda: self.model(input_ids=1, position_ids=2, attention_mask=None), self.model)\n")
    assert _inside_guard(fixed) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="verify_drafts allocates on cuda")
def test_verify_drafts_forced_overflow_returns_rerun_results(monkeypatch):
    """Drive verify_drafts with an EP lockstep filler and a real-row forward, both overflowing once.

    The overflowed pass's logits pick the wrong token everywhere (as dropped rows would); the rerun's logits pick
    the draft. The result must be the rerun's: the full draft accepted, the rerun's next token.
    """
    from megatron.core import parallel_state as ps

    monkeypatch.setattr(ps, "get_data_parallel_world_size", lambda *a, **k: 1)
    monkeypatch.setattr(ps, "get_data_parallel_rank", lambda *a, **k: 0)
    dd = _DD(monkeypatch, n=2)
    V, plen, draft, nxt, wrong = 16, 3, [5, 6, 7], 9, 2
    row = torch.tensor([1, 2, 3] + draft)
    seq = draft + [nxt]  # token emitted at positions plen .. plen+len(draft)
    calls = []

    class Model(torch.nn.Module):
        def forward(self, input_ids, position_ids, attention_mask):
            calls.append(dd.active)
            b, s = input_ids.shape
            out = torch.zeros((b, s, V), device="cuda")
            if dd.active:  # overflowed pass: garbage
                out[:, :, wrong] = 1e4
            else:
                for j, t in enumerate(seq):
                    if plen - 1 + j < s:  # the 4-token EP filler is shorter than the real row
                        out[:, plen - 1 + j, t] = 1e4
            return out

    class Fake:
        cfg = {"make_sequence_length_divisible_by": 4}
        model = Model()

        def _expert_parallel_lockstep(self, plans):
            return [[] for _ in range(len(plans) + 1)]  # one EP filler forward

    before = learner_graphs.stats().get("capacity_rerun", 0)
    with torch.no_grad():
        res = _worker_cls().verify_drafts(Fake(), [row], [plen], [123], V, 1, 64)
    assert calls == [True, False, True, False]  # filler + rerun, real rows + rerun
    assert learner_graphs.stats().get("capacity_rerun", 0) == before + 2
    assert dd.active
    ((i, r),) = res
    assert i == 0 and r["accepted"] == len(draft) and r["next"] == nxt

    # Negative control: without the guard the overflowed logits reach the result (the test is not vacuous).
    calls.clear()
    dd.n = 2
    monkeypatch.setattr(learner_graphs, "guarded_forward", lambda fn, model=None: fn())
    with torch.no_grad():
        ((_, r_bad),) = _worker_cls().verify_drafts(Fake(), [row], [plen], [123], V, 1, 64)
    assert calls == [True, True] and r_bad["accepted"] == 0 and r_bad["next"] == wrong


def test_score_full_unguarded_is_only_called_from_score_full():
    """The scorer guard holds only if nothing bypasses _score_full; scan every thundersync_rl module."""
    import pathlib

    import thundersync_rl

    bad = []
    for p in pathlib.Path(thundersync_rl.__file__).parent.rglob("*.py"):
        tree = ast.parse(p.read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or fn.name == "_score_full":
                continue
            for c in ast.walk(fn):
                if isinstance(c, ast.Attribute) and c.attr == "_score_full_unguarded" and not (
                    fn.name == "_score_full_unguarded"
                ):
                    bad.append((p.name, fn.name, c.lineno))
    assert bad == [], bad
