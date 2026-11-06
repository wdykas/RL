"""On/off env flags: "0" / "false" / "" mean off (THUNDERSYNC_SPEC_PROF=0 used to turn the profiler, and its device
syncs, ON), and no on/off flag in the package is read by plain truthiness of os.environ.get any more."""

import ast
import pathlib

import pytest
from thundersync_rl import engine_prof, prof
from thundersync_rl.envflag import env_flag

PKG = pathlib.Path(__file__).resolve().parents[2] / "thundersync_rl"
# Variables that carry a value (a path prefix / directory), where presence is the switch.
# NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP_NOGRAD is a number (fraction of R_cap) read as `envs.get(..) or os.environ.get(..) or 0`
# -> float: unset / "" give 0, "0" gives 0.0, so the `or` chain is a fallback between sources, not an on/off test.
VALUE_VARS = {"THUNDERSYNC_TAIL_DUMP", "THUNDERSYNC_TORCH_PROF", "THUNDERSYNC_ACC_DUMP",  # value (path) variables, not on/off flags
              "NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP_NOGRAD"}


@pytest.mark.parametrize("value", ["0", "false", "False", "FALSE", "", " 0 ", "no", "off"])
def test_off_values(monkeypatch, value):
    monkeypatch.setenv("TS_TEST_FLAG", value)
    assert env_flag("TS_TEST_FLAG") is False
    assert env_flag("TS_TEST_FLAG", default=True) is False


@pytest.mark.parametrize("value", ["1", "true", "True", "yes", "on", "2"])
def test_on_values(monkeypatch, value):
    monkeypatch.setenv("TS_TEST_FLAG", value)
    assert env_flag("TS_TEST_FLAG") is True


def test_unset_gives_default(monkeypatch):
    monkeypatch.delenv("TS_TEST_FLAG", raising=False)
    assert env_flag("TS_TEST_FLAG") is False
    assert env_flag("TS_TEST_FLAG", default=True) is True


@pytest.mark.parametrize("mod,var", [(prof, "THUNDERSYNC_SPEC_PROF"), (engine_prof, "THUNDERSYNC_ENGINE_PROF")])
def test_profiler_gates(monkeypatch, mod, var):
    monkeypatch.setenv(var, "0")
    assert mod.enabled() is False
    monkeypatch.setenv(var, "false")
    assert mod.enabled() is False
    monkeypatch.setenv(var, "1")
    assert mod.enabled() is True


def _is_env_get(n):
    return (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("get", "getenv")
            and len(n.args) == 1 and not n.keywords and isinstance(n.args[0], ast.Constant)
            and isinstance(n.args[0].value, str) and n.args[0].value not in VALUE_VARS
            and ast.unparse(n.func.value) in ("os.environ", "os"))


def test_no_truthiness_env_flags_left():
    bad = []
    for f in sorted(PKG.glob("*.py")):
        for node in ast.walk(ast.parse(f.read_text())):
            tests = []
            if isinstance(node, (ast.If, ast.IfExp, ast.While, ast.Assert)):
                tests.append(node.test)
            elif isinstance(node, ast.BoolOp):
                tests.extend(node.values)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                tests.append(node.operand)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "bool":
                tests.extend(node.args)
            bad += [f"{f.name}:{t.lineno} {ast.unparse(t)}" for t in tests if _is_env_get(t)]
    assert bad == []
