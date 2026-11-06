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
"""module_provenance: one source tree per package, a mix (image copy + shared-filesystem copy) raises."""

import os
from types import SimpleNamespace

import pytest

from nemo_rl.utils.module_provenance import check_module_provenance, module_roots

A = "/shared/x/RL/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
B = "/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"


def _mods(**files):
    return {name.replace("__", "."): SimpleNamespace(__file__=path) for name, path in files.items()}


def test_single_tree_per_package_passes_and_reports_roots(capsys):
    mods = _mods(
        megatron__core=f"{A}/megatron/core/__init__.py",
        megatron__core__inference__batch_dimensions_utils=f"{A}/megatron/core/inference/batch_dimensions_utils.py",
        nemo_rl=os.path.join("/t/RL-clean-scale-x", "nemo_rl", "__init__.py"),
        transformer_engine__pytorch="/w/overlay_bi/transformer_engine/pytorch/__init__.py",
    )
    roots = check_module_provenance("driver", mods)
    assert roots["megatron.core"] == {A: 2}
    assert roots["nemo_rl"] == {"/t/RL-clean-scale-x": 1}
    assert roots["transformer_engine"] == {"/w/overlay_bi": 1}
    assert "[provenance] driver" in capsys.readouterr().out


def test_two_trees_of_one_package_raise(monkeypatch):
    monkeypatch.delenv("NEMO_RL_PROVENANCE_CHECK", raising=False)
    mods = _mods(
        megatron__core__inference__batch_dimensions_utils=f"{A}/megatron/core/inference/batch_dimensions_utils.py",
        megatron__core__process_groups_config=f"{B}/megatron/core/process_groups_config.py",
    )
    with pytest.raises(RuntimeError, match="more than one source tree"):
        check_module_provenance("worker x", mods)


def test_warn_mode_prints_and_continues(monkeypatch, capsys):
    monkeypatch.setenv("NEMO_RL_PROVENANCE_CHECK", "warn")
    mods = _mods(nemo_rl__a="/t1/nemo_rl/a.py", nemo_rl__b="/t2/nemo_rl/b.py")
    check_module_provenance("worker x", mods)
    assert "continuing" in capsys.readouterr().out


def test_off_and_unwatched_and_unrooted_modules(monkeypatch):
    monkeypatch.setenv("NEMO_RL_PROVENANCE_CHECK", "0")
    assert check_module_provenance("driver", _mods(nemo_rl__a="/t1/nemo_rl/a.py", nemo_rl__b="/t2/nemo_rl/b.py")) == {}
    mods = {"torch": SimpleNamespace(__file__="/x/torch/__init__.py"), "nemo_rl.ext": SimpleNamespace(__file__="/x/lib.so"),
            "nemo_rl.ns": SimpleNamespace(__file__=None)}
    assert module_roots(mods) == {}


def test_live_interpreter_has_one_nemo_rl_tree():
    roots = module_roots()
    assert len(roots.get("nemo_rl", {})) == 1


def _pinned_tree(tmp_path, name, rel, text):
    root = tmp_path / name
    (root / rel).mkdir(parents=True)
    (root / rel / "__init__.py").write_text(text)
    return root


def test_snapshot_pins_match_and_mismatch(tmp_path, monkeypatch, capsys):
    import nemo_rl.utils.module_provenance as mp

    monkeypatch.delenv("NEMO_RL_PROVENANCE_CHECK", raising=False)
    monkeypatch.setattr(mp, "_have_git", lambda: False)  # content hashes are the enforced pin
    mlm = _pinned_tree(tmp_path, "mlm", "megatron/core", "x = 1\n")
    te = _pinned_tree(tmp_path, "te", "transformer_engine", "y = 2\n")
    pins = tmp_path / ".snapshot_pins"
    pins.write_text(
        f"MEGATRON_LM_ROOT={mlm}\nMEGATRON_LM_SHA=abc\nMEGATRON_LM_HASH={mp.tree_hash(str(mlm), 'megatron.core')}\n"
        f"TE_ROOT={te}\nTE_HASH={mp.tree_hash(str(te), 'transformer_engine')}\n"
    )
    monkeypatch.setenv("NEMO_RL_SNAPSHOT_PINS", str(pins))
    mods = _mods(megatron__core=f"{mlm}/megatron/core/__init__.py", transformer_engine=f"{te}/transformer_engine/__init__.py")
    check_module_provenance("worker x", mods)
    assert "matches the snapshot pins" in capsys.readouterr().out
    # an edited pinned tree
    (mlm / "megatron/core/__init__.py").write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="content hash"):
        check_module_provenance("worker x", mods)
    # a package loaded from another tree than the pinned one
    other = _mods(megatron__core=f"{B}/megatron/core/__init__.py", transformer_engine=f"{te}/transformer_engine/__init__.py")
    with pytest.raises(RuntimeError, match="pinned"):
        check_module_provenance("worker x", other)
