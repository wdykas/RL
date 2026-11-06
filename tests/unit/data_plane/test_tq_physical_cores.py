# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""BUGS #13: TQ operations must not depend on a per-call sysfs walk.

``transfer_queue.utils.common.limit_pytorch_auto_parallel_threads`` runs on
every client put / get and every storage-unit put / get / clear and compares
``TQ_NUM_THREADS`` with ``psutil.cpu_count(logical=False)``. That call returns
``None`` while the ``/sys/devices/system/cpu`` topology walk fails (on aarch64
it is the only source), and the comparison raises
``'>' not supported between instances of 'int' and 'NoneType'``. One of our runs
died of it at step 10.

Each test makes ``psutil.cpu_count(logical=False)`` return ``None`` after the
process has started. That is the state of a node whose walk fails mid-run.
The test then drives TQ's real code: the context manager on the client side,
and a storage unit's real ``_handle_put`` / ``_handle_clear``, in this process,
in a fresh process that never imports the adapter, and inside a real Ray
actor. Before the fix, each of these returns the error that run hit.
"""

from __future__ import annotations

import errno
import os
import platform
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

pytest.importorskip("transfer_queue", reason="transfer_queue not installed")

import psutil  # noqa: E402


_TQ_ERROR = "'>' not supported between instances of 'int' and 'NoneType'"

# Imports TQ's storage-unit class from whichever module this TQ revision defines it in.
_STORAGE_MODULE_SNIPPET = textwrap.dedent(
    """
    import importlib
    def storage_module():
        for path in ("transfer_queue.storage.simple_storage", "transfer_queue.storage.simple_backend"):
            try:
                return importlib.import_module(path)
            except ImportError:
                continue
        raise ImportError("no SimpleStorageUnit module in this transfer_queue")
    """
)
exec(_STORAGE_MODULE_SNIPPET)  # defines storage_module()


def _none_physical_cores(real):
    def cpu_count(logical: bool = True):
        return real(logical=True) if logical else None

    return cpu_count


def _bare_unit(storage_mod):
    """A storage unit without its ZMQ server: only the state the handlers use."""
    base = storage_mod.SimpleStorageUnit.__ray_metadata__.modified_class
    unit = base.__new__(base)
    unit.storage_unit_id = "TQ_STORAGE_UNIT_bug13"
    unit.storage_data = storage_mod.StorageUnitData(16)
    return unit


def _put_and_clear(storage_mod, unit) -> tuple[str, str, dict, dict]:
    put = unit._handle_put(
        SimpleNamespace(body={"global_indexes": [0], "data": {"x": [1]}})
    )
    clear = unit._handle_clear(SimpleNamespace(body={"global_indexes": [0]}))
    return str(put.request_type), str(clear.request_type), put.body, clear.body


@pytest.mark.skipif(
    platform.machine() != "aarch64", reason="arm64 has no /proc/cpuinfo fallback"
)
def test_failing_sysfs_walk_makes_psutil_return_none(monkeypatch) -> None:
    """The mechanism: glob swallows the walk's OSError and psutil returns None."""
    real_scandir = os.scandir

    def scandir(path="."):
        if str(path).startswith("/sys/devices/system/cpu"):
            raise OSError(errno.ENOMEM, "Cannot allocate memory", str(path))
        return real_scandir(path)

    assert psutil.cpu_count(logical=False)
    monkeypatch.setattr(os, "scandir", scandir)
    assert psutil.cpu_count(logical=False) is None


def test_client_side_tq_survives_none_core_count(monkeypatch) -> None:
    """Every client put / get enters this context manager (client.py async_put / async_get_data)."""
    from nemo_rl.data_plane.adapters import transfer_queue  # noqa: F401  pins at import
    from transfer_queue.utils import common

    cores = psutil.cpu_count(logical=False)
    assert cores
    monkeypatch.setattr(psutil, "cpu_count", _none_physical_cores(psutil.cpu_count))
    with common.limit_pytorch_auto_parallel_threads(
        target_num_threads=8, info="bug13"
    ):
        pass
    assert common.psutil.cpu_count(logical=False) == cores


def test_storage_unit_handlers_survive_none_core_count(monkeypatch) -> None:
    """The failing call: a storage unit's clear and put."""
    from nemo_rl.data_plane.adapters import transfer_queue  # noqa: F401

    mod = storage_module()
    unit = _bare_unit(mod)
    monkeypatch.setattr(psutil, "cpu_count", _none_physical_cores(psutil.cpu_count))
    put_t, clear_t, put_b, clear_b = _put_and_clear(mod, unit)
    assert _TQ_ERROR not in str(put_b) and _TQ_ERROR not in str(clear_b)
    assert put_t.endswith("PUT_DATA_RESPONSE"), put_b
    assert clear_t.endswith("CLEAR_DATA_RESPONSE"), clear_b


def _run(script: str) -> str:
    out = subprocess.run(
        [sys.executable, "-c", _STORAGE_MODULE_SNIPPET + textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout


def test_storage_unit_process_pins_without_the_adapter() -> None:
    """A storage-unit process never imports the adapter: the actor class must pin.

    Fresh interpreter, adapter not imported. The unit is built through the
    class the driver hands TQ's bootstrap, with TQ's constructor body (ZMQ
    server, threads) skipped.
    """
    stdout = _run(
        """
        import sys
        from types import SimpleNamespace
        import psutil
        from transfer_queue.utils import common
        from nemo_rl.utils import tq_physical_cores as t

        assert "nemo_rl.data_plane.adapters.transfer_queue" not in sys.modules
        assert common.psutil is psutil
        mod = storage_module()
        pinned = t.pinned_simple_storage_unit_class(mod.SimpleStorageUnit)
        cls = pinned.__ray_metadata__.modified_class

        def tq_init(self, storage_unit_size):
            self.storage_unit_id = "TQ_STORAGE_UNIT_bug13"
            self.storage_data = mod.StorageUnitData(storage_unit_size)

        mod.SimpleStorageUnit.__ray_metadata__.modified_class.__ray_actor_class__.__init__ = tq_init
        unit = cls.__new__(cls)
        cls.__init__(unit, storage_unit_size=16)
        real = psutil.cpu_count
        psutil.cpu_count = lambda logical=True: real(logical=True) if logical else None
        put = unit._handle_put(SimpleNamespace(body={"global_indexes": [0], "data": {"x": [1]}}))
        clear = unit._handle_clear(SimpleNamespace(body={"global_indexes": [0]}))
        print("PUT", put.request_type, put.body)
        print("CLEAR", clear.request_type, clear.body)
        """
    )
    assert _TQ_ERROR not in stdout, stdout
    assert "PUT_DATA_RESPONSE" in stdout and "CLEAR_DATA_RESPONSE" in stdout, stdout


def test_driver_swaps_the_class_tq_bootstrap_instantiates() -> None:
    """_init_tq's hook replaces the module global TQ's bootstrap calls ``.options`` on."""
    stdout = _run(
        """
        import importlib
        from nemo_rl.utils import tq_physical_cores as t

        t.pin_tq_storage_unit_core_count()
        t.pin_tq_storage_unit_core_count()  # idempotent: no double wrapping
        hits = 0
        for path in t._STORAGE_UNIT_FACTORY_MODULES:
            try:
                m = importlib.import_module(path)
            except ImportError:
                continue
            cls = getattr(m, "SimpleStorageUnit", None)
            if cls is None:
                continue
            assert getattr(cls, "_nrl_physical_cores_pinned", False), path
            assert cls._default_options.get("num_cpus") == 1, cls._default_options
            hits += 1
        print("HITS", hits)
        """
    )
    assert "HITS 0" not in stdout, stdout


def test_real_ray_storage_unit_actor_survives_none_core_count() -> None:
    """End to end in a Ray actor process: the pinned class TQ would start, then a put and a clear."""
    pytest.importorskip("ray")
    stdout = _run(
        """
        from types import SimpleNamespace
        import ray
        from nemo_rl.utils import tq_physical_cores as t

        mod = storage_module()
        ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
        try:
            pinned = t.pinned_simple_storage_unit_class(mod.SimpleStorageUnit)
            unit = pinned.options(num_cpus=1).remote(storage_unit_size=16)

            def probe(self):
                import psutil
                from transfer_queue.utils import common
                real = psutil.cpu_count
                psutil.cpu_count = lambda logical=True: real(logical=True) if logical else None
                put = self._handle_put(SimpleNamespace(body={"global_indexes": [0], "data": {"x": [1]}}))
                clear = self._handle_clear(SimpleNamespace(body={"global_indexes": [0]}))
                return type(common.psutil).__name__, str(put.request_type), put.body, str(clear.request_type), clear.body

            print("ACTOR", ray.get(unit.__ray_call__.remote(probe), timeout=300))
        finally:
            ray.shutdown()
        """
    )
    assert "_FixedPhysicalCores" in stdout, stdout
    assert _TQ_ERROR not in stdout, stdout
    assert "PUT_DATA_RESPONSE" in stdout and "CLEAR_DATA_RESPONSE" in stdout, stdout


def test_init_tq_hands_tq_the_pinned_storage_unit_class() -> None:
    """R34 (a): ``_init_tq`` (simple backend) itself must swap the class TQ instantiates.

    Fresh interpreter, so no earlier test's swap can satisfy it. ``tq.init`` is
    stubbed to record, at the moment it runs, the module-global
    ``SimpleStorageUnit`` of every factory module this TQ revision has: that is
    the class TQ's bootstrap calls ``.options`` on. Fails if the
    ``pin_tq_storage_unit_core_count()`` call in ``_init_tq`` is removed.
    """
    stdout = _run(
        """
        import importlib
        from nemo_rl.data_plane.adapters import transfer_queue as adapter
        from nemo_rl.utils import tq_physical_cores as t

        def factory_classes():
            out = {}
            for path in t._STORAGE_UNIT_FACTORY_MODULES:
                try:
                    m = importlib.import_module(path)
                except ModuleNotFoundError:
                    continue
                cls = getattr(m, "SimpleStorageUnit", None)
                if cls is not None:
                    out[path] = cls
            return out

        before = factory_classes()
        assert before, "no SimpleStorageUnit factory module in this transfer_queue"
        # Non-vacuous: importing the adapter pins clients only, not the unit class.
        assert not any(getattr(c, "_nrl_physical_cores_pinned", False) for c in before.values()), before
        seen = {}

        def init(*, conf):
            for path, cls in factory_classes().items():
                seen[path] = (
                    bool(getattr(cls, "_nrl_physical_cores_pinned", False)),
                    cls.__ray_metadata__.modified_class.__ray_actor_class__.__name__,
                    cls._default_options.get("num_cpus"),
                )

        adapter.tq.init = init
        adapter._init_tq(
            {
                "enabled": True,
                "impl": "transfer_queue",
                "claim_meta_poll_interval_s": 0.5,
                "backend": "simple",
                "simple": {"num_storage_units": 2},
            }
        )
        assert seen, "tq.init was not called"
        print("SEEN", seen)
        """
    )
    seen = stdout.split("SEEN", 1)[1]
    assert "(False" not in seen, stdout
    assert "(True, 'NrlSimpleStorageUnit', 1)" in seen, stdout


def _fake_factory(monkeypatch, failures: dict):
    """Point pin_tq_storage_unit_core_count at fake factory modules; ``failures``: path -> exception."""
    from nemo_rl.utils import tq_physical_cores as t

    def import_module(path):
        if path in failures:
            raise failures[path]
        return SimpleNamespace(SimpleStorageUnit=object())

    monkeypatch.setattr(t, "importlib", SimpleNamespace(import_module=import_module))
    monkeypatch.setattr(t, "pinned_simple_storage_unit_class", lambda cls: "PINNED")
    return t


def test_storage_unit_swap_skips_only_a_missing_factory_module(monkeypatch) -> None:
    """R34 (b): a factory module absent from this TQ revision (or its parent package) is skipped."""
    first, second = _fake_factory(monkeypatch, {})._STORAGE_UNIT_FACTORY_MODULES
    for missing in (first, first.rsplit(".", 1)[0]):
        t = _fake_factory(
            monkeypatch, {first: ModuleNotFoundError(f"No module named {missing!r}", name=missing)}
        )
        t.pin_tq_storage_unit_core_count()  # the other module swaps; no error


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("No module named 'some_dependency'", name="some_dependency"),
        ImportError("cannot import name 'X' from 'transfer_queue.utils'"),
    ],
    ids=["transitive_module_not_found", "import_error"],
)
def test_storage_unit_swap_surfaces_a_broken_factory_module(monkeypatch, error) -> None:
    """R34 (b): a factory module that exists but fails its own imports must raise, not be skipped.

    The other module's swap would satisfy the "nothing swapped" guard, so a
    silent skip would hide units started unpinned.
    """
    for path in _fake_factory(monkeypatch, {})._STORAGE_UNIT_FACTORY_MODULES:
        t = _fake_factory(monkeypatch, {path: error})
        with pytest.raises(type(error)):
            t.pin_tq_storage_unit_core_count()
