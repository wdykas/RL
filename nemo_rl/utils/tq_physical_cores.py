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
"""Read the node's physical core count once per process for ``transfer_queue``.

Every TQ put / get / clear, on the client and in each ``SimpleStorageUnit``,
enters ``transfer_queue.utils.common.limit_pytorch_auto_parallel_threads``,
which re-reads the core count from the kernel on every call::

    physical_cores = psutil.cpu_count(logical=False)
    ...
    if target_num_threads > physical_cores:      # TQ_NUM_THREADS (8) > None

``psutil.cpu_count(logical=False)`` returns ``None`` when it cannot determine
the count. On aarch64 (GB200) its only working source is a walk of
``/sys/devices/system/cpu/cpu[0-9]*/topology/core_cpus_list``: the
``/proc/cpuinfo`` fallback looks for x86's ``physical id`` / ``cpu cores``
lines, which arm64 does not print. ``glob`` swallows any ``OSError`` from that
walk and returns ``[]``. So whenever the walk fails on a node, every TQ
operation in every process on that node raises
``TypeError: '>' not supported between instances of 'int' and 'NoneType'``
until the walk works again (several of our runs
and the stock baseline; all units on one node fail at
once, for up to ~52 s).

The physical core count of a node does not change while a process runs, so it
is read once here, at process start. TQ's ``common`` module is then pointed at
a stand-in for ``psutil`` that returns that count. The stand-in is bound in
that one module only: ``psutil`` itself, and every other user of it, stay
untouched. A count that cannot be read at process start raises there, before
the process holds any data. It is not replaced by a default.

``transfer_queue`` has no configuration hook for this:
``TQ_NUM_THREADS`` only sets the left operand of the comparison. The upstream
pin in pyproject.toml has the same code, so the fix has to live in the
processes that run TQ.

* Client processes: :mod:`nemo_rl.data_plane.adapters.transfer_queue` calls
  :func:`pin_tq_physical_core_count` at import.
* Storage-unit actors never import that adapter (this module is kept free of
  NeMo-RL imports so they do not have to either). :func:`_init_tq` swaps the
  actor class that TQ's bootstrap instantiates for
  :func:`pinned_simple_storage_unit_class`, whose ``__init__`` pins the count
  in the actor's own process before TQ's constructor starts its worker thread.
"""

from __future__ import annotations

import importlib
from typing import Any

import psutil

# The modules that create SimpleStorageUnits, by TQ revision: the container's
# TQ builds them in ``interface``; the pyproject pin (05d4cc33) moved that to the
# SimpleStorage bootstrap.
_STORAGE_UNIT_FACTORY_MODULES = (
    "transfer_queue.storage.bootstrap.simple_storage_bootstrap",
    "transfer_queue.interface",
)


class _FixedPhysicalCores:
    """Stands in for ``psutil`` inside ``transfer_queue.utils.common`` only."""

    def __init__(self, physical_cores: int) -> None:
        self.physical_cores = physical_cores

    def cpu_count(self, logical: bool = True) -> int | None:
        if logical:
            return psutil.cpu_count(logical=True)
        return self.physical_cores

    def __getattr__(self, name: str) -> Any:
        return getattr(psutil, name)


def _drift_error(missing: str) -> RuntimeError:
    return RuntimeError(
        f"transfer_queue's {missing}, so the physical core count could not be "
        "fixed for TQ (BUGS #13: psutil.cpu_count(logical=False) returns None "
        "when the sysfs topology walk fails, and TQ compares it with an int on "
        "every put / get / clear). The TQ pin has moved: re-point "
        "nemo_rl/utils/tq_physical_cores.py at the new code."
    )


def pin_tq_physical_core_count() -> int:
    """Fix TQ's view of this node's physical core count for this process.

    Idempotent. Returns the count. Raises if ``transfer_queue`` no longer
    reads the count through ``utils.common.psutil``, or if the count cannot
    be read now.
    """
    from transfer_queue.utils import common

    pinned = getattr(common, "psutil", None)
    if isinstance(pinned, _FixedPhysicalCores):
        return pinned.physical_cores
    if pinned is not psutil or not hasattr(
        common, "limit_pytorch_auto_parallel_threads"
    ):
        raise _drift_error(
            "utils.common no longer reads the core count through a module-level "
            "psutil in limit_pytorch_auto_parallel_threads"
        )
    cores = psutil.cpu_count(logical=False)
    if cores is None:
        raise RuntimeError(
            "psutil.cpu_count(logical=False) returned None at process start: "
            "the /sys/devices/system/cpu topology walk failed on this node "
            "(BUGS #13). transfer_queue cannot run without the count."
        )
    common.psutil = _FixedPhysicalCores(int(cores))
    return int(cores)


def pinned_simple_storage_unit_class(actor_cls: Any) -> Any:
    """Return TQ's ``SimpleStorageUnit`` actor class, pinning the count in the actor.

    ``actor_cls`` is TQ's ``@ray.remote`` class. The returned actor class has
    the same Ray options and the same methods. Its ``__init__`` runs
    :func:`pin_tq_physical_core_count` in the actor's process before TQ's
    constructor.
    """
    import ray

    if getattr(actor_cls, "_nrl_physical_cores_pinned", False):
        return actor_cls
    try:
        base = actor_cls.__ray_metadata__.modified_class.__ray_actor_class__
        options = dict(actor_cls._default_options)
    except AttributeError as e:
        raise _drift_error(
            f"SimpleStorageUnit is no longer a Ray actor class ({e})"
        ) from e

    class NrlSimpleStorageUnit(base):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pin_tq_physical_core_count()
            super().__init__(*args, **kwargs)

    NrlSimpleStorageUnit.__qualname__ = NrlSimpleStorageUnit.__name__
    pinned = ray.remote(**options)(NrlSimpleStorageUnit)
    pinned._nrl_physical_cores_pinned = True
    return pinned


def _is_module_or_parent(name: str | None, path: str) -> bool:
    """True if ``name`` is ``path`` or one of its parent packages."""
    parts = path.split(".")
    return name in {".".join(parts[: i + 1]) for i in range(len(parts))}


def pin_tq_storage_unit_core_count() -> None:
    """Make TQ's SimpleStorage bootstrap start units that pin the core count.

    Called by the driver right before ``tq.init`` builds the units.
    """
    swapped = False
    for path in _STORAGE_UNIT_FACTORY_MODULES:
        try:
            mod = importlib.import_module(path)
        except ModuleNotFoundError as e:
            # Only "this TQ revision has no such module" is skipped. A module
            # that exists but fails one of its own imports must surface: a
            # silent skip could leave the units unpinned while the other
            # module's swap satisfies the guard below.
            if not _is_module_or_parent(e.name, path):
                raise
            continue
        actor_cls = getattr(mod, "SimpleStorageUnit", None)
        if actor_cls is None:
            continue
        mod.SimpleStorageUnit = pinned_simple_storage_unit_class(actor_cls)
        swapped = True
    if not swapped:
        raise _drift_error(
            "SimpleStorageUnit is no longer created from "
            + " or ".join(_STORAGE_UNIT_FACTORY_MODULES)
        )
