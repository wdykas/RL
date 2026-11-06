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
"""Long-run leak / health probe (``THUNDERSYNC_LEAK_PROBE``, default on).

Every step the loop records ``leak/*`` metrics:

* per worker (``longrun_snapshot``; learner ranks and every engine rank of every lane): host RSS, CUDA allocated /
  reserved, open file descriptors, threads, and the sizes of the per-draft bookkeeping that used to grow with the run
  (rank 0's draft tables, the hidden-state / route logs' open and closed keys, verification captures, head stashes,
  engine anchor handles, prefix cache) -- reduced to max (and rank 0) over ranks;
* driver: RSS, the speculative wrappers' containers (plans, ⊥-BV slots, route blocks, stream owners, cohorts);
* Ray object store: plasma bytes / objects in use and spilled bytes (``memory_summary(stats_only=True)``);
* log sizes: the run directory and the Ray session's log directory.

``leak_verdict(history)`` fits a robust slope (Theil-Sen) to every ``leak/*`` series over the steps after a warm-up and
flags a series whose projected growth over ``HORIZON`` steps exceeds both a relative and an absolute tolerance
(memory flatness). ``python -m thundersync_rl.leak_probe RUN_DIR`` prints per-key slopes and ``[leak] PASS|FAIL``.

Per-rank cost: one small Ray call per worker per step (dict lengths, allocator counters, /proc reads); one
``memory_summary`` call and a directory walk on the driver. Nothing per token; at Ultra scale the per-step fan-out is
one call per learner / engine rank (the same fan-out as the ordered-call digests already read every step).
"""

from __future__ import annotations

import ctypes
import math
import os
import re
import threading
from typing import Any, Iterable, Optional

HORIZON = 500  # steps a long run must stay flat over (the campaign's longest planned run)
WARMUP_FRAC = 0.2  # leading fraction of the steps ignored (allocator / cache warm-up, first epoch)
MIN_STEPS = 8  # fewer post-warm-up points: no verdict


def enabled() -> bool:
    return os.environ.get("THUNDERSYNC_LEAK_PROBE", "1") != "0"


def process_stats() -> dict[str, float]:
    """This process: host RSS (GB), CUDA allocated / reserved (GB, if CUDA is initialized), fds, threads."""
    out: dict[str, float] = {}
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        out["rss_gb"] = pages * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (OSError, ValueError, IndexError):
        pass
    # Split of the RSS (final50, 10-20): engines / learners map Ray's plasma store (hidden blocks put / fetched
    # zero-copy), and touched shared pages count in RSS up to the store's size -- bounded, not a leak. RssAnon is the
    # private heap (Python objects, pinned host buffers); the verdict judges it, RssShmem / RssFile are reported.
    try:
        with open("/proc/self/status") as f:
            for line in f:
                for tag, key in (("RssAnon:", "rss_anon_gb"), ("RssShmem:", "rss_shmem_gb"), ("RssFile:", "rss_file_gb")):
                    if line.startswith(tag):
                        out[key] = int(line.split()[1]) / 2**20  # kB -> GB
    except (OSError, ValueError, IndexError):
        pass
    out.update(heap_stats())
    if "rss_anon_gb" in out and "heap_free_gb" in out:
        # Live private memory: RssAnon net of the free bytes glibc's arenas keep resident (retention of freed
        # transient buffers, reused by later allocations). Everything else stays in: glibc in-use chunks, pymalloc
        # arenas (Python objects), other anonymous mappings. The verdict judges this, not RssAnon (leak
        # investigation 10-21, research/leak/README.md): a per-step append grows it, allocator retention does not.
        out["rss_live_gb"] = max(0.0, out["rss_anon_gb"] - out["heap_free_gb"])
    try:
        out["fds"] = float(len(os.listdir("/proc/self/fd")))
    except OSError:
        pass
    out["threads"] = float(threading.active_count())
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            out["cuda_alloc_gb"] = torch.cuda.memory_allocated() / 2**30
            out["cuda_reserved_gb"] = torch.cuda.memory_reserved() / 2**30
            # Device memory in use by every process on this GPU (one worker per GPU here) and the part outside
            # PyTorch's allocator (CUDA context, NCCL / symmetric-memory buffers, cuBLAS workspaces): final50
            # OOMed with 13.5 GiB outside PyTorch, which the allocator counters cannot show.
            free, total = torch.cuda.mem_get_info()
            out["cuda_device_used_gb"] = (total - free) / 2**30
            out["cuda_nontorch_gb"] = max(0.0, (total - free) / 2**30 - out["cuda_reserved_gb"])
    except Exception:  # noqa: BLE001
        pass
    return out


class _MallInfo2(ctypes.Structure):
    _fields_ = [(k, ctypes.c_size_t) for k in
                ("arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost")]


def _libc():
    try:
        lib = ctypes.CDLL("libc.so.6")
        lib.mallinfo2.restype = _MallInfo2
        return lib
    except (OSError, AttributeError):  # not glibc / glibc < 2.33
        return None


def heap_stats() -> dict[str, float]:
    """glibc heap of this process (GB): ``heap_inuse_gb`` = live malloc'd bytes (arena chunks + mmapped chunks),
    ``heap_free_gb`` = free bytes the arenas hold (counted in RssAnon until trimmed). Splits RssAnon growth into live
    allocations vs allocator retention (leak investigation 10-21). {} without glibc's mallinfo2."""
    lib = _libc()
    if lib is None:
        return {}
    m = lib.mallinfo2()
    return {"heap_inuse_gb": (m.uordblks + m.hblkhd) / 2**30, "heap_free_gb": m.fordblks / 2**30}


_PLASMA = re.compile(r"Plasma memory usage (\d+) MiB, (\d+) objects")
_SPILLED = re.compile(r"Spilled (\d+) MiB, (\d+) objects")


def object_store_stats() -> dict[str, float]:
    """Plasma usage over the cluster (GB, objects) and spilled bytes, from Ray's memory summary."""
    try:
        from ray._private.internal_api import memory_summary

        text = memory_summary(stats_only=True)
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    m = _PLASMA.search(text)
    if m:
        out["object_store_gb"] = int(m.group(1)) / 1024
        out["object_store_objects"] = float(m.group(2))
    m = _SPILLED.search(text)
    out["object_store_spilled_gb"] = (int(m.group(1)) / 1024) if m else 0.0
    return out


def dir_bytes(path: Optional[str]) -> Optional[float]:
    if not path or not os.path.isdir(path):
        return None
    total = 0
    stack = [path]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        elif e.is_file(follow_symlinks=False):
                            total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return float(total)


def ray_log_dir() -> Optional[str]:
    try:
        import ray

        node = ray._private.worker._global_node  # noqa: SLF001
        return os.path.join(node.get_session_dir_path(), "logs") if node is not None else None
    except Exception:  # noqa: BLE001
        return None


def reduce_snapshots(prefix: str, snaps: list[dict[str, float]]) -> dict[str, float]:
    """``leak/{prefix}_{key}_max`` over ranks (and ``_r0`` for rank 0's value, ``_sum`` for memory keys)."""
    out: dict[str, float] = {}
    keys = sorted({k for s in snaps if s for k in s})
    for k in keys:
        xs = [float(s[k]) for s in snaps if s and isinstance(s.get(k), (int, float))]
        if not xs:
            continue
        out[f"leak/{prefix}_{k}_max"] = max(xs)
        if k.startswith("rank0_"):
            continue
        if k.endswith("_gb"):
            out[f"leak/{prefix}_{k}_sum"] = sum(xs)
    return out


def container_sizes(spec: Any) -> dict[str, float]:
    """Driver-side containers of one SpeculativeGeneration (sizes only)."""
    d = getattr(spec, "__dict__", {})
    out = {}
    for name in ("plans", "_null_slots", "_route_blocks", "cohorts", "_stream_owner", "_anchor_claims", "_row_plans"):
        v = d.get(name)
        if v is None:
            continue
        try:
            n = len(v)
        except TypeError:
            continue
        out[name.strip("_")] = float(n)
        if name == "_route_blocks":
            out["route_block_refs"] = float(sum(len(x) for x in v.values()))
        if name == "_stream_owner":
            out["stream_owner_indices"] = float(sum(len(ix) for gens in v.values() for ix in gens.values()))
    vp = d.get("vit_predictor")
    if vp is not None and hasattr(vp, "steps"):
        out["vit_predictor_steps"] = float(len(vp.steps))
    return out


def collect(
    worker_groups: dict[str, Any],
    specs: Iterable[Any],
    run_dirs: Iterable[Optional[str]] = (),
    history_len: int = 0,
    get=None,
) -> dict[str, float]:
    """One step's ``leak/*`` metrics. ``worker_groups``: {"learner": group, "engine": group, "engine1": ...}.

    Each group is asked for ``longrun_snapshot`` on every worker; a group that fails (a stub, a dead worker) is
    skipped and counted in ``leak/probe_errors``.
    """
    if get is None:
        import ray

        get = ray.get
    out: dict[str, float] = {}
    errors = 0
    for name, group in worker_groups.items():
        if group is None:
            continue
        try:
            snaps = get(group.run_all_workers_single_data("longrun_snapshot"))
        except Exception:  # noqa: BLE001
            errors += 1
            continue
        out.update(reduce_snapshots(name, list(snaps)))
    drv = process_stats()
    out.update({f"leak/driver_{k}": v for k, v in drv.items()})
    for i, sp in enumerate(specs):
        if sp is None:
            continue
        try:
            out.update({f"leak/spec{i}_{k}": v for k, v in container_sizes(sp).items()})
        except Exception:  # noqa: BLE001  (a probe never stops the run)
            errors += 1
    out.update({f"leak/{k}": v for k, v in object_store_stats().items()})
    total = 0.0
    seen = False
    for d in run_dirs:
        b = dir_bytes(d)
        if b is not None:
            total += b
            seen = True
    if seen:
        out["leak/run_dir_gb"] = total / 2**30
    b = dir_bytes(ray_log_dir())
    if b is not None:
        out["leak/ray_log_gb"] = b / 2**30
    out["leak/history_len"] = float(history_len)
    out["leak/probe_errors"] = float(errors)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Verdict


def theil_sen(xs: list[float], ys: list[float]) -> float:
    """Median of pairwise slopes (robust to the spikes of single steps)."""
    n = len(xs)
    sl = []
    stride = max(1, n // 200)  # O(n^2) pairs: subsample for long runs
    for i in range(0, n, stride):
        for j in range(i + 1, n, stride):
            if xs[j] != xs[i]:
                sl.append((ys[j] - ys[i]) / (xs[j] - xs[i]))
    if not sl:
        return 0.0
    sl.sort()
    m = len(sl) // 2
    return sl[m] if len(sl) % 2 else 0.5 * (sl[m - 1] + sl[m])


# Keys that legitimately grow with the step count (the history list itself, cumulative logs): reported, not judged
# against the flatness tolerance but against their own (linear growth of a log is expected; superlinear is not).
CUMULATIVE = ("leak/history_len", "leak/run_dir_gb", "leak/ray_log_gb", "leak/probe_errors")
# Absolute tolerance of the projected growth over HORIZON steps, by key suffix.
ABS_TOL = (("_gb_sum", 2.0), ("_gb_max", 1.0), ("_gb", 1.0), ("", 2000.0))
REL_TOL = 0.25


def _abs_tol(key: str) -> float:
    for suf, tol in ABS_TOL:
        if key.endswith(suf):
            return tol
    return ABS_TOL[-1][1]


def series(history: list[dict[str, Any]]) -> dict[str, tuple[list[float], list[float]]]:
    out: dict[str, tuple[list[float], list[float]]] = {}
    for i, h in enumerate(history):
        step = h.get("step", i)
        for k, v in h.items():
            if not k.startswith("leak/") or isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if isinstance(v, float) and not math.isfinite(v):
                continue
            xs, ys = out.setdefault(k, ([], []))
            xs.append(float(step))
            ys.append(float(v))
    return out


def late_growth(xs: list[float], ys: list[float]) -> float:
    """Growth per step over the final half of (xs, ys): the larger of its Theil-Sen slope and the rate between the
    medians of its first and last thirds (a level shift inside the final half counts even when most pairwise slopes
    are 0). +inf with fewer than MIN_STEPS points in the final half (no plateau claim on short windows)."""
    h = len(xs) // 2
    fx, fy = xs[h:], ys[h:]
    if len(fx) < MIN_STEPS:
        return float("inf")
    t = len(fx) // 3
    a, b = sorted(fy[:t]), sorted(fy[-t:])
    ca, cb = sorted(fx[:t])[t // 2], sorted(fx[-t:])[t // 2]
    level = (b[t // 2] - a[t // 2]) / (cb - ca) if cb > ca else 0.0
    return max(theil_sen(fx, fy), level)


def envelope_growth(xs: list[float], ys: list[float]) -> float:
    """Growth per step of the upper envelope: (max over the final half - max over the first half) / half the span.
    A leak keeps setting new highs; a bounded gauge sampled at random phases (0 or up to its cap: alternating lanes'
    route / draft stores) and a plateau do not, however their medians or pairwise slopes fall."""
    h = len(xs) // 2
    if h < MIN_STEPS // 2 or xs[-1] <= xs[0]:
        return float("inf")
    return (max(ys[h:]) - max(ys[:h])) / (0.5 * (xs[-1] - xs[0]))


def leak_verdict(
    history: list[dict[str, Any]],
    horizon: int = HORIZON,
    warmup_frac: float = WARMUP_FRAC,
    ignore: Iterable[str] = (),
) -> dict[str, Any]:
    """{"ok": bool, "steps": n, "rows": [(key, first, last, slope/step, projected, tol, ok)], "failed": [...]}.

    A non-cumulative series fails when its Theil-Sen slope x ``horizon`` exceeds max(REL_TOL x its median level,
    the absolute tolerance of its kind) AND its growth over the final half of the window (``late_growth``) x
    ``horizon`` exceeds it too (a plateau reached before the final half passes; growth that persists fails) AND its
    upper envelope (``envelope_growth``) keeps rising at that rate. Cumulative series (logs, history) fail only when their slope over the last
    half exceeds 2x the slope over the first half (superlinear growth) and the projection exceeds 10 GB.
    """
    rows, failed = [], []
    ign = tuple(ignore)
    n_steps = len(history)
    ser = series(history)
    for k, (xs, ys) in sorted(ser.items()):
        if any(k.startswith(p) for p in ign):
            continue
        info = False
        if "_rss_gb" in k and k.replace("_rss_gb", "_rss_anon_gb") in ser:
            info = True  # total RSS includes mapped plasma pages; judged through rss_anon (private memory)
        if "_rss_shmem_gb" in k or "_rss_file_gb" in k or "cuda_device_used_gb" in k:
            info = True  # shared / file mappings (plasma, libraries) and device totals: reported, not judged
        if "_rss_anon_gb" in k and k.replace("_rss_anon_gb", "_rss_live_gb") in ser:
            info = True  # RssAnon includes glibc's retained free pages; judged through rss_live (live memory)
        if "heap_free_gb" in k:
            info = True  # allocator retention (freed pages kept for reuse): reported, not judged
        lo = int(math.ceil(warmup_frac * len(xs)))
        xs_w, ys_w = xs[lo:], ys[lo:]
        if len(xs_w) < MIN_STEPS:
            continue
        slope = theil_sen(xs_w, ys_w)
        proj = slope * horizon
        med = sorted(ys_w)[len(ys_w) // 2]
        if k in CUMULATIVE:
            h = len(xs_w) // 2
            s1 = theil_sen(xs_w[:h], ys_w[:h]) if h >= 3 else slope
            s2 = theil_sen(xs_w[h:], ys_w[h:]) if len(xs_w) - h >= 3 else slope
            tol = 10.0 if k.endswith("_gb") else float("inf")
            ok = not (s2 > 2.0 * max(s1, 0.0) + 1e-12 and proj > tol)
        else:
            tol = max(REL_TOL * abs(med), _abs_tol(k))
            # Growth must persist into the final half of the window: a series that rose and then stayed flat
            # (allocator / cache warm-up that saturated, a one-off larger buffer) is a plateau, not a leak; the
            # whole-window slope over-projects such step-like data (final50b learner RssAnon max: +1.1 GB by step
            # 18, flat for the 31 steps after it, projected +15 GB). A linear leak grows in every window and fails.
            # A bounded gauge whose samples alternate between ~0 and its cap (leak-l8-big-live 10-21: engine route
            # rows stored, 0.15M / 2.1M of a fixed 4.2M cap) has a noisy slope; its upper envelope is flat.
            ok = (proj <= tol or late_growth(xs_w, ys_w) * horizon <= tol
                  or envelope_growth(xs_w, ys_w) * horizon <= tol)
        if info:
            ok = True
        rows.append((k, ys_w[0], ys_w[-1], slope, proj, tol, ok))
        if not ok:
            failed.append(k)
    return {"ok": not failed and bool(rows), "steps": n_steps, "rows": rows, "failed": failed}


def format_verdict(v: dict[str, Any], tag: str = "", top: int = 12) -> list[str]:
    lines = []
    rows = sorted(v["rows"], key=lambda r: (r[6], -abs(r[4]) / (r[5] or 1.0)))
    for k, first, last, slope, proj, tol, ok in rows[:top]:
        lines.append(
            f"[leak] {'ok  ' if ok else 'GROW'} {k} first={first:.4g} last={last:.4g} slope/step={slope:.3g} "
            f"proj{HORIZON}={proj:.3g} tol={tol:.3g}"
        )
    head = "PASS" if v["ok"] else ("FAIL" if v["rows"] else "NO DATA")
    lines.append(
        f"[leak] {head} {tag} steps={v['steps']} series={len(v['rows'])} growing={','.join(v['failed']) or '-'}"
    )
    return lines


def main(argv: Optional[list[str]] = None) -> int:
    """``python -m thundersync_rl.leak_probe RUN_DIR|history.json ... [--horizon N] [--all]``."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--all", action="store_true", help="print every series")
    a = ap.parse_args(argv)
    rc = 0
    for run in a.runs:
        path = run if run.endswith(".json") else os.path.join(run, "history.json")
        try:
            with open(path) as f:
                hist = json.load(f)
        except (OSError, ValueError) as e:
            print(f"[leak] NO DATA {run}: {e}")
            rc = 1
            continue
        v = leak_verdict(hist, horizon=a.horizon)
        for line in format_verdict(v, run, top=10**6 if a.all else 12):
            print(line)
        rc |= not v["ok"]
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
