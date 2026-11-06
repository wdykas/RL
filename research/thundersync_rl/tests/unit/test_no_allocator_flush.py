# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""No runtime allocator management on the learner step path (no per-step empty_cache, no flush knob)."""

import ast
import inspect
import textwrap

from thundersync_rl import worker


def _calls(src: str) -> set[str]:
    """Names of the functions / methods called in ``src`` (comments and strings do not count)."""
    out = set()
    for node in ast.walk(ast.parse(textwrap.dedent(src))):
        if isinstance(node, ast.Call):
            f = node.func
            out.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return out


def _string_constants(src: str) -> set[str]:
    return {
        n.value
        for n in ast.walk(ast.parse(textwrap.dedent(src)))
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }


def test_no_empty_cache_frac_knob():
    assert "THUNDERSYNC_EMPTY_CACHE_FRAC" not in _string_constants(inspect.getsource(worker))


def test_allocator_stats_reports_churn_counters():
    consts = _string_constants(inspect.getsource(worker.allocator_stats))
    assert {"learner_mem_device_allocs", "learner_mem_device_frees", "num_device_alloc", "num_device_free"} <= consts
    if worker.torch.cuda.is_available():
        assert {"learner_mem_device_allocs", "learner_mem_device_frees"} <= set(worker.allocator_stats())
