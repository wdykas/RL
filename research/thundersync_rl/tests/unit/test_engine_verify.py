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
"""Engine-side verification capture (B', verify_engine_share): CPU pieces."""

import numpy as np
import torch
from thundersync_rl.engine_verify import CaptureStore, capture_selection

def test_capture_store_positions_and_completeness():
    st = CaptureStore()
    st.begin("a", 4, 8, 3, torch.float32, "cpu")  # positions 4..7
    rows = torch.arange(12, dtype=torch.float32).view(4, 3)
    st.put("a", np.array([2, 3, 4, 5]), rows)  # 2, 3 outside: dropped
    assert not st.complete("a")
    st.put("a", np.array([6, 7]), rows[:2] + 100)
    assert st.complete("a")
    out = st.pop("a")
    assert torch.equal(out[0], rows[2]) and torch.equal(out[1], rows[3])
    assert torch.equal(out[2], rows[0] + 100)
    assert st.pop("a") is None


def test_capture_selection_keeps_tagged_window_only():
    # Slots: 0 untagged, 1 key 0 (lo 2, hi 5), 2 key 1 (lo 0, hi 1).
    slot_kid = np.array([-1, 0, 1])
    lo, hi = np.array([2, 0]), np.array([5, 1])
    slots = np.array([0, 1, 1, 1, 1, 1, 2, 2])
    pos = np.array([9, 0, 1, 2, 3, 4, 0, 1])
    sel, kid, p = capture_selection(slots, pos, slot_kid, lo, hi)
    assert sel.tolist() == [3, 4, 5, 6]
    assert kid.tolist() == [0, 0, 0, 1]
    assert p.tolist() == [2, 3, 4, 0]



def test_capture_buffer_is_engine_scratch():
    """The head-input capture's static buffer is allocated in the engine-scratch region (released during colocated
    training with torch_memory_saver, like the dk row workspace). Source check: importing the worker needs the
    engine stack."""
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "thundersync_rl" / "worker.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_install_verify_capture")
    withs = [w for w in ast.walk(fn) if isinstance(w, ast.With)
             and any(isinstance(i.context_expr, ast.Call) and getattr(i.context_expr.func, "id", "") ==
                     "engine_scratch_region" for i in w.items)]
    assert withs, "the capture buffer must be allocated inside engine_scratch_region()"
    assigned = {t.id for w in withs for s in w.body if isinstance(s, ast.Assign) for t in s.targets
                if isinstance(t, ast.Name)}
    assert "static" in assigned
