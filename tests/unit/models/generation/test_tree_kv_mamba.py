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
"""tree_spec / tree_kv are self-contained and run on Megatron-LM main (a11c84aec) as on earlier pins:

- no import of the multi-token tail decode modules (tail_mtd, mamba_bi_lookahead), which colo-slim does not carry;
- the Mamba node scratch rows are allocated explicitly on the decode buffers MCore hands out (``_get_buffers``),
  whoever allocated them (main allocates z-free buffers itself and never runs nemo_rl's allocate hook), registered
  once per buffers object, in layer order;
- the scratch conv rows sit after ALL of the context's conv-state slots (main may reserve a dummy recurrent slot);
- the controller's async-schedule state is found under main's name (_async_sched_forward) and the earlier one;
- the retired gate-tool env name is refused, not silently ignored.
"""

import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
TREE_FILES = [ROOT / "nemo_rl/models/generation/megatron/tree_spec.py",
              ROOT / "nemo_rl/models/generation/megatron/tree_kv.py"]
RETIRED = {"nemo_rl.models.generation.megatron.tail_mtd", "nemo_rl.models.megatron.mamba_bi_lookahead"}

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _imports(path: Path) -> set:
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
            out |= {f"{node.module}.{a.name}" for a in node.names}
    return out


@pytest.mark.parametrize("path", TREE_FILES, ids=lambda p: p.name)
def test_no_mtd_imports(path):
    bad = sorted(m for m in _imports(path) if any(m == r or m.startswith(r + ".") for r in RETIRED))
    assert not bad, f"{path.name} imports retired speculation modules: {bad}"


def test_import_with_mtd_modules_absent(monkeypatch):
    """tree_spec / tree_kv import with tail_mtd and mamba_bi_lookahead unimportable (as on colo-slim)."""
    import importlib
    import sys

    for m in RETIRED:
        monkeypatch.setitem(sys.modules, m, None)  # `import m` raises ImportError
    for m in ("nemo_rl.models.generation.megatron.tree_spec", "nemo_rl.models.generation.megatron.tree_kv"):
        monkeypatch.delitem(sys.modules, m, raising=False)
        mod = importlib.import_module(m)
        assert mod is not None
    from nemo_rl.models.generation.megatron import tree_spec as ts

    assert ts.MTD_REFS_ATTR == "ts_mtd_refs" and ts.SEP == -1  # the drivers' wire format is unchanged
    packed = ts.pack_refs([[5, 6], [7]])
    assert packed.tolist() == [5, 6, -1, 7, -1]
    assert ts._ListSample([1, [2, 3]]).tolist() == [1, [2, 3]]


def test_async_sched_state_both_pins():
    from nemo_rl.models.generation.megatron import tree_spec as ts

    main, old = object(), object()
    assert ts.async_sched_state(SimpleNamespace(_async_sched_forward=main)) is main
    assert ts.async_sched_state(SimpleNamespace(_async_sched_logits=old)) is old
    assert ts.async_sched_state(SimpleNamespace(_async_sched_forward=main, _async_sched_logits=old)) is main
    with pytest.raises(RuntimeError, match="async-schedule state"):
        ts.async_sched_state(SimpleNamespace())


def test_force_width_env(monkeypatch):
    from nemo_rl.models.generation.megatron import tree_spec as ts

    for k in (ts.FORCE_W_ENV, ts.RETIRED_FORCE_ENV, "THUNDERSYNC_NL_CHECKS"):
        monkeypatch.delenv(k, raising=False)
    assert ts.force_width() is None
    monkeypatch.setenv(ts.FORCE_W_ENV, "8")
    with pytest.raises(RuntimeError, match="gate tool"):
        ts.force_width()
    monkeypatch.setenv("THUNDERSYNC_NL_CHECKS", "1")
    assert ts.force_width() == 8
    monkeypatch.setenv(ts.RETIRED_FORCE_ENV, "4")
    with pytest.raises(RuntimeError, match="retired"):
        ts.force_width()


@pytest.mark.parametrize("dummy_slot", [0, 1])
def test_conv_scratch_rows_after_every_slot(monkeypatch, dummy_slot):
    """_allocate_mamba_states: MCore's slots (+ a dummy recurrent slot) then node_cap scratch rows, one storage; MCore's
    view keeps its shape and values; the scratch base is the slot count."""
    from nemo_rl.models.generation.megatron import tree_kv

    L, slots, cap = 3, 6 + dummy_slot, 5

    def mcore_alloc(self):
        self.mamba_conv_states = torch.arange(L * slots * 4 * 2, dtype=torch.float32).view(L, slots, 4, 2)

    monkeypatch.setitem(tree_kv._ORIG, "allocate_states", mcore_alloc)
    monkeypatch.setitem(tree_kv._STATE, "cap", cap)
    ctx = SimpleNamespace(is_hybrid_model=True, batch_invariant_mode=True)
    tree_kv._allocate_mamba_states(ctx)
    full = tree_kv.conv_states_full(ctx)
    assert full.shape == (L, slots + cap, 4, 2)
    assert ctx.mamba_conv_states.shape == (L, slots, 4, 2)
    assert ctx.mamba_conv_states.untyped_storage().data_ptr() == full.untyped_storage().data_ptr()
    assert torch.equal(ctx.mamba_conv_states, torch.arange(L * slots * 8, dtype=torch.float32).view(L, slots, 4, 2))
    assert tree_kv.conv_scratch_base(ctx) == slots
    # the per-layer conv state MCore passes to ssm_decode, widened: node rows land after every slot of that layer
    wide = tree_kv._full_of(ctx.mamba_conv_states[1])
    assert wide.shape[0] == slots + cap and wide.data_ptr() == full[1].data_ptr()
    wide[slots].fill_(-7.0)
    assert bool((full[1, slots] == -7.0).all())
    # a non-batch-invariant / non-hybrid context is left as MCore built it
    ctx2 = SimpleNamespace(is_hybrid_model=True, batch_invariant_mode=False)
    tree_kv._allocate_mamba_states(ctx2)
    assert tree_kv.conv_states_full(ctx2) is None


def _main_like_buffers(R, chunk, nh, hd, ng, ds, device, z=False):
    """A stand-in for Megatron-LM main's z-free BatchInvariantDecodeBuffers dataclass (fields as on a11c84aec)."""

    @dataclasses.dataclass
    class Buffers:
        x: torch.Tensor
        dt: torch.Tensor
        B: torch.Tensor
        C: torch.Tensor
        num_buffered: torch.Tensor

    f = dict(device=device, dtype=torch.float32)
    b = Buffers(x=torch.zeros(R, chunk, nh, hd, **f), dt=torch.zeros(R, chunk, nh, **f),
                B=torch.zeros(R, chunk, ng, ds, **f), C=torch.zeros(R, chunk, ng, ds, **f),
                num_buffered=torch.zeros(R, device=device, dtype=torch.int32))
    if z:
        b.z = torch.zeros(R, chunk, nh, hd, **f)
    return b


@cuda
def test_scratch_rows_allocated_explicitly_on_mcore_buffers(monkeypatch):
    """The ``_get_buffers`` hook gives MCore-allocated buffers their scratch rows (no nemo_rl allocate hook runs), once,
    registered in layer order whatever the call order; unsupported buffers are refused loudly."""
    from tests.unit.models.megatron.test_mamba_bi_fused_decode import SHAPES

    from nemo_rl.models.generation.megatron import tree_kv
    from nemo_rl.models.megatron import mamba_bi_fused_decode as fd

    nh, hd, ng, ds, chunk = SHAPES["super_tp2"]
    if not fd.supported_config(nh, hd, ng, ds, chunk, torch.float32, torch.device("cuda")):
        pytest.skip("fused decode step does not serve this shape on this GPU")
    cap = 7
    monkeypatch.setitem(tree_kv._STATE, "cap", cap)
    monkeypatch.setattr(tree_kv, "_MAMBA", {"bufs": [], "nbuf_ptrs": None, "replay_ptrs": None})
    made = {}

    def mcore_get_buffers(self, max_requests, x, B):  # MCore's lazy allocation (main: z-free, its own allocate)
        if self.mixer.layer_number not in made:
            made[self.mixer.layer_number] = _main_like_buffers(max_requests, chunk, nh, hd, ng, ds, x.device)
        return made[self.mixer.layer_number]

    monkeypatch.setitem(tree_kv._ORIG, "get_buffers", mcore_get_buffers)
    x = torch.zeros(1, device="cuda")
    adapters = {n: SimpleNamespace(mixer=SimpleNamespace(layer_number=n)) for n in (9, 2, 5)}
    for n in (9, 2, 5, 2, 9):  # first use out of layer order, then repeated uses
        buf = tree_kv._get_buffers(adapters[n], 12, x, x)
        assert buf is made[n]
        assert [t.shape for t in buf.scratch] == [(cap, nh, hd), (cap, nh), (cap, ng, ds), (cap, ng, ds)]
    assert tree_kv.mamba_layers() == 3
    ptrs = tree_kv.num_buffered_ptrs(torch.device("cuda")).tolist()
    assert ptrs == [made[n].num_buffered.data_ptr() for n in (2, 5, 9)]
    rp, sizes = tree_kv.replay_ptrs(torch.device("cuda"))
    assert rp.shape == (8, 3) and rp[4].tolist() == [made[n].scratch[0].data_ptr() for n in (2, 5, 9)]
    assert sizes == (nh * hd, nh, ng * ds)
    # buffers the fused decode does not serve (z buffered) cannot carry node rows: refused
    with pytest.raises(RuntimeError, match="fused decode"):
        tree_kv.attach_scratch(_main_like_buffers(4, chunk, nh, hd, ng, ds, "cuda", z=True), 11)


@cuda
def test_installs_on_the_pinned_megatron(monkeypatch):
    """The hooks tree_kv installs exist on the pinned Megatron-Core with the signatures it wraps (main a11c84aec and
    a68555ff alike)."""
    import inspect

    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
    from megatron.core.ssm.mamba_mixer import MambaMixer
    from megatron.core.ssm.ops.mamba2.batch_invariant_decode import MambaBatchInvariantDecode

    assert list(inspect.signature(MambaBatchInvariantDecode._get_buffers).parameters) == ["self", "max_requests", "x",
                                                                                          "B"]
    assert list(inspect.signature(MambaMixer.ssm_decode).parameters)[:4] == ["self", "zxBCdt", "conv_state",
                                                                             "ssm_state"]
    assert callable(DynamicInferenceContext._allocate_mamba_states)
    assert callable(DynamicInferenceContext.append_key_value_cache)
