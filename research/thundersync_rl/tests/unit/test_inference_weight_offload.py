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
"""Colocated inference-weight offload without a CPU backup (torch_memory_saver, TMS=1 overlay; GPU).

When every wake reshards fresh weights in, the inference model's region is paused without a CPU backup: the
refit tensors (parameters, persistent buffers) are rewritten by the reshard after the resume, and the region's
other buffers (non-persistent) are kept by pause / resume_inference_weights. Checked on a small module.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class _Block(torch.nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.lin = torch.nn.Linear(2048, 2048, device=device)
        self.register_buffer("scale", torch.ones(2048, device=device))  # persistent: refit
        self.register_buffer("table", torch.zeros(4096, 1024, device=device), persistent=False)  # not refit

    def forward(self, x):
        return self.lin(x) * self.scale + self.table[: x.shape[0], :1]


def test_reshard_rewrites_weights_except_negative_control():
    import os

    from nemo_rl.models.megatron import memory_saver as ms

    old = os.environ.pop("NEMO_RL_DEBUG_COLOC_SKIP_RESHARD", None)
    try:
        assert ms.reshard_rewrites_weights()
        os.environ["NEMO_RL_DEBUG_COLOC_SKIP_RESHARD"] = "1"
        assert not ms.reshard_rewrites_weights()
    finally:
        os.environ.pop("NEMO_RL_DEBUG_COLOC_SKIP_RESHARD", None)
        if old is not None:
            os.environ["NEMO_RL_DEBUG_COLOC_SKIP_RESHARD"] = old


# Region tensors must not be freed outside a region before process teardown (torch_memory_saver exits 1): the tests
# keep theirs alive here; memory_saver's teardown hook makes the frees at exit legal.
_HELD: list = []


def _need_tms():
    try:
        import torch_memory_saver  # noqa: F401
    except ImportError:
        pytest.skip("torch_memory_saver (see docs/colo/HANDOFF.md, overlays) not on PYTHONPATH")


def _build_in_region(ms, extra=None):
    dev = torch.cuda.current_device()
    torch.cuda.synchronize()
    before = ms.active_blocks(dev)
    with ms.inference_model_alloc_region(cpu_backup=False):
        tgt = _Block(device="cuda")
        tgt.cache = torch.full((512, 512), 3.0, device="cuda")  # a plain tensor attribute (a derived cache)
        stray = torch.empty(1 << 20, device="cuda") if extra else None
    torch.cuda.synchronize()
    built = {a: n for a, n in ms.active_blocks(dev).items() if a not in before}
    _HELD.extend([tgt, stray])
    return tgt, built, stray


def test_inference_weights_released_without_backup_rebuilt_by_reshard():
    """Review E2 / E4 (batch 19): the region accounting keeps every non-refit tensor (buffers and tensor attributes),
    the first pause checks the release, and after the wake's reshard the model computes the same bits."""
    _need_tms()
    from megatron.core.resharding.utils import named_refit_tensors

    from nemo_rl.models.megatron import memory_saver as ms

    torch.manual_seed(0)
    src = _Block(device="cuda")
    src.scale.uniform_(0.5, 1.5)
    src.table.normal_()
    tgt, built, _ = _build_in_region(ms)
    keep, region_bytes = ms.account_inference_region(tgt, built)
    assert {id(t) for t in keep} == {id(tgt.table), id(tgt.cache)}
    ms.set_inference_region_bytes(region_bytes)
    with torch.no_grad():
        tgt.load_state_dict(src.state_dict())  # the build-time weights (params + persistent buffers)
        tgt.table.copy_(src.table)  # non-persistent state the model computed at build
    x = torch.randn(64, 2048, device="cuda")
    ref = src(x)
    assert torch.equal(tgt(x), ref)

    torch.cuda.synchronize()
    free0 = torch.cuda.mem_get_info()[0]
    kept = ms.pause_inference_weights(keep)  # first pause: checks >= 0.9 x region_bytes came back
    assert torch.cuda.mem_get_info()[0] - free0 >= 0.9 * region_bytes
    ms.resume_inference_weights(kept)
    assert torch.equal(tgt.table, src.table) and bool((tgt.cache == 3.0).all())  # kept across the pause
    src_t = dict(named_refit_tensors(src))  # the wake's reshard: every refit tensor rewritten
    with torch.no_grad():
        for name, t in named_refit_tensors(tgt):
            t.copy_(src_t[name])
    assert torch.equal(tgt(x), ref)


def test_unreachable_region_allocation_is_refused():
    """Review E2: an allocation of the no-backup region that no model tensor holds would come back as garbage."""
    _need_tms()
    from nemo_rl.models.megatron import memory_saver as ms

    tgt, built, stray = _build_in_region(ms, extra=True)
    with pytest.raises(RuntimeError, match="hold no tensor reachable"):
        ms.account_inference_region(tgt, built)


def test_release_check_refuses_a_pause_that_frees_nothing():
    """Review E4: TMS requested but not functional (nothing released) is an error."""
    _need_tms()
    from nemo_rl.models.megatron import memory_saver as ms

    free = torch.cuda.mem_get_info()[0]
    with pytest.raises(RuntimeError, match="not functional"):
        ms._released_or_raise("a test region", free, 1 << 30)


def test_process_with_paused_regions_exits_zero():
    """Review E3: a process holding paused and live torch_memory_saver regions exits with status 0 (without the
    teardown hook torch_memory_saver exits 1 when interpreter shutdown frees region tensors outside a region)."""
    _need_tms()
    import subprocess
    import sys

    code = (
        "import torch\n"
        "from nemo_rl.models.megatron import memory_saver as ms\n"
        "from nemo_rl.models.megatron import fp16_moe_dk as dk\n"
        "with ms.inference_model_alloc_region(cpu_backup=False):\n"
        "    w = torch.ones(1 << 24, device='cuda')\n"
        "dk.reserve_row_workspace(torch.device('cuda'), 4096, 6, 16, 256, 128, colocated=True)\n"
        "ms.set_inference_region_bytes(w.numel() * 4)\n"
        "ms.pause_inference_weights(())\n"
        "ms.pause_engine_scratch()\n"
        "print('paused', flush=True)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "paused" in r.stdout, (r.returncode, r.stdout[-2000:], r.stderr[-2000:])
