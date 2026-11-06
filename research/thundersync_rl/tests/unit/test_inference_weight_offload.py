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


def test_weights_built_outside_then_moved_into_one_region_allocation():
    """The no-backup path builds the model in the default pool and moves its tensors into one region allocation
    (torch_memory_saver's pause / resume cost is per allocation): one region block, same values / dtypes / strides /
    Parameter objects / views; the moved tensors no refit tensor shares a storage with are kept across the pause;
    after the wake's reshard the model computes the same bits."""
    _need_tms()
    from megatron.core.resharding.utils import named_refit_tensors

    from nemo_rl.models.megatron import memory_saver as ms

    dev = torch.cuda.current_device()
    torch.cuda.synchronize()
    before = ms.active_blocks(dev)
    m = _Block(device="cuda")  # default pool
    m.scale.uniform_(0.5, 1.5)
    m.table.normal_()
    m.half_w = torch.randn(300, 7, device="cuda").half()  # a tensor attribute of another dtype
    m.w_t = m.lin.weight.t()  # a strided view of a parameter
    torch.cuda.synchronize()
    built = {a: n for a, n in ms.active_blocks(dev).items() if a not in before}
    ref = {k: v.detach().cpu() for k, v in [("w", m.lin.weight), ("b", m.lin.bias), ("scale", m.scale),
                                           ("table", m.table), ("half_w", m.half_w)]}
    params = [m.lin.weight, m.lin.bias]
    from megatron.core.resharding.utils import get_refit_tensor_dict

    get_refit_tensor_dict(m)  # MCore's refit tensor cache, filled before the move (as a plan built at init would be)
    pre = ms.active_blocks(dev)
    reserved0 = torch.cuda.memory_reserved()
    with ms.inference_model_alloc_region(cpu_backup=False):
        nbytes, keep, (n_left, _) = ms.consolidate_into_arena(m, built)
    assert n_left == 0
    _HELD.append(m)
    torch.cuda.synchronize()
    region = {a: n for a, n in ms.active_blocks(dev).items() if a not in pre}
    # the refit cache after the move holds the moved tensors: a reshard writes the arena
    arena_ptr = m.lin.weight.untyped_storage().data_ptr()
    assert all(t.untyped_storage().data_ptr() == arena_ptr for t in get_refit_tensor_dict(m).values())
    # the build's originals went back to the default caching allocator: an allocation of the same bytes there needs
    # no new memory (the learner reuses them)
    moved = sum(n for n in built.values())
    reserved1 = torch.cuda.memory_reserved()
    reuse = [torch.empty(n, dtype=torch.uint8, device="cuda") for n in built.values()]
    assert torch.cuda.memory_reserved() == reserved1, (reserved0, reserved1, torch.cuda.memory_reserved(), moved)
    del reuse
    assert len(region) == 1 and nbytes >= sum(t.numel() * t.element_size() for t in ref.values()), region
    assert {id(t) for t in keep} == {id(m.table), id(m.half_w)}  # not refit: kept across a pause
    assert params[0] is m.lin.weight and params[1] is m.lin.bias
    for k, t in [("w", m.lin.weight), ("b", m.lin.bias), ("scale", m.scale), ("table", m.table), ("half_w", m.half_w)]:
        assert t.dtype == ref[k].dtype and t.shape == ref[k].shape and torch.equal(t.cpu(), ref[k]), k
    assert m.w_t.untyped_storage().data_ptr() == m.lin.weight.untyped_storage().data_ptr()
    assert torch.equal(m.w_t, m.lin.weight.t()) and m.w_t.stride() == (1, 2048)
    x = torch.randn(8, 2048, device="cuda")
    out = m(x)
    ms.set_inference_region_bytes(nbytes)
    src = {k: t.detach().clone() for k, t in named_refit_tensors(m)}  # the training weights a wake reshards in
    kept = ms.pause_inference_weights(keep)
    ms.resume_inference_weights(kept)
    assert torch.equal(m.table.cpu(), ref["table"]) and torch.equal(m.half_w.cpu(), ref["half_w"])
    with torch.no_grad():
        for k, t in named_refit_tensors(m):
            t.copy_(src[k])
    assert torch.equal(m(x), out)


def test_arena_refuses_a_hidden_holder_of_a_moved_storage():
    """Review batch 31 (b): a storage something outside the module tree still holds stays allocated after the move
    (twice the memory, stale reads after a reshard): refused."""
    _need_tms()
    from nemo_rl.models.megatron import memory_saver as ms

    dev = torch.cuda.current_device()
    torch.cuda.synchronize()
    before = ms.active_blocks(dev)
    m = _Block(device="cuda")
    torch.cuda.synchronize()
    built = {a: n for a, n in ms.active_blocks(dev).items() if a not in before}
    hidden = {"w": m.lin.weight.detach()}  # a cache outside the module tree, on the original storage
    with ms.inference_model_alloc_region(cpu_backup=False):
        with pytest.raises(RuntimeError, match="still allocated after the move"):
            ms.consolidate_into_arena(m, built)
    _HELD.extend([m, hidden])


class _Ctx:
    """A stand-in for DynamicInferenceContext: KV buffer, Mamba state buffers, allocator and slot bookkeeping."""


def _kv_ctx(proxy, tag, stray=False):
    import types

    pool, layers, bs, heads, dim, slots = 64, 2, 16, 4, 32, 8
    ctx = _Ctx()
    with proxy.region(tag=tag, enable_cpu_backup=True):
        ctx.memory_buffer = torch.randn(2, layers, pool, bs, heads, dim, device="cuda")
        ctx.mamba_conv_states = torch.randn(layers, slots, 3, 5, device="cuda")
        ctx.mamba_ssm_states = torch.randn(layers, slots, 4, 6, device="cuda")
        ctx.step_scratch = torch.randn(100, device="cuda")  # another region tensor: kept whole
        if stray:
            _HELD.append(torch.empty(1 << 20, device="cuda"))  # held by nothing the context reaches
    ctx.kv_block_allocator = types.SimpleNamespace(
        block_bag=torch.randperm(pool - 1).int().tolist() + [pool - 1], pool_avail=pool - 1 - 10, pool_size=pool
    )
    ctx.kv_block_allocator.block_bag = torch.tensor(ctx.kv_block_allocator.block_bag, dtype=torch.int32)
    ctx.mamba_metadata = types.SimpleNamespace(
        mamba_state_free_slots=torch.randperm(slots).int(), mamba_state_free_slot_count=5, max_requests=slots
    )
    # owners: 3 resident requests holding the 10 allocated blocks (rows padded with -1) and the 3 used slots
    used = ctx.kv_block_allocator.block_bag[ctx.kv_block_allocator.pool_avail : pool - 1].tolist()
    ctx.request_to_kv_block_ids = torch.tensor([used[0:4], used[4:8], used[8:10] + [-1, -1]])
    ctx.total_request_count = 3
    ctx.mamba_metadata.request_to_mamba_state_idx = ctx.mamba_metadata.mamba_state_free_slots[5:].clone()
    _HELD.append(ctx)
    return ctx


def test_kv_offload_saves_only_resident_blocks_and_slots():
    """kv_live_offload: the KV / state region has no CPU backup; a suspend copies the blocks outside the free pool
    (10 + the dummy) and the slots outside the free-slot stack (3), plus other region tensors whole, and the resume
    writes them back bitwise. A region allocation no context tensor holds is refused."""
    _need_tms()
    import weakref

    from torch_memory_saver import torch_memory_saver as real

    from nemo_rl.models.megatron import kv_live_offload as kl

    tag = "test_kv_live"
    proxy = kl._SaverProxy(real, tag)
    kl._STATE.update(blocks={}, saved=None)
    ctx = _kv_ctx(proxy, tag)
    kl._STATE["ctx"] = weakref.ref(ctx)
    used = ctx.kv_block_allocator.block_bag[ctx.kv_block_allocator.pool_avail :].long().cuda()
    live_slots = ctx.mamba_metadata.mamba_state_free_slots[5:].long().cuda()
    ref_kv = ctx.memory_buffer.index_select(2, used).cpu()
    ref_conv = ctx.mamba_conv_states.index_select(1, live_slots).cpu()
    ref_ssm = ctx.mamba_ssm_states.index_select(1, live_slots).cpu()
    ref_scratch = ctx.step_scratch.cpu()
    torch.cuda.synchronize()
    free0 = torch.cuda.mem_get_info()[0]
    proxy.pause(tag)
    assert torch.cuda.mem_get_info()[0] > free0  # released
    st = kl.last_stats()
    want = 11 * ctx.memory_buffer[:, :, 0].numel() * 4 + 3 * (
        ctx.mamba_conv_states[:, 0].numel() + ctx.mamba_ssm_states[:, 0].numel()) * 4 + 100 * 4
    assert (st["live_blocks"], st["live_slots"], st["saved_bytes"]) == (11, 3, want), st
    proxy.resume(tag)
    assert torch.equal(ctx.memory_buffer.index_select(2, used).cpu(), ref_kv)
    assert torch.equal(ctx.mamba_conv_states.index_select(1, live_slots).cpu(), ref_conv)
    assert torch.equal(ctx.mamba_ssm_states.index_select(1, live_slots).cpu(), ref_ssm)
    assert torch.equal(ctx.step_scratch.cpu(), ref_scratch)
    # the owners' digest check (exactness checks on): a suspend that drops a live block fails at the resume
    import os

    os.environ["THUNDERSYNC_NL_CHECKS"] = "1"
    try:
        proxy.pause(tag)
        proxy.resume(tag)  # every live block / slot restored: passes
        assert kl.last_stats()["checked_blocks"] == 11
        kl.raise_if_check_failed()
        os.environ[kl.DROP_ONE_ENV] = "1"
        proxy.pause(tag)
        proxy.resume(tag)
        with pytest.raises(RuntimeError, match="changed across the suspend"):
            kl.raise_if_check_failed()
    finally:
        os.environ.pop(kl.DROP_ONE_ENV, None)
        os.environ.pop("THUNDERSYNC_NL_CHECKS", None)
    # a region allocation the context does not hold
    kl._STATE.update(blocks={})
    bad = _kv_ctx(kl._SaverProxy(real, tag + "_2"), tag + "_2", stray=True)
    with pytest.raises(RuntimeError, match="hold no context tensor"):
        kl._region_tensors(bad)


def test_kv_live_set_is_the_complement_of_the_free_stacks():
    """Review batch 36: Megatron's free pools are stacks and releases come out of order, so the entries above the free
    count are not the allocated set. The live set is the complement of the free part: allocate a,b,c then d,e,
    release b (MCore's push) -> d stays live, b is free; same for Mamba slots. A resident request's block / slot in
    the free set (or an allocated block no owner holds) is refused."""
    import types

    from nemo_rl.models.megatron import kv_live_offload as kl

    pool = 11  # 10 usable + the dummy (id 10)
    alloc = types.SimpleNamespace(block_bag=torch.arange(pool, dtype=torch.int32), pool_avail=pool - 1, pool_size=pool)

    def allocate(n):
        alloc.pool_avail -= n
        return alloc.block_bag[alloc.pool_avail : alloc.pool_avail + n].clone()

    def release(ids):
        alloc.block_bag[alloc.pool_avail : alloc.pool_avail + len(ids)] = ids
        alloc.pool_avail += len(ids)

    abc = allocate(3)
    de = allocate(2)
    release(abc[1:2])  # b
    md = types.SimpleNamespace(mamba_state_free_slots=torch.arange(6, dtype=torch.int32), mamba_state_free_slot_count=6,
                               max_requests=6)

    def take():
        md.mamba_state_free_slot_count -= 1
        return int(md.mamba_state_free_slots[md.mamba_state_free_slot_count])

    s1, s2, s3 = take(), take(), take()
    md.mamba_state_free_slots[md.mamba_state_free_slot_count] = s2  # free s2 (MCore free_slot)
    md.mamba_state_free_slot_count += 1
    ctx = _Ctx()
    ctx.kv_block_allocator, ctx.mamba_metadata = alloc, md
    ctx.total_request_count = 2
    ctx.request_to_kv_block_ids = torch.tensor([[int(abc[0]), int(abc[2])], [int(de[0]), int(de[1])]])
    md.request_to_mamba_state_idx = torch.tensor([s1, s3])
    blocks, slots = kl._live_index(ctx)
    want = sorted([int(abc[0]), int(abc[2]), int(de[0]), int(de[1]), pool - 1])
    assert sorted(blocks.cpu().tolist()) == want, (blocks, want)  # d live, b not, dummy kept
    assert sorted(slots.cpu().tolist()) == sorted([s1, s3])
    ctx.request_to_kv_block_ids[0, 0] = int(abc[1])  # a request claiming the freed b (and a left unowned)
    with pytest.raises(RuntimeError, match="not the allocator's non-free blocks"):
        kl._live_index(ctx)


def test_kv_live_owners_main_pin_prefix_cache_and_registered_holders():
    """Review batch 36 K2 / K3: on Megatron-LM main the prefix cache's hashes are on alloc.pc_state; a cached block
    (hash set, ref 0, outside the free stack) is an owner there too. A block reserved by another component (EX-T's
    private tree blocks) is an owner once registered, and refused before."""
    import types

    from nemo_rl.models.megatron import kv_live_offload as kl

    pool = 11
    bag = torch.arange(pool, dtype=torch.int32)
    alloc = types.SimpleNamespace(block_bag=bag, pool_avail=pool - 1 - 3, pool_size=pool)  # 7,8,9 allocated
    hashes = torch.full((pool,), -1, dtype=torch.int64)
    hashes[9] = 12345  # block 9: cached prefix block, no request holds it
    alloc.pc_state = types.SimpleNamespace(block_hashes=hashes)  # main-pin layout
    ctx = _Ctx()
    ctx.kv_block_allocator = alloc
    ctx.total_request_count = 1
    ctx.request_to_kv_block_ids = torch.tensor([[7, -1]])  # block 8: held by no request (a reserved block)
    with pytest.raises(RuntimeError, match="unowned"):
        kl._live_index(ctx)
    kl.register_block_owner("test_private_pool", lambda: [8])
    try:
        blocks, _ = kl._live_index(ctx)
        assert sorted(blocks.cpu().tolist()) == [7, 8, 9, 10]
    finally:
        kl._BLOCK_OWNERS.pop("test_private_pool")
    alloc.pc_state = None  # main pin, prefix caching off: block 9 is then unowned
    with pytest.raises(RuntimeError, match="unowned"):
        kl._live_index(ctx)


def test_kv_suspend_side_failure_reaches_the_caller_and_keeps_the_state():
    """Review batch 36 F1: a suspend whose live set cannot be established (an allocated block nobody owns) does not
    raise inside Megatron's pause (the engine loop thread would stall): the region stays mapped (nothing lost), the
    resume is a no-op, and raise_if_check_failed (the worker's sleep path) raises it."""
    _need_tms()
    import weakref

    from torch_memory_saver import torch_memory_saver as real

    from nemo_rl.models.megatron import kv_live_offload as kl

    tag = "test_kv_live_f1"
    proxy = kl._SaverProxy(real, tag)
    kl._STATE.update(blocks={}, saved=None)
    ctx = _kv_ctx(proxy, tag)
    kl._STATE["ctx"] = weakref.ref(ctx)
    ctx.request_to_kv_block_ids[0, 0] = -1  # one allocated block loses its owner
    ref = ctx.memory_buffer.cpu()
    torch.cuda.synchronize()
    free0 = torch.cuda.mem_get_info()[0]
    proxy.pause(tag)  # does not raise
    assert torch.cuda.mem_get_info()[0] <= free0 + (1 << 20)  # not released
    with pytest.raises(RuntimeError, match="unowned"):
        kl.raise_if_check_failed()
    proxy.resume(tag)  # no-op
    assert torch.equal(ctx.memory_buffer.cpu(), ref)
