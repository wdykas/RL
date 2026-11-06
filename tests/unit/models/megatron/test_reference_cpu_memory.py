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

"""setup_reference_model_state: unpinned CPU snapshot under CPU offload; the temporary reference DDP model is freed.

Source: rl-determinism 69c3fd21 (tests/unit/models/megatron/test_reference_cpu_memory.py, the two setup tests)."""

import gc
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

pytestmark = [
    pytest.mark.mcore,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _model() -> torch.nn.Linear:
    model = torch.nn.Linear(384, 256, bias=False, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        model.weight.fill_(3)
    model.register_buffer("scale", torch.arange(8, device="cuda", dtype=torch.float32))
    model.register_buffer("transposed", torch.arange(63, device="cuda").reshape(7, 9).T)
    return model


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_reference_setup_copies_independent_cpu_state(
    cpu_offload: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nemo_rl.models.megatron import setup

    model = _model()
    original = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    config = {"megatron_cfg": {"optimizer": {"optimizer_cpu_offload": cpu_offload}}}
    mcfg = SimpleNamespace(
        model=SimpleNamespace(),
        checkpoint=None,
        logger=None,
        train=None,
        optimizer=SimpleNamespace(overlap_param_gather_with_optimizer_step=False),
        ddp=None,
        scheduler=None,
        dataset=None,
        tokenizer=None,
        dist=SimpleNamespace(use_torch_fsdp2=False),
        rng=SimpleNamespace(data_parallel_random_init=False),
    )
    # Replace model construction/checkpoint I/O, keeping the actual state-copy path.
    monkeypatch.setattr(setup, "ConfigContainer", SimpleNamespace)
    monkeypatch.setattr(setup, "init_checkpointing_context", Mock(return_value={}))
    monkeypatch.setattr(setup, "get_model", Mock(return_value=[model]))
    monkeypatch.setattr(setup, "checkpoint_exists", Mock(return_value=True))
    monkeypatch.setattr(setup, "load_checkpoint", Mock())
    monkeypatch.setattr(setup, "clear_global_router_replay_instances", Mock())
    monkeypatch.setattr(
        setup.ProcessGroupCollection, "use_mpu_process_groups", Mock(return_value=None)
    )
    state = setup.setup_reference_model_state(config, mcfg, "/unused-checkpoint")
    torch.cuda.synchronize()
    for key, value in state.items():
        assert value.device.type == "cpu"
        assert torch.equal(value, original[key])
        if cpu_offload:
            assert not value.is_pinned()
    with torch.no_grad():
        for value in model.state_dict().values():
            value.zero_()
    for key, value in state.items():
        assert torch.equal(value, original[key])


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_reference_setup_releases_temporary_ddp_buffers(
    cpu_offload: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # These dependencies are available only in the Megatron test environment.
    from megatron.core import parallel_state
    from megatron.core.distributed import (
        DistributedDataParallel,
        DistributedDataParallelConfig,
    )
    from megatron.core.transformer import TransformerConfig

    from nemo_rl.models.megatron import setup

    references: list[weakref.ReferenceType[DistributedDataParallel]] = []

    def build(*args: Any, **kwargs: Any) -> list[DistributedDataParallel]:
        module = _model()
        model = DistributedDataParallel(
            TransformerConfig(
                num_layers=1,
                hidden_size=256,
                num_attention_heads=8,
                params_dtype=torch.bfloat16,
                bf16=True,
            ),
            DistributedDataParallelConfig(
                use_distributed_optimizer=True,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
                grad_reduce_in_fp32=False,
            ),
            module,
        )
        references.append(weakref.ref(model))
        return [model]

    config = {"megatron_cfg": {"optimizer": {"optimizer_cpu_offload": cpu_offload}}}
    mcfg = SimpleNamespace(
        model=SimpleNamespace(),
        checkpoint=None,
        logger=None,
        train=None,
        optimizer=SimpleNamespace(overlap_param_gather_with_optimizer_step=False),
        ddp=None,
        scheduler=None,
        dataset=None,
        tokenizer=None,
        dist=SimpleNamespace(use_torch_fsdp2=False),
        rng=SimpleNamespace(data_parallel_random_init=False),
    )
    monkeypatch.setattr(setup, "ConfigContainer", SimpleNamespace)
    monkeypatch.setattr(setup, "init_checkpointing_context", Mock(return_value={}))
    monkeypatch.setattr(setup, "get_model", build)
    monkeypatch.setattr(setup, "checkpoint_exists", Mock(return_value=True))
    # Mock.call_args would itself retain the temporary model under test.
    monkeypatch.setattr(setup, "load_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(setup, "clear_global_router_replay_instances", Mock())
    created_process_group = False
    gc_was_enabled = gc.isenabled()
    try:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                "nccl",
                rank=0,
                world_size=1,
                init_method=f"file://{tmp_path / 'reference_pg'}",
            )
            created_process_group = True
        parallel_state.initialize_model_parallel()
        gc.collect()
        # Cleanup must not depend on when Python next schedules cyclic GC.
        gc.disable()
        allocated = torch.cuda.memory_allocated()
        state = setup.setup_reference_model_state(config, mcfg, "/unused-checkpoint")
        torch.cuda.synchronize()
        assert references[0]() is None
        assert torch.cuda.memory_allocated() == allocated
        assert all(value.device.type == "cpu" for value in state.values())
        assert torch.equal(state["weight"], torch.full_like(state["weight"], 3))
    finally:
        if gc_was_enabled:
            gc.enable()
        parallel_state.destroy_model_parallel()
        if created_process_group:
            torch.distributed.destroy_process_group()
