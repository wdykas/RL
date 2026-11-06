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
"""The synchronous torch-dist save runs DCP's plan exchange on a Gloo group (setup._patch_bridge_sync_save_plan_group).

On the NCCL world group DCP's gather_object / scatter_object_list open persistent NCCL P2P connections from the
coordinator to every peer (+0.375 GiB per peer on GB200, kept for the run). Real 2-process Gloo world, real MCore
strategy and DCP; the save itself is real (files written, loadable)."""

import os
import socket
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("megatron.core.dist_checkpointing")
pytest.importorskip("megatron.bridge.training.checkpointing")

CODE = textwrap.dedent(
    """
    import os, sys, torch, torch.distributed as dist
    rank, world, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from nemo_rl.models.megatron import setup
    from megatron.core.dist_checkpointing import ShardedTensor
    from megatron.core.dist_checkpointing.strategies import torch as mcore_torch
    import torch.distributed.checkpoint as dcp
    setup._patch_bridge_sync_save_plan_group()
    from megatron.bridge.training import checkpointing as bridge_ckpt
    seen = []
    real = mcore_torch.checkpoint.save
    def spy(*a, **k):
        seen.append(k.get("process_group"))
        return real(*a, **k)
    mcore_torch.checkpoint.save = spy
    # A gather/scatter on the default group fails the test: the plan exchange must not touch it.
    default = dist.distributed_c10d._get_default_group()
    for name in ("gather_object", "scatter_object_list"):
        f = getattr(dist, name)
        def guard(*a, _f=f, _n=name, **k):
            assert k.get("group") not in (None, default), f"{_n} on the default group"
            return _f(*a, **k)
        setattr(dist, name, guard)
        setattr(dist.distributed_c10d, name, guard)
    t = torch.full((4,), float(rank))
    sd = {"w": ShardedTensor.from_rank_offsets("w", t, (0, rank, world))}
    s = bridge_ckpt.TorchDistSaveShardedStrategy("torch_dist", 1)
    s.save(sd, out)
    pg = seen[0]
    assert pg is not None and pg is not default and dist.get_backend(pg) == "gloo", pg
    assert dist.get_world_size(pg) == world
    dist.barrier()
    if rank == 0:
        from torch.distributed.checkpoint import FileSystemReader
        md = FileSystemReader(out).read_metadata()
        assert "w" in md.state_dict_metadata, md.state_dict_metadata.keys()
    print("ok", rank)
    """
)


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_sync_save_plan_exchange_on_gloo(tmp_path):
    env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT=str(_port()), CUDA_VISIBLE_DEVICES="")
    procs = [
        subprocess.Popen([sys.executable, "-c", CODE, str(r), "2", str(tmp_path / "ckpt")], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for r in range(2)
    ]
    outs = [p.communicate(timeout=300) for p in procs]
    for p, (o, e) in zip(procs, outs):
        assert p.returncode == 0 and "ok" in o, e[-3000:]
