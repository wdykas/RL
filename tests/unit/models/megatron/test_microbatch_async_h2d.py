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
"""Micro-batch host-to-device move (megatron/data.py ``to_cuda_async``): pinned, asynchronous, same values."""

import time

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.data import to_cuda_async

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


def _batch():
    g = torch.Generator().manual_seed(0)  # the same batch on every call
    ids = torch.randint(0, 131072, (2, 8448), dtype=torch.long, generator=g)  # a verify / training micro-batch
    return BatchedDataDict(
        {
            "input_ids": ids,
            "input_lengths": torch.tensor([8448, 8300]),
            "routed_experts": torch.randint(-1, 512, (2, 8448, 4, 22), dtype=torch.int16, generator=g),
            "tag": ["a", "b"],
        }
    )


@cuda
def test_values_devices_and_dtypes_are_those_of_to_cuda():
    want = _batch().to("cuda")
    got = to_cuda_async(_batch())
    torch.cuda.synchronize()
    for k in ("input_ids", "input_lengths", "routed_experts"):
        assert got[k].is_cuda and got[k].dtype == want[k].dtype
        assert torch.equal(got[k], want[k])
    assert got["tag"] == ["a", "b"]


@cuda
def test_already_on_device_is_untouched():
    b = _batch().to("cuda")
    ref = b["input_ids"]
    assert to_cuda_async(b)["input_ids"] is ref


def _host_seconds(move):
    torch.cuda.synchronize()
    b = _batch()
    torch.cuda._sleep(int(2e9))  # ~1 s of queued GPU work on the current stream
    t0 = time.perf_counter()
    out = move(b)
    dt = time.perf_counter() - t0
    torch.cuda.synchronize()
    return dt, out


@cuda
def test_does_not_wait_for_queued_kernels():
    """The point of the change: the host returns while earlier kernels are still running."""
    dt_new, out = _host_seconds(to_cuda_async)
    dt_old, ref = _host_seconds(lambda b: b.to("cuda"))
    assert torch.equal(out["input_ids"], ref["input_ids"])
    print(f"host time with ~1 s queued: pinned async {dt_new * 1e3:.1f} ms, pageable {dt_old * 1e3:.1f} ms")
    assert dt_new < 0.1
    # Negative control (documents the bug on this GPU / driver): the pageable copy waited for the queue.
    assert dt_old > 0.3


@cuda
def test_no_synchronizing_call():
    b = _batch()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        to_cuda_async(b)
    finally:
        torch.cuda.set_sync_debug_mode("default")
