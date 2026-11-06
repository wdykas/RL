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
"""Order-invariant exact gradient accumulator (nemo_rl/algorithms/exact_grad_accum.py): the accumulated gradient is
bitwise identical under random arrival permutations; plain fp32 accumulation is not (the gate's negative)."""

import pytest
import torch

from nemo_rl.algorithms.exact_grad_accum import OrderInvariantGradAccumulator, from_fixed, to_fixed


def _groups(n_groups: int, sizes: tuple[int, ...], seed: int) -> list[list[torch.Tensor]]:
    """Per-group gradients over several flat buffers: wide dynamic range, signs, exact zeros, tiny values."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_groups):
        bufs = []
        for n in sizes:
            mag = torch.exp(torch.empty(n).uniform_(-25.0, 12.0, generator=g))
            sign = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
            x = (mag * sign).float()
            x[torch.rand(n, generator=g) < 0.1] = 0.0
            bufs.append(x)
        out.append(bufs)
    return out


def _accumulate(groups, order, sizes, chunk=1 << 24) -> list[torch.Tensor]:
    grads = [torch.zeros(n, dtype=torch.float32) for n in sizes]
    acc = OrderInvariantGradAccumulator(grads, chunk=chunk)
    for i in order:
        for buf, x in zip(grads, groups[i]):
            buf.add_(x)  # the backward writes into the (zeroed) scratch buffers
        acc.capture()
        assert all(not buf.any() for buf in grads)  # scratch zeroed after every capture
    acc.finalize()
    return grads


def _naive(groups, order, sizes) -> list[torch.Tensor]:
    grads = [torch.zeros(n, dtype=torch.float32) for n in sizes]
    for i in order:
        for buf, x in zip(grads, groups[i]):
            buf.add_(x)
    return grads


def _bits(ts):
    return [t.view(torch.int32).clone() for t in ts]


@pytest.mark.parametrize("seed", range(4))
def test_bitwise_identical_under_random_arrival_permutations(seed):
    sizes = (4099, 1000, 37)
    groups = _groups(24, sizes, seed)
    g = torch.Generator().manual_seed(1000 + seed)
    ref = _bits(_accumulate(groups, list(range(24)), sizes))
    for _ in range(12):
        perm = torch.randperm(24, generator=g).tolist()
        assert all(torch.equal(a, b) for a, b in zip(ref, _bits(_accumulate(groups, perm, sizes))))
    # chunking of the fp64 scratch does not change the bits either
    assert all(torch.equal(a, b) for a, b in zip(ref, _bits(_accumulate(groups, list(range(24)), sizes, chunk=97))))


def test_plain_fp32_accumulation_is_order_dependent():
    """The gate's negative: the same groups summed in fp32 in two arrival orders give different bits."""
    sizes = (4099,)
    groups = _groups(24, sizes, 7)
    a = _bits(_naive(groups, list(range(24)), sizes))
    b = _bits(_naive(groups, list(reversed(range(24))), sizes))
    assert not torch.equal(a[0], b[0])


def test_accuracy_against_fp64_sum():
    sizes = (5000,)
    groups = _groups(16, sizes, 3)
    got = _accumulate(groups, list(range(16)), sizes)[0].double()
    exact = sum(gr[0].double() for gr in groups)
    # fp32 rounding of the exact sum plus the 2^-31 per-capture rounding
    tol = exact.abs() * 2.0**-24 + 16 * 2.0**-31
    assert torch.all((got - exact).abs() <= tol)


def test_fixed_roundtrip_exact_for_values_on_the_grid():
    x = torch.tensor([1.0, -1.5, 3.0e5, -0.0078125, 2.0**-30, 123456.789, 0.0], dtype=torch.float32)
    q, bad = to_fixed(x)
    assert not bad.any()
    assert torch.equal(from_fixed(q).view(torch.int32), x.view(torch.int32))


def test_too_many_captures_raise():
    from nemo_rl.algorithms.exact_grad_accum import MAX_CAPTURES

    grads = [torch.zeros(4)]
    acc = OrderInvariantGradAccumulator(grads)
    for _ in range(MAX_CAPTURES + 1):
        acc.capture()
    with pytest.raises(RuntimeError, match="captures"):
        acc.finalize()


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), 2.0**24])
def test_nonfinite_or_out_of_range_raises(bad):
    grads = [torch.zeros(8)]
    acc = OrderInvariantGradAccumulator(grads)
    grads[0][3] = bad
    acc.capture()
    with pytest.raises(RuntimeError, match="non-finite"):
        acc.finalize()


def _run(cls, groups, order, sizes, check):
    grads = [torch.zeros(n, dtype=torch.float32) for n in sizes]
    acc = cls(grads, check_elems=check)
    for _ in range(2):  # reused across steps: reset re-zeroes the limbs, nothing is reallocated
        acc.reset()
        for i in order:
            for buf, x in zip(grads, groups[i]):
                buf.add_(x)
            acc.capture()
        acc.finalize()
    return grads, acc.order_check


def test_in_loop_order_check_passes_exact_and_fails_naive():
    from nemo_rl.algorithms.exact_grad_accum import NaiveArrivalOrderAccumulator

    sizes = (3000, 500)
    groups = _groups(16, sizes, 11)
    order = [3, 1, 4, 0, 5, 9, 2, 6, 15, 8, 7, 10, 14, 13, 12, 11]  # an "arrival" order
    g_exact, chk = _run(OrderInvariantGradAccumulator, groups, order, sizes, 1024)
    assert chk["mismatch"] == 0 and chk["checked"] == 1024 + 500 and chk["captures"] == 16
    assert chk["naive_order_diff"] > 0  # the check has power on this data
    g_naive, chk_n = _run(NaiveArrivalOrderAccumulator, groups, order, sizes, 1024)
    assert chk_n["mismatch"] > 0  # the negative control fails the same check
    # reuse across steps leaves the exact result unchanged
    assert all(torch.equal(a.view(torch.int32), b.view(torch.int32)) for a, b in zip(
        g_exact, _accumulate(groups, list(range(16)), sizes)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_triton_capture_equals_torch_capture(monkeypatch):
    """The fused Triton capture and the chunked torch path give the same int64 sums and bad counts."""
    from nemo_rl.algorithms.exact_grad_accum import capture_into

    g = torch.Generator().manual_seed(5)
    xs = [(torch.exp(torch.empty(100003).uniform_(-25.0, 17.5, generator=g))
           * torch.where(torch.rand(100003, generator=g) < 0.5, -1.0, 1.0)).float() for _ in range(5)]
    xs[2][7] = float("inf")
    xs[3][11] = float("nan")
    out = {}
    for path in ("0", "1"):
        monkeypatch.setenv("NEMO_RL_EXACT_GRAD_ACCUM_TORCH", path)
        acc = torch.zeros(100003, dtype=torch.int64, device="cuda")
        bad = torch.zeros((), dtype=torch.int64, device="cuda")
        for x in xs:
            buf = x.cuda()
            capture_into(buf, acc, bad, chunk=4099)
            assert not buf.any()
        out[path] = (acc.cpu(), int(bad))
    assert torch.equal(out["0"][0], out["1"][0]) and out["0"][1] == out["1"][1] > 2


def test_replay_compare_and_counts_untouched():
    """R32 item 1: a re-run of the first group compares equal (bitwise + digest) and adds nothing."""
    sizes = (2000, 300)
    groups = _groups(6, sizes, 21)
    grads = [torch.zeros(n) for n in sizes]
    acc = OrderInvariantGradAccumulator(grads, check_elems=512)
    acc.reset()
    for i in range(6):
        for b, x in zip(grads, groups[i]):
            b.add_(x)
        acc.capture()
    for b, x in zip(grads, groups[0]):  # replay the first group
        b.add_(x)
    acc.replay_compare()
    assert acc.replay["replay_mismatch"] == 0 and acc.replay["replay_digest_equal"] == 1.0
    assert acc.captures == 6 and not any(b.any() for b in grads)
    for b, x in zip(grads, groups[0]):  # a perturbed replay is seen
        b.add_(x * (1 + 1e-3))
    acc.replay_compare()
    assert acc.replay["replay_mismatch"] > 0 and 5e-4 < acc.replay["replay_rel_l2"] < 2e-3
    acc.finalize()
    assert all(torch.equal(a.view(torch.int32), b.view(torch.int32)) for a, b in zip(grads, _accumulate(groups, range(6), sizes)))


def test_accuracy_report_and_adam_sim():
    """R32 item 2: rel_l2 vs the plain fp32 path, the fp32 order floor, the zero-here count, the Adam-update rel_l2."""
    sizes = (5000,)
    groups = _groups(16, sizes, 8)
    grads = [torch.zeros(5000)]
    acc = OrderInvariantGradAccumulator(grads, check_elems=5000)
    acc.reset()
    for i in range(16):
        grads[0].add_(groups[i][0])
        acc.capture()
    acc.finalize()
    oc = acc.order_check
    assert 0 <= oc["grad_rel_l2_vs_fp32"] < 1e-6 and 0 <= oc["grad_rel_l2_fp32_order_floor"] < 1e-6
    assert oc["zero_here_nonzero_fp32"] >= 0
    sim = {}
    out = acc.adam_sim(1.0 / 4096, sim, 0.9, 0.999, 1e-8)
    assert set(out) == {"adam_rel_l2_vs_fp32", "adam_rel_l2_fp32_order_floor"} and sim["t"] == 1
    assert out["adam_rel_l2_vs_fp32"] < 1e-3


def test_acc_views_persistent_buffer():
    """R32 item 3: the sums can live in views of one caller-owned buffer (reset zeroes it; same bits)."""
    sizes = (1000, 77)
    groups = _groups(5, sizes, 2)
    buf = torch.full((sum(sizes) * 8,), 7, dtype=torch.uint8)  # dirty bytes (as after holding the moments)
    views, off = [], 0
    for n in sizes:
        views.append(buf[off : off + 8 * n].view(torch.int64))
        off += 8 * n
    grads = [torch.zeros(n) for n in sizes]
    acc = OrderInvariantGradAccumulator(grads, acc_views=views)
    acc.reset()
    for i in [4, 2, 0, 1, 3]:
        for b, x in zip(grads, groups[i]):
            b.add_(x)
        acc.capture()
    acc.finalize()
    ref = _accumulate(groups, range(5), sizes)
    assert all(torch.equal(a.view(torch.int32), b.view(torch.int32)) for a, b in zip(grads, ref))
    with pytest.raises(ValueError):
        OrderInvariantGradAccumulator(grads, acc_views=[views[0]])


def test_both_accumulators_share_the_constructor_signature():
    """The worker builds either class with the same keywords (BUGS #12: the negative control lacked acc_views)."""
    from nemo_rl.algorithms.exact_grad_accum import NaiveArrivalOrderAccumulator

    for cls in (OrderInvariantGradAccumulator, NaiveArrivalOrderAccumulator):
        acc = cls([torch.zeros(4)], check_elems=2, acc_views=None)
        acc.reset()
        acc.capture()
        acc.finalize()
