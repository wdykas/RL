# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""stash_head / stash_weights: the new version reuses a dropped version's buffers (no clone next to it)."""

import torch

from thundersync_rl.worker import _restash


def test_reuses_dropped_buffer_bitwise():
    stash = {1: [torch.zeros(4, 3)], 2: [torch.ones(4, 3)]}
    keep_ptr = stash[1][0].data_ptr()
    src = [torch.randn(4, 3)]
    out = _restash(stash, 3, [2, 3], src)
    assert sorted(stash) == [2]  # 1 dropped
    assert out[0].data_ptr() == keep_ptr  # its buffer reused
    assert torch.equal(out[0], src[0]) and out[0].data_ptr() != src[0].data_ptr()
    src[0].add_(1.0)
    assert not torch.equal(out[0], src[0])  # a copy, not an alias


def test_clone_when_nothing_dropped_or_shape_differs():
    stash = {1: [torch.zeros(2)]}
    out = _restash(stash, 2, [1, 2], [torch.ones(2)])
    assert sorted(stash) == [1] and torch.equal(out[0], torch.ones(2))
    stash = {1: [torch.zeros(3)]}
    out = _restash(stash, 2, [2], [torch.ones(2)])
    assert stash == {} and out[0].shape == (2,)


def test_restash_same_version_overwrites_in_place():
    stash = {5: [torch.zeros(2)]}
    p = stash[5][0].data_ptr()
    out = _restash(stash, 5, [5], [torch.ones(2)])
    assert out[0].data_ptr() == p and torch.equal(out[0], torch.ones(2))
