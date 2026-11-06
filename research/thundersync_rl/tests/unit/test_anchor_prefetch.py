"""Verify-batch anchor prefetch: the anchors built ahead are the ones the forward would build itself.

The verifier calls ``_anchor_batch(anchors, len(rows), self._padded_width(rows))`` in a background thread and passes
the result as ``routed=``; ``_training_forward_rows`` must then use it unchanged and not rebuild it, and with no
``routed`` it must build exactly that call (same width) inline.
"""

import pytest
import torch
from thundersync_rl import worker as W


class _Stop(Exception):
    pass


def _worker(monkeypatch, calls):
    seen = {}

    def stop(data, _tok):
        seen["data"] = data
        raise _Stop

    monkeypatch.setattr(W, "attach_media_token_validity_mask", stop)
    w = object.__new__(W.ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)
    w.cfg = {"make_sequence_length_divisible_by": 8}
    w.media_placeholder_token_id = None

    def anchor_batch(anchors, batch, width, starts=None):
        calls.append((list(anchors) if anchors else anchors, batch, width, starts))
        return torch.full((batch, width, 2, 1), 3, dtype=torch.int32)

    w._anchor_batch = anchor_batch
    return w, seen


def _rows():
    return [torch.arange(1, 12), torch.arange(1, 20), torch.arange(1, 5)]


def test_padded_width_is_the_forward_width(monkeypatch):
    w, _ = _worker(monkeypatch, [])
    assert w._padded_width([torch.arange(9)]) == 16
    assert w._padded_width([torch.arange(8)]) == 8
    assert w._padded_width(_rows()) == 24


def test_inline_build_uses_the_prefetch_call(monkeypatch):
    calls = []
    w, seen = _worker(monkeypatch, calls)
    rows, anchors = _rows(), ["a", "b", "c"]
    with pytest.raises(_Stop):
        w._training_forward_rows(rows, anchors=anchors)
    # The prefetch thread's call: (anchors, len(rows), _padded_width(rows)), no starts.
    assert calls == [(anchors, 3, w._padded_width(rows), None)]
    assert seen["data"]["input_ids"].shape == (3, w._padded_width(rows))
    assert torch.equal(seen["data"]["routed_experts"], torch.full((3, 24, 2, 1), 3, dtype=torch.int32))


def test_prefetched_anchors_are_used_unchanged(monkeypatch):
    calls = []
    w, seen = _worker(monkeypatch, calls)
    routed = torch.randint(0, 7, (3, 24, 2, 1), dtype=torch.int32)
    with pytest.raises(_Stop):
        w._training_forward_rows(_rows(), anchors=["a", "b", "c"], routed=routed)
    assert calls == []
    assert seen["data"]["routed_experts"] is routed
