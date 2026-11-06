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
"""Lookahead drafts (X5, ``THUNDERSYNC_LOOKAHEAD_DRAFT_ALPHA``, opt-in): drafting lanes hold theta~ = theta_k + alpha D.

D = theta_k - theta_{k-1} is the last update of the learner's weights. Every drafter refit (a cohort lane that only
drafts: lookahead-2 early drafts, the lookahead-1 drafting lane) sends, per element of every refit tensor,

    theta~ = dtype(fp32(theta_k) + alpha * (fp32(theta_k) - fp32(theta_{k-1})))

instead of theta_k (``LookaheadSend``, the native MCore refit's source-side ``ReshardTransform``). Lanes that decode
target rows at theta_k (fresh rows) are refit with theta_k unchanged. alpha = 0 sends theta_k bitwise.

Exactness (research/verify0attr/X5_LOOKAHEAD_DRAFTS.md): drafts are verified against p_{k+h} with their logged q, so
the sampling law is unchanged. Under sticky routing the trained policy is F(theta_{k+h}; A(theta~)), with anchors from
theta~: a deterministic function of past iterates, alpha and the prefix. Its gradient is exact as before. theta~ must
be the same on every rank (elementwise, same arithmetic) and recomputable after a resume (the theta_{k-1} snapshot
is saved with the checkpoint).

The learner keeps theta_{k-1} of its refit tensors in pinned host memory (``PrevSnapshot``, double-buffered): after
every optimizer step the new live weights are copied out on a side stream, and the next step's optimizer write joins
that copy first. The weight sentinel and the drafting-head stash use the same theta~ (``extrapolate``).
Per rank: 2 x the refit tensors' bytes of pinned host memory, one tensor's prev on the device at a time, no
collectives (the transform runs inside the refit's own sends).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

ENV = "THUNDERSYNC_LOOKAHEAD_DRAFT_ALPHA"


def alphas() -> Optional[tuple[float, ...]]:
    """The configured alpha schedule (``None``: off). One value: every version; a comma list ``a0,a1,...``: version k
    drafts with ``a_{k mod n}`` (an alpha sweep at matched phase in one run, to fit the KL parabola in alpha and read
    alpha* off its vertex). ``0`` is on with theta~ = theta_k (the matched control arm)."""
    v = os.environ.get(ENV, "").strip().lower()
    if v in ("", "off", "none"):
        return None
    return tuple(float(x) for x in v.split(","))


def alpha() -> Optional[float]:
    """The first alpha of the schedule (``None``: off)."""
    a = alphas()
    return None if a is None else a[0]


def alpha_at(schedule: tuple[float, ...], version: int) -> float:
    return float(schedule[int(version) % len(schedule)])


@torch.no_grad()
def extrapolate(cur: torch.Tensor, prev: torch.Tensor, a: float) -> torch.Tensor:
    """theta~ = cur.dtype(fp32(cur) + a * (fp32(cur) - fp32(prev))), elementwise (any slice of a tensor gives the same
    elements as the whole tensor)."""
    c = cur.float()
    return (c + a * (c - prev.to(device=cur.device, dtype=torch.float32))).to(cur.dtype)


def _key(t: torch.Tensor) -> tuple:
    return (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), t.dtype)


class PrevSnapshot:
    """theta_{k-1} of the learner's refit tensors in pinned host memory (double buffer).

    ``capture(named)`` starts copying the live tensors into the spare buffer (side stream, ordered after the work
    queued now); ``commit()`` (before the next parameter write) waits for it and makes it the previous version.
    ``prev(name)`` / ``prev_for(tensor)``: the previous version of a refit tensor (host), or None.
    """

    def __init__(self) -> None:
        self.bufs: list[dict[str, torch.Tensor]] = [{}, {}]
        self.keys: list[dict[tuple, str]] = [{}, {}]
        self.cur = 0  # index of the committed previous version
        self.pending: Optional[torch.cuda.Event] = None
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.committed = False

    @torch.no_grad()
    def capture(self, named: dict[str, torch.Tensor]) -> None:
        if self.pending is not None:
            raise RuntimeError("lookahead snapshot: capture while the previous one is pending")
        i = 1 - self.cur
        spare, keys = self.bufs[i], {}
        if self.stream is not None:
            ev = torch.cuda.Event()
            ev.record()
            self.stream.wait_event(ev)
        with (torch.cuda.stream(self.stream) if self.stream is not None else _Null()):
            for name, t in named.items():
                if not t.is_floating_point():
                    continue
                h = spare.get(name)
                if h is None or h.shape != t.shape or h.dtype != t.dtype:
                    h = spare[name] = torch.empty(
                        t.shape, dtype=t.dtype, device="cpu", pin_memory=torch.cuda.is_available()
                    )
                h.copy_(t.detach(), non_blocking=True)
                keys[_key(t)] = name
                if t.is_contiguous():
                    keys[("storage", t.untyped_storage().data_ptr())] = (name, t.storage_offset(), t.numel())
        self.keys[i] = keys
        if self.stream is not None:
            self.pending = torch.cuda.Event()
            self.pending.record(self.stream)
        else:
            self.pending = _Done()

    def commit(self) -> None:
        """Wait for the pending capture and make it the previous version."""
        if self.pending is None:
            return
        self.pending.synchronize()
        self.pending = None
        self.cur = 1 - self.cur
        self.committed = True

    def capture_now(self, named: dict[str, torch.Tensor]) -> None:
        """Capture and commit synchronously (start of a run: prev = theta_0, so D = 0 at version 0)."""
        self.capture(named)
        self.commit()

    def prev(self, name: Optional[str]) -> Optional[torch.Tensor]:
        if not self.committed or name is None:
            return None
        return self.bufs[self.cur].get(name)

    def prev_for(self, t: torch.Tensor) -> Optional[torch.Tensor]:
        """The previous version of ``t``: a refit tensor itself, or a view into a contiguous refit tensor's storage
        (e.g. one expert of a grouped weight), as the same view of the host copy."""
        keys = self.keys[self.cur]
        p = self.prev(keys.get(_key(t)))
        if p is not None or not self.committed:
            return p
        base = keys.get(("storage", t.untyped_storage().data_ptr()))
        if base is None:
            return None
        name, off, n = base
        host = self.bufs[self.cur].get(name)
        rel = t.storage_offset() - off
        if host is None or rel < 0 or rel + t.numel() > n:
            return None
        return host.reshape(-1).as_strided(tuple(t.shape), tuple(t.stride()), rel)

    # --- checkpoint / resume ------------------------------------------------------------------------------------
    def state(self) -> dict[str, torch.Tensor]:
        return dict(self.bufs[self.cur]) if self.committed else {}

    def load_state(self, state: dict[str, torch.Tensor], named: dict[str, torch.Tensor]) -> None:
        """Restore the previous version saved with a checkpoint; ``named``: the live refit tensors (keys)."""
        pin = torch.cuda.is_available()
        self.bufs[self.cur] = {k: (v.pin_memory() if pin else v.clone()) for k, v in state.items()}
        keys = {}
        for n, t in named.items():
            if n in state:
                keys[_key(t)] = n
                if t.is_contiguous():
                    keys[("storage", t.untyped_storage().data_ptr())] = (n, t.storage_offset(), t.numel())
        self.keys[self.cur] = keys
        self.committed = bool(state)

    def digest(self) -> str:
        """Bitwise digest of the committed previous version (resume tests)."""
        import hashlib

        from thundersync_rl.sentinel import tensor_digest

        h = hashlib.blake2b(digest_size=8)
        if self.committed:
            for name in sorted(self.bufs[self.cur]):
                h.update(repr((name, tensor_digest(self.bufs[self.cur][name]))).encode())
        return h.hexdigest()


class _Done:
    def synchronize(self) -> None:
        return None


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def make_send_transform(snapshot: PrevSnapshot, a: float):
    """The native MCore refit's source-side transform: send theta~ for every refit tensor with a previous version.

    Same wire format as the default send (one tensor of the slice's shape and dtype), so the receiver (default
    writeback or its fp16-held transform) is unchanged. One tensor's previous version is on the device at a time.
    """
    from megatron.core.resharding.transforms import ReshardTransform

    class LookaheadSend(ReshardTransform):
        def __init__(self) -> None:
            self._dev: tuple[Optional[str], Optional[torch.Tensor]] = (None, None)
            self.sent = 0

        def should_transform(self, param_name: str) -> bool:
            return snapshot.prev(param_name) is not None

        def prepare_send(self, param_name, src_slice, src_param):
            name, dev = self._dev
            if name != param_name:
                dev = snapshot.prev(param_name).to(src_param.device, non_blocking=True)
                self._dev = (param_name, dev)
            self.sent += 1
            return [extrapolate(src_param.detach()[src_slice], dev[src_slice], a).contiguous()]

        def prepare_recv(self, param_name, dst_slice):
            raise RuntimeError("LookaheadSend is a source-side transform")

        def finalize_recv(self, param_name, dst_slice, recv_buffers):
            raise RuntimeError("LookaheadSend is a source-side transform")

    return LookaheadSend()
