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
"""Training LM head whose logits are never materialized for a whole micro-batch.

Under the fp16-GEMM recipe the training LM head is the fp32 head
(``fp16_gemm.fp32_gemm``) and the loss only needs the target tokens'
log-probs (``zero_train_gen_mismatch._BatchInvariantLogprob``). Holding the
logits of a whole micro-batch costs ``tokens x vocab x 4 B`` for the forward
output, the same again for its gradient and half again for the backward's
bf16 copy: at 8 x 8192 tokens x 131k vocab that is ~85 GB per learner rank,
and it grows with tokens per micro-batch.

``DeferredHeadLogits`` stands in for the logits: a tensor of the logits' shape
and dtype with no storage that remembers the head input ``x`` ``[s, b, h]``
and weight. The view ops between the model and the loss (Megatron's
``transpose(0, 1).contiguous()``, ``.to(float32)``, sequence ``narrow`` /
slicing) are recorded; the log-prob op then calls ``target_logprobs``, which
runs ``FusedHeadLogprob`` over fixed sequence chunks:

- forward, per chunk: ``fp32_gemm`` of the chunk's rows (row-invariant: a
  row's logits do not depend on which rows share the GEMM) and the same
  log-softmax + gather as ``_gathered_target_logprobs``, so the log-probs are
  bitwise those of the materialized path;
- backward, per chunk: recompute the chunk's logits, form ``g (onehot -
  softmax)`` (``_gathered_logprob_grad``), then the head's dgrad / wgrad GEMMs
  exactly as ``Fp16Linear.backward`` forms them (bf16 operands, fp32
  accumulation), wgrad summed over chunks in chunk order.

Per rank: at most two ``[b, chunk, vocab]`` fp32 buffers at a time (chunk x
b <= ``CHUNK_TOKENS``; the logits and their log-softmax while it runs, 4 GiB
at 4096 x 131k), whatever the micro-batch's tokens; the backward's gradient
reuses the log-softmax storage. The head input is kept (``[b, s, h]``). The wgrad's chunk-order sum differs from the one-GEMM sum by
rounding only (gradients need not be bitwise the materialized path's; the
forward log-probs are). Chunk boundaries depend only on the sequence length
and the micro-batch's row count, never on other ranks or timing.

Any other op on a ``DeferredHeadLogits`` materializes the logits (through the
autograd-tracked head, so it stays correct, without the memory saving);
in-place ops raise, since they cannot act on the stand-in.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

# Tokens (rows x sequence positions) per logits chunk: bounds the fp32 chunk
# buffer to CHUNK_TOKENS x vocab x 4 B (2 GiB at 4096 x 131k).
CHUNK_TOKENS = 4096

_VIEW_FNS = None


def _view_fns():
    global _VIEW_FNS
    if _VIEW_FNS is None:
        T = torch.Tensor
        _VIEW_FNS = {
            "transpose": {T.transpose, torch.transpose, T.swapaxes, torch.swapaxes},
            "identity": {T.contiguous},
            "to": {T.to, T.float, T.type_as},
            "narrow": {T.narrow, torch.narrow},
            "getitem": {T.__getitem__},
        }
    return _VIEW_FNS


# Metadata queries answered by the wrapper itself (no data access).
_METADATA = {
    "__get__", "size", "dim", "numel", "stride", "is_contiguous", "element_size",
    "__len__", "nelement", "ndimension", "get_device", "__repr__", "__format__",
}


def _chunk_len(batch: int) -> int:
    return max(1, CHUNK_TOKENS // max(1, batch))


class FusedHeadLogprob(torch.autograd.Function):
    """Target log-probs of ``x @ weight.T`` (fp32 head) without whole-batch logits.

    ``x`` ``[b, s, h]``, ``weight`` ``[v, h]``, ``target`` ``[b, s]`` (token at
    each position, already shifted). Returns ``[b, s]`` fp32.
    """

    @staticmethod
    def forward(ctx, x, weight, target, chunk):  # noqa: D102
        from nemo_rl.models.megatron import fp16_gemm
        from nemo_rl.models.megatron.zero_train_gen_mismatch import (
            _gathered_target_logprobs,
        )

        def chunk_of(start: int, end: int) -> torch.Tensor:
            # The materialized head's forward: fp32 operand, fp32_gemm.
            return fp16_gemm.fp32_gemm(x[:, start:end].to(torch.float32), weight)

        out = _gathered_target_logprobs(chunk_of, target, chunk)
        ctx.save_for_backward(x, weight, target)
        ctx.chunk = chunk
        from nemo_rl.models.megatron import exact_nucleus

        ctx.nucleus_anchors = exact_nucleus.current_anchors()
        return out

    @staticmethod
    def backward(ctx, grad_out):  # noqa: D102
        from nemo_rl.models.megatron import fp16_gemm
        from nemo_rl.models.megatron.zero_train_gen_mismatch import (
            _gathered_logprob_grad,
            _seq_chunks,
        )

        x, weight, target = ctx.saved_tensors
        b, s, h = x.shape
        v = weight.shape[0]
        w16 = weight.to(torch.bfloat16)
        dx = torch.empty((b, s, h), dtype=x.dtype, device=x.device)
        dw = torch.zeros((v, h), dtype=torch.float32, device=x.device)
        from nemo_rl.models.megatron import exact_nucleus

        anchors_ctx = exact_nucleus.anchors_context(ctx.nucleus_anchors)
        anchors_ctx.__enter__()
        for start, end in _seq_chunks(s, ctx.chunk):
            xc = x[:, start:end].to(torch.float32)
            grad = _gathered_logprob_grad(
                lambda a, e, xc=xc: fp16_gemm.fp32_gemm(xc[:, a:e], weight),
                target[:, start:end],
                grad_out[:, start:end],
                vocab_start_index=0,
                vocab_end_index=v,
                chunk_size=end - start,
                seq_offset=start,
                # One chunk: its gradient comes back in the log-softmax output's
                # storage (no separate [b, chunk, v] fp32 buffer).
                grad_input=None,
            )
            # As Fp16Linear.backward: bf16 operands, fp32 accumulation.
            g2 = grad.reshape(-1, v).to(torch.bfloat16)
            del grad
            dx[:, start:end] = (
                fp16_gemm._mm(g2, w16, torch.float32).view(b, end - start, h).to(x.dtype)
            )
            dw += fp16_gemm._mm(g2.t().contiguous(), xc.reshape(-1, h).to(torch.bfloat16), torch.float32)
            del g2, xc
        anchors_ctx.__exit__(None, None, None)
        # Merge (ts-nano-scale-v11): exact top-p's anchors context closes first (the head
        # backward's nucleus masking is done), then learnfuse's fused untied-head wgrad.
        from nemo_rl.models.megatron import fused_learner_ops as flo

        if flo.wgrad_fusable((weight,), dw):  # untied head: dw straight into main_grad (bitwise)
            return dx, flo.accumulate_wgrad(dw, (weight,))[0], None, None
        return dx, dw.to(weight.dtype), None, None



# Full-logit materializations of the deferred LM head (silent_bug_sweep.md): each one
# costs the memory the deferred head exists to save, and one below autograd under
# grad would silently drop that term's gradient. Counted (``materialized_take``),
# warned once per op; THUNDERSYNC_STRICT_DEFERRED_HEAD=1 raises on a below-autograd
# materialization while grad is enabled.
_MATERIALIZED: dict[str, int] = {"function": 0, "dispatch": 0, "dispatch_grad": 0}
_WARNED: set = set()


def _note_materialized(func: Any, path: str, grad: bool, below_autograd: bool = False) -> None:
    import os
    import warnings

    _MATERIALIZED[path] += 1
    if below_autograd and grad:
        _MATERIALIZED["dispatch_grad"] += 1
        if os.environ.get("THUNDERSYNC_STRICT_DEFERRED_HEAD") == "1":
            raise RuntimeError(
                f"DeferredHeadLogits materialized below autograd by {func} with grad enabled: "
                "no gradient would flow through this use of the logits"
            )
    name = getattr(func, "__name__", str(func))
    if (path, name) not in _WARNED:
        _WARNED.add((path, name))
        warnings.warn(
            f"DeferredHeadLogits: {name} materializes the full logits ({path} path"
            + (", below autograd, grad enabled: no gradient" if below_autograd and grad else "")
            + ")"
        )


def materialized_take() -> dict[str, int]:
    out = dict(_MATERIALIZED)
    for k in _MATERIALIZED:
        _MATERIALIZED[k] = 0
    return out

class DeferredHeadLogits(torch.Tensor):
    """Stand-in for the training LM head's logits (see the module docstring).

    Logical value: ``fp32_gemm(x, weight)`` ``[s, b, v]`` with the recorded
    views applied: ``batch_first`` (transposed to ``[b, s, v]``) and a
    sequence slice ``seq`` (start, end).
    """

    @staticmethod
    def __new__(cls, x, weight, batch_first=False, seq=None):
        s, b = int(x.shape[0]), int(x.shape[1])
        start, end = (0, s) if seq is None else seq
        n = end - start
        shape = (b, n, weight.shape[0]) if batch_first else (n, b, weight.shape[0])
        out = torch.Tensor._make_wrapper_subclass(
            cls, shape, dtype=torch.float32, device=x.device, requires_grad=False
        )
        out._x, out._w, out._batch_first, out._seq = x, weight, batch_first, (start, end)
        return out

    def __repr__(self):  # noqa: D105
        return f"DeferredHeadLogits(shape={tuple(self.shape)}, batch_first={self._batch_first})"

    def _derive(self, batch_first=None, seq=None) -> "DeferredHeadLogits":
        return DeferredHeadLogits(
            self._x,
            self._w,
            self._batch_first if batch_first is None else batch_first,
            self._seq if seq is None else seq,
        )

    def _detached(self) -> "DeferredHeadLogits":
        return DeferredHeadLogits(
            self._x.detach(), self._w.detach(), self._batch_first, self._seq
        )

    def _seq_dim(self) -> int:
        return 1 if self._batch_first else 0

    def head_input(self) -> torch.Tensor:
        """``x`` restricted to the slice, ``[b, n, h]`` (autograd-tracked)."""
        start, end = self._seq
        return self._x[start:end].transpose(0, 1).contiguous()

    def row_input(self, b: int) -> torch.Tensor:
        """Head input of batch row ``b`` over the slice, ``[n, h]`` (no autograd)."""
        start, end = self._seq
        return self._x[start:end, b].detach()

    @property
    def head_weight(self) -> torch.Tensor:
        return self._w.detach()

    def materialize(self) -> torch.Tensor:
        """The logits through the autograd-tracked head (no memory saving)."""
        from nemo_rl.models.megatron import fp16_gemm

        start, end = self._seq
        logits = fp16_gemm.fp16_linear(
            self._x[start:end], self._w, fp16_gemm.COMM_NONE, None, fp16_gemm.FP32_HEAD
        )
        return logits.transpose(0, 1) if self._batch_first else logits

    def target_logprobs(self, target: torch.Tensor, chunk_size: Optional[int]) -> torch.Tensor:
        """Log-probs of ``target`` ``[b, n]`` under these logits (requires ``batch_first``)."""
        if not self._batch_first:
            raise ValueError("DeferredHeadLogits.target_logprobs expects [b, s, v] logits")
        x = self.head_input()
        chunk = chunk_size if chunk_size is not None else _chunk_len(int(x.shape[0]))
        return FusedHeadLogprob.apply(x, self._w, target, chunk)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):  # noqa: D105
        kwargs = kwargs or {}
        fns = _view_fns()
        self = args[0] if args and isinstance(args[0], DeferredHeadLogits) else None
        if self is not None and (
            func == torch.Tensor.detach or func == torch.Tensor.data.__get__
        ):
            # A detached stand-in (same logical values, no autograd history).
            return self._detached()
        if self is not None and getattr(func, "__name__", "") in _METADATA:
            # Shape / dtype / device queries: the wrapper's own metadata.
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)
        if self is not None:
            if func in fns["identity"]:
                return self
            if func in fns["to"]:
                target_dtype = kwargs.get("dtype")
                for a in args[1:]:
                    if isinstance(a, torch.dtype):
                        target_dtype = a
                    elif isinstance(a, torch.Tensor):
                        target_dtype = a.dtype
                if func is torch.Tensor.float or target_dtype in (None, torch.float32):
                    if all(not isinstance(a, torch.device) for a in args[1:]) and not kwargs.get(
                        "device"
                    ):
                        return self
            if func in fns["transpose"]:
                dims = sorted(int(d) % 3 for d in (args[1:3] if len(args) >= 3 else kwargs.values()))
                if dims == [0, 1]:
                    return self._derive(batch_first=not self._batch_first)
            if func in fns["narrow"]:
                dim, start, length = (list(args[1:]) + [kwargs.get(k) for k in ("dim", "start", "length")])[:3]
                if int(dim) % 3 == self._seq_dim():
                    s0 = self._seq[0]
                    return self._derive(seq=(s0 + int(start), s0 + int(start) + int(length)))
            if func in fns["getitem"]:
                idx = args[1]
                sl = _seq_slice(idx, self._seq_dim())
                if sl is not None:
                    a, b_ = sl.indices(self._seq[1] - self._seq[0])[:2]
                    s0 = self._seq[0]
                    return self._derive(seq=(s0 + a, s0 + b_))
        if getattr(func, "__name__", "").endswith("_") and self is not None:
            raise RuntimeError(
                f"in-place {func.__name__} on DeferredHeadLogits (the training LM head's "
                "logits are not materialized; see nemo_rl/models/megatron/deferred_head.py)"
            )
        # Anything else: materialize (correct, without the memory saving).
        _note_materialized(func, "function", torch.is_grad_enabled())

        def unwrap(a):
            return a.materialize() if isinstance(a, DeferredHeadLogits) else a

        args = tuple(unwrap(a) for a in args)
        kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # noqa: D105
        # Below autograd (ops reached without __torch_function__, e.g. property
        # getters such as ``.data``): the stand-in has requires_grad=False, so
        # its value is computed without autograd history.
        kwargs = kwargs or {}
        if func is torch.ops.aten.detach.default:
            return args[0]._detached()

        def unwrap(a):
            if isinstance(a, DeferredHeadLogits):
                # Below autograd: the value carries no gradient. A loss term that
                # reached the logits this way would get none, silently.
                _note_materialized(func, "dispatch", torch.is_grad_enabled(), below_autograd=True)
                with torch.no_grad():
                    return a._detached().materialize()
            return a

        return func(*(unwrap(a) for a in args), **{k: unwrap(v) for k, v in kwargs.items()})


def _seq_slice(idx: Any, seq_dim: int) -> Optional[slice]:
    """The sequence slice of an index that only slices the sequence dim (step 1), else None."""
    if not isinstance(idx, tuple):
        idx = (idx,)
    full = slice(None)
    if len(idx) > 3 or any(i is Ellipsis or i is None for i in idx):
        return None
    idx = idx + (full,) * (3 - len(idx))
    for d, i in enumerate(idx):
        if d == seq_dim:
            if not isinstance(i, slice) or i.step not in (None, 1):
                return None
        elif i != full:
            return None
    return idx[seq_dim]


def deferred_head_logits(x: torch.Tensor, weight: torch.Tensor) -> DeferredHeadLogits:
    """The stand-in for ``fp16_linear(x, weight, FP32_HEAD)`` (``x`` ``[s, b, h]``)."""
    return DeferredHeadLogits(x, weight)


class verification_logits:
    """Context: no-grad forwards return ``DeferredHeadLogits`` too (fp32 head, TP=1).

    Verification needs log-probs of the drafted tokens and a few full rows; the
    caller computes them per row in chunks of at most ``CHUNK_TOKENS`` positions
    (``thundersync_rl.engine_q.DeferredLogprobRows``), each chunk's logits by the
    same row-invariant ``fp32_gemm`` and row-wise log-softmax as the materialized
    path, so the values are bitwise those of a whole-batch head. Outside the fp16
    head recipe (or at TP > 1) the head is unchanged and logits are materialized.
    """

    def __enter__(self):
        from nemo_rl.models.megatron import fp16_gemm

        self._prev = fp16_gemm._STATE.get("deferred_head_nograd")
        fp16_gemm._STATE["deferred_head_nograd"] = True
        return self

    def __exit__(self, *exc):
        from nemo_rl.models.megatron import fp16_gemm

        fp16_gemm._STATE["deferred_head_nograd"] = self._prev
        return False
