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
"""Forward + backward CUDA graphs of a Megatron learner forward with activation recompute.

``torch.cuda.make_graphed_callables`` captures the backward with
``torch.autograd.grad``; Megatron's activation checkpoint (the learner's selective
MoE recompute, ``recompute_modules: [moe]``) raises "Checkpointing is not compatible
with .grad()" there, unless Megatron's own "graph capturing" flag is set, and with
that flag set during a *forward* the checkpoint stops recomputing (it would save
every MoE activation in the graph's pool).

``capture_fwd_bwd`` therefore captures the forward with the flag off (the
checkpoint records its recompute as usual) and the backward with the flag on (the
recompute runs inside the captured backward). Gradients w.r.t. parameters with a
Megatron ``main_grad`` are added into it inside the backward graph as they are produced
(post-accumulate hooks on stand-in leaves, under ``no_sync``), so a graph's pool holds
saved activations and one layer's transients, never a whole-model gradient set; the
others are handed to autograd as in eager mode.

Forward outputs, log-probs and loss are bitwise eager's; gradients are the same
per-parameter values summed into main_grad in a possibly different order (fp32
rounding level).
"""

from __future__ import annotations

import contextlib

import torch


@contextlib.contextmanager
def _swapped(module: torch.nn.Module, stand_ins: dict):
    """Module parameters -> stand-ins (attributes and helper objects such as Mamba's
    ``MambaContextParallel``, which keeps references to the live parameters), for the
    whole warm-up / capture, so that recompute inside the backward sees them too."""
    from torch.nn.utils.stateless import _reparametrize_module

    from nemo_rl.models.megatron.sticky_rounding import _captured_params_swapped

    # Helpers first: they are found by the identity of the live parameters, which
    # ``named_parameters`` no longer returns once the module is reparametrized.
    with _captured_params_swapped(module, stand_ins), _reparametrize_module(
        module, stand_ins, strict=False
    ):
        yield


class mcore_capture_flag:
    """Megatron's "graph capturing" flag for the duration of a backward capture."""

    def __enter__(self):
        from megatron.core.transformer import cuda_graphs

        cuda_graphs._set_capture_start()
        return self

    def __exit__(self, *exc):
        from megatron.core.transformer import cuda_graphs

        cuda_graphs._set_capture_end()
        return False


def capture_fwd_bwd(
    module: torch.nn.Module,
    *static_args: torch.Tensor,
    warmup: int = 1,
    pool=None,
    main_grad_in_graph: bool = False,
    stage_input=None,
    on_backward=None,
    shared_out=None,
    bwd_pool=None,
):
    """Forward and backward graphs of ``module(*static_args)`` as one autograd callable.

    Returns ``call()``: replays the forward and returns its (single tensor) output,
    whose backward replays the captured backward and returns every trainable
    parameter's gradient to autograd. The caller writes new inputs into
    ``static_args`` before each call; they must not require grad. ``pool``: a
    memory pool shared with other captures whose replays never interleave (saved
    activations live from a forward replay to its backward replay); default a
    private one. ``warmup=0``: no eager run here (the caller ran this shape eagerly
    before, or captures directly; nothing executes until the first replay, which keeps
    expert-parallel exchanges paired across ranks). Side effects of the warm-up /
    capture runs on persistent buffers are the caller's to undo.

    ``main_grad_in_graph``: parameters with a Megatron ``main_grad`` get
    ``main_grad += grad`` inside the backward graph as each gradient is produced (a
    post-accumulate hook on the stand-in leaf, what Megatron DDP's hook does under
    ``no_sync``), and return no gradient to autograd. A gradient lives only from its
    producing kernel to its add: the captured backward's pool holds one layer's
    weight-sized transients (e.g. an expert linear's fp32 wgrad), never the whole
    model's gradients (``torch.autograd.grad`` would return every gradient at once:
    a full bf16 gradient copy in every training graph's pool). Only valid while
    gradient sync is off (``no_sync``) and without TE's fused wgrad accumulation; the
    caller checks.

    ``shared_out(shape, dtype, name)``: optional provider of buffers shared by every
    graph (None: keep a private one). The forward output is copied into
    ``shared_out(.., 'train_out')`` and the output gradient read from
    ``shared_out(.., 'train_gout')``, so a capture keeps no per-shape output.

    ``bwd_pool``: capture the backward into this pool instead of ``pool`` (several
    in-flight graphs, e.g. 1F1B slots at PP > 1: per-slot ``pool`` holds what lives from
    a forward to its backward; the backward's transients come from one pool shared by
    every slot, since backward replays are serialized). Gradients returned to autograd
    are then copies (another slot's backward replay may reuse their blocks).

    ``stage_input`` (pipeline parallel, non-first stage): ``(buf, bind)``. ``buf`` is the
    static leaf the module reads as its stage input (``bind(t)`` makes the decoder read
    ``t``, as ``set_input_tensor`` does; called around the warm-up / capture); ``call(x)``
    then takes the received activation ``x``, copies it into ``buf`` before the forward
    replay, and its backward returns ``x``'s gradient (a copy), which the schedule sends
    to the previous stage. ``on_backward()``: called after each backward replay.
    """
    named = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
    params = tuple(p for _, p in named)
    # Gradients are taken with respect to stand-in leaves that share each parameter's
    # storage, not the parameters themselves: an eager backward (and Megatron DDP, which
    # keeps every parameter's AccumulateGrad node in ``grad_accs``) ties the parameter's
    # AccumulateGrad node to the default stream, and a backward through it on the capture
    # stream makes autograd synchronize with the default stream, which invalidates the
    # capture (measured: research/lgraphs/bwd_cap_min.py MODES=ddp).
    stand_ins = {}
    for name, p in named:
        q = torch.nn.Parameter(p.detach(), requires_grad=True)
        q.__dict__.update({k: v for k, v in p.__dict__.items() if not k.startswith("_")})
        stand_ins[name] = q
    leaves = tuple(stand_ins.values())
    in_graph = [main_grad_in_graph and getattr(p, "main_grad", None) is not None for p in params]
    stage_leaf = None
    if stage_input is not None:
        buf, bind = stage_input
        stage_leaf = buf.detach().requires_grad_(True)
        bind(stage_leaf)

    def fwd(*args):
        return module(*args)

    hooks_on = [False]

    def _hook_for(p):
        def hook(q):
            if hooks_on[0] and q.grad is not None:
                p.main_grad.add_(q.grad)
                q.grad = None

        return hook

    handles = [
        q.register_post_accumulate_grad_hook(_hook_for(p))
        for q, p, ig in zip(leaves, params, in_graph)
        if ig
    ]
    outer_leaves = tuple(q for q, ig in zip(leaves, in_graph) if not ig) + (
        (stage_leaf,) if stage_leaf is not None else ()
    )

    def grads_of(out, gout):
        # One backward into the leaves' .grad: the outer graph's gradients and an
        # activation-recompute region's own inner ``torch.autograd.backward`` (Megatron
        # checkpoint) both accumulate there. In-graph leaves go to main_grad from their
        # post-accumulate hooks as they are produced; the others are collected here.
        for q in leaves:
            q.grad = None
        if stage_leaf is not None:
            stage_leaf.grad = None
        torch.autograd.backward(out, gout, inputs=list(leaves) + ([stage_leaf] if stage_leaf is not None else []))
        res = []
        for q in outer_leaves:
            res.append(q.grad)
            q.grad = None
        return tuple(res)

    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st), _swapped(module, stand_ins):
        for _ in range(warmup):
            out = fwd(*static_args)
            with mcore_capture_flag():
                grads_of(out, torch.empty_like(out))
            del out
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    pool = torch.cuda.graph_pool_handle() if pool is None else pool
    fwd_graph, bwd_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    import os

    # thread_local: the learner actor runs other threads (Ray's event loop, hidden-row
    # prefetch, host copies) whose CUDA API calls would invalidate a "global" capture
    # (lgf2-x1 xovf: "operation failed due to a previous error during capture" in a
    # training forward capture of step 1; no-grad captures were thread_local already).
    mode = os.environ.get("NEMO_RL_GRAPH_CAPTURE_ERROR_MODE", "thread_local")
    try:
        with _swapped(module, stand_ins):
            with torch.cuda.graph(fwd_graph, pool=pool, capture_error_mode=mode):
                live_out = fwd(*static_args)
                static_out = None
                if shared_out is not None:
                    static_out = shared_out(tuple(live_out.shape), live_out.dtype, "train_out")
                if static_out is not None:
                    static_out.copy_(live_out)
                else:
                    static_out = live_out
            static_gout = None
            if shared_out is not None:
                static_gout = shared_out(tuple(live_out.shape), live_out.dtype, "train_gout")
            if static_gout is None:
                static_gout = torch.empty_like(live_out)
            hooks_on[0] = True
            with mcore_capture_flag(), torch.cuda.graph(
                bwd_graph, pool=pool if bwd_pool is None else bwd_pool, capture_error_mode=mode
            ):
                static_grads = grads_of(live_out, static_gout)
            hooks_on[0] = False
            del live_out
    finally:
        hooks_on[0] = False
        for h in handles:
            h.remove()
    static_stage_grad = None
    if stage_leaf is not None:
        static_stage_grad = static_grads[-1]
        if static_stage_grad is None:
            static_stage_grad = torch.zeros_like(stage_leaf)
        static_grads = static_grads[:-1]
    # Tensors the module kept from the capture (e.g. a deferred LM head's weight, used
    # outside the graph) must be the live parameters, not the stand-ins.
    if hasattr(module, "remap_params"):
        real = {id(q): p for q, p in zip(leaves, params)}
        module.remap_params(lambda t: real.get(id(t), t))
    # Only parameters whose gradient goes back through autograd are inputs of the
    # autograd node (the others' grad hooks must not run: their gradient is already in
    # main_grad); a dummy leaf keeps the output differentiable when none is left.
    outer = [(p, g) for p, g in zip((p for p, ig in zip(params, in_graph) if not ig), static_grads)]
    dummy = torch.zeros((), device=static_out.device, requires_grad=True)
    inputs = tuple(p for p, _ in outer) + (dummy,)
    outer_grads = tuple(g for _, g in outer) + (None,)
    # Without Megatron's DDP hook (which moves .grad into main_grad at once) the next
    # replay would rewrite a .grad that aliases these buffers: hand autograd copies.
    clone = [bwd_pool is not None or not hasattr(p, "main_grad") for p, _ in outer] + [False]
    done_box = [None]

    class _Graphed(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, *_params):  # noqa: D102
            if stage_leaf is not None:
                stage_leaf.detach().copy_(x)
            fwd_graph.replay()
            return static_out.detach()

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad):  # noqa: D102
            if grad.data_ptr() != static_gout.data_ptr():
                static_gout.copy_(grad)
            bwd_graph.replay()
            gx = None if stage_leaf is None else static_stage_grad.clone()
            if done_box[0] is not None:
                done_box[0][0] = True
            if on_backward is not None:  # the caller's slot is free again (its buffers were consumed)
                on_backward()
            return (gx,) + tuple(
                None if g is None else (g.clone() if c else g.detach())
                for g, c in zip(outer_grads, clone)
            )

    def call(x=None):
        if stage_leaf is not None and x is None:
            raise ValueError("capture_fwd_bwd: a stage-input graph needs the received activation")
        return _Graphed.apply(x if stage_leaf is not None else None, *inputs)

    def set_done(box):
        """``box[0] = True`` after the next backward replay (the caller's open-window flag)."""
        done_box[0] = box

    call.set_done = set_done
    return call
