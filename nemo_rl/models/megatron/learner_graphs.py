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
"""CUDA graphs of whole learner forwards (opt-in, ``NEMO_RL_LEARNER_GRAPHS=1``).

A learner call of the batch-invariant fp16 recipe is host bound: full
Nemotron-3-Nano launches ~4k kernels per forward and ~9k per forward + backward.
Once the MoE layers keep their routing counts on the GPU (``device_dispatch``,
``NEMO_RL_MOE_DEVICE_DISPATCH=1``, required for MoE models), a forward has no host
sync and is replayed from a CUDA graph. Prefix-state cache forwards (pooled K/V pages and
Mamba slots addressed by static device metadata) are graphed the same way.

No-grad forwards (verification, log-probs; ThunderSync's critical path):

- token buckets: the sequence length is padded up to a bucket (grid step
  2^floor(log2 S) / ``NEMO_RL_LEARNER_GRAPHS_BUCKET_DIV``, default 8: at most
  12.5% padding) when B x bucket fits the MoE dispatch token cap. A forward's
  values at the real positions do not depend on later positions (causal Mamba /
  attention, per-token MoE, row-invariant GEMMs: batch invariance), so the caller
  gets exactly its unpadded result: the output is cut back to S rows;
- sticky routing: each router's anchors are copied into a static buffer
  (padding positions -1 = plain routing, as the caller's own padding);
- sticky-routing check statistics (CHECKS=1) are graph outputs: the captured
  forward adds each router call's (kept anchors, differs-from-plain) counts into a
  per-graph device accumulator; the host appends one aggregated record per replay
  with the real (unpadded) token count, so ``sticky_routing.pop_stats`` returns the
  same integers as the eager forwards' per-call records;
- log-prob forwards whose output is the full logits (> ``NEMO_RL_LEARNER_GRAPHS_MAX_OUT_MB``,
  e.g. 20480 tokens x 131k vocab x 4 B = 10.7 GB): the graph ends at the LM head's
  input (the deferred head, fp32 head at TP 1) and the logits are formed after the
  replay by the same ``fp16_linear`` on the same rows (row-invariant GEMM: bitwise the
  in-graph head), so no static logits buffer is kept;
- head-input outputs (deferred head, verification) are graphed up to a regime-aware cap
  (``_head_out_bytes``): max(1 GiB, T_cap x hidden x 4 B, bounded by 4% of the GPU) unless
  ``NEMO_RL_LEARNER_GRAPHS_MAX_OUT_MB`` is set; T_cap is sized by the run from its row length,
  so one 128k Nano row (1.43 GB head input) is graphed by default;
- outputs are copied out on return (the caller may keep them while graphs run again).

Forward + backward (training, ``NEMO_RL_LEARNER_GRAPHS_TRAIN=1``): one graph pair
per micro-batch bucket (``graph_capture.capture_fwd_bwd``: Megatron's MoE activation
recompute runs inside the captured backward; gradients go into ``main_grad`` inside
the graph, which is valid while Megatron DDP's gradient sync is off, i.e. under the
learner's ``no_sync``; otherwise eager). Training buckets
(``NEMO_RL_LEARNER_GRAPHS_TRAIN_BUCKET_DIV``, default 16: at most 6.25% padding; 0 =
exact shapes): the padded positions are causal successors of the real ones with a
zero loss mask, so the real tokens' forward values, log-probs and the loss are
bitwise the unpadded eager call's (gate, measured); gradients are exact up to the
summation order of token reductions (padding rows contribute exact zeros; reductions
whose split depends on the row count, e.g. RMSNorm's weight-gradient column sum, round
differently: fp32 rounding level, measured as relative error vs eager, research/lgfull;
gradients need not be bitwise, user rule 10-19).

Memory (budgeted, one pool; ``NEMO_RL_LEARNER_GRAPHS_SHARED_POOL=1``, default):

- every graph of a model (verification, log-prob and training) is captured into ONE
  private memory pool. Graph replays never interleave (ordered learner calls;
  verification runs between micro-batches, after a backward), so a capture may reuse
  every block another capture freed: the pool holds max over graphs of one graph's
  transient, not the sum, and no release / recapture is needed when the learner
  switches between verification and training;
- static inputs (ids, positions, sticky anchors) and outputs (head input; training
  output gradient) are shared buffers sized for the dispatch token cap T_cap, viewed
  per shape: per graph only the CUDA graph itself is kept. Training graphs are
  "open" from the forward replay to the backward replay; any other learner call in
  that window runs eagerly (it would overwrite the shared inputs / the pool's
  activations), counted as ``why_*_train_open``;
- per rank: static + max(pool, eager transient of a call that is not graphed) +
  outside-PyTorch, the pool being max(training micro-batch transient at its bucket,
  verification transient). An eager call that needs more than the free memory releases
  the pool first (``release_<cause>`` counters): an eager training forward always
  (its activations are of the pool's size), an eager no-grad forward when
  driver-free + cached-free < ``NEMO_RL_LEARNER_GRAPHS_NOGRAD_EAGER_GB`` (default 12,
  measured no-grad transient at T_cap ~10 GiB). With every call graphed nothing is
  released (steady state);
- first sighting of a shape: eager (``NEMO_RL_LEARNER_GRAPHS_DIRECT=0``, default). Direct
  capture at the first sighting (=1) fails in the loop: torch.compile'd functions
  (Megatron's ``squared_relu`` jit_fuser in the shared experts) recompile for a new
  shape inside the capture ("operation failed due to a previous error during capture",
  lgf-x1 @51cd9554). An eager training first sighting releases the pool first (budget
  rule above), so the number of releases is bounded by the number of distinct training
  buckets (warm-up only; none once every bucket was seen).
- ``NEMO_RL_LEARNER_GRAPHS_SHARED_POOL=0``: the earlier layout (separate no-grad and
  training pools, the no-grad graphs released at the first grad-enabled call, v11oom).

Pipeline parallel > 1 (ts-scaleup 14bg, merged onto this core in ts-final): a non-first
stage reads its input from the decoder's ``input_tensor`` (``set_input_tensor`` by the
schedule), not from the call's arguments, so each graph owns a static stage-input buffer:
the received tensor is copied into it before the eager first run / capture / replay and
the decoder reads the buffer (``_StageInput``); its shape and dtype are part of the graph
key. Exact lengths at PP > 1 (``_no_buckets``). Training graphs at PP > 1: 1F1B runs
several micro-batches' forwards before the first backward, so each in-flight micro-batch
gets its own entry ("slot", key includes the slot index) with private static inputs and
outputs and a private pool for what lives from its forward to its backward; backward
transients come from one backward pool shared by the slots (backward replays are
serialized; ``capture_fwd_bwd(bwd_pool=)``). The stage input is a leaf of the captured
backward (``capture_fwd_bwd(stage_input=...)``); the replayed backward returns the
received activation's gradient, which the schedule sends to the previous stage. A slot is
``busy`` from its forward replay to its backward replay; the shared-pool budget release
never runs while a slot is busy. PP 1 keeps the shared-pool "open window" rule above.

Eligible calls: no attention mask / packed sequences / multimodal inputs, a
prefix-state cache forward only when ``prefix_state_cache.graph_key`` allows it
(pooled K/V pages and Mamba slots addressed by static device metadata;
else ``why_fwd_prefix_cache``), no sticky-rounding hooks / statistics, router replay only under sticky routing, keyword
options among ``fp32_output``. Others run eagerly with the same kernels; every eager
call is counted by reason (``why_fwd_<reason>`` / ``why_train_<reason>``) and by
caller feature (``feature``; ``feat_<name>_graph`` / ``feat_<name>_eager``).

MoE capacity (expert parallel > 1): if a device-dispatch exchange overflowed its
receive buffer (``device_dispatch.overflowed``, the same on every EP rank), the
call is rerun eagerly on Megatron's dispatcher, which stays selected through the
call's backward (until the next call).

"""

from __future__ import annotations

import collections
import contextlib
import os
import weakref
from typing import Any, Optional

import torch

_ALLOWED_KW = {"fp32_output"}


def enabled() -> bool:
    """Opt-in switch (env, read in every learner worker)."""
    return os.environ.get("NEMO_RL_LEARNER_GRAPHS", "0") == "1"


def _max_graphs() -> int:
    return int(os.environ.get("NEMO_RL_LEARNER_GRAPHS_MAX", "64"))


def _bucket_div() -> int:
    return int(os.environ.get("NEMO_RL_LEARNER_GRAPHS_BUCKET_DIV", "8"))


def _train_bucket_div() -> int:
    return int(os.environ.get("NEMO_RL_LEARNER_GRAPHS_TRAIN_BUCKET_DIV", "16"))


def _train_graphs() -> bool:
    return os.environ.get("NEMO_RL_LEARNER_GRAPHS_TRAIN", "0") == "1"


def _shared_pool() -> bool:
    return os.environ.get("NEMO_RL_LEARNER_GRAPHS_SHARED_POOL", "1") == "1"


def _direct_capture() -> bool:
    return os.environ.get("NEMO_RL_LEARNER_GRAPHS_DIRECT", "0") == "1"


def _nograd_eager_bytes() -> int:
    return int(float(os.environ.get("NEMO_RL_LEARNER_GRAPHS_NOGRAD_EAGER_GB", "12")) * 2**30)


def _max_out_bytes() -> int:
    return int(float(os.environ.get("NEMO_RL_LEARNER_GRAPHS_MAX_OUT_MB", "1024")) * 2**20)


def _row_token_cap() -> int:
    """Most tokens one learner forward returns: the dispatch token cap T_cap, which the run sizes
    from its configured row length (run_ab_arm.sh: T_cap = max(MBS x (PROMPT_CAP + MAXLEN), 18432),
    whole rows, not divided by CP, so it also covers a BI CP verification forward that gathers
    whole rows on every CP rank)."""
    from nemo_rl.models.megatron import device_dispatch

    try:
        return int(device_dispatch.token_cap())
    except Exception:  # noqa: BLE001  (CPU tests)
        return 0


def _head_out_bytes(width: int) -> int:
    """Graph output cap for a head-input output (deferred LM head; verification / log-prob).

    ``NEMO_RL_LEARNER_GRAPHS_MAX_OUT_MB`` when set; else regime aware: max(1 GiB, the largest
    head input a call can return = ``_row_token_cap()`` x hidden x 4 B), bounded by
    ``NEMO_RL_LEARNER_GRAPHS_HEAD_OUT_FRAC`` (default 0.04) of this GPU's memory. Nano (hidden
    2688, 10.75 KB/token): 8k (T_cap 20480) 0.22 GB -> 1 GiB (unchanged); 128k rows (T_cap >=
    2048 + 131072) >= 1.43 GB -> a one-piece 128k verification is graphed without a manual env.
    """
    if os.environ.get("NEMO_RL_LEARNER_GRAPHS_MAX_OUT_MB"):
        return _max_out_bytes()
    need = _row_token_cap() * max(int(width), 1) * 4
    try:
        bound = int(float(os.environ.get("NEMO_RL_LEARNER_GRAPHS_HEAD_OUT_FRAC", "0.04"))
                    * torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)
    except Exception:  # noqa: BLE001
        bound = need
    return max(2**30, min(need, bound))


def bucket_length(s: int, div: Optional[int] = None) -> int:
    """Smallest bucket >= s: multiples of max(128, 2^floor(log2(s - 1)) / div)."""
    div = _bucket_div() if div is None else div
    if div <= 0 or s <= 128:
        return s
    step = max(128, (1 << ((s - 1).bit_length() - 1)) // div)
    return -(-s // step) * step


# -- callers' features (coverage per feature) ----------------------------------
_FEATURE: list[str] = ["other"]


@contextlib.contextmanager
def feature(name: str):
    """Tag the learner forwards inside this context with ``name`` (coverage counters)."""
    prev = _FEATURE[0]
    _FEATURE[0] = name
    try:
        yield
    finally:
        _FEATURE[0] = prev


class _Wrapped(torch.nn.Module):
    """``model(input_ids, position_ids, attention_mask=None, **kw)`` as a function of two tensors."""

    def __init__(self, model: Any, kw: dict, force_deferred: bool = False):
        super().__init__()
        self.model = model
        self.kw = kw
        self.force_deferred = force_deferred
        self.deferred = None  # (weight, batch_first, seq) when the head is deferred

    def forward(self, ids, pos):  # noqa: D102
        from nemo_rl.models.megatron.deferred_head import DeferredHeadLogits

        ctx = contextlib.nullcontext()
        if self.force_deferred:
            from nemo_rl.models.megatron.deferred_head import verification_logits

            ctx = verification_logits()
        with ctx:
            out = self.model(input_ids=ids, position_ids=pos, attention_mask=None, **self.kw)
        out = out[0] if isinstance(out, tuple) else out
        if isinstance(out, DeferredHeadLogits):
            # Graph up to the head input; the stand-in is rebuilt around the replayed
            # input (its fused log-prob / backward run outside the graph).
            self.deferred = (out._w, out._batch_first, out._seq)
            return out._x
        self.deferred = None
        return out

    def remap_params(self, fn):
        """Replace tensors kept from a capture (graph_capture stand-ins) by ``fn(tensor)``."""
        if self.deferred is not None:
            w, batch_first, seq = self.deferred
            self.deferred = (fn(w), batch_first, seq)

    def rebuild(self, x):
        """The model's output from the graphed tensor (identity unless the head is deferred)."""
        if self.deferred is None:
            return x
        from nemo_rl.models.megatron.deferred_head import DeferredHeadLogits

        w, batch_first, seq = self.deferred
        return DeferredHeadLogits(x, w, batch_first, seq)


def _cut(out: Any, wrapped: _Wrapped, b: int, s: int, s_pad: int, copy: bool, materialize: bool = False):
    """The unpadded model output (first s positions) of a forward run at s_pad.

    ``materialize``: the graph ended at a deferred head that the caller did not ask
    for (large-logit log-prob forward): return the logits the in-model head forms
    (``DeferredHeadLogits.materialize``: the same fp32 head GEMM on the same rows).
    """
    if wrapped.deferred is not None:
        # Head input x is [s, b, h] (sequence first); seq slices are within s.
        x = out[:s] if s_pad != s else out
        if copy and not materialize:
            x = x.clone()
        w, batch_first, seq = wrapped.deferred
        from nemo_rl.models.megatron.deferred_head import DeferredHeadLogits

        if seq is not None and tuple(seq) == (0, s_pad):
            seq = None
        elif s_pad != s and seq is not None:
            raise RuntimeError(f"learner_graphs: padded forward with a sliced head {seq}")
        d = DeferredHeadLogits(x, w, batch_first, seq)
        if materialize:
            return d.materialize().contiguous()
        return d
    if s_pad != s:
        if out.shape[:2] == (b, s_pad):
            out = out[:, :s]
        elif out.shape[:2] == (s_pad, b):
            out = out[:s]
        else:  # pragma: no cover (guarded by _no_grad)
            raise RuntimeError(f"learner_graphs: cannot cut output {tuple(out.shape)} to {s}")
    return out.clone() if copy else out


def _replay_instances(model: Any) -> list:
    from nemo_rl.models.megatron.router_replay import _router_replay_instances_for_model

    return [inst for inst, _ in _router_replay_instances_for_model(model)]


def _replay_state(model: Any) -> tuple[Optional[tuple], Optional[str]]:
    """(state, reason): state () without router replay, ('sticky', k, n[, 'stats']) under a
    sticky replay forward (with check statistics: 'stats'), None (ineligible, reason given)."""
    try:
        from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
    except ImportError:
        return (), None
    insts = getattr(RouterReplay, "global_router_replay_instances", None) or []
    actions = {getattr(i, "router_replay_action", None) for i in insts}
    if actions <= {None}:
        return (), None
    from nemo_rl.models.megatron import sticky_routing

    if actions != {RouterReplayAction.REPLAY_FORWARD}:
        return None, "replay_action"
    if sticky_routing.get_delta() is None:
        return None, "replay_plain"
    mine = _replay_instances(model)
    if not mine or any(i.target_topk_idx is None for i in mine):
        return None, "replay_no_target"
    k = int(mine[0].target_topk_idx.shape[-1])
    if sticky_routing._STATE.get("stats") is not None:
        return ("sticky", k, len(mine), "stats"), None
    return ("sticky", k, len(mine)), None


def _hook_state() -> tuple[Optional[tuple], Optional[str]]:
    """(key part, reason): Python-side switches that change what a forward computes.

    None: a hook a graph cannot replay is installed (sticky rounding's expert hook or
    statistics) or the active prefix-state cache forward is not graph-safe
    (``prefix_state_cache.graph_key``).
    """
    from nemo_rl.models.megatron import fp16_gemm, fp16_moe, prefix_state_cache, sticky_rounding

    if fp16_moe._HOOKS.get("fc2_input") is not None:
        return None, "sround_hook"
    if sticky_rounding._STATE.get("stats") is not None:
        return None, "sround_stats"
    # Prefix-state cache forwards: the cache says whether its active forward
    # is graph-safe (pooled state addressed by static device index tensors) and what of it
    # belongs in the key (pool addresses / sizes); () when no cache forward is active.
    prefix = prefix_state_cache.graph_key()
    if prefix is None:
        return None, "prefix_cache"
    st = fp16_gemm._STATE
    from nemo_rl.models.megatron import bi_context_parallel as _cpx

    return prefix + (
        # BI CP: whole-row (verification) vs CP-local outputs are different graphs (the wrapper's gather).
        _cpx._GATHER_OUT[0] > 0,
        bool(st.get("deferred_head")),
        bool(st.get("deferred_head_nograd")),
        id(st.get("site_cast")),
        id(st.get("moe_preprocess")),
        st.get("head"),
        st.get("down_scale"),
    ), None


def _causal_mask_ok(model: Any, mask: Optional[torch.Tensor], ids: torch.Tensor) -> bool:
    """No mask, or the unpacked path's causal [1|B, 1, S, S] mask under flash attention.

    Megatron's unpacked micro-batches carry ``get_ltor_masks_and_position_ids``'s causal
    mask; flash attention (causal mask type) does not read it and Mamba layers ignore
    it, so graphed / padded forwards run without it (checked bitwise by verify_probe).
    """
    if mask is None:
        return True
    if mask.dim() != 4 or mask.shape[1] != 1 or mask.shape[0] not in (1, ids.shape[0]):
        return False
    if tuple(mask.shape[-2:]) != (ids.shape[1], ids.shape[1]):
        return False
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.utils import get_model_config

    return get_model_config(model).attention_backend == AttnBackend.flash


def _train_max_tokens() -> int:
    """Largest training micro-batch (rows x bucket length) captured. Shared pool: the
    dispatch token cap (every micro-batch); separate pools: 8192 (the earlier default)."""
    v = os.environ.get("NEMO_RL_LEARNER_GRAPHS_TRAIN_MAX_TOKENS")
    if v:
        return int(v)
    if _shared_pool():
        from nemo_rl.models.megatron import device_dispatch

        return device_dispatch.token_cap()
    return 8192


def _grad_sync_off(model: Any) -> bool:
    """Megatron DDP inside ``no_sync`` (its grad hook only adds into main_grad) and no
    TE fused wgrad accumulation: the backward graph may add into main_grad itself."""
    groups = list(getattr(model, "bucket_groups", []) or []) + list(
        getattr(model, "expert_parallel_bucket_groups", []) or []
    )
    if not groups or any(getattr(g, "is_last_microbatch", True) for g in groups):
        return False
    # fused_learner_ops' eager wgrad -> main_grad fusion sets the flag too (it is off inside
    # captures, where graph_capture adds into main_grad); only TE's fused accumulation counts.
    return not any(
        getattr(p, "grad_added_to_main_grad", False) and not getattr(p, "_learnfuse_wgrad", False)
        for p in model.parameters()
    )


def _buffer_ptrs(model: Any) -> tuple:
    """Addresses of Megatron DDP's contiguous parameter and gradient buffers.

    Captured kernels read parameters and (training graphs) add into ``main_grad`` by
    address. ``offload_to_cpu`` / ``reload_from_cpu`` (``move_model``) resize these
    storages, which reallocates them: a graph captured before would then read / write
    freed memory, so the addresses are part of a graph's key. A few buffers per model
    (per rank cost: one int per buffer and call).
    """
    out = []
    for name in ("buffers", "expert_parallel_buffers"):
        bufs = getattr(model, name, None)
        if not isinstance(bufs, (list, tuple)):  # nn.Module.buffers is a method
            continue
        for buf in bufs:
            for attr in ("param_data", "grad_data"):
                t = getattr(buf, attr, None)
                out.append(t.data_ptr() if isinstance(t, torch.Tensor) else 0)
    return tuple(out)


def _sp_size(model: Any) -> int:
    """TP size when sequence parallel is on (token rows split over TP), else 1.

    Under SP a padded length would move token rows between TP ranks, so no token buckets
    there, and sticky anchors are per-rank rows (ts-scaleup dedd0d0d: Super-L8 TP2 loop
    "size of tensor a (22) must match ... (11)" when the anchors assumed every row)."""
    from megatron.core import parallel_state as ps
    from megatron.core.utils import get_model_config

    try:
        tp = ps.get_tensor_model_parallel_world_size()
        return tp if tp > 1 and getattr(get_model_config(model), "sequence_parallel", False) else 1
    except (AssertionError, RuntimeError, AttributeError, ValueError):  # pragma: no cover
        return 1


def _no_buckets(model: Any) -> bool:
    """Exact lengths only: sequence parallel (rows move between TP ranks), BI context
    parallel (the chunk layout), pipeline parallel (received activations' length)."""
    if _sp_size(model) > 1:
        return True
    try:
        from nemo_rl.models.megatron import bi_context_parallel

        if bi_context_parallel.size() > 1:
            return True
    except (ImportError, AttributeError):
        pass
    from megatron.core import parallel_state as ps

    try:
        return ps.get_pipeline_model_parallel_world_size() > 1
    except (AssertionError, RuntimeError):  # pragma: no cover
        return False


def _pp_size() -> int:
    from megatron.core import parallel_state as ps

    try:
        return ps.get_pipeline_model_parallel_world_size()
    except (AssertionError, RuntimeError):  # pragma: no cover
        return 1


def _cp_size() -> int:
    """BI context parallel size (bi_context_parallel.py); 1 without it."""
    try:
        from nemo_rl.models.megatron import bi_context_parallel

        return bi_context_parallel.size()
    except (ImportError, AttributeError):  # pragma: no cover
        return 1


def key_slot(entry: Any) -> int:
    return int(getattr(entry, "slot", 0))


def _decoder(model: Any) -> Any:
    """The model's decoder block (``TransformerBlock`` / ``HybridStack``) under DDP / Float16 wrappers."""
    m = model
    for _ in range(4):
        if hasattr(m, "decoder"):
            return m.decoder
        m = getattr(m, "module", None)
        if m is None:
            return None
    return None


def _stage_input(model: Any) -> Optional[torch.Tensor]:
    """The received activation of a non-first pipeline stage (None on the first stage / PP 1)."""
    dec = _decoder(model)
    t = getattr(dec, "input_tensor", None) if dec is not None else None
    return t if isinstance(t, torch.Tensor) else None


class _StageInput:
    """Route a non-first stage's received input through a static buffer for one forward."""

    def __init__(self, model: Any, buf: Optional[torch.Tensor]):
        self.dec = _decoder(model) if buf is not None else None
        self.buf = buf

    def __enter__(self):
        if self.dec is None:
            return self
        self.saved = self.dec.input_tensor
        self.buf.copy_(self.saved)
        self.dec.input_tensor = self.buf
        return self

    def __exit__(self, *exc):
        if self.dec is not None:
            self.dec.input_tensor = self.saved
        return False


def _ep_size() -> int:
    from megatron.core import parallel_state as ps

    try:
        return ps.get_expert_model_parallel_world_size()
    except (AssertionError, RuntimeError):  # pragma: no cover
        return 1


def _model_width(model: Any) -> int:
    try:
        from megatron.core.utils import get_model_config

        return int(get_model_config(model).hidden_size)
    except Exception:  # noqa: BLE001  (CPU tests / non-Megatron models)
        return 0


def _local_vocab(model: Any) -> int:
    """Rows of the LM head weight on this rank (vocab / TP), 0 without one."""
    v = _VOCAB.get(id(model))
    if v is not None:
        return v
    _VOCAB[id(model)] = 0
    for m in model.modules():
        head = getattr(m, "output_layer", None)
        w = getattr(head, "weight", None) if head is not None else None
        if isinstance(w, torch.Tensor):
            _VOCAB[id(model)] = int(w.shape[0])
            break
    return _VOCAB[id(model)]


_VOCAB: dict[int, int] = {}


def _logits_plan(model: Any, b: int, s_pad: int):
    """How a no-grad forward of [b, s_pad] returns its output: None (as is), 'materialize'
    (full logits above the output cap: graph to the deferred head's input, logits formed
    after the replay), ('shared_logits', bytes) (TP > 1: logits in one shared buffer) or
    'too_large' (eager)."""
    from nemo_rl.models.megatron import fp16_gemm

    st = fp16_gemm._STATE
    try:
        from megatron.core import parallel_state as ps

        tp = ps.get_tensor_model_parallel_world_size()
    except Exception:  # noqa: BLE001
        tp = 1
    if st.get("deferred_head_nograd") and st.get("deferred_head") and tp == 1:
        return None  # verification at TP 1: the model already returns the head input
    v_local = _local_vocab(model)
    vocab = v_local * tp
    if not vocab or b * s_pad * (vocab // max(tp, 1)) * 4 <= _max_out_bytes():
        return None
    deferrable = (
        st.get("deferred_head")
        and st.get("head") == fp16_gemm.FP32_HEAD
        and tp == 1
        and st.get("site_cast") is None
    )
    if deferrable:
        return "materialize"
    # TP > 1 (vocab-parallel head, no deferred head yet): one shared [T_cap, V/TP] fp32
    # logits buffer for every graph, within NEMO_RL_LEARNER_GRAPHS_LOGITS_GB.
    from nemo_rl.models.megatron import device_dispatch

    nbytes = device_dispatch.token_cap() * (vocab // max(tp, 1)) * 4
    if _shared_pool() and nbytes <= float(os.environ.get("NEMO_RL_LEARNER_GRAPHS_LOGITS_GB", "8")) * 2**30:
        return ("shared_logits", nbytes)
    return "too_large"


class _IO:
    """Static input / output buffers shared by every graph of one model.

    Sized for the MoE dispatch token cap T_cap (the largest call the learner makes):
    ids / positions 2 x T_cap int64, one [T_cap, k] int64 anchor table per router,
    the head input [T_cap, hidden] fp32 and its gradient. Full Nano (T_cap 20480,
    23 routers x top-6, hidden 2688): 0.33 MB + 22.6 MB + 2 x 220 MB per rank, once,
    instead of per graph. Graphs read / write views ``[:tokens]``; replays never
    interleave (training graphs are open forward -> backward and block others).
    """

    def __init__(self, cap: int, width: int):
        self.cap, self.width = int(cap), int(width)
        dev = torch.cuda.current_device()
        self.dev = dev
        self.ids = torch.zeros(self.cap, dtype=torch.long, device=dev)
        self.pos = torch.zeros(self.cap, dtype=torch.long, device=dev)
        self.anchor_tabs: dict[tuple[int, int], list[torch.Tensor]] = {}
        self.outs: dict[str, torch.Tensor] = {}

    def fits(self, tokens: int) -> bool:
        return tokens <= self.cap

    def anchors(self, n: int, k: int) -> list[torch.Tensor]:
        tabs = self.anchor_tabs.get((n, k))
        if tabs is None:
            tabs = self.anchor_tabs[(n, k)] = [
                torch.full((self.cap, k), -1, dtype=torch.long, device=self.dev) for _ in range(n)
            ]
        return tabs

    def prepare(self, name: str, nbytes: int) -> None:
        """Allocate shared buffer ``name`` (bytes) outside any capture (a buffer first
        allocated inside a capture would live in the graph pool)."""
        buf = self.outs.get(name)
        if buf is None or buf.numel() < nbytes:
            self.outs[name] = torch.empty(int(nbytes), dtype=torch.uint8, device=self.dev)

    def out(self, name: str, shape: tuple, dtype: torch.dtype) -> Optional[torch.Tensor]:
        """A view of shared output buffer ``name`` (None: not prepared / does not fit: the
        caller keeps a private one)."""
        numel = 1
        for d in shape:
            numel *= int(d)
        esz = torch.empty((), dtype=dtype).element_size()
        buf = self.outs.get(name)
        if buf is None or numel * esz > buf.numel():
            return None
        return buf[: numel * esz].view(dtype).view(shape)


class _FwdEntry:
    """One captured shape: static input views, then (after capture) its graph."""

    def __init__(self, b, s_pad, n_replay, k, io: Optional[_IO] = None, sp: int = 1):
        dev = torch.cuda.current_device()
        n = b * s_pad
        # Sequence parallel: each rank's routers see only its own token rows (n / TP).
        self.sp = sp
        if io is not None and io.fits(n):
            self.ids = io.ids[:n].view(b, s_pad)
            self.pos = io.pos[:n].view(b, s_pad)
            self.anchors = [t[: n // sp] for t in io.anchors(n_replay, k)] if n_replay else []
        else:
            self.ids = torch.zeros((b, s_pad), dtype=torch.long, device=dev)
            self.pos = torch.zeros((b, s_pad), dtype=torch.long, device=dev)
            # Sticky anchors per router, [s_pad * b, k] token-major (s major, b minor) as
            # Megatron's flattened [s, b] tokens; padding positions stay -1 (plain routing).
            self.anchors = [
                torch.full((s_pad * b // sp, k), -1, dtype=torch.long, device=dev) for _ in range(n_replay)
            ]
        self.graph = None
        self.exchanges = False
        self.out = None
        self.wrapped = None
        self.capturable = True
        self.materialize = False  # graph ends at a forced deferred head: logits formed after replay
        self.logits_bytes = 0  # > 0: full logits kept in the shared logits buffer (TP > 1)
        self.acc = None  # sticky-routing check statistics accumulator (device int64 [2])
        self.router_calls = 0  # sticky top-k calls per forward (counted at capture)
        self.stage_in: Optional[torch.Tensor] = None  # PP > 1, non-first stage
        self.busy = False  # PP > 1 training slot: forward replayed, backward not yet
        self.slot = 0

    def load(self, ids, pos, insts, s):
        b = ids.shape[0]
        s_pad = self.ids.shape[1]
        self.ids[:, :s].copy_(ids)
        self.pos[:, :s].copy_(pos)
        if s < s_pad:
            # Padded positions: token 0 (a fixed id: shared buffers must not carry another
            # call's tokens), positions continuing past the row (causal: no padded value
            # reaches a real position; their gradients are exactly zero).
            self.ids[:, s:].zero_()
            ext = torch.arange(1, s_pad - s + 1, device=pos.device)
            self.pos[:, s:].copy_(pos[:, -1:] + ext)
        for buf, inst in zip(self.anchors, insts):
            src = inst.target_topk_idx.to(buf.device, torch.long)
            if self.sp > 1:  # this rank's SP rows, unpadded (s == s_pad under SP)
                buf.copy_(src.view(buf.shape))
                continue
            view = buf.view(-1, b, buf.shape[-1])
            view[:s].copy_(src.view(s, b, -1))
            if s < s_pad:
                view[s:].fill_(-1)


class _Swap:
    """Point the routers' replay targets at a graph's static anchor buffers for one forward."""

    def __init__(self, insts, bufs):
        self.insts, self.bufs = insts, bufs

    def __enter__(self):
        self.saved = [i.target_topk_idx for i in self.insts]
        for i, b in zip(self.insts, self.bufs):
            i.target_topk_idx = b
        return self

    def __exit__(self, *exc):
        for i, t in zip(self.insts, self.saved):
            i.target_topk_idx = t
        return False


@contextlib.contextmanager
def _stats_fixup(real_tokens: int, padded_tokens: int):
    """Eager padded forward under sticky check statistics: count real tokens only.

    Padded positions carry anchor -1 (kept False, differs-from-plain False), so only the
    token count of each record needs the unpadded value."""
    from nemo_rl.models.megatron import sticky_routing

    stats = sticky_routing._STATE.get("stats")
    n0 = len(stats) if stats is not None else 0
    try:
        yield
    finally:
        if stats is not None and real_tokens != padded_tokens:
            for i in range(n0, len(stats)):
                t, kept, diff = stats[i]
                if t == padded_tokens:
                    stats[i] = (real_tokens, kept, diff)


@contextlib.contextmanager
def _stats_capture(entry: _FwdEntry):
    """Sticky check statistics as graph outputs: route each router call's counts into
    ``entry.acc`` (zeroed inside the graph) and count the calls."""
    from nemo_rl.models.megatron import sticky_routing

    if sticky_routing._STATE.get("stats") is None:
        yield
        return
    if entry.acc is None:
        entry.acc = torch.zeros(2, dtype=torch.long, device=torch.cuda.current_device())
    entry.acc.zero_()
    sticky_routing._STATE["stats_dev"] = entry.acc
    sticky_routing._STATE["stats_dev_calls"] = 0
    try:
        yield
    finally:
        entry.router_calls = int(sticky_routing._STATE.pop("stats_dev_calls", 0))
        sticky_routing._STATE.pop("stats_dev", None)


def _stats_after_replay(entry: _FwdEntry, real_tokens: int) -> None:
    from nemo_rl.models.megatron import sticky_routing

    stats = sticky_routing._STATE.get("stats")
    if stats is None or entry.acc is None:
        return
    a = entry.acc.clone()
    stats.append((real_tokens * entry.router_calls, a[0], a[1]))


class LearnerGraphs:
    """Per-model cache of captured forwards (see module docstring)."""

    def __init__(self, model: Any):
        self.model = model
        self.seen: collections.Counter = collections.Counter()
        self.fwd: "collections.OrderedDict[tuple, _FwdEntry]" = collections.OrderedDict()
        self.train: "collections.OrderedDict[tuple, _FwdEntry]" = collections.OrderedDict()
        self.pool = None
        self.train_pool = None
        self.mempool = None  # shared mode: graph_pool_handle of every graph
        self.pool_reserved = 0  # bytes reserved by captures since the pool was created
        self.params: Optional[list] = None
        self.has_moe: Optional[bool] = None
        self._dd_ok: Optional[bool] = None
        self.stats = collections.Counter()
        self.calls = 0
        self.warm = {"fwd": False, "train": False}
        self.io: Optional[_IO] = None
        self._open_ref = None  # weakref to an open training graph's output
        self._open_done = [True]
        self.slot_bwd_pool = None  # PP > 1: backward pool shared by the in-flight training slots

    # -- bookkeeping ----------------------------------------------------------
    def _why(self, kind: str, reason: str) -> None:
        self.stats["eager_ineligible" if kind == "fwd" else "eager_train"] += 1
        self.stats[f"why_{kind}_{reason}"] += 1

    def _feat(self, graphed: bool) -> None:
        self.stats[f"feat_{_FEATURE[0]}_{'graph' if graphed else 'eager'}"] += 1

    def _io(self) -> Optional[_IO]:
        if not _shared_pool():
            return None
        if self.io is None:
            from nemo_rl.models.megatron import device_dispatch

            self.io = _IO(device_dispatch.token_cap(), _model_width(self.model))
        return self.io

    def _pool_handle(self, kind: str):
        if _shared_pool():
            # A graph_pool_handle (freed with the last graph that uses it), not a
            # torch.cuda.MemPool: dropping a MemPool while graph-owned blocks are freed later
            # trips CUDACachingAllocator's "owner_PrivatePool->cudaMalloc_count > 0" assert in
            # the next capture's empty_cache (lgf-ab @cdd20499, full Nano, after a release).
            if self.mempool is None:
                self.mempool = torch.cuda.graph_pool_handle()
                self.pool_reserved = 0
            return self.mempool
        if kind == "train":
            if self.train_pool is None:
                self.train_pool = torch.cuda.graph_pool_handle()
            return self.train_pool
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        return self.pool

    def pool_bytes(self) -> tuple[int, int]:
        """(reserved, allocated) bytes of the shared graph pool: reserved = the sum of
        reserved-memory growth over the captures since the pool was created; allocated reported
        as 0 (unknown), so the eager cached-free estimate reserved - allocated - (res - alloc)
        is a lower bound (conservative budget)."""
        return self.pool_reserved, 0

    @contextlib.contextmanager
    def _measure_pool(self):
        """Count the reserved-memory growth of one capture against the shared pool (a capture's
        torch.cuda.graph empties the eager cache first; so do we, to measure from there)."""
        if not _shared_pool():
            yield
            return
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        r0 = torch.cuda.memory_reserved()
        try:
            yield
        finally:
            self.pool_reserved += max(0, torch.cuda.memory_reserved() - r0)

    def is_open(self) -> bool:
        """A training graph replayed its forward and its backward has not run yet."""
        if self._open_ref is None or self._open_done[0]:
            return False
        if self._open_ref() is None:  # output dropped without a backward
            self._open_ref = None
            return False
        return True

    def _mark_open(self, out: torch.Tensor) -> None:
        done = [False]
        self._open_done = done
        self._open_ref = weakref.ref(out)
        return done

    def slots_busy(self) -> bool:
        """PP > 1: an in-flight training slot replayed its forward, its backward has not run."""
        return any(e.busy for e in self.train.values())

    def release_fwd(self) -> int:
        """Drop every captured no-grad graph and its outputs (static inputs and ``seen``
        stay: the next sighting of a shape captures again without an eager run).
        Returns the number of graphs dropped; the pool's memory goes back to the device
        at the caller's ``empty_cache``."""
        n = 0
        for entry in self.fwd.values():
            if entry.graph is not None:
                n += 1
            entry.graph, entry.out, entry.wrapped = None, None, None
        if n:
            self.pool = None  # a fresh pool for the next captures
            self.stats["fwd_released"] += n
        return n

    def release_all(self, cause: str) -> int:
        """Shared pool: drop every graph (no-grad and training) and the pool itself.
        Not while a training graph is open (its activations live in the pool) or a PP > 1
        training slot is in flight (its graph and private pool must survive to its backward)."""
        if self.is_open() or self.slots_busy():
            return 0
        n = sum(e.graph is not None for e in self.fwd.values()) + sum(
            e.graph is not None for e in self.train.values()
        )
        for entry in list(self.fwd.values()) + list(self.train.values()):
            entry.graph, entry.out, entry.wrapped = None, None, None
        if self.mempool is not None:
            self.mempool = None
            n = max(n, 1)
        if n:
            # A training graph's autograd.Function class (graph_capture._Graphed) sits in a
            # reference cycle (every class does): its graphs, and so the pool's segments, are
            # freed by the cyclic collector only (lgf2-full: 75 GiB still in the pool after the
            # release -> OOM at the next eager first sighting).
            import gc

            gc.collect()
            self.stats[f"release_{cause}"] += 1
            self.stats["graphs_released"] += n
        return n

    def _budget_release(self, grad: bool) -> None:
        """Before an eager call: free the pool when the call may not fit next to it."""
        if not _shared_pool() or self.mempool is None or self.is_open() or self.slots_busy():
            return
        if grad:
            # An eager training forward's activations are of the pool's own size.
            if self.release_all("eager_train"):
                _empty_cache()
            return
        free, _total = torch.cuda.mem_get_info()
        p_res, p_alloc = self.pool_bytes()
        cached_free = max(0, torch.cuda.memory_reserved() - torch.cuda.memory_allocated() - (p_res - p_alloc))
        if free + cached_free < _nograd_eager_bytes():
            if self.release_all("eager_fwd"):
                _empty_cache()

    # -- eligibility -------------------------------------------------------
    def _common_key(self, input_ids, position_ids, attention_mask, kw) -> tuple[Optional[tuple], Optional[str]]:
        from megatron.core import parallel_state as ps

        if position_ids is None:
            return None, "no_positions"
        if not _causal_mask_ok(self.model, attention_mask, input_ids):
            return None, "mask"
        if set(kw) - _ALLOWED_KW:
            return None, "kwargs_" + "_".join(sorted(set(kw) - _ALLOWED_KW))
        stage = ()
        if ps.get_pipeline_model_parallel_world_size() != 1:
            t = _stage_input(self.model)
            if t is not None:
                stage = (tuple(t.shape), t.dtype)
            elif _decoder(self.model) is None:
                return None, "pp"
        if input_ids.dim() != 2 or position_ids.shape != input_ids.shape or not input_ids.is_cuda:
            return None, "input_layout"
        state, why = _hook_state()
        if state is None:
            return None, why
        if self.has_moe is None:
            from megatron.core.transformer.moe.moe_layer import MoELayer

            self.has_moe = any(isinstance(m, MoELayer) for m in self.model.modules())
        if self.has_moe:
            from nemo_rl.models.megatron import device_dispatch

            if self._dd_ok is None or not device_dispatch.graph_safe():
                ok, why = device_dispatch.model_graph_safe(self.model)
                self._dd_ok = ok or None  # re-checked until it holds (workspaces appear lazily)
                if not ok:
                    return None, why  # Megatron's dispatcher syncs the host
        if self.params is None:
            self.params = [p for p in self.model.parameters()]
        return (
            bool(self.model.training),
            tuple(sorted(kw.items())),
            state,
            # Captured kernels read parameters by address: a swapped parameter
            # (e.g. a weight stash installed into .data) needs its own graph.
            tuple(p.data_ptr() for p in self.params[:: max(1, len(self.params) // 8)]),
            _buffer_ptrs(self.model),
            stage,
        ), None

    # -- eager with the capacity guard ----------------------------------------
    def _eager(self, input_ids, position_ids, attention_mask, kw):
        from nemo_rl.models.megatron import device_dispatch

        grad = torch.is_grad_enabled()
        self._budget_release(grad)
        snap = None
        if grad and _ep_size() > 1 and device_dispatch.enabled():
            snap = _snapshot(self.model)
        out = self.model(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, **kw
        )
        if device_dispatch.overflowed():
            out = self._megatron_rerun(snap, input_ids, position_ids, attention_mask, kw)
        return out

    def _megatron_rerun(self, snap, input_ids, position_ids, attention_mask, kw):
        """The call again on Megatron's dispatcher (capacity overflow; every EP rank does it)."""
        from nemo_rl.models.megatron import device_dispatch

        self.stats["capacity_rerun"] += 1
        print(f"[learner_graphs] MoE dispatch capacity overflow: rerun on Megatron's dispatcher "
              f"(shape {tuple(input_ids.shape)}, reruns {self.stats['capacity_rerun']})", flush=True)
        if snap is not None:
            _restore(snap)
        self._budget_release(torch.is_grad_enabled())
        # Stays selected through this call's backward (MoE recompute); the next call
        # switches back.
        device_dispatch.set_active(False)
        return self.model(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, **kw
        )

    def _entry(self, table, key, b, s_pad, insts, replay, private: bool = False):
        entry = table.get(key)
        if entry is None:
            while len(table) >= _max_graphs():
                # never evict an in-flight PP slot (its backward replay is still to come)
                victim = next((k for k, e in table.items() if not e.busy), None)
                if victim is None:
                    break
                table.pop(victim)
                self.stats["evicted"] += 1
            entry = table[key] = _FwdEntry(
                b, s_pad, len(insts), replay[1] if replay else 1, None if private else self._io(),
                _sp_size(self.model),
            )
            t = _stage_input(self.model)
            if t is not None:
                entry.stage_in = torch.empty(t.shape, dtype=t.dtype, device=t.device)
        table.move_to_end(key)
        return entry

    # -- no-grad forwards ----------------------------------------------------
    def _no_grad(self, common, input_ids, position_ids, attention_mask, kw):
        from nemo_rl.models.megatron import device_dispatch

        replay, why = _replay_state(self.model)
        if replay is None:
            self._why("fwd", why)
            return self._eager(input_ids, position_ids, attention_mask, kw)
        b, s = input_ids.shape
        s_pad = bucket_length(s)
        if s_pad != s and (b * s_pad > device_dispatch.token_cap() or b == s_pad or _no_buckets(self.model)):
            s_pad = s
        key = (common, False, b, s_pad, replay)
        insts = _replay_instances(self.model) if replay else []
        entry = self._entry(self.fwd, key, b, s_pad, insts, replay)
        if entry.graph is None and entry.capturable and not entry.materialize:
            plan = _logits_plan(self.model, b, s_pad)
            if plan == "materialize":
                entry.materialize = True
            elif plan == "too_large":
                entry.capturable = False
            elif isinstance(plan, tuple):
                entry.logits_bytes = plan[1]
        if not entry.capturable:
            self._why("fwd", "out_too_large")
            return self._eager(input_ids, position_ids, attention_mask, kw)
        entry.load(input_ids, position_ids, insts, s)
        direct = _direct_capture() and self.warm["fwd"]
        if entry.graph is None and not direct and self.seen[key] == 0:
            # First sighting: eager at the padded shape.
            self.seen[key] += 1
            self.stats["eager_first"] += 1
            self._budget_release(False)
            wrapped = _Wrapped(self.model, dict(kw))
            with _Swap(insts, entry.anchors), _stats_fixup(b * s, b * s_pad), _StageInput(self.model, entry.stage_in):
                out = wrapped(entry.ids, entry.pos)
            if device_dispatch.overflowed():
                return self._megatron_rerun(None, input_ids, position_ids, attention_mask, kw)
            self.warm["fwd"] = True
            if wrapped.deferred is None and out.numel() * out.element_size() > _max_out_bytes() and not entry.materialize and not entry.logits_bytes:
                entry.capturable = False  # unforeseen large output: this shape stays eager
            return _cut(out, wrapped, b, s, s_pad, copy=False)
        if entry.graph is None:
            try:
                with self._measure_pool():
                    self._capture_fwd(entry, kw, insts)
            except _NotCapturable as exc:
                entry.capturable = False
                self._why("fwd", str(exc))
                return self._eager(input_ids, position_ids, attention_mask, kw)
            self.stats["captured"] += 1
        if entry.acc is not None:
            entry.acc.zero_()
        with _StageInput(self.model, entry.stage_in):  # the received activation -> graph input
            entry.graph.replay()
        self.stats["replay_fwd"] += 1
        if device_dispatch.overflowed(force=entry.exchanges):
            return self._megatron_rerun(None, input_ids, position_ids, attention_mask, kw)
        _stats_after_replay(entry, b * s)
        self._feat(True)
        return _cut(entry.out, entry.wrapped, b, s, s_pad, copy=True, materialize=entry.materialize)

    def _capture_fwd(self, entry: _FwdEntry, kw, insts):
        from nemo_rl.models.megatron import device_dispatch

        ensure_graph_safe_rng()
        pool = self._pool_handle("fwd")
        wrapped = _Wrapped(self.model, dict(kw), force_deferred=entry.materialize)
        g = torch.cuda.CUDAGraph()
        n0 = device_dispatch.exchanges_counted()
        io = self._io()
        if io is not None:
            io.prepare("fwd_out", io.cap * max(io.width, 1) * 4)
            if entry.logits_bytes:
                io.prepare("fwd_logits", entry.logits_bytes)
        # Capture executes nothing (no exchange runs): the replay below is this call's
        # only forward on this rank, as an eager call would be on its EP peers.
        with _Swap(insts, entry.anchors), _StageInput(self.model, entry.stage_in), _stats_capture(
            entry
        ), torch.no_grad(), torch.cuda.graph(g, pool=pool, capture_error_mode="thread_local"):
            # (the stage-input copy runs before the capture: replays copy it themselves)
            out = wrapped(entry.ids, entry.pos)
            if entry.materialize and wrapped.deferred is None:
                raise _NotCapturable("logits_not_deferrable")
            shared = (
                io.out("fwd_logits" if entry.logits_bytes else "fwd_out", tuple(out.shape), out.dtype)
                if io is not None and isinstance(out, torch.Tensor) and out.is_contiguous()
                else None
            )
            if shared is not None:
                # The output goes to the shared buffer; the pool block is free again for
                # other captures (only the graph is kept per shape).
                shared.copy_(out)
                del out
                out = shared
            elif out.numel() * out.element_size() > (
                _head_out_bytes(_model_width(self.model)) if wrapped.deferred is not None else _max_out_bytes()
            ):
                raise _NotCapturable("out_too_large")
        # Exchanges inside the graph: each replay must read the overflow flag.
        entry.exchanges = device_dispatch.exchanges_counted() > n0
        entry.graph, entry.out, entry.wrapped = g, out, wrapped

    # -- forward + backward --------------------------------------------------
    def _train(self, common, input_ids, position_ids, attention_mask, kw):
        from nemo_rl.models.megatron import device_dispatch

        replay, why = _replay_state(self.model)
        b, s = input_ids.shape
        div = _train_bucket_div()
        s_pad = bucket_length(s, div) if div > 0 and not _no_buckets(self.model) else s
        if b * s_pad > _train_max_tokens() and b * s <= _train_max_tokens():
            s_pad = s
        if not _train_graphs():
            why = "train_off"
        elif replay is None:
            pass
        elif not _grad_sync_off(self.model):
            why = "grad_sync_on"
        elif b * s_pad > _train_max_tokens():
            why = "train_too_large"
        elif replay and len(replay) > 3:
            why = "train_stats"  # check statistics are collected on no-grad forwards only
        if why is not None:
            self._why("train", why)
            return self._eager(input_ids, position_ids, attention_mask, kw)
        # In-flight slot (PP > 1, 1F1B): the first entry of this shape whose last replay's
        # backward has run. PP 1 runs every backward before the next forward: slot 0 always.
        slots = _pp_size() > 1
        slot = 0
        while slots and (e := self.train.get((common, True, b, s_pad, replay, slot))) is not None and e.busy:
            slot += 1
        key = (common, True, b, s_pad, replay, slot)
        insts = _replay_instances(self.model) if replay else []
        entry = self._entry(self.train, key, b, s_pad, insts, replay, private=slots)
        entry.slot = slot
        entry.load(input_ids, position_ids, insts, s)
        ep = _ep_size() > 1 and device_dispatch.enabled()
        snap = _snapshot(self.model) if ep else None
        direct = _direct_capture() and self.warm["train"]
        if entry.graph is None and not direct and self.seen[key] == 0:
            # First sighting: eager (the same computation the graph replays later;
            # kernel autotuning and workspaces happen here).
            self.seen[key] += 1
            self.stats["eager_first"] += 1
            self._budget_release(True)
            wrapped = _Wrapped(self.model, dict(kw))
            # Private copies: the backward (embedding gradient) reads the ids after this
            # call returns, while the shared input buffers may be reloaded.
            with _Swap(insts, [a.clone() for a in entry.anchors]):
                out = wrapped(entry.ids.clone(), entry.pos.clone())
            if device_dispatch.overflowed():
                return self._megatron_rerun(snap, input_ids, position_ids, attention_mask, kw)
            self.warm["train"] = True
            return _cut(out, wrapped, b, s, s_pad, copy=False)
        if entry.graph is None:
            # PP > 1 slots capture into private pools, not the shared one: not counted against it.
            with self._measure_pool() if not slots else contextlib.nullcontext():
                self._capture_train(entry, kw, insts)
            self.stats["captured"] += 1
        if slots:
            entry.busy = True
            out = entry.graph(_stage_input(self.model)) if entry.stage_in is not None else entry.graph()
        else:
            out = entry.graph()
        self.stats["replay_train"] += 1
        if device_dispatch.overflowed(force=entry.exchanges):
            entry.busy = False  # its output is dropped: no backward will free the slot
            # The replayed forward's output is dropped before any backward. The rerun is
            # an eager forward whose backward recompute pops the worker's anchors from
            # replay_backward_list (REPLAY_BACKWARD), so they must still be there
            # (lg4-x xovf2: popping first -> "pop from empty list" in the recompute).
            self._open_done[0] = True
            del out
            return self._megatron_rerun(snap, input_ids, position_ids, attention_mask, kw)
        for inst in insts:
            # The worker appended this call's anchors for the backward's recompute
            # (REPLAY_BACKWARD pops them); the replayed backward reads the routes the
            # replayed forward wrote into the graph's buffer instead.
            if inst.replay_backward_list:
                inst.replay_backward_list.pop()
        if isinstance(out, torch.Tensor) and not slots:
            entry_done = self._mark_open(out)
            if hasattr(entry.graph, "set_done"):
                entry.graph.set_done(entry_done)
        self._feat(True)
        return _cut(out, entry.wrapped, b, s, s_pad, copy=False)


    def _capture_train(self, entry: _FwdEntry, kw, insts):
        """Forward + backward graphs (graph_capture.capture_fwd_bwd, no warm-up run).

        Parameter gradients go into ``main_grad`` inside the backward graph (gradient
        sync is off, checked by the caller), so a graph keeps no gradient buffers; the
        forward output and its gradient live in the shared buffers (shared pool).
        """
        from nemo_rl.models.megatron import device_dispatch
        from nemo_rl.models.megatron.graph_capture import capture_fwd_bwd

        ensure_graph_safe_rng()
        slots = _pp_size() > 1
        bwd_pool = None
        if slots:
            # PP > 1 slot: a private pool for what lives forward -> backward (other slots replay
            # meanwhile), backward transients from one pool shared by every slot. A private pool
            # takes fresh device memory, never the eager cache: return the cache first
            # (Super-L8 TP1 x PP2 x EP2: 73 GiB cached-free eager blocks -> OOM at slot 2's capture).
            # Every PP > 1 training entry (any shape, any slot) owns its forward pool: 1F1B keeps
            # forwards of several micro-batches in flight together (ts-scaleup 1a8ab43f: slot 0 of
            # two shapes sharing one pool overwrote each other's saved activations).
            _empty_cache()
            pool = torch.cuda.graph_pool_handle()
            if self.slot_bwd_pool is None:
                self.slot_bwd_pool = torch.cuda.graph_pool_handle()
            bwd_pool = self.slot_bwd_pool
        else:
            pool = self._pool_handle("train")
        wrapped = _Wrapped(self.model, dict(kw))
        n0 = device_dispatch.exchanges_counted()
        torch.cuda.synchronize()
        r0 = torch.cuda.memory_stats().get("reserved_bytes.all.current", 0)
        io = None if slots else self._io()
        shared = None
        stage = None
        if entry.stage_in is not None:
            dec = _decoder(self.model)
            received = dec.input_tensor

            def bind(t, _dec=dec):
                _dec.input_tensor = t

            stage = (entry.stage_in, bind)
        if io is not None:
            io.prepare("train_out", io.cap * max(io.width, 1) * 4)
            io.prepare("train_gout", io.cap * max(io.width, 1) * 4)
            def shared(shape, dtype, name):  # noqa: E306
                return io.out(name, shape, dtype)
        try:
            with _Swap(insts, entry.anchors):
                entry.graph = capture_fwd_bwd(
                    wrapped, entry.ids, entry.pos, warmup=0, pool=pool, main_grad_in_graph=True,
                    shared_out=shared, stage_input=stage, bwd_pool=bwd_pool,
                    on_backward=(lambda e=entry: setattr(e, "busy", False)) if slots else None,
                )
        finally:
            if stage is not None:
                dec.input_tensor = received
        entry.exchanges = device_dispatch.exchanges_counted() > n0
        entry.wrapped = wrapped
        if os.environ.get("NEMO_RL_LEARNER_GRAPHS_POOL_LOG") == "1":
            torch.cuda.synchronize()
            st = torch.cuda.memory_stats()
            print(f"[learner_graphs] train capture slot {key_slot(entry)} shape {tuple(entry.ids.shape)}: "
                  f"reserved {st.get('reserved_bytes.all.current', 0) / 2**30:.2f} GiB, allocated "
                  f"{st.get('allocated_bytes.all.current', 0) / 2**30:.2f} GiB, pool delta "
                  f"{(st.get('reserved_bytes.all.current', 0) - r0) / 2**30:.2f} GiB", flush=True)

    # -- call --------------------------------------------------------------
    def __call__(self, input_ids, position_ids, attention_mask, **kw):
        from nemo_rl.models.megatron import device_dispatch

        self.calls += 1
        device_dispatch.set_active(True)  # a capacity rerun's switch lasts one call
        grad = torch.is_grad_enabled()
        kind = "train" if grad else "fwd"
        if (
            grad
            and not _shared_pool()
            and _CACHES
            and not torch.cuda.is_current_stream_capturing()
        ):
            # Separate pools (v11oom): a training forward needs the memory the no-grad
            # graphs pin. No-op without graphs.
            release_fwd_graphs()
        if not enabled():
            return self._eager(input_ids, position_ids, attention_mask, kw)
        common, why = self._common_key(input_ids, position_ids, attention_mask, kw)
        if common is None:
            self._why(kind, why)
            self._feat(False)
            return self._eager(input_ids, position_ids, attention_mask, kw)
        if _shared_pool() and self.is_open():
            # Another call between a training graph's forward and backward replays: the
            # shared inputs and the pool's activations belong to the open graph.
            self._why(kind, "train_open")
            self._feat(False)
            return self._eager(input_ids, position_ids, attention_mask, kw)
        n0 = self.stats["replay_fwd"] + self.stats["replay_train"]
        out = (self._train if grad else self._no_grad)(common, input_ids, position_ids, attention_mask, kw)
        if self.stats["replay_fwd"] + self.stats["replay_train"] == n0:
            self._feat(False)
        return out


class _NotCapturable(RuntimeError):
    """Raised inside a capture whose result would not be kept (reason as message)."""


def _empty_cache() -> None:
    before = torch.cuda.memory_reserved()
    torch.cuda.empty_cache()
    _RELEASE["bytes"] += max(0, before - torch.cuda.memory_reserved())
    _RELEASE["times"] += 1


def _snapshot(model: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out = []
    for m in model.modules():
        t = getattr(m, "local_tokens_per_expert", None)
        if isinstance(t, torch.Tensor):
            out.append((t, t.clone()))
    return out


def _restore(snap) -> None:
    for t, v in snap or []:
        t.copy_(v)


def ensure_graph_safe_rng() -> None:
    """Make Megatron's CUDA RNG tracker graph safe (its forks run inside captures).

    Attention and recompute fork the tracker on every forward; the default tracker
    sets the generator's seed, which capture forbids. Re-initialized once with
    ``use_cudagraphable_rng`` and the seed in ``NEMO_RL_LEARNER_GRAPHS_SEED``
    (default 1234). The learner recipe has no dropout, so no value changes.
    """
    from megatron.core.tensor_parallel import random as mrandom

    tracker = mrandom.get_cuda_rng_tracker()
    if getattr(tracker, "use_cudagraphable_rng", False) or _RNG_DONE[0]:
        return
    mrandom.model_parallel_cuda_manual_seed(
        int(os.environ.get("NEMO_RL_LEARNER_GRAPHS_SEED", "1234")),
        use_cudagraphable_rng=True,
        force_reset_rng=True,
    )
    new = mrandom.get_cuda_rng_tracker()
    if new is not tracker and type(new) is type(tracker):
        # Modules keep bound methods of the original tracker (Transformer Engine's
        # attention stores tracker.fork): turn that object into the graph-safe one.
        tracker.__dict__.clear()
        tracker.__dict__.update(new.__dict__)
        mrandom._CUDA_RNG_STATE_TRACKER = tracker
    _RNG_DONE[0] = True


_RNG_DONE = [False]

_CACHES: dict[int, LearnerGraphs] = {}


def active() -> bool:
    """Route ``model_forward`` through ``call`` (graphs on, or device dispatch's capacity guard)."""
    from nemo_rl.models.megatron import device_dispatch

    return enabled() or device_dispatch.enabled()


def call(model: Any, input_ids, position_ids, attention_mask, **kw):
    """``model(...)`` through the graph cache of ``model`` (eager when not enabled / eligible)."""
    if not active():
        return model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, **kw)
    cache = _CACHES.get(id(model))
    if cache is None or cache.model is not model:
        cache = _CACHES[id(model)] = LearnerGraphs(model)
    out = cache(input_ids, position_ids, attention_mask, **kw)
    n = cache.calls
    if enabled() and (n in (1, 10, 100) or n % 500 == 0):
        print(f"[learner_graphs] calls={n} {dict(cache.stats)} graphs fwd="
              f"{sum(e.graph is not None for e in cache.fwd.values())} "
              f"train={sum(e.graph is not None for e in cache.train.values())}",
              flush=True)
    return out


_RELEASE = {"bytes": 0, "times": 0}


def release_fwd_graphs() -> int:
    """Release the no-grad graphs of every model and return their pools to the device.

    Separate-pool mode only (v11oom): called at the first grad-enabled learner call after
    a verification burst. Per call: a scan of the cached entries and, only when a graph
    was dropped, one ``torch.cuda.empty_cache``.
    """
    n = sum(c.release_fwd() for c in _CACHES.values())
    if n:
        _empty_cache()
    return n


def released() -> dict[str, int]:
    """Totals of pool releases in this process (times, bytes returned)."""
    return dict(_RELEASE)


def summary_line() -> Optional[str]:
    """The ``[learner_graphs] calls=..`` line with this process's current totals (None when off)."""
    if not enabled() or not _CACHES:
        return None
    st = stats()
    n = sum(c.calls for c in _CACHES.values())
    fwd = sum(sum(e.graph is not None for e in c.fwd.values()) for c in _CACHES.values())
    train = sum(sum(e.graph is not None for e in c.train.values()) for c in _CACHES.values())
    pool = sum(c.pool_bytes()[0] for c in _CACHES.values())
    if pool:
        st["pool_mib"] = pool >> 20
    return f"[learner_graphs] calls={n} {st} graphs fwd={fwd} train={train}"


def stats() -> dict[str, int]:
    """Calls per outcome (eager / captured / replayed, eager reasons, features) summed over models."""
    out: collections.Counter = collections.Counter()
    for c in _CACHES.values():
        out.update(c.stats)
    out.update(_GUARD_STATS)
    return dict(out)


# Capacity reruns of forwards that run outside ``call`` (``guarded_forward``).
_GUARD_STATS: collections.Counter = collections.Counter()


def guarded_forward(fn: Any, model: Any = None) -> Any:
    """``fn()`` (a forward that bypasses ``call``) with the MoE device dispatch's capacity guard.

    Device dispatch is patched class-wide, so a forward that calls the model directly (verification
    scorers, EP lockstep fillers) also runs the symmetric exchange; if it overflowed its receive rows,
    rows were dropped. Same rule as ``LearnerGraphs._eager``: the overflow flag is the same on every EP
    rank, so every rank reruns ``fn`` on Megatron's dispatcher (exact; ``model``'s routing counters are
    restored first when grad is on). The switch back happens here for no-grad calls; with grad it lasts
    through the backward (the next ``call`` re-activates, as after ``_megatron_rerun``).
    """
    from nemo_rl.models.megatron import device_dispatch

    if not device_dispatch.enabled():
        return fn()
    grad = torch.is_grad_enabled()
    snap = _snapshot(model) if grad and model is not None and _ep_size() > 1 else None
    device_dispatch.set_active(True)
    out = fn()
    if not device_dispatch.overflowed():
        return out
    _GUARD_STATS["capacity_rerun"] += 1
    print(f"[learner_graphs] MoE dispatch capacity overflow outside call: rerun on Megatron's dispatcher "
          f"(reruns {_GUARD_STATS['capacity_rerun']})", flush=True)
    del out
    _restore(snap)
    device_dispatch.set_active(False)
    try:
        return fn()
    finally:
        if not grad:
            device_dispatch.set_active(True)


def reset() -> None:
    """Drop every captured graph (e.g. after the model's parameters are reallocated)."""
    _CACHES.clear()
