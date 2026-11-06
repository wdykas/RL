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
"""Host-sync-free MoE token dispatch for the learner (opt-in, ``NEMO_RL_MOE_DEVICE_DISPATCH=1``).

Megatron-Core's AlltoAll dispatcher copies the routing counts to the host on every
MoE layer (``_maybe_dtoh_and_synchronize``): the AlltoAll split sizes, the chunk
sort of permutation 2 and the grouped experts' split sizes are host lists. Each copy
stalls the host until the GPU reaches the router, so the CPU cannot run ahead of
the GPU, and a forward with a host sync cannot be captured in a CUDA graph.

This module keeps every count on the GPU, for the batch-invariant fp16 MoE path
(dropless top-k routing, ``moe_permute_fusion=False``, expert TP 1):

- EP 1: permutation 1 is Megatron's (a fixed-size stable argsort, already capture
  safe); the chunk sort of permutation 2 is the identity (one rank, so the
  expert-major / rank-minor order is permutation 1's order); there is no AlltoAll.
  ``tokens_per_expert`` stays a CUDA tensor and the grouped expert linears read it
  on the device (``fp16_moe.GroupedFp16Linear``, segment kernels).
- EP > 1: the AlltoAll becomes a push exchange over torch symmetric memory (the CUDA
  backend the engine's NVLS dispatcher uses; one NVLink domain, e.g. a GB200 NVL72
  rack): every rank writes its routing counts into each peer's [EP, E] count table,
  a device barrier, then every rank derives all offsets on the GPU and writes each
  permutation-1 row straight into its owner's receive buffer at the row Megatron's
  AlltoAll + chunk sort would give it (local expert major, source rank minor, the
  source's order within). The combine writes expert outputs back to the source's
  rows in permutation-1 order; backward runs the same two exchanges in reverse.
  Ranks may hold different token counts (no padding, no shape agreement).

Same rows in the same order into the same kernels, so the same bits: the expert
GEMMs are the segment kernels, which are bitwise the per-expert cuBLAS GEMMs
; permutation and unpermutation are
Megatron's (or the fused batch-invariant unpermute) unchanged; the exchange only
moves rows.

Per-rank cost (EP > 1): one symmetric workspace sized for T_rank tokens per rank.
``NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS`` = T_cap is a learner *call's* token budget (the same on every
rank; it must cover the largest call: the training micro-batch and the verification batch, which
is what every config check and learner_graphs' buckets compare it with). A rank's MoE layer sees
only its sequence shard of the call: T_rank = ceil(T_cap / (TP x CP)) under sequence parallelism
(Megatron refuses MoE + TP > 1 without it in training; the MoE input is [S / TP, B, H]; the latent projection is
duplicated, so the dispatched token count is the shard's), ceil(T_cap / CP) without
(``token_split``). Until 11-05 T_rank was T_cap itself, i.e. TP x too large: every receive buffer
and every per-expert-row activation of a call was sized for TP x the provable worst case
(Ultra-L27 TP8 EP32, T_cap 18432: 1,105,920 receive rows, a 21.09 GiB rows x F fp32 expert
activation per MoE layer where a verify call of 10240 tokens routes a mean of 1280 x 22 = 28,160 rows to
an owner; the verify forward's OOM).
Rows are the *dispatched* rows: width D = ``moe_latent_size`` for latent MoE
(Nemotron Super / Ultra project to the latent before dispatch), else hidden_size.
Receive rows (``receive_rows``):

    R_cap = min(EP x T_rank x min(top-k, local experts),  ceil(c x T_rank x top-k))

The first term is the hard worst case (every peer sends every token's
min(top-k, local experts) rows to this rank). The second is c times the mean: every
rank sends T_rank x top-k rows spread over EP owners, so an owner receives T_rank x
top-k rows on average whatever EP and the local expert count are; c =
``NEMO_RL_MOE_DEVICE_DISPATCH_CAPACITY`` (default 2.0) is the load-imbalance headroom
over that mean. (Until 10-16 the second term was c x T_cap x min(top-k, local
experts): below the mean whenever top-k > local experts, e.g. Ultra top-22 over 8
local experts at EP64 gave 16 rows/token < 22 on average -> an overflow rerun on
every call.) Workspace: R_cap x (D x 4 B + 4 B) receive rows (token rows, reused
for the output gradients in backward) + T_rank x top-k x (D x 4 B + 4 B) return rows
(combine outputs, reused for the returned token gradients) + the [EP, E] counts.
Token arrays hold fp32 rows (the learner recipe's fp32 residual stream).
- Full Nemotron-3-Nano at EP 4 (D = H 2688, top-k 6, 32 local experts), T_cap 20480,
  c 2: R_cap = 245,760 rows, ~4.0 GB per rank (unchanged by the 10-16 fix: top-k <=
  local experts).
- Nemotron Ultra at EP 64 (D = latent 2048, top-k 22, 8 local experts), T_cap 4096,
  c 2: R_cap = 180,224 rows (1.48 GB) + return 90,112 rows (0.74 GB) = 2.2 GB per rank
  (the old formula: 65,536 rows of hidden 8192 = 5.1 GB and an overflow every call).
(Both examples at TP 1 / no sequence parallelism, T_rank = T_cap.)
It grows with c x T_rank x top-k x D, never with EP or the expert count.

Sparse cross-fade routing (``crossfade_routing_sparse``: top-k = C = k + extra, the band's zero-weight
candidates are not rows) sizes every row array by the live rows: top-k above becomes the stock k
(``row_slots``), so R_cap = c x T_rank x k and the return arrays hold c x T_rank x k rows (not T_rank x C).
Nano 16k EP8 (k 6, C 14, T_cap 20480, D 2688): receive 573,440 -> 245,760 rows, return 286,720 -> 245,760,
workspace 8.62 -> 4.92 GiB per rank. Ultra EP32 (k 22, extra 8, T_rank 2304 at T_cap 18432 / TP8 SP, latent
2048): receive 138,240 -> 101,376 rows, return rows stay T_rank x C = 69,120 (below c x T_rank x k), 1.58 -> 1.30
GiB. A source routing more than its return rows overflows like an owner does (same flag, every rank, exact
rerun).

Capacity overflow: every rank holds the whole [EP, E] count table after the count
exchange, so every rank computes every owner's received rows and the same overflow
decision. Rows beyond R_cap are dropped (memory safe, values wrong) and a device
flag is raised; ``overflowed`` (one host sync per learner call, at its end) tells
the caller to rerun the call on Megatron's dispatcher (``learner_graphs`` does,
consistently on every EP rank): the result is always exact (Megatron's dispatcher
gives the same bits), overflow only costs time, and with R_cap over the mean it
happens only when one owner receives more than c times the mean (an imbalance
event, not every call). The expert activations of a call are R_cap rows
(row-wise kernels touch the whole buffer; the GEMMs only the live rows). Beyond one
NVLink domain: not built yet (Megatron's dispatcher).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

try:  # MCore's symmetric-memory barrier (the engine's NVLS collectives use it)
    from megatron.core.inference.communication.torch_symm_triton.barrier import symm_mem_sync

    HAVE_SYMM = HAVE_TRITON
except ImportError:  # pragma: no cover
    HAVE_SYMM = False

_STATE: dict[str, Any] = {
    "patched": False,
    "logged": set(),
    "active": True,
    "ws": {},
    "overflow": None,  # device int32 flag: some layer's receive rows exceeded R_cap
    "symm_calls": 0,  # symmetric exchanges since the last overflowed() check
}


def enabled() -> bool:
    """Opt-in switch (env, so learner workers and probes see the same value)."""
    return os.environ.get("NEMO_RL_MOE_DEVICE_DISPATCH", "0") == "1"


def _epinv_ok() -> bool:
    """Device dispatch with the EP-invariant combine (default on; NEMO_RL_DEVICE_DISPATCH_EPINV=0: off)."""
    return os.environ.get("NEMO_RL_DEVICE_DISPATCH_EPINV", "1") == "1"


def graph_safe() -> bool:
    """Do learner MoE layers run without host syncs (installed, active, no limb combine)?"""
    from nemo_rl.models.megatron import ep_invariant_combine

    return bool(
        _STATE["patched"]
        and _STATE["active"]
        and not (ep_invariant_combine.enabled() and not _epinv_ok())
        # an EP group whose symmetric workspace failed keeps Megatron's AlltoAll
        and all(ws is not None for ws in _STATE["ws"].values())
    )


def model_graph_safe(model: Any) -> tuple[bool, str]:
    """Every MoE token dispatcher of ``model`` takes the host-sync-free path (graph_safe()
    plus per layer: supported layout, e.g. expert-TP 1; EP > 1: its symmetric workspace
    exists already -- created by an eager call, never inside a capture). (ok, reason)."""
    if not graph_safe():
        return False, "dispatch_off"
    try:
        from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher as D
    except ImportError:  # pragma: no cover
        return True, ""
    for m in model.modules():
        d = getattr(m, "token_dispatcher", None)
        if d is None:
            continue
        if not isinstance(d, D):
            return False, "dispatch_type"
        if not _base_supported(d):
            return False, "dispatch_etp" if getattr(d, "tp_size", 1) != 1 else "dispatch_layout"
        if d.ep_size > 1:
            if _STATE["ws"].get(_ws_key(d)) is None:
                return False, "dispatch_no_workspace"
    return True, ""


def set_active(flag: bool) -> None:
    """Switch the patched dispatcher on / off at runtime (A/B tests in one process)."""
    _STATE["active"] = bool(flag)


def token_cap() -> int:
    """T_cap: most tokens of one learner call (the micro-batch / verify batch budget, before the sequence split)."""
    return int(os.environ.get("NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS", "20480"))


def token_split(config: Any) -> int:
    """Ranks a call's tokens are split over before an MoE layer: TP under sequence parallelism, times CP."""
    tp = int(getattr(config, "tensor_model_parallel_size", 1) or 1) if getattr(config, "sequence_parallel", False) else 1
    cp = int(getattr(config, "context_parallel_size", 1) or 1)
    return max(1, tp * cp)


def rank_tokens(t_cap: int, split: int) -> int:
    """T_rank: most tokens one rank passes through an MoE layer in a call of at most T_cap tokens.

    The sequence shards of a call are equal (sequence parallelism and CP need the padded length divisible by
    the split), so a rank holds call_tokens / split <= T_cap / split tokens.
    """
    return -(-int(t_cap) // max(1, int(split)))


def capacity_factor() -> float:
    """c: receive rows per rank = c x the mean (T_rank x top-k), capped at the worst case."""
    return float(os.environ.get("NEMO_RL_MOE_DEVICE_DISPATCH_CAPACITY", "2.0"))


def call_cap_enabled() -> bool:
    """Per-call receive rows (``NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP=1``, opt-in): a call's expert
    buffers hold min(R_cap, ceil(c x this rank's routed rows)) rows instead of R_cap, so a small
    call (verification piece, PP micro-batch, probe) does not carry T_cap-sized expert activations
    (Super-L8 PP2 training graph at 1 x 512 tokens: 45 of 53 GiB of the capture's peak were
    R_cap-row expert buffers). Every owner's cap travels with its counts,
    so all EP ranks take the same overflow decision (an owner receiving more than its cap: rerun on
    Megatron's dispatcher, exact). Floor: ``NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP_MIN`` rows (default
    1/8 of R_cap) so fillers next to full calls do not overflow at every layer."""
    return os.environ.get("NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP", "0") == "1"


def call_rows(r_cap: int, n_rows: int, c: float) -> int:
    """Receive rows of one call on this rank (see ``call_cap_enabled``)."""
    if not call_cap_enabled():
        return r_cap
    floor = int(os.environ.get("NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP_MIN", str(max(1, r_cap // 8))))
    if not torch.is_grad_enabled():
        floor = max(floor, nograd_floor(r_cap))
    return int(min(r_cap, max(floor, -(-c * n_rows // 1))))


def nograd_floor(r_cap: int) -> int:
    """Receive-row floor of no-grad calls (verification forwards): ``NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP_NOGRAD``
    = fraction of R_cap (default 0: the common floor). A no-grad call frees its expert buffers layer by layer, so a
    larger floor costs one layer's transient buffers, not stored activations; it lets verify forwards larger than
    a training micro-batch run without overflowing (an overflow reruns the whole call on Megatron's dispatcher)."""
    frac = float(os.environ.get("NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP_NOGRAD", "0") or 0)
    return int(min(r_cap, -(-frac * r_cap // 1))) if frac > 0 else 0


def receive_rows(world: int, num_experts: int, topk: int, t_cap: int, c: float, row_k: Optional[int] = None) -> int:
    """R_cap: receive rows per owner rank (module docstring); ``t_cap`` is T_rank (``rank_tokens``).

    min(hard worst case EP x T_rank x min(top-k, local experts), ceil(c x T_rank x row_k)):
    an owner receives T_rank x row_k rows on average (every source spreads its T_rank x
    row_k rows over EP owners), so c is headroom over the mean for any EP / expert count.
    ``row_k`` (``row_slots``): routed rows per token the capacity is sized for; top-k unless
    the routing map is sparse.
    """
    ne = num_experts // world
    worst = world * t_cap * min(topk, ne)
    return int(min(worst, -(-c * t_cap * (topk if row_k is None else row_k) // 1)))


def row_slots(topk: int) -> int:
    """Routed rows per token that size the exchange's row arrays.

    Fixed-width routing: top-k, every slot is a row. Sparse cross-fade routing (``crossfade_routing``
    sparse dispatch: zero-weight slots are not rows): the stock k. The band's weight-0 candidates are dropped
    and a token's live slots are k plus the extras inside the band. The band does not bound how many extras
    that is (near-tied scores can put every candidate inside it), so no per-token bound below k + extra
    exists. Their mean is a fraction of k (Nano 6.55 / 7.0 / 7.44 live slots for bands cf01 / S30 / u03,
    next/16-f3), so it lies inside the capacity headroom c. The worst case stays exact: a receive or return
    array that would overflow raises the overflow flag on every rank, and the call reruns on Megatron's
    dispatcher.
    """
    try:
        from nemo_rl.models.megatron import crossfade_routing
    except ImportError:  # pragma: no cover
        return topk
    st = crossfade_routing.get_state()
    if st.get("m") is None or not st.get("sparse") or st.get("k") is None:
        return topk
    return min(int(topk), int(st["k"]))


def dispatch_width(config: Any) -> int:
    """Width of a dispatched token row: the latent for latent MoE, else hidden_size."""
    return int(getattr(config, "moe_latent_size", None) or config.hidden_size)


def _overflow_flag(device) -> torch.Tensor:
    f = _STATE["overflow"]
    if f is None:
        f = _STATE["overflow"] = torch.zeros((), dtype=torch.int32, device=device)
    return f


def exchanges_counted() -> int:
    """Symmetric exchanges launched from Python since the last ``overflowed()`` check."""
    return _STATE["symm_calls"]


def overflowed(force: bool = False) -> bool:
    """Did a symmetric exchange since the last call drop rows (capacity)? Resets the flag.

    One host sync when an exchange ran (none otherwise). A replayed CUDA graph runs
    exchanges without Python seeing them: its caller passes ``force=True``. The flag
    is the same on every rank of an EP group (computed from the shared count table),
    so callers that rerun on overflow stay in lockstep.
    """
    if not _STATE["symm_calls"] and not force:
        return False
    _STATE["symm_calls"] = 0
    f = _STATE["overflow"]
    if f is None:
        return False
    hit = bool(f.item())
    if hit:
        f.zero_()
    return hit


def _log_once(key: str, msg: str) -> None:
    if key not in _STATE["logged"]:
        _STATE["logged"].add(key)
        print(f"[device_dispatch] {msg}", flush=True)


def _base_supported(d: Any) -> bool:
    from nemo_rl.models.megatron import ep_invariant_combine

    cfg = d.config
    return (
        _STATE["active"]
        # The EP-invariant limb combine only replaces the unpermute (bi_unpermute ->
        # ep_invariant_combine.unpermute_rows: same rows, same inverse_map, exact limb
        # sums, Triton, no host sync); device dispatch only moves rows (its combine
        # returns them in Megatron's permutation order), so the two compose. Before
        # ts-exact-harden this switched device dispatch off whenever the combine was on,
        # i.e. in every lanes arm (GEN_EP != EP): learner graphs were 100% eager_ineligible
        # there. NEMO_RL_DEVICE_DISPATCH_EPINV=0 restores that.
        and not (ep_invariant_combine.enabled() and not _epinv_ok())
        and d.tp_size == 1
        and not d.drop_and_pad
        and cfg.moe_expert_capacity_factor is None
        and not cfg.moe_router_padding_for_quantization
        and not cfg.moe_permute_fusion
    )


def _inference_active() -> bool:
    try:
        from megatron.core.inference.utils import InferenceMode
    except ImportError:  # pragma: no cover
        return False
    return bool(InferenceMode.is_active())


def _sparse_routing() -> bool:
    """Learner routing maps may hold fewer than top-k experts per token (crossfade_routing sparse dispatch)."""
    try:
        from nemo_rl.models.megatron import crossfade_routing
    except ImportError:  # pragma: no cover
        return False
    st = crossfade_routing.get_state()
    return bool(st.get("m") is not None and st.get("sparse"))


def _mode(d: Any) -> Optional[str]:
    """'local' (EP 1), 'symm' (EP > 1, symmetric-memory exchange) or None (Megatron's path)."""
    if not _base_supported(d) or _inference_active():  # engine forwards keep their path
        return None
    if d.ep_size == 1:
        return "local"
    return "symm" if HAVE_SYMM and _workspace(d) is not None else None


# ---------------------------------------------------------------------------
# EP > 1: push exchange over symmetric memory.
# ---------------------------------------------------------------------------

if HAVE_TRITON:

    @triton.jit
    def _put_counts_kernel(counts_ptr, buffer_ptrs, cnt_off, E, RANK: tl.constexpr, BLOCK: tl.constexpr):
        # Program p writes this rank's [E] counts into peer p's count table row RANK.
        p = tl.program_id(0)
        base = tl.load(buffer_ptrs.to(tl.pointer_type(tl.uint64)) + p)
        dst = (base + cnt_off).to(tl.pointer_type(tl.int32))
        idx = tl.arange(0, BLOCK)
        for c0 in range(0, E, BLOCK):
            m = c0 + idx < E
            v = tl.load(counts_ptr + c0 + idx, mask=m, other=0).to(tl.int32)
            tl.store(dst + RANK * E + c0 + idx, v, mask=m)

    @triton.jit
    def _push_rows_kernel(
        src_ptr,  # [N, H] rows to send (local)
        rank_ptr,  # [N] int64 destination rank of each row
        pos_ptr,  # [N] int64 destination row
        live_ptr,  # scalar int64: rows [0, live) are sent
        buffer_ptrs,  # [W] uint64 symmetric buffer bases
        dst_off,  # byte offset of the destination array in the symmetric buffer
        H,
        cap,  # rows of the destination array: rows with pos >= cap are dropped (overflow)
        ELEM_BYTES: tl.constexpr,
        BLOCK_H: tl.constexpr,
        CAST: tl.constexpr = 0,  # 0: rows as is; 1: fp16; 2: bf16 (round to nearest even, as torch .to)
    ):
        live = tl.load(live_ptr)
        cols = tl.arange(0, BLOCK_H)
        for row in range(tl.program_id(0), live, tl.num_programs(0)):
            row64 = row.to(tl.int64)
            p = tl.load(rank_ptr + row64)
            pos = tl.load(pos_ptr + row64)
            if pos < cap:
                base = tl.load(buffer_ptrs.to(tl.pointer_type(tl.uint64)) + p)
                if CAST == 1:
                    dst = (base + dst_off + pos * H * ELEM_BYTES).to(tl.pointer_type(tl.float16))
                elif CAST == 2:
                    dst = (base + dst_off + pos * H * ELEM_BYTES).to(tl.pointer_type(tl.bfloat16))
                else:
                    dst = (base + dst_off + pos * H * ELEM_BYTES).to(src_ptr.dtype)
                for h0 in range(0, H, BLOCK_H):
                    m = h0 + cols < H
                    v = tl.load(src_ptr + row64 * H + h0 + cols, mask=m)
                    if CAST == 1:
                        v = v.to(tl.float16)
                    elif CAST == 2:
                        v = v.to(tl.bfloat16)
                    tl.store(dst + h0 + cols, v, mask=m)

    @triton.jit
    def _copy_live_kernel(src_ptr, dst_ptr, live_ptr, H, BLOCK_H: tl.constexpr):
        # dst[:live] = src[:live] (rows of H elements); rows past live untouched.
        live = tl.load(live_ptr)
        cols = tl.arange(0, BLOCK_H)
        for row in range(tl.program_id(0), live, tl.num_programs(0)):
            row64 = row.to(tl.int64)
            for h0 in range(0, H, BLOCK_H):
                m = h0 + cols < H
                tl.store(dst_ptr + row64 * H + h0 + cols, tl.load(src_ptr + row64 * H + h0 + cols, mask=m), mask=m)

    @triton.jit
    def _barrier_kernel(signal_pad_ptrs, RANK: tl.constexpr, WORLD: tl.constexpr):
        symm_mem_sync(
            signal_pad_ptrs, 0, RANK, WORLD, hasPreviousMemAccess=True, hasSubsequentMemAccess=True
        )


def _row_block(h: int) -> int:
    return min(triton.next_power_of_2(max(h, 1)), 2048)


def _programs(rows: int) -> int:
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return max(1, min(rows, 4 * sms))


class _Workspace:
    """Symmetric workspace of one EP group (allocated once, reused by every MoE layer)."""

    def __init__(
        self, group: Any, num_experts: int, topk: int, hidden: int, t_cap: int, split: int = 1,
        row_k: Optional[int] = None,
    ):
        import torch.distributed._symmetric_memory as symm_mem

        self.group = group
        self.world = group.size()
        self.rank = group.rank()
        # t_call: the call budget T_cap; t_cap: T_rank, this rank's share (rank_tokens), which sizes every array.
        self.t_call, self.split = t_cap, split
        self.E, self.k, self.H, self.t_cap = num_experts, topk, hidden, rank_tokens(t_cap, split)
        self.ne = num_experts // self.world
        # Row arrays are sized for row_k routed rows per token (``row_slots``): top-k for fixed-width routing,
        # the stock k for sparse cross-fade routing (the band's live extras ride the headroom c; past it the
        # overflow flag reruns the call).
        self.row_k = topk if row_k is None else int(row_k)
        self.r_cap = receive_rows(self.world, num_experts, topk, self.t_cap, capacity_factor(), self.row_k)
        # s_max: rows of this rank's permutation 1 (T_rank x top-k, a hard bound); s_cap: rows of its return
        # arrays. Fixed width: every permutation row is real, s_cap = s_max. Sparse: only the live rows come back,
        # at most c x T_rank x row_k of them (more: overflow).
        self.s_max = self.t_cap * topk
        self.s_cap = (
            self.s_max if self.row_k >= topk else int(min(self.s_max, -(-capacity_factor() * self.t_cap * self.row_k // 1)))
        )
        # (name, elements, bytes per element): token rows take the dispatched tokens'
        # dtype (fp32 under fp32_residual_connection, else 16-bit), so they are sized
        # for 4 bytes and viewed per call. Forward and backward arrays share storage
        # ("g_recv" = "recv", "dx_ret" = "ret"): every exchange copies its rows out and
        # passes a barrier before any rank writes the array again.
        layout = [
            ("cnt", self.world * (num_experts + 1), 4),  # [EP, E] counts + each owner's call cap
            ("recv", self.r_cap * hidden, 4),
            ("recv_p", self.r_cap, 4),
            ("ret", self.s_cap * hidden, 4),
            ("dp_ret", self.s_cap, 4),
        ]
        self.off, self.nbytes, total = {}, {}, 0
        for name, numel, size in layout:
            total = -(-total // 256) * 256
            self.off[name] = total
            self.nbytes[name] = numel * size
            total += numel * size
        for alias, name in (("g_recv", "recv"), ("dx_ret", "ret")):
            self.off[alias], self.nbytes[alias] = self.off[name], self.nbytes[name]
        self.cap_rows = {
            "recv": self.r_cap, "recv_p": self.r_cap, "g_recv": self.r_cap,
            "ret": self.s_cap, "dx_ret": self.s_cap, "dp_ret": self.s_cap,
        }
        self.buf = symm_mem.empty(total, dtype=torch.uint8, device="cuda")
        self.hdl = symm_mem.rendezvous(self.buf, group.group_name)
        self.bytes = total

    def view(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        """Symmetric array ``name`` as a flat tensor of ``dtype``."""
        return self.buf[self.off[name] : self.off[name] + self.nbytes[name]].view(dtype)

    def barrier(self) -> None:
        _barrier_kernel[(1,)](self.hdl.signal_pad_ptrs_dev, RANK=self.rank, WORLD=self.world, num_warps=4)

    def push(
        self,
        src: torch.Tensor,
        ranks: torch.Tensor,
        pos: torch.Tensor,
        live: torch.Tensor,
        dst: str,
        wire_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Write rows [0, live) of src [N, H] to row pos[i] of array ``dst`` on rank ranks[i].

        ``wire_dtype`` (fp16 / bf16): rows are rounded to it in the kernel (round to nearest
        even, bitwise ``src.to(wire_dtype)``) and the array holds that dtype.
        """
        n = src.shape[0]
        if n == 0:
            return
        h = src.shape[1] if src.dim() == 2 else 1
        cast = {None: 0, torch.float16: 1, torch.bfloat16: 2}[wire_dtype]
        _push_rows_kernel[(_programs(n),)](
            src,
            ranks,
            pos,
            live,
            self.hdl.buffer_ptrs_dev,
            self.off[dst],
            h,
            self.cap_rows[dst],
            ELEM_BYTES=src.element_size() if cast == 0 else 2,
            BLOCK_H=_row_block(h),
            CAST=cast,
        )

    def take(
        self, name: str, rows: int, cols: int, live: torch.Tensor, dtype: torch.dtype = torch.float32,
        zero_tail: bool = False,
    ) -> torch.Tensor:
        """Copy rows [0, live) of symmetric array ``name`` (as ``dtype``) into a fresh [rows, cols] tensor.

        ``zero_tail``: rows [live, rows) are zero (padding rows of a sparse routing map: their outputs and input
        gradients must be exact zeros, the permutation's backward adds them into token gradients).
        """
        flat = self.view(name, dtype)
        n = min(rows, self.cap_rows[name])  # the array's rows; live <= n (the plan clamps it)
        src = flat[: n * cols].view(n, cols) if cols > 1 else flat[:n]
        shape = (rows, cols) if cols > 1 else (rows,)
        out = (torch.zeros if zero_tail else torch.empty)(shape, dtype=src.dtype, device=src.device)
        if n:
            _copy_live_kernel[(_programs(n),)](src, out, live, cols, BLOCK_H=_row_block(cols))
        return out


def _ws_key(d: Any) -> tuple:
    return (d.ep_group.group_name, d.num_experts, d.config.moe_router_topk, dispatch_width(d.config),
            token_split(d.config), row_slots(d.config.moe_router_topk))


def _workspace(d: Any) -> Optional[_Workspace]:
    """The EP group's workspace (None: unavailable, the layer keeps Megatron's path)."""
    group = d.ep_group
    key = _ws_key(d)
    width, split, row_k = key[3], key[4], key[5]
    ws = _STATE["ws"].get(key)
    if ws is None and key not in _STATE["ws"]:
        try:
            ws = _Workspace(group, d.num_experts, d.config.moe_router_topk, width, token_cap(), split, row_k)
            _log_once(
                f"ws{key}",
                f"EP={ws.world} symmetric exchange: T_cap={ws.t_call} tokens/call, sequence split {split} -> "
                f"T_rank={ws.t_cap} tokens/rank, row width {ws.H}, rows/token {ws.row_k} of top-{ws.k}, receive rows "
                f"{ws.r_cap}, return rows {ws.s_cap} (capacity {capacity_factor()}), "
                f"workspace {ws.bytes / 2**30:.2f} GiB per rank",
            )
        except (RuntimeError, ImportError) as exc:
            _log_once(f"wsfail{key}", f"EP={group.size()} symmetric memory unavailable ({exc}); Megatron's AlltoAll")
            ws = None
        _STATE["ws"][key] = ws
    return ws


def route_tables(
    C: torch.Tensor, me: int, n_rows: int, r_cap: int, caps: Optional[torch.Tensor] = None,
    s_cap: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Device-side routes of one exchange from the [EP, E] count table (pure tensor math).

    ``C[s, e]``: rows source rank s routes to global expert e (int64); ``me``: this
    rank; ``n_rows``: this rank's permutation-1 rows; ``r_cap``: receive rows per owner.
    Returns fwd_rank / fwd_pos [n_rows] (owner and owner row of each of my rows),
    back_rank / back_pos [r_cap] (source and source row of each received row), live
    (received rows kept, scalar), tokens_per_expert [local experts] and overflow
    (scalar bool: some owner receives more than r_cap rows, or some source routes more than
    ``s_cap`` rows (its return arrays' rows; None: unbounded); the same on every rank).
    CPU-testable; ``_Plan`` runs it on the GPU after the count exchange.
    """
    W, E = C.shape
    ne = E // W
    dev = C.device
    src_start = C.cumsum(1) - C  # [W, E] first row of each expert in each source's permutation 1
    sizes = C.view(W, W, ne).permute(1, 2, 0).reshape(W, ne * W)  # [owner, (e, src)]
    off = (sizes.cumsum(1) - sizes).view(W, ne, W)  # owner row of segment (e, src)
    # Forward: my permutation-1 rows -> owner rows.
    my_end = C[me].cumsum(0)
    rows = torch.arange(n_rows, device=dev)
    # Rows past this rank's real count (a sparse routing map under the static T x top-k capacity, cross-fade
    # routing's zero-weight slots) are padding: never pushed (rows [0, live) only); the clamp keeps their table
    # entries in range.
    g = torch.searchsorted(my_end, rows, right=True).clamp_max(E - 1)
    fwd_rank = g // ne
    fwd_pos = off[:, :, me].reshape(E)[g] + rows - src_start[me][g]
    # Capacity: every rank sees every owner's total (same decision on all ranks).
    # ``caps`` [W]: every owner's receive rows of this call (per-call caps); default r_cap for all.
    overflow = (sizes.sum(1) > (r_cap if caps is None else caps)).any()
    if s_cap is not None:
        overflow = overflow | (C.sum(1) > s_cap).any()
    # Received rows (this rank as owner): segment (e, src) -> the source's row.
    mine = sizes[me]
    seg_end = mine.cumsum(0)
    live = seg_end[-1:].clamp_max(r_cap).reshape(())
    r = torch.arange(r_cap, device=dev)
    seg = torch.searchsorted(seg_end, r, right=True).clamp_max(ne * W - 1)
    e, src = seg // W, seg % W
    back_rank = src
    back_pos = src_start[src, me * ne + e] + r - (seg_end[seg] - mine[seg])
    # [ne] rows per local expert (clamped to the buffer on overflow: memory safe).
    ends = mine.view(ne, W).sum(1).cumsum(0).clamp_max(r_cap)
    tokens_per_expert = ends - torch.cat([ends.new_zeros(1), ends[:-1]])
    return dict(fwd_rank=fwd_rank, fwd_pos=fwd_pos, back_rank=back_rank, back_pos=back_pos,
                live=live, tokens_per_expert=tokens_per_expert, overflow=overflow)


class _Plan:
    """Device-side routes of one MoE layer call (all from the exchanged [EP, E] counts)."""

    def __init__(self, ws: _Workspace, local_counts: torch.Tensor, n_rows: int, sparse: bool = False):
        if n_rows > ws.s_max:
            raise RuntimeError(
                f"device_dispatch: {n_rows} routed rows > T_rank x top-k = {ws.t_cap} x {ws.k} = {ws.s_max} "
                f"(T_rank = ceil(T_cap {ws.t_call} / sequence split {ws.split})): the call has more than T_cap "
                "tokens; raise NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS (it must be the same on every rank)"
            )
        W, E, me = ws.world, ws.E, ws.rank
        dev = local_counts.device
        # This call's receive rows on this rank (R_cap unless per-call caps are on); its value
        # travels as column E of the count table, so every rank sees every owner's cap.
        # Per-call caps scale with the call's rows at row_k per token (sparse: the live rows, not the padding).
        self.r_call = call_rows(ws.r_cap, -(-n_rows * ws.row_k // ws.k), capacity_factor())
        ext = torch.cat([local_counts.to(torch.int32).reshape(-1),
                         torch.full((1,), self.r_call, dtype=torch.int32, device=dev)])
        _put_counts_kernel[(W,)](
            ext, ws.hdl.buffer_ptrs_dev, ws.off["cnt"], E + 1, RANK=me, BLOCK=256
        )
        ws.barrier()
        Cx = ws.view("cnt", torch.int32)[: W * (E + 1)].view(W, E + 1).to(torch.int64)  # copy: rewritten next layer
        C, caps = Cx[:, :E], Cx[:, E]
        t = route_tables(C, me, n_rows, self.r_call, caps, ws.s_cap if ws.s_cap < ws.s_max else None)
        self.fwd_rank, self.fwd_pos = t["fwd_rank"], t["fwd_pos"]
        self.n_src = n_rows
        # Sparse routing map (cross-fade zero-weight slots dropped on the learner): the real rows are the counted
        # ones, the rest of the static T x top-k permutation is padding.
        self.sparse = sparse
        # Clamped to the return arrays' rows (rows past it are an overflow: flagged above, the call reruns).
        self.src_live = (
            local_counts.to(torch.int64).sum().clamp_max(ws.s_cap).reshape(())
            if sparse
            else torch.full((), n_rows, dtype=torch.int64, device=dev)
        )
        _overflow_flag(dev).bitwise_or_(t["overflow"].to(torch.int32))
        _STATE["symm_calls"] += 1
        self.live = t["live"]
        self.back_rank, self.back_pos = t["back_rank"], t["back_pos"]
        self.tokens_per_expert = t["tokens_per_expert"]
        self.ws = ws


class GradBox:
    """Side channel for a gradient whose dtype differs from its forward tensor's.

    A 16-bit wire (``fused_learner_ops``) makes the exchanged tensor fp16 / bf16 while its
    gradient must stay fp32 (dispatch: the fc1 input gradient) or is consumed 16-bit
    (combine backward: fc2 casts its output gradient to bf16 first). Autograd would cast
    a gradient to its tensor's dtype, so the producer stores it here and returns None;
    the consumer takes it (and raises if it is missing, never computes with a guess).
    """

    __slots__ = ("grad",)

    def __init__(self) -> None:
        self.grad: Optional[torch.Tensor] = None

    def take(self, what: str) -> torch.Tensor:
        g, self.grad = self.grad, None
        if g is None:
            raise RuntimeError(f"device_dispatch 16-bit wire: {what} gradient was not handed over")
        return g


BOX_ATTR = "_learnfuse_gbox"


def wire_ok() -> bool:
    """16-bit wire fusions enabled and no sticky-rounding hooks (they edit the exchanged rows)."""
    from nemo_rl.models.megatron import fp16_gemm, fp16_moe
    from nemo_rl.models.megatron import fused_learner_ops as flo

    return (
        flo.enabled()
        and flo.ENABLED.get("wire", True)
        and fp16_gemm.enabled()
        and fp16_gemm._STATE["site_cast"] is None
        and fp16_gemm._STATE["moe_preprocess"] is None
        and fp16_moe._HOOKS["fc2_input"] is None
    )


def _wire16(d: Any, tokens: torch.Tensor) -> bool:
    """fp16 forward dispatch: the only consumer of the dispatched rows is the fp16 expert fc1,
    which rounds them to fp16 first (fp16_moe.training_experts_forward), so rounding at the
    source is the same value. Off with sticky-rounding hooks or shared-expert overlap."""
    return wire_ok() and tokens.dtype == torch.float32 and d.shared_experts is None


class _Dispatch(torch.autograd.Function):
    """Source permutation-1 rows -> owner rows (expert major, source minor)."""

    @staticmethod
    def forward(ctx, tokens, probs, plan, wire16=False):  # noqa: D102
        ws = plan.ws
        ctx.set_materialize_grads(False)
        if wire16:
            # fp16 rows on the wire (half the bytes), rounded in the push kernel.
            ws.push(tokens.contiguous(), plan.fwd_rank, plan.fwd_pos, plan.src_live, "recv", torch.float16)
        else:
            ws.push(tokens.contiguous(), plan.fwd_rank, plan.fwd_pos, plan.src_live, "recv")
        ws.push(probs.float().contiguous().view(-1, 1), plan.fwd_rank, plan.fwd_pos, plan.src_live, "recv_p")
        ws.barrier()
        h = tokens.shape[1]  # the call's row width (<= ws.H: latent / hidden rows, or narrower key rows)
        x = ws.take("recv", plan.r_call, h, plan.live, torch.float16 if wire16 else tokens.dtype)
        p = ws.take("recv_p", plan.r_call, 1, plan.live)
        ws.barrier()  # every owner holds its copy before any peer reuses the receive arrays
        ctx.plan, ctx.h = plan, h
        ctx.tok_dtype, ctx.p_dtype = tokens.dtype, probs.dtype
        ctx.box = None
        if wire16:
            ctx.box = GradBox()
            setattr(x, BOX_ATTR, ctx.box)  # read by fp16_moe.GroupedFp16Linear (fc1)
            return x, p.to(probs.dtype)
        return x.to(tokens.dtype), p.to(probs.dtype)

    @staticmethod
    def backward(ctx, gx, gp):  # noqa: D102
        plan, ws = ctx.plan, ctx.plan.ws
        if ctx.box is not None:
            gx = ctx.box.take("expert fc1 input (fp16 dispatch)")
        gx = (torch.zeros((plan.r_call, ctx.h), dtype=ctx.tok_dtype, device=plan.live.device) if gx is None
              else gx.to(ctx.tok_dtype).contiguous())
        gp = (torch.zeros((plan.r_call,), dtype=torch.float32, device=plan.live.device) if gp is None
              else gp.float().contiguous())
        ws.push(gx, plan.back_rank, plan.back_pos, plan.live, "dx_ret")
        ws.push(gp.view(-1, 1), plan.back_rank, plan.back_pos, plan.live, "dp_ret")
        ws.barrier()
        dx = ws.take("dx_ret", plan.n_src, ctx.h, plan.src_live, ctx.tok_dtype, zero_tail=plan.sparse)
        dp = ws.take("dp_ret", plan.n_src, 1, plan.src_live, zero_tail=plan.sparse)
        ws.barrier()
        return dx.to(ctx.tok_dtype), dp.view(-1).to(ctx.p_dtype), None, None


def _skip_recompute_combine(d: Any) -> bool:
    """Inside Megatron's selective MoE recompute (the backward re-runs the MoE layer forward
    only to rebuild its autograd graph), the combine output's values are never read: the
    region's remaining ops (batch-invariant unpermute = row sums, + shared-expert output)
    have value-independent backwards and the region's output only seeds
    ``torch.autograd.backward``. Not with a latent MoE (fc2 norm / projection after the
    combine save their inputs) or other recompute granularities."""
    from nemo_rl.models.megatron import fused_learner_ops as flo

    if not (flo.enabled() and flo.ENABLED.get("skip_recompute_combine", True) and torch.is_grad_enabled()):
        return False
    cfg = d.config
    if (
        getattr(cfg, "recompute_granularity", None) != "selective"
        or "moe" not in (getattr(cfg, "recompute_modules", None) or ())
        or getattr(cfg, "moe_latent_size", None)
    ):
        return False
    from megatron.core.tensor_parallel.random import is_checkpointing

    return bool(is_checkpointing())


class _Combine(torch.autograd.Function):
    """Owner expert-output rows -> the source's permutation-1 rows."""

    @staticmethod
    def forward(ctx, y, plan, skip=False):  # noqa: D102
        ws = plan.ws
        ctx.plan, ctx.h = plan, y.shape[1]  # row width: latent or hidden (ts-latent-moe)
        ctx.y_dtype = y.dtype
        # bf16 output gradient on the wire when the producer (expert fc2) takes it from a box
        # and casts it to bf16 first anyway (fp16_moe.GroupedFp16Linear): same bits.
        ctx.box = getattr(y, BOX_ATTR, None)
        if skip:  # recompute: values unused (_skip_recompute_combine); all EP ranks skip alike.
            # Zeros, not uninitialized memory: finite values for any monitor downstream (one memset).
            return torch.zeros((plan.n_src, ctx.h), dtype=y.dtype, device=y.device)
        ws.push(y.float().contiguous(), plan.back_rank, plan.back_pos, plan.live, "ret")
        ws.barrier()
        out = ws.take("ret", plan.n_src, y.shape[1], plan.src_live, zero_tail=plan.sparse)
        ws.barrier()
        return out.to(y.dtype)

    @staticmethod
    def backward(ctx, g):  # noqa: D102
        plan, ws = ctx.plan, ctx.plan.ws
        if ctx.box is not None:
            ws.push(g.float().contiguous(), plan.fwd_rank, plan.fwd_pos, plan.src_live, "g_recv", torch.bfloat16)
            ws.barrier()
            ctx.box.grad = ws.take("g_recv", plan.r_call, ctx.h, plan.live, torch.bfloat16)
            ws.barrier()
            return None, None, None
        ws.push(g.float().contiguous(), plan.fwd_rank, plan.fwd_pos, plan.src_live, "g_recv")
        ws.barrier()
        gy = ws.take("g_recv", plan.r_call, ctx.h, plan.live)
        ws.barrier()
        return gy.to(ctx.y_dtype), None, None


class _ZeroTailGrad(torch.autograd.Function):
    """Identity forward; backward zeroes the gradient rows at and past ``live`` (padding rows of a sparse
    routing map: the grouped GEMM leaves them unwritten, and the permute's backward would scatter them into
    real tokens and router probabilities)."""

    @staticmethod
    def forward(ctx, x, probs, live):
        ctx.save_for_backward(live)
        return x.view_as(x), (probs.view_as(probs) if probs is not None else None)

    @staticmethod
    def backward(ctx, gx, gp):
        (live,) = ctx.saved_tensors

        def zero(g):
            if g is None:
                return None
            keep = torch.arange(g.shape[0], device=g.device) < live
            return torch.where(keep.view(-1, *([1] * (g.dim() - 1))), g, torch.zeros((), dtype=g.dtype, device=g.device))

        return zero(gx), zero(gp), None


# ---------------------------------------------------------------------------
# Dispatcher patch.
# ---------------------------------------------------------------------------


def install() -> None:
    """Patch Megatron's AlltoAll dispatcher (idempotent); a no-op unless ``enabled()``."""
    if _STATE["patched"] or not enabled():
        return
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher as D

    orig = {
        name: getattr(D, name)
        for name in (
            "preprocess",
            "_maybe_dtoh_and_synchronize",
            "dispatch_postprocess",
            "combine_preprocess",
            "token_dispatch",
            "token_combine",
        )
    }

    def preprocess(self, routing_map):
        mode = _mode(self)
        if mode != "symm":
            out = orig["preprocess"](self, routing_map)
            self._dd_live = None
            if _sparse_routing() and not _inference_active() and not getattr(self, "drop_and_pad", False):
                if mode == "local":
                    # Sparse routing map (cross-fade zero-weight slots dropped): the static T x top-k output keeps
                    # padding rows past the live count; their gradients are zeroed at dispatch (no host sync).
                    self._dd_live = routing_map.sum()
                else:
                    # Megatron's AlltoAll splits need the exact row count (a host sync; stock path only).
                    self.num_out_tokens = int(routing_map.sum())
            return out
        # Dropless top-k: static output size, counts on the device only.
        self.num_out_tokens = routing_map.size(0) * self.config.moe_router_topk
        self._dd_counts = routing_map.sum(dim=0).to(torch.int32)
        return None

    def _maybe_dtoh_and_synchronize(self, point, tokens_per_expert=None):
        if _mode(self) is not None:
            return tokens_per_expert  # counts stay on the GPU; nothing waits for them
        return orig["_maybe_dtoh_and_synchronize"](self, point, tokens_per_expert)

    def token_dispatch(self, permutated_local_input_tokens, permuted_probs):
        mode = _mode(self)
        if mode is None:
            return orig["token_dispatch"](self, permutated_local_input_tokens, permuted_probs)
        if self.shared_experts is not None:
            self.shared_experts.wait_current_stream()
        if mode == "symm":
            ws = _workspace(self)
            if permutated_local_input_tokens.shape[-1] > ws.H:
                raise RuntimeError(
                    f"device_dispatch: dispatched rows have width {permutated_local_input_tokens.shape[-1]} > "
                    f"workspace width {ws.H} (moe_latent_size or hidden_size)"
                )
            plan = _Plan(ws, self._dd_counts, permutated_local_input_tokens.shape[0], sparse=_sparse_routing())
            self._dd_plan = plan
            self.tokens_per_expert = plan.tokens_per_expert
            out = _Dispatch.apply(
                permutated_local_input_tokens, permuted_probs, plan, _wire16(self, permutated_local_input_tokens)
            )
        else:
            out = (permutated_local_input_tokens, permuted_probs)
            if getattr(self, "_dd_live", None) is not None:
                out = _ZeroTailGrad.apply(permutated_local_input_tokens, permuted_probs, self._dd_live)
        if self.shared_experts is not None:
            self.shared_experts.linear_fc1_forward_and_act(out[0])
        return out

    def dispatch_postprocess(self, global_input_tokens, global_probs):
        mode = _mode(self)
        if mode is None:
            return orig["dispatch_postprocess"](self, global_input_tokens, global_probs)
        _log_once(
            f"mode{mode}{self.ep_size}",
            f"MoE dispatch without host syncs: ep={self.ep_size} ({mode}) "
            f"local_experts={self.num_local_experts}",
        )
        # Rows are already expert major (EP 1: permutation 1; EP > 1: the exchange
        # placed them), so the chunk sort of permutation 2 is not needed.
        tokens_per_expert = self.tokens_per_expert
        self.tokens_per_expert = None
        return global_input_tokens, tokens_per_expert, global_probs

    def combine_preprocess(self, hidden_states):
        if _mode(self) is None:
            return orig["combine_preprocess"](self, hidden_states)
        return hidden_states  # no unsort: the combine exchange restores permutation-1 order

    def token_combine(self, hidden_states, async_finish=True, allocate_on_comm_stream=True):
        mode = _mode(self)
        if mode is None:
            return orig["token_combine"](self, hidden_states, async_finish, allocate_on_comm_stream)
        if self.shared_experts is not None:
            self.shared_experts.wait_current_stream()
        out = (
            _Combine.apply(hidden_states, self._dd_plan, _skip_recompute_combine(self))
            if mode == "symm"
            else hidden_states
        )
        if self.shared_experts is not None:
            self.shared_experts.linear_fc2_forward(out)
            self.shared_experts.post_forward_comm()
        return out

    D.preprocess = preprocess
    D._maybe_dtoh_and_synchronize = _maybe_dtoh_and_synchronize
    D.token_dispatch = token_dispatch
    D.dispatch_postprocess = dispatch_postprocess
    D.combine_preprocess = combine_preprocess
    D.token_combine = token_combine
    _STATE["patched"] = True
