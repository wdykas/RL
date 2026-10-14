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
"""Engine-sourced q for block verification under batch invariance.

With batch-invariant kernels the inference engine's decode forward is bitwise
the training forward, so the true sampling law q of a drafted token is the
engine's own distribution:

* log q(x_i) of every drafted token is the engine's reported log-prob
  (``generated_log_probs``, already streamed with the draft);
* a full q row, needed only where the lazy scan computes a residual mass and
  at the residual sample, is ``log_softmax(W_head^(v) h_i)`` with h_i the
  engine's final hidden state (LM-head input, after the final norm) at that
  step and W_head^(v) the LM-head weight of the drafting version v.

So the verifier needs no scorer copy and no whole-model weight stash: the
engine logs h_i per drafted token (d x 2 bytes, ``HiddenLog``) and the learner
keeps only the LM head per pending version (V x d x 2 bytes). p comes from the
training model's own bf16 forward at theta_{k+1}.

Tensor parallelism (training TP = inference TP, as batch invariance requires):
the LM head is vocab-parallel, so each rank stashes only its shard
(V/TP x d), computes its logits shard of a row, and the full row is the
shards all-gathered in rank order followed by one fp32 log-softmax: what the
engine does (``ColumnParallelLinear`` with gathered output, then the
controller's ``log_softmax(logits.float())``) and what batch-invariant
training does (``_GatheredVocabLogprob``). All-gathers are copies, so the row
is bitwise the TP=1 row of the same logits.

This module holds the parts that need no Megatron: the engine-side hidden
store and per-step row mapping, the gathered log-prob rows, and the per-draft
verification rule.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import torch

# Sampling-params attribute that tags a draft request for hidden-state logging.
HIDDEN_KEY_ATTR = "ts_hidden_key"


def step_rows(
    request_ids: list[int],
    kv_offsets: list[int],
    query_lengths: list[int],
    lookup: Callable[[int], Optional[tuple[str, int]]],
) -> list[tuple[int, str, int]]:
    """Rows of one engine forward that sample a tagged request's next token.

    The forward produces one LM-head row per active request (request order),
    the row of its last input token, from which that request's next token is
    sampled. ``lookup(request_id)`` gives (draft key, prompt length) for
    tagged requests, else None. Returns (row, key, t): row ``row`` produced
    generated token ``t`` of draft ``key``, where t = (position of the last
    input token) + 1 - prompt length. Rows with t < 0 (an intermediate prefill
    chunk, which samples nothing) are skipped. Recomputation after an engine
    eviction rewrites the same t, so a draft's rows are keyed by position, not
    by arrival order.
    """
    out = []
    for row, (rid, kv, ql) in enumerate(zip(request_ids, kv_offsets, query_lengths)):
        info = lookup(int(rid))
        if info is None:
            continue
        key, plen = info
        t = int(kv) + int(ql) - plen
        if t >= 0:
            out.append((row, key, t))
    return out


class HiddenLog:
    """Final hidden states of tagged draft requests, by (key, generated index).

    Engine side. ``record`` copies the selected rows of a forward's LM-head
    input to (pinned, when CUDA is available) host memory without blocking
    the engine; ``pop`` returns rows 0..n-1 of a draft, or None if any is
    missing. Memory: d x 2 bytes (bf16) per drafted token, released as drafts
    are popped. Thread-safe (the engine records on its loop thread).
    """

    def __init__(self, chunk_rows: int = 8192):
        self.chunk_rows = chunk_rows
        self._lock = threading.Lock()
        self._rows: dict[str, dict[int, tuple[list, int]]] = {}
        self._closed: set[str] = set()
        self._cur: Optional[list] = None  # [buffer, used, live rows, event]
        self.recorded = 0

    def _alloc(self, k: int, like: torch.Tensor) -> tuple[list, int]:
        cur = self._cur
        if cur is None or cur[1] + k > cur[0].shape[0]:
            n = max(self.chunk_rows, k)
            buf = torch.empty(
                (n, like.shape[-1]),
                dtype=like.dtype,
                pin_memory=like.is_cuda,
            )
            cur = self._cur = [buf, 0, 0, None]
        off = cur[1]
        cur[1] += k
        return cur, off

    def record(self, entries: list[tuple[int, str, int]], hidden: torch.Tensor) -> None:
        """Store ``hidden[row]`` as token ``t`` of draft ``key`` for each entry."""
        with self._lock:
            entries = [e for e in entries if e[1] not in self._closed]
            if not entries:
                return
            idx = torch.tensor([r for r, _, _ in entries])
            if hidden.is_cuda:
                # Pinned and non-blocking: a pageable copy would make the host
                # wait for the forward that produced ``hidden``.
                idx = idx.pin_memory().to(hidden.device, non_blocking=True)
            sel = hidden.detach().index_select(0, idx)
            chunk, off = self._alloc(len(entries), sel)
            chunk[0][off : off + len(entries)].copy_(sel, non_blocking=sel.is_cuda)
            if sel.is_cuda:
                ev = torch.cuda.Event()
                ev.record()
                chunk[3] = ev
            for j, (_, key, t) in enumerate(entries):
                slots = self._rows.setdefault(key, {})
                old = slots.get(t)
                if old is not None:
                    old[0][2] -= 1
                slots[t] = (chunk, off + j)
                chunk[2] += 1
            self.recorded += len(entries)

    def pop(self, key: str, n: int) -> Optional[torch.Tensor]:
        """Rows 0..m-1 of draft ``key`` [m, d] (CPU): the longest logged prefix (m <= n).

        Closes the key: rows the engine records for it later are dropped.
        """
        with self._lock:
            self._closed.add(key)
            slots = self._rows.pop(key, {})
            for chunk, _ in slots.values():
                chunk[2] -= 1
            # The longest contiguous prefix of rows: a row the engine did not
            # log (or the source check rejected) truncates the draft there.
            m = 0
            while m < n and m in slots:
                m += 1
            if m == 0:
                return None if n else torch.empty(0, 0)
            for ev in {id(slots[t][0]): slots[t][0][3] for t in range(m)}.values():
                if ev is not None:
                    ev.synchronize()
            return torch.stack([slots[t][0][0][slots[t][1]] for t in range(m)])

    def take(self, key: str, lo: int, n: int) -> Optional[torch.Tensor]:
        """Rows lo..lo+m-1 of a running draft [m, d] (the longest logged run, m <= n).

        The rows leave the log; the key stays open (later rows keep being
        logged), so a draft can be taken in pieces while it is decoded.
        """
        with self._lock:
            slots = self._rows.get(key, {})
            m = 0
            while m < n and lo + m in slots:
                m += 1
            if m == 0:
                return None if n else torch.empty(0, 0)
            taken = [slots.pop(t) for t in range(lo, lo + m)]
            for chunk, _ in taken:
                chunk[2] -= 1
            for ev in {id(c[0]): c[3] for c, _ in taken}.values():
                if ev is not None:
                    ev.synchronize()
            return torch.stack([c[0][i] for c, i in taken])

    def close(self, key: str) -> None:
        """Drop a draft's rows and the rows the engine logs for it later."""
        self.pop(key, 0)

    def live_rows(self) -> int:
        with self._lock:
            return sum(len(s) for s in self._rows.values())


# ---------------------------------------------------------------------------
# Moving hidden states: generation worker -> object store -> verifying learner
# ---------------------------------------------------------------------------
#
# A draft's hidden states are d x 2 bytes per drafted token, but only drafts
# the lazy scan does not accept whole ever read a q row, and those read only
# the rows the scan visits (top candidates down to tau). So the generation
# worker puts each draft's rows of one drafting round into the Ray object
# store once, in blocks of ``HIDDEN_BLOCK_ROWS`` rows, and only the handles
# travel engine -> driver -> learner. The learner fetches a block the first
# time its lazy scan asks for a row in it (``HiddenRows``): fully accepted
# drafts fetch nothing, and the data moves from the generation node's object
# store to the learner's node without passing through the driver. The driver
# frees the objects (``free_hidden``) once the draft is verified or dropped.


@dataclass
class HiddenSeg:
    """Handle to one block of a draft's hidden states in the object store.

    ``ref``: ObjectRef of a [rows, d] tensor (put by the generation worker
    that drafted these rows; it owns the object), None once released. The
    driver holds only these handles, never the rows.
    """

    ref: Any
    rows: int
    nbytes: int

    def release(self) -> None:
        """Drop this handle's reference (``free_hidden`` also deletes the object)."""
        self.ref = None


@dataclass
class HiddenRefs:
    """A draft's hidden states as object-store block handles, in row order.

    Draft tuple index 7 under ``hidden_transport="object_store"``; a [g, d]
    tensor under ``"driver"``. A cohort draft chains the blocks of its
    drafting rounds; the handles are shared with the cohort dict, so releasing
    them clears the draft's references everywhere on the driver.
    """

    segs: list[HiddenSeg] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return sum(s.rows for s in self.segs)

    @property
    def nbytes(self) -> int:
        return sum(s.nbytes for s in self.segs)

    def release(self) -> None:
        for s in self.segs:
            s.release()

    @classmethod
    def from_blocks(cls, blocks: list[tuple]) -> "HiddenRefs":
        """From ``put_hidden``'s [(ref, rows, bytes)]."""
        return cls([HiddenSeg(*b) for b in blocks])


def ray_free(refs: list) -> None:
    """Delete objects from every object store now (Ray internal API).

    Called by the driver once the learner calls that could read them have
    returned, so a lingering Python reference anywhere (driver, owner,
    learner) cannot keep a step's hidden states alive.
    """
    from ray._private import internal_api

    internal_api.free(refs, local_only=False)


def free_hidden(hiddens, free: Optional[Callable[[list], None]] = None) -> int:
    """Release the handles of ``HiddenRefs`` in ``hiddens`` and free their objects.

    Other entries (tensors, None) are ignored. Returns the bytes released.
    """
    refs, nbytes = [], 0
    for h in hiddens:
        if not isinstance(h, HiddenRefs):
            continue
        for s in h.segs:
            if s.ref is not None:
                refs.append(s.ref)
                nbytes += s.nbytes
                s.release()
    if refs:
        (free or ray_free)(refs)
    return nbytes


# Rows per object-store object. The learner fetches whole blocks, so this
# bounds what a rejected draft pulls beyond the rows its lazy scan visits
# (at most one partial block per visited run); 256 rows are 2 MB at d = 4096
# and 3.7 MB at d = 7168 (bf16), well above Ray's inline-object size. Each
# block is one ray.put, so larger blocks cut per-put overhead at the step
# deadline (THUNDERSYNC_HIDDEN_BLOCK_ROWS overrides it for measurements).
HIDDEN_BLOCK_ROWS = int(os.environ.get("THUNDERSYNC_HIDDEN_BLOCK_ROWS", "256"))


def put_hidden(
    h: torch.Tensor,
    put: Optional[Callable[[Any], Any]] = None,
    block_rows: int = HIDDEN_BLOCK_ROWS,
) -> list[tuple]:
    """Generation-worker side: put rows [n, d] in the object store once.

    One object per ``block_rows`` rows. Returns [(ref, rows, bytes)] per
    block, the only thing sent back to the driver (``HiddenRefs.from_blocks``).
    """
    if put is None:
        import ray

        put = ray.put
    out = []
    for a in range(0, int(h.shape[0]), block_rows):
        b = h[a : a + block_rows].contiguous()
        out.append((put(b), int(b.shape[0]), b.numel() * b.element_size()))
    return out


def cat_hidden(parts: list) -> Any:
    """A draft's hidden states from its per-segment parts.

    Tensors (``hidden_transport="driver"``) are concatenated, skipping empty
    segments; ``HiddenRefs`` are chained (no data is touched).
    """
    if parts and isinstance(parts[0], HiddenRefs):
        return HiddenRefs([s for p in parts for s in p.segs if s.rows])
    parts = [h for h in parts if h.numel()]
    return torch.cat(parts) if parts else torch.empty(0, 0)


def hidden_num_rows(h: Any) -> int:
    """Rows of draft tuple index 7 (tensor or ``HiddenRefs``)."""
    return h.rows if isinstance(h, HiddenRefs) else int(h.shape[0])


def hidden_nbytes(h: Any) -> int:
    if h is None:
        return 0
    if isinstance(h, HiddenRefs):
        return h.nbytes
    return h.numel() * h.element_size()


class HiddenRows:
    """Learner side: rows of one draft's hidden states, fetched on first use.

    ``src`` is a [g, d] tensor (driver transport: already here) or a
    ``HiddenRefs``. ``__call__(positions)`` returns rows [k, d] (CPU). With
    refs, each block that holds a requested position is fetched once
    (``get``, ``ray.get`` by default) and cached for this draft; untouched
    blocks are never fetched. ``fetched_bytes`` counts what was pulled.
    Tensor-parallel ranks of a replica run the same lazy scan, so they ask
    for the same positions in the same order and fetch the same objects
    (immutable, so every rank sees identical bytes; ranks on one node share
    the node's copy in the object store).
    """

    def __init__(self, src: Any, get: Optional[Callable[[Any], Any]] = None):
        self.src = src
        self.fetched_bytes = 0
        self._get = get
        if isinstance(src, HiddenRefs):
            self._starts = []
            a = 0
            for s in src.segs:
                self._starts.append(a)
                a += s.rows
            self._cache: dict[int, torch.Tensor] = {}

    def _block(self, k: int) -> torch.Tensor:
        h = self._cache.get(k)
        if h is None:
            seg = self.src.segs[k]
            if seg.ref is None:
                raise RuntimeError("hidden states of this draft were already released")
            get = self._get
            if get is None:
                import ray

                get = ray.get
            h = get(seg.ref)
            if h.shape[0] != seg.rows:
                raise RuntimeError(
                    f"hidden block has {h.shape[0]} rows, expected {seg.rows}"
                )
            self._cache[k] = h
            self.fetched_bytes += seg.nbytes
        return h

    def __call__(self, positions: list[int]) -> torch.Tensor:
        if not isinstance(self.src, HiddenRefs):
            return self.src[positions]
        import bisect

        out = []
        for t in positions:
            k = bisect.bisect_right(self._starts, t) - 1
            out.append(self._block(k)[t - self._starts[k]])
        return torch.stack(out)


# Rows per full-vocabulary log-softmax when log-probs of many positions are
# gathered at once (bounds the [rows, V] fp32 temporaries, as training's
# gathered TP log-probs do).
GATHER_CHUNK_ROWS = 4096

VocabGather = Callable[[torch.Tensor], torch.Tensor]


def no_gather(x: torch.Tensor) -> torch.Tensor:
    """The vocab gather of tensor-parallel size 1."""
    return x


def gathered_log_softmax(
    local: torch.Tensor, gather: VocabGather, vocab_limit: int
) -> torch.Tensor:
    """Full-vocabulary fp32 log-prob rows of vocab-sharded logits [k, V/TP].

    ``gather`` concatenates the tensor-parallel ranks' shards along the last
    dimension in rank (= vocabulary) order (``no_gather`` at TP=1): the
    engine's logits all-gather. Then the engine's ``raw_logprobs`` path: fp32
    cast and one full-row ``log_softmax``. A collective at TP>1: every rank
    of the group must call it with the same row count.
    """
    z = gather(local.contiguous())
    return torch.log_softmax(z[..., :vocab_limit].float(), -1)


class GatheredLogprobRows:
    """Log-prob rows of one draft's vocab-sharded logits, gathered on demand.

    ``local`` [n, V/TP]: this rank's logits shard per position. Rows are
    computed by ``gathered_log_softmax`` only where asked (lazy scan) and,
    for the drafted tokens' log-probs, ``GATHER_CHUNK_ROWS`` rows at a time.
    Under batch-invariant kernels a row's log-softmax does not depend on how
    many rows run together, so both give the bits of training's gathered
    log-probs. Memory: the local shard plus one chunk of full rows, instead of
    [n, V] fp32. Every call is a collective at TP>1.
    """

    def __init__(
        self,
        local: torch.Tensor,
        gather: VocabGather,
        vocab_limit: int,
        chunk_rows: int = GATHER_CHUNK_ROWS,
    ):
        self.local = local
        self.gather = gather
        self.vocab_limit = vocab_limit
        self.chunk_rows = chunk_rows

    def __call__(self, idx: torch.Tensor) -> torch.Tensor:
        return gathered_log_softmax(
            self.local[idx.to(self.local.device)], self.gather, self.vocab_limit
        )

    def token_logprobs(self, tokens: torch.Tensor) -> torch.Tensor:
        """[k] fp32 log-prob of ``tokens[j]`` at row j, for rows 0..k-1."""
        k = tokens.numel()
        out = []
        for a in range(0, k, self.chunk_rows):
            b = min(k, a + self.chunk_rows)
            lp = gathered_log_softmax(self.local[a:b], self.gather, self.vocab_limit)
            out.append(lp.gather(-1, tokens[a:b].view(-1, 1).to(lp.device))[:, 0])
            del lp
        if not out:
            return torch.zeros(0, dtype=torch.float32, device=self.local.device)
        return torch.cat(out)


def head_stash_bytes(
    vocab: int, hidden: int, tp: int, versions: int = 1, elem_bytes: int = 2
) -> int:
    """GPU bytes of the LM-head stash on one tensor-parallel rank.

    Each rank keeps its vocab-parallel shard per pending version: ceil(V/TP)
    x d elements (Megatron pads V to a multiple of TP, so V/TP exactly).
    """
    return versions * (-(-vocab // tp)) * hidden * elem_bytes


def residual_rows_needed(n_eval: int, tau: int, g: int) -> int:
    """Q rows a lazy verification computed: residual masses plus the emit row."""
    return n_eval + (tau < g)


def verify_engine_q(
    p_logprobs: Union[torch.Tensor, GatheredLogprobRows],
    draft: torch.Tensor,
    engine_logq: torch.Tensor,
    q_logprob_rows: Callable[[torch.Tensor], torch.Tensor],
    generator: Optional[torch.Generator] = None,
    null_m: float = 1.0,
) -> tuple[int, int, torch.Tensor, int]:
    """Block-verify one draft with p from training and q from the engine.

    ``p_logprobs``: [g+1, V] fp32 log-probs of the training model at
    theta_{k+1} (log-softmax rows, as training computes them), or a
    ``GatheredLogprobRows`` over its vocab-sharded logits (tensor parallelism:
    rows gathered on demand, the same values).
    ``engine_logq``: [g] fp32 log q(x_i), the engine's log-probs of the drafted
    tokens. ``q_logprob_rows(idx)``: [k, V] fp32 q log-prob rows at positions
    ``idx`` (reconstructed from the engine's hidden states), called only where
    the lazy scan needs a residual mass and for the residual sample.

    Same rule and generator calls as ``block_verify_logits`` (one ``rand(g+1)``,
    one ``multinomial``), with log-prob rows standing in for logits (their
    logsumexp is 0). Returns (tau, y, logp, residual masses computed); logp are
    entries of ``p_logprobs``, i.e. training's own log-probs of the kept tokens.
    ``null_m`` > 1: null-augmented block verification (y = ``NULL_TOKEN``: BOT,
    the draft yields no output).
    """
    from thundersync_rl.block_verification_fused import block_verify_stats_lazy

    g = draft.numel()
    if isinstance(p_logprobs, GatheredLogprobRows):
        dev = p_logprobs.local.device
        lpx = p_logprobs.token_logprobs(draft.to(dev))
        p_rows = p_logprobs
    else:
        dev = p_logprobs.device
        lpx = p_logprobs[torch.arange(g, device=dev), draft]

        def p_rows(idx):
            return p_logprobs[idx]

    lqx = engine_logq.to(device=dev, dtype=torch.float32)
    zeros = torch.zeros(g + 1, dtype=torch.float32, device=dev)
    return block_verify_stats_lazy(
        lpx, lqx, zeros, zeros[:g], p_rows, q_logprob_rows, generator, null_m
    )


def verify_engine_q_grid(
    p_logprobs: Union[torch.Tensor, GatheredLogprobRows],
    draft: torch.Tensor,
    engine_logq: torch.Tensor,
    q_logprob_rows: Callable[[torch.Tensor], torch.Tensor],
    generator: Optional[torch.Generator],
    envelopes: tuple[float, ...],
) -> tuple[dict[float, tuple[int, int, Optional[float]]], torch.Tensor, int]:
    """``verify_engine_q`` at every ⊥-BV envelope in ``envelopes``, one set of draws.

    Returns ({M: (tau, y, log p(y) or None for BOT)}, log p of the drafted
    tokens [g], residual masses computed); see
    ``block_verification_fused.block_verify_stats_lazy_grid``.
    """
    from thundersync_rl.block_verification_fused import block_verify_stats_lazy_grid

    g = draft.numel()
    if isinstance(p_logprobs, GatheredLogprobRows):
        dev = p_logprobs.local.device
        lpx = p_logprobs.token_logprobs(draft.to(dev))
        p_rows = p_logprobs
    else:
        dev = p_logprobs.device
        lpx = p_logprobs[torch.arange(g, device=dev), draft]

        def p_rows(idx):
            return p_logprobs[idx]

    lqx = engine_logq.to(device=dev, dtype=torch.float32)
    zeros = torch.zeros(g + 1, dtype=torch.float32, device=dev)
    out, n_eval = block_verify_stats_lazy_grid(
        lpx, lqx, zeros, zeros[:g], p_rows, q_logprob_rows, generator, envelopes
    )
    return out, lpx, n_eval
