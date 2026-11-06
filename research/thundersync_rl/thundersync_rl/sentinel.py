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
"""Always-on engine-vs-learner weight sentinel.

Exactness depends on three facts no other always-on check covers (accepted
draft tokens store the learner's own p, so the log-prob mismatch check only
sees fresh and tail tokens):

(a) after every refit, each engine rank holds bitwise the learner's weights
    of the same version (theta_k, or theta_hat_{k+1} for momentum drafts),
    after the dtype cast the engine stores them in (fp16 GEMM weights);
(b) the LM head stashed on the learner for q-row rebuilds is the head the
    engine drafted with;
(c) MoE ``expert_bias`` buffers never change (bias updates are rejected).

Each side digests its own shards: per tensor, two 64-bit sums over the raw
bits (plain and position-weighted), no gather, O(params / rank) reads. Tensor
names are canonicalized so both sides agree whatever their pipeline / expert
layout: ``module.`` wrappers stripped, decoder layer indices made global
(``layer_number``), local expert indices made global (expert-parallel rank x
local experts + local index). Digests are keyed by (canonical name, tensor-
parallel rank); within a side, replicas (data / expert-data parallel) must
agree; across sides, every tensor present on both must agree bitwise, with the
learner's tensor first cast to the engine's storage dtype. A mismatch raises
``WeightSentinelError`` naming the tensors.
"""

from __future__ import annotations

import os
import time

import re
from typing import Any, Callable, Iterable, Iterator, Optional

import torch

# Elements per digest chunk (bounds the int64 temporaries).
DIGEST_CHUNK = 1 << 24
_POS_MOD = 65521  # position weight (i mod p) + 1: catches permutations
_BITS_DTYPE = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
_MASK64 = (1 << 64) - 1

Digest = tuple[str, int, int, int]  # (dtype, numel, sum, position-weighted sum)


class WeightSentinelError(RuntimeError):
    """Engine and learner weights (or replicas of one side) differ."""


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def name_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


@torch.no_grad()
def tensor_digest(t: torch.Tensor, dtype: Optional[torch.dtype] = None) -> Digest:
    """Bitwise digest of ``t`` (cast to ``dtype`` first, chunk by chunk, if given)."""
    flat = t.detach().reshape(-1)
    out_dtype = dtype if dtype is not None else flat.dtype
    n = flat.numel()
    s1 = torch.zeros((), dtype=torch.int64, device=flat.device)
    s2 = torch.zeros((), dtype=torch.int64, device=flat.device)
    for a in range(0, n, DIGEST_CHUNK):
        b = min(n, a + DIGEST_CHUNK)
        x = flat[a:b]
        if x.dtype != out_dtype:
            x = x.to(out_dtype)
        if x.dtype == torch.bool:
            x = x.to(torch.int8)
        bits = x.contiguous().view(_BITS_DTYPE[x.element_size()]).to(torch.int64)
        w = torch.arange(a, b, device=x.device, dtype=torch.int64) % _POS_MOD + 1
        s1 += bits.sum()
        s2 += (bits * w).sum()
    return dtype_name(out_dtype), n, int(s1) & _MASK64, int(s2) & _MASK64


def _bits64(x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """``x`` (flat) cast to ``out_dtype``, its bit patterns widened to int64."""
    if x.dtype != out_dtype:
        x = x.to(out_dtype)
    if x.dtype == torch.bool:
        x = x.to(torch.int8)
    return x.contiguous().view(_BITS_DTYPE[x.element_size()]).to(torch.int64)


@torch.no_grad()
def digest_many(
    items: Iterable[tuple[str, torch.Tensor, Optional[torch.dtype]]],
    pack_elems: int = 1 << 16,
    chunk: Optional[int] = None,
) -> dict[str, Digest]:
    """``{key: tensor_digest(t, dtype)}`` for every ``(key, t, dtype)``, bitwise equal.

    ``tensor_digest`` synchronizes the host once per tensor and launches ~10
    kernels per tensor; a full-Nano rank has ~6k tensors, so one sentinel check
    took ~2 s on the refit's critical path (``sentinel/time_s``, ab2 runs).
    Here tensors smaller than ``pack_elems`` are packed (same device) into
    buffers of at most ``pack_elems`` elements and digested together with
    integer segment sums (``index_add_`` on int64: exact and order-free, the
    same sums mod 2**64), larger ones chunk by chunk as before, and the host
    reads every sum with one transfer per device. O(params / rank) work, no
    gather, one sync.

    ``pack_elems`` is 2**16: the segment sums are int64 ``index_add_`` atomics on
    one address per tensor, so a packed tensor of n elements serializes n
    atomics (2**22 made every Super latent expert slice, 1024 x 2688 = 2.75M
    elements, a single-segment pack: 4.5 ms per tensor vs 0.16 ms chunked, a
    22-27 s sentinel per refit at Super-L22 S2). Small tensors (norms, biases,
    routers) still share packs; digests are per tensor, so the threshold never
    changes a digest.
    """
    chunk = chunk or DIGEST_CHUNK  # read at call time
    out_meta: dict[str, tuple[str, int]] = {}
    sums: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    packs: dict[torch.device, list[tuple[str, torch.Tensor]]] = {}
    pos_cache: dict[torch.device, torch.Tensor] = {}

    def positions(device: torch.device, n: int) -> torch.Tensor:
        base = pos_cache.get(device)
        if base is None or base.numel() < n:
            base = torch.arange(max(n, 1), device=device, dtype=torch.int64)
            pos_cache[device] = base
        return base[:n]

    for key, t, dtype in items:
        flat = t.detach().reshape(-1)
        out_dtype = dtype if dtype is not None else flat.dtype
        n = flat.numel()
        out_meta[key] = (dtype_name(out_dtype), n)
        if n < pack_elems:
            packs.setdefault(flat.device, []).append((key, _bits64(flat, out_dtype)))
            continue
        s1 = torch.zeros((), dtype=torch.int64, device=flat.device)
        s2 = torch.zeros((), dtype=torch.int64, device=flat.device)
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            bits = _bits64(flat[a:b], out_dtype)
            w = (positions(flat.device, b - a) + a) % _POS_MOD + 1
            s1 += bits.sum()
            s2 += (bits * w).sum()
        sums[key] = (s1, s2)

    for device, group in packs.items():
        start = 0
        while start < len(group):
            stop, total = start, 0
            while stop < len(group) and (stop == start or total + group[stop][1].numel() <= pack_elems):
                total += group[stop][1].numel()
                stop += 1
            part = group[start:stop]
            lens = torch.tensor([b.numel() for _, b in part], device=device, dtype=torch.int64)
            seg = torch.repeat_interleave(
                torch.arange(len(part), device=device), lens, output_size=total
            )  # output_size: no host sync per pack
            offsets = torch.cumsum(lens, 0) - lens
            bits = torch.cat([b for _, b in part]) if len(part) > 1 else part[0][1]
            w = (positions(device, total) - offsets[seg]) % _POS_MOD + 1
            s1 = torch.zeros(len(part), dtype=torch.int64, device=device).index_add_(0, seg, bits)
            s2 = torch.zeros(len(part), dtype=torch.int64, device=device).index_add_(0, seg, bits * w)
            for j, (key, _) in enumerate(part):
                sums[key] = (s1[j], s2[j])
            start = stop

    keys = list(sums)
    by_device: dict[torch.device, list[str]] = {}
    for k in keys:
        by_device.setdefault(sums[k][0].device, []).append(k)
    out: dict[str, Digest] = {}
    for device, ks in by_device.items():
        vals = torch.stack([torch.stack(sums[k]) for k in ks]).cpu().tolist()  # one sync
        for k, (a, b) in zip(ks, vals):
            name, n = out_meta[k]
            out[k] = (name, n, int(a) & _MASK64, int(b) & _MASK64)
    return {k: out[k] for k in out_meta}


_LAYER = re.compile(r"(^|\.)layers\.(\d+)(?=\.|$)")
_LOCAL_EXPERT = re.compile(r"(\.experts\.local_experts\.)(\d+)(\.)")
_GROUPED_EXPERT = re.compile(r"(\.experts\.linear_fc\d\.(?:weight|bias))(\d+)$")


def _strip(name: str) -> str:
    while name.startswith("module."):
        name = name[len("module."):]
    return name.replace(".module.", ".")


def _chunks(model: Any) -> list:
    return list(model) if isinstance(model, (list, tuple)) else [model]


def _canon_fn(chunk: Any, ep_rank: int):
    """name -> canonical name for the parameters of one model chunk."""
    modules = dict(chunk.named_modules())
    layer_map: dict[str, int] = {}
    experts_local: dict[str, int] = {}
    for name, mod in modules.items():
        m = _LAYER.search(name)
        if m is not None and m.end() == len(name):
            ln = getattr(mod, "layer_number", None)
            if isinstance(ln, int):
                layer_map[name] = ln - 1  # Megatron layer numbers are global, 1-based
        if name.endswith("experts"):
            nl = getattr(mod, "num_local_experts", None)
            if isinstance(nl, int):
                experts_local[name] = nl

    def canon(name: str) -> str:
        # Global expert index (needs the module path before renaming layers).
        for pat in (_LOCAL_EXPERT, _GROUPED_EXPERT):
            m = pat.search(name)
            if m is None:
                continue
            owner = name[: name.index(".experts.") + len(".experts")]
            nl = experts_local.get(owner)
            if nl is not None:
                j = ep_rank * nl + int(m.group(2))
                if pat is _LOCAL_EXPERT:
                    name = name[: m.start(2)] + str(j) + name[m.end(2):]
                else:
                    name = name[: m.start(2)] + str(j)
        # Global layer index: longest matching module prefix first.
        out, pos = [], 0
        for m in _LAYER.finditer(name):
            mod_name = name[: m.end()]
            if mod_name in layer_map:
                out.append(name[pos : m.start(2)] + str(layer_map[mod_name]))
                pos = m.end(2)
        out.append(name[pos:])
        return _strip("".join(out))

    return canon


DEBUG_FLIP_ENV = "THUNDERSYNC_DEBUG_SENTINEL_FLIP"


@torch.no_grad()
def debug_flip_byte(named: list, call_no: int, environ: Optional[dict] = None) -> Optional[str]:
    """Negative control (test only, ``THUNDERSYNC_DEBUG_SENTINEL_FLIP=<n>``): at the n-th engine digest of this rank,
    flip the low bit of one byte of the first held weight in place, i.e. a corrupted refit the weight sentinel must
    report. The weight stays corrupted (later decodes use it). Returns the tensor name, else None."""
    v = (os.environ if environ is None else environ).get(DEBUG_FLIP_ENV, "")
    if not v or int(v) != int(call_no) or not named:
        return None
    name, t = named[0]
    flat = t.detach().reshape(-1)[:1]
    flat.view(torch.uint8)[0] ^= 1
    return name


def canonical_named_tensors(
    model: Any, ep_rank: int = 0, include_buffers: bool = True
) -> Iterator[tuple[str, torch.Tensor]]:
    """(canonical name, tensor) of every parameter (and ``expert_bias`` buffer).

    Shared tensors (tied embeddings) are listed once, under their first name.
    """
    seen: set[tuple[int, int, torch.dtype]] = set()
    for chunk in _chunks(model):
        canon = _canon_fn(chunk, ep_rank)
        items: list[tuple[str, torch.Tensor]] = list(chunk.named_parameters())
        if include_buffers:
            items += [(n, b) for n, b in chunk.named_buffers() if n.endswith("expert_bias")]
        for name, t in items:
            key = (t.data_ptr(), t.numel(), t.dtype)
            if key in seen:
                continue
            seen.add(key)
            yield canon(name), t


# Compute tensors are digested under "<canonical source parameter>#<kind>"; the
# learner digests the source parameter cast to the engine tensor's dtype under the
# same key, so they compare like parameters.
ALIAS = "#"


def compute_tensors(model: Any, ep_rank: int = 0) -> Iterator[tuple[str, torch.Tensor, bool]]:
    """The tensors the engine computes with besides its parameters.

    Yields (``<source param>#buf``, tensor, aliased) for every expert of every
    grouped-MLP module holding concatenated expert weights (``_fc1_weight`` /
    ``_fc2_weight`` [E, N, K]: MCore's inference concatenation, or the fp16
    expert buffers of ``fp16_moe._ensure_fp16_expert_weights``), the storage the
    expert kernels read. ``aliased``: the parameter's data is a view of that
    slice (refit writes land there); a non-aliased buffer is a separate copy
    that must still equal the parameter.
    """
    for chunk in _chunks(model):
        canon = _canon_fn(chunk, ep_rank)
        for mname, mod in chunk.named_modules():
            for lin_name, buf_name in (("linear_fc1", "_fc1_weight"), ("linear_fc2", "_fc2_weight")):
                buf = getattr(mod, buf_name, None)
                lin = getattr(mod, lin_name, None)
                if not isinstance(buf, torch.Tensor) or lin is None or buf.dim() < 2:
                    continue
                for i in range(buf.shape[0]):
                    p = getattr(lin, f"weight{i}", None)
                    if not isinstance(p, torch.Tensor) or tuple(p.shape) != tuple(buf.shape[1:]):
                        continue
                    src = canon(f"{mname}.{lin_name}.weight{i}" if mname else f"{lin_name}.weight{i}")
                    aliased = p.data_ptr() == buf[i].data_ptr() and p.dtype == buf.dtype
                    yield f"{src}{ALIAS}buf", buf[i], aliased


def digest_named(
    named: Iterable[tuple[str, torch.Tensor]],
    dtypes: Optional[dict[str, str]] = None,
    chunk: Optional[int] = None,
) -> dict[str, Digest]:
    """Digest every tensor; ``dtypes[name]``: cast to the other side's storage dtype (``chunk``: elements per
    int64 temporary; does not change a digest).

    ``dtypes`` keys ``<name>#<kind>`` (engine compute tensors) are digested from
    the tensor ``<name>`` cast to that dtype, under the alias key.
    """
    named = list(named)
    items = []
    for name, t in named:
        want = None if dtypes is None or name not in dtypes else name_dtype(dtypes[name])
        items.append((name, t, want))
    if dtypes:
        by_name = dict(named)
        for key, dt in dtypes.items():
            if ALIAS in key:
                src = by_name.get(key.split(ALIAS, 1)[0])
                if src is not None:
                    items.append((key, src, name_dtype(dt)))
    # Same digests as tensor_digest per tensor, one host sync (digest_many).
    return digest_many(items, chunk=chunk)


def shard_rank(name: str, tp: Any) -> int:
    """The tensor-parallel rank ``name`` is sharded by on a rank keyed ``tp``.

    ``tp``: an int (dense TP rank) or ``(dense TP rank, expert TP rank)``. Expert
    tensors (``.experts.``, incl. ``#buf`` compute buffers) use the expert TP rank:
    with expert_tensor_parallel_size 1 the dense TP rank that holds an expert depends
    on the EP size, which may differ between engine and learner.
    """
    if isinstance(tp, (tuple, list)):
        return int(tp[1]) if ".experts." in name else int(tp[0])
    return int(tp)


def merge_side(
    ranks: list[tuple[Any, dict[str, Digest]]], side: str
) -> dict[tuple[str, int], Digest]:
    """One side's digests keyed by (name, tp rank); replicas must agree bitwise."""
    merged: dict[tuple[str, int], Digest] = {}
    bad = []
    for tp, digests in ranks:
        for name, d in digests.items():
            k = (name, shard_rank(name, tp))
            if k in merged and merged[k] != d:
                bad.append(f"{name}[tp{k[1]}]")
            merged.setdefault(k, d)
    if bad:
        raise WeightSentinelError(
            f"weight sentinel: {side} replicas disagree on {len(bad)} tensors: {bad[:20]}"
        )
    return merged


# Canonical tensor names that may legitimately exist on one side only (regular expressions, full match). A
# one-sided (name, tp) key is a tensor the sentinel never compares, e.g. experts keyed by
# the dense TP rank at TP > 1 with engine EP != learner EP. Empty: every Nano / Super run logged so far that had no
# keying bug shows engine_only == learner_only == 0. Add an entry only for a tensor that is
# shown to exist on one side by design (e.g. a head the engine does not load), with the run that showed it.
ONE_SIDED_ALLOWLIST: tuple[str, ...] = ()


def one_sided(
    e: dict[tuple[str, int], Digest], l_: dict[tuple[str, int], Digest]
) -> tuple[list[str], list[str], int]:
    """(engine-only, learner-only) ``name[tpN]`` keys outside ``ONE_SIDED_ALLOWLIST``, and the allowed count."""
    allow = [re.compile(p) for p in ONE_SIDED_ALLOWLIST]
    out: list[list[str]] = [[], []]
    allowed = 0
    for side, keys in enumerate((set(e) - set(l_), set(l_) - set(e))):
        for name, tp in sorted(keys):
            if any(a.fullmatch(name) for a in allow):
                allowed += 1
            else:
                out[side].append(f"{name}[tp{tp}]")
    return out[0], out[1], allowed


def compare(
    engine: list[tuple[int, dict[str, Digest]]],
    learner: list[tuple[int, dict[str, Digest]]],
    label: str,
) -> dict[str, float]:
    """Raise ``WeightSentinelError`` on any mismatch or uncompared tensor; return coverage counters.

    Coverage is part of the check: a (name, tp) key present on one side only is a tensor that is never compared,
    so it raises unless its name is in ``ONE_SIDED_ALLOWLIST`` (run rule 1).
    """
    e, l_ = merge_side(engine, "engine"), merge_side(learner, "learner")
    common = sorted(set(e) & set(l_))
    if not common:
        raise WeightSentinelError(
            f"weight sentinel ({label}): no tensor name in common between engine "
            f"({len(e)}) and learner ({len(l_)}); e.g. engine {sorted(e)[:5]} "
            f"learner {sorted(l_)[:5]} (vacuous check)"
        )
    bad = [k for k in common if e[k] != l_[k]]
    if bad:
        detail = [
            f"{n}[tp{tp}] engine={e[(n, tp)]} learner={l_[(n, tp)]}" for n, tp in bad[:20]
        ]
        raise WeightSentinelError(
            f"weight sentinel ({label}): {len(bad)}/{len(common)} tensors differ "
            f"between engine and learner: " + "; ".join(detail)
        )
    e_only, l_only, allowed = one_sided(e, l_)
    if e_only or l_only:
        raise WeightSentinelError(
            f"weight sentinel ({label}): {len(e_only)} engine-only and {len(l_only)} learner-only tensor keys "
            f"are never compared ({len(common)} compared): engine-only {e_only[:10]} learner-only {l_only[:10]} "
            f"(keys disagree between the sides' layouts; ONE_SIDED_ALLOWLIST lists names allowed on one side)"
        )
    return {
        "sentinel/tensors": float(len(common)),
        "sentinel/elems": float(sum(e[k][1] for k in common)),
        "sentinel/engine_only": float(len(set(e) - set(l_))),
        "sentinel/learner_only": float(len(set(l_) - set(e))),
        "sentinel/expert_bias": float(sum(1 for n, _ in common if n.endswith("expert_bias"))),
        "sentinel/one_sided_allowed": float(allowed),
    }


def _sum_cov(covs) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in covs:
        for k, v in (c or {}).items():
            out[k] = out.get(k, 0) + int(v)
    return out


NOT_CHECKED = (
    "per-call casts that are not stored (bf16 params cast to fp16 inside each GEMM when the "
    "engine does not hold fp16 weights, e.g. refit_transport mcore/m2n: a deterministic "
    "function of the checked parameter)",
    "the hi/lo split of the fp32 LM head (it splits activations; the head weight itself is checked)",
    "KV cache, Mamba conv/SSM state and CUDA-graph static activation buffers (activations, not weights)",
    "the learner fp16 expert cache while theta_hat is loaded (it keeps theta_k's casts by design)",
)


def coverage_report(label, cov, e_cov, l_cov, e_only, l_only) -> str:
    """What the sentinel compares and what it does not (printed at the first check)."""
    lines = [
        f"[sentinel] coverage at {label}:",
        f"[sentinel]   checked: {int(cov['sentinel/tensors'])} engine/learner tensor pairs "
        f"({int(cov['sentinel/elems'])} elements, all ranks), of which "
        f"{int(cov.get('sentinel/compute_buffers', 0))} expert compute buffers (#buf, the storage "
        f"the expert kernels read; {e_cov.get('compute_buffers_aliased', 0)}/"
        f"{e_cov.get('compute_buffers', 0)} rank-slices alias their parameter) and "
        f"{int(cov['sentinel/expert_bias'])} expert_bias buffers; engine parameters held in fp16 "
        f"(GEMM / router / head weights, compared after the learner's fp16 cast): "
        f"{e_cov.get('fp16_params', 0)} rank-tensors",
        f"[sentinel]   engine-only (not compared) {len(e_only)}: {e_only[:12]}",
        f"[sentinel]   learner-only (not compared) {len(l_only)}: {l_only[:12]}",
    ]
    lines += [f"[sentinel]   not checked: {x}" for x in NOT_CHECKED]
    return "\n".join(lines)


def unmatched(
    engine: list[tuple[int, dict[str, Digest]]],
    learner: list[tuple[int, dict[str, Digest]]],
) -> tuple[list[str], list[str]]:
    """``name[tpN]`` keys present on one side only (for the coverage report; the comparison is keyed by (name,
    tp rank), so a name present on both sides under different TP ranks is still uncompared)."""
    e = {f"{n}[tp{tp}]" for tp, d in engine for n in d}
    l_ = {f"{n}[tp{tp}]" for tp, d in learner for n in d}
    return sorted(e - l_), sorted(l_ - e)


def engine_dtypes(engine: list[tuple[int, dict[str, Digest]]]) -> dict[str, str]:
    """Storage dtype of every engine tensor (the learner casts to it)."""
    out: dict[str, str] = {}
    for _, digests in engine:
        for n, d in digests.items():
            out.setdefault(n, d[0])
    return out


# ---------------------------------------------------------------------------
# Driver side (Ray): run both sides after a refit and compare.
# ---------------------------------------------------------------------------


def _la_kw(lookahead: bool) -> dict[str, bool]:
    """``lookahead=True`` only when set (workers without lookahead drafts keep their signature)."""
    return {"lookahead": True} if lookahead else {}


def _async_enabled() -> bool:
    return os.environ.get("THUNDERSYNC_SENTINEL_ASYNC", "0") == "1"


class WeightSentinel:
    """Driver helper: engine-vs-learner digests after every refit (always on).

    ``check(generation, label)`` digests every engine rank of ``generation``
    and every learner rank (cast to the engine dtypes) and compares them; the
    learner must hold the weights that were just refit. ``engine_digest`` /
    ``check_engine_against`` compare an engine with a digest taken earlier
    (momentum host stage reloaded). Coverage counters accumulate per step in
    ``stats`` (``pop_stats``).
    """

    def __init__(self, policy: Any, enabled: bool = True) -> None:
        self.policy = policy
        self.enabled = enabled
        self._sampled: dict = {}
        self.stats: dict[str, float] = {}
        self._reported_coverage = False
        # Engine storage dtypes per engine (they do not change across refits):
        # lets the learner digest run concurrently with the engine's.
        self._dtypes: dict[int, dict[str, str]] = {}
        # Compared-tensor count per (engine, kind): constant across refits (run rule 1). A change means a refit
        # added or dropped tensors from the comparison (e.g. a layout or keying change mid-run).
        self._tensor_counts: dict[tuple[int, str], int] = {}
        # THUNDERSYNC_SENTINEL_ASYNC=1 (sentinel_async): learner digests in flight, compared at the next check of the
        # same engine or at ``pop_stats`` (step end): token -> (engine key, label, engine digests).
        self._pending: dict[int, tuple[int, str, list, bool]] = {}
        self._token = 0

    @staticmethod
    def _engine_group(generation: Any):
        g = getattr(generation, "base", None)  # SpeculativeGeneration wrapper
        gen = g if g is not None else generation
        return gen._policy.worker_group

    def engine_digest(self, generation: Any) -> list:
        """[(tp rank, digests, coverage)] of every engine rank of ``generation``."""
        return self._submit_engine(generation)()

    def learner_digest(self, dtypes: dict[str, str]) -> list:
        return self._submit_learner(dtypes)()

    def _submit_engine(self, generation: Any) -> Callable[[], list]:
        """Start the engine digest; the returned thunk waits for it."""
        import ray

        refs = self._engine_group(generation).run_all_workers_single_data(
            "sentinel_weight_digest"
        )
        return lambda: ray.get(refs)

    def _submit_learner(self, dtypes: dict[str, str], lookahead: bool = False) -> Callable[[], list]:
        import ray

        refs = self.policy.worker_group.run_all_workers_single_data(
            "sentinel_weight_digest", dtypes=dtypes, learner=True, **_la_kw(lookahead)
        )
        return lambda: ray.get(refs)

    @staticmethod
    def _engine_key(generation: Any) -> int:
        g = getattr(generation, "base", None)
        return id(g if g is not None else generation)

    def _sample_skip(self, key: Any) -> bool:
        """THUNDERSYNC_SENTINEL_EVERY=k (perf / gate runs; correctness runs keep every refit): check the first refit of
        each engine and then every k-th one (a function of the refit count only: each checked refit is a sample)."""
        every = int(os.environ.get("THUNDERSYNC_SENTINEL_EVERY", "1") or 1)
        if every <= 1:
            return False
        n = self._sampled.get(key, 0)
        self._sampled[key] = n + 1
        if n % every:
            self.stats["sentinel/skipped"] = self.stats.get("sentinel/skipped", 0.0) + 1.0
            return True
        return False

    def check(
        self,
        generation: Any,
        label: str,
        need_engine_digest: bool = False,
        lookahead: bool = False,
    ) -> list[tuple[int, dict[str, Digest]]]:
        """Compare ``generation``'s engine with the learner's live weights now.

        Returns the engine digests (``[]`` when THUNDERSYNC_SENTINEL_ASYNC defers them; ``need_engine_digest``
        keeps the engine side synchronous for a caller that uses them). ``lookahead``: the engine was refit with
        theta~ (lookahead drafts, lookahead_draft.py); the learner digests theta~ of its live weights.
        """
        if not self.enabled:
            return []
        if self._sample_skip(self._engine_key(generation)):
            return []

        t0 = time.perf_counter()
        key = self._engine_key(generation)
        cached = self._dtypes.get(key)
        if cached is not None and _async_enabled():
            return self._check_async(
                generation, label, key, cached, t0, not need_engine_digest, lookahead=lookahead
            )
        # Both sides at once when this engine's dtypes are known (different GPUs;
        # about half of the serial engine-then-learner time, results_audit 4.1).
        wait_eng = self._submit_engine(generation)
        wait_lrn = self._submit_learner(cached, **_la_kw(lookahead)) if cached is not None else None
        eng_full = wait_eng()
        eng = [(tp, d) for tp, d, *_ in eng_full]
        dtypes = engine_dtypes(eng)
        if wait_lrn is None or dtypes != cached:
            if wait_lrn is not None:
                wait_lrn()  # digested with stale dtypes: discard, redo with the real ones
                self.stats["sentinel/dtype_redo"] = self.stats.get("sentinel/dtype_redo", 0.0) + 1.0
            wait_lrn = self._submit_learner(dtypes, **_la_kw(lookahead))
            self._dtypes[key] = dtypes
        lrn_full = wait_lrn()
        self._finish(key, label, eng_full, lrn_full, t0)
        return eng

    def _check_async(
        self,
        generation: Any,
        label: str,
        key: int,
        cached: dict[str, str],
        t0: float,
        engine_async: bool,
        lookahead: bool = False,
    ) -> list[tuple[int, dict[str, Digest]]]:
        """Learner digest (and the engine's, ``engine_async``) started off the training / decode path on both sides
        and compared later (``_resolve``); same digests and comparison as the synchronous check."""
        import ray

        self._resolve(lambda k: k == key)  # one comparison in flight per engine
        self._token += 1
        token = self._token
        lrn_begin = self.policy.worker_group.run_all_workers_single_data(
            "sentinel_digest_begin", token=token, dtypes=cached, **_la_kw(lookahead)
        )
        if engine_async:
            egroup = self._engine_group(generation)
            ray.get(egroup.run_all_workers_single_data("sentinel_digest_begin", token=token, learner=False))
            ray.get(lrn_begin)
            self._pending[token] = (key, label, None, egroup, cached)
            eng = []
        else:
            eng_full = self._submit_engine(generation)()
            ray.get(lrn_begin)
            self._pending[token] = (key, label, eng_full, None, cached)
            eng = [(tp, d) for tp, d, *_ in eng_full]
            if engine_dtypes(eng) != cached:
                # The learner digested with stale dtypes: compare nothing from it, redo synchronously.
                self._pending.pop(token, None)
                self._collect(token, self.policy.worker_group, "learner")
                self.stats["sentinel/dtype_redo"] = self.stats.get("sentinel/dtype_redo", 0.0) + 1.0
                self._dtypes.pop(key, None)
                return self._check_sync_after_redo(generation, label, t0, lookahead)
        self.stats["sentinel/async_begun"] = self.stats.get("sentinel/async_begun", 0.0) + 1.0
        self.stats["sentinel/time_s"] = self.stats.get("sentinel/time_s", 0.0) + (time.perf_counter() - t0)
        return eng

    def _check_sync_after_redo(
        self, generation: Any, label: str, t0: float, lookahead: bool = False
    ):
        key = self._engine_key(generation)
        eng_full = self._submit_engine(generation)()
        eng = [(tp, d) for tp, d, *_ in eng_full]
        self._dtypes[key] = engine_dtypes(eng)
        lrn_full = self._submit_learner(self._dtypes[key], **_la_kw(lookahead))()
        self._finish(key, label, eng_full, lrn_full, t0)
        return eng

    def _collect(self, token: int, group: Any, side: str) -> list:
        """``side``'s results of digest ``token`` from every worker of ``group`` (raises what a digest raised)."""
        import ray

        res = ray.get(group.run_all_workers_single_data("sentinel_digest_result", token=token))
        secs = max((r[1] for r in res), default=0.0)
        waited = max((r[2] for r in res), default=0.0)
        k = f"sentinel/async_{side}_s"
        self.stats[k] = self.stats.get(k, 0.0) + secs
        k = f"sentinel/async_{side}_write_wait_s"
        self.stats[k] = self.stats.get(k, 0.0) + waited
        return [r[0] for r in res]

    def _resolve(self, which: Callable[[int], bool] = lambda k: True) -> None:
        """Compare every pending asynchronous check whose engine key satisfies ``which`` (in begin order)."""
        for token in sorted(t for t, p in getattr(self, "_pending", {}).items() if which(p[0])):
            key, label, eng_full, egroup, cached = self._pending.pop(token)
            t0 = time.perf_counter()
            if eng_full is None:
                eng_full = self._collect(token, egroup, "engine")
            lrn_full = self._collect(token, self.policy.worker_group, "learner")
            if eng_full is not None and engine_dtypes([(tp, d) for tp, d, *_ in eng_full]) != cached:
                raise WeightSentinelError(
                    f"weight sentinel ({label}): the engine's storage dtypes changed since its first check; the "
                    f"learner digest (cast to the earlier dtypes) cannot be compared"
                )
            self._finish(key, label, eng_full, lrn_full, t0)

    def _finish(self, key: int, label: str, eng_full: list, lrn_full: list, t0: float) -> None:
        """The comparison and coverage accounting of one check (synchronous or asynchronous); ``t0``: start of
        the driver time it accounts."""
        eng = [(tp, d) for tp, d, *_ in eng_full]
        lrn = [(tp, d) for tp, d, *_ in lrn_full]
        cov = compare(eng, lrn, label)
        self._gate_count((key, "learner"), cov, label)
        e_cov = _sum_cov(x[2] if len(x) > 2 else {} for x in eng_full)
        l_cov = _sum_cov(x[2] if len(x) > 2 else {} for x in lrn_full)
        cov["sentinel/compute_buffers"] = float(
            sum(1 for n, _ in set(merge_side(eng, "engine")) if ALIAS in n)
        )
        if not self._reported_coverage:
            self._reported_coverage = True
            e_only, l_only = unmatched(eng, lrn)
            print(coverage_report(label, cov, e_cov, l_cov, e_only, l_only), flush=True)
        self._account(cov, time.perf_counter() - t0)

    def check_engine_against(
        self, generation: Any, reference: list[tuple[int, dict[str, Digest]]], label: str
    ) -> None:
        """The engine now holds bitwise what ``reference`` digested earlier."""
        if not self.enabled or not reference:
            return
        import time

        t0 = time.perf_counter()
        eng = [(tp, d) for tp, d, *_ in self.engine_digest(generation)]
        cov = compare(eng, reference, label)
        self._gate_count((self._engine_key(generation), "reference"), cov, label)
        self._account(cov, time.perf_counter() - t0)

    def _gate_count(self, key: tuple[int, str], cov: dict[str, float], label: str) -> None:
        """Raise if this engine's compared-tensor count differs from its first check (one int per engine)."""
        n = int(cov["sentinel/tensors"])
        first = self._tensor_counts.setdefault(key, n)
        if n != first:
            raise WeightSentinelError(
                f"weight sentinel ({label}): {n} tensors compared, {first} at this engine's first check: the "
                f"compared set changed between refits"
            )

    def _account(self, cov: dict[str, float], secs: float) -> None:
        self.stats["sentinel/checks"] = self.stats.get("sentinel/checks", 0.0) + 1.0
        self.stats["sentinel/time_s"] = self.stats.get("sentinel/time_s", 0.0) + secs
        for k, v in cov.items():
            self.stats[k] = v

    def pop_stats(self) -> dict[str, float]:
        self._resolve()  # asynchronous checks of this step: compared (and raised) before the step's metrics
        out, self.stats = self.stats, {}
        return out
