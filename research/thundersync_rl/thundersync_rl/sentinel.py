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
"""Always-on engine-vs-learner weight sentinel (DESIGN.md section 14s).

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

import re
from typing import Any, Iterable, Iterator, Optional

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
) -> dict[str, Digest]:
    """Digest every tensor; ``dtypes[name]``: cast to the other side's storage dtype.

    ``dtypes`` keys ``<name>#<kind>`` (engine compute tensors) are digested from
    the tensor ``<name>`` cast to that dtype, under the alias key.
    """
    named = list(named)
    out = {}
    for name, t in named:
        want = None if dtypes is None or name not in dtypes else name_dtype(dtypes[name])
        out[name] = tensor_digest(t, want)
    if dtypes:
        by_name = dict(named)
        for key, dt in dtypes.items():
            if ALIAS in key:
                src = by_name.get(key.split(ALIAS, 1)[0])
                if src is not None:
                    out[key] = tensor_digest(src, name_dtype(dt))
    return out


@torch.no_grad()
def check_expert_weight_cache(model: Any) -> int:
    """Learner: every cached stacked fp16 expert weight equals its parameters cast now.

    ``fp16_moe``'s cache is keyed by parameter identity and data pointer, not
    by value: an in-place weight change that skipped ``invalidate_weight_cache``
    would leave stale fp16 experts in the verification / training forwards.
    Returns the number of expert slices checked; raises ``WeightSentinelError``.
    """
    try:
        from nemo_rl.models.megatron import fp16_moe
    except ImportError:
        return 0
    params = {}
    for chunk in _chunks(model):
        for name, p in chunk.named_parameters():
            params[id(p)] = (name, p)
    n, bad = 0, []
    for key, (ptrs, w16) in list(fp16_moe._WEIGHT_CACHE.items()):
        hits = [params.get(pid) for pid in key]
        if any(h is None for h in hits) or tuple(h[1].data_ptr() for h in hits) != ptrs:
            continue  # not this model's, or a miss on next use (rebuilt): never read stale
        for i, (name, p) in enumerate(hits):
            n += 1
            if not torch.equal(w16[i], p.detach().to(torch.float16)):
                bad.append(name)
    if bad:
        raise WeightSentinelError(
            f"weight sentinel: {len(bad)} cached fp16 expert weights differ from their "
            f"parameters (stale cache): {bad[:10]}"
        )
    return n


def merge_side(
    ranks: list[tuple[int, dict[str, Digest]]], side: str
) -> dict[tuple[str, int], Digest]:
    """One side's digests keyed by (name, tp rank); replicas must agree bitwise."""
    merged: dict[tuple[str, int], Digest] = {}
    bad = []
    for tp, digests in ranks:
        for name, d in digests.items():
            k = (name, tp)
            if k in merged and merged[k] != d:
                bad.append(f"{name}[tp{tp}]")
            merged.setdefault(k, d)
    if bad:
        raise WeightSentinelError(
            f"weight sentinel: {side} replicas disagree on {len(bad)} tensors: {bad[:20]}"
        )
    return merged


def compare(
    engine: list[tuple[int, dict[str, Digest]]],
    learner: list[tuple[int, dict[str, Digest]]],
    label: str,
) -> dict[str, float]:
    """Raise ``WeightSentinelError`` on any mismatch; return coverage counters."""
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
    return {
        "sentinel/tensors": float(len(common)),
        "sentinel/elems": float(sum(e[k][1] for k in common)),
        "sentinel/engine_only": float(len(set(e) - set(l_))),
        "sentinel/learner_only": float(len(set(l_) - set(e))),
        "sentinel/expert_bias": float(sum(1 for n, _ in common if n.endswith("expert_bias"))),
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
        f"{e_cov.get('fp16_params', 0)} rank-tensors; learner fp16 expert cache slices checked "
        f"against the parameters: {l_cov.get('learner_fp16_expert_cache', 0)}",
        f"[sentinel]   engine-only (not compared) {len(e_only)}: {e_only[:12]}",
        f"[sentinel]   learner-only (not compared) {len(l_only)}: {l_only[:12]}",
    ]
    lines += [f"[sentinel]   not checked: {x}" for x in NOT_CHECKED]
    return "\n".join(lines)


def unmatched(
    engine: list[tuple[int, dict[str, Digest]]],
    learner: list[tuple[int, dict[str, Digest]]],
) -> tuple[list[str], list[str]]:
    """Canonical names present on one side only (for the coverage report)."""
    e = {n for _, d in engine for n in d}
    l_ = {n for _, d in learner for n in d}
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
        self.stats: dict[str, float] = {}
        self._reported_coverage = False

    @staticmethod
    def _engine_group(generation: Any):
        g = getattr(generation, "base", None)  # SpeculativeGeneration wrapper
        gen = g if g is not None else generation
        return gen._policy.worker_group

    def engine_digest(self, generation: Any) -> list:
        """[(tp rank, digests, coverage)] of every engine rank of ``generation``."""
        import ray

        return ray.get(
            self._engine_group(generation).run_all_workers_single_data(
                "sentinel_weight_digest"
            )
        )

    def learner_digest(self, dtypes: dict[str, str], caches: bool = False) -> list:
        import ray

        return ray.get(
            self.policy.worker_group.run_all_workers_single_data(
                "sentinel_weight_digest", dtypes=dtypes, learner=True, caches=caches
            )
        )

    def check(
        self, generation: Any, label: str, caches: bool = True
    ) -> list[tuple[int, dict[str, Digest]]]:
        """Compare ``generation``'s engine with the learner's live weights now.

        ``caches``: also check the learner's fp16 expert-weight cache (False while
        the learner holds draft weights: the cache keeps theta_k's casts then).
        """
        if not self.enabled:
            return []
        import time

        t0 = time.perf_counter()
        eng_full = self.engine_digest(generation)
        eng = [(tp, d) for tp, d, *_ in eng_full]
        lrn_full = self.learner_digest(engine_dtypes(eng), caches=caches)
        lrn = [(tp, d) for tp, d, *_ in lrn_full]
        cov = compare(eng, lrn, label)
        e_cov = _sum_cov(x[2] if len(x) > 2 else {} for x in eng_full)
        l_cov = _sum_cov(x[2] if len(x) > 2 else {} for x in lrn_full)
        cov["sentinel/compute_buffers"] = float(
            sum(1 for n, _ in set(merge_side(eng, "engine")) if ALIAS in n)
        )
        cov["sentinel/learner_cache_slices"] = float(l_cov.get("learner_fp16_expert_cache", 0))
        if not self._reported_coverage:
            self._reported_coverage = True
            e_only, l_only = unmatched(eng, lrn)
            print(coverage_report(label, cov, e_cov, l_cov, e_only, l_only), flush=True)
        self._account(cov, time.perf_counter() - t0)
        return eng

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
        self._account(cov, time.perf_counter() - t0)

    def _account(self, cov: dict[str, float], secs: float) -> None:
        self.stats["sentinel/checks"] = self.stats.get("sentinel/checks", 0.0) + 1.0
        self.stats["sentinel/time_s"] = self.stats.get("sentinel/time_s", 0.0) + secs
        for k, v in cov.items():
            self.stats[k] = v

    def pop_stats(self) -> dict[str, float]:
        out, self.stats = self.stats, {}
        return out
