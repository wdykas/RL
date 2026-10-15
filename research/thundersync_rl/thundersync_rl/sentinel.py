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
"""Always-on engine-vs-learner weight sentinel (DESIGN.md section 14q).

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


def canonical_named_tensors(
    model: Any, ep_rank: int = 0, include_buffers: bool = True
) -> Iterator[tuple[str, torch.Tensor]]:
    """(canonical name, tensor) of every parameter (and ``expert_bias`` buffer).

    Shared tensors (tied embeddings) are listed once, under their first name.
    """
    chunks = model if isinstance(model, (list, tuple)) else [model]
    seen: set[tuple[int, int, torch.dtype]] = set()
    for chunk in chunks:
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

        items: list[tuple[str, torch.Tensor]] = list(chunk.named_parameters())
        if include_buffers:
            items += [(n, b) for n, b in chunk.named_buffers() if n.endswith("expert_bias")]
        for name, t in items:
            key = (t.data_ptr(), t.numel(), t.dtype)
            if key in seen:
                continue
            seen.add(key)
            yield canon(name), t


def digest_named(
    named: Iterable[tuple[str, torch.Tensor]],
    dtypes: Optional[dict[str, str]] = None,
) -> dict[str, Digest]:
    """Digest every tensor; ``dtypes[name]``: cast to the other side's storage dtype."""
    out = {}
    for name, t in named:
        want = None if dtypes is None or name not in dtypes else name_dtype(dtypes[name])
        out[name] = tensor_digest(t, want)
    return out


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

    def engine_digest(self, generation: Any) -> list[tuple[int, dict[str, Digest]]]:
        import ray

        return ray.get(
            self._engine_group(generation).run_all_workers_single_data(
                "sentinel_weight_digest"
            )
        )

    def learner_digest(self, dtypes: dict[str, str]) -> list[tuple[int, dict[str, Digest]]]:
        import ray

        return ray.get(
            self.policy.worker_group.run_all_workers_single_data(
                "sentinel_weight_digest", dtypes=dtypes, learner=True
            )
        )

    def check(self, generation: Any, label: str) -> list[tuple[int, dict[str, Digest]]]:
        """Compare ``generation``'s engine with the learner's live weights now."""
        if not self.enabled:
            return []
        import time

        t0 = time.perf_counter()
        eng = self.engine_digest(generation)
        lrn = self.learner_digest(engine_dtypes(eng))
        cov = compare(eng, lrn, label)
        if not self._reported_coverage:
            self._reported_coverage = True
            e_only, l_only = unmatched(eng, lrn)
            print(
                f"[sentinel] {label}: {int(cov['sentinel/tensors'])} tensors compared "
                f"({int(cov['sentinel/elems'])} elements, {int(cov['sentinel/expert_bias'])} "
                f"expert_bias); engine-only {len(e_only)} {e_only[:8]}; learner-only "
                f"{len(l_only)} {l_only[:8]}",
                flush=True,
            )
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
        cov = compare(self.engine_digest(generation), reference, label)
        self._account(cov, time.perf_counter() - t0)

    def _account(self, cov: dict[str, float], secs: float) -> None:
        self.stats["sentinel/checks"] = self.stats.get("sentinel/checks", 0.0) + 1.0
        self.stats["sentinel/time_s"] = self.stats.get("sentinel/time_s", 0.0) + secs
        for k, v in cov.items():
            self.stats[k] = v

    def pop_stats(self) -> dict[str, float]:
        out, self.stats = self.stats, {}
        return out
