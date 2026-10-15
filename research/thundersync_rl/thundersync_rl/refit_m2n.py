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
"""Many-to-many (m2n) refit for ThunderSync, and its bitwise crosscheck.

Transport (``policy.generation.refit_transport: mcore``): Megatron Core's native
refit. Every learner rank sends its own TP/EP/PP-local shards straight into the
engine ranks' parameters (Megatron names, no HF export, no rank-0 funnel); the
plan is built once per lane and cached, and the source of each shard is
balanced over the learner's DP replicas. ``mcore_generation_config.refit_backend``
picks the copy service: ``nccl_m2n`` (native NCCL M2N, needs nccl-extensions)
or ``nccl`` (batched NCCL point-to-point). The engine receives its fp16-held
weights through ``Fp16HeldWeightReceive`` (megatron_worker.py). Cohort lanes
keep one native refit group per lane (``keep_refit_group``).

Crosscheck (``THUNDERSYNC_REFIT_CROSSCHECK=1``, debug): every refit runs the
packed broadcast (the reference) into the engine, takes a bitwise digest of
every engine tensor, poisons every refit tensor, runs the m2n refit and takes
the digest again. Equal digests on every rank = the m2n refit wrote bitwise
the broadcast's weights, fp16 casts and derived caches included, for theta and
the momentum draft theta_hat alike (both go through ``sync_weights``). The run
then continues on the m2n weights. One line per refit:
``[m2n-crosscheck] PASS|FAIL refit=<n> ...``.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import torch

# Digest: two position-weighted sums of the raw bits, modulo two primes < 2^31.
_P1 = 2_147_483_647  # 2^31 - 1
_P2 = 2_147_483_629
_CHUNK = 1 << 22
_INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def crosscheck_enabled() -> bool:
    return os.environ.get("THUNDERSYNC_REFIT_CROSSCHECK", "0") not in ("", "0")


@torch.no_grad()
def tensor_digest(t: torch.Tensor) -> str:
    """Bitwise digest of ``t``: equal for equal bits, different for any bit change.

    Raw bits (an integer view of the storage dtype) weighted by a per-position
    pseudo-random factor, summed modulo two primes; a change of one element by
    d changes a sum by d * w mod p, which is nonzero unless p divides d (|d| <
    2^32, so at most a few multiples) for both primes at once.
    """
    flat = t.detach().contiguous().reshape(-1)
    n = flat.numel()
    if n == 0:
        return f"{t.dtype}:0"
    bits = flat.view(_INT_VIEW[flat.element_size()])
    s1 = s2 = 0
    for start in range(0, n, _CHUNK):
        v = bits[start : start + _CHUNK].to(torch.int64)
        pos = torch.arange(start, start + v.numel(), device=v.device, dtype=torch.int64)
        w1 = (pos * 40503 + 2654435761) % _P1
        w2 = (pos * 69069 + 1013904223) % _P2
        s1 = (s1 + int(((v % _P1) * w1 % _P1).sum().item())) % _P1
        s2 = (s2 + int((((v + 12345) % _P2) * w2 % _P2).sum().item())) % _P2
    return f"{t.dtype}:{n}:{s1:x}:{s2:x}"


def _chunks(model: Any) -> list:
    return list(model) if isinstance(model, (list, tuple)) else [model]


@torch.no_grad()
def model_digest(model: Any) -> tuple[list[str], list[str], list[str]]:
    """(names, digests, dtypes) of every parameter and buffer of ``model``."""
    entries: dict[str, torch.Tensor] = {}
    for i, chunk in enumerate(_chunks(model)):
        for name, p in chunk.named_parameters():
            entries[f"{i}/{name}"] = p
        for name, b in chunk.named_buffers():
            if b is not None:
                entries[f"{i}/buffer/{name}"] = b
    names = sorted(entries)
    with torch.inference_mode():  # engine tensors may be inference tensors
        digests = [tensor_digest(entries[n]) for n in names]
    return names, digests, [str(entries[n].dtype) for n in names]


@torch.no_grad()
def poison_refit_tensors(model: Any) -> int:
    """Fill every tensor the refit writes (MCore's refit tensor set) with a sentinel."""
    from megatron.core.resharding.utils import get_refit_tensor_dict
    from megatron.core.utils import unwrap_model

    n = 0
    # Engine tensors allocated during generation are inference tensors.
    with torch.inference_mode():
        for chunk in _chunks(model):
            for t in get_refit_tensor_dict(unwrap_model(chunk)).values():
                if t.is_floating_point():
                    t.data.fill_(float("nan"))
                else:
                    t.data.fill_(-7)
                n += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return n


def compare_digests(reference: list[dict], candidate: list[dict]) -> list[str]:
    """Names (``rank<r>:<tensor>``) whose digests differ, rank by rank."""
    bad = []
    for rank, (ref, got) in enumerate(zip(reference, candidate)):
        if ref["names"] != got["names"]:
            missing = sorted(set(ref["names"]) ^ set(got["names"]))
            bad += [f"rank{rank}:<tensor set differs: {missing[:5]}>"]
            continue
        bad += [
            f"rank{rank}:{name}"
            for name, a, b in zip(ref["names"], ref["digests"], got["digests"])
            if a != b
        ]
    return bad


def _make_crosscheck_cls():
    # Deferred: the synchronizer module imports NeMo-RL's generation stack.
    import ray

    from nemo_rl.weight_sync.megatron_weight_synchronizer import (
        MegatronWeightSynchronizer,
    )

    class CrosscheckRefitSynchronizer(MegatronWeightSynchronizer):
        """Native (m2n) refit, checked bitwise against the packed broadcast on every refit."""

        refits = 0
        failures = 0

        def init_reference(self) -> None:
            """Build the packed collective (the reference) next to the native refit group."""
            policy, gen = self._policy, self._generation
            state_dict_info = policy.prepare_refit_info(
                refit_payload_mode=gen.get_refit_payload_mode()
            )
            engine = gen._policy
            # MegatronGeneration routes prepare_refit_info / init_collective to the
            # native refit under refit_transport mcore: call its workers directly.
            ray.get(
                engine.worker_group.run_all_workers_single_data(
                    "prepare_refit_info", state_dict_info=state_dict_info
                )
            )
            ip, port = self._train_cluster.get_master_address_and_port()
            train_world_size = self._train_cluster.world_size()
            world_size = train_world_size + self._inference_cluster.world_size()
            sender_spec = gen.get_collective_sender_spec()
            futures = policy.init_collective(
                ip,
                port,
                world_size,
                train_world_size=train_world_size,
                nccl_peer=sender_spec.nccl_peer,
            ) + engine.init_collective(
                ip,
                port,
                world_size,
                train_world_size=train_world_size,
                rank_offset=train_world_size,
            )
            ray.get(futures)

        def init_communicator(self) -> None:
            super().init_communicator()
            self.init_reference()

        def sync_weights(self, *, timer=None, kv_scales=None) -> Optional[dict]:
            if self._colocated or self._transport is not None:
                raise RuntimeError("the refit crosscheck needs refit_transport mcore, non-colocated")
            gen, policy = self._generation, self._policy
            engine = gen._policy.worker_group
            gen.suspend_for_refit()
            gen.prepare_for_generation(tags=["weights"])
            # 1. Reference: packed broadcast of the learner's live weights.
            t0 = time.perf_counter()
            futures_train = policy.broadcast_weights_for_collective(kv_scales=kv_scales)
            futures_engine = engine.run_all_workers_single_data(
                "update_weights_from_collective"
            )
            ray.get(futures_train)
            if not all(r for r in ray.get(futures_engine) if r is not None):
                raise RuntimeError("crosscheck: reference broadcast refit failed")
            t1 = time.perf_counter()
            reference = ray.get(engine.run_all_workers_single_data("engine_weight_digest"))
            ray.get(engine.run_all_workers_single_data("engine_poison_refit_weights"))
            # 2. Candidate: the native m2n refit, into the poisoned engine.
            t2 = time.perf_counter()
            futures_engine = gen.update_weights_from_collective()
            futures_train = policy.swap_weights_via_reshard(is_source=True)
            ray.get(futures_train)
            ray.get(futures_engine)
            t3 = time.perf_counter()
            candidate = ray.get(engine.run_all_workers_single_data("engine_weight_digest"))
            bad = compare_digests(reference, candidate)
            type(self).refits += 1
            type(self).failures += bool(bad)
            n_tensors = sum(len(r["names"]) for r in reference)
            n_fp16 = sum(d == "torch.float16" for r in candidate for d in r["dtypes"])
            print(
                f"[m2n-crosscheck] {'FAIL' if bad else 'PASS'} refit={type(self).refits} "
                f"ranks={len(reference)} tensors={n_tensors} fp16_tensors={n_fp16} "
                f"mismatched={len(bad)} broadcast_s={t1 - t0:.3f} m2n_s={t3 - t2:.3f} "
                f"failures_so_far={type(self).failures}"
                + (f" first={bad[:20]}" if bad else ""),
                flush=True,
            )
            gen.prepare_for_generation(tags=["kv_cache"])
            gen.resume_after_refit()
            self._stale = False
            if bad and os.environ.get("THUNDERSYNC_REFIT_CROSSCHECK") == "strict":
                raise RuntimeError(f"m2n refit differs from the broadcast refit: {bad[:20]}")
            return {}

    return CrosscheckRefitSynchronizer


_CLS = None


def crosscheck_cls():
    global _CLS
    if _CLS is None:
        _CLS = _make_crosscheck_cls()
    return _CLS


def make_lane_synchronizer(
    policy: Any, generation: Any, train_cluster: Any, inference_cluster: Any
) -> Any:
    """The weight synchronizer of a new engine lane (crosscheck-wrapped when enabled)."""
    if crosscheck_enabled() and generation.uses_native_refit:
        return crosscheck_cls()(
            policy,
            generation,
            colocated=False,
            train_cluster=train_cluster,
            inference_cluster=inference_cluster,
        )
    from nemo_rl.weight_sync.factory import create_weight_synchronizer

    return create_weight_synchronizer(
        policy=policy,
        generation=generation,
        generation_backend="megatron",
        colocated=False,
        train_cluster=train_cluster,
        inference_cluster=inference_cluster,
    )


def install_crosscheck(
    policy: Any, generation: Any, train_cluster: Any, inference_cluster: Any
) -> None:
    """Wrap the setup engine's (already initialized) native refit with the crosscheck."""
    if not (crosscheck_enabled() and generation.uses_native_refit):
        return
    sync = crosscheck_cls()(
        policy,
        generation,
        colocated=False,
        train_cluster=train_cluster,
        inference_cluster=inference_cluster,
    )
    sync._refit_backend = generation.cfg["mcore_generation_config"]["refit_backend"]
    sync.init_reference()
    sync._stale = False
    generation.weight_synchronizer = sync
