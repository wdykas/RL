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
"""Engine side of the nolanes mode (nolanes.py): version tags, per-request hidden keys, noise-key registry.

Generation workers only. ``NolanesEngineMixin.nl_engine_setup`` installs, on an engine that runs the SC loop's
requests (``generate_async``):
- the wide keyed sampler (noise keyed by (salt, engine request id, position)) and the draft hidden-state log;
- a hidden-log key on every request (``_nl_request_attrs``), handed back with the result (``ts_draft``);
- the version tag of every logged token: the resident weight version of the FORWARD whose logits it was sampled
  from (stamped on the host by the forward wrapper on every engine step, eager or graph replay; R23 MEDIUM-3c). The
  driver announces the refit's ABSOLUTE version (``nl_announce_version``) and every rank sets it in
  ``suspend_for_refit``, with its engine suspended and before any weight is written (not in ``resume_after_refit``:
  engine rank 0 resumes every engine through the coordinator before the other ranks' resume calls arrive). A sample
  of a forward more than one version behind the resident one raises;
- the noise-key registry (E4 / MEDIUM-5): a (key request id, position) pair is drawn at most once per engine
  process; a reuse raises ``SamplerKeyError``.
"""

from __future__ import annotations

import itertools
import os
import threading
from array import array
from typing import Any, Optional

import numpy as np

NL_KEY_PREFIX = "nl"
# SamplingParams attribute: noise-key request id override (negative control key_reuse only).
KEY_RID_ATTR = "ts_nl_key_rid"
# SamplingParams attribute set on the engine rank: the draft was cut at a stand-down (wake_verify); it travels back
# with the result (SamplingParams serialize their __dict__) and marks the draft unfinished.
CUT_ATTR = "ts_nl_cut"
# BX-SD fork lane (bxsd_lane.py / bxsd_route.py): the request's target engine DP rank, and the fork-row bit (L1).
BX_DP_RANK_ATTR = "ts_dp_rank"
# Lag-K wake_verify: the training step a draft's group targets (the sampler's target_step stamp, driver data field
# nl_target_step). A stand-down cuts only the drafts of the batch that trains next; later batches' drafts pause.
TARGET_ATTR = "ts_nl_target"


def cuttable(sp: Any, max_target: Optional[int]) -> bool:
    """A draft is cut at this stand-down iff no bound is given (lag 1: every draft) or it targets <= max_target
    (unstamped drafts are cut: the lag-1 rule)."""
    if max_target is None:
        return True
    t = getattr(sp, TARGET_ATTR, None)
    return t is None or int(t) <= int(max_target)
BX_FORK_ATTR = "ts_bx_fork"


def normalize_empty_results(result: Any) -> int:
    """A request that finished without a generated token carries ``generated_log_probs = None`` (MCore's default
    before the first append): a draft cut at a stand-down in the step it was first scheduled (the cut step's sampled
    token is not appended) ends with zero tokens. Give it an empty log-prob list, as its empty token list; returns the
    number of results changed. (Under the carry_unverified negative the short continuation phase left many fresh
    drafts at the cut: the parser's torch.tensor(None) failed at step 3 of a gate.)"""
    n = 0
    for r in result:
        if getattr(r, "generated_log_probs", None) is None and not list(getattr(r, "generated_tokens", None) or []):
            r.generated_log_probs = []
            n += 1
    return n


def mark_cut(engine: Any, max_target: Optional[int] = None) -> dict[str, int]:
    """Mark every draft request (hidden-log key) of a PAUSED dynamic engine for a cut (``nl_cut_drafts``): resident
    requests finish at the engine's next step through the stop-word path (that step's sampled token is not appended);
    a waiting request (not prefilled) finishes after its first token. Each marked request carries ``CUT_ATTR`` back
    with its result. Continuations / fresh rows (no hidden-log key) are never touched."""
    from thundersync_rl.engine_q import HIDDEN_KEY_ATTR

    counts = {"cut": 0, "waiting": 0, "marked": 0, "kept": 0}
    waiting = {int(r) for r in engine.waiting_request_ids}
    for rid, entry in list(engine.requests.items()):
        sp = entry.record[-1].sampling_params
        if sp is None or getattr(sp, HIDDEN_KEY_ATTR, None) is None:
            continue
        if not cuttable(sp, max_target):
            counts["kept"] += 1  # a later batch's draft (lag K): it pauses at the stand-down and resumes at the wake
            continue
        if getattr(sp, CUT_ATTR, 0):
            counts["marked"] += 1
            continue
        for r in entry.record.requests:
            if getattr(r, "sampling_params", None) is not None:
                r.sampling_params.add_attributes({CUT_ATTR: 1})
        if int(rid) in waiting:
            sp.num_tokens_to_generate = 1
            counts["waiting"] += 1
        else:
            engine.stop_word_finished_request_ids.add(int(rid))
            counts["cut"] += 1
    return counts


class VersionTagLog:
    """Per hidden-log key: the weight version of each logged token (4 bytes / token, run on the logging rank)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tags: dict[str, array] = {}
        self.recorded = 0

    def record(self, keys, ts, version: int) -> None:
        with self._lock:
            for k, t in zip(keys, ts):
                a = self._tags.get(k)
                if a is None:
                    a = self._tags[k] = array("i")
                t = int(t)
                if t >= len(a):
                    a.extend([-1] * (t + 1 - len(a)))
                a[t] = int(version)
                self.recorded += 1

    def pop(self, key: str, n: int) -> list[int]:
        """Versions of tokens 0..m-1 (m <= n: the longest tagged prefix). Vectorized (the hand-out pops every draft
        of the step; this was a per-token Python loop); same list of ints."""
        with self._lock:
            a = self._tags.pop(key, None)
        if a is None:
            return []
        v = np.frombuffer(a, dtype=f"i{a.itemsize}")[: max(int(n), 0)]
        neg = np.flatnonzero(v < 0)
        return v[: int(neg[0]) if neg.size else v.size].tolist()

    def close(self, key: str) -> None:
        with self._lock:
            self._tags.pop(key, None)

    def __len__(self) -> int:
        return len(self._tags)


class KeyRegistry:
    """Every (noise-key request id, position) drawn by this engine process, at most once (E4 / MEDIUM-5).

    Per key id: the highest position drawn. A draw at a position <= that is a reuse (a continuation keyed like the
    draft it replaces, a re-admitted request id); raises. Positions of one request increase by one per draw (the
    sampler's KeyContinuity), so the highest position is the whole history.
    """

    def __init__(self) -> None:
        self.last: dict[int, int] = {}
        self.draws = 0
        self.violations = 0

    def observe(self, key_ids: np.ndarray, pos: np.ndarray) -> list[str]:
        bad = []
        for k, p in zip(key_ids.tolist(), pos.tolist()):
            if p < 0:
                continue
            q = self.last.get(k)
            if q is not None and p <= q:
                bad.append(f"key {k} position {p} already drawn (up to {q})")
                continue
            self.last[k] = p
            self.draws += 1
        self.violations += len(bad)
        return bad


class NolanesEngineMixin:
    """Mixed into ThunderSyncMegatronPolicyWorker (before the stock worker)."""

    def nl_engine_setup(self, salt: int, key64: bool = True, seal_blocks: bool = True, version: int = 0) -> Any:
        """Install the nolanes engine side (idempotent). Returns True on logging ranks, None on ranks without the
        head (hidden log not installed there), False when unsupported."""
        if getattr(self, "dynamic_inference_engine", None) is None:
            return False
        if self.__dict__.get("_nl") is not None:
            return self._nl["logging"]
        if getattr(self, "is_generation_colocated", False):
            # Colocated (every colocated nolanes config, a hard check before any install so a retry runs it again):
            # every admitted row keeps its KV at full length (engine_slots.py), never paused / evicted.
            from thundersync_rl.engine_slots import check_engine_reservation

            tree = os.environ.get("THUNDERSYNC_TAIL_TREE", "").strip()
            kv = check_engine_reservation(self.dynamic_inference_engine,
                                          reserved_blocks=int(tree.split(":")[2]) if tree else 0)
            print(f"[nolanes engine] rank={getattr(self, 'rank', '?')} kv reservation {kv}", flush=True)
        self.install_wide_sampler(salt=int(salt), key64=bool(key64), fresh_salt=None)
        logging_rank = self.install_draft_hidden_log(seal_blocks=bool(seal_blocks))
        if logging_rank is False:
            return False
        self._nl = {
            "version": int(version),
            "fwd_version": int(version),
            "refits": 0,
            "counter": itertools.count(),
            "tags": VersionTagLog(),
            "keyreg": KeyRegistry(),
            "logging": logging_rank,
            "tag_mismatch": 0,
            "neg": os.environ.get("THUNDERSYNC_NL_NEG", "").strip(),
            "neg_tokens": 0,
            "steps_since_refit": 0,
        }
        controller = self.dynamic_inference_engine.controller
        orig = controller._dynamic_step_forward_logits
        nl = self._nl
        # Gate tool (X1): a fixed host delay per engine step while no continuation runs (engine rank 0 knows), so
        # drafts outlive a training step and refits land inside rows at short caps (mixed-version drafts), while
        # the step's own continuations are not slowed. Content independent: only the proposal timing changes.
        step_sleep = float(os.environ.get("THUNDERSYNC_NL_DEBUG_STEP_SLEEP_MS", "0") or 0) / 1000.0
        nl["cont_inflight"] = 0

        def forward_logits(*args, **kwargs):
            if step_sleep > 0 and nl["cont_inflight"] <= 0:
                import time

                time.sleep(step_sleep)
            # Host side of every engine step (eager or graph replay): the version of the weights this forward reads.
            nl["fwd_version"] = nl["version"]
            nl["steps_since_refit"] += 1
            return orig(*args, **kwargs)

        controller._dynamic_step_forward_logits = forward_logits
        self._nl_install_tail_mtd()
        self._nl_install_tail_tree()
        print(
            f"[nolanes engine] rank={getattr(self, 'rank', '?')} version tags + hidden keys + key registry installed "
            f"(logging={logging_rank}, version={version})",
            flush=True,
        )
        return logging_rank

    def _nl_install_tail_mtd(self) -> None:
        """thundersync.tail_mtd (env THUNDERSYNC_TAIL_MTD = K:rows:ref_tokens, set by nolanes.configure_driver): the
        multi-token tail decode runtime on this engine (nemo_rl/models/generation/megatron/tail_mtd.py)."""
        spec = os.environ.get("THUNDERSYNC_TAIL_MTD", "").strip()
        if not spec:
            return
        k, rows, ref_cap = (int(x) for x in spec.split(":"))
        if os.environ.get("THUNDERSYNC_TAIL_MTD_DEBUG_FORCE_K") and os.environ.get("THUNDERSYNC_NL_CHECKS") != "1":
            raise RuntimeError("THUNDERSYNC_TAIL_MTD_DEBUG_FORCE_K is a gate tool: refused without CHECKS")
        from nemo_rl.models.generation.megatron import tail_mtd
        from nemo_rl.models.megatron import mamba_bi_lookahead

        if mamba_bi_lookahead.capacity() == 0:
            raise RuntimeError("[nolanes engine] tail_mtd: the Mamba lookahead rows were not installed before the "
                               "engine was built (megatron_worker._prepare_engine_numerics)")
        engine = self.dynamic_inference_engine
        nl = self._nl
        self._nl_mtd = tail_mtd.install(
            engine, K=k, t_max=64, ref_cap=ref_cap, sample_rows=self._nl_mtd_sample,
            key_check=getattr(self, "_wide_key_check", None), keyreg=nl["keyreg"],
            key_ids=lambda rids: self._nl_key_ids(engine, rids),
        )
        print(f"[nolanes engine] rank={getattr(self, 'rank', '?')} tail_mtd K={k} rows/step={rows} "
              f"ref tokens={ref_cap}", flush=True)

    def _nl_install_tail_tree(self) -> None:
        """thundersync.tail_tree (env THUNDERSYNC_TAIL_TREE = W:D:rows:ref_tokens, set by nolanes.configure_driver):
        the exact tree speculation runtime on this engine (nemo_rl/models/generation/megatron/tree_spec.py)."""
        spec = os.environ.get("THUNDERSYNC_TAIL_TREE", "").strip()
        if not spec:
            return
        w, d, rows, ref_cap = (int(x) for x in spec.split(":"))
        if os.environ.get("THUNDERSYNC_TAIL_MTD_DEBUG_FORCE_K") and os.environ.get("THUNDERSYNC_NL_CHECKS") != "1":
            raise RuntimeError("THUNDERSYNC_TAIL_MTD_DEBUG_FORCE_K is a gate tool: refused without CHECKS")
        from nemo_rl.models.generation.megatron import tree_kv, tree_spec
        from nemo_rl.models.megatron import mamba_bi_lookahead

        if mamba_bi_lookahead.tree_capacity() == 0 or tree_kv.capacity() == 0:
            raise RuntimeError("[nolanes engine] tail_tree: the node rows (Mamba scratch rows, KV gather) were not "
                               "installed before the engine was built (megatron_worker._prepare_engine_numerics)")
        engine = self.dynamic_inference_engine
        nl = self._nl
        self._nl_tree = tree_spec.install(
            engine, W=w, D=d, t_max=64, ref_cap=ref_cap, node_cap=rows, sample_rows=self._nl_tree_sample,
            key_check=getattr(self, "_wide_key_check", None), keyreg=nl["keyreg"],
            key_ids=lambda rids: self._nl_key_ids(engine, rids),
            scores_rows=self._nl_tree_scores, uniform_rows=self._nl_tree_uniform, top2_rows=self._nl_tree_top2,
        )
        print(f"[nolanes engine] rank={getattr(self, 'rank', '?')} tail_tree W={w} D={d} rows/step={rows} "
              f"ref tokens={ref_cap}", flush=True)
        mtp = os.environ.get("THUNDERSYNC_TAIL_TREE_MTP", "").strip()
        if mtp:
            self._nl_install_tree_mtp(mtp)

    def _nl_install_tree_mtp(self, spec: str) -> None:
        """thundersync.tail_tree_mtp (env THUNDERSYNC_TAIL_TREE_MTP = head path:slots): the MTP draft source of the
        tree (tree_mtp.MTPHead on the engine model's embedding / output layer; the decoder-output capture was
        installed by megatron_worker before the engine was built)."""
        from megatron.core.utils import unwrap_model

        from nemo_rl.models.generation.megatron import tree_mtp

        path, slots = spec.rsplit(":", 1)
        if tree_mtp.hidden_buffer() is None:
            raise RuntimeError("[nolanes engine] tail_tree_mtp: the decoder-output capture was not installed before "
                               "the engine was built (megatron_worker._prepare_engine_numerics)")
        m = unwrap_model(self._gen_model())
        cfg = m.config
        if int(getattr(cfg, "tensor_model_parallel_size", 1) or 1) != 1:
            raise RuntimeError("[nolanes engine] tail_tree_mtp needs an engine with tensor_model_parallel_size 1")
        ctx = self.dynamic_inference_engine.context
        head = tree_mtp.MTPHead(
            path, m.embedding.word_embeddings.weight, m.output_layer.weight, slots=int(slots),
            max_seq=int(ctx.max_sequence_length), num_heads=int(cfg.num_attention_heads),
            num_kv_heads=int(cfg.num_query_groups or cfg.num_attention_heads), head_dim=int(cfg.kv_channels),
            eps=float(cfg.layernorm_epsilon),
        )
        self._nl_tree.attach_mtp(head)
        print(f"[nolanes engine] rank={getattr(self, 'rank', '?')} tail_tree MTP head {path} slots={slots} "
              f"KV {2 * head.kc.numel() * 2 / 2**20:.0f} MiB", flush=True)

    def _nl_tree_sample(self, logits, rids: np.ndarray, pos, temp, kid_xor: int = 0):
        """The keyed sampler S on node rows: noise keyed by (salt, key id of the request, position), positions as a
        device tensor (the device knows each node's depth). ``kid_xor``: negative control tree_fresh_keys only."""
        import torch

        from thundersync_rl import keyed_sampling_s as s_mod
        from thundersync_rl.worker import _min_sampling_temperature

        engine = self.dynamic_inference_engine
        dev = logits.device
        if torch.is_tensor(rids):  # device key ids, already mapped (tree_spec stages them with the layout)
            kid_d = rids.to(device=dev, dtype=torch.int64)
            if kid_xor:
                kid_d = kid_d ^ int(kid_xor)
        else:
            kid = self._nl_key_ids(engine, rids).astype(np.int64)
            if kid_xor:
                kid = kid ^ np.int64(kid_xor)
            kid_d = torch.from_numpy(kid).pin_memory().to(dev, non_blocking=True)
        pos_d = pos.to(device=dev, dtype=torch.int64)
        rows = logits if logits.stride(-1) == 1 else logits.contiguous()
        n, v = rows.shape
        ws = getattr(self, "_nl_mtd_ws", None)
        need = s_mod.workspace_bytes(n, v)
        if ws is None or ws.numel() < need:
            ws = self._nl_mtd_ws = torch.empty(max(need, 1), dtype=torch.uint8, device=dev)
        out = torch.empty(n, dtype=torch.int64, device=dev)
        assert getattr(self, "_wide_key64", True), "tail_tree: the key64 sampler (thundersync.wide_sampler_key64)"
        s_mod.wide_sample(rows, kid_d, pos_d, temp.clamp(min=_min_sampling_temperature()), self._wide_salt,
                          out=out, workspace=ws)
        return out

    def _nl_tree_scores(self, logits, rids: np.ndarray, pos, temp):
        """Checks / negative controls only: the keyed race's full scores of the given rows (keyed_scores)."""
        import torch

        from thundersync_rl.keyed_scores import keyed_scores
        from thundersync_rl.worker import _min_sampling_temperature

        kid = self._nl_key_ids(self.dynamic_inference_engine, np.asarray(rids, dtype=np.int64)).astype(np.int64)
        kid_d = torch.from_numpy(kid).pin_memory().to(logits.device, non_blocking=True)
        rows = logits if logits.stride(-1) == 1 else logits.contiguous()
        return keyed_scores(rows, kid_d, pos.to(device=logits.device, dtype=torch.int64),
                            temp.clamp(min=_min_sampling_temperature()), self._wide_salt)

    def _nl_tree_top2(self, logits, kid_dev, pos_dev, temp):
        """Proposals only (tree_spec's MTP depth-1 sibling): the runner-up of each row's keyed race (S's keys and
        noise, fp32 scores), -1 where none. Device key ids and positions; fixed shapes (graph-capturable)."""
        import torch

        from thundersync_rl import keyed_sampling_s as s_mod
        from thundersync_rl.worker import _min_sampling_temperature

        rows = logits if logits.stride(-1) == 1 else logits.contiguous()
        n, v = rows.shape
        need = s_mod.top2_workspace_bytes(n, v)
        ws = self.__dict__.get("_nl_top2_ws")
        if ws is None or ws.numel() < need:
            ws = self._nl_top2_ws = torch.empty(max(need, 1), dtype=torch.uint8, device=rows.device)
        o1 = torch.empty(n, dtype=torch.int64, device=rows.device)
        o2 = torch.empty(n, dtype=torch.int64, device=rows.device)
        s_mod.keyed_top2(rows, kid_dev.to(torch.int64), pos_dev.to(torch.int64),
                         temp.clamp(min=_min_sampling_temperature()), self._wide_salt, out1=o1, out2=o2, workspace=ws)
        return o2

    def _nl_tree_uniform(self, kid_dev, pos_dev):
        from thundersync_rl.keyed_scores import keyed_uniform

        return keyed_uniform(kid_dev, pos_dev, self._wide_salt)

    def nl_tree_stats(self) -> dict:
        """tree_spec counters of this engine rank (nolanes tail_tree or colo tree_xssd: whichever installed it)."""
        m = self.__dict__.get("_nl_tree")
        if m is None:
            m = getattr(self.__dict__.get("dynamic_inference_engine"), "_tree_spec", None)
        return {} if m is None else m.summary()

    def _nl_mtd_sample(self, logits, rids: np.ndarray, pos: np.ndarray, temp):
        """The keyed sampler S on given rows: noise keyed by (salt, key id of the request, position), the draw
        one-token decode makes for that (request, position) (wide_kernel's S branch, same arguments)."""
        import torch

        from thundersync_rl import keyed_sampling_s as s_mod
        from thundersync_rl.worker import _min_sampling_temperature

        engine = self.dynamic_inference_engine
        kid = self._nl_key_ids(engine, rids)
        dev = logits.device
        pack = torch.from_numpy(np.stack([kid.astype(np.int64), pos.astype(np.int64)])).pin_memory()
        pack = pack.to(dev, non_blocking=True)
        rows = logits if logits.stride(-1) == 1 else logits.contiguous()
        n, v = rows.shape
        ws = getattr(self, "_nl_mtd_ws", None)
        need = s_mod.workspace_bytes(n, v)
        if ws is None or ws.numel() < need:
            ws = self._nl_mtd_ws = torch.empty(max(need, 1), dtype=torch.uint8, device=dev)
        out = torch.empty(n, dtype=torch.int64, device=dev)
        assert getattr(self, "_wide_key64", True), "tail_mtd: the key64 sampler (thundersync.wide_sampler_key64)"
        s_mod.wide_sample(rows, pack[0], pack[1], temp.clamp(min=_min_sampling_temperature()), self._wide_salt,
                          out=out, workspace=ws)
        return out

    def nl_mtd_stats(self) -> dict:
        m = self.__dict__.get("_nl_mtd")
        return {} if m is None else m.summary()

    def nl_set_version(self, version: int) -> int:
        nl = self.__dict__.get("_nl")
        if nl is None:
            return -1
        nl["version"] = nl["fwd_version"] = int(version)
        return nl["version"]

    def nl_announce_version(self, version: int) -> int:
        """Driver, right before a refit: the absolute weight version the next suspend_for_refit installs."""
        nl = self.__dict__.get("_nl")
        if nl is None:
            return -1
        nl["pending"] = int(version)
        return nl["pending"]

    def nl_version(self) -> int:
        nl = self.__dict__.get("_nl")
        return -1 if nl is None else int(nl["version"])


    # -- hooks called from worker.py ------------------------------------------------------------------------

    def _nl_tag_version(self) -> int:
        """Version tag of the tokens sampled now: the version of the FORWARD whose logits they come from.

        Under MCore's async scheduling a step's logits are sampled at the start of the next step, so the forward
        launched just before a refit is sampled after it: those tokens carry the forward's (previous) version. The
        forward ran on the old weights (``suspend_for_refit`` synchronizes the device before any write); the engine
        q check rebuilds them with the old head. More than one version apart is impossible and raises.
        """
        nl = self._nl
        v = nl["fwd_version"]
        if v != nl["version"]:
            if v != nl["version"] - 1:
                nl["tag_mismatch"] += 1
                raise RuntimeError(
                    f"[nolanes engine] sampled logits of a version-{v} forward with version {nl['version']} resident"
                )
            nl["straddle_samples"] = nl.get("straddle_samples", 0) + 1
        if nl["neg"] == "tag_off_by_one" and nl["refits"] > 0 and nl["steps_since_refit"] <= 4 and v == nl["version"]:
            nl["neg_tokens"] += 1
            return v - 1  # negative control E1c: the first steps after a refit keep the previous tag
        return v

    def suspend_for_refit(self, *args, **kwargs):
        out = super().suspend_for_refit(*args, **kwargs)
        nl = self.__dict__.get("_nl")
        if nl is not None:
            import torch

            # Every forward launched before the pause finishes on the old weights before the refit writes any.
            torch.cuda.synchronize()
            # Advance the version here, with this rank's engine suspended and before any weight is written. Not in
            # resume_after_refit: engine rank 0 resumes and unpauses EVERY engine through the coordinator, so another
            # rank's engine may already step on the new weights before its own resume call arrives (X1 dev
            # runs: samples of a forward stamped with the old version after the resume).
            pending = nl.pop("pending", None)
            if pending is None or int(pending) != nl["version"] + 1:
                raise RuntimeError(
                    f"[nolanes engine] refit without (or with a non-consecutive) announced version: resident "
                    f"{nl['version']}, announced {pending}"
                )
            nl["version"] = int(pending)
            nl["refits"] += 1
            nl["steps_since_refit"] = 0
        return out

    def prepare_for_generation(self, *args, **kwargs):
        """Colocated engine (docs/colo/THEORY.md v2): the wake IS the refit; no suspend_for_refit
        runs (the engine was put to sleep before training). An announced version is installed here, with the engine
        asleep (every forward finished at its sleep) and before the wake's reshard writes any weight. A wake without
        an announcement keeps the resident version: the colocated verify stage's mid-step wake reshards the same
        theta_k (the driver checks every rank's tag against the trainer version after each wake)."""
        nl = self.__dict__.get("_nl")
        if nl is not None and getattr(self, "is_generation_colocated", False) and "pending" in nl:
            import torch

            torch.cuda.synchronize()
            pending = int(nl.pop("pending"))
            if pending != nl["version"] + 1:
                raise RuntimeError(
                    f"[nolanes engine] colocated wake with a non-consecutive announced version: resident "
                    f"{nl['version']}, announced {pending}"
                )
            nl["version"] = pending
            nl["refits"] += 1
            nl["steps_since_refit"] = 0
        return super().prepare_for_generation(*args, **kwargs)

    def _nl_record_tags(self, keys, ts) -> None:
        nl = self.__dict__.get("_nl")
        if nl is None or not keys:
            return
        nl["tags"].record(keys, ts, self._nl_tag_version())

    def _nl_key_ids(self, engine: Any, rids: np.ndarray) -> np.ndarray:
        """Noise-key request ids of the sampled rows: the engine request id, or the request's KEY_RID_ATTR override
        when it carries one (a consumer continuing a row under the original row's keys, e.g. a keyed check's
        in-place continuation; INTERFACE.md). nolanes' own requests carry it only under the key_reuse negative, so its
        fresh-key rule (E4) is unchanged; the negative's tokens are counted."""
        nl = self.__dict__.get("_nl")
        cache = self.__dict__.setdefault("_nl_kid_cache", {})  # rid -> override or None (attributes are immutable)
        if len(cache) > 1 << 16:
            cache.clear()
        out = rids.copy()
        for i, rid in enumerate(rids.tolist()):
            if rid in cache:
                k = cache[rid]
            else:
                entry = engine.requests.get(int(rid))
                if entry is None:
                    continue
                k = getattr(entry.record[-1].sampling_params, KEY_RID_ATTR, None)
                cache[rid] = k
            if k is not None:
                out[i] = int(k)
                if nl is not None and nl["neg"] == "key_reuse":
                    nl["neg_tokens"] += 1
        return out

    def _nl_observe_keys(self, key_ids: np.ndarray, pos: Optional[np.ndarray]) -> None:
        nl = self.__dict__.get("_nl")
        if nl is None or pos is None:
            return
        bad = nl["keyreg"].observe(key_ids, pos)
        if bad:
            from thundersync_rl.key_check import SamplerKeyError

            raise SamplerKeyError(
                f"[nolanes engine] noise key reused ({len(bad)} rows; negative tokens {nl['neg_tokens']}): {bad[:5]} "
                "-- E4: every (request, position) key is drawn once (continuations are new requests)"
            )

    def _nl_request_attrs(self, data: Any, sps: list) -> None:
        """Rank 0 (it builds the requests): a hidden-log key on every SC request, unless the data says the request
        needs none (continuations / fresh slots: exact target decodes, never verified)."""
        nl = self.__dict__.get("_nl")
        if nl is None:
            return
        from thundersync_rl.engine_q import HIDDEN_KEY_ATTR

        no_log = data.get("nl_no_log") if data is not None and "nl_no_log" in data else None
        if no_log is not None:
            nl["cont_inflight"] = nl.get("cont_inflight", 0) + sum(1 for x in no_log if bool(x))
        key_rid = data.get("nl_key_rid") if data is not None and "nl_key_rid" in data else None
        mtd_refs = data.get("nl_mtd_refs") if data is not None and "nl_mtd_refs" in data else None
        lane = data.get("nl_lane") if data is not None and "nl_lane" in data else None
        tgt = data.get("nl_target_step") if data is not None and "nl_target_step" in data else None
        placer = (self.__dict__.get("_bx") or {}).get("placer")
        for i, sp in enumerate(sps):
            attrs = {}
            if no_log is None or not bool(no_log[i]):
                attrs[HIDDEN_KEY_ATTR] = f"{NL_KEY_PREFIX}{next(nl['counter'])}"
            if key_rid is not None and int(key_rid[i]) >= 0:
                attrs[KEY_RID_ATTR] = int(key_rid[i])
            if mtd_refs is not None and mtd_refs[i] is not None:
                from nemo_rl.models.generation.megatron.tail_mtd import MTD_REFS_ATTR

                attrs[MTD_REFS_ATTR] = np.ascontiguousarray(mtd_refs[i], dtype=np.int32).tobytes()
            if tgt is not None and tgt[i] is not None and attrs.get(HIDDEN_KEY_ATTR) is not None:
                attrs[TARGET_ATTR] = int(tgt[i])  # lag-K wake_verify: the draft's training step (selective cut)
            if placer is not None:
                # BX-SD lane: the target engine DP rank (routed by bxsd_route in the coordinator); fork rows to the lane
                fork = lane is not None and lane[i] is not None and bool(int(lane[i]))
                attrs[BX_DP_RANK_ATTR] = int(placer.place(fork))
                if fork:
                    attrs[BX_FORK_ATTR] = 1
            if attrs:
                sp.add_attributes(attrs)

    def _parse_result_to_batched_data_dict(self, data, result):
        normalize_empty_results(result)
        out = super()._parse_result_to_batched_data_dict(data, result)
        placer = (self.__dict__.get("_bx") or {}).get("placer")
        if placer is not None:
            for r in result:
                rk = getattr(r.sampling_params, BX_DP_RANK_ATTR, None)
                if rk is not None and int(rk) >= 0:
                    placer.release(int(rk), bool(getattr(r.sampling_params, BX_FORK_ATTR, 0)))
        nl = self.__dict__.get("_nl")
        if nl is not None and data is not None and "nl_call_id" in data:
            # The engine request finished: report the driver's call id (append-only; read by cursor, never cleared).
            nl.setdefault("done_call_ids", []).extend(int(x) for x in data["nl_call_id"])
        if nl is not None and data is not None and "nl_no_log" in data:
            nl["cont_inflight"] = nl.get("cont_inflight", 0) - sum(1 for x in data["nl_no_log"] if bool(x))
        if self.__dict__.get("_nl") is not None:
            from thundersync_rl.engine_q import HIDDEN_KEY_ATTR

            out["ts_draft"] = [
                {"key": getattr(r.sampling_params, HIDDEN_KEY_ATTR, None), "rid": int(r.request_id),
                 "cut": int(getattr(r.sampling_params, CUT_ATTR, 0) or 0)}
                for r in result
            ]
            ts = self.__dict__.get("_nl_tree")
            if ts is not None and ts.per_request:
                # CHECKS: per request (tree rounds, accepted nodes, nodes laid out) -- the gate's non-vacuity check
                out["nl_tree_stats"] = [ts.pop_request_stats(int(r.request_id)) for r in result]
                if ts.debug_ru:  # THUNDERSYNC_TREE_DEBUG_RUNNERUP: {position: keyed-race runner-up} per request
                    out["nl_tree_runner_up"] = [ts.pop_runner_ups(int(r.request_id)) for r in result]
        return out

    # -- wake_verify: cut the drafts at a stand-down ------------------------------------------------------------

    def nl_cut_drafts(self, max_target: Optional[int] = None) -> dict[str, int]:
        """Cut every draft request on this engine rank where it stands (thundersync.wake_verify).

        All engine ranks pause through the coordinator (one step boundary for every TP / EP / DP rank, as at a
        sleep), every rank marks the same requests, then the engines unpause: at their next step the marked
        resident requests finish through the stop-word path (the step's sampled token is not appended: the draft
        ends with the tokens it had), and a marked waiting request (not prefilled yet) finishes after its first
        token. Only requests with a hidden-log key (drafts) are cut; continuations / fresh rows are never cut.
        Returns this rank's counts (resident, waiting, already marked)."""
        nl = self.__dict__.get("_nl")
        engine = getattr(self, "dynamic_inference_engine", None)
        if nl is None or engine is None or getattr(self, "_inference_engine_asleep", False):
            return {"cut": 0, "waiting": 0, "marked": 0}
        import asyncio

        import torch
        from megatron.core.inference.engines.dynamic_engine import EngineState

        counts = {"cut": 0, "waiting": 0, "marked": 0, "kept": 0}

        async def pause():
            if torch.distributed.get_rank() == 0:
                await self._client_send(self.inference_client.pause_engines)
            await engine.wait_until(EngineState.PAUSED)

        async def mark():
            counts.update(mark_cut(engine, max_target))

        async def unpause():
            if torch.distributed.get_rank() == 0:
                await self._client_send(self.inference_client.unpause_engines)
            await engine.wait_until(EngineState.RUNNING)

        # pause -> mark -> barrier -> unpause, nothing after the unpause: once a rank's engine runs again its loop may
        # issue TP / EP collectives, and a default-group barrier interleaved with them can deadlock (the B' engine /
        # learner hang). The barrier before the unpause makes every rank's marks precede any engine step.
        loop = self._inference_loop
        asyncio.run_coroutine_threadsafe(pause(), loop).result()
        asyncio.run_coroutine_threadsafe(mark(), loop).result()
        torch.distributed.barrier()
        asyncio.run_coroutine_threadsafe(unpause(), loop).result()
        nl["cut_total"] = nl.get("cut_total", 0) + counts["cut"] + counts["waiting"]
        return counts

    def nl_done_call_ids(self, since: int = 0) -> tuple[list[int], int]:
        """(driver call ids whose engine request finished, from position ``since`` on; the new cursor). Rank 0 only
        builds requests; other ranks return ([], 0). Append-only list: a concurrent append is never lost."""
        nl = self.__dict__.get("_nl")
        ids = (nl or {}).get("done_call_ids") or []
        n = len(ids)
        return list(ids[int(since):n]), n

    def nl_draft_requests(self, max_target: Optional[int] = None) -> list[int]:
        """[draft requests (hidden-log key) this engine rank still holds, of which not marked for a cut]
        (wake_verify: [0, 0] once the cut ones are out)."""
        engine = getattr(self, "dynamic_inference_engine", None)
        if self.__dict__.get("_nl") is None or engine is None or getattr(self, "_inference_engine_asleep", False):
            return [0, 0]
        import asyncio

        from thundersync_rl.engine_q import HIDDEN_KEY_ATTR

        async def count():
            tot = unmarked = 0
            for e in list(engine.requests.values()):
                sp = e.record[-1].sampling_params
                if sp is None or getattr(sp, HIDDEN_KEY_ATTR, None) is None or not cuttable(sp, max_target):
                    continue
                tot += 1
                unmarked += 0 if getattr(sp, CUT_ATTR, 0) else 1
            return [tot, unmarked]

        return asyncio.run_coroutine_threadsafe(count(), self._inference_loop).result()

    # -- hand-out ---------------------------------------------------------------------------------------------

    def nl_pop_drafts(self, keys: list[str], lengths: list[int]) -> dict[str, Any]:
        """{key: (hidden blocks [(ref, rows, bytes)], per-token versions)} of the drafts this rank logged (closed)."""
        nl = self.__dict__.get("_nl")
        log = getattr(self, "_hidden_log", None)
        if nl is None or log is None:
            return {}
        from thundersync_rl.engine_q import HIDDEN_STATS_KEY

        hid = self._hand_out_hidden(log, keys, [0] * len(keys), lengths, True, True)
        stats = hid.pop(HIDDEN_STATS_KEY, {})
        out: dict[str, Any] = {}
        for k, n in zip(keys, lengths):
            vers = nl["tags"].pop(k, n)
            if k in hid:
                out[k] = (hid[k], vers)
        out["__stats__"] = {
            **stats,
            "tag_mismatch": nl["tag_mismatch"],
            "neg_tokens": nl["neg_tokens"],
            "key_draws": nl["keyreg"].draws,
            "version": nl["version"],
        }
        return out

    def nl_close_drafts(self, keys: list[str]) -> None:
        nl = self.__dict__.get("_nl")
        if nl is None:
            return
        for k in keys:
            nl["tags"].close(k)
        self.close_draft_logs(keys)

    # -- BX-SD fork lane ------------------------------------------------------------------------------------------

    def nl_bx_info(self) -> Optional[dict]:
        """This engine rank's DP rank (the coordinator identity mp-coord-<dp_rank>) and the global ranks it steps in
        lockstep with (its expert-parallel and model-parallel groups). Read by the driver to build the lane layout
        without a collective (an engine-time collective could interleave with the engine loop's)."""
        engine = getattr(self, "dynamic_inference_engine", None)
        if engine is None:
            return None
        import torch.distributed as dist
        from megatron.core.utils import get_pg_rank

        pgc = engine.pg_collection
        return {
            "global_rank": int(dist.get_rank()),
            "dp_rank": int(get_pg_rank(pgc.dp)),
            "lockstep": [sorted(int(x) for x in dist.get_process_group_ranks(g)) for g in (pgc.ep, pgc.mp)],
        }

    def nl_bx_begin_phase(self, n_bulk_new: int, n_fork_new: int, b: int, it_bulk: float, fork_depth_max: float,
                          groups: list, row_cap: Optional[int] = None) -> dict:
        """Open a BX-SD phase. Every rank: the L1 horizon (lane rows draft only while behind it). The
        request-building rank (global rank 0, which owns the inference client): the placer's quotas."""
        import torch.distributed as dist

        from thundersync_rl import bxsd_lane

        bx = self.__dict__.setdefault("_bx", {"iters": 0, "horizon": 0.0, "bulk": 0.0, "l1_on": 0, "l1_off": 0, "routed": 0,
                                              "misrouted": 0})
        engine = getattr(self, "dynamic_inference_engine", None)
        if engine is not None and "dp_rank" not in bx:
            self._bx_install(engine)
        ts = self.__dict__.get("_nl_tree")
        s_hat = 1.0
        if ts is not None and int(ts.stats.get("crit_iters", 0)) > 0:
            # measured tokens per iteration of the critical tails so far (run total); 1 before any (report only)
            s_hat = max(1.0, float(ts.stats["crit_tokens"]) / float(ts.stats["crit_iters"]))
        bx["bulk"] = float(it_bulk)  # L1: the deepest non-lane deadline row (tokens; an upper bound in iterations)
        bx["horizon"] = max(float(it_bulk), float(fork_depth_max) / s_hat)  # report: the predicted lane phase
        bx["iters"] = 0
        if int(dist.get_rank()) != 0:
            return {}
        if bx.get("placer") is None:
            layout = bxsd_lane.LaneLayout(groups=tuple(tuple(int(r) for r in g) for g in groups))
            bx["placer"] = bxsd_lane.LanePlacer(layout, row_cap=None if row_cap is None else int(row_cap))
        p = bx["placer"]
        prev = [list(p.phase_placed[r]) for r in range(p.layout.dp_size)]  # where the previous phase's rows went
        b = p.begin_phase(int(n_bulk_new), int(n_fork_new), int(b))
        return {"b": int(b), "live": [list(v) for v in p.loads().values()], "prev_placed": prev, "placed": p.placed,
                "released": p.released, "horizon": round(bx["horizon"], 1), "s_hat": round(s_hat, 3)}

    def _bx_install(self, engine: Any) -> None:
        """Every engine rank, once: count routed / misrouted requests (a request carrying another rank's target is a
        routing bug), and the L1 rule before each tree plan (lane fork rows: width cap W while behind the horizon,
        0 otherwise; proposals only, exact by construction)."""
        from megatron.core.utils import get_pg_rank

        from thundersync_rl import bxsd_lane

        bx = self._bx
        bx["dp_rank"] = int(get_pg_rank(engine.pg_collection.dp))
        orig_add = engine.add_request

        def add_request(request_id, prompt, sampling_params=None, *a, **kw):
            tgt = getattr(sampling_params, BX_DP_RANK_ATTR, None) if sampling_params is not None else None
            if tgt is not None and int(tgt) >= 0:
                bx["routed" if int(tgt) == bx["dp_rank"] else "misrouted"] += 1
            return orig_add(request_id, prompt, sampling_params, *a, **kw)

        engine.add_request = add_request
        ts = self.__dict__.get("_nl_tree")
        l1 = os.environ.get("THUNDERSYNC_BX_LANE", "").strip() == "l1"
        if ts is None or not l1:
            return
        orig_plan = ts.plan
        cap0: dict[int, int] = {}

        def plan(ctx, is_dummy):
            # one count per engine step on every rank of a lockstep group (dummy steps included: they are steps)
            it = bx["iters"]
            bx["iters"] = it + 1
            if not is_dummy and ts.tails:
                if len(cap0) > 4 * max(len(ts.tails), 1024):
                    for k in [k for k in cap0 if k not in ts.tails]:
                        del cap0[k]
                forks = []
                for rid, t in list(ts.tails.items()):
                    req = engine.get_request(int(rid))
                    if getattr(req.sampling_params, BX_FORK_ATTR, 0):
                        forks.append((int(rid), cap0.setdefault(int(rid), int(t.w_cap)),
                                      t.budget_end - (t.plen + len(req.generated_tokens))))
                if forks:
                    ci = int(ts.stats.get("crit_iters", 0))
                    s_crit = float(ts.stats["crit_tokens"]) / ci if ci > 0 else 1.0
                    on = bxsd_lane.l1_drafts([r for _, _, r in forks], s_crit, max(bx["bulk"] - it, 0.0))
                    for (rid, w0, _r), d in zip(forks, on):
                        ts.set_width_cap(rid, w0 if d else 0)
                        bx["l1_on" if d else "l1_off"] += 1
            return orig_plan(ctx, is_dummy)

        ts.plan = plan

    def nl_bx_stats(self) -> dict:
        bx = self.__dict__.get("_bx") or {}
        return {k: bx[k] for k in ("routed", "misrouted", "l1_on", "l1_off", "iters") if k in bx}

    def nl_engine_stats(self) -> dict[str, Any]:
        nl = self.__dict__.get("_nl")
        if nl is None:
            return {}
        ws = self.wide_sampler_stats() or {}
        return {
            "version": nl["version"],
            "refits": nl["refits"],
            "tag_mismatch": nl["tag_mismatch"],
            "straddle_samples": nl.get("straddle_samples", 0),
            "neg_tokens": nl["neg_tokens"] + int((self.nl_mtd_stats() or {}).get("neg_tokens", 0))
            + int((self.nl_tree_stats() or {}).get("neg_tokens", 0)),
            "key_draws": nl["keyreg"].draws,
            "key_violations": nl["keyreg"].violations,
            "open_tag_keys": len(nl["tags"]),
            # Keyed sampler S: calls that took S vs all wide-sampler calls (R35 3(h)).
            "s_calls": int(ws.get("s_calls", 0)),
            "wide_calls": int(ws.get("calls", 0)),
        }
