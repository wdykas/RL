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


def mark_cut(engine: Any) -> dict[str, int]:
    """Mark every draft request (hidden-log key) of a PAUSED dynamic engine for a cut (``nl_cut_drafts``): resident
    requests finish at the engine's next step through the stop-word path (that step's sampled token is not appended);
    a waiting request (not prefilled) finishes after its first token. Each marked request carries ``CUT_ATTR`` back
    with its result. Continuations / fresh rows (no hidden-log key) are never touched."""
    from thundersync_rl.engine_q import HIDDEN_KEY_ATTR

    counts = {"cut": 0, "waiting": 0, "marked": 0}
    waiting = {int(r) for r in engine.waiting_request_ids}
    for rid, entry in list(engine.requests.items()):
        sp = entry.record[-1].sampling_params
        if sp is None or getattr(sp, HIDDEN_KEY_ATTR, None) is None:
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

            kv = check_engine_reservation(self.dynamic_inference_engine)
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
        print(
            f"[nolanes engine] rank={getattr(self, 'rank', '?')} version tags + hidden keys + key registry installed "
            f"(logging={logging_rank}, version={version})",
            flush=True,
        )
        return logging_rank

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
        """Colocated engine (docs/colo/THEORY.md, "XPR"): the wake IS the refit; no suspend_for_refit
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
        """Noise-key request ids of the sampled rows (the engine id; a key_reuse negative's override)."""
        nl = self.__dict__.get("_nl")
        if nl is None or nl["neg"] != "key_reuse":
            return rids
        out = rids.copy()
        for i, rid in enumerate(rids.tolist()):
            entry = engine.requests.get(int(rid))
            if entry is None:
                continue
            k = getattr(entry.record[-1].sampling_params, KEY_RID_ATTR, None)
            if k is not None:
                out[i] = int(k)
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
        for i, sp in enumerate(sps):
            attrs = {}
            if no_log is None or not bool(no_log[i]):
                attrs[HIDDEN_KEY_ATTR] = f"{NL_KEY_PREFIX}{next(nl['counter'])}"
            if key_rid is not None and int(key_rid[i]) >= 0:
                attrs[KEY_RID_ATTR] = int(key_rid[i])
            if attrs:
                sp.add_attributes(attrs)

    def _parse_result_to_batched_data_dict(self, data, result):
        normalize_empty_results(result)
        out = super()._parse_result_to_batched_data_dict(data, result)
        nl = self.__dict__.get("_nl")
        if nl is not None and data is not None and "nl_no_log" in data:
            nl["cont_inflight"] = nl.get("cont_inflight", 0) - sum(1 for x in data["nl_no_log"] if bool(x))
        if self.__dict__.get("_nl") is not None:
            from thundersync_rl.engine_q import HIDDEN_KEY_ATTR

            out["ts_draft"] = [
                {"key": getattr(r.sampling_params, HIDDEN_KEY_ATTR, None), "rid": int(r.request_id),
                 "cut": int(getattr(r.sampling_params, CUT_ATTR, 0) or 0)}
                for r in result
            ]
        return out

    # -- wake_verify: cut the drafts at a stand-down ------------------------------------------------------------

    def nl_cut_drafts(self) -> dict[str, int]:
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

        counts = {"cut": 0, "waiting": 0, "marked": 0}

        async def pause():
            if torch.distributed.get_rank() == 0:
                await self._client_send(self.inference_client.pause_engines)
            await engine.wait_until(EngineState.PAUSED)

        async def mark():
            counts.update(mark_cut(engine))

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

    def nl_draft_requests(self) -> list[int]:
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
                if sp is None or getattr(sp, HIDDEN_KEY_ATTR, None) is None:
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
            "neg_tokens": nl["neg_tokens"],
            "key_draws": nl["keyreg"].draws,
            "key_violations": nl["keyreg"].violations,
            "open_tag_keys": len(nl["tags"]),
            # Keyed sampler S: calls that took S vs all wide-sampler calls (R35 3(h)).
            "s_calls": int(ws.get("s_calls", 0)),
            "wide_calls": int(ws.get("calls", 0)),
        }
