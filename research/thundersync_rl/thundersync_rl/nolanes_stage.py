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
"""The verify stage of the nolanes mode, inside the SingleController actor (nolanes.py has the conditions).

``VerifyStage.run(meta, k)`` replaces the drafted rows of training step k's batch by verified output rows:
1. the rollout records of the batch's groups (``record_sink``, kept per group by the rollout manager) give each
   draft's tokens, engine log q and engine draft reference; the engine hands out each draft's logged head inputs
   and per-token version tags (``nl_pop_drafts``);
2. ⊥-BV of every prompt's pool (drafts in sample-index order) against theta_k on the learner: the existing
   verify0 machinery (``SpeculativeGeneration.verify``: null_bv pools, DS2, grouping, F3 row seeds);
3. each group's G slots: whole accepted drafts, prefixes to continue (residual / bonus token included), and fresh
   slots; continuations and fresh slots run as NEW engine requests at theta_k (``AsyncRolloutImpl.run_rows``),
   and every row is scored by the environment;
4. the group's rows in the data plane are rewritten (G rows, tagged with the verified version), generation
   log-probs = verified p / theta_k engine log-probs (E7).
"""

from __future__ import annotations

import asyncio
import math
import os
import time
import zlib
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import ray
import torch

from thundersync_rl import edv, nolanes
from thundersync_rl.nolanes import (
    VERIFIED_TAG,
    DraftRow,
    NolanesConfig,
    OutputRow,
    StepStats,
    continuation_budget,
    drafts_of_record,
    outputs_of_group,
    shift_segments,
    version_segments,
)
from thundersync_rl.finalize_pipeline import step_output_ids, trained_once_diff


def _spec_class():
    """SpeculativeGeneration with the nolanes hooks (imported lazily: heavy module)."""
    from thundersync_rl.speculative import SpeculativeGeneration, _null_result

    class NolanesSpec(SpeculativeGeneration):
        def _null_plan(self, d, res):
            # Engine-sourced q: the verifier's log q of every accepted token is the engine's, bitwise, so the
            # stock q'/q correction in behavior_logprobs never fires and generation log-probs are the verified p (E7).
            n_acc = int(res.get("accepted", 0))
            dq, eq = res.get("draft_logq") or [], d[6] if len(d) > 6 else None
            if eq is not None and any(float(a) != float(b) for a, b in zip(list(eq)[:n_acc], list(dq)[:n_acc])):
                raise RuntimeError(f"[nolanes] verifier log q != engine log q on draft {d[4]} (E1/E7)")
            plan = super()._null_plan(d, res)
            plan.engine_key = d[4]  # the draft this output came from
            g = len(d[2])
            if n_acc < g:
                # Per-row rejection diagnostics (R30 acceptance sanity): position, draft versions, log p - log q at
                # the rejected token and the mean |log p - log q| over the accepted ones.
                lpx = res.get("token_logprobs") or []
                lq = list(eq) if eq is not None else []
                at = (float(lpx[n_acc]) - float(lq[n_acc])) if n_acc < min(len(lpx), len(lq)) else float("nan")
                acc = [abs(float(a) - float(b)) for a, b in zip(lpx[:n_acc], lq[:n_acc])]
                rec = {
                    "key": d[4], "g": g, "tau": n_acc, "versions": sorted({int(v) for v, _, _ in (d[5] or ())}),
                    "dlog_at_tau": at, "mean_abs_dlog_acc": (sum(acc) / len(acc)) if acc else 0.0,
                    "m": res.get("null_m"), "next": int(res.get("next", -1)),
                }
                diag = self.__dict__.setdefault("_nl_reject_diag", [])
                diag.append(rec)
            return plan

        def _null_apply(self, pools, groups, flat, results):
            if nolanes.negative() != "content_m":
                return super()._null_apply(pools, groups, flat, results)
            # NEGATIVE CONTROL (E3): the envelope of a draft chosen from its own first token's engine log q
            # against the pool's median: high-q first tokens get the largest envelope (more often ⊥).
            med = self.__dict__.setdefault("_nl_pool_median", {})
            for i in sorted(results):
                run, r, d = pools[groups[i]], results[i], flat[i][1]
                if groups[i] not in med:
                    vals = [float(flat[j][1][6][0]) for j in run.env if len(flat[j][1][6])]
                    med[groups[i]] = float(np.median(vals)) if vals else 0.0
                m_counts = run.state.next_m()
                grid = sorted(r.get("null_grid", {}))
                if m_counts is None or len(grid) < 2 or not len(d[6]):
                    run.decide(i, r, False, 0)
                    continue
                m = grid[-1] if float(d[6][0]) > med[groups[i]] else grid[0]
                run.inflight -= 1
                tau, y, _ = r["null_grid"][m]
                run.state.record(y >= 0)
                run.out[i] = _null_result(r, m)
                self.stats["nl/neg_content_m_rows"] = self.stats.get("nl/neg_content_m_rows", 0.0) + 1.0
                if r.get("edv_deferred") or r.get("edv_rewalk"):
                    # EDV rows the negative moved: deferred rows (walked as whole) and rows of an EDV re-walk.
                    self.stats["nl/neg_content_m_edv_rows"] = self.stats.get("nl/neg_content_m_edv_rows", 0.0) + 1.0

    return NolanesSpec


def edv_e_of_step(e: float) -> float:
    """This step's EDV threshold: thundersync.edv_threshold, or THUNDERSYNC_NL_DEBUG_EDV_E (X1 gate tool, CHECKS only)."""
    v = os.environ.get("THUNDERSYNC_NL_DEBUG_EDV_E", "").strip()
    return float(v) if v else float(e)


class VerifyStage:
    """Driver side of the nolanes mode; owned by the SingleController actor."""

    def __init__(self, sc: Any, master_config: dict, cfg: NolanesConfig):
        self.sc = sc
        self.mc = master_config
        self.cfg = cfg
        # EDV (edv.py, DESIGN §22.9, §22.15): a fixed threshold e; None = off.
        self.edv_e = float(cfg.edv_threshold) if cfg.edv_threshold is not None else None
        # Train overlap (R31): set by run() when the stage trained the step's groups itself.
        self.trained_in_stage = False
        self.stage_has_valid_tokens = False
        self._groups_trained = 0
        # Gate tool (X1; THUNDERSYNC_NL_DEBUG_ prefix, refused without CHECKS): hold new admissions for the whole step
        # (the admission gate reads min(its version, max(admit_hold_version, trainer_version - 1))), i.e. until the
        # optimizer step bumps the trainer version, ~1 s before the refit lands. New drafts then prefill and start
        # decoding at theta_k and are still decoding when the refit lands: mixed-version drafts (rejected / continued
        # under stress) with admit_after_refit on, at any row length and without an engine-step sleep (R35 2(e);
        # the pre-admit_after_refit admission timing). Timing only, content independent.
        self.debug_admit_after_tails = os.environ.get("THUNDERSYNC_NL_DEBUG_ADMIT_AFTER_TAILS", "") == "1"
        self.admit_hold_version: Optional[int] = None
        self.g = int(master_config["grpo"]["num_generations_per_prompt"])
        self.n = nolanes.drafts_per_prompt(cfg, self.g)
        self.max_new = int(master_config["policy"]["generation"]["max_new_tokens"])
        self.max_seq = master_config["policy"].get("max_total_sequence_length")
        self.lag = int(master_config["async_rl"]["sampler"].get("max_lookahead_versions", 1))
        self.seed = int(master_config["grpo"].get("seed", 0))
        if self.debug_admit_after_tails and self.lag < 1:
            raise ValueError("THUNDERSYNC_NL_DEBUG_ADMIT_AFTER_TAILS needs max_lookahead_versions >= 1 (lag 0 would wait "
                             "on its own step)")
        self.records: dict[str, tuple[Any, Any]] = {}
        self.spec = None
        self.eod: Optional[int] = None
        self.history: list[dict[str, Any]] = []
        self.neg = nolanes.negative()
        self.prev_from_verify = bool(cfg.prev_logprobs_from_verify)
        # GRPO-Z (grpo_z.py): row advantages from the siblings' draft rewards, rows train on arrival. loo_c scale s_k:
        # a constant at the first step, then the previous step's mean within-group std of final rewards.
        self.grpo_z = str(getattr(cfg, "grpo_z", "off") or "off")
        self.gz_scale = 1.0
        # Colocated engine (research/colo/MODEL.md v2, XPR-1): set in setup() from the generation backend.
        self.colocated = False
        nolanes.apply_driver_env(cfg)

    # -- setup ----------------------------------------------------------------------------------------------

    def record_sink(self, group_id: str, input_sample: Any, record: Any) -> None:
        self.records[group_id] = (input_sample, record)

    def _gen_wg(self):
        return self.sc._gen._policy.worker_group

    def setup(self, version: int) -> None:
        """After the engine exists and holds theta_version: verifier, sampler, logs, version tags, head stash."""
        if self.cfg.train_overlap and not getattr(self.sc._advantage_stage_config, "shardable", False):
            raise RuntimeError("[nolanes] train_overlap needs a group-local (shard-invariant) advantage estimator: "
                               "each group's advantages are computed when it becomes final")
        self.colocated = bool(self.sc._gen.blocks_training())
        self.slots = None
        if self.colocated:
            # Engine row slots, continuations first (engine_slots.py): capacity = max_requests x engine DP by rule;
            # every generation call of the driver takes one before it reaches Ray / the engine.
            from thundersync_rl import engine_slots

            mcore = self.sc._gen.cfg["mcore_generation_config"]
            engine_slots.require_least_loaded(mcore)
            cap, mr, dp = engine_slots.capacity_of(mcore, len(self._gen_wg().workers))
            impl = getattr(getattr(self.sc, "_rollout_manager", None), "_impl", None)
            self.slots = engine_slots.install([self.sc._gen, getattr(impl, "_policy_generation", None)], cap)
            print(f"[engine slots] capacity={cap} (max_requests {mr} x engine DP {dp}), continuations first", flush=True)
        cls = _spec_class()
        tok = self.sc._rollout_manager._impl._tokenizer
        self.spec = cls(
            self.sc._gen,
            self.sc._trainer,
            vocab_limit=len(tok),
            head_k=0,
            draft_budget=self.max_new,
            max_new_tokens=self.max_new,
            pad_token_id=tok.pad_token_id,
            draft_variants=1,
            variant_eps=0.0,
            verify_mode="block",
            verify_precision="bf16",
            # resolved by nolanes.configure_driver (effective_verify_batch_tokens)
            verify_batch_tokens=int(self.cfg.verify_batch_tokens),
            longest_first=False,
            verify_on=str(self.cfg.verify_on),
            # Engine verify: a verification request is prompt + draft + one sampled token; drafts longer than the
            # engine's max_model_len - 1 are verified as their prefix that fits (a cut at a fixed position: exact).
            engine_max_len=(int((self.sc._gen.cfg.get("mcore_generation_config") or {}).get("max_model_len")
                                or self.max_seq) if str(self.cfg.verify_on) == "engine" else None),
            pool_weights=None,
            fused_verify_train=False,
            fused_max_retained_tokens=0,
            verify_source="engine",
            hidden_transport="object_store",
            sticky_delta=None,
            null_bv=True,
            null_bv_policy={
                "logratio_var": float(self.cfg.null_bv_logratio_var),
                "ref_len": self.max_new,
                "fixed_m": self.cfg.null_bv_fixed_m,
                "prompt_drop": bool(int(self.cfg.spare_prompts)),
                "tail_weight": float(self.cfg.null_bv_tail_weight),
                "calibrate": False,
            },
            max_seq_len=self.max_seq,
            seed=self.seed,
            seed_stream=self.cfg.seed_stream,
            engine_sampler="wide",
            sampler_segment=int(version),
            wide_key64=bool(self.cfg.wide_sampler_key64),
            null_bv_spare_prompts=int(self.cfg.spare_prompts),
        )
        if int(self.cfg.spare_prompts):
            # XPR-S prompt selection counted in SLOTS (a pool holds every group of a duplicated prompt): keep the
            # undropped pools in the fixed order while they fit in the trained slots, then fill a shortfall from the
            # dropped ones in order (they decode fresh). Inputs: ⊥ counts (dropped), slot counts and the order only.
            spec_s = self.spec
            spare_slots = int(self.cfg.spare_prompts) * self.g

            def _select_slots(score: Optional[dict] = None) -> None:
                order = list(spec_s._null_all_slots)
                slots = {k: int(spec_s._null_all_slots[k]) for k in order}
                train = sum(slots.values()) - spare_slots
                dropped = spec_s.null_dropped_keys
                cand = [k for k in order if k not in dropped]
                if score is not None:  # NEGATIVE CONTROLS only: rank by content
                    cand = sorted(order, key=lambda k: (-score.get(k, 0.0), order.index(k)))
                kept, used = [], 0
                for k in cand:
                    if used + slots[k] <= train:
                        kept.append(k); used += slots[k]
                for k in [k for k in order if k in dropped and k not in kept]:
                    if used + slots[k] <= train:
                        kept.append(k); used += slots[k]
                spec_s.null_selected_keys = kept
                for key in [k for k in spec_s.plans if k not in set(kept)]:
                    del spec_s.plans[key]

            spec_s.__dict__["_select_prompts"] = _select_slots
            if self.neg == "spare_by_logp":
                spec_s.__dict__["_select_prompts"] = lambda: _select_slots(spec_s.__dict__.get("_neg_pool_score", {}))
        if self.neg == "spare_tail_drop":
            # NEGATIVE CONTROL (XPR-S, row-level selection on content): the pool walk's drop-IS mode -- an output that
            # would need a continuation is dropped and the next draft takes its slot -- without the importance weight
            # that would make it unbiased. Kept rows are accepted drafts only (law tilted toward q): must FAIL.
            self.spec.__dict__["null_bv_drop_is"] = True
        if self.neg == "spare_by_content":
            # NEGATIVE CONTROL (XPR-S): the verifier keeps the P pools whose verified plans continue the fewest rows
            # (a selection on content), replacing the count-only prompt selection; the law check must fail.
            spec_ = self.spec

            def _select_by_content() -> None:
                order = list(spec_._null_all_slots)
                train = len(order) - spec_.null_bv_spare_prompts
                cont = {k: sum(1 for p_ in spec_.plans.get(k, []) if not p_.complete) for k in order}
                kept = sorted(order, key=lambda k: (cont[k], k in spec_.null_dropped_keys, order.index(k)))[:train]
                spec_.null_selected_keys = kept
                for key in [k for k in spec_.plans if k not in set(kept)]:
                    del spec_.plans[key]

            spec_.__dict__["_select_prompts"] = _select_by_content  # instance dict (the class forwards unknown attrs)
        self.spec.verify_in_train = False
        self.spec.verify_yield = False
        # Lazy ⊥-BV waves (``_NullPoolRun``, ``SpeculativeGeneration._null_wave``): each wave forwards ceil(s x M)
        # rows per active pool (s slots left), and spare pools only once an earlier pool drops, so only drafts that
        # can still fill a trained slot are verified. Which rows are forwarded is a function of the counts and the
        # fixed order, so the decisions are the count process's, walked in drafting order. EDV draws every row's
        # eta before the walk, so it forwards every draft of a pool in one wave.
        self.spec.null_one_wave = self.edv_e is not None
        self.eod = int(self.spec.eod)
        salt = zlib.crc32(f"wide/{self.seed}/{self.cfg.seed_stream}/seg{int(version)}".encode()) | (
            (self.seed & 0x7FFFFFFF) << 32
        )
        ok = ray.get(
            self._gen_wg().run_all_workers_single_data(
                "nl_engine_setup", salt=salt, key64=bool(self.cfg.wide_sampler_key64), seal_blocks=True,
                version=int(version),
            )
        )
        if not any(x is True for x in ok) or any(x is False for x in ok):
            raise RuntimeError(f"[nolanes] engine setup failed per rank: {ok}")
        if self.colocated:
            # Colocated: the learner's weights are not resident while the engine is awake (one model's weights at a
            # time in the shared allocation region), so theta_version's head is stashed at the first stand-down.
            self._colo_stash_pending = int(version)
        else:
            self.stash_head(version)
        self.engine_verify = str(self.cfg.verify_on) == "engine"
        if self.engine_verify:
            # DX: the inference pool verifies (one-token prefill requests, head-input capture, ⊥-BV decided on the
            # engine rank that holds the rows); q rows need the LM head of every drafting version on the engine ranks.
            from thundersync_rl.speculative import EngineVerifier

            self.spec.engine_verifier = EngineVerifier(self._gen_wg())
            self.stash_engine_head(version)
        print(f"[nolanes] verify stage ready: G={self.g} N={self.n} lag={self.lag} version={version} "
              f"negative={self.neg or 'none'}", flush=True)

    async def cut_drafts(self) -> dict:
        """wake_verify: cut every draft still decoding on the engine (``nl_cut_drafts``) and wait until each is back.

        Stop rule, counted: no draft request on any engine rank AND (with engine slots) every granted draft call's engine
        request reported finished by the engine (``nl_done_call_ids``): a result still travelling back to the driver is
        safe and drains during training.
        Under the slots' cut mode (``cut_waiting_drafts``) a draft call that has not been granted a slot never reaches
        an engine (it ends at the gate with zero tokens), so a granted, unreturned call is the only way a draft can
        still land on an engine: the count is complete, whatever is still in dispatch prep. The cut groups'
        post-processing (env scoring, buffer commit) needs no engine and overlaps training. Without slots: no draft on
        the engines and no rollout group in flight (as before). Draft requests that reach an engine after a cut round
        are cut by the next round."""
        wg = self._gen_wg()
        t0 = time.perf_counter()
        rounds, cut, waiting = 0, 0, 0
        while True:
            rs = await asyncio.to_thread(ray.get, wg.run_all_workers_single_data("nl_cut_drafts"))
            rounds += 1
            # every TP / EP rank of an engine replica holds the same requests: the replica's count is the rank max
            cut = max(cut, max((int(r.get("cut", 0)) for r in rs if r), default=0))
            waiting = max(waiting, max((int(r.get("waiting", 0)) for r in rs if r), default=0))
            while True:
                cnt = await asyncio.to_thread(ray.get, wg.run_all_workers_single_data("nl_draft_requests"))
                left = sum(int(c[0]) for c in cnt if c)
                unmarked = sum(int(c[1]) for c in cnt if c)
                inflight = len(getattr(self.sc, "_inflight_by_group_id", {}) or {})
                pending = None
                if self.slots is not None:
                    # engine-finished call ids since the last poll (rank 0 builds every request; cursor per rank)
                    cur = self.__dict__.setdefault("_done_cursor", {})
                    reps = await asyncio.to_thread(ray.get, [w.nl_done_call_ids.remote(cur.get(r, 0))
                                                             for r, w in enumerate(wg.workers)])
                    for r, (ids, n) in enumerate(reps):
                        cur[r] = n
                        self.slots.note_engine_done(ids)
                    pending = self.slots.drafts_not_engine_done()
                if left == 0 and (pending == 0 if pending is not None else inflight == 0):
                    return {"cut_rounds": rounds, "cut_resident_max": cut, "cut_waiting_max": waiting,
                            "cut_s": round(time.perf_counter() - t0, 2), "cut_inflight_groups": inflight}
                if time.perf_counter() - t0 > 300:
                    raise RuntimeError(f"[nolanes] wake_verify cut: {left} draft requests on the engines and {inflight} "
                                       f"rollout groups in flight after {rounds} cut rounds / 300 s")
                if unmarked > 0:
                    break  # drafts that reached an engine after the last round: cut them too
                await asyncio.sleep(0.05)

    def defers_post_step_wake(self) -> bool:
        """wake_verify (colocated): the loop does not wake the engine after the optimizer step; the next step's stage
        verifies with the engine still stood down and its own wake installs theta_{k+1} (the announced version)."""
        return bool(getattr(self.cfg, "wake_verify", False) and self.colocated)

    def stash_head(self, version: int) -> None:
        """Learner: stash theta_version's LM head (q rows of tokens drafted at version), keep lag + 1 versions."""
        keep = list(range(max(0, version - self.lag - 1), version + 1))
        # Materialize the optimizer step first: with overlap_param_gather (learner DP > 1) the updated parameters
        # are gathered only at the next forward or refit, and the stash read the previous version's head shard
        # (Super-L22 4-node check: engine q check 289125 of 1046821 tokens exact at the first mixed step).
        self.sc._trainer.sync_params_before_refit()
        ray.get(
            self.sc._trainer.worker_group.run_all_workers_single_data("stash_head", version=int(version), keep=keep)
        )

    def announce_version(self, version: int) -> None:
        """Before the refit to ``version``: every engine rank records it (set absolutely at its suspend)."""
        vs = ray.get(self._gen_wg().run_all_workers_single_data("nl_announce_version", version=int(version)))
        if any(v != int(version) for v in vs):
            raise RuntimeError(f"[nolanes] engine ranks did not take the announced version {version}: {vs}")

    def check_engine_version(self, version: int) -> None:
        """After every refit: EVERY engine rank's resident version tag equals the trainer version."""
        vs = ray.get(self._gen_wg().run_all_workers_single_data("nl_version"))
        if not vs or any(v != version for v in vs):
            raise RuntimeError(f"[nolanes] engine version tags {vs} != trainer version {version}")
        if getattr(self, "engine_verify", False):
            self.stash_engine_head(version)

    def stash_engine_head(self, version: int) -> None:
        """Engine verify: every engine rank stashes its live LM head as ``version`` (it holds theta_version right after
        the refit); keeps the versions a batch's drafts can carry (lag + 1 back)."""
        keep = list(range(max(0, version - self.lag - 1), version + 1))
        ray.get(self._gen_wg().run_all_workers_single_data("stash_head", version=int(version), keep=keep))

    def _edv_configure(self) -> str:
        """EDV shadow counter (DESIGN §22.8): ``<run dir>/edv_shadow.jsonl``, log only. Configured at the first step:
        the stage is built in the controller's __init__, before the controller's Logger exists."""
        from thundersync_rl import edv_shadow

        if edv_shadow._PATH is None:
            edv_shadow.configure(getattr(getattr(self.sc, "_logger", None), "base_log_dir", None))
            if edv_shadow._PATH is None:
                raise RuntimeError("[edv shadow] no run dir: the controller has no logger base_log_dir")
        return edv_shadow._PATH

    # -- the stage ------------------------------------------------------------------------------------------

    async def run(self, meta: Any, version: int) -> Any:
        from nemo_rl.data_plane.schema import GROUP_ID_TAG
        from thundersync_rl import edv_shadow

        self._edv_configure()
        edv_shadow.set_step(int(version))

        st = StepStats()
        t0 = time.perf_counter()
        mixed_drafts: set[int] = set()  # id() of the drafts sampled under more than one version
        if self.debug_admit_after_tails:
            self.admit_hold_version = int(version) - 1
        # Groups in a fixed order (R2): dataset prompt index, then group id. Never arrival or engine timing.
        gids: list[str] = []
        for t in meta.tags:
            gid = t[GROUP_ID_TAG]
            if gid not in gids:
                gids.append(gid)
        prompt_idx = {t[GROUP_ID_TAG]: t.get("prompt_idx") for t in meta.tags}
        gids.sort(key=lambda g: (str(prompt_idx.get(g)), g))
        missing = [g for g in gids if g not in self.records]
        if missing:
            raise RuntimeError(f"[nolanes] no rollout record for groups {missing[:5]} (record sink not installed?)")
        recs = {g: self.records.pop(g) for g in gids}
        old_ids = {g: [s for s, t in zip(meta.sample_ids, meta.tags) if t[GROUP_ID_TAG] == g] for g in gids}
        disp_version = {
            g: min(int(t.get("weight_version", version)) for t in meta.tags if t[GROUP_ID_TAG] == g) for g in gids
        }
        drafts = {g: drafts_of_record(g, recs[g][0]["message_log"], recs[g][1]) for g in gids}
        for g in gids:
            if len(drafts[g]) != self.n:
                raise RuntimeError(f"[nolanes] group {g}: {len(drafts[g])} drafts, expected N={self.n}")
        n_cut = 0
        for g in gids:
            for d in drafts[g]:
                if not d.finished:
                    if not self.cfg.wake_verify:
                        raise RuntimeError(f"[nolanes] draft {g}/{d.index} was cut at a stand-down without wake_verify")
                    if d.tokens and d.tokens[-1] == self.eod:
                        d.finished = True  # its last kept token ended it: finished by its own content, not by the cut
                    else:
                        n_cut += 1
        if self.cfg.wake_verify:
            st.crit["cut_drafts"] = n_cut
        # Engine hand-out: head inputs + version tags of every logged draft.
        flat_d = [d for g in gids for d in drafts[g]]
        keys = [d.key for d in flat_d if d.key is not None]
        lens = [len(d.tokens) for d in flat_d if d.key is not None]
        merged: dict[str, Any] = {}
        hstats: list[dict] = []
        for r in await asyncio.to_thread(
            ray.get, self._gen_wg().run_all_workers_single_data("nl_pop_drafts", keys=keys, lengths=lens)
        ):
            if r:
                hstats.append(r.pop("__stats__", {}))
                merged.update(r)
        st.times["handout"] = time.perf_counter() - t0
        if self.colocated:
            # Colocated: the learner verifies with the engine stood down (megatron's InferenceMode is process-wide
            # while the engine is awake). The lookahead batches' requests are paused, KV kept (persist), and resume
            # at the wake below: their tokens so far stay drafts of their own versions.
            tc = time.perf_counter()
            await asyncio.to_thread(self.sc._gen.finish_generation)
            # Resident colocated learner (MODEL.md v2 §11): the grad buffers stay through the verify. Releasing them here
            # (resize_(0)) and growing them back for training after the verify transients took their place left
            # +11 GiB of re-grown buffers and +20 GiB of allocator holes at the training peak (Super-L8 snapshot,
            # research/colo/mem/, XPR vs XS) -- the speculative arm's 4-node OOM.
            await asyncio.to_thread(ray.get, self.sc._trainer.worker_group.run_all_workers_single_data(
                "prepare_for_lp_inference", keep_train_buffers=True))
            pend = self.__dict__.pop("_colo_stash_pending", None)
            if pend is not None:
                await asyncio.to_thread(self.stash_head, pend)
            st.times["colo_standdown"] = time.perf_counter() - tc
        # Draft tuples per prompt key (pools), sample-index order.
        from thundersync_rl.engine_q import HiddenRefs

        pools: dict[tuple[int, ...], list[tuple]] = defaultdict(list)
        slots: dict[tuple[int, ...], int] = defaultdict(int)
        pool_groups: dict[tuple[int, ...], list[str]] = defaultdict(list)
        draft_by_key: dict[str, DraftRow] = {}
        vkey_of: dict[int, str] = {}  # id(DraftRow) -> draft key (EDV)

        def walk_empty(g: str, pk: tuple, d: DraftRow) -> None:
            """An empty / unverifiable draft stays in its pool at its drafting index: zero tokens, decided by its keyed
            ⊥ coin in the walk (⊥ with probability 1 - 1/M, else its slot decodes fresh), so the pool's ⊥ pattern and
            prompt drop never depend on which drafts were empty (slot timing)."""
            from thundersync_rl.engine_q import HiddenRefs as _HR

            vk = f"{version}:{g}:{d.index}"
            draft_by_key[vk] = d
            vkey_of[id(d)] = vk
            pools[pk].append((torch.tensor(d.prompt, dtype=torch.long), -1, [], False, vk, [], [], _HR()))
        for g in gids:
            pk = tuple(drafts[g][0].prompt)
            slots[pk] += self.g
            pool_groups[pk].append(g)
            for d in drafts[g]:
                st.drafts += 1
                if not d.tokens:
                    # Cut before its first token (at the slot gate, or in the step it was first scheduled): walked at its
                    # drafting index by its keyed ⊥ coin (speculative.empty_draft_uniform), never dropped.
                    st.crit["empty_drafts"] = st.crit.get("empty_drafts", 0) + 1
                    walk_empty(g, pk, d)
                    continue
                got = merged.get(d.key) if d.key is not None else None
                if got is None or len(d.logq) != len(d.tokens):
                    st.dropped_unlogged += 1
                    walk_empty(g, pk, d)  # unlogged: its tokens cannot be verified; walked as an empty draft
                    continue
                blocks, vers = got
                hid = HiddenRefs.from_blocks(blocks)
                from thundersync_rl.engine_q import hidden_num_rows

                m = min(len(d.tokens), hidden_num_rows(hid), len(vers))
                if m != hidden_num_rows(hid):
                    # Tags shorter than the logged hidden rows (a version tag missing): the block handles cannot be
                    # cut, so the draft is not verified (its slot decodes fresh). Tag presence is engine bookkeeping,
                    # never a function of the draft's tokens.
                    st.dropped_unlogged += 1
                    from thundersync_rl.engine_q import free_hidden

                    free_hidden([hid])
                    walk_empty(g, pk, d)  # unverifiable: walked as an empty draft
                    continue
                toks, lq, finished = d.tokens[:m], d.logq[:m], d.finished and m == len(d.tokens)
                segs = version_segments(vers[:m])
                if len(segs) > 1:  # maximal runs: >1 segment = >1 version (true tags, before a negative rewrites them)
                    mixed_drafts.add(id(d))
                if self.neg == "wrong_tag":
                    # NEGATIVE CONTROL (E1): every token tagged with the group's dispatch version.
                    segs = shift_segments(segs, "wrong_tag", disp_version[g])
                    st.neg_tokens += sum(1 for v in vers[:m] if v != disp_version[g])
                if id(d) in mixed_drafts:  # true tags: a negative that rewrites them does not hide the coverage
                    st.mixed_version_rows += 1
                vkey = f"{version}:{g}:{d.index}"
                draft_by_key[vkey] = d
                vkey_of[id(d)] = vkey
                pools[pk].append(
                    (torch.tensor(d.prompt, dtype=torch.long), -1, toks, finished, vkey, segs, lq, hid)
                )
        spec = self.spec
        if self.neg == "spare_by_logp":
            # NEGATIVE CONTROL input: each pool's mean FIRST-token draft log-prob (drafts are tuples (..., lq, hid));
            # aimed at the law check's first-token test (a whole-row mean is too diluted to detect at lr 1e-6).
            spec.__dict__["_neg_pool_score"] = {
                pk: sum(float(d[6][0]) for d in rows if len(d[6])) / max(len(rows), 1) for pk, rows in pools.items()
            }
            st.neg_tokens += 1
        for k in [k for k in spec.stats if k.startswith("vprof/")]:
            del spec.stats[k]  # this step's verify timing only
        spec.__dict__["_vp_calls"] = []
        t_pools = time.perf_counter() - t0 - st.times["handout"]
        neg_m0 = spec.stats.get("nl/neg_content_m_rows", 0.0)
        spec.current_step = int(version)
        impl = self.sc._rollout_manager._impl
        tasks: dict[str, Any] = {}

        metas: dict[str, Any] = {}
        has_valid_of: dict[str, bool] = {}  # filled where each group is finalized (trained-once check, R35 1(a))
        recs_out: dict[str, Any] = {}
        self.trained_in_stage = False
        self.stage_has_valid_tokens = False
        self._groups_trained = 0
        count_cont_as_neg = self.neg == "cf_engine_m"
        # cf_engine_m: every engine-sampled token of the step is off the learner's routing; the target-sampled ones
        # (fresh / continuation tokens, counted below) are the ones a check can see.
        from nemo_rl.experience.rollout_manager import PromptGroupRecord

        cont_max: dict[str, int] = {}

        # EDV (edv.py, DESIGN §22.9): deferred rows by draft key -> (chunk handle, flat row); their η_g and envelope;
        # the rows the data plane holds per group (a retrained group is rewritten); the loss's decisions.
        edv_on = self.edv_e is not None
        edv_e = edv_e_of_step(self.edv_e) if edv_on else None
        edv_rows: dict[str, tuple[dict, int]] = {}  # deferred and not (yet) verified
        edv_eta: dict[str, float] = {}
        edv_replay: dict[str, dict] = {}  # CHECKS: the deferred rows' own verification (compared, never used)
        edv_committed: dict[str, bool] = {}
        edv_loss_bg: dict[str, tuple[float, float]] = {}
        edv_stats = {"deferred": 0, "surprises": 0, "fallback": 0, "replay_rows": 0, "replay_mismatch": 0}
        written_ids: dict[str, list] = {}

        def edv_m_of(k: str) -> float:
            h, i = edv_rows[k]
            return float(h["pools"][h["groups"][i]].out.get(i, {}).get("null_m") or 1.0)

        def edv_slots(g: str) -> dict[int, str]:
            """Slot -> draft key of g's deferred whole rows."""
            out = {}
            for k, o in enumerate(specs[g]):
                key = vkey_of.get(id(o.draft)) if o.draft is not None else None
                if key in edv_rows and o.kind == "whole":
                    out[k] = key
            return out

        def edv_fields_of(g: str) -> dict:
            sl = edv_slots(g)
            return edv.row_fields(
                len(specs[g]), {k: (edv_eta[key], edv_m_of(key), len(draft_by_key[key].tokens)) for k, key in sl.items()}
            )

        async def finish_group(g: str, res) -> tuple[Any, Any]:
            """Rewrite group g's G rows (verified tag) into the buffer; returns (meta, record)."""
            sample, rec = recs[g]
            comps = [c for c, _ in res]
            for c, o in zip(comps, specs[g]):
                a = nolanes._assistant(c.message_log)
                a.pop(nolanes.DRAFT_REF, None)
                st.continuation_tokens += len(a["token_ids"]) - len(o.prefix)
                cont_max[g] = max(cont_max.get(g, 0), len(a["token_ids"]) - len(o.prefix))
                if count_cont_as_neg:
                    st.neg_tokens += len(a["token_ids"]) - len(o.prefix)
            new_rec = PromptGroupRecord(
                prompt_idx=rec.prompt_idx,
                prompt=rec.prompt,
                extra_env_info=rec.extra_env_info,
                metadata=rec.metadata,
                completions=comps,
                rollout_metrics=rec.rollout_metrics,
                loss_multiplier=rec.loss_multiplier,
            )
            m = await self.sc._buffer.rewrite_group_rows(
                group_id=g, record=new_rec, weight_version=disp_version[g],
                clear_sample_ids=written_ids.get(g, old_ids[g]),
                extra_fields=edv_fields_of(g) if edv_on else None,
                extra_tags={VERIFIED_TAG: int(version)},
                prev_logprobs_from_generation=self.prev_from_verify,
            )
            written_ids[g] = list(m.sample_ids)
            return m, new_rec

        pipe = None
        # GRPO-Z train-on-arrival: a group's rows are finalized and trained in UNITS as they become final: one unit
        # "<g>#c" of the rows complete at verification (whole drafts, scored only) and one unit "<g>#<slot>" per row
        # that decodes on the engine. Each unit is one capture of the exact accumulator; its composition is a function
        # of the verification outcome only, so theta_{k+1} does not depend on the tails' timing.
        gz_on = self.grpo_z != "off"
        gz_units: dict[str, tuple[str, list[int]]] = {}
        gz_metas: dict[str, Any] = {}
        gz_valid: dict[str, bool] = {}
        gz_comps: dict[str, list] = {}
        gz_draft_r: dict[str, list[float]] = {}
        gz_slot_sid: dict[tuple[str, int], Any] = {}  # (group, slot) -> the sample id its row was written under
        gz_relaunches = [0]
        if gz_on:
            if not self.cfg.train_overlap or (edv_on and self.n != self.g):
                raise RuntimeError("[grpo_z] needs train_overlap; with EDV, k_drafts 1 (validator)")
            for g in gids:
                gz_draft_r[g] = [float(c.reward) for c in recs[g][1].completions]
                if len(gz_draft_r[g]) != self.n:
                    raise RuntimeError(f"[grpo_z] group {g}: {len(gz_draft_r[g])} draft rewards for N={self.n} drafts")
            # The drafts' rows leave the data plane up front; each unit then writes only its own slots.
            await self.sc._call_dp("clear_samples", sample_ids=[sid for g in gids for sid in old_ids[g]],
                                   partition_id=meta.partition_id)
            for g in gids:
                written_ids[g] = []

        async def gz_finalize(u: str, res) -> tuple[Any, bool]:
            """Unit u's rows: rewrite into the data plane at their slots, GRPO-Z row advantages, advantage stage."""
            from nemo_rl.experience.rollout_manager import PromptGroupRecord as _PGR
            from thundersync_rl import grpo_z as _gz

            g, sl = gz_units[u]
            sample, rec = recs[g]
            comps = [c for c, _ in res]
            for c, k in zip(comps, sl):
                o = specs[g][k]
                a = nolanes._assistant(c.message_log)
                a.pop(nolanes.DRAFT_REF, None)
                st.continuation_tokens += len(a["token_ids"]) - len(o.prefix)
                cont_max[g] = max(cont_max.get(g, 0), len(a["token_ids"]) - len(o.prefix))
            part = _PGR(prompt_idx=rec.prompt_idx, prompt=rec.prompt, extra_env_info=rec.extra_env_info,
                        metadata=rec.metadata, completions=comps, rollout_metrics=rec.rollout_metrics,
                        loss_multiplier=rec.loss_multiplier)
            # A slot written before (its unit's capture was dropped by an EDV surprise) is rewritten in place.
            stale = [gz_slot_sid.pop((g, k)) for k in sl if (g, k) in gz_slot_sid]
            extra = None
            if edv_on:
                full = edv_fields_of(g)
                idx = torch.tensor(list(sl), dtype=torch.long)
                extra = {name: col.index_select(0, idx) for name, col in full.items()}
            m = await self.sc._buffer.rewrite_group_rows(
                group_id=g, record=part, weight_version=disp_version[g], clear_sample_ids=stale,
                extra_tags={VERIFIED_TAG: int(version)}, prev_logprobs_from_generation=self.prev_from_verify,
                generation_indices=list(sl), extra_fields=extra,
            )
            written_ids[g] = [x for x in written_ids[g] if x not in set(stale)] + list(m.sample_ids)
            for k, sid in zip(sl, m.sample_ids):
                gz_slot_sid[(g, k)] = sid
            # Row k's source draft (its index in g's drafts), None when it came from another group's draft of the same
            # prompt (a shared pool) or is fresh: its baseline is then every draft of g (all independent of Y_k).
            srcs = [specs[g][k].draft.index if specs[g][k].draft is not None and specs[g][k].draft.group_id == g
                    else None for k in sl]
            adv = _gz.group_row_advantages([float(c.reward) for c in comps], srcs, gz_draft_r[g], self.grpo_z,
                                           step_scale=self.gz_scale if self.grpo_z == "loo_c" else None)
            m, has_valid = await self.sc._advantage_stage(m, row_advantages=torch.tensor(adv, dtype=torch.float32))
            gz_metas[u], gz_valid[u], gz_comps[u] = m, bool(has_valid), comps
            return m, bool(has_valid)

        if self.cfg.train_overlap:
            # Finalize overlap (DESIGN §13; part of train_overlap): each group is rewritten and gets its group-local advantages the moment its
            # rows are final (here: while the learner verifies later chunks / trains other groups); the learner trains
            # from the queue. Train-call contents are unchanged; order is free under the exact accumulator (R31).
            from thundersync_rl.finalize_pipeline import FinalizePipeline

            async def _finalize(g: str, res) -> tuple[Any, bool]:
                m, recs_out[g] = await finish_group(g, res)
                m, has_valid = await self.sc._advantage_stage(m)
                metas[g] = m
                has_valid_of[g] = bool(has_valid)
                return m, bool(has_valid)

            pipe = FinalizePipeline(gz_finalize if gz_on else _finalize)

        def count_specs(outs: list, sign: int) -> None:
            """The step's output-row statistics of one group's outputs (sign -1: an EDV re-walk replaces them)."""
            for o in outs:
                st.rows += sign
                st.whole += sign * (o.kind == "whole")
                st.continued += sign * (o.kind == "continued")
                st.fresh += sign * (o.kind == "fresh")
                if o.kind == "continued" and o.prefix:
                    st.reject_hist[nolanes.reject_bin(len(o.prefix) - 1)] += sign
                if o.kind == "continued" and o.draft is not None and id(o.draft) in mixed_drafts:
                    st.mixed_continued += sign
                if o.draft is not None:
                    st.accepted_tokens += sign * (len(o.prefix) - (0 if o.kind == "whole" else 1))

        n_gen_calls = [0]  # rows launched that call the engine (not complete): continuation / fresh calls

        def launch(g: str, relaunch: bool = False, gz_slots: Optional[list[int]] = None) -> None:
            """Group g's continuations / fresh slots / scoring: new engine requests at theta_k, started now.
            ``relaunch`` (EDV surprise): the group's outputs were re-walked; its rows replace the submitted ones."""
            sample, rec = recs[g]
            plen = len(drafts[g][0].prompt)
            rows = []
            count_specs(specs[g] if gz_slots is None else [specs[g][k] for k in gz_slots], +1)
            for k_, o in enumerate(specs[g]):
                budget = continuation_budget(plen, len(o.prefix), self.max_new, self.max_seq)
                complete = o.complete or budget == 0
                n_gen_calls[0] += 0 if (complete or (gz_slots is not None and k_ not in gz_slots)) else 1
                gd = {"nl_no_log": 1, "max_new_tokens": int(budget)}
                if self.neg == "key_reuse" and o.kind == "continued" and o.draft is not None and o.draft.rid is not None:
                    gd["nl_key_rid"] = int(o.draft.rid)  # NEGATIVE CONTROL (E4): the rejected draft's noise key
                    st.neg_tokens += 1
                else:
                    gd["nl_key_rid"] = -1
                if (self.cfg.tail_mtd or getattr(self.cfg, "tail_tree", 0)) and not complete:
                    # Multi-token tail decode: lookup references (proposals only; any content is exact, DESIGN E6
                    # concerns targets): the rejected draft from tau on, then the group's other drafts.
                    from nemo_rl.models.generation.megatron.tail_mtd import pack_refs

                    refs = []
                    if o.kind == "continued" and o.draft is not None and o.prefix:
                        refs.append(o.draft.tokens[len(o.prefix) - 1:])
                    refs += [d.tokens for d in drafts[g] if d is not o.draft]
                    gd["nl_mtd_refs"] = pack_refs(refs)
                rows.append(
                    {
                        "prefix": list(o.prefix),
                        "prefix_logprobs": list(o.prefix_logprobs),
                        "complete": complete,
                        # The Megatron engine path reports no truncation flag (stock: False for every row);
                        # rows finished here keep the same function of their tokens.
                        "truncated": False,
                        "generation_data": gd,
                    }
                )
            if gz_on:
                # GRPO-Z units (see gz_finalize): the rows complete now, then each engine row on its own. An EDV
                # re-walk (gz_slots: the dropped unit's slots) relaunches only those slots, as new units.
                ks = list(range(len(rows))) if gz_slots is None else sorted(gz_slots)
                cname = f"{g}#c" if gz_slots is None else f"{g}#c{gz_relaunches[0]}"
                done_now = [k for k in ks if rows[k]["complete"]]
                parts = ([(cname, done_now)] if done_now else []) + [
                    (f"{g}#{k}", [k]) for k in ks if not rows[k]["complete"]]
                unit_tasks = []
                for u, sl in parts:
                    if u in gz_units:
                        raise RuntimeError(f"[grpo_z] unit {u} launched twice")
                    gz_units[u] = (g, sl)
                    tu = asyncio.ensure_future(impl.run_rows(sample, [rows[k] for k in sl]))
                    unit_tasks.append(tu)
                    if gz_slots is None:
                        pipe.submit(u, tu)
                    else:
                        pipe.submit_extra(u, tu)
                if gz_slots is None:
                    tasks[g] = asyncio.ensure_future(asyncio.gather(*unit_tasks))
                else:
                    tasks[g] = asyncio.ensure_future(asyncio.gather(tasks[g], *unit_tasks))
            else:
                tasks[g] = asyncio.ensure_future(impl.run_rows(sample, rows))
            t_launch[g] = time.perf_counter()
            tasks[g].add_done_callback(lambda _f, g=g: t_done.__setitem__(g, time.perf_counter()))
            if pipe is not None and not gz_on:
                if relaunch:
                    pipe.resubmit(g, tasks[g])
                else:
                    pipe.submit(g, tasks[g])
            early = [
                len(o.prefix) - 1
                for o in specs[g]
                if o.kind == "continued" and o.prefix and len(o.prefix) - 1 < nolanes.EARLY_TAIL_TOKENS
            ]
            if early and "first_early_launch" not in st.crit:
                st.crit["first_early_launch"] = round(t_launch[g] - t0, 2)
                st.crit["first_early_chunk"] = chunk_of[g]

        # Streaming continuation start: contiguous runs of the fixed pool order are verified one call each; a run's
        # rows start decoding on the engine as soon as its call returns, while the learner verifies the next run.
        # Row seeds (F3) and the BI forward make every decision and p independent of the call it rides in.
        pks = list(pool_groups)
        chunks = nolanes.stream_chunks(pks, [len(pools.get(pk, ())) for pk in pks], self.cfg.verify_stream_chunks)
        chunk_of = {g: ci for ci, run in enumerate(chunks) for pk in run for g in pool_groups[pk]}
        t_launch: dict[str, float] = {}
        t_done: dict[str, float] = {}
        specs: dict[str, list[OutputRow]] = {}
        call_stats: list[dict] = []
        t1 = time.perf_counter()
        st.times["pre_verify"] = t1 - t0 - st.times["handout"]  # driver: pools, hidden refs, segments (learner idle)
        t2 = None
        st.times["verify"] = 0.0
        chunk_s: list[float] = []
        t_launch_total = 0.0

        dropped_pks: set = set()
        spare_sel: Optional[set] = None  # set after the verification (spec.null_selected_keys)

        def publish(run_pks, has_pools: bool, plans_of: Optional[dict] = None, relaunch: bool = False) -> None:
            """Outputs per group of a verified run of pools, and their continuations started on the engine."""
            nonlocal t2, t_launch_total
            tl = time.perf_counter()
            # Outputs per group: a pool's plans (drafting order) fill its groups' slots in group order (duplicate
            # prompts share one pool; a draft is only a proposal, whichever group's slot it fills).
            for pk in run_pks:
                groups = pool_groups[pk]
                if spare_sel is not None and pk not in spare_sel:
                    # Spare prompts (XPR-S): not among the first P undropped pools (⊥ counts + fixed order only):
                    # discarded unseen by the learner; no continuation is decoded for it.
                    dropped_pks.add(pk)
                    continue
                if not has_pools:
                    plans = []  # no drafts: every slot decodes fresh
                elif plans_of is not None:
                    plans = list(plans_of[pk])  # EDV re-walk
                else:
                    plans = list(spec.plans.get(pk, []))
                for g in groups:
                    take, plans = plans[: self.g], plans[self.g :]
                    specs[g] = outputs_of_group([draft_by_key[p.engine_key] for p in take], take, self.g)
                if plans:
                    raise RuntimeError(f"[nolanes] pool of {groups}: {len(plans)} outputs beyond its slots")
                # E7: generation log-probs of verified tokens are p (negative control: keep the engine's q).
                if self.neg == "engine_q_genlp":
                    st.neg_tokens += sum(nolanes.keep_engine_q(specs[g]) for g in groups)
                if self.neg == "carry_unverified":
                    st.neg_tokens += sum(nolanes.accept_unverified(specs[g]) for g in groups)
                if t2 is None:
                    t2 = time.perf_counter()
                for g in groups:
                    launch(g, relaunch=relaunch)
            t_launch_total += time.perf_counter() - tl

        async def verify_phase() -> None:
            """theta_k's verification of every pool (learner chunks, or the engine: DX) and the continuations' start."""
            nonlocal gids, spare_sel
            # Every chunk's call is posted up front; the learner runs them in posting order back to back (no driver
            # round trip between chunks), and the driver collects and publishes them in the same order as they finish.
            posted = []
            tv = time.perf_counter()
            if not edv_on:
                for run_pks in chunks:
                    run_pools = [pools[pk] for pk in run_pks if pk in pools]
                    h = None
                    if run_pools:
                        spec._null_slots[int(version)] = {pk: slots[pk] for pk in run_pks}
                        h = await asyncio.to_thread(spec.verify_post, run_pools)
                    posted.append((run_pks, run_pools, h))
            else:
                # EDV: prepare every chunk, draw every row's η_g in one learner call, defer the finished rows with
                # η_g <= e of single-group pools (predicted whole: decided by their group's training forward), and post
                # each chunk's other rows (drafting order, as without EDV). CHECKS: the deferred rows are also posted, last.
                for run_pks in chunks:
                    run_pools = [pools[pk] for pk in run_pks if pk in pools]
                    h = None
                    if run_pools:
                        spec._null_slots[int(version)] = {pk: slots[pk] for pk in run_pks}
                        h = await asyncio.to_thread(spec.verify_prepare, run_pools)
                    posted.append((run_pks, run_pools, h))
                idx_all = [(h, i) for _, _, h in posted if h is not None for i in h["batch"]]
                if idx_all:
                    seeds, lens = [], []
                    for _, _, h in posted:
                        if h is not None:
                            sd, ln = spec.edv_seeds(h, h["batch"])
                            seeds += sd
                            lens += ln
                    etas_r = await asyncio.to_thread(
                        ray.get, self.sc._trainer.worker_group.run_all_workers_single_data(
                            "nl_row_etas", seeds=seeds, lengths=lens)
                    )
                    etas = next(e for e in etas_r if e is not None)
                    for (h, i), eta in zip(idx_all, etas):
                        d = h["flat"][i][1]
                        pk_row = tuple(int(t) for t in d[0].tolist())
                        if len(pool_groups.get(pk_row, ())) == 1 and edv.eligible(d[3], eta, edv_e):
                            h["edv_done"][i] = edv.deferred_result(d, tuple(h["pools"][h["groups"][i]].env[i]))
                            edv_rows[d[4]] = (h, i)
                            edv_eta[d[4]] = float(eta)
                    edv_stats["deferred"] = len(edv_rows)
                for _, _, h in posted:
                    if h is not None:
                        h["sub"] = [i for i in h["batch"] if i not in h["edv_done"]]
                        h["futs"] = await asyncio.to_thread(spec.verify_submit, h, h["sub"])
                replay_posted = []
                if nolanes.checks_on():
                    for _, _, h in posted:
                        if h is not None and h["edv_done"]:
                            rk = sorted(h["edv_done"])
                            replay_posted.append((h, rk, await asyncio.to_thread(spec.verify_submit, h, rk)))
            st.times["verify"] += time.perf_counter() - tv
            for run_pks, run_pools, h in posted:
                if h is not None:
                    tv = time.perf_counter()
                    await asyncio.to_thread(spec.verify_collect, h)
                    st.times["verify"] += time.perf_counter() - tv
                    if int(self.cfg.spare_prompts):
                        # XPR-S: the verifier's prompt selection (first P undropped pools, fixed order; one verify call per
                        # step, the validator requires it). Unselected pools are discarded at publish.
                        if spec.null_selected_keys is None:
                            raise RuntimeError("[nolanes] spare prompts: the verifier selected no prompts (prompt drop off?)")
                        spare_sel = {tuple(k) for k in spec.null_selected_keys}
                    chunk_s.append(round(time.perf_counter() - tv, 2))
                    call_stats.append({k: v for k, v in spec.stats.items() if k.startswith("spec/null_")})
                if not self.colocated:
                    publish(run_pks, bool(run_pools))
            if self.colocated:
                # Wake on the same theta_k (no version announced: the tag stays k; checked on every rank), then start the
                # continuations; they co-batch with the resumed lookahead drafting.
                tc = time.perf_counter()
                try:
                    if self.slots is not None:
                        # the step's continuations take their slots before the next batch's drafts (released in finally:
                        # also when the wake raises and the loop retries)
                        self.slots.hold_drafts()
                    if self.defers_post_step_wake():
                        # This wake is the step's refit (the loop deferred its post-step wake): the rollouts it permits
                        # are stamped with theta_k's version, set before the dispatch gate opens.
                        self.sc._rollout_manager.set_weight_version(int(self.sc._trainer_version))
                    await self.sc._sync_weights()
                    await asyncio.to_thread(self.check_engine_version, int(version))
                    st.times["colo_wake"] = time.perf_counter() - tc
                    arrivals0 = self.slots.cont_arrivals if self.slots is not None else 0
                    calls0 = n_gen_calls[0]
                    for run_pks, run_pools, _h in posted:
                        publish(run_pks, bool(run_pools))
                    if self.slots is not None:
                        # every launched engine call is queued at the gate before the drafts are released
                        await self.slots.wait_cont_arrivals(arrivals0 + n_gen_calls[0] - calls0)
                finally:
                    if self.slots is not None:
                        self.slots.open_drafts()
            if edv_on:
                for h, rk, futs in replay_posted:
                    got = await asyncio.to_thread(spec.verify_results, h, rk, futs)
                    edv_replay.update({h["flat"][i][1][4]: r for i, r in got.items()})
            st.times["verify_wall"] = time.perf_counter() - t1
            vprof = {k[6:]: round(v, 3) for k, v in spec.stats.items() if k.startswith("vprof/")}
            vprof.update(nolanes.call_idle(spec.__dict__.get("_vp_calls", []), time.time() - (time.perf_counter() - t1)))
            vprof.update({"chunks_s": chunk_s, "launch_s": round(t_launch_total, 3), "pools_s": round(t_pools, 3)})
            st.vprof = vprof
            st.neg_tokens += int(spec.stats.get("nl/neg_content_m_rows", 0.0) - neg_m0)  # rows the E3 negative moved
            n_admitted = len(gids)
            dropped_gids = [g for pk in dropped_pks for g in pool_groups[pk]]
            if dropped_pks:
                st.crit["spare_dropped_groups"] = len(dropped_gids)
                gids = [g for g in gids if g not in set(dropped_gids)]
                # Release the discarded groups' rows from the data plane (the loop only clears the groups it trains).
                ids = [sid for g in dropped_gids for sid in old_ids[g]]
                await self.sc._call_dp("clear_samples", sample_ids=ids, partition_id=meta.partition_id)
                for g in dropped_gids:
                    self.sc._finalizer_metrics_by_group.pop(g, None)
                    self.sc._finalizer_rewards_by_group.pop(g, None)
            if int(self.cfg.spare_prompts) and len(gids) != n_admitted - int(self.cfg.spare_prompts):
                raise RuntimeError(f"[nolanes] spare prompts: {len(gids)} of {n_admitted} groups kept, expected "
                                   f"{n_admitted - int(self.cfg.spare_prompts)}")
            missing = [g for g in gids if g not in tasks]
            if missing:
                raise RuntimeError(f"[nolanes] groups {missing[:5]} got no outputs (pool / chunk bookkeeping)")
            st.draft_tokens = sum(len(d.tokens) for d in flat_d)

        engine_mode = bool(getattr(self, "engine_verify", False))
        if not engine_mode:
            await verify_phase()
        tw = time.perf_counter()
        if self.cfg.train_overlap:
            # R31 Option A (exact variant): each group trains the moment its G rows are final, in ARRIVAL order,
            # one train call per group (micro-batches hold only that group's rows, in slot order), into the
            # learner's order-invariant int64-limb gradient accumulator (NEMO_RL_EXACT_GRAD_ACCUM); one optimizer
            # step closes after the last group (finish_train_step, SC loop). theta_{k+1}'s bits are a function of
            # the outputs only, whatever the tails' timing.
            trainer = self.sc._trainer
            tb = time.perf_counter()
            await asyncio.to_thread(trainer.prepare_for_training)
            st.times["prep_train"] = time.perf_counter() - tb
            tb = time.perf_counter()
            await asyncio.to_thread(trainer.begin_train_step, self.sc._loss_fn, **self.sc._begin_train_step_kwargs())
            st.times["begin_step"] = time.perf_counter() - tb
            self.trained_in_stage = True
            train_fields = tuple(self.sc._train_fields) + (edv.FIELDS if edv_on else ())
            if edv_on:
                await asyncio.to_thread(
                    ray.get, trainer.worker_group.run_all_workers_single_data("nl_edv_arm", on=True)
                )

            async def edv_verify(keys: list[str]) -> dict[str, dict]:
                """The regular verification of deferred rows (CHECKS: their replay, verified already)."""
                out = {k: edv_replay[k] for k in keys if k in edv_replay}
                by_h: dict[int, tuple[dict, list[int]]] = {}
                for k in keys:
                    if k not in out:
                        h, i = edv_rows[k]
                        by_h.setdefault(id(h), (h, []))[1].append(i)
                for h, idx in by_h.values():
                    futs = await asyncio.to_thread(spec.verify_submit, h, idx)
                    got = await asyncio.to_thread(spec.verify_results, h, idx, futs)
                    out.update({h["flat"][i][1][4]: r for i, r in got.items()})
                edv_replay.update(out)
                return out

            def unit_group(u: str) -> str:
                return gz_units[u][0] if gz_on else u

            def edv_slots_of(u: str) -> dict[int, str]:
                """Slot -> draft key of the deferred whole rows of group / unit u (a unit: only its slots)."""
                sl = edv_slots(unit_group(u))
                if gz_on:
                    mine = set(gz_units[u][1])
                    sl = {k: key for k, key in sl.items() if k in mine}
                return sl

            async def gz_edv_rewalk(u: str, keys: list[str]) -> None:
                """GRPO-Z + EDV surprise (k_drafts 1: N = G drafts, envelope 1, no ⊥): verify the deferred rows, walk
                the group's pool again, and relaunch the dropped unit's slots as new units. Only the surprised slots
                can change (validated): every other unit of the group stands."""
                g, sl = gz_units[u]
                got = await edv_verify(keys)
                h, _ = edv_rows[keys[0]]
                name = h["groups"][edv_rows[keys[0]][1]]
                once = ("null_grid_pit", "edv_shadow")
                for i in [i for i, gg in enumerate(h["groups"]) if gg == name]:
                    k = h["flat"][i][1][4]
                    h["res"][i] = got[k] if k in keys else {a: v for a, v in h["res"][i].items() if a not in once}
                plans = await asyncio.to_thread(spec.edv_rewalk, h, name)
                for k in keys:
                    edv_rows.pop(k)
                pk = next(pk for pk in pool_groups if g in pool_groups[pk])
                if pool_groups[pk] != [g] or len(plans) != self.g:
                    raise RuntimeError(f"[grpo_z] EDV re-walk of {g}: {len(plans)} plans for a pool of {pool_groups[pk]}")
                new = outputs_of_group([draft_by_key[p.engine_key] for p in plans], plans, self.g)
                changed = [k for k, (a, b) in enumerate(zip(specs[g], new))
                           if (a.kind, list(a.prefix), id(a.draft)) != (b.kind, list(b.prefix), id(b.draft))]
                if not set(changed) <= set(sl):
                    raise RuntimeError(f"[grpo_z] EDV re-walk of {g} changed slots {changed} outside the dropped unit "
                                       f"{u} {sl} (needs k_drafts 1: no ⊥)")
                count_specs([specs[g][k] for k in sl], -1)
                specs[g] = new
                gz_metas.pop(u, None)
                gz_units.pop(u)
                gz_valid.pop(u, None)
                gz_comps.pop(u, None)
                gz_relaunches[0] += 1
                launch(g, relaunch=True, gz_slots=list(sl))

            async def edv_rewalk(g: str, keys: list[str]) -> None:
                """Deferred rows of g were not accepted whole: verify them, walk the pool again with the true outcomes,
                and launch what the walk gives; the group is finalized and trained again when its rows are final."""
                got = await edv_verify(keys)
                h, _ = edv_rows[keys[0]]
                name = h["groups"][edv_rows[keys[0]][1]]
                once = ("null_grid_pit", "edv_shadow")
                for i in [i for i, gg in enumerate(h["groups"]) if gg == name]:
                    k = h["flat"][i][1][4]
                    h["res"][i] = got[k] if k in keys else {a: v for a, v in h["res"][i].items() if a not in once}
                plans = await asyncio.to_thread(spec.edv_rewalk, h, name)
                for k in keys:
                    edv_rows.pop(k)
                diag = spec.__dict__.get("_nl_reject_diag")
                if diag:  # the re-walk re-plans the pool's rejected rows: one diagnostic record per draft
                    diag[:] = list({r["key"]: r for r in diag}.values())
                pk = next(pk for pk in pool_groups if g in pool_groups[pk])
                for g2 in pool_groups[pk]:
                    count_specs(specs[g2], -1)  # replaced by the re-walk's outputs (counted by launch)
                publish([pk], True, {pk: plans}, relaunch=True)

            async def edv_after_call(groups_called: list[str]) -> set[str]:
                """Read each group's agreed surprise mask and the loss's decisions (one entry per group of the call, in
                call order); start the surprise path; return the groups whose capture was dropped."""
                logs = await asyncio.to_thread(
                    ray.get, trainer.worker_group.run_all_workers_single_data("nl_edv_take")
                )
                for lg in logs:
                    if len(lg) != len(groups_called):
                        raise RuntimeError(f"[edv] {len(lg)} capture entries for a call of {len(groups_called)} groups")
                dropped: set[str] = set()
                for j, g in enumerate(groups_called):
                    masks = {lg[j]["mask"] for lg in logs}
                    if len(masks) != 1:
                        raise RuntimeError(f"[edv] group {g}: learner ranks disagree on the surprise mask {masks}")
                    mask = masks.pop()
                    sl = edv_slots_of(g)
                    decided = {}
                    for lg in logs:
                        for slot, bg, eta, ok in lg[j]["decisions"]:
                            decided[int(slot)] = (float(bg), float(eta), bool(ok))
                    if not sl:
                        if mask:
                            raise RuntimeError(f"[edv] group {g}: surprise mask {mask:#x} without deferred rows")
                        continue
                    missing = [key for k, key in sl.items() if k not in decided]
                    if missing and mask == 0:
                        edv_stats["fallback"] += len(missing)
                        got = await edv_verify(missing)
                        bad = [k for k in missing if got[k]["null_grid"][edv_m_of(k)][0] != len(draft_by_key[k].tokens)]
                        if bad:
                            raise RuntimeError(f"[edv] group {g}: fallback-verified rows {bad} are not whole, but the "
                                               "group's capture is already committed")
                    for k, key in sl.items():
                        if k in decided:
                            edv_loss_bg[key] = decided[k][:2]
                            edv_committed[key] = not (mask >> k) & 1
                    if mask:
                        keys = [sl[k] for k in range(64) if (mask >> k) & 1 and k in sl]
                        if len(keys) != bin(mask).count("1"):
                            raise RuntimeError(f"[edv] group {g}: surprise mask {mask:#x} names slots without a deferred row")
                        edv_stats["surprises"] += len(keys)
                        dropped.add(g)
                        if gz_on:
                            await gz_edv_rewalk(g, keys)
                        else:
                            await edv_rewalk(g, keys)
                return dropped
            st.times["train_in_stage"] = 0.0
            arrival: list[str] = []
            trained_order: list[str] = []
            trained_ids: list[Any] = []  # sample ids handed to the train calls (trained-once check, P5)
            neg_late = self.neg == "drop_late_group"

            def edv_risk(g: str) -> float:
                """The largest η_g among g's deferred rows (its surprise risk is increasing in η_g); 0 without one."""
                return max((edv_eta[k] for k in edv_slots_of(g).values()), default=0.0)

            # GRPO-Z: finalize units are re-chunked into TRAIN units of exactly D = learner DP x MBS rows (any groups,
            # arrival order) so every DP rank (and every EP peer) trains one row per capture; the step's P x G rows
            # are a multiple of D, so nothing is left over. Train-unit composition follows arrival (theta_{k+1}'s
            # bits may follow tail timing; gradients need accuracy only, user rule 10-19); the rows' law is unchanged.
            gz_dp = int(self.sc._trainer.sharding_annotations.get_axis_size("data_parallel")) if gz_on else 1
            gz_carry: list[tuple[str, Any]] = []
            gz_chunk_metas: dict[str, Any] = {}

            async def train_ready(batch: list[tuple[str, tuple[Any, bool]]]) -> None:
                """EDV: the riskiest groups (deferred rows with η_g near e) train first, in a call of their own, so a
                surprise is found at the start of training and its continuation launches early. Order only: the
                accumulator is order-invariant (R31)."""
                if gz_on:
                    for u, (m, has_valid) in batch:
                        arrival.append(u)
                        self.stage_has_valid_tokens |= bool(has_valid)
                        gz_carry.extend((u, m.subset([i])) for i in range(len(m.sample_ids)))
                    chunks: list[tuple[str, tuple[Any, bool]]] = []
                    while len(gz_carry) >= gz_dp:
                        part, gz_carry[:] = gz_carry[:gz_dp], gz_carry[gz_dp:]
                        tm = part[0][1] if gz_dp == 1 else part[0][1].concat(*[x[1] for x in part[1:]])
                        name = f"t{len(gz_chunk_metas)}"
                        gz_chunk_metas[name] = tm
                        chunks.append((name, (tm, True)))
                    if chunks:
                        await train_call(chunks, record_arrival=False)
                    return
                if not edv_on or len(batch) < 2:
                    return await train_call(batch)
                ranked = sorted(batch, key=lambda x: -edv_risk(x[0]))
                first = [x for x in ranked if edv_risk(x[0]) > edv_e - 0.15][:8]
                rest = [x for x in ranked if x not in first]
                if first:
                    await train_call(first)
                if rest:
                    await train_call(rest)

            async def train_call(batch: list[tuple[str, tuple[Any, bool]]], record_arrival: bool = True) -> None:
                # Every group finalized when the learner became free trains in ONE learner call, one group after the
                # other with one accumulator capture per group (PERF_RECONCILE_1102 lever 1: no driver round trip
                # between groups). What each group trains is unchanged.
                call: list[Any] = []
                called: list[str] = []
                for g, (m, has_valid) in batch:
                    if record_arrival:
                        arrival.append(g)
                        self.stage_has_valid_tokens |= bool(has_valid)
                    if not has_valid:
                        continue
                    n_units = (sum(len(sl) for _, sl in gz_units.values()) // gz_dp) if gz_on else len(gids)
                    if neg_late and len(trained_order) == n_units - 1 and trained_order and trained_order[0] != g:
                        # NEGATIVE CONTROL (P5): the step does not wait for its last-finishing group; the first group
                        # trains a second time in its place (same group and capture counts).
                        st.neg_tokens += len(m.sample_ids)
                        g, m = trained_order[0], (gz_chunk_metas if gz_on else metas)[trained_order[0]]
                    self._groups_trained += 1
                    trained_order.append(g)
                    trained_ids.extend(m.sample_ids)
                    call.append(m)
                    called.append(g)
                if not call:
                    return
                tt = time.perf_counter()
                await asyncio.to_thread(trainer.train_groups_from_metas, call, train_fields=train_fields)
                st.times["train_in_stage"] += time.perf_counter() - tt
                st.train_calls += 1
                if edv_on and any(edv_slots_of(g) for g in called):
                    # Dropped captures: those groups did not train (their re-walked rows train when final).
                    for g in await edv_after_call(called):
                        m = call[called.index(g)]
                        self._groups_trained -= 1
                        trained_order.remove(g)
                        for sid in m.sample_ids:
                            trained_ids.remove(sid)
                elif edv_on:
                    await asyncio.to_thread(  # drain the call's (empty) capture entries
                        ray.get, trainer.worker_group.run_all_workers_single_data("nl_edv_take")
                    )

            if engine_mode:
                # DX: the learner trains whatever is final while the engine verifies the step's pools (open drain:
                # the verify phase closes the pipeline once every pool is published).
                vtask = asyncio.ensure_future(verify_phase())

                def _verify_done(f) -> None:
                    if f.cancelled():
                        return
                    err = f.exception()
                    if err is not None:
                        pipe.fail(err)
                    else:
                        pipe.close()

                vtask.add_done_callback(_verify_done)
                try:
                    probes = await pipe.drain(None, train_ready)
                finally:
                    if not vtask.done():
                        vtask.cancel()
                await vtask
            else:
                probes = await pipe.drain(len(gz_units) if gz_on else len(gids), train_ready)
            if gz_on and gz_carry:
                raise RuntimeError(f"[grpo_z] {len(gz_carry)} finalized rows left untrained (not a multiple of DP "
                                   f"{gz_dp}): the step's rows must be a multiple of learner DP x MBS")
            if gz_on:
                # Per-group views for the checks and the step meta: units in slot order.
                from nemo_rl.experience.rollout_manager import PromptGroupRecord as _PGR

                for g in gids:
                    us = sorted((u for u, (gg, _) in gz_units.items() if gg == g), key=lambda u: gz_units[u][1][0])
                    mg = None
                    comps_g: dict[int, Any] = {}
                    for u in us:
                        mg = gz_metas[u] if mg is None else mg.concat(gz_metas[u])
                        for k, c in zip(gz_units[u][1], gz_comps[u]):
                            comps_g[k] = c
                    metas[g] = mg
                    has_valid_of[g] = True  # every finalized row trains (in DP-sized train units), valid tokens or not
                    rec = recs[g][1]
                    recs_out[g] = _PGR(prompt_idx=rec.prompt_idx, prompt=rec.prompt, extra_env_info=rec.extra_env_info,
                                       metadata=rec.metadata, completions=[comps_g[k] for k in sorted(comps_g)],
                                       rollout_metrics=rec.rollout_metrics, loss_multiplier=rec.loss_multiplier)
                    if sorted(comps_g) != list(range(self.g)):
                        raise RuntimeError(f"[grpo_z] group {g}: slots {sorted(comps_g)} finalized, expected 0..{self.g - 1}")
            st.times["learner_wait"] = probes["learner_wait"]
            st.times["finalize_hidden"] = pipe.prepare_s
            st.crit["tail_exposed"] = round(probes["tail_exposed"], 2)
            # Outputs are taken from the finalize side (every step group, fixed order), not from the train calls, so a
            # queue / drain bug that skips or repeats a group cannot add to both sides (R35 1(a)). An all-invalid step
            # (no group with valid tokens) trains nothing and is not an error (R35 1(b); stock skips such a step).
            output_ids = step_output_ids(gids, metas, has_valid_of)
            missing, extra = trained_once_diff(trained_ids, output_ids)
            line = (f"[trained once check] step={int(version)} trained={len(trained_ids)} outputs={len(output_ids)} "
                    f"missing={len(missing)} extra={len(extra)}")
            bad = bool(missing or extra)
            print(line + (f" negative tokens {st.neg_tokens}" if neg_late else "") + (" FAIL" if bad else " ok"),
                  flush=True)
            if bad and not neg_late:
                raise RuntimeError(f"[nolanes] step {int(version)}: the trained rows are not the step's output rows "
                                   f"exactly once (missing {missing[:3]}, extra / duplicated {extra[:3]})")
            if nolanes.checks_on() and trained_order:
                # R32 item 1: re-run the window's FIRST trained group at its end; the learner compares the gradient
                # with that group's first contribution (bitwise / digest) and drops it.
                await asyncio.to_thread(
                    ray.get, trainer.worker_group.run_all_workers_single_data("exact_accum_arm_replay")
                )
                await asyncio.to_thread(
                    trainer.train_groups_from_metas, [(gz_chunk_metas if gz_on else metas)[trained_order[0]]],
                    train_fields=train_fields
                )
            st.times["continue_after_verify"] = time.perf_counter() - tw
            t_post = time.perf_counter()
            st.times["continue"] = time.perf_counter() - (t2 if t2 is not None else tw)
            st.arrival_inversions = sum(
                1 for i in range(len(arrival)) for j in range(i + 1, len(arrival))
                if gids.index(arrival[i].rsplit("#", 1)[0] if gz_on else arrival[i])
                > gids.index(arrival[j].rsplit("#", 1)[0] if gz_on else arrival[j])
            )
        else:
            results = [await tasks[g] for g in gids]  # fixed group order (R2), whatever order they finish in
            st.times["continue_after_verify"] = time.perf_counter() - tw
            if self.colocated:
                # Stand down for training (the loop's own stand-down after this stage is then a no-op sleep).
                tc = time.perf_counter()
                self.sc._rollout_permitted.clear()
                if self.cfg.wake_verify and self.slots is not None:
                    # Drafts waiting for a slot end here with zero tokens, unsent (and so does any draft call reaching
                    # the gate before the next wake): only drafts holding a slot are on the engines to be cut.
                    st.crit["cut_at_gate"] = self.slots.cut_waiting_drafts()
                if self.cfg.wake_verify:
                    # The next batch's drafts end here, unfinished where they stand: that batch is complete at this
                    # stand-down and is verified against theta_{k+1} at the start of the next phase.
                    st.crit.update(await self.cut_drafts())
                    st.times["colo_cut"] = time.perf_counter() - tc
                await asyncio.to_thread(self.sc._gen.finish_generation)
                st.times["colo_standdown2"] = time.perf_counter() - tc
            t_post = time.perf_counter()
            st.times["continue"] = time.perf_counter() - (t2 if t2 is not None else tw)
            t3 = time.perf_counter()
            for g, res in zip(gids, results):
                metas[g], recs_out[g] = await finish_group(g, res)
            st.times["rewrite"] = time.perf_counter() - t3
        if t_done:
            # Critical path: the group whose tail finished last, its tail duration and its longest row's tokens.
            gl = max(t_done, key=lambda g: t_done[g])
            gt = max(t_done, key=lambda g: t_done[g] - t_launch[g])
            dur = t_done[gt] - t_launch[gt]
            st.crit.update(
                last_tail_done=round(t_done[gl] - t0, 2), last_tail_chunk=chunk_of.get(gl, -1),
                last_tail_launch=round(t_launch[gl] - t0, 2),
                tail_max_s=round(dur, 2), tail_max_tok=int(cont_max.get(gt, 0)),
                tail_ms_per_tok=round(1000.0 * dur / max(cont_max.get(gt, 0), 1), 2),
            )
        if self.cfg.tail_mtd:
            # Multi-token tail decode: per-engine-rank totals since the start (tail iterations, emitted tokens, extra
            # rows, width histogram); the step's tokens per tail iteration from the deltas.
            ms = [x for x in await asyncio.to_thread(
                ray.get, self._gen_wg().run_all_workers_single_data("nl_mtd_stats")) if x]
            tot = {k: sum(int(m.get(k, 0)) for m in ms) for k in
                   ("tail_iters", "tokens", "extra_rows", "accepted_extra", "forwards_with_extras", "registered",
                    "proposed", "neg_tokens")}
            prev = getattr(self, "_mtd_prev", {k: 0 for k in tot})
            d = {k: tot[k] - prev.get(k, 0) for k in tot}
            self._mtd_prev = tot
            widths = [sum(int(m.get("widths", [0])[i]) if i < len(m.get("widths", [])) else 0 for m in ms)
                      for i in range(int(self.cfg.tail_mtd) + 1)]
            st.crit.update(mtd_tok_per_iter=round(d["tokens"] / max(d["tail_iters"], 1), 4))
            print(f"[mtd] step={int(version)} ranks={len(ms)} tail_iters={d['tail_iters']} tokens={d['tokens']} "
                  f"tok/iter={d['tokens'] / max(d['tail_iters'], 1):.4f} extra_rows={d['extra_rows']} "
                  f"accepted_extra={d['accepted_extra']} proposed={d['proposed']} fwd_with_extras={d['forwards_with_extras']} "
                  f"registered={d['registered']} widths_total={widths}", flush=True)
        if getattr(self.cfg, "tail_tree", 0):
            # Tree speculation: per-engine-rank totals since the start; the step's deltas. crit = each rank's tail
            # with the longest remaining budget (the critical row's tokens per iteration).
            ms = [x for x in await asyncio.to_thread(
                ray.get, self._gen_wg().run_all_workers_single_data("nl_tree_stats")) if x]
            from nemo_rl.models.generation.megatron import tree_spec as _ts

            line, tot = _ts.stats_line(int(version), ms, getattr(self, "_tree_prev", None))
            prev = getattr(self, "_tree_prev", None) or {k: 0 for k in tot}
            self._tree_prev = tot
            d_tok, d_it = tot["tokens"] - prev["tokens"], tot["tail_iters"] - prev["tail_iters"]
            d_ct, d_ci = tot["crit_tokens"] - prev["crit_tokens"], tot["crit_iters"] - prev["crit_iters"]
            st.crit.update(tree_tok_per_iter=round(d_tok / max(d_it, 1), 4),
                           tree_crit_tok_per_iter=round(d_ct / max(d_ci, 1), 4))
            print(line, flush=True)
        new_meta = None
        batch_rows = []
        def edv_finish_step(step: int) -> None:
            """EDV bookkeeping: CHECKS replay comparison (and the records' deferred log-probs := the replay's p, so
            the mismatch check covers them), hidden of deferred rows freed, the controller fed, the [edv] line."""
            if nolanes.checks_on():
                for key, ok in sorted(edv_committed.items()):
                    res = edv_replay.get(key)
                    if res is None:
                        continue
                    edv_stats["replay_rows"] += 1
                    m = edv_m_of(key) if key in edv_rows else float(res.get("null_m") or 0.0)
                    if key not in edv_rows:  # verified after a surprise: committed False; its replay says not whole
                        if ok:
                            edv_stats["replay_mismatch"] += 1
                        continue
                    tau = res["null_grid"][m][0]
                    sh = res.get("edv_shadow") or {}
                    bg_ok = (key not in edv_loss_bg
                             or (edv_loss_bg[key][0] == sh.get("b_g", {}).get(m) and edv_loss_bg[key][1] == sh.get("eta_g")))
                    if (tau == len(draft_by_key[key].tokens)) != ok or not bg_ok:
                        edv_stats["replay_mismatch"] += 1
                key_lp = {}
                for k, r in edv_replay.items():
                    if k not in edv_rows:
                        continue
                    h_, i_ = edv_rows[k]
                    m_ = h_["pools"][h_["groups"][i_]].out.get(i_, {}).get("null_m")  # None: not an output
                    if m_ is not None and r["null_grid"][float(m_)][0] == len(draft_by_key[k].tokens):
                        key_lp[k] = r["token_logprobs"]
                for g in gids:
                    for c, o in zip(recs_out[g].completions, specs[g]):
                        k = vkey_of.get(id(o.draft)) if o.draft is not None else None
                        if k in key_lp and o.kind == "whole":
                            a = nolanes._assistant(c.message_log)
                            a["generation_logprobs"] = torch.tensor(
                                key_lp[k][: len(a["token_ids"])], dtype=a["generation_logprobs"].dtype)
            spec._release_hidden(h["flat"][i][1] for h, i in edv_rows.values())
            n_sur = edv_stats["surprises"]
            n_neg_edv = int(spec.stats.pop("nl/neg_content_m_edv_rows", 0.0))
            line = (f"[edv] step={step} e={edv_e:.3f} deferred={edv_stats['deferred']} surprises={n_sur} "
                    f"neg_edv_rows={n_neg_edv} "
                    f"fallback={edv_stats['fallback']} replay_rows={edv_stats['replay_rows']} "
                    f"replay_mismatch={edv_stats['replay_mismatch']}")
            neg = edv.negative()
            neg_any = nolanes.negative()
            # Negative controls: the rows they touched on the EDV path (defer_no_decide: committed rows its verifier
            # rejects; content_m: EDV rows whose envelope it chose).
            n_neg = edv_stats["replay_mismatch"] if neg else n_neg_edv
            print(line + (f" negative tokens {n_neg}" if neg_any else ""), flush=True)
            st.crit.update({f"edv_{k}": v for k, v in edv_stats.items()})
            if edv_stats["replay_mismatch"] and not neg:
                raise RuntimeError(f"[edv] step {step}: {edv_stats['replay_mismatch']} deferred rows' committed "
                                   "decision differs from their verification")

        for g in gids:  # fixed group order (R2) for the step's metadata and checks
            batch_rows.append(recs_out[g])
            new_meta = metas[g] if new_meta is None else new_meta.concat(metas[g])
        nolanes.check_provenance(new_meta.tags, int(version), "verify stage output")
        if edv_on:
            edv_finish_step(int(version))
        if gz_on:
            # loo_c: the next step's scale from this step's FINAL rewards (F_{k+1}-measurable for step k+1).
            from thundersync_rl import grpo_z as _gz

            fr = [float(c.reward) for g in gids for c in recs_out[g].completions]
            gi = [i for i, g in enumerate(gids) for _ in recs_out[g].completions]
            self.gz_scale = _gz.next_step_scale(torch.tensor(gi), torch.tensor(fr))
            n_units = len(gz_units)
            print(f"[grpo_z] step={int(version)} mode={self.grpo_z} units={n_units} "
                  f"row_units={sum(1 for u in gz_units if not u.endswith('#c'))} next_scale={self.gz_scale:.4f}",
                  flush=True)
        if st.dropped_unlogged:
            msg = (f"[nolanes] WARN step={int(version)}: {st.dropped_unlogged} of {st.drafts} drafts dropped (hidden log or "
                   "version tags shorter than the draft); their slots decode fresh")
            print(msg, flush=True)
            if nolanes.checks_on() or st.dropped_unlogged > max(1, st.drafts // 100):
                raise RuntimeError(msg + " -- refused (CHECKS run, or more than 1% of the drafts)")
        st.times["total"] = time.perf_counter() - t0
        st.times["post_train"] = time.perf_counter() - t_post  # driver after the last train call (learner idle)
        diag = spec.__dict__.pop("_nl_reject_diag", [])
        early = [r for r in diag if r["tau"] < nolanes.EARLY_REJECT_POS]
        st.early_rejects = len(early)
        if nolanes.checks_on() and not self.neg:
            # Acceptance sanity (R30): the step's own rejection hazard per scanned position predicts the rejections
            # below position 8; a mispaired p / q rejects early far more often. Fail when the observed count is
            # beyond the Poisson tail (p < 1e-4) of 3x that prediction.
            p_tail = nolanes.early_reject_tail(len(diag), st.accepted_tokens, st.whole + st.continued, st.early_rejects)
            if p_tail < 1e-4:
                raise RuntimeError(
                    f"[nolanes] CHECKS step {int(version)}: {st.early_rejects} rejections below position 8 "
                    f"(Poisson tail {p_tail:.2e} against 3x the step's own hazard)"
                )
        for r in sorted(diag, key=lambda r: r["tau"])[:12]:
            print(f"[nolanes reject] step={int(version)} {r}", flush=True)
        print(f"[edv shadow] step={int(version)} rows={edv_shadow.COUNTS['rows']} missing={edv_shadow.COUNTS['missing']} "
              f"path={edv_shadow._PATH}", flush=True)
        edv_shadow.flush()
        line = st.line(int(version))
        stats = {k: v for k, v in spec.stats.items() if k.startswith(("spec/accepted", "spec/ds2", "nl/"))}
        if call_stats:
            stats.update(nolanes.merge_call_stats(call_stats))
        stats["nl/verify_calls"] = float(len(chunks))
        if (
            nolanes.checks_on()
            and os.environ.get("THUNDERSYNC_HIDDEN_DEVICE_GATHER_EXPECTED", "1") == "1"
            and stats.get("spec/ds2_rows", 0) >= 32
            and stats.get("spec/ds2_dg_checked", 0) == 0
        ):
            print(f"[nolanes] WARN step={int(version)}: device hidden gather tripwire ran 0 comparisons "
                  f"over {int(stats.get('spec/ds2_rows', 0))} DS2 rows (spec/ds2_dg_checked 0)", flush=True)
        print(line + f" hstats={hstats[:1]} spec={ {k: round(v, 4) for k, v in stats.items()} }", flush=True)
        if self.slots is not None:
            from thundersync_rl import engine_slots

            print(engine_slots.summary_line(int(version), self.slots.capacity, self.slots.take_stats()), flush=True)
        self.history.append({"step": int(version), **st.__dict__, **stats})
        if nolanes.checks_on():
            await asyncio.to_thread(self.exactness_checks, batch_rows, int(version))
        return new_meta

    def admission_version(self, engine_version: int, trainer_version: int) -> int:
        """The version the sampler's admission gate compares against (nolanes): the version the engine has installed,
        set right after the refit lands, so every admitted row prefills and starts on the newest weights
        (ACCEPT_DECAY.md §2; the trainer version is bumped ~1 s before the refit). X1 gate tool only
        (THUNDERSYNC_NL_DEBUG_ADMIT_AFTER_TAILS, refused without CHECKS): the stage holds admissions for its whole step
        (admit_hold_version = k - 1) until the optimizer step bumps the trainer version."""
        if self.admit_hold_version is None:
            return int(engine_version)
        return min(int(engine_version), max(self.admit_hold_version, int(trainer_version) - 1))

    def check_router_frozen(self, policy_result: dict, version: int) -> None:
        """User rule 11-02 (always on): after every optimizer step, every learner rank's router weights / bias /
        expert bias are bitwise those after the first step, and no router parameter is trainable (router_freeze.py;
        the fingerprints ride on finish_train_step's result)."""
        from thundersync_rl.router_freeze import check_router_reports

        reps = policy_result.get("nl_router_fp") or []
        if not reps:
            raise RuntimeError(f"[router frozen] step {version}: no router fingerprint from the learner")
        ok, key, n, trainable = check_router_reports(reps, getattr(self, "_router_fp0", None))
        if getattr(self, "_router_fp0", None) is None:
            self._router_fp0 = key
        print(f"[router frozen] step={version} ranks={len(reps)} tensors={n} trainable={len(trainable)} "
              f"{'unchanged ok' if ok else 'CHANGED FAIL'}", flush=True)
        if not ok:
            raise RuntimeError(f"[router frozen] step {version}: router weights / bias changed since the first step, or "
                               f"trainable router parameters {trainable[:3]} (tensors {n}); the router must be frozen")

    def check_grad_order(self, policy_result: dict, version: int) -> None:
        """Train overlap (R31): every learner rank folded exactly the step's trained groups into the order-invariant
        accumulator; CHECKS runs: re-summing the sampled gradient elements in other orders changes no bit (the
        grad_order negative, a plain fp32 arrival-order sum, must change some)."""
        if not self.cfg.train_overlap:
            return
        reps = policy_result.get("exact_grad_accum") or []
        if not reps:
            raise RuntimeError("[nolanes] train overlap: no gradient-accumulator report from the learner "
                               "(NEMO_RL_EXACT_GRAD_ACCUM not active)")
        caps = sorted({int(r["captures"]) for r in reps})
        gib = max(float(r.get("bytes", 0.0)) for r in reps) / 2**30
        mx = lambda k: max(float(r.get(k, 0.0)) for r in reps)  # noqa: E731
        print(f"[nolanes grad accum] step={version} ranks={len(reps)} captures={caps} groups_trained={self._groups_trained} "
              f"accumulator {gib:.2f} GiB/rank capture {mx('capture_s'):.2f} s/step park {mx('park_s'):.2f} s "
              f"unpark {mx('unpark_s'):.2f} s reserved {mx('reserved_gib'):.2f} GiB peak {mx('peak_gib'):.2f} GiB "
              f"(max rank)", flush=True)
        if caps != [self._groups_trained]:
            raise RuntimeError(f"[nolanes] train overlap step {version}: learner captures {caps} != groups trained "
                               f"{self._groups_trained} (a train call did not fold into the accumulator)")
        if not nolanes.checks_on():
            return
        checked = sum(float(r.get("checked", 0.0)) for r in reps)
        mism = int(sum(float(r.get("mismatch", 0.0)) for r in reps))
        naive = int(sum(float(r.get("naive_order_diff", 0.0)) for r in reps))
        neg = os.environ.get("THUNDERSYNC_NL_NEG", "").strip()
        acc = (f" grad_rel_l2_vs_fp32={mx('grad_rel_l2_vs_fp32'):.3e} floor_fp32_order={mx('grad_rel_l2_fp32_order_floor'):.3e}"
               f" adam_rel_l2_vs_fp32={mx('adam_rel_l2_vs_fp32'):.3e} adam_floor={mx('adam_rel_l2_fp32_order_floor'):.3e}"
               f" zero_here_nonzero_fp32={int(sum(float(r.get('zero_here_nonzero_fp32', 0.0)) for r in reps))}"
               f" replay_mismatch={int(sum(float(r.get('replay_mismatch', -1.0)) for r in reps))}"
               f" replay_rel_l2={mx('replay_rel_l2'):.3e}"
               f" replay_digest_equal={int(min(float(r.get('replay_digest_equal', 0.0)) for r in reps))}"
               f" moments_roundtrip_exact={int(min(float(r.get('moments_roundtrip_exact', 1.0)) for r in reps))}")
        line = (f"[grad order check] step={version} ranks={len(reps)} checked={int(checked)} mismatch={mism} "
                f"naive_order_diff={naive}" + acc + (f" negative tokens {mism}" if neg == "grad_order" else ""))
        print(line + (" FAIL" if mism or checked <= 0 else " ok"), flush=True)
        if (mism or checked <= 0) and not neg:
            raise RuntimeError(f"[nolanes] CHECKS step {version}: the step gradient depends on group arrival order "
                               f"({mism} of {int(checked)} sampled elements change) or nothing was checked")
        if not neg and (mx("replay_rel_l2") > 1e-2 or min(float(r.get("moments_roundtrip_exact", 1.0)) for r in reps) < 1):
            raise RuntimeError(f"[nolanes] CHECKS step {version}: a group's re-run gradient differs grossly from its "
                               "first run, or a parked Adam moment did not come back bitwise")

    def check_train_lp(self, step_metrics: dict, version: int) -> None:
        """Always-on sentinel: the training forward's log-prob of EVERY trained token equals its generation log-prob
        (the verified p / theta_k engine decode), bitwise. Replaces the one-row re-scoring forward (≈22 s/step at
        Super-L22) with the loss's own comparison (train_gen_lp_*, THUNDERSYNC_TRAIN_LP_CHECK set by configure_driver)."""
        if "train_gen_lp_inexact_tokens" not in step_metrics:
            raise RuntimeError("[nolanes] train-lp sentinel missing from the step metrics (THUNDERSYNC_TRAIN_LP_CHECK)")
        bad = float(step_metrics["train_gen_lp_inexact_tokens"])
        dmax = float(step_metrics.get("train_gen_lp_abs_diff_max", 0.0))
        tok = float(step_metrics.get("train_gen_lp_tokens", 0.0))
        print(f"[nolanes sentinel] step={version} train-lp tokens={tok:.0f} inexact={bad:.0f} max|d|={dmax:.3g}", flush=True)
        if bad != 0 or dmax != 0 or tok <= 0:
            if os.environ.get("THUNDERSYNC_NL_NEG", "").strip() and nolanes.checks_on():
                return  # negative-control arms (CHECKS runs only; R29 item 8) report and run on
            raise RuntimeError(
                f"[nolanes sentinel] step {version}: {bad:.0f} trained tokens' training log-probs differ from their "
                f"generation log-probs (max |d| {dmax:.3g}): engine / learner divergence"
            )

    # -- CHECKS=1 -------------------------------------------------------------------------------------------

    def _check_batch(self, records: list[Any], first_token_only: bool):
        """Learner check batch of the output rows: right-padded input ids, masks, generation log-probs."""
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict
        from nemo_rl.experience.payload import record_to_train_batch

        pad = int(self.sc._rollout_manager._impl._tokenizer.pad_token_id)
        rows = []
        for rec in records:
            tb = record_to_train_batch(rec, pad_value_dict={"token_ids": pad}, include_message_violation_fields=False)
            plen = sum(len(m["token_ids"]) for m in rec.prompt)
            for i in range(int(tb["input_lengths"].shape[0])):
                n = int(tb["input_lengths"][i])
                ids = tb["input_ids"][i, :n]
                tm = tb["token_mask"][i, :n].clone()
                lp = tb["generation_logprobs"][i, :n]
                if first_token_only:
                    n = min(n, plen + 1)
                    ids, lp = ids[:n], lp[:n]
                    tm = torch.zeros(n, dtype=tm.dtype)
                    if n > plen:
                        tm[plen] = 1
                rows.append((ids, tm, lp))
        width = max(int(r[0].numel()) for r in rows)
        width = ((width + 63) // 64) * 64
        b = len(rows)
        ids = torch.full((b, width), pad, dtype=torch.long)
        tm = torch.zeros((b, width), dtype=torch.float32)
        lp = torch.zeros((b, width), dtype=torch.float32)
        lens = torch.zeros(b, dtype=torch.long)
        for i, (x, m, l) in enumerate(rows):
            n = int(x.numel())
            ids[i, :n], tm[i, :n], lp[i, :n], lens[i] = x, m.float(), l.float(), n
        return BatchedDataDict(
            {"input_ids": ids, "input_lengths": lens, "token_mask": tm, "sample_mask": torch.ones(b),
             "generation_logprobs": lp}
        )

    def sentinel_check(self, records: list[Any], step: int) -> None:
        """Always on (R29): prev_logprobs := generation log-probs hides an engine / learner divergence from the stock
        monitors, so every step one row per learner data-parallel rank (chosen by a hash of the step, never by
        content) is re-scored by the learner and must match its generation log-probs bitwise."""
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        trainer = self.sc._trainer
        dp = max(1, int(getattr(trainer, "data_parallel_size", 1) or 1))
        flat = [(r, i) for r in records for i in range(len(r.completions))]
        if not flat:
            return
        start = zlib.crc32(f"nl-sentinel/{self.seed}/{step}".encode()) % len(flat)
        pick = [flat[(start + j) % len(flat)] for j in range(min(dp, len(flat)))]
        sub = []
        for r, i in pick:
            sub.append(type(r)(prompt_idx=r.prompt_idx, prompt=r.prompt, extra_env_info=r.extra_env_info,
                               metadata=r.metadata, completions=[r.completions[i]], rollout_metrics=r.rollout_metrics,
                               loss_multiplier=r.loss_multiplier))
        data = self._check_batch(sub, first_token_only=False)
        ray.get(trainer.worker_group.run_all_workers_single_data(
            "prepare_for_lp_inference", keep_train_buffers=bool(self.trained_in_stage) or self.colocated))  # open step / resident colocated: keep buffers
        lp = trainer.get_logprobs(data)["logprobs"]
        mask = (data["token_mask"] * data["sample_mask"].unsqueeze(-1)).bool()
        mask[:, 0] = False
        diff = (lp - data["generation_logprobs"])[mask]
        n, exact = int(diff.numel()), int((diff == 0).sum())
        print(f"[nolanes sentinel] step={step} rows={len(pick)} tok={n} exact={exact}", flush=True)
        if exact != n:
            raise RuntimeError(f"[nolanes sentinel] step {step}: {n - exact} of {n} tokens' learner log-probs differ "
                               "from their generation log-probs (engine / learner divergence)")

    def exactness_checks(self, records: list[Any], step: int) -> None:
        from thundersync_rl import law_check

        trainer = self.sc._trainer
        ray.get(trainer.worker_group.run_all_workers_single_data(
            "prepare_for_lp_inference", keep_train_buffers=bool(self.trained_in_stage) or self.colocated))  # open step / resident colocated: keep buffers
        data = self._check_batch(records, first_token_only=False)
        out = trainer.get_logprobs(data, law_check={"seed": self.seed * 7919 + step})
        lp = out["logprobs"]
        mask = (data["token_mask"] * data["sample_mask"].unsqueeze(-1)).bool()
        mask[:, 0] = False
        diff = (lp - data["generation_logprobs"])[mask]
        n, exact = int(diff.numel()), int((diff == 0).sum())
        print(f"[nolanes mismatch] step={step} tok={n} exact={exact} max|d|={float(diff.abs().max()) if n else 0.0:.6g}",
              flush=True)
        failures = [] if (n and exact == n) else [f"mismatch {n - exact} of {n} tokens"]
        stats = out["law_stats"].double().sum(0)[: law_check.N_STATS]
        print(law_check.format_line(step, stats), flush=True)
        st_null = law_check.pop_null_emission()
        if st_null is not None:
            print(law_check.format_line(step, st_null).replace("[law check]", "[emission law nullbv]"), flush=True)
        em = [x for x in ray.get(trainer.worker_group.run_all_workers_single_data("pop_emission_law")) if x]
        for kind in ("residual", "bonus"):
            s = torch.tensor([x[kind] for x in em], dtype=torch.float64).sum(0) if em else None
            if s is not None and s[law_check.N_BINS] > 0:
                print(law_check.format_line(step, s).replace("[law check]", f"[emission law {kind}]"), flush=True)
        # First output token of every row (row-selection law: ⊥-BV pool choices, E3).
        data1 = self._check_batch(records, first_token_only=True)
        out1 = trainer.get_logprobs(data1, law_check={"seed": self.seed * 7919 + step + 104729})
        s1 = out1["law_stats"].double().sum(0)[: law_check.N_STATS]
        print(law_check.format_line(step, s1).replace("[law check]", "[law check first-token]"), flush=True)
        # The same rows pooled over every step so far: a row-selection bias (E3) is a small per-step shift.
        self._first_cum = s1 if getattr(self, "_first_cum", None) is None else self._first_cum + s1
        print(law_check.format_line(step, self._first_cum).replace("[law check]", "[law check first-token cumulative]"),
              flush=True)
        for name, st_ in (("law", stats), ("first-token", s1), ("first-token cumulative", self._first_cum)):
            if law_check.law_test(st_).get("p", 1.0) < law_check.alpha():
                failures.append(f"{name} law")
        if failures and os.environ.get("THUNDERSYNC_NL_NEG", "").strip() == "":
            # Negative-control arms run to the end so every check reports; a positive run stops here.
            raise RuntimeError(f"[nolanes] CHECKS step {step} failed: {failures}")
        eng = [x for x in ray.get(self._gen_wg().run_all_workers_single_data("nl_engine_stats")) if x]
        print(f"[nolanes engine] step={step} {eng[:1]}", flush=True)
