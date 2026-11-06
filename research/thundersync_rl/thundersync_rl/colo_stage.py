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
"""Driver stage of the colo mode (thundersync_rl/colo.py has the conditions): the hooks the SingleController loop
calls on its ``_nl``. Sync (lag 0): no verification; sentinel, router check, and the CHECKS=1 exactness tests."""

from __future__ import annotations

import asyncio
import os
import time
import zlib
from typing import Any, Optional

from thundersync_rl.colo import SKIP_RESHARD_ENV, ColoConfig
from thundersync_rl.nolanes_stage import VerifyStage


class ColoStage(VerifyStage):
    """Driver side of colo mode, owned by the SingleController actor (the hooks the loop calls on its ``_nl``)."""

    def __init__(self, sc: Any, master_config: dict, cfg: ColoConfig):  # noqa: D107 (no VerifyStage init)
        from thundersync_rl import nolanes

        self.sc = sc
        self.mc = master_config
        self.cfg = cfg
        self.trained_in_stage = False
        self.stage_has_valid_tokens = False
        self._groups_trained = 0
        self.admit_hold_version: Optional[int] = None
        self.g = int(master_config["grpo"]["num_generations_per_prompt"])
        self.n = self.g
        self.seed = int(master_config["grpo"].get("seed", 0))
        self.records: dict[str, tuple[Any, Any]] = {}
        self.history: list[dict[str, Any]] = []
        self.neg = nolanes.negative()
        # Generation log-probs are theta_k's own (bitwise the learner's; the sentinel checks every trained token).
        self.prev_from_verify = False

    def setup(self, version: int) -> None:
        import ray

        salt = zlib.crc32(f"wide/{self.seed}/{self.cfg.seed_stream}".encode()) | ((self.seed & 0x7FFFFFFF) << 32)
        fresh_salt = None
        if getattr(self.cfg, "stable_keys", False):
            from thundersync_rl.speculative import wide_fresh_salt

            fresh_salt = wide_fresh_salt(self.seed)
            self._install_stable_keys()
        out = ray.get(self._gen_wg().run_all_workers_single_data(
            "install_wide_sampler", salt=salt, key64=bool(self.cfg.wide_sampler_key64), fresh_salt=fresh_salt))
        if getattr(self.cfg, "tree_xssd", None):
            tr = [x for x in ray.get(self._gen_wg().run_all_workers_single_data(
                "install_tree_xssd", fresh_salt=fresh_salt)) if x]
            if not tr:
                raise RuntimeError("[colo] tree_xssd: tree_spec not installed on any engine rank")
            print(f"[colo] XS-SD on tree_spec: {tr[0]} on {len(tr)} ranks", flush=True)
        if not any(isinstance(x, dict) for x in out):
            raise RuntimeError(f"[colo] keyed sampler not installed on any engine rank: {out}")
        print(f"[colo] stage ready: G={self.g} version={version} engine sampler=wide on "
              f"{sum(isinstance(x, dict) for x in out)} ranks negative={self.neg or 'none'}", flush=True)

    def stash_head(self, version: int) -> None:
        return None

    def announce_version(self, version: int) -> None:
        return None

    def check_engine_version(self, version: int) -> None:
        return None

    def check_grad_order(self, policy_result: dict, version: int) -> None:
        return None

    def admission_version(self, engine_version: int, trainer_version: int) -> int:
        # Lag 0: the in_order gate admits batch k at trainer version k, which a colocated loop installs in the engine
        # at the wake before any rollout of batch k runs (rollouts are stood down while training).
        self._key_version = int(trainer_version)
        return int(trainer_version)

    def _install_stable_keys(self) -> None:
        """Rows keyed by (seed, step, prompt idx, sample index): ``noise_seed`` in each fresh row's generation data."""
        impl = self.sc._rollout_manager._impl
        if getattr(impl, "_colo_stable_keys", False):
            return
        orig = impl._run_single_rollout
        stage = self

        async def run_single(input_sample, traj_idx, forced=None):
            if forced is None:
                idx = int(input_sample["idx"])
                key = zlib.crc32(f"colo/{stage.seed}/{stage._key_version}/{idx}/{int(traj_idx)}".encode())
                key |= (zlib.crc32(f"{idx}/{stage._key_version}".encode()) & 0x3FFFFFFF) << 32
                forced = {"generation_data": {"noise_seed": int(key)}}
            return await orig(input_sample, traj_idx, forced=forced)

        impl._run_single_rollout = run_single
        impl._colo_stable_keys = True
        self._key_version = 0

    def dump_rows(self, recs: list[Any], gids: list[str], step: int) -> None:
        """T1 material: every trained row's tokens and generation log-prob bits, keyed by (step, prompt idx, j)."""
        import json

        import numpy as np

        from thundersync_rl import nolanes

        d = self.cfg.row_dump
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"rows_{step}.jsonl"), "w") as f:
            for gid, rec in zip(gids, recs):
                idx = getattr(rec, "prompt_idx", None)
                for j, c in enumerate(rec.completions):
                    m = nolanes._assistant(c.message_log)
                    lp = np.asarray(m["generation_logprobs"].float().numpy(), dtype=np.float32)
                    f.write(json.dumps({"step": step, "gid": gid, "idx": None if idx is None else str(idx), "j": j,
                                        "tokens": [int(t) for t in m["token_ids"].tolist()],
                                        "lp_bits": lp.view(np.uint32).tolist()}) + "\n")

    async def run(self, meta: Any, version: int) -> Any:
        from nemo_rl.data_plane.schema import GROUP_ID_TAG

        from thundersync_rl import nolanes

        t0 = time.perf_counter()
        gids: list[str] = []
        for t in meta.tags:
            gid = t[GROUP_ID_TAG]
            if gid not in gids:
                gids.append(gid)
        prompt_idx = {t[GROUP_ID_TAG]: t.get("prompt_idx") for t in meta.tags}
        gids.sort(key=lambda g: (str(prompt_idx.get(g)), g))
        missing = [g for g in gids if g not in self.records]
        if missing:
            raise RuntimeError(f"[colo] no rollout record for groups {missing[:5]} (record sink not installed?)")
        stale = {int(t.get("weight_version", version)) for t in meta.tags} - {int(version)}
        if stale:
            raise RuntimeError(f"[colo] step {version}: rows tagged with weight versions {sorted(stale)} (lag 0 "
                               "admits only theta_k rows)")
        recs = [self.records.pop(g)[1] for g in gids]
        rows = sum(len(r.completions) for r in recs)
        toks = sum(len(nolanes._assistant(c.message_log)["token_ids"]) for r in recs for c in r.completions)
        print(f"[colo stage] step={int(version)} groups={len(gids)} rows={rows} gen_tokens={toks} "
              f"t={time.perf_counter() - t0:.2f}s", flush=True)
        if getattr(self.cfg, "tree_xssd", None):  # per-step tree acceptance (tree_spec counters of every engine rank)
            import ray

            from nemo_rl.models.generation.megatron import tree_spec as _ts

            ms = [x for x in await asyncio.to_thread(
                ray.get, self._gen_wg().run_all_workers_single_data("nl_tree_stats")) if x]
            line, self._tree_prev = _ts.stats_line(int(version), ms, getattr(self, "_tree_prev", None))
            print(line, flush=True)
        if getattr(self.cfg, "row_dump", None):
            self.dump_rows(recs, gids, int(version))
        self.history.append({"step": int(version), "groups": len(gids), "rows": rows, "gen_tokens": toks})
        if nolanes.checks_on():
            # Learner forwards need the colocated engine stood down: megatron's InferenceMode flag is process-wide and
            # set while the engine is awake (a learner forward would take the inference paths). The loop's own
            # finish_generation right after this stage is then a no-op sleep (CHECKS runs only).
            if self.sc._gen.blocks_training():
                await asyncio.to_thread(self.sc._gen.finish_generation)
            await asyncio.to_thread(self.exactness_checks, recs, int(version))
        return meta

    def exactness_checks(self, records: list[Any], step: int) -> None:
        """Mismatch (every output token, bitwise) + law tests (all tokens, first tokens, first tokens cumulative)."""
        import ray
        import torch

        from thundersync_rl import law_check, nolanes

        trainer = self.sc._trainer
        ray.get(trainer.worker_group.run_all_workers_single_data("prepare_for_lp_inference", keep_train_buffers=False))
        data = self._check_batch(records, first_token_only=False)
        out = trainer.get_logprobs(data, law_check={"seed": self.seed * 7919 + step})
        lp = out["logprobs"]
        mask = (data["token_mask"] * data["sample_mask"].unsqueeze(-1)).bool()
        mask[:, 0] = False
        diff = (lp - data["generation_logprobs"])[mask]
        n, exact = int(diff.numel()), int((diff == 0).sum())
        print(f"[colo mismatch] step={step} tok={n} exact={exact} max|d|={float(diff.abs().max()) if n else 0.0:.6g}",
              flush=True)
        failures = [] if (n and exact == n) else [f"mismatch {n - exact} of {n} tokens"]
        stats = out["law_stats"].double().sum(0)[: law_check.N_STATS]
        print(law_check.format_line(step, stats), flush=True)
        data1 = self._check_batch(records, first_token_only=True)
        out1 = trainer.get_logprobs(data1, law_check={"seed": self.seed * 7919 + step + 104729})
        s1 = out1["law_stats"].double().sum(0)[: law_check.N_STATS]
        print(law_check.format_line(step, s1).replace("[law check]", "[law check first-token]"), flush=True)
        self._first_cum = s1 if getattr(self, "_first_cum", None) is None else self._first_cum + s1
        print(law_check.format_line(step, self._first_cum).replace("[law check]", "[law check first-token cumulative]"),
              flush=True)
        for name, st_ in (("law", stats), ("first-token", s1), ("first-token cumulative", self._first_cum)):
            if law_check.law_test(st_).get("p", 1.0) < law_check.alpha():
                failures.append(f"{name} law")
        negative = bool(os.environ.get("THUNDERSYNC_NL_NEG", "").strip()) or os.environ.get(SKIP_RESHARD_ENV) == "1"
        print(f"[colo checks] step={step} {'FAIL ' + str(failures) if failures else 'EXACT'}", flush=True)
        if failures and not negative:
            raise RuntimeError(f"[colo] CHECKS step {step} failed: {failures}")
        del torch

    def check_train_lp(self, step_metrics: dict, version: int) -> None:
        if os.environ.get(SKIP_RESHARD_ENV) == "1":
            # Negative control: report the sentinel, never stop on it (the run must reach its later steps).
            bad = float(step_metrics.get("train_gen_lp_inexact_tokens", -1))
            print(f"[nolanes sentinel] step={version} train-lp tokens={float(step_metrics.get('train_gen_lp_tokens', 0)):.0f} "
                  f"inexact={bad:.0f} max|d|={float(step_metrics.get('train_gen_lp_abs_diff_max', 0.0)):.3g} "
                  "(negative control skip_reshard)", flush=True)
            return
        super().check_train_lp(step_metrics, version)
