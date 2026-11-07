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
"""Driver hooks of exact SPR (thundersync_rl/xspr.py has the method). The rows are made exact in the rollout path
(``ExactRows``); this stage installs the engine side, checks every trained row's certificate, logs the per-step
check statistics, and keeps colo mode's always-on sentinel / router check. CHECKS=1 adds, per step, before the
stand-down:

- T1: every trained row decoded again from scratch under theta_k with its key (tokens AND log-probs bitwise);
- P1: every trained row checked again under theta_k (tau = its length, log-probs bitwise: engine prefill = decode);

then colo mode's learner checks (every output token's training log-prob bitwise; the law tests)."""

from __future__ import annotations

import asyncio
import os
import time
import zlib
from typing import Any

from thundersync_rl import xspr
from thundersync_rl.colo_stage import ColoStage


class XsprStage(ColoStage):
    """Driver side of xspr mode, owned by the SingleController actor (the hooks the loop calls on its ``_nl``).

    Subclasses reuse the engine side and the row machinery through ``rows_cls`` (the row loop), ``wake_cut`` (the
    engine's cut at the wake; a property, overridable) and ``negative`` (the active mutation)."""

    rows_cls = xspr.ExactRows
    # Negative-control hook (xspr_engine.install_check_cache_rule): check requests salted with the previous weight
    # epoch, so prefix-cache hits read the older version's KV.
    stale_cache = False
    # Negative-control hook (xspr_engine.cut_touched): requests part-way through a chunked prefill not cut at a wake.
    prefill_cut = True

    @property
    def wake_cut(self) -> bool:
        return self.rows.neg != xspr.NEG_NO_WAKE_CUT

    @staticmethod
    def negative() -> str:
        return xspr.negative()

    def __init__(self, sc: Any, master_config: Any, cfg: xspr.XsprConfig):  # noqa: D107
        seed = int(getattr(sc._algo_cfg, "seed", 0) or 0)
        g = int(sc._algo_cfg.num_generations_per_prompt)
        super().__init__(sc, {"grpo": {"num_generations_per_prompt": g, "seed": seed}}, cfg)
        self.max_new = int(master_config.policy["generation"]["max_new_tokens"])
        # Where the engine's non-finite sampled-log-prob guard dumps the row (xspr_engine.nonfinite_dump).
        self.dump_dir = os.path.join(master_config.logger.log_dir, "nonfinite_lp")
        self.rows = self.rows_cls(seed, cfg.seed_stream, self.max_new, sc._rollout_manager.weight_version,
                                  neg=self.negative(), keep_forks=bool(cfg.row_dump_dir))
        self.rows.dump_dir = self.dump_dir
        impl = sc._rollout_manager._impl
        if not hasattr(impl, "_generate_response"):
            raise RuntimeError("[xspr] needs the native rollout path (AsyncRolloutImpl)")
        impl.xspr = self.rows
        sc._rollout_manager.nl_record_sink = self.record_sink
        self._t_step = time.perf_counter()

    def setup(self, version: int) -> None:
        import ray

        salt = zlib.crc32(f"wide/{self.seed}/{self.cfg.seed_stream}".encode()) | ((self.seed & 0x7FFFFFFF) << 32)
        wg = self._gen_wg()
        out = ray.get(wg.run_all_workers_single_data("install_wide_sampler", salt=salt, key64=True, fresh_salt=None))
        if not any(isinstance(x, dict) for x in out):
            raise RuntimeError(f"[xspr] keyed sampler not installed on any engine rank: {out}")
        ok = ray.get(wg.run_all_workers_single_data("xspr_setup", chunk_rows=int(self.cfg.check_chunk_rows),
                                                       wake_cut=bool(self.wake_cut), dump_dir=self.dump_dir,
                                                       stale_cache=bool(self.stale_cache),
                                                       prefill_cut=bool(self.prefill_cut)))
        if not any(ok):
            raise RuntimeError(f"[xspr] check not installed on any engine rank: {ok}")
        print(f"[xspr] stage ready: G={self.g} version={version} engine ranks={sum(bool(x) for x in ok)} "
              f"mutation={self.rows.neg or 'none'}", flush=True)

    async def run(self, meta: Any, version: int) -> Any:
        import ray

        from nemo_rl.data_plane.schema import GROUP_ID_TAG

        from thundersync_rl import nolanes

        gids: list[str] = []
        for t in meta.tags:
            if t[GROUP_ID_TAG] not in gids:
                gids.append(t[GROUP_ID_TAG])
        missing = [g for g in gids if g not in self.records]
        if missing:
            raise RuntimeError(f"[xspr] no rollout record for groups {missing[:5]}")
        groups = [self.records.pop(g) for g in gids]
        certs = [self.rows.pop_cert(version, rec.prompt_idx, j) for _, rec in groups for j in range(len(rec.completions))]
        bad = [c for c in certs if c.version != int(version)]
        st = xspr.tau_stats(certs)
        eng = [x for x in ray.get(self._gen_wg().run_all_workers_single_data("xspr_engine_stats")) if x]
        evicted = sum(int(x.get("evicted", 0)) for x in eng)
        now = time.perf_counter()
        print(f"[xspr] step={int(version)} rows={st['rows']:.0f} uncertified={len(bad)} checked={st['checked']:.0f} "
              f"checks={st['checks']:.0f} mismatched={st['mismatched']:.0f} tau_min={st['tau_min']:.0f} "
              f"tau_p10={st['tau_p10']:.0f} tau_p50={st['tau_p50']:.0f} "
              f"tau_hist[0-15,16-127,128-1023,1024-4095,4096+]={st['tau_hist']} redecoded={st['redecoded']:.0f} "
              f"at_tau(med): lq_x={st['lq_x']:.4g} lp_x={st['lp_x']:.4g} lp_y={st['lp_y']:.4g} "
              f"dlp_max_before={st['dlp_max']:.3g} "
              f"check_prefill_tok={st['check_prefill']:.0f} cont_prefill_tok={st['cont_prefill']:.0f} "
              f"eos_earlier={st['eos_earlier']:.0f} eos_later={st['eos_later']:.0f} "
              f"engine_check_s={sum(float(x.get('check_s', 0.0)) for x in eng):.2f} "
              f"engine_check_rows={sum(int(x.get('check_rows', 0)) for x in eng)} "
              f"cut={sum(int(x.get('cut', 0)) for x in eng)} evicted={evicted} "
              f"occ_tail_rows={_mean(eng, 'occ_tail_rows'):.1f} occ_rows={_mean(eng, 'occ_rows'):.1f} "
              f"ms_per_iter={_mean(eng, 'ms_per_iter'):.2f} t_since_last={now - self._t_step:.1f}s", flush=True)
        self._t_step = now
        self.history.append({"step": int(version), **st, "uncertified": len(bad)})
        if self.cfg.row_dump_dir:
            await asyncio.to_thread(self.dump_rows, groups, certs, int(version))
        negative = bool(os.environ.get("THUNDERSYNC_NL_NEG", "").strip())
        if evicted:
            raise RuntimeError(f"[xspr] step {version}: {evicted} engine evictions (D5)")
        if bad and not negative:
            raise RuntimeError(f"[xspr] step {version}: {len(bad)} rows not certified at theta_{version} "
                               f"(versions {sorted({c.version for c in bad})[:5]})")
        if nolanes.checks_on():
            # Lookahead rows still decoding are cut first (they resume through their check at once), so the gate
            # exercises the cut path although the replays below outlast every lookahead row.
            cut = await asyncio.to_thread(ray.get, self._gen_wg().run_all_workers_single_data("xspr_cut_now"))
            print(f"[xspr] step={int(version)} CHECKS cut before replay: {sum(cut)} (rank sum)", flush=True)
            await self.replay_checks(groups, int(version))
            if self.sc._gen.blocks_training():
                await asyncio.to_thread(self.sc._gen.finish_generation)
            await asyncio.to_thread(self.exactness_checks, [rec for _, rec in groups], int(version))
        return meta

    def dump_rows(self, groups: list, certs: list, version: int) -> None:
        """``row_dump_dir``: one JSON line per trained row (prompt idx, sample j, prompt length and, on sample 0,
        prompt tokens, final tokens, every mismatch's tau and discarded suffix, pre-check length)."""
        import json

        from thundersync_rl import nolanes

        os.makedirs(self.cfg.row_dump_dir, exist_ok=True)
        it = iter(certs)
        with open(os.path.join(self.cfg.row_dump_dir, f"rows_step{version}.jsonl"), "w") as f:
            for sample, rec in groups:
                prompt = [int(t) for m in rec.prompt for t in m["token_ids"].tolist()]
                plen = len(prompt)
                for j, comp in enumerate(rec.completions):
                    c = next(it)
                    toks = [int(t) for t in nolanes._assistant(comp.message_log)["token_ids"].tolist()]
                    row = {"step": version, "idx": xspr._plain(rec.prompt_idx), "j": j, "plen": plen,
                           "prompt": prompt if j == 0 else None, "tokens": toks, "pre_check_len": c.pre_check_len,
                           "forks": [[int(t), [int(x) for x in d], q, p] for t, d, q, p in c.forks]}
                    f.write(json.dumps(row) + "\n")

    async def replay_checks(self, groups: list, version: int) -> None:
        """T1 (from-scratch keyed decode) and P1 (re-check) of every trained row under theta_version, bitwise."""
        from nemo_rl.data.multimodal_utils import VLLM_PROMPT_KEYS

        from thundersync_rl import nolanes

        impl = self.sc._rollout_manager._impl
        jobs = []
        for sample, rec in groups:
            native = {k: sample[k] for k in VLLM_PROMPT_KEYS if k in sample}
            for j, comp in enumerate(rec.completions):
                msg = nolanes._assistant(comp.message_log)
                key = self.rows.key_of(version, rec.prompt_idx, j)
                jobs.append((sample["message_log"], native, key, [int(t) for t in msg["token_ids"].tolist()],
                             msg["generation_logprobs"].float()))

        async def one(prompt_log, native, key, tokens, lps):
            import torch

            m, _, _ = await impl._generate_response(
                prompt_log, None, native_generation_data=native,
                forced={"prefix": [], "prefix_logprobs": [], "generation_data": {"xspr_key": key,
                                                                                 "max_new_tokens": self.max_new}})
            r_tok = [int(t) for t in m["token_ids"].tolist()]
            r_lp = m["generation_logprobs"].float()
            t1_tok = r_tok == tokens
            t1_lp = t1_tok and bool(torch.equal(r_lp, lps))
            _, _, _, info, _ = await self.rows._segment(impl, prompt_log, native, key, tokens,
                                                         lps.tolist(), check="only")
            chk = info["check"] or {}
            p1_tau = int(chk.get("tau", -1)) == len(tokens)
            p1_lp = p1_tau and bool(torch.equal(torch.tensor(chk.get("lps", []), dtype=torch.float32), lps))
            first = next((i for i, (a, b) in enumerate(zip(r_tok, tokens)) if a != b), min(len(r_tok), len(tokens)))
            return t1_tok, t1_lp, p1_tau, p1_lp, first

        res = await asyncio.gather(*(one(*j) for j in jobs))
        n = len(res)
        t1_tok, t1_lp, p1_tau, p1_lp = (sum(int(r[i]) for r in res) for i in range(4))
        diverge = sorted(r[4] for r in res if not r[0])[:5]
        t1_ok, p1_ok = t1_tok == n and t1_lp == n, p1_tau == n and p1_lp == n
        print(f"[xspr T1] step={version} rows={n} tokens_equal={t1_tok} lp_equal={t1_lp} first_divergence={diverge} "
              f"{'EXACT' if t1_ok else 'FAIL'}", flush=True)
        print(f"[xspr P1] step={version} rows={n} tau_full={p1_tau} lp_equal={p1_lp} {'EXACT' if p1_ok else 'FAIL'}",
              flush=True)
        if not (t1_ok and p1_ok) and not os.environ.get("THUNDERSYNC_NL_NEG", "").strip():
            raise RuntimeError(f"[xspr] CHECKS step {version}: T1 {t1_tok}/{t1_lp} P1 {p1_tau}/{p1_lp} of {n} rows")


def _mean(stats: list[dict], key: str) -> float:
    vals = [float(x[key]) for x in stats if key in x]
    return sum(vals) / len(vals) if vals else 0.0
