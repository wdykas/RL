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
"""Driver hooks of IS-SPR (thundersync_rl/isspr.py has the method). Per step, before the stand-down: every trained
row's certificate (log-probs of theta_k), the group weights W_g written as the ``is_weight`` column the loss reads,
and the weight log (eps_hat, mean / var of s_g, ESS). The train-lp sentinel (T3) runs every step (colo mode's).

CHECKS=1 adds the rescore audit: every trained row rescored again under theta_k from scratch. Its log-probs must
equal the row's generation log-probs bitwise, and the audited s_g (= s_g + sum(p_audit - p)) must equal the logged
s_g. The stale-KV negative fails both: its rows carry q, not theta_k's log-probs (the learner sentinel fails too),
and the audited weights show the stale-state divergence (Var(s_g) up, ESS down)."""

from __future__ import annotations

import asyncio
import math
import os
import time
from typing import Any

from thundersync_rl import isspr, nolanes
from thundersync_rl.xspr_stage import XsprStage, _mean


class IsSprStage(XsprStage):
    """Driver side of isspr mode, owned by the SingleController actor (the hooks the loop calls on its ``_nl``)."""

    extra_train_fields = (isspr.IS_WEIGHT_FIELD,)
    rows_cls = isspr.IsRows
    negative = staticmethod(isspr.negative)

    @property
    def wake_cut(self) -> bool:
        # Stale-KV negative: no wake cut, so paused rows resume on their old state (stock persist).
        return self.rows.neg != isspr.NEG_STALE_KV

    @property
    def prefill_cut(self) -> bool:
        return self.rows.neg != isspr.NEG_NO_PREFILL_CUT

    @property
    def stale_cache(self) -> bool:
        return self.rows.neg == isspr.NEG_STALE_CACHE

    def __init__(self, sc: Any, master_config: Any, cfg: isspr.IsSprConfig):  # noqa: D107
        isspr.require_group_local(sc._advantage_stage_config)
        isspr.require_rloo(sc._advantage_stage_config, cfg)
        super().__init__(sc, master_config, cfg)
        # Row unit: s_r, w_r and their log come from the learner's forward (isspr.py "Row weights from the learner").
        self.learner_s = cfg.unit == "row"
        self.rows.learner_s = self.learner_s
        if self.learner_s:
            self.extra_train_fields = (isspr.IS_WEIGHT_FIELD, isspr.IS_OLD_FIELD) + (
                (isspr.IS_AUDIT_FIELD,) if nolanes.checks_on() else ())

    async def run(self, meta: Any, version: int) -> Any:
        import ray
        import torch

        from nemo_rl.data_plane.column_io import write_columns
        from nemo_rl.data_plane.schema import GROUP_ID_TAG

        gids: list[str] = []
        for t in meta.tags:
            if t[GROUP_ID_TAG] not in gids:
                gids.append(t[GROUP_ID_TAG])
        missing = [g for g in gids if g not in self.records]
        if missing:
            raise RuntimeError(f"[isspr] no rollout record for groups {missing[:5]}")
        groups = [self.records.pop(g) for g in gids]
        # The step's rows (prompt and token ids, row keys), dumped if the learner's step goes non-finite.
        self._step_rows = (int(version), groups)
        certs = [[self.rows.pop_cert(version, rec.prompt_idx, j) for j in range(len(rec.completions))]
                 for _, rec in groups]
        flat = [c for cs in certs for c in cs]
        s = [math.fsum(c.s for c in cs) for cs in certs]
        s_rows = [c.s for c in flat]
        row = self.cfg.unit == "row"
        bad = [c for c in flat if c.version != int(version)]
        eng = [x for x in ray.get(self._gen_wg().run_all_workers_single_data("xspr_engine_stats")) if x]
        evicted = sum(int(x.get("evicted", 0)) for x in eng)
        now = time.perf_counter()
        if self.learner_s:
            # s_r and the weight log are the learner's (check_train_lp prints them after the step).
            weights = "s=learner "
        else:
            st = isspr.weight_stats(s_rows if row else s, self.cfg.clip_c)
            st_other = isspr.weight_stats(s if row else s_rows, self.cfg.clip_c)
            dp = isspr.reward_shift(s_rows, [float(c.reward) for _, rec in groups for c in rec.completions])
            weights = (f"{'group' if row else 'row'}_ess={st_other['ess']:.6f} "
                       f"{'group' if row else 'row'}_s_var={st_other['s_var']:.6g} dp_hat={dp:.6g} "
                       f"eps_hat={st['eps_hat']:.6g} s_mean={st['s_mean']:.6g} s_var={st['s_var']:.6g} "
                       f"s_min={st['s_min']:.6g} s_max={st['s_max']:.6g} ess={st['ess']:.6f} "
                       f"ess_clip={st['ess_clip']:.6f} clipped={st['clipped']:.0f} w_mean={st['w_mean']:.6f} "
                       f"age={_ages(isspr.age_stats(int(version), flat))} ")
        print(f"[isspr] step={int(version)} groups={len(s)} rows={len(flat)} uncertified={len(bad)} "
              f"weight={self.cfg.weight} unit={self.cfg.unit} {weights}"
              f"rescored_rows={sum(1 for c in flat if c.rescores)} rescores={sum(c.rescores for c in flat)} "
              f"rescored_tok={sum(c.rescored_tokens for c in flat)} prefill_tok={sum(c.prefill_tokens for c in flat)} "
              f"only_tok={sum(c.only_tokens for c in flat)} resume_tok={sum(c.resume_tokens for c in flat)} "
              f"old_tok={sum(c.old_tokens for c in flat)} "
              f"engine_check_s={sum(float(x.get('check_s', 0.0)) for x in eng):.2f} "
              f"cut={sum(int(x.get('cut', 0)) for x in eng)} evicted={evicted} "
              f"pc_skipped_tok={sum(int(x.get('pc_skipped_tok', 0)) for x in eng)} "
              f"pc_matched_blocks={sum(int(x.get('pc_matched_blocks', 0)) for x in eng)} "
              f"check_chunked={sum(int(x.get('check_chunked', 0)) for x in eng)} "
              f"check_void={sum(int(x.get('check_void', 0)) for x in eng)} "
              f"cut_midprefill={sum(int(x.get('cut_midprefill', 0)) for x in eng)} "
              f"check_max_blocks={max([int(x.get('check_max_blocks', 0)) for x in eng] or [0])} "
              f"check_max_shared={max([int(x.get('check_max_shared', 0)) for x in eng] or [0])} "
              f"ms_per_iter={_mean(eng, 'ms_per_iter'):.2f} t_since_last={now - self._t_step:.1f}s", flush=True)
        self._t_step = now
        if not self.learner_s:
            self.history.append({"step": int(version), **st, "dp_hat": dp, "uncertified": len(bad)})
            print(f"[isspr rows] step={int(version)} s_r={','.join(f'{x:.4g}' for x in s_rows)}", flush=True)
        tk = isspr.token_stats([c.log_ratios for c in flat if c.rescored_tokens and c.log_ratios is not None])
        if tk["tokens"]:
            print(f"[isspr tokens] step={int(version)} tokens={tk['tokens']} "
                  f"r_quantiles({','.join(f'{x:g}' for x in isspr.TOKEN_QUANTILES)})="
                  f"{','.join(f'{x:.3g}' for x in tk['q'])} abs_share_top0.1%={tk['abs_top0.001']:.3f} "
                  f"abs_share_top1%={tk['abs_top0.01']:.3f} sq_share_top0.1%={tk['sq_top0.001']:.3f} "
                  f"sq_share_top1%={tk['sq_top0.01']:.3f}", flush=True)
        negative = bool(self.rows.neg)
        if evicted:
            raise RuntimeError(f"[isspr] step {version}: {evicted} engine evictions (D5)")
        if bad and not negative:
            raise RuntimeError(f"[isspr] step {version}: {len(bad)} rows' log-probs not theta_{version}'s "
                               f"(versions {sorted({c.version for c in bad})[:5]})")
        # Row (g, j) is the data-plane sample "{g}_g{j}" (payload.pack_payload: generation index j = completion j).
        # Per-row column: the row's clipped weight (its own, or its group's), times l_r / L_c under the constant length
        # normalizer (isspr.train_coefficients), l_r the row's trained token count.
        if self.learner_s:
            w = [1.0] * len(flat)  # the length coefficient alone: w_r is formed in the loss
        elif row:
            w = isspr.clipped_weights(s_rows, self.cfg.clip_c)
        else:
            w = [wg for wg, cs in zip(isspr.clipped_weights(s, self.cfg.clip_c), certs) for _ in cs]
        lengths = [nolanes._assistant(comp.message_log)["token_ids"].numel() for _, rec in groups
                   for comp in rec.completions]
        off = [(c.key, n, c.tokens) for c, n in zip(flat, lengths) if n != c.tokens]
        if off:
            raise RuntimeError(f"[isspr] step {version}: trained token counts differ from the certified rows' "
                               f"(key, trained, certified) {off[:3]}")
        keys = [f"{g}_g{j}" for g, cs in zip(gids, certs) for j in range(len(cs))]
        w_of = dict(zip(keys, isspr.train_coefficients(w, lengths, self.cfg)))
        unit = list(meta.sample_ids)
        unknown = [k for k in unit if k not in w_of]
        if unknown or len(unit) != len(flat):
            raise RuntimeError(f"[isspr] step {version}: {len(unit)} batch rows for {len(flat)} certified rows, "
                               f"unknown {unknown[:3]}")
        cols = {isspr.IS_WEIGHT_FIELD: torch.tensor([w_of[k] for k in unit], dtype=torch.float32)}
        if self.learner_s:
            old_of = dict(zip(keys, [c.old_tokens for c in flat]))
            cols[isspr.IS_OLD_FIELD] = torch.tensor([old_of[k] for k in unit], dtype=torch.int64)
            if nolanes.checks_on():
                lps_of = dict(zip(keys, await self.rescore_audit(groups, certs, s, int(version))))
                cols[isspr.IS_AUDIT_FIELD] = self._audit_column(meta, unit, lps_of)
        write_columns(self.sc._dp_client, meta, cols)
        if nolanes.checks_on() and not self.learner_s:
            await self.rescore_audit(groups, certs, s, int(version))
        return meta.with_fields(list(cols))

    def dump_step_rows(self, version: int) -> str:
        """Every row of the step that trained at ``version`` (key, prompt / token ids, generation log-probs, version,
        step) as JSON lines under the stage's dump directory; returns the path ("" without the step's rows)."""
        import json

        step, groups = getattr(self, "_step_rows", (None, None))
        if step is None:
            return ""
        os.makedirs(self.dump_dir, exist_ok=True)
        path = os.path.join(self.dump_dir, f"step{int(version)}_rows.jsonl")
        with open(path, "w") as f:
            for sample, rec in groups:
                prompt = [int(t) for m in sample["message_log"] for t in m["token_ids"]]
                for j, comp in enumerate(rec.completions):
                    msg = nolanes._assistant(comp.message_log)
                    f.write(json.dumps({"step": int(step), "version": int(version), "prompt_idx": str(rec.prompt_idx),
                                        "sample": j, "key": self.rows.key_of(step, rec.prompt_idx, j),
                                        "prompt_token_ids": prompt,
                                        "token_ids": [int(t) for t in msg["token_ids"]],
                                        "generation_logprobs": [float(x) for x in msg["generation_logprobs"]]}) + "\n")
        return path

    def _audit_column(self, meta: Any, unit: list, lps_of: dict) -> Any:
        """The trained rows' generation log-probs with every trained token's replaced by the audit's rescore (same
        layout; the loss compares the training forward with it bitwise, loss_functions.is_row_weights)."""
        import torch

        from nemo_rl.data_plane.column_io import read_columns

        data = read_columns(self.sc._dp_client, meta, ["generation_logprobs", "token_mask"])
        col = data["generation_logprobs"].clone().float()
        for i, k in enumerate(unit):
            at = torch.nonzero(data["token_mask"][i] != 0).flatten()
            lps = lps_of[k]
            if at.numel() != lps.numel():
                raise RuntimeError(f"[isspr audit] row {k}: {lps.numel()} audit log-probs for {at.numel()} tokens")
            col[i, at] = lps.to(col.dtype)
        return col

    def check_train_lp(self, step_metrics: dict, version: int) -> None:
        bad = [k for k in ("grad_norm", "loss", "train_gen_lp_abs_diff_max")
               if k in step_metrics and not math.isfinite(float(step_metrics[k]))]
        if float(step_metrics.get("is_nonfinite", 0.0)):
            bad.append("is_nonfinite")
        if bad:
            path = self.dump_step_rows(version)
            raise RuntimeError(f"[isspr] step {version}: the learner's step is non-finite ({', '.join(bad)}); its "
                               f"rows dumped to {path or '(none)'} for an offline replay")
        super().check_train_lp(step_metrics, version)
        if not self.learner_s:
            return
        m = {k: float(step_metrics.get(k, 0.0)) for k in (
            "is_rows", "is_s_sum", "is_s2_sum", "is_w_sum", "is_w2_sum", "is_e_sum", "is_e2_sum", "is_excess_sum",
            "is_clipped", "is_nonfinite", "is_old_tok", "is_audit_tok", "is_audit_inexact")}
        n = m["is_rows"]
        # Every trained row counted once (TP / CP / PP copies would inflate the sums and the ESS; REVIEW batch 50 L2).
        _, groups = getattr(self, "_step_rows", (None, []))
        rows = sum(len(rec.completions) for _, rec in groups)
        tokens = sum(nolanes._assistant(c.message_log)["token_ids"].numel() for _, rec in groups for c in rec.completions)
        if n != rows:
            raise RuntimeError(f"[isspr] step {version}: the learner weighted {n:.0f} rows, the step trained {rows}")
        if nolanes.checks_on() and m["is_audit_tok"] != tokens:
            raise RuntimeError(f"[isspr] step {version}: the learner audited {m['is_audit_tok']:.0f} tokens, the step "
                               f"trained {tokens}")
        if m["is_nonfinite"]:
            raise RuntimeError(f"[isspr] step {version}: {m['is_nonfinite']:.0f} rows with a non-finite learner s_r")
        mean = m["is_s_sum"] / n
        print(f"[isspr learner] step={int(version)} rows={n:.0f} unit=row s_mean={mean:.6g} "
              f"s_var={max(0.0, m['is_s2_sum'] / n - mean * mean):.6g} "
              f"ess={m['is_e_sum'] ** 2 / (n * m['is_e2_sum']):.6f} ess_clip={m['is_w_sum'] ** 2 / (n * m['is_w2_sum']):.6f} "
              f"eps_hat={m['is_excess_sum'] / n:.6g} clipped={m['is_clipped']:.0f} w_mean={m['is_w_sum'] / n:.6f} "
              f"old_tok={m['is_old_tok']:.0f}", flush=True)
        if nolanes.checks_on():
            ok = m["is_audit_tok"] > 0 and m["is_audit_inexact"] == 0
            print(f"[isspr audit] step={int(version)} learner tokens={m['is_audit_tok']:.0f} "
                  f"inexact={m['is_audit_inexact']:.0f} {'EXACT' if ok else 'FAIL'}", flush=True)
            if not ok and not self.rows.neg:
                raise RuntimeError(f"[isspr] CHECKS step {version}: {m['is_audit_inexact']:.0f} of "
                                   f"{m['is_audit_tok']:.0f} training log-probs differ from the engine's rescore")

    async def rescore_audit(self, groups: list, certs: list, s: list[float], version: int) -> Any:
        """Every trained row rescored under theta_version from scratch: log-probs bitwise the row's, s_g as logged.
        Once the audit's requests are prefilling, every xspr request in flight is cut (``xspr_cut_now``): audit checks
        caught mid-prefill come back void and are redone (with chunked prefill this is the path of a check a wake cuts
        mid-prefill); the redone count is logged, and the redone checks must still match bitwise."""
        import ray
        import torch

        from nemo_rl.data.multimodal_utils import VLLM_PROMPT_KEYS

        impl = self.sc._rollout_manager._impl

        async def one(prompt_log, native, key, tokens, lps):
            # No prefix-cache hit: an independent reference prefill under theta_k.
            redone = -1
            info = {"check": None}
            while info["check"] is None:
                redone += 1
                _, _, _, info, _ = await self.rows._segment(impl, prompt_log, native, key, tokens, lps.tolist(),
                                                            check="only", nocache=True)
            audit = torch.tensor(info["check"]["lps"], dtype=torch.float32)
            if audit.numel() != lps.numel():
                raise RuntimeError(f"[isspr audit] key {key}: {audit.numel()} log-probs for {lps.numel()} tokens")
            return bool(torch.equal(audit, lps)), math.fsum((audit.double() - lps.double()).tolist()), redone, audit

        jobs = []
        for (sample, rec), cs in zip(groups, certs):
            native = {k: sample[k] for k in VLLM_PROMPT_KEYS if k in sample}
            for comp, c in zip(rec.completions, cs):
                msg = nolanes._assistant(comp.message_log)
                jobs.append(one(sample["message_log"], native, c.key, [int(t) for t in msg["token_ids"].tolist()],
                                msg["generation_logprobs"].float()))
        tasks = [asyncio.ensure_future(j) for j in jobs]
        await asyncio.sleep(1.0)
        if not all(t.done() for t in tasks):
            await asyncio.to_thread(ray.get, self._gen_wg().run_all_workers_single_data("xspr_cut_now"))
        res = await asyncio.gather(*tasks)
        if self.learner_s:
            # The rows carry q: the comparison is the learner's (check_train_lp), against these log-probs.
            print(f"[isspr audit] step={version} rows={len(res)} redone={sum(r[2] for r in res)} "
                  f"rescored for the learner comparison", flush=True)
            return [r[3] for r in res]
        k, s_audit = 0, []
        for g, cs in enumerate(certs):
            s_audit.append(s[g] + math.fsum(res[k + j][1] for j in range(len(cs))))
            k += len(cs)
        lp_equal = sum(int(r[0]) for r in res)
        s_equal = sum(int(a == b) for a, b in zip(s_audit, s))
        st = isspr.weight_stats(s_audit, self.cfg.clip_c)
        ok = lp_equal == len(res) and s_equal == len(s)
        print(f"[isspr audit] step={version} rows={len(res)} lp_equal={lp_equal} groups={len(s)} s_equal={s_equal} "
              f"redone={sum(r[2] for r in res)} "
              f"audit_s_var={st['s_var']:.6g} audit_ess={st['ess']:.6f} audit_eps_hat={st['eps_hat']:.6g} "
              f"{'EXACT' if ok else 'FAIL'}", flush=True)
        if not ok and not os.environ.get("THUNDERSYNC_NL_NEG", "").strip():
            raise RuntimeError(f"[isspr] CHECKS step {version}: audit lp {lp_equal}/{len(res)} s {s_equal}/{len(s)}")


def _ages(st: dict[int, tuple[int, float]]) -> str:
    """age:rows:KL-per-token items, e.g. 1:512:8.1e-05."""
    return ",".join(f"{a}:{n}:{kl:.3g}" for a, (n, kl) in st.items())
