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
"""Exact SPR (``thundersync.mode=xspr``; docs/colo/THEORY.md): stock colocated
partial rollout (SingleController loop, in_order sampler at lag K, requests paused in the engine at the stand-down),
with every trained row an exact keyed sample of the weights it trains under.

Keyed sample-and-match. Token t of a row is y_t = argmax(logits_theta(prefix_<t) + g(row key, t)): the keyed
Gumbel race of the engine sampler with the row's key (``row_key``: unique per seed, target step, prompt, sample; one
key for the row's whole life). A row of batch k must train as the keyed decode of theta_k. Its tokens may come from
older weights (lookahead rows run during earlier phases; paused rows resume after the wake). So before batch k
trains, every row whose tokens are not all from theta_k decodes is CHECKED under theta_k: a prefill of prompt +
tokens recomputes y_t at every position t from 0; tau = the first t with y_t != x_t. Tokens before tau are theta_k's
own keyed tokens (they equal them), so they stay, with theta_k's log-probs from that prefill; the row resumes at tau
as a new request on prompt + x_<tau with the same key: its first token is y_tau, the rest theta_k's keyed decode. The
row is then token for token the keyed theta_k decode from scratch, whatever the pause points (pathwise).

Version label. ``ExactRows`` labels each piece of a row with the rollout weight version current when its request
was sent. The engine has installed that version by then, so the label never exceeds the version of any forward of
the request; and a row of batch k never runs after the stand-down of step k (the in_order batch is complete before
it). So a row labelled k throughout is all theta_k. Rows labelled lower are checked: rows the wake cut (they ran
before it) at once, finished rows when theta_k is installed (D1). Nothing is dropped (D5): a failure, retry,
eviction or abort raises.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass, field, fields
from typing import Any, Literal, Optional

MODE = "xspr"
# Mutation controls (THUNDERSYNC_NL_NEG; each must FAIL T1 or T3).
NEG_SKIP_CHECK = "xspr_skip_check"  # rows are never checked
NEG_NEW_ONLY = "xspr_new_only"  # the check covers only the tokens added since the row's last check / start
NEG_SKIP_FINISHED = "xspr_skip_finished"  # rows that finished before their batch's wake are not checked
NEG_FRESH_KEYS = "xspr_fresh_keys"  # the continuation after tau draws with a fresh key
NEG_KEEP_LP = "xspr_keep_lp"  # accepted tokens keep their generation log-probs (not theta_k's from the check)
NEG_STATE_PREFIX_END = "xspr_state_prefix_end"  # the continuation resumes from the end of the old prefix
# Stock SPR's assumption: no cut, paused rows resume on their stale KV, and a request is labelled with the version
# at its return. (The cut alone is not needed for exactness: with submission labels a resumed row is checked anyway.)
NEG_NO_WAKE_CUT = "xspr_no_wake_cut"
NEGATIVES = (NEG_SKIP_CHECK, NEG_NEW_ONLY, NEG_SKIP_FINISHED, NEG_FRESH_KEYS, NEG_KEEP_LP, NEG_STATE_PREFIX_END,
             NEG_NO_WAKE_CUT)
_POLL_S = 0.005


def _get(d: Any, path: str, default: Any = None) -> Any:
    cur = d
    for part in path.split("."):
        if cur is None:
            return default
        cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
    return default if cur is None else cur


@dataclass
class XsprConfig:
    """``thundersync:`` block of an SC config in xspr mode (every key optional except ``mode``)."""

    mode: Literal["xspr"] = MODE
    seed_stream: str = "xspr"
    # Positions whose logits exist at once in a check (each [V] fp32).
    check_chunk_rows: int = 256
    # Cross-fade routing (nolanes.crossfade_problems: required; off only with crossfade_off_reason).
    crossfade_routing_m: Optional[float] = None
    crossfade_routing_m_profile: Optional[list] = None
    crossfade_routing_extra: int = 8
    crossfade_routing_ramp_stopgrad: bool = True
    crossfade_routing_sparse: bool = True
    crossfade_routing_sparse_engine: bool = True
    crossfade_off_reason: Optional[str] = None
    # Measurement output: per step, every trained row's tokens and every mismatch's (tau, discarded suffix) as JSON
    # lines (<dir>/rows_step<k>.jsonl): the fork-tail speculation study (TAIL_FLOOR §7.2) replays them on the host.
    row_dump_dir: Optional[str] = None

    @classmethod
    def from_config(cls, config: Any) -> "XsprConfig":
        block = _get(config, "thundersync") or {}
        block = dict(block) if not isinstance(block, dict) else block
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(block) - known)
        if unknown:
            raise ValueError(f"thundersync (mode=xspr): unknown keys {unknown} (known: {sorted(known)})")
        out = cls(**{k: v for k, v in block.items() if k in known})
        if out.mode != MODE:
            raise ValueError(f"thundersync.mode must be {MODE!r} (got {out.mode!r})")
        if int(out.check_chunk_rows) < 1:
            raise ValueError("thundersync.check_chunk_rows must be >= 1")
        return out


def enabled(config: Any) -> bool:
    return _get(config, "thundersync.mode") == MODE


def config_problems(config: Any) -> list[str]:
    from thundersync_rl import nolanes

    p: list[str] = []
    pol = _get(config, "policy") or {}
    gen = _get(pol, "generation") or {}
    mcfg = _get(pol, "megatron_cfg") or {}
    mcore = _get(gen, "mcore_generation_config") or {}
    if _get(gen, "backend") != "megatron" or not _get(gen, "colocated.enabled", False):
        p.append("xspr needs colocated megatron generation")
    if not _get(mcfg, "batch_invariant_mode", False) or not _get(mcfg, "zero_train_gen_mismatch", False):
        p.append("xspr needs batch-invariant numerics (engine prefill = engine decode = learner, bitwise)")
    # KV: persist (paused requests keep their KV) or offload (released during training, restored at the wake). Either
    # way the wake cut + rescore rebuilds every carried row's state under the new weights; MCore's own recompute mode
    # would rebuild it a second time.
    if _get(mcore, "kv_cache_management_mode") not in ("persist", "offload") or _get(
            config, "async_rl.recompute_kv_cache_after_weight_updates", False):
        p.append("xspr needs kv_cache_management_mode persist or offload (the wake cut + rescore rebuilds the state)")
    max_seq = _get(pol, "max_total_sequence_length")
    if (not _get(mcore, "enable_chunked_prefill", False) and max_seq is not None
            and int(_get(mcore, "max_tokens", 0) or 0) < int(max_seq)):
        p.append(f"mcore_generation_config.max_tokens must be >= max_total_sequence_length ({max_seq}) without "
                 "chunked prefill: a check prefills prompt + row in one forward")
    if _get(mcore, "logprobs_mode") != "raw_logprobs" or not _get(mcore, "materialize_only_last_token_logits", False):
        p.append("xspr needs logprobs_mode raw_logprobs and materialize_only_last_token_logits (the check's rows)")
    if _get(gen, "top_k") not in (None, 0, -1) or float(_get(gen, "top_p", 1.0)) != 1.0:
        p.append("xspr needs the full keyed sampler (no top-k / top-p: those rows bypass the keyed race)")
    if _get(gen, "stop_strings"):
        p.append("xspr: no stop strings (a continuation's stop match would not see the kept prefix)")
    if _get(pol, "offload_optimizer_for_refit", True):
        p.append("policy.offload_optimizer_for_refit must be false (resident colocated learner)")
    if _get(config, "async_rl.sampler.name") != "in_order":
        p.append("xspr needs the in_order sampler (batch k complete before the stand-down of step k)")
    lag = int(_get(config, "async_rl.sampler.max_lookahead_versions", 0) or 0)
    prompts = _get(config, "grpo.num_prompts_per_step")
    if prompts is not None:
        need = (lag + 1) * int(prompts)
        for k in ("max_inflight_prompts", "max_buffered_rollouts"):
            if int(_get(config, f"async_rl.{k}", 0) or 0) < need:
                p.append(f"async_rl.{k} must be >= (K + 1) P = {need}: rows of a batch hold their permits until its "
                         "wake (finished rows wait for theta_k there)")
    if int(_get(config, "grpo.max_rollout_turns", 1) or 1) != 1:
        p.append("xspr is single-turn (grpo.max_rollout_turns 1)")
    if _get(config, "token_capture.enabled", False) or _get(config, "env.should_use_nemo_gym", False):
        p.append("xspr runs on the native rollout path (no token capture / NeMo Gym)")
    if _get(config, "async_rl.trajectory_streaming") is not None:
        p.append("xspr: no trajectory streaming (rows publish only when certified)")
    p += nolanes.router_frozen_problems(config)
    p += nolanes.crossfade_problems(crossfade_view(config), mcfg)
    return p


CROSSFADE_KEYS = ("crossfade_routing_m", "crossfade_routing_m_profile", "crossfade_routing_extra",
                  "crossfade_routing_ramp_stopgrad", "crossfade_routing_sparse", "crossfade_routing_sparse_engine",
                  "crossfade_off_reason")


def crossfade_view(config: Any):
    """The cross-fade fields of the config's thundersync block (XsprConfig's defaults where unset), whatever its
    mode (modes built on xspr share them)."""
    from types import SimpleNamespace

    block = dict(_get(config, "thundersync") or {})
    defaults = {f.name: f.default for f in fields(XsprConfig) if f.name in CROSSFADE_KEYS}
    return SimpleNamespace(**{k: block.get(k, v) for k, v in defaults.items()})


def validate_config(config: Any) -> XsprConfig:
    problems = config_problems(config)
    if problems:
        raise ValueError("thundersync.mode=xspr: " + "; ".join(problems) + ".")
    return XsprConfig.from_config(config)


def configure_worker(config: dict, cfg: Any) -> None:
    """The ThunderSync worker (keyed sampler, engine check, router fingerprints), cross-fade routing from ``cfg``
    (the mode's thundersync block), the engine's head-input capture (the check's rows) and the always-on train-lp
    sentinel."""
    from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES

    from thundersync_rl import nolanes

    ACTOR_ENVIRONMENT_REGISTRY[nolanes.WORKER_FQN] = PY_EXECUTABLES.MCORE
    config["policy"]["worker_extension_cls_fqn"] = nolanes.WORKER_FQN
    mcfg = config["policy"]["megatron_cfg"]
    nolanes.apply_crossfade(cfg, mcfg)
    mcfg["engine_verify"] = True
    config.setdefault("loss_fn", {})["train_lp_check"] = True


def configure_driver(config: dict) -> XsprConfig:
    """Driver side, before setup_single_controller: validate, layout fields, ``configure_worker``."""
    from thundersync_rl import nolanes

    cfg = validate_config(config)
    nolanes.derive_layout_fields(config)
    configure_worker(config, cfg)
    neg = nolanes.negative()
    lag = int(_get(config, "async_rl.sampler.max_lookahead_versions", 0) or 0)
    print(f"[xspr] mode=xspr exact colocated partial rollout lag={lag} check_chunk_rows={cfg.check_chunk_rows} "
          f"cf m={cfg.crossfade_routing_m} extra={cfg.crossfade_routing_extra} sparse={cfg.crossfade_routing_sparse} "
          f"checks={int(nolanes.checks_on())}" + (f" MUTATION {neg}" if neg else ""), flush=True)
    return cfg


def row_key(seed: int, stream: str, target_step: int, prompt_idx: Any, j: int) -> int:
    """The row's noise key id (62 bits): unique per (run seed, target step, prompt, sample index), never reused (a
    prompt that recurs in a later epoch trains at another step, so it gets another key)."""
    h = hashlib.blake2b(f"xspr/{seed}/{stream}/{int(target_step)}/{prompt_idx}/{int(j)}".encode(), digest_size=8)
    return int.from_bytes(h.digest(), "little") & ((1 << 62) - 1)


class KeyLedger:
    """The positions each row key has drawn (D4). A key belongs to one row; its draws are contiguous from 0 except
    the intended redraw: after a check found tau, the continuation redraws from tau (``permit``), once."""

    def __init__(self) -> None:
        self.hi: dict[int, int] = {}
        self.permit_at: dict[int, int] = {}

    def open(self, key: int) -> None:
        if key in self.hi:
            raise RuntimeError(f"[xspr] row key {key} used by two rows (D4: one key per row, never reused)")
        self.hi[key] = 0

    def permit(self, key: int, tau: int) -> None:
        if tau > self.hi[key]:
            raise RuntimeError(f"[xspr] redraw of key {key} from {tau} beyond its draws ({self.hi[key]})")
        self.permit_at[key] = int(tau)

    def draw(self, key: int, start: int, end: int) -> None:
        """A decode of the row drew positions [start, end)."""
        hi = self.hi.get(key)
        if hi is None:
            raise RuntimeError(f"[xspr] draw of unknown row key {key}")
        if start != hi:
            if self.permit_at.get(key) != start:
                raise RuntimeError(f"[xspr] key {key} drew from {start} with {hi} drawn and no redraw permit there "
                                   "(D4: the same key stream continues at tau, nothing else redraws)")
        self.permit_at.pop(key, None)
        self.hi[key] = int(end)

    def close(self, key: int) -> None:
        self.hi.pop(key, None)
        self.permit_at.pop(key, None)


@dataclass
class RowCert:
    """A finished row: the version all its tokens are certified at, and what it took."""

    version: int
    key: int
    tokens: int
    checks: int = 0
    taus: tuple = ()  # tau of every check (n where the check matched all n tokens)
    mism: tuple = ()  # tau of every check that found a mismatch
    forks: list = field(default_factory=list)  # (tau, discarded suffix, old lps, check lps) (keep_forks)
    # per mismatch: (log q(x_tau), log p(x_tau), log p(y_tau), max |log p - log q| over the kept tokens t < tau)
    flips: list = field(default_factory=list)
    redecoded: int = 0
    check_prefill: int = 0
    cont_prefill: int = 0
    pre_check_len: int = -1


class ExactRows:
    """The row loop of exact SPR (AsyncRolloutImpl._run_single_rollout calls ``respond`` for each row)."""

    def __init__(self, seed: int, stream: str, max_new_tokens: int, version_fn, neg: str = "",
                 keep_forks: bool = False) -> None:
        self.seed = int(seed)
        self.stream = stream
        self.max_new = int(max_new_tokens)
        self.version = version_fn
        self.neg = neg
        self.keep_forks = keep_forks
        self.ledger = KeyLedger()
        self.certs: dict[tuple, RowCert] = {}

    def key_of(self, target_step: int, prompt_idx: Any, j: int) -> int:
        return row_key(self.seed, self.stream, target_step, _plain(prompt_idx), j)

    async def wait_version(self, target: int) -> None:
        """Until the engine runs theta_target (rows of batch target hold here until its wake)."""
        while True:
            v = int(self.version())
            if v == target:
                return
            if v > target:
                raise RuntimeError(f"[xspr] a row of batch {target} is still open at version {v}")
            await asyncio.sleep(_POLL_S)

    async def _segment(self, impl, message_log, native, key: int, prefix: list, lps: list, check: str = "",
                       nocache: bool = False):
        """One engine request of the row on prompt + ``prefix``: a keyed decode (``check`` ""), the check of
        ``prefix`` alone ("only": one discarded token), or the check that decodes on from the end of ``prefix``
        when every token matches ("continue": the engine stops it at a mismatch) or always ("continue_always").
        ``nocache``: the request matches no prefix-cache block (an independent reference, e.g. a CHECKS audit).
        A check request the wake cut while its (chunked) prefill was part-way returns no result (info["check"] None,
        info["cut"] 1): nothing was checked and the caller redoes it. Returns (assistant message, input length, gen
        metrics, xspr info, version label)."""
        v = int(self.version())
        data = {"xspr_key": key, "max_new_tokens": self.max_new - len(prefix)}
        if check != "only" and data["max_new_tokens"] < 1:
            raise RuntimeError(f"[xspr] key {key}: a decode of a row already at the length cap ({len(prefix)})")
        if nocache:
            data["xspr_nocache"] = 1
        if check:
            data["xspr_check"] = len(prefix)
            data["xspr_continue"] = {"only": 0, "continue": 1, "continue_always": 2}[check]
        msg, in_len, metrics = await impl._generate_response(
            message_log, None, native_generation_data=native,
            forced={"prefix": list(prefix), "prefix_logprobs": list(lps), "generation_data": data})
        if self.neg == NEG_NO_WAKE_CUT:
            v = int(self.version())
        info = msg.pop("xspr", None)
        if info is None:
            raise RuntimeError("[xspr] the engine returned no xspr record (xspr_setup not run on the engine?)")
        if check and info["check"] is None and info["cut"]:
            if len(msg["token_ids"]) != len(prefix):
                raise RuntimeError(f"[xspr] key {key}: a check cut mid-prefill returned new tokens")
        elif check and (info["check"] is None or int(info["check"]["n"]) != len(prefix)):
            raise RuntimeError(f"[xspr] check of key {key}: missing or wrong result {info['check']}")
        # Every log-prob the row will carry is finite (host values already here: free). The sampler cannot draw a
        # token its logged law gives probability 0; a NaN / -inf means the engine's log-prob path and sampler disagree.
        require_finite(msg["generation_logprobs"][len(prefix):].tolist(), "decoded", key, len(prefix))
        if check and info["check"] is not None:
            require_finite(info["check"]["lps"], "rescored", key, 0)
        return msg, in_len, metrics, info, v

    def _finished(self, info: dict, tokens: list) -> bool:
        """A request's row is complete unless the engine cut it with room left under the length cap (a row cut in
        the step it reached the cap is complete)."""
        return not info["cut"] or len(tokens) >= self.max_new

    async def respond(self, impl, message_log, stop_strings, native, input_sample, traj_idx: int):
        """``_generate_response`` of one row of batch ``target`` (the rollout's target step), certified at theta_target.

        A finished row not all decoded at theta_target waits for it and is checked alone; an unfinished row (cut at
        a wake) is checked at once by a request that decodes on when nothing mismatches. At a mismatch tau the row
        resumes as a new request on prompt + x_<tau with the same key (its state is the prefill to tau)."""
        import torch

        from nemo_rl.experience.rollout_manager import ROLLOUT_TARGET_STEP

        if stop_strings:
            raise RuntimeError("[xspr] stop strings are not supported (validator)")
        target = ROLLOUT_TARGET_STEP.get()
        if target is None:
            raise RuntimeError("[xspr] a row without a target step (in_order sampler)")
        idx = _plain(input_sample["idx"])
        key = self.key_of(target, idx, traj_idx)
        self.ledger.open(key)
        cert = RowCert(version=-1, key=key, tokens=0)
        msg, in_len, metrics, info, label = await self._segment(impl, message_log, native, key, [], [])
        tokens = [int(t) for t in msg["token_ids"].tolist()]
        lps = [float(x) for x in msg["generation_logprobs"].tolist()]
        self.ledger.draw(key, 0, len(tokens))
        done, seg_start = self._finished(info, tokens), 0
        while not (done and label == target):
            if done:
                if self.neg in (NEG_SKIP_CHECK, NEG_SKIP_FINISHED):
                    break
                await self.wait_version(target)
            n = len(tokens)
            if n == 0 or (self.neg == NEG_SKIP_CHECK and not done):
                # nothing to check (cut before its first token), or the mutation: decode on from the end
                msg, in_len, metrics, info, v = await self._segment(impl, message_log, native, key, tokens, lps)
                tokens = [int(t) for t in msg["token_ids"].tolist()]
                lps = [float(x) for x in msg["generation_logprobs"].tolist()]
                self.ledger.draw(key, n, len(tokens))
                done, label, seg_start = self._finished(info, tokens), (v if n == 0 else min(label, v)), n
                continue
            cmsg, in_len, metrics_c, cinfo, checked = await self._segment(
                impl, message_log, native, key, tokens, lps, check="only" if done else "continue")
            if cinfo["check"] is None:
                continue  # cut mid-prefill by a wake: nothing checked, redone under the new weights
            label = checked
            chk = cinfo["check"]
            tau, y = int(chk["tau"]), int(chk["y"])
            all_p = [float(x) for x in chk["lps"]]
            chk_lps = all_p[:tau]
            cert.checks += 1
            cert.taus += (tau,)
            cert.check_prefill += int(in_len) + n
            if cert.pre_check_len < 0 and done:  # a finished row's length before its check (EOS moves)
                cert.pre_check_len = n
            if self.neg == NEG_NEW_ONLY and tau < seg_start:
                tau, chk_lps = n, lps  # the tokens before the last segment taken as checked
            if self.neg == NEG_KEEP_LP:
                chk_lps = lps[:tau]
            if tau == n:
                if done:
                    lps = chk_lps
                    continue
                # the check request decoded on from the end of the row
                new = [int(t) for t in cmsg["token_ids"].tolist()]
                lps = chk_lps + [float(x) for x in cmsg["generation_logprobs"].tolist()][n:]
                self.ledger.draw(key, n, len(new))
                tokens, msg, metrics, seg_start = new, cmsg, metrics_c, n
                done = self._finished(cinfo, tokens)
                continue
            cert.redecoded += n - tau
            cert.mism += (tau,)
            if self.keep_forks:
                cert.forks.append((tau, tokens[tau:], lps, all_p))
            dmax = max((abs(a - b) for a, b in zip(all_p[:tau], lps[:tau])), default=0.0)
            self.ledger.permit(key, tau)
            prefix, plps, draw_key = tokens[:tau], chk_lps[:tau], key
            if self.neg == NEG_FRESH_KEYS:
                draw_key = row_key(self.seed, self.stream + "/fresh", target, idx, traj_idx)
            if self.neg == NEG_STATE_PREFIX_END and n < self.max_new:
                prefix, plps = tokens, lps
            msg, in_len, metrics, info, v = await self._segment(impl, message_log, native, draw_key, prefix, plps)
            cert.cont_prefill += int(in_len) + len(prefix)
            new = [int(t) for t in msg["token_ids"].tolist()]
            new_lps = [float(x) for x in msg["generation_logprobs"].tolist()]
            if self.neg == NEG_STATE_PREFIX_END and prefix is tokens:
                # resumed from the end of the old prefix, its tokens spliced in at tau
                new, new_lps = tokens[:tau] + new[n:], chk_lps[:tau] + new_lps[n:]
            elif draw_key == key and len(new) > tau and new[tau] != y:
                raise RuntimeError(f"[xspr] key {key}: the continuation's first token {new[tau]} != the check's "
                                   f"y_tau {y} at tau {tau}: engine prefill and decode disagree (P1)")
            self.ledger.draw(key, tau, len(new))
            cert.flips.append((lps[tau], all_p[tau], new_lps[tau] if len(new_lps) > tau else float("nan"), dmax))
            tokens, lps, seg_start = new, new_lps, tau
            done, label = self._finished(info, tokens), min(label, v)
        self.ledger.close(key)
        out = dict(msg)
        out["token_ids"] = torch.tensor(tokens, dtype=msg["token_ids"].dtype)
        out["generation_logprobs"] = torch.tensor(lps, dtype=torch.float32)
        out["content"] = impl._tokenizer.decode(out["token_ids"], skip_special_tokens=True)
        cert.version, cert.tokens = int(label), len(tokens)
        self.certs[(int(target), idx, int(traj_idx))] = cert
        return out, in_len, metrics

    def pop_cert(self, target: int, prompt_idx: Any, j: int) -> RowCert:
        c = self.certs.pop((int(target), _plain(prompt_idx), int(j)), None)
        if c is None:
            raise RuntimeError(f"[xspr] no certificate for row ({target}, {prompt_idx}, {j})")
        return c


# tau histogram bins (FIRSTPRINCIPLES_B Theorem F: early forks at tau < 16 / < 128)
TAU_BINS = ((0, 16), (16, 128), (128, 1024), (1024, 4096), (4096, 1 << 62))


def tau_stats(certs: list[RowCert]) -> dict[str, float]:
    """Per-step stats of the checks: rows checked / mismatched, tau quantiles over the mismatches, re-decoded
    tokens, check / continuation prefill tokens, rows whose EOS moved earlier / later than before their check."""
    import numpy as np

    red = np.array([t for c in certs for t in c.mism] or [-1])
    last = np.array([c.mism[-1] for c in certs if c.mism] or [-1])  # the mismatch each re-decoded row resumed from
    fl = np.array([f for c in certs for f in c.flips] or [[np.nan] * 4], dtype=np.float64)
    checked = [c for c in certs if c.checks]
    fin = [c for c in checked if c.redecoded and c.pre_check_len >= 0]  # re-decoded rows that had finished
    return {
        "rows": float(len(certs)),
        "checked": float(len(checked)),
        "checks": float(sum(c.checks for c in certs)),
        "mismatched": float(sum(1 for c in certs if c.redecoded > 0)),
        "redecoded": float(sum(c.redecoded for c in certs)),
        "check_prefill": float(sum(c.check_prefill for c in certs)),
        "cont_prefill": float(sum(c.cont_prefill for c in certs)),
        "eos_earlier": float(sum(1 for c in fin if c.tokens < c.pre_check_len)),
        "eos_later": float(sum(1 for c in fin if c.tokens > c.pre_check_len)),
        "tau_min": float(red.min()),
        "tau_p10": float(np.percentile(red, 10)),
        "tau_p50": float(np.percentile(red, 50)),
        "tau_hist": [int(((last >= a) & (last < b)).sum()) for a, b in TAU_BINS],
        # at the mismatches (medians): log q(x_tau), log p(x_tau), log p(y_tau), max |log p - log q| before tau
        "lq_x": float(np.nanmedian(fl[:, 0])),
        "lp_x": float(np.nanmedian(fl[:, 1])),
        "lp_y": float(np.nanmedian(fl[:, 2])),
        "dlp_max": float(np.nanmedian(fl[:, 3])),
    }


def require_finite(lps: list, what: str, key: int, first: int) -> None:
    """Raise on the first non-finite log-prob, naming the row key and the absolute token position."""
    import math

    bad = next((j for j, x in enumerate(lps) if not math.isfinite(x)), None)
    if bad is not None:
        raise RuntimeError(f"[xspr] non-finite {what} log-prob {lps[bad]!r} at token position {first + bad} of row key "
                           f"{key} ({len(lps)} {what} tokens from position {first})")


def _plain(idx: Any) -> Any:
    """A prompt index as a plain Python value (tensors / numpy scalars -> int)."""
    try:
        return int(idx)
    except (TypeError, ValueError):
        return str(idx)


def negative() -> str:
    v = os.environ.get("THUNDERSYNC_NL_NEG", "").strip()
    return v if v in NEGATIVES else ""
