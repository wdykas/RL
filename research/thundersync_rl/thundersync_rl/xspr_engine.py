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
"""Engine side of exact SPR (``thundersync.mode=xspr``; docs/colo/THEORY.md).

Generation workers only. ``XsprEngineMixin.xspr_setup`` installs, on the colocated engine:

- row keys: every xspr request carries its row's noise key id (``KEY_RID_ATTR``); the keyed sampler draws token t of
  the row with noise g(row key, t), t the absolute position, whatever engine request carries the row (the wide
  kernel reads the key through ``_request_keys``);
- the check: a request tagged ``CHECK_ATTR = L`` holds the row's L candidate tokens x as the end of its prompt. In
  its prefill forward the engine takes the head-input rows that predict x_0 .. x_{L-1} (the verify capture's
  static copy, ``_install_verify_capture``), computes their logits with the engine's own head GEMM and vocab gather,
  and per row j: y_j = the keyed sampler S on that logits row at (row key, position of x_j), and log p(x_j) by the
  engine's raw log-prob function. The first j with y_j != x_j is tau (L if none). The result (tau, log p(x_0 ..
  x_{L-1}), y_tau) rides back on the request's sampling params (``RESULT_ATTR``). Logits exist only for
  ``chunk_rows`` positions at a time. A check request either generates one discarded token, or (``CONTINUE_ATTR``)
  decodes on from the end of its prefill, with the same key, unless tau < L: then the engine stops it;
- the cut at the wake: before a colocated wake resumes the engine, every xspr request that has run a forward
  (resident, not waiting) is marked to finish at its next step without a new token (``CUT_ATTR``, the stop-word
  path of ``nolanes_engine.mark_cut``), so a paused row returns its tokens and is checked under the new weights
  instead of decoding on its stale prefix;
- the eviction guard: an engine eviction of any request raises (D5).
"""

from __future__ import annotations

import itertools
import time
from typing import Any, Callable, Optional

import numpy as np

from thundersync_rl.nolanes_engine import CUT_ATTR, KEY_RID_ATTR, normalize_empty_results

# SamplingParams attributes (set on rank 0 from the generation data, or on the engine rank).
CHECK_ATTR = "ts_xspr_check"  # L: the last L prompt tokens are the row's candidate tokens to check
CONTINUE_ATTR = "ts_xspr_continue"  # the check request decodes on after its prefill: 1 unless a token mismatched, 2 always
RESULT_ATTR = "ts_xspr_result"  # {"tau", "lps", "y"} of a check request (engine rank)
NOCACHE_ATTR = "ts_xspr_nocache"  # the request matches no prefix-cache block (an independent reference prefill)


def check_rows(
    rows,
    x,
    key_id: int,
    first_pos: int,
    temperature: float,
    logits_fn: Callable,
    sample_fn: Callable,
    logprob_fn: Callable,
    chunk_rows: int,
) -> tuple[int, list[float], int]:
    """(tau, log p(x_0 .. x_{n-1}), y_tau or -1) of one row's n candidate tokens (log-probs of all n: rescoring
    users need them past tau).

    ``rows`` [L, d]: head inputs; row j predicts x_j, the token at absolute position ``first_pos + j``.
    ``logits_fn(h)`` -> full-vocabulary logits rows; ``sample_fn(z, keys, pos, temp)`` -> the keyed sampler's tokens;
    ``logprob_fn(z)`` -> log-prob rows. ``chunk_rows`` logits rows exist at a time. One device sync.
    """
    import torch

    n = int(x.shape[0])
    if int(rows.shape[0]) != n:
        raise RuntimeError(f"[xspr engine] check: {rows.shape[0]} head-input rows for {n} candidate tokens")
    if n == 0:
        return 0, [], -1
    dev = rows.device
    ys, lps = [], []
    for a in range(0, n, chunk_rows):
        b = min(n, a + chunk_rows)
        z = logits_fn(rows[a:b])
        keys = torch.full((b - a,), int(key_id), dtype=torch.int64, device=dev)
        pos = torch.arange(first_pos + a, first_pos + b, dtype=torch.int64, device=dev)
        temp = torch.full((b - a,), float(temperature), dtype=torch.float32, device=dev)
        ys.append(sample_fn(z, keys, pos, temp))
        lps.append(logprob_fn(z).gather(-1, x[a:b].view(-1, 1).to(dev))[:, 0].float())
        del z
    y = torch.cat(ys)
    lp = torch.cat(lps)
    bad = torch.nonzero(y != x.to(dev), as_tuple=False)
    tau = int(bad[0, 0]) if bad.numel() else n
    lp_host = lp.cpu().tolist()
    y_tau = int(y[tau]) if tau < n else -1
    return tau, lp_host, y_tau


def nonfinite_dump(worker: Any, engine: Any, logits, new_tokens, selected, dump_dir: str) -> None:
    """A sampled token's log-prob is non-finite: per bad row, the request id, row key, absolute position, sampled id,
    its log-prob, the row's max logit / LSE / NaN and inf counts; torch.save of the full fp32 logits row and the
    request's last head-input row (the verify capture) to ``dump_dir``; then raise."""
    import os

    import torch

    from thundersync_rl.worker import _request_keys, _request_positions

    ctx = engine.context
    lo = int(ctx.paused_request_count)
    n = int(new_tokens.shape[0])
    rids = ctx.request_ids[lo: lo + n].to(torch.int64).numpy()
    lens = worker._request_lengths()
    pos = _request_positions(engine, rids, lens)
    keys = _request_keys(engine, rids, lens)
    rows = logits.squeeze(0)
    bad = torch.nonzero(~torch.isfinite(selected), as_tuple=False).flatten().tolist()
    static = (getattr(worker, "_verify_cap", None) or {}).get("static")
    tok_req = ctx.token_to_request_idx[: int(ctx.active_token_count)].numpy()
    lines = []
    for i in bad[:8]:
        z = rows[i].float()
        rec = {"rid": int(rids[i]), "key": int(keys[i]), "pos": int(pos[i]), "token": int(new_tokens[i]),
               "lp": float(selected[i]), "max_logit": float(z.max()), "lse": float(torch.logsumexp(z, -1)),
               "nan": int(torch.isnan(z).sum()), "inf": int(torch.isinf(z).sum()),
               "rank": int(getattr(worker, "rank", -1))}
        lines.append(rec)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            t = np.flatnonzero(tok_req == lo + i)
            head = static[int(t[-1])].float().cpu() if static is not None and t.size else None
            torch.save({**rec, "logits": z.cpu(), "head_input": head},
                       os.path.join(dump_dir, f"nonfinite_r{rec['rank']}_rid{rec['rid']}_pos{rec['pos']}.pt"))
    print(f"[xspr engine] NON-FINITE sampled log-prob: {len(bad)} of {n} rows {lines} dump_dir={dump_dir!r}",
          flush=True)
    raise RuntimeError(f"[xspr engine] non-finite log-prob of a sampled token: {lines[:2]}")


_PRIVATE_HASHES = itertools.count()


def check_cache_blocks(hashes: list, prompt_len: int, n_check: int, block: int, nocache: bool) -> list:
    """The block hashes of a request, with the blocks it must compute itself made private. A check request (its last
    ``n_check`` prompt tokens are the candidates) keeps the real hashes only of blocks that end before position
    prompt_len - n_check - 1; the head-input rows of every candidate (positions first - 1 .. end - 2) are then computed
    by this prefill, never read from a cached block. Shared prompt blocks still match (within a weight epoch: MCore
    salts the cache by the epoch at every resume, so a wake's recompute never matches an older version's KV).
    ``nocache``: every block private (an independent reference). Private blocks get fresh unique hashes rather than
    none: MCore registers every complete prompt block by its hash (its chain needs one per block), and a hash no
    other request computes is never matched."""
    keep = 0 if nocache else (len(hashes) if not n_check else max(0, (prompt_len - n_check - 1) // block))
    keep = min(keep, len(hashes))
    return list(hashes[:keep]) + [(1 << 62) | next(_PRIVATE_HASHES) for _ in range(len(hashes) - keep)]


_CACHE_RULE: dict[str, Any] = {"engine": None, "stale": False, "max_blocks": 0, "max_shared": 0}


def install_check_cache_rule(engine: Any, stale_cache: bool) -> None:
    """Apply ``check_cache_blocks`` wherever MCore (re)computes a request's block hashes (creation and the resume
    re-salt; one engine per process). ``stale_cache`` (negative control isspr_stale_cache): check requests are salted
    with the previous weight epoch, so their prompt blocks match the older version's KV."""
    from megatron.core.inference.engines import dynamic_engine as de
    from megatron.core.inference.inference_request import DynamicInferenceRequest as R

    _CACHE_RULE.update(engine=engine, stale=bool(stale_cache))
    if getattr(R._compute_block_hashes, "_ts_check_rule", False):
        return
    orig = R._compute_block_hashes

    def compute(self) -> None:
        sp = self.sampling_params
        n = int(getattr(sp, CHECK_ATTR, 0) or 0)
        eng = _CACHE_RULE["engine"]
        if _CACHE_RULE["stale"] and n and eng is not None and eng._weight_epoch > 0:
            self.block_hash_salt = de._weight_scoped_salt(eng._weight_epoch - 1, None)
        orig(self)
        full = list(self.precomputed_block_hashes)
        self.precomputed_block_hashes = check_cache_blocks(
            full, len(self.prompt_tokens), n, int(self.block_size_tokens), bool(getattr(sp, NOCACHE_ATTR, 0)))
        if n:  # non-vacuity: the most complete / shareable blocks a check request had since the last stats call
            shared = sum(1 for a, b in zip(full, self.precomputed_block_hashes) if a == b)
            _CACHE_RULE["max_blocks"] = max(_CACHE_RULE["max_blocks"], len(full))
            _CACHE_RULE["max_shared"] = max(_CACHE_RULE["max_shared"], shared)

    compute._ts_check_rule = True
    R._compute_block_hashes = compute


class CheckAccumulator:
    """A check request's result over the prefill forwards that bring its n candidate rows (one forward without
    chunked prefill; several, in position order, with it): log-probs of all n, the first mismatch tau over all of
    them and its y. Bitwise the one-forward result when the chunks' head-input rows are (BI chunked prefill)."""

    def __init__(self, n: int) -> None:
        self.n = int(n)
        self.done = 0
        self.parts = 0  # prefill forwards that brought rows (> 1: a chunked check)
        self.lps: list[float] = []
        self.tau = self.n
        self.y = -1

    def add(self, j0: int, tau: int, lps: list[float], y: int) -> bool:
        """Rows j0 .. j0 + len(lps) - 1 (their local tau / y); True once all n rows are in."""
        if j0 != self.done:
            raise RuntimeError(f"[xspr engine] check rows from {j0}, expected {self.done} (out of order or repeated)")
        if self.tau == self.n and tau < len(lps):
            self.tau, self.y = j0 + tau, y
        self.lps += lps
        self.done += len(lps)
        self.parts += 1
        if self.done > self.n:
            raise RuntimeError(f"[xspr engine] {self.done} check rows for {self.n} candidate tokens")
        return self.done == self.n

    def result(self) -> tuple[int, list[float], int]:
        return self.tau, self.lps, self.y


def occupancy(occ: list[tuple[float, int]], tail: float = 0.25) -> dict[str, float]:
    """Engine occupancy over the forwards ``occ`` [(time, active rows)]: mean active rows over all of them and over
    the last ``tail`` of their time span (the end of a phase, where a re-decode tail would show), ms per forward."""
    if len(occ) < 2:
        return {}
    t0, t1 = occ[0][0], occ[-1][0]
    cut = t1 - tail * (t1 - t0)
    late = [a for t, a in occ if t >= cut]
    return {"occ_rows": sum(a for _, a in occ) / len(occ), "occ_tail_rows": sum(late) / len(late),
            "ms_per_iter": 1000.0 * (t1 - t0) / (len(occ) - 1)}


def check_capture(overflow_before: int, overflow_after: int) -> None:
    """A forward that holds check rows must have copied its head-input rows: the verify capture skips the copy when
    a forward has more rows than its static buffer (overflow count), and the check would then read stale rows."""
    if overflow_after != overflow_before:
        raise RuntimeError("[xspr engine] a forward with check rows overflowed the head-input capture (rows > "
                           "max_tokens + 1024): its rows were not copied")


def record_failure(xs: dict, err: Exception) -> None:
    """A guard failed on the engine loop thread: keep the first failure for raise_failure (printing it once)."""
    if "error" not in xs:
        xs["error"] = f"{type(err).__name__}: {err}"
        print(f"[xspr engine] FAILURE (raised on the worker's next result / stats / wake call): {err}", flush=True)


def raise_failure(xs: Optional[dict]) -> None:
    """Raise a failure the engine loop recorded (record_failure), on a worker call that reaches the driver."""
    if xs is not None and "error" in xs:
        raise RuntimeError(xs["error"])


def prefill_one_epoch(seen: dict, rids, epoch: int, is_cut: Callable[[int], bool]) -> None:
    """Every forward of one request's prefill runs on one weight version, unless the wake cut it (its result is void
    and it ends without a token): a request seen in prefill under two weight epochs and not cut had KV (and check
    rows) from both (a chunked prefill the cut missed). Raises."""
    for rid in rids.tolist():
        first = seen.setdefault(int(rid), epoch)
        if first != epoch and not is_cut(int(rid)):
            raise RuntimeError(f"[xspr engine] request {int(rid)}: its prefill ran under weight epochs {first} and "
                               f"{epoch} (a chunked prefill across a wake that the cut missed)")


def cut_touched(engine: Any, stats: Optional[dict] = None, prefill_cut: bool = True) -> int:
    """Mark every xspr request of a suspended engine that has run a forward to finish at its next step, its tokens
    as they stand (the stop-word path: that step's sampled token is not appended). Requests still waiting for their
    first forward are left alone; a request part-way through a chunked prefill (MCore keeps it at the head of the
    waiting queue) has run forwards and is cut too: its remaining chunks would run on the new weights over KV of the
    old ones (MCore finishes it, without a token, at its last chunk). Returns the number marked; ``stats["midprefill"]``
    counts those cut part-way through a prefill. ``prefill_cut`` False (negative control isspr_no_prefill_cut):
    those are skipped, as before 4389aae5."""
    waiting = {int(r) for r in getattr(engine, "resume_request_ids", None) or ()}
    waiting |= {int(r) for r in engine.waiting_request_ids}
    n = 0
    for rid, entry in list(engine.requests.items()):
        sp = entry.record[-1].sampling_params
        started = int(getattr(entry.record[-1], "finished_chunk_token_count", 0) or 0) > 0
        mid = int(rid) in waiting and started
        if sp is None or getattr(sp, KEY_RID_ATTR, None) is None:
            continue
        if int(rid) in waiting and not (started and prefill_cut):
            continue
        if getattr(sp, CUT_ATTR, 0):
            continue
        for r in entry.record.requests:
            if getattr(r, "sampling_params", None) is not None:
                r.sampling_params.add_attributes({CUT_ATTR: 1})
        engine.stop_word_finished_request_ids.add(int(rid))
        n += 1
        if stats is not None and mid:
            stats["midprefill"] = stats.get("midprefill", 0) + 1
    return n


class _CheckTags:
    """Context slot -> check request (key id into ``meta``), refreshed per forward (``routes.SlotTags``)."""

    def __init__(self, engine: Any) -> None:
        from thundersync_rl.routes import SlotTags

        self.engine = engine
        self.meta: list[Optional[dict[str, Any]]] = []
        self.by_rid: dict[int, int] = {}
        self.free: list[int] = []
        self.lo = np.zeros(0, dtype=np.int64)
        self.hi = np.zeros(0, dtype=np.int64)
        self.tags = SlotTags(self._lookup, lambda k: k)

    def _lookup(self, rid: int) -> Optional[int]:
        import torch

        entry = self.engine.requests.get(rid)
        if entry is None:
            return None
        req = entry.record[-1]
        sp = req.sampling_params
        n = getattr(sp, CHECK_ATTR, None)
        if not n:
            return None
        if len(entry.record.requests) != 1:
            raise RuntimeError(f"[xspr engine] check request {rid} was checkpointed (evicted): D5")
        prompt = req.prompt_tokens
        total = int(len(prompt))
        n = int(n)
        kid = self.by_rid.get(rid)
        if kid is None:
            kid = self.free.pop() if self.free else len(self.meta)
            self.by_rid[rid] = kid
            if kid == len(self.meta):
                self.meta.append(None)
            self.meta[kid] = ({
                "rid": int(rid), "sp": sp, "entry": entry, "first": total - n, "n": n,
                "key": int(getattr(sp, KEY_RID_ATTR)), "temp": float(sp.temperature),
                "x": torch.as_tensor(prompt[total - n:]).to("cpu", copy=True).long(),
                "cont": int(getattr(sp, CONTINUE_ATTR, 0) or 0),
                "acc": CheckAccumulator(n),
            })
            if kid >= self.lo.shape[0]:
                grow = max(256, kid + 1 - self.lo.shape[0])
                self.lo = np.concatenate([self.lo, np.zeros(grow, dtype=np.int64)])
                self.hi = np.concatenate([self.hi, np.zeros(grow, dtype=np.int64)])
            # Rows at positions first-1 .. total-2 predict x_0 .. x_{n-1}.
            self.lo[kid], self.hi[kid] = total - n - 1, total - 1
        return kid

    def prune(self) -> int:
        """Forget the check requests the engine no longer holds (finished); their ids are reused. Requests still
        waiting for their prefill keep their entries. Returns the number pruned."""
        gone = [rid for rid in self.by_rid if rid not in self.engine.requests]
        for rid in gone:
            kid = self.by_rid.pop(rid)
            self.meta[kid] = None
            self.free.append(kid)
        return len(gone)


class XsprEngineMixin:
    """Mixed into ThunderSyncMegatronPolicyWorker (before the stock worker)."""

    def xspr_setup(self, chunk_rows: int, wake_cut: bool = True, dump_dir: str = "", stale_cache: bool = False,
                   prefill_cut: bool = True) -> Any:
        """Install the check, the row keys and (``wake_cut``) the wake cut (idempotent). Needs the wide sampler
        (installed first) and the verify capture (built with the engine: ``policy.megatron_cfg.engine_verify``).
        Without the wake cut, paused requests resume on their stale KV as in stock SPR. Also the sampled-token
        log-prob guard (``dump_dir``: where a non-finite one is dumped before raising). Returns True on ranks with
        an engine."""
        engine = getattr(self, "dynamic_inference_engine", None)
        if engine is None:
            return False
        if self.__dict__.get("_xspr") is not None:
            return True
        sampling = engine.controller._sampling
        if not getattr(sampling, "_wide_installed", False):
            raise RuntimeError("[xspr engine] install_wide_sampler first (the check and decode share its keys)")
        if getattr(self, "_verify_cap", None) is None:
            raise RuntimeError("[xspr engine] no verify capture on this engine (policy.megatron_cfg.engine_verify)")
        self._xspr = {
            "chunk_rows": int(chunk_rows),
            "wake_cut": bool(wake_cut),
            "checks": 0,
            "check_rows": 0,
            "check_s": 0.0,
            "check_chunked": 0,
            "check_void": 0,
            "cut": 0,
            "cut_stats": {},  # cut_touched: requests cut part-way through a prefill ("midprefill")
            "prefill_cut": bool(prefill_cut),
            "prefill_epoch": {},  # request id -> the weight epoch of its prefill forwards (one epoch per prefill)
            "evicted0": int(getattr(engine, "evicted_request_count", 0)),
            "occ": [],  # (time, active rows) of every forward since the last stats call
            "head": {},
        }
        self._xspr_wrap_forward(engine)
        install_check_cache_rule(engine, stale_cache)
        from thundersync_rl import nolanes

        if nolanes.checks_on():
            # A host sync per engine step: CHECKS runs only, never timed pairs (REVIEW batch 27, N1). Timed runs keep
            # the free driver-side check (ExactRows._segment: require_finite).
            self._xspr_guard_logprobs(engine, dump_dir)
        print(f"[xspr engine] rank={getattr(self, 'rank', '?')} check + row keys + wake cut installed "
              f"(chunk_rows={chunk_rows})", flush=True)
        return True

    def _xspr_guard_logprobs(self, engine: Any, dump_dir: str) -> None:
        """Every sampled token's engine log-prob must be finite: the sampler cannot draw a token the logged law gives
        probability 0 (a NaN / -inf there means the log-prob path and the sampler disagree). One reduction and host
        sync per step (CHECKS only); on a non-finite one, dump the row(s) with their logits and raise
        (``nonfinite_dump``)."""
        import torch

        ctx = engine.context
        orig = ctx.calculate_log_probs_tensors

        def calculate_log_probs_tensors(logits, new_tokens, *args, **kwargs):
            selected, log_probs = orig(logits, new_tokens, *args, **kwargs)
            n = int(new_tokens.shape[0])
            if selected.shape[0] == n and not bool(torch.isfinite(selected).all()):
                try:
                    nonfinite_dump(self, engine, logits, new_tokens, selected, dump_dir)
                except Exception as err:  # the engine loop thread: recorded, raised on the worker path
                    record_failure(self._xspr, err)
            return selected, log_probs

        ctx.calculate_log_probs_tensors = calculate_log_probs_tensors

    # -- hooks called from worker.py ------------------------------------------------------------------------

    def _xspr_on(self) -> bool:
        return self.__dict__.get("_xspr") is not None

    def _xspr_request_attrs(self, data: Any, sps: list) -> None:
        """Rank 0 (it builds the requests): the row key of every xspr request; a check request generates one token
        (discarded) after the prefill that checks its candidate tokens."""
        if data is None or "xspr_key" not in data:
            return
        keys = data["xspr_key"]
        checks = data["xspr_check"] if "xspr_check" in data else None
        conts = data["xspr_continue"] if "xspr_continue" in data else None
        nocache = data["xspr_nocache"] if "xspr_nocache" in data else None
        for i, sp in enumerate(sps):
            attrs = {KEY_RID_ATTR: int(keys[i])}
            if nocache is not None and int(nocache[i]):
                attrs[NOCACHE_ATTR] = 1
            n = int(checks[i]) if checks is not None else 0
            if n > 0:
                attrs[CHECK_ATTR] = n
                if conts is not None and int(conts[i]):
                    attrs[CONTINUE_ATTR] = int(conts[i])
                else:
                    sp.num_tokens_to_generate = 1
            sp.add_attributes(attrs)

    def prepare_for_generation(self, *args, **kwargs):
        """Colocated wake: cut every xspr request that ran a forward before the stand-down (engine still suspended,
        before the wake's reshard), then wake."""
        xs = self.__dict__.get("_xspr")
        raise_failure(xs)
        engine = getattr(self, "dynamic_inference_engine", None)
        if (xs is not None and xs["wake_cut"] and engine is not None and getattr(self, "is_generation_colocated", False)
                and getattr(self, "_inference_engine_asleep", False)):
            import asyncio

            import torch

            async def mark() -> int:
                return cut_touched(engine, xs["cut_stats"], xs["prefill_cut"])

            xs["cut"] += asyncio.run_coroutine_threadsafe(mark(), self._inference_loop).result()
            torch.distributed.barrier()
        if xs is not None:
            xs["head"].clear()
        return super().prepare_for_generation(*args, **kwargs)

    def xspr_cut_now(self) -> int:
        """Cut every xspr request that has run a forward, with the engine awake (CHECKS gate tool: the stage cuts
        the lookahead rows before its from-scratch replays, so the cut rows' check / resume path runs in a gate
        whose replays would otherwise outlast every lookahead row). All engine ranks pause through the coordinator,
        mark, meet at a barrier, unpause (``nl_cut_drafts``' order). Returns this rank's count."""
        xs = self.__dict__.get("_xspr")
        engine = getattr(self, "dynamic_inference_engine", None)
        if xs is None or engine is None or getattr(self, "_inference_engine_asleep", False):
            return 0
        import asyncio

        import torch
        from megatron.core.inference.engines.dynamic_engine import EngineState

        async def pause():
            if torch.distributed.get_rank() == 0:
                await self._client_send(self.inference_client.pause_engines)
            await engine.wait_until(EngineState.PAUSED)

        async def mark() -> int:
            return cut_touched(engine, xs["cut_stats"], xs["prefill_cut"])

        async def unpause():
            if torch.distributed.get_rank() == 0:
                await self._client_send(self.inference_client.unpause_engines)
            await engine.wait_until(EngineState.RUNNING)

        loop = self._inference_loop
        asyncio.run_coroutine_threadsafe(pause(), loop).result()
        n = asyncio.run_coroutine_threadsafe(mark(), loop).result()
        torch.distributed.barrier()
        asyncio.run_coroutine_threadsafe(unpause(), loop).result()
        xs["cut"] += n
        return n

    def _parse_result_to_batched_data_dict(self, data, result):
        if self.__dict__.get("_xspr") is None:
            return super()._parse_result_to_batched_data_dict(data, result)
        raise_failure(self.__dict__.get("_xspr"))
        normalize_empty_results(result)
        out = super()._parse_result_to_batched_data_dict(data, result)
        out["xspr"] = [
            {"cut": int(getattr(r.sampling_params, CUT_ATTR, 0) or 0),
             "check": getattr(r.sampling_params, RESULT_ATTR, None)}
            for r in result
        ]
        return out

    def xspr_engine_stats(self) -> dict[str, Any]:
        """Since the last call: checks, checked positions, check seconds, rows cut at wakes, occupancy; evictions
        since setup."""
        xs = self.__dict__.get("_xspr")
        engine = getattr(self, "dynamic_inference_engine", None)
        if xs is None or engine is None:
            return {}
        raise_failure(xs)
        out = {k: xs[k] for k in ("checks", "check_rows", "check_s", "check_chunked", "check_void", "cut")}
        out["cut_midprefill"] = int(xs["cut_stats"].pop("midprefill", 0))
        out["evicted"] = int(getattr(engine, "evicted_request_count", 0)) - xs["evicted0"]
        out.update(occupancy(xs["occ"]))
        # Prefill tokens served from the prefix cache since the last call (non-vacuity of the cache rule and of the
        # stale-cache negative; MCore's accumulator may be reset by its own metrics, hence the guard).
        # With memory-only hybrid prefix caching (prefix_caching_mamba_gb null) no prefill token is skipped, but matched
        # blocks still hand their attention KV to the request: count both.
        for name, attr in (("pc_skipped_tok", "_prefill_tokens_skipped"), ("pc_matched_blocks",
                                                                           "_prefix_cache_blocks_matched")):
            cur = int(getattr(engine, attr, 0) or 0)
            last = xs.get(name + "0", 0)
            out[name] = cur - last if cur >= last else cur
            xs[name + "0"] = cur
        out["check_max_blocks"], out["check_max_shared"] = _CACHE_RULE["max_blocks"], _CACHE_RULE["max_shared"]
        _CACHE_RULE.update(max_blocks=0, max_shared=0)
        xs.update(checks=0, check_rows=0, check_s=0.0, check_chunked=0, check_void=0, cut=0, occ=[])
        return out

    # -- the check --------------------------------------------------------------------------------------------

    def _xspr_head(self):
        """(logits fn [k, d] -> full-vocabulary fp32 rows, logprob fn) of the engine's model: the engine's head GEMM
        (``fp16_gemm.head_matmul``, logit scaling), its rank-order vocab all-gather, its raw log-prob rows."""
        import torch

        from megatron.core import parallel_state as ps

        from nemo_rl.models.megatron.fp16_gemm import head_input_dtype, head_matmul
        from nemo_rl.models.megatron.zero_train_gen_mismatch import _all_gather_vocab

        from thundersync_rl.worker import _unwrap

        gpt = _unwrap(self._gen_model())
        layer = gpt.output_layer
        weight = (gpt.shared_embedding_or_output_weight()
                  if getattr(gpt, "share_embeddings_and_output_weights", False) else layer.weight)
        if weight.untyped_storage().size() == 0:
            raise RuntimeError("[xspr engine] the engine's LM-head weight has no storage (offloaded)")
        scale = getattr(gpt, "_scale_logits", None)
        dtype = head_input_dtype(weight.dtype)
        group = ps.get_tensor_model_parallel_group() if ps.get_tensor_model_parallel_world_size() > 1 else None
        ctx = self.dynamic_inference_engine.context
        width = int(self.dynamic_inference_engine.controller.vocab_size)

        def logits(h: torch.Tensor) -> torch.Tensor:
            z = head_matmul(h.to(device=weight.device, dtype=dtype), weight)
            if scale is not None:
                z = scale(z)
            if group is not None:
                z = _all_gather_vocab(z.contiguous(), group)
            if int(z.shape[-1]) < width:
                raise RuntimeError(f"[xspr engine] head rows of width {z.shape[-1]} < engine vocab {width}")
            return z[:, :width].float()

        def logprobs(z: torch.Tensor) -> torch.Tensor:
            # The engine's raw log-prob rows (DynamicInferenceContext._processed_log_probs, raw_logprobs mode).
            return ctx._processed_log_probs(z, 0, None, None).float()

        return logits, logprobs

    def _xspr_sample_fn(self):
        import torch

        from thundersync_rl import keyed_sampling_s as s_mod
        from thundersync_rl.worker import _min_sampling_temperature

        salt = int(self._wide_salt)
        lo = _min_sampling_temperature()
        ws = [None]

        def sample(z, keys, pos, temp):
            n, v = z.shape
            need = s_mod.workspace_bytes(n, v)
            if ws[0] is None or ws[0].numel() < need:
                ws[0] = torch.empty(max(need, 1), dtype=torch.uint8, device=z.device)
            out = torch.empty(n, dtype=torch.int64, device=z.device)
            s_mod.wide_sample(z.contiguous(), keys, pos, temp.clamp(min=lo), salt, out=out, workspace=ws[0])
            return out

        return sample

    def _xspr_wrap_forward(self, engine: Any) -> None:
        """After every engine forward, check the candidate tokens of the check requests it prefilled."""
        import torch

        from thundersync_rl.engine_verify import capture_selection

        xs = self._xspr
        cap = self._verify_cap
        static = cap["static"]
        controller = engine.controller
        ctx = engine.context
        tags = _CheckTags(engine)
        dummy = [False]
        orig_init = controller._dynamic_step_context_init
        orig = controller._dynamic_step_forward_logits
        head = xs["head"] = {}

        def is_cut(rid: int) -> bool:
            entry = engine.requests.get(rid)
            return entry is not None and bool(getattr(entry.record[-1].sampling_params, CUT_ATTR, 0))  # the engine head's fns, rebuilt after every wake (the reshard may move weights)

        def context_init(*args, **kwargs):
            dummy[0] = bool(kwargs.get("is_dummy_forward", False))
            return orig_init(*args, **kwargs)

        def select():
            """Before the forward: the D5 guard, the one-epoch-per-prefill guard, and the check rows it holds."""
            if int(getattr(engine, "evicted_request_count", 0)) != xs["evicted0"]:
                raise RuntimeError("[xspr engine] the engine evicted a request (D5: no evictions in exact SPR)")
            if dummy[0]:
                return None
            n = int(ctx.active_token_count)
            e = int(ctx.total_request_count)
            xs["occ"].append((time.perf_counter(), e - int(ctx.paused_request_count)))
            if n <= 0 or e <= 0:
                return None
            in_prefill = ctx.request_in_prefill_status_tensor[:e].numpy() != 0
            prefill_one_epoch(xs["prefill_epoch"], ctx.request_ids[:e].numpy()[in_prefill],
                              int(getattr(engine, "_weight_epoch", 0)), is_cut)
            slot_kid = tags.tags.refresh(ctx.request_ids[:e].numpy(), in_prefill)
            if not (slot_kid >= 0).any():
                return None
            sel, kids, pos_sel = capture_selection(
                ctx.token_to_request_idx[:n].numpy(), ctx.token_to_pos_ids[:n].numpy(), slot_kid, tags.lo, tags.hi)
            return (sel, kids, pos_sel) if sel.size else None

        def score(sel, kids, pos_sel, ovf0) -> None:
            """After the forward: the check rows' log-probs / keyed matches, accumulated per request (chunks)."""
            check_capture(ovf0, cap["overflow"][0])
            t0 = time.perf_counter()
            rows_all = static.index_select(0, torch.from_numpy(sel).pin_memory().to(static.device, non_blocking=True))
            if not head:
                head["logits"], head["logprobs"] = self._xspr_head()
                head["sample"] = self._xspr_sample_fn()
            with torch.no_grad():
                for kid in np.unique(kids):
                    m = np.flatnonzero(kids == kid)
                    meta = tags.meta[int(kid)]
                    # Rows predicting x_j sit at positions first - 1 + j; a chunked prefill brings them over
                    # several forwards, in order (CheckAccumulator).
                    j0 = int(pos_sel[m[0]]) - (meta["first"] - 1)
                    j1 = j0 + m.size
                    if not np.array_equal(pos_sel[m], np.arange(j0, j1) + meta["first"] - 1):
                        raise RuntimeError(f"[xspr engine] check request {meta['rid']}: rows not contiguous")
                    rows = rows_all.index_select(0, torch.from_numpy(m).to(static.device))
                    part = check_rows(
                        rows, meta["x"][j0:j1], meta["key"], meta["first"] + j0, meta["temp"],
                        head["logits"], head["sample"], head["logprobs"], xs["chunk_rows"])
                    if not meta["acc"].add(j0, *part):
                        continue
                    xs["check_chunked"] += int(meta["acc"].parts > 1)
                    if getattr(meta["sp"], CUT_ATTR, 0):
                        # Cut by a wake while its (chunked) prefill was part-way: its rows straddle two weight
                        # versions. No result; the driver redoes the check (MCore finishes a partial prefill
                        # only at its last chunk, so the remaining chunks ran on the new weights).
                        xs["check_void"] += 1
                        continue
                    tau, lps, y = meta["acc"].result()
                    res = {"tau": tau, "lps": lps, "y": y, "n": meta["n"]}
                    for r in meta["entry"].record.requests:
                        if getattr(r, "sampling_params", None) is not None:
                            r.sampling_params.add_attributes({RESULT_ATTR: res})
                    if meta["cont"] == 1 and tau < meta["n"]:
                        # a mismatch: the row resumes at tau as a new request; stop this one (the tokens it
                        # may still append are past tau and dropped by the driver)
                        engine.stop_word_finished_request_ids.add(meta["rid"])
                    xs["checks"] += 1
                    xs["check_rows"] += int(meta["n"])
            xs["check_s"] += time.perf_counter() - t0

        def forward_logits(*args, **kwargs):
            # The engine loop thread: a raise here stalls the run (MCore's coordinator loop dies and the driver waits),
            # so a failed guard is recorded (record_failure) and raised on the worker's result / stats / wake paths.
            # After a failure no check result is attached, so pending checks fail at the driver too.
            picked = None
            if "error" not in xs:
                try:
                    picked = select()
                except Exception as err:  # any error on the engine loop thread would kill it: record, raise later
                    record_failure(xs, err)
            ovf0 = cap["overflow"][0]
            out = orig(*args, **kwargs)
            if picked is not None:
                try:
                    score(*picked, ovf0)
                except Exception as err:
                    record_failure(xs, err)
            if len(tags.by_rid) > 1024:
                tags.prune()
            if len(xs["prefill_epoch"]) > 4096:
                live = engine.requests
                for rid in [r for r in xs["prefill_epoch"] if r not in live]:
                    del xs["prefill_epoch"][rid]
            return out

        controller._dynamic_step_context_init = context_init
        controller._dynamic_step_forward_logits = forward_logits

