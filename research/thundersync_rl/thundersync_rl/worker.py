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
"""Megatron policy worker with gradient-streaming (ThunderSyncRL) methods.

Builds on the worker's split train-step API (``begin_train_step`` /
``train_microbatch`` / ``finish_train_step``), which already accumulates
un-normalized gradients locally under ``model.no_sync()`` and defers the 1/N
normalization, the DP reduction, clipping and the optimizer step to a single
barrier. On top of that we use the DDP ``grad_data`` buffers as scratch: every
streamed chunk's backward lands there and is immediately moved into a
per-(group, reward) accumulator, so its advantage can be applied once the
group closes. At the barrier, the folded batch gradient is written back into
the grad buffers and the regular ``finish_train_step`` runs unchanged.
"""

import functools
import os
import time
from contextlib import nullcontext
from typing import Any, Optional

import ray
import torch

from nemo_rl.algorithms.grad_streaming import GradStreamingSpec, StreamBucket
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.data.multimodal_utils import attach_media_token_validity_mask
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron import train as megatron_train
from nemo_rl.models.megatron.data import get_microbatch_iterator
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)
from thundersync_rl.fused_train import (
    FusedGraph,
    FusedGraphStore,
    FusedOutcome,
    make_fused_model_forward,
    substitute_model_forward,
)

WORKER_FQN = "thundersync_rl.worker.ThunderSyncMegatronPolicyWorker"


def _model_in_eval(method):
    """Run ``method`` with ``self.model`` in eval mode.

    Learner: restore the previous mode afterwards, since verification runs while
    a streamed train step is open and leaving eval mode on would silently disable
    dropout for the rest of the step. Generation workers: leave it in eval mode;
    the inference engine runs on the same module concurrently and requires it.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "dynamic_inference_engine", None) is not None:
            self.model.eval()
            return method(self, *args, **kwargs)
        was_training = self.model.training
        self.model.eval()
        try:
            return method(self, *args, **kwargs)
        finally:
            self.model.train(was_training)

    return wrapper


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """The innermost module under DDP / Float16Module wrappers."""
    while hasattr(model, "module"):
        model = model.module
    return model


def _plain_floats(x: Any) -> list[float] | None:
    """Engine log-probs (list, tensor or a serialized ("tensor", data) pair) as floats."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.float().tolist()
    if isinstance(x, (list, tuple)) and len(x) == 2 and x[0] == "tensor":
        from megatron.core.inference.inference_request import deserialize_tensor

        return deserialize_tensor(x[1]).float().tolist()
    return [float(v) for v in x]


def group_pipeline_calls(
    items: list[dict[str, Any]], *, merge: bool
) -> list[list[int]]:
    """Group a dispatch's items into pipeline schedule calls (lists of indices).

    Under pipeline parallelism each ``forward_backward_func`` call pays a
    fill/drain bubble of ``P - 1`` micro-batch slots, so a rank's items run as
    few calls as their gradient destinations allow:

    - all final items (``group=None``: advantages in the loss) share one call;
      their gradient accumulates in the grad buffers (or is captured into the
      batch accumulator once, after the call);
    - the items of one open-group bucket ``(group, reward)`` share one call,
      captured into that bucket once;
    - an item with a ``fused_key`` (retained verification forward, PP = 1
      only) runs alone.

    Calls are ordered by their first item, and items keep their order within
    a call, so micro-batches run in dispatch order per destination. Only the
    order in which gradients are summed changes (fp32 rounding).
    ``merge=False`` returns one call per item (the previous behavior).
    """
    if not merge:
        return [[i] for i in range(len(items))]
    calls: list[list[int]] = []
    by_key: dict[Any, list[int]] = {}
    for i, item in enumerate(items):
        if item.get("fused_key") is not None:
            calls.append([i])
            continue
        key = (item["group"], None if item["group"] is None else item["reward"])
        if key not in by_key:
            by_key[key] = []
            calls.append(by_key[key])
        by_key[key].append(i)
    return calls


def apply_draft_frame(
    frame: dict[str, Any],
    tokens: list[int],
    logprobs: list[list[float] | None],
    i: int,
) -> bool:
    """Apply one streaming frame of draft ``i``; return True on the final frame.

    ``tokens`` is draft i's token list (updated in place). ``logprobs[i]`` holds
    the engine's log-prob of each token, or None once a frame did not carry one
    per new token (the driver then reports the scorer's log p, see
    ``SpeculativeGeneration._plan``).
    """
    if "partial" in frame:
        part = frame["partial"]
        new = list(part["new_tokens"])
        tokens.extend(new)
        lps = _plain_floats(part.get("new_log_probs"))
        if logprobs[i] is not None:
            if lps is not None and len(lps) == len(new):
                logprobs[i].extend(lps)
            else:
                logprobs[i] = None
        return False
    if "final" in frame:
        final = frame["final"]
        get = (
            final.get if isinstance(final, dict) else lambda k: getattr(final, k, None)
        )
        tokens[:] = list(get("generated_tokens"))
        lps = _plain_floats(get("generated_log_probs"))
        logprobs[i] = lps if lps is not None and len(lps) == len(tokens) else None
        return True
    return False


def snapshot_drafts(
    tokens: list[list[int]],
    logprobs: list[list[float] | None],
    done: list[bool],
) -> list[tuple[list[int], bool, list[float] | None]]:
    """(tokens, finished, engine logprobs or None) per draft, copied."""
    out = []
    for t, lp, d in zip(tokens, logprobs, done):
        ok = lp is not None and len(lp) == len(t)
        out.append((list(t), d, list(lp) if ok else None))
    return out


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker")
)  # pragma: no cover
class ThunderSyncMegatronPolicyWorker(MegatronPolicyWorkerImpl):
    """Driver-friendly wrappers over the core split API's streaming mode."""

    def stream_begin_step(
        self,
        loss_fn: LossFunction,
        *,
        gbs: int,
        mbs: int,
        storage_device: str,
        max_buckets_per_group: int,
        max_open_groups: int,
        merge_pipeline_calls: bool = True,
    ) -> None:
        """Open an optimizer step whose gradient arrives in streamed chunks.

        ``merge_pipeline_calls``: run all chunks of a dispatch that share a
        gradient destination as one pipeline schedule call (see
        :func:`group_pipeline_calls`); ``False`` runs one call per chunk.
        """
        # Retained verification forwards of this step (fused_verify_train).
        self._fused_graphs = FusedGraphStore()
        self.begin_train_step(
            loss_fn=loss_fn,
            gbs=gbs,
            mbs=mbs,
            grad_streaming=GradStreamingSpec(
                storage_device=storage_device,
                max_buckets_per_group=max_buckets_per_group,
                max_open_groups=max_open_groups,
            ),
        )
        self._stream_busy_s = 0.0
        self._stream_merge_calls = merge_pipeline_calls

    def _with_routes(self, data: BatchedDataDict) -> BatchedDataDict:
        """Give a batch ``routed_experts`` when sticky routing is on.

        Rows the driver gave anchors carry them; a batch without the field
        (no rollout of it was drafted) gets -1 everywhere: anchor = self,
        i.e. plain routing, which is the sticky policy of undrafted rows.
        """
        if (
            "routed_experts" in data
            or self._anchor_batch([], [], 0) is None
            or "input_ids" not in data
        ):
            return data
        b, width = data["input_ids"].shape[:2]
        data["routed_experts"] = self._anchor_batch([None] * b, None, width)
        return data

    def train_microbatches(self, datas, *args, **kwargs):
        return super().train_microbatches(
            [self._with_routes(d) for d in datas], *args, **kwargs
        )

    def get_logprobs(self, *, data, **kwargs):
        return super().get_logprobs(data=self._with_routes(data), **kwargs)

    def finish_train_step(self) -> dict[str, Any]:
        """Finish the train step; optionally perturb expert gradients (test only).

        ``THUNDERSYNC_DEBUG_EXPERT_GRAD_NOISE=<scale>`` adds seeded noise to the MoE
        expert gradients before each optimizer step, so expert weights change even when
        every reward is 0 (random tiny models). Exercises refit into the engine's expert
        weights: any stale engine weight shows up in the mismatch check.
        """
        scale = os.environ.get("THUNDERSYNC_DEBUG_EXPERT_GRAD_NOISE")
        if not scale:
            return super().finish_train_step()
        step = self.optimizer.step
        n_done = getattr(self, "_debug_noise_steps", 0)

        def noisy_step(*args, **kwargs):
            # Same seed and parameter order on every rank: replicas of an expert agree.
            g = torch.Generator(device="cuda").manual_seed(1234 + n_done)
            chunks = self.model if isinstance(self.model, list) else [self.model]
            for chunk in chunks:
                for name, p in chunk.named_parameters():
                    if (
                        ".experts." in name
                        and getattr(p, "main_grad", None) is not None
                    ):
                        p.main_grad.add_(
                            torch.randn(
                                p.main_grad.shape,
                                generator=g,
                                device=p.main_grad.device,
                            )
                            * float(scale)
                        )
            return step(*args, **kwargs)

        self.optimizer.step = noisy_step
        try:
            return super().finish_train_step()
        finally:
            self.optimizer.step = step
            self._debug_noise_steps = n_done + 1

    def _log_mem(self, where: str) -> None:
        if os.environ.get("THUNDERSYNC_MEM_LOG"):
            print(
                f"[thundersync mem] rank={self.rank} {where}: "
                f"alloc={torch.cuda.memory_allocated() / 2**30:.1f}G "
                f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}G",
                flush=True,
            )
            torch.cuda.reset_peak_memory_stats()

    def stream_accumulate(
        self,
        items: list[dict[str, Any]],
        closes: list[tuple[Any, dict[float, float]]],
    ) -> dict[str, Any]:
        """Backprop streamed chunks, then fold any groups that closed.

        Each item is ``{"data": BatchedDataDict, "group": gid | None,
        "reward": float | None, "fused_key": str | None}``; ``group=None`` items
        carry final advantages. An item with a ``fused_key`` is one row whose
        model forward is the retained verification forward of that key
        (fused_train.py); everything else about its training is unchanged.

        Items with the same gradient destination (all final items; the items
        of one (group, reward) bucket) run as one pipeline schedule call with
        all their micro-batches (:func:`group_pipeline_calls`), so under
        pipeline parallelism a dispatch pays one fill/drain bubble per
        destination instead of one per chunk.
        """
        t0 = time.perf_counter()
        fused_trained = fused_tokens = fused_fallback = 0
        state = self._train_step_state
        calls = group_pipeline_calls(
            items, merge=getattr(self, "_stream_merge_calls", True)
        )
        prof = os.environ.get("THUNDERSYNC_LEARNER_PROF")
        tprof = os.environ.get("THUNDERSYNC_LEARNER_TORCHPROF")
        for call in calls:
            call_items = [items[i] for i in call]
            first = call_items[0]
            bucket = (
                None
                if first["group"] is None
                else StreamBucket(group=first["group"], reward=first["reward"])
            )
            # Chunks smaller than a micro-batch run as one micro-batch of exactly
            # their rows (no dummy rows); larger chunks arrive padded to a
            # multiple of the configured size and use it. Keeping the configured
            # size bounds memory and matches the synchronous trainer's numerics
            # (Megatron fp32 gradients shift ~1e-4 with micro-batch size).
            step_mbs = state["mbs"]
            mbs_per_item = [min(it["data"].size, step_mbs) for it in call_items]
            if prof:
                torch.cuda.synchronize()
                t_call = time.perf_counter()
            for it in call_items:
                adv0 = it["data"].get("advantages")
                if adv0 is None or bool((adv0 != 0).any()):
                    self._n_items = getattr(self, "_n_items", 0) + 1
            fused_key = first.get("fused_key")
            assert fused_key is None or len(call_items) == 1
            graph = None if fused_key is None else self._fused_graphs.pop(fused_key)
            graph_tokens = 0 if graph is None else graph.num_tokens
            # A key without a graph (over the budget, or dropped) trains normally.
            outcome = FusedOutcome(
                used=False, fallback=fused_key is not None and graph is None
            )
            substitute = (
                nullcontext()
                if graph is None
                else substitute_model_forward(
                    megatron_train,
                    make_fused_model_forward(
                        graph,
                        megatron_train.model_forward,
                        outcome,
                        check=bool(os.environ.get("THUNDERSYNC_FUSED_CHECK")),
                    ),
                )
            )
            del graph  # only the substitute forward holds it now
            try:
                with substitute:
                    self._stream_train_call(call_items, mbs_per_item, bucket, tprof)
            finally:
                # Frees the graph's activations also when no forward ran (a
                # zero-advantage row skipped by the worker).
                del substitute
            fused_trained += outcome.used
            fused_tokens += graph_tokens if outcome.used else 0
            fused_fallback += outcome.fallback
            if prof:
                torch.cuda.synchronize()
                rows = sum(it["data"].size for it in call_items)
                nz = 0
                for it in call_items:
                    adv = it["data"].get("advantages")
                    nz += int((adv.abs().sum(-1) > 0).sum()) if adv is not None else 0
                shapes = [tuple(it["data"]["input_ids"].shape) for it in call_items]
                print(
                    f"[learner prof] rank={self.rank} chunks={len(call_items)} "
                    f"rows={rows} nonzero={nz} shapes={shapes} "
                    f"secs={time.perf_counter() - t_call:.4f}",
                    flush=True,
                )
        if closes:
            self.close_stream_groups(closes)
        torch.cuda.synchronize()
        self._log_mem("stream_accumulate")
        self._stream_busy_s += time.perf_counter() - t0
        acc = self._train_step_state["grad_stream"]
        return {
            "num_items": len(items),
            "num_calls": len(calls),
            "open_buffers": acc.num_open_buffers(),
            "fused_trained": fused_trained,
            "fused_tokens": fused_tokens,
            "fused_fallback": fused_fallback,
        }

    def _stream_train_call(
        self,
        call_items: list[dict[str, Any]],
        mbs_per_item: list[int],
        bucket: StreamBucket | None,
        tprof: str | None,
    ) -> None:
        """``train_microbatches`` on one call's items (optionally under the profiler)."""
        datas = [it["data"] for it in call_items]
        if (
            tprof
            and getattr(self, "_n_items", 0) >= int(tprof)
            and not getattr(self, "_tprof_done", False)
        ):
            self._tprof_done = True
            from torch.profiler import ProfilerActivity, profile

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as tp:
                self.train_microbatches(
                    datas, stream_bucket=bucket, micro_batch_sizes=mbs_per_item
                )
                torch.cuda.synchronize()
            print(
                f"[learner torchprof] shapes="
                f"{[tuple(d['input_ids'].shape) for d in datas]}\n"
                + tp.key_averages().table(
                    sort_by="cpu_time_total",
                    row_limit=30,
                    max_name_column_width=60,
                ),
                flush=True,
            )
        else:
            self.train_microbatches(
                datas, stream_bucket=bucket, micro_batch_sizes=mbs_per_item
            )

    def stream_finish_step(self) -> dict[str, Any]:
        """Barrier: write back, normalize, DP-reduce, clip, step."""
        peak = self._train_step_state["grad_stream"].peak_open_buffers
        # Retained verification graphs never outlive the weights they were
        # taken at; any left (their rollout trained the regular way) go now.
        unclaimed = self._fused_graphs.clear()
        results = self.finish_train_step()
        results["stream_peak_open_buffers"] = peak
        results["stream_learner_busy_s"] = self._stream_busy_s
        results["stream_fused_unclaimed"] = unclaimed
        return results

    def stream_abort_step(self) -> None:
        """Abort the open streamed step and drop its retained verification graphs."""
        self._fused_graphs = FusedGraphStore()
        self.abort_train_step()

    def get_flat_grads(self) -> torch.Tensor:
        """Return this rank's gradient buffers on CPU (for equivalence tests).

        Valid after a step with DP=1, where the buffers hold the full
        normalized gradient until the next step zeroes them.
        """
        buffers = self.model.buffers + self.model.expert_parallel_buffers
        return torch.cat([b.grad_data.detach().float().cpu() for b in buffers])

    def get_flat_params(self) -> dict[str, torch.Tensor]:
        """Return this rank's parameters on CPU (for equivalence tests)."""
        return {
            name: p.detach().float().cpu().clone()
            for name, p in self.model.named_parameters()
        }

    # ---- Exact cross-iteration speculative rollouts (keyed sampling) ----

    def install_lookup_references(self) -> None:
        """Generation workers: forward per-request ``lookup_references`` to the engine.

        Requests whose data carries ``lookup_references`` (a list of token
        sequences per row) get them on their SamplingParams, for the engine's
        exact lookup speculative decoding (``speculative_method="lookup"``).
        """
        if getattr(self, "_lookup_refs_installed", False):
            return
        inner = self._prepare_data_for_generation

        def prepare(data, greedy=False):
            prompts, mm, sps = inner(data, greedy)
            if "lookup_references" in data:
                for i, sp in enumerate(sps):
                    sp.lookup_references = [
                        list(r) for r in data["lookup_references"][i]
                    ]
            return prompts, mm, sps

        self._prepare_data_for_generation = prepare
        self._lookup_refs_installed = True

    def install_keyed_sampler(self, vocab_limit: int, head_k: int) -> int | None:
        """Generation workers: sample requests that carry ``noise_seed`` with keyed noise.

        The token at absolute position L of a request with seed s is
        ``keyed_sample(p, s, L)``, so a continuation request (prompt + accepted
        prefix) reproduces the uninterrupted decode exactly. Rows without a seed
        use the engine's own sampler. Also routes per-row ``noise_seed`` and
        ``max_new_tokens`` fields of the generation data into SamplingParams.
        """
        from thundersync_rl.keyed_sampling import keyed_sample_logits

        engine = getattr(self, "dynamic_inference_engine", None)
        if engine is None:
            return None
        eod = int(self.megatron_tokenizer.eod)
        sampling = engine.controller._sampling
        if getattr(sampling, "_keyed_installed", False):
            return eod
        orig_kernel = sampling.sample_kernel

        def keyed_kernel(
            logits,
            n,
            context,
            *,
            gather_indices=None,
            token_to_request_index=None,
            output=None,
            **kw,
        ):
            if token_to_request_index is not None:
                return orig_kernel(
                    logits,
                    n,
                    context,
                    gather_indices=gather_indices,
                    token_to_request_index=token_to_request_index,
                    output=output,
                    **kw,
                )
            lo, hi = context.paused_request_count, context.total_request_count
            # Key = (seed, index of the token being sampled). The context's
            # sequence length counts the current step's input token during
            # decode, so take the index from the request itself.
            seeds, positions, jitter = [], [], []
            for rid in context.request_ids[lo:hi].tolist()[:n]:
                req = engine.get_request(rid)
                sp = req.sampling_params
                seeds.append(int(getattr(sp, "noise_seed", -1)))
                positions.append(len(req.prompt_tokens) + len(req.generated_tokens))
                jitter.append(
                    (
                        float(getattr(sp, "draft_variant", 0)),
                        float(getattr(sp, "variant_eps", 0.0)),
                    )
                )
            if all(s < 0 for s in seeds):
                return orig_kernel(
                    logits,
                    n,
                    context,
                    gather_indices=gather_indices,
                    output=output,
                    **kw,
                )
            if all(s >= 0 for s in seeds):
                out = (
                    output
                    if output is not None
                    else torch.empty(n, device=logits.device, dtype=torch.int64)
                )
            else:
                out = orig_kernel(
                    logits,
                    n,
                    context,
                    gather_indices=gather_indices,
                    output=output,
                    **kw,
                )
            rows = (
                logits[gather_indices[:n]] if gather_indices is not None else logits[:n]
            )
            # Pinned host tensors + non-blocking copies: no per-step stream sync,
            # so the engine's async scheduling overlap is preserved.
            seed_c = torch.tensor(seeds, dtype=torch.long).pin_memory()
            pos_c = torch.tensor(positions, dtype=torch.long).pin_memory()
            jit_c = torch.tensor(jitter, dtype=torch.float64).pin_memory()
            if all(s >= 0 for s in seeds):
                out.copy_(keyed_sample_logits(rows, seed_c, pos_c, vocab_limit, jit_c))
            else:
                idx = [j for j, s in enumerate(seeds) if s >= 0]
                idx_t = (
                    torch.tensor(idx, dtype=torch.long)
                    .pin_memory()
                    .to(rows.device, non_blocking=True)
                )
                out[idx_t] = keyed_sample_logits(
                    rows[idx_t], seed_c[idx], pos_c[idx], vocab_limit, jit_c[idx]
                ).to(out.dtype)
            return out

        sampling.sample_kernel = keyed_kernel
        sampling._keyed_installed = True

        orig_prepare = self._prepare_data_for_generation

        def prepare(data, greedy=False):
            prompts, mm, sps = orig_prepare(data, greedy)
            if "noise_seed" in data:
                for i, sp in enumerate(sps):
                    sp.noise_seed = int(data["noise_seed"][i])
            if "max_new_tokens" in data:
                for i, sp in enumerate(sps):
                    sp.num_tokens_to_generate = int(data["max_new_tokens"][i])
            if "draft_variant" in data:
                for i, sp in enumerate(sps):
                    sp.draft_variant = int(data["draft_variant"][i])
                    sp.variant_eps = float(data["variant_eps"][i])
            return prompts, mm, sps

        self._prepare_data_for_generation = prepare
        return eod

    @torch.no_grad()
    @_model_in_eval
    def verify_drafts(
        self,
        rows: list[torch.Tensor],
        prompt_lens: list[int],
        seeds: list[int],
        vocab_limit: int,
        head_k: int,
        batch_tokens: int,
        position_shift: int = 0,
    ) -> list[tuple[int, dict[str, Any]]]:
        """Learner: keep each draft's longest prefix the current policy would emit.

        ``rows[i]`` = prompt + draft tokens. Position j (predicting token j+1) emits
        ``keyed_sample(p_theta(. | row[:j+1]), seed, j+1)``; the draft is kept up to
        the first disagreement, where the emitted token replaces it. If the whole
        draft agrees, the next token after it is emitted too. Rows are split across
        DP ranks and batched right-padded (causal attention). Returns
        [(row_index, {"accepted", "next", "logprobs"})] for this rank's rows.
        """
        from thundersync_rl.keyed_sampling import keyed_sample_logits

        from megatron.core import parallel_state as ps

        # By data-parallel rank: the tensor-parallel ranks of a replica share rows.
        dp, dp_rank = ps.get_data_parallel_world_size(), ps.get_data_parallel_rank()
        mine = sorted(
            (i for i in range(len(rows)) if i % dp == dp_rank),
            key=lambda i: rows[i].numel(),
        )
        groups = []
        b = 0
        while b < len(mine):
            e = b + 1
            while e < len(mine) and rows[mine[e]].numel() * (e + 1 - b) <= batch_tokens:
                e += 1
            groups.append(mine[b:e])
            b = e
        # Expert parallelism: every forward is an all-to-all over the EP group.
        lockstep = self._expert_parallel_lockstep([[] for _ in groups])
        for _ in range(len(groups), len(groups) if lockstep is None else len(lockstep)):
            n = self.cfg["make_sequence_length_divisible_by"]
            self.model(
                input_ids=torch.zeros((1, n), dtype=torch.long, device="cuda"),
                position_ids=torch.arange(n, device="cuda")[None],
                attention_mask=None,
            )
        results = []
        for group in groups:
            s = max(rows[i].numel() for i in group)
            # Pad like training (make_sequence_length_divisible_by): batch-invariant
            # kernels are invariant within a token-count shape class only.
            m = self.cfg["make_sequence_length_divisible_by"]
            s = -(-s // m) * m
            ids = torch.zeros((len(group), s), dtype=torch.long)
            for r, i in enumerate(group):
                ids[r, : rows[i].numel()] = rows[i]
            ids = ids.cuda()
            pos = torch.arange(s, device="cuda").expand(len(group), -1)
            logits = self.model(input_ids=ids, position_ids=pos, attention_mask=None)
            for r, i in enumerate(group):
                n, plen = rows[i].numel(), prompt_lens[i]
                lp = torch.log_softmax(
                    logits[r, plen - 1 : n, :vocab_limit].float(), -1
                )
                positions = torch.arange(plen, n + 1, device="cuda") + position_shift
                emitted = keyed_sample_logits(
                    lp, torch.full_like(positions, seeds[i]), positions
                )
                draft = ids[r, plen:n]
                agree = emitted[: draft.numel()] == draft
                n_acc = int(agree.long().cumprod(0).sum())
                kept = torch.cat([draft[:n_acc], emitted[n_acc : n_acc + 1]])
                # Sampling uses the vocab_limit slice (as the engine's keyed sampler
                # does); the reported log-probs are over the full vocabulary, as
                # training computes them (bitwise equal under batch invariance).
                lps = (
                    torch.log_softmax(logits[r, plen - 1 : plen + n_acc].float(), -1)
                    .gather(-1, kept.unsqueeze(-1))[:, 0]
                    .tolist()
                )
                results.append(
                    (
                        i,
                        {
                            "accepted": n_acc,
                            "next": int(emitted[n_acc]),
                            "logprobs": lps,
                        },
                    )
                )
            del logits
        return results

    # ---- Verification scorer: fp32 Megatron copy of the policy ----

    def _fp32_scorer(self):
        """Megatron copy of the policy with fp32 params/activations.

        Built from a shallow copy of the training model provider (same
        architecture, layer spec and process groups; fp32 dtypes; no DDP/optimizer) and refreshed by copying
        the training model's own bf16-valued parameters. Scoring p and q with it
        removes the bf16 activation-rounding noise that dominates the p/q ratio.
        """
        import copy

        t0 = time.perf_counter()
        if getattr(self, "_mfp32", None) is None:
            from megatron.core import parallel_state as ps

            if ps.get_context_parallel_world_size() > 1:
                # NeMo-RL runs MCore context parallelism only with sequence
                # packing (THD); the scorer would have to score packed, CP-sharded
                # batches through that path, which also needs packing support in
                # the streaming trainer.
                raise NotImplementedError(
                    "speculative verification scorer does not support context "
                    "parallelism (requires packed THD scoring)"
                )
            # Shallow copy: only top-level dtype/recompute fields change; process
            # groups (not copyable) are shared with the training model.
            provider = copy.copy(self.megatron_cfg.model)
            # "fp32" (default): fp32 params/activations, removes bf16 rounding
            # noise from p/q (~99% kept). "model": keep the training dtypes
            # (half the memory, bf16 speed, ~91% kept).
            if getattr(self, "_scorer_precision", "fp32") == "fp32":
                # Batch-invariant kernels are BF16-only; an fp32 scorer is built
                # outside batch-invariant mode (its higher precision, not
                # bitwise parity with training, is what it is for). TE attention
                # settings are process-wide, so they stay as configured; its
                # attention is replaced by SDPA below.
                provider.batch_invariant_mode = False
                provider.params_dtype = torch.float32
                provider.pipeline_dtype = torch.float32
                provider.bf16 = False
                provider.fp16 = False
            # On inference workers the provider builds MCore's inference-optimized
            # layers (engine-owned buffers): sharing them corrupted the engine's
            # weights/logits. The scorer always uses the standard TE layers.
            if getattr(provider, "transformer_impl", None) == "inference_optimized":
                from megatron.bridge.models.gpt_provider import (
                    GPTModelProvider,
                    default_layer_spec,
                )

                provider.transformer_impl = "transformer_engine"
                if isinstance(provider, GPTModelProvider):
                    provider.transformer_layer_spec = default_layer_spec
            # The scorer never runs under CUDA graphs; in an inference process a
            # graph-enabled copy would join the engine's graph registry.
            provider.cuda_graph_impl = "none"
            if hasattr(provider, "inference_cuda_graph_scope"):
                from megatron.core.transformer.enums import InferenceCudaGraphScope

                provider.inference_cuda_graph_scope = InferenceCudaGraphScope.none
            provider.recompute_granularity = None
            provider.recompute_method = None
            provider.recompute_num_layers = None
            if provider.params_dtype == torch.float32:
                # The fused bias-SwiGLU kernel is ~4x off the memory bound in
                # fp32; the unfused path is replaced by a Triton SwiGLU below.
                # (A training-dtype scorer keeps the training op sequence, which
                # batch-invariant kernels need to reproduce training numerics.)
                provider.bias_activation_fusion = False
            provider.finalize()
            self._mfp32 = provider.provide().cuda().eval()
            # RotaryEmbedding.forward is a process-wide lru_cache(32). In an
            # inference process the engine's CUDA graphs read tables from that
            # cache, so the scorer's own RoPE calls must not evict them.
            rope = getattr(self._mfp32, "rotary_pos_emb", None)
            if rope is not None and hasattr(rope.forward, "cache_parameters"):
                import types

                rope.forward = types.MethodType(type(rope).forward.__wrapped__, rope)
            if provider.params_dtype == torch.float32:
                # Megatron/TE fall back to unfused fp32 attention and GLU paths.
                from thundersync_rl.scorer_kernels import (
                    use_fused_swiglu,
                    use_sdpa_attention,
                )

                attn_bf16 = os.environ.get("THUNDERSYNC_SCORER_ATTN_BF16") == "1"
                n_attn = use_sdpa_attention(
                    self._mfp32, provider, torch.bfloat16 if attn_bf16 else None
                )
                n_mlp = use_fused_swiglu(self._mfp32)
            else:
                n_attn = n_mlp = 0
            from thundersync_rl.scorer_kernels import use_selective_output

            selective = use_selective_output(self._mfp32)
            if self.rank == 0:
                print(
                    f"[thundersync scorer] dtype={provider.params_dtype} "
                    f"sdpa_attention={n_attn} fused_swiglu={n_mlp} "
                    f"selective_lm_head={selective}",
                    flush=True,
                )
            src_names = [n for n, _ in self.model.named_parameters()]
            self._mfp32_map = []
            src = dict(self.model.named_parameters())
            for name, p in self._mfp32.named_parameters():
                match = [n for n in src_names if n == name or n.endswith("." + name)]
                assert len(match) == 1, (name, match[:3])
                self._mfp32_map.append((p, src[match[0]]))
        with torch.no_grad():
            for dst, src in self._mfp32_map:
                dst.copy_(src.detach().to(dst.dtype))
        torch.cuda.synchronize()
        self._prof["export"] = self._prof.get("export", 0.0) + time.perf_counter() - t0
        return self._mfp32

    def termination_id(self) -> int | None:
        tok = getattr(self, "megatron_tokenizer", None)
        return None if tok is None else int(tok.eod)

    # ---- Deadline-truncated streaming drafts (generation rank 0) ----

    def start_drafts(
        self,
        prompts: list[list[int]],
        max_new_tokens: int | list[int],
        stream_interval: int,
        hidden_keys: list[str] | None = None,
    ) -> bool:
        """Submit draft requests as streams and return immediately.

        Tokens and the engine's log-probs of them accumulate in
        ``self._draft_tokens`` / ``self._draft_logprobs`` as partial frames
        arrive; ``collect_drafts`` aborts whatever is unfinished at the deadline.
        Any prefix of a draft is a valid draft, so truncation is harmless.

        ``hidden_keys[i]`` (engine-sourced q): tag draft i so the engine logs
        its final hidden states under that key (``install_draft_hidden_log``,
        fetched with ``pop_draft_hidden``).
        """
        import asyncio

        from thundersync_rl.engine_q import HIDDEN_KEY_ATTR

        if self.inference_client is None or self._inference_loop is None:
            return False
        # Fresh buffers per call, captured by the coroutines below: streams aborted
        # at an earlier deadline may still deliver late frames, which must not
        # land in this round's drafts.
        tokens: list[list[int]] = [[] for _ in prompts]
        logprobs: list[list[float] | None] = [[] for _ in prompts]
        done = [False] * len(prompts)
        streams = {}
        (
            self._draft_tokens,
            self._draft_logprobs,
            self._draft_done,
            self._draft_streams,
        ) = (tokens, logprobs, done, streams)

        async def run_one(i, prompt):
            # return_log_probs=True: the engine reports log q'(x) of each sampled
            # token (temperature 1, no top-k/top-p under speculation, so raw and
            # processed log-probs coincide).
            sp = self._build_sampling_params(greedy=False, stop_words=None)
            sp.num_tokens_to_generate = (
                max_new_tokens[i]
                if isinstance(max_new_tokens, list)
                else max_new_tokens
            )
            sp.streaming_interval = stream_interval
            if hidden_keys is not None:
                # Travels with the serialized sampling params to the engine.
                sp.add_attributes({HIDDEN_KEY_ATTR: hidden_keys[i]})
            stream = self.inference_client.add_request_streaming(prompt, sp)
            streams[i] = stream
            async for frame in stream:
                done[i] = apply_draft_frame(frame, tokens[i], logprobs, i) or done[i]

        for i, p in enumerate(prompts):
            asyncio.run_coroutine_threadsafe(run_one(i, list(p)), self._inference_loop)
        return True

    def draft_request_ids(self) -> dict[int, int]:
        """Engine request id of each unfinished draft (rank 0 holds the streams)."""
        done = getattr(self, "_draft_done", None) or []
        return {
            i: int(st.request_id)
            for i, st in (getattr(self, "_draft_streams", None) or {}).items()
            if not done[i]
        }

    def draft_records(self, request_ids: list[int]) -> dict[int, tuple]:
        """(tokens, engine log-probs) of the listed requests this engine holds.

        Read from the engine's request records (``record.merge()``, as its final
        reply does), not from streamed partial frames: after an eviction the
        engine resumes a request as a new record whose generated tokens restart
        at 0 while its partial-frame counter does not, so the frames silently
        skip tokens (Megatron DynamicInferenceEngine._try_send_streaming_partials).
        """
        import asyncio

        engine = getattr(self, "dynamic_inference_engine", None)
        loop = getattr(self, "_inference_loop", None)
        if engine is None or loop is None:
            return {}

        async def read():
            out = {}
            for rid in request_ids:
                entry = engine.requests.get(rid)
                if entry is not None:
                    merged = entry.record.merge()
                    out[rid] = (
                        list(merged.generated_tokens),
                        _plain_floats(merged.generated_log_probs),
                    )
            return out

        return asyncio.run_coroutine_threadsafe(read(), loop).result()

    def collect_drafts(
        self, records: dict[int, tuple] | None = None
    ) -> list[tuple[list[int], bool, list[float] | None]]:
        """Abort unfinished drafts; return (tokens, finished, engine logprobs) per draft.

        The log-probs are None when the engine did not report one per token.
        """
        import asyncio

        tokens, logprobs, done, streams = (
            self._draft_tokens,
            self._draft_logprobs,
            self._draft_done,
            self._draft_streams,
        )

        async def abort_and_snapshot():
            # Snapshot on the loop thread so no frame is half-applied.
            stats = {"from_record": 0, "fallback": 0, "differed": 0}
            for i, stream in list(streams.items()):
                if done[i]:
                    continue
                rec = (records or {}).get(int(stream.request_id))
                if rec is None:
                    stats["fallback"] += 1
                else:
                    toks, lps = rec
                    stats["from_record"] += 1
                    stats["differed"] += int(list(toks) != tokens[i])
                    tokens[i][:] = list(toks)
                    if logprobs[i] is not None or lps is not None:
                        logprobs[i] = (
                            lps
                            if lps is not None and len(lps) == len(tokens[i])
                            else None
                        )
                self.inference_client.abort_request(stream.request_id)
            if os.environ.get("THUNDERSYNC_HIDDEN_DEBUG"):
                print(f"[collect dbg] {stats}", flush=True)
            return snapshot_drafts(tokens, logprobs, done)

        return asyncio.run_coroutine_threadsafe(
            abort_and_snapshot(), self._inference_loop
        ).result()

    # ---- Memory-scalable block verification: stash weights, not q ----
    # q is recomputed at verification time from a bf16 stash of the weights that
    # drafted each position (2 bytes/param per model-parallel shard per stashed
    # version), so memory is O(params + one batch x vocab) instead of
    # O(all draft tokens x vocab).

    @torch.no_grad()
    def stash_weights(self, version: int, keep: list[int], precision: str) -> int:
        """Stash the current training parameters as ``version``; drop others not in ``keep``."""
        self._prof = {}
        self._scorer_precision = precision
        self._fp32_scorer()  # builds the scorer and its parameter map once
        if not hasattr(self, "_weight_stash"):
            self._weight_stash: dict[int, list[torch.Tensor]] = {}
        self._weight_stash[version] = [
            src.detach().clone() for _, src in self._mfp32_map
        ]
        for v in list(self._weight_stash):
            if v not in keep and v != version:
                del self._weight_stash[v]
        self._log_mem(f"stash_weights v={version} n={len(self._weight_stash)}")
        return len(self._weight_stash)

    @torch.no_grad()
    def _load_scorer(self, version: int | None) -> None:
        """Load stashed ``version`` (None: current training weights) into the scorer."""
        t0 = time.perf_counter()
        srcs = (
            [src for _, src in self._mfp32_map]
            if version is None
            else self._weight_stash[version]
        )
        for (dst, _), src in zip(self._mfp32_map, srcs):
            dst.copy_(src)
        torch.cuda.synchronize()
        self._prof["load"] = self._prof.get("load", 0.0) + time.perf_counter() - t0

    @torch.no_grad()
    @_model_in_eval
    def verify_drafts_block_stashed(
        self,
        rows: list[torch.Tensor],
        prompt_lens: list[int],
        segments: list[list[tuple[int, int, int]]],
        vocab_limit: int,
        seed: int,
        keys: list[str],
        batch_tokens: int,
        precision: str,
        lazy: bool = False,
    ) -> list[tuple[int, dict[str, Any]]]:
        """Block-verify drafts whose q is recomputed per batch from stashed weights.

        ``segments[i]``: [(version, start, end)] - draft tokens [start, end) of row
        i were drafted by stashed weights ``version``. p is the current weights.
        Returns [(row_index, {"accepted", "next", "logprobs", "draft_logq"})]:
        "logprobs" = log p of the kept tokens (accepted prefix + next),
        "draft_logq" = log q of the accepted prefix.

        ``lazy``: lazy block verification (residual masses only where the
        backward scan needs them; pathwise identical to the eager rule given the
        same logits). Logit-free when the scorer supports position selection:
        keep final hidden states [T, d] instead of [T, V] logits, compute the
        logsumexps and draft-token logits in row chunks, and recompute full rows
        only at visited positions (``THUNDERSYNC_LAZY_LOGITS=1`` keeps the logits
        and only makes the scan lazy).
        """
        from thundersync_rl.block_verification_fused import (
            block_verify_logits,
            block_verify_logits_lazy,
        )
        from thundersync_rl.scorer_kernels import supports_selection

        self._prof = {}
        t_all = time.perf_counter()
        self._scorer_precision = precision
        scorer = self._fp32_scorer()
        from megatron.core import parallel_state as ps

        logit_free = (
            lazy
            and supports_selection(scorer)
            and os.environ.get("THUNDERSYNC_LAZY_LOGITS") != "1"
        )
        verify_fn = block_verify_logits_lazy if lazy else block_verify_logits

        # Same stream on every TP rank of a replica: identical decisions.
        gen = torch.Generator(device="cuda").manual_seed(
            seed * 1_000_003 + ps.get_data_parallel_rank()
        )
        results = []
        batches = list(self._iter_row_id_batches(rows, batch_tokens, keys))
        plans = [
            sorted({v for i in group for v, a, b in segments[i] if b > a})
            for group, _ in batches
        ]
        lockstep = self._expert_parallel_lockstep(plans)
        for j in range(len(batches) if lockstep is None else len(lockstep)):
            if j >= len(batches):
                # Expert parallelism: out of rows, join the other ranks' forwards.
                for v in lockstep[j]:
                    self._load_scorer(v)
                    self._dummy_score(scorer)
                self._load_scorer(None)
                self._dummy_score(scorer)
                continue
            group, ids = batches[j]
            pos = torch.arange(ids.shape[1], device="cuda").expand(len(group), -1)
            versions = plans[j] if lockstep is None else lockstep[j]
            if logit_free:
                results.extend(
                    self._verify_batch_logit_free(
                        scorer,
                        group,
                        ids,
                        pos,
                        versions,
                        rows,
                        prompt_lens,
                        segments,
                        vocab_limit,
                        gen,
                    )
                )
                continue
            lq_rows = self._stashed_q_logits(
                scorer,
                group,
                ids,
                pos,
                prompt_lens,
                segments,
                vocab_limit,
                lockstep_versions=None if lockstep is None else lockstep[j],
            )
            self._load_scorer(None)
            spans = [(prompt_lens[i] - 1, rows[i].numel()) for i in group]
            logits = self._score(
                scorer,
                ids,
                pos,
                select=(
                    torch.cat([torch.arange(a, e) for a, e in spans]).cuda(),
                    torch.cat(
                        [torch.full((e - a,), r) for r, (a, e) in enumerate(spans)]
                    ).cuda(),
                ),
            )
            if logits is None:
                continue
            off = 0
            for r, i in enumerate(group):
                n, plen = rows[i].numel(), prompt_lens[i]
                g = n - plen
                p_logits = logits[off : off + g + 1, :vocab_limit]
                off += g + 1
                parts = [t for _, t in sorted(lq_rows.pop(i), key=lambda x: x[0])]
                q_logits = torch.cat(parts) if parts else p_logits[:0]
                assert q_logits.shape[0] == g, (q_logits.shape, g)
                draft = ids[r, plen:n]
                # Raw logits in, no [g, V] log-softmax / fp64 temporaries.
                tau, y, logp = verify_fn(p_logits, q_logits, draft, gen)
                # log q of the accepted draft tokens (same vocab slice as the
                # acceptance ratios) for the behavior log-probs (speculative._plan).
                qa = q_logits[:tau].float()
                logq = (
                    qa.gather(-1, draft[:tau].view(-1, 1))[:, 0]
                    - torch.logsumexp(qa, -1)
                    if tau
                    else qa.new_empty(0)
                )
                if self._batch_invariant:
                    # Report log-probs exactly as training computes them (fp32
                    # log-softmax + gather), so they match the trainer bitwise.
                    kept = torch.cat([draft[:tau], torch.tensor([y], device="cuda")])
                    logp = torch.log_softmax(p_logits[: tau + 1].float(), -1).gather(
                        -1, kept.unsqueeze(-1)
                    )[:, 0]
                results.append(
                    (
                        i,
                        {
                            "accepted": tau,
                            "next": y,
                            "logprobs": logp.tolist(),
                            # log q of the accepted draft tokens (same vocab
                            # slice as verification), for the behavior
                            # log-probs log p + log q' - log q (speculative._plan).
                            "draft_logq": logq.tolist(),
                        },
                    )
                )
            del logits
        self._log_mem("verify_drafts_block_stashed")
        if os.environ.get("THUNDERSYNC_SPEC_PROF"):
            pool = (
                "gen"
                if getattr(self, "dynamic_inference_engine", None) is not None
                else "learner"
            )
            ntok = sum(rows[i].numel() for i in range(len(rows)))
            print(
                f"[spec prof] verify_stashed pool={pool} rank={self.rank} rows={len(rows)} tok={ntok} lazy={int(lazy)} logit_free={int(logit_free)} total={time.perf_counter() - t_all:.3f} {self._prof}",
                flush=True,
            )
        if lazy and os.environ.get("THUNDERSYNC_LAZY_CHECK") == "1":
            # Debug: rerun the eager logits path with the same seed and compare.
            ref = dict(
                self.verify_drafts_block_stashed(
                    rows,
                    prompt_lens,
                    segments,
                    vocab_limit,
                    seed,
                    keys,
                    batch_tokens,
                    precision,
                    lazy=False,
                )
            )
            diff = sum(
                (o["accepted"], o["next"]) != (ref[i]["accepted"], ref[i]["next"])
                for i, o in results
            )
            dlp = max(
                (
                    max(abs(a - b) for a, b in zip(o["logprobs"], ref[i]["logprobs"]))
                    for i, o in results
                    if o["accepted"] == ref[i]["accepted"]
                ),
                default=0.0,
            )
            print(
                f"[lazy check] rank={self.rank} rows={len(results)} logit_free="
                f"{int(logit_free)} decision_mismatches={diff} max|dlogp|={dlp:.3e}",
                flush=True,
            )
        return results

    @torch.no_grad()
    def _stashed_q_logits(
        self,
        scorer,
        group: list[int],
        ids: torch.Tensor,
        pos: torch.Tensor,
        prompt_lens: list[int],
        segments: list[list[tuple[int, int, int]]],
        vocab_limit: int,
        lockstep_versions: list[int] | None = None,
    ) -> dict[int, list[tuple[int, torch.Tensor]]]:
        """Q logits of a batch's draft positions, from the weights that drafted them.

        Returns row -> [(segment start, [segment length, V] logits)]; empty off
        the last pipeline stage. ``lockstep_versions``: the expert-parallel
        group's common versions for this batch slot (see
        ``_expert_parallel_lockstep``); versions this rank has no rows for run a
        dummy forward.
        """
        lq_rows: dict[int, list[tuple[int, torch.Tensor]]] = {i: [] for i in group}
        versions = lockstep_versions
        if versions is None:
            versions = sorted({v for i in group for v, _, _ in segments[i]})
        for v in versions:
            # q of version v is needed only at its segments' positions; causal
            # attention lets each row stop at its last such position.
            segs = [
                (r, i, a, b)
                for r, i in enumerate(group)
                for sv, a, b in segments[i]
                if sv == v and b > a
            ]
            if not segs:
                if lockstep_versions is not None:
                    # Expert parallelism: another rank needs version v here.
                    self._load_scorer(v)
                    self._dummy_score(scorer)
                continue
            sub_rows = sorted({r for r, _, _, _ in segs})
            local = {r: k for k, r in enumerate(sub_rows)}
            width = max(prompt_lens[i] - 1 + b for _, i, _, b in segs)
            s_idx = torch.cat(
                [
                    torch.arange(prompt_lens[i] - 1 + a, prompt_lens[i] - 1 + b)
                    for _, i, a, b in segs
                ]
            )
            b_idx = torch.cat([torch.full((b - a,), local[r]) for r, _, a, b in segs])
            self._load_scorer(v)
            logits = self._score(
                scorer,
                ids[sub_rows, :width],
                pos[: len(sub_rows), :width],
                select=(s_idx.cuda(), b_idx.cuda()),
            )
            if logits is None:  # not the last pipeline stage
                continue
            off = 0
            for _, i, a, b in segs:
                lq_rows[i].append((a, logits[off : off + b - a, :vocab_limit]))
                off += b - a
            del logits
        return lq_rows

    def _verify_batch_logit_free(
        self,
        scorer,
        group,
        ids,
        pos,
        versions,
        rows,
        prompt_lens,
        segments,
        vocab_limit,
        gen,
    ) -> list[tuple[int, dict[str, Any]]]:
        """One batch of ``verify_drafts_block_stashed`` without [T, V] logits.

        Same forwards, version order and generator calls as the logits path; q
        and p come as final hidden states, reduced to (logsumexp, draft-token
        logit) per position in row chunks (``ChunkedHead``). The lazy scan then
        recomputes full rows only where it needs r_i, plus the row it samples
        from. q rows of version v are recomputed with the head weight of stash v
        (the values the scorer held during that forward).
        """
        from thundersync_rl.block_verification_fused import block_verify_stats_lazy
        from thundersync_rl.logit_free import ChunkedHead

        chunk = int(os.environ.get("THUNDERSYNC_LAZY_CHUNK", "512"))
        # Row id -> [(start, head, base, length, lse, picked)] per q segment.
        q_parts: dict[int, list[tuple]] = {i: [] for i in group}
        heads = []
        for v in versions:
            segs = [
                (r, i, a, b)
                for r, i in enumerate(group)
                for sv, a, b in segments[i]
                if sv == v and b > a
            ]
            if not segs:
                # Expert parallelism: another rank needs version v here.
                self._load_scorer(v)
                self._dummy_score(scorer)
                continue
            sub_rows = sorted({r for r, _, _, _ in segs})
            local = {r: k for k, r in enumerate(sub_rows)}
            width = max(prompt_lens[i] - 1 + b for _, i, _, b in segs)
            s_idx = torch.cat(
                [
                    torch.arange(prompt_lens[i] - 1 + a, prompt_lens[i] - 1 + b)
                    for _, i, a, b in segs
                ]
            )
            b_idx = torch.cat([torch.full((b - a,), local[r]) for r, _, a, b in segs])
            self._load_scorer(v)
            out = self._score_hidden(
                scorer,
                ids[sub_rows, :width],
                pos[: len(sub_rows), :width],
                (s_idx.cuda(), b_idx.cuda()),
                vocab_limit,
            )
            if out is None:  # not the last pipeline stage
                continue
            hidden, logits_fn, w_live = out
            tokens = torch.cat(
                [
                    rows[i][prompt_lens[i] + a : prompt_lens[i] + b]
                    for _, i, a, b in segs
                ]
            ).cuda()
            head = ChunkedHead(hidden, logits_fn, lambda w=w_live: w, chunk)
            lse, picked = head.stats(tokens)
            # Rows recomputed later: the scorer then holds other weights.
            head.weight = self._stashed_head_weight(v, w_live)
            heads.append(head)
            off = 0
            for _, i, a, b in segs:
                n = b - a
                q_parts[i].append(
                    (a, head, off, n, lse[off : off + n], picked[off : off + n])
                )
                off += n
        self._load_scorer(None)
        spans = [(prompt_lens[i] - 1, rows[i].numel()) for i in group]
        out = self._score_hidden(
            scorer,
            ids,
            pos,
            (
                torch.cat([torch.arange(a, e) for a, e in spans]).cuda(),
                torch.cat(
                    [torch.full((e - a,), r) for r, (a, e) in enumerate(spans)]
                ).cuda(),
            ),
            vocab_limit,
        )
        if out is None:
            return []
        hidden, logits_fn, w_live = out
        tokens, pins, off = [], [], 0
        for i in group:
            draft = rows[i][prompt_lens[i] :]
            tokens += [draft, torch.zeros(1, dtype=draft.dtype)]  # bonus: no token
            off += draft.numel() + 1
            pins.append(off - 1)
        head_p = ChunkedHead(hidden, logits_fn, lambda: w_live, chunk)
        heads.append(head_p)
        lse_p_all, picked_p_all = head_p.stats(torch.cat(tokens).cuda(), pins)
        results = []
        off = 0
        empty = torch.empty(0, dtype=torch.float32, device="cuda")
        for i in group:
            g = rows[i].numel() - prompt_lens[i]
            lse_p = lse_p_all[off : off + g + 1]
            lpx = picked_p_all[off : off + g] - lse_p[:g]
            parts = sorted(q_parts.pop(i), key=lambda x: x[0])
            lse_q = torch.cat([pt[4] for pt in parts]) if parts else empty
            lqx = (torch.cat([pt[5] for pt in parts]) if parts else empty) - lse_q
            assert lqx.numel() == g, (lqx.numel(), g)

            def q_rows(idx, parts=parts):
                out = []
                for t in idx.tolist():
                    a, head, base, _, _, _ = next(
                        pt for pt in parts if pt[0] <= t < pt[0] + pt[3]
                    )
                    out.append(head.row(base + t - a))
                return torch.stack(out)

            tau, y, logp, n_eval = block_verify_stats_lazy(
                lpx,
                lqx,
                lse_p,
                lse_q,
                lambda idx, off=off: head_p.rows(idx + off),
                q_rows,
                gen,
            )
            self._prof["lazy_rows"] = self._prof.get("lazy_rows", 0) + n_eval
            results.append(
                (
                    i,
                    {
                        "accepted": tau,
                        "next": y,
                        "logprobs": logp.tolist(),
                        "draft_logq": lqx[:tau].tolist(),
                    },
                )
            )
            off += g + 1
        for head in heads:
            head.release()
        return results

    def _stashed_head_weight(self, version, w_live):
        """Callable giving the LM head weight of stash ``version`` (scorer dtype).

        Bitwise the values ``_load_scorer(version)`` copied into ``w_live``.
        """
        k = next(j for j, (dst, _) in enumerate(self._mfp32_map) if dst is w_live)
        return lambda: self._weight_stash[version][k].to(w_live.dtype)

    def _score_hidden(self, scorer, ids, pos, select, vocab_limit):
        """Final hidden states at ``select`` plus a function computing their logits.

        Returns (hidden [n, h], logits_fn(h, weight) -> fp32 [m, vocab_limit],
        head weight) on the last pipeline stage, else None. ``logits_fn`` is the
        forward's LM head (with its logit scaling), all-gathered over the
        vocabulary under tensor parallelism one row chunk at a time.
        """
        from megatron.core import parallel_state as ps
        from thundersync_rl.scorer_kernels import capture_head_input

        with capture_head_input(scorer, select) as cap:
            out = self._score_full(scorer, ids, pos)
        if out is None:
            return None
        ((hidden, weight, rgo),) = cap
        layer = scorer.output_layer
        w_live = layer.weight if weight is None else weight
        tp = ps.get_tensor_model_parallel_world_size()
        gathered = layer.gather_output if rgo is None else rgo
        scale = getattr(scorer, "_scale_logits", None)

        def logits_fn(h, w):
            z, _ = layer._ts_orig_forward(
                h.unsqueeze(1), weight=w, runtime_gather_output=rgo
            )
            if scale is not None:
                z = scale(z)
            z = z[:, 0].float()
            if tp > 1 and not gathered:
                # Vocab-parallel -> full vocab, one chunk of rows at a time.
                parts = [torch.empty_like(z) for _ in range(tp)]
                torch.distributed.all_gather(
                    parts, z.contiguous(), group=ps.get_tensor_model_parallel_group()
                )
                z = torch.cat(parts, dim=-1)
            return z[:, :vocab_limit]

        return hidden, logits_fn, w_live

    # ---- Engine-sourced q under batch invariance (verify_source="engine") ----
    # q of a drafted token is the engine's own log-prob; full q rows come from
    # the engine's logged final hidden states and an LM-head-only stash per
    # pending version (engine_q.py). p is the training model's own forward.

    def _initialize_inference_engine(self, mcore_generation_config: dict) -> None:
        """Build the inference engine with the draft hidden-state copy installed.

        The engine captures its CUDA graphs in its constructor; the LM-head copy
        pre-hook must exist by then on every rank (the captures run TP/EP
        collectives, so all ranks capture the same graphs), or graph replays
        skip it. The constructor is wrapped for the duration of the build.
        """
        from megatron.core.inference.engines import dynamic_engine as de

        orig = de.DynamicInferenceEngine
        worker = self

        class _Engine(orig):  # type: ignore[misc, valid-type]
            def __init__(self, controller, context, *args, **kwargs):
                worker._install_hidden_copy(context)
                worker._install_route_record(context)
                super().__init__(controller, context, *args, **kwargs)

        de.DynamicInferenceEngine = _Engine
        try:
            super()._initialize_inference_engine(mcore_generation_config)
        finally:
            de.DynamicInferenceEngine = orig

    def _install_hidden_copy(self, context) -> None:
        """Copy each forward's LM-head input rows into a static buffer.

        One [max_requests, d] buffer per engine (bf16: 14 MB at d=7168 and
        1024 requests), written by a pre-hook on the output layer; the copy is
        captured into the CUDA graphs. Only the last pipeline stage has an
        output layer. Rows are read by ``install_draft_hidden_log``'s sampler.
        """
        model = _unwrap(self._gen_model())
        layer = getattr(model, "output_layer", None)
        if layer is None or getattr(self, "_hidden_copy", None) is not None:
            return
        try:
            from megatron.core.inference.utils import InferenceMode
        except ImportError:  # pragma: no cover
            InferenceMode = None
        materialized = bool(context.config.materialize_only_last_token_logits)
        # Tied embeddings: the output layer holds no weight of its own.
        weight = layer.weight
        if weight is None:
            weight = model.shared_embedding_or_output_weight()
        from nemo_rl.models.megatron.fp16_gemm import head_input_dtype

        # The head input dtype: fp16 under forward_gemm_precision=fp16, where the
        # head GEMM rounds its fp32 input to fp16 first (the copy rounds the same
        # way, so q recomputed from these rows is exact); else the weight dtype.
        static = torch.empty(
            (int(context.max_requests), weight.shape[1]),
            dtype=head_input_dtype(weight.dtype),
            device=weight.device,
        )
        filled = [False]

        def hook(module, args, kwargs):
            if InferenceMode is not None and not InferenceMode.is_active():
                return  # not an engine forward
            h = args[0] if args else kwargs["input_"]
            rows = h.reshape(-1, h.shape[-1])
            if not materialized:
                if torch.cuda.is_current_stream_capturing():
                    return  # per-token rows: only copied on eager forwards
                rows = context.last_token_logits(rows.unsqueeze(0))
            k = min(rows.shape[0], static.shape[0])
            static[:k].copy_(rows[:k])
            filled[0] = True

        layer.register_forward_pre_hook(hook, with_kwargs=True)
        self._hidden_copy = {
            "static": static,
            "filled": filled,
            "weight": weight,
            "materialized": materialized,
        }

    # ---- Sticky MoE routing: the drafter's routes (anchors), engine side ----

    def _install_route_record(self, context) -> None:
        """Record every engine token's plain top-k into one static buffer.

        Sticky routing only (``policy.megatron_cfg.sticky_routing_delta``).
        MCore ``RouterReplay`` RECORD with a static buffer per MoE layer (a
        [max_tokens, L_local, K] int16 tensor, one column per local layer):
        the copy is part of each router's forward, so it is captured into the
        engine's CUDA graphs, which is why it is armed here, before the engine
        constructor captures them. Recording is restricted to inference
        forwards (``sticky_routing``'s wrapper), so a training-shaped forward
        in this process neither records nor overruns the buffer.
        """
        from megatron.core.transformer.moe.router_replay import RouterReplayAction

        from nemo_rl.models.megatron import sticky_routing
        from nemo_rl.models.megatron.router_replay import (
            _router_replay_instances_for_model,
        )

        if sticky_routing.get_delta() is None or getattr(self, "_route_record", None):
            return
        sticky_routing.install()
        insts = sorted(
            _router_replay_instances_for_model(self._gen_model()), key=lambda x: x[1]
        )
        if not insts:
            return
        from megatron.core.utils import get_model_config

        cfg = get_model_config(self._gen_model())
        k = int(cfg.moe_router_topk)
        dev = torch.device("cuda", torch.cuda.current_device())
        buf = torch.full(
            (int(context.max_tokens), len(insts), k), -1, dtype=torch.int16, device=dev
        )
        for j, (replay, _) in enumerate(insts):
            replay.set_static_buffer(buf[:, j, :])
            replay._nrl_record_inference_only = True
            replay.set_router_replay_action(RouterReplayAction.RECORD)
        self._route_record = {
            "buf": buf,
            "layers": [ln for _, ln in insts],
            "sequence_parallel": bool(getattr(cfg, "sequence_parallel", False)),
        }

    def install_draft_route_log(self) -> bool | None:
        """Log the routes of tagged draft requests (prompt and drafted tokens).

        Generation workers, sticky routing. Wraps the controller's
        ``_dynamic_step_forward_logits``, which runs in Python for every engine
        forward (eager or CUDA-graph replay, legacy or async scheduling): before
        the forward it snapshots the step's token layout from the context's
        CPU bookkeeping (which at launch time describes exactly this forward:
        slot and absolute position of every token), after it enqueues the
        copy of this forward's route rows of tagged tokens to host memory, on
        the forward's stream and before the next forward can overwrite the
        buffer. Tags are the hidden-state log's draft keys
        (``HIDDEN_KEY_ATTR``). Every rank that routes tokens logs: pipeline
        stages their layers, sequence-parallel TP ranks their token shard
        (TP rank 0 alone without sequence parallelism, where routes are
        replicated). Returns True when installed, None on ranks with nothing
        to log, False when unsupported.
        """
        from megatron.core import parallel_state as ps
        from thundersync_rl.engine_q import HIDDEN_KEY_ATTR
        from thundersync_rl.routes import RouteLog, step_route_entries

        engine = getattr(self, "dynamic_inference_engine", None)
        ctx = getattr(self, "inference_context", None)
        rec = getattr(self, "_route_record", None)
        if engine is None or ctx is None or rec is None:
            return False
        if getattr(self, "_route_log", None) is not None:
            return True
        tp = ps.get_tensor_model_parallel_world_size()
        tp_rank = ps.get_tensor_model_parallel_rank()
        sp = rec["sequence_parallel"] and tp > 1
        if tp_rank != 0 and not sp:
            return None
        log = RouteLog()
        buf = rec["buf"]
        controller = engine.controller
        orig = controller._dynamic_step_forward_logits

        def snapshot():
            n = int(ctx.active_token_count)
            if n <= 0:
                return []
            slots = ctx.token_to_request_idx[:n].tolist()
            keys: dict[int, str] = {}
            eng = self.dynamic_inference_engine
            for slot in set(slots):
                rid = int(ctx.request_ids[slot])
                entry = eng.requests.get(rid)
                if entry is None:
                    continue
                key = getattr(entry.record[-1].sampling_params, HIDDEN_KEY_ATTR, None)
                if key is not None:
                    keys[slot] = key
            if not keys:
                return []
            pos = ctx.token_to_pos_ids[:n].tolist()
            if sp:
                chunk = int(ctx.padded_active_token_count) // tp
                return step_route_entries(slots, pos, keys, tp_rank * chunk, chunk)
            return step_route_entries(slots, pos, keys, 0, n)

        def forward_logits(*args, **kwargs):
            entries = snapshot()
            out = orig(*args, **kwargs)
            if entries:
                log.record(entries, buf[: max(r for r, _, _ in entries) + 1])
            return out

        controller._dynamic_step_forward_logits = forward_logits
        self._route_log = log
        return True

    def pop_draft_routes(self, keys: list[str]) -> dict[str, tuple]:
        """{key: (layer numbers, positions [m], routes [m, L_local, K] int16)}.

        Drafts this rank logged; every key is closed (late rows dropped).
        """
        log = getattr(self, "_route_log", None)
        if log is None:
            return {}
        layers = self._route_record["layers"]
        out = {}
        for k in keys:
            r = log.pop(k)
            if r is not None:
                out[k] = (layers, r[0], r[1])
        return out

    def drafts_done(self) -> list[bool]:
        """Finished flag per draft of the last ``start_drafts`` call (rank 0)."""
        return list(getattr(self, "_draft_done", None) or [])

    def install_draft_hidden_log(self) -> bool | None:
        """Log the engine's LM-head input for tagged draft requests.

        Generation workers. A forward pre-hook on the engine model's output
        layer copies, per forward, the row of each tagged request's last input
        token (the hidden state its next token is sampled from) to host memory,
        keyed by (draft key, generated index). Returns True when installed,
        None on ranks that hold no rows to log (not the last pipeline stage,
        tensor-parallel rank > 0), False when unsupported (no engine, MTP
        speculative decoding inside the engine, sequence-parallel head input).

        Tensor parallelism: with ``materialize_only_last_token_logits`` the
        engine all-gathers the sequence-parallel hidden states and selects each
        request's last row before the output layer (with its sequence
        parallelism switched off), so the hook's input is the full set of rows,
        replicated on every TP rank. TP rank 0 alone logs them (one device-to-
        host copy per forward, no extra collective). Without it the head input
        is sequence-sharded; that layout is unsupported (False).

        Pipeline parallelism: the output layer exists only on the engine's last
        stage, which computes the logits (the controller then broadcasts them
        to the other stages for sampling), so the hook is installed there; the
        dynamic engine's requests, with their sampling params, reach every
        model-parallel rank, so the last stage sees the tags. Other stages
        return None. Expert parallelism: each engine replica logs the drafts
        it runs; an expert-parallel dummy step has no requests and logs
        nothing.
        """
        from megatron.core import parallel_state as ps
        from thundersync_rl.engine_q import HIDDEN_KEY_ATTR, HiddenLog

        from nemo_rl.models.megatron.fp16_gemm import head_matmul

        engine = getattr(self, "dynamic_inference_engine", None)
        ctx = getattr(self, "inference_context", None)
        if engine is None or ctx is None:
            return False
        if getattr(self, "_hidden_log", None) is not None:
            return True
        if not ps.is_pipeline_last_stage(ignore_virtual=True):
            return None
        if ps.get_tensor_model_parallel_rank() != 0:
            return None
        if getattr(ctx, "num_speculative_tokens", 0):
            return False
        materialized = bool(ctx.config.materialize_only_last_token_logits)
        layer = _unwrap(self._gen_model()).output_layer
        if getattr(layer, "sequence_parallel", False) and not materialized:
            return False
        log = HiddenLog()

        def lookup(rid, engine):
            # Not cached: the engine may reuse request ids, so a cached id
            # could point at an earlier request's draft key.
            entry = engine.requests.get(rid)
            if entry is None:
                return None
            req = entry.record[-1]
            key = getattr(req.sampling_params, HIDDEN_KEY_ATTR, None)
            return None if key is None else (key, len(req.prompt_tokens))

        # The LM-head input rows are copied into a static buffer by a pre-hook
        # installed on every rank when the engine was built, before it captured
        # its CUDA graphs (``_install_hidden_copy``), so the copy replays with
        # every graph. Here only the logging (sampler side) is added.
        copy = getattr(self, "_hidden_copy", None)
        if copy is None:
            return False
        static, filled, weight = copy["static"], copy["filled"], copy["weight"]
        dbg = {"kernel": 0, "rows": 0, "rejected": 0}
        probe = os.environ.get("THUNDERSYNC_ENGINE_Q_CHECK") == "1"
        debug = bool(os.environ.get("THUNDERSYNC_HIDDEN_DEBUG"))

        sampling = engine.controller._sampling
        inner = sampling.sample_kernel

        def logging_kernel(logits, n, context, **kw):
            dbg["kernel"] += 1
            # With materialized last-token rows every forward (eager or graph
            # replay) refills the buffer; per-token rows are only copied on
            # eager forwards, flagged from Python.
            entries = []
            if materialized or filled[0]:
                filled[0] = False
                engine_now = self.dynamic_inference_engine
                a, e = context.paused_request_count, context.total_request_count
                for row, rid in enumerate(context.request_ids[a:e].tolist()):
                    info = lookup(rid, engine_now)
                    if info is None:
                        continue
                    rec = engine_now.requests.get(rid)
                    if rec is None:
                        continue
                    # Index of the token this row samples, relative to the
                    # request's original prompt: after an eviction the engine
                    # resumes the request as a new record whose prompt includes
                    # the earlier tokens and whose generated tokens restart at 0.
                    cur = rec.record[-1]
                    t = (
                        len(cur.prompt_tokens)
                        + len(cur.generated_tokens)
                        - len(rec.record[0].prompt_tokens)
                    )
                    entries.append((row, info[0], t))
            out = inner(logits, n, context, **kw)
            if not entries:
                return out
            # Source check (THUNDERSYNC_ENGINE_Q_CHECK=1 only: its .tolist() is
            # a GPU->CPU sync per decode step): a logged row must reproduce
            # 16 probe logits of the row the engine sampled from. Rows follow
            # the controller's async-schedule compaction (compact() below), so
            # the check should never reject; rejected rows are not recorded and
            # their drafts are truncated there (HiddenLog.pop returns the
            # longest logged prefix).
            if probe and logits is not None and torch.is_tensor(logits):
                rows_t = torch.tensor([r for r, _, _ in entries], device=static.device)
                lg = logits.reshape(-1, logits.shape[-1])
                w = weight
                lo = 0
                if w.shape[0] < lg.shape[-1]:  # vocab-parallel shard (TP > 1)
                    from megatron.core import parallel_state as ps_

                    lo = ps_.get_tensor_model_parallel_rank() * w.shape[0]
                # Probe columns of this rank's vocabulary shard (any TP, any
                # sampled token): a stale row disagrees on all of them.
                cols = torch.linspace(0, w.shape[0] - 1, 16, device=w.device).long()
                mine = head_matmul(static[rows_t], w[cols])
                ref = lg[rows_t][:, lo + cols].float()
                ok = ((mine - ref).abs().amax(-1) <= 1.0).tolist()
                if not all(ok):
                    dbg["rejected"] += len(entries) - sum(ok)
                    entries = [en for en, k_ in zip(entries, ok) if k_]
            if entries:
                dbg["rows"] += len(entries)
                log.record(entries, static[: max(r for r, _, _ in entries) + 1])
            if debug and dbg["kernel"] % 500 == 1:
                print(f"[hidden dbg] {dbg}", flush=True)
            return out

        sampling.sample_kernel = logging_kernel

        # Under async scheduling the controller launches the next forward
        # before finished requests are resolved, then compacts the surviving
        # rows and reorders the pending logits to match
        # (``_compact_async_sched_logits``). Apply the same permutation to the
        # hidden-copy buffer, on the same stream, so row i of the buffer stays
        # the hidden state of logits row i.
        controller = engine.controller
        orig_compact = controller._compact_async_sched_logits

        def compact(survivor_idxs, *args, **kwargs):
            n = survivor_idxs.numel()
            if n and not torch.equal(
                survivor_idxs, torch.arange(n, device=survivor_idxs.device)
            ):
                static[:n].copy_(static[survivor_idxs.to(static.device)])
            return orig_compact(survivor_idxs, *args, **kwargs)

        controller._compact_async_sched_logits = compact
        self._hidden_log = log
        return True

    def pop_draft_hidden(
        self, keys: list[str], lengths: list[int], as_refs: bool = False
    ) -> dict[str, Any]:
        """Logged hidden states [n, d] (CPU) of the drafts this engine ran.

        Drafts another engine ran, or with a missing row, are left out. Every
        key is closed: rows of later engine steps (an aborted stream finishing
        its step) are discarded.

        ``as_refs``: put each draft's rows into the Ray object store (one
        object per ``HIDDEN_BLOCK_ROWS`` rows) and return [(ObjectRef, rows,
        bytes)] per block instead of the tensor, so the caller only handles references
        and the verifying learner fetches the rows directly from this node,
        only if its lazy scan needs them. This worker owns the objects but
        keeps no reference: the driver frees them after verification.
        """
        from thundersync_rl.engine_q import put_hidden

        log = getattr(self, "_hidden_log", None)
        if log is None:
            return {}
        out = {}
        for k, n in zip(keys, lengths):
            h = log.pop(k, n)
            if h is not None and n > 0:
                out[k] = put_hidden(h) if as_refs else h
        return out

    def _lm_head(self):
        """(output layer, live head weight, logit scaling or None) of the training model.

        None off the last pipeline stage: Megatron builds the output layer only
        there (under tied embeddings the last stage keeps its own copy of the
        embedding weight as the output weight, synchronized by the embedding
        group, which ``shared_embedding_or_output_weight`` returns).
        """
        gpt = _unwrap(self.model)
        layer = getattr(gpt, "output_layer", None)
        if layer is None or not getattr(gpt, "post_process", True):
            return None
        weight = (
            gpt.shared_embedding_or_output_weight()
            if getattr(gpt, "share_embeddings_and_output_weights", False)
            else layer.weight
        )
        return layer, weight, getattr(gpt, "_scale_logits", None)

    @torch.no_grad()
    def stash_head(self, version: int, keep: list[int]) -> int:
        """Stash only the LM-head weight as ``version``.

        The live weight is this rank's vocab-parallel shard (V/TP x d; under
        tied embeddings the embedding's shard, same vocabulary split), so each
        tensor-parallel rank keeps 1/TP of the head per pending version.
        Pipeline parallelism: only the last stage holds the head and verifies;
        other stages stash nothing and return 0.
        """
        head = self._lm_head()
        if head is None:
            return 0
        _, weight, _ = head
        if not hasattr(self, "_head_stash"):
            self._head_stash: dict[int, torch.Tensor] = {}
        self._head_stash[version] = weight.detach().clone()
        for v in list(self._head_stash):
            if v not in keep and v != version:
                del self._head_stash[v]
        self._log_mem(
            f"stash_head v={version} n={len(self._head_stash)} "
            f"bytes={self.head_stash_bytes()}"
        )
        return len(self._head_stash)

    def head_stash_bytes(self) -> int:
        """Bytes of the LM-head stashes on this GPU (all pending versions)."""
        return sum(
            w.numel() * w.element_size()
            for w in getattr(self, "_head_stash", {}).values()
        )

    def _tp_vocab_gather(self):
        """Rank-order all-gather of vocab shards over the tensor-parallel group.

        The concatenation the engine's gathered output layer and
        batch-invariant training (``_GatheredVocabLogprob``) perform; identity
        at TP=1.
        """
        from megatron.core import parallel_state as ps
        from thundersync_rl.engine_q import no_gather

        from nemo_rl.models.megatron.zero_train_gen_mismatch import _all_gather_vocab

        if ps.get_tensor_model_parallel_world_size() == 1:
            return no_gather
        group = ps.get_tensor_model_parallel_group()
        return lambda x: _all_gather_vocab(x, group)

    def _head_logprobs(
        self, hidden: torch.Tensor, version: int, vocab_limit: int
    ) -> torch.Tensor:
        """Q log-prob rows [n, V] (fp32) of ``hidden`` [n, d] under head ``version``.

        The training model's own output layer with the stashed weight (this
        rank's vocab shard at TP>1), run as the engine runs it on its gathered
        last-token rows: sequence parallelism off, local logits, logit scaling,
        then the shards all-gathered in rank order and one fp32 log-softmax
        (``gathered_log_softmax``). Under batch invariance the GEMM gives each
        row's logits independently of the row count, so a row equals the
        engine's row for the same hidden state. A collective at TP>1: every
        rank of the group calls it with the same rows (the lazy scan's
        decisions are identical on all of them).
        """
        from thundersync_rl.engine_q import gathered_log_softmax

        layer, _, scale = self._lm_head()
        from nemo_rl.models.megatron.fp16_gemm import head_input_dtype

        w = self._head_stash[version]
        h = hidden.to(device=w.device, dtype=head_input_dtype(w.dtype))
        sp = getattr(layer, "sequence_parallel", False)
        if sp:
            # The rows are full (not sequence-sharded), as in the engine.
            layer.sequence_parallel = False
        try:
            z, _ = layer(h.unsqueeze(1), weight=w, runtime_gather_output=False)
        finally:
            if sp:
                layer.sequence_parallel = True
        if scale is not None:
            z = scale(z)
        return gathered_log_softmax(z[:, 0], self._tp_vocab_gather(), vocab_limit)

    @torch.no_grad()
    @_model_in_eval
    def verify_drafts_block_engine(
        self,
        rows: list[torch.Tensor],
        prompt_lens: list[int],
        segments: list[list[tuple[int, int, int]]],
        engine_logq: list[list[float]],
        hidden: list[Any],
        vocab_limit: int,
        seed: int,
        keys: list[str],
        batch_tokens: int = 0,
        anchors: list[Any] | None = None,
        null_groups: list[Optional[str]] | None = None,
        null_slots: dict[str, int] | None = None,
        null_policy: dict[str, Any] | None = None,
        **_: Any,
    ) -> list[tuple[int, dict[str, Any]]]:
        """Block verification with q from the engine and p from the training model.

        Pooled ⊥-BV (``null_bv.py``): ``null_groups[i]`` names row i's prompt
        group (None: plain block verification). The rows of one group are
        owned by one data-parallel rank (hash of the group name) and decided in
        their order in ``rows`` (the drafting order, fixed before
        verification): each with envelope M = policy(drafts left, slots left)
        until ``null_slots[group]`` outputs exist; later rows of the group are
        returned as ``"null": "unused"`` (forwarded but not decided). Every
        decided row reports ``"null_m"`` and ``"null"`` = "out" or "bot"
        (BOT: no output). ``null_policy``: ``NullPolicy`` keyword arguments.

        ``anchors[i]`` (sticky MoE routing): the drafter's routes of row i,
        [positions, L, K] int16 (-1: self); p is then F(theta; anchors), the
        sticky policy the rollouts are sampled from.

        Requires batch-invariant kernels (the engine's decode is the training
        forward) and the same parallel layout as the engine (TP, PP, EP; batch
        invariance enforces it). Per draft i:
        p = log-softmax of the training model's forward of ``rows[i]`` at the
        current weights (a training-shaped forward, as
        ``verify_drafts_block_fused`` runs it, without autograd; at TP>1 its
        vocab-parallel logits are all-gathered in rank order before the
        log-softmax, as ``_GatheredVocabLogprob`` does); log q(x_j) =
        ``engine_logq[i][j]``; q rows only where the lazy scan needs them, from
        ``hidden[i]`` (the engine's final hidden states: a [g, d] tensor, or
        ``HiddenRefs`` object-store block handles) and the LM-head stash
        of the version that drafted position j (``segments``). With handles,
        a block is fetched (``ray.get``, directly from the generation
        worker's node) only when the lazy scan first needs a q row in it:
        phase 1 (p, b, the whole-draft test) reads only log q of the drafted
        tokens, so a fully accepted draft fetches nothing.
        No scorer copy and no whole-model stash. Returns the same per-row dicts
        as ``verify_drafts_block_stashed``: "logprobs" are training's own
        log-probs of the kept tokens and "draft_logq" the engine's, so the
        behavior-log-prob correction is exactly zero; plus
        "hidden_bytes_fetched" (bytes this draft pulled from the object store).

        Tensor parallelism: every TP rank of a replica verifies the same rows
        (``_my_rows`` hashes by data-parallel rank) with the same generator
        seed and bitwise the same gathered p and q rows, so the lazy scans
        take the same decisions and call the row gathers in lockstep.

        Pipeline parallelism: every stage of a replica runs the same rows'
        forwards through the forward-only pipeline schedule
        (``_training_forward``); only the last stage holds logits (and the LM
        head and its stashes), so it alone runs the lazy scans and returns the
        results. Other stages return [] (the driver merges all ranks' rows).

        Expert parallelism: each training forward runs all-to-alls over the
        expert-parallel group, which spans data-parallel replicas with
        different row counts. The group agrees on the largest count
        (``_expert_parallel_lockstep`` with empty version plans: q needs no
        forward here) and short ranks pad with one-row dummy forwards
        (``_dummy_training_forward``, output discarded). The q rows (LM head
        only) and the lazy scan's vocab gathers involve no experts and only
        the tensor-parallel group, i.e. ranks of one replica that verify the
        same rows in the same order, so they need no lockstep across replicas.

        ``THUNDERSYNC_ENGINE_Q_CHECK=1``: also rebuild every q row of each draft
        and print how many drafted tokens' log q match the engine bitwise and
        the largest difference (expected: all, 0).
        """
        from megatron.core import parallel_state as ps
        from thundersync_rl.engine_q import (
            GatheredLogprobRows,
            HiddenRows,
            hidden_num_rows,
            verify_engine_q,
        )

        if not self.cfg["megatron_cfg"].get("batch_invariant_mode"):
            raise RuntimeError(
                "verify_source='engine' requires policy.megatron_cfg.batch_invariant_mode"
            )
        self._check_dropless_moe()
        tp = ps.get_tensor_model_parallel_world_size()
        last_stage = (
            ps.get_pipeline_model_parallel_world_size() == 1
            or ps.is_pipeline_last_stage(ignore_virtual=True)
        )
        gather = self._tp_vocab_gather() if last_stage else None
        self._prof = {}
        t_all = time.perf_counter()
        gen = None  # created on the logits' device (last stage only)
        check = os.environ.get("THUNDERSYNC_ENGINE_Q_CHECK") == "1"
        chk_max, chk_tok, chk_exact = 0.0, 0, 0
        results = []
        n_rows = 0
        null_state: dict[str, Any] = {}
        if null_groups is not None:
            from thundersync_rl.null_bv import NullGroup, NullPolicy

            # A group's rows all on one rank, decided in index order.
            own = [g if g is not None else k for g, k in zip(null_groups, keys)]
            mine = self._my_rows(rows, own)
            policy = NullPolicy(**(null_policy or {}))
            longest: dict[str, int] = {}
            count: dict[str, int] = {}
            for i in mine:
                grp = null_groups[i]
                if grp is not None:
                    longest[grp] = max(longest.get(grp, 0), rows[i].numel())
                    count[grp] = count.get(grp, 0) + 1
            # Groups ordered by their longest row (padding of the batched
            # forwards), rows of a group consecutive in index order.
            mine.sort(
                key=lambda i: (
                    longest.get(null_groups[i], rows[i].numel()),
                    null_groups[i] or "",
                    i,
                )
            )
            null_state = {
                grp: NullGroup(n, int((null_slots or {})[grp]), policy)
                for grp, n in count.items()
            }
        else:
            mine = self._my_rows(rows, keys)
        # Several drafts per training-shaped forward (padded as training pads a
        # micro-batch): batch-invariant kernels give each row the same logits
        # as a one-row forward, and per-call overhead dominates small rows.
        batches = self._row_batches([rows[i] for i in mine], batch_tokens)
        batches = [[mine[j] for j in b] for b in batches]
        lockstep = self._expert_parallel_lockstep([[] for _ in batches])
        n_dummy = 0 if lockstep is None else len(lockstep) - len(batches)

        def row_logits():
            for batch in batches:
                t0 = time.perf_counter()
                if anchors is not None:
                    lb = self._training_forward_rows(
                        [rows[i] for i in batch],
                        anchors=[anchors[i] for i in batch],
                    )
                elif len(batch) == 1:
                    lb = self._training_forward(rows[batch[0]])
                else:
                    lb = self._training_forward_rows([rows[i] for i in batch])
                self._prof["forward"] = (
                    self._prof.get("forward", 0.0) + time.perf_counter() - t0
                )
                for k, i in enumerate(batch):
                    yield i, (None if lb is None else lb[k : k + 1])
                del lb

        for i, logits in row_logits():
            n, plen = rows[i].numel(), prompt_lens[i]
            g = n - plen
            if logits is None:  # not the last pipeline stage
                continue
            dev = logits.device
            if gen is None:
                gen = torch.Generator(device=dev).manual_seed(
                    seed * 1_000_003 + ps.get_data_parallel_rank()
                )
            null_m, grp = 1.0, None if null_groups is None else null_groups[i]
            if grp is not None:
                m = null_state[grp].next_m()
                if m is None:
                    # Slots filled: never decided (the stopping rule depends on
                    # counts only, so dropping it biases nothing).
                    del logits
                    results.append(
                        (
                            i,
                            {
                                "accepted": 0,
                                "next": -1,
                                "logprobs": [],
                                "draft_logq": [],
                                "hidden_bytes_fetched": 0,
                                "null": "unused",
                            },
                        )
                    )
                    continue
                null_m = m
            if tp == 1:
                # Log-probs exactly as training computes them (fp32 log-softmax).
                p_lp = torch.log_softmax(
                    logits[0, plen - 1 : n, :vocab_limit].float(), -1
                )
            else:
                # Vocab-parallel logits: rows gathered on demand, as batch-
                # invariant training's gathered log-softmax computes them.
                p_lp = GatheredLogprobRows(logits[0, plen - 1 : n], gather, vocab_limit)
            del logits
            draft = rows[i][plen:].to(dev)
            lq = torch.tensor(engine_logq[i], dtype=torch.float32, device=dev)
            assert lq.numel() == g, (lq.numel(), g)
            assert hidden_num_rows(hidden[i]) == g, (hidden_num_rows(hidden[i]), g)
            # Phase 2 only: rows (per segment) fetched when the scan asks.
            h = HiddenRows(hidden[i])
            version = [0] * g
            for v, a, b in segments[i]:
                version[a:b] = [v] * (b - a)

            def q_rows(idx, h=h, version=version):
                idx_l = idx.tolist()
                out: list[Any] = [None] * len(idx_l)
                by_v: dict[int, list[int]] = {}
                for j, t in enumerate(idx_l):
                    by_v.setdefault(version[t], []).append(j)
                for v, js in by_v.items():
                    lp = self._head_logprobs(h([idx_l[j] for j in js]), v, vocab_limit)
                    for k, j in enumerate(js):
                        out[j] = lp[k]
                return torch.stack(out)

            tau, y, logp, n_eval = verify_engine_q(
                p_lp, draft, lq, q_rows, gen, null_m
            )
            null_info: dict[str, Any] = {}
            if grp is not None:
                null_state[grp].record(y >= 0)
                null_info = {"null": "out" if y >= 0 else "bot", "null_m": null_m}
            n_rows += n_eval + (tau < g)
            fetched = h.fetched_bytes  # what the scan needed (not the check)
            if check:
                for a in range(0, g, 512):
                    idx = torch.arange(a, min(g, a + 512))
                    rec = q_rows(idx).gather(-1, draft[idx.to(dev)].view(-1, 1))[:, 0]
                    d = (rec - lq[a : a + idx.numel()]).abs()
                    chk_max = max(chk_max, float(d.max()))
                    chk_exact += int((d == 0).sum())
                    chk_tok += idx.numel()
                    bad = (d > 0).nonzero().flatten()
                    if bad.numel() and os.environ.get("THUNDERSYNC_HIDDEN_DEBUG"):
                        pos = (bad + a).tolist()[:8]
                        print(
                            f"[engine q check] inexact positions {pos} of g={g} "
                            f"d={[round(float(d[b]), 2) for b in bad[:8]]} "
                            f"rec={[round(float(rec[b]), 2) for b in bad[:4]]} "
                            f"eng={[round(float(lq[a + int(b)]), 2) for b in bad[:4]]} "
                            f"eng_prev={[round(float(lq[a + int(b) - 1]), 2) if a + int(b) > 0 else None for b in bad[:4]]} "
                            f"eng_next={[round(float(lq[a + int(b) + 1]), 2) if a + int(b) + 1 < g else None for b in bad[:4]]} "
                            f"versions_at={[version[t] for t in pos]} segs={segments[i]}",
                            flush=True,
                        )
            results.append(
                (
                    i,
                    {
                        "accepted": tau,
                        "next": y,
                        "logprobs": logp.tolist(),
                        "draft_logq": lq[:tau].tolist(),
                        "hidden_bytes_fetched": fetched,
                        **null_info,
                    },
                )
            )
            del p_lp, h
        for _ in range(n_dummy):
            # Expert parallelism: out of rows, join the other replicas' forwards.
            self._dummy_training_forward()
        self._prof["ep_dummy_forwards"] = n_dummy
        self._prof["q_rows"] = n_rows
        self._log_mem("verify_drafts_block_engine")
        if check and last_stage:
            print(
                f"[engine q check] rank={self.rank} rows={len(results)} tok={chk_tok} "
                f"exact={chk_exact} max|dlogq|={chk_max:.3e}",
                flush=True,
            )
        if os.environ.get("THUNDERSYNC_SPEC_PROF"):
            ntok = sum(rows[i].numel() for i, _ in results)
            print(
                f"[spec prof] verify_engine rank={self.rank} rows={len(results)} "
                f"tok={ntok} total={time.perf_counter() - t_all:.3f} {self._prof}",
                flush=True,
            )
        return results

    def _check_dropless_moe(self) -> None:
        """Raise if MoE layers drop tokens (expert capacity factor set).

        With a capacity factor, whether a token reaches its experts depends on
        every other token routed to them in the same forward (other rows, other
        expert-parallel ranks' rows, lockstep dummies), so p of a row is not a
        function of the row and its weights: verification against it is not
        exact. Dropless MoE (capacity factor None, Megatron's default) only.
        """
        from megatron.core.utils import get_model_config

        try:
            config = get_model_config(self.model)
        except (AttributeError, RuntimeError):  # no Megatron config (fakes)
            return
        if getattr(config, "num_moe_experts", None) and (
            getattr(config, "moe_expert_capacity_factor", None) is not None
        ):
            raise RuntimeError(
                "verify_source='engine' requires dropless MoE "
                "(moe_expert_capacity_factor=None): token dropping makes a "
                "row's p depend on the other tokens in its forward"
            )

    # ---- Fused verification + training (batch-invariant kernels; fused_train.py) ----

    def _check_fused_verify_supported(self, state: dict[str, Any]) -> None:
        """Raise unless the verification forward can stand in for training's."""
        from megatron.core import parallel_state as ps

        problems = []
        if state["grad_stream"] is None:
            problems.append("the train step was opened without gradient streaming")
        if not self.cfg["megatron_cfg"].get("batch_invariant_mode"):
            problems.append("batch_invariant_mode is off")
        if state["mtp_enabled"]:
            problems.append("multi-token prediction needs the loss mask in the forward")
        if self.draft_model is not None:
            problems.append(
                "draft-model training captures hidden states in the forward"
            )
        if self.cfg["megatron_cfg"].get("use_fused_linear_logprobs"):
            problems.append("fused linear log-probs produce no logits")
        if ps.get_pipeline_model_parallel_world_size() != 1:
            problems.append("pipeline parallelism (the graph spans stages)")
        if ps.get_tensor_model_parallel_world_size() != 1:
            problems.append("tensor parallelism (training logits are vocab-sharded)")
        if ps.get_context_parallel_world_size() != 1:
            problems.append("context parallelism")
        if problems:
            raise RuntimeError(
                "fused verification + training unsupported: " + "; ".join(problems)
            )

    def _row_batches(
        self, rows: list[torch.Tensor], batch_tokens: int
    ) -> list[list[int]]:
        """Consecutive groups of ``rows`` whose padded micro-batch fits ``batch_tokens``.

        A group costs len(group) x its longest row padded to
        ``make_sequence_length_divisible_by``; a row longer than the budget
        forms its own group. ``batch_tokens <= 0`` gives one row per group.
        """
        m = self.cfg.get("make_sequence_length_divisible_by", 1)
        out: list[list[int]] = []
        longest = 0
        for j, row in enumerate(rows):
            padded = -(-row.numel() // m) * m
            if (
                out
                and batch_tokens > 0
                and (len(out[-1]) + 1) * max(longest, padded) <= batch_tokens
            ):
                out[-1].append(j)
                longest = max(longest, padded)
            else:
                out.append([j])
                longest = padded
        return out

    def _training_forward(
        self, row: torch.Tensor, anchors: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """Logits [1, S, V] of one row, computed exactly as a training forward."""
        if anchors is None:
            return self._training_forward_rows([row])
        return self._training_forward_rows([row], anchors=[anchors])

    def _anchor_batch(
        self, rows: list[torch.Tensor], anchors: list[Any] | None, width: int
    ) -> torch.Tensor | None:
        """``routed_experts`` [B, width, L, K] for a sticky-routing forward.

        None when sticky routing is off. Rows without anchors (and padding)
        get -1: anchor = self, i.e. plain routing for those tokens.
        """
        from nemo_rl.models.megatron import sticky_routing

        if sticky_routing.get_delta() is None or not self._router_replay_enabled:
            return None
        from nemo_rl.models.megatron.router_replay import router_replay_dimensions
        from megatron.core.utils import get_model_config

        layers, k = router_replay_dimensions(get_model_config(self.model))
        out = torch.full((len(rows), width, layers, k), -1, dtype=torch.int16)
        for b, a in enumerate(anchors or []):
            if a is None:
                continue
            n = min(int(a.shape[0]), width)
            out[b, :n] = a[:n].to(torch.int16)
        return out

    def _training_forward_rows(
        self, rows: list[torch.Tensor], anchors: list[Any] | None = None
    ) -> torch.Tensor | None:
        """Logits [B, S, V] of ``rows`` in one micro-batch, exactly as a training forward.

        Same padding (``make_sequence_length_divisible_by``), micro-batch
        processing and ``model_forward`` call as ``train_microbatch``; autograd
        follows the caller's grad mode.

        Pipeline parallelism: the same forward through Megatron's pipeline
        schedule, forward-only (``_pipeline_forward_only``), as training runs
        each micro-batch. Every stage must call it for the same row; the last
        stage returns the logits (vocab-parallel at TP>1), the others None.
        """
        from megatron.core import parallel_state as ps

        m = self.cfg["make_sequence_length_divisible_by"]
        lens = [row.numel() for row in rows]
        ids = torch.zeros((len(rows), -(-max(lens) // m) * m), dtype=torch.long)
        for b, row in enumerate(rows):
            ids[b, : lens[b]] = row
        data = BatchedDataDict({"input_ids": ids, "input_lengths": torch.tensor(lens)})
        routed = self._anchor_batch(rows, anchors, ids.shape[1])
        if routed is not None:
            # Sticky routing: the forward that defines p replays the anchors
            # through the training path's RouterReplay (REPLAY_FORWARD).
            data["routed_experts"] = routed
        attach_media_token_validity_mask(data, self.media_placeholder_token_id)
        iterator, _, _, _, padded_seq_length = get_microbatch_iterator(
            data,
            self.cfg,
            len(rows),
            straggler_timer=self.mcore_state.straggler_timer,
            delegate_pack_to_model=self.delegate_pack_to_model,
            delegate_mtp_loss_mask_to_model=self.delegate_mtp_loss_mask_to_model,
            model_slices_context_parallel_inputs=self.model_slices_context_parallel_inputs,
            mtp_enabled=self.mtp_enabled,
        )
        mb = next(iterator)

        def forward(model):
            if routed is None:
                return _model_forward(model)
            from nemo_rl.models.megatron.router_replay import (
                clear_router_replay,
                set_router_replay_forward,
            )

            set_router_replay_forward(model, mb.routed_experts_cp_sharded)
            try:
                return _model_forward(model)
            finally:
                clear_router_replay(model)

        def _model_forward(model):
            return megatron_train.model_forward(
                model=model,
                data_dict=mb.data_dict,
                input_ids_cp_sharded=mb.input_ids_cp_sharded,
                position_ids=mb.position_ids,
                attention_mask=mb.attention_mask,
                packed_seq_params=mb.packed_seq_params,
                defer_fp32_logits=self.defer_fp32_logits,
                mtp_loss_mask=mb.mtp_loss_mask,
                padding_mask=mb.padding_mask,
                straggler_timer=self.mcore_state.straggler_timer,
                use_fused_linear_logprobs=False,
                media_token_validity_mask=mb.media_token_validity_mask,
                model_slices_context_parallel_inputs=self.model_slices_context_parallel_inputs,
            )

        if ps.get_pipeline_model_parallel_world_size() == 1:
            return forward(self.model)
        return self._pipeline_forward_only(forward, padded_seq_length, len(rows))

    def _pipeline_forward_only(
        self, forward, seq_length: int, micro_batch_size: int = 1
    ) -> torch.Tensor | None:
        """``forward(model)`` of one micro-batch through the pipeline schedule.

        Megatron's ``get_forward_backward_func`` with ``forward_only=True`` and
        ``collect_non_loss_data=True`` (one micro-batch of size 1, sequence
        length ``seq_length`` as training passes it): each stage runs its layers
        on the activations received from the previous stage, and the last stage
        collects the output (logits). Returns them on the last stage, None on
        the others. No autograd: a graph across stages would need the backward
        schedule. The DDP ``no_sync`` hook is bypassed for the call (the
        forward-only schedule enters it without leaving it, and verification
        must not change gradient-sync state mid-step).
        """
        from megatron.core import parallel_state as ps
        from megatron.core.pipeline_parallel import get_forward_backward_func
        from megatron.core.utils import get_model_config

        if torch.is_grad_enabled():
            raise NotImplementedError(
                "autograd through a pipeline-parallel verification forward is "
                "not implemented (it needs the backward schedule)"
            )

        def step(data_iterator, model):
            next(data_iterator)
            return forward(model), lambda out, non_loss_data=True: out.detach()

        config = get_model_config(self.model)
        no_sync = config.no_sync_func
        config.no_sync_func = None
        try:
            out = get_forward_backward_func()(
                forward_step_func=step,
                data_iterator=iter([None]),
                model=self.model,
                num_microbatches=1,
                seq_length=seq_length,
                micro_batch_size=micro_batch_size,
                decoder_seq_length=seq_length,
                forward_only=True,
                collect_non_loss_data=True,
            )
        finally:
            config.no_sync_func = no_sync
        if not ps.is_pipeline_last_stage(ignore_virtual=True):
            return None
        return out[0]

    def _dummy_training_forward(self) -> None:
        """One-row training forward whose output is discarded (EP lockstep filler).

        A single token padded to ``make_sequence_length_divisible_by`` (a
        multiple of TP, so sequence parallelism splits it evenly), no autograd,
        through the pipeline schedule at PP>1: it joins the expert-parallel
        group's all-to-alls like a real row.
        """
        self._training_forward(torch.zeros(1, dtype=torch.long))

    def verify_drafts_block_fused(
        self,
        rows: list[torch.Tensor],
        prompt_lens: list[int],
        segments: list[list[tuple[int, int, int]]],
        vocab_limit: int,
        seed: int,
        keys: list[str],
        batch_tokens: int,
        precision: str,
        retain: list[bool],
        max_retained_tokens: int,
        lazy: bool = False,
    ) -> list[tuple[int, dict[str, Any]]]:
        """Block verification with p from the training model; keep training forwards.

        As ``verify_drafts_block_stashed`` (q from the stashed weights), but p
        is the training model's forward of each row at the current weights,
        run like a training forward inside the open train step. For a row with
        ``retain[i]`` (the driver would turn its full acceptance into a
        finished rollout) the forward runs with autograd; if verification
        accepts the whole draft and the rank's retained graphs stay within
        ``max_retained_tokens``, the graph is kept for the row's training call
        and the result carries ``"fused": {"key", "dp_rank"}``. Other graphs
        are freed right away.
        """
        from megatron.core import parallel_state as ps

        # Triton kernel: imported where it runs (GPU workers only).
        from thundersync_rl.block_verification_fused import (
            block_verify_logits,
            block_verify_logits_lazy,
        )

        # ``lazy``: same decisions from the lazy scan (no residual pass on
        # whole-accepted drafts).
        verify_fn = block_verify_logits_lazy if lazy else block_verify_logits

        state = self._assert_step_open()
        self._check_fused_verify_supported(state)
        self._prof = {}
        t_all = time.perf_counter()
        self._scorer_precision = precision
        scorer = self._fp32_scorer()
        self.model.train()  # the mode training runs in (the step is open)
        dp_rank = ps.get_data_parallel_rank()
        gen = torch.Generator(device="cuda").manual_seed(seed * 1_000_003 + dp_rank)
        results = []
        for group, ids in self._iter_row_id_batches(rows, batch_tokens, keys):
            pos = torch.arange(ids.shape[1], device="cuda").expand(len(group), -1)
            lq_rows = self._stashed_q_logits(
                scorer, group, ids, pos, prompt_lens, segments, vocab_limit
            )
            for r, i in enumerate(group):
                n, plen = rows[i].numel(), prompt_lens[i]
                g = n - plen
                keep = retain[i] and self._fused_graphs.can_retain(
                    n, max_retained_tokens
                )
                t0 = time.perf_counter()
                with torch.set_grad_enabled(keep):
                    logits = self._training_forward(rows[i])
                self._prof["forward"] = (
                    self._prof.get("forward", 0.0) + time.perf_counter() - t0
                )
                with torch.no_grad():
                    p_logits = logits[0, plen - 1 : n, :vocab_limit].detach().float()
                    parts = [t for _, t in sorted(lq_rows.pop(i), key=lambda x: x[0])]
                    q_logits = torch.cat(parts) if parts else p_logits[:0]
                    assert q_logits.shape[0] == g, (q_logits.shape, g)
                    draft = ids[r, plen:n]
                    tau, y, _ = verify_fn(p_logits, q_logits, draft, gen)
                    # Log-probs exactly as training computes them (fp32
                    # log-softmax + gather): generation log-probs of the plan
                    # equal the trainer's bitwise.
                    kept = torch.cat([draft[:tau], torch.tensor([y], device="cuda")])
                    logp = torch.log_softmax(p_logits[: tau + 1], -1).gather(
                        -1, kept.unsqueeze(-1)
                    )[:, 0]
                res: dict[str, Any] = {
                    "accepted": tau,
                    "next": y,
                    "logprobs": logp.tolist(),
                }
                if keep and tau == g:
                    self._fused_graphs.put(
                        keys[i],
                        FusedGraph(
                            ids=rows[i].detach().cpu().clone(), logits=logits[:, :n]
                        ),
                    )
                    res["fused"] = {"key": keys[i], "dp_rank": dp_rank}
                del logits, p_logits, q_logits
                results.append((i, res))
        self._log_mem("verify_drafts_block_fused")
        if os.environ.get("THUNDERSYNC_SPEC_PROF"):
            ntok = sum(rows[i].numel() for i in range(len(rows)))
            print(
                f"[spec prof] verify_fused rank={self.rank} rows={len(rows)} tok={ntok} "
                f"retained={len(self._fused_graphs)} "
                f"retained_tok={self._fused_graphs.retained_tokens} "
                f"total={time.perf_counter() - t_all:.3f} {self._prof}",
                flush=True,
            )
        return results

    @property
    def _batch_invariant(self) -> bool:
        """Whether the scorer runs batch-invariant kernels (bf16 scorer under BI).

        An fp32 scorer is built outside batch-invariant mode, so training-parity
        measures (full-row scoring, training's log-softmax) do not apply to it.
        """
        return bool(self.cfg["megatron_cfg"].get("batch_invariant_mode")) and (
            getattr(self, "_scorer_precision", "fp32") != "fp32"
        )

    def _score(self, scorer, ids, pos, select=None):
        """Scorer logits: [b, s, V], or [n, V] at ``select = (seq_idx, batch_idx)``."""
        from megatron.core import parallel_state as ps
        from thundersync_rl.scorer_kernels import select_positions, supports_selection

        if select is None:
            return self._score_full(scorer, ids, pos)
        # Batch-invariant GEMMs are invariant within a token-count shape class
        # only: score full rows, as training does, to reproduce its numerics.
        if not supports_selection(scorer) or self._batch_invariant:
            logits = self._score_full(scorer, ids, pos)
            return None if logits is None else logits[select[1], select[0]]
        with select_positions(scorer, select):
            logits = self._score_full(scorer, ids, pos)
        if logits is None:  # not the last pipeline stage
            return None
        if (
            os.environ.get("THUNDERSYNC_SELECT_CHECK")
            and ps.get_pipeline_model_parallel_world_size() == 1
        ):
            # Debug: selected logits vs the full forward (PP=1 only: a second
            # pipeline schedule on the last stage alone would deadlock).
            full = self._score_full(scorer, ids, pos)[select[1], select[0]]
            diff = torch.log_softmax(logits[0].float(), -1) - torch.log_softmax(
                full.float(), -1
            )
            print(
                f"[select check] n={full.shape[0]} max|dlogp|={diff.abs().max().item():.3e}",
                flush=True,
            )
        return logits[0]

    def _score_full(self, scorer, ids, pos):
        from megatron.core import parallel_state as ps

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        seq_len = ids.shape[1]
        m = self.cfg["make_sequence_length_divisible_by"]
        padded = (
            self._batch_invariant
            and ps.get_tensor_model_parallel_world_size() > 1
            and seq_len % m
        )
        if padded:
            # Batch-invariant scorer at TP>1: sequence parallelism splits the
            # sequence over TP ranks, and each rank's norm row count must stay
            # in training's shape class, so pad like training. Causal attention
            # leaves the logits of the real positions unchanged (sliced below).
            padded = -(-seq_len // m) * m
            ids = torch.nn.functional.pad(ids, (0, padded - seq_len))
            pos = torch.arange(padded, device=ids.device).expand(ids.shape[0], -1)
        # On generation workers Megatron's global InferenceMode is active while the
        # engine runs; it requires gathered TP logits from any forward.
        gather_kw = {}
        try:
            from megatron.core.inference.utils import InferenceMode

            if InferenceMode.is_active():
                gather_kw["runtime_gather_output"] = True
        except ImportError:
            pass
        if (
            os.environ.get("THUNDERSYNC_SPEC_TORCHPROF")
            and not getattr(self, "_profiled", False)
            and ps.get_pipeline_model_parallel_world_size() == 1
        ):
            # Debug: profile one extra forward (PP=1 only: outside the schedule).
            self._profiled = True
            from torch.profiler import ProfilerActivity, profile

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
            ) as prof:
                scorer(
                    input_ids=ids, position_ids=pos, attention_mask=None, **gather_kw
                )
                torch.cuda.synchronize()
            print(
                f"[spec torchprof] ids={tuple(ids.shape)}\n"
                + prof.key_averages(group_by_input_shape=True).table(
                    sort_by="self_cuda_time_total",
                    row_limit=25,
                    max_name_column_width=40,
                    max_shapes_column_width=90,
                ),
                flush=True,
            )
        if ps.get_pipeline_model_parallel_world_size() == 1:
            logits = scorer(
                input_ids=ids, position_ids=pos, attention_mask=None, **gather_kw
            ).float()
        else:
            # Run the pipeline schedule forward-only; logits exist on the last stage.
            from megatron.core.pipeline_parallel import get_forward_backward_func

            captured: list[torch.Tensor] = []

            def fwd_step(data_iterator, model):
                b_ids, b_pos = next(data_iterator)
                out = model(input_ids=b_ids, position_ids=b_pos, attention_mask=None)

                def collect(output_tensor):
                    captured.append(output_tensor.detach().float())
                    return torch.zeros((), device=output_tensor.device), {}

                return out, collect

            get_forward_backward_func()(
                forward_step_func=fwd_step,
                data_iterator=iter([(ids, pos)]),
                model=[scorer],
                num_microbatches=1,
                seq_length=ids.shape[1],
                micro_batch_size=ids.shape[0],
                forward_only=True,
            )
            if not ps.is_pipeline_last_stage(ignore_virtual=True):
                torch.cuda.synchronize()
                self._prof["forward"] = (
                    self._prof.get("forward", 0.0) + time.perf_counter() - t0
                )
                return None
            logits = captured[0]
        tp = ps.get_tensor_model_parallel_world_size()
        if tp > 1 and not gather_kw:
            # Vocab-parallel logits -> full vocab for this batch only (memory is
            # bounded by batch_tokens, not by the number of draft tokens).
            parts = [torch.empty_like(logits) for _ in range(tp)]
            torch.distributed.all_gather(
                parts, logits.contiguous(), group=ps.get_tensor_model_parallel_group()
            )
            logits = torch.cat(parts, dim=-1)
        if padded:
            # Only full-row forwards are padded (batch invariance never selects
            # positions); selected logits are [1, n_selected, V] and kept whole.
            logits = logits[:, :seq_len]
        torch.cuda.synchronize()
        self._prof["forward"] = (
            self._prof.get("forward", 0.0) + time.perf_counter() - t0
        )
        return logits

    def _expert_parallel_lockstep(
        self, plans: list[list[int]]
    ) -> list[list[int]] | None:
        """Common forward schedule of the expert-parallel group, or None without EP.

        ``plans[j]``: stash versions batch j of this rank scores q with (p follows).
        Every scorer forward runs an all-to-all over the EP group, and each rank
        holds only its own experts, so all EP ranks must run the same number of
        forwards with the same weights loaded: batch j scores, in order, the union
        of the ranks' versions for j, and ranks without rows there run a dummy.
        """
        from megatron.core import parallel_state as ps

        if ps.get_expert_model_parallel_world_size() <= 1:
            return None
        gathered: list[Any] = [None] * ps.get_expert_model_parallel_world_size()
        torch.distributed.all_gather_object(
            gathered, plans, group=ps.get_expert_model_parallel_group()
        )
        n = max(len(p) for p in gathered)
        return [
            sorted({v for p in gathered if j < len(p) for v in p[j]}) for j in range(n)
        ]

    def _dummy_score(self, scorer) -> None:
        """One-row scorer forward whose output is discarded (EP lockstep filler)."""
        n = self.cfg["make_sequence_length_divisible_by"]
        ids = torch.zeros((1, n), dtype=torch.long, device="cuda")
        pos = torch.arange(n, device="cuda")[None]
        zero = torch.zeros(1, dtype=torch.long, device="cuda")
        self._score(scorer, ids, pos, select=(zero, zero))

    def _my_rows(self, rows, keys) -> list[int]:
        """Indices of the rows this data-parallel rank owns, shortest first.

        Ownership is a stable hash of the draft key, so the tensor-parallel
        ranks of one replica process the same rows together.
        """
        import zlib

        from megatron.core import parallel_state as ps

        dp, dp_rank = ps.get_data_parallel_world_size(), ps.get_data_parallel_rank()
        return sorted(
            (
                i
                for i in range(len(rows))
                if zlib.crc32(keys[i].encode()) % dp == dp_rank
            ),
            key=lambda i: rows[i].numel(),
        )

    def _iter_row_id_batches(self, rows, batch_tokens, keys):
        """This rank's rows, length-sorted, as padded token-id batches.

        Rows are owned by data-parallel rank (stable hash of the key), so the
        tensor-parallel ranks of one replica process the same rows together.
        """
        mine = self._my_rows(rows, keys)
        b = 0
        while b < len(mine):
            e = b + 1
            while e < len(mine) and rows[mine[e]].numel() * (e + 1 - b) <= batch_tokens:
                e += 1
            group = mine[b:e]
            b = e
            s = max(rows[i].numel() for i in group)
            ids = torch.zeros((len(group), s), dtype=torch.long)
            for r, i in enumerate(group):
                ids[r, : rows[i].numel()] = rows[i]
            yield group, ids.cuda()

    @torch.no_grad()
    def scorer_token_logprobs(
        self, rows: list[torch.Tensor]
    ) -> list[torch.Tensor] | None:
        """Debug/validation: next-token logprobs of ``rows`` under the fp32 scorer.

        Uses the current weights; returned on last-pipeline-stage DP-rank-0 workers.
        """
        from megatron.core import parallel_state as ps

        self._prof = {}
        scorer = self._fp32_scorer()
        out = []
        for row in rows:
            ids = row.view(1, -1).cuda()
            pos = torch.arange(ids.shape[1], device="cuda").view(1, -1)
            logits = self._score(scorer, ids, pos)
            if logits is None:
                continue
            lp = torch.log_softmax(logits[0, :-1].float(), -1)
            out.append(lp.gather(-1, ids[0, 1:, None]).squeeze(-1).cpu())
        last = ps.is_pipeline_last_stage(ignore_virtual=True)
        return (
            out
            if (
                last
                and ps.get_data_parallel_rank() == 0
                and ps.get_tensor_model_parallel_rank() == 0
            )
            else None
        )

    @torch.no_grad()
    def model_param_checksum(self) -> float:
        """Debug: sum of the (refit target) model parameters, to compare pools."""
        return float(sum(p.detach().double().sum() for p in self.model.parameters()))

    def debug_refit_export_names(self) -> dict[str, Any]:
        """Debug: names (and a checksum) this rank would send at refit."""
        names, total = [], 0.0
        for name, t in self._iter_params_with_optional_kv_scales(kv_scales=None):
            names.append(name)
            total += float(t.detach().double().sum())
        layers = sorted(
            {int(n.split("layers.")[1].split(".")[0]) for n in names if "layers." in n}
        )
        return {
            "rank": self.rank,
            "n": len(names),
            "layers": (layers[:3], layers[-3:], len(layers)),
            "sum": total,
        }
