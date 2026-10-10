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

import os
import time
from typing import Any

import ray
import torch

from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)
from nemo_rl.algorithms.grad_streaming import GradStreamingSpec, StreamBucket

WORKER_FQN = "thundersync_rl.worker.ThunderSyncMegatronPolicyWorker"


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
    ) -> None:
        """Open an optimizer step whose gradient arrives in streamed chunks."""
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

    def stream_accumulate(
        self,
        items: list[dict[str, Any]],
        closes: list[tuple[Any, dict[float, float]]],
    ) -> dict[str, Any]:
        """Backprop streamed chunks, then fold any groups that closed.

        Each item is ``{"data": BatchedDataDict, "group": gid | None,
        "reward": float | None}``; ``group=None`` items carry final advantages.
        """
        t0 = time.perf_counter()
        for item in items:
            bucket = (
                None
                if item["group"] is None
                else StreamBucket(group=item["group"], reward=item["reward"])
            )
            # Chunks smaller than a micro-batch run as one micro-batch of exactly
            # their rows (no dummy rows); larger chunks arrive padded to a
            # multiple of the configured size and use it. Keeping the configured
            # size bounds memory and matches the synchronous trainer's numerics
            # (Megatron fp32 gradients shift ~1e-4 with micro-batch size).
            state = self._train_step_state
            step_mbs = state["mbs"]
            state["mbs"] = min(item["data"].size, step_mbs)
            try:
                self.train_microbatch(item["data"], stream_bucket=bucket)
            finally:
                state["mbs"] = step_mbs
        if closes:
            self.close_stream_groups(closes)
        torch.cuda.synchronize()
        self._stream_busy_s += time.perf_counter() - t0
        acc = self._train_step_state["grad_stream"]
        return {"num_items": len(items), "open_buffers": acc.num_open_buffers()}

    def stream_finish_step(self) -> dict[str, Any]:
        """Barrier: write back, normalize, DP-reduce, clip, step."""
        peak = self._train_step_state["grad_stream"].peak_open_buffers
        results = self.finish_train_step()
        results["stream_peak_open_buffers"] = peak
        results["stream_learner_busy_s"] = self._stream_busy_s
        return results

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

    # ---- Cross-iteration speculation experiments (policy forecasting) ----
    # Snapshots and linear combinations of this rank's fp32 master-weight
    # shards. Requires the distributed optimizer without the precision-aware
    # optimizer, so master weights live in ``shard_fp32_from_float16_groups``.

    def _master_shards(self) -> list[torch.Tensor]:
        shards = []
        for opt in getattr(self.optimizer, "chained_optimizers", [self.optimizer]):
            assert not opt.config.use_precision_aware_optimizer, (
                "forecasting needs master weights in the dist optimizer; set "
                "policy.megatron_cfg.optimizer.use_precision_aware_optimizer=false"
            )
            for group in opt.shard_fp32_from_float16_groups + opt.shard_fp32_groups:
                shards.extend(group)
        return shards

    def save_master_weights(self, tag: str) -> None:
        """Snapshot the fp32 master shards under ``tag``."""
        if not hasattr(self, "_master_snapshots"):
            self._master_snapshots: dict[str, list[torch.Tensor]] = {}
        self._master_snapshots[tag] = [s.detach().clone() for s in self._master_shards()]

    def rename_master_weights(self, src: str, dst: str) -> None:
        self._master_snapshots[dst] = self._master_snapshots.pop(src)

    @torch.no_grad()
    def load_master_combination(self, coeffs: dict[str, float]) -> None:
        """Set master weights to sum_tag coeff * snapshot[tag] and refresh the model.

        Copies master shards into the model's param buffers and all-gathers them
        across DP, exactly as the optimizer does after a step.
        """
        for i, shard in enumerate(self._master_shards()):
            acc = torch.zeros_like(shard)
            for tag, c in coeffs.items():
                acc.add_(self._master_snapshots[tag][i], alpha=c)
            shard.copy_(acc)
        for opt in getattr(self.optimizer, "chained_optimizers", [self.optimizer]):
            opt._copy_main_params_to_model_params()
            opt.start_param_sync_for_bucket_group_subset()
        torch.cuda.synchronize()

    @torch.no_grad()
    def save_adam_forecast(self, tag: str) -> dict[str, float]:
        """Snapshot the next Adam(W) step's weights assuming a zero gradient.

        theta_hat = theta - lr * (m_hat / (sqrt(v_hat) + eps) + wd * theta), with
        m, v decayed one step and bias-corrected at t + 1: the part of the next
        update the optimizer state already determines. Call before the step.
        """
        shards, n_missing = [], 0
        for opt in getattr(self.optimizer, "chained_optimizers", [self.optimizer]):
            inner = opt.optimizer
            for group in inner.param_groups:
                b1, b2 = group["betas"]
                for p in group["params"]:
                    st = inner.state.get(p, {})
                    if "exp_avg" not in st:
                        n_missing += 1
                        shards.append((p, p.detach().clone()))
                        continue
                    t = int(st.get("step", group.get("step", 0))) + 1
                    m_hat = st["exp_avg"].float() * b1 / (1 - b1**t)
                    v_hat = st["exp_avg_sq"].float() * b2 / (1 - b2**t)
                    upd = m_hat / (v_hat.sqrt() + group["eps"])
                    upd.add_(p, alpha=group["weight_decay"])
                    shards.append((p, p - group["lr"] * upd))
        # Order must match _master_shards(): map by identity.
        by_id = {id(p): hat for p, hat in shards}
        self._master_snapshots[tag] = [by_id[id(s)] for s in self._master_shards()]
        return {"params_without_state": float(n_missing)}

    @torch.no_grad()
    def master_distances(self, ref: str, others: list[str]) -> dict[str, float]:
        """fp32 L2 distance and bf16 disagreement count of snapshots vs ``ref``."""
        out: dict[str, float] = {}
        refs = self._master_snapshots[ref]
        out["numel"] = float(sum(r.numel() for r in refs))
        for tag in others:
            sq = flips = 0.0
            for a, b in zip(self._master_snapshots[tag], refs):
                sq += float((a - b).double().pow(2).sum())
                flips += float((a.bfloat16() != b.bfloat16()).sum())
            out[f"{tag}/l2"] = sq**0.5
            out[f"{tag}/bf16_flips"] = flips
        return out

    def drop_master_weights(self, tag: str) -> None:
        getattr(self, "_master_snapshots", {}).pop(tag, None)

    @torch.no_grad()
    def coupled_samples(
        self, rows: list[torch.Tensor], seed: int, chunk: int = 256
    ) -> torch.Tensor | None:
        """Teacher-forced shared-randomness samples under the current weights.

        For each row and position j (predicting token j+1) returns the token each
        coupling scheme would emit given a fixed per-(row, position) random
        draw: [:, :, 0] inverse CDF in token-id order, [:, :, 1] inverse CDF in
        descending-probability order, [:, :, 2] Gumbel-max. Two calls with
        different weights and the same ``seed`` disagree exactly where a
        drafter/verifier pair would. Returns a [num_rows, max_len, 3] int32
        tensor on rank 0, None elsewhere.
        """
        self.model.eval()
        max_len = max(int(r.numel()) for r in rows)
        out = torch.full((len(rows), max_len, 3), -1, dtype=torch.int32)
        for i, ids in enumerate(rows):
            ids = ids.cuda().view(1, -1)
            s = ids.shape[1]
            pos = torch.arange(s, device="cuda").view(1, -1)
            logits = self.model(input_ids=ids, position_ids=pos, attention_mask=None)
            logits = logits[0].float()  # [s, V]
            g_u = torch.Generator(device="cuda").manual_seed(seed * 1_000_003 + i)
            u = torch.rand(s, device="cuda", generator=g_u)
            g_g = torch.Generator(device="cuda").manual_seed(seed * 1_000_033 + i)
            for a in range(0, s, chunk):
                lg = logits[a : a + chunk]
                p = torch.softmax(lg, dim=-1)
                uu = u[a : a + chunk].unsqueeze(-1)
                cdf = torch.cumsum(p, dim=-1)
                x_id = torch.searchsorted(cdf, uu).clamp(max=p.shape[-1] - 1)
                ps, idx = torch.sort(p, dim=-1, descending=True)
                x_s = torch.searchsorted(torch.cumsum(ps, dim=-1), uu).clamp(
                    max=p.shape[-1] - 1
                )
                x_sorted = idx.gather(-1, x_s)
                gumbel = -torch.log(
                    -torch.log(
                        torch.rand(lg.shape, device="cuda", generator=g_g).clamp_min(
                            1e-20
                        )
                    )
                )
                x_g = torch.argmax(torch.log_softmax(lg, -1) + gumbel, dim=-1, keepdim=True)
                out[i, a : a + lg.shape[0]] = torch.cat(
                    [x_id, x_sorted, x_g], dim=-1
                ).int().cpu()
        return out if self.rank == 0 else None

    # ---- Exact cross-iteration speculative rollouts (keyed sampling) ----

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

        def keyed_kernel(logits, n, context, *, gather_indices=None,
                         token_to_request_index=None, output=None, **kw):
            if token_to_request_index is not None:
                return orig_kernel(logits, n, context, gather_indices=gather_indices,
                                   token_to_request_index=token_to_request_index,
                                   output=output, **kw)
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
                    (float(getattr(sp, "draft_variant", 0)), float(getattr(sp, "variant_eps", 0.0)))
                )
            if os.environ.get("THUNDERSYNC_SPEC_DEBUG"):
                ctx_len = context.get_active_sequence_lengths()[:n].tolist()
                pre = context.request_in_prefill_status_tensor[lo:hi].tolist()[:n]
                for j, rid in enumerate(context.request_ids[lo:hi].tolist()[:n]):
                    if seeds[j] == 1 and positions[j] - len(engine.get_request(rid).prompt_tokens) < 8:
                        print(f"[keyed dbg] call ctx_len={ctx_len[j]} prefill={pre[j]} "
                              f"req_pos={positions[j]} prompt={len(engine.get_request(rid).prompt_tokens)}",
                              flush=True)
            if all(s < 0 for s in seeds):
                return orig_kernel(logits, n, context, gather_indices=gather_indices,
                                   output=output, **kw)
            if all(s >= 0 for s in seeds):
                out = output if output is not None else torch.empty(
                    n, device=logits.device, dtype=torch.int64
                )
            else:
                out = orig_kernel(logits, n, context, gather_indices=gather_indices,
                                  output=output, **kw)
            rows = logits[gather_indices[:n]] if gather_indices is not None else logits[:n]
            # Pinned host tensors + non-blocking copies: no per-step stream sync,
            # so the engine's async scheduling overlap is preserved.
            seed_c = torch.tensor(seeds, dtype=torch.long).pin_memory()
            pos_c = torch.tensor(positions, dtype=torch.long).pin_memory()
            jit_c = torch.tensor(jitter, dtype=torch.float64).pin_memory()
            if all(s >= 0 for s in seeds):
                out.copy_(keyed_sample_logits(rows, seed_c, pos_c, vocab_limit, jit_c))
            else:
                idx = [j for j, s in enumerate(seeds) if s >= 0]
                idx_t = torch.tensor(idx, dtype=torch.long).pin_memory().to(rows.device, non_blocking=True)
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
    def verify_drafts(
        self,
        rows: list[torch.Tensor],
        prompt_lens: list[int],
        seeds: list[int],
        vocab_limit: int,
        head_k: int,
        batch_tokens: int = 32768,
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

        self.model.eval()
        world = torch.distributed.get_world_size()
        mine = sorted(
            (i for i in range(len(rows)) if i % world == self.rank),
            key=lambda i: rows[i].numel(),
        )
        results = []
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
            ids = ids.cuda()
            pos = torch.arange(s, device="cuda").expand(len(group), -1)
            logits = self.model(input_ids=ids, position_ids=pos, attention_mask=None)
            for r, i in enumerate(group):
                n, plen = rows[i].numel(), prompt_lens[i]
                lp = torch.log_softmax(logits[r, plen - 1 : n, :vocab_limit].float(), -1)
                positions = torch.arange(plen, n + 1, device="cuda") + position_shift
                emitted = keyed_sample_logits(
                    lp, torch.full_like(positions, seeds[i]), positions
                )
                draft = ids[r, plen:n]
                agree = emitted[: draft.numel()] == draft
                n_acc = int(agree.long().cumprod(0).sum())
                kept = torch.cat([draft[:n_acc], emitted[n_acc : n_acc + 1]])
                lps = lp[torch.arange(n_acc + 1, device="cuda"), kept].tolist()
                results.append(
                    (i, {"accepted": n_acc, "next": int(emitted[n_acc]), "logprobs": lps})
                )
            del logits
        return results

    # ---- Block verification (Sun et al. 2024) with learner-side q ----

    def _fp32_scorer(self):
        """Megatron copy of the policy with fp32 params/activations.

        Built from a deep copy of the training model provider (same architecture
        and layer spec, fp32 dtypes, no DDP/optimizer) and refreshed by copying
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
                provider.params_dtype = torch.float32
                provider.pipeline_dtype = torch.float32
                provider.bf16 = False
                provider.fp16 = False
            provider.recompute_granularity = None
            provider.recompute_method = None
            provider.recompute_num_layers = None
            # The fused bias-SwiGLU kernel is ~4x off the memory bound in fp32;
            # the unfused path is replaced by a Triton SwiGLU below.
            provider.bias_activation_fusion = False
            provider.finalize()
            self._mfp32 = provider.provide().cuda().eval()
            if provider.params_dtype == torch.float32:
                # Megatron/TE fall back to unfused fp32 attention and GLU paths.
                from thundersync_rl.scorer_kernels import use_fused_swiglu, use_sdpa_attention

                use_sdpa_attention(self._mfp32, provider)
                use_fused_swiglu(self._mfp32)
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

    def _iter_row_batches(
        self, rows: list[torch.Tensor], batch_tokens: int, keys: list[str] | None = None
    ):
        """This rank's rows, length-sorted, as padded batches.

        Rows are owned by i % world, or by a stable hash of ``keys[i]`` so a draft
        scored across several calls always lands on the same rank.
        """
        import zlib

        world = torch.distributed.get_world_size()

        def owner(i):
            return (zlib.crc32(keys[i].encode()) if keys is not None else i) % world

        mine = sorted(
            (i for i in range(len(rows)) if owner(i) == self.rank),
            key=lambda i: rows[i].numel(),
        )
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
            ids = ids.cuda()
            pos = torch.arange(s, device="cuda").expand(len(group), -1)
            torch.cuda.synchronize()
            t_fwd = time.perf_counter()
            scorer = getattr(self, "_active_scorer", None)
            if scorer is not None:
                prev = torch.get_float32_matmul_precision()
                torch.set_float32_matmul_precision("high" if self._scorer_tf32 else "highest")
                try:
                    if os.environ.get("THUNDERSYNC_SPEC_TORCHPROF") and not getattr(self, "_profiled", False):
                        self._profiled = True
                        from torch.profiler import ProfilerActivity, profile

                        with profile(activities=[ProfilerActivity.CUDA]) as prof:
                            scorer(input_ids=ids, position_ids=pos, attention_mask=None)
                            torch.cuda.synchronize()
                        print("[spec torchprof]\n" + prof.key_averages().table(
                            sort_by="cuda_time_total", row_limit=14), flush=True)
                    logits = scorer(
                        input_ids=ids, position_ids=pos, attention_mask=None
                    ).float()
                finally:
                    torch.set_float32_matmul_precision(prev)
            else:
                logits = self.model(input_ids=ids, position_ids=pos, attention_mask=None)
            torch.cuda.synchronize()
            self._prof["forward"] = self._prof.get("forward", 0.0) + time.perf_counter() - t_fwd
            self._prof["tokens"] = self._prof.get("tokens", 0) + int(ids.numel())
            yield group, ids, logits
            del logits

    @torch.no_grad()
    def score_drafts_q(
        self,
        rows: list[torch.Tensor],
        prompt_lens: list[int],
        vocab_limit: int,
        batch_tokens: int = 16384,
        precision: str = "model",
        keys: list[str] | None = None,
        from_lens: list[int] | None = None,
    ) -> int:
        """Learner at theta_k (before its step): keep each draft's full log q.

        Stores log q(. | prompt, draft[:i]) for i = 0..len(draft)-1 in bf16 on GPU
        for ``verify_drafts_block`` at the next iteration.
        """
        self._prof = {}
        t_all = time.perf_counter()
        self.model.eval()
        # Keyed mode appends the q of tokens drafted since the last deadline
        # (from_lens[i] onwards) to what earlier iterations stored: each
        # position keeps the q of the weights that actually drafted it.
        if keys is None:
            self._spec_q = {}
        elif not hasattr(self, "_spec_q"):
            self._spec_q = {}
        self._scorer_tf32 = precision == "tf32"
        self._active_scorer = self._fp32_scorer() if precision in ("fp32", "tf32") else None
        for group, ids, logits in self._iter_row_batches(rows, batch_tokens, keys):
            for r, i in enumerate(group):
                n, plen = rows[i].numel(), prompt_lens[i]
                start = plen - 1 + (from_lens[i] if from_lens is not None else 0)
                lq = torch.log_softmax(logits[r, start : n - 1, :vocab_limit].float(), -1)
                lq = lq.to(torch.bfloat16)
                key = keys[i] if keys is not None else i
                prev = self._spec_q.get(key) if keys is not None else None
                self._spec_q[key] = lq if prev is None else torch.cat([prev, lq])
        self._active_scorer = None
        if os.environ.get("THUNDERSYNC_SPEC_PROF"):
            print(f"[spec prof] score rank={self.rank} total={time.perf_counter() - t_all:.3f} {self._prof}", flush=True)
        return len(self._spec_q)

    @torch.no_grad()
    def verify_drafts_block(
        self,
        rows: list[torch.Tensor],
        prompt_lens: list[int],
        vocab_limit: int,
        seed: int,
        batch_tokens: int = 16384,
        precision: str = "model",
        keys: list[str] | None = None,
    ) -> list[tuple[int, dict[str, Any]]]:
        """Learner at theta_{k+1}: block-verify each draft against the stored q.

        Keeps X[:tau] and emits Y: the bonus token from p if the whole draft is
        accepted, else a sample from the block residual max(b_tau p - q, 0).
        Output is distributed exactly as p given that q is the draft's sampling
        distribution (Sun et al. 2024, "Block Verification Accelerates
        Speculative Decoding").
        """
        self._prof = {}
        t_all = time.perf_counter()
        self.model.eval()
        from thundersync_rl.block_verification import block_verify

        gen = torch.Generator(device="cuda").manual_seed(seed * 1_000_003 + self.rank)
        results = []
        self._scorer_tf32 = precision == "tf32"
        self._active_scorer = self._fp32_scorer() if precision in ("fp32", "tf32") else None
        for group, ids, logits in self._iter_row_batches(rows, batch_tokens, keys):
            for r, i in enumerate(group):
                n, plen = rows[i].numel(), prompt_lens[i]
                g = n - plen
                lp = torch.log_softmax(logits[r, plen - 1 : n, :vocab_limit].float(), -1)
                lq = self._spec_q.pop(keys[i] if keys is not None else i).float()  # [g, V]
                assert lq.shape[0] == g, (lq.shape, g)
                draft = ids[r, plen:n]
                tau, y = block_verify(lp, lq, draft, gen)
                kept = torch.cat([draft[:tau], torch.tensor([y], device="cuda")])
                lps = lp[torch.arange(tau + 1, device="cuda"), kept].tolist()
                results.append((i, {"accepted": tau, "next": y, "logprobs": lps}))
        self._active_scorer = None
        if os.environ.get("THUNDERSYNC_SPEC_PROF"):
            print(f"[spec prof] verify rank={self.rank} total={time.perf_counter() - t_all:.3f} {self._prof}", flush=True)
        return results

    def termination_id(self) -> int | None:
        tok = getattr(self, "megatron_tokenizer", None)
        return None if tok is None else int(tok.eod)

    # ---- Deadline-truncated streaming drafts (generation rank 0) ----

    def start_drafts(
        self, prompts: list[list[int]], max_new_tokens: int | list[int], stream_interval: int
    ) -> bool:
        """Submit draft requests as streams and return immediately.

        Tokens accumulate in ``self._draft_tokens`` as partial frames arrive;
        ``collect_drafts`` aborts whatever is unfinished at the deadline. Any
        prefix of a draft is a valid draft, so truncation is harmless.
        """
        import asyncio

        if self.inference_client is None or self._inference_loop is None:
            return False
        # Fresh buffers per call, captured by the coroutines below: streams aborted
        # at an earlier deadline may still deliver late frames, which must not
        # land in this round's drafts.
        tokens = [[] for _ in prompts]
        done = [False] * len(prompts)
        streams = {}
        self._draft_tokens, self._draft_done, self._draft_streams = tokens, done, streams

        async def run_one(i, prompt):
            sp = self._build_sampling_params(greedy=False, stop_words=None)
            sp.num_tokens_to_generate = (
                max_new_tokens[i] if isinstance(max_new_tokens, list) else max_new_tokens
            )
            sp.streaming_interval = stream_interval
            stream = self.inference_client.add_request_streaming(prompt, sp)
            streams[i] = stream
            async for frame in stream:
                if "partial" in frame:
                    tokens[i].extend(frame["partial"]["new_tokens"])
                elif "final" in frame:
                    final = frame["final"]
                    toks = (
                        final["generated_tokens"]
                        if isinstance(final, dict)
                        else final.generated_tokens
                    )
                    tokens[i][:] = list(toks)
                    done[i] = True

        for i, p in enumerate(prompts):
            asyncio.run_coroutine_threadsafe(run_one(i, list(p)), self._inference_loop)
        return True

    def collect_drafts(self) -> list[tuple[list[int], bool]]:
        """Abort unfinished drafts; return (tokens, finished) per draft."""
        import asyncio

        tokens, done, streams = self._draft_tokens, self._draft_done, self._draft_streams

        async def abort_and_snapshot():
            # Snapshot on the loop thread so no frame is half-applied.
            for i, stream in list(streams.items()):
                if not done[i]:
                    self.inference_client.abort_request(stream.request_id)
            return [(list(t), d) for t, d in zip(tokens, done)]

        return asyncio.run_coroutine_threadsafe(
            abort_and_snapshot(), self._inference_loop
        ).result()

    # ---- Memory-scalable block verification: stash weights, not q ----
    # q is recomputed at verification time from a bf16 stash of the weights that
    # drafted each position (2 bytes/param per model-parallel shard per stashed
    # version), so memory is O(params + one batch x vocab) instead of
    # O(all draft tokens x vocab).

    @torch.no_grad()
    def stash_weights(self, version: int, keep: list[int], precision: str = "fp32") -> int:
        """Stash the current training parameters as ``version``; drop others not in ``keep``."""
        self._prof = {}
        self._scorer_precision = precision
        self._fp32_scorer()  # builds the scorer and its parameter map once
        if not hasattr(self, "_weight_stash"):
            self._weight_stash: dict[int, list[torch.Tensor]] = {}
        self._weight_stash[version] = [src.detach().clone() for _, src in self._mfp32_map]
        for v in list(self._weight_stash):
            if v not in keep and v != version:
                del self._weight_stash[v]
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
    def verify_drafts_block_stashed(
        self,
        rows: list[torch.Tensor],
        prompt_lens: list[int],
        segments: list[list[tuple[int, int, int]]],
        vocab_limit: int,
        seed: int,
        keys: list[str],
        batch_tokens: int = 16384,
        precision: str = "fp32",
    ) -> list[tuple[int, dict[str, Any]]]:
        """Block-verify drafts whose q is recomputed per batch from stashed weights.

        ``segments[i]``: [(version, start, end)] - draft tokens [start, end) of row
        i were drafted by stashed weights ``version``. p is the current weights.
        """
        from thundersync_rl.block_verification import block_verify

        self._prof = {}
        t_all = time.perf_counter()
        self.model.eval()
        self._scorer_precision = precision
        scorer = self._fp32_scorer()
        from megatron.core import parallel_state as ps

        # Same stream on every TP rank of a replica: identical decisions.
        gen = torch.Generator(device="cuda").manual_seed(
            seed * 1_000_003 + ps.get_data_parallel_rank()
        )
        results = []
        for group, ids in self._iter_row_id_batches(rows, batch_tokens, keys):
            pos = torch.arange(ids.shape[1], device="cuda").expand(len(group), -1)
            lq_rows: dict[int, list[tuple[int, torch.Tensor]]] = {i: [] for i in group}
            versions = sorted({v for i in group for v, _, _ in segments[i]})
            for v in versions:
                self._load_scorer(v)
                logits = self._score(scorer, ids, pos)
                if logits is None:  # not the last pipeline stage
                    continue
                for r, i in enumerate(group):
                    plen = prompt_lens[i]
                    for sv, a, b in segments[i]:
                        if sv == v and b > a:
                            lq = torch.log_softmax(
                                logits[r, plen - 1 + a : plen - 1 + b, :vocab_limit].float(), -1
                            )
                            lq_rows[i].append((a, lq))
                del logits
            self._load_scorer(None)
            logits = self._score(scorer, ids, pos)
            if logits is None:
                continue
            for r, i in enumerate(group):
                n, plen = rows[i].numel(), prompt_lens[i]
                g = n - plen
                lp = torch.log_softmax(logits[r, plen - 1 : n, :vocab_limit].float(), -1)
                parts = [t for _, t in sorted(lq_rows.pop(i), key=lambda x: x[0])]
                lq = torch.cat(parts) if parts else lp[:0]
                assert lq.shape[0] == g, (lq.shape, g)
                draft = ids[r, plen:n]
                tau, y = block_verify(lp, lq, draft, gen)
                kept = torch.cat([draft[:tau], torch.tensor([y], device="cuda")])
                lps = lp[torch.arange(tau + 1, device="cuda"), kept].tolist()
                results.append((i, {"accepted": tau, "next": y, "logprobs": lps}))
            del logits
        if os.environ.get("THUNDERSYNC_SPEC_PROF"):
            pool = "gen" if getattr(self, "dynamic_inference_engine", None) is not None else "learner"
            ntok = sum(rows[i].numel() for i in range(len(rows)))
            print(f"[spec prof] verify_stashed pool={pool} rank={self.rank} rows={len(rows)} tok={ntok} total={time.perf_counter() - t_all:.3f} {self._prof}", flush=True)
        return results

    def _score(self, scorer, ids, pos):
        from megatron.core import parallel_state as ps

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # On generation workers Megatron's global InferenceMode is active while the
        # engine runs; it requires gathered TP logits from any forward.
        gather_kw = {}
        try:
            from megatron.core.inference.utils import InferenceMode

            if InferenceMode.is_active():
                gather_kw["runtime_gather_output"] = True
        except ImportError:
            pass
        if os.environ.get("THUNDERSYNC_SPEC_TORCHPROF") and not getattr(self, "_profiled", False):
            self._profiled = True
            from torch.profiler import ProfilerActivity, profile

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
            ) as prof:
                scorer(input_ids=ids, position_ids=pos, attention_mask=None, **gather_kw)
                torch.cuda.synchronize()
            print(f"[spec torchprof] ids={tuple(ids.shape)}\n" + prof.key_averages(
                group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=25,
                max_name_column_width=40, max_shapes_column_width=90), flush=True)
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
                self._prof["forward"] = self._prof.get("forward", 0.0) + time.perf_counter() - t0
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
        torch.cuda.synchronize()
        self._prof["forward"] = self._prof.get("forward", 0.0) + time.perf_counter() - t0
        return logits

    def _iter_row_id_batches(self, rows, batch_tokens, keys):
        """Like _iter_row_batches but yields token ids only (no forward).

        Rows are owned by data-parallel rank (stable hash of the key), so the
        tensor-parallel ranks of one replica process the same rows together.
        """
        import zlib

        from megatron.core import parallel_state as ps

        dp, dp_rank = ps.get_data_parallel_world_size(), ps.get_data_parallel_rank()
        mine = sorted(
            (i for i in range(len(rows)) if zlib.crc32(keys[i].encode()) % dp == dp_rank),
            key=lambda i: rows[i].numel(),
        )
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
    def scorer_token_logprobs(self, rows: list[torch.Tensor]) -> list[torch.Tensor] | None:
        """Debug/validation: next-token logprobs of ``rows`` under the fp32 scorer
        (current weights), returned on last-pipeline-stage DP-rank-0 workers."""
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
        return out if (last and ps.get_data_parallel_rank() == 0 and ps.get_tensor_model_parallel_rank() == 0) else None

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
        layers = sorted({int(n.split("layers.")[1].split(".")[0]) for n in names if "layers." in n})
        return {"rank": self.rank, "n": len(names), "layers": (layers[:3], layers[-3:], len(layers)), "sum": total}
