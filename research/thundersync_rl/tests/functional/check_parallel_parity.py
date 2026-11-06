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
"""GPU check: engine log-probs == learner log-probs, bitwise, at TP / EP / PP > 1.

torchrun with world = TP x PP x DP ranks (DP = 1 for the engine: every rank holds the
same requests; EP is carved out of TP x DP, expert TP 1, as the BI recipe requires).
Same recipe as the in-loop engine and learner (check_precision_raise_parity.py, which is
the one-GPU version): BI kernels, fp32 residual, fp16 GEMM forward, fp32 hi/lo LM head,
engine weights held in fp16, BI Mamba decode, Megatron dynamic inference with optional
CUDA graphs; learner = TE layers, sequence parallel at TP > 1, the learner's vocab-parallel
log-prob op (``model_utils.from_parallel_logits_to_logprobs`` = the BI gathered path at
TP > 1). Every generated token's log-prob must be bitwise equal, in a padded bshd batch and
(``--thd 1``) a packed THD batch.

  torchrun --nproc-per-node 4 check_parallel_parity.py --model <L8> --tp 4 --out r.json
  --tp_reduce stock|ordered : the row-parallel fp32 reduce-scatter (NEMO_RL_TP_REDUCE).

Writes <out>.lp.pt (rank 0) with engine and learner log-probs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

os.environ.setdefault("CUBLASLT_WORKSPACE_SIZE", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":0:0")
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

RANK = int(os.environ.get("RANK", "0"))


def log(*a):
    if RANK == 0 or os.environ.get("PARITY_ALL_RANKS") == "1":
        print(time.strftime("[%H:%M:%S]"), f"r{RANK}", *a, flush=True)


def provider(args, inference: bool, graphs: bool, ddp: bool = False, pre_wrap_hook=None):
    from megatron.bridge import AutoBridge
    from megatron.core.transformer.enums import AttnBackend

    prov = AutoBridge.from_hf_pretrained(args.model, torch_dtype=torch.bfloat16,
                                         trust_remote_code=True).to_megatron_provider(load_weights=True)
    prov.tensor_model_parallel_size = args.tp
    prov.pipeline_model_parallel_size = args.pp if not inference else args.engine_pp
    prov.expert_model_parallel_size = args.ep
    prov.expert_tensor_parallel_size = 1
    prov.sequence_parallel = args.tp > 1
    if prov.pipeline_model_parallel_size > 1:
        prov.pipeline_dtype = torch.float32
    prov.params_dtype = torch.bfloat16
    prov.bf16, prov.fp16 = True, False
    prov.gradient_accumulation_fusion = False
    prov.fp32_residual_connection = True
    prov.moe_router_dtype = "fp32"
    prov.moe_permute_fusion = False
    prov.attention_backend = AttnBackend.flash
    prov.flash_attention_version = 4
    prov.batch_invariant_mode = True
    prov.batch_invariant_backend = "te_native"
    prov.batch_invariant_collective = "ordered"
    prov.bias_activation_fusion = False
    prov.moe_shared_expert_overlap = False
    if hasattr(prov, "inference_fuse_tp_communication"):
        prov.inference_fuse_tp_communication = False
    if hasattr(prov, "tp_comm_overlap"):
        prov.tp_comm_overlap = False
    if getattr(prov, "moe_router_num_groups", None) == 1:
        prov.moe_router_num_groups = prov.moe_router_group_topk = None
    if args.layers:
        prov.num_layers = args.layers
    if inference:
        prov.transformer_impl = "inference_optimized"
        prov.inference_moe_token_dispatcher_type = args.engine_dispatcher
        if graphs:
            from megatron.core.transformer.enums import InferenceCudaGraphScope

            prov.cuda_graph_impl = "local"
            prov.use_te_rng_tracker = True
            prov.inference_cuda_graph_scope = InferenceCudaGraphScope["block"]
    else:
        prov.moe_token_dispatcher_type = "alltoall"
    prov.finalize()
    if ddp:
        from megatron.core.distributed import DistributedDataParallelConfig

        m = prov.provide_distributed_model(
            ddp_config=DistributedDataParallelConfig(
                grad_reduce_in_fp32=True, overlap_grad_reduce=False, use_distributed_optimizer=False),
            wrap_with_ddp=True, pre_wrap_hook=pre_wrap_hook)
        return prov, m
    m = prov.provide_distributed_model(wrap_with_ddp=False, pre_wrap_hook=pre_wrap_hook)
    for c in m:
        c.eval()
    return prov, m


def build_engine(args, model):
    from megatron.bridge.training.config import TokenizerConfig
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
    from megatron.core.inference.config import (AsyncScheduleMode, InferenceConfig, KVCacheManagementMode,
                                                MambaInferenceStateConfig)
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
    from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
    from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
    from megatron.core.inference.text_generation_controllers.text_generation_controller import (
        TextGenerationController)
    from megatron.core.utils import get_attr_wrapped_model, unwrap_model

    from nemo_rl.models.generation.megatron.megatron_worker import _pin_rotary_tables

    tok = build_tokenizer(TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=args.model,
                                          hf_tokenizer_kwargs={"trust_remote_code": True, "use_fast": True}),
                          make_vocab_size_divisible_by=128, tensor_model_parallel_size=args.tp)
    _pin_rotary_tables(unwrap_model(model))
    mamba = MambaInferenceStateConfig.from_model(model)
    if mamba is not None:
        mamba.conv_states_dtype = torch.float32
    kw = dict(
        block_size_tokens=256, buffer_size_gb=args.buffer_gb, num_cuda_graphs=16 if args.graphs else None,
        max_tokens=4096, max_sequence_length=args.max_len,
        kv_cache_management_mode=KVCacheManagementMode("persist"), static_kv_memory_pointers=False,
        use_cuda_graphs_for_non_decode_steps=bool(args.graphs), use_flashinfer_fused_rope=False,
        sampling_backend="flashinfer", use_synchronous_zmq_collectives=True,
        materialize_only_last_token_logits=True, enable_chunked_prefill=False,
        enable_prefix_caching=False, pg_collection=get_attr_wrapped_model(model, "pg_collection"),
        num_speculative_tokens=0, logprobs_mode="raw_logprobs", max_requests=64,
        async_sched_mode=AsyncScheduleMode("legacy"), mamba_inference_state_config=mamba,
        mamba_memory_ratio=None)
    import dataclasses

    fields = {f.name for f in dataclasses.fields(InferenceConfig)}
    kw = {k: v for k, v in kw.items() if k in fields}
    if "cuda_graph_max_tokens" in fields:
        kw["cuda_graph_max_tokens"] = 512
    ctx = DynamicInferenceContext(model.config, InferenceConfig(**kw))
    return DynamicInferenceEngine(TextGenerationController(GPTInferenceWrapper(model, ctx), tokenizer=tok), ctx)


def generate(eng, prompts, gen, seed):
    from megatron.core.inference.sampling_params import SamplingParams

    done = {}
    for i, p in enumerate(prompts):
        sp = SamplingParams(temperature=1.0, top_k=0, num_tokens_to_generate=gen, termination_id=-1,
                            return_log_probs=True, skip_prompt_log_probs=True, detokenize_generations=False)
        eng.add_request(1000 + i, p, sp)
    for _ in range(100000):
        for r in eng.step_modern()["finished_requests"]:
            done[int(r.request_id) - 1000] = (list(r.generated_tokens),
                                              torch.as_tensor(r.generated_log_probs, dtype=torch.float64))
        if len(done) == len(prompts):
            break
    return done


def _round_up(n, m):
    return (n + m - 1) // m * m


def _call(model, ids, pos, psp):
    """The learner's model call: through learner_graphs (graphs / device-dispatch guard) when active."""
    from nemo_rl.models.megatron import learner_graphs

    kw = {} if psp is None else {"packed_seq_params": psp}
    if learner_graphs.active() and psp is None:
        return learner_graphs.call(model, ids, pos, None)
    return model(input_ids=ids, position_ids=pos, attention_mask=None, **kw)


def _forward_logits(chunks, ids, pos, psp=None):
    """Model forward over every pipeline stage (forward-only schedule); last stage's logits."""
    from megatron.core import parallel_state as ps

    if len(chunks) == 1 and ps.get_pipeline_model_parallel_world_size() == 1:
        return _call(chunks[0], ids, pos, psp)
    from megatron.core.pipeline_parallel import get_forward_backward_func

    out = {}

    def step(data_iter, model):
        o = _call(model, ids, pos, psp)

        def fin(t):
            out["logits"] = t
            return t.float().sum() * 0, {}

        return o, fin

    fb = get_forward_backward_func()
    s = ids.shape[1] if psp is None else ids.shape[1]
    fb(forward_step_func=step, data_iterator=iter([None]), model=chunks, num_microbatches=1,
       seq_length=s, micro_batch_size=ids.shape[0], forward_only=True, collect_non_loss_data=False)
    return out.get("logits")


def learner_logprobs(args, chunks, rows, n_prompt, thd: bool):
    """Per row: log-probs of the generated tokens through the learner's log-prob op (float64)."""
    from megatron.core import parallel_state as ps

    from nemo_rl.distributed import model_utils

    tp = args.tp
    tpg = ps.get_tensor_model_parallel_group()
    mult = math.lcm(64, 32 * tp) if tp > 1 else 64
    with torch.no_grad():
        if not thd:
            s = _round_up(max(len(r) for r in rows), mult)
            ids = torch.zeros(len(rows), s, dtype=torch.long, device="cuda")
            for b, r in enumerate(rows):
                ids[b, : len(r)] = torch.tensor(r)
            pos = torch.arange(s, device="cuda").unsqueeze(0).expand(len(rows), -1).contiguous()
            logits = _forward_logits(chunks, ids, pos)
            if logits is None:
                return None
            vl = logits.shape[-1]
            r0 = ps.get_tensor_model_parallel_rank()
            lp_all = model_utils.from_parallel_logits_to_logprobs(
                logits.float(), ids, r0 * vl, (r0 + 1) * vl, tpg, inference_only=True)
            return [lp_all[b, n_prompt[b] - 1: len(r) - 1].double().cpu() for b, r in enumerate(rows)]
        from megatron.core.packed_seq_params import PackedSeqParams

        lens = [len(r) for r in rows]
        plens = [_round_up(n, mult) for n in lens]  # each sequence padded to the SP / norm multiple
        cu = torch.tensor([0] + list(torch.tensor(plens).cumsum(0)), dtype=torch.int32, device="cuda")
        flat = []
        for r, p in zip(rows, plens):
            flat += r + [0] * (p - len(r))
        ids = torch.tensor(flat, device="cuda")[None]
        pos = torch.cat([torch.arange(p) for p in plens]).cuda()[None]
        psp = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max(plens),
                              max_seqlen_kv=max(plens), cu_seqlens_q_padded=cu, cu_seqlens_kv_padded=cu)
        logits = _forward_logits(chunks, ids, pos, psp)
        if logits is None:
            return None
        vl = logits.shape[-1]
        r0 = ps.get_tensor_model_parallel_rank()
        lps = []
        for b, r in enumerate(rows):
            s0 = int(cu[b])
            seg = logits[:, s0: s0 + plens[b]].float()
            lp = model_utils.from_parallel_logits_to_logprobs(
                seg, ids[:, s0: s0 + plens[b]], r0 * vl, (r0 + 1) * vl, tpg, inference_only=True)
            lps.append(lp[0, n_prompt[b] - 1: len(r) - 1].double().cpu())
        return lps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1, help="learner PP")
    ap.add_argument("--engine_pp", type=int, default=1)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--engine_dispatcher", default="nvls")
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--buffer_gb", type=float, default=8)
    ap.add_argument("--graphs", type=int, default=1)
    ap.add_argument("--thd", type=int, default=0)  # BI Mamba training does not pack (hybrid: bshd only)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raise", dest="raise_", default="none")
    ap.add_argument("--latent_shard", type=int, default=0, help="TP-shard the latent projections (both sides)")
    ap.add_argument("--reps", type=int, default=1, help="learner calls per format (graphs: >= 3)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from nemo_rl.models.megatron import fp16_gemm, fp32_residual, precision_raise
    from nemo_rl.models.megatron.zero_train_gen_mismatch import enable_batch_invariant_kernels

    enable_batch_invariant_kernels({"megatron_cfg": {
        "batch_invariant_mode": True, "batch_invariant_backend": "te_native",
        "batch_invariant_collective": "ordered", "flash_attention_version": 4, "attention_backend": "flash"}})
    # As the policy worker: fused BI unpermute, learner MoE device dispatch (opt-in env).
    from nemo_rl.models.megatron import bi_unpermute, device_dispatch

    bi_unpermute.install()
    if device_dispatch.enabled():
        device_dispatch.install()
    fp32_residual.install()
    fp16_gemm.install()
    spec = args.raise_
    pcfg = {"preset": spec} if spec in precision_raise.PRESETS else {
        "raise_sites": [s for s in spec.split(",") if s and s != "none"]}
    precision_raise.install(pcfg, {})

    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    def init_ps(pp):
        ps.destroy_model_parallel()
        ps.initialize_model_parallel(args.tp, pp, expert_model_parallel_size=args.ep,
                                     expert_tensor_parallel_size=1)
        model_parallel_cuda_manual_seed(args.seed, te_rng_tracker=bool(args.graphs),
                                        inference_rng_tracker=bool(args.graphs))
        torch.manual_seed(args.seed)

    assert world == args.tp * max(args.pp, args.engine_pp) or world == args.tp, (world, args)
    init_ps(args.engine_pp)
    log(f"[parity] world {world} tp {args.tp} ep {args.ep} pp {args.pp} engine_pp {args.engine_pp} "
        f"tp_reduce={os.environ.get('NEMO_RL_TP_REDUCE', 'default')} graphs {args.graphs}")

    _, echunks = provider(args, True, bool(args.graphs))
    emodel = echunks[0]
    if args.latent_shard:
        from nemo_rl.models.megatron import latent_tp_shard

        log(f"[parity] engine: {latent_tp_shard.shard_model(echunks)} latent projections TP-sharded")
    fp16_gemm.hold_fp16_weights(emodel)
    from nemo_rl.models.megatron import mamba_bi_decode

    mamba_bi_decode.install()
    if args.engine_pp > 1:
        from nemo_rl.models.megatron import inference_pp_cuda_graph

        inference_pp_cuda_graph.install()
    eng = build_engine(args, emodel)
    rng = random.Random(args.seed)
    plens = [rng.choice([1, 7, 63, 64, 65, 200, 511, 1023, 1100, 1500]) for _ in range(args.rows)]
    prompts = [[rng.randrange(100, 30000) for _ in range(n)] for n in plens]
    t0 = time.perf_counter()
    done = generate(eng, prompts, args.gen, args.seed)
    log(f"engine: {len(done)} rows generated in {time.perf_counter() - t0:.1f} s (graphs {args.graphs})")
    # Every rank must hold the same tokens (sampling is replicated over TP / PP).
    tok = torch.tensor([t for i in range(args.rows) for t in done[i][0]], device="cuda")
    tok0 = tok.clone()
    dist.broadcast(tok0, 0)
    if world // (args.tp * args.engine_pp) > 1:
        # Engine data parallel (learner PP > engine PP): each replica samples on its own; the
        # learner checks replica 0's rows (its tokens and log-probs, on every rank).
        box = [done]
        dist.broadcast_object_list(box, src=0)
        done = {i: (t, lp.cpu()) for i, (t, lp) in box[0].items()}
    else:
        assert torch.equal(tok, tok0), "engine ranks sampled different tokens"
    del eng
    from megatron.core.inference.utils import InferenceMode

    InferenceMode.unset_active()  # the learner forward is a training-mode model call
    del echunks, emodel
    torch.cuda.empty_cache()

    if args.pp != args.engine_pp:
        init_ps(args.pp)
    _, lchunks = provider(args, False, False)
    if args.latent_shard:
        from nemo_rl.models.megatron import latent_tp_shard

        log(f"[parity] learner: {latent_tp_shard.shard_model(lchunks)} latent projections TP-sharded")
    rows = [prompts[i] + done[i][0] for i in range(args.rows)]
    res = {"args": vars(args), "prompt_lens": plens, "world": world,
           "tp_reduce": os.environ.get("NEMO_RL_TP_REDUCE", "default")}
    all_lp = {}
    last = ps.is_pipeline_last_stage()
    ok_all = True
    for fmt in (["bshd", "thd"] if args.thd else ["bshd"]):
        lps = learner_logprobs(args, lchunks, rows, plens, fmt == "thd")
        # --reps > 1: the same learner call again (learner graphs: eager first, capture, replay);
        # every repeat must give the first call's bits.
        rep_diff = 0
        for rep in range(args.reps - 1):
            # Odd repeats run the rows in another order (same shape, so a graph replay with new
            # inputs): a replay that kept the capture's inputs or activations would differ.
            perm = list(range(len(rows)))
            if rep % 2 == 0:
                perm = perm[1:] + perm[:1]
            again = learner_logprobs(args, lchunks, [rows[i] for i in perm], [plens[i] for i in perm],
                                     fmt == "thd")
            if last:
                rep_diff += sum(int((lps[i] != again[j]).sum()) for j, i in enumerate(perm))
        if args.reps > 1:
            from nemo_rl.models.megatron import learner_graphs

            t = torch.tensor([rep_diff], device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            rep_diff = int(t)

            res[fmt + "_reps"] = {"reps": args.reps, "differing": rep_diff, "graphs": learner_graphs.stats()}
            log(f"[parity] {fmt} repeats: {args.reps} calls, differing tokens vs first {rep_diff}, "
                f"learner_graphs {learner_graphs.stats()}")
            ok_all &= rep_diff == 0
        n_tok = mism = 0
        maxd = 0.0
        if last:
            for i in range(args.rows):
                e, l_ = done[i][1], lps[i]
                assert e.shape == l_.shape, (e.shape, l_.shape)
                n_tok += e.numel()
                mism += int((e != l_).sum())
                maxd = max(maxd, float((e - l_).abs().max()))
            all_lp[fmt] = lps
        stats = torch.tensor([n_tok, mism, maxd], dtype=torch.float64, device="cuda")
        gathered = [torch.zeros_like(stats) for _ in range(world)]
        dist.all_gather(gathered, stats)
        per_rank = [(int(g[0]), int(g[1]), float(g[2])) for g in gathered]
        res[fmt] = {"per_rank": per_rank, "tokens": max(p[0] for p in per_rank),
                    "mismatch": max(p[1] for p in per_rank), "max_abs_diff": max(p[2] for p in per_rank)}
        ok_all &= res[fmt]["mismatch"] == 0 and res[fmt]["tokens"] > 0
        log(f"[parity] {fmt}: tokens {res[fmt]['tokens']} mismatch {res[fmt]['mismatch']} "
            f"max|d| {res[fmt]['max_abs_diff']:.3e} per-rank {per_rank}")
    res["pass"] = bool(ok_all)
    writer = last and ps.get_tensor_model_parallel_rank() == 0 and ps.get_data_parallel_rank() == 0
    if writer:  # the last pipeline stage holds the learner log-probs
        all_lp["engine"] = [done[i][1] for i in range(args.rows)]
        all_lp["tokens"] = [done[i][0] for i in range(args.rows)]
        torch.save(all_lp, args.out + ".lp.pt")
        json.dump(res, open(args.out, "w"), indent=1)
    log(f"[parity] {'PASS' if ok_all else 'FAIL'} tp {args.tp} ep {args.ep} pp {args.pp}/{args.engine_pp}")
    sys.stdout.flush()
    sys.stderr.flush()
    dist.barrier()
    # Engine threads / NCCL communicators can hang a clean teardown: exit hard.
    os._exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
