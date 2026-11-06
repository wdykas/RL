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
"""GPU check: engine log-probs == learner log-probs, bitwise, under ``precision_raise`` sites.

One process, one GPU (EP 1). Builds the engine model the way the in-loop engine does
(``transformer_impl=inference_optimized``, fp16-held weights, BI Mamba decode, Megatron dynamic
inference with optional CUDA graphs) and the learner model the way the learner does (TE layers),
both from the same checkpoint, with BI kernels + fp32 residual + fp16 GEMM forward + the given
``precision_raise`` config installed in the process. The engine samples (temperature 1, seeded)
``--rows`` requests of mixed prompt lengths for ``--gen`` tokens (decode steps mix rows at
different positions; prompt prefill is graphed when graphs are on); the learner then runs one
padded batch forward (bshd) and, with ``--thd 1``, one packed THD forward over prompt + generated
tokens. Every generated token's log-prob must be bitwise equal to the engine's.

  python check_precision_raise_parity.py --model <nano-L8> --raise attn_core_fp32 --out r.json
  --raise none : the control (stock recipe) on the same harness.
Writes the learner log-probs to <out>.lp.pt so two configs can be diffed (non-vacuity: a raise
must change the numbers).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

os.environ.setdefault("CUBLASLT_WORKSPACE_SIZE", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":0:0")
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def provider(model, inference: bool, graphs: bool):
    from megatron.bridge import AutoBridge
    from megatron.core.transformer.enums import AttnBackend

    prov = AutoBridge.from_hf_pretrained(model, torch_dtype=torch.bfloat16,
                                         trust_remote_code=True).to_megatron_provider(load_weights=True)
    prov.tensor_model_parallel_size = prov.pipeline_model_parallel_size = 1
    prov.expert_model_parallel_size = prov.expert_tensor_parallel_size = 1
    prov.sequence_parallel = False
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
    if getattr(prov, "moe_router_num_groups", None) == 1:
        prov.moe_router_num_groups = prov.moe_router_group_topk = None
    if inference:
        prov.transformer_impl = "inference_optimized"
        prov.inference_moe_token_dispatcher_type = "nccl"
        if graphs:
            from megatron.core.transformer.enums import InferenceCudaGraphScope

            prov.cuda_graph_impl = "local"
            prov.use_te_rng_tracker = True
            prov.inference_cuda_graph_scope = InferenceCudaGraphScope["block"]
    prov.finalize()
    m = prov.provide_distributed_model(wrap_with_ddp=False)[0]
    m.eval()
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
                          make_vocab_size_divisible_by=128, tensor_model_parallel_size=1)
    _pin_rotary_tables(unwrap_model(model))
    mamba = MambaInferenceStateConfig.from_model(model)
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


def learner_logprobs(model, rows, n_prompt, thd: bool):
    """Per row: log-probs of the generated tokens (fp32 log_softmax of the fp32 logits, as float64)."""
    with torch.no_grad():
        if not thd:
            s = max(len(r) for r in rows)
            ids = torch.zeros(len(rows), s, dtype=torch.long, device="cuda")
            for b, r in enumerate(rows):
                ids[b, : len(r)] = torch.tensor(r)
            pos = torch.arange(s, device="cuda").unsqueeze(0).expand(len(rows), -1).contiguous()
            logits = model(input_ids=ids, position_ids=pos, attention_mask=None).float()
            lps = []
            for b, r in enumerate(rows):
                lp = torch.log_softmax(logits[b, : len(r) - 1], -1)
                tgt = torch.tensor(r[1:], device="cuda")
                lps.append(lp.gather(-1, tgt[:, None])[:, 0][n_prompt[b] - 1:].double().cpu())
            return lps
        from megatron.core.packed_seq_params import PackedSeqParams

        lens = [len(r) for r in rows]
        cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
        ids = torch.tensor([t for r in rows for t in r], device="cuda")[None]
        pos = torch.cat([torch.arange(n) for n in lens]).cuda()[None]
        psp = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max(lens),
                              max_seqlen_kv=max(lens))
        logits = model(input_ids=ids, position_ids=pos, attention_mask=None, packed_seq_params=psp).float()[0]
        lps = []
        for b, r in enumerate(rows):
            s0 = int(cu[b])
            lp = torch.log_softmax(logits[s0: s0 + len(r) - 1], -1)
            tgt = torch.tensor(r[1:], device="cuda")
            lps.append(lp.gather(-1, tgt[:, None])[:, 0][n_prompt[b] - 1:].double().cpu())
        return lps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--raise", dest="raise_", default="attn_core_fp32",
                    help="comma list of precision_raise sites, a preset name (32k/128k/...), or none")
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--buffer_gb", type=float, default=8)
    ap.add_argument("--graphs", type=int, default=1)
    ap.add_argument("--thd", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from nemo_rl.models.megatron import fp16_gemm, fp32_residual, precision_raise
    from nemo_rl.models.megatron.zero_train_gen_mismatch import enable_batch_invariant_kernels

    enable_batch_invariant_kernels({"megatron_cfg": {
        "batch_invariant_mode": True, "batch_invariant_backend": "te_native",
        "batch_invariant_collective": "ordered", "flash_attention_version": 4, "attention_backend": "flash"}})
    fp32_residual.install()
    fp16_gemm.install()
    spec, c16 = (args.raise_[:-4], True) if args.raise_.endswith("+c16") else (args.raise_, False)
    if spec in precision_raise.PRESETS:
        pcfg = {"preset": spec}
    else:
        pcfg = {"raise_sites": [s for s in spec.split(",") if s and s != "none"]}
    pcfg["combine_payload_fp16"] = c16  # '+c16' suffix: combine_payload_fp16
    precision_raise.install(pcfg, {})
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29700 + os.getpid() % 500))
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=0, world_size=1)
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    ps.initialize_model_parallel(1, 1, expert_model_parallel_size=1)
    model_parallel_cuda_manual_seed(args.seed, te_rng_tracker=bool(args.graphs),
                                    inference_rng_tracker=bool(args.graphs))
    torch.manual_seed(args.seed)

    _, emodel = provider(args.model, True, bool(args.graphs))
    fp16_gemm.hold_fp16_weights(emodel)
    from nemo_rl.models.megatron import mamba_bi_decode

    mamba_bi_decode.install()
    eng = build_engine(args, emodel)
    rng = random.Random(args.seed)
    plens = [rng.choice([1, 7, 63, 64, 65, 200, 511, 1023, 1100, 1500]) for _ in range(args.rows)]
    prompts = [[rng.randrange(100, 30000) for _ in range(n)] for n in plens]
    t0 = time.perf_counter()
    done = generate(eng, prompts, args.gen, args.seed)
    log(f"engine: {len(done)} rows generated in {time.perf_counter() - t0:.1f} s (graphs {args.graphs})")
    del eng
    torch.cuda.empty_cache()
    from megatron.core.inference.utils import InferenceMode

    InferenceMode.unset_active()  # the learner forward is a training-mode model call

    _, lmodel = provider(args.model, False, False)
    rows = [prompts[i] + done[i][0] for i in range(args.rows)]
    res = {"raise": sorted(precision_raise.sites()) + (["combine_payload_fp16"] if c16 else []), "args": vars(args), "prompt_lens": plens}
    all_lp = {}
    for fmt in (["bshd", "thd"] if args.thd else ["bshd"]):
        lps = learner_logprobs(lmodel, rows, plens, fmt == "thd")
        n_tok = mism = 0
        maxd = 0.0
        for i in range(args.rows):
            e, l_ = done[i][1], lps[i]
            assert e.shape == l_.shape, (e.shape, l_.shape)
            n_tok += e.numel()
            ne = (e != l_)
            mism += int(ne.sum())
            maxd = max(maxd, float((e - l_).abs().max()))
        res[fmt] = {"tokens": n_tok, "mismatch": mism, "max_abs_diff": maxd}
        all_lp[fmt] = lps
        log(f"[parity] {fmt}: raise={res['raise']} tokens {n_tok} mismatch {mism} max|d| {maxd:.3e}")
    if "attn_core_qp" in precision_raise.sites():
        # Non-vacuity: the learner with the stock fp16 attention (q/k/v cast restored) must differ.
        from nemo_rl.models.megatron import hp_attention_mcore as hpm

        hp_cast = fp16_gemm._cast_qkv
        fp16_gemm._cast_qkv = hpm._STATE["orig"]["cast"]
        try:
            lps0 = learner_logprobs(lmodel, rows, plens, False)
        finally:
            fp16_gemm._cast_qkv = hp_cast
        d0 = max(float((a - b).abs().max()) for a, b in zip(lps0, all_lp["bshd"]))
        n0 = sum(int((a != b).sum()) for a, b in zip(lps0, all_lp["bshd"]))
        res["nonvacuity_fp16_attention"] = {"differing_tokens": n0, "max_abs_diff": d0}
        log(f"[parity] non-vacuity: stock fp16 attention differs on {n0} tokens, max|d| {d0:.3e}")
    all_lp["engine"] = [done[i][1] for i in range(args.rows)]
    all_lp["tokens"] = [done[i][0] for i in range(args.rows)]
    torch.save(all_lp, args.out + ".lp.pt")
    ok = all(res[f]["mismatch"] == 0 for f in res if f in ("bshd", "thd"))
    res["pass"] = ok
    json.dump(res, open(args.out, "w"), indent=1)
    log(f"[parity] {'PASS' if ok else 'FAIL'} raise={res['raise']}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
