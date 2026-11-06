"""Engine decode bench (helper module for tree_engine_long.py; also runnable standalone).

Options beyond a plain decode bench: --tp (tensor parallel, sequence parallel at TP > 1, EP = world / TP with ETP 1),
--cf_m / --cf_extra (cross-fade theta-only routing on the engine), --nl 1 (no-lanes engine side: wide keyed sampler +
q hidden log).

The engine is built as the in-loop RL runs build it (buffer 48 GB, max_requests 128, max_model_len 10240, max_tokens
16384, 16 graphs to 512 tokens, hybrid sizing, block-scope local graphs, non-decode steps graphed, async scheduling,
BI te_native + FA4, fp16-GEMM recipe, fp32 residual, fp32 router, NVLS AGV/RSV, hold_fp16_weights + mamba_bi_decode).
One process per GPU, EP = world size. Each point keeps exactly N rows decoding (max_requests is set, never-ending
requests) at context C, and checks it.

Per point N:C (rows per rank : prompt length):
  wall      graphs on: median wall ms/step over --steps steps (step_modern loop)
  host      per-step host time of the engine/controller phases (perf_counter wrappers; host waits on CUDA events =
            "wait"): how much of the step the host is busy vs blocked on the GPU
  --nsys 1  wraps the measured steps in cudaProfilerStart/Stop + NVTX (run under nsys --capture-range=cudaProfilerApi)
  --kinds 1 graphs off: torch-profiler device ms/step per kernel kind + top kernels
  --ts_hooks q,route  install the ThunderSync engine hooks as the worker does (q / hidden-row log, sticky route
            record + log) for the in-loop host-cost comparison
  --greedy 1 --dump P   top-1 sampling, per-rank generated tokens + log-probs of every finished request -> P.rank<R>.pt

  torchrun --nproc-per-node 4 decode_bench.py --model <hf id|dir> --points 1:2048,8:2048,64:2048,256:2048 --out x.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import statistics
import time

os.environ.setdefault("CUBLASLT_WORKSPACE_SIZE", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":0:0")
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL = int(os.environ.get("LOCAL_RANK", "0"))


def log(*a):
    if RANK == 0:
        print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def build(args):
    from nemo_rl.models.megatron import fp16_gemm, fp32_residual
    from nemo_rl.models.megatron.zero_train_gen_mismatch import enable_batch_invariant_kernels

    STOCK = bool(args.stock)
    if not STOCK:
      enable_batch_invariant_kernels({"megatron_cfg": {"batch_invariant_mode": True, "batch_invariant_backend": "te_native",
                                                     "batch_invariant_collective": "ordered",
                                                     "flash_attention_version": 4, "attention_backend": "flash"}})
    if args.epinv:
        # EP-invariant MoE combine (exact limb sums), as run_ab_arm.sh turns it on for lanes (engine EP != learner EP).
        from nemo_rl.models.megatron import ep_invariant_combine

        ep_invariant_combine.set_enabled(True)
        ep_invariant_combine.install_engine_buffers()
    if not STOCK:
        fp32_residual.install()
        fp16_gemm.install()
    if args.min_bucket:
        # decode_lat probe: smallest BI decode bucket below 64 tokens (MCore TOKEN_ROUNDER); speed + dump A/B only
        import megatron.core.inference.batch_dimensions_utils as _bdu
        _bdu.TOKEN_ROUNDER = int(args.min_bucket)
    torch.cuda.set_device(LOCAL)
    dist.init_process_group("nccl")
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    TP = args.tp
    EP = args.ep or WORLD
    ps.initialize_model_parallel(TP, 1, expert_model_parallel_size=EP, expert_tensor_parallel_size=1)
    model_parallel_cuda_manual_seed(0, te_rng_tracker=bool(args.graphs), inference_rng_tracker=bool(args.graphs))
    if args.sticky:
        from nemo_rl.models.megatron import sticky_routing

        sticky_routing.set_delta(0.03)

    from megatron.bridge import AutoBridge
    from megatron.core.transformer.enums import AttnBackend

    # --mcore_ckpt <torch_dist dir> | random: the HF dir supplies config + tokenizer only (no safetensors, e.g. Super-120B);
    # weights come from the Megatron checkpoint (dist_checkpointing, resharded to this EP) or stay at random init.
    mck = getattr(args, "mcore_ckpt", "")
    prov = AutoBridge.from_hf_pretrained(args.model, torch_dtype=torch.bfloat16,
                                         trust_remote_code=True).to_megatron_provider(load_weights=not mck)
    if mck and hasattr(prov, "mtp_num_layers"):
        prov.mtp_num_layers = None  # engine decode: no MTP head
    prov.tensor_model_parallel_size = TP
    prov.pipeline_model_parallel_size = 1
    prov.expert_model_parallel_size = EP
    prov.expert_tensor_parallel_size = 1
    prov.sequence_parallel = TP > 1
    prov.params_dtype = torch.bfloat16
    prov.bf16, prov.fp16 = True, False
    prov.gradient_accumulation_fusion = False
    prov.fp32_residual_connection = not STOCK
    prov.moe_router_dtype = "fp32"
    prov.moe_permute_fusion = False
    prov.attention_backend = AttnBackend.flash
    if not STOCK:
        prov.flash_attention_version = 4  # BI overlay FA4 (stock: the provider's default flash path)
    prov.batch_invariant_mode = not STOCK
    if not STOCK:
        prov.batch_invariant_backend = "te_native"
        prov.batch_invariant_collective = "ordered"
    prov.transformer_impl = "inference_optimized"
    prov.inference_moe_token_dispatcher_type = args.dispatcher
    prov.moe_shared_expert_overlap = False
    if args.sticky:
        prov.moe_enable_routing_replay = True
    if args.cf_m > 0:
        from nemo_rl.models.megatron import crossfade_routing

        crossfade_routing.configure({"megatron_cfg": {"crossfade_routing_m": args.cf_m,
                                                      "crossfade_routing_extra": args.cf_extra,
                                                      "crossfade_routing_sparse_engine": bool(args.cf_sparse)}}, prov)
        log("cross-fade:", crossfade_routing.get_state(), "topk", prov.moe_router_topk)
    if args.layers > 0:
        prov.num_layers = args.layers
        prov.hybrid_override_pattern = prov.hybrid_override_pattern[: args.layers]
    if getattr(prov, "moe_router_num_groups", None) == 1:
        prov.moe_router_num_groups = prov.moe_router_group_topk = None
    if args.graphs:
        from megatron.core.transformer.enums import InferenceCudaGraphScope

        prov.cuda_graph_impl = "local"
        prov.use_te_rng_tracker = True
        prov.inference_cuda_graph_scope = InferenceCudaGraphScope[args.scope]
    prov.finalize()
    model = prov.provide_distributed_model(wrap_with_ddp=False)[0]
    model.eval()
    if mck and mck != "random":
        load_mcore_ckpt(model, mck)
    elif mck == "random":
        deterministic_init(model)
    n = 0
    if not STOCK:
        n = fp16_gemm.hold_fp16_weights(model)
        from nemo_rl.models.megatron import mamba_bi_decode

        mamba_bi_decode.install()  # on the D2 tree this also installs the decode step
    from nemo_rl.models.generation.megatron.megatron_worker import install_engine_latency_levers

    log("STOCK (BI off)" if STOCK else "BI on", "engine latency levers:", {} if STOCK else install_engine_latency_levers(model),
        "sampler S", os.environ.get("THUNDERSYNC_WIDE_SAMPLER_S", "0"), "sched", args.sched)
    log("model built", f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params/rank, EP={WORLD}",
        f"fp16-held {n}", f"TP={TP} EP={EP}", f"layers {prov.num_layers} pattern {prov.hybrid_override_pattern}",
        f"mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

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
                          make_vocab_size_divisible_by=128, tensor_model_parallel_size=TP)
    _pin_rotary_tables(unwrap_model(model))
    mamba = MambaInferenceStateConfig.from_model(
        model, ssm_states_dtype=torch.float32 if getattr(args, "ssm_fp32", 0) else None)
    mamba.conv_states_dtype = torch.float32  # resolve_mamba_conv_states_dtype under the fp16 GEMM forward
    if args.buffer_gb < 0:  # -R: every free GiB on the device except R (graphs / activations / transient)
        torch.cuda.empty_cache()  # checkpoint-load staging blocks
        free_gib, reserve = torch.cuda.mem_get_info()[0] / 2**30, -args.buffer_gb
        t_ = torch.tensor([free_gib - reserve], device="cuda")
        dist.all_reduce(t_, op=dist.ReduceOp.MIN)
        args.buffer_gb = float(t_.item())
        log(f"buffer_gb: device free {free_gib:.1f} GiB (rank 0) - reserve {reserve:.1f} -> {args.buffer_gb:.1f} GiB (min over ranks)")
    kw = dict(
        block_size_tokens=256, buffer_size_gb=args.buffer_gb, num_cuda_graphs=args.num_graphs if args.graphs else None,
        max_tokens=args.max_tokens, max_sequence_length=args.max_len,
        kv_cache_management_mode=KVCacheManagementMode("persist"), static_kv_memory_pointers=False,
        use_cuda_graphs_for_non_decode_steps=bool(args.graphs), use_flashinfer_fused_rope=False,
        sampling_backend="flashinfer", use_synchronous_zmq_collectives=True,
        materialize_only_last_token_logits=not (args.prompt_lp or getattr(args, "verify_prefill", 0)), enable_chunked_prefill=bool(getattr(args, "chunked", 0)),
        enable_prefix_caching=False, pg_collection=get_attr_wrapped_model(model, "pg_collection"),
        num_speculative_tokens=0, logprobs_mode="raw_logprobs", max_requests=args.max_requests,
        async_sched_mode=AsyncScheduleMode(args.sched), mamba_inference_state_config=mamba,
        mamba_memory_ratio=None)
    import dataclasses

    fields = {f.name for f in dataclasses.fields(InferenceConfig)}
    for k, v in (("cuda_graph_max_tokens", 512), ("cuda_graph_sizing_distribution", "hybrid"),
                 ("cuda_graph_all_prefills", bool(int(os.environ.get("DL_ALL_PREFILLS", "1"))))):
        if k in fields:
            kw[k] = v
    if getattr(args, "verify_prefill", 0):
        # Verify-in-prefill probe (colo MODEL.md): every position's log p (prompt log-probs on every request, set in
        # points()) plus a head-input capture of every position copied to host, as a forward pre-hook on the output
        # layer (installed before graph capture, so graphs replay it): rows -> static device buffer -> pinned host.
        from megatron.core.utils import unwrap_model as _uw

        _ol = getattr(_uw(model), "output_layer", None)
        assert _ol is not None, "verify_prefill: model has no output_layer"
        _cap = {}

        def _pre(mod, inp):
            x = inp[0]
            rows = x.reshape(-1, x.shape[-1])
            buf = _cap.get("host")
            if buf is None or buf.shape[0] < rows.shape[0] or buf.dtype != rows.dtype:
                n = max(int(args.max_tokens) + 1024, rows.shape[0])
                _cap["host"] = buf = torch.empty((n, rows.shape[1]), dtype=rows.dtype, pin_memory=True)
            buf[: rows.shape[0]].copy_(rows, non_blocking=True)
            _cap["rows"] = _cap.get("rows", 0) + rows.shape[0]

        _ol.register_forward_pre_hook(_pre)
        globals()["VERIFY_CAP"] = _cap
        log("verify_prefill: all-position prompt log-probs + head-input capture to pinned host")
    ctx = DynamicInferenceContext(model.config, InferenceConfig(**kw))
    if args.nl and not args.ts_hooks:
        args.ts_hooks = "q"  # nolanes engine: q hidden log (nl_engine_setup -> install_draft_hidden_log)
    hooks = TsHooks(model, ctx, args.ts_hooks) if args.ts_hooks else None
    # With hooks: the ThunderSync worker's engine class (installs the hidden copy and the route record before
    # graph capture, drops MCore's route bookkeeping, allows async scheduling with the route record).
    eng_cls = hooks.engine_class(DynamicInferenceEngine) if hooks is not None else DynamicInferenceEngine
    eng = eng_cls(TextGenerationController(GPTInferenceWrapper(model, ctx), tokenizer=tok), ctx)
    if args.wide or args.nl:
        # The 53-bit keyed engine sampler (thundersync.engine_sampler: wide, the default), installed by
        # the worker's own method before the q log (as SpeculativeDriver does).
        import types

        from thundersync_rl import worker as W

        wcls = W.ThunderSyncMegatronPolicyWorker
        wcls = getattr(wcls, "__ray_actor_class__", None) or getattr(getattr(wcls, "__ray_metadata__", None),
                                                                       "modified_class", None) or wcls
        st = (hooks if hooks is not None else TsHooks(model, ctx, "")).obj
        st.dynamic_inference_engine = eng
        st.rank = RANK
        global WIDE_STATS
        WIDE_STATS = st.install_wide_sampler(salt=12345, key64=True, fresh_salt=None)
        log("wide sampler:", WIDE_STATS)
    if hooks is not None:
        hooks.after_engine(eng)
    if args.tail_hold:
        from nemo_rl.models.generation.megatron import tail_hold
        tail_hold.install(eng)
    log("engine built", f"max_requests {ctx.max_requests}", f"mem {torch.cuda.memory_allocated() / 2**30:.2f} GiB",
        f"graphs {len(getattr(ctx, 'cuda_graph_batch_dimensions_list', []) or [])}")
    log("MEM " + json.dumps(mem_info(ctx)))
    dec = sorted({int(getattr(d, "token_count", 0)) for d in (getattr(ctx, "cuda_graph_batch_dimensions_list", []) or [])
                  if int(getattr(d, "prefill_req_count", 1) or 0) == 0})
    log("decode graph token counts", dec)
    return eng, model, prov, hooks


def deterministic_init(model):
    """--mcore_ckpt random: weights as a function of (parameter name, rank) only, so two runs of the same layout
    hold the same bits (Megatron's random init is not reproducible across processes here). Matrices N(0, 0.02);
    Mamba A_log / dt_bias / D fixed ramps; other vectors (norm weights, biases) keep their constant init."""
    import math
    import zlib

    n = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            # seed by (name, shard): TP-sharded params by TP rank, expert params by EP rank, replicated params the
            # same on every rank (a per-RANK seed made TP-replicated weights differ across TP ranks, so sequence-
            # parallel token placement changed a row's numerics: a bench artifact, not engine batch dependence)
            from megatron.core import parallel_state as _ps2
            shard = 0
            if getattr(p, "tensor_model_parallel", False):
                shard += 1 + _ps2.get_tensor_model_parallel_rank()
            if not getattr(p, "allreduce", True):
                shard += 1000 * (1 + _ps2.get_expert_model_parallel_rank())
            g = torch.Generator(device=p.device).manual_seed((zlib.crc32(name.encode()) * 1000003 + shard) % (2**62))
            if p.dim() >= 2:
                p.copy_(torch.empty(p.shape, device=p.device, dtype=torch.float32).normal_(0.0, 0.02, generator=g))
            elif name.endswith("A_log"):
                p.copy_(torch.log(torch.linspace(1.0, 16.0, p.numel(), device=p.device)))
            elif name.endswith("dt_bias"):
                dt = torch.linspace(1e-3, 1e-1, p.numel(), device=p.device)
                p.copy_(dt + torch.log(-torch.expm1(-dt)))
            elif name.endswith(".D") or name == "D":
                p.fill_(1.0)
            else:
                continue
            n += 1
    log(f"deterministic init: {n} parameters")


def load_mcore_ckpt(model, path):
    """Load a Megatron torch_dist checkpoint into the built (EP-sharded) model; _extra_state skipped; missing keys logged."""
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.validation import StrictHandling

    from megatron.core.utils import unwrap_model

    model = unwrap_model(model)
    t = time.perf_counter()
    from megatron.core.dist_checkpointing.mapping import ShardedObject

    # weights only: TE / inference-layer _extra_state (ShardedObjects, per-expert shards) carry no tensors we need
    sd = {k: v for k, v in model.sharded_state_dict().items()
          if "_extra_state" not in k and not isinstance(v, ShardedObject)}
    loaded, missing, unexpected = dist_checkpointing.load(sd, path, strict=StrictHandling.RETURN_ALL)
    res = model.load_state_dict(loaded, strict=False)
    miss = [k for k in res.missing_keys if "_extra_state" not in k]
    log(f"mcore ckpt {path}: {len(loaded)} tensors in {time.perf_counter() - t:.0f} s; ckpt-missing {len(missing)}, "
        f"model-missing {len(miss)} {miss[:6]}, unexpected (ckpt keys not in model) {len(unexpected)}")
    # dist_checkpointing's mismatch sets only hold _extra_state / MTP keys we skip on purpose; every model weight must load
    other = [k for k in set(missing) | set(unexpected) if "_extra_state" not in k and not k.startswith("mtp")]
    assert not miss and not other, (other, miss)
    del loaded, sd
    torch.cuda.empty_cache()  # release the load's staging blocks before the engine sizes its buffers / graphs


def mem_info(ctx):
    """Engine memory: KV block pool, Mamba state tensors (conv / SSM + any BI replay buffers on the context), totals."""
    mamba = {}
    for k, v in vars(ctx).items():
        if isinstance(v, torch.Tensor) and "mamba" in k.lower():
            mamba[k] = v.numel() * v.element_size()
    free, total = torch.cuda.mem_get_info()
    al = ctx.kv_block_allocator
    return {"max_requests": int(ctx.max_requests), "block_size_tokens": int(ctx.block_size_tokens),
            "block_size_bytes": int(ctx.block_size_bytes), "kv_pool_blocks": int(al.pool_size),
            "kv_paused_limit": int(getattr(al, "paused_limit", 0)),
            "kv_pool_gib": al.pool_size * ctx.block_size_bytes / 2**30,
            "kv_bytes_per_token": ctx.block_size_bytes / ctx.block_size_tokens,
            "mamba_tensors_gib": {k: b / 2**30 for k, b in mamba.items()},
            "mamba_bytes_per_request": sum(mamba.values()) / max(1, int(ctx.max_requests)),
            "allocated_gib": torch.cuda.memory_allocated() / 2**30, "reserved_gib": torch.cuda.memory_reserved() / 2**30,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "device_free_gib": free / 2**30, "device_total_gib": total / 2**30}


class TsHooks:
    """ThunderSync engine hooks, installed by the worker's own functions on a stand-in object.

    q: ``_install_hidden_copy`` (LM-head input copy, captured into the graphs) + ``install_draft_hidden_log`` (the
    sampler-side q-row log), every request tagged as a draft. route: sticky route record (before graph capture) +
    ``install_draft_route_log``. The functions are the ThunderSync worker's, unbound, so the bench measures the code
    the loop runs.
    """

    def __init__(self, model, ctx, which):
        import types

        from thundersync_rl import worker as W

        cls = W.ThunderSyncMegatronPolicyWorker
        cls = getattr(cls, "__ray_actor_class__", None) or getattr(getattr(cls, "__ray_metadata__", None),
                                                                     "modified_class", None) or cls
        self.cls = cls
        self.which = set(which.split(","))
        class _Stand(types.SimpleNamespace):  # worker methods bound on demand (the worker grows new helpers)
            def __getattr__(self, name):
                f = getattr(cls, name, None)
                if f is None or not callable(f):
                    raise AttributeError(name)
                return types.MethodType(f, self)

        self.obj = _Stand()
        self.obj._gen_model = lambda: model
        self.obj.inference_context = ctx
        self.ctx = ctx
        for name in ("_install_hidden_copy", "install_draft_hidden_log", "_install_route_record",
                     "install_draft_route_log", "_wrap_route_log", "_engine_class", "_install_anchor_apply",
                     "_wrap_anchor_apply"):
            setattr(self.obj, name, types.MethodType(getattr(cls, name), self.obj))

    def engine_class(self, orig):
        return self.obj._engine_class(orig)

    def after_engine(self, eng):
        self.obj.dynamic_inference_engine = eng
        if "q" in self.which:
            log("q hidden log installed:", self.obj.install_draft_hidden_log(seal_blocks=bool(int(os.environ.get("EB_SEAL", "0")))))
        if "route" in self.which:
            log("route log installed:", self.obj.install_draft_route_log())

    def tag(self, sp, key):
        from thundersync_rl.engine_q import HIDDEN_KEY_ATTR

        setattr(sp, HIDDEN_KEY_ATTR, key)


_KINDS = (  # first match wins
    ("expert_act", r"_relu2_probs|_swiglu_probs"),
    ("mamba", r"chunk_scan|chunk_state|state_passing|bmm_chunk|chunk_cumsum|causal_conv|masked_update|gate_target|"
              r"selective_state|ssd|mamba|_bi_decode|replay"),
    ("attention", r"flash|fmha|attn|attention|rotary|rope|splitkv|anchor"),
    ("moe_dispatch", r"multimem|all_gather|allgather|reduce_scatter|ordered_reduce|nvls|symm|nccl|limbs"),
    ("moe", r"grouped_gemm|permute|unpermute|topk|router|moe|expert|_relu2|fp64"),
    ("dense_gemm", r"gemm|cutlass|cublas|nvjet|xmma|sm90_|sm100_|hilo|matmul"),
    ("sampling", r"sampl|softmax|log_softmax|argmax|flashinfer|curand|philox|gumbel|exponential|cumsum|sort"),
    ("cast_fp16", r"float16_copy|BFloat16_copy|bfloat16_copy|half"),
    ("copy", r"direct_copy|copy_kernel|Memcpy|memcpy|CatArray|index|gather|scatter"),
    ("elementwise", r"elementwise|fill|reduce_kernel|norm"),
)


def kind_of(name):
    for k, rx in _KINDS:
        if re.search(rx, name, re.IGNORECASE):
            return k
    return "other"


# ---- host phase timers (perf_counter; no device sync) ----
_T = collections.defaultdict(float)
_C = collections.defaultdict(int)


def _wrap(obj, name, label, is_async=False):
    f = getattr(obj, name, None)
    if f is None:
        return
    if is_async:
        async def w(*a, **k):
            t = time.perf_counter()
            try:
                return await f(*a, **k)
            finally:
                _T[label] += time.perf_counter() - t
                _C[label] += 1
    else:
        def w(*a, **k):
            t = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                _T[label] += time.perf_counter() - t
                _C[label] += 1
    setattr(obj, name, w)


def install_host_timers(eng):
    c = eng.controller
    ctx = eng.context
    for n in ("_run_async_sched_prepare", "_run_async_sched_sample", "_run_async_sched_log_probs",
              "_copy_async_sched_log_probs_to_cpu", "_run_async_sched_publish_bookkeeping",
              "_run_async_sched_forward", "_run_async_sched_resolve", "_materialize_async_sched_log_probs",
              "_dynamic_step_context_init", "_dynamic_step_forward_logits", "_dynamic_step_sample_logits",
              "_compact_async_sched_logits", "_build_async_sched_step_result"):
        _wrap(c, n, "ctl." + n.lstrip("_"))
    _wrap(c, "_synchronize_async_sched_event", "WAIT.sync_event")
    for n in ("commit_sampled_tokens", "prepare_requests", "resolve_requests", "transfer_bookkeeping_to_gpu",
              "copy_async_sched_sample_to_forward"):
        _wrap(ctx, n, "ctx." + n)
    _wrap(eng, "post_process_requests", "eng.post_process_requests")
    _wrap(eng, "async_forward", "eng.async_forward", is_async=True)
    _wrap(eng, "async_bookkeep", "eng.async_bookkeep", is_async=True)


def _all_ranks(flag: bool) -> bool:
    t = torch.tensor([1 if flag else 0], dtype=torch.int32, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())


def points(eng, args, prov, hooks):
    from megatron.core.inference.sampling_params import SamplingParams

    out = {"ep": args.ep or WORLD, "tp": args.tp, "model": args.model, "layers": prov.num_layers, "pattern": prov.hybrid_override_pattern,
           "max_requests": int(eng.context.max_requests), "graphs": args.graphs, "scope": args.scope,
           "ts_hooks": args.ts_hooks, "args": vars(args), "points": {}}
    done = {}

    def step():
        for r in eng.step_modern()["finished_requests"]:
            plp = getattr(r, "prompt_log_probs", None)
            done[int(r.request_id)] = (torch.tensor(list(r.generated_tokens), dtype=torch.int64),
                                       torch.as_tensor(r.generated_log_probs, dtype=torch.float32),
                                       torch.as_tensor(plp if plp is not None else [], dtype=torch.float32))

    if args.timers:
        install_host_timers(eng)
    for k, pt in enumerate(args.points.split(",")):
        # N:C = N rows at context C; N:C:L = the same with ONE of the N rows at context L (longest-live-row probe)
        pt_, _, peer = pt.partition("@")
        f_ = [int(x) for x in pt_.split(":")]
        Np, C = f_[0], f_[1]
        from megatron.core import parallel_state as _ps
        dpr = _ps.get_data_parallel_rank()
        if peer and dpr != 0:
            Np = int(peer)  # rows on every DP rank except DP rank 0 (the measured "tail" rank)
        lens = ([f_[2]] + [C] * (Np - 1)) if len(f_) > 2 else [C] * Np
        rng = random.Random(500 + k)  # same prompts on every rank: lockstep steps
        ntok = args.ntok if args.ntok > 0 else args.warm + args.steps + 64 + (args.prof_steps if args.nsys else 0)
        ctx_ = eng.context
        need_blocks = sum(-(-(c + ntok) // ctx_.block_size_tokens) for c in lens)
        have_blocks = ctx_.kv_block_allocator.pool_size - 1
        if Np > ctx_.max_requests or need_blocks > have_blocks or max(lens) + ntok > ctx_.max_sequence_length:
            log(f"SKIP point {pt}: needs {Np} requests / {need_blocks} KV blocks / len {C + ntok}; engine has "
                f"{ctx_.max_requests} / {have_blocks} / {ctx_.max_sequence_length}")
            out["points"][pt] = {"skipped": True, "need_blocks": need_blocks, "have_blocks": have_blocks}
            continue
        torch.cuda.reset_peak_memory_stats()
        for i in range(Np):
            tail = bool(args.tail_rows) and dpr == 0 and i < args.tail_rows  # DP rank 0's first rows: tails
            sp = SamplingParams(temperature=1.0, top_k=1 if args.greedy else 0,
                                num_tokens_to_generate=(args.tail_ntok or ntok) if tail else ntok,
                                termination_id=-1, return_log_probs=True, skip_prompt_log_probs=not ((args.prompt_lp and i % 2 == 0) or getattr(args, "verify_prefill", 0)),
                                detokenize_generations=False)
            if hooks is not None:
                hooks.tag(sp, f"bench-{k}-{i}")
            if tail:
                from nemo_rl.models.generation.megatron.tail_hold import TAIL_ATTR
                sp.add_attributes({TAIL_ATTR: 1})
            eng.add_request(800_000_000 + 10000 * k + i, [rng.randrange(100, 30000) for _ in range(lens[i])], sp)
        t0 = time.perf_counter()
        for _ in range(100000):
            step()
            d = eng.decode_only
            live = int(eng.context.total_request_count - (0 if args.tail_hold else eng.context.paused_request_count))
            ok = bool((d.launched if d.launched is not None else d.consumed) and live == Np and not eng.waiting_request_ids)
            if _all_ranks(ok):  # collective exit: DP ranks with different row counts keep stepping in lockstep (EP)
                break
        prefill_s = time.perf_counter() - t0
        for _ in range(args.warm):
            step()
        torch.cuda.synchronize()
        dist.barrier()
        live = int(eng.context.total_request_count - eng.context.paused_request_count)
        res = {"live_rows": live, "prefill_s": prefill_s}
        if live != Np:
            log(f"WARNING point {pt}: {live} live rows, wanted {Np} (max_requests {eng.context.max_requests})")
        if args.kinds:
            from torch.profiler import ProfilerActivity, profile

            acts = [ProfilerActivity.CUDA] + ([ProfilerActivity.CPU] if args.kinds > 1 else [])
            with profile(activities=acts, with_stack=args.kinds > 1) as prof:
                for _ in range(10):
                    step()
                torch.cuda.synchronize()
            if args.kinds > 1 and RANK == 0:
                # Which Python call sites launch the small kernels: CPU ops by device time, grouped by stack.
                tab = prof.key_averages(group_by_stack_n=6).table(sort_by="self_device_time_total", row_limit=60,
                                                                   max_name_column_width=60)
                open(f"{args.out}.N{Np}.stacks.txt", "w").write(tab)
            kinds, rows = {}, []
            for e in prof.key_averages():
                if e.self_device_time_total > 0 and e.device_type == torch.autograd.DeviceType.CUDA:
                    ms = e.self_device_time_total / 10 / 1e3
                    kinds[kind_of(e.key)] = kinds.get(kind_of(e.key), 0.0) + ms
                    rows.append((ms, e.count / 10, e.key[:110]))
            top = sorted(rows, reverse=True)[:40]
            res["kinds_ms"], res["top"] = kinds, top
            log(f"KINDS N={Np} C={C} device ms/step total {sum(kinds.values()):.3f} " +
                " ".join(f"{a}={b:.3f}" for a, b in sorted(kinds.items())))
            for ms, cnt, name in top[:25]:
                log(f"   {ms:8.3f} ms  x{cnt:6.1f}/step  {name}")
        if args.callsites:
            # Graphs off: every torch call of one step, by (function, nearest megatron/nemo_rl/thundersync frame).
            import traceback

            from torch.overrides import TorchFunctionMode

            counts = collections.Counter()

            class _Mode(TorchFunctionMode):
                def __torch_function__(self, func, types, a=(), kw=None):
                    fr = [f for f in traceback.extract_stack(limit=14)[:-1]
                          if ("megatron" in f.filename or "nemo_rl" in f.filename or "thundersync" in f.filename)
                          and "decode_bench" not in f.filename]
                    site = " < ".join(f"{f.filename.split('/')[-1]}:{f.lineno}:{f.name}" for f in fr[-3:][::-1])
                    counts[(getattr(func, "__name__", str(func)), site)] += 1
                    return func(*a, **(kw or {}))

            with _Mode():
                step()
            torch.cuda.synchronize()
            if RANK == 0:
                with open(f"{args.out}.N{Np}.callsites.txt", "w") as fh:
                    for (fn, site), c in counts.most_common():
                        fh.write(f"{c:5d}  {fn:28s} {site}\n")
                log(f"CALLSITES N={Np}: {sum(counts.values())} torch calls in one step -> {args.out}.N{Np}.callsites.txt")
        if args.gprof:
            # Graphs on: GPU busy vs idle over 10 steps (CUPTI kernel + memcpy intervals, union over streams).
            from torch.profiler import ProfilerActivity, profile

            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                t0 = time.perf_counter()
                for _ in range(10):
                    step()
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0
            tr = f"{args.out}.N{Np}.trace.json"
            prof.export_chrome_trace(tr)
            ev = json.load(open(tr))["traceEvents"]
            iv = sorted((e["ts"], e["ts"] + e.get("dur", 0)) for e in ev
                        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("dur") is not None)
            busy, cs, ce, gaps = 0.0, None, None, []
            for a, b in iv:
                if ce is None or a > ce:
                    if ce is not None:
                        busy += ce - cs
                        gaps.append(a - ce)
                    cs, ce = a, b
                else:
                    ce = max(ce, b)
            if ce is not None:
                busy += ce - cs
            span = (iv[-1][1] - iv[0][0]) if iv else 0.0
            big = sorted(gaps, reverse=True)[:10]
            res["gprof"] = {"wall_ms": wall * 1e3 / 10, "gpu_span_ms": span / 1e3 / 10, "gpu_busy_ms": busy / 1e3 / 10,
                            "gpu_idle_ms": (span - busy) / 1e3 / 10, "n_gpu_ops": len(iv) / 10,
                            "gaps_gt_20us_ms": sum(g for g in gaps if g > 20) / 1e3 / 10,
                            "n_gaps_gt_20us": sum(1 for g in gaps if g > 20) / 10, "largest_gaps_us": big}
            log(f"GPROF N={Np} " + json.dumps(res["gprof"]))
            os.remove(tr) if args.keep_trace == 0 else None
        if args.nsys:
            from megatron.core.utils import configure_nvtx_profiling

            configure_nvtx_profiling(True)
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            for i in range(args.prof_steps):
                torch.cuda.nvtx.range_push(f"bench_step N={Np} i={i}")
                step()
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            configure_nvtx_profiling(False)
        _T.clear()
        _C.clear()
        walls = []
        last = time.perf_counter()
        for _ in range(args.steps):
            step()
            now = time.perf_counter()
            walls.append(now - last)
            last = now
        torch.cuda.synchronize()
        med = 1e3 * statistics.median(walls)
        res.update(ms=med, ms_p10=1e3 * sorted(walls)[len(walls) // 10],
                   ms_p90=1e3 * sorted(walls)[(9 * len(walls)) // 10])
        if args.timers:
            res["host_ms_per_step"] = {k: 1e3 * v / args.steps for k, v in sorted(_T.items())}
            res["host_calls_per_step"] = {k: v / args.steps for k, v in sorted(_C.items())}
            log(f"HOST N={Np} " + " ".join(f"{k}={v:.3f}" for k, v in res["host_ms_per_step"].items()))
        res["mem"] = mem_info(eng.context)
        log(f"POINT N={Np}/rank C={C}{(' long1=' + str(lens[0])) if len(f_) > 2 else ''} wall/step median {med:.3f} ms (p10 {res['ms_p10']:.3f} p90 {res['ms_p90']:.3f})"
            f" live {live} peak mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB prefill {prefill_s:.3f} s prefill_tok_per_s_fleet {Np * C * (WORLD // args.tp) / max(prefill_s, 1e-9):.0f}")
        out["points"][pt] = res
        while not _all_ranks(not eng.has_unfinished_requests()):
            if eng.has_unfinished_requests():
                step()
            else:
                eng.controller.dummy_forward()  # EP peer still decoding: take part in its collectives
        dist.barrier()
    if args.wide or args.nl:
        # The keyed sampler must have run during decode (the install-time line always reads calls 0).
        ws = dict(getattr(eng.controller._sampling, "_wide_stats", None) or globals().get("WIDE_STATS") or {})
        log("wide sampler at end:", ws)
        assert int(ws.get("calls", 0)) > 0, f"--wide 1 but the keyed sampler never ran: {ws}"
    if args.dump:
        torch.save(done, f"{args.dump}.rank{RANK}.pt")
        log(f"dumped {len(done)} finished requests to {args.dump}.rank*.pt")
    if RANK == 0:
        json.dump(out, open(args.out, "w"), indent=1)


def main():
    import faulthandler
    import sys as _sys
    faulthandler.dump_traceback_later(int(os.environ.get("DL_HANG_DUMP_S", "420")), repeat=True, file=_sys.stderr)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    ap.add_argument("--points", default="1:2048,8:2048,64:2048,256:2048")
    ap.add_argument("--ntok", type=int, default=0, help="tokens per request (0: just enough for the measurement)")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warm", type=int, default=12)
    ap.add_argument("--prof_steps", type=int, default=12)
    ap.add_argument("--buffer_gb", type=float, default=48)
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--max_len", type=int, default=10240)
    ap.add_argument("--max_requests", type=int, default=256)
    ap.add_argument("--sched", default="async")
    ap.add_argument("--scope", default="block")
    ap.add_argument("--dispatcher", default="nvls", choices=["nvls", "nccl"])
    ap.add_argument("--graphs", type=int, default=1)
    ap.add_argument("--kinds", type=int, default=0)
    ap.add_argument("--nsys", type=int, default=0)
    ap.add_argument("--gprof", type=int, default=0, help="graphs-on torch-profiler GPU busy/idle per step")
    ap.add_argument("--keep_trace", type=int, default=0)
    ap.add_argument("--callsites", type=int, default=0, help="graphs off: torch calls of one step by call site")
    ap.add_argument("--timers", type=int, default=1)
    ap.add_argument("--ts_hooks", default="", help="comma list of q,route")
    ap.add_argument("--sticky", type=int, default=0)
    ap.add_argument("--greedy", type=int, default=0)
    ap.add_argument("--dump", default="")
    ap.add_argument("--num_graphs", type=int, default=16)
    ap.add_argument("--epinv", type=int, default=0, help="EP-invariant MoE combine (lanes recipe at engine EP != learner EP)")
    ap.add_argument("--wide", type=int, default=0, help="install the 53-bit keyed engine sampler (the default sampler)")
    ap.add_argument("--chunked", type=int, default=0, help="chunked prefill (long prompts > max_tokens; decode steps unchanged)")
    ap.add_argument("--mcore_ckpt", default="", help="Megatron torch_dist checkpoint dir (HF dir = config only) or 'random'")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=0, help="0: world size")
    ap.add_argument("--cf_m", type=float, default=0.0, help="cross-fade routing m (0: off; nl_base 0.003)")
    ap.add_argument("--cf_extra", type=int, default=4)
    ap.add_argument("--nl", type=int, default=0, help="nolanes engine side: wide keyed sampler + q hidden log")
    ap.add_argument("--prompt_lp", type=int, default=0, help="even requests return prompt log-probs (prefill check)")
    ap.add_argument("--cf_sparse", type=int, default=1, help="engine expert GEMMs skip zero-weight cross-fade slots (nl_super)")
    ap.add_argument("--tail_hold", type=int, default=0, help="install nemo_rl tail_hold on the engine")
    ap.add_argument("--tail_rows", type=int, default=0, help="DP rank 0's first rows are tails (TAIL_ATTR)")
    ap.add_argument("--tail_ntok", type=int, default=0, help="tokens per tail row (0: as the others)")
    ap.add_argument("--stock", type=int, default=0, help="BI off: no BI kernels, no fp16-GEMM / fp32-residual recipe, no BI Mamba decode")
    ap.add_argument("--min_bucket", type=int, default=0, help="override MCore TOKEN_ROUNDER (smallest BI graph bucket)")
    ap.add_argument("--verify_prefill", type=int, default=0, help="every request returns all-position prompt log-probs + head-input rows copied to pinned host")
    ap.add_argument("--ssm_fp32", type=int, default=0, help="fp32 Mamba SSM states (the stock recipe's mamba_inference_ssm_states_dtype)")
    ap.add_argument("--layers", type=int, default=0, help="truncate the model to its first N layers (0: all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    eng, _model, prov, hooks = build(args)
    points(eng, args, prov, hooks)
    dist.barrier()
    torch.cuda.synchronize()
    log("bench done")
    # The engine holds background resources (symmetric-memory handles, NCCL comms in use by captured graphs) whose
    # teardown can hang at interpreter exit; every output is written by now.
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
