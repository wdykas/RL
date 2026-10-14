#!/bin/bash
# Nemotron-3-Nano (hybrid Mamba-2 + attention + MoE: sigmoid router + expert bias, squared-ReLU
# experts, shared expert) truncated to L layers, fp16 GEMM forward + batch invariance, engine-sourced
# q, sticky routing: the train == engine exactness check on 2 generation + 2 training GPUs (EP=2).
#
#   bash research/thundersync_rl/scripts/bi_tp/nano_hybrid_fp16.sh        # L8 (MEMEM*EM), sticky
#   L=4 bash .../nano_hybrid_fp16.sh                                        # L4 (MEME, no attention)
#   STICKY=0 bash .../nano_hybrid_fp16.sh                                   # plain routing A/B
#   TF32=ieee bash .../nano_hybrid_fp16.sh                                  # IEEE fp32 Mamba SSD dots
#   SCHED=legacy bash .../nano_hybrid_fp16.sh                               # MCore legacy engine scheduling (default async)
#
# Pass (exit 0): as sticky_moe.sh (mismatch 0, exact_token_frac 1, engine-q exact, no scorer
# fallback; non-vacuity: weights change every step (expert-grad noise), steps >= 1 accept draft
# tokens, and with STICKY=1 sticky stats present, [route check] exact, the rule changes routes).
# Hybrid specifics: the engine's Mamba conv state is fp32 under the fp16 GEMM forward (default,
# resolve_mamba_conv_states_dtype); SSM states are fp32 (batch-invariant mode); the SSD Triton dots
# use TF32 in training and engine alike unless TF32=ieee sets TRITON_F32_DEFAULT on both.
# Truncated checkpoints: thundersync_work/models/nemotron3-nano-L$L (research/hybrid/make_trunc_nemotron.py).
W=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/wdykas/code/on-olicy-rl/thundersync_work
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
L=${L:-8}
export MODEL=${MODEL:-$W/models/nemotron3-nano-L$L}
if [ ! -f "$MODEL/model.safetensors" ]; then
  /opt/nemo_rl_venv/bin/python $W/research/hybrid/make_trunc_nemotron.py "$L" "$MODEL" || exit 2
fi
export EP=${EP:-2} TP=${TP:-1} PP=${PP:-1} MAXLEN=${MAXLEN:-1024} STEPS=${STEPS:-4}
export RUNS=${RUNS:-runs_nano_hybrid}
export SCHED=${SCHED:-async}
export TAG=${TAG:-nano-L$L-fp16-sticky${STICKY:-1}-ep$EP-$SCHED${TF32:+-$TF32}}
RECIPE="++policy.megatron_cfg.optimizer.use_precision_aware_optimizer=false
 ++policy.megatron_cfg.fp32_residual_connection=true ++policy.megatron_cfg.forward_gemm_precision=fp16
 policy.megatron_cfg.bias_activation_fusion=false"
if [ -n "${TF32:-}" ]; then
  export TRITON_F32_DEFAULT=$TF32
  RECIPE+=" ++policy.megatron_cfg.env_vars.TRITON_F32_DEFAULT=$TF32"
fi
export ARGS="$(echo $RECIPE) ${ARGS:-}"
exec bash "$HERE/sticky_moe.sh"
