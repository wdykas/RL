# Arm presets (sourced by jobs/*.sh): the exact knob sets of the verified runs, per arm and model.
#   preset <arm> <model> [K]    sets ARM_LAUNCHER (colo_arm.sh | stock_arm.sh) and exports the arm's env
# arm:   wvs  = PRODUCTION: ours, exact colocated partial rollout, fixed G, spare prompts, verify at the wake (colo_wv.yaml:
#               XPR-S + wake_verify; engine row slots, continuations first), lag K
#        xprs = ours, XPR-S with the verify at the stand-down (colo_xprs.yaml, SPARE=S: P + S admitted, P trained), lag K
#        xpr  = ours, exact colocated partial rollout without spares (colo_xpr.yaml), lag K
#        vg   = ours, ALTERNATIVE: exact partial rollout with variable group size (colo_vg.yaml, M 1.4, wake_verify),
#               lag K; rows in flight capped at SPR's
#        xs   = ours, exact colocated sync (colo.yaml), lag 0              (reference)
#        xssd = ours, XS + exact keyed tree speculative decode (colo.yaml + tail tree 8:4:256, MTP draft head
#               COLO_XSSD_HEAD), lag 0; engine rows MAX_REQ on every arm of its set
#        spr  = stock colocated partial rollout (stock tree, in_order lag K)  (baseline a)
#        sprbi = stock colocated partial rollout on OUR tree, engine and BI numerics (colo_sprbi.yaml)  (baseline b:
#               ours / sprbi = method overhead; give it the same MAX_REQ / ENGINE_BUF as the arm it is compared to)
#        cs   = stock colocated sync (K = 0)                                (kernel normalization only)
# model: l8 (Super-L8, 1 node, TP2 EP4) | sl22 (Super-L22, 4 nodes, TP2 EP8) | nano (full Nano, 4 nodes, TP1 EP8)
#        | ul27 (Ultra-L27, 16 nodes, TP8 EP32)
# Same in every arm of a comparison: model, layout, prompts x G, MAXLEN, steps, seed, routers frozen,
# policy.offload_optimizer_for_refit=false (both arms), MTD off.
# Anything already exported (e.g. STEPS=6) wins over the preset.
preset() {
  local arm=$1 m=$2 k=${3:-1}
  unset CFG_REL ARM_LAUNCHER
  local _TP _EP _GPUS _P=64 _ML=8192 _PC=2048 _ST=12 _PAR=1 _MODEL _MBS _LMBS=2 _LDIV=128 _SBUF=24 _SREQ=512 _XE=""
  case $m in
    l8)   _MODEL=$MODEL_SUPER_L8; _TP=2; _EP=4; _GPUS=4; _P=16; _ML=512; _PC=1024; _ST=4; _MBS=""; _SBUF=16; _SREQ=64 ;;   # MBS: config default, as the gates
    sl22) _MODEL=$MODEL_SUPER_L22; _TP=2; _EP=8; _GPUS=16; _MBS=2 ;;
    nano) _MODEL=$MODEL_NANO; _TP=1; _EP=8; _GPUS=16; _PAR=0; _MBS=1; _SREQ=256 ;;   # our learner needs MBS 1 at Nano
    ul27) _MODEL=$MODEL_ULTRA_L27; _TP=8; _EP=32; _GPUS=64; _ST=8; _MBS=1; _LMBS=1; _LDIV=256
          _XE='++policy.megatron_cfg.env_vars.NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS="18432"' ;;
    *) echo "[preset] model must be l8 | sl22 | nano | ul27" >&2; return 2 ;;
  esac
  export MODEL=${MODEL:-$_MODEL} TP=${TP:-$_TP} EP=${EP:-$_EP} GPUS=${GPUS:-$_GPUS} PROMPTS=${PROMPTS:-$_P} G=${G:-8}
  export MAXLEN=${MAXLEN:-$_ML} PROMPT_CAP=${PROMPT_CAP:-$_PC} STEPS=${STEPS:-$_ST} PARITY=${PARITY:-$_PAR} SEED=${SEED:-42}
  export RUN_TIMEOUT=${RUN_TIMEOUT:-5400}
  # Engine settings on Megatron-LM main (our tree): prefix caching on (weight-scoped), coordinator policy load_balanced
  # (the engine row slots need it), prefills eager as stock. Chunked prefill: on for XS and its SPR-BI; held off for the
  # speculative arms (a draft cut mid chunked prefill at the stand-down: fix pending), and so for their SPR-BI too.
  local _MG=policy.generation.mcore_generation_config
  # Stock's engine set (prefix caching with its coordinator default, eager prefills, stock graph pools); the speculative
  # arms add the load_balanced coordinator their engine row slots need.
  local _ENG_SYNC="$_MG.enable_prefix_caching=true $_MG.cuda_graph_all_prefills=false $_MG.num_cuda_graphs=-1"
  local _ENG="$_ENG_SYNC $_MG.prefix_caching_coordinator_policy=load_balanced"
  local _CP_ON="$_MG.enable_chunked_prefill=true" _CP_OFF="$_MG.enable_chunked_prefill=false"
  case $arm in
    wvs|xpr|xprs|xs|xssd|vg|sprbi|isspr)
      ARM_LAUNCHER=colo_arm.sh
      case $arm in
        xpr)  export CFG_REL=research/thundersync_rl/configs/colo/colo_xpr.yaml K=$k; _XE="$_XE $_ENG $_CP_OFF" ;;
        # wvs: engine rows per rank by rule in colo_arm.sh: ceil(4/3 (P G + (P+S) N) / engine DP), rounded up to 16
        # (Super-L22 256, measured; Nano 128, rule-derived).
        wvs)  export CFG_REL=$COLO_PROD_CFG K=$k; _XE="$_XE $_ENG $_CP_OFF"
              export SPARE=${SPARE:-$( [ $m = l8 ] && echo 2 || echo 8 )}
              _XE="$_XE policy.generation.mcore_generation_config.num_cuda_graphs=-1" ;;
        xprs) export CFG_REL=research/thundersync_rl/configs/colo/colo_xprs.yaml K=$k; _XE="$_XE $_ENG $_CP_OFF"
              export SPARE=${SPARE:-$( [ $m = l8 ] && echo 2 || echo 8 )}   # Super-L22: S 8 of 72 (TAIL_FLOOR §3)
              _XE="$_XE policy.generation.mcore_generation_config.num_cuda_graphs=-1" ;;   # stock's graph set
        # xs: engine rows = SPR's rows in flight (2 P G / engine DP), as the fair XS / SPR-BI pair (SPRBI_MATCH=xs).
        xs)   export CFG_REL=research/thundersync_rl/configs/colo/colo.yaml; _XE="$_XE $_ENG_SYNC $_CP_ON"
              [ $m = l8 ] || export MAX_REQ=${MAX_REQ:-$(( 2 * _P * 8 * _TP / _GPUS ))} ;;
        # xssd: XS + the exact tree speculative decode. Tree node rows count against max_requests (tree_spec.plan:
        # room = max_requests - live rows), so MAX_REQ = max((K+1) P G, 4 P G) / engine DP; set the same MAX_REQ on every
        # arm of its set. The MTP head's K / V slots = MAX_REQ.
        xssd) export CFG_REL=research/thundersync_rl/configs/colo/colo.yaml; _XE="$_XE $_ENG_SYNC $_CP_ON"
              [ $m = l8 ] || export MAX_REQ=${MAX_REQ:-$(( 4 * _P * 8 * _TP / _GPUS ))}
              _XE="$_XE ++thundersync.tree_xssd=${XSSD_TREE:-8:4:256} ++thundersync.tree_xssd_mtp=${XSSD_HEAD:-${COLO_XSSD_HEAD:?set COLO_XSSD_HEAD to the MTP draft head file (head_v2)}}:${MAX_REQ:-64}" ;;
        # vg: admits ceil(1.4 P) prompts (Super-L22: 90 -> ~512 trained rows); engine requests per rank capped at
        # SPR's rows in flight ((K+1) P G / engine DP: 128 at Super-L22), the measured VG-cap arm.
        vg)   export CFG_REL=research/thundersync_rl/configs/colo/colo_vg.yaml K=$k; _XE="$_XE $_ENG $_CP_OFF"
              export MAX_REQ=${MAX_REQ:-$(( (k + 1) * _P * 8 * _TP / _GPUS ))}
              export PROMPTS=$( [ $m = l8 ] && echo ${VG_PROMPTS:-24} || echo ${VG_PROMPTS:-$(( (_P * 14 + 9) / 10 ))} ) ;;
        # sprbi: same engine capacity as vg by default (the method-overhead pair c = vg, b = sprbi); set MAX_REQ /
        # ENGINE_BUF to the compared arm's values for another arm.
        # isspr: IS-SPR (colo_isspr.yaml: stock pause / resume + keyed rescore at the wake, group IS weights, cross-fade). Same
        # engine as XS and its SPR-BI: prefix caching + chunked prefill on, SPR's rows in flight, stock coordinator (no slots).
        isspr) export CFG_REL=research/thundersync_rl/configs/colo/colo_isspr.yaml K=$k
               export MAX_REQ=${MAX_REQ:-$(( (k + 1) * _P * 8 * _TP / _GPUS ))}
               _XE="$_XE $_ENG_SYNC $_CP_ON" ;;
        # SPRBI_MATCH=wvs (default): the production arm's engine requests and graph set; xprs: XPR-S's ((K+1) (P+S) 13 /
        # engine DP); vg: SPR's rows in flight; xs: XS's (one batch) with chunked prefill.
        sprbi) export CFG_REL=research/thundersync_rl/configs/colo/colo_sprbi.yaml K=$k
               # same engine settings as the arm it is compared to (SPRBI_MATCH): chunked prefill only with xs
               case ${SPRBI_MATCH:-wvs} in xs|isspr) _XE="$_XE $_ENG_SYNC $_CP_ON" ;; *) _XE="$_XE $_ENG $_CP_OFF" ;; esac
               if [ "${SPRBI_MATCH:-wvs}" = wvs ]; then
                 local _S=$( [ $m = l8 ] && echo 2 || echo 8 ) _DP=$(( _GPUS / _TP ))
                 export MAX_REQ=${MAX_REQ:-$(( ( (4 * (_P * 8 + (_P + _S) * 14) + 3 * _DP - 1) / (3 * _DP) + 15) / 16 * 16 ))}
                 _XE="$_XE policy.generation.mcore_generation_config.num_cuda_graphs=-1"
               elif [ "${SPRBI_MATCH}" = xprs ]; then
                 local _S=$( [ $m = l8 ] && echo 2 || echo 8 ) _DP=$(( _GPUS / _TP ))
                 export MAX_REQ=${MAX_REQ:-$(( ( ((k + 1) * (_P + _S) * 13 + _DP - 1) / _DP + 15) / 16 * 16 ))}
                 _XE="$_XE policy.generation.mcore_generation_config.num_cuda_graphs=-1"
               else
                 export MAX_REQ=${MAX_REQ:-$(( (k + 1) * _P * 8 * _TP / _GPUS ))}
               fi ;;
      esac
      # F3: the Nano cross-fade band cfS30 (m .03 x [.3, 1, .5] by MoE depth, extra 4) in every arm on our tree. SPR-BI's
      # stock loop reads it from policy.megatron_cfg, ThunderSync arms from the thundersync block. Other models keep the
      # configs' band (m .01, extra 8). No spaces inside the profile: XA is split on whitespace.
      if [ $m = nano ]; then
        local _BP=$( [ $arm = sprbi ] && echo policy.megatron_cfg || echo thundersync )
        _XE="$_XE ++$_BP.crossfade_routing_m=0.03 ++$_BP.crossfade_routing_m_profile=[[0.0,0.2608,0.3],[0.2608,0.7826,1.0],[0.7826,1.0,0.5]] ++$_BP.crossfade_routing_extra=4"
      fi
      [ -n "$_MBS" ] && export MBS=${MBS:-$_MBS}
      # MAX_REQ / ENGINE_BUF: by rule in colo_arm.sh (lib/engine_buf.py), except the 1-node values the gates used.
      [ $m = l8 ] && { export ENGINE_BUF=${ENGINE_BUF:-16}; [ $arm = xpr ] && export MAX_REQ=${MAX_REQ:-$(( (_P * 9 * (k + 1) + 1) / 2 / 16 * 16 + 16 ))}; }
      export XA="policy.offload_optimizer_for_refit=false $_XE ${XA_EXTRA:-}" ;;
    spr|cs)
      ARM_LAUNCHER=stock_arm.sh
      [ $arm = cs ] && k=0
      # Stock as is, except the knobs a fair comparison must hold equal to b / c: learner micro-batch (our learner's), engine
      # requests = its rows in flight ((K+1) P G / engine DP, which the in-flight prompt cap enforces anyway), the engine
      # buffer by the same rule as ours (stock_arm.sh). Eager prefills are stock's default (our arms set it explicitly).
      export K=$k VARIANT=ltuned LMBS=${LMBS:-${_MBS:-$_LMBS}} LDIV=${LDIV:-$_LDIV} KV=$COLO_KV
      [ $m = l8 ] && export ENGINE_BUF=${ENGINE_BUF:-$_SBUF} MAX_REQ=${MAX_REQ:-$_SREQ}
      export MAX_REQ=${MAX_REQ:-$(( (k + 1) * _P * 8 * _TP / _GPUS ))}
      # Stock's best config also carries the generic learner lever NEMO_RL_LEARNER_DEFER_METRICS (stock code, no BI;
      # same-job A/B: 57.06 vs 57.61 s/step, training phase equal).
      export ARGS="policy.offload_optimizer_for_refit=false async_rl.max_inflight_prompts=$(( _P * (k + 1) )) ++policy.megatron_cfg.env_vars={NEMO_RL_LEARNER_DEFER_METRICS:'1'} ${XA_EXTRA:-}" ;;
    *) echo "[preset] arm must be wvs | xprs | xpr | vg | xs | xssd | isspr | spr | sprbi | cs" >&2; return 2 ;;
  esac
  export ARM_LAUNCHER
}
