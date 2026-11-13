Ultra 128-GPU native Math qualification helpers

Scope: three functional native Math GRPO updates, not the full external reward
mixture, numerical equivalence, convergence, or steady throughput. All guards
are from the matched GPU-only qualification. This package contains no recorded
cluster results and does not require a historical job or completed gate file.

Add this qualification directory to PYTHONPATH in the driver AND actual policy
actor runtime environments. Before importing the native entry, register
math_worker.MathAuditWorker with the same ACTOR_ENVIRONMENT_REGISTRY value as
nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker.
Set policy.worker_extension_cls_fqn to math_worker.MathAuditWorker in both arms.
Use the native worker interpreter containing Torch, Ray, Transformer Engine,
Megatron-Core and Megatron-Bridge; postrun verification also needs TensorBoard.

Actual worker environment:
  NRL_ULTRA_AUDIT_DIR          RUN/observer
  ULTRA_DIRECT64_RUN_DIR       RUN (the existing variable name covers world128)
  NRL_ULTRA_MTP_FIX_SOURCE     the portable ../mtp_fix/SOURCE.json
  NRL_ULTRA_MCORE_SOURCE       the supplied, actually mounted Megatron-Core root
Create a fresh RUN for each invocation. Worker records contain actual runtime
paths and identities; the source package contains no installation-specific path.

Before model startup, run mtp_fix_guard.py --source-manifest SOURCE.json
--mcore-source MCORE_ROOT --output RUN/actual-mtp-import-guard.json in a bounded,
CUDA_VISIBLE_DEVICES='' subprocess using the policy interpreter. Preserve the
actual driver/worker PYTHONPATH and helper binds. Each actual worker independently
repeats the helper path/SHA and controller-alias checks before construction.
The SOURCE.json describes exact native-relative helper paths and fixed SHA256s;
this import check is not a replacement for native algorithm tests.

Keep all actual observer wrappers: gathered DDP storage/publication readiness,
reader drain, optimizer, private router-bias copy, shared aliases and epochs.
GPU optimizer inventory requires actual native DistributedOptimizer children,
TE FusedAdam, precision-aware BF16 CUDA shards, dense group16/expert group2,
CPU offload disabled, and CUDA FP32 moments/INT16 remainders after each update.
Raw generated-token/logprob lengths are checked before original native packing;
missing entries fail rather than being padded or trimmed.

After native entry completion, run:
  python -m ultra_qualification.validate --records RUN/observer --mode ARM \
    --updates 3 --world-size 128 --output RUN/observer-validation.json
  python qualification/verify_run.py --run RUN --arm ARM
ARM is alternating or concurrent. The latter command defaults SOURCE.json to
../mtp_fix/SOURCE.json relative to this directory; --source-manifest overrides it.

Required launcher startup evidence:
  RUN/actual-mtp-import-guard.json: ACTUAL_MTP_MODULE_AND_CONTROLLER_IDENTITY_PASS
  RUN/topology.json: status PRE_MODEL_NATIVE_TOPOLOGY_PASS,
                    physical.reserved_gpus=128 and two rank_islands
Required postrun data:
  RUN/training-logs/**/events.out.tfevents.*
  RUN/observer/*.json and RUN/observer-validation.json
  RUN/gpu-optimizer-inventory/*.json: 128 learner construction inventories
  RUN/mtp-import-guards/*.json: 128 alternating / 256 concurrent processes
  RUN/raw-generation-payloads/*.jsonl: at least 192 unique raw requests
Verification requires three64-sequence updates, finite/nonzero native GRPO,
lag0/1, zero dropped/replaced/evicted work, 384 optimizer-returned state records,
and all-rank observer checks. It writes RUN/functional-qualification.json.
Allocation completion/cleanup, actual MPS registrations/caps, and physical memory
peaks remain external evidence. A local verifier PASS is not a performance claim.

CPU tests: put qualification on PYTHONPATH and run pytest on qualification/tests.
They use synthetic records/classes and native-parser AST CPU tests; they do not
initialize a GPU, native distributed workers, or a model. Historical cache/gate
artifacts are not dependencies. The base checkout's production raw-payload guard
is also exercised; synthetic shape-only evidence cannot qualify a run.
