# Ultra MPS transfer scripts

These scripts run the two Ultra **functional qualification** configurations:
native colocated Megatron (`alternating`) and shared-weight concurrent Megatron
under MPS (`concurrent`). They use SingleController, DAPOMath17K and the native
math verifier. They do not launch the public Ultra recipe's full NeMo-Gym reward
mixture, judge models or sandbox services.

This directory lives on a separate branch above the clean MPS implementation.
It contains the launch scripts, both configurations, runtime observers and the
MTP helper fix required by the pinned Megatron version. Checkpoints, container
images, Hugging Face caches and historical run outputs are not included.

## Preserved configuration

| Setting | Both arms |
|---|---|
| Allocation | 32 nodes × 4 GPUs = 128 GPUs |
| Training | TP8 / CP8 / EP64 / expert-TP1 / PP1 |
| Inference | TP8 / CP1 / EP64 / expert-TP1 / PP1 |
| Optimizer | GPU-only distributed precision-aware Adam; no optimizer parking |
| Work | 4 prompts × 16 responses; 64 sequences per update; 3 updates |
| Sequence limits | 2,048 input tokens; 49,152 maximum output/total tokens |
| Inference capacity | 8 requests per context; 128 aggregate request slots |
| Policy lag | At most one version |
| MPS arm | Shared weights; learner cap 58%; inference uncapped |

The allocation must provide **two groups of 16 four-GPU nodes**, with each group
inside one NVLink fabric domain. The native launcher advertises fabric metadata,
and the entrypoint validates the actual placement-group rank order before model
construction. This package deliberately rejects a different node/GPU geometry;
changing only the Slurm node count is insufficient to port it to a different
topology.

## Destination prerequisites

- Slurm with Pyxis/Enroot and the options required by the repository's `ray.sub`.
- GB200-class GPU capacity and the NVLink topology above. GPU and host memory fit
  still require completing qualification on the destination.
- A container with the NeMo RL driver and Megatron policy environments already
  installed. Supply their Python paths explicitly. The container's `ray` command
  must use the same Ray version as both Python environments.
- A shared filesystem mounted at the same absolute path on every node and in the
  containers. Put this checkout, model, dependency checkouts, cache and outputs
  below one shared root. The launcher rejects whitespace and shell metacharacters
  in these paths because the native launcher generates shell commands.
- The same local Hugging Face Ultra checkpoint for both arms, including tokenizer
  files and any required model code. `prepare_config.py` checks the architecture
  metadata (108 layers, 512 experts and MTP); it does not authenticate weight
  contents. Checkpoint weights are not distributed by this branch.
- DAPOMath17K available through Hugging Face or a populated shared cache. Native
  driver preflight exercises the actual dataset loader and math verifier.

The original runtime used PyTorch 2.13 with CUDA 13.0, Transformer Engine 2.18,
and Ray 2.56.1. An arbitrary older container is not a substitute: in particular,
the qualification expects precision-aware GPU optimizer state and the pinned
host-allocation options. No container installation or credential setup is
performed by these scripts.

Place these dependency revisions under the shared root:

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git /shared/ultra/Megatron-LM
git -C /shared/ultra/Megatron-LM checkout 3e95097b4a9ab6ea6b2a39eb2b2b24e4a6dfa54a
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git /shared/ultra/Megatron-Bridge
git -C /shared/ultra/Megatron-Bridge checkout 7e9dcd609edf2bb9478d3fa4d9b9f8ee3875c16e
```

Both must be usable with the installed container dependencies. The launcher
checks their revisions and rejects tracked source modifications. It bind-mounts
the two bundled MTP helper files over the pinned Megatron files **inside the
container only**. These are the exact upstream fix at
`0a513268eee94d8e903fecbc21fde862eecc31f3`; `mtp_fix/SOURCE.json` records original
and fixed hashes. Driver and worker checks verify the actual imported helpers.
The host checkouts are not patched.

## Configure and submit

Check out this branch under the shared root. Set destination-specific values in
your shell; `/shared/ultra` below is an example, not a required mount name:

```bash
export ULTRA_SHARED_ROOT=/shared/ultra
export ULTRA_MODEL_PATH=/shared/ultra/checkpoints/ultra
export ULTRA_MCORE_ROOT=/shared/ultra/Megatron-LM
export ULTRA_BRIDGE_ROOT=/shared/ultra/Megatron-Bridge
export ULTRA_OUTPUT_ROOT=/shared/ultra/results
export ULTRA_HF_HOME=/shared/ultra/huggingface
export ULTRA_CONTAINER=/shared/ultra/images/nemo-rl.sqsh
export ULTRA_DRIVER_PYTHON=/opt/nemo_rl_venv/bin/python
export ULTRA_POLICY_PYTHON=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
export ULTRA_ACCOUNT=YOUR_ACCOUNT
export ULTRA_PARTITION=YOUR_PARTITION
export ULTRA_CPUS_PER_NODE=144
export ULTRA_PLACEMENT_ARGS='--segment=16'
```

Use your site's equivalent placement option if it does not support `--segment`.
The runtime placement checks remain mandatory; there are no fixed host requests.
Set `ULTRA_CPUS_PER_NODE` to the destination node's CPU allocation. The optional
`ULTRA_TIME_LIMIT` defaults to `02:30:00`.

From the checkout, inspect both commands without submitting:

```bash
bash examples/ultra_mps/submit.sh alternating
bash examples/ultra_mps/submit.sh concurrent
```

Submit each arm as its own allocation:

```bash
ULTRA_SUBMIT=1 bash examples/ultra_mps/submit.sh alternating
ULTRA_SUBMIT=1 bash examples/ultra_mps/submit.sh concurrent
```

Additional arguments are forwarded to `sbatch`, for example `--qos=YOUR_QOS`.
Do not override the fixed GPU/node geometry. The training body has a 90-minute
timeout, separate from the Slurm allocation limit.

The driver and worker use the explicitly selected interpreters rather than
resolving new environments during a large allocation. Model loading follows the
native loader and its cache conventions. A cache from another checkpoint path
may not be reusable: prepare the native model cache on the destination or budget
for conversion/loading. No historical cache fingerprint is treated as evidence
that the destination cache is complete.

## Checks and outputs

Each allocation writes a fresh directory under `ULTRA_OUTPUT_ROOT`, containing
the resolved configuration, model-metadata hash, source revision, preflight
reports, topology, per-node memory samples, native training logs and all-rank
qualification records. The concurrent arm starts private MPS daemons only after
the native placement group selects the policy nodes.

Preflight checks actual driver math dependencies and actual policy schema,
inference capacity and MTP helper imports. Runtime checks retain shared-storage
identity, optimizer/publication ordering, GPU optimizer state and exact raw
token/log-prob counts. Successful qualification requires three completed native
updates with accepted work and a nonzero GRPO signal. Inspect the post-run
qualification report, Slurm exit status and native cleanup records together;
an observer pass alone does not prove allocation cleanup.

This is a portable adaptation of the queued configuration, not a destination
GPU-tested result. The qualification observers also add overhead. Do not use
these three-update jobs to report a steady-state speedup.
