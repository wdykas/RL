# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Check actual driver/policy dependencies in a CUDA-hidden child before models."""

import argparse
import inspect
import json
import os
from pathlib import Path
import sys

from prepare_config import sha256, verify_materialized


def policy_checks(config: dict) -> dict:
    """Validate installed native schema and eager capacity without constructing models."""
    # These native dependencies belong to the actual policy interpreter only.
    from megatron.core.inference.config import InferenceConfig, KVCacheManagementMode
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
    from nemo_rl.algorithms.single_controller_utils.config import (
        MasterConfig,
        validate_single_controller_config,
    )
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    master = MasterConfig(**config)
    validate_single_controller_config(master)
    generation = config["policy"]["generation"]["mcore_generation_config"]
    source = Path(inspect.getfile(DynamicInferenceContext)).resolve()
    expected = (
        Path(config["policy"]["megatron_cfg"]["env_vars"]["NRL_ULTRA_MCORE_SOURCE"])
        / "megatron/core/inference/contexts/dynamic_context.py"
    )
    if not source.samefile(expected):
        raise ValueError("Imported MCore context does not match supplied source root")
    padded = DynamicInferenceContext.round_up_tokens(
        generation["max_tokens"], tp_size=generation["tensor_model_parallel_size"]
    )
    if padded != generation["max_tokens"]:
        raise ValueError(
            "Native eager prefill padding exceeds configured input capacity"
        )
    inference = InferenceConfig(
        max_tokens=generation["max_tokens"],
        max_requests=generation["max_requests"],
        max_sequence_length=generation["max_model_len"],
        buffer_size_gb=generation["buffer_size_gb"],
        block_size_tokens=generation["block_size_tokens"],
        num_speculative_tokens=generation["num_speculative_tokens"],
        logprobs_mode=generation["logprobs_mode"],
        kv_cache_management_mode=KVCacheManagementMode(
            generation["kv_cache_management_mode"]
        ),
        enable_chunked_prefill=generation["enable_chunked_prefill"],
        enable_prefix_caching=generation["enable_prefix_caching"],
        sampling_backend="flashinfer",
    )
    if inference.sampling_backend != "flashinfer":
        raise ValueError("Native FlashInfer sampling dependency is missing")
    return {
        "native_schema": type(master).__name__,
        "worker_module": inspect.getfile(MegatronPolicyWorkerImpl),
        "mcore_context": str(source),
        "mcore_context_sha256": sha256(source),
        "max_tokens": inference.max_tokens,
        "padded_capacity": padded,
        "scope": "Native schema and inference constructor subset; no model, process-group or memory-fit proof",
    }


def driver_checks() -> dict:
    """Exercise the actual native Math verifier and dataset loader without actors."""
    # Driver-only dependencies must be checked in the environment used by Math actors.
    from nemo_rl.data.datasets.response_datasets.dapo_math import DAPOMath17KDataset
    from nemo_rl.environments.dapo_math_verifier import compute_score
    from nemo_rl.environments.math_environment import HFVerifyWorker, MathEnvironment

    positive = compute_score(r"Answer: \boxed{2}", "2")
    negative = compute_score(r"Answer: \boxed{3}", "2")
    if positive["score"] != 1 or negative["score"] != 0:
        raise ValueError("Native DAPO verifier positive/negative checks failed")
    verifier_class = HFVerifyWorker.__ray_metadata__.modified_class.__ray_actor_class__
    scores, _ = verifier_class().verify(
        [r"Answer: \boxed{2}", r"Answer: \boxed{3}"],
        ["2", "2"],
        return_extracted_answer=True,
        math_verify_impl="dapo_math_verify",
    )
    if scores != [1.0, 0.0]:
        raise ValueError("Native HFVerifyWorker verification failed")
    dataset = DAPOMath17KDataset().dataset
    if len(dataset) < 20 or dataset[0]["task_name"] != "DAPOMath17K":
        raise ValueError("DAPOMath17K must provide enough complete P4 cohorts")
    for row in dataset.select(range(20)):
        if not row["messages"][0]["content"] or not row["messages"][1]["content"]:
            raise ValueError("Dataset prompt or ground truth is empty")
    return {
        "math_actor": str(MathEnvironment),
        "verifier_scores": scores,
        "dataset_rows": len(dataset),
        "dataset_fingerprint": dataset._fingerprint,
        "scope": "Actual imports, local original verifier class and dataset; no Ray actor or training result",
    }


def main() -> None:
    """Run the selected role preflight in the interpreter supplied by the launcher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("driver", "policy"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Preflight requires an explicitly CUDA-hidden child")
    config = json.loads(args.config.read_text())
    arm = verify_materialized(config)
    # Imported after the visibility guard, in the actual selected interpreter.
    import ray
    import torch

    previous = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/ultra-schema-only-not-a-daemon"
    try:
        evidence = policy_checks(config) if args.role == "policy" else driver_checks()
    finally:
        if previous is None:
            os.environ.pop("CUDA_MPS_PIPE_DIRECTORY", None)
        else:
            os.environ["CUDA_MPS_PIPE_DIRECTORY"] = previous
    if torch.cuda.is_initialized() or ray.is_initialized():
        raise ValueError("Preflight unexpectedly initialized CUDA or Ray")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "PREFLIGHT_PASS",
                "role": args.role,
                "arm": arm,
                "executable": sys.executable,
                "config_sha256": sha256(args.config),
                "cuda_initialized": False,
                "ray_initialized": False,
                "evidence": evidence,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
