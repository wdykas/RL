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

"""Materialize the fixed 128-GPU Ultra Math qualification with local paths."""

import argparse
import hashlib
import json
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parent
ARMS = ("alternating", "concurrent")
MODEL_SHAPE = {
    "model_type": "nemotron_h",
    "architectures": ["NemotronHForCausalLM"],
    "num_hidden_layers": 108,
    "hidden_size": 8192,
    "vocab_size": 131072,
    "num_attention_heads": 64,
    "num_key_value_heads": 2,
    "n_routed_experts": 512,
    "num_experts_per_tok": 22,
    "moe_intermediate_size": 5120,
    "moe_latent_size": 2048,
    "num_nextn_predict_layers": 1,
    "mtp_hybrid_override_pattern": "*E",
    "hybrid_override_pattern": "MEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
    "mamba_num_heads": 256,
    "mamba_head_dim": 64,
    "ssm_state_size": 128,
    "moe_shared_expert_intermediate_size": 10240,
}
MODE_PATHS = {
    "policy.generation.colocated.enabled",
    "policy.generation.colocated.engine_sm_fraction",
    "policy.generation.refit_transport",
    "policy.generation.mcore_generation_config.refit_backend",
}


def sha256(path: Path) -> str:
    """Return the SHA256 of one input file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def differences(left: object, right: object, prefix: str = "") -> list[str]:
    """Return changed leaf paths, including missing fields."""
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            path = f"{prefix}.{key}".lstrip(".")
            if key not in left or key not in right:
                result.append(path)
            else:
                result.extend(differences(left[key], right[key], path))
        return result
    return [] if left == right else [prefix]


def validate_model(model: Path) -> dict:
    """Check local checkpoint metadata; weight contents still require native loading."""
    if not model.is_dir():
        raise ValueError("--model must be an existing local checkpoint directory")
    metadata = model / "config.json"
    config = json.loads(metadata.read_text())
    for key, value in MODEL_SHAPE.items():
        if config.get(key) != value:
            raise ValueError(f"Unsupported Ultra model field {key}: expected {value!r}")
    return {
        "config_sha256": sha256(metadata),
        "shape": MODEL_SHAPE,
        "scope": "Metadata only; no checkpoint payload, tokenizer or numerical parity proof",
    }


def materialize(
    *, arm: str, model: Path, run_dir: Path, mcore_root: Path, bridge_root: Path
) -> dict:
    """Apply only portable paths and the common qualification worker profile."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    model = model.expanduser().resolve(strict=True)
    run_dir = run_dir.expanduser().resolve()
    mcore_root = mcore_root.expanduser().resolve(strict=True)
    bridge_root = bridge_root.expanduser().resolve(strict=True)
    validate_model(model)
    if not (
        mcore_root / "megatron/core/inference/contexts/dynamic_context.py"
    ).is_file():
        raise ValueError("--mcore-root must contain the actual Megatron source tree")
    if not (bridge_root / "src/megatron/bridge").is_dir():
        raise ValueError("--bridge-root must contain src/megatron/bridge")
    pair = {
        name: json.loads((EXAMPLE / "configs" / f"{name}.json").read_text())
        for name in ARMS
    }
    if set(differences(pair["alternating"], pair["concurrent"])) != MODE_PATHS:
        raise ValueError("Templates differ beyond the four native mode fields")
    config = pair[arm]
    config["policy"]["model_name"] = str(model)
    config["policy"]["tokenizer"]["name"] = str(model)
    config["logger"]["log_dir"] = str(run_dir / "training-logs")
    config["checkpointing"]["checkpoint_dir"] = str(run_dir / "checkpoints")
    config["policy"]["worker_extension_cls_fqn"] = "math_worker.MathAuditWorker"
    config["policy"]["megatron_cfg"]["env_vars"].update(
        {
            "NRL_ULTRA_AUDIT_DIR": str(run_dir / "observer"),
            "ULTRA_DIRECT64_RUN_DIR": str(run_dir),
            "NRL_ULTRA_MTP_FIX_SOURCE": str(EXAMPLE / "mtp_fix/SOURCE.json"),
            "NRL_ULTRA_MCORE_SOURCE": str(mcore_root),
            "ULTRA_BRIDGE_ROOT": str(bridge_root),
            "PYTHONPATH": ":".join(
                map(
                    str,
                    (
                        EXAMPLE,
                        EXAMPLE / "qualification",
                        EXAMPLE.parents[1],
                        bridge_root / "src",
                        mcore_root,
                    ),
                )
            ),
        }
    )
    return config


def verify_materialized(config: dict) -> str:
    """Reject altered workload, mode, instrumentation or materialized paths."""
    arm = (
        "alternating"
        if config["policy"]["generation"]["colocated"]["enabled"]
        else "concurrent"
    )
    env = config["policy"]["megatron_cfg"]["env_vars"]
    expected = materialize(
        arm=arm,
        model=Path(config["policy"]["model_name"]),
        run_dir=Path(env["ULTRA_DIRECT64_RUN_DIR"]),
        mcore_root=Path(env["NRL_ULTRA_MCORE_SOURCE"]),
        bridge_root=Path(env["ULTRA_BRIDGE_ROOT"]),
    )
    changed = differences(config, expected)
    if changed:
        raise ValueError(f"Prepared configuration changed: {changed}")
    return arm


def prepare(
    *, arm: str, model: Path, run_dir: Path, mcore_root: Path, bridge_root: Path
) -> dict:
    """Write fresh config and audit files, refusing to replace an existing run."""
    config = materialize(
        arm=arm,
        model=model,
        run_dir=run_dir,
        mcore_root=mcore_root,
        bridge_root=bridge_root,
    )
    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if any((run_dir / name).exists() for name in ("config.json", "config-audit.json")):
        raise FileExistsError("Run already has prepared config/audit files")
    target = run_dir / "config.json"
    with target.open("x") as stream:
        stream.write(json.dumps(config, indent=2) + "\n")
    audit = {
        "arm": arm,
        "config_sha256": sha256(target),
        "template_sha256": sha256(EXAMPLE / "configs" / f"{arm}.json"),
        "model": validate_model(Path(config["policy"]["model_name"])),
        "contract": {
            "updates": 3,
            "accepted_sequences_per_update": 64,
            "world_size": 128,
            "nodes": 32,
            "gpus_per_node": 4,
            "required_nvlink_islands": 2,
            "gpus_per_island": 64,
            "expert_optimizer_group_size": 2,
            "dense_optimizer_group_size": 16,
        },
        "scope": "Prepared configuration only; actual topology, work, gradients and memory fit require qualification",
    }
    with (run_dir / "config-audit.json").open("x") as stream:
        stream.write(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> None:
    """Prepare the selected native arm using explicit local paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mcore-root", type=Path, required=True)
    parser.add_argument("--bridge-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
