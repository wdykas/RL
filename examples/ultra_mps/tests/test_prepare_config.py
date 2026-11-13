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

"""CPU checks for portable paths and the closed Ultra configuration pair."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_config as config_tool


class PrepareConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model with spaces"
        self.model.mkdir()
        (self.model / "config.json").write_text(json.dumps(config_tool.MODEL_SHAPE))
        self.mcore = self.root / "mcore"
        source = self.mcore / "megatron/core/inference/contexts/dynamic_context.py"
        source.parent.mkdir(parents=True)
        source.write_text("# Test fixture: no runtime imports.\n")
        self.bridge = self.root / "bridge"
        (self.bridge / "src/megatron/bridge").mkdir(parents=True)
        self.options = dict(
            model=self.model,
            run_dir=self.root / "run",
            mcore_root=self.mcore,
            bridge_root=self.bridge,
        )

    def test_pair_preserves_work_and_native_modes(self) -> None:
        pair = {
            arm: config_tool.materialize(arm=arm, **self.options)
            for arm in config_tool.ARMS
        }
        self.assertEqual(
            set(config_tool.differences(pair["alternating"], pair["concurrent"])),
            config_tool.MODE_PATHS,
        )
        for arm, config in pair.items():
            with self.subTest(arm=arm):
                self.assertEqual(config_tool.verify_materialized(config), arm)
                self.assertEqual(
                    config["cluster"],
                    dict(num_nodes=32, gpus_per_node=4, segment_size=None),
                )
                self.assertEqual(
                    [
                        config["grpo"][k]
                        for k in (
                            "max_num_steps",
                            "num_prompts_per_step",
                            "num_generations_per_prompt",
                        )
                    ],
                    [3, 4, 16],
                )
                policy = config["policy"]
                self.assertEqual(policy["train_global_batch_size"], 64)
                optimizer = policy["megatron_cfg"]["optimizer"]
                self.assertIs(optimizer["optimizer_cpu_offload"], False)
                self.assertEqual(optimizer["optimizer_offload_fraction"], 0)
                self.assertFalse(policy["offload_optimizer_for_refit"])
                self.assertFalse(policy["offload_optimizer_for_logprob"])
                self.assertEqual(policy["generation"]["max_new_tokens"], 49152)
                generation = policy["generation"]["mcore_generation_config"]
                self.assertEqual(
                    (
                        generation["max_tokens"],
                        generation["num_speculative_tokens"],
                        generation["max_requests"],
                    ),
                    (1088, 5, 8),
                )

    def test_model_shape_and_mtp_rejected(self) -> None:
        for key, value in (
            ("hidden_size", 2048),
            ("num_nextn_predict_layers", 0),
            ("n_routed_experts", 128),
        ):
            with self.subTest(key=key):
                config = dict(config_tool.MODEL_SHAPE)
                config[key] = value
                (self.model / "config.json").write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, key):
                    config_tool.materialize(arm="concurrent", **self.options)

    def test_missing_local_model_and_source_rejected(self) -> None:
        for key in ("model", "mcore_root", "bridge_root"):
            with self.subTest(key=key):
                options = {**self.options, key: self.root / "missing"}
                with self.assertRaises(FileNotFoundError):
                    config_tool.materialize(arm="concurrent", **options)

    def test_changed_work_paths_and_worker_rejected(self) -> None:
        original = config_tool.materialize(arm="concurrent", **self.options)
        changes = (
            ("grpo", "max_num_steps", 25),
            ("policy", "worker_extension_cls_fqn", None),
            ("policy", "train_global_batch_size", 32),
            ("logger", "log_dir", "elsewhere"),
        )
        for section, key, value in changes:
            with self.subTest(key=key):
                config = copy.deepcopy(original)
                config[section][key] = value
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    config_tool.verify_materialized(config)

    def test_actor_paths_and_allocator_caps(self) -> None:
        config = config_tool.materialize(arm="concurrent", **self.options)
        self.assertEqual(
            config["logger"]["log_dir"], str(self.options["run_dir"] / "training-logs")
        )
        env = config["policy"]["megatron_cfg"]["env_vars"]
        self.assertEqual(
            env["PYTORCH_ALLOC_CONF"],
            "pinned_max_round_threshold_mb:16,pinned_max_cached_size_mb:16",
        )
        self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], env["PYTORCH_ALLOC_CONF"])
        self.assertEqual(
            env["PYTHONPATH"].split(":"),
            list(
                map(
                    str,
                    (
                        config_tool.EXAMPLE,
                        config_tool.EXAMPLE / "qualification",
                        config_tool.EXAMPLE.parents[1],
                        self.bridge / "src",
                        self.mcore,
                    ),
                )
            ),
        )
        self.assertEqual(
            config["policy"]["model_name"], config["policy"]["tokenizer"]["name"]
        )

    def test_cli_output_audit_and_no_overwrite(self) -> None:
        command = [
            sys.executable,
            str(config_tool.EXAMPLE / "prepare_config.py"),
            "--arm",
            "alternating",
        ]
        for key, value in self.options.items():
            command.extend(("--" + key.replace("_", "-"), str(value)))
        subprocess.run(command, check=True, capture_output=True, text=True)
        audit = json.loads((self.options["run_dir"] / "config-audit.json").read_text())
        self.assertEqual(
            audit["config_sha256"],
            config_tool.sha256(self.options["run_dir"] / "config.json"),
        )
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already has prepared", result.stderr)


if __name__ == "__main__":
    unittest.main()
