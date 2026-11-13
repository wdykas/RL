# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
"""Exercise native and qualification prepack payload-length rejection."""

import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raw_logprob_guard import RawLogprobGuardMixin, require_raw_logprob_lengths

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "nemo_rl/models/generation/megatron/megatron_worker.py"


class Batch(dict):
    @classmethod
    def from_batches(cls, values):
        return cls(values[0])

    def to(self, device):
        return self


def native_parser():
    tree = ast.parse(SOURCE.read_text())
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "_parse_result_to_batched_data_dict"
    )
    # Exact native function body, CPU tensors only; no native module import/Ray.
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    namespace = {"torch": torch, "BatchedDataDict": Batch}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, method], type_ignores=[])
            ),
            str(SOURCE),
            "exec",
        ),
        namespace,
    )
    return namespace[method.name]


class RawLogprobGuardTests(unittest.TestCase):
    def request(self, tokens, logprobs):
        return SimpleNamespace(
            request_id=7,
            finish_reason="stop",
            prompt_tokens=None,
            generated_tokens=tokens,
            generated_log_probs=logprobs,
        )

    def test_exact_native_parser_and_prepack_guard_reject_missing_tail(self):
        parser = native_parser()
        worker = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
        data = {
            "input_ids": torch.tensor([[10, 20]]),
            "input_lengths": torch.tensor([2]),
        }
        request = self.request([1561, 15, 1010, 11], [-0.3, -1.2])
        with self.assertRaisesRegex(RuntimeError, "log probabilities"):
            parser(worker, data, [request])
        with self.assertRaisesRegex(ValueError, "before packing"):
            require_raw_logprob_lengths([request], requested=True)

    def test_conditional_lengths_and_no_mutation(self):
        request = self.request([1, 11], [-0.3, -1.2])
        self.assertEqual(
            require_raw_logprob_lengths([request], requested=True)[0][
                "raw_token_count"
            ],
            2,
        )
        self.assertEqual(request.generated_tokens, [1, 11])
        self.assertEqual(request.generated_log_probs, [-0.3, -1.2])
        for values in [[], [-0.3], [-0.3, -1.2, -0.4], None]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                require_raw_logprob_lengths(
                    [self.request([1, 11], values)], requested=True
                )
        self.assertEqual(
            require_raw_logprob_lengths([self.request([1], None)], requested=False), []
        )
        self.assertEqual(
            require_raw_logprob_lengths([self.request([], [])], requested=True)[0][
                "raw_token_count"
            ],
            0,
        )

    def test_mixin_checks_before_original_and_preserves_valid_result(self):
        calls = []
        sentinel = object()

        class Native:
            def _parse_result_to_batched_data_dict(self, data, result):
                calls.append((data, result))
                return sentinel

        class Guarded(RawLogprobGuardMixin, Native):
            pass

        worker = Guarded()
        with self.assertRaises(ValueError):
            worker._parse_result_to_batched_data_dict(
                "input", [self.request([1, 11], [])]
            )
        self.assertEqual(calls, [])
        valid = [self.request([1, 11], [-0.3, -1.2])]
        self.assertIs(
            worker._parse_result_to_batched_data_dict("input", valid), sentinel
        )
        self.assertIs(calls[0][1], valid)

    def test_actual_native_direct_request_always_asks_for_logprobs(self):
        tree = ast.parse(SOURCE.read_text())
        builder = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_build_sampling_params"
        )
        settings = [
            n
            for n in ast.walk(builder)
            if isinstance(n, ast.keyword) and n.arg == "return_log_probs"
        ]
        self.assertEqual(len(settings), 1)
        self.assertIs(ast.literal_eval(settings[0].value), True)


if __name__ == "__main__":
    unittest.main()
