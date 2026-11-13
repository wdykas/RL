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

"""Synthetic CPU records exercise coverage, ownership and clock rejection."""

import unittest

from ultra_qualification.validate import check_catalog, validate


def catalog(rank: int) -> list[dict]:
    rows = []
    modules = [f"decoder.layers.{layer}.mlp" for layer in range(48)]
    modules.append("mtp.layers.0.mtp_model_layer.layers.1.mlp")
    for module in modules:
        for projection, shape in (
            ("linear_fc1", [5120, 2048]),
            ("linear_fc2", [2048, 5120]),
        ):
            for local in range(8):
                rows.append(
                    {
                        "name": f"{module}.experts.{projection}.weight{local}",
                        "resolved_name": f"{module}.experts.{projection}.weight{rank * 8 + local}",
                        "shape": shape,
                        "stride": [shape[1], 1],
                        "bytes": 20971520,
                        "dtype": "torch.bfloat16",
                        "parameter": True,
                        "aliases_mapped_source": True,
                    }
                )
    return rows


def fixture() -> list[dict]:
    records = []
    for rank in range(64):
        tensors = catalog(rank)

        def row(
            event: str,
            *,
            engine: bool,
            sequence: int,
            clock: int,
            calls: int = 0,
            epoch: int | None = None,
            full: bool = False,
        ) -> dict:
            return {
                "event": event,
                "rank": rank,
                "hostname": "same-node",
                "pid": 1000 + rank + (100 if engine else 0),
                "process_role": "engine" if engine else "learner",
                "role": "engine" if engine else "learner",
                "device_uuid": f"GPU-{rank}",
                "rank_coordinates": {"tp": [rank % 8, 8], "pp": [0, 1]},
                "clock_domain": ["boot-id", "time-namespace"],
                "concurrent": True,
                "sequence": sequence,
                "monotonic_ns": clock,
                "wall_ns": clock + 100000,
                "optimizer_calls": calls,
                "weight_epoch": epoch,
                "publication": {"rejected_dispatches": 0},
                "bucket_states": [[True, False]],
                "ep_coordinate": [rank, 64],
                "catalog": tensors if full else None,
                "expert_storage_digest": f"stable-{rank}",
                "expert_stacks": [
                    {"module": f"module-{i}", "built": True} for i in range(49)
                ],
                "memory": {
                    "allocated": 10,
                    "reserved": 20,
                    "cumulative_peak_allocated": 30,
                    "cumulative_peak_reserved": 40,
                },
            }

        for sequence, event in enumerate(
            (
                "constructed",
                "train_begin_before",
                "train_begin_after",
                "first_microbatch_returned",
                "optimizer_before",
                "optimizer_returned",
                "parameters_published",
            ),
            1,
        ):
            clocks = (1, 5, 6, 7, 20, 30, 35)
            records.append(
                row(
                    event,
                    engine=False,
                    sequence=sequence,
                    clock=clocks[sequence - 1],
                    calls=int(event in ("optimizer_returned", "parameters_published")),
                    full=event == "constructed",
                )
            )
        for sequence, event in enumerate(
            (
                "constructed",
                "bound",
                "generation_prepared",
                "reader_drained",
                "private_buffers_published",
                "reader_before_unpause",
                "reader_unpaused",
            ),
            1,
        ):
            clocks = (1, 2, 3, 10, 40, 50, 60)
            record = row(
                event,
                engine=True,
                sequence=sequence,
                clock=clocks[sequence - 1],
                epoch=int(sequence >= 6),
                full=event in ("bound", "generation_prepared"),
            )
            if event == "private_buffers_published":
                record["router_biases"] = [
                    {"private": True, "equal": True} for _ in range(49)
                ]
            records.append(record)
    return records


class ValidationTests(unittest.TestCase):
    def test_explicit_world128_keeps_EP64_catalog_and_rejects_missing_replica(self):
        import copy

        original = fixture()
        extra = copy.deepcopy(original)
        for row in extra:
            row["rank"] += 64
            row["pid"] += 10000
            row["device_uuid"] += "-replica2"
        combined = original + extra
        self.assertTrue(
            validate(combined, concurrent=True, updates=1, world_size=128)[
                "observer_checks_pass"
            ]
        )
        with self.assertRaises(ValueError):
            validate(original, concurrent=True, updates=1, world_size=128)

    def test_all_ranks_and_same_host_lifecycle(self) -> None:
        result = validate(fixture(), concurrent=True, updates=1)
        self.assertTrue(result["observer_checks_pass"])
        self.assertFalse(result["native_full_qualification_pass"])
        self.assertEqual(result["processes"], 128)

    def test_alternating_shared_actors_keep_separate_engine_model_catalog(self) -> None:
        records = []
        for record in fixture():
            if record["process_role"] == "engine":
                if record["event"] != "generation_prepared":
                    continue
                record["process_role"] = "learner"
                record["pid"] -= 100
            record["concurrent"] = False
            records.append(record)
        for rank in range(64):
            process = sorted(
                (row for row in records if row["rank"] == rank),
                key=lambda row: row["monotonic_ns"],
            )
            for sequence, record in enumerate(process, 1):
                record["sequence"] = sequence
        result = validate(records, concurrent=False, updates=1)
        self.assertEqual(result["processes"], 64)
        self.assertTrue(result["observer_checks_pass"])

    def test_missing_engine_rank_rejected(self) -> None:
        records = [
            row
            for row in fixture()
            if not (row["rank"] == 63 and row["process_role"] == "engine")
        ]
        with self.assertRaisesRegex(ValueError, "Expected 64"):
            validate(records, concurrent=True, updates=1)

    def test_clock_namespace_and_late_pause_rejected(self) -> None:
        for field, value, message in (
            ("clock_domain", ["other-boot", "time"], "clock namespaces"),
            ("monotonic_ns", 21, "publication/reader order"),
        ):
            with self.subTest(field=field):
                records = fixture()
                target = next(
                    row
                    for row in records
                    if row["rank"] == 0
                    and row["event"]
                    == ("bound" if field == "clock_domain" else "reader_drained")
                )
                target[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate(records, concurrent=True, updates=1)

    def test_global_expert_ownership_and_private_fallback_rejected(self) -> None:
        record = next(row for row in fixture() if row["event"] == "bound")
        record["catalog"][0]["resolved_name"] = (
            "decoder.layers.0.mlp.experts.linear_fc1.weight999"
        )
        with self.assertRaisesRegex(ValueError, "global IDs"):
            check_catalog(record, shared=True)
        record["catalog"][0]["resolved_name"] = (
            "decoder.layers.0.mlp.experts.linear_fc1.weight0"
        )
        record["catalog"][0]["aliases_mapped_source"] = False
        with self.assertRaisesRegex(ValueError, "does not alias"):
            check_catalog(record, shared=True)


if __name__ == "__main__":
    unittest.main()
