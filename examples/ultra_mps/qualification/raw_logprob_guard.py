# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
"""Reject incomplete raw native request payloads before packing.

Native direct generation requests return_log_probs=True. The qualification
worker checks lengths before invoking the original native packing method.
This detects missing payload entries; equality alone does not prove indexing
or numerical correctness of the entries.
"""


def require_raw_logprob_lengths(results, *, requested: bool) -> list[dict]:
    """Check complete raw request payloads without padding, trimming or mutation."""
    if not requested:
        return []
    evidence = []
    failures = []
    for index, request in enumerate(results):
        tokens = request.generated_tokens
        logprobs = request.generated_log_probs
        row = {
            "index": index,
            "request_id": getattr(request, "request_id", None),
            "finish_reason": getattr(request, "finish_reason", None),
            "raw_token_count": None if tokens is None else len(tokens),
            "raw_logprob_count": None if logprobs is None else len(logprobs),
        }
        evidence.append(row)
        if tokens is None or logprobs is None or len(tokens) != len(logprobs):
            failures.append(row)
    if failures:
        raise ValueError(
            "Native raw generated-token/log-probability payload mismatch before "
            f"packing (no padding/trimming permitted): {failures!r}"
        )
    return evidence


class RawLogprobGuardMixin:
    """Optional research adapter for the native always-requested-logprobs path."""

    def _parse_result_to_batched_data_dict(self, data, result):
        # Native direct _build_sampling_params sets return_log_probs=True; this
        # seam must be revisited if that request contract becomes configurable.
        require_raw_logprob_lengths(result, requested=True)
        return super()._parse_result_to_batched_data_dict(data, result)
