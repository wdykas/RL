# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Reproduce the FlashInfer probability-CDF last-token sampling bias.

NeMo-RL previously forced MCore's FlashInfer backend. With top_p=1.0, that
backend samples a float32 probability CDF. If its accumulated mass is slightly
below the uniform random draw, FlashInfer returns the last positive vocabulary
id. The returned logprob is still the model's real logprob for that id, which
makes the sample look statistically impossible while training/inference
logprobs continue to agree.

This script constructs a two-token categorical distribution with a small mass
shortfall comparable to the observed failure rate. The final token has
logprob=-40. A correct categorical sampler effectively never selects it, but
FlashInfer's probability samplers select it at roughly the shortfall rate.
The logits sampler and torch.multinomial are included as controls.

The shortfall is deliberately constructed so this is a quick, deterministic
mechanism repro. In a real model it comes from float32 softmax/CDF reductions
over a large vocabulary, not from intentionally malformed probabilities.

Usage (requires a CUDA environment with flashinfer-python):
    python tools/model_diagnostics/7.flashinfer_sampling_tail_bias.py
    python tools/model_diagnostics/7.flashinfer_sampling_tail_bias.py \
        --draws 10000000 --shortfall 3.1e-6
"""

import argparse
import math
from importlib.metadata import PackageNotFoundError, version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--draws",
        type=int,
        default=5_000_000,
        help="Number of samples per sampler (default: 5,000,000).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Samples per GPU launch (default: 1,000,000).",
    )
    parser.add_argument(
        "--shortfall",
        type=float,
        default=3.1e-6,
        help="Missing probability mass used to expose the fallback (default: 3.1e-6).",
    )
    parser.add_argument(
        "--tail-logprob",
        type=float,
        default=-40.0,
        help="True logprob of the last token (default: -40).",
    )
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.draws <= 0:
        raise ValueError("--draws must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if not 0.0 < args.shortfall < 1.0:
        raise ValueError("--shortfall must be in (0, 1)")
    if args.tail_logprob >= 0.0:
        raise ValueError("--tail-logprob must be negative")

    # Keep --help usable outside the optional MCore/CUDA environment.
    try:
        import flashinfer
        import torch
    except ImportError as error:
        raise SystemExit(
            "This diagnostic requires torch and flashinfer-python. "
            "Run it in the NeMo-RL MCore worker environment."
        ) from error

    if not torch.cuda.is_available():
        raise SystemExit("This diagnostic requires a CUDA GPU.")

    device = torch.device("cuda")
    tail_probability = math.exp(args.tail_logprob)
    head_probability = 1.0 - args.shortfall - tail_probability
    if head_probability <= 0.0:
        raise ValueError("--shortfall and --tail-logprob leave no head probability")

    # Store exactly what the CUDA kernels receive, then compute expectations
    # from those fp32 values rather than the higher-precision CLI inputs.
    probs = torch.tensor(
        [[head_probability, tail_probability]], dtype=torch.float32, device=device
    )
    stored_head, stored_tail = (float(value) for value in probs[0].cpu())
    stored_total = stored_head + stored_tail
    stored_shortfall = 1.0 - stored_total
    normalized_tail_probability = stored_tail / stored_total
    expected_tail_count = args.draws * normalized_tail_probability
    expected_fallback_count = args.draws * stored_shortfall
    logits = probs.log()

    def flashinfer_count(kind: str) -> int:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        tail_count = 0
        for start in range(0, args.draws, args.chunk_size):
            count = min(args.chunk_size, args.draws - start)
            indices = torch.zeros(count, dtype=torch.int32, device=device)
            if kind == "top_p_probs":
                samples = flashinfer.sampling.top_p_sampling_from_probs(
                    probs,
                    1.0,
                    indices=indices,
                    deterministic=True,
                    generator=generator,
                )
            elif kind == "probs":
                samples = flashinfer.sampling.sampling_from_probs(
                    probs,
                    indices=indices,
                    deterministic=True,
                    generator=generator,
                )
            elif kind == "logits":
                samples = flashinfer.sampling.sampling_from_logits(
                    logits,
                    indices=indices,
                    deterministic=True,
                    generator=generator,
                )
            else:  # pragma: no cover - internal call sites are fixed above
                raise AssertionError(f"unknown sampler kind: {kind}")
            tail_count += int((samples == 1).sum().item())
        return tail_count

    def torch_count() -> int:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        tail_count = 0
        for start in range(0, args.draws, args.chunk_size):
            count = min(args.chunk_size, args.draws - start)
            samples = torch.multinomial(
                probs[0], count, replacement=True, generator=generator
            )
            tail_count += int((samples == 1).sum().item())
        return tail_count

    results = [
        (
            "FlashInfer top_p_sampling_from_probs(p=1)",
            flashinfer_count("top_p_probs"),
        ),
        ("FlashInfer sampling_from_probs", flashinfer_count("probs")),
        ("FlashInfer sampling_from_logits", flashinfer_count("logits")),
        ("torch.multinomial", torch_count()),
    ]

    try:
        flashinfer_version = version("flashinfer-python")
    except PackageNotFoundError:
        flashinfer_version = getattr(flashinfer, "__version__", "unknown")

    print(f"FlashInfer version: {flashinfer_version}")
    print(f"draws per sampler: {args.draws:,}")
    print(f"stored fp32 mass: {stored_total:.10f}")
    print(f"stored mass shortfall: {stored_shortfall:.10g}")
    print(f"last-token logprob: {math.log(normalized_tail_probability):.6f}")
    print(f"correct expected last-token count: {expected_tail_count:.6g}")
    print(f"expected CDF-fallback count: {expected_fallback_count:.3f}\n")

    name_width = max(len(name) for name, _ in results)
    print(f"{'sampler':<{name_width}}  {'last-id count':>13}  {'observed rate':>13}")
    print(f"{'-' * name_width}  {'-' * 13}  {'-' * 13}")
    for name, count in results:
        print(f"{name:<{name_width}}  {count:13,d}  {count / args.draws:13.6g}")

    top_p_count = results[0][1]
    logits_count = results[2][1]
    torch_tail_count = results[3][1]
    if top_p_count == 0:
        raise SystemExit(
            "\nINCONCLUSIVE: no fallback was drawn. Increase --draws or --shortfall."
        )
    if logits_count != 0 or torch_tail_count != 0:
        raise SystemExit(
            "\nUnexpected control result: a safe sampler selected the synthetic tail."
        )

    print(
        "\nCONFIRMED: the probability-CDF path selected the final token many "
        "orders of magnitude more often than its reported probability."
    )


if __name__ == "__main__":
    main()
