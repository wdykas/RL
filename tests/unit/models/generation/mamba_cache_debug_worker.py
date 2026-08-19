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
"""Debug worker extension for the Mamba decode-cache reproduction matrix.

Injected via ``Policy(worker_extension_cls_fqn=...)`` so the freshness tests
can inspect and perturb ``MambaMixer`` state inside the Ray workers. Test-only;
requires the repo root on the workers' import path (the unit harness provides
it). See ``test_megatron_generation.py::test_mamba_decode_cache_freshness``.
"""

from typing import Any

import ray
import torch
from megatron.core.ssm.mamba_mixer import MambaMixer

from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)


def _resident_models(worker: Any) -> list[tuple[str, Any]]:
    """(tag, model) pairs for every model this worker holds."""
    models = []
    model = getattr(worker, "model", None)
    if model is not None:
        models.append(("train", model))
    inference_model = getattr(worker, "inference_model", None)
    if inference_model is not None:
        models.append(("inference", inference_model))
    return models


def _iter_mixers(model: Any):
    root = model[0] if isinstance(model, (list, tuple)) else model
    for name, module in root.named_modules():
        if isinstance(module, MambaMixer):
            yield name, module


@ray.remote(runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker"))
class MambaCacheDebugMegatronWorker(MegatronPolicyWorkerImpl):
    """MegatronPolicyWorker plus white-box probes for the A-cache repro test."""

    def debug_mamba_cache_report(self) -> list[dict[str, Any]]:
        """Per-mixer freshness report: cached ``-exp(A_log)`` vs the live parameter."""
        report = []
        for model_tag, model in _resident_models(self):
            for name, mixer in _iter_mixers(model):
                cache = getattr(mixer, "_A_neg_exp_cache", None)
                if cache is None:
                    continue
                with torch.no_grad():
                    expected = -torch.exp(mixer.A_log.float())
                    max_abs_diff = (cache - expected).abs().max().item()
                report.append(
                    {
                        "model": model_tag,
                        "mixer": name,
                        "stale_flag": bool(mixer._A_neg_exp_cache_stale),
                        "max_abs_diff": max_abs_diff,
                        "cache_fresh": max_abs_diff < 1e-5,
                    }
                )
        return report

    def debug_graphs_report(self) -> dict[str, dict[str, Any]]:
        """Graphs-engaged probe: manager presence + configured impl per model.

        Guards against the silent degeneration where a mode requests CUDA
        graphs but decodes eagerly (managers never existed at build).
        """
        out = {}
        for model_tag, model in _resident_models(self):
            root = model[0] if isinstance(model, (list, tuple)) else model
            managed = [
                name
                for name, module in root.named_modules()
                if hasattr(module, "cudagraph_manager")
            ]
            impl = getattr(getattr(root, "config", None), "cuda_graph_impl", None)
            out[model_tag] = {
                "cuda_graph_impl": impl,
                "num_managed_modules": len(managed),
            }
        return out

    def debug_perturb_mamba_a_log(self, delta: float) -> int:
        """Shift every training-model ``A_log`` by ``delta`` (log space).

        A deterministic stand-in for RL drift: +1.0 changes every head's decay
        rate by a factor of e, so a stale consumer fails parity loudly instead
        of within noise. Returns the number of mixers perturbed.
        """
        count = 0
        model = getattr(self, "model", None)
        if model is None:
            return 0
        with torch.no_grad():
            for _, mixer in _iter_mixers(model):
                mixer.A_log.add_(delta)
                count += 1
        return count

    def debug_toggle_train_mode(self) -> None:
        """Emulate a training phase's cache arming: ``train()`` then ``eval()``."""
        model = getattr(self, "model", None)
        if model is None:
            return
        root = model[0] if isinstance(model, (list, tuple)) else model
        root.train()
        root.eval()
