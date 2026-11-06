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
"""MegatronGenerationMixin._merge_stop_strings: the same list in every process (source: rl-determinism 69c3fd21).

The merged stop strings came from ``list(set)``, whose order follows the string hash seed, so two processes (e.g. two
engine ranks, or a run and its resume) passed differently ordered stop lists to the engine."""

import os
import subprocess
import sys

SNIPPET = r"""
from types import SimpleNamespace
from nemo_rl.models.generation.megatron.megatron_worker import MegatronGenerationMixin
w = SimpleNamespace(cfg={"generation": {"stop_strings": ["a", "bb", "ccc", "dddd", "eeeee", "ffffff"]}})
print(MegatronGenerationMixin._merge_stop_strings(w, [["x", "yy"], None, ["zzz"]]))
"""


def _run(seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=seed, CUDA_VISIBLE_DEVICES="")
    out = subprocess.run([sys.executable, "-c", SNIPPET], env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


def test_merged_stop_strings_do_not_depend_on_the_hash_seed():
    results = {_run(seed) for seed in ("0", "1", "2", "3", "4", "5")}
    assert len(results) == 1, results
    assert results.pop() == str(sorted(["a", "bb", "ccc", "dddd", "eeeee", "ffffff", "x", "yy", "zzz"]))
