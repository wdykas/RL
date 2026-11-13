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
"""Parse actual per-GPU native fabric evidence without host inventories."""

import re


def parse_gpu_query(*, gpu_uuid: str, query: str) -> dict[str, str]:
    """Extract one GPU's completed native fabric registration, rejecting ambiguity."""
    uuids = re.findall(r"^\s*GPU UUID\s*:\s*(\S+)\s*$", query, re.MULTILINE)
    clusters = re.findall(r"^\s*ClusterUUID\s*:\s*(\S+)\s*$", query, re.MULTILINE)
    fabric = re.search(
        r"^    (?:GPU )?Fabric\s*\n((?:[ ]{8}.*\n)+)", query, re.MULTILINE
    )
    if uuids != [gpu_uuid] or len(clusters) != 1 or fabric is None:
        raise ValueError("Missing or ambiguous per-GPU UUID/fabric evidence")
    if not re.search(
        r"^\s*State\s*:\s*Completed\s*$", fabric[1], re.MULTILINE
    ) or not re.search(r"^\s*Status\s*:\s*Success\s*$", fabric[1], re.MULTILINE):
        raise ValueError("Native GPU fabric registration is not Completed/Success")
    return {"gpu_uuid": gpu_uuid, "cluster_uuid": clusters[0]}
