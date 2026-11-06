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
"""Negative-control hooks (test only, off unless set; R-E1 suite, CONFERENCE_PLAN §5).

``THUNDERSYNC_DEBUG_VERIFY_NO_BI=1``: the learner's block verification (``verify_drafts_block_engine``) runs with
Megatron's batch-invariant mode switched off (``set_batch_invariant_mode(False)``), i.e. a verifier whose
numerics depend on the batch. The in-loop checks must report it.
"""

from __future__ import annotations

import functools

from thundersync_rl.envflag import env_flag

VERIFY_NO_BI_ENV = "THUNDERSYNC_DEBUG_VERIFY_NO_BI"


def verify_no_bi(fn):
    """Decorator: run ``fn`` outside batch-invariant mode when ``THUNDERSYNC_DEBUG_VERIFY_NO_BI`` is on."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not env_flag(VERIFY_NO_BI_ENV):
            return fn(*args, **kwargs)
        from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode

        with set_batch_invariant_mode(False):
            return fn(*args, **kwargs)

    return wrapper
