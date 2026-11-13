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

"""Qualification counters for the pinned BF16 DDP gather boundary."""

from dataclasses import dataclass


@dataclass
class PublicationGuard:
    """Reject a new bucket gather after declaring its parameters published."""

    published: bool = False
    api_calls: int = 0
    existing_handle_drains: int = 0
    potential_dispatches: int = 0
    rejected_dispatches: int = 0

    def before_gather(self, *, force_sync: bool, has_handle: bool) -> None:
        self.api_calls += 1
        # The pinned force-sync branch waits and returns when a handle exists.
        if force_sync and has_handle:
            self.existing_handle_drains += 1
            return
        self.potential_dispatches += 1
        if self.published:
            self.rejected_dispatches += 1
            raise RuntimeError("New parameter gather in published-reader window")

    def before_optimizer(self) -> None:
        """Close the window at the native post-pause optimizer boundary."""
        self.published = False

    def publish_initial(self, *, bucket_states: tuple[tuple[bool, bool], ...]) -> None:
        """Allow initialized parameters before any optimizer update exists."""
        if not bucket_states or any(has_handle for _, has_handle in bucket_states):
            raise RuntimeError("Initial publication has an outstanding gather")
        self.published = True

    def publish(self, *, bucket_states: tuple[tuple[bool, bool], ...]) -> None:
        """Require all buckets dispatched with no remaining gather handles."""
        if not bucket_states or any(
            not dispatched or has_handle for dispatched, has_handle in bucket_states
        ):
            raise RuntimeError("Cannot publish incomplete parameter gather state")
        self.published = True
