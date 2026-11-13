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

"""Exercise state transitions around native gather publication."""

import unittest

from ultra_qualification.publication import PublicationGuard


class PublicationTests(unittest.TestCase):
    def test_optimizer_publish_next_forward_and_reset(self) -> None:
        guard = PublicationGuard()
        guard.publish_initial(bucket_states=((False, False),))
        guard.before_optimizer()
        guard.before_gather(force_sync=False, has_handle=False)
        guard.before_gather(force_sync=True, has_handle=True)
        guard.publish(bucket_states=((True, False), (True, False)))
        self.assertTrue(guard.published)
        self.assertEqual(guard.potential_dispatches, 1)
        self.assertEqual(guard.existing_handle_drains, 1)
        guard.before_optimizer()
        guard.before_gather(force_sync=True, has_handle=False)
        guard.publish(bucket_states=((True, False),))
        self.assertEqual(guard.rejected_dispatches, 0)

    def test_force_sync_existing_handle_is_not_a_new_dispatch(self) -> None:
        guard = PublicationGuard()
        guard.publish(bucket_states=((True, False),))
        guard.before_gather(force_sync=True, has_handle=True)
        self.assertEqual(guard.api_calls, 1)
        self.assertEqual(guard.existing_handle_drains, 1)
        self.assertEqual(guard.potential_dispatches, 0)

    def test_new_dispatch_rejected_before_caller_enqueues(self) -> None:
        for force in (False, True):
            with self.subTest(force_sync=force):
                guard = PublicationGuard()
                guard.publish(bucket_states=((True, False),))
                enqueued = []
                with self.assertRaisesRegex(RuntimeError, "published-reader"):
                    guard.before_gather(force_sync=force, has_handle=False)
                    enqueued.append(True)
                self.assertFalse(enqueued)
                self.assertEqual(guard.rejected_dispatches, 1)

    def test_incomplete_and_empty_publication_rejected(self) -> None:
        for states in ((), ((False, False),), ((True, True),)):
            with self.subTest(states=states):
                guard = PublicationGuard()
                with self.assertRaises(RuntimeError):
                    guard.publish(bucket_states=states)
                self.assertFalse(guard.published)

    def test_initial_publication_cannot_hide_an_inflight_gather(self) -> None:
        guard = PublicationGuard()
        with self.assertRaises(RuntimeError):
            guard.publish_initial(bucket_states=((False, True),))
        self.assertFalse(guard.published)


if __name__ == "__main__":
    unittest.main()
