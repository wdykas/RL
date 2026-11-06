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
"""Releasable per-draft tables of generation rank 0 (long runs).

Rank 0 of a drafting engine keeps, per draft stream it submitted, the streamed token list, the engine log-probs,
a finished flag and the stream handle. ``start_drafts`` used to replace them with fresh Python lists, ``add_drafts``
to append to them; under anchor epochs (one ``start_drafts`` per epoch, ``add_drafts`` for every later cohort and
every continuation stream) they grew with every draft of the run: an 8k draft holds ~0.5 MB of Python ints and floats,
so ~0.5 GB per full-Nano step on rank 0. A ``DraftTable`` is indexed like the list it replaces (indices 0..n-1, stable
for the table's life, ``+=`` appends, iteration in index order) but an entry can be released: it then takes no memory
and reads as the table's empty value (a fresh one per read: a late frame of an aborted stream lands in a throwaway),
and writes to it are dropped. Nothing here is a function of tokens; releasing an entry never changes another one.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

_MISSING = object()


class DraftTable:
    """A list of per-draft entries whose entries can be released (sparse, index-stable)."""

    def __init__(self, empty: Callable[[], Any]):
        self._d: dict[int, Any] = {}
        self._n = 0
        self._empty = empty
        self.released = 0

    def __len__(self) -> int:
        return self._n

    def __iadd__(self, values) -> "DraftTable":
        for v in values:
            self._d[self._n] = v
            self._n += 1
        return self

    def _index(self, i: int) -> int:
        i = int(i)
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError(f"draft index {i} out of range (n={self._n})")
        return i

    def __getitem__(self, i: int) -> Any:
        v = self._d.get(self._index(i), _MISSING)
        return self._empty() if v is _MISSING else v

    def __setitem__(self, i: int, v: Any) -> None:
        i = self._index(i)
        if i in self._d:  # released entries stay released
            self._d[i] = v

    def __iter__(self) -> Iterator[Any]:
        for i in range(self._n):
            yield self[i]

    def release(self, i: int) -> bool:
        """Drop entry ``i``; True if it was live."""
        if self._d.pop(int(i), _MISSING) is _MISSING:
            return False
        self.released += 1
        return True

    def is_live(self, i: int) -> bool:
        return int(i) in self._d

    def live(self) -> int:
        """Entries held (not released)."""
        return len(self._d)


def release_entries(tables: dict[str, Any], indices, running: Callable[[int], bool]) -> dict[str, int]:
    """Release ``indices`` from every ``DraftTable`` / dict in ``tables`` unless ``running(i)``.

    Returns counts: released, still running (kept; the caller retries later), already gone, tokens freed (from the
    "tokens" table when present).
    """
    out = {"released": 0, "running": 0, "gone": 0, "tokens": 0}
    tok = tables.get("tokens")
    for i in indices:
        i = int(i)
        if running(i):
            out["running"] += 1
            continue
        if tok is not None and isinstance(tok, DraftTable) and tok.is_live(i):
            out["tokens"] += len(tok[i])
        hit = False
        for t in tables.values():
            if isinstance(t, DraftTable):
                hit = t.release(i) or hit
            elif isinstance(t, dict):
                hit = (t.pop(i, _MISSING) is not _MISSING) or hit
            elif isinstance(t, set):
                if i in t:
                    t.discard(i)
        out["released" if hit else "gone"] += 1
    return out
