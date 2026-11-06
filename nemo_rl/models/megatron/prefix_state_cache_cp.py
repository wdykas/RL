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
"""Context-parallel layout of the learner prefix-state cache (first build).

Layout (contiguous CP chunks at Mamba-chunk boundaries, the CP design of 14bg item 4):
- K / V of a row's prefix stay sharded in place: CP rank r keeps positions [lo_r, hi_r) of every attention layer;
- a boundary's Mamba state (fp32 SSM state + conv inputs) lives on the rank that owns the boundary's position;
- a continuation piece [s, n) runs on the rank owning boundary s (the last prefix rank); its new queries attend
  to every rank's K / V shard.

Operations (each a function of the per-rank ``PrefixStateCache``; all ranks of the CP group call them together):
- ``send_entry`` / ``recv_entry``: hand a row's boundary state (and, in this first build, its cached K / V) from the
  rank that produced it to the next CP rank -- the state chain of a CP prefill;
- ``drop_kv_below``: a rank frees (here: poisons with NaN, so a missing gather cannot pass silently) the K / V
  positions it does not own;
- ``gather_kv``: every rank sends its owned K / V shard of a row to the piece's rank, which writes it into its entry
  before the piece forward. The piece's logits then equal the uncached whole-row forward bitwise (the attention
  inputs are the same K / V values; Mamba restarts from the owner's state).

Per rank cost (row length L, CP ranks c, attention layers A, KV heads per TP rank h, head dim d, piece p):
K / V held 2 A h d L / c x 2 bytes (fp16) instead of 2 A h d L x 2; boundary states only on their owner
(Ultra TP8: ~0.05 GB per boundary). A piece moves (c-1)/c of the prefix K / V to its rank:
2 A h d L (c-1)/c x 2 bytes (Ultra, 12 attention layers, 2 KV heads replicated under TP8, 128k: 1.4 GB, ~1.5 ms at
~0.9 TB/s) and holds it transiently. The scalable form (not built): no gather -- each rank computes FA4 partials
(o, lse) of the piece's queries against its own shard on split-KV blocks anchored at absolute position 0 (shard
boundaries multiples of the split size S), and the piece's rank merges every split in split order with FA4's
combine, which is independent of the number of splits (bi_split_kv): same bits, moves p x heads x (d + 1) fp32 per
rank instead of the prefix K / V.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


def _entry_to_obj(e: Any) -> dict:
    return {
        "tokens": e.tokens,
        "kv_len": e.kv_len,
        "kv": {ln: (k.cpu(), v.cpu()) for ln, (k, v) in e.kv.items()},
        "states": {
            s: {ln: (c.cpu(), x.cpu()) for ln, (c, x) in per.items()} for s, per in e.states.items()
        },
    }


def send_entry(cache: Any, key: str, dst: int, group: Any = None) -> None:
    """Send this rank's entry of ``key`` (boundary states + cached K / V) to global rank ``dst``."""
    dist.send_object_list([_entry_to_obj(cache.entries[key])], dst=dst, group=group)


def recv_entry(cache: Any, key: str, src: int, group: Any = None) -> None:
    """Install the entry of ``key`` sent by global rank ``src`` (on this rank's GPU)."""
    from nemo_rl.models.megatron.prefix_state_cache import _Entry

    box = [None]
    dist.recv_object_list(box, src=src, group=group)
    o = box[0]
    dev = torch.cuda.current_device()
    cache.entries[key] = _Entry(
        tokens=o["tokens"],
        kv_len=o["kv_len"],
        kv={ln: (k.to(dev), v.to(dev)) for ln, (k, v) in o["kv"].items()},
        states={s: {ln: (c.to(dev), x.to(dev)) for ln, (c, x) in per.items()} for s, per in o["states"].items()},
    )


def drop_kv_below(cache: Any, key: str, lo: int) -> None:
    """This rank does not own K / V positions [0, lo) of ``key``: poison them (NaN) so only a gather can restore them."""
    e = cache.entries[key]
    for k, v in e.kv.values():
        k[:lo].fill_(float("nan"))
        v[:lo].fill_(float("nan"))


def gather_kv(cache: Any, key: str, owner: int, ranges: list[tuple[int, int]], group: Any = None) -> None:
    """Every CP rank sends its owned K / V shard ``ranges[cp_rank]`` of ``key`` to global rank ``owner``.

    ``ranges``: per CP-group rank (group order), the [lo, hi) positions it owns. The owner writes each shard into
    its own entry; the other ranks' entries are unchanged.
    """
    ranks = dist.get_process_group_ranks(group) if group is not None else list(range(dist.get_world_size()))
    me = dist.get_rank()
    layers = sorted(cache.entries[key].kv) if key in cache.entries else None
    box = [layers]
    dist.broadcast_object_list(box, src=owner, group=group)
    layers = box[0]
    for j, r in enumerate(ranks):
        if r == owner:
            continue
        lo, hi = ranges[j]
        for ln in layers:
            if me == r:
                k, v = cache.entries[key].kv[ln]
                dist.send(k[lo:hi].contiguous(), dst=owner, group=group)
                dist.send(v[lo:hi].contiguous(), dst=owner, group=group)
            elif me == owner:
                k, v = cache.entries[key].kv[ln]
                bk = torch.empty_like(k[lo:hi])
                bv = torch.empty_like(v[lo:hi])
                dist.recv(bk, src=r, group=group)
                dist.recv(bv, src=r, group=group)
                k[lo:hi].copy_(bk)
                v[lo:hi].copy_(bv)
