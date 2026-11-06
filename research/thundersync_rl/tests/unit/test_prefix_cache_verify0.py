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
"""verify0 capture for the learner prefix-state cache: bookkeeping on CPU.

The bitwise property (first piece from a verify0-captured state == whole-row forward) needs
GPU kernels: tests/functional/check_prefix_cache.py --mode v0 (Nano L8, EP2, negative control).
"""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from thundersync_rl.prefix_cache import PrefixCacheDirectory, lpt_owners

GIB = 2**30


def _cache(chunk=4, max_keep=4, page=8, n_pages=64, n_slots=32, max_len=256):
    """A PrefixStateCache on CPU pools: one attention layer (1 head x 1 dim) and one Mamba layer."""
    from nemo_rl.models.megatron.prefix_state_cache import PrefixStateCache, _Pools

    c = PrefixStateCache.__new__(PrefixStateCache)
    c.chunk, c.max_keep, c.gpu_bytes, c.max_len = chunk, max_keep, 1 << 40, max_len
    c.entries = OrderedDict()
    c.weights_version = None
    c.attention_layers, c.mamba_layers = [1], [2]
    c.stats = {k: 0 for k in ("hits", "rows", "tokens_skipped", "tokens_forwarded", "v0_captured",
                              "v0_hits", "v0_trimmed", "v0_trim_dropped", "pool_fallback")}
    c.negctl, c.graphs, c.last_from_v0 = 0.0, True, []
    c.pools = _Pools({1: (1, 1, torch.float32)}, {2: (1, 1, torch.float32, 1, 1, 1)}, page, n_pages,
                     n_slots, -(-(max_len + page) // page), max_keep, torch.device("cpu"))
    return c


def _fwd(c, keys, rows, starts, floors=None, pins=None):
    """``cache.forward`` with the hooks emulated through the same device metadata.

    Before the writes it checks what a real forward reads: every cached position [0, start)
    of a row holds that row's token as K (page table), and the input state slot holds the
    state of boundary ``start`` (the zero slot for 0). Writes: K = token id at every
    forwarded position (``kv_write_index``), and each kept boundary's state = the boundary.
    """
    from nemo_rl.models.megatron.prefix_state_cache import TRASH_SLOT, _ACTIVE, kv_write_index

    with torch.no_grad(), c.forward(keys, rows, starts, pins, floors):
        call = _ACTIVE[0]
        meta, p = call.meta, c.pools
        s = max(r.length for r in call.rows)
        for b, r in enumerate(call.rows):
            assert int(meta["start"][b]) == r.start and int(meta["length"][b]) == r.length
            for pos in range(r.start):
                pg = int(meta["page_table"][b, pos // p.page])
                assert p.k[1][pg, pos % p.page, 0, 0] == float(r.tokens[pos]), (b, pos)
            assert float(p.ssm[2][int(meta["slot_in"][b])].item()) == float(r.start)
        valid, pg, off = kv_write_index(meta, s, p.page, p.maxp)
        tok = torch.zeros(len(call.rows), s)
        for b, r in enumerate(call.rows):
            tok[b, : r.length] = r.tokens[r.start :].float()
        p.k[1].index_put_((pg.reshape(-1), off.reshape(-1)), tok.reshape(-1, 1, 1))
        for b, r in enumerate(call.rows):
            for j in range(p.K):
                slot = int(meta["keep_slot"][b, j])
                if slot != TRASH_SLOT:
                    p.ssm[2][slot] = float(r.start + int(meta["keep_off"][b, j]))
    _check_ownership(c)


def _check_ownership(c):
    """Every page / slot is free or owned by exactly one resident entry (no leak, no sharing)."""
    from nemo_rl.models.megatron.prefix_state_cache import TRASH_PAGE, TRASH_SLOT

    p = c.pools
    owned_p = [pg for e in c.entries.values() if e.host is None for pg in e.pages]
    owned_s = [s for e in c.entries.values() if e.host is None for s in e.states.values()]
    assert len(owned_p) == len(set(owned_p)) and len(owned_s) == len(set(owned_s))
    assert sorted(owned_p + p.free_pages) == list(range(TRASH_PAGE + 1, p.n_pages))
    assert sorted(owned_s + p.free_slots) == list(range(TRASH_SLOT + 1, p.n_slots))
    for e in c.entries.values():
        if e.host is None:
            assert len(e.pages) == -(-e.kv_len // p.page)
            for b, slot in e.states.items():
                assert float(p.ssm[2][slot].item()) == float(b)


# ---- learner: capture boundaries, trim ---------------------------------------------------
def test_keep_from_the_prompt_boundary():
    c = _cache(chunk=4, max_keep=4)
    # verify0 of prompt 10 + draft: boundaries from 8 (the prompt's boundary), not from 0
    assert c.keep(0, 21, lo=10) == [8, 12, 16, 20]
    assert c.keep(0, 64, lo=10) == [8, 24, 44, 60]  # long: first, last, evenly spaced
    assert c.keep(0, 9, lo=3) == [0, 4, 8]  # prompt below one chunk: from 0 as before
    assert c.keep(0, 9) == [0, 4, 8] and c.keep(0, 3, lo=1) == []


def test_capture_is_committed_as_v0_and_trim_keeps_the_first_piece_boundary():
    c = _cache(chunk=4, page=8)
    prompt, draft = torch.arange(100, 110), torch.arange(200, 230)  # plen 10, n 40
    row = torch.cat([prompt, draft])
    _fwd(c, ["a"], [row], [0], floors=[10])
    e = c.entries["a"]
    assert e.origin == "v0" and sorted(e.states) == [8, 16, 28, 36] and e.kv_len == 36
    assert len(e.pages) == 5
    # rejection at draft position 13 (row position 23): first piece prompt = 10 + 13 + 1
    rej = torch.cat([row[:23], torch.tensor([7]), torch.arange(300, 304)])
    start_untrimmed = c.start("a", rej, 24)
    assert start_untrimmed == 16
    free_s, free_p = len(c.pools.free_slots), len(c.pools.free_pages)
    c.trim("a", 23)  # bound = first piece's prompt length - 1
    _check_ownership(c)
    e = c.entries["a"]
    assert sorted(e.states) == [16] and e.kv_len == 16 and torch.equal(e.tokens, row[:16])
    assert len(e.pages) == 2 and e.origin == "v0t"
    assert len(c.pools.free_slots) == free_s + 3 and len(c.pools.free_pages) == free_p + 3
    assert c.start("a", rej, 24) == start_untrimmed  # memory only: same start
    _fwd(c, ["a"], [rej], [16])  # reads [0, 16) from the trimmed pages (checked in _fwd)


def test_trim_is_memory_only_for_any_decision():
    """For every rejection position (and whole acceptance) the trimmed entry gives the
    first piece the same start as the untrimmed capture, and that forward reads its prefix."""
    gen = torch.Generator().manual_seed(0)
    for chunk in (4, 8):
        for plen in (3, 10, 17):
            g = 37
            row = torch.randint(5, 1000, (plen + g,), generator=gen)
            for tau in range(0, g + 1, 3):
                full, trimmed = _cache(chunk=chunk), _cache(chunk=chunk)
                for c in (full, trimmed):
                    _fwd(c, ["a"], [row], [0], floors=[plen])
                if tau == g:  # whole accepted, cut: first piece = row + piece
                    nxt = torch.cat([row, torch.arange(2000, 2005)])
                    p1 = plen + g
                else:  # rejected at tau: prefix draft[:tau] + y
                    nxt = torch.cat([row[: plen + tau], torch.tensor([1]), torch.arange(2000, 2005)])
                    p1 = plen + tau + 1
                trimmed.trim("a", p1 - 1)
                _check_ownership(trimmed)
                s_full = full.start("a", nxt, p1)
                s_trim = trimmed.start("a", nxt, p1)
                assert s_full == s_trim, (chunk, plen, tau, s_full, s_trim)
                if (plen + g) // chunk - plen // chunk < 4:
                    assert p1 - 1 - s_trim < chunk
                _fwd(trimmed, ["a"], [nxt], [s_trim])


def test_trim_leaves_piece_entries_alone_and_drops_unusable_captures():
    c = _cache(chunk=4)
    row = torch.arange(100, 121)
    _fwd(c, ["p"], [row], [0])  # a piece forward's entry
    states = sorted(c.entries["p"].states)
    c.trim("p", 5)
    assert sorted(c.entries["p"].states) == states  # trims apply to verify0 captures only
    _fwd(c, ["v"], [row], [0], floors=[10])
    c.trim("v", 7)  # nothing <= 7 kept from the prompt boundary 8 on
    assert "v" not in c.entries
    _check_ownership(c)
    c.trim("missing", 3)  # no-op


def test_first_piece_after_capture_replaces_the_entry():
    c = _cache(chunk=4)
    row = torch.arange(100, 130)  # plen 10, n 30
    _fwd(c, ["a"], [row], [0], floors=[10])
    c.trim("a", 29)
    assert sorted(c.entries["a"].states) == [28]
    nxt = torch.cat([row, torch.arange(300, 307)])  # cut row's first piece, n 37
    s = c.start("a", nxt, 30)
    assert s == 28
    _fwd(c, ["a"], [nxt], [s])
    e = c.entries["a"]
    assert e.origin == "piece" and sorted(e.states) == [28, 32, 36]
    c.trim("a", 29)  # a late trim must not cut the piece's newer boundaries
    assert sorted(c.entries["a"].states) == [28, 32, 36]


def test_negative_control_scales_only_captured_states():
    from nemo_rl.models.megatron.prefix_state_cache import TRASH_SLOT

    c = _cache(chunk=4)
    c.negctl = 2.0**-6
    row = torch.arange(100, 130)
    with torch.no_grad(), c.forward(["a"], [row], [0], None, [10]):
        from nemo_rl.models.megatron.prefix_state_cache import _ACTIVE, kv_write_index

        meta, p = _ACTIVE[0].meta, c.pools
        valid, pg, off = kv_write_index(meta, 30, p.page, p.maxp)
        p.k[1].index_put_((pg.reshape(-1), off.reshape(-1)), row.float().reshape(-1, 1, 1))
        for j in range(p.K):
            slot = int(meta["keep_slot"][0, j])
            if slot != TRASH_SLOT:
                p.ssm[2][slot] = float(meta["keep_off"][0, j])
    e = c.entries["a"]
    s = max(e.states)
    f = 1 + 2.0**-6
    assert float(c.pools.ssm[2][e.states[s]].item()) == float(s) * f
    assert float(c.pools.k[1][e.pages[0], 3, 0, 0]) == float(row[3]) * f
    c.negctl = 0.0
    c.release(["a"])
    _fwd(c, ["b"], [row], [0])  # piece entries untouched
    assert float(c.pools.k[1][c.entries["b"].pages[0], 3, 0, 0]) == float(row[3])


# ---- driver: admission, settle, trims ----------------------------------------------------
def _directory(max_gb=1.0, kv=1024, state=2**20, chunk=128, **kw):
    d = PrefixCacheDirectory(gpu_gb=0.5, max_gb=max_gb, **kw)
    d.configure([{"kv_bytes_per_token": kv, "state_bytes_per_boundary": state, "chunk": chunk, "max_keep": 4}])
    return d


def test_capture_places_by_lpt_and_admits_against_the_hard_cap():
    d = _directory(max_gb=0.02, row_cap_tokens=2048)  # ~20 MiB per rank; a row reserves ~7 MiB
    d.plan_step(1, 2)  # auto budget of an earlier step (small) must not bound captures
    lengths, plens = [2000, 1500, 1800, 900, 1200, 1000, 1100, 1300], [100] * 8
    keys = [d.new_key() for _ in lengths]
    owners, sent = d.capture(keys, lengths, plens, ranks=2)
    assert owners == lpt_owners(lengths, 2)
    per_rank = d.row_bytes(100)
    fits = int(d.cap_bytes // per_rank)
    assert sum(k is not None for k in sent) == 2 * fits
    c = d.take_counts()
    assert c["v0_captured"] == 2 * fits and c["v0_denied"] == 8 - 2 * fits
    assert all(d.owner[k] == r for k, r in zip(keys, owners) if k in d.owner)


def test_settle_keeps_continuing_rows_releases_the_rest_and_pieces_stay_on_the_owner():
    d = _directory(max_gb=1.0, row_cap_tokens=4096)
    keys = [d.new_key() for _ in range(4)]
    owners, sent = d.capture(keys, [3000, 2000, 1000, 500], [100] * 4, ranks=2)
    assert sent == keys
    d.settle(keys[0], 3099)  # cut row: continues
    d.settle(keys[1], 150)  # rejected: continues from position 150
    d.settle(keys[2], None)  # finished rollout
    d.settle(keys[3], None)  # a ⊥ / unused pool row
    d.settle(keys[3], 7)  # settled twice: ignored
    assert d.take_trim() == [(keys[0], 3099), (keys[1], 150)] and d.take_trim() == []
    assert sorted(d.take_release()) == sorted(keys[2:])
    c = d.take_counts()
    assert (c["v0_kept"], c["v0_released"]) == (2, 2)
    # first pieces of the kept rows: sticky owner, counted as eligible re-verified rows
    own2, sent2 = d.place([keys[1], keys[0]], [281, 3228], [151, 3100], ranks=2)
    assert own2 == [owners[1], owners[0]] and sent2 == [keys[1], keys[0]]
    assert d.take_counts()["eligible"] == 2
    assert d.forwarded(keys[0], 3228, 3100) == 3228 - 3072


def test_keys_are_unique_across_directories():
    a, b = _directory(), _directory()
    ka = {a.new_key() for _ in range(5)}
    kb = {b.new_key() for _ in range(5)}
    assert not ka & kb


def test_reset_forgets_captures():
    d = _directory()
    k = d.new_key()
    d.capture([k], [1000], [100], ranks=1)
    d.reset()
    d.settle(k, 500)
    assert d.take_trim() == [] and not d.captured


# ---- driver: verify0 call -> plans -> settle -> first piece --------------------------------
class _Pool:
    """A learner pool of ``dp`` data-parallel ranks that records its calls."""

    def __init__(self, dp):
        from nemo_rl.distributed.named_sharding import NamedSharding

        layout = np.arange(dp).reshape(1, dp, 1, 1)
        names = ["pipeline_parallel", "data_parallel", "context_parallel", "tensor_parallel"]
        self.sharding_annotations = NamedSharding(layout, names)
        self.dp_size = dp
        self.workers = [None] * dp
        self.calls = []

    def run_all_workers_single_data(self, method, **kw):
        self.calls.append((method, kw))
        return []


def _spec(dp=2, verify0=True):
    from thundersync_rl.speculative import SpeculativeGeneration

    spec = object.__new__(SpeculativeGeneration)
    spec.base = SimpleNamespace()
    spec.max_new_tokens, spec.max_seq_len, spec.draft_budget = 64, None, 64
    spec.sticky_delta, spec.eod = None, 0
    spec.verify_source, spec.verify_on, spec.hidden_transport = "engine", "learner", "object_store"
    spec.verify_batch_tokens, spec.vocab_limit = 16384, 1000
    spec.verify_precision, spec.verify_lazy = None, True
    spec.verifiers = [_Pool(dp)]
    spec.current_step = 3
    spec.stats = {}
    spec.turn_env, spec.anchor_horizon = None, None
    spec.learner_ordered = False
    spec._verify_seeds = SimpleNamespace(keyed=lambda *a: 7)
    d = PrefixCacheDirectory(gpu_gb=0.5, max_gb=4.0, row_cap_tokens=512)
    d.configure([{"kv_bytes_per_token": 6144, "state_bytes_per_boundary": 2**20, "chunk": 128, "max_keep": 4}])
    spec.prefix_cache = d
    spec._prefix_step = 3  # this step already begun (no worker cost query)
    spec.prefix_cache_verify0 = verify0
    spec._v0_cap, spec._v0_plan, spec._v0_step_keys = {}, {}, []
    return spec


def _draft(plen, g, finished, key):
    prompt = torch.arange(1, plen + 1)
    return (prompt, -1, list(range(10, 10 + g)), finished, key, [(2, 0, g)], [-0.5] * g, None)


def test_verify0_capture_settle_and_first_piece_placement():
    spec = _spec(dp=2)
    drafts = [_draft(100, 40, False, "a"), _draft(90, 30, False, "b"), _draft(80, 20, True, "c"),
              _draft(70, 10, False, "d")]
    flat = list(enumerate(drafts))
    rows = [torch.cat([d[0], torch.tensor(d[2])]) for d in drafts]
    spec._submit_verify_block(flat, rows, capture=True)
    method, kw = spec.verifiers[0].calls[-1]
    assert method == "verify_drafts_block_engine" and kw["prefix_capture"] is True
    keys = kw["prefix_keys"]
    # c finished (a stopped draft cannot continue): not captured, forwarded uncached
    assert keys[2] is None and all(k is not None for k in keys[:2] + keys[3:])
    assert len(set(keys[:2] + keys[3:])) == 3
    assert kw["row_ranks"] == lpt_owners([r.numel() for r in rows], 2)
    owners = dict(zip(keys, kw["row_ranks"]))
    # decisions: a cut (whole accepted, unfinished), b rejected at 5, c finished, d: no plan (⊥)
    lp = [-0.1] * 40
    plan_a = spec._plan(drafts[0], {"accepted": 40, "next": 55, "logprobs": lp})
    plan_b = spec._plan(drafts[1], {"accepted": 5, "next": 77, "logprobs": lp[:6]})
    plan_c = spec._plan(drafts[2], {"accepted": 20, "next": 0, "logprobs": lp[:21]})
    assert not plan_a.complete and not plan_b.complete and plan_c.complete
    assert plan_a.prefix_key == keys[0] and plan_b.prefix_key == keys[1]
    spec._v0_settle()
    assert plan_c.prefix_key is None and not spec._v0_cap and not spec._v0_plan
    d = spec.prefix_cache
    # first pieces: the next call carries the trims / releases and keeps the owners
    pa = torch.cat([rows[0], torch.tensor([55]), torch.arange(500, 520)])
    pb = torch.cat([rows[1][: 90 + 5], torch.tensor([77]), torch.arange(600, 620)])
    pflat = [(0, (pa[:141], -1, list(range(500, 520)), False, "a1", [(2, 0, 20)], [-0.5] * 20, None)),
             (1, (pb[:96], -1, list(range(600, 620)), False, "b1", [(2, 0, 20)], [-0.5] * 20, None))]
    spec._submit_verify_block(pflat, [pa, pb], prefix_keys=spec._prefix_keys([plan_a, plan_b]))
    method, kw = spec.verifiers[0].calls[-1]
    assert kw["prefix_keys"] == [keys[0], keys[1]] and "prefix_capture" not in kw
    assert kw["row_ranks"] == [owners[keys[0]], owners[keys[1]]]
    assert sorted(kw["prefix_trim"]) == sorted([(keys[0], 100 + 41 - 1), (keys[1], 90 + 6 - 1)])
    assert sorted(kw["prefix_release"]) == [keys[3]]
    c = d.take_counts()
    assert (c["v0_captured"], c["v0_kept"], c["v0_released"], c["eligible"]) == (3, 2, 1, 2)


def test_a_call_of_stopped_drafts_only_keeps_the_plain_verify0_forward():
    """Full-draft cohorts (every draft stopped: EOS or the cap) pay nothing for the capture."""
    spec = _spec(dp=2)
    drafts = [_draft(100, 40, True, "a"), _draft(90, 30, True, "b"), _draft(80, 64, False, "c")]
    flat = list(enumerate(drafts))
    rows = [torch.cat([d[0], torch.tensor(d[2])]) for d in drafts]
    spec._submit_verify_block(flat, rows, capture=True)
    method, kw = spec.verifiers[0].calls[-1]
    assert method == "verify_drafts_block_engine"
    assert "prefix_keys" not in kw and "prefix_capture" not in kw
    assert kw["row_ranks"] == lpt_owners([r.numel() for r in rows], 2)
    assert not spec._v0_cap and spec.prefix_cache.take_counts().get("v0_captured", 0) == 0


def test_directory_capture_skips_rows_without_a_key():
    d = _directory(max_gb=1.0, row_cap_tokens=4096)
    keys = [d.new_key(), None, d.new_key()]
    owners, sent = d.capture(keys, [3000, 2000, 1000], [100] * 3, ranks=2)
    assert owners == lpt_owners([3000, 2000, 1000], 2)
    assert sent == [keys[0], None, keys[2]]
    c = d.take_counts()
    assert c["v0_captured"] == 2 and c.get("v0_denied", 0) == 0


def test_capture_off_sends_no_keys_and_vit_calls_never_capture():
    spec = _spec(dp=2, verify0=False)
    drafts = [_draft(100, 40, False, "a"), _draft(90, 30, False, "b")]
    flat = list(enumerate(drafts))
    rows = [torch.cat([d[0], torch.tensor(d[2])]) for d in drafts]
    spec._submit_verify_block(flat, rows, capture=True)
    _, kw = spec.verifiers[0].calls[-1]
    assert "prefix_keys" not in kw and kw["row_ranks"] == lpt_owners([r.numel() for r in rows], 2)
    spec = _spec(dp=2, verify0=True)
    spec._submit_verify_block(flat, rows, capture=True, row_seed_base=11)
    _, kw = spec.verifiers[0].calls[-1]
    assert "prefix_keys" not in kw
    spec.turn_env = object()  # multi-turn: off
    assert not spec._v0_capture_on()


def test_validation_of_the_flag():
    from thundersync_rl.grpo_loop import validate_learner_prefix_cache

    def cfg(**ts):
        base = dict(
            learner_prefix_cache=False, learner_prefix_cache_verify0=False, sticky_routing=True,
            verify_source="engine", verify_on="learner", sticky_rounding=False,
            learner_prefix_cache_max_gb=32.0, learner_prefix_cache_gpu_gb=8.0,
            learner_prefix_cache_size="auto", learner_prefix_cache_policy="pin",
            sticky_anchor_horizon=None,
        )
        base.update(ts)
        return SimpleNamespace(
            thundersync=SimpleNamespace(**base),
            policy={"megatron_cfg": {"batch_invariant_mode": True}},
            grpo=SimpleNamespace(max_rollout_turns=1),
        )

    validate_learner_prefix_cache(cfg(learner_prefix_cache=True, learner_prefix_cache_verify0=True))
    with pytest.raises(ValueError, match="needs thundersync.learner_prefix_cache"):
        validate_learner_prefix_cache(cfg(learner_prefix_cache_verify0=True))
    with pytest.raises(ValueError, match="anchor_horizon"):
        validate_learner_prefix_cache(
            cfg(learner_prefix_cache=True, learner_prefix_cache_verify0=True, sticky_anchor_horizon=64)
        )
