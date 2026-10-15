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
"""Run control (run_state.py): seeds, epochs, resumable data position, boundaries, resume."""

import json
from types import SimpleNamespace

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from thundersync_rl.grpo_loop import validate_run_control
from thundersync_rl.run_state import (
    STATE_FILE,
    Boundaries,
    PromptStream,
    SeedStream,
    check_fingerprint,
    config_fingerprint,
    derive_seed,
    latest_checkpoint,
    load_state,
    resolve_resume,
    save_state,
)

# ---------------------------------------------------------------------------
# Verification seeds (issue 5)
# ---------------------------------------------------------------------------


def test_seeds_depend_on_grpo_seed_stream_and_step_and_never_repeat():
    step = [0]
    a = SeedStream(42, "verify/0", lambda: step[0])
    b = SeedStream(43, "verify/0", lambda: step[0])  # another grpo.seed
    c = SeedStream(42, "verify/1", lambda: step[0])  # another lane
    seen = set()
    for s in range(5):
        step[0] = s
        for _ in range(20):
            for x in (next(a), next(b), next(c)):
                assert 1 <= x < 2**40  # seed * 1_000_003 + rank fits manual_seed
                assert x not in seen
                seen.add(x)


def test_seed_counter_resumes_exactly_and_refuses_other_seed():
    step = [3]
    a = SeedStream(7, "verify/0", lambda: step[0])
    [next(a) for _ in range(5)]
    state = a.state_dict()
    want = [next(a) for _ in range(4)]
    b = SeedStream(7, "verify/0", lambda: step[0])
    b.load_state_dict(state)
    assert [next(b) for _ in range(4)] == want
    with pytest.raises(ValueError, match="grpo.seed"):
        SeedStream(8, "verify/0", lambda: 0).load_state_dict(state)
    assert derive_seed(1, "x", 2, 3) == derive_seed(1, "x", 2, 3)  # stable


def test_speculative_wrapper_uses_derived_seed_streams(monkeypatch):
    import thundersync_rl.speculative as spec_mod
    from test_speculative_driver import _spec

    monkeypatch.setattr(spec_mod.ray, "get", lambda x: x)

    s = _spec({}, seed=11, seed_stream="1")
    s.current_step = 4
    first = next(s._verify_seeds)
    assert first == derive_seed(11, "verify/1", 4, 1)
    st = s.seed_state()
    t = _spec({}, seed=11, seed_stream="1")
    t.current_step = 4
    t.load_seed_state(st)
    assert next(t._verify_seeds) == next(s._verify_seeds)


def _call_seed_spec(step=3):
    from thundersync_rl.speculative import SpeculativeGeneration

    sp = object.__new__(SpeculativeGeneration)
    sp.__dict__.update(
        current_step=step,
        _verify_seeds=SeedStream(11, "verify/0", lambda: sp.current_step),
    )
    return sp


def _d(key, toks=(1, 2)):
    return (torch.tensor([5, 6, 7]), -1, list(toks), False, key, [], None, None)


def test_call_seed_is_independent_of_arrival_order():
    """Block and piece verification calls get the same seeds whichever submits first."""
    block = [_d("1:0"), _d("1:1")]
    piece = [_d("c:0")]
    a = _call_seed_spec()
    sa = (a._call_seed("block", block), a._call_seed("block", piece))
    b = _call_seed_spec()
    sb_piece = b._call_seed("block", piece)
    sb_block = b._call_seed("block", block)
    assert sa == (sb_block, sb_piece)


def test_call_seed_is_fresh_per_row_ordinal_and_step():
    """A row's next call (next piece) and the next step never reuse a seed."""
    sp = _call_seed_spec(step=3)
    first = sp._call_seed("block", [_d("c:0")])
    second = sp._call_seed("block", [_d("c:0")])
    assert first != second
    sp.current_step = 4
    assert sp._call_seed("block", [_d("c:0")]) not in (first, second)
    # Unkeyed rows are named by their content.
    u = _call_seed_spec()
    assert u._call_seed("block", [_d(None, (1, 2))]) != u._call_seed("block", [_d(None, (1, 3))])


# ---------------------------------------------------------------------------
# Data: epochs and an exact resume position (issues 1, 3)
# ---------------------------------------------------------------------------


class _DS(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def _collate(xs):
    return BatchedDataDict({"idx": list(xs), "message_log": [[{"token_ids": torch.tensor([x])}] for x in xs]})


def _loader(n=10, bs=3, workers=0):
    torch.manual_seed(0)
    return StatefulDataLoader(
        _DS(n), batch_size=bs, shuffle=True, drop_last=True, collate_fn=_collate, num_workers=workers
    )


@pytest.mark.parametrize("rebatch", [None, 4])
def test_prompt_stream_epochs_and_resume_from_every_position(rebatch):
    full = PromptStream(_loader(), max_epochs=3, rebatch_size=rebatch)
    tagged = list(full)
    seq = [t.batch["idx"] for t in tagged]
    per_epoch = 3  # 10 // 3 with drop_last
    if rebatch is None:
        assert len(seq) == 3 * per_epoch
        # Each epoch is a permutation slice of the dataset; epochs reshuffle.
        assert seq[:3] != seq[3:6]
    else:
        assert all(len(b) == rebatch for b in seq) and len(seq) == (3 * 9) // rebatch
    for k, t in enumerate(tagged):
        resumed = PromptStream(_loader(), max_epochs=3, rebatch_size=rebatch, state=t.state)
        assert [x.batch["idx"] for x in resumed] == seq[k:], k


def test_prompt_stream_state_after_last_batch_is_exhausted():
    s = PromptStream(_loader(), max_epochs=1)
    list(s)
    assert s.exhausted
    assert list(PromptStream(_loader(), max_epochs=1, state=s.state())) == []


# ---------------------------------------------------------------------------
# Boundaries: no draft crosses a checkpoint / validation step
# ---------------------------------------------------------------------------


def test_boundaries_and_draft_limit():
    b = Boundaries(max_steps=10, save_period=4, val_period=3, val_start_at=6)
    assert [s for s in range(10) if b.saves_after(s)] == [3, 7, 9]  # last step saves too
    assert [s for s in range(10) if b.validates_after(s)] == [5, 8]
    for s in range(10):
        lim = b.draft_limit(s)
        # Nothing between s+1 and lim-1 is a boundary; lim is one (or max_steps).
        assert all(not b.is_boundary(t - 1) for t in range(s + 2, lim))
        assert lim == 10 or b.is_boundary(lim - 1)
        assert b.may_draft_next(s) == (not b.is_boundary(s) and s + 1 < 10)
    off = Boundaries(max_steps=5)
    assert [off.may_draft_next(s) for s in range(5)] == [True] * 4 + [False]
    off.requested.add(3)
    assert not off.may_draft_next(2) and off.may_draft_next(3)


# ---------------------------------------------------------------------------
# Explicit resume (issue 2)
# ---------------------------------------------------------------------------


def test_resolve_resume(tmp_path):
    cfg = {"checkpoint_dir": str(tmp_path / "ck"), "enabled": False}
    assert resolve_resume(cfg) is None  # nothing there
    (tmp_path / "ck" / "step_4").mkdir(parents=True)
    (tmp_path / "ck" / "tmp_step_9").mkdir()
    with pytest.raises(RuntimeError, match="checkpointing.enabled is false"):
        resolve_resume(cfg)  # NeMo-RL setup() would load it silently
    cfg["enabled"] = True
    with pytest.raises(RuntimeError, match="not written by the ThunderSync loop"):
        resolve_resume(cfg)
    (tmp_path / "ck" / "step_12").mkdir()
    save_state(tmp_path / "ck" / "step_12", {"next_step": 12})
    assert latest_checkpoint(tmp_path / "ck").name == "step_12"
    assert resolve_resume(cfg).name == "step_12"
    assert load_state(tmp_path / "ck" / "step_12")["next_step"] == 12
    assert (tmp_path / "ck" / "step_12" / STATE_FILE).exists()


def _master(**over):
    cfg = SimpleNamespace(
        grpo=SimpleNamespace(
            seed=42, num_prompts_per_step=4, num_generations_per_prompt=8, max_num_epochs=2,
            val_period=0, val_at_start=False, val_at_end=False,
        ),
        data={"shuffle": True},
        checkpointing={"enabled": True, "save_period": 2, "save_optimizer": True},
        thundersync=SimpleNamespace(
            speculate=True, draft_weights="momentum", cohort_lanes=1, null_bv_prompt_drop=False,
            null_bv_spare_prompts=0, rollout_lag=0, stream_training_over_tails=True,
        ),
    )
    for k, v in over.items():
        sec, key = k.split("__")
        part = getattr(cfg, sec)
        if isinstance(part, dict):
            part[key] = v
        else:
            setattr(part, key, v)
    return cfg


def test_fingerprint_refuses_changed_settings():
    a = config_fingerprint(_master())
    check_fingerprint(json.loads(json.dumps(a)), config_fingerprint(_master()))
    with pytest.raises(RuntimeError, match="grpo.seed"):
        check_fingerprint(a, config_fingerprint(_master(grpo__seed=1)))


def test_run_control_validation():
    validate_run_control(_master())
    for over, msg in (
        ({"checkpointing__save_optimizer": False}, "save_optimizer"),
        ({"thundersync__rollout_lag": 1}, "rollout_lag"),
        ({"checkpointing__save_period": 0}, "save_period"),
        ({"grpo__max_num_epochs": 0}, "max_num_epochs"),
        ({"thundersync__cohort_lanes": 2, "thundersync__stream_training_over_tails": False}, "non-streaming lanes"),
    ):
        with pytest.raises(ValueError, match=msg):
            validate_run_control(_master(**over))


def test_resume_refuses_a_changed_run_length(tmp_path):
    import yaml

    from thundersync_rl.run_state import check_schedule_unchanged

    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"grpo": {"max_num_steps": 4, "max_num_epochs": 1}}))
    check_schedule_unchanged(tmp_path, 4, 1)
    with pytest.raises(RuntimeError, match="same run length"):
        check_schedule_unchanged(tmp_path, 2, 1)  # the scheduler was sized for 4 steps
