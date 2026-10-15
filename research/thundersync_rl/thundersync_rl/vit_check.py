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
"""Debug check for verify-in-train (VIT): ``THUNDERSYNC_VIT_CHECK=1`` (learner worker).

Verify-in-train runs block verification *as* the training forward of finished
drafts: the decisions are read from the training forward's log-probs, rejected
rows are masked before the backward. This check measures, on real canonical
training calls of a running ThunderSync step, everything that design's
exactness rests on. Per checked call (final-advantage chunks only), with the
step's gradient buffers snapshotted before and restored after:

  R0  regular training call                                   -> G0
  R1  same call, ClippedPGLossFn patched to read the training
      forward's sampled-token log-probs (read-only)           -> G1
      * ``ctl``: G1 == G0 bitwise (determinism control; also shows that reading
        the decisions' inputs from the training graph does not perturb it)
      * ``p``: training-forward log-prob == the row's generation_logprobs on every
        valid token. For accepted draft tokens those are the verifier's p
        (speculative.py ``token_logprobs``); for engine-decoded tokens the BI
        engine's. Equal p => the same block-verification decisions (they are a
        deterministic function of (p, q, u)).
  M   row j (first live row of the rank) masked like a padding row
      (sample_mask = token_mask = advantages = 0)              -> GM
  M'  as M, row j's tokens replaced (other content, same length) -> GM'
      * ``mask``: GM == GM' bitwise: a masked row contributes exactly zero and
        nothing of its content reaches the gradient (MoE aux loss, routing,
        normalization would show up here).
  A   row j removed from its micro-batch (only when that slice is a single
      micro-batch; else as M, keeping expert-parallel forward counts equal) -> GA
      * ``absent``: GA vs GM (information: a smaller micro-batch changes wgrad
        reduction shapes; not required by the VIT design, which never re-forms
        a micro-batch)
  P   provisional advantages a_hat = 0.7 a, then a correction call with (a - a_hat)
      on the same rows                                         -> GP
      * ``corr``: GP vs G0 (the "linear correction" is exact in exact arithmetic;
        this measures its fp difference)
  WZ  every micro-batch's loss x 0 (verify-in-train's masking of a micro-batch
      with a rejection); WM  every row of every item masked as padding:
      * ``wz`` / ``wm`` vs the gradient before the call: must be 0 (a whole
        masked micro-batch contributes exact zeros; the production condition)
  final: restore the snapshot and rerun R0 (the regular result is what trains;
      ``ctl2``: equal to G0).

Every rank runs the same passes with the same micro-batch counts (expert-parallel
lockstep). Output: one ``[vit check]`` line per rank and call; summarize with
``research/verify_in_train/parse_vit_check.py``.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Callable, Optional

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

STEP_COUNTERS = (
    "num_chunks",
    "local_valid_seqs",
    "local_valid_toks",
    "trained_valid_seqs",
    "trained_valid_toks",
    "total_num_microbatches",
    "num_pipeline_calls",
)


def copy_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Training items with their data copied (tensors cloned, lists copied)."""
    out = []
    for it in items:
        data = BatchedDataDict()
        for k, v in it["data"].items():
            data[k] = v.clone() if isinstance(v, torch.Tensor) else (list(v) if isinstance(v, list) else v)
        out.append({**it, "data": data})
    return out


def param_grads(worker) -> list[tuple[str, torch.Tensor]]:
    return [
        (n, p.main_grad)
        for n, p in worker.model.named_parameters()
        if getattr(p, "main_grad", None) is not None
    ]


def snapshot(worker) -> dict[str, Any]:
    state = worker._train_step_state
    bufs = list(worker.model.buffers) + list(worker.model.expert_parallel_buffers)
    return {
        "grads": [(b.grad_data, b.grad_data.clone()) for b in bufs]
        + [(g, g.clone()) for _, g in param_grads(worker)],
        "counters": {
            k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k in STEP_COUNTERS
            if (v := state.get(k)) is not None
        },
        "lists": {k: len(state[k]) for k in ("all_mb_metrics", "mb_losses") if k in state},
    }


def restore(worker, snap: dict[str, Any]) -> None:
    state = worker._train_step_state
    for live, saved in snap["grads"]:
        live.copy_(saved)
    for p in worker.model.parameters():
        p.grad = None
    state.update(snap["counters"])
    for k, n in snap["lists"].items():
        del state[k][n:]


def grads_to_host(worker) -> dict[str, torch.Tensor]:
    return {n: g.detach().to("cpu", copy=True) for n, g in param_grads(worker)}


def compare(worker, ref: dict[str, torch.Tensor]) -> tuple[int, float, float, int, int]:
    """(differing elements, max |d|, max |d| / max |ref| over params, params differing,
    differing elements in MoE expert weights)."""
    bad, worst_abs, worst_rel, nparams, bad_exp = 0, 0.0, 0.0, 0, 0
    for n, g in param_grads(worker):
        r = ref[n].to(g.device, non_blocking=True)
        if torch.equal(r, g):
            continue
        ne = int((r != g).sum())
        if ne == 0:  # NaN == NaN never; equal otherwise
            continue
        bad += ne
        nparams += 1
        if ".experts." in n:
            bad_exp += ne
        d = float((r.float() - g.float()).abs().max())
        m = float(r.float().abs().max())
        worst_abs = max(worst_abs, d)
        worst_rel = max(worst_rel, d / m if m > 0 else float("inf"))
    return bad, worst_abs, worst_rel, nparams, bad_exp


def _live(data) -> torch.Tensor:
    tm = data["token_mask"][:, 1:] * data["sample_mask"].unsqueeze(-1)
    return (data["advantages"][:, 1:] * tm).ne(0).any(dim=-1)


def pick_row(items) -> Optional[tuple[int, int]]:
    """(item, row) of the first row with a nonzero advantage on a valid token."""
    for i, it in enumerate(items):
        live = _live(it["data"])
        if bool(live.any()):
            return i, int(torch.nonzero(live)[0])
    return None


def mask_row(items, pos, replace: bool) -> None:
    if pos is None:
        return
    i, j = pos
    d = items[i]["data"]
    for k in ("sample_mask", "token_mask", "advantages"):
        d[k][j] = 0
    if replace:
        ids = d["input_ids"][j]
        n = int(d["input_lengths"][j]) if "input_lengths" in d else ids.numel()
        # Other content, same length and vocabulary range: a fixed permutation of the row's tokens.
        ids[:n] = torch.roll(ids[:n], shifts=max(1, n // 3)).clone()


def drop_row(items, pos, mbs_per_item) -> bool:
    """Remove row j when its slice runs as one micro-batch (forward count unchanged)."""
    if pos is None:
        return False
    i, j = pos
    d = items[i]["data"]
    if d.size <= 1 or d.size > mbs_per_item[i]:
        return False
    keep = [r for r in range(d.size) if r != j]
    items[i]["data"] = d.select_indices(keep)
    mbs_per_item[i] = len(keep)
    return True


def scale_adv(items, f: Callable[[torch.Tensor], torch.Tensor]) -> None:
    for it in items:
        it["data"]["advantages"] = f(it["data"]["advantages"].clone())


def mask_all(items) -> None:
    """Every row of every item masked like a padding row (whole micro-batches)."""
    for it in items:
        d = it["data"]
        for k in ("sample_mask", "token_mask", "advantages"):
            d[k] = torch.zeros_like(d[k])


@contextmanager
def zero_loss():
    """Every micro-batch's loss multiplied by 0 (what verify-in-train does to a
    micro-batch with a rejection: the backward runs, contributing exact zeros)."""
    from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossFn

    orig = ClippedPGLossFn.__call__

    def patched(self, *args, **kwargs):
        loss, metrics = orig(self, *args, **kwargs)
        return loss * 0.0, metrics

    ClippedPGLossFn.__call__ = patched
    try:
        yield
    finally:
        ClippedPGLossFn.__call__ = orig


@contextmanager
def capture_logprobs(stats: dict[str, float]):
    """Patch ClippedPGLossFn.__call__ to compare the training forward's p with generation_logprobs."""
    from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossFn

    orig = ClippedPGLossFn.__call__

    def patched(self, next_token_logprobs=None, data=None, *args, **kwargs):
        if next_token_logprobs is not None and data is not None:
            with torch.no_grad():
                m = (data["token_mask"][:, 1:] * data["sample_mask"].unsqueeze(-1)).bool()
                lp = next_token_logprobs.detach()
                gen = data["generation_logprobs"][:, 1:].to(lp.device)
                if lp.shape == gen.shape:
                    d = (lp.float() - gen.float())[m.to(lp.device)]
                    stats["p_tokens"] += float(d.numel())
                    stats["p_diff_tokens"] += float((d != 0).sum())
                    if d.numel():
                        stats["p_max_abs"] = max(stats["p_max_abs"], float(d.abs().max()))
                else:
                    stats["p_shape_skips"] += 1
        return orig(self, next_token_logprobs, data, *args, **kwargs)

    ClippedPGLossFn.__call__ = patched
    try:
        yield
    finally:
        ClippedPGLossFn.__call__ = orig


def run_check(worker, call_items, mbs_per_item, bucket, tprof, call_index: int) -> None:
    """Replaces one ``_stream_train_call`` (final chunks); leaves the regular result in place."""
    pristine = copy_items(call_items)
    snap = snapshot(worker)
    g_pre = grads_to_host(worker)
    train = lambda items, mbs: worker._stream_train_call(items, list(mbs), bucket, tprof)
    out: dict[str, Any] = {}

    # R0
    train(copy_items(pristine), mbs_per_item)
    g0 = grads_to_host(worker)
    # R1: read p from the training forward
    restore(worker, snap)
    pst = dict(p_tokens=0.0, p_diff_tokens=0.0, p_max_abs=0.0, p_shape_skips=0.0)
    with capture_logprobs(pst):
        train(copy_items(pristine), mbs_per_item)
    out["ctl"] = compare(worker, g0)
    out.update(pst)
    pos = pick_row(pristine)
    out["row"] = pos
    # M, M'
    restore(worker, snap)
    items = copy_items(pristine)
    mask_row(items, pos, replace=False)
    train(items, mbs_per_item)
    gm = grads_to_host(worker)
    restore(worker, snap)
    items = copy_items(pristine)
    mask_row(items, pos, replace=True)
    train(items, mbs_per_item)
    out["mask"] = compare(worker, gm)
    # A (row absent) vs M
    restore(worker, snap)
    items = copy_items(pristine)
    mbs = list(mbs_per_item)
    dropped = drop_row(items, pos, mbs)
    if not dropped:
        mask_row(items, pos, replace=False)
    train(items, mbs)
    out["absent"] = compare(worker, gm) if dropped else None
    del gm
    # P: provisional 0.7 a, then correction (a - 0.7 a), vs G0
    restore(worker, snap)
    items = copy_items(pristine)
    scale_adv(items, lambda a: a * 0.7)
    train(items, mbs_per_item)
    items = copy_items(pristine)
    scale_adv(items, lambda a: a - a * 0.7)
    train(items, mbs_per_item)
    out["corr"] = compare(worker, g0)
    # WZ / WM: whole micro-batches masked (the production condition): the call
    # must leave the accumulated gradient bitwise unchanged.
    restore(worker, snap)
    with zero_loss():
        train(copy_items(pristine), mbs_per_item)
    out["wz"] = compare(worker, g_pre)
    restore(worker, snap)
    items = copy_items(pristine)
    mask_all(items)
    train(items, mbs_per_item)
    out["wm"] = compare(worker, g_pre)
    del g_pre
    # final: the regular result
    restore(worker, snap)
    train(copy_items(pristine), mbs_per_item)
    out["ctl2"] = compare(worker, g0)
    del g0, snap
    print(
        f"[vit check] rank={worker.rank} call={call_index} row={out['row']} "
        f"p_tokens={int(out['p_tokens'])} p_diff_tokens={int(out['p_diff_tokens'])} "
        f"p_max_abs={out['p_max_abs']:.3e} p_shape_skips={int(out['p_shape_skips'])} "
        + " ".join(
            f"{k}=" + ("None" if out[k] is None else f"{out[k][0]}/{out[k][1]:.3e}/{out[k][2]:.3e}/{out[k][3]}/{out[k][4]}")
            for k in ("ctl", "ctl2", "mask", "absent", "corr", "wz", "wm")
        ),
        flush=True,
    )


def enabled() -> bool:
    return os.environ.get("THUNDERSYNC_VIT_CHECK", "0") == "1"


def max_calls() -> int:
    return int(os.environ.get("THUNDERSYNC_VIT_CHECK_CALLS", "2"))
