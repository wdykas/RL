r"""Is a TP collective's result for a fixed token independent of the batch size?

Batch invariance at TP>1 (DESIGN.md section 14) needs the tensor-parallel
reductions of the row-parallel layers to give a token the same bits whatever
the total token count T is. Ops:
  rs        sequence-parallel path: reduce_scatter_tensor over T rows (each rank
            keeps T/tp rows), all-gathered back for the comparison;
  ar        non-SP path: all_reduce over T rows;
  ordered   candidate fix (all_to_all of the partial rows, then an fp32 sum in
            ascending rank order, one bf16 rounding): invariant by construction.

For every TP size and op, each rank holds fixed random bf16 partials for a pool
of tokens and reduces the first T of them for many T. A token's output must be
bitwise identical for every T (reference: the largest T). Outputs are also
compared with round_bf16(fp32 sum of the partials in rank order).

Expected: tp=2 invariant for rs/ar (a 2-term sum is one commutative IEEE add);
tp>=3 may depend on T for rs/ar (the ring/tree order of a token's partial sums
depends on which rank owns it, i.e. on T); ordered always invariant. If tp>=3
shows a dependence, the fix is an ordered reduce-scatter (DESIGN.md section 14,
follow-up).

Run (4 GPUs, a few minutes):
  /opt/nemo_rl_venv/bin/python -m torch.distributed.run --nproc-per-node 4 \
      nccl_tp_invariance.py [--tp 2 3 4] [--ops rs ar ordered]
Optionally repeat with NCCL_ALGO=Ring, NCCL_ALGO=NVLS or NCCL_PROTO=Simple.
Exit status 1 if an rs/ar case at tp=2, or any ordered case, depends on T.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.distributed as dist


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tp", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument(
        "--ops",
        nargs="+",
        default=["rs", "ar", "ordered"],
        choices=["rs", "ar", "ordered"],
    )
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--pool", type=int, default=8192, help="largest T")
    p.add_argument(
        "--step", type=int, default=64, help="T grid step (made a multiple of tp)"
    )
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args(argv)


def reduce_tokens(op: str, x: torch.Tensor, group, tp: int) -> torch.Tensor:
    """Reduced [T, H] rows of ``x`` (this rank's partials) over ``group``."""
    if op == "ar":
        y = x.clone()
        dist.all_reduce(y, group=group)
        return y
    t, h = x.shape
    if op == "rs":
        local = torch.empty(t // tp, h, dtype=x.dtype, device=x.device)
        dist.reduce_scatter_tensor(local, x.contiguous(), group=group)
    else:  # ordered
        recv = torch.empty_like(x)
        dist.all_to_all_single(recv, x.contiguous(), group=group)
        parts = recv.view(tp, t // tp, h).float()
        acc = parts[0].clone()
        for r in range(1, tp):  # ascending rank order, fp32
            acc += parts[r]
        local = acc.to(x.dtype)
    full = torch.empty_like(x)
    dist.all_gather_into_tensor(full, local, group=group)
    return full


def check_tp(args, tp: int, rank: int, group, dev) -> bool:
    """Run every op at this TP size; return True if an expected invariance failed."""
    gen = torch.Generator(device=dev).manual_seed(args.seed + rank)
    partial = torch.randn(args.pool, args.hidden, generator=gen, device=dev).to(
        torch.bfloat16
    )
    gathered = [torch.empty_like(partial) for _ in range(tp)]
    dist.all_gather(gathered, partial, group=group)
    exact = gathered[0].float()
    for r in range(1, tp):
        exact += gathered[r].float()
    exact = exact.to(torch.bfloat16)
    del gathered
    step = math.lcm(args.step, tp)
    sizes = list(range(step, args.pool + 1, step))
    failed = False
    for op in args.ops:
        ref = reduce_tokens(op, partial[: sizes[-1]], group, tp)
        n_bad_t, worst = 0, 0
        for t in sizes:
            out = reduce_tokens(op, partial[:t], group, tp)
            diff = int((out != ref[:t]).any(dim=1).sum())
            if diff:
                n_bad_t += 1
                worst = max(worst, diff)
        vs_exact = int((ref != exact[: sizes[-1]]).any(dim=1).sum())
        invariant = n_bad_t == 0
        if rank == 0:
            print(
                f"tp={tp} op={op:8s} T={sizes[0]}..{sizes[-1]} step {step}: "
                f"{'INVARIANT' if invariant else 'DEPENDS ON T'} "
                f"(T values with differing tokens {n_bad_t}/{len(sizes)}, "
                f"max differing tokens {worst}; tokens != round(fp32 rank-order "
                f"sum) {vs_exact}/{sizes[-1]})",
                flush=True,
            )
        if not invariant and (tp == 2 or op == "ordered"):
            failed = True
    return failed


def main(argv=None) -> int:
    args = parse_args(argv)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda")
    if rank == 0:
        print(
            f"world={world} NCCL_ALGO={os.environ.get('NCCL_ALGO')} "
            f"NCCL_PROTO={os.environ.get('NCCL_PROTO')} torch={torch.__version__} "
            f"nccl={torch.cuda.nccl.version()}",
            flush=True,
        )
    failed = False
    for tp in args.tp:
        if tp > world:
            if rank == 0:
                print(f"skip tp={tp}: world size {world}", flush=True)
            continue
        group = dist.new_group(list(range(tp)))  # collective: every rank calls it
        if rank < tp:
            failed |= check_tp(args, tp, rank, group, dev)
        dist.barrier()
    flag = torch.tensor([int(failed)], device=dev)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    dist.destroy_process_group()
    return int(flag.item())


if __name__ == "__main__":
    sys.exit(main())
