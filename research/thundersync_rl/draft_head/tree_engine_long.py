"""Long real-text rollouts with the MTP draft source: --prompt_keep (0 = the full held-out prompt) and per-request
finish steps ("fin": rid -> [first step, finish step, tokens]) for the slowest-row tokens/forward.

Engine-level exactness of tree speculation (tree_spec), adapted from the multi-token-decode (MTD) engine test.

  --arm tree : tree_spec installed (node rows, private KV blocks); requests 0..T-1 opt in with references:
                'true' (the plain continuation), 'half' (every 3rd token corrupted), 'mix' (a corrupted copy, then the
                true one: the trie branches where they differ), 'rand', 'none'. Every request's tokens and log-probs
                must equal the plain arm's bitwise; THUNDERSYNC_NL_NEG=tree_* negatives must NOT.

Original doc (MTD):

Two processes of the same engine (dl_bench's build: the in-loop BI recipe, CUDA graphs, async scheduling, the keyed
sampler S), the same request ids, prompts and noise keys:

  --arm plain : every request decodes one token per step; dump tokens + log-probs per request.
  --arm mtd   : tail_mtd installed; requests 0..T-1 opt in with references:
                  'true'   the plain arm's continuation of the same request (proposals mostly right),
                  'half'   the plain continuation with every 3rd token corrupted,
                  'rand'   random tokens (proposals never right),
                  'none'   no references (no proposals: the tail decodes one token per iteration);
                the other requests are ordinary one-token rows next to them.
                Tokens and log-probs of EVERY request must equal the plain arm's bitwise; MTD rows must have emitted
                more than one token per iteration in the 'true' arm.

  torchrun --nproc-per-node N tree_engine_long.py --model M --arm plain --out P ; ... --arm tree --mtp <head .pt> --ref P --out T
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # dl_bench
os.environ.setdefault("CUBLASLT_WORKSPACE_SIZE", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":0:0")
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

import dl_bench as B  # noqa: E402


def bench_args(a):
    return argparse.Namespace(
        model=a.model, tp=a.tp, ep=a.ep, graphs=1, scope="block", sticky=0, mcore_ckpt="", cf_m=a.cf_m, cf_extra=4,
        cf_sparse=1, layers=0, stock=0, epinv=0, min_bucket=0, dispatcher="nvls", buffer_gb=a.buffer_gb,
        num_graphs=16, max_tokens=a.max_tokens, max_len=a.max_len, prompt_lp=0, chunked=0, max_requests=a.max_requests,
        sched="async", nl=0, ts_hooks="", wide=1, tail_hold=0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["plain", "tree"], required=True)
    ap.add_argument("--prof", type=int, default=0, help="accumulate host time of the MTD hooks (last run)")
    ap.add_argument("--model", required=True, help="HF id or local dir of the policy (e.g. the Nano snapshot)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--cf_m", type=float, default=0.0)
    ap.add_argument("--buffer_gb", type=float, default=20)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--max_requests", type=int, default=128)
    ap.add_argument("--n", type=int, default=24, help="requests per rank")
    ap.add_argument("--tails", type=int, default=8, help="MTD requests per rank (first ones)")
    ap.add_argument("--ntok", type=int, default=700)
    ap.add_argument("--W", type=int, default=16)
    ap.add_argument("--D", type=int, default=4)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--ref", default="")
    ap.add_argument("--kinds", default="true,half,mix,rand,none", help="reference kinds cycled over the tails")
    ap.add_argument("--repeat", type=int, default=1, help="run the request set N times; time the last run only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mtp", default="", help="tree_mtp head (head_mcore.pt): the MTP draft source (TP 1)")
    ap.add_argument("--mtp_slots", type=int, default=64)
    ap.add_argument("--prompts", choices=["rand", "real"], default="rand",
                    help="real: XS-SD held-out prompts (text the MTP head was trained on), truncated to 400 tokens")
    ap.add_argument("--prompt_keep", type=int, default=400)
    ap.add_argument("--eos", type=int, default=-1, help="termination id (Nano: 11); -1 = none (forced length)")
    a = ap.parse_args()
    if a.arm == "tree":
        from nemo_rl.models.generation.megatron import tree_kv
        from nemo_rl.models.megatron import mamba_bi_lookahead

        mamba_bi_lookahead.install(a.rows, tree=True)  # before the context is built and the graphs captured
        tree_kv.install(a.rows)
    if a.mtp:
        # the decoder-output capture goes on the model before the engine / graphs are built (as megatron_worker does):
        # dl_bench imports install_engine_latency_levers from megatron_worker right after the model is built
        from nemo_rl.models.generation.megatron import megatron_worker as mw
        from nemo_rl.models.generation.megatron import tree_mtp

        _lev = mw.install_engine_latency_levers

        def _lev_and_capture(model_, *aa, **kk):
            tree_mtp.install_hidden_capture(model_, a.max_tokens, int(model_.config.hidden_size),
                                            feat_layers=tree_mtp.head_feature_layers(a.mtp))
            return _lev(model_, *aa, **kk)
        mw.install_engine_latency_levers = _lev_and_capture
    eng, model, prov, hooks = B.build(bench_args(a))
    rank = B.RANK
    from megatron.core.inference.sampling_params import SamplingParams

    mtd = None
    if a.arm == "tree":
        from nemo_rl.models.generation.megatron import tail_mtd, tree_spec

        st = WIDE_HOLDER["obj"]
        mtd = tree_spec.install(eng, W=a.W, D=a.D, t_max=64, ref_cap=16384, node_cap=a.rows,
                                own_history="own" in a.kinds.split(","),
                                sample_rows=st._nl_tree_sample, key_check=getattr(st, "_wide_key_check", None),
                                keyreg=None, key_ids=None, scores_rows=st._nl_tree_scores,
                                uniform_rows=st._nl_tree_uniform,
                                top2_rows=st._nl_tree_top2 if hasattr(st, "_nl_tree_top2") else None)
        if a.mtp:
            from megatron.core.utils import unwrap_model

            m = unwrap_model(model)
            c = m.config
            mtd.attach_mtp(tree_mtp.MTPHead(
                a.mtp, m.embedding.word_embeddings.weight, m.output_layer.weight, slots=a.mtp_slots,
                max_seq=int(eng.context.max_sequence_length), num_heads=c.num_attention_heads,
                num_kv_heads=c.num_query_groups or c.num_attention_heads, head_dim=c.kv_channels,
                eps=c.layernorm_epsilon))
    if mtd is not None:
        c_ = eng.context
        print(f"[tree test] kv buffer {tuple(c_.memory_buffer.shape)} numel {c_.memory_buffer.numel()} dummy "
              f"{int(c_.kv_block_allocator.dummy_block_idx)} private {mtd.priv.host[:3]}..{mtd.priv.host[-1:]} "
              f"max_requests {int(c_.max_requests)}", flush=True)
    prof = {}
    if mtd is not None and a.prof:
        for name in ("plan", "finish_layout", "expand", "after_forward", "after_sample", "resolve", "_commit_keys",
                     "tracked_rids"):  # tree_spec has the same hook names
            f0 = getattr(mtd, name)

            def timed(*aa, _f=f0, _n=name, **kk):
                t = time.perf_counter()
                try:
                    return _f(*aa, **kk)
                finally:
                    prof[_n] = prof.get(_n, 0.0) + time.perf_counter() - t
            setattr(mtd, name, timed)
    ref = torch.load(f"{a.ref}.rank{rank}.pt") if a.ref else {}
    # Same requests on every rank (TP ranks of one engine must see identical requests; EP ranks run in lockstep).
    kinds = a.kinds.split(",")
    real = []
    if a.prompts == "real":
        import pickle

        dd = os.environ.get("DRAFT_HEAD_DATA", "")
        if not dd:
            raise SystemExit("--prompts real needs DRAFT_HEAD_DATA (prep_data.py output dir with heldout.pkl)")
        ho = pickle.load(open(os.path.join(dd, "heldout.pkl"), "rb"))
        real = [list(map(int, ho["prompts"][k]))[-a.prompt_keep:] for k in sorted(ho["prompts"])[: a.n]]
    for rep_i in range(a.repeat):
        rng = random.Random(1234)
        rng_refs = random.Random(99)  # references only: the requests are the same in both arms
        done = {}
        fin = {}
        base = 7_000_000 + 1_000_000 * rep_i
        for i in range(a.n):
            rid = base + i
            plen = rng.randrange(60, 400)
            prompt = [rng.randrange(100, 30000) for _ in range(plen)]
            if real:
                prompt = real[i % len(real)]
                plen = len(prompt)
            ntok = a.ntok + rng.randrange(0, 200)
            sp = SamplingParams(temperature=1.0, top_k=0, num_tokens_to_generate=ntok, termination_id=a.eos,
                                return_log_probs=True, skip_prompt_log_probs=True, detokenize_generations=False)
            if mtd is not None and i < a.tails:
                kind = kinds[i % len(kinds)]
                cont = ref[7_000_000 + i][0].tolist() if (7_000_000 + i) in ref else []
                if kind == "true":
                    refs = [cont]
                elif kind == "half":
                    refs = [[t if j % 3 else (t + 7) % 30000 for j, t in enumerate(cont)]]
                elif kind == "mix":
                    refs = [[t if j % 5 else (t + 11) % 30000 for j, t in enumerate(cont)], cont]
                elif kind == "rand":
                    refs = [[rng_refs.randrange(100, 30000) for _ in range(max(len(cont), 512))]]
                else:
                    refs = []
                if kind == "own":  # src bit1 (XS-SD): candidates from the request's own history, no references
                    import numpy as _np

                    sp.add_attributes({tree_spec.TREE_ATTR: _np.asarray([a.W, a.D, 2], dtype=_np.int32).tobytes()})
                else:
                    sp.add_attributes({tail_mtd.MTD_REFS_ATTR: tail_mtd.pack_refs(refs).tobytes()})
            eng.add_request(rid, prompt, sp)
            DEBUG_PLEN[rid] = plen
        prof.clear()
        marks = {}
        t0 = time.perf_counter()
        steps = 0
        while True:
            res = eng.step_modern()
            steps += 1
            if steps in (25, 50, 100, 200, 400, 800, 1200, 1600):
                marks[steps] = round(time.perf_counter() - t0, 4)
            for r in res["finished_requests"]:
                fin[int(r.request_id) - 1_000_000 * rep_i] = [0, steps, len(r.generated_tokens)]
                done[int(r.request_id) - 1_000_000 * rep_i] = (
                    torch.tensor(list(r.generated_tokens), dtype=torch.int64),
                    torch.as_tensor(r.generated_log_probs, dtype=torch.float32))
            if B._all_ranks(not eng.has_unfinished_requests()):
                break
            if not eng.has_unfinished_requests():
                while not B._all_ranks(not eng.has_unfinished_requests()):
                    eng.controller.dummy_forward()
                break
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        if rep_i == 0:
            done0 = done
    done = done0  # exactness: the first run (the plain arm's reference keys); timing: the last run
    torch.save(done, f"{a.out}.rank{rank}.pt")
    out = {"arm": a.arm, "rank": rank, "steps": steps, "wall_s": wall, "requests": len(done)}
    out["marks"] = marks
    out["fin"] = {str(k): v for k, v in fin.items()}  # last repeat; all requests start at step 1 (one prefill)
    if prof:
        out["prof_ms_per_step"] = {k: round(1e3 * v / max(steps, 1), 4) for k, v in prof.items()}
    if mtd is not None:
        out["tree"] = mtd.summary()
        if getattr(mtd, "_debug", None) is not None and ref:
            # per iteration: the first extra row whose inputs were right but whose draw differs from the plain run
            bad_rows = {}
            n_ok_rows = 0
            for d in mtd._debug:
                plain = ref[d["rid"]][0].tolist()
                pl = len(plain)
                gen0 = d["p"] - (d["p"] - len(plain)) if False else None
                del gen0
                # position of the token row 0 samples = p (absolute); plain holds generated tokens only
                base = d["p"] - DEBUG_PLEN[d["rid"]]
                if base + d["kk"] >= pl:
                    continue
                for i in range(0, d["kk"] + 1):
                    # row i's inputs are right iff proposals d_1..d_i equal the plain tokens at base .. base+i-1
                    if i > 0 and d["prop"][i - 1] != plain[base + i - 1]:
                        break
                    if d["s"][i] != plain[base + i]:
                        bad_rows.setdefault(i, 0)
                        bad_rows[i] += 1
                        break
                    n_ok_rows += 1
            out["debug_wrong_rows_by_index"] = bad_rows
            out["debug_right_rows"] = n_ok_rows
            out["debug_first"] = mtd._debug[:6]
            json.dump({"debug": mtd._debug, "plain": {str(r): ref[r][0].tolist() for r in ref},
                       "plen": {str(k): v for k, v in DEBUG_PLEN.items()}}, open(f"{a.out}.rank{rank}.debug.json", "w"))
    if ref:
        bad = []
        for rid, (tk, lp) in done.items():
            rt, rl = ref[rid]
            if not (torch.equal(tk, rt) and torch.equal(lp.view(torch.int32), rl.view(torch.int32))):
                n = min(len(tk), len(rt))
                first = next((j for j in range(n) if tk[j] != rt[j] or lp[j].item() != rl[j].item()), n)
                bad.append((rid, int(len(tk)), int(len(rt)), int(first)))
        out["mismatch"] = bad
        out["exact"] = not bad and len(done) == len(ref)
        per = {}
        for rid in done:
            i = rid - 7_000_000
            per[i] = len(done[rid][0])
        out["tokens"] = per
    print(f"[tree test] {json.dumps(out)}", flush=True)
    json.dump(out, open(f"{a.out}.rank{rank}.json", "w"))
    dist.barrier()
    os._exit(0)


WIDE_HOLDER: dict = {}
DEBUG_PLEN: dict = {}

_orig_init = B.TsHooks.__init__


def _remember(self, *aa, **kk):
    _orig_init(self, *aa, **kk)
    WIDE_HOLDER["obj"] = self.obj


B.TsHooks.__init__ = _remember

if __name__ == "__main__":
    main()
