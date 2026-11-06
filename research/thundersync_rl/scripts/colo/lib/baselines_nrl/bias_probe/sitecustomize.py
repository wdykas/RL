"""Read-only MoE router expert-bias probe for the nrl bias-off arms (NRL_BIAS_PROBE=1; dir on PYTHONPATH).

Wraps megatron.core.distributed.finalize_model_grads._update_router_expert_bias (called once per optimizer step from
finalize_model_grads when moe_router_enable_expert_bias) and prints, on the process with torch.distributed rank 0:
  [nrl-bias] call=N rate=R n_bias=M numel=E max_abs_delta=D l1_from_first=F sum=S
max_abs_delta = max |expert_bias after - before| over every unfrozen expert_bias buffer of this rank (the same
selection Megatron updates); l1_from_first = sum |bias - bias at the first call| (drift over the run). Also prints
  [nrl-router] call=N n_router=M requires_grad=G numel=E equal_first=True|False max_abs_from_first=D checksum=C
for the router gate weights (bitwise comparison with the first call). Nothing is
written to the model; the wrapped call runs unchanged. Then chains to any other sitecustomize on sys.path.
"""
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

_TARGET = "megatron.core.distributed.finalize_model_grads"


def _install(mod):
    orig = getattr(mod, "_update_router_expert_bias", None)
    if orig is None or getattr(orig, "_nrl_bias_probe", False):
        return
    state = {"n": 0, "first": None}

    def _biases(model):
        import torch  # noqa: F401
        out = []
        for chunk in model:
            for m in chunk.modules():
                b = getattr(m, "expert_bias", None)
                if b is not None and not getattr(m, "frozen_expert_bias", False):
                    out.append(b)
        return out

    def wrapped(model, config, *a, **kw):
        import torch
        bl = _biases(model)
        before = [b.detach().clone() for b in bl]
        r = orig(model, config, *a, **kw)
        state["n"] += 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            if bl:
                d = max(float((b.detach() - p).abs().max()) for b, p in zip(bl, before))
                cur = torch.cat([b.detach().float().flatten() for b in bl])
                if state["first"] is None:
                    state["first"] = torch.cat([p.float().flatten() for p in before])
                l1 = float((cur - state["first"]).abs().sum())
                s = float(cur.double().sum())
                numel = cur.numel()
            else:
                d = l1 = s = 0.0
                numel = 0
            print(f"[nrl-bias] call={state['n']} rate={getattr(config, 'moe_router_bias_update_rate', None)} "
                  f"n_bias={len(bl)} numel={numel} max_abs_delta={d:.6g} l1_from_first={l1:.6g} sum={s:.9g}",
                  flush=True)
        _router_weights(model, rank)
        return r

    def _router_weights(model, rank):
        # Router gate weights (every megatron Router module of this rank): requires_grad count, bitwise equality with
        # the copy taken at the first call, max |w - w_first|, float64 checksum. finalize_model_grads runs before the
        # optimizer step, so call N sees the weights after N - 1 optimizer steps.
        import torch
        from megatron.core.transformer.moe.router import Router
        ws = []
        for chunk in model:
            for m in chunk.modules():
                if isinstance(m, Router) and getattr(m, "weight", None) is not None:
                    ws.append(m.weight)
        if rank != 0:
            return
        ids = [id(w) for w in ws]
        if state.get("wids") != ids:  # first call (or a different model object): take the reference copy
            state["wids"] = ids
            state["wfirst"] = [w.detach().clone() for w in ws]
        eq = all(torch.equal(w.detach(), f) for w, f in zip(ws, state["wfirst"]))
        d = max((float((w.detach().float() - f.float()).abs().max()) for w, f in zip(ws, state["wfirst"])), default=0.0)
        cs = sum(float(w.detach().double().sum()) for w in ws)
        nrg = sum(int(w.requires_grad) for w in ws)
        print(f"[nrl-router] call={state['n']} n_router={len(ws)} requires_grad={nrg} "
              f"numel={sum(w.numel() for w in ws)} equal_first={eq} max_abs_from_first={d:.6g} checksum={cs:.12g}",
              flush=True)

    wrapped._nrl_bias_probe = True
    mod._update_router_expert_bias = wrapped
    print(f"[nrl-bias] probe installed pid={os.getpid()}", flush=True)


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        _install(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        for f in sys.meta_path:
            if f is self:
                continue
            spec = f.find_spec(name, path, target) if hasattr(f, "find_spec") else None
            if spec is not None and spec.loader is not None:
                spec.loader = _Loader(spec.loader)
                return spec
        return None


if os.environ.get("NRL_BIAS_PROBE", "0") == "1":
    if _TARGET in sys.modules:
        _install(sys.modules[_TARGET])
    else:
        sys.meta_path.insert(0, _Finder())

# Chain to the next sitecustomize on sys.path (if any), so this file shadows nothing.
_here = os.path.dirname(os.path.abspath(__file__))
_rest = [p for p in sys.path if os.path.abspath(p or ".") != _here]
_spec = importlib.machinery.PathFinder.find_spec("sitecustomize", _rest)
if _spec is not None and _spec.origin and os.path.dirname(os.path.abspath(_spec.origin)) != _here:
    _m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
