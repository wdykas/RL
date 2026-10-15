"""carried_piece / carried_scan on GPU rows (Triton residual masses) decide as on CPU."""

import pytest
import torch

from thundersync_rl.carried_walk import DecisionRule, carried_piece, carried_scan

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("trial", range(40))
def test_gpu_matches_cpu(trial):
    g = torch.Generator().manual_seed(trial)
    V, n = 4096, 64
    P = torch.log_softmax(torch.randn(n + 1, V, generator=g) * 2, -1)
    Q = torch.log_softmax(P[:n] + torch.randn(n, V, generator=g) * 0.05, -1)
    draft = torch.multinomial(Q.exp(), 1, generator=g)[:, 0]
    ar = torch.arange(n)
    lpx, lqx = P[ar, draft], Q[ar, draft]
    walk = {"log_b": -0.03 * (trial % 3), "floor": 0, "n": 0}
    outs = []
    for dev in ("cpu", "cuda"):
        Pd, Qd = P.to(dev), Q.to(dev)
        gen = torch.Generator(device=dev).manual_seed(5)
        out = carried_piece(lpx.to(dev), lqx.to(dev), torch.zeros(n + 1, device=dev), torch.zeros(n, device=dev),
                            lambda i: Pd[i], lambda i: Qd[i], None if dev == "cuda" else gen, walk,
                            DecisionRule("row"), True)
        outs.append(out)
    a, b = outs
    assert a["walk"] == b["walk"]
    if a["decision"] == "whole" or b["decision"] == "whole":
        return  # generators differ across devices; the walk is what must agree
    # with the same draws (eta drawn on CPU and passed through), r on GPU == r on CPU
    from thundersync_rl.carried_walk import _hits_desc, _rows_mass, advance

    lb, fl = advance((lpx - lqx).tolist(), walk["log_b"], 0, 0)
    eta = torch.rand(n, generator=torch.Generator().manual_seed(trial), dtype=torch.float64)
    m_cpu = _rows_mass(lambda i: P[i], lambda i: Q[i], torch.zeros(n + 1), torch.zeros(n))
    Pc, Qc = P.cuda(), Q.cuda()
    m_gpu = _rows_mass(lambda i: Pc[i], lambda i: Qc[i], torch.zeros(n + 1, device="cuda"), torch.zeros(n, device="cuda"))
    assert _hits_desc(lb, fl - 1, eta, 1e-4, m_cpu)[0] == _hits_desc(lb, fl - 1, eta.cuda(), 1e-4, m_gpu)[0]


@pytest.mark.parametrize("n,chunk", [(37, 8), (300, 4096), (5000, 4096)])
def test_deferred_logprob_rows_equal_materialized(n, chunk):
    """Chunked verification head rows == log_softmax of the whole-batch fp32 head (bitwise)."""
    from nemo_rl.models.megatron import fp16_gemm
    from thundersync_rl.engine_q import DeferredLogprobRows

    g = torch.Generator(device="cuda").manual_seed(n)
    h, V, vlim = 256, 3000, 2900
    x = torch.randn(n, h, device="cuda", generator=g)
    w = (torch.randn(V, h, device="cuda", generator=g) * 0.05).to(torch.bfloat16)
    full = torch.log_softmax(fp16_gemm.fp32_gemm(x.unsqueeze(1), w)[:, 0, :vlim].float(), -1)
    rows = DeferredLogprobRows(x, w, vlim, chunk_rows=chunk)
    tok = torch.randint(0, vlim, (n,), device="cuda", generator=g)
    assert torch.equal(rows.token_logprobs(tok), full[torch.arange(n, device="cuda"), tok])
    idx = torch.tensor([0, n // 2, n - 1], device="cuda")
    assert torch.equal(rows(idx), full[idx])
