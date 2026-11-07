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
"""tree_mtp.MTPHead (the MTP draft source of tree_spec) against a dense reference of MCore's MTP layer: the
incremental KV-cached row forward (append_kv of the prefix, forward_rows at the last position) equals the full causal
forward of the layer, per slot, with masked rows (slot -1) writing only to the sink."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

HID, NH, NKV, HD, FFN, V, EPS = 256, 8, 2, 32, 192, 512, 1e-5


def _sd(gen):
    def r(*s):
        return (torch.randn(*s, generator=gen) * 0.05).to(torch.bfloat16)

    def ln():
        return (1.0 + 0.1 * torch.randn(HID, generator=gen)).to(torch.bfloat16)

    p, a, f = "mtp.layers.0.", "mtp.layers.0.mtp_model_layer.layers.0.self_attention.", \
        "mtp.layers.0.mtp_model_layer.layers.1.mlp."
    return {
        p + "enorm.weight": ln(), p + "hnorm.weight": ln(), p + "eh_proj.weight": r(HID, 2 * HID),
        a + "linear_qkv.layer_norm_weight": ln(), a + "linear_qkv.weight": r((NH + 2 * NKV) * HD, HID),
        a + "linear_proj.weight": r(HID, NH * HD),
        f + "linear_fc1.layer_norm_weight": ln(), f + "linear_fc1.weight": r(FFN, HID), f + "linear_fc2.weight": r(HID, FFN),
        p + "final_layernorm.weight": ln(),
    }


def _rms(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()).to(torch.bfloat16)


def _ref_layer(sd, emb, out, h, tok_next):
    """Dense MCore MTP layer over one sequence: position i reads (h_i, x_{i+1}); causal attention, grouped QKV
    (per KV group: its query heads, then k, then v), squared-ReLU MLP, final norm, output layer. fp32 attention."""
    g = lambda k: sd[k].cuda()  # noqa: E731
    p = "mtp.layers.0."
    e = _rms(emb[tok_next], g(p + "enorm.weight"))
    u = torch.cat([e, _rms(h, g(p + "hnorm.weight"))], -1) @ g(p + "eh_proj.weight").t()
    return _ref_body(sd, out, u)


def _ref_body(sd, out, u):
    """The dense layer after eh_proj: causal attention, MLP, final norm, output layer."""
    g = lambda k: sd[k].cuda()  # noqa: E731
    a, f = "mtp.layers.0.mtp_model_layer.layers.0.self_attention.", "mtp.layers.0.mtp_model_layer.layers.1.mlp."
    p = "mtp.layers.0."
    n = u.shape[0]
    gq = NH // NKV
    x = (_rms(u, g(a + "linear_qkv.layer_norm_weight")) @ g(a + "linear_qkv.weight").t()).view(n, NKV, gq + 2, HD)
    q = x[:, :, :gq].reshape(n, NH, HD).float()
    k = x[:, :, gq].float().repeat_interleave(gq, dim=1)
    v = x[:, :, gq + 1].float().repeat_interleave(gq, dim=1)
    s = torch.einsum("qhd,khd->hqk", q, k) / HD**0.5
    s = s.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool, device=s.device), 1), float("-inf"))
    o = torch.einsum("hqk,khd->qhd", s.softmax(-1), v).to(torch.bfloat16)
    u = u + o.reshape(n, -1) @ g(a + "linear_proj.weight").t()
    z = torch.relu(_rms(u, g(f + "linear_fc1.layer_norm_weight")) @ g(f + "linear_fc1.weight").t())
    u = u + (z * z) @ g(f + "linear_fc2.weight").t()
    hf = _rms(u, g(p + "final_layernorm.weight"))
    return hf, hf @ out.t()


@cuda
def test_incremental_rows_equal_dense_layer(tmp_path):
    from nemo_rl.models.generation.megatron import tree_mtp

    gen = torch.Generator().manual_seed(0)
    sd = _sd(gen)
    path = tmp_path / "head.pt"
    torch.save(sd, path)
    emb = (torch.randn(V, HID, generator=gen) * 0.5).to(torch.bfloat16).cuda()
    out = (torch.randn(V, HID, generator=gen) * 0.05).to(torch.bfloat16).cuda()
    head = tree_mtp.MTPHead(str(path), emb, out, slots=3, max_seq=160, num_heads=NH, num_kv_heads=NKV, head_dim=HD,
                            eps=EPS)
    lens = {0: 37, 2: 131}  # slot -> sequence length (131 > one attention block of 64)
    seqs = {}
    for s, n in lens.items():
        h = torch.randn(n, HID, generator=gen).to(torch.bfloat16).cuda()
        tok = torch.randint(0, V, (n,), generator=gen).cuda()
        seqs[s] = (h, tok)
        # prefix K / V (positions 0 .. n-2), in two batches, plus a masked row that must not touch any slot
        cut = n // 2
        for lo, hi in ((0, cut), (cut, n - 1)):
            sl = torch.full((hi - lo + 1,), s, device="cuda")
            sl[-1] = -1
            ps = torch.cat([torch.arange(lo, hi, device="cuda"), torch.tensor([5], device="cuda")])
            head.append_kv(sl, ps, torch.cat([h[lo:hi], h[:1]]), torch.cat([tok[lo:hi], tok[:1]]))
    # both slots' last positions in one batch, with a masked row between them
    sl = torch.tensor([0, -1, 2], device="cuda")
    ps = torch.tensor([lens[0] - 1, 3, lens[2] - 1], device="cuda")
    hh = torch.stack([seqs[0][0][-1], seqs[0][0][0], seqs[2][0][-1]])
    tt = torch.stack([seqs[0][1][-1], seqs[0][1][0], seqs[2][1][-1]])
    hf, logits = head.forward_rows(sl, ps, hh, tt)
    for i, s in ((0, 0), (2, 2)):
        rh, rl = _ref_layer(sd, emb, out, *seqs[s])
        torch.testing.assert_close(hf[i].float(), rh[-1].float(), atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(logits[i].float(), rl[-1].float(), atol=3e-2, rtol=3e-2)
        assert torch.equal(logits[i].argmax(), rl[-1].argmax()) or (rl[-1].topk(2).values.diff().abs() < 2e-2).item()
    # the masked row attends nothing and wrote only the sink slot; slot 1 untouched
    assert head.kc[1].abs().sum() == 0 and head.vc[1].abs().sum() == 0


@cuda
def test_slot_start_masks_positions_without_kv(tmp_path):
    """Prefix caching: a slot whose request skipped cached prompt tokens has K / V from `start` on; attention reads
    only [start, pos] (== the dense layer over that suffix), whatever the positions before hold."""
    from nemo_rl.models.generation.megatron import tree_mtp

    gen = torch.Generator().manual_seed(1)
    sd = _sd(gen)
    path = tmp_path / "head.pt"
    torch.save(sd, path)
    emb = (torch.randn(V, HID, generator=gen) * 0.5).to(torch.bfloat16).cuda()
    out = (torch.randn(V, HID, generator=gen) * 0.05).to(torch.bfloat16).cuda()
    head = tree_mtp.MTPHead(str(path), emb, out, slots=2, max_seq=1200, num_heads=NH, num_kv_heads=NKV, head_dim=HD,
                            eps=EPS)
    s0, n = 600, 700  # start past one attention split (512)
    h = torch.randn(n, HID, generator=gen).to(torch.bfloat16).cuda()
    tok = torch.randint(0, V, (n,), generator=gen).cuda()
    head.kc[1].normal_()  # garbage before the start must not be read
    head.vc[1].normal_()
    head.set_start(1, s0)
    sl = torch.full((n - 1 - s0,), 1, device="cuda")
    head.append_kv(sl, torch.arange(s0, n - 1, device="cuda"), h[s0: n - 1], tok[s0: n - 1])  # (h_i, x_i+1) per row
    hf, logits = head.forward_rows(torch.tensor([1], device="cuda"), torch.tensor([n - 1], device="cuda"),
                                   h[n - 1: n], tok[n - 1: n])
    rh, rl = _ref_layer(sd, emb, out, h[s0:], tok[s0:])
    torch.testing.assert_close(hf[0].float(), rh[-1].float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(logits[0].float(), rl[-1].float(), atol=3e-2, rtol=3e-2)


@cuda
def test_v2_feature_head_rows_equal_dense_layer(tmp_path):
    """Input contract v2 (draft-head2): rows carry [h | f_0 | f_1]; eh_proj reads [enorm(e) | hnorm(h) | fnorm_0(f_0) |
    fnorm_1(f_1)]. The incremental row forward equals the dense layer on the same inputs; chain_input keeps the base
    position's features next to the previous depth's output."""
    from nemo_rl.models.generation.megatron import tree_mtp

    gen = torch.Generator().manual_seed(2)
    sd = _sd(gen)
    p = "mtp.layers.0."
    sd[p + "eh_proj.weight"] = (torch.randn(HID, 4 * HID, generator=gen) * 0.05).to(torch.bfloat16)
    for i in range(2):
        sd[p + f"fnorm{i}.weight"] = (1.0 + 0.1 * torch.randn(HID, generator=gen)).to(torch.bfloat16)
    sd["draft_head2.feats"] = torch.tensor([17, 34], dtype=torch.int32)
    path = tmp_path / "head_v2.pt"
    torch.save(sd, path)
    assert tree_mtp.head_feature_layers(str(path)) == [17, 34]
    emb = (torch.randn(V, HID, generator=gen) * 0.5).to(torch.bfloat16).cuda()
    out = (torch.randn(V, HID, generator=gen) * 0.05).to(torch.bfloat16).cuda()
    head = tree_mtp.MTPHead(str(path), emb, out, slots=1, max_seq=128, num_heads=NH, num_kv_heads=NKV, head_dim=HD,
                            eps=EPS)
    assert head.in_width == 3 * HID and head.feat_layers == [17, 34]
    n = 40
    rows = torch.randn(n, 3 * HID, generator=gen).to(torch.bfloat16).cuda()
    tok = torch.randint(0, V, (n,), generator=gen).cuda()
    sl = torch.zeros(n - 1, dtype=torch.long, device="cuda")
    head.append_kv(sl, torch.arange(n - 1, device="cuda"), rows[: n - 1], tok[: n - 1])
    hf, logits = head.forward_rows(torch.tensor([0], device="cuda"), torch.tensor([n - 1], device="cuda"),
                                   rows[n - 1:], tok[n - 1:])
    # dense reference: eh_proj's v2 input built explicitly, then the same layer body
    g = lambda k: sd[k].cuda()  # noqa: E731
    x = torch.cat([_rms(emb[tok], g(p + "enorm.weight")), _rms(rows[:, :HID], g(p + "hnorm.weight")),
                   _rms(rows[:, HID: 2 * HID], g(p + "fnorm0.weight")),
                   _rms(rows[:, 2 * HID:], g(p + "fnorm1.weight"))], -1)
    u = x @ g(p + "eh_proj.weight").t()
    rh, rl = _ref_body(sd, out, u)
    torch.testing.assert_close(hf[0].float(), rh[-1].float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(logits[0].float(), rl[-1].float(), atol=3e-2, rtol=3e-2)
    ci = head.chain_input(hf, rows[n - 1:])
    assert ci.shape == (1, 3 * HID) and torch.equal(ci[:, :HID], hf) and torch.equal(ci[:, HID:], rows[n - 1:, HID:])
