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
"""MTP draft source of tree_spec (INTERFACE.md v2): a 1-layer Megatron-Core MTP head ("*-": attention + dense MLP),
run as a few fixed-shape device rows per tail right after the tree's accept, drafting the next iteration's candidate
chain with the target's own keys (coupled draws: per-token acceptance ~ 1 - TV).

Head (MCore MultiTokenPredictionLayer semantics): at position i, u = eh_proj([enorm(emb(x_{i+1})), hnorm(h_i)]) with
h_i the decoder output after the final norm (the LM head's input); u + attn(LN(u)) over the head's own KV of every
committed position; + MLP (squared ReLU); final norm; the main model's output layer -> logits for x_{i+2}. Depth k > 1
chains the layer on its own output hidden (after its final norm, as MCore chains depths) and the drawn token.

Proposals only change how many iterations a row takes, never what it samples (tree_spec's exactness argument), so the
head's arithmetic need not be batch invariant: it only has to be identical on every TP rank of an engine (same inputs,
same kernels). Built for TP 1 (Nano's engine); TP > 1 needs a vocab gather of the head's logits (refused).

Per rank: head weights (~48M params bf16 at Nano: 96 MB) + its KV for `slots` requests ([slots, max_seq, kv heads,
head dim] x 2, bf16: 16 x 10240 x 2 x 128 x 2 x 2 B = 168 MB at Nano) + the decoder-output capture buffer
([max tokens, hidden] bf16). Per iteration and tail: <= 8 KV rows (the accepted positions) and K draft rows
(K full layer forwards + K LM-head rows). No communication.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

SPLIT = 512  # positions per attention split (flash decoding: a long context spreads over many programs)
_HID: dict = {}  # decoder-output capture: {"buf": [max_tokens, hidden] bf16, "n": rows of the last forward}


def install_hidden_capture(model, max_tokens: int, hidden: int) -> None:
    """Copy the decoder's final-norm output of every forward into a fixed buffer (before the engine is built: if the
    final norm runs inside a captured graph, the copy is captured with it)."""
    if _HID:
        return
    dev = torch.device("cuda", torch.cuda.current_device())
    _HID["buf"] = torch.zeros(int(max_tokens), int(hidden), dtype=torch.bfloat16, device=dev)
    from megatron.core.utils import unwrap_model

    m = unwrap_model(model)
    norm = m.decoder.final_norm

    def hook(_mod, _inp, out):
        o = out[0] if isinstance(out, tuple) else out
        flat = o.reshape(-1, o.shape[-1])
        n = min(flat.shape[0], _HID["buf"].shape[0])
        _HID["buf"][:n].copy_(flat[:n])
        return None

    norm.register_forward_hook(hook)


def hidden_buffer() -> Optional[torch.Tensor]:
    return _HID.get("buf")


if HAVE_TRITON:

    @triton.jit
    def _attn_kernel(q, kc, vc, slot, start, length, pm, pl, pacc, s_q_r, s_q_h, s_c_s, s_c_t, s_c_h,
                     GQ: tl.constexpr, GQP: tl.constexpr, HD: tl.constexpr, BT: tl.constexpr, SPL: tl.constexpr,
                     NS: tl.constexpr, NH: tl.constexpr):
        """Program (row r, KV head g, split c): the online-softmax partial (max, sum, weighted V) of the GQ query
        heads of KV head g over positions [max(c * SPL, start[slot]), min((c + 1) * SPL, length[r])) of the row's
        slot (flash decoding: each K / V block read once per group; splits past the length write max -inf; positions
        before the slot's start have no K / V: prefix-cache skipped prompt tokens)."""
        r = tl.program_id(0)
        g = tl.program_id(1)
        c = tl.program_id(2)
        s = tl.load(slot + r)
        n = tl.load(length + r)
        d = tl.arange(0, HD)
        hh = tl.arange(0, GQP)
        mh = hh < GQ
        h = g * GQ + hh
        lo = tl.maximum(c * SPL, tl.load(start + tl.maximum(s, 0)))
        hi = tl.minimum(c * SPL + SPL, n)
        m = tl.full([GQP], float("-inf"), tl.float32)
        l = tl.zeros([GQP], tl.float32)
        acc = tl.zeros([GQP, HD], tl.float32)
        if (s >= 0) & (lo < hi):
            qv = tl.load(q + r * s_q_r + h[:, None] * s_q_h + d[None, :], mask=mh[:, None], other=0.0)
            base = s.to(tl.int64) * s_c_s + g * s_c_h
            scale = 1.0 / math.sqrt(HD)
            for t0 in range(lo, hi, BT):
                t = t0 + tl.arange(0, BT)
                mt = t < hi
                kk = tl.load(kc + base + t[:, None].to(tl.int64) * s_c_t + d[None, :], mask=mt[:, None], other=0.0)
                sc = tl.dot(qv, tl.trans(kk)).to(tl.float32) * scale  # [GQP, BT]
                sc = tl.where(mt[None, :], sc, float("-inf"))
                mn = tl.maximum(m, tl.max(sc, axis=1))
                p = tl.exp(sc - mn[:, None])
                alpha = tl.exp(m - mn)
                vv = tl.load(vc + base + t[:, None].to(tl.int64) * s_c_t + d[None, :], mask=mt[:, None], other=0.0)
                acc = acc * alpha[:, None] + tl.dot(p.to(vv.dtype), vv).to(tl.float32)
                l = l * alpha + tl.sum(p, axis=1)
                m = mn
        pidx = (r * NH + h) * NS + c  # [GQP]
        tl.store(pm + pidx, m, mask=mh)
        tl.store(pl + pidx, l, mask=mh)
        tl.store(pacc + pidx[:, None].to(tl.int64) * HD + d[None, :], acc, mask=mh[:, None])

    @triton.jit
    def _attn_combine_kernel(pm, pl, pacc, out, s_o_r, s_o_h, NH: tl.constexpr, NS: tl.constexpr,
                             NSP: tl.constexpr, HD: tl.constexpr):
        """Program (row r, head h): merge the NS split partials (rows with no position: zeros)."""
        r = tl.program_id(0)
        h = tl.program_id(1)
        c = tl.arange(0, NSP)
        mc = c < NS
        d = tl.arange(0, HD)
        base = (r * NH + h) * NS
        m = tl.load(pm + base + c, mask=mc, other=float("-inf"))
        l = tl.load(pl + base + c, mask=mc, other=0.0)
        mx = tl.max(m, axis=0)
        w = tl.where(m > float("-inf"), tl.exp(m - mx), 0.0)
        a = tl.load(pacc + (base + c)[:, None].to(tl.int64) * HD + d[None, :], mask=mc[:, None], other=0.0)
        den = tl.sum(w * l, axis=0)
        num = tl.sum(w[:, None] * a, axis=0)
        o = tl.where(den > 0, num / tl.where(den > 0, den, 1.0), 0.0)
        tl.store(out + r * s_o_r + h * s_o_h + d, o.to(out.dtype.element_ty))

    @triton.jit
    def _rms_kernel(x, w, out, s_x, s_o, N: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr):
        """Program (row): RMSNorm in fp32 (bf16 out), x * rsqrt(mean(x^2) + eps) * w."""
        r = tl.program_id(0)
        i = tl.arange(0, BLOCK)
        mk = i < N
        v = tl.load(x + r.to(tl.int64) * s_x + i, mask=mk, other=0.0).to(tl.float32)
        rs = tl.rsqrt(tl.sum(v * v, axis=0) / N + EPS)
        ww = tl.load(w + i, mask=mk, other=0.0).to(tl.float32)
        tl.store(out + r.to(tl.int64) * s_o + i, (v * rs * ww).to(out.dtype.element_ty), mask=mk)


def _rms(x, w, eps, out=None):
    """RMSNorm of the rows of x (fp32 math, bf16 out; one fused kernel), optionally into a column slice ``out``."""
    if x.stride(-1) != 1:
        x = x.contiguous()
    n, hid = x.shape
    if out is None:
        out = torch.empty(n, hid, dtype=torch.bfloat16, device=x.device)
    if n:
        _rms_kernel[(n,)](x, w, out, x.stride(0), out.stride(0), N=hid, BLOCK=triton.next_power_of_2(hid),
                          EPS=float(eps))
    return out


class MTPHead:
    """The trained head on device, its KV cache, and the batched row forward."""

    def __init__(self, path: str, embedding: torch.Tensor, output: torch.Tensor, slots: int, max_seq: int,
                 num_heads: int, num_kv_heads: int, head_dim: int, eps: float = 1e-5):
        dev = embedding.device
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        g = lambda k: sd[k].to(device=dev, dtype=torch.bfloat16).contiguous()  # noqa: E731
        p = "mtp.layers.0."
        self.enorm, self.hnorm, self.eh = g(p + "enorm.weight"), g(p + "hnorm.weight"), g(p + "eh_proj.weight")
        a = p + "mtp_model_layer.layers.0.self_attention."
        self.ln_qkv, self.w_qkv, self.w_proj = g(a + "linear_qkv.layer_norm_weight"), g(a + "linear_qkv.weight"), \
            g(a + "linear_proj.weight")
        f = p + "mtp_model_layer.layers.1.mlp."
        self.ln_fc1, self.w_fc1, self.w_fc2 = g(f + "linear_fc1.layer_norm_weight"), g(f + "linear_fc1.weight"), \
            g(f + "linear_fc2.weight")
        self.final = g(p + "final_layernorm.weight")
        self.emb, self.out = embedding, output  # the main model's (TP 1: full vocab)
        assert self.emb.shape[0] == self.out.shape[0], "tree_mtp needs the full vocabulary on the rank (TP 1)"
        self.nh, self.nkv, self.hd, self.eps = num_heads, num_kv_heads, head_dim, eps
        self.gq = num_heads // num_kv_heads
        self.slots, self.max_seq = int(slots), int(max_seq)
        self.kc = torch.zeros(self.slots + 1, self.max_seq, num_kv_heads, head_dim, dtype=torch.bfloat16, device=dev)
        self.vc = torch.zeros_like(self.kc)  # the extra slot is the masked rows' sink
        self.start = torch.zeros(self.slots + 1, dtype=torch.int32, device=dev)  # first position with K / V

    def set_start(self, slot: int, pos: int) -> None:
        """A slot's request has K / V from position ``pos`` on (a prefix-cache hit skipped the tokens before it)."""
        self.start[int(slot)] = int(pos)

    # ---- the layer, in pieces -------------------------------------------------------------------------------------
    def _u(self, h, tok):
        """eh_proj([enorm(emb(tok)), hnorm(h)])."""
        hid = h.shape[-1]
        x = torch.empty(h.shape[0], 2 * hid, dtype=torch.bfloat16, device=h.device)  # [enorm(e) | hnorm(h)], no cat
        _rms(self.emb.index_select(0, tok.clamp(min=0)), self.enorm, self.eps, out=x[:, :hid])
        _rms(h, self.hnorm, self.eps, out=x[:, hid:])
        return x @ self.eh.t()

    def _qkv(self, u):
        """MCore's grouped QKV layout: per KV group, its query heads, then k, then v."""
        x = (_rms(u, self.ln_qkv, self.eps) @ self.w_qkv.t()).view(-1, self.nkv, self.gq + 2, self.hd)
        q = x[:, :, : self.gq].reshape(-1, self.nh, self.hd)
        return q, x[:, :, self.gq], x[:, :, self.gq + 1]

    def append_kv(self, slot, pos, h, tok):
        """K / V of rows (slot, position) for the head's input (h_i, x_{i+1}); slot -1 rows go to the sink."""
        u = self._u(h, tok)
        _q, k, v = self._qkv(u)
        sl = torch.where(slot >= 0, slot, torch.full_like(slot, self.slots))
        ps = pos.clamp(0, self.max_seq - 1)
        self.kc[sl, ps] = k
        self.vc[sl, ps] = v
        return u

    def _attend(self, q, sl, length):
        """Attention of query rows q [n, heads, dim] over positions 0 .. length - 1 of their slots sl (split + merge)."""
        n = q.shape[0]
        o = torch.empty_like(q)
        if n == 0:
            return o
        ns = triton.cdiv(self.max_seq, SPLIT)
        pm = torch.empty(n * self.nh * ns, dtype=torch.float32, device=q.device)
        pl = torch.empty_like(pm)
        pacc = torch.empty(n * self.nh * ns, self.hd, dtype=torch.float32, device=q.device)
        _attn_kernel[(n, self.nkv, ns)](q, self.kc, self.vc, sl.to(torch.int32), self.start, length, pm, pl, pacc,
                                        q.stride(0), q.stride(1), self.kc.stride(0), self.kc.stride(1),
                                        self.kc.stride(2), GQ=self.gq, GQP=max(16, triton.next_power_of_2(self.gq)),
                                        HD=self.hd, BT=64, SPL=SPLIT, NS=ns, NH=self.nh)
        _attn_combine_kernel[(n, self.nh)](pm, pl, pacc, o, o.stride(0), o.stride(1), NH=self.nh, NS=ns,
                                           NSP=max(2, triton.next_power_of_2(ns)), HD=self.hd)
        return o

    def forward_rows(self, slot, pos, h, tok):
        """Full layer at rows (slot, position): appends their K / V, attends over positions 0..pos, returns (the layer
        output after the head's final norm (MCore's MTP output hidden, the chaining input of the next depth), logits
        for the token two positions on)."""
        u = self._u(h, tok)
        q, k, v = self._qkv(u)
        sl = torch.where(slot >= 0, slot, torch.full_like(slot, self.slots))
        ps = pos.clamp(0, self.max_seq - 1)
        self.kc[sl, ps] = k
        self.vc[sl, ps] = v
        n = q.shape[0]
        o = self._attend(q, sl, torch.where(slot >= 0, ps + 1, torch.zeros_like(ps)).to(torch.int32))
        u = u + o.reshape(n, -1) @ self.w_proj.t()
        z = torch.relu(_rms(u, self.ln_fc1, self.eps) @ self.w_fc1.t())
        u = u + (z * z) @ self.w_fc2.t()
        hf = _rms(u, self.final, self.eps)
        return hf, hf.to(self.out.dtype) @ self.out.t()  # the output layer may be held in fp16 (fp16_gemm)
