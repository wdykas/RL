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
"""CUDA target-row chunk scan of the batch-invariant Mamba decode step (``mamba_bi_fused_decode``).

Per (decode row, head) the same arithmetic, instruction for instruction, as the training chunk-scan
kernel's chain for that row (MCore ``_chunk_scan_fwd_kernel`` on its pinned tile) followed by its
gate, as the Triton kernels compile it on sm_100:

    acc = mma.sync m16n8k8 tf32 chain over d_state (k ascending, raw fp32 operands, zero start) of C[tr] x S
    acc = acc * ex2.approx(dA[tr] * log2e)                                   (mul.rn)
    acc = the chain continued over k < K_MAX (K_MAX = end of tr's training M tile), A = cbw[tr, k],
          cbw = where(k <= tr, (CB[tr, k] * ex2.approx(min(dA[tr] - dA[k], 0) * log2e)) * dt[k], 0)
    acc = acc + x[tr] * D                                                      (mul.rn, add.rn)
    out = acc * (z * div.full(1, 1 + ex2.approx((0 - z) * log2e)))            (the gate)

A tensor-core MMA computes each row of its M tile from that row of A alone, so the target row sits
in row 0 of the 16-row tile (other rows 0) and every output column's chain is the training tile's.
mma.sync and tcgen05 TF32 give the same bits for the pinned SSD tiles on GB200 (research/mamba_dk2/
mma_equiv.py), and every result is checked bitwise against MCore's tile pipeline and the learner's
training forward in tests/unit/models/megatron/test_mamba_bi_fused_decode.py.

Data path (what differs from the Triton tile kernel): a program is one (row, head) with head_dim / 8
warps, each owning one 8-column MMA n-tile; the head's state rows and the replay rows below K_MAX
are streamed into padded shared memory by 16-byte cp.async, all requested before the first MMA (one
DRAM round trip per program), while the per-row scalars, the C row and the chunk weights are loaded
in one round trip. The program also advances the slot's write cursor. The SMALL variant (steps
whose programs fit in one wave of the GPU) also computes the CB row and, for a row that fills its
chunk, the chunk state + state passing (the pinned chunk-state tile's chain, then
``fma(exp(dA_last), S, new)``) in the same program.

The kernel is compiled per shape (head_dim, d_state, chunk, training-tile M) for the device's
architecture through torch's JIT extension path (``load_inline``, cached in TORCH_EXTENSIONS_DIR).
Launches go to the current stream with a fixed grid and no host sync or allocation (CUDA-graph
capturable).
"""

from __future__ import annotations

import threading

import torch

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Shape (compile-time, one build per shape): HD head_dim, DS d_state, CHUNK chunk size, REF_M the pinned
// training chunk-scan tile's M (the decode chain's K bound is the end of the target row's M tile).
#define NWARPS (HD / 8)          // one 8-column MMA n-tile per warp
#define NT (NWARPS * 32)
#define SPAD (DS + 4)            // padded smem state row: conflict-free B-fragment reads, 16-byte aligned rows
#define XP (HD + 8)              // padded smem replay row: conflict-free B-fragment reads, 16-byte aligned rows
#define SMEM_FLOATS (HD * SPAD + CHUNK * XP + DS + 2 * CHUNK)

__device__ __forceinline__ float ex2_approx(float x) {
  float r; asm volatile("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float tl_exp(float x) {   // Triton tl.exp (fp32): mul.f32 by log2e, ex2.approx.f32
  return ex2_approx(__fmul_rn(x, __uint_as_float(0x3FB8AA3Bu)));
}
__device__ __forceinline__ float min_f32(float a, float b) {
  float r; asm volatile("min.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ float div_full(float a, float b) {
  float r; asm volatile("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ void mma_tf32(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_row0(float* d, uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
  mma_tf32(d, a0, 0u, a2, 0u, b0, b1);   // A rows g + 8 are zero: only tile row 0 carries the target row
}
__device__ __forceinline__ float load_D(const void* p, int64_t i, int dtype) {
  // D is loaded in its parameter dtype and converted to fp32 (exact for bf16 / fp16), as the Triton kernels do
  if (dtype == 0) return reinterpret_cast<const float*>(p)[i];
  if (dtype == 1) return __uint_as_float(static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(p)[i]) << 16);
  return __half2float(reinterpret_cast<const __half*>(p)[i]);
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}

template <bool SMALL>
__global__ void __launch_bounds__(NT) scan_rows_kernel(
    const float* __restrict__ x, const float* __restrict__ C, const float* __restrict__ dt,
    const float* __restrict__ dA, const float* __restrict__ cb, const float* __restrict__ z,
    const void* __restrict__ Dp, float* __restrict__ states, float* __restrict__ out,
    const int* __restrict__ slots, const int* __restrict__ rows, const int* __restrict__ starts,
    const int* __restrict__ flags, int* __restrict__ nbuf, const float* __restrict__ Bb,
    int64_t sx_seq, int64_t sx_head, int64_t sc_seq, int64_t sc_head, int64_t sb_seq, int64_t sb_head,
    int64_t sdt_head, int64_t sdt_chunk, int64_t sdA_head, int64_t sdA_chunk,
    int64_t scb_row, int64_t scb_grp, int64_t sz_row, int64_t sz_head, int64_t sD_head,
    int64_t sst_batch, int64_t sst_head, int64_t sout_row, int64_t sout_head,
    int ratio, int has_D, int D_hdim, int D_dtype) {
  extern __shared__ __align__(16) float smem[];
  float* sS = smem;                        // [HD][SPAD]   the head's state
  float* sX = sS + HD * SPAD;              // [CHUNK][XP]  replay x rows below K_MAX
  float* sC = sX + CHUNK * XP;             // [DS]         C[tr]
  float* sW = sC + DS;                     // [CHUNK]      the target row's chunk weights
  float* sCB = sW + CHUNK;                 // [CHUNK]      CB[tr, :] (SMALL)
  const int64_t c = blockIdx.x;
  const int h = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  // the per-row scalars together (one round trip before the copies are issued)
  const int tr = rows[c];
  const int64_t start = starts[c];
  const int64_t slot = slots[c];
  float* o = out + c * sout_row + h * sout_head;
  if (tr < 0) {   // padding entry: zero output row
    for (int i = tid; i < HD; i += NT) o[i] = 0.f;
    return;
  }
  const int grp = h / ratio;
  const int K_MAX = min((tr / REF_M + 1) * REF_M, CHUNK);
  const int nks = K_MAX / 8;
  // ---- the head's state and the replay rows below K_MAX -> padded smem (cp.async, all issued first) ----
  float* S = states + slot * sst_batch + h * sst_head;   // [HD][DS] contiguous
  for (int j = tid; j < HD * (DS / 4); j += NT) {
    const int r = j / (DS / 4), c4 = j % (DS / 4);
    cp_async16(&sS[r * SPAD + c4 * 4], S + r * DS + c4 * 4);
  }
  const float* xh = x + start * sx_seq + h * sx_head;
  for (int j = tid; j < K_MAX * (HD / 4); j += NT) {
    const int r = j / (HD / 4), c4 = j % (HD / 4);
    cp_async16(&sX[r * XP + c4 * 4], xh + (int64_t)r * sx_seq + c4 * 4);
  }
  asm volatile("cp.async.commit_group;\n" ::);
  const float* dth = dt + c * sdt_chunk + h * sdt_head;
  const float* dAh = dA + c * sdA_chunk + h * sdA_head;
  const float* Cr = C + (start + tr) * sc_seq + grp * sc_head;
  const float* Bg = Bb + start * sb_seq + grp * sb_head;
  const float dA_m = dAh[tr];
  const float* cbr = cb + c * scb_row + grp * scb_grp;
  if (SMALL) {
    for (int k = tid; k < DS; k += NT) sC[k] = Cr[k];
    // CB[tr, j] for j <= tr: the CB tile's per-element chain (tf32 MMA over d_state, k ascending, from 0)
    __syncthreads();
    for (int jt = warp; jt <= tr / 8; jt += NWARPS) {
      float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int ks = 0; ks < DS / 8; ++ks) {
        const uint32_t a0 = g == 0 ? __float_as_uint(sC[ks * 8 + t]) : 0u;
        const uint32_t a2 = g == 0 ? __float_as_uint(sC[ks * 8 + t + 4]) : 0u;
        const float* br = Bg + (int64_t)(jt * 8 + g) * sb_seq + ks * 8 + t;
        mma_row0(d, a0, a2, __float_as_uint(br[0]), __float_as_uint(br[4]));
      }
      if (g == 0) { sCB[jt * 8 + 2 * t] = d[0]; sCB[jt * 8 + 2 * t + 1] = d[1]; }
    }
    __syncthreads();
    cbr = sCB;
  }
  // ---- the C row (large path) and the target row's chunk weights: one k per thread, the loads issued together
  // (one round trip, overlapping the copies) ----
  for (int k = tid; k < (DS > CHUNK ? DS : CHUNK); k += NT) {
    const float ck = (!SMALL && k < DS) ? Cr[k] : 0.f;
    if (k < CHUNK) {
      const float cbk = cbr[k], dAk = dAh[k], dtk = dth[k];
      float w = 0.f;
      if (k <= tr) w = __fmul_rn(__fmul_rn(cbk, tl_exp(min_f32(__fsub_rn(dA_m, dAk), 0.f))), dtk);
      sW[k] = w;   // where(k <= tr, w, 0)
    }
    if (!SMALL && k < DS) sC[k] = ck;
  }
  const float scale = tl_exp(dA_m);
  // epilogue operands (lanes g == 0 own columns 2t, 2t + 1 of the warp's n-tile)
  const int p0 = warp * 8 + 2 * t;
  float zz[2], xr[2], Dv[2];
  if (g == 0) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      zz[e] = z[c * sz_row + h * sz_head + p0 + e];
      xr[e] = xh[(int64_t)tr * sx_seq + p0 + e];
      Dv[e] = has_D ? load_D(Dp, D_hdim ? h * sD_head + p0 + e : h * sD_head, D_dtype) : 0.f;
    }
  }
  asm volatile("cp.async.wait_all;\n" ::);
  __syncthreads();

  // ---- acc = C[tr] . S (k ascending), then * exp(dA[tr]) ----
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const bool r0 = g == 0;
  const float* srow = &sS[(warp * 8 + g) * SPAD + t];
#pragma unroll
  for (int ks = 0; ks < DS / 8; ++ks) {
    const uint32_t a0 = r0 ? __float_as_uint(sC[ks * 8 + t]) : 0u;
    const uint32_t a2 = r0 ? __float_as_uint(sC[ks * 8 + t + 4]) : 0u;
    mma_row0(acc, a0, a2, __float_as_uint(srow[ks * 8]), __float_as_uint(srow[ks * 8 + 4]));
  }
#pragma unroll
  for (int e = 0; e < 4; ++e) acc[e] = __fmul_rn(acc[e], scale);
  // ---- acc += cbw . x over the K steps below K_MAX (k ascending) ----
  const int col = warp * 8 + g;
  for (int ks = 0; ks < nks; ++ks) {
    const uint32_t a0 = r0 ? __float_as_uint(sW[ks * 8 + t]) : 0u;
    const uint32_t a2 = r0 ? __float_as_uint(sW[ks * 8 + t + 4]) : 0u;
    mma_row0(acc, a0, a2, __float_as_uint(sX[(ks * 8 + t) * XP + col]),
             __float_as_uint(sX[(ks * 8 + t + 4) * XP + col]));
  }
  // ---- D, gate, store ----
  if (r0) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      float v = acc[e];
      if (has_D) v = __fadd_rn(v, __fmul_rn(xr[e], Dv[e]));
      const float zv = zz[e];
      const float sg = div_full(1.f, __fadd_rn(tl_exp(__fsub_rn(0.f, zv)), 1.f));
      o[p0 + e] = __fmul_rn(v, __fmul_rn(sg, zv));
    }
  }
  // ---- chunk boundary (SMALL; the row filled its chunk): chunk state + state passing for this head ----
  // new[p, n] = sum_k x[k, p] * (B[k, n] * w[k]) as the chunk-state tile's tf32 MMA chain over k ascending from 0,
  // w[k] = exp(min(dA[last] - dA[k], 0)) * dt[k]; then state[p, n] = fma(exp(dA[last]), state[p, n], new[p, n]).
  const int flag = flags[c];
  if (h == 0 && tid == 0) nbuf[slot] = flag != 0 ? 0 : tr + 1;   // advance the slot's write cursor
  if (SMALL && flag != 0) {
    __syncthreads();   // every warp is done with sW
    const float dA_last = dAh[CHUNK - 1];
    for (int k = tid; k < CHUNK; k += NT) sW[k] = __fmul_rn(tl_exp(min_f32(__fsub_rn(dA_last, dAh[k]), 0.f)), dth[k]);
    __syncthreads();
    const float decay = tl_exp(dA_last);
    for (int u = warp; u < (HD / 16) * (DS / 64); u += NWARPS) {   // (16-row M tile, 8 n-tiles) units
      const int m0 = (u % (HD / 16)) * 16, nbase = (u / (HD / 16)) * 64;
      float sacc[8][4];
#pragma unroll
      for (int i = 0; i < 8; ++i) { sacc[i][0] = sacc[i][1] = sacc[i][2] = sacc[i][3] = 0.f; }
#pragma unroll 4
      for (int ks = 0; ks < CHUNK / 8; ++ks) {
        const int k0 = ks * 8 + t, k1 = k0 + 4;
        const uint32_t a0 = __float_as_uint(sX[k0 * XP + m0 + g]);
        const uint32_t a1 = __float_as_uint(sX[k0 * XP + m0 + g + 8]);
        const uint32_t a2 = __float_as_uint(sX[k1 * XP + m0 + g]);
        const uint32_t a3 = __float_as_uint(sX[k1 * XP + m0 + g + 8]);
        const float w0 = sW[k0], w1 = sW[k1];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int n = nbase + i * 8 + g;
          mma_tf32(sacc[i], a0, a1, a2, a3, __float_as_uint(__fmul_rn(Bg[(int64_t)k0 * sb_seq + n], w0)),
                   __float_as_uint(__fmul_rn(Bg[(int64_t)k1 * sb_seq + n], w1)));
        }
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int n = nbase + i * 8 + 2 * t;
#pragma unroll
        for (int hr = 0; hr < 2; ++hr) {
          const int pr = m0 + g + 8 * hr;
          float2 v;
          v.x = __fmaf_rn(decay, sS[pr * SPAD + n], sacc[i][2 * hr]);
          v.y = __fmaf_rn(decay, sS[pr * SPAD + n + 1], sacc[i][2 * hr + 1]);
          *reinterpret_cast<float2*>(&S[pr * DS + n]) = v;
        }
      }
    }
  }
}

static int smem_bytes() { return SMEM_FLOATS * sizeof(float); }

template <bool SMALL>
static void set_attrs() {
  static bool done = false;
  if (!done) {
    cudaFuncSetAttribute(scan_rows_kernel<SMALL>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes());
    cudaFuncSetAttribute(scan_rows_kernel<SMALL>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    done = true;
  }
}

int64_t resident_programs_per_sm(bool small) {
  int n = 0;
  if (small) { set_attrs<true>(); cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, scan_rows_kernel<true>, NT, smem_bytes()); }
  else { set_attrs<false>(); cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, scan_rows_kernel<false>, NT, smem_bytes()); }
  return n;
}

void scan_rows(torch::Tensor x, torch::Tensor C, torch::Tensor dt, torch::Tensor dA, torch::Tensor cb,
               torch::Tensor z, c10::optional<torch::Tensor> D, torch::Tensor states, torch::Tensor out,
               torch::Tensor slots, torch::Tensor rows, torch::Tensor starts, torch::Tensor flags,
               torch::Tensor nbuf, torch::Tensor B, int64_t n, int64_t ratio, bool small) {
  const int nheads = x.size(1);
  TORCH_CHECK(x.size(2) == HD && C.size(2) == DS && states.size(2) == HD && states.size(3) == DS && B.size(2) == DS);
  TORCH_CHECK(states.stride(3) == 1 && states.stride(2) == DS && x.stride(2) == 1 && C.stride(2) == 1 && B.stride(2) == 1);
  TORCH_CHECK(z.stride(2) == 1 && out.stride(2) == 1 && dt.stride(2) == 1 && dA.stride(2) == 1 && cb.stride(2) == 1);
  TORCH_CHECK(z.scalar_type() == at::kFloat && x.scalar_type() == at::kFloat && states.scalar_type() == at::kFloat);
  if (n == 0) return;
  const bool has_D = D.has_value();
  const void* Dp = has_D ? D->data_ptr() : nullptr;
  const int D_hdim = has_D && D->dim() == 2;
  int D_dtype = 0;
  if (has_D) {
    const auto st = D->scalar_type();
    TORCH_CHECK(st == at::kFloat || st == at::kBFloat16 || st == at::kHalf, "D: fp32 / bf16 / fp16");
    TORCH_CHECK(D->stride(-1) == 1, "D: contiguous last dim");
    D_dtype = st == at::kFloat ? 0 : (st == at::kBFloat16 ? 1 : 2);
  }
  dim3 grid(n, nheads);
  auto stream = at::cuda::getCurrentCUDAStream();
#define SCAN_ARGS                                                                                                  \
  x.data_ptr<float>(), C.data_ptr<float>(), dt.data_ptr<float>(), dA.data_ptr<float>(), cb.data_ptr<float>(),      \
      z.data_ptr<float>(), Dp, states.data_ptr<float>(), out.data_ptr<float>(), slots.data_ptr<int>(),           \
      rows.data_ptr<int>(), starts.data_ptr<int>(), flags.data_ptr<int>(), nbuf.data_ptr<int>(), B.data_ptr<float>(), \
      x.stride(0), x.stride(1), C.stride(0), C.stride(1), B.stride(0), B.stride(1), dt.stride(0), dt.stride(1),    \
      dA.stride(0), dA.stride(1), cb.stride(0), cb.stride(1), z.stride(0), z.stride(1), has_D ? D->stride(0) : 0, \
      states.stride(0), states.stride(1), out.stride(0), out.stride(1), (int)ratio, (int)has_D, D_hdim, D_dtype
  if (small) {
    set_attrs<true>();
    scan_rows_kernel<true><<<grid, NT, smem_bytes(), stream>>>(SCAN_ARGS);
  } else {
    set_attrs<false>();
    scan_rows_kernel<false><<<grid, NT, smem_bytes(), stream>>>(SCAN_ARGS);
  }
#undef SCAN_ARGS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_CPP = (
    "void scan_rows(torch::Tensor x, torch::Tensor C, torch::Tensor dt, torch::Tensor dA, torch::Tensor cb,"
    " torch::Tensor z, c10::optional<torch::Tensor> D, torch::Tensor states, torch::Tensor out,"
    " torch::Tensor slots, torch::Tensor rows, torch::Tensor starts, torch::Tensor flags,"
    " torch::Tensor nbuf, torch::Tensor B, int64_t n, int64_t ratio, bool small);\n"
    "int64_t resident_programs_per_sm(bool small);"
)

_MODS: dict = {}
_LOCK = threading.Lock()


def smem_bytes(hd: int, ds: int, chunk: int) -> int:
    """Dynamic shared memory per program (state + replay rows + C row + weights + CB row)."""
    return 4 * (hd * (ds + 4) + chunk * (hd + 8) + ds + 2 * chunk)


def supported(hd: int, ds: int, chunk: int, ref_m: int, device) -> bool:
    """Shapes the kernel is written for, on a device with mma.sync TF32 and cp.async (sm_80+)."""
    if torch.cuda.get_device_capability(device) < (8, 0):
        return False
    if hd % 16 or ds % 64 or chunk % ref_m or ref_m % 8 or hd // 8 > 32:
        return False
    optin = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    return smem_bytes(hd, ds, chunk) <= optin


def module(hd: int, ds: int, chunk: int, ref_m: int, device=None):
    """The extension for one shape on ``device``'s architecture, through torch's JIT extension build (cached in
    ``TORCH_EXTENSIONS_DIR``; the launchers point it, with the Triton cache, at the job's cache directory)."""
    major, minor = torch.cuda.get_device_capability(device)
    key = (hd, ds, chunk, ref_m, major, minor)
    if key in _MODS:
        return _MODS[key]
    with _LOCK:
        if key not in _MODS:
            from torch.utils.cpp_extension import load_inline

            arch = f"{major}{minor}"
            _MODS[key] = load_inline(
                name=f"mamba_bi_scan_cuda_h{hd}d{ds}c{chunk}m{ref_m}_sm{arch}",
                cpp_sources=_CPP,
                cuda_sources=_SRC,
                functions=["scan_rows", "resident_programs_per_sm"],
                extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "--fmad=false",
                                   f"-DHD={hd}", f"-DDS={ds}", f"-DCHUNK={chunk}", f"-DREF_M={ref_m}"],
                verbose=False,
            )
    return _MODS[key]
