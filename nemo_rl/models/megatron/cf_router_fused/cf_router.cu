// Fused midpoint cross-fade MoE router (cf_router.md): one warp per token row.
//
// Bits: every fp32 op is the reference's (PyTorch) op with the same IEEE rounding:
//   s = 1 / (1 + expf(-x))            torch.sigmoid's CUDA kernel (precise expf, IEEE divide)
//   t = s + b
//   selection: the C largest t, ties to the lower id (stable sort of -t), as 64-bit keys (ordered t, ~id)
//   tau = (v[k-1] + v[k]) * 0.5
//   c = clamp(u, 0, 1), u = (v - tau) / m + 0.5; torch divides a tensor by a Python scalar as a multiply by the
//       host-computed fp32 reciprocal (RECIP=1), or as an IEEE divide (RECIP=0): the variant is picked by test
//   w = s[idx] * c, W = sequential sum over the C slots, p = w / (W + 1e-20f), p = p * sf
// Explicit __f*_rn intrinsics: nothing is contracted into an fma (the reference runs each op as its own kernel).
// Lane j of the warp owns candidate slot j (C <= 32).
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

namespace {

__device__ __forceinline__ uint32_t ord_key(float t) {
  // order-preserving map of a non-NaN float to uint32 (+0 and -0 equal, as the sort compares them)
  if (t == 0.f) t = 0.f;
  uint32_t u = __float_as_uint(t);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float sigmoid_ref(float x) {
  return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-x)));
}

// Selection shared by forward and backward: on return, lane j < C holds the expert id of slot j and recomputes
// its (s, v) with the same ops. Per round: one redux.sync max over the ordered t of every lane's current best,
// one redux.sync min over the ids of the lanes holding that t (ties to the lower id); the winning lane pops it.
template <int PL>
__device__ __forceinline__ int select_row(const float* __restrict__ xrow, const float* __restrict__ bias, int E,
                                          int C, int lane) {
  uint32_t kt[PL];
#pragma unroll
  for (int i = 0; i < PL; ++i) {
    const int e = lane + 32 * i;
    kt[i] = (e < E) ? ord_key(__fadd_rn(sigmoid_ref(xrow[e]), bias[e])) : 0u;
  }
  uint32_t bt = 0u;
  int bi = 0;
#pragma unroll
  for (int i = 0; i < PL; ++i)
    if (kt[i] > bt) { bt = kt[i]; bi = i; }  // strict: the first (lowest id) of equal keys
  int slot = 0;
  for (int j = 0; j < C; ++j) {
    const uint32_t mt = __reduce_max_sync(0xffffffffu, bt);
    const uint32_t wid = __reduce_min_sync(0xffffffffu, bt == mt ? static_cast<uint32_t>(lane + 32 * bi) : 0xffffffffu);
    if (lane == j) slot = static_cast<int>(wid);
    if (lane == static_cast<int>(wid & 31u)) {
#pragma unroll
      for (int i = 0; i < PL; ++i)
        if (i == bi) kt[i] = 0u;
      bt = 0u; bi = 0;
#pragma unroll
      for (int i = 0; i < PL; ++i)
        if (kt[i] > bt) { bt = kt[i]; bi = i; }
    }
  }
  return slot;
}

// Selection, radix variant: the C-th largest key by an MSB-first 4 x 8-bit radix select (warp-private shared-memory
// histograms), the ties at that key resolved to the lowest ids by ballot ranks, the C winners compacted and sorted
// (key descending, id ascending) by a 32-wide warp bitonic sort. No serial per-slot rounds.
template <int PL>
__device__ __forceinline__ int select_row_radix(const float* __restrict__ xrow, const float* __restrict__ bias,
                                                int E, int C, int lane) {
  __shared__ int hist_all[4][256];
  __shared__ unsigned long long ent_all[4][32];
  const int wv = (threadIdx.x >> 5) & 3;
  int* hist = hist_all[wv];
  unsigned long long* ent = ent_all[wv];
  uint32_t kt[PL];
#pragma unroll
  for (int i = 0; i < PL; ++i) {
    const int e = lane + 32 * i;
    kt[i] = (e < E) ? ord_key(__fadd_rn(sigmoid_ref(xrow[e]), bias[e])) : 0u;
  }
  uint32_t prefix = 0u, pmask = 0u;
  int need = C;
#pragma unroll
  for (int pass = 3; pass >= 0; --pass) {
    const int shift = 8 * pass;
#pragma unroll
    for (int q = 0; q < 8; ++q) hist[lane * 8 + q] = 0;
    __syncwarp();
#pragma unroll
    for (int i = 0; i < PL; ++i)
      if ((kt[i] & pmask) == prefix && (lane + 32 * i) < E) atomicAdd(&hist[(kt[i] >> shift) & 255u], 1);
    __syncwarp();
    // lane l owns bins 255-8l .. 248-8l (descending); suffix counts from the top
    int b[8];
    int lsum = 0;
#pragma unroll
    for (int q = 0; q < 8; ++q) { b[q] = hist[255 - lane * 8 - q]; lsum += b[q]; }
    int incl = lsum;  // inclusive prefix over lanes 0..lane (= bins from the top down to this lane's range)
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += y;
    }
    const int excl = incl - lsum;
    const unsigned found = __ballot_sync(0xffffffffu, excl < need && incl >= need);
    const int fl = __ffs(found) - 1;
    int digit = 0, above = 0;
    if (lane == fl) {
      int run = excl;
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        if (run + b[q] >= need) { digit = 255 - lane * 8 - q; above = run; break; }
        run += b[q];
      }
    }
    digit = __shfl_sync(0xffffffffu, digit, fl);
    above = __shfl_sync(0xffffffffu, above, fl);
    need -= above;
    prefix |= static_cast<uint32_t>(digit) << shift;
    pmask |= 255u << shift;
    __syncwarp();
  }
  // winners: key > prefix, plus the `need` lowest ids with key == prefix
  int base = 0;  // running count of earlier winners, in (i, lane) order
  int tie_base = 0;
  const unsigned lt = (1u << lane) - 1u;
#pragma unroll
  for (int i = 0; i < PL; ++i) {
    const bool valid = (lane + 32 * i) < E;
    const bool eq = valid && kt[i] == prefix;
    const unsigned beq = __ballot_sync(0xffffffffu, eq);
    const int trank = tie_base + __popc(beq & lt);
    tie_base += __popc(beq);
    const bool win = valid && (kt[i] > prefix || (eq && trank < need));
    const unsigned bw = __ballot_sync(0xffffffffu, win);
    if (win) {
      const int pos = base + __popc(bw & lt);
      ent[pos] = (static_cast<unsigned long long>(kt[i]) << 32) |
                 static_cast<unsigned long long>(0xFFFFFFFFu - static_cast<uint32_t>(lane + 32 * i));
    }
    base += __popc(bw);
  }
  __syncwarp();
  unsigned long long key = lane < C ? ent[lane] : 0ull;
  __syncwarp();
  // bitonic sort, descending
#pragma unroll
  for (int size = 2; size <= 32; size <<= 1) {
#pragma unroll
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      const unsigned long long other = __shfl_xor_sync(0xffffffffu, key, stride);
      const bool up = ((lane & size) == 0);           // descending block
      const bool lower = ((lane & stride) == 0);
      const bool take_max = (lower == up);
      key = take_max ? (other > key ? other : key) : (other < key ? other : key);
    }
  }
  return static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(key & 0xFFFFFFFFull));
}

// Per-row math after selection, run by one warp whose lane j holds the expert id of slot j (shared by the
// warp-per-row and block-per-row kernels, so both give the same bits).
template <bool RECIP, typename IdxT>
__device__ __forceinline__ void fwd_finish(const float* __restrict__ xrow, const float* __restrict__ bias, int row,
                                           int lane, int id, int k, int C, float m_or_inv, float sf,
                                           float* __restrict__ P, IdxT* __restrict__ IDX, bool* __restrict__ MASK,
                                           int* __restrict__ TPE, int* __restrict__ STOCK) {
  const float s = lane < C ? sigmoid_ref(xrow[id]) : 0.f;
  const float v = lane < C ? __fadd_rn(s, bias[id]) : 0.f;
  const float vk1 = __shfl_sync(0xffffffffu, v, k - 1);
  const float vk = __shfl_sync(0xffffffffu, v, k);
  const float tau = __fmul_rn(__fadd_rn(vk1, vk), 0.5f);
  const float d = __fsub_rn(v, tau);
  const float q = RECIP ? __fmul_rn(d, m_or_inv) : __fdiv_rn(d, m_or_inv);
  float c = __fadd_rn(q, 0.5f);
  c = c < 0.f ? 0.f : (c > 1.f ? 1.f : c);
  const float w = (lane < C) ? __fmul_rn(s, c) : 0.f;
  float wa[32];
#pragma unroll
  for (int j = 0; j < 32; ++j) wa[j] = __shfl_sync(0xffffffffu, w, j);
  float W = wa[0];
#pragma unroll
  for (int j = 1; j < 32; ++j)
    if (j < C) W = __fadd_rn(W, wa[j]);
  if (lane < C) {
    float p = __fdiv_rn(w, __fadd_rn(W, 1e-20f));
    if (sf != 0.f) p = __fmul_rn(p, sf);
    const long long o = static_cast<long long>(row) * C + lane;
    P[o] = p;
    IDX[o] = static_cast<IdxT>(id);
    if (MASK) MASK[o] = c > 0.f;
    if (TPE) atomicAdd(TPE + id, 1);
    if (STOCK && lane < k) atomicAdd(STOCK + id, 1);
  }
}

template <bool RECIP>
__device__ __forceinline__ void bwd_finish(const float* __restrict__ xrow, const float* __restrict__ bias,
                                           const float* __restrict__ G, int row, int lane, int id, int E, int k,
                                           int C, float m_or_inv, float sf, float* __restrict__ GX) {
  const float s = lane < C ? sigmoid_ref(xrow[id]) : 0.f;
  const float v = lane < C ? __fadd_rn(s, bias[id]) : 0.f;
  const float vk1 = __shfl_sync(0xffffffffu, v, k - 1);
  const float vk = __shfl_sync(0xffffffffu, v, k);
  const float tau = __fmul_rn(__fadd_rn(vk1, vk), 0.5f);
  const float inv_m = RECIP ? m_or_inv : 1.0f / m_or_inv;
  const float u = (v - tau) * inv_m + 0.5f;
  const float c = fminf(fmaxf(u, 0.f), 1.f);
  const bool act = lane < C;
  const float w = act ? s * c : 0.f;
  float W = 0.f;
  for (int j = 0; j < C; ++j) W += __shfl_sync(0xffffffffu, w, j);
  const float rho = sf != 0.f ? sf : 1.f;
  const float Wd = W + 1e-20f;
  const float g = act ? G[static_cast<long long>(row) * C + lane] : 0.f;
  float gw = g * w;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) gw += __shfl_xor_sync(0xffffffffu, gw, o);
  const float qv = gw / Wd;
  const float dw = act ? rho * (g - qv) / Wd : 0.f;
  float ds = dw * c;
  const float dc = dw * s;
  const float du = (act && u > 0.f && u < 1.f) ? dc * inv_m : 0.f;  // dv_j from u_j
  // dtau = -sum_j du_j (warp reduction); tau = (v_{k-1} + v_k) / 2
  float sdu = du;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sdu += __shfl_xor_sync(0xffffffffu, sdu, o);
  float dv = du;
  if (lane == k - 1 || lane == k) dv += -0.5f * sdu;
  ds += dv;  // v = s + b
  const float gx = ds * s * (1.f - s);
  if (act) GX[static_cast<long long>(row) * E + id] = gx;
}

template <int PL, bool RECIP, typename IdxT>
__global__ void __launch_bounds__(128) cf_forward_kernel(
    const float* __restrict__ X, const float* __restrict__ bias, int T, int E, int k, int C, float m_or_inv,
    float sf, float* __restrict__ P, IdxT* __restrict__ IDX, bool* __restrict__ MASK, int* __restrict__ TPE,
    int* __restrict__ STOCK, int sel) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (row >= T) return;
  const float* xrow = X + static_cast<long long>(row) * E;
  const int id = sel ? select_row_radix<PL>(xrow, bias, E, C, lane) : select_row<PL>(xrow, bias, E, C, lane);
  fwd_finish<RECIP, IdxT>(xrow, bias, row, lane, id, k, C, m_or_inv, sf, P, IDX, MASK, TPE, STOCK);
}

// d(sum_j g_j p_j)/dx (fp32): through p = rho w / (W + eps), w = s c, c = clamp(u), u = (v - tau)/m + .5,
// tau = (v_{k-1} + v_k)/2, v = s + b, s = sigmoid(x). One-sided at the clamp kinks (inactive at u <= 0, u >= 1).
template <int PL, bool RECIP>
__global__ void __launch_bounds__(128) cf_backward_kernel(
    const float* __restrict__ X, const float* __restrict__ bias, const float* __restrict__ G, int T, int E, int k,
    int C, float m_or_inv, float sf, float* __restrict__ GX, int sel) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (row >= T) return;
  const float* xrow = X + static_cast<long long>(row) * E;
  const int id = sel ? select_row_radix<PL>(xrow, bias, E, C, lane) : select_row<PL>(xrow, bias, E, C, lane);
  float* out = GX + static_cast<long long>(row) * E;
  for (int e = lane; e < E; e += 32) out[e] = 0.f;
  __syncwarp();
  bwd_finish<RECIP>(xrow, bias, G, row, lane, id, E, k, C, m_or_inv, sf, GX);
}

// Block-per-row selection (decode-size batches: per-row latency, not throughput, sets the time). 256 threads hold
// ceil(E / 256) keys each; an MSB-first 4 x 8-bit radix select over the block (shared histogram) finds the C-th key,
// the ties at it go to the lowest ids (block ballot ranks, id order = (i, thread) order), the C winners are
// compacted and sorted by a 32-wide bitonic sort in warp 0. Returns the slot's id in warp 0's lanes.
template <int PB>
__device__ __forceinline__ int select_block(const float* __restrict__ xrow, const float* __restrict__ bias, int E,
                                            int C) {
  __shared__ int hist[256];
  __shared__ int wsum[8];
  __shared__ int bc[2];
  __shared__ unsigned long long ent[32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  uint32_t kt[PB];
#pragma unroll
  for (int i = 0; i < PB; ++i) {
    const int e = tid + 256 * i;
    kt[i] = (e < E) ? ord_key(__fadd_rn(sigmoid_ref(xrow[e]), bias[e])) : 0u;
  }
  uint32_t prefix = 0u, pmask = 0u;
  int need = C;
#pragma unroll
  for (int pass = 3; pass >= 0; --pass) {
    const int shift = 8 * pass;
    hist[tid] = 0;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < PB; ++i)
      if ((tid + 256 * i) < E && (kt[i] & pmask) == prefix) atomicAdd(&hist[(kt[i] >> shift) & 255u], 1);
    __syncthreads();
    if (warp == 0) {
      int b[8];
      int lsum = 0;
#pragma unroll
      for (int q = 0; q < 8; ++q) { b[q] = hist[255 - lane * 8 - q]; lsum += b[q]; }
      int incl = lsum;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += y;
      }
      const int excl = incl - lsum;
      const unsigned found = __ballot_sync(0xffffffffu, excl < need && incl >= need);
      const int fl = __ffs(found) - 1;
      if (lane == fl) {
        int run = excl;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          if (run + b[q] >= need) { bc[0] = 255 - lane * 8 - q; bc[1] = run; break; }
          run += b[q];
        }
      }
    }
    __syncthreads();
    need -= bc[1];
    prefix |= static_cast<uint32_t>(bc[0]) << shift;
    pmask |= 255u << shift;
    __syncthreads();
  }
  const unsigned lt = (1u << lane) - 1u;
  int tie_base = 0, base = 0;
#pragma unroll
  for (int i = 0; i < PB; ++i) {
    const int e = tid + 256 * i;
    const bool valid = e < E;
    const bool eq = valid && kt[i] == prefix;
    const unsigned beq = __ballot_sync(0xffffffffu, eq);
    if (lane == 0) wsum[warp] = __popc(beq);
    __syncthreads();
    int eqoff = 0, eqtot = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) { const int c = wsum[w]; eqoff += w < warp ? c : 0; eqtot += c; }
    const int trank = tie_base + eqoff + __popc(beq & lt);
    tie_base += eqtot;
    const bool win = valid && (kt[i] > prefix || (eq && trank < need));
    const unsigned bw = __ballot_sync(0xffffffffu, win);
    __syncthreads();
    if (lane == 0) wsum[warp] = __popc(bw);
    __syncthreads();
    int woff = 0, wtot = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) { const int c = wsum[w]; woff += w < warp ? c : 0; wtot += c; }
    if (win) {
      ent[base + woff + __popc(bw & lt)] = (static_cast<unsigned long long>(kt[i]) << 32) |
                                           static_cast<unsigned long long>(0xFFFFFFFFu - static_cast<uint32_t>(e));
    }
    base += wtot;
    __syncthreads();
  }
  int id = 0;
  if (warp == 0) {
    unsigned long long key = lane < C ? ent[lane] : 0ull;
#pragma unroll
    for (int size = 2; size <= 32; size <<= 1) {
#pragma unroll
      for (int stride = size >> 1; stride > 0; stride >>= 1) {
        const unsigned long long other = __shfl_xor_sync(0xffffffffu, key, stride);
        const bool up = ((lane & size) == 0);
        const bool lower = ((lane & stride) == 0);
        key = (lower == up) ? (other > key ? other : key) : (other < key ? other : key);
      }
    }
    id = static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(key & 0xFFFFFFFFull));
  }
  return id;
}

template <int PB, bool RECIP, typename IdxT>
__global__ void __launch_bounds__(256) cf_forward_block_kernel(
    const float* __restrict__ X, const float* __restrict__ bias, int T, int E, int k, int C, float m_or_inv,
    float sf, float* __restrict__ P, IdxT* __restrict__ IDX, bool* __restrict__ MASK, int* __restrict__ TPE,
    int* __restrict__ STOCK) {
  const int row = blockIdx.x;
  const float* xrow = X + static_cast<long long>(row) * E;
  const int id = select_block<PB>(xrow, bias, E, C);
  if ((threadIdx.x >> 5) == 0)
    fwd_finish<RECIP, IdxT>(xrow, bias, row, threadIdx.x & 31, id, k, C, m_or_inv, sf, P, IDX, MASK, TPE, STOCK);
}

template <int PB, bool RECIP>
__global__ void __launch_bounds__(256) cf_backward_block_kernel(
    const float* __restrict__ X, const float* __restrict__ bias, const float* __restrict__ G, int T, int E, int k,
    int C, float m_or_inv, float sf, float* __restrict__ GX) {
  const int row = blockIdx.x;
  const float* xrow = X + static_cast<long long>(row) * E;
  const int id = select_block<PB>(xrow, bias, E, C);
  float* out = GX + static_cast<long long>(row) * E;
  for (int e = threadIdx.x; e < E; e += 256) out[e] = 0.f;
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) bwd_finish<RECIP>(xrow, bias, G, row, threadIdx.x & 31, id, E, k, C, m_or_inv, sf, GX);
}

// Profiling only (cf_phase): 0 = loads + sigmoid + keys, 1 = + selection.
template <int PL, int PHASE>
__global__ void __launch_bounds__(128) cf_phase_kernel(const float* __restrict__ X, const float* __restrict__ bias,
                                                       int T, int E, int C, int* __restrict__ OUT) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (row >= T) return;
  const float* xrow = X + static_cast<long long>(row) * E;
  if (PHASE == 0) {
    uint32_t acc = 0u;
#pragma unroll
    for (int i = 0; i < PL; ++i) {
      const int e = lane + 32 * i;
      if (e < E) acc ^= ord_key(__fadd_rn(sigmoid_ref(xrow[e]), bias[e]));
    }
    OUT[row * 32 + lane] = static_cast<int>(acc);
  } else if (PHASE == 1) {
    OUT[row * 32 + lane] = select_row<PL>(xrow, bias, E, C, lane);
  } else {
    OUT[row * 32 + lane] = select_row_radix<PL>(xrow, bias, E, C, lane);
  }
}

int pl_for(int E) { return E <= 32 ? 1 : E <= 64 ? 2 : E <= 128 ? 4 : E <= 256 ? 8 : 16; }

}  // namespace

void cf_forward(torch::Tensor x, torch::Tensor bias, int64_t k, int64_t C, double m_or_inv, bool recip, double sf,
                torch::Tensor probs, torch::Tensor idx, c10::optional<torch::Tensor> mask,
                c10::optional<torch::Tensor> tpe, c10::optional<torch::Tensor> stock, int64_t sel) {
  const int T = x.size(0), E = x.size(1);
  cudaStream_t st = c10::cuda::getCurrentCUDAStream(x.device().index()).stream();
  if (tpe) cudaMemsetAsync(tpe->data_ptr(), 0, sizeof(int) * E, st);
  if (stock) cudaMemsetAsync(stock->data_ptr(), 0, sizeof(int) * E, st);
  if (T == 0) return;
  const int threads = 128;
  const int blocks = (T * 32 + threads - 1) / threads;
  bool* mp = mask ? mask->data_ptr<bool>() : nullptr;
  int* tp = tpe ? tpe->data_ptr<int>() : nullptr;
  int* sp = stock ? stock->data_ptr<int>() : nullptr;
  const float mi = static_cast<float>(m_or_inv), sff = static_cast<float>(sf);
#define CF_LAUNCH(PLV, RV, IT)                                                                                  \
  cf_forward_kernel<PLV, RV, IT><<<blocks, threads, 0, st>>>(x.data_ptr<float>(), bias.data_ptr<float>(), T, E, \
      static_cast<int>(k), static_cast<int>(C), mi, sff, probs.data_ptr<float>(),                               \
      reinterpret_cast<IT*>(idx.data_ptr()), mp, tp, sp, static_cast<int>(sel))
#define CF_PL(RV, IT)                                    \
  switch (pl_for(E)) {                                   \
    case 1: CF_LAUNCH(1, RV, IT); break;                 \
    case 2: CF_LAUNCH(2, RV, IT); break;                 \
    case 4: CF_LAUNCH(4, RV, IT); break;                 \
    case 8: CF_LAUNCH(8, RV, IT); break;                 \
    default: CF_LAUNCH(16, RV, IT); break;               \
  }
  const bool i64 = idx.scalar_type() == torch::kInt64;
  if (sel == 2) {
#define CFB(PBV, RV, IT)                                                                                         \
  cf_forward_block_kernel<PBV, RV, IT><<<T, 256, 0, st>>>(x.data_ptr<float>(), bias.data_ptr<float>(), T, E,     \
      static_cast<int>(k), static_cast<int>(C), mi, sff, probs.data_ptr<float>(),                                 \
      reinterpret_cast<IT*>(idx.data_ptr()), mp, tp, sp)
#define CFB_PB(RV, IT) if (E <= 256) { CFB(1, RV, IT); } else { CFB(2, RV, IT); }
    if (recip) { if (i64) { CFB_PB(true, int64_t) } else { CFB_PB(true, int32_t) } }
    else { if (i64) { CFB_PB(false, int64_t) } else { CFB_PB(false, int32_t) } }
#undef CFB_PB
#undef CFB
    return;
  }
  if (recip) { if (i64) { CF_PL(true, int64_t) } else { CF_PL(true, int32_t) } }
  else { if (i64) { CF_PL(false, int64_t) } else { CF_PL(false, int32_t) } }
#undef CF_PL
#undef CF_LAUNCH
}

void cf_backward(torch::Tensor x, torch::Tensor bias, torch::Tensor g, int64_t k, int64_t C, double m_or_inv,
                 bool recip, double sf, torch::Tensor gx, int64_t sel) {
  const int T = x.size(0), E = x.size(1);
  if (T == 0) return;
  cudaStream_t st = c10::cuda::getCurrentCUDAStream(x.device().index()).stream();
  const int threads = 128;
  const int blocks = (T * 32 + threads - 1) / threads;
  const float mi = static_cast<float>(m_or_inv), sff = static_cast<float>(sf);
#define CB_LAUNCH(PLV, RV)                                                                                      \
  cf_backward_kernel<PLV, RV><<<blocks, threads, 0, st>>>(x.data_ptr<float>(), bias.data_ptr<float>(),         \
      g.data_ptr<float>(), T, E, static_cast<int>(k), static_cast<int>(C), mi, sff, gx.data_ptr<float>(),         \
      static_cast<int>(sel))
#define CB_PL(RV)                                      \
  switch (pl_for(E)) {                                 \
    case 1: CB_LAUNCH(1, RV); break;                   \
    case 2: CB_LAUNCH(2, RV); break;                   \
    case 4: CB_LAUNCH(4, RV); break;                   \
    case 8: CB_LAUNCH(8, RV); break;                   \
    default: CB_LAUNCH(16, RV); break;                 \
  }
  if (sel == 2) {
#define CBB(PBV, RV)                                                                                             \
  cf_backward_block_kernel<PBV, RV><<<T, 256, 0, st>>>(x.data_ptr<float>(), bias.data_ptr<float>(),             \
      g.data_ptr<float>(), T, E, static_cast<int>(k), static_cast<int>(C), mi, sff, gx.data_ptr<float>())
    if (E <= 256) { if (recip) { CBB(1, true); } else { CBB(1, false); } }
    else { if (recip) { CBB(2, true); } else { CBB(2, false); } }
#undef CBB
    return;
  }
  if (recip) { CB_PL(true) } else { CB_PL(false) }
#undef CB_PL
#undef CB_LAUNCH
}

void cf_phase(torch::Tensor x, torch::Tensor bias, int64_t C, int64_t phase, torch::Tensor out) {
  const int T = x.size(0), E = x.size(1);
  cudaStream_t st = c10::cuda::getCurrentCUDAStream(x.device().index()).stream();
  const int blocks = (T * 32 + 127) / 128;
#define CP(PLV)                                                                                                  \
  if (phase == 0) cf_phase_kernel<PLV, 0><<<blocks, 128, 0, st>>>(x.data_ptr<float>(), bias.data_ptr<float>(), T, \
      E, static_cast<int>(C), out.data_ptr<int>());                                                               \
  else if (phase == 1) cf_phase_kernel<PLV, 1><<<blocks, 128, 0, st>>>(x.data_ptr<float>(), bias.data_ptr<float>(), \
      T, E, static_cast<int>(C), out.data_ptr<int>());                                                            \
  else cf_phase_kernel<PLV, 2><<<blocks, 128, 0, st>>>(x.data_ptr<float>(), bias.data_ptr<float>(), T, E,         \
      static_cast<int>(C), out.data_ptr<int>());
  switch (pl_for(E)) { case 4: CP(4) break; case 8: CP(8) break; default: CP(16) break; }
#undef CP
}
