// PoI v2 (sd-turbo-int) device kernels. Bit-exact mirrors of contrib/poi-sd/sdint.py.
#include "kernels.h"
#include "intops.cuh"
#include <climits>
#include <stdexcept>

namespace meow {
namespace {
constexpr int NT = 256;

__device__ __forceinline__ i64 warp_sum(i64 v) { for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o); return v; }
__device__ __forceinline__ int warp_max(int v) { for (int o = 16; o; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, o)); return v; }
__device__ __forceinline__ int warp_min(int v) { for (int o = 16; o; o >>= 1) v = min(v, __shfl_xor_sync(0xffffffffu, v, o)); return v; }
__device__ __forceinline__ u32 warp_umax(u32 v) { for (int o = 16; o; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, o)); return v; }
/** Block-wide reductions (blockDim.x <= 1024, multiple of 32). `sh` is 32 x 8 bytes of shared memory. Result broadcast to all threads. */
template <typename T, typename Op> __device__ __forceinline__ T block_reduce(T v, Op warp_op, T ident, T* sh) {
    v = warp_op(v);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    v = (threadIdx.x < nw) ? sh[threadIdx.x] : ident;
    if (wid == 0) v = warp_op(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    v = sh[0];
    __syncthreads();
    return v;
}
struct SumOp { __device__ i64 operator()(i64 v) const { return warp_sum(v); } };
struct MaxOp { __device__ int operator()(int v) const { return warp_max(v); } };
struct MinOp { __device__ int operator()(int v) const { return warp_min(v); } };
struct UMaxOp { __device__ u32 operator()(u32 v) const { return warp_umax(v); } };

__device__ __forceinline__ u32 iabs32(int v) { return v < 0 ? u32(-i64(v)) : u32(v); }
__device__ __forceinline__ int epi1v(int chi, int clo, const int4 c, int er) {
    i64 acc = 256LL * chi + clo + 128LL * c.w;
    return int(sat32(rshr(acc * c.x, c.y - er) + c.z));
}

// ---------------------------------------------------------------- quantisation of GEMM A operands
__global__ void quant_rows_k(const int32_t* __restrict__ X, int M, int K, int ldx, int8_t* __restrict__ hi, int8_t* __restrict__ lo, int32_t* __restrict__ e, int Kp) {
    __shared__ i64 sh[32];
    int r = blockIdx.x;
    int8_t* h = hi + size_t(r) * Kp; int8_t* l = lo + size_t(r) * Kp;
    if (r >= M) { for (int k = threadIdx.x; k < Kp; k += blockDim.x) { h[k] = 0; l[k] = -128; } return; }
    const int32_t* x = X + size_t(r) * ldx;
    u32 m = 0;
    for (int k = threadIdx.x; k < K; k += blockDim.x) m = max(m, iabs32(x[k]));
    m = block_reduce<u32>(m, UMaxOp(), 0u, (u32*)sh);
    int ee = row_exp(m);
    if (threadIdx.x == 0) e[r] = ee;
    for (int k = threadIdx.x; k < Kp; k += blockDim.x) {
        int q = k < K ? quant16_32(x[k], ee) : 0;
        h[k] = int8_t(q >> 8); l[k] = int8_t((q & 255) - 128);
    }
}

__device__ __forceinline__ int epi1(const int32_t* __restrict__ Chi, const int32_t* __restrict__ Clo, size_t ci, const int4 c, int er) {
    i64 acc = 256LL * Chi[ci] + Clo[ci] + 128LL * c.w;
    return int(sat32(rshr(acc * c.x, c.y - er) + c.z));
}
// 4 consecutive columns per thread (N % 4 == 0, ldc/ldy % 4 == 0): one int4 load per plane, one int4 store.
__global__ void epilogue_k(const int32_t* __restrict__ Chi, const int32_t* __restrict__ Clo, int M, int N, int ldc, const int4* __restrict__ cst,
                           const int32_t* __restrict__ e, int32_t* __restrict__ Y, int ldy, const int32_t* R) {
    const int tpr = N >> 2;
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= size_t(M) * tpr) return;
    const int r = int(idx / tpr), n = int(idx - size_t(r) * tpr) * 4;
    const int er = e[r];
    const int4 h = *reinterpret_cast<const int4*>(Chi + size_t(r) * ldc + n), l = *reinterpret_cast<const int4*>(Clo + size_t(r) * ldc + n);
    int4 y; y.x = epi1v(h.x, l.x, cst[n], er); y.y = epi1v(h.y, l.y, cst[n + 1], er); y.z = epi1v(h.z, l.z, cst[n + 2], er); y.w = epi1v(h.w, l.w, cst[n + 3], er);
    if (R) { const int4 rr = *reinterpret_cast<const int4*>(R + size_t(r) * ldy + n); y.x = int(sat32(i64(y.x) + rr.x)); y.y = int(sat32(i64(y.y) + rr.y)); y.z = int(sat32(i64(y.z) + rr.z)); y.w = int(sat32(i64(y.w) + rr.w)); }
    *reinterpret_cast<int4*>(Y + size_t(r) * ldy + n) = y;
}

// Epilogue of a merged GEMM whose N = parts*Cs: column n -> Y[n / Cs][row][n % Cs]
__global__ void epilogue_split_k(const int32_t* __restrict__ Chi, const int32_t* __restrict__ Clo, int M, int N, int ldc, const int4* __restrict__ cst,
                                 const int32_t* __restrict__ e, int Cs, int32_t* __restrict__ Y0, int32_t* __restrict__ Y1, int32_t* __restrict__ Y2) {
    int n = blockIdx.x * blockDim.x + threadIdx.x; if (n >= N) return;
    int part = n >= 2 * Cs ? 2 : (n >= Cs ? 1 : 0), c = n - part * Cs;
    int32_t* Y = part == 0 ? Y0 : (part == 1 ? Y1 : Y2); const int4 cc = cst[n];
    for (int r = blockIdx.y; r < M; r += gridDim.y) Y[size_t(r) * Cs + c] = epi1v(Chi[size_t(r) * ldc + n], Clo[size_t(r) * ldc + n], cc, e[r]);
}

__global__ void pixmax_k(const int32_t* __restrict__ X, int rows, int C, u32* __restrict__ pm) {
    // one warp per row
    int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (w >= rows) return;
    const int32_t* x = X + size_t(w) * C;
    u32 m = 0;
    for (int c = lane; c < C; c += 32) m = max(m, iabs32(x[c]));
    m = warp_umax(m);
    if (lane == 0) pm[w] = m;
}

// K index is tap-major: k = tap*C + c, tap = ky*3+kx. (Weights are permuted to match at load time.)
__global__ void im2col3_k(const int32_t* __restrict__ X, const u32* __restrict__ pm, int B, int H, int W, int C, int stride,
                          int8_t* __restrict__ hi, int8_t* __restrict__ lo, int32_t* __restrict__ e, int Kp, int Ho, int Wo) {
    __shared__ int she;
    int r = blockIdx.x;                         // output row = b*Ho*Wo + oy*Wo + ox
    int b = r / (Ho * Wo), p = r % (Ho * Wo), oy = p / Wo, ox = p % Wo;
    int iy0 = oy * stride - 1, ix0 = ox * stride - 1;
    if (threadIdx.x == 0) {
        u32 m = 0;
        for (int ky = 0; ky < 3; ++ky) for (int kx = 0; kx < 3; ++kx) {
            int iy = iy0 + ky, ix = ix0 + kx;
            if (iy >= 0 && iy < H && ix >= 0 && ix < W) m = max(m, pm[(size_t(b) * H + iy) * W + ix]);
        }
        she = row_exp(m); e[r] = she;
    }
    __syncthreads();
    int ee = she; int K = 9 * C;
    int8_t* h = hi + size_t(r) * Kp; int8_t* l = lo + size_t(r) * Kp;
    for (int tap = 0; tap < 9; ++tap) {
        int ky = tap / 3, kx = tap - ky * 3, iy = iy0 + ky, ix = ix0 + kx;
        bool in = iy >= 0 && iy < H && ix >= 0 && ix < W;
        const int32_t* xp = in ? X + ((size_t(b) * H + iy) * W + ix) * C : nullptr;
        int8_t* ht = h + tap * C; int8_t* lt = l + tap * C;
        for (int c4 = threadIdx.x; c4 < (C >> 2); c4 += blockDim.x) {
            int4 xv = in ? *reinterpret_cast<const int4*>(xp + c4 * 4) : make_int4(0, 0, 0, 0);
            int q0 = in ? quant16_32(xv.x, ee) : 0, q1 = in ? quant16_32(xv.y, ee) : 0, q2 = in ? quant16_32(xv.z, ee) : 0, q3 = in ? quant16_32(xv.w, ee) : 0;
            u32 hv = (u32(uint8_t(q0 >> 8))) | (u32(uint8_t(q1 >> 8)) << 8) | (u32(uint8_t(q2 >> 8)) << 16) | (u32(uint8_t(q3 >> 8)) << 24);
            u32 lv = (u32(uint8_t((q0 & 255) - 128))) | (u32(uint8_t((q1 & 255) - 128)) << 8) | (u32(uint8_t((q2 & 255) - 128)) << 16) | (u32(uint8_t((q3 & 255) - 128)) << 24);
            *reinterpret_cast<u32*>(ht + c4 * 4) = hv; *reinterpret_cast<u32*>(lt + c4 * 4) = lv;
        }
    }
    for (int k = K + threadIdx.x; k < Kp; k += blockDim.x) { h[k] = 0; l[k] = -128; }
}

// ---------------------------------------------------------------- GroupNorm (NHWC, stats via atomics)
__global__ void gn_init_k(GnStat* gs, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { gs[i].sum = 0; gs[i].mn = INT_MAX; gs[i].mx = INT_MIN; gs[i].ss = 0; } }
// grid (groups, B, splits)
__global__ void gn_stats1_k(const int32_t* __restrict__ X, int T, int C, int groups, GnStat* gs) {
    __shared__ i64 sh[32];
    int g = blockIdx.x, b = blockIdx.y, cg = C / groups, n = T * cg;
    int splits = gridDim.z, per = (n + splits - 1) / splits, i0 = blockIdx.z * per, i1 = min(n, i0 + per);
    const int32_t* x = X + size_t(b) * T * C + g * cg;
    i64 sum = 0; int mn = INT_MAX, mx = INT_MIN;
    for (int i = i0 + threadIdx.x; i < i1; i += blockDim.x) { int v = x[size_t(i / cg) * C + i % cg]; sum += v; mn = min(mn, v); mx = max(mx, v); }
    sum = block_reduce<i64>(sum, SumOp(), 0, sh); mn = block_reduce<int>(mn, MinOp(), INT_MAX, (int*)sh); mx = block_reduce<int>(mx, MaxOp(), INT_MIN, (int*)sh);
    if (threadIdx.x == 0) { GnStat* s = gs + b * groups + g; atomicAdd((unsigned long long*)&s->sum, (unsigned long long)sum); atomicMin(&s->mn, mn); atomicMax(&s->mx, mx); }
}
__global__ void gn_fin1_k(GnStat* gs, int cnt, i64 n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= cnt) return;
    GnStat s = gs[i];
    s.mean = floor_div(s.sum + n / 2, n);
    i64 m = max(s.mx - s.mean, s.mean - s.mn);
    s.e = max(0, row_exp(m) - 5);
    gs[i] = s;
}
__global__ void gn_stats2_k(const int32_t* __restrict__ X, int T, int C, int groups, GnStat* gs) {
    __shared__ i64 sh[32];
    int g = blockIdx.x, b = blockIdx.y, cg = C / groups, n = T * cg;
    int splits = gridDim.z, per = (n + splits - 1) / splits, i0 = blockIdx.z * per, i1 = min(n, i0 + per);
    const int32_t* x = X + size_t(b) * T * C + g * cg;
    GnStat* s = gs + b * groups + g; i64 mean = s->mean; int e = s->e;
    i64 ss = 0;
    for (int i = i0 + threadIdx.x; i < i1; i += blockDim.x) { i64 dd = rshr(i64(x[size_t(i / cg) * C + i % cg]) - mean, e); ss += dd * dd; }
    ss = block_reduce<i64>(ss, SumOp(), 0, sh);
    if (threadIdx.x == 0) atomicAdd((unsigned long long*)&s->ss, (unsigned long long)ss);
}
__global__ void gn_fin2_k(GnStat* gs, int cnt, i64 n, i64 den) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= cnt) return;
    GnStat s = gs[i];
    i64 var = floor_div(s.ss, n);
    int sh = 32 - 2 * s.e; i64 one = sh >= 0 ? (1LL << sh) : 0;
    i64 eps = floor_div(one + den / 2, den);
    s.sd = isqrt_i64((var + eps) << 16);
    gs[i] = s;
}
__global__ void gn_apply_k(const int32_t* __restrict__ X, int B, int T, int C, int groups, const GnStat* __restrict__ gs, const i64* __restrict__ g, const i64* __restrict__ beta, int32_t* __restrict__ Y) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; size_t tot = size_t(B) * T * C;
    if (i >= tot) return;
    int c = int(i % C); int b = int(i / (size_t(T) * C)); int cg = C / groups; int grp = c / cg;
    GnStat s = gs[b * groups + grp];
    i64 d = i64(X[i]) - s.mean;
    i64 y = floor_div(d * (1LL << (24 - s.e)) * 2 + s.sd, 2 * s.sd);
    Y[i] = int32_t(sat32(rshr(y * g[c], 16) + beta[c]));
}

// ---------------------------------------------------------------- LayerNorm (row in registers)
__global__ void lnorm_k(const int32_t* __restrict__ X, int C, const i64* __restrict__ g, const i64* __restrict__ beta, i64 den, int32_t* __restrict__ Y) {
    __shared__ i64 sh[32];
    const int32_t* x = X + size_t(blockIdx.x) * C; int32_t* y = Y + size_t(blockIdx.x) * C;
    int v[8]; int cnt = 0;
    i64 sum = 0; int mn = INT_MAX, mx = INT_MIN;
    for (int c = threadIdx.x; c < C; c += NT) { int t = x[c]; v[cnt++] = t; sum += t; mn = min(mn, t); mx = max(mx, t); }
    sum = block_reduce<i64>(sum, SumOp(), 0, sh); mn = block_reduce<int>(mn, MinOp(), INT_MAX, (int*)sh); mx = block_reduce<int>(mx, MaxOp(), INT_MIN, (int*)sh);
    i64 n = C, mean = floor_div(sum + n / 2, n);
    int e = max(0, row_exp(max(mx - mean, mean - mn)) - 5);
    i64 ss = 0;
    for (int j = 0; j < cnt; ++j) { i64 dd = rshr(i64(v[j]) - mean, e); ss += dd * dd; }
    ss = block_reduce<i64>(ss, SumOp(), 0, sh);
    i64 var = floor_div(ss, n);
    int shb = 32 - 2 * e; i64 one = shb >= 0 ? (1LL << shb) : 0;
    i64 eps = floor_div(one + den / 2, den);
    i64 sd = isqrt_i64((var + eps) << 16);
    cnt = 0;
    for (int c = threadIdx.x; c < C; c += NT) {
        i64 d = i64(v[cnt++]) - mean;
        i64 yy = floor_div(d * (1LL << (24 - e)) * 2 + sd, 2 * sd);
        y[c] = int32_t(sat32(rshr(yy * g[c], 16) + beta[c]));
    }
}

// ---------------------------------------------------------------- fused row kernels with quantised output
// Row of C values held in registers (R per thread): max|v| -> e -> hi/lo.
template <int R> __device__ __forceinline__ void quant_row_out(const int* v, int cnt, int C, int8_t* h, int8_t* l, int32_t* e, int Kp, i64* sh) {
    u32 m = 0;
    for (int j = 0; j < cnt; ++j) m = max(m, iabs32(v[j]));
    m = block_reduce<u32>(m, UMaxOp(), 0u, (u32*)sh);
    int ee = row_exp(m);
    if (threadIdx.x == 0) *e = ee;
    int j = 0;
    for (int c = threadIdx.x; c < C; c += NT, ++j) { int q = quant16_32(v[j], ee); h[c] = int8_t(q >> 8); l[c] = int8_t((q & 255) - 128); }
    for (int c = C + threadIdx.x; c < Kp; c += NT) { h[c] = 0; l[c] = -128; }
}
template <int R>
__global__ void lnorm_q_k(const int32_t* __restrict__ X, int C, const int32_t* __restrict__ g, const int32_t* __restrict__ beta, i64 den, int8_t* __restrict__ hi, int8_t* __restrict__ lo, int32_t* __restrict__ e, int Kp) {
    __shared__ i64 sh[32];
    const int32_t* x = X + size_t(blockIdx.x) * C;
    int v[R]; int cnt = 0;
    i64 sum = 0; int mn = INT_MAX, mx = INT_MIN;
    for (int c = threadIdx.x; c < C; c += NT) { int t = x[c]; v[cnt++] = t; sum += t; mn = min(mn, t); mx = max(mx, t); }
    sum = block_reduce<i64>(sum, SumOp(), 0, sh); mn = block_reduce<int>(mn, MinOp(), INT_MAX, (int*)sh); mx = block_reduce<int>(mx, MaxOp(), INT_MIN, (int*)sh);
    i64 n = C, mean = floor_div(sum + n / 2, n);
    int ex = max(0, row_exp(max(mx - mean, mean - mn)) - 5);
    i64 ss = 0;
    for (int j = 0; j < cnt; ++j) { i64 dd = rshr(i64(v[j]) - mean, ex); ss += dd * dd; }
    ss = block_reduce<i64>(ss, SumOp(), 0, sh);
    i64 var = floor_div(ss, n);
    int shb = 32 - 2 * ex; i64 one = shb >= 0 ? (1LL << shb) : 0;
    i64 eps = floor_div(one + den / 2, den);
    i64 sd = isqrt_i64((var + eps) << 16);
    cnt = 0;
    for (int c = threadIdx.x; c < C; c += NT, ++cnt) {
        i64 d = i64(v[cnt]) - mean;
        i64 yy = floor_div(d * (1LL << (24 - ex)) * 2 + sd, 2 * sd);
        v[cnt] = int(sat32(rshr(yy * g[c], 16) + beta[c]));
    }
    quant_row_out<R>(v, cnt, C, hi + size_t(blockIdx.x) * Kp, lo + size_t(blockIdx.x) * Kp, e + blockIdx.x, Kp, sh);
}
template <int R>
__global__ void gn_apply_row_k(const int32_t* __restrict__ X, int T, int C, int groups, const GnStat* __restrict__ gs, const int32_t* __restrict__ g, const int32_t* __restrict__ beta, const int32_t* __restrict__ Lsig,
                               int32_t* __restrict__ Y, u32* __restrict__ pm, int8_t* __restrict__ hi, int8_t* __restrict__ lo, int32_t* __restrict__ e, int Kp) {
    __shared__ i64 sh[32];
    size_t row = blockIdx.x; int b = int(row / T); int cg = C / groups;
    const int32_t* x = X + row * C;
    int v[R]; int cnt = 0; u32 m = 0;
    for (int c = threadIdx.x; c < C; c += NT, ++cnt) {
        GnStat st = gs[b * groups + c / cg];
        i64 d = i64(x[c]) - st.mean;
        i64 yy = floor_div(d * (1LL << (24 - st.e)) * 2 + st.sd, 2 * st.sd);
        i64 y = sat32(rshr(yy * g[c], 16) + beta[c]);
        if (Lsig) y = sat32(rshr(y * lut16i(y, Lsig), 16));
        v[cnt] = int(y); m = max(m, iabs32(int(y)));
    }
    if (hi) { quant_row_out<R>(v, cnt, C, hi + row * Kp, lo + row * Kp, e + row, Kp, sh); return; }
    int32_t* y = Y + row * C; cnt = 0;
    for (int c = threadIdx.x; c < C; c += NT) y[c] = v[cnt++];
    m = block_reduce<u32>(m, UMaxOp(), 0u, (u32*)sh);
    if (threadIdx.x == 0) pm[row] = m;
}
template <int R>
__global__ void geglu_q_k(const int32_t* __restrict__ A, const int32_t* __restrict__ G, int C, const int32_t* __restrict__ L, int8_t* __restrict__ hi, int8_t* __restrict__ lo, int32_t* __restrict__ e, int Kp) {
    __shared__ i64 sh[32];
    size_t row = blockIdx.x; const int32_t* a = A + row * C; const int32_t* gg = G + row * C;
    int v[R]; int cnt = 0;
    for (int c = threadIdx.x; c < C; c += NT, ++cnt) { i64 x = gg[c]; i64 ge = sat32(rshr(x * lut16i(x, L), 16)); v[cnt] = int(sat32(rshr(i64(a[c]) * ge, 16))); }
    quant_row_out<R>(v, cnt, C, hi + row * Kp, lo + row * Kp, e + row, Kp, sh);
}

// Warp-per-row variants: lane handles 4-column chunks c4 = lane + 32*j (columns 4*c4 .. 4*c4+3) -> int4 loads, u32 byte stores.
// No shared memory, no block barriers. 8 rows per block. R4 = chunks per lane (C <= 128*R4).
template <int R4> __device__ __forceinline__ void quant_row_out_w4(const int* v, int cnt, int8_t* h, int8_t* l, int32_t* e, int C, int Kp) {
    const int lane = threadIdx.x & 31;
    u32 m = 0;
    for (int j = 0; j < 4 * cnt; ++j) m = max(m, iabs32(v[j]));
    m = warp_umax(m);
    const int ee = row_exp(m);
    if (lane == 0) *e = ee;
    for (int j = 0; j < cnt; ++j) {
        const int c = (lane + 32 * j) * 4;
        int q0 = quant16_32(v[4 * j], ee), q1 = quant16_32(v[4 * j + 1], ee), q2 = quant16_32(v[4 * j + 2], ee), q3 = quant16_32(v[4 * j + 3], ee);
        *reinterpret_cast<u32*>(h + c) = u32(uint8_t(q0 >> 8)) | (u32(uint8_t(q1 >> 8)) << 8) | (u32(uint8_t(q2 >> 8)) << 16) | (u32(uint8_t(q3 >> 8)) << 24);
        *reinterpret_cast<u32*>(l + c) = u32(uint8_t((q0 & 255) - 128)) | (u32(uint8_t((q1 & 255) - 128)) << 8) | (u32(uint8_t((q2 & 255) - 128)) << 16) | (u32(uint8_t((q3 & 255) - 128)) << 24);
    }
    for (int c = C + lane; c < Kp; c += 32) { h[c] = 0; l[c] = -128; }
}
template <int R4>
__global__ void lnorm_qw_k(const int32_t* __restrict__ X, int M, int C, const int32_t* __restrict__ g, const int32_t* __restrict__ beta, i64 den, int8_t* __restrict__ hi, int8_t* __restrict__ lo, int32_t* __restrict__ e, int Kp) {
    const int lane = threadIdx.x & 31; const size_t row = size_t(blockIdx.x) * 8 + (threadIdx.x >> 5);
    if (row >= size_t(M)) return;
    const int32_t* x = X + row * C; const int C4 = C >> 2;
    int v[4 * R4]; int cnt = 0;
    i64 sum = 0; int mn = INT_MAX, mx = INT_MIN;
    for (int c4 = lane; c4 < C4; c4 += 32, ++cnt) {
        int4 t = *reinterpret_cast<const int4*>(x + c4 * 4);
        v[4 * cnt] = t.x; v[4 * cnt + 1] = t.y; v[4 * cnt + 2] = t.z; v[4 * cnt + 3] = t.w;
        sum += i64(t.x) + t.y + t.z + t.w; mn = min(mn, min(min(t.x, t.y), min(t.z, t.w))); mx = max(mx, max(max(t.x, t.y), max(t.z, t.w)));
    }
    sum = warp_sum(sum); mn = warp_min(mn); mx = warp_max(mx);
    const i64 n = C, mean = floor_div(sum + n / 2, n);
    const int ex = max(0, row_exp(max(mx - mean, mean - mn)) - 5);
    i64 ss = 0;
    for (int j = 0; j < 4 * cnt; ++j) { i64 dd = rshr(i64(v[j]) - mean, ex); ss += dd * dd; }
    ss = warp_sum(ss);
    const i64 var = floor_div(ss, n);
    const int shb = 32 - 2 * ex; const i64 one = shb >= 0 ? (1LL << shb) : 0;
    const i64 eps = floor_div(one + den / 2, den);
    const i64 sd = isqrt_i64((var + eps) << 16), sd2 = 2 * sd; const i64 sc = 1LL << (24 - ex);
    for (int j = 0; j < cnt; ++j) {
        const int c = (lane + 32 * j) * 4;
        const int4 gg = *reinterpret_cast<const int4*>(g + c), bb = *reinterpret_cast<const int4*>(beta + c);
        const int gv[4] = {gg.x, gg.y, gg.z, gg.w}, bv[4] = {bb.x, bb.y, bb.z, bb.w};
        #pragma unroll
        for (int k = 0; k < 4; ++k) { i64 d = i64(v[4 * j + k]) - mean; i64 yy = floor_div(d * sc * 2 + sd, sd2); v[4 * j + k] = int(sat32(rshr(yy * gv[k], 16) + bv[k])); }
    }
    quant_row_out_w4<R4>(v, cnt, hi + row * Kp, lo + row * Kp, e + row, C, Kp);
}
template <int R4>
__global__ void gn_apply_roww_k(const int32_t* __restrict__ X, size_t M, int T, int C, int groups, const GnStat* __restrict__ gs, const int32_t* __restrict__ g, const int32_t* __restrict__ beta, const int32_t* __restrict__ Lsig,
                                int32_t* __restrict__ Y, u32* __restrict__ pm, int8_t* __restrict__ hi, int8_t* __restrict__ lo, int32_t* __restrict__ e, int Kp) {
    __shared__ GnStat sgs[8][32];
    const int lane = threadIdx.x & 31; const int wrow = threadIdx.x >> 5; const size_t row = size_t(blockIdx.x) * 8 + wrow;
    const int b = int(row / T); const int cg = C / groups; const float icg = 1.0f / float(cg);
    if (row < M) sgs[wrow][lane] = gs[b * groups + lane];
    __syncwarp();
    if (row >= M) return;
    const int32_t* x = X + row * C; const int C4 = C >> 2;
    int v[4 * R4]; int cnt = 0; u32 m = 0;
    for (int c4 = lane; c4 < C4; c4 += 32, ++cnt) {
        const int c = c4 * 4;
        const int4 t = *reinterpret_cast<const int4*>(x + c), gg = *reinterpret_cast<const int4*>(g + c), bb = *reinterpret_cast<const int4*>(beta + c);
        const int xv[4] = {t.x, t.y, t.z, t.w}, gv[4] = {gg.x, gg.y, gg.z, gg.w}, bv[4] = {bb.x, bb.y, bb.z, bb.w};
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const GnStat& st = sgs[wrow][int((float(c + k) + 0.5f) * icg)];
            i64 d = i64(xv[k]) - st.mean;
            i64 yy = floor_div(d * (1LL << (24 - st.e)) * 2 + st.sd, 2 * st.sd);
            i64 y = sat32(rshr(yy * gv[k], 16) + bv[k]);
            if (Lsig) y = sat32(rshr(y * lut16i(y, Lsig), 16));
            v[4 * cnt + k] = int(y); m = max(m, iabs32(int(y)));
        }
    }
    if (hi) { quant_row_out_w4<R4>(v, cnt, hi + row * Kp, lo + row * Kp, e + row, C, Kp); return; }
    int32_t* y = Y + row * C;
    for (int j = 0; j < cnt; ++j) *reinterpret_cast<int4*>(y + (lane + 32 * j) * 4) = make_int4(v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3]);
    m = warp_umax(m);
    if (lane == 0) pm[row] = m;
}
template <int R4>
__global__ void quant_rows_w_k(const int32_t* __restrict__ X, int M, int K, int ldx, int8_t* __restrict__ hi, int8_t* __restrict__ lo, int32_t* __restrict__ e, int Kp) {
    const int lane = threadIdx.x & 31; const size_t r = size_t(blockIdx.x) * 8 + (threadIdx.x >> 5);
    int8_t* h = hi + r * Kp; int8_t* l = lo + r * Kp;
    if (r >= size_t(M)) { if (r < size_t((M + 31) / 32 * 32)) for (int k = lane; k < Kp; k += 32) { h[k] = 0; l[k] = -128; } return; }
    const int32_t* x = X + r * ldx; const int K4 = K >> 2;
    int v[4 * R4]; int cnt = 0;
    for (int c4 = lane; c4 < K4; c4 += 32, ++cnt) { int4 t = *reinterpret_cast<const int4*>(x + c4 * 4); v[4 * cnt] = t.x; v[4 * cnt + 1] = t.y; v[4 * cnt + 2] = t.z; v[4 * cnt + 3] = t.w; }
    quant_row_out_w4<R4>(v, cnt, h, l, e + r, K, Kp);
}

template <int R4>
__global__ void epilogue_geglu_qw_k(const int32_t* __restrict__ Chi, const int32_t* __restrict__ Clo, int M, int F, int ldc, const int4* __restrict__ cst,
                                    const int32_t* __restrict__ e, const int32_t* __restrict__ L, int8_t* __restrict__ hi, int8_t* __restrict__ lo, int32_t* __restrict__ eo, int Kp) {
    const int lane = threadIdx.x & 31; const size_t row = size_t(blockIdx.x) * 8 + (threadIdx.x >> 5);
    if (row >= size_t(M)) return;
    const int er = e[row]; const size_t base = row * ldc; const int F4 = F >> 2;
    int v[4 * R4]; int cnt = 0;
    for (int c4 = lane; c4 < F4; c4 += 32, ++cnt) {
        const int c = c4 * 4;
        const int4 ah = *reinterpret_cast<const int4*>(Chi + base + c), al = *reinterpret_cast<const int4*>(Clo + base + c);
        const int4 gh = *reinterpret_cast<const int4*>(Chi + base + F + c), gl = *reinterpret_cast<const int4*>(Clo + base + F + c);
        const int ahv[4] = {ah.x, ah.y, ah.z, ah.w}, alv[4] = {al.x, al.y, al.z, al.w}, ghv[4] = {gh.x, gh.y, gh.z, gh.w}, glv[4] = {gl.x, gl.y, gl.z, gl.w};
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            i64 a = epi1v(ahv[k], alv[k], cst[c + k], er);
            i64 gg = epi1v(ghv[k], glv[k], cst[F + c + k], er);
            i64 ge = sat32(rshr(gg * lut16i(gg, L), 16));
            v[4 * cnt + k] = int(sat32(rshr(a * ge, 16)));
        }
    }
    quant_row_out_w4<R4>(v, cnt, hi + row * Kp, lo + row * Kp, eo + row, F, Kp);
}
template <int R>
__global__ void epilogue_geglu_q_k(const int32_t* __restrict__ Chi, const int32_t* __restrict__ Clo, int F, int ldc, const int4* __restrict__ cst,
                                   const int32_t* __restrict__ e, const int32_t* __restrict__ L, int8_t* __restrict__ hi, int8_t* __restrict__ lo, int32_t* __restrict__ eo, int Kp) {
    __shared__ i64 sh[32];
    const size_t row = blockIdx.x; const int er = e[row]; const size_t base = row * ldc;
    int v[R]; int cnt = 0;
    for (int c = threadIdx.x; c < F; c += NT, ++cnt) {
        i64 a = epi1(Chi, Clo, base + c, cst[c], er);
        i64 gg = epi1(Chi, Clo, base + F + c, cst[F + c], er);
        i64 ge = sat32(rshr(gg * lut16i(gg, L), 16));
        v[cnt] = int(sat32(rshr(a * ge, 16)));
    }
    quant_row_out<R>(v, cnt, F, hi + row * Kp, lo + row * Kp, eo + row, Kp, sh);
}

// ---------------------------------------------------------------- elementwise
__global__ void act_k(int32_t* X, size_t n, const int32_t* __restrict__ L) { size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i < n) { i64 x = X[i]; X[i] = int32_t(sat32(rshr(x * lut16i(x, L), 16))); } }
__global__ void geglu_k(int32_t* A, const int32_t* __restrict__ G, size_t n, const int32_t* __restrict__ L) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i >= n) return;
    i64 gg = G[i]; i64 ge = sat32(rshr(gg * lut16i(gg, L), 16));
    A[i] = int32_t(sat32(rshr(i64(A[i]) * ge, 16)));
}
__global__ void add_k(int32_t* X, const int32_t* __restrict__ Y, size_t n) { size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i < n) X[i] = int32_t(sat32(i64(X[i]) + Y[i])); }
__global__ void concat_k(const int32_t* __restrict__ A, int Ca, const int32_t* __restrict__ Bm, int Cb, int rows, int32_t* __restrict__ out) {
    int Co = Ca + Cb; size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i >= size_t(rows) * Co) return;
    size_t r = i / Co; int c = int(i % Co);
    out[i] = c < Ca ? A[r * Ca + c] : Bm[r * Cb + (c - Ca)];
}
__global__ void upsample2_k(const int32_t* __restrict__ X, int B, int H, int W, int C, int32_t* __restrict__ out) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; size_t tot = size_t(B) * 4 * H * W * C; if (i >= tot) return;
    int c = int(i % C); size_t p = i / C; int x = int(p % (2 * W)); p /= 2 * W; int y = int(p % (2 * H)); int b = int(p / (2 * H));
    out[i] = X[((size_t(b) * H + (y >> 1)) * W + (x >> 1)) * C + c];
}

// ---------------------------------------------------------------- ChaCha20 noise
__device__ __forceinline__ u32 rotl32(u32 v, int n) { return (v << n) | (v >> (32 - n)); }
#define QR(a, b, c, d) a += b; d ^= a; d = rotl32(d, 16); c += d; b ^= c; b = rotl32(b, 12); a += b; d ^= a; d = rotl32(d, 8); c += d; b ^= c; b = rotl32(b, 7);
// grid (512 blocks per sample /NT, B); each thread one 64-byte keystream block -> 32 u16 -> 32 latent values
__global__ void noise_k(const uint8_t* __restrict__ seeds, const i64* __restrict__ gauss, i64 cin, int32_t* __restrict__ n, int32_t* __restrict__ x) {
    int blk = blockIdx.x * blockDim.x + threadIdx.x; int b = blockIdx.y;
    if (blk >= 512) return;
    const uint8_t* key = seeds + 32 * b;
    u32 st[16]; u32 in[16];
    in[0] = 0x61707865u; in[1] = 0x3320646eu; in[2] = 0x79622d32u; in[3] = 0x6b206574u;
    for (int i = 0; i < 8; ++i) in[4 + i] = u32(key[4 * i]) | (u32(key[4 * i + 1]) << 8) | (u32(key[4 * i + 2]) << 16) | (u32(key[4 * i + 3]) << 24);
    in[12] = u32(blk); in[13] = 0; in[14] = 0; in[15] = 0;
    for (int i = 0; i < 16; ++i) st[i] = in[i];
    for (int r = 0; r < 10; ++r) {
        QR(st[0], st[4], st[8], st[12]) QR(st[1], st[5], st[9], st[13]) QR(st[2], st[6], st[10], st[14]) QR(st[3], st[7], st[11], st[15])
        QR(st[0], st[5], st[10], st[15]) QR(st[1], st[6], st[11], st[12]) QR(st[2], st[7], st[8], st[13]) QR(st[3], st[4], st[9], st[14])
    }
    int32_t* nb = n + size_t(b) * 16384; int32_t* xb = x + size_t(b) * 16384;
    for (int i = 0; i < 16; ++i) {
        u32 w = st[i] + in[i];
        u32 u0 = w & 0xffffu, u1 = w >> 16;      // little-endian u16 pairs
        int idx = blk * 32 + 2 * i;
        i64 g0 = gauss[u0], g1 = gauss[u1];
        nb[idx] = int32_t(g0); nb[idx + 1] = int32_t(g1);
        int c0 = idx >> 12, p0 = idx & 4095, c1 = (idx + 1) >> 12, p1 = (idx + 1) & 4095;
        xb[p0 * 4 + c0] = int32_t(rshr(g0 * cin, 30));
        xb[p1 * 4 + c1] = int32_t(rshr(g1 * cin, 30));
    }
}
__global__ void finalize_k(const int32_t* __restrict__ n, const int32_t* __restrict__ eps, int B, i64 sig, uint8_t* __restrict__ out) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i >= size_t(B) * 16384) return;
    int b = int(i / 16384), idx = int(i % 16384), c = idx >> 12, p = idx & 4095;
    i64 x0 = rshr((i64(n[i]) - eps[size_t(b) * 16384 + p * 4 + c]) * sig, 16);
    i64 q = rshr(x0, 8); q = q < -32767 ? -32767 : (q > 32767 ? 32767 : q);
    uint16_t u = uint16_t(int16_t(q));
    out[2 * i] = uint8_t(u & 0xff); out[2 * i + 1] = uint8_t(u >> 8);
}
__global__ void text_embed_k(const int32_t* __restrict__ ids, int T, int rowsPad, const int8_t* __restrict__ tw, const i64* __restrict__ tm, const i64* __restrict__ ts, const i64* __restrict__ pos, int D, int32_t* __restrict__ X) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i >= size_t(rowsPad) * D) return;
    int t = int(i / D), d = int(i % D);
    if (t >= T) { X[i] = 0; return; }
    int id = ids[t];
    i64 v = rshr(i64(tw[size_t(id) * D + d]) * tm[id], ts[id] - 16);
    X[i] = int32_t(sat32(v + pos[size_t(t) * D + d]));
}

// ---------------------------------------------------------------- attention operands
// one warp per (g, t) row of 64 values
__global__ void quant_heads_k(const int32_t* __restrict__ X, int B, int T, int C, int heads, int Tp,
                              int8_t* __restrict__ hi, int8_t* __restrict__ lo, int32_t* __restrict__ e, i64* __restrict__ hs, i64* __restrict__ ls, i64* __restrict__ tsum) {
    int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    int G = B * heads; if (w >= G * Tp) return;
    int g = w / Tp, t = w % Tp, b = g / heads, h = g % heads;
    size_t o = size_t(w) * 64;
    if (t >= T) {
        hi[o + lane] = 0; hi[o + 32 + lane] = 0; lo[o + lane] = -128; lo[o + 32 + lane] = -128;
        if (lane == 0) { e[w] = 0; hs[w] = 0; ls[w] = -128 * 64; tsum[w] = 0; }
        return;
    }
    const int32_t* x = X + (size_t(b) * T + t) * C + h * 64;
    int v0 = x[lane], v1 = x[lane + 32];
    u32 m = warp_umax(max(iabs32(v0), iabs32(v1)));
    int ee = row_exp(m);
    i64 q0 = quant16(v0, ee), q1 = quant16(v1, ee);
    int8_t h0, l0, h1, l1; split16(q0, h0, l0); split16(q1, h1, l1);
    hi[o + lane] = h0; hi[o + 32 + lane] = h1; lo[o + lane] = l0; lo[o + 32 + lane] = l1;
    i64 sh = warp_sum(i64(h0) + h1), sl = warp_sum(i64(l0) + l1), st = warp_sum(q0 + q1);
    if (lane == 0) { e[w] = ee; hs[w] = sh; ls[w] = sl; tsum[w] = st; }
}
// per (g, d) column of V: column max (atomic), then ev + vsum. grid (G, splits), block 256 = 64 dims x 4 t-slices
__global__ void quant_v_max_k(const int32_t* __restrict__ X, int Tk, int C, int heads, u32* __restrict__ pm) {
    __shared__ u32 sh[4][64];
    int g = blockIdx.x, b = g / heads, h = g % heads, d = threadIdx.x & 63, ts = threadIdx.x >> 6;
    int per = (Tk + gridDim.y - 1) / gridDim.y, t0 = blockIdx.y * per, t1 = min(Tk, t0 + per);
    const int32_t* x = X + size_t(b) * Tk * C + h * 64 + d;
    u32 m = 0;
    for (int t = t0 + ts; t < t1; t += 4) m = max(m, iabs32(x[size_t(t) * C]));
    sh[ts][d] = m; __syncthreads();
    if (ts == 0) { m = max(max(sh[0][d], sh[1][d]), max(sh[2][d], sh[3][d])); atomicMax(pm + g * 64 + d, m); }
}
__global__ void quant_v_sum_k(const int32_t* __restrict__ X, int Tk, int C, int heads, const u32* __restrict__ pm, int32_t* __restrict__ ev, i64* __restrict__ vsum) {
    __shared__ i64 sh[4][64];
    int g = blockIdx.x, b = g / heads, h = g % heads, d = threadIdx.x & 63, ts = threadIdx.x >> 6;
    int per = (Tk + gridDim.y - 1) / gridDim.y, t0 = blockIdx.y * per, t1 = min(Tk, t0 + per);
    const int32_t* x = X + size_t(b) * Tk * C + h * 64 + d;
    int e = row_exp(pm[g * 64 + d]); i64 s = 0;
    for (int t = t0 + ts; t < t1; t += 4) s += quant16(x[size_t(t) * C], e);
    sh[ts][d] = s; __syncthreads();
    if (ts == 0) { s = sh[0][d] + sh[1][d] + sh[2][d] + sh[3][d]; atomicAdd((unsigned long long*)(vsum + g * 64 + d), (unsigned long long)s); if (blockIdx.y == 0) ev[g * 64 + d] = e; }
}
// transposed write: block per (g, 64-wide t tile); smem tile [64 t][65]
__global__ void quant_v_write_k(const int32_t* __restrict__ X, int B, int Tk, int C, int heads, int Tkp, const int32_t* __restrict__ ev,
                                int8_t* __restrict__ vt_hi, int8_t* __restrict__ vt_lo) {
    __shared__ int32_t tile[64][65];
    int g = blockIdx.y, t0 = blockIdx.x * 64, b = g / heads, h = g % heads;
    int tx = threadIdx.x & 63, ty = threadIdx.x >> 6;     // 256 threads: 4 rows at a time
    for (int r = ty; r < 64; r += 4) { int t = t0 + r; tile[r][tx] = t < Tk ? X[(size_t(b) * Tk + t) * C + h * 64 + tx] : 0; }
    __syncthreads();
    // write vt[g][d][t]: thread handles d = ty.., t = t0+tx
    for (int d = ty; d < 64; d += 4) {
        int t = t0 + tx; int e = ev[g * 64 + d];
        i64 q = t < Tk ? quant16(tile[tx][d], e) : 0;
        int8_t a, bb; split16(q, a, bb);
        size_t o = (size_t(g) * 64 + d) * Tkp + t;
        vt_hi[o] = a; vt_lo[o] = bb;
    }
}
// block per (g, query row); row of Tkp scores in registers (<= 16 per thread for Tkp <= 4096)
template <int PER>
__global__ void softmax_k(const int32_t* __restrict__ Shh, const int32_t* __restrict__ Shl, const int32_t* __restrict__ Slh, const int32_t* __restrict__ Sll,
                          const i64* __restrict__ qhs, const i64* __restrict__ qls, const int32_t* __restrict__ eq, const i64* __restrict__ ksum, const int32_t* __restrict__ ek,
                          int Tqp, int q0, int Qc, int Tk, int Tkp, int causal, const i64* __restrict__ Lexp,
                          int8_t* __restrict__ Phi, int8_t* __restrict__ Plo, i64* __restrict__ phs, i64* __restrict__ pls) {
    __shared__ i64 sh[32];
    int a = blockIdx.x % Qc, g = blockIdx.x / Qc, q = q0 + a;
    size_t row = (size_t(g) * Qc + a) * Tkp;
    int rq = g * Tqp + q;
    i64 cq = 32768LL * qhs[rq] + 128LL * qls[rq]; int eqq = eq[rq];
    int s[PER]; int mx = INT_MIN;
    for (int j = 0; j < PER; ++j) {
        int t = threadIdx.x + j * NT; int v = INT_MIN;
        if (t < Tk && !(causal && t > q)) {
            int rk = g * Tkp + t;
            i64 S = 65536LL * Shh[row + t] + 256LL * (i64(Shl[row + t]) + Slh[row + t]) + Sll[row + t] + cq + 128LL * ksum[rk];
            v = int(rshr(S, 19 - eqq - ek[rk]));
        }
        s[j] = v; mx = max(mx, v);
    }
    mx = block_reduce<int>(mx, MaxOp(), INT_MIN, (int*)sh);
    i64 E[PER]; i64 sum = 0;
    for (int j = 0; j < PER; ++j) {
        int t = threadIdx.x + j * NT; i64 ev = 0;
        if (t < Tk && !(causal && t > q)) {
            i64 z = i64(mx) - s[j];
            if (z < (16LL << 16)) { i64 i = z >> 10, f = z & 1023; ev = Lexp[i] + (((Lexp[i + 1] - Lexp[i]) * f) >> 10); }
        }
        E[j] = ev; sum += ev;
    }
    sum = block_reduce<i64>(sum, SumOp(), 0, sh);
    i64 r = floor_div(1LL << 62, sum);
    i64 hsum = 0, lsum = 0;
    for (int j = 0; j < PER; ++j) {
        int t = threadIdx.x + j * NT; if (t >= Tkp) break;
        i64 P = 0;
        if (E[j] > 0) { P = (E[j] * r + (1LL << 46)) >> 47; if (P > 32767) P = 32767; }
        int8_t a8, b8; split16(P, a8, b8);
        Phi[row + t] = a8; Plo[row + t] = b8; hsum += a8; lsum += b8;
    }
    hsum = block_reduce<i64>(hsum, SumOp(), 0, sh); lsum = block_reduce<i64>(lsum, SumOp(), 0, sh);
    if (threadIdx.x == 0) { phs[size_t(g) * Qc + a] = hsum; pls[size_t(g) * Qc + a] = lsum; }
}
__global__ void attn_out_k(const int32_t* __restrict__ Ohh, const int32_t* __restrict__ Ohl, const int32_t* __restrict__ Olh, const int32_t* __restrict__ Oll,
                           const i64* __restrict__ phs, const i64* __restrict__ pls, const i64* __restrict__ vsum, const int32_t* __restrict__ ev,
                           int B, int heads, int Tq, int q0, int Qc, int C, int32_t* __restrict__ out) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; int G = B * heads; if (i >= size_t(G) * Qc * 64) return;
    int d = int(i & 63); size_t ra = i >> 6; int a = int(ra % Qc); int g = int(ra / Qc); int b = g / heads, h = g % heads;
    int q = q0 + a; if (q >= Tq) return;
    i64 O = 65536LL * Ohh[i] + 256LL * (i64(Ohl[i]) + Olh[i]) + Oll[i] + 32768LL * phs[ra] + 128LL * pls[ra] + 128LL * vsum[g * 64 + d];
    out[(size_t(b) * Tq + q) * C + h * 64 + d] = int32_t(rshr(O, 15 - ev[g * 64 + d]));
}
inline unsigned gridFor(size_t n) { return unsigned((n + NT - 1) / NT); }
}  // namespace

#define WARPDISPATCH(C, KERN, GRID, ...) do { if (C <= 384) KERN<3><<<GRID, NT, 0, st>>>(__VA_ARGS__); else if (C <= 640) KERN<5><<<GRID, NT, 0, st>>>(__VA_ARGS__); else if (C <= 1024) KERN<8><<<GRID, NT, 0, st>>>(__VA_ARGS__); else KERN<10><<<GRID, NT, 0, st>>>(__VA_ARGS__); } while (0)
void k_quant_rows(cudaStream_t st, const int32_t* X, int M, int K, int ldx, int8_t* hi, int8_t* lo, int32_t* e, int Kp) {
    int Mp = int((M + 31) / 32 * 32);
    if (K <= 1280 && K % 4 == 0 && ldx % 4 == 0) WARPDISPATCH(K, quant_rows_w_k, unsigned((Mp + 7) / 8), X, M, K, ldx, hi, lo, e, Kp);
    else quant_rows_k<<<Mp, NT, 0, st>>>(X, M, K, ldx, hi, lo, e, Kp);
}
void k_epilogue(cudaStream_t st, const int32_t* Chi, const int32_t* Clo, int M, int N, int ldc, const int4* cst, const int32_t* e, int32_t* Y, int ldy, const int32_t* residual) {
    if (N % 4 || ldc % 4 || ldy % 4) throw std::runtime_error("epilogue: N/ld must be multiples of 4");
    epilogue_k<<<gridFor(size_t(M) * (N / 4)), NT, 0, st>>>(Chi, Clo, M, N, ldc, cst, e, Y, ldy, residual);
}
void k_pixmax(cudaStream_t st, const int32_t* X, int rows, int C, uint32_t* pm) { pixmax_k<<<gridFor(size_t(rows) * 32), NT, 0, st>>>(X, rows, C, pm); }
void k_im2col3(cudaStream_t st, const int32_t* X, const uint32_t* pm, int B, int H, int W, int C, int stride, int8_t* hi, int8_t* lo, int32_t* e, int Kp) {
    int Ho = (H - 1) / stride + 1, Wo = (W - 1) / stride + 1; int rows = B * Ho * Wo; int Mp = (rows + 31) / 32 * 32;
    im2col3_k<<<rows, NT, 0, st>>>(X, pm, B, H, W, C, stride, hi, lo, e, Kp, Ho, Wo);
    if (Mp > rows) { cudaMemsetAsync(hi + size_t(rows) * Kp, 0, size_t(Mp - rows) * Kp, st); cudaMemsetAsync(lo + size_t(rows) * Kp, 0x80, size_t(Mp - rows) * Kp, st); }
}
void k_gn_stats1(cudaStream_t st, const int32_t* X, int B, int T, int C, int groups, GnStat* gs) {
    int cnt = B * groups; gn_init_k<<<gridFor(cnt), NT, 0, st>>>(gs, cnt);
    int splits = T >= 1024 ? 8 : (T >= 256 ? 4 : 1);
    gn_stats1_k<<<dim3(groups, B, splits), NT, 0, st>>>(X, T, C, groups, gs);
    gn_fin1_k<<<gridFor(cnt), NT, 0, st>>>(gs, cnt, i64(T) * (C / groups));
}
void k_gn_stats2(cudaStream_t st, const int32_t* X, int B, int T, int C, int groups, i64 den, GnStat* gs) {
    int cnt = B * groups; int splits = T >= 1024 ? 8 : (T >= 256 ? 4 : 1);
    gn_stats2_k<<<dim3(groups, B, splits), NT, 0, st>>>(X, T, C, groups, gs);
    gn_fin2_k<<<gridFor(cnt), NT, 0, st>>>(gs, cnt, i64(T) * (C / groups), den);
}
void k_gn_apply(cudaStream_t st, const int32_t* X, int B, int T, int C, int groups, const GnStat* gs, const i64* g, const i64* beta, int32_t* Y) {
    gn_apply_k<<<gridFor(size_t(B) * T * C), NT, 0, st>>>(X, B, T, C, groups, gs, g, beta, Y);
}
void k_lnorm(cudaStream_t st, const int32_t* X, int M, int C, const i64* g, const i64* beta, i64 den, int32_t* Y) {
    if (C > 8 * NT) throw std::runtime_error("lnorm: C too large");
    lnorm_k<<<M, NT, 0, st>>>(X, C, g, beta, den, Y);
}
void k_act(cudaStream_t st, int32_t* X, size_t n, const int32_t* L) { act_k<<<gridFor(n), NT, 0, st>>>(X, n, L); }
void k_geglu(cudaStream_t st, int32_t* A, const int32_t* G, size_t n, const int32_t* L) { geglu_k<<<gridFor(n), NT, 0, st>>>(A, G, n, L); }
void k_add(cudaStream_t st, int32_t* X, const int32_t* Y, size_t n) { add_k<<<gridFor(n), NT, 0, st>>>(X, Y, n); }
void k_concat(cudaStream_t st, const int32_t* A, int Ca, const int32_t* B, int Cb, int rows, int32_t* out) { concat_k<<<gridFor(size_t(rows) * (Ca + Cb)), NT, 0, st>>>(A, Ca, B, Cb, rows, out); }
void k_upsample2(cudaStream_t st, const int32_t* X, int B, int H, int W, int C, int32_t* out) { upsample2_k<<<gridFor(size_t(B) * 4 * H * W * C), NT, 0, st>>>(X, B, H, W, C, out); }
void k_noise(cudaStream_t st, const uint8_t* seeds, int B, const i64* gauss, i64 cin, int32_t* n, int32_t* x) { noise_k<<<dim3(2, B), NT, 0, st>>>(seeds, gauss, cin, n, x); }
void k_finalize(cudaStream_t st, const int32_t* n, const int32_t* eps, int B, i64 sig, uint8_t* out) { finalize_k<<<gridFor(size_t(B) * 16384), NT, 0, st>>>(n, eps, B, sig, out); }
void k_text_embed(cudaStream_t st, const int32_t* ids, int T, int rowsPad, const int8_t* tw, const i64* tm, const i64* ts, const i64* pos, int D, int32_t* X) {
    text_embed_k<<<gridFor(size_t(rowsPad) * D), NT, 0, st>>>(ids, T, rowsPad, tw, tm, ts, pos, D, X);
}
void k_quant_heads(cudaStream_t st, const int32_t* X, int B, int T, int C, int heads, int Tp, int8_t* hi, int8_t* lo, int32_t* e, i64* hs, i64* ls, i64* tsum) {
    quant_heads_k<<<gridFor(size_t(B) * heads * Tp * 32), NT, 0, st>>>(X, B, T, C, heads, Tp, hi, lo, e, hs, ls, tsum);
}
void k_quant_v(cudaStream_t st, const int32_t* X, int B, int Tk, int C, int heads, int Tkp, int8_t* vt_hi, int8_t* vt_lo, int32_t* ev, i64* vsum, uint32_t* scratch) {
    int G = B * heads; int splits = Tk >= 1024 ? 16 : (Tk >= 256 ? 4 : 1);
    cudaMemsetAsync(scratch, 0, size_t(G) * 64 * 4, st); cudaMemsetAsync(vsum, 0, size_t(G) * 64 * 8, st);
    quant_v_max_k<<<dim3(G, splits), NT, 0, st>>>(X, Tk, C, heads, scratch);
    quant_v_sum_k<<<dim3(G, splits), NT, 0, st>>>(X, Tk, C, heads, scratch, ev, vsum);
    quant_v_write_k<<<dim3(Tkp / 64, G), NT, 0, st>>>(X, B, Tk, C, heads, Tkp, ev, vt_hi, vt_lo);
}
void k_softmax(cudaStream_t st, const int32_t* Shh, const int32_t* Shl, const int32_t* Slh, const int32_t* Sll, const i64* qhs, const i64* qls, const int32_t* eq, const i64* ksum, const int32_t* ek,
               int G, int Tqp, int q0, int Qc, int Tk, int Tkp, bool causal, const i64* Lexp, int8_t* Phi, int8_t* Plo, i64* phs, i64* pls) {
    unsigned blocks = unsigned(G) * Qc;
#define SM(P) softmax_k<P><<<blocks, NT, 0, st>>>(Shh, Shl, Slh, Sll, qhs, qls, eq, ksum, ek, Tqp, q0, Qc, Tk, Tkp, causal, Lexp, Phi, Plo, phs, pls)
    if (Tkp <= 256) SM(1); else if (Tkp <= 512) SM(2); else if (Tkp <= 1024) SM(4); else if (Tkp <= 2048) SM(8); else if (Tkp <= 4096) SM(16); else throw std::runtime_error("softmax: Tk too large");
#undef SM
}
void k_attn_out(cudaStream_t st, const int32_t* Ohh, const int32_t* Ohl, const int32_t* Olh, const int32_t* Oll, const i64* phs, const i64* pls, const i64* vsum, const int32_t* ev,
                int B, int heads, int Tq, int q0, int Qc, int C, int32_t* out) {
    attn_out_k<<<gridFor(size_t(B) * heads * Qc * 64), NT, 0, st>>>(Ohh, Ohl, Olh, Oll, phs, pls, vsum, ev, B, heads, Tq, q0, Qc, C, out);
}
void k_zero(cudaStream_t st, void* p, size_t bytes) { cudaMemsetAsync(p, 0, bytes, st); }
__global__ void quant_heads_pad_k(int G, int T, int Tp, int8_t* __restrict__ hi, int8_t* __restrict__ lo, int32_t* __restrict__ e, i64* __restrict__ hs, i64* __restrict__ ls, i64* __restrict__ tsum) {
    const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31; const int npad = Tp - T;
    if (w >= G * npad) return;
    const int g = w / npad, t = T + w % npad; const size_t row = size_t(g) * Tp + t;
    hi[row * 64 + lane] = 0; hi[row * 64 + 32 + lane] = 0; lo[row * 64 + lane] = -128; lo[row * 64 + 32 + lane] = -128;
    if (lane == 0) { e[row] = 0; hs[row] = 0; ls[row] = -128 * 64; tsum[row] = 0; }
}
void k_quant_heads_pad(cudaStream_t st, int G, int T, int Tp, int8_t* hi, int8_t* lo, int32_t* e, i64* hs, i64* ls, i64* tsum) {
    if (Tp <= T) return;
    quant_heads_pad_k<<<gridFor(size_t(G) * (Tp - T) * 32), NT, 0, st>>>(G, T, Tp, hi, lo, e, hs, ls, tsum);
}
void k_epilogue_split(cudaStream_t st, const int32_t* Chi, const int32_t* Clo, int M, int N, int ldc, const int4* cst, const int32_t* e, int Cs, int32_t* Y0, int32_t* Y1, int32_t* Y2) {
    epilogue_split_k<<<dim3(gridFor(N), unsigned(std::min(M, 65535))), NT, 0, st>>>(Chi, Clo, M, N, ldc, cst, e, Cs, Y0, Y1, Y2);
}
void k_epilogue_geglu_q(cudaStream_t st, const int32_t* Chi, const int32_t* Clo, int M, int F, int ldc, const int4* cst, const int32_t* e, const int32_t* Lphi, int8_t* hi, int8_t* lo, int32_t* eo, int Kp) {
    if (F <= 1280) WARPDISPATCH(F, epilogue_geglu_qw_k, unsigned((M + 7) / 8), Chi, Clo, M, F, ldc, cst, e, Lphi, hi, lo, eo, Kp);
    else if (F <= 12 * NT) epilogue_geglu_q_k<12><<<M, NT, 0, st>>>(Chi, Clo, F, ldc, cst, e, Lphi, hi, lo, eo, Kp);
    else if (F <= 20 * NT) epilogue_geglu_q_k<20><<<M, NT, 0, st>>>(Chi, Clo, F, ldc, cst, e, Lphi, hi, lo, eo, Kp);
    else throw std::runtime_error("epilogue_geglu_q: F too large");
}
#define ROWDISPATCH(C, KERN, ...) do { if (C <= 4 * NT) KERN<4><<<__VA_ARGS__; else if (C <= 8 * NT) KERN<8><<<__VA_ARGS__; else if (C <= 12 * NT) KERN<12><<<__VA_ARGS__; else if (C <= 20 * NT) KERN<20><<<__VA_ARGS__; else throw std::runtime_error("row kernel: C too large"); } while (0)
void k_lnorm_q(cudaStream_t st, const int32_t* X, int M, int C, const int32_t* g, const int32_t* beta, i64 den, int8_t* hi, int8_t* lo, int32_t* e, int Kp) {
    if (C > 1280 || C % 4) throw std::runtime_error("lnorm_q: C must be <= 1280 and a multiple of 4");
    WARPDISPATCH(C, lnorm_qw_k, unsigned((M + 7) / 8), X, M, C, g, beta, den, hi, lo, e, Kp);
}
void k_gn_apply_row(cudaStream_t st, const int32_t* X, int B, int T, int C, int groups, const GnStat* gs, const int32_t* g, const int32_t* beta, const int32_t* Lsig, int32_t* Y, uint32_t* pm, int8_t* hi, int8_t* lo, int32_t* e, int Kp) {
    size_t M = size_t(B) * T;
    if (C % 4) throw std::runtime_error("gn_apply_row: C % 4");
    if (C <= 1280) WARPDISPATCH(C, gn_apply_roww_k, unsigned((M + 7) / 8), X, M, T, C, groups, gs, g, beta, Lsig, Y, pm, hi, lo, e, Kp);
    else if (C <= 2560) gn_apply_roww_k<20><<<unsigned((M + 7) / 8), NT, 0, st>>>(X, M, T, C, groups, gs, g, beta, Lsig, Y, pm, hi, lo, e, Kp);
    else throw std::runtime_error("gn_apply_row: C too large");
}
}  // namespace meow
