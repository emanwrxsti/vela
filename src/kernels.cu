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
        i64 q = k < K ? quant16(x[k], ee) : 0;
        int8_t a, b; split16(q, a, b); h[k] = a; l[k] = b;
    }
}

__global__ void epilogue_k(const int32_t* __restrict__ Chi, const int32_t* __restrict__ Clo, int M, int N, int ldc,
                           const i64* __restrict__ wsum, const i64* __restrict__ m, const i64* __restrict__ s, const i64* __restrict__ b,
                           const int32_t* __restrict__ e, int32_t* __restrict__ Y, int ldy, const int32_t* R) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(M) * N) return;
    int r = int(i / N), n = int(i % N);
    size_t ci = size_t(r) * ldc + n;
    i64 acc = 256LL * Chi[ci] + Clo[ci] + 128LL * wsum[n];
    i64 y = sat32(rshr(acc * m[n], s[n] - e[r]) + b[n]);
    if (R) y = sat32(y + R[size_t(r) * ldy + n]);
    Y[size_t(r) * ldy + n] = int32_t(y);
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
    for (int k = threadIdx.x; k < Kp; k += blockDim.x) {
        i64 q = 0;
        if (k < K) {
            int tap = k / C, c = k - tap * C, ky = tap / 3, kx = tap - ky * 3;
            int iy = iy0 + ky, ix = ix0 + kx;
            if (iy >= 0 && iy < H && ix >= 0 && ix < W) q = quant16(X[((size_t(b) * H + iy) * W + ix) * C + c], ee);
        }
        int8_t a, bb; split16(q, a, bb); h[k] = a; l[k] = bb;
    }
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
    for (int c = threadIdx.x; c < C; c += NT, ++j) { i64 q = quant16(v[j], ee); int8_t a, b; split16(q, a, b); h[c] = a; l[c] = b; }
    for (int c = C + threadIdx.x; c < Kp; c += NT) { h[c] = 0; l[c] = -128; }
}
template <int R>
__global__ void lnorm_q_k(const int32_t* __restrict__ X, int C, const i64* __restrict__ g, const i64* __restrict__ beta, i64 den, int8_t* __restrict__ hi, int8_t* __restrict__ lo, int32_t* __restrict__ e, int Kp) {
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
__global__ void gn_apply_row_k(const int32_t* __restrict__ X, int T, int C, int groups, const GnStat* __restrict__ gs, const i64* __restrict__ g, const i64* __restrict__ beta, const i64* __restrict__ Lsig,
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
        if (Lsig) y = sat32(rshr(y * lut16(y, Lsig), 16));
        v[cnt] = int(y); m = max(m, iabs32(int(y)));
    }
    if (hi) { quant_row_out<R>(v, cnt, C, hi + row * Kp, lo + row * Kp, e + row, Kp, sh); return; }
    int32_t* y = Y + row * C; cnt = 0;
    for (int c = threadIdx.x; c < C; c += NT) y[c] = v[cnt++];
    m = block_reduce<u32>(m, UMaxOp(), 0u, (u32*)sh);
    if (threadIdx.x == 0) pm[row] = m;
}
template <int R>
__global__ void geglu_q_k(const int32_t* __restrict__ A, const int32_t* __restrict__ G, int C, const i64* __restrict__ L, int8_t* __restrict__ hi, int8_t* __restrict__ lo, int32_t* __restrict__ e, int Kp) {
    __shared__ i64 sh[32];
    size_t row = blockIdx.x; const int32_t* a = A + row * C; const int32_t* gg = G + row * C;
    int v[R]; int cnt = 0;
    for (int c = threadIdx.x; c < C; c += NT, ++cnt) { i64 x = gg[c]; i64 ge = sat32(rshr(x * lut16(x, L), 16)); v[cnt] = int(sat32(rshr(i64(a[c]) * ge, 16))); }
    quant_row_out<R>(v, cnt, C, hi + row * Kp, lo + row * Kp, e + row, Kp, sh);
}

// ---------------------------------------------------------------- elementwise
__global__ void act_k(int32_t* X, size_t n, const i64* __restrict__ L) { size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i < n) { i64 x = X[i]; X[i] = int32_t(sat32(rshr(x * lut16(x, L), 16))); } }
__global__ void geglu_k(int32_t* A, const int32_t* __restrict__ G, size_t n, const i64* __restrict__ L) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i >= n) return;
    i64 gg = G[i]; i64 ge = sat32(rshr(gg * lut16(gg, L), 16));
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

void k_quant_rows(cudaStream_t st, const int32_t* X, int M, int K, int ldx, int8_t* hi, int8_t* lo, int32_t* e, int Kp) {
    int Mp = int((M + 31) / 32 * 32);
    quant_rows_k<<<Mp, NT, 0, st>>>(X, M, K, ldx, hi, lo, e, Kp);
}
void k_epilogue(cudaStream_t st, const int32_t* Chi, const int32_t* Clo, int M, int N, int ldc, const i64* wsum, const i64* m, const i64* s, const i64* b, const int32_t* e, int32_t* Y, int ldy, const int32_t* residual) {
    epilogue_k<<<gridFor(size_t(M) * N), NT, 0, st>>>(Chi, Clo, M, N, ldc, wsum, m, s, b, e, Y, ldy, residual);
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
void k_act(cudaStream_t st, int32_t* X, size_t n, const i64* L) { act_k<<<gridFor(n), NT, 0, st>>>(X, n, L); }
void k_geglu(cudaStream_t st, int32_t* A, const int32_t* G, size_t n, const i64* L) { geglu_k<<<gridFor(n), NT, 0, st>>>(A, G, n, L); }
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
#define ROWDISPATCH(C, KERN, ...) do { if (C <= 4 * NT) KERN<4><<<__VA_ARGS__; else if (C <= 8 * NT) KERN<8><<<__VA_ARGS__; else if (C <= 12 * NT) KERN<12><<<__VA_ARGS__; else if (C <= 20 * NT) KERN<20><<<__VA_ARGS__; else throw std::runtime_error("row kernel: C too large"); } while (0)
void k_lnorm_q(cudaStream_t st, const int32_t* X, int M, int C, const i64* g, const i64* beta, i64 den, int8_t* hi, int8_t* lo, int32_t* e, int Kp) {
    ROWDISPATCH(C, lnorm_q_k, M, NT, 0, st>>>(X, C, g, beta, den, hi, lo, e, Kp));
}
void k_gn_apply_row(cudaStream_t st, const int32_t* X, int B, int T, int C, int groups, const GnStat* gs, const i64* g, const i64* beta, const i64* Lsig, int32_t* Y, uint32_t* pm, int8_t* hi, int8_t* lo, int32_t* e, int Kp) {
    ROWDISPATCH(C, gn_apply_row_k, unsigned(size_t(B) * T), NT, 0, st>>>(X, T, C, groups, gs, g, beta, Lsig, Y, pm, hi, lo, e, Kp));
}
void k_geglu_q(cudaStream_t st, const int32_t* A, const int32_t* G, int M, int C, const i64* Lphi, int8_t* hi, int8_t* lo, int32_t* e, int Kp) {
    ROWDISPATCH(C, geglu_q_k, M, NT, 0, st>>>(A, G, C, Lphi, hi, lo, e, Kp));
}
}  // namespace meow
