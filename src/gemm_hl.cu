#include "gemm_hl.h"
#include "intops.cuh"
#include <cstdint>
#include <stdexcept>

namespace meow {
namespace {
constexpr int BK = 64, NTHR = 256;
// Tile configuration: BM output rows per block (A tile = BM hi rows + BM lo rows), BN columns, WM x WN warps (8 total).
template <int BM_, int BN_, int STAGES_, int WM_> struct Cfg {
    static constexpr int BM = BM_, BN = BN_, STAGES = STAGES_, WM = WM_, WN = 8 / WM_;
    static constexpr int A_ROWS = 2 * BM, A_BYTES = A_ROWS * BK, B_BYTES = BN * BK, STAGE_BYTES = A_BYTES + B_BYTES, SMEM = STAGES * STAGE_BYTES;
    static constexpr int WTM = BM / WM, WTN = BN / WN, MI = WTM / 16, NI = WTN / 8;     // per-warp output tile and mma tiles per plane
    static_assert(MI >= 1 && NI >= 2 && NI % 2 == 0 && 2 * MI * NI <= 16, "tile too large for registers");
    static constexpr int A_CHUNKS = A_ROWS * 4 / NTHR, B_CHUNKS = BN * 4 / NTHR;        // 16-byte chunks per thread per stage
    static_assert(A_ROWS * 4 % NTHR == 0 && BN * 4 % NTHR == 0, "loader mapping");
};

__device__ __forceinline__ int swz(int row, int chunk) { return row * BK + ((chunk ^ ((row >> 1) & 3)) << 4); }   // 16-byte chunk address, bank-conflict free for ldmatrix
__device__ __forceinline__ void cp16z(void* sm, const void* gm, bool valid) {
    unsigned sa = unsigned(__cvta_generic_to_shared(sm)); int sz = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(sa), "l"(gm), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }
__device__ __forceinline__ void ldm4(uint32_t* r, const void* p) {
    unsigned a = unsigned(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma_s8(int32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int epi1(int chi, int clo, const int4 c, int er) {
    i64 acc = 256LL * chi + clo + 128LL * c.w;
    return int(sat32(rshr(acc * c.x, c.y - er) + c.z));
}

__device__ __forceinline__ int rowexp_u32(uint32_t m) { return m ? max(0, 32 - __clz(m) - 15) : 0; }
/** quantise 16 int32 values with shift e into packed hi/lo bytes (4 x u32 each) */
__device__ __forceinline__ void quant16x(const int32_t* v, int e, uint32_t* hi, uint32_t* lo) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        int q0 = quant16_32(v[4 * j], e), q1 = quant16_32(v[4 * j + 1], e), q2 = quant16_32(v[4 * j + 2], e), q3 = quant16_32(v[4 * j + 3], e);
        hi[j] = uint32_t(uint8_t(q0 >> 8)) | (uint32_t(uint8_t(q1 >> 8)) << 8) | (uint32_t(uint8_t(q2 >> 8)) << 16) | (uint32_t(uint8_t(q3 >> 8)) << 24);
        lo[j] = uint32_t(uint8_t((q0 & 255) - 128)) | (uint32_t(uint8_t((q1 & 255) - 128)) << 8) | (uint32_t(uint8_t((q2 & 255) - 128)) << 16) | (uint32_t(uint8_t((q3 & 255) - 128)) << 24);
    }
}

template <int EPI, int SRC, class C>
__global__ void __launch_bounds__(NTHR) gemm_hl_k(const ASrc A, int Mreal, const int8_t* __restrict__ W, int ldw, int Nreal, int Kp, const int4* __restrict__ cst, const AEpi E) {
    extern __shared__ __align__(128) unsigned char smem[];
    constexpr int BM = C::BM, BN = C::BN, STAGES = C::STAGES, MI = C::MI, NI = C::NI;
    static_assert(SRC == SRC_I8 || BM == 64, "register-path loaders assume 64 output rows per block (one 16-wide chunk per thread per stage)");
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, wm = w % C::WM, wn = w / C::WM;
    const int gid = lane >> 2, t4 = lane & 3, lm = lane >> 3, lr = lane & 7;
    const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
    const int nK = (Kp + BK - 1) / BK;
    const int32_t* __restrict__ e = A.e;
    // ---- B (weights): always cp.async
    auto load_B = [&](int stage, int kt) {
        unsigned char* sB = smem + stage * C::STAGE_BYTES + C::A_BYTES;
        const int k0 = kt * BK;
        #pragma unroll
        for (int i = 0; i < C::B_CHUNKS; ++i) {
            const int idx = tid + i * NTHR, row = idx >> 2, c = idx & 3;
            const int kc = k0 + c * 16; const bool valid = kc < Kp;
            cp16z(sB + swz(row, c), W + size_t(n0 + row) * ldw + (valid ? kc : 0), valid);
        }
    };
    // ---- A: planar int8 via cp.async
    auto load_A8 = [&](int stage, int kt) {
        unsigned char* sA = smem + stage * C::STAGE_BYTES;
        const int k0 = kt * BK;
        #pragma unroll
        for (int i = 0; i < C::A_CHUNKS; ++i) {
            const int idx = tid + i * NTHR, row = idx >> 2, c = idx & 3;
            const int kc = k0 + c * 16; const bool valid = kc < Kp;
            const int8_t* src = (row < BM ? A.hi + size_t(m0 + row) * A.ld8 : A.lo + size_t(m0 + row - BM) * A.ld8) + (valid ? kc : 0);
            cp16z(sA + swz(row, c), src, valid);
        }
    };
    // ---- A: register path (SRC_X32 / SRC_CONV32): this thread owns output row (tid>>2), chunk (tid&3) of every stage
    const int arow = tid >> 2, achunk = tid & 3;
    const int grow = m0 + arow;
    int er_a = 0;                                    // quantisation shift of this thread's row
    if (SRC == SRC_X32) er_a = grow < Mreal ? rowexp_u32(A.rowmax[grow]) : 0;
    if (SRC == SRC_CONV32) er_a = grow < Mreal ? e[grow] : 0;
    int cb = 0, ciy0 = 0, cix0 = 0;                  // conv: sample and top-left input coords of this output pixel
    if (SRC == SRC_CONV32) { const int pp = grow % (A.Ho * A.Wo); cb = grow / (A.Ho * A.Wo); const int oy = pp / A.Wo, ox = pp - oy * A.Wo; ciy0 = oy * A.stride - 1; cix0 = ox * A.stride - 1; }
    auto load_raw = [&](int kt, int4* raw) {
        const int k = kt * BK + achunk * 16;
        const int32_t* src = nullptr;
        if (SRC == SRC_X32) { if (grow < Mreal && k < A.K) src = A.x + size_t(grow) * A.ldx + k; }
        else { if (grow < Mreal && k < A.K) { const int tap = k / A.C, c0 = k - tap * A.C, ky = tap / 3, kx = tap - ky * 3, iy = ciy0 + ky, ix = cix0 + kx;
                 if (iy >= 0 && iy < A.H && ix >= 0 && ix < A.W) src = A.img + ((size_t(cb) * A.H + iy) * A.W + ix) * A.C + c0; } }
        if (src) { raw[0] = *reinterpret_cast<const int4*>(src); raw[1] = *reinterpret_cast<const int4*>(src + 4); raw[2] = *reinterpret_cast<const int4*>(src + 8); raw[3] = *reinterpret_cast<const int4*>(src + 12); }
        else raw[0] = raw[1] = raw[2] = raw[3] = make_int4(0, 0, 0, 0);
    };
    auto store_quant = [&](int stage, const int4* raw) {
        unsigned char* sA = smem + stage * C::STAGE_BYTES;
        uint32_t hi[4], lo[4];
        quant16x(reinterpret_cast<const int32_t*>(raw), er_a, hi, lo);
        *reinterpret_cast<uint4*>(sA + swz(arow, achunk)) = make_uint4(hi[0], hi[1], hi[2], hi[3]);
        *reinterpret_cast<uint4*>(sA + swz(BM + arow, achunk)) = make_uint4(lo[0], lo[1], lo[2], lo[3]);
    };
    int32_t acc[2][MI][NI][4];
    #pragma unroll
    for (int p = 0; p < 2; ++p) for (int mi = 0; mi < MI; ++mi) for (int ni = 0; ni < NI; ++ni) for (int j = 0; j < 4; ++j) acc[p][mi][ni][j] = 0;
    int4 raw0[4], raw1[4];                            // two in-flight raw stages (static names: keep them in registers)
    if (SRC == SRC_I8) {
        #pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) { if (s < nK) { load_A8(s, s); load_B(s, s); } cp_commit(); }
    } else {
        { int4 r0[4]; load_raw(0, r0); store_quant(0, r0); }
        load_raw(1, raw0); load_raw(2, raw1);
        #pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) { if (s < nK) load_B(s, s); cp_commit(); }
    }
    auto compute = [&](int kt) {
        const unsigned char* sA = smem + (kt % STAGES) * C::STAGE_BYTES; const unsigned char* sB = sA + C::A_BYTES;
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            uint32_t af[2][MI][4];
            #pragma unroll
            for (int p = 0; p < 2; ++p)
                #pragma unroll
                for (int mi = 0; mi < MI; ++mi) {
                    const int row = p * BM + wm * C::WTM + mi * 16 + (lm & 1) * 8 + lr, chunk = ks * 2 + (lm >> 1);
                    ldm4(af[p][mi], sA + swz(row, chunk));
                }
            #pragma unroll
            for (int np = 0; np < NI / 2; ++np) {
                uint32_t bf[4];
                const int nt = wn * NI + np * 2 + (lm >> 1), row = nt * 8 + lr, chunk = ks * 2 + (lm & 1);
                ldm4(bf, sB + swz(row, chunk));
                #pragma unroll
                for (int p = 0; p < 2; ++p)
                    #pragma unroll
                    for (int mi = 0; mi < MI; ++mi) { mma_s8(acc[p][mi][np * 2], af[p][mi], bf[0], bf[1]); mma_s8(acc[p][mi][np * 2 + 1], af[p][mi], bf[2], bf[3]); }
            }
        }
    };
    auto step = [&](int kt, int4* rw) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        if (SRC == SRC_I8) { const int nx = kt + STAGES - 1; if (nx < nK) { load_A8(nx % STAGES, nx); load_B(nx % STAGES, nx); } cp_commit(); }
        else {
            if (kt + 1 < nK) store_quant((kt + 1) % STAGES, rw);          // stage kt+1 (loaded two iterations ago)
            if (kt + 3 < nK) load_raw(kt + 3, rw);
            { const int nx = kt + STAGES - 1; if (nx < nK) load_B(nx % STAGES, nx); cp_commit(); }
        }
        compute(kt);
    };
    for (int kt = 0; kt < nK; kt += 2) { step(kt, raw0); if (kt + 1 < nK) step(kt + 1, raw1); }
    cp_wait<0>();
    int32_t* __restrict__ Y = E.Y; int32_t* __restrict__ Y1 = E.Y1; int32_t* __restrict__ Y2 = E.Y2; const int ldy = E.ldy; const int32_t* __restrict__ R = E.residual; const int Cs = E.Cs; const int32_t* __restrict__ Lphi = E.Lphi;
    #pragma unroll
    for (int mi = 0; mi < MI; ++mi) {
        const int r0 = m0 + wm * C::WTM + mi * 16 + gid, r1 = r0 + 8;
        int e0 = 0, e1 = 0;
        if (SRC == SRC_X32) { e0 = r0 < Mreal ? rowexp_u32(A.rowmax[r0]) : 0; e1 = r1 < Mreal ? rowexp_u32(A.rowmax[r1]) : 0; }
        else { e0 = r0 < Mreal ? e[r0] : 0; e1 = r1 < Mreal ? e[r1] : 0; }
        uint32_t mx0 = 0, mx1 = 0;
        #pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
            const int n = n0 + (wn * NI + ni) * 8 + t4 * 2;
            if (n >= Nreal) continue;
            const int4 c0 = cst[n], c1 = cst[n + 1];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = h ? r1 : r0; if (r >= Mreal) continue;
                const int er = h ? e1 : e0;
                int y0 = epi1(acc[0][mi][ni][2 * h], acc[1][mi][ni][2 * h], c0, er), y1 = epi1(acc[0][mi][ni][2 * h + 1], acc[1][mi][ni][2 * h + 1], c1, er);
                if (EPI == EPI_PLAIN || EPI == EPI_GELU) {
                    if (EPI == EPI_GELU) { y0 = int(sat32(rshr(i64(y0) * lut16i(y0, Lphi), 16))); y1 = int(sat32(rshr(i64(y1) * lut16i(y1, Lphi), 16))); }
                    int32_t* dst = Y + size_t(r) * ldy + n;
                    if (R) { const int2 rr = *reinterpret_cast<const int2*>(R + size_t(r) * ldy + n); y0 = int(sat32(i64(y0) + rr.x)); y1 = int(sat32(i64(y1) + rr.y)); }
                    *reinterpret_cast<int2*>(dst) = make_int2(y0, y1);
                    if (E.rowmax) { const uint32_t m = max(uint32_t(y0 < 0 ? -i64(y0) : y0), uint32_t(y1 < 0 ? -i64(y1) : y1)); if (h) mx1 = max(mx1, m); else mx0 = max(mx0, m); }
                } else if (EPI == EPI_SPLIT3) {
                    const int part = n >= 2 * Cs ? 2 : (n >= Cs ? 1 : 0); int32_t* dst = (part == 0 ? Y : (part == 1 ? Y1 : Y2)) + size_t(r) * Cs + (n - part * Cs);
                    *reinterpret_cast<int2*>(dst) = make_int2(y0, y1);
                } else {
                    const i64 ge = sat32(rshr(i64(y1) * lut16i(y1, Lphi), 16));
                    const int v = int(sat32(rshr(i64(y0) * ge, 16)));
                    Y[size_t(r) * ldy + (n >> 1)] = v;
                    if (E.rowmax) { const uint32_t m = uint32_t(v < 0 ? -i64(v) : v); if (h) mx1 = max(mx1, m); else mx0 = max(mx0, m); }
                }
            }
        }
        if (E.rowmax) {      // reduce over the 4 lanes sharing a row, one atomic per row per warp
            for (int o = 1; o < 4; o <<= 1) { mx0 = max(mx0, __shfl_xor_sync(0xffffffffu, mx0, o)); mx1 = max(mx1, __shfl_xor_sync(0xffffffffu, mx1, o)); }
            if (t4 == 0) { if (r0 < Mreal && mx0) atomicMax(E.rowmax + r0, mx0); if (r1 < Mreal && mx1) atomicMax(E.rowmax + r1, mx1); }
        }
    }
}
typedef Cfg<64, 64, 4, 4> CfgSmall;     // 64x64 tile, 48 KB smem, 2 blocks/SM   (small K)
typedef Cfg<128, 64, 4, 4> CfgTall;     // 128x64 tile, 80 KB smem, 1 block/SM   (large K: more reuse per byte)
typedef Cfg<64, 128, 3, 2> CfgWide;     // 64x128 tile, 48 KB
template <int EPI, int SRC, class C> void launch(dim3 grid, cudaStream_t st, const ASrc& A, int Mreal, const int8_t* W, int ldw, int Nreal, int Kp, const int4* cst, const AEpi& E) {
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(gemm_hl_k<EPI, SRC, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM); attr = true; }
    gemm_hl_k<EPI, SRC, C><<<grid, NTHR, C::SMEM, st>>>(A, Mreal, W, ldw, Nreal, Kp, cst, E);
}
template <int SRC, class C> void launch_epi(int epi, dim3 grid, cudaStream_t st, const ASrc& A, int Mreal, const int8_t* W, int ldw, int Nreal, int Kp, const int4* cst, const AEpi& E) {
    switch (epi) {
        case EPI_SPLIT3: launch<EPI_SPLIT3, SRC, C>(grid, st, A, Mreal, W, ldw, Nreal, Kp, cst, E); break;
        case EPI_GEGLU: launch<EPI_GEGLU, SRC, C>(grid, st, A, Mreal, W, ldw, Nreal, Kp, cst, E); break;
        case EPI_GELU: launch<EPI_GELU, SRC, C>(grid, st, A, Mreal, W, ldw, Nreal, Kp, cst, E); break;
        default: launch<EPI_PLAIN, SRC, C>(grid, st, A, Mreal, W, ldw, Nreal, Kp, cst, E);
    }
}
__global__ void conv_e_k(const uint32_t* __restrict__ pm, int B, int H, int W, int stride, int Ho, int Wo, int32_t* __restrict__ e) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x; if (r >= B * Ho * Wo) return;
    const int pp = r % (Ho * Wo), b = r / (Ho * Wo), oy = pp / Wo, ox = pp - oy * Wo;
    uint32_t m = 0;
    for (int ky = 0; ky < 3; ++ky) for (int kx = 0; kx < 3; ++kx) { const int iy = oy * stride - 1 + ky, ix = ox * stride - 1 + kx; if (iy >= 0 && iy < H && ix >= 0 && ix < W) m = max(m, pm[(size_t(b) * H + iy) * W + ix]); }
    e[r] = rowexp_u32(m);
}
}  // namespace

bool gemm_hl_supported(int Mp, int Np, int Kp) { return Mp % 64 == 0 && Np % 64 == 0 && Kp % 16 == 0; }
int g_gemm_hl_variant = -1;     // -1 auto, 0 small, 1 tall, 2 wide (benchmark override)
void k_conv_e(cudaStream_t st, const uint32_t* pm, int B, int H, int W, int stride, int32_t* e) {
    const int Ho = (H - 1) / stride + 1, Wo = (W - 1) / stride + 1; const int rows = B * Ho * Wo;
    conv_e_k<<<(rows + 255) / 256, 256, 0, st>>>(pm, B, H, W, stride, Ho, Wo, e);
}
void k_gemm_hl2(cudaStream_t st, const ASrc& A, int Mp, int Mreal, const int8_t* W, int ldw, int Np, int Nreal, int Kp, const int4* cst, const AEpi& E) {
    if (!gemm_hl_supported(Mp, Np, Kp)) throw std::runtime_error("gemm_hl: unsupported shape");
    int v = g_gemm_hl_variant;
    if (v < 0) v = Kp >= 2048 ? (Np % 128 == 0 ? 2 : (Mp % 128 == 0 ? 1 : 0)) : 0;   // deep K: bigger tiles (more reuse); shallow K: 64x64 (2 blocks/SM)
    if (v == 1 && Mp % 128) v = 0;
    if (v == 2 && Np % 128) v = 0;
    if (A.src != SRC_I8) v = v == 1 ? 0 : v;                                           // register-path loaders need 64 output rows per block
    if (A.src == SRC_CONV32 && (A.C % 16 || A.K != 9 * A.C)) throw std::runtime_error("gemm_hl conv source: C % 16 or K");
    if (A.src == SRC_I8) {
        if (v == 1) { dim3 grid(Np / 64, Mp / 128); launch_epi<SRC_I8, CfgTall>(E.epi, grid, st, A, Mreal, W, ldw, Nreal, Kp, cst, E); }
        else if (v == 2) { dim3 grid(Np / 128, Mp / 64); launch_epi<SRC_I8, CfgWide>(E.epi, grid, st, A, Mreal, W, ldw, Nreal, Kp, cst, E); }
        else { dim3 grid(Np / 64, Mp / 64); launch_epi<SRC_I8, CfgSmall>(E.epi, grid, st, A, Mreal, W, ldw, Nreal, Kp, cst, E); }
    } else if (A.src == SRC_X32) {
        if (v == 2) { dim3 grid(Np / 128, Mp / 64); launch_epi<SRC_X32, CfgWide>(E.epi, grid, st, A, Mreal, W, ldw, Nreal, Kp, cst, E); }
        else { dim3 grid(Np / 64, Mp / 64); launch_epi<SRC_X32, CfgSmall>(E.epi, grid, st, A, Mreal, W, ldw, Nreal, Kp, cst, E); }
    } else {
        if (v == 2) { dim3 grid(Np / 128, Mp / 64); launch_epi<SRC_CONV32, CfgWide>(E.epi, grid, st, A, Mreal, W, ldw, Nreal, Kp, cst, E); }
        else { dim3 grid(Np / 64, Mp / 64); launch_epi<SRC_CONV32, CfgSmall>(E.epi, grid, st, A, Mreal, W, ldw, Nreal, Kp, cst, E); }
    }
}
void k_gemm_hl(cudaStream_t st, const int8_t* Ahi, const int8_t* Alo, int lda, int Mp, int Mreal, const int8_t* W, int ldw, int Np, int Nreal, int Kp,
               const int4* cst, const int32_t* e, int32_t* Y, int ldy, const int32_t* residual, int epi, int Cs, int32_t* Y1, int32_t* Y2, const int32_t* Lphi) {
    ASrc A; A.src = SRC_I8; A.hi = Ahi; A.lo = Alo; A.ld8 = lda; A.e = e;
    AEpi E; E.epi = epi; E.Y = Y; E.ldy = ldy; E.residual = residual; E.Cs = Cs; E.Y1 = Y1; E.Y2 = Y2; E.Lphi = Lphi;
    k_gemm_hl2(st, A, Mp, Mreal, W, ldw, Np, Nreal, Kp, cst, E);
}
}  // namespace meow
