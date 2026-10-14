#include "gemm_hl.h"
#include "intops.cuh"
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

template <int EPI, class C>
__global__ void __launch_bounds__(NTHR) gemm_hl_k(const int8_t* __restrict__ Ahi, const int8_t* __restrict__ Alo, int lda, int Mreal,
                                                  const int8_t* __restrict__ W, int ldw, int Nreal, int Kp,
                                                  const int4* __restrict__ cst, const int32_t* __restrict__ e, int32_t* __restrict__ Y, int ldy, const int32_t* __restrict__ R,
                                                  int Cs, int32_t* __restrict__ Y1, int32_t* __restrict__ Y2, const int32_t* __restrict__ Lphi) {
    extern __shared__ __align__(128) unsigned char smem[];
    constexpr int BM = C::BM, BN = C::BN, STAGES = C::STAGES, MI = C::MI, NI = C::NI;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, wm = w % C::WM, wn = w / C::WM;
    const int gid = lane >> 2, t4 = lane & 3, lm = lane >> 3, lr = lane & 7;
    const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
    const int nK = (Kp + BK - 1) / BK;
    auto load_stage = [&](int stage, int kt) {
        unsigned char* sA = smem + stage * C::STAGE_BYTES; unsigned char* sB = sA + C::A_BYTES;
        const int k0 = kt * BK;
        #pragma unroll
        for (int i = 0; i < C::A_CHUNKS; ++i) {
            const int idx = tid + i * NTHR, row = idx >> 2, c = idx & 3;
            const int kc = k0 + c * 16; const bool valid = kc < Kp;
            const int8_t* src = (row < BM ? Ahi + size_t(m0 + row) * lda : Alo + size_t(m0 + row - BM) * lda) + (valid ? kc : 0);
            cp16z(sA + swz(row, c), src, valid);
        }
        #pragma unroll
        for (int i = 0; i < C::B_CHUNKS; ++i) {
            const int idx = tid + i * NTHR, row = idx >> 2, c = idx & 3;
            const int kc = k0 + c * 16; const bool valid = kc < Kp;
            cp16z(sB + swz(row, c), W + size_t(n0 + row) * ldw + (valid ? kc : 0), valid);
        }
    };
    int32_t acc[2][MI][NI][4];
    #pragma unroll
    for (int p = 0; p < 2; ++p) for (int mi = 0; mi < MI; ++mi) for (int ni = 0; ni < NI; ++ni) for (int j = 0; j < 4; ++j) acc[p][mi][ni][j] = 0;
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) { if (s < nK) load_stage(s, s); cp_commit(); }
    for (int kt = 0; kt < nK; ++kt) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        { const int nx = kt + STAGES - 1; if (nx < nK) load_stage(nx % STAGES, nx); cp_commit(); }
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
    }
    cp_wait<0>();
    #pragma unroll
    for (int mi = 0; mi < MI; ++mi) {
        const int r0 = m0 + wm * C::WTM + mi * 16 + gid, r1 = r0 + 8;
        const int e0 = r0 < Mreal ? e[r0] : 0, e1 = r1 < Mreal ? e[r1] : 0;
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
                if (EPI == EPI_PLAIN) {
                    int32_t* dst = Y + size_t(r) * ldy + n;
                    if (R) { const int2 rr = *reinterpret_cast<const int2*>(R + size_t(r) * ldy + n); y0 = int(sat32(i64(y0) + rr.x)); y1 = int(sat32(i64(y1) + rr.y)); }
                    *reinterpret_cast<int2*>(dst) = make_int2(y0, y1);
                } else if (EPI == EPI_SPLIT3) {
                    const int part = n >= 2 * Cs ? 2 : (n >= Cs ? 1 : 0); int32_t* dst = (part == 0 ? Y : (part == 1 ? Y1 : Y2)) + size_t(r) * Cs + (n - part * Cs);
                    *reinterpret_cast<int2*>(dst) = make_int2(y0, y1);
                } else {
                    const i64 ge = sat32(rshr(i64(y1) * lut16i(y1, Lphi), 16));
                    Y[size_t(r) * ldy + (n >> 1)] = int(sat32(rshr(i64(y0) * ge, 16)));
                }
            }
        }
    }
}
typedef Cfg<64, 64, 4, 4> CfgSmall;     // 64x64 tile, 48 KB smem, 2 blocks/SM   (small K)
typedef Cfg<128, 64, 4, 4> CfgTall;     // 128x64 tile, 80 KB smem, 1 block/SM   (large K: more reuse per byte)
typedef Cfg<64, 128, 3, 2> CfgWide;     // 64x128 tile, 48 KB
template <int EPI, class C> void launch(dim3 grid, cudaStream_t st, const int8_t* Ahi, const int8_t* Alo, int lda, int Mreal, const int8_t* W, int ldw, int Nreal, int Kp,
                                        const int4* cst, const int32_t* e, int32_t* Y, int ldy, const int32_t* R, int Cs, int32_t* Y1, int32_t* Y2, const int32_t* Lphi) {
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(gemm_hl_k<EPI, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM); attr = true; }
    gemm_hl_k<EPI, C><<<grid, NTHR, C::SMEM, st>>>(Ahi, Alo, lda, Mreal, W, ldw, Nreal, Kp, cst, e, Y, ldy, R, Cs, Y1, Y2, Lphi);
}
}  // namespace

bool gemm_hl_supported(int Mp, int Np, int Kp) { return Mp % 64 == 0 && Np % 64 == 0 && Kp % 16 == 0; }
int g_gemm_hl_variant = -1;     // -1 auto, 0 small, 1 tall, 2 wide (benchmark override)
void k_gemm_hl(cudaStream_t st, const int8_t* Ahi, const int8_t* Alo, int lda, int Mp, int Mreal, const int8_t* W, int ldw, int Np, int Nreal, int Kp,
               const int4* cst, const int32_t* e, int32_t* Y, int ldy, const int32_t* residual, int epi, int Cs, int32_t* Y1, int32_t* Y2, const int32_t* Lphi) {
    if (!gemm_hl_supported(Mp, Np, Kp)) throw std::runtime_error("gemm_hl: unsupported shape");
    int v = g_gemm_hl_variant;
    if (v < 0) v = Kp >= 2048 ? (Np % 128 == 0 ? 2 : (Mp % 128 == 0 ? 1 : 0)) : 0;   // deep K: bigger tiles (more reuse); shallow K: 64x64 (2 blocks/SM)
    if (v == 1 && Mp % 128) v = 0;
    if (v == 2 && Np % 128) v = 0;
#define GEMM_HL_DISPATCH(C, BMv, BNv) { dim3 grid(Np / BNv, Mp / BMv); \
        if (epi == EPI_PLAIN) launch<EPI_PLAIN, C>(grid, st, Ahi, Alo, lda, Mreal, W, ldw, Nreal, Kp, cst, e, Y, ldy, residual, Cs, Y1, Y2, Lphi); \
        else if (epi == EPI_SPLIT3) launch<EPI_SPLIT3, C>(grid, st, Ahi, Alo, lda, Mreal, W, ldw, Nreal, Kp, cst, e, Y, ldy, residual, Cs, Y1, Y2, Lphi); \
        else launch<EPI_GEGLU, C>(grid, st, Ahi, Alo, lda, Mreal, W, ldw, Nreal, Kp, cst, e, Y, ldy, residual, Cs, Y1, Y2, Lphi); }
    if (v == 1) GEMM_HL_DISPATCH(CfgTall, 128, 64)
    else if (v == 2) GEMM_HL_DISPATCH(CfgWide, 64, 128)
    else GEMM_HL_DISPATCH(CfgSmall, 64, 64)
#undef GEMM_HL_DISPATCH
}
}  // namespace meow
