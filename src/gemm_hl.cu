#include "gemm_hl.h"
#include "intops.cuh"
#include <stdexcept>

namespace meow {
namespace {
constexpr int BM = 64;          // output rows per block (A tile = 64 hi rows + 64 lo rows)
constexpr int BN = 64, BK = 64, STAGES = 4, NTHR = 256;
constexpr int A_ROWS = 2 * BM;  // 128 int8 rows of 64 bytes
constexpr int A_BYTES = A_ROWS * BK, B_BYTES = BN * BK, STAGE_BYTES = A_BYTES + B_BYTES;

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

template <int EPI>
__global__ void __launch_bounds__(NTHR, 2) gemm_hl_k(const int8_t* __restrict__ Ahi, const int8_t* __restrict__ Alo, int lda, int Mreal,
                                                     const int8_t* __restrict__ W, int ldw, int Nreal, int Kp,
                                                     const int4* __restrict__ cst, const int32_t* __restrict__ e, int32_t* __restrict__ Y, int ldy, const int32_t* __restrict__ R,
                                                     int Cs, int32_t* __restrict__ Y1, int32_t* __restrict__ Y2, const int32_t* __restrict__ Lphi) {
    extern __shared__ __align__(128) unsigned char smem[];
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, wm = w & 3, wn = w >> 2;
    const int gid = lane >> 2, t4 = lane & 3, lm = lane >> 3, lr = lane & 7;
    const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
    const int nK = (Kp + BK - 1) / BK;
    // tile loader: A rows 0..63 = hi rows m0.., 64..127 = lo rows m0..; B rows = W rows n0..
    auto load_stage = [&](int stage, int kt) {
        unsigned char* sA = smem + stage * STAGE_BYTES; unsigned char* sB = sA + A_BYTES;
        const int k0 = kt * BK;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid + i * NTHR, row = idx >> 2, c = idx & 3;         // 512 chunks
            const int kc = k0 + c * 16; const bool valid = kc < Kp;
            const int8_t* src = (row < BM ? Ahi + size_t(m0 + row) * lda : Alo + size_t(m0 + row - BM) * lda) + (valid ? kc : 0);
            cp16z(sA + swz(row, c), src, valid);
        }
        {
            const int row = tid >> 2, c = tid & 3;                                 // 256 chunks
            const int kc = k0 + c * 16; const bool valid = kc < Kp;
            cp16z(sB + swz(row, c), W + size_t(n0 + row) * ldw + (valid ? kc : 0), valid);
        }
    };
    int32_t acc[2][4][4];
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) for (int ni = 0; ni < 4; ++ni) for (int j = 0; j < 4; ++j) acc[mi][ni][j] = 0;
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) { if (s < nK) load_stage(s, s); cp_commit(); }
    for (int kt = 0; kt < nK; ++kt) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        { const int nx = kt + STAGES - 1; if (nx < nK) load_stage(nx % STAGES, nx); cp_commit(); }
        const unsigned char* sA = smem + (kt % STAGES) * STAGE_BYTES; const unsigned char* sB = sA + A_BYTES;
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            uint32_t af[2][4];
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {              // mi 0: hi rows, mi 1: lo rows (same output rows)
                const int row = mi * BM + wm * 16 + (lm & 1) * 8 + lr, chunk = ks * 2 + (lm >> 1);
                ldm4(af[mi], sA + swz(row, chunk));
            }
            #pragma unroll
            for (int np = 0; np < 2; ++np) {              // n-tile pairs: 4 n-tiles of 8 per warp
                uint32_t bf[4];
                const int nt = wn * 4 + np * 2 + (lm >> 1), row = nt * 8 + lr, chunk = ks * 2 + (lm & 1);
                ldm4(bf, sB + swz(row, chunk));
                mma_s8(acc[0][np * 2], af[0], bf[0], bf[1]); mma_s8(acc[0][np * 2 + 1], af[0], bf[2], bf[3]);
                mma_s8(acc[1][np * 2], af[1], bf[0], bf[1]); mma_s8(acc[1][np * 2 + 1], af[1], bf[2], bf[3]);
            }
        }
    }
    cp_wait<0>();
    // epilogue: thread holds rows (wm*16 + gid, +8), cols nt*8 + t4*2 (+1) for nt = wn*4 + ni; hi = acc[0], lo = acc[1]
    const int r0 = m0 + wm * 16 + gid, r1 = r0 + 8;
    const int e0 = r0 < Mreal ? e[r0] : 0, e1 = r1 < Mreal ? e[r1] : 0;
    #pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
        const int n = n0 + (wn * 4 + ni) * 8 + t4 * 2;
        if (n >= Nreal) continue;
        const int4 c0 = cst[n], c1 = cst[n + 1];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = h ? r1 : r0; if (r >= Mreal) continue;
            const int er = h ? e1 : e0;
            int y0 = epi1(acc[0][ni][2 * h], acc[1][ni][2 * h], c0, er), y1 = epi1(acc[0][ni][2 * h + 1], acc[1][ni][2 * h + 1], c1, er);
            if (EPI == EPI_PLAIN) {
                int32_t* dst = Y + size_t(r) * ldy + n;
                if (R) { const int2 rr = *reinterpret_cast<const int2*>(R + size_t(r) * ldy + n); y0 = int(sat32(i64(y0) + rr.x)); y1 = int(sat32(i64(y1) + rr.y)); }
                *reinterpret_cast<int2*>(dst) = make_int2(y0, y1);
            } else if (EPI == EPI_SPLIT3) {              // column n -> Y[n / Cs][r][n % Cs]  (Cs multiple of 8 keeps the pair in one part)
                const int part = n >= 2 * Cs ? 2 : (n >= Cs ? 1 : 0); int32_t* dst = (part == 0 ? Y : (part == 1 ? Y1 : Y2)) + size_t(r) * Cs + (n - part * Cs);
                *reinterpret_cast<int2*>(dst) = make_int2(y0, y1);
            } else {                                     // EPI_GEGLU: (y0, y1) = (a_c, g_c), c = n/2
                const i64 ge = sat32(rshr(i64(y1) * lut16i(y1, Lphi), 16));
                Y[size_t(r) * ldy + (n >> 1)] = int(sat32(rshr(i64(y0) * ge, 16)));
            }
        }
    }
}
}  // namespace

bool gemm_hl_supported(int Mp, int Np, int Kp) { return Mp % BM == 0 && Np % BN == 0 && Kp % 16 == 0; }
void k_gemm_hl(cudaStream_t st, const int8_t* Ahi, const int8_t* Alo, int lda, int Mp, int Mreal, const int8_t* W, int ldw, int Np, int Nreal, int Kp,
               const int4* cst, const int32_t* e, int32_t* Y, int ldy, const int32_t* residual, int epi, int Cs, int32_t* Y1, int32_t* Y2, const int32_t* Lphi) {
    if (!gemm_hl_supported(Mp, Np, Kp)) throw std::runtime_error("gemm_hl: unsupported shape");
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(gemm_hl_k<EPI_PLAIN>, cudaFuncAttributeMaxDynamicSharedMemorySize, STAGES * STAGE_BYTES); cudaFuncSetAttribute(gemm_hl_k<EPI_SPLIT3>, cudaFuncAttributeMaxDynamicSharedMemorySize, STAGES * STAGE_BYTES); cudaFuncSetAttribute(gemm_hl_k<EPI_GEGLU>, cudaFuncAttributeMaxDynamicSharedMemorySize, STAGES * STAGE_BYTES); attr = true; }
    dim3 grid(Np / BN, Mp / BM);
    if (epi == EPI_PLAIN) gemm_hl_k<EPI_PLAIN><<<grid, NTHR, STAGES * STAGE_BYTES, st>>>(Ahi, Alo, lda, Mreal, W, ldw, Nreal, Kp, cst, e, Y, ldy, residual, Cs, Y1, Y2, Lphi);
    else if (epi == EPI_SPLIT3) gemm_hl_k<EPI_SPLIT3><<<grid, NTHR, STAGES * STAGE_BYTES, st>>>(Ahi, Alo, lda, Mreal, W, ldw, Nreal, Kp, cst, e, Y, ldy, residual, Cs, Y1, Y2, Lphi);
    else gemm_hl_k<EPI_GEGLU><<<grid, NTHR, STAGES * STAGE_BYTES, st>>>(Ahi, Alo, lda, Mreal, W, ldw, Nreal, Kp, cst, e, Y, ldy, residual, Cs, Y1, Y2, Lphi);
}
}  // namespace meow
