// Custom int8 GEMM with the hi/lo split accumulated in-register and the exact PoI epilogue fused (gemm_hl.cu).
//   Y[r, n] = sat32(rshr((256*Ahi[r]·W[n] + Alo[r]·W[n] + 128*wsum[n]) * m[n], s[n] - e[r]) + b[n])  (+ residual, saturated)
// A planes: Ahi/Alo [Mp rows, ld lda] (Mp multiple of 64), W [Np rows, ld ldw] (Np multiple of 64), K = Kp (multiple of 16).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
namespace meow {
enum GemmEpi { EPI_PLAIN = 0, EPI_SPLIT3 = 1, EPI_GEGLU = 2, EPI_GELU = 3, EPI_QKV = 4 };   // QKV: q,k written head-quantised (k_quant_heads layout), v as int32 (64x64 tiles only)   // GELU: y = sat32(rshr(y*phi(y),16)) (text fc1)
enum GemmSrc { SRC_I8 = 0, SRC_X32 = 1, SRC_CONV32 = 2 };
/** A operand description. SRC_I8: planar int8 hi/lo rows (ld8) + per-row e. SRC_X32: int32 X[row, k] (ldx) quantised on load with
 *  e = row_exp(rowmax[row]). SRC_CONV32: NHWC int32 image, implicit 3x3/pad-1 im2col (K = 9C, C % 16 == 0), e[row] precomputed (k_conv_e). */
struct ASrc {
    int src = SRC_I8;
    const int8_t* hi = nullptr; const int8_t* lo = nullptr; int ld8 = 0; const int32_t* e = nullptr;
    const int32_t* x = nullptr; int ldx = 0; const uint32_t* rowmax = nullptr; int K = 0;
    const int32_t* img = nullptr; int H = 0, W = 0, C = 0, stride = 1, Ho = 0, Wo = 0;
};
struct AEpi {
    int epi = EPI_PLAIN; int32_t* Y = nullptr; int ldy = 0; const int32_t* residual = nullptr;
    int Cs = 0; int32_t* Y1 = nullptr; int32_t* Y2 = nullptr; const int32_t* Lphi = nullptr;
    uint32_t* rowmax = nullptr;              // optional: atomicMax of |y| per output row (feeds a SRC_X32 consumer)
    void* gnstat = nullptr; int gnT = 0;     // optional (EPI_PLAIN, 64x64 tiles, Nreal % 32 == 0): accumulate GroupNorm sum/min/max per (sample, group) into GnStat[b*32+g]
    // EPI_QKV: N = 3*C, C = heads*64, rows = B*T. q -> (qhi,qlo,qe,qhs,qls,qts) with Tp = TpQ, k -> (khi,...) with Tp = TpK, v -> Y2 [rows, C]
    int heads = 0, T = 0, TpQ = 0, TpK = 0, C = 0;
    int8_t *qhi = nullptr, *qlo = nullptr, *khi = nullptr, *klo = nullptr; int32_t *qe = nullptr, *ke = nullptr; int64_t *qhs = nullptr, *qls = nullptr, *qts = nullptr, *khs = nullptr, *kls = nullptr, *kts = nullptr;
};
void k_gemm_hl2(cudaStream_t st, const ASrc& A, int Mp, int Mreal, const int8_t* W, int ldw, int Np, int Nreal, int Kp, const int4* cst, const AEpi& E);
/** e[row] = row_exp(max over the 3x3 window of pm) for the implicit-im2col source (rows = B*Ho*Wo). */
void k_conv_e(cudaStream_t st, const uint32_t* pm, int B, int H, int W, int stride, int32_t* e);   // GEGLU: W rows interleaved (2c = a_c, 2c+1 = g_c), writes sat32(rshr(a*gelu(g),16)) to Y[r, c] (ldy), Lphi = int32 phi LUT
void k_gemm_hl(cudaStream_t st, const int8_t* Ahi, const int8_t* Alo, int lda, int Mp, int Mreal,
               const int8_t* W, int ldw, int Np, int Nreal, int Kp,
               const int4* cst, const int32_t* e, int32_t* Y, int ldy, const int32_t* residual,
               int epi = EPI_PLAIN, int Cs = 0, int32_t* Y1 = nullptr, int32_t* Y2 = nullptr, const int32_t* Lphi = nullptr);
bool gemm_hl_supported(int Mp, int Np, int Kp);
extern int g_gemm_hl_variant;   // -1 auto, 0 64x64, 1 128x64, 2 64x128
}
