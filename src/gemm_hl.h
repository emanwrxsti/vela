// Custom int8 GEMM with the hi/lo split accumulated in-register and the exact PoI epilogue fused (gemm_hl.cu).
//   Y[r, n] = sat32(rshr((256*Ahi[r]·W[n] + Alo[r]·W[n] + 128*wsum[n]) * m[n], s[n] - e[r]) + b[n])  (+ residual, saturated)
// A planes: Ahi/Alo [Mp rows, ld lda] (Mp multiple of 64), W [Np rows, ld ldw] (Np multiple of 64), K = Kp (multiple of 16).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
namespace meow {
enum GemmEpi { EPI_PLAIN = 0, EPI_SPLIT3 = 1, EPI_GEGLU = 2 };   // GEGLU: W rows interleaved (2c = a_c, 2c+1 = g_c), writes sat32(rshr(a*gelu(g),16)) to Y[r, c] (ldy), Lphi = int32 phi LUT
void k_gemm_hl(cudaStream_t st, const int8_t* Ahi, const int8_t* Alo, int lda, int Mp, int Mreal,
               const int8_t* W, int ldw, int Np, int Nreal, int Kp,
               const int4* cst, const int32_t* e, int32_t* Y, int ldy, const int32_t* residual,
               int epi = EPI_PLAIN, int Cs = 0, int32_t* Y1 = nullptr, int32_t* Y2 = nullptr, const int32_t* Lphi = nullptr);
bool gemm_hl_supported(int Mp, int Np, int Kp);
}
