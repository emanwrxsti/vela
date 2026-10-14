// Host launch wrappers for the PoI v2 kernels (kernels.cu). All pointers are device pointers.
// Activations are int32, row-major [rows, C] (tokens x channels, NHWC for images).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace meow {
typedef int64_t i64; typedef unsigned int u32;
struct GnStat { i64 sum; int mn, mx; i64 mean; int e; i64 ss; i64 sd; };

// Per-row power-of-two quantisation of X[M, K] (ld ldx) into hi/lo int8 [Mp, Kp] + e[M]. Pads cols [K,Kp) with value 0 = (0,-128).
void k_quant_rows(cudaStream_t st, const int32_t* X, int M, int K, int ldx, int8_t* hi, int8_t* lo, int32_t* e, int Kp);
// y[r,n] = sat32(rshr((256*Chi + Clo + 128*wsum[n]) * m[n], s[n] - e[r]) + b[n]); with residual R: y = sat32(R[r,n] + y) (R may alias Y)
// cst[n] = {m, s, b, wsum} packed per output column.
void k_epilogue(cudaStream_t st, const int32_t* Chi, const int32_t* Clo, int M, int N, int ldc, const int4* cst, const int32_t* e, int32_t* Y, int ldy, const int32_t* residual = nullptr);
// pm[b*T+p] = max_c |X[(b*T+p), c]|
void k_pixmax(cudaStream_t st, const int32_t* X, int rows, int C, uint32_t* pm);
// 3x3 (pad 1, stride) im2col with per-output-pixel quantisation: cols rows r = b*Ho*Wo + p, K index c*9+ky*3+kx.
void k_im2col3(cudaStream_t st, const int32_t* X, const uint32_t* pm, int B, int H, int W, int C, int stride,
               int8_t* hi, int8_t* lo, int32_t* e, int Kp);
// GroupNorm on X[B, T, C] with `groups`: stats then apply (g/beta per channel, Q16).
void k_gn_stats1(cudaStream_t st, const int32_t* X, int B, int T, int C, int groups, GnStat* gs);
void k_gn_stats2(cudaStream_t st, const int32_t* X, int B, int T, int C, int groups, i64 den, GnStat* gs);
void k_gn_apply(cudaStream_t st, const int32_t* X, int B, int T, int C, int groups, const GnStat* gs, const i64* g, const i64* beta, int32_t* Y);
// LayerNorm over rows of X[M, C]
void k_lnorm(cudaStream_t st, const int32_t* X, int M, int C, const i64* g, const i64* beta, i64 den, int32_t* Y);
void k_act(cudaStream_t st, int32_t* X, size_t n, const int32_t* L);                     // x = sat32(rshr(x*lut(x),16)) in place
void k_geglu(cudaStream_t st, int32_t* A, const int32_t* G, size_t n, const int32_t* Lphi); // a = sat32(rshr(a*gelu(g),16))
void k_add(cudaStream_t st, int32_t* X, const int32_t* Y, size_t n);                  // x = sat32(x+y)
void k_concat(cudaStream_t st, const int32_t* A, int Ca, const int32_t* B, int Cb, int rows, int32_t* out);
void k_upsample2(cudaStream_t st, const int32_t* X, int B, int H, int W, int C, int32_t* out);
// ChaCha20(seed) keystream -> uint16 -> gauss LUT -> n (channel-major int32 [B,4,4096]) and x = rshr(n*cin,30) (NHWC [B,4096,4]).
void k_noise(cudaStream_t st, const uint8_t* seeds, int B, const i64* gauss, i64 cin, int32_t* n, int32_t* x);
// x0 = rshr((n - eps)*sig, 16); q = clamp(rshr(x0,8)); out int16le [B, 32768 bytes], eps NHWC [B,4096,4]
void k_finalize(cudaStream_t st, const int32_t* n, const int32_t* eps, int B, i64 sig, uint8_t* out);
// Text embedding: X[t,d] = sat32(rshr(tw[id][d]*tm[id], ts[id]-16) + pos[t][d]) for t<77; rows [77,rowsPad) zeroed.
void k_text_embed(cudaStream_t st, const int32_t* ids, int T, int rowsPad, const int8_t* tw, const i64* tm, const i64* ts, const i64* pos, int D, int32_t* X);
// Attention operands. heads of 64. G = B*heads. Layout: per (g, t) 64 values.
//  Q/K: X[B, T, C] -> hi/lo [G, Tp, 64], e[G, Tp], hs/ls = row sums of hi/lo (A-operand), tsum = sum of true quantised values (B-operand). Rows t>=T are value 0.
void k_quant_heads(cudaStream_t st, const int32_t* X, int B, int T, int C, int heads, int Tp,
                   int8_t* hi, int8_t* lo, int32_t* e, i64* hs, i64* ls, i64* tsum);
//  V: X[B, Tk, C] -> transposed vt hi/lo [G, 64, Tkp] with per-(g,d) exps ev[G,64] and vsum[G,64] = sum_t of true quantised v.
//  scratch: u32[G*64] for the column max (zeroed inside).
void k_quant_v(cudaStream_t st, const int32_t* X, int B, int Tk, int C, int heads, int Tkp,
               int8_t* vt_hi, int8_t* vt_lo, int32_t* ev, i64* vsum, uint32_t* scratch);
//  Softmax over scores for query rows [q0, q0+Qc) of every g: 4 GEMM partials [G, Qc, Tkp] -> P hi/lo [G, Qc, Tkp] + row sums.
void k_softmax(cudaStream_t st, const int32_t* Shh, const int32_t* Shl, const int32_t* Slh, const int32_t* Sll,
               const i64* qhs, const i64* qls, const int32_t* eq, const i64* ksum, const int32_t* ek,
               int G, int Tqp, int q0, int Qc, int Tk, int Tkp, bool causal, const i64* Lexp,
               int8_t* Phi, int8_t* Plo, i64* phs, i64* pls);
//  out[b, q0+a, h*64+d] = rshr(O, 15 - ev[g,d]) from 4 PV partials [G, Qc, 64]
void k_attn_out(cudaStream_t st, const int32_t* Ohh, const int32_t* Ohl, const int32_t* Olh, const int32_t* Oll,
                const i64* phs, const i64* pls, const i64* vsum, const int32_t* ev,
                int B, int heads, int Tq, int q0, int Qc, int C, int32_t* out);
void k_zero(cudaStream_t st, void* p, size_t bytes);
// Row kernels that emit the quantised GEMM operand directly (hi/lo [*, Kp], e[row]); C <= 5120, Kp >= C.
void k_lnorm_q(cudaStream_t st, const int32_t* X, int M, int C, const i64* g, const i64* beta, i64 den, int8_t* hi, int8_t* lo, int32_t* e, int Kp);
// GroupNorm apply (+ optional SiLU); block per pixel row. Either quantised output (hi/lo/e, when hi != nullptr) or int32 Y plus the pixel max pm.
void k_gn_apply_row(cudaStream_t st, const int32_t* X, int B, int T, int C, int groups, const GnStat* gs, const i64* g, const i64* beta, const int32_t* Lsig,
                    int32_t* Y, uint32_t* pm, int8_t* hi, int8_t* lo, int32_t* e, int Kp);
// a = sat32(rshr(a*gelu(g),16)) quantised per row
void k_geglu_q(cudaStream_t st, const int32_t* A, const int32_t* G, int M, int C, const int32_t* Lphi, int8_t* hi, int8_t* lo, int32_t* e, int Kp);
// Epilogue of a merged GEMM whose N = parts*Cs: column n -> output Y[n / Cs][row][n % Cs] (up to 3 outputs).
void k_epilogue_split(cudaStream_t st, const int32_t* Chi, const int32_t* Clo, int M, int N, int ldc, const int4* cst, const int32_t* e, int Cs, int32_t* Y0, int32_t* Y1, int32_t* Y2);
// Epilogue of the merged GEGLU GEMM (N = 2F: [a | g]) fused with a = sat32(rshr(a*gelu(g),16)) and row quantisation -> hi/lo/e [*, Kp>=F].
void k_epilogue_geglu_q(cudaStream_t st, const int32_t* Chi, const int32_t* Clo, int M, int F, int ldc, const int4* cst, const int32_t* e, const int32_t* Lphi, int8_t* hi, int8_t* lo, int32_t* eo, int Kp);
}  // namespace meow
namespace meow {
// Fused exact attention on tensor cores (attn.cu). Tqp, Tkp multiples of 64; K/V operands from k_quant_heads / k_quant_v.
void k_fused_attention(cudaStream_t st, const int8_t* qhi, const int8_t* qlo, const int32_t* eq, const i64* qhs, const i64* qls,
                       const int8_t* khi, const int8_t* klo, const int32_t* ek, const i64* ksum,
                       const int8_t* vthi, const int8_t* vtlo, const int32_t* ev, const i64* vsum,
                       int B, int heads, int Tq, int Tqp, int Tk, int Tkp, bool causal, const i64* Lexp, int C, int32_t* gmx, i64* gr, int32_t* out);
}
