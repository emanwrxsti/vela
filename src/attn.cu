// Fused exact attention (PoI v2 semantics) on int8 tensor cores via mma.sync.m16n8k32 (sm_80+).
// Never materialises the score matrix: three passes over the keys per 64-query tile
//   pass 0: row max of s = rshr(S, 19-eq-ek)         pass 1: sumE = Σ LUT(mx - s)        pass 2: P = ((E*r + 2^46)>>47) -> P·V
// S is exact: S = 65536*hh + 256*(hl+lh) + ll + 32768*qhs + 128*qls + 128*ksum (hi/lo int8 split of 16-bit operands).
// Block = 64 queries x one (batch, head); 8 warps: warp w -> query rows (w&3)*16.., key half (w>>2) of each 64-key tile;
// in the PV stage warp w -> rows (w&3)*16, output dims (w>>2)*32... Accumulator fragments are combined in registers.
#include "kernels.h"
#include "intops.cuh"
#include <climits>
#include <stdexcept>

namespace meow {
namespace {
constexpr int TQ = 64, TK = 64, NW = 8, NTHR = NW * 32, LD = 80;   // LD: smem row stride in bytes (bank-conflict free 4-byte fragment loads)

struct __align__(16) Smem {
    int8_t khi[TK * LD], klo[TK * LD];      // K tile  [key][dim]
    int8_t vhi[64 * LD], vlo[64 * LD];      // V tile  [dim][key]
    int8_t phi[TQ * LD], plo[TQ * LD];      // P tile  [query][key]
    i64 ksum[TK]; int32_t ek[TK];
    i64 cq[TQ]; int32_t eq[TQ];
    int32_t mxh[2][TQ]; i64 sumh[2][TQ]; i64 phsh[2][TQ], plsh[2][TQ];
    int32_t mx[TQ]; i64 r[TQ];
    int32_t vev[64]; i64 vvsum[64];
    i64 lexp[1025];
};

__device__ __forceinline__ void mma_s8(int32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t ld32(const int8_t* p) { return *reinterpret_cast<const uint32_t*>(p); }
__device__ __forceinline__ i64 lutexp(i64 z, const i64* L) {
    if (z >= (16LL << 16)) return 0;
    i64 i = z >> 10, f = z & 1023;
    i64 a = L[i], b = L[i + 1];
    return a + (((b - a) * f) >> 10);
}

__global__ void __launch_bounds__(NTHR, 2) fused_attn_k(
    const int8_t* __restrict__ qhi, const int8_t* __restrict__ qlo, const int32_t* __restrict__ eq, const i64* __restrict__ qhs, const i64* __restrict__ qls,
    const int8_t* __restrict__ khi, const int8_t* __restrict__ klo, const int32_t* __restrict__ ek, const i64* __restrict__ ksum,
    const int8_t* __restrict__ vthi, const int8_t* __restrict__ vtlo, const int32_t* __restrict__ ev, const i64* __restrict__ vsum,
    int Tq, int Tqp, int Tk, int Tkp, int causal, const i64* __restrict__ Lexp, int heads, int C, int32_t* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem& S = *reinterpret_cast<Smem*>(smem_raw);
    const int g = blockIdx.y, qt = blockIdx.x, q0 = qt * TQ;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, rw = w & 3, kh = w >> 2;
    const int gid = lane >> 2, t4 = lane & 3;                    // mma fragment coordinates
    const int b = g / heads, h = g % heads;
    if (q0 >= Tq) return;
    for (int i = tid; i < TQ; i += NTHR) { int rq = g * Tqp + q0 + i; S.cq[i] = 32768LL * qhs[rq] + 128LL * qls[rq]; S.eq[i] = eq[rq]; }
    for (int i = tid; i < 64; i += NTHR) { S.vev[i] = ev[g * 64 + i]; S.vvsum[i] = vsum[g * 64 + i]; }
    for (int i = tid; i < 1025; i += NTHR) S.lexp[i] = Lexp[i];
    // Q A-fragments (rows rw*16 + gid, +8; 2 k-steps of 32 dims) for hi and lo, kept in registers
    uint32_t qa_hi[2][4], qa_lo[2][4];
    {
        const int8_t* qh = qhi + (size_t(g) * Tqp + q0 + rw * 16) * 64; const int8_t* ql = qlo + (size_t(g) * Tqp + q0 + rw * 16) * 64;
        for (int ks = 0; ks < 2; ++ks) {
            int c0 = ks * 32 + t4 * 4;
            qa_hi[ks][0] = ld32(qh + gid * 64 + c0); qa_hi[ks][1] = ld32(qh + (gid + 8) * 64 + c0); qa_hi[ks][2] = ld32(qh + gid * 64 + c0 + 16); qa_hi[ks][3] = ld32(qh + (gid + 8) * 64 + c0 + 16);
            qa_lo[ks][0] = ld32(ql + gid * 64 + c0); qa_lo[ks][1] = ld32(ql + (gid + 8) * 64 + c0); qa_lo[ks][2] = ld32(ql + gid * 64 + c0 + 16); qa_lo[ks][3] = ld32(ql + (gid + 8) * 64 + c0 + 16);
        }
    }
    const int row0 = rw * 16 + gid, row1 = row0 + 8;             // this thread's two local query rows
    const int qrow0 = q0 + row0, qrow1 = q0 + row1;
    const int nTilesMasked = causal ? min(Tkp / TK, qt + 1) : Tkp / TK;
    int32_t o_hh[4][4], o_mx[4][4], o_ll[4][4];                  // PV accumulators: 4 n-tiles of 8 dims x 3 partials
    for (int pass = 0; pass < 3; ++pass) {
        int mx0 = INT_MIN, mx1 = INT_MIN; i64 sum0 = 0, sum1 = 0, phs0 = 0, phs1 = 0, pls0 = 0, pls1 = 0;
        if (pass == 2) for (int n = 0; n < 4; ++n) for (int j = 0; j < 4; ++j) { o_hh[n][j] = 0; o_mx[n][j] = 0; o_ll[n][j] = 0; }
        const int nTiles = pass == 2 ? Tkp / TK : nTilesMasked;
        for (int kt = 0; kt < nTiles; ++kt) {
            const int k0 = kt * TK;
            __syncthreads();
            {   // K tile (and V tile in pass 2) -> smem, 16-byte chunks
                const int4* sh = reinterpret_cast<const int4*>(khi + (size_t(g) * Tkp + k0) * 64); const int4* sl = reinterpret_cast<const int4*>(klo + (size_t(g) * Tkp + k0) * 64);
                for (int i = tid; i < TK * 4; i += NTHR) { int key = i >> 2, c = i & 3; *reinterpret_cast<int4*>(S.khi + key * LD + c * 16) = sh[i]; *reinterpret_cast<int4*>(S.klo + key * LD + c * 16) = sl[i]; }
                for (int i = tid; i < TK; i += NTHR) { S.ksum[i] = ksum[g * Tkp + k0 + i]; S.ek[i] = ek[g * Tkp + k0 + i]; }
                if (pass == 2) for (int i = tid; i < 64 * 4; i += NTHR) {
                    int d = i >> 2, c = i & 3;
                    *reinterpret_cast<int4*>(S.vhi + d * LD + c * 16) = *reinterpret_cast<const int4*>(vthi + (size_t(g) * 64 + d) * Tkp + k0 + c * 16);
                    *reinterpret_cast<int4*>(S.vlo + d * LD + c * 16) = *reinterpret_cast<const int4*>(vtlo + (size_t(g) * 64 + d) * Tkp + k0 + c * 16);
                }
            }
            __syncthreads();
            const i64 cq0 = S.cq[row0], cq1 = S.cq[row1]; const int eq0 = S.eq[row0], eq1 = S.eq[row1];
            // scores: this warp -> n-tiles kh*4 .. kh*4+3 (8 keys each)
            for (int nt2 = 0; nt2 < 4; ++nt2) {
                const int nt = kh * 4 + nt2, kb = nt * 8;             // local key base of this n-tile
                int32_t hh[4] = {0, 0, 0, 0}, hl[4] = {0, 0, 0, 0}, lh[4] = {0, 0, 0, 0}, ll[4] = {0, 0, 0, 0};
                for (int ks = 0; ks < 2; ++ks) {
                    const int8_t* kh_p = S.khi + (kb + gid) * LD + ks * 32 + t4 * 4; const int8_t* kl_p = S.klo + (kb + gid) * LD + ks * 32 + t4 * 4;
                    uint32_t bh0 = ld32(kh_p), bh1 = ld32(kh_p + 16), bl0 = ld32(kl_p), bl1 = ld32(kl_p + 16);
                    mma_s8(hh, qa_hi[ks], bh0, bh1); mma_s8(hl, qa_hi[ks], bl0, bl1); mma_s8(lh, qa_lo[ks], bh0, bh1); mma_s8(ll, qa_lo[ks], bl0, bl1);
                }
                // this thread's 4 elements: (row0, kb+t4*2), (row0, +1), (row1, kb+t4*2), (row1, +1)
                int8_t ph[4], pl[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int kl = kb + t4 * 2 + (j & 1), key = k0 + kl; const bool r1 = j >= 2;
                    const int qrow = r1 ? qrow1 : qrow0; const int row = r1 ? row1 : row0;
                    i64 Sx = 65536LL * hh[j] + 256LL * (i64(hl[j]) + lh[j]) + ll[j] + (r1 ? cq1 : cq0) + 128LL * S.ksum[kl];
                    const bool valid = key < Tk && !(causal && key > qrow);
                    const int s = valid ? int(rshr(Sx, 19 - (r1 ? eq1 : eq0) - S.ek[kl])) : INT_MIN;
                    if (pass == 0) { if (r1) mx1 = max(mx1, s); else mx0 = max(mx0, s); }
                    else {
                        const i64 E = valid ? lutexp(i64(S.mx[row]) - s, S.lexp) : 0;
                        if (pass == 1) { if (r1) sum1 += E; else sum0 += E; }
                        else {
                            i64 P = 0; if (E > 0) { P = (E * S.r[row] + (1LL << 46)) >> 47; if (P > 32767) P = 32767; }
                            split16(P, ph[j], pl[j]);
                            if (r1) { phs1 += ph[j]; pls1 += pl[j]; } else { phs0 += ph[j]; pls0 += pl[j]; }
                        }
                    }
                }
                if (pass == 2) {
                    const int c = kb + t4 * 2;
                    *reinterpret_cast<uint16_t*>(S.phi + row0 * LD + c) = uint16_t(uint8_t(ph[0]) | (uint16_t(uint8_t(ph[1])) << 8));
                    *reinterpret_cast<uint16_t*>(S.plo + row0 * LD + c) = uint16_t(uint8_t(pl[0]) | (uint16_t(uint8_t(pl[1])) << 8));
                    *reinterpret_cast<uint16_t*>(S.phi + row1 * LD + c) = uint16_t(uint8_t(ph[2]) | (uint16_t(uint8_t(ph[3])) << 8));
                    *reinterpret_cast<uint16_t*>(S.plo + row1 * LD + c) = uint16_t(uint8_t(pl[2]) | (uint16_t(uint8_t(pl[3])) << 8));
                }
            }
            if (pass == 2) {
                __syncthreads();                                     // P tile complete
                // PV: A = P rows rw*16.. (K = 64 keys, 2 k-steps), B = V^T [dim][key]; this warp -> dims kh*32 .. +32 (4 n-tiles)
                for (int ks = 0; ks < 2; ++ks) {
                    const int c0 = ks * 32 + t4 * 4;
                    uint32_t pa_h[4] = {ld32(S.phi + row0 * LD + c0), ld32(S.phi + row1 * LD + c0), ld32(S.phi + row0 * LD + c0 + 16), ld32(S.phi + row1 * LD + c0 + 16)};
                    uint32_t pa_l[4] = {ld32(S.plo + row0 * LD + c0), ld32(S.plo + row1 * LD + c0), ld32(S.plo + row0 * LD + c0 + 16), ld32(S.plo + row1 * LD + c0 + 16)};
                    for (int nd = 0; nd < 4; ++nd) {
                        const int d = kh * 32 + nd * 8 + gid;
                        const int8_t* vh_p = S.vhi + d * LD + c0; const int8_t* vl_p = S.vlo + d * LD + c0;
                        uint32_t vh0 = ld32(vh_p), vh1 = ld32(vh_p + 16), vl0 = ld32(vl_p), vl1 = ld32(vl_p + 16);
                        mma_s8(o_hh[nd], pa_h, vh0, vh1); mma_s8(o_mx[nd], pa_h, vl0, vl1); mma_s8(o_mx[nd], pa_l, vh0, vh1); mma_s8(o_ll[nd], pa_l, vl0, vl1);
                    }
                }
            }
        }
        __syncthreads();
        // combine across the 4 t4 lanes (same rows) and the two key-half warps
        if (pass == 0) {
            for (int o = 1; o < 4; o <<= 1) { mx0 = max(mx0, __shfl_xor_sync(0xffffffffu, mx0, o)); mx1 = max(mx1, __shfl_xor_sync(0xffffffffu, mx1, o)); }
            if (t4 == 0) { S.mxh[kh][row0] = mx0; S.mxh[kh][row1] = mx1; }
            __syncthreads();
            for (int i = tid; i < TQ; i += NTHR) S.mx[i] = max(S.mxh[0][i], S.mxh[1][i]);
        } else if (pass == 1) {
            for (int o = 1; o < 4; o <<= 1) { sum0 += __shfl_xor_sync(0xffffffffu, sum0, o); sum1 += __shfl_xor_sync(0xffffffffu, sum1, o); }
            if (t4 == 0) { S.sumh[kh][row0] = sum0; S.sumh[kh][row1] = sum1; }
            __syncthreads();
            for (int i = tid; i < TQ; i += NTHR) { i64 se = S.sumh[0][i] + S.sumh[1][i]; S.r[i] = se > 0 ? floor_div(1LL << 62, se) : 0; }
        } else {
            for (int o = 1; o < 4; o <<= 1) { phs0 += __shfl_xor_sync(0xffffffffu, phs0, o); phs1 += __shfl_xor_sync(0xffffffffu, phs1, o); pls0 += __shfl_xor_sync(0xffffffffu, pls0, o); pls1 += __shfl_xor_sync(0xffffffffu, pls1, o); }
            if (t4 == 0) { S.phsh[kh][row0] = phs0; S.phsh[kh][row1] = phs1; S.plsh[kh][row0] = pls0; S.plsh[kh][row1] = pls1; }
            __syncthreads();
            const i64 ps0 = 32768LL * (S.phsh[0][row0] + S.phsh[1][row0]) + 128LL * (S.plsh[0][row0] + S.plsh[1][row0]);
            const i64 ps1 = 32768LL * (S.phsh[0][row1] + S.phsh[1][row1]) + 128LL * (S.plsh[0][row1] + S.plsh[1][row1]);
            for (int nd = 0; nd < 4; ++nd) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const bool r1 = j >= 2; const int q = r1 ? qrow1 : qrow0; if (q >= Tq) continue;
                    const int d = kh * 32 + nd * 8 + t4 * 2 + (j & 1);
                    i64 O = 65536LL * o_hh[nd][j] + 256LL * o_mx[nd][j] + o_ll[nd][j] + (r1 ? ps1 : ps0) + 128LL * S.vvsum[d];
                    out[(size_t(b) * Tq + q) * C + h * 64 + d] = int32_t(rshr(O, 15 - S.vev[d]));
                }
            }
        }
        __syncthreads();
    }
}
}  // namespace

void k_fused_attention(cudaStream_t st, const int8_t* qhi, const int8_t* qlo, const int32_t* eq, const i64* qhs, const i64* qls,
                       const int8_t* khi, const int8_t* klo, const int32_t* ek, const i64* ksum,
                       const int8_t* vthi, const int8_t* vtlo, const int32_t* ev, const i64* vsum,
                       int B, int heads, int Tq, int Tqp, int Tk, int Tkp, bool causal, const i64* Lexp, int C, int32_t* out) {
    if (Tqp % 64 || Tkp % 64) throw std::runtime_error("fused attention: Tqp/Tkp must be multiples of 64");
    static bool attrSet = false;
    if (!attrSet) { cudaFuncSetAttribute(fused_attn_k, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(Smem))); attrSet = true; }
    dim3 grid(Tqp / 64, B * heads);
    fused_attn_k<<<grid, NTHR, sizeof(Smem), st>>>(qhi, qlo, eq, qhs, qls, khi, klo, ek, ksum, vthi, vtlo, ev, vsum, Tq, Tqp, Tk, Tkp, causal, Lexp, heads, C, out);
}
}  // namespace meow
