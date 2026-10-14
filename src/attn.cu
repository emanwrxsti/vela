// Fused exact attention (PoI v2 semantics) on int8 tensor cores via mma.sync.m16n8k32 (sm_80+).
// Never materialises the score matrix. Three kernels (one per pass) over the keys per 64-query tile:
//   pass 0: row max of s = rshr(S, 19-eq-ek)   pass 1: sumE = Σ LUT(mx - s) -> r = 2^62/sumE   pass 2: P = ((E*r + 2^46)>>47) -> P·V
// S is exact: S = 65536*hh + T, T = 256*(hl+lh) + ll + 32768*qhs + 128*qls + 128*ksum (|T| < 2^31) (hi/lo int8 split of 16-bit operands).
// Block = 64 queries x one (batch, head); 8 warps: warp w -> query rows (w&3)*16.., key half (w>>2) of each 64-key tile;
// in the PV stage warp w -> rows (w&3)*16, output dims (w>>2)*32... Passes 0/1 need few registers/smem -> high occupancy.
#include "kernels.h"
#include "intops.cuh"
#include <climits>
#include <stdexcept>
#include <cstddef>

namespace meow {
namespace {
constexpr int TQ = 64, TK = 64, NW = 8, NTHR = NW * 32, LD = 80, LDP = 64;   // LD: smem row stride in bytes (bank-conflict free 4-byte fragment loads)

struct __align__(16) Smem {
    // ---- pass 0 ----
    int8_t khi[2][TK * LD], klo[2][TK * LD];   // K tile  [key][dim], double-buffered (cp.async)
    i64 ksum[2][TK]; int32_t ek[2][TK];
    int32_t cq[TQ]; int32_t eq[TQ];            // 32768*qhs + 128*qls fits int32
    int32_t mxh[2][TQ]; i64 sumh[2][TQ];
    // ---- pass 1 ----
    int32_t lexp[1028];                        // exp LUT, values <= 2^30 (padded to keep 16-byte alignment below)
    // ---- pass 2 ----
    __align__(16) int8_t vhi[64 * LD], vlo[64 * LD];      // V tile  [dim][key]
    int8_t phi[TQ * LDP], plo[TQ * LDP];       // P tile  [query][key]
    i64 phsh[2][TQ], plsh[2][TQ];
    int32_t vev[64]; i64 vvsum[64];
};
__device__ __forceinline__ void cp16(void* sm, const void* gm) { unsigned sa = unsigned(__cvta_generic_to_shared(sm)); asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(gm)); }
__device__ __forceinline__ void cp8(void* sm, const void* gm) { unsigned sa = unsigned(__cvta_generic_to_shared(sm)); asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(sa), "l"(gm)); }
__device__ __forceinline__ void cp4(void* sm, const void* gm) { unsigned sa = unsigned(__cvta_generic_to_shared(sm)); asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(sa), "l"(gm)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }

__device__ __forceinline__ void mma_s8(int32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t ld32(const int8_t* p) { return *reinterpret_cast<const uint32_t*>(p); }
/** Exact score: s = rshr((hh<<16) + T, sh), saturated to int32. */
__device__ __forceinline__ int score(int hh, int T, int sh) {
    i64 S = (i64(hh) << 16) + T;
    i64 v;
    if (sh >= 0) v = (S + ((1LL << sh) >> 1)) >> sh;
    else { int neg = -sh; i64 lim = 1LL << (62 - neg); v = (S < -lim ? -lim : (S > lim ? lim : S)) << neg; }
    return int(v < INT_MIN ? INT_MIN : (v > INT_MAX ? INT_MAX : v));
}
/** E = LUT(z) for z = mx - s >= 0 (uint32), 0 when z >= 2^20. Branchless. */
__device__ __forceinline__ int lutexp32(unsigned z, const int32_t* L) {
    unsigned zc = z < (16u << 16) ? z : ((16u << 16) - 1);
    int i = int(zc >> 10), f = int(zc & 1023);
    int a = L[i], b = L[i + 1];
    int v = a + int((i64(b - a) * f) >> 10);
    return z < (16u << 16) ? v : 0;
}

template <int PASS>
__global__ void __launch_bounds__(NTHR, (PASS == 2 ? 2 : 3)) attn_pass_k(
    const int8_t* __restrict__ qhi, const int8_t* __restrict__ qlo, const int32_t* __restrict__ eq, const i64* __restrict__ qhs, const i64* __restrict__ qls,
    const int8_t* __restrict__ khi, const int8_t* __restrict__ klo, const int32_t* __restrict__ ek, const i64* __restrict__ ksum,
    const int8_t* __restrict__ vthi, const int8_t* __restrict__ vtlo, const int32_t* __restrict__ ev, const i64* __restrict__ vsum,
    int Tq, int Tqp, int Tk, int Tkp, int causal, const i64* __restrict__ Lexp, int heads, int C, int32_t* __restrict__ gmx, i64* __restrict__ gr, int32_t* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem& S = *reinterpret_cast<Smem*>(smem_raw);
    const int g = blockIdx.y, qt = blockIdx.x, q0 = qt * TQ;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, rw = w & 3, kh = w >> 2;
    const int gid = lane >> 2, t4 = lane & 3;
    const int b = g / heads, h = g % heads;
    if (q0 >= Tq) return;
    const size_t rowBase = size_t(g) * Tqp + q0;
    for (int i = tid; i < TQ; i += NTHR) { size_t rq = rowBase + i; S.cq[i] = int(32768LL * qhs[rq] + 128LL * qls[rq]); S.eq[i] = eq[rq]; }
    if (PASS >= 1) for (int i = tid; i < 1025; i += NTHR) S.lexp[i] = int(Lexp[i]);
    if (PASS == 2) for (int i = tid; i < 64; i += NTHR) { S.vev[i] = ev[g * 64 + i]; S.vvsum[i] = vsum[g * 64 + i]; }
    uint32_t qa_hi[2][4], qa_lo[2][4];
    {
        const int8_t* qh = qhi + (rowBase + rw * 16) * 64; const int8_t* ql = qlo + (rowBase + rw * 16) * 64;
        for (int ks = 0; ks < 2; ++ks) {
            int c0 = ks * 32 + t4 * 4;
            qa_hi[ks][0] = ld32(qh + gid * 64 + c0); qa_hi[ks][1] = ld32(qh + (gid + 8) * 64 + c0); qa_hi[ks][2] = ld32(qh + gid * 64 + c0 + 16); qa_hi[ks][3] = ld32(qh + (gid + 8) * 64 + c0 + 16);
            qa_lo[ks][0] = ld32(ql + gid * 64 + c0); qa_lo[ks][1] = ld32(ql + (gid + 8) * 64 + c0); qa_lo[ks][2] = ld32(ql + gid * 64 + c0 + 16); qa_lo[ks][3] = ld32(ql + (gid + 8) * 64 + c0 + 16);
        }
    }
    const int row0 = rw * 16 + gid, row1 = row0 + 8;
    const int qrow0 = q0 + row0, qrow1 = q0 + row1;
    __syncthreads();
    const int cq0 = S.cq[row0], cq1 = S.cq[row1], eq0 = S.eq[row0], eq1 = S.eq[row1];
    int mxr0 = 0, mxr1 = 0; unsigned r0lo = 0, r0hi = 0, r1lo = 0, r1hi = 0;
    if (PASS >= 1) { mxr0 = gmx[rowBase + row0]; mxr1 = gmx[rowBase + row1]; }
    if (PASS == 2) { i64 rr0 = gr[rowBase + row0], rr1 = gr[rowBase + row1]; r0lo = unsigned(rr0); r0hi = unsigned(rr0 >> 32); r1lo = unsigned(rr1); r1hi = unsigned(rr1 >> 32); }
    // passes 0/1 may skip fully-masked tiles; pass 2 must include every key tile (the 128*vsum correction assumes all keys went through the MMA with P = 0 = (0,-128)).
    const int nTiles = (PASS == 2 || !causal) ? Tkp / TK : min(Tkp / TK, qt + 1);
    int mx0 = INT_MIN, mx1 = INT_MIN; i64 sum0 = 0, sum1 = 0, phs0 = 0, phs1 = 0, pls0 = 0, pls1 = 0;
    int32_t o_hh[4][4], o_mx[4][4], o_ll[4][4];
    if (PASS == 2) for (int n = 0; n < 4; ++n) for (int j = 0; j < 4; ++j) { o_hh[n][j] = 0; o_mx[n][j] = 0; o_ll[n][j] = 0; }
    auto loadK = [&](int buf, int kt) {
        const int k0 = kt * TK;
        const int8_t* sh = khi + (size_t(g) * Tkp + k0) * 64; const int8_t* sl = klo + (size_t(g) * Tkp + k0) * 64;
        for (int i = tid; i < TK * 4; i += NTHR) { int key = i >> 2, c = i & 3; cp16(S.khi[buf] + key * LD + c * 16, sh + i * 16); cp16(S.klo[buf] + key * LD + c * 16, sl + i * 16); }
        for (int i = tid; i < TK; i += NTHR) { cp8(&S.ksum[buf][i], ksum + g * Tkp + k0 + i); cp4(&S.ek[buf][i], ek + g * Tkp + k0 + i); }
    };
    auto loadV = [&](int kt) {
        const int k0 = kt * TK;
        for (int i = tid; i < 64 * 4; i += NTHR) { int d = i >> 2, c = i & 3; cp16(S.vhi + d * LD + c * 16, vthi + (size_t(g) * 64 + d) * Tkp + k0 + c * 16); cp16(S.vlo + d * LD + c * 16, vtlo + (size_t(g) * 64 + d) * Tkp + k0 + c * 16); }
    };
    loadK(0, 0); cp_commit();
    for (int kt = 0; kt < nTiles; ++kt) {
        const int k0 = kt * TK; const int buf = kt & 1;
        cp_wait<0>();                                            // K(kt) landed
        __syncthreads();                                         // ...and visible to all; everyone finished tile kt-1 (incl. its PV reads of V / P)
        if (PASS == 2) { loadV(kt); cp_commit(); }               // V(kt): waited for right before the PV stage
        if (kt + 1 < nTiles) { loadK(buf ^ 1, kt + 1); cp_commit(); }
        const int8_t* Kh = S.khi[buf]; const int8_t* Kl = S.klo[buf]; const i64* Ks = S.ksum[buf]; const int32_t* Ke = S.ek[buf];
        const bool fullTile = (k0 + TK <= Tk) && !causal;       // no per-element validity checks needed
        // scores: this warp -> n-tiles kh*4 .. kh*4+3 (8 keys each), two n-tiles per scalar block for ILP
        for (int np = 0; np < 2; ++np) {
            int32_t hh[2][4], hl[2][4], lh[2][4], ll[2][4];
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int nt = kh * 4 + np * 2 + u, kb = nt * 8;
                for (int j = 0; j < 4; ++j) { hh[u][j] = 0; hl[u][j] = 0; lh[u][j] = 0; ll[u][j] = 0; }
                for (int ks = 0; ks < 2; ++ks) {
                    const int8_t* kh_p = Kh + (kb + gid) * LD + ks * 32 + t4 * 4; const int8_t* kl_p = Kl + (kb + gid) * LD + ks * 32 + t4 * 4;
                    uint32_t bh0 = ld32(kh_p), bh1 = ld32(kh_p + 16), bl0 = ld32(kl_p), bl1 = ld32(kl_p + 16);
                    mma_s8(hh[u], qa_hi[ks], bh0, bh1); mma_s8(hl[u], qa_hi[ks], bl0, bl1); mma_s8(lh[u], qa_lo[ks], bh0, bh1); mma_s8(ll[u], qa_lo[ks], bl0, bl1);
                }
            }
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int nt = kh * 4 + np * 2 + u, kb = nt * 8;
                const int kl0 = kb + t4 * 2; const int key0 = k0 + kl0;
                const int ekk[2] = {Ke[kl0], Ke[kl0 + 1]}; const int kk[2] = {int(Ks[kl0] * 128), int(Ks[kl0 + 1] * 128)};
                int8_t ph[4], pl[4]; unsigned es0 = 0, es1 = 0; int phs_t0 = 0, pls_t0 = 0, phs_t1 = 0, pls_t1 = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int kc = j & 1, key = key0 + kc; const bool r1 = j >= 2;
                    const int T = ((hl[u][j] + lh[u][j]) << 8) + ll[u][j] + (r1 ? cq1 : cq0) + kk[kc];
                    const bool valid = fullTile || (key < Tk && !(causal && key > (r1 ? qrow1 : qrow0)));
                    const int sv = valid ? score(hh[u][j], T, 19 - (r1 ? eq1 : eq0) - ekk[kc]) : INT_MIN;
                    if (PASS == 0) { if (r1) mx1 = max(mx1, sv); else mx0 = max(mx0, sv); }
                    else {
                        const unsigned z = valid ? unsigned(r1 ? mxr1 : mxr0) - unsigned(sv) : 0xffffffffu;
                        const int E = lutexp32(z, S.lexp);
                        if (PASS == 1) { if (r1) es1 += unsigned(E); else es0 += unsigned(E); }
                        else {
                            const unsigned rlo = r1 ? r1lo : r0lo, rhi = r1 ? r1hi : r0hi;
                            unsigned long long prod = (unsigned long long)unsigned(E) * rlo + ((unsigned long long)(unsigned(E) * rhi) << 32);
                            prod = (prod + (1ull << 46)) >> 47; const int P = prod > 32767 ? 32767 : int(prod);
                            ph[j] = int8_t(P >> 8); pl[j] = int8_t((P & 255) - 128);
                            if (r1) { phs_t1 += ph[j]; pls_t1 += pl[j]; } else { phs_t0 += ph[j]; pls_t0 += pl[j]; }
                        }
                    }
                }
                if (PASS == 1) { sum0 += es0; sum1 += es1; }
                else if (PASS == 2) {
                    phs0 += phs_t0; pls0 += pls_t0; phs1 += phs_t1; pls1 += pls_t1;
                    const int c = kb + t4 * 2;
                    *reinterpret_cast<uint16_t*>(S.phi + row0 * LDP + c) = uint16_t(uint8_t(ph[0]) | (uint16_t(uint8_t(ph[1])) << 8));
                    *reinterpret_cast<uint16_t*>(S.plo + row0 * LDP + c) = uint16_t(uint8_t(pl[0]) | (uint16_t(uint8_t(pl[1])) << 8));
                    *reinterpret_cast<uint16_t*>(S.phi + row1 * LDP + c) = uint16_t(uint8_t(ph[2]) | (uint16_t(uint8_t(ph[3])) << 8));
                    *reinterpret_cast<uint16_t*>(S.plo + row1 * LDP + c) = uint16_t(uint8_t(pl[2]) | (uint16_t(uint8_t(pl[3])) << 8));
                }
            }
        }
        if (PASS == 2) {
            cp_wait<0>();                                        // V(kt) (and K(kt+1)) landed
            __syncthreads();                                     // P tile complete, V visible
            for (int ks = 0; ks < 2; ++ks) {
                const int c0 = ks * 32 + t4 * 4;
                uint32_t pa_h[4] = {ld32(S.phi + row0 * LDP + c0), ld32(S.phi + row1 * LDP + c0), ld32(S.phi + row0 * LDP + c0 + 16), ld32(S.phi + row1 * LDP + c0 + 16)};
                uint32_t pa_l[4] = {ld32(S.plo + row0 * LDP + c0), ld32(S.plo + row1 * LDP + c0), ld32(S.plo + row0 * LDP + c0 + 16), ld32(S.plo + row1 * LDP + c0 + 16)};
                for (int nd = 0; nd < 4; ++nd) {
                    const int d = kh * 32 + nd * 8 + gid;
                    const int8_t* vh_p = S.vhi + d * LD + c0; const int8_t* vl_p = S.vlo + d * LD + c0;
                    uint32_t vh0 = ld32(vh_p), vh1 = ld32(vh_p + 16), vl0 = ld32(vl_p), vl1 = ld32(vl_p + 16);
                    mma_s8(o_hh[nd], pa_h, vh0, vh1); mma_s8(o_mx[nd], pa_h, vl0, vl1); mma_s8(o_mx[nd], pa_l, vh0, vh1); mma_s8(o_ll[nd], pa_l, vl0, vl1);
                }
            }
        }
    }
    cp_wait<0>();
    __syncthreads();
    if (PASS == 0) {
        for (int o = 1; o < 4; o <<= 1) { mx0 = max(mx0, __shfl_xor_sync(0xffffffffu, mx0, o)); mx1 = max(mx1, __shfl_xor_sync(0xffffffffu, mx1, o)); }
        if (t4 == 0) { S.mxh[kh][row0] = mx0; S.mxh[kh][row1] = mx1; }
        __syncthreads();
        for (int i = tid; i < TQ; i += NTHR) gmx[rowBase + i] = max(S.mxh[0][i], S.mxh[1][i]);
    } else if (PASS == 1) {
        for (int o = 1; o < 4; o <<= 1) { sum0 += __shfl_xor_sync(0xffffffffu, sum0, o); sum1 += __shfl_xor_sync(0xffffffffu, sum1, o); }
        if (t4 == 0) { S.sumh[kh][row0] = sum0; S.sumh[kh][row1] = sum1; }
        __syncthreads();
        for (int i = tid; i < TQ; i += NTHR) { i64 se = S.sumh[0][i] + S.sumh[1][i]; gr[rowBase + i] = se > 0 ? floor_div(1LL << 62, se) : 0; }
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
}
constexpr size_t SMEM0 = offsetof(Smem, lexp), SMEM1 = offsetof(Smem, vhi), SMEM2 = sizeof(Smem);
}  // namespace

void k_fused_attention(cudaStream_t st, const int8_t* qhi, const int8_t* qlo, const int32_t* eq, const i64* qhs, const i64* qls,
                       const int8_t* khi, const int8_t* klo, const int32_t* ek, const i64* ksum,
                       const int8_t* vthi, const int8_t* vtlo, const int32_t* ev, const i64* vsum,
                       int B, int heads, int Tq, int Tqp, int Tk, int Tkp, bool causal, const i64* Lexp, int C, int32_t* gmx, i64* gr, int32_t* out) {
    if (Tqp % 64 || Tkp % 64) throw std::runtime_error("fused attention: Tqp/Tkp must be multiples of 64");
    static bool attrSet = false;
    if (!attrSet) {
        cudaFuncSetAttribute(attn_pass_k<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEM0));
        cudaFuncSetAttribute(attn_pass_k<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEM1));
        cudaFuncSetAttribute(attn_pass_k<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEM2));
        attrSet = true;
    }
    dim3 grid(Tqp / 64, B * heads);
    attn_pass_k<0><<<grid, NTHR, SMEM0, st>>>(qhi, qlo, eq, qhs, qls, khi, klo, ek, ksum, vthi, vtlo, ev, vsum, Tq, Tqp, Tk, Tkp, causal, Lexp, heads, C, gmx, gr, out);
    attn_pass_k<1><<<grid, NTHR, SMEM1, st>>>(qhi, qlo, eq, qhs, qls, khi, klo, ek, ksum, vthi, vtlo, ev, vsum, Tq, Tqp, Tk, Tkp, causal, Lexp, heads, C, gmx, gr, out);
    attn_pass_k<2><<<grid, NTHR, SMEM2, st>>>(qhi, qlo, eq, qhs, qls, khi, klo, ek, ksum, vthi, vtlo, ev, vsum, Tq, Tqp, Tk, Tkp, causal, Lexp, heads, C, gmx, gr, out);
}
}  // namespace meow
