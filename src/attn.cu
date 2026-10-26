// Fused exact attention (PoI v2 semantics) on int8 tensor cores via mma.sync.m16n8k32 (sm_80+).
// Never materialises the score matrix. Three kernels (one per pass) over the keys per 64-query tile:
//   pass 0: row max of s = rshr(S, 19-eq-ek)   pass 1: sumE = Σ LUT(mx - s) -> r = 2^62/sumE   pass 2: P = ((E*r + 2^46)>>47) -> P·V
// S is exact: S = 65536*hh + T, T = 256*(hl+lh) + ll + 32768*qhs + 128*qls + 128*ksum (|T| < 2^31) (hi/lo int8 split of 16-bit operands).
// Block = 64 queries x one (batch, head); 8 warps: warp w -> query rows (w&3)*16.., key half (w>>2) of each 64-key tile;
// in the PV stage warp w -> rows (w&3)*16, output dims (w>>2)*32... Passes 0/1 need few registers/smem -> high occupancy.
#include "kernels.h"
#include "common.h"
#include "intops.cuh"
#include <climits>
#include <stdexcept>
#include <cstddef>

namespace meow {
namespace {
#ifndef MEOW_ATTN_TQ
#define MEOW_ATTN_TQ 64
#endif
constexpr int TQ = MEOW_ATTN_TQ, TK = 64, RW = TQ / 16, NW = 2 * RW, NTHR = NW * 32, LD = 80, LDP = 64;   // RW row-groups of 16 queries x 2 key halves   // LD: smem row stride in bytes (bank-conflict free 4-byte fragment loads)

struct __align__(16) Smem {
    // ---- pass 0 ----
    int8_t khi[2][TK * 64], klo[2][TK * 64];   // K tile  [key][dim], double-buffered (cp.async)
    int32_t ek[2][TK];
    int32_t cq[TQ]; int32_t eq[TQ];            // 128 * sum_d Q_d = 32768*qhs + 128*qls + 2^20 fits int32
    int32_t mxh[2][TQ]; i64 sumh[2][TQ];
    // ---- pass 1 ----
    int2 lexp2[1024];                          // passes 1/2: exp LUT as (value, slope to next entry): one 8-byte load per lookup; values <= 2^30
    union {
        struct {                               // ---- pass 2 ----
            __align__(16) int8_t vhi[64 * 64], vlo[64 * 64];      // V tile  [dim][key]
            int8_t phi[TQ * LDP], plo[TQ * LDP];       // P tile  [query][key]: hi = P >> 8 (0..127), lo = P & 255 (unsigned)
            int32_t phsh[2][TQ], plsh[2][TQ];
            int32_t pflag[RW][2];                      // P-hi plane nonzero anywhere in (16-row group, 32-key half) of the current tile
            int32_t vev[64];
        } p2;
    } u;
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
/** A operand unsigned (u8: lo planes stored as value & 255), B signed. */
__device__ __forceinline__ void mma_u8s8(int32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t ld32(const int8_t* p) { return *reinterpret_cast<const uint32_t*>(p); }
/** ldmatrix.x4: four 8x16-byte matrices; lane l supplies the row address of matrix l>>3, row l&7. Result reg i = matrix i, row lane>>2, bytes (lane&3)*4. */
__device__ __forceinline__ void ldm4(uint32_t* r, const void* rowAddr) {
    unsigned a = unsigned(__cvta_generic_to_shared(rowAddr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
/** K/V tiles [row][64 bytes]: 16-byte chunk index swizzled by row so 8-row ldmatrix reads are bank-conflict free (no padding). */
__device__ __forceinline__ int kswz(int row, int chunk) { return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4); }
/** P tile [query][key] (LDP = 64): 16-byte chunk index swizzled by row so 8-row ldmatrix reads are bank-conflict free. */
__device__ __forceinline__ int pswz(int row, int key) { return row * LDP + (((key >> 4) ^ ((row >> 1) & 3)) << 4) + (key & 15); }
/** Exact score: s = rshr((hh<<16) + T, sh), saturated to int32. */
__device__ __forceinline__ int score(int hh, int T, int sh) {
    if (sh >= 16) {            // floor((hh*2^16 + T + h) / 2^sh) = (hh + ((T + h) >> 16)) >> (sh-16); |T + h| < 2^31, result < 2^22: pure 32-bit
        const int h = 1 << (sh - 1);
        return (hh + ((T + h) >> 16)) >> (sh - 16);
    }
    if (sh >= 0) {             // hh*2^16 divisible by 2^sh: s = hh << (16-sh) + rshr(T, sh), saturated
        const int t = sh ? ((T >> sh) + ((T >> (sh - 1)) & 1)) : T;
        // Quantised Q/K have 64 entries in [-32767,32767]. Thus
        // |S| <= 64*32767^2 and round(S/2^sh) fits int32 for sh >= 5.
        // The two terms may individually wrap (e.g. hh=64*128^2),
        // so add as unsigned limbs; their final signed sum is in range.
        if (sh >= 5) return int((unsigned(hh) << (16 - sh)) + unsigned(t));
        const i64 v = (i64(hh) << (16 - sh)) + t;
        return int(v < INT_MIN ? INT_MIN : (v > INT_MAX ? INT_MAX : v));
    }
    const i64 S = (i64(hh) << 16) + T;
    const int neg = -sh; const i64 lim = 1LL << (62 - neg); const i64 v = (S < -lim ? -lim : (S > lim ? lim : S)) << neg;
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
__device__ __forceinline__ int lutexp32(unsigned z, const int2* L) {
    unsigned zc = z < (16u << 16) ? z : ((16u << 16) - 1);
    int i = int(zc >> 10), f = int(zc & 1023);
    const int2 e = L[i];                 // (L[i], L[i+1] - L[i])
    int v = e.x + int((i64(e.y) * f) >> 10);
    return z < (16u << 16) ? v : 0;
}
#ifndef MEOW_ATTN_UNIFORM
#define MEOW_ATTN_UNIFORM 1
#endif
/** score() with a warp-uniform shift class (0: every sh >= 16, 1: every sh in [5,15], 2: every sh < 0, 3: mixed -> general path). */
__device__ __forceinline__ int score_u(int hh, int T, int sh, int cls) {
#if MEOW_ATTN_UNIFORM
    if (cls == 0) { const int h = 1 << (sh - 1); return (hh + ((T + h) >> 16)) >> (sh - 16); }
    if (cls == 1) { const int t = (T >> sh) + ((T >> (sh - 1)) & 1); return int((unsigned(hh) << (16 - sh)) + unsigned(t)); }
    if (cls == 2) { const i64 S = (i64(hh) << 16) + T; const int neg = -sh; const i64 lim = 1LL << (62 - neg); const i64 v = (S < -lim ? -lim : (S > lim ? lim : S)) << neg;
                    return int(v < INT_MIN ? INT_MIN : (v > INT_MAX ? INT_MAX : v)); }
#endif
    return score(hh, T, sh);
}

template <int PASS, bool Unmasked>
__global__ void __launch_bounds__(NTHR, (TQ == 64 ? (PASS == 2 ? 2 : 3) : 1)) attn_pass_k(
    const int8_t* __restrict__ qhi, const int8_t* __restrict__ qlo, const int32_t* __restrict__ eq, const i64* __restrict__ qhs, const i64* __restrict__ qls,
    const int8_t* __restrict__ khi, const int8_t* __restrict__ klo, const int32_t* __restrict__ ek, const i64* __restrict__ ksum,
    const int8_t* __restrict__ vthi, const int8_t* __restrict__ vtlo, const int32_t* __restrict__ ev, const i64* __restrict__ vsum,
    int Tq, int Tqp, int Tk, int Tkp, int causal, const i64* __restrict__ Lexp, int heads, int C, int32_t* __restrict__ gmx, i64* __restrict__ gr, int32_t* __restrict__ out, uint32_t* __restrict__ rowmax) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem& S = *reinterpret_cast<Smem*>(smem_raw);
    const int g = blockIdx.y, qt = blockIdx.x, q0 = qt * TQ;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, rw = w % RW, kh = w / RW;
    const int gid = lane >> 2, t4 = lane & 3;
    const int lm = lane >> 3, lr = lane & 7;                       // ldmatrix: matrix index, row within matrix
    const int b = g / heads, h = g % heads;
    if (q0 >= Tq) return;
    const size_t rowBase = size_t(g) * Tqp + q0;
    // Q = 256*qh + qlu with qlu = q & 255 (unsigned lo plane), K = 256*kh + kl + 128 (signed lo plane):
    // S = 65536*hh + 256*(qh.kl + qlu.kh) + qlu.kl + 128*sum_d Q_d, and 128*sum_d Q_d = 32768*qhs + 128*qls + 2^20 (qls is the signed-plane sum).
    for (int i = tid; i < TQ; i += NTHR) { size_t rq = rowBase + i; S.cq[i] = int(32768LL * qhs[rq] + 128LL * qls[rq]) + (1 << 20); S.eq[i] = eq[rq]; }
    if (PASS >= 1) for (int i = tid; i < 1024; i += NTHR) S.lexp2[i] = make_int2(int(Lexp[i]), int(Lexp[i + 1]) - int(Lexp[i]));
    if (PASS == 2) for (int i = tid; i < 64; i += NTHR) S.u.p2.vev[i] = ev[g * 64 + i];
    uint32_t qa_hi[2][4], qa_lo[2][4];
    {
        const int8_t* qh = qhi + (rowBase + rw * 16) * 64; const int8_t* ql = qlo + (rowBase + rw * 16) * 64;
        for (int ks = 0; ks < 2; ++ks) {
            int c0 = ks * 32 + t4 * 4;
            qa_hi[ks][0] = ld32(qh + gid * 64 + c0); qa_hi[ks][1] = ld32(qh + (gid + 8) * 64 + c0); qa_hi[ks][2] = ld32(qh + gid * 64 + c0 + 16); qa_hi[ks][3] = ld32(qh + (gid + 8) * 64 + c0 + 16);
            // stored lo plane is (q & 255) - 128; xor 0x80 per byte turns it into the unsigned q & 255
            qa_lo[ks][0] = ld32(ql + gid * 64 + c0) ^ 0x80808080u; qa_lo[ks][1] = ld32(ql + (gid + 8) * 64 + c0) ^ 0x80808080u; qa_lo[ks][2] = ld32(ql + gid * 64 + c0 + 16) ^ 0x80808080u; qa_lo[ks][3] = ld32(ql + (gid + 8) * 64 + c0 + 16) ^ 0x80808080u;
        }
    }
    const int row0 = rw * 16 + gid, row1 = row0 + 8;
    const int qrow0 = q0 + row0, qrow1 = q0 + row1;
    __syncthreads();
    const int cq0 = S.cq[row0], cq1 = S.cq[row1], eq0 = S.eq[row0], eq1 = S.eq[row1];
    int mxr0 = 0, mxr1 = 0; unsigned r0lo = 0, r0hi = 0, r1lo = 0, r1hi = 0;
    if (PASS >= 1) { mxr0 = gmx[rowBase + row0]; mxr1 = gmx[rowBase + row1]; }
    if (PASS == 2) { i64 rr0 = gr[rowBase + row0], rr1 = gr[rowBase + row1]; r0lo = unsigned(rr0); r0hi = unsigned(rr0 >> 32); r1lo = unsigned(rr1); r1hi = unsigned(rr1 >> 32); }
    // Fully-masked key tiles may be skipped in every pass: masked/padded keys have P = 0 (both planes zero) and contribute nothing.
    const int nTiles = (Unmasked || !causal) ? Tkp / TK : min(Tkp / TK, (qt + 1) * (TQ / TK));
    int mx0 = INT_MIN, mx1 = INT_MIN; i64 sum0 = 0, sum1 = 0;
    // At most 4096 keys: sum_k P_k <= 2^15 + Tk/2, so plane sums fit int32.
    int phs0 = 0, phs1 = 0, pls0 = 0, pls1 = 0;
    // PV accumulators: accH = sum_k P_k * vh_k, accL = sum_k P_k * vl_k (P = 256*ph + plu folded per 32-key step; |acc| <= 2^15.1 * 128).
    int32_t accH[4][4], accL[4][4];
    if (PASS == 2) for (int n = 0; n < 4; ++n) for (int j = 0; j < 4; ++j) { accH[n][j] = 0; accL[n][j] = 0; }
    auto loadK = [&](int buf, int kt) {
        const int k0 = kt * TK;
        const int8_t* sh = khi + (size_t(g) * Tkp + k0) * 64; const int8_t* sl = klo + (size_t(g) * Tkp + k0) * 64;
        for (int i = tid; i < TK * 4; i += NTHR) { int key = i >> 2, c = i & 3; cp16(S.khi[buf] + kswz(key, c), sh + i * 16); cp16(S.klo[buf] + kswz(key, c), sl + i * 16); }
        for (int i = tid; i < TK; i += NTHR) cp4(&S.ek[buf][i], ek + g * Tkp + k0 + i);
    };
    auto loadV = [&](int kt) {
        const int k0 = kt * TK;
        for (int i = tid; i < 64 * 4; i += NTHR) { int d = i >> 2, c = i & 3; cp16(S.u.p2.vhi + kswz(d, c), vthi + (size_t(g) * 64 + d) * Tkp + k0 + c * 16); cp16(S.u.p2.vlo + kswz(d, c), vtlo + (size_t(g) * 64 + d) * Tkp + k0 + c * 16); }
    };
    loadK(0, 0); cp_commit();
    for (int kt = 0; kt < nTiles; ++kt) {
        const int k0 = kt * TK; const int buf = kt & 1;
        cp_wait<0>();                                            // K(kt) landed
        __syncthreads();                                         // ...and visible to all; everyone finished tile kt-1 (incl. its PV reads of V / P)
        if (PASS == 2) { loadV(kt); cp_commit(); }               // V(kt): waited for right before the PV stage
        if (kt + 1 < nTiles) { loadK(buf ^ 1, kt + 1); cp_commit(); }
        const int8_t* Kh = S.khi[buf]; const int8_t* Kl = S.klo[buf]; const int32_t* Ke = S.ek[buf];
        bool anyHi = false;                                      // pass 2: P >= 256 somewhere in this warp's (16 rows x 32 keys) P block
        const bool fullTile = Unmasked || ((k0 + TK <= Tk) && !causal);       // no per-element validity checks needed
        // Warp-uniform shift class of this tile: every (row, key) pair this warp scores has sh = 19 - eq - ek in one of the score() ranges,
        // so the per-element branches in score() become uniform. Each thread owns rows row0/row1 and keys kh*32 + nt8*8 + t4*2 + {0,1}.
        int cls = 3;
        {
            int ekmin = INT_MAX, ekmax = INT_MIN;
            #pragma unroll
            for (int nt8 = 0; nt8 < 4; ++nt8) { const int kl = (kh * 4 + nt8) * 8 + t4 * 2; const int a = Ke[kl], c = Ke[kl + 1]; ekmin = min(ekmin, min(a, c)); ekmax = max(ekmax, max(a, c)); }
            const int shmin = 19 - max(eq0, eq1) - ekmax, shmax = 19 - min(eq0, eq1) - ekmin;
            if (__all_sync(0xffffffffu, shmin >= 16)) cls = 0;
            else if (__all_sync(0xffffffffu, shmin >= 5 && shmax <= 15)) cls = 1;
            else if (__all_sync(0xffffffffu, shmax < 0)) cls = 2;
        }
        // scores: this warp -> n-tiles kh*4 .. kh*4+3 (8 keys each), two n-tiles per scalar block for ILP
        // On consumer Blackwell, the unmasked PV path needs a smaller live
        // score tile to reduce register spills. Ada retains the paired tile.
#ifndef MEOW_ATTN_NU_SM120_P2
#define MEOW_ATTN_NU_SM120_P2 1
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
        constexpr int NU = (PASS == 2 && Unmasked) ? MEOW_ATTN_NU_SM120_P2 : 2;
#else
        constexpr int NU = 2;
#endif
        for (int np = 0; np < 4 / NU; ++np) {
            int32_t hh[NU][4], hl[NU][4], ll[NU][4];
            #pragma unroll
            for (int u = 0; u < NU; ++u) {
                const int nt = kh * 4 + np * NU + u, kb = nt * 8;
                for (int j = 0; j < 4; ++j) { hh[u][j] = 0; hl[u][j] = 0; ll[u][j] = 0; }
                for (int ks = 0; ks < 2; ++ks) {
                    uint32_t bf[4];      // {bh0, bh1, bl0, bl1}: matrices 0/1 = K hi dims +0/+16, 2/3 = K lo
                    ldm4(bf, (lm < 2 ? Kh : Kl) + kswz(kb + lr, ks * 2 + (lm & 1)));
                    // The cross products enter the score only as their sum.
                    // Their combined magnitude is <= 2*64*128^2, so using
                    // one int32 accumulator is exact and saves registers.
                    mma_s8(hh[u], qa_hi[ks], bf[0], bf[1]); mma_s8(hl[u], qa_hi[ks], bf[2], bf[3]); mma_u8s8(hl[u], qa_lo[ks], bf[0], bf[1]); mma_u8s8(ll[u], qa_lo[ks], bf[2], bf[3]);
                }
            }
            #pragma unroll
            for (int u = 0; u < NU; ++u) {
                const int nt = kh * 4 + np * NU + u, kb = nt * 8;
                const int kl0 = kb + t4 * 2; const int key0 = k0 + kl0;
                const int ekk[2] = {Ke[kl0], Ke[kl0 + 1]};
                int ph[4], pl[4]; unsigned es0 = 0, es1 = 0; int phs_t0 = 0, pls_t0 = 0, phs_t1 = 0, pls_t1 = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int kc = j & 1, key = key0 + kc; const bool r1 = j >= 2;
                    const int T = (hl[u][j] << 8) + ll[u][j] + (r1 ? cq1 : cq0);
                    const bool valid = fullTile || (key < Tk && !(causal && key > (r1 ? qrow1 : qrow0)));
                    const int sv = valid ? score_u(hh[u][j], T, 19 - (r1 ? eq1 : eq0) - ekk[kc], cls) : INT_MIN;
                    if (PASS == 0) { if (r1) mx1 = max(mx1, sv); else mx0 = max(mx0, sv); }
                    else {
                        const unsigned z = valid ? unsigned(r1 ? mxr1 : mxr0) - unsigned(sv) : 0xffffffffu;
                        const int E = lutexp32(z, S.lexp2);
                        if (PASS == 1) { if (r1) es1 += unsigned(E); else es0 += unsigned(E); }
                        else {
                            const unsigned rlo = r1 ? r1lo : r0lo, rhi = r1 ? r1hi : r0hi;
                            // P = (E*r + 2^46) >> 47. r = floor(2^62 / sumE) and sumE >= max E, so normally r < 2^33 (rhi <= 1): the low word of
                            // E*r never reaches bit 46, hence P = (hi32(E*rlo) + E*rhi + 2^14) >> 15 in 32-bit arithmetic (E <= 2^30, no overflow).
                            unsigned Pp;
                            if (rhi <= 1u) { const unsigned hi = __umulhi(unsigned(E), rlo) + unsigned(E) * rhi; Pp = (hi + (1u << 14)) >> 15; }
                            else { unsigned long long prod = (unsigned long long)unsigned(E) * rlo + ((unsigned long long)(unsigned(E) * rhi) << 32); Pp = unsigned((prod + (1ull << 46)) >> 47); }
                            const int P = Pp > 32767u ? 32767 : int(Pp);
                            ph[j] = P >> 8; pl[j] = P & 255;                     // both planes unsigned: P = 256*ph + pl
                            if (r1) { phs_t1 += ph[j]; pls_t1 += pl[j]; } else { phs_t0 += ph[j]; pls_t0 += pl[j]; }
                        }
                    }
                }
                if (PASS == 1) { sum0 += es0; sum1 += es1; }
                else if (PASS == 2) {
                    phs0 += phs_t0; pls0 += pls_t0; phs1 += phs_t1; pls1 += pls_t1;
                    anyHi |= (ph[0] | ph[1] | ph[2] | ph[3]) != 0;
                    const int c = kb + t4 * 2; const int o0 = pswz(row0, c), o1 = pswz(row1, c);
                    *reinterpret_cast<uint16_t*>(S.u.p2.phi + o0) = uint16_t(uint8_t(ph[0]) | (uint16_t(uint8_t(ph[1])) << 8));
                    *reinterpret_cast<uint16_t*>(S.u.p2.plo + o0) = uint16_t(uint8_t(pl[0]) | (uint16_t(uint8_t(pl[1])) << 8));
                    *reinterpret_cast<uint16_t*>(S.u.p2.phi + o1) = uint16_t(uint8_t(ph[2]) | (uint16_t(uint8_t(ph[3])) << 8));
                    *reinterpret_cast<uint16_t*>(S.u.p2.plo + o1) = uint16_t(uint8_t(pl[2]) | (uint16_t(uint8_t(pl[3])) << 8));
                }
            }
        }
        if (PASS == 2) {
            // this warp wrote P for rows rw*16.. and keys kh*32..+31 of the tile: publish whether any P >= 256 there
            if (__any_sync(0xffffffffu, anyHi)) { if (lane == 0) S.u.p2.pflag[rw][kh] = 1; } else if (lane == 0) S.u.p2.pflag[rw][kh] = 0;
            cp_wait<0>();                                        // V(kt) (and K(kt+1)) landed
            __syncthreads();                                     // P tile + flags complete, V visible
            for (int ks = 0; ks < 2; ++ks) {
                // A fragments of P: matrices 0/1 = rows +0/+8 at keys ks*32+0..15, 2/3 = same rows at keys +16
                const int prow = rw * 16 + (lm & 1) * 8 + lr, pkey = ks * 32 + (lm >> 1) * 16;
                const bool hiPlane = S.u.p2.pflag[rw][ks] != 0;       // warp-uniform
                uint32_t pa_h[4], pa_l[4];
                ldm4(pa_l, S.u.p2.plo + pswz(prow, pkey));
                if (hiPlane) ldm4(pa_h, S.u.p2.phi + pswz(prow, pkey));
                for (int nd = 0; nd < 4; ++nd) {
                    uint32_t vf[4];      // {vh0, vh1, vl0, vl1}
                    ldm4(vf, (lm < 2 ? S.u.p2.vhi : S.u.p2.vlo) + kswz(kh * 32 + nd * 8 + lr, ks * 2 + (lm & 1)));
                    mma_u8s8(accH[nd], pa_l, vf[0], vf[1]); mma_u8s8(accL[nd], pa_l, vf[2], vf[3]);
                    if (hiPlane) {       // P >= 256 somewhere in this 16x32 block: add the hi plane, folded as 256 * (ph . v)
                        int32_t th[4] = {0, 0, 0, 0}, tl[4] = {0, 0, 0, 0};
                        mma_s8(th, pa_h, vf[0], vf[1]); mma_s8(tl, pa_h, vf[2], vf[3]);
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) { accH[nd][j] += th[j] << 8; accL[nd][j] += tl[j] << 8; }
                    }
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
        if (t4 == 0) { S.u.p2.phsh[kh][row0] = phs0; S.u.p2.phsh[kh][row1] = phs1; S.u.p2.plsh[kh][row0] = pls0; S.u.p2.plsh[kh][row1] = pls1; }
        __syncthreads();
        // O = sum_k P_k V_k with V = 256*vh + vl + 128: 256*accH + accL + 128 * sum_k P_k, sum_k P_k = 256*phs + pls (unsigned planes)
        const i64 ps0 = 128LL * (256LL * (S.u.p2.phsh[0][row0] + S.u.p2.phsh[1][row0]) + (S.u.p2.plsh[0][row0] + S.u.p2.plsh[1][row0]));
        const i64 ps1 = 128LL * (256LL * (S.u.p2.phsh[0][row1] + S.u.p2.phsh[1][row1]) + (S.u.p2.plsh[0][row1] + S.u.p2.plsh[1][row1]));
        unsigned am0 = 0, am1 = 0;
        for (int nd = 0; nd < 4; ++nd) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const bool r1 = j >= 2; const int q = r1 ? qrow1 : qrow0; if (q >= Tq) continue;
                const int d = kh * 32 + nd * 8 + t4 * 2 + (j & 1);
                i64 O = 256LL * accH[nd][j] + accL[nd][j] + (r1 ? ps1 : ps0);
                const int32_t ov = int32_t(rshr(O, 15 - S.u.p2.vev[d]));
                out[(size_t(b) * Tq + q) * C + h * 64 + d] = ov;
                const unsigned a = unsigned(ov < 0 ? -i64(ov) : ov); if (r1) am1 = max(am1, a); else am0 = max(am0, a);
            }
        }
        if (rowmax) {
            for (int o = 1; o < 4; o <<= 1) { am0 = max(am0, __shfl_xor_sync(0xffffffffu, am0, o)); am1 = max(am1, __shfl_xor_sync(0xffffffffu, am1, o)); }
            if (t4 == 0) { if (qrow0 < Tq && am0) atomicMax(rowmax + size_t(b) * Tq + qrow0, am0); if (qrow1 < Tq && am1) atomicMax(rowmax + size_t(b) * Tq + qrow1, am1); }
        }
    }
}

// ---------------------------------------------------------------- single-launch attention for Tkp <= 128 (cross-attention, text encoder)
// Identical arithmetic to the three passes, but with at most two key tiles the scores stay in registers: QK^T once instead of three
// times, row max / sum exchanged through shared memory instead of global round trips, one LUT fill, one launch.
struct __align__(16) SmemS {
    int8_t khi[2][TK * 64], klo[2][TK * 64];   // K tiles [key][dim]; reused for the V tiles [dim][key] once the scores are in registers
    int32_t ek[2][TK];
    int32_t cq[TQ], eq[TQ];
    int32_t mxh[2][TQ]; i64 sumh[2][TQ];
    int2 lexp[1024];
    int8_t phi[TQ * LDP], plo[TQ * LDP];       // P tile of the current key tile
    int32_t phsh[2][TQ], plsh[2][TQ];
    int32_t pflag[RW][2];
    int32_t vev[64];
};
template <bool Unmasked>
__global__ void __launch_bounds__(NTHR, 2) attn_small_k(
    const int8_t* __restrict__ qhi, const int8_t* __restrict__ qlo, const int32_t* __restrict__ eq, const i64* __restrict__ qhs, const i64* __restrict__ qls,
    const int8_t* __restrict__ khi, const int8_t* __restrict__ klo, const int32_t* __restrict__ ek,
    const int8_t* __restrict__ vthi, const int8_t* __restrict__ vtlo, const int32_t* __restrict__ ev,
    int Tq, int Tqp, int Tk, int Tkp, int causal, const i64* __restrict__ Lexp, int heads, int C, int32_t* __restrict__ out, uint32_t* __restrict__ rowmax) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    SmemS& S = *reinterpret_cast<SmemS*>(smem_raw);
    const int g = blockIdx.y, qt = blockIdx.x, q0 = qt * TQ;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, rw = w % RW, kh = w / RW;
    const int gid = lane >> 2, t4 = lane & 3;
    const int lm = lane >> 3, lr = lane & 7;
    const int b = g / heads, h = g % heads;
    if (q0 >= Tq) return;
    const size_t rowBase = size_t(g) * Tqp + q0;
    const int nT = Tkp / TK;                                                // 1 or 2 (uniform)
    for (int i = tid; i < TQ; i += NTHR) { size_t rq = rowBase + i; S.cq[i] = int(32768LL * qhs[rq] + 128LL * qls[rq]) + (1 << 20); S.eq[i] = eq[rq]; }
    for (int i = tid; i < 1024; i += NTHR) S.lexp[i] = make_int2(int(Lexp[i]), int(Lexp[i + 1]) - int(Lexp[i]));
    for (int i = tid; i < 64; i += NTHR) S.vev[i] = ev[g * 64 + i];
    for (int t = 0; t < nT; ++t) {
        const int k0 = t * TK;
        const int8_t* sh = khi + (size_t(g) * Tkp + k0) * 64; const int8_t* sl = klo + (size_t(g) * Tkp + k0) * 64;
        for (int i = tid; i < TK * 4; i += NTHR) { int key = i >> 2, c = i & 3; cp16(S.khi[t] + kswz(key, c), sh + i * 16); cp16(S.klo[t] + kswz(key, c), sl + i * 16); }
        for (int i = tid; i < TK; i += NTHR) cp4(&S.ek[t][i], ek + g * Tkp + k0 + i);
    }
    cp_commit();
    uint32_t qa_hi[2][4], qa_lo[2][4];
    {
        const int8_t* qh = qhi + (rowBase + rw * 16) * 64; const int8_t* ql = qlo + (rowBase + rw * 16) * 64;
        for (int ks = 0; ks < 2; ++ks) {
            int c0 = ks * 32 + t4 * 4;
            qa_hi[ks][0] = ld32(qh + gid * 64 + c0); qa_hi[ks][1] = ld32(qh + (gid + 8) * 64 + c0); qa_hi[ks][2] = ld32(qh + gid * 64 + c0 + 16); qa_hi[ks][3] = ld32(qh + (gid + 8) * 64 + c0 + 16);
            qa_lo[ks][0] = ld32(ql + gid * 64 + c0) ^ 0x80808080u; qa_lo[ks][1] = ld32(ql + (gid + 8) * 64 + c0) ^ 0x80808080u; qa_lo[ks][2] = ld32(ql + gid * 64 + c0 + 16) ^ 0x80808080u; qa_lo[ks][3] = ld32(ql + (gid + 8) * 64 + c0 + 16) ^ 0x80808080u;
        }
    }
    const int row0 = rw * 16 + gid, row1 = row0 + 8;
    const int qrow0 = q0 + row0, qrow1 = q0 + row1;
    cp_wait<0>();
    __syncthreads();
    const int cq0 = S.cq[row0], cq1 = S.cq[row1], eq0 = S.eq[row0], eq1 = S.eq[row1];
    // ---- scores of every (row, key) this thread owns: sc[tile][nt8][j], keys kh*32 + nt8*8 + t4*2 + (j&1), rows row0 (j<2) / row1 ----
    int sc[2][4][4];
    int mx0 = INT_MIN, mx1 = INT_MIN;
    #pragma unroll
    for (int t = 0; t < 2; ++t) {
        if (t >= nT) break;
        const int k0 = t * TK;
        const int8_t* Kh = S.khi[t]; const int8_t* Kl = S.klo[t]; const int32_t* Ke = S.ek[t];
        const bool fullTile = Unmasked || ((k0 + TK <= Tk) && !causal);
        int cls = 3;
        {
            int ekmin = INT_MAX, ekmax = INT_MIN;
            #pragma unroll
            for (int nt8 = 0; nt8 < 4; ++nt8) { const int kl = (kh * 4 + nt8) * 8 + t4 * 2; const int a = Ke[kl], c = Ke[kl + 1]; ekmin = min(ekmin, min(a, c)); ekmax = max(ekmax, max(a, c)); }
            const int shmin = 19 - max(eq0, eq1) - ekmax, shmax = 19 - min(eq0, eq1) - ekmin;
            if (__all_sync(0xffffffffu, shmin >= 16)) cls = 0;
            else if (__all_sync(0xffffffffu, shmin >= 5 && shmax <= 15)) cls = 1;
            else if (__all_sync(0xffffffffu, shmax < 0)) cls = 2;
        }
        #pragma unroll
        for (int np = 0; np < 2; ++np) {
            int32_t hh[2][4], hl[2][4], ll[2][4];
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int nt = kh * 4 + np * 2 + u, kb = nt * 8;
                for (int j = 0; j < 4; ++j) { hh[u][j] = 0; hl[u][j] = 0; ll[u][j] = 0; }
                for (int ks = 0; ks < 2; ++ks) {
                    uint32_t bf[4];
                    ldm4(bf, (lm < 2 ? Kh : Kl) + kswz(kb + lr, ks * 2 + (lm & 1)));
                    mma_s8(hh[u], qa_hi[ks], bf[0], bf[1]); mma_s8(hl[u], qa_hi[ks], bf[2], bf[3]); mma_u8s8(hl[u], qa_lo[ks], bf[0], bf[1]); mma_u8s8(ll[u], qa_lo[ks], bf[2], bf[3]);
                }
            }
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int nt = kh * 4 + np * 2 + u, kb = nt * 8;
                const int kl0 = kb + t4 * 2; const int key0 = k0 + kl0;
                const int ekk[2] = {Ke[kl0], Ke[kl0 + 1]};
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int kc = j & 1, key = key0 + kc; const bool r1 = j >= 2;
                    const int T = (hl[u][j] << 8) + ll[u][j] + (r1 ? cq1 : cq0);
                    const bool valid = fullTile || (key < Tk && !(causal && key > (r1 ? qrow1 : qrow0)));
                    const int sv = valid ? score_u(hh[u][j], T, 19 - (r1 ? eq1 : eq0) - ekk[kc], cls) : INT_MIN;
                    sc[t][np * 2 + u][j] = sv;
                    if (valid) { if (r1) mx1 = max(mx1, sv); else mx0 = max(mx0, sv); }
                }
            }
        }
    }
    // ---- row max over the 4 lanes of a row and the two key halves ----
    for (int o = 1; o < 4; o <<= 1) { mx0 = max(mx0, __shfl_xor_sync(0xffffffffu, mx0, o)); mx1 = max(mx1, __shfl_xor_sync(0xffffffffu, mx1, o)); }
    if (t4 == 0) { S.mxh[kh][row0] = mx0; S.mxh[kh][row1] = mx1; }
    __syncthreads();                                                        // all QK reads of the K tiles are done: the buffers may take V
    const int mxr0 = max(S.mxh[0][row0], S.mxh[1][row0]), mxr1 = max(S.mxh[0][row1], S.mxh[1][row1]);
    for (int t = 0; t < nT; ++t) {
        const int k0 = t * TK;
        for (int i = tid; i < 64 * 4; i += NTHR) { int d = i >> 2, c = i & 3; cp16(S.khi[t] + kswz(d, c), vthi + (size_t(g) * 64 + d) * Tkp + k0 + c * 16); cp16(S.klo[t] + kswz(d, c), vtlo + (size_t(g) * 64 + d) * Tkp + k0 + c * 16); }
    }
    cp_commit();
    // ---- sum of E ----
    auto validAt = [&](int t, int nt8, int j) -> bool {
        if (Unmasked) return true;
        const int key = t * TK + (kh * 4 + nt8) * 8 + t4 * 2 + (j & 1);
        return key < Tk && !(causal && key > (j >= 2 ? qrow1 : qrow0));
    };
    i64 sum0 = 0, sum1 = 0;
    #pragma unroll
    for (int t = 0; t < 2; ++t) {
        if (t >= nT) break;
        #pragma unroll
        for (int nt8 = 0; nt8 < 4; ++nt8) {
            unsigned es0 = 0, es1 = 0;                                      // 2 values <= 2^30 per row per nt8: fits unsigned; the tile sum does not
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const bool r1 = j >= 2;
                const unsigned z = validAt(t, nt8, j) ? unsigned(r1 ? mxr1 : mxr0) - unsigned(sc[t][nt8][j]) : 0xffffffffu;
                const int E = lutexp32(z, S.lexp);
                if (r1) es1 += unsigned(E); else es0 += unsigned(E);
            }
            sum0 += es0; sum1 += es1;
        }
    }
    for (int o = 1; o < 4; o <<= 1) { sum0 += __shfl_xor_sync(0xffffffffu, sum0, o); sum1 += __shfl_xor_sync(0xffffffffu, sum1, o); }
    if (t4 == 0) { S.sumh[kh][row0] = sum0; S.sumh[kh][row1] = sum1; }
    __syncthreads();
    unsigned r0lo, r0hi, r1lo, r1hi;
    { const i64 se0 = S.sumh[0][row0] + S.sumh[1][row0], se1 = S.sumh[0][row1] + S.sumh[1][row1];
      const i64 rr0 = se0 > 0 ? floor_div(1LL << 62, se0) : 0, rr1 = se1 > 0 ? floor_div(1LL << 62, se1) : 0;
      r0lo = unsigned(rr0); r0hi = unsigned(rr0 >> 32); r1lo = unsigned(rr1); r1hi = unsigned(rr1 >> 32); }
    // ---- P and P.V per key tile ----
    int32_t accH[4][4], accL[4][4];
    for (int n = 0; n < 4; ++n) for (int j = 0; j < 4; ++j) { accH[n][j] = 0; accL[n][j] = 0; }
    int phs0 = 0, phs1 = 0, pls0 = 0, pls1 = 0;
    #pragma unroll
    for (int t = 0; t < 2; ++t) {
        if (t >= nT) break;
        bool anyHi = false;
        #pragma unroll
        for (int nt8 = 0; nt8 < 4; ++nt8) {
            const int kb = (kh * 4 + nt8) * 8;
            int ph[4], pl[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const bool r1 = j >= 2;
                const unsigned z = validAt(t, nt8, j) ? unsigned(r1 ? mxr1 : mxr0) - unsigned(sc[t][nt8][j]) : 0xffffffffu;
                const int E = lutexp32(z, S.lexp);
                const unsigned rlo = r1 ? r1lo : r0lo, rhi = r1 ? r1hi : r0hi;
                unsigned Pp;
                if (rhi <= 1u) { const unsigned hi = __umulhi(unsigned(E), rlo) + unsigned(E) * rhi; Pp = (hi + (1u << 14)) >> 15; }
                else { unsigned long long prod = (unsigned long long)unsigned(E) * rlo + ((unsigned long long)(unsigned(E) * rhi) << 32); Pp = unsigned((prod + (1ull << 46)) >> 47); }
                const int P = Pp > 32767u ? 32767 : int(Pp);
                ph[j] = P >> 8; pl[j] = P & 255;
                if (r1) { phs1 += ph[j]; pls1 += pl[j]; } else { phs0 += ph[j]; pls0 += pl[j]; }
            }
            anyHi |= (ph[0] | ph[1] | ph[2] | ph[3]) != 0;
            const int c = kb + t4 * 2; const int o0 = pswz(row0, c), o1 = pswz(row1, c);
            *reinterpret_cast<uint16_t*>(S.phi + o0) = uint16_t(uint8_t(ph[0]) | (uint16_t(uint8_t(ph[1])) << 8));
            *reinterpret_cast<uint16_t*>(S.plo + o0) = uint16_t(uint8_t(pl[0]) | (uint16_t(uint8_t(pl[1])) << 8));
            *reinterpret_cast<uint16_t*>(S.phi + o1) = uint16_t(uint8_t(ph[2]) | (uint16_t(uint8_t(ph[3])) << 8));
            *reinterpret_cast<uint16_t*>(S.plo + o1) = uint16_t(uint8_t(pl[2]) | (uint16_t(uint8_t(pl[3])) << 8));
        }
        if (__any_sync(0xffffffffu, anyHi)) { if (lane == 0) S.pflag[rw][kh] = 1; } else if (lane == 0) S.pflag[rw][kh] = 0;
        cp_wait<0>();                                                       // V tiles landed (first iteration)
        __syncthreads();                                                    // P tile + flags complete, V visible
        const int8_t* Vh = S.khi[t]; const int8_t* Vl = S.klo[t];
        for (int ks = 0; ks < 2; ++ks) {
            const int prow = rw * 16 + (lm & 1) * 8 + lr, pkey = ks * 32 + (lm >> 1) * 16;
            const bool hiPlane = S.pflag[rw][ks] != 0;
            uint32_t pa_h[4], pa_l[4];
            ldm4(pa_l, S.plo + pswz(prow, pkey));
            if (hiPlane) ldm4(pa_h, S.phi + pswz(prow, pkey));
            for (int nd = 0; nd < 4; ++nd) {
                uint32_t vf[4];
                ldm4(vf, (lm < 2 ? Vh : Vl) + kswz(kh * 32 + nd * 8 + lr, ks * 2 + (lm & 1)));
                mma_u8s8(accH[nd], pa_l, vf[0], vf[1]); mma_u8s8(accL[nd], pa_l, vf[2], vf[3]);
                if (hiPlane) {
                    int32_t th[4] = {0, 0, 0, 0}, tl[4] = {0, 0, 0, 0};
                    mma_s8(th, pa_h, vf[0], vf[1]); mma_s8(tl, pa_h, vf[2], vf[3]);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) { accH[nd][j] += th[j] << 8; accL[nd][j] += tl[j] << 8; }
                }
            }
        }
        __syncthreads();                                                    // P tile free for the next key tile
    }
    // ---- output ----
    for (int o = 1; o < 4; o <<= 1) { phs0 += __shfl_xor_sync(0xffffffffu, phs0, o); phs1 += __shfl_xor_sync(0xffffffffu, phs1, o); pls0 += __shfl_xor_sync(0xffffffffu, pls0, o); pls1 += __shfl_xor_sync(0xffffffffu, pls1, o); }
    if (t4 == 0) { S.phsh[kh][row0] = phs0; S.phsh[kh][row1] = phs1; S.plsh[kh][row0] = pls0; S.plsh[kh][row1] = pls1; }
    __syncthreads();
    const i64 ps0 = 128LL * (256LL * (S.phsh[0][row0] + S.phsh[1][row0]) + (S.plsh[0][row0] + S.plsh[1][row0]));
    const i64 ps1 = 128LL * (256LL * (S.phsh[0][row1] + S.phsh[1][row1]) + (S.plsh[0][row1] + S.plsh[1][row1]));
    unsigned am0 = 0, am1 = 0;
    for (int nd = 0; nd < 4; ++nd) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const bool r1 = j >= 2; const int q = r1 ? qrow1 : qrow0; if (q >= Tq) continue;
            const int d = kh * 32 + nd * 8 + t4 * 2 + (j & 1);
            i64 O = 256LL * accH[nd][j] + accL[nd][j] + (r1 ? ps1 : ps0);
            const int32_t ov = int32_t(rshr(O, 15 - S.vev[d]));
            out[(size_t(b) * Tq + q) * C + h * 64 + d] = ov;
            const unsigned a = unsigned(ov < 0 ? -i64(ov) : ov); if (r1) am1 = max(am1, a); else am0 = max(am0, a);
        }
    }
    if (rowmax) {
        for (int o = 1; o < 4; o <<= 1) { am0 = max(am0, __shfl_xor_sync(0xffffffffu, am0, o)); am1 = max(am1, __shfl_xor_sync(0xffffffffu, am1, o)); }
        if (t4 == 0) { if (qrow0 < Tq && am0) atomicMax(rowmax + size_t(b) * Tq + qrow0, am0); if (qrow1 < Tq && am1) atomicMax(rowmax + size_t(b) * Tq + qrow1, am1); }
    }
}
constexpr size_t SMEMS = sizeof(SmemS);
#ifndef MEOW_ATTN_SMALL
#define MEOW_ATTN_SMALL 1
#endif
constexpr size_t SMEM0 = offsetof(Smem, lexp2), SMEM1 = offsetof(Smem, u), SMEM2 = sizeof(Smem);
}  // namespace

void k_fused_attention(cudaStream_t st, const int8_t* qhi, const int8_t* qlo, const int32_t* eq, const i64* qhs, const i64* qls,
                       const int8_t* khi, const int8_t* klo, const int32_t* ek, const i64* ksum,
                       const int8_t* vthi, const int8_t* vtlo, const int32_t* ev, const i64* vsum,
                       int B, int heads, int Tq, int Tqp, int Tk, int Tkp, bool causal, const i64* Lexp, int C, int32_t* gmx, i64* gr, int32_t* out, uint32_t* rowmax) {
    if (Tqp % TQ || Tkp % 64) throw std::runtime_error("fused attention: Tqp/Tkp alignment");
    // Kernel attributes belong to the current device, not the whole process.
    // Each lane has its own cache, avoiding a host data race; reconfigure when
    // the same thread switches devices (e.g. checking several GPUs in order).
    static thread_local int attrDevice = -1;
    static thread_local bool specializeUnmasked = false;
    int device = -1;
    CUDA_CHECK(cudaGetDevice(&device));
    if (attrDevice != device) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        specializeUnmasked = (prop.major == 8 && prop.minor == 9) || prop.major == 12;
        CUDA_CHECK(cudaFuncSetAttribute(attn_pass_k<0, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEM0)));
        CUDA_CHECK(cudaFuncSetAttribute(attn_pass_k<0, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEM0)));
        CUDA_CHECK(cudaFuncSetAttribute(attn_pass_k<1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEM1)));
        CUDA_CHECK(cudaFuncSetAttribute(attn_pass_k<1, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEM1)));
        CUDA_CHECK(cudaFuncSetAttribute(attn_pass_k<2, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEM2)));
        CUDA_CHECK(cudaFuncSetAttribute(attn_pass_k<2, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEM2)));
        attrDevice = device;
    }
    dim3 grid(Tqp / TQ, B * heads);
#if MEOW_ATTN_SMALL
    if (Tkp <= 2 * TK) {       // <= 2 key tiles: single-launch kernel with register-resident scores (same arithmetic)
        static thread_local int smallDevice = -1;
        if (smallDevice != device) {
            CUDA_CHECK(cudaFuncSetAttribute(attn_small_k<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEMS)));
            CUDA_CHECK(cudaFuncSetAttribute(attn_small_k<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEMS)));
            smallDevice = device;
        }
        if (!causal && Tk == Tkp) attn_small_k<true><<<grid, NTHR, SMEMS, st>>>(qhi, qlo, eq, qhs, qls, khi, klo, ek, vthi, vtlo, ev, Tq, Tqp, Tk, Tkp, causal, Lexp, heads, C, out, rowmax);
        else attn_small_k<false><<<grid, NTHR, SMEMS, st>>>(qhi, qlo, eq, qhs, qls, khi, klo, ek, vthi, vtlo, ev, Tq, Tqp, Tk, Tkp, causal, Lexp, heads, C, out, rowmax);
        return;
    }
#endif
    // All model self-attention keys are real and noncausal. Compile out
    // element masking only when the entire padded key domain is valid.
    if (specializeUnmasked && !causal && Tk == Tkp) {
        attn_pass_k<0, true><<<grid, NTHR, SMEM0, st>>>(qhi, qlo, eq, qhs, qls, khi, klo, ek, ksum, vthi, vtlo, ev, vsum, Tq, Tqp, Tk, Tkp, causal, Lexp, heads, C, gmx, gr, out, rowmax);
        attn_pass_k<1, true><<<grid, NTHR, SMEM1, st>>>(qhi, qlo, eq, qhs, qls, khi, klo, ek, ksum, vthi, vtlo, ev, vsum, Tq, Tqp, Tk, Tkp, causal, Lexp, heads, C, gmx, gr, out, rowmax);
        attn_pass_k<2, true><<<grid, NTHR, SMEM2, st>>>(qhi, qlo, eq, qhs, qls, khi, klo, ek, ksum, vthi, vtlo, ev, vsum, Tq, Tqp, Tk, Tkp, causal, Lexp, heads, C, gmx, gr, out, rowmax);
    } else {
        attn_pass_k<0, false><<<grid, NTHR, SMEM0, st>>>(qhi, qlo, eq, qhs, qls, khi, klo, ek, ksum, vthi, vtlo, ev, vsum, Tq, Tqp, Tk, Tkp, causal, Lexp, heads, C, gmx, gr, out, rowmax);
        attn_pass_k<1, false><<<grid, NTHR, SMEM1, st>>>(qhi, qlo, eq, qhs, qls, khi, klo, ek, ksum, vthi, vtlo, ev, vsum, Tq, Tqp, Tk, Tkp, causal, Lexp, heads, C, gmx, gr, out, rowmax);
        attn_pass_k<2, false><<<grid, NTHR, SMEM2, st>>>(qhi, qlo, eq, qhs, qls, khi, klo, ek, ksum, vthi, vtlo, ev, vsum, Tq, Tqp, Tk, Tkp, causal, Lexp, heads, C, gmx, gr, out, rowmax);
    }
}
}  // namespace meow
