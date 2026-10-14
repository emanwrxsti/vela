// Exact integer primitives of PoI v2 (contrib/poi-sd/sdint.py), device side.
// Every function here must be bit-identical to the reference; int64 throughout.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace meow {
typedef int64_t i64; typedef unsigned int u32;
__host__ __device__ __forceinline__ i64 sat32(i64 v) { return v < -2147483648LL ? -2147483648LL : (v > 2147483647LL ? 2147483647LL : v); }
/** rshr(v, s): rounding arithmetic right shift (s>=0), saturating left shift (s<0); result saturated to int32. */
__host__ __device__ __forceinline__ i64 rshr(i64 v, i64 s) {
    if (s >= 0) { int p = s > 62 ? 62 : int(s); return sat32((v + ((1LL << p) >> 1)) >> p); }
    int neg = -s > 31 ? 31 : int(-s);
    i64 lim = 1LL << (62 - neg);
    i64 r = v < -lim ? -lim : (v > lim ? lim : v);
    return sat32(r * (1LL << neg));
}
__host__ __device__ __forceinline__ i64 floor_div(i64 a, i64 b) { i64 q = a / b; if ((a % b != 0) && ((a < 0) != (b < 0))) --q; return q; }
/** e = number of k in [15,32) with m >= 2^k  ==  max(0, bitlen(m) - 15) for m < 2^32. */
__host__ __device__ __forceinline__ int row_exp(i64 m) { int e = 0; for (int k = 15; k < 32; ++k) e += (m >= (1LL << k)); return e; }
__host__ __device__ __forceinline__ i64 quant16(i64 v, int e) { i64 q = rshr(v, e); return q < -32767 ? -32767 : (q > 32767 ? 32767 : q); }
/** quant16 for an int32 input (e in [0,16]): rshr == (v >> e) + bit(e-1) of v, all 32-bit, no overflow. */
__device__ __forceinline__ int quant16_32(int v, int e) { int q = e ? ((v >> e) + ((v >> (e - 1)) & 1)) : v; return q < -32767 ? -32767 : (q > 32767 ? 32767 : q); }
/** Interpolated table over [-16,16) with int32 entries (values <= 65536). */
__device__ __forceinline__ int lut16i(i64 x, const int32_t* __restrict__ L) {
    i64 z = x + (16LL << 16);
    if (z < 0) return 0;
    if (z >= (32LL << 16)) return 65536;
    int i = int(z >> 10), f = int(z & 1023);
    int a = L[i], b = L[i + 1];
    return a + (((b - a) * f) >> 10);
}
__host__ __device__ __forceinline__ i64 isqrt_i64(i64 v) {
    i64 s = (i64)sqrt((double)v);
    while (s > 0 && s * s > v) --s;
    while ((s + 1) * (s + 1) <= v) ++s;
    return s;
}
/** Interpolated table over [-16,16): Q16 sigmoid/phi of a Q16 input. L has 2049 entries. */
__device__ __forceinline__ i64 lut16(i64 x, const i64* __restrict__ L) {
    i64 z = x + (16LL << 16);
    if (z < 0) return 0;
    if (z >= (32LL << 16)) return 65536;
    i64 i = z >> 10, f = z & 1023;
    return L[i] + (((L[i + 1] - L[i]) * f) >> 10);
}
/** 16-bit split: a = 256*hi + lo + 128, hi = a>>8, lo = (a&255)-128. Value 0 -> (0,-128). */
__device__ __forceinline__ void split16(i64 a, int8_t& hi, int8_t& lo) { hi = (int8_t)(a >> 8); lo = (int8_t)((a & 255) - 128); }
}  // namespace meow
