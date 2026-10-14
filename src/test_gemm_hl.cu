// Standalone: compare k_gemm_hl against cuBLASLt + epilogue on random data (exactness + timing).
#include "gemm_hl.h"
#include "gemm.h"
#include "kernels.h"
#include "common.h"
#include <random>
#include <cstdio>
using namespace meow;
int main(int argc, char** argv) {
    struct S { int M, N, K; } shapes[] = {{16384, 320, 2880}, {16384, 960, 320}, {16384, 2560, 320}, {16384, 320, 1280}, {4096, 640, 5760}, {1024, 1280, 11520}, {96, 1024, 1024}, {16384, 4, 2880}, {4096, 1280, 640}};
    Gemm lt; lt.init(32 << 20); cudaStream_t st = 0; std::mt19937 rng(1);
    for (auto& s : shapes) {
        int Mp = int(align_up(s.M, 64)), Kp = int(align_up(s.K, 16)), Np32 = int(align_up(s.N, 32)), Np64 = int(align_up(s.N, 64));
        std::vector<int8_t> hA(size_t(2) * Mp * Kp), hW(size_t(Np64) * Kp, 0); std::vector<int32_t> hE(Mp), hc(size_t(Np64) * 4, 0), hR(size_t(s.M) * s.N);
        for (auto& v : hA) v = int8_t(rng()); for (int n = 0; n < s.N; ++n) for (int k = 0; k < s.K; ++k) hW[size_t(n) * Kp + k] = int8_t(rng());
        for (auto& v : hE) v = int(rng() % 12); for (auto& v : hR) v = int32_t(rng());
        std::vector<i64> wsum(Np64, 0), m(Np64, 1), sv(Np64, 0), b(Np64, 0);
        for (int n = 0; n < s.N; ++n) { i64 ws = 0; for (int k = 0; k < s.K; ++k) ws += hW[size_t(n) * Kp + k]; wsum[n] = ws; m[n] = 16384 + int(rng() % 16384); sv[n] = 20 + int(rng() % 10); b[n] = int32_t(rng() % 100000) - 50000; hc[4 * n] = int(m[n]); hc[4 * n + 1] = int(sv[n]); hc[4 * n + 2] = int(b[n]); hc[4 * n + 3] = int(ws); }
        int8_t *dA, *dW; int32_t *dE, *dC, *dY1, *dY2, *dR; int4* dcst; i64 *dws, *dm, *ds, *db;
        cudaMalloc(&dA, hA.size()); cudaMalloc(&dW, hW.size()); cudaMalloc(&dE, hE.size() * 4); cudaMalloc(&dC, size_t(2) * Mp * Np32 * 4); cudaMalloc(&dY1, size_t(s.M) * s.N * 4); cudaMalloc(&dY2, size_t(s.M) * s.N * 4); cudaMalloc(&dR, hR.size() * 4); cudaMalloc(&dcst, hc.size() * 4);
        cudaMalloc(&dws, Np64 * 8); cudaMalloc(&dm, Np64 * 8); cudaMalloc(&ds, Np64 * 8); cudaMalloc(&db, Np64 * 8);
        cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice); cudaMemcpy(dW, hW.data(), hW.size(), cudaMemcpyHostToDevice); cudaMemcpy(dE, hE.data(), hE.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dR, hR.data(), hR.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(dcst, hc.data(), hc.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dws, wsum.data(), Np64 * 8, cudaMemcpyHostToDevice); cudaMemcpy(dm, m.data(), Np64 * 8, cudaMemcpyHostToDevice); cudaMemcpy(ds, sv.data(), Np64 * 8, cudaMemcpyHostToDevice); cudaMemcpy(db, b.data(), Np64 * 8, cudaMemcpyHostToDevice);
        const int8_t* Ahi = dA; const int8_t* Alo = dA + size_t(Mp) * Kp;
        auto ref = [&]() { lt.run(st, dA, Kp, 0, dW, Kp, 0, dC, Np32, 0, 2 * Mp, Np32, Kp); k_epilogue(st, dC, dC + size_t(Mp) * Np32, s.M, s.N, Np32, dcst, dE, dY1, s.N, dR); };
        auto mine = [&]() { k_gemm_hl(st, Ahi, Alo, Kp, Mp, s.M, dW, Kp, Np64, s.N, Kp, dcst, dE, dY2, s.N, dR); };
        ref(); mine(); cudaDeviceSynchronize(); cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { printf("CUDA error %s\n", cudaGetErrorString(err)); return 1; }
        std::vector<int32_t> y1(hR.size()), y2(hR.size()); cudaMemcpy(y1.data(), dY1, y1.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(y2.data(), dY2, y2.size() * 4, cudaMemcpyDeviceToHost);
        long bad = 0; for (size_t i = 0; i < y1.size(); ++i) if (y1[i] != y2[i]) { if (!bad) printf("  first mismatch at %zu: ref %d got %d\n", i, y1[i], y2[i]); ++bad; }
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float tr, tm;
        cudaEventRecord(e0); for (int i = 0; i < 20; ++i) ref(); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&tr, e0, e1);
        cudaEventRecord(e0); for (int i = 0; i < 20; ++i) mine(); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&tm, e0, e1);
        printf("M=%5d N=%4d K=%5d  %s  cublasLt+epilogue %.3f ms  gemm_hl %.3f ms  (%.2fx)\n", s.M, s.N, s.K, bad ? "MISMATCH" : "OK      ", tr / 20, tm / 20, tr / tm);
        cudaFree(dA); cudaFree(dW); cudaFree(dE); cudaFree(dC); cudaFree(dY1); cudaFree(dY2); cudaFree(dR); cudaFree(dcst); cudaFree(dws); cudaFree(dm); cudaFree(ds); cudaFree(db);
    }
}
