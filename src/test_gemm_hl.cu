// Standalone: compare k_gemm_hl against cuBLASLt + epilogue on random data (exactness + timing).
#include "gemm_hl.h"
#include "gemm.h"
#include "kernels.h"
#include "common.h"
#include <random>
#include <cstdio>
using namespace meow;
// --- X32 / CONV32 source tests: compare against quant_rows / pixmax+im2col + planar gemm_hl
static int test_sources() {
    cudaStream_t st = 0; std::mt19937 rng(7); int fails = 0;
    struct S { int B, H, W, C, N, stride; bool conv; } cases[] = {{2, 64, 64, 320, 320, 1, true}, {1, 32, 32, 640, 640, 2, true}, {1, 16, 16, 1280, 1280, 1, true}, {1, 8, 8, 2560, 1280, 1, true}, {4, 64, 64, 320, 960, 1, false}, {2, 32, 32, 1280, 640, 1, false}, {1, 96, 1, 1024, 4096, 1, false}};
    for (auto& c : cases) {
        const int rows = c.conv ? c.B * ((c.H - 1) / c.stride + 1) * ((c.W - 1) / c.stride + 1) : c.B * c.H * c.W;
        const int K = c.conv ? 9 * c.C : c.C, Kp = int(align_up(K, 16)), Mp = int(align_up(rows, 64)), Np = int(align_up(c.N, 64));
        std::vector<int32_t> hX(size_t(c.B) * c.H * c.W * c.C); for (auto& v : hX) { int bits = 1 + int(rng() % 31); v = int32_t(int64_t(rng() & ((1ull << bits) - 1)) - (1ll << (bits - 1))); }
        std::vector<int8_t> hW(size_t(Np) * Kp, 0); for (int n = 0; n < c.N; ++n) for (int k = 0; k < K; ++k) hW[size_t(n) * Kp + k] = int8_t(rng());
        std::vector<int32_t> hc(size_t(Np) * 4, 0); for (int n = 0; n < c.N; ++n) { i64 ws = 0; for (int k = 0; k < K; ++k) ws += hW[size_t(n) * Kp + k]; hc[4 * n] = 16384 + int(rng() % 16384); hc[4 * n + 1] = 20 + int(rng() % 10); hc[4 * n + 2] = int32_t(rng() % 100000) - 50000; hc[4 * n + 3] = int(ws); }
        int32_t *dX, *dY1, *dY2, *dE; int8_t *dW, *dQ; uint32_t *dpm, *drm; int4* dcst;
        cudaMalloc(&dX, hX.size() * 4); cudaMalloc(&dW, hW.size()); cudaMalloc(&dcst, hc.size() * 4); cudaMalloc(&dY1, size_t(rows) * c.N * 4); cudaMalloc(&dY2, size_t(rows) * c.N * 4); cudaMalloc(&dE, Mp * 4);
        cudaMalloc(&dQ, size_t(2) * Mp * Kp); cudaMalloc(&dpm, size_t(c.B) * c.H * c.W * 4); cudaMalloc(&drm, Mp * 4);
        cudaMemcpy(dX, hX.data(), hX.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(dW, hW.data(), hW.size(), cudaMemcpyHostToDevice); cudaMemcpy(dcst, hc.data(), hc.size() * 4, cudaMemcpyHostToDevice);
        int8_t* hi = dQ; int8_t* lo = dQ + size_t(Mp) * Kp;
        ASrc A; AEpi E; E.epi = EPI_PLAIN; E.ldy = c.N;
        if (c.conv) {
            k_pixmax(st, dX, c.B * c.H * c.W, c.C, dpm); k_im2col3(st, dX, dpm, c.B, c.H, c.W, c.C, c.stride, hi, lo, dE, Kp);
            E.Y = dY1; k_gemm_hl(st, hi, lo, Kp, Mp, rows, dW, Kp, Np, c.N, Kp, dcst, dE, dY1, c.N, nullptr);
            k_conv_e(st, dpm, c.B, c.H, c.W, c.stride, dE);
            A.src = SRC_CONV32; A.img = dX; A.H = c.H; A.W = c.W; A.C = c.C; A.stride = c.stride; A.Ho = (c.H - 1) / c.stride + 1; A.Wo = (c.W - 1) / c.stride + 1; A.K = K; A.e = dE;
        } else {
            k_quant_rows(st, dX, rows, K, K, hi, lo, dE, Kp);
            E.Y = dY1; k_gemm_hl(st, hi, lo, Kp, Mp, rows, dW, Kp, Np, c.N, Kp, dcst, dE, dY1, c.N, nullptr);
            // rowmax from the data
            std::vector<uint32_t> rm(Mp, 0); for (int r = 0; r < rows; ++r) { uint32_t m = 0; for (int k = 0; k < K; ++k) { int32_t v = hX[size_t(r) * K + k]; m = std::max(m, uint32_t(v < 0 ? -int64_t(v) : v)); } rm[r] = m; }
            cudaMemcpy(drm, rm.data(), Mp * 4, cudaMemcpyHostToDevice);
            A.src = SRC_X32; A.x = dX; A.ldx = K; A.rowmax = drm; A.K = K;
        }
        E.Y = dY2; k_gemm_hl2(st, A, Mp, rows, dW, Kp, Np, c.N, Kp, dcst, E);
        cudaDeviceSynchronize(); cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { printf("CUDA error %s\n", cudaGetErrorString(err)); return 1; }
        std::vector<int32_t> y1(size_t(rows) * c.N), y2(y1.size()); cudaMemcpy(y1.data(), dY1, y1.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(y2.data(), dY2, y2.size() * 4, cudaMemcpyDeviceToHost);
        long bad = 0; for (size_t i = 0; i < y1.size(); ++i) if (y1[i] != y2[i]) { if (!bad) printf("  first mismatch at %zu: ref %d got %d\n", i, y1[i], y2[i]); ++bad; }
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float ta, tb;
        auto ref = [&]() { if (c.conv) { k_pixmax(st, dX, c.B * c.H * c.W, c.C, dpm); k_im2col3(st, dX, dpm, c.B, c.H, c.W, c.C, c.stride, hi, lo, dE, Kp); } else k_quant_rows(st, dX, rows, K, K, hi, lo, dE, Kp); k_gemm_hl(st, hi, lo, Kp, Mp, rows, dW, Kp, Np, c.N, Kp, dcst, dE, dY1, c.N, nullptr); };
        auto mine = [&]() { if (c.conv) k_conv_e(st, dpm, c.B, c.H, c.W, c.stride, dE); k_gemm_hl2(st, A, Mp, rows, dW, Kp, Np, c.N, Kp, dcst, E); };
        cudaEventRecord(e0); for (int i = 0; i < 20; ++i) ref(); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ta, e0, e1);
        cudaEventRecord(e0); for (int i = 0; i < 20; ++i) mine(); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&tb, e0, e1);
        printf("%s B=%d %dx%d C=%d N=%d s=%d  %s  planar(quant/im2col+gemm) %.3f ms  fused-loader %.3f ms (%.2fx)\n", c.conv ? "CONV32" : "X32   ", c.B, c.H, c.W, c.C, c.N, c.stride, bad ? "MISMATCH" : "OK      ", ta / 20, tb / 20, ta / tb);
        fails += bad != 0;
        cudaFree(dX); cudaFree(dW); cudaFree(dcst); cudaFree(dY1); cudaFree(dY2); cudaFree(dE); cudaFree(dQ); cudaFree(dpm); cudaFree(drm);
    }
    return fails;
}
int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "sources") return test_sources();
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
        float tv[3];
        for (int v = 0; v < 3; ++v) { g_gemm_hl_variant = v; if ((v == 1 && Mp % 128) || (v == 2 && Np64 % 128)) { tv[v] = -1; continue; } mine(); cudaDeviceSynchronize(); cudaEvent_t a0, a1; cudaEventCreate(&a0); cudaEventCreate(&a1); cudaEventRecord(a0); for (int i = 0; i < 20; ++i) mine(); cudaEventRecord(a1); cudaEventSynchronize(a1); cudaEventElapsedTime(&tv[v], a0, a1); tv[v] /= 20;
            std::vector<int32_t> yv(hR.size()); cudaMemcpy(yv.data(), dY2, yv.size() * 4, cudaMemcpyDeviceToHost); std::vector<int32_t> yr(hR.size()); lt.run(st, dA, Kp, 0, dW, Kp, 0, dC, Np32, 0, 2 * Mp, Np32, Kp); k_epilogue(st, dC, dC + size_t(Mp) * Np32, s.M, s.N, Np32, dcst, dE, dY1, s.N, dR); cudaMemcpy(yr.data(), dY1, yr.size() * 4, cudaMemcpyDeviceToHost); long bb = 0; for (size_t i = 0; i < yv.size(); ++i) bb += yv[i] != yr[i]; if (bb) printf("  variant %d MISMATCH %ld\n", v, bb); }
        g_gemm_hl_variant = -1;
        printf("   variants: 64x64 %.3f  128x64 %.3f  64x128 %.3f ms\n", tv[0], tv[1], tv[2]);
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
