// int8 x int8 -> int32 GEMMs on tensor cores through cuBLASLt.
// Row-major view:  C[M, N] = A[M, K] * W[N, K]^T   (A = activations, W = weights, both int8, ld in elements).
// Requirements: K % 16 == 0, N % 4 == 0 (N % 32 preferred), M any (pad rows exist in the buffers), 16-byte aligned pointers.
#pragma once
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <map>
#include <tuple>

namespace meow {
class Gemm {
public:
    void init(int wsBytes);
    ~Gemm();
    void run(cudaStream_t st, const int8_t* A, int lda, long long strideA, const int8_t* W, int ldw, long long strideW,
             int32_t* C, int ldc, long long strideC, int M, int N, int K, int batch = 1);
private:
    struct Plan { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; cublasLtMatmulAlgo_t algo; bool haveAlgo; };
    Plan& plan(int M, int N, int K, int lda, int ldw, int ldc, long long sA, long long sW, long long sC, int batch);
    cublasLtHandle_t lt_ = nullptr; void* ws_ = nullptr; size_t wsBytes_ = 0;
    std::map<std::tuple<int, int, int, int, int, int, long long, long long, long long, int>, Plan> plans_;
};
}  // namespace meow
