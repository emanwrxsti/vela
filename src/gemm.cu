#include "gemm.h"
#include "common.h"

#define LT_CHECK(x) do { cublasStatus_t _s = (x); if (_s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("cublasLt error ") + std::to_string(int(_s)) + " at " #x); } while (0)

namespace meow {
void Gemm::init(int wsBytes) {
    LT_CHECK(cublasLtCreate(&lt_));
    wsBytes_ = size_t(wsBytes);
    CUDA_CHECK(cudaMalloc(&ws_, wsBytes_));
}
Gemm::~Gemm() {
    for (auto& kv : plans_) { cublasLtMatmulDescDestroy(kv.second.op); cublasLtMatrixLayoutDestroy(kv.second.a); cublasLtMatrixLayoutDestroy(kv.second.b); cublasLtMatrixLayoutDestroy(kv.second.c); }
    if (ws_) cudaFree(ws_);
    if (lt_) cublasLtDestroy(lt_);
}
Gemm::Plan& Gemm::plan(int M, int N, int K, int lda, int ldw, int ldc, long long sA, long long sW, long long sC, int batch) {
    auto key = std::make_tuple(M, N, K, lda, ldw, ldc, sA, sW, sC, batch);
    auto it = plans_.find(key);
    if (it != plans_.end()) return it->second;
    Plan p{};
    LT_CHECK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    cublasOperation_t tT = CUBLAS_OP_T, tN = CUBLAS_OP_N;
    LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &tT, sizeof(tT)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tN, sizeof(tN)));
    // cuBLAS is column-major: C^T[N, M] = W^T(col K x N, ld ldw)^T * A^T(col K x M, ld lda)
    LT_CHECK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_8I, K, N, ldw));
    LT_CHECK(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_8I, K, M, lda));
    LT_CHECK(cublasLtMatrixLayoutCreate(&p.c, CUDA_R_32I, N, M, ldc));
    if (batch > 1) {
        for (auto* L : {&p.a, &p.b, &p.c}) LT_CHECK(cublasLtMatrixLayoutSetAttribute(*L, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        LT_CHECK(cublasLtMatrixLayoutSetAttribute(p.a, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sW, sizeof(sW)));
        LT_CHECK(cublasLtMatrixLayoutSetAttribute(p.b, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sA, sizeof(sA)));
        LT_CHECK(cublasLtMatrixLayoutSetAttribute(p.c, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sC, sizeof(sC)));
    }
    cublasLtMatmulPreference_t pref; LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBytes_, sizeof(wsBytes_)));
    cublasLtMatmulHeuristicResult_t res[16]; int nres = 0;
    LT_CHECK(cublasLtMatmulAlgoGetHeuristic(lt_, p.op, p.a, p.b, p.c, p.c, pref, 16, res, &nres));
    cublasLtMatmulPreferenceDestroy(pref);
    if (nres == 0) throw std::runtime_error("cublasLt: no int8 GEMM algorithm for M=" + std::to_string(M) + " N=" + std::to_string(N) + " K=" + std::to_string(K) + " batch=" + std::to_string(batch));
    p.algo = res[0].algo; p.haveAlgo = true; p.nCand = 0;
    for (int i = 0; i < nres && i < 16; ++i) if (res[i].state == CUBLAS_STATUS_SUCCESS) p.cand[p.nCand++] = res[i].algo;
    return plans_.emplace(key, p).first->second;
}
void Gemm::run(cudaStream_t st, const int8_t* A, int lda, long long strideA, const int8_t* W, int ldw, long long strideW,
               int32_t* C, int ldc, long long strideC, int M, int N, int K, int batch) {
    Plan& p = plan(M, N, K, lda, ldw, ldc, strideA, strideW, strideC, batch);
    int32_t alpha = 1, beta = 0;
    if (p.nCand > 1) {
        // Autotune once per shape (eager runs only — never inside stream capture): time every heuristic candidate, keep the fastest.
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone; cudaStreamIsCapturing(st, &cs);
        if (cs == cudaStreamCaptureStatusNone) {
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            float best = 1e30f; int bi = 0;
            for (int i = 0; i < p.nCand; ++i) {
                if (cublasLtMatmul(lt_, p.op, &alpha, W, p.a, A, p.b, &beta, C, p.c, C, p.c, &p.cand[i], ws_, wsBytes_, st) != CUBLAS_STATUS_SUCCESS) continue;   // warm
                cudaEventRecord(e0, st);
                for (int r = 0; r < 3; ++r) cublasLtMatmul(lt_, p.op, &alpha, W, p.a, A, p.b, &beta, C, p.c, C, p.c, &p.cand[i], ws_, wsBytes_, st);
                cudaEventRecord(e1, st); cudaEventSynchronize(e1);
                float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                if (ms < best) { best = ms; bi = i; }
            }
            cudaEventDestroy(e0); cudaEventDestroy(e1);
            p.algo = p.cand[bi];
        }
        p.nCand = 0;      // tuned (or capture-time fallback to the heuristic's first choice)
    }
    LT_CHECK(cublasLtMatmul(lt_, p.op, &alpha, W, p.a, A, p.b, &beta, C, p.c, C, p.c, &p.algo, ws_, wsBytes_, st));
}
}  // namespace meow
