// PoI v2 inference engine: device model + per-lane forward (text encoder eager, UNet in a CUDA graph).
#pragma once
#include "poisd.h"
#include "gemm.h"
#include "kernels.h"
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace meow {
struct Lin { int N = 0, K = 0, Np = 0, Kp = 0; int8_t* w = nullptr; i64 *wsum = nullptr, *m = nullptr, *s = nullptr, *b = nullptr; int4* cst = nullptr; bool conv3 = false; };   // cst[n] = {m, s, b, wsum} packed
struct Norm { i64 *g = nullptr, *beta = nullptr; int32_t *g32 = nullptr, *beta32 = nullptr; i64 den = 0; int C = 0; };   // g/beta are Q16 int32 values

/** Weights of one GPU, shared by all lanes on it. */
class DevModel {
public:
    void load(int device, const Poisd& model);
    const Lin& lin(const std::string& n) const;
    /** Row-concatenation of linears sharing K (N = ΣN_i), e.g. to_q|to_k|to_v, ff a|g; built once at load. */
    void mergeLins(const std::string& name, const std::vector<std::string>& parts, bool interleave = false);   // interleave: rows of the parts alternate (a0,g0,a1,g1,…)
    const Norm& norm(const std::string& n) const;
    bool hasLin(const std::string& n) const { return lins_.count(n) != 0; }
    int32_t *lutSigmoid32 = nullptr, *lutPhi32 = nullptr;
    i64 *lutSigmoid = nullptr, *lutPhi = nullptr, *lutExp = nullptr, *lutGauss = nullptr, *tePos = nullptr, *tokM = nullptr, *tokS = nullptr;
    int8_t* tokW = nullptr;
    i64 cSig = 0, cCin = 0;
    size_t bytes = 0;
    int device = 0;
private:
    std::map<std::string, Lin> lins_; std::map<std::string, Norm> norms_;
    std::vector<void*> allocs_;
    void* dalloc(size_t n);
};

struct Tensor { int32_t* d = nullptr; int B = 0, T = 0, C = 0; uint32_t* rowmax = nullptr; size_t n() const { return size_t(B) * T * C; } };   // rowmax: per-row max|x| when the producer provided it

struct Arena {
    uint8_t* base = nullptr; size_t cap = 0, off = 0, peak = 0;
    void* alloc(size_t n);
    size_t mark() const { return off; }
    void release(size_t m) { off = m; }
    void reset() { off = 0; }
};

struct QuantIn { int8_t* hi = nullptr; int8_t* lo = nullptr; int32_t* e = nullptr; int B = 0, T = 0, K = 0, Kp = 0, Mp = 0; int rows() const { return B * T; } };
struct AttnKV {   // quantised key/value operands of one attention call
    int8_t *khi = nullptr, *klo = nullptr, *vthi = nullptr, *vtlo = nullptr;
    int32_t *ek = nullptr, *ev = nullptr; i64 *ksum = nullptr, *vsum = nullptr;
    int Tk = 0, Tkp = 0;
};

struct EngineConfig { int batch = 4; bool fusedAttention = true; int customGemmMaxK = 1536; /* gemm_hl for Kp <= this (0 = always cuBLASLt); deep-K convs stay on cuBLASLt (measured faster) */ bool fusedLoader = true; /* quantise A inside the GEMM (int32 + rowmax, implicit im2col) */ size_t arenaBytes = 0; size_t scoreBytes = 256u << 20; int gemmWorkspace = 32 << 20; bool graph = true; };

class Engine {
public:
    Engine(const DevModel& dm, const EngineConfig& cfg);
    ~Engine();
    /** Text encoder + cross-attention KV for a prompt (eager, cached until the next call). */
    void setPrompt(const std::vector<int>& tokens);
    /** seeds: batch x 32 bytes -> x0: batch x 32768 bytes. First call runs eagerly and captures the graph. */
    void run(const uint8_t* seeds, uint8_t* x0);
    int batch() const { return cfg_.batch; }
    size_t arenaPeak() const { return arena_.peak; }
    cudaStream_t stream() const { return st_; }
    double lastMs = 0;
    /** Compare the fused attention against the cuBLASLt path on random data; returns mismatching elements. */
    long testAttention(int B, int Tq, int Tk, int heads, bool causal, int scaleBits);
private:
    const DevModel& dm_; EngineConfig cfg_; cudaStream_t st_ = nullptr; Gemm gemm_;
    Arena arena_; std::vector<void*> persist_;
    cudaGraphExec_t gexec_ = nullptr; bool captured_ = false;
    uint8_t *dSeeds_ = nullptr, *dOut_ = nullptr, *hSeeds_ = nullptr, *hOut_ = nullptr;
    int32_t *dNoise_ = nullptr, *dX_ = nullptr, *dEps_ = nullptr, *dIds_ = nullptr;
    cudaEvent_t ev0_ = nullptr, ev1_ = nullptr;
    std::map<std::string, AttnKV> crossKV_;      // per transformer block prefix (replicated for batch)
    std::vector<int> promptTokens_; bool havePrompt_ = false;
    void* palloc(size_t n);
    Tensor talloc(int B, int T, int C);
    // ops (all on st_)
    Tensor lin(const Tensor& x, const Lin& L, int32_t* out = nullptr, const int32_t* residual = nullptr);
    QuantIn qalloc(int B, int T, int K);                      // operand buffers for a [B*T, K] row-quantised input
    Tensor linQ(const QuantIn& q, const Lin& L, int32_t* out = nullptr, const int32_t* residual = nullptr);
    QuantIn layernormQ(const Tensor& x, const Norm& nm);
    QuantIn groupnormQ(const Tensor& x, const Norm& nm);
    Tensor groupnormSilu(const Tensor& x, const Norm& nm, u32*& pm);   // GN + SiLU + pixel max (for conv3)
    void linQ3(const QuantIn& q, const Lin& L, int Cs, Tensor& y0, Tensor& y1, Tensor& y2);   // merged GEMM, split epilogue
    QuantIn linQGegluQ(const QuantIn& q, const Lin& L, int F, const Lin* Li = nullptr);          // merged [a|g] GEMM + GEGLU + quant (Li = interleaved weights for the custom GEMM)
    bool useHl(const Lin& L) const;
    Tensor gegluX(const QuantIn& q, const Lin& Li, int F);          // interleaved a|g GEMM with GEGLU epilogue -> int32 [rows, F] + rowmax
    /** GEMM on an int32 input quantised on load (needs x.rowmax); falls back to lin() when unavailable. */
    Tensor linX(const Tensor& x, const Lin& L, int32_t* out = nullptr, const int32_t* residual = nullptr, int epi = 0, bool wantRowmax = false);
    uint32_t* rowmaxAlloc(int rows);
    Tensor conv3(const Tensor& x, int H, int W, const Lin& L, int stride, int32_t* out = nullptr, const int32_t* residual = nullptr, const u32* pmIn = nullptr);
    Tensor groupnorm(const Tensor& x, const Norm& nm);
    Tensor layernorm(const Tensor& x, const Norm& nm);
    void silu(Tensor& x);
    AttnKV quantKV(const Tensor& k, const Tensor& v, int heads);
    Tensor attention(const Tensor& q, const AttnKV& kv, int heads, bool causal, int Tq);
    Tensor resnet(const Tensor& x, int H, int W, const std::string& p);
    Tensor transformer(const Tensor& x, const std::string& p, int heads);
    void unet(const Tensor& x, Tensor& eps);
    Tensor textEncoder(const std::vector<int>& ids);
    void forward();
};
}  // namespace meow
