#include "engine.h"
#include "gemm_hl.h"
#include "common.h"
#include <cstring>
#include <algorithm>

namespace meow {
// ------------------------------------------------------------------ DevModel
void* DevModel::dalloc(size_t n) { void* p = nullptr; CUDA_CHECK(cudaMalloc(&p, n)); allocs_.push_back(p); bytes += n; return p; }

void DevModel::load(int dev, const Poisd& model) {
    device = dev;
    CUDA_CHECK(cudaSetDevice(dev));
    auto up64 = [&](const std::vector<i64>& v) { i64* p = (i64*)dalloc(v.size() * 8); CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * 8, cudaMemcpyHostToDevice)); return p; };
    auto is3x3 = [](const std::string& n) {
        auto ends = [&](const char* s) { size_t l = std::strlen(s); return n.size() >= l && n.compare(n.size() - l, l, s) == 0; };
        return ends(".conv1") || ends(".conv2") || ends("conv_in") || ends("conv_out") || ends("downsamplers.0.conv") || ends("upsamplers.0.conv");
    };
    std::vector<int8_t> hbuf;
    for (const auto& kv : model.entries()) {
        const std::string& name = kv.first; const PoisdTensor& t = kv.second;
        if (t.dtype != 0 || name.size() < 2 || name.compare(name.size() - 2, 2, ".w") != 0 || t.dims.size() != 2) continue;
        std::string base = name.substr(0, name.size() - 2);
        if (base == "te.tok") continue;
        if (!model.has(base + ".m")) continue;
        Lin L; L.N = int(t.dims[0]); L.K = int(t.dims[1]); L.Np = int(align_up(L.N, 64)); L.Kp = int(align_up(L.K, 16)); L.conv3 = is3x3(base);
        const int8_t* w = model.i8(name);
        hbuf.assign(size_t(L.Np) * L.Kp, 0);
        std::vector<i64> wsum(L.Np, 0), m(L.Np, 1), s(L.Np, 0), b(L.Np, 0);
        auto mv = model.i32v(base + ".m"), sv = model.i32v(base + ".s"), bv = model.i32v(base + ".b");
        int C = L.conv3 ? L.K / 9 : 0;
        if (L.conv3 && C * 9 != L.K) throw std::runtime_error("conv weight K not multiple of 9: " + name);
        for (int n = 0; n < L.N; ++n) {
            const int8_t* row = w + size_t(n) * L.K; int8_t* dst = hbuf.data() + size_t(n) * L.Kp; i64 sum = 0;
            for (int k = 0; k < L.K; ++k) {
                int kk = L.conv3 ? ((k % 9) * C + k / 9) : k;     // tap-major permutation for im2col
                dst[kk] = row[k]; sum += row[k];
            }
            wsum[n] = sum; m[n] = mv[n]; s[n] = sv[n]; b[n] = bv[n];
        }
        L.w = (int8_t*)dalloc(hbuf.size()); CUDA_CHECK(cudaMemcpy(L.w, hbuf.data(), hbuf.size(), cudaMemcpyHostToDevice));
        L.wsum = up64(wsum); L.m = up64(m); L.s = up64(s); L.b = up64(b);
        { std::vector<int32_t> c(size_t(L.Np) * 4); for (int n = 0; n < L.Np; ++n) { c[4 * n] = int32_t(m[n]); c[4 * n + 1] = int32_t(s[n]); c[4 * n + 2] = int32_t(b[n]); c[4 * n + 3] = int32_t(wsum[n]); }
          L.cst = (int4*)dalloc(c.size() * 4); CUDA_CHECK(cudaMemcpy(L.cst, c.data(), c.size() * 4, cudaMemcpyHostToDevice)); }
        lins_[base] = L;
    }
    for (const auto& kv : model.entries()) {
        const std::string& name = kv.first;
        if (name.size() < 2 || name.compare(name.size() - 2, 2, ".g") != 0) continue;
        std::string base = name.substr(0, name.size() - 2);
        if (!model.has(base + ".beta") || !model.has(base + ".eps")) continue;
        Norm nm; auto g = model.i32v(name); auto bt = model.i32v(base + ".beta"); nm.C = int(g.size()); nm.g = up64(g); nm.beta = up64(bt); nm.den = model.i32s(base + ".eps");
        { std::vector<int32_t> h(g.begin(), g.end()); nm.g32 = (int32_t*)dalloc(h.size() * 4); CUDA_CHECK(cudaMemcpy(nm.g32, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
          std::vector<int32_t> hb(bt.begin(), bt.end()); nm.beta32 = (int32_t*)dalloc(hb.size() * 4); CUDA_CHECK(cudaMemcpy(nm.beta32, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice)); }
        norms_[base] = nm;
    }
    lutSigmoid = up64(model.i32v("lut.sigmoid")); lutPhi = up64(model.i32v("lut.phi"));
    { auto up32 = [&](const std::vector<i64>& v) { std::vector<int32_t> h(v.begin(), v.end()); int32_t* p = (int32_t*)dalloc(h.size() * 4); CUDA_CHECK(cudaMemcpy(p, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); return p; };
      lutSigmoid32 = up32(model.i32v("lut.sigmoid")); lutPhi32 = up32(model.i32v("lut.phi")); } lutExp = up64(model.i32v("lut.exp")); lutGauss = up64(model.i32v("lut.gauss"));
    tePos = up64(model.i32v("te.pos")); tokM = up64(model.i32v("te.tok.m")); tokS = up64(model.i32v("te.tok.s"));
    const PoisdTensor& tw = model.get("te.tok.w"); tokW = (int8_t*)dalloc(tw.bytes()); CUDA_CHECK(cudaMemcpy(tokW, model.i8("te.tok.w"), tw.bytes(), cudaMemcpyHostToDevice));
    cSig = model.i32s("c.sig"); cCin = model.i32s("c.cin");
    // merged projections: one GEMM instead of three / two
    std::vector<std::string> blocks;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 2; ++j) blocks.push_back("unet.down_blocks." + std::to_string(i) + ".attentions." + std::to_string(j));
    blocks.push_back("unet.mid_block.attentions.0");
    for (int i = 1; i < 4; ++i) for (int j = 0; j < 3; ++j) blocks.push_back("unet.up_blocks." + std::to_string(i) + ".attentions." + std::to_string(j));
    for (auto& bl : blocks) { std::string b = bl + ".transformer_blocks.0"; mergeLins(b + ".attn1.to_qkv", {b + ".attn1.to_q", b + ".attn1.to_k", b + ".attn1.to_v"}); mergeLins(b + ".ff.net.0.proj.ag", {b + ".ff.net.0.proj.a", b + ".ff.net.0.proj.g"}); mergeLins(b + ".ff.net.0.proj.agi", {b + ".ff.net.0.proj.a", b + ".ff.net.0.proj.g"}, true); }
    for (int l = 0; l < 23; ++l) { std::string p = "te.layers." + std::to_string(l); mergeLins(p + ".self_attn.qkv", {p + ".self_attn.q_proj", p + ".self_attn.k_proj", p + ".self_attn.v_proj"}); }
}
void DevModel::mergeLins(const std::string& name, const std::vector<std::string>& parts, bool interleave) {
    std::vector<const Lin*> ls; int N = 0, K = -1;
    for (auto& pn : parts) { auto it = lins_.find(pn); if (it == lins_.end()) return; ls.push_back(&it->second); N += it->second.N; if (K < 0) K = it->second.K; else if (K != it->second.K) throw std::runtime_error("mergeLins: K mismatch"); }
    Lin L; L.N = N; L.K = K; L.Np = int(align_up(N, 64)); L.Kp = ls[0]->Kp; L.conv3 = false;
    L.w = (int8_t*)dalloc(size_t(L.Np) * L.Kp); CUDA_CHECK(cudaMemset(L.w, 0, size_t(L.Np) * L.Kp));
    L.wsum = (i64*)dalloc(size_t(L.Np) * 8); L.m = (i64*)dalloc(size_t(L.Np) * 8); L.s = (i64*)dalloc(size_t(L.Np) * 8); L.b = (i64*)dalloc(size_t(L.Np) * 8);
    std::vector<i64> ones(L.Np, 1), zeros(L.Np, 0);
    CUDA_CHECK(cudaMemcpy(L.m, ones.data(), size_t(L.Np) * 8, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(L.s, zeros.data(), size_t(L.Np) * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(L.b, zeros.data(), size_t(L.Np) * 8, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(L.wsum, zeros.data(), size_t(L.Np) * 8, cudaMemcpyHostToDevice));
    int off = 0;
    if (interleave) {
        const int P = int(ls.size()); const int Ni = ls[0]->N;
        for (int i = 0; i < P; ++i) {
            const Lin* p = ls[i]; if (p->N != Ni) throw std::runtime_error("mergeLins interleave: N mismatch");
            CUDA_CHECK(cudaMemcpy2D(L.w + size_t(i) * L.Kp, size_t(P) * L.Kp, p->w, L.Kp, L.Kp, Ni, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy2D(L.wsum + i, size_t(P) * 8, p->wsum, 8, 8, Ni, cudaMemcpyDeviceToDevice)); CUDA_CHECK(cudaMemcpy2D(L.m + i, size_t(P) * 8, p->m, 8, 8, Ni, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy2D(L.s + i, size_t(P) * 8, p->s, 8, 8, Ni, cudaMemcpyDeviceToDevice)); CUDA_CHECK(cudaMemcpy2D(L.b + i, size_t(P) * 8, p->b, 8, 8, Ni, cudaMemcpyDeviceToDevice));
        }
    } else for (const Lin* p : ls) {
        CUDA_CHECK(cudaMemcpy(L.w + size_t(off) * L.Kp, p->w, size_t(p->N) * L.Kp, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(L.wsum + off, p->wsum, size_t(p->N) * 8, cudaMemcpyDeviceToDevice)); CUDA_CHECK(cudaMemcpy(L.m + off, p->m, size_t(p->N) * 8, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(L.s + off, p->s, size_t(p->N) * 8, cudaMemcpyDeviceToDevice)); CUDA_CHECK(cudaMemcpy(L.b + off, p->b, size_t(p->N) * 8, cudaMemcpyDeviceToDevice));
        off += p->N;
    }
    { std::vector<i64> hm(L.Np), hs(L.Np), hb(L.Np), hw(L.Np); std::vector<int32_t> c(size_t(L.Np) * 4);
      CUDA_CHECK(cudaMemcpy(hm.data(), L.m, size_t(L.Np) * 8, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(hs.data(), L.s, size_t(L.Np) * 8, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hb.data(), L.b, size_t(L.Np) * 8, cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(hw.data(), L.wsum, size_t(L.Np) * 8, cudaMemcpyDeviceToHost));
      for (int n = 0; n < L.Np; ++n) { c[4 * n] = int32_t(hm[n]); c[4 * n + 1] = int32_t(hs[n]); c[4 * n + 2] = int32_t(hb[n]); c[4 * n + 3] = int32_t(hw[n]); }
      L.cst = (int4*)dalloc(c.size() * 4); CUDA_CHECK(cudaMemcpy(L.cst, c.data(), c.size() * 4, cudaMemcpyHostToDevice)); }
    lins_[name] = L;
}
const Lin& DevModel::lin(const std::string& n) const { auto it = lins_.find(n); if (it == lins_.end()) throw std::runtime_error("no linear " + n); return it->second; }
const Norm& DevModel::norm(const std::string& n) const { auto it = norms_.find(n); if (it == norms_.end()) throw std::runtime_error("no norm " + n); return it->second; }

// ------------------------------------------------------------------ Arena / Engine
void* Arena::alloc(size_t n) {
    n = align_up(n, 256);
    if (off + n > cap) throw std::runtime_error("arena exhausted (" + std::to_string((off + n) >> 20) + " MiB needed, " + std::to_string(cap >> 20) + " MiB reserved) — lower --batch or raise --mem");
    void* p = base + off; off += n; peak = std::max(peak, off); return p;
}
void* Engine::palloc(size_t n) { void* p = nullptr; CUDA_CHECK(cudaMalloc(&p, n)); persist_.push_back(p); return p; }
Tensor Engine::talloc(int B, int T, int C) { Tensor t; t.B = B; t.T = T; t.C = C; t.d = (int32_t*)arena_.alloc(t.n() * 4); return t; }

Engine::Engine(const DevModel& dm, const EngineConfig& cfg) : dm_(dm), cfg_(cfg) {
    CUDA_CHECK(cudaSetDevice(dm.device));
    CUDA_CHECK(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking));
    gemm_.init(cfg.gemmWorkspace);
    size_t freeB = 0, totB = 0; CUDA_CHECK(cudaMemGetInfo(&freeB, &totB));
    arena_.cap = cfg.arenaBytes ? cfg.arenaBytes : (freeB > (1536u << 20) ? freeB - (1024u << 20) : freeB / 2);
    arena_.base = (uint8_t*)palloc(arena_.cap);
    int B = cfg.batch;
    dSeeds_ = (uint8_t*)palloc(32 * B); dOut_ = (uint8_t*)palloc(32768 * B);
    dNoise_ = (int32_t*)palloc(size_t(B) * 16384 * 4); dX_ = (int32_t*)palloc(size_t(B) * 16384 * 4); dEps_ = (int32_t*)palloc(size_t(B) * 16384 * 4);
    dIds_ = (int32_t*)palloc(96 * 4);
    CUDA_CHECK(cudaHostAlloc((void**)&hSeeds_, 32 * B, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&hOut_, 32768 * B, cudaHostAllocDefault));
    CUDA_CHECK(cudaEventCreate(&ev0_)); CUDA_CHECK(cudaEventCreate(&ev1_));
}
Engine::~Engine() {
    cudaStreamSynchronize(st_);
    if (gexec_) cudaGraphExecDestroy(gexec_);
    for (void* p : persist_) cudaFree(p);
    if (hSeeds_) cudaFreeHost(hSeeds_); if (hOut_) cudaFreeHost(hOut_);
    if (ev0_) cudaEventDestroy(ev0_); if (ev1_) cudaEventDestroy(ev1_);
    if (st_) cudaStreamDestroy(st_);
}

// ------------------------------------------------------------------ ops
QuantIn Engine::qalloc(int B, int T, int K) {
    QuantIn q; q.B = B; q.T = T; q.K = K; q.Kp = int(align_up(K, 16)); q.Mp = int(align_up(B * T, 64));
    q.hi = (int8_t*)arena_.alloc(size_t(2) * q.Mp * q.Kp); q.lo = q.hi + size_t(q.Mp) * q.Kp;      // stacked [hi; lo] -> one GEMM
    q.e = (int32_t*)arena_.alloc(size_t(q.Mp) * 4);
    return q;
}
Tensor Engine::linQ(const QuantIn& q, const Lin& L, int32_t* out, const int32_t* residual) {
    if (q.K != L.K || q.Kp != L.Kp) throw std::runtime_error("linQ: K mismatch " + std::to_string(q.K) + " vs " + std::to_string(L.K));
    int rows = q.rows(), Mp = q.Mp;
    Tensor y; y.B = q.B; y.T = q.T; y.C = L.N; y.d = out ? out : (int32_t*)arena_.alloc(y.n() * 4);
    if (useHl(L)) { k_gemm_hl(st_, q.hi, q.lo, L.Kp, Mp, rows, L.w, L.Kp, L.Np, L.N, L.Kp, L.cst, q.e, y.d, L.N, residual); return y; }
    size_t mk = arena_.mark();
    int32_t* Chi = (int32_t*)arena_.alloc(size_t(2) * Mp * L.Np * 4); int32_t* Clo = Chi + size_t(Mp) * L.Np;
    gemm_.run(st_, q.hi, L.Kp, 0, L.w, L.Kp, 0, Chi, L.Np, 0, 2 * Mp, L.Np, L.Kp);
    k_epilogue(st_, Chi, Clo, rows, L.N, L.Np, L.cst, q.e, y.d, L.N, residual);
    arena_.release(mk);
    return y;
}
uint32_t* Engine::rowmaxAlloc(int rows) { uint32_t* p = (uint32_t*)arena_.alloc(size_t(rows) * 4); cudaMemsetAsync(p, 0, size_t(rows) * 4, st_); return p; }
Tensor Engine::linX(const Tensor& x, const Lin& L, int32_t* out, const int32_t* residual, int epi, bool wantRowmax) {
    if (!(cfg_.fusedLoader && x.rowmax && useHl(L) && L.K % 16 == 0)) {
        if (epi == EPI_GELU) { Tensor y = lin(x, L, out, residual); k_act(st_, y.d, y.n(), dm_.lutPhi32); if (wantRowmax) { QuantIn dummy; (void)dummy; } return y; }
        return lin(x, L, out, residual);
    }
    int rows = x.B * x.T, Mp = int(align_up(rows, 64));
    Tensor y; y.B = x.B; y.T = x.T; y.C = L.N; y.d = out ? out : (int32_t*)arena_.alloc(y.n() * 4);
    if (wantRowmax) y.rowmax = rowmaxAlloc(rows);
    ASrc A; A.src = SRC_X32; A.x = x.d; A.ldx = x.C; A.rowmax = x.rowmax; A.K = L.K;
    AEpi E; E.epi = epi; E.Y = y.d; E.ldy = L.N; E.residual = residual; E.Lphi = dm_.lutPhi32; E.rowmax = y.rowmax;
    k_gemm_hl2(st_, A, Mp, rows, L.w, L.Kp, L.Np, L.N, L.Kp, L.cst, E);
    return y;
}
bool Engine::useHl(const Lin& L) const { return cfg_.customGemmMaxK > 0 && L.Kp <= cfg_.customGemmMaxK && gemm_hl_supported(64, L.Np, L.Kp); }
Tensor Engine::lin(const Tensor& x, const Lin& L, int32_t* out, const int32_t* residual) {
    if (x.C != L.K) throw std::runtime_error("lin: K mismatch " + std::to_string(x.C) + " vs " + std::to_string(L.K));
    size_t mk = arena_.mark();
    Tensor y; y.B = x.B; y.T = x.T; y.C = L.N; y.d = out ? out : (int32_t*)arena_.alloc(y.n() * 4);
    size_t mk2 = arena_.mark();
    QuantIn q = qalloc(x.B, x.T, x.C);
    k_quant_rows(st_, x.d, q.rows(), L.K, L.K, q.hi, q.lo, q.e, L.Kp);
    linQ(q, L, y.d, residual);
    arena_.release(mk2); (void)mk;
    return y;
}
void Engine::linQ3(const QuantIn& q, const Lin& L, int Cs, Tensor& y0, Tensor& y1, Tensor& y2) {
    if (q.K != L.K || L.N != 3 * Cs) throw std::runtime_error("linQ3: shape mismatch");
    int rows = q.rows(), Mp = q.Mp;
    y0 = talloc(q.B, q.T, Cs); y1 = talloc(q.B, q.T, Cs); y2 = talloc(q.B, q.T, Cs);
    if (useHl(L) && Cs % 8 == 0) { k_gemm_hl(st_, q.hi, q.lo, L.Kp, Mp, rows, L.w, L.Kp, L.Np, L.N, L.Kp, L.cst, q.e, y0.d, Cs, nullptr, EPI_SPLIT3, Cs, y1.d, y2.d); return; }
    size_t mk = arena_.mark();
    int32_t* Chi = (int32_t*)arena_.alloc(size_t(2) * Mp * L.Np * 4); int32_t* Clo = Chi + size_t(Mp) * L.Np;
    gemm_.run(st_, q.hi, L.Kp, 0, L.w, L.Kp, 0, Chi, L.Np, 0, 2 * Mp, L.Np, L.Kp);
    k_epilogue_split(st_, Chi, Clo, rows, L.N, L.Np, L.cst, q.e, Cs, y0.d, y1.d, y2.d);
    arena_.release(mk);
}
QuantIn Engine::linQGegluQ(const QuantIn& q, const Lin& L, int F, const Lin* Li) {
    if (q.K != L.K || L.N != 2 * F) throw std::runtime_error("linQGegluQ: shape mismatch");
    int rows = q.rows(), Mp = q.Mp;
    QuantIn out = qalloc(q.B, q.T, F);
    size_t mk = arena_.mark();
    if (Li && useHl(*Li)) {                       // interleaved a|g weights: GEGLU in the GEMM epilogue, then row quantisation
        int32_t* v = (int32_t*)arena_.alloc(size_t(Mp) * F * 4);
        k_gemm_hl(st_, q.hi, q.lo, Li->Kp, Mp, rows, Li->w, Li->Kp, Li->Np, Li->N, Li->Kp, Li->cst, q.e, v, F, nullptr, EPI_GEGLU, 0, nullptr, nullptr, dm_.lutPhi32);
        k_quant_rows(st_, v, rows, F, F, out.hi, out.lo, out.e, out.Kp);
        arena_.release(mk);
        return out;
    }
    int32_t* Chi = (int32_t*)arena_.alloc(size_t(2) * Mp * L.Np * 4); int32_t* Clo = Chi + size_t(Mp) * L.Np;
    gemm_.run(st_, q.hi, L.Kp, 0, L.w, L.Kp, 0, Chi, L.Np, 0, 2 * Mp, L.Np, L.Kp);
    k_epilogue_geglu_q(st_, Chi, Clo, rows, F, L.Np, L.cst, q.e, dm_.lutPhi32, out.hi, out.lo, out.e, out.Kp);
    arena_.release(mk);
    return out;
}
Tensor Engine::gegluX(const QuantIn& q, const Lin& Li, int F) {
    int rows = q.rows(), Mp = q.Mp;
    Tensor v; v.B = q.B; v.T = q.T; v.C = F; v.d = (int32_t*)arena_.alloc(size_t(Mp) * F * 4); v.rowmax = rowmaxAlloc(rows);
    ASrc A; A.src = SRC_I8; A.hi = q.hi; A.lo = q.lo; A.ld8 = Li.Kp; A.e = q.e;
    AEpi E; E.epi = EPI_GEGLU; E.Y = v.d; E.ldy = F; E.Lphi = dm_.lutPhi32; E.rowmax = v.rowmax;
    k_gemm_hl2(st_, A, Mp, rows, Li.w, Li.Kp, Li.Np, Li.N, Li.Kp, Li.cst, E);
    return v;
}
QuantIn Engine::layernormQ(const Tensor& x, const Norm& nm) {
    if (nm.C != x.C) throw std::runtime_error("layernormQ: C mismatch");
    QuantIn q = qalloc(x.B, x.T, x.C);
    k_lnorm_q(st_, x.d, q.rows(), x.C, nm.g32, nm.beta32, nm.den, q.hi, q.lo, q.e, q.Kp);
    return q;
}
QuantIn Engine::groupnormQ(const Tensor& x, const Norm& nm) {
    if (nm.C != x.C) throw std::runtime_error("groupnormQ: C mismatch");
    QuantIn q = qalloc(x.B, x.T, x.C);
    size_t mk = arena_.mark();
    GnStat* gs = (GnStat*)arena_.alloc(sizeof(GnStat) * x.B * 32);
    k_gn_stats1(st_, x.d, x.B, x.T, x.C, 32, gs);
    k_gn_stats2(st_, x.d, x.B, x.T, x.C, 32, nm.den, gs);
    k_gn_apply_row(st_, x.d, x.B, x.T, x.C, 32, gs, nm.g32, nm.beta32, nullptr, nullptr, nullptr, q.hi, q.lo, q.e, q.Kp);
    arena_.release(mk);
    return q;
}
Tensor Engine::groupnormSilu(const Tensor& x, const Norm& nm, u32*& pm) {
    if (nm.C != x.C) throw std::runtime_error("groupnormSilu: C mismatch");
    Tensor y = talloc(x.B, x.T, x.C);
    pm = (u32*)arena_.alloc(size_t(x.B) * x.T * 4);
    size_t mk = arena_.mark();
    GnStat* gs = (GnStat*)arena_.alloc(sizeof(GnStat) * x.B * 32);
    k_gn_stats1(st_, x.d, x.B, x.T, x.C, 32, gs);
    k_gn_stats2(st_, x.d, x.B, x.T, x.C, 32, nm.den, gs);
    k_gn_apply_row(st_, x.d, x.B, x.T, x.C, 32, gs, nm.g32, nm.beta32, dm_.lutSigmoid32, y.d, pm, nullptr, nullptr, nullptr, 0);
    arena_.release(mk);
    return y;
}
Tensor Engine::conv3(const Tensor& x, int H, int W, const Lin& L, int stride, int32_t* out, const int32_t* residual, const u32* pmIn) {
    if (!L.conv3 || x.C * 9 != L.K) throw std::runtime_error("conv3: shape mismatch");
    int Ho = (H - 1) / stride + 1, Wo = (W - 1) / stride + 1; int rows = x.B * Ho * Wo, Mp = int(align_up(rows, 64));
    Tensor y; y.B = x.B; y.T = Ho * Wo; y.C = L.N; y.d = out ? out : (int32_t*)arena_.alloc(y.n() * 4);
    size_t mk = arena_.mark();
    u32* pm = const_cast<u32*>(pmIn);
    if (!pm) { pm = (u32*)arena_.alloc(size_t(x.B) * H * W * 4); k_pixmax(st_, x.d, x.B * H * W, x.C, pm); }
    int32_t* e = (int32_t*)arena_.alloc(size_t(Mp) * 4);
    if (cfg_.fusedLoader && useHl(L) && x.C % 16 == 0) {           // implicit im2col inside the GEMM (no int8 planes)
        k_conv_e(st_, pm, x.B, H, W, stride, e);
        ASrc A; A.src = SRC_CONV32; A.img = x.d; A.H = H; A.W = W; A.C = x.C; A.stride = stride; A.Ho = Ho; A.Wo = Wo; A.K = L.K; A.e = e;
        AEpi E; E.epi = EPI_PLAIN; E.Y = y.d; E.ldy = L.N; E.residual = residual;
        k_gemm_hl2(st_, A, Mp, rows, L.w, L.Kp, L.Np, L.N, L.Kp, L.cst, E);
        arena_.release(mk);
        return y;
    }
    int8_t* hi = (int8_t*)arena_.alloc(size_t(2) * Mp * L.Kp); int8_t* lo = hi + size_t(Mp) * L.Kp;
    k_im2col3(st_, x.d, pm, x.B, H, W, x.C, stride, hi, lo, e, L.Kp);
    if (useHl(L)) { k_gemm_hl(st_, hi, lo, L.Kp, Mp, rows, L.w, L.Kp, L.Np, L.N, L.Kp, L.cst, e, y.d, L.N, residual); arena_.release(mk); return y; }
    int32_t* Chi = (int32_t*)arena_.alloc(size_t(2) * Mp * L.Np * 4); int32_t* Clo = Chi + size_t(Mp) * L.Np;
    gemm_.run(st_, hi, L.Kp, 0, L.w, L.Kp, 0, Chi, L.Np, 0, 2 * Mp, L.Np, L.Kp);
    k_epilogue(st_, Chi, Clo, rows, L.N, L.Np, L.cst, e, y.d, L.N, residual);
    arena_.release(mk);
    return y;
}
Tensor Engine::groupnorm(const Tensor& x, const Norm& nm) {
    if (nm.C != x.C) throw std::runtime_error("groupnorm: C mismatch");
    Tensor y = talloc(x.B, x.T, x.C);
    size_t mk = arena_.mark();
    GnStat* gs = (GnStat*)arena_.alloc(sizeof(GnStat) * x.B * 32);
    k_gn_stats1(st_, x.d, x.B, x.T, x.C, 32, gs);
    k_gn_stats2(st_, x.d, x.B, x.T, x.C, 32, nm.den, gs);
    k_gn_apply(st_, x.d, x.B, x.T, x.C, 32, gs, nm.g, nm.beta, y.d);
    arena_.release(mk);
    return y;
}
Tensor Engine::layernorm(const Tensor& x, const Norm& nm) {
    if (nm.C != x.C) throw std::runtime_error("layernorm: C mismatch");
    Tensor y = talloc(x.B, x.T, x.C);
    k_lnorm(st_, x.d, x.B * x.T, x.C, nm.g, nm.beta, nm.den, y.d);
    return y;
}
void Engine::silu(Tensor& x) { k_act(st_, x.d, x.n(), dm_.lutSigmoid32); }

AttnKV Engine::quantKV(const Tensor& k, const Tensor& v, int heads) {
    AttnKV kv; kv.Tk = k.T; kv.Tkp = int(align_up(k.T, 64)); int G = k.B * heads;
    kv.khi = (int8_t*)arena_.alloc(size_t(G) * kv.Tkp * 64); kv.klo = (int8_t*)arena_.alloc(size_t(G) * kv.Tkp * 64);
    kv.ek = (int32_t*)arena_.alloc(size_t(G) * kv.Tkp * 4); kv.ksum = (i64*)arena_.alloc(size_t(G) * kv.Tkp * 8);
    i64* hs = (i64*)arena_.alloc(size_t(G) * kv.Tkp * 8); i64* ls = (i64*)arena_.alloc(size_t(G) * kv.Tkp * 8);
    kv.vthi = (int8_t*)arena_.alloc(size_t(G) * 64 * kv.Tkp); kv.vtlo = (int8_t*)arena_.alloc(size_t(G) * 64 * kv.Tkp);
    kv.ev = (int32_t*)arena_.alloc(size_t(G) * 64 * 4); kv.vsum = (i64*)arena_.alloc(size_t(G) * 64 * 8);
    k_quant_heads(st_, k.d, k.B, k.T, k.C, heads, kv.Tkp, kv.khi, kv.klo, kv.ek, hs, ls, kv.ksum);
    u32* scratch = (u32*)arena_.alloc(size_t(G) * 64 * 4);
    k_quant_v(st_, v.d, v.B, v.T, v.C, heads, kv.Tkp, kv.vthi, kv.vtlo, kv.ev, kv.vsum, scratch);
    return kv;
}
HeadQ Engine::headAlloc(int B, int heads, int T, int Tp) {
    HeadQ h; h.B = B; h.T = T; h.Tp = Tp; h.heads = heads; const size_t G = size_t(B) * heads;
    h.hi = (int8_t*)arena_.alloc(G * Tp * 64); h.lo = (int8_t*)arena_.alloc(G * Tp * 64);
    h.e = (int32_t*)arena_.alloc(G * Tp * 4); h.hs = (i64*)arena_.alloc(G * Tp * 8); h.ls = (i64*)arena_.alloc(G * Tp * 8); h.tsum = (i64*)arena_.alloc(G * Tp * 8);
    return h;
}
void Engine::linQ3H(const QuantIn& q, const Lin& L, int C, int heads, int Treal, HeadQ& qh, HeadQ& kh, Tensor& v) {
    if (q.K != L.K || L.N != 3 * C || C != heads * 64) throw std::runtime_error("linQ3H: shape mismatch");
    const int rows = q.rows(), Mp = q.Mp;
    const int TpQ = int(align_up(Treal, 128)), TpK = int(align_up(Treal, 64));
    qh = headAlloc(q.B, heads, Treal, TpQ); kh = headAlloc(q.B, heads, Treal, TpK);
    v = talloc(q.B, q.T, C);
    ASrc A; A.src = SRC_I8; A.hi = q.hi; A.lo = q.lo; A.ld8 = L.Kp; A.e = q.e;
    AEpi E; E.epi = EPI_QKV; E.Y2 = v.d; E.heads = heads; E.T = q.T; E.TpQ = TpQ; E.TpK = TpK; E.C = C;
    E.qhi = qh.hi; E.qlo = qh.lo; E.qe = qh.e; E.qhs = qh.hs; E.qls = qh.ls; E.qts = qh.tsum; E.khi = kh.hi; E.klo = kh.lo; E.ke = kh.e; E.khs = kh.hs; E.kls = kh.ls; E.kts = kh.tsum;
    k_gemm_hl2(st_, A, Mp, rows, L.w, L.Kp, L.Np, L.N, L.Kp, L.cst, E);
    const int G = q.B * heads;
    k_quant_heads_pad(st_, G, q.T, TpQ, qh.hi, qh.lo, qh.e, qh.hs, qh.ls, qh.tsum);     // rows the GEMM did not write
    k_quant_heads_pad(st_, G, q.T, TpK, kh.hi, kh.lo, kh.e, kh.hs, kh.ls, kh.tsum);
}
AttnKV Engine::quantKVH(const HeadQ& k, const Tensor& v, int heads) {
    AttnKV kv; kv.Tk = k.T; kv.Tkp = k.Tp; const int G = v.B * heads;
    kv.khi = k.hi; kv.klo = k.lo; kv.ek = k.e; kv.ksum = k.tsum;
    kv.vthi = (int8_t*)arena_.alloc(size_t(G) * 64 * kv.Tkp); kv.vtlo = (int8_t*)arena_.alloc(size_t(G) * 64 * kv.Tkp);
    kv.ev = (int32_t*)arena_.alloc(size_t(G) * 64 * 4); kv.vsum = (i64*)arena_.alloc(size_t(G) * 64 * 8);
    u32* scratch = (u32*)arena_.alloc(size_t(G) * 64 * 4);
    k_quant_v(st_, v.d, v.B, k.T, v.C, heads, kv.Tkp, kv.vthi, kv.vtlo, kv.ev, kv.vsum, scratch);
    return kv;
}
Tensor Engine::attentionH(const HeadQ& q, const AttnKV& kv, int heads, bool causal, int Tq, int C) {
    const int B = q.B, G = B * heads, Tqp = q.Tp;
    Tensor out = talloc(B, Tq, C);
    if (cfg_.fusedLoader) out.rowmax = rowmaxAlloc(B * Tq);
    size_t mk = arena_.mark();
    int32_t* gmx = (int32_t*)arena_.alloc(size_t(G) * Tqp * 4); i64* gr = (i64*)arena_.alloc(size_t(G) * Tqp * 8);
    k_fused_attention(st_, q.hi, q.lo, q.e, q.hs, q.ls, kv.khi, kv.klo, kv.ek, kv.ksum, kv.vthi, kv.vtlo, kv.ev, kv.vsum, B, heads, Tq, Tqp, kv.Tk, kv.Tkp, causal, dm_.lutExp, C, gmx, gr, out.d, out.rowmax);
    arena_.release(mk);
    return out;
}
Tensor Engine::attention(const Tensor& q, const AttnKV& kv, int heads, bool causal, int Tq) {
    int B = q.B, C = q.C, G = B * heads, Tqp = int(align_up(q.T, 32));
    if (C != heads * 64) throw std::runtime_error("attention: C != heads*64");
    if (cfg_.fusedAttention) Tqp = int(align_up(q.T, 128));   // works for 64- and 128-query attention blocks
    Tensor out = talloc(B, q.T, C);
    if (cfg_.fusedAttention && cfg_.fusedLoader) out.rowmax = rowmaxAlloc(B * q.T);
    size_t mk = arena_.mark();
    int8_t* qhi = (int8_t*)arena_.alloc(size_t(G) * Tqp * 64); int8_t* qlo = (int8_t*)arena_.alloc(size_t(G) * Tqp * 64);
    int32_t* eq = (int32_t*)arena_.alloc(size_t(G) * Tqp * 4); i64* qhs = (i64*)arena_.alloc(size_t(G) * Tqp * 8); i64* qls = (i64*)arena_.alloc(size_t(G) * Tqp * 8); i64* qts = (i64*)arena_.alloc(size_t(G) * Tqp * 8);
    k_quant_heads(st_, q.d, B, Tq, C, heads, Tqp, qhi, qlo, eq, qhs, qls, qts);
    if (cfg_.fusedAttention) {
        int32_t* gmx = (int32_t*)arena_.alloc(size_t(G) * Tqp * 4); i64* gr = (i64*)arena_.alloc(size_t(G) * Tqp * 8);
        k_fused_attention(st_, qhi, qlo, eq, qhs, qls, kv.khi, kv.klo, kv.ek, kv.ksum, kv.vthi, kv.vtlo, kv.ev, kv.vsum, B, heads, Tq, Tqp, kv.Tk, kv.Tkp, causal, dm_.lutExp, C, gmx, gr, out.d, out.rowmax);
        arena_.release(mk);
        return out;
    }
    // query chunk: keep the 4 int32 score partials inside the budget
    size_t perRow = size_t(G) * kv.Tkp * 16;
    int Qc = int(std::min<size_t>(Tqp, std::max<size_t>(32, cfg_.scoreBytes / perRow) / 32 * 32));
    int32_t* S[4]; for (auto& p : S) p = (int32_t*)arena_.alloc(size_t(G) * Qc * kv.Tkp * 4);
    int8_t* phi = (int8_t*)arena_.alloc(size_t(G) * Qc * kv.Tkp); int8_t* plo = (int8_t*)arena_.alloc(size_t(G) * Qc * kv.Tkp);
    i64* phs = (i64*)arena_.alloc(size_t(G) * Qc * 8); i64* pls = (i64*)arena_.alloc(size_t(G) * Qc * 8);
    int32_t* O[4]; for (auto& p : O) p = (int32_t*)arena_.alloc(size_t(G) * Qc * 64 * 4);
    for (int q0 = 0; q0 < Tq; q0 += Qc) {
        long long sQ = (long long)Tqp * 64, sK = (long long)kv.Tkp * 64, sS = (long long)Qc * kv.Tkp;
        const int8_t* qa[2] = {qhi + size_t(q0) * 64, qlo + size_t(q0) * 64}; const int8_t* kb[2] = {kv.khi, kv.klo};
        // S[0]=hh S[1]=hl S[2]=lh S[3]=ll  (first index = query operand)
        for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j)
            gemm_.run(st_, qa[i], 64, sQ, kb[j], 64, sK, S[i * 2 + j], kv.Tkp, sS, Qc, kv.Tkp, 64, G);
        k_softmax(st_, S[0], S[1], S[2], S[3], qhs, qls, eq, kv.ksum, kv.ek, G, Tqp, q0, Qc, kv.Tk, kv.Tkp, causal, dm_.lutExp, phi, plo, phs, pls);
        const int8_t* pa[2] = {phi, plo}; const int8_t* vb[2] = {kv.vthi, kv.vtlo};
        for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j)
            gemm_.run(st_, pa[i], kv.Tkp, sS, vb[j], kv.Tkp, (long long)64 * kv.Tkp, O[i * 2 + j], 64, (long long)Qc * 64, Qc, 64, kv.Tkp, G);
        k_attn_out(st_, O[0], O[1], O[2], O[3], phs, pls, kv.vsum, kv.ev, B, heads, Tq, q0, Qc, C, out.d);
    }
    arena_.release(mk);
    return out;
}
Tensor Engine::resnet(const Tensor& x, int H, int W, const std::string& p) {
    const Lin& c1 = dm_.lin(p + ".conv1"); const Lin& c2 = dm_.lin(p + ".conv2");
    Tensor out = talloc(x.B, x.T, c2.N);
    size_t mk = arena_.mark();
    u32* pm1 = nullptr; u32* pm2 = nullptr;
    Tensor h = groupnormSilu(x, dm_.norm(p + ".norm1"), pm1);
    h = conv3(h, H, W, c1, 1, nullptr, nullptr, pm1);
    Tensor h2 = groupnormSilu(h, dm_.norm(p + ".norm2"), pm2);
    if (dm_.hasLin(p + ".conv_shortcut")) { Tensor xs = lin(x, dm_.lin(p + ".conv_shortcut")); conv3(h2, H, W, c2, 1, out.d, xs.d, pm2); }
    else conv3(h2, H, W, c2, 1, out.d, x.d, pm2);
    arena_.release(mk);
    return out;
}
Tensor Engine::transformer(const Tensor& x, const std::string& p, int heads) {
    Tensor out = talloc(x.B, x.T, x.C);
    size_t mk = arena_.mark();
    std::string b = p + ".transformer_blocks.0";
    Tensor h = talloc(x.B, x.T, dm_.lin(p + ".proj_in").N);
    {
        size_t m2 = arena_.mark();
        QuantIn xq = groupnormQ(x, dm_.norm(p + ".norm"));
        linQ(xq, dm_.lin(p + ".proj_in"), h.d);
        arena_.release(m2);
    }
    {
        size_t m2 = arena_.mark();
        QuantIn n = layernormQ(h, dm_.norm(b + ".norm1"));
        Tensor a;
        if (cfg_.fusedAttention && useHl(dm_.lin(b + ".attn1.to_qkv"))) {
            HeadQ qh, kh; Tensor v; linQ3H(n, dm_.lin(b + ".attn1.to_qkv"), h.C, heads, h.T, qh, kh, v);
            AttnKV kv = quantKVH(kh, v, heads);
            a = attentionH(qh, kv, heads, false, h.T, h.C);
        } else {
            Tensor q, k, v; linQ3(n, dm_.lin(b + ".attn1.to_qkv"), h.C, q, k, v);
            AttnKV kv = quantKV(k, v, heads);
            a = attention(q, kv, heads, false, q.T);
        }
        linX(a, dm_.lin(b + ".attn1.to_out.0"), h.d, h.d);
        arena_.release(m2);
    }
    {
        size_t m2 = arena_.mark();
        QuantIn n = layernormQ(h, dm_.norm(b + ".norm2"));
        Tensor q = linQ(n, dm_.lin(b + ".attn2.to_q"));
        auto it = crossKV_.find(b); if (it == crossKV_.end()) throw std::runtime_error("no cross KV for " + b);
        Tensor a = attention(q, it->second, heads, false, q.T);
        linX(a, dm_.lin(b + ".attn2.to_out.0"), h.d, h.d);
        arena_.release(m2);
    }
    {
        size_t m2 = arena_.mark();
        QuantIn n = layernormQ(h, dm_.norm(b + ".norm3"));
        const int F = dm_.lin(b + ".ff.net.0.proj.a").N; const Lin& Li = dm_.lin(b + ".ff.net.0.proj.agi");
        if (cfg_.fusedLoader && useHl(Li)) { Tensor v = gegluX(n, Li, F); Tensor hh = linX(v, dm_.lin(b + ".ff.net.2"), h.d, h.d, EPI_PLAIN, true); h.rowmax = hh.rowmax; }
        else { QuantIn aq = linQGegluQ(n, dm_.lin(b + ".ff.net.0.proj.ag"), F, &Li); linQ(aq, dm_.lin(b + ".ff.net.2"), h.d, h.d); }
        arena_.release(m2);
    }
    linX(h, dm_.lin(p + ".proj_out"), out.d, x.d);
    arena_.release(mk);
    return out;
}
void Engine::unet(const Tensor& x, Tensor& eps) {
    static const int HEADS[4] = {5, 10, 20, 20};
    int H = 64, W = 64;
    Tensor h = conv3(x, H, W, dm_.lin("unet.conv_in"), 1);
    std::vector<Tensor> skips; skips.push_back(h);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            h = resnet(h, H, W, "unet.down_blocks." + std::to_string(i) + ".resnets." + std::to_string(j));
            if (i < 3) h = transformer(h, "unet.down_blocks." + std::to_string(i) + ".attentions." + std::to_string(j), HEADS[i]);
            skips.push_back(h);
        }
        if (i < 3) { h = conv3(h, H, W, dm_.lin("unet.down_blocks." + std::to_string(i) + ".downsamplers.0.conv"), 2); H /= 2; W /= 2; skips.push_back(h); }
    }
    h = resnet(h, H, W, "unet.mid_block.resnets.0");
    h = transformer(h, "unet.mid_block.attentions.0", 20);
    h = resnet(h, H, W, "unet.mid_block.resnets.1");
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            Tensor s = skips.back(); skips.pop_back();
            Tensor c = talloc(h.B, h.T, h.C + s.C);
            k_concat(st_, h.d, h.C, s.d, s.C, h.B * h.T, c.d);
            h = resnet(c, H, W, "unet.up_blocks." + std::to_string(i) + ".resnets." + std::to_string(j));
            if (i > 0) h = transformer(h, "unet.up_blocks." + std::to_string(i) + ".attentions." + std::to_string(j), HEADS[3 - i]);
        }
        if (i < 3) {
            Tensor u = talloc(h.B, h.T * 4, h.C);
            k_upsample2(st_, h.d, h.B, H, W, h.C, u.d); H *= 2; W *= 2;
            h = conv3(u, H, W, dm_.lin("unet.up_blocks." + std::to_string(i) + ".upsamplers.0.conv"), 1);
        }
    }
    u32* pmo = nullptr;
    Tensor n = groupnormSilu(h, dm_.norm("unet.conv_norm_out"), pmo);
    conv3(n, H, W, dm_.lin("unet.conv_out"), 1, eps.d, nullptr, pmo);
}
Tensor Engine::textEncoder(const std::vector<int>& ids) {
    const int T = 77, Tp = 96, D = 1024;
    if (ids.size() != size_t(T)) throw std::runtime_error("text: need 77 ids");
    std::vector<int32_t> h(ids.begin(), ids.end());
    CUDA_CHECK(cudaMemcpyAsync(dIds_, h.data(), T * 4, cudaMemcpyHostToDevice, st_));
    Tensor x = talloc(1, Tp, D);
    k_text_embed(st_, dIds_, T, Tp, dm_.tokW, dm_.tokM, dm_.tokS, dm_.tePos, D, x.d);
    for (int l = 0; l < 23; ++l) {
        std::string p = "te.layers." + std::to_string(l);
        size_t mk = arena_.mark();
        QuantIn n = layernormQ(x, dm_.norm(p + ".layer_norm1"));
        Tensor a;
        if (cfg_.fusedAttention && useHl(dm_.lin(p + ".self_attn.qkv"))) {
            HeadQ qh, kh; Tensor v; linQ3H(n, dm_.lin(p + ".self_attn.qkv"), D, 16, T, qh, kh, v);   // Treal = 77 (rows 77..95 of x are padding)
            AttnKV kv = quantKVH(kh, v, 16);
            a = attentionH(qh, kv, 16, true, T, D);
        } else {
            Tensor q, k, v; linQ3(n, dm_.lin(p + ".self_attn.qkv"), D, q, k, v);
            k.T = T; v.T = T;                    // only the 77 real rows are keys/values
            AttnKV kv = quantKV(k, v, 16);
            a = attention(q, kv, 16, true, T);
        }
        linX(a, dm_.lin(p + ".self_attn.out_proj"), x.d, x.d);
        arena_.release(mk); mk = arena_.mark();
        QuantIn n2 = layernormQ(x, dm_.norm(p + ".layer_norm2"));
        Tensor f;
        if (cfg_.fusedLoader && useHl(dm_.lin(p + ".mlp.fc1"))) {
            const Lin& L1 = dm_.lin(p + ".mlp.fc1"); f.B = 1; f.T = Tp; f.C = L1.N; f.d = (int32_t*)arena_.alloc(f.n() * 4); f.rowmax = rowmaxAlloc(Tp);
            ASrc A; A.src = SRC_I8; A.hi = n2.hi; A.lo = n2.lo; A.ld8 = L1.Kp; A.e = n2.e;
            AEpi E; E.epi = EPI_GELU; E.Y = f.d; E.ldy = L1.N; E.Lphi = dm_.lutPhi32; E.rowmax = f.rowmax;
            k_gemm_hl2(st_, A, n2.Mp, Tp, L1.w, L1.Kp, L1.Np, L1.N, L1.Kp, L1.cst, E);
            linX(f, dm_.lin(p + ".mlp.fc2"), x.d, x.d);
        } else { f = linQ(n2, dm_.lin(p + ".mlp.fc1")); k_act(st_, f.d, f.n(), dm_.lutPhi32); lin(f, dm_.lin(p + ".mlp.fc2"), x.d, x.d); }
        arena_.release(mk);
    }
    return layernorm(x, dm_.norm("te.final_layer_norm"));
}
void Engine::setPrompt(const std::vector<int>& tokens) {
    if (havePrompt_ && tokens == promptTokens_) return;
    CUDA_CHECK(cudaSetDevice(dm_.device));
    std::vector<int> ids; ids.push_back(49406); for (int t : tokens) ids.push_back(t); ids.push_back(49407);
    if (ids.size() > 77) throw std::runtime_error("prompt too long");
    while (ids.size() < 77) ids.push_back(0);
    arena_.reset();
    Tensor ctx = textEncoder(ids);           // [1, 96, 1024]
    ctx.T = 77;
    // cross-attention KV per transformer block, replicated for the batch
    static const int HEADS[4] = {5, 10, 20, 20};
    std::vector<std::pair<std::string, int>> blocks;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 2; ++j) blocks.push_back({"unet.down_blocks." + std::to_string(i) + ".attentions." + std::to_string(j), HEADS[i]});
    blocks.push_back({"unet.mid_block.attentions.0", 20});
    for (int i = 1; i < 4; ++i) for (int j = 0; j < 3; ++j) blocks.push_back({"unet.up_blocks." + std::to_string(i) + ".attentions." + std::to_string(j), HEADS[3 - i]});
    int B = cfg_.batch;
    for (auto& bl : blocks) {
        std::string b = bl.first + ".transformer_blocks.0"; int heads = bl.second;
        size_t mk = arena_.mark();
        Tensor k = lin(ctx, dm_.lin(b + ".attn2.to_k")), v = lin(ctx, dm_.lin(b + ".attn2.to_v"));
        AttnKV one = quantKV(k, v, heads);          // G = heads
        AttnKV& kv = crossKV_[b];
        int G1 = heads, Tkp = one.Tkp;
        if (!kv.khi) {
            kv.Tk = one.Tk; kv.Tkp = Tkp;
            kv.khi = (int8_t*)palloc(size_t(B) * G1 * Tkp * 64); kv.klo = (int8_t*)palloc(size_t(B) * G1 * Tkp * 64);
            kv.ek = (int32_t*)palloc(size_t(B) * G1 * Tkp * 4); kv.ksum = (i64*)palloc(size_t(B) * G1 * Tkp * 8);
            kv.vthi = (int8_t*)palloc(size_t(B) * G1 * 64 * Tkp); kv.vtlo = (int8_t*)palloc(size_t(B) * G1 * 64 * Tkp);
            kv.ev = (int32_t*)palloc(size_t(B) * G1 * 64 * 4); kv.vsum = (i64*)palloc(size_t(B) * G1 * 64 * 8);
        }
        for (int bb = 0; bb < B; ++bb) {
            size_t o = size_t(bb) * G1;
            CUDA_CHECK(cudaMemcpyAsync(kv.khi + o * Tkp * 64, one.khi, size_t(G1) * Tkp * 64, cudaMemcpyDeviceToDevice, st_));
            CUDA_CHECK(cudaMemcpyAsync(kv.klo + o * Tkp * 64, one.klo, size_t(G1) * Tkp * 64, cudaMemcpyDeviceToDevice, st_));
            CUDA_CHECK(cudaMemcpyAsync(kv.ek + o * Tkp, one.ek, size_t(G1) * Tkp * 4, cudaMemcpyDeviceToDevice, st_));
            CUDA_CHECK(cudaMemcpyAsync(kv.ksum + o * Tkp, one.ksum, size_t(G1) * Tkp * 8, cudaMemcpyDeviceToDevice, st_));
            CUDA_CHECK(cudaMemcpyAsync(kv.vthi + o * 64 * Tkp, one.vthi, size_t(G1) * 64 * Tkp, cudaMemcpyDeviceToDevice, st_));
            CUDA_CHECK(cudaMemcpyAsync(kv.vtlo + o * 64 * Tkp, one.vtlo, size_t(G1) * 64 * Tkp, cudaMemcpyDeviceToDevice, st_));
            CUDA_CHECK(cudaMemcpyAsync(kv.ev + o * 64, one.ev, size_t(G1) * 64 * 4, cudaMemcpyDeviceToDevice, st_));
            CUDA_CHECK(cudaMemcpyAsync(kv.vsum + o * 64, one.vsum, size_t(G1) * 64 * 8, cudaMemcpyDeviceToDevice, st_));
        }
        arena_.release(mk);
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
    CUDA_CHECK(cudaGetLastError());
    promptTokens_ = tokens; havePrompt_ = true;
}
long Engine::testAttention(int B, int Tq, int Tk, int heads, bool causal, int scaleBits) {
    CUDA_CHECK(cudaSetDevice(dm_.device));
    int C = heads * 64;
    auto randTensor = [&](int T, uint32_t seed) {
        Tensor t = talloc(B, T, C); std::vector<int32_t> h(t.n()); uint64_t x = 0x9E3779B97F4A7C15ull ^ seed;
        for (auto& v : h) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; int bits = 1 + int((x >> 40) % scaleBits); v = int32_t(int64_t(x & ((1ull << bits) - 1)) - (1ll << (bits - 1))); }
        CUDA_CHECK(cudaMemcpyAsync(t.d, h.data(), t.n() * 4, cudaMemcpyHostToDevice, st_)); return t;
    };
    arena_.reset();
    Tensor q = randTensor(Tq, 1), k = randTensor(Tk, 2), v = randTensor(Tk, 3);
    AttnKV kv = quantKV(k, v, heads);
    bool saved = cfg_.fusedAttention;
    cfg_.fusedAttention = false; Tensor ref = attention(q, kv, heads, causal, Tq);
    cfg_.fusedAttention = true; Tensor got = attention(q, kv, heads, causal, Tq);
    cfg_.fusedAttention = saved;
    std::vector<int32_t> hr(ref.n()), hg(got.n());
    CUDA_CHECK(cudaMemcpyAsync(hr.data(), ref.d, ref.n() * 4, cudaMemcpyDeviceToHost, st_)); CUDA_CHECK(cudaMemcpyAsync(hg.data(), got.d, got.n() * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_)); CUDA_CHECK(cudaGetLastError());
    long bad = 0; long firstBad = -1;
    for (size_t i = 0; i < hr.size(); ++i) if (hr[i] != hg[i]) { if (firstBad < 0) firstBad = long(i); ++bad; }
    if (bad) { size_t i = size_t(firstBad); std::printf("  first mismatch at [b=%zu t=%zu c=%zu]: ref %d got %d\n", i / (size_t(Tq) * C), (i / C) % Tq, i % C, hr[i], hg[i]); }
    return bad;
}
void Engine::forward() {
    int B = cfg_.batch;
    CUDA_CHECK(cudaMemcpyAsync(dSeeds_, hSeeds_, 32 * B, cudaMemcpyHostToDevice, st_));
    arena_.reset();
    k_noise(st_, dSeeds_, B, dm_.lutGauss, dm_.cCin, dNoise_, dX_);
    Tensor x; x.d = dX_; x.B = B; x.T = 4096; x.C = 4;
    Tensor eps; eps.d = dEps_; eps.B = B; eps.T = 4096; eps.C = 4;
    unet(x, eps);
    k_finalize(st_, dNoise_, dEps_, B, dm_.cSig, dOut_);
    CUDA_CHECK(cudaMemcpyAsync(hOut_, dOut_, 32768 * B, cudaMemcpyDeviceToHost, st_));
}
void Engine::run(const uint8_t* seeds, uint8_t* x0) {
    if (!havePrompt_) throw std::runtime_error("run: no prompt set");
    CUDA_CHECK(cudaSetDevice(dm_.device));
    int B = cfg_.batch;
    std::memcpy(hSeeds_, seeds, 32 * B);
    CUDA_CHECK(cudaEventRecord(ev0_, st_));
    if (!cfg_.graph) {
        forward();
    } else if (!captured_) {
        forward();                                   // eager warm-up: builds GEMM plans, measures arena
        CUDA_CHECK(cudaStreamSynchronize(st_)); CUDA_CHECK(cudaGetLastError());
        cudaGraph_t g;
        CUDA_CHECK(cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal));
        forward();
        CUDA_CHECK(cudaStreamEndCapture(st_, &g));
        CUDA_CHECK(cudaGraphInstantiate(&gexec_, g, 0));
        cudaGraphDestroy(g);
        captured_ = true;
        CUDA_CHECK(cudaGraphLaunch(gexec_, st_));
    } else {
        CUDA_CHECK(cudaGraphLaunch(gexec_, st_));
    }
    CUDA_CHECK(cudaEventRecord(ev1_, st_));
    CUDA_CHECK(cudaEventSynchronize(ev1_));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0; cudaEventElapsedTime(&ms, ev0_, ev1_); lastMs = ms;
    std::memcpy(x0, hOut_, 32768 * B);
}
}  // namespace meow
