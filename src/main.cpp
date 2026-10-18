// supr-meow-vela — Vela (PoI v2 / sd-turbo-int) GPU miner and verify-server.
#include "common.h"
#include "poisd.h"
#include "sha256.h"
#include "engine.h"
#include "stratum.h"
#include "httpd.h"
#include "device.h"
#include <cuda_profiler_api.h>
#include <nlohmann/json.hpp>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <exception>
#include <mutex>
#include <random>
#include <thread>
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace meow;
using json = nlohmann::json;
static const char* VERSION_STR = "0.1.1";
static std::atomic<bool> g_stop{false};
static std::atomic<bool> g_minerFailed{false};
static std::mutex g_logmu;
static void logln(const std::string& s) {
    std::lock_guard<std::mutex> l(g_logmu);
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); char buf[32]; std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&t));
    std::cout << "[" << buf << "] " << s << std::endl;
}
struct Opts {
    std::string model = "sd-turbo-int.poisd", url, wallet, password = "x", host = "0.0.0.0";
    std::vector<int> devices; int batch = 4; int lanes = 2; bool tls = false; bool graph = true;
    bool fusedAttn = true; bool testAttn = false; int gemmMaxK = 1536; bool fusedLoader = true; bool gnFuse = false; bool check = false, bench = false, verifyServer = false; int benchIters = 20; int port = 9010; size_t memMB = 0; size_t scoreMB = 256;
    std::string seedHex, headerHex, tokensStr, x0Out; double suggestDiff = 0; int threads = 4;
};
static void usage() {
    std::cout << "supr-meow-vela " << VERSION_STR << " — Vela PoI v2 GPU miner\n"
        "  -o, --url HOST:PORT      stratum endpoint (e.g. vela.suprnova.cc:7601)\n"
        "  -u, --user WALLET[.rig]  vela1… address with optional worker name\n"
        "  -p, --pass X             password (ignored by the pool)\n"
        "      --tls                TLS (port 7604)\n"
        "      --model FILE         sd-turbo-int.poisd (default ./sd-turbo-int.poisd)\n"
        "      --devices 0,1        CUDA devices (default all)\n"
        "      --batch N            inferences per lane step (default 4)\n"
        "      --lanes N            lanes (streams) per GPU (default 2)\n"
        "      --mem MB             arena per lane (default: auto)\n"
        "      --score-mb MB        attention score budget per lane (default 256)\n"
        "      --suggest-diff D     send mining.suggest_difficulty after connect\n"
        "      --no-graph           run eagerly (debug)\n"
        "      --check              verify the spec test vector on every selected GPU\n"
        "      --bench [N]          throughput benchmark\n"
        "      --seed HEX | --header HEX80  --tokens 320,736,3240 [--x0-out FILE]   single inference, print sha256(x0)/hashpoi\n"
        "      --verify-server --port 9010 --host 0.0.0.0   pool verifier (HTTP /verify, /health)\n";
}
static std::vector<int> parseInts(const std::string& s) { std::vector<int> v; std::stringstream ss(s); std::string t; while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::stoi(t)); return v; }
static Opts parse(int argc, char** argv) {
    Opts o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i]; auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if (a == "-o" || a == "--url") o.url = next(); else if (a == "-u" || a == "--user") o.wallet = next(); else if (a == "-p" || a == "--pass") o.password = next();
        else if (a == "--tls") o.tls = true; else if (a == "--model") o.model = next(); else if (a == "--devices") o.devices = parseInts(next());
        else if (a == "--batch") o.batch = std::stoi(next()); else if (a == "--lanes") o.lanes = std::stoi(next()); else if (a == "--mem") o.memMB = std::stoul(next());
        else if (a == "--score-mb") o.scoreMB = std::stoul(next()); else if (a == "--suggest-diff") o.suggestDiff = std::stod(next()); else if (a == "--no-graph") o.graph = false; else if (a == "--attn-cublas") o.fusedAttn = false; else if (a == "--test-attn") o.testAttn = true; else if (a == "--gemm-maxk") o.gemmMaxK = std::stoi(next()); else if (a == "--no-fused-loader") o.fusedLoader = false; else if (a == "--gn-fuse") o.gnFuse = true;
        else if (a == "--check") o.check = true; else if (a == "--bench") { o.bench = true; if (i + 1 < argc && argv[i + 1][0] != '-') o.benchIters = std::stoi(argv[++i]); }
        else if (a == "--seed") o.seedHex = next(); else if (a == "--header") o.headerHex = next(); else if (a == "--x0-out") o.x0Out = next(); else if (a == "--tokens") o.tokensStr = next();
        else if (a == "--verify-server") o.verifyServer = true; else if (a == "--port") o.port = std::stoi(next()); else if (a == "--host") o.host = next(); else if (a == "--threads") o.threads = std::stoi(next());
        else if (a == "-h" || a == "--help") { usage(); std::exit(0); }
        else throw std::runtime_error("unknown option " + a);
    }
    return o;
}
static std::array<uint8_t, 32> poiSeed(const uint8_t* header80, const std::array<uint8_t, 32>& modelHashU256) {
    TaggedHasher h("POI/seed"); h.update(header80, 80); h.update(modelHashU256.data(), 32); return h.digest();
}
static std::array<uint8_t, 32> poiOut(const std::array<uint8_t, 32>& seed, const uint8_t* x0) { TaggedHasher h("POI/out"); h.update(seed.data(), 32); h.update(x0, 32768); return h.digest(); }

// ------------------------------------------------------------------ single inference / check
static int runSingle(const Opts& o, const Poisd& model, const std::vector<int>& devs, std::array<uint8_t, 32> seed, const std::vector<int>& tokens, const char* expectHex) {
    DevModel dm; dm.load(devs[0], model);
    EngineConfig cfg; cfg.batch = o.batch; cfg.graph = o.graph; cfg.fusedAttention = o.fusedAttn; cfg.customGemmMaxK = o.gemmMaxK; cfg.fusedLoader = o.fusedLoader; cfg.gnFuse = o.gnFuse; cfg.scoreBytes = o.scoreMB << 20; if (o.memMB) cfg.arenaBytes = o.memMB << 20;
    Engine eng(dm, cfg);
    auto t0 = std::chrono::steady_clock::now();
    eng.setPrompt(tokens);
    double promptMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::vector<uint8_t> seeds(size_t(cfg.batch) * 32), x0(size_t(cfg.batch) * 32768);
    for (int b = 0; b < cfg.batch; ++b) std::memcpy(&seeds[b * 32], seed.data(), 32);
    eng.run(seeds.data(), x0.data());
    eng.run(seeds.data(), x0.data());
    auto d = sha256(x0.data(), 32768);
    std::string got = hex(d.data(), 32);
    auto hp = poiOut(seed, x0.data());
    bool same = true; for (int b = 1; b < cfg.batch; ++b) if (std::memcmp(x0.data(), &x0[b * 32768], 32768) != 0) same = false;
    logln("prompt (text encoder + cross KV): " + std::to_string(int(promptMs)) + " ms; batch " + std::to_string(cfg.batch) + " forward " + std::to_string(eng.lastMs) + " ms (" + std::to_string(1000.0 * cfg.batch / eng.lastMs) + " inf/s); arena peak " + std::to_string(eng.arenaPeak() >> 20) + " MiB");
    logln("seed " + hex(seed.data(), 32));
    logln("sha256(x0) = " + got);
    logln("hashpoi    = " + hex(hp.data(), 32));
    if (!o.x0Out.empty()) { FILE* f = std::fopen(o.x0Out.c_str(), "w"); if (f) { std::string hx = hex(x0.data(), 32768); std::fwrite(hx.data(), 1, hx.size(), f); std::fclose(f); logln("x0 hex written to " + o.x0Out); } }
    if (!same) { logln("FAIL: batch lanes disagree"); return 2; }
    if (expectHex) { if (got == expectHex) { logln("CHECK OK — matches the sdint.py test vector"); return 0; } logln(std::string("CHECK FAILED — expected ") + expectHex); return 1; }
    return 0;
}

// ------------------------------------------------------------------ benchmark
#ifdef MEOW_ATTN_STATS
namespace meow { void k_attn_stats_dump(); }
#endif
static int runBench(const Opts& o, const Poisd& model, const std::vector<int>& devs) {
    std::vector<int> tokens = {320, 736, 3240};
    std::atomic<uint64_t> total{0};
    std::atomic<bool> failed{false};
    std::vector<std::thread> ths; std::mutex mu;
    auto t0 = std::chrono::steady_clock::now();
    for (int dev : devs) ths.emplace_back([&, dev] {
        try {
            DevModel dm; dm.load(dev, model);
            size_t freeB = 0, totB = 0; cudaMemGetInfo(&freeB, &totB);
            EngineConfig cfg; cfg.batch = o.batch; cfg.graph = o.graph; cfg.fusedAttention = o.fusedAttn; cfg.customGemmMaxK = o.gemmMaxK; cfg.fusedLoader = o.fusedLoader; cfg.gnFuse = o.gnFuse; cfg.scoreBytes = o.scoreMB << 20;
            cfg.arenaBytes = o.memMB ? (o.memMB << 20) : (freeB - (768u << 20)) / o.lanes;
            std::vector<std::unique_ptr<Engine>> lanes; for (int l = 0; l < o.lanes; ++l) { lanes.emplace_back(new Engine(dm, cfg)); lanes.back()->setPrompt(tokens); }
            std::vector<std::thread> lt; std::atomic<uint64_t> devTotal{0};
            std::exception_ptr laneError; std::mutex laneErrorMu;
            // warm-up (graph capture + cuBLASLt autotune) is excluded from the timed window
            { std::vector<uint8_t> seeds(size_t(o.batch) * 32, 7), x0(size_t(o.batch) * 32768); for (auto& e : lanes) { e->submit(seeds.data()); e->submit(seeds.data()); e->collect(x0.data()); e->collect(x0.data()); } }   // both pipeline slots captured
            cudaProfilerStart();                       // nsys --capture-range=cudaProfilerApi: profile only the steady state
            auto td = std::chrono::steady_clock::now();
            for (int l = 0; l < o.lanes; ++l) lt.emplace_back([&, l] {
                try {
                    std::mt19937_64 rng(dev * 100 + l); std::vector<uint8_t> seeds(size_t(o.batch) * 32), x0(size_t(o.batch) * 32768);
                    // two batches in flight per lane: the GPU never waits for the host-side hashing / copies
                    for (auto& b : seeds) b = uint8_t(rng());
                    lanes[l]->submit(seeds.data());
                    for (int it = 0; it < o.benchIters; ++it) {
                        if (it + 1 < o.benchIters) { for (auto& b : seeds) b = uint8_t(rng()); lanes[l]->submit(seeds.data()); }
                        lanes[l]->collect(x0.data());
                        for (int b = 0; b < o.batch; ++b) { auto s = std::array<uint8_t, 32>{}; std::memcpy(s.data(), &seeds[b * 32], 32); poiOut(s, &x0[b * 32768]); }
                        devTotal += o.batch; total += o.batch;
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> lock(laneErrorMu);
                    if (!laneError) laneError = std::current_exception();
                }
            });
            for (auto& t : lt) t.join();
            cudaProfilerStop();
            if (laneError) std::rethrow_exception(laneError);
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - td).count();
            std::lock_guard<std::mutex> l(mu);
            logln("GPU " + std::to_string(dev) + ": " + std::to_string(devTotal.load()) + " inferences in " + std::to_string(s) + " s = " + std::to_string(devTotal / s) + " inf/s (batch " + std::to_string(o.batch) + " x " + std::to_string(o.lanes) + " lanes, last step " + std::to_string(lanes[0]->lastMs) + " ms, arena peak " + std::to_string(lanes[0]->arenaPeak() >> 20) + " MiB)");
        } catch (const std::exception& e) { failed = true; logln("GPU " + std::to_string(dev) + " bench error: " + e.what()); }
    });
    for (auto& t : ths) t.join();
#ifdef MEOW_ATTN_STATS
    meow::k_attn_stats_dump();
#endif
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    logln("TOTAL " + std::to_string(total.load()) + " inferences, " + std::to_string(total / s) + " inf/s incl. setup");
    return failed ? 1 : 0;
}

// ------------------------------------------------------------------ verify server
static int runVerifyServer(const Opts& o, const Poisd& model, const std::vector<int>& devs) {
    struct Slot { std::unique_ptr<DevModel> dm; std::unique_ptr<Engine> eng; std::mutex mu; };
    std::vector<std::unique_ptr<Slot>> slots;
    for (int dev : devs) {
        auto s = std::make_unique<Slot>(); s->dm = std::make_unique<DevModel>(); s->dm->load(dev, model);
        EngineConfig cfg; cfg.batch = 1; cfg.graph = o.graph; cfg.fusedAttention = o.fusedAttn; cfg.customGemmMaxK = o.gemmMaxK; cfg.fusedLoader = o.fusedLoader; cfg.gnFuse = o.gnFuse; cfg.scoreBytes = o.scoreMB << 20; if (o.memMB) cfg.arenaBytes = o.memMB << 20;
        s->eng = std::make_unique<Engine>(*s->dm, cfg); s->eng->setPrompt({320, 736, 3240});
        std::vector<uint8_t> seed(32, 1), x0(32768); s->eng->run(seed.data(), x0.data());
        slots.push_back(std::move(s));
        logln("verify-server: GPU " + std::to_string(dev) + " ready (" + std::to_string(s ? 0 : slots.back()->eng->lastMs) + ")");
    }
    std::string mh = model.model_hash_natural(); auto mhU = model.model_hash_uint256();
    std::atomic<size_t> rr{0}; std::atomic<uint64_t> nreq{0}, nvalid{0};
    Httpd srv(o.host, o.port, [&](const HttpRequest& r) -> HttpResponse {
        HttpResponse resp;
        if (r.method == "GET" && r.path == "/health") { resp.body = json{{"status", "ok"}, {"engine", "supr-meow-vela"}, {"version", VERSION_STR}, {"model_hash", mh}, {"gpus", slots.size()}, {"requests", nreq.load()}, {"valid", nvalid.load()}}.dump(); return resp; }
        if (r.method != "POST" || r.path != "/verify") { resp.status = 404; resp.body = "{\"error\":\"not found\"}"; return resp; }
        json q; try { q = json::parse(r.body); } catch (...) { resp.status = 400; resp.body = "{\"error\":\"bad json\"}"; return resp; }
        auto fail = [&](const char* reason) { resp.body = json{{"valid", false}, {"verify_ms", 0}, {"recomputed_hpoi_ok", false}, {"x0_match", false}, {"reason", reason}}.dump(); return resp; };
        nreq++;
        std::string hh = q.value("header80_hex", ""), xh = q.value("x0_hex", ""), ph = q.value("hashpoi_hex", ""), mhq = q.value("model_hash_hex", "");
        std::vector<int> tokens; for (auto& t : q.value("prompt_tokens", json::array())) tokens.push_back(t.get<int>());
        for (auto& c : mhq) c = char(tolower(c));
        if (mhq != mh) return fail("model_mismatch");
        std::vector<uint8_t> header, x0c, hp;
        try { header = unhex(hh); x0c = unhex(xh); hp = unhex(ph); } catch (...) { resp.status = 400; resp.body = "{\"error\":\"bad hex\"}"; return resp; }
        if (header.size() != 80) return fail("bad_header_len");
        if (x0c.size() != 32768) return fail("bad_x0_len");
        if (hp.size() != 32) return fail("bad_hpoi_len");
        if (tokens.empty() || tokens.size() > 75) return fail("bad_tokens");
        auto t0 = std::chrono::steady_clock::now();
        auto seed = poiSeed(header.data(), mhU);
        std::vector<uint8_t> x0(32768);
        { Slot& s = *slots[rr++ % slots.size()]; std::lock_guard<std::mutex> l(s.mu); s.eng->setPrompt(tokens); s.eng->run(seed.data(), x0.data()); }
        auto hpoi = poiOut(seed, x0.data());
        int ms = int(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        bool x0ok = x0 == x0c, hpok = std::memcmp(hpoi.data(), hp.data(), 32) == 0, valid = x0ok && hpok;
        if (valid) nvalid++;
        resp.body = json{{"valid", valid}, {"verify_ms", ms}, {"recomputed_hpoi_ok", hpok}, {"x0_match", x0ok}, {"reason", valid ? json(nullptr) : json(!x0ok ? "x0_mismatch" : "hpoi_mismatch")}}.dump();
        logln(std::string("verify ") + (valid ? "VALID" : "INVALID") + " " + std::to_string(ms) + " ms height-prompt " + std::to_string(tokens.size()) + " tokens");
        return resp;
    }, o.threads);
    srv.start();
    logln("verify-server listening on " + o.host + ":" + std::to_string(o.port) + " (model " + mh.substr(0, 16) + "…, " + std::to_string(slots.size()) + " GPU)");
    while (!g_stop) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    srv.stop();
    return 0;
}

// ------------------------------------------------------------------ mining
struct LaneStats { std::atomic<uint64_t> inferences{0}; std::atomic<double> bestDiff{0}; };
static double hashDifficulty(const uint8_t* hLE) {
    // difficulty = (2^256-1) / hash  (hash as little-endian integer)
    long double v = 0; for (int i = 31; i >= 0; --i) v = v * 256 + hLE[i];
    if (v <= 0) return 1e300;
    return (long double)std::ldexp(1.0L, 256) / v;
}
static void mineLane(const Opts& o, const DevModel& dm, EngineConfig cfg, StratumClient& client, int dev, int lane, uint32_t en2, LaneStats& ls, std::array<uint8_t, 32> modelHashU256, std::string modelHashNat) {
    std::string tag = "GPU " + std::to_string(dev) + " lane " + std::to_string(lane) + ": ";
    try {
        Engine eng(dm, cfg);
        int B = cfg.batch;
        std::vector<uint8_t> seeds(size_t(B) * 32), x0(size_t(B) * 32768);
        std::vector<std::array<uint8_t, 32>> seedArr(B);
        std::vector<std::array<uint8_t, 80>> headers(B);
        uint64_t lastGen = 0; Job job; uint8_t tmpl[80]; bool have = false; uint32_t nonce = uint32_t(lane) << 28; std::string en2Hex;
        Job pendJob; std::vector<std::array<uint8_t, 80>> pendHeaders(B); std::vector<std::array<uint8_t, 32>> pendSeeds(B);
        { char b[16]; std::snprintf(b, sizeof(b), "%08x", en2); en2Hex = b; }
        bool warned = false;
        while (!g_stop) {
            Job j;
            if (!client.currentJob(j) || client.extranonce1Hex().empty()) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); continue; }
            if (j.generation != lastGen) {
                if (!j.modelHash.empty() && j.modelHash != modelHashNat) { if (!warned) { logln(tag + "job model_hash " + j.modelHash + " != loaded model " + modelHashNat + " — refusing to mine"); warned = true; } std::this_thread::sleep_for(std::chrono::seconds(5)); continue; }
                // coinbase + merkle root with our extranonce
                std::vector<uint8_t> cb = j.coinbase1; auto en1 = unhex(client.extranonce1Hex()); cb.insert(cb.end(), en1.begin(), en1.end()); auto e2 = unhex(en2Hex); cb.insert(cb.end(), e2.begin(), e2.end()); cb.insert(cb.end(), j.coinbase2.begin(), j.coinbase2.end());
                auto root = sha256d(cb.data(), cb.size());
                for (auto& br : j.branch) { uint8_t buf[64]; std::memcpy(buf, root.data(), 32); std::memcpy(buf + 32, br.data(), 32); root = sha256d(buf, 64); }
                auto le32 = [](uint8_t* p, uint32_t v) { p[0] = v; p[1] = v >> 8; p[2] = v >> 16; p[3] = v >> 24; };
                le32(tmpl, j.version); std::memcpy(tmpl + 4, j.prevHashLE.data(), 32); std::memcpy(tmpl + 36, root.data(), 32); le32(tmpl + 68, j.ntime); le32(tmpl + 72, j.bits);
                if (!have || j.promptTokens != job.promptTokens) { auto t0 = std::chrono::steady_clock::now(); eng.setPrompt(j.promptTokens); logln(tag + "prompt set (" + std::to_string(j.promptTokens.size()) + " tokens, " + std::to_string(int(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count())) + " ms)"); }
                job = j; lastGen = j.generation; have = true;
            }
            // batch k is submitted while batch k-1 is being verified/hashed on the host (two in flight per lane)
            auto prepare = [&](std::vector<std::array<uint8_t, 80>>& hdrs, std::vector<std::array<uint8_t, 32>>& sds, std::vector<uint8_t>& sbuf) {
                for (int b = 0; b < B; ++b) {
                    std::memcpy(hdrs[b].data(), tmpl, 76); uint32_t n = nonce++; hdrs[b][76] = n; hdrs[b][77] = n >> 8; hdrs[b][78] = n >> 16; hdrs[b][79] = n >> 24;
                    sds[b] = poiSeed(hdrs[b].data(), modelHashU256); std::memcpy(&sbuf[b * 32], sds[b].data(), 32);
                }
            };
            if (eng.inFlight() == 0) { prepare(headers, seedArr, seeds); eng.submit(seeds.data()); pendJob = job; pendHeaders = headers; pendSeeds = seedArr; }
            // next batch on the current job
            prepare(headers, seedArr, seeds); eng.submit(seeds.data());
            // collect the previous batch (belongs to pendJob)
            eng.collect(x0.data());
            ls.inferences += B;
            for (int b = 0; b < B; ++b) {
                auto hp = poiOut(pendSeeds[b], &x0[b * 32768]);
                uint8_t h112[112]; std::memcpy(h112, pendHeaders[b].data(), 80); std::memcpy(h112 + 80, hp.data(), 32);
                auto hash = sha256d(h112, 112);
                double d = hashDifficulty(hash.data());
                double best = ls.bestDiff.load(); while (d > best && !ls.bestDiff.compare_exchange_weak(best, d)) {}
                if (hashMeetsTarget(hash.data(), pendJob.target)) {
                    uint32_t n = uint32_t(pendHeaders[b][76]) | (uint32_t(pendHeaders[b][77]) << 8) | (uint32_t(pendHeaders[b][78]) << 16) | (uint32_t(pendHeaders[b][79]) << 24);
                    char nb[16]; std::snprintf(nb, sizeof(nb), "%08x", n);
                    client.submit(pendJob.id, en2Hex, pendJob.ntimeHex, nb, hex(&x0[b * 32768], 32768), hex(hp.data(), 32));
                    logln(tag + "share found, diff " + std::to_string((long long)d) + " (job " + pendJob.id + ", target " + std::to_string((long long)pendJob.difficulty) + ")");
                }
            }
            pendJob = job; pendHeaders = headers; pendSeeds = seedArr;
        }
    } catch (const std::exception& e) { logln(tag + "FATAL " + e.what()); g_minerFailed = true; g_stop = true; }
}
static int runMiner(const Opts& o, const Poisd& model, const std::vector<int>& devs) {
    if (o.url.empty() || o.wallet.empty()) { usage(); return 1; }
    std::string host = o.url; int port = 7601; size_t c = host.rfind(':'); if (c != std::string::npos) { port = std::stoi(host.substr(c + 1)); host = host.substr(0, c); }
    if (host.rfind("stratum+tcp://", 0) == 0) host = host.substr(14); if (host.rfind("stratum+ssl://", 0) == 0) host = host.substr(14);
    StratumClient client(host, port, o.tls, o.wallet, o.password, std::string("supr-meow-vela/") + VERSION_STR);
    client.log = logln;
    if (o.suggestDiff > 0) client.suggestDifficulty(o.suggestDiff);
    client.start();
    std::vector<std::unique_ptr<DevModel>> dms; std::vector<std::thread> ths; std::vector<std::unique_ptr<LaneStats>> stats;
    std::mt19937 rng(std::random_device{}());
    auto mhU = model.model_hash_uint256(); std::string mhN = model.model_hash_natural();
    int laneId = 0;
    for (int dev : devs) {
        auto dm = std::make_unique<DevModel>(); dm->load(dev, model);
        size_t freeB = 0, totB = 0; cudaMemGetInfo(&freeB, &totB);
        EngineConfig cfg; cfg.batch = o.batch; cfg.graph = o.graph; cfg.fusedAttention = o.fusedAttn; cfg.customGemmMaxK = o.gemmMaxK; cfg.fusedLoader = o.fusedLoader; cfg.gnFuse = o.gnFuse; cfg.scoreBytes = o.scoreMB << 20;
        cfg.arenaBytes = o.memMB ? (o.memMB << 20) : (freeB > (1024u << 20) ? (freeB - (768u << 20)) / o.lanes : freeB / 2 / o.lanes);
        logln("GPU " + std::to_string(dev) + ": model " + std::to_string(dm->bytes >> 20) + " MiB on device, " + std::to_string(freeB >> 20) + " MiB free, arena " + std::to_string(cfg.arenaBytes >> 20) + " MiB x " + std::to_string(o.lanes) + " lanes, batch " + std::to_string(o.batch));
        for (int l = 0; l < o.lanes; ++l) {
            stats.emplace_back(new LaneStats()); uint32_t en2 = (uint32_t(rng()) & 0xffff0000u) | uint32_t(laneId++);
            ths.emplace_back(mineLane, std::cref(o), std::cref(*dm), cfg, std::ref(client), dev, l, en2, std::ref(*stats.back()), mhU, mhN);
        }
        dms.push_back(std::move(dm));
    }
    // stats reporter
    uint64_t lastTot = 0; auto lastT = std::chrono::steady_clock::now(); auto start = lastT;
    while (!g_stop) {
        for (int i = 0; i < 100 && !g_stop; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        auto now = std::chrono::steady_clock::now(); double dt = std::chrono::duration<double>(now - lastT).count(); lastT = now;
        uint64_t tot = 0; std::string per; size_t si = 0; double best = 0;
        for (size_t di = 0; di < devs.size(); ++di) {
            uint64_t dsum = 0; for (int l = 0; l < o.lanes; ++l, ++si) { dsum += stats[si]->inferences; best = std::max(best, stats[si]->bestDiff.load()); }
            static std::vector<uint64_t> lastDev; if (lastDev.size() < devs.size()) lastDev.resize(devs.size(), 0);
            double r = (dsum - lastDev[di]) / dt; lastDev[di] = dsum; tot += dsum;
            GpuTelemetry t = gpuTelemetry(devs[di]);
            char b[160]; std::snprintf(b, sizeof(b), " GPU%d %.1f inf/s%s%s", devs[di], r, t.tempC >= 0 ? (" " + std::to_string(t.tempC) + "C").c_str() : "", t.powerW >= 0 ? (" " + std::to_string(t.powerW) + "W").c_str() : "");
            per += b;
        }
        double rate = (tot - lastTot) / dt; lastTot = tot;
        double up = std::chrono::duration<double>(now - start).count();
        char line[256]; std::snprintf(line, sizeof(line), "speed %.1f inf/s |%s | shares A%llu R%llu S%llu | best %.0f | up %.0fm", rate, per.c_str(), (unsigned long long)client.stats.accepted.load(), (unsigned long long)client.stats.rejected.load(), (unsigned long long)client.stats.stale.load(), best, up / 60);
        logln(line);
    }
    client.stop();
    for (auto& t : ths) t.join();
    return g_minerFailed ? 1 : 0;
}

int main(int argc, char** argv) {
    Opts o;
    try { o = parse(argc, argv); } catch (const std::exception& e) { std::cerr << e.what() << "\n"; usage(); return 1; }
    std::signal(SIGINT, [](int) { g_stop = true; }); std::signal(SIGTERM, [](int) { g_stop = true; });
    auto gpus = listGpus();
    if (gpus.empty()) { std::cerr << "no CUDA devices\n"; return 1; }
    std::vector<int> devs = o.devices; if (devs.empty()) for (auto& g : gpus) devs.push_back(g.index);
    for (auto& g : gpus) logln("GPU " + std::to_string(g.index) + ": " + g.name + " (sm_" + std::to_string(g.cc) + ", " + std::to_string(g.memBytes >> 20) + " MiB, " + std::to_string(g.sms) + " SMs)");
    Poisd model; std::string err;
    auto t0 = std::chrono::steady_clock::now();
    if (!model.load(o.model, err)) { std::cerr << "model: " << err << "\n"; return 1; }
    logln("model " + o.model + ": " + std::to_string(model.entries().size()) + " tensors, hash " + model.model_hash_natural() + " (" + std::to_string(int(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count())) + " ms)");
    try {
        if (o.testAttn) {
            DevModel dm; dm.load(devs[0], model); EngineConfig cfg; cfg.batch = 2; cfg.graph = false; Engine eng(dm, cfg);
            struct T { int B, Tq, Tk, heads; bool causal; int bits; } tests[] = {{1, 64, 64, 1, false, 20}, {2, 96, 77, 16, true, 24}, {1, 256, 256, 2, false, 28}, {2, 1024, 64, 10, false, 30}, {2, 4096, 77, 5, false, 31}, {1, 4096, 4096, 5, false, 31}};
            int fails = 0;
            for (int bits : {0, 14, 17, 20, 22, 23, 28, 31}) {
                long bad = eng.testAttention(1, 65, 257, 2, false, bits);
                logln("test-attn boundary scale=" + std::to_string(bits) + ": " + (bad ? std::to_string(bad) + " mismatches" : "OK"));
                fails += bad != 0;
            }
            for (auto& t : tests) { long bad = eng.testAttention(t.B, t.Tq, t.Tk, t.heads, t.causal, t.bits); logln("test-attn B=" + std::to_string(t.B) + " Tq=" + std::to_string(t.Tq) + " Tk=" + std::to_string(t.Tk) + " heads=" + std::to_string(t.heads) + (t.causal ? " causal" : "") + ": " + (bad ? std::to_string(bad) + " mismatches" : "OK")); fails += bad != 0; }
            return fails ? 1 : 0;
        }
        if (o.check) {
            auto seed = sha256("x", 1);
            for (int dev : devs) {
                logln("Checking GPU " + std::to_string(dev));
                int rc = runSingle(o, model, {dev}, seed, {320, 736, 3240}, "bef682cc9651fe8d42f075e98b907432f86a0b9e9a40e772fef99fc424f5c518");
                if (rc) return rc;
            }
            return 0;
        }
        if (!o.headerHex.empty()) {
            auto hdr = unhex(o.headerHex); if (hdr.size() != 80) throw std::runtime_error("--header must be 80 bytes");
            auto seed = poiSeed(hdr.data(), model.model_hash_uint256());
            return runSingle(o, model, devs, seed, o.tokensStr.empty() ? std::vector<int>{320, 736, 3240} : parseInts(o.tokensStr), nullptr);
        }
        if (!o.seedHex.empty()) { auto s = unhex(o.seedHex); if (s.size() != 32) throw std::runtime_error("--seed must be 32 bytes"); std::array<uint8_t, 32> seed; std::copy(s.begin(), s.end(), seed.begin()); return runSingle(o, model, devs, seed, o.tokensStr.empty() ? std::vector<int>{320, 736, 3240} : parseInts(o.tokensStr), nullptr); }
        if (o.bench) return runBench(o, model, devs);
        if (o.verifyServer) return runVerifyServer(o, model, devs);
        return runMiner(o, model, devs);
    } catch (const std::exception& e) { logln(std::string("FATAL: ") + e.what()); return 1; }
}
