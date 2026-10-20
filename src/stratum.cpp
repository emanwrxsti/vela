#include "stratum.h"
#include "common.h"
#include <nlohmann/json.hpp>
#include "compat.h"
#include <cmath>
#include <cstring>
#include <chrono>
#ifdef MEOW_HAVE_OPENSSL
#include <openssl/ssl.h>
#include <openssl/err.h>
#endif
#if !defined(_WIN32)
#include <fcntl.h>
#include <poll.h>
#endif

using json = nlohmann::json;
namespace meow {

std::array<uint8_t, 32> targetFromDifficulty(double difficulty) {
    std::array<uint8_t, 32> t{};
    if (!(difficulty >= 1)) { t.fill(0xff); return t; }
    uint64_t d = uint64_t(std::ceil(difficulty));            // stricter when fractional
    // long division of 2^256-1 (all 0xff) by d, most significant byte first
    uint64_t rem = 0;
    for (int i = 31; i >= 0; --i) t[i] = div256_step(rem, d);
    return t;
}
bool hashMeetsTarget(const uint8_t* h, const std::array<uint8_t, 32>& t) {
    for (int i = 31; i >= 0; --i) { if (h[i] < t[i]) return true; if (h[i] > t[i]) return false; }
    return true;
}

StratumClient::StratumClient(std::string host, int port, bool tls, std::string wallet, std::string password, std::string ua)
    : host_(std::move(host)), port_(port), tls_(tls), wallet_(std::move(wallet)), password_(std::move(password)), ua_(std::move(ua)) {}
StratumClient::~StratumClient() { stop(); }
void StratumClient::start() { run_ = true; th_ = std::thread([this] { loop(); }); }
void StratumClient::stop() { run_ = false; { std::lock_guard<std::mutex> l(wmu_); stats.connected = false; if (fd_ >= 0) net_shutdown(fd_); } if (th_.joinable()) th_.join(); disconnect(); }
bool StratumClient::currentJob(Job& out) const { std::lock_guard<std::mutex> l(mu_); if (!stats.connected || job_.id.empty() || job_.extranonce1Hex.empty()) return false; out = job_; return true; }

bool StratumClient::connect() {
    net_global_init();
    struct addrinfo hints{}; hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* res = nullptr;
    if (getaddrinfo(host_.c_str(), std::to_string(port_).c_str(), &hints, &res) != 0 || !res) { log("stratum: cannot resolve " + host_); return false; }
    sock_t fd = (sock_t)MEOW_INVALID_SOCK;
    for (auto* p = res; p; p = p->ai_next) {
        fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if ((intptr_t)fd == MEOW_INVALID_SOCK) continue;
        net_set_timeout((intptr_t)fd, 15);
        if (::connect(fd, p->ai_addr, (int)p->ai_addrlen) == 0) break;
        net_close((intptr_t)fd); fd = (sock_t)MEOW_INVALID_SOCK;
    }
    freeaddrinfo(res);
    if ((intptr_t)fd == MEOW_INVALID_SOCK) { log("stratum: connect to " + host_ + ":" + std::to_string(port_) + " failed: " + net_strerror()); return false; }
    int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof(one)); setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, (const char*)&one, sizeof(one));
    void* newSsl = nullptr; void* newCtx = nullptr;
    if (tls_) {
#ifdef MEOW_HAVE_OPENSSL
        OPENSSL_init_ssl(0, nullptr);
        SSL_CTX* ctx = SSL_CTX_new(TLS_client_method()); SSL* ssl = SSL_new(ctx);
        SSL_set_tlsext_host_name(ssl, host_.c_str()); SSL_set_fd(ssl, fd);
        ERR_clear_error();
        if (SSL_connect(ssl) != 1) { log("stratum: TLS handshake failed"); SSL_free(ssl); SSL_CTX_free(ctx); net_close((intptr_t)fd); return false; }
#if !defined(_WIN32)
        { int fl = fcntl(fd, F_GETFL, 0); if (fl < 0 || fcntl(fd, F_SETFL, fl | O_NONBLOCK) < 0) {
            log("stratum: cannot make TLS socket non-blocking");
            SSL_free(ssl); SSL_CTX_free(ctx); net_close((intptr_t)fd); return false;
        } }
#endif
        newSsl = ssl; newCtx = ctx;
#else
        log("stratum: built without OpenSSL, --tls unavailable"); net_close((intptr_t)fd); return false;
#endif
    }
    {
        // Publish connection and job state together. Submissions hold the same
        // I/O lock from validation through writing. Lock order: wmu_ then mu_.
        std::lock_guard<std::mutex> io(wmu_);
        std::lock_guard<std::mutex> state(mu_);
        fd_ = (intptr_t)fd; ssl_ = newSsl; sslCtx_ = newCtx;
        pending_.clear(); job_ = Job(); haveDiff_ = false; extranonce1_.clear(); connJobs_.clear();
        ++connectionId_;
        stats.connected = run_.load();
    }
    rbuf_.clear();
    lastRx_ = lastPing_ = std::chrono::steady_clock::now();
    return true;
}
void StratumClient::disconnect() {
    std::lock_guard<std::mutex> l(wmu_);          // never free the SSL*/close the fd under a concurrent submit
    {
        std::lock_guard<std::mutex> state(mu_);
        stats.connected = false; job_ = Job(); extranonce1_.clear(); connJobs_.clear(); pending_.clear();
    }
#ifdef MEOW_HAVE_OPENSSL
    // Fatal TLS errors also reach this path; SSL_shutdown is invalid after those.
    if (ssl_) { SSL_free((SSL*)ssl_); ssl_ = nullptr; }
    if (sslCtx_) { SSL_CTX_free((SSL_CTX*)sslCtx_); sslCtx_ = nullptr; }
#endif
    if (fd_ >= 0) { net_close(fd_); fd_ = -1; }
}
bool StratumClient::sendLine(const std::string& s) {
    std::lock_guard<std::mutex> l(wmu_);
    return sendLineLocked(s);
}
bool StratumClient::sendLineLocked(const std::string& s) {
    if (fd_ < 0) return false;
    std::string line = s + "\n"; size_t off = 0;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (off < line.size()) {
        long long n;
#ifdef MEOW_HAVE_OPENSSL
        if (ssl_) {
            ERR_clear_error();
            n = SSL_write((SSL*)ssl_, line.data() + off, int(line.size() - off));
            if (n <= 0) {
                int e = SSL_get_error((SSL*)ssl_, int(n));
#if !defined(_WIN32)
                if ((e == SSL_ERROR_WANT_WRITE || e == SSL_ERROR_WANT_READ) && std::chrono::steady_clock::now() < deadline) {
                    pollfd p{int(fd_), short(e == SSL_ERROR_WANT_WRITE ? POLLOUT : POLLIN), 0}; poll(&p, 1, 200); continue;   // same buffer, as OpenSSL requires
                }
#endif
                stats.connected = false;
                net_shutdown(fd_);                    // wake the reader -> immediate reconnect
                return false;
            }
        } else
#endif
        {
            n = net_send(fd_, line.data() + off, line.size() - off);
            if (n <= 0) { stats.connected = false; net_shutdown(fd_); return false; }
        }
        off += size_t(n);
    }
    (void)deadline;
    return true;
}
// One read attempt. TLS: non-blocking SSL_read under wmu_ (never concurrent with SSL_write);
// waits for data with poll() *outside* the lock. Plain TCP: blocking recv with SO_RCVTIMEO.
long long StratumClient::ioRead(char* buf, size_t n, bool& wouldBlock) {
    wouldBlock = false;
#ifdef MEOW_HAVE_OPENSSL
    if (ssl_) {
        long long r; int e;
        { std::lock_guard<std::mutex> l(wmu_); if (!ssl_) return 0; ERR_clear_error(); r = SSL_read((SSL*)ssl_, buf, int(n)); e = r > 0 ? SSL_ERROR_NONE : SSL_get_error((SSL*)ssl_, int(r)); }
        if (r > 0) return r;
        if (e == SSL_ERROR_WANT_READ || e == SSL_ERROR_WANT_WRITE) {
#if !defined(_WIN32)
            pollfd p{int(fd_), short(e == SSL_ERROR_WANT_WRITE ? POLLOUT : POLLIN), 0}; poll(&p, 1, 1000);
#endif
            wouldBlock = true; return -1;
        }
        return 0;
    }
#endif
    long long r = net_recv(fd_, buf, n);
    if (r < 0 && net_would_block()) wouldBlock = true;
    return r;
}
bool StratumClient::readLine(std::string& out) {
    for (;;) {
        size_t nl = rbuf_.find('\n');
        if (nl != std::string::npos) { out = rbuf_.substr(0, nl); rbuf_.erase(0, nl + 1); return true; }
        if (rbuf_.size() > (2u << 20)) return false;
        char buf[65536]; bool wb = false;
        long long n = ioRead(buf, sizeof(buf), wb);
        if (n > 0) { rbuf_.append(buf, size_t(n)); lastRx_ = std::chrono::steady_clock::now(); continue; }
        if (!wb || !run_) return false;               // EOF / error / stopping
        // Liveness: blocks can be 30+ min apart, so silence alone is normal. Ping after 2 min of
        // silence (pool answers "pong"); give up and reconnect after 5 min with nothing at all.
        auto now = std::chrono::steady_clock::now();
        if (now - lastRx_ > std::chrono::minutes(5)) { log("stratum: no data from pool for 5 min, reconnecting"); return false; }
        if (now - lastRx_ > std::chrono::minutes(2) && now - lastPing_ > std::chrono::minutes(2)) {
            lastPing_ = now; int id = nextId_++;
            { std::lock_guard<std::mutex> l(mu_); pending_[id] = "mining.ping"; }
            if (!sendLine(json{{"id", id}, {"method", "mining.ping"}, {"params", json::object()}}.dump())) return false;
        }
    }
}
void StratumClient::loop() {
    int backoff = 2;
    while (run_) {
        if (!connect()) { for (int i = 0; i < backoff * 10 && run_; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100)); backoff = std::min(backoff * 2, 60); continue; }
        if (!run_) { disconnect(); break; }
        backoff = 2; stats.reconnects++;
        log("stratum: connected to " + host_ + ":" + std::to_string(port_) + (tls_ ? " (TLS)" : ""));
        json sub = {{"id", 1}, {"method", "mining.subscribe"}, {"params", {{"user_agent", ua_}}}};
        json auth = {{"id", 2}, {"method", "mining.authorize"}, {"params", {{"wallet", wallet_}, {"password", password_}}}};
        if (!sendLine(sub.dump()) || !sendLine(auth.dump())) { disconnect(); continue; }
        std::string suggestion;
        { std::lock_guard<std::mutex> l(mu_); suggestion = suggestDiff_; }
        if (!suggestion.empty()) sendLine(suggestion);
        std::string line;
        while (run_ && readLine(line)) { try { handle(line); } catch (const std::exception& e) { log(std::string("stratum: bad message: ") + e.what()); } }
        if (run_) log("stratum: connection lost, reconnecting");
        disconnect();
    }
}
void StratumClient::handle(const std::string& line) {
    if (line.empty()) return;
    json m = json::parse(line);
    if (m.contains("method")) {
        std::string method = m["method"].get<std::string>();
        json p = m.value("params", json::object());
        if (method == "mining.set_difficulty") {
            double d = p.is_object() ? p.value("difficulty", 0.0) : (p.is_array() && !p.empty() ? p[0].get<double>() : 0.0);
            std::lock_guard<std::mutex> l(mu_); difficulty_ = d; haveDiff_ = true;
            log("stratum: difficulty " + std::to_string(d));
        } else if (method == "mining.notify") {
            Job j;
            j.id = p.at("job_id").get<std::string>(); j.prevHashHex = p.at("prev_hash").get<std::string>();
            j.coinbase1Hex = p.at("coinbase1").get<std::string>(); j.coinbase2Hex = p.at("coinbase2").get<std::string>();
            j.versionHex = p.at("version").get<std::string>(); j.bitsHex = p.at("bits").get<std::string>(); j.ntimeHex = p.at("ntime").get<std::string>();
            j.clean = p.value("clean_jobs", false); j.height = p.value("height", 0); j.poiVersion = p.value("poi_version", 2);
            j.modelHash = p.value("model_hash", std::string());
            for (auto& b : p.at("merkle_branch")) j.merkleBranch.push_back(b.get<std::string>());
            for (auto& t : p.at("prompt_tokens")) j.promptTokens.push_back(t.get<int>());
            j.version = uint32_t(std::stoul(j.versionHex, nullptr, 16)); j.ntime = uint32_t(std::stoul(j.ntimeHex, nullptr, 16)); j.bits = uint32_t(std::stoul(j.bitsHex, nullptr, 16));
            auto ph = unhex(j.prevHashHex); if (ph.size() != 32) throw std::runtime_error("prev_hash length");
            for (int i = 0; i < 32; ++i) j.prevHashLE[i] = ph[31 - i];
            j.coinbase1 = unhex(j.coinbase1Hex); j.coinbase2 = unhex(j.coinbase2Hex);
            for (auto& b : j.merkleBranch) { auto v = unhex(b); if (v.size() != 32) throw std::runtime_error("branch length"); std::array<uint8_t, 32> a; std::copy(v.begin(), v.end(), a.begin()); j.branch.push_back(a); }
            {
                std::lock_guard<std::mutex> l(mu_);
                j.difficulty = haveDiff_ ? difficulty_ : 2048; j.target = targetFromDifficulty(j.difficulty);
                j.connectionId = connectionId_; j.extranonce1Hex = extranonce1_;
                j.generation = ++gen_; job_ = j; connJobs_.insert(j.id);
                if (!j.modelHash.empty()) modelHash_ = j.modelHash;
            }
            log("stratum: job " + j.id + " height " + std::to_string(j.height) + (j.clean ? " (clean)" : "") + " diff " + std::to_string((long long)j.difficulty) + " tokens " + std::to_string(j.promptTokens.size()));
            if (onJob) onJob(j);
        } else if (method == "mining.ping") {
            json r = {{"id", m.value("id", json())}, {"result", "pong"}, {"error", nullptr}}; sendLine(r.dump());
        } else {
            log("stratum: unhandled method " + method);
        }
        return;
    }
    if (!m.contains("id") || m["id"].is_null()) return;
    int id = m["id"].get<int>();
    bool ok = m.contains("result") && !m["result"].is_null() && (m["result"].is_boolean() ? m["result"].get<bool>() : true);
    std::string err;
    if (m.contains("error") && !m["error"].is_null()) {
        auto& e = m["error"]; ok = false;
        if (e.is_object()) { err = "code " + std::to_string(e.value("code", 0)) + " " + e.value("msg", e.value("message", std::string())); if (e.contains("data")) err += " " + e["data"].dump(); }
        else err = e.dump();
    }
    if (id == 1) {
        if (!ok) { log("stratum: subscribe failed: " + err); return; }
        auto& r = m["result"];
        std::lock_guard<std::mutex> l(mu_);
        extranonce1_ = r.value("extranonce1", std::string());
        job_.extranonce1Hex = extranonce1_;
        if (r.contains("model") && r["model"].is_object()) modelHash_ = r["model"].value("model_hash", std::string());
        log("stratum: subscribed, extranonce1 " + extranonce1_ + " model " + modelHash_.substr(0, 16) + "…");
        return;
    }
    if (id == 2) { log(ok ? "stratum: authorized " + wallet_ : "stratum: authorize failed: " + err); return; }
    std::string method;
    { std::lock_guard<std::mutex> l(mu_); auto it = pending_.find(id); if (it != pending_.end()) { method = it->second; pending_.erase(it); } }
    if (method == "mining.submit") {
        if (ok) { stats.accepted++; log("share accepted (" + std::to_string(stats.accepted.load()) + "/" + std::to_string(stats.submitted.load()) + ")"); }
        else { if (err.find("code 21") != std::string::npos) stats.stale++; else stats.rejected++; log("share REJECTED: " + err); }
    } else if (method == "mining.ping") {
        // keepalive answer; lastRx_ was already refreshed by readLine
    } else if (method == "mining.suggest_difficulty") {
        log(ok ? "stratum: suggest_difficulty accepted" : "stratum: suggest_difficulty: " + err);
    }
}
void StratumClient::submit(const Job& job, const std::string& en2, const std::string& ntime, const std::string& nonce, const std::string& x0Hex, const std::string& hpoi) {
    // A share is only valid on the connection whose job (and extranonce1) produced it. Shares found
    // while reconnecting, or for a job from the previous connection, are discarded instead of being
    // sent (they would be rejected and can count towards the pool's invalid-share ban).
    int id = nextId_++;
    json r = {{"id", id}, {"method", "mining.submit"}, {"params", {{"worker", wallet_}, {"job_id", job.id}, {"extranonce2", en2}, {"ntime", ntime}, {"nonce", nonce}, {"x0_hex", x0Hex}, {"hashpoi_hex", hpoi}}}};
    std::string line = r.dump();
    // Teardown/reconnect cannot slip between validation and the actual write.
    // A connection token also catches pools that reuse job ids after reconnect.
    std::lock_guard<std::mutex> io(wmu_);
    {
        std::lock_guard<std::mutex> l(mu_);
        if (!stats.connected || job.connectionId != connectionId_ || !connJobs_.count(job.id)) {
            stats.discarded++;
            log(std::string("stratum: share discarded (") + (stats.connected ? "job " + job.id + " is from a previous connection" : std::string("not connected, reconnecting")) + ")");
            return;
        }
        pending_[id] = "mining.submit";
    }
    if (sendLineLocked(line)) { stats.submitted++; return; }
    { std::lock_guard<std::mutex> l(mu_); pending_.erase(id); }
    stats.discarded++;
    log("stratum: share discarded (connection lost while sending, reconnecting)");
}
void StratumClient::suggestDifficulty(double d) {
    int id = nextId_++;
    json r = {{"id", id}, {"method", "mining.suggest_difficulty"}, {"params", {{"difficulty", d}}}};
    std::string line = r.dump();
    { std::lock_guard<std::mutex> l(mu_); pending_[id] = "mining.suggest_difficulty"; suggestDiff_ = line; }
    sendLine(line);
}
}  // namespace meow
