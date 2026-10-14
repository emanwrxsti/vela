#include "stratum.h"
#include "common.h"
#include <nlohmann/json.hpp>
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <poll.h>
#include <cmath>
#include <cstring>
#include <chrono>
#ifdef MEOW_HAVE_OPENSSL
#include <openssl/ssl.h>
#include <openssl/err.h>
#endif

using json = nlohmann::json;
namespace meow {

std::array<uint8_t, 32> targetFromDifficulty(double difficulty) {
    std::array<uint8_t, 32> t{};
    if (!(difficulty >= 1)) { t.fill(0xff); return t; }
    uint64_t d = uint64_t(std::ceil(difficulty));            // stricter when fractional
    // long division of 2^256-1 (all 0xff) by d, most significant byte first
    unsigned __int128 rem = 0;
    for (int i = 31; i >= 0; --i) { rem = (rem << 8) | 0xff; t[i] = uint8_t(rem / d); rem %= d; }
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
void StratumClient::stop() { run_ = false; if (fd_ >= 0) ::shutdown(fd_, SHUT_RDWR); if (th_.joinable()) th_.join(); disconnect(); }
bool StratumClient::currentJob(Job& out) const { std::lock_guard<std::mutex> l(mu_); if (job_.id.empty()) return false; out = job_; return true; }

bool StratumClient::connect() {
    struct addrinfo hints{}; hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* res = nullptr;
    if (getaddrinfo(host_.c_str(), std::to_string(port_).c_str(), &hints, &res) != 0 || !res) { log("stratum: cannot resolve " + host_); return false; }
    int fd = -1;
    for (auto* p = res; p; p = p->ai_next) {
        fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) continue;
        struct timeval tv{15, 0}; setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)); setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
        ::close(fd); fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) { log("stratum: connect to " + host_ + ":" + std::to_string(port_) + " failed: " + std::strerror(errno)); return false; }
    int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)); setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
    fd_ = fd;
    if (tls_) {
#ifdef MEOW_HAVE_OPENSSL
        static bool init = false; if (!init) { SSL_library_init(); SSL_load_error_strings(); init = true; }
        SSL_CTX* ctx = SSL_CTX_new(TLS_client_method()); SSL* ssl = SSL_new(ctx);
        SSL_set_tlsext_host_name(ssl, host_.c_str()); SSL_set_fd(ssl, fd);
        if (SSL_connect(ssl) != 1) { log("stratum: TLS handshake failed"); SSL_free(ssl); SSL_CTX_free(ctx); ::close(fd); fd_ = -1; return false; }
        ssl_ = ssl; sslCtx_ = ctx;
#else
        log("stratum: built without OpenSSL, --tls unavailable"); ::close(fd); fd_ = -1; return false;
#endif
    }
    rbuf_.clear();
    return true;
}
void StratumClient::disconnect() {
#ifdef MEOW_HAVE_OPENSSL
    if (ssl_) { SSL_shutdown((SSL*)ssl_); SSL_free((SSL*)ssl_); ssl_ = nullptr; }
    if (sslCtx_) { SSL_CTX_free((SSL_CTX*)sslCtx_); sslCtx_ = nullptr; }
#endif
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    stats.connected = false;
}
bool StratumClient::sendLine(const std::string& s) {
    std::lock_guard<std::mutex> l(wmu_);
    if (fd_ < 0) return false;
    std::string line = s + "\n"; size_t off = 0;
    while (off < line.size()) {
        ssize_t n;
#ifdef MEOW_HAVE_OPENSSL
        if (ssl_) n = SSL_write((SSL*)ssl_, line.data() + off, int(line.size() - off)); else
#endif
        n = ::send(fd_, line.data() + off, line.size() - off, MSG_NOSIGNAL);
        if (n <= 0) return false;
        off += size_t(n);
    }
    return true;
}
bool StratumClient::readLine(std::string& out) {
    for (;;) {
        size_t nl = rbuf_.find('\n');
        if (nl != std::string::npos) { out = rbuf_.substr(0, nl); rbuf_.erase(0, nl + 1); return true; }
        if (rbuf_.size() > (2u << 20)) return false;
        char buf[65536]; ssize_t n;
#ifdef MEOW_HAVE_OPENSSL
        if (ssl_) n = SSL_read((SSL*)ssl_, buf, sizeof(buf)); else
#endif
        n = ::recv(fd_, buf, sizeof(buf), 0);
        if (n <= 0) {
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) { if (!run_) return false; continue; }   // timeout: keep waiting
            return false;
        }
        rbuf_.append(buf, size_t(n));
    }
}
void StratumClient::loop() {
    int backoff = 2;
    while (run_) {
        if (!connect()) { for (int i = 0; i < backoff * 10 && run_; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100)); backoff = std::min(backoff * 2, 60); continue; }
        backoff = 2; stats.connected = true; stats.reconnects++;
        log("stratum: connected to " + host_ + ":" + std::to_string(port_) + (tls_ ? " (TLS)" : ""));
        { std::lock_guard<std::mutex> l(mu_); pending_.clear(); job_ = Job(); haveDiff_ = false; }
        json sub = {{"id", 1}, {"method", "mining.subscribe"}, {"params", {{"user_agent", ua_}}}};
        json auth = {{"id", 2}, {"method", "mining.authorize"}, {"params", {{"wallet", wallet_}, {"password", password_}}}};
        if (!sendLine(sub.dump()) || !sendLine(auth.dump())) { disconnect(); continue; }
        if (!suggestDiff_.empty()) sendLine(suggestDiff_);
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
                j.generation = ++gen_; job_ = j;
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
    } else if (method == "mining.suggest_difficulty") {
        log(ok ? "stratum: suggest_difficulty accepted" : "stratum: suggest_difficulty: " + err);
    }
}
void StratumClient::submit(const std::string& jobId, const std::string& en2, const std::string& ntime, const std::string& nonce, const std::string& x0Hex, const std::string& hpoi) {
    int id = nextId_++;
    json r = {{"id", id}, {"method", "mining.submit"}, {"params", {{"worker", wallet_}, {"job_id", jobId}, {"extranonce2", en2}, {"ntime", ntime}, {"nonce", nonce}, {"x0_hex", x0Hex}, {"hashpoi_hex", hpoi}}}};
    { std::lock_guard<std::mutex> l(mu_); pending_[id] = "mining.submit"; }
    stats.submitted++;
    if (!sendLine(r.dump())) log("stratum: submit failed (not connected)");
}
void StratumClient::suggestDifficulty(double d) {
    int id = nextId_++;
    json r = {{"id", id}, {"method", "mining.suggest_difficulty"}, {"params", {{"difficulty", d}}}};
    { std::lock_guard<std::mutex> l(mu_); pending_[id] = "mining.suggest_difficulty"; }
    suggestDiff_ = r.dump();
    if (fd_ >= 0) sendLine(suggestDiff_);
}
}  // namespace meow
