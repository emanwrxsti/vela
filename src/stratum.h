// Stratum client for the Vela named-params dialect (VELA-STRATUM-PROTOCOL.md v1.1).
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <array>
#include <chrono>
#include <set>

namespace meow {
struct Job {
    std::string id, prevHashHex, coinbase1Hex, coinbase2Hex, versionHex, bitsHex, ntimeHex, modelHash;
    std::vector<std::string> merkleBranch;
    std::vector<int> promptTokens;
    int height = 0; bool clean = false; int poiVersion = 2;
    double difficulty = 0;                      // difficulty in effect when the job was issued
    uint64_t generation = 0;                    // increases on every notify
    uint32_t version = 0, ntime = 0, bits = 0;
    std::array<uint8_t, 32> prevHashLE{};       // internal byte order (as it goes into the header)
    std::array<uint8_t, 32> target{};           // share target, little-endian 256-bit
    std::vector<uint8_t> coinbase1, coinbase2;
    std::vector<std::array<uint8_t, 32>> branch;
};
struct StratumStats { std::atomic<uint64_t> accepted{0}, rejected{0}, submitted{0}, stale{0}, discarded{0}; std::atomic<bool> connected{false}; std::atomic<int> reconnects{0}; };

class StratumClient {
public:
    StratumClient(std::string host, int port, bool tls, std::string wallet, std::string password, std::string userAgent);
    ~StratumClient();
    void start();
    void stop();
    /** Snapshot of the current job (empty id when none). */
    bool currentJob(Job& out) const;
    uint64_t generation() const { return gen_.load(); }
    std::string extranonce1Hex() const { std::lock_guard<std::mutex> l(mu_); return extranonce1_; }
    std::string modelHash() const { std::lock_guard<std::mutex> l(mu_); return modelHash_; }
    /** Submit a share; result delivered asynchronously via stats + log. */
    void submit(const std::string& jobId, const std::string& extranonce2Hex, const std::string& ntimeHex, const std::string& nonceHex, const std::string& x0Hex, const std::string& hashpoiHex);
    void suggestDifficulty(double d);
    StratumStats stats;
    std::function<void(const std::string&)> log = [](const std::string&) {};
    std::function<void(const Job&)> onJob;
private:
    void loop();
    bool connect();
    void disconnect();
    bool sendLine(const std::string& s);
    bool sendLineLocked(const std::string& s);   // caller holds wmu_
    bool readLine(std::string& out);
    long long ioRead(char* buf, size_t n, bool& wouldBlock);
    void handle(const std::string& line);
    std::string host_; int port_; bool tls_; std::string wallet_, password_, ua_;
    intptr_t fd_ = -1; void* ssl_ = nullptr; void* sslCtx_ = nullptr; std::string rbuf_;
    std::thread th_; std::atomic<bool> run_{false};
    mutable std::mutex mu_; Job job_; std::atomic<uint64_t> gen_{0};
    std::string extranonce1_, modelHash_; double difficulty_ = 0; bool haveDiff_ = false;
    // wmu_ serialises EVERY use of the socket/TLS object that can race with another thread:
    // writes (submits come from the GPU lane threads), TLS reads (OpenSSL forbids concurrent
    // SSL_read/SSL_write on one SSL*) and teardown in disconnect(). 0.1.0-0.1.2 only locked
    // writes, which corrupted TLS sessions under load and could use a freed SSL* on reconnect.
    std::mutex wmu_; std::atomic<int> nextId_{10};
    std::chrono::steady_clock::time_point lastRx_{}, lastPing_{};
    std::set<std::string> connJobs_;             // job ids issued on the current connection
    std::map<int, std::string> pending_;         // id -> method
    std::string suggestDiff_;                    // sent after authorize if set
};
/** target = floor((2^256-1)/difficulty) as 32 little-endian bytes. */
std::array<uint8_t, 32> targetFromDifficulty(double difficulty);
/** little-endian 256-bit compare: hash <= target */
bool hashMeetsTarget(const uint8_t* hashLE, const std::array<uint8_t, 32>& target);
}  // namespace meow
