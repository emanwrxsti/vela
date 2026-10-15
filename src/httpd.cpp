#include "httpd.h"
#include "compat.h"
#include <cstring>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>

namespace meow {
namespace { std::mutex qmu; std::condition_variable qcv; std::queue<intptr_t> q; }

Httpd::Httpd(std::string host, int port, std::function<HttpResponse(const HttpRequest&)> h, int threads) : host_(std::move(host)), port_(port), handler_(std::move(h)), threads_(threads) {}
Httpd::~Httpd() { stop(); }
void Httpd::start() {
    net_global_init();
    sock_t ls = ::socket(AF_INET, SOCK_STREAM, 0);
    lfd_ = (intptr_t)ls;
    int one = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof(one));
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(uint16_t(port_)); a.sin_addr.s_addr = host_.empty() || host_ == "0.0.0.0" ? INADDR_ANY : inet_addr(host_.c_str());
    if (::bind(ls, (sockaddr*)&a, sizeof(a)) != 0) throw std::runtime_error(std::string("httpd bind: ") + net_strerror());
    if (::listen(ls, 64) != 0) throw std::runtime_error("httpd listen");
    run_ = true;
    for (int i = 0; i < threads_; ++i) workers_.emplace_back([this] {
        for (;;) {
            intptr_t fd; { std::unique_lock<std::mutex> l(qmu); qcv.wait(l, [&] { return !q.empty() || !run_; }); if (!run_ && q.empty()) return; fd = q.front(); q.pop(); }
            serve(fd); net_close(fd);
        }
    });
    acceptor_ = std::thread([this] { acceptLoop(); });
}
void Httpd::stop() {
    if (!run_) return; run_ = false;
    if (lfd_ >= 0) { net_shutdown(lfd_); net_close(lfd_); lfd_ = -1; }
    qcv.notify_all();
    if (acceptor_.joinable()) acceptor_.join();
    for (auto& t : workers_) if (t.joinable()) t.join();
}
void Httpd::acceptLoop() {
    while (run_) {
        sock_t fd = ::accept((sock_t)lfd_, nullptr, nullptr);
        if ((intptr_t)fd == MEOW_INVALID_SOCK) { if (!run_) return; continue; }
        net_set_timeout((intptr_t)fd, 30);
        int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof(one));
        { std::lock_guard<std::mutex> l(qmu); q.push((intptr_t)fd); } qcv.notify_one();
    }
}
void Httpd::serve(intptr_t fd) {
    std::string buf; char tmp[65536];
    size_t hdrEnd = std::string::npos;
    while (hdrEnd == std::string::npos) {
        long long n = net_recv(fd, tmp, sizeof(tmp)); if (n <= 0) return; buf.append(tmp, size_t(n));
        hdrEnd = buf.find("\r\n\r\n"); if (buf.size() > (1u << 20) && hdrEnd == std::string::npos) return;
    }
    std::string head = buf.substr(0, hdrEnd); HttpRequest req;
    size_t sp1 = head.find(' '), sp2 = head.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) return;
    req.method = head.substr(0, sp1); req.path = head.substr(sp1 + 1, sp2 - sp1 - 1);
    size_t clen = 0; std::string lower = head; for (auto& c : lower) c = char(tolower(c));
    size_t cl = lower.find("content-length:"); if (cl != std::string::npos) clen = std::stoul(lower.substr(cl + 15));
    if (clen > (4u << 20)) return;
    req.body = buf.substr(hdrEnd + 4);
    while (req.body.size() < clen) { long long n = net_recv(fd, tmp, sizeof(tmp)); if (n <= 0) return; req.body.append(tmp, size_t(n)); }
    req.body.resize(clen);
    HttpResponse r;
    try { r = handler_(req); } catch (const std::exception& e) { r.status = 500; r.body = std::string("{\"error\":\"") + e.what() + "\"}"; }
    const char* st = r.status == 200 ? "OK" : (r.status == 400 ? "Bad Request" : (r.status == 404 ? "Not Found" : "Error"));
    std::string out = "HTTP/1.1 " + std::to_string(r.status) + " " + st + "\r\nContent-Type: " + r.contentType + "\r\nContent-Length: " + std::to_string(r.body.size()) + "\r\nConnection: close\r\n\r\n" + r.body;
    size_t off = 0; while (off < out.size()) { long long n = net_send(fd, out.data() + off, out.size() - off); if (n <= 0) return; off += size_t(n); }
}
}  // namespace meow
