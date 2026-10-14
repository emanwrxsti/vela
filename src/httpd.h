// Minimal HTTP/1.1 server (POST /verify, GET /health) for --verify-server mode.
#pragma once
#include <functional>
#include <string>
#include <thread>
#include <atomic>

namespace meow {
struct HttpRequest { std::string method, path, body; };
struct HttpResponse { int status = 200; std::string contentType = "application/json"; std::string body; };
class Httpd {
public:
    Httpd(std::string host, int port, std::function<HttpResponse(const HttpRequest&)> handler, int threads = 4);
    ~Httpd();
    void start();
    void stop();
private:
    void acceptLoop();
    void serve(int fd);
    std::string host_; int port_; std::function<HttpResponse(const HttpRequest&)> handler_;
    int threads_; int lfd_ = -1; std::vector<std::thread> workers_; std::thread acceptor_; std::atomic<bool> run_{false};
};
}  // namespace meow
