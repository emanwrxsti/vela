// CPU-only self-test for the cross-platform shim (no GPU needed; runs in CI).
//  1. share-target long division (exercises the MSVC _udiv128 path in compat.h)
//  2. Winsock/BSD-socket loopback round-trip through net_send/net_recv
#include "compat.h"
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

using meow::sock_t;
using meow::MEOW_INVALID_SOCK;

// Same algorithm as meow::targetFromDifficulty, built on the portable div256_step.
static std::array<uint8_t, 32> target_from_diff(double difficulty) {
    std::array<uint8_t, 32> t{};
    if (!(difficulty >= 1)) { t.fill(0xff); return t; }
    uint64_t d = (uint64_t)std::ceil(difficulty);
    uint64_t rem = 0;
    for (int i = 31; i >= 0; --i) t[i] = meow::div256_step(rem, d);
    return t;
}

struct Vec { uint64_t d; uint8_t t[32]; };
static const Vec VECS[] = {
  {2048ULL, {0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0x1f,0x00}},
  {1000003ULL, {0x9d,0x13,0x7d,0xfe,0xaf,0x43,0xf9,0x2c,0xab,0x45,0x50,0x0c,0xb4,0x4f,0xc6,0x00,0x3a,0xbb,0x0c,0x94,0xe5,0x8d,0xc6,0x59,0xcb,0x49,0x54,0xf4,0xc6,0x10,0x00,0x00}},
  {1152921504606846976ULL, {0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0x0f,0x00,0x00,0x00,0x00,0x00,0x00,0x00}},
  {18014398509481984ULL, {0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0x03,0x00,0x00,0x00,0x00,0x00,0x00}},
};

static int test_target() {
    int fail = 0;
    for (const auto& v : VECS) {
        auto got = target_from_diff((double)v.d);
        if (std::memcmp(got.data(), v.t, 32) != 0) { printf("  target FAIL d=%llu\n", (unsigned long long)v.d); fail++; }
    }
    printf("  target math: %s (%d vectors)\n", fail ? "FAIL" : "OK", (int)(sizeof(VECS) / sizeof(VECS[0])));
    return fail;
}

static int test_loopback() {
    meow::net_global_init();
    sock_t ls = ::socket(AF_INET, SOCK_STREAM, 0);
    if ((intptr_t)ls == MEOW_INVALID_SOCK) { printf("  socket FAIL\n"); return 1; }
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = 0;
    if (::bind(ls, (sockaddr*)&a, sizeof(a)) != 0) { printf("  bind FAIL\n"); return 1; }
    socklen_t al = sizeof(a);
    if (::getsockname(ls, (sockaddr*)&a, &al) != 0) { printf("  getsockname FAIL\n"); return 1; }
    uint16_t port = ntohs(a.sin_port);
    if (::listen(ls, 1) != 0) { printf("  listen FAIL\n"); return 1; }

    std::atomic<int> srv_ok{-1};
    std::thread srv([&] {
        sock_t c = ::accept(ls, nullptr, nullptr);
        if ((intptr_t)c == MEOW_INVALID_SOCK) { srv_ok = 0; return; }
        char buf[64] = {0};
        long long n = meow::net_recv((intptr_t)c, buf, sizeof(buf) - 1);
        bool got_ping = (n > 0 && std::string(buf, (size_t)n).rfind("ping", 0) == 0);
        meow::net_send((intptr_t)c, "pong\n", 5);
        meow::net_close((intptr_t)c);
        srv_ok = got_ping ? 1 : 0;
    });

    sock_t cl = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in sa{}; sa.sin_family = AF_INET; sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK); sa.sin_port = htons(port);
    meow::net_set_timeout((intptr_t)cl, 5);
    int fail = 0;
    if (::connect(cl, (sockaddr*)&sa, sizeof(sa)) != 0) { printf("  connect FAIL\n"); fail = 1; }
    else {
        meow::net_send((intptr_t)cl, "ping\n", 5);
        char buf[64] = {0};
        long long n = meow::net_recv((intptr_t)cl, buf, sizeof(buf) - 1);
        if (!(n > 0 && std::string(buf, (size_t)n).rfind("pong", 0) == 0)) { printf("  client recv FAIL\n"); fail = 1; }
    }
    meow::net_close((intptr_t)cl);
    srv.join();
    meow::net_close((intptr_t)ls);
    if (srv_ok != 1) { printf("  server side FAIL\n"); fail = 1; }
    printf("  socket loopback: %s\n", fail ? "FAIL" : "OK");
    return fail;
}

int main() {
    printf("supr-meow-vela compat self-test\n");
    int fail = test_target() + test_loopback();
    printf("%s\n", fail ? "SELFTEST FAILED" : "SELFTEST OK");
    return fail ? 1 : 0;
}
