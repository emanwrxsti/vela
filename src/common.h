#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    throw std::runtime_error(std::string("CUDA error ") + cudaGetErrorString(_e) + " at " #x " (" __FILE__ ":" + std::to_string(__LINE__) + ")"); } } while (0)

namespace meow {
inline std::string hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef"; std::string s; s.resize(2 * n);
    for (size_t i = 0; i < n; ++i) { s[2 * i] = d[p[i] >> 4]; s[2 * i + 1] = d[p[i] & 15]; }
    return s;
}
inline std::string hex(const std::vector<uint8_t>& v) { return hex(v.data(), v.size()); }
inline std::vector<uint8_t> unhex(const std::string& s) {
    if (s.size() % 2) throw std::runtime_error("odd hex length");
    std::vector<uint8_t> v(s.size() / 2);
    auto nib = [](char c) -> int { if (c >= '0' && c <= '9') return c - '0'; c |= 0x20; if (c >= 'a' && c <= 'f') return c - 'a' + 10; throw std::runtime_error("bad hex char"); };
    for (size_t i = 0; i < v.size(); ++i) v[i] = uint8_t((nib(s[2 * i]) << 4) | nib(s[2 * i + 1]));
    return v;
}
inline std::vector<uint8_t> rev(std::vector<uint8_t> v) { std::vector<uint8_t> r(v.rbegin(), v.rend()); return r; }
inline size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }
}  // namespace meow
