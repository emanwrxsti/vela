// Cross-platform shim: BSD sockets vs Winsock, dlopen vs LoadLibrary, and a
// portable 128/64 long-division step (MSVC has no __int128). Host code only.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

#if defined(_WIN32)
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #include <windows.h>
  #include <intrin.h>
  #ifndef MSG_NOSIGNAL
    #define MSG_NOSIGNAL 0
  #endif
  #ifndef SHUT_RDWR
    #define SHUT_RDWR SD_BOTH
  #endif
#else
  #include <arpa/inet.h>
  #include <netdb.h>
  #include <netinet/in.h>
  #include <netinet/tcp.h>
  #include <sys/socket.h>
  #include <unistd.h>
  #include <dlfcn.h>
  #include <cerrno>
  #include <cstring>
#endif

namespace meow {

#if defined(_WIN32)
using sock_t = SOCKET;
static const intptr_t MEOW_INVALID_SOCK = (intptr_t)INVALID_SOCKET;   // == -1

// One-time Winsock init; a static instance brings the stack up before main's networking.
struct NetInit { NetInit() { WSADATA w; WSAStartup(MAKEWORD(2, 2), &w); } ~NetInit() { WSACleanup(); } };
inline void net_global_init() { static NetInit g; }

inline int  net_close(intptr_t fd)        { return ::closesocket((sock_t)fd); }
inline void net_shutdown(intptr_t fd)     { ::shutdown((sock_t)fd, SHUT_RDWR); }
inline int  net_errno()                   { return ::WSAGetLastError(); }
inline bool net_would_block()             { int e = ::WSAGetLastError(); return e == WSAEWOULDBLOCK || e == WSAETIMEDOUT; }
inline std::string net_strerror()         { return "winsock error " + std::to_string(::WSAGetLastError()); }
// SO_RCVTIMEO / SO_SNDTIMEO take a DWORD of milliseconds on Windows.
inline void net_set_timeout(intptr_t fd, int seconds) {
    DWORD ms = (DWORD)seconds * 1000;
    ::setsockopt((sock_t)fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&ms, sizeof(ms));
    ::setsockopt((sock_t)fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&ms, sizeof(ms));
}
inline void* dl_open(const char* name)    { return (void*)::LoadLibraryA(name); }
inline void* dl_sym(void* h, const char* s){ return h ? (void*)::GetProcAddress((HMODULE)h, s) : nullptr; }
inline const char* nvml_lib()             { return "nvml.dll"; }

#else
using sock_t = int;
static const intptr_t MEOW_INVALID_SOCK = -1;

inline void net_global_init() {}
inline int  net_close(intptr_t fd)        { return ::close((int)fd); }
inline void net_shutdown(intptr_t fd)     { ::shutdown((int)fd, SHUT_RDWR); }
inline int  net_errno()                   { return errno; }
inline bool net_would_block()             { return errno == EAGAIN || errno == EWOULDBLOCK; }
inline std::string net_strerror()         { return std::strerror(errno); }
inline void net_set_timeout(intptr_t fd, int seconds) {
    struct timeval tv{seconds, 0};
    ::setsockopt((int)fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt((int)fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}
inline void* dl_open(const char* name)    { return ::dlopen(name, RTLD_NOW); }
inline void* dl_sym(void* h, const char* s){ return h ? ::dlsym(h, s) : nullptr; }
inline const char* nvml_lib()             { return "libnvidia-ml.so.1"; }
#endif

// send/recv normalised: Windows wants char* and int length and has no MSG_NOSIGNAL semantics.
inline long long net_send(intptr_t fd, const void* b, size_t n) {
    return (long long)::send((sock_t)fd, (const char*)b, (int)n, MSG_NOSIGNAL);
}
inline long long net_recv(intptr_t fd, void* b, size_t n) {
    return (long long)::recv((sock_t)fd, (char*)b, (int)n, 0);
}

// One step of long-dividing 2^256-1 by d, most-significant byte first:
// quotient byte = ((rem<<8)|0xff) / d, rem updated to the remainder (always < d).
inline uint8_t div256_step(uint64_t& rem, uint64_t d) {
#if defined(_MSC_VER)
    uint64_t high = rem >> 56;
    uint64_t low  = (rem << 8) | 0xffULL;
    uint64_t r = 0;
    uint64_t q = _udiv128(high, low, d, &r);   // high < d is guaranteed since rem < d
    rem = r;
    return (uint8_t)q;
#else
    unsigned __int128 cur = ((unsigned __int128)rem << 8) | 0xff;
    uint8_t q = (uint8_t)(cur / d);
    rem = (uint64_t)(cur % d);
    return q;
#endif
}

}  // namespace meow
