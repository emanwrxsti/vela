#include "device.h"
#include "common.h"
#include "compat.h"
#include <mutex>

namespace meow {
std::vector<GpuInfo> listGpus() {
    int n = 0; if (cudaGetDeviceCount(&n) != cudaSuccess) return {};
    std::vector<GpuInfo> v;
    for (int i = 0; i < n; ++i) { cudaDeviceProp p{}; cudaGetDeviceProperties(&p, i); v.push_back({i, p.name, p.major * 10 + p.minor, p.totalGlobalMem, p.multiProcessorCount}); }
    return v;
}
namespace {
typedef int (*nvmlInit_t)(); typedef int (*nvmlGetByPci_t)(const char*, void**); typedef int (*nvmlTemp_t)(void*, int, unsigned*); typedef int (*nvmlPower_t)(void*, unsigned*);
typedef int (*nvmlUtil_t)(void*, unsigned*); typedef int (*nvmlFan_t)(void*, unsigned*);
struct Nvml { void* h = nullptr; nvmlGetByPci_t byPci = nullptr; nvmlTemp_t temp = nullptr; nvmlPower_t power = nullptr; nvmlUtil_t util = nullptr; nvmlFan_t fan = nullptr; bool ok = false; };
Nvml& nvml() {
    static Nvml n; static std::once_flag once;
    std::call_once(once, [] {
        n.h = dl_open(nvml_lib()); if (!n.h) return;
        auto init = (nvmlInit_t)dl_sym(n.h, "nvmlInit_v2"); if (!init || init() != 0) return;
        n.byPci = (nvmlGetByPci_t)dl_sym(n.h, "nvmlDeviceGetHandleByPciBusId_v2"); n.temp = (nvmlTemp_t)dl_sym(n.h, "nvmlDeviceGetTemperature");
        n.power = (nvmlPower_t)dl_sym(n.h, "nvmlDeviceGetPowerUsage"); n.util = (nvmlUtil_t)dl_sym(n.h, "nvmlDeviceGetUtilizationRates"); n.fan = (nvmlFan_t)dl_sym(n.h, "nvmlDeviceGetFanSpeed");
        n.ok = n.byPci && n.temp && n.power;
    });
    return n;
}
}  // namespace
GpuTelemetry gpuTelemetry(int idx) {
    GpuTelemetry t; Nvml& n = nvml(); if (!n.ok) return t;
    char pci[32]; if (cudaDeviceGetPCIBusId(pci, sizeof(pci), idx) != cudaSuccess) return t;
    void* dev = nullptr; if (n.byPci(pci, &dev) != 0) return t;
    unsigned v = 0; if (n.temp(dev, 0, &v) == 0) t.tempC = int(v);
    if (n.power(dev, &v) == 0) t.powerW = int(v / 1000);
    if (n.util) { unsigned u[2] = {0, 0}; if (n.util(dev, u) == 0) t.util = int(u[0]); }
    if (n.fan && n.fan(dev, &v) == 0) t.fan = int(v);
    return t;
}
}  // namespace meow
