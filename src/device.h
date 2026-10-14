#pragma once
#include <string>
#include <vector>
namespace meow {
struct GpuInfo { int index; std::string name; int cc; size_t memBytes; int sms; };
std::vector<GpuInfo> listGpus();
/** NVML (dlopen'ed): temperature C, power W, utilisation %; -1 when unavailable. */
struct GpuTelemetry { int tempC = -1; int powerW = -1; int util = -1; int fan = -1; };
GpuTelemetry gpuTelemetry(int cudaIndex);
}  // namespace meow
