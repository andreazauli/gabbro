// src/platform/unified_memory.cpp - cudaMemGetInfo on a unified-memory GPU (DGX Spark / GB10).
//
// GB10 has no VRAM: the GPU allocates from the same LPDDR5x pool as the CPU, and the driver's cudaMemGetInfo reports
// as "free" only the pages the kernel has free right now. Reclaimable page cache (a model file just read or
// downloaded) is not counted, so after reading a 20 GB pack the card can look 2-5 GB "free" while 115 GB is
// available, and every memory plan in the engine refuses to start. cudaMalloc itself does reclaim the cache.
//
// This definition, linked into the executable, interposes the libcudart one for every call in the program (the
// executable's own symbol wins over the shared library's). It asks the real function and, on an integrated GPU,
// raises `free` to the kernel's MemAvailable, capped at the device total. On a discrete card it changes nothing.
#if !(defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) && defined(__linux__)
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

using MemGetInfoFn = cudaError_t (*)(size_t*, size_t*);

unsigned long long meminfo_bytes(const char* key) {
    unsigned long long kb = 0;
    const size_t kl = std::strlen(key);
    if (std::FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f))
            if (std::strncmp(line, key, kl) == 0) { kb = std::strtoull(line + kl, nullptr, 10); break; }
        std::fclose(f);
    }
    return kb * 1024ull;
}

// integrated per the driver, or a device whose total is the machine's RAM (within 1/8): one shared pool
bool integrated_device(size_t total) {
    int dev = 0, integrated = 0;
    if (cudaGetDevice(&dev) == cudaSuccess && cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev) == cudaSuccess &&
        integrated != 0)
        return true;
    const unsigned long long mt = meminfo_bytes("MemTotal:");
    const unsigned long long d = mt > total ? mt - total : total - mt;
    return mt > 0 && d < mt / 8;
}

}  // namespace

extern "C" cudaError_t cudaMemGetInfo(size_t* free_b, size_t* total_b) {
    static MemGetInfoFn real = (MemGetInfoFn) dlsym(RTLD_NEXT, "cudaMemGetInfo");
    if (real == nullptr) return cudaErrorUnknown;
    const cudaError_t e = real(free_b, total_b);
    if (e != cudaSuccess || free_b == nullptr || total_b == nullptr) return e;
    static const bool off = std::getenv("BASALT_UMA_FREE") != nullptr && std::strcmp(std::getenv("BASALT_UMA_FREE"), "0") == 0;
    if (off || !integrated_device(*total_b)) return e;
    const unsigned long long avail = meminfo_bytes("MemAvailable:");
    if (avail > *free_b) *free_b = (size_t) (avail < *total_b ? avail : *total_b);
    return e;
}
#endif
