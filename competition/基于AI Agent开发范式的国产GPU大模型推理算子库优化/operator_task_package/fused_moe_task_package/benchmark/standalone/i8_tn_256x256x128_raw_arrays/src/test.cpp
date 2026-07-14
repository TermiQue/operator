#include <mc_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../include/gemm_i8_tn_256x256x128_raw_arrays.hpp"

namespace {

using namespace standalone_i8_tn_256x256x128_raw_arrays;

struct BenchmarkResult {
    float avg_ms = 0.0f;
    double tflops = 0.0;
    int warmup_iterations = 0;
    int measured_iterations = 0;
};

void check_mc(mcError_t status, const char *expr) {
    if (status != mcSuccess) {
        std::cerr << expr << " failed: " << mcGetErrorString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void fill_row_major_a(std::vector<int8_t> &a, int m, int k) {
    a.resize(static_cast<size_t>(m) * k);
    for (int row = 0; row < m; ++row) {
        for (int kk = 0; kk < k; ++kk) {
            a[static_cast<size_t>(row) * k + kk] =
                static_cast<int8_t>(((row * 13 + kk * 7 + 5) % 9) - 4);
        }
    }
}

void fill_col_major_b(std::vector<int8_t> &b, int n, int k) {
    b.resize(static_cast<size_t>(n) * k);
    for (int col = 0; col < n; ++col) {
        for (int kk = 0; kk < k; ++kk) {
            b[static_cast<size_t>(col) * k + kk] =
                static_cast<int8_t>(((kk * 11 + col * 5 + 3) % 7) - 3);
        }
    }
}

std::vector<int32_t> reference_gemm_tn(const std::vector<int8_t> &a,
                                       const std::vector<int8_t> &b_col_major,
                                       int m,
                                       int n,
                                       int k) {
    std::vector<int32_t> out(static_cast<size_t>(m) * n, 0);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            int32_t acc = 0;
            for (int kk = 0; kk < k; ++kk) {
                acc += static_cast<int32_t>(a[static_cast<size_t>(row) * k + kk]) *
                       static_cast<int32_t>(b_col_major[static_cast<size_t>(col) * k + kk]);
            }
            out[static_cast<size_t>(row) * n + col] = acc;
        }
    }
    return out;
}

bool run_case(int m, int n, int k, const char *tag) {
    std::vector<int8_t> host_a;
    std::vector<int8_t> host_b;
    std::vector<int32_t> host_d(static_cast<size_t>(m) * n, -1);
    fill_row_major_a(host_a, m, k);
    fill_col_major_b(host_b, n, k);
    const std::vector<int32_t> reference = reference_gemm_tn(host_a, host_b, m, n, k);

    int8_t *dev_a = nullptr;
    int8_t *dev_b = nullptr;
    int32_t *dev_d = nullptr;
    check_mc(mcMalloc(reinterpret_cast<void **>(&dev_a), host_a.size() * sizeof(int8_t)), "mcMalloc(dev_a)");
    check_mc(mcMalloc(reinterpret_cast<void **>(&dev_b), host_b.size() * sizeof(int8_t)), "mcMalloc(dev_b)");
    check_mc(mcMalloc(reinterpret_cast<void **>(&dev_d), host_d.size() * sizeof(int32_t)), "mcMalloc(dev_d)");

    check_mc(mcMemcpy(dev_a, host_a.data(), host_a.size() * sizeof(int8_t), mcMemcpyHostToDevice), "mcMemcpy(dev_a)");
    check_mc(mcMemcpy(dev_b, host_b.data(), host_b.size() * sizeof(int8_t), mcMemcpyHostToDevice), "mcMemcpy(dev_b)");
    check_mc(mcMemset(dev_d, 0, host_d.size() * sizeof(int32_t)), "mcMemset(dev_d)");

    dim3 grid((m + KernelConfig::kTileM - 1) / KernelConfig::kTileM,
              (n + KernelConfig::kTileN - 1) / KernelConfig::kTileN,
              1);
    gemm_i8_tn_256x256x128_raw_arrays_kernel<<<grid, KernelConfig::kThreadCount>>>(dev_a, dev_b, dev_d, m, n, k);
    check_mc(mcDeviceSynchronize(), "mcDeviceSynchronize");
    check_mc(mcGetLastError(), "mcGetLastError");

    check_mc(mcMemcpy(host_d.data(), dev_d, host_d.size() * sizeof(int32_t), mcMemcpyDeviceToHost), "mcMemcpy(host_d)");

    check_mc(mcFree(dev_a), "mcFree(dev_a)");
    check_mc(mcFree(dev_b), "mcFree(dev_b)");
    check_mc(mcFree(dev_d), "mcFree(dev_d)");

    size_t mismatch_count = 0;
    size_t first_bad = 0;
    for (size_t i = 0; i < host_d.size(); ++i) {
        if (host_d[i] != reference[i]) {
            if (mismatch_count == 0) {
                first_bad = i;
            }
            ++mismatch_count;
        }
    }

    if (mismatch_count != 0) {
        const int row = static_cast<int>(first_bad / n);
        const int col = static_cast<int>(first_bad % n);
        std::cerr << tag << " validation failed. mismatches=" << mismatch_count
                  << ", first mismatch at (" << row << ", " << col << ")"
                  << ", got=" << host_d[first_bad]
                  << ", expected=" << reference[first_bad] << '\n';
        return false;
    }

    std::cout << tag << " passed: M=" << m
              << ", N=" << n
              << ", K=" << k
              << ", sample D[0]=" << host_d[0]
              << ", D[last]=" << host_d.back() << '\n';
    return true;
}

double compute_tflops(int m, int n, int k, float avg_ms) {
    if (avg_ms <= 0.0f) {
        return 0.0;
    }
    const double operations = 2.0 * static_cast<double>(m) * n * k;
    return operations / (static_cast<double>(avg_ms) * 1.0e9);
}

BenchmarkResult run_benchmark(int m, int n, int k, int warmup_iterations, int measured_iterations) {
    std::vector<int8_t> host_a;
    std::vector<int8_t> host_b;
    std::vector<int32_t> host_d(static_cast<size_t>(m) * n, 0);
    fill_row_major_a(host_a, m, k);
    fill_col_major_b(host_b, n, k);

    int8_t *dev_a = nullptr;
    int8_t *dev_b = nullptr;
    int32_t *dev_d = nullptr;
    check_mc(mcMalloc(reinterpret_cast<void **>(&dev_a), host_a.size() * sizeof(int8_t)), "mcMalloc(dev_a)");
    check_mc(mcMalloc(reinterpret_cast<void **>(&dev_b), host_b.size() * sizeof(int8_t)), "mcMalloc(dev_b)");
    check_mc(mcMalloc(reinterpret_cast<void **>(&dev_d), host_d.size() * sizeof(int32_t)), "mcMalloc(dev_d)");

    check_mc(mcMemcpy(dev_a, host_a.data(), host_a.size() * sizeof(int8_t), mcMemcpyHostToDevice), "mcMemcpy(dev_a)");
    check_mc(mcMemcpy(dev_b, host_b.data(), host_b.size() * sizeof(int8_t), mcMemcpyHostToDevice), "mcMemcpy(dev_b)");
    check_mc(mcMemset(dev_d, 0, host_d.size() * sizeof(int32_t)), "mcMemset(dev_d)");

    dim3 grid((m + KernelConfig::kTileM - 1) / KernelConfig::kTileM,
              (n + KernelConfig::kTileN - 1) / KernelConfig::kTileN,
              1);

    for (int iter = 0; iter < warmup_iterations; ++iter) {
        gemm_i8_tn_256x256x128_raw_arrays_kernel<<<grid, KernelConfig::kThreadCount>>>(dev_a, dev_b, dev_d, m, n, k);
    }
    check_mc(mcDeviceSynchronize(), "mcDeviceSynchronize(warmup)");

    mcEvent_t start;
    mcEvent_t stop;
    check_mc(mcEventCreate(&start), "mcEventCreate(start)");
    check_mc(mcEventCreate(&stop), "mcEventCreate(stop)");
    check_mc(mcEventRecord(start), "mcEventRecord(start)");
    for (int iter = 0; iter < measured_iterations; ++iter) {
        gemm_i8_tn_256x256x128_raw_arrays_kernel<<<grid, KernelConfig::kThreadCount>>>(dev_a, dev_b, dev_d, m, n, k);
    }
    check_mc(mcEventRecord(stop), "mcEventRecord(stop)");
    check_mc(mcEventSynchronize(stop), "mcEventSynchronize(stop)");
    check_mc(mcGetLastError(), "mcGetLastError");

    float elapsed_ms = 0.0f;
    check_mc(mcEventElapsedTime(&elapsed_ms, start, stop), "mcEventElapsedTime");
    check_mc(mcEventDestroy(start), "mcEventDestroy(start)");
    check_mc(mcEventDestroy(stop), "mcEventDestroy(stop)");

    check_mc(mcFree(dev_a), "mcFree(dev_a)");
    check_mc(mcFree(dev_b), "mcFree(dev_b)");
    check_mc(mcFree(dev_d), "mcFree(dev_d)");

    BenchmarkResult result;
    result.warmup_iterations = warmup_iterations;
    result.measured_iterations = measured_iterations;
    result.avg_ms = elapsed_ms / static_cast<float>(measured_iterations);
    result.tflops = compute_tflops(m, n, k, result.avg_ms);
    return result;
}

}  // namespace

int main() {
    constexpr int kExactM = 2048;
    constexpr int kExactN = 2048;
    constexpr int kExactK = 2048;
    constexpr int kBenchM = 2048;
    constexpr int kBenchN = 2048;
    constexpr int kBenchK = 2048;
    constexpr int kWarmupIterations = 3;
    constexpr int kMeasuredIterations = 10;

    int device_count = 0;
    check_mc(mcGetDeviceCount(&device_count), "mcGetDeviceCount");
    if (device_count <= 0) {
        std::cerr << "No MACA device is visible.\n";
        return EXIT_FAILURE;
    }
    check_mc(mcSetDevice(0), "mcSetDevice");

    if (!run_case(kExactM, kExactN, kExactK, "standalone_i8_tn_256x256x128_raw_arrays_exact")) {
        return EXIT_FAILURE;
    }

    const BenchmarkResult benchmark =
        run_benchmark(kBenchM, kBenchN, kBenchK, kWarmupIterations, kMeasuredIterations);
    std::cout << std::fixed << std::setprecision(3)
              << "standalone_i8_tn_256x256x128_raw_arrays benchmark: M=" << kBenchM
              << ", N=" << kBenchN
              << ", K=" << kBenchK
              << ", avg_ms=" << benchmark.avg_ms
              << ", TFLOPS=" << benchmark.tflops
              << ", warmup=" << benchmark.warmup_iterations
              << ", iters=" << benchmark.measured_iterations << '\n';

    return EXIT_SUCCESS;
}
