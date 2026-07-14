#include <mc_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "fused_moe_i8_tn_runner.h"

namespace {

constexpr int kNumExperts = fused_moe_i8_tn::kDefaultNumExperts;
constexpr int kTileM = fused_moe_i8_tn::kDefaultTileM;
constexpr int kN = fused_moe_i8_tn::kDefaultN;
constexpr int kK = fused_moe_i8_tn::kDefaultK;
constexpr int kDefaultWarmupIterations = 20;
constexpr int kDefaultMeasuredIterations = 100;

struct CaseConfig {
    const char *tag;
    int num_tokens;
    int topk;
    int em;
    std::vector<int> tile_experts;
};

struct BenchmarkConfig {
    int warmup_iterations = 0;
    int measured_iterations = 0;
};

struct BenchmarkResult {
    float avg_ms = 0.0f;
    double tops = 0.0;
    int warmup_iterations = 0;
    int measured_iterations = 0;
};

void check_mc(mcError_t status, const char *expr) {
    if (status != mcSuccess) {
        std::cerr << expr << " failed: " << mcGetErrorString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

int read_env_int(const char *name, int default_value) {
    const char *value = std::getenv(name);
    if (!value || !value[0]) {
        return default_value;
    }

    char *end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || (end && *end != '\0') || parsed < 0) {
        std::cerr << "Ignoring invalid " << name << '=' << value
                  << ", using default " << default_value << '\n';
        return default_value;
    }
    return static_cast<int>(parsed);
}

BenchmarkConfig make_benchmark_config() {
    BenchmarkConfig config;
    config.warmup_iterations = read_env_int("MCTLASS_MOE_WARMUP", kDefaultWarmupIterations);
    config.measured_iterations = read_env_int("MCTLASS_MOE_ITERS", kDefaultMeasuredIterations);
    if (config.measured_iterations <= 0) {
        std::cerr << "MCTLASS_MOE_ITERS must be > 0\n";
        std::exit(EXIT_FAILURE);
    }
    return config;
}

double compute_tops(int m, int n, int k, float avg_ms) {
    if (avg_ms <= 0.0f) {
        return 0.0;
    }
    const double operations = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    return operations / (static_cast<double>(avg_ms) * 1.0e9);
}

void fill_inputs(const CaseConfig &cfg,
                 std::vector<int8_t> &a,
                 std::vector<int8_t> &b_col_major,
                 std::vector<float> &scale_a,
                 std::vector<float> &scale_b,
                 std::vector<float> &moe_weights,
                 std::vector<int> &token_ids,
                 std::vector<int> &expert_ids) {
    const int total_rows = cfg.num_tokens * cfg.topk;

    a.resize(static_cast<size_t>(cfg.num_tokens) * kK);
    scale_a.resize(cfg.num_tokens);
    moe_weights.resize(total_rows);
    token_ids.resize(total_rows);
    expert_ids = cfg.tile_experts;

    for (int row = 0; row < cfg.num_tokens; ++row) {
        for (int kk = 0; kk < kK; ++kk) {
            a[static_cast<size_t>(row) * kK + kk] =
                static_cast<int8_t>(((row * 13 + kk * 7 + cfg.topk * 5 + 3) % 11) - 5);
        }
        scale_a[row] = 0.125f + 0.015625f * static_cast<float>((row + cfg.topk) % 7);
    }

    for (int routed_row = 0; routed_row < total_rows; ++routed_row) {
        token_ids[routed_row] = routed_row;
        moe_weights[routed_row] = 0.5f + 0.03125f * static_cast<float>((routed_row + cfg.topk) % 5);
    }

    b_col_major.resize(static_cast<size_t>(kNumExperts) * kN * kK);
    scale_b.resize(static_cast<size_t>(kNumExperts) * kN);
    for (int expert = 0; expert < kNumExperts; ++expert) {
        for (int col = 0; col < kN; ++col) {
            scale_b[static_cast<size_t>(expert) * kN + col] =
                0.25f + 0.03125f * static_cast<float>((expert * 3 + col + cfg.topk) % 9);
            for (int kk = 0; kk < kK; ++kk) {
                const int value = ((expert * 17 + col * 5 + kk * 3 + cfg.topk) % 9) - 4;
                b_col_major[(static_cast<size_t>(expert) * kN + col) * kK + kk] =
                    static_cast<int8_t>(value);
            }
        }
    }
}

std::vector<fused_moe_i8_tn::BFloat16> reference_fused_moe(const CaseConfig &cfg,
                                                           const std::vector<int8_t> &a,
                                                           const std::vector<int8_t> &b_col_major,
                                                           const std::vector<float> &scale_a,
                                                           const std::vector<float> &scale_b,
                                                           const std::vector<float> &moe_weights,
                                                           const std::vector<int> &token_ids,
                                                           const std::vector<int> &expert_ids) {
    const int total_rows = cfg.num_tokens * cfg.topk;
    std::vector<fused_moe_i8_tn::BFloat16> out(static_cast<size_t>(total_rows) * kN,
                                               fused_moe_i8_tn::float_to_bf16(0.0f));

    for (int routed_row = 0; routed_row < total_rows; ++routed_row) {
        const int token = token_ids[routed_row] / cfg.topk;
        const int tile_idx = routed_row / kTileM;
        const int expert = expert_ids[tile_idx];
        const float row_scale = scale_a[token] * moe_weights[routed_row];

        for (int col = 0; col < kN; ++col) {
            int32_t acc = 0;
            for (int kk = 0; kk < kK; ++kk) {
                const int32_t lhs = static_cast<int32_t>(a[static_cast<size_t>(token) * kK + kk]);
                const int32_t rhs =
                    static_cast<int32_t>(b_col_major[(static_cast<size_t>(expert) * kN + col) * kK + kk]);
                acc += lhs * rhs;
            }
            const float scaled = static_cast<float>(acc) * row_scale *
                                 scale_b[static_cast<size_t>(expert) * kN + col];
            out[static_cast<size_t>(routed_row) * kN + col] = fused_moe_i8_tn::float_to_bf16(scaled);
        }
    }

    return out;
}

bool validate_result(const std::vector<fused_moe_i8_tn::BFloat16> &got,
                     const std::vector<fused_moe_i8_tn::BFloat16> &expected,
                     const CaseConfig &cfg) {
    size_t mismatch_count = 0;
    size_t first_bad = 0;
    float max_abs = 0.0f;
    const int total_rows = cfg.num_tokens * cfg.topk;

    for (size_t i = 0; i < got.size(); ++i) {
        const float got_f = fused_moe_i8_tn::bf16_to_float(got[i]);
        const float exp_f = fused_moe_i8_tn::bf16_to_float(expected[i]);
        const float abs_err = std::fabs(got_f - exp_f);
        max_abs = std::max(max_abs, abs_err);
        if (abs_err > 1e-2f) {
            if (mismatch_count == 0) {
                first_bad = i;
            }
            ++mismatch_count;
        }
    }

    if (mismatch_count != 0) {
        const int row = static_cast<int>(first_bad / kN);
        const int col = static_cast<int>(first_bad % kN);
        std::cerr << cfg.tag << " failed"
                  << ": mismatches=" << mismatch_count
                  << ", first mismatch at (" << row << ", " << col << ")"
                  << ", got=" << fused_moe_i8_tn::bf16_to_float(got[first_bad])
                  << ", expected=" << fused_moe_i8_tn::bf16_to_float(expected[first_bad])
                  << ", max_abs=" << max_abs << '\n';
        return false;
    }

    std::cout << cfg.tag
              << " passed"
              << ": rows=" << total_rows
              << ", topk=" << cfg.topk
              << ", N=" << kN
              << ", K=" << kK
              << ", sample C[0]=" << fused_moe_i8_tn::bf16_to_float(got.front())
              << ", C[last]=" << fused_moe_i8_tn::bf16_to_float(got.back())
              << ", max_abs=" << max_abs << '\n';
    return true;
}

template <typename LaunchFn>
BenchmarkResult run_benchmark(LaunchFn &&launch, int m, int n, int k, const BenchmarkConfig &config) {
    for (int iter = 0; iter < config.warmup_iterations; ++iter) {
        launch();
    }
    check_mc(mcDeviceSynchronize(), "mcDeviceSynchronize(warmup)");

    mcEvent_t start;
    mcEvent_t stop;
    check_mc(mcEventCreate(&start), "mcEventCreate(start)");
    check_mc(mcEventCreate(&stop), "mcEventCreate(stop)");

    check_mc(mcEventRecord(start, nullptr), "mcEventRecord(start)");
    for (int iter = 0; iter < config.measured_iterations; ++iter) {
        launch();
    }
    check_mc(mcEventRecord(stop, nullptr), "mcEventRecord(stop)");
    check_mc(mcEventSynchronize(stop), "mcEventSynchronize(stop)");

    float elapsed_ms = 0.0f;
    check_mc(mcEventElapsedTime(&elapsed_ms, start, stop), "mcEventElapsedTime");
    check_mc(mcEventDestroy(start), "mcEventDestroy(start)");
    check_mc(mcEventDestroy(stop), "mcEventDestroy(stop)");

    BenchmarkResult result;
    result.warmup_iterations = config.warmup_iterations;
    result.measured_iterations = config.measured_iterations;
    result.avg_ms = elapsed_ms / static_cast<float>(config.measured_iterations);
    result.tops = compute_tops(m, n, k, result.avg_ms);
    return result;
}

bool run_case(const CaseConfig &cfg, const BenchmarkConfig &benchmark_config) {
    std::vector<int8_t> host_a;
    std::vector<int8_t> host_b;
    std::vector<float> host_scale_a;
    std::vector<float> host_scale_b;
    std::vector<float> host_moe_weights;
    std::vector<int> host_token_ids;
    std::vector<int> host_expert_ids;

    fill_inputs(cfg,
                host_a,
                host_b,
                host_scale_a,
                host_scale_b,
                host_moe_weights,
                host_token_ids,
                host_expert_ids);

    const std::vector<fused_moe_i8_tn::BFloat16> expected = reference_fused_moe(
        cfg, host_a, host_b, host_scale_a, host_scale_b, host_moe_weights, host_token_ids, host_expert_ids);

    fused_moe_i8_tn::HostInputs inputs;
    inputs.num_tokens = cfg.num_tokens;
    inputs.topk = cfg.topk;
    inputs.em = cfg.em;
    inputs.num_experts = kNumExperts;
    inputs.n = kN;
    inputs.k = kK;
    inputs.a = host_a;
    inputs.b_col_major = host_b;
    inputs.scale_a = host_scale_a;
    inputs.scale_b = host_scale_b;
    inputs.moe_weights = host_moe_weights;
    inputs.token_ids = host_token_ids;
    inputs.expert_ids = host_expert_ids;

    const fused_moe_i8_tn::RunResult run_result = fused_moe_i8_tn::run_fused_moe_i8_tn(inputs);
    const bool valid = validate_result(run_result.output, expected, cfg);
    if (!valid) {
        return false;
    }

    const BenchmarkResult benchmark = run_benchmark([&]() { fused_moe_i8_tn::run_fused_moe_i8_tn(inputs); },
                                                    cfg.em,
                                                    kN,
                                                    kK,
                                                    benchmark_config);
    std::cout << cfg.tag
              << " benchmark"
              << ": avg_ms=" << benchmark.avg_ms
              << ", TOPS=" << benchmark.tops
              << ", warmup=" << benchmark.warmup_iterations
              << ", iters=" << benchmark.measured_iterations
              << '\n';
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    check_mc(mcGetDeviceCount(&device_count), "mcGetDeviceCount");
    if (device_count <= 0) {
        std::cerr << "No MACA device is visible.\n";
        return EXIT_FAILURE;
    }
    check_mc(mcSetDevice(0), "mcSetDevice");

    const BenchmarkConfig benchmark_config = make_benchmark_config();
    std::cout << "Benchmark config"
              << ": warmup=" << benchmark_config.warmup_iterations
              << ", iters=" << benchmark_config.measured_iterations
              << '\n';

    const std::vector<CaseConfig> cases = {
        {"fused_moe_i8_tn_topk1", 256, 1, 256, {0, 1}},
        {"fused_moe_i8_tn_topk2", 256, 2, 512, {0, 1, 1, 0}},
        {"fused_moe_i8_tn_topk3", 128, 3, 384, {0, 1, 0}},
    };

    bool ok = true;
    for (const CaseConfig &cfg : cases) {
        ok &= run_case(cfg, benchmark_config);
    }
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
