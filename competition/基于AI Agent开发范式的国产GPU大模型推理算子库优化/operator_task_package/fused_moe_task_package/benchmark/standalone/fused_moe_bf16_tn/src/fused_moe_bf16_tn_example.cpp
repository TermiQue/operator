#include <mc_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "mctlass/bfloat16.h"
#include "mctlass/frontend_op/gemm_config.h"
#include "mctlass/frontend_op/mctlass_moe_gemm.h"

namespace {

using Bf16 = maca_bfloat16;
using LayoutA = mctlass::layout::RowMajor;
using LayoutB = mctlass::layout::ColumnMajor;
using LayoutC = mctlass::layout::RowMajor;
using GemmOp = mctlassMoeGemm<Bf16, LayoutA, Bf16, LayoutB, Bf16, LayoutC, float>;

constexpr int kNumExperts = 2;
constexpr int kNumTokens = 256;
constexpr int kTopK = 1;
constexpr int kEM = 256;
constexpr int kN = 128;
constexpr int kK = 64;
constexpr int kTileM = 128;

void check_mc(mcError_t status, const char *expr) {
    if (status != mcSuccess) {
        std::cerr << expr << " failed: " << mcGetErrorString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void check_mctlass(mctlass::Status status, const char *expr) {
    if (status != mctlass::Status::kSuccess) {
        std::cerr << expr << " failed: " << mctlass::mctlassGetStatusString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

Bf16 float_to_bf16(float value) {
    return mctlass::bfloat16_t(value).to_bfloat();
}

float bf16_to_float(Bf16 value) {
    return static_cast<float>(mctlass::bfloat16_t(value));
}

void fill_inputs(std::vector<Bf16> &a,
                 std::vector<Bf16> &b_col_major,
                 std::vector<float> &moe_weights,
                 std::vector<int> &token_ids,
                 std::vector<int> &expert_ids) {
    a.resize(static_cast<size_t>(kNumTokens) * kK);
    b_col_major.resize(static_cast<size_t>(kNumExperts) * kN * kK);
    moe_weights.resize(kEM);
    token_ids.resize(kEM);
    expert_ids = {0, 1};

    for (int row = 0; row < kNumTokens; ++row) {
        for (int kk = 0; kk < kK; ++kk) {
            const float value = ((row * 11 + kk * 5 + 7) % 29 - 14) * 0.125f;
            a[static_cast<size_t>(row) * kK + kk] = float_to_bf16(value);
        }
        token_ids[row] = row;
        moe_weights[row] = 0.5f + 0.03125f * static_cast<float>(row % 7);
    }

    for (int expert = 0; expert < kNumExperts; ++expert) {
        for (int col = 0; col < kN; ++col) {
            for (int kk = 0; kk < kK; ++kk) {
                const float value = ((expert * 13 + col * 3 + kk * 7 + 1) % 31 - 15) * 0.0625f;
                b_col_major[(static_cast<size_t>(expert) * kN + col) * kK + kk] = float_to_bf16(value);
            }
        }
    }
}

std::vector<Bf16> reference_fused_moe(const std::vector<Bf16> &a,
                                      const std::vector<Bf16> &b_col_major,
                                      const std::vector<float> &moe_weights,
                                      const std::vector<int> &token_ids,
                                      const std::vector<int> &expert_ids) {
    std::vector<Bf16> out(static_cast<size_t>(kEM) * kN, float_to_bf16(0.0f));

    for (int routed_row = 0; routed_row < kEM; ++routed_row) {
        const int token = token_ids[routed_row];
        const int tile_idx = routed_row / kTileM;
        const int expert = expert_ids[tile_idx];
        const float moe_weight = moe_weights[routed_row];

        for (int col = 0; col < kN; ++col) {
            float acc = 0.0f;
            for (int kk = 0; kk < kK; ++kk) {
                acc += bf16_to_float(a[static_cast<size_t>(token) * kK + kk]) *
                       bf16_to_float(b_col_major[(static_cast<size_t>(expert) * kN + col) * kK + kk]);
            }
            out[static_cast<size_t>(routed_row) * kN + col] = float_to_bf16(acc * moe_weight);
        }
    }

    return out;
}

bool validate_result(const std::vector<Bf16> &got, const std::vector<Bf16> &expected) {
    constexpr float kTolerance = 2e-2f;
    size_t mismatch_count = 0;
    size_t first_bad = 0;
    float max_abs = 0.0f;

    for (size_t i = 0; i < got.size(); ++i) {
        const float got_f = bf16_to_float(got[i]);
        const float exp_f = bf16_to_float(expected[i]);
        const float abs_err = std::fabs(got_f - exp_f);
        max_abs = std::max(max_abs, abs_err);
        if (abs_err > kTolerance) {
            if (mismatch_count == 0) {
                first_bad = i;
            }
            ++mismatch_count;
        }
    }

    if (mismatch_count != 0) {
        const int row = static_cast<int>(first_bad / kN);
        const int col = static_cast<int>(first_bad % kN);
        std::cerr << "fused_moe_bf16_tn failed"
                  << ": mismatches=" << mismatch_count
                  << ", first mismatch at (" << row << ", " << col << ")"
                  << ", got=" << bf16_to_float(got[first_bad])
                  << ", expected=" << bf16_to_float(expected[first_bad])
                  << ", max_abs=" << max_abs << '\n';
        return false;
    }

    std::cout << "fused_moe_bf16_tn passed"
              << ": rows=" << kEM
              << ", topk=" << kTopK
              << ", N=" << kN
              << ", K=" << kK
              << ", sample C[0]=" << bf16_to_float(got.front())
              << ", C[last]=" << bf16_to_float(got.back())
              << ", max_abs=" << max_abs << '\n';
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

    std::vector<Bf16> host_a;
    std::vector<Bf16> host_b;
    std::vector<float> host_moe_weights;
    std::vector<int> host_token_ids;
    std::vector<int> host_expert_ids;
    std::vector<int32_t> host_num_tokens_post_padded(1, kEM);
    std::vector<Bf16> host_output(static_cast<size_t>(kEM) * kN, float_to_bf16(0.0f));

    fill_inputs(host_a, host_b, host_moe_weights, host_token_ids, host_expert_ids);
    const std::vector<Bf16> expected =
        reference_fused_moe(host_a, host_b, host_moe_weights, host_token_ids, host_expert_ids);

    Bf16 *dev_a = nullptr;
    Bf16 *dev_b = nullptr;
    float *dev_moe_weights = nullptr;
    int *dev_token_ids = nullptr;
    int *dev_expert_ids = nullptr;
    int32_t *dev_num_tokens_post_padded = nullptr;
    Bf16 *dev_c = nullptr;

    check_mc(mcMalloc(reinterpret_cast<void **>(&dev_a), host_a.size() * sizeof(Bf16)), "mcMalloc(dev_a)");
    check_mc(mcMalloc(reinterpret_cast<void **>(&dev_b), host_b.size() * sizeof(Bf16)), "mcMalloc(dev_b)");
    check_mc(mcMalloc(reinterpret_cast<void **>(&dev_moe_weights), host_moe_weights.size() * sizeof(float)),
             "mcMalloc(dev_moe_weights)");
    check_mc(mcMalloc(reinterpret_cast<void **>(&dev_token_ids), host_token_ids.size() * sizeof(int)),
             "mcMalloc(dev_token_ids)");
    check_mc(mcMalloc(reinterpret_cast<void **>(&dev_expert_ids), host_expert_ids.size() * sizeof(int)),
             "mcMalloc(dev_expert_ids)");
    check_mc(mcMalloc(reinterpret_cast<void **>(&dev_num_tokens_post_padded),
                      host_num_tokens_post_padded.size() * sizeof(int32_t)),
             "mcMalloc(dev_num_tokens_post_padded)");
    check_mc(mcMalloc(reinterpret_cast<void **>(&dev_c), host_output.size() * sizeof(Bf16)), "mcMalloc(dev_c)");

    check_mc(mcMemcpy(dev_a, host_a.data(), host_a.size() * sizeof(Bf16), mcMemcpyHostToDevice), "mcMemcpy(dev_a)");
    check_mc(mcMemcpy(dev_b, host_b.data(), host_b.size() * sizeof(Bf16), mcMemcpyHostToDevice), "mcMemcpy(dev_b)");
    check_mc(mcMemcpy(dev_moe_weights,
                      host_moe_weights.data(),
                      host_moe_weights.size() * sizeof(float),
                      mcMemcpyHostToDevice),
             "mcMemcpy(dev_moe_weights)");
    check_mc(mcMemcpy(dev_token_ids, host_token_ids.data(), host_token_ids.size() * sizeof(int), mcMemcpyHostToDevice),
             "mcMemcpy(dev_token_ids)");
    check_mc(mcMemcpy(dev_expert_ids,
                      host_expert_ids.data(),
                      host_expert_ids.size() * sizeof(int),
                      mcMemcpyHostToDevice),
             "mcMemcpy(dev_expert_ids)");
    check_mc(mcMemcpy(dev_num_tokens_post_padded,
                      host_num_tokens_post_padded.data(),
                      host_num_tokens_post_padded.size() * sizeof(int32_t),
                      mcMemcpyHostToDevice),
             "mcMemcpy(dev_num_tokens_post_padded)");
    check_mc(mcMemset(dev_c, 0, host_output.size() * sizeof(Bf16)), "mcMemset(dev_c)");

    GemmOp gemm_op;
    typename GemmOp::Arguments args(
        mctlass::gemm::GemmUniversalMode::kGemm,
        mctlass::gemm::BatchedGemmCoord(kEM, kN, kK, kNumExperts),
        typename GemmOp::epilogueParams(dev_moe_weights),
        dev_a,
        dev_b,
        dev_c,
        typename GemmOp::moeParams(dev_token_ids,
                                   dev_expert_ids,
                                   dev_num_tokens_post_padded,
                                   kEM,
                                   kTopK,
                                   true));

    check_mctlass(gemm_op(args, nullptr, nullptr), "gemm_op");
    check_mc(mcDeviceSynchronize(), "mcDeviceSynchronize");
    check_mc(mcGetLastError(), "mcGetLastError");

    check_mc(mcMemcpy(host_output.data(), dev_c, host_output.size() * sizeof(Bf16), mcMemcpyDeviceToHost),
             "mcMemcpy(host_output)");

    check_mc(mcFree(dev_a), "mcFree(dev_a)");
    check_mc(mcFree(dev_b), "mcFree(dev_b)");
    check_mc(mcFree(dev_moe_weights), "mcFree(dev_moe_weights)");
    check_mc(mcFree(dev_token_ids), "mcFree(dev_token_ids)");
    check_mc(mcFree(dev_expert_ids), "mcFree(dev_expert_ids)");
    check_mc(mcFree(dev_num_tokens_post_padded), "mcFree(dev_num_tokens_post_padded)");
    check_mc(mcFree(dev_c), "mcFree(dev_c)");

    return validate_result(host_output, expected) ? EXIT_SUCCESS : EXIT_FAILURE;
}
