#pragma once

#include <mc_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "fused_moe_i8_tn_kernel.h"
#include "fused_moe_i8_tn_types.h"

namespace fused_moe_i8_tn {

constexpr int kDefaultNumExperts = 2;
constexpr int kDefaultTileM = 128;
constexpr int kDefaultN = 128;
constexpr int kDefaultK = 128;

struct HostInputs {
    int num_tokens = 0;
    int topk = 0;
    int em = 0;
    int num_experts = kDefaultNumExperts;
    int n = kDefaultN;
    int k = kDefaultK;
    std::vector<int8_t> a;
    std::vector<int8_t> b_col_major;
    std::vector<float> scale_a;
    std::vector<float> scale_b;
    std::vector<float> moe_weights;
    std::vector<int> token_ids;
    std::vector<int> expert_ids;
};

struct RunResult {
    int rows = 0;
    int cols = 0;
    std::vector<BFloat16> output;
};

inline float bf16_to_float(BFloat16 value) {
    return static_cast<float>(value);
}

inline BFloat16 float_to_bf16(float value) {
    return BFloat16(value);
}

inline std::vector<float> bf16_vector_to_float(const std::vector<BFloat16> &input) {
    std::vector<float> out(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        out[i] = bf16_to_float(input[i]);
    }
    return out;
}

inline void check_mc_or_throw(mcError_t status, const char *expr) {
    if (status != mcSuccess) {
        throw std::runtime_error(std::string(expr) + " failed: " + mcGetErrorString(status));
    }
}

inline void check_status_or_throw(Status status, const char *expr) {
    if (status != Status::kSuccess) {
        throw std::runtime_error(std::string(expr) + " failed: " + get_status_string(status));
    }
}

inline bool is_log2_value(int value) {
    return value > 0 && ((value & (value - 1)) == 0);
}

inline void validate_host_inputs(const HostInputs &inputs) {
    if (inputs.num_tokens <= 0) {
        throw std::invalid_argument("num_tokens must be > 0");
    }
    if (inputs.topk <= 0) {
        throw std::invalid_argument("topk must be > 0");
    }
    if (inputs.n != kDefaultN) {
        throw std::invalid_argument("only N=128 is supported by fused_moe_i8_tn");
    }
    if (inputs.k != kDefaultK) {
        throw std::invalid_argument("only K=128 is supported by fused_moe_i8_tn");
    }
    if (inputs.em <= 0) {
        throw std::invalid_argument("em must be > 0");
    }
    if (inputs.em != inputs.num_tokens * inputs.topk) {
        throw std::invalid_argument("em must equal num_tokens * topk");
    }
    if (inputs.em % kDefaultTileM != 0) {
        throw std::invalid_argument("em must be a multiple of 128");
    }
    if (inputs.num_experts <= 0) {
        throw std::invalid_argument("num_experts must be > 0");
    }

    const size_t total_rows = static_cast<size_t>(inputs.em);
    const size_t expected_a = static_cast<size_t>(inputs.num_tokens) * inputs.k;
    const size_t expected_b = static_cast<size_t>(inputs.num_experts) * inputs.n * inputs.k;
    const size_t expected_scale_a = static_cast<size_t>(inputs.num_tokens);
    const size_t expected_scale_b = static_cast<size_t>(inputs.num_experts) * inputs.n;
    const size_t expected_moe_weights = total_rows;
    const size_t expected_token_ids = total_rows;
    const size_t expected_expert_ids = static_cast<size_t>(inputs.em / kDefaultTileM);

    if (inputs.a.size() != expected_a) {
        throw std::invalid_argument("A size mismatch");
    }
    if (inputs.b_col_major.size() != expected_b) {
        throw std::invalid_argument("B size mismatch");
    }
    if (inputs.scale_a.size() != expected_scale_a) {
        throw std::invalid_argument("scale_a size mismatch");
    }
    if (inputs.scale_b.size() != expected_scale_b) {
        throw std::invalid_argument("scale_b size mismatch");
    }
    if (inputs.moe_weights.size() != expected_moe_weights) {
        throw std::invalid_argument("moe_weights size mismatch");
    }
    if (inputs.token_ids.size() != expected_token_ids) {
        throw std::invalid_argument("token_ids size mismatch");
    }
    if (inputs.expert_ids.size() != expected_expert_ids) {
        throw std::invalid_argument("expert_ids size mismatch");
    }

    for (int expert : inputs.expert_ids) {
        if (expert < 0 || expert >= inputs.num_experts) {
            throw std::invalid_argument("expert_ids contains out-of-range expert index");
        }
    }
}

template <typename GemmKernel>
inline typename GemmKernel::Arguments make_direct_arguments(const HostInputs &inputs,
                                                            int total_rows,
                                                            int8_t *dev_a,
                                                            int8_t *dev_b,
                                                            float *dev_scale_a,
                                                            float *dev_scale_b,
                                                            float *dev_moe_weights,
                                                            int *dev_token_ids,
                                                            int *dev_expert_ids,
                                                            int32_t *dev_num_tokens_post_padded,
                                                            BFloat16 *dev_c) {
    return typename GemmKernel::Arguments(
        BatchedGemmCoord(total_rows, inputs.n, inputs.k, inputs.num_experts),
        typename GemmKernel::EpilogueOutputOp::Params(dev_scale_a, dev_scale_b, dev_moe_weights),
        dev_a,
        dev_b,
        dev_c,
        MoeParams(dev_token_ids, dev_expert_ids, dev_num_tokens_post_padded, inputs.em, inputs.topk, true));
}

inline RunResult run_fused_moe_i8_tn(const HostInputs &inputs, int device_id = 0) {
    validate_host_inputs(inputs);

    int device_count = 0;
    check_mc_or_throw(mcGetDeviceCount(&device_count), "mcGetDeviceCount");
    if (device_count <= 0) {
        throw std::runtime_error("No MACA device is visible.");
    }
    if (device_id < 0 || device_id >= device_count) {
        throw std::invalid_argument("device_id is out of range");
    }
    check_mc_or_throw(mcSetDevice(device_id), "mcSetDevice");

    const int total_rows = inputs.em;
    std::vector<int32_t> host_num_tokens_post_padded(1, inputs.em);
    std::vector<BFloat16> host_output(static_cast<size_t>(total_rows) * inputs.n, float_to_bf16(0.0f));

    int8_t *dev_a = nullptr;
    int8_t *dev_b = nullptr;
    float *dev_scale_a = nullptr;
    float *dev_scale_b = nullptr;
    float *dev_moe_weights = nullptr;
    int *dev_token_ids = nullptr;
    int *dev_expert_ids = nullptr;
    int32_t *dev_num_tokens_post_padded = nullptr;
    BFloat16 *dev_c = nullptr;

    try {
        check_mc_or_throw(mcMalloc(reinterpret_cast<void **>(&dev_a), inputs.a.size() * sizeof(int8_t)), "mcMalloc(dev_a)");
        check_mc_or_throw(mcMalloc(reinterpret_cast<void **>(&dev_b), inputs.b_col_major.size() * sizeof(int8_t)), "mcMalloc(dev_b)");
        check_mc_or_throw(mcMalloc(reinterpret_cast<void **>(&dev_scale_a), inputs.scale_a.size() * sizeof(float)),
                          "mcMalloc(dev_scale_a)");
        check_mc_or_throw(mcMalloc(reinterpret_cast<void **>(&dev_scale_b), inputs.scale_b.size() * sizeof(float)),
                          "mcMalloc(dev_scale_b)");
        check_mc_or_throw(mcMalloc(reinterpret_cast<void **>(&dev_moe_weights), inputs.moe_weights.size() * sizeof(float)),
                          "mcMalloc(dev_moe_weights)");
        check_mc_or_throw(mcMalloc(reinterpret_cast<void **>(&dev_token_ids), inputs.token_ids.size() * sizeof(int)),
                          "mcMalloc(dev_token_ids)");
        check_mc_or_throw(mcMalloc(reinterpret_cast<void **>(&dev_expert_ids), inputs.expert_ids.size() * sizeof(int)),
                          "mcMalloc(dev_expert_ids)");
        check_mc_or_throw(mcMalloc(reinterpret_cast<void **>(&dev_num_tokens_post_padded),
                                   host_num_tokens_post_padded.size() * sizeof(int32_t)),
                          "mcMalloc(dev_num_tokens_post_padded)");
        check_mc_or_throw(mcMalloc(reinterpret_cast<void **>(&dev_c), host_output.size() * sizeof(BFloat16)), "mcMalloc(dev_c)");

        check_mc_or_throw(mcMemcpy(dev_a, inputs.a.data(), inputs.a.size() * sizeof(int8_t), mcMemcpyHostToDevice),
                          "mcMemcpy(dev_a)");
        check_mc_or_throw(mcMemcpy(dev_b,
                                   inputs.b_col_major.data(),
                                   inputs.b_col_major.size() * sizeof(int8_t),
                                   mcMemcpyHostToDevice),
                          "mcMemcpy(dev_b)");
        check_mc_or_throw(mcMemcpy(dev_scale_a,
                                   inputs.scale_a.data(),
                                   inputs.scale_a.size() * sizeof(float),
                                   mcMemcpyHostToDevice),
                          "mcMemcpy(dev_scale_a)");
        check_mc_or_throw(mcMemcpy(dev_scale_b,
                                   inputs.scale_b.data(),
                                   inputs.scale_b.size() * sizeof(float),
                                   mcMemcpyHostToDevice),
                          "mcMemcpy(dev_scale_b)");
        check_mc_or_throw(mcMemcpy(dev_moe_weights,
                                   inputs.moe_weights.data(),
                                   inputs.moe_weights.size() * sizeof(float),
                                   mcMemcpyHostToDevice),
                          "mcMemcpy(dev_moe_weights)");
        check_mc_or_throw(mcMemcpy(dev_token_ids,
                                   inputs.token_ids.data(),
                                   inputs.token_ids.size() * sizeof(int),
                                   mcMemcpyHostToDevice),
                          "mcMemcpy(dev_token_ids)");
        check_mc_or_throw(mcMemcpy(dev_expert_ids,
                                   inputs.expert_ids.data(),
                                   inputs.expert_ids.size() * sizeof(int),
                                   mcMemcpyHostToDevice),
                          "mcMemcpy(dev_expert_ids)");
        check_mc_or_throw(mcMemcpy(dev_num_tokens_post_padded,
                                   host_num_tokens_post_padded.data(),
                                   host_num_tokens_post_padded.size() * sizeof(int32_t),
                                   mcMemcpyHostToDevice),
                          "mcMemcpy(dev_num_tokens_post_padded)");
        check_mc_or_throw(mcMemset(dev_c, 0, host_output.size() * sizeof(BFloat16)), "mcMemset(dev_c)");

        const bool topk_is_log2 = is_log2_value(inputs.topk);
        if (topk_is_log2) {
            using GemmKernel = DirectMoeKernel<true>;
            auto const args = make_direct_arguments<GemmKernel>(inputs,
                                                                total_rows,
                                                                dev_a,
                                                                dev_b,
                                                                dev_scale_a,
                                                                dev_scale_b,
                                                                dev_moe_weights,
                                                                dev_token_ids,
                                                                dev_expert_ids,
                                                                dev_num_tokens_post_padded,
                                                                dev_c);
            check_status_or_throw(launch<GemmKernel>(args), "direct_moe_kernel");
        } else {
            using GemmKernel = DirectMoeKernel<false>;
            auto const args = make_direct_arguments<GemmKernel>(inputs,
                                                                total_rows,
                                                                dev_a,
                                                                dev_b,
                                                                dev_scale_a,
                                                                dev_scale_b,
                                                                dev_moe_weights,
                                                                dev_token_ids,
                                                                dev_expert_ids,
                                                                dev_num_tokens_post_padded,
                                                                dev_c);
            check_status_or_throw(launch<GemmKernel>(args), "direct_moe_kernel");
        }

        check_mc_or_throw(mcDeviceSynchronize(), "mcDeviceSynchronize");
        check_mc_or_throw(mcGetLastError(), "mcGetLastError");
        check_mc_or_throw(mcMemcpy(host_output.data(),
                                   dev_c,
                                   host_output.size() * sizeof(BFloat16),
                                   mcMemcpyDeviceToHost),
                          "mcMemcpy(host_output)");
    } catch (...) {
        if (dev_a) {
            mcFree(dev_a);
        }
        if (dev_b) {
            mcFree(dev_b);
        }
        if (dev_scale_a) {
            mcFree(dev_scale_a);
        }
        if (dev_scale_b) {
            mcFree(dev_scale_b);
        }
        if (dev_moe_weights) {
            mcFree(dev_moe_weights);
        }
        if (dev_token_ids) {
            mcFree(dev_token_ids);
        }
        if (dev_expert_ids) {
            mcFree(dev_expert_ids);
        }
        if (dev_num_tokens_post_padded) {
            mcFree(dev_num_tokens_post_padded);
        }
        if (dev_c) {
            mcFree(dev_c);
        }
        throw;
    }

    check_mc_or_throw(mcFree(dev_a), "mcFree(dev_a)");
    check_mc_or_throw(mcFree(dev_b), "mcFree(dev_b)");
    check_mc_or_throw(mcFree(dev_scale_a), "mcFree(dev_scale_a)");
    check_mc_or_throw(mcFree(dev_scale_b), "mcFree(dev_scale_b)");
    check_mc_or_throw(mcFree(dev_moe_weights), "mcFree(dev_moe_weights)");
    check_mc_or_throw(mcFree(dev_token_ids), "mcFree(dev_token_ids)");
    check_mc_or_throw(mcFree(dev_expert_ids), "mcFree(dev_expert_ids)");
    check_mc_or_throw(mcFree(dev_num_tokens_post_padded), "mcFree(dev_num_tokens_post_padded)");
    check_mc_or_throw(mcFree(dev_c), "mcFree(dev_c)");

    RunResult result;
    result.rows = total_rows;
    result.cols = inputs.n;
    result.output = std::move(host_output);
    return result;
}

}  // namespace fused_moe_i8_tn
