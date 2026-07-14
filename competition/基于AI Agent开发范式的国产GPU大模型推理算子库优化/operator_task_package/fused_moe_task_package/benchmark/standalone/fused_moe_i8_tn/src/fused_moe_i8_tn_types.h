#pragma once

#include <cstdint>
#include <cstring>

#include <maca_bfloat16.h>
#include <mc_runtime_api.h>
#include <mc_runtime_types.h>

namespace fused_moe_i8_tn {

#if defined(__MXCC__) || (defined(__clang__) && defined(__MACA__))
#define FUSED_MOE_HOST_DEVICE __forceinline__ __device__ __host__
#define FUSED_MOE_DEVICE __forceinline__ __device__
#else
#define FUSED_MOE_HOST_DEVICE inline
#define FUSED_MOE_DEVICE inline
#endif

enum class Status {
    kSuccess,
    kErrorInternal,
};

inline const char *get_status_string(Status status) {
    switch (status) {
        case Status::kSuccess:
            return "Success";
        case Status::kErrorInternal:
            return "Error Internal";
    }
    return "Invalid status";
}

struct alignas(2) BFloat16 {
    uint16_t storage;

    FUSED_MOE_HOST_DEVICE
    BFloat16() : storage(0) {}

    FUSED_MOE_HOST_DEVICE
    explicit BFloat16(float x) {
#if defined(__MACA_ARCH__)
        auto tmp = __float2bfloat16(x);
        storage = reinterpret_cast<uint16_t const &>(tmp);
#else
        uint32_t bits;
        std::memcpy(&bits, &x, sizeof(bits));
        bits += ((bits >> 16) & 1) + 0x7fff;
        storage = static_cast<uint16_t>(bits >> 16);
#endif
    }

    FUSED_MOE_HOST_DEVICE
    operator float() const {
#if defined(__MACA_ARCH__)
        __maca_bfloat16_raw raw;
        raw.x = storage;
        return __bfloat162float(__maca_bfloat16(raw));
#else
        uint32_t bits = static_cast<uint32_t>(storage) << 16;
        float out;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
#endif
    }
};

struct BatchedGemmCoord {
    int m_;
    int n_;
    int k_;
    int batch_;

    FUSED_MOE_HOST_DEVICE
    BatchedGemmCoord() : m_(0), n_(0), k_(0), batch_(0) {}

    FUSED_MOE_HOST_DEVICE
    BatchedGemmCoord(int m, int n, int k, int batch) : m_(m), n_(n), k_(k), batch_(batch) {}

    FUSED_MOE_HOST_DEVICE
    int m() const { return m_; }

    FUSED_MOE_HOST_DEVICE
    int n() const { return n_; }

    FUSED_MOE_HOST_DEVICE
    int k() const { return k_; }

    FUSED_MOE_HOST_DEVICE
    int batch() const { return batch_; }
};

struct MoeParams {
    int *token_ids;
    int *expert_ids;
    int *num_tokens_post_padded_ptr;
    int32_t EM;
    int32_t topk;
    bool mul_weight;
    int topk_bits;

    FUSED_MOE_HOST_DEVICE
    MoeParams()
        : token_ids(nullptr),
          expert_ids(nullptr),
          num_tokens_post_padded_ptr(nullptr),
          EM(0),
          topk(0),
          mul_weight(false),
          topk_bits(0) {}

    FUSED_MOE_HOST_DEVICE
    MoeParams(int *token_ids_,
              int *expert_ids_,
              int *num_tokens_post_padded_ptr_,
              int EM_,
              int topk_,
              bool mul_weight_)
        : token_ids(token_ids_),
          expert_ids(expert_ids_),
          num_tokens_post_padded_ptr(num_tokens_post_padded_ptr_),
          EM(EM_),
          topk(topk_),
          mul_weight(mul_weight_),
          topk_bits(0) {
        int num = topk_;
        while (num >>= 1) {
            ++topk_bits;
        }
    }
};

struct EpilogueOutputOp {
    using ElementOutput = BFloat16;
    using ElementCompute = float;
    static constexpr int kCount = 2;
    static constexpr bool MUL_WEIGHTS = true;

    struct Params {
        ElementCompute const *scale_a;
        ElementCompute const *scale_b;
        ElementCompute const *moe_weights;

        FUSED_MOE_HOST_DEVICE
        Params() : scale_a(nullptr), scale_b(nullptr), moe_weights(nullptr) {}

        FUSED_MOE_HOST_DEVICE
        Params(ElementCompute const *scale_a_,
               ElementCompute const *scale_b_,
               ElementCompute const *moe_weights_)
            : scale_a(scale_a_), scale_b(scale_b_), moe_weights(moe_weights_) {}
    };

    ElementCompute const *scale_a_;
    ElementCompute const *scale_b_;
    ElementCompute const *moe_weights_;

    FUSED_MOE_HOST_DEVICE
    EpilogueOutputOp() : scale_a_(nullptr), scale_b_(nullptr), moe_weights_(nullptr) {}

    FUSED_MOE_HOST_DEVICE
    explicit EpilogueOutputOp(Params const &params)
        : scale_a_(params.scale_a), scale_b_(params.scale_b), moe_weights_(params.moe_weights) {}
};

}  // namespace fused_moe_i8_tn
