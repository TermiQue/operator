#pragma once

#include <algorithm>
#include <cstdint>

#include <cute/tensor.hpp>

#include "fused_moe_i8_tn_macros.h"
#include "fused_moe_i8_tn_types.h"

namespace fused_moe_i8_tn {

using ElementA = int8_t;
using ElementB = int8_t;
using ElementC = BFloat16;
using ElementAccumulator = int32_t;
using ElementCompute = float;

using INT1 = __NATIVE_VECTOR__(1, int32_t);
using INT4 = __NATIVE_VECTOR__(4, int32_t);
using FLOAT2 = __NATIVE_VECTOR__(2, float);
using FLOAT4 = __NATIVE_VECTOR__(4, float);
using LdgType = __NATIVE_VECTOR__(4, int32_t);
using StsType = LdgType;
using LdsType = LdgType;
using StgType = __NATIVE_VECTOR__(2, uint);
using Tc = maca_bfloat16;

constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kTileK = 128;
constexpr int kThreadCount = 256;
constexpr int kWaveSize = 64;
constexpr int kWaveNum = kThreadCount / kWaveSize;
constexpr int kWaveM = 4;
constexpr int kWaveN = kWaveNum / kWaveM;
constexpr int kLdgSize = sizeof(LdgType) * kThreadCount;
constexpr int kMNPerLdg = kLdgSize / kTileK;
constexpr int kLdgSizePerWave = kLdgSize / kWaveNum;
constexpr int kSizeA = kTileM * kTileK * sizeof(ElementA);
constexpr int kSizeB = kTileN * kTileK * sizeof(ElementB);
constexpr int kLdgNumA = kSizeA / kLdgSize;
constexpr int kLdgNumB = kSizeB / kLdgSize;
constexpr int kLdsNumA = kSizeA / (kLdgSizePerWave * kWaveM);
constexpr int kLdsNumB = kSizeB / (kLdgSizePerWave * kWaveN);
constexpr int kStsNumA = kLdgNumA;
constexpr int kStsNumB = kLdgNumB;
constexpr int kMmaM = kTileM / 16 / kWaveM;
constexpr int kMmaN = kTileN / 16 / kWaveN;
constexpr int kMmaK = kTileK / 16;
constexpr int kRowCSize = 8;
constexpr int kOutputCount = 16;
constexpr int kSmemSize = kSizeA + kSizeB;

template <bool IsTopkLog2>
struct DirectMoeKernel {
    static constexpr bool kIsTopkLog2 = IsTopkLog2;
    using EpilogueOutputOp = fused_moe_i8_tn::EpilogueOutputOp;

    struct Arguments {
        BatchedGemmCoord problem_size;
        typename EpilogueOutputOp::Params output_op;
        void const *ptr_A;
        void const *ptr_B;
        void *ptr_C;
        MoeParams moe_params;

        FUSED_MOE_HOST_DEVICE
        Arguments() : ptr_A(nullptr), ptr_B(nullptr), ptr_C(nullptr) {}

        FUSED_MOE_HOST_DEVICE
        Arguments(BatchedGemmCoord problem_size_,
                  typename EpilogueOutputOp::Params output_op_,
                  void const *ptr_A_,
                  void const *ptr_B_,
                  void *ptr_C_,
                  MoeParams moe_params_)
            : problem_size(problem_size_),
              output_op(output_op_),
              ptr_A(ptr_A_),
              ptr_B(ptr_B_),
              ptr_C(ptr_C_),
              moe_params(moe_params_) {}
    };
};

template <bool IsTopkLog2>
__global__ void direct_moe_kernel(typename DirectMoeKernel<IsTopkLog2>::Arguments args) {
    using namespace cute;

#define MMA_STAGE_MNKX2(m, n, k)                                                                    \
    accum[m][n] = FUSED_MOE_BUILTIN_MMA_16X16X16_I8(a[m][k], b[n][k], accum[m][n]);                \
    accum[m][n] = FUSED_MOE_BUILTIN_MMA_16X16X16_I8(a[m][k + 1], b[n][k + 1], accum[m][n])

#define LDG_A_STAGE_I(ldgi)                                                                         \
    A[ldgi] = __builtin_mxc_ldg_b128_predicator(Aaddr + ldg_a_offs_m[ldgi] + ldg_k,                 \
                                                0,                                                  \
                                                true,                                               \
                                                true,                                               \
                                                false,                                              \
                                                false,                                              \
                                                rowA_mask[ldgi],                                    \
                                                1,                                                  \
                                                MACA_ICMP_EQ)

#define LDG_B_STAGE_I(ldgi)                                                                         \
    B[ldgi] = __builtin_mxc_ldg_b128(&(gB(ldg_n[ldgi], ldg_k, tile_k)),                             \
                                     0,                                                             \
                                     -1,                                                            \
                                     true,                                                          \
                                     true,                                                          \
                                     false,                                                         \
                                     false)

#define LDS_A_B128(rowi, coli) FUSED_MOE_LDS(a[rowi][coli * 4], sA(lds_row_A[rowi], lds_col[coli]), LdsType)
#define LDS_B_B128(rowi, coli) FUSED_MOE_LDS(b[rowi][coli * 4], sB(lds_row_B[rowi], lds_col[coli]), LdsType)

#define CVT_F32_TO_BF16(dst, src0, src1)                                                            \
    src0 = ((src0 >> 16) & 1) + src0 + 0x7fff;                                                      \
    src1 = ((src1 >> 16) & 1) + src1 + 0x7fff;                                                      \
    dst = __builtin_mxc_byte_perm(src0, src1, 0x03020706)

    int *token_ids_ptr = args.moe_params.token_ids;
    int *expert_ids_ptr = args.moe_params.expert_ids;
    int *num_tokens_post_padded_ptr = args.moe_params.num_tokens_post_padded_ptr;

    int num_tokens_post_padded = num_tokens_post_padded_ptr[0];
    int tid = threadIdx.x;
    int bidx = blockIdx.x + blockIdx.z * gridDim.x;
    int bidy = blockIdx.y;
    int wave = tid / kWaveSize;
    int lane = tid % kWaveSize;

    if (bidx * kTileM >= num_tokens_post_padded) {
        return;
    }

    EpilogueOutputOp output_op(args.output_op);

    __shared__ int8_t smem_data[kSmemSize];
    int8_t *smem_A = smem_data;
    int8_t *smem_B = smem_A + kSizeA;

    int group_idx = expert_ids_ptr[bidx];
    int prev_m = bidx * kTileM;
    ElementB *Baddr = (ElementB *)args.ptr_B + uint64_t(group_idx) * args.problem_size.n() * args.problem_size.k();

    Tensor mB = make_tensor(make_gmem_ptr((ElementB *)Baddr),
                            make_shape(args.problem_size.n(), args.problem_size.k()),
                            make_stride(args.problem_size.k(), Int<1>{}));
    Tensor gB = local_tile(mB, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(bidy, _));

    LdgType A[kLdgNumA], B[kLdgNumB];
    int k_head = (args.problem_size.k() - 1) % kTileK + 1;
    int col_limit = min(kTileN, args.problem_size.n() - bidy * kTileN);
    int ldg_n[kLdgNumB], ldg_a_offs_m[kLdgNumA];
    bool rowA_mask[kLdgNumA];
    int ldg_m_base = tid / 8;
    int ldg_n_base = tid / 8 * kLdgNumB;
    int ldg_k = (lane % 8) * 16;
    int num_tile_k = size<2>(gB);

    ElementA *Aaddr = (ElementA *)args.ptr_A + (num_tile_k - 1) * kTileK;

#pragma unroll
    for (uint32_t ldgi = 0; ldgi < kLdgNumA; ++ldgi) {
        int idx_row_a = ldg_m_base + kMNPerLdg * ldgi;
        reinterpret_cast<INT1 *>(&ldg_a_offs_m)[ldgi] =
            __builtin_mxc_ldg_b32(token_ids_ptr + idx_row_a + prev_m, 0, -1, true, true, false, false);
    }
#pragma unroll
    for (uint32_t ldgi = 0; ldgi < kLdgNumB; ++ldgi) {
        ldg_n[ldgi] = min(ldg_n_base + ldgi, col_limit - 1);
        B[ldgi] = __builtin_mxc_ldg_b128_predicator(&(gB(ldg_n[ldgi], ldg_k, num_tile_k - 1)),
                                                    0,
                                                    true,
                                                    true,
                                                    false,
                                                    false,
                                                    ldg_k,
                                                    k_head,
                                                    MACA_ICMP_SLT);
    }
#pragma unroll
    for (uint32_t ldgi = 0; ldgi < kLdgNumA; ++ldgi) {
        rowA_mask[ldgi] = ldg_a_offs_m[ldgi] < args.problem_size.m();
        if constexpr (IsTopkLog2) {
            ldg_a_offs_m[ldgi] = (ldg_a_offs_m[ldgi] >> args.moe_params.topk_bits) * args.problem_size.k();
        } else {
            ldg_a_offs_m[ldgi] = (ldg_a_offs_m[ldgi] / args.moe_params.topk) * args.problem_size.k();
        }
        A[ldgi] = __builtin_mxc_ldg_b128_predicator(Aaddr + ldg_a_offs_m[ldgi] + ldg_k,
                                                    0,
                                                    true,
                                                    true,
                                                    false,
                                                    false,
                                                    (ldg_k < k_head) && rowA_mask[ldgi],
                                                    1,
                                                    MACA_ICMP_EQ);
    }

    Tensor sA = make_tensor(make_smem_ptr((ElementA *)smem_A),
                            make_shape(Int<kTileM>{}, Int<kTileK>{}),
                            make_stride(Int<kTileK>{}, Int<1>{}));
    Tensor sB = make_tensor(make_smem_ptr((ElementB *)smem_B),
                            make_shape(Int<kTileN>{}, Int<kTileK>{}),
                            make_stride(Int<kTileK>{}, Int<1>{}));

    int sts_rowA[kStsNumA], sts_rowB[kStsNumB];
    int sts_col = (((tid / 8) + (tid % 8)) % 8) * 16;
#pragma unroll
    for (uint32_t i = 0; i < kStsNumB; ++i) {
        sts_rowB[i] = tid / 8 + kMNPerLdg * i;
        FUSED_MOE_STS(sB(sts_rowB[i], sts_col), B[i], StsType);
    }
#pragma unroll
    for (uint32_t i = 0; i < kStsNumA; ++i) {
        sts_rowA[i] = wave * 32 + lane / 8 + i * 8;
    }
    FUSED_MOE_STS(sA(sts_rowA[0], sts_col), A[0], StsType);
    FUSED_MOE_STS(sA(sts_rowA[1], sts_col), A[1], StsType);

    INT4 accum[kMmaM][kMmaN] = {0};
    int32_t a[kMmaM][kMmaK], b[kMmaN][kMmaK];
    int lds_row_A[2], lds_row_B[8], lds_col[2];

#pragma unroll
    for (int i = 0; i < 2; ++i) {
        lds_col[i] = (((tid % 16) + (lane / 16) + 4 * i) % 8) * 16;
        lds_row_A[i] = (tid % 16) + wave * 32 + 16 * i;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        lds_row_B[i] = (tid % 16) + 16 * i;
    }

    __syncthreadshared();

    LDS_A_B128(0, 0);
    LDS_B_B128(0, 0);
    LDS_B_B128(1, 0);
    LDS_B_B128(2, 0);
    LDS_B_B128(3, 0);

    int loop_tile_k = size<2>(gB) - 1;
    Aaddr = (ElementA *)args.ptr_A;
    for (uint32_t tile_k = 0; tile_k < loop_tile_k; ++tile_k) {
        LDG_B_STAGE_I(0);
        LDG_B_STAGE_I(1);
        MMA_STAGE_MNKX2(0, 0, 0);
        LDS_B_B128(4, 0);
        MMA_STAGE_MNKX2(0, 0, 2);
        LDS_B_B128(5, 0);
        MMA_STAGE_MNKX2(0, 1, 0);
        LDS_B_B128(6, 0);
        LDG_B_STAGE_I(2);
        MMA_STAGE_MNKX2(0, 1, 2);
        LDS_B_B128(7, 0);
        MMA_STAGE_MNKX2(0, 2, 0);
        LDG_B_STAGE_I(3);
        MMA_STAGE_MNKX2(0, 2, 2);
        MMA_STAGE_MNKX2(0, 3, 0);
        LDG_A_STAGE_I(0);
        MMA_STAGE_MNKX2(0, 3, 2);
        LDG_A_STAGE_I(1);

        MMA_STAGE_MNKX2(0, 4, 0);
        LDS_A_B128(0, 1);
        MMA_STAGE_MNKX2(0, 4, 2);
        LDS_B_B128(0, 1);
        MMA_STAGE_MNKX2(0, 5, 0);
        LDS_B_B128(1, 1);
        MMA_STAGE_MNKX2(0, 5, 2);
        LDS_B_B128(2, 1);
        MMA_STAGE_MNKX2(0, 6, 0);
        LDS_B_B128(3, 1);
        MMA_STAGE_MNKX2(0, 6, 2);
        MMA_STAGE_MNKX2(0, 7, 0);
        MMA_STAGE_MNKX2(0, 7, 2);

        LDS_B_B128(4, 1);
        MMA_STAGE_MNKX2(0, 0, 4);
        LDS_B_B128(5, 1);
        MMA_STAGE_MNKX2(0, 0, 6);
        LDS_B_B128(6, 1);
        MMA_STAGE_MNKX2(0, 1, 4);
        LDS_B_B128(7, 1);
        MMA_STAGE_MNKX2(0, 1, 6);
        MMA_STAGE_MNKX2(0, 2, 4);
        MMA_STAGE_MNKX2(0, 2, 6);
        FUSED_MOE_STS(sA(sts_rowA[2], sts_col), A[2], StsType);
        MMA_STAGE_MNKX2(0, 3, 4);
        MMA_STAGE_MNKX2(0, 3, 6);
        FUSED_MOE_STS(sA(sts_rowA[3], sts_col), A[3], StsType);

        MMA_STAGE_MNKX2(0, 4, 4);
        LDG_A_STAGE_I(2);
        MMA_STAGE_MNKX2(0, 4, 6);
        LDG_A_STAGE_I(3);
        MMA_STAGE_MNKX2(0, 5, 4);
        MMA_STAGE_MNKX2(0, 5, 6);
        MMA_STAGE_MNKX2(0, 6, 4);
        LDS_A_B128(1, 0);
        MMA_STAGE_MNKX2(0, 6, 6);
        MMA_STAGE_MNKX2(0, 7, 4);
        Aaddr += kTileK;
        MMA_STAGE_MNKX2(0, 7, 6);

        __syncthreadshared();
        MMA_STAGE_MNKX2(1, 0, 0);
        LDS_A_B128(1, 1);
        MMA_STAGE_MNKX2(1, 0, 2);
        MMA_STAGE_MNKX2(1, 1, 0);
        MMA_STAGE_MNKX2(1, 1, 2);
        MMA_STAGE_MNKX2(1, 2, 0);
        MMA_STAGE_MNKX2(1, 2, 2);
        MMA_STAGE_MNKX2(1, 3, 0);
        MMA_STAGE_MNKX2(1, 3, 2);

        MMA_STAGE_MNKX2(1, 4, 0);
        FUSED_MOE_STS(sB(sts_rowB[0], sts_col), B[0], StsType);
        MMA_STAGE_MNKX2(1, 4, 2);
        MMA_STAGE_MNKX2(1, 5, 0);
        MMA_STAGE_MNKX2(1, 5, 2);
        FUSED_MOE_STS(sB(sts_rowB[1], sts_col), B[1], StsType);
        MMA_STAGE_MNKX2(1, 6, 0);
        MMA_STAGE_MNKX2(1, 6, 2);
        MMA_STAGE_MNKX2(1, 7, 0);
        FUSED_MOE_STS(sB(sts_rowB[2], sts_col), B[2], StsType);
        MMA_STAGE_MNKX2(1, 7, 2);

        MMA_STAGE_MNKX2(1, 0, 4);
        MMA_STAGE_MNKX2(1, 0, 6);
        FUSED_MOE_STS(sB(sts_rowB[3], sts_col), B[3], StsType);
        MMA_STAGE_MNKX2(1, 1, 4);
        MMA_STAGE_MNKX2(1, 1, 6);
        MMA_STAGE_MNKX2(1, 2, 4);
        FUSED_MOE_STS(sA(sts_rowA[0], sts_col), A[0], StsType);
        MMA_STAGE_MNKX2(1, 2, 6);
        MMA_STAGE_MNKX2(1, 3, 4);
        MMA_STAGE_MNKX2(1, 3, 6);
        FUSED_MOE_STS(sA(sts_rowA[1], sts_col), A[1], StsType);

        MMA_STAGE_MNKX2(1, 4, 4);
        MMA_STAGE_MNKX2(1, 4, 6);
        MMA_STAGE_MNKX2(1, 5, 4);
        __syncthreadshared();
        MMA_STAGE_MNKX2(1, 5, 6);
        LDS_A_B128(0, 0);
        LDS_B_B128(0, 0);
        MMA_STAGE_MNKX2(1, 6, 4);
        LDS_B_B128(1, 0);
        MMA_STAGE_MNKX2(1, 6, 6);
        LDS_B_B128(2, 0);
        MMA_STAGE_MNKX2(1, 7, 4);
        LDS_B_B128(3, 0);
        MMA_STAGE_MNKX2(1, 7, 6);
    }

    int rowC[kRowCSize];
    MMA_STAGE_MNKX2(0, 0, 0);
    LDS_B_B128(4, 0);
    MMA_STAGE_MNKX2(0, 0, 2);
    LDS_B_B128(5, 0);
    MMA_STAGE_MNKX2(0, 1, 0);
    LDS_B_B128(6, 0);
    MMA_STAGE_MNKX2(0, 1, 2);
    LDS_B_B128(7, 0);
    MMA_STAGE_MNKX2(0, 2, 0);
    int token_row_m = prev_m + ((lane / 16) % 2) * 4 + wave * 8 + (lane / 32) * 32;
    MMA_STAGE_MNKX2(0, 2, 2);
    MMA_STAGE_MNKX2(0, 3, 0);
    MMA_STAGE_MNKX2(0, 3, 2);

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        *(reinterpret_cast<INT1 *>(&rowC) + j) =
            __builtin_mxc_ldg_b32(token_ids_ptr + token_row_m + j, 0, -1, true, true, false, false);
    }

    MMA_STAGE_MNKX2(0, 4, 0);
    LDS_A_B128(0, 1);
    MMA_STAGE_MNKX2(0, 4, 2);
    LDS_B_B128(0, 1);
    MMA_STAGE_MNKX2(0, 5, 0);
    LDS_B_B128(1, 1);
    MMA_STAGE_MNKX2(0, 5, 2);
    LDS_B_B128(2, 1);
    MMA_STAGE_MNKX2(0, 6, 0);
    LDS_B_B128(3, 1);
    MMA_STAGE_MNKX2(0, 6, 2);
    MMA_STAGE_MNKX2(0, 7, 0);
    MMA_STAGE_MNKX2(0, 7, 2);

    LDS_B_B128(4, 1);
    MMA_STAGE_MNKX2(0, 0, 4);
    LDS_B_B128(5, 1);
    MMA_STAGE_MNKX2(0, 0, 6);
    LDS_B_B128(6, 1);
    MMA_STAGE_MNKX2(0, 1, 4);
    LDS_B_B128(7, 1);
    MMA_STAGE_MNKX2(0, 1, 6);
    MMA_STAGE_MNKX2(0, 2, 4);
    FUSED_MOE_STS(sA(sts_rowA[2], sts_col), A[2], StsType);
    MMA_STAGE_MNKX2(0, 2, 6);
    MMA_STAGE_MNKX2(0, 3, 4);
    MMA_STAGE_MNKX2(0, 3, 6);
    FUSED_MOE_STS(sA(sts_rowA[3], sts_col), A[3], StsType);

    MMA_STAGE_MNKX2(0, 4, 4);
    MMA_STAGE_MNKX2(0, 4, 6);
    MMA_STAGE_MNKX2(0, 5, 4);
    MMA_STAGE_MNKX2(0, 5, 6);
    MMA_STAGE_MNKX2(0, 6, 4);
    LDS_A_B128(1, 0);
    MMA_STAGE_MNKX2(0, 6, 6);
    MMA_STAGE_MNKX2(0, 7, 4);
    MMA_STAGE_MNKX2(0, 7, 6);

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        *(reinterpret_cast<INT1 *>(&rowC) + 4 + j) =
            __builtin_mxc_ldg_b32(token_ids_ptr + token_row_m + 64 + j, 0, -1, true, true, false, false);
    }

    MMA_STAGE_MNKX2(1, 0, 0);
    MMA_STAGE_MNKX2(1, 0, 2);
    MMA_STAGE_MNKX2(1, 1, 0);
    MMA_STAGE_MNKX2(1, 1, 2);
    MMA_STAGE_MNKX2(1, 2, 0);
    MMA_STAGE_MNKX2(1, 2, 2);
    MMA_STAGE_MNKX2(1, 3, 0);
    MMA_STAGE_MNKX2(1, 3, 2);

    MMA_STAGE_MNKX2(1, 4, 0);
    MMA_STAGE_MNKX2(1, 4, 2);
    LDS_A_B128(1, 1);
    MMA_STAGE_MNKX2(1, 5, 0);
    MMA_STAGE_MNKX2(1, 5, 2);
    MMA_STAGE_MNKX2(1, 6, 0);
    MMA_STAGE_MNKX2(1, 6, 2);
    MMA_STAGE_MNKX2(1, 7, 0);
    MMA_STAGE_MNKX2(1, 7, 2);

    MMA_STAGE_MNKX2(1, 0, 4);
    MMA_STAGE_MNKX2(1, 0, 6);
    MMA_STAGE_MNKX2(1, 1, 4);
    MMA_STAGE_MNKX2(1, 1, 6);
    MMA_STAGE_MNKX2(1, 2, 4);
    MMA_STAGE_MNKX2(1, 2, 6);
    MMA_STAGE_MNKX2(1, 3, 4);
    MMA_STAGE_MNKX2(1, 3, 6);

    MMA_STAGE_MNKX2(1, 4, 4);
    MMA_STAGE_MNKX2(1, 4, 6);
    MMA_STAGE_MNKX2(1, 5, 4);
    MMA_STAGE_MNKX2(1, 5, 6);
    MMA_STAGE_MNKX2(1, 6, 4);
    MMA_STAGE_MNKX2(1, 6, 6);
    MMA_STAGE_MNKX2(1, 7, 4);
    MMA_STAGE_MNKX2(1, 7, 6);

    INT4 output[kOutputCount];
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            output[i * 8 + 2 * j][0] = accum[i][0][j];
            output[i * 8 + 2 * j][1] = accum[i][2][j];
            output[i * 8 + 2 * j][2] = accum[i][4][j];
            output[i * 8 + 2 * j][3] = accum[i][6][j];
            output[i * 8 + 2 * j + 1][0] = accum[i][1][j];
            output[i * 8 + 2 * j + 1][1] = accum[i][3][j];
            output[i * 8 + 2 * j + 1][2] = accum[i][5][j];
            output[i * 8 + 2 * j + 1][3] = accum[i][7][j];
        }
    }

    int colC[2];
    bool colC_mask[2];
    colC[0] = (tid % 16) * 4;
    colC[1] = colC[0] + 64;
    colC_mask[0] = colC[0] < col_limit;
    colC_mask[1] = colC[1] < col_limit;

    float weights[2][4], a_scale[2][4];
    FLOAT4 b_scale[2];

#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            if (output_op.MUL_WEIGHTS) {
                const void *moe_weights_ptr = output_op.moe_weights_ + rowC[i * 4 + j];
                *(reinterpret_cast<INT1 *>(&weights[i]) + j) =
                    __builtin_mxc_ldg_b32_predicator(const_cast<void *>(moe_weights_ptr),
                                                     0,
                                                     true,
                                                     true,
                                                     false,
                                                     false,
                                                     rowC[i * 4 + j],
                                                     args.problem_size.m(),
                                                     MACA_ICMP_SLT);
            }

            int row_a_scale;
            if constexpr (IsTopkLog2) {
                row_a_scale = (rowC[i * 4 + j] >> args.moe_params.topk_bits);
            } else {
                row_a_scale = (rowC[i * 4 + j] / args.moe_params.topk);
            }

            const void *scale_a_ptr = output_op.scale_a_ + row_a_scale;
            *(reinterpret_cast<INT1 *>(&a_scale[i]) + j) =
                __builtin_mxc_ldg_b32_predicator(const_cast<void *>(scale_a_ptr),
                                                 0,
                                                 true,
                                                 true,
                                                 false,
                                                 false,
                                                 rowC[i * 4 + j],
                                                 args.problem_size.m(),
                                                 MACA_ICMP_SLT);
        }
    }

#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const void *scale_b_ptr =
            (const float *)output_op.scale_b_ + group_idx * args.problem_size.n() + bidy * kTileN + colC[i];
        b_scale[i] = __builtin_mxc_ldg_b128_predicator(const_cast<void *>(scale_b_ptr),
                                                       0,
                                                       true,
                                                       true,
                                                       false,
                                                       false,
                                                       colC_mask[i],
                                                       1,
                                                       MACA_ICMP_EQ);
    }

    Tc *Caddr = (Tc *)args.ptr_C + bidy * kTileN;
    FLOAT2 zero2 = {0.f, 0.f};
    StgType tempC;

#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            float out[8];
            out[0] = output[i * 8 + 2 * j][0];
            out[1] = output[i * 8 + 2 * j][1];
            out[2] = output[i * 8 + 2 * j][2];
            out[3] = output[i * 8 + 2 * j][3];
            out[4] = output[i * 8 + 2 * j + 1][0];
            out[5] = output[i * 8 + 2 * j + 1][1];
            out[6] = output[i * 8 + 2 * j + 1][2];
            out[7] = output[i * 8 + 2 * j + 1][3];

            if (output_op.MUL_WEIGHTS) {
                a_scale[i][j] *= weights[i][j];
            }

            FLOAT2 a_scale_f2 = {a_scale[i][j], a_scale[i][j]};
            FLOAT2 scale[4];
            scale[0] = __builtin_mxc_pk_fma_f32(reinterpret_cast<FLOAT2 *>(&b_scale[0])[0], a_scale_f2, zero2);
            scale[1] = __builtin_mxc_pk_fma_f32(reinterpret_cast<FLOAT2 *>(&b_scale[0])[1], a_scale_f2, zero2);
            scale[2] = __builtin_mxc_pk_fma_f32(reinterpret_cast<FLOAT2 *>(&b_scale[1])[0], a_scale_f2, zero2);
            scale[3] = __builtin_mxc_pk_fma_f32(reinterpret_cast<FLOAT2 *>(&b_scale[1])[1], a_scale_f2, zero2);
            *reinterpret_cast<FLOAT2 *>(&out[0]) =
                __builtin_mxc_pk_fma_f32(*reinterpret_cast<FLOAT2 *>(&out[0]), scale[0], zero2);
            *reinterpret_cast<FLOAT2 *>(&out[2]) =
                __builtin_mxc_pk_fma_f32(*reinterpret_cast<FLOAT2 *>(&out[2]), scale[1], zero2);
            *reinterpret_cast<FLOAT2 *>(&out[4]) =
                __builtin_mxc_pk_fma_f32(*reinterpret_cast<FLOAT2 *>(&out[4]), scale[2], zero2);
            *reinterpret_cast<FLOAT2 *>(&out[6]) =
                __builtin_mxc_pk_fma_f32(*reinterpret_cast<FLOAT2 *>(&out[6]), scale[3], zero2);

            CVT_F32_TO_BF16(tempC[0], reinterpret_cast<uint *>(&out)[0], reinterpret_cast<uint *>(&out)[1]);
            CVT_F32_TO_BF16(tempC[1], reinterpret_cast<uint *>(&out)[2], reinterpret_cast<uint *>(&out)[3]);
            __builtin_mxc_stg_b64_predicator(Caddr + rowC[i * 4 + j] * args.problem_size.n() + colC[0],
                                             0,
                                             *(reinterpret_cast<uint64_t *>(&tempC)),
                                             true,
                                             false,
                                             false,
                                             (rowC[i * 4 + j] < args.problem_size.m()) && colC_mask[0],
                                             1,
                                             MACA_ICMP_EQ);

            CVT_F32_TO_BF16(tempC[0], reinterpret_cast<uint *>(&out)[4], reinterpret_cast<uint *>(&out)[5]);
            CVT_F32_TO_BF16(tempC[1], reinterpret_cast<uint *>(&out)[6], reinterpret_cast<uint *>(&out)[7]);
            __builtin_mxc_stg_b64_predicator(Caddr + rowC[i * 4 + j] * args.problem_size.n() + colC[1],
                                             0,
                                             *(reinterpret_cast<uint64_t *>(&tempC)),
                                             true,
                                             false,
                                             false,
                                             (rowC[i * 4 + j] < args.problem_size.m()) && colC_mask[1],
                                             1,
                                             MACA_ICMP_EQ);
        }
    }
}

template <bool IsTopkLog2>
using DirectMoeGemmKernel = DirectMoeKernel<IsTopkLog2>;

template <typename Kernel>
inline dim3 get_grid_shape(typename Kernel::Arguments const &args) {
    const int grid_m = (args.moe_params.EM + kTileM - 1) / kTileM;
    const int group_bidx = std::max(1, std::min(8, (args.problem_size.m() / args.problem_size.batch() + kTileM - 1) / kTileM));
    const int grid_x = std::min(grid_m, group_bidx);
    const int grid_z = (grid_m + grid_x - 1) / grid_x;
    const int grid_y = (args.problem_size.n() + kTileN - 1) / kTileN;
    return dim3(grid_x, grid_y, grid_z);
}

template <typename Kernel>
inline Status launch(typename Kernel::Arguments const &args, mcStream_t stream = nullptr) {
    dim3 const block(kThreadCount, 1, 1);
    dim3 const grid = get_grid_shape<Kernel>(args);
    direct_moe_kernel<Kernel::kIsTopkLog2><<<grid, block, 0, stream>>>(args);
    return Status::kSuccess;
}

}  // namespace fused_moe_i8_tn
