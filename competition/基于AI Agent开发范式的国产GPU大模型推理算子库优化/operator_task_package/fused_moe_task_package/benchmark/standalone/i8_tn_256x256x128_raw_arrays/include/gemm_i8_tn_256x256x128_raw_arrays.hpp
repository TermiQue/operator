#pragma once

#include "standalone_maca_kernel_utils.hpp"

namespace standalone_i8_tn_256x256x128_raw_arrays {

struct KernelConfig {
    static constexpr int kTileM = 256;
    static constexpr int kTileN = 256;
    static constexpr int kTileK = 128;
    static constexpr int kThreadCount = 512;
    static constexpr int kWaveSize = 64;
    static constexpr int kWaveCount = kThreadCount / kWaveSize;
    static constexpr int kRowsPerWaveGroup = 64;
    static constexpr int kColsPerWaveGroup = 128;
    static constexpr int kRowsPerMicroTile = 16;
    static constexpr int kColsPerMicroTile = 32;
    static constexpr int kColumnGroupsPerWave = 2;
    static constexpr int kColBlocksPerWaveGroup = 2;
    static constexpr int kRowBlocksPerWave = 4;
    static constexpr int kOutputVectorsPerMicroTile = 4;
    static constexpr int kOutputVectorsPerColBlock = kRowBlocksPerWave * kOutputVectorsPerMicroTile;
    static constexpr int kElementsPer128b = 16;
    static constexpr int kSharedBytesA = kTileM * kTileK;
    static constexpr int kSharedBytesB = kTileN * kTileK;
    static constexpr int kSmemSize = kSharedBytesA + kSharedBytesB;
};

using StoreVector = __NATIVE_VECTOR__(2, int32_t);
using LdsTypeI8Mma = __NATIVE_VECTOR__(4, int32_t);
using AbTypeI8Mma = int32_t;
using AccumTypeI8Mma = __NATIVE_VECTOR__(4, int32_t);

struct ThreadContext {
    int a_store_offset[4];
    int a_row_local[4];
    int a_load_k[4];
    int b_store_offset[4];
    int b_col_local[4];
    int b_load_k[4];
    int a_lds_offset[KernelConfig::kRowBlocksPerWave][2];
    int b_lds_offset[KernelConfig::kColumnGroupsPerWave][KernelConfig::kColBlocksPerWaveGroup][2][2];
};

__forceinline__ __device__ int swizzled_slot8(int row_or_col, int q) {
    return q ^ (row_or_col & 7);
}

__forceinline__ __device__ void clear_accumulators(
    AccumTypeI8Mma (&accum)[KernelConfig::kColumnGroupsPerWave]
                            [KernelConfig::kColBlocksPerWaveGroup]
                            [KernelConfig::kRowBlocksPerWave][2]) {
#pragma unroll
    for (int column_group = 0; column_group < KernelConfig::kColumnGroupsPerWave; ++column_group) {
#pragma unroll
        for (int col_block = 0; col_block < KernelConfig::kColBlocksPerWaveGroup; ++col_block) {
#pragma unroll
            for (int row_block = 0; row_block < KernelConfig::kRowBlocksPerWave; ++row_block) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
#pragma unroll
                    for (int i = 0; i < KernelConfig::kOutputVectorsPerMicroTile; ++i) {
                        accum[column_group][col_block][row_block][half][i] = 0;
                    }
                }
            }
        }
    }
}

__forceinline__ __device__ void build_thread_context(ThreadContext &ctx,
                                                     int tid,
                                                     int wave_row_group,
                                                     int wave_col_group,
                                                     int lane16,
                                                     int lane_q_block,
                                                     int K) {
    const int row_or_col_local = tid >> 3;
    const int slot8_phys = tid & 7;

#pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        const int row_or_col = pass * 64 + row_or_col_local;
        const int q = slot8_phys ^ (row_or_col & 7);
        ctx.a_store_offset[pass] = row_or_col * 128 + slot8_phys * KernelConfig::kElementsPer128b;
        ctx.a_row_local[pass] = row_or_col;
        ctx.a_load_k[pass] = q * KernelConfig::kElementsPer128b;
        ctx.b_store_offset[pass] = row_or_col * 128 + slot8_phys * KernelConfig::kElementsPer128b;
        ctx.b_col_local[pass] = row_or_col;
        ctx.b_load_k[pass] = q * KernelConfig::kElementsPer128b;
    }

    const int q_values[2] = {lane_q_block, lane_q_block + 4};

#pragma unroll
    for (int row_block = 0; row_block < KernelConfig::kRowBlocksPerWave; ++row_block) {
        const int a_row =
            wave_row_group * KernelConfig::kRowsPerWaveGroup + row_block * KernelConfig::kRowsPerMicroTile + lane16;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int slot8_phys_local = swizzled_slot8(a_row, q_values[half]);
            ctx.a_lds_offset[row_block][half] = a_row * 128 + slot8_phys_local * KernelConfig::kElementsPer128b;
        }
    }

    const int lds_k_b = (lane_q_block ^ (lane16 & 3)) * KernelConfig::kElementsPer128b;
#pragma unroll
    for (int column_group = 0; column_group < KernelConfig::kColumnGroupsPerWave; ++column_group) {
#pragma unroll
        for (int col_block = 0; col_block < KernelConfig::kColBlocksPerWaveGroup; ++col_block) {
            const int b_chunk = wave_col_group * 4 + column_group * 2 + col_block;
            const int b_col0 = b_chunk * KernelConfig::kColsPerMicroTile + lane16;
            const int b_col1 = b_col0 + 16;
            const int cols[2] = {b_col0, b_col1};
#pragma unroll
            for (int half = 0; half < 2; ++half) {
#pragma unroll
                for (int which = 0; which < 2; ++which) {
                    const int row_or_col = cols[which];
                    const int slot8_phys_local = swizzled_slot8(row_or_col, q_values[half]);
                    ctx.b_lds_offset[column_group][col_block][half][which] =
                        row_or_col * 128 + slot8_phys_local * KernelConfig::kElementsPer128b;
                }
            }
        }
    }
}

template <int Pass>
__forceinline__ __device__ void load_a_pass(int8_t *smem_a,
                                            const int8_t *a_ptr,
                                            int K,
                                            int global_row_base,
                                            int k_tile_base,
                                            ThreadContext const &ctx) {
    const int row = global_row_base + ctx.a_row_local[Pass];
    __builtin_mxc_ldg_b128_bsm(
        smem_a + ctx.a_store_offset[Pass],
        const_cast<void *>(reinterpret_cast<void const *>(
            a_ptr + static_cast<size_t>(row) * K + k_tile_base + ctx.a_load_k[Pass])),
        0,
        -1,
        true,
        true,
        false,
        true);
}

template <int Pass>
__forceinline__ __device__ void load_b_pass(int8_t *smem_b,
                                            const int8_t *b_ptr,
                                            int K,
                                            int global_col_base,
                                            int k_tile_base,
                                            ThreadContext const &ctx) {
    const int col = global_col_base + ctx.b_col_local[Pass];
    __builtin_mxc_ldg_b128_bsm(
        smem_b + ctx.b_store_offset[Pass],
        const_cast<void *>(reinterpret_cast<void const *>(
            b_ptr + static_cast<size_t>(col) * K + k_tile_base + ctx.b_load_k[Pass])),
        0,
        -1,
        true,
        true,
        false,
        true);
}

__forceinline__ __device__ void wait_for_tile_load() {
    standalone_arrive_gvmcnt(0);
    __builtin_mxc_barrier_inst();
}

__forceinline__ __device__ void mma_on_pack16(AccumTypeI8Mma &accum_left,
                                              AccumTypeI8Mma &accum_right,
                                              LdsTypeI8Mma const &a_pack,
                                              LdsTypeI8Mma const &b0_pack,
                                              LdsTypeI8Mma const &b1_pack) {
    AbTypeI8Mma const *a_frag = reinterpret_cast<AbTypeI8Mma const *>(&a_pack);
    AbTypeI8Mma const *b0_frag = reinterpret_cast<AbTypeI8Mma const *>(&b0_pack);
    AbTypeI8Mma const *b1_frag = reinterpret_cast<AbTypeI8Mma const *>(&b1_pack);
#pragma unroll
    for (int step = 0; step < 4; ++step) {
        accum_left = STANDALONE_BUILTIN_MMA_16X16X16_I8(a_frag[step], b0_frag[step], accum_left);
        accum_right = STANDALONE_BUILTIN_MMA_16X16X16_I8(a_frag[step], b1_frag[step], accum_right);
    }
}

__forceinline__ __device__ void consume_full_k128_from_shared(
    AccumTypeI8Mma (&accum)[KernelConfig::kColumnGroupsPerWave]
                            [KernelConfig::kColBlocksPerWaveGroup]
                            [KernelConfig::kRowBlocksPerWave][2],
    int8_t const *smem_a,
    int8_t const *smem_b,
    ThreadContext const &ctx) {
    LdsTypeI8Mma b0_pack_low[KernelConfig::kColumnGroupsPerWave][KernelConfig::kColBlocksPerWaveGroup];
    LdsTypeI8Mma b1_pack_low[KernelConfig::kColumnGroupsPerWave][KernelConfig::kColBlocksPerWaveGroup];
    LdsTypeI8Mma b0_pack_high[KernelConfig::kColumnGroupsPerWave][KernelConfig::kColBlocksPerWaveGroup];
    LdsTypeI8Mma b1_pack_high[KernelConfig::kColumnGroupsPerWave][KernelConfig::kColBlocksPerWaveGroup];

#pragma unroll
    for (int column_group = 0; column_group < KernelConfig::kColumnGroupsPerWave; ++column_group) {
#pragma unroll
        for (int col_block = 0; col_block < KernelConfig::kColBlocksPerWaveGroup; ++col_block) {
            STANDALONE_LDS(
                b0_pack_low[column_group][col_block],
                *const_cast<int8_t *>(smem_b + ctx.b_lds_offset[column_group][col_block][0][0]),
                LdsTypeI8Mma);
            STANDALONE_LDS(
                b1_pack_low[column_group][col_block],
                *const_cast<int8_t *>(smem_b + ctx.b_lds_offset[column_group][col_block][0][1]),
                LdsTypeI8Mma);
            STANDALONE_LDS(
                b0_pack_high[column_group][col_block],
                *const_cast<int8_t *>(smem_b + ctx.b_lds_offset[column_group][col_block][1][0]),
                LdsTypeI8Mma);
            STANDALONE_LDS(
                b1_pack_high[column_group][col_block],
                *const_cast<int8_t *>(smem_b + ctx.b_lds_offset[column_group][col_block][1][1]),
                LdsTypeI8Mma);
        }
    }

#pragma unroll
    for (int row_block = 0; row_block < KernelConfig::kRowBlocksPerWave; ++row_block) {
        LdsTypeI8Mma a_pack_low;
        LdsTypeI8Mma a_pack_high;
        STANDALONE_LDS(a_pack_low, *const_cast<int8_t *>(smem_a + ctx.a_lds_offset[row_block][0]), LdsTypeI8Mma);
        STANDALONE_LDS(a_pack_high, *const_cast<int8_t *>(smem_a + ctx.a_lds_offset[row_block][1]), LdsTypeI8Mma);

#pragma unroll
        for (int column_group = 0; column_group < KernelConfig::kColumnGroupsPerWave; ++column_group) {
#pragma unroll
            for (int col_block = 0; col_block < KernelConfig::kColBlocksPerWaveGroup; ++col_block) {
                mma_on_pack16(
                    accum[column_group][col_block][row_block][0],
                    accum[column_group][col_block][row_block][1],
                    a_pack_low,
                    b0_pack_low[column_group][col_block],
                    b1_pack_low[column_group][col_block]);
                mma_on_pack16(
                    accum[column_group][col_block][row_block][0],
                    accum[column_group][col_block][row_block][1],
                    a_pack_high,
                    b0_pack_high[column_group][col_block],
                    b1_pack_high[column_group][col_block]);
            }
        }
    }
}

__forceinline__ __device__ void store_accumulators_to_global(
    int32_t *D,
    int M,
    int N,
    int bidx,
    int bidy,
    int bidz,
    AccumTypeI8Mma const (&accum)[KernelConfig::kColumnGroupsPerWave]
                                  [KernelConfig::kColBlocksPerWaveGroup]
                                  [KernelConfig::kRowBlocksPerWave][2]) {
    const int row_limit = ((M - bidy * KernelConfig::kTileM) < KernelConfig::kTileM)
                              ? (M - bidy * KernelConfig::kTileM)
                              : KernelConfig::kTileM;
    const int col_limit = ((N - bidx * KernelConfig::kTileN) < KernelConfig::kTileN)
                              ? (N - bidx * KernelConfig::kTileN)
                              : KernelConfig::kTileN;
    const int tid = threadIdx.x;
    const int wave_idx = tid / KernelConfig::kWaveSize;
    const int lane_idx = tid % KernelConfig::kWaveSize;
    const int lane_col = lane_idx % 16;
    const int lane_row_group = lane_idx / 16;
    const int wave_row_base = (wave_idx / 2) * KernelConfig::kRowsPerWaveGroup;
    const int wave_col_base = (wave_idx % 2) * KernelConfig::kColsPerWaveGroup;

    int32_t *d_ptr = D + static_cast<size_t>(bidz) * M * N;

#pragma unroll
    for (int column_group = 0; column_group < KernelConfig::kColumnGroupsPerWave; ++column_group) {
#pragma unroll
        for (int col_block = 0; col_block < KernelConfig::kColBlocksPerWaveGroup; ++col_block) {
            const int microtile_col_base =
                wave_col_base + column_group * 64 + col_block * KernelConfig::kColsPerMicroTile;
            const int col0 = microtile_col_base + lane_col;
            const int col1 = col0 + 16;

#pragma unroll
            for (int row_block = 0; row_block < KernelConfig::kRowBlocksPerWave; ++row_block) {
                const int output_base =
                    column_group * (KernelConfig::kColBlocksPerWaveGroup * KernelConfig::kOutputVectorsPerColBlock) +
                    col_block * KernelConfig::kOutputVectorsPerColBlock +
                    row_block * KernelConfig::kOutputVectorsPerMicroTile;
#pragma unroll
                for (int i = 0; i < KernelConfig::kOutputVectorsPerMicroTile; ++i) {
                    const int row = wave_row_base + row_block * KernelConfig::kRowsPerMicroTile +
                                    lane_row_group * 4 + i;
                    if (row >= row_limit) {
                        continue;
                    }
                    const size_t row_offset = static_cast<size_t>(bidy * KernelConfig::kTileM + row) * N +
                                              bidx * KernelConfig::kTileN;
                    if (col0 < col_limit) {
                        d_ptr[row_offset + col0] = accum[column_group][col_block][row_block][0][i];
                    }
                    if (col1 < col_limit) {
                        d_ptr[row_offset + col1] = accum[column_group][col_block][row_block][1][i];
                    }
                }
            }
        }
    }
}

__global__ void gemm_i8_tn_256x256x128_raw_arrays_kernel(const int8_t *A,
                                                         const int8_t *B,
                                                         int32_t *D,
                                                         int M,
                                                         int N,
                                                         int K) {
    __shared__ int8_t smem_data[KernelConfig::kSmemSize];

    int8_t *smem_a = smem_data;
    int8_t *smem_b = smem_data + KernelConfig::kSharedBytesA;

    const int tid = threadIdx.x;
    const int bidx = blockIdx.y;
    const int bidy = blockIdx.x;
    const int bidz = blockIdx.z;

    const int wave_idx = tid / KernelConfig::kWaveSize;
    const int lane_idx = tid % KernelConfig::kWaveSize;
    const int wave_row_group = wave_idx / 2;
    const int wave_col_group = wave_idx % 2;
    const int lane16 = lane_idx % 16;
    const int lane_q_block = lane_idx / 16;
    const int global_row_base = bidy * KernelConfig::kTileM;
    const int global_col_base = bidx * KernelConfig::kTileN;

    const int8_t *a_ptr = A + static_cast<size_t>(bidz) * M * K;
    const int8_t *b_ptr = B + static_cast<size_t>(bidz) * N * K;

    AccumTypeI8Mma accum[KernelConfig::kColumnGroupsPerWave]
                        [KernelConfig::kColBlocksPerWaveGroup]
                        [KernelConfig::kRowBlocksPerWave][2];
    ThreadContext ctx;
    clear_accumulators(accum);
    build_thread_context(ctx, tid, wave_row_group, wave_col_group, lane16, lane_q_block, K);

    for (int k_tile = 0; k_tile < K; k_tile += KernelConfig::kTileK) {
        load_a_pass<0>(smem_a, a_ptr, K, global_row_base, k_tile, ctx);
        load_a_pass<1>(smem_a, a_ptr, K, global_row_base, k_tile, ctx);
        load_a_pass<2>(smem_a, a_ptr, K, global_row_base, k_tile, ctx);
        load_a_pass<3>(smem_a, a_ptr, K, global_row_base, k_tile, ctx);

        load_b_pass<0>(smem_b, b_ptr, K, global_col_base, k_tile, ctx);
        load_b_pass<1>(smem_b, b_ptr, K, global_col_base, k_tile, ctx);
        load_b_pass<2>(smem_b, b_ptr, K, global_col_base, k_tile, ctx);
        load_b_pass<3>(smem_b, b_ptr, K, global_col_base, k_tile, ctx);

        wait_for_tile_load();
        consume_full_k128_from_shared(accum, smem_a, smem_b, ctx);
        __syncthreadshared();
    }

    store_accumulators_to_global(D, M, N, bidx, bidy, bidz, accum);
}

}  // namespace standalone_i8_tn_256x256x128_raw_arrays
