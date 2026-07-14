import torch
import tilelang
import tilelang.language as T


_KERNEL_CACHE = {}
_WORKSPACE_CACHE = {}


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def _moe_forward_kernel(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    num_blocks_m,
):
    scale = 1.44269504
    dtype = T.float16
    accum_dtype = T.float32
    block_token = 128
    block_dhidden = 128
    block_dexpert = 128
    threads = 512
    num_stages = 1
    swizzle_panel = 8 if hidden <= 3584 else 10

    input_shape = (total_padded_tokens, hidden)
    intermediate_shape = (total_padded_tokens, intermediate)
    gate_shape = (num_experts, intermediate, hidden)
    up_shape = (num_experts, intermediate, hidden)
    down_shape = (num_experts, hidden, intermediate)

    @T.prim_func
    def kernel(
        stacked_expert_tokens: T.Tensor(input_shape, dtype),
        gate_w: T.Tensor(gate_shape, dtype),
        up_w: T.Tensor(up_shape, dtype),
        down_w: T.Tensor(down_shape, dtype),
        routed_expert_weights: T.Tensor((total_valid_tokens,), T.float32),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        up_logits: T.Tensor(intermediate_shape, dtype),
        out: T.Tensor(input_shape, dtype),
    ):
        with T.Kernel(
            num_blocks_m * 2,
            T.ceildiv(intermediate, block_dexpert),
            threads=threads,
        ) as (bx, by):
            input_shared = T.alloc_fragment(
                (block_token, block_dhidden), dtype=dtype
            )
            gate_shared = T.alloc_shared(
                (block_dexpert, block_dhidden), dtype=dtype
            )
            up_shared = T.alloc_shared(
                (block_dexpert, block_dhidden), dtype=dtype
            )
            gate_local = T.alloc_fragment(
                (block_token, block_dexpert), dtype=accum_dtype
            )
            up_local = T.alloc_fragment(
                (block_token, block_dexpert), dtype=accum_dtype
            )

            T.use_swizzle(swizzle_panel)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * block_token
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            actual_rows = T.max(
                0,
                T.min(
                    block_token,
                    group_size - (block_start - padded_start),
                ),
            )

            T.clear(gate_local)
            T.clear(up_local)

            for k in T.Pipelined(
                T.ceildiv(hidden, block_dhidden),
                num_stages=num_stages,
            ):
                T.copy(
                    stacked_expert_tokens[
                        block_start : block_start + block_token,
                        k * block_dhidden : (k + 1) * block_dhidden,
                    ],
                    input_shared,
                )
                T.copy(
                    gate_w[
                        expert_id,
                        by * block_dexpert : (by + 1) * block_dexpert,
                        k * block_dhidden : (k + 1) * block_dhidden,
                    ],
                    gate_shared,
                )
                T.copy(
                    up_w[
                        expert_id,
                        by * block_dexpert : (by + 1) * block_dexpert,
                        k * block_dhidden : (k + 1) * block_dhidden,
                    ],
                    up_shared,
                )
                T.gemm(
                    input_shared,
                    gate_shared,
                    gate_local,
                    transpose_B=True,
                )
                T.gemm(
                    input_shared,
                    up_shared,
                    up_local,
                    transpose_B=True,
                )

            for i, j in T.Parallel(block_token, block_dexpert):
                if i < actual_rows:
                    up_logits[
                        block_start + i,
                        by * block_dexpert + j,
                    ] = up_local[i, j] * gate_local[i, j] * (
                        1.0
                        / (
                            1.0
                            + T.exp2(-gate_local[i, j] * scale)
                        )
                    )

        with T.Kernel(
            num_blocks_m,
            T.ceildiv(hidden, block_dhidden),
            threads=threads,
        ) as (bx, by):
            up_shared = T.alloc_fragment(
                (block_token, block_dexpert), dtype=dtype
            )
            down_shared = T.alloc_shared(
                (block_dhidden, block_dexpert), dtype=dtype
            )
            out_local = T.alloc_fragment(
                (block_token, block_dhidden), dtype=accum_dtype
            )
            routed_weight_local = T.alloc_fragment(
                (block_token,), dtype=dtype
            )

            T.use_swizzle(swizzle_panel)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * block_token
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(
                0,
                T.min(block_token, group_size - token_offset),
            )

            T.clear(out_local)

            for i in T.Parallel(block_token):
                if i < actual_rows:
                    routed_weight_local[i] = routed_expert_weights[
                        raw_start + token_offset + i
                    ]

            for k in T.Pipelined(
                T.ceildiv(intermediate, block_dexpert),
                num_stages=num_stages,
            ):
                T.copy(
                    up_logits[
                        block_start : block_start + block_token,
                        k * block_dexpert : (k + 1) * block_dexpert,
                    ],
                    up_shared,
                )
                T.copy(
                    down_w[
                        expert_id,
                        by * block_dhidden : (by + 1) * block_dhidden,
                        k * block_dexpert : (k + 1) * block_dexpert,
                    ],
                    down_shared,
                )
                T.gemm(
                    up_shared,
                    down_shared,
                    out_local,
                    transpose_B=True,
                )

            for i, j in T.Parallel(block_token, block_dhidden):
                if i < actual_rows:
                    out[
                        block_start + i,
                        by * block_dhidden + j,
                    ] = out_local[i, j] * routed_weight_local[i]

    return kernel


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def _moe_forward_kernel_tail32(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    num_blocks_m,
):
    scale = 1.44269504
    dtype = T.float16
    accum_dtype = T.float32
    parent_token = 128
    tail_token = 64
    block_dhidden = 128
    block_dexpert = 128
    full_threads = 512
    tail_threads = 256
    num_stages = 1
    swizzle_panel = 8 if hidden <= 3584 else 10

    input_shape = (total_padded_tokens, hidden)
    intermediate_shape = (total_padded_tokens, intermediate)
    gate_shape = (num_experts, intermediate, hidden)
    up_shape = (num_experts, intermediate, hidden)
    down_shape = (num_experts, hidden, intermediate)

    @T.prim_func
    def kernel(
        stacked_expert_tokens: T.Tensor(input_shape, dtype),
        gate_w: T.Tensor(gate_shape, dtype),
        up_w: T.Tensor(up_shape, dtype),
        down_w: T.Tensor(down_shape, dtype),
        routed_expert_weights: T.Tensor((total_valid_tokens,), T.float32),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        up_logits: T.Tensor(intermediate_shape, dtype),
        out: T.Tensor(input_shape, dtype),
    ):
        # Full gate/up blocks retain the high-throughput 128-row tile.
        with T.Kernel(
            num_blocks_m * 2,
            T.ceildiv(intermediate, block_dexpert),
            threads=full_threads,
        ) as (bx, by):
            input_shared = T.alloc_fragment(
                (parent_token, block_dhidden), dtype=dtype
            )
            gate_shared = T.alloc_shared(
                (block_dexpert, block_dhidden), dtype=dtype
            )
            up_shared = T.alloc_shared(
                (block_dexpert, block_dhidden), dtype=dtype
            )
            gate_local = T.alloc_fragment(
                (parent_token, block_dexpert), dtype=accum_dtype
            )
            up_local = T.alloc_fragment(
                (parent_token, block_dexpert), dtype=accum_dtype
            )
            T.use_swizzle(swizzle_panel)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * parent_token
            padded_start = group_padded_offsets[expert_id]
            remaining = group_sizes[expert_id] - (block_start - padded_start)

            if remaining >= parent_token:
                T.clear(gate_local)
                T.clear(up_local)
                for k in T.Pipelined(
                    T.ceildiv(hidden, block_dhidden),
                    num_stages=num_stages,
                ):
                    T.copy(
                        stacked_expert_tokens[
                            block_start : block_start + parent_token,
                            k * block_dhidden : (k + 1) * block_dhidden,
                        ],
                        input_shared,
                    )
                    T.copy(
                        gate_w[
                            expert_id,
                            by * block_dexpert : (by + 1) * block_dexpert,
                            k * block_dhidden : (k + 1) * block_dhidden,
                        ],
                        gate_shared,
                    )
                    T.copy(
                        up_w[
                            expert_id,
                            by * block_dexpert : (by + 1) * block_dexpert,
                            k * block_dhidden : (k + 1) * block_dhidden,
                        ],
                        up_shared,
                    )
                    T.gemm(
                        input_shared,
                        gate_shared,
                        gate_local,
                        transpose_B=True,
                    )
                    T.gemm(
                        input_shared,
                        up_shared,
                        up_local,
                        transpose_B=True,
                    )
                for i, j in T.Parallel(parent_token, block_dexpert):
                    up_logits[
                        block_start + i,
                        by * block_dexpert + j,
                    ] = up_local[i, j] * gate_local[i, j] * (
                        1.0 / (1.0 + T.exp2(-gate_local[i, j] * scale))
                    )

        # Tail gate/up blocks split one padded parent block into two 64-row tiles.
        with T.Kernel(
            num_blocks_m * 2,
            T.ceildiv(intermediate, block_dexpert),
            threads=tail_threads,
        ) as (tbx, by):
            input_shared = T.alloc_fragment(
                (tail_token, block_dhidden), dtype=dtype
            )
            gate_shared = T.alloc_shared(
                (block_dexpert, block_dhidden), dtype=dtype
            )
            up_shared = T.alloc_shared(
                (block_dexpert, block_dhidden), dtype=dtype
            )
            gate_local = T.alloc_fragment(
                (tail_token, block_dexpert), dtype=accum_dtype
            )
            up_local = T.alloc_fragment(
                (tail_token, block_dexpert), dtype=accum_dtype
            )
            T.use_swizzle(swizzle_panel)

            parent_bx = tbx // 2
            tail_id = tbx % 2
            expert_id = group_idx_for_bx[parent_bx]
            parent_start = parent_bx * parent_token
            block_start = parent_start + tail_id * tail_token
            padded_start = group_padded_offsets[expert_id]
            parent_remaining = group_sizes[expert_id] - (
                parent_start - padded_start
            )
            actual_rows = T.max(
                0,
                T.min(tail_token, parent_remaining - tail_id * tail_token),
            )

            if parent_remaining < parent_token and actual_rows > 0:
                T.clear(gate_local)
                T.clear(up_local)
                for k in T.Pipelined(
                    T.ceildiv(hidden, block_dhidden),
                    num_stages=num_stages,
                ):
                    T.copy(
                        stacked_expert_tokens[
                            block_start : block_start + tail_token,
                            k * block_dhidden : (k + 1) * block_dhidden,
                        ],
                        input_shared,
                    )
                    T.copy(
                        gate_w[
                            expert_id,
                            by * block_dexpert : (by + 1) * block_dexpert,
                            k * block_dhidden : (k + 1) * block_dhidden,
                        ],
                        gate_shared,
                    )
                    T.copy(
                        up_w[
                            expert_id,
                            by * block_dexpert : (by + 1) * block_dexpert,
                            k * block_dhidden : (k + 1) * block_dhidden,
                        ],
                        up_shared,
                    )
                    T.gemm(
                        input_shared,
                        gate_shared,
                        gate_local,
                        transpose_B=True,
                    )
                    T.gemm(
                        input_shared,
                        up_shared,
                        up_local,
                        transpose_B=True,
                    )
                for i, j in T.Parallel(tail_token, block_dexpert):
                    if i < actual_rows:
                        up_logits[
                            block_start + i,
                            by * block_dexpert + j,
                        ] = up_local[i, j] * gate_local[i, j] * (
                            1.0
                            / (1.0 + T.exp2(-gate_local[i, j] * scale))
                        )

        # Full down blocks.
        with T.Kernel(
            num_blocks_m * 2,
            T.ceildiv(hidden, block_dhidden),
            threads=full_threads,
        ) as (bx, by):
            up_shared = T.alloc_fragment(
                (parent_token, block_dexpert), dtype=dtype
            )
            down_shared = T.alloc_shared(
                (block_dhidden, block_dexpert), dtype=dtype
            )
            out_local = T.alloc_fragment(
                (parent_token, block_dhidden), dtype=accum_dtype
            )
            routed_weight_local = T.alloc_fragment(
                (parent_token,), dtype=dtype
            )
            T.use_swizzle(swizzle_panel)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * parent_token
            padded_start = group_padded_offsets[expert_id]
            raw_start = group_offsets[expert_id]
            token_offset = block_start - padded_start
            remaining = group_sizes[expert_id] - token_offset

            if remaining >= parent_token:
                T.clear(out_local)
                for i in T.Parallel(parent_token):
                    routed_weight_local[i] = routed_expert_weights[
                        raw_start + token_offset + i
                    ]
                for k in T.Pipelined(
                    T.ceildiv(intermediate, block_dexpert),
                    num_stages=num_stages,
                ):
                    T.copy(
                        up_logits[
                            block_start : block_start + parent_token,
                            k * block_dexpert : (k + 1) * block_dexpert,
                        ],
                        up_shared,
                    )
                    T.copy(
                        down_w[
                            expert_id,
                            by * block_dhidden : (by + 1) * block_dhidden,
                            k * block_dexpert : (k + 1) * block_dexpert,
                        ],
                        down_shared,
                    )
                    T.gemm(
                        up_shared,
                        down_shared,
                        out_local,
                        transpose_B=True,
                    )
                for i, j in T.Parallel(parent_token, block_dhidden):
                    out[
                        block_start + i,
                        by * block_dhidden + j,
                    ] = out_local[i, j] * routed_weight_local[i]

        # Tail down blocks.
        with T.Kernel(
            num_blocks_m * 2,
            T.ceildiv(hidden, block_dhidden),
            threads=tail_threads,
        ) as (tbx, by):
            up_shared = T.alloc_fragment(
                (tail_token, block_dexpert), dtype=dtype
            )
            down_shared = T.alloc_shared(
                (block_dhidden, block_dexpert), dtype=dtype
            )
            out_local = T.alloc_fragment(
                (tail_token, block_dhidden), dtype=accum_dtype
            )
            routed_weight_local = T.alloc_fragment(
                (tail_token,), dtype=dtype
            )
            T.use_swizzle(swizzle_panel)

            parent_bx = tbx // 2
            tail_id = tbx % 2
            expert_id = group_idx_for_bx[parent_bx]
            parent_start = parent_bx * parent_token
            block_start = parent_start + tail_id * tail_token
            padded_start = group_padded_offsets[expert_id]
            raw_start = group_offsets[expert_id]
            parent_offset = parent_start - padded_start
            parent_remaining = group_sizes[expert_id] - parent_offset
            token_offset = parent_offset + tail_id * tail_token
            actual_rows = T.max(
                0,
                T.min(tail_token, parent_remaining - tail_id * tail_token),
            )

            if parent_remaining < parent_token and actual_rows > 0:
                T.clear(out_local)
                for i in T.Parallel(tail_token):
                    if i < actual_rows:
                        routed_weight_local[i] = routed_expert_weights[
                            raw_start + token_offset + i
                        ]
                for k in T.Pipelined(
                    T.ceildiv(intermediate, block_dexpert),
                    num_stages=num_stages,
                ):
                    T.copy(
                        up_logits[
                            block_start : block_start + tail_token,
                            k * block_dexpert : (k + 1) * block_dexpert,
                        ],
                        up_shared,
                    )
                    T.copy(
                        down_w[
                            expert_id,
                            by * block_dhidden : (by + 1) * block_dhidden,
                            k * block_dexpert : (k + 1) * block_dexpert,
                        ],
                        down_shared,
                    )
                    T.gemm(
                        up_shared,
                        down_shared,
                        out_local,
                        transpose_B=True,
                    )
                for i, j in T.Parallel(tail_token, block_dhidden):
                    if i < actual_rows:
                        out[
                            block_start + i,
                            by * block_dhidden + j,
                        ] = out_local[i, j] * routed_weight_local[i]

    return kernel


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def _moe_forward_kernel_all64(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    num_blocks_m,
):
    scale = 1.44269504
    dtype = T.float16
    accum_dtype = T.float32
    parent_token = 128
    block_token = 64
    block_dhidden = 128
    block_dexpert = 128
    threads = 512
    num_stages = 1
    swizzle_panel = 8 if hidden <= 3584 else 16
    down_pack = 2
    down_swizzle = 8
    gate_pack = 2

    input_shape = (total_padded_tokens, hidden)
    intermediate_shape = (total_padded_tokens, intermediate)
    gate_shape = (num_experts, intermediate, hidden)
    up_shape = (num_experts, intermediate, hidden)
    down_shape = (num_experts, hidden, intermediate)

    @T.prim_func
    def kernel(
        stacked_expert_tokens: T.Tensor(input_shape, dtype),
        gate_w: T.Tensor(gate_shape, dtype),
        up_w: T.Tensor(up_shape, dtype),
        down_w: T.Tensor(down_shape, dtype),
        routed_expert_weights: T.Tensor((total_valid_tokens,), T.float32),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        up_logits: T.Tensor(intermediate_shape, dtype),
        out: T.Tensor(input_shape, dtype),
    ):
        with T.Kernel(
            num_blocks_m * 2,
            T.ceildiv(intermediate, block_dexpert * gate_pack),
            threads=threads,
        ) as (bx, by):
            T.annotate_l2_hit_ratio({up_logits: 0.8})
            input_shared = T.alloc_fragment(
                (block_token, block_dhidden), dtype=dtype
            )
            gate_shared = T.alloc_shared(
                (block_dexpert, block_dhidden), dtype=dtype
            )
            up_shared = T.alloc_shared(
                (block_dexpert, block_dhidden), dtype=dtype
            )
            gate_local_0 = T.alloc_fragment(
                (block_token, block_dexpert), dtype=accum_dtype
            )
            gate_local_1 = T.alloc_fragment(
                (block_token, block_dexpert), dtype=accum_dtype
            )
            up_local_0 = T.alloc_fragment(
                (block_token, block_dexpert), dtype=accum_dtype
            )
            up_local_1 = T.alloc_fragment(
                (block_token, block_dexpert), dtype=accum_dtype
            )
            T.use_swizzle(swizzle_panel)

            parent_bx = bx // 2
            tile_id = bx % 2
            expert_id = group_idx_for_bx[parent_bx]
            block_start = parent_bx * parent_token + tile_id * block_token
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(
                0,
                T.min(block_token, group_sizes[expert_id] - token_offset),
            )

            if actual_rows > 0:
                T.clear(gate_local_0)
                T.clear(gate_local_1)
                T.clear(up_local_0)
                T.clear(up_local_1)
                for k in T.serial(T.ceildiv(hidden, block_dhidden)):
                    T.copy(
                        stacked_expert_tokens[
                            block_start : block_start + block_token,
                            k * block_dhidden : (k + 1) * block_dhidden,
                        ],
                        input_shared,
                    )
                    T.copy(
                        gate_w[
                            expert_id,
                            by * gate_pack * block_dexpert : (by * gate_pack + 1)
                            * block_dexpert,
                            k * block_dhidden : (k + 1) * block_dhidden,
                        ],
                        gate_shared,
                    )
                    T.copy(
                        up_w[
                            expert_id,
                            by * gate_pack * block_dexpert : (by * gate_pack + 1)
                            * block_dexpert,
                            k * block_dhidden : (k + 1) * block_dhidden,
                        ],
                        up_shared,
                    )
                    T.gemm(
                        input_shared,
                        gate_shared,
                        gate_local_0,
                        transpose_B=True,
                    )
                    T.gemm(
                        input_shared,
                        up_shared,
                        up_local_0,
                        transpose_B=True,
                    )
                    T.copy(
                        gate_w[
                            expert_id,
                            (by * gate_pack + 1)
                            * block_dexpert : (by * gate_pack + 2)
                            * block_dexpert,
                            k * block_dhidden : (k + 1) * block_dhidden,
                        ],
                        gate_shared,
                    )
                    T.copy(
                        up_w[
                            expert_id,
                            (by * gate_pack + 1)
                            * block_dexpert : (by * gate_pack + 2)
                            * block_dexpert,
                            k * block_dhidden : (k + 1) * block_dhidden,
                        ],
                        up_shared,
                    )
                    T.gemm(
                        input_shared,
                        gate_shared,
                        gate_local_1,
                        transpose_B=True,
                    )
                    T.gemm(
                        input_shared,
                        up_shared,
                        up_local_1,
                        transpose_B=True,
                    )
                for i, j in T.Parallel(block_token, block_dexpert):
                    if i < actual_rows:
                        up_logits[
                            block_start + i,
                            by * gate_pack * block_dexpert + j,
                        ] = up_local_0[i, j] * gate_local_0[i, j] * (
                            1.0
                            / (1.0 + T.exp2(-gate_local_0[i, j] * scale))
                        )
                        up_logits[
                            block_start + i,
                            (by * gate_pack + 1) * block_dexpert + j,
                        ] = up_local_1[i, j] * gate_local_1[i, j] * (
                            1.0
                            / (1.0 + T.exp2(-gate_local_1[i, j] * scale))
                        )

        with T.Kernel(
            num_blocks_m * 2,
            T.ceildiv(hidden, block_dhidden * down_pack),
            threads=threads,
        ) as (bx, by):
            T.annotate_l2_hit_ratio({up_logits: 0.8})
            up_shared = T.alloc_fragment(
                (block_token, block_dexpert), dtype=dtype
            )
            down_shared_0 = T.alloc_shared(
                (block_dhidden, block_dexpert), dtype=dtype
            )
            down_shared_1 = T.alloc_shared(
                (block_dhidden, block_dexpert), dtype=dtype
            )
            out_local_0 = T.alloc_fragment(
                (block_token, block_dhidden), dtype=accum_dtype
            )
            out_local_1 = T.alloc_fragment(
                (block_token, block_dhidden), dtype=accum_dtype
            )
            routed_weight_local = T.alloc_fragment(
                (block_token,), dtype=dtype
            )
            T.use_swizzle(down_swizzle)

            parent_bx = bx // 2
            tile_id = bx % 2
            expert_id = group_idx_for_bx[parent_bx]
            block_start = parent_bx * parent_token + tile_id * block_token
            padded_start = group_padded_offsets[expert_id]
            raw_start = group_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(
                0,
                T.min(block_token, group_sizes[expert_id] - token_offset),
            )

            if actual_rows > 0:
                T.clear(out_local_0)
                T.clear(out_local_1)
                for i in T.Parallel(block_token):
                    if i < actual_rows:
                        routed_weight_local[i] = routed_expert_weights[
                            raw_start + token_offset + i
                        ]
                for k in T.Pipelined(
                    T.ceildiv(intermediate, block_dexpert),
                    num_stages=num_stages,
                ):
                    T.copy(
                        up_logits[
                            block_start : block_start + block_token,
                            k * block_dexpert : (k + 1) * block_dexpert,
                        ],
                        up_shared,
                    )
                    T.copy(
                        down_w[
                            expert_id,
                            by * down_pack * block_dhidden : (by * down_pack + 1)
                            * block_dhidden,
                            k * block_dexpert : (k + 1) * block_dexpert,
                        ],
                        down_shared_0,
                    )
                    T.copy(
                        down_w[
                            expert_id,
                            (by * down_pack + 1)
                            * block_dhidden : (by * down_pack + 2)
                            * block_dhidden,
                            k * block_dexpert : (k + 1) * block_dexpert,
                        ],
                        down_shared_1,
                    )
                    T.gemm(
                        up_shared,
                        down_shared_0,
                        out_local_0,
                        transpose_B=True,
                    )
                    T.gemm(
                        up_shared,
                        down_shared_1,
                        out_local_1,
                        transpose_B=True,
                    )
                for i, j in T.Parallel(block_token, block_dhidden):
                    if i < actual_rows:
                        out[
                            block_start + i,
                            by * down_pack * block_dhidden + j,
                        ] = out_local_0[i, j] * routed_weight_local[i]
                        out[
                            block_start + i,
                            (by * down_pack + 1) * block_dhidden + j,
                        ] = out_local_1[i, j] * routed_weight_local[i]

    return kernel


def _get_kernel(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    num_blocks_m,
):
    key = (
        int(hidden),
        int(intermediate),
        int(num_experts),
        int(total_padded_tokens),
        int(total_valid_tokens),
        int(num_blocks_m),
    )
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = _moe_forward_kernel_all64(*key)
        _KERNEL_CACHE[key] = kernel
    return kernel


def _get_workspace(stacked_expert_tokens, intermediate):
    key = (
        int(stacked_expert_tokens.device.index or 0),
        int(stacked_expert_tokens.shape[0]),
        int(intermediate),
        str(stacked_expert_tokens.dtype),
    )
    up_logits = _WORKSPACE_CACHE.get(key)
    if up_logits is None:
        up_logits = torch.empty(
            (
                int(stacked_expert_tokens.shape[0]),
                int(intermediate),
            ),
            device=stacked_expert_tokens.device,
            dtype=stacked_expert_tokens.dtype,
        )
        _WORKSPACE_CACHE[key] = up_logits
    return up_logits


def run_kernel(
    stacked_expert_tokens,
    gate_w,
    up_w,
    down_w,
    routed_expert_weights,
    group_sizes,
    group_offsets,
    group_padded_offsets,
    group_idx_for_bx,
    out,
):
    hidden = int(stacked_expert_tokens.shape[1])
    intermediate = int(gate_w.shape[1])
    num_experts = int(gate_w.shape[0])
    total_padded_tokens = int(stacked_expert_tokens.shape[0])
    total_valid_tokens = int(routed_expert_weights.shape[0])
    num_blocks_m = int(group_idx_for_bx.shape[0])

    up_logits = _get_workspace(stacked_expert_tokens, intermediate)
    kernel = _get_kernel(
        hidden,
        intermediate,
        num_experts,
        total_padded_tokens,
        total_valid_tokens,
        num_blocks_m,
    )
    kernel(
        stacked_expert_tokens,
        gate_w,
        up_w,
        down_w,
        routed_expert_weights,
        group_sizes,
        group_offsets,
        group_padded_offsets,
        group_idx_for_bx,
        up_logits,
        out,
    )
