from __future__ import annotations

from typing import Any

import numpy as np


K_TILE_M = 128
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32
GROUP_SIZE_M = 8

_TRITON = None
_TL = None
triton = None
tl = None


def _require_triton_runtime() -> tuple[Any, Any, Any]:
    global _TRITON, _TL, triton, tl
    try:
        import torch
        import triton as triton_mod
        import triton.language as tl_mod
    except ImportError as exc:
        raise RuntimeError("torch and triton are required for the Triton backend") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("the Triton backend currently requires a CUDA-capable torch runtime")

    _TRITON = triton_mod
    _TL = tl_mod
    triton = triton_mod
    tl = tl_mod
    return torch, triton_mod, tl_mod


def triton_backend_available() -> tuple[bool, str]:
    try:
        _require_triton_runtime()
    except RuntimeError as exc:
        return False, str(exc)
    return True, ""


def _build_blocked_routing(token_ids: np.ndarray, expert_ids: np.ndarray, num_experts: int, block_size_m: int):
    total_rows = int(token_ids.shape[0])
    routed_rows_per_expert = [[] for _ in range(num_experts)]

    for routed_row in range(total_rows):
        tile_idx = routed_row // K_TILE_M
        expert = int(expert_ids[tile_idx])
        routed_rows_per_expert[expert].append(routed_row)

    sorted_routed_rows: list[int] = []
    block_expert_ids: list[int] = []
    invalid_row = total_rows

    for expert, rows in enumerate(routed_rows_per_expert):
        if not rows:
            continue
        sorted_routed_rows.extend(rows)
        padded = (-len(rows)) % block_size_m
        if padded:
            sorted_routed_rows.extend([invalid_row] * padded)
        block_count = (len(rows) + padded) // block_size_m
        block_expert_ids.extend([expert] * block_count)

    num_tokens_post_padded = len(sorted_routed_rows)
    return (
        np.asarray(sorted_routed_rows, dtype=np.int32),
        np.asarray(block_expert_ids, dtype=np.int32),
        np.asarray([num_tokens_post_padded], dtype=np.int32),
    )


def _ensure_triton_symbols():
    if _TRITON is None or _TL is None:
        _require_triton_runtime()
    return _TRITON, _TL


def _get_fused_moe_kernel():
    triton, tl = _ensure_triton_symbols()

    @triton.jit
    def _fused_moe_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        b_bias_ptr,
        scale_a_ptr,
        scale_b_ptr,
        moe_weights_ptr,
        sorted_routed_rows_ptr,
        block_expert_ids_ptr,
        num_tokens_post_padded_ptr,
        n_dim,
        k_dim,
        em,
        num_valid_tokens,
        stride_am,
        stride_ak,
        stride_be,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        stride_bbe,
        stride_bbn,
        group_n: tl.constexpr,
        group_k: tl.constexpr,
        naive_block_assignment: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        SPLIT_K: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
        use_fp8_w8a8: tl.constexpr,
        use_int8_w8a8: tl.constexpr,
        use_int8_w8a16: tl.constexpr,
        per_channel_quant: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(em, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(n_dim, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return

        if not naive_block_assignment:
            offs_token_id = pid_m * BLOCK_SIZE_M + offs
            offs_token = tl.load(sorted_routed_rows_ptr + offs_token_id)
        else:
            offs_token = tl.where(
                offs == 0,
                pid_m,
                num_valid_tokens,
            )

        offs_token = offs_token.to(tl.int64)
        token_mask = offs_token < num_valid_tokens

        off_experts = tl.load(block_expert_ids_ptr + pid_m).to(tl.int64)
        if off_experts == -1:
            zero_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
            zero_offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            zero_c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * zero_offs_cn[None, :]
            zero_c_mask = token_mask[:, None] & (zero_offs_cn[None, :] < n_dim)
            tl.store(zero_c_ptrs, zero_acc, mask=zero_c_mask)
            return

        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % n_dim
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        if use_int8_w8a16:
            b_scale_ptrs = scale_b_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            b_scale = tl.load(b_scale_ptrs)

        if use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                a_scale_ptrs = scale_a_ptr + (offs_token // top_k) * stride_asm
                offs_bsn = offs_bn // group_n
                b_scale_ptrs = scale_b_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            elif per_channel_quant:
                b_scale_ptrs = scale_b_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
                b_scale = tl.load(b_scale_ptrs)
                a_scale_ptrs = scale_a_ptr + (offs_token // top_k) * stride_asm
                a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
            else:
                a_scale = tl.load(scale_a_ptr)
                b_scale = tl.load(scale_b_ptr + off_experts)

        if HAS_BIAS:
            bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
            bias = tl.load(bias_ptrs, mask=(offs_bn < n_dim), other=0.0)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(k_dim, BLOCK_SIZE_K)):
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < k_dim - k * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_dim - k * BLOCK_SIZE_K, other=0.0)
            if use_int8_w8a16:
                accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
            elif use_fp8_w8a8 or use_int8_w8a8:
                if group_k > 0 and group_n > 0:
                    k_start = k * BLOCK_SIZE_K
                    offs_ks = k_start // group_k
                    a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0)
                    b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
                else:
                    if use_fp8_w8a8:
                        accumulator = tl.dot(a, b, acc=accumulator)
                    else:
                        accumulator += tl.dot(a, b)
            else:
                accumulator += tl.dot(a, b)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if use_int8_w8a16:
            accumulator = accumulator * b_scale
        elif (use_fp8_w8a8 or use_int8_w8a8) and not (group_k > 0 and group_n > 0):
            accumulator = accumulator * a_scale * b_scale

        if HAS_BIAS:
            accumulator += bias[None, :]

        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(
                moe_weights_ptr + offs_token,
                mask=token_mask,
                other=0,
            )
            accumulator *= moe_weight[:, None]

        accumulator = accumulator.to(compute_type)

        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < n_dim)
        tl.store(c_ptrs, accumulator, mask=c_mask)

    return _fused_moe_kernel


def run_fused_moe_i8_tn_triton(
    a: np.ndarray,
    b_col_major: np.ndarray,
    scale_a: np.ndarray,
    scale_b: np.ndarray,
    moe_weights: np.ndarray,
    token_ids: np.ndarray,
    expert_ids: np.ndarray,
    topk: int,
    device: str = "cuda",
) -> np.ndarray:
    # Adapted from vLLM's fused MoE Triton path:
    # https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py
    torch, triton, tl = _require_triton_runtime()

    if a.ndim != 2:
        raise ValueError("a must be a 2D array")
    if b_col_major.ndim != 3:
        raise ValueError("b_col_major must be a 3D array")
    if scale_a.ndim != 1:
        raise ValueError("scale_a must be a 1D array")
    if scale_b.ndim != 2:
        raise ValueError("scale_b must be a 2D array")
    if moe_weights.ndim != 1:
        raise ValueError("moe_weights must be a 1D array")
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be a 1D array")
    if expert_ids.ndim != 1:
        raise ValueError("expert_ids must be a 1D array")
    if topk <= 0:
        raise ValueError("topk must be > 0")

    num_tokens, k_dim = a.shape
    num_experts, n_dim, b_k = b_col_major.shape
    total_rows = moe_weights.shape[0]

    if b_k != k_dim:
        raise ValueError("B K dimension must match A K dimension")
    if scale_a.shape[0] != num_tokens:
        raise ValueError("scale_a size mismatch")
    if scale_b.shape != (num_experts, n_dim):
        raise ValueError("scale_b shape mismatch")
    if token_ids.shape[0] != total_rows:
        raise ValueError("token_ids size mismatch")
    if total_rows != num_tokens * topk:
        raise ValueError("moe_weights size must equal num_tokens * topk")
    if total_rows % K_TILE_M != 0:
        raise ValueError("num_tokens * topk must be a multiple of 128")
    if expert_ids.shape[0] != total_rows // K_TILE_M:
        raise ValueError("expert_ids size mismatch")

    sorted_routed_rows, block_expert_ids, num_tokens_post_padded = _build_blocked_routing(
        token_ids, expert_ids, num_experts, BLOCK_SIZE_M
    )

    a_t = torch.as_tensor(np.ascontiguousarray(a), device=device, dtype=torch.int8)
    b_t = torch.as_tensor(np.ascontiguousarray(b_col_major), device=device, dtype=torch.int8)
    scale_a_t = torch.as_tensor(np.ascontiguousarray(scale_a), device=device, dtype=torch.float32)
    scale_b_t = torch.as_tensor(np.ascontiguousarray(scale_b), device=device, dtype=torch.float32)
    moe_weights_t = torch.as_tensor(np.ascontiguousarray(moe_weights), device=device, dtype=torch.float32)
    sorted_routed_rows_t = torch.as_tensor(sorted_routed_rows, device=device, dtype=torch.int32)
    block_expert_ids_t = torch.as_tensor(block_expert_ids, device=device, dtype=torch.int32)
    num_tokens_post_padded_t = torch.as_tensor(num_tokens_post_padded, device=device, dtype=torch.int32)
    dummy_bias_t = torch.zeros((num_experts, n_dim), device=device, dtype=torch.float32)
    out_t = torch.empty((total_rows, n_dim), device=device, dtype=torch.float32)

    fused_moe_kernel = _get_fused_moe_kernel()

    grid = (triton.cdiv(int(num_tokens_post_padded[0]), BLOCK_SIZE_M) * triton.cdiv(n_dim, BLOCK_SIZE_N),)
    fused_moe_kernel[grid](
        a_t,
        b_t,
        out_t,
        dummy_bias_t,
        scale_a_t,
        scale_b_t,
        moe_weights_t,
        sorted_routed_rows_t,
        block_expert_ids_t,
        num_tokens_post_padded_t,
        n_dim,
        k_dim,
        int(num_tokens_post_padded[0]),
        total_rows,
        a_t.stride(0),
        a_t.stride(1),
        b_t.stride(0),
        b_t.stride(2),
        b_t.stride(1),
        out_t.stride(0),
        out_t.stride(1),
        scale_a_t.stride(0),
        0,
        scale_b_t.stride(0),
        0,
        scale_b_t.stride(1),
        dummy_bias_t.stride(0),
        dummy_bias_t.stride(1),
        group_n=0,
        group_k=0,
        naive_block_assignment=False,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        SPLIT_K=1,
        MUL_ROUTED_WEIGHT=True,
        top_k=topk,
        compute_type=tl.float32,
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        use_int8_w8a16=False,
        per_channel_quant=True,
        HAS_BIAS=False,
    )

    return out_t.cpu().numpy()
