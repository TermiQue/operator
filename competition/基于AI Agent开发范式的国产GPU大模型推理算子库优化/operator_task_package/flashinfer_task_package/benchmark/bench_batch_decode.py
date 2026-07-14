"""
Benchmark script for BatchDecodeWithPagedKVCacheWrapper
seq_len_q=1 (decode mode), seq_len_kv from 1K to 16K
"""

import itertools
import pandas as pd
import torch

import flashinfer
from bench_common import dtype, page_block_size, setup_workspace, setup_paged_kv_indptr, run_with_profiler, get_csv_path, compute_reps

target_kernels = ["BatchPrefillWithPagedKVCacheKernel"]


def bench_batch_decode(
    batch_size,
    seq_len_kv,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_block_size,
):
    """Benchmark BatchDecodeWithPagedKVCacheWrapper"""
    seq_lens = [seq_len_kv] * batch_size

    kv_indptr, last_page_len, num_blocks = setup_paged_kv_indptr(batch_size, seq_lens)

    q = torch.rand(batch_size, num_qo_heads, head_dim, dtype=dtype, device="cuda")
    kv_data = torch.randn(num_blocks, 2, page_block_size, num_kv_heads, head_dim, dtype=dtype, device="cuda")

    workspace_buffer = setup_workspace()
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD", use_tensor_cores=True
    )
    wrapper.plan(
        kv_indptr.to("cuda"),
        torch.arange(num_blocks, dtype=torch.int32, device="cuda"),
        last_page_len.to("cuda"),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_block_size,
        data_type=dtype,
        q_data_type=dtype,
    )

    reps = compute_reps(batch_size, seq_len_kv, head_dim, base_reps=100)
    ms = run_with_profiler(lambda: wrapper.run(q, kv_data), target_kernels=target_kernels, reps=reps)

    io = q.numel() * q.element_size() + kv_data.numel() * kv_data.element_size()
    flops = 2 * batch_size * seq_len_kv * num_qo_heads * num_kv_heads * head_dim
    return ms, io, flops


def run_benchmark():
    records = []

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    head_dims = [64, 128, 256]
    seq_lens_kv = [512, 1024, 2048, 4096, 8192, 16384]

    api_name = "BatchDecodeWithPagedKVCacheWrapper"
    test_cases = list(itertools.product(batch_sizes, seq_lens_kv, head_dims))
    total_cases = len(test_cases)

    print(f"[{api_name}] Starting benchmark, total cases: {total_cases}")
    print(f"  seq_len_q=1 (decode mode), causal=False")
    for idx, (bs, sl_kv, hd) in enumerate(test_cases, 1):
        num_qo_heads = 32
        num_kv_heads = 8 if hd == 64 else 4
        ms, io, flops = bench_batch_decode(bs, sl_kv, num_qo_heads, num_kv_heads, hd, page_block_size)
        bw = io / ms / 1e6
        tflops = flops / ms / 1e9
        records.append({
            "api": api_name,
            "batch_size": bs,
            "seq_len_q": 1,
            "seq_len_kv": sl_kv,
            "num_qo_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": hd,
            "time_ms": ms,
            "bandwidth_GB_s": bw,
            "tflops": tflops,
        })
        print(f"  [{idx}/{total_cases}] bs={bs}, kv_len={sl_kv}, hd={hd}: {ms:.3f}ms, {bw:.2f} GB/s, {tflops:.2f} TFLOPs")

    return records


if __name__ == "__main__":
    import numpy as np
    np.random.seed(42)
    torch.random.manual_seed(42)

    records = run_benchmark()
    df = pd.DataFrame(records)
    csv_path = get_csv_path("BatchDecodeWithPagedKVCacheWrapper")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")