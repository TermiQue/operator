"""
Benchmark script for BatchPrefillWithPagedKVCacheWrapper
headdim: 64/128/256
"""

import itertools
import pandas as pd
import torch

import flashinfer
from bench_common import (
    dtype,
    setup_workspace,
    setup_paged_kv_indptr,
    run_with_profiler,
    get_csv_path,
    compute_reps,
)

target_kernels = ["BatchPrefillWithPagedKVCacheKernel"]


def bench_batch_prefill_with_paged_kv_cache(
    batch_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    causal=True,
):
    """Benchmark BatchPrefillWithPagedKVCacheWrapper"""
    q_lens = [seq_len] * batch_size
    kv_lens = [seq_len] * batch_size

    qo_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(torch.tensor(q_lens), 0)], dim=0
    ).int()
    kv_indptr, last_page_len, num_blocks = setup_paged_kv_indptr(batch_size, kv_lens)

    q = torch.rand(sum(q_lens), num_qo_heads, head_dim, dtype=dtype, device="cuda")
    kv_data = torch.randn(
        num_blocks, 2, 16, num_kv_heads, head_dim, dtype=dtype, device="cuda"
    )

    workspace_buffer = setup_workspace()
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD", backend="auto"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        torch.arange(num_blocks, dtype=torch.int32, device="cuda"),
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        16,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    reps = compute_reps(batch_size, seq_len, head_dim, base_reps=100)
    ms = run_with_profiler(
        lambda: wrapper.run(q, kv_data), target_kernels=target_kernels, reps=reps
    )

    io = q.numel() * q.element_size() + kv_data.numel() * kv_data.element_size()
    # Attention FLOPs calculation:
    # - causal=True: triangular pattern
    # - causal=False: full attention
    flops = (
        2
        * batch_size
        * seq_len
        * seq_len
        * num_qo_heads
        * head_dim
        * (1 if causal else 2)
    )
    return ms, io, flops


def run_benchmark():
    records = []

    batch_sizes = [1, 4, 16, 64]
    seq_lens = [1024, 4096, 8192, 16384]
    head_dims = [128, 256]

    api_name = "BatchPrefillWithPagedKVCacheWrapper"
    test_cases = list(itertools.product(head_dims, batch_sizes, seq_lens))
    total_cases = len(test_cases)

    print(f"[{api_name}] Starting benchmark, total cases: {total_cases}")
    for idx, (head_dim, bs, sl) in enumerate(test_cases, 1):
        num_qo_heads = 32
        num_kv_heads = 8 if head_dim == 64 else 4
        ms, io, flops = bench_batch_prefill_with_paged_kv_cache(
            bs, sl, num_qo_heads, num_kv_heads, head_dim
        )
        bw = io / ms / 1e6
        tflops = flops / ms / 1e9
        records.append(
            {
                "api": api_name,
                "batch_size": bs,
                "seq_len": sl,
                "num_qo_heads": num_qo_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "time_ms": ms,
                "bandwidth_GB_s": bw,
                "tflops": tflops,
            }
        )
        print(
            f"  [{idx}/{total_cases}] bs={bs}, sl={sl}, hd={head_dim}: {ms:.3f}ms, {bw:.2f} GB/s, {tflops:.2f} TFLOPs"
        )

    return records


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    torch.random.manual_seed(42)

    records = run_benchmark()
    df = pd.DataFrame(records)
    csv_path = get_csv_path("BatchPrefillWithPagedKVCacheWrapper")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
