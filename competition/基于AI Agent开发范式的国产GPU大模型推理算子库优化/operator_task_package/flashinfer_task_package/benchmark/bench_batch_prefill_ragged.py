"""
Benchmark script for BatchPrefillWithRaggedKVCacheWrapper
headdim configurations: [64,64], [128,128], [192,128], [256,256]
"""

import itertools
import pandas as pd
import torch

import flashinfer
from bench_common import (
    dtype,
    setup_workspace,
    run_with_profiler,
    get_csv_path,
    compute_reps,
)

target_kernels = [
    "BatchPrefillWithRaggedKVCacheKernel",
    "PersistentVariableLengthMergeStates",
]


def bench_batch_prefill_with_ragged_kv_cache(
    batch_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo,
    causal=True,
):
    """Benchmark BatchPrefillWithRaggedKVCacheWrapper for MLA"""
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
    kv_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * seq_len
    )

    q = torch.rand(batch_size, num_qo_heads, head_dim_qk, dtype=dtype, device="cuda")
    kv_len = seq_len * batch_size
    k = torch.rand(kv_len, num_kv_heads, head_dim_qk, dtype=dtype, device="cuda")
    v = torch.rand(kv_len, num_kv_heads, head_dim_vo, dtype=dtype, device="cuda")

    workspace_buffer = setup_workspace()
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD", backend="auto"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    reps = compute_reps(batch_size, seq_len, head_dim_qk + head_dim_vo, base_reps=100)
    ms = run_with_profiler(
        lambda: wrapper.run(q, k, v), target_kernels=target_kernels, reps=reps
    )

    io = (
        q.numel() * q.element_size()
        + k.numel() * k.element_size()
        + v.numel() * v.element_size()
    )

    flops = (
        batch_size
        * seq_len
        * seq_len
        * num_qo_heads
        * (head_dim_qk + head_dim_vo)
        * (1 if causal else 2)
    )

    return ms, io, flops


def run_benchmark():
    records = []

    # headdim combinations: [qk, vo]
    head_dim_configs = [(128, 128), (192, 128), (256, 256)]
    batch_sizes = [1, 4, 16, 64]
    seq_lens = [1024, 4096, 8192, 16384]

    api_name = "BatchPrefillWithRaggedKVCacheWrapper"
    test_cases = list(itertools.product(head_dim_configs, batch_sizes, seq_lens))
    total_cases = len(test_cases)

    print(f"[{api_name}] Starting benchmark, total cases: {total_cases}")
    for idx, ((head_dim_qk, head_dim_vo), bs, sl) in enumerate(test_cases, 1):
        num_qo_heads = 32
        num_kv_heads = 4
        ms, io, flops = bench_batch_prefill_with_ragged_kv_cache(
            bs, sl, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo
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
                "head_dim_qk": head_dim_qk,
                "head_dim_vo": head_dim_vo,
                "time_ms": ms,
                "bandwidth_GB_s": bw,
                "tflops": tflops,
            }
        )
        print(
            f"  [{idx}/{total_cases}] bs={bs}, sl={sl}, hd=[{head_dim_qk},{head_dim_vo}]: {ms:.3f}ms, {bw:.2f} GB/s, {tflops:.2f} TFLOPs"
        )

    return records


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    torch.random.manual_seed(42)

    records = run_benchmark()
    df = pd.DataFrame(records)
    csv_path = get_csv_path("BatchPrefillWithRaggedKVCacheWrapper")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
