"""
Benchmark script for BatchMLAPagedAttentionWrapper
headdim: ckv=512, kpe=64 (DeepSeek MLA configuration)
"""

import itertools
import pandas as pd
import torch

import flashinfer
from bench_common import dtype, page_block_size, setup_workspace, run_with_profiler, get_csv_path, compute_reps

target_kernels = ["BatchMLAPagedAttentionKernel"]


def bench_batch_mla_paged_attention(
    batch_size,
    seq_len,
    num_heads,
    head_dim_ckv,
    head_dim_kpe,
):
    """Benchmark BatchMLAPagedAttentionWrapper for DeepSeek MLA"""
    # MLA decode mode: q has length 1, not seq_len
    q_nope = torch.randn(batch_size, num_heads, head_dim_ckv, dtype=dtype, device="cuda")
    q_pe = torch.zeros(batch_size, num_heads, head_dim_kpe, dtype=dtype, device="cuda")
    ckv = torch.randn(batch_size * seq_len, 1, head_dim_ckv, dtype=dtype, device="cuda")
    kpe = torch.zeros(batch_size * seq_len, 1, head_dim_kpe, dtype=dtype, device="cuda")

    sm_scale = 1.0 / ((head_dim_ckv + head_dim_kpe) ** 0.5)

    # q_indptr for decode: each query has length 1
    q_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * seq_len
    kv_indices = torch.arange(0, batch_size * seq_len, dtype=torch.int32, device="cuda")
    kv_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")

    page_size = 1  # MLA uses page_size=1

    workspace_buffer = setup_workspace()
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace_buffer, backend="auto")
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        False,  # causal
        sm_scale,
        q_nope.dtype,
        ckv.dtype,
    )

    reps = compute_reps(batch_size, seq_len, head_dim_ckv + head_dim_kpe, base_reps=100)
    ms = run_with_profiler(lambda: wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False), target_kernels=target_kernels, reps=reps)

    io = sum([t.numel() * t.element_size() for t in [q_nope, q_pe, ckv, kpe]])
    # MLA FLOPs: 2 * batch_size * num_heads * (2 * head_dim_ckv + head_dim_kpe) * seq_len
    flops = 2 * batch_size * num_heads * (2 * head_dim_ckv + head_dim_kpe) * seq_len
    return ms, io, flops


def run_benchmark():
    records = []

    # MLA configuration - same as DeepSeek
    head_dim_ckv = 512
    head_dim_kpe = 64
    batch_sizes = [1, 4, 16, 64]
    seq_lens = [1024, 4096, 8192, 16384]
    num_heads_list = [64, 128]

    api_name = "BatchMLAPagedAttentionWrapper"
    test_cases = list(itertools.product(num_heads_list, batch_sizes, seq_lens))
    total_cases = len(test_cases)

    print(f"[{api_name}] Starting benchmark, total cases: {total_cases}")
    for idx, (num_heads, bs, sl) in enumerate(test_cases, 1):
        ms, io, flops = bench_batch_mla_paged_attention(bs, sl, num_heads, head_dim_ckv, head_dim_kpe)
        bw = io / ms / 1e6
        tflops = flops / ms / 1e9
        records.append({
            "api": api_name,
            "batch_size": bs,
            "seq_len": sl,
            "num_heads": num_heads,
            "head_dim_ckv": head_dim_ckv,
            "head_dim_kpe": head_dim_kpe,
            "time_ms": ms,
            "bandwidth_GB_s": bw,
            "tflops": tflops,
        })
        print(f"  [{idx}/{total_cases}] bs={bs}, sl={sl}, num_heads={num_heads}: {ms:.3f}ms, {bw:.2f} GB/s, {tflops:.2f} TFLOPs")

    return records


if __name__ == "__main__":
    import numpy as np
    np.random.seed(42)
    torch.random.manual_seed(42)

    records = run_benchmark()
    df = pd.DataFrame(records)
    csv_path = get_csv_path("BatchMLAPagedAttentionWrapper")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")