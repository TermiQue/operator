from flash_attn.flash_attn_interface import flash_attn_with_kvcache
import torch
import math
from einops import rearrange
from datetime import datetime
import csv


def run_with_profiler(fn, warmup=10, reps=100, print_result=False, target_kernels=None):
    """Run function with torch.profiler and return sum of specific kernel times in ms"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()

    if print_result:
        print(prof.key_averages().table(sort_by="device_time", row_limit=20))

    if target_kernels is None:
        target_kernels = []

    kernel_times_us = 0.0
    for evt in prof.key_averages():
        if any(k in evt.key for k in target_kernels):
            kernel_times_us += evt.device_time

    ms = kernel_times_us / 1e3
    return ms


def calc_bandwidth(batch_size, seqlen_q, seqlen_k, num_heads, num_heads_k, headdim, dtype, ms):
    """Calculate bandwidth in GB/s"""
    bytes_per_elem = 2 if dtype == torch.bfloat16 else 4
    q_bytes = batch_size * seqlen_q * num_heads * headdim * bytes_per_elem
    kv_bytes = batch_size * seqlen_k * num_heads_k * headdim * bytes_per_elem * 2
    total_bytes = q_bytes + kv_bytes
    bw_gb_s = (total_bytes / 1e9) / (ms / 1e3)
    return bw_gb_s


def benchmark_kvcache(batch_size, seqlen_k, seqlen_q, num_heads, num_heads_k, headdim, page_block_size, device, dtype=torch.bfloat16, causal=False):
    num_blocks = math.ceil(seqlen_k / page_block_size) * batch_size * 3
    num_blocks = max(1024, num_blocks)
    paged_kv_block_size = page_block_size

    nheads = num_heads
    nheads_k = num_heads_k
    d = headdim

    torch.random.manual_seed(0)
    window_size = (-1, -1)

    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)

    k_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )
    v_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )
    block_table = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=batch_size,
    )

    cache_seqlens = torch.full((batch_size,), seqlen_k, dtype=torch.int32, device=device)

    def run_fn():
        flash_attn_with_kvcache(
            q, k_cache_paged, v_cache_paged, None, None,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=None,
            block_table=block_table,
            causal=causal,
            window_size=window_size,
            rotary_interleaved=False,
            alibi_slopes=None,
            num_splits=1,
        )

    return run_fn


def main():
    headdims = [256]
    page_block_size = 16
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    seq_lens_kv = [512, 1024, 2048, 4096, 8192, 16384]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    causal = False
    warmup = 10
    repeat = 100

    num_heads = 8
    num_heads_k = 8
    seqlen_q = 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"benchmark_kvcache_{timestamp}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "seq_len_kv", "heads", "headdim", "time_ms", "bandwidth_GB_s"])
        print(f"{'batch_size':>10} {'seq_len_kv':>12} {'heads':>6} {'headdim':>8} {'time_ms':>10} {'bandwidth_GB_s':>15}")
        print("-" * 75)
        for headdim in headdims:
            for seqlen_k in seq_lens_kv:
                for batch_size in batch_sizes:
                    try:
                        run_fn = benchmark_kvcache(
                            batch_size=batch_size,
                            seqlen_k=seqlen_k,
                            seqlen_q=seqlen_q,
                            num_heads=num_heads,
                            num_heads_k=num_heads_k,
                            headdim=headdim,
                            page_block_size=page_block_size,
                            device=device,
                            dtype=dtype,
                            causal=causal,
                        )
                        ms = run_with_profiler(run_fn, warmup=warmup, reps=repeat, target_kernels=["flash"])
                        bw = calc_bandwidth(batch_size, seqlen_q, seqlen_k, num_heads, num_heads_k, headdim, dtype, ms)
                        writer.writerow([batch_size, seqlen_k, num_heads, headdim, f"{ms:.4f}", f"{bw:.2f}"])
                        print(f"{batch_size:>10} {seqlen_k:>12} {num_heads:>6} {headdim:>8} {ms:>10.4f} {bw:>15.2f}")
                    except Exception as e:
                        writer.writerow([batch_size, seqlen_k, num_heads, headdim, "OOM", "OOM"])
                        print(f"{batch_size:>10} {seqlen_k:>12} {num_heads:>6} {headdim:>8} {'OOM':>10} {'OOM':>15}  # {e}")

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
