"""
Common utilities for FlashInfer benchmarks
"""

import os
import random
from datetime import datetime
import numpy as np
import torch

page_block_size = 16
dtype = torch.bfloat16


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_csv_path(prefix):
    """Generate CSV path in current execution directory with timestamp"""
    return f"{prefix}_{get_timestamp()}.csv"


def generate_random_seqlens(batch_size, min_len=1024, max_len=16384):
    """Generate random sequence lengths simulating real LLM workloads"""
    return [random.randint(min_len, max_len) for _ in range(batch_size)]


def setup_workspace():
    """Create workspace buffer for FlashInfer"""
    return torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")


def setup_paged_kv_indptr(batch_size, seq_lens):
    """Setup paged KV cache indptr and last_page_len"""
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)
    seq_lens_blocks = torch.ceil(seq_lens_tensor / page_block_size).int()
    kv_indptr = torch.cat([torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0).int()
    num_blocks = kv_indptr[-1].item()
    last_page_len = (seq_lens_tensor - 1) % page_block_size + 1
    return kv_indptr, last_page_len, num_blocks


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

    # Sum device_time of specific kernels
    if target_kernels is None:
        target_kernels = []

    kernel_times_us = 0.0
    for evt in prof.key_averages():
        if any(k in evt.key for k in target_kernels):
            kernel_times_us += evt.device_time

    ms = kernel_times_us / 1e3
    return ms


def compute_reps(batch_size, seq_len, head_dim, base_reps=100):
    """Dynamically compute repetition count based on workload size"""
    # Estimate workload: batch_size * seq_len * head_dim
    workload = batch_size * seq_len * head_dim
    if workload < 1e5:  # tiny workload
        return base_reps
    elif workload < 1e6:  # small workload
        return base_reps // 2
    elif workload < 1e7:  # medium workload
        return base_reps // 4
    elif workload < 1e8:  # large workload
        return base_reps // 8
    elif workload < 1e9:  # very large workload
        return base_reps // 16
    else:  # huge workload
        return base_reps // 32