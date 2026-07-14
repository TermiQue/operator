import argparse
import importlib
import json
import math
import os

import torch


run_kernel = importlib.import_module(
    os.environ.get("SUBMISSION_MODULE", "submission")
).run_kernel


def make_case(
    hidden, intermediate, num_experts, total_valid, seed, random_groups=False
):
    torch.manual_seed(seed)
    if random_groups:
        assignments = torch.randint(num_experts, (total_valid,), device="cpu")
        sizes = torch.bincount(assignments, minlength=num_experts).tolist()
    else:
        base = total_valid // num_experts
        extra = total_valid % num_experts
        sizes = [base + (1 if i < extra else 0) for i in range(num_experts)]

    raw_offsets = [0]
    padded_offsets = [0]
    block_to_expert = []
    for expert_id, size in enumerate(sizes):
        raw_offsets.append(raw_offsets[-1] + size)
        padded_size = math.ceil(size / 128) * 128
        padded_offsets.append(padded_offsets[-1] + padded_size)
        block_to_expert.extend([expert_id] * (padded_size // 128))

    device = torch.device("cuda")
    dtype = torch.float16
    total_padded = padded_offsets[-1]

    x = torch.randn(
        (total_padded, hidden), device=device, dtype=dtype
    ).contiguous()
    gate = torch.empty(
        (num_experts, intermediate, hidden), device=device, dtype=dtype
    ).uniform_(-0.02, 0.02)
    up = torch.empty_like(gate).uniform_(-0.02, 0.02)
    down = torch.empty(
        (num_experts, hidden, intermediate), device=device, dtype=dtype
    ).uniform_(-0.02, 0.02)
    route = torch.rand((total_valid,), device=device, dtype=torch.float32)
    group_sizes = torch.tensor(sizes, device=device, dtype=torch.int32)
    group_offsets = torch.tensor(
        raw_offsets, device=device, dtype=torch.int32
    )
    group_padded_offsets = torch.tensor(
        padded_offsets, device=device, dtype=torch.int32
    )
    group_idx_for_bx = torch.tensor(
        block_to_expert, device=device, dtype=torch.int32
    )
    out = torch.zeros((total_padded, hidden), device=device, dtype=dtype)

    return (
        x,
        gate,
        up,
        down,
        route,
        group_sizes,
        group_offsets,
        group_padded_offsets,
        group_idx_for_bx,
        out,
    )


def reference(case):
    (
        x,
        gate,
        up,
        down,
        route,
        group_sizes,
        group_offsets,
        group_padded_offsets,
        _,
        out,
    ) = case
    expected = torch.zeros_like(out)
    for expert_id in range(group_sizes.numel()):
        size = int(group_sizes[expert_id].item())
        raw_start = int(group_offsets[expert_id].item())
        padded_start = int(group_padded_offsets[expert_id].item())
        x_e = x[padded_start : padded_start + size].float()
        gate_e = x_e @ gate[expert_id].float().transpose(0, 1)
        up_e = x_e @ up[expert_id].float().transpose(0, 1)
        hidden_e = torch.nn.functional.silu(gate_e) * up_e
        y_e = hidden_e @ down[expert_id].float().transpose(0, 1)
        y_e *= route[raw_start : raw_start + size].unsqueeze(1)
        expected[padded_start : padded_start + size] = y_e.half()
    return expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--valid", type=int, default=568)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=81394)
    parser.add_argument("--random-groups", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    case = make_case(
        args.hidden,
        args.intermediate,
        args.experts,
        args.valid,
        args.seed,
        args.random_groups,
    )
    expected = reference(case) if args.check else None

    for _ in range(args.warmup):
        run_kernel(*case)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        run_kernel(*case)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / args.iters

    result = {
        "hidden": args.hidden,
        "intermediate": args.intermediate,
        "experts": args.experts,
        "total_valid": args.valid,
        "total_padded": int(case[0].shape[0]),
        "mean_ms": elapsed_ms,
    }
    if expected is not None:
        actual = case[-1]
        diff = (actual.float() - expected.float()).abs()
        result["max_abs_error"] = float(diff.max().item())
        result["mean_abs_error"] = float(diff.mean().item())
        result["allclose"] = bool(
            torch.allclose(actual, expected, rtol=0.05, atol=0.05)
        )

    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
