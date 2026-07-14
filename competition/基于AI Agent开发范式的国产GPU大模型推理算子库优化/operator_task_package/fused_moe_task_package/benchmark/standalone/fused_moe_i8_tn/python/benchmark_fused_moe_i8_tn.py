import argparse
import time

from test_fused_moe_i8_tn_pybind import fill_inputs, resolve_backends


K_N = 128
K_K = 128


def compute_tops(rows: int, cols: int, k_dim: int, avg_ms: float) -> float:
    if avg_ms <= 0.0:
        return 0.0
    operations = 2.0 * float(rows) * float(cols) * float(k_dim)
    return operations / (avg_ms * 1.0e9)


def benchmark_backend_case(backend: str, backend_fn, tag: str, num_tokens: int, topk: int, em: int, tile_experts: list[int], warmup: int, iters: int):
    a, b, scale_a, scale_b, moe_weights, token_ids, expert_ids = fill_inputs(num_tokens, topk, tile_experts)

    for _ in range(warmup):
        backend_fn(a, b, scale_a, scale_b, moe_weights, token_ids, expert_ids, topk)

    start = time.perf_counter()
    for _ in range(iters):
        backend_fn(a, b, scale_a, scale_b, moe_weights, token_ids, expert_ids, topk)
    elapsed_s = time.perf_counter() - start

    avg_ms = elapsed_s * 1000.0 / iters
    tops = compute_tops(em, K_N, K_K, avg_ms)
    print(
        f"{backend}:{tag} benchmark: avg_ms={avg_ms:.6f}, "
        f"TOPS={tops:.6f}, warmup={warmup}, iters={iters}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("pybind", "triton", "reference", "all"),
        default="pybind",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    cases = [
        ("fused_moe_i8_tn_topk1", 256, 1, 256, [0, 1]),
        ("fused_moe_i8_tn_topk2", 256, 2, 512, [0, 1, 1, 0]),
        ("fused_moe_i8_tn_topk3", 128, 3, 384, [0, 1, 0]),
    ]
    backends = resolve_backends(args.backend)
    for backend, backend_fn in backends:
        for case in cases:
            benchmark_backend_case(backend, backend_fn, *case, warmup=args.warmup, iters=args.iters)


if __name__ == "__main__":
    main()
