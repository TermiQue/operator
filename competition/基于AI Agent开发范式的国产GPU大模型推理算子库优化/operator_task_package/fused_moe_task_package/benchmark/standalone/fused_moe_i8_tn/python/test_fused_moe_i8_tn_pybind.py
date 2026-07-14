import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np

from fused_moe_i8_tn_triton import run_fused_moe_i8_tn_triton, triton_backend_available


K_NUM_EXPERTS = 2
K_TILE_M = 128
K_N = 128
K_K = 128


def load_extension():
    build_dir = Path(__file__).resolve().parents[1] / "build"
    candidates = sorted(build_dir.glob("fused_moe_i8_tn_pybind*.so"))
    if not candidates:
        raise FileNotFoundError(f"no built extension found under {build_dir}")
    module_path = candidates[0]
    spec = importlib.util.spec_from_file_location("fused_moe_i8_tn_pybind", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fill_inputs(num_tokens: int, topk: int, tile_experts: list[int]):
    total_rows = num_tokens * topk

    a = np.empty((num_tokens, K_K), dtype=np.int8)
    scale_a = np.empty((num_tokens,), dtype=np.float32)
    moe_weights = np.empty((total_rows,), dtype=np.float32)
    token_ids = np.empty((total_rows,), dtype=np.int32)
    expert_ids = np.asarray(tile_experts, dtype=np.int32)

    for row in range(num_tokens):
        for kk in range(K_K):
            a[row, kk] = ((row * 13 + kk * 7 + topk * 5 + 3) % 11) - 5
        scale_a[row] = 0.125 + 0.015625 * ((row + topk) % 7)

    for routed_row in range(total_rows):
        token_ids[routed_row] = routed_row
        moe_weights[routed_row] = 0.5 + 0.03125 * ((routed_row + topk) % 5)

    b = np.empty((K_NUM_EXPERTS, K_N, K_K), dtype=np.int8)
    scale_b = np.empty((K_NUM_EXPERTS, K_N), dtype=np.float32)
    for expert in range(K_NUM_EXPERTS):
        for col in range(K_N):
            scale_b[expert, col] = 0.25 + 0.03125 * ((expert * 3 + col + topk) % 9)
            for kk in range(K_K):
                b[expert, col, kk] = ((expert * 17 + col * 5 + kk * 3 + topk) % 9) - 4

    return a, b, scale_a, scale_b, moe_weights, token_ids, expert_ids


def reference_fused_moe(a, b, scale_a, scale_b, moe_weights, token_ids, expert_ids, topk):
    total_rows = token_ids.shape[0]
    out = np.zeros((total_rows, K_N), dtype=np.float32)
    for routed_row in range(total_rows):
        token = token_ids[routed_row] // topk
        tile_idx = routed_row // K_TILE_M
        expert = expert_ids[tile_idx]
        row_scale = scale_a[token] * moe_weights[routed_row]
        for col in range(K_N):
            acc = 0
            for kk in range(K_K):
                acc += int(a[token, kk]) * int(b[expert, col, kk])
            out[routed_row, col] = np.float32(acc * row_scale * scale_b[expert, col])
    return out


def run_pybind_backend(module, a, b, scale_a, scale_b, moe_weights, token_ids, expert_ids, topk):
    return module.run_fused_moe_i8_tn(
        a,
        b,
        scale_a,
        scale_b.reshape(-1),
        moe_weights,
        token_ids,
        expert_ids,
        topk,
        K_NUM_EXPERTS,
        int(os.environ.get("MCTLASS_PY_DEVICE_ID", "0")),
    )


def run_reference_backend(a, b, scale_a, scale_b, moe_weights, token_ids, expert_ids, topk):
    return reference_fused_moe(a, b, scale_a, scale_b, moe_weights, token_ids, expert_ids, topk)


def run_case(backend: str, backend_fn, tag: str, num_tokens: int, topk: int, em: int, tile_experts: list[int]):
    assert em == num_tokens * topk
    a, b, scale_a, scale_b, moe_weights, token_ids, expert_ids = fill_inputs(num_tokens, topk, tile_experts)
    expected = reference_fused_moe(a, b, scale_a, scale_b, moe_weights, token_ids, expert_ids, topk)
    got = backend_fn(a, b, scale_a, scale_b, moe_weights, token_ids, expert_ids, topk)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-2)
    print(
        f"{backend}:{tag} passed: rows={got.shape[0]}, cols={got.shape[1]}, "
        f"sample C[0]={got.reshape(-1)[0]}, C[last]={got.reshape(-1)[-1]}"
    )


def resolve_backends(requested_backend: str):
    backends = []

    if requested_backend in {"pybind", "all"}:
        module = load_extension()
        backends.append(("pybind", lambda a, b, scale_a, scale_b, moe_weights, token_ids, expert_ids, topk: run_pybind_backend(
            module, a, b, scale_a, scale_b, moe_weights, token_ids, expert_ids, topk
        )))

    if requested_backend in {"reference", "all"}:
        backends.append(("reference", run_reference_backend))

    if requested_backend in {"triton", "all"}:
        available, reason = triton_backend_available()
        if available:
            backends.append(("triton", run_fused_moe_i8_tn_triton))
        elif requested_backend == "triton":
            raise RuntimeError(f"Triton backend is unavailable: {reason}")
        else:
            print(f"skip triton backend: {reason}")

    return backends


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("pybind", "triton", "reference", "all"),
        default=os.environ.get("MCTLASS_FUSED_MOE_BACKEND", "pybind"),
    )
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
            run_case(backend, backend_fn, *case)


if __name__ == "__main__":
    main()
