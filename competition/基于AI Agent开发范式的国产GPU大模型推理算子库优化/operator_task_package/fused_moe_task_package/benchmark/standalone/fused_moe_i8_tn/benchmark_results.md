# Fused MoE i8 TN Benchmark Results

Remote run environment:

- host: `10.2.118.21`
- repo: `/home/acl_dnn/mcTlass`
- binary: `standalone/fused_moe_i8_tn/build/fused_moe_i8_tn_example`

Benchmark command used:

```bash
MCTLASS_MOE_WARMUP=2 MCTLASS_MOE_ITERS=5 ./standalone/fused_moe_i8_tn/build/fused_moe_i8_tn_example
```

Results:

```text
Benchmark config: warmup=2, iters=5
fused_moe_i8_tn_topk1 passed: rows=256, topk=1, N=128, K=128, sample C[0]=0.695312, C[last]=-0.445312, max_abs=0
fused_moe_i8_tn_topk1 benchmark: avg_ms=0.0137216, TOPS=0.611343, warmup=2, iters=5
fused_moe_i8_tn_topk2 passed: rows=512, topk=2, N=128, K=128, sample C[0]=-0.578125, C[last]=-0.498047, max_abs=0
fused_moe_i8_tn_topk2 benchmark: avg_ms=0.0121856, TOPS=1.37681, warmup=2, iters=5
fused_moe_i8_tn_topk3 passed: rows=384, topk=3, N=128, K=128, sample C[0]=-1.08594, C[last]=-0.335938, max_abs=0
fused_moe_i8_tn_topk3 benchmark: avg_ms=0.0124416, TOPS=1.01136, warmup=2, iters=5
```

Notes:

- current numbers are from a short sanity benchmark, not a long stabilized run
- default code path still uses `warmup=20` and `iters=100` if env vars are not set
