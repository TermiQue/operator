# DeepSeek MLA Decode 提交说明

## 当前结果

| Status | Score | Time | Memory | Platform |
| --- | ---: | ---: | ---: | --- |
| Accepted | 49.5 | 23 ms | 22.2 G | TileLang Maca C500 / 10.1 K |

得分说明：

- 50 分左右基本对应和题目 baseline 的加速比约为 `1:1`。
- 当前 `49.5` 可以理解为 baseline 档位附近的 Accepted 结果。
- 该结果主要用于确认提交接口、TileLang kernel 调用和输出正确性已经跑通。
- 本文档只记录提交模板、改算子位置和当前结果，不展开进一步算子优化。

## 提交文件

提交文件为：

```text
race_tests/mla/submission.py
```

评测只要求 Python 文件中暴露 `run_kernel` 函数，函数名、参数顺序必须和题目一致：

```python
def run_kernel(
    q,
    q_pe,
    kv,
    k_pe,
    output,
    batch,
    heads,
    kv_heads,
    kv_ctx,
    dim,
    pe_dim,
):
    ...
```

提交时使用 `submission.py` 的内容即可，不需要提交 benchmark、reference 或测试脚本。

## 算子来源

当前提交模板基于 `race_tests/mla/test_tilelang_mla.py` 中的 `flashattn` 算子封装而来。

题目计算语义为：

```text
score = (q @ kv^T + q_pe @ k_pe^T) / sqrt(576)
attention = softmax(score, dim=-1)
output = attention @ kv
```

固定约束：

```text
dim = 512
pe_dim = 64
kv_heads = 1
heads = 16
```

## 模板结构

`submission.py` 里主要有三部分：

```text
flashattn(...)     TileLang MLA kernel
_get_kernel(...)   按 shape 缓存编译后的 kernel
run_kernel(...)    OJ 调用入口，写入 output
```

`run_kernel` 不做同步，不分配最终输出，只负责取得缓存 kernel 并调用：

```python
kernel = _get_kernel(...)
kernel(q, q_pe, kv, k_pe, output)
```

## 改算子的位置

只改 MLA 算子时，主要看两个位置：

```text
race_tests/mla/submission.py
```

1. `flashattn(...)`
   - TileLang kernel 主体。
   - QK、QK_pe、online softmax、PV 都在这里。

2. `_get_kernel(...)`
   - 设置 `block_n`、`block_h`、`num_split`。
   - 控制不同 shape 的 kernel 缓存 key。

一般不要改 `run_kernel` 的函数签名；评测器按固定签名调用。

## 完整算子代码

下面是当前 `race_tests/mla/submission.py` 的完整提交代码：

```python
import tilelang
import tilelang.language as T


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def flashattn(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, block_N, block_H, num_split, softmax_scale):
    scale = float(softmax_scale * 1.44269504)
    dtype = T.float16
    accum_dtype = T.float32
    kv_group_num = heads // kv_head_num
    valid_block_h = min(block_H, kv_group_num)
    assert kv_head_num == 1, "kv_head_num must be 1"

    @T.prim_func
    def main_split(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        glse = T.alloc_global([batch, heads, num_split], dtype)
        output_partial = T.alloc_global([batch, heads, num_split, dim], dtype)

        with T.Kernel(batch, heads // min(block_H, kv_group_num), num_split, threads=256) as (bid, hid, bz):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
            KV_shared = T.alloc_shared([block_N, dim], dtype)
            K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
            O_shared = T.alloc_shared([block_H, dim], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)
            cur_kv_head = hid // (kv_group_num // block_H)

            T.use_swizzle(10)
            T.copy(Q[bid, hid * valid_block_h : (hid + 1) * valid_block_h, :], Q_shared)
            T.copy(Q_pe[bid, hid * valid_block_h : (hid + 1) * valid_block_h, :], Q_pe_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(seqlen_kv // num_split, block_N)
            for k in T.Pipelined(loop_range, num_stages=2):
                kv_start = (seqlen_kv // num_split) * bz + k * block_N
                kv_end = (seqlen_kv // num_split) * bz + (k + 1) * block_N
                T.copy(KV[bid, kv_start:kv_end, cur_kv_head, :], KV_shared)
                T.copy(K_pe[bid, kv_start:kv_end, cur_kv_head, :], K_pe_shared)

                T.clear(acc_s)
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)

                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_shared)
                T.copy(S_shared, acc_s_cast)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(acc_s_cast, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

            for i, j in T.Parallel(block_H, dim):
                acc_o[i, j] /= logsum[i]
            for i in T.Parallel(block_H):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

            T.copy(logsum, glse[bid, hid * valid_block_h : (hid + 1) * valid_block_h, bz])
            T.copy(acc_o, O_shared)
            T.copy(O_shared, output_partial[bid, hid * valid_block_h : (hid + 1) * valid_block_h, bz, :])

        with T.Kernel(heads, batch, threads=128) as (hid, bz):
            po_local = T.alloc_fragment([dim], dtype)
            o_accum_local = T.alloc_fragment([dim], accum_dtype)
            lse_local_split = T.alloc_var(accum_dtype)
            lse_logsum_local = T.alloc_var(accum_dtype)
            lse_max_local = T.alloc_var(accum_dtype)
            scale_local = T.alloc_var(accum_dtype)

            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            lse_max_local = -T.infinity(accum_dtype)

            for k in T.serial(num_split):
                lse_max_local = T.max(lse_max_local, glse[bz, hid, k])
            for k in T.Pipelined(num_split, num_stages=1):
                lse_local_split = glse[bz, hid, k]
                lse_logsum_local += T.exp2(lse_local_split - lse_max_local)
            lse_logsum_local = T.log2(lse_logsum_local) + lse_max_local

            for k in T.serial(num_split):
                for i in T.Parallel(dim):
                    po_local[i] = output_partial[bz, hid, k, i]
                lse_local_split = glse[bz, hid, k]
                scale_local = T.exp2(lse_local_split - lse_logsum_local)
                for i in T.Parallel(dim):
                    o_accum_local[i] += po_local[i] * scale_local

            for i in T.Parallel(dim):
                Output[bz, hid, i] = o_accum_local[i]

    @T.prim_func
    def main_no_split(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        with T.Kernel(heads // min(block_H, kv_group_num), batch, threads=128) as (hid, bid):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
            KV_shared = T.alloc_shared([block_N, dim], dtype)
            K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
            O_shared = T.alloc_shared([block_H, dim], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)
            cur_kv_head = hid // (kv_group_num // block_H)

            T.copy(Q[bid, hid * valid_block_h : (hid + 1) * valid_block_h, :], Q_shared)
            T.copy(Q_pe[bid, hid * valid_block_h : (hid + 1) * valid_block_h, :], Q_pe_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(seqlen_kv, block_N)
            for k in T.Pipelined(loop_range, num_stages=0):
                T.copy(KV[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], KV_shared)
                T.copy(K_pe[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], K_pe_shared)
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)

                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_shared)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

            for i, j in T.Parallel(block_H, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bid, hid * valid_block_h : (hid + 1) * valid_block_h, :])

    if num_split > 1:
        return main_split
    return main_no_split


_KERNEL_CACHE = {}


def _get_kernel(batch, heads, kv_heads, kv_ctx, dim, pe_dim):
    block_n = 32
    block_h = min(16, heads // kv_heads)
    num_split = 1
    softmax_scale = (dim + pe_dim) ** -0.5
    key = (batch, heads, kv_heads, kv_ctx, dim, pe_dim, block_n, block_h, num_split)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = flashattn(batch, heads, kv_heads, kv_ctx, dim, pe_dim, block_n, block_h, num_split, softmax_scale)
        _KERNEL_CACHE[key] = kernel
    return kernel


def run_kernel(
    q,
    q_pe,
    kv,
    k_pe,
    output,
    batch,
    heads,
    kv_heads,
    kv_ctx,
    dim,
    pe_dim,
):
    kernel = _get_kernel(int(batch), int(heads), int(kv_heads), int(kv_ctx), int(dim), int(pe_dim))
    kernel(q, q_pe, kv, k_pe, output)
```

## 本地验证

本地验证时需要使用已经编译好的 TileLang，并把仓库根目录和 TVM Python 路径加入 `PYTHONPATH`。

在仓库根目录执行：

```bash
TILELANG_CACHE_DIR=/tmp/tilelang-cache \
MACA_PATH=${MACA_PATH:-/opt/maca} \
LD_LIBRARY_PATH="$(pwd)/build/lib:${MACA_PATH:-/opt/maca}/lib:${MACA_PATH:-/opt/maca}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}" \
PATH="${MACA_PATH:-/opt/maca}/bin:${MACA_PATH:-/opt/maca}/mxgpu_llvm/bin:${PATH}" \
PYTHONPATH="$(pwd):$(pwd)/3rdparty/tvm/python:$(pwd)/race_tests/mla:${PYTHONPATH}" \
python race_tests/mla/test_tilelang_mla.py \
  --no-json \
  --batch 1 \
  --heads 16 \
  --kv_heads 1 \
  --kv_ctx 2048 \
  --dim 512 \
  --pe_dim 64
```

已用 `submission.run_kernel` 验证过：

```text
kv_ctx=2048  allclose True
kv_ctx=8192  allclose True
```

## 注意事项

- `run_kernel` 内不要调用 `torch.cuda.synchronize()`。
- `output` 是评测器传入的缓冲区，必须原地写入。
- 当前文档只说明提交模板和改算子入口，不涉及进一步算子优化。
