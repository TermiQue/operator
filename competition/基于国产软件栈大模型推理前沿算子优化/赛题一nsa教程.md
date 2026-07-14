# Native Sparse Attention 提交说明

## 当前结果

| Status | Score | Case | Time | Memory | Platform | Submit Time |
| --- | ---: | --- | ---: | ---: | --- | --- |
| Accepted | 53.64 | muxitest001 | 619 us | 22.2 G | TileLang Maca C500 / 4.9 K | 06/18 15:53:16 |

得分说明：

- 50 分左右基本对应和题目 baseline 的加速比约为 `1:1`。
- 当前 `53.64` 是 baseline 档位附近、略高于 50 分线的 Accepted 结果。
- 该结果主要用于确认提交接口、TileLang kernel 调用和输出正确性已经跑通。
- 本文档只记录提交模板、改算子位置和当前结果，不展开进一步算子优化。

## 提交文件

提交文件为：

```text
race_tests/nsa/submission.py
```

评测只要求 Python 文件中暴露 `run_kernel` 函数，函数名、参数顺序必须和题目一致：

```python
def run_kernel(
    q,
    k,
    v,
    block_indices,
    output,
    B,
    seq_len,
    H,
    HQ,
    D,
    S,
    block_size,
    is_causal,
):
    ...
```

提交时使用 `submission.py` 的内容即可，不需要提交 benchmark、reference 或测试脚本。

## 算子来源

当前提交模板基于 `race_tests/nsa/test_tilelang_nsa_fwd.py` 中的 `native_sparse_attention` 算子封装而来。

题目计算语义为：

```text
score = q @ k_selected^T / sqrt(D)
attention = softmax(score)
output = attention @ v_selected
```

其中 `block_indices` 指定每个 query token 选中的 KV block，`is_causal=1` 时需要屏蔽未来 token。

## 模板结构

`submission.py` 里主要有三部分：

```text
native_sparse_attention(...)  TileLang NSA kernel
_get_kernel(...)              按 shape 缓存编译后的 kernel
run_kernel(...)               OJ 调用入口，写入 output
```

`run_kernel` 不做同步，不分配最终输出，只负责取得缓存 kernel 并调用：

```python
kernel = _get_kernel(...)
kernel(q, k, v, block_indices, output)
```

## 改算子的位置

只改 NSA 算子时，主要看两个位置：

```text
race_tests/nsa/submission.py
```

1. `native_sparse_attention(...)`
   - TileLang kernel 主体。
   - block 读取、causal mask、online softmax、PV 都在这里。

2. `_get_kernel(...)`
   - 设置 shape cache key。
   - 控制不同 `(B, seq_len, H, HQ, D, S, block_size, is_causal)` 的 kernel 缓存。

一般不要改 `run_kernel` 的函数签名；评测器按固定签名调用。

## 完整算子代码

下面是当前 `race_tests/nsa/submission.py` 的完整提交代码：

```python
import tilelang
import tilelang.language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def native_sparse_attention(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
    scale = float((dim**-0.5) * 1.44269504)
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    block_indices_shape = [batch, seq_len, head_kv, selected_blocks]

    dtype = T.float16
    accum_dtype = T.float32
    block_t = min(128, tilelang.math.next_power_of_2(dim))

    assert tilelang.cdiv(dim, block_t) == 1, "The key dimension can not be larger than 128"

    S = selected_blocks
    G = groups
    BS = block_size
    BK = BV = block_t
    num_stages = 2
    threads = 64

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        BlockIndices: T.Tensor(block_indices_shape, T.int32),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(seq_len, tilelang.cdiv(dim, BV), batch * head_kv, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([G, BK], dtype)
            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            O_shared = T.alloc_shared([G, BV], dtype)

            acc_s = T.alloc_fragment([G, BS], accum_dtype)
            acc_s_cast = T.alloc_fragment([G, BS], dtype)
            acc_o = T.alloc_fragment([G, BV], accum_dtype)
            scores_max = T.alloc_fragment([G], accum_dtype)
            scores_max_prev = T.alloc_fragment([G], accum_dtype)
            scores_scale = T.alloc_fragment([G], accum_dtype)
            scores_sum = T.alloc_fragment([G], accum_dtype)
            logsum = T.alloc_fragment([G], accum_dtype)

            i_t = bx
            i_v = by
            i_bh = bz
            i_b = i_bh // head_kv
            i_h = i_bh % head_kv

            T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for s in T.Pipelined(S, num_stages=num_stages):
                i_s = BlockIndices[i_b, i_t, i_h, s] * BS
                if i_s <= i_t and i_s >= 0:
                    T.copy(K[i_b, i_s : i_s + BS, i_h, :], K_shared)

                    if is_causal:
                        for i, j in T.Parallel(G, BS):
                            acc_s[i, j] = T.if_then_else(i_t >= i_s + j, 0, -T.infinity(acc_s.dtype))
                    else:
                        T.clear(acc_s)

                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=True)
                    for i in T.Parallel(G):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(G, BS):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)

                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(G):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)

                    for i, j in T.Parallel(G, BV):
                        acc_o[i, j] *= scores_scale[i]

                    T.copy(V[i_b, i_s : i_s + BS, i_h, i_v * BV : (i_v + 1) * BV], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(G, BV):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[i_b, i_t, i_h * G : (i_h + 1) * G, i_v * BV : (i_v + 1) * BV])

    return kernel


_KERNEL_CACHE = {}


def _get_kernel(B, seq_len, H, HQ, D, S, block_size, is_causal):
    groups = HQ // H
    key = (B, seq_len, H, HQ, D, S, block_size, int(is_causal))
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = native_sparse_attention(
            batch=B,
            heads=HQ,
            seq_len=seq_len,
            dim=D,
            is_causal=bool(is_causal),
            block_size=block_size,
            groups=groups,
            selected_blocks=S,
        )
        _KERNEL_CACHE[key] = kernel
    return kernel


def run_kernel(
    q,
    k,
    v,
    block_indices,
    output,
    B,
    seq_len,
    H,
    HQ,
    D,
    S,
    block_size,
    is_causal,
):
    kernel = _get_kernel(
        int(B),
        int(seq_len),
        int(H),
        int(HQ),
        int(D),
        int(S),
        int(block_size),
        int(is_causal),
    )
    kernel(q, k, v, block_indices, output)
```

## 本地验证

本地验证时需要使用已经编译好的 TileLang，并把仓库根目录和 TVM Python 路径加入 `PYTHONPATH`。

在仓库根目录执行：

```bash
TILELANG_CACHE_DIR=/tmp/tilelang-cache \
MACA_PATH=${MACA_PATH:-/opt/maca} \
LD_LIBRARY_PATH="$(pwd)/build/lib:${MACA_PATH:-/opt/maca}/lib:${MACA_PATH:-/opt/maca}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}" \
PATH="${MACA_PATH:-/opt/maca}/bin:${MACA_PATH:-/opt/maca}/mxgpu_llvm/bin:${PATH}" \
PYTHONPATH="$(pwd):$(pwd)/3rdparty/tvm/python:$(pwd)/race_tests/nsa:${PYTHONPATH}" \
python race_tests/nsa/test_tilelang_nsa_fwd.py
```

已用 `submission.run_kernel` 验证过：

```text
(B=1, seq_len=64, H=1, HQ=16, D=32, S=1, block_size=16)    allclose True
(B=1, seq_len=128, H=2, HQ=32, D=64, S=4, block_size=16)   allclose True
(B=1, seq_len=128, H=1, HQ=16, D=128, S=2, block_size=32)  allclose True
```

## 注意事项

- `run_kernel` 内不要调用 `torch.cuda.synchronize()`。
- `output` 是评测器传入的缓冲区，必须原地写入。
- `block_indices` 中无效 block 使用 `seq_len` 作为哨兵值；当前 kernel 通过 `i_s <= i_t` 和 `i_s >= 0` 跳过无效或未来 block。
- 当前文档只说明提交模板和改算子入口，不涉及进一步算子优化。
