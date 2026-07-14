# Agent 推理算子库优化 - FlashInfer Ragged Prefill

当前题目说明来源为 [*XPU-OJ 20001*](https://xpuoj.com/contest/2/problem/1)，内容以 XPU-OJ 线上为准。

## 1. 题目描述
你需要实现 FlashInfer ragged KV cache prefill 的CUDA C++前向算子。

本题输入采用 FlashInfer BatchPrefillWithRaggedKVCacheWrapper 的 ragged NHD 布局。每个 batch 段的 query/KV 长度由 qo_indptr 和 kv_indptr 给出；seq_len 只是所有段长度的上界，真实总长度分别是 qo_indptr[batch_size] 和 kv_indptr[batch_size]。

其中 query heads 采用 GQA 布局：num_qo_heads 个 query/output heads 共享 num_kv_heads 个 KV heads，G = num_qo_heads / num_kv_heads。

评测程序会调用你提交代码中的 run_kernel 函数。你需要根据 qo_indptr 和 kv_indptr 读取 ragged Q/K/V，并将结果写入 output。

baseline 使用 FlashInfer ragged prefill 的 Python API：

``` python
wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD", backend="auto")
wrapper.plan(qo_indptr, kv_indptr, num_qo_heads, num_kv_heads,
             head_dim_qk, head_dim_vo, causal=True,
             q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
wrapper.run(q, k, v, out=output)

```

如何提交代码详见 [*评测指南*](https://xpuoj.com/d/2)。

## 2. 接口约定

### 2.1 CUDA

你必须在提交的 CUDA 源码中提供如下 C 符号，函数名、参数类型、顺序必须完全一致，并使用 extern "C" 防止 name mangling：

``` cpp
#include <stdint.h>
#include <cuda_bf16.h>

extern "C" void run_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    __nv_bfloat16* output,
    const int32_t* qo_indptr,
    const int32_t* kv_indptr,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_qo_heads,
    int64_t num_kv_heads,
    int64_t head_dim_qk,
    int64_t head_dim_vo,
    int64_t causal
);

```

**参数说明**

- q：query tensor，shape (total_q, num_qo_heads, head_dim_qk)，连续 bf16，其中 total_q = qo_indptr[batch_size]
- k：key tensor，shape (total_kv, num_kv_heads, head_dim_qk)，连续 bf16，其中 total_kv = kv_indptr[batch_size]
- v：value tensor，shape (total_kv, num_kv_heads, head_dim_vo)，连续 bf16
- output：输出缓冲区，shape (total_q, num_qo_heads, head_dim_vo)，连续 bf16
- qo_indptr：query/output ragged indptr，shape (batch_size + 1)，连续 int32
- kv_indptr：KV ragged indptr，shape (batch_size + 1)，连续 int32
- seq_len：所有 query/KV 段长度的上界，可用于 launch grid；真实段长必须由 indptr 读取
- causal：是否启用 causal mask，评测中固定为 1

部分测试点是等长段，但也包含 q_len != kv_len 和不同 batch 段长度不相等的 ragged 测试点。实现不能假设 qo_indptr[b + 1] - qo_indptr[b] == seq_len 或 kv_indptr[b + 1] - kv_indptr[b] == seq_len。

run_kernel 内部需要自行计算合适的 launch 配置并启动 CUDA kernel。为保证计时准确，不建议在 run_kernel 内部做 cudaDeviceSynchronize() 或显式同步。

### 2.2 Triton

你必须在提交的 Python 代码中提供 run_kernel 函数，函数名、参数顺序、类型必须完全一致：

``` python
import triton
import triton.language as tl

@triton.jit
def your_kernel(...):
    ...

def run_kernel(
    q,  # Tensor[bf16], shape (total_q, num_qo_heads, head_dim_qk)
    k,  # Tensor[bf16], shape (total_kv, num_kv_heads, head_dim_qk)
    v,  # Tensor[bf16], shape (total_kv, num_kv_heads, head_dim_vo)
    output,  # Tensor[bf16], shape (total_q, num_qo_heads, head_dim_vo)
    qo_indptr,  # Tensor[int32], shape (batch_size + 1)
    kv_indptr,  # Tensor[int32], shape (batch_size + 1)
    batch_size,  # int64
    seq_len,  # int64, max segment length bound
    num_qo_heads,  # int64
    num_kv_heads,  # int64
    head_dim_qk,  # int64
    head_dim_vo,  # int64
    causal,  # int64
):
    ...

```

**参数说明**

- q/k/v：FlashInfer ragged prefill 输入 tensor，连续 bfloat16
- output：输出缓冲区，连续 bfloat16，需要写入结果
- qo_indptr/kv_indptr：ragged indptr，连续 int32；真实段长和 total_q/total_kv 以 indptr 为准
- causal：是否启用 causal mask，评测中固定为 1

run_kernel 内部需要自行计算合适的 grid/block，并 launch 你实现的 Triton kernel。

### 2.3 TileLang

你必须在提交的 Python 代码中提供 run_kernel 函数，函数名、参数顺序、类型必须完全一致：

``` python
import tilelang
import tilelang.language as T
from tilelang import jit

real_kernel = None

@jit
def build_kernel(*args):
    @T.prim_func
    def kernel(*args):
        ...
    return kernel

def run_kernel(
    q,  # Tensor[bf16], shape (total_q, num_qo_heads, head_dim_qk)
    k,  # Tensor[bf16], shape (total_kv, num_kv_heads, head_dim_qk)
    v,  # Tensor[bf16], shape (total_kv, num_kv_heads, head_dim_vo)
    output,  # Tensor[bf16], shape (total_q, num_qo_heads, head_dim_vo)
    qo_indptr,  # Tensor[int32], shape (batch_size + 1)
    kv_indptr,  # Tensor[int32], shape (batch_size + 1)
    batch_size,  # int64
    seq_len,  # int64, max segment length bound
    num_qo_heads,  # int64
    num_kv_heads,  # int64
    head_dim_qk,  # int64
    head_dim_vo,  # int64
    causal,  # int64
):
    global real_kernel
    if real_kernel is None:
        real_kernel = build_kernel(...)
    real_kernel(q, k, v, output, qo_indptr, kv_indptr,
                batch_size, seq_len, num_qo_heads, num_kv_heads,
                head_dim_qk, head_dim_vo, causal)

```

**参数说明**

- q/k/v：FlashInfer ragged prefill 输入 tensor，连续 bfloat16
- output：输出缓冲区，连续 bfloat16，需要写入结果
- qo_indptr/kv_indptr：ragged indptr，连续 int32；真实段长和 total_q/total_kv 以 indptr 为准
- causal：是否启用 causal mask，评测中固定为 1

run_kernel 内部需要自行计算合适的 grid/block，并 launch 你实现的 TileLang kernel。

## 3. 输入格式

本题输入由评测程序在 GPU 上构造，并按接口约定中的顺序传入 run_kernel。

所有 q/k/v/output 均为连续 torch.bfloat16 CUDA tensor，qo_indptr/kv_indptr 为连续 torch.int32 CUDA tensor。

张量布局固定为 FlashInfer ragged prefill 的 NHD 布局。

## 4. 输出格式

输出写入 output，shape 为 (total_q, num_qo_heads, head_dim_vo)，类型为 bfloat16，其中 total_q = qo_indptr[batch_size]。

## 5. 样例

若 batch_size = 1、seq_len = 4、num_qo_heads = 1、num_kv_heads = 1，则：

```
qo_indptr = [0, 4]
kv_indptr = [0, 4]
```

第 t 个 query 会访问同一 batch 内的 KV token 前缀；启用 causal mask 时，只能看到位置不超过 t 的 token。例如 t = 2 时：

```
attention = softmax(q[2, 0, :] @ k[0:3, 0, :].T / sqrt(head_dim_qk))
output[2, 0, :] = attention @ v[0:3, 0, :]
```

若某个 varlen case 中 q_len=2、kv_len=4，则 causal mask 采用 FlashInfer/sol-execbench 的 bottom-right 对齐：第 t 个 query 可见的 KV 上界为 t + 1 + (kv_len - q_len)。例如 t=0 时可见 k[0:3]，t=1 时可见 k[0:4]。

## 6. 数据范围与提示

- 数据类型：q/k/v/output 均为 bfloat16
- KV layout：NHD
- num_qo_heads = 32
- num_kv_heads = 4
- causal = 1
- head_dim_qk, head_dim_vo 取值为 (128, 128)
- batch_size 取值随测试点变化，覆盖 1, 2, 4, 15, 16, 27, 33
- seq_len 参数表示所有 query/KV 段长度的上界，各测试点的段长上界覆盖 1, 65, 123, 873, 987, 1024, 1280, 2048, 4096, 16384（变长测试点内部还包含 512、640 等更短的真实段长）
- total_q = qo_indptr[batch_size]
- total_kv = kv_indptr[batch_size]

注意：

- G = num_qo_heads / num_kv_heads，同一个 KV head 服务连续的 G 个 query heads。
- 对 query head h_q，对应的 KV head 为 h_q / G。
- 真实段长必须从 qo_indptr 和 kv_indptr 读取，不能假设每段长度相同。
- 启用 causal mask 后，采用 bottom-right 对齐。若当前段 q_len != kv_len，第 t 个 query 可访问的位置满足 kv_pos < t + 1 + (kv_len - q_len)。
- 输出校验容差为 rtol=1.6e-2, atol=1.6e-2，且允许不超过 1% 的元素超差（匹配率需 ≥ 0.99）。
- 被容忍的超差元素其绝对误差仍不得超过 8 × (atol + rtol · |ref|)，避免个别段被整段算错而蒙混通过。
- 单 token 边界（用例 14）和非 2 的幂尾段（用例 15）为小规模确定性用例，要求逐元素通过（匹配率需 = 1.0）。
- q/k/v 使用标准正态分布生成，避免均匀正输入导致长序列 softmax 退化成近似 prefix mean。

## 7. 测试用例尺寸

测试点顺序与 testcase_config.py 的 TESTCASES 一致。共 15 个测试点，全部 head_dim_qk = head_dim_vo = 128，覆盖等长长序列、变长 ragged、q_len < kv_len、短段和非 2 的幂长度。

<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse; width:100%;">
<thead>
<tr style="text-align:center; vertical-align:middle;">
<th style="padding:6px 10px;">测试用例ID</th>
<th>类型</th>
<th>batch</th>
<th>total_q</th>
<th>total_kv</th>
<th>max_q</th>
<th>max_kv</th>
<th>heads</th>
<th>head_dim</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>混合 ragged 长序列</td>
<td>33</td>
<td colspan="2">16294</td>
<td colspan="2">987</td>
<td rowspan="15">32/4</td>
<td rowspan="15">128/128</td>
</tr>
<tr>
<td>2</td>
<td rowspan="7">等长序列</td>
<td rowspan="3">1</td>
<td colspan="4">1024</td>
</tr>
<tr>
<td>3</td>
<td colspan="4">4096</td>
</tr>
<tr>
<td>4</td>
<td colspan="4">16384</td>
</tr>
<tr>
<td>5</td>
<td rowspan="2">4</td>
<td colspan="2">4096</td>
<td colspan="2">1024</td>
</tr>
<tr>
<td>6</td>
<td rowspan="2" colspan="2">16384</td>
<td colspan="2">4096</td>
</tr>
<tr>
<td>7</td>
<td rowspan="2">16</td>
<td colspan="2">1024</td>
</tr>
<tr>
<td>8</td>
<td colspan="2">32768</td>
<td colspan="2">2048</td>
</tr>
<tr>
<td>9</td>
<td>变长 <code>q_len &lt; kv_len</code></td>
<td rowspan="2">4</td>
<td>2048</td>
<td>4096</td>
<td>512</td>
<td>1024</td>
</tr>
<tr>
<td>10</td>
<td>混合变长 <code>q_len &lt; kv_len</code></td>
<td>1536</td>
<td>3584</td>
<td>640</td>
<td>1280</td>
</tr>
<tr>
<td>11</td>
<td>双段变长 <code>q_len &lt; kv_len</code></td>
<td>2</td>
<td>1024</td>
<td>3072</td>
<td>512</td>
<td>2048</td>
</tr>
<tr>
<td>12</td>
<td>混合 ragged 中长序列</td>
<td>27</td>
<td colspan="2">12251</td>
<td colspan="2">873</td>
</tr>
<tr>
<td>13</td>
<td>混合 ragged 短序列</td>
<td>15</td>
<td colspan="2">969</td>
<td colspan="2">123</td>
</tr>
<tr>
<td>14</td>
<td>单 token 边界</td>
<td colspan="5">1</td>
</tr>
<tr>
<td>15</td>
<td>非 2 的幂尾段</td>
<td>2</td>
<td colspan="2">98</td>
<td colspan="2">65</td>
</tr>
</tbody>
</table>

说明：变长测试点的真实段长由 qo_indptr 和 kv_indptr 给出；参赛实现应始终以 indptr 为准，而不是从 seq_len、total_q 或 total_kv 反推出每段长度。

## 8. PyTorch 参考实现

``` python
def baseline(q, k, v, output, qo_indptr, kv_indptr,
             batch_size, seq_len, num_qo_heads, num_kv_heads,
             head_dim_qk, head_dim_vo, causal):
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer,
        kv_layout="NHD",
        backend="auto",
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo,
        causal=bool(causal),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    wrapper.run(q, k, v, out=output)

```
