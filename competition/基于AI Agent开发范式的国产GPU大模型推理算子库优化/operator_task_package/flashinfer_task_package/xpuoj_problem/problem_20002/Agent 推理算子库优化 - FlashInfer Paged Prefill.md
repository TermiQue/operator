# Agent 推理算子库优化 - FlashInfer Paged Prefill

当前题目说明来源为 [*XPU-OJ 20002*](https://xpuoj.com/contest/2/problem/2)，内容以 XPU-OJ 线上为准。

## 1. 题目描述
你需要实现 FlashInfer paged KV cache prefill 的CUDA C++前向算子。

本题输入采用 FlashInfer BatchPrefillWithPagedKVCacheWrapper 的 paged NHD 布局。每个 batch 中有 seq_len 个 query token，KV cache 也有 seq_len 个 token，并按 page 存储。

评测程序会调用你提交代码中的 run_kernel 函数。你需要根据 qo_indptr、kv_indptr、kv_indices 和 last_page_len 读取 paged KV cache，并将结果写入 output。

baseline 使用 FlashInfer paged prefill 的 Python API：

``` python
wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD", backend="auto")
wrapper.plan(qo_indptr, kv_indptr, kv_indices, last_page_len,
             num_qo_heads, num_kv_heads, head_dim, page_block_size,
             causal=bool(causal),
             q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
wrapper.run(q, kv_data, out=output)

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
    const __nv_bfloat16* kv_data,
    __nv_bfloat16* output,
    const int32_t* qo_indptr,
    const int32_t* kv_indptr,
    const int32_t* kv_indices,
    const int32_t* last_page_len,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_qo_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t page_block_size,
    int64_t causal
);

```

**参数说明**

- q：query tensor，shape (batch_size * seq_len, num_qo_heads, head_dim)，连续 bf16
- kv_data：paged KV cache，shape (num_blocks, 2, page_block_size, num_kv_heads, head_dim)，连续 bf16，其中 kv_data[:, 0] 为 key，kv_data[:, 1] 为 value
- output：输出缓冲区，shape (batch_size * seq_len, num_qo_heads, head_dim)，连续 bf16
- qo_indptr：query/output indptr，shape (batch_size + 1)，连续 int32
- kv_indptr：paged KV indptr，shape (batch_size + 1)，连续 int32
- kv_indices：page index，shape (num_blocks)，连续 int32
- last_page_len：每个 batch 最后一个 page 的有效 token 数，shape (batch_size)，连续 int32
- page_block_size：page size，评测中固定为 16
- causal：是否启用 causal mask，本题按 benchmark case 固定为 0

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
    q,  # Tensor[bf16], shape (batch_size * seq_len, num_qo_heads, head_dim)
    kv_data,  # Tensor[bf16], shape (num_blocks, 2, page_block_size, num_kv_heads, head_dim)
    output,  # Tensor[bf16], shape (batch_size * seq_len, num_qo_heads, head_dim)
    qo_indptr,  # Tensor[int32], shape (batch_size + 1)
    kv_indptr,  # Tensor[int32], shape (batch_size + 1)
    kv_indices,  # Tensor[int32], shape (num_blocks)
    last_page_len,  # Tensor[int32], shape (batch_size)
    batch_size,  # int64
    seq_len,  # int64
    num_qo_heads,  # int64
    num_kv_heads,  # int64
    head_dim,  # int64
    page_block_size,  # int64
    causal,  # int64
):
    ...

```

**参数说明**

- q：query tensor，连续 bfloat16
- kv_data：paged KV cache，连续 bfloat16
- output：输出缓冲区，连续 bfloat16，需要写入结果
- qo_indptr/kv_indptr/kv_indices/last_page_len：paged KV metadata，连续 int32
- page_block_size：评测中固定为 16
- causal：评测中固定为 0

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
    q,  # Tensor[bf16], shape (batch_size * seq_len, num_qo_heads, head_dim)
    kv_data,  # Tensor[bf16], shape (num_blocks, 2, page_block_size, num_kv_heads, head_dim)
    output,  # Tensor[bf16], shape (batch_size * seq_len, num_qo_heads, head_dim)
    qo_indptr,  # Tensor[int32], shape (batch_size + 1)
    kv_indptr,  # Tensor[int32], shape (batch_size + 1)
    kv_indices,  # Tensor[int32], shape (num_blocks)
    last_page_len,  # Tensor[int32], shape (batch_size)
    batch_size,  # int64
    seq_len,  # int64
    num_qo_heads,  # int64
    num_kv_heads,  # int64
    head_dim,  # int64
    page_block_size,  # int64
    causal,  # int64
):
    global real_kernel
    if real_kernel is None:
        real_kernel = build_kernel(...)
    real_kernel(q, kv_data, output, qo_indptr, kv_indptr, kv_indices, last_page_len,
                batch_size, seq_len, num_qo_heads, num_kv_heads,
                head_dim, page_block_size, causal)

```

**参数说明**

- q：query tensor，连续 bfloat16
- kv_data：paged KV cache，连续 bfloat16
- output：输出缓冲区，连续 bfloat16，需要写入结果
- qo_indptr/kv_indptr/kv_indices/last_page_len：paged KV metadata，连续 int32
- page_block_size：评测中固定为 16
- causal：评测中固定为 0

run_kernel 内部需要自行计算合适的 grid/block，并 launch 你实现的 TileLang kernel。

## 3. 输入格式

本题输入由评测程序在 GPU 上构造，并按接口约定中的顺序传入 run_kernel。

q/kv_data/output 均为连续 torch.bfloat16 CUDA tensor，qo_indptr/kv_indptr/kv_indices/last_page_len 均为连续 torch.int32 CUDA tensor。

KV layout 固定为 FlashInfer paged prefill 的 NHD 布局，page size 固定为 16。

## 4. 输出格式

输出写入 output，shape 为 (batch_size * seq_len, num_qo_heads, head_dim)，类型为 bfloat16。

## 5. 样例

若 batch_size = 1、seq_len = 32、page_block_size = 16，则：

```
qo_indptr = [0, 32]
kv_indptr = [0, 2]
kv_indices = [0, 1]
last_page_len = [16]
```

第 0 个 batch 的 KV token 存放在 page 0 和 page 1 中，每个 page 有 16 个 token。
