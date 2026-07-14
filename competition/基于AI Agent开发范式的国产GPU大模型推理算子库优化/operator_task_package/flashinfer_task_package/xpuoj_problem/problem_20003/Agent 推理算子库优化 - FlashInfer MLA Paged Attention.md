# Agent 推理算子库优化 - FlashInfer MLA Paged Attention

当前题目说明来源为 [*XPU-OJ 20003*](https://xpuoj.com/contest/2/problem/3)，内容以 XPU-OJ 线上为准。

## 1. 题目描述
你需要实现 FlashInfer MLA paged attention 的 CUDA C++ 前向算子。

本题使用 BatchMLAPagedAttentionWrapper 的 DeepSeek MLA decode 配置：q_nope 表示不含 RoPE 的 query，q_pe 表示 RoPE 部分的 query，ckv 与 kpe 分别表示 compressed KV cache 与 RoPE KV cache。MLA page size 固定为 1。

评测程序会调用你提交代码中的 run_kernel 函数。你需要根据 q_indptr、kv_indptr、kv_indices 和 kv_lens 读取 cache，并将结果写入 output。

baseline 使用 FlashInfer MLA 的 Python API：

``` python
wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="auto")
wrapper.plan(q_indptr, kv_indptr, kv_indices, kv_lens,
             num_heads, head_dim_ckv, head_dim_kpe,
             page_size, False, sm_scale,
             q_nope.dtype, ckv.dtype)
wrapper.run(q_nope, q_pe, ckv, kpe, out=output, return_lse=False)

```

如何提交代码详见 [*评测指南*](https://xpuoj.com/d/2)。

## 2. 接口约定

### 2.1 CUDA

你必须在提交的 CUDA 源码中提供如下 C 符号，函数名、参数类型、顺序必须完全一致，并使用 extern "C" 防止 name mangling：

``` cpp
#include <stdint.h>
#include <cuda_bf16.h>

extern "C" void run_kernel(
    const __nv_bfloat16* q_nope,
    const __nv_bfloat16* q_pe,
    const __nv_bfloat16* ckv,
    const __nv_bfloat16* kpe,
    __nv_bfloat16* output,
    const int32_t* q_indptr,
    const int32_t* kv_indptr,
    const int32_t* kv_indices,
    const int32_t* kv_lens,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim_ckv,
    int64_t head_dim_kpe,
    int64_t page_size,
    int64_t causal
);

```

**参数说明**

- q_nope：query 的 compressed/nope 部分，shape (batch_size, num_heads, head_dim_ckv)，连续 bf16
- q_pe：query 的 RoPE 部分，shape (batch_size, num_heads, head_dim_kpe)，连续 bf16
- ckv：compressed KV cache，shape (batch_size * seq_len, 1, head_dim_ckv)，连续 bf16
- kpe：RoPE KV cache，shape (batch_size * seq_len, 1, head_dim_kpe)，连续 bf16
- output：输出缓冲区，shape (batch_size, num_heads, head_dim_ckv)，连续 bf16
- q_indptr：decode query indptr，shape (batch_size + 1)，内容为 [0, 1, ..., batch_size]
- kv_indptr：KV indptr，shape (batch_size + 1)，每段长度为 seq_len
- kv_indices：page index，shape (batch_size * seq_len)，连续 int32
- kv_lens：每个 batch 的 KV 长度，shape (batch_size)，连续 int32
- page_size：评测中固定为 1
- causal：评测中固定为 0

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
    q_nope,  # Tensor[bf16], shape (batch_size, num_heads, head_dim_ckv)
    q_pe,  # Tensor[bf16], shape (batch_size, num_heads, head_dim_kpe)
    ckv,  # Tensor[bf16], shape (batch_size * seq_len, 1, head_dim_ckv)
    kpe,  # Tensor[bf16], shape (batch_size * seq_len, 1, head_dim_kpe)
    output,  # Tensor[bf16], shape (batch_size, num_heads, head_dim_ckv)
    q_indptr,  # Tensor[int32], shape (batch_size + 1)
    kv_indptr,  # Tensor[int32], shape (batch_size + 1)
    kv_indices,  # Tensor[int32], shape (batch_size * seq_len)
    kv_lens,  # Tensor[int32], shape (batch_size)
    batch_size,  # int64
    seq_len,  # int64
    num_heads,  # int64
    head_dim_ckv,  # int64
    head_dim_kpe,  # int64
    page_size,  # int64
    causal,  # int64
):
    ...

```

**参数说明**

- q_nope/q_pe/ckv/kpe：MLA attention 输入 tensor，连续 bfloat16
- output：输出缓冲区，连续 bfloat16，需要写入结果
- q_indptr/kv_indptr/kv_indices/kv_lens：paged attention metadata，连续 int32
- page_size：评测中固定为 1
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
    q_nope,  # Tensor[bf16], shape (batch_size, num_heads, head_dim_ckv)
    q_pe,  # Tensor[bf16], shape (batch_size, num_heads, head_dim_kpe)
    ckv,  # Tensor[bf16], shape (batch_size * seq_len, 1, head_dim_ckv)
    kpe,  # Tensor[bf16], shape (batch_size * seq_len, 1, head_dim_kpe)
    output,  # Tensor[bf16], shape (batch_size, num_heads, head_dim_ckv)
    q_indptr,  # Tensor[int32], shape (batch_size + 1)
    kv_indptr,  # Tensor[int32], shape (batch_size + 1)
    kv_indices,  # Tensor[int32], shape (batch_size * seq_len)
    kv_lens,  # Tensor[int32], shape (batch_size)
    batch_size,  # int64
    seq_len,  # int64
    num_heads,  # int64
    head_dim_ckv,  # int64
    head_dim_kpe,  # int64
    page_size,  # int64
    causal,  # int64
):
    global real_kernel
    if real_kernel is None:
        real_kernel = build_kernel(...)
    real_kernel(q_nope, q_pe, ckv, kpe, output,
                q_indptr, kv_indptr, kv_indices, kv_lens,
                batch_size, seq_len, num_heads,
                head_dim_ckv, head_dim_kpe, page_size, causal)

```

**参数说明**

- q_nope/q_pe/ckv/kpe：MLA attention 输入 tensor，连续 bfloat16
- output：输出缓冲区，连续 bfloat16，需要写入结果
- q_indptr/kv_indptr/kv_indices/kv_lens：paged attention metadata，连续 int32
- page_size：评测中固定为 1
- causal：评测中固定为 0

run_kernel 内部需要自行计算合适的 grid/block，并 launch 你实现的 TileLang kernel。

## 3. 输入格式

本题输入由评测程序在 GPU 上构造，并按接口约定中的顺序传入 run_kernel。

q_nope/q_pe/ckv/kpe/output 均为连续 torch.bfloat16 CUDA tensor，q_indptr/kv_indptr/kv_indices/kv_lens 均为连续 torch.int32 CUDA tensor。

## 4. 输出格式

输出写入 output，shape 为 (batch_size, num_heads, head_dim_ckv)，类型为 bfloat16。

## 5. 样例

若 batch_size = 2、seq_len = 4，则：

```
q_indptr = [0, 1, 2]
kv_indptr = [0, 4, 8]
kv_indices = [0, 1, 2, 3, 4, 5, 6, 7]
kv_lens = [4, 4]
```

每个 batch 只有 1 个 decode query，会访问对应 batch 的全部 KV cache。
