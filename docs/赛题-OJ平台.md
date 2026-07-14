# 1. Tilelang 算子优化 - Fused MoE GEMM

传统

10000 ms

4096 MiB

[返回比赛](https://xpuoj.com/contest/5)

0

通过

0

提交

题目描述

你需要实现 DeepSeek 风格 MoE 中的 pre-routed fused expert GEMM。

**该题目的 GPU 计算算子不能使用 PyTorch，请只用 TileLang 完成相关计算算子。**

这道题参考的是 TileLang MetaX 仓库中这个提交的 `race_tests/moe/` 相关代码：

- commit: `ee6db4376484f2f7270183c01fd0d90f794965cb`
- 入口文件： `race_tests/moe/custom_fusedmoe.py`
- 参考实现： `race_tests/moe/ref_fusedmoe.py`

评测程序会调用你提交代码中的 `run_kernel` 函数，参数形式约定为：

```python
run_kernel(
    stacked_expert_tokens,
    gate_w,
    up_w,
    down_w,
    routed_expert_weights,
    group_sizes,
    group_offsets,
    group_padded_offsets,
    group_idx_for_bx,
    out,
)
```

你需要完成的是 routed expert MLP 的 fused 部分，不包含 router：

```text
gate_logits = x @ gate_w[expert]^T
up_logits   = x @ up_w[expert]^T
hidden      = silu(gate_logits) * up_logits
output      = hidden @ down_w[expert]^T
output     *= routed_expert_weights
```

其中输入 `stacked_expert_tokens` 已经按 expert 分组并按 `M=128` 对齐 padding。

如何提交代码详见[评测指南](https://xpuoj.com/d/2)。

接口约定

你提交的代码需要暴露一个名为 `run_kernel` 的 Python 函数，评测程序会直接调用它。

函数签名约定为：

```python
def run_kernel(
    stacked_expert_tokens: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    routed_expert_weights: torch.Tensor,
    group_sizes: torch.Tensor,
    group_offsets: torch.Tensor,
    group_padded_offsets: torch.Tensor,
    group_idx_for_bx: torch.Tensor,
    out: torch.Tensor,
) -> None:
    ...
```

要求如下：

- 不需要返回新 tensor，直接原地写 `out`
- `out` 是唯一 `INOUT` 参数，其余参数都视为只读
- `out` 的 shape 和 dtype 由评测程序提前分配好
- padded token 对应的输出行应保持为 `0`

输入格式

本题输入由评测程序在 GPU 上构造，并按接口约定中的顺序传入 `run_kernel`。

张量均为连续存储。`expert_tokens` 已经按照 expert 分组，无需在本题中实现 router、top-k 或 scatter reduce。

输出格式

输出写入 `output`，shape 为 `(group_sum, d_hidden)`，类型为 `float16`。

`up_logits` 是可复用 workspace，不作为最终评测输出。

样例

若 `n_routed_experts = 2`，`group_sizes = [3, 2]`，`group_offsets = [0, 3]`，则：

- `expert_tokens[0:3]` 使用 expert 0 的三组权重计算
- `expert_tokens[3:5]` 使用 expert 1 的三组权重计算
- 每一行输出再乘以对应的 `routed_expert_weights`

实际评测数据由 `testcase_config.py` 随机生成。

数据范围与提示

测试点覆盖以下典型范围：

- `d_hidden`：2048 或 7168
- `d_expert`：2048 或 8192
- `n_routed_experts`：16、32 或 64
- `group_sum`：2272 到 9088
- `block_token`：128
- 输入、权重、workspace 和输出均为 `float16`
- group metadata 为 `int32`

## 测试用例尺寸

| 测试用例ID | n_routed_experts | d_hidden | d_expert | group_sum | warmup | iters |
| ---------- | ---------------- | -------- | -------- | --------- | ------ | ----- |
| 1          | 16               | 2048     | 8192     | 2272      | 5      | 30    |
| 2          | 32               | 7168     | 2048     | 4544      | 20     |       |
| 3          | 64               | 9088     |          |           |        |       |

注意：

- 输出只检查 `output`，但建议使用 `up_logits` 作为中间缓冲区，避免重复计算。
- 为保证计时准确，不建议在 `run_kernel` 内部调用 `torch.cuda.synchronize()`。
- 大测试点显存占用较高，评测配置已经降低 warmup/iters。

PyTorch 参考实现

```python
import torch

def run_kernel(
    stacked_expert_tokens: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    routed_expert_weights: torch.Tensor,
    group_sizes: torch.Tensor,
    group_offsets: torch.Tensor,
    group_padded_offsets: torch.Tensor,
    group_idx_for_bx: torch.Tensor,
    out: torch.Tensor,
) -> None:
    out.zero_()
    num_experts = int(group_sizes.numel())

    x_f32 = stacked_expert_tokens.to(torch.float32)
    gate_f32 = gate_w.to(torch.float32)
    up_f32 = up_w.to(torch.float32)
    down_f32 = down_w.to(torch.float32)

    for expert_idx in range(num_experts):
        valid_count = int(group_sizes[expert_idx].item())
        if valid_count == 0:
            continue

        raw_start = int(group_offsets[expert_idx].item())
        padded_start = int(group_padded_offsets[expert_idx].item())

        x_e = x_f32[padded_start:padded_start + valid_count]
        gate_logits = torch.matmul(x_e, gate_f32[expert_idx].transpose(0, 1))
        up_logits = torch.matmul(x_e, up_f32[expert_idx].transpose(0, 1))
        hidden_act = torch.sigmoid(gate_logits) * gate_logits * up_logits
        y_e = torch.matmul(hidden_act, down_f32[expert_idx].transpose(0, 1))
        y_e *= routed_expert_weights[raw_start:raw_start + valid_count].to(torch.float32).unsqueeze(1)

        out[padded_start:padded_start + valid_count].copy_(y_e.to(torch.float16))
```

这份参考实现对应的数学语义是：

```text
for each expert e:
    x_e = stacked_expert_tokens[padded_start : padded_start + valid_count]

    gate_logits = x_e @ gate_w[e]^T
    up_logits   = x_e @ up_w[e]^T
    hidden      = silu(gate_logits) * up_logits
    y_e         = hidden @ down_w[e]^T
    y_e        *= routed_expert_weights[raw_start : raw_start + valid_count]

    out[padded_start : padded_start + valid_count] = y_e
```

其中：

- `raw_start = group_offsets[e]`
- `padded_start = group_padded_offsets[e]`
- `valid_count = group_sizes[e]`

实现要点：

**该题目的 GPU 计算算子不能使用 PyTorch，请只用 TileLang 完成相关计算算子。**

1. `group_offsets` 和 `group_padded_offsets` 需要分别使用，不能混用。

2. `routed_expert_weights` 的索引基于真实 token 顺序。

3. `stacked_expert_tokens` 与 `out` 的索引基于 padding 后顺序。

4. padding 行不参与计算，输出保持为 `0`。

5. 当前 baseline 主路径会把 `group_idx_for_bx` 用作 block 到 expert group 的映射。

6. 为了对齐 TileLang kernel 的常见实现，最终结果按 `float16` 写回。

   
   