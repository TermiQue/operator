# Agent推理算子库优化-FlashAttention KV Cache Decode Benchmark 实战：从性能基线到 XPU-OJ 评测

## 1. 教程定位

本教程面向 **沐曦-揭榜挂帅-Agent推理算子库优化-FlashAttention任务**，是围绕题目 **Agent推理算子库优化-FlashAttention KV Cache Decode** 的 **FlashAttention Benchmark 入门与评测提交衔接** 模块，主要帮助用户跑通 FlashAttention paged KV-cache 推理核函数 `flash_attn_with_kvcache` 的基准测试流程，理解 benchmark 脚本的输入输出、性能指标和评测含义，并基于 XPU-OJ 题包完成一个最小正确版 `run_kernel` 的实现与提交。

需要特别说明：本教程中的 benchmark 脚本主要用于帮助参赛者理解目标算子的调用方式、输入输出结构和性能基线，benchmark 脚本不是最终提交物。最终评测以 XPU-OJ 题包为准，参赛者需要根据题包中的接口约定实现自己的 `run_kernel`，并在输出结果对齐 OJ 参考结果的前提下提升性能。

完成本教程后，学员应能够：

* 完成环境验证与依赖检查
* 理解并配置 KV-Cache Benchmark 的核心参数
* 明确 benchmark 脚本的含义，理解 KV-Cache 为何成为推理性能瓶颈
* 运行 `flash_attn_with_kvcache` 的 Benchmark 测试并获取性能数据
* 输出一份性能基线结果记录表，为后续算子优化提供对比基准
* 理解 XPU-OJ 评测的 `run_kernel` 接口规范与精度要求
* 基于 OJ 题包接口实现一个最小正确版 `run_kernel`，通过正确性校验

## 2. 学习目标

完成本模块后，你将能够：

1. **理解 FlashAttention Paged KV-Cache 算子的作用**：明白 `flash_attn_with_kvcache` 在 LLM 推理 decode 阶段如何高效利用分页 KV Cache，减少显存碎片并提升吞吐。
2. **完成环境准备与 Benchmark 运行**：安装所需依赖，运行 `benchmark_kvcache.py`，生成包含执行时间与显存带宽的 CSV 性能记录。
3. **读懂 benchmark 脚本并定位性能瓶颈**：分析不同 `batch_size`、`seq_len_kv` 下的带宽曲线，理解显存带宽对 decode 阶段的影响。
4. **理清 benchmark 脚本与 OJ 题包的关系**：明确 benchmark 脚本用于建立性能基线，OJ 题包定义最终提交接口、数据范围和精度校验标准。
5. **实现并提交一个最小正确版 `run_kernel`**：根据题包中的接口约定编写 CUDA Maca 算子，通过 OJ 正确性校验并记录首次提交耗时。

## 3. 适用对象

本模块适合以下人员：

* 参与 **沐曦-揭榜挂帅-Agent推理算子库优化-FlashAttention任务** 的参赛者
* 对 GPU 算子性能优化感兴趣的开发者
* 需要了解 FlashAttention KV-Cache 推理性能的研究人员

**基础知识要求：**

* 了解 Python 编程基础
* 了解 PyTorch 基本用法
* 了解 GPU 推理的基本概念

## 4. 前置准备

开始实战前，请确认你已经完成以下准备。

### 4.1 环境准备

**设置领取与兑换算力券**

1. 前往沐曦开发者社区注册账号并完成邮箱验证，申请并获取 MACA 算力代金券兑换码。链接：https://developer.metax-tech.com/activities/6
2. 登录模力方舟平台，在“费用中心 -> 算力券”页面输入兑换码完成充值。链接：https://ai.gitee.com/

**创建并启动实例**

1. 进入算力市场，筛选“沐曦”芯片厂商，选择合适的 GPU 规格，推荐曦云 C500 节点。
2. 关键配置：在预装镜像处，务必选择专属开发镜像 PyTorch Agent / 2.8.0 / Python 3.12 / maca 3.7.2.1。
3. 创建完成后，进入算力容器，点击“工具-lab”即可打开 JupyterLab 终端开始项目创作。

![image](https://origin.picgo.net/2026/06/23/image86811f30a57ae7b8.png)

**说明：** 由于本次使用的是预装的专属镜像，环境中已经默认安装并配置好了 PyTorch、FlashAttention、einops 等依赖包。因此在启动实例后，无需再进行繁琐的依赖库版本验证即可直接进入测试环节。

### 4.2 代码准备

* 获取目标源码，包含 `benchmark_kvcache.py` 及 OJ 相关提交材料
* 进入项目目录 `op_optimization/基于AI\ Agent开发范式的国产GPU大模型推理算子库优化/operator_task_package/flashattn_task_package/benchmark/`
* 准备 Benchmark 脚本与 OJ 测试脚本

## 5. 知识速览

### 5.1 关键术语

* **KV-Cache：** 缓存历史 Token 的 Key / Value 向量，避免 Transformer 推理时重复计算。
* **Paged KV-Cache：** 将 KV-Cache 分页管理，减少显存碎片，提高利用率。
* **Batch Size：** 一次处理的样本数，越大并行度越高，但显存占用越大。
* **seq_len_kv：** KV-Cache 中已缓存的历史 Token 数量。
* **headdim：** 注意力头维度，常见为 64 / 128 / 256。

### 5.2 核心知识

* **正确性测试：** 验证算子输出结果的数学精度是否与标准实现一致，这是绝对底线。
* **性能测试：** 在正确的前提下测算速度与吞吐。通过建立性能基线，才能量化后续每次代码修改带来的真实收益。
* **Benchmark：** 在固定条件下反复运行同一任务，获取可重复的性能指标，用于建立基线、量化优化效果和定位瓶颈。
* **XPU-OJ：** 比赛官方在线评测平台，最终评测会调用参赛者提交代码中的 `run_kernel`。

### 5.3 关键指标

* **Kernel 执行时间：** GPU 核函数运行耗时，使用 GPU 端同步计时获得。
* **有效带宽：** 数据传输量除以 Kernel 时间，越接近理论峰值说明显存带宽利用越充分。

### 5.4 其他要点

* **Warmup：** 预热若干次，不记录，使 GPU 进入稳定状态。
* **Repeat：** 正式运行多次，取平均值或中位数以消除波动。
* **同步：** 调用 `torch.cuda.synchronize()` 确保精确计时。
* **数据类型：** 本教程使用 `bfloat16`，在精度和性能之间取得平衡。
* **显存占用估算：** KV-Cache 约等于 `batch × seq_len_kv × num_heads_k × headdim × 2(K+V) × 字节数`。
* **OOM 应对：** 减小 `batch` / `seq_len_kv`、使用更小 dtype 或释放中间变量。
* **Tensor Core：** 现代 GPU（含沐曦 C500）的矩阵乘法专用单元，通常要求维度对齐为 8 或 16 的倍数。

### 5.5 开源仓库参考

[GitHub - MetaX-MACA/flashattn](https://github.com/MetaX-MACA/flashattn)

> 链接内容可供用于学习 API、算子实现思路、benchmark 方法和优化策略。选手仍需根据 XPU-OJ 题包接口**自行实现**可提交的 `run_kernel(...)`。

[GitHub - MetaX-MACA/mcTlass](https://github.com/MetaX-MACA/mcTlass)

> 沐曦版 CUTLASS 组件，本题包要求核心矩阵计算必须基于此实现。仓库包含头文件和示例代码，可参考其接口设计和使用方式。

## 6. 项目实践：FlashAttention KV-Cache Benchmark

本章对 FlashAttention 的 paged KV-cache 推理核函数 `flash_attn_with_kvcache` 进行自动化性能基准测试，覆盖多种 `batch_size × seq_len_kv` 组合，输出执行时间和有效显存带宽。

### Step 1：进入创建的实例环境

打开模力方舟实例页面：https://ai.gitee.com/fwlhecko/dashboard/compute/instances

选择“工具-lab”进入实例环境。

![b5d7d783 3106 4a4d 97f5 c7ef4d7fa537](https://origin.picgo.net/2026/06/18/b5d7d783-3106-4a4d-97f5-c7ef4d7fa5379e806361950ffeed.png)

### Step 2：检查运行环境

**目标：** 确认当前环境满足本模块运行要求。

在 JupyterLab Terminal 中检查 GPU 状态、Python 版本和依赖版本。

![339ac3f8 e31d 43c6 992b 07e20e35ef95](https://origin.picgo.net/2026/06/18/339ac3f8-e31d-43c6-992b-07e20e35ef95004ae954fadfe267.png)

**命令示例：**

```bash
# 检查沐曦 GPU 状态
mx-smi

# 检查 Python 版本
python --version

# 检查 PyTorch 是否能识别 GPU
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# 检查依赖版本
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import flash_attn; print(f'flash-attn {flash_attn.__version__}')"
python -c "import einops; print('einops OK')"

# 检查 mctlass 头文件是否可用
ls $MACA_PATH/include/mctlass/ 2>/dev/null || echo "mctlass headers not found in default path"
```

**预期结果：**

* `mx-smi` 显示沐曦 GPU 信息

![mx-smi](https://origin.picgo.net/2026/06/23/image4591670d6c9f207d.png)

* Python 版本为 3.12

![image](https://origin.picgo.net/2026/06/23/image5f5fb78c43a0c220.png)

* `torch.cuda.is_available()` 返回 `True`

![image](https://origin.picgo.net/2026/06/23/image10a748b701eed6f6.png)

* FlashAttention、einops 等依赖可用

![image](https://origin.picgo.net/2026/06/23/image62099cfafe955cac.png)

* mctlass 头文件可用

![image](https://origin.picgo.net/2026/06/26/image1d2949c260b3eae8.png)

**常见问题：**

| 问题 | 解决方法 |
| --- | --- |
| `torch.cuda.is_available()` 返回 `False` | 检查 MXMACA 环境变量是否正确配置 |
| `ModuleNotFoundError: No module named 'flash_attn'` | 确认 flash-attn 已安装且版本符合镜像要求 |
| `mx-smi` 命令不存在 | 确认已进入沐曦 GPU 镜像环境 |

### Step 3：进入项目目录

**目标：** 进入本模块所需的源码目录。

1. 克隆代码仓库：

```bash
git clone https://gitlink.org.cn/metax-maca/op_optimization.git
```

2. 切换到 FlashAttention 任务包中的 benchmark 目录：

```bash
cd op_optimization/基于AI\ Agent开发范式的国产GPU大模型推理算子库优化/operator_task_package/flashattn_task_package/benchmark/
ls
```

**预期结果：**

```text
benchmark_kvcache.py
```

### Step 4：配置基准测试参数

**目标：** 根据测试需求配置基准测试参数。

在 `benchmark_kvcache.py` 的 `main()` 函数中配置基准测试参数。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `headdims` | `[256]` | head dimension，可改为 `[128]` 等 |
| `page_block_size` | `16` | paged KV-cache 的 block 大小 |
| `batch_sizes` | `[1, 2, 4, 8, 16, 32, 64, 128]` | 批大小扫描范围 |
| `seq_lens_kv` | `[512, 1024, 2048, 4096, 8192, 16384]` | KV 序列长度扫描范围 |
| `num_heads` | `8` | query head 数量 |
| `num_heads_k` | `8` | KV head 数量 |
| `seqlen_q` | `1` | query 序列长度，单 token 推理 |
| `dtype` | `torch.bfloat16` | 数据类型 |
| `causal` | `False` | 是否启用 causal mask |
| `warmup` | `10` | 预热迭代次数 |
| `repeat` | `100` | 正式 profiling 迭代次数 |

如需测试 `headdim=128`，修改对应列表：

```python
headdims = [128]
```

如需启用 causal mask：

```python
causal = True
```

### Step 5：运行基准测试

**目标：** 运行基准测试脚本，收集性能数据。

```bash
python benchmark_kvcache.py
```

脚本运行时会在终端实时打印结果表格：

```text
batch_size   seq_len_kv  heads  headdim    time_ms  bandwidth_GB_s
---------------------------------------------------------------------------
         1          512      8      256     0.0948           44.27
         2          512      8      256     0.0988           84.95
       ...
```

同时生成带时间戳的 CSV 文件，命名格式为 `benchmark_kvcache_YYYYMMDD_HHMMSS.csv`。

**常见问题：**

| 问题 | 解决方法 |
| --- | --- |
| 出现 `OOM` 标记 | 该配置超出 GPU 显存容量，可减小 `batch_size` 或 `seq_len_kv` |
| 脚本运行缓慢 | 减少 `repeat` 次数或缩小扫描范围 |

### Step 6：查看与分析结果

**目标：** 理解输出结果格式，分析性能数据。

**CSV 输出格式：**

| 列名 | 说明 |
| --- | --- |
| `batch_size` | 批大小 |
| `seq_len_kv` | KV 序列长度 |
| `heads` | head 数量 |
| `headdim` | head 维度 |
| `time_ms` | kernel 执行时间，单位 ms |
| `bandwidth_GB_s` | 有效显存带宽，单位 GB/s |

如果某个配置因显存不足而失败，对应的 `time_ms` 和 `bandwidth_GB_s` 列会标记为 `OOM`。

**带宽计算公式：**

```text
total_bytes = q_bytes + kv_bytes
q_bytes   = batch_size × seqlen_q × num_heads × headdim × bytes_per_elem
kv_bytes  = batch_size × seqlen_k × num_heads_k × headdim × bytes_per_elem × 2
bandwidth = (total_bytes / 1e9) / (time_ms / 1e3)   [GB/s]
```

其中 `bytes_per_elem` 在 `bfloat16` 下为 2，在 `float32` 下为 4。

**性能观察：**

* 小 batch 时带宽较低：`batch_size=1` 时 kernel 无法充分利用 GPU 并行度。
* 大 batch + 长序列时带宽较高：`batch_size=128` 时更容易接近 GPU 显存带宽上限。
* `headdim` 增大时 OOM 风险增加：`headdim=256` 的显存占用约为 `headdim=128` 的两倍。

**测试结果示例：**

**headdim=128（2026-05-26）**

所有 48 个配置均成功运行，峰值带宽约 1251 GB/s。

| batch_size | seq_len_kv | time_ms | bandwidth_GB_s |
| --- | --- | --- | --- |
| 1 | 512 | 0.0322 | 65.27 |
| 128 | 512 | 0.2453 | 1095.50 |
| 1 | 16384 | 0.8356 | 80.32 |
| 128 | 16384 | 6.8668 | 1250.98 |

**headdim=256（2026-05-27）**

48 个配置中有 3 个因显存不足（OOM）而失败，峰值带宽约 807 GB/s。

| batch_size | seq_len_kv | time_ms | bandwidth_GB_s |
| --- | --- | --- | --- |
| 1 | 512 | 0.0948 | 44.27 |
| 128 | 4096 | 5.3221 | 807.10 |
| 128 | 8192 | OOM | OOM |
| 64 | 16384 | OOM | OOM |
| 128 | 16384 | OOM | OOM |

### Step 7：自定义扩展（可选）

如需进一步测试，可参考以下扩展方法。

修改 `headdim`：

```python
headdims = [128, 256]
```

启用 causal mask：

```python
causal = True
```

调整 profiling 精度：

```python
warmup = 20
repeat = 200
```

启用详细 profiler 输出：

```python
ms = run_with_profiler(run_fn, warmup=warmup, reps=repeat, print_result=True, target_kernels=["flash"])
```

## 7. XPU-OJ 在线评测与提交

完成 Benchmark 后，下一步是理解 XPU-OJ 的题目接口、精度要求、提交方式和结果反馈，并完成一次冒烟提交。

### Step 8：从 Benchmark 到 XPU-OJ 提交

Benchmark 脚本用于理解目标算子的调用方式、输入输出 shape 和性能基线；XPU-OJ 题包用于定义最终评测接口、数据范围、参考输出和精度要求。跑完 benchmark 后，选手需要完成从 Python API 到可提交算子实现 `run_kernel` 的转换。

**Benchmark 与 XPU-OJ 的关系对比：**

| 维度 | Benchmark 脚本 | XPU-OJ 提交 |
| --- | --- | --- |
| 目的 | 理解算子接口、建立性能基线 | 统一环境下的正确性与性能评测 |
| 接口形式 | Python API：`flash_attn_with_kvcache` | 本教程展开 CUDA Maca 的 `extern "C" void run_kernel(...)`；Triton / TileLang 以 OJ 题目界面为准 |
| 数据范围 | 多种组合扫描，用于观察性能趋势 | 固定参数范围，以 OJ 题包为准 |
| 验证方式 | 主要产出性能记录和基线数据 | 强制通过 `torch.allclose(rtol=1e-2, atol=1e-2)` |
| 输出 | CSV 性能记录 | 测试点结果、SPJ Report、排行榜得分 |

跑完 benchmark、建立性能基线后，选手需要完成以下转换：

1. 从 benchmark 脚本中理解目标 API，本任务对应 `flash_attn.flash_attn_interface` 中的 `flash_attn_with_kvcache`，使用 paged KV cache 布局。
2. 在 XPU-OJ 平台上查看 **Agent推理算子库优化-FlashAttention KV Cache Decode** 题目的接口约定。
3. 对照题包中的输入 shape、数据范围和精度要求。
4. 编写自己的 `run_kernel(...)`。
5. 提交 OJ，先通过正确性。
6. 正确性通过后，再对比 benchmark 耗时 / OJ 耗时继续优化。

### Step 9：题目说明与提交入口

本教程只覆盖 **Agent推理算子库优化-FlashAttention KV Cache Decode** 这一题。正式接口、数据范围和精度要求以 XPU-OJ 题目界面为准。本节演示从 benchmark 到 XPU-OJ 提交的完整流程。

* **算子说明**：实现 paged KV cache 下的 decode 注意力，每个 batch 只有 1 个 query token，KV cache 按 page 存储，长度由 `seqlen_k` 决定。
* **对应 OJ 题目**：XPU-OJ 上 **Agent推理算子库优化-FlashAttention KV Cache Decode** 题。

使用组委会统一发放的账号登录 XPU-OJ，并进入对应比赛页面和题目页面。

1. 打开 XPU-OJ 平台：https://xpuoj.com/
2. 使用组委会统一发放的账号和初始密码登录。
3. 登录后进入比赛 / 题目列表页面，找到比赛 **沐曦-揭榜挂帅-Agent推理算子库优化-FlashAttention任务**。
4. 找到对应题目 **Agent推理算子库优化-FlashAttention KV Cache Decode**。
5. 点击进入题目详情页，查看题目描述、接口约定、数据范围和提交入口。

![image](https://origin.picgo.net/2026/06/23/image123512a0b877cd31.png)

### Step 10：理解 CUDA Maca 接口约定与精度要求

下面给出 CUDA Maca 的接口约定。Triton / TileLang 等其他提交语言的接口细节，请以 OJ 题目界面为准。

**必须实现的接口**

在 XPU-OJ 平台的题目详情页中，查看 **接口约定** 部分。使用 CUDA Maca 提交时，选手需要在提交源码中提供如下 C 符号，函数名、参数类型、顺序必须完全一致，并使用 `extern "C"` 防止 name mangling：

```cpp
#include <stdint.h>
#include <cuda_bf16.h>

extern "C" void run_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache_paged,
    const __nv_bfloat16* v_cache_paged,
    __nv_bfloat16* output,
    const int32_t* cache_seqlens,
    const int32_t* block_table,
    int64_t batch_size,
    int64_t seqlen_k,
    int64_t seqlen_q,
    int64_t num_heads,
    int64_t num_heads_k,
    int64_t headdim,
    int64_t page_block_size,
    int64_t num_blocks,
    int64_t causal
);
```

>**重要提交限制：** 在使用 CUDA Maca 提交 FlashAttention KV cache decode 算子时，`QK^T`、`PV` 等 attention 核心矩阵计算应使用沐曦提供的 `mctlass` 组件或其基础计算原语实现，不应完全采用手写 CUDA 循环替代 `mctlass` 完成核心矩阵乘法逻辑。可用头文件可参考 `#include "mctlass/mctlass.h"`。

**参数说明**

| 参数 | 说明 |
| --- | --- |
| `q` | decode query tensor，shape `(batch_size, seqlen_q, num_heads, headdim)`，连续 `bf16` |
| `k_cache_paged` | paged key cache，shape `(num_blocks, page_block_size, num_heads_k, headdim)`，连续 `bf16` |
| `v_cache_paged` | paged value cache，shape `(num_blocks, page_block_size, num_heads_k, headdim)`，连续 `bf16` |
| `output` | 输出缓冲区，shape `(batch_size, seqlen_q, num_heads, headdim)`，连续 `bf16` |
| `cache_seqlens` | 每个 batch 的 KV 长度，shape `(batch_size)`，连续 `int32` |
| `block_table` | 每个 batch 的 page 映射表，shape `(batch_size, num_blocks / batch_size)`，连续 `int32` |
| `seqlen_q` | query 长度，评测中固定为 `1` |
| `page_block_size` | page size，评测中固定为 `16` |
| `causal` | 是否启用 causal mask，评测中固定为 `0` |

`run_kernel` 内部需要自行计算合适的 launch 配置并启动 CUDA kernel。为保证计时准确，不建议在 `run_kernel` 内部做 `cudaDeviceSynchronize()` 或显式同步。

**KV cache 布局**

KV cache layout 固定为 `flash_attn_with_kvcache` 的 paged cache 布局：

```text
(num_blocks, page_block_size, num_heads_k, headdim)
```

第 `t` 个 KV token 位于 `block_table[batch_idx, t / page_block_size]` 指向的物理 page 中，page 内偏移为 `t % page_block_size`。

**评测数据范围与精度要求**

题包中定义的本题测试配置如下，正式配置仍以 OJ 题目界面为准：

```python
HEAD_DIMS = [128]
BATCH_SIZES = [1, 4, 16]
SEQ_LENS_KV = [1024, 4096, 8192, 16384]
SEQ_LEN_Q = 1
NUM_HEADS = 8
NUM_HEADS_K = 8
PAGE_BLOCK_SIZE = 16
CAUSAL = 0
```

当前 **Agent推理算子库优化-FlashAttention KV Cache Decode** 题的校验方式为：

```python
torch.allclose(output_t.float(), output_ref.float(), rtol=1e-2, atol=1e-2)
```

### Step 11：提交代码并查看结果

**目标：** 提交冒烟代码，确认 OJ 提交链路、语言环境和 `run_kernel(...)` 接口可用。

1. 在语言下拉框中选择本题支持的提交语言。本教程的接口和附录代码对应 `CUDA Maca`，其他语言以 OJ 题目界面为准。
2. 将实现了题目要求接口的代码复制到提交框中。
3. 本任务提供的 CUDA Maca 冒烟代码较长，完整源码统一收录在本文末尾的“附录：完整代码参考”中。
4. 点击“提交”按钮，等待评测结果。

评测时间与题目测试点数量、队列状态和平台负载有关，通常需要等待数十秒到数分钟。以平台实际返回为准。

![95a36602 dcd7 48ea 9010 0dc4e7bb45d6](https://origin.picgo.net/2026/06/18/95a36602-dcd7-48ea-9010-0dc4e7bb45d647d8de7b01ae032b.png)

**预期结果：**

* 提交成功后，系统会自动运行评测程序。
* 首先进行正确性校验，如果输出结果与参考实现差异超过 `rtol=1e-2, atol=1e-2`，则标记为 `Wrong Answer`。
* 正确性通过后，进行性能评测，计算加速比和得分。

OJ 对每次提交大致会走以下流程：

```text
1. 选手提交代码
2. 平台按所选语言编译或加载提交代码
3. 评测程序构造测试输入
4. 调用选手代码中的 run_kernel(...)
5. OJ 后台参考实现生成 output_ref
6. 将 run_kernel(...) 的输出与 output_ref 做正确性校验
7. 正确性通过后，统计运行耗时或性能指标
8. 根据题目评分规则换算该题得分
9. 更新该题历史最好成绩
10. 在榜单中展示本题得分和排名
```

### Step 12：分析评测结果与评分机制

**目标：** 理解 OJ 评测结果的含义，分析性能表现。

在 XPU-OJ 平台上查看提交结果。提交详情会显示状态、本题得分、时间、内存、编译信息以及各测试点结果。

![image](https://origin.picgo.net/2026/06/24/image158f29857ec8755e.png)

**单测试点查看：** 直接点击测试点可以看到测试点的详细检查器信息。

![image](https://origin.picgo.net/2026/06/24/imaged28bd6c2aa18a143.png)

在单测试点详情中，**检查器信息**（SPJ Report）会展示该测试点中的详细性能评估结果，以下是检查器中每行的意思：

- **Config**：`batch=1, seqlen_k=1024, seqlen_q=1, heads=8, kv_heads=8, headdim=128, page_size=16, causal=0`  
  测试用例的参数配置，定义了算子运行的具体场景（如批量大小、序列长度、注意力头数等），用于复现测试环境。

- **Baseline**：`0.120000 ms`  
  基准算子的执行时间（参考实现，优化前的版本），作为性能对比的标准。

- **User kernel**：`0.564000 ms`  
  你提交的算子的实际运行耗时。

- **Hardware bound**：`0.037287 ms`  
  硬件限制的理论最小耗时（理想情况下，算子受 GPU 硬件能达到的最快速度），若为 0 表示该场景下硬件未成为瓶颈，或基准 / 你提交的算子已接近硬件极限。

- **Speedup vs base**：`0.213 x`  
  **加速比**：你的算子相对于基准的加速倍数。计算方式为 `Baseline 时间 / 你的算子执行时间`。  
  当加速比 > 1 时，表示你的算子比基准快（性能提升）。  
  当加速比 < 1 时，表示你的算子比基准慢（性能下降）。  
  当前 `0.213 x` 意味着你的算子比基准慢了约 4.7 倍（1 / 0.213 ≈ 4.7），需要优化。

- **Score ratio**：`0.135722 (13.57%)`  
  **得分比例**：反映你的算子性能与基准的差距。根据评测系统，当你的算子与基准等速时（$T_k = T_b$），得分为 50 分；当达到硬件理论下限时（$T_k = T_h$），得分为 100 分。当前 `13.57%` 表示你的算子性能还有很大提升空间，得分约为 13 分（满分 100 分）。

- **Display score**：`13 / 100`  
  **最终得分**：由 Score ratio 映射而来的百分制分数。满分 100 分，分数越高表示性能越好。当前 `13 分` 表示你的算子性能还有很大提升空间。

- **Pass**：`OK`  
  测试用例的通过状态。`OK` 表示算子功能正确、输出与预期一致；若为 `FAIL`，则表示功能错误。

通过这些信息，你可以快速定位算子的性能瓶颈（如 User kernel 时间远大于 Baseline），或确认功能是否正确，从而针对性优化代码。

**OJ 评分机制解析：**

OJ 平台对单测试点的评分遵循以下公式：

```math
S(T_k) = \frac{100}{1 + \left(\frac{1}{0.5} - 1\right) \cdot \frac{T_k - T_h}{T_b - T_h}}
```

其中：

* $T_k$：你的 kernel 平均执行时间
* $T_b$：Baseline 参考实现平均执行时间，对应 50 分
* $T_h$：硬件理论下限耗时，对应 100 分，计算方式为：

```math
T_h = \max\left(
  \frac{\mathrm{FLOPs}}{\mathrm{peak\_tflops}},
  \frac{\mathrm{bytes}}{\mathrm{peak\_bw}}
\right)
```

**关键分数节点：**

| 你的性能 | 得分 | 含义 |
| --- | ---: | --- |
| $T_k = T_b$ | 50 分 | 与 Baseline 等速 |
| $T_k = T_h$ | 100 分 | 达到硬件理论上限 |
| $T_k < T_h$ | 大于 100 分 | 超越理论估算，可能因估算偏保守 |
| $T_k \gg T_b$ | 接近 0 分 | 远慢于 Baseline |

当单测试点得分超过 150 分时，平台会按对数压缩规则显示：

```math
S_{\mathrm{display}} = 150 + 10 \cdot \log_{10}\left(\frac{S}{150}\right)
```

本题得分为各测试点得分的算术平均，本题总耗时为各测试点 $T_k$ 的求和。

### Step 13：榜单查看与初步优化方向

在榜单页面，可以查看所有参赛者的排名情况：

![image](https://origin.picgo.net/2026/06/23/image4035d30ecf0356f4.png)

![image](https://origin.picgo.net/2026/06/24/image6f3c0164de8a994e.png)

* **本题得分：** 展示该题的历史最好成绩，排名按本题得分从高到低排序。
* **个人排名：** 页面顶部会显示你的当前排名和当前得分，方便快速了解自己的位置。
* **提交次数：** 分数下方括号中的数字表示该账号在本题下的提交次数。

初步优化方向包括：

| 方向 | 说明 |
| --- | --- |
| 算子融合 | 将矩阵乘、scale、softmax 等步骤合并为单个内核，减少 HBM 往返 |
| 并行策略调整 | 在 Decode 阶段采用分块或 Split-K 思路，提升长 KV 序列并行度 |
| 在线 Softmax | 引入局部最大值和局部求和动态缩放，保证数值稳定并减少访存 |
| 显存访问合并 | 保证相邻线程访问相邻地址，按 HeadDim 等连续维度向量化加载 |
| 利用内存层级 | 将频繁更新的标量放入寄存器，将块内复用数据放入共享内存 |
| 软件流水线 | 在当前分块计算时预取下一分块数据，隐藏内存加载延迟 |
| 减少冗余计算与分支 | 提取循环不变量，处理变长序列时减少复杂分支 |

## 8. Agent 辅助开发指南

OJ 提交可以使用 AI Agent 辅助完成题目阅读、接口理解、冒烟代码生成、错误定位和性能分析。本教程以 OpenCode 为例演示 Agent 辅助开发流程，也可以根据自己的习惯使用 Claude Code、Codex、Cursor 等工具。

### Step 14：OpenCode 安装与工作目录

如果你还没有自己的 `run_kernel`，可以先让 Agent 阅读题目界面和任务包材料，总结接口契约，再生成一个最小正确版实现思路。

在镜像 JupyterLab Terminal 中安装 OpenCode：

```bash
curl -fsSL https://opencode.ai/install | bash
```

![b4d06ad0 183b 47ce a673 27a6a69ef348](https://origin.picgo.net/2026/06/18/b4d06ad0-183b-47ce-a673-27a6a69ef34860481172500d1bb3.png)

安装完成后，进入 FlashAttention 任务包目录。注意：直接在任务包目录下工作即可，不需要切换到额外题包子目录。

```bash
cd op_optimization/基于AI\ Agent开发范式的国产GPU大模型推理算子库优化/operator_task_package/flashattn_task_package/
opencode
```

![opencode](https://origin.picgo.net/2026/06/25/-2026-06-25-1122089c7b677348527ea6.png)

### Step 15：Agent Prompt 模板与使用示例

下面的 Prompt 模板覆盖从题目解析、冒烟代码生成到 OJ 报错调试、性能瓶颈分析的完整流程。实际使用时，把 OJ 题目界面、提交代码和 SPJ Report 按需粘贴给 Agent。

| 任务阶段 | 参考 Prompt 模板 | 核心目的 |
| --- | --- | --- |
| 题包解析 | 请阅读 **Agent推理算子库优化-FlashAttention KV Cache Decode** 的题目界面和 FlashAttention 任务包材料，总结 CUDA Maca `run_kernel` 函数签名、Paged KV Cache 寻址公式、精度校验方式和测试数据范围。 | 提取接口契约，明确参数 shape 和边界条件 |
| 生成冒烟代码 | 请生成一个最小可运行的 CUDA Maca `run_kernel` 实现，要求严格匹配 `extern "C"` 签名，支持 Paged KV Cache 的 `block_table` 寻址，支持 head 映射，优先保证正确性。 | 快速验证接口和环境 |
| OJ 报错调试 | 我的代码提交后 `Wrong Answer`。这是我的代码和 SPJ Report。请检查 Paged KV 地址映射、bf16 到 float32 的计算转换、尾部 page 有效 token 判断是否正确。 | 结构化排查功能错误 |
| 性能瓶颈分析 | 这是测试点的 SPJ Report。请分析 `User kernel` 与 `Hardware bound` 的差距，判断更接近 compute-bound 还是 memory-bound，并给出具体的 mctlass 或访存优化建议。 | 将 OJ 反馈转化为优化行动 |

**题包解析 Prompt：**

```text
请阅读 XPU-OJ 上 **Agent推理算子库优化-FlashAttention KV Cache Decode** 的题目说明，以及 flashattn_task_package 中与 benchmark / 提交相关的材料。

请输出以下内容：
1. CUDA Maca 版本 run_kernel 的完整函数签名；
2. q、k_cache_paged、v_cache_paged、output、cache_seqlens、block_table 的 shape 和数据类型；
3. Paged KV Cache 的寻址公式，特别是 block_table、page_block_size、page_offset 的关系；
4. 精度校验方式，包含 rtol=1e-2、atol=1e-2；
5. 本题测试数据范围；
6. 容易写错的边界条件。

注意：本教程只讨论 CUDA Maca 接口约定。Triton / TileLang 的细节请以 OJ 题目界面为准。
```

**冒烟代码生成 Prompt：**

```text
请生成一个最小可运行的 CUDA Maca run_kernel 实现，用于 FlashAttention paged KV cache decode 题。

要求：
1. 严格匹配 OJ 题目界面中的 extern "C" void run_kernel(...) 签名；
2. k_cache_paged / v_cache_paged 布局为 (num_blocks, page_block_size, num_heads_k, headdim)；
3. 根据 cache_seqlens 和 block_table 从 paged cache 中读取 K / V；
4. 支持 num_heads 和 num_heads_k 的 head 映射；
5. bf16 输入需要转换为 float32 累加，最终写回 bf16；
6. 不在 run_kernel 内部调用 cudaDeviceSynchronize()；
7. 优先保证正确性，不追求性能。

请先解释核心索引公式，再给出代码。
```

在 OpenCode 中输入 prompt 后，可以让 Agent 先生成接口理解、实现思路和初版代码。

![2efeb3c0 77b0 407b 9bff 815670bcbf00](https://origin.picgo.net/2026/06/18/2efeb3c0-77b0-407b-9bff-815670bcbf0063a5b5b673b82532.png)

输出示例：

![095af1c8 aa7e 44e3 ad0e 620206aaba1e](https://origin.picgo.net/2026/06/18/095af1c8-aa7e-44e3-ad0e-620206aaba1e23a43b2eb7b0ead3.png)

**Wrong Answer 调试 Prompt：**

```text
我的 Agent推理算子库优化-FlashAttention KV Cache Decode 代码提交后出现 Wrong Answer。

这是我的代码：
[粘贴代码]

这是 OJ 的 SPJ Report：
[粘贴报告]

请按下面顺序排查：
1. run_kernel 函数签名、参数顺序和数据类型是否与 CUDA Maca 接口一致；
2. q / k_cache_paged / v_cache_paged / output 的 layout 是否匹配；
3. block_table 的 Paged KV Cache 寻址是否越界；
4. cache_seqlens 与尾部 page 的有效 token 判断是否正确；
5. bf16 是否转换为 float32 后再做累加和 softmax；
6. 是否存在数据竞争、错误同步或输出未完整写入。

请给出最可能的 3 个错误点，以及每个错误点对应的修改建议。
```

**性能瓶颈分析 Prompt：**

```text
这是 Agent推理算子库优化-FlashAttention KV Cache Decode 某个测试点的 SPJ Report：
[粘贴报告]

请分析：
1. User kernel、Baseline、Hardware bound 分别说明什么；
2. 当前性能更接近 compute-bound 还是 memory-bound；
3. 与 Hardware bound 的差距主要来自访存、并行度、softmax、mctlass 使用方式还是 launch 配置；
4. 给出 3 条可以优先尝试的优化建议；
5. 每条建议预期影响哪个指标，以及可能带来的正确性风险。
```

### Step 16：优化追踪与路径回顾

建议维护一份优化日志，记录每次优化的改动和性能变化，用于复盘和对比。

| 优化版本 | 改动描述 | 目标配置 | Baseline (ms) | 优化前 | 优化后 | 加速比 | OJ 得分 | 备注 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| v1.0 | 初始冒烟代码 | bs=1, sl=1024 | 0.120 | - | 0.564 | 0.21x | 13 | 验证链路 |
| v1.1 | 增加 mctlass 矩阵乘 | bs=1, sl=1024 | 0.120 | 0.564 | 0.150 | 0.80x | 45 | 计算优化 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

从性能基线到参赛作品的路径如下：

1. 跑通 benchmark，建立性能基线。
2. 提交冒烟代码，确认 OJ 链路正常。
3. 使用 Agent 阅读题目界面和任务包材料，理解输入输出、数据范围和精度要求。
4. 在 `run_kernel(...)` 中实现最小正确版算子。
5. 提交 OJ，先通过正确性。
6. 正确性通过后，再让 Agent 辅助分析性能瓶颈。
7. 围绕访存、softmax、线程划分、head_dim 特化、K/V 复用等方向迭代优化。
8. 保存每轮 Agent Prompt、代码改动、OJ 结果和性能变化，形成可复现的 Agent / Skill 优化流程。

## 9. 常见问题

| 类别 | 问题 | 可能原因 | 处理建议 |
| --- | --- | --- | --- |
| 环境 | `mx-smi` 命令无输出或报错 | 沐曦 GPU 驱动未正确加载，或没有进入正确镜像 | 确认实例类型和镜像，检查 MXMACA 环境变量 |
| 环境 | `ModuleNotFoundError: No module named 'flash_attn'` | flash-attn 未安装或版本不兼容 | 使用镜像内置环境，必要时检查 `pip show flash-attn` |
| Benchmark | 所有配置都显示 OOM | GPU 显存不足，或 `batch_size` / `seq_len_kv` 设置过大 | 缩小扫描范围，优先跑小 batch 和短序列 |
| Benchmark | 带宽数值异常低 | warmup 不足，GPU 未达到稳态，或 batch 太小 | 增加 `warmup`，记录多次运行结果，优先对比同配置变化 |
| 提交 | `Compile Error` 或 `Undefined reference to run_kernel` | 函数名被 C++ name mangling，或签名与 OJ 不一致 | 添加 `extern "C"`，逐字核对参数类型和顺序 |
| 正确性 | `Wrong Answer`，`torch.allclose` 失败 | layout 错、Paged KV 寻址错、bf16 精度处理错、尾部 token 越界 | 检查 `block_table` 寻址、float32 累加、有效 token 边界 |
| 运行 | `Runtime Error` 或非法内存访问 | page 索引越界、共享内存大小不足、线程块配置错误 | 核对 `num_blocks / batch_size`、动态共享内存大小和 launch 参数 |
| 性能 | `Time Limit Exceeded` | 串行循环过重、同步位置错误、并行度不足 | 移除不必要同步，优化 grid / block，按 KV 长度拆分并行 |
| 性能 | `User kernel` 远慢于 `Baseline` | 访存未合并、softmax 重复计算、未利用 mctlass / Tensor Core | 从访存合并、在线 softmax、矩阵计算原语和 head_dim 特化入手 |

## 10. 下一步学习建议

### 10.1 深入理解 FlashAttention 与 Paged KV Cache

阅读 FlashAttention 的源码与相关文档，重点理解 `flash_attn_with_kvcache` 的输入布局、KV cache 更新逻辑、paged cache 映射方式和 decode 阶段的访存瓶颈。

### 10.2 掌握 mctlass 组件

学习 mctlass 的基础数据类型、矩阵计算原语、tile 组织方式和示例代码。本题要求核心矩阵计算基于 mctlass 或其基础原语实现，因此熟悉组件接口会直接影响后续优化上限。

### 10.3 建立可复现的优化实验

每次修改只改变一个关键因素，并记录 benchmark / OJ 的前后结果。推荐至少记录配置、耗时、带宽、得分、修改点、失败现象和回滚结论。

### 10.4 结合 OJ Report 做定向优化

不要只看本题最终得分。单测试点的 `Config`、`User kernel`、`Hardware bound` 和 `Speedup vs base` 更适合指导下一轮优化方向。长序列、大 batch、小 batch 的瓶颈可能完全不同。

### 10.5 继续优化本题

完成 Agent推理算子库优化-FlashAttention KV Cache Decode 的冒烟提交后，可以继续围绕不同 batch、KV 长度和 head dimension 配置做定向优化。

## 附录:完整代码参考

本节为 CUDA Maca 冒烟代码完整版，主要用于验证接口签名和平台环境是否正常。它不是最优实现，也不作为评分参考。

```cpp
#include <stdint.h>
#include <cuda_bf16.h>
#include <math.h>

#include "mctlass/mctlass.h"
#include "mctlass/bfloat16.h"
#include "mctlass/numeric_conversion.h"

static_assert(sizeof(mctlass::bfloat16_t) == sizeof(__nv_bfloat16),
              "mctlass::bfloat16_t and __nv_bfloat16 must both be 16-bit values");

union Bf16Bits {
    __nv_bfloat16 nv;
    uint16_t bits;
};

MCTLASS_DEVICE float mctlass_bf16_to_float(__nv_bfloat16 x) {
    Bf16Bits u;
    u.nv = x;
    mctlass::bfloat16_t mx = mctlass::bfloat16_t::bitcast(u.bits);
    mctlass::NumericConverter<float, mctlass::bfloat16_t> convert;
    return convert(mx);
}

MCTLASS_DEVICE __nv_bfloat16 mctlass_float_to_bf16(float x) {
    mctlass::NumericConverter<mctlass::bfloat16_t, float> convert;
    mctlass::bfloat16_t mx = convert(x);
    Bf16Bits u;
    u.bits = mx.raw();
    return u.nv;
}

__global__ void paged_kv_decode_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache_paged,
    const __nv_bfloat16* __restrict__ v_cache_paged,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int64_t batch_size,
    int64_t seqlen_k,
    int64_t seqlen_q,
    int64_t num_heads,
    int64_t num_heads_k,
    int64_t headdim,
    int64_t page_block_size,
    int64_t max_num_blocks_per_seq)
{
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int num_warps = blockDim.y;

    int seqlen = cache_seqlens[b];

    extern __shared__ char smem[];
    float* q_smem = (float*)smem;
    float* smem_m = (float*)(q_smem + headdim);
    float* smem_d = (float*)(smem_m + num_warps);
    float* smem_acc = (float*)(smem_d + num_warps);

    int num_iters = (headdim + 31) / 32;

    if (seqlen == 0) {
        if (ty == 0) {
            for (int step = 0; step < num_iters; ++step) {
                int i = step * 32 + tx;
                if (i < headdim) {
                    output[(int64_t)b * num_heads * headdim + (int64_t)h * headdim + i] =
                        mctlass_float_to_bf16(0.0f);
                }
            }
        }
        return;
    }

    if (ty == 0) {
        for (int step = 0; step < num_iters; ++step) {
            int i = step * 32 + tx;
            if (i < headdim) {
                int64_t q_idx = (int64_t)b * num_heads * headdim + (int64_t)h * headdim + i;
                q_smem[i] = mctlass_bf16_to_float(q[q_idx]);
            }
        }
    }
    __syncthreads();

    float m_warp = -1e20f;
    float d_warp = 0.0f;
    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        acc[i] = 0.0f;
    }

    float scale = 1.0f / sqrtf((float)headdim);
    int kv_h = h / (num_heads / num_heads_k);

    for (int t = ty; t < seqlen; t += num_warps) {
        int page_idx = t / page_block_size;
        int page_offset = t % page_block_size;
        int block_id = block_table[b * max_num_blocks_per_seq + page_idx];

        int64_t k_base = (int64_t)block_id * (page_block_size * num_heads_k * headdim)
                       + (int64_t)page_offset * (num_heads_k * headdim)
                       + (int64_t)kv_h * headdim;

        float score = 0.0f;
        for (int step = 0; step < num_iters; ++step) {
            int i = step * 32 + tx;
            if (i < headdim) {
                float k_val = mctlass_bf16_to_float(k_cache_paged[k_base + i]);
                score += q_smem[i] * k_val;
            }
        }
        score *= scale;

        for (int mask = 16; mask > 0; mask /= 2) {
            score += __shfl_xor_sync(0xffffffff, score, mask);
        }

        float m_old = m_warp;
        m_warp = fmaxf(m_warp, score);
        float exp_val = expf(score - m_warp);
        float exp_old = expf(m_old - m_warp);
        d_warp = d_warp * exp_old + exp_val;

        int64_t v_base = (int64_t)block_id * (page_block_size * num_heads_k * headdim)
                       + (int64_t)page_offset * (num_heads_k * headdim)
                       + (int64_t)kv_h * headdim;

        for (int step = 0; step < num_iters; ++step) {
            int i = step * 32 + tx;
            if (i < headdim) {
                float v_val = mctlass_bf16_to_float(v_cache_paged[v_base + i]);
                acc[step] = acc[step] * exp_old + exp_val * v_val;
            }
        }
    }

    if (tx == 0) {
        smem_m[ty] = m_warp;
        smem_d[ty] = d_warp;
    }
    for (int step = 0; step < num_iters; ++step) {
        int i = step * 32 + tx;
        if (i < headdim) {
            smem_acc[ty * headdim + i] = acc[step];
        }
    }
    __syncthreads();

    if (ty == 0) {
        float global_m = -1e20f;
        for (int w = 0; w < num_warps; ++w) {
            global_m = fmaxf(global_m, smem_m[w]);
        }

        float global_d = 0.0f;
        for (int w = 0; w < num_warps; ++w) {
            global_d += smem_d[w] * expf(smem_m[w] - global_m);
        }

        for (int step = 0; step < num_iters; ++step) {
            int i = step * 32 + tx;
            if (i < headdim) {
                float global_acc = 0.0f;
                for (int w = 0; w < num_warps; ++w) {
                    global_acc += smem_acc[w * headdim + i] * expf(smem_m[w] - global_m);
                }
                float out_val = global_acc / global_d;
                int64_t out_idx = (int64_t)b * num_heads * headdim + (int64_t)h * headdim + i;
                output[out_idx] = mctlass_float_to_bf16(out_val);
            }
        }
    }
}

extern "C" void run_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache_paged,
    const __nv_bfloat16* v_cache_paged,
    __nv_bfloat16* output,
    const int32_t* cache_seqlens,
    const int32_t* block_table,
    int64_t batch_size,
    int64_t seqlen_k,
    int64_t seqlen_q,
    int64_t num_heads,
    int64_t num_heads_k,
    int64_t headdim,
    int64_t page_block_size,
    int64_t num_blocks,
    int64_t causal)
{
    int num_warps = 8;
    dim3 block(32, num_warps);
    dim3 grid(batch_size, num_heads);

    size_t smem_size = (headdim + num_warps * 2 + num_warps * headdim) * sizeof(float);
    int64_t max_num_blocks_per_seq = num_blocks / batch_size;

    paged_kv_decode_kernel<<<grid, block, smem_size>>>(
        q, k_cache_paged, v_cache_paged, output,
        cache_seqlens, block_table,
        batch_size, seqlen_k, seqlen_q,
        num_heads, num_heads_k, headdim,
        page_block_size, max_num_blocks_per_seq
    );
}
```
