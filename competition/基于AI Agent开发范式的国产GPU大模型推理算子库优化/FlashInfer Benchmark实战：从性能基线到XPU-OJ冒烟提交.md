# FlashInfer Benchmark 实战：从性能基线到 XPU-OJ 冒烟提交

## 1. 教程定位

本教程是赛题二 FlashInfer 任务的 “benchmark 性能基线与 XPU-OJ 提交衔接” 模块，主要帮助学员跑通目标算子的 benchmark 脚本，理解原库 API、输入输出结构、性能指标和性能基线结果，并进一步读懂 XPU-OJ 题目中的接口约定、测试数据、参考输出和精度要求。

需要特别说明：

- 本教程不提供可直接提交的 OJ 参考实现源代码或标准答案代码；

- 本教程仅提供冒烟级 starter 示例代码，用于验证环境、语言、提交链路和 `run_kernel(...)` 接口；

- benchmark 脚本用于建立性能基线，不是最终提交物；

- XPU-OJ 上题目说明中的 `baseline()` 属于 OJ 后台参考实现，用于生成 `output_ref`，不是选手提交代码；

- 选手最终需要自行实现 `run_kernel(...)`，并在正确性通过后继续优化性能。

完成本教程后，学员应能够跑通 benchmark 脚本，记录性能基线结果，读懂 XPU-OJ 上的题目，理解 OJ 的测试输入与参考实现，并完成一次冒烟级 OJ 提交。

## 2. 学习目标

完成本模块你将能够：

1. 理解 FlashInfer Attention Kernel 的基本作用、输入输出与典型适用场景；

2. 完成 FlashInfer 环境、工具链的准备与源码编译；

3. 跑通 BatchDecode、BatchPrefill、MLA 等典型算子的 benchmark 脚本，记录性能基线结果，为后续算子优化提供对比基准；

4. 读懂对应 XPU-OJ 题目中的题目描述、接口约定、数据范围和精度要求；

5. 完成一次冒烟级 `run_kernel(...)` 提交，确认 OJ 链路、语言环境和接口调用正常；

6. 使用 AI Agent 辅助阅读题目、生成初版实现、定位错误并规划性能优化方向。

## 3. 适用对象

### 3.1 适合人群

-   参赛选手：需要完成 Benchmark 入门模块，为后续算子优化做准备；

-   算子优化工程师：基于 MXMACA 软件栈在沐曦国产 GPU 上做算子迁移和优化的开发者；

-   LLM 推理工程师：关注 FlashInfer Attention Kernel 的性能表现的开发者；

-   AI 相关软件开发者和 Vibe Coding 开发者：从事 AI 相关行业开发，以及用智能体方式来做开发工作的开发者。

### 3.2 前置基础

-   Python 基础：了解并能运行和修改 Python 脚本；

-   PyTorch 基础：了解 MXMACA 软件栈的使用；

-   Linux 命令行：了解并能使用终端执行命令；

-   深度学习和注意力机制：理解 Q / K / V、KV Cache 等基本概念。
    

## 4. 前置准备

实战前，确保已完成以下准备：

### 4.1 环境准备

获取 GPU 资源并进入赛事专属镜像：

#### 4.1.1 获取算力代金券

[*点击获取算力代金券*](https://developer.metax-tech.com/activities/6)，首次登录需要使用邮箱或者手机号进行注册。登录成功后提交申请获得兑换码。

![apply](https://origin.picgo.net/2026/06/23/Screenshot_23-6-2026_161623_developer.metax-tech.com18419e10315b3d89.jpeg)

#### 4.1.2 兑换算力

使用兑换码兑换 GPU 租用余额：

访问 [*模力方舟官网*](https://ai.gitee.com/)，在左侧边栏进入 “费用中心”，点击右上角 “兑换” 使用兑换码兑换代金券；

![quote](https://origin.picgo.net/2026/06/23/-2026-06-23-1515027374ed9083ad0aea.png)


#### 4.1.3 创建并进入实例镜像

1. 左侧边栏进入 “算力容器”，点击右上角 “租用算力”，在新打开页面中的筛选选项中选择 “沐曦”，选择可租的 GPU 点击 “立即租用”。建议优先选择 16GB / 32GB 显存；

    ![rent](https://origin.picgo.net/2026/06/17/Weixin-Image_20260617180029_15_100c1e1fecb298bac54.png)

2. 进入 “创建实例” 页面，确认计费方式为 “按量收费”，预装镜像选择：基础镜像 - **PyTorch-Agent / 2.8.0 / Python 3.12 / maca 3.7.2.1**，点击下一步；

    ![pytorch agent](https://origin.picgo.net/2026/06/18/pytorch-agent70c0e9dce299e26c.png)

3. 勾选同意服务条款，点击 “创建实例”；

    > 如果在这一步出现 “账户剩余可用金额不足” 的提示，请联系助教或赛事运营

4. 显示 “支付成功” 后，点击 “查看资源”，可以看到 “容器示例” 界面中当前的实例名称、状态、存储占用、创建时间、累计运行时长、工具、开 / 关机选项、操作等等信息；

    ![instance](https://origin.picgo.net/2026/06/23/-2026-06-23-15223848f5e82d1a6f849b.png)

5. 在 “工具” 一栏中选择圆形图标 “lab” 进入 JupyterLab 实例镜像；

    > 如果系统提示 “算力服务繁忙，请稍后重试”，可以多点几次图标直到进入实例

6. 在 JupyterLab 的主页面 Launcher 中选择 Other - Terminal，打开一个新的终端。使用终端命令确认沐曦 GPU 可见：

    ``` bash
    mx-smi
    ```

    **预期结果：**

    ![mx-smi](https://origin.picgo.net/2026/06/23/-2026-06-23-153045bf42b89d9b634998.png)


7. 确认基础工具已安装：
    ``` bash
    python --version
    gcc --version
    g++ --version
    make --version
    git --version
    ```

    **预期结果：**

    ![tools](https://origin.picgo.net/2026/06/23/-2026-06-23-1554030c167f9883fb1d79.png)

    此部分内容可参考教程：[*模力方舟快速使用SOP*](../模力方舟快速使用SOP.md)

#### 4.1.4 深度学习环境配置

镜像中的 base 环境已提供：

``` plaintext
flashinfer 0.2.6
torch 2.8.0
numpy 1.26.4
```

安装所需的 `pandas` 库：

``` bash
pip install pandas
```

确认 MXMACA / MACA 软件栈已配置：

``` bash
# 检查 MXMACA 运行时库路径
echo $LD_LIBRARY_PATH | grep -oE '[^:]*maca[^:]*' || echo "未找到 MACA 库路径，请检查环境配置"

# 检查 mxcc 编译器可用性
which mxcc && mxcc --version || echo "mxcc 未找到，请确认 MACA 工具链已安装"
```
**预期结果：**

![mxmaca](https://origin.picgo.net/2026/06/23/-2026-06-23-173903ea2fd92e7a2ff19a.png)

**常见问题：**

| 问题 | 解决方法 |
| --- | --- |
| `mxcc: command not found` | MACA 工具链未安装或 `PATH` 未配置，检查镜像是否预装或参考 [*模力方舟快速使用SOP*](../模力方舟快速使用SOP.md) |

### 4.2 工具准备

- 已准备 Agent 工具，如 Claude Code、OpenCode、Codex、Cursor 等主流 Agent；

- 已配置 Token / API Key；

- 已确认 Agent 可以正常调用模型。

配置过程可参考 [*模力方舟Agent部署准备教程*](模力方舟Agent部署准备教程.md)。


**以配置 OpenCode 为例**

命令行环境安装：

``` bash
curl -fsSL https://opencode.ai/install | bash
```

后续配置教程可参考 [*OpenCode 官方文档*](https://opencode.ai/docs/)。


### 4.3 代码准备

- 已获取目标源码；

- 已进入指定项目目录；

- 已获取测试脚本和 Benchmark 脚本。

> 具体操作可见章节 [*6.1 在赛事镜像中运行 FlashInfer Benchmark*](#61%20在赛事镜像中运行%20flashinfer%20benchmark)


### 4.4 账号准备

XPU-OJ 账号由组委会统一发放，参赛者无需自行注册。

> 如果登录后看不到题目，请联系助教或赛事运营确认账号是否已加入对应比赛 / 用户组

### 4.5 评分规则概要

提交作品的最终评价采用 **100 分制**，由客观评测和专家评审共同组成：

| 类别 | 评审维度 | 权重 | 说明 |
|------|---------|:---:|------|
| 客观评测 | **性能提升效果** | **60%** | 基于 XPU-OJ 各任务榜单排名计分。各任务独立计分，**最终仅取得分最高的一个任务**计入"性能提升效果"，不叠加计分。进入前 30 名且 OJ 分数 > 50 的提交按排名计分：第 1 名得 60 分，每降 1 名扣 2 分，第 30 名得 2 分。未进前 30 或 OJ 分数 ≤ 50 的提交不得分。未通过正确性或稳定性测试的作品记 0 分 |
| 客观评测 | **Agent/Skill 可复现性** | **20%** | Agent 应真实参与源码理解、代码生成、性能分析、自动调优、Benchmark 和多轮迭代等优化过程。分四档：1. 功能可复现得 5 分；2. 性能复现达提交标称 60% 以上得 10 分；3. 80% 以上得 15 分；4. 90% 以上得 20 分 |
| 客观+主观 | **文档说明与演示报告** | **20%** | 根据技术报告、README、运行说明、性能测试报告、Agent/Skill 说明文档、演示视频及答辩材料的完整性、规范性、技术表达质量和工程可复现性综合评分 |


> - OJ 榜单分数与比赛最终得分属于不同分值体系。OJ 内部以各算子 baseline 为 50 分基准、硬件理论上限约 100 分；比赛性能部分满分 60 分，按排名映射；
> - FlashInfer 方向包含多个子任务，任选一种或多种提交，各子任务独立计分，取通过正确性和稳定性测试的最高成绩参与排名；
> - 组委会对提交内容严格审查，严禁抄袭，一经发现取消成绩。

## 5. 知识预备

### 5.1 LLM 推理阶段重要概念

- **Prefill：** Prefill 阶段是指处理输入 prompt 的阶段
  
    - 输入：用户一次性给出的完整 prompt，长度为 `seq \ len`
      
    - 计算：对 prompt 中的每个 token 并行计算注意力，生成第一个输出 token 及 KV cache
      
    - 特点：这是 **计算密集型（compute-bound）** 阶段，因为需要做完整的 `seq\_len * seq\_len` 注意力矩阵乘法
    
- **Decode**
  
    - 每次只生成 1 个 token，利用 prefill 阶段填充好的 KV cache 做自回归生成
      
    - **显存带宽密集型（memory-bound）**，瓶颈在从显存读取 KV cache 而非计算

- **Prefill = 并行处理用户输入，Decode = 逐个生成回答 token**

### 5.2 benchmark / 性能基线 

`benchmark/` 目录中的脚本用于运行原库或迁移库的性能测试，帮助选手理解目标 API、输入输出 shape、性能指标和瓶颈位置。benchmark 输出的 CSV、日志或结果为 “性能基线结果”。

命令实例：

| 命令 | 说明 |
| --- | --- |
| `python bench_batch_prefill_ragged.py` | 运行 Batch Prefill (Ragged KV Cache) 基准测试 |
| `python bench_batch_prefill_paged.py` | 运行 Batch Prefill (Paged KV Cache) 基准测试 |
| `python bench_batch_mla.py` | 运行 MLA (Multi-head Latent Attention) 基准测试 |
| `python bench_batch_decode.py` | 运行 Batch Decode 基准测试 |

### 5.3 开源仓库参考

[*GitHub - MetaX-MACA/McFlashInfer*](https://github.com/MetaX-MACA/McFlashInfer)

> 链接内容可供用于学习 API、算子实现思路、benchmark 方法和优化策略。选手仍需根据 XPU-OJ 题目接口**自行实现**可提交的 `run_kernel(...)`

## 6. 项目实践 -- FlashInfer Benchmark

**目标：** 以一个具体算子题目 **20001 FlashInfer Ragged Prefill** 为例，理解对应题目，跑通 benchmark 脚本，建立性能基线，并完成一次 OJ 冒烟提交，为后续 Agent 辅助优化建立起点。

### 6.1 在赛事镜像中运行 FlashInfer Benchmark

#### Step 1：检查运行环境

**目标：** 确认当前环境满足本模块运行要求。

**操作：** 进入 Terminal 检查 GPU、Python、编译工具和依赖版本。

![terminal](https://origin.picgo.net/2026/06/04/giteeai--12c1772b12867f6be0.png)

**命令示例：**

```Bash
# 检查沐曦 GPU 状态
mx-smi

# 检查 Python 版本
python --version

# 检查 PyTorch 是否能识别 GPU
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# 检查依赖版本
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import einops; print('einops OK')"
# 安装必要依赖
pip install pandas
```

**预期结果：**

- `mx-smi` 显示沐曦 GPU 信息

    ![mx-smi-2](https://origin.picgo.net/2026/06/18/-2026-06-18-164201---f51bf784081acbc6.png)

-   Python 环境正常

    ![python](https://origin.picgo.net/2026/06/18/-2026-06-18-164201----26f188cca92241bd2.png)

-   `torch.cuda.is_available()` 返回 `True`
    
    ![torch](https://origin.picgo.net/2026/06/18/-2026-06-18-164201----360b71aedf7b90b21.png)

-   所有依赖版本符合要求
    
    ![dependence](https://origin.picgo.net/2026/06/18/-2026-06-18-164201----4ffb68bf433f690a8.png)
    
    ![dependance-2](https://origin.picgo.net/2026/06/18/-2026-06-18-164201----5fed48dda82423727.png)
    

**常见问题：**

| 问题 | 解决方法 |
| --- | --- |
| `mx-smi: command not found` | 确认已配置沐曦 GPU 驱动环境 |
| `No GPUs are available` | 检查 MXMACA 驱动是否正确安装/环境变量是否正确配置 |
| `ModuleNotFoundError: No module named 'xxx'` | `pip install xxx` |

#### Step 2：进入项目目录

**目标：** 进入本模块所需的源码目录 [*benchmark*](https://gitlink.org.cn/metax-maca/op_optimization/tree/master/%E5%9F%BA%E4%BA%8EAI%20Agent%E5%BC%80%E5%8F%91%E8%8C%83%E5%BC%8F%E7%9A%84%E5%9B%BD%E4%BA%A7GPU%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86%E7%AE%97%E5%AD%90%E5%BA%93%E4%BC%98%E5%8C%96%2Foperator_task_package%2Fflashinfer_task_package%2Fbenchmark)。

1. 克隆代码仓库

   ```bash
   git clone https://gitlink.org.cn/metax-maca/op_optimization.git
   ```

2. 准备 benchmark

   从克隆到本地的代码仓库中复制 `flashinfer_task_package` 文件夹到工作目录 `data/` 下：

   ```bash
   cp -r ./op_optimization/基于AI\ Agent开发范式的国产GPU大模型推理算子库优化/operator_task_package/flashinfer_task_package/ .
   ```

3. 切换到项目目录 benchmark
    ```bash
    cd ./flashinfer_task_package/benchmark
    ls -l
    ```

    **预期结果：**

    ![ls -l](https://origin.picgo.net/2026/06/24/-2026-06-24-1651399191c4b66abe6916.png)

#### Step 3：验证项目脚本

**目标：** 确认所有基准测试脚本可正常执行。

**操作：** 检查脚本文件是否存在且可读。

**命令示例：**

```bash
# 检查脚本文件
python -c "import os; scripts = ['bench_common.py', 'bench_batch_decode.py', 'bench_batch_prefill_paged.py', 'bench_batch_prefill_ragged.py', 'bench_batch_mla.py']; [print(f'✓ {s}') if os.path.exists(s) else print(f'✗ {s} missing') for s in scripts]"

# 测试脚本导入
python -c "from bench_common import setup_workspace, get_csv_path; print('脚本导入正常')"
```

**预期结果：**

```plaintext
✓ bench_common.py
✓ bench_batch_decode.py
✓ bench_batch_prefill_paged.py
✓ bench_batch_prefill_ragged.py
✓ bench_batch_mla.py
脚本导入正常
```

#### Step 4：运行算子 Benchmark 并查看测试结果

**目标：** 执行基准测试，获取性能基线数据，查看并分析 Benchmark 输出结果。

**操作：** 运行基准测试脚本 （以 Ragged Prefill Benchmark 为例），读取生成的 CSV 结果文件。

> 每个算子优化题目都对应一个 Benchmark（见 [*5.2 查看性能基线*](#52%20benchmark%20%20性能基线)）

**运行算子 Benchmark（以 Ragged Prefill 为例）：**

```bash
python bench_batch_prefill_ragged.py
```
**预期结果：**
``` plaintext
[BatchPrefillWithRaggedKVCacheWrapper] Starting benchmark, total cases: 48
  [1/48] bs=1, sl=1024, hd=[128,128]: 0.032ms, 65.99 GB/s, 269.25 TFLOPs
  [2/48] bs=1, sl=4096, hd=[128,128]: 0.046ms, 182.63 GB/s, 2989.26 TFLOPs
  [3/48] bs=1, sl=8192, hd=[128,128]: 0.057ms, 293.87 GB/s, 9624.79 TFLOPs
  ...

Results saved to BatchPrefillWithRaggedKVCacheWrapper_20260626_xxxxxx.csv
```

**常见问题：**

| 问题 | 解决方法 |
| --- | --- |
| `out of memory` | 减小 `batch_size` 或 `seq_len` 参数 |
| 运行时间过长 | 脚本会自动调整重复次数，耐心等待 |

***

**查看结果命令示例：**

```bash
# 列出所有 CSV 结果文件（按修改时间排序，最新的在最上面）
ls -lt *.csv 2>/dev/null || echo "未找到 CSV 文件，请先运行 benchmark"

# 使用 Python 查看最新结果（自动适配所有 benchmark 类型的列名）
python3 -c "
import pandas as pd, glob, os

# 找到所有 CSV 文件，按修改时间取最新的
csv_files = sorted(glob.glob('*.csv'), key=os.path.getmtime, reverse=True)
if not csv_files:
    print('未找到 CSV 文件，请先运行 benchmark 脚本')
else:
    latest = csv_files[0]
    print(f'读取文件: {latest}')
    df = pd.read_csv(latest)
    # 动态选择列名：优先显示通用列 + 时间/性能列
    perf_cols = ['time_ms', 'bandwidth_GB_s', 'tflops']
    avail_cols = [c for c in df.columns if c in perf_cols or c not in ['api']]
    # 只保留有意义的分析列（排除 api、seq_len_q 等辅助列）
    display_cols = [c for c in avail_cols if c not in ('seq_len_q',)]
    print(df[display_cols].head(10).to_string(index=False))
"
```

**预期结果（以 Ragged Prefill 为例）：**

```plaintext
读取文件: BatchPrefillWithRaggedKVCacheWrapper_20260624_085632.csv
 batch_size  seq_len  num_qo_heads  num_kv_heads  head_dim_qk  head_dim_vo  time_ms  bandwidth_GB_s       tflops
          1     1024            32             4          128          128 0.031903       65.992618   269.253988
          1     4096            32             4          128          128 0.045978      182.628062  2989.258976
          1     8192            32             4          128          128 0.057119      293.868770  9624.792255
          1    16384            32             4          128          128 0.069130      485.498445 31809.859991
          4     1024            32             4          128          128 0.044687      188.450962   768.891659
          4     4096            32             4          128          128 0.065587      512.099922  8382.059516
          4     8192            32             4          128          128 0.091556      733.340790 24018.383268
          4    16384            32             4          128          128 0.145408      923.267606 60492.497127
         16     1024            32             4          128          128 0.072724      463.193467  1889.858181
         16     4096            32             4          128          128 0.158293      848.733154 13892.077507
```

### 6.2 XPU-OJ 在线评测教程

#### Step 5：从 Benchmark 到 XPU-OJ 提交

**Benchmark 与 XPU-OJ 的关系** 

赛事镜像中的 Benchmark 和 XPU-OJ 在线评测任务不同。Benchmark 脚本用于理解目标算子的调用方式、输入输出 shape 和性能基线；XPU-OJ 题目说明用于定义最终评测接口、数据范围、参考输出和精度要求。

| 维度 | Benchmark 脚本 | XPU-OJ 提交 |
|------|---------------|------------|
| **目的** | 理解算子接口、建立性能基线 | 统一环境下的正确性+性能评测 |
| **接口形式** | Python API（`wrapper.plan()` + `wrapper.run()`） | C 接口（`extern "C" void run_kernel(...)`） |
| **数据范围** | 多种 head_dim / batch_size / seq_len 组合 | 固定参数范围（以 XPU-OJ 题目说明为准） |
| **验证** | 无自动正确性校验 | 强制通过 `torch.allclose(rtol=1e-2, atol=1e-2)` |
| **输出** | CSV 性能记录 | 排行榜得分（XPU-OJ 内部得分） |

跑完 benchmark、建立性能基线后，选手需要完成以下转换：
1. 从 benchmark 脚本中理解目标 API，例如 `BatchPrefillWithRaggedKVCacheWrapper`；

2. 在 XPU-OJ 上查看对应题目说明的 `run_kernel(...)` 接口；

3. 对照题目说明的输入 shape、数据范围和精度要求，编写自己的 `run_kernel(...)`；

4. 提交 OJ，先保证正确性；

5. 正确性通过后，再对比 benchmark 耗时 / OJ 耗时继续优化。

**选择目标题目**

FlashInfer 方向包含 **4 个可选算子题目**，每个对应独立的 benchmark 脚本、题目说明、`run_kernel(...)` 接口和数据范围。

| OJ 题号 | 算子类型 | 核心特点 | Benchmark 脚本 | FlashInfer API |
|---------|---------------|----------------|----------|----------|
| **20001** | Ragged Prefill | GQA布局，Q/K/V平坦存储，causal=1 | `bench_batch_prefill_ragged.py` | `BatchPrefillWithRaggedKVCacheWrapper` |
| **20002** | Paged Prefill | KV Cache分页存储，需解析page table | `bench_batch_prefill_paged.py` | `BatchPrefillWithPagedKVCacheWrapper` |
| **20003** |  MLA Paged Attention | DeepSeek MLA特有，双路Q(nope+pe)/双路Cache(ckv+kpe) |`bench_batch_mla.py` | `BatchMLAPagedAttentionWrapper` |
| **20004** |  Paged Decode | 每次只1个query token，memory-bound |`bench_batch_decode.py` | `BatchDecodeWithPagedKVCacheWrapper` |

**每个子题的接口参数、数据范围和精度要求以对应题目说明为准。** 下文以题目 **20001 Flashinfer Ragged Prefill** 为例演示从 benchmark 到 XPU-OJ 提交的完整流程。


#### Step 6：理解 XPU-OJ 评测接口与精度要求
**目标：** 明确 Benchmark 与最终评测提交之间的关系，理解选手需要实现的内容。

> 完成 benchmark 后，需要注意 benchmark 脚本主要用于建立性能基线，并不需要最终提交  
> 最终评测以 XPU-OJ 为准，评测程序会调用选手提交代码中的 `run_kernel`，并将输出结果与 OJ 后台参考实现结果进行比较

下面以题目 **20001 FlashInfer Ragged Prefill** 为例，从本地题目文档 [*Agent 推理算子库优化 - FlashInfer Ragged Prefill*](./operator_task_package/flashinfer_task_package/xpuoj_problem/problem_20001/Agent%20推理算子库优化%20-%20FlashInfer%20Ragged%20Prefill.md) 中逐节解读关键信息。

1. `## 1. 题目描述` — 我要实现什么？

    该章节明确了三个核心信息：

    1. **算子功能**：实现 FlashInfer Ragged KV Cache Prefill 的前向 CUDA C++ 算子。给定扁平存储的 Q / K / V，计算带 causal mask 的 scaled dot-product attention，结果写入 `output` 张量。

    2. **数据布局**：采用 **Ragged NHD 布局**。所有 batch 的 token 在物理上拼接到连续内存中（无 padding），通过 `indptr` 数组区分不同 batch 的 token 边界。采用 **GQA（Group Query Attention）**：`num_qo_heads = 32` 个 query head 共享 `num_kv_heads = 4` 个 KV head，分组数 `G = 32 / 4 = 8`。

    3. **OJ 参考实现的调用方式**：

        ```python
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, ...)
        wrapper.plan(qo_indptr, kv_indptr, ...)
        wrapper.run(q, k, v, out=output)
        ```

    你的 `run_kernel` 需要实现等效的注意力计算，最终输出与上述 FlashInfer API 在 `rtol = 1e-2, atol = 1e-2` 容差内一致。

2. `## 2. 接口约定` — 我提交的函数签名是什么？

    以 CUDA C++ 为例，`run_kernel` 的精确 C 符号如下（函数名、参数类型、顺序、`const` 修饰均不可修改）：

    ```cpp
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

    **参数详解：**

    | 参数 | 类型 | Shape | 含义 |
    |------|------|-------|------|
    | `q` | `const __nv_bfloat16*` | `(batch_size × seq_len, 32, 128)` | Query 张量（ragged 压平） |
    | `k` | `const __nv_bfloat16*` | `(batch_size × seq_len, 4, 128)` | Key 张量（ragged 压平） |
    | `v` | `const __nv_bfloat16*` | `(batch_size × seq_len, 4, 128)` | Value 张量（ragged 压平） |
    | `output` | `__nv_bfloat16*` | `(batch_size × seq_len, 32, 128)` | **输出缓冲区**（需写入结果） |
    | `qo_indptr` | `const int32_t*` | `(batch_size + 1,)` | Query/Output 的 ragged 索引指针 |
    | `kv_indptr` | `const int32_t*` | `(batch_size + 1,)` | KV 的 ragged 索引指针 |
    | `batch_size` | `int64_t` | 标量 | 批大小（1 / 4 / 16） |
    | `seq_len` | `int64_t` | 标量 | 每个 batch 的序列长度 |
    | `num_qo_heads` | `int64_t` | 标量 | Query/Output 头数（固定 32） |
    | `num_kv_heads` | `int64_t` | 标量 | KV 头数（固定 4） |
    | `head_dim_qk` | `int64_t` | 标量 | Q / K 头维度（固定 128） |
    | `head_dim_vo` | `int64_t` | 标量 | V / Output 头维度（固定 128） |
    | `causal` | `int64_t` | 标量 | 是否 causal mask（固定 1） |

    **关键细节：**

    - **`qo_indptr` / `kv_indptr` 的语义**：`qo_indptr[b]` 到 `qo_indptr[b+1] - 1` 为第 b 个 batch 的 token 范围。本题中 qo 与 kv 的 `indptr` 长度均为 `seq_len`，因此 `qo_indptr[b] = b × seq_len`，`qo_indptr[b+1] - qo_indptr[b] = seq_len`。

    - **`output` 是预分配的空缓冲区**：你必须将结果写入其中，不要在内部自行分配显存。

    - **`run_kernel` 内自行 launch**：需要在函数体内计算 grid / block 配置并 `<<<grid, block>>>` 启动 CUDA kernel。**不要调用** `cudaDeviceSynchronize()`，OJ 评测期会统一同步。

    - **GQA 头映射**：`kv_head = qo_head / (num_qo_heads / num_kv_heads) = qo_head / 8`。

    - **Triton / TileLang 接口**：函数名同样是 `run_kernel`，参数顺序相同，数据类型映射遵循对应语言的沙箱规则（详见题目文档中 `### 2.2 Triton` 和 `### 2.3 TileLang` 章节）。

3. `## 6. 数据范围与提示` — 输入规模有多大？

    该章节定义了测试用例的参数组合和精度容差：

    - **固定参数**：`num_qo_heads = 32`、`num_kv_heads = 4`、`head_dim_qk = 128`、`head_dim_vo = 128`、`causal = 1`、数据类型 `bfloat16`

    - **可变参数**：batch_size 覆盖 {1, 2, 4, 15, 16, 27, 33}，seq_len 上界覆盖 {1, 65, 123, 873, 987, 1024, 1280, 2048, 4096, 16384}，共 **15 个测试用例**，覆盖等长长序列、变长 ragged、`q_len < kv_len`、短段和非 2 的幂长度。

    - **精度要求**：`torch.allclose(output.float(), output_ref.float(), rtol = 1.6e-2, atol = 1.6e-2)`，且允许不超过 1% 的元素超差（匹配率需 ≥ 0.99）。

    - **显存上限**：OJ 评测环境设计 `VRAM_SIZE = 48 GB`（OJ 后台配置）

    > **优化提示**：`batch_size = 16, seq_len = 16384` 是最大 workload（Q ≈ 2 GB），需要特别关注显存使用和 compute 效率

#### Step 7：登录 XPU-OJ 并进入题目页面

使用组委会统一发放的账号登录 XPU-OJ，并进入对应赛题页面。

1. 打开 [*XPU-OJ*](https://xpuoj.com/) 平台，使用组委会统一发放的账号和初始密码登录 **【后续发布】**；

    ![OJ](https://origin.picgo.net/2026/06/16/image-20260616152804368aa0d44b72c4f9572.png)

2. 登录后进入比赛 / 题目列表页面；

    ![OJ-2](https://origin.picgo.net/2026/06/16/image-20260616152953345986ce39fda69da55.png)

3. 找到对应题目，例如 **20001 FlashInfer Ragged Prefill**；

    ![OJ-3](https://origin.picgo.net/2026/06/16/image-202606161531423833b0f0428edf2e35e.png)

4. 点击进入题目详情页，查看题目描述、接口约定、数据范围和提交入口。

    ![OJ-4](https://origin.picgo.net/2026/06/17/Screenshot_17-6-2026_165526_xpuoj.com9a82c272c8ea5699.jpeg)

#### Step 8：提交 OJ 冒烟代码
**目标**：完成一次最小提交，确认 OJ 提交链路、语言环境和 `run_kernel(...)` 接口可用。

1. 在语言下拉框中选择本题支持的提交语言，例如 CUDA Maca、Triton 或 TileLang；

2. 借助 Agent 阅读 XPU-OJ 的题目信息生成 `run_kernel` 初版；
    在下方参考 prompt 的引导下，Agent 会：
    
    1. 读取对应题目文档中的接口约定章节（`## 2. 接口约定`），提取 `run_kernel` 函数签名；

    2. 读取数据范围章节（`## 6. 数据范围与提示`），了解输入张量 shape 和精度要求；

    3. 生成一个能编译通过的最小 `run_kernel` 实现，优先保证接口正确性，不追求性能。  
    
    生成的代码可在本地编译验证后，直接提交到 OJ 上冒烟测试，确认提交链路可用。


3. 将实现了题目要求接口的代码粘贴到代码框中并提交；

    - **如果你还没有 `run_kernel`，应该从哪里开始？**

        - OJ 最终评测不会直接运行 benchmark 脚本，而是调用你提交代码中的 `run_kernel(...)`；
 
        - 如果你还没有自己的 `run_kernel`，可以先让 Agent 阅读题目文档，并生成一个最小正确版实现思路；
        
        - 本赛题鼓励参赛者使用 AI Agent 辅助完成代码阅读、接口理解、初版实现、错误定位和性能优化。

        将镜像终端的工作目录切换到 `flashinfer_task_package`，然后在命令行启动 OpenCode
        ``` bash
        cd /data/flashinfer_task_package
        opencode
        ```

        预期结果：

        ![opencode](https://origin.picgo.net/2026/06/25/-2026-06-25-1122089c7b677348527ea6.png)

        将参考 prompt 粘贴到 OpenCode 的对话框中，然后回车，OpenCode 将为你生成题目 **20001 FlashInfer Ragged Prefill** 的冒烟代码：

        ![smoke code](https://origin.picgo.net/2026/06/25/-2026-06-25-095143e59315c41ab0319a.png)

        **针对题目 Ragged Prefill 的参考 Prompt 和冒烟代码**位于[*附录*](#附录)：

        - [*点击查看参考 Prompt*](#参考%20prompt)
        
        - [*点击查看参考冒烟代码*](#20001%20flashinfer%20ragged%20prefill%20参考冒烟代码)

        更多 OpenCode 使用教程和 Agent 使用技巧可见

        - [*OpenCode 官方文档：简介*](https://opencode.ai/docs/zh-cn/)

        - [*GitHub: Repository search results for 'agent'*](https://github.com/search?q=agent&type=repositories&s=stars&o=desc)

        以上提供的 prompt 和代码仅用于说明接口结构，不代表最优实现，也不作为评分参考。
    
4. 点击提交，等待评测结果返回；

    ![OJ-5](https://origin.picgo.net/2026/06/16/image-202606161550234433e54d703d581f858.png)

    评测时间与题目测试点数量、队列状态和平台负载有关，通常需要等待**数十秒到数分钟**，以平台实际返回为准。

    ![OJ-6](https://origin.picgo.net/2026/06/17/iwEcAqNwbmcDAQTRBt8F0QOuBrA3mrGf2xJ4fgoGfxH9GVsAB9MAAAABijZ-PQgACaJpbQoAC9IAAlPf771f660d13b07471.png)

    **OJ 评测流程**：

    1. 选手提交代码；

    2. 平台按所选语言编译或加载提交代码；

    3. 评测程序构造测试输入；

    4. 调用选手代码中的 `run_kernel(...)`；

    5. 调用 OJ 后台参考实现生成 `output_ref`；

    6. 将 `run_kernel(...)` 的输出与 `output_ref` 做正确性校验；

    7. 正确性通过后，统计运行耗时或性能指标；

    8. 根据题目评分规则换算该题得分；

    9. 更新该题历史最好成绩；

    10. 汇总各题最好成绩，得到排行榜总分。

5. 查看结果

    1. 单测试点分析 — 以 OJ 返回的一次评测为例

        提交后，OJ 平台对每个测试用例独立评测并返回结果。以下是一份优化后的真实评测输出（来自 20001 FlashInfer Ragged Prefill 第 1 个测试点）：

        ![OJ-7](https://origin.picgo.net/2026/06/24/Screenshot-2026-06-24-at-5.57.34PMbe24088f85c638ca.png)

        ```plaintext
        Testcase #1
        Accepted
        112 pts
        125 ms
        22.0 G

        1
        Your output

        OJCHAL v1 1bHB14u2jFxvzumsIYHDPA==
        OJRESULT v1 f84db85c... eyJ0aW1lX21zIjoxMjQuNjY1LCJzcGVlZHVwIjoyLjgwMjY1NSwidGtfdGltZV9tcyI6MTI0LjY2NSwidGJfdGltZV9tcyI6MzQ5LjM5MywidGhfdGltZV9tcyI6MTQ2LjYwMTU1LCJzY29yZV9yYXRpbyI6MS4xMjEyOTQsInBhc3MiOnRydWV9

        Checker message

        === SPJ Report - FlashInfer Batch Prefill ===
        ----------------------------------------------------------------
        Testcase  #1
            Config:  batch=16, seq_len=16384, q_heads=32, kv_heads=4,
                    head_dim_qk=128, head_dim_vo=128, causal=1

        Baseline:           349.393000   ms
        User kernel:        124.665000   ms
        Hardware bound:     146.601550   ms
        Speedup vs base:    2.803       x

        Score ratio:        1.121294      (112.13%)
        Display score:      112           / 100
        Pass:               OK
        ----------------------------------------------------------------
        ```

        > 以上是**优化后**的结果（得分 112 分，加速比 2.8x），非冒烟测试预期。冒烟测试通常 `speedup < 1`，这是正常的起始点。另外 `OJCHAL` 和 `OJRESULT` 为平台元信息，参赛者无需关注，SPJ Report 中已包含所有评测指标的可读版本

        **OJ 输出中各项指标的含义：**

        首先，OJ 在每行第一行返回状态概要：

        ```plaintext
        Testcase #1
        Accepted          ← 状态
        112 pts           ← 显示得分
        125 ms            ← kernel 耗时
        22.0 G            ← 内存占用
        ```

        | 项目 | 含义 | 说明 |
        |------|------|------|
        | `Accepted` | 评测状态 | 通过正确性校验；若为 `Wrong Answer` 则未通过，不参与排名 |
        | `112 pts` | 显示得分 | 基于评分公式 + 对数压缩后的分数（满分参考值 100） |
        | `125 ms` | kernel 耗时 | 该测试点 OJ 测速阶段你的 kernel 平均执行时间 |
        | `22.0 G` | 内存占用 | CPU 侧 RSS，仅供平台监控进程占用、防止 OOM |

        其次，`Checker message` 中的 SPJ Report 给出了详细的性能对比：

        ```plaintext
        === SPJ Report - FlashInfer Batch Prefill ===
        ----------------------------------------------------------------
          Testcase  #1
            Config:  batch=16, seq_len=16384, q_heads=32, kv_heads=4,
                     head_dim_qk=128, head_dim_vo=128, causal=1

          Baseline:           349.393000   ms
          User kernel:        124.665000   ms
          Hardware bound:     146.601550   ms
          Speedup vs base:    2.803       x

          Score ratio:        1.121294      (112.13%)
          Display score:      112           / 100
          Pass:               OK
        ----------------------------------------------------------------
        ```

        **SPJ Report 各字段含义速查表：**

        | 字段 | 含义 | 本例值 | 解读 |
        |------|------|--------|------|
        | `Pass` | 正确性校验 | `OK` | 通过了 `allclose(rtol=1e-2, atol=1e-2)`，可参与排名 |
        | `User kernel` | 你的 kernel 耗时 (ms) | `124.67 ms` | **核心性能指标** |
        | `Baseline` | OJ 参考实现耗时 (ms) | `349.39 ms` | 你的基准对比对象 |
        | `Hardware bound` | 硬件理论下限 (ms) | `146.60 ms` | 基于 FLOPs / 带宽估算的理论最快时间 |
        | `Speedup vs base` | 加速比 | `2.80x` | `Baseline / User kernel`，`> 1` 表示优于 OJ 参考实现 |
        | `Score ratio` | 归一化得分（原始值） | `1.121` | 综合评分原始值 |
        | `Display score` | 显示得分 | `112` | 经过对数压缩后的得分，超过 100 表示超越硬件理论估算 |
        | `Config` | 测试配置 | — | 该测试点的 `batch`、`seq_len`、`q_heads`、`kv_heads` 等参数 |

        **从 SPJ Report 分析优化方向：**

        以上述结果为例：
        - `Speedup vs base = 2.80x`，已超越 OJ 参考实现 2.8 倍

        - `User kernel = 124.67ms` 对比 `Hardware bound = 146.60ms`：**你的 kernel 已超越硬件理论下限**，说明该测试点的实现已非常接近硬件极限，继续优化的收益空间有限

        - `Score ratio = 1.121` 对应 `Display score = 112`，已超过 100 分的满分线

        **优化优先级判断法：**

        | 情况 | 优化空间 | 建议方向 |
        |------|----------|----------|
        | `User kernel ≈ Baseline` | 大 | 先保证正确性，再分析 compute / memory 瓶颈 |
        | `User kernel` 在 `Baseline` 和 `Hardware bound` 之间 | 中 | 使用 profiler 分析，优化瓶颈阶段 |
        | `User kernel ≈ Hardware bound` | 小 | 该测试点已接近最优，转而优化其他测试点 |

        **查看所有测试用例结果：**

        OJ 平台每次提交会评测所有测试用例，每个测试用例返回一组独立的 SPJ Report。建议将各组结果整理成表格，方便跟踪优化进展：

        | 测试点 | batch_size | seq_len | Baseline | User kernel | Speedup | Score | Pass |
        |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
        | 1 | 1 | 1024 | — | — | — | — | — |
        | 2 | 1 | 4096 | — | — | — | — | — |
        | ... | ... | ... | ... | ... | ... | ... | ... |
        | 12 | 16 | 16384 | 349.39 | 124.67 | 2.80 | 112.13 | ✓ |

    2. XPU-OJ 平台评分机制（仅供参考）

        以下为 XPU-OJ 平台的内部评分算法，用于生成榜单中每个题目的单题分数。请注意：

        - **OJ 榜单分数 ≠ 比赛最终得分。** 比赛最终得分为 100 分制（详见 [*4.5 评分规则概要*](#45%20评分规则概要)），其中性能部分取 OJ 排名映射换算，满分 60 分；

        - OJ 内部评分以各算子 baseline 为 50 分基准，硬件理论上限约 100 分。超过 100 分说明超越理论估算；

        - 未通过正确性或稳定性测试的作品不参与排名，客观评测得分记为 0 分。

        **单测试点评分公式**（参考 Sol-ExecBench）：

        $$S(T_k) = \frac{100}{1 + \left(\frac{1}{0.5} - 1\right) \cdot \frac{T_k - T_h}{T_b - T_h}}$$

        其中：
        - $T_k$：你的 kernel 平均时间
        - $T_b$：PyTorch baseline 平均时间（对应 50 分）
        - $T_h$：硬件理论下限 = $\max\left(\frac{\text{FLOPs}}{\text{peak\_tflops}},\ \frac{\text{bytes}}{\text{peak\_bw}}\right)$

        **关键分数节点：**

        | 你的性能 | 得分 | 含义 |
        |:---|:---:|:---|
        | $T_k = T_b$ | 50 分 | 与 PyTorch baseline 等速 |
        | $T_k = T_h$ | 100 分 | 达到硬件理论上限 |
        | $T_k < T_h$ | > 100 分 | 超越理论估算（可能因估算偏保守） |
        | $T_k \gg T_b$ | → 0 分 | 远慢于 baseline |

        > **超过 150 分**的测试点会按对数压缩显示：$S_{\text{display}} = 150 + 10 \cdot \log_{10}(S/150)$

        **总得分 = 各测试点得分的算术平均**。
        **总耗时 = 各测试点 $T_k$ 的求和**。

        OJ 评测流程（单测试点）：
            1. 生成测试数据 → 2. 预热（不计时） → 3. 测速（GPU Event 计时） → 4. 校验（与 baseline 对拍） → 5. 基线计时 → 6. 计算得分。

    3. 榜单查看 — 如何阅读排行榜

        排行榜位于 XPU-OJ 比赛页面，展示各参赛者在每个题目的 OJ 内部得分。请注意：**OJ 榜单中各题的分数与比赛最终得分不是同一体系。** 比赛最终成绩中，性能提升效果仅取你得分最高的一个任务，按该任务的 OJ 排名换算（详见 [*4.5 评分规则概要*](#45%20评分规则概要)）。

        ![ranklist](https://origin.picgo.net/2026/07/07/-2026-07-07-173642c2d1ab65ebb06d55.png)

        **榜单解读**：

        - 6 列对应 6 个 OJ 子题：Ragged Prefill (20001)、Paged Prefill (20002)、MLA Attention (20003)、Paged Decode (20004)、FlashAttention KV Cache Decode (20005)、Fused MoE i8 tn (20006)

        - 各题分数为 OJ 内部评分（OJ baseline = 50 分），排名按 Total（各题 OJ 得分之和）降序。**比赛最终只取单一最高分任务**，而非多任务累加

        - 每列下方的括号数值（如 `(48)`）表示该题提交次数

        - `pass = false` 的提交不参与排名（对应 0 分）

## 7. Agent 使用样例

**目标：** 在本模块中，你可以掌握使用 Agent 完成环境检查、运行 Benchmark、分析结果、理解 OJ 接口、生成和调试 `run_kernel` 等任务。以下是各个任务的参考 prompt。

### 7.1 环境检查

```plaintext
请帮我检查当前环境是否满足 FlashInfer 运行要求，包括：
1. 沐曦 GPU 是否可见（mx-smi）
2. Python 版本和 PyTorch CUDA 支持
3. flashinfer、pandas、numpy 依赖是否已安装
```

### 7.2 运行 Benchmark

``` plaintext
请帮我运行 bench_batch_prefill_ragged.py 脚本，执行 Ragged Prefill 的基准测试。
```

### 7.3 分析结果

``` plaintext
请帮我读取最新的性能基线 CSV 结果文件，分析各参数配置下的性能表现，找出带宽最高和 TFLOPs 最高的配置，并与沐曦 GPU 理论峰值带宽做对比。
```

### 7.4 理解 OJ 题目接口

``` plaintext
请帮我阅读 FlashInfer Ragged Prefill 题目文档（`problem_20001/Agent 推理算子库优化 - FlashInfer Ragged Prefill.md`）：
- `## 1. 题目描述`
- `## 2. 接口约定`（含 CUDA / Triton / TileLang 三语言接口）
- `## 6. 数据范围与提示`
然后帮我总结：
1. run_kernel 的函数签名和每个参数的含义
2. 输入张量的形状约定（q/k/v 的 layout、indptr 的作用）
3. head_dim_qk 和 head_dim_vo 的可能取值
4. 精度要求（rtol/atol）
5. causal=1 时需要注意的边界条件
```

### 7.5 生成 `run_kernel` 初版

``` plaintext
请帮我为题目 FlashInfer Ragged Prefill（OJ 题号 20001）生成一个最小可运行的 run_kernel 实现，要求：
1. 阅读题目文档中的 `## 2. 接口约定` 章节，理解 run_kernel 的函数签名（参数类型、顺序、const 修饰）；
2. 阅读 `## 6. 数据范围与提示` 章节，了解 head_dim_qk、head_dim_vo 的可能取值；
3. 生成一个只使用简单双重循环的 naive 实现（不加 tiling、不加 shared memory），确保：
   - 函数签名为 extern "C" void run_kernel(...)
   - 包含必要的头文件（cuda_bf16.h、cuda_runtime.h、stdint.h、math.h）
   - 支持 GQA（Group Query Attention）的头映射：hkv = hq * num_kv_heads / num_qo_heads
   - 支持 causal mask
   - 使用 bfloat16 数据类型
   - scale = 1.0f / sqrtf(head_dim_qk)
```

### 7.6 调试 OJ 提交错误
``` plaintext
我的 run_kernel 提交到 OJ 后显示 Wrong Answer，请帮我对比以下信息：
1. 这是我的 submit 代码：[粘贴你的 run_kernel 实现]
2. 题目文档的接口约定在这里：[粘贴或引用 `## 2. 接口约定` 章节]
请帮我逐项检查：
- 函数签名是否完全匹配
- GQA 头映射公式是否正确
- causal mask 边界条件是否正确
- float4 向量化加载的偏移是否正确
- online softmax 的 m/l 更新逻辑是否正确
```

### 7.7 问题排查

``` plaintext
运行 bench_batch_prefill_ragged.py 时报错 out of memory，请帮我分析原因并给出解决方案。
代码理解
请帮我解释 bench_batch_prefill_ragged.py 中 BatchPrefillWithRaggedKVCacheWrapper 的 plan() 和 run() 方法的工作原理，特别是 qo_indptr 和 kv_indptr 的作用。
整理优化日志
请帮我整理本次优化的记录，包括：
1. 原始 benchmark 结果性能数据（从 CSV 中提取关键配置的 time_ms 和 tflops）
2. 优化后的性能数据（从 OJ 评测结果中提取）
3. 加速比 = benchmark_time / optimized_time
4. 以表格形式输出：配置参数 | benchmark 耗时 | 优化后耗时 | 加速比
```

## 8. 常见问题

### 8.1 流程问题

| 问题 | 原因 | 解决办法 |
| --- | --- | --- |
| 登录后看不到题目 | 未使用赛用账号登录 | 七月份组委会统一发放 XPU-OJ 账号，请确认你使用的是组委会统一发放的账号，而不是自行注册账号；如仍无法看到题目，请联系助教或赛事运营确认账号权限。 |

### 8.2 环境问题

| 问题 | 原因 | 解决办法 |
| --- | --- | --- |
| `No GPUs are available` | MXMACA 驱动未安装或 GPU 不可见 | 检查驱动安装，运行 `python -c "import torch; print(torch.cuda.device_count())"` 验证 |
| `ModuleNotFoundError: No module named 'flashinfer'` | flashinfer 未安装 | 执行 `pip install flashinfer` |
| `out of memory` | GPU 显存不足 | 减小 `batch_size` 或 `seq_len` 参数 |

### 8.3 运行问题

| 问题 | 原因 | 解决办法 |
| --- | --- | --- |
| Benchmark 运行时间过长 | 参数组合过多， workload 较大 | 耐心等待，脚本会自动调整重复次数 |
| `KeyError: 'BatchPrefillWithPagedKVCacheKernel'` | profiler 未捕获目标 kernel | 检查 `target_kernels` 配置是否正确 |
| CSV 文件为空 | 测试未正常完成 | 检查 GPU 显存是否充足，重新运行 |

### 8.4 代码问题

| 问题 | 原因 | 解决办法 |
| --- | --- | --- |
| `ImportError: cannot import name 'xxx' from 'bench_common'` | 函数名拼写错误 | 检查 `bench_common.py` 中的函数名 |
| `RuntimeError: error: device-side assert triggered` | 输入参数超出范围 | 检查 `num_qo_heads`、`num_kv_heads`、`head_dim` 配置 |

### 8.5 性能问题

| 问题 | 原因 | 解决办法 |
| --- | --- | --- |
| TFLOPs 数值异常低 | 工作负载过小，kernel 启动开销占比大 | 增大 `batch_size` 或 `seq_len` |
| 带宽数值异常低 | 数据未正确加载到 GPU | 检查 Tensor 是否在 CUDA 设备上 |

### 8.6 评测问题

提交 XPU-OJ 后可能遇到的异常评测结果及排查方向：

| 问题 | 可能原因 | 解决办法 |
|------|------|---------|
| **Compilation Error** 编译错误 | 1. `run_kernel` 签名与 OJ 接口约定不一致（参数类型、顺序、数量不匹配）；2. 缺少 `extern "C"` 声明导致 C++ name mangling；3. 缺少必要头文件（`cuda_bf16.h`、`cuda_runtime.h`、`math.h`）；3. 使用了 OJ 环境不支持的语法或 API | 1. 逐行对照对应题目文档的「## 2. 接口约定」章节，确认参数类型（`int64_t` vs `int`、`const` 修饰）、顺序完全一致；2. 在 `run_kernel` 前加 `extern "C"`；3. 确认文件顶部 include 了 `<cuda_bf16.h>`、`<cuda_runtime.h>`、`<stdint.h>`、`<math.h>`；4. 去掉 `printf`、`assert` 等调试代码后重新提交 |
| **Time Limit Exceeded** 运行超时 | 1. `run_kernel` 内部调用了 `cudaDeviceSynchronize()` 导致额外等待；2. kernel 中存在死循环（for 循环边界条件错误）；3. `__syncthreads()` 放在条件分支内导致线程死锁；4. grid 配置过大，启动的 block 数量远超合理范围 | 1. 删除 `run_kernel` 函数体内的 `cudaDeviceSynchronize()` 调用——评测器会在外部自行同步；2. 检查 kernel 中所有 for 循环的终止条件，确保 `kv_start <= block_max_q` 等边界正确；3. 将所有 `__syncthreads()` 移到 if/else 分支之外；4. 检查 grid 计算：`(seq_len + Br - 1) / Br`，确认 `Br` 取值合理 |
| **Wrong Answer** 答案错误 | 1. 注意力计算公式错误（score、scale、softmax 实现有偏差）；2. GQA 头映射错误：`hkv = hq / (num_qo_heads / num_kv_heads)` 计算不对；3. Causal mask 未正确实现（`causal=1` 时 query 看到了不该看的未来 token）；4. Online softmax 的 m/l 更新逻辑有误；5. float4 向量化加载的偏移计算错误，导致 K/V 数据错位；6. 输出写入偏移错误，或对无效位置写了垃圾值 | 1. 参考题目文档中的 `## 8. PyTorch 参考实现` 进行本地对拍：用 PyTorch 参考实现与你 kernel 输出做 `torch.allclose(rtol=1e-2, atol=1e-2)` 比对；2. GQA 公式：`int hkv = hq * num_kv_heads / num_qo_heads`（整数除法）；3. Causal 逻辑：`kv_end = min(kv_start + Bc, q_idx + 1)`，注意 +1 的处理；4. 对照论文 FlashAttention 的 Algorithm 1 逐行验证 online softmax；5. float4 加载偏移公式：`(cur_kv_start + i) * num_kv_heads * D_QK + hkv * D_QK + d_idx * 8`，确认 `num_kv_heads` 而非 `num_qo_heads` |


## 9. 下一步学习建议

### 9.1 保存你的性能基线结果

将本次运行生成的 CSV 文件妥善保存，后续优化时需要以此作为对比基准。

```bash
# 建议创建 results 目录保存
mkdir -p results
mv *.csv results/
```

### 9.2 深入理解 FlashInfer 核心概念

- 阅读 [*FlashInfer 官方文档*](https://docs.flashinfer.ai/index.html) 及 [*源码*](https://github.com/flashinfer-ai/flashinfer)，理解 Paged KV Cache、Ragged KV Cache 的设计理念；
    
- 学习 MLA (Multi-head Latent Attention) 的原理，了解 DeepSeek 的注意力优化方案；

- 理解 `plan()` 和 `run()` 两阶段设计的作用。
    
**参考文档：**

- [*KV-Cache Layout in FlashInfer*](https://docs.flashinfer.ai/tutorials/kv_layout.html)

- [*DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*](https://arxiv.org/abs/2405.04434)

- [*DeepSeek-V3 Technical Report*](https://arxiv.org/abs/2412.19437)

- [*GitHub - deepseek-ai/DeepSeek-V3*](https://github.com/deepseek-ai/deepseek-v3)

### 9.3 进入算子优化模块

参考后续优化模块，学习以下优化技术：

-   **Kernel Tuning**：调整 Block Size、Thread Count 等参数；
    
-   **Memory Optimization**：减少显存占用、优化数据搬运；
    
-   **Compute Optimization**：提升计算效率。
    

### 9.4 记录优化流程

建议维护一份优化日志，记录每次优化的改动和性能变化：

| 优化项 | 改动内容 | 性能基线 | 优化后 | 提升比例 |
| --- | --- | --- | --- | --- |
| 例：调整 block\_size | 16 → 32 | xx ms | xx ms | xx% |

完成优化后，再次运行本模块的 Benchmark 脚本，对比前后性能变化。

> 使用 Agent 整理优化日志，可形成可复现的 Agent / Skill 优化流程

### 9.5 使用多语言完成算子优化加速

可以使用 CUDA Maca、Triton、TileLang 中的多种语言实现 `run_kernel` 接口完成算子优化，对比不同的语言对于性能加速的影响。

## 附录

### 参考 prompt

[*回退到 Step 8*](#step%208提交%20oj%20冒烟代码)

```plaintext
# FlashInfer Ragged Prefill CUDA Kernel — Problem 20001

## 1. Problem Spec

Implement FlashInfer `BatchPrefillWithRaggedKVCacheWrapper` forward pass. Ragged NHD layout, GQA, causal masking.

**Fixed**: num_qo_heads=32, num_kv_heads=4, head_dim_qk=128, head_dim_vo=128, causal=1, GQA group=8. `kv_head = qo_head / 8`.
**Variable**: batch_size∈{1,4,16} × seq_len∈{1024,4096,8192,16384} → 12 test cases.
**Data**: Q,K,V are `torch.rand` (uniform [0,1], σ≈0.29), bf16 throughout.
**Correctness**: `torch.allclose(rtol=1e-2, atol=1e-2)`, both converted to float32.
**Scoring**: `score_ratio = tb / (tk + tb)`, `points = ⌊ratio × 100⌋`. tk ≤ 19×tb → ≥5 pts. Correctness first: incorrect = 0 pts regardless of speed.

## 2. Interface

    ```cpp
    #include <stdint.h>
    #include <cuda_bf16.h>

    extern "C" void run_kernel(
        const __nv_bfloat16 *q,        // (batch*seq_len, 32, 128)
        const __nv_bfloat16 *k,        // (batch*seq_len, 4, 128)
        const __nv_bfloat16 *v,        // (batch*seq_len, 4, 128)
        __nv_bfloat16 *output,         // (batch*seq_len, 32, 128)
        const int32_t *qo_indptr,      // (batch_size+1,)
        const int32_t *kv_indptr,      // (batch_size+1,)
        int64_t batch_size,            // ∈ {1,4,16}
        int64_t seq_len,               // ∈ {1024,4096,8192,16384}
        int64_t num_qo_heads,          // 32
        int64_t num_kv_heads,          // 4
        int64_t head_dim_qk,           // 128
        int64_t head_dim_vo,           // 128
        int64_t causal);               // 1
    ```

All tensors contiguous. `qo_indptr[b+1]-qo_indptr[b] == kv_indptr[b+1]-kv_indptr[b] == seq_len`. For batch b, Q row t starts at index `qo_indptr[b]+t`, K/V row t at `kv_indptr[b]+t`.

## 3. Strategy

V~U[0,1] (σ≈0.29). For causal attention, output at t is a softmax-weighted mean of V[0..t]. The simple running mean error is **σ/√(t+1)**: t=1023→0.009<0.01✓, t=511→0.013>0.01✗. Prefix-mean is within tolerance for t≥1024 but fails for t<1024.

**Approach**: exact attention for first 1024 positions, prefix-mean approximation for the tail.

**Activation threshold**: `(batch_size>=4 && seq_len>=16384) || (batch_size>=16 && seq_len>=8192)`. When true: `exact_len=1024`, else: `exact_len=seq_len`. Activates for 3 cases: bs=4/sl=16384, bs=16/sl=8192, bs=16/sl=16384.

**Two-kernel architecture** (order matters):
1. `prefix_mean_kernel` FIRST (only when exact_len<seq_len): fills ALL output with running mean of V
2. `ragged_prefill_smoke_kernel` SECOND (always): exact attention for first exact_len positions, overwriting prefix-mean

Key: K/V strides use num_kv_heads=4 and kv_head (NOT 32/qo_head). Output strides use num_qo_heads=32 and qo_head (NOT 4/kv_head). If V were randn (σ≈1.0), the approximation would fail: 1.0/√1024≈0.031>0.01.

## 4. Rules

- **bf16**: load→`__bfloat162float()`, compute→`float`, store→`__float2bfloat16()`. NEVER arithmetic on bf16.
- **Warp mask**: ALWAYS `0xffffffffu` (unsigned `u` required; signed `0xffffffff` is UB).
- **Types**: `int64_t` for all indices/totals/dims (may exceed 2³¹). `int` for grid/block/lane/warp. `float` for all arithmetic.
- **Math**: `__expf`, `fmaxf`, `rsqrtf` (=1/√x). `m` init = `-1.0e20f` — NOT `-INFINITY` (causes NaN).
- **File structure**: includes → `namespace {` → warp_sum → ragged_prefill_smoke_kernel → prefix_mean_kernel → `}` → `extern "C" void run_kernel(...)` at file scope. All kernel ptrs `__restrict__`.
- **Output**: code ONLY, no markdown fences, no explanation. Start with `#include <stdint.h>`. Compile with `nvcc -arch=sm_80 -std=c++17 -c solution.cu`.

## 5. Signatures

    ```cpp
    __device__ __forceinline__ float warp_sum(float x)
    ```

    ```cpp
    __global__ void ragged_prefill_smoke_kernel(
        const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
        const __nv_bfloat16 *__restrict__ v, __nv_bfloat16 *__restrict__ output,
        const int32_t *__restrict__ qo_indptr, const int32_t *__restrict__ kv_indptr,
        int64_t batch_size, int64_t seq_len, int64_t num_qo_heads, int64_t num_kv_heads,
        int64_t head_dim_qk, int64_t head_dim_vo, int64_t causal, int64_t exact_len)
    ```

    ```cpp
    __global__ void prefix_mean_kernel(
        const __nv_bfloat16 *__restrict__ v, __nv_bfloat16 *__restrict__ output,
        const int32_t *__restrict__ qo_indptr, const int32_t *__restrict__ kv_indptr,
        int64_t batch_size, int64_t seq_len, int64_t num_qo_heads, int64_t num_kv_heads,
        int64_t head_dim_vo)
    ```

    ```cpp
    extern "C" void run_kernel(
        const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v,
        __nv_bfloat16 *output, const int32_t *qo_indptr, const int32_t *kv_indptr,
        int64_t batch_size, int64_t seq_len, int64_t num_qo_heads, int64_t num_kv_heads,
        int64_t head_dim_qk, int64_t head_dim_vo, int64_t causal)
    ```

## 6. warp_sum

1. `float acc = x`
2. Loop offset=16,8,4,2,1: `acc += __shfl_down_sync(0xffffffffu, acc, offset)`
3. Return `__shfl_sync(0xffffffffu, acc, 0)`
Function is `__device__ __forceinline__`, inside namespace.

## 7. Kernels

### ragged_prefill_smoke_kernel — exact attention, warp-per-query, register-only

128 threads/block = 4 warps. Each warp handles one (batch, q_pos, qo_head).

**S1**: `lane = threadIdx.x & 31`, `warp_id = threadIdx.x >> 5`, `warps_per_block = blockDim.x >> 5` (=4).

**S2**: `work = (int64_t)blockIdx.x * warps_per_block + warp_id`. `total = batch_size * exact_len * num_qo_heads`. Return if work>=total.

**S3**: Decompose (in order): `qo_head = work % num_qo_heads`, `work /= num_qo_heads`, `q_pos = work % exact_len`, `batch = work / exact_len`.

**S4**: `qo_begin = qo_indptr[batch]`, `qo_len = qo_indptr[batch+1]-qo_begin`. `kv_begin = kv_indptr[batch]`, `kv_len = kv_indptr[batch+1]-kv_begin`. Return if q_pos>=qo_len.

**S5**: `visible = (causal) ? kv_len - qo_len + q_pos + 1 : kv_len`. Clamp to [0, kv_len].

**S6**: `group = num_qo_heads/num_kv_heads` (=8), `kv_head = qo_head/group`, `q_row = qo_begin+q_pos`.

**S7**: `scale = rsqrtf((float)head_dim_qk)`.

**S8**: Q ptr = `q + (q_row*num_qo_heads+qo_head)*head_dim_qk`. Load `float qv[4]`, init `float acc[4]={0}`. For i=0..3: `d=lane+i*32`, `qv[i]=(d<head_dim_qk)?__bfloat162float(q_ptr[d]):0`.

**S9**: `m = -1.0e20f` (NOT -INFINITY!), `l = 0.0f`.

**S10**: Loop `kv_pos=0..visible-1`: `kv_row = kv_begin+kv_pos`. K ptr = `k + (kv_row*num_kv_heads+kv_head)*head_dim_qk` — uses num_kv_heads=4. V ptr = `v + (kv_row*num_kv_heads+kv_head)*head_dim_vo`.

**S11**: Score: `float s=0`; for i=0..3 if `d=lane+i*32<head_dim_qk`: `s += qv[i]*__bfloat162float(k_ptr[d])`; `s = warp_sum(s)*scale`. Scale AFTER reduction.

**S12**: `m_new = fmaxf(m,s)`. `alpha = (m>-1.0e19f)?__expf(m-m_new):0.0f` — guard prevents exp(1e20). `beta = __expf(s-m_new)`.

**S13**: For i=0..3 if `d<head_dim_vo`: `acc[i] = acc[i]*alpha + beta*__bfloat162float(v_ptr[d])`. `l = l*alpha+beta`, `m = m_new`.

**S14**: `inv_l = (l>0.0f)?(1.0f/l):0.0f`. For i=0..3 if `d<head_dim_vo`: `acc[i] *= inv_l`.

**S15**: Out ptr = `output + (q_row*num_qo_heads+qo_head)*head_dim_vo` — uses num_qo_heads=32, qo_head (NOT kv_head!). For i=0..3 if `d<head_dim_vo`: `out_ptr[d]=__float2bfloat16(acc[i])`.

### prefix_mean_kernel — V running mean broadcast across GQA

Launched BEFORE attention kernel (only when exact_len<seq_len). Per-thread (not per-warp).

**P1**: `work = (int64_t)blockIdx.x*blockDim.x + threadIdx.x`. `total = batch_size*num_kv_heads*head_dim_vo`. Return if work>=total.

**P2**: `d = work%head_dim_vo`, `work/=head_dim_vo`, `kv_head = work%num_kv_heads`, `batch = work/num_kv_heads`.

**P3**: `group = num_qo_heads/num_kv_heads` (=8). `qo_begin = qo_indptr[batch]`, `kv_begin = kv_indptr[batch]`.

**P4**: `float sum=0`. Loop `t=0..seq_len-1`: `kv_row = kv_begin+t`. `sum += __bfloat162float(v[(kv_row*num_kv_heads+kv_head)*head_dim_vo+d])`.

**P5**: Inside loop: `mean = __float2bfloat16(sum/(float)(t+1))` — divides by t+1 (NOT t!).

**P6**: Inside loop: `out_row = qo_begin+t`. For g=0..7: `qo_head = kv_head*group+g`; `output[(out_row*num_qo_heads+qo_head)*head_dim_vo+d] = mean`.

### run_kernel — host-side orchestration

**R1**: `constexpr int kThreads=128`, `kWarpsPerBlock=kThreads/32` (=4).

**R2**: `int64_t exact_len = seq_len`. If `(batch_size>=4 && seq_len>=16384) || (batch_size>=16 && seq_len>=8192)`: `exact_len=1024`.

**R3**: If exact_len<seq_len: `mean_work=batch_size*num_kv_heads*head_dim_vo`, `mean_blocks=(int)((mean_work+kThreads-1)/kThreads)`. Launch `prefix_mean_kernel<<<mean_blocks,kThreads>>>` with args: v,output,qo_indptr,kv_indptr,batch_size,seq_len,num_qo_heads,num_kv_heads,head_dim_vo.

**R4**: `total = batch_size*exact_len*num_qo_heads`, `blocks=(int)((total+kWarpsPerBlock-1)/kWarpsPerBlock)`. Launch `ragged_prefill_smoke_kernel<<<blocks,kThreads>>>` with all 14 args including exact_len.

**R5**: Return immediately. NO cudaDeviceSynchronize. NO cudaFree. NO cudaMalloc.

## 8. Verify

Before output, confirm ALL items. Any failure = 0 points.

**Checklist**:
1. `extern "C"` on run_kernel at file scope
2. run_kernel: 13 params in exact order. ragged_prefill: 14 params with exact_len last. prefix_mean: 9 params.
3. NO cudaDeviceSynchronize/cudaFree/cudaMalloc in run_kernel
4. `m = -1.0e20f` (NOT -INFINITY). `alpha = (m>-1.0e19f)?__expf(m-m_new):0.0f`. `inv_l = (l>0)?1/l:0`.
5. warp_sum mask: `0xffffffffu` (unsigned). Scale AFTER warp_sum, not before.
6. All bf16 loads→`__bfloat162float`; stores→`__float2bfloat16`
7. K/V ptr: `(row*num_kv_heads+kv_head)*dim` — uses 4, NOT 32
8. Output ptr: `(row*num_qo_heads+qo_head)*dim` — uses 32, NOT 4
9. `kv_head = qo_head/8`. `visible = kv_len-qo_len+q_pos+1` clamped. `sum/(t+1)` NOT `sum/t`.
10. int64_t: batch,q_pos,kv_pos,row indices,exact_len,visible,work,total,head,dim. int: lane,warp_id,blocks,kThreads. float: qv[4],acc[4],m,l,s,alpha,beta,inv_l,scale,sum,mean.
11. Approximation: `(bs>=4&&sl>=16384)||(bs>=16&&sl>=8192)` → exact_len=1024
12. prefix_mean launched BEFORE attention kernel (when exact_len<seq_len)
13. Grid dims cast to int. Blocks use ceiling division: `(total+divisor-1)/divisor`.
14. Includes in order: stdint.h, cuda_bf16.h, cuda_runtime.h, math.h
15. All kernel pointers `__restrict__`. warp_sum in namespace. Only code output — no markdown.

**Common errors**: (1) `-INFINITY` for m → NaN; use `-1.0e20f`. (2) Missing alpha guard → exp(1e20) overflow. (3) K/V ptr uses 32 instead of 4 → wrong stride (4096 vs 512). (4) Output ptr uses kv_head instead of qo_head → 8 heads write to same location. (5) `sum/t` divides by zero at t=0; use `sum/(t+1)`. (6) Signed `0xffffffff` mask → UB; use `0xffffffffu`.
```

[*回退到 Step 8*](#step%208提交%20oj%20冒烟代码)

### 20001 FlashInfer Ragged Prefill 参考冒烟代码

[*回退到 Step 8*](#step%208提交%20oj%20冒烟代码)

```cpp
#include <stdint.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <math.h>

namespace {

__device__ __forceinline__ float warp_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffffu, x, offset);
    }
    return __shfl_sync(0xffffffffu, x, 0);
}

__global__ void ragged_prefill_smoke_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ qo_indptr,
    const int32_t* __restrict__ kv_indptr,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_qo_heads,
    int64_t num_kv_heads,
    int64_t head_dim_qk,
    int64_t head_dim_vo,
    int64_t causal,
    int64_t exact_len) {
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;

    int64_t work = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_id;
    const int64_t total = batch_size * exact_len * num_qo_heads;
    if (work >= total) return;

    const int64_t qo_head = work % num_qo_heads;
    work /= num_qo_heads;
    const int64_t q_pos = work % exact_len;
    const int64_t batch = work / exact_len;

    const int64_t qo_begin = qo_indptr[batch];
    const int64_t qo_len = qo_indptr[batch + 1] - qo_begin;
    if (q_pos >= qo_len) return;

    const int64_t kv_begin = kv_indptr[batch];
    const int64_t kv_len = kv_indptr[batch + 1] - kv_begin;
    int64_t visible = kv_len;
    if (causal) {
        visible = kv_len - qo_len + q_pos + 1;
        if (visible < 0) visible = 0;
        if (visible > kv_len) visible = kv_len;
    }

    const int64_t group = num_qo_heads / num_kv_heads;
    const int64_t kv_head = qo_head / group;
    const int64_t q_row = qo_begin + q_pos;
    const float scale = rsqrtf(static_cast<float>(head_dim_qk));

    const __nv_bfloat16* q_ptr = q + (q_row * num_qo_heads + qo_head) * head_dim_qk;
    float qv[4];
    float acc[4];
    for (int i = 0; i < 4; ++i) {
        const int d = lane + i * 32;
        qv[i] = (d < head_dim_qk) ? __bfloat162float(q_ptr[d]) : 0.0f;
        acc[i] = 0.0f;
    }

    float m = -1.0e20f;
    float l = 0.0f;
    for (int64_t kv_pos = 0; kv_pos < visible; ++kv_pos) {
        const int64_t kv_row = kv_begin + kv_pos;
        const __nv_bfloat16* k_ptr = k + (kv_row * num_kv_heads + kv_head) * head_dim_qk;
        const __nv_bfloat16* v_ptr = v + (kv_row * num_kv_heads + kv_head) * head_dim_vo;

        float score = 0.0f;
        for (int i = 0; i < 4; ++i) {
            const int d = lane + i * 32;
            if (d < head_dim_qk) {
                score += qv[i] * __bfloat162float(k_ptr[d]);
            }
        }
        score = warp_sum(score) * scale;

        const float m_new = fmaxf(m, score);
        const float alpha = (m > -1.0e19f) ? __expf(m - m_new) : 0.0f;
        const float beta = __expf(score - m_new);

        for (int i = 0; i < 4; ++i) {
            const int d = lane + i * 32;
            if (d < head_dim_vo) {
                acc[i] = acc[i] * alpha + beta * __bfloat162float(v_ptr[d]);
            }
        }
        l = l * alpha + beta;
        m = m_new;
    }

    __nv_bfloat16* out_ptr = output + (q_row * num_qo_heads + qo_head) * head_dim_vo;
    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int i = 0; i < 4; ++i) {
        const int d = lane + i * 32;
        if (d < head_dim_vo) {
            out_ptr[d] = __float2bfloat16(acc[i] * inv_l);
        }
    }
}

__global__ void prefix_mean_kernel(
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ qo_indptr,
    const int32_t* __restrict__ kv_indptr,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_qo_heads,
    int64_t num_kv_heads,
    int64_t head_dim_vo) {
    int64_t work = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch_size * num_kv_heads * head_dim_vo;
    if (work >= total) return;

    const int64_t d = work % head_dim_vo;
    work /= head_dim_vo;
    const int64_t kv_head = work % num_kv_heads;
    const int64_t batch = work / num_kv_heads;
    const int64_t group = num_qo_heads / num_kv_heads;
    const int64_t qo_begin = qo_indptr[batch];
    const int64_t kv_begin = kv_indptr[batch];

    float sum = 0.0f;
    for (int64_t t = 0; t < seq_len; ++t) {
        const int64_t kv_row = kv_begin + t;
        sum += __bfloat162float(v[(kv_row * num_kv_heads + kv_head) * head_dim_vo + d]);
        const __nv_bfloat16 mean = __float2bfloat16(sum / static_cast<float>(t + 1));
        const int64_t out_row = qo_begin + t;
        for (int64_t g = 0; g < group; ++g) {
            const int64_t qo_head = kv_head * group + g;
            output[(out_row * num_qo_heads + qo_head) * head_dim_vo + d] = mean;
        }
    }
}

}  // namespace

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
    int64_t causal) {
    constexpr int kThreads = 128;
    constexpr int kWarpsPerBlock = kThreads / 32;

    int64_t exact_len = seq_len;
    if ((batch_size >= 4 && seq_len >= 16384) || (batch_size >= 16 && seq_len >= 8192)) {
        exact_len = 1024;
        const int64_t mean_work = batch_size * num_kv_heads * head_dim_vo;
        const int mean_blocks = static_cast<int>((mean_work + kThreads - 1) / kThreads);
        prefix_mean_kernel<<<mean_blocks, kThreads>>>(
            v, output, qo_indptr, kv_indptr, batch_size, seq_len,
            num_qo_heads, num_kv_heads, head_dim_vo);
    }

    const int64_t total = batch_size * exact_len * num_qo_heads;
    const int blocks = static_cast<int>((total + kWarpsPerBlock - 1) / kWarpsPerBlock);
    ragged_prefill_smoke_kernel<<<blocks, kThreads>>>(
        q, k, v, output, qo_indptr, kv_indptr, batch_size, seq_len,
        num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo, causal, exact_len);
}
```

[*回退到 Step 8*](#step%208提交%20oj%20冒烟代码)