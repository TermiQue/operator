# Agent 使用说明
本说明涵盖三个算子的 Agent 辅助能力（FlashInfer、FlashAttention、Fused MoE）以及使用 Agent 时的推荐方式和注意事项。

## 一、模块功能速查
### 1.FlashInferFlashInfer：
在本模块中，Agent 可以帮助你完成以下任务：

#### 环境检查

```plaintext
请帮我检查当前环境是否满足 FlashInfer 运行要求，包括 GPU、Python、PyTorch 和 flashinfer 依赖。
```

#### 运行测试

```plaintext
请帮我运行 bench_batch_decode.py 脚本，执行 BatchDecode 的基准测试。
```

#### 分析结果

```plaintext
请帮我读取最新的 CSV 结果文件，分析各参数配置下的性能表现，找出带宽最高和 TFLOPs 最高的配置。
```

#### 问题排查

```plaintext
运行时报错 out of memory，请帮我分析原因并给出解决方案。
```

#### 代码理解

```plaintext
请帮我解释 bench_common.py 中 run_with_profiler 函数的工作原理。
```

### 2. FlashAttention：
在本模块中，Agent 可用于以下场景：
#### Prompt 模板

**环境验证：**

```Plain
请帮我验证当前环境是否满足 FlashAttention KV-Cache Benchmark 的运行要求，包括：
1. 沐曦 GPU 是否可见
2. PyTorch 版本和 CUDA 支持
3. flash-attn 和 einops 是否已安装

```

**参数配置建议：**

```Plain
我需要测试 headdim=128 和 headdim=256 的性能差异，请帮我推荐合适的 batch_sizes 和 seq_lens_kv 扫描范围。

```

**结果分析：**

```Plain
请帮我分析这份 benchmark 结果 CSV 文件，找出峰值带宽配置和 OOM 边界。

```
---
### 3. Fused MoE：
在本模块中，Agent可以帮助你完成以下任务：

#### 环境检查
    
    ```plaintext
    我正在算力平台部署 fused_moe_baseline 源码。
    需要的环境信息如下：
    - Python 3.12
    - g++ 13.3.0
    - mxcc 已安装
    - numpy / torch / triton 已安装
    
    请帮我确认：
    1. 当前环境是否满足编译与运行要求？
    2. 是否有潜在的不兼容风险（如 Python 与 libpython 版本）？
    ```
    
#### 运行测试
    
    ```plaintext
    请帮我运行 scripts/run_fused_moe_i8_tn_pybind_test.sh 脚本 
    ```
    
#### 分析结果
    
    ```plaintext
    这是性能测试结果：
    pybind:    avg_ms=0.30,  TOPS=0.027
    triton:    avg_ms=19.01, TOPS=0.0004
    reference: avg_ms=1685,  TOPS=0.000005
    
    请分析：
    1. 为什么 pybind 比 Triton 快这么多？
    2. TOPS 指标是否可信？
    3. 当前结果是否已经具备提交价值？
    ```
    
#### 报错检查
    
    ```plaintext
    编译 pybind 时出现以下错误：
    /usr/bin/ld: cannot find -lpython3.10
    
    已知：
    - 使用的是 Conda Python 3.10
    - mxcc 编译正常
    
    请一步一步告诉我：
    1. 错误原因是什么？
    2. 如何用 find 命令定位 libpython3.10.so？
    3. 如何在 build_fused_moe_i8_tn_pybind.sh 中正确指定路径？
    ```
    
#### 代码理解
    
    ```plaintext
    请帮我梳理释 benchmark_fused_moe_i8_tn.py 代码整体框架
    ```
    
#### KernelSwift 系统搜索算子
    

```plaintext
请帮我在算子广场检索 fused_moe 算子
```
## 二、推荐工作方式
使用 Agent 时，不要一开始就让它“直接优化到最快”。推荐节奏是：

先让它检查目录。

再让它跑通原始 baseline。

然后让它只做一处小修改。

每次修改后必须 build、test、benchmark。

每轮都记录结果。

## 三、注意事项
使用 Agent 不要相信没有命令输出支持的环境判断。

不要让 Agent 同时修改很多文件。

不要让 Agent 修改测试标准。