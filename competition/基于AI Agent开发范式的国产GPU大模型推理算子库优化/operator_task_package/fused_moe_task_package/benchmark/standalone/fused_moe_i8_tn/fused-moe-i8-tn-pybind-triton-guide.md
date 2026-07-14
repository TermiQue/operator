# fused_moe_i8_tn Python / Triton / Pybind 使用说明

## 1. 目标

当前仓库已经为 `standalone/fused_moe_i8_tn` 提供了三套可用于 Python 层验证与对比的实现：

1. `pybind`
   调用 MACA / MCTLASS C++ kernel，通过 Python 扩展模块暴露给 Python。
2. `triton`
   使用 Triton 实现的 `fused moe` 路径，结构参考 vLLM 的 `fused_moe_kernel`。
3. `reference`
   纯 Python / NumPy 参考实现，用于结果校验，不用于性能。

这三套后端都已经接入统一测试和统一 benchmark 脚本，可以直接按后端切换。

## 2. 相关文件

### 2.1 C++ / Pybind

- [standalone/fused_moe_i8_tn/src/fused_moe_i8_tn_runner.h](/home/zguo/mcTlass/standalone/fused_moe_i8_tn/src/fused_moe_i8_tn_runner.h)
- [standalone/fused_moe_i8_tn/src/fused_moe_i8_tn_pybind.cpp](/home/zguo/mcTlass/standalone/fused_moe_i8_tn/src/fused_moe_i8_tn_pybind.cpp)
- [scripts/build_fused_moe_i8_tn_pybind.sh](/home/zguo/mcTlass/scripts/build_fused_moe_i8_tn_pybind.sh)

### 2.2 Triton

- [standalone/fused_moe_i8_tn/python/fused_moe_i8_tn_triton.py](/home/zguo/mcTlass/standalone/fused_moe_i8_tn/python/fused_moe_i8_tn_triton.py)

### 2.3 测试与性能

- [standalone/fused_moe_i8_tn/python/test_fused_moe_i8_tn_pybind.py](/home/zguo/mcTlass/standalone/fused_moe_i8_tn/python/test_fused_moe_i8_tn_pybind.py)
- [standalone/fused_moe_i8_tn/python/benchmark_fused_moe_i8_tn.py](/home/zguo/mcTlass/standalone/fused_moe_i8_tn/python/benchmark_fused_moe_i8_tn.py)
- [scripts/run_fused_moe_i8_tn_pybind_test.sh](/home/zguo/mcTlass/scripts/run_fused_moe_i8_tn_pybind_test.sh)
- [scripts/run_fused_moe_i8_tn_benchmark.sh](/home/zguo/mcTlass/scripts/run_fused_moe_i8_tn_benchmark.sh)

## 3. 默认远端环境

当前脚本已经默认适配远端环境：

```bash
PYTHON_BIN=/home/wtliu/miniforge3/envs/py310/bin/python
MACA_PATH=/opt/maca-20260318
LD_LIBRARY_PATH=$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/lib:$LD_LIBRARY_PATH
```

其中：

1. `py310` 环境中已确认存在：
   - `torch`
   - `triton`
   - `numpy`
2. `torch` 实际版本为：
   - `2.8.0+metax3.6.0.5`
3. `triton` 实际版本为：
   - `3.0.0`

如果你需要切换 Python 环境，也可以在命令前覆盖：

```bash
PYTHON_BIN=/path/to/python bash scripts/run_fused_moe_i8_tn_pybind_test.sh --backend triton
```

## 4. Pybind 编译

远端进入仓库后，编译 `pybind` 模块：

```bash
cd /home/acl_dnn/mcTlass
bash scripts/build_fused_moe_i8_tn_pybind.sh
```

编译成功后，会生成：

```bash
standalone/fused_moe_i8_tn/build/fused_moe_i8_tn_pybind.cpython-310-x86_64-linux-gnu.so
```

## 5. 正确性测试

### 5.1 只测 pybind

```bash
cd /home/acl_dnn/mcTlass
bash scripts/run_fused_moe_i8_tn_pybind_test.sh --backend pybind
```

### 5.2 只测 triton

```bash
cd /home/acl_dnn/mcTlass
bash scripts/run_fused_moe_i8_tn_pybind_test.sh --backend triton
```

### 5.3 跑全部后端

```bash
cd /home/acl_dnn/mcTlass
bash scripts/run_fused_moe_i8_tn_pybind_test.sh --backend all
```

支持的后端选项：

- `pybind`
- `triton`
- `reference`
- `all`

## 6. 性能测试

### 6.1 只测 triton

```bash
cd /home/acl_dnn/mcTlass
bash scripts/run_fused_moe_i8_tn_benchmark.sh --backend triton --warmup 5 --iters 20
```

### 6.2 跑全部后端

```bash
cd /home/acl_dnn/mcTlass
bash scripts/run_fused_moe_i8_tn_benchmark.sh --backend all --warmup 5 --iters 20
```

参数说明：

- `--warmup`
  预热次数
- `--iters`
  正式计时次数

## 7. 当前 Triton 实现说明

当前 Triton 后端不是最初那版“逐行加载、逐行计算”的简化实现，而是已经改成参考 vLLM `fused_moe_kernel` 的结构。

核心特征包括：

1. 使用 `sorted routed rows`
2. 使用 `block expert ids`
3. 使用 `num_tokens_post_padded`
4. 使用 grouped `pid_m / pid_n` 调度方式
5. 使用 `MUL_ROUTED_WEIGHT`
6. 使用 `int8_w8a8 + per_channel_quant` 这条固定路径

当前为了适配本仓库现有 `fused_moe_i8_tn` 输入定义，做了以下简化：

1. `HAS_BIAS=False`
2. `use_int8_w8a8=True`
3. `use_fp8_w8a8=False`
4. `use_int8_w8a16=False`
5. `group_k=0, group_n=0`
6. `naive_block_assignment=False`

因此它的目的目前是：

1. 让 Triton 路径和当前 `fused_moe_i8_tn` 测试数据对齐
2. 保持与 vLLM fused moe kernel 结构尽量接近
3. 在 MetaX + Triton 环境中可实际运行

## 8. 当前对比结论

### 8.1 正确性

远端实际验证结果表明：

1. `triton` 和 `reference` 的输出一致
2. `pybind` 也通过校验，但由于其输出走 BF16 路径，和 `reference/triton` 相比会存在 BF16 舍入差

这属于当前实现预期行为，不是错误。

### 8.2 性能

远端使用：

```bash
bash scripts/run_fused_moe_i8_tn_benchmark.sh --backend all --warmup 5 --iters 20
```

得到的结果为：

```text
pybind:fused_moe_i8_tn_topk1 benchmark: avg_ms=0.109936, TOPS=0.076305
pybind:fused_moe_i8_tn_topk2 benchmark: avg_ms=0.126724, TOPS=0.132392
pybind:fused_moe_i8_tn_topk3 benchmark: avg_ms=0.113708, TOPS=0.110660

triton:fused_moe_i8_tn_topk1 benchmark: avg_ms=9.841149, TOPS=0.000852
triton:fused_moe_i8_tn_topk2 benchmark: avg_ms=9.924612, TOPS=0.001690
triton:fused_moe_i8_tn_topk3 benchmark: avg_ms=9.938139, TOPS=0.001266

reference:fused_moe_i8_tn_topk1 benchmark: avg_ms=717.996218, TOPS=0.000012
reference:fused_moe_i8_tn_topk2 benchmark: avg_ms=1437.042784, TOPS=0.000012
reference:fused_moe_i8_tn_topk3 benchmark: avg_ms=1077.384005, TOPS=0.000012
```

当前结论：

1. `pybind` 最快
2. `triton` 明显快于 `reference`
3. `triton` 仍显著慢于 `pybind`

也就是说，当前 Triton 路径已经具备：

1. 结构正确
2. 数值正确
3. 能在远端真实运行

但它还不是性能优化完成版。

## 9. 推荐后续工作

如果后续继续优化，优先建议做：

1. 分析 MetaX Triton backend 的 `tl.dot(int8, int8)` lowering 是否真正命中高效硬件路径
2. 调整 `BLOCK_SIZE_K / BLOCK_SIZE_N / GROUP_SIZE_M`
3. 进一步减少 Python 侧 routing / packing 的额外开销
4. 对齐 C++ kernel 的 tile 组织方式，逐步缩小 `pybind` 与 `triton` 的性能差距
