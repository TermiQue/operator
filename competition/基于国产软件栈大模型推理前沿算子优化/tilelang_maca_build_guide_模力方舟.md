# TileLang on MACA 编译与导入完整指南

> 记录从源码编译 tilelang-metax 并启用 MACA 后端的完整过程  
> 环境：MetaX C500 / MACA 3.5.3.20 / Python 3.12 / Conda

---

## 1. 项目背景

**代码位置**：`/app/tilelang-metax`（沐曦适配分支，dev 分支，commit f1ca0fb9）

**版本信息**：
- 基础版本：`0.1.9`
- Git 描述：`v0.1.9-73-gf1ca0fb9`
- 构建标识：`+maca.gitf1ca0fb9`（启用 MACA 后）

---

## 2. 初始状态排查

### 2.1 检查是否已安装

```bash
pip show tilelang
# 结果：未安装

python -c "import tilelang; print(tilelang.__version__)"
# 结果：import 失败
```

### 2.2 查找源码位置

```bash
find / -name "tilelang*" -type d 2>/dev/null | head -10
# 结果：/app/tilelang-metax
```

### 2.3 检查 GPU 环境

```bash
mx-smi
# 结果：MetaX C500，MACA 3.5.3.20，/opt/maca 已安装
```

---

## 3. 第一次编译尝试（失败）

### 3.1 命令

```bash
cd /app/tilelang-metax
pip install -e . -v
```

### 3.2 失败原因

**错误信息**：
```
ninja: error: Makefile:5: expected '=', got ':'
default_target: all
              ^ near here
```

**根因**：`/app/tilelang-metax/build/` 目录下已有之前用 **Unix Makefiles** 生成的构建产物（包含 `Makefile`），但 pip 安装时使用了 **Ninja** 构建系统���Ninja 误把 `Makefile` 当成 ninja 构建文件解析，导致语法错误。

---

## 4. 清理后重新编译（成功，但缺少 MACA）

### 4.1 命令

```bash
cd /app/tilelang-metax
rm -rf build
pip install -e . -v
```

### 4.2 编译结果

- **构建成功**，版本号为 `0.1.9+cuda.gitf1ca0fb9`
- **但 MACA 后端未启用**：CMake 日志显示 `-- CUDA toolkit not found; building without CUDA support by default.`
- 未检测到 `USE_MACA` 环境变量，因此 TVM 未注册 `maca` target

---

## 5. 导入报错与修复

### 5.1 导入失败

```bash
python -c "import tilelang"
```

**错误信息**：

```
ValueError: Target kind "maca" is not defined.
Target creation from string failed: maca
```

**根因**：`tilelang/utils/target.py` 的 `determine_target("auto")` 检测到系统有 MACA 环境（`mxcc.find_maca_path()` 成功），于是返回 `"maca"`，但当前构建未包含 MACA 后端，`Target("maca")` 创建失败。

### 5.2 修复 1：gemm_sp.py 导入保护

**文件**：`tilelang/language/experimental/gemm_sp.py`

**修改前**：
```python
_is_maca_target = target_is_maca(determine_target(return_object=True))
```

**修改后**：
```python
try:
    _is_maca_target = target_is_maca(determine_target(return_object=True))
except Exception:
    _is_maca_target = False
```

**作用**：模块导入时如果 maca target 不可用，静默降级为 `False`，避免阻塞整个包的导入。

### 5.3 修复 2：determine_target() 降级保护

**文件**：`tilelang/utils/target.py`

**修改前**：

```python
if is_maca_available:
    return_var = "maca"
```

**修改后**：

```python
if is_maca_available:
    try:
        Target("maca")
        return_var = "maca"
    except Exception:
        is_maca_available = False
if is_maca_available:
    return_var = "maca"
```

**作用**：返回 "maca" 之前先验证 `Target("maca")` 是否真的能创建成功，不能则降级到 CUDA/HIP/Metal 等其他后端。

---

## 6. 启用 MACA 后端重新编译（最终成功）

### 6.1 关键发现

查看 `CMakeLists.txt` 和 `cmake/FindMACA.cmake`：

```cmake
# CMakeLists.txt
elseif($ENV{USE_MACA})
    set(USE_MACA ON)

# cmake/FindMACA.cmake
macro(find_maca use_maca)
  if(IS_DIRECTORY /opt/maca)
    set(__maca_sdk /opt/maca)
  endif()
  find_library(MACA_MACAMCC_LIBRARY mcruntime ${__maca_sdk}/lib)
endmacro()
```

**结论**：需要显式设置 `USE_MACA=ON` 环境变量，CMake 才会启用 MACA 后端。

### 6.2 编译命令

```bash
cd /app/tilelang-metax
rm -rf build
export USE_MACA=ON
pip install -e . -v
```

### 6.3 编译过程

1. **安装构建依赖**：`scikit-build-core`, `cython`, `z3-solver`, `patchelf`
2. **CMake 配置**：
   ```
   -- Found MACA_INCLUDE_DIRS=/opt/maca/include
   -- Found MACA_MACAMCC_LIBRARY=/opt/maca/lib/libmcruntime.so
   ```
3. **Ninja 编译**：692 个编译目标，约 1 分 38 秒完成
4. **生成 wheel**：`tilelang-0.1.9+maca.gitf1ca0fb9-cp38-abi3-linux_x86_64.whl`

### 6.4 版本变化

| 阶段 | 版本号 | 说明 |
|------|--------|------|
| 未启用 MACA | `0.1.9+cuda.gitf1ca0fb9` | 无 MACA 后端 |
| 启用 MACA 后 | `0.1.9+maca.gitf1ca0fb9` | ✅ MACA 后端已编译 |

---

## 7. 验证安装

### 7.1 基础导入验证

```bash
python -c "import tilelang; print(tilelang.__version__)"
# 输出：0.1.9+maca.gitf1ca0fb9
```

### 7.2 MACA Target 验证

```bash
python -c "from tvm.target import Target; t = Target('maca'); print(t)"
# 输出：
# maca -keys=maca,gpu -max_local_memory_per_block=4095 -max_num_threads=1024
#   -max_shared_memory_per_block=65536 -max_threads_per_block=1024
#   -mcpu=xcore1000 -mtriple=mxc-metax-macahca -thread_warp_size=64
```

### 7.3 Kernel 测试验证

```bash
cd /app/tilelang-metax
python -m pytest testing/python/kernel/test_tilelang_kernel_gemm.py::test_gemm_f16f16f32_nn -v
# 结果：PASSED ✅
```

---

## 8. 测试通过率汇总

| 测试模块 | 启用 MACA 前 | 启用 MACA 后 |
|----------|-------------|-------------|
| `testing/python/cpu/` | 14/14 ✅ | 14/14 ✅ |
| `testing/python/arith/` | 87/87 ✅ | 87/87 ✅ |
| `testing/python/kernel/` | ❌ 全部失败 | **5 passed, 2 failed, 9 skipped** |
| `testing/python/transform/` | ❌ 8 errors | **164 passed** |
| `testing/python/language/` | ❌ 141 failed | **330 passed** |

**仅剩失败**：
- `test_gemm_i8i8i32_nt` — INT8 精度偏差
- `test_gemm_i8i8i32_tn` — INT8 精度偏差

---

## 9. 复现步骤（一键脚本）

```bash
#!/bin/bash
set -e

# 1. 进入源码目录
cd /app/tilelang-metax

# 2. 清理旧构建
rm -rf build

# 3. 启用 MACA 后端
export USE_MACA=ON

# 4. 编译安装（可编辑模式）
pip install -e . -v

# 5. 验证导入
python -c "import tilelang; print('Version:', tilelang.__version__)"
python -c "from tvm.target import Target; print('MACA target:', Target('maca'))"

# 6. 运行核心测试
python -m pytest testing/python/kernel/test_tilelang_kernel_gemm.py::test_gemm_f16f16f32_nn -v

echo "TileLang MACA build completed successfully!"
```

---

## 10. 关键踩坑记录

| 坑 | 现象 | 根因 | 解决 |
|----|------|------|------|
| **Ninja vs Makefile 冲突** | `ninja: error: Makefile:5: expected '='` | build 目录残留旧 Makefile | `rm -rf build` |
| **未启用 MACA** | `Target kind "maca" is not defined` | 缺少 `USE_MACA=ON` | 重新导出环境变量并编译 |
| **导入时急切检测** | `import tilelang` 崩溃 | `gemm_sp.py` 模块级别求值 | `try-except` 保护 |
| **auto-detect 降级缺失** | 所有测试崩溃 | `determine_target` 直接返回未注册的 target | 创建前验证 target 可用性 |

---

*文档由实际操作过程整理，基于 tilelang-metax dev 分支（commit f1ca0fb9）*