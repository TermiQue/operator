# 模力方舟Agent部署准备教程

预计用时：40 分钟

本教程带你完成比赛开发环境准备：申请算力、进入模力方舟镜像、准备 baseline 代码、配置 Agent 工具和模力方舟 API Token，并完成一次本地 build / test / benchmark 验证。

本教程截止到“本地 Agent 跑通”。OJ 适配与统一提交格式由主办方对 baseline 进行整理，有专门教学。

## 一、教程定位

本教程面向第一次在模力方舟算力实例中使用 Agent 开发的同学。完成后，你将拥有一个可直接进行代码阅读、编译、测试、benchmark 和小步优化的本地环境。

本节重点解决五件事：

1.  专属镜像与算力环境准备
    
2.  MXMACA / MACA 软件栈认识
    
3.  Agent 工具安装与配置
    
4.  源码与项目目录准备
    
5.  连通性与可用性测试
    

基础镜像：

```plaintext
PyTorch-Agent / 2.8.0 / Python 3.12 / maca 3.7.2.1
```

## 二、学习目标

完成本教程后，你将能够：

1.  申请并启动模力方舟算力实例。
    
2.  进入 JupyterLab，并打开终端。
    
3.  确认镜像内的 Python、PyTorch、MACA / MXMACA、mxcc 等环境可用。
    
4.  将 baseline 源码放入 `/data` 工作目录。
    
5.  配置模力方舟 API Token。
    
6.  使用 OpenCode 作为 Agent 工具。
    
7.  让 Agent 在本地完成 candidate 0 的 build、test、benchmark。
    
8.  让 Agent 进行一次 candidate 1 小步优化，并记录 benchmark 对比。
    

## 三、适用对象

本教程适合：

1.  已报名比赛、需要在统一镜像中开发 baseline 的选手。
    
2.  希望使用 Agent 辅助阅读代码、定位入口、运行测试和做小步优化的选手。
    
3.  对 MACA / MXMACA 环境不熟悉，但希望先把本地开发闭环跑通的选手。
    

你不需要提前掌握 MACA kernel 开发。本教程的目标是先把环境和 Agent 闭环跑通。

## 四、前置准备

开始前，请确认你已经准备好以下内容。

### 4.1 账号与算力

1.  在沐曦开发者社区领取算力券
   *   领取链接：[https://developer.metax-tech.com/activities/6](https://developer.metax-tech.com/activities/6)
        
        *   登录平台
            
        
        ![platform login](https://origin.picgo.net/2026/06/04/platform-login626620122b08424d.png)
        
        *   首次登录需要先进行注册（使用邮箱或者手机号进行注册）
            
        
        ![platform registration](https://origin.picgo.net/2026/06/04/platform-registrationdb267074af39bf4c.png)
        
        *   登录成功后进行第二步-邮箱验证，填入自己的邮箱。
            
        
        ![email verification](https://origin.picgo.net/2026/06/04/email-verificationc3f391bb747318e0.png)
        
        *   第三步，提交申请。
            
        
        ![submit application](https://origin.picgo.net/2026/06/04/submit-application3bf7ac4724e13ae8.png)
        
        *   获得兑换码
            
    
    ![get redeem code](https://origin.picgo.net/2026/06/04/get-redeem-code3f6a20e5f9cbbd38.png)领取链接：https://developer.metax-tech.com/activities/6
    
   
2.  在模力方舟平台兑换算力券：
    
    *   平台链接：[https://ai.gitee.com/](https://ai.gitee.com/)
        
    *   1.登录模力方舟平台
        
    
    ![ai.gitee login](https://origin.picgo.net/2026/06/04/ai.gitee-login7d9fe2b5e35e3a92.png)
    
    *   2.进入费用中心 - 算力券 ， 点击右上角“兑换”
        
    
    ![redeem compute voucher](https://origin.picgo.net/2026/06/04/redeem-compute-voucher0eb15e2f3f9b7bbd.png)
3.  租用算力：

    *   模力方舟算力市场链接：https://ai.gitee.com/compute
        
    *   选择沐曦芯片厂商，并根据项目要求选择相应的配置。
        
    
    ![rent compute](https://origin.picgo.net/2026/06/04/rent-compute1197cc6d884ce429.png)
    
4.  创建实例：

专属镜像文件：

![create instance1](https://origin.picgo.net/2026/06/04/create-instance10ab33dd1b7e14727.png)

![create instance2](https://origin.picgo.net/2026/06/04/create-instance23666efb720fefa60.png)

![create instance3](https://origin.picgo.net/2026/06/04/create-instance3f18b4323644d3447.png)

  进入算力容器，刚创建的实例默认开机状态，点击工具-lab开始项目创作。

**重要说明：**由于本次使用的是预装的专属镜像，环境中已经默认安装并配置好了 PyTorch、FlashAttention、einops 等所有依赖包。因此在启动实例后，无需再进行繁琐的依赖库版本验证即可直接进入测试环节。

    
5.  日常开发与调试：选用沐曦 GPU，显存 16–32 GB 即可满足需求。

    **注意：运行性能跑分（Benchmark）时，请务必使用整张单卡（64 GB）。**


### 4.2 购买模型资源包

在兑换代金/算力券之后，点击模力方舟主页左侧导航栏“Token资源包”

点击右上角“购买订阅套餐” -> 选择购买沐曦Token套餐Plus 29元/月

支持MiniMax 2.7，Qwen3.6系列，Qwen-Image，IndexTTS-2等模型
模型购买链接：[https://ai.gitee.com/serverless-api/packages/13525?tab=subscription](https://ai.gitee.com/serverless-api/packages/13525?tab=subscription)
MiniMax-M2.7：推荐用于通用 AI 编程、Agent/OpenClaw 工作流、长上下文文本生成与代码辅助场景。
Qwen3.5 / Qwen3.6：推荐用于需要较强推理、Function Calling、多模态理解或 OpenClaw Agent 执行的复杂任务。
Qwen-Image：推荐用于文生图、图像生成、图像处理与视觉创意类场景。

### 4.3 设置API Key

你需要一个模力方舟 API Key，用于让 Agent 调用模型。

完成 4.2 的模型资源包购买后即可申请。

申请方式：

1.  点击进入模力方舟网页首页
    
2.  从左侧导航栏进入“访问令牌”或“API Key”页面。
    
3.  新建一个访问令牌。
    
4.  复制并保存该 API Key
    

后续教程中统一使用环境变量名：

```plaintext
MOARK_API_KEY
```

### 4.4 本地文件（可选）

##### 如果还未获取baseline文件可跳过这一步先进行agent部署。

请准备 baseline 压缩包或源码目录。示例目录名：

```plaintext
fusedmoe_v2.1
```

在模力方舟实例中，建议将代码放在：

```plaintext
/data/fusedmoe_v2.1
```

如果你的比赛材料中提供的是其他 baseline 名称，请以实际发放文件为准。

## 五、知识预备

### 5.1 什么是专属镜像

专属镜像是主办方提前准备好的开发环境。它通常已经安装好 PyTorch、MACA / MXMACA 软件栈、编译器、驱动运行时和常用依赖。

本教程使用的镜像是：

```plaintext
PyTorch-Agent / 2.8.0 / Python 3.12 / maca 3.7.2.1
```

这意味着你不需要从零安装 PyTorch、MACA、mxcc 等底层组件。你需要做的是进入镜像、确认环境、放入源码并运行 baseline。

### 5.2 什么是 MACA / MXMACA

MACA / MXMACA 是本次算力环境中的核心软件栈，负责让程序在对应加速硬件上编译和运行。

你会在命令中看到这些路径或工具：

```plaintext
MACA_PATH
mxcc
mxgpu_llvm
LD_LIBRARY_PATH
```

常见路径可能是：

```plaintext
/opt/maca
/opt/maca-20260318
```

实际路径以镜像内查询结果为准。

### 5.3 什么是 Agent

Agent 是能读代码、运行命令、修改文件并总结结果的编程助手。本教程推荐使用 OpenCode，并接入模力方舟的 MiniMax 模型。

本教程中推荐模型：

```plaintext
MiniMax-M2.7
```

你也可以使用沐曦-模型资源包中支持的其他模型。注意：模型名必须以 API 返回的准确模型 id 为准，并且需要通过 `/v1/chat/completions` 实测可用。

## 六、项目实践：环境准备与本地迭代

### 步骤 1：创建算力实例

目标：使用指定镜像创建比赛开发实例。

操作：

1.  打开模力方舟控制台，点击上方控制栏中的“算力市场”
    
2.  进入算力实例创建页面，选择沐曦专区VRAM16-32GB算力容器
    
3.  选择镜像：
    
    ```plaintext
    PyTorch-Agent / 2.8.0 / Python 3.12 / maca 3.7.2.1
    ```
    
4.  点击创建实例。
    
5.  创建完成后回到主页点击左侧导航栏进入“算力容器”界面
    

### 步骤 2：进入 JupyterLab

进入镜像中的开发环境。

操作：

1.  在实例列表中找到刚刚创建的实例。
    
2.  点击“JupyterLab”或类似按钮。
    
    如图所示右侧按钮
    
    ![image](https://origin.picgo.net/2026/06/04/jupyterlab-entryf4a35ad7aeb7b847.png)
    
3.  页面打开后，点击 `Terminal` 新建终端。
    

### 步骤 3：确认基础环境

目标：确认 Python、PyTorch、mxcc、MACA 路径可用。

在 JupyterLab Terminal 中执行：

```plaintext
pwd
python --version
which python

```

期望输出：

![image](https://origin.picgo.net/2026/06/04/terminal-python-checkd568cc610011191e.png)

继续检查 PyTorch：

```plaintext
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY
```

期望输出：torch: 2.8.0+metax3.7.1.3

检查 MACA / mxcc：

```plaintext
which mxcc || true
find /opt -name mxcc 2>/dev/null | head
ls /opt
```

预期结果：

`which mxcc` 如果已经输出 mxcc 路径，说明环境变量已经配置好，可以继续下一步。

如果 `which mxcc` 没有结果，但 `find /opt -name mxcc` 找到了类似路径：

```plaintext
/opt/maca/mxgpu_llvm/bin/mxcc
```

或：

```plaintext
/opt/maca-20260318/mxgpu_llvm/bin/mxcc
```

就设置环境变量。以 `/opt/maca` 为例：

```plaintext
export MACA_PATH=/opt/maca
export PATH=$MACA_PATH/mxgpu_llvm/bin:$PATH
export LD_LIBRARY_PATH=$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$LD_LIBRARY_PATH
```

预期结果：

```plaintext
which mxcc
```

能够输出 `mxcc` 路径。

### 步骤 4：准备源码目录

目标：将三个 baseline 放到 `/data` 下（左侧文件栏），作为后续 Agent 工作目录。

进入 `/data`：

```plaintext
cd /data
ls
```

假设上传后的文件名是：

```plaintext
fusedmoe_v2.1.zip
```

解压：

```plaintext
cd /data
python - <<'PY'
import zipfile
from pathlib import Path

zip_path = Path("fusedmoe_v2.1.zip")
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(".")
print("extracted", zip_path)
PY
```

说明：请把上面整段 Python 命令一次性复制到终端中运行。

进入项目：

```plaintext
cd /data/fusedmoe_v2.1
find . -maxdepth 3 -type f | sort | head -120
```

注意：

解压后目录名可能会改变。如果 `cd /data/fusedmoe_v2.1` 提示目录不存在，请先执行 `ls /data` 查看实际解压出的目录名，再进入对应目录。

你现在应该能看到类似入口：

```plaintext
scripts/build_fused_moe_i8_tn_pybind.sh
scripts/run_fused_moe_i8_tn_pybind_test.sh
scripts/run_fused_moe_i8_tn_benchmark.sh
standalone/fused_moe_i8_tn/
```

源码目录准备完成，后续命令都在 `/data/fusedmoe_v2.1` 或实际解压出的 baseline 目录中执行。

### 步骤 5：配置模力方舟 API Key

目标：让后续 Agent 能调用模力方舟模型。

在终端中设置：

```plaintext
echo 'export MOARK_API_KEY="你的真实 API Key"' >> ~/.bashrc
source ~/.bashrc
```

确认环境变量存在：

```plaintext
test -n "$MOARK_API_KEY" && echo "MOARK_API_KEY is set" || echo "MOARK_API_KEY is missing"
```

期望输出：

MOARK\_API\_KEY is set

注意：

不要把 API Key 粘贴到公开仓库、提交代码或截图中。

如果新开 Terminal 后显示 `MOARK_API_KEY is missing`，说明环境变量还没有写入当前 shell。重新执行上面的 `source ~/.bashrc` 即可。

如果后续更换 API Key，请先打开 `~/.bashrc` 删除旧的 `MOARK_API_KEY` 行，或执行下面命令追加新 key 后重新加载：

```bash
echo 'export MOARK_API_KEY="你的新 API Key"' >> ~/.bashrc
source ~/.bashrc
```

如果同一个文件里出现多行 `MOARK_API_KEY`，通常最后一行会生效。为了避免混乱，建议只保留一行。

### 步骤 6：测试模力方舟 API 连通性

目标：确认 Token 与模型可用。

测试文档中使用 `MiniMax-M2.7`：

```plaintext
curl https://api.moark.com/v1/chat/completions \
  -H "Authorization: Bearer $MOARK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMax-M2.7",
    "messages": [{"role": "user", "content": "你好，请用一句话回复。"}],
    "stream": false
  }'
```

预期结果：

返回 JSON，并且能看到模型回复内容。

#### 替换模型

只能替换沐曦模型资源包中支持的模型。先用以下命令查询当前 API Key 能看到的模型 id：

```bash
curl https://api.moark.com/v1/models \
  -H "Authorization: Bearer $MOARK_API_KEY" \
  | python -m json.tool
```

返回结果中 `"id"` 后面的字段就是模型 id。请复制完整 id，不要手动猜模型名。

注意：`/v1/models` 中能看到某个模型，不代表它一定能用于 `/v1/chat/completions`。复制模型 id 后，还需要把步骤 6 第一段 curl 中的 `"model"` 字段改成该 id 并再次测试。只有 curl 实测返回正常回复的模型，才建议写入 OpenCode 配置。

### 步骤 7：安装 OpenCode

目标：安装并启动 Agent 工具。

先检查 Node.js 和 npm：

```plaintext
node --version
npm --version
```

如果镜像中已经安装 npm，执行：

```plaintext
npm install -g opencode-ai
```

验证：

```plaintext
opencode --version
```

### 步骤 8：配置 OpenCode 使用 Moark

目标：让 OpenCode 通过模力方舟 API 调用 `MiniMax-M2.7`。

创建配置目录：

```plaintext
mkdir -p ~/.config/opencode
```

写入配置（可以后续覆写添加模型）：

```plaintext
python - <<'PY'
import json
import os
from pathlib import Path

key = os.environ["MOARK_API_KEY"]

cfg = {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
        "moark": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Moark",
            "options": {
                "baseURL": "https://api.moark.com/v1",
                "apiKey": key,
            },
            "models": {
                "MiniMax-M2.7": {
                    "name": "MiniMax-M2.7",
                    "limit": {
                        "context": 200000,
                        "output": 200000,
                    },
                },

                # 可选：如需添加其他模型，请先用步骤6中 curl 指令验证该模型可用，
                # 然后取消下面示例配置的注释，并把“其他模型名”
                # 替换为 /v1/models 返回的准确模型 id。
                #
                # "其他模型名": {
                #     "name": "其他模型名",
                #     "limit": {
                #         "context": 200000,
                #         "output": 200000,
                #     },
                # },
            },
        }
    },
}

config_path = Path.home() / ".config/opencode/opencode.json"
config_path.parent.mkdir(parents=True, exist_ok=True)
config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))

print("wrote", config_path)
PY
```

如需添加多个模型，可以在配置文件的 `models` 字段中添加多个下面这种模块。添加前必须先用步骤 6 的 curl 命令确认该模型可用于 `/v1/chat/completions`：

```python
"其他模型名": {
    "name": "其他模型名",
    "limit": {
        "context": 200000,
        "output": 200000,
    },
},
```

注意：重新运行上面的 Python 配置脚本会覆盖整个 `~/.config/opencode/opencode.json`。如果要保留多个模型，请在同一次配置中把所有模型都写进 `models` 字段。

检查配置：

```plaintext
grep -n "baseURL\|MiniMax\|apiKey" ~/.config/opencode/opencode.json
```

如果配置中显示：

```plaintext
"apiKey": "${MOARK_API_KEY}"
```

说明 OpenCode 可能不会自动展开环境变量，导致 `Unauthorized`。为了先跑通，本教程推荐用上面的 Python 脚本把真实 key 写入本机配置文件。

#### 如何使用其他模型：

如果需要使用其他模型，请先用步骤 6 的 curl 方法把 "model" 字段改成目标模型名，确认 API 能返回结果。

确认可用后，再把该模型加入 OpenCode 配置中的 models 字段。启动 OpenCode 后使用 /models 切换 Moark 下的模型。

### 步骤 9：启动 OpenCode

目标：进入项目目录并启动 Agent。

```plaintext
cd /data/fusedmoe_v2.1
opencode
```

页面加载完毕后如果需要选择模型，请输入：

```plaintext
/models
```

只能选择 Moark 下面已经配置过、且 curl 实测可用的模型。其他模型可能因为资源包、接口类型或模型名不匹配而报错。

进入后，先发送以下测试提示词测试运行情况：

```text
请通过真实 shell 命令检查当前目录、Python 版本、mxcc 路径、MACA_PATH 和 baseline 脚本入口。然后检查当前目录结构，并找出 fusedmoe_v2.1 的 build、test、benchmark 入口。先不要修改代码。
```


## 七、常见问题

### 问题 1：OpenCode 报 Unauthorized

可能原因：

1.  `MOARK_API_KEY` 没有设置。
    
2.  OpenCode 配置没有读到真实 key。
    
3.  配置里写的是字面量 `"${MOARK_API_KEY}"`。
    

检查：

```plaintext
test -n "$MOARK_API_KEY" && echo "MOARK_API_KEY is set" || echo "MOARK_API_KEY is missing"
grep -n "apiKey" ~/.config/opencode/opencode.json
```

解决：

重新执行“步骤 8：配置 OpenCode 使用 Moark”中的 Python 配置脚本。

### 问题 2：MiniMax-M2.1 不可用

如果接口返回资源包不支持 `MiniMax-M2.1`，直接使用：

```plaintext
MiniMax-M2.7
```

或任意沐曦-模型资源包包含的模型

本教程使用 `MiniMax-M2.7`。

### 问题 3：找不到 mxcc

检查：

```plaintext
which mxcc || true
find /opt -name mxcc 2>/dev/null | head
```

如果找到了 mxcc，但 `which mxcc` 为空，说明环境变量没有设置。根据实际路径设置 `MACA_PATH`、`PATH` 和 `LD_LIBRARY_PATH`。

### 问题 4：Python.h file not found

先检查 Python 头文件：

```plaintext
python - <<'PY'
import sysconfig, os
inc = sysconfig.get_paths()["include"]
print("python include:", inc)
print("Python.h exists:", os.path.exists(os.path.join(inc, "Python.h")))
print("LIBDIR:", sysconfig.get_config_var("LIBDIR"))
print("LDLIBRARY:", sysconfig.get_config_var("LDLIBRARY"))
PY
```

如果 `Python.h exists: True`，但 build 仍失败，请用：

```plaintext
bash -x scripts/build_fused_moe_i8_tn_pybind.sh
```

查看完整编译命令中是否带上了 Python include 路径。

### 问题 5：加载到旧的 .so

如果测试时报类似旧 Python 动态库错误，例如：

```plaintext
libpython3.10.so.1.0: cannot open shared object file
```

说明可能加载了旧编译产物。可以清理后重新 build：

```plaintext
cd /data/fusedmoe_v2.1
rm -rf build standalone/fused_moe_i8_tn/build
find . -name "*.so" -delete
bash scripts/build_fused_moe_i8_tn_pybind.sh
```

### 问题 6：Agent 编造路径或测试结果

处理方式：

让 Agent 先执行真实命令：

```plaintext
请不要猜测。先运行 pwd、find . -maxdepth 3 -type f、python --version、which mxcc，并基于真实输出继续分析。
```