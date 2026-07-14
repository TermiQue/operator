# Fused MoE GEMM 优化实验记录

> 重要范围说明（2026-07-13）：第 0～24 轮使用仓库旧版
> `fusedmoe_benchmark.py` 的两组样例进行探索，不能直接视为正式 XPU-OJ
> 提升。正式成绩必须以 `submission.py` 的 `run_kernel(...)` 接口、OJ 三个
> 测试点时间与 OJ 分数为准。旧实验仍可用于筛选优化方向，但所有保留改动
> 都需要在正式接口和正式 shape 上重新验证。

## 1. 实验约定

- 硬件：MetaX C500，50% 算力切片，32 GB 显存
- 服务器命名（2026-07-13 起）：原 32 GB 实例称为“服务器-32-1”；新申请的
  两个 64 GB 实例分别称为“服务器-64-1”和“服务器-64-2”。服务器地址和
  账号密码不写入本日志。
- 软件：MACA 3.7.1.5、PyTorch 2.8.0+metax3.7.1.3、TileLang 0.1.9
- 正确性门槛：官方功能测试 2/2 通过
- 性能测量：官方 `fusedmoe_benchmark.py`，warmup 10 次、迭代 100 次
- 大配置：`dhidden=7168, dexpert=2048, experts=8, topk=4, bs=4, seqlen=8192`
- 小配置：`dhidden=3584, dexpert=1024, experts=4, topk=2, bs=8, seqlen=4096`
- 提升计算：`(原始耗时 - 当前耗时) / 原始耗时 * 100%`
- 原始基线：大配置 424.856953 ms，小配置 61.963486 ms

### 1.1 正式 XPU-OJ 评测约定（2026-07-13 更新）

- 唯一提交入口：无装饰器、无类型标注的 `run_kernel(...)`，直接原地写 `out`。
- GPU 计算只能使用 TileLang，不能用 PyTorch 完成算子计算。
- `out` 是唯一 INOUT 参数；其余张量只读；padding 行必须保持为 0。
- 正式范围：`d_hidden` 2048/7168，`d_expert` 2048/8192，experts 16/32/64，
  `group_sum` 2272～9088，`block_token=128`。
- 正式评测有三个测试点；每点先预热再用 GPU Event 多次计时并取平均。
- 总分为各测试点分数算术平均，总时为各测试点耗时之和。
- OJ baseline 模板公开记录约为 50 分、66 ms；后续优化基线应以我们自己的
  首次 Accepted 提交结果为准。
- TileLang 沙箱仅允许 `torch`、`tilelang`、`tilelang.language`、
  `tilelang.intrinsics`、`math`；提交代码不能依赖 `torch.nn`、`typing`、
  相对导入或通配符导入。

### 1.2 当前正式 OJ 基线（提交 51562）

- 提交时间：2026-07-13 16:14:32
- 状态：Accepted
- 总分：53.33
- 总耗时：58 ms
- 测试点：#1 54 分 / 9 ms，#2 53 分 / 16 ms，#3 53 分 / 32 ms
- 相对官方公开模板的 50 分、66 ms：得分提高 6.66%，总耗时降低 12.12%。
- 此结果是后续正式 OJ 优化的比较基线；旧 benchmark 的 12.10% 结果不再用于
  代替正式比赛成绩。

### 1.3 当前正式 OJ 最佳（提交 52152）

- 提交时间：2026-07-13 22:24:12
- 状态：Accepted
- 总分：60.67
- 总耗时：44 ms
- 测试点：#1 61 分 / 7 ms，#2 60 分 / 13 ms，#3 61 分 / 24 ms
- 相对提交 52062：得分提高 4.01%，总耗时降低 8.33%；相对提交 51562：得分提高 13.76%，总耗时降低 24.14%。
- 核心结构：128 行父块拆成两个 64 行计算块，保持 gate/up 与 down 总计两次
  kernel launch，512 threads；无有效行的 64 行子块跳过 GEMM；gate/up 阶段一个 CTA 顺序计算两个 intermediate N tile并共享 input load；down 阶段一个 CTA 并行计算两个 hidden N tile并共享 `up_logits` activation load；down 使用 swizzle 8。

## 2. 旧样例当前最佳（待 OJ 复验）

当前最佳由以下改动组成：

1. `threads: 256 -> 512`
2. gate/up 阶段先加载 gate 和 up 两块权重，再执行两个 GEMM
3. swizzle 按静态 shape 分派：`dhidden <= 3584` 使用 8，否则使用 16
4. 将 SiLU、逐元素乘法和 `up_logits` 写回合并为一轮 `T.Parallel`
5. down 阶段将每个 token 的路由权重先加载到 fragment，再供 128 个输出列复用

| 指标 | 原始基线 | 当前最佳 | 累计提升 |
|---|---:|---:|---:|
| 大配置 | 424.856953 ms | 371.767461 ms | 12.50% |
| 小配置 | 61.963486 ms | 54.712935 ms | 11.70% |
| 两配置等权平均提升 | - | - | 12.10% |

当前最佳功能测试：2/2 通过。

## 3. 逐轮记录

| 轮次 | 单变量/改动 | 正确性 | 大配置 ms | 小配置 ms | 相对原始表现 | 结论 |
|---:|---|---|---:|---:|---|---|
| 0 | 官方 baseline：128×128 tile、256 threads、1 stage、swizzle 10 | 2/2 | 424.857 | 61.963 | 0% / 0% | 基线 |
| 1 | `num_stages: 1 -> 2` | 失败 | - | - | - | 动态共享内存增至 128 KB，超过 C500 64 KB 上限，淘汰 |
| 2 | `block_token: 128 -> 64` | 失败 | - | - | - | `group_idx_for_bx` 由测试框架按 128 构造，接口 shape 不匹配，淘汰 |
| 3 | `threads: 256 -> 128` | 2/2 | 792.987 | 105.092 | -86.65% / -69.61% | 严重退化，淘汰 |
| 4 | `threads: 256 -> 512` | 2/2 | 373.865 | 55.425 | +12.00% / +10.55% | 首个有效优化，保留 |
| 5 | `threads: 512 -> 1024` | 2/2 | 641.531 | 88.894 | 相对当前最佳明显退化 | 线程甜点在 512，淘汰 |
| 6 | `block_dexpert: 128 -> 64`，threads 512 | 2/2 | 476.213 | 67.729 | 相对当前最佳退化 27.37% / 22.20% | 计算密度不足，淘汰 |
| 7 | gate/up 512 threads，down 256 threads | 2/2 | 374.501 | 55.535 | +11.85% / +10.37% | 接近双阶段 512，但略慢；说明 down 线程数不敏感 |
| 8 | gate/up 256 threads，down 512 threads | 2/2 | 423.655 | 61.502 | +0.28% / +0.74% | 512 线程收益主要来自 gate/up 阶段，淘汰 |
| 9 | gate/up 两个 shared 权重缓冲区复用为一个 | 编译失败 | - | - | - | PipelinePlanning 禁止流水 stage 重叠写同一 buffer，淘汰 |
| 10 | shared 复用，同时把 gate/up K 循环改为 `T.serial` | 2/2 | 381.368 | 56.303 | +10.24% / +9.14% | shared 减少有益，但失去流水调度代价更大，淘汰 |
| 11 | 恢复双 shared；循环内先加载 gate/up，再执行两个 GEMM | 2/2 | 373.734 | 55.344 | +12.03% / +10.68% | 微幅且复测可重复，保留 |
| 12 | 两阶段 `swizzle: 10 -> 8` | 2/2 | 374.215 | 55.244 | +11.92% / +10.85% | 大 shape 略差，小 shape 更好，转为按 shape 分派 |
| 13 | 大 shape swizzle 10，小 shape swizzle 8 | 2/2 | 373.742 | 55.244 | +12.03% / +10.85% | 合并两个 shape 的较优值，保留 |
| 14 | 小 shape `swizzle: 8 -> 6` | 2/2 | 373.694 | 55.285 | 大 shape 为噪声；小 shape 略退化 | 小 shape 回到 8 |
| 15 | `block_dhidden: 128 -> 64` | 2/2 | 510.006 | 71.833 | 相对当前最佳严重退化 | 128×128 为当前 tile 甜点，淘汰 |
| 16 | 合并 SiLU、up 乘法与 `up_logits` 写回循环 | 2/2 | 372.756 | 55.171 | +12.26% / +10.96% | 消除一次 fragment 遍历，保留 |
| 17 | down 阶段路由权重预取到 fragment 后复用 | 2/2 | 371.767 | 54.713 | +12.50% / +11.70% | 明显且双 shape 一致改善，保留 |
| 18 | 最佳结构上测试 `threads=384` | 编译失败 | - | - | - | 128×128 fragment layout inference 冲突；该线程轴仅支持特定布局族，淘汰 |
| 19 | 最佳结构上全局 `swizzle=4` | 2/2 | 372.606 | 55.009 | +12.30% / +11.22% | 被第 17 轮双 shape 同时支配，淘汰 |
| 20 | 最佳结构上全局 `swizzle=12` | 2/2 | 372.387 | 54.894 | +12.35% / +11.41% | 接近但仍被第 17 轮支配，淘汰 |
| 21 | 最佳结构上全局 `swizzle=16` | 2/2 | 372.887 | 54.885 | +12.23% / +11.42% | 小 shape 尚可，大 shape 退化；仍被第 17 轮支配，淘汰 |
| 22 | 最佳结构上恢复原 gate/up 顺序：load gate→GEMM→load up→GEMM | 2/2 | 372.183 | 54.891 | +12.40% / +11.41% | 与预加载顺序差异仅 0.01%～0.06%，属于噪声；顺序不是核心收益 |
| 23 | gate/up 按 shape swizzle 10/8，down 单独 swizzle 16 | 2/2 | 372.386 | 54.939 | +12.35% / +11.34% | 被第 17 轮支配，淘汰 |
| 24 | gate/up 按 shape swizzle 10/8，down 单独 swizzle 4 | 2/2 | 372.478 | 54.862 | +12.33% / +11.46% | 被第 17 轮支配，淘汰 |
| 25 | 迁移为正式 `run_kernel`；512 threads；hidden 2048 用 swizzle 8、7168 用 10；合并 SiLU 写回；路由权重 fragment 复用 | OJ Accepted 3/3 | 58 ms（OJ 总时） | 53.33 分 | 相对官方模板：分数 +6.66%，耗时 -12.12% | 建立首个正式 OJ 基线，保留 |
| 26 | 服务器-64-1：full M=128 + tail M=32，tail 128 threads；四个 kernel launch | 小规模 allclose | 点1 12.684 ms | 点2 22.858 ms | 相对同机基线退化 4.19% / 7.92% | 小 M 节省不足以覆盖额外 launch 和低 GEMM 效率，淘汰；不提交 OJ |
| 27 | full M=128 + tail M=64；tail threads 保持 128，其余同轮26 | 小规模 allclose | 点1 15.905 ms | 点2 28.268 ms | 相对同机基线退化 30.64% / 33.46% | M64 与 128 threads 布局效率很差，淘汰；继续单测 256 threads |
| 28 | 轮27仅改 tail threads：128→256 | 小规模 allclose | 点1 13.603 ms | 未测 | 点1相对同机基线退化 11.74% | 比128 threads好，但仍被轮26的M32支配，提前淘汰；不提交 OJ |
| 29 | 全量 M=64、256 threads；保持两次 kernel launch，128 父块拆成两个子块 | 小规模 allclose | 点1 10.882 ms | 点2 20.776 ms | 相对同机基线提升 10.61% / 1.91%，两点合计提升 5.09% | 首个正式 shape 双点均提升的结构候选，保留并继续广搜 |
| 30 | 全量 M=32、128 threads；保持两次 kernel launch，128 父块拆成四个子块 | 小规模 allclose | 点1 12.386 ms | 点2 23.574 ms | 相对同机基线退化 1.74% / 11.30% | 粒度过细导致 GEMM 效率下降，被轮29支配，淘汰 |
| 31 | 轮29仅改全量 M64 threads：256→512 | 小规模 allclose；OJ 3/3 Accepted | 点1 10.444 ms | 点2 19.452 ms；点3 39.125 ms | 同机提升 14.21% / 8.16% / 6.90%；OJ 53.33→55 分、58→55 ms | 提交 51719，成为当前正式最佳 |
| 32 | 轮31的三个 GEMM 显式指定 `GemmWarpPolicy.FullRow` | 小规模 allclose | 点1 10.444 ms | 点2 19.456 ms | 相对轮31约 0.00% / -0.02% | 与默认策略等价，淘汰；不提交 OJ |
| 33 | 轮31的三个 GEMM 显式指定 `GemmWarpPolicy.FullCol` | 小规模 allclose | 0.1603 ms（小规模） | 未测正式点 | 小规模相对轮31的 0.0749 ms 退化约 114% | 明显错误方向，提前淘汰；不提交 OJ |
| 34 | 全量 M64 threads：512→1024 | 小规模 allclose | 0.1026 ms（小规模） | 未测正式点 | 小规模相对轮31退化约 37% | 线程过多降低效率，提前淘汰；不提交 OJ |

说明：表中“当前最佳”指对应轮次发生时的最佳结果。微小差异需复测后才纳入保留版本。

## 4. 已确认的参数规律

- `block_token=128` 受测试数据结构约束，不能仅修改 kernel 默认值。
- 2-stage 双缓冲需要约 128 KB shared memory，超过当前 C500 单 block 上限。
- `threads=512` 是已测范围 128/256/512/1024 中的最佳点。
- `threads=384` 无法为当前 128×128 fragment 推导合法布局；线程数不是可连续调节参数。
- 线程提升的主要收益来自 gate/up 双 GEMM 阶段，down 阶段对 256/512 不敏感。
- `block_dhidden=128`、`block_dexpert=128` 均明显优于单独降至 64。
- swizzle 对 shape 敏感：7168 hidden 更适合 10，3584 hidden 更适合 8。
- 已完成 swizzle 4/6/8/10/12/16 的广度覆盖；全局 4/12/16 均不超过按 shape 分派的 10/8。
- down 阶段单独使用 swizzle 4 或 16 均无收益；两阶段共用按 shape 的 10/8 更稳。
- gate/up 的“两次 load 后再双 GEMM”和原始交错顺序性能差异处于噪声范围，不应作为主要创新点。
- 减少 epilogue 的 fragment 遍历和重复全局索引可带来稳定收益。

## 5. 后续搜索方法

后续改为“先广度、后深度”：

1. 构建自动化候选生成、同步、测试、解析和 CSV 记录脚本。
2. 广度扫描单参数及结构开关，失败配置自动记录并跳过。
3. 对每个候选至少记录功能通过率、大/小配置耗时、相对基线提升和 Pareto 状态。
4. 只对广度扫描中的 Pareto 前沿候选做组合搜索。
5. 最终候选至少复测两次，再同步为本地与远端最佳版本。

## 6. 下一阶段正式 OJ 优化方案：padding-aware full/tail split

### 6.1 差距判断

- 2026-07-13 排行榜第一名为 71.33 分，我们为 53.33 分；排行榜不公开其
  分测试点耗时和源码，不能直接归因于某个公开参数。
- 我们三个测试点分别为 54/53/53 分，差距是整体性的，不是某一个 shape 异常。
- 三个正式点的平均有效 token/专家恰好都约为 142：
  `2272/16 = 142`、`4544/32 = 142`、`9088/64 = 142`。
- 当前 kernel 以 128 行为固定 M tile。`actual_rows` 只屏蔽最终写回，gate、up、
  down 三个 GEMM 仍对 padding 行做完整计算。若专家负载接近平均值，每个专家
  通常占两个 128 行块，即约 256 行计算服务于 142 行有效数据，近似有效利用率
  只有 55.5%。
- 因此下一阶段不再优先做 swizzle 微调，而是先消除尾块 padding 计算。

### 6.2 结构方案

将 gate/up 和 down 两个阶段都拆成 full/tail 两条路径：

1. full kernel：仅处理 `remaining_rows >= 128` 的完整块，保留当前 128×128 tile。
2. tail kernel：仅处理 `0 < remaining_rows < 128` 的尾块，把一个父 128 行块拆成
   16/32/64 行子块；超出 `actual_rows` 的子块直接跳过 GEMM。
3. 继续使用 `group_idx_for_bx` 的 128 行父块映射，避免改变 OJ 元数据约定。
4. padding 行不写回，保持输出为 0；路由权重继续按 raw offset 索引。

### 6.3 广度扫描（先单变量）

在正式三个 shape 上分别记录每点耗时，不用一个配置强行覆盖全部 shape：

1. `tail_M`: 16、32、64。
2. tail threads：128、256、512（不合法布局直接记录淘汰）。
3. tail N tile：64、128。
4. GEMM warp policy：默认、FullRow、FullCol。
5. tail swizzle：4、8、10、16。
6. K tile/stage：128×1 stage，以及 64×1/2 stages。

每次只改变一个维度，记录正确性、三个测试点耗时、总分估计、编译资源错误和
Pareto 状态。

### 6.4 深度组合与提交门槛

1. 只组合各 shape 单变量扫描的前两名配置。
2. 比较“全量小 M”与“full 128 + tail 小 M”两种结构，确认额外 kernel launch
   是否值得。
3. 本地/远端固定输入至少复测两次；预计总耗时低于 52 ms 才提交 OJ。
4. OJ Accepted 后以 53.33 分、58 ms 为基线报告正式提升百分比。

首要候选：full M=128 保持现状，tail M=32、threads=128、N=128；这是第一轮
结构验证配置，不提前认定为最终最优。

## 7. 服务器扩容与 M64 后续扫描（2026-07-13）

- 新增 `服务器-64-2`，硬件同属 64 GB 测试服务器；源码由服务器-64-1 的已验证版本完整迁移，避免远端代码源版本不一致。
- 已完成 TileLang/TVM C++ 核心编译，并通过显式源码版本补齐无 `.git` 迁移包的 FFI 构建元数据；TileLang 0.1.9 导入验证通过，开始与服务器-64-1 并行扫描。
- 轮次 35：全量 M64、512 threads，仅将 `hidden=7168` 的 swizzle 从 10 改为 12。测试点 2 为 **19.219 ms**，相对轮次 31 的 19.452 ms 提升 **1.20%**；测试点 3 为 **38.973 ms**，相对 39.125 ms 提升 **0.39%**。两个大隐藏维度点均提升，升级为当前候选最佳，待点 1 回归检查与 OJ 复验。
- 轮次 36：服务器-64-2 同机比较大隐藏维度 swizzle 16 与 12；两者正确性均通过。测试点 2 分别为 **17.470 ms** 与 **19.129 ms**，swizzle 16 相对提升 **8.67%**。由于该趋势与服务器-64-1 不同，先重复 swizzle 16 并补测点 3，确认不是切片负载或首次运行异常后再保留。
- 轮次 37：swizzle 16 加长复测并跨服务器验证。服务器-64-2 点 2 重复结果 **17.467 ms**，与首次 17.470 ms 相差约 0.02%；服务器-64-1 点 2 为 **17.528 ms**，相对该机 swizzle 12 的 19.219 ms 提升 **8.80%**；服务器-64-2 点 3 为 **34.858 ms**，相对 swizzle 12 的 38.973 ms 提升 **10.56%**。三次均 allclose，故将大隐藏维度 swizzle 16 升级为新的正式候选最佳。
- 轮次 38：补测 swizzle 16 的点 1，服务器-64-1 为 **10.443 ms**，与轮次 31 的 10.444 ms 基本一致且 allclose。至此三个正式 shape 均通过正确性验证，并提交为 OJ 记录 **51776**。
- 轮次 39：深度组合实验采用 gate/up M64、down M128，目的是保留前半段 padding-aware 收益并将 down 计算块数减半。服务器-64-1 点 2 allclose，耗时 **20.399 ms**；相对同机全 M64 的 17.528 ms 退化 **16.39%**。说明 down 阶段的 padding 削减收益明显大于减少计算块数量的收益，组合淘汰；本地与服务器-64-1 均已恢复全 M64。
- 轮次 40：服务器-64-1 对全 M64 单变量测试 `block_dexpert: 128 -> 64`（N tile 64）。点 2 allclose，耗时 **21.717 ms**，相对同机当前最佳 17.528 ms 退化 **23.90%**；N tile 64 淘汰。
- 轮次 41：与轮次 40 并行，在服务器-64-2 单变量测试 `block_dhidden: 128 -> 64`（K tile 64）。点 2 allclose，耗时 **22.330 ms**，相对同机当前最佳 17.467 ms 退化 **27.84%**；K tile 64 淘汰。两台服务器随后均恢复 M64、128×128 tile、512 threads、swizzle 16 的候选最佳。
- 轮次 42：OJ 提交 **51776** Accepted，总分 **55.33**、总耗时 **53 ms**；三点分别为 **56/55/55 分**与 **9/15/29 ms**。相对提交 51719，总分提高 **0.60%**、总耗时降低 **3.64%**；点 3 从 31 ms 降至 29 ms（**6.45%**）并由 54 分升至 55 分，点 1/2 未跨越 OJ 的整数毫秒或评分档位。swizzle 16 正式升级为当前 OJ 最佳。

## 8. 独立子智能体复盘与结构性新方向（2026-07-13）

为减少既有实验路径造成的思路锚定，创建只读子智能体；仅提供正式 shape、当前最佳结构与明确淘汰项，不允许修改代码、连接服务器或提交 OJ。独立复盘得到以下关键判断：

1. 平均每专家约 142 个有效 token。当前全 M64 通常为每专家生成 4 个子 CTA，其中 3 个执行 GEMM、1 个分支退出；有效行利用率约为 `142 / 192 = 74%`。
2. 当前每个 M64 CTA 都重新加载 gate/up/down 权重。同一专家通常执行 3 个有效 M64 CTA，因此权重 tile 可能重复加载 3 次；这可能比 `up_logits` 的一次写回和一次读取更主要。
3. `up_logits` 全局往返仍是明确成本，但彻底消除并不现实：down 的每个 hidden 输出 tile 依赖全部 intermediate tile；直接融合要么重复 gate/up，要么同时保留过多 down accumulator。更可行的是紧凑/块化 workspace、提高 L2 命中或单 launch 双阶段。

新的优先级如下：

- **Top 1：父 CTA 内串行两个 M64。** 网格恢复为 128 行父任务，但不使用原来的 M128 GEMM；CTA 内保留两个独立 M64 fragment，在同一次权重加载后分别执行两个 M64 GEMM。完整父块共享权重，尾父块只执行有效半块。先分别实现 gate/up-only、down-only、both 三个单变量版本，预期重点观察寄存器 spill 与 shared/pipeline 复用。
- **Top 2：CTA 内融合相邻 N 微块。** gate/up 的一个 CTA 顺序处理两个 `by`，共享 input tile；down 顺序处理 2 或 4 个输出 hidden tile，共享 activation tile。测试矩阵为 `GU pack={1,2} × down pack={1,2,4}`。
- **Top 3：expert-major raster / persistent。** 先将 `(expert, N, row)` 任务按同一专家权重连续访问的顺序映射，再考虑 persistent CTA；先验证纯一维 raster，避免直接引入 cooperative/persistent 高风险实现。
- 后续候选：权重 L2 驻留提示、缓存紧凑任务描述符、三个正式 shape 的专用调度族、producer/consumer 一致的 tile-major workspace，以及带 grid sync 的单 launch 双阶段原型。

两服务器安排：服务器-64-1 扫父 CTA 复用的 gate/up-only、down-only、both；服务器-64-2 扫 N-pack 组合与 expert-major raster。每个候选先做 allclose，再跑三个正式点，Pareto 前沿版本长测两次后才允许提交 OJ。

## 9. 父 CTA 双 M64 权重复用实测（轮次 43～44）

- 轮次 43：只在 gate/up 阶段把两个相邻 M64 放入同一 128 行父 CTA，共享 gate/up 权重 tile；down 保持全 M64。服务器-64-1 点 2 allclose，耗时 **21.272 ms**，相对同机最佳 17.528 ms 退化 **21.36%**。双 input fragment 与四个 FP32 accumulator 带来的寄存器压力、条件控制和并发度下降明显超过权重少加载一次的收益，gate/up-only 淘汰。
- 轮次 44：只在 down 阶段使用父 CTA 双 M64，共享 down 权重 tile；gate/up 保持全 M64。首个补丁因通用文本匹配误改 gate/up 网格，产生错误结果，已定位且该次性能不计入候选比较。修正网格后服务器-64-2 点 2 allclose，耗时 **18.726 ms**，相对同机最佳 17.466 ms 退化 **7.21%**，down-only 淘汰。由于两个单阶段均退化，不再测试 both 组合。
- 回归：清理全部实验补丁后，两台服务器和本地恢复全 M64、128×128 tile、512 threads、swizzle 8/16；服务器-64-2 点 2 回归为 **17.4665 ms** 且 allclose，与实验前 17.4670 ms 一致。

结论：减少权重加载次数并不自动提高性能；在 C500 当前 TileLang 布局下，增加并行 fragment/accumulator 会显著压低 occupancy。后续 N-pack 实验必须优先控制寄存器数量，分别测试“顺序复用 shared”与“多 accumulator”，不能直接照搬父 CTA 双 M64 结构。

## 10. Down N-pack 融合（轮次 45～46）

- 轮次 45：`down_pack=2`，单 accumulator 在一个 CTA 内顺序完成两个 hidden N tile。该版本只减少 CTA/元数据与 routed weight 重复加载，不复用每轮 K 的 activation。服务器-64-1 点 2 allclose，耗时 **17.895 ms**，相对 17.528 ms 退化 **2.10%**；说明单纯减少 CTA 数不足以抵消并行度下降，淘汰。
- 轮次 46：`down_pack=2`，保留一份 `up_shared` activation fragment，同时使用两份 down weight shared buffer 和两个 FP32 output accumulator，并行计算相邻两个 hidden N tile。服务器-64-2 点 2 为 **15.813 ms**，相对同机 17.466 ms 提升 **9.46%**；服务器-64-1复现为 **15.905 ms**，相对同机 17.528 ms 提升 **9.26%**，两机差异约 0.58%，均 allclose。
- 三点补测：点 1 **9.083 ms**（原 10.443 ms，提升 **13.02%**）；点 2 **15.813 ms**（原 17.466 ms，提升 **9.46%**）；点 3 **31.547 ms**（原 34.858 ms，提升 **9.50%**）。三点本地合计约 56.44 ms，相对原候选 62.77 ms 降低约 **10.08%**。
- 一次误用了 `hidden=7168, intermediate=8192, experts=64, valid=9088` 的非正式组合，结果 125.097 ms 且 allclose；该结果不纳入正式三点比较。正式点 3 按 `hidden=7168, intermediate=2048, experts=64, valid=9088` 重测。
- 双 accumulator N-pack=2 已升级为当前候选并同步到两台服务器；首次 OJ 提交记录为 **52042**。

机制结论：该方向的核心收益是一次读取 `up_logits` activation 后服务两个 down 输出 tile。与父 CTA 双 M64 的退化相反，down N 维融合虽然增加 accumulator，但同时减少了高价值 activation 流量，因此整体收益显著。

### 10.1 OJ 52042/52062 状态说明

- 提交 52042 的点 1 Accepted：**58 分 / 8 ms**，相对提交 51776 的 56 分 / 9 ms 提升 2 分并降低 1 ms。
- 点 2/3 被平台标为 Wrong Answer，但展开 stderr 后确认错误发生在 `/sandbox/binary/testcase_config.py` 生成 `up_w` 输入时；当时评测 GPU 仅剩 **313.94 MiB** 可用显存，申请 1.75 GiB 时触发 `torch.OutOfMemoryError`。候选 kernel 尚未进入该点计时，因此这不是算法正确性失败。
- 原样重提交为记录 **52062**；最终 Accepted，得分 **58.33**、总耗时 **48 ms**。三点分别为 **59/58/58 分**与 **8/14/26 ms**。相对提交 51776，总分提高 **5.42%**、总耗时降低 **9.43%**；N-pack=2 正式升级为当前 OJ 最佳。
- 为覆盖 OJ 可能使用的非均匀专家负载，本地 `benchmark_submission.py` 增加可选 `--random-groups` 开关；默认行为不变。当时上传受外部命令额度限制，后续已在轮次 52 补齐多 seed 压力测试。

## 11. N-pack=2 后续资源与调度扫描（轮次 47～53）

- 轮次 47：恢复 XPUOJ 访问后读取提交 **52062**，最终 Accepted：**58.33 分 / 48 ms**；三点为 **59/58/58 分**与 **8/14/26 ms**。相对提交 51776，总分提高 **5.42%**、总耗时降低 **9.43%**，升级为正式最佳。
- 轮次 48：N-pack=2 上单变量测试 down threads 512→256，服务器-64-1 点 2 为 **16.506 ms**，相对该机 15.905 ms 退化 **3.78%**，淘汰。服务器-64-2同时测试 down 独立 swizzle 8，点 2 为 **15.769 ms**，相对 swizzle 16 的 15.813 ms 提升约 **0.28%**，进入复测。
- 轮次 49：加长 80 iter 复测。服务器-64-2 swizzle 8 为 **15.763 ms**；服务器-64-1 swizzle 12 为 **15.948 ms**，略慢于该机 pack=2基线。swizzle 12 淘汰。
- 轮次 50：swizzle 8 跨服务器复现，服务器-64-1 点 2 为 **15.853 ms**，相对该机 swizzle 16 的 15.905 ms 提升 **0.33%**；服务器-64-2点 3 为 **31.362 ms**，相对 swizzle 16 的 31.547 ms 提升 **0.59%**。两点均 allclose，down swizzle 8 升级为本地候选。
- 轮次 51：补齐低 swizzle 广度点。服务器-64-1 swizzle 4 为 **15.925 ms**；服务器-64-2 swizzle 6 为 **16.265 ms**，均被 swizzle 8 支配，淘汰。
- 轮次 52：随机专家负载正确性压力测试。seed=1 的总 padding 为 7808，seed=20260713 为 7552；两台服务器均 allclose，耗时分别 **15.681 ms**与 **14.850 ms**。这验证了 N-pack 对非均匀 `group_sizes`、变化的 padding 总量和尾块分布是稳健的。
- 轮次 53：测试 `down_pack=4` 的双 pair 顺序版本，在不增加 shared/accumulator 数量的前提下进一步减少 CTA；服务器-64-1点 2 为 **16.218 ms**，相对 pack=2/swizzle8 退化约 **2.30%**，淘汰。服务器-64-2同时补测 down swizzle 10，为 **15.797 ms**，略慢于 swizzle 8约 **0.21%**，淘汰。

当前结论：down 的资源甜点为 `pack=2 + 512 threads + swizzle 8`。该版本已同步到本地和两台服务器；相对 OJ 52062 仅改变 down 大 shape swizzle 16→8，本地收益约 0.3%～0.6%，尚未单独提交 OJ。

## 12. Gate/up N-pack=2 与双 N-pack 正式突破（轮次 54～57）

- 轮次 54：在 down pack=2/swizzle8候选上实现 gate/up N-pack=2。一个 CTA 顺序处理两个 intermediate N tile，每轮 K 只加载一次 input fragment；gate/up shared buffer 顺序复用以维持64 KB上限，使用四个 FP32 accumulator，K 循环改为 `T.serial`。服务器-64-1点 1 allclose，耗时 **8.054 ms**，相对 down-only候选 9.083 ms 提升 **11.33%**。
- 轮次 55：同一原型扩展到大 hidden。服务器-64-1点 2为 **14.671 ms**，相对 15.853 ms 提升 **7.46%**；点 3为 **28.339 ms**，相对 31.362 ms 提升 **9.64%**，均 allclose。说明 input复用收益在三个正式 shape 上均超过四 accumulator与失去 K pipeline的代价。
- 轮次 56：双 N-pack候选同步到服务器-64-2，点 2复现为 **14.574 ms**，与服务器-64-1趋势一致且 allclose。三点本地合计约 **51.06 ms**，相对 down-only候选约再降 **10.06%**。
- 轮次 57：OJ 提交 **52152** Accepted，总分 **60.67**、总耗时 **44 ms**；三点为 **61/60/61 分**与 **7/13/24 ms**。相对提交 52062，总分提高 **4.01%**、总耗时降低 **8.33%**。gate/up pack=2 + down pack=2/swizzle8 正式升级为当前最佳，并已同步本地与两台服务器。

机制结论：gate/up 阶段的主要可复用对象是 input tile，down 阶段的主要可复用对象是 `up_logits` activation tile。沿 N 维成对融合在两个阶段都能显著减少高价值全局内存读取；相比沿 M 维合并，N 维融合更能抵消额外 accumulator造成的资源压力。

## 13. 激进结构搜索与换轨复盘（轮次 58～69）

### 13.1 本轮不可变基线与目标

- 正式版本：OJ 提交 **52152**，`submission.py` SHA-256 为
  `980B77034ECF564E25596D11C7AA1CD0E556B9D7A805BDDF2B37BC920E4DB6D6`。
- OJ 成绩：**60.67 分 / 44 ms**，三点 **7/13/24 ms**。
- 服务器-64-1 长测基线（warmup 10、iters 100、seed 81394）：
  **8.052790 / 14.670006 / 28.350161 ms**。
- 服务器-64-2 长测基线：**7.997606 / 14.574448 / 28.115742 ms**。
  点1跨机差约0.69%，满足约1%的基线一致性要求。
- 根据历次分点反推，评分近似满足 `score ∝ 1/sqrt(time)`；80分对应三点约
  **4.1/7.3/14.0 ms**、总时约 **25～26 ms**。当前44 ms仍需约1.7倍加速，
  因此本轮不再把小于1%的参数噪声视为有效进展。

### 13.2 第一阶段：低成本高赔率探针

- **轮次58 / R58-kpack2（服务器-64-1）**：仅在 all64 双 N-pack 的六处
  `T.gemm` 显式加入 `k_pack=2`。小shape与三正式点均allclose；三点
  **8.050744 / 14.671808 / 28.330181 ms**，相对同机基线改善
  **+0.025% / -0.012% / +0.070%**，合计约0.04%，属于噪声，淘汰。
- **轮次59 / R59-l2-persist（服务器-64-2）**：仅对两阶段 `up_logits`
  添加 `T.annotate_l2_hit_ratio({up_logits: 0.8})`。三点
  **7.996180 / 14.574789 / 28.101545 ms**，改善
  **+0.018% / -0.002% / +0.050%**。TVMFFI host source未出现
  set/reset/access-policy调用，提示在默认后端疑似被静默丢弃；机制与计时均无效，淘汰。

### 13.3 第二阶段：down结构重构

- **轮次60 / R60-down-pack4-true（服务器-64-1）**：down改为真正的pack4；
  每个K tile只读一次activation，使用一份32 KB权重shared和四个FP32 accumulator，
  `T.serial` 中静态展开四次copy/GEMM。小shape与点1正确，但点1为
  **9.108915 ms**，相对8.052790 ms退化 **13.12%**。四accumulator与串行权重
  复用导致的occupancy/同步损失超过activation节省，硬停止。
- **轮次61 / R61-down-tailM32-H7168（服务器-64-2）**：GU保持M64；H7168
  的down拆为完整M64与M32/256线程尾块，总launch由2增至3。`64+11`与
  `32+13`定向尾块均allclose；p2/p3为 **14.495432 / 27.920693 ms**，合计仅
  提升 **0.642%**，低于3%结构门槛，淘汰。
- **轮次64 / R64-H7168-down-fullM32-t256-pack2（服务器-64-1）**：只把
  H7168 down全量改为M32/256线程，保留pack2。定向随机边界正确，但p2为
  **16.168766 ms**，退化 **10.22%**。这补齐了旧轮30仅测M32/128的空白，
  证明M32/256同样无法抵消CTA翻倍与MMA效率损失。

### 13.4 第三阶段：高风险数值与单launch

- **轮次62 / R62-fp16-GU-only（服务器-64-1）**：仅将GU四个accumulator
  改为FP16，down保持FP32。MACA设备编译失败：生成 `float16x4`，而当前GEMM
  intrinsic固定要求 `float4`。未生成可运行kernel；不修改intrinsic/policy，永久淘汰
  FP16 accumulator路线。
- **轮次65 / R65-M64-owner-singlelaunch（服务器-64-2）**：单个M64 CTA先
  串行完成该tile全部GU N-pack，再立即消费workspace完成全部down N-pack；复用
  两块32 KB shared、显式CTA barrier，无grid sync。小shape正确，但p2为
  **28.434136 ms**，相对14.574448 ms退化 **95.10%**。N维CTA并行损失远大于
  单launch与cache收益，淘汰。

### 13.5 外部一手源码启发：GROUP_SIZE_M raster

- 只读检索公开的 `tile-ai/TileOPs` 最新MoE grouped-GEMM源码。其一维
  `GROUP_SIZE_M` raster用于平衡activation与同专家权重的L2复用；本地只改CTA
  映射，保持总CTA、tile、threads、pack、shared和数学不变。
- **轮次66 / R66-raster-gsm4（服务器-64-1）**：group size 4；小shape与p1
  正确，但p1为 **10.385413 ms**，退化 **28.97%**，淘汰。
- **轮次67 / R67-raster-gsm2（服务器-64-2）**：group size 2；三点
  **10.659000 / 18.716465 / 37.543042 ms**，均allclose但合计退化 **32.02%**。
  说明NVIDIA TileOPs的M-major局部性不能直接迁移到C500；当前二维/N-major调度
  对专家权重连续wave复用更重要。停止整个raster家族，不扫描更多group size。

### 13.6 编译器级终止探针

- **轮次68 / R68-enable-warp-specialized（服务器-64-1）**：只将实际all64
  kernel的 `TL_DISABLE_WARP_SPECIALIZED` 从True改为False。小shape与p2正确；
  p2 **14.674148 ms**，退化0.028%，淘汰。
- **轮次69 / R69-enable-fast-math（服务器-64-2）**：只增加
  `TL_ENABLE_FAST_MATH=True`。p1误差未明显放大，耗时 **7.978040 ms**，仅提升
  **0.245%**，低于1%门槛，淘汰。

### 13.7 阶段结论

1. 本轮所有候选均保存在 `experiments/R58-*` 至 `experiments/R69-*`，每个已结束
   实验均有 `result.json`；失败、退化和平台/编译器限制均未省略。
2. 没有候选满足晋级门槛，因此**未修改正式 `submission.py`、未提交OJ**；正式最佳
   仍为提交52152、60.67分/44 ms。
3. 已明确耗尽：K-pack2、默认后端L2 hint、true down-pack4、M32 tail/full、
   FP16 accumulator、M64 owner single-launch、GROUP_SIZE_M raster、自动warp
   specialization与fast-math。
4. 若继续冲击80分，应换到新的底层能力，而不是继续组合上述失败项：优先寻找
   C500/MACA可用的异步copy/producer-consumer GEMM、允许低于64 KB峰值的真实
   多stage调度，或比赛方/高分实现所用的专用matrix-core布局。没有新机制证据前，
   不再做swizzle、threads或group-size无界扫描。

## 14. MACA 原生搬运与底层流水审计（轮次 70～72）

### 14.1 提交边界与后端事实

- 正式基线继续固定为提交 **52152**；`submission.py` SHA-256 仍为
  `980B77034ECF564E25596D11C7AA1CD0E556B9D7A805BDDF2B37BC920E4DB6D6`。
- OJ只接受可由白名单 `torch/tilelang/math` 源码现场重编译的实现；外部动态库、
  预编译MACA二进制、`ctypes`、custom callback、私有编译扩展和直接修改本地
  TileLang后端均不可提交。本轮只使用公开 `tilelang.language` API和字面量
  `pass_configs`。
- 源码审计确认：MACA的通用 PipelinePlanning 强制关闭 generic async-copy；
  `T.Pipelined` 在该后端只提供软件流水。`T.async_copy` 无法携带MACA所需barrier，
  不是合法入口。唯一能生成原生搬运的公开DSL路径是
  `T.maca_async_copy + T.alloc_maca_barrier + 显式 barrier wait`。
- 当前正式GEMM为fragment×shared。MACA lowering只在shared×shared路径消费
  `T.gemm(..., mbar=...)`；本算子必须在每个对应GEMM前显式调用
  `T.maca_barrier_arrive_and_wait`，且不同权重/双缓冲bank不能共用被覆盖的barrier。

### 14.2 R70：原生 async-copy 代码生成成功但严格正确性失败

- **轮次70 / R70-maca-async-weights（服务器-64-1）**：保持M64、N-pack2、
  K128、512线程、FP32 accumulator与数学不变；仅把all64 gate/up/down的权重
  global→shared搬运替换为 `T.maca_async_copy`，为各权重使用独立barrier并在
  GEMM前显式等待。
- TileLang safe-memory pass在RewriteCPAsync处触发内部bool类型错误；使用公开
  `TL_DISABLE_SAFE_MEMORY_ACCESS=True` 后完成lowering与MXCC。生成MACA源码含
  **8处 `memcpy_async<16>` 与8处 `barrier_arrive_and_wait`**，证明原生入口确实
  可达；该pass设置本身可由提交源码表达，但不能弥补运行正确性问题。
- H512/I512第一次与独立第二进程都能通过宽松allclose，但相对reference的
  `max_abs` 为 **0.04219～0.04408**、`mean_abs` 为 **0.0016667～0.0016712**；
  baseline同shape仅 `3.0518e-5 / 3.4508e-7`。
- 同一输入连续10次的候选输出两两最大差达到 **0.0275383**；无NaN/Inf，但存在
  明确非确定race。按严格正确性门槛直接淘汰，未做正式点性能测试；停止所有
  显式MACA async-copy、手写barrier ping-pong和producer/consumer组合路线。

### 14.3 R71：安全K64两级软件流水回退

- **轮次71 / R71-k64-stage2（服务器-64-2）**：首版同时修改GU/down时，GU在
  同一K迭代中pack0/pack1两次写同一shared buffer，被PipelinePlanner判定stage
  重叠写并拒绝。随后按单变量原则将GU完全恢复baseline，只保留down K tile
  `128→64`、`num_stages=2`，M64/N128/down-pack2/512线程不变。
- 修正版小shape编译并allclose，误差与baseline一致；生成代码不含
  `memcpy_async`，验证其仅为安全的软件流水。
- p2短测（warmup 3、iters 20）为 **15.017664 ms**，相对服务器-64-2基线
  **14.574448 ms** 回退 **3.041%**。K循环翻倍和软件多版本化成本高于潜在重叠，
  直接淘汰，未做长测。

### 14.4 R72：普通权重copy已达公开128-bit上限

- baseline H512/I512生成源码中，all64权重load静态8处全部为 `*(uint4*)`，即
  每线程 **16 B / 8×FP16 / 128-bit**；合法的 `coalesced_width=8`不会改变代码。
- **轮次72 / R72-force-weight-copy-width16（服务器-64-1）**：仅给all64六类
  权重global→shared `T.copy` 增加公开参数 `coalesced_width=16`，尝试请求
  16×FP16=32 B/256-bit；input/up_logits fragment copy及所有调度参数不变。
- MACA xcore1000 VectorizePlanner的该路径上限为128-bit；LayoutInference在
  `src/op/parallel.cc:690` 精确失败：
  `Vector size 8 is not divisible by coalesced width 16`。候选未进入MACA源码生成
  与MXCC，淘汰；不扫描已被baseline覆盖的宽度8或更窄宽度。

### 14.5 阶段结论

1. R70～R72均有隔离 `submission.py` 与 `result.json`，正式文件未修改、未提交OJ。
2. 原生 `memcpy_async` 在当前TileLang 0.1.9/MACA 3.7/C500组合上可生成，但严格
   重复测试暴露非确定race，不满足比赛正确性；不得以宽松allclose掩盖。
3. 不使用原生异步时，K64两级软件流水明显回退；普通同步copy又已自动使用公开
   planner允许的128-bit上限。继续微调stage、barrier数量或copy宽度没有机制空间。
4. `T.gemm` 已自动lower到MACA MMA；直接使用内部mma emitter不会获得新的矩阵
   指令，却会引入非公开接口、lane layout与沙箱风险。本轮不以私改intrinsic绕过
   FP16 accumulator或async正确性限制。
5. 若继续冲击80分，需要新的、可随提交源码进入OJ且经过严格确定性验证的后端能力，
   或高分实现所采用的不同算法分解；当前公开TileLang/MACA流水接口不足以支撑从
   44 ms降至约25 ms的目标。
