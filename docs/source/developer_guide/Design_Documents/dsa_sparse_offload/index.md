# DSA 稀疏卸载：架构与工程指南

> 基线：vLLM / vLLM-Ascend v0.23.0。
> 本文档集按总设计、特性路线、已确认设计、未决草案和工程经验组织。

## 阅读导航

| 分类 | 文档 | 回答的问题 |
|---|---|---|
| 总设计 | [系统架构](architecture.md) | 系统边界、数据与控制流、模块所有权、统一元数据、公共算子合同 |
| 特性路线图 | [特性演进与实施路线图](feature_roadmap.md) | 哪些特性已适配、哪些待实施或待定稿、下一步有哪些依赖 |
| 已确认设计 | [多级 KV 缓存](features/multilevel_kv_cache.md) | L2目录、L1/L2/L3匹配、回填、存取与一致性 |
| 已确认设计 | [MTP 与推测解码](features/speculative_decoding.md) | 固定MTP3、双尾、七输出、proposer和多级prefix |
| 已确认设计 | [数据并行推理](features/data_parallel_inference.md) | DP/TP/EP基础、单机DP资源预算、dummy与管理事务 |
| 未决设计草案 | [自适应推测解码](proposals/adaptive_speculation.md) | 理想版MTP的容量适应、有效query数、采样与图合同 |
| 详细设计草案 | [P/D分离](proposals/prefill_decode_disaggregation.md) | 已确认首版范围、v0.23/新版依据、接管/尾页/MTP/DP合同及验证门禁；待评审、未实现 |
| 未决设计草案 | [分布式执行与状态恢复](proposals/execution_and_recovery.md) | 打断恢复、跨机DP及异步加载的候选方案与待决策项；PD转到独立草案 |
| 工程经验 | [问题分析与工程经验](engineering/diagnostics_and_lessons.md) | 排查方法、已修复机制、未定界问题、证据和测试边界 |

推荐顺序：**系统架构 → 对应专题 → 特性路线图**。遇到故障再查工程经验；规划新能力时才进入草案。

首次了解图模式，可以先读[图模式背景与调用链](architecture.md#136-图模式入门执行程序与记录设备任务)：
从Dynamo/FX和设备图的区别，到compile/warmup/capture/replay、bucket/PAD、固定地址、
npugraph_ex及attention参数更新，再进入DSA的统一metadata设计。性能问题另见
[Host准备与同步等待归因](engineering/diagnostics_and_lessons.md#12-性能归因与元数据准备)。

总设计保留插件安装、spec/容量、plan/commit、SoA、first-fill、dump、图与模块边界的
详细说明。算子按家族独立核对：总设计11节包含A5五输出LIDU/七输出LIM和packed KSC，
12.6节列三种满块复制，18节列原生Quant-LI/QSFA调用，19至23节详述A3单query三算子。
多级缓存专题保留job/fence、地址、后端、失败/淘汰、LMCache对照和回填例子；
DP专题先介绍基线逐token教学，再解释DSA已确认的增量。

## 设计状态与实施状态

这两个维度必须分开记录：

- **设计已确认 / 已适配**：当前三个专题的已实施主路径，受各自支持矩阵约束。
- **设计已确认 / 待适配**：只有接口、所有权、时序和失败语义都已确认，才能标为此状态。后续出现这类任务时，放入相应专题并在路线图列明实施缺口。
- **目标已确认 / 方案待定或待评审**：例如更完整的分层复用、动态MTP、PD与恢复。PD已有独立详细设计草案，但仍需评审与介质门禁；目标共识不等于可直接实现的方案，详细讨论留在`proposals/`。
- **已适配 / 验证有边界**：功能可运行不等于任意模型、拓扑、数值质量、性能与长稳都已认证。

各特性的当前状态以路线图和对应专题为准。草案不解除现有gate，也不构成自动开工指令。

## 文档分工

1. `architecture.md`维护全局不变量和模块边界，不记录逐轮debug进度。
2. `features/`维护已确认合同；专题更新时同步总设计中的跨模块关系。
3. `feature_roadmap.md`只记录特性适配、实施依赖与支持状态，不堆砌bugfix、run ID或UT数量。
4. `proposals/`明确可选方案、代价与未决点；方案确认后将冻结合同移入专题，避免同一方案多处维护。
5. `engineering/`统一维护分析方法、修复案例和残留风险；不将相关性升级成根因，不将CPU模拟升级成NPU验证。

历史ABI与撤回原型仅在解释差异、反例或方案取舍时保留，并明确历史/候选状态。
临时探针启动命令与重复阶段流水不另建归档。
原始模型数据、tensor、私有日志和构建产物不属于设计文档。

## 测试入口

部署与执行命令就近维护于仓库 `examples/dsa_demo/README.md`，不在设计页复制服务器参数。
同事入门使用 `simple_dp_prompt_test.py`，直接修改脚本顶部配置，按三轮基础流程运行；自动摘要是证据整理，不是全量精度认证。

:::{toctree}
:maxdepth: 1
:caption: 架构与特性
architecture
feature_roadmap
features/multilevel_kv_cache
features/speculative_decoding
features/data_parallel_inference
:::

:::{toctree}
:maxdepth: 1
:caption: 设计草案与工程经验
proposals/adaptive_speculation
proposals/prefill_decode_disaggregation
proposals/execution_and_recovery
engineering/diagnostics_and_lessons
:::
