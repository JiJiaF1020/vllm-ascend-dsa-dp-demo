# 特性演进与实施路线图

> 更新：2026-09-11；当前集成快照`b9d12863c2`。
> 本页记录特性适配，不记录逐轮debug、bugfix清单或实验日志。工程质量风险集中在[问题分析与工程经验](engineering/diagnostics_and_lessons.md)。

## 1. 状态定义

| 标记 | 含义 |
|---|---|
| 已适配 | 代码与约定主路径已集成，验证范围需单独限定 |
| 设计确认 / 待实施 | 接口、所有权、时序与失败语义均已确认，尚有明确实施任务 |
| 目标确认 / 待定稿 | 产品/架构方向已认可，详细方案未完成共识 |
| 研究候选 | 尚需可行性、收益或基线对齐分析 |

已适配不等于任意模型、拓扑、执行模式的全组合质量认证。功能阶段号也不能复用为修复序号。

## 2. 基础架构演进

### 2.1 基线迁移沿革

项目从v0.13原型迁移到v0.19，再迁移到当前v0.23。版本迁移不等于重做全部功能，
也不能把后续新特性的验证范围反向计入旧版本。

| 阶段 | 保留的主线 | 当前如何理解 |
|---|---|---|
| v0.13原型 | DSA稀疏卸载的起点 | 此处记录项目沿革，不重新认证旧版本功能或性能 |
| v0.19迁移与收敛 | Indexer/MLA解耦、LIDU/KSC/SFA链、row-mode FULL decode、独立满块dump、统一metadata；后期将修改集中到Ascend插件 | 旧版已探索插件化，不应把“上游vLLM零修改”全部说成v0.23首次实现；旧patch拓扑和接口不直接复用 |
| v0.23迁移与扩展 | 延续卸载和固定owner合同，适配新版spec/group、scheduler扩展、输入投影与图准入；随后叠加A5、MTP、多级prefix和单机DP | 当前支持状态与实现以本文以下表格和正式专题为准，不照搬旧版启动配置 |

### 2.2 当前基础能力

| 特性 | 当前进展 | 关键交付 |
|---|---|---|
| 完整Indexer与resident MLA解耦 | 已适配 | 独立spec、group、BlockPool、容量与slot mapping |
| 基础稀疏decode | 已适配 | LI/selection、resident更新、KSC miss换入、SFA与固定尾区 |
| v0.23基线迁移 | 已适配 | 平台扩展点、薄scheduler、原生Request生命周期，不复制主调度循环 |
| 请求布局事务与冻结budget | 已适配 | PREFILL/DENSE/ENTER/SPARSE、plan/commit与多池admission |
| Chunked prefill | 已适配 | 完整prompt准入、逐chunk增量分配和verified满块dump |
| Eager/FULL decode Graph统一 | 已适配 | 同一InputBatch/runtime、固定地址scratch、PAD/no-op合同 |
| A5 packed-C8 | 已适配 | 独立Indexer key/scale、656-byte MLA、融合LIDU与packed KSC/QSFA |
| GLM-5.2 IndexShare | 已适配 | 21个full selection group与57个shared follower，本层payload独立 |

这些能力不是后来L3或DP的附属补丁，而是其必要基础。当前公共合同见[系统架构](architecture.md)。

## 3. 缓存与推测解码演进

| 特性 | 当前进展 | 关键交付 |
|---|---|---|
| Scheduler-owned L2目录 | 已适配 | 统一physical ID、hash/refcount/LRU、worker只镜像地址 |
| 完整L2 bundle | 已适配 | target Indexer+MLA，MTP模式增加BF16 full-cache component |
| Local-L2 prefix cache | 已适配 | 完整块最长连续命中、跨请求保留、恢复与去重 |
| Mooncake L2→L3 write-behind | 已适配 | TP0生产、源pin/reuse fence、完成回报和有界并发 |
| L3 prefix loadback | 已适配 | 连续suffix、各TP L3→L2→HBM、本域成功汇总与错误传播 |
| HBM参与prefix匹配 | 已适配 | 完整pair/triple与L2独立检索、最长本地优先、首prefill补L2 gap |
| 固定MTP3 target双尾 | 已适配 | 多query验算、冻结预算档位、跨128边界的parity尾区 |
| Proposer原生BF16 full cache | 已适配 | active decode全HBM；不依赖缺失的BF16 sparse swap算子 |
| MTP多级prefix | 已适配 | verified三组件bundle、联合回退边界、proposer后完成发布 |
| LIM七输出与2304选择策略 | 已适配 | 共同durable TopK+因果双尾、query与request分轴、后续融合接口 |
| 算子跨特性集成 | 已适配 | LIDU/LIM/KSC更新与prefix、MTP、DP共享同一元数据合同 |

当前L3后端是Mooncake Store-owned TCP。更细的component源选择和direct-HBM不由以上状态自动推出。详见[多级缓存](features/multilevel_kv_cache.md)、[MTP设计](features/speculative_decoding.md)。

## 4. 单机数据并行演进

以下P0-P6是**DP方案自己的阶段号**，与早期缓存计划的P0/P1/P2不同。

| DP阶段 | 当前进展 | 功能覆盖 |
|---|---|---|
| P0：支持边界与基线接入 | 已适配 | 每域EngineCore/Scheduler，配置守卫，非DSA不受干扰 |
| P1：节点L2预算 | 已适配 | 分配前共同观测、等额字节配额、本域L1比例格式化 |
| P2：执行/Graph/dummy | 已适配 | 原生模式协商、不分裂metadata、空域不触发真实IO |
| P3：Local prefix | 已适配 | 两域L1/L2隔离，冷热、非对称与重入主路径 |
| P4：MTP Local | 已适配 | 固定MTP3与Local prefix、双尾、proposer时序 |
| P5：共享L3 | 已适配 | 各域TP0写者、跨域回读、连接身份与内容key分离 |
| P6-A：MTP三组件L3 | 已适配 | 单长请求及长短请求交换域、local reset后回读 |
| P6-B：管理生命周期 | 已适配 | 全域pause/drain/reset/resume、统一epoch、预暂停状态保持 |
| 离线SPMD与简明入口 | 已适配 | 每域独立客户端，三轮首次/本地/跨域L3流程 |

约定收口范围为A5单机DP2/TP4/EP8、GLM-5.2 W4A4C8、mp、同步调度、固定MTP3、Mooncake TCP。它不是跨机能力或故障恢复认证。具体设计见[数据并行推理](features/data_parallel_inference.md)。

## 5. 待推进特性

| 特性 | 设计状态 | 实施状态与依赖 |
|---|---|---|
| 请求preemption/resume | 目标确认 / 待定稿 | 未开放；需先决定L2寿命、恢复水位、尾块和在途IO合同 |
| P/D分离 | 目标确认 / 待定稿 | 未开放；依赖完整状态接管、失败协议及基线connector选型 |
| 跨机DP | 目标确认 / 待定稿 | 未开放；需节点级预算域、身份/可达性和全域管理 |
| 理想版自适应MTP | 目标确认 / 待定稿 | 未实施；有效query前缀必须贯通算子、采样、scheduler与Graph |
| MTP proposer Graph | 研究候选 | 当前保持eager；需覆盖独立cache与多步draft控制流 |
| Layerwise async load/wait | 目标确认 / 待定稿 | 当前同步恢复；需逐层readiness、资源寿命及失败协商 |
| L3↔HBM直通 | 目标确认 / 待定稿 | 当前目标环境不满足统一数据面前提；先验证transport兼容组合 |
| 任意component独立源计划 | 目标确认 / 待定稿 | 完整pair/triple已实现，独立component mask仍未开放 |
| Offload-aware partial-prefix prefill | 研究候选 | 现有部分命中先恢复dense历史；直接以L2历史做多query或跳过dense恢复需新的attention与阶段合同 |
| L3写入策略与可观测性扩展 | 研究候选 | 现有TP0 write-behind已适配；轮转writer、热度admission、retention/TTL及KV events/metrics另行设计 |
| 更多后端、拓扑和模型 | 研究候选 | MemCache正式后端、CP/PP、更多硬件/模型需独立准入 |

“目标确认”表示方向成立，不表示旧草案已可执行。当前没有另行冻结、尚未实施的完整新特性设计；后续确认后再明确标为“设计确认 / 待实施”，移入`features/`。

## 6. 实施依赖与决策顺序

```text
已完成：基础卸载/统一metadata
        -> L2目录与完整bundle -> L3与HBM prefix
        -> 固定MTP3/多级prefix -> 单机DP与组合适配

后续先决策：恢复的数据所有权与失败语义
            + 跨实例/跨节点通信与部署边界
            -> 对应preemption、PD或跨机DP实现

性能扩展：layerwise/direct/proposer Graph/动态MTP分别立项
```

后续优先级由新的需求和资源条件确认，不把旧聊天计划当作自动开工授权。每项进入实施前必须写清状态所有者、数据路径、失败边界、最小验收矩阵；不得先删除启动拒绝来制造“支持”。

### 6.1 目标场景决定优先级

| 目标场景 | 已有基础与后续依赖 |
|---|---|
| TTFT、重复长prompt、多轮对话 | L1/L2/L3及dense恢复式partial-prefix已适配；是否继续做layerwise或offload-aware prefill，由真实瓶颈与容量需求决定 |
| 单请求decode性能、MTP效率 | 固定MTP3及proposer BF16 full-HBM已适配；proposer Graph与有效query前缀自适应各自立项，不阻塞缓存主线 |
| 跨域/跨实例池化 | 约定单机DP共享L3已适配；跨机仍需节点级预算、可达性、共享身份和管理协议，不由backend有共享池自动推出 |
| P/D分离 | 先解决单实例preemption/resume的所有权与恢复合同，再决定L3中转或P2P；P/D物理group和TP可不同，需语义manifest及D侧完整backing/first-fill |

### 6.2 进入详细设计的决策门禁

- 冻结配对vLLM/vLLM-Ascend版本和connector ABI；升级时先做接口差异核对，不按名称相同假设兼容。
- 缓存扩展明确节点/EngineCore预算域、必选payload（含scale和MTP）、TP副本字节一致性，以及full/partial-prefix的实际恢复路径。
- P/D明确语义component导出、接收端L2/HBM落点、原子ready、first-fill及失败接管；直接P2P是否能写D侧L2必须单独确认。
- 理想MTP先决定greedy、MTP3、固定Q4的最小范围；max15、随机采样及processor/logprobs泛化不是自动包含的能力。
- 每项依次给出数据合同、组件测试、集成测试与端到端门禁，不同时改变所有权、hash、异步完成和first-fill却缺少隔离验证。

早期人周、月数和里程碑只记录当时范围下的工作量判断，不是当前剩余工作量或工期承诺。
详细性能分析、候选source plan与实施方案归入[执行与恢复草案](proposals/execution_and_recovery.md)或
[自适应推测解码草案](proposals/adaptive_speculation.md)，不在本页展开。

## 7. 维护边界

本页只在特性支持、设计状态或实施依赖改变时更新。具体bug修复、日志run ID、UT数量、数值边界与环境问题写入[工程经验](engineering/diagnostics_and_lessons.md)。部署命令保持在`examples/dsa_demo/README.md`，不复制个人路径。
