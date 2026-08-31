# DSA 稀疏卸载当前设计

> - 最后更新：2026-08-31
> - 目标基线：vLLM v0.23.0 + vLLM-Ascend v0.23.0 正式版
>   `5cb98caa`（CANN 9.1.0、TorchNPU 2.10.0.post4、
>   Triton Ascend 3.2.2）
> - 当前完成度：核心控制面、eager 和 FULL decode graph 已完成 910C 初验；
>   chunked prefill 已完成分段 prefill/首 token 初验，完整 sparse decode
>   验收待补；A5 packed C8 已完成 GLM-5.1 eager/graph 端到端初验；
>   GLM-5.2 MTP 已完成 UT、离线 eager/FULL graph 与在线 FULL graph 冒烟
> - 本地 DRAM complete-bundle prefix cache 已完成第一版上板；Mooncake
>   P1-W rank0 异步 write-behind 与 P1-R 同步 all-rank prefix loadback
>   已完成代码、UT 和 TP8 收缩验收，在线长稳与统一 P2 回归仍待补齐
> - 首要验收模型：GLM-5.1、GLM-5.2；兼容回归模型：DeepSeek-V3.2

## 1. 文档定位

本文描述 **v0.23 分支当前已经落地的设计**，是公开项目中关于 DSA
“当前代码是什么结构、谁持有什么真源、稳定不变量是什么”的唯一设计真源。
后续每完成一个扩展场景，都应先按实际代码更新本文，再更新对应 demo 和
验收矩阵。尚未实现的能力只列接口边界，不写成可运行能力。

## 2. 目标与当前边界

DSA 稀疏卸载最终目标是在长上下文 decode 中实现：

1. Indexer K 保存完整上下文并驻留 HBM；
2. target Indexer+MLA 完整满块卸载到 worker 本地 hot DRAM；
3. HBM MLA 只保留有界 resident budget 与固定尾区；普通 decode 为一块，
   固定 MTP3 为两块 parity tail；
4. LIDU 选择重要 token 并更新 resident 映射；
5. KSC 只换入本轮 miss token；
6. SFA-Offload 消费重要 token 与尾块完成注意力计算。

当前代码已经完成：

- `additional_config["dsa_sparse_config"]` 的类型化解析和支持矩阵校验；
- 模型能力判断；
- Indexer/MLA HBM spec、容量、tensor、block pool 和绑定解耦；
- scheduler/core 侧请求 cache 布局规划；
- PREFILL、DENSE、ENTER、SPARSE 的 component-wise block 分配；
- prefill 输出返回后的 resident 满块释放时序；
- 不复制上游 `Scheduler.schedule()` 的薄调度适配；
- scheduler/core 已提交状态到 worker 最终 `InputBatch` 行序的列式投影；
- ENTER 的 resident MLA block table 全量替换；
- 稳定 resident token pool 与逐 selection group 的 `cache_slots`；
- 固定容量 hot DRAM arena 和请求逻辑块 ledger；
- prefill/dense/sparse 共用的双 plane slot mapping；
- 基于 v0.23 原生调度状态的 chunked prefill、增量双 plane 分配与逐 chunk
  满块 dump；
- eager LIDU→KSC→SFA-Offload 与 full-block dump 数据面；
- 复用原生 FULL decode capture/replay 的 row-mode 图数据面；
- A5 C8 的独立 Indexer key/scale、packed resident/DRAM、decode 融合
  Quant-LI/resident manager、纯 IO packed KSC、社区 QSFA 与 packed dump
  源码集成；prefill 仍复用社区 Quant-LI。
- GLM-5.2 IndexShare：78 个 resident MLA 层中仅 21 个 full 层创建
  Indexer cache/LIDU 持久状态，57 个 shared 层复用所属 full 层的 topK、
  copy plan 与 resident slot 映射，同时保留本层独立 MLA HBM/DRAM payload。
- A5 packed-C8 固定 MTP3：target 使用多 query LIM、请求级去重 KSC 和
  原生 QSFA，并使用固定双尾 parity 布局；MTP proposer 保持基线 BF16
  Indexer+MLA forward，其 Cache 作为第三个原生 full-cache group 全量驻留
  HBM，不创建 DSA selection state 或 DRAM arena。框架侧已切换到该结构并
  完成服务器阶段冒烟。
- scheduler-owned DRAM complete bundle 的 hash-ready、LRU/refcount、
  请求结束保留、最长连续命中、target 分平面 restore，以及可选 MTP
  full-HBM EAGLE 公共命中第一版；
- `DSAL3Connector` 的 Mooncake Store-owned TCP：TP0 异步上传完整 L2
  Indexer/MLA component、physical-id reuse fence、完成回报与 shutdown drain；
- L2 命中后的 L3 连续 suffix lookup、联合 HBM/L2 admission、all-rank
  `L3 -> L2` 同步恢复、TP success consensus 和既有 `L2 -> HBM` restore；
- no-MTP 下成对 HBM Indexer+dense-MLA prefix 独立参与本地最长命中，允许
  越过 L2 complete bundle，并在首个 prefill forward 中补齐 L2 gap。

正式版基线同时提供并由 DSA 原样继承：

- MTP draft slot mapping 的固定容量补偿，覆盖 BlockTable 与 proposer buffer；
- spawn 后 `enable_npugraph_ex`、static-kernel 等最终图配置回写；
- AscendStore layerwise MTP 层注册与 PCP/DCP/PP/rank 隔离 GVA key；
- MemCache lazy-init 查询保护、在线 `shm_broadcast` 槽位释放修复；
- async accepted-token/SWA 生命周期修复。DSA 当前仍显式拒绝 async
  scheduling，但不能绕开或覆盖这些基线补丁。

当前尚未完成：

- Indexer-only/MLA-only 的 component-wise source mask 与精细 IO 去重；
- L3 在线 continuous batching/FD/连接长稳、统一 load/store drain、
  layerwise async load、HBM 直通、MTP/DP/非 Mooncake backend；
- prefill/decode mixed、preemption/resume、P/D 分离、非 MTP speculative
  method、MTP draft 数不等于 3/非 greedy MTP、MTP proposer graph、
  async scheduling、KV-cache metrics/events；
- GLM-5.2 A5 正式 QA/性能验收、A5 BF16 算子链和 DeepSeek-V3.2 回归。

当前已在 Linux + Ascend 环境完成 DSA UT，并取得 GLM-5.1 W4A8/W4A4C8、
TP16/EP 的 disabled、cache-init、eager 与 FULL decode graph 初验证据。
graph 已覆盖 bsz=4 的同长度约 8K prompt，以及约
5K/20K/8K/40K 的混合长度与不同 resident budget。该结果证明
capture/replay 主路径可运行，但仍不替代完整 QA 数据集、长时间连续调度、
请求结束/行复用和 DeepSeek-V3.2 回归。chunked prefill 已在
`max_num_batched_tokens=4096/8192/16384` 下完成四条长短混合请求的
baseline/DSA 首 token 对照；由于该组测试只生成一个 token，它尚不能证明
后续 sparse decode 的完整正确性。

### 2.1 妥协版 MTP 后续 RoadMap 位置

当前实现已将 L3 数据路径阶段性收口并进入 HBM 热层 P1-HOT：scheduler-owned
DRAM、完整 Indexer+MLA 本地 prefix 和 L2/L3 第一版读写闭环已经打通；
graph token 确定性与 L3 长稳作为遗留继续跟踪，不能直接把项目状态描述成
已经进入 P/D 分离。

| RoadMap 工作项 | 当前状态 |
|---|---|
| scheduler 统一管理 DRAM physical block | 已完成 |
| 完整 Indexer+MLA bundle、block hash 与 LocalDRAM prefix | 第一版已完成并上板 |
| L2→L3 异步 write-behind | P1-W 已完成并初验 |
| L3→L2→HBM prefix loadback | P1-R 同步版已完成并初验 |
| HBM target prefix 与 L2/L3 component-wise source | 妥协版 pair 已开发；HBM 可越过 L2，L3 固定经 L2，待上板 |
| MTP 多级 prefix | 未实现；L3 当前显式拒绝 MTP |
| preemption/resume | 未实现，scheduler 显式失败 |
| P/D 分离 | 未开始 |
| 理想版动态 MTP | 未开始；当前固定 MTP3 可用 |
| MTP proposer graph | 未实现；target uniform Q=4 FULL graph 已完成 |

默认推进顺序为：先上板收口成对 HBM target prefix 与 L2 backfill，随后依次
处理 MTP 多级 prefix、preemption/resume 和 P/D 分离。独立 component source
mask、L3 layerwise/direct 与理想版 MTP 属于性能或高风险扩展，不应阻塞上述
主线。详细阶段门禁见
{doc}`dsa_offload_roadmap`。

## 3. 当前总体架构

```mermaid
flowchart TB
    CFG["additional_config.dsa_sparse_config"] --> AC["AscendConfig.dsa_offload_config"]
    AC --> PLATFORM["AscendPlatform 配置收敛与支持矩阵"]
    PLATFORM --> SPEC["DSA KV specs 与 group 规划"]
    SPEC --> CACHECFG["ratio 感知的 KVCacheConfig"]
    CACHECFG --> WORKER["NPUModelRunner 多 group tensor 分配、reshape、绑定"]
    CACHECFG --> COORD["DSAKVCacheCoordinator"]
    COORD --> IDX["Indexer 独立 BlockPool / manager"]
    COORD --> MLA["resident MLA 独立 BlockPool / manager"]
    COORD --> MTP["MTP BF16 full-cache BlockPool / baseline manager"]
    REQ["vLLM Request 只读视图"] --> PLAN["DSARequestCachePlanner"]
    PLAN --> ALLOC["allocate_dsa_slots"]
    ALLOC --> IDX
    ALLOC --> MLA
    ALLOC --> MTP
    SCHED["DSAOffloadScheduler 薄适配"] --> COORD
    COORD --> PROJECTION["DSA cache-layout 列式投影"]
    PROJECTION --> INPUT["NPUInputBatch 固定容量行状态"]
    INPUT --> POOL["稳定 resident token pool / cache_slots"]
    INPUT --> RUNTIME["DSAOffloadRuntime 固定 metadata owner"]
    COORD --> DIRECTORY["scheduler DRAM block directory"]
    DIRECTORY --> PROJECTION
    RUNTIME --> DRAM["固定完整 Indexer+MLA arena / worker mirror"]
    POOL --> DATA["LIDU -> KSC -> SFA-Offload"]
    DRAM --> DATA
    DATA --> DUMP["SFA 后 full-block dump"]
    DUMP --> DRAM
    INPUT --> GATE["DSA row-mode graph gate"]
    GATE --> GRAPH["原生 FULL graph dispatcher/capture/replay"]
    RUNTIME --> GRAPH
```

当前有三个不同层次的真源：

| 层次 | 真源 | 当前所有者 |
|---|---|---|
| 用户配置 | 解析后的不可变 DSA 配置 | `AscendConfig.dsa_offload_config` |
| 请求 cache 布局 | 阶段、冻结预算、resident 有效长度 | `DSARequestCachePlanner` |
| 物理块表 | target 两 plane 与可选 MTP full group 的 request→blocks 映射 | `DSAKVCacheCoordinator` 下的各 group manager |
| worker resident 映射 | request→稳定 pool row、逐层 token→slot | `DSAResidentTokenPool` |
| DRAM 逻辑真源 | request→logical full block→DRAM block | `DSAKVCacheCoordinator` |
| DRAM worker 镜像 | pool row→logical full block→DRAM block | `DSAWorkerDRAMStore` |

worker 行状态是上述 scheduler/core 真源的 **投影**，不是第二套请求阶段
账本。eager 与 graph 后续都必须消费这个投影和同一个 buffer owner。

## 4. 与 v0.23 基线的集成方式

### 4.1 Python 插件发现与安装时序

DSA 实现只修改 vLLM-Ascend，不修改配套 vLLM checkout。这里的“零修改”
不等于 vLLM 不参与：vLLM 提供硬件插件发现与若干扩展点，Python 提供
distribution metadata 和运行时导入机制。

vLLM-Ascend 安装时在 `setup.py` 注册：

```text
vllm.platform_plugins:
  ascend = vllm_ascend:register

vllm.general_plugins:
  ascend_* = vllm_ascend:register_*
```

`from vllm import LLM` 首次导入 vLLM 后，vLLM 通过
`importlib.metadata.entry_points()` 查询当前 Python 环境中已安装
distribution 的这些 entry point。平台插件的 `register()` 返回
`vllm_ascend.platform.NPUPlatform`，后续配置收敛、模型 runner 和
attention backend 才进入 Ascend 实现。

engine-core 子进程还会加载 general plugins。它们通过
`vllm_ascend._ensure_global_patch()` 调用原生 `adapt_patch()`；
`_GLOBAL_PATCH_APPLIED` 保证同一进程重复触发时幂等。patch 必须早于被替换
函数首次用于构造 scheduler/cache manager；仅早于一次普通函数调用还不够，
因为其他模块可能已经用 `from ... import ...` 缓存了旧引用。

```mermaid
sequenceDiagram
    participant U as "用户脚本"
    participant V as "vLLM import/plugin loader"
    participant A as "vLLM-Ascend"
    participant E as "EngineCore/Worker"

    U->>V: from vllm import LLM
    V->>A: 加载 platform entry point
    A-->>V: NPUPlatform
    U->>V: LLM(...)
    V->>A: NPUPlatform.check_and_update_config
    A->>A: 解析 DSAOffloadConfig/安装 scheduler_cls
    V->>E: 创建 engine-core 与 worker
    E->>A: 加载 general plugins
    A->>A: 幂等安装 vLLM-Ascend 全局 patch
    E->>A: 构造 DSA coordinator/model runner
```

### 4.2 扩展点与窄 patch 的边界

| v0.23 扩展点 | DSA 用法 |
|---|---|
| `AscendConfig` | 解析并持有 `DSAOffloadConfig` |
| `KVCacheSpecRegistry` | 注册 DSA Indexer/resident spec 与 manager |
| KV group/config hook | 构造 target 两组、可选 MTP 第三组和 ratio 容量 |
| coordinator factory | 创建 `DSAKVCacheCoordinator` |
| `scheduler_config.scheduler_cls` | 安装薄 `DSAOffloadScheduler` |
| `SchedulerOutput` dataclass | 浅包装为只增加一个 projection 的 Ascend 子类 |
| `NPUInputBatch` | 持有固定容量 DSA SoA 行状态 |
| `NPUModelRunner` cache 初始化 | 分配、reshape、绑定 target split plane 与原生 MTP full cache |
| common/SFA attention metadata | 在同一份 resident metadata 上附带 Indexer/DSA view |
| `AscendSFAImpl` | DSA 开启时执行 LIDU/KSC/SFA-Offload 和层后 dump |

能够直接修改 vLLM-Ascend 的位置都采用显式类型和普通调用，不再为 DSA
单独制造 patch 层。vLLM v0.23 尚未给出正式扩展点的少数位置，DSA 复用
vLLM-Ascend 已有 patch 模块：

| 现有 patch 模块 | DSA 窄适配 |
|---|---|
| `patch_kv_cache_utils.py` | 识别 DSA split spec/group，接管分组、ratio 容量和容量报告 |
| `patch_kv_cache_coordinator.py` | 识别 DSA group 创建 coordinator，并仅对 `DSAKVCacheCoordinator` 调用 `allocate_dsa_slots()` |

每个 wrapper 都先用类型化 spec/group/coordinator 判断是否为 DSA；不满足
时调用保存的上游原函数。不能再为同一目标函数叠加第二个 DSA wrapper，
因为 Python 属性替换遵循最后一次赋值，多个互不知情的 wrapper 会产生安装
顺序依赖。新逻辑应合并到现有 wrapper，并继续依赖 vLLM-Ascend 全局 patch
的进程内幂等安装。

DSA 没有 patch `Request`，没有复制或 patch
`Scheduler.schedule()`，也没有全局替换 `SchedulerOutput`。它通过
`scheduler_config.scheduler_cls` 安装薄 `DSAOffloadScheduler`，再用
vLLM-Ascend 自有的 `DSAOffloadSchedulerOutput` 子类附加一个类型化
projection。v0.19 的 patch 拓扑只作为历史参考，不是 v0.23 的实现范式。

## 5. 配置与模型能力

配置入口保持为：

```python
additional_config={
    "dsa_sparse_config": {
        "enabled": True,
        "split_indexer_cache": True,
        "indexer_mla_block_ratio": 3,
        "sparse_activation_tokens": 6144,
        "prompt_budget_thresholds": [32768, 65536],
        "resident_budget_tokens": [6144, 10240, 12288],
        "max_active_reqs": 256,
        "hot_cpu_block_multiple": 3.0,
        "enable_row_mode_decode_graph": True,
    }
}
```

`config.py` 在拉起期完成类型转换、未知字段拒绝和支持矩阵校验。当前支持
v0.23 原生显式 chunked prefill，以及
`long_prefill_token_threshold` 形成的固定上限 prefill chunk；两者都要求
`scheduler_reserve_full_isl=True`，在首个 chunk 入场前验证完整 prompt 能
同时容纳于所有启用的 full-context cache group。speculative decode 仅放行 A5 packed-C8
下的 `method=mtp`、固定 3 个 draft 和 greedy 请求；MTP resident 最低档为
8192。本地 DRAM complete-bundle prefix cache 已开放配置；当前仍显式拒绝
async scheduling、其他 speculative method、KV transfer、context/pipeline parallel、
KV-cache metrics/events 等未建立完整合同的组合。

| 字段 | 默认值 | 当前语义 |
|---|---:|---|
| `enabled` | `False` | DSA 总开关；关闭时不创建 DSA cache、scheduler 或 runtime |
| `split_indexer_cache` | `True` | 兼容性输入；DSA 开启时必须为真，解析后由 `enabled` 派生 |
| `indexer_mla_block_ratio` | `3` | Indexer HBM blocks 与 resident MLA base blocks 的容量比 |
| `sparse_activation_tokens` | `6144` | 完整上下文超过该值后允许进入 sparse 布局 |
| `prompt_budget_thresholds` | `[32768, 65536]` | admission 时按 prompt 长度选择 resident 档位 |
| `resident_budget_tokens` | `[6144, 10240, 12288]` | N 个 prompt 阈值划分 N+1 个区间，每个区间对应一个冻结 resident budget |
| `max_active_reqs` | `256` | 配置上界，必须覆盖 `max_num_seqs`；不是实际预分配行数 |
| `hot_cpu_block_multiple` | `3.0` | 期望 DRAM usable blocks 相对 Indexer HBM blocks 的浮点倍数，最终向上取整；单机内存不足可缩容但不得低于 1x Indexer blocks |
| `enable_row_mode_decode_graph` | `False` | 允许普通 Q=1 或 MTP3 uniform Q=4 decode 进入原生 FULL graph |
| `trace_points` | 关闭 | 拉起期解析的预留调测合同；当前仅接受 `first_sample`，尚无稳定日志 consumer |

实际 `DSAInputBatchCacheLayout` 列式状态按 `max_num_seqs` 分配；
`DSAResidentTokenPool.cache_slots` 额外增加一条共享 PAD 物理行，即
`max_num_seqs + 1`。两者都不按 `max_active_reqs=256` 分配。后者保留为
未来将 DRAM ledger 与 resident 活跃行生命周期解耦时的请求账本容量合同，
当前首先用于拒绝 `max_num_seqs > max_active_reqs` 的配置。

静态算子 ABI 还要求：

- `block_size=128`；
- `max_model_len <= 262144`；
- non-MTP resident budget 只能取 `6144/10240/12288`；MTP3 只能取
  `8192/10240/12288`，且都与 block size 对齐；
- LIDU caller-owned 输出列宽固定为 `16384`；
- Indexer head dim 为 128，MLA latent/rope 维度为 512/64；
- 当前 Indexer heads 支持 32 或 64。

模型支持采用能力判断，不仅依赖 architecture 名称。模型必须具备：

- MLA attention；
- sparse Indexer；
- 与当前 SFA 合同一致的 `index_topk`；
- DSA 所需的模型维度和 cache dtype。

模型若声明 `indexer_types`，其长度必须与 `num_hidden_layers` 相同且每项只能
为 `full/shared`。该列表是 checkpoint 物理拓扑的真源，不能只用
`index_topk_freq` 猜测哪些层缺少 Indexer 权重。GLM-5.2 的真实拓扑为：

```text
78 resident layers = 21 full Indexer layers + 57 shared followers
full layer ids      = 0, 1, 2, 6, 10, ..., 74
shared source       = 其前方最近一个 full layer
```

层 0、1 各自计算 topK；从层 2 开始，每个 full 层与后续三个 shared 层组成
一个四层 selection group。当前算子 ABI 仍限制 `max_model_len <= 262144`，
因此并不等同于已经开放 GLM-5.2 checkpoint 声明的 1M 最大上下文。

## 6. HBM 物理 Cache groups

### 6.1 Spec

`DSAIndexerKVSpec` 描述完整 Indexer K。每 token 只有一个 K 向量，因此覆盖
`FullAttentionSpec` 默认 K+V page 字节数。

`DSAResidentMLAAttentionSpec` 描述 resident MLA。它的
`sparse_head_dim` 不再包含 Indexer 维度，防止 MLA page 重复核算 Indexer。

两个不同的 DSA spec identity 让 v0.23 registry 将它们保留为独立 cache group。
普通模型两个 plane 的层集合相同；GLM-5.2 的 target Indexer group 只包含
full 层，是 target resident MLA 层的真子集。

启用 MTP3 后，模型自带 MTP layer 使用基线原生
`AscendMLAAttentionSpec`，`sparse_head_dim` 同时包含 BF16 MLA latent、RoPE
和 Indexer K。它不转换为 DSA spec，而是形成标记
`is_eagle_group=True` 的第三个 full-cache group。

### 6.2 容量与 group

finalized `KVCacheConfig.num_blocks` 表示 target resident MLA 的 base blocks。
Target Indexer 和可选 MTP full-cache 容量通过最终 `KVCacheTensor.size`
表达为：

```text
target indexer blocks = resident base blocks * indexer_mla_block_ratio
MTP full blocks       = resident base blocks * indexer_mla_block_ratio
```

KV group 顺序固定为 target Indexer、target resident MLA、可选 MTP full；运行期仍通过
`DSAKVCacheGroupIds` 按 spec identity 解析 group id，不在热路径假设固定
下标。

worker 的同层绑定顺序则固定为 resident MLA 在前、Indexer 在后。这是绑定
ABI，不等同于 group 顺序。

容量公式按最终 group 的真实层集合和 block multiplier 计费。增加一个 MTP
draft layer 会增加一份 `ratio * native BF16 full page` 成本，因而在固定 HBM
字节下自然降低 base block 数；容量报告、scheduler admission 和实际 tensor
分配使用同一结果，不会出现 target 可容纳请求数与 proposer cache 容量不一致。
启动容量报告会单独打印 MTP full group 的 layer 数、blocks、tokens、字节占用、
占总 KV HBM 比例及其 `max_model_len` 请求上限。

### 6.3 Worker tensor

每个 full 层的 Indexer plane 恢复为：

```text
[num_indexer_blocks, block_size, num_kv_heads, head_size]
```

它没有独立 attention backend 或 metadata builder，只作为对应 selection
group 的 full 层 LIDU cache 输入。shared 层不创建空壳 Indexer tensor。

resident MLA 继续复用 Ascend MLA backend 的 tensor 表示，但不再内嵌
Indexer K。两张 tensor 在初始化后分别绑定到共享模型层。

MTP full group 复用基线 allocation/reshape，得到恰好三项 BF16 tuple：

```text
(mla_nope_cache, mla_rope_cache, indexer_key_cache)
```

它使用独立 block table 与 slot mapping；物理 block ID 无需等于 target
Indexer，但 token coverage 必须相同。

## 7. 请求 cache 布局账本

### 7.1 为什么不是新的 request lifecycle

`request_cache_layout.py` 不接管 vLLM 的 waiting/running/finished/preempted
状态。它只记录 DSA cache 应采用的物理布局：

| 阶段 | Indexer plane | resident MLA plane |
|---|---|---|
| `PREFILL` | 完整 prompt | 完整 prompt，供 prefill 与后续 dump |
| `DENSE_DECODE` | 完整上下文 | 完整上下文 |
| `ENTER_SPARSE_DECODE` | 完整上下文继续增长 | 一次性替换为 budget + 固定尾区 |
| `SPARSE_DECODE` | 完整上下文继续增长 | 物理表稳定，只刷新有效长度/内容 |

可选 MTP full group 不参与上述阶段切换：PREFILL/DENSE/ENTER/SPARSE 均按
完整上下文增长，lookahead slot 复用基线 Eagle/MTP 分配语义。

### 7.2 持久状态

每个活动请求只持有一个 slotted `DSARequestCacheState`：

- `stage`；
- `target_resident_budget_tokens`；
- `sparse_budget_tokens`；
- `resident_valid_tokens`；
- `prefill_resident_released`。

目标 budget 在请求首次成功 admission 时按 prompt token 数选择，decode
期间不换档。`sparse_budget_tokens` 与 `resident_valid_tokens` 保留为
scheduler→worker 投影真源，避免 worker、eager 和 graph 各自重新计算。

### 7.3 Plan/commit

每次 `allocate_slots` 只创建一个 slotted、不可变
`DSARequestCachePlan`：

1. `plan()` 计算候选阶段和 target 两 plane 的需求，不修改持久 state；
2. manager 分别检查 target 两池与可选 MTP full-cache 池；
3. 容量满足后修改 block table；
4. `commit()` 原地刷新请求唯一 state；
5. 容量不足返回 `None`，state 与 resident 表不推进。

`tail_tokens` 和 ENTER 的 resident 整表替换标志是 plan 的派生属性，不额外
存储。planner 不在 steady decode 重建跨 step state。

## 8. 独立物理 pool 分配

`DSAKVCacheCoordinator` 为每个 group 持有独立 `BlockPool` 和 manager：

- `DSAIndexerKVCacheManager` 保存完整上下文块；
- `DSAResidentMLAKVCacheManager` 保存 dense 或 resident 布局块。
- 启用 MTP 时，原生 `FullAttentionManager` 保存 MTP BF16 full-cache 块。

这些 group 复用同一个基线 `BlockPool` 类型，但不是同一个实例：Indexer、
resident MLA 和可选 MTP 的物理 block 数及 id 空间彼此独立。当前所有 HBM
target Indexer 与 resident MLA pool 仍以 `enable_caching=False` 构造，尤其
resident MLA 的 budget/tail block 不代表完整上下文块，不能产生 prefix hash。
启用 prefix cache 时，DRAM complete-bundle pool 与可选 MTP full-HBM pool
启用原生 hash/LRU；MTP 继续使用基线 `FullAttentionManager`。

分配规则：

- PREFILL/DENSE 同时扩展 target 两 plane 与可选 MTP full cache；
- ENTER 先完成所有 pool 容量预检，再释放 target resident 旧满块、按预算重建表并
  保留原不满尾块；MTP3 同时补齐另一 parity tail，并把保留块放入与其
  logical block parity 对应的固定列；
- SPARSE 扩展 target Indexer 与可选 MTP full cache，target resident 物理块表
  必须与预期 `budget + tail_block_count` 容量一致；
- free 同时清理所有 group manager 和请求 cache state。

第一版 prefix hit 的 target 数据只来自 DRAM complete bundle；命中 token
通过 Request-aware DSA 分配事务处理，不开放通用 KV connector 的 external
computed tokens 或 delayed allocation。固定 MTP3 的三个 lookahead token
仍由同一个多 group allocator 处理，其他 speculative method 在配置边界拒绝。

## 9. 薄 Scheduler

`DSAOffloadScheduler` 继承上游 Scheduler，但不复制主循环。它只补充：

1. prefill/decode phase barrier；
2. waiting prefill 只有在所有物理 pool 都可 admission 时才阻塞 decode；
3. model output 返回后释放已同步 dump 的 prefill resident 满块；
4. preemption/resume 尚未支持时显式失败。

请求是否仍在 prefill 只由
`num_computed_tokens < num_prompt_tokens` 判定，不依赖 output 列表更新时序。
每个 chunk 都复用上游 token budget、`_inflight_prefills` 和
`_update_after_schedule()`；DSA 不复制 chunk 调度循环。中间 chunk 继续
保持 PREFILL，target 两 plane 与可选 MTP full cache 按本轮结束位置增量扩展。最后一个 chunk
仍以 PREFILL 数据面执行；output 返回后，scheduler 才释放已经在同一 stream
完成 dump 的 resident 满块。下一轮才进入 DENSE 或 ENTER decode。

waiting/skipped-waiting 为空且上游 `_inflight_prefills` 也为空的 steady
decode 直接进入 `super().schedule()` 快路径，不做额外 batch 扫描。

当前满块 dump 目标设计假设与模型 forward 位于同一 NPU stream，stream 内
保序。未来改为异步多 stream 时，必须新增 event/readiness 协议，不能只凭
host 输出返回释放 HBM 块。

## 10. Scheduler→worker 投影与执行接口

### 10.1 Scheduler→worker 投影

当前投影层实现以下合同：

- `DSARequestCacheLayoutProjection` 按 scheduled request 顺序列式承载
  stage、target/sparse budget、resident valid length；
- `DSAOffloadSchedulerOutput` 浅包装原生输出，基线字段继续共享原对象；
- coordinator 按已提交的 Indexer block table 扩展 DRAM directory，投影只
  携带本轮新增的连续 assignment 后缀，不重传完整 DRAM 表；
- 多进程 pickle 往返保留 projection 类型和内容；
- worker 先执行原生 `_update_states()` 的 remove/add/condense/reorder，
  再通过 `req_id_to_index` 投影到最终行序；
- `DSAInputBatchCacheLayout` 使用一个 `[7, max_num_reqs]` 的固定容量
  `CpuGpuBuffer`。四个 scheduler 投影列加 `row_mode`、
  `resident_pool_index`、`candidate_len`，七个 SoA 列在 device 侧均为
  连续向量；
- eager 后续使用 active-prefix，graph 使用 captured-prefix + PAD，
  二者共享同一个 owner；
- steady 行序与 scheduler projection 一致时，四个语义列批量写入；
  只有请求增删或重排才做一次 request-id 映射，并在同一结构刷新中同步
  resident/DRAM pool row；
- PAD 行在 owner 初始化时一次设置；batch 缩小时只清理由 active 退回
  PAD 的尾部，不在每个 step 重写整个剩余容量；
- 只有 ENTER 行携带 resident 全量 block IDs，并覆盖 worker request
  账本与对应 `MultiGroupBlockTable` 行；其他阶段不额外改写 block table。

该投影层明确不做：

- 在 worker 各 rank 根据长度重新决定阶段；
- 为 eager 和 graph 创建两套语义状态；
- 每 step 构造多份 request-id→scalar Python 字典。
- 为了消除 scheduler 的一次 O(B) projection 而复制上游
  `Scheduler.schedule()`；v0.23 输出构造处没有逐请求扩展钩子。

### 10.2 Eager 数据面

eager 路径按以下结构接通：

- `DSAResidentTokenPool` 为活跃请求分配与 InputBatch 行号解耦的稳定行；
- `cache_slots[selection_group, pool_row, position]` 是 LIDU 跨 step 原址更新
  的持久状态；普通模型一层一组，GLM-5.2 仅 full Indexer 层占一组；
  最后一列保存未初始化、first-fill 或 steady budget 标记；
- pool row 释放时只归还行号，下一次分配前执行唯一一次整行清理，避免
  request release 和 row reuse 重复写同一块大状态 tensor；
- `DSAWorkerDRAMStore` 按层持有固定地址 target MLA arena，并为 full Indexer
  层额外持有完整 Indexer plane；逻辑块表使用
  `pool_row × logical_block`，但只镜像 scheduler assignment，不在 worker
  分配 physical id；MTP proposer 不创建 DRAM payload arena；
- `DSAOffloadRuntime` 是 eager/graph 共用的物理 metadata owner，持有
  resident positions、active DRAM table、dump 列和 LIDU scratch；
- LIDU 的 topK/slot/miss/tail 输出只在当前 selection group 内存活：full 层
  写入，shared follower 依次消费，下一 full 层再覆盖；所有 group 串行复用
  同一套固定地址 scratch；跨 step 持久状态只保留 `cache_slots`；
- target Indexer、target resident 与 MTP full group 继续复用基线
  `MultiGroupBlockTable`。target Indexer 和 MTP full 使用完整序列 position；
  target MTP3 sparse resident 使用 `budget + position % 256`，普通 decode
  resident 仍使用 `budget + position % 128`；
- target prefill 使用基线 lightning-indexer 读取独立 Indexer plane；target
  decode 对整 batch 执行 LIDU→KSC→SFA-Offload；MTP proposer 继续走基线
  BF16 lightning-indexer/SFA；
- 普通 prefill/decode 在 SFA 后同 stream dump 本轮新满块；MTP target 验算只在
  下一 outer step 按 scheduler 已确认长度推进 durable watermark，并在当前层
  写 KV 前 dump，避免 parity tail 复用覆盖源块；
- 满块边界判定使用 worker-lifetime NumPy scratch；steady 无 dump step
  不构造 job 列，DRAM 表版本未变化时不重复 H2D。

eager 只消费 active-prefix；graph 通过同一 owner 的 captured-prefix + PAD
扩展执行 view，没有改变 eager 的请求语义。

### 10.3 图模式

复用 v0.23 原生 FULL decode capture/replay：

- `graph_gate.py` 只读取统一 InputBatch 投影，允许普通 Q=1 或 MTP3
  uniform Q=4 的 DENSE/ENTER/SPARSE 混排；prefill、非 uniform query 和
  capture-size miss 正常走 true eager；
- chunked prefill 不创建独立图或图专属元数据。每个 multi-token chunk
  复用 eager 数据面；最后一个 chunk 完成后，后续 decode 继续复用原生
  FULL capture/replay；
- 原生 dispatcher 的最终 uniform FULL keys 是 capture 容量唯一真源，
  继续负责图形状与向上 padding；仅支持具有独立 decode routine 的模式
  （例如 `FULL_DECODE_ONLY`），不支持 mixed/decode 共用的精确 `FULL`；
- DP 以原生全局图模式决议为准：任一 replica 处于 prefill 或
  capture-size miss 时共同 true eager；DP=1 且没有原生动态 blocker 时，
  gate 允许却未选中 FULL 才视为合同破坏；
- `DSAInputBatchCacheLayout` 的 device view 提供 captured-prefix + PAD，
  CPU 请求真源不变；
- `DSAResidentTokenPool` 与 `DSAOffloadRuntime` 在 dummy warmup/capture
  期间临时安装合法 SPARSE first-fill，forward 后统一恢复；
- graph 的 full-block dump 固定为 captured-row 宽度，未使用行以
  `src=0, dst=-1` 空转；eager 仍提交紧凑 jobs；
- `_build_attention_metadata`、attention builder、模型 forward 和 ACLGraph
  wrapper 均复用基线，不维护 graph-only 元数据类。

### 10.4 扩展场景边界

chunked prefill 已复用当前 request ledger 和固定 runtime owner；它没有开放
prefill/decode mixed forward。固定 MTP3 的 target 验算复用原生 compact TND
与 uniform FULL graph，双尾块直接覆盖跨块边界，不降级 Q=1；GLM proposer
继续复用社区 eager 路径及其独立 BF16 full cache。理想版动态 query 截短仍未实现。
本地 DRAM prefix hit 在 prefill 前完成 restore，后续 decode 继续使用同一
block table 和 graph 元数据；它不把 restore 算子捕获进 decode graph。
preemption、mixed prefill/decode 和 KV transfer 仍会改变 ledger 或状态恢复
合同，应按独立能力继续设计。

### 10.5 固定 MTP3 的双尾块与 proposer cache

妥协版固定 `num_speculative_tokens=3`，target 每请求最多形成 4 个 query。
最小 resident budget 提升为 8192，从而严格覆盖四路 top-2048 的最坏并集。
它不实现理想版的动态 query 截短，也不修改 rejection sampler 的接受语义。

SPARSE resident block table 固定为：

```text
[budget blocks][tail parity 0][tail parity 1]
```

完整序列位置 `p` 的尾区逻辑槽位为：

```text
budget + ((p // 128) & 1) * 128 + p % 128
= budget + p % 256
```

scheduler 只在 ENTER 时一次性分配并排列两块 tail；后续不交换 current/next
地址。provisional draft 被拒绝后无需清零，下一轮只以 scheduler 已回滚的
confirmed length 作为可见边界，陈旧字节不会进入 LI candidate、attention
tail 或 dump。

MTP draft 模块保留独立 native BF16 full-cache group。它复用基线 MTP
proposer 的 `kv_cache_gid`、block table、每 draft step slot mapping、seq lens
与 query start 元数据，不调用 DSA resident position 映射，也不占用 target
`cache_slots` selection state 或 DRAM payload arena。该 group 始终按完整上下文
增长，只在请求结束时释放；target 从 DENSE 进入 SPARSE 时只收缩 target
resident group。

LIM 的请求级 `candidate_len` 在 SPARSE 行表示本轮开始时已经 durable 的 DRAM
完整块前缀。SPARSE 每个 query 真正参与 LI 排名的历史边界由设备侧根据
`final_seq_len` 和 `actual_seq_lengths_query` 独立计算：

```text
visible_len(q)   = final_len - later_queries(q)
query_candidate = floor((visible_len(q) - 1) / 128) * 128
```

位于 `query_candidate` 但尚未进入 durable DRAM 的 token 只能来自当前两块
tail，LIM 直接把它映射为 parity slot，不为 KSC 生成 IO。四个 query 的 DRAM
miss 先做请求内稳定去重，再一次性更新 `cache_slots` 和 copy plan；不允许四行
依次驱逐同一请求的 resident 集合。

DENSE 不使用上述“完整块 + tail”拆分：其 KV 全在 HBM，每个 query 直接对完整
`visible_len` 排 top-2048；`visible_len<=2048` 时输出完整 causal prefix，超过
2048 时只输出 top-2048，不产生 DRAM IO，也不额外追加 tail。

MTP target 验算产生的 target 满块在下一 outer step 才成为 durable。runtime 比较请求级
`confirmed_full_blocks` 与 DRAM watermark，预先构造一份固定地址 dump plan；
每个 target 物理层在写入当前 KV 前，把同一 logical block 写到自己的 arena。
对应 DRAM id 也只在这个下一 outer step 分配；provisional query 即使跨过
128-token 边界并写满一块双尾区，也不会提前占用或写入 DRAM。MTP full
cache 不参与该 dump。当前实现保持单 stream 顺序；未来若改为异步
多流，必须增加 event 和逐层可见性协议。

启用 prefix cache 后，请求结束只释放 DRAM block 引用，已经 ready 的完整
bundle 保留 hash/LRU。MTP terminal flush 仍未实现：若最后一次验算新确认的
token 恰好补满 full block，而请求随后立刻 EOS/length finish，不会再有下一
outer step 执行 pre-write dump。第一版不会把该尾块标记 hash-ready，因此
保持正确但命中前缀少一块；后续再用显式 flush-only 数据面补齐。

## 11. Decode 数据面与算子 ABI

### 11.1 三种编号不要混用

| 名称 | 含义 | 典型消费者 |
|---|---|---|
| token position | token 在完整请求序列中的位置，范围 `[0, seq_len)` | Indexer、`topk_index`、DRAM logical block 推导 |
| resident logical slot | token 当前在该请求 resident 预算区中的逻辑槽位 | `cache_slots`、`topk_slots`、SFA-Offload |
| physical block id | HBM 或 DRAM arena 中的物理块编号 | block table、KSC、full-block dump |

`topk_index` 不是 HBM slot；`topk_slots` 也不是 token id。KSC 使用前者定位
DRAM 中的源 token，使用后者定位 resident HBM 的目标槽位。SFA-Offload
只消费换入完成后的 `topk_slots` 和尾块信息。

### 11.2 `cache_slots` 持久状态

每个 selection group 的状态形状为：

```text
[max_num_seqs + 1 PAD, aligned(max_model_len + 1)]
```

前 `W-1` 列是 `token position -> resident logical slot` 映射，最后一列
是该行的状态元数据：

- `0`：尚未建立 sparse resident；
- `-budget`：下一次 SPARSE 行执行 first-fill；
- `+budget`：该层 resident 映射已经建立，进入 steady update。

它是唯一按 selection group、跨 decode step 的 tokenwise 真源。
`DSAOffloadRuntime` 中的 LIDU 输出只是当前 group 的短生命周期 scratch，
不能替代 `cache_slots`。GLM-5.2 的 shared 层没有第二份映射真源。

### 11.3 LIDU 输出合同

下表是 A3 BF16 LIDU 使用的 caller-owned 固定地址输出；这些 buffer 由
runtime 按 selection group 串行复用：

| 输出 | 形状 | 语义 |
|---|---|---|
| `topk_index` | `int32[B, 1, 16384]` | 完整序列 token position；KSC 的源 token |
| `topk_slots` | `int32[B, 1, 16384]` | 对应 resident logical slot；KSC 目标与 SFA 稀疏索引 |
| `miss_count` | `int32[B]` | KSC 只消费 `[0, miss_count)` 前缀 |
| `tail_info` | `int32[B, 2]` | `[tail_slot_start, tail_token_count]` |

逐行语义：

| row mode | 行为 |
|---|---|
| `PAD=0` | 输出无效、`miss_count=0`，不修改持久状态 |
| `DENSE=1` | 计算完整 Indexer 序列的 top2048；`topk_slots=topk_index`、`miss_count=0`、无尾块追加；不修改 `cache_slots` |
| `SPARSE=2` 且 metadata<0 | first-fill：建立 target budget resident 映射，产生需要换入的 miss prefix |
| `SPARSE=2` 且 metadata>0 | steady：保留命中槽，只为 miss token 分配/复用槽并原址更新映射 |
| `SPARSE=2` 且 metadata=0 | 非法生命周期输入，不允许静默按 DENSE 处理 |

`ENTER_SPARSE_DECODE` 在 scheduler 仍是独立阶段，但映射到设备
`row_mode=SPARSE`，由负预算区分 first-fill。因此 DENSE、ENTER、steady
SPARSE 和 PAD 能共用同一条算子拓扑和同一张 captured graph。

“DENSE”表示 MLA 物理 cache 尚未收缩，完整上下文仍在 HBM 可寻址；它不
表示 SFA 做全量稠密 attention。当前 DSA decode 与模型原生 DSA 语义一致，
仍由 LIDU 从完整 Indexer 序列选 top2048 交给 SFA。

### 11.4 KSC 与 SFA-Offload

KSC 不重新判断命中，也不检查 `miss_count` 之后的输出：

```text
src_token_ids = topk_index[:, :, :miss_count]
dst_slots     = topk_slots[:, :, :miss_count]
```

因此“miss token 必须紧凑放在输出前缀”是 LIDU 与 KSC 的硬 ABI。host
不得用 `.item()` 读取 `miss_count` 后再拆请求；copy count 直接作为设备
tensor 交给 KSC。

SFA-Offload 读取每行前 2048 个 `topk_slots`，SPARSE 行再根据
`tail_info` 追加预算区之后的不满尾块有效 token。尾块保持全量参与的原因
是它尚未成为可卸载的完整 block。DENSE 行的 `tail_info=[-1, 0]`，其
top2048 slot 直接等于完整序列 position。

A5 C8 的物理 ABI 有一处受控差异：融合 LIDU 直接生成社区 QSFA 消费的固定
`int32[B,1,2176]` slot 行与 `resident_seq_lengths[B]`，同时生成 KSC 的
`copy_src_ids/copy_dst_slots/copy_counts`。packed KSC 只搬运有效 copy-prefix，
不再承担 attention metadata 构造。该差异没有引入新的 request 真源，也不改变
miss-prefix 合同；eager/graph 仍复用同一组固定 buffer。DENSE 行是零 IO，
resident length 保持完整 `actual_len`，因此 dense 序列长度不受 2176 限制。
A5 路径不消费 A3 的 `tail_info` scratch；融合算子已把有效尾部 slot 直接追加到
2176 列 QSFA 索引行中。

MTP3 时请求轴与 query 轴分离：copy plan/`cache_slots` 仍为 `B` 行，QSFA
attention slots 为 `T=sum(Q_b)` 行。SPARSE 的 `candidate_lens[B]` 是 durable
DRAM prefix，不是四个 query 共用的排名长度；LIM 在设备侧计算每 query
candidate，并对跨界后新完成、仍位于双尾中的 selected token 直接给出 parity
slot。此时 `resident_seq_lengths[B]` 为 `budget+256` 的物理可寻址 span，
而每条 attention row 仍只追加自身当前逻辑块内最多 128 个 causal tail slot。

GLM-5.2 的 full 层仍生成同一份输出。紧随其后的 shared 层不执行 LI/LIDU，
而是用该输出的 `copy_src_ids/copy_dst_slots/copy_counts` 对自己的 packed
MLA arena 执行 KSC，再用同一 `attention_slots/resident_seq_lengths` 执行
QSFA。所有操作在当前单 stream 内按层保序，所以下一 full 层覆盖 scratch
之前，三个 follower 已经完成消费；图模式捕获的也是这条固定地址依赖链。

这里没有复制基线的 IndexCache 元数据面：prefill、mixed forward 与非 DSA
路径仍使用模型初始化期已有的 `topk_indices_buffer`，full 层写入、shared 层
读取。DSA decode 才改用 LIDU scratch，因为基线 buffer 只保存 token topK，
无法表达卸载额外需要的 miss-prefix、resident slot 和 KSC copy plan。scratch
仍是 `DSAOffloadRuntime` 中 eager/graph 共用的一份固定地址 owner。

KSC 必须对四层中的每一层执行：四层只共享 token 选择，MLA KV payload 与
HBM/DRAM arena 仍逐层独立。现有 KSC ABI 一次只接收一层 arena，因此把三个
follower 的 KSC 提前塞到 full 层只会移动三次调用的位置，不会减少 launch
数或搬运字节，还会让 full context 越权持有后续层资源。当前选择在每层 SFA
前就地执行 KSC，保持基线的 layer-local 所有权和最短数据生命周期。若后续
开发可接收多层 arena 的 batched KSC，才值得评估每个 selection group 一次
批量换入及多流预取。

### 11.5 单行 first-fill 示例

假设 prompt admission 后冻结 `target_budget=6144`，当前完整长度为
`6500`：

1. scheduler 计算最后一个不满块之前共有 6400 个 token，提交 ENTER，
   resident MLA 表被替换成 6144 预算槽加一个尾块；
2. worker 写 `row_mode=SPARSE`，并将该请求所有层的 metadata 列设为
   `-6144`；
3. LIDU first-fill 选择并排列 6144 个 resident token，写
   `cache_slots[token_position]=resident_slot`；
4. `miss_count=6144`，KSC 将 miss-prefix 从 DRAM 换入对应 HBM slot；
5. 中间已经完成的 256 个 token 位于 DRAM，当前不满尾块有 100 个 token；
   A3 ABI 写 `tail_info=[6144, 100]`，A5 融合 ABI 则直接把这 100 个尾部 slot
   追加到 2176 列 attention row，二者都让 SFA 使用重要 top2048 加完整尾部；
6. LIDU 把 metadata 改为 `+6144`，下一 step 进入 steady。

后续若 top2048 只有 37 个 token 不在 resident，LIDU 只把这 37 个 miss
放在输出前缀，KSC 只搬 37 个 token；更大的 6144 resident budget 正是
用空间降低跨 step miss 的手段。

### 11.6 Attention 接入点

v0.23 不再保留 v0.19 的通用 `attention_begin/after_indexer/
attention_finished` hook 对象。当前接入直接落在 vLLM-Ascend 自有
`AscendSFAImpl` 的既有阶段：

| 位置 | DSA 行为 |
|---|---|
| model runner cache 初始化 | 为每个 SFA layer 绑定 `DSALayerOffloadContext` |
| `indexer_select_post_process()` prefill 分支 | 复用基线 lightning-indexer，但读取独立 Indexer plane/table |
| `indexer_select_post_process()` decode 分支 | 调用该层 context 的 LIDU→KSC，返回 `DSAOffloadSelectionOutput` |
| shared 层 `skip_topk` decode 分支 | 不执行 LIDU；复用所属 full 层输出，对本层 arena 执行 KSC |
| `_execute_sparse_flash_attention_process()` | 识别上述 selection，调用 SFA-Offload |
| SFA 返回后、`v_up_proj` 前 | 调用该层 full-block dump |

这里仍保留“选择/物化”和“attention 计算”两个清晰阶段，只是使用真实
`AscendSFAImpl` 调用点，而不再额外维护一套空的 hook 生命周期。层的
Indexer cache、layer id 和 runtime 在初始化期绑定，热路径不重复遍历
`static_forward_context`。

## 12. Hot DRAM 与满块卸载

### 12.1 Arena 和 ledger

EngineCore 在基线 HBM `KVCacheConfig` 已完成各 rank 收敛后，只观测一次
当前单机可用内存（同时考虑常见 cgroup 上限），生成
`DSADRAMCapacityPlan`：

该自动规划首版只承诺单机 TP（PP 仍由配置合同拒绝）；多机 TP 需要按节点
而非全局 TP 数切分内存，尚未开放。

```text
minimum = HBM Indexer blocks per rank
desired = ceil(minimum * hot_cpu_block_multiple)
per_rank_available = host_available / tensor_parallel_size
max_usable = floor(per_rank_available / complete_block_bytes) - 1
usable = max(minimum, min(desired, max_usable))
```

减一是因为所有 worker arena 都保留已清零的 block `0` 作为空映射。若完整
block 的最大可用数量小于 HBM Indexer blocks，启动直接失败；若只是不足
配置倍数，则统一缩容并在容量报告中打印实际倍数。计划随类型化
`DSAKVCacheConfig` 同时下发 scheduler 和所有 TP worker；worker 不允许自行
缩容，任一 swapped-memory arena 创建失败会终止启动。

一个逻辑 DRAM block 包含本 rank 的全部 target dense 语义 payload：

```text
A3: all full Indexer BF16 planes + all resident MLA NOPE/ROPE planes
A5: all full Indexer FP8-key/FP32-scale planes + all packed-C8 MLA planes
```

GLM-5.2 只保存 21 个 full Indexer 的 payload，57 个 shared follower 只保存
自己的 MLA；它们复用所属 full 层的 Indexer 结果，不重复落一份 Indexer。
妥协版 MTP proposer 的 BF16 full cache 仍独立全驻留 HBM，不进入 DRAM
容量和 payload bundle。

`DSASchedulerDRAMBlockDirectory` 是 physical id 唯一分配者。它直接复用
vLLM 原生 `BlockPool(usable_blocks + 1)` 的 null block、refcount 和 free
queue，并按本轮确定可持久化的 durable full-block 前缀增量扩展；该前缀
始终受已提交 Indexer block table 覆盖：

durable block 数必须使用 finalized Indexer spec 的物理 block size。vLLM
通用 scheduler/hash 粒度在 hybrid group、context parallel 或显式 hash 配置
下允许与单个物理 group 不同，不能用于 DRAM 物理块计数。
此外，DSA projection 发生在基线 `_update_after_schedule()` 之后，此时
`request.num_computed_tokens` 已包含本轮 scheduled tokens：prefill/普通路径
直接使用该 end 值；MTP pre-write dump 则减回 scheduled tokens 得到本轮
开始时已确认的 durable 前缀，任何路径都不能再次把 scheduled 相加。

```text
request_id -> [logical block 0 -> dram id, logical block 1 -> dram id, ...]
```

只有新增连续后缀通过 scheduler output 发送。worker 的
`DSAWorkerDRAMStore` 使用 stable resident pool row 保存镜像：


```text
[resident_pool_storage_rows, ceil(max_model_len / block_size)]
```

worker 只校验越界和同一行内的重复 id，绝不在 dump 热路径补分配。prefix
命中允许多个 active request row 共享同一个只读 DRAM block；hash/refcount/LRU
唯一真源位于 scheduler 的原生 `BlockPool`，worker 只维护 active refcount。
请求结束后 worker 通过基线 `finished_req_ids` 清镜像行；scheduler 只释放
请求引用，ready block 的 payload/hash 保留到物理块真正被重新分配。

### 12.2 Dump 计划与执行时序

model-forward 元数据准备阶段仍只批量识别本轮新完成的 full block；对应
DRAM id 已由 scheduler 预先下发。逐层 attention 完成 SFA 后：

```text
resident MLA source block ids
dense Indexer source block ids
dst_dram_block_ids
```

每个 target 层把 MLA payload 写到同一逻辑 id；full Indexer 层还通过
`KvCacheFullBlockDumpBytes` 写入 Indexer plane。后者把初始化期建立的连续
cache byte view 当作 opaque payload，支持 A3 BF16 与 A5 FP8/FP32 的不同
dtype，而不在热路径执行转换或 `.contiguous()`。算子不感知请求、stage 或
token position；prefill 与 decode 使用相同 job 列。

当前所有 attention 和 dump 位于同一 NPU stream，stream 内严格保序：

```text
本层 SFA 读取 HBM -> 本层 MLA/可选 Indexer full-block dump -> 后续层
```

该 block 到下一 model-forward 才可能被 LIDU/KSC 消费。因此当前不需要
额外的“dump finished”状态、D2H polling 或 event。scheduler 在 prefill
model output 返回后释放已经 dump 的 resident 满块。未来引入异步多 stream
时必须重新设计 event/readiness 协议，不能沿用这一同步假设。

### 12.3 eager 与 graph 空转合同

- eager：没有新满块时直接跳过算子；有 job 时提交紧凑 src/dst 前缀；
- graph：为了保持 captured topology，始终传 captured-row 固定宽度；
  MLA 与 full Indexer dump 都使用同一组固定 src/dst owner，空行使用
  `src=0, dst=-1`；
- `dst=-1` 是 dump 算子的唯一空转哨兵，不能和 DRAM ledger 的空映射
  block `0` 混用。

### 12.4 本地 DRAM prefix cache 第一版

EngineCore 继续复用唯一 `Request.block_hashes`；DSA 不构造第二份 token hash
链。查询阶段只生成零引用 hit plan，不能提前 touch。HBM 三个 component 的
容量预检全部成功后，才一次提交：

```text
DRAM target hit -> touch complete bundle
MTP enabled     -> touch baseline full-HBM hit after EAGLE drop
target          -> allocate Indexer/MLA HBM destinations
worker          -> restore Indexer and MLA by independent destination columns
```

DRAM assignment 不等于 hash-ready。本轮所有 TP worker 在同一 stream 完成
全部 target layer dump，并成功返回 model output 后，scheduler 才为新增完整块
发布 hash。未 ready 的 assignment 在请求释放时放到 free queue 前端优先复用；
ready prefix 按基线 reverse-free 保持 tail-first eviction bias。

当前 phase barrier 会在 ready decode/ENTER step 隐藏 waiting prefill，因此
dense resident 释放/重分配与新请求 prefix lookup 不会同批交错。更根本的
安全性来自 scheduler 单线程事务和 `BlockPool.ref_cnt`：查询阶段不修改状态，
全组件 admission 成功后先 touch 命中块，再允许同物理 pool 分配新块。未来
开放 mixed batch 或 HBM 热层复用时必须保持该顺序，不能依赖调度遍历顺序。

命中请求通过同一个 `DSAOffloadSchedulerOutput` 获得：增量 DRAM assignment、
Indexer destination、MLA destination 和 ready watermark。worker 在基线
`_update_states()` 形成最终 HBM block table 后准备固定 restore buffer，并在
进入 model forward 前按已注册 target layer 恢复 DRAM payload。所有 copy 与
后续 forward 位于同一 stream；restore 不进入 attention hook，也不进入 decode
graph。prefix hit 水位同时初始化 worker durable watermark，避免同一 forward
把已恢复前缀重复 dump。

MTP proposer 仍全驻留 HBM，不执行 DRAM restore。MTP raw hit 复用基线
`is_eagle_group` 的“多检查一块、drop 最后一块”逻辑，生成 drafting hidden
states 所需的有效长度；最终命中是 target DRAM 长度与 MTP 有效长度的公共
前缀。MTP terminal step 尚未 flush 的新满块不发布，只会让可复用前缀略短。

最终目标不是固定“只查 DRAM”：后续应按每个逻辑块、每个 component 选择
`HBM > DRAM > L3` 中最热的可用副本，再在 Indexer、MLA、MTP 三个必需
component 之间取公共前缀。当前接口已把 Indexer/MLA restore 分列，避免锁死
未来的不等长回载。

当前功能验证阶段使用 INFO 日志 `DSA prefix lookup` 输出逐请求命中；容量
报告中的 prefix-enabled 状态和 vLLM 原生聚合 hit rate 同时保留。进入性能
验收后应将逐请求日志降为 DEBUG、采样或移除，避免在线高 QPS 日志放大。

### 12.5 Rank-local L2 与 AscendStore L3 对接状态

完整接口、异步 job/reuse-fence、component schema、支持矩阵和测试门禁见
{doc}`dsa_l2_l3_cache_design`。本节只保留总览。

RoadMap 中的 scheduler-owned DRAM pool 和完整 Indexer+MLA 本地 prefix bundle
已经落地；Mooncake Store-owned TCP 已通过 swapped L2 单卡、跨 NUMA、TP8
120-plane 逐字节门禁。P1-W rank0 异步 write-behind 与 P1-R 同步 prefix
loadback 均已接入当前代码；下一阶段不再重新设计 L2，而是关闭在线长稳，
再加入 HBM target hot tier。AscendDirect HBM 基线仍保留回归，但不作为
DSA Host H2H P1-W 数据面。

第一版采用以下读路径：

```text
scheduler 查询 rank-local L2 完整 bundle
    -> AscendStore 查询 L3 中更长的公共 Indexer/MLA 前缀
    -> 为 L3-only suffix 分配各 rank local L2 bundle
    -> Store-owned TCP 将 suffix 恢复到各 rank L2
    -> 为 L2 hit + L3-only suffix + 本轮新 token 一次完成 HBM admission
    -> worker 执行既有 L2 -> HBM restore
    -> model forward
    -> 既有 full-block dump 只物化本轮新满块
    -> worker 成功返回后发布新增 L2 hash
```

TCP 首版显式采用 `L3 -> L2 -> HBM`。这样对齐 LMCache 的固定 L2 registration
模式，不要求 TCP 识别 NPU HBM，并保证 L3 suffix 在 forward 前已经成为可供
KSC/prefix reuse 使用的完整 L2 bundle。外部 IO、L2 restore 都位于 model
forward / graph replay 之前，eager 与 graph 继续消费同一份最终 HBM block table
和 DSA metadata。
该路径是当前 P1-R/P1-HOT 的正式妥协版数据面。HBM pair 可以覆盖本地 L2
缺口，但所有 L3 suffix 仍先落 L2；请求释放 dense HBM 或进入 sparse decode
前必须提交完整 L2 ready 不变量。direct-HBM 仅保留为后续研究项。
chunked prefill 也沿用同一流程：首个有 external hit 的 chunk 完成恢复，后续 chunk
继续按当前 durable watermark 增量 dump。

L3 对同一个 token block 使用两个独立组件对象：

```text
dsa-target-indexer-v1(hash) -> 仅 full Indexer 层的全部物理 plane
dsa-target-mla-v1(hash)     -> 全部 target MLA dense plane
```

scheduler 只把二者最长连续公共前缀视为 L3 hit。本地 L2 仍保持一个 physical id
绑定完整 bundle；L3 不复用该 id，而以 block hash 为身份。这既保留本地原子
bundle 的简单性，也为后续 Indexer/MLA 从不同层级恢复留下空间。

L3 key 必须包含模型命名空间、DSA schema 版本、block size、层拓扑、plane
顺序/字节数和 component，不能直接复用 DSA-disabled 基线 key。A5 GLM-5.2
实测中，DSA packed-C8 完整 L2 block 为 6,904,320 bytes，而 DSA-disabled
AscendStore HBM block 为 12,189,696 bytes；即使 token hash 相同，payload ABI
也不同。

框架侧采用“DSA schema 适配器 + 既有 AscendStore Backend”的分层：

- DSA connector/scheduler 负责本地命中长度、外部增量命中、HBM admission、
  request 生命周期和 component 公共前缀；
- DSA worker 负责把 split HBM group 或 rank-local DRAM arena 映射为
  `keys/addrs/sizes`；
- Mooncake、MemCache 差异继续封装在现有 `Backend.exists/put/get/register_buffer`
  接口内，DSA 控制面不按 backend 分叉；
- Backend 的 put/get 必须返回可校验的逐 key 状态。DSA 初版固定 fail-fast，
  任何 rank、任何 component 失败都不能进入 forward 或静默 recompute。

L2 -> L3 使用异步 write-behind，而不是当前 AscendStore 非 layerwise 路径中的
每轮 `request_queue.join()`：

1. scheduler 在 L2 ready watermark 推进后生成尚未远端发布的 hash/component；
2. 仅 TP rank 0 提交共享 payload，其他副本 rank 不重复写；
3. scheduler 记录 physical block id -> pending job；该 id 即将被重新分配时，
   通过 `jobs_to_flush` 要求 worker 在 overwrite 前等待上传完成；
4. worker 返回明确完成状态后清理 reuse fence 并标记 remote-ready；
5. put 失败只损失 L3 副本，不破坏本地 L2；失败 key 不得对 lookup 可见。

当前 A5 使用 Store-owned TCP 连接 Mooncake，完整 L2 arena 直接作为 Host
source/destination；不引入 staging ring。AscendDirect/global_ub 与
direct-HBM 仅保留为后续研究，不属于当前部署合同。

当前阶段：

1. **P0 介质与 schema 门禁：完成**。21 个 full Indexer（key/scale）plane、
   78 个 packed MLA plane 和真实 6,904,320-byte bundle 已完成 TP8 门禁；
2. **P1-W rank0 异步写回：完成首版**。按 ready watermark 增量上传，具备
   reuse fence、完成聚合、去重、失败处理和 shutdown drain；
3. **P1-R 同步只读：完成首版**。`L2 local hit + Mooncake suffix -> all-rank
   L2 -> HBM -> forward` 已接入，保持 `kv_load_failure_policy=fail`；
4. **P2 稳态关闭：阶段性完成**。在线 continuous batching 已覆盖；graph
   确定性、FD 长稳和 external namespace GC 作为独立遗留继续跟踪；
5. **P1-HOT 妥协版：代码完成，待上板**。完整 HBM Indexer+MLA pair 可独立
   于 L2 命中并作为本地最长前缀；L3 suffix 固定 `L3 -> L2 -> HBM`，HBM-only
   L2 gap 由首个 chunk 的既有 dump 补齐，worker 返回后才推进 ready/hash；
6. MTP L3、preemption/resume、P/D、跨节点 TP 和 L3 eviction event 均在上述
   正确性闭环后单独开放，当前配置继续显式拒绝。

## 13. 统一 metadata 与图模式

### 13.1 生命周期分层

| 生命周期 | 数据 | owner |
|---|---|---|
| 拉起期 | 类型化配置、spec、固定容量 tensor | `AscendConfig` / model runner |
| 请求跨 step | 阶段、冻结预算、resident 有效长度 | scheduler/core planner |
| scheduler 跨 step | 请求 DRAM logical block→physical id 目录 | `DSAKVCacheCoordinator` |
| worker 跨 step | InputBatch 七列投影、stable pool row、`cache_slots`、DRAM 目录镜像 | `NPUInputBatch` / resident pool / DRAM store |
| 单次 model forward | token row mode、slot mapping position、active DRAM table、dump jobs | `DSAOffloadRuntime` |
| 单层 | LIDU 四个输出和 SFA selection view | `DSALayerOffloadContext` |

这套分层防止把 request lifetime、step lifetime 和 layer lifetime 混进一个
大对象。`DSAOffloadRuntime` 不复制 scheduler 阶段账本，只从 InputBatch
投影构造本轮设备 view。

### 13.2 七列 InputBatch SoA

`DSAInputBatchCacheLayout` 持有一个 `[7, max_num_seqs]`
`CpuGpuBuffer`：

```text
stage
target_resident_budget_tokens
sparse_budget_tokens
resident_valid_tokens
row_mode
resident_pool_index
candidate_len
```

这里的 SoA（Structure of Arrays）表示同一种字段沿 batch 连续，而不是每个
请求保存一个 Python 对象。scheduler 输出前四列；worker 派生 row mode、
resident pool 行和本轮 LI 候选前缀长度。
原生 `_update_states()` 完成 add/remove/condense/reorder 后再刷新 DSA
active-prefix，因此最终行序与基线 `InputBatch` 完全一致。

### 13.3 eager 与 graph 的关系

eager 和 graph 并不是两套 DSA 元数据：

```text
同一个 CPU 真源 / 同一个 CpuGpuBuffer / 同一个 resident pool
                   |
        +----------+----------+
        |                     |
 active-prefix view   captured-prefix + PAD view
        |                     |
      eager             FULL graph replay
```

DSA 不另建完整图编译流程。原生 vLLM/vLLM-Ascend 继续负责：

- `VLLM_COMPILE` 与 piecewise 编译；
- FULL decode capture size 过滤、向上 padding 和 dispatcher；
- `_build_attention_metadata`；
- ACL graph capture/replay 和 DP 全局图模式决议。

DSA 只增加：

- 读取统一行状态的准入 gate；
- captured-prefix 的 PAD 行；
- capture dummy 所需的合法 SPARSE first-fill 临时状态；
- 固定地址的 LIDU/KSC/SFA/dump 元数据 view。

dummy capture 不代表真实请求。捕获前临时安装，捕获后在 `finally` 恢复，
不能推进 CPU 请求阶段、占用真实 DRAM ledger 或把 LIDU 原址更新遗留给首个
真实 replay。

### 13.4 DSA 图准入失败为什么是 true eager

vLLM/vLLM-Ascend 基线在 `VLLM_COMPILE` 下同时具备 piecewise compiled
forward 和 FULL ACL graph replay。不过当前 DSA 对预期的 row-mode gate
miss 会显式设置 `force_eager`，并在模型调用处设置
`skip_compiled_model_forward`，因此不会落到 piecewise：

- `enforce_eager=True`：全程 true eager；
- `VLLM_COMPILE + DSA gate miss`：跳过 compiled model，本轮 true eager；
- `VLLM_COMPILE + DSA gate hit`：FULL ACL graph replay。

prefill、multi-token、capture-size miss、cascade/encoder blocker，以及 DP
全局图模式决议降级，都是当前 DSA 的预期 true-eager 场景。piecewise
backend 仍属于基线编译体系的一部分，但不是 DSA gate miss 的执行后端。

## 14. 请求生命周期与资源回收

```mermaid
stateDiagram-v2
    [*] --> PREFILL
    PREFILL --> DENSE: "prompt 未达到 sparse 条件"
    PREFILL --> ENTER: "首次 decode 已满足激活与 resident 条件"
    DENSE --> DENSE: "完整上下文继续增长"
    DENSE --> ENTER: "首次满足 sparse 条件"
    ENTER --> SPARSE: "resident 整表替换 + first-fill"
    SPARSE --> SPARSE: "steady LIDU/KSC/SFA"
    DENSE --> [*]: "finished/free"
    SPARSE --> [*]: "finished/free"
```

请求结束时：

1. 基线移除 InputBatch 行；
2. resident pool 归还 stable row，但不立即清大 tensor；
3. scheduler DRAM directory 释放请求引用；启用 prefix 时 ready block 留在
   hash/LRU 中，未 ready 后缀优先回收；worker 收到 `finished_req_ids` 后
   清除 stable row 镜像；
4. 该 resident row 下次 `acquire()` 时只做一次所有层整行清理；
5. scheduler/core 同时释放所有 HBM group manager 和请求 cache-layout state。

把大 `cache_slots` 清理延迟到 row reuse，避免 release 和 acquire 各写一次。
“临时未被本轮调度”不等于 request finished；persistent InputBatch 在
condense/reorder 时必须保留该请求的 resident/DRAM 所有权。

当前 preemption 明确失败。未来目标不是把全部状态绑死在一个生命周期：

- tokenwise resident row 按 `max_num_seqs + PAD` 分配，preempt 时可释放；
- DRAM ledger 应有独立、可跨 preempt 保留的请求行；
- resume 重新分配 resident row，并用 DRAM ledger 执行 first-fill。

该拆分尚未实现，因此当前文档不能把 `max_active_reqs` 描述成已经存在的
独立 DRAM 请求池。

## 15. 性能不变量与已知成本

热路径必须保持：

- 不按 DENSE/SPARSE 拆 Python 子 batch；
- 不用 `.item()`/`.to("cpu")` 读取 miss、stage 或 row mode；
- 不逐层创建大 tensor，LIDU 输出由所有层串行复用；
- DRAM table 版本不变时不重复 H2D；
- prefix 关闭时 restore 固定 buffer 只保留一个哨兵槽且不调用 prepare；
  prefix 开启但无命中时只在 `_update_states` 后执行一次空任务判断，不增加
  逐层 attention 热路径；
- 无 dump 的 eager step 不调用 dump 算子；
- graph 使用固定地址 view，不在 replay 前替换 owner；
- request-id 映射只在 InputBatch 结构变化时做，steady 行序批量刷新。

当前仍有三个已知成本：

1. scheduler 需要在 `super().schedule()` 返回后对最终 scheduled requests
   做一次 O(B) projection。vLLM v0.23 没有逐请求输出扩展钩子；为了消掉
   这一次遍历去复制 `schedule()` 得不偿失。
2. v0.23 的 fused Indexer 投影返回带行 stride 的 `weights` 后缀 view。
   A5 融合 LIDU 已通过 `weight_stride` 直接读取该 view，不再逐层执行
   `weights.contiguous()`；A3 路径继续沿用社区算子的既有布局合同。后续改动
   必须保留这一零临时拷贝性质。
3. 完整 bundle 使 full Indexer 层在 full-block dump 时增加一个或两个
   byte-plane copy kernel；graph 空转 step 也会保留这些节点。首版优先保证
   目录与 payload 语义完整，上板后必须单独比较无 dump steady step 和
   128-token 边界 step，再决定是否把同层 MLA/Indexer copy 融成单算子。

## 16. 代码索引

| 模块 | 当前职责 |
|---|---|
| `dsa_offload/config.py` | 类型化配置、算子 ABI 和支持矩阵校验 |
| `dsa_offload/contracts.py` | 框架与四个算子共享的静态常量 |
| `dsa_offload/model_support.py` | 模型能力判断 |
| `dsa_offload/kv_cache.py` | spec、HBM group、DRAM 容量合同、绑定顺序和报告 |
| `dsa_offload/kv_cache_coordinator.py` | HBM 多物理 pool、DRAM/MTP 公共命中与 scheduler DRAM directory |
| `dsa_offload/kv_cache_manager.py` | 阶段感知分配与 prefix-hit admission/commit 事务 |
| `dsa_offload/request_cache_layout.py` | 请求 cache 布局 plan/commit |
| `dsa_offload/scheduler.py` | 薄 phase barrier 与输出后释放 |
| `dsa_offload/scheduler_output.py` | scheduler→worker 最小列式投影 |
| `dsa_offload/input_batch.py` | worker 固定容量行状态与 ENTER 整表覆盖 |
| `dsa_offload/graph_gate.py` | 原生 FULL decode graph 的纯准入策略 |
| `dsa_offload/resident_pool.py` | stable pool row 与逐层 LIDU `cache_slots` |
| `dsa_offload/scheduler_dram_pool.py` | 单机容量计划、DRAM hash/LRU 与 scheduler physical-id 目录 |
| `dsa_offload/worker_dram_store.py` | 固定 payload arena、共享 active refcount 与 worker 逻辑表镜像 |
| `dsa_offload/runtime.py` | eager/graph 共用 metadata、preforward restore owner 与逐层上下文 |
| `dsa_offload/ops.py` | LIDU、KSC、SFA-Offload、双向 byte-block copy tensor ABI |
| `core/kv_cache_interface.py` | DSA spec/manager registry 注册 |
| `worker/npu_input_batch.py` | 可选 DSA buffer owner |
| `worker/model_runner_v1.py` | cache 初始化、行投影、多 group slot mapping 与 runtime 绑定 |
| `attention/sfa_v1.py` | DSA attention 算子链和层后满块 dump |
| `platform.py` | 配置收敛、scheduler 类和启动期校验 |

## 17. 当前验证状态

已获得的 910C 证据：

- `python -m pytest tests/ut/dsa_offload -vv --tb=short` 共 114 项通过；
- DSA disabled GLM-5.1 回归通过；
- DSA `cache-init` 成功，HBM 容量报告只打印一次；
- `async_scheduling=True` 按支持矩阵拒绝；
- Indexer/MLA 3:1 容量和双 tensor 初始化符合预期；
- GLM-5.1 W4A8、bsz=4、约 8K prompt 的 eager 生成初验通过；
- FULL decode graph 已覆盖 bsz=4 的四条相同约 8K prompt；
- FULL decode graph 已覆盖约 5K/20K/8K/40K 混合长度和不同预算档位。
- chunked prefill 的配置、阶段、增量 component-wise 分配、连续 chunk dump 和
  phase barrier 已补 UT；
- 约 70K/40K/8K/5K 的四条混合请求在
  `max_num_batched_tokens=4096/8192/16384` 下，DSA 与 baseline 的
  首 token 均对齐。

已获得的 A5 packed-C8 证据：

- `vllm_a5_li_manage_nomtp_c8` 的 DENSE、三档 SPARSE budget、PAD、
  first-fill/steady 与 graph replay 算子用例通过；
- packed KSC 的独立搬运及 LIDU→KSC 可见性用例通过；
- GLM-5.1 W4A4C8 的 eager 与 FULL decode graph 端到端初验通过；
- GLM-5.2 的 21 full + 57 shared 拓扑、紧凑 selection state、
  full→shared→next-full 固定地址 graph 链已有 UT/E2E 算子覆盖。
- DSA+MTP 全量 UT 通过；GLM-5.2 W4A4C8 已通过离线 eager、离线 FULL
  graph 和在线 FULL graph 冒烟。FULL graph 覆盖单请求约 40K，以及
  8K/5K/70K/40K 的 bsz=4 混合长度。
- scheduler-owned complete DRAM pool 的 Python UT 与新增 byte-dump A5
  eager/graph 算子 UT 通过；GLM-5.2 容量报告确认 21 个 full Indexer、
  78 个 resident MLA，完整 bundle 未为 shared Indexer padding；
- no-MTP 已覆盖 bsz=4 的约 8K/5K/70K/40K eager/FULL graph，以及
  active=2（约 5K/40K）、capture sizes 仅 1/4 的向上 padding和 eager；
- MTP3 已覆盖约 8K/5K/70K/40K 的 bsz=4 FULL graph，以及单请求约 40K
  的在线推理；优化版 LIM 的 32/64 heads、shared durable-prefix、miss union、
  双尾与严格单调 TopK 共 24 项 A5 算子用例通过。上述结果作为 scheduler
  DRAM pool 与妥协版 MTP 第一版常见场景门禁。

本地 DRAM 与 L3 prefix cache 第一版的当前证据：

- Windows 隔离环境中，新增 hash-ready/LRU/shared-ref、MTP 公共命中、
  prefix allocation、restore metadata 和既有 DSA 回归共 171 项通过；
- Ascend 服务器上的 DSA 全量 Python UT、A2 SFA UT 与 A5 C8 算子 E2E
  已通过；generic byte-copy 的 HBM→swapped DRAM 和 swapped DRAM→HBM
  round-trip 实测为 `2 passed, 3 skipped`，A5 跳过项均为不适用的 BF16
  paired-plane 算子；
- Python `compileall`、Ruff、`git diff --check` 与在线脚本 `bash -n` 通过；
- Mooncake P1-W 已覆盖 TP8 eager/graph、chunked prefill、混合长度和在线
  graph，rank0-only put、remote-ready key、reuse fence 与 shutdown drain
  均有成功日志；
- P1-R 已覆盖 all-rank `L3 -> L2` get、TP consensus、既有 restore 和
  Local-L2/L3 replay 对照；bsz=1/eager 保持严格 token 门禁，多请求 graph
  的数值变化已通过 Local-L2/Local-L2 与启停 L3 对照归因。

这些结果证明当前核心控制面、eager、graph、L2 与 L3 P1 主路径可运行。
尚需补齐：

- QA 数据集 disabled/eager/graph 精度对照；
- bsz=1 与 bsz>1、全 DENSE、全 SPARSE、ENTER mixed；
- active rows 小于 captured rows 的 PAD；
- 请求完成、行复用和长时间 continuous batching；
- chunked prefill 在 `max_tokens>=32` 或正式 QA 数据集上的 eager/graph、
  bsz=1/bsz>1、ENTER/steady sparse decode 与并发调度回归；
- DeepSeek-V3.2 强制回归；
- profiling 性能基线；
- LocalDRAM/L3 prefix 的在线 continuous batching、长稳 FD/连接、request
  finish、row/physical-id reuse 与 load/store completion 联合 drain；
- HBM target prefix、MTP 多级 prefix、L3 layerwise/direct 与故障恢复；
- GLM-5.2 W4A4C8 的完整 QA、长稳并发、请求结束/行复用压力与性能回归。

完整测试命令和验收要求维护在 `examples/dsa_demo/README.md`。

## 18. v0.19 知识归档说明

本设计文档已经吸收 v0.19 原型中仍有价值的内容：插件/patch 时序、双 HBM
plane、hot DRAM、四阶段请求布局、列式元数据、stable resident row、
LIDU/KSC/SFA ABI、full-block dump、eager/graph 共用 owner、生命周期和
测试矩阵。以下旧形态有意不保留为当前实现：

- 全局 Scheduler monkey patch 与输出类重绑定；
- 复制 `_update_states()`、`_prepare_inputs()` 或图 dispatcher；
- packed MLA+Indexer page-size 补丁；
- 旧 GatherSelection 算子链和兼容 fallback；
- 运行时环境变量式 DSA trace 开关；
- 已无消费方的异步 dump 状态。

因此 v0.19 checkout 不再是理解、运行或测试 v0.23 DSA 的必要依赖；它只
保留历史提交溯源价值。
