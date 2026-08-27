# DSA Rank-local L2 与 AscendStore L3 分层缓存设计

## 1. 文档状态

本文是 DSA 稀疏卸载接入远端 KV pool 的详细设计，基于：

- vLLM v0.23；
- vLLM-Ascend v0.23.0 正式版；
- 当前 DSA custom 分支；
- 单机 8 x Ascend 950DT；
- GLM-5.1/GLM-5.2 W4A4C8 为首要验收模型；
- Mooncake `0.3.11.post1` + Store-owned TCP 为第一后端；
- MemCache `1.2.0` 临时 local-EID 版本为第二后端。

主 DSA 架构、模型执行和算子 ABI 仍以
{doc}`dsa_offload_design` 为总设计。本文只展开 L1/L2/L3 分层缓存、
Connector 生命周期、异步上传和后续 HBM 热层复用。

当前状态：**P0 Store-owned TCP 的单卡、跨 NUMA 与 TP8 120-plane 门禁已通过；
P1-W rank0 异步 write-behind 已完成 TCP 框架适配、UT，以及 TP8 离线单 40K
长请求的 eager/graph、5K+8K+40K+70K 混合长度多请求 graph/chunked prefill
和在线 graph 单 40K 流式请求的完整闭环验收，首版正确性阶段已经关闭。
L3 prefix lookup/loadback 尚未实现，下一阶段进入 P1-R。**

## 2. 目标与非目标

### 2.1 目标

1. 保留当前 scheduler-owned DSA L2 DRAM directory，不重新建立第二套本地
   block manager。
2. 新请求按唯一 `Request.block_hashes` 查询本地 L2 与远端 L3。
3. L3 只补充 L2 之后的连续 prefix suffix，不重复加载 L2 已命中的部分。
4. TCP 首版将 L3-only suffix 恢复到各 rank 已注册的 L2，再复用现有
   L2->HBM restore；不为 TCP 另建 HBM staging 或第二套 read metadata。
5. prefill/decode 新 durable 满块完成同步 L1->L2 dump 后，由 TP rank0
   异步执行 L2->L3 write-behind。
6. 请求完成不等待 L3 put；块即将被复用时才使用基线风格 reuse fence。
7. eager 与 graph 共用相同 scheduler/worker 状态和最终 HBM block table；
   外部 IO 不进入 graph。
8. Mooncake/MemCache 复用同一 DSA 控制面，后端差异封装在 AscendStore
   `Backend` 层。
9. 为未来 component-wise `HBM > L2 > L3` source plan 保留接口。

### 2.2 第一版非目标

- MTP target cache 的 L3 存储；
- request preemption/resume；
- async scheduling；
- PP、DCP、PCP 与跨节点 TP；
- P/D 分离；
- L3 eviction policy 定制；
- L3 SSD tier；
- L3 load failure 后静默 recompute；
- 修改 Mooncake/MemCache 社区源码的淘汰算法。

## 3. 当前基线盘点

### 3.1 DSA L1/L2 已有能力

当前 target cache 具有三个不同的物理语义：

| 层级/平面 | 当前物理语义 | block id 空间 |
|---|---|---|
| HBM Indexer | 完整上下文 dense Indexer | 独立 pool |
| HBM resident MLA | prefill 时 dense；sparse decode 时 budget+tail | 独立 pool |
| rank-local DRAM | 完整 dense Indexer+MLA bundle | scheduler-owned 独立 pool |
| MTP HBM | BF16 full cache，不卸载 | 可选独立 pool |

`DSASchedulerDRAMBlockDirectory` 已经复用 vLLM `BlockPool` 的 null block、
refcount、hash map、free queue 和 LRU。scheduler 是 DRAM physical id、hash、
ready watermark 与 request->logical block 映射的唯一真源。

`DSAWorkerDRAMStore` 在每个 TP worker 创建相同容量、固定地址的
`empty_with_swapped_memory` arena，只镜像 scheduler assignment；worker 不分配
DRAM id。A5 GLM-5.2 一个逻辑 L2 block 当前包含：

```text
21 个 full Indexer layer:
    key bytes + scale bytes
78 个 target MLA layer:
    packed C8 [128, 1, 656]
总计 6,904,320 bytes / rank / logical block
```

当前本地 prefix cache 只把完整 L2 bundle 作为 target 命中源。MTP 开启时，
target L2 hit 还会与 MTP full-HBM hit 取公共前缀。

### 3.2 当前 HBM target prefix 限制

目前 HBM physical pool 中只有 MTP group 开启 prefix caching。target Indexer
和 resident MLA pool 不发布 hash，原因不是 HBM 无法复用，而是 resident MLA
同时容纳：

```text
dense prefix block      有文本前缀语义
sparse budget block     无文本前缀语义
KSC resident slots      无连续前缀语义
tail/provisional block  未完全 durable
```

在缺少 block-role 状态前直接给整个 resident pool 开 hash 会产生错误命中。
因此第一版 L3 仍以 L2 为唯一 target 本地 prefix tier；HBM target 热层列为本文
第 18 节的高优先级后续项。

### 3.3 vLLM Connector 与 OffloadingConnector 可复用能力

vLLM v0.23 已经提供：

- `KVConnectorOutput` worker->scheduler 输出；
- `Scheduler.update_from_output()` 单线程更新入口；
- `kv_connector_worker_meta` 跨 TP 聚合；
- `kv_connector_no_forward()` 零 token control step；
- `OffloadingConnector` 的 job id、异步 store/load、完成回报；
- `_block_id_to_pending_jobs` 与 `jobs_to_flush` block-reuse fence；
- request finish 后 pending store 不阻塞客户端响应；
- `OffloadingManager.prepare/complete_store/load` 生命周期；
- worker `TransferResult(job_id, success, bytes, time, type)`。

这些结构应作为 DSA L3 的异步生命周期骨架。DSA 不应新建 scheduler 接收
线程，也不应使用独立 RPC 直接修改 scheduler BlockPool。

不能直接原样使用通用 `OffloadingConnector` 的原因是：

- 它的 store source 是普通 GPU/HBM block table；
- DSA sparse decode 的 resident block table 是 budget+tail，不再对应完整
  logical context；
- DSA 完整 source 位于自定义 rank-local DRAM arena；
- DSA 的 Indexer/MLA physical pool 容量和 block id 不同；
- DSA L2 一个 physical id 对应两个 L3 component；
- TP rank0-only store 的完成计数不是 `world_size`。

因此实现应复用 Connector/job 模型，但使用 DSA-specific scheduler/worker
schema adapter。

### 3.4 AscendStore/Mooncake 已验证能力

已验证：

- Mooncake `global_ub_device` 单卡、TP2、TP4、TP8 multi-buffer round trip；
- TP8 环形 remote get 逐字节一致；
- DSA-disabled GLM-5.2 完整模型 3200/3287 external hit；
- 每个 rank 25/25 keys、`failed=0`；
- 每 rank 单次恢复 304,742,400 bytes；
- warm replay token/logprobs 稳定；
- 零 HBM prefix hit 下发生真实 L3->HBM load。
- `empty_with_swapped_memory` 的完整 A5 DSA schema 可通过 Mooncake
  `global_ub` 直接 put/get，无需 staging ring；
- 单卡 exact schema 包含 42 个 Indexer plane、78 个 MLA plane、242 个
  registration region，总注册 110,477,312 bytes；
- 两个 component 的 put/get、interior block 地址和逐字节 digest 均通过；
- 同一 TransferEngine 同时注册普通 HBM 与 swapped LocalDRAM 可行。

`global_ub_device` 只创建 device endpoint。它会把 host memory 再次执行
`aclrtHostRegister` 后映射到 device 链路；而
`empty_with_swapped_memory` 在 torch_npu 分配期已经完成该注册，二次注册会以
`halHostRegister drvRetCode=12` 失败。`global_ub` 同时创建 device/host
endpoint，使 HBM 走 device、DSA L2 走 host，是当前正向合同。

尚未验证：

- DSA split HBM schema 的 AscendStore load；
- rank0-only DSA store + 全 rank DSA load；
- DSA L3 正常 shutdown/drain。

### 3.5 vLLM、vLLM-Ascend 与 DSA 的复用边界

当前实现不是让 `DSAL3Connector` 与 `AscendStoreConnector` 同时运行，而是按
职责做三层组合：

| 层次 | 复用内容 | 当前所有者 |
|---|---|---|
| vLLM Connector 规范 | Factory、`KVConnectorBase_V1`、scheduler metadata、worker hooks、`KVConnectorOutput` 聚合、shutdown | vLLM |
| Ascend L3 数据面 | `Backend`、Mooncake Store-owned TCP；可选 GlobalTE/HIXL AscendDirect | vLLM-Ascend |
| DSA 策略适配 | split schema/key、scheduler-owned L2 IDs、rank0-only store、dump event、reuse fence | DSA |

`AscendStoreConnector` 本身同样继承 vLLM `KVConnectorBase_V1`，因此
vLLM-Ascend 并未抛弃 vLLM 生命周期；它是在接口内部增加
`KVPoolScheduler/KVPoolWorker/Backend`。DSA 也应遵循同一个外层接口，但不能
直接复用两类 pool 对象，因为它们假定 store source 是普通 HBM block table、
各 group block identity 可按基线解释，并按 worker/rank 保存。DSA 的 durable
source 则是完整 LocalDRAM bundle，resident HBM 是 budget+tail，且首版只允许
rank0 写。强行套用会把错误的 HBM block 语义隐藏在“复用”名义下。

因此当前选择是组合 `MooncakeBackend`，而不是复制 Mooncake/HIXL 实现，也不是
继承后覆盖大部分 `AscendStoreConnector`。同一进程只配置一个 connector；DSA
配置校验拒绝 `AscendStoreConnector`、`MultiConnector` 等并行组合，避免双
scheduler 状态机和重复注册。

对 vLLM-Ascend 公共模块的修改限制为可独立回归的通用增强：GlobalTE 增量幂等
注册/解注册、Mooncake put 返回原始逐 key 结果、显式 close。未来若
AscendStore 抽象出可插拔 source schema、rank policy 和成功结果合同，DSA
应继续下沉复用这些公共能力，使 `DSAL3Connector` 收敛为薄策略层。

## 4. 不变量

### 4.1 唯一 token identity

所有层级继续使用 vLLM 创建的唯一 `Request.block_hashes`。DSA 不重新计算
token hash，不按 request id 建 L3 identity。

### 4.2 logical block 与 physical id

```text
logical block index  请求完整上下文中的第几个 128-token block
L1 physical id       Indexer/MLA/MTP 各自 pool 内编号
L2 physical id       完整 DSA bundle 的 rank-local arena 行
L3 key               内容寻址 key，不等于任何 L1/L2 physical id
```

### 4.3 durable 规则

只有以下 token 可以写入 L2/L3 prefix cache：

- 已完成 target forward；
- 已通过 MTP target verification；
- 位于完整 block；
- 对应 Indexer 和 MLA payload 都已完成；
- scheduler 已拥有该 logical block 的 block hash。

不满尾块和 MTP provisional token 不得发布 L2/L3 key。

### 4.4 scheduler 单线程真源

所有 physical-id reuse、job state、remote-ready、failure 和 prefix lookup
结果只在 scheduler 主线程更新。worker 后台线程只能向本地 completion queue
写结果，不能回调或直接修改 scheduler 对象。

## 5. L3 对象与 key schema

### 5.1 component 拆分

每个 token block 在 L3 使用两个对象：

```text
INDEXER component -> 所有 full Indexer layer 的 key/scale plane
MLA component     -> 所有 target MLA layer 的 dense packed plane
```

只有两个 component 都 ready 时，该 token block 才是完整 target L3 hit。

拆分的理由：

- Indexer/MLA 在 HBM 已是独立物理池；
- 未来允许两个 component 从不同层级恢复；
- component 失败可以独立重试；
- 避免将 L3 ABI 锁死为当前 L2 bundle；
- GLM-5.2 只保存 21 个 full Indexer，不为 shared follower padding。

### 5.2 key 必备字段

L3 key 至少包含：

```text
namespace/version
model identity
quant/cache format
block size
component
schema fingerprint
request block hash
```

示意：

```text
vllm-ascend:dsa-l3:v1:<model>:a5-c8:bs128:<schema>:indexer:<hash>
vllm-ascend:dsa-l3:v1:<model>:a5-c8:bs128:<schema>:mla:<hash>
```

schema fingerprint 应覆盖：

- full Indexer layer id 顺序；
- MLA layer id 顺序；
- 每个 plane dtype、byte size 和顺序；
- packed/非 packed 模式；
- Indexer shared topology；
- TP replica/shard 语义。

不能直接复用 DSA-disabled AscendStore key。当前 A5 GLM-5.2 的基线 HBM
对象约为 12,189,696 bytes/block，而 DSA packed-C8 L2 对象为
6,904,320 bytes/block，二者 payload ABI 不同。

### 5.3 TP identity

GLM/DeepSeek MLA 与 Indexer target payload 当前按 TP 副本处理：

- L2：每个 rank 保留本地副本；
- L2->L3：初版只由 TP rank0 上传一份；
- L3->HBM：每个 rank 分别加载到自己的 HBM block；
- key 不包含 TP rank。

Mooncake 的“贡献池容量”和“写入 DSA 副本”是两个独立维度，不能都按
rank0-only 处理。当前 vLLM-Ascend AscendStore 采用 Mooncake 内嵌 client
模式：每个 TP worker 都以 `contribute_memory=True` 创建 backend，并向全局池
挂载一段 `global_segment_size`；只有 scheduler 查询客户端使用
`contribute_memory=False`。因此单机 TP8 配置每 rank 4 GiB 时，名义全局池容量
为 32 GiB，而不是 4 GiB。

DSA 应按同一基线范式划分角色：

- 所有 TP rank：创建可贡献存储的 Mooncake backend/TransferEngine，将各自的
  segment 汇入全局池，并在 load 时注册本 rank HBM destination；
- TP rank0：作为当前副本型 Indexer/MLA payload 的唯一 producer，发起一次
  L2->L3 put，避免把相同 KV 数据按 TP 数重复写入；Mooncake Master 可把该次
  put 的对象分配到任一可用 segment；
- 其他 TP rank：不重复发起相同 put，但其贡献的 segment 仍是池容量的一部分，
  并在 load 时从全局池读取到自己的 HBM；
- scheduler：只拥有 lookup/admission 决策。远端 exists 可沿用 AscendStore 的
  rank0 lookup service/RPC 范式，不能让所有 rank 重复查询同一组 key；
- store completion 的 expected producer count 为 1，load completion 的
  expected worker count 为参与 TP load 的 rank 数。

Mooncake 也支持“每个 rank 为 dummy client、每个推理实例仅一个 real client
贡献整机内存”的部署方式，但这要求单独的 dummy-real/store-service 架构，
不是当前 vLLM-Ascend AscendStore backend 的执行方式。P1-W 初版曾只有 TP0
初始化 Mooncake，会把名义池容量从 `TP * per_rank_segment` 缩成一个 rank 的
segment；现已改为 all-rank segment contribution、rank0-only put。P1-W 验收
必须同时检查每个 rank 的 backend 初始化日志和唯一 producer 的 store 日志。

两卡 swapped Host 跨 segment 基础门禁已通过：TP0 的 1 MiB swapped buffer
可强制定向写入 TP1 segment，并由 TP0 从全局池完整读回。Mooncake/HIXL 基础
H2H 池化能力成立。120-buffer、6,904,320-byte 小 arena scatter/gather 也已
通过；每个 plane 使用 4,353-block swapped arena、TP0 注册总量约 30 GiB 并从
非零 block offset 读取的门禁同样通过。Mooncake API 从主线程移至 background
thread 后，put/get 与逐 buffer 校验仍通过，因此真实 write-behind 的线程设备/
通信上下文也已排除。真实 GLM-5.2 A5 双 key 布局同样通过：Indexer key 含
42 个 16,384/512-byte 交错 plane，MLA key 含 78 个 83,968-byte packed
plane，put/get 分别返回 `[0,0]` 与 `[354816,6549504]`。

TP8 框架日志暴露了两卡门禁没有覆盖的拓扑风险。TP0 producer 对由 Mooncake
Master 自动选择的 TP1/TP2/TP3 segment 可以继续推进，而对 TP4/TP5/TP7
segment 的 120-descriptor Host write 均在 30 秒后失败；NPU0-3 与 NPU4-7
恰好位于不同 NUMA/卡组。因而“all-rank contribution + rank0-only put”目前只
是待验证方案，不能再视作已完成设计。下一门禁必须依次覆盖：

1. TP2 进程、物理 NPU0->NPU4；
2. TP8 进程、显式 NPU0->NPU4；
3. TP8 进程、`preferred_segment=false` 的 Master automatic placement；
4. 与框架相同的 host IP、4 GiB global segment、1 GiB local buffer。

其中第一项已经复现失败：仅创建 NPU0/NPU4 两个 probe worker，固定 loopback
控制地址并将双 key 显式放到 NPU4 segment 后，TransferEngine 建链、segment
mount 和 source arena 注册均成功，但 Host put 在
`HcommChannelFenceOnThread` 返回 `hccl_ret=19 (HCCL_E_NETWORK)`。因此当前
问题不是 TP8 数量、模型通信并发、Master 放置或 DSA payload，而是 NPU0/NPU4
之间这条 `global_ub` Host endpoint direct data path。该结果仍不能证明硬件本身
不支持跨 NUMA；CANN/HComm/HIXL endpoint 选择、路由配置或当前 wheel 实现均
可能是根因。

同一 NPU0/NPU4 目标下，将 source/destination 改为 HBM tensor，并改用
`global_ub_device` 后完整往返通过：put 为 `[0,0]`，remote get 为
`[354816,6549504]`，逐 plane 内容一致。这条路径实际覆盖
`NPU0 HBM -> NPU4 进程 Host L3 segment -> NPU0 HBM`，说明跨 NUMA/卡组的
Device-to-remote-Host 与 remote-Host-to-Device 能力可用；故障不能归因于
Mooncake 全局池或跨卡组网络整体不可达，而应严格限定为 swapped Host source
与 Host endpoint 参与的 direct Host-to-Host 路径。

Mooncake/HIXL 源码进一步显示该失败路径不是唯一实现。HIXL 正式接口提供
`OPTION_BUFFER_POOL`，用于 Host direct 不可用、Host registration 受限或大量
小块传输时的中转模式；未注册地址会按 Host memory 处理。Mooncake Ascend
TransferEngine 却在初始化时显式设置 `adxl.BufferPool=0:0`，只有环境变量
`ASCEND_BUFFER_POOL` 非零时才重新启用。启用后，Mooncake 对 Host region 的
`registerMem` 会直接返回，不再把 30 GiB swapped arena 注册为 direct Host
memory，而由固定 staging pool 负责传输。当前 GLM-5.2 DSA 逻辑块为 6.59 MiB，
`ASCEND_BUFFER_POOL=4:8` 的单 buffer 为 8 MiB；HIXL 当前创建两组 device
pool，每组 `4 * 8 MiB`，合计约 64 MiB HBM/进程。对象大于单 buffer 时会按
8 MiB 切片循环复用，因此 staging 容量与 L2 总容量无关。该路径需先完成功能
诊断，但额外 H2D/D2H、跨卡 Device 带宽和 HBM 占用使其不作为 DSA 正式
fallback。

需要注意 HIXL 当前存在两条 engine 路由：设置 `GlobalResourceConfig`/
`protocol_desc`（包括 `global_ub`）时，`EngineFactory` 选择新版
`HixlEngine`，该实现明确只允许 `BufferPool=0:0`；不设置 protocol desc 时选择
兼容 `CommEngine/AdxlInnerEngine`，后者才实现上述 staging pool。实测
`global_ub + ASCEND_BUFFER_POOL=4:8` 在 TransferEngine 初始化期即以
`PARAM_INVALID` 拒绝，未进入数据传输。buffer pool 能力门禁因此必须先用
`comm_mode=auto`，后续若成立，再评估如何在正式 backend 中显式选择兼容 engine
而不是依赖模糊的环境默认值。

`auto + BufferPool` 已确认能进入 `CommEngine/AdxlInnerEngine`：两端 staging
pool 初始化成功，Host segment 与 swapped buffer registration 均被正确跳过。
当前失败推进到 `HcclCommInitClusterInfoMemConfig(ret=1)`，属于兼容 engine 生成的
rank-table/通信参数错误。HIXL 仓内所有同机双卡 CommEngine 样例均显式设置
`HCCL_INTRA_ROCE_ENABLE=1`，下一门禁补齐该变量，并在失败收集中保留
`rank_table` 与 `HcclCommInit` 日志。

补齐 `HCCL_INTRA_ROCE_ENABLE=1` 后，同卡组 NPU0/NPU1 与跨卡组 NPU0/NPU6
仍以相同方式失败。诊断日志给出确定根因：HIXL v1 generator 输出
`server_count/server_list/status/version`，但没有当前 HCCL parser 必需的顶层
`rank_count`。该问题位于 HIXL/HCCL 兼容层，不属于本项目允许修改的边界；本地
HIXL 私有补丁已撤销，也不把修通旧 CommEngine staging 作为 DSA L3 交付前提。
同步检查 HIXL `origin/master`（2026-08-28 拉取到 `61a2ee4`）后，主线 v1/v2
generator 仍未输出 `rank_count`，简单升级到 master 也不能消除该诊断缺口。

当前 `mooncake-transfer-engine-npu` wheel 以 `USE_ASCEND` 构建，会无条件安装
AscendDirectTransport；传入 `protocol=tcp|rdma` 不会选择通用 Host transport，
仍会进入 HIXL/HCCL。因此 NPU 探针撤销 TCP/RDMA 选项；在不重新构建或修改
Mooncake 的边界下，不能把通用 Mooncake Host transport 视为可用方案。

NPU wheel 已有的 direct Host 路径已分别用 `global_ub+pinned` 隔离 swapped SVM
地址、用 `global_ub_host+pinned` 隔离 mixed endpoint 选路。pinned 始终只是诊断
介质，不能替换 KSC、dump 和 restore 共用的 `empty_with_swapped_memory` L2 arena。

`global_ub+pinned` 的 NPU0/NPU6 单 descriptor 门禁仍以
`HcommChannelFenceOnThread: HCCL_E_NETWORK` 失败；allocation、Host registration
与 segment connect 均成功。因此 swapped SVM、scatter/gather 数量和大 arena
均已排除。

`global_ub_host+pinned` 的相同门禁也失败。日志确认协议只包含
`ub_ctp:host`，两端均由 `aclrtMallocHost` 分配并按 Host memory 注册，segment
连接成功，最终仍在 NPU0 到 NPU6 的 `HcommChannelFenceOnThread` 返回
`HCCL_E_NETWORK`。因此 mixed Device/Host endpoint 选路也已排除。当前
Mooncake NPU wheel/HIXL/CANN 组合不能证明跨卡组 direct Host-to-Host 可用；除非
通信侧提供经过验证的显式 HiXLEP route，否则停止继续排列组合 transport 参数。

该结论不否定 Mooncake 整体能力：基线 HBM KV Connector、
`global_ub_device` TP8 以及同组门禁仍然有效。阻塞边界仅是 DSA 所需的“任意
rank-local swapped Host L2 直接发布到全局 L3，并可跨组恢复”的数据面合同。

源码也不能被解读为“明确拒绝 H2H”。Mooncake Ascend Direct 可以注册
`MEM_HOST`，其 HIXL 依赖会把 Host/Host 描述符分类为 `COMM_TYPE_UB_H2H`，A5
接口样例也列出了 `ub_ctp:host`。但 Mooncake 的 Ascend Direct 文档只明确承诺
H2D、D2H、D2D；Mooncake 与 HIXL 都没有进一步承诺 A5 单机跨 NUMA/跨卡组
H2H。通用 Transfer Engine 的 NUMA-aware NIC 选择也没有在该 Ascend Direct
远端路径中实现；Mooncake 只选择 ADXL endpoint 并把传输交给 HIXL。因此当前
结果应定性为“抽象层支持 H2H，但目标拓扑在当前依赖和配置下未实现或未打通”，
而不是上层主动拒绝。

跨卡组 preferred put 失败后，不能把 NUMA/rank 拓扑直接写进 DSA scheduler。
DSA 对 L3 的合同仍应是任意合法 L2 source 可发布为全局对象、任意 consumer 可
恢复；direct、staging、relay、replica 和本地 ingest 都属于 L3 backend/transport
实现。**唯一对象、分片本地 producer**只能作为 backend 内部优化或临时诊断，
不能改变 key/schema 或让 scheduler 依赖物理 NPU 编号。HBM staging/relay 不作为
本项目可接受的生产 fallback。Mooncake AscendDirect 路径继续作为外部依赖阻塞；
Store-owned TCP 已通过 swapped Host、跨进程、跨 NUMA、TP8 与 120-plane 门禁，
因此成为 P1-W 正式 fallback。其他 L3 后端也必须先通过相同门禁，才能进入完整
框架集成。

P0 必须通过 TP8 逐字节比较证明 DSA payload 确实可按副本处理；未通过时不得
使用 rank0-only store。

## 6. 总体架构

```mermaid
flowchart TB
    P["Prompt token list"]:::prompt
    S["Scheduler<br/>split HBM block managers (L1)<br/>DSA DRAM block manager (L2)"]:::scheduler
    KO["KVConnectorOutput<br/>job result metadata"]:::metadata
    L3["L3 KV pool<br/>Mooncake / MemCache<br/>backend-managed eviction"]:::l3

    subgraph WORKERS["TP workers"]
        direction LR
        subgraph R0["Worker 0 / TP rank0"]
            direction TB
            C0["worker control<br/>IO job queue"]:::worker
            H0["split HBM arena (L1)<br/>Indexer / resident MLA"]:::worker
            D0["rank0 LocalDRAM arena (L2)<br/>complete dense bundle"]:::l2
            K0["③ KSC<br/>token-wise topK miss"]:::ksc
        end

        subgraph RN["Worker N / TP rankN"]
            direction TB
            CN["worker control<br/>load/completion"]:::worker
            HN["split HBM arena (L1)<br/>Indexer / resident MLA"]:::worker
            DN["rankN LocalDRAM arena (L2)<br/>complete dense bundle"]:::l2
            KN["③ KSC<br/>token-wise topK miss"]:::ksc
        end
    end

    LEGEND["① TCP prefix load: L3-only suffix -> L2 -> L1<br/>② Durable full block: L1 -> L2 sync; rank0 L2 -> L3 async<br/>③ Sparse decode: KSC token-wise L2 -> resident L1<br/>Future P3-HOT: component-wise HBM -> L2 -> L3 lookup"]:::legend

    P -. "compute Request.block_hashes" .-> S
    S -. "L3 block-hash lookup" .-> L3
    S -. "SchedulerOutput + Connector metadata" .-> C0
    S -. "SchedulerOutput + Connector metadata" .-> CN
    C0 -. "completed job results" .-> KO
    KO -. "scheduler-thread state update" .-> S

    D0 -->|"① local prefix restore, sync"| H0
    DN -->|"① local prefix restore, sync"| HN
    L3 -->|"① L3-only suffix to local L2, sync"| D0
    L3 -->|"① L3-only suffix to local L2, sync"| DN

    H0 -->|"② full-block dump, sync"| D0
    HN -->|"② full-block dump, sync"| DN
    D0 -->|"② rank0 write-behind, async"| L3

    D0 --> K0
    K0 --> H0
    DN --> KN
    KN --> HN

    classDef prompt fill:#1f6d8c,color:#ffffff,stroke:#102f3d,stroke-width:2px;
    classDef scheduler fill:#ed762f,color:#ffffff,stroke:#8d3f12,stroke-width:2px;
    classDef worker fill:#169dcc,color:#ffffff,stroke:#07546f,stroke-width:2px;
    classDef l2 fill:#8fc3e8,color:#111111,stroke:#1786bd,stroke-width:2px;
    classDef l3 fill:#4ba82e,color:#ffffff,stroke:#1e5d12,stroke-width:2px;
    classDef metadata fill:#ffffff,color:#111111,stroke:#555555,stroke-width:1px;
    classDef ksc fill:#ffd54f,color:#111111,stroke:#c28b00,stroke-width:2px;
    classDef legend fill:#f6f6f6,color:#111111,stroke:#777777,stroke-width:1px;

    linkStyle 0,1,2,3,4,5 stroke:#222222,stroke-width:2px;
    linkStyle 6,7,8,9 stroke:#e31a1c,stroke-width:3px;
    linkStyle 10,11,12 stroke:#169bd5,stroke-width:3px;
    linkStyle 13,14,15,16 stroke:#e5ad00,stroke-width:3px;
```

图中虚线是 scheduler/worker/L3 lookup 的控制信息，红、蓝、黄实线分别对应
① prefix load、② full-block 下沉、③ decode KSC miss IO。L3->L2 红线只传
本地最长命中之后的 L3-only suffix，再复用现有 L2->L1 restore；L2->L3 蓝线
初版只有 rank0 提交。

## 7. Prefix lookup 与 load path

### 7.0 当前实施顺序

为先闭合 durable source 生命周期，代码实施顺序调整为：

1. **P1-W store-only**：同轮 projection 规划 durable blocks，TP rank0 在
   forward/dump 后记录预分配 event，后台执行 L2->L3 put；scheduler 通过
   `KVConnectorOutput` 接收显式结果，并在 L2 physical ID 复用前 flush 相关
   job。request finish 不等待。
2. **P1-R sync-read**：write-behind 稳定后，再开放 L2 hit 之后的 L3 suffix
   lookup、L2/HBM 联合 admission、all-rank `L3->local L2` 与既有
   `L2->HBM` restore。

两阶段共用 `DSAL3Connector`、key/schema 与 completion 通道，不建立临时旁路。

### 7.1 第一版顺序

1. DSA `get_computed_blocks()` 查询 L2 完整 bundle 的最长连续 prefix。
2. vLLM 基线继续调用 Connector，传入 L2 hit token count。
3. Connector 只查询 L2 之后的 L3 suffix。
4. Indexer/MLA L3 component hit 取公共连续前缀。
5. scheduler 为 L3-only suffix 分配完整 Indexer+MLA L2 bundle，并对
   `L2 hit + L3-only hit + 本轮 compute` 一次完成 HBM admission。
6. Connector metadata 携带 L3 component keys 与每 rank local L2 destination。
7. all-rank worker 在 forward 前用 Store-owned TCP 将 L3 suffix 写入各自 L2。
8. L3 get 全部成功后发布本轮 L2 ready watermark，并执行既有 L2->HBM restore。
9. forward 只计算未命中部分；已恢复 suffix 不重复执行 HBM->L2 dump。
10. 任一 rank/component load 失败则 fail-fast，不发布 L2 hash。

### 7.2 为什么 TCP 第一版选择 L3->L2->HBM

优点：

- 对齐已通过的 Store-owned TCP swapped Host 数据面；
- 对齐 LMCache 的“注册固定 L2，get into 预分配 L2”模式；
- 不要求 TCP transport 识别或直接写 NPU HBM；
- L3 suffix 在 forward 前已经成为完整 L2 bundle，可直接供后续 KSC、prefix
  reuse 与 L2 eviction 使用；
- 只执行必要的 L2->HBM restore，不再在 forward 后增加 HBM->L2 回写。

代价：

- scheduler 必须联合预留 L2 bundle 与 split HBM blocks；
- pending L3 load 期间 L2 destination 不可被复用；
- all-rank get 完成后才能原子发布 Indexer+MLA 公共前缀；
- 初版必须 fail-fast，不能静默 recompute。

### 7.3 L3->HBM direct 后续路径

AscendDirect 修复跨 NUMA Host H2H、或未来后端能可靠直接写 split HBM 后，可在
不改变 lookup/schema 的前提下增加 direct destination capability。它必须复用同一
scheduler source plan，不能另建一套 eager/graph metadata。该优化保留给：

- 不要求 L2 常驻副本的 P/D consumer；
- 大容量只读 prefix；
- 已证明 direct load 后无需立刻物化 L2 的场景。

## 8. DSA allocator 改造

当前 `allocate_dsa_slots()` 显式拒绝：

```text
num_external_computed_tokens
delay_cache_blocks
reserved_blocks
```

P1 只开放同步 L3 load：

- `num_external_computed_tokens > 0`；
- `delay_cache_blocks=False`；
- `reserved_blocks=0`；
- `load_async=False`。

planner 的已计算长度必须改为：

```text
request.num_computed_tokens
+ local_l2_hit_tokens
+ external_l3_hit_tokens
```

HBM target block 分配覆盖全部 local+external prefix；local 部分由 DSA restore
写入，external 部分由 Connector 写入。不能调用基线
`allocate_new_computed_blocks()` 并假设 local hit 已经是 HBM block，因为 DSA
local target hit 实际来自 L2。

admission 必须在任何 pool 修改前同时检查：

- HBM Indexer；
- HBM resident MLA；
- 当前请求后续 full sequence reservation；
- 可选 MTP HBM；
- L2 logical directory 容量。

## 9. 异步 L2->L3 write-behind

### 9.1 触发点

scheduler 在构造本轮 DSA projection 时已经知道 forward 成功后将推进到的
durable full-block watermark。它可以生成本轮尚未远端发布的：

```text
logical block index
L2 physical block id
block hash
INDEXER/MLA component key
job id
```

任务随同一次 forward 的 Connector metadata 下发。worker 不得在 dump 之前
读取 L2。

这里存在一个明确的基线时序差异：vLLM `Scheduler.schedule()` 会在
`_update_after_schedule()` 和返回之前调用 Connector `build_connector_meta()`；
当前 DSA DRAM assignment/projection 则在 `super().schedule()` 返回后，由
`DSAOffloadScheduler._attach_dsa_cache_layout()` 生成。普通 Connector metadata
因此天然看不到同轮新增的 L2 physical ids。

第一版应增加一个只对 DSA connector 生效的类型化后置钩子：

```text
base Scheduler.schedule()
    -> base connector metadata
    -> DSA projection / DRAM assignment
    -> dsa_connector.extend_store_jobs(scheduler_output, projection)
    -> return final SchedulerOutput
```

该钩子同时把新分配/复用的 L2 physical ids 交给 reuse-fence 逻辑。不能只在
下一轮普通 `build_connector_meta()` 中扫描，因为最后一个 forward 之后可能只有
zero-token control step；也不能让 worker 根据 row 长度自行推导 store job。

### 9.2 worker 提交时序

```text
本轮所有 layer L1->L2 dump 入同一 NPU stream
    -> forward 结束
    -> Connector get_finished()/post-forward 记录或继承 dump-ready event
    -> rank0 把 store job 放入本地 pending queue
    -> 下一 engine step 的 start/handle hook 提交后台 put
```

这复用 vLLM `OffloadingConnector` 的 store deferral：上传在下一步开始，而不是
当前 token generation 的尾部同步 `join()`。

### 9.3 block reuse fence

复核 vLLM 基线后，第一选择不是长期增加 transfer ref，而是复用
`_block_id_to_pending_jobs + jobs_to_flush`：

1. pending job 记录它正在读取的 L2 physical block ids；
2. request finish 可以正常释放自己的 L2 refs；
3. block 可进入 L2 cached free queue，不长期减少可用容量；
4. scheduler 若再次分配同一个 physical id，先把关联 job id 放入
   `jobs_to_flush`；
5. worker 在应用新 assignment、执行任何 overwrite 之前等待这些 job 完成；
6. put 完成后再允许新请求覆盖该 L2 block。

该机制仅在真实复用压力出现时阻塞，比 request-finish 全量等待和永久 pin 更适合
在线吞吐。作为保守回退，也可以对小规模 P0/P1 使用 transfer ref pin；正式
P2 以 reuse fence 为目标。

### 9.4 request finish

request finish 不调用 `join()`，客户端响应不等待 L3。pending job 只保留：

- job metadata；
- old hash/component keys；
- source physical ids；
- reuse fence。

请求对象不再是 L3 object 的身份。所有请求完成但仍有 pending job 时，Connector
通过 `has_pending_transfers()` 让 `DSAOffloadScheduler.has_requests()` 保持为真，
复用 vLLM 的 zero-token connector progress step 收集 completion。客户端响应不
等待该 control step；但 EngineCore shutdown 必须等 pending job 清零后，才能
关闭各 TP rank 贡献的 Mooncake segment，避免 rank0 put 访问已退出的远端
segment owner。vLLM 外层 process manager 默认只等待 5 秒，无法覆盖 Mooncake
首次跨 segment 懒建链，因此 DSA L3 要求配置 `shutdown_timeout>0`；示例使用
300 秒上限，稳态通常会更早退出。vLLM 离线 `LLM` 的 finalizer 当前不向
MPClient 传递该 timeout，DSA 离线 demo 需显式调用 engine-core shutdown；在线
launcher 已按基线传递该配置。

## 10. job 与状态机

### 10.1 scheduler job

```text
job_id
direction: LOAD / STORE
request_id（诊断与 load resume 使用）
block_hashes
components
source/destination block ids
expected_worker_acks
status
bytes
```

store job 的 `expected_worker_acks=1`，因为只有 rank0 写。load job 若走所有 TP
worker，则 expected count 等于参与 load 的本地 worker 数。

### 10.2 状态

```text
LOCAL_READY
  -> STORE_PLANNED
  -> STORE_SUBMITTED
  -> REMOTE_READY
  -> STORE_FAILED

REMOTE_READY
  -> LOAD_PLANNED
  -> LOAD_SUBMITTED
  -> LOAD_COMPLETE
  -> LOAD_FAILED
```

远端状态以 `(schema namespace, component, block hash)` 为身份；在飞 source
保护以 `(physical block id, generation/hash, job id)` 为身份。迟到事件必须同时
匹配 job id 和旧 hash，不能只凭可复用的 block id 更新状态。

### 10.3 完成回报

worker 后台线程完成阻塞式 Backend `put/get` 后，将结果写入本地线程安全队列。
worker 主线程在正常或 zero-token Connector `get_finished()` 中清空队列，生成
`KVConnectorOutput.kv_connector_worker_meta`。scheduler 在
`connector.update_connector_output()` 中消费。

P1-W 验收期间 scheduler 与 TP0 worker 分别维护 planned/submitted/finished/
failed/bytes 计数，并在 shutdown 打印汇总。普通 poll 与 reuse/shutdown flush
必须共用同一 completion 记录函数，避免“future 已完成但无 completed 日志”。
scheduler shutdown 若仍有 unacknowledged job 必须失败，不能清空账本后静默退出。
DSA L3 模块复用 `vllm.logger.logger`，与当前大多数 vLLM-Ascend 热路径一致；不能
使用只落到 `vllm_ascend` file handler 的独立 logger 后再凭 console 判断状态。

不创建 scheduler 接收线程；即使未来增加 IPC receiver，receiver 也只能入队，
不能直接修改 BlockPool。

### 10.4 当前基线缺口

现有 AscendStore/Offloading 代码有以下不能原样沿用的行为：

- AscendStore `completed_events` 只有计数，没有 success/error；
- AscendStore sending thread 在 `finally` 中也标记 event 完成；
- Mooncake/MemCache `Backend.put()` 当前不返回可靠逐 key 结果；
- upstream `OffloadingConnectorWorker` 当前对 `TransferResult.success=False`
  直接 assert；
- `pending_count` 默认使用 world size，不适合 rank0-only store；
- AscendStore 非 layerwise `wait_for_save()` 会 `request_queue.join()`。

DSA connector 必须补充显式 `JobResult(success, failed_components,
error_codes)`，不得把“线程结束”当成“远端发布成功”。

## 11. worker 地址映射与 buffer registration

### 11.1 DSA L2 component view

`DSAWorkerDRAMStore` 需要新增只读 API，按 L2 physical ids 生成：

```text
Indexer: keys, list[list[address]], list[list[size]]
MLA:     keys, list[list[address]], list[list[size]]
```

地址来自初始化期固定 arena；热路径不 `.contiguous()`、不创建 tensor、不重新
组织 payload。

### 11.2 registration 顺序问题

Mooncake `GlobalTE.register_buffer()` 已从单个 `is_register_buffer` 改为增量
interval 账本：

- 维护已注册 interval 集合，而不是全局 bool；
- 支持 HBM/L2 分阶段增量注册；
- 合并同 storage 连续区间；
- 重复区间幂等；
- partial failure 回滚或明确失败；
- shutdown 时 unregister/close。

GLM-5.2 A5 L2 当前约有 78 个 MLA arena 和 42 个 Indexer plane arena；P0
必须验证约 120 个 multi-buffer component 的 registration 数量与资源限制。

### 11.3 swapped-memory P0

当前结果：

以下单卡项已经通过：swapped arena registration、swapped put/get、NPU 回读、
逐字节一致、HBM+swapped 同引擎注册。仍待验证 TP8 rank0 put/all-rank get 与
正式 connector shutdown/drain。

固定 device staging ring 仅保留为能力定位手段，不进入正式 DSA 设计。若未来
某个 L3 backend 在其内部透明使用 staging，它仍必须满足框架可见的 Host H2H
合同和性能门槛，DSA 不自行维护 ring slot 或拓扑路由。

## 12. 各模型阶段

### 12.1 prefill

- 每个 chunk 都可能产生新 durable 满块；
- L1->L2 继续使用当前同步 full-block dump；
- 本轮新增 ready block 生成异步 L3 store job；
- L3 external hit 只发生在请求第一次进入 scheduler 的 prefix lookup；
- chunked prefill 后续 chunk 不重复 lookup 已提交 prefix。

### 12.2 dense decode

- 新满块与 prefill 使用同一 dump/job 语义；
- HBM resident block 尚有 dense 语义，但第一版不作为 target HBM prefix tier；
- L3 store source 始终取 L2，不依赖 HBM 请求布局。

### 12.3 sparse decode

- KSC 只执行 token-wise L2->resident-HBM miss copy；
- sparse budget block 不进入 L3；
- 新 durable tail 满块先同步写完整 L2 bundle，再异步上传 L3；
- active request 持有 L2 refs，因此 KSC 不会读到被重分配的 block；
- L3 不参与 decode 热路径的 token-wise miss。

### 12.4 compromise MTP

初版 DSA+L3 显式拒绝 MTP。后续开放时：

- target 只有 verified token 可以推进 L2/L3 watermark；
- provisional 双尾块不能上传；
- MTP BF16 full cache 仍只在 HBM；
- target L2/L3 hit 与 MTP HBM hit 取公共前缀；
- terminal 未 flush 满块只能降低命中长度，不能发布错误 key。

## 13. eager 与 graph

L3 lookup、load/store metadata、buffer registration 和 completion 都位于 host
控制面。L3 load 和 L2 restore 都在 model forward/graph replay 前完成，forward
只读取最终 HBM block table。

必须保持：

- 一个 `SchedulerOutput`/Connector metadata 真源；
- worker 先执行基线 `_update_states()`，再按最终行序应用 DSA projection；
- eager/graph 不维护两份 L3 row metadata；
- background store job 不进入 captured tensor；
- zero-token completion 不运行模型图；
- graph PAD 行不创建真实 L2/L3 job。

## 14. failure policy

### 14.1 load

第一版固定 `kv_load_failure_policy=fail`。任一 rank、任一 Indexer/MLA
component load 失败：

- 不进入 forward；
- 不发布 L2 ready/hash；
- 释放本轮 HBM/L2 admission；
- 返回明确请求错误或终止当前测试；
- 不使用裸 `invalid_block_ids`，因为 DSA 多物理 pool 重用 block id。

后续若支持 recompute，必须使用 group/tier-qualified failure identity 或 request
级失败结果，不能广播裸 block id。

### 14.2 store

store 失败不影响已经完成的推理：

- 对应 component 不标 remote-ready；
- reuse fence 解除；
- L2 副本若尚在，可按限次策略重试；
- L2 已复用则放弃该次 L3 promotion；
- 不允许部分 Indexer/MLA object 构成完整 prefix hit。

### 14.3 backend 原子可见性

worker `put()` 成功回报是 source-lifetime 完成依据。未来 prefix lookup 仍以
Backend `exists/lookup` 为远端可用真值。debug 模式可在 worker success 后做
二次 exists，但 scheduler 不轮询 L3 来判断 source 是否可复用。

## 15. L2 与 L3 淘汰

### 15.1 L2

L2 继续使用 vLLM BlockPool 全局 LRU。request free 使用 reverse-free 保留
tail-first eviction bias，但不承诺严格按每个请求尾部淘汰。

正确性来自：

- prefix lookup 从 logical block 0 开始；
- 遇到第一个 miss 停止；
- pending store source 在 overwrite 前执行 reuse fence；
- active request ref 阻止 KSC source 被复用。

### 15.2 L3

Mooncake/MemCache 自主管理容量和淘汰。DSA 不修改后端源码，也不假设它按请求
尾部淘汰。Indexer/MLA component 分别 lookup，并从 block 0 求公共连续 prefix。

未来可增加 retention/TTL/parent-hash hint，但不是第一版正确性条件。

## 16. 背压与性能

配置建议：

```text
max_inflight_l3_store_jobs
max_inflight_l3_store_bytes
max_descriptors_per_l3_put
l3_completion_poll_interval
l3_connection_warmup
```

达到上限时不得阻塞 decode：

- 暂缓新的 L3 promotion；
- 保留普通 L2 cache；
- 若 block 后续仍在且队列有空位再补传；
- L2 即将复用时只 flush 真正相关 job。

已观察到 Mooncake 第一次 warm load 因 segment 懒建链耗时 10~24s，连接复用后
各 rank get 为约 17~327ms。正式性能测试前需要连接预热，不能把首轮建链混入
稳态 L3 带宽。

## 17. Backend 抽象

DSA connector 不直接 import Mooncake/MemCache API，复用 vLLM-Ascend
AscendStore `Backend`：

```text
exists(keys)
register_buffer(ptrs, sizes)
put(keys, addrs, sizes)
get(keys, addrs, sizes)
shutdown/close/drain
```

### 17.1 LMCache Mooncake 对照

LMCache 主线的 `MooncakestoreConnector` 为 DSA 的 L2/L3 边界提供了直接参考：

- 配置默认 `protocol=tcp`；
- 调用 `MooncakeDistributedStore.setup()` 时不注入外部 TransferEngine，由 Store
  自己创建对应协议的 engine；
- 初始化期把 LocalCPUBackend allocator 的整块固定 CPU buffer 通过
  `store.register_buffer()` 注册；
- put 直接使用该 L2 MemoryObj 的地址，get 先从 LocalCPUBackend 分配目标对象，
  再用 `batch_get_into()` 写回同一注册 arena；
- connector 关闭时先 unregister，再关闭 Store；
- NUMA 绑定发生在 Store setup 之前，但 block 分配、引用计数和淘汰仍由
  LocalCPUBackend 管理，Mooncake 不接管 L2 ownership。

因此“把 DSA L2 地址注册给 Mooncake”不构成层级耦合。注册仅是 transport 的
memory-region 合同；DSA scheduler 继续独占 L2 block 生命周期，Mooncake 只拥有
自己的 L3 segment 和 object。Store-owned TCP 真实介质门禁已经通过：原始
`mooncake-transfer-engine-npu==0.3.11.post1` 可直接注册 swapped L2，并在
NPU0↔1、跨 NUMA NPU0↔6 两组 worker 间完成双向逐字节往返。因此 TCP 首版确定为
`TP0 L2 -> L3` 写入、`L3 -> all-rank local L2 -> HBM` 读取。它仍需与本文前述
AscendDirect `L3->HBM` 方案统一收敛，不能同时维护两套 scheduler/read metadata。
无论最终选择哪条数据面，schema、job、reuse fence 与 L2 ownership 均不得因
transport 分叉。

TP8 放大门禁也已经通过：4-plane/1 MiB 与 120-plane/6,904,320-byte 两组均由
8 个 rank 同时贡献 segment、注册 source/destination L2 regions，并完成环形
remote get。每 rank 结果均为 put=`[0]`、get 等于完整 payload bytes。该结果
覆盖进程数、region 数与单逻辑块 descriptor/payload，但测试 arena 只有 2 个
physical blocks；正式 scheduler 规划的几十 GiB L2 MR 注册仍属于框架 E2E 门禁。

当前 vLLM-Ascend `MooncakeBackend` 仍拒绝非 `ascend` 配置，并向 Store 注入由
`GlobalTE` 以 `ascend` 初始化的外部 engine。这是 Store-owned TCP 不可达的框架
约束，不是 wheel 能力缺失。TCP 模式必须让 Store 自建 engine，并把
register/unregister 转发给 Store；Ascend 模式继续复用现有外部 `GlobalTE`。

NPU wheel 的 TransferEngine 会同时安装 AscendDirect 和 TCP，且普通 memory
registration 会遍历全部已安装 transport，所以 TCP 日志仍可能出现 HIXL 初始化和
Host registration。实际 payload transport 由目标 segment descriptor 的
`protocol` 决定；TCP cross 门禁已证明跨 NUMA payload 不经过 HComm。首版可接受
未使用 AscendDirect 的初始化开销，后续再评估 TCP-only wheel 或 Mooncake 上游
按需安装 transport，不能为消除日志而修改 DSA scheduler。

需要统一返回合同：

```text
len(result) == len(keys)
0           == success
nonzero     == explicit failure
exception   == transport/backend failure
```

Mooncake multi-buffer 调用仍受 descriptor 数量和单次同步时延影响。A5 GLM-5.2
一个逻辑块当前包含 42 个 Indexer plane 与 78 个 MLA plane；若把 64 个块一次
交给 `batch_put_from_multi_buffers()`，会形成 7,680 个 descriptor。
`DSAL3BackendAdapter` 因此按 `max_descriptors_per_l3_put` 对 key 批次做贪心切分，
默认预算 128。Store-owned TCP 的 TP8 120-plane 单块门禁已通过；多块合批上限
仍应由 P1-W profiling 决定。该切分只改变
一次 backend 调用包含多少 key，不改变 L3 object key、component schema、job
完成语义或 scheduler 元数据。该 128 上限也与 Mooncake HiCache 文档的经验
batch page 上限一致。仅 AscendDirect 模式补齐 `ASCEND_CONNECT_TIMEOUT` 与
`ASCEND_TRANSFER_TIMEOUT`；TCP 不设置 HComm/HIXL timeout。

该预算是跨 key 合批的软限制，不是模型结构上限。若未来单个 Indexer 或 MLA
component 自身超过 128 个 descriptor，它仍保持一个完整 Mooncake key，由
Mooncake transport 在该 key 内部分 chunk；backend 只将其隔离成 single-key
put 并告警。scheduler
始终只接收逻辑 component/job 的最终完成状态，不跟踪 HIXL packet。

Mooncake 优先实现。MemCache 后续复用相同 DSA key/schema/job 状态机，只保留：

- Local EID 初始化；
- AICPU run 包检查；
- Backend put/get 实现；
- lease/GVA 特有能力（第一版 non-layerwise 不依赖）。

## 18. HBM target prefix 高优先级后续

该适配点不能遗忘。L3 第一版完成后，应尽快把 HBM target 热副本加入
component-wise source plan：

```text
Indexer: HBM -> L2 -> L3
MLA:     dense HBM -> L2 -> L3
MTP:     HBM
```

必要改造：

1. Indexer HBM pool 开启 hash；
2. resident MLA block 增加 `DENSE_PREFIX/BUDGET/TAIL` role；
3. 只给 `DENSE_PREFIX` 发布 hash；
4. prefix lookup 按 logical block 和 component 选最热 source；
5. Indexer/MLA source 长度允许不同，但最终必须覆盖同一连续 token prefix；
6. restore plan 拆成独立 component source/destination/mask；
7. eviction 与 invalid-block identity 使用 group-qualified id；
8. 保持 worker 最终 HBM block table 为 eager/graph 唯一真源。

该项优先级定为 **P3-HOT（高）**，排在 L3 正确性闭环与 rank0 async store
之后，不归入低优先级远期优化。

## 19. 配置与初版支持矩阵

建议沿用标准 `KVTransferConfig`，使用能力命名的 Connector，例如：

```python
KVTransferConfig(
    kv_connector="DSAL3Connector",
    kv_role="kv_producer",  # P1-W；开放 load 后改为 kv_both
    kv_load_failure_policy="fail",
    kv_connector_extra_config={
        "backend": "mooncake",
        "max_inflight_l3_store_jobs": 8,
        "max_active_channels": 8,
        "max_descriptors_per_l3_put": 128,
    },
)
```

DSA 配置校验第一版要求：

| 能力 | 第一版 |
|---|---|
| 单机 TP | 支持 |
| GLM-5.1/5.2 A5 C8 | 首要 |
| DeepSeek-V3.2 | 回归 |
| Mooncake | 支持 |
| MemCache | 后续同 schema |
| eager | 支持 |
| FULL graph | 支持，IO 在图外 |
| chunked prefill | 支持 |
| async scheduling | 拒绝 |
| MTP | 拒绝 |
| preemption | 拒绝 |
| PP/DCP/PCP | 拒绝 |
| load failure recompute | 拒绝 |

当前 `dsa_offload/config.py` 对所有 KV connector 的无条件拒绝，应改成只允许
经过验证的 DSA connector，而不是笼统放开 AscendStoreConnector。

## 20. 计划代码结构

建议新增：

```text
vllm_ascend/dsa_offload/l3/
    __init__.py
    connector.py          标准 KVConnector 生命周期入口
    scheduler.py          lookup、job、reuse fence、completion
    worker.py             rank0 uploader、全 rank loader、completion queue
    metadata.py           pickle-safe job/result dataclass
    schema.py             component/key/schema fingerprint
    backend_adapter.py    对现有 AscendStore Backend 做严格返回封装
```

现有文件窄改：

| 文件 | 改造 |
|---|---|
| `dsa_offload/config.py` | 支持矩阵与 Connector 校验 |
| `kv_cache_manager.py` | 接受 external tokens，component admission |
| `kv_cache_coordinator.py` | local hit commit、L2 assignment、未来 HBM source |
| `scheduler.py` | 将新增/复用 L2 IDs 提供给 connector fence |
| `worker_dram_store.py` | component 地址视图和 registration regions |
| `model_runner_v1.py` | bind DSA L3 worker/store，保持基线 connector hooks |
| `mooncake_backend.py` | 严格 put/get 结果、close |
| `memcache_backend.py` | 同一结果合同、close |
| `mooncake_transfer_engine.py` | 增量注册与 unregister |

不新增模型名特化模块，不修改 vLLM 源码，不复制 Scheduler 主循环。

## 21. 分阶段实施

### P0：介质与 schema

- exact 6,904,320-byte DSA block；
- split HBM TP8 round trip；
- swapped L2 put/get；
- HBM+L2 同进程增量注册；
- rank0 put/all-rank get；
- strict status 与 teardown。

### P1-W：rank0 异步 write-behind

- Store-owned TCP engine 与 per-rank segment identity；
- TCP L2 registration 转发给 Store，Ascend 路径继续使用 GlobalTE；
- all-rank Mooncake segment contribution；
- store job 和 dump-ready event；
- rank0-only Mooncake put；
- reuse fence；
- request finish 不等待；
- pending store 通过 zero-token control step 推进；
- EngineCore shutdown 在 all-rank segment teardown 前完成 drain；
- 显式 success/error result；
- prefill/chunk/decode durable block。

当前代码项与服务器 UT 已完成。2026-08-29 的 TP8 离线 eager 单 40K 验收中，
TP0 成功注册 120 个 LocalDRAM region、30,054,504,960 bytes（约 27.99 GiB），
scheduler 按 `64 + 64 + 64 + 64 + 64 + 5` 增量规划 325 个 durable block；
TP0 提交并完成 6 个任务，累计 2,243,904,000 bytes，失败数和残留任务均为 0。
scheduler 最终发布 650 个 remote-ready key，恰好等于 325 个 block 的
Indexer/MLA 两个 component。其他 TP rank 的提交数均为 0，符合 rank0-only put。

该结果关闭了“真实几十 GiB L2 MR + 单请求离线 eager/graph + chunked prefill
write-behind”门禁：`max_num_batched_tokens=8192`、`block_size=128`，所以前五个
chunk 各推进 64 个 durable block，最后推进 5 个，且没有重复上传已经发布的块。
graph 复测的 block/job/key/byte 计数与 eager 完全一致，输出 token 也逐个一致；
真实请求阶段出现 ACL graph replay，未因 PAD 或 replay 额外创建 store job，8 个
worker 均正常退出。

混合长度 graph/chunked 复测最终 durable watermark 为 `(60, 39, 574, 325)`，
合计 998 个逻辑满块；其中 5K 请求在 decode 阶段跨过块边界，新增的第 39 块被
独立规划。scheduler 按内容寻址规划 944 个唯一块、1,888 个 component key、
6,517,678,080 bytes（约 6.07 GiB），相对逻辑块少 54 个，符合请求间公共
block hash 复用。16 个异步任务全部成功，最后两个任务在请求输出后由 shutdown
drain 完成，未阻塞请求 finish；所有 worker 正常退出。该轮 `planner_calls=1856`
暴露出等待异步 IO 时存在较多 zero-token polling，作为 P1-W 正确性关闭后的性能
收敛项，不影响本轮数据闭环结论。

在线 TP8 graph 服务使用 `max_num_batched_tokens=16384`，40K 请求的 prefill
按 `128 + 128 + 69` 规划 325 个块；生成阶段又形成第 326、327 个 durable 满块。
5 个 job 全部返回 `success=True`，累计 327 blocks、654 component keys、
2,257,712,640 bytes。ACL graph replay、流式 HTTP 服务与异步 write-behind 并行
工作正常；服务持续运行，因此该轮没有 shutdown summary，外部 prefix hit 为 0
符合 producer-only 边界。由此关闭 P1-W 首版正确性阶段，在线多请求 continuous
batching 与长稳转入 P2 统一回归。

已知但不阻塞关闭的事项：eager 轮退出时曾出现 6 个 worker 超过
executor grace period 后被发送 SIGTERM，但该提示发生在 6 个 store job 全部完成、
各 rank summary 输出及 Mooncake segment 解注册之后；暂归类为 teardown 时延问题，
不计作数据写失败，后续长稳测试继续观察。混合请求轮的 `planner_calls=1856`
表明 zero-token polling 尚有性能收敛空间，同样不影响 P1-W 正确性结论。

### P1-R：同步只读

- DSA connector 注册；
- L2 hit 后 L3 suffix lookup；
- external L2/HBM 联合 admission；
- all-rank `L3 -> local L2` Store get；
- 复用现有 L2->HBM restore 后进入 forward；
- no-MTP、fail-fast、Mooncake only。

### P2：读写闭环与稳态

- load/store completion 联合 drain；
- L3 object lookup 与连续公共前缀；
- store/load failure policy；
- TP8 rank0 store/all-rank load；
- 在线 continuous batching 与长稳。

### P3-HOT：HBM target prefix

- component-wise HBM/L2/L3 source；
- dense MLA role；
- independent restore masks；
- LRU/eviction 性能。

### P4：扩展

- MTP target+MTP 公共前缀；
- preemption/resume；
- P/D；
- 多节点与 CP/PP；
- MemCache 正式版；
- L3 event/metrics/retention。

## 22. 测试门禁

### 22.1 Python UT

- key/schema 稳定与 namespace 隔离；
- Indexer/MLA component 公共前缀；
- L2 hit + L3 suffix；
- L3 hole/partial component；
- external token allocator；
- rank0 store expected count；
- success/failure completion；
- request finish 不等待；
- physical id reuse 触发 flush；
- stale job/hash generation 防护；
- zero-token completion；
- backpressure；
- shutdown/reset。

### 22.2 NPU component

- realistic HBM/L2 schema bitwise；
- TP8 rank0 put/all-rank get；
- same-process HBM+L2 registration；
- empty/error return；
- first connection 与 steady bandwidth；
- in-flight reuse fence。

### 22.3 E2E

1. DSA disabled control；
2. DSA L2 only control；
3. DSA+Mooncake，清 L2/缩小 L2 强制 L3 hit；
4. eager/graph；
5. bsz=1/bsz>1；
6. 相同/混合长度；
7. chunked prefill；
8. prefill 新块与 decode 新块远端增长；
9. online continuous batching；
10. request finish/row reuse/L2 physical id reuse；
11. output token 与 top-logprobs 对照；
12. L3 put/get/object 数和 external hit 证据；
13. profiling 与长稳。

GLM-5.2 当前冷算与 HBM/L3 warm prefix 路径存在相同的 top-token 边界变化；
L3 验收应继续对照 HBM-prefix warm 结果，不能只比较 cold/warm token 是否完全相同。

## 23. 已识别风险

| 风险 | 当前处理 |
|---|---|
| AscendDirect swapped H2H 跨 NUMA 失败 | P1-W 改用已通过 TP8 的 Store-owned TCP；保留 direct 结果作历史门禁 |
| 正式几十 GiB L2 MR 尚未以 TCP 注册 | cache-init/P1-W E2E 必测，独立 2-block probe 不替代 |
| Backend put 吞错误 | 严格结果 adapter |
| completion 把异常当成功 | 显式 JobResult |
| physical id 在上传中复用 | upstream-style jobs_to_flush fence |
| base Connector metadata 早于 DSA projection | DSA 类型化后置 metadata 扩展钩子 |
| split pool 裸 block id 冲突 | job/component/group-qualified identity |
| rank0-only store 误按 world size 等待 | per-job expected ack count |
| DSA target key 与基线 key 冲突 | schema namespace/fingerprint |
| 首次 Mooncake 建链过慢 | connection warmup 与稳态指标分离 |
| Connector teardown 泄漏 | stop/join/unregister/close/drain |
| sparse resident 被通用 save 错读 | store source 固定为 DSA L2 component view |
| MTP provisional block 提前发布 | durable watermark + 初版拒绝 MTP |
| eager/graph 元数据分裂 | 所有 IO 在图外，共用 scheduler plan/HBM table |

## 24. 决策摘要

1. L2 block id 继续绑定完整 Indexer+MLA bundle。
2. L3 按 Indexer/MLA component 分对象，以 hash 为身份。
3. TCP 第一版 read 为 L2 hit + `L3-only suffix -> local L2 -> HBM`。
4. L3 suffix 在 forward 前已物化到 L2，不再执行一次 HBM->L2 回写。
5. L2->L3 只由 rank0 异步写。
6. 不在 request finish 等待。
7. 优先复用 vLLM OffloadingConnector job/reuse-fence/no-forward 模型。
8. worker 完成事件是 source lifetime 真值；scheduler 不轮询 L3 判断完成。
9. L3 eviction 由后端自治，DSA 只做连续公共前缀。
10. HBM target prefix 是 P3-HOT 高优先级项，不得遗忘。
