# L3 当前测试清单

本文记录 2026-08-25 起基于以下环境继续执行的最短 L3 门禁：

- 单机 8 x Ascend 950DT；
- CANN 9.1.0；
- PyTorch 2.10.0；
- `torch_npu==2.10.0.post6`，`git_version=a8f2f4fda83732c0d02244a46815a81f21fe3439`；
- vLLM-Ascend 当前 DSA custom-official 分支。

社区原生 `AscendStoreConnector` 与 MemCache/Mooncake 的基础链路、DSA
Store-owned TCP 单卡/跨 NUMA/TP8 精确介质门禁已经通过。当前进入 P1-W
`L2 -> L3` TCP 框架级 write-behind 验收。

## 当前结果

- [x] 环境诊断：TorchNPU post6 / `a8f2f4fd` 生效，NPU 0、1 健康且空闲。
- [x] MemCache capabilities、Host、AICPU、单卡 NPU 与 GVA/LAYERWISE API。
- [x] 真实 `MemcacheBackend` scheduler/worker 单卡跨进程往返。
- [x] Mooncake runtime import、单卡 NPU multi-buffer 往返。
- [x] 真实 `MooncakeBackend` scheduler/worker 单卡跨进程往返。
- [x] MemCache NPU 0、1 跨 rank 边界已定位：`device_urma` 固定使用 rootinfo
  中的出框 EID，当前实现没有同机 peer 的内部 fullmesh 选路；该项不是通过，
  而是确认不适用于当前单机环境。
- [x] MemFabric 临时包支持按进程设置 `USE_LOCAL_EID`；8 个物理 NPU 使用
  ping 筛选出的内部 fullmesh EID 后，完整 mesh 的 `store.init` 全部返回 0。
- [x] MemCache 临时包 TP8 真实数据面：逐 rank local put、环形 remote get 和
  1 MiB payload 逐字节校验通过，最终 gate 退出码为 0。
- [x] Mooncake NPU 0、1 跨 segment 门禁：`global_ub_device` 双向 1 MiB
  remote get 与内容校验通过。
- [x] Mooncake NPU 0-3 四卡轻量门禁。
- [x] Mooncake TP8 轻量门禁：`global_ub_device` 下 8 个 rank 双向环形
  remote get 与内容校验通过。共享节点前一轮的 AICPU/HComm 初始化失败确认为
  瞬时资源竞争，不计作功能失败。
- [x] DSA-disabled MemCache 完整模型：3200/3287 tokens external hit，热重放
  token/logprobs 稳定且与 HBM prefix 热路径完全一致；仍保留 GLM-5.2 冷/热
  prefix 路径输出差异作为独立精度问题。
- [x] DSA-disabled Mooncake 完整模型：TP8 `global_ub_device` 下 3200/3287
  tokens external hit；两轮 warm replay 稳定。每轮 8 个 rank 均完成 25/25
  keys、`failed=0`，每 rank 实际恢复 304,742,400 bytes。首次 warm load 的
  10.7~23.8s 来自跨 segment 懒建链，连接复用后各 rank get 降至
  17~327ms。
- [x] DSA swapped smoke schema：单卡 `global_ub` 下 1 个 Indexer key plane
  和 1 个 packed MLA plane，source/destination 均由
  `empty_with_swapped_memory` 分配；注册 6 个 region 后，两个 component 的
  put/get、interior block 回读和 SHA-256 校验通过，gate `exit_code=0`。
- [x] DSA swapped exact schema：单卡 `global_ub` 下 42 个 Indexer plane 和
  78 个 packed MLA plane，完整逻辑块为 6,904,320 bytes；242 个 region、
  110,477,312 registered bytes 全部注册成功，put/get 分别返回
  `[0, 0]` 和 `[354816, 6549504]`，两个 component 的逐字节 digest 校验及
  gate `exit_code=0` 均通过。
- [x] 两卡 swapped Host 跨 segment 基础门禁：TP0 的 1 MiB swapped buffer
  强制定向写入 TP1 segment，`put=[0]`，随后跨 segment get 返回 1 MiB 并
  逐字节校验通过。Mooncake/HIXL 基础 H2H 池化能力成立。
- [x] 两卡 120-buffer、6,904,320-byte swapped scatter/gather 跨 segment
  门禁通过，`put=[0]`、get=6,904,320 bytes 且逐 buffer 校验一致。
- [x] 两卡 120-buffer、4,353-block 大 arena interior-address 门禁通过，真实
  DSA TP0 约 30 GiB 注册范围与非零 block offset 本身可用。
- [x] 在相同大 arena 条件下从独立 Python background thread 调用 Mooncake
  put/get；`put=[0]`、remote get=6,904,320 bytes、逐 buffer 校验和 gate
  `exit_code=0` 均通过，write-behind 线程设备/通信上下文本身可用。
- [x] 复刻真实 GLM-5.2 A5 payload：Indexer 42 个非等长 plane 与 MLA
  78 个 packed plane 分别组成两个 key，在相同 background/large-arena
  条件下完成跨 segment 往返；`put=[0,0]`、get 分别返回
  `[354816,6549504]`，逐 plane 校验通过。
- [x] 用完全相同的 payload 验证 NPU0→NPU4 跨 NUMA/卡组 preferred put：
  当前 `global_ub` swapped Host direct 路径失败。建链、segment mount、约
  30 GiB source registration 均成功，首个 120-descriptor put 在
  `HcommChannelFenceOnThread` 返回 `hccl_ret=19 (HCCL_E_NETWORK)`；错误与
  TP8、模型/HCCL 并发、Master 自动放置和 Host IP 无关。
- [x] NPU0→NPU4 基线式 HBM `global_ub_device` 对照通过：相同双 key、120
  plane 和 6,904,320-byte payload 的 put 返回 `[0,0]`，remote get 返回
  `[354816,6549504]`，逐 plane 校验及 gate `exit_code=0` 均通过。跨卡组
  Device→remote Host→Device 路径成立，失败边界仅位于 Host direct 路径。
- [x] staging 仅完成可行性诊断，不作为 DSA 正式 fallback：HBM 中转会额外
  消耗 H2D/D2H、跨卡 Device 带宽和每 rank HBM，偏离目标 Host H2H 数据面。
- [x] `global_ub + ASCEND_BUFFER_POOL=4:8` 负向门禁：该组合在初始化期被
  新版 `HixlEngine` 以 `OPTION_BUFFER_POOL ... only supports 0:0` 拒绝，尚未
  分配 staging 或执行 IO。带 `protocol_desc` 的 global 模式选择
  `HixlEngine`；buffer pool 必须先用 `auto` 进入兼容 `CommEngine` 验证。
- [x] `auto + BufferPool + HCCL_INTRA_ROCE_ENABLE=1` 在 NPU0→NPU1 和
  NPU0→NPU6 均于 HCCL communicator 初始化失败；生成的 v1 rank table 含
  两个 device，但缺少当前 HCCL 必需的顶层 `rank_count`，报错为
  `json object has no property called 'rank_count'`。这与 NUMA 无关，尚未执行
  staging 数据传输。
- [x] HIXL `rank_count` 缺失已定位但不在本项目修改边界内；不维护 HIXL 私有
  补丁，也不以修通旧 CommEngine staging 作为 DSA L3 交付前提。
- [x] `mooncake.engine.TransferEngine.initialize(protocol=tcp)` 的原始 NPU
  wheel 对照不可用：该入口仍因 auto-discovery 进入 AscendDirect/HIXL，不能
  作为 Store TCP 能力证据。Mooncake NPU release 中 `USE_TCP` 默认开启，且
  Store 自有 `Client::InitTransferEngine` 有显式 TCP 分支，因此不再把该结果
  扩大解释成“wheel 未编译 TCP”。
- [x] Store-owned TCP 原始 wheel 门禁通过：不创建/注入外部 TransferEngine，
  `MooncakeDistributedStore.setup(protocol=tcp)` 后通过 `store.register_buffer`
  注册真实 swapped L2。单 worker、NPU0↔1 和跨 NUMA NPU0↔6 的 4-plane/
  1 MiB 双向往返均逐字节通过，put=`[0]`、get=`[1048576]`。
- [x] 传输选路源码确认：NPU wheel 会额外安装 AscendDirect，普通
  `registerLocalMemory()` 也会遍历所有已安装 transport，因此日志仍出现 HIXL
  registration；但 mounted segment 的 descriptor 明确记录 `protocol=tcp`，
  `MultiTransport::selectTransport()` 按目标 segment protocol 选择 TcpTransport。
  上述 cross payload 不经过 HComm。原始 wheel 足以支撑 DSA TCP 首版。
- [x] Store-owned TCP TP8 门禁通过：4 plane/1 MiB 下 8 个进程全部完成
  并发初始化和环形 remote get；120 plane/6,904,320 bytes 下每个 rank 均返回
  put=`[0]`、get=`[6904320]` 并逐 plane 校验通过。每 rank 同时注册 120 source
  与 120 destination regions，未使用的 AscendDirect 没有在 TP8 初始化或
  region-count 放大下耗尽资源。该门禁 arena 仅含 2 blocks，正式大容量 L2 MR
  的字节规模仍由框架 E2E 验收。
- [x] P1-W Backend TCP gate：真实 `MooncakeBackend` worker 使用 Store-owned
  engine，scheduler-only client 使用独立 segment identity；put/get 和 teardown
  通过，Ascend 旧 gate 同时回归。
- [x] P1-W UT：`tests/ut/distributed/ascend_store/test_backend.py` 与
  `tests/ut/dsa_offload/test_l3.py` 全部通过，再跑 `tests/ut/dsa_offload`。
- [ ] P1-W cache-init：TP8 worker 全部贡献 segment，TP0 完整几十 GiB L2 arena
  通过 Store registration；日志不得显示 DSA 主动配置 global_ub。完整 E2E 已证明
  TP0 可注册 120 regions/30,054,504,960 bytes，独立 cache-init 模式尚未重跑。
- [x] P1-W 离线 eager：长 prompt 产生 durable 满块，日志依次出现唯一 TP0 的
  `store planned -> submitted -> completed success=True bytes>0`，请求输出正常且
  shutdown drain 完成。scheduler summary 必须满足
  `planned_jobs=completed_jobs`、`failed_jobs=0`、`inflight_jobs=0`；TP0 worker
  summary 必须满足 `submitted_jobs=finished_jobs`、`pending_results=0`。shutdown
  遇到未 ack job 会直接失败，不再静默清空。
  - 2026-08-29 TP8 单 40K：325 blocks，6 jobs，650 component keys，
    2,243,904,000 bytes；scheduler/worker 两侧计数与字节完全一致，0 failed、
    0 inflight、0 pending。TP1-TP7 均为 0 submitted，rank0-only put 成立。
  - durable watermark 按 64/128/192/256/320/325 递增，未重复规划旧 block。
    退出时 6 个 worker 超过 executor grace period，但发生在 store 闭环和 segment
    解注册之后；作为 teardown 时延项继续观察，不判定为本轮 P1-W 失败。
- [x] DSA L3 console 日志缺失已定位：vLLM-Ascend 先安装 file handler，后续
  console 配置因“已有 handler”提前返回，使 `init_logger(vllm_ascend.*)` 只写入
  `~/ascend/log/vllm_ascend`。DSA L3 已对齐仓内主流模块，改用
  `vllm.logger.logger`；历史运行可从该目录 grep，不需要仅为取日志重跑。
- [x] P1-W 单请求 chunked prefill：40K prompt、
  `max_num_batched_tokens=8192`、`block_size=128` 下，durable watermark 按
  64/128/192/256/320/325 推进；每个 chunk 的新增满块被增量规划，未重复上传
  已发布 key，6 个任务均在请求执行期间异步完成。
- [x] P1-W 多请求 chunked prefill：混合长度请求交错推进时，各 request watermark
  独立、无重复 key，且 finish/reuse fence 不会错误等待其他请求的任务。首个
  TP8 graph 用例采用 5K+8K+40K+70K、`max_num_batched_tokens=8192`；按既有
  tokenizer 长度预计 prompt 产生 997 个满块、1,994 个 component key、
  6,883,607,040 bytes。若每请求生成 32 tokens 且 5K 请求跨过下一个块边界，
  上界为 998 blocks、1,996 keys、6,890,511,360 bytes。DSA TCP 样例固定为
  4 GiB/rank、TP8 合计 32 GiB，不配置额外 L3 副本。
  - 实测最终 watermark 为 `(60, 39, 574, 325)`，合计 998 个逻辑块；5K 请求
    在 decode 中新增第 39 个满块。
  - 内容寻址后实际写入 944 个唯一块、1,888 keys、6,517,678,080 bytes；少掉的
    54 个逻辑块符合跨请求公共 block hash 去重。16 jobs 全部成功，0 failed、
    0 inflight、0 pending。
- [x] P1-W 单请求 graph：与 eager 使用相同 40K/chunked 配置，仍为 325 blocks、
  6 jobs、650 keys、2,243,904,000 bytes，0 failed/inflight/pending；输出 token 与
  eager 逐个一致。真实请求发生 ACL graph replay，PAD/capture/replay 未产生额外
  store job，8 个 worker 全部 graceful exit。
- [x] P1-W 多请求 graph：混合长度、动态 batch 行序和请求完成次序下维持相同
  job/key 状态机；ACL graph replay 未创建额外 store job，TP0-only put 与 8-rank
  graceful teardown 均成立。
- [x] P1-W 在线服务冒烟：TP8 graph、单 40K 流式请求、
  `max_num_batched_tokens=16384` 下，prefill 按 128/128/69 生成 325 个块，decode
  继续生成第 326、327 块；5 jobs、654 keys、2,257,712,640 bytes 全部成功。
  ACL graph replay 与服务持续运行正常；无 shutdown summary 属于常驻服务预期，
  `External prefix cache hit rate=0` 符合 producer-only 边界。
- [x] P1-W 首版正确性阶段关闭：可以开始 P1-R lookup/loadback。
- [x] P1-R UT：`test_l3.py` 覆盖 paired-key 最长前缀、L2+L3 metadata、
  all-rank worker get 与 component failure；`test_kv_cache_layout.py` 覆盖 external
  L2/HBM 联合 admission 和 combined restore；随后跑完整 `tests/ut/dsa_offload`。
- [x] P1-R cache-init：配置 `kv_role=kv_both`、`enable_prefix_caching=True`，确认
  scheduler-only Mooncake client 不贡献容量，所有 TP worker 注册本 rank 完整 L2
  arena；P1-W 的 all-rank segment/rank0-only put 保持不变。
- [ ] P1-R 离线 eager：先用单 8K 降低首轮调试成本，再扩到单 40K；均运行三轮
  相同 prompt：cold、local-L2 replay、L3 replay。第二轮后只执行
  `reset_prefix_cache(reset_connector=False)` 清本地 HBM/L2，第三轮必须出现
  `prefix lookup -> load planned -> TP0..TP7 get/load completed -> prefix restore`
  且 `external_tokens>0`。local-L2 与 L3 replay 的 greedy token IDs 必须一致；
  cold 与 prefix replay 仅记录差异，不作为 L3 IO 精度断言。
  - [x] 2026-08-29 单 8K eager：cold 为 0 hit，TP0 写入 60 blocks/120 keys/
    414,259,200 bytes；第二轮 Local-L2 命中 60 blocks；reset 后第三轮 L3 命中
    60 blocks、`external_tokens=7680`。TP0-TP7 均完成 414,259,200-byte get 和
    60-block/78-layer restore，Local-L2 与 L3 replay token IDs 完全一致。
    scheduler/worker 均为 0 failed/inflight/pending。
  - [ ] 单 40K eager。
- [x] P1-R 离线 graph：单 5K 三轮中 Local-L2/L3 对 38 blocks、
  262,364,160 bytes/rank 与 32 个生成 token 严格一致。混合 bsz=4 的 5K 请求
  在第 5 token 出现分叉，但关闭 L3 loadback 后 Local-L2/Local-L2 在相同位置
  复现同一分叉；启用/禁用 L3 的输出按 replay 轮次逐条相同，确认不是 L3
  数据差异。多请求 graph 门禁改用双方首个分歧 token 互在 top-k 的数值等价
  规则，bsz=1/eager 仍保持严格 token 一致。
- [ ] P1-R fail-fast：破坏任一 Indexer/MLA key 或让任一 rank get 失败，必须在
  TP success all-reduce 后让所有 rank 在 forward 前明确失败，不能发布 L2 hash、
  不能让成功 rank 提前进入 HCCL forward，也不能静默 recompute。
- [ ] P1-R 阶段关闭后进入 P2 读写闭环、online continuous batching 与长稳。
- [ ] P2-LAYERWISE：先复用基线 attention `wait_for_layer_load()` 将 L2->HBM
  restore 按层流水；确认收益后再评估 per-layer L3 key/schema，避免当前整组件
  Mooncake object 冒充逐层异步。GLM-5.2 shared Indexer 按 physical source layer
  映射。
- [ ] P2 在线 continuous batching 与长稳：在读写闭环后统一回归多请求进入/完成
  交错、store/load completion、zero-token polling 与 teardown 时延。在线三轮门禁
  先调用 `/pause?mode=wait&clear_cache=false`，利用 scheduler pending-transfer
  状态确定性 drain，再 reset 本地 prefix 并 resume；不再用固定 sleep 猜测上传完成。
- [x] `global_ub + pinned` 的 NPU0→NPU6 1 MiB/单 descriptor 门禁失败：pinned
  allocation、Host registration、segment connect 均成功，最终仍在
  `HcommChannelFenceOnThread` 返回 `HCCL_E_NETWORK`。故障与 swapped SVM、
  descriptor 数量和 arena 大小无关。
- [x] `global_ub_host + pinned` 的相同门禁失败：协议只有 `ub_ctp:host`，两端
  均为 `aclrtMallocHost` Host memory，单 descriptor/1 MiB 仍在 NPU0→NPU6 的
  `HcommChannelFenceOnThread` 返回 `HCCL_E_NETWORK`。mixed endpoint 误选已排除。
- [x] 源码合同复核：Mooncake/HIXL 存在 `MEM_HOST`、`COMM_TYPE_UB_H2H` 和
  A5 `ub_ctp:host` 路径，没有主动拒绝 H2H；但没有对 A5 跨 NUMA/跨卡组 H2H
  作出保证，Ascend Direct 远端路径也没有 Mooncake 通用 NUMA-aware fallback。
- [ ] 仅当通信侧提供经过验证的显式 HiXLEP route，或新版本 Mooncake/HIXL
  明确修复跨卡组 Host H2H 后，才重启 pinned→swapped direct 门禁；不再随机组合
  transport 参数，也不允许用 pinned 替换 KSC DRAM pool。
- [ ] NPU4 将 swapped source 本地写入自己的 segment，NPU0 独立 consumer
  从该对象恢复到 swapped destination，验证 PD 所需的 remote L3→local L2。
- [ ] TP8 下分别验证指定 NPU4 segment 与 Master automatic placement，区分
  跨卡组连通性、TP8 资源放大和框架计算并发三个因素。
- [x] `global_ub_device` 负向门禁：swapped tensor 的物理介质是已由
  torch_npu 执行过 `aclrtHostRegister` 的 host memory；device-only endpoint
  会再次执行 host mapping，并以 `halHostRegister drvRetCode=12` 失败。DSA L2
  直通 Mooncake 必须优先使用同时包含 device/host endpoint 的 `global_ub`。

## P1-W TCP 框架验收顺序

1. 运行 Backend/DSA UT；
2. 运行 `ascend-store-mooncake-tcp`，随后回归 `ascend-store-mooncake`；
3. 启动 Mooncake Master，运行 `cache-init + ENABLE_L3_WRITE_BEHIND=True`，只验
   all-rank segment 与 TP0 大 L2 MR registration/teardown；
4. 运行单个长 prompt eager，核对 planned/submitted/completed 与输出；
5. 开启 chunked prefill，核对每 chunk 增量 durable block；
6. eager 稳定后再跑 graph 和在线服务。

该顺序是 P1-W 的历史验收记录；P1-W 已关闭，当前执行上面的 P1-R 门禁。

Mooncake AscendDirect 的 DSA Host H2H 路径因跨 NUMA 失败而暂停；以下条目保留
为历史证据。P1-W 正式路线已经切换到 Store-owned TCP：

- [x] 单卡真实 DSA swapped schema：42 个 Indexer plane + 78 个 MLA plane，
  普通 HBM 先注册、swapped range 后注册，两个 component put/get 逐字节一致；
- [ ] 用 DSA A5 真实 split schema（21 个 full Indexer key/scale plane +
  78 个 packed MLA plane）完成 TP8 swapped Host put/get 逐字节往返；
- [x] 验证 `empty_with_swapped_memory` L2 arena 能直接作为 Mooncake
  `global_ub` put 源和 get 目标，无需固定 staging ring。

当前先执行单卡 swapped gate：

```bash
cd /home/y00813308/vllm-ascend-v0.23/examples/dsa_demo/L3_cache_sys_api_test

ASCEND_VISIBLE_DEVICES=0 \
ASCEND_RT_VISIBLE_DEVICES=0 \
DEVICE_ID=0 \
DSA_SWAPPED_COMPONENT=all \
DSA_SWAPPED_SCHEMA_MODE=smoke \
DSA_SWAPPED_MOONCAKE_COMM_MODE=global_ub \
bash run_l3_cache_api_tests.sh dsa-swapped-mooncake

ASCEND_VISIBLE_DEVICES=0 \
ASCEND_RT_VISIBLE_DEVICES=0 \
DEVICE_ID=0 \
DSA_SWAPPED_COMPONENT=all \
DSA_SWAPPED_SCHEMA_MODE=exact \
DSA_SWAPPED_MOONCAKE_COMM_MODE=global_ub \
bash run_l3_cache_api_tests.sh dsa-swapped-mooncake
```

回传完整日志，尤其保留：schema、`register_memory` 失败位置、put/get 返回值、
registered region 数量、digest 和最终退出码。`global_ub_device` 只保留为二次
host mapping 的负向诊断，不作为 DSA L2 正向门禁。若 `exact` 失败，再分别执行：

```bash
DSA_SWAPPED_SCHEMA_MODE=smoke bash run_l3_cache_api_tests.sh dsa-swapped-mooncake
DSA_SWAPPED_COMPONENT=indexer bash run_l3_cache_api_tests.sh dsa-swapped-mooncake
DSA_SWAPPED_COMPONENT=mla bash run_l3_cache_api_tests.sh dsa-swapped-mooncake
```

## 0. 公共准备

确保节点独占、NPU 空闲，且没有遗留的 vLLM、MetaService 或 Mooncake Master。

```bash
cd /home/y00813308/vllm-ascend-v0.23/examples/dsa_demo/L3_cache_sys_api_test
export LOG_DIR=/tmp/dsa_l3_post6_$(date +%Y%m%d_%H%M%S)
bash run_l3_cache_api_tests.sh diagnostics
```

`diagnostics.log` 必须显示上述 TorchNPU `git_version`。

## 1. 单卡回归

安装 TorchNPU post6 后，先确认既有单卡能力没有回归。

### MemCache

```bash
ASCEND_VISIBLE_DEVICES=0 \
ASCEND_RT_VISIBLE_DEVICES=0 \
DEVICE_ID=0 \
DEVICE_PROTOCOL=device_urma \
bash run_l3_cache_api_tests.sh memcache-all

ASCEND_VISIBLE_DEVICES=0 \
ASCEND_RT_VISIBLE_DEVICES=0 \
DEVICE_ID=0 \
DEVICE_PROTOCOL=device_urma \
bash run_l3_cache_api_tests.sh ascend-store-memcache
```

### Mooncake

```bash
ASCEND_VISIBLE_DEVICES=0 \
ASCEND_RT_VISIBLE_DEVICES=0 \
DEVICE_ID=0 \
bash run_l3_cache_api_tests.sh mooncake-npu

ASCEND_VISIBLE_DEVICES=0 \
ASCEND_RT_VISIBLE_DEVICES=0 \
DEVICE_ID=0 \
bash run_l3_cache_api_tests.sh ascend-store-mooncake
```

任一单卡门禁失败时停止，不进入多卡测试。

## 2. 两卡关键门禁

优先使用健康的物理 NPU 0、1，排除此前 NPU 7 `Warning` 状态的干扰。

### MemCache

默认包/rootinfo 路径在当前单机不再执行。若安装了支持 `USE_LOCAL_EID` 的临时包，
可直接执行下面的 here-doc，生成当前实验节点已经通过 ping 筛选的 EID 映射：

```bash
cat >/tmp/memcache-local-eids.json <<'JSON'
{
  "0": "000000000041050000100000df001200",
  "1": "000000000047050000100000df003801",
  "2": "000000000041050000100000df005202",
  "3": "000000000046050000100000df007703",
  "4": "000000000042050000100000df009304",
  "5": "000000000042050000100000df00b305",
  "6": "000000000042050000100000df00d306",
  "7": "000000000042050000100000df00f307"
}
JSON
```

这组值是机器拓扑数据，不是 950DT 的通用常量。换节点、换卡或拓扑变化后必须
重新逐卡验证，不能直接复用。

终端 1 保持 MetaService 运行，终端 2 执行：

```bash
ASCEND_VISIBLE_DEVICES=0,1 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
MEMCACHE_TP_ACTIVE_DEVICES=0,1 \
MEMCACHE_TP_LOCAL_EID_MAP_FILE=/tmp/memcache-local-eids.json \
bash run_l3_cache_api_tests.sh memcache-tp
```

必须输出 `CROSS-RANK ROUND TRIP PASSED`。EID 映射按物理 NPU ID 索引，不能按
MemCache auto-ranking 产生的动态 rank 索引。既有脚本和日志也继续用于：

- 多服务器超节点环境验证出框通信；
- MemFabric 增加同机 peer 内部 fullmesh 选路后的回归；
- 验证 Host transport 功能兜底，但 Host 路径不能代表 NPU 直通通过。

### Mooncake

终端 1 保持 Master 前台运行：

```bash
cd /home/y00813308/vllm-ascend-v0.23/examples/dsa_demo/L3_cache_sys_api_test
bash baseline_e2e/start_mooncake_master.sh
```

终端 2 执行门禁：

```bash
ASCEND_VISIBLE_DEVICES=0,1 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
MOONCAKE_TP_ACTIVE_DEVICES=0,1 \
MOONCAKE_TP_COMM_MODE=global_ub_device \
bash run_l3_cache_api_tests.sh mooncake-tp
```

随后固定生产者为 NPU0，验证另一卡组的 NPU4：

```bash
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MOONCAKE_TP_ACTIVE_DEVICES=0,4 \
MOONCAKE_TP_COMM_MODE=global_ub \
MOONCAKE_TP_LOCAL_HOST=127.0.0.1 \
MOONCAKE_TP_BUFFER_MODE=swapped \
MOONCAKE_TP_PAYLOAD_LAYOUT=dsa_a5_glm52 \
MOONCAKE_TP_SEGMENT_BYTES=4294967296 \
MOONCAKE_TP_ARENA_BLOCKS=4353 \
MOONCAKE_TP_BLOCK_INDEX=1 \
MOONCAKE_TP_IO_THREAD=background \
MOONCAKE_TP_TARGET_DEVICE=4 \
MOONCAKE_TP_PRODUCER_RANK=0 \
bash run_l3_cache_api_tests.sh mooncake-tp
```

当前该用例已经失败，先停止 TP8 扩展。使用相同 NPU0/NPU4 验证基线式 HBM
`global_ub_device` 路径：

```bash
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MOONCAKE_TP_ACTIVE_DEVICES=0,4 \
MOONCAKE_TP_COMM_MODE=global_ub_device \
MOONCAKE_TP_LOCAL_HOST=127.0.0.1 \
MOONCAKE_TP_BUFFER_MODE=device \
MOONCAKE_TP_PAYLOAD_LAYOUT=dsa_a5_glm52 \
MOONCAKE_TP_SEGMENT_BYTES=4294967296 \
MOONCAKE_TP_ARENA_BLOCKS=1 \
MOONCAKE_TP_BLOCK_INDEX=0 \
MOONCAKE_TP_IO_THREAD=background \
MOONCAKE_TP_TARGET_DEVICE=4 \
MOONCAKE_TP_PRODUCER_RANK=0 \
bash run_l3_cache_api_tests.sh mooncake-tp
```

以下命令仅保留为已完成的 staging 诊断记录，不再作为正式 DSA fallback 或
后续必跑门禁。Mooncake 默认强制 `adxl.BufferPool=0:0`；重新启用时必须改用
`auto` 选择兼容 CommEngine：

```bash
ASCEND_BUFFER_POOL=4:8 \
HCCL_INTRA_ROCE_ENABLE=1 \
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MOONCAKE_TP_ACTIVE_DEVICES=0,6 \
MOONCAKE_TP_COMM_MODE=auto \
MOONCAKE_TP_LOCAL_HOST=100.102.207.172 \
MOONCAKE_TP_BUFFER_MODE=swapped \
MOONCAKE_TP_PAYLOAD_LAYOUT=uniform \
MOONCAKE_TP_NUM_BUFFERS=1 \
MOONCAKE_TP_PAYLOAD_BYTES=1048576 \
MOONCAKE_TP_SEGMENT_BYTES=1073741824 \
MOONCAKE_TP_ARENA_BLOCKS=1 \
MOONCAKE_TP_BLOCK_INDEX=0 \
MOONCAKE_TP_IO_THREAD=background \
MOONCAKE_TP_TARGET_DEVICE=6 \
MOONCAKE_TP_PRODUCER_RANK=0 \
bash run_l3_cache_api_tests.sh mooncake-tp
```

最小门禁通过后，再复测原始 swapped exact 路径：

```bash
ASCEND_BUFFER_POOL=4:8 \
HCCL_INTRA_ROCE_ENABLE=1 \
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MOONCAKE_TP_ACTIVE_DEVICES=0,4 \
MOONCAKE_TP_COMM_MODE=auto \
MOONCAKE_TP_LOCAL_HOST=100.102.207.172 \
MOONCAKE_TP_BUFFER_MODE=swapped \
MOONCAKE_TP_PAYLOAD_LAYOUT=dsa_a5_glm52 \
MOONCAKE_TP_SEGMENT_BYTES=4294967296 \
MOONCAKE_TP_ARENA_BLOCKS=4353 \
MOONCAKE_TP_BLOCK_INDEX=1 \
MOONCAKE_TP_IO_THREAD=background \
MOONCAKE_TP_TARGET_DEVICE=4 \
MOONCAKE_TP_PRODUCER_RANK=0 \
bash run_l3_cache_api_tests.sh mooncake-tp
```

上式通过后，验证远端 L3 恢复到另一卡组的 LocalDRAM；`producer_rank=1`
对应 active device 列表中的 NPU4：

```bash
ASCEND_BUFFER_POOL=4:8 \
HCCL_INTRA_ROCE_ENABLE=1 \
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MOONCAKE_TP_ACTIVE_DEVICES=0,4 \
MOONCAKE_TP_COMM_MODE=auto \
MOONCAKE_TP_LOCAL_HOST=100.102.207.172 \
MOONCAKE_TP_SOURCE_BUFFER_MODE=swapped \
MOONCAKE_TP_DESTINATION_BUFFER_MODE=swapped \
MOONCAKE_TP_PAYLOAD_LAYOUT=uniform \
MOONCAKE_TP_NUM_BUFFERS=1 \
MOONCAKE_TP_PAYLOAD_BYTES=1048576 \
MOONCAKE_TP_SEGMENT_BYTES=1073741824 \
MOONCAKE_TP_ARENA_BLOCKS=1 \
MOONCAKE_TP_BLOCK_INDEX=0 \
MOONCAKE_TP_IO_THREAD=background \
MOONCAKE_TP_TARGET_DEVICE=4 \
MOONCAKE_TP_PRODUCER_RANK=1 \
MOONCAKE_TP_CONSUMER_DEVICE=0 \
bash run_l3_cache_api_tests.sh mooncake-tp
```

只有 Host 与 HBM 两条跨卡组路径都通过后，才把 active devices 扩为 TP8，
并仍显式定向 NPU4：

```bash
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MOONCAKE_TP_ACTIVE_DEVICES=0,1,2,3,4,5,6,7 \
MOONCAKE_TP_COMM_MODE=global_ub \
MOONCAKE_TP_LOCAL_HOST=127.0.0.1 \
MOONCAKE_TP_BUFFER_MODE=swapped \
MOONCAKE_TP_PAYLOAD_LAYOUT=dsa_a5_glm52 \
MOONCAKE_TP_SEGMENT_BYTES=4294967296 \
MOONCAKE_TP_ARENA_BLOCKS=4353 \
MOONCAKE_TP_BLOCK_INDEX=1 \
MOONCAKE_TP_IO_THREAD=background \
MOONCAKE_TP_TARGET_DEVICE=4 \
MOONCAKE_TP_PRODUCER_RANK=0 \
bash run_l3_cache_api_tests.sh mooncake-tp
```

显式 TP8 定向也通过后，再让 Master 自动放置复刻框架配置：

```bash
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MOONCAKE_TP_ACTIVE_DEVICES=0,1,2,3,4,5,6,7 \
MOONCAKE_TP_COMM_MODE=global_ub \
MOONCAKE_TP_LOCAL_HOST=127.0.0.1 \
MOONCAKE_TP_BUFFER_MODE=swapped \
MOONCAKE_TP_PAYLOAD_LAYOUT=dsa_a5_glm52 \
MOONCAKE_TP_SEGMENT_BYTES=4294967296 \
MOONCAKE_TP_ARENA_BLOCKS=4353 \
MOONCAKE_TP_BLOCK_INDEX=1 \
MOONCAKE_TP_IO_THREAD=background \
MOONCAKE_TP_PLACEMENT_MODE=automatic \
MOONCAKE_TP_PRODUCER_RANK=0 \
bash run_l3_cache_api_tests.sh mooncake-tp
```

必须输出：

```text
CROSS-SEGMENT ROUND TRIP PASSED
```

DSA 正式 write-behind 还要求 swapped Host 跨 segment 门禁：

```bash
ASCEND_VISIBLE_DEVICES=0,1 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
MOONCAKE_TP_ACTIVE_DEVICES=0,1 \
MOONCAKE_TP_COMM_MODE=global_ub \
MOONCAKE_TP_BUFFER_MODE=swapped \
MOONCAKE_TP_PAYLOAD_LAYOUT=dsa_a5_glm52 \
MOONCAKE_TP_ARENA_BLOCKS=4353 \
MOONCAKE_TP_BLOCK_INDEX=1 \
MOONCAKE_TP_IO_THREAD=background \
MOONCAKE_TP_FORCE_REMOTE_PUT=true \
MOONCAKE_TP_PRODUCER_RANK=0 \
bash run_l3_cache_api_tests.sh mooncake-tp
```

若 Mooncake 仍停在 `HcommEndpointCreate`、`ThreadAlloc` 或 remote get，保留
自动生成的主日志和 `*_hcomm.log`，并停止完整模型测试。

## 3. TP8 轻量门禁

只有对应后端的两卡门禁通过后才执行。当前 NPU 4-7 为 `Warning`，Mooncake
先用健康的 NPU 0-3 做四卡扩展：

```bash
ASCEND_VISIBLE_DEVICES=0,1,2,3 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
MOONCAKE_TP_ACTIVE_DEVICES=0,1,2,3 \
MOONCAKE_TP_COMM_MODE=global_ub_device \
bash run_l3_cache_api_tests.sh mooncake-tp
```

四卡通过后，再决定是否使用包含告警卡的 TP8。

### MemCache

仅在上述两卡真实 IO 通过后执行。映射文件必须覆盖 0-7：

```bash
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MEMCACHE_TP_ACTIVE_DEVICES=0,1,2,3,4,5,6,7 \
MEMCACHE_TP_LOCAL_EID_MAP_FILE=/tmp/memcache-local-eids.json \
bash run_l3_cache_api_tests.sh memcache-tp
```

该门禁会让每个 rank 发布一个 1 MiB NPU buffer，并从下一 rank 读回后逐字节校验。
八卡 `store.init` 通过本身不能替代此门禁。

### Mooncake

```bash
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MOONCAKE_TP_ACTIVE_DEVICES=0,1,2,3,4,5,6,7 \
MOONCAKE_TP_COMM_MODE=global_ub_device \
bash run_l3_cache_api_tests.sh mooncake-tp
```

## 4. DSA-disabled 完整模型

只选择已经通过 TP8 数据面门禁的后端。优先离线，再测在线；此阶段仍关闭 DSA。

### MemCache 服务

终端 1：

```bash
cd /home/y00813308/vllm-ascend-v0.23/examples/dsa_demo/L3_cache_sys_api_test/baseline_e2e
bash start_memcache_meta.sh 2>&1 | tee memcache-meta.log
```

终端 2 保持 `offline_baseline_l3.py` 顶部为 `L3_BACKEND = "memcache"`，确认
`MEMCACHE_LOCAL_EID_MAP_PATH` 指向刚才生成的文件，然后执行：

```bash
python offline_baseline_l3.py 2>&1 | tee baseline-memcache-offline.log
```

离线通过后，再按 `baseline_e2e/README.md` 启动在线服务和客户端。

### Mooncake 服务

终端 1：

```bash
cd /home/y00813308/vllm-ascend-v0.23/examples/dsa_demo/L3_cache_sys_api_test/baseline_e2e
bash start_mooncake_master.sh 2>&1 | tee mooncake-master.log
```

终端 2 将 `offline_baseline_l3.py` 顶部设置为：

```python
L3_BACKEND = "mooncake"
ENABLE_PREFIX_CACHING = False
MOONCAKE_COMM_MODE = "global_ub_device"
```

然后执行：

```bash
python offline_baseline_l3.py 2>&1 | tee baseline-l3-offline.log
```

离线通过后，再按 `baseline_e2e/README.md` 拉起在线服务并运行
`online_baseline_l3.py`。

完整模型验收必须同时满足：

- 第 2、3 轮热重放 token/logprobs 稳定；
- 第二轮 external lookup/load 非零；
- 后端对象或 Put/Get 指标增长；
- 无 load failure、静默 recompute 和 DSA 容量/算子日志。

## 5. 失败时回传

只回传首个失败门禁，避免重复执行后续无效测试：

- `$LOG_DIR/diagnostics.log`；
- `$LOG_DIR/<gate>.log`；
- `$LOG_DIR/<gate>_hcomm.log`（多卡失败时）；
- `npu-smi info` 中相关设备健康状态。
