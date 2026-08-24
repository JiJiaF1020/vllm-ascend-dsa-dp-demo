# MemCache A5 TP 阻塞记录

本文记录 2026-08-24 在单机 Ascend 950DT 上验证 vLLM-Ascend 原生
`AscendStoreConnector + MemcacheBackend` 的多 rank 数据面时遇到的阻塞。

## 已通过能力

1. `memcache_hybrid==1.2.0`、`memfabric_hybrid==1.2.0` 可正常导入。
2. 与 native commit `e973ccbb8fa8ea92c9efe0821df731182d86234e`
   匹配的 AICPU run 包状态为 `READY`。
3. 单 LocalService Host、NPU L2G/G2L、GVA alloc/copy/publish/lease 均逐字节通过。
4. 真实 `MemcacheBackend` 的一个 NPU worker 与一个 `init_bm=False` scheduler
   metadata client 往返通过。

以上门禁均未创建两个真实 LocalService 之间的数据 channel，因此不能覆盖 TP
场景的 `GroupJoin/HcommChannelCreate`。

## 完整模型首个失败

DSA disabled、TP8 初始化时，8 个 worker 均完成：

- 56-bit GVA/LVA 预留；
- 1 GiB DRAM hugepage 申请；
- Host/Device 内存注册与导出；
- AICPU kernel 加载；
- MetaService/ConfigStore 连接。

随后 rank 0 与 rank 7 在 `GroupJoin` 中等待约 120 秒并失败：

```text
device_urma Prepare HcommChannelCreate failed, peer: 7 ret: 9
device_urma Prepare HcommChannelCreate failed, peer: 0 ret: 9
GroupJoin signal_ret:-2004
```

`ret=9` 对应 `HCCL_E_TIMEOUT`。持续出现的 `TryRemovePrefixKey ... ret:-404`
是 GroupJoin 等待交换信息期间轮询尚不存在的 key，不是最终首因。

## 两卡最小复现

将逻辑池从 probe 参数收敛为：

```text
world_size=33
max.dram.size=1TB
dram.size=1GB
```

`1TB x 33 = 33TiB`，仍满足 `device_urma` 总逻辑池大于 32TiB、启用 56-bit
GVA 的约束；只启动 NPU0、7 两个 LocalService。结果为：

- 两端均成功初始化 endpoint、DRAM 和 NPU buffer；
- 一端可以完成本地 put；
- 两端在首次相互 `HcommChannelCreate` 时同时超时；
- 不加载模型也能稳定复现。

这证明失败不属于 vLLM KV cache 数量、模型结构、DSA 或 256-rank 配置。

## 与 Mooncake 的对照

Mooncake 和 MemCache 的单卡 API 都能通过，但两者不是同一套多 rank 选路。
Mooncake 已使用 `global_ub_device` 通过同机 TP8 内部 UB/fullmesh 门禁；MemCache
`device_urma/UBC_CTP` 则固定从 `/etc/hccl_rootinfo.json` 选择 endpoint EID，
当前没有根据 peer 是否同机切换通信平面。现场日志还显示：

```text
server_id=65535
superpod id=4294967295
```

这两个值呈现为常见的 unsigned 最大值哨兵。CANN 9.1 HComm 源码中的 direct UBC_CTP
`EndpointDescPairToLinkData()` 只消费本地/远端 EID 与物理 device id，不消费
`serverIdx/superPodIdx`，因此不能把这两个哨兵当作首要根因。Mooncake 的通过
结果进一步说明本机内部 UB 通路可用，MemCache 失败应归因于选中的通信平面，
而不是泛化为整机 HComm 失效。

## CANN 9.1 源码审计

- `HcommChannelCreate` 的实现同时接受 `COMM_ENGINE_AICPU` 与
  `COMM_ENGINE_AICPU_TS`，二者都会创建 `AicpuTsUrmaChannel`；MemFabric/HIXL
  使用 `AICPU` 不是当前 timeout 的充分根因。
- direct UBC_CTP channel 通过 EID、local/remote physical device id 构造
  `LinkData`，然后建立 UB socket 与 `DevUbCtpConnection`。
- 下一轮最小复现改用健康状态均为 `OK` 的 NPU 0、1，并自动收集最近十分钟
  `AicpuTsUrmaChannel`、Socket、`DevUbCtpConnection` 和 `UbMemTransport` plog。

## 阶段性状态（历史）

MemCache A5 TP 标记为 **当前实现不支持本单机内部 fullmesh 选路**。在上游能力
或测试环境发生变化前：

- 不再运行完整模型；
- 保留 `test_memcache_tp.py` 作为最小验收门禁；
- 单卡 gate 的通过结论继续有效，但不得外推为 TP 数据面通过；
- DSA 本地 DRAM 功能不受影响；L3 主路线转向 Mooncake。

## 2026-08-25 正式版镜像复测

使用 vLLM-Ascend v0.23.0 正式版镜像重新创建容器，并补齐
`/dev/ummu`、`/dev/uburma`、`/lib/route.conf`、`/etc/hccn.conf`、
`/etc/hccl_rootinfo.json` 与 `/etc/hixlep` 后，两卡最小门禁仍未通过：

- CANN 9.1.0、`torch_npu==2.10.0.post4`；
- `memfabric_hybrid==1.2.0` 的 native commit 为
  `999034e9320abf27bbde7be83a32d3bf4d0f7940`；
- AICPU kernel 预检为 `READY`，endpoint、内存注册/导出和本地 put 成功；
- 两端 `HcommChannelCreate` 均返回有效 channel/thread handle；
- channel 状态持续为 `IN_PROGRESS(1)`，100 秒后
  `WaitForChannelReady` 返回 `BM_TIMEOUT(-7)`；
- `server_id=65535`、`superpod_id=4294967295` 仍未变化。

相较旧 wheel 在 `HcommChannelCreate` 内直接返回 `HCCL_E_TIMEOUT(9)`，新
wheel 只是将异步建链状态显式轮询出来；首个失败边界仍是 HComm/UB channel
无法进入 ready。由此可排除旧容器漏挂 A5 设备或拓扑文件是唯一根因，也确认
本次 v0.23.0 正式版 DSA 迁移不会自行修复该外部阻塞。

本轮使用 NPU 0、7，其中 NPU 7 为 `Warning`。仍需以 NPU 0、1 做一次最终
消歧；若同样停在 `IN_PROGRESS`，即可排除告警卡因素。

## 2026-08-25 TorchNPU post6 与健康卡复测

在 NPU 0、1 均为 `OK`、节点无运行中 NPU 进程的条件下，使用：

- `torch_npu==2.10.0.post6`；
- `git_version=a8f2f4fda83732c0d02244a46815a81f21fe3439`；
- MemFabric native commit `999034e9320abf27bbde7be83a32d3bf4d0f7940`。

单卡 Host、NPU、GVA/LAYERWISE 和真实 `MemcacheBackend` 门禁全部通过；两卡
门禁仍在相同位置失败：

- NPU1/BM rank0 成功执行 `RaSocketListenStart`，在 UB 地址
  `::3f:300:10:0:df00:2b00` 上监听端口 `60001`；
- NPU0/BM rank1 成功执行 `RaSocketBatchConnectAsync`，从
  `::3f:300:10:0:df00:b00` 连接上述远端地址；
- 两端均成功获得 HComm channel/thread handle；
- channel 状态持续为 `IN_PROGRESS(1)`，100 秒后两端
  `WaitForChannelReady` 均返回 `BM_TIMEOUT(-7)`。

该结果排除了 TorchNPU post4、NPU7 `Warning` 和旧容器挂载缺失是首因，并把
失败边界进一步收敛到本机 HCCP/UB socket 从 listen/connect 到 ready 的建链
阶段。完整模型和 TP8 MemCache 测试继续暂停。

## 最终选路结论

结合平台通信同事说明与 MemFabric 1.2.0 源码，当前失败原因收敛为：

1. Ascend 950 机器存在多级通信拓扑；当前 `/etc/hccl_rootinfo.json` 中由
   MemFabric 规则筛选出的 EID 对应出框/超节点通信平面。
2. 当前只有一台 8 卡服务器，同机 NPU 间实际可用的是内部 fullmesh 通路。
3. `DeviceUrmaTransportManager::BuildLocalEndpointDescLocked(UBC_CTP)` 无条件调用
   `GetDeviceUrmaEid()`，没有按 peer locality 为同机 rank 选择内部 fullmesh EID。
4. 因而 rootinfo 可以合法解析、channel/thread handle 也可以创建，但连接目标在
   当前单机部署中不可达，状态会持续 `IN_PROGRESS` 并最终超时。

这也解释了单卡通过、两卡首次建链失败，以及 Mooncake `global_ub_device` 能够
通过 TP8。当前 rootinfo 不应被判定为损坏，更不能为了单机 MemCache 测试手工
替换为内部 EID，否则可能破坏 HCCL 或多服务器超节点通信语义。

## 2026-08-25 临时包本地 EID 覆盖

MemFabric 临时包 native commit `c9258a05149447577a9f7e3eab434c757b59d23c`
增加了 `USE_LOCAL_EID`。每个 NPU 进程在 `store.init` 前设置本物理卡对应、经 ping
筛选可达的内部 fullmesh EID 后，8 卡最小初始化结果为：

- 8 个进程都打印 `using EID from USE_LOCAL_EID`；
- 物理 device 0-7 全部成功 auto-ranking，动态 rank 顺序与物理卡顺序不同；
- 所有 peer channel 均进入 `Prepare success`，未出现 `BM_TIMEOUT`；
- 8 个 `store.init` 均返回 0；
- barrier 后可以正常销毁 channel、unimport memory 并关闭 LocalService；
- 子进程退出码为 `[0, 0, 0, 0, 0, 0, 0, 0]`。

这证明临时覆盖能修复当前单机的建链选择，但该最小脚本没有执行 payload IO。
下一门禁必须通过真实 `MemcacheBackend` 的 local put、环形 remote get 和逐字节
校验，才可把 MemCache 恢复为 vLLM-Ascend 候选。未来框架接入时必须按物理 NPU
ID 设置 EID，不能按 MemCache 动态 rank 设置。

## 恢复条件

满足以下任一条件后再恢复 MemCache 多 rank 验证：

1. 获得具备出框通信通路的多服务器超节点环境，并确认 rootinfo EID 可达；
2. MemFabric 正式支持按 peer locality 为同机 rank 选择内部 fullmesh 通路；
3. 使用明确支持单机 A5 的其他 Device transport，并完成相同两卡逐字节门禁。

临时包的 `USE_LOCAL_EID` 属于条件 2 的验证接口，尚不能替代正式选路机制。

`test_memcache_tp.py` 输出 `CROSS-RANK ROUND TRIP PASSED` 后，才恢复 TP8 模型
冷暖两轮 external hit。Host transport 可以做功能兜底，但不能作为 NPU 直通
数据面通过的证明。
