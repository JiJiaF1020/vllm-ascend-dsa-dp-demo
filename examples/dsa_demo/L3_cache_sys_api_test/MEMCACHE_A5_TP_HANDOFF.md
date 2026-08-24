# MemCache A5 两卡问题交接说明

> 默认包结论：当前并非 rootinfo 文件损坏，也不是 vLLM/DSA 初始化错误。MemFabric
> `device_urma/UBC_CTP` 使用 rootinfo 中的出框 EID，未为同机 peer 选择内部
> fullmesh 通路，因此在只有一台 Ascend 950DT 服务器时无法完成两卡建链。
> 临时包已增加进程级 `USE_LOCAL_EID`，八卡完整 mesh 初始化通过，真实数据 IO
> 待 `test_memcache_tp.py` 验证。

## 一句话现象

MemCache/MemFabric 在单张 Ascend 950DT 上的 Host、NPU、GVA 和真实
`MemcacheBackend` 数据往返全部通过；NPU0、1 两个 LocalService 组网时，服务端
已监听、客户端已发起 UB socket 连接，但 HComm channel 一直停在
`IN_PROGRESS(1)`，100 秒后双方 `BM_TIMEOUT(-7)`，尚未进入跨 rank 数据复制。

## 环境

- 单机 8 x Ascend 950DT，本次只使用健康且空闲的 NPU0、1；
- 驱动/HDK：`npu-smi 25.6.rc1`；
- CANN：9.1.0；
- Python：3.12.13；
- PyTorch：2.10.0；
- TorchNPU：2.10.0.post6，git commit
  `a8f2f4fda83732c0d02244a46815a81f21fe3439`；
- `memcache_hybrid==1.2.0`；
- `memfabric_hybrid==1.2.0`，native commit
  `999034e9320abf27bbde7be83a32d3bf4d0f7940`；
- HYBM AICPU kernel 预检：`READY`；
- 测试代码：vLLM-Ascend commit `d4460039` 中
  `examples/dsa_demo/L3_cache_sys_api_test/test_memcache_tp.py`。

## 已通过，基本可排除

1. MemCache Python API 与 native 库可加载。
2. Host object、Host batch put/get 逐字节通过。
3. 单卡 NPU L2G/G2L 往返通过。
4. GVA alloc、batch copy、write finish、lease 和 key info 通过。
5. 真实 vLLM-Ascend `MemcacheBackend` scheduler/worker 单卡跨进程往返通过。
6. 两卡均完成 Device endpoint、AICPU kernel、1 GiB DRAM、Host/Device 内存
   注册与导出，本地 put 通过。
7. TorchNPU post4 与 post6 结果一致。
8. 使用 NPU0、1 复现，已排除先前 NPU7 `Warning` 状态。
9. 轻量探针不加载模型、不启用 DSA，已排除 vLLM 模型和 DSA 逻辑。
10. 同节点 Mooncake 使用 device-only UB endpoint 已通过 TP8，说明基础 UB
    Host↔NPU 搬运并非整体不可用；但 Mooncake 与 MemFabric 不是同一建链实现。

## 复现方式

### 最小裸 MemCache 复现

不导入 vLLM/vLLM-Ascend，也不创建 KV tensor 或执行 put/get；两个子进程仅调用
`DistributedObjectStore.init(init_bm=True)`：

终端 1：

```bash
cd /home/y00813308/vllm-ascend-v0.23/examples/dsa_demo/L3_cache_sys_api_test
bash baseline_e2e/start_memcache_meta.sh
```

终端 2：

```bash
python repro_memcache_device_urma_2card.py
```

唯一触发调用位于脚本的 `store.init(device, init_bm=True)`。当前环境预期在约
100 秒后至少一个子进程打印 `ret=-1`，parent 打印非零 exitcode。若两边均
`ret=0`，表示建链成功。

### 带完整数据校验的探针

终端 1：

```bash
cd /home/y00813308/vllm-ascend-v0.23/examples/dsa_demo/L3_cache_sys_api_test
bash baseline_e2e/start_memcache_meta.sh
```

终端 2：

```bash
ASCEND_VISIBLE_DEVICES=0,1 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
MEMCACHE_TP_ACTIVE_DEVICES=0,1 \
bash run_l3_cache_api_tests.sh memcache-tp
```

LocalService 配置：

```text
world_size=33
protocol=device_urma
dram.size=1GB
max.dram.size=1TB
hbm.size=0
max.hbm.size=0
```

`1TB x 33 = 33TiB` 仅用于满足 56-bit GVA 大于 32 TiB 的启用条件；实际只有
两个 active LocalService，各分配 1 GiB。

## 精确失败边界

本轮 auto-ranking 按启动顺序分配：NPU1 为 BM rank0，NPU0 为 BM rank1。

NPU1/BM rank0：

```text
RaSocketListenStart
localIp=::3f:300:10:0:df00:2b00
port=60001
RsSocketListenBindListen success
```

NPU0/BM rank1：

```text
RaSocketBatchConnectAsync
localIp=::3f:300:10:0:df00:b00
remoteIp=::3f:300:10:0:df00:2b00
port=60001
```

两端均完成 `HcommChannelCreate` 并获得有效 channel/thread handle，但：

```text
WaitForChannelReady status=IN_PROGRESS(1)
timeout_ms=100000
ret=BM_TIMEOUT(-7)
```

未观察到 connect/accept ready，随后 `GroupJoin` 失败。另一个 worker 的
`BrokenBarrierError` 是同伴初始化失败后的次生错误。

## 对应源码调用栈

轻量复现入口：

```text
repro_memcache_device_urma_2card.py
  DistributedObjectStore.init(device, init_bm=True)

test_memcache_tp.py:78
  MemcacheBackend(..., init_bm=True)
```

vLLM-Ascend Python 路径：

```text
pool_worker.py:107       KVPoolWorker.__init__ -> _init_backend
pool_worker.py:255       real_backend(...) -> MemcacheBackend
memcache_backend.py:58   self._setup_store()
memcache_backend.py:87   store.init(local_rank, init_bm=True)
```

MemCache native 路径：

```text
pymmc.cpp:699                    DistributedObjectStore.init binding
mmcache_store.cpp:180            mmc_init(...)
mmc.cpp:267                      mmcs_local_service_start(...)
mmc_local_service_default.cpp:48 InitBm()
mmc_local_service_default.cpp:180 bmProxy->InitBm(...)
mmc_bm_proxy.cpp:54              SmemBmInit(...)
mmc_bm_proxy.cpp:70              SmemBmJoin(...)
```

MemFabric/HYBM 建链路径：

```text
smem_bm_entry.cpp:469                 globalGroup_->GroupJoin()
smem_bm_entry.cpp:271/317             JoinHandle -> hybm_import(...)
hybm_entity_default.cpp:608           transportManager_->Prepare(...)
device_urma_transport_manager.cpp:1414 HcommChannelCreate(...)
device_urma_transport_manager.cpp:1443 WaitForChannelReady(...)
hcomm_transport_manager.cpp:58         轮询 HcommChannelGetStatus
```

当前真正超时的代码行为是
`device_urma_transport_manager.cpp:1443 -> hcomm_transport_manager.cpp:58`：
`HcommChannelCreate` 已成功返回，随后 `HcommChannelGetStatus` 始终得到
`IN_PROGRESS(1)`，直到 100 秒截止。

## 对应 vLLM 场景

该问题发生在配置以下能力时：

```text
kv_connector=AscendStoreConnector
kv_connector_extra_config.backend=memcache
TP > 1
protocol=device_urma
```

`AscendStoreConnector` 初始化 worker 侧 `KVPoolWorker` 时，会为每个 TP worker
立即创建 `MemcacheBackend(init_bm=True)`。`device_urma` 不走 lazy init，因此
各 worker 在引擎拉起阶段就加入 MemFabric Big Memory group 并建立两两通道。

因此它对应的是 **vLLM 服务/离线 LLM 初始化阶段**，发生在第一条请求、prefix
lookup、prefill、decode 和实际 KV put/get 之前。scheduler 侧只创建
`init_bm=False` 的元数据 client，不触发该建链；TP1 不需要远端 peer，也不会
复现。

## 历史待确认项

- 两卡 EID 可以正确交换，NPU1 listen、NPU0 async connect 的地址和端口相互
  对应。
- 日志中的 `server_id=65535`、`superpod_id=4294967295` 为哨兵值；请确认在
  单机 950DT UBC_CTP 场景是否合法。
- 两个 LocalService 均显示默认 `transUrl=tcp://127.0.0.1:7000`。源码审计中
  `device_urma::OpenDevice` 没有消费 `options.nic`，但请确认多 rank 是否仍要求
  每个 LocalService 配置唯一 `hcom_url`。
- auto-ranking 与测试进程 rank/物理卡顺序可以不同；请确认这是支持的合同，还是
  device_urma 要求固定 rank→physical device 映射。
- 容器已直通 `/dev/uburma`、`/dev/ummu` 和各 `/dev/davinci*`，并挂载
  `/lib/route.conf`、`/etc/hccn.conf`、`/etc/hccl_rootinfo.json`、`/etc/hixlep`
  与驱动目录；请确认当前 HCCP/UB route、白名单或 CTP 配置是否还缺必需项。
- 请确认 CANN 9.1.0、HDK 25.6.rc1、MemFabric 1.2.0 commit `999034e9` 是否为
  受支持组合，以及 `device_urma` 是否是 950DT 本机 MemCache 的推荐协议。

这些问题促成了后续 rootinfo、URMA 与 Mooncake 对照测试；当前首个问题已经由
通信同事解释并由源码行为支撑，不再继续把它当作未知 HCCP 故障。

## 最终根因

MemFabric 1.2.0 固定读取 `/etc/hccl_rootinfo.json`，再读取其中的
`topo_file_path`。对 `Atlas 950 SuperPoD`，它会在每个 `device_id` 的
`level_list/rank_addr_list` 中选择唯一一个 `addr_type=EID` 且 `ports` 数量为 6
的地址作为本卡 endpoint。

当前文件中被选中的地址属于出框/超节点通信平面；单机 NPU 间实际走内部
fullmesh。MemFabric 的 `UBC_CTP` endpoint 构造没有按同机/跨机 peer 分流，
所以两张本地卡仍尝试使用出框 EID。地址解析和 channel 创建可以成功，但实际
通路不可达，最终表现为 `IN_PROGRESS -> BM_TIMEOUT(-7)`。

对照证据是 Mooncake `global_ub_device` 已在同一节点通过 NPU 0-1、0-3 和 TP8
内部 UB 往返，排除了本机内部通信硬件整体不可用。

## RootInfo 只读核对

审计工具保留用于解释 MemFabric 选中了哪个 EID，不用于指导业务侧修改文件：

```bash
python audit_hccl_rootinfo.py /etc/hccl_rootinfo.json \
  | tee /tmp/hccl_rootinfo.audit.txt
urma_admin show topo > /tmp/urma_topo.current.txt
```

当前不要重新生成或替换 rootinfo。该文件可能完全符合多服务器超节点部署需求；
把内部 fullmesh EID 手工填入会改变 HCCL 的通信平面语义。只有平台团队确认文件
本身确实偏离整机部署设计时，才应使用与 HDK/MindCluster 配套的正式工具重建。

## 后续选择

1. 当前 DSA L3 集成优先使用已经通过单机 TP8 门禁的 Mooncake
   `global_ub_device`。
2. MemCache 使用临时包和显式物理卡 EID 映射，依次验证两卡、TP8 的真实跨 rank
   payload IO；建链成功不能替代数据校验。
3. 若数据面通过，再评估在 vLLM-Ascend worker 初始化 MemcacheBackend 前按物理
   NPU ID 设置 EID；正式方案仍应由 MemFabric 提供 peer-locality 选路或配置合同。
