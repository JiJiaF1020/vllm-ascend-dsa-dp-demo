# vLLM-Ascend L3 Backend 独立 API 探针

本目录统一维护 DSA 对接 L3 前后的 API、环境、框架拉起和回归脚本。第一阶段
提供两个不依赖模型权重、也不拉起 vLLM engine 的测试入口：

- `test_mooncake.py`
- `test_memcache.py`（实现位于 `memcache_a5_probe.py`）

目标不是证明 Prefix Cache 已集成，而是提前验证 vLLM-Ascend backend 实际
使用的底层 IO API 在 A5 环境可用。

后续新增内容也必须放在本目录，包括：

- 社区 baseline + AscendStoreConnector 拉起脚本；
- L3 cold/warm prefix 测试；
- DSA L1/L2/L3 component source 测试；
- chunked prefill 增量下沉测试；
- eager/graph、在线/离线和多 TP rank IO 测试。

## 文件说明

| 文件 | 作用 |
|---|---|
| `CURRENT_TEST_CHECKLIST.md` | 当前环境应执行的最短门禁顺序、停止条件和回传日志 |
| `test_mooncake.py` | Mooncake A5 TransferEngine 与 multi-buffer IO 探针 |
| `test_mooncake_tp.py` | 8 个 A5 worker 的跨 segment HIXL 往返探针 |
| `MOONCAKE_A5_TP_BLOCKER.md` | Mooncake A5 TP 历史问题、兼容路径与 TP8 通过结论 |
| `test_memcache.py` | MemCache 探针统一入口 |
| `test_memcache_tp.py` | 多个 A5 LocalService 的跨 rank URMA 数据往返探针 |
| `repro_memcache_device_urma_2card.py` | 不依赖 vLLM 的最小两卡 MemCache 建链复现 |
| `start_a5_l3_container.sh` | 宿主机执行的 A5 L3 调试容器启动与挂载预检 |
| `MEMCACHE_A5_TP_BLOCKER.md` | MemCache A5 TP 失败证据与单机内部 fullmesh 选路限制 |
| `MEMCACHE_A5_TP_HANDOFF.md` | 可直接转交 MemCache/MemFabric/HComm 同事的定位结论 |
| `test_memfabric_aicpu.py` | A5 URMA 所需 HYBM AICPU kernel 预检/provision |
| `test_ascend_store_backends.py` | 真实 vLLM-Ascend Backend scheduler/worker 集成探针 |
| `check_hixlep.py` | 校验 A5 TP worker 使用的逐卡 HiXLEP LocalCommRes 文件 |
| `audit_hccl_rootinfo.py` | 只读汇总 MemFabric 实际消费的 rootinfo EID 与拓扑文件 |
| `memcache_a5_probe.py` | MemCache Host/NPU/GVA/lease 完整实现 |
| `run_l3_cache_api_tests.sh` | 分 Gate 执行、记录日志并保留真实退出码 |
| `README_memcache_a5_probe.md` | MemCache 分阶段命令和故障定位 |
| `baseline_e2e/` | DSA disabled 的配置模板、L3 服务和在线/离线双轮验证 |

## 对齐版本

| 组件 | 版本/commit |
|---|---|
| vLLM-Ascend baseline | v0.23.0 official `5cb98caaad` |
| Mooncake NPU wheel/source | `v0.3.11.post1` / `e9c6107572` |
| MemCache | `v1.2.0` / `4310d68fa8` |
| MemFabric submodule | `9a9cfbc8a1` |
| HIXL source for CANN 9.1 | `v9.1.0` / `c12f9a56ab` |
| HComm source for CANN 9.1 | `v9.1.0` / `4ca1db912c` |
| LMCache / LMCache-Ascend reference | `v0.3.12` / `78697950e9dc`, `58e0aef462ed` |
| UCM reference | `v0.5.0` / `52b0f545d993` |

正式版固定 Mooncake `0.3.11.post1`，部署文档使用 LMCache `0.3.12`，并要求
MemCache `>=1.2.0`；UCM 文档没有锁定源码版本。因此这里只将 HIXL 对齐到
CANN 9.1.0 tag，其余仓保持已匹配版本，UCM 继续作为此前分析使用的稳定参考，
不追随远端 master。

当前单机 A5 的主验证路线调整为 **Mooncake + `global_ub_device`**。该路径已经
通过单卡、两卡、四卡和 TP8 跨 segment 逐字节往返；CANN 9.1.0 中
`ub_ctp:device + ub_ctp:host` 的历史 D2rH 问题仍保留为运行时限制，但不再阻塞
device-only 兼容路径。完整证据见
[Mooncake A5 TP 记录](MOONCAKE_A5_TP_BLOCKER.md)。

MemCache 的单卡 Host/NPU/GVA 与单 worker Backend gate 继续有效，但其
`device_urma/UBC_CTP` 会固定使用 `/etc/hccl_rootinfo.json` 中筛选出的出框 EID，
没有为同机 peer 切换到内部 fullmesh 通路。当前只有单机 8 卡环境，因此两卡
channel 停在 `IN_PROGRESS` 并最终超时属于适用边界，而不是 DSA、vLLM、容器
挂载或 rootinfo 文件损坏。不要为单机测试手改 rootinfo；默认选路只用于上游
适配或多服务器超节点环境的回归，临时 EID 覆盖路径见下文及
[MemCache A5 TP 记录](MEMCACHE_A5_TP_BLOCKER.md)。

2026-08-25 的 MemFabric 临时包增加了进程级 `USE_LOCAL_EID` 覆盖。为 8 个 worker
按物理 NPU ID 设置已验证可达的内部 fullmesh EID 后，完整 mesh 初始化、内存
import 和正常清理均通过，原先的 channel timeout 消失。后续 TP8
`test_memcache_tp.py` 又完成每个 rank 的 local put、环形 remote get 和 1 MiB
payload 逐字节校验，gate 退出码为 0。Mooncake 继续作为优先主路线，MemCache
恢复为可并行推进到 DSA-disabled 完整模型的候选。

## A5 调试容器

当前 CANN/HComm/HiXLEP 问题必须在挂载完整 A5 路由资源的容器中复测。宿主机
执行：

```bash
bash start_a5_l3_container.sh <official-image> <container-name>
```

脚本会在 `docker run` 前检查 8 张 NPU、`/dev/ummu` 与 `/dev/uburma` 设备目录、
`/lib/route.conf`、`/etc/hccn.conf`、`/etc/hccl_rootinfo.json`、`/etc/hixlep`、
`hccn_tool` 与 `urma_admin`，缺失时直接退出。拓扑文件以只读方式挂载，模型和
工程目录保持读写挂载。

默认不挂载宿主机整个 `/usr/lib64`，确保新镜像的用户态依赖不被覆盖。只有镜像
官方说明明确要求时才设置：

```bash
MOUNT_HOST_USR_LIB64=1 \
  bash start_a5_l3_container.sh <official-image> <container-name>
```

MemCache TP8 历史配置使用 `world_size=33`、`max.dram.size=1TB`，用于满足
`device_urma` 大于 32 TiB 的逻辑 GVA envelope 约束。该配置和最小两卡复现仍
保留给 MemFabric 维护者，但不再作为当前单机 A5 的模型前置门禁。

`mooncake-transfer-engine-npu` 是 NPU 发行包名；该 wheel 同时包含
`mooncake.engine`、`mooncake.store` 和 `mooncake_master`。详细导入检查见
[MemCache/A5 探针说明](README_memcache_a5_probe.md)。

## API 覆盖

### MooncakeBackend

| vLLM-Ascend 调用 | 探针覆盖 |
|---|---|
| `TransferEngine.initialize(..., "ascend", ...)` | 是 |
| `TransferEngine.register_memory/unregister_memory` | 是 |
| `MooncakeDistributedStore.setup` | 是 |
| `batch_is_exist` | 是 |
| `batch_put_from_multi_buffers` | 是 |
| `batch_get_into_multi_buffers` | 是 |
| `ReplicateConfig.prefer_alloc_in_same_node` | 是 |
| bytes `put/get` 服务健康检查 | 是 |
| SSD offload | 否，第一版不测 |

### MemcacheBackend 非 layerwise

| vLLM-Ascend 调用 | 探针覆盖 |
|---|---|
| `MetaService.setup/main` | 是 |
| `DistributedObjectStore.setup/init/close` | 是 |
| `register_buffer/unregister_buffer` | 是 |
| `batch_is_exist` | 是 |
| `batch_put_from_layers` | 是 |
| `batch_get_into_layers` | 是 |
| Host H2G/G2H | 是 |
| NPU L2G/G2L | 是 |

### MemcacheBackend layerwise/GVA

| vLLM-Ascend 调用 | 探针覆盖 |
|---|---|
| `batch_alloc` | 是 |
| `batch_copy` 写/读 | 是 |
| `batch_write_finish` | 是 |
| `batch_get_key_info` / `gva_list` | 是 |
| `batch_add_lease` | 是 |
| `batch_remove_lease` | 是 |

MemCache v1.2.0 的 native `ResourceTracker/atexit` 存在析构顺序缺陷。探针会先
正常关闭 LocalService，再跳过有缺陷的 native 全局 finalizer；自己创建的临时
MetaService 则以 SIGKILL 结束。因此成功日志中的 MetaService `exitcode=-9` 是
探针预期行为，不是数据面失败。详见
[MemCache 探针说明](README_memcache_a5_probe.md)。

## MemCache 概念分层

这里有两组彼此正交的概念，不能混在一起理解：

1. `--mode` 决定探针验证哪一层 **API 与数据流**；
2. `--protocol` 决定 MemFabric 使用哪一种 **底层搬运/互联机制**。

### 探针 mode

| mode | 本地缓冲区 | API 路径 | 对应的 vLLM-Ascend 用途 |
|---|---|---|---|
| `capabilities` | 无 | 只检查 wheel、类和方法 | Gate 0，不创建内存池 |
| `host` | CPU DRAM tensor | `batch_put/get_*_layers`，H2G/G2H | 验证对象、MetaService 和 Host 池，不验证 HBM IO |
| `npu` | NPU HBM tensor | `batch_put/get_*_layers`，L2G/G2L | 非 layerwise 的 key/object 型直接 HBM IO |
| `gva` | NPU HBM tensor | `batch_alloc/copy/write_finish/key_info/lease` | MemCache layerwise 路径使用的显式地址型 IO |
| `both` | CPU DRAM + NPU HBM | `host` 后接 `npu` | 组合测试，不包含 GVA |
| `all` | CPU DRAM + NPU HBM | `host`、`npu`、`gva` | 全数据面测试 |

`npu` 与 `gva` 都会真实搬运 HBM 数据，差别在控制接口：

- `npu` 把 `key + 多段本地地址 + size` 直接交给 MemCache，由它完成全局空间
  分配、写完成发布和按 key 读取；
- `gva` 先显式 `batch_alloc` 得到 GVA，再执行 `batch_copy`，写端调用
  `batch_write_finish` 发布对象；读端先查 `KeyInfo/GVA`、加 lease，再按地址读取。

GVA 是 **Global Virtual Address（全局虚拟地址）**。它是 MemFabric 编排的一枚
`uint64` 逻辑地址，各进程看到一致的线性全局地址布局；它不是某个 Python tensor
自身的本地 `data_ptr()`。lease 用来保证按 GVA 异步读取期间对象不被回收。

### MemFabric protocol

源码当前接受八种配置：

| protocol | 发起/执行侧 | 典型硬件或用途 | 备注 |
|---|---|---|---|
| `host_shm` | Host | 同节点 CPU 共享内存 | 不用 HCOM；只允许 DRAM 池，最适合 Host Gate |
| `host_tcp` | Host | TCP 兼容/调试路径 | 需要 HCOM，通常不是性能首选 |
| `host_rdma` | Host NIC | A2/A3 Host RoCE/RDMA | Host 侧网络搬运 |
| `host_urma` | Host UB | 鲲鹏 K5 等 Host UB | URMA 为 Unified Remote Memory Access |
| `device_sdma` | NPU Device | A3 Device UB 1.0 | 由 Device SDMA 搬运 |
| `device_rdma` | NPU Device | A2 Device RoCE/RDMA | NPU 侧直接 RDMA |
| `device_urma` | NPU Device | A5 Device UB | 使用 URMA；MemCache 会按总地址空间自动启用 56-bit GVA |
| `device_uboe` | NPU Device | A5 跨节点 UBoE | UBoE 即 UB over Ethernet |

`device_uboe` 的含义不是“使用普通 socket 把 tensor 发出去”，而是让 A5 的
Device 侧通过 **UnifiedBus over Ethernet** 访问全局池。它与 `device_urma`
共享 URMA 风格的数据操作实现，但对应的互联资源配置不同。单机 A5 首轮验证仍需
根据现场到底提供本机 UB 还是 UBoE 资源选择协议；不能只因为硬件是 A5 就认定
二者都已配置。`device_urma/device_uboe` 的总虚拟池空间需超过 32 TiB，探针用
`256GB * 256` 触发 56-bit GVA，但每个本地服务实际只分配 `DRAM_SIZE=1GB`。

## 推荐执行顺序

建议在本目录执行统一入口。日志默认写入
`/tmp/dsa_l3_cache_probe_logs`；脚本通过 `PIPESTATUS[0]` 返回 Python 的真实
退出码，不会被 `tee` 的退出码掩盖。

```bash
cd examples/dsa_demo/L3_cache_sys_api_test

# 0. 不上 NPU 的安全检查；不会加载已知会崩溃的 Mooncake native 模块。
bash run_l3_cache_api_tests.sh safe

# 1. MemCache A5 单卡普通地址型 IO，仅作既有能力回归。
DEVICE_ID=0 DEVICE_PROTOCOL=device_uboe \
  bash run_l3_cache_api_tests.sh memcache-npu

# 2. MemCache 单卡 layerwise/GVA API，不外推到多 rank。
DEVICE_ID=0 DEVICE_PROTOCOL=device_uboe \
  bash run_l3_cache_api_tests.sh memcache-gva

# 2.1 临时包按物理 NPU ID 覆盖内部 fullmesh EID 后，验证跨 rank 数据 IO。
MEMCACHE_TP_ACTIVE_DEVICES=0,1 \
MEMCACHE_TP_LOCAL_EID_MAP_FILE=/tmp/memcache-local-eids.json \
  bash run_l3_cache_api_tests.sh memcache-tp

# 3. 收集包、环境、NPU 和拓扑信息。
bash run_l3_cache_api_tests.sh diagnostics

# 4. TP 多卡 Mooncake 前校验逐卡 HiXLEP endpoint 文件。
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash run_l3_cache_api_tests.sh hixlep-config

# 5. 外部 Mooncake Master 保持运行时，验证当前主路线的 TP8 数据面。
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MOONCAKE_TP_COMM_MODE=global_ub_device \
  bash run_l3_cache_api_tests.sh mooncake-tp
```

统一入口同时接受 `ASCEND_RT_VISIBLE_DEVICES` 和 `ASCEND_VISIBLE_DEVICES`。TP
门禁至少需要两张卡。当前 CANN 9.1.0 环境使用 `global_ub_device`：只生成
`ub_ctp:device` endpoint，并复用该 endpoint 的 Host VA mapping 完成 Host DRAM
segment 与 NPU KV multi-buffer IO。

`mooncake-tp` 的 `global_ub` 模式通过 `ASCEND_GLOBAL_RESOURCE_CONFIG` 请求
HIXL 自动生成 `ub_ctp:device + ub_ctp:host`。CANN 9.1.0 所带 HIXL 在该组合的
跨 segment D2rH 上存在 `register_dev_addr == nullptr` 缺陷；
`global_ub_device` 只生成 `ub_ctp:device`，让 Host segment 走 Device endpoint
的 Host VA mapping；该兼容路径已经通过当前单机 A5 TP8 门禁，是后续框架验证
的默认选择。若获得包含对应 HIXL 修复的正式运行时，再回归
`ub_ctp:device + ub_ctp:host`；`hixlep` 与 `auto` 目前只用于隔离诊断。

`MOONCAKE_TP_ACTIVE_DEVICES` 可在保持整机设备可见的同时只启动指定 worker，
用于定位某张卡或规避全量初始化噪声。例如仅验证 NPU 0 与 NPU 5：

```bash
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MOONCAKE_TP_ACTIVE_DEVICES=0,5 \
MOONCAKE_TP_COMM_MODE=global_ub \
bash run_l3_cache_api_tests.sh mooncake-tp
```

门禁默认把 `comm_resource_config.max_active_channels` 限制为 8，避免为了单条测试
链路预留过多 HComm Thread/Stream/Notify。可通过
`MOONCAKE_TP_MAX_ACTIVE_CHANNELS` 覆盖；该值只服务环境诊断，不代表正式推理的
性能配置。

若门禁报告缺少 `placement=host`，先采集当前镜像具备的生成工具，不要编辑现有
JSON 或复制其他机器的 EID：

```bash
bash run_l3_cache_api_tests.sh hixlep-tooling
```

### A5 URMA 的 AICPU 前置项

`device_urma/device_uboe` 在创建 MemFabric BigMemory 时需要加载：

```text
$ASCEND_HOME_PATH/opp/vendors/cust/op_impl/aicpu/kernel/cann-hybm-compat.tar.gz
$ASCEND_HOME_PATH/opp/vendors/cust/op_impl/aicpu/config/libcann_hybm_kernel.json
$ASCEND_HOME_PATH/opp/vendors/cust/op_impl/aicpu/config/cann_hybm_kernel_version
```

先执行：

```bash
bash run_l3_cache_api_tests.sh memcache-aicpu-check
```

输出含义：

| status | 含义 |
|---|---|
| `READY` | kernel、JSON、版本和 `ascend_package_load.ini` 均完整 |
| `AUTO_PROVISION_AVAILABLE` | wheel 携带构建资源，可执行 `memcache-aicpu-provision` |
| `VERSION_MATCHED_RUN_REQUIRED` | 当前 wheel 较旧，必须安装同 native commit 的独立 run 包 |

当前服务器日志显示 `memfabric_hybrid==1.2.0` 的 native commit 为
`e973ccbb8fa8ea92c9efe0821df731182d86234e`。该版本尚未包含 wheel import 自动
provision，因此应从同一 commit 构建并安装，不能直接使用更新源码生成的 kernel：

```bash
git clone https://gitcode.com/Ascend/memfabric_hybrid.git
cd memfabric_hybrid
git checkout --detach e973ccbb8fa8ea92c9efe0821df731182d86234e
source /usr/local/Ascend/ascend-toolkit/set_env.sh

bash script/kernel/build_ops_run.sh --clean -j 8
./output/memfabric_hybrid_aicpu_kernel.run --version
./output/memfabric_hybrid_aicpu_kernel.run --check
./output/memfabric_hybrid_aicpu_kernel.run --install --install-for-all
```

首次安装不应加 `--force`。也不要预先关闭 A5 验签；只有安装明确报验签失败时，
再按现场安全流程处理。安装后重新执行 `memcache-aicpu-check`，再运行 NPU/GVA
gate。

### 当前 A5 单卡实测状态

在 `Ascend950DT + CANN 9.1.0 + HDK 25.6.rc1` 环境，安装与
`memfabric_hybrid==1.2.0` native commit `e973ccbb...` 匹配的 AICPU run 包后：

| Gate | 结果 |
|---|---|
| AICPU kernel/archive/INI | `READY` |
| MemCache Host H2G/G2H | 通过 |
| MemCache NPU L2G/G2L multi-key/multi-layer | 通过，bitwise 一致 |
| MemCache GVA alloc/copy/publish/lease | 通过，bitwise 一致 |

`e973ccbb...` 在注销用户注册的 HBM buffer 时还会调用
`HalHostUnregisterEx`，驱动可能返回 `ret:8`。该调用失败只记录日志，随后
`UnRegisterMem` 成功且 teardown 继续；更新的 MemFabric 源码已将同一信息从
`ERROR` 下调为 `WARN`。当前将其记录为已知、非阻断的 teardown 告警，但正式
压测仍需观察反复初始化/退出后的资源占用是否稳定。

MemCache Host 成功应同时满足：

```text
[memcache-probe] HOST ROUND TRIP PASSED
[memcache-probe] ALL REQUESTED TESTS PASSED
[memcache-probe] MetaService stopped exitcode=-9
[memcache-probe] explicit cleanup complete; bypassing MemCache v1.2.0 native finalizers
[l3-gate] exit_code=0 gate=memcache_host
```

其中 `exitcode=-9` 是临时 MetaService 的预期退出方式；判定门禁是否通过应看
最末的 `exit_code=0`。

以下为不经过统一入口的等价原始命令，便于单步调试：

```bash
# 1. 只检查 wheel 与 API
python test_memcache.py --mode capabilities
python test_mooncake.py --mode capabilities

# 2. MemCache 服务与 Host IO
python test_memcache.py \
  --mode host \
  --protocol host_shm \
  --dram-size 1GB \
  --max-dram-size 4GB \
  --world-size 8

# 3. MemCache A5 NPU 普通 + GVA API
export ASCEND_RT_VISIBLE_DEVICES=0
python test_memcache.py \
  --mode all \
  --protocol device_uboe \
  --device-id 0 \
  --dram-size 1GB \
  --max-dram-size 256GB \
  --world-size 256

# 4. Mooncake A5 Ascend multi-buffer
python test_mooncake.py \
  --mode all \
  --device-id 0 \
  --local-host 127.0.0.1 \
  --master-address 127.0.0.1:50088
```

若 Mooncake 在 capability 输出完成后出现 glibc heap corruption，先分别运行：

```bash
python -c 'import mooncake.store; print("store-only ok")'; echo $?
python -c 'import mooncake.engine; print("engine-only ok")'; echo $?
python -c 'import mooncake.store; import mooncake.engine; print("store-engine ok")'; echo $?
python -c 'import mooncake.engine; import mooncake.store; print("engine-store ok")'; echo $?
```

任何一条退出码非 0 都属于 wheel/动态库问题；不要继续启动 master 或执行 IO。

统一入口中的等价命令为：

```bash
bash run_l3_cache_api_tests.sh mooncake-imports
bash run_l3_cache_api_tests.sh mooncake-linkage
```

`mooncake-linkage` 会输出 glibc 版本、两个扩展模块路径及完整 `ldd` 结果。
只有四组导入均为退出码 0，且 `ldd` 不含 `not found`，才执行：

```bash
DEVICE_ID=0 bash run_l3_cache_api_tests.sh mooncake-npu
```

若 import 仍然在进程退出时触发 `SIGABRT`，采集两个 native 回溯：

```bash
bash run_l3_cache_api_tests.sh mooncake-gdb-store
bash run_l3_cache_api_tests.sh mooncake-gdb-engine
```

若提示 `gdb is not installed`，先安装与容器系统匹配的 `gdb`，或至少反馈
`mooncake-linkage` 日志。GDB 回溯用于区分 Mooncake 自身析构、CANN/Ascend
transport 清理和同名动态库 ABI 冲突。

`mooncake-imports` 还会对比三种加载顺序：Mooncake 直接加载、先加载并初始化
ACL、先由 `torch_npu` 设置设备。vLLM worker 通常属于第三种；如果只有直接加载
会崩，优先修正 connector 的初始化/退出顺序。若三种顺序全部崩，再评估替换 wheel
或使用当前 CANN 环境源码重编：

```bash
bash run_l3_cache_api_tests.sh mooncake-imports
```

正式 NPU IO gate 不会重复执行这些预期失败的裸 import，而是只检查 vLLM-like
顺序：

```bash
bash run_l3_cache_api_tests.sh mooncake-runtime-import
DEVICE_ID=0 bash run_l3_cache_api_tests.sh mooncake-npu
```

GDB 自身可能在成功捕获 inferior 的 `SIGABRT` 后返回 0；统一脚本会额外解析
`received signal`，将这种情况判为 gate 失败，不能把 GDB 的退出码 0 当成 Mooncake
通过。

探针直接启动 wheel 内的 native `mooncake_master`，并为其创建独立进程组；测试
结束后整组回收。不要经 Python console wrapper 只终止外层 PID，否则 native
master 会继续打印 metrics、持有 `tee` 管道，使命令看起来一直不退出。

Mooncake 的 `--local-host` 最终应换成容器内实际可达 IP；`127.0.0.1` 只用于
单进程、单节点最小尝试。A5 还需按现场选择 UB/UBOE 资源环境。

## 结果边界

探针通过只证明：

- Python wheel 与动态库可用；
- Meta/Master、LocalService/TransferEngine 可建立；
- vLLM-Ascend 使用的地址型 IO API 能正确往返数据。

它不证明：

- Scheduler prefix hash 与 block 生命周期正确；
- 多 TP rank 去重写入正确；
- eager/graph metadata 正确；
- DSA Indexer/MLA component schema 已适配；
- 真实长上下文性能满足要求。

## 后续框架级门禁

当前仅以 Mooncake 为后端，独立 API 通过后依次增加：

1. DSA disabled、单实例 `kv_both`、相同 prompt 两轮 external hit；
2. eager 后 graph；
3. DSA enabled、MTP disabled、Indexer/MLA component key；
4. chunked prefill 每个 chunk 只发布新增 hash-ready 完整块；
5. 受控清理 L1/L2、保留 L3，证明真实 L3 loadback；
6. bsz/mixed stage、多 TP rank owner、容量压力和异步 IO。

所有脚本必须打印实际 backend、key 数、传输方向、字节数和完成状态，避免只凭
聚合 Prefix Cache hit rate 判断路径。

### 真实 Backend 集成门禁

该门禁不加载模型。它启动真实 MetaService/Mooncake Master，并用两个独立 spawn
子进程分别构造 scheduler client 与 NPU worker backend，验证 worker 写入的 key
能被 scheduler 看见，随后经 backend wrapper 读回并逐字节校验：

```bash
DEVICE_ID=0 DEVICE_PROTOCOL=device_urma \
  bash run_l3_cache_api_tests.sh ascend-store-memcache

DEVICE_ID=0 \
  bash run_l3_cache_api_tests.sh ascend-store-mooncake
```

通过标志：

```text
[ascend-store-backend] PASSED backend=memcache ...
[ascend-store-backend] PASSED backend=mooncake ...
[l3-gate] exit_code=0 ...
```

这一级验证 `MemcacheBackend/MooncakeBackend` 的真实配置加载、scheduler 零容量
client、worker buffer 注册、返回码归一化与跨进程 key 可见性。通过后再进入
DSA disabled 的完整 `AscendStoreConnector + 模型` 冷/热两轮测试。

完整基线验证入口见
[baseline_e2e/README.md](baseline_e2e/README.md)。外部 Master/MetaService 由用户
启动，vLLM-Ascend 在 scheduler/worker 内分别创建 metadata client 与嵌入式
Backend/LocalService；vLLM 不会自动部署第三方 L3 控制面。

MemCache v1.2.0 即使显式完成 buffer 注销和 `store.close()`，Python 正常退出仍
可能在 native `ResourceTracker/atexit` 阶段触发 heap corruption 或挂住。Backend
集成探针的 MemCache 子进程在显式清理后使用 `os._exit()`，避免把已通过的数据面
误判为失败。这不是生产修复；正式服务的优雅停止仍要求升级/修复 MemCache native
生命周期，并为 `MemcacheBackend` 接入幂等 shutdown hook。
