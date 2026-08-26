# L3 当前测试清单

本文记录 2026-08-25 起基于以下环境继续执行的最短 L3 门禁：

- 单机 8 x Ascend 950DT；
- CANN 9.1.0；
- PyTorch 2.10.0；
- `torch_npu==2.10.0.post6`，`git_version=a8f2f4fda83732c0d02244a46815a81f21fe3439`；
- vLLM-Ascend 当前 DSA custom-official 分支。

当前只验证社区原生 `AscendStoreConnector` 与 MemCache/Mooncake 的基础链路。
在本清单通过前，不运行 DSA Indexer/MLA 对接 L3 的测试。

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
- [ ] DSA-disabled Mooncake 完整模型冷/热两轮。

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

必须输出：

```text
CROSS-SEGMENT ROUND TRIP PASSED
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
