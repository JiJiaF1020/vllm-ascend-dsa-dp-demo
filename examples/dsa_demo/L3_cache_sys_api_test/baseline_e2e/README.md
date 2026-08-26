# DSA Disabled Baseline L3 End-to-End Smoke

本目录验证社区 vLLM/vLLM-Ascend 的原生 `AscendStoreConnector` 能否在不开启
DSA 时对接 Mooncake 或 MemCache。测试刻意固定为：

- `kv_role=kv_both`；
- eager；
- async scheduling 关闭；
- vLLM HBM prefix caching 关闭；
- chunked prefill 关闭；
- MTP 关闭；
- 不传 `additional_config` 或 `dsa_sparse_config`；
- 一个多满块 prompt，第一轮冷写，等待 10 秒后第二轮热读；
- 每轮只生成一个 token，减少与 L3 验证无关的 decode 开销。

## 文件

| 文件 | 用途 |
|---|---|
| `configs/mmc-meta.conf` | MemCache MetaService 配置 |
| `configs/mmc-local.conf` | vLLM worker 内嵌 MemCache LocalService 配置 |
| `configs/mooncake.json` | Mooncake Backend 配置 |
| `start_memcache_meta.sh` | 前台启动 MemCache MetaService |
| `start_mooncake_master.sh` | 前台启动 native Mooncake Master |
| `serve_baseline_l3.sh` | DSA disabled 在线 vLLM 服务 |
| `online_baseline_l3.py` | 对在线服务发送两轮相同请求 |
| `offline_baseline_l3.py` | 同一 `LLM` 内执行两轮离线 generate |
| `baseline_prompt.py` | 在线/离线共享的确定性长前缀 |

当前单机 A5 仍优先推进 Mooncake `global_ub_device`。独立探针已通过 TP8，证据见
[Mooncake A5 TP 记录](../MOONCAKE_A5_TP_BLOCKER.md)。

MemFabric 临时包支持按进程设置 `USE_LOCAL_EID` 后，MemCache 已通过单机 TP8
完整建链和 1 MiB 环形 remote-get payload 校验。本目录现通过
`kv_connector_extra_config.memcache_local_eid_map_path` 在每个 vLLM worker
初始化时选择物理卡 EID，可进入 DSA-disabled baseline E2E。详见
[MemCache A5 TP 记录](../MEMCACHE_A5_TP_BLOCKER.md)。

本目录的在线、离线样例会设置
`kv_connector_extra_config.log_transfer_summary=true`。每次真实外部 KV load
完成后，各 TP rank 都会输出一条 `[KVPOOL-IO]`：只有 `status=success`、
`returned=keys` 且 `failed=0` 才表示后端 `get` 成功返回。`elapsed_ms` 包含
Mooncake 首次访问远端 segment 时的懒建链开销，后续请求可用它区分首次建链与
稳态数据传输。

## 前置条件

先完成上一级独立 API 与真实 Backend gate。MemCache 还必须安装与 wheel native
commit 匹配的 `memfabric_hybrid_aicpu_kernel.run`，并按
[`CURRENT_TEST_CHECKLIST.md`](../CURRENT_TEST_CHECKLIST.md) 生成
`/tmp/memcache-local-eids.json`。该映射只适用于当前实验节点。

默认配置仅面向当前单机 A5：

```text
Mooncake Master  127.0.0.1:50088
Mooncake Metrics 127.0.0.1:9003
MemCache Meta    127.0.0.1:5000
MemCache Config  127.0.0.1:6000
MemCache Metrics 127.0.0.1:18000
vLLM Online      127.0.0.1:8035
```

在线/离线样例默认暴露 NPU `0..7` 并采用 TP8。MemCache 使用
`device_urma`，每个 worker 内嵌一个 LocalService，并向同一个外部 MetaService
注册 1 GiB DRAM；worker 在创建 LocalService 前按物理 NPU ID 从 EID 映射文件
设置进程级 `USE_LOCAL_EID`。scheduler 的 `init_bm=False` metadata client 不设置
EID，也不创建数据 channel。

在线和离线脚本的 `MOONCAKE_COMM_MODE` 默认均为 `global_ub_device`。它在首次导入
vLLM 前设置 `ASCEND_GLOBAL_RESOURCE_CONFIG`，仅请求 `ub_ctp:device` endpoint，
并将 `max_active_channels` 限制为 8；同时固定单机 `VLLM_HOST_IP=127.0.0.1`。

Master 保持运行时，先用不加载模型的 TP8 门禁验证跨 segment 通道：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MOONCAKE_TP_COMM_MODE=global_ub_device \
  bash ../run_l3_cache_api_tests.sh mooncake-tp
```

只有输出
`CROSS-SEGMENT ROUND TRIP PASSED` 才继续拉起完整模型。

单卡探针在 `auto` 下通过，只能证明一个 TransferEngine 的本地 IO；TP8 的复制
MLA cache 在 L3 只保存一份，写入按 rank 分片、恢复时需要跨 segment 传输，因此
必须显式构造所有 worker 一致可达的 endpoint。仅设置
`ASCEND_LOCAL_COMM_RES={"version":"1.3"}` 不能替代这些硬件拓扑信息。`uboe`
模式仍保留用于现场明确配置了 UBoE 的环境，`auto` 仅用于隔离诊断。

Mooncake Store 的 `local_buffer_size` 和非 fabric `global_segment_size` 会注册 Host
内存，显式 KV multi-buffer IO 还需要 Device endpoint。CANN 9.1.0 的混合
Device+Host endpoint 路径存在已知 D2rH 缺陷；当前验证通过的兼容方式是仅生成
`ub_ctp:device`，让 Host segment 复用 Device endpoint 的 Host VA mapping。

Mooncake 每个 worker 默认贡献 1 GiB global segment 和 1 GiB local buffer。
MemCache 每个 worker 默认贡献 1 GiB DRAM；`max.dram.size=1TB` 与
`world_size=33` 构造 33 TiB 逻辑 GVA envelope，以满足 `device_urma` 总池空间
必须大于 32 TiB、启用 56-bit GVA 的约束。当前只有 8 个 active worker，实际仍
只申请 8 GiB DRAM。修改 `world_size` 后必须重启 MetaService。

## MemCache 在线

终端 1：

```bash
cd examples/dsa_demo/L3_cache_sys_api_test/baseline_e2e
bash start_memcache_meta.sh 2>&1 | tee memcache-meta.log
```

终端 2：确认 `serve_baseline_l3.sh` 顶部为：

```bash
L3_BACKEND="memcache"
```

脚本默认读取 `/tmp/memcache-local-eids.json`。使用其他路径时设置：

```bash
export MEMCACHE_LOCAL_EID_MAP_PATH=/path/to/memcache-local-eids.json
```

然后运行：

```bash
bash serve_baseline_l3.sh 2>&1 | tee baseline-memcache-online.log
```

终端 3：

```bash
python online_baseline_l3.py
```

## Mooncake 在线

终端 1：

```bash
cd examples/dsa_demo/L3_cache_sys_api_test/baseline_e2e
bash start_mooncake_master.sh 2>&1 | tee mooncake-master.log
```

终端 2：编辑 `serve_baseline_l3.sh`：

```bash
L3_BACKEND="mooncake"
```

并保持：

```bash
MOONCAKE_COMM_MODE="global_ub_device"
```

然后运行：

```bash
bash serve_baseline_l3.sh 2>&1 | tee baseline-mooncake-online.log
```

终端 3 仍执行：

```bash
python online_baseline_l3.py
```

## 离线

离线测试同样要求终端 1 中的对应 Master/MetaService 保持运行。编辑
`offline_baseline_l3.py` 顶部的 `MODEL_PATH`、`L3_BACKEND`、
`MEMCACHE_LOCAL_EID_MAP_PATH` 与 TP 配置，然后：

```bash
python offline_baseline_l3.py 2>&1 | tee baseline-l3-offline.log
```

不要同时运行在线服务和离线脚本；二者默认使用相同的 NPU 和 connector IPC
标识。样例将 `shutdown_timeout` 设置为 30 秒，使 EngineCore 在请求完成后走 drain
teardown；不要改回默认的 0 秒 abort，否则 Mooncake native 资源可能在 worker
被强制 SIGTERM 时留下 ZMQ、semaphore 或 shared-memory 告警。

### 冷热输出不一致诊断

离线脚本默认执行三轮，并打印首个生成位置的 top-5 logprobs。出现冷/热 token
不一致时，使用完全相同的模型、prompt 和采样参数分别运行三组独立进程：

| 对照 | `L3_BACKEND` | `ENABLE_PREFIX_CACHING` | 目的 |
|---|---|---|---|
| 全量重算 | `"none"` | `False` | 判断模型重复执行本身是否稳定 |
| HBM prefix | `"none"` | `True` | 判断跳过已缓存前缀后的基线数值路径是否稳定 |
| MemCache | `"memcache"` | `False` | 验证外部恢复相对 HBM prefix 是否引入额外差异 |

每组测试都重新启动 Python 进程。MemCache 组还应重启 MetaService，避免沿用上轮
对象。若全量重算稳定、HBM prefix 与 MemCache 都以相近 logprobs 发生相同变化，
优先判断为缓存命中改变 prefill shape 后的数值边界；若仅 MemCache 明显偏离 HBM
prefix，则继续定位外部 KV 的保存、恢复或布局问题。

GLM-5.2 的 78 个目标层中，前 2 层各有独立 Indexer，后续 76 层每 4 层共享一
个 full Indexer。因此 AscendStore 注册摘要应体现 78 个 MLA 物理层和 21 个
Indexer 物理层。shared 层复用的 topK buffer 是单轮 forward 临时状态，不属于
L3 持久化对象；若日志把 78 层都登记成独立 Indexer cache，才说明基线 spec 或
注册布局存在问题。

当前 GLM-5.2 W4A4C8 MXFP4 实测中，全量重算三轮完全一致；HBM prefix 与
MemCache 的两次热重放分别稳定，并且两种热路径的 top-5 token、排序和 logprob
到 7 位小数完全一致。两种 prefix-hit 路径都与冷路径不同，因此该差异归属于
GLM-5.2 基线 prefix-hit 计算路径，而不是 MemCache 引入的额外偏差。离线脚本将
热重放稳定作为 L3 功能门禁，同时继续打印冷/热差异，不能据此宣称模型 prefix
精度问题已经解决。

## 验收

客户端/离线脚本必须打印：

```text
round=1 ...
round=2 ...
OUTPUT CONSISTENCY PASSED
```

并满足：

1. 两轮输出 token IDs 一致；若在线 OpenAI SDK 丢弃扩展字段，则文本一致；
2. 第一轮结束后 L3 中出现对象，第二轮发生 external lookup/load；
3. vLLM 日志无 Backend put/get 错误和 load failure recompute；
4. 不出现 DSA HBM/DRAM capacity report，也不出现 DSA 算子日志；
5. Mooncake Master 的 keys/Put/Get 指标增长，或 MemCache 日志出现对应对象操作；
6. 第二轮请求仍会重算不完整尾块，不能把“未做到 100% token 跳过”误判为 miss。

仅凭两轮结果一致不能证明 L3 命中。样例使用
`kv_load_failure_policy=fail`，避免已选择的外部 KV 在加载失败后静默重算，但若
第一轮根本没有成功发布对象，第二轮仍可能按 cold miss 正常重算。因此首次框架
联调必须同时保留 vLLM 与 L3 服务完整日志，并确认 external hit 非零；若日志证据
仍不够，再给 AscendStore 增加短期、定向的 INFO 探针。

MemCache v1.2.0 的数据面已通过，但其 native 正常退出仍有
`ResourceTracker/atexit` 生命周期缺陷。它不阻塞常驻服务冒烟，不能视为生产级
优雅停止已经通过。
