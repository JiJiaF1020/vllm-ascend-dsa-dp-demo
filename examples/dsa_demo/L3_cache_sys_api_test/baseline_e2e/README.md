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

当前单机 A5 的主路线已切换为 Mooncake `global_ub_device`。独立探针已通过
TP8，但本目录的在线/离线包装脚本仍保留历史 `memcache/hixlep` 默认值，下一步
需要先收敛通信模式，再运行完整模型。证据见
[Mooncake A5 TP 记录](../MOONCAKE_A5_TP_BLOCKER.md)。

MemCache 两卡最小门禁已确认受单机内部 fullmesh 选路边界限制；其命令只保留
给多服务器超节点或上游适配后的回归，不再作为本机完整模型候选。详见
[MemCache A5 TP 记录](../MEMCACHE_A5_TP_BLOCKER.md)。

## 前置条件

先完成上一级独立 API 与真实 Backend gate。MemCache 还必须安装与 wheel native
commit 匹配的 `memfabric_hybrid_aicpu_kernel.run`。

默认配置仅面向当前单机 A5：

```text
Mooncake Master  127.0.0.1:50088
Mooncake Metrics 127.0.0.1:9003
MemCache Meta    127.0.0.1:5000
MemCache Config  127.0.0.1:6000
MemCache Metrics 127.0.0.1:18000
vLLM Online      127.0.0.1:8035
```

在线/离线样例默认暴露 NPU `0..7` 并采用 TP8。MemCache 历史路径使用
`device_urma`，每个 worker 内嵌一个 LocalService，并向同一个外部 MetaService
注册 1 GiB DRAM；该路径当前不用于单机 TP8。

历史脚本的 `MOONCAKE_COMM_MODE` 默认是 `hixlep`：样例先检查
`/etc/hixlep/ub_endpoint_npu_<physical-id>.json`，再通过
`kv_connector_extra_config.ascend_local_comm_res_path` 让每个 A5 worker 加载
与物理卡对应的完整 `ASCEND_LOCAL_COMM_RES`。可用以下命令先做门禁：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash ../run_l3_cache_api_tests.sh hixlep-config
```

Master 保持运行时，先用不加载模型的 TP8 门禁验证跨 segment 通道：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MOONCAKE_TP_COMM_MODE=global_ub_device \
  bash ../run_l3_cache_api_tests.sh mooncake-tp
```

当前应显式使用 `MOONCAKE_TP_COMM_MODE=global_ub_device`。只有输出
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

## MemCache 在线（仅保留回归资料）

终端 1：

```bash
cd examples/dsa_demo/L3_cache_sys_api_test/baseline_e2e
bash start_memcache_meta.sh 2>&1 | tee memcache-meta.log
```

终端 2：确认 `serve_baseline_l3.sh` 顶部为：

```bash
L3_BACKEND="memcache"
```

然后运行：

```bash
bash serve_baseline_l3.sh 2>&1 | tee baseline-memcache-online.log
```

终端 3：

```bash
python online_baseline_l3.py
```

## Mooncake 在线（当前主路线，包装脚本待收敛）

终端 1：

```bash
cd examples/dsa_demo/L3_cache_sys_api_test/baseline_e2e
bash start_mooncake_master.sh 2>&1 | tee mooncake-master.log
```

终端 2：编辑 `serve_baseline_l3.sh`：

```bash
L3_BACKEND="mooncake"
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
`offline_baseline_l3.py` 顶部的 `MODEL_PATH`、`L3_BACKEND` 与 TP 配置，然后：

```bash
python offline_baseline_l3.py 2>&1 | tee baseline-l3-offline.log
```

不要同时运行在线服务和离线脚本；二者默认使用相同的 NPU 和 connector IPC
标识。

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
