# Mooncake A5 TP 阻塞记录

本文记录 2026-08-24 在单机 Ascend 950DT 上验证 vLLM-Ascend 原生
`AscendStoreConnector + MooncakeBackend` 时遇到的运行时问题及兼容结论。历史
阻塞已经由 `global_ub_device` 绕开，并通过单机 TP8 轻量数据面门禁；本文保留
失败证据，避免后续重新走回不兼容配置。

## 环境基线

| 组件 | 版本或状态 |
|---|---|
| 硬件 | 单机 8 x Ascend 950DT |
| HDK | `25.6.rc1` |
| CANN | `9.1.0` |
| Mooncake NPU wheel | `0.3.11.post1` |
| Mooncake source | `e9c6107572` |
| vLLM-Ascend baseline | `1471b7702767ac351c8955fa286b4913b631bdd7` |
| 测试形态 | DSA disabled，`AscendStoreConnector`，`kv_role=kv_both` |

## 已通过能力

以下结果说明 Mooncake Python 包、Master 和基础数据 API 并非整体不可用：

1. `torch_npu` 初始化后导入 `mooncake.store/engine` 正常。
2. 单卡 `TransferEngine`、Host 对象和 NPU multi-buffer 往返通过。
3. 真实 `MooncakeBackend` scheduler/worker 跨进程 key 可见和逐字节回读通过。
4. TP8 首轮中 Master 能注册 8 个 client/segment；部分 rank 能完成 Host buffer、
   global segment、NPU buffer 注册和本地 `put=[0]`。

## 阻塞现象

### 逐卡 D2D LocalCommRes

平台 `/etc/hixlep/ub_endpoint_npu_*.json` 均为 `version=1.3`，但只有
`placement=device`。Mooncake 的非 fabric Host DRAM segment 需要注册 Host
地址，因此初始化报错：

```text
no endpoint matches mem type:1 for register mem
Failed to register mem, type:host
```

这说明平台默认 D2D 文件不能直接作为 Host DRAM Store 的完整 LocalCommRes，
并不说明文件自身损坏。

### GlobalResourceConfig 自动生成

使用以下协议范围后，Host/Device 地址注册能够成功：

```json
{
  "comm_resource_config.protocol_desc": [
    "ub_ctp:device",
    "ub_ctp:host"
  ]
}
```

但多进程初始化不稳定：同一轮中部分 rank 成功，部分 rank 在本地
`HcommEndpointCreate` 失败；反复运行后失败 rank 增多。定向两卡也可能在
`TransferEngine.initialize` 阶段同时失败，尚未进入远端 get。

关键 plog：

```text
HcommProxy::EndpointCreate(...) failed, ret: 0x1
HcommProxy::ThreadAlloc(...) failed, ret: 0xF
```

其中 `0xF` 对应 `HCCL_E_RUNTIME`，失败位于 HComm Thread/Stream/Notify 资源
申请路径，不是 Mooncake Master、对象容量或远端网络连接。

## 当前判断

| 假设 | 证据 | 状态 |
|---|---|---|
| HIXL 初始化失败未回滚已创建 endpoint | 首轮部分成功，重复后失败增多；官方后续提交明确修复同类泄漏 | 高概率 |
| 共享节点 HComm 资源压力 | 采样时 NPU6 有两个 Python 业务进程且利用率 100%，NPU4-7 为 Warning | 未排除 |
| NPU5 单卡故障 | 后续失败 rank 变化，定向测试中 NPU0 与 NPU5 同时失败 | 基本排除 |
| Mooncake Master 或容量不足 | Master ready、8 GiB segment 注册成功、部分 key 写入成功 | 排除 |
| DSA 框架逻辑 | 轻量探针不加载模型，也不启用 DSA | 排除 |

## 2026-08-25 NPU 0、1 跨 segment 复测

在 TorchNPU post6、NPU 0、1 均健康的环境中，`global_ub` 模式已完成两个
TransferEngine 初始化、Host/Device 内存注册、segment mount 和 local put；双向
remote get 均返回 `-800/TRANSFER_FAIL`。首个明确错误为：

```text
remote_region.register_dev_addr is nullptr
ConvertHostMappedDescs failed
```

HIXL 主线提交
[`4fdd7c8`](https://gitcode.com/cann/hixl/commit/4fdd7c84e2cd46291d4db0e5c1ec14ee9616eced)
给出了完全一致的根因：同时配置 Device 和 Host UBC CTP endpoint 时，D2rH
匹配 Device→Host endpoint，客户端却仍按 Device→Device 规则要求远端 Host VA
映射；Host endpoint 注册不会生成该地址。该修复晚于当前 CANN 9.1.0 HIXL。

在获得修复包前，使用 `global_ub_device` 仅生成 Device endpoint，验证 HIXL
现有 Host VA mapping 兼容路径；这只是运行时门禁，不代表最终部署必须采用该
配置。

### Device-only 兼容路径结果

`global_ub_device` 在健康的 NPU 0、1 上通过：

- 两个 TransferEngine、Host segment 与 NPU buffer 初始化成功；
- 两端 local put 均成功；
- 两端分别从对端 segment remote get 1 MiB；
- 返回长度均为 1,048,576 bytes，逐字节内容校验通过；
- 两个 ADXL engine 正常断开。

这说明 Mooncake、Master、跨进程 segment 寻址和 A5 Host↔NPU 搬运能力可用；
阻塞只存在于当前 HIXL 的 Device+Host 混合 endpoint 选择逻辑。当时 Mooncake
路线恢复为“有兼容配置、待多卡扩展验证”，随后依次完成 NPU 0-3 和 TP8 门禁。

NPU 0-3 四卡兼容路径随后通过。首次 NPU 0-7 测试发生在共享节点：device0 在
`TransferEngine.initialize` 内的 `RunAicpuThreadInit/Hcomm ThreadAlloc` 阶段
失败，device6 未在超时前返回，其余 rank 因 barrier 被打断而退出。该轮没有
执行 remote get，且同配置此前已在包含 device0 的四卡场景通过，因此标记为
共享 HComm/AICPU 资源竞争下的无效结果，等待独占空闲节点复测，不作为 Mooncake
TP8 功能失败证据。

随后在资源状态恢复后以完全相同配置复测 TP8：8 个 rank 均完成 local put，
并按环形关系从下一 rank remote get 1 MiB，8 份返回长度与内容校验全部通过，
最终正常断开 ADXL engine。由此确认 `global_ub_device` 已通过当前单机 A5 TP8
轻量数据面门禁；前一轮失败属于共享节点的瞬时 HComm/AICPU 资源竞争。

HIXL 官方提交
[`f99216e`](https://gitcode.com/cann/hixl/commit/f99216e6a1b8b50873783091ea432f819093e3df)
在 `HixlCSServer::Initialize` 失败时增加 `Finalize()` 回滚，并明确说明旧路径会
泄漏 hcomm endpoint。该提交当前晚于 `v9.2.0-beta.1`，尚未包含在已发布 tag
中，因此不能直接建议用某个已发布版本替换当前 CANN 9.1.0。

## 当前推进边界

Mooncake A5 TP 当前标记为 **轻量数据面与 DSA-disabled 框架 E2E 已通过，进入
DSA split-cache schema 验证**：

- CANN 9.1.0 下固定使用已验证的 `global_ub_device`；
- 不把未发布 HIXL master 动态库覆盖到现有 CANN；
- 共享节点出现初始化失败时，先排除 HComm/AICPU 资源竞争；
- `ub_ctp:device + ub_ctp:host` 仅在获得正式修复运行时后回归；
- MemCache 的单机选路限制不再阻塞 Mooncake 路线。
- DSA-disabled GLM-5.2 TP8 已确认 3200/3287 external hit，所有 rank 的
  `batch_get_into_multi_buffers` 均为 25/25 keys、`failed=0`；首次热请求慢于
  冷算来自 segment 懒建链，不是超时回算。

## 下一步

1. 用 DSA A5 的 21 full Indexer + 78 packed MLA 真实 component schema 完成
   TP8 HBM multi-buffer 往返；
2. 验证 swapped-memory L2 arena 是否可由 `global_ub_device` 直接 put/get；
3. 打通 DSA 本地 L2 hit 后继续查询 Mooncake，并将 L3-only suffix 直接恢复到
   HBM、经同轮 full-block dump 物化回 L2；
4. 增加 rank0 异步 L2->L3 write-behind，再覆盖 eager/graph、在线/离线和
   chunked prefill 回归。
