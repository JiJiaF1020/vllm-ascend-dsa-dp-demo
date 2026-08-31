# DSA 稀疏卸载后续开发 RoadMap

> - 最后更新：2026-09-01
> - 基线：vLLM-Ascend v0.23.0 `5cb98caa`
> - 当前实现基点：DSA custom `4c979ffd`
> - 当前阶段：P1-HOT 妥协版开发完成，等待成对 HBM-only backfill 上板

## 1. 文档定位

本文只维护当前项目从“可运行原型”走向完整分层缓存与调度能力的开发顺序、
阶段边界和退出门禁。已经落地的数据结构、所有权和算子 ABI 以
{doc}`dsa_offload_design` 为真源；L2/L3 job、schema 和后端细节以
{doc}`dsa_l2_l3_cache_design` 为真源。

优先级遵循以下原则：

1. 先关闭已有能力的生命周期和长稳，再叠加新 cache tier；
2. 先建立 component-wise 正确性合同，再做 layerwise/direct 性能优化；
3. 先完成单实例状态恢复，再进入跨实例 P/D 分离；
4. 当前妥协版 MTP3 可用时，不让理想版算子重写阻塞基础缓存能力。

## 2. 当前完成位置

| 原 RoadMap 工作项 | 状态 | 当前边界 |
|---|---|---|
| scheduler 统一管理 DRAM block | 完成 | scheduler 是 physical-id/hash/refcount 真源 |
| 完整 Indexer+MLA LocalDRAM bundle | 完成 | worker 只持固定 arena 与 assignment 镜像 |
| LocalDRAM prefix cache | 第一版完成 | complete bundle 最长连续命中与分平面 restore |
| L2→L3 write-behind | P1-W 完成 | TP0 异步 put、reuse fence、shutdown drain |
| L3→L2→HBM loadback | P1-R 完成首版 | no-MTP、Mooncake、同步 all-rank get、fail-fast |
| HBM target prefix | P1-HOT 妥协版待上板 | HBM pair 可越过 L2；L3 固定经 L2；首 chunk 补 L2 gap |
| MTP 多级 prefix | 未开始 | L3 当前显式拒绝 MTP |
| preemption/resume | 未开始 | scheduler 当前显式失败 |
| P/D 分离 | 未开始 | 等待 tier/source 与恢复合同稳定 |
| 理想版动态 MTP | 未开始 | 当前固定 MTP3、greedy、双尾可用 |
| MTP proposer graph | 未开始 | target uniform Q=4 FULL graph 已完成 |

## 3. P0：L3 P2 稳态关闭

该阶段不扩展数据格式，目标是证明 P1-W/P1-R 在在线服务生命周期中稳定。
统一验收入口位于 `examples/dsa_demo/online_l2_l3_closeout_test.py`。
A5 TP8 graph 已完成串行 Local-L2/L3、L3/L3 与三轮混合 `bsz=4`
continuous batching：串行跨介质分叉均能由同介质相同签名复现，并发差异在
Local-L2 自对照中已经广泛存在，没有形成 L3 独有错误证据。按项目进度将
L3 数据路径阶段性收口；graph token 确定性、eager 严格回归、FD 长稳和
external namespace 物理 GC 保留为独立遗留，不作为 P1-HOT 阻塞项。

### 工作项

- 在线 continuous batching，覆盖 5K/8K/40K/70K 混合请求；
- cold、Local-L2、清本地后的 L3 三轮 prefix replay；
- 请求完成、InputBatch row reuse 和 L2 physical-id reuse；
- pending load/store completion 联合 drain；
- Mooncake FD、TCP connection pool、segment teardown 与长稳；
- external reset 的 namespace 生命周期与远端对象回收；当前只做逻辑轮换，
  测试默认复用 namespace，物理 GC 作为本阶段遗留项；
- lookup 后对象被淘汰时保持 all-rank fail-fast；
- 收敛 zero-token polling 与 shutdown grace-period 延迟；
- 普通 DSA DP 冒烟；L3 继续限制 `data_parallel_size=1`。

### 退出门禁

- bsz=1/eager Local-L2 与 L3 输出严格一致；
- component payload bitwise、hit blocks、all-rank get 和 restore 全部成功；
- bsz>1 graph 差异必须能由 Local-L2/Local-L2 对照复现，不能只凭文本放宽；
- 无丢失 completion、stale job、refcount underflow、FD 单调增长或 pending job；
- 不轮换 namespace 时，相同 prompt 的重复测试不得重复写入相同完整前缀；
- 请求 finish 不等待普通 write-behind，但 reuse/shutdown 必须正确 drain。

## 4. P1：HBM target prefix 与 component source

这是完成 P0 后的默认下一功能优先级。

当前妥协版先实现完整 HBM pair：无 MTP 时，Indexer 与 dense MLA 同时命中的
公共前缀直接成为请求 block table 前段，并允许比 L2 complete bundle 更长。
L3 仍统一经 L2 恢复，不引入 direct-HBM 或独立 component mask。

当前准确分层关系是：

```text
local_l2_blocks = complete L2 prefix
hbm_pair_blocks = min(HBM Indexer, HBM dense MLA)
local_blocks = max(local_l2_blocks, hbm_pair_blocks)
external_l3_blocks = local_blocks 之后的连续 L3 suffix

[0, hbm_pair)                  直接复用 HBM pair
[hbm_pair, local_l2)           L2 -> HBM restore（若 L2 更长）
[local_l2, hbm_pair)           首个 prefill forward 执行 HBM -> L2 backfill
[local_blocks, local_blocks+L3) L3 -> L2 -> HBM
```

当 HBM pair 越过 L2 时，scheduler 先为缺口分配 L2 physical IDs，但 ready
水位停留在原 L2 边界。即使更靠后的 L3 suffix 已写入 L2，也不能越过中间
缺口发布 hash；首个 prefill forward 的既有 full-block dump 会把缺口、新满块
以及必要的连续后缀一起物化，worker 成功返回后才统一推进 ready。

本阶段不可放松的最终不变量：请求释放任何 dense MLA block 或进入
`ENTER/SPARSE` decode 前，L2 ready 前缀必须覆盖请求全部
`durable_full_blocks`，且每个逻辑块同时包含完整 Indexer+MLA payload。HBM
命中只减少 restore，不能代替 durable backing；当前 HBM-only 命中必须先
完成 backfill，再推进 ready 水位和释放 HBM。

当前退出门禁以完整 HBM pair 为准，不要求 Indexer-only/MLA-only 去重。
逐 block/component source mask、L3 direct-HBM 与 layerwise async 统一后移到
性能/研究阶段，不能阻塞 MTP 多级 prefix、preemption 和 P/D 主线。

### 工作项

1. 为 resident HBM block 建立 `dense-context`、`sparse-budget`、`tail`
   角色，只有 dense-context MLA 可参与 prefix hash；
2. Indexer HBM 始终保留完整上下文语义，但 lookup 后必须通过 refcount/pin
   防止 admission 前被重分配；
3. scheduler 独立查询 HBM pair 与 L2 complete bundle，并以较长者作为本地
   effective prefix；
4. L3 从 effective prefix 后查询，并固定执行 `L3 -> L2 -> HBM`；
5. HBM-only hit 通过首个 forward 回填 L2，ready/hash 只在 worker 成功后推进；
6. eager/graph 继续共用同一 projection、HBM block table 与 runtime owner。

### 退出门禁

- HBM complete hit、HBM>L2、L2>HBM、HBM-only+L3 suffix；
- HBM block 在 lookup/admission 事务内不被复用；
- HBM→L2 backfill 不提前发布 hash，L3 suffix 不跨越未 ready 缺口；
- 同一 prompt 在 tier 组合变化后 component bytes 与最终输出满足既有门禁；
- 无命中 steady decode 不增加逐层 Python 或 tensor 分配。

## 5. P2：MTP 多级 prefix

先支持同进程 HBM/L2 公共命中，再扩展 L3。

- target Indexer、target MLA 与 MTP full-HBM cache 取公共长度；
- provisional MTP token 永远不发布 hash；
- L3 方案需在“第三个 MTP component”与“恢复 target、重算 proposer prefix”
  之间冻结一个精确合同；
- MTP cache 未恢复时不能仅凭 target L3 hit 跳过 prefill；
- 保持固定 MTP3、greedy 和双尾，不在本阶段引入动态验算窗口。

## 6. P3：preemption/resume

- preempt 时释放 HBM，但保留或重新获取 L2/L3 durable 引用；
- 丢弃 provisional tail，恢复后重新建立 resident `cache_slots` first-fill；
- 请求恢复时重新执行 component admission，不能复用陈旧 physical id；
- 覆盖多次 preempt、DRAM eviction、L3-only restore 和 MTP/no-MTP。

该阶段是 P/D 分离前的单实例恢复门禁。

## 7. P4：P/D 分离

- Prefill 端输出完整 Indexer+MLA component；
- Decode 端接管完整 Indexer、MLA durable prefix 与必要的 proposer cache；
- 同时支持 connector 直传和 L3 中转；
- Decode 完成 first-fill 后进入固定 resident sparse decode；
- decode 新增满块继续发布到 L3，支持后续多轮前缀复用。

## 8. P5：性能与研究项

这些项目不应先于上述正确性合同，内部顺序由实际 KPI 决定：

- L3 layerwise async load 与逐层 wait，优化 TTFT；
- HBM↔L3 direct，面向 Prefill/P-D 特定部署，不能破坏 L2 durable backing；
- Mooncake 批量/并发 IO、zero-token polling 和连接预热；
- MTP proposer graph；
- 理想版动态 MTP：运行时 union 容量决定有效验算长度，联动 sampler/rejector；
- MemCache 正式后端、多节点、CP/PP、KV metrics/events 和 L3 retention。

## 9. 默认推进顺序

```text
L3 P2 在线/长稳收口
  -> HBM target prefix + L2 backfill/去重
  -> MTP 多级 prefix
  -> preemption/resume
  -> P/D 分离
  -> layerwise/direct/图模式/理想 MTP
```

任何阶段若需要临时删除启动期拒绝，必须先补齐相应数据所有权、失败语义和
回归门禁；不能把“请求能跑”当作该能力已经支持。
