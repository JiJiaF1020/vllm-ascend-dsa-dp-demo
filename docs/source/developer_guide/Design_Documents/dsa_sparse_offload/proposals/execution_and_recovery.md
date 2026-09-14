# 分布式执行与状态恢复设计草案

> 状态：目标与部分原则已确认，实施方案未定稿。覆盖打断恢复、跨机DP及相关加载优化，不是当前支持矩阵。
> PD已于2026-09-14转入[独立详细设计草案](prefill_decode_disaggregation.md)，本页不再维护第二份PD协议。

## 1. 共同不变量

复用v0.23的请求生命周期、connector完成协议与图机制；按已有资源owner扩展，不新增组合专属runtime。跨实例/节点移动的是明确语义的payload及接管状态，不是把本进程physical block id原样发送过去。

无论传输路径如何，Decode使用重要token前必须有完整Indexer及足够durable backing，MTP还要明确proposer和drafting边界。最长最热优先复用不能豁免这些条件。

## 2. 请求打断与恢复

### 2.1 四种不同的“恢复”

| 机制 | 触发及行为 | 与本草案的关系 |
|---|---|---|
| 请求级 preemption/resume | 本 engine HBM 扩表失败，释放 victim 的块，保留 token 历史，重新入队 | DSA 尚未支持，本文主要研究对象 |
| DP wave pause/wake | MoE 全域空闲后停波，新请求重新唤醒，idle 域仍按协议陪跑 | 已有基线执行协同，不恢复请求 KV |
| 管理 pause/resume | 暂停新 admission，排空 running/IO，排队请求保留 | 单机 DP 已适配，见[DP 管理合同](../features/data_parallel_inference.md)，不等于 preemption |
| Ascend `RecomputeScheduler` | PD Decode consumer 容量不足时交回代理重算或调用 connector preempt hook | 专用 PD 语义，当前 DSA 拒绝组合，不可直接作为普通 DP 恢复实现 |

### 2.2 基线 victim 与 resumed block table

vLLM v0.23 同步 scheduler 在 running 请求扩展 block table 失败时，选择低优先级
或 running 尾部 victim。它调用 `kv_cache_manager.free(request)` 释放请求全部
HBM KV blocks，释放 encoder cache、移出 inflight prefill，改为 `PREEMPTED`，
令 `num_computed_tokens=0` 并清 speculative tokens。Prompt、已生成 output token
IDs 保留，请求 prepend 回 waiting。再次 admission 时重新查本地/外部 prefix，
为未命中部分重算；`CachedRequestData.resumed_req_ids` 通知 worker 用新 block
IDs **替换**旧表，而不是追加。

因此基线并不承诺保留 CPU KV 副本。“Resume”默认可以是从 token 0 重算，cache
只减少重算量。DSA 当前 `_preempt_request()` 明确报不支持，不能靠删除 gate 或
把状态改为 PREEMPTED 就宣布完成；`BalanceScheduler`、PD recompute、async
scheduling 的组合也仍需独立适配。

### 2.3 需要共同评审的候选

基线preemption可释放HBM、重置computed、清理speculative状态并重新入队；resumed worker应替换旧表。DSA还持有L2目录、resident映射、在途L3 job及双尾，不能直接复制基线状态赋值。

| 待决策项 | 候选 | 主要代价 |
|---|---|---|
| L2寿命 | Pinned suspended ledger | 恢复源稳定，但占DRAM，需限额/淘汰 |
| L2寿命 | 释放引用后按hash重取 | 利用率高，但需要L3或重算fallback |
| 打断流程 | 先持久化再释放的两阶段协议 | 正确性清楚，但引入完成等待与失败状态 |
| 恢复路径 | Dense catch-up后重新ENTER | 可复用部分现有路径，但有额外计算与边界风险 |
| 恢复路径 | 完整Indexer恢复后直接sparse first-fill | 潜在省时，需要算子/尾部协议充分支持 |
| 异步状态 | 复用KVConnector blocked/completion | 需明确L2是内部tier还是connector职责 |

此前pinned L2+dense catch-up临时原型已撤销，没有通过正式方案评审，不予保留为可恢复代码或现成部署方式。值得保留的是以下问题：

- 保存哪个verified/computed边界，如何保留尚未计算的frontier，避免恢复阶段额外采样。
- MTP provisional后缀、拒绝状态、双尾与proposer cache怎样丢弃或重建。
- 恢复中再次preempt、多个suspended请求、L2/L3淘汰、priority scheduling怎样处理。
- Pin/reuse fence如何覆盖尚在读取源的write-behind；不能仅按request finish释放。

## 3. P/D 分离

首版范围与详细合同现集中在[P/D分离设计](prefill_decode_disaggregation.md)，包括：

- 同TP、各P/D服务内部单机；D重算N-1，P构建必要MTP历史但不额外多步投机。
- D完整Indexer/MTP HBM、MLA完整L2 backing、固定resident及私有边界页。
- 独立PD交付与可选共享L3分开；P/D新满块可供后续复用。
- 复用原生Request/KVConnector和现有DSA owner，Graph/Eager同源。
- 保守准入、明确失败、安全回收；不以先完成自动preemption/resume为前置。

状态仍是详细设计待评审、未实现，不能据此移除配置gate。早期“任意TP”、
“P首输出直接交接”及“默认Store中转”不再作为首版并列方案维护；介质未通过
独立PD门禁时需重新讨论范围，不静默强制Store。

### 3.1 语义manifest不绑定物理group

为保留旧链接，本节转向新草案第5节的组件/身份/传输描述；不再复制字段表。

### 3.2 接收端落点与候选状态流

见新草案第6至10节：介质、D-first、资源预留、N-1私有边界和直接first-fill。
特别是“received有效范围”不等于“computed/公共durable”，不能直接套共部恢复公式。

### 3.3 Decode增量与最小接管门禁

见新草案第11至14节的MTP、prefix、DP与失败合同，以及第19节验证矩阵。
后续更新只修改该草案，不在此保留容易失配的简化协议。

## 4. 跨机DP

当前boot-ID检查只承诺单共享host范围。跨机需要每节点预算域、各节点L3 client预留、可达地址、部署级身份与统一配置检查，不能先删单机拒绝再测试。

Launcher、executor、API负载均衡、模型collective是不同层；跨机不自动要求Ray，Ray也不会自动解决KV ownership或管理事务。必须先选部署拓扑与进程启动方式，再设计节点故障和全域管理参与者。

请求仍可保持DP-local；运行中跨DP迁移是另一项状态转移能力，不与跨机通信混为一谈。

### 4.1 预算的分子、分母与观测窗口

单机已采用分配前共同观测和静态字节配额；跨机不能把全局 DP0 的 MemAvailable
复制给所有节点。候选须先明确“预算共享域”：物理宿主、容器 cgroup 限额、实际
本地 DP 数与每域 TP 放置，不靠重复 hostname 或一个 local-DP 配置值猜测。

```text
节点 h 可用于本部署的预算 B_h
节点 h 上的 DP 域数 D_h，每域 T 个本地 workers
每域配额 = floor(B_h / D_h)
每worker配额 = floor((每域配额 - 该域后续预留) / T)
```

两台机器各一个 DP、global DP2 时，各自分母是本节点的1，不是全局2。一个 TP
组本身跨机时，该公式也不能直接套用，需要按 worker 实际放置重新定义预算与
完整镜像合同。离线 `VLLM_DP_*` 恢复全局 size/rank，不保证同一处已恢复本节点
DP 总数；单机 SPMD 的 `data_parallel_size_local=1` 不能被当成只有一个参与者。

“每 core 看到多少余额就再除 DP”有启动顺序偏差：初始600 GiB，DP0先看到600、
按DP2分300并分配；DP1之后只看到300，再除2只能拿150，本来可容纳的第二份300
可能被错误拒绝。共同快照解决这类自我竞争，但不是物理预留，观测后其他作业
仍可占用内存。不同容器已各有独立限额时，不得把各自限额当成同一共享总预算
再次均分。跨容器、异构节点的公平份额与失败传播尚未定稿。

采样应在任何 L2 分配前的可靠窗口，分析模型加载/profile 暂态和之后的常驻
CPU/L3 开销，不是越早越好。后续 segment、local buffer、staging 分别按实际
worker/DP/node 创建份数扣除，已反映在观测中的内存不再扣一次。最终 L1 spec
确定后计算 bundle 布局即可，不必为此新增全体 HBM 分配后第二轮初始化 RPC。

### 4.2 部署与故障参与者

优先比较“每节点一个 DP、TP 留在本机”与“单 TP 跨机”，不要混成同一个扩展。
基线逐节点 serve/headless 和 Ray 均可作为编排参考，但要逐项定义全局 rank、
本地 rank/start rank、共同 master 地址/端口、HCCL EP 可达性与每节点资源释放。
Ray 的 actor 放置不替代 HCCL 或缓存完成协议。

当前单机管理会检查全体 DP 参与者；多节点若 API 只连接本地子集，不能继续用
它宣称全 deployment reset。需决定节点失联时是全服务 fail-fast 还是可恢复接管，
哪些节点保存 owner/epoch，怎样防止迟到 commit、旧 writer 或失联节点恢复后
访问旧代数据。当前没有跨进程故障接管或持久管理日志，不能只删 boot-ID gate。

## 5. 多级源计划与异步加载

已确认方向是最长最热优先：重叠区优先利用L1回填L2，L3补本地独占缺口。当前完整pair/triple+bundle已实现；任意Indexer/MLA/MTP独立源mask仍需细化admission和原子ready语义。

L3→HBM直通的前提是目标内存、注册与实际transport能在同一配置下满足所有路径。Host TCP通过与另一次Device direct通过不能拼成完整支持证明；device staging不是默认兜底。

Layerwise async需定义每层ready event、TP失败一致性、buffer寿命、partial completion、首prefill消费与Graph等待。当前全后缀同步恢复不能只挪到线程里就称为异步支持。

### 5.1 独立component命中的公式与例子

本节是候选的最少搬运模型，当前实现仍按完整HBM pair/triple与complete L2 bundle
选择来源。先限定no-MTP，且每个来源都是从逻辑block 0起的连续前缀；以下长度
全用block为单位，区间为左闭右开：

```text
I = HBM Indexer continuous prefix
A = HBM dense MLA continuous prefix
D = complete L2 bundle continuous prefix

Indexer_available = max(I, D)
MLA_available     = max(A, D)
common_hit        = min(max(I, D), max(A, D))
                  = max(D, min(I, A))

ideal Indexer restore = [I, common_hit) if I < common_hit
ideal MLA restore     = [A, common_hit) if A < common_hit
```

例`I=6、A=3、D=5`：公共命中5块；候选可直接复用Indexer前5块和MLA前3块，
仅从L2恢复MLA的块3、4。当前pair策略同样命中5块，但完整HBM pair只有3块，
后2块从L2同时恢复Indexer和MLA，因而可能重复搬运仍在HBM的Indexer。区别在
copy范围而非命中长度；这是已采用的保守策略，不是当前prefix命中bug。

只有仍具有dense文本语义的MLA块可计入A，sparse budget、resident slots及
provisional tail不在其中。若各层留下散落残片，三个标量不足，必须逐逻辑块/
component检查，不能越过第一个缺口把后面仍在池中的块拼成连续prefix。

### 5.2 加入L3与MTP的SourcePlan

候选按每个逻辑块、每个必需component选择最热且ready的来源，顺序为L1、L2、L3。
当前MTP已有verified BF16 L2镜像，不能沿用早期“MTP只有HBM或L3”的来源集合。

| 逻辑范围示例 | Indexer来源 | MLA来源 | MTP来源 |
|---|---|---|---|
| 0-99 | L1 | L2 | L1 |
| 100-199 | L2 | L2 | L3 |
| 200 | 任一可用层 | 缺失 | 任一可用层 |

该例公共raw前缀止于200之前；MTP还必须按基线Eagle规则换成可执行边界，不能
对每个tier分别减块再相加。Eagle末块回退用于重建drafting hidden states，不是
证明旧KV未填满或含未验证token。计算时统一用blocks，最后乘block size换tokens，
不要混用`target_blocks`和`mtp_tokens`比较。

Scheduler应只生成一次带类型SourcePlan，worker不再查hash或自行重新选tier：

| 计划部分 | 需要表达的内容 |
|---|---|
| HBM reuse/touch | 每component逻辑范围、命中物理块与引用保护 |
| L2→HBM restore | source L2 ID、独立component目的HBM ID、copy范围 |
| L3→L2 promotion | keys、预留L2 destination、完成与失败身份 |
| HBM→L2 mirror/backfill | 仅L2缺失的完整durable范围、source pin及提交边界 |
| L3→HBM direct（另行启用的候选） | 兼容transport、目的schema和随后L2补齐责任；当前不能启用 |
| Remaining compute | raw命中转换后的可执行边界、需重建/重算的suffix |

各rank按同一逻辑计划解释本地地址；全组件容量预检、touch、分配、load、restore、
ready提交必须有明确顺序，防止查询后source被复用。列出两种L3目的地不表示要
同时维护两套eager/Graph元数据，也不改变当前fail-fast而非可回滚load的边界。

### 5.3 L2物理解耦的收益门槛

独立component来源不要求同时拆L2 allocator。当前一个L2 bundle ID索引各个
Indexer/MLA/MTP arena，统一ready、refcount和LRU，已能表达完整durable副本。
只有需要独立淘汰或分别配置component容量时，才考虑如下目录形态：

```text
BundleRecord(
    block_hash,
    indexer_payload_id,
    mla_payload_id,
    mtp_payload_id,       # optional
    component_ready_mask,
)
```

这只是字段示意，不是已存在类。payload ID可不同，外部identity仍按
`hash + component + schema`，不依赖任一物理ID。代价包括多套free/LRU/refcount、
独立restore列、部分component保留/淘汰以及原子完整ready的失败处理；必须证明
独立容量或少搬字节的收益足以覆盖这些成本，再决定是否拆分。

早期内部接口候选`DSADRAMPrimaryTier`可按`lookup_ready/reserve_write/
publish_ready/acquire_read/complete_read/release_or_evict`划分职责，secondary可按
`lookup/submit_store/submit_load/poll_finished`划分职责。这不要求引入新类或
Python Protocol，更不是新增第二个block owner；实施时优先在现有typed directory、
connector和backend adapter上表达确有必要的合同。

### 5.4 Direct路径与完整backing

一条候选direct路线是：

```text
L3 -> dense HBM
   -> prefix-prefill按层补齐缺失的完整L2 bundle
   -> worker成功返回，提交durable ready
   -> scheduler才可释放dense HBM
   -> 下一forward按约定进入ENTER/SPARSE
```

对长DSA请求，它与`L3 -> L2 -> HBM`都包含两段大传输，不能预先保证更快。
短dense请求或另有backing合同的consumer可能有不同代价，但需要独立证据；
直接跳过dense恢复构建sparse resident会改变PREFILL/first-fill合同，不属于简单
transport开关。当前所有protocol的direct-HBM均未放行，device staging/relay
不是默认生产兜底。无论选择哪条路线，L2 gap未补齐时不能发布跨洞ready。

## 6. 其他候选

Proposer Graph需覆盖BF16 cache、draft循环与durable dump，不是删掉enforce_eager。其他后端、CP/PP、动态配额、权重在线更新、L3物理GC和多租户隔离各有独立协议，不在一个“泛化支持”开关下同时开放。

### 6.1 写入时机与源寿命

当前已适配的是每域TP0异步write-behind、有界job及physical-ID reuse fence。
以下策略仅供性能阶段比较，不能将“选用候选”写成已实现开关：

| 策略 | 可能收益 | 代价与待定合同 |
|---|---|---|
| 每个durable新块增量下沉 | 远端覆盖及时，行为容易验证 | IO较多；旧文称write-through，但当前异步路径不能被误解成同步等待put |
| 接近L2淘汰时再写（write-back） | 减少只使用一次的远端写入 | eviction/finish时可能集中burst；先完成source保护，不能先复用再补传 |
| 热度admission | 同hash第二次出现或达到阈值才晋升L3，减少无效存储 | 热度身份、统计范围、低频长prefix的覆盖损失需量化 |

旧三引用模型`request_ref/cache_ref/io_ref`是另一种候选表达，不是当前实际字段，
也不要求永久pin所有pending source。无论采用何种模型，request finish不能取消
仍被合法IO读取的source；当前normal active ref与reuse fence保护不得丢失。
队列满时延后promotion并保留普通L2，不应为抢占远端带宽阻塞decode；新增强背压
必须作为不同方案评审，不能冒充当前行为。

`STORE_PENDING`本身不是remote hit。同机L2完整可读时仍可复用；若所需L2/L3
均不可读，应按已定义的miss/重算或失败规则处理，不等待未知时长的store。
同hash重复提交、partial component、源已复用后的重试与取消须有幂等边界。

### 6.2 轮转writer不是交付前提

冻结基线的AscendStore可按`put_step=tp_size/num_kv_heads`分担副本型KV写入；
MLA单head、TP8时每rank负责每8个候选块中的一块。早期DSA据此提出：

```text
owner_rank(block) = logical_block_ordinal % tp_replica_group_size
```

若评估该候选，同一logical block的全部component应选同一owner，且key不能编码
physical writer rank，避免换writer后无法复用已有对象。需定义ordinal的稳定范围，
不能因chunk切分改变owner含义。它只是在TP副本组内分摊IO，不自动提供跨DP唯一
writer或原子bundle发布；并发writer协议仍见第13.2节。

当前每域TP0写者已适配，轮转只有在profiling证明单点带宽/排队构成瓶颈后才值得
承担额外job分派、完成汇总与fence成本。不能为规避某NUMA Host路径失败而把物理
NPU拓扑写进scheduler；后端路由、relay或replication应留在backend边界。

### 6.3 性能指标与控制项分开

性能比较应固定模型/schema、prompt分布、全局并发、命中层级与backend版本，
分别报告建链冷启动、稳态传输、L2/HBM搬运及模型计算。旧记录中的首次建链秒级
与连接复用后毫秒级不能混成一个“L3带宽”，本草案不据历史耗时承诺当前收益。

GLM-5.2 no-MTP每块有42个Indexer plane和78个MLA plane，64块合批会产生
`64*(42+78)=7680`个descriptor。当前adapter的128预算是跨key合批软上限：
单component超预算时隔离为single-key并告警，不为满足128把一个逻辑key任意拆开；
transport内部如何chunk与scheduler完成状态分开。MTP还需加入自身plane/bytes，
不能照搬no-MTP的总量。

应分开评估job数上限、在途bytes配额、每次descriptor数、completion poll间隔、
连接预热和实际网络并发；并非这些建议都已成为公共配置，`max_active_channels`
也不等于TCP并发数。候选最小指标包括：

- TTFT、TPOT、吞吐及队列等待，而非只报命中率。
- 每层真实bytes、descriptor数、H2D/D2H次数、restore wait及计算重叠比例。
- 每个writer的job/bytes分布、zero-token polls、source pin量和reuse-fence等待。
- FD/ESTAB、segment注册/关闭资源与长稳趋势；不能只以短请求完成宣称无泄漏。
- L3对象增长、重复put、partial对象、retention/TTL/GC与KV events/metrics的准确含义。

输出相同、top-k互含或同源非确定性复现都不是吞吐收益或全量logits等价证明。
任何优化仍需单独执行正确性门禁，不能用性能数据替代内容/生命周期验证。

### 6.4 Backend备选与准入

下表是已读冻结资料中的能力与待验证方向，不是对外部最新版或A5支持范围的新认证。
当前DSA正式路径仍是Mooncake Store-owned TCP；新增后端必须满足同一key/schema、
完整字节数、source lifetime及失败完成合同，不能通过接入另一个connector引入第二套L2 owner。

| 备选 | 可参考的能力 | DSA接入前仍需验证 |
|---|---|---|
| MemCache/MemFabric | G2H/H2G/H2GH，DRAM/L2.5 write cache/SSD，layerwise及GVA/lease接口 | Local EID、AICPU包、真实swapped Host与NPU IO、lease/close；当时A2 RDMA/A3 SDMA建议不构成A5证明 |
| LMCache/LMCache-Ascend | LocalCPUBackend、远端backend、CPU/NPU与PD channel；注册固定L2再get into的模式 | wrapper HMA声明、DSA多group公共命中、固定arena所有权、目标硬件与外部依赖版本 |
| UCM | 内部DRAM cache与PipelineStore，Posix/NFS/3FS持久层 | 内部CPU缓存不直接给KSC稳定arena；不能只凭具有DRAM层就替代DSA L2 |
| Yuanrong | Datasystem+etcd，`mset_d2h/mget_h2d`，可选remote H2D，无额外预注册模式 | HugeTLB/外部worker部署、Host L2 adapter、A5及注销/关闭合同 |
| vLLM FS secondary | CPU primary memoryview上的Python线程文件IO，可用于协议原型 | DSA scatter-gather、完整bundle、并发/失败与后端持久性，不强制改KSC友好arena布局 |
| vLLM object secondary | NIXL与S3兼容对象存储 | 平台依赖、延迟、CPU gateway及DSA地址/完成适配；源码存在不等于NPU可用 |

后端内部DRAM/SSD冷热迁移或TTL不是DSA active L2的可淘汰授权。旧
NPUOffloadingSpec相关deprecated/skip记录也不是当前正式byte-copy测试可skip的依据。
候选应先在无模型probe验证register/put/get/bitwise/close，再用DSA-disabled基线
验证真实external hit，之后才测DSA三component、TP/DP及Graph组合；Host成功不能
代替NPU、单卡不能代替跨NUMA、注册成功不能代替payload与关闭成功。

## 7. 定稿与实施门槛

每个候选定稿前至少确认：目标场景、数据/状态owner、完整时序、容量约束、失败/重试/取消行为、Eager/Graph同源设计、基线复用点及最小验收矩阵。

确认后将正式合同移入相应`features/`专题，并在[特性路线图](../feature_roadmap.md)标为“设计确认 / 待实施”；本草案保留未决部分，不继续维护第二份正式设计。

### 7.1 估算边界、依赖与不确定性

重新估算只覆盖尚未实现并已明确范围的工作，不迁移早期人周/月数或按完成比例
扣减旧总工期。scheduler-owned L2、完整bundle、Local/L3 prefix和约定单机DP
已经是现有基础，不能再次计为从零开发；独立source mask、layerwise、P/D与恢复
又有不同前置条件，不能当成几个互不依赖的小开关。

估算应先冻结配对vLLM/vLLM-Ascend版本、connector ABI、backend、硬件和验收
范围；明确假设的人员经验、可用测试环境及协作边界。计算/编码投入与机器排队、
外部存储部署、跨团队联调等待分开说明，不能把后者默认为零。

按“协议/所有权决策 -> API和介质probe -> component验证 -> 集成 -> 端到端及
生命周期”列依赖，再区分可复用工作与独立新增工作，避免重复相加。网络瓶颈是否
需要per-layer schema、P2P能否写D L2、TP payload是否可互换、恢复尾部能否不
改变采样，以及Graph等待/失败传播都是可能改变范围的不确定性，须先以探针收敛。
工具可减少阅读与样板编码，不会消除A5编译、精度、异步寿命或多机联调的验证成本。

## 8. 一次恢复究竟要保存什么

考虑请求已计算到C，durable满块边界为D，且target已经进入sparse。候选恢复记录至少要
区分以下语义，字段命名与持有模块仍待确认：

| 信息 | 为什么不能省略 |
|---|---|
| 请求token历史与已确认输出 | 重新构建block hash、保证恢复不是另一个请求 |
| 旧computed/verified边界 | 分清已算KV、已确认token和尚未计算的frontier |
| Durable logical block范围 | 表示哪些完整组件真的可恢复，不是只记录已分配块数 |
| 当前可用源及其寿命 | L2 pin、可失效hash或L3引用具有不同保证 |
| MTP accepted/provisional与drafting边界 | 不能把未验后缀当成可复用前缀 |
| Generation/operation identity | 防止旧job或迟到completion更新新一代请求状态 |

旧HBM physical ID和resident slot不应直接作为跨实例/重分配后的地址。恢复前先确定
逻辑数据，再绑定新物理资源；若未恢复同代完整resident映射，应重新first-fill。

## 9. 两种恢复流程的对照

### 9.1 Pinned-L2 + dense catch-up候选

```text
preempt前确认durable源 -> 暂存有配额的suspended L2引用
  -> 释放HBM与旧worker row -> waiting中保留请求语义
  -> 再admission全部component -> 分配新HBM表并恢复L2前缀
  -> 恢复不足的边界计算 -> 重新ENTER/first-fill -> 正常decode
```

优势是恢复源相对稳定；代价是请求虽然释放HBM，却仍占L2，多个suspended请求可能把
压力转移给主机内存。必须定义配额、驱逐和再次preempt策略。边界catch-up不能意外采样
额外输出，不能把旧computed和最终可复用prefix混成一个数字。

#### 撤回原型曾使用的精确顺序

以下是未评审原型的研究记录，**不是当前恢复能力，也不是已批准实施方案**。
保留细节是为了比较方案时能重现其假设，而不是恢复已撤回代码。

1. Scheduler 始终维护 `ready_block_count`；APC 开启时才进一步发布 hash，
   APC 关闭也将 ready 作为 durability 事实。分配了块不代表可恢复。
2. Preempt 前确认已计算的完整块全部在 L2 ready；释放 Indexer HBM、resident
   MLA HBM 及 worker resident row，不保存旧 HBM physical ID 作为恢复地址。
3. Ready active L2 row 转入按 request ID 索引的 pinned suspended ledger，
   暂不归还 block refcount；请求生命周期与 payload 生命周期从此分开。
4. 保存 `resume_target_tokens = old num_computed_tokens`。再次 admission
   为全部 HBM components 重新分配，并从 logical block 0 下发 L2 assignment。
5. Prefix lookup 的上限收缩至 `resume_target_tokens - 1`。先只推进一个 dense
   token，以实际 restore 结果确定 computed boundary，再以一个或多个 dense
   chunks 精确追平旧水位；**下一 step** 才重新 ENTER sparse 并 first-fill。
6. Worker 接到 `preempted_req_ids` 后释放旧 resident row、`cache_slots` 和 L2
   logical-table 镜像；物理 L2 payload 是否保留由 scheduler ledger 决定。
7. 新 admission 因边界回退而不再需要的 suspended suffix 可以释放；若旧 L3
   store 仍读其物理块，必须先满足 physical-ID reuse fence。

本原型只考虑 no-MTP、同步调度和 chunked prefill，不能机械外推到当前固定
MTP3 的七输出2304、双尾和 BF16 full-HBM proposer。MTP 还须明确释放/重建
full-cache、accepted/provisional、dump epoch 与 drafting boundary 的时机，
旧动态 query/2176 研究中的缺陷也不能记作当前实现仍有的 bug。

#### 为什么必须留下一个尚未计算的 frontier

设抢占前已计算 C 个位置，同步 no-MTP decode 有
`num_tokens - num_computed_tokens == 1`。位置 `[0,C)` 已有计算语义，位置 C
是已经采样但尚未作为输入计算 KV 的 frontier。恢复只应追平 C，不能在 dense
recovery 中连 frontier 一起消费并额外采样；否则恢复本身改变了输出和 sparse
进入时机。

例如 C=41672、block_size=128，完整持久化边界 D=41600。候选允许先恢复
`[0,41600)`，通过受控 dense catch-up 重建剩余72个已计算位置，但停止在41672；
位置41672留给重新建立的 sparse 路径。这只是边界说明，实际 prefix 可能被淘汰
或因模型规则回退，必须以 restore 结果决定起点，而不是总假定命中 D。

不完整尾块是否保存、能否精确重算且不改变后续数值，需要独立评审。将水位写成
整数 C/D 不能替代这些字节/采样合同，尤其不能把 verified、computed 和 durable
三个边界当成同一个数字。

#### 原型未解决的限制

- 不支持恢复完成前再次 preempt，也不支持 active request 的
  `reset_running_requests=True`。
- Suspended L2 不允许淘汰，缺少配额/驱逐，多个请求可能把 HBM 压力转移到 DRAM。
- Priority scheduling、多 suspended 请求、L3-only 恢复尚未证明。
- Dense catch-up 引入额外 forward，eager/Graph 路由及每 step ready 维护开销未评估。
- 尚未对齐基线 offloader、KVConnector 和 DP 生命周期，不能原样采用。

### 9.2 释放引用 + 重新按内容查找候选

```text
保存逻辑恢复边界 -> 释放active L2/HBM引用
  -> 再admission时重新查询L1/L2/L3 -> 找到则恢复，缺失则走已定义的重算/失败路径
  -> 重新建立物理表与resident状态
```

该方式更利于容量利用，但不保证原来的块还在。必须允许远端查询与读取之间失效，不能
把preempt时的hit plan长期留作有效地址。选择这条路线前，要确认是否接受重算成本，以及
恢复时的数值/采样边界是否仍满足项目目标。

两条流程都需要在途write-behind源保护。管理pause只停止新请求调度并排空，不能代替
单个请求的suspended恢复协议；这也是不能直接复用P6-B来宣称preemption已支持的原因。

### 9.3 其他候选的性能与实现取舍

| 候选 | 可能节省什么 | 必须付出的成本/解决的失败窗口 |
|---|---|---|
| Pinned ledger | 稳定 L2 源减少重复 get/重算 | pin 占用 DRAM，需 suspended 上限、驱逐与公平性；HBM 获释不代表总内存获释 |
| 释放后 hash 重取 | 释放容量，让活跃请求利用 L2 | lookup/get 之间失效、旧地址覆盖，需要 L3 或重算 fallback；hit 不是恢复保证 |
| 两阶段 preemption | 尾部与 metadata 持久化完成后再释放，边界更明确 | worker 完成回报前不能回收；scheduler preemption 变成异步过程，需失败、取消和再次抢占状态 |
| 直接 sparse first-fill | 完整 Indexer 恢复后以 KSC 类路径建立 resident，避免 dense tail forward | 当前算子是否支持所需输入/partial-tail、完整 backing、首轮候选和双尾初始化尚未确认 |
| Connector 管理恢复 | 复用 `KVConnectorOutput`、blocked waiting 和 completion | 先决定 L2 是 scheduler 内部 tier 还是 connector；不能两个 owner 同时释放同一源 |

性能比较至少分开：释放了多少 HBM、额外 pin 多少 L2、网络/L2->HBM 字节、恢复
forward 数、阻塞 admission 时间、恢复后首 token 延迟及稳态吞吐。稀疏 first-fill
看起来少算，不代表在网络或 Indexer 恢复主导时一定更快；固定 L2 pin 也不代表
高压力下整体吞吐更好。当前没有这些候选的 A5 性能结论。

### 9.4 采用前的学习与测试合同

先跟读基线 allocation failure、victim selection、preempted queue、resumed
worker 整表替换，再比较 Ascend recompute scheduler、CPU offloader 与
AscendStore/Mooncake 的 completion；新社区实现须标明版本，不直接套到 v0.23。
此外要确认 internal-DP 与离线/external 前端拓扑中的 request ID、L2 ledger、
L3 namespace 归属，恢复不默认跨 DP 迁移。

最小测试须覆盖：APC 开/关、完整/部分尾块、无命中/L2/L3、源在途被复用、读回
短缺、优先级和多个 suspended、恢复再次抢占、取消/finish 与迟到 completion、
reset/epoch 变化、P/D/idle 相遇，以及将来 MTP 的 accepted/provisional 与三组件
水位。每项都要检查新表替换旧表、无旧 physical ID 泄漏、无额外输出、无跨缺口
ready、Graph/eager 同源；先 no-MTP 证明，再讨论 MTP，不以整网完成代替这些合同。

## 10. P/D接管的确认边界

为保留旧链接，本节转向[P/D详细设计](prefill_decode_disaggregation.md)第7、9、14节。
生产可读、目标可见、布局提交和源允许回收仍是四类不同事件；详细条件与失败窗口仅在新草案维护。

## 11. Layerwise加载为何不能只开线程

现有L3 component key包含跨层plane集合，get成功首先是对象级完成；恢复本地HBM则
逐层消费已经完整落到L2的数据。第一种候选只重叠L2→HBM与prefill，保留远端schema；
第二种候选进一步调整为per-layer对象，使网络读取也逐层化。二者改动与收益不同。

要为每个阶段定义固定buffer、event、wait位置、源pin和失败汇总；GLM shared Indexer
按真实full源层映射，不按78个MLA层重复取Indexer。任何优化都不能在L2 gap未补齐时
发布完整bundle，或让一个TP进入该层forward而另一个仍在失败读取中。

### 11.1 第一步：保留远端schema，只流水化本地restore

候选先等待所需L3对象完整进入L2，维持当前跨层key与后端原子可读边界，再按层
发起L2→HBM restore。复用基线attention侧`wait_for_layer_load()`思路，用固定
`num_prefetch_layers`窗口提前搬运后续层，在该层消费前等待其HBM可见：

```text
complete L3 -> L2 load
  -> issue bounded layer H2D prefetch
  -> wait before consuming layer -> run layer prefill
  -> advance the fixed prefetch window
```

潜在收益只覆盖H2D与prefill计算重叠，不隐含网络读取也已流水化。必须保证L2
source及HBM目的块在copy完成前不可复用，固定buffer/event在初始化期准备，
不能在每层热路径重新分配tensor或backend RPC。

### 11.2 第二步：按证据决定per-layer远端对象

只有第一步profiling仍显示网络IO主导TTFT，才评估schema版本升级为per-layer
object，使`L3 -> L2 -> HBM`都能逐层推进。当前no-MTP一个Indexer key跨42个
plane、一个MLA key跨78个plane；仅打开`use_layerwise/load_async`无法产生各层
独立get完成事件，不能把一个对象完成误当成所有后续attention都已ready。

升级必须定义key/version隔离、每层lookup/partial completion、L2逐component
状态和最终完整bundle提交。旧跨层对象是否继续可读或需要重新生成是迁移问题，
不能用相同key解释两套payload。该步骤比第一步多了远端对象数、lookup和完成
管理成本，收益不应仅以“支持异步”描述。

### 11.3 Shared physical source与MTP

GLM-5.2只有21个真实full Indexer source layer，57个shared follower应引用其
对应source；不能按78个MLA层重复存取Indexer。候选metadata必须同时表达
physical source到消费者的映射和消费者等待关系：一个source取回一次，所有
依赖它的层仅在source ready后消费，而各层自己的MLA仍独立恢复。

开启MTP时还必须覆盖BF16 component和proposer消费/dump epoch，不能只对target
层做完prefetch就提交三component完整ready。这是加载依赖问题，不改变当前
固定MTP3、七输出2304与proposer full-HBM，也不在本节引入动态query方案。

### 11.4 Event、失败与Graph验收

每阶段分别标识“网络写L2完成”“H2D写HBM完成”“该层可消费”“完整durable
bundle可提交”。Event只能证明其记录的工作完成，不能替代scheduler的完整ready
提交；即使部分层已算，也不能跨L2 gap发布hash。

候选需确认wait位于图外、图内合法事件还是piecewise边界，并让eager/Graph消费
同一份SourcePlan和固定owner；不捕获动态RPC。某rank/layer失败时，未完成copy、
已消费层、源pin与请求取消如何收尾必须一并定义，不能一个TP继续forward、另一个
停在失败get。最小验收包括首prefill长gap、部分层短读、shared source被多个层
使用、prefetch窗口边界、MTP组件晚完成、取消与迟到completion、重复Graph replay
及source复用。两步分别给出bitwise、时序和端到端证据，不用线程已启动代替通过。

## 12. 全局阶段调度与负载均衡候选

当前已选方案 A：保持本地 phase barrier，修补 worker 执行合同；该事实放在
features，本节只保留未采用方向及重新考虑它们的条件。

**方案 B：统一全局 prefill/decode 阶段。** 这不只是一个 phase bit，还需要
执行 epoch、真正可接纳性、idle、waiting、公平性、零 token IO 与异常传播。
若某域 waiting prefill 因容量不足不能 admission，而全局规则又禁止其他 decode，
原本需要 decode 完成才能释放的资源就无法释放。不能以“更保守”当安全证明。
只有必要通信协议确实要求阶段一致、又无法局部适配时才重新评估；不能拿它
掩盖合法 P/D 场景的数值差异。

**方案 C：本地混合 prefill/decode batch。** 涉及 row mode、算子、dump 和
dense-to-sparse 转换，不是支持跨 DP 不同阶段的先决条件；不能随 DP 接入顺便开放。

**入口与预算的性能策略。** 长度/阶段/cache-aware 路由、离线按 token 工作量分片、
按 KV capacity 加权及每域 chunk budget 配合可以单独评估。当前 internal LB 的
`waiting*4+running` 和100ms级异步统计不是逐轮全局 token 配额器，不自动借用
空闲域 budget，也不 work-steal 已归属请求。EP/EPLB 平衡专家不等于平衡 Attention。
评估要固定全局负载，测混合长度、P/D、idle、prefix 亲和带来的收益与等待，分别
报告 TTFT、TPOT、吞吐、容量；不要仅用等长大 batch 或缓存命中率推导端到端收益。

## 13. 共享内容、持久身份与发布协议的未决部分

### 13.1 静态身份与运行时 epoch

当前单机 external reset 使用统一 UUID epoch，不再使用早期各 scheduler 独立
`generation++` 的方案；后者曾可能从0/0变成1/0，再重试成2/1，重试或 bool AND
都不能撤销部分提交。这个历史反例用于解释当前全域协议，不是报告当前仍有此 bug。

静态 schema/version/model/component/hash 与运行时 reset epoch 仍解决不同问题：

| 场景 | 正确性或产品要求 |
|---|---|
| 模型/量化/cache 语义不变 | 旧内容本应可复用，不必每请求换代 |
| 清本地后验证 L3 loadback | 保留原 namespace，否则测不到旧对象读回 |
| 验证真正 cold store | 用明确隔离的 salt/epoch，不能仅凭清本地假设远端冷 |
| API 明确 external reset | 需要逻辑失效或删除的真实效果，不能以默认 no-op 当完成 |
| 在线换权重但 key 不变 | 旧 KV 可能错误命中，需要可靠权重版本和失效合同 |
| Payload 语义变化但 schema 不变 | 属于版本合同错误，hash 相同不能证明兼容 |
| 要求物理删除或立即回收容量 | epoch 轮换不够，旧对象仍由后端 lease/淘汰/segment 生命周期处理 |

早期进程内 generation 从0重启还可能撞回旧代；当前 UUID reset 改善一次运行中
的统一换代，但不等于任意跨实例持久化、热更新、多租户都已有部署身份协议。
模型路径不同可能导致相同权重不共享，同路径被替换也不证明内容相同。是否采用
权重 digest、显式 deployment cache epoch、用户 namespace、持久管理日志与 GC
仍待产品场景明确后定稿。

### 13.2 多 writer 完整发布

当前复用 Mooncake 单 key 互斥/幂等与 post-put 全组件可读检查，不先创建分布式
claim/manifest。待验证的假设是：相同 compatibility/schema/hash 下，两个 DP
生产的同 component 语义等价，允许一个 logical bundle 的不同组件来自不同 writer。
“全部 key 可读”不证明来自同一次 producer 发布，也不证明跨 batch 字节一致。

候选升级门槛是实际证据表明上述可互换假设不成立，或产品要求单 producer 原子
发布。届时需定义 claim 的 owner/失联接管、manifest 的 complete 提交、partial
components 的寿命、重试幂等、取消和 GC，不只再加一个锁。应先用独立 client
测试同 key processing、writer 失败、部分组件可读、淘汰与重试，再决定是否承担
这套协议。普通 put、允许覆盖的 upsert、remove/recreate 必须分别定义，不能混用。
