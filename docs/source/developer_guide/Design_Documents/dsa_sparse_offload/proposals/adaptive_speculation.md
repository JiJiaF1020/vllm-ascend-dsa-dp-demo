# 自适应推测解码设计草案

> 状态：目标已确认，详细方案未定稿、未实施。本文替代早期“理想版MTP强制合同”原稿；候选字段和公式不是当前ABI。
> 当前固定MTP3的七输出、预算和因果双尾仍以[已确认设计](../features/speculative_decoding.md)为准。

## 1. 目标

在不因为draft数增加而永久抬高resident预算的前提下，利用query之间TopK重合程度，自适应选择能精确计算的最大query前缀。不得以随机丢重要token或伪造reject来掩盖容量不足。

当前固定MTP3最低budget为8192。本草案讨论恢复较小预算等可能性，不表示6144已能在当前MTP启动配置中使用。

## 2. 候选容量模型

对某请求，令Kq为query q在既定durable历史上的TopK集合，R为请求冻结resident预算，尾区单独管理：

```text
U(m) = union(K0, ..., K[m-1])
m = 最大满足 |U(m)| <= R 的连续query前缀长度
effective_draft_count = m - 1
```

保留最后一组有效target logits用于bonus/recovery/frontier采样，而不是将m个query都当成m个已验draft。以调度3 draft、4 target query、R=6144、TopK=2048为例，三个集合最坏并集6144；第四组加入若超限，可只保留m=3，实际验2 draft，从第三组有效logits产生下一frontier。若并集不超限则m=4。

该推导依赖tail容量与历史候选范围正确、每个有效query确实完成精确attention，不能仅凭集合计数就声称整网分布无偏。

## 3. 跨层、跨rank与有效后缀

候选策略是各层得到本层最大有效query前缀，再使请求的有效数跨层单调不增。后层截短后，后缀输出不得进入最终采样、KV发布或下一轮draft状态。

这种设计需要复核因果attention及MoE无token-drop等条件。实际量化、padding、专家分组和归约可能改变数值路径，不能仅凭“Transformer按token独立”忽略实现细节。

TP域必须对有效query数和resident状态取得一致结果。不能依赖物理pool行、未定义tie-break或核调度顺序，也不能未经性能设计就在每层增加host同步/collective。允许的协议与验证方案尚需确认。

## 4. 需要协同的接口

| 模块 | 待定义合同 |
|---|---|
| Scheduler | 调度draft数与有效draft数分离，computed回退、截短统计、完成边界 |
| LIM | 逐请求前缀union、容量判定、映射更新和错误传播 |
| Attention | 固定物理query容量中的有效前缀/无效后缀，因果tail |
| Sampler/rejector | 最后一组有效logits、bonus/recovery、未验suffix语义 |
| Proposer | 接受/拒绝/容量截短后的输入、slot和cache边界 |
| L2/L3 | 仅发布verified满块，不把已算但无效的后缀当durable |
| Graph/DP | 请求级有效长度、跨域collective形态与统一metadata |

`valid_target_query_count[B]`等名称只是候选。不能现在改七输出接口，也不能将query数量误称为topK token数量。

未来实施宜用显式版本化接口或独立实验算子验证有效query协议，保留当前固定MTP3的
七输出路径作为回归对照；不能在同名同schema下悄悄改变有效范围和消费者含义。

主动容量截短与普通rejection不同。随机采样中的reject可能使用target/draft修正分布；伪造reject不等价于从最后有效target分布采样。是否先限制greedy、如何再推广采样，需要明确方案。

## 5. Eager与Graph候选

逻辑有效长度应共用同一真源；Graph可探索固定物理query容量加mask/PAD，Eager可保留紧凑TND视图。物理布局不同不应产生两套有效query状态。

不能从底层attention宣称某query上限，直接外推平台、binding和当前框架全部支持。上限、可见长度、workspace、capture shape、DP同步与proposer循环需联合门禁。

无效suffix还有两种待选实现：扩展attention接口，使其在任何数据读取前按有效数跳过；
或保持现有attention接口，给无效行提供完全合法的空选择/安全输入，随后禁止sampler
消费这些输出。后者可能多做无用计算，也必须证明空选择合法，不能仅假设乘零可阻止
非法读。下文保留旧交接稿的前一种严格合同；两条方案尚未统一，不能同时声称接口
不变且已经实现了kernel内的有效query提前返回。

## 6. 必须讨论的决策

1. 是否只做精确前缀截短，拒绝近似union裁剪；tail与resident预算是否保持独立。
2. 有效数在哪一层产生、如何传递到sampler，后缀是否仍计算部分层以及收益是否值得。
3. TP一致性与相等分数tie-break的协议，异构query下的图策略。
4. 拒绝、全接受、容量截短、EOS和max_tokens各自怎样推进frontier与KV。
5. 先支持greedy还是同时支持一般采样，以及对应数学/数值验收。

在这些决策完成前，不将历史示意中的“唯一首版策略”当作当前共识。

## 7. 未来验收框架

先做CPU集合/采样状态模型，再做单卡1..支持上限的query组合、最坏不重合/全重合、跨块双parity、连续截短与状态重用；随后覆盖同输入TP一致性、Eager/Graph、MTP多级prefix及DP空域。

验收既要证明正确，也要测量截短频率、有效验算收益、同步与额外scratch成本。不能仅因某批输出相同就认定方案完成。

## 8. 有效query为何比有效draft多一行

调度3个draft时，4个target query的用途是：

| target query输入 | 完整验算时logits用途 | 截短到3行时 |
|---|---|---|
| frontier | 验draft0 | 验draft0 |
| draft0 | 验draft1 | 验draft1 |
| draft1 | 验draft2 | 从该组有效target分布产生下一frontier |
| draft2 | 全接受后产生bonus | 无效后缀，不参与采样/持久化 |

所以m=3时只验证两个draft，最后一组有效logits用于保留frontier/bonus位置，而不是被
浪费。若把第三行也算作已验draft，再读取不存在的第四组有效logits，就会错位或越界。

一般采样中容量截短不是target拒绝了某draft。普通reject可能采用target/draft分布差的
修正采样；容量截短需要最后有效target分布，不能通过伪造reject索引直接复用全部旧路径。
Greedy可作为较小验证范围，但不证明一般采样自动正确。

## 9. 跨层单调收缩的候选协议

```text
scheduled_target_queries[B]  # 原始物理输入与调度布局
valid_target_queries[B]      # 各层已认可的共同有效前缀
layer_valid_queries[B]       # 本层union容量上限
valid = min(valid, layer_valid)
effective_drafts = valid - 1
```

这些是候选语义名，尚未冻结为ABI。每层只允许缩短，不能在后层重新使用前层已废弃的
query后缀。Sampler、accepted统计、scheduler computed回退与proposer必须读取同一份
最终有效数，不能各自推导。

因果性使前面query在数学上不依赖未来query，但实现仍需验证量化/归约/padding、MoE
路由与token-drop条件；不能用一句“Transformer按token独立”替代实现层面的证明。

## 10. 批内异构有效长度与固定图容量

两个请求都调度4行，最终有效数可为`[3,4]`。候选Graph仍使用每请求4行固定容量，
只屏蔽第一个请求的最后一行，不能把全batch统一截成3行。Eager可以消费紧凑TND视图，
但有效长度向量仍来自同一owner。

基线`num_draft_tokens`、`cu_num_draft_tokens`、`cu_num_sampled_tokens`区分逐请求
draft长度，例如`[3,0,2,0,1]`及其累计offset。新方案应延续这种request/query分轴，
不能把“语义支持异构长度”误读为“原生FULL graph已支持所有异构布局”。

待评审点包括无效后缀的attention/no-op行为、MoE collective兼容shape，以及是否仍算
部分无效行换取固定图。不能逐层`.item()`回读到host后重建图，也不能让TP各自截短。

## 11. 必须构造的反例

- 四组TopK完全重合、部分重合、前三组恰好填满预算、第四组只超1个token。
- 请求预算与有效长度不同，相等分数下的TopK/victim仍确定。
- 前层认可4行、后层缩到3行，无效后缀不进入下一轮可见KV或L3。
- 全接受、首/中间draft拒绝、容量截短遇EOS/max_tokens，输出与computed边界一致。
- D跨128边界与两种parity、first-fill/steady、prefix恢复后新pool行。
- TP物理block/pool ID不同但逻辑数据等价，有效query数不能因此变化。

先做集合与状态模型，再做真实算子和整网。历史原案提出的1..16 query不是当前支持
承诺，实际测试上限需要平台、binding、workspace和Graph策略联合确认。

## 12. 候选有效行的完整生命周期

这一节保留旧原案中可继续评审的设备状态协议，并非当前已实现能力。有效长度应是
**一次target forward内的临时设备状态**，不加入scheduler跨step请求账本。

| 对象 | 候选owner及寿命 |
|---|---|
| Scheduled布局 | 基线scheduler/`SpecDecodeMetadata`，本轮物理span和packed偏移不可变 |
| `valid_target_query_count` | Runtime预分配int32 `[max_num_reqs]`，每forward重置、层间收缩、sampler/proposer消费结束即失效 |
| 各层`cache_slots` | 每selection group、每stable pool row的持久token→resident slot映射 |
| Query/copy plan | 固定scratch，按本层LIM→KSC→attention及shared follower消费顺序复用 |

旧原案提出用首个target selection层的静态`reset_valid_target_query_count=true`
在算子内完成初始化，省去单独H2D/初始化kernel；后层传false。这一选择仍须评审
shared-indexer拓扑，不能把“每个Transformer层都重新初始化”作为实现。

1. 每forward由累计query end的差分覆盖active计数，PAD请求写0，即使batch和行序未变也必须重置。
2. 本层仅对输入有效前缀求union，再原址写`min(input_valid,layer_valid)`；无效suffix不能修改本层映射或copy plan。
3. 同层KSC只消费这个最终前缀生成的copy plan，attention只读更新后的有效数。无效query在读query/plan/table/KV前跳过并输出确定性零，不能先非法读再乘零。
4. 后层进一步缩短时，前层多换入的历史token仍合法，无须跨层回滚；它们只影响以后命中率。
5. 最终有效数随model执行状态交给sampler，只传实际请求前缀view，不D2H、不改scheduled计数或packed offset。
6. Sampler/proposer消费完毕后，下轮重新从新的scheduled布局开始，禁止沿用上轮截短值。

例如初始化`[4,4]`，三层局部容量分别为`[4,3]`、`[3,4]`、`[4,2]`：

```text
有效数: [4,4] -> [4,3] -> [3,3] -> [3,2]
最终有效draft数: [2,1]
下轮若仍安排每请求4行: 必须重新从[4,4]开始
```

不另持久维护一个等价的`effective_draft_token_count` tensor，它在消费者内部由
`valid-1`派生。活动请求至少留1行用于下一frontier分布，PAD为0；预算应至少容纳一组
Top2048。跨层协议必须覆盖row释放/复用，不能只验证同一batch的一次forward。

## 13. 物理偏移不变的Sampler接线

### 13.1 动态Bonus与Hidden State

假设两请求均安排3个draft，累计query end为`[4,8]`，最后有效query数为`[3,4]`：

```text
物理TND: req0=[q0,q1,q2,q3], req1=[q0,q1,q2,q3]
query_start                       = [0,4]
effective_draft_token_count        = [2,3]
dynamic_bonus_logits_indices       = query_start + effective_drafts = [2,7]
```

Req0失效的是q3，即0起索引3的第四行，不移动req1的起点，不改累计end，也不重新打包draft IDs。
候选`AscendRejectionSampler`接收独立可选的device有效数参数；非自适应路径保持原有
行为。不能在forward后回读有效数去改`SpecDecodeMetadata.num_draft_tokens`或
`cu_num_draft_tokens`，这些字段还负责真实输入定位。

Sampler仍按scheduled偏移定位，只验证`draft_index < effective_drafts[b]`。
若有效draft全接受，在输出列`effective_drafts[b]`写入动态bonus，其余列保留
placeholder。不能固定取每请求物理末行作为bonus。

Req0两个有效draft全接受时：

```text
output = [draft_0, draft_1, bonus_from_q2]
valid_sampled_tokens_count = 3
num_rejected_or_truncated = scheduled_drafts + 1 - valid_sampled_tokens_count
                         = 3 + 1 - 3 = 1
target_hidden_state_row   = 原物理末行q3向前回退1行 = q2
```

Proposer消费bonus的embedding与q2的target hidden state。Bonus尚无自己的target
hidden state；如果更早draft拒绝，仍沿用基线
`valid_sampled_tokens_count/num_rejected_tokens_gpu/token_indices_to_sample`
链路选择对应行，不能另建一套DSA hidden-state索引账本。

未触发停止截断的基本账务可写成：

```text
num_rejected_or_truncated = scheduled_drafts - (len(output_tokens) - 1)
```

这样scheduler不必在调度前知道union容量。EOS/max_tokens、stop导致更短输出时，
必须和基线完成账务联合验证，不能只套这个公式。性能统计要区分真实reject与容量截短；
可复用回退计数，不应把截短次数误报成模型接受率下降，也不能为统计新增逐层同步。

### 13.2 Prefix-Aware采样处理

只更换bonus索引仍不够。Req0的q2原来用于验draft2，截短后提供bonus；它的penalties、
bad words、grammar、thinking budget只能看见`history + draft0 + draft1`，不能把
未验draft2当历史。因此每个target query必须形成正确的前缀分布，再在device侧动态
gather验算行和bonus行。不能把完整`sampling_metadata.spec_token_ids`交给固定末行
bonus sampler；一般随机拒绝中的修正分布也不能代替容量截短的原始target分布。

旧原案曾提出“仅无history-sensitive processor的greedy首版”，其具体范围值得保留
为待评审准入表，而不是声称当前固定MTP3已经执行下面所有限制：

| 候选范围 | 参数 |
|---|---|
| 允许 | 归一化后的greedy；max_tokens、stop string/token、EOS/ignore_eos等采样后停止条件；不影响greedy选择的seed |
| 暂拒绝，待补前缀语义 | 非greedy；非默认presence/frequency/repetition penalty；非零min_tokens；bad words、allowed-token mask、structured output/grammar、thinking-token budget、自定义logits processor |
| 暂拒绝，待补行映射 | output/prompt logprobs、指定token logprobs等 |

原案基线将`temperature<1e-5`归一化为greedy并重置top_p/top_k/min_p；未来准入仍应检查
归一化的`SamplingType`而不是原始浮点温度，新基线须重新核验该阈值。逐请求校验应在
进入scheduler前给出具体字段错误，worker/sampler再校验active batch；不能混入一条
随机请求后才让整批切换到尚未适配的分支。当前源码已确认的明确限制是greedy，本文
更细的参数表仅服务自适应方案评审。

推广随机采样需同时完成rejection kernel、动态bonus、processor前缀历史、logprobs
索引和相应数学/数值验证，不能仅扩展greedy argmax分支便宣称无偏。

## 14. Prefix-Union与Resident候选策略

只有连续前缀有意义，不能为装满预算跳过q1却保留q2。一个具体容量反例为：

```text
R = 6144
|U(1)|=2048, |U(2)|=3510, |U(3)|=5890, |U(4)|=6480
=> 本层保留q0/q1/q2，验2个draft，q2提供下一frontier分布
```

必须先确定最终有效前缀，再保护其union、选择victim和分配slot。旧原案提出的
first-fill候选不同于当前“union优先、token顺序filler”：

```text
anchor = topR(q0)
mandatory_new = U - anchor
evictable_filler = anchor - U
final_resident = anchor
                 - q0排名最低的|mandatory_new|个evictable_filler
                 + mandatory_new
```

因为`|U|<=R`，可驱逐filler必然足够；q0自身TopK属于U，不被驱逐。只保留q0时退化
成`topR(q0)`；`|U|=R`时无需filler。候选顺序是按query和query内rank稳定去重，
相等score优先保留较小token position，anchor hit保留原rank slot，mandatory token
依次复用filler slot，first-fill按最终slot顺序输出`copy_count=R`。

选择q0的理由是它必然存在，且比最后有效query更少依赖可能被拒绝的draft；不对跨query
raw score做max/sum/average，因为不同query的score尺度未必可比，还会引入workspace
与归约。Steady候选同样先保护U，再用q0 score评价非U resident。是否保留这套策略、
改用当前顺序filler或其他确定性策略，需要测量，不能混称为当前实现。

例如R=6，旧映射`10→0,20→1,30→2,40→3,50→4,60→5`，三个query选择：

```text
q0=[10,20,70], q1=[20,30,80], q2=[90,10,30]
U =[10,20,70,30,80,90]
一种合法计划: 70→3,80→4,90→5，驱逐40/50/60
attention slots: q0=[0,1,3], q1=[1,2,4], q2=[5,0,2]
```

若q3独有token使union超6，q3不参与任何eviction/copy/映射提交。先读旧状态完成
plan，再更新一次请求级映射；不能把旧单query“遇miss立即改映射”外面套Q次循环。
单stream的LIM→KSC→attention可维持当前失败即终止的边界，多流则必须增设readiness
和commit协议。Capacity truncation不是算子错误，非法ID/布局也不能被包装为正常截短。

## 15. 候选算子接口与有效区

本节用`H_idx=32/64`表示Indexer query head数，`num_resident_blocks`表示MLA
resident cache的物理块数；它们均不同于容量公式中的resident base block数N和
full Indexer层数F。MLA query head数单独记为`H_attn`，不能与H_idx混用。

`B_capacity`为预分配请求容量，`B_view`为本次active/captured请求view，
`T_capacity=B_capacity*Qmax`为query预分配容量，`T_view`为本次物理query view，
`T_scheduled`为累计query end末值。PAD请求计入captured view但不计真实请求；
物理query容量尾部不归属于任何真实请求。

原始BF16交接稿曾定义`npu_lightning_indexer_decode_update_out`接受以下语义：

```text
query, indexer_key_cache, key_weights_projection,
req_pool_entries, cache_slots(a!), row_modes,
actual_seq_lengths_query, actual_seq_lengths_key, indexer_block_table,
valid_target_query_count(b!), attention_slots_out(c!),
copy_src_token_ids_out(d!), copy_dst_slots_out(e!), copy_counts_out(f!),
tail_info_out(g!), weights_offset, reset_valid_target_query_count -> ()
```

这是**旧BF16候选schema记录**，不是当前A5七输出ABI；当前使用FP8 query/key、
stride权重输入和2304列paired source/slot。重新设计时须明确选择继承哪些语义，
不能同时拼接两套输入，或把旧`weights_offset`当成当前可传参数。

| 候选语义 | 原案合同与仍需决定之处 |
|---|---|
| 旧BF16 Indexer输入 | query为BF16/FP16 `[T_view,H_idx,128]`；合并投影`[T_view,128+H_idx]`，静态weights_offset=128；key为完整dense Indexer cache |
| Query分段 | 复用累计`actual_seq_lengths_query[B_view]`，不新增等价私有offset/count tensor |
| 物理容量 | `0<=T_scheduled<=T_view<=T_capacity`；累计end非降，PAD请求差分0；图容量尾部不属于新请求 |
| 有效数 | int32 caller-owned in/out `[B_view]`，首selection层重置，后层只缩不增；attention/sampler只读 |
| Request copy | int32 `[B_view,1,16384]` src/dst及`copy_count[B_view]`；KSC仍纯IO，不感知截短/接受率/tail |
| 旧BF16 attention表达 | 连续`[T_view,1,2048]`历史slot和`tail_info[T_view,2]`，后者为start/count；不能套给固定parity下可能回绕的当前tail |
| 当前A5可复用基础 | 七输出2304列显式表达每个tail slot/source；若增有效数，必须重新评审schema、图和消费者，不撤销当前空位-1合同 |
| 原生MLA载荷 | 旧BF16/FP16 NOPE `[num_resident_blocks,128,1,512]`、RoPE `[num_resident_blocks,128,1,64]`，key/value alias；当前C8为opaque packed载荷，不能互换 |
| 旧BF16 attention输入/输出 | query/output `[T_view,H_attn,512]`，query_rope `[T_view,H_attn,64]`，与query同dtype和TND行序；output无效行写零 |

旧SFAO提案把`valid_target_query_count`放在`actual_seq_lengths_query`与
`actual_seq_lengths_kv`之间，属性为`layout_query=TND`、`layout_kv=PA_BSND`、
`sparse_block_size=1`、`sparse_mode=3`，MLA query head数H_attn候选为1/2/4/8/16/32/64/128。
这些只说明旧接口的维度与寻址意图；不能声称当前A5 binding已支持同一范围，更不能
只删旧SFAO的`T==B`校验而不修复kernel中request行和query行的混用。

历史前缀与因果长度的候选推导为：

```text
Q_b = query_end[b] - query_start[b]
durable_kv_length = optimistic_kv_length - Q_b
D = floor(durable_kv_length / 128) * 128
Vq = durable_kv_length + local_query_index + 1
SPARSE历史候选 = [0,D)，tail = [D,Vq)
DENSE候选 = [0,Vq)，不追加SPARSE tail
```

D还必须对应真实可读的durable前缀，不能仅由算术推导代替L2 ready/dump合同。
旧canonical current/next双尾通过交换物理block ID维持逻辑连续，因此start/count足够；
当前固定parity无需每轮交换表，tail可能回绕，已用显式2304列plan表达。二者只共享
“历史与tail不重复、每query因果可见”原则，不共享tail ABI。

### 15.1 读写与Poison测试

旧交接稿要求：无效query不读query/attention plan/tail/block table/KV，输出确定性零；
KSC只读`[0,copy_count)`，zero-count不读cache/table。旧研究稿同时出现“未用suffix
未定义”与“必须-1”两种表述，未来ABI必须消除歧义：只有消费者先依据有效长度跳过的
区域才可未定义；若attention按固定宽度读取，则必须给合法空值。当前2304版本明确用-1，
不能以旧稿的免清零优化为由删除。

所有mutable state和caller-owned输出需同时贯通Torch schema、Meta/fake、CANN
OpDef、ACLNN adapter；mutation/alias不得丢失，mutable输入不能通过
`AutoContiguous()`生成临时副本。旧提案的合并投影`[T,128+H_idx]`加静态offset，
与当前A5的`index_weights`行stride方案都是避免隐藏打包的选择，最终只保留一份真源。
从16384宽copy buffer截2048列仍非紧凑矩阵，禁止用隐式copy掩盖ABI差异。

Eager与Graph共享固定owner；可预知metadata/scratch初始化时分配，forward仅view、
原址刷新或out操作，不能新增`.cpu()`/`.item()`、host callback、list-to-tensor或
逐层设备分配。普通attention激活输出仍可走框架返回tensor，但需能正确capture。
Workspace按shape/attr给出上界，不能以runtime union大小触发host重分配。

### 15.2 TP一致性的前提

原案依据的是DeepSeek/GLM Indexer在TP域复制：`wq_b`为replicated linear、融合
`wk_weights_proj`使用`disable_tp=True`，每rank计算全部Indexer heads，并有逻辑
等价、物理独立的完整key cache。主干MLA query head切分不等于Indexer也被切分。
未来实现必须重新核验模型拓扑、量化和并行配置，不能只依赖该历史事实。

一致性指逻辑query/key/weights、长度/mode、候选token集合和resident映射演进一致；
不要求TP各rank的物理block ID或pool行号逐整数相同。Tie-break可采用逻辑token
position再resident slot等明确规则，不能受pool行或核调度影响。调试可选择少量层/step
比对valid/copy_count/cache摘要，但默认热路径不应未经设计加入逐层TP collective。
DCP/PCP/KVP等若改变Indexer拥有范围，必须重新建立协议，不能外推当前单机DP适配。

## 16. 未采用的容量回退路线

| 路线 | 公式/做法 | 取舍 |
|---|---|---|
| 独立verification scratch | `S=max(0,Qmax*2048-R)`；布局为`[resident R][scratch S][tail A][tail B]` | 保持原resident档位但新增scratch生命周期、两类slot编码、attention统一寻址、graph/容量报告维度及回收策略；不能称为零HBM成本 |
| 近似union裁剪 | `union_score(t)=max_q(alpha[q]*score[q,t])`，alpha可按到达该query的经验概率递减 | 可能同时改变target logits、接受率、最终分布及档位精度；不属于精确自适应，不在本方案默认采用 |
| 按query串行执行 | 每query完成LIM→KSC→attention后才处理下一query | 可作为不丢TopK的正确性reference，但增加launch和IO轮数，不是简单把四个query依次改映射后最后一起attention |
| 静态提高budget | 当前固定MTP3使用R至少8192 | 无容量截短协议，代价是更高resident占用；这已是当前实现，不是本草案提出的新fallback |

旧原案不选scratch的理由是R=6144已无条件容纳3个query，可验2个draft并保留一行
frontier分布，通常不值得再增加一个地址域。若更大Q或多层MTP改变收益，应重新测量。
近似路线只能在另行确认精度目标后研究，至少同时报告union overflow rate、逐query
裁剪token数、KSC miss/copy、接受长度和任务精度，不能仅以省IO论证。

## 17. MTP Cache路线的容量与取舍

自适应query和proposer是否卸载是两个正交选择。当前已选择方案二BF16 full-HBM；
**target/proposer一起强制C8稀疏的旧实现已放弃**。下述方案一是保持BF16后再做
proposer稀疏卸载的未采用方案，不是恢复旧非法BF16写FP8签名。

单块字节符号沿用[当前设计的容量推导](../features/speculative_decoding.md#13-full-hbm的容量推导)：
`P_TR=83968,P_TI=16896,P_MR=147456,P_MI=32768`；L为target MLA层数，F为target
full Indexer数，r为full/resident block比例，N为resident base blocks。

```text
无MTP target DSA:
  C0 = L*P_TR + r*F*P_TI
方案一，MTP也做BF16稀疏卸载:
  C1 = L*P_TR + P_MR + r*(F*P_TI + P_MI)
方案二，MTP原生BF16 full-HBM:
  C2 = L*P_TR + r*(F*P_TI + P_MI + P_MR)
HBM_KV_i ~= N*Ci
N_i ~= floor(H/Ci)
C2-C1 = (r-1)*P_MR
```

方案一的MTP Indexer仍覆盖全序列HBM，MLA才变为resident+DRAM；不能照抄旧对照表
把方案一Indexer也写成token-wise swap。方案二active decode不做token-wise IO，
但当前verified满块已经镜像到L2/L3，旧表中的“仅HBM/无任何MTP DRAM IO”已过期。

取`L=78,F=21,r=3`：

| 成本 | bytes/base-N |
|---|---:|
| C0 | 7,613,952 |
| C1 | 7,859,712 |
| C2 | 8,154,624 |
| C2-C1 | 294,912 |

方案二相对方案一的成本分母增加约3.75%，固定H下base-N减少约3.62%；相对无MTP
增加7.10%成本，base-N减少6.63%。这些不是实测吞吐，也未计入对齐/null block和
额外workspace。多MTP层应累加每层真实page size，full-HBM成本随层数增加。

### 17.1 方案一必须付出的工程成本

| 模块 | 必须补齐的合同 |
|---|---|
| Spec/容量 | 同一runtime下逐层packed-C8 target与split-BF16 proposer布局；按page size加权，不能假设resident层同构 |
| Runtime/DRAM | `packed_c8`按层真实布局；target packed arena与MTP BF16 NOPE/ROPE arena；独立selection state、dump/miss plan和release生命周期 |
| A5算子 | 验证或补齐BF16 full-block dump、BF16 KSC、BF16 SFAO，支持BF16/C8 mixed dispatcher；不能只添加SoC注册名 |
| 调度/验算 | 每outer step三次draft、accept/reject、confirmed watermark、双尾与row reuse组合；明确各层数据完成时刻 |
| 图与验证 | Proposer eager不免除target Q4 Graph同owner要求；需单算子、三算子串联及整网精度、时延和内存验证 |

收益是进一步减少长上下文HBM，未来多MTP层可能更有价值；代价是每draft新增selection/
IO、混合layout组合和更大的精度/回滚状态空间。原决策的定性估计为框架及算子“大到
特大”、约方案二3~5倍工作量，属于历史估算，不是当前排期承诺。

### 17.2 为何当前选择方案二

方案二只需第三full group及三pool加权容量、component-wise admission、原生proposer
group/table/slot接线，保留target双尾和Q4执行链；不新建A5 BF16 token-wise卸载链。
一层MTP的额外HBM有限，而三个串行draft避免了额外selection/swap时延。其风险也必须
保留：第三pool可能成为新瓶颈、`is_eagle_group`/tuple格式要正确、所有输入准备路径
都要刷新第三组slot mapping，多MTP层会线性放大内存成本。

因此“复用full-cache管理范式”不等于合并物理group，更不等于让MTP共用target的
resident position。方案二当前的prefix镜像与active full-HBM并不矛盾；研究方案一
必须以这条已适配基线为对照，而不是重新声称MTP/prefix尚未接入。

## 18. 研究交付与验收门禁

未来实现先固定接口/状态与CPU golden，再做设备链式、模型eager、Graph/DP及性能。
下面补充第7/11节的细粒度覆盖；它们不扩大当前引擎准入范围：

- Union恰为R-1/R/R+1；只留1行、全部有效、连续数轮截短后恢复scheduled Q；q0 anchor候选与当前filler做对照。
- 累计end `[4,6]`、有效数`[3,2]`；物理span不压紧；query、plan、tail、table及copy suffix分别poison，证明无效行在读取前跳过。
- Tail旧长度0/1/126/127、最大候选Q跨两块；DENSE可见长度小于/等于/大于2048；历史TopK/tail无重复，source/slot重排仍匹配golden。
- First-fill→steady，全hit/部分/全miss；同token多query只copy一次；row释放复用、prefix恢复、PAD数在多次replay间变化。
- 设备有效数跨层min和跨step重置；TP逻辑等价但物理pool/table不同；相等score与重复运行结果一致。
- 首draft拒绝、中间拒绝、全接受、只有reject suffix跨块、accepted前缀跨块后的prewrite dump，以及容量截短遇EOS/stop/max_tokens。
- Sampler动态bonus行、processor前缀和proposer hidden-state索引联合验证；一般采样扩展另做分布与logprobs测试。

旧稿曾计划算子Q=1/2/4/8/15/16、框架到15，以及BF16/FP16、Indexer heads=32/64、
attention heads=1/2/4/8/16/32/64/128的910C/A5矩阵。保留这些作为历史扩展测试意图，
实际新矩阵必须重新确认CANN/SoC、binding、workspace、capture和模型限制，不能现在
宣传为支持范围。双尾覆盖所需`Qmax<=block_size`也只是必要条件，不是硬件充分条件。

旧2176版本还曾研究“对完整因果上下文取Top2176，再过滤当前尾块至多128个token，
从而保留2048个历史候选”的替代排名路径；若只取Top2048再过滤，历史候选可能不足。
该证明只针对旧逐query当前尾块候选语义，而且依赖原生Quant-LI支持2176，原稿没有
提供这项能力的最终证据。当前2304语义直接在共同durable `[0,D)`选Top2048，再追加
双尾，不能把旧oversampling候选当成当前实现或不经重新证明沿用。

性能要同时测SQ回归、MQ相对Q次串行reference、截短频率/长度、TopK overlap、KSC
copy字节、有效/无效Q增量时延、all-zero launch、p50/p95、workspace与核利用率、
host prepare、HBM并发容量和TPOT/吞吐。不为未消费16384列后缀或PAD做昂贵计算；
不通过逐层D2H/TP同步换取表面正确。微秒目标待真实profile，不凭工作量估算填数。

交付包应包含匹配的OpDef/infer-shape/type/tiling/kernel/ACLNN、Torch schema/adapter、
mutation/alias说明、Meta/fake、CPU golden、UT/ST、多轮ACL Graph、SoC构建和
性能复现命令。硬件阻塞需给出SoC、CANN/torch_npu版本、失败阶段、最小shape/命令、
完整日志、缺失能力及已验证替代方案代价；不能把实现复杂或旧代码单query作为硬件限制。
