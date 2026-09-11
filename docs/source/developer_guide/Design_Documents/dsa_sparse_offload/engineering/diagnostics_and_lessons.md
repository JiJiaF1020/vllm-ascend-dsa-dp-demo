# 问题分析方法与工程经验

> 2026-09-11整理。本文集中维护分析方法、修复机制、关键证据与残留风险；不保存逐轮聊天/TODO或已撤销探针的运行说明。
> **已确认并修复**、**相关性但未定界**、**环境限制**必须分别标注；特性支持进度不在本文维护。

## 1. 先建立证据层次

| 观察 | 能证明 | 不能证明 |
|---|---|---|
| HTTP 200 / 脚本 COMPLETED | 请求完成、测试流程走通 | token正确、L3实际读取、worker干净退出 |
| `cached_tokens>0` | scheduler报告命中 | 命中来自L1/L2还是L3、payload逐字节正确 |
| 非零 load completed | 相应TP的IO完成回报 | 其他TP完成、模型输出正确 |
| `pending=0` / all drained | 逻辑job已排空 | buffer已注销、native资源全部关闭 |
| `Replaying aclgraph` | 实际执行过图 | 所有轮次、所有forward都执行图 |
| 本地CPU/mock UT | 可覆盖的控制面、shape、生命周期合同 | 设备DMA、通信、NPU数值 |
| token相同 | 本段输出一致 | top-k概率或全量logits相同 |
| atexit进入 | Python退出回调开始 | 后续TorchNPU/CANN析构已完成 |

先核对实际脚本路径、源码/so版本、模型、TP/DP、图模式、MTP、chunk budget、输出长度和namespace。源文件更新不改变已启动进程；只更新draft脚本也不会更新项目模块。

### 标准分析流程

1. **确定要检验的性质**：功能完成、payload字节、相同输入数值、确定性、性能或资源退出，不混用结论。
2. **固定实验条件**：记录实际代码/算子安装版本、输入token、模型量化、并行拓扑、chunk/图/缓存/MTP和namespace；用run ID与request/DP/TP关联证据。
3. **建立自对照**：Local/Local、同row重跑或同模式重复先稳定，再与L3/另一row/另一模式比较。
4. **找到首个可信分歧**：沿生产者→消费者检查输入、shape、有效范围、地址映射、水位和payload，而不是从最终文本倒猜根因。
5. **构造最小反例**：能用CPU状态模型或单卡合成实验验证的，不直接要求重新拉八卡大模型。
6. **只改有证据的边界**：保持原复现条件，验证修复前失败、修复后通过；不能靠关特性、固定分配或改容差代替修复。
7. **形成可维护回归**：把关键不变量放入常规UT；删除专用trace与消融分支，避免上手脚本变成诊断框架。

### 从症状到证据

| 症状 | 首先核查 | 不应直接下的结论 |
|---|---|---|
| 只有prefix replay不同 | 命中范围、来源、水位、相同历史概率、Local自对照 | L3数据损坏 / 硬件误差 |
| 换行/换块后不同 | 逻辑选择、slot、block table、实际payload、尾块位置 | allocator必须固定某个ID |
| UT全过但整网失败 | UT是否覆盖真实分支、mixed batch、有效输出与生产者时序 | 一定是模型或基线问题 |
| 小TP通过、大TP失败 | 卡组/内存类型/transport/每rank完成证据 | TP数量本身就是根因 |
| 日志缺失 | 真实logger handler、导入路径、运行版本和打点触发条件 | 函数未执行 / 用户没上传 |
| 关闭超时 | IO、region注销、backend、atexit和native阶段分别采样 | 当前栈顶就是永久死锁根因 |

## 2. LIM 物理行号影响数值：已修复

### 现象与定界

七输出/2304升级后的8k+70k混合请求中，同域70k热回放呈现A/B/A。两次本地热复用之间已经分叉，因此L3 get并非必要条件；Eager复现与Graph对应的16条输出，Graph也不是必要条件。

有效pool日志显示R2/R3/R4的stable pool行是1/0/1。释放与LIFO重用是合法行为，不能通过固定row0掩盖问题。

### 从复杂症状到最小反例

| 实验 | 观察 | 这一步排除了什么 / 没排除什么 |
|---|---|---|
| 四长度三轮、末轮交换DP | 5k/8k/40k热比较一致，70k概率在输出index7先变、token在index10分叉 | 同时改变了DP归属、等待与数据来源，尚不能归因L3或LIM |
| 每域单70k、不换域 | 本地/本地和本地/L3的128-token/top-k对照一致 | 证明单请求控制可稳定，不排除混合请求、部分重叠prefix |
| 加回8k+70k、不换域 | 70k恢复A/B/A；R2/R3已是HBM573块、external=0、restore=0 | L3 get与换DP不是必要条件，但write-behind仍开，不能说整个L3已消融 |
| 仅Graph改Eager | 对应16条输出与Graph相同，A/B/A仍在 | Graph replay不是复现必要条件 |
| 有效pool日志 | 两域均1/0/1，对应A/B/A；LIFO释放顺序可解释 | 只建立关联，仍需相同输入下干预行号 |
| 单卡row0/row1/row0，CPU按映射物化KV | 同row稳定；仅换pool行即改变steady布局和QSFA | 隔离掉模型、DP/MoE、L3与KSC，但不是独立LI数学golden |

首次70k差异中，1.061640739440918是输出index8的**非首选候选**98676的共同logprob差，
不是top1漂移1.06。index7/8仍选相同token，直到index10才分叉；这些概率对照尚处于相同
生成历史内。输出token位置也不是MTP target forward轮次，缺少accepted/query-row记录时
不能从“第8个输出”直接推成“第8次LIM”。R1的70k已命中6784 tokens，不是纯冷算。

受控单卡用例用相同逻辑初态、输入和query序列执行row0/row1/row0。KV由CPU按实际映射物化，刻意绕过KSC：

| 步骤 | 修复前观察 |
|---|---|
| first-fill | 一致 |
| steady 1 | 选择token集合/顺序仍相同，但resident映射/slots不同，QSFA出现差异 |
| steady 2 | 状态已受上一步影响，顺序和miss数也分叉，copy数6435/6434 |

单卡具体差值与干预边界如下；数字用于保存反例，不作为通用容差：

| step | raw QSFA max_abs | 按logical source排序后的控制 max_abs | copy count |
|---|---:|---:|---|
| first-fill | 0 | 0 | 12288/12288 |
| steady 1 | 0.000030517578125 | 0.000152587890625 | 6494/6494 |
| steady 2 | 0.000762939453125 | 0.000152587890625 | 6435/6434 |

Step1前逻辑状态相同、选择集合与顺序仍相同；step2开始前的resident状态已经分叉，
不能把它描述成每步输入状态都完全一致，也不能把6435/6434直接解释为漏拷一个token。
逻辑排序控制未消除差异，说明只排序attention列表不足以处理该布局依赖。

`HashVictimScanSeed(candidate,poolRow)`把物理位置混入替换策略。改为candidate-only，保持旧row0扫描规则，快速扫描与精确补扫一起修改。真实row仍负责寻址，未修改allocator、双尾、ABI或正式attention排序。

该seed在迁移前已经存在，不是新调用了随机数API。算子优化扩大执行路径覆盖面时，旧逻辑
可能第一次在新的mixed组合下暴露；归因需要同时比较旧/新路径选择，不能把所有异常都当成
本次新增算法。修复保留旧row0的算术：candidate=73472时旧row0/1扫描起点32/30，新为32/32。

### 修复验证与边界

- 单卡三步的状态、选择、slots、copy counts、原始/逻辑排序QSFA均严格一致。
- Eager `cc0755df...` 与Graph `3ca6ff2e...`：两域70k仍为1/0/1，8组热比较的128 tokens及返回top-5全部一致。
- 两种模式的16条完整token/text逐条相同；实际L3回载仍覆盖53本地+520外部，没有靠跳过IO通过。
- R1/R2仍有首次生成/前缀复用差异；70k R1已命中6784 tokens，不是纯冷算。它不在本次修复结论内。
- 不由此认定QSFA内部某种归约/量化实现有bug，也不认证所有形状或性能。

关键证据已汇总于本节，严格回归保留在 `test_dsa_a5_c8_ops.py` 的pool行置换用例；不再保留临时四轮脚本和整篇试错日志。

截至本次整理，原始算子迁移有框架487项/native62项通过反馈；后来新增的行置换用例为1 passed/62 deselected，不能据此前全量结果推定最终内核的全量门禁已完成。清理后相关框架CPU子集144通过、demo235通过/12跳过是本地验证，不是新增NPU证据。

**经验：相同TopK集合不等于相同驻留布局；布局变化也可能改变数值。修复验证必须确认原来的行号变化仍然发生。**

## 3. Block table 与真实尾块：已修复的独立问题

### Budget块物理顺序

较早的重复请求测试曾暴露free-list重分配后，尚未first-fill的budget块顺序变化与QSFA输出变化相关。当前仅在ENTER阶段规范化无有效KV、无prefix hash的budget块，随后按最终表填充。

**尾块不排序**，也不能重排已有有效KV的block而不搬payload。MTP双尾按逻辑parity定位，不靠physical id最大或最小判断。

### 非活动表列

短请求复用长请求的固定宽block table行后，旧的非活动列可能继续被算子读取。add、ENTER replace、condense move、clear要维护同一合同，不能只修一个入口。此类清理放在行生命周期路径，不应对每个steady step加全表清零。

### Lookahead与待dump满块

`req_blocks[-1]`可能是lookahead提前分配的空块，不是真正部分尾块。例如8319个已计算token、lookahead=3时应按已计算长度定位，而不是保留最后分配块。

首次ENTER还必须保护尚待dump的已验证满块。共享prefix引用不等于本请求可回收容量；错误回收会同时影响L2 backing和parity tail。回归应直接构造lookahead跨块、pending dump与共享引用，不能以普通长prompt“应该会覆盖”替代。

### 8319-token反例：最后分配块没有真实尾数据

已修复的旧路径可用以下整数状态复现，不必先跑整网：prefix=8064、prompt=8319，
最后一次prefill的Q=255，lookahead=3。按8322个位置分配得到66块logical 0..65；
但真实部分尾是logical 64，包含token 8192..8318，共127个token。令physical ID为
logical ID+1，正确应保留physical 65，旧`req_blocks[-1]`却保留了lookahead-only的66。

随后旧ENTER释放65，把66放入parity0。最后prefill只dump到8192，丢失的127个token
没有L2 backing；shared-D策略又会直接从HBM尾区消费它们，因此KSC无法从L2补救。
地址66可以合法，内容身份却错了，这不是越界断言能完全检出的错误。

修复以明确computed长度选择logical tail，并识别prefill已经压缩成单尾的表。
轻量回归使用真实BlockPool及C=383的同构跨块边界，检查保留/释放的物理ID与refcount；
矩阵应包含prompt余数125/126/127、prefix有无及lookahead是否跨块。CPU反例证明旧
所有权错误，不等于当时已经在NPU复现了特定数值分叉。当前做法见[总架构](../architecture.md)§8、§11.4。

### 8191到8192：prewrite之前源块已经丢失

另一项独立修复不能只用“保留部分尾块”解释。设prompt=8191、budget=8192，
prefill后computed C=8191、durable D=8064；下一轮仍为DENSE验证，draft全拒后
确认C=8192，但prewrite使用上轮C，因此D仍停在8064。下一context=8193首次ENTER：

```text
已验证、尚未dump的logical 63 -> 正确源physical 64
旧判断C % 128 == 0 -> 认为没有尾块 -> 释放physical 64
新resident表parity1列65 -> 示例新physical 10065
runtime按新表做prewrite -> 从错误源dump，却把D推进到8192
```

Indexer/proposer仍保留完整表时可能都正确，只有同bundle的MLA payload错误。
所以“prewrite发生在本层KV写前”不是充分保护，源可能早在scheduler阶段被回收。

修复从成功输出的ready水位到`ceil(C/128)`保留pending满块及部分块，最多两尾，
按logical parity映射。保留的满块要经公开`evict_blocks`去hash；去hash不会解除peer
引用，refcount>1时先保持一轮DENSE完成dump再ENTER，部分尾必须独占。Prefix关闭时
也维护MTP ready事实，但不因此开启hash发布或改free-queue策略。

正常一步满足`D=128*floor(C/128)`、`Vq<=C+Q`、`Q<=4`时，`Vq-D<=131<256`，
只能证明双尾容量足够，不能证明物理块内是正确payload。HBM-only大gap应在prefill补齐，
不能靠SPARSE一次跨多个新durable块掩盖。Terminal没有下一outer step时可能只是少缓存
一个满块，不应直接报数据错误。水位与双尾的现行合同见[推测解码](../features/speculative_decoding.md)。

### Refcount反例：释放65个引用只增加33个空闲块

旧ENTER预检曾把“本请求有多少existing块”当成“可以回收多少块”。构造65个existing
块，其中32个refcount=2、33个refcount=1，池中另有free=1，ENTER需要66块：

```text
旧估计：1 + 65 = 66，误判可分配
真实新增free：33，实际只有1 + 33 = 34
```

旧流程到`get_new_blocks(66)`才失败，此时resident表已清空、Indexer也已经分配过，
破坏了“容量不足时不改变状态”。这是合法共享prefix加低余量，不是allocator允许跨请求
可写alias；free充足的正例确认新budget与peer持有的32块不相交。

修复只计本次释放后归零的独占非null块，保留源不计入可回收量；所有pool预检先于
改表、阶段和hash。CPU回归直接断言失败返回None且表/refcount/free不变，比“请求最终
完成”更能检验事务边界。这不扩展为preemption或新的COW协议；详见[总架构](../architecture.md)§11.4。

## 4. MTP L3 发布早于 proposer：已修复

Target执行结束时，MTP proposer的BF16 cache可能还没写完或dump到L2。若在target的`get_finished()`直接把整个三组件bundle标为ready，L3可能保存旧MTP payload，即使target Indexer/MLA完全正确。

当前在target完成边界暂存job，proposer完成对应dump并核对epoch后，才发布保存事件并提交。验证不能只看generic byte-copy round-trip；它能证明复制算子，却不能证明框架发布时序正确。

同理，scheduler computed、已接受token、L2 ready、L3 finished是不同水位。后缀已回载但前面的HBM-only gap尚未回填时，不得跨洞发布完整prefix。

固定MTP3在DP下还需验证本域epoch和任务归属：另一域proposer idle不能完成或清除本域
暂存job，`job_id=0`在两个域同时出现不是冲突。已有单40k三组件回读各目标TP为324块、
2,295,392,256 bytes，恢复78个target层及1个MTP层；8k/70k交换域回读分别为59/573块、
177/1719 keys、417,988,096/4,059,443,712 bytes。两域prewrite/postwrite和全TP恢复
提供真实路径证据，但不替代同key不同writer的组件字节等价、processing失败或完整数值验收。

2176到2304的选择策略变化与本节发布时序错误是两件事，旧MTP并非没有双尾存储。
策略和跨块四query对照见[推测解码](../features/speculative_decoding.md)，这里不重列另一份ABI。

### 独立的dtype接入实障：W4A4C8名称没有量化layer78

更早的一次方案将GLM-5.2的MTP layer78与78个target层统一绑定packed-C8 resident。
算子UT通过，首次端到端却在`npu_mla_prolog_v3`报161002。读取checkpoint safetensors
dtype元数据后发现，该层Indexer、MLA、MoE/norm权重实际是BF16；当时非法组合是
BF16 token/投影权重与FP8 packed `kvCacheRef`，不是“整个W4A4C8模型就该用C8”的推论。

诊断顺序应分别查weight/token计算dtype、cache生产算子、cache布局和消费者签名。
MTP是同一个完整层执行三次，不是三份可任意换格式的cache。临时“BF16 compute+C8
MTP cache”修改被撤回，强转spec/dtype不能补出量化生产者和对应消费链。

当前采用独立native BF16 Indexer+MLA full group，第三pool与target Indexer同coverage，
proposer eager且不走LIM/KSC/QSFA token-wise卸载。现行dtype/tuple与容量合同见
[推测解码](../features/speculative_decoding.md)§1。保留此例是提醒检查checkpoint事实，
不是宣称所有CANN版本永久不支持混合签名；若重研BF16卸载方案，须完整评审mixed-layer链。

### 第三pool创建了，第三份slot mapping仍可能漏刷

从两pool迁到三pool时，spec、BF16 metadata、`kv_cache_gid`和分配可以全都正确，
但若`_prepare_inputs()`仍只刷新target Indexer/resident两组，MTP消费的寻址就会陈旧。
来源记录了这一已修接线遗漏，没有提供可引用的错误token或崩溃日志，不能编造症状。

MTP full group必须以真实`positions`及本轮query span刷新，不能用target resident
压缩position；proposer有自己的draft-step数组也不能省掉首次第三group映射。之后才由
基线`slot_mapping_group/seq_lens_group/query_start_loc_group`推进draft子步骤。
轻量回归让各group物理ID不同，覆盖B>1、长短混批、row重排/复用、target ENTER后
MTP仍full，以及连续请求释放；只断言“group数等于3”抓不到此错。

旧`compact_dump_jobs`清理针对的是稀疏proposer上下文，不否定当前full-cache prefix
hook。双尾防覆盖和confirmed prewrite的完整时序仍见[推测解码](../features/speculative_decoding.md)；
本页第3节保留scheduler更早丢失源块的独立反例。

## 5. DP / MoE 分支相关差异：未最终定界

此前**关闭MTP**的双40k并发热缓存测试，首decode与对侧72-token prefix-prefill相遇时，MC2/ALLGATHER及padding规模发生变化，首概率漂移与其时序吻合。全Eager仍复现；固定ALLGATHER消融内14组比较一致。

这仅证明执行路径相关性，不证明MC2有bug，也不证明DSA完全无关。尚缺不启用DSA的干净同条件基线或相同输入的逐模块数值对照。BF16 dispatch消融没有A5整网反馈。

该问题与MTP LIM pool-row问题不同，不能随LIM修复一并关闭。正式代码仍使用基线自动MoE选择，不保留固定通信方式或BF16诊断分支。

保留的关键复现条件是：A5 DP2/TP4/EP8、关闭MTP、双41672-token请求、热命中41600、64输出、greedy、chunk budget16384。自动MoE的Graph和Eager轮均为9组匹配/5组不匹配，概率先在输出index1漂移、token在index9分叉；固定ALLGATHER轮14组内部对照一致，且保留了首decode/对侧72-token prefill相遇条件。

后者的输出也不必等于前两轮，因此不能用它证明哪条路径数学上正确。已有CPU元数据摘要没有覆盖全部设备输入/原始KV，也不能当作payload逐字节一致性证明。继续定位应补干净基线或同输入模块对照，而不是恢复临时开关强行通过。

### 必须保留的实验条件与证据范围

每域的对照序列是seed、local-0、local-1、external-0、post-local-0、external-1、post-local-1；
external前清两域本地、保留L3，cold seed不与热路径作同算法通过判定。每run新salt、每域
salt不同、域内各阶段固定。`max_num_seqs=4`、`max_model_len=131072`、seed=1024、
ignore_eos=true、top5 logprobs也是这组条件，不能换一套chunk/batch参数后称为严格消融。

| 运行ID后缀 | 条件 | 结果 |
|---|---|---|
| `d1d4fb082c9c4c91a7f1a1389f8bf61d` | DP0串行40k | 7组64-token/top-k一致，真实L3读回 |
| `d725dfc1682647f28f2f0d3f0e7749bc` | DP1串行40k | 7组一致，但服务重启过，不是同进程并发消融 |
| `0a307109a437427f830aa6d717b8b377` | 同服务8k/70k并发 | 14组一致，不替代双40k条件 |
| `6b8e0fdbe13b4167915c47e11d26930d` | Graph、自动MoE | 9组匹配/5组不匹配 |
| `46a4ebfb665341b482d14de1ba360f01` | Eager、自动MoE | 同类分歧，9组匹配/5组不匹配 |
| `64d5118384fb41ae8d242693b88a2f40` | Eager、固定ALLGATHER消融 | 14组内部一致，真实P/D相遇覆盖 |

最后三轮各186条选定执行记录，仅覆盖decode input offset 0..11，不能说完整捕获64个输出的
所有forward。固定ALLGATHER的P/D与D/D同域对照只在DP0覆盖；不能扩大为两个域都覆盖。
自动模式首差token为98841/99185，最大共同候选logprob差3.357579231262207；固定模式的
第10个热token又是98940，进一步说明“本轮自洽”不等于“它数学上比另一条路径正确”。

首步已记录CPU字段相同（computed=41672、candidate=41600、budget=10240、pool_index=0）
也没有覆盖设备table、slot和KV字节。`pad_size=0`不代表MoE内部没有按DP最大token数padding。
MC2融合量化、独立dynamic_mx_quant、scale、专家分组和归约都是候选变量；仅凭文件diff或栈
落在基线代码，不能完成责任归属。

### 好坏路径对照如何缩小假设

早期双40k一轮中，DP1本地重复、external重复各自稳定，但跨路径4组比较失败；另一次
同run、同进程、同salt却是external-0坏、external-1好。坏路径由DP1先完成load，DP0
随后才开始；好路径两域同时开始。两次都是Local miss、L3 hit325、每TP读取
2,243,904,000 bytes并非零restore，因此不能把“每次读取该对象都固定损坏”当唯一解释。
反过来，没有逐字节校验，也不能据一次好读回完全排除恢复或初始布局问题。

同服务8k/70k通过时，DP0/DP1实际prompt为7752/73496，热prefill剩余query为72/24，
不是双40k的72/72；load仍错开，只是先到域相反。它排除了“任何错位必错”，不证明
只有并发是根因。DP1串行通过发生在重启后，缓存、分配与capture初态也改变，不能称为
同进程仅关并发的消融。Cold seed首token55959、热路径5161的另一差异不在这组热门禁内。

完整trace补上了同轮对侧状态，而不只靠wall-clock猜测：

| Graph AUTO中的同步轮次 | DP0 | DP1 | 观察 |
|---|---|---|---|
| 298/394 | 首decode，q=1，ENTER stage=2 | prefix-prefill，q=72，stage=0 | DP0为NONE/ALLGATHER，两个异常请求 |
| 299/395 | decode offset1，stage=3 | 首decode，stage=2 | 恢复双方decode及FULL/MC2 |

两请求后11个已记录decode步骤都恢复FULL/MC2；先前一步的影响不会因此自动消失。
9 matched/5 mismatch中的5是比较关系数，只涉及这2个异常请求。全eager在另一轮的
228/452也出现对应P/D首步，故无需FULL replay即可复现。单请求无MTP的prefill产生
输出index0，首decode产生index1，恰好解释为何应从首概率漂移处而非index9才追查。

### 采集方法本身也要验收

旧定点trace的186条由80+80+26三段合并，校验14请求offset0..11连续、无重复或破损JSON；
只读既有CPU同步结果/InputBatch，不新增NPU读回、collective或barrier。同步可能被基线
bypass时显式记`performed=false/reason`，不能沿用旧快照；模式在dispatch后仍可能覆盖，
所以实际forward context建立后还要记录最终mode、skip_compiled和MoE方法。

`exchange`是本worker实际同步计数，不是scheduler step，更不能跨PID对齐。原始上报可能
已有SP/capture padding，因此称`reported_tokens`。本例本地actual/padded/num_reqs仍为1，
改变的是global max 1→72、context padded 4→72；candidate=41600不是topk=2048。
未知设备scalar不能为打印隐式转int/str；`post-local`是客户端阶段名，不能替代IO来源日志。

客户端pair模式用单调时钟检查同轮两域RPC在途重叠、成员及签名完整，失败先保存同伴响应，
不自动取消/重试。缺域、不同批次或无重叠不能仅凭token相同PASS；RPC重叠仍不是server
forward同轮证明。1e-6只是首个可见logprob漂移的标记，不是量化模型容差。

未上板的BF16 dispatch候选有明确隔离目标：保持AUTO selector和MXFP4权重，仅在MC2
用quant_mode=0并移除FP4 y_dtype，检查返回BF16/full hidden，再令dynamic_scale=None
走MLP独立量化；combine使用本次原生tp_recv_counts，不能沿用量化dispatch的占位counts。
有效实验还需8worker返回回执及同域P/D、D/D覆盖。它增加通信/临时activation成本，不是
性能方案或已验证修复；正式分支已撤出，现有日志研究不要求恢复该开关。

## 6. 冷算、prefix和batch差异怎样比较

1. 确认输入token完全相同，不只看文本标签；删除序号前缀会改变输入，本身也是实验变量。
2. 确认生成历史仍相同，再比较概率。首个token分叉后的logprobs已处于不同上下文，不能当根因。
3. 先做同模式、同长度、同命中边界的Local/Local和L3/L3控制，再看Local/L3。
4. 冷算与长前缀复用的prefill范围不同；batch形状、chunk切分、MoE实现也可能变化。不要求无证据地归咎L3，更不能无证据地归为硬件噪声。
5. `max_num_seqs=1`通过只能排除该串行条件，不证明多请求路径正确。在线和离线共用模型引擎，但到达时序、batch组成、缓存历史可能不同。
6. 基线DSA与本分支的尾块全保留策略并不总是相同；跨策略结果不能作为同算法严格golden。

摘要应记录相同历史内首次概率变化、候选集合变化、首次token差异以及最大共同候选差值；只检查候选交集差值0会漏掉候选集合不同。

### 热命中不等于执行过恢复

DP Local prefix的16请求试验中，7冷请求命中0、9热请求有命中；无MTP时5k/8k/40k/70k
分别命中4864/7680/41600/73472 tokens，固定MTP3时为4736/7552/41472/73344。
每项少128是Eagle边界回退；但两轮热请求均HBM pair/triple与L2等长，hbm_only=0，
restore=0。因此它们证明目录/隔离与HBM复用，不证明非零L2→HBM或HBM-only gap回填。
后续跨域L3回读的非零restore才补上该条搬运链，仍不能替代所有独立容量组合。

缓存命中也不保证端到端低延迟：一次热40k仍命中41600，却约28.30秒才结束，与对端冷70k
约28.50秒接近；MTP下也有30.76/31.40秒的对应观察。共享EP等待与此相容，不应因热请求
慢就诊断本地缓存失效。这些是总请求耗时，不是TTFT、纯IO带宽或性能承诺。

### 配对控制与旧脚本判据的强弱

无MTP的A5 TP8单5k，Local-L2/L3各38块、每rank 262,364,160 bytes，32输出token严格
一致；混合8k/5k/70k/40k时5k在第5个生成token分叉。关闭L3 load后的两轮Local/Local
复现相同请求/位置的token与top-logprobs，启用/禁用L3的四条结果按replay轮次对应。
这排除了“分歧只发生于L3恢复”，不证明所有Graph数值稳定，也不免除组件bitwise门禁。

MTP TP8单40k的eager与FULL Graph曾分别通过Local-L2/L3严格token控制，Graph的
cold/local/L3三轮也一致；这不替代混合请求。MTP混合回放有两处分歧在无L3的Local/Local中按同request、common-prefix、token对
复现，期间实际触发一个decode prewrite新增durable块。历史在线证据还包含24个真实
L3 load plan，每个在TP0..7完成三组件get/restore；不能仅用HTTP 200或top-k互含代替它们。

旧`online_l2_l3_closeout_test.py`的输出必须按实际判据解释：

| 判据 | 实际比较什么 | 结论强度 |
|---|---|---|
| 严格token比较 | 相同历史下的token IDs | 有差异仍是严格比较未通过 |
| 同源精确分叉签名 | 同request/位置/common-prefix及token对已在同源trace复现 | 支持该分歧并非跨介质独有，不证明全部payload正确 |
| 同prompt波动包络加reciprocal top-k | 同prompt已不稳定，跨源候选互含；仅Graph且放宽配置开启 | 比精确同源签名弱，不能说该次差异已精确复现 |
| `exact_fingerprint()` | text、finish_reason、token_ids | 不比较logits；名字中的exact不扩大比较对象 |

`build_same_source_variance_envelope()`汇总Local-L2内部和L3内部trace对；
`find_cross_variance_absent_from_replay_controls()`可能接受后两类不同强度的归因。
历史标签`top-k-equivalent`不表示数学等价。不满足相应规则的跨源独有分歧仍应失败调查；
bsz1/eager严格门禁不被豁免。只有组件bitwise、命中长度、全TP get/restore及生命周期
等证据分别成立后，才能限定地记录数据路径通过与同源非确定性，不能单凭Local分叉免检L3。

## 7. Mooncake / H2H / TCP：先分辨路径

裸CPU跨NUMA memcpy通过，不代表注册后的HCCL/HIXL/URMA路径同样通过；也不能从一组失败推导“Ascend或Mooncake完全不支持H2H”。必须区分主机/设备内存、分配方式、注册范围、实际transport、卡组与两端进程。

本项目的A5跨卡组Host H2H曾在底层URMA完成队列报错，包括`HostCpuUrmaChannel::ChannelFence`的status:4。日志位置不自动等于根因归属，不能仅凭数字猜测通用错误含义。

原始NPU wheel的某些使用路径硬编码或自动安装AscendDirect，不等于该wheel必然没有TCP。Store自管engine的`setup(protocol="tcp", engine=None)`加真实buffer注册、descriptor和跨进程回读，才证明具体TCP payload路径。

当前采用Store-owned TCP的L2↔L3，回载L3→L2→HBM。不能把TCP Host路径通过和另一次AscendDirect HBM路径通过拼成“二者在同一Store配置中可任意混用”。也不引入device中转占用HBM/H2D/D2H来默认兜底。

MemCache是否支持同一场景取决于它实际用的transport和版本，不能仅凭也是NPU后端就推定结果。当前可运行探针入口见{download}`API测试资料 <../../../../../../examples/dsa_demo/L3_cache_sys_api_test/README.md>`，其早期实验状态不替代当前特性支持矩阵。

### H2H失败的逐级排除链

以下是当时A5、Mooncake `0.3.11.post1` 依赖组合的历史证据，不能推广成任意版本的
H2H能力结论。保留每次改变的变量，比只记最后的URMA/HCCL错误码更有价值。

| 门禁/对照 | 实际证据 | 能缩小的范围 |
|---|---|---|
| 基线HBM路径 | `global_ub_device` 单卡/TP2/TP4/TP8 multi-buffer和TP8环形remote get逐字节通过；DSA-disabled GLM-5.2有3200/3287 external hit、每rank 25/25 keys、304,742,400 bytes、failed=0，零HBM hit下真实L3→HBM且warm token/logprobs稳定 | 证明基线HBM数据面，不证明DSA swapped Host |
| 单卡exact schema | 42个Indexer plane、78个MLA plane、242个registration regions、110,477,312 bytes；两个component、interior地址、digest及同engine HBM+swapped注册通过 | 先验证真实布局，不把简单连续buffer当完整DSA schema |
| 同卡组Host放大 | TP0用1 MiB swapped source显式写TP1 segment并完整回读；120-buffer/6,904,320-byte小arena通过，再扩到每plane 4353 blocks、约30 GiB注册及非零offset，主线程/后台线程都逐buffer一致 | 大MR、scatter-gather、offset、后台线程设备上下文并非单独的充分失败条件 |
| 卡组分界 | TP0→TP1/2/3可推进，→TP4/5/7的120-descriptor Host write约30秒后失败 | 观察到NPU0..3与4..7卡组/NUMA相关性，尚不是根因 |
| 缩为两worker | NPU0/NPU4、loopback控制地址、双key显式放NPU4；mount/register/connect成功，Host put在`HcommChannelFenceOnThread`返回19（该日志的HCCL_E_NETWORK） | 排除必须TP8、模型并发、Master随机放置等解释，不证明硬件不支持跨NUMA |
| 同端点改HBM | NPU0 HBM→NPU4进程Host L3→NPU0 HBM，以`global_ub_device`逐plane通过 | 不是整个跨组网络或共享池都不可达，收窄到Host source/Host endpoint direct H2H |
| 换Host分配介质 | `global_ub+pinned` 的NPU0/NPU6单descriptor仍在fence失败，分配/注册/建链均成功 | SVM、大arena、descriptor数不是必要条件 |
| 去掉mixed endpoint | `global_ub_host+pinned`，只有`ub_ctp:host`，两端`aclrtMallocHost`仍失败 | 不能简单归为Host/device混合endpoint选路 |

同组双key真实布局是Indexer 42个交错16,384/512-byte planes，MLA 78个83,968-byte
planes；put返回`[0,0]`，get返回`[354816,6549504]`，总计6,904,320 bytes。
NPU0/NPU4的HBM控制也得到这些字节数及逐plane一致。它们比“get成功”更能说明
完整对象已回读；region数量则取决于探针source/destination组织，不能直接当生产plane数。

还曾遇到更早的**重复注册**问题：`global_ub_device`只有device endpoint，会对已由
torch_npu注册的swapped Host再次执行`aclrtHostRegister`，报`halHostRegister drvRetCode=12`。
`global_ub`的host/device双endpoint避开了该路径，但之后跨组Host fence仍失败。
不能把修好注册等同于修好传输，也不能用pinned诊断替换dump/KSC/restore共用的swapped arena。

源码允许Mooncake注册MEM_HOST，HIXL也将Host/Host分类为COMM_TYPE_UB_H2H，A5示例有
`ub_ctp:host`；但这些入口存在不等于目标跨NUMA路径已获保证。通用NUMA-aware NIC
选择也不能证明Ascend Direct远端路由实现。没有经验证的显式route时，不再排列组合
参数，更不把NUMA/rank拓扑写入DSA scheduler。后续Store-owned TCP通过相同Host门禁，
所以旧失败不能写成当前整个L3不可用。

### Staging失败发生在哪一层

HIXL `OPTION_BUFFER_POOL` 可用于Host direct不可用、注册受限或小块传输，但路径
选择有前提。默认`adxl.BufferPool=0:0`，当时通过非零`ASCEND_BUFFER_POOL`启用；
诊断值`4:8`表示每组4个8 MiB buffer，
两组约64 MiB HBM/进程。6.59 MiB逻辑块可装入，更大对象按8 MiB切片循环，池大小
不随30 GiB L2 arena同比增长。额外H2D/D2H、Device带宽及HBM成本使它不是当前默认fallback。

| 诊断配置 | 失败阶段 | 正确解释 |
|---|---|---|
| 显式GlobalResourceConfig/protocol_desc，例如`global_ub+4:8` | 新HixlEngine初始化即PARAM_INVALID，只允许0:0 | 还没传数据，不是Host put失败 |
| `comm_mode=auto`进入CommEngine/AdxlInnerEngine | 两端pool已创建、Host注册跳过，随后`HcclCommInitClusterInfoMemConfig(ret=1)` | 已越过原注册问题，又遇到另一条初始化兼容边界 |
| 补示例要求的`HCCL_INTRA_ROCE_ENABLE=1` | NPU0/1与0/6仍相同失败 | 不能将剩余问题简化成少了这个环境变量或仅跨组失败 |

当时旧rank-table generator有server_count/server_list/status/version，却缺所用HCCL
parser要求的顶层rank_count；2026-08-28检查的HIXL master `61a2ee4` v1/v2 generator
仍缺该字段。这是冻结依赖组合的兼容证据，不是当前DSA必须修的内核bug；项目未接管
HIXL/HCCL兼容层，私有补丁已撤销，也不从这一轮推断新版本仍然相同。

### Wheel、注册日志与实际payload transport

中途曾误判USE_ASCEND wheel强制AscendDirect、因而TCP不可用。实际是旧AscendStore
wrapper注入外部ascend GlobalTE，使该调用配置到不了Store-owned TCP；不注入外部engine，
使用`setup(protocol="tcp", engine=None)`的Store自管方式随后在原NPU wheel通过。
因此“必须重编wheel才有TCP”不能保留为最终结论。

Wheel可能安装多个transport，registration也可能遍历它们；TCP配置日志里出现HIXL，
不证明payload走HComm。应核对目标segment descriptor的protocol，再看实际双向byte
回读，不用初始化日志替代数据路由证据。当前支持路径与注册合同见[分层缓存](../features/multilevel_kv_cache.md)。

另有更早的环境Gate 0：`mooncake.store`或`mooncake.engine`单独import后退出即报
`corrupted size vs. prev_size`，两种组合顺序也exit134，发生在Master/IO之前，不能
归为KV精度失败。同期MemCache/MemFabric Hybrid 1.2.0的host_shm MetaService、
ConfigStore、LocalService及Host scatter-gather bitwise通过，但显式close/SIGTERM触发native double-cleanup；探针避免重复cleanup只保护
当时测试，不能宣布正式退出已验。它们是不同依赖、不同阶段的问题，不与今日DP post6
native退出自动合并归因，也不覆盖后来的Mooncake主路径成功证据。

### 依赖审计快照

以下是上述能力判断及后端选型所依据的冻结源码，供复核API和实现边界，不表示当前
服务安装了全部组件，也不代表各项目最新能力。同版本号的服务器wheel仍需实际门禁，
不能仅凭版本字符串认定其注册、transport或退出行为相同。

| 项目 | 冻结版本 | 源码commit |
|---|---|---|
| Mooncake | v0.3.11.post1 | `e9c61075720039bcfc5fffd19f847608402be3d0` |
| MemCache | v1.2.0 | `4310d68fa8964a3032b71c09296b0ffa80330033` |
| MemFabric | MemCache固定submodule | `9a9cfbc8a1e026e7c73a558ff27d75372cb2e803` |
| LMCache | v0.3.12 | `78697950e9dc243087c5ef8683e693258d7f6df3` |
| LMCache-Ascend | v0.3.12 | `58e0aef462eda7c451f84735b277b35e02dce227` |
| kvcache-ops | LMCache-Ascend固定submodule | `98fe68db9a2cef360244f3e0487ce564667e61a8` |
| UCM | v0.5.0 | `52b0f545d993b6cf76323826d37683c750098891` |

### TCP从小探针到真实请求的计数闭环

原`mooncake-transfer-engine-npu=0.3.11.post1`以Store-owned TCP直接注册swapped L2，
NPU0↔1及跨NUMA0↔6双向bitwise通过；TP8每rank贡献segment，4-plane/1 MiB和
120-plane/6,904,320-byte两组source/destination环形get，put为0、get为完整字节数。
小arena只有2个physical blocks，这还不能替代真实几十GiB注册和后台增量发布。

| 后续门禁 | 能相互核算的数字 | 对应证明 |
|---|---|---|
| TP8离线40k，block128/chunk8192 | TP0注册120 regions、30,054,504,960 bytes；`64*5+5=325` durable块，6 jobs、650 component keys、2,243,904,000 bytes，其他TP提交0 | 大MR、每chunk增量和唯一producer；Graph复测计数同eager、token逐个一致、实际replay且无PAD额外store，该轮8worker正常退出 |
| 四长度Graph/chunked | watermark按当时请求顺序为`(60,39,574,325)`，合998逻辑块；去重944唯一块、1888 keys、6,517,678,080 bytes，少54块是共同hash | 5k decode新增第39块与跨请求去重并存，不应把逻辑块总和当上传对象数 |
| 同组后台完成 | 16 jobs全部成功，最后2个在输出后shutdown drain完成；planner_calls=1856 | finish不阻塞write-behind，不等于这些job失败；高zero-token polling仍有性能观察价值 |
| 在线TP8，chunk16384 | 40k `128+128+69=325`，decode再到326/327；5 jobs、327 blocks、654 keys、2,257,712,640 bytes | 流式HTTP、Graph和增量写并行；producer-only external hit=0合理，服务未关闭所以无shutdown验收 |

跨DP的同salt并发中，也观察过一方先写完、另一方650 keys已存在、bytes=0；那证明
去重，不证明真正processing writer失败/重试窗口。40k尾部72 tokens时，32输出还未跨
128边界，64输出出现单块6,904,320 bytes上传，才有decode新增durable块证据。

读路径也要闭环：一次在线no-MTP控制中943个唯一prompt bundle只写一次，3组L3回放
每rank加载`3*943=2829`块，并核对全部rank的restore、无重复store/残留；HTTP全200
只是其中一项。早期eager另有6worker在job和segment注销后超grace被SIGTERM，应单列
teardown时延，不能算put失败或与后来DP native停留点合并根因。

### 构建和容器教训

- Ubuntu容器挂载整套宿主`/usr/lib64`可能混入另一发行版的linker script，出现引用不存在`/lib64/libm.so.6`；不应靠盲建软链接继续混用glibc。按需挂载驱动/通信依赖。
- pip显示成功不代表native依赖齐全；检查yaml-cpp/jsoncpp/gflags等CMake目标及最终link line。发现`-lyaml-cpp`失败时需查实际库搜索路径，不是只补一个Python包。
- CMake版本与第三方项目的最低policy可能不兼容；隔离构建工具链，不随意升级推理环境setuptools或替换TorchNPU。
- APT临时文件/GPG错误先检查`/tmp`权限和写入，不把它直接当作仓库签名失效。
- 本项目提供的隔离构建属于诊断/兼容工具，不要求原始wheel已通过TCP门禁的环境重新编译。详见{download}`构建说明 <../../../../../../examples/dsa_demo/L3_cache_sys_api_test/README_mooncake_npu_tcp_build.md>`。

## 8. DP管理与身份：已修复的集成缺口

同hostname不等于同共享内存范围，不同hostname也不等于不同宿主；当前单机预算用boot ID确认范围，且要求显式一致的hash seed。DP域ID属于连接/control身份，不属于共享内容key。

管理广播只返回某个engine的结果，不能证明所有域都成功。reset必须校验全域参与者和namespace，未知失败不自动resume。反过来，DSA专用管理不能覆盖启动时的`reset_mm_cache`等基线无关utility；曾经出现的启动干扰已修正为限定DSA管理入口。

`out-of-order step`告警要区分统计通道与模型执行协议。它本身不证明forward乱序或缓存IO错误，需结合请求进度、实际通信与状态证据。

### 两个启动字段相同，实际身份却不同

旧compatibility signature只记录`PYTHONHASHSEED`字符串；两域都未设置时同为空，
签名可相同，但真实`init_none_hash()`分别调用`os.urandom(32)`，首块hash不同并沿
父链扩散。模型sampling seed=1024和选择sha256均不能消除这个随机根。用真实hash函数
构造两根即可在CPU复现；固定114514是对照组，旧示例本就固定，因此已有跨域命中不矛盾。
现要求显式有效seed并保留原文，`1`与`01`不能先转整数再比较。

另一旧协议用hostname判定共享内存域。同名`dsa-host`容器可能在不同宿主，同宿主也可
不同名。真实双进程Gloo的最小回归覆盖同boot异名成功、异boot同名拒绝、单侧probe
失败后双方共同错误；探测失败先参加同一次汇总，检查typed observation、rank全集及
本地样本，不先让一域退出而留下另一域等待。单机SPMD的local-DP=1/global-DP=2
可以合法，简单强制相等会误伤它。这些是CPU协议验证，不是跨机/NPU部署验收。

字段一致也不能保证未来分配成功：公共最低mirror检查发生在扣L3预留前，逐域最终
容量检查还可能单侧失败；共同快照不是物理预留。准确预算及host/seed合同见
[数据并行推理](../features/data_parallel_inference.md)§7，不把身份修复扩大为极限内存/失败传播全验。

### Bool AND不能修复已经分叉的namespace

旧独立`generation++`反例：初态0/0，DP0空闲允许reset到1，DP1因pending store拒绝仍0；
首engine返回True时前端可能误报成功。即使改成AND，也撤不回DP0；排空后重试可能成2/1。
因此需要全域prepare与统一epoch，而不是只换返回值。当前已修，不能写成仍在发生的bug。

正常8请求管理试验先实际命中DP1 running=1/l3_pending，最后store完成后才两域
prepare/commit；23个store全部配对（19put/4dedup），4次回读各含全4TP非零get与
78 target+1 MTP层restore。External reset使两域旧salt冷，新epoch随后双向可读。

扩展9请求进一步把A放进已暂停DP0的waiting：两次prepare均明确waiting=1，对端为0；
local和external reset后均未擅自resume，显式resume后A按新epoch冷算，对端再命中41472。
6次reset两域commit一致、local不换代，28store全部完成。**Client未返回不够证明已进
Scheduler队列，waiting=1才补上这个观察。** 最后pause不释放NPU/内存；该次服务后来强杀，
不能追认正常退出。

PAUSED_NEW已经禁止admission，若本地phase gate仍因waiting prefill阻止running decode，
排空会被自己的暂停条件卡住。这项交互已修；不能先停掉connector stepping再等pending IO。
管理CAS owner、HTTP取消后的shield任务、关键段未知失败保留owner/fatal的现行协议见
[数据并行推理](../features/data_parallel_inference.md)§9。无TTL接管或持久rollback；
多前端竞争、取消/超时测试主要是TCPStore/BlockPool/原生pause Future的CPU证据，不是上板故障注入。

管理接入还曾误禁`AsyncLLM.reset_mm_cache`，启动即报DSA不支持该utility，尚未到模型或
Mooncake问题。回归用1/2个API client到2 DP的真实入口及patch/EngineCore调用链，只替代
socket/设备，检查MM/encoder仍走原生广播、不触发prefix reset或DSA管理。缺BalanceScheduler
字段的测试夹具问题要另修，不能靠把基线调用全mock掉掩盖接入范围。

### out-of-order的最小统计反例

Coordinator先收DP0 `(wave=1, step=181)`，再收DP1 `(1,180)`，用跨engine共用高水位
即可发warning；不要求任何单域发送序号倒退。Wave是整体活跃期，step含dummy，不是
请求编号、层号或MTP子步骤。相等高水位实际可接受，日志`expected >`不是完整判定规则。

Warning后仍更新该engine的waiting/running，不回滚KV、token或请求归属；显式rank请求
不依赖这条LB评分。落后一轮且请求完成可记非阻塞观测，但持续扩大、单域自身回退或不再
推进必须另查队列和wave，不以此例保证所有同名warning无害，也不为消日志加forward barrier。

## 9. 退出超时：已知生命周期限制

全域L3 drain及region注销/backend关闭已有证据；后续八worker仍在原生退出窗口内未结束。早期快照看到`empty_cache`、process-group销毁，只能说明采样时所处位置，不足以断言永久卡在那里。

延续到atexit之后的快照显示全部worker停在TorchNPU `2.10.0.post6`、git `a8f2f4fda83732c0d02244a46815a81f21fe3439` 的 `_C._npu_shutdown(success)`调用处。该定位说明等待已进入native清理，内部具体步骤及DSA资源使用是否触发仍未完全确定。

因此不把它当作L3 IO未排空继续反复改Connector，也不通过延长超时或`os._exit`伪装干净退出。保留限定单机DP正常功能闭环与独立退出限制。后续若继续调查，应看已有native shutdown阶段日志或更小资源复现，不重复申请机器验证已排空的L3路径。

### Summary在close之前，不能证明资源关闭

一次离线DP2/TP4、Graph+MTP3/TCP的12请求cold/local/cross全部完成128输出，
33store（25put/8dedup）配对、16条目标TP非零get/restore、各域inflight及worker pending
均0，全域drained后却依次4秒等待、SIGTERM、再4秒、8worker SIGKILL。这来自原生
executor策略，不是日志显示用户外部抢占；worker summary当时在backend.close之前。

为分清边界，正常关闭移到全域drain之后、death-pipe关闭之前，复用存活worker RPC：
仅同步当前模型stream，注销L2、close backend、清理原生KVConnector引用；本域必须
收齐4个True，再全域closed后执行最后CPU barrier。失败、缺rank、False、timeout
不能计closed，timeout=0的abort不加提前关闭RPC；父级总预算仍有效，不给各层叠加300秒。
不做全设备同步去等待SDK常驻通信流。完整协议见[数据并行推理](../features/data_parallel_inference.md)§10。

后续六请求实测把资源边界真正补齐：八worker当前模型流同步0.205–0.299毫秒，
123 regions注销1.924–1.985毫秒，backend close为2.094–3.182秒；两Core各在
2.440/3.185秒获全4TP回执，随后仍SIGKILL。因此L3资源关闭已验，不能把未完成的
OS退出继续写成L3在途IO，也不能反过来把第二层完成写成第四层干净退出。

注册量也要分清：这类离线记录中每rank的39,907,236,352 bytes是L2 arena注册量，
123 regions是布局/注册计数，不是该rank贡献的L3 segment容量。父client退出和
`INFERENCE/CACHE PASSED`都不能替代全部worker的退出状态。

### 日志文字可能误导退出定位

`Parent process exited, terminating worker queues`来自death-pipe EOF，表示开始通知
退出，不证明EngineCore已死亡或worker主线程已完成队列关闭。它使用local scope的
info_once，只见TP0也不能推断其他TP未收到。更晚出现的`_host_emptyCache`兼容warning
来自Core、在worker SIGKILL之后，捕获的是AttributeError，不是本次worker卡点证据。

应分别核对逻辑job、region/backend、Python回调及OS终止，不能把不同进程/时间的
最后一条warning拼成一条因果链。

### 探针第一处盲区：两跳序列化

旧专用退出探针经`LLM.collective_rpc`传callable。首轮只验证executor→worker的
cloudpickle，遗漏client→EngineCore默认Msgpack安全门禁，实际在第一跳拒绝function，
虽然生成zip却`worker_files=0/8`，根本没有线程栈。这是探针交付问题，不是模型/传输回归。

当时修正只在可信离线子进程中按需开启官方pickle fallback，并在加载模型前测试真实
Msgpack、两跳回执和失败不创建LLM；不能复制到对外服务或不可信输入环境。现在正式
示例已经移除探针、callable诊断RPC、zip和专用pickle开关；普通status/reset Msgpack
无需这个开关。保留方法而不恢复旧操作入口。

### 探针第二处盲区：到达atexit就停止采样

早期八worker在3秒快照均接近`_cleanup_profiling_kv_cache -> empty_cache`，但
3.588704–3.596955秒都已到python-atexit标记，绝对时间跨度8.199毫秒且早于后续kill。
所以单次empty_cache快照不能认定最终阻塞。探针晚注册的atexit回调按逆序先执行，
它取消采样时，较早注册的TorchNPU等库回调可能还没运行，产生后半程盲区。

Through-atexit版本保持文件描述符至进程结束，标记`sampling-continues`，不取消或
重注册库回调、不提前手动shutdown。CPU真实子进程按同步、PG析构、runtime关闭三阶段，
各覆盖正常/强杀，共六场景，证明标记后仍能采样；定时`Timeout`标题不是新推理超时。
Python finalization之后纯native析构仍不保证能提供Python栈。

最终实机约3.37秒到atexit，6/9秒两次快照停在同版`runtime_lifecycle.py:15`；前面的
`_npu_shutdown_synchronize`和Python `_destructor_process_group`已返回，但栈不证明
`success=True`。Native内部还有ReleaseHcclCommList、HostFinalize、host/device/swapped
allocator、NpuSysCtrl Finalize；swapped allocator与DSA L2资源类型有关，Python栈
不足以细分归因。原生路径文件无DSA差异也不能排除资源交互，INFO没开启时无plog打印
不等于调用未发生。结论始终是未在原生窗口内完成，而非已证明永久死锁。

## 10. 测试交付准则

- 先让轻量UT/合成实验验证真正的假设，再申请八卡模型；编译慢时明确哪些文件改动需要编译。
- 探针先验证真实输出链路：曾用未配置INFO handler的logger导致整轮无日志，mock `logger.info`不能发现这种缺陷。
- 原始输入、run ID、配置、摘要持久化到`results/`等明确目录，避免只放`/tmp`；日志过滤保留request/DP/TP身份。
- 判断“通过”要有断言。只打印EVIDENCE_ONLY的诊断不能作为精度gate，pytest无异常也不能自动升级结论。
- 修复后保留可复现的最小回归，清除运行时trace、临时通信分支和测试专用开关；上手脚本恢复简明默认流程。
- 变更后的完整测试数、skipped和依赖隔离方式必须记录；不要用一次全量门禁覆盖后来新增的case。

算子UT还要区分有效copy-prefix与未承诺写出的scratch尾部。只能对合同规定的有效范围作精确比较；不能读无效槽位后把差异误判为算子错误。整数metadata错不能靠放宽浮点容差解决；负例设备断言需独立执行，不与正常回归混在同一个损坏上下文中。

修复前测试应真实失败，修复后通过，并确认原分支、行号变换、跨块及IO条件仍发生。单卡CPU物化KV绕过KSC的实验有明确隔离价值，但它不是完整搬运链验证，也不是独立LI数学golden。

### 元数据重构的字段级对照

元数据重构先验证语义，再改变存储。两条管线最终得到同一种 forward view，并不证明中途没有各自重算阶段、预算、候选域或块表。应从同一份最终 InputBatch 行序构造 eager active-prefix 与 graph captured-prefix，对真实请求逐字段比较 row mode、stable pool entry、candidate、resident position、HBM/DRAM 表及 dump 的有效 job；额外 PAD 行单独验证安全值，不混入 active-prefix 等价断言。

先保留这些断言，再去除没有消费者的桥接和重复 request/list 物化，最后验证 owner 复用后的固定地址、row reuse、多轮 mutation 和 dump 可见性。一个字段不被当前 graph 直接读取，不足以证明它可删：它可能仍负责下一轮调度、durable/hash 发布或资源释放。该方法只比较同一版本、同一 attention 语义的两种执行路径，不要求 A3/A5 或旧新 MTP 策略逐 token 等价；CPU parity 也不能代替 NPU/Graph/性能验证。

### 日志探针的真实门禁

曾经`init_logger(__name__)`创建的`vllm_ascend.dsa_offload.resident_pool`继承root的WARNING，
而vLLM默认只给`vllm`树配置INFO handler，导致整轮没有pool日志。测试mock掉`logger.info`
后只证明调用发生，绕过了真正的level/handler gate。后来用真实StreamHandler构造
WARNING root / INFO vllm层级，旧实现实际0行、新实现3个事件，才确认了修复。

因此“没打印”先检查运行版本、触发条件、logger有效级别与最终handler，再判断函数是否未执行。
不能因为探针自身失效，先让使用者反复重新部署模型或推定代码没有上传。探针修好也只是让证据
可见，不是业务根因已修复；少量日志仍可能改变host时序，不应当作性能基线。

### DENSE边界不能只测2048以内

早期non-MTP接入候选只在SPARSE做LI，DENSE直接发布连续slots且假设长度不超过2176。
框架的sparse activation默认6144，实际2177..6144仍可以是合法DENSE，因此“DENSE”
不代表不需要Indexer。正确区分是长度<=2048才可连续发布，>2048必须真正执行LI，
最后partial块也不能选择超出有效cache的token。

轻量边界组取1/2048/2049/2176/2177/6144，分别检查是否实际进入计算、输出token范围，
而不是只检查mode字段和返回shape。该候选差距已由当前实现覆盖，不能称为当前MTP
2304选择bug；算法与DENSE合同见[总架构](../architecture.md)§6.5、§11.8–11.9。

### View的stride与mutable storage是两道门禁

32头例中`weights=kw[:,128:]`的stride可为(160,1)，query scale却紧凑为(32,1)。
若共用weights行offset，q=1将从scale160而非32读起；只有q=0的单行用例和全相同scale
会隐藏误址。回归须多行且各行scale不同，分别验证tensorWeightsOffset和tensorQueryScaleOffset。
当前A5接受weights行stride，合法(33,1)等非对齐行距也有逐行DMA路径，不用额外
`.contiguous()`掩盖接口；旧BF16接口传完整kw加offset只是当时的替代设计。

另一个容易漏的view是`[B,1,16384][..., :2048]`：B>1时行stride仍为16384，并非紧凑
2048列attention plan；也不能为了多query按Q复制整份16384列scratch来回避寻址设计。

Mutable `cache_slots`或out buffer若被AutoContiguous复制，device写入的是临时storage，
runtime/Graph持有的原地址未更新。Schema标了mutation不意味着adapter自动正确。测试应
对照schema、Meta/fake、OpDef、ACLNN、真实storage/stride，并多轮replay验证原buffer
确实被更新，不能只比较临时返回值的首轮数值。当前QSFA query连续化是激活布局处理，
因此“weights零copy”也不能扩大成整个forward零copy。详见[推测解码](../features/speculative_decoding.md)§14。

### 删掉T等于B检查，不等于单query内核支持TND

旧BF16 SFAO的设计反例使用两请求Q=[4,2]、累计end=[4,6]、最终KV长度[1004,702]：

```text
T=6, B=2
请求0四行的因果可见长度：1001,1002,1003,1004
请求1两行的因果可见长度：701,702
```

Host原来拒绝T!=B；只删除检查，kernel仍以bIdx=0/1读前两行query/topK/tail，不能得到
上述六行结果。必须真实区分request span、local query和TND行，block table仍按B行，
不可把所有数组一律扩成T。用不同请求长度、不同query值及PAD/query/table poison能在
小样本揭露错误；无效行要先判定，不可先读取非法值再乘零。

这是旧接口扩展反例，不是当前A5七输出实现仍有的bug。未来有效query截短导致bonus
行改变、不能伪造reject等更复杂反例，已在[自适应推测草案](../proposals/adaptive_speculation.md)
§12–15按候选保存，不作为当前固定MTP3采样缺陷再复制一遍。

### R2有shape，未必真正执行QLI

历史ragged native测试包含R2，但R2在第一状态为DENSE16、后两状态为PAD，都会被
`TaskNeedsQli`跳过。Head32、groups7给B28/T70才进入tile4，小T10只覆盖tile2；不能
以测试名字有R1..4或张量shape含R2认定各计算分支已验。

具体检查发现第一状态实际LI宽度仅{3,4}，第二状态全局cold不能计作fast partial，
第三状态为{1,3,4}，仍缺R2。当前已增加无first-fill的R2 SPARSE-steady状态补齐。
首组一个entry置负budget触发global cold，同时检查其他steady是合法混合覆盖；
late-group cold、短Q cold和all-firstfill大batch还需各自构造，不能互相替代。
独立native R1 oracle和真实swapped copy有各自价值，但单算子ragged Graph不认证
model runner mixed-Q FULL；覆盖范围见[总架构](../architecture.md)§17。

### Dummy的B、Bp和初始化语义必须同时正确

旧padding用`repeat(Bp)`处理长度B的counts，得到的是B*Bp而非Bp。取B=2、Q=4、
Tp=16、Bp=4，就会用8个counts给4个ends赋值；B=3时得到12。B=1→Bp4恰巧正确，
标准B=Bp capture也可能避开原条件，故不能说默认demo必崩。轻量测试必须刻意构造
真实行数与最终descriptor行数不同。

仅修数组长度还不够：旧capture只给原B写seq_len，却令全部Bp行SPARSE。B1→4时
ends=[4,8,12,16]，final=[8196,0,0,0]，新增三行并非合法first-fill。Runtime idle
全PAD允许final=0，capture synthetic SPARSE不允许借用这个豁免。修复用native最终
Tp/Bp/Q统一counts、ends、seq_len、dummy owners，并阻止后续FIA padding退回原B。

DP proposer还有独立的行数误推：对端长prefill把共同padded tokens对齐到16384，
旧`num_tokens/(3+1)`会推成4096条dummy请求。本域应按num_reqs有界准备，实机曾
观察16384 tokens/1 dummy request/3 draft steps，以及8 tokens/2 rows。共同token
宽度与本地请求数不是同一维度，仍需保留三次通信调用。

Dummy slot失效化也必须早于复制它的attention metadata构建；事后清generic表无法
清掉已复制旧slot的后端表。Capture helper的“无真实owner”假设不能移给仍有pending IO
的runtime idle。当前PAD/no-op的完整合同见[总架构](../architecture.md)§13.5与
[DP集成](../features/data_parallel_inference.md)§6；模式选择日志不能替代真实L1/L2 canary。

早期实际Graph矩阵还专门使用active=2的5k/40k，capture sizes只有1/4，覆盖PAD和
非精确capture尺寸；另有四长度Graph/eager及MTP3组合。不要把它缩写成“有Graph测试”
而丢掉此配置。旧MTP尚无DRAM镜像的容量结论不能用于今天三组件预算。

### Schema没有加载，测试还没进入kernel

早期prefix full-block测试5项全在schema查询失败，根本未执行kernel。A5的
`enable_custom_op()`遵循社区策略禁用全局旧算子选择，测试应显式
`load_custom_op_library()`；缺schema不能定性为copy kernel错误。

之后2 passed/3 skipped：A5不构建的三个BF16 paired-plane用例按注册能力skip，
generic byte D2H/H2D仍是硬门禁。它证明既有`KvCacheFullBlockDumpBytes`可反向用于
prefix loadback，无须再造一个copy kernel；不等于五项全过，也不是当前测试总数。
MTP3/FULL、同LLM四请求两轮聚合hit rate从5.3%到52.3%只证明命中/replay发生，
长输出不同仍需no-MTP/eager/bsz1/MAX_TOKENS=1等分层控制，聚合比例不替代请求级水位/字节。

### UB live-range与安装包完整性

融合工程的历史64头实验静态UB约261664 bytes，超过248 KiB=253952 bytes上限。
复用LI结束后的resMm1UB和TopK merge结束后的mrgValue，将占用压到249376 bytes，
余4576 bytes。可复用方法是按最后消费者分析live-range，而不是只把所有InitBuffer
尺寸相加后盲删buffer；这些数值属于历史实验，不是当前七输出LIM的UB或性能实测。
同理query从GM只读真实G行，2G是dual-destination的L1对齐，不能把padding当真实输入。

QLI的InitBuffer结束后Reset切入manager，不能在运行中再调用破坏该生命周期的
InitGlobalMemory；当前使用已有manager UB做Duplicate+DataCopyPad。Tokenwise pool
通常以256个INT32行宽隔开以避免多AIV共享64B DCache line，紧凑miss_count/tail_info
经MTE3发布；极限raw-width非对齐写回仍需单独核对，不能仅因边界合法就宣称性能相同。

融合还能避免把每请求8 KiB的topK写到GM后再读回，以及一次ACLNN/tiling/kernel边界，
但这是减少搬运和launch的设计动机，不是测出的吞吐提升。正式性能仍需profile。

打包也有独立门禁：OpDef/InferShape/Tiling按固定后缀分文件，CMake直接挂def，
`add_modules_sources`另收集推形/tiling。聚合成普通.cpp或只复制kernel可能漏掉对应
对象库，符号源码存在不代表包可部署。应验证实际产物注册链，见[总架构](../architecture.md)§16.5。

### 离线预检不能只相信脚本开关

一次旧离线脚本打印L3=True，四个cold请求也完成128输出，但两域L3 reservation=0；
external reset返回成功后，“必须拒绝”的负例断言失败。实际没配置connector时，原生
external部分可no-op，同时本地清理成功，并非DSA guard失效。旧脚本借用其他demo构造
函数且不检查kwargs可在CPU复现此漂移；服务器具体在哪一步丢配置没有证据，不编造归因。

后来改为自包含参数，加载前核验kwargs/依赖/Master，加载后检查真正EngineCore的
DP/TP、MTP、connector和fingerprint。正向cold/local/交换域回读留给昂贵推理，配置
拒绝留在UT。12请求修正轮中local的16条restore全0，cross四请求各4TP非零restore，
区分了HBM复用与实际L3；失败原文在cleanup前落盘，避免被退出日志盖住。

另一个预检bug使用`env_value or file_value`，而backend只有None才回退文件；显式
空字符串本应覆盖且报错，子进程NATIVE_RUNTIME_ENV还可覆盖父MOONCAKE_MASTER。
八个轻量用例修复前4失败/4通过，修复后8通过，覆盖文件、父环境、子覆盖、空值且
不修改父环境。这比启动大模型后才发现预检错地址更直接，不改变实际后端策略。

### 报告日志锁与清理等价性

多行容量报告经v0.23前缀包装逐段write，进程内logger锁不能避免两个EngineCore交错。
仅对DP>1容量报告使用同UID临时FileLock，持锁覆盖logger+flush，finally释放，5秒获取
失败时告警降级；不增加collective、不改预算，也不保证所有模块日志全局串行。
Windows释放后可能删除lock文件，UT应验证共享路径及可重新获取，不以文件残留当锁证据。
实机四份报告各自完整、报告间仍穿插worker日志，正符合这个限定范围。

探针清理不能只把默认值设False。历史清理核对eager/Graph×MTP×L3八组合的LLM kwargs
不变，参数构造/校验/引擎核验/数据读取/空闲等待/输出检查的AST不变，worker仅去除诊断
调用；保留drain、全TP回执和幂等关闭。Framework UT不导入examples，示例自测却必须
加载真实脚本，不能复制被测函数自证；缺脚本同步也不应轻率扩展成新的目录兼容框架。
不同轮CPU passed/skipped不能累加，抽取原方法体17项通过仍不等于正式proposer模块导入
和NPU执行。normal poll与reuse/shutdown flush应共用完成记录，避免future已完成而
console无completed的假阴性。

## 11. KSC错误传播、混装与有效范围

### 安全跳过不等于按合同失败

曾审查的候选KSC删除了设备断言，将负count改成0、超cap改成cap，或在有效prefix里
发现坏pair时静默跳过。当前迁移已恢复错误传播。保留以下反例是为了避免未来优化重复引入，
不是说当前仍使用旧候选的静默行为：

| 输入 | 旧候选表现 | 必须保留的合同 |
|---|---|---|
| `copy_count=-1` | 0次pair读取/复制，正常返回 | 错误不能等价于正常0-miss/PAD |
| `copy_count=16385` | 只复制前16384项 | 必须报告超过输出容量 |
| `copy_count=1, src=-1, dst=3` | 跳过坏pair，无明确失败 | 有效prefix内不能出现静默空洞 |

结构检查只验证shape/dtype/device，不读取设备count值；不能在host加`.item()`来补错误链。
目的地址合法也不证明里面属于该请求，错误owner的block ID可能完全落在有效范围。因此需要
producer、L2目录、KSC及后续QSFA的分层验证，不能把KSC ASSERT当作payload身份校验。

### LIM入口gate不是2304选择策略bug

旧LIM入口让各AIC/AIV独立读可变的正/负budget决定cold/fast分支。B1/R4通常调度两个
mixed groups，但cold只由core0实际工作；若早到manager已把负budget改正，迟到group
可能读到新值而走另一分支。READY/ACK与AIV-wide barrier的作用是在所有核完成入口
判定后才放行写入。这是源码时序推导及对应保护，不能充当历史所有精度差异的根因证明。

当前gate已存在。它与2176→2304更改选择集合是独立维度，不应绑定成同一个bug；
AIV-only KSC也不能照搬这套AIC/AIV配对事件。回归应刻意覆盖迟到group、一个entry触发
全局cold以及其他steady参与者，而不是只看B1最早group成功。现行gate和workspace
合同见[总架构](../architecture.md)§11.9，KSC的错误传播与混装案例仍按本节分别检查。

### 未使用的新字段仍会改变二进制布局

KSC曾在`totalPairSlots`前增加两个uint32字段。按当时自然布局模型，结构40→48字节，
该uint64字段偏移32→40；旧kernel读新host数据时，会将`{128,83968}`拼成
360639813910656，而不是`B*16384`。这是混装反例，不是同版本host/kernel必然有错，
实际编译器布局仍需检查。字段即便没有被kernel显式消费，也不能因此认为只替换kernel安全。

升级门禁要覆盖host/tiling/kernel、生成API、binding与graph重捕获；同名符号存在不证明
输出数量、alias和结构布局已一致。负例设备断言独立进程执行，避免污染后续正常测试上下文。

### 字节复制与队列的验证方法

有效packed row为656字节，UB按32字节向上取整到672只是分配容量，不能复制672字节
覆盖邻行。行首相对32字节对齐在0/16间交替；数学区间不重叠也不能替代真实MTE粒度、
邻行guard和swapped-memory readback验证。

深度2 FIFO跨batch排空时，已经入队项必须使用当时保存的完整uint64源/目的offset，不能
用后来刷新到另一请求的cached表基址重新解释旧项。队列满、只有1项、末尾排空、count全0、
跨batch仍有inflight都需要构造；0-copy只意味着不访问payload，不意味着零元数据读取。

历史CPU调度/FIFO模型枚举6708组并检查大于4GiB偏移，只证明控制流/算术在模型中符合预期，
不模拟MTE2/MTE3、TQue事件、UB可见性或真实相邻AIV写入。LIM内部gate也不是KSC的
跨kernel同步协议，不应为“稳妥”给KSC加同ID的WaitFlag。当前producer→KSC→QSFA与
full→shared→next-full的同流消费关系仍需真实设备验证。

## 12. 性能归因与元数据准备

以下保留旧 v0.19 Host prepare 分析中仍有用的计时方法、数量级反例和解释链，
实现结论已核对 v0.23 源码。`a73948c7d`、`cc3da68bc`、`fd7643757` 是历史
样本身份，不是当前性能基准；旧阶段待办、测试计数及毫秒数均不代表本版本已验收益。

### 计时边界、wrapper 与不双算

先确认所测的是 Host enqueue、Host blocking wait、设备完成还是端到端输出。
当前 `execute_model` 的 `prepare input` 包含状态更新、输入准备、图选择和 attention
metadata；`forward` 还包住 forward context 与 connector 上下文，不等于纯设备图。
`_model_forward -> self.model(...)` 会进入模型/编译/ACLGraph wrapper；wrapper
里的 replay 前同步、提交及参数更新不能被误标为 Transformer kernel 自身耗时。
这些实际范围见 [model runner](../../../../../../vllm_ascend/worker/model_runner_v1.py)。

同一线程、同一轮的嵌套范围应按 inclusive/exclusive 记账：

```text
T_inclusive(P) = end(P) - begin(P)
T_exclusive(P) = T_inclusive(P) - duration(union(child_intervals within P))
```

只在子区间互不重叠时才能直接减其耗时之和；不要把父区间、孩子、孙子再全部相加。
例如 `_prepare_inputs` 已包含 `compute_slot_mapping`，一次 `_update_states`
也可能包含多次 row replacement。不同线程/stream 的重叠工作不能套用这个单线程
公式相减，device kernel 时长也不能再加到已包含等待的 Host wall time 上。
先逐轮计算差值再求分位数，`p90(parent)-p90(child)` 不是 exclusive 的 p90。

已 capture 的算子在正常 replay 中不会重新逐层执行原 Python 包装、binding 参数
检查和 host tiling；这些是 capture/Eager 的成本，不能按层数再加进 steady replay
的 Host 准备账本。图内 kernel、图内 copy/no-op 节点和 graph replay runtime 本身
仍有设备/提交成本，是否重新 capture 或走了 true eager 必须从实际执行确认。

Profiler span 也不是零成本：进入/退出、事件记录、名字构造及逐层次数都会扰动短函数。
配套 vLLM 的 `record_function_or_nullcontext` 会选择并缓存 record_function、NVTX
或 nullcontext 工厂；“不记录 trace”不等于所有调用/对象开销都消失。局部细分用于
定位，最终收益需用同样条件的低扰动测量复核。这里不恢复旧 trace points、开关或脚本。

### 同步等待为什么会移动

同步 API 的 Python 父函数表示“Host 在哪里等待”，不表示“谁发起了所有被等待的工作”。
对某个 fence F，可用以下式子描述尚未完成的依赖：

```text
HostWait(F) = max(0, finish_time(dependencies_of_F) - host_arrival_time(F))
```

Host 准备更快、提交更异步时，可能更早抵达后面的 fence，使那段 wait 反而变长。
前面加同步可让后面 wait 变短，删除同步也可只移动时间而不减少设备计算。
不能以“sample_tokens 变长”认定 forward 被搬进 sampler，也不能以某个 sync 变短
单独证明优化有效。

当前 [ACLGraphWrapper](../../../../../../vllm_ascend/compilation/acl_graph.py) 的
普通 FULL 主模型路径仍是 `current_stream().synchronize()` 在 `replay()` **之前**。
它可以等上一轮尾部或本轮已提交的 input/slot metadata，却不可能等尚未提交的本轮
replay。ENPU 与 Eagle draft 等分支有例外；SFA 的 `update_graph_params()` 是 no-op，
也不能单凭这一点删除具有并发排序背景的通用 barrier。

旧 DeepSeek-V3.2、无 MTP、batch FULL decode 的现象可由两种时间线解释：

```text
async scheduling:
  replay(i-1) -> lm_head/sampler(i-1) 异步入队
  Host进入_model_forward(i) -> pre-replay sync等待已有工作 -> replay(i)

sync scheduling，无MTP:
  replay(i) -> lm_head(i) -> sampler(i) -> sampled IDs D2H
  sample_tokens -> _bookkeeping_sync -> _to_list -> event.synchronize
```

第二条 event 不只等少量 token 的复制，还等 token 产生所依赖的整条计算链；因此
下一轮 replay 前的同步可能很短。旧报告推测“基线自动启用 async、DSA 显式关闭”
与现象吻合，但没有两份有效启动日志和展开到最底层的同步栈，仍不能记为已确认因果。

v0.23 配套 vLLM 的 async 默认配置仍可能从 `None` 解析为启用；当前 DSA 配置则
明确要求 `async_scheduling=False`。公平对照应让基线也显式使用 False，不能为追求
相同 profile 外观打开 DSA 尚未适配的 async。当前无 MTP 单输出仍调用 `_to_list`
的 pinned copy/event fence；固定 MTP3 的多输出走 `RejectionSampler.parse_output`
并读取 CPU 结果，不能把所有 sampling 等待都写成同一条 `_to_list` 栈。

继续展开实际调用还能区分 grammar 的 logits D2H、logprobs 转换、随机数辅助 stream
event 和其他显式计时同步。两边配置相同却位置不同，应按 rank、单调 iteration、
原始时间戳及跨 stream 依赖核对，排除 i/i-1 错位、实际 Eager、不同 capture 阶段及
profiler correlation 偏差。诊断标记不能为了打印而 `.item()`/`.cpu()` 回读设备状态。

### 两组历史准备开销案例

**案例一：不足 3 ms 到约 40 ms 不是 37 ms 纯 Python 计算。** 2026-07-12 的
`a73948c7d` 分析记录了这个用户观察，源码中发现最多五组串行 gate CPU collective
（TP/EP 还可能覆盖相同 ranks）、CPU query layout 先上设备又回 CPU 的 barrier，
以及 eager device batch 再复制到 graph batch 的双重物化。未上板测量，不能确定
它们分别贡献多少，也不能把静态候选链写成全部 37 ms 已归因。

当时的保守源码提交数估算为：

```text
约18次 list -> NPU tensor
+ (1 + 2B) 次 query-position full/cast/copy
+ 约7次 HBM/DRAM gather/cast
+ 21..26次 graph-buffer copy/fill
= 约47..52 + 2B次图外任务；B=4时约55..60次
```

No-op cast 的处理依赖运行库，源码调用数不是精确 runtime task 数。其价值在于解释
“少量 metadata 为何仍可能慢”：提交、allocator、同步的固定成本可压过 payload。
21 个复制字段中当时至少 7 个没有有效图内消费者，因此应先做消费者审计，而不是
为所有中间值建立永久 graph 输入；这不是要求删除当前 MTP 有效 query/plan 字段。

同一旧布局取 B=4、请求容量 C=256、最大长度约31K、block size S=128、M=243、
budget=2048，HBM 表宽17：21个输入共253364 bytes，约247.4 KiB；其中不被图内
消费者使用的全局 DRAM 表为 `256*243*4=248832 bytes`，占约98.2%。固定表规模是：

```text
global_table_bytes = 4 * C * ceil(max_model_len / S)
```

即使按读写双向计也不到0.5 MiB。不能仅凭这些字节解释40 ms，更应检查同步等待与
碎片化任务。该比例不包含跨 step 的 resident status，也不是当前七列/DRAM布局。
后续 `cc3da68bc` 的另一小 slab 估算为 `B*(7*4+16*4+8+1)=101B bytes`
（未含 alignment），同样说明不应先把小表归为带宽瓶颈；两份快照的字段/表宽不能混算。

**案例二：B=23 的逐行更新比字段数量更值得先查。** 2026-07-25、`fd7643757`、
上下文约21K 的旧观察如下。模型未记录，数字主要来自可视化单次范围，不是 p50：

| 旧函数/范围 | 观察值 | 当时需核查的机制 |
|---|---:|---|
| DSA adapter `update_states` | 约4.67 ms | 逐请求、逐cache group的reset再append |
| `BlockTable.reset_row` | 单次100+ us | 同时清NPU/CPU/NumPy，当时后两者是同一storage的两个view |
| `MultiGroupBlockTable.append_row` | 单次50+ us | 成功路径也构造诊断list，多group写行 |
| input-state refresh | 700+ us | O(B)字典扫描、逐scalar列写入 |
| `_prepare_inputs` | 1.8+ ms | 含原生输入、resident与split-group slot mapping |
| `compute_slot_mapping` | 300+ us | 先排首次Triton JIT，再数各group实际launch |
| graph gate | 367+ us | 重复NumPy扫描与小数组创建 |
| DSA forward metadata | 310+ us | 重复校验、语义到物理slab序列化、表staging |
| attention metadata | 740+ us | 分group builder测量，不能先怪新增view字段 |

这些范围有嵌套，不能求和称为总准备时长。当时 P0/P1/P2 的“已实现待复测”没有
后续性能数填入台账；更不能沿用为本版本“已消除多少毫秒”的承诺。

### 旧优化在 v0.23 的实际落点

当前架构合同已在[总架构](../architecture.md)§10、§13、§15说明，这里只补迁移
时容易误读的性能差异，不重新列 ABI、旧六阶段计划或旧回调流程：

| 旧优化方向 | 当前源码事实 | 不能扩大的结论 |
|---|---|---|
| 避免整体fork `_update_states` | ENTER前窄范围正规化，调用 `super()._update_states`，最终行序再投影 | 不等于scheduler/worker控制面没有O(B)工作 |
| 去掉逐行NPU reset | `replace_row`/`clear_row`/`move_row`在CPU清陈旧列，ENTER后用原生表提交 | 当前replace清整条CPU行；没有旧dirty/unchanged跳过H2D合同 |
| 合并语义与物理owner | `DSAInputBatchCacheLayout`七列CPU/device镜像，Eager/Graph共享view | 不是旧mixed-dtype byte slab，也不再经过旧eager real_batch二次物化 |
| 减少重复查询与校验 | 行序一致时批量写列，改变时按request映射；candidate/dump用持久NumPy scratch | gate仍有阶段范围的NumPy检查；不是旧summary-only实现，也不是零扫描/零对象 |
| DRAM增量刷新 | `(mapping_version,table_version,active_num_reqs)`签名，int32底表，`np.take(...,out=...)` | 不再构造旧pool-row tuple签名，但request-id tuple及索引dtype适配仍在 |
| 去掉重复HBM staging | SFA复用原生block table，metadata引用统一state/runtime的前缀 | 原生 `commit_block_table` 仍逐group clone/pin并H2D active rows，不能沿用旧“表未变0次copy” |
| 去掉query-layout D2H回环 | prepare从CPU账本/统一投影得到所需控制量，device query ends供attention消费 | 没有旧 `build_dsa_meta` 回读链；不等于整个forward/sampling没有D2H |
| 避免逐层输出owner | runtime初始化scratch，full层写、shared层消费、下一full才覆盖 | `cache_slots`仍跨step持久，不能与scratch一起复用或删除 |
| 缩短长hash传递 | 现typed projection携带布局和DRAM assignment，steady assignment通常为空 | 不把旧hash suffix游标协议当现接口，scheduler目录/hash账本仍不可删除 |

可复核 [InputBatch投影](../../../../../../vllm_ascend/dsa_offload/input_batch.py)、
[运行期准备](../../../../../../vllm_ascend/dsa_offload/runtime.py)及
[原生表提交](../../../../../../vllm_ascend/worker/block_table.py)。旧报告“steady只要两次H2D”
只是当时DSA相关metadata的近似，不是当前整个input preparation的任务数。
当前Indexer/resident/可选MTP full group分别计算slot mapping，MTP完整position不能
因省一次launch而改成resident压缩position；这个寻址反例已在本页第4、10节保留。

### 固定 owner 不等于没有动态成本

地址、shape、dtype、stride稳定只解决捕获绑定问题，不表示值可以缓存不更新。请求
阶段、position/candidate、水位和dump列仍要随本轮变化；InputBatch行号与stable pool
行号也不是同一个身份。暂时未调度的请求不能按请求完成释放。生命周期合同见
[总架构](../architecture.md)§13，不恢复旧 sidecar 或按层回调。

旧每步复制完整hash历史的工作量随 `B*ceil(context_len/S)` 增长，长上下文与大batch
会共同放大它。当前布局/assignment投影已替换那条传递链；只在边界需要的跨step账本
应按生命周期使用，不能因为steady算子不直接读取就删除scheduler的hash或资源目录。

当前七列是 `[7,C]` INT32 SoA，`copy_to_device`虽然检查active/captured行范围，
实际仍复制整个owner：`7*C*4=28C bytes`，C=256时为7168 bytes。缩batch时只把
原active尾部清成PAD；不每轮清全部未用CPU尾行。只传active列前缀会遇到二维切片
stride及多次提交的权衡，不能看到B小就断定整块copy一定更慢。

DRAM缓存命中时不提交表H2D；版本或映射变化时，gather active rows，并清除缩行后
的旧尾部，复制行数为 `max(B_previous,B_active)`，表字节为该行数乘 `4*M`。
全局table version仍可能因无关pool行变化使当前view重建，所以要分开稳态命中与
新增满块、请求增删/重排后的尖峰。指针没变不证明映射正确，缓存失效测试必须改变
值、行序、版本与active数量，再检查消费者看到的新值和PAD尾部。

`prepare_execution_view`提前在attention metadata构建前提交dump列，允许Host后续
工作与H2D重叠；它并不只有一次packed copy。Graph按captured Bp复制resident src、
Indexer src、dst三列，MTP durable开启时再加一列MTP src；Eager只提交真实紧凑jobs。
因此零真实dump也不等于Graph零元数据提交、零dump节点成本。

另一个需要分清的span是算子物理可寻址范围，而非有效key数或profiler span：MTP
QSFA消费的resident span为budget+256，选择plan宽2304，并不是每query都使用2304个
有效key。PAD、无效槽及0-copy不意味着所有元数据读取都消失。扩大span/captured rows
的成本要用实际kernel与任务数衡量，不可按有效TopK数推断相同时延，也不可声称一定
扫描完整span。现行plan合同与query连续化成本见[总架构](../architecture.md)§15、§18。

### 图覆盖率与容量账本

旧 `cc3da68bc` 确有“任一请求满块，整batch绕过ACLGraph和torch.compile”的限制。
对普通Q1、每S个token一次边界、请求相位近似独立的模型，至少一个请求到边界的概率为：

```text
P(any boundary) = 1 - (1 - 1/S)^B
```

同相位请求会集中触发，错相位会分散触发。这是批放大风险模型，不是测出的退图率，
也不能直接套到MTP每步接受数不同的情形。**当前已不是满块一律退图**：纯准入gate
允许uniform Q1或固定MTP3 Q4的DENSE/ENTER/SPARSE混排，Graph含固定dump节点及
`dst=-1`空转行。普通路径attention后dump，MTP已确认前缀按prewrite时序保护源，
不能把旧“replay后追加dump”建议替代现行水位合同。

仍要核对最终native dispatcher与DP决议以及实际replay。Prefill、non-uniform、
capture-size miss、cascade/encoder blocker或对侧DP降级可以合法true eager；
不要求恢复旧 `dsa_graph_phase` 标记。细节见[总架构](../architecture.md)§10.3、§13.4
和[DP集成](../features/data_parallel_inference.md)。真实Graph里的无dump稳态、边界
copy与Eager应分别统计，不能把“局部gate允许”直接算作Graph命中。

总平均时延可按实际互斥类别加权：`E[T]=sum(p_k*E[T|k])`；各类p99不能同样加权。
这解释了为何steady准备变快仍可能被覆盖率、周期性copy或跨DP等待掩盖。DSA新增的
LIM/KSC/DRAM读取和resident更新也是真实设备工作，与Host在哪个fence等到它是两本账。

旧图输出常驻量的例子也值得保留，但必须更新所有权解释：当时按
`num_layers*captured_rows*attention_width*sizeof(int32)`估算，61层、B=256、
width=2048+2*128=2304时单capture size约137MiB，多尺寸再累加。它是旧逐层分配
方案的容量风险，不是Host 40ms的根因，也不是今日分配公式。当前runtime同一套
scratch跨selection group及capture前缀共用；持久cache_slots、实际各输出宽度、
图runtime及模型激活另计。审计容量应按真实storage owner去重，不能按view数量相乘。

### 复测条件与尚未证明的结论

将模型/量化、硬件、CANN/torch_npu与代码版本、TP/DP/EP/CP、B、context、capture
size、MTP、输出长度、请求相位、缓存来源和有效async设置固定后再比较。Capture、
首次JIT、warmup、真实steady、ENTER、dump边界、DRAM重建及fallback分开；可从
10–20轮warmup、至少100个正式样本起步，并记录样本数与p50/p90/p99，尾延迟需要
足够边界/重建样本，不能用这个起点代替统计充分性。

同时看Host enqueue、各blocking fence的去重等待、设备图区间、lm_head/sampler、
H2D/D2H/D2D次数与字节及端到端TPOT/吞吐。诊断用event或前后同步测completion会
改变原有重叠，应与自然异步运行的端到端结果分开；MTP还要记录每outer step实际
接受/输出token数，不能将一轮target时长直接当成每token时延。

先按B与context/容量做缩放，确定是循环、固定表、launch还是等待主导，再决定优化。
无需重做已从当前路径消失的旧gate collective/旧serializer消融。当前仍需实测的
包括原生表clone/pin/copy、七列整块提交、DRAM cache失效、各group slot mapping、
boundary/no-op节点与最终graph覆盖率；没有新profile不能给出收益或“已追平基线”。

容量报告也要区分模型配置的理论 `max_position_embeddings` 与给定硬件、TP/EP及
当前功能组合下实际可启动并完成服务的长度。只测试到某值，就记“最大已验证值”，
不能当作已搜索出的硬件极限。缩减 `num_hidden_layers` 的smoke可缩小复现、加快
定位，但改变了计算/容量与通信负载，不能替代全层模型验收或线性外推性能。

CPU/mock能检查owner指针、PAD、版本失效与任务列，不能证明真实NPU DMA、数值或
延迟；当前相关测试存在也不等于本次执行通过。旧39 passed、旧“结构优化完成”与
今天的NPU性能门禁分别记录。本文仅迁移方法与代码核验，没有新增性能实测结论。

## 13. 还没解决什么

全量QA/其他模型精度、冷算/复用等价性、上述非MTP DP/MoE分歧、native干净退出、长稳/FD/淘汰/故障注入、namespace物理GC和性能收益均不因本次文档整理被标为通过。后续工作应分别建立可复现条件和验收指标。
