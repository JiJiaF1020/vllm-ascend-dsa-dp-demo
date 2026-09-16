# DSA 稀疏卸载测试入口

本目录是 v0.23 DSA 的 tester-facing 验收入口。脚本以 GLM-5.1/GLM-5.2
为首要模型，DeepSeek-V3.2 使用相同配置做强制回归。

随仓提供 `novel_dataset.py`：`chinese_5k`、`chinese_10k`、`chinese_23k`、
`chinese_40k`、`chinese_64k`、`chinese_104k`。每项是包含一条 prompt 的列表。
当前客户端已将旧 `8k/70k` 标签替换为 `10k/64k`；下方历史验收记录中的
旧标签和测量结果保留，仅表示当时使用的数据。

## 公开版 GLM-5.2 权重

A5 DSA offload 的 Indexer C8 类型按 `quant_model_description.json` 中
逐层 `*.indexer.quant_type` 优先、全局 `indexer_quant_type` 兜底识别，
支持 `INT8_DYNAMIC` / `FP8_DYNAMIC` / `W8A8_MXFP8`。
不再用 `wq_b.weight` 的量化方式替代缓存类型。缺少上述声明的内部版保持
原有行为；关闭 LI C8 开关仍有效，baseline 保留原生量化路径。
测试时只需把 `MODEL_PATH` 指向公开版权重目录，不要修改权重描述文件。

## GLM MTP 三轮合并图（卸载与 baseline 共用）

`simple_dp_prompt_test.py` 设置 `RUN_MODE="graph"`、`ENABLE_MTP=True`、
`ENABLE_MTP_GRAPH=True` 即可启用；`ENABLE_DSA_OFFLOAD` 独立控制卸载。
`ENABLE_MTP_GRAPH` 默认 `False`，保留已有 eager drafter 对照。
其它入口设置 `speculative_config.enforce_eager=False`、`num_speculative_tokens=3`，
并令 `draft_tensor_parallel_size` 等于 target TP。

支持 A5 GLM DSA 模型、target `FULL_DECODE_ONLY`、PP/PCP/DCP=1、同步调度，
不启用 SP、shared-expert-DP、FlashComm2、lm-head TP。不满足条件时 drafter 回退 eager。
普通 decode 使用独立的三轮 MTP 合并图，设备端 padding、slot/position 和 RoPE
前处理包含在图内；host staging 和 DP 模式协商仍在图外。
baseline 的 MTP MLA/Indexer 缓存保持 BF16 full cache，target C8 不变。

custom 的 MTP durable dump step（prefix/L3）继续走 eager，任何 DP rank
需要此路径时全组回退，避免捕获动态 dump job/epoch；prefill/mixed 也走 eager。
确认实际回放请查看 `[GLM MTP graph] Replaying separate merged 3-step draft graph`。
建议先短 prompt eager/graph 对照，再验证长 prompt、跨满块及 prefix/L3 回载精度。

## CopySFA 融合开关（默认关闭）

在现有 `additional_config["dsa_sparse_config"]` 内添加
`"enable_fused_copy_sfa": True` 开启融合；省略或设为 `False` 时保持
copy + SFA 分离路径。不要用此字段覆盖其它 DSA 配置。

目前仅支持 A5 packed-C8 + DSA offload + MTP 的 TopK2048 接口，
不满足条件时显式报错。只改变 target 稀疏 attention 的 copy/SFA 调用，
不改变 LIM、prefix/L3 或 MTP durable cache 生命周期。
算子内部保留大 copy-count 回退路径；开启不代表每种形状都获得加速。

首次使用需在昇腾环境重新编译安装本仓库的 CANN 算子和 Torch 扩展。
单卡 eager/graph 对照测试：

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_dsa_a5_c8_ops.py -k copysfa
```

随后用相同输入分别设置开关为 `False`/`True` 做整网精度及性能对照；
单算子通过不能替代整网（包括 prefix/L3）验收。

| 文件 | 用途 |
|---|---|
| `simple_prompt_test.py` | 基线隔离、cache-init、eager 和 graph 冒烟 |
| `simple_dp_prompt_test.py` | 同事入门用的单机DP离线脚本，顶部按特性分组配置，输出完整文本/token IDs及自动精度证据摘要 |
| `run_a5_mtp_operator_tests.py` | MTP全可见域TopK2048升级门禁，默认框架UT＋单卡算子，不加载模型或启动Mooncake |
| `qa_dataset_test.py` | LongBench 风格数据集精度回归，可切 disabled/eager/graph |
| `eval_dataset_acc_score.py` | 对 QA 脚本生成的 JSONL 计算 LongBench 指标 |
| `serve_dsa.sh` | 以 OpenAI 兼容在线服务拉起 DSA eager/graph |
| `serve_dp_mtp_l3.sh` | P6-A单机DP2/TP4/EP8、MTP3与共享Mooncake的阶段启动配置 |
| `stream_chat_client.py` | 交互式多轮流式客户端 |
| `online_l2_l3_closeout_test.py` | 5k/8k/40k/70k 在线 L2/L3 生命周期收口 |
| `online_dp_foundation_test.py` | 单机 DP2 基础阶段的混合并发与 real/idle graph smoke |
| `online_dp_prefix_test.py` | 单机 DP2 Local L1/L2 prefix，显式 DP 路由、冷/热命中断言与持久响应记录 |
| `online_dp_l3_test.py` | 单机 DP2 共享 Mooncake：跨域首次 load、local reset 后读回与同 key 并发请求 |
| `online_dp_l3_accuracy_test.py` | 串行或显式双域并发的热缓存/L3精度对照，首差异logprobs及持久JSONL离线分析 |
| `online_dp_cache_admin_test.py` | 顶部配置直接运行，默认P6-B九请求，结束恢复运行，不停止服务 |
| `offline_dp_closeout_test.py` | 顶部配置直接运行，默认DP2/TP4 Graph+MTP3+L3六请求，检查正常缓存流程 |
| `online_long_context_dataset_test.py` | 在线长上下文数据集分桶回归 |
| `L3_cache_sys_api_test/` | Mooncake/MemCache 独立 API、baseline L3 与后续 DSA 分层缓存测试 |

总架构、特性路线图、已确认专题、未决草案和工程经验统一见
[DSA 文档中心](../../docs/source/developer_guide/Design_Documents/dsa_sparse_offload/index.md)。
当前架构和稳定 ABI 见
[DSA 稀疏卸载详细设计](../../docs/source/developer_guide/Design_Documents/dsa_sparse_offload/architecture.md)。
该文档包含模块职责、v0.23基线对照和统一元数据合同；
按资源所有权复用现有模块，不按DP/PD/MTP组合复制运行时。
L3 接入、异步 write-behind 和 HBM 热层复用设计见
[DSA L1/L2/L3 分层缓存详细设计](../../docs/source/developer_guide/Design_Documents/dsa_sparse_offload/features/multilevel_kv_cache.md)。
特性适配进展与后续实施方向见
[DSA 稀疏卸载后续开发 RoadMap](../../docs/source/developer_guide/Design_Documents/dsa_sparse_offload/feature_roadmap.md)。

## A5 MTP 算子升级验证

### 专项回归：pool行置换

修复机制、关键证据与分析方法统一维护于
[问题分析与工程经验](../../docs/source/developer_guide/Design_Documents/dsa_sparse_offload/engineering/diagnostics_and_lessons.md)。
已部署配套算子的环境，可单独执行不加载模型的严格回归：

```bash
ASCEND_VISIBLE_DEVICES=0 ASCEND_RT_VISIBLE_DEVICES=0 \
python -m pytest tests/e2e/nightly/single_node/ops/singlecard_ops/test_dsa_a5_c8_ops.py \
  -k pool_row_relocation_diagnostic -vv -s --tb=short 2>&1 | tee mtp_pool_row_fixed.log
```

0改为分配到的空闲A5卡号；该测试不需要模型权重、Mooncake或DP集群。
名称保留diagnostic以兼容旧命令，但用例是严格断言。它覆盖DENSE+SPARSE、first-fill和
两个steady步骤，比较row0/row1/row0的映射、选择、copy count及QSFA；CPU物化KV刻意
绕过KSC，所以不能替代完整搬运链门禁。成功末尾打印`POOL_ROW_INVARIANCE_PASSED`。
默认simple-DP仍为三轮基础流程，不带已撤销的行trace/独立salt开关。

`simple_dp_prompt_test.py`正常完成推理后会自动生成精度证据摘要，不需要独立过滤脚本。
同一次运行目录下的`precision_compact/precision_summary.txt`用于先反馈，
`precision_details.json`保留完整生成token/text及首差异附近概率，
不含完整prompt或全量logprobs；原始`dp*.jsonl`保留不变。
摘要按实际DP数和相邻轮次配对，支持末轮交换DP与replicate；单轮只整理输出。
已有结果也可直接处理：把本脚本顶部`SUMMARY_ONLY_DIR`改为对应的`simple-dp-xxx`
目录，再执行`python simple_dp_prompt_test.py`。此模式不导入`novel_dataset`、
不加载模型或连接Mooncake；重新推理前将`SUMMARY_ONLY_DIR`改回空字符串。
摘要同时记录首次共同候选概率变化、候选集合变化和最大变化位置；候选交集
差值为0不等于完整top-k集合相同，也不等于全量logits一致。
仅整理已有结果不需要NPU、vLLM或重启服务；差异仅作为诊断证据，不自动判精度通过/失败。

MTP路径同步单算子仓 `6732a649a36e4829994aa6cbdbd7ad6616333180`：
逐query在全部因果可见KV上选Top2048，入选tail为hit，不再追加dense tail。
LIM输入 `offloaded_lens` 仍传已卸载的durable前缀长度；输出
`topk_slots/topk_index` 均为 `[T,1,2048]`，官方QSFA使用 `sparse_mode=0`。
双parity tail、缓存预算、块分配/倒手及KSC不改，resident长度仍为C+256。
仅升级MTP3配置的target路径，保留其运行时R=1～4、PAD/DENSE/first-fill/steady；
非MTP继续使用原算子、2176输出及QSFA mode=3。MTP Attention语义标识已更新，
旧L3对象隔离；不会改变非MTP的语义标识。本轮框架集成仍需以下A5验收。

这是CANN算子、生成ACLNN API、Torch binding、Python owner的联合升级，
**只同步Python、复用旧so或只编译kernel都不够**。沿用现有成功的A5/CANN环境，
在项目根目录重编译安装，例如：

```bash
COMPILE_CUSTOM_KERNELS=1 python -m pip install -e . --no-build-isolation --no-deps
python examples/dsa_demo/run_a5_mtp_operator_tests.py
```

`SOC_VERSION`沿用测试机真实A5型号；自动识别失败再填写以`ascend950`开头的值。
不要求先全量删除build/output，但必须确认本轮host/kernel/扩展均实际更新。
LIM/CopySFA符号名不变，启动检查通过新参数名拒绝旧schema，native测试拒绝旧
2176/2304的MTP输出。新Torch binding搭配旧CANN so仍不合法，必须配套重建并重启，
重新capture graph。短DENSE/PAD仍可能有无效索引，不宣称所有官方QSFA精度问题已解决。

脚本开关直接在顶部修改。默认顺序：

1. `tests/ut/dsa_offload`：owner、请求/query分轴、L3语义指纹及既有prefix/DP回归。
2. `test_dsa_a5_c8_ops.py`全量：非MTP回归、七输出形状校验、跨128边界与两种parity、
   first-fill/steady、ragged/PAD/DENSE、真实swapped DRAM逐字节复制、eager/graph及
   full/shared/下一full缓冲复用。这里的MTP是target C8算子，不是proposer BF16算子。

任一阶段失败即停止。日志和`summary.json`在脚本旁的`results/a5_mtp_ops/<时间-pid>/`，
反馈失败阶段日志或全量结果目录即可。复制到draft也可自动定位已安装的源码仓库；
有多份安装时直接填写`PROJECT_ROOT`。脚本不改NPU可见设备设置，请沿用原单卡UT环境。

同日跨特性复核还补充了真实BlockPool的ENTER回归：lookahead后的真实尾块、
待dump满块、共享pending块延后一轮DENSE以及低余量失败不改表。算子用例增加
head32/T=70的四-query tile（含真正执行LI的R=2行）、共享只读L2源的独立HBM写回。
这些会随上述入口自动执行，不需再加开关。`simple_dp_prompt_test.py`仍保留
MTP+prefix+L3组合；当前已恢复四长度三轮Graph入门测试，说明见下方单机DP入门入口。

`RUN_INVALID_PLAN_PROBES=False`默认关闭故意触发设备断言的负例。只有允许设备异常的
独占环境才打开；它单独启动两个进程检查-1 count和坏pair，禁用core文件、设置超时，
且必须看到对应错误信息，不能将任意进程异常当成PASS。正常门禁不包含这些破坏性测试。

算子门禁通过后，用`simple_dp_prompt_test.py`的Graph＋MTP＋prefix＋L3配置
复测，再按需回归在线。需要重启服务/重新capture graph。packed-MTP的L3指纹已隔离
旧attention策略，不需清空公共Mooncake；旧对象不会被本版复用。新旧策略的生成token
不要求逐个相同，精度对照应比较同一新策略下local replay与external loadback。

## 1. 前置条件

- 配套 vLLM v0.23 和当前 vLLM-Ascend 已安装；
- 当前 checkout 的自定义算子已经完整编译并重新安装；
- Ascend 910C 首版使用 PP=1、DCP=1、PCP=1；
- `block_size=128`；
- 关闭 async scheduling 和 KV-cache metrics/events；KV transfer 只允许当前
  首版 `DSAL3Connector`，其他 Connector 仍拒绝；本地
  DRAM prefix cache 可通过脚本中的 `ENABLE_PREFIX_CACHING` 单独开启；
  如启用 speculative decoding，只允许 A5 packed-C8
  下固定 MTP3 与 greedy sampling；
- 正式精度和性能测试关闭 DSA trace points。

脚本只设置必要的 vLLM/vLLM-Ascend 原生环境变量。DSA 策略参数全部位于
`additional_config["dsa_sparse_config"]`，不再通过零散 DSA 环境变量控制。
A5 packed C8 物理布局另外复用 vLLM-Ascend 原生的
`enable_sparse_sfa_c8` 与 `enable_sparse_li_c8`；示例用一个开关同时控制二者，
不暴露 DSA 不支持的半开组合。该开关在 `disabled` 模式下也会保留，以便
使用相同 packed C8 cache 编码公平对照原生路径与 DSA 路径。

首次拉起应分别看到且只看到一份：

```text
================ DSA HBM CACHE CAPACITY REPORT ================
================ DSA DRAM CAPACITY REPORT ====================
```

报告中的 Indexer blocks 应等于 resident MLA base blocks 乘
`indexer_mla_block_ratio`。DRAM 的 selected usable blocks 必须不少于
HBM Indexer blocks；若主机内存不足配置倍数但仍满足 1x 完整 mirror，日志会
明确显示 downsized 和实际倍数，任一 worker arena 创建失败则启动失败。

## 2. 快速冒烟

`simple_prompt_test.py` 延续 v0.19 的直接用法，不提供一长串 CLI 参数。
先修改文件顶部“用户配置”区：

```python
MODEL_PATH = "/mnt/kv_dpc/weight/GLM-5.1-w4a8"
RUN_MODE = "eager"  # disabled / cache-init / eager / graph
PROMPTS = [
    "第一个测试文本",
    "第二个测试文本",
]
MAX_NUM_SEQS = 2
MAX_MODEL_LEN = 8192
MAX_NUM_BATCHED_TOKENS = 8192
ENABLE_CHUNKED_PREFILL = False
ENABLE_PREFIX_CACHING = False  # DSA 下启用 LocalDRAM prefix；target HBM 尚不匹配
PREFIX_CACHE_REPLAY_ROUNDS = 1  # prefix 验收时与上项一起改为 2
LOGGING_LEVEL = "INFO"
ENABLE_A5_PACKED_C8_DSA = False  # A5 W4A4C8 设 True；A3 W4A8 保持 False
ENABLE_MTP = False  # A5 C8 上设 True 后固定启用 3 个 draft token
ENABLE_L3_WRITE_BEHIND = False  # 首版 Mooncake rank0 L2->L3 异步写
ENABLE_L3_PREFIX_LOADBACK = False  # P1-R: L3 suffix -> all-rank L2 -> HBM
ENABLE_L3_DIRECT_HBM_LOAD = False  # 当前固定 L3 -> L2 -> HBM
RESULT_JSON = None  # 需要保存对照 token IDs 时填写路径
```

然后直接运行：

```bash
python examples/dsa_demo/simple_prompt_test.py
```

四种模式的作用：

- `disabled`：不传 `dsa_sparse_config`；A5 C8 开关可独立保留，用于验证 DSA
  修改未影响相同 cache 编码下的原生路径；
- `cache-init`：只构造 `LLM`，验证 HBM 双平面、scheduler DRAM 容量合同和
  各 worker 完整 Indexer+MLA arena 初始化；
- `eager`：执行 DSA LIDU/KSC/SFA-Offload 和满块 dump；
- `graph`：执行 `VLLM_COMPILE + FULL_DECODE_ONLY`。

Prefix cache 验收必须在同一个 `LLM`/在线服务进程中先完成一次请求，再发送
第二次相同前缀请求；把相同 prompt 同时放进同一个首次 batch 不能证明命中，
因为第一份 bundle 在该 batch 的 model output 返回前尚未 hash-ready。
当前 INFO 模式会为每个新请求打印一次 `DSA prefix lookup`，包含有效命中
token 数、DRAM blocks 和 MTP blocks；待功能稳定并进入性能验收后，再评估
降为 DEBUG 或改成采样日志。vLLM 原生聚合 prefix hit rate 仍会保留。

`cache-init` 不执行 `generate()`，因此不能证明数据面算子已经运行。结果 JSON
也不捕获 engine-core 子进程日志；容量报告是否出现且只出现一次，必须在
完整控制台日志中单独核对。

默认短 prompt 主要覆盖 DENSE。构造 sparse/ENTER 或混合预算 batch 时，直接
替换 `PROMPTS` 并相应增大 `MAX_NUM_SEQS`、`MAX_MODEL_LEN` 和
`MAX_NUM_BATCHED_TOKENS`。实际阈值覆盖应使用 tokenizer token 长度，不要把
字符数当 token 数。

graph 模式下，普通 Q=1 或 MTP3 的 uniform Q=4
DENSE/ENTER/SPARSE 任意混排可进入统一 FULL graph；prefill、非 uniform
query、capture-size miss 或其他原生动态 blocker 会被 DSA 显式送入 true
eager，不会执行一张仅覆盖部分 DSA metadata 的 piecewise graph。MTP3 的
双尾块覆盖跨 128-token 边界，不通过临时降级 Q=1 规避。非法 async 配置由
`tests/ut/dsa_offload/test_config.py` 覆盖。

chunked prefill 复用 vLLM v0.23 原生调度。把
`ENABLE_CHUNKED_PREFILL=True`，并把 `MAX_NUM_BATCHED_TOKENS` 调到小于
单条长 prompt 的 token 数，即可覆盖多个 chunk。DSA 仍保持
prefill/decode phase barrier：每个 prefill chunk 走 eager，最后一个 chunk
完成并同步 dump 后才释放 resident 满块，后续单 token decode 可正常进入
FULL graph。`scheduler_reserve_full_isl` 必须保持默认的 `True`，保证首个
chunk 入场前完整 prompt 能同时容纳于 Indexer 与 resident MLA 两个 dense
plane。

## 3. QA 数据集回归

`qa_dataset_test.py` 延续旧项目对测试同事更直接的用法：修改文件顶部“用户
配置”区，不强制记忆大量 CLI 参数。至少设置：

```python
MODEL_PATH = "/mnt/kv_dpc/weight/GLM-5.1-w4a8"
DATASET_FILE = "/home/data/longbench/multifieldqa_zh.jsonl"
DATASET_START = 0
DATASET_LIMIT = 100
RESULT_DIR = "LongBenchResult/glm51_dsa"
RUN_MODE = "eager"
BATCH_SIZE = 4
MAX_MODEL_LEN = 131072
MAX_NUM_BATCHED_TOKENS = 131072
ENABLE_CHUNKED_PREFILL = False
```

然后运行：

```bash
python examples/dsa_demo/qa_dataset_test.py
```

建议对同一数据切片跑三次：

1. `RUN_MODE="disabled"`：原生基线；
2. `RUN_MODE="eager"`：DSA eager；
3. `RUN_MODE="graph"`：DSA FULL decode graph。

脚本自动在 `RESULT_DIR` 下增加 `disabled/eager/graph` 子目录，不会让三次
结果相互覆盖。

输出 JSONL 保留：

- `pred`；
- `answers`；
- `all_classes`；
- LongBench `length`；
- `sample_id`、`sample_index`；
- 实际 `prompt_tokens`。

这既可用于自动评分，也可直接定位异常请求。若一批 prompt token 总数超过
`max_num_batched_tokens`，脚本会提示 scheduler 可能分多轮完成 prefill；
打开 `ENABLE_CHUNKED_PREFILL` 后，单个长 prompt 也可以按 token budget
切成多个 prefill chunk。

脚本默认 `MIN_TOKENS=0`，不会强制模型绕过正常 EOS。评分器也仅按
LongBench 官方规则对 `trec/triviaqa/samsum/lsht` 截取首行；QA 和
retrieval 任务保留完整输出参与评分，以免掩盖后续乱码、重复 prompt 或异常
提前结束。

## 4. LongBench 评分

评分脚本需要环境中存在 `jieba`、`fuzzywuzzy` 和 `rouge`。对一个目录：

```bash
python examples/dsa_demo/eval_dataset_acc_score.py \
  --result-path \
  examples/dsa_demo/LongBenchResult/glm51_dsa/eager
```

对单个文件：

```bash
python examples/dsa_demo/eval_dataset_acc_score.py \
  --result-path /path/to/multifieldqa_zh.jsonl
```

增加 `--longbench-e` 可输出 `0-4k`、`4-8k` 和 `8k+` 三档分数。

## 5. 最小回归矩阵

| 维度 | 至少覆盖 |
|---|---|
| 模式 | disabled、eager、graph |
| batch | bsz=1、bsz>1 |
| 长度 | 超短、DENSE、三档 resident budget |
| row mode | 全 DENSE、全 SPARSE、DENSE/ENTER/SPARSE mixed |
| graph | active=captured、active<captured 的 PAD |
| dump | prefill 多满块、decode 跨满块 |
| chunked prefill | 中间 chunk、最终 chunk、完成后首个 decode |
| 生命周期 | 请求结束、InputBatch condense/reorder、stable row 复用 |
| 模型 | GLM-5.1/GLM-5.2 主验收、DeepSeek-V3.2 回归 |

不要只看“进程跑完”。至少核对：

- 无乱码和异常提前 EOS；
- 同一请求在 bsz=1 与 bsz>1 下没有明显语义断裂；
- eager 与 graph 没有单行串扰；
- finish reason 与输出 token 数合理；
- 容量报告的 blocks、tokens、bytes 自洽；
- 没有 silent fallback 到未启用 DSA 的路径。

## 6. 问题反馈工件

出现问题时保留：

- vLLM 和 vLLM-Ascend commit；
- 完整脚本配置或结果 JSON；
- 模型 architecture、量化方式、TP/EP/DP；
- prompt token 长度和出错 sample index；
- 从 `non-default args` 到异常结束的完整日志；
- DSA HBM 容量报告；
- eager/graph 对照 token IDs 与 finish reason；
- CANN、torch、torch-npu、设备型号；
- 性能问题对应的 profiler 目录和测试场景。

`trace_points` 当前是拉起期可解析的预留合同，尚无稳定日志 consumer。不要
把“没有 trace 输出”解释成 DSA 未运行；确认路径应结合容量报告、自定义
算子 profiling 和输出结果。

## 7. 在线流式服务

### DP入门离线脚本

只改`simple_dp_prompt_test.py`顶部配置，然后在实际脚本目录直接运行：

```bash
python simple_dp_prompt_test.py
```

配置按长条注释分成输入/采样、基线通用、DP/TP、DSA、prefix/replay、MTP、L3、
运行环境/结果八组，另有顶部的已有结果摘要入口。不导入其他demo的配置，不要求命令行环境变量。
当前叠加A5八卡DP2/TP4、Graph、chunked prefill、DSA、prefix、MTP3、L3读写，
每域max_num_seqs=4，生成128 tokens并ignore_eos=True，保存top-5 logprobs。
默认`PROMPTS=chinese_8k + chinese_5k + chinese_70k + chinese_40k`，
`PROMPT_DISTRIBUTION="round_robin"`、`REPLAY_ROUNDS=3`。
数据导入和`PROMPTS`由使用者直接维护，不自动补请求或增加序号前缀。
四份数据各一条时，前两轮DP0跑8k/70k，DP1跑5k/40k；最后一轮交换输入归属。
运行前核对模型/设备/Mooncake配置并手动启动Master。已部署修复后算子的环境无需重编译。

默认三轮共12个请求，不把不同DP输出当作数学标杆，也不自动放宽精度判据：

| 轮次 | 日志phase | 作用 |
|---|---|---|
| R1 | `seed` | 首次生成并写L3；70k可能复用同域8k刚生成的公共前缀，不能一律当成全量重算 |
| R2 | `local-replay-1` | 保留本地缓存的首次复用 |
| R3 | `after-local-reset` | 两域排空并只清本地、保留L3，再交换各域输入；与该prompt的R2比较 |

运行级salt在所有DP和各轮保持相同，隔离旧实验缓存但不破坏本次跨域内容共享。
原四轮独立salt/行号日志专用于LIM定位，修复已通过单卡及整网Eager/Graph后删除。

同步最新脚本到实际运行目录后执行：

```bash
python -u simple_dp_prompt_test.py 2>&1 | tee simple_dp.log
grep -E '\[simple-dp|Replaying aclgraph|DSA L3|DSA MTP durable|DSA prefix|ERROR|Traceback' simple_dp.log > simple_dp_key.log
```

反馈`simple_dp_key.log`及本次`results/simple_dp/<run_id>/precision_compact/`
下的`precision_summary.txt`、`precision_details.json`即可，不用传原始大JSONL。
R3应有非零L3 load/restore完成，具体块数取决于本次prompt共享前缀和调度顺序，
不能只靠`cached_tokens`或phase名称证明实际发生IO。精度差异需结合执行条件分析，
当前样例不声明首次生成/前缀复用、不同DP或不同模式之间数学等价。

常见用法，均直接在脚本内修改：

| 场景 | 设置 |
|---|---|
| 基线DP对照 | 关闭`ENABLE_DSA_OFFLOAD`和两个L3开关；MTP按对照需要选择，RUN_MODE可独立选eager/graph |
| 调整输入 | 直接修改脚本顶部的数据导入及`PROMPTS` |
| 每个DP跑相同输入 | `PROMPT_DISTRIBUTION="replicate"`、`SWAP_DP_ON_LAST_REPLAY=False`；仍共享本次run salt |
| 仅Local prefix重放 | 关闭两个L3开关及reset/swap，保留prefix，`REPLAY_ROUNDS=2` |
| 非MTP对照 | `ENABLE_MTP=False`；MTP3默认已开启，proposer默认eager，可设`ENABLE_MTP_GRAPH=True` |
| 混合三轮跨域读取 | 当前默认；配好`PROMPTS`，保留round_robin、三轮、L3及reset/swap开关 |
| 不加载模型的配置检查 | `CHECK_ONLY=True`，先检查配置、路径、依赖以及启用时的Master监听 |

L3当前固定Mooncake TCP，Master需自己启动；MTP proposer默认eager，入图配置见文首。
Master预检遵循实际worker的地址优先级：脚本内`NATIVE_RUNTIME_ENV`的覆盖、父进程
`MOONCAKE_MASTER`环境变量、最后才是配置文件。显式空值会在加载模型前报错，
不表示取消覆盖；要使用文件地址，应移除对应覆盖项，而不是赋空字符串。
`CHECK_ONLY=False`才能实际推理，未支持的async scheduling、direct-HBM仍关闭，
“默认叠加特性”不包含这些尚未支持的分支。只观察短输出/性能时可降低MAX_TOKENS或设LOGPROBS=0。
`MAX_NUM_SEQS`与token预算均是每域值，输入多于并发上限可以正常排队。
可见设备数必须等于DP*TP，不按每域手工裁切环境变量；每个客户端由脚本自动设置
`VLLM_DP_*`，LLM kwargs不重复传data_parallel_size。
round_robin要求每域至少一条输入，不伪造placeholder。DP=1可作为无L3对照，须同步
调整TP或可见卡；DP1+L3沿用`simple_prompt_test.py`，避免调用仅DP具备的管理接口。
首个已验证的DSA拓扑为DP2/TP4/EP8，其他拓扑不据此承诺已通过模型/算子约束。

每次运行结果存于`results/simple_dp/<run_id>/`：配置、LLM参数、实际engine信息及
逐域JSONL（含原prompt、输出文本、token IDs和可选logprobs）。每轮只打印简要进度，
主进程最后按DP/轮次统一打印完整输出，避免文本穿插。脚本用全域共享、轮间不变的
run salt隔离旧实验缓存，不修改prompt文本。每条记录保存phase和实际salt，摘要保留命中长度，
便于识别请求或对照条件是否发生变化。摘要比较的是相邻轮次，不自动给出精度PASS。

本脚本供配置探索和学习，不自动放宽/判定数值精度，不把cache hit或COMPLETED
当成IO字节正确性和worker优雅退出的证明。需要严格功能验收仍使用
`offline_dp_closeout_test.py`及相应在线测试；已知TorchNPU native退出限制仍适用。

### DP 脚本配置索引

DP 服务统一使用 `serve_dsa.sh`；文中的 `dp_serve_dsa.sh` 指使用者自己的副本，
不是仓库中的另一套启动入口。下面四个客户端需要同步同一版本的文件：

| 客户端 | 默认请求安排 | 默认生成长度 | 服务要求 |
|---|---|---|---|
| `online_dp_foundation_test.py` | 混合5k/8k/40k/70k，再单40k，共5请求，服务自动路由 | 32 | DSA DP2，prefix/MTP/L3关闭 |
| `online_dp_prefix_test.py` | 显式DP0/1的冷/热、混合、非对称、重入，共16请求 | 32 | Local prefix开启，L3关闭；可选MTP |
| `online_dp_l3_test.py` | 默认40k共7请求；`--concurrent-pair`每组8请求、交换DP回载 | 32 | P5无MTP；P6-A加`--mtp`并建议生成128 tokens |
| `online_dp_l3_accuracy_test.py` | 默认DP0串行40k共7请求；`--concurrent-pair`每组14请求 | 64 | 同P5配置；比较热路径，不以冷/热不同直接判失败 |

共用客户端设置实际在 `online_dp_foundation_test.py` 导入时读取。prefix 导入
这些设置；L3 也显式导入 `MAX_TOKENS`，传给 prefix 的 `_run_case(max_tokens=...)`。
精度客户端复用连接设置，但单独读取同一个 `DSA_DP_MAX_TOKENS`，以64为默认值。
因此不用在每个文件中重复写 `os.getenv`；设置必须发生在启动客户端 Python 之前。
所有入口均固定 `ignore_eos=True`、temperature=0，且禁止HTTP自动重试；精度入口
另固定请求seed=1024和top5 logprobs，功能入口不进行严格数值对照。

上述环境变量配置说明仅适用于表中四个旧客户端。当前收口入口
`online_dp_cache_admin_test.py`和`offline_dp_closeout_test.py`改为顶部“用户配置”
作为默认来源，直接运行即可，不需要DSA_DP_*或MODEL_PATH环境变量。旧CLI可选覆盖。

| 环境变量 | 默认值 | 作用范围 |
|---|---|---|
| `DSA_DP_API_BASE` | `http://127.0.0.1:8000/v1` | 全部客户端；服务使用8035时必须显式覆盖 |
| `DSA_DP_API_KEY` | `EMPTY` | 全部客户端；需与服务端 `API_KEY` 一致，不打印密钥 |
| `DSA_DP_MODEL_NAME` | 未设置 | 全部客户端；服务只有一个模型时自动从 `/models` 选择 |
| `DSA_DP_MAX_TOKENS` | 功能入口32，精度入口64 | 全部客户端，至少2；不会改变服务端batch budget |
| `DSA_DP_REQUEST_TIMEOUT_SECONDS` | 900 | 全部客户端，必须为有限正数 |
| `DSA_DP_PREFIX_RESULT_DIR` | 脚本旁 `results/dp_prefix` | prefix客户端 |
| `DSA_DP_L3_LABELS` | `40k` | L3功能客户端；逗号分隔5k/8k/40k/70k |
| `DSA_DP_L3_STORE_WAIT_SECONDS` | 10 | L3功能和精度客户端；有限非负秒数，等待本身不证明put成功 |
| `DSA_DP_L3_RESULT_DIR` | 脚本旁 `results/dp_l3` | L3功能客户端 |
| `DSA_DP_ACCURACY_LABELS` / `DSA_DP_ACCURACY_RANKS` | `40k` / `0` | 精度串行模式；`0,1`仍为顺序测试，并非并发 |
| `DSA_DP_ACCURACY_RESULT_DIR` | 脚本旁 `results/dp_accuracy` | 精度客户端 |

精度客户端的`--concurrent-pair 10k,64k`左右分别指定DP0/DP1，此时不使用精度串行的
LABELS/RANKS选择场景；可重复传参。功能L3客户端也支持该参数，但流程不同：
左右指定初始写入域，跨域读取时交换，两次并发回载共8请求，并忽略`DSA_DP_L3_LABELS`。
精度客户端的`--analyze <JSONL>`只分析已有结果，
不连接服务。功能入口没有 `PROMPT_LABELS`、`REPEATS`、`IGNORE_EOS` 开关，
不要把旧确定性测试脚本的配置复制过来。foundation/prefix会读取四种数据，
L3/精度入口也通过共用loader读取 `novel_dataset.py` 中的四个条目后再选场景。

服务端配置与客户端环境彼此独立：`PORT`不自动变成`DSA_DP_API_BASE`，
`SERVED_MODEL_NAME`不自动变成`DSA_DP_MODEL_NAME`；`MAX_NUM_BATCHED_TOKENS`
控制每个scheduler一轮的token预算，`DSA_DP_MAX_TOKENS`控制每个请求的输出长度。
只改客户端生成长度、标签或结果目录，不需要重启服务；改服务端DP/TP、模型、
prefix/MTP/L3开关则需要重启。客户端启动日志会打印实际生效的 `max_tokens`。

`serve_dsa.sh` 支持其 `${NAME:-默认值}` 形式配置项的环境变量覆盖，但以下
内容在脚本中固定：`PYTHONHASHSEED=114514`、block_size=128、关闭async scheduling、
EP开启、`mooncake-dsa-tcp.json`路径、每worker的4 GiB segment。JSON中的Master
地址和1 GiB local buffer需要改配置文件；不能仅在命令前写同名变量就认为已覆盖。
P6-A已开放单机DP+固定MTP3+TCP L3；P6-B在线internal-DP的协调external reset已
通过正常管理及暂停排队中reset验证。启用DSAL3时，裸Connector/离线单域external
reset仍拒绝。未配置connector时沿用基线行为，不适用该拒绝断言。direct-HBM仍拒绝。

先修改 `serve_dsa.sh` 顶部用户配置。首次在线验证建议从
`RUN_MODE="eager"` 开始，确认请求生命周期、连续请求和流式返回正常后，再
切换为 `RUN_MODE="graph"`：

### 单机 DP2 基础阶段

DP>1的L2预算用`/proc/sys/kernel/random/boot_id`识别宿主内核内存域，不再用
hostname判断。同一宿主的不同hostname可以通过；同名但不同boot ID会被拒绝。
该文件必须在每个EngineCore中可读且反映同一宿主内核；若容器屏蔽或虚拟化它，
先修正部署视图，不回退到hostname。读取失败会随原有CPU汇总传播后由各域拒绝，
不新增forward同步。DP1不需要这项检测，单机SPMD也不要求local-DP等于global-DP。

当前支持 DSA + 单机 mp/internal-DP/EP，可选 Local L1/L2 prefix 和固定 MTP3（proposer eager、与 target 相同 TP）。P5 无MTP共享L3已验证单40k Graph闭环，P6-A的DP MTP与L3组合也已通过单40k及8k/70k长短并发功能闭环，完整精度与生命周期仍待验收。下面的 foundation 测试保持 prefix/MTP/L3 关闭，prefix/L3 使用后面的专用客户端。脚本默认仍是 DP1；验证 A5 八卡 `DP2*TP4*EP8` 时设置：

```bash
MODEL_PATH="/models/GLM-5.2-w4a4c8-mxfp4" \
SERVED_MODEL_NAME="glm-5.2-dsa-dp2" \
RUN_MODE="eager" \
ENABLE_A5_PACKED_C8_DSA="true" \
DATA_PARALLEL_SIZE=2 \
DATA_PARALLEL_SIZE_LOCAL=2 \
TENSOR_PARALLEL_SIZE=4 \
bash examples/dsa_demo/serve_dsa.sh 2>&1 | tee dsa-dp2-server.log
```

脚本用户配置需保持：

```bash
ENABLE_A5_PACKED_C8_DSA="true"
ENABLE_PREFIX_CACHING="false"
ENABLE_MTP="false"
ENABLE_L3_WRITE_BEHIND="false"
ENABLE_L3_PREFIX_LOADBACK="false"
```

容量报告应在两个 EngineCore 中给出相同的 node snapshot、per-DP byte quota 和完整 `L1 Indexer blocks by DP`，但各 DP 的 selected L2 blocks 允许随本域 L1 不同。graph 的 D/idle 阶段还应出现：

```text
DSA runtime idle dummy: graph_mode=FULL ... io=device-pad/noop
```

该日志表示 idle DP 也重放 FULL 图；是否无副作用仍需结合无异常、真实缓存 canary/算子 UT 与请求结果验收，不能只凭一行日志判定。

提交服务器后先执行：

```bash
python -m pytest tests/ut/dsa_offload -vv --tb=short
python -m pytest tests/ut/worker/a2/test_worker_v1.py -vv --tb=short
python -m pytest \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_dsa_a5_c8_ops.py \
  -k 'graph_replay_pad_row_is_noop' -vv -s --tb=short
python -m pytest \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_kv_cache_full_block_dump.py \
  -k 'noop_rows_preserve_destination' -vv -s --tb=short
```

本阶段没有修改自定义算子源码；后两条使用现有 kernel 的 PAD/`dst=-1` 合同，不需要仅为本次 Python 改动全量重编 csrc。

服务就绪后运行一次客户端即可同时制造 mixed real work 和 real/idle：

```bash
python examples/dsa_demo/online_dp_foundation_test.py \
  2>&1 | tee dsa-dp2-client.log
```

服务端提取本阶段证据：

```bash
grep -E \
  'DSA DRAM CAPACITY|DP rank / size|Host observations by DP|L1 Indexer blocks by DP|This-DP byte quota|Compatibility signature|DSA runtime idle dummy|Replaying aclgraph|ERROR|Traceback' \
  dsa-dp2-server.log
```

### 单机 DP2 Local L1/L2 Prefix（P3）

基础 eager/Graph 冒烟之后，开启 Local prefix。以下配置保留两域独立
Scheduler、HBM/L2 block manager 和请求级水位；不共享 L2，不取跨 DP
prefix 长度的 Min。HBM pair 与 L2 bundle 复用、首个真实 prefill 的
HBM-only 回填继续走既有数据面，eager/Graph 使用相同 metadata owner。

在项目根目录执行，模型路径按实际位置修改：

```bash
MODEL_PATH=/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4 \
SERVED_MODEL_NAME=glm-5.2-dsa PORT=8035 \
RUN_MODE=graph ENABLE_A5_PACKED_C8_DSA=true \
DATA_PARALLEL_SIZE=2 DATA_PARALLEL_SIZE_LOCAL=2 TENSOR_PARALLEL_SIZE=4 \
MAX_NUM_SEQS=4 MAX_MODEL_LEN=131072 MAX_NUM_BATCHED_TOKENS=16384 \
ENABLE_CHUNKED_PREFILL=true GPU_MEMORY_UTILIZATION=0.9 \
DSA_INDEXER_MLA_BLOCK_RATIO=2 DSA_HOT_CPU_BLOCK_MULTIPLE=1.0 \
ENABLE_PREFIX_CACHING=true ENABLE_MTP=false \
ENABLE_L3_WRITE_BEHIND=false ENABLE_L3_PREFIX_LOADBACK=false \
bash examples/dsa_demo/serve_dsa.sh 2>&1 | tee dsa-dp2-prefix-server.log
```

同一配置改 `RUN_MODE=eager` 可做 eager 对照。脚本已加入基线参数
`--enable-prompt-tokens-details`，用于返回每请求的 `cached_tokens`；若沿用
自建的 `dp_serve_dsa.sh`，也需补这个参数，并去掉旧的 DP prefix 禁用判断。
其余 DP/MTP/L3 约束保持上述配置，普通 `--enable-prefix-caching` 就是本轮开关。

先运行 UT（不需要为本轮重编自定义算子）：

```bash
python -m pytest tests/ut/dsa_offload -vv --tb=short
```

服务就绪后，在项目根目录执行：

```bash
DSA_DP_API_BASE=http://127.0.0.1:8035/v1 DSA_DP_MAX_TOKENS=32 \
python examples/dsa_demo/online_dp_prefix_test.py 2>&1 | tee dsa-dp2-prefix-client.log
```

脚本固定面向两个 DP，共 16 请求：

1. 40k 在 DP0 冷算后同域热重放；相同 salt 的 40k 再路由到 DP1，仍应先冷后热。
2. 5k/8k/40k/70k 按显式路由并发冷算，再按相同路由并发热重放。
3. DP0 冷 70k 与 DP1 热 40k 并发，制造不同 prefill 工作量和 real/idle 转换。
4. DP0、DP1 分别再次执行热 40k，覆盖 idle 后重新入场。

显式路由复用 vLLM 的 `X-data-parallel-rank` 请求头，没有修改服务端 LB。
每次脚本运行产生独立 `cache_salt`，同一轮对比的 prompt/salt 保持一致，
不会向文本添加编号，也不依赖全局清缓存才能重跑。16 请求是顺序阶段中的
总数，最大并发仍为 4；不要求四条 mixed 请求落在同一 forward。

客户端冷请求必须 `cached_tokens=0`，热请求必须 `cached_tokens>0`，
命中长度按 128 对齐且短于 prompt；全部请求须完成指定 token 数。
基线冷命中会省略 `prompt_tokens_details`，按 0 处理；若热请求也缺该字段，
客户端会失败并提醒检查服务参数。`cached_tokens` 是 Local 命中总数，
区分 HBM pair 与 L2 restore 仍需查看服务端 `hbm_pair_blocks/l2_blocks`。

响应、token ID、request ID、路由和 usage 逐条写入
`examples/dsa_demo/results/dp_prefix/<run-id>.jsonl`，可用
`DSA_DP_PREFIX_RESULT_DIR` 指向其他持久目录。断言前先保存响应，便于追溯。
脚本不把不同 batch/冷算/复用的 token 不一致直接判为缓存功能失败，也不声明
这种差异必然来自硬件；精度检查是独立验收项。

服务端证据可过滤：

```bash
grep -E 'DSA prefix lookup|DSA prefix restore submitted|DSA runtime idle dummy|Replaying aclgraph|ERROR|Traceback' \
  dsa-dp2-prefix-server.log
```

`DSA prefix lookup` 现在末尾有 `dp_rank`，可与 client request ID 对齐。
本轮负载不保证自然产生 HBM-only/L2-only 的每种容量组合；相同物理编号的
跨 DP 隔离、L1/L2 长度组合、ENTER 与 dummy 后回填由 `test_dp_prefix.py`
和 `test_runtime.py` 覆盖。Graph 还需已有 PAD/dump no-op 的 A5 算子 UT 证据。

管理面说明：DSA internal-DP 的 local reset 会等待并检查所有 EngineCore
结果；任一拒绝/失败将抛出异常，使 HTTP 接口不再返回误导性的 200。
上述为P3时的结果聚合边界；P6-B实现后，在线internal-DP reset统一走下述
暂停、全域预检和提交协议。offline入口不自动获得该协调能力。
若某域繁忙导致部分清理失败，可在排空后重试；本轮客户端不调用 reset。
`reset_running_requests=true` 遇到运行中请求会在队列修改前拒绝，DSA
尚不支持抢占恢复。共享 L3 namespace 的统一 epoch/quiesce 协议仍在后续阶段。

### 单机 DP2 MTP + Local Prefix（P4）

P4 延用 P3 的独立 L1/L2 目录与统一元数据，启用固定 MTP3、target 双尾与
MTP BF16 满块持久化。proposer 必须 eager，draft TP 与 target TP 一致；
target 仍可使用 FULL decode 图。DP 下 L3、抢占恢复与 async scheduling
继续关闭。新增的 idle proposer 没有真实 attention metadata，但仍沿基线
merged-draft 执行三次模型调用，与真实域保持所需 collective 次序。

本阶段框架 UT：

```bash
python -m pytest tests/ut/dsa_offload -vv --tb=short
python -m pytest tests/ut/spec_decode/test_llm_base_proposer.py -vv --tb=short
```

本轮修改均在 Python 框架/测试层，不需要重新编译自定义算子。
在已验证 P3 的服务配置上设置 `ENABLE_MTP=true` 并重启；例如：

```bash
MODEL_PATH=/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4 \
SERVED_MODEL_NAME=glm-5.2-dsa PORT=8035 \
RUN_MODE=graph ENABLE_A5_PACKED_C8_DSA=true \
DATA_PARALLEL_SIZE=2 DATA_PARALLEL_SIZE_LOCAL=2 TENSOR_PARALLEL_SIZE=4 \
MAX_NUM_SEQS=4 MAX_MODEL_LEN=131072 MAX_NUM_BATCHED_TOKENS=16384 \
ENABLE_CHUNKED_PREFILL=true GPU_MEMORY_UTILIZATION=0.9 \
DSA_INDEXER_MLA_BLOCK_RATIO=2 DSA_HOT_CPU_BLOCK_MULTIPLE=1.0 \
ENABLE_PREFIX_CACHING=true ENABLE_MTP=true \
ENABLE_L3_WRITE_BEHIND=false ENABLE_L3_PREFIX_LOADBACK=false \
bash examples/dsa_demo/serve_dsa.sh 2>&1 | tee dsa-dp2-mtp-prefix-server.log
```

启动脚本为 MTP 明确传入
`{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}`，将
最小 sparse 阈值/预算设为 8,192，并捕获 target `[4,8,16]` token 图。
`enforce_eager` 在此只约束 proposer，不改变 `RUN_MODE=graph` 的 target。
若使用自建 `dp_serve_dsa.sh`，同步这些配置及移除旧的 DP MTP 禁用判断；
不要移除 L3 gate。

沿用显式路由客户端，生成 128 tokens 覆盖 40k/70k 的满块边界和后续 dump：

```bash
DSA_DP_API_BASE=http://127.0.0.1:8035/v1 DSA_DP_MAX_TOKENS=128 \
python examples/dsa_demo/online_dp_prefix_test.py 2>&1 | tee dsa-dp2-mtp-prefix-client.log
```

仍是 16 请求、最大并发 4，`ignore_eos=True` 已在脚本内设置；无需改脚本。
MTP 的热前缀会按 Eagle 边界回退一个满块。缓存足够时，对当前数据集通常
可见 5k/8k/40k/70k 分别命中 4,736/7,552/41,472/73,344 tokens；实际验收
仍以非零、块对齐及服务端 component 命中信息为准。

```bash
grep -E 'DSA prefix lookup|DSA prefix restore submitted|DSA MTP durable dump|DSA MTP idle proposer|DSA runtime idle dummy|Replaying aclgraph|ERROR|Traceback' \
  dsa-dp2-mtp-prefix-server.log
```

期望两域均能完成请求并出现 MTP durable `postwrite`（prefill）和
`prewrite`（decode 满块）完成日志；idle 时可见 target FULL/PAD 和
`DSA MTP idle proposer: ... draft_steps=3 attention=none`。后者的
`padded_tokens` 可因对端 prefill 很大，但 `dummy_reqs` 不超过本域上限。
日志按签名去重，不代表每轮都打印；prefix 的非零 L2 restore 和 HBM-only
回填仍需定向负载覆盖，不能用全 HBM 命中替代。

### 单机 DP2 共享 Mooncake L3（P5）

2026-09-09 启动保护修复和客户端配置整理后的回归再次通过，run ID 为
`dsa-dp-l3-73180e6e945f4752b2c235ff67d4e833`：7请求各完成32 tokens，三次外部
读取均先本地miss，再命中325 blocks/41,600 tokens，目标DP的4个TP均完成
2,243,904,000 bytes读取及325块HBM恢复。两个DP的内容fingerprint一致，八个
worker和两个scheduler的TCP身份互不重名。冷写的128+128+69块全部有
`readable=True`与成功completion；同salt并发时本次DP1写入，DP0三批全key命中
而跳过put。`store completed bytes=0`在这些记录中是去重，不是丢写。

该附件含client摘要和过滤后的server日志，没有token/logprobs或精度断言失败，
只能收口本轮P5功能回归，不能宣称模型精度通过，也不能据此归因为基线MoE。
此前双40k并发的数值差异仍是独立待查项，不因本轮功能通过而消失。正常退出、
压力/淘汰/故障注入及HBM-only回填仍不在本轮证据范围；不要求为这次正常反馈
原样再跑同一组用例。后续P6-A实现与待验场景见专节，不能用本轮结果替代MTP验收。

2026-09-08 已收到默认单40k、7请求的Graph闭环通过反馈：三次外部命中均有
所在DP全部4TP的非零L3读取和325块HBM恢复，覆盖两个DP；同key并发请求中
观察到一域先上传、另一域exists去重。此结果不包括精度、HBM-only回填、
在途竞争失败重试、压力淘汰或退出生命周期验证，也不代表DP+MTP+L3已开放。

首版为 `no-MTP + TCP + kv_both + Local prefix`，数据流保持
`L3 -> 每 TP rank 的 L2 -> HBM`。各 DP 的 TP0 分别负责上传，所有 worker
仍按配置贡献 segment；内容 key 不含 DP/rank，两个域可以读取同一份对象。

共享L3要求在启动Python前，为所有参与进程设置同一个`PYTHONHASHSEED`十进制
整数（0..4294967295）。`serve_dsa.sh`已有固定值114514，无需新增开关；自建
启动脚本或离线入口需要显式设置，例如`export PYTHONHASHSEED=114514`。
模型采样的`--seed`不能替代它：vLLM在该环境变量缺失时使用进程独立的随机
首块哈希根，即使sha256也会产生不同prefix key。当前框架会拒绝缺失、空值或
random模式，并比较seed的原字符串（例如1和01不能混用），不擅自改写哈希算法。
这项限制只用于DSA DP共享L3，不改变纯Local prefix或DP1已有行为。

沿用已验证的 Mooncake Master；若尚未启动，在另一终端运行：

```bash
bash examples/dsa_demo/L3_cache_sys_api_test/baseline_e2e/start_mooncake_master.sh
```

先验证框架与 backend UT，本轮无需重编算子或修改 Mooncake 源码：

```bash
python -m pytest tests/ut/dsa_offload -vv --tb=short
python -m pytest tests/ut/distributed/ascend_store -vv --tb=short
```

只回归这两项启动保护时，可先执行不加载模型的UT：

```bash
python -m pytest tests/ut/dsa_offload/test_dp_context.py \
  tests/ut/dsa_offload/test_config.py tests/ut/dsa_offload/test_kv_cache.py -q --tb=short
```

已有P5配置无需改推理参数或重编算子。同步Python代码后，下次有设备时按下方
原配置重启一次vLLM并运行`online_dp_l3_test.py`，检查跨DP首读和local reset即可；
Mooncake Master可保留，不必因这次启动保护修复重跑整套精度矩阵。

随后使用更新后的启动脚本重启推理服务：

```bash
MODEL_PATH=/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4 \
SERVED_MODEL_NAME=glm-5.2-dsa PORT=8035 \
RUN_MODE=graph ENABLE_A5_PACKED_C8_DSA=true \
DATA_PARALLEL_SIZE=2 DATA_PARALLEL_SIZE_LOCAL=2 TENSOR_PARALLEL_SIZE=4 \
MAX_NUM_SEQS=4 MAX_MODEL_LEN=131072 MAX_NUM_BATCHED_TOKENS=16384 \
ENABLE_CHUNKED_PREFILL=true GPU_MEMORY_UTILIZATION=0.9 \
DSA_INDEXER_MLA_BLOCK_RATIO=2 DSA_HOT_CPU_BLOCK_MULTIPLE=1.0 \
ENABLE_PREFIX_CACHING=true ENABLE_MTP=false \
ENABLE_L3_WRITE_BEHIND=true ENABLE_L3_PREFIX_LOADBACK=true \
bash examples/dsa_demo/serve_dsa.sh 2>&1 | tee dsa-dp2-l3-server.log
```

脚本使用 `mooncake-dsa-tcp.json`，Master 默认 `127.0.0.1:50088`，每 worker
贡献4 GiB，共32 GiB；此外每 worker 的 local buffer 为1 GiB。L2 预算会从
各自 DP quota 中扣除本域 L3 segment/local buffer 预留。两个 Scheduler
与所有 workers 必须使用相同 backend 配置；初始化签名现在也比较这些字段。

默认 TCP segment 身份为 workers `host:45000..45007`、两个 Scheduler
`host:55000/55001`。这些是 Store 身份，不应与实际 TE RPC 监听端口混淆。
在同一 Master/host 上并行部署另一服务时，要为它另设
`MOONCAKE_TCP_SEGMENT_PORT_BASE`，避免两套服务重名。

先用默认单40k，运行7个请求：

```bash
DSA_DP_API_BASE=http://127.0.0.1:8035/v1 DSA_DP_MAX_TOKENS=32 \
python examples/dsa_demo/online_dp_l3_test.py 2>&1 | tee dsa-dp2-l3-client.log
```

流程依次为 DP0 冷写、DP1 首次跨域读取、DP1 本地重放、清两域本地后由
DP0 读回、两域同 salt 并发请求、再清本地后由 DP1 读回。默认给每批异步
上传10秒窗口，可用 `DSA_DP_L3_STORE_WAIT_SECONDS` 调整；等待本身不是 IO
成功证明，热命中断言与服务端日志才是验收依据。不自动重试请求，避免首次
失败后的本地缓存掩盖 L3 问题。响应逐条保存到 `results/dp_l3/<run-id>.jsonl`。

测试服务需无其他流量，因为 local reset 会影响整个服务。脚本只调用
`reset_external=false`，保留 L3 对象；P6-B另开放在线全域协调的`reset_external=true`，
原始Connector/离线单域调用仍于本地修改前拒绝。每次脚本运行
使用新 salt，会增加新的测试对象，由后端自己淘汰；不是相同 key 被重复扩容。

服务端应能明确看到：

```bash
grep -E 'DSA prefix tiers|Store-owned TCP|DSA L3 (prefix lookup|load|store)|DSA prefix lookup|DSA prefix restore|DSA runtime idle|ERROR|Traceback' \
  dsa-dp2-l3-server.log
```

跨域首次读取与 reset 后读回应为 local L1/L2 miss、L3 hit，命中域的四个
TP worker 均有非零 `load completed ... destination=l2`，随后非零
`prefix restore submitted`。这样才能补上 DP 实机 L2->HBM restore 的证据。
普通 local-replay 仍可 `blocks=0`。同 key 请求的实际 put 是否重叠取决于调度，
该阶段允许被对侧先完成的上传命中；在途竞争由框架 UT 的受控事件测试补充。

并发 put 的零返回值不再直接等同于 ready：adapter 会再次检查全部组件可读。
尚未确认时回报 `success=False, retryable=True`，Scheduler 不保留 ready 提示，
活跃请求可重新扫描，后续同前缀请求也能重试。该状态打印 warning；实际 get
失败仍按 fail policy 终止，不静默重算。日志里的 put `bytes` 是成功调用提交的
payload 大小，重复 key 场景不能用它精确计算实际网络流量。

正常停止服务时，应等待各 EngineCore 的模型与 IO 工作排空；新增退出协调
使用已有 DP Store 公布本地 drain，再用已有 Gloo DP group 做一次退出 barrier，
本域 drain 后也屏蔽迟到的 START_DP_WAVE，之后才释放任何 embedded segment。
该协调不进入每轮 forward；强制 kill 或
超过基线 shutdown timeout 不提供 graceful drain 保证。不要在同一服务里
单独提前结束某个 DP 实例。

### P5 受控精度对照与容量输出

2026-09-08默认DP0/40k/64tokens的7请求实机反馈已通过：7组比较的全部64个
输出token以及返回top-k候选集合/概率均一致，最大可见logprob差值为0；
两次external均确认四TP实际非零L3->L2->HBM。本轮也观察到容量报告按整段
输出。此结论不覆盖冷算与热复用等价，也不覆盖两个DP同时处理真实请求的精度。

先在P5已有服务上做小范围精度门禁，再叠加P6的DP+MTP+L3，不等所有适配
完成才比较精度。当前测试是输出一致性与定位入口，不替代模型质量评测。

沿用上一节的 `DP2/TP4/EP8 + Graph + no-MTP + Local prefix + Mooncake TCP kv_both`。
新客户端可直接连接当前服务，无需为了精度测试重启或重编算子：

```bash
DSA_DP_API_BASE=http://127.0.0.1:8035/v1 \
DSA_DP_ACCURACY_RANKS=0 DSA_DP_ACCURACY_LABELS=40k DSA_DP_MAX_TOKENS=64 \
python examples/dsa_demo/online_dp_l3_accuracy_test.py 2>&1 | tee dsa-dp2-l3-accuracy-client.log
```

默认7请求全部串行固定路由到DP0，DP1只陪跑；不得混入其他用户流量。
输入文本不加前缀，同组使用同一salt、固定seed1024、temperature0，生成64tokens，
ignore_eos=true并请求top5 logprobs。64tokens也会越过这条40k输入的尾块边界。
要在同服务上验证另一个DP，将 `DSA_DP_ACCURACY_RANKS=1`；`0,1` 表示两个域
各做一组，共14请求，仍然串行，不是跨域同时请求。

| 阶段 | 请求数 | 用途 |
|---|---:|---|
| seed | 1 | 冷算与写缓存，只留档，不作为热路径精度参考 |
| local-0 / local-1 | 2 | 同一DP本地热复用的自身稳定性 |
| external-0 / post-local-0 | 2 | 清两域本地后L3读回，再在本域热重放 |
| external-1 / post-local-1 | 2 | 再次清本地后重复上一对，检查读回自身稳定性 |

每次reset前给write-behind默认10秒窗口（`DSA_DP_L3_STORE_WAIT_SECONDS`）；
等待不等于IO成功，仍需服务端证据。命中长度不同会标为`incomparable`，
而不是把不同计算条件下的结果误判为新精度问题。token/logprobs缺失、非有限值
会标为无效证据或直接失败，不能用空token列表获得假PASS。

同一请求条件下分别比较local-repeat、external-repeat、本地与external、以及
external后的本地再重放。任一token不同，脚本以非零退出并保存完整比较结果；
若local-repeat自己就不稳定，不能直接归因于L3。若自身稳定但跨路径不同，
则保留为待查缓存/计算路径差异，不能直接归因为硬件，也不靠放宽断言收口。

logprobs仅比较共同生成历史及首个分歧位置，不把分叉后不同历史上的概率继续
相减。`max_shared_topk_logprob_delta`只覆盖共同候选，候选集合变化另列；
`first_logprob_drift`用1e-6定位首个可见漂移，不是量化模型精度验收阈值。
`PASSED token replay consistency`只表示这组输出token一致，logprobs漂移仍需查看，
不宣称KV逐字节一致或全模型精度通过。

完整响应含token/logprobs、request ID、采样配置、prompt摘要、命中及耗时，逐条
保存到 `results/dp_accuracy/<run-id>.jsonl`；汇总为同名 `.summary.json`。
可用 `DSA_DP_ACCURACY_RESULT_DIR` 指向其他持久目录。比较在HTTP客户端关闭后
进行，避免断言触发时留下异步连接清理噪声。重复分析无需再跑推理：

```bash
python examples/dsa_demo/online_dp_l3_accuracy_test.py \
  --analyze examples/dsa_demo/results/dp_accuracy/<run-id>.jsonl
```

服务端仍需按本次run ID对应确认：local阶段是本地命中，external阶段是
local miss + L3 hit，并有本域4TP非零get和restore。客户端usage本身不能区分
HBM、L2和L3；本地参考可能直接复用HBM，不应称为纯L2 restore参考。
优先反馈本次client日志与summary，只有出现差异时再提供相应request的server片段。

#### 双域同时真实请求的精度对照

2026-09-08已验证DP0/DP1分别串行40k，以及同服务的8k/70k并发对照。
双40k在原生MoE自动选择下仍有已知精度差异，调查已收敛到首decode与对侧
prefill同轮时的执行路径差别，尚未确认数值根因。历史实验结果不改变当前
精度断言；以下脚本保留为回归与问题反馈入口，不表示该组合精度已经验收通过。
详细实验矩阵见下方调查记录链接，不要求每次代码清理都重复整套测试。

受控串行门禁通过后，使用新增的 `--concurrent-pair`，不要把
`DSA_DP_ACCURACY_RANKS=0,1` 的串行两组当作并发。该参数左侧固定发给DP0，
右侧固定发给DP1；此模式不使用串行模式的RANKS/LABELS环境变量选择请求。
同一P5服务可直接继续，无需重启、修改server或重编算子；仍然关闭MTP。

首组为DP0和DP1各一个40k，生成64 tokens：

```bash
DSA_DP_API_BASE=http://127.0.0.1:8035/v1 DSA_DP_MAX_TOKENS=64 \
python examples/dsa_demo/online_dp_l3_accuracy_test.py \
  --concurrent-pair 40k,40k 2>&1 | tee dsa-dp2-concurrent-40k-client.log
```

一组共7轮、14请求：seed、local-0、local-1、external-0、post-local-0、
external-1、post-local-1。每轮两个任务先在客户端会合再一起开始RPC，只有
两个请求都返回后才进入下一轮。两次external之前全服务local-only reset，
保留L3；等待write-behind窗口仍为10秒。不得混入其他真实流量。

请求保持原prompt、temperature0、seed1024、ignore_eos=true、top5 logprobs。
每个DP在同一scenario内复用自己的salt；两个DP的salt不同，有意先隔离
同key并发写竞争，而不是修改生产框架的key schema。同key竞争功能已有P5
独立测试，跨DP producer内容等价仍属后续单项。

每个DP独立做7组精度比较，不把DP0输出当成DP1的标杆；冷seed不与热输出
比较。并发组合、路由、prompt摘要、采样条件和cached长度必须一致。
本地重复自身不稳定、external重复不稳定、跨缓存路径不同分别保留原始证据，
不因为“并发可能不确定”而自动跳过差异或放宽断言。

JSONL增加execution_mode、scenario、batch_members、batch_signature、batch_id
以及本次运行内的单调时钟起止时间。离线分析还检查每轮恰有DP0/DP1各一条，
并输出 `client_overlap_ms`、`start_skew_ms`；缺一个域、没有在途重叠或请求
失败都会使门禁不通过。一边异常时仍等待另一边完成并保存响应，之后才报错，
不取消同伴或提前reset掩盖现场。旧串行JSONL继续支持 `--analyze`。

**重叠证据的边界**：client overlap只证明RPC在途窗口重叠，可能含排队/IO；
不能证明逐个forward的batch/EP token数完全相同。因此报告保留
`server_batch_alignment_verified=false`，这不是失败，而是明确未自动验证。
必须结合两个EngineCore同轮阶段的request ID、lookup、load、restore及完成
日志判断覆盖范围；即便请求集相同，服务端动态调度也可能不同。若token有
差异，先核对调度和路径，不直接判为L3数据损坏或硬件误差。

第一组通过后，可在同一服务验证长短不均衡：

```bash
DSA_DP_API_BASE=http://127.0.0.1:8035/v1 DSA_DP_MAX_TOKENS=64 \
python examples/dsa_demo/online_dp_l3_accuracy_test.py \
  --concurrent-pair 8k,70k 2>&1 | tee dsa-dp2-concurrent-asymmetric-client.log
```

也可重复传参，如 `--concurrent-pair 5k,70k --concurrent-pair 8k,40k`。
两组scenario顺序执行，各自14请求，salt及比较结果互相隔离；仍为每域同时
一个真实请求，不宣称覆盖每域多请求混合batch。完整响应与summary仍在
`results/dp_accuracy`，反馈本次client及对应server关键日志即可，不必重跑
已通过的单域串行测试。本节记录的是无MTP的P5精度调查；P6-A需另按专节验证。

#### 已知精度问题与清理边界

双40k并发在原生MoE自动选择下出现过热路径差异；全eager仍复现，
历史固定ALLGATHER实验在有效P/D覆盖下通过。尚未在不启用DSA的干净基线
中复现相同条件，也未获得同输入模块对照，不能写成已确认的纯基线bug。

本次清理撤掉运行时dp_execution trace、强制ALLGATHER和BF16 dispatch消融，
保留P0–P5实现及上述精度客户端，不放宽断言。普通服务恢复基线自动MoE选择，
已知差异仍可能出现。当时清理没有开放P6；后续P6-A的代码接入不等于已解决该数值问题。

详细调查记录仅在维护者本地保留，不作为仓库文档或测试依赖。

运行时trace、两个MoE消融、专用启动入口，以及只服务于这些历史日志的解析器
和专属测试均已退出仓库。不要再使用旧诊断配置。原有`first_sample`配置不属于
此次删除范围。既有JSONL、summary和原始日志应保留；不必为清理重跑精度矩阵。

当前DP测试入口分工如下，不依赖被删的诊断模块：

| 脚本 | 保留用途 |
|---|---|
| `online_dp_foundation_test.py` | P0-P2无prefix/MTP/L3的路由、mixed和idle陪跑冒烟 |
| `online_dp_prefix_test.py` | P3/P4各域Local prefix、非对称负载及idle重入 |
| `online_dp_l3_test.py` | P5跨域L3首读、local reset及同key并发写入 |
| `online_dp_l3_accuracy_test.py` | 串行/并发热路径比较及已有JSONL离线分析；不放宽已知差异的判据 |

上述客户端共用foundation的数据/连接辅助函数及prefix的usage检查，应同步同一
版本的相关脚本。它们仍需服务和`novel_dataset.py`；缺少部署依赖不表示脚本失效。
对应`tests/examples/dsa_demo`保留客户端与Mooncake探针的纯脚本自测，不导入它们
到框架UT，也不复制被测实现。DP配置、预算、dummy、控制面及缓存生命周期UT保留。

#### 容量报告互斥

容量日志也已调整：同机、同用户、共享临时目录的DP EngineCore使用同一把
`filelock`文件锁，完整HBM/DRAM报告输出后释放，DP1仍独立打印。v0.23的进程
前缀装饰会把多行日志拆成多次write，这正是此前报告穿插的原因。新锁仅包住
报告，既不增加DP collective，也不改变容量计算或forward。它不规定DP0必须
先打印，也不锁住其他模块的日志；自定义异步日志聚合器不在这层保证范围内。
锁文件无法打开或5秒内未取得时告警并降级输出，避免为日志互斥卡住引擎启动。
使用的是vLLM已有filelock依赖，不需新装NPU包。该打印改动下次重启生效，
无需单独占用一次服务拉起来验证。

本轮最小UT：

```bash
python -m pytest tests/ut/dsa_offload/test_kv_cache.py -vv --tb=short
python -m pytest tests/examples/dsa_demo -q
```

### 框架 UT 与示例自测

`tests/ut/dsa_offload` 不再导入 `examples/dsa_demo`。在线客户端的响应断言
自测和 Mooncake 探针的布局/角色自测已移到 `tests/examples/dsa_demo`，
它们验证示例脚本本身，按需独立执行，不需要模型服务或真实 L3 后端：

```bash
python -m pytest tests/examples/dsa_demo -q
```

框架 UT 与示例自测分开维护，不把示例配置逻辑迁入运行时模块来满足测试依赖。

这组是开发测试工具的自测，不是每轮服务器DSA回归的必跑门禁，也不能用其
通过数量宣称框架或模型精度通过。修改在线测试脚本/探针时由开发侧运行即可。
它有意调用仓库中真实examples脚本，而不是在测试里复制被测函数；请求、
响应和后端等测试替身可在测试内构造，生产框架UT仍不得依赖examples。
运行需要完整checkout及Python的pytest/openai/httpx，不需要novel_dataset、
模型权重、NPU或Mooncake native包。不要只复制tests目录后期望它能测不存在的脚本。
Mooncake探针现在按测试文件推导出的绝对路径加载，不再依赖当前工作目录中的
`examples`包；从项目外也可以用测试目录的绝对路径执行pytest。

### P6-A：单机 DP2 MTP 共享 L3

2026-09-09代码接入，单40k/128输出tokens/7请求及8k/70k长短并发/8请求的
A5功能闭环均已收到通过反馈；后者也补齐了两个DP实际Graph replay证据。
保留P4的固定MTP3、proposer eager、
target/draft同TP，以及P5的每域TP0异步上传、目标域全部TP同步读回、
L3->L2->HBM和fail-fast。没有新增全局phase scheduler或每轮DP collective，
不改LIDU/LIM/KSC算子、真实forward/MoE算法或量化路径。

本次run ID为`dsa-dp-l3-cc1773a4df22418b9dc5d28ac0c98e3c`：两域`mtp=True`且
fingerprint一致，热命中41472 tokens；三次外部回载均读取324 blocks、972 keys，
每个目标TP读取2,295,392,256 bytes，随后四TP全部恢复78 target layers和1 MTP层。
每块7,084,544 bytes包含180,224 bytes MTP，123 descriptors包含3个MTP plane。
两域四TP均有postwrite/prewrite完成记录；16个已提交store任务全部有成功回报，
其中11个非零、5个全key命中去重，上传只出现在各域TP0。

local-replay中的`local_blocks=324, hit_blocks=1, effective_blocks=0`符合MTP
最后一个完整块保留重算的规则，并未再丢掉一个本地块。并发prompt写入观察到
DP0先完成、DP1去重；这不替代真正重叠在途put冲突和失败重试的实测。
附件是过滤日志，不含token/logprobs、容量报告和target capture/replay标记，
不据此宣布数值等价、Graph逐轮覆盖或完整P6完成；该轮未覆盖P6-B。本轮没有发现
需要修改运行时的新故障。随后长短并发补测的结果见下节；仍不据此宣称完整P6已验收。

本阶段补强了以下组合边界：

- MTP模型路径、revision、配置hash及投机宽度纳入DP启动签名和MTP L3 schema，
  不含具体DP/rank。无MTP的L3 schema不变；旧MTP namespace内容不再误复用，
  对象由Mooncake自行淘汰，不需要手工清池或重启Master。
- proposer dummy的独立ForwardContext继续携带`skip_kv_transfer`，connector
  对显式dummy误入生命周期直接报错，不允许它放行真实任务。
- 真实target forward只暂存MTP store计划；本域proposer完成durable copy、
  runtime校验通过后，仍经既有`wait_for_save()`记录event并放行TP0后台上传。
- UT覆盖三组件的跨域同key竞争、MTP组件尚不可读的假成功、只补写缺失组件、
  短读/同TP组失败、不完整bundle miss、两域不同L2长度时的一次Eagle回退。
  这些是CPU控制面/字节替身测试，不代替真实NPU吞吐、精度或通信验证。

先同步代码并跑UT，本轮没有修改自定义算子，不需要重新编译csrc：

```bash
python -m pytest tests/ut/dsa_offload -vv --tb=short
python -m pytest tests/ut/spec_decode/test_llm_base_proposer.py -vv --tb=short
```

沿用P5的Mooncake Master，正常停止旧vLLM服务后启动；所有命令从项目根目录执行：

```bash
MODEL_PATH=/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4 \
bash examples/dsa_demo/serve_dp_mtp_l3.sh 2>&1 | tee dsa-p6a-server.log
```

该入口只设置阶段默认值并调用`serve_dsa.sh`，不复制vLLM参数拼装：DP2/TP4/EP8、
八卡可见、target Graph、MTP3 proposer eager、prefix/L3写读开启、direct-HBM关闭。
maxseq=4、maxlen=131072、chunked prefill=16384、HBM比例2、L2倍数1.0、
gpu_memory_utilization=0.9，API为8035，模型名glm-5.2-dsa。模型路径可覆盖，
`RUN_MODE=eager`可切eager对照；TP/DP和阶段开关在此入口固定。

服务就绪且没有其他流量后：

```bash
DSA_DP_API_BASE=http://127.0.0.1:8035/v1 \
DSA_DP_L3_LABELS=40k DSA_DP_MAX_TOKENS=128 \
python examples/dsa_demo/online_dp_l3_test.py --mtp 2>&1 | tee dsa-p6a-client.log
```

仍是7个请求，固定ignore_eos=True；32/64/128等生成长度由已有环境变量控制，
`--mtp`不暗改长度。该选项检查完整热前缀一次末块回退：当前40k条目输入41672
tokens时，应为41472 cached tokens（324块），而非无MTP的41600（325块）。
通过marker、容量报告和服务端日志确认实际MTP已开启，不只凭客户端cached长度。

```bash
grep -E 'DSA prefix tiers|DSA MTP (durable|L3|idle)|DSA L3 (prefix|store|load)|DSA prefix restore|ERROR|Traceback' \
  dsa-p6a-server.log > dsa-p6a-server-key.log
```

本轮至少检查：两域初始化`mtp=True`且fingerprint相同；各域TP0的store先staged、
再released after proposer completion；MTP `postwrite`及跨满块`prewrite`均完成。
外部load应有三component keys（324块对应972 keys），命中域四个TP非零读取后
恢复target与MTP，`mtp_layers=1`（当前GLM模型）。local重放可零restore。
同key竞争允许先完成者去重；不可读时必须重试/报错，不能假成功。

先反馈client和server-key即可，不要求同时扩成40k/40k精度矩阵。功能通过后，
再追加混合长度/不同接受率和同缓存路径数值对照；已有AUTO MoE差异继续单列。
P6-B的在线管理协议与待验场景见下节；P6-A功能测试仍要求无其他流量。
正常退出沿用P5 drain保护，强杀不保证数据保留。

#### P6-A 长短不均衡补测（沿用当前服务）

2026-09-09已通过：run ID为`dsa-dp-l3-2fd870ac772441f983420a32570578de`，
服务重启后参数按原P6-A保持，8请求均完成128 tokens。8k/70k热命中为
7552/73344 tokens；交换域和local reset后的4个外部请求各有4TP非零读取，
分别为59块/177 keys/417,988,096 bytes与573块/1719 keys/4,059,443,712 bytes。
16条TP load完成后均恢复78 target layers+1 MTP layer；本地重放8条restore为0。
14个store提交均成功完成（11非零、3全key去重），没有本轮可见的漏回报或错误。
两域四TP都有postwrite/prewrite记录，decode新增满块分别为物理块61和575；
11:23:05两个DP的TP0均打印`Replaying aclgraph`，两域均有FULL/PAD idle记录。

该结果补齐长短角色交换、实际Graph执行及三组件回载的功能证据，不提供每轮
执行签名、设备canary或token/logprobs对照。短请求与长请求响应时间接近不能
作为IO失败的证据：cross-load时8k于11:23:20完成get，而70k于11:23:23完成，
两域后续staged记录同在11:23:23，与共享EP耦合相容，但未采集性能分解。
P6-A已验证主路径可阶段性收口，下一步进入P6-B；数值、故障/压力/长稳和
完整管理生命周期仍独立待验。本次评估只更新文档，不修改运行时代码或要求原样复测。

单40k/128 tokens闭环后，先补一组8k/70k，再进入P6-B。仅更新客户端及其
prefix/foundation辅助脚本，不修改服务器配置或框架，不需重启服务/Master。
保持原P6-A的MTP3、Graph和L3配置，无其他流量；本轮不是严格精度测试。

```bash
DSA_DP_API_BASE=http://127.0.0.1:8035/v1 DSA_DP_MAX_TOKENS=128 \
python examples/dsa_demo/online_dp_l3_test.py --mtp --concurrent-pair 8k,70k \
  2>&1 | tee dsa-p6a-pair-client.log
```

| 阶段 | DP0 | DP1 | 预期来源 |
|---|---|---|---|
| pair0-seed | 8k | 70k | 各自新salt冷算并写L3 |
| pair0-cross-load | 70k | 8k | 交换来源，两个域都必须首次从共享L3取对应前缀 |
| pair0-local-replay | 70k | 8k | 当前域本地热复用 |
| pair0-reset-load | 8k | 70k | 清两个域本地后，两个域都从L3取回 |

共8请求，每轮并发发起2请求，等两者都结束后再推进下一阶段。seed后、reset前
均保留10秒write-behind窗口；一侧失败会等另一侧返回再报错，不重试、不继续reset。
两份来源的salt各自固定并跨域复用；即使指定40k/40k也不会被两个域都已持有同salt
的HBM命中掩盖外部读取。`--concurrent-pair`可重复使用，各组顺序执行、salt隔离。

MTP阈值为8192：当前8k条目实际7752 tokens，生成128后仍在dense范围；70k为
73496 tokens，覆盖长prefill和sparse decode。交换角色覆盖两域的长/短路径。
热命中应分别为7552/73344 tokens（59/573块），每次外部读取预期177/1719 keys，
命中域4TP均有对应非零load与`mtp_layers=1`恢复。合计4个外部请求的16条TP完成
记录；本地重放可restore=0。响应、request ID和实际cache_salt保存在本轮JSONL。

客户端并发RPC不保证server逐forward对齐。结合server日志确认真实load、restore、
MTP dump以及模式转换，不把client耗时相近当作逐轮同步或数值等价证明。
`DSA_DP_L3_LABELS=8k,70k`仍是逐标签执行旧7请求流程，不能代替此长短并发场景。

保留现有服务日志，不要清空正在写入的文件。若日志文件为`dp_for_mtp.log`，可在
运行client前记录`START_LINE=$(wc -l < dp_for_mtp.log)`，client结束后提取增量：

```bash
tail -n +"$((START_LINE + 1))" dp_for_mtp.log | \
  grep -E 'DSA MTP (durable|L3|idle)|DSA L3 (prefix|store|load)|DSA prefix restore|DSA runtime idle|Replaying aclgraph|ERROR|Traceback' \
  > dsa-p6a-pair-server-key.log
```

已有去重日志可能不重复打印，尤其MTP prewrite/postwrite和首个Graph replay。
可另附旧日志中的模式/初始化标记，不因本轮未重打印而判断能力失效，也不需要
为重打这些日志重启服务。反馈client和增量server-key即可。

### P6-B：在线全域缓存管理（正常主路径已验证）

首版代码、CPU控制面验证及A5正常八请求管理流程已完成，后续九请求补测也覆盖了
预先pause且有请求排队时的reset/resume；具体证据见下文验收记录。故障/长稳与退出仍单列。
只支持现有单机mp/internal-DP部署，不新增线程/通信组，不改变MoE与缓存数据格式。
使用已有DP TCPStore的compare_set，在多个API前端之间持有同一管理owner锁。

`POST /reset_prefix_cache?reset_external=false|true`统一执行：

```text
获取全部署管理锁 -> pause(wait, clear_cache=False)
 -> 所有域运行请求和connector任务排空
 -> 全域PREPARE只读检查 -> COMMIT清本地hash/未分配候选
 -> external=true时使用同一UUID epoch切换L3 namespace
 -> 确认所有commit结果/fingerprint一致 -> 恢复原先运行的部署 -> 释放锁
```

基线PAUSED_NEW会保留waiting请求，新请求也可入队，但不admit；已有running请求
继续执行。DSA waiting-prefill gate现在尊重此状态，避免排队prefill反过来阻塞
decode排空。EngineCore.has_work包含DSA pending IO，pause Future全部完成后再
预检各物理池只剩null-block引用，且没有待消费的restore/load计划。queued请求
本身不删除，恢复后重新查询cache；已经paused的部署reset后仍保持paused。

边界与失败规则：

- HTTP成功表示所有域都确认，而不是仅取第一个结果。并发管理请求在修改前拒绝，
  不同时启动两轮pause/reset。状态查询也汇总所有域。
- 暂停开始后的超时、部分失败或不确定结果保持owner锁并标记fail-closed，禁止
  继续管理/resume；需要正常停止并重启全部DP。已排队请求可能等待到重启；不承诺
  故障时保住这些请求。新请求在已收到fatal通知的EngineCore中直接返回失败。
- 管理owner不设自动过期。前端进程失联/网络故障时不由另一个前端自动接手。
  通知也可能失败，因此不承诺所有进程在故障瞬间同步停下或恢复。
- HTTP调用方取消等待不会取消已开始的后台管理任务；该任务由原API客户端对象
  保持引用。API进程整体退出则不保证完成，不允许把HTTP取消视为回滚。
- pause等待超时复用服务shutdown_timeout（未设置则300秒，至少30秒）；其他
  utility阶段前端等待上限30秒。超时不等于服务器动作已停止，不可盲目重试。
- external reset只是本部署逻辑失效，旧Mooncake对象仍占空间直到后端淘汰；
  不影响其他部署的namespace，也不是物理清池。重启后的namespace持久化不在本版。
- 手工`/pause`仅允许`mode=wait&clear_cache=false`，`/resume`恢复，`/is_paused`
  汇总状态。keep/abort、pause自动清缓存、sleep/wake、reset_running_requests=true
  暂拒绝；不要拿未适配的RLHF/sleep路径绕过管理协议。
- `reset_mm_cache`和`reset_encoder_cache`继续走原生utility，不参与DSA管理事务。
  v0.23的API启动会自动调用前者清理dummy多模态缓存，即使当前部署是纯文本模型；
  不能把它当成L1/L2/L3 prefix reset拦截，否则API服务在就绪前就会退出。
- DP1原有行为不变；启用DSAL3的offline SPMD单域external reset仍拒绝，不把在线协议误用于它。

`dp_admin.py`是EngineCore参与者，`dp_control.py`是API协调端，注册仍在
`patch_dsa_data_parallel.py`。正常退出保留P5的全域drain/最后barrier再释放segment。
管理/paused期间退出不自动resume；未admit请求可能被终止，强杀不保证排空。

#### P6-B验收记录（2026-09-09）

运行`dsa-dp-admin-08506ef24e1c4d88a1bad02c8163b1a3`已通过正常管理主闭环，
8个40k请求均完成128输出tokens，无client断言或所提供server关键日志中的失败。

- 首次local reset并非空闲演示：07:25:56 DP1报告`running=1 l3_pending=True`；
  07:26:01最后store完成，07:26:03两域才prepare/commit并resume。
- 3次reset均有两域prepare/commit/resume及API完成记录。首次local reset保留
  指纹`8ed58c365f5de72bb466fa9965da41f2`；external reset两域统一切换至
  `8be1c295fa8ea9ac60ec63bdd5d195d0`；第二次local reset仍保持新指纹。
  管理请求分别由ApiServer_0/1处理，均成功，但不是并发管理竞争的实机证据。
- 两个不同salt在external reset后分别于DP0/DP1得到`cached_tokens=0`；
  新epoch数据随后可双向跨域命中41472 tokens，没有只有一域切换的迹象。
- `local-reset-load`、`new-epoch-cross-a/b`及`queued-while-paused`共4次回读，
  各有目标域TP0-3完整get/完成记录：每rank 324块、972 keys、2295392256 bytes，
  随后`target_layers=78 mtp_layers=1` restore。不是仅依靠HTTP命中数或时延判断IO。
- 以`(DP rank, job ID)`匹配，23个planned/submitted/completed任务全部闭合；
  19次非零put并确认`readable=True`，4次`hit all keys`去重对应`bytes=0`成功完成。
- 手工pause后，client检查`is_paused=True`且排队任务未完成；resume后该请求完成。
  server中此请求的lookup/get位于resume日志之后。精确排队到达时刻未单独采样。

结论仅覆盖上述正常管理、在途排空及数据恢复。数值精度、故障/超时/HTTP取消、
并发管理竞争、预先暂停时reset、暂停期间退出和长稳不由此宣称实机通过。
过滤日志未保留Graph模式信息，不据此新增逐轮Graph覆盖结论。无需原样重复此组。

同步Python代码后需重启一次vLLM（不能只换client），无需重编C++或重启Master。
继续使用P6-A的DP2/TP4、MTP3、Graph和L3参数：

```bash
python -m pytest tests/ut/dsa_offload -vv --tb=short
MODEL_PATH=/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4 \
bash examples/dsa_demo/serve_dp_mtp_l3.sh 2>&1 | tee dsa-p6b-server.log
```

以下保留原八请求验证方式：先在`online_dp_cache_admin_test.py`顶部设置
`PAUSED_RESET=False`、`LEAVE_PAUSED=False`，API_BASE/模型名/API_KEY也在该文件中修改。
服务就绪后，在无其他业务流量的测试服务上：

```bash
python examples/dsa_demo/online_dp_cache_admin_test.py 2>&1 | tee dsa-p6b-client.log
```

共8请求、两个独立salt：分别在DP0/1冷写，local reset后跨域读旧内容；external
reset后两域分别必须冷算，再清本地并交叉读取新namespace。最后手工pause、
发送一个排队请求、确认未完成，再resume让它返回。首轮local reset尝试与第二个
seed请求重叠，0.5秒只用于触发机会，不是排空依据；看pause日志running/l3_pending
判断是否实际命中在途状态。控制面UT另外用受控Future覆盖pending IO。

```bash
grep -E 'DSA DP cache admin|DSA L3 namespace reset|DSA L3 (prefix|load|store)|DSA prefix restore|ERROR|Traceback' \
  dsa-p6b-server.log > dsa-p6b-server-key.log
```

关注每个operation两域均prepared/committed，external=true的fingerprint一致且
变化，local-only时不变，resume在全部commit之后；client期望`PASSED requests=8`。
这是管理/IO功能验证，不宣称数值精度通过。遇到fail-closed停止测试并反馈日志，
不要继续手动resume。正常完成后可在结束测试时正常停止服务，附退出摘要检查
各域inflight_jobs=0及worker退出；无需为关服验证另外重拉一轮模型。

### 单机DP收尾回归

限定A5单机DP2/TP4/EP8、GLM-5.2 W4A4C8、Mooncake TCP及固定MTP3/proposer eager，
P0-P6正常功能主路径已有在线/离线验证。完整精度、故障/长稳及干净退出不据此宣称通过。
实现与已知限制见总设计16.6节；本节只保留当前可执行的回归入口，不重扫历史调试轮次。

`cc328ec9`收口确认：用户已反馈UT及最新离线三轮正常完成。限定单机DP正常功能
阶段性闭环，以下命令保留供后续改动回归，不要求为本次文档更新再次运行。

**后续代码改动的回归入口：UT与示例自测。**

在vllm-ascend项目根目录运行：

```bash
set -o pipefail
python -m pytest tests/ut/dsa_offload tests/examples/dsa_demo -vv --tb=short 2>&1 | tee dp_closeout_ut.log
```

需要同步项目代码和`examples/dsa_demo`内相关脚本。框架UT不依赖示例；
`tests/examples/dsa_demo`测试的是示例本身，故需要对应脚本，二者职责不同。
本次只清理诊断/示例/文档，没有修改生产框架或算子，不需要编译算子、重启在线服务
或再次申请八卡跑同一矩阵。反馈最后的pytest汇总；有失败时带完整失败栈。

**离线推理：后续设备空闲时的可选回归，不是本次清理的强制门禁。**

`offline_dp_closeout_test.py`顶部已经预置：

| 设置 | 默认值 |
|---|---|
| 模型 | `/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4`，按机器实际路径修改 |
| 拓扑 | 单机八卡，DP2/TP4/EP8 |
| 执行 | target Graph，固定MTP3/proposer eager，同步调度 |
| 缓存 | L1/L2 prefix + Mooncake TCP L3，direct-HBM关闭 |
| 输入/输出 | 8k/40k，128输出tokens，固定ignore_eos=True |
| 容量 | 每域max_num_seqs=2，max_model_len=100000，chunk=8192，ratio=2，L2 multiple=1.0 |
| L3容量 | 每worker贡献4GB，合计32GB |

Master仍需手动启动；在线服务和离线任务不能占用同一组卡或复用冲突的segment端口。
推理脚本可按习惯放在draft等目录，只需它、可导入的`novel_dataset.py`和顶部指定的
Mooncake配置文件。它不导入其他推理demo的配置。在实际脚本目录运行：

```bash
set -o pipefail
python -u offline_dp_closeout_test.py 2>&1 | tee dp_offline_closeout.log
```

脚本先检查模型/依赖/Master监听，加载后核对真实DP/TP、MTP、connector与namespace。
随后两域分别cold、local、清本地后交换域读取，共6请求；冷算cached=0，本地/跨域
命中8k=7552、40k=41472（当前数据集和MTP3配置）。跨域读取需目标域所有TP非零get，
随后restore 78 target layers和1 MTP层。完整四长度矩阵可把`PROMPT_LABELS`改回
`("5k", "8k", "40k", "70k")`，该12请求矩阵已有实机记录，不要求本轮再跑。
`CHECK_ONLY=True`只做启动前检查并保存参数，不加载模型。

结果逐条写入`results/dp_offline/<run_id>/`，包含配置、实际engine信息、dp0/dp1 JSONL；
错误在清理前打印并保存`dpN-failure.txt`。结尾`INFERENCE/CACHE PASSED`仅表示
正常功能断言通过，不代表输出精度或worker干净退出。需要精简服务日志时：

```bash
grep -E 'dsa-dp-offline|DSA DP L3|DSA L3 (load|store|worker summary|scheduler summary|resource close)|DSA prefix restore|Replaying aclgraph|shutdown|ERROR|Traceback' dp_offline_closeout.log > dp_offline_key.log
```

正式脚本已移除临时退出探针、诊断RPC、专用pickle设置和`shutdown-debug.zip`生成，
因此后续不会再打印`shutdown trace armed`；这不是上传遗漏。正常status/reset RPC仍使用
原生默认序列化。不要为了常规测试手动设置`VLLM_ALLOW_INSECURE_SERIALIZATION=1`。

**在线推理：沿用已有配置与客户端，无需为本次清理重拉服务。**

后续专用八卡空闲时，先核对`serve_dp_mtp_l3.sh`的模型路径及设备，再从项目根目录启动：

```bash
bash examples/dsa_demo/serve_dp_mtp_l3.sh 2>&1 | tee dp_server.log
```

另一终端执行（已运行的同配置服务可直接复用）：

```bash
python examples/dsa_demo/online_dp_cache_admin_test.py 2>&1 | tee dp_admin_client.log
```

客户端默认`PAUSED_RESET=True`、`LEAVE_PAUSED=False`、128输出tokens，覆盖9请求；
测试后恢复服务，不自动关闭服务器。不应在有其他业务的服务上执行缓存管理测试。

**已知退出限制，单独记账。**

离线`98e3407215834cb7af79b374a9e39f3f`已经确认所有DSA请求/IO排空、八worker
L3关闭成功，但随后均停留在TorchNPU post6的`_C._npu_shutdown(success)`，
超过原生executor的4秒+4秒等待窗口后被SIGKILL。内部具体步骤和归属未完全确定，
不能标为干净退出通过，也不将它误报成L3读写失败或直接认定与DSA使用方式无关。
无需为补相同日志再次拉起大模型；已有诊断实现保留在`999c9727`历史中。

### 单机 DP1 在线配置

对 A5/950 W4A4C8，至少修改：

```bash
MODEL_PATH="/path/to/GLM-5.1-or-5.2-w4a4c8"
SERVED_MODEL_NAME="glm-5.2-dsa"
ENABLE_A5_PACKED_C8_DSA="true"
ENABLE_PREFIX_CACHING="true"  # 需要验证本地 DRAM prefix 时开启
SAFETENSORS_LOAD_STRATEGY="prefetch"
```

该开关会在 `additional_config` 同时写入
`enable_sparse_sfa_c8=true` 与 `enable_sparse_li_c8=true`。不要只开其中
一个；当前 A5 DSA 会在初始化期拒绝半开布局。W4A4C8 仍使用
`--quantization ascend`。需要验证 MTP3 时再设置：

```bash
ENABLE_MTP="true"
```

启动脚本会同时改用 `sparse_activation_tokens=8192`、resident 档位
`8192/10240/12288`，并传入
`{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}`。MTP3 只接受
`temperature=0.0`；target 可复用 uniform FULL graph，GLM proposer 按社区
基线保持 eager。MTP proposer 使用独立 BF16 full-cache group，不进入 target
DSA resident 的 token-wise swap；启用 prefix 时，其 verified full blocks 会作为
第三个 durable component 进入 L2/L3。默认 ratio=3 时启动容量报告应显示 target
Indexer `3N`、target resident `N`、MTP full `3N` 三个 HBM pool，并在 DRAM
complete block bytes 中增加 MTP BF16 page。
若 prefix caching 与 L3 都关闭，MTP 继续保持原来的纯 HBM 行为，不创建该
DRAM plane，也不增加 steady proposer copy。
若修改了 `SERVED_MODEL_NAME`，同步把 `stream_chat_client.py` 顶部的
`MODEL_NAME` 改成相同值。

DSA L3 P1-W/P1-R 都需要先启动 Mooncake Master。仅验证 write-behind 时设置：

```bash
# no-MTP 保持 false；验证固定 MTP3 多级 prefix 时设为 true。
ENABLE_MTP="false"
ENABLE_L3_WRITE_BEHIND="true"
MOONCAKE_CONFIG_PATH="${SCRIPT_DIR}/L3_cache_sys_api_test/baseline_e2e/configs/mooncake-dsa-tcp.json"
MOONCAKE_GLOBAL_SEGMENT_SIZE="4GB"  # 单 40k DSA block 约占 2.2GB
DSA_L3_MAX_DESCRIPTORS_PER_PUT="128"
DSA_L3_SHUTDOWN_TIMEOUT_SECONDS="300"
```

该模式使用 `DSAL3Connector` 和 Mooncake Store-owned TCP。所有 TP rank 都按
vLLM-Ascend embedded-client 范式贡献一段 `MOONCAKE_GLOBAL_SEGMENT_SIZE`，
但只有 TP rank0 上传一份完整 durable L2 bundle；request finish 不等待，
L2 physical block 被复用前才触发相关 job fence。P1-W 使用
`kv_role=kv_producer`，因此 TTFT 不应出现 external hit。验收时应看到每个 TP rank 的
`contributes_segment=True`，以及唯一 producer 的 `store planned/submitted/`
`completed success=True` 和 Mooncake 对象增长。请求响应不等待异步 put；当
离线脚本随后销毁 EngineCore 时，框架会继续执行 zero-token connector progress
step，待 pending store 完成后再关闭各 rank 的 Mooncake segment。DSA L3
显式要求 `shutdown_timeout>0`；示例预留 300 秒是为了覆盖首次跨 segment 懒建链，
并不改变请求完成时延，只限制进程退出最多等待多久。离线 `LLM` 的默认
finalizer 不向 MPClient 传递该 timeout，因此 `simple_prompt_test.py` 会在成功
生成后显式调用 engine-core shutdown；在线 launcher 原生会传递该配置。
当前 A5 GLM-5.2 no-MTP 完整逻辑块包含 42 个 Indexer plane 和 78 个 MLA
plane；MTP3 再增加 3 个 BF16 plane，总计 123 descriptors，仍低于 128 的单块
预算。示例按 128 个 Mooncake descriptor 切分 backend put，因此一次传一个完整
bundle，避免把多块合成数千个 descriptor 后触发 10 秒同步传输
超时；后续性能调优可在实测门禁下提高该预算。TCP 模式不创建外部
`GlobalTE`，而是由 Mooncake Store 创建 engine，并通过
`store.register_buffer()` 注册 TP0 固定 L2 arena。配置中的
`tcp_segment_port_base + rank` 的保留区间仅用于生成 worker segment identity；
scheduler-only client 使用额外固定 offset，真实 RPC/TCP 监听端口仍由 Mooncake
分配。社区 `mooncake.json` 保持 `protocol=ascend`，不受 DSA TCP 改造影响。

P1-R 同步 prefix loadback 在上述配置上再打开：

```python
ENABLE_PREFIX_CACHING = True
ENABLE_L3_PREFIX_LOADBACK = True
PREFIX_CACHE_REPLAY_ROUNDS = 3
```

当前 Mooncake Store-owned TCP wheel 不支持 NPU device pointer：TCP transport
只为 CUDA/HIP 等设备实现 host staging，直接写 NPU HBM 会在
`ClientSession::readBody` 返回 `EFAULT (Bad address)`。框架因此会在启动期拒绝
任何 `direct_hbm_load=true`；不要通过切换 protocol 或
`MC_STORE_MEMCPY=1` 绕过。当前正式部署必须保持
`ENABLE_L3_DIRECT_HBM_LOAD=False`，统一使用
`L3 -> L2 -> HBM`。direct-HBM 探针只保留为后续后端能力研究，不属于当前
模型服务测试矩阵。

Store-owned TCP 还必须在首次导入 Mooncake 前启用连接复用：

```bash
export MC_TCP_ENABLE_CONNECTION_POOL=1
```

`simple_prompt_test.py` 已自动设置该变量。GLM-5.2 A5 的一个完整 DSA
block 包含 120 个 descriptor；连接池关闭时，TP8 大前缀同步回读会迅速累积
短连接和 `TIME_WAIT`，最终表现为 `connect: Cannot assign requested address`
与 Mooncake `-800 TRANSFER_FAIL`。

Mooncake TCP 用 `host:tcp_segment_port_base+rank` 作为 segment identity，而
不是实际监听端口。同一 Master 上并行运行或新旧实例交叠退出时，各实例必须设置
不同的 `MOONCAKE_TCP_SEGMENT_PORT_BASE`，例如 `45000` 与 `30000`；否则一个
实例挂载/卸载同名 segment 会污染另一个实例的 Master 元数据。离线和在线 demo
均显式暴露该设置。

此时 Connector 改用 `kv_role=kv_both`。离线 demo 依次执行冷算、本地 L2 replay，
然后等待后台 put 一个受控窗口并调用
`reset_prefix_cache(reset_connector=False)`：只清理本地 HBM/L2 hash，保留
Mooncake 对象；最后执行 L3 replay。严格精度断言比较本地 L2 replay 与 L3
replay，避免把 full-prefill 与 prefix-suffix 的数值路径差异误判为 L3 数据损坏。
最后一轮同 prompt 应打印：

```text
DSA L3 prefix lookup: ... hit_blocks>0 external_tokens>0
DSA L3 load planned: ... blocks>0
DSA L3 load backend get finished: ...
DSA L3 load completed: tp_rank=0..TP-1 ...
DSA prefix restore submitted: ...
[dsa-smoke] local-L2/L3 replay consistency passed
```

P1-R 同步执行 `L3 -> 每 rank LocalDRAM -> split HBM`。所有 rank 完成本地
get 后用预分配状态 tensor 做 TP all-reduce；任一 rank 或任一
Indexer/MLA/可选 MTP component 失败都会在 forward 前让全体 fail-fast。DP、
异步 load 与 recompute 仍不支持。Mooncake lookup/load 成功后，第二轮只计算未命中的尾部，
不把已恢复 prefix 再次当作新增块上传。

L3 key 不包含 TP rank，故 Mooncake 中只有一份池化对象；all-rank load 是同一
对象向各 rank 本地 L2 的读出扇出，并不会在 L3 再保存 TP 份副本。

固定 MTP3 多级 prefix 使用同一入口：

```python
ENABLE_MTP = True
ENABLE_PREFIX_CACHING = True
ENABLE_L3_WRITE_BEHIND = True
ENABLE_L3_PREFIX_LOADBACK = True
ENABLE_L3_DIRECT_HBM_LOAD = False
PREFIX_CACHE_REPLAY_ROUNDS = 3
```

此时 GLM-5.2 每个 complete bundle 为 `7,084,544` bytes，L3 每个 block
包含 `indexer/mla/mtp` 三个 key 和 123 descriptors。第二轮本地 replay 与清
本地后的第三轮 L3 replay 都会按基线 MTP 语义在最长 raw prefix 末端回退一个
block；这是重建 proposer 边界状态，不是 cache miss。日志应同时显示非零
`mtp_blocks`、MTP-aware bundle bytes，以及 L3 load 后 target+MTP restore。

### P1-R 精度收缩矩阵

多请求 graph 下若 Local-L2/L3 token 不完全一致，先不要降低通过标准。
`simple_prompt_test.py` 会请求 top-5 logprobs，并在第一个分歧位置打印双方
候选。先用 Case A/B 隔离 L3，再用 Case C 继续归因 graph 数值变化：

```python
# Case A：5k 单请求仍走 dense decode，但覆盖完整 L3 prefix loadback。
PROMPTS = chinese_5k
RUN_MODE = "graph"
MAX_NUM_SEQS = 1
MAX_TOKENS = 5
ENABLE_CHUNKED_PREFILL = True
ENABLE_PREFIX_CACHING = True
PREFIX_CACHE_REPLAY_ROUNDS = 3
ENABLE_L3_WRITE_BEHIND = True
ENABLE_L3_PREFIX_LOADBACK = True
```

```python
# Case B：不清 Local-L2，检查混合 bsz=4 graph 自身是否严格稳定。
PROMPTS = chinese_8k + chinese_5k + chinese_70k + chinese_40k
RUN_MODE = "graph"
MAX_NUM_SEQS = 4
MAX_TOKENS = 5
ENABLE_CHUNKED_PREFILL = True
ENABLE_PREFIX_CACHING = True
PREFIX_CACHE_REPLAY_ROUNDS = 3
ENABLE_L3_WRITE_BEHIND = True
ENABLE_L3_PREFIX_LOADBACK = False
```

```python
# Case C：关闭 DSA/L3，检查原生 HBM prefix 的混合 graph 稳定性。
PROMPTS = chinese_8k + chinese_5k + chinese_70k + chinese_40k
RUN_MODE = "disabled"
MAX_NUM_SEQS = 4
MAX_TOKENS = 5
ENABLE_CHUNKED_PREFILL = True
ENABLE_PREFIX_CACHING = True
PREFIX_CACHE_REPLAY_ROUNDS = 3
ENABLE_L3_WRITE_BEHIND = False
ENABLE_L3_PREFIX_LOADBACK = False
```

三组都保持：

```python
PREFIX_REPLAY_LOGPROBS = 5
COMPARE_LAST_TWO_PREFIX_REPLAYS = True
STRICT_PREFIX_REPLAY_TOKEN_MATCH = True
ALLOW_MULTI_REQUEST_GRAPH_TOPK_VARIANCE = True
```

Case A 比较 Local-L2/L3；Case B 因不执行本地 reset，会比较连续两轮
Local-L2。实际 A5 GLM-5.2 结果中，Case A 的 38 个块、262,364,160
bytes/rank 严格一致；Case B 在没有任何 L3 load 的情况下，于 5k 请求第 5
个生成 token 复现了混合 bsz=4 的同一分叉。更重要的是，启用和禁用 L3
loadback 的 round2/round3 四条输出按 replay 轮次逐条相同，因此该差异属于
多请求 graph replay 的既有数值变化，不是 L3 数据损坏。

当前自动门禁采用分层规则：bsz=1 或 eager 仍要求 token 完全一致；仅对
bsz>1 graph，若第一个分歧前已有相同 token，且双方选中 token 都出现在对方
同位置的 top-k 中，则记录为 `top-k numerical consistency passed`。缺失
logprobs、首 token 即分叉、候选不互相包含等情况仍严格失败。该规则不能替代
NPU component bitwise UT、all-rank get 成功、block 数一致和 restore 完成日志。
Case C 不阻塞 L3 P1-R 验收，仅用于进一步判断多请求 graph 数值变化来自社区
HBM prefix 基线，还是 DSA Local-L2 restore/调度路径。

注意：该配置下 `enable_prefix_caching=True` 不代表 target HBM Indexer/MLA 已参与
匹配。DSA 已拦截基线 target HBM hit，当前查询顺序是 LocalDRAM L2 后接 L3；
HBM component-wise 命中属于后续 P3-HOT。

```bash
bash examples/dsa_demo/serve_dsa.sh 2>&1 | tee dsa-online.log
```

服务就绪后，另开终端检查：

```bash
curl -H "Authorization: Bearer EMPTY" \
  http://127.0.0.1:8000/v1/models
```

运行交互式流式客户端：

```bash
python examples/dsa_demo/stream_chat_client.py
```

客户端会保留并在下一轮重新发送完整消息历史。设置
`LONG_CONTEXT_FILE="/path/to/long_context.txt"` 可以把长文本加入 system
消息；只有实际 token 长度超过 `sparse_activation_tokens`，才能验证
ENTER/SPARSE 路径，而不是只验证在线 API 和 DENSE 路径。默认
`ENABLE_PREFIX_CACHING=false` 时，多轮长上下文测试每次都会重新 prefill
完整历史；设为 `true` 后，可用相同前缀的第二轮请求验证本地 DRAM 命中。

若要复现离线 `llm.generate()` 的原始小说续写，而不是让 chat template
把小说包装成一轮对话，可在 `stream_chat_client.py` 的用户配置区设置：

```python
from novel_dataset import chinese_40k

CONTINUATION_PROMPT = chinese_40k[0]
MAX_TOKENS = 512
L3_PREFIX_REPLAY_ROUNDS = 3
```

此时脚本会调用流式 `/v1/completions`，将 `chinese_40k[0]` 原样作为
prompt。P1-R 三轮模式依次验证 cold、Local-L2 replay，并调用 vLLM 原生
`/pause?mode=wait&clear_cache=false` 确定性 drain pending store，再调用
`/reset_prefix_cache?reset_external=false` 并 `/resume` 后执行 L3 replay；严格比较
后两轮文本。固定 sleep 不能证明异步上传完成，不再作为在线验收屏障。
`serve_dsa.sh` 只在 `ENABLE_L3_PREFIX_LOADBACK=true` 时为该内部门禁设置
`VLLM_SERVER_DEV_MODE=1`，生产服务不要启用开发接口。
`chinese_40k` 在测试数据中是只包含一条
文本的 list，因此这里需要取 `[0]`。

### 在线 L2/L3 收口矩阵

完成单条 40k 冒烟后，使用 `online_l2_l3_closeout_test.py` 一次覆盖四种
长度、chunked prefill、continuous batching、row reuse 和本地 cache reset
后的 L3 loadback。服务端建议配置：

```bash
MODEL_PATH="/models/GLM-5.2-w4a4c8-mxfp4"
SERVED_MODEL_NAME="glm-5.2-dsa"
RUN_MODE="graph"                 # 首轮也可先用 eager
ENABLE_A5_PACKED_C8_DSA="true"
ENABLE_MTP="false"
ENABLE_L3_WRITE_BEHIND="true"
ENABLE_L3_PREFIX_LOADBACK="true"
ENABLE_PREFIX_CACHING="true"
TENSOR_PARALLEL_SIZE="8"
MAX_NUM_SEQS="4"
MAX_MODEL_LEN="131072"
MAX_NUM_BATCHED_TOKENS="16384"
ENABLE_CHUNKED_PREFILL="true"
```

这些值位于 `serve_dsa.sh` 顶部用户配置区。先启动同目录文档指定的 Mooncake
Master，再拉起服务：

```bash
bash examples/dsa_demo/serve_dsa.sh 2>&1 | tee dsa-online-l2-l3.log
```

确保服务器上的 `novel_dataset.py` 可被 Python 导入，且包含
`chinese_5k`、`chinese_10k`、`chinese_40k`、`chinese_64k`。客户端顶部的
`SERVER_RUN_MODE` 必须与服务端 `RUN_MODE` 一致，然后运行：

```bash
python examples/dsa_demo/online_l2_l3_closeout_test.py \
  2>&1 | tee dsa-online-l2-l3-client.log
```

脚本不把四条 prompt 同时作为首次 seed batch，因为同批请求不能复用尚未
发布的 prefix。它按以下顺序执行：

1. 先只清 HBM/L2，四条请求逐条 seed；已有 L3 对象则加载，缺失才 cold
   计算并触发异步 L2→L3 put；
2. 四条请求逐条执行 Local-L2 A/B 自对照；
3. pause/drain 后只清 HBM/L2 namespace，逐条执行两次强制 L3 A/B；
4. 对 Local-L2/L3 分叉做同介质重放归因；
5. 多轮 `bsz=4`，每轮执行 Local-L2 A、Local-L2 B、清本地后的 L3；
6. 最后再次 pause/drain，结果写入
   `/tmp/dsa-online-l2-l3-closeout.json`。

eager 始终要求 token IDs 严格一致。graph 若发生 top-token 边界分叉，首个
分叉 token 必须互相位于对方 top-k，而且同一请求的差异必须先在对应的
Local-L2/Local-L2 或 L3/L3 自对照中出现；稳定但跨介质不同的结果仍直接失败。
脚本会给 request ID 加上 `dsa-closeout-<run>-<phase>-<length>` 前缀，并打印
首个分叉位置的实际 top-k logprob，便于关联服务端 hit/load/restore 日志。
单个对照失败时不会立即中止，脚本会采完串行与并发全部阶段，再统一以非零
状态退出并把 `diagnostic_failures` 写入结果 JSON。
脚本默认 `FRESH_EXTERNAL_NAMESPACE_AT_START=False`，因此相同服务上反复运行
会复用相同 Mooncake key，不会整份重写 prompt。只有需要隔离旧对象的 cold
门禁时才临时设为 `True`；该操作只轮换 namespace，不物理删除旧对象，会一直
占用容量直到 Mooncake 淘汰。5k 请求默认只生成 8 tokens，避免它的 109/128
尾块在不同续写中形成新的 durable key。顺序阶段保持 SSE 流式打印，并发阶段
缓冲输出以真正形成 `bsz=4`。

服务日志还必须同时满足：

- 首次或 L3 缺失的 seed 阶段出现非零
  `store planned/submitted/completed success=True`；已有对象的重复测试应看到
  backend hit，而不是相同完整前缀再次增加容量；
- L3 阶段四条请求均出现非零 external hit、all-rank get 和 restore；
- reset 前后没有 stale job、refcount underflow、连接失败或 pending store；
- 客户端最终打印 `PASSED`，JSON 中 `status` 为 `passed`。

### P1-HOT 成对 HBM target prefix

P1-HOT 不增加启动参数，沿用 no-MTP、`ENABLE_PREFIX_CACHING=true` 的 L2/L3
配置。首次 seed 完成后，相同 prompt 的下一次本地 replay 应打印：

```text
DSA prefix lookup: ... hbm_pair_blocks>0 l2_blocks=... hbm_only_blocks=...
DSA L3 prefix lookup: ... local_blocks=max(hbm_pair_blocks,l2_blocks)
```

HBM pair 只覆盖同时命中的 Indexer+dense MLA，但不再要求同长度 L2 bundle
仍存在。若 `hbm_pair_blocks > l2_blocks`，首个 prefill forward 会产生非零
full-block dump，model output 返回后 L2 ready/hash 才推进到完整 durable
前缀。L3 suffix 始终打印 `destination=l2`，随后执行既有 restore。MTP 开启时
HBM 使用 target pair+MTP triple，L2/L3 使用三 component complete bundle；
Indexer-only、MLA-only 与逐 component IO 去重仍属于后续阶段。

建议按以下顺序收口：

1. `tests/ut/dsa_offload` 全量通过，重点覆盖 pure-HBM、HBM>L2、
   HBM-only gap+L3 suffix 和 restore-ready 水位；
2. no-MTP、无 L3、5K dense prompt 连续 replay 两轮，分别跑 eager/graph，
   第二轮确认 `hbm_pair_blocks>0`；
3. no-MTP、开启 L3 TCP、5K/8K/40K/70K graph/chunked 三轮 replay，确认
   L3 load 始终为 `destination=l2`，且没有 ready watermark 回退；
4. 最后补在线 graph 单 40K 与混合并发。当前必须保持
   `ENABLE_L3_DIRECT_HBM_LOAD=False`。

不安装 Python OpenAI SDK 时，也可以直接观察 SSE 流：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer EMPTY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.1-dsa",
    "messages": [{"role": "user", "content": "你好，请介绍一下你自己。"}],
    "temperature": 0,
    "max_tokens": 128,
    "stream": true
  }'
```

在线冒烟至少核对：

- 启动日志中 DSA HBM/DRAM 两份容量报告各只出现一次；
- `/v1/models` 返回 `glm-5.1-dsa`；
- SSE/客户端内容逐步输出，而非等待完整回答后一次性打印；
- 同一短请求在在线 eager 与离线 eager 下无明显语义差异；
- 长上下文请求能够进入 DSA 算子路径；
- 连续多轮、`/clear` 后新请求以及请求结束后的 row 复用均无串扰；
- eager 通过后再复测 graph，比较 token IDs、结束原因和服务日志。

### 在线长上下文数据集

`online_long_context_dataset_test.py` 可直接复用已经拉起的在线服务。测试数据包
包含 LV-Eval 中文 multifield QA 的 16K/32K/64K/128K 档，以及 CLongEval
小说问答和键值检索的 small/medium/large 档。

先把数据包解压到服务器，例如：

```bash
unzip -q dsa_long_context_dataset_pack.zip -d /home/data
```

确认脚本顶部配置：

```python
API_BASE = "http://127.0.0.1:8000/v1"
MODEL_NAME = "glm-5.1-dsa"
LOCAL_TOKENIZER_PATH = None
DATASET_ROOT = "/home/data/dsa_long_context_dataset_pack"
RUN_LABEL = "dsa-eager-online"
ENABLE_THINKING = False
MAX_SAMPLES_PER_FILE = None
```

服务无需重启，直接运行：

```bash
python examples/dsa_demo/online_long_context_dataset_test.py
```

脚本按照 `dataset_specs()` 中的数据集顺序和 JSONL 原始行序完整处理样本，
不按长度选样或重排。`MAX_SAMPLES_PER_FILE = None` 表示完整处理每个文件；
仅在快速调试脚本时才设置正整数上限。脚本从 `/v1/models` 返回的模型
`root` 本地加载同一 tokenizer（不加载模型权重），提前计算 prompt token 数
并拦截超过 `max_model_len` 的样本。服务使用单独的 `--tokenizer` 路径时，
需要通过 `LOCAL_TOKENIZER_PATH` 显式指定。

第一条有效样本会额外调用一次服务端 `/tokenize`，确认本地 tokenizer、chat
template 和 `enable_thinking` 得到的计数完全一致；其余样本均在本地预检。
正式响应的 `usage.prompt_tokens` 会再次与预检值核对。样本按实际长度标记为
`0-6K`、`6-16K`、`16-32K`、`32-48K`、`48-64K`、`64-96K` 或
`96K+`。

每 `REQUEST_CONCURRENCY` 条样本组成一个客户端并发组。脚本会在发送前逐行
打印数据集、子集、样本 ID、任务要求、问题（或检索键）和 golden answer；
同时打印预检 token 数和长度区间。组完成后再打印正式响应中的实际 token
数。脚本不会人为安排 mixed-length 组合；实际进入同一个引擎 batch 的请求仍
由在线调度器决定。结果保存为 JSONL，并生成按数据集、任务类型和实际 token
档汇总的 `summary.json`。

当前接入的任务包括：

- CLongEval `long_story_qa`：长篇小说问答；
- CLongEval `key_passage_retrieval`：从 JSON 长上下文中检索指定键的值；
- LV-Eval `multifieldqa_zh_mixup`：多文章混合场景问答；
- 可选 LongBench `multifieldqa_zh`：长上下文问答。

数据包中的 NeedleBench 文件目前只是后续扩展所需的原始语料和 needle 定义，
尚未被该脚本组装成测试样本。

GLM-5.1 的 chat template 默认开启 thinking。短答案精度回归保持
`ENABLE_THINKING = False`；否则较小的 `MAX_TOKENS` 可能全部成为
reasoning token，最终答案为空。

`character_f1` 和 `contains_answer` 是不依赖额外包的冒烟指标，不等价于
LV-Eval 或 CLongEval 官方排行榜分数。正式精度对比应保持相同数据包、服务
参数和 `RUN_LABEL`，分别拉起 baseline、DSA eager 与 DSA graph 后运行。

## 8. 当前不支持

- prefill/decode mixed batch（chunked prefill 本身已支持）；
- async scheduling；
- HBM Indexer-only/MLA-only 的独立 source mask 与细粒度 IO 去重；完整 HBM
  pair 与 HBM-only L2 backfill 不在该未支持项中；
- preemption/resume；
- 非 MTP speculative method、MTP draft 数不等于 3、非 greedy MTP；
- L3 的跨机DP、layerwise async load 和 HBM 直通；固定MTP3多级prefix及
  单机DP2/TP4/EP8共享Mooncake TCP已有上述阶段性验收，不再列为未支持；
- `DSAL3Connector` 之外的 DSA KV transfer connector，以及非 Mooncake 后端；
- context parallel 和 pipeline parallel；
- KV-cache metrics/events；
- A5 BF16 算子链、完整 QA 与性能验收。

这些能力会在后续按独立合同逐项扩展。在相应实现和回归完成前，不要通过
删除启动期校验强行打开。
