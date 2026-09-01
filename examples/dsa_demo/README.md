# DSA 稀疏卸载测试入口

本目录是 v0.23 DSA 的 tester-facing 验收入口。脚本以 GLM-5.1/GLM-5.2
为首要模型，DeepSeek-V3.2 使用相同配置做强制回归。

| 文件 | 用途 |
|---|---|
| `simple_prompt_test.py` | 基线隔离、cache-init、eager 和 graph 冒烟 |
| `qa_dataset_test.py` | LongBench 风格数据集精度回归，可切 disabled/eager/graph |
| `eval_dataset_acc_score.py` | 对 QA 脚本生成的 JSONL 计算 LongBench 指标 |
| `serve_dsa.sh` | 以 OpenAI 兼容在线服务拉起 DSA eager/graph |
| `stream_chat_client.py` | 交互式多轮流式客户端 |
| `online_l2_l3_closeout_test.py` | 5k/8k/40k/70k 在线 L2/L3 生命周期收口 |
| `online_long_context_dataset_test.py` | 在线长上下文数据集分桶回归 |
| `L3_cache_sys_api_test/` | Mooncake/MemCache 独立 API、baseline L3 与后续 DSA 分层缓存测试 |

当前架构和稳定 ABI 见
[DSA 稀疏卸载详细设计](../../docs/source/developer_guide/Design_Documents/dsa_offload_design.md)。
L3 接入、异步 write-behind 和 HBM 热层后续方案见
[DSA L1/L2/L3 分层缓存详细设计](../../docs/source/developer_guide/Design_Documents/dsa_l2_l3_cache_design.md)。
阶段优先级与退出门禁见
[DSA 稀疏卸载后续开发 RoadMap](../../docs/source/developer_guide/Design_Documents/dsa_offload_roadmap.md)。

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

先修改 `serve_dsa.sh` 顶部用户配置。首次在线验证建议从
`RUN_MODE="eager"` 开始，确认请求生命周期、连续请求和流式返回正常后，再
切换为 `RUN_MODE="graph"`：

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
`{"method":"mtp","num_speculative_tokens":3}`。MTP3 只接受
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
`chinese_5k`、`chinese_8k`、`chinese_40k`、`chinese_70k`。客户端顶部的
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
  pair 与 HBM-only L2 backfill 已完成代码改造、待 A5 上板；
- preemption/resume；
- 非 MTP speculative method、MTP draft 数不等于 3、非 greedy MTP；
- 固定 MTP3 多级 prefix 尚待 A5 上板；L3 的 DP>1、layerwise async load 和
  HBM 直通；
- `DSAL3Connector` 之外的 DSA KV transfer connector，以及非 Mooncake 后端；
- context parallel 和 pipeline parallel；
- KV-cache metrics/events；
- A5 BF16 算子链、完整 QA 与性能验收。

这些能力会在后续按独立合同逐项扩展。在相应实现和回归完成前，不要通过
删除启动期校验强行打开。
