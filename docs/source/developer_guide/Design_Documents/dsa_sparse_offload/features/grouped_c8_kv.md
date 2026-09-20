# 四层 IndexShare Grouped C8 KV（实验路径）

默认关闭。仅 A5 packed-C8、DSA offload、MTP3、copy + SFA 分离路径，
PP/PCP/DCP=1。DP/TP 数量不参与分组；每个 worker 独立按模型 `indexer_types` 分组。
初版拒绝 prefix caching、KV transfer/L3 和 CopySFA 融合；不改变 baseline、非 MTP 或旧算子的 ABI。

## 开关

在已有 `additional_config["dsa_sparse_config"]` 中加
`"enable_grouped_c8_kv": True`。省略或设为 `False` 走原来的逐层存储路径。
不要覆盖其它 DSA 配置，`"enable_fused_copy_sfa"` 保持 `False`。

`simple_dp_prompt_test.py` 可直接设置：

```python
ENABLE_GROUPED_C8_KV = True
ENABLE_DSA_OFFLOAD = True
ENABLE_A5_PACKED_C8_DSA = True
ENABLE_MTP = True
MTP_NUM_SPECULATIVE_TOKENS = 3
ENABLE_FUSED_COPYSFA = False
ENABLE_PREFIX_CACHING = False
ENABLE_L3_WRITE_BEHIND = False
ENABLE_L3_PREFIX_LOADBACK = False
REPLAY_ROUNDS = 1
ENABLE_PROFILE = False
```

其它单机/双机入口使用同一 `additional_config` 字段，不需要按 DP/TP 组合另建实现。
先保持原来已通过精度验证的 prompt、budget、采样和并行配置，**只改变此开关**做 A/B。
首次先用 eager，再用 target `FULL_DECODE_ONLY` + MTP 三轮合并图。

## 存储及调用

| 部分 | 新路径 |
| --- | --- |
| target HBM、L2 DRAM | 同一 IndexShare 组 `[B,128,1,2624]`；四个 lane 各保存原来的 656B |
| 分组边界 | full 开始，紧随其后的 shared 跟随；只接受独立层或 full+3 shared。典型 GLM 为 0、1 独立，2–5 为首个四层组 |
| LIM | 原算子、原状态、原 block table。owner 做一次选择，三层复用；不改变 hash、淘汰、topK 或 tail 规则 |
| copy | 新 `VllmA5KvcacheScatterCopyGroup4C8`，owner 一次搬运四层；每个 `DataCopyPad` 搬 2624B，shared 不再 copy |
| Prolog / KV 写入 | 零拷贝 PA view `[4B,128,1,656]`，仅 KV 写入地址转换为 `4*slot+lane`；Indexer 的 slot 不变 |
| SFA | 复用原生 QSFA，对 sparse slot 和 block table 做等价地址转换；每层只读取自己的 lane |
| 满块 dump | 复用原有通用 `kv_cache_full_block_dump_bytes`，每个块搬运 `128*2624` 字节 |
| Indexer、MTP draft | Indexer 仍逐 owner 存储；draft 仍 BF16 full cache，不做四层打包 |

四层合并**不是**把一个层的 656B 扩成四倍缓存，模型总 payload 字节数不变。
allocator 直接分配一个组，不先分配四层旧 cache 再拼接，也不额外保存一份旧布局。

对一个逻辑 sparse slot `s` 和 lane `l`：新地址为 `4*s+l`，
block table 每项 `b` 展开为 `4*b+[0,1,2,3]`。寻址结果等于
`(physical_block*128 + token_in_block)*2624 + l*656`。
负数 padding 保留负数；实际 query/KV 长度及有效注意力项数量**不乘四**。
prefill 保持 causal mode=3，LIM MTP 已排除未来项的 selection 保持 mode=0。
这些转换是独立的新路径，不修改原生 Prolog/SFA 内核。

decode 的 dump 在 owner 写入任何新 KV **之前**完成全组旧满块搬运。
prefill 的 dump 在组内最后一层写完 **之后**完成，不能在 owner 写完就把其余三层旧数据卸载。
因此 owner 的单次分组 copy 可以访问完整的四层 DRAM 数据，不依赖后续层再补 dump。

## 编译与验证

需要在 A5 环境重新编译安装本仓库的 **CANN 自定义算子和 Torch 扩展**；只更新 Python 不够。
先检查新的 Torch 入口存在：

```bash
python -c 'from vllm_ascend.utils import load_custom_op_library; import torch; load_custom_op_library(); print(torch.ops._C_ascend.npu_dsa_a5_kvcache_scatter_copy_group4_c8_out.default._schema)'
```

单卡算子验收（不加载模型）：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 ASCEND_VISIBLE_DEVICES=0 python -m pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_dsa_grouped_c8_ops.py
```

覆盖 grouped dump/copy、无效行、跨 128-token 边界、四个 lane 的 QSFA 与旧布局逐元素对照、
eager/图重放、变化的索引和 block table，并打印 2048-token copy 微基准。
不对耗时设硬阈值：DRAM 链路负载与 host 下发开销会影响测量，不能把 copy 提速直接等同于 ITL 提速。

整网必须再验证：短 prompt → 超过 activation 的长 prompt → 跨 tail 边界生成至少 256 token →
BS1/BS8、DP idle/padding、不同 TP、eager/graph。关闭 profile 做逐 token 对照，再单独采性能。
日志应出现 `[DSA grouped C8 KV] layers=... shape=(...,128,1,2624)`。
本地 CPU 测试不能证明 NPU 编译成功、Prolog/QSFA 版本兼容或整网精度；这些是启用该实验路径的设备验收门禁。
