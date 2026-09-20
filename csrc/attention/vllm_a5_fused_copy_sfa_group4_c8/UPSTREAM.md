# Group4 CopySFA

独立新增的 `VllmA5FusedCopySfaGroup4C8`，不修改原 `VllmA5FusedCopySfaC8`。

SCFA/MLA、反量化、softmax、matmul 等源码直接 include 相邻
`vllm_a5_fused_copy_sfa_c8/op_kernel/fused_c8/`，保留其 Huawei 版权声明及许可证。
来源及许可见 [原算子说明](../vllm_a5_fused_copy_sfa_c8/UPSTREAM.md) 和
[LICENSE](../vllm_a5_fused_copy_sfa_c8/LICENSE)。CMake 声明同仓依赖，源码构建与
打包后的扁平 kernel 路径均有对应 include，不依赖外部算子仓。

新增入口采用完整四层 union copy、全核同步、owner SFA 的顺序；不启用 source-aware
DRAM gather 或预取，不宣称 copy/SFA 流水重叠。`fused_group4_c8_copy.h` 基于原
`fused_c8_copy.h`，仅将 row/UB 单缓冲粒度改为 2624B、采用逻辑 copy 页宽，保留
双缓冲显式事件与全核 barrier。两侧 UB 前缀均为 5248B，避免 SFA Fixpipe 地址不一致。
所有 union 条目都持久写回，不能套用原算子 small-miss 只读本层 attention 的策略。

复用原 tiling 数据结构；其中 SFA 的页宽为逻辑页宽的四倍，copy 时还原为逻辑页宽。
原逻辑 copy metadata 与新增虚拟 SFA slots/table 独立输入，长度与有效项数量不乘四。
Torch 适配复用原 shape checker、stream-ordered launcher 和 workspace 生命周期逻辑。
新 operator、Torch schema、ACLNN 符号均独立命名，原 ABI 和默认开关不变。

设备门禁：`tests/e2e/nightly/single_node/ops/singlecard_ops/test_dsa_grouped_copysfa_ops.py`。
CPU 搬运 emulation 和 ATen 语法测试不能证明 CANN 编译、设备同步或端到端精度通过。
