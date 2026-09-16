# 源码来源

`op_kernel/fused_c8/` 的 SCFA/MLA、反量化、softmax、matmul/buffer/VF 基于本地 `vLLM-ascend-DSA` 提交 `3b5d0ecdc` 的 `csrc/attention/kv_quant_sparse_flash_attention/op_kernel/arch35`、`csrc/attention/common/op_kernel` 及 `sparse_flash_attention/op_kernel/arch35/util_regbase.h`；`util.h` 来自 `ops_copysfa_a5` 的 A5 SFA 依赖。保留 Huawei 版权声明，许可证见本目录 `LICENSE`。所有依赖已放入本目录，构建不依赖外部仓库。

本仓修改：本地 include 路径、固定 TND/PA_BSND/FP8 packed KV 的精简 tiling、紧凑 query 分核、有效 slot 前缀长度、source-aware gather、逐 query 唯一写回归属及 MTE 预取。修改集中在入口、`fused_c8_copy.h`、`fused_c8_schedule.h`、`fused_source_gather.h` 和 kernel/vector 服务；反量化公式、两次矩阵乘及在线 softmax 复用上游实现。两种 dtype 分别以 query dtype 解释 packed RoPE。`buffer.h` 的显式同步 ID 设置接口取自 `ops_copysfa_a5`；MTE gather/padding 使用 `uint8_t`，只在 VF 反量化入口按 FP8 重解释，避免 Bisheng 不支持的 FP8 标量赋值。

流水参考：`ops_copysfa_a5` 的 hit 阶段预取、V1/V2 后等待写回；`ops_copysfa_mtp` 的每 query source/miss 输入。A3 指令未直接搬到 A5。有效 slot 长度用于屏蔽 `-1`，不把旧版官方 SFA 的 padding 精度风险当作目标语义；单测分别报告 fused 与官方的独立逻辑 golden 误差。

大 copy-count 阶段使用两个 672-byte UB buffer 和显式 MTE2/MTE3 事件；排空读写并释放事件后，所有 AIC/AIV 执行 `SyncAll<false>()` 再初始化 SFA。两侧预留相同的 1344-byte UB 前缀并保留至 kernel 结束，确保后续 Fixpipe 缓冲区地址一致；不执行 `TPipe.Reset()`，避免重置随后 SFA 依赖的事件分配状态。

框架集成来源：`ops_lidu_scattercopy_sfa_a5` 提交 `19e5615`。`op_host` 的 C++ 与全部 `op_kernel` 文件保持源码一致；新增 framework CMake、`_C_ascend` caller-owned 适配及局部 workspace cache。保留预取默认开启、512 fallback 阈值、miss pair 合并写回。eager workspace 按 device/stream 复用；capture 使用独立图池，不复用 eager cache。

本次同步单算子仓 `6732a649a36e4829994aa6cbdbd7ad6616333180` 的全可见域 TopK2048 版本：host C++ 和 kernel 与源仓一致；配对输入改为 `topk_slots/topk_index[T,1,2048]`，不再追加 tail，等价于 COPY + 官方 QSFA `sparse_mode=0`。保留 >512 内部回退与现有流水。框架 CMake、Torch 适配和 workspace cache 保留，仍在 `_C_ascend` 注册；默认不开启融合。MTP 的 L3 Attention 语义指纹同步升级，但不改块分配和缓存生命周期。必须连同新版 LIM、CANN so、Torch binding 和 Python 一起部署，重新 capture graph。目标 A5 验证入口见 `examples/dsa_demo/README.md`，单算子通过不等于框架编译和整网验收。
