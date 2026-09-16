// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cmath>
#include <limits>
#include <tuple>
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "copysfa_workspace_cache.h"

// CopySFA-only queue-ordered launcher; the other operators' allocator is unchanged.
namespace vllm_ascend {
namespace copysfa_detail {
using InitPTACacheThreadLocal = void (*)();
using SetPTAHashKey = void (*)(uint64_t);
using UnInitPTACacheThreadLocal = void (*)();
using OpApiFunc = int (*)(void*, uint64_t, aclOpExecutor*, const aclrtStream);
using AclUseStreamResFunc = void (*)(void*);

inline AclUseStreamResFunc GetUseStreamResFuncCoreNum() {
    static void* handle = dlopen("libascendcl.so", RTLD_NOW);
    static auto func = handle == nullptr ? nullptr :
        reinterpret_cast<AclUseStreamResFunc>(dlsym(handle, "aclrtUseStreamResInCurrentThread"));
    return func;
}
inline void UnInitCacheThreadLocal() {
    static auto func = reinterpret_cast<UnInitPTACacheThreadLocal>(
        GetOpApiFuncAddr("UnInitPTACacheThreadLocal"));
    if (func) func();
}
inline at::Tensor AllocateCopySfaWorkspace(uint64_t bytes, aclrtStream stream) {
    TORCH_CHECK(bytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "CopySFA workspace exceeds tensor size limits");
    const int device = c10_npu::getCurrentNPUStream().device_index();
    auto allocate = [device](uint64_t size) {
        return at::empty({static_cast<int64_t>(size)},
            at::TensorOptions().device(c10::Device(c10::DeviceType::PrivateUse1, device)).dtype(at::kByte));
    };
    if (c10_npu::currentStreamCaptureStatusMayInitCtx() != c10_npu::CaptureStatus::None) {
        // A graph may replay on another stream. Never share eager workspace
        // with a graph, or retain graph-private allocations in the eager cache.
        return allocate(bytes);
    }
    // Deliberately process-lived: one grow-only high-water buffer per device/stream.
    // This operator-local persistent state is necessary for workspace reuse;
    // the queued closure retains any old buffer while the cache grows.
    static auto* cache = new vllm_dsa_a5::CopySfaWorkspaceCache<at::Tensor>();
    return cache->acquire(device, reinterpret_cast<uintptr_t>(stream), bytes, allocate).tensor;
}
}  // namespace copysfa_detail
#define EXEC_DSA_COPYSFA_ORDERED_IMPL(aclnn_api, tensor_keepalive, allocate_workspace, ...)                               \
    do {                                                                                                               \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                  \
        static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                                \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                                    \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                                \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                                         \
        static const auto initPTACacheThreadLocalAddr = GetOpApiFuncAddr("InitPTACacheThreadLocal");                   \
        static const auto setPTAHashKeyAddr = GetOpApiFuncAddr("SetPTAHashKey");                                       \
        TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr, #aclnn_api, " or ",               \
                    #aclnn_api "GetWorkspaceSize", " not in ", GetOpApiLibName(), ", or ", GetOpApiLibName(),          \
                    "not found.");                                                                                     \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                                \
        if (c10_npu::check_enqueue_need_use(acl_stream)) {                                                             \
            auto useStreamResFunc = copysfa_detail::GetUseStreamResFuncCoreNum();                                                      \
            TORCH_CHECK(useStreamResFunc != nullptr, "aclrtUseStreamResInCurrentThread is unavailable.");            \
            useStreamResFunc(acl_stream);                                                                              \
        }                                                                                                              \
        uint64_t workspace_size = 0;                                                                                   \
        uint64_t *workspace_size_addr = &workspace_size;                                                               \
        aclOpExecutor *executor = nullptr;                                                                             \
        aclOpExecutor **executor_addr = &executor;                                                                     \
        InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                    \
        UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);            \
        copysfa_detail::InitPTACacheThreadLocal initPTACacheThreadLocalFunc =                                                          \
            reinterpret_cast<copysfa_detail::InitPTACacheThreadLocal>(initPTACacheThreadLocalAddr);                                    \
        copysfa_detail::SetPTAHashKey setPTAHashKeyFunc = reinterpret_cast<copysfa_detail::SetPTAHashKey>(setPTAHashKeyAddr);                          \
        if (initPTACacheThreadLocalFunc && setPTAHashKeyFunc) {                                                        \
            initPTACacheThreadLocalFunc();                                                                             \
            setPTAHashKeyFunc(0);                                                                                      \
        }                                                                                                              \
        if (initMemFunc) {                                                                                             \
            initMemFunc(nullptr, false);                                                                               \
        }                                                                                                              \
        auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);                         \
        static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);             \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                          \
        TORCH_CHECK(workspace_status == 0, "call " #aclnn_api "GetWorkspaceSize failed, detail:",                   \
                    aclGetRecentErrMsg());                                                                             \
        void *workspace_addr = nullptr;                                                                                \
        at::Tensor workspace_tensor;                                                                                   \
        if (workspace_size != 0) {                                                                                     \
            workspace_tensor = allocate_workspace(workspace_size, acl_stream);                                        \
            workspace_addr = const_cast<void *>(workspace_tensor.storage().data());                                    \
        }                                                                                                              \
        auto ordered_tensor_keepalive = tensor_keepalive;                                                              \
        auto acl_call = [converted_params, workspace_tensor, workspace_addr, workspace_size, acl_stream, executor,     \
                         ordered_tensor_keepalive]()->int {                                                            \
            (void)workspace_tensor;                                                                                    \
            (void)ordered_tensor_keepalive;                                                                            \
            if (c10_npu::check_dequeue_need_use(acl_stream)) {                                                         \
                auto useStreamResFunc = copysfa_detail::GetUseStreamResFuncCoreNum();                                                  \
                TORCH_CHECK(useStreamResFunc != nullptr, "aclrtUseStreamResInCurrentThread is unavailable.");        \
                useStreamResFunc(acl_stream);                                                                          \
            }                                                                                                          \
            copysfa_detail::OpApiFunc opApiFunc = reinterpret_cast<copysfa_detail::OpApiFunc>(opApiFuncAddr);                                          \
            auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                            \
            TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg());                 \
            ReleaseConvertTypes(converted_params);                                                                     \
            ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                          \
            if (releaseMemFunc) {                                                                                      \
                releaseMemFunc(nullptr, false);                                                                        \
            }                                                                                                          \
            return api_ret;                                                                                            \
        };                                                                                                             \
        at_npu::native::OpCommand::RunOpApiV2(#aclnn_api, acl_call);                                                   \
        if (unInitMemFunc) {                                                                                           \
            unInitMemFunc(nullptr, false);                                                                             \
        }                                                                                                              \
        copysfa_detail::UnInitCacheThreadLocal();                                                                                      \
    } while (false)

inline void check_fused_copy_sfa_c8(const at::Tensor& query,
    const at::Tensor& hbm_kv_bytes,
    const at::Tensor& dram_kv_bytes,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    const at::Tensor& topk_slots,
    const at::Tensor& topk_index,
    const at::Tensor& per_query_miss_counts,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& resident_seq_lengths,
    const at::Tensor& copy_src_ids,
    const at::Tensor& copy_dst_slots,
    const at::Tensor& copy_counts,
    double scale_value, const at::Tensor& attention_out) {
  const auto b = copy_counts.numel();
  const auto t = query.dim() == 3 ? query.size(0) : 0;
  TORCH_CHECK(query.dim() == 3 && t > 0 && query.size(2) == 576 &&
      query.size(1) > 0 && (query.size(1) <= 64 || query.size(1) == 128),
      "fused_copy_sfa_c8 query must be [T,H,576], H=1..64 or 128");
  TORCH_CHECK(query.scalar_type() == at::kBFloat16 || query.scalar_type() == at::kHalf,
      "query must be bf16/fp16");
  TORCH_CHECK(copy_counts.dim() == 1 && b > 0 && t >= b && t <= 4 * b,
      "copy_counts must be [B], B <= T <= 4B");
  for (const auto* cache : {&hbm_kv_bytes, &dram_kv_bytes}) {
    TORCH_CHECK(cache->dim() == 4 && cache->size(0) > 0 && cache->size(1) == 128 &&
        cache->size(2) == 1 && cache->size(3) == 656 && cache->scalar_type() == at::kChar,
        "KV caches must be int8 [blocks,128,1,656] byte views of FP8 E4M3 packed KV");
  }
  for (const auto* table : {&hbm_block_table, &dram_block_table}) {
    TORCH_CHECK(table->dim() == 2 && table->size(0) == b && table->size(1) > 0 &&
        table->size(1) <= 2048, "block tables must be [B,1..2048]");
  }
  for (const auto* ids : {&topk_slots, &topk_index}) {
    TORCH_CHECK(ids->dim() == 3 && ids->size(0) == t && ids->size(1) == 1 &&
        ids->size(2) == 2048, "MTP attention indices must be [T,1,2048]");
  }
  for (const auto* ids : {&copy_src_ids, &copy_dst_slots}) {
    TORCH_CHECK(ids->dim() == 3 && ids->size(0) == b && ids->size(1) == 1 &&
        ids->size(2) == 16384, "union copy indices must be [B,1,16384]");
  }
  TORCH_CHECK(per_query_miss_counts.dim() == 1 && per_query_miss_counts.size(0) == t &&
      actual_seq_lengths_query.dim() == 1 && actual_seq_lengths_query.size(0) == b &&
      resident_seq_lengths.dim() == 1 && resident_seq_lengths.size(0) == b,
      "misses [T], query ends and resident lengths [B] required");
  for (const auto* meta : {&hbm_block_table, &dram_block_table, &topk_slots, &topk_index, &per_query_miss_counts, &actual_seq_lengths_query, &resident_seq_lengths, &copy_src_ids, &copy_dst_slots, &copy_counts}) {
    TORCH_CHECK(meta->scalar_type() == at::kInt, "all metadata must be int32");
  }
  TORCH_CHECK(attention_out.dim() == 3 && attention_out.size(0) == t &&
      attention_out.size(1) == query.size(1) && attention_out.size(2) == 512 &&
      attention_out.scalar_type() == query.scalar_type(),
      "attention_out must be caller-owned [T,H,512] with query dtype");
  TORCH_CHECK(std::isfinite(scale_value), "scale_value must be finite");
  for (const auto* tensor : {&query, &hbm_kv_bytes, &dram_kv_bytes, &hbm_block_table, &dram_block_table, &topk_slots, &topk_index, &per_query_miss_counts, &actual_seq_lengths_query, &resident_seq_lengths, &copy_src_ids, &copy_dst_slots, &copy_counts, &attention_out}) {
    TORCH_CHECK(tensor->device() == query.device() && tensor->is_contiguous(),
        "fused_copy_sfa_c8 tensors must be contiguous on one device");
  }
}
inline void npu_dsa_a5_fused_copy_sfa_c8_out(const at::Tensor& query,
    const at::Tensor& hbm_kv_bytes,
    const at::Tensor& dram_kv_bytes,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    const at::Tensor& topk_slots,
    const at::Tensor& topk_index,
    const at::Tensor& per_query_miss_counts,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& resident_seq_lengths,
    const at::Tensor& copy_src_ids,
    const at::Tensor& copy_dst_slots,
    const at::Tensor& copy_counts,
    double scale_value, const at::Tensor& attention_out) {
  check_fused_copy_sfa_c8(query, hbm_kv_bytes, dram_kv_bytes, hbm_block_table, dram_block_table, topk_slots, topk_index, per_query_miss_counts, actual_seq_lengths_query, resident_seq_lengths, copy_src_ids, copy_dst_slots, copy_counts, scale_value, attention_out);
  TORCH_CHECK(query.device().is_privateuseone(), "fused_copy_sfa_c8 requires NPU");
  auto keepalive = std::make_tuple(query, hbm_kv_bytes, dram_kv_bytes, hbm_block_table, dram_block_table, topk_slots, topk_index, per_query_miss_counts, actual_seq_lengths_query, resident_seq_lengths, copy_src_ids, copy_dst_slots, copy_counts, attention_out);
  EXEC_DSA_COPYSFA_ORDERED_IMPL(aclnnVllmA5FusedCopySfaC8, keepalive, copysfa_detail::AllocateCopySfaWorkspace,
      query,
      hbm_kv_bytes,
      dram_kv_bytes,
      hbm_block_table,
      dram_block_table,
      topk_slots,
      topk_index,
      per_query_miss_counts,
      actual_seq_lengths_query,
      resident_seq_lengths,
      copy_src_ids,
      copy_dst_slots,
      copy_counts, scale_value, attention_out);
}
#undef EXEC_DSA_COPYSFA_ORDERED_IMPL
}  // namespace vllm_ascend
