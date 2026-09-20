// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../vllm_a5_fused_copy_sfa_c8/vllm_a5_fused_copy_sfa_c8_torch_adpt.h"

namespace vllm_ascend {
// Reuse the original ordered-launch helpers and per-stream workspace cache.
// Keep a separately named macro: the original operator/adapter stays unchanged.
#define EXEC_DSA_GROUP4_COPYSFA_ORDERED_IMPL(aclnn_api, tensor_keepalive, allocate_workspace, ...)                               \
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


inline void check_fused_copy_sfa_group4_c8(const at::Tensor& query,
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
    const at::Tensor& attention_slots,
    const at::Tensor& attention_block_table,
    double scale_value, const at::Tensor& attention_out) {

  for (const auto* cache : {&hbm_kv_bytes, &dram_kv_bytes}) {
    TORCH_CHECK(cache->dim() == 4 && cache->size(0) > 0 &&
        cache->size(0) <= std::numeric_limits<int32_t>::max() / 4 &&
        cache->size(1) == 128 && cache->size(2) == 1 && cache->size(3) == 2624 &&
        cache->is_contiguous(), "Group4 CopySFA requires contiguous [blocks,128,1,2624]");
  }
  // Share the existing dtype/shape/device checks on zero-copy PA aliases.
  check_fused_copy_sfa_c8(query, hbm_kv_bytes.view({-1,128,1,656}),
      dram_kv_bytes.view({-1,128,1,656}), hbm_block_table, dram_block_table,
      topk_slots, topk_index, per_query_miss_counts, actual_seq_lengths_query,
      resident_seq_lengths, copy_src_ids, copy_dst_slots, copy_counts, scale_value, attention_out);
  TORCH_CHECK(attention_slots.sizes() == topk_slots.sizes() &&
      attention_block_table.dim() == 2 &&
      attention_block_table.size(0) == hbm_block_table.size(0) &&
      attention_block_table.size(1) == 4 * hbm_block_table.size(1),
      "Group4 CopySFA requires lane-0 slots [T,1,2048] and virtual block table [B,4*M]");
  for (const auto* tensor : {&attention_slots, &attention_block_table}) {
    TORCH_CHECK(tensor->scalar_type() == at::kInt && tensor->is_contiguous() &&
        tensor->device() == query.device(), "virtual PA metadata must be contiguous int32 on the query device");
  }
}

inline void npu_dsa_a5_fused_copy_sfa_group4_c8_out(const at::Tensor& query,
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
    const at::Tensor& attention_slots,
    const at::Tensor& attention_block_table,
    double scale_value, const at::Tensor& attention_out) {
  check_fused_copy_sfa_group4_c8(query, hbm_kv_bytes, dram_kv_bytes, hbm_block_table, dram_block_table, topk_slots, topk_index, per_query_miss_counts, actual_seq_lengths_query, resident_seq_lengths, copy_src_ids, copy_dst_slots, copy_counts, attention_slots, attention_block_table, scale_value, attention_out);
  TORCH_CHECK(query.device().is_privateuseone(), "fused_copy_sfa_group4_c8 requires NPU");
  auto keepalive = std::make_tuple(query, hbm_kv_bytes, dram_kv_bytes, hbm_block_table, dram_block_table, topk_slots, topk_index, per_query_miss_counts, actual_seq_lengths_query, resident_seq_lengths, copy_src_ids, copy_dst_slots, copy_counts, attention_slots, attention_block_table, attention_out);
  EXEC_DSA_GROUP4_COPYSFA_ORDERED_IMPL(aclnnVllmA5FusedCopySfaGroup4C8, keepalive, copysfa_detail::AllocateCopySfaWorkspace,
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
      copy_counts, attention_slots, attention_block_table, scale_value, attention_out);
}

#undef EXEC_DSA_GROUP4_COPYSFA_ORDERED_IMPL
} // namespace vllm_ascend
