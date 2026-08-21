/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef KV_CACHE_FULL_BLOCK_DUMP_BYTES_TORCH_ADPT_H
#define KV_CACHE_FULL_BLOCK_DUMP_BYTES_TORCH_ADPT_H

namespace vllm_ascend {

inline void npu_kv_cache_full_block_dump_bytes(
    const at::Tensor& src_cache,
    const at::Tensor& dst_cache,
    const at::Tensor& src_block_ids,
    const at::Tensor& dst_block_ids)
{
    TORCH_CHECK(src_cache.device().is_privateuseone() &&
                    dst_cache.device().is_privateuseone(),
                "byte-block dump caches must be on NPU");
    TORCH_CHECK(src_cache.device() == dst_cache.device() &&
                    src_block_ids.device() == src_cache.device() &&
                    dst_block_ids.device() == src_cache.device(),
                "byte-block dump tensors must share one NPU device");
    TORCH_CHECK(src_cache.scalar_type() == at::ScalarType::Char &&
                    dst_cache.scalar_type() == at::ScalarType::Char,
                "byte-block dump cache views must be int8");
    TORCH_CHECK(src_cache.dim() == 2 && dst_cache.dim() == 2 &&
                    src_cache.size(1) == dst_cache.size(1) &&
                    src_cache.size(1) > 0,
                "byte-block dump caches must be [blocks, bytes_per_block] "
                "with matching positive widths");
    TORCH_CHECK(src_cache.is_contiguous() && dst_cache.is_contiguous(),
                "byte-block dump caches must be contiguous");
    TORCH_CHECK(src_block_ids.dim() == 1 && dst_block_ids.dim() == 1 &&
                    src_block_ids.numel() == dst_block_ids.numel(),
                "byte-block dump block-id tensors must have matching rows");
    TORCH_CHECK(src_block_ids.scalar_type() == at::ScalarType::Int &&
                    dst_block_ids.scalar_type() == at::ScalarType::Int &&
                    src_block_ids.is_contiguous() &&
                    dst_block_ids.is_contiguous(),
                "byte-block dump block ids must be contiguous int32 tensors");
    TORCH_CHECK(src_cache.data_ptr() != dst_cache.data_ptr(),
                "byte-block dump source and destination must not alias");

    EXEC_NPU_CMD(aclnnKvCacheFullBlockDumpBytes,
                 src_cache,
                 dst_cache,
                 src_block_ids,
                 dst_block_ids);
}

}  // namespace vllm_ascend

#endif
