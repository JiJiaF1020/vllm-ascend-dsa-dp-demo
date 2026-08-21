/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "../op_kernel/kv_cache_full_block_dump_bytes_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t SRC_CACHE = 0;
constexpr size_t DST_CACHE = 1;
constexpr size_t SRC_BLOCK_IDS = 2;
constexpr size_t DST_BLOCK_IDS = 3;
constexpr uint32_t COPY_CHUNK_BYTES = 32U * 1024U;
}  // namespace

namespace optiling {
static ge::graphStatus TilingKvCacheFullBlockDumpBytes(
    gert::TilingContext* context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = SRC_CACHE; index <= DST_BLOCK_IDS; ++index) {
        if (context->GetInputShape(index) == nullptr ||
            context->GetInputDesc(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    if (context->GetInputDesc(SRC_CACHE)->GetDataType() != ge::DT_INT8 ||
        context->GetInputDesc(DST_CACHE)->GetDataType() != ge::DT_INT8 ||
        context->GetInputDesc(SRC_BLOCK_IDS)->GetDataType() != ge::DT_INT32 ||
        context->GetInputDesc(DST_BLOCK_IDS)->GetDataType() != ge::DT_INT32) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape src =
        context->GetInputShape(SRC_CACHE)->GetStorageShape();
    const gert::Shape dst =
        context->GetInputShape(DST_CACHE)->GetStorageShape();
    const gert::Shape srcIds =
        context->GetInputShape(SRC_BLOCK_IDS)->GetStorageShape();
    const gert::Shape dstIds =
        context->GetInputShape(DST_BLOCK_IDS)->GetStorageShape();
    if (src.GetDimNum() != 2 || dst.GetDimNum() != 2 ||
        src.GetDim(0) <= 0 || dst.GetDim(0) <= 0 ||
        src.GetDim(1) <= 0 || src.GetDim(1) != dst.GetDim(1) ||
        srcIds.GetDimNum() != 1 || dstIds.GetDimNum() != 1 ||
        srcIds.GetDim(0) <= 0 || srcIds.GetDim(0) != dstIds.GetDim(0)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t uint32Max = static_cast<int64_t>(
        std::numeric_limits<uint32_t>::max());
    if (src.GetDim(0) > uint32Max || dst.GetDim(0) > uint32Max ||
        src.GetDim(1) > uint32Max || srcIds.GetDim(0) > uint32Max) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aivCount = platform.GetCoreNumAiv();
    if (aivCount == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t bytesPerBlock64 = static_cast<uint64_t>(src.GetDim(1));
    const uint32_t bytesPerBlock = static_cast<uint32_t>(bytesPerBlock64);
    const uint32_t tasksPerRow =
        (bytesPerBlock + COPY_CHUNK_BYTES - 1) / COPY_CHUNK_BYTES;
    const uint32_t rowCount = static_cast<uint32_t>(srcIds.GetDim(0));
    const uint64_t taskCount =
        static_cast<uint64_t>(rowCount) * tasksPerRow;
    const uint32_t usedCoreNum = static_cast<uint32_t>(
        std::min<uint64_t>(taskCount, aivCount));

    auto* tiling = context->GetTilingData<
        KvCacheFullBlockDumpBytesTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->usedCoreNum = usedCoreNum;
    tiling->rowCount = rowCount;
    tiling->srcBlockNum = static_cast<uint32_t>(src.GetDim(0));
    tiling->dstBlockNum = static_cast<uint32_t>(dst.GetDim(0));
    tiling->bytesPerBlock = bytesPerBlock;
    tiling->chunkBytes = COPY_CHUNK_BYTES;
    tiling->tasksPerRow = tasksPerRow;
    tiling->taskCount = taskCount;
    context->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

struct KvCacheFullBlockDumpBytesCompileInfo {};

static ge::graphStatus TilingParseKvCacheFullBlockDumpBytes(
    gert::TilingParseContext*)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KvCacheFullBlockDumpBytes)
    .Tiling(TilingKvCacheFullBlockDumpBytes)
    .TilingParse<KvCacheFullBlockDumpBytesCompileInfo>(
        TilingParseKvCacheFullBlockDumpBytes);
}  // namespace optiling
