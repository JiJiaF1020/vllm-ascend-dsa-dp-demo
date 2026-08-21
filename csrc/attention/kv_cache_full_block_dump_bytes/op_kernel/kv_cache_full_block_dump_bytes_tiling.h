/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef KV_CACHE_FULL_BLOCK_DUMP_BYTES_TILING_H
#define KV_CACHE_FULL_BLOCK_DUMP_BYTES_TILING_H

#include <cstdint>

struct KvCacheFullBlockDumpBytesTilingData {
    uint32_t usedCoreNum;
    uint32_t rowCount;
    uint32_t srcBlockNum;
    uint32_t dstBlockNum;
    uint32_t bytesPerBlock;
    uint32_t chunkBytes;
    uint32_t tasksPerRow;
    uint64_t taskCount;
};

#endif
