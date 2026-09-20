#ifndef VLLM_A5_KVCACHE_SCATTER_COPY_GROUP4_C8_TILING_H
#define VLLM_A5_KVCACHE_SCATTER_COPY_GROUP4_C8_TILING_H

#include <cstdint>

struct VllmA5KvcacheScatterCopyGroup4C8TilingData {
    uint32_t usedCoreNum;
    uint32_t batchSize;
    uint32_t copyCap;
    uint32_t hbmMaxBlockNum;
    uint32_t dramMaxBlockNum;
    uint32_t hbmPhysicalBlockCount;
    uint32_t dramPhysicalBlockCount;
    uint32_t packedRowBytes;
    uint32_t blockSize;
    uint32_t blockRowBytes;
    uint64_t totalPairSlots;
};

#endif
