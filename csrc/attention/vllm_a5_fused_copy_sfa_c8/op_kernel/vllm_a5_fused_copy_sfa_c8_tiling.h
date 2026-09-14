#ifndef VLLM_A5_FUSED_COPY_SFA_C8_TILING_H
#define VLLM_A5_FUSED_COPY_SFA_C8_TILING_H
#include <cstdint>

// Only the fields consumed by the arch35 SCFA/MLA implementation are needed.
struct FusedC8BaseParams {
    uint32_t batchSize, seqSize, qSeqSize, nNumOfQInOneGroup;
    uint32_t sparseBlockCount, sparseMode, outputLayout, blockSize;
    uint32_t maxBlockNumPerBatch, isActualLenDimsNull, isActualLenDimsKVNull;
    uint32_t dSizeVInput;
    float scaleValue;
};
struct FusedC8SingleCoreParams { uint32_t usedCoreNum; };
struct VllmA5FusedCopySfaC8TilingData {
    FusedC8BaseParams baseParams;
    FusedC8SingleCoreParams singleCoreParams;
    uint32_t dramMaxBlocks, hbmPhysicalBlocks, dramPhysicalBlocks;
};
#endif
