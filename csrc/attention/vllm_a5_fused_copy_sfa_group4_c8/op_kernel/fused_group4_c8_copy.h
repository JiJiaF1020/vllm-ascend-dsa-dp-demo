// Group4-only copy phase: two 2624-byte buffers; all four lanes are durable
// before the shared SFA math consumes the owner's zero-copy virtual PA view.
#ifndef FUSED_GROUP4_C8_COPY_H
#define FUSED_GROUP4_C8_COPY_H
#include "kernel_operator.h"
#include "vllm_a5_fused_copy_sfa_group4_c8_tiling.h"

namespace fused_group4_c8 {
using namespace AscendC;
constexpr uint32_t ROW_BYTES = 2624, UB_ROW_BYTES = 2624, COPY_CAP = 16384;

__aicore__ inline int64_t RowOffsetInRequest(GlobalTensor<int32_t> &table, uint32_t width,
    uint32_t blocks, uint64_t tableRow, int32_t token)
{
    if (token < 0 || static_cast<uint32_t>(token) / 128 >= width) return -1;
    const int32_t block = table.GetValue(tableRow + token / 128);
    if (block < 0 || static_cast<uint32_t>(block) >= blocks) return -1;
    return (static_cast<int64_t>(block) * 128 + token % 128) * ROW_BYTES;
}

__aicore__ inline int64_t RowOffset(GlobalTensor<int32_t> &table, uint32_t width,
    uint32_t blocks, uint32_t batch, int32_t token)
{
    return RowOffsetInRequest(table, width, blocks, static_cast<uint64_t>(batch) * width, token);
}

__aicore__ inline void UnionCopy(TPipe &pipe, GM_ADDR hbm, GM_ADDR dram,
    GM_ADDR ht, GM_ADDR dt, GM_ADDR src, GM_ADDR dst, GM_ADDR counts,
    const VllmA5FusedCopySfaGroup4C8TilingData &td)
{
    // Reserve the SAME UB prefix on AIC and AIV: SFA's Fixpipe buffers are
    // independently allocated on both sides and must have identical offsets.
    // Keep it until kernel exit. Resetting TPipe here would also reset the
    // event allocator before SFA initializes its L0/UB buffers.
    TBuf<TPosition::VECIN> storage;
    pipe.InitBuffer(storage, 2 * UB_ROW_BYTES);
    if ASCEND_IS_AIV {
        GlobalTensor<int8_t> hg, dg;
        GlobalTensor<int32_t> htable, dtable, sources, dests, sizes;
        hg.SetGlobalBuffer((__gm__ int8_t *)hbm); dg.SetGlobalBuffer((__gm__ int8_t *)dram);
        htable.SetGlobalBuffer((__gm__ int32_t *)ht); dtable.SetGlobalBuffer((__gm__ int32_t *)dt);
        sources.SetGlobalBuffer((__gm__ int32_t *)src); dests.SetGlobalBuffer((__gm__ int32_t *)dst);
        sizes.SetGlobalBuffer((__gm__ int32_t *)counts);
        // Explicit events leave no copy-queue mutexes for the SFA phase.
        auto buffers = storage.Get<int8_t>();
        event_t readReady[2], writeDone[2];
        bool wrote[2] = {false, false};
        int64_t pendingDst[2];
        uint32_t issued = 0, finished = 0;
        for (uint32_t slot = 0; slot < 2; ++slot) {
            readReady[slot] = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_MTE3>());
            writeDone[slot] = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_MTE2>());
        }
        const uint32_t lanes = td.singleCoreParams.usedCoreNum * 2;
        // Interleave pairs across all AIVs; first-fill rows may be much larger than TopK.
        for (uint32_t b = 0; b < td.baseParams.batchSize; ++b) {
            const int32_t n = sizes.GetValue(b);
            for (uint32_t i = GetBlockIdx(); i < n; i += lanes) {
                const uint64_t p = static_cast<uint64_t>(b) * COPY_CAP + i;
                const auto so = RowOffset(dtable, td.dramMaxBlocks, td.dramPhysicalBlocks, b, sources.GetValue(p));
                const auto dest = RowOffset(htable, td.baseParams.maxBlockNumPerBatch / 4,
                    td.hbmPhysicalBlocks, b, dests.GetValue(p));
                ASSERT(so >= 0 && dest >= 0);
                const uint32_t slot = issued % 2;
                if (wrote[slot]) {
                    WaitFlag<HardEvent::MTE3_MTE2>(writeDone[slot]);
                    wrote[slot] = false;
                }
                auto buf = buffers[slot * UB_ROW_BYTES];
                DataCopyPad(buf, dg[so], DataCopyExtParams{1, ROW_BYTES, 0, 0, 0},
                    DataCopyPadExtParams<int8_t>{false, 0, 0, 0});
                pendingDst[slot] = dest;
                SetFlag<HardEvent::MTE2_MTE3>(readReady[slot]);
                ++issued;
                if (issued - finished == 2) {
                    const uint32_t out = finished % 2;
                    WaitFlag<HardEvent::MTE2_MTE3>(readReady[out]);
                    DataCopyPad(hg[pendingDst[out]], buffers[out * UB_ROW_BYTES],
                        DataCopyExtParams{1, ROW_BYTES, 0, 0, 0});
                    SetFlag<HardEvent::MTE3_MTE2>(writeDone[out]);
                    wrote[out] = true;
                    ++finished;
                }
            }
        }
        while (finished < issued) {
            const uint32_t out = finished % 2;
            WaitFlag<HardEvent::MTE2_MTE3>(readReady[out]);
            DataCopyPad(hg[pendingDst[out]], buffers[out * UB_ROW_BYTES],
                DataCopyExtParams{1, ROW_BYTES, 0, 0, 0});
            SetFlag<HardEvent::MTE3_MTE2>(writeDone[out]);
            wrote[out] = true;
            ++finished;
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            if (wrote[slot]) {
                WaitFlag<HardEvent::MTE3_MTE2>(writeDone[slot]);
            }
            pipe.ReleaseEventID<HardEvent::MTE2_MTE3>(readReady[slot]);
            pipe.ReleaseEventID<HardEvent::MTE3_MTE2>(writeDone[slot]);
        }
    }
    // Every launched AIC and AIV participates, including lanes with no pairs.
    // Complete ALL HBM writes before ANY query can gather another lane's data.
    // SyncAll owns its reserved IDs; SFA keeps private IDs 0..8 exclusively.
    SyncAll<false>();
}
}
#endif
