// Included inside BaseApi, after QSFAVectorService is fully defined.
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::EnqueueSourceAwareKv(
    uint32_t indexBase, int64_t first, uint32_t count, int64_t *writebacks)
{
    auto packed = stage0InQue.template AllocTensor<uint8_t>();
    const DataCopyExtParams params{1, 656, 0, 0, 0};
    const DataCopyPadExtParams<uint8_t> pad{true, 0, 16, 0};
    for (uint32_t j = 0; j < count; ++j) {
        // Mapping is constant across this contiguous AIV slice, not per token.
        const uint32_t index = indexBase + first + j;
        writebacks[j] = -1;
        if (index < queryMisses && index < prefetchedUntil && OwnsMiss(index)) {
            // Prefetch already resolved ownership and persisted the owner's row.
            // Do not reload source IDs/block tables just to read our workspace.
            DataCopyPad(packed[j * 672], prefetchGm[prefetchRow + index * 672], params, pad);
            continue;
        }
        const int32_t slot = SparseIndicesGm.GetValue(queryRow + index);
        if (index < queryMisses) {
            ASSERT(slot >= 0 && slot < 16384);
            const int32_t source = sourcesGm.GetValue(queryRow + index);
            const int64_t offset = fused_c8::RowOffsetInRequest(dramTableGm, copyTiling->dramMaxBlocks,
                copyTiling->dramPhysicalBlocks, dramTableRow, source);
            ASSERT(offset >= 0);
            DataCopyPad(packed[j * 672], dramGm[offset], params, pad);
            if ((earlierMissSlots[slot / 32] & (1U << (slot % 32))) == 0) {
                writebacks[j] = fused_c8::RowOffsetInRequest(blockTableGm,
                    copyTiling->baseParams.maxBlockNumPerBatch, copyTiling->hbmPhysicalBlocks, hbmTableRow, slot);
                ASSERT(writebacks[j] >= 0);
            }
        } else {
            const int64_t offset = fused_c8::RowOffsetInRequest(blockTableGm,
                copyTiling->baseParams.maxBlockNumPerBatch, copyTiling->hbmPhysicalBlocks, hbmTableRow, slot);
            ASSERT(offset >= 0);
            DataCopyPad(packed[j * 672], keyGm[offset], params, pad);
        }
    }
    stage0InQue.EnQue(packed);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::ProcessSourceAwareKv(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    const RunInfo &runInfo, ConstInfo &constInfo, uint32_t indexBase)
{
    PrepareOwners(runInfo.boIdx, queryIndex);
    int64_t writebacks[2][16];
    uint32_t lane = 0;
    EnqueueSourceAwareKv(indexBase, sparseS2Start, Min(static_cast<int64_t>(16), sparseS2End - sparseS2Start),
        writebacks[lane]);
    for (int64_t first = sparseS2Start; first < sparseS2End; first += 16) {
        const uint32_t count = Min(static_cast<int64_t>(16), sparseS2End - first);
        auto packed = stage0InQue.template DeQue<uint8_t>();
        const auto readEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        // Fence this batch BEFORE issuing the next MTE2 read: its writeback must
        // not wait for the following DRAM batch. Queue events also order MTE2->V.
        SetFlag<HardEvent::MTE2_MTE3>(readEvent);
        WaitFlag<HardEvent::MTE2_MTE3>(readEvent);
        if (first + 16 < sparseS2End) {
            EnqueueSourceAwareKv(indexBase, first + 16, Min(static_cast<int64_t>(16), sparseS2End - first - 16),
                writebacks[lane ^ 1U]);
        }
        bool wrote = false;
        // MTE3 persistence and V dequantization both consume the same DRAM read.
        // Their completion is joined before this UB buffer can be reused.
        for (uint32_t j = 0; j < count; ++j) {
            if (writebacks[lane][j] < 0) continue;
            // Only coalesce adjacent owner rows in this bank. Prefetched rows,
            // earlier-query owners and hits retain -1 and must not be persisted.
            const auto plan = fused_c8::PlanMissWritebackPair(writebacks[lane][j],
                j + 1 < count ? writebacks[lane][j + 1] : -1);
            DataCopyPad(keyGm[writebacks[lane][j]], packed[j * 672],
                DataCopyExtParams{plan.blockCount, 656, 0, plan.dstStride, 0});
            j += plan.blockCount - 1;
            wrote = true;
        }
        auto dequant = stage0OutQue.template AllocTensor<Q_T>();
        DequantKv(dequant, packed, count, constInfo);
        if (wrote) {
            const auto event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(event);
            WaitFlag<HardEvent::MTE3_MTE2>(event);
        }
        stage0InQue.FreeTensor(packed);
        stage0OutQue.EnQue(dequant);
        dequant = stage0OutQue.template DeQue<Q_T>();
        if constexpr (IS_SPLIT_G) CopyOutKvUb2Gm(v0ResGm, dequant, count, first, runInfo, constInfo);
        else CopyOutKvUb2L1(outputL1, dequant, count, first, runInfo, constInfo);
        stage0OutQue.FreeTensor(dequant);
        lane ^= 1U;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::SubmitMissPrefetch(
    const RunInfo &runInfo, ConstInfo &constInfo)
{
    if (!sourceAware) return;
    const uint32_t query = queryIndex;
    const uint32_t miss = queryMisses;
    if (miss == 0 || miss > 512) return;
    // Only prefetch during the hit chunks, before this query reaches its miss chunk.
    const uint32_t logicalChunk = runInfo.s2LoopCount / 4;
    const uint32_t missChunk = 3 - runInfo.boIdx % 3;
    if (logicalChunk >= missChunk) return;
    PrepareOwners(runInfo.boIdx, query);
    auto io = missPrefetchBuf.template Get<uint8_t>();
    uint32_t scan = prefetchedUntil;
    while (scan < miss && pendingRows < 4) {
        const uint32_t index = scan++;
        if (!OwnsMiss(index)) continue;
        const uint64_t p = queryRow + index;
        const int32_t source = sourcesGm.GetValue(p), slot = SparseIndicesGm.GetValue(p);
        const auto offset = fused_c8::RowOffsetInRequest(dramTableGm, copyTiling->dramMaxBlocks,
            copyTiling->dramPhysicalBlocks, dramTableRow, source);
        ASSERT(offset >= 0 && slot >= 0 && slot < 16384);
        pendingIndices[pendingRows] = index;
        pendingDestinations[pendingRows] = -1;
        if ((earlierMissSlots[slot / 32] & (1U << (slot % 32))) == 0) {
            pendingDestinations[pendingRows] = fused_c8::RowOffsetInRequest(blockTableGm,
                copyTiling->baseParams.maxBlockNumPerBatch, copyTiling->hbmPhysicalBlocks, hbmTableRow, slot);
            ASSERT(pendingDestinations[pendingRows] >= 0);
        }
        DataCopyPad(io[pendingRows * 672], dramGm[offset], DataCopyExtParams{1, 656, 0, 0, 0},
            DataCopyPadExtParams<uint8_t>{true, 0, 16, 0});
        ++pendingRows;
    }
    pendingEnd = scan;
    if (pendingRows) {
        missReadEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(missReadEvent);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::CompleteMissPrefetch()
{
    if (!pendingRows) return;
    WaitFlag<HardEvent::MTE2_MTE3>(missReadEvent);
    auto io = missPrefetchBuf.template Get<uint8_t>();
    for (uint32_t j = 0; j < pendingRows; ++j) {
        const DataCopyExtParams params{1, 656, 0, 0, 0};
        DataCopyPad(prefetchGm[prefetchRow + pendingIndices[j] * 672],
            io[j * 672], params);
        if (pendingDestinations[j] >= 0) DataCopyPad(keyGm[pendingDestinations[j]], io[j * 672], params);
    }
    const auto event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(event);
    WaitFlag<HardEvent::MTE3_MTE2>(event);
    prefetchedUntil = pendingEnd;
    pendingRows = 0;
}
