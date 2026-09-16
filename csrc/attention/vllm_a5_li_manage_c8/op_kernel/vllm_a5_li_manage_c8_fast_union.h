/** Stage 2-4: ordered union, victim selection, cache update and slot repair. */

#ifndef VLLM_A5_LI_MANAGE_C8_FAST_UNION_H
#define VLLM_A5_LI_MANAGE_C8_FAST_UNION_H

#include "kernel_operator.h"
#include "vllm_a5_li_manage_c8_fast_union_vf.h"
#include "vllm_a5_li_manage_c8_fast_victim_vf.h"
#include "vllm_a5_li_manage_c8_fast_workspace.h"
#include "vllm_a5_li_manage_c8_victim_scan.h"
#include "lim_simt_postprocess.h"

namespace vllm_a5_li_manage_c8_fast {
using namespace AscendC;

constexpr uint32_t UNION_ROUTES = 4U;
constexpr uint32_t UNION_TOPK = 2048U;
constexpr int32_t UNION_ROW_MODE_PAD = 0;
constexpr int32_t UNION_ROW_MODE_DENSE = 1;
constexpr int32_t UNION_ROW_MODE_SPARSE = 2;
constexpr uint32_t UNION_CAPACITY = UNION_ROUTES * UNION_TOPK;
constexpr uint32_t UNION_PAIR_WORDS = UNION_TOPK * 2U;
constexpr uint32_t UNION_SOURCE_MASK = (1U << 18U) - 1U;
constexpr uint32_t B32_VECTOR_ELEMENTS = 64U;
constexpr uint32_t VICTIM_SCAN_CHUNK = 2048U;
constexpr uint32_t VICTIM_SLOT_SHIFT = 18U;
constexpr uint32_t VICTIM_SLOT_MASK = (1U << 14U) - 1U;

template <HardEvent event>
__aicore__ inline void UnionSync(HardEvent value)
{
    event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(value));
    SetFlag<event>(id);
    WaitFlag<event>(id);
}

class OrderedMissUnion {
public:
    __aicore__ inline void Init(
        GM_ADDR routePairs, GM_ADDR routeThresholds, GM_ADDR routeCounts,
        GM_ADDR scoreWorkspace, GM_ADDR actualSeqLengthsQuery,
        GM_ADDR offloadedLens,
        GM_ADDR finalSeqLengthsKv, GM_ADDR rowModes,
        GM_ADDR reqPoolEntries, GM_ADDR cacheSlots,
        GM_ADDR unionSources, GM_ADDR unionDestinations,
        GM_ADDR unionCounts, GM_ADDR topkSlots, GM_ADDR topkIndex,
        GM_ADDR perQueryMissCounts, GM_ADDR residentSeqLengths,
        uint32_t tokenCapacity, uint32_t outputCapacity,
        uint32_t scoreRowStride, uint32_t batchSize,
        uint32_t poolSize, uint32_t totalQueryRows,
        uint32_t maxCandidateLen, TPipe *pipe)
    {
        routePairsGm_.SetGlobalBuffer((__gm__ int32_t *)routePairs);
        routeThresholdsGm_.SetGlobalBuffer(
            (__gm__ uint16_t *)routeThresholds);
        routeCountsGm_.SetGlobalBuffer((__gm__ int32_t *)routeCounts);
        scoreWorkspaceGm_.SetGlobalBuffer((__gm__ uint16_t *)scoreWorkspace);
        actualSeqLengthsQueryGm_.SetGlobalBuffer(
            (__gm__ int32_t *)actualSeqLengthsQuery);
        offloadedLensGm_.SetGlobalBuffer((__gm__ int32_t *)offloadedLens);
        finalSeqLengthsKvGm_.SetGlobalBuffer(
            (__gm__ int32_t *)finalSeqLengthsKv);
        rowModesGm_.SetGlobalBuffer((__gm__ int32_t *)rowModes);
        reqPoolEntriesGm_.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries);
        cacheSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)cacheSlots);
        unionSourcesGm_.SetGlobalBuffer((__gm__ int32_t *)unionSources);
        unionDestinationsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)unionDestinations);
        unionCountsGm_.SetGlobalBuffer((__gm__ int32_t *)unionCounts);
        topkSlotsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)topkSlots);
        topkIndexGm_.SetGlobalBuffer(
            (__gm__ int32_t *)topkIndex);
        perQueryMissCountsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)perQueryMissCounts);
        residentSeqLengthsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)residentSeqLengths);
        sourceCapacity_ = tokenCapacity;
        poolStride_ = tokenCapacity + 1U;
        outputCapacity_ = outputCapacity;
        scoreRowStride_ = scoreRowStride;
        batchSize_ = batchSize;
        poolSize_ = poolSize;
        totalQueryRows_ = totalQueryRows;
        maxCandidateLen_ = maxCandidateLen;
        pipe->InitBuffer(pairInputBuf_,
                         UNION_CAPACITY * 2U * sizeof(float));
        pipe->InitBuffer(pairOutputBuf_,
                         UNION_CAPACITY * 2U * sizeof(float));
        pipe->InitBuffer(sourceBuf_,
                         UNION_CAPACITY * sizeof(int32_t));
        // Three independent aligned rows let MTE3 publish per-query counts,
        // union count and resident length together without reusing scalar UB.
        pipe->InitBuffer(countBuf_, 3U * 32U);
        pipe->InitBuffer(
            thresholdBuf_,
            UNION_ROUTES *
                vllm_a5_li_manage_c8_fast_workspace::THRESHOLD_STRIDE *
                sizeof(uint16_t));
    }

    __aicore__ inline void Process(uint32_t first, uint32_t stride)
    {
        for (uint32_t batch = first; batch < batchSize_; batch += stride) {
            ProcessRequest(batch);
        }
    }

private:
#if LIM_SIMT_REQUEST_META_CACHE
    struct SparseRequestMetadata {
        uint32_t candidate;
        uint32_t finalLen;
        uint32_t poolRow;
        uint32_t budget;
    };
#endif

    __aicore__ inline uint32_t Deduplicate(
        LocalTensor<float> merged, uint32_t total,
        LocalTensor<int32_t> output)
    {
        VllmA5LiManageC8FastUnionVF::DeduplicateSortedPairs(
            (__ubuf__ uint32_t *)output.GetPhyAddr(),
            (__ubuf__ uint32_t *)merged.GetPhyAddr(), total);
        const uint32_t count = static_cast<uint32_t>(
            GetSpr<SpecialPurposeReg::AR>() / sizeof(uint32_t));
        PipeBarrier<PIPE_V>();
        // The following victim phase reuses the merged-pair UB allocation.
        UnionSync<HardEvent::V_S>(HardEvent::V_S);
        return count;
    }

    __aicore__ inline uint32_t ReadBudget(
        uint32_t batch, uint32_t poolRow)
    {
        (void)batch;
        return static_cast<uint32_t>(cacheSlotsGm_.GetValue(
            static_cast<uint64_t>(poolRow) * poolStride_ +
            sourceCapacity_));
    }

    __aicore__ inline bool QueryRange(
        uint32_t batch, uint32_t &queryStart,
        uint32_t &queryEnd, uint32_t &routeCount)
    {
        const int32_t end = actualSeqLengthsQueryGm_.GetValue(batch);
        const int32_t start = batch == 0U
            ? 0 : actualSeqLengthsQueryGm_.GetValue(batch - 1U);
        if (start < 0 || end <= start ||
            end > static_cast<int32_t>(totalQueryRows_) ||
            end - start > static_cast<int32_t>(UNION_ROUTES)) {
            return false;
        }
        queryStart = static_cast<uint32_t>(start);
        queryEnd = static_cast<uint32_t>(end);
        routeCount = queryEnd - queryStart;
        return true;
    }

    __aicore__ inline void LoadThresholds(
        uint32_t batch, LocalTensor<uint16_t> local,
        uint32_t routeCount, uint16_t values[UNION_ROUTES])
    {
        constexpr uint32_t STRIDE =
            vllm_a5_li_manage_c8_fast_workspace::THRESHOLD_STRIDE;
        const uint64_t routeBase =
            static_cast<uint64_t>(batch) * UNION_ROUTES;
        for (uint32_t route = 0U; route < routeCount; ++route) {
            DataCopyPad(
                local[route * STRIDE],
                routeThresholdsGm_[(routeBase + route) * STRIDE],
                {1, STRIDE * static_cast<uint32_t>(sizeof(uint16_t)),
                 0, 0, 0},
                {false, 0, 0, 0});
        }
        UnionSync<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        for (uint32_t route = 0U; route < routeCount; ++route) {
            values[route] = local.GetValue(route * STRIDE);
        }
        // The victim VF tests score < threshold for all four lanes.  An
        // absent speculative route must therefore be neutral, not a reason
        // to reject an otherwise safe victim.  Stage A does not publish an
        // unused score row; CompactSafeVictims supplies a zero row and this
        // maximal threshold makes that lane conservatively pass.  A real
        // score equal to 0xffff can only make the fast scan miss a victim;
        // AppendExactVictims still completes the request exactly.
        for (uint32_t route = routeCount; route < UNION_ROUTES; ++route) {
            values[route] = 0xffffU;
        }
    }

    __aicore__ inline uint32_t CompactSafeVictims(
        uint32_t batch, uint32_t candidate, uint32_t poolRow,
        uint32_t budget, uint32_t required, uint32_t routeCount,
        uint32_t scanSeed,
        const uint16_t thresholds[UNION_ROUTES],
        LocalTensor<int32_t> destinations,
        LocalTensor<int32_t> victimSources)
    {
        LocalTensor<uint8_t> scratch = pairInputBuf_.Get<uint8_t>();
        LocalTensor<uint16_t> score0 =
            scratch.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> score1 = score0[VICTIM_SCAN_CHUNK];
        LocalTensor<uint16_t> score2 =
            score0[VICTIM_SCAN_CHUNK * 2U];
        LocalTensor<uint16_t> score3 =
            score0[VICTIM_SCAN_CHUNK * 3U];
        LocalTensor<int32_t> slots32 =
            scratch[VICTIM_SCAN_CHUNK * UNION_ROUTES *
                    sizeof(uint16_t)].ReinterpretCast<int32_t>();
        LocalTensor<int16_t> slots16 =
            scratch[VICTIM_SCAN_CHUNK *
                    (UNION_ROUTES * sizeof(uint16_t) + sizeof(int32_t))]
                .ReinterpretCast<int16_t>();
        LocalTensor<uint32_t> compact =
            scratch[VICTIM_SCAN_CHUNK *
                    (UNION_ROUTES * sizeof(uint16_t) + sizeof(int32_t) +
                     sizeof(int16_t))].ReinterpretCast<uint32_t>();

        const uint32_t chunks =
            (candidate + VICTIM_SCAN_CHUNK - 1U) / VICTIM_SCAN_CHUNK;
        const uint32_t firstChunk = scanSeed % chunks;
        const uint64_t requestScoreBase =
            static_cast<uint64_t>(batch) * UNION_ROUTES * scoreRowStride_;
        const uint64_t cacheBase =
            static_cast<uint64_t>(poolRow) * poolStride_;
        uint32_t written = 0U;
        for (uint32_t visit = 0U;
             visit < chunks && written < required; ++visit) {
            const uint32_t chunk = (firstChunk + visit) % chunks;
            const uint32_t chunkBase = chunk * VICTIM_SCAN_CHUNK;
            const uint32_t chunkLen =
                chunkBase + VICTIM_SCAN_CHUNK > candidate
                    ? candidate - chunkBase
                    : VICTIM_SCAN_CHUNK;
            const uint32_t alignedLen =
                (chunkLen + B32_VECTOR_ELEMENTS - 1U) /
                B32_VECTOR_ELEMENTS * B32_VECTOR_ELEMENTS;
            const DataCopyExtParams scoreCopy{
                1, chunkLen * static_cast<uint32_t>(sizeof(uint16_t)),
                0, 0, 0};
            const DataCopyPadExtParams<uint16_t> scorePad{
                true, 0,
                static_cast<uint8_t>(alignedLen - chunkLen), 0U};
            if (routeCount > 0U) {
                DataCopyPad(
                    score0,
                    scoreWorkspaceGm_[requestScoreBase + chunkBase],
                    scoreCopy, scorePad);
            } else {
                Duplicate(score0, static_cast<uint16_t>(0), alignedLen);
            }
            if (routeCount > 1U) {
                DataCopyPad(
                    score1,
                    scoreWorkspaceGm_[requestScoreBase + scoreRowStride_ +
                                      chunkBase],
                    scoreCopy, scorePad);
            } else {
                Duplicate(score1, static_cast<uint16_t>(0), alignedLen);
            }
            if (routeCount > 2U) {
                DataCopyPad(
                    score2,
                    scoreWorkspaceGm_[requestScoreBase +
                                      scoreRowStride_ * 2U + chunkBase],
                    scoreCopy, scorePad);
            } else {
                Duplicate(score2, static_cast<uint16_t>(0), alignedLen);
            }
            if (routeCount > 3U) {
                DataCopyPad(
                    score3,
                    scoreWorkspaceGm_[requestScoreBase +
                                      scoreRowStride_ * 3U + chunkBase],
                    scoreCopy, scorePad);
            } else {
                Duplicate(score3, static_cast<uint16_t>(0), alignedLen);
            }
            DataCopyPad(
                slots32, cacheSlotsGm_[cacheBase + chunkBase],
                {1, chunkLen * static_cast<uint32_t>(sizeof(int32_t)),
                 0, 0, 0},
                {false, 0, 0, 0});
            UnionSync<HardEvent::MTE2_V>(HardEvent::MTE2_V);

            Cast(slots16, slots32, RoundMode::CAST_NONE, chunkLen);
            if (alignedLen > chunkLen) {
                Duplicate(
                    slots16[chunkLen], static_cast<int16_t>(-1),
                    alignedLen - chunkLen);
            }
            PipeBarrier<PIPE_V>();
            VllmA5LiManageC8FastVictimVF::CompactEligiblePayloads(
                (__ubuf__ uint32_t *)compact.GetPhyAddr(),
                (__ubuf__ uint16_t *)score0.GetPhyAddr(),
                (__ubuf__ uint16_t *)score1.GetPhyAddr(),
                (__ubuf__ uint16_t *)score2.GetPhyAddr(),
                (__ubuf__ uint16_t *)score3.GetPhyAddr(),
                (__ubuf__ uint16_t *)slots16.GetPhyAddr(),
                thresholds[0], thresholds[1], thresholds[2],
                thresholds[3], static_cast<uint16_t>(budget),
                chunkBase, alignedLen / B32_VECTOR_ELEMENTS);
            const uint32_t compactCount = static_cast<uint32_t>(
                GetSpr<SpecialPurposeReg::AR>() / sizeof(uint32_t));
            PipeBarrier<PIPE_V>();
#if !LIM_SIMT_VICTIM_UNPACK
            UnionSync<HardEvent::V_S>(HardEvent::V_S);
#endif
            const uint32_t remaining = required - written;
            const uint32_t keep =
                compactCount < remaining ? compactCount : remaining;
#if LIM_SIMT_VICTIM_UNPACK
            if (keep != 0U) {
                asc_vf_call<lim_simt::UnpackVictims>(
                    dim3(LIM_SIMT_THREADS),
                    (__ubuf__ uint32_t *)compact.GetPhyAddr(),
                    (__ubuf__ int32_t *)destinations.GetPhyAddr() + written,
                    (__ubuf__ int32_t *)victimSources.GetPhyAddr() + written,
                    keep);
            }
            // SIMD compaction and SIMT unpack share PIPE_V. Wait after
            // unpack (instead of before scalar unpack) before reusing the
            // scratch or letting AppendExactVictims read destinations.
            // This also completes compaction when keep==0; no empty VF.
            UnionSync<HardEvent::V_S>(HardEvent::V_S);
#else
            for (uint32_t index = 0U; index < keep; ++index) {
                const uint32_t payload = compact.GetValue(index);
                destinations.SetValue(
                    written + index,
                    static_cast<int32_t>(
                        (payload >> VICTIM_SLOT_SHIFT) & VICTIM_SLOT_MASK));
                victimSources.SetValue(
                    written + index,
                    static_cast<int32_t>(payload & UNION_SOURCE_MASK));
            }
#endif
            written += keep;
        }
        return written;
    }

    __aicore__ inline uint32_t AppendExactVictims(
        uint32_t batch, uint32_t candidate, uint32_t poolRow,
        uint32_t budget, uint32_t written, uint32_t required,
        uint32_t routeCount, uint32_t queryStart,
        uint32_t scanSeed,
        LocalTensor<int32_t> destinations,
        LocalTensor<int32_t> victimSources)
    {
        if (written >= required) {
            return written;
        }
        // One int32 marker per physical cache slot fits in pairInputBuf_
        // because the MTP cache budget is capped at 12288.
        LocalTensor<int32_t> protectedSlots =
            pairInputBuf_.Get<int32_t>();
        Duplicate(protectedSlots, static_cast<int32_t>(0), budget);
        PipeBarrier<PIPE_V>();
        UnionSync<HardEvent::V_S>(HardEvent::V_S);
        for (uint32_t index = 0U; index < written; ++index) {
            const int32_t slot = destinations.GetValue(index);
            if (slot >= 0 && static_cast<uint32_t>(slot) < budget) {
                protectedSlots.SetValue(static_cast<uint32_t>(slot), 1);
            }
        }
        // Stage 1 writes the selected prefix/tail union to [T,2048].
        // Only prefix slots (< budget) belong to the eviction domain.
        for (uint32_t route = 0U; route < routeCount; ++route) {
            const uint64_t topkBase =
                (static_cast<uint64_t>(queryStart) + route) *
                vllm_a5_li_manage_c8_fast_workspace::ATTENTION_CAPACITY;
            for (uint32_t index = 0U; index < UNION_TOPK; ++index) {
                const int32_t slot =
                    topkSlotsGm_.GetValue(topkBase + index);
                if (slot >= 0 && static_cast<uint32_t>(slot) < budget) {
                    protectedSlots.SetValue(
                        static_cast<uint32_t>(slot), 1);
                }
            }
        }

        const uint32_t chunks =
            (candidate + VICTIM_SCAN_CHUNK - 1U) / VICTIM_SCAN_CHUNK;
        const uint32_t firstChunk = scanSeed % chunks;
        const uint64_t cacheBase =
            static_cast<uint64_t>(poolRow) * poolStride_;
        for (uint32_t visit = 0U;
             visit < chunks && written < required; ++visit) {
            const uint32_t chunk = (firstChunk + visit) % chunks;
            const uint32_t begin = chunk * VICTIM_SCAN_CHUNK;
            const uint32_t end =
                begin + VICTIM_SCAN_CHUNK < candidate
                    ? begin + VICTIM_SCAN_CHUNK
                    : candidate;
            for (uint32_t source = begin;
                 source < end && written < required; ++source) {
                const int32_t slot =
                    cacheSlotsGm_.GetValue(cacheBase + source);
                if (slot < 0 || static_cast<uint32_t>(slot) >= budget ||
                    protectedSlots.GetValue(
                        static_cast<uint32_t>(slot)) != 0) {
                    continue;
                }
                destinations.SetValue(written, slot);
                victimSources.SetValue(
                    written, static_cast<int32_t>(source));
                protectedSlots.SetValue(static_cast<uint32_t>(slot), 1);
                ++written;
            }
        }
        return written;
    }

    __aicore__ inline uint32_t FindVictims(
        uint32_t batch, uint32_t count, uint32_t routeCount,
        uint32_t queryStart, uint32_t scanSeed,
#if LIM_SIMT_REQUEST_META_CACHE
        const SparseRequestMetadata &metadata,
#endif
        LocalTensor<int32_t> destinations,
        LocalTensor<int32_t> victimSources)
    {
        if (count == 0U) {
            return 0U;
        }
        Duplicate(destinations, static_cast<int32_t>(-1), count);
        Duplicate(victimSources, static_cast<int32_t>(-1), count);
        PipeBarrier<PIPE_V>();

        LocalTensor<uint16_t> thresholdLocal = thresholdBuf_.Get<uint16_t>();
        uint16_t thresholds[UNION_ROUTES];
        LoadThresholds(batch, thresholdLocal, routeCount, thresholds);
#if LIM_SIMT_REQUEST_META_CACHE
        const uint32_t candidate = metadata.candidate;
        const uint32_t poolRow = metadata.poolRow;
        const uint32_t budget = metadata.budget;
#else
        const uint32_t candidate = static_cast<uint32_t>(
            offloadedLensGm_.GetValue(batch));
        const uint32_t poolRow = static_cast<uint32_t>(
            reqPoolEntriesGm_.GetValue(batch));
        const uint32_t budget = ReadBudget(batch, poolRow);
#endif
        uint32_t written = CompactSafeVictims(
            batch, candidate, poolRow, budget, count, routeCount,
            scanSeed, thresholds,
            destinations, victimSources);
        written = AppendExactVictims(
            batch, candidate, poolRow, budget, written, count,
            routeCount, queryStart, scanSeed,
            destinations, victimSources);
        return written;
    }

    __aicore__ inline uint32_t ApplyCacheUpdates(
        uint32_t batch, uint32_t count,
#if LIM_SIMT_REQUEST_META_CACHE
        const SparseRequestMetadata &metadata,
#endif
        LocalTensor<int32_t> sources,
        LocalTensor<int32_t> destinations,
        LocalTensor<int32_t> victimSources)
    {
#if LIM_SIMT_REQUEST_META_CACHE
        (void)batch;
        const uint32_t candidate = metadata.candidate;
        const uint32_t poolRow = metadata.poolRow;
        const uint32_t budget = metadata.budget;
#else
        const uint32_t candidate = static_cast<uint32_t>(
            offloadedLensGm_.GetValue(batch));
        const uint32_t poolRow = static_cast<uint32_t>(
            reqPoolEntriesGm_.GetValue(batch));
        const uint32_t budget = ReadBudget(batch, poolRow);
#endif
        const uint64_t cacheBase =
            static_cast<uint64_t>(poolRow) * poolStride_;
        uint32_t updated = 0U;
#if LIM_SIMT_POSTPROCESS
        // countBuf_[1] is idle until PublishFinalOutputs rewrites all count
        // rows. Keep it separate from the live source/destination arrays.
        LocalTensor<uint32_t> validPrefix = countBuf_.Get<uint32_t>()[1U];
        validPrefix.SetValue(0U, count);
        UnionSync<HardEvent::S_V>(HardEvent::S_V);
        asc_vf_call<lim_simt::ApplyUpdates>(
            dim3(LIM_SIMT_THREADS),
            (__gm__ int32_t *)cacheSlotsGm_.GetPhyAddr(), cacheBase,
            (__ubuf__ int32_t *)sources.GetPhyAddr(),
            (__ubuf__ int32_t *)destinations.GetPhyAddr(),
            (__ubuf__ int32_t *)victimSources.GetPhyAddr(),
            (__ubuf__ uint32_t *)validPrefix.GetPhyAddr(),
            count, candidate, budget);
        UnionSync<HardEvent::V_S>(HardEvent::V_S);
        updated = validPrefix.GetValue(0U);
#else
        for (uint32_t index = 0U; index < count; ++index) {
            const int32_t source = sources.GetValue(index);
            const int32_t destination = destinations.GetValue(index);
            const int32_t victimSource = victimSources.GetValue(index);
            if (source < 0 || victimSource < 0 || destination < 0 ||
                static_cast<uint32_t>(source) >= candidate ||
                static_cast<uint32_t>(victimSource) >= candidate ||
                static_cast<uint32_t>(destination) >= budget) {
                break;
            }
            cacheSlotsGm_.SetValue(
                cacheBase + static_cast<uint32_t>(victimSource), -1);
            cacheSlotsGm_.SetValue(
                cacheBase + static_cast<uint32_t>(source), destination);
            ++updated;
        }
#endif
        if (updated != 0U) {
            PipeBarrier<PIPE_ALL>();
        }
        return updated;
    }

    __aicore__ inline void PrepareTopkMissPrefixes(
        uint32_t batch, uint32_t routeCount,
        const uint32_t lengths[UNION_ROUTES],
        uint32_t unionCount, LocalTensor<int32_t> unionSources,
        LocalTensor<int32_t> unionDestinations,
        LocalTensor<int32_t> allDestinations)
    {
        LocalTensor<int32_t> routePairs = pairInputBuf_.Get<int32_t>();
#if LIM_SIMT_BATCHED_REPAIR
        static_assert(UNION_ROUTES == 4U && UNION_TOPK == lim_simt::REPAIR_TOPK,
            "batched repair must match the fixed route-pair workspace rows");
        // Each DMA owns a separate pairInput row. No route overwrites data
        // still being read by another route, so one MTE2_V wait is sufficient.
        const uint64_t requestPairBase =
            static_cast<uint64_t>(batch) * UNION_CAPACITY * 2U;
        for (uint32_t route = 0U; route < routeCount; ++route) {
            const uint32_t length = lengths[route];
            if (length != 0U) {
                const uint32_t pairOffset = route * UNION_PAIR_WORDS;
                DataCopyPad(
                    routePairs[pairOffset],
                    routePairsGm_[requestPairBase + pairOffset],
                    {1, length * 2U * static_cast<uint32_t>(sizeof(int32_t)),
                     0, 0, 0},
                    {false, 0, 0, 0});
            }
        }
        UnionSync<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        asc_vf_call<lim_simt::RepairMissSlotsBatched>(
            dim3(LIM_SIMT_THREADS),
            (__ubuf__ int32_t *)routePairs.GetPhyAddr(),
            (__ubuf__ int32_t *)unionSources.GetPhyAddr(),
            (__ubuf__ int32_t *)unionDestinations.GetPhyAddr(),
            (__ubuf__ int32_t *)allDestinations.GetPhyAddr(),
            lengths[0], lengths[1], lengths[2], lengths[3], unionCount);
        // Complete pair reads before publication and the next request can
        // reuse their UB. Keep the existing final V_MTE3 ordering.
        UnionSync<HardEvent::V_MTE2>(HardEvent::V_MTE2);
#else
        // pairInputBuf_ has 16384 int32 words. One 4096-word interleaved
        // route-pair row and four 2048-word destination prefixes are disjoint.
        const uint64_t requestPairBase =
            static_cast<uint64_t>(batch) * UNION_CAPACITY * 2U;
        for (uint32_t route = 0U; route < routeCount; ++route) {
            const uint32_t length = lengths[route];
            if (length == 0U) {
                continue;
            }
            const uint32_t pairOffset = route * UNION_PAIR_WORDS;
            DataCopyPad(
                routePairs,
                routePairsGm_[requestPairBase + pairOffset],
                {1, length * 2U * static_cast<uint32_t>(sizeof(int32_t)),
                 0, 0, 0},
                {false, 0, 0, 0});
            LocalTensor<int32_t> rowDestinations =
                allDestinations[route * UNION_TOPK];
#if LIM_SIMT_POSTPROCESS
            UnionSync<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            asc_vf_call<lim_simt::RepairMissSlots>(
                dim3(LIM_SIMT_THREADS),
                (__ubuf__ int32_t *)routePairs.GetPhyAddr(),
                (__ubuf__ int32_t *)unionSources.GetPhyAddr(),
                (__ubuf__ int32_t *)unionDestinations.GetPhyAddr(),
                (__ubuf__ int32_t *)rowDestinations.GetPhyAddr(),
                length, unionCount);
            // Next route reuses routePairs for MTE2. Do not overwrite it
            // while the current VF still reads it. Final MTE3 publication
            // retains the existing V_MTE3 dependency.
            UnionSync<HardEvent::V_MTE2>(HardEvent::V_MTE2);
#else
            UnionSync<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            uint32_t unionCursor = 0U;
            for (uint32_t miss = 0U; miss < length; ++miss) {
                // BuildSortedMissPairs emits source IDs in ascending order,
                // matching the deduplicated union and the public source
                // prefix already published once by Stage 1.
                const int32_t source = routePairs.GetValue(miss * 2U + 1U);
                while (unionCursor < unionCount &&
                       unionSources.GetValue(unionCursor) < source) {
                    ++unionCursor;
                }
                const int32_t destination =
                    unionCursor < unionCount &&
                            unionSources.GetValue(unionCursor) == source
                        ? unionDestinations.GetValue(unionCursor)
                        : -1;
                rowDestinations.SetValue(miss, destination);
            }
#endif
        }
#endif
    }

    __aicore__ inline void PublishFinalOutputs(
        uint32_t batch, uint32_t queryStart,
        uint32_t routeCount,
        const uint32_t lengths[UNION_ROUTES],
        uint32_t count, LocalTensor<int32_t> countLocal,
#if LIM_SIMT_REQUEST_META_CACHE
        const SparseRequestMetadata &metadata,
#endif
        LocalTensor<int32_t> unionSources,
        LocalTensor<int32_t> unionDestinations,
        LocalTensor<int32_t> topkMissDestinations)
    {
        constexpr uint32_t TAIL = VllmA5LiManageC8FastUnionVF::TAIL_CAPACITY;
#if LIM_SIMT_REQUEST_META_CACHE
        const uint32_t budget = metadata.budget;
#else
        const uint32_t poolRow = static_cast<uint32_t>(
            reqPoolEntriesGm_.GetValue(batch));
        const uint32_t budget = ReadBudget(batch, poolRow);
#endif
        // Independent 32-byte scalar rows: no DMA source is overwritten
        // until the single completion wait at the end of this request.
        LocalTensor<int32_t> unionCountLocal = countLocal[8U];
        LocalTensor<int32_t> residentLocal = countLocal[16U];
        for (uint32_t route = 0U; route < routeCount; ++route) {
            countLocal.SetValue(route, static_cast<int32_t>(lengths[route]));
        }
        unionCountLocal.SetValue(0U, static_cast<int32_t>(count));
        residentLocal.SetValue(0U, static_cast<int32_t>(budget + TAIL));
        UnionSync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        UnionSync<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopyPad(
            perQueryMissCountsGm_[queryStart], countLocal,
            {1, static_cast<uint16_t>(
                    routeCount * sizeof(int32_t)), 0, 0});
        DataCopyPad(
            unionCountsGm_[batch], unionCountLocal,
            {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        DataCopyPad(
            residentSeqLengthsGm_[batch], residentLocal,
            {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        if (count != 0U) {
            const uint64_t unionOffset =
                static_cast<uint64_t>(batch) * outputCapacity_;
            const uint16_t unionBytes = static_cast<uint16_t>(
                count * static_cast<uint32_t>(sizeof(int32_t)));
            DataCopyPad(
                unionSourcesGm_[unionOffset], unionSources,
                {1, unionBytes, 0, 0});
            DataCopyPad(
                unionDestinationsGm_[unionOffset], unionDestinations,
                {1, unionBytes, 0, 0});
        }
        for (uint32_t route = 0U; route < routeCount; ++route) {
            const uint64_t publicRow =
                (static_cast<uint64_t>(queryStart) + route) *
                vllm_a5_li_manage_c8_fast_workspace::ATTENTION_CAPACITY;
            if (count != 0U && lengths[route] != 0U) {
                DataCopyPad(
                    topkSlotsGm_[publicRow],
                    topkMissDestinations[route * UNION_TOPK],
                    {1, static_cast<uint16_t>(lengths[route] * sizeof(int32_t)),
                     0, 0});
            }
        }
        UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

    __aicore__ inline void FillAttentionRows(
        uint32_t queryStart, uint32_t routeCount, int32_t value)
    {
        LocalTensor<int32_t> row = sourceBuf_.Get<int32_t>();
        Duplicate(
            row, value,
            vllm_a5_li_manage_c8_fast_workspace::ATTENTION_CAPACITY);
        UnionSync<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        for (uint32_t route = 0U; route < routeCount; ++route) {
            const uint64_t publicRow =
                (static_cast<uint64_t>(queryStart) + route) *
                vllm_a5_li_manage_c8_fast_workspace::ATTENTION_CAPACITY;
            DataCopy(
                topkSlotsGm_[publicRow], row,
                vllm_a5_li_manage_c8_fast_workspace::ATTENTION_CAPACITY);
            DataCopy(
                topkIndexGm_[publicRow], row,
                vllm_a5_li_manage_c8_fast_workspace::ATTENTION_CAPACITY);
        }
        UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

    __aicore__ inline void PublishRequestMetadata(
        uint32_t batch, uint32_t queryStart, uint32_t routeCount,
        int32_t perQueryCount, int32_t copyCount, int32_t residentLength)
    {
        LocalTensor<int32_t> values = countBuf_.Get<int32_t>();
        for (uint32_t route = 0U; route < routeCount; ++route) {
            values.SetValue(route, perQueryCount);
        }
        UnionSync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(
            perQueryMissCountsGm_[queryStart], values,
            {1, static_cast<uint16_t>(routeCount * sizeof(int32_t)), 0, 0});
        UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);

        // Every UB source passed to MTE3 must be 32-byte aligned. Reuse the
        // first aligned scalar lane sequentially instead of addressing
        // values[4]/values[5], whose byte offsets are only 16/20.
        values.SetValue(0U, copyCount);
        UnionSync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(
            unionCountsGm_[batch], values,
            {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);

        values.SetValue(0U, residentLength);
        UnionSync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(
            residentSeqLengthsGm_[batch], values,
            {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

    __aicore__ inline void PublishPad(
        uint32_t batch, uint32_t queryStart, uint32_t routeCount)
    {
        // PAD owns real packed query rows but has no request-pool state.
        // Clear only those compact public rows and never dereference its
        // undefined pool entry.
        FillAttentionRows(queryStart, routeCount, -1);
        PublishRequestMetadata(batch, queryStart, routeCount, 0, 0, 0);
    }

    __aicore__ inline void PublishError(
        uint32_t batch, uint32_t queryStart, uint32_t routeCount)
    {
        FillAttentionRows(queryStart, routeCount, -1);
        PublishRequestMetadata(batch, queryStart, routeCount, -1, -1, 0);
    }

    __aicore__ inline void PublishDense(
        uint32_t batch, uint32_t queryStart, uint32_t queryEnd)
    {
        const int32_t finalRaw = finalSeqLengthsKvGm_.GetValue(batch);
        const uint32_t routeCount = queryEnd - queryStart;
        if (finalRaw < static_cast<int32_t>(routeCount) ||
            finalRaw > static_cast<int32_t>(sourceCapacity_) ||
            finalRaw > static_cast<int32_t>(maxCandidateLen_)) {
            PublishError(batch, queryStart, routeCount);
            return;
        }

        LocalTensor<int32_t> row = sourceBuf_.Get<int32_t>();
        const uint32_t capacity = static_cast<uint32_t>(
            vllm_a5_li_manage_c8_fast_workspace::ATTENTION_CAPACITY);
        for (uint32_t route = 0U; route < routeCount; ++route) {
            const uint32_t queryRow = queryStart + route;
            const uint32_t later = queryEnd - 1U - queryRow;
            const uint32_t visible = static_cast<uint32_t>(finalRaw) - later;
            const uint64_t publicRow =
                static_cast<uint64_t>(queryRow) * capacity;
            if (visible <= UNION_TOPK) {
                Duplicate(row, static_cast<int32_t>(-1), capacity);
                PipeBarrier<PIPE_V>();
                if (visible != 0U) {
                    CreateVecIndex(row, static_cast<int32_t>(0), visible);
                    PipeBarrier<PIPE_V>();
                }
                UnionSync<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                DataCopy(topkSlotsGm_[publicRow], row, capacity);
                DataCopy(topkIndexGm_[publicRow], row, capacity);
                UnionSync<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            } // Long DENSE rows were already published by Stage A.
        }
        UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        PublishRequestMetadata(
            batch, queryStart, routeCount, 0, 0, finalRaw);
    }

    __aicore__ inline bool ValidateSparse(
        uint32_t batch, uint32_t routeCount,
        uint32_t &candidate, uint32_t &finalLen,
        uint32_t &poolRow, uint32_t &budget)
    {
        const int32_t candidateRaw = offloadedLensGm_.GetValue(batch);
        const int32_t finalRaw = finalSeqLengthsKvGm_.GetValue(batch);
        const int32_t poolRowRaw = reqPoolEntriesGm_.GetValue(batch);
        if (poolRowRaw < 0 ||
            poolRowRaw >= static_cast<int32_t>(poolSize_) ||
            candidateRaw < static_cast<int32_t>(UNION_TOPK) ||
            candidateRaw > static_cast<int32_t>(sourceCapacity_) ||
            finalRaw > static_cast<int32_t>(maxCandidateLen_) ||
            candidateRaw % 128 != 0 ||
            finalRaw < static_cast<int32_t>(routeCount) ||
            finalRaw > static_cast<int32_t>(sourceCapacity_)) {
            return false;
        }
        candidate = static_cast<uint32_t>(candidateRaw);
        finalLen = static_cast<uint32_t>(finalRaw);
        poolRow = static_cast<uint32_t>(poolRowRaw);
        const uint32_t earliestVisible = finalLen - (routeCount - 1U);
        const uint32_t tailCapacity = static_cast<uint32_t>(
            VllmA5LiManageC8FastUnionVF::TAIL_CAPACITY);
        if (candidate > earliestVisible ||
            finalLen - candidate > tailCapacity) {
            return false;
        }
        const int32_t metadata = cacheSlotsGm_.GetValue(
            static_cast<uint64_t>(poolRow) * poolStride_ + sourceCapacity_);
        // A negative value denotes first-fill and must have selected the
        // correctness-first batch path before Stage A.  The unified fast
        // path accepts only a populated cache.
        if (metadata <= 0) {
            return false;
        }
        budget = static_cast<uint32_t>(metadata);
        return (budget == 8192U || budget == 10240U || budget == 12288U) &&
            budget <= candidate;
    }

    __aicore__ inline void ProcessSparse(
        uint32_t batch, uint32_t queryStart, uint32_t routeCount)
    {
        uint32_t candidate = 0U;
        uint32_t finalLen = 0U;
        uint32_t poolRow = 0U;
        uint32_t budget = 0U;
        if (!ValidateSparse(
                batch, routeCount, candidate, finalLen, poolRow, budget)) {
            PublishError(batch, queryStart, routeCount);
            return;
        }
#if LIM_SIMT_REQUEST_META_CACHE
        // ValidateSparse has already read and checked these four values.
        // Keep only a request-local snapshot, never a member / GM / UB cache:
        // the next request and every graph replay validate fresh metadata.
        // Steady updates change token columns, not the positive budget word.
        const SparseRequestMetadata metadata{candidate, finalLen, poolRow, budget};
#endif

        LocalTensor<float> pairs = pairInputBuf_.Get<float>();
        LocalTensor<float> merged = pairOutputBuf_.Get<float>();
        LocalTensor<int32_t> sources = sourceBuf_.Get<int32_t>();
        LocalTensor<int32_t> countLocal = countBuf_.Get<int32_t>();
        const uint64_t routeBase =
            static_cast<uint64_t>(batch) * UNION_CAPACITY * 2U;
        // Internal workspaces retain four fixed lanes per request, while the
        // public ABI is compact TND.  Read only the actually present lanes.
        DataCopyPad(
            countLocal, routeCountsGm_[batch * UNION_ROUTES],
            {1, static_cast<uint32_t>(routeCount * sizeof(int32_t)),
             0, 0, 0},
            {false, 0, 0, 0});
        UnionSync<HardEvent::MTE2_S>(HardEvent::MTE2_S);

        uint32_t lengths[UNION_ROUTES] = {0U, 0U, 0U, 0U};
        uint32_t total = 0U;
        LocalTensor<int32_t> pairWords = pairs.ReinterpretCast<int32_t>();
        for (uint32_t route = 0U; route < routeCount; ++route) {
            const int32_t rawLength = countLocal.GetValue(route);
            if (rawLength < 0 || rawLength > static_cast<int32_t>(UNION_TOPK)) {
                PublishError(batch, queryStart, routeCount);
                return;
            }
            const uint32_t length = static_cast<uint32_t>(rawLength);
            lengths[route] = length;
            total += length;
            if (length != 0U) {
                const uint32_t pairOffset = route * UNION_PAIR_WORDS;
                DataCopyPad(
                    pairWords[pairOffset],
                    routePairsGm_[routeBase + pairOffset],
                    {1, static_cast<uint32_t>(
                            length * 2U * sizeof(int32_t)),
                     0, 0, 0},
                    {false, 0, 0, 0});
            }
        }
        if (total == 0U) {
            PublishFinalOutputs(
                batch, queryStart, routeCount, lengths,
                0U, countLocal,
#if LIM_SIMT_REQUEST_META_CACHE
                metadata,
#endif
                sources, sources, sources);
            return;
        }

        UnionSync<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        MrgSort4Info params;
        params.elementLengths[0] = lengths[0];
        params.elementLengths[1] = lengths[1];
        params.elementLengths[2] = lengths[2];
        params.elementLengths[3] = lengths[3];
        params.ifExhaustedSuspension = false;
        params.validBit =
            (lengths[0] > 0U ? 0b0001 : 0U) |
            (lengths[1] > 0U ? 0b0010 : 0U) |
            (lengths[2] > 0U ? 0b0100 : 0U) |
            (lengths[3] > 0U ? 0b1000 : 0U);
        params.repeatTimes = 1;
        MrgSortSrcList<float> inputs;
        inputs.src1 = pairs;
        inputs.src2 = pairs[UNION_PAIR_WORDS];
        inputs.src3 = pairs[UNION_PAIR_WORDS * 2U];
        inputs.src4 = pairs[UNION_PAIR_WORDS * 3U];
        MrgSort<float>(merged, inputs, params);
        PipeBarrier<PIPE_V>();

        const uint32_t count = Deduplicate(merged, total, sources);
        if (count == 0U || count > total ||
            count > budget || count > outputCapacity_) {
            PublishError(batch, queryStart, routeCount);
            return;
        }
        LocalTensor<int32_t> victimStorage =
            pairOutputBuf_.Get<int32_t>();
        LocalTensor<int32_t> destinations = victimStorage;
        LocalTensor<int32_t> victimSources =
            victimStorage[UNION_CAPACITY];
        // Reuse the validated request-local length; both victim passes must
        // start from the same seed, refreshed on every invocation/replay.
        const uint32_t scanSeed =
            vllm_a5_li_manage_c8_victim_scan::HashVictimScanSeed(
                finalLen);
        const uint32_t found = FindVictims(
            batch, count, routeCount, queryStart, scanSeed,
#if LIM_SIMT_REQUEST_META_CACHE
            metadata,
#endif
            destinations, victimSources);
        if (found != count) {
            PublishError(batch, queryStart, routeCount);
            return;
        }
        const uint32_t updated = ApplyCacheUpdates(
            batch, count,
#if LIM_SIMT_REQUEST_META_CACHE
            metadata,
#endif
            sources, destinations, victimSources);
        if (updated != count) {
            PublishError(batch, queryStart, routeCount);
            return;
        }
#if LIM_SIMT_BATCHED_REPAIR
        // ApplyCacheUpdates has completed (V_S + PIPE_ALL). Victim sources
        // are now dead; reuse these 8192 words for four repaired prefixes.
        // The first half of pairOutput (union destinations) stays read-only
        // until PublishFinalOutputs finishes MTE3. No extra UB allocation.
        LocalTensor<int32_t> topkMissDestinations = victimSources;
#else
        LocalTensor<int32_t> topkScratch = pairInputBuf_.Get<int32_t>();
        LocalTensor<int32_t> topkMissDestinations =
            topkScratch[UNION_PAIR_WORDS];
#endif
        PrepareTopkMissPrefixes(
            batch, routeCount, lengths, updated,
            sources, destinations, topkMissDestinations);
        PublishFinalOutputs(
            batch, queryStart, routeCount, lengths, updated,
            countLocal,
#if LIM_SIMT_REQUEST_META_CACHE
            metadata,
#endif
            sources, destinations, topkMissDestinations);
    }

    __aicore__ inline void ProcessRequest(uint32_t batch)
    {
        uint32_t queryStart = 0U;
        uint32_t queryEnd = 0U;
        uint32_t routeCount = 0U;
        if (!QueryRange(batch, queryStart, queryEnd, routeCount)) {
            // No safe compact TND range exists to clear. Publish request
            // scalars only; the fast gate should normally reject this case.
            LocalTensor<int32_t> values = countBuf_.Get<int32_t>();
            values.SetValue(0U, -1);
            UnionSync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(
                unionCountsGm_[batch], values,
                {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
            UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
            values.SetValue(0U, 0);
            UnionSync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(
                residentSeqLengthsGm_[batch], values,
                {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
            UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
            return;
        }

        const int32_t mode = rowModesGm_.GetValue(batch);
        if (mode == UNION_ROW_MODE_PAD) {
            PublishPad(batch, queryStart, routeCount);
        } else if (mode == UNION_ROW_MODE_DENSE) {
            PublishDense(batch, queryStart, queryEnd);
        } else if (mode == UNION_ROW_MODE_SPARSE) {
            ProcessSparse(batch, queryStart, routeCount);
        } else {
            PublishError(batch, queryStart, routeCount);
        }
    }

    GlobalTensor<int32_t> routePairsGm_;
    GlobalTensor<uint16_t> routeThresholdsGm_;
    GlobalTensor<int32_t> routeCountsGm_;
    GlobalTensor<uint16_t> scoreWorkspaceGm_;
    GlobalTensor<int32_t> actualSeqLengthsQueryGm_;
    GlobalTensor<int32_t> offloadedLensGm_;
    GlobalTensor<int32_t> finalSeqLengthsKvGm_;
    GlobalTensor<int32_t> rowModesGm_;
    GlobalTensor<int32_t> reqPoolEntriesGm_;
    GlobalTensor<int32_t> cacheSlotsGm_;
    GlobalTensor<int32_t> unionSourcesGm_;
    GlobalTensor<int32_t> unionDestinationsGm_;
    GlobalTensor<int32_t> unionCountsGm_;
    GlobalTensor<int32_t> topkSlotsGm_;
    GlobalTensor<int32_t> topkIndexGm_;
    GlobalTensor<int32_t> perQueryMissCountsGm_;
    GlobalTensor<int32_t> residentSeqLengthsGm_;
    TBuf<TPosition::VECCALC> pairInputBuf_;
    TBuf<TPosition::VECCALC> pairOutputBuf_;
    TBuf<TPosition::VECCALC> sourceBuf_;
    TBuf<TPosition::VECCALC> countBuf_;
    TBuf<TPosition::VECCALC> thresholdBuf_;
    uint32_t sourceCapacity_ = 0U;
    uint32_t poolStride_ = 0U;
    uint32_t outputCapacity_ = 0U;
    uint32_t scoreRowStride_ = 0U;
    uint32_t batchSize_ = 0U;
    uint32_t poolSize_ = 0U;
    uint32_t totalQueryRows_ = 0U;
    uint32_t maxCandidateLen_ = 0U;
};
} // namespace vllm_a5_li_manage_c8_fast

#endif
