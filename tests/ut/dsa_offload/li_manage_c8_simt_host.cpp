// Host arithmetic self-test, NOT an Ascend compiler / concurrency simulator.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#define __simt_callee__
#define __aicore__
#define __gm__
#define __ubuf__
#include "lim_simt_postprocess_algorithms.h"
#include "lim_simt_config.h"
#include "vllm_a5_li_manage_c8_budget.h"

static void CheckBudgetBounds()
{
    uint32_t accepted = 0U;
    for (uint32_t budget = 0U; budget <= 32768U; ++budget) {
        const bool expected = budget >= 8192U && budget <= 16000U && budget % 128U == 0U;
        assert(vllm_a5_li_manage_c8_budget::IsSupported(budget) == expected);
        if (!expected) {
            continue;
        }
        ++accepted;
        // Both prefix slot C-1 and all 256 tail slots round-trip unchanged,
        // even with the maximum 18-bit token ID. Neither miss marker aliases.
        for (uint32_t slot = budget - 1U; slot < budget + 256U; ++slot) {
            const uint32_t payload = (slot << 18U) | 262143U;
            int32_t decodedSlot = -1, decodedToken = -1;
            lim_simt::UnpackVictim(payload, &decodedSlot, &decodedToken, 0U);
            assert(decodedSlot == static_cast<int32_t>(slot));
            assert(decodedToken == 262143);
            assert(slot < 0x3ffeU && payload != 0xffffffffU);
        }
    }
    assert(accepted == 62U);
    assert(!vllm_a5_li_manage_c8_budget::IsSupported(0xffffffffU));
}

static void CheckVictimUnpack(uint32_t required)
{
    constexpr uint32_t guard = 7U; // deliberately not a 32B-aligned prefix
    constexpr int32_t sentinel = -777;
    for (uint32_t budget = 8192U; budget <= 16000U; budget += 128U) {
        std::vector<int32_t> destinations(required + 2U * guard, sentinel);
        std::vector<int32_t> victims(destinations);
        auto expectedDestinations = destinations;
        auto expectedVictims = victims;
        uint32_t written = 0U;
        // Empty chunks, unaligned output offsets, multi-thread strides, and
        // a final partial chunk all occur in the real threshold scan.
        const uint32_t chunkCounts[] = {0U, 1U, 31U, 257U, 0U, 511U, 63U, 2048U};
        for (uint32_t chunk = 0U; written < required; ++chunk) {
            const uint32_t compactCount = chunkCounts[chunk % 8U];
            const uint32_t keep = std::min(compactCount, required - written);
            std::vector<uint32_t> payloads(compactCount);
            for (uint32_t i = 0U; i < compactCount; ++i) {
                const uint32_t token = 262143U - (chunk * 2048U + i) % 262144U;
                const uint32_t slot = (budget - 1U + i * 17U) % budget;
                payloads[i] = (slot << 18U) | token;
                if (i < keep) {
                    expectedDestinations[guard + written + i] = static_cast<int32_t>(slot);
                    expectedVictims[guard + written + i] = static_cast<int32_t>(token);
                }
            }
            const auto beforePayloads = payloads;
            for (uint32_t tid = 0U; tid < LIM_SIMT_THREADS; ++tid) {
                for (uint32_t i = tid; i < keep; i += LIM_SIMT_THREADS) {
                    lim_simt::UnpackVictim(
                        payloads[i], destinations.data() + guard + written,
                        victims.data() + guard + written, i);
                }
            }
            written += keep;
            assert(payloads == beforePayloads);
            assert(destinations == expectedDestinations);
            assert(victims == expectedVictims); // includes prefix + suffix guards
        }
    }
    // Upper encoding bits must not be sign-extended; decoding does not
    // invent a stricter range check than the original scalar operation.
    int32_t destination = -1, victim = -1;
    lim_simt::UnpackVictim(0xffffffffU, &destination, &victim, 0U);
    assert(destination == 16383 && victim == 262143);
}

static void CheckUpdates(uint32_t count, uint32_t firstInvalid, uint32_t budget)
{
    constexpr uint32_t capacity = 1U << 18U;
    constexpr uint64_t stride = capacity + 1U;
    constexpr uint64_t rowBase = 2U * stride;
    std::vector<int32_t> pool(4U * stride, -1);
    std::vector<int32_t> source(count), victim(count), destination(count);
    pool[rowBase + capacity] = budget;
    for (uint32_t i = 0; i < budget; ++i) {
        pool[rowBase + i] = static_cast<int32_t>(i);
    }
    for (uint32_t i = 0; i < count; ++i) {
        source[i] = static_cast<int32_t>(capacity - count + i);
        victim[i] = static_cast<int32_t>(budget - count + i);
        destination[i] = static_cast<int32_t>(budget - count + i);
    }
    if (firstInvalid < count) {
        destination[firstInvalid] = budget;
    }
    const auto initial = pool;
    auto expected = pool;
    uint32_t prefix = 0;
    for (; prefix < count; ++prefix) {
        if (!lim_simt::ValidPair(source[prefix], destination[prefix], victim[prefix], capacity, budget)) {
            break;
        }
        expected[rowBase + victim[prefix]] = -1;
        expected[rowBase + source[prefix]] = destination[prefix];
    }

    // Model parallel validation followed by its barrier, then arbitrarily
    // interleave independent writes. The real VF uses the same helpers.
    uint32_t validPrefix = count;
    for (uint32_t tid = 0; tid < LIM_SIMT_THREADS; ++tid) {
        for (uint32_t i = tid; i < count; i += LIM_SIMT_THREADS) {
            if (!lim_simt::ValidPair(source[i], destination[i], victim[i], capacity, budget)) {
                validPrefix = std::min(validPrefix, i);
            }
        }
    }
    assert(validPrefix == prefix);
    std::vector<uint32_t> order(validPrefix);
    std::iota(order.begin(), order.end(), 0U);
    std::mt19937 random(count + firstInvalid);
    std::shuffle(order.begin(), order.end(), random);
    for (const auto i : order) {
        lim_simt::UpdatePair(pool.data(), rowBase, source[i], destination[i], victim[i]);
    }
    assert(pool == expected); // includes metadata and three inactive rows
    if (count == 0U) {
        assert(pool == initial);
    }
}

static void CheckRepair(uint32_t count)
{
    std::vector<int32_t> sources(count), destinations(count);
    for (uint32_t i = 0; i < count; ++i) {
        sources[i] = static_cast<int32_t>(262144U - 2U * count + 2U * i);
        destinations[i] = static_cast<int32_t>((i * 17U) % 12288U);
    }
    assert(lim_simt::LookupSlot(-1, sources.data(), destinations.data(), count) == -1);
    assert(lim_simt::LookupSlot(262144, sources.data(), destinations.data(), count) == -1);
    for (uint32_t routes = 1; routes <= 4; ++routes) {
        for (uint32_t route = 0; route < routes; ++route) {
            std::vector<int32_t> pairs, expected;
            for (uint32_t i = 0; i < count && expected.size() < 2048; ++i) {
                if (i % 5 == route || i % 7 == 0) {
                    pairs.push_back(-999); // score word is not a token ID
                    pairs.push_back(sources[i]);
                    expected.push_back(destinations[i]);
                }
            }
            std::vector<int32_t> actual(2048 + 256, -777);
            for (uint32_t tid = 0; tid < LIM_SIMT_THREADS; ++tid) {
                for (uint32_t i = tid; i < expected.size(); i += LIM_SIMT_THREADS) {
                    actual[i] = lim_simt::LookupSlot(
                        pairs[2U*i+1U], sources.data(), destinations.data(), count);
                }
            }
            assert(std::equal(expected.begin(), expected.end(), actual.begin()));
            assert(std::all_of(actual.begin() + expected.size(), actual.end(),
                               [](int32_t v) { return v == -777; }));
        }
    }
    for (uint32_t i = 0; i < count; ++i) {
        assert(lim_simt::LookupSlot(sources[i], sources.data(), destinations.data(), count) == destinations[i]);
        assert(lim_simt::LookupSlot(sources[i]+1, sources.data(), destinations.data(), count) == -1);
    }
}

static void CheckBatchedRepair(const std::array<uint32_t, 4> &lengths, uint32_t step)
{
    constexpr uint32_t topk = lim_simt::REPAIR_TOPK;
    constexpr uint32_t capacity = 4U * topk;
    constexpr uint32_t guard = 8U;
    constexpr int32_t sentinel = -777;
    // Model the real alias plan: union destinations occupy the first half
    // of pairOutput, repaired prefixes reuse its now-dead victim half.
    std::vector<int32_t> pairInput(2U * capacity + 2U * guard, sentinel);
    std::vector<int32_t> pairOutput(2U * capacity + 2U * guard, sentinel);
    std::vector<int32_t> sources;
    for (uint32_t route = 0U; route < 4U; ++route) {
        for (uint32_t i = 0U; i < lengths[route]; ++i) {
            const int32_t token = static_cast<int32_t>(245760U + 2U * (route * step + i));
            pairInput[guard + 2U * (route * topk + i)] = -999; // not a token ID
            pairInput[guard + 2U * (route * topk + i) + 1U] = token;
            sources.push_back(token);
        }
    }
    std::sort(sources.begin(), sources.end());
    sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
    for (uint32_t i = 0U; i < sources.size(); ++i) {
        pairOutput[guard + i] = static_cast<int32_t>((8192U + i * 17U) % 12288U);
    }
    const auto beforeInput = pairInput;
    const auto beforeSources = sources;
    auto expected = pairOutput;
    // Independent per-route reference, including every inactive suffix.
    for (uint32_t route = 0U; route < 4U; ++route) {
        for (uint32_t i = 0U; i < lengths[route]; ++i) {
            const int32_t token = pairInput[guard + 2U * (route * topk + i) + 1U];
            const auto found = std::lower_bound(sources.begin(), sources.end(), token);
            assert(found != sources.end() && *found == token);
            expected[guard + capacity + route * topk + i] =
                pairOutput[guard + static_cast<uint32_t>(found - sources.begin())];
        }
    }
    const uint32_t end0 = lengths[0];
    const uint32_t end1 = end0 + lengths[1];
    const uint32_t end2 = end1 + lengths[2];
    const uint32_t total = end2 + lengths[3];
    for (uint32_t tid = 0U; tid < LIM_SIMT_THREADS; ++tid) {
        for (uint32_t index = tid; index < total; index += LIM_SIMT_THREADS) {
            lim_simt::RepairBatchedMiss(
                pairInput.data() + guard, sources.data(), pairOutput.data() + guard,
                pairOutput.data() + guard + capacity, index, end0, end1, end2,
                static_cast<uint32_t>(sources.size()));
        }
    }
    assert(pairInput == beforeInput);
    assert(sources == beforeSources);
    assert(pairOutput == expected); // live union destinations + all row/outer guards
    // After the VF finishes, tail publication may overwrite input pairs,
    // but must not change either the union or per-query destination slots.
    std::fill_n(pairInput.begin() + guard, 2U * 4U * 256U, -123);
    assert(pairOutput == expected);
}

static void CheckBatchedRepairs()
{
    const std::array<uint32_t, 14> edges = {
        0U, 1U, 31U, 32U, 33U, 63U, 64U, 65U, 255U, 256U, 257U, 511U, 513U, 2048U};
    for (uint32_t routes = 1U; routes <= 4U; ++routes) {
        for (uint32_t active = 0U; active < routes; ++active) {
            for (const uint32_t length : edges) {
                std::array<uint32_t, 4> lengths{};
                lengths[active] = length; // leading, middle and trailing empty lanes
                CheckBatchedRepair(lengths, 2048U);
            }
        }
        for (uint32_t start = 0U; start < edges.size(); ++start) {
            std::array<uint32_t, 4> lengths{};
            for (uint32_t route = 0U; route < routes; ++route) {
                lengths[route] = edges[(start + route * 3U) % edges.size()];
            }
            for (const uint32_t step : {0U, 512U, 2048U}) {
                CheckBatchedRepair(lengths, step); // shared, overlapping, disjoint misses
            }
        }
        std::array<uint32_t, 4> full{};
        std::fill_n(full.begin(), routes, 2048U);
        CheckBatchedRepair(full, 0U); // all routes share the same 2048 misses
        CheckBatchedRepair(full, 2048U); // up to 8192 distinct union misses
    }
}

// Peak sums of the sequential TPipe allocations (all are 32B aligned).
// Phases reuse the arena after pipe.Reset(), so their sizes are not added.
constexpr uint32_t FastUb(uint32_t heads, uint32_t tile)
{
    return tile*heads*128U*4U + 3U*tile*512U + 16384U + tile*128U*2U +
        36864U + 8448U + 4096U + 56576U + 8192U + 8192U;
}
constexpr uint32_t ColdUb(uint32_t heads)
{
    return 131072U + 1024U + 512U + 1024U + 1024U + (256U/heads)*128U*2U +
        20480U + 8448U + 4096U + 40192U + 8192U;
}
static_assert(FastUb(32, 4) == 211456U);
static_assert(ColdUb(32) == 218112U);
static_assert(FastUb(32, 4) <= LIM_SIMT_USER_UB_BYTES);
static_assert(FastUb(64, 2) <= LIM_SIMT_USER_UB_BYTES);
static_assert(ColdUb(32) <= LIM_SIMT_USER_UB_BYTES);
static_assert(ColdUb(64) <= LIM_SIMT_USER_UB_BYTES);
static_assert(2U*65536U + 32768U + 96U + 128U <= LIM_SIMT_USER_UB_BYTES);
static_assert(4U*2048U*4U + vllm_a5_li_manage_c8_budget::MAX_CACHE_TOKENS +
              16384U*4U + 32U <= LIM_SIMT_USER_UB_BYTES); // Cold manager buffers.

int main()
{
    CheckBudgetBounds();
    assert(!lim_simt::ValidPair(-1, 0, 0, 262144, 12288));
    assert(!lim_simt::ValidPair(0, -1, 0, 262144, 12288));
    assert(!lim_simt::ValidPair(0, 0, -1, 262144, 12288));
    assert(!lim_simt::ValidPair(262144, 0, 0, 262144, 12288));
    for (const uint32_t count : {0U, 1U, 31U, 32U, 33U, 255U, 256U, 257U,
                                300U, 500U, 511U, 512U, 513U, 2048U, 2049U, 8192U}) {
        CheckVictimUnpack(count);
        for (const uint32_t budget : {8192U, 12288U, 16000U}) {
            CheckUpdates(count, count, budget);
            if (count) {
                CheckUpdates(count, 0, budget);
                CheckUpdates(count, count/2, budget);
                CheckUpdates(count, count-1, budget);
            }
        }
        CheckRepair(count);
    }
    CheckBatchedRepairs();
    std::cout << "SIMT_POSTPROCESS_HOST_OK victim_unpack=1 batched_repair=1 indexing=1 bounds=1 update_order=1 ub_plan=1 budgets=62\n";
}
