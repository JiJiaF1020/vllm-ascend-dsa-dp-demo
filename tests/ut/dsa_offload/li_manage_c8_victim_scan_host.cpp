// Portable hash/scan arithmetic check; not an A5 execution test.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <set>
#include <vector>

#define __aicore__
#include "vllm_a5_li_manage_c8_victim_scan.h"

using vllm_a5_li_manage_c8_victim_scan::HashVictimScanSeed;

int main()
{
    assert(HashVictimScanSeed(0U) == 0U);
    assert(HashVictimScanSeed(65540U) == 0x5649ea50U);
    assert(HashVictimScanSeed(65541U) == 0x3dc0110dU);
    assert(HashVictimScanSeed(131076U) == 0x0eec3350U);
    assert(HashVictimScanSeed(262143U) == 0x723088e6U);

    constexpr uint32_t chunkSize = 2048U;
    // Includes a final partial chunk and the 18-bit boundary.
    for (const uint32_t candidate : {8192U, 16512U, 65536U, 131072U, 261888U}) {
        const uint32_t chunks = (candidate + chunkSize - 1U) / chunkSize;
        std::set<uint32_t> starts;
        for (uint32_t tail = 4U; tail <= 256U; ++tail) {
            const uint32_t finalLen = candidate + tail;
            const uint32_t first = HashVictimScanSeed(finalLen) % chunks;
            starts.insert(first);
            // Full ring covers the durable prefix exactly once; never tail.
            std::vector<uint32_t> visits(candidate, 0U);
            for (uint32_t visit = 0U; visit < chunks; ++visit) {
                const uint32_t begin = ((first + visit) % chunks) * chunkSize;
                const uint32_t end = std::min(begin + chunkSize, candidate);
                for (uint32_t token = begin; token < end; ++token) {
                    ++visits[token];
                }
            }
            assert(std::all_of(visits.begin(), visits.end(),
                [](uint32_t n) { return n == 1U; }));
            assert(HashVictimScanSeed(finalLen) % chunks == first);
        }
        // We do NOT require adjacent lengths to avoid every hash collision.
        assert(starts.size() > 1U);
    }
    std::cout << "LI_MTP_VICTIM_SCAN_HOST_OK\n";
}
