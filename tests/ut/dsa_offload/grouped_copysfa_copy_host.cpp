// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <iostream>
#include "fused_group4_c8_copy.h"

int main() {
    for (int32_t count : {0, 1, 2, 4, 127, 128, 129, 512, 1024, 1025, 8192, 16000, 16384}) {
        const uint32_t width = (count + 127) / 128 + 3;
        const uint32_t rows = width * 128;
        std::vector<uint8_t> hbm(rows * 2624, 0), dram(rows * 2624), expected(hbm);
        for (size_t i = 0; i < dram.size(); ++i) dram[i] = (i * 19 + 7) % 251;
        std::vector<int32_t> ht(2 * width), dt(2 * width), src(2 * 16384, -1), dst(src), counts{count, 0};
        for (uint32_t b = 0; b < 2; ++b) for (uint32_t i = 0; i < width; ++i) {
            ht[b * width + i] = width - 1 - i;
            dt[b * width + i] = (i + 1) % width;
        }
        for (int32_t i = 0; i < count; ++i) {
            src[i] = rows - 1 - i;
            dst[i] = (i + 127) % rows;
            const auto so = (dt[src[i] / 128] * 128 + src[i] % 128) * 2624;
            const auto offset = (ht[dst[i] / 128] * 128 + dst[i] % 128) * 2624;
            std::copy_n(dram.begin() + so, 2624, expected.begin() + offset);
        }
        VllmA5FusedCopySfaGroup4C8TilingData td{};
        td.baseParams.batchSize = 2;
        td.baseParams.maxBlockNumPerBatch = 4 * width; // SFA virtual pages, copy must undo x4.
        td.dramMaxBlocks = width;
        td.hbmPhysicalBlocks = td.dramPhysicalBlocks = width;
        td.singleCoreParams.usedCoreNum = 2;
        AscendC::hostCopyCalls = 0;
        auto run = [&] {
            AscendC::TPipe pipe;
            fused_group4_c8::UnionCopy(pipe, hbm.data(), dram.data(),
                reinterpret_cast<GM_ADDR>(ht.data()), reinterpret_cast<GM_ADDR>(dt.data()),
                reinterpret_cast<GM_ADDR>(src.data()), reinterpret_cast<GM_ADDR>(dst.data()),
                reinterpret_cast<GM_ADDR>(counts.data()), td);
            assert(pipe.allocated == 2 * 2624); // same UB prefix on both AIC and AIV.
        };
        AscendC::hostIsAiv = false;
        run();
        assert(hbm == std::vector<uint8_t>(hbm.size(), 0));
        AscendC::hostIsAiv = true;
        for (uint32_t core = 0; core < 4; ++core) {
            AscendC::hostCore = core;
            run();
        }
        assert(hbm == expected); // includes every byte of all four lanes, plus untouched rows.
        assert(AscendC::hostCopyCalls == static_cast<uint64_t>(count) * 2);
    }
    std::cout << "GROUP4_COPYSFA_COPY_HOST_OK\n";
}
