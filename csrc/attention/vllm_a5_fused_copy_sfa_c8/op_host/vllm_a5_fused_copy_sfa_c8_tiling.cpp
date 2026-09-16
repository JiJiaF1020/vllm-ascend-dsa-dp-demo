#include <algorithm>
#include <cmath>
#include <cstdint>
#include "../op_kernel/vllm_a5_fused_copy_sfa_c8_tiling.h"
#include "../op_kernel/vllm_a5_fused_copy_sfa_c8_template_tiling_key.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext *c)
{
    if (!c || !c->GetPlatformInfo() || !c->GetAttrs()) return ge::GRAPH_FAILED;
    for (int i = 0; i < 13; ++i) {
        if (!c->GetInputShape(i) || !c->GetInputDesc(i)) return ge::GRAPH_FAILED;
        const auto dtype = c->GetInputDesc(i)->GetDataType();
        if ((i == 0 && dtype != ge::DT_BF16 && dtype != ge::DT_FLOAT16) ||
            (i > 0 && i < 3 && dtype != ge::DT_INT8) ||
            (i >= 3 && dtype != ge::DT_INT32)) return ge::GRAPH_FAILED;
    }
    const auto &q = c->GetInputShape(0)->GetStorageShape();
    const auto &h = c->GetInputShape(1)->GetStorageShape();
    const auto &d = c->GetInputShape(2)->GetStorageShape();
    const auto &ht = c->GetInputShape(3)->GetStorageShape();
    const auto &dt = c->GetInputShape(4)->GetStorageShape();
    const auto &counts = c->GetInputShape(12)->GetStorageShape();
    if (q.GetDimNum() != 3 || q.GetDim(0) <= 0 || q.GetDim(2) != 576 ||
        q.GetDim(1) <= 0 || (q.GetDim(1) > 64 && q.GetDim(1) != 128) ||
        counts.GetDimNum() != 1 || counts.GetDim(0) <= 0) return ge::GRAPH_FAILED;
    const auto b = counts.GetDim(0), t = q.GetDim(0);
    if (b > INT32_MAX / 4 || t > INT32_MAX || t < b || t > 4 * b ||
        ht.GetDimNum() != 2 || dt.GetDimNum() != 2 ||
        ht.GetDim(0) != b || dt.GetDim(0) != b || ht.GetDim(1) <= 0 || dt.GetDim(1) <= 0 ||
        ht.GetDim(1) > 2048 || dt.GetDim(1) > 2048) return ge::GRAPH_FAILED;
    for (const auto *cache : {&h, &d}) {
        if (cache->GetDimNum() != 4 || cache->GetDim(0) <= 0 ||
            cache->GetDim(0) > UINT32_MAX || cache->GetDim(1) != 128 ||
            cache->GetDim(2) != 1 || cache->GetDim(3) != 656) return ge::GRAPH_FAILED;
    }
    for (int i = 5; i <= 12; ++i) {
        const auto &s = c->GetInputShape(i)->GetStorageShape();
        const bool matrix = i == 5 || i == 6 || i == 10 || i == 11;
        const auto rows = i <= 7 ? t : b;
        if (s.GetDimNum() != (matrix ? 3 : 1) || s.GetDim(0) != rows ||
            (matrix && (s.GetDim(1) != 1 || s.GetDim(2) != (i < 7 ? 2048 : 16384))))
            return ge::GRAPH_FAILED;
    }
    const float *scale = c->GetAttrs()->GetAttrPointer<float>(0);
    if (!scale || !std::isfinite(*scale)) return ge::GRAPH_FAILED;
    platform_ascendc::PlatformAscendC platform(c->GetPlatformInfo());
    const uint32_t cores = platform.GetCoreNumAic();
    if (cores == 0 || platform.GetCoreNumAiv() < 2 * cores || (cores & 1)) return ge::GRAPH_FAILED;
    auto *td = c->GetTilingData<VllmA5FusedCopySfaC8TilingData>();
    if (!td || !c->GetWorkspaceSizes(1)) return ge::GRAPH_FAILED;
    *td = {};
    auto &p = td->baseParams;
    p.batchSize = b; p.qSeqSize = t; p.seqSize = ht.GetDim(1) * 128;
    p.nNumOfQInOneGroup = q.GetDim(1); p.sparseBlockCount = 2048;
    // LIM already enforced logical causality before mapping resident slots.
    p.sparseMode = 0; p.outputLayout = 1; p.blockSize = 128;
    p.maxBlockNumPerBatch = ht.GetDim(1); p.dSizeVInput = 656; p.scaleValue = *scale;
    td->singleCoreParams.usedCoreNum = cores;
    td->dramMaxBlocks = dt.GetDim(1);
    td->hbmPhysicalBlocks = h.GetDim(0); td->dramPhysicalBlocks = d.GetDim(0);
    c->SetBlockDim(cores);
    c->SetTilingKey(GET_TPL_TILING_KEY(q.GetDim(1) == 128));
    // G=128: three NZ KV tiles per pair of AICs. Otherwise only runtime sync workspace.
    c->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() +
        static_cast<uint64_t>(cores) * 3 * 128 * 576 * 2 + static_cast<uint64_t>(t) * 512 * 672;
    return ge::GRAPH_SUCCESS;
}
struct CompileInfo {};
static ge::graphStatus Parse(gert::TilingParseContext *) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(VllmA5FusedCopySfaC8).Tiling(Tiling).TilingParse<CompileInfo>(Parse);
}
