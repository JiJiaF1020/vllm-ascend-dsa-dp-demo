// One Ascend950 MIX kernel: source-aware C8 gather + official SCFA math.
// build.sh selects this value in its generated copy; direct builds default to on.
#ifndef VLLM_A5_DSA_COPYSFA_PREFETCH
#define VLLM_A5_DSA_COPYSFA_PREFETCH 1
#endif

#include "kernel_operator.h"
#include "vllm_a5_fused_copy_sfa_c8_tiling.h"
#include "vllm_a5_fused_copy_sfa_c8_template_tiling_key.h"
#include "fused_c8_copy.h"
using KvQuantSparseFlashAttentionTilingDataMla = VllmA5FusedCopySfaC8TilingData;
#include "fused_c8/arch35/kv_quant_sparse_flash_attention_kernel_mla.h"

template<bool SPLIT_HEADS>
__global__ __aicore__ void vllm_a5_fused_copy_sfa_c8(
    GM_ADDR query, GM_ADDR hbmKvBytes, GM_ADDR dramKvBytes,
    GM_ADDR hbmBlockTable, GM_ADDR dramBlockTable, GM_ADDR sparseAndTailSlots,
    GM_ADDR sparseAndTailSrcIds, GM_ADDR perQueryMissCounts,
    GM_ADDR actualSeqLengthsQuery, GM_ADDR residentSeqLengths,
    GM_ADDR copySrcIds, GM_ADDR copyDstSlots, GM_ADDR copyCounts,
    GM_ADDR hbmKvAlias, GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(VllmA5FusedCopySfaC8TilingData);
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe pipe;
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    const bool fallback = fused_c8::RequiresUnionCopy(copyCounts, perQueryMissCounts,
        actualSeqLengthsQuery, tilingData);
    if (fallback) {
        fused_c8::UnionCopy(pipe, hbmKvBytes, dramKvBytes, hbmBlockTable, dramBlockTable,
            copySrcIds, copyDstSlots, copyCounts, tilingData);
    }
    // int8 is an opaque byte view, NOT the numerical latent dtype.
    using Q = typename std::conditional<ORIG_DTYPE_QUERY == DT_BF16, bfloat16_t, half>::type;
    using Cube = typename std::conditional<g_coreType == AscendC::AIC,
        BaseApi::QSFAMatmulService<Q, fp8_e4m3fn_t, float, Q, false, true,
            QSFA_LAYOUT::TND, QSFA_LAYOUT::PA_BSND, QSFATemplateMode::SCFA_TEMPLATE_MODE, SPLIT_HEADS>,
        BaseApi::QSFAMatmulServiceDummy<Q, fp8_e4m3fn_t, float, Q, false, true,
            QSFA_LAYOUT::TND, QSFA_LAYOUT::PA_BSND, QSFATemplateMode::SCFA_TEMPLATE_MODE, SPLIT_HEADS>>::type;
    using Vec = typename std::conditional<g_coreType == AscendC::AIC,
        BaseApi::QSFAVectorServiceDummy<Q, fp8_e4m3fn_t, float, Q, false, true,
            QSFA_LAYOUT::TND, QSFA_LAYOUT::PA_BSND, QSFATemplateMode::SCFA_TEMPLATE_MODE, SPLIT_HEADS>,
        BaseApi::QSFAVectorService<Q, fp8_e4m3fn_t, float, Q, false, true,
            QSFA_LAYOUT::TND, QSFA_LAYOUT::PA_BSND, QSFATemplateMode::SCFA_TEMPLATE_MODE, SPLIT_HEADS>>::type;
    BaseApi::KvQuantSparseFlashAttentionMla<Cube, Vec> op;
    op.Init(query, hbmKvBytes, hbmKvBytes, sparseAndTailSlots, nullptr, nullptr,
        hbmBlockTable, actualSeqLengthsQuery, residentSeqLengths, attentionOut,
        userWorkspace, &tilingData, &pipe);
    op.InitSourceAwareGather(dramKvBytes, dramBlockTable, sparseAndTailSlots,
        sparseAndTailSrcIds, perQueryMissCounts, &tilingData, !fallback,
        userWorkspace +
            static_cast<uint64_t>(tilingData.singleCoreParams.usedCoreNum) * 3 * 128 * 576 * 2);
    op.Process();
}
