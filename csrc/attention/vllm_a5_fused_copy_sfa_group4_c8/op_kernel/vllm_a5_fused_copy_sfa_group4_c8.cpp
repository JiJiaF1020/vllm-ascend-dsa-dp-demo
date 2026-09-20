// One Ascend950 MIX kernel: full four-lane union copy, barrier, owner SFA.
// SFA addresses are virtual PA indices; copy addresses remain logical.
// All group data is already in HBM before SFA; do not compile DRAM prefetch.
#ifndef VLLM_A5_DSA_COPYSFA_PREFETCH
#define VLLM_A5_DSA_COPYSFA_PREFETCH 0
#endif

#include "kernel_operator.h"
#include "vllm_a5_fused_copy_sfa_group4_c8_tiling.h"
#include "vllm_a5_fused_copy_sfa_group4_c8_template_tiling_key.h"
#include "fused_group4_c8_copy.h"
using KvQuantSparseFlashAttentionTilingDataMla = VllmA5FusedCopySfaGroup4C8TilingData;
// The original math and its disabled source-aware helpers are reused verbatim.
#if __has_include("../../vllm_a5_fused_copy_sfa_c8/op_kernel/fused_c8_copy.h")
#include "../../vllm_a5_fused_copy_sfa_c8/op_kernel/fused_c8_copy.h"
#include "../../vllm_a5_fused_copy_sfa_c8/op_kernel/fused_c8/arch35/kv_quant_sparse_flash_attention_kernel_mla.h"
#else
#include "../vllm_a5_fused_copy_sfa_c8/fused_c8_copy.h"
#include "../vllm_a5_fused_copy_sfa_c8/fused_c8/arch35/kv_quant_sparse_flash_attention_kernel_mla.h"
#endif

template<bool SPLIT_HEADS>
__global__ __aicore__ void vllm_a5_fused_copy_sfa_group4_c8(
    GM_ADDR query, GM_ADDR hbmKvBytes, GM_ADDR dramKvBytes,
    GM_ADDR hbmBlockTable, GM_ADDR dramBlockTable, GM_ADDR topkSlots,
    GM_ADDR topkIndex, GM_ADDR perQueryMissCounts,
    GM_ADDR actualSeqLengthsQuery, GM_ADDR residentSeqLengths,
    GM_ADDR copySrcIds, GM_ADDR copyDstSlots, GM_ADDR copyCounts,
    GM_ADDR attentionSlots, GM_ADDR attentionBlockTable,
    GM_ADDR hbmKvAlias, GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(VllmA5FusedCopySfaGroup4C8TilingData);
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe pipe;
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    // Validate MTP counts/ends, but ALWAYS persist the complete union, even
    // for counts <= 1024 or entries absent from the owner's attention subset.
    (void)fused_c8::RequiresUnionCopy(copyCounts, perQueryMissCounts,
        actualSeqLengthsQuery, tilingData);
    fused_group4_c8::UnionCopy(pipe, hbmKvBytes, dramKvBytes, hbmBlockTable,
        dramBlockTable, copySrcIds, copyDstSlots, copyCounts, tilingData);
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
    op.Init(query, hbmKvBytes, hbmKvBytes, attentionSlots, nullptr, nullptr,
        attentionBlockTable, actualSeqLengthsQuery, residentSeqLengths, attentionOut,
        userWorkspace, &tilingData, &pipe);
    op.InitSourceAwareGather(dramKvBytes, dramBlockTable, attentionSlots,
        topkIndex, perQueryMissCounts, &tilingData, false, userWorkspace);
    op.Process();
}
