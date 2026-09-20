// SPDX-License-Identifier: Apache-2.0
#pragma once
// Source-tree and CANN package/staging layouts differ. Both use the SAME
// original SCFA metadata layout, without changing the old operator's ABI.
#if __has_include("../../vllm_a5_fused_copy_sfa_c8/op_kernel/vllm_a5_fused_copy_sfa_c8_tiling.h")
#include "../../vllm_a5_fused_copy_sfa_c8/op_kernel/vllm_a5_fused_copy_sfa_c8_tiling.h"
#else
#include "../vllm_a5_fused_copy_sfa_c8/vllm_a5_fused_copy_sfa_c8_tiling.h"
#endif
using VllmA5FusedCopySfaGroup4C8TilingData = VllmA5FusedCopySfaC8TilingData;
