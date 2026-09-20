# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Reuse the original CopySFA math in both source and packaged builds.
set(vllm_a5_fused_copy_sfa_group4_c8_depends
    "attention/vllm_a5_fused_copy_sfa_c8" CACHE INTERNAL "Group4 CopySFA dependency" FORCE)
add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE vllm_a5_fused_copy_sfa_group4_c8_def.cpp)
endif()
add_ops_compile_options(
    OP_NAME VllmA5FusedCopySfaGroup4C8
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        --op_relocatable_kernel_binary=true
)
if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE vllm_a5_fused_copy_sfa_group4_c8 ACLNNTYPE aclnn)
endif()
