#ifndef VLLM_A5_LI_MANAGE_C8_BUDGET_H
#define VLLM_A5_LI_MANAGE_C8_BUDGET_H

#include <cstdint>

namespace vllm_a5_li_manage_c8_budget {

constexpr uint32_t MIN_CACHE_TOKENS = 8192U; // Four disjoint TopK2048 rows.
constexpr uint32_t MAX_CACHE_TOKENS = 16000U;
constexpr uint32_t CACHE_ALIGNMENT = 128U;

// Keep the existing {token18, slot14} encoding and its miss sentinels.
// At C=16000 the largest tail slot is 16255, below both reserved values.
static_assert(MAX_CACHE_TOKENS + 255U < 0x3ffeU);
static_assert(MAX_CACHE_TOKENS % CACHE_ALIGNMENT == 0U);

__aicore__ inline bool IsSupported(uint32_t budget)
{
    return budget >= MIN_CACHE_TOKENS && budget <= MAX_CACHE_TOKENS &&
        budget % CACHE_ALIGNMENT == 0U;
}

} // namespace vllm_a5_li_manage_c8_budget

#endif
