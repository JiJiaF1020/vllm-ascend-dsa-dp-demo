#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <utility>

namespace vllm_dsa_a5 {

// Eager only. Graph-capture allocations must stay in the graph's private pool.
// Buffer is an owning handle: a queued launch retains its copy during growth.
template <typename Buffer>
class CopySfaWorkspaceCache {
 public:
  struct Allocation {
    Buffer tensor;
    uint64_t capacity = 0;
    bool allocated = false;
  };

  template <typename Allocate>
  Allocation acquire(int device, uintptr_t stream, uint64_t bytes, Allocate allocate) {
    if (bytes == 0) return {};
    std::lock_guard<std::mutex> lock(mutex_);
    auto& entry = entries_[{device, stream}];
    const bool grow = entry.capacity < bytes;
    if (grow) {
      auto tensor = allocate(bytes);
      entry.tensor = std::move(tensor);
      entry.capacity = bytes;
    }
    return {entry.tensor, entry.capacity, grow};
  }

 private:
  struct Entry {
    Buffer tensor;
    uint64_t capacity = 0;
  };
  std::mutex mutex_;
  std::map<std::pair<int, uintptr_t>, Entry> entries_;
};

}  // namespace vllm_dsa_a5
