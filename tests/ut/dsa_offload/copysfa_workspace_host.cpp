// Host-only self-check: g++ -std=c++17 -pthread tests/_copysfa_workspace_cache_test.cpp -o /tmp/copysfa_workspace_check
#include "../../../csrc/attention/vllm_a5_fused_copy_sfa_c8/copysfa_workspace_cache.h"
#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

int main() {
  using Buffer = std::shared_ptr<std::vector<unsigned char>>;
  vllm_dsa_a5::CopySfaWorkspaceCache<Buffer> cache;
  int allocations = 0;
  auto allocate = [&](uint64_t bytes) {
    ++allocations;
    return std::make_shared<std::vector<unsigned char>>(bytes);
  };
  assert(!cache.acquire(0, 1, 0, allocate).tensor && allocations == 0);
  auto first = cache.acquire(0, 1, 128, allocate);
  assert(first.allocated && first.capacity == 128 && allocations == 1);
  auto same = cache.acquire(0, 1, 128, allocate);
  auto smaller = cache.acquire(0, 1, 64, allocate);
  assert(!same.allocated && !smaller.allocated && smaller.capacity == 128);
  assert(same.tensor == first.tensor && smaller.tensor == first.tensor && allocations == 1);
  (*first.tensor)[0] = 42;
  auto grown = cache.acquire(0, 1, 512, allocate);
  assert(grown.allocated && grown.capacity == 512 && grown.tensor != first.tensor);
  assert((*first.tensor)[0] == 42);  // queued launch still owns the old buffer
  assert(cache.acquire(0, 1, 128, allocate).tensor == grown.tensor);
  assert(cache.acquire(0, 2, 128, allocate).tensor != grown.tensor);
  assert(cache.acquire(1, 1, 128, allocate).tensor != grown.tensor);
  try {
    cache.acquire(0, 1, 1024, [](uint64_t) -> Buffer { throw std::bad_alloc(); });
    assert(false);
  } catch (const std::bad_alloc&) {}
  assert(cache.acquire(0, 1, 512, allocate).tensor == grown.tensor);

  // Concurrent host callers on the same stream share one allocation.
  Buffer received[8];
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&, i] { received[i] = cache.acquire(0, 3, 256, allocate).tensor; });
  }
  for (auto& thread : threads) thread.join();
  for (auto& tensor : received) assert(tensor == received[0]);
  assert(allocations == 5);
  std::cout << "COPYSFA_WORKSPACE_CACHE_OK\n";
}
