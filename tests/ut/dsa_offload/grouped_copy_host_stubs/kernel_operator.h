// SPDX-License-Identifier: Apache-2.0
// CPU address/byte-count harness only; does NOT emulate Ascend event ordering.
#pragma once
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#define __aicore__
#define __gm__
#define ASCEND_IS_AIV (AscendC::hostIsAiv)
#define ASSERT(condition) assert(condition)
using GM_ADDR = uint8_t*;
namespace AscendC {
inline bool hostIsAiv = true;
inline uint32_t hostCore = 0;
inline uint64_t hostCopyCalls = 0;
inline uint32_t GetBlockIdx() { return hostCore; }
enum class TPosition { VECIN };
enum class HardEvent { MTE2_MTE3, MTE3_MTE2 };
using event_t = uint32_t;
template<class T> struct LocalTensor {
    T* data;
    LocalTensor operator[](int64_t offset) const { return {data + offset}; }
};
template<class T> struct GlobalTensor {
    T* data;
    void SetGlobalBuffer(T* pointer) { data = pointer; }
    T GetValue(int64_t offset) const { return data[offset]; }
    GlobalTensor operator[](int64_t offset) const { return {data + offset}; }
};
template<TPosition> struct TBuf {
    std::vector<uint64_t> storage;
    template<class T> LocalTensor<T> Get() { return {reinterpret_cast<T*>(storage.data())}; }
};
struct TPipe {
    uint32_t allocated = 0;
    template<TPosition P> void InitBuffer(TBuf<P>& buffer, uint32_t bytes) {
        allocated += bytes;
        buffer.storage.resize((bytes + 7) / 8);
    }
    template<HardEvent> event_t AllocEventID() { return 0; }
    template<HardEvent> void ReleaseEventID(event_t) {}
};
template<HardEvent> void SetFlag(event_t) {}
template<HardEvent> void WaitFlag(event_t) {}
template<bool> void SyncAll() {}
struct DataCopyExtParams { uint16_t blockCount; uint32_t blockLen, srcStride, dstStride, rsv; };
template<class T> struct DataCopyPadExtParams { bool pad; uint8_t left, right; T value; };
template<class T> void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src,
    DataCopyExtParams p, DataCopyPadExtParams<T>) {
    assert(p.blockCount == 1 && p.blockLen == 2624 && !p.srcStride && !p.dstStride);
    ++hostCopyCalls;
    std::memcpy(dst.data, src.data, p.blockLen);
}
template<class T> void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams p) {
    assert(p.blockCount == 1 && p.blockLen == 2624 && !p.srcStride && !p.dstStride);
    ++hostCopyCalls;
    std::memcpy(dst.data, src.data, p.blockLen);
}
} // namespace AscendC
