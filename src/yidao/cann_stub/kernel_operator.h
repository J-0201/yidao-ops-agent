#ifndef YIDAO_CANN_STUB_H
#define YIDAO_CANN_STUB_H
// ============================================================================
// CANN / AscendC 桩头文件（CPU 语义实现）
//
// 目的：让「真正的 AscendC kernel 源码」在没有昇腾 NPU 的 x86 沙箱里，
//       也能被 g++ 编译并跑出数值，从而与独立 oracle 对拍。
//
// 这不是伪造后端：生成的 kernel 源码用的是真实 AscendC 编程模型
// （GM_ADDR / __global__ / GlobalTensor / LocalTensor / TPipe / TQue /
//  DataCopy / Init-Process / CopyIn-Compute-CopyOut）。把本桩头文件换成
// 真机 ${ASCEND_HOME}/include 的 kernel_operator.h，同一份源码即可由 ccec
// 编译烧录。本桩仅把这些 API 等价实现为 host 内存操作，供语义验证。
// ============================================================================
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>

using GM_ADDR = void*;
#define __gm__
#define __aicore__
#define __global__
#define __host__

namespace AscendC {

enum class QuePosition { VECIN, VECOUT, VECCALC };

template <typename T>
struct GlobalTensor {
    T* ptr = nullptr;
    uint32_t size_ = 0;
    __aicore__ inline void SetGlobalBuffer(__gm__ T* p, uint32_t n) { ptr = p; size_ = n; }
    __aicore__ inline T& operator[](uint32_t i) { return ptr[i]; }
    __aicore__ inline const T& operator[](uint32_t i) const { return ptr[i]; }
};

template <typename T>
struct LocalTensor {
    std::vector<T> buf;
    __aicore__ inline void SetSize(uint32_t n) { buf.resize(n); }
    __aicore__ inline uint32_t GetSize() const { return (uint32_t)buf.size(); }
    __aicore__ inline T& operator[](uint32_t i) { return buf[i]; }
    __aicore__ inline const T& operator[](uint32_t i) const { return buf[i]; }
};

template <QuePosition Q, uint32_t D>
struct TQue {
    uint32_t buf_size = 0;      // 元素数（由 TPipe::InitBuffer 的字节数换算）
    LocalTensor<float> slot;    // 桩：单槽持有真实存储（EnQue 后持久，供 Compute 读）

    template <typename T>
    __aicore__ inline LocalTensor<T> AllocTensor() {
        LocalTensor<T> t; t.SetSize(buf_size); return t;
    }
    __aicore__ inline void EnQue(const LocalTensor<float>& t) { slot = t; }
    template <typename T>
    __aicore__ inline LocalTensor<T> DeQue() {
        LocalTensor<T> t; t.buf = slot.buf; return t;
    }
    __aicore__ inline void FreeTensor(const LocalTensor<float>&) {}
};

struct TPipe {
    template <QuePosition Q, uint32_t D>
    __aicore__ inline void InitBuffer(TQue<Q, D>& q, uint32_t num, uint32_t bytes) {
        q.buf_size = bytes / sizeof(float);
    }
};

template <typename T>
__aicore__ inline void DataCopy(LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t n) {
    dst.SetSize(n);
    if (n) std::memcpy(dst.buf.data(), src.ptr, (size_t)n * sizeof(T));
}
template <typename T>
__aicore__ inline void DataCopy(GlobalTensor<T>& dst, const LocalTensor<T>& src, uint32_t n) {
    if (n) std::memcpy(dst.ptr, src.buf.data(), (size_t)n * sizeof(T));
}

}  // namespace AscendC
using namespace AscendC;
#endif
