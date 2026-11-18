"""AscendC 设备侧 kernel 生成器（v6）——MatMul 组合子（向量化点积累加）。

【诚实定位】这是【向量化实现】的 MatMul：用 AI Core 向量单元（Mul + WholeReduceSum）
逐行逐列做点积，不是 Cube/张量核的 Matmul 高阶 API（那需要 host 侧 TCubeTiling，
链路复杂）。它是真 AscendC 设备 kernel、真机 昇腾 NPU 运行、语义正确，
但【不使用 Cube 加速】——性能不是本项目目标，文档如实标注，知止。

【规避该 CANN 的两个坑（gather/动态 tiling 实测得出）】
  1. LocalTensor/GlobalTensor 的【变址】GetValue(i) 不可靠（恒读首元素）：
     故输出列 c 用 C++17 模板递归展开成【编译期常量】，GetValue(0) 读归约标量；
  2. GM DataCopy 地址/长度须 32B(8 float) 对齐：
     故 B 由 host 转置为 BT[N,K]，按整行(K=32，8 对齐) DataCopy；
     输出按整行(N=32) DataCopy。

接口：A[M,K]、BT[K 转置后的 N,K]、C[M,N]，均 float32；M=N=K=32，8 核。
C[r,c] = sum_k A[r,k] * BT[c,k]。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatMulKernel:
    name: str
    source: str
    m: int
    k: int
    n: int
    core_num: int


def compile_matmul_kernel(
    name: str, *, m: int = 32, k: int = 32, n: int = 32, core_num: int = 8
) -> MatMulKernel:
    """生成向量化 MatMul 设备 kernel（B 需 host 转置为 BT[N,K]）。"""
    if m % core_num != 0:
        raise ValueError("MatMul 要求 m 能被 core_num 整除")
    if k % 8 or n % 8:
        raise ValueError("MatMul 要求 k、n 为 8 的倍数（GM 32B 对齐）")
    rows_per_core = m // core_num

    src = f'''/**
 * {name} · 由易道算子 Agent v6 自动生成（AscendC 向量化 MatMul kernel）
 * C[r,c] = sum_k A[r,k]*BT[c,k]；向量 Mul + WholeReduceSum，模板展开规避变址读坑。
 * 注意：向量化实现，非 Cube 张量核加速。
 */
#include "kernel_operator.h"

constexpr int32_t M = {m};
constexpr int32_t K = {k};
constexpr int32_t N = {n};
constexpr int32_t CORE_NUM = {core_num};
constexpr int32_t ROWS_PER_CORE = M / CORE_NUM;

class Kernel{name} {{
public:
    __aicore__ inline Kernel{name}() {{}}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR bt, GM_ADDR c) {{
        aGm.SetGlobalBuffer((__gm__ float*)a, M * K);
        btGm.SetGlobalBuffer((__gm__ float*)bt, N * K);
        cGm.SetGlobalBuffer((__gm__ float*)c, M * N);
        pipe.InitBuffer(aBuf, K * sizeof(float));
        pipe.InitBuffer(bBuf, K * sizeof(float));
        pipe.InitBuffer(prodBuf, K * sizeof(float));
        pipe.InitBuffer(sumBuf, 64);
        pipe.InitBuffer(resBuf, N * sizeof(float));
    }}
    __aicore__ inline void Process() {{
        int32_t bid = AscendC::GetBlockIdx();
        AscendC::LocalTensor<float> aLocal = aBuf.Get<float>();
        AscendC::LocalTensor<float> bLocal = bBuf.Get<float>();
        AscendC::LocalTensor<float> prod = prodBuf.Get<float>();
        AscendC::LocalTensor<float> sum = sumBuf.Get<float>();
        AscendC::LocalTensor<float> res = resBuf.Get<float>();
        for (int32_t r = 0; r < ROWS_PER_CORE; r++) {{
            int32_t row = bid * ROWS_PER_CORE + r;
            AscendC::DataCopy(aLocal, aGm[row * K], K);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            DotCols<0>(aLocal, bLocal, prod, sum, res);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::DataCopy(cGm[row * N], res, N);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }}
    }}
private:
    // 模板展开列索引 C，规避 GetValue/setValue 变址不可靠问题
    template<int C>
    __aicore__ inline void DotCols(AscendC::LocalTensor<float>& aLocal,
                                   AscendC::LocalTensor<float>& bLocal,
                                   AscendC::LocalTensor<float>& prod,
                                   AscendC::LocalTensor<float>& sum,
                                   AscendC::LocalTensor<float>& res) {{
        if constexpr (C < N) {{
            AscendC::DataCopy(bLocal, btGm[C * K], K);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::Mul(prod, aLocal, bLocal, K);
            AscendC::SetFlag<AscendC::HardEvent::V_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_V>(EVENT_ID0);
            AscendC::WholeReduceSum<float>(sum, prod, (int32_t)K, 1, 1, 1, 1);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            res.SetValue(C, sum.GetValue(0));
            DotCols<C + 1>(aLocal, bLocal, prod, sum, res);
        }}
    }}
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> prodBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> resBuf;
    AscendC::GlobalTensor<float> aGm;
    AscendC::GlobalTensor<float> btGm;
    AscendC::GlobalTensor<float> cGm;
}};

extern "C" __global__ __aicore__ void {name}(GM_ADDR a, GM_ADDR bt, GM_ADDR c) {{
    Kernel{name} op; op.Init(a, bt, c); op.Process();
}}
'''
    return MatMulKernel(name=name, source=src, m=m, k=k, n=n, core_num=core_num)
