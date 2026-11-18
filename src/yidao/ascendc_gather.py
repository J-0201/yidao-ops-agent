"""AscendC 设备侧 kernel 生成器（v6）——Gather 组合子：按索引收集行。

语义：y[k, c] = data[idx[k], c]，data 为 [ROWS, COLS] float32，
idx 为 [K] int32，输出 y 为 [K, COLS] float32。

设备范式（昇腾 NPU / CANN9 可编译）：
  - 多核分块：K 个查询行分给 CORE_NUM 个核，每核 ROWS_PER_CORE=K/CORE_NUM 行；
  - CopyIn：先 DataCopy 本核负责的 idx 段到 UB（LocalTensor<int32_t>），
    逐行 GetValue 取索引 j = idx[k]；
  - 再 DataCopy data[j*COLS] 的整行到 UB；
  - CopyOut：把该行搬到 y[k*COLS]。
  - 索引越界由调用方在 Python 侧拒绝（设备 kernel 假定 idx 合法）。

这是与 elementwise / 归约族互补的第三类设备范式（内存收集/查表），
补齐 v5 文档中「Gather 设备侧实现未覆盖」的下一步。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GatherKernel:
    name: str
    source: str
    rows: int            # data 行数
    cols: int            # 每行元素数
    k: int               # 查询索引数
    core_num: int


def compile_gather_kernel(
    name: str, *, rows: int = 64, cols: int = 64, k: int = 64, core_num: int = 8
) -> GatherKernel:
    """生成按行 Gather 的 AscendC 设备 kernel 源码。

    要求 k 能被 core_num 整除；cols 为 64 的倍数（整 repeat）。
    """
    if k % core_num != 0:
        raise ValueError("Gather 要求 k 能被 core_num 整除")
    if cols % 64 != 0:
        raise ValueError("Gather 要求 cols 为 64 的倍数")
    k_per_core = k // core_num

    src = f'''/**
 * {name} · 由易道算子 Agent v6 自动生成（AscendC Gather 设备 kernel）
 * y[k,c] = data[idx[k], c]；多核分块 + 索引查表 + 整行搬运
 */
#include "kernel_operator.h"

constexpr int32_t ROWS = {rows};
constexpr int32_t COLS = {cols};
constexpr int32_t K = {k};
constexpr int32_t CORE_NUM = {core_num};
constexpr int32_t K_PER_CORE = K / CORE_NUM;

class Kernel{name} {{
public:
    __aicore__ inline Kernel{name}() {{}}
    __aicore__ inline void Init(GM_ADDR data, GM_ADDR idx, GM_ADDR y) {{
        int32_t bid = AscendC::GetBlockIdx();
        dataGm.SetGlobalBuffer((__gm__ float*)data, ROWS * COLS);
        // 索引以 float32 传输（该 CANN 版本对 int32 GlobalTensor 读取不可靠）
        idxGm.SetGlobalBuffer((__gm__ float*)idx + bid * K_PER_CORE, K_PER_CORE);
        yGm.SetGlobalBuffer((__gm__ float*)y + bid * K_PER_CORE * COLS, K_PER_CORE * COLS);
        pipe.InitBuffer(idxBuf, K_PER_CORE * sizeof(float));
        pipe.InitBuffer(rowBuf, COLS * sizeof(float));
    }}
    __aicore__ inline void Process() {{
        AscendC::LocalTensor<float> idxLocal = idxBuf.Get<float>();
        AscendC::LocalTensor<float> row = rowBuf.Get<float>();
        AscendC::DataCopy(idxLocal, idxGm, K_PER_CORE);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        GatherRows<0>(idxLocal, row);
    }}
private:
    // 用模板递归把索引 I 变成编译期常量：该 CANN 上 GetValue(变址) 不可靠，
    // 常量索引 GetValue(I) 已由 reduce kernel 验证可靠。if constexpr 终止递归。
    template<int I>
    __aicore__ inline void GatherRows(AscendC::LocalTensor<float>& idxLocal,
                                      AscendC::LocalTensor<float>& row) {{
        if constexpr (I < K_PER_CORE) {{
            int32_t j = (int32_t)idxLocal.GetValue(I);
            AscendC::DataCopy(row, dataGm[j * COLS], COLS);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::DataCopy(yGm[I * COLS], row, COLS);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            GatherRows<I + 1>(idxLocal, row);
        }}
    }}
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> idxBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowBuf;
    AscendC::GlobalTensor<float> dataGm;
    AscendC::GlobalTensor<float> idxGm;
    AscendC::GlobalTensor<float> yGm;
}};

extern "C" __global__ __aicore__ void {name}(GM_ADDR data, GM_ADDR idx, GM_ADDR y) {{
    Kernel{name} op; op.Init(data, idx, y); op.Process();
}}
'''
    return GatherKernel(name=name, source=src, rows=rows, cols=cols, k=k,
                        core_num=core_num)
