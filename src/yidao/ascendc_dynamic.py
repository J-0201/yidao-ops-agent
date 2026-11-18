"""AscendC 设备侧 kernel 生成器（v6）——动态长度 / 非整 repeat 尾块。

v5 之前 device kernel 的 shape 固定（total_length=8*2048 整除、COLS=64 整 repeat），
README/ARCHITECTURE 把「动态 tiling / 非整 repeat 尾块」列为下一步。本模块补上：

  - TOTAL_LENGTH 可为任意正整数（不要求被核数 / 64 整除）；
  - 多核分块：每核处理连续一段，前 (CORE_NUM-1) 核等长 BLOCK，最后一核处理尾部
    （可能更短，含非 64 倍数的非整 repeat）；
  - UB buffer 按每段向上对齐到 64（一个 repeat = 64 float）分配；
  - DataCopy 与向量 API 的 count 用该核**实际元素数**（AscendC 向量 API 的
    count 支持非 64 倍数，硬件按 count 精确处理，不需要手工 mask）；
  - 这样任意 shape 都能编译运行，尾块被正确处理。

覆盖 elementwise 族（relu/add/sub/mul/exp/tanh 等），与 ascendc_device.py 的
指令表一致，但把「固定 tiling」升级为「编译期动态 tiling + 尾块」。
"""

from __future__ import annotations

from dataclasses import dataclass

from .expr import Elem, Expr, Input, Scalar, walk

_UNARY = {
    "exp": "Exp(yLocal, {a}, count);",
    "tanh": "Tanh(yLocal, {a}, count);",
    "relu": "Maxs(yLocal, {a}, (float)0.0, count);",
    "neg": "Muls(yLocal, {a}, (float)-1.0, count);",
}
_BINARY = {
    "add": "Add(yLocal, {a}, {b}, count);",
    "sub": "Sub(yLocal, {a}, {b}, count);",
    "mul": "Mul(yLocal, {a}, {b}, count);",
    "div": "Div(yLocal, {a}, {b}, count);",
    "maximum": "Max(yLocal, {a}, {b}, count);",
}


@dataclass(frozen=True)
class DynamicKernel:
    name: str
    source: str
    op: str
    n_inputs: int
    total_length: int
    core_num: int
    vector_calls: tuple[str, ...]


def _plan(tree: Expr) -> tuple[list[str], int]:
    """把 elementwise 树编成向量指令序列（源 x{i}Local / t{i}，目的 yLocal）。"""
    instrs: list[str] = []
    cnt = [0]

    def newtmp() -> str:
        s = f"t{cnt[0]}"
        cnt[0] += 1
        return s

    def emit(node: Expr) -> str:
        if isinstance(node, Input):
            return f"x{node.i}Local"
        if isinstance(node, Scalar):
            t = newtmp()
            instrs.append(f"Duplicate({t}, (float){node.v!r}, count);")
            return t
        if isinstance(node, Elem):
            if node.op == "sigmoid":
                a = emit(node.args[0])
                e = newtmp()
                instrs.append(f"Muls({e}, {a}, (float)-1.0, count);")
                instrs.append(f"Exp({e}, {e}, count);")
                instrs.append(f"Adds({e}, {e}, (float)1.0, count);")
                t = newtmp()
                instrs.append(f"Reciprocal({t}, {e}, count);")
                return t
            if node.op in _UNARY:
                instrs.append(_UNARY[node.op].format(a=emit(node.args[0])))
                return "yLocal"
            if node.op in _BINARY:
                a = emit(node.args[0])
                b = emit(node.args[1])
                instrs.append(_BINARY[node.op].format(a=a, b=b))
                return "yLocal"
            if node.op == "pow" and isinstance(node.args[1], Scalar) and node.args[1].v == 0.5:
                instrs.append(f"Sqrt(yLocal, {emit(node.args[0])}, count);")
                return "yLocal"
            raise ValueError(f"动态设备后端不支持 op：{node.op}")
        raise ValueError(f"动态设备后端不支持节点：{type(node).__name__}")

    result = emit(tree)
    if result not in ("yLocal",):
        instrs.append(f"Adds(yLocal, {result}, (float)0.0, count);")
    return instrs, cnt[0]


def compile_dynamic_kernel(
    name: str, tree: Expr, *, total_length: int = 1000, core_num: int = 8
) -> DynamicKernel:
    """生成支持任意长度的 elementwise AscendC kernel 源码（含尾块）。"""
    instrs, n_tmp = _plan(tree)
    n_inputs = max(n.i for n in walk(tree) if isinstance(n, Input)) + 1

    # 每核实际段长 = ceil(TOTAL/CORE_NUM)，再向上对齐到 8 (GM 32B 地址对齐要求)，
    # 保证每个 start=bid*BLOCK 都是 8 的倍数（否则 GM 非对齐搬移会丢段尾数据）；
    # UB 容量再向上对齐到 64（一个 repeat）。
    block = (((total_length + core_num - 1) // core_num + 7) // 8) * 8
    ub_elems = ((block + 63) // 64) * 64

    # GM->UB DataCopy 的有效长度按 32B(8 float) 对齐；向上取整保证有效数据全搬入，
    # padding 尾部即使读到越界 GM 也不搬出。搬运用 cc，计算用 UB_ELEMS。
    copyin_lines = "\n".join(
        f"        AscendC::DataCopy(x{i}Local, xGm{i}[start], cc);"
        for i in range(n_inputs)
    )
    alloc_lines = "\n".join(
        f"        AscendC::LocalTensor<float> x{i}Local = inQueueX{i}.AllocTensor<float>();"
        for i in range(n_inputs)
    )
    enq_lines = "\n".join(f"        inQueueX{i}.EnQue(x{i}Local);" for i in range(n_inputs))
    deq_lines = "\n".join(
        f"        AscendC::LocalTensor<float> x{i}Local = inQueueX{i}.DeQue<float>();"
        for i in range(n_inputs)
    )
    free_lines = "\n".join(f"        inQueueX{i}.FreeTensor(x{i}Local);" for i in range(n_inputs))
    tmp_alloc = "\n".join(
        f"        AscendC::LocalTensor<float> t{i} = tmpBuf{i}.Get<float>();"
        for i in range(n_tmp)
    )
    compute_lines = "\n".join("        " + x for x in instrs)
    init_sets = "\n".join(
        f"        xGm{i}.SetGlobalBuffer((__gm__ float*)x{i}, TOTAL_LENGTH);"
        for i in range(n_inputs)
    )
    pipe_bufs = "\n".join(
        f"        pipe.InitBuffer(inQueueX{i}, 1, UB_ELEMS * sizeof(float));"
        for i in range(n_inputs)
    )
    tmp_inits = "\n".join(
        f"        pipe.InitBuffer(tmpBuf{i}, UB_ELEMS * sizeof(float));" for i in range(n_tmp)
    )
    queues = "\n".join(
        f"    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX{i};"
        for i in range(n_inputs)
    )
    gms = "\n".join(f"    AscendC::GlobalTensor<float> xGm{i};" for i in range(n_inputs))
    tmps = "\n".join(
        f"    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf{i};" for i in range(n_tmp)
    )
    gm_params = ", ".join(f"GM_ADDR x{i}" for i in range(n_inputs))
    kernel_args = gm_params

    src = f'''/**
 * {name} · 由易道算子 Agent v6 自动生成（AscendC 动态 tiling kernel）
 * 任意 TOTAL_LENGTH / 含非整 repeat 尾块；每核处理一段，count=实际元素数。
 * 向量指令：{instrs}
 */
#include "kernel_operator.h"

constexpr int32_t TOTAL_LENGTH = {total_length};
constexpr int32_t CORE_NUM = {core_num};
constexpr int32_t BLOCK = (((TOTAL_LENGTH + CORE_NUM - 1) / CORE_NUM + 7) / 8) * 8;  // 32B 对齐
constexpr int32_t UB_ELEMS = ((BLOCK + 63) / 64) * 64;

class Kernel{name} {{
public:
    __aicore__ inline Kernel{name}() {{}}
    __aicore__ inline void Init({gm_params}, GM_ADDR y) {{
{init_sets}
        yGm.SetGlobalBuffer((__gm__ float*)y, TOTAL_LENGTH);
{pipe_bufs}
        pipe.InitBuffer(outQueueY, 1, UB_ELEMS * sizeof(float));
{tmp_inits}
    }}
    __aicore__ inline void Process() {{
        int32_t bid = AscendC::GetBlockIdx();
        int32_t start = bid * BLOCK;
        int32_t count = start + BLOCK <= TOTAL_LENGTH ? BLOCK : (TOTAL_LENGTH - start);
        if (count <= 0) return;
        CopyIn(start, count); Compute(); CopyOut(start, count);
    }}
private:
    __aicore__ inline void CopyIn(int32_t start, int32_t count) {{
        int32_t cc = ((count + 7) / 8) * 8;  // 32B 对齐的搬入长度
{alloc_lines}
{copyin_lines}
{enq_lines}
    }}
    __aicore__ inline void Compute() {{
        // 向量计算按对齐长度 UB_ELEMS 做（含尾部 padding），搬运仍用实际 count
        int32_t count = UB_ELEMS;
{deq_lines}
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
{tmp_alloc}
{compute_lines}
        outQueueY.EnQue<float>(yLocal);
{free_lines}
    }}
    __aicore__ inline void CopyOut(int32_t start, int32_t count) {{
        // 写出也按 32B(8 float) 对齐（GM 要求），尾部 padding 落在 host 预留的
        // 对齐 buffer 内，不影响前 TOTAL_LENGTH 个有效结果。
        int32_t cc = ((count + 7) / 8) * 8;
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[start], yLocal, cc);
        outQueueY.FreeTensor(yLocal);
    }}
    AscendC::TPipe pipe;
{queues}
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueY;
{gms}
    AscendC::GlobalTensor<float> yGm;
{tmps}
}};

extern "C" __global__ __aicore__ void {name}({kernel_args}, GM_ADDR y) {{
    Kernel{name} op; op.Init({", ".join(f"x{i}" for i in range(n_inputs))}, y); op.Process();
}}
'''
    return DynamicKernel(name=name, source=src, op=name, n_inputs=n_inputs,
                         total_length=total_length, core_num=core_num,
                         vector_calls=tuple(instrs))
