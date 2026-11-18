"""AscendC 设备侧 kernel 生成器（真机范式，v4 hardware-real 层）。

与 ascendc.py（桩/标量求值后端）不同，本模块产出的是能直接被 ccec 编译、
在 AI Core 上运行的高性能 kernel：
  - 多核分块：GetBlockIdx() + BLOCK_LENGTH
  - 三段式流水：CopyIn(DataCopy GM->UB) / Compute(Vector API) / CopyOut
  - 双缓冲：TQue<VECIN/VECOUT, 2> + TILE_NUM
  - 逐元素向量 API：Maxs/Add/Sub/Mul/Div/Exp/Reciprocal...

当前覆盖 elementwise 族。新增算子 = 在指令表里声明，不改引擎。
"""

from __future__ import annotations

from dataclasses import dataclass

from .expr import Elem, Expr, Input, Scalar, walk


@dataclass(frozen=True)
class DeviceKernel:
    name: str
    source: str
    op: str
    n_inputs: int
    vector_calls: tuple[str, ...]


# 逐元素运算 -> AscendC 向量 API；{d}目的 {a}{b}源 n=count
_UNARY = {
    "exp": "Exp({d}, {a}, n);",
    "tanh": "Tanh({d}, {a}, n);",
    "relu": "Maxs({d}, {a}, (float)0.0, n);",
    "neg": "Muls({d}, {a}, (float)-1.0, n);",
}
_BINARY = {
    "add": "Add({d}, {a}, {b}, n);",
    "sub": "Sub({d}, {a}, {b}, n);",
    "mul": "Mul({d}, {a}, {b}, n);",
    "div": "Div({d}, {a}, {b}, n);",
    "maximum": "Max({d}, {a}, {b}, n);",
}


def _plan(tree: Expr) -> tuple[list[str], int]:
    """把表达式树编译成向量指令序列。源用 x{i}Local / t{i}，目的用 yLocal / t{i}。"""
    instrs: list[str] = []
    cnt = [0]

    def newtmp() -> str:
        s = f"t{cnt[0]}"
        cnt[0] += 1
        return s

    # 返回"承载该子树结果的 LocalTensor 名字"
    def emit(node: Expr) -> str:
        if isinstance(node, Input):
            return f"x{node.i}Local"
        if isinstance(node, Scalar):
            # 常量张量：先用 Duplicate 填充分量（随后参与运算）
            t = newtmp()
            instrs.append(f"Duplicate({t}, (float){node.v!r}, n);")
            return t
        if isinstance(node, Elem):
            if node.op == "sigmoid":
                # sigmoid(x) = 1 / (1 + exp(-x))
                # 【精度修复 v8】绝不用 Reciprocal：
                #   AscendC Reciprocal 底层是 vrec 倒数【近似】指令，内部 fp16 精度，
                #   1/1 都会得到 0.998047(=2043/2048) 的量化网格值，MERE 卡在 ~1e-3，
                #   超 CANN-Bench 2^-13 阈值约 8 倍（易道 v7 与 同类工具链 共同踩坑）。
                # 改用 Div：底层 vdiv 是硬件 fp32 除法，精度 ~1e-7，MERE 可压到 2^-13 以下。
                a = emit(node.args[0])
                e = newtmp()
                instrs.append(f"Muls({e}, {a}, (float)-1.0, n);")   # -x
                instrs.append(f"Exp({e}, {e}, n);")                 # exp(-x)  vexp fp32
                instrs.append(f"Adds({e}, {e}, (float)1.0, n);")    # 1+exp(-x)
                one = newtmp()
                instrs.append(f"Duplicate({one}, (float)1.0, n);")  # 分子常量 1（fp32）
                t = newtmp()
                instrs.append(f"Div({t}, {one}, {e}, n);")          # 1/(...)  vdiv fp32 精确除法
                return t
            if node.op in _UNARY:
                a = emit(node.args[0])
                d = "yLocal" if _is_root(node) else newtmp()
                instrs.append(_UNARY[node.op].format(d=d, a=a))
                return d
            if node.op in _BINARY:
                a = emit(node.args[0])
                b = emit(node.args[1])
                d = "yLocal"
                instrs.append(_BINARY[node.op].format(d=d, a=a, b=b))
                return d
            if node.op == "pow":
                # pow(x, c) 常量指数：用 Muls 近似仅支持 c=0.5（sqrt）等特例
                a = emit(node.args[0])
                if isinstance(node.args[1], Scalar) and node.args[1].v == 0.5:
                    instrs.append(f"Sqrt(yLocal, {a}, n);")
                    return "yLocal"
                raise ValueError("设备后端 pow 仅支持 0.5 (Sqrt)")
            raise ValueError(f"设备后端不支持 op：{node.op}")
        raise ValueError(f"设备后端不支持节点：{type(node).__name__}")

    # 根节点的单目运算目的应为 yLocal
    root_holder = [id(tree)]

    def _is_root(node: Expr) -> bool:
        return id(node) == root_holder[0]

    result = emit(tree)
    if result != "yLocal":
        instrs.append(f"Adds(yLocal, {result}, (float)0.0, n);")
    return instrs, cnt[0]


def compile_device_kernel(name: str, tree: Expr, total_length: int = 8 * 2048,
                          use_core_num: int = 8, tile_num: int = 4,
                          buffer_num: int = 2) -> DeviceKernel:
    """生成一段可被 ccec 编译的 elementwise AscendC kernel 源码。

    tile_num 默认 4（v9 性能工程）：昇腾 NPU 实测 TILE_LENGTH=256（8 次循环）
    比原来的 TILE_LENGTH=128（16 次循环）快约 30%，且稳定性相当（CV 1.7%）。
    这与业界通用模板的 tile 粒度一致——性能差距的根因是 tile 大小，
    而非 DataCopyPad/Reciprocal 等指令选择。更大 tile（TILE512）均值更快但
    抖动大（CV 8.9%），故不取。仅影响 tiling，不影响向量指令与精度语义。
    """
    instrs, n_tmp = _plan(tree)
    n_inputs = max(n.i for n in walk(tree) if isinstance(n, Input)) + 1
    # 统一 count 用 TILE_LENGTH（替换占位 n）
    vec = [i.replace(", n)", ", TILE_LENGTH)") for i in instrs]

    copyin_lines = "\n".join(
        f"        AscendC::LocalTensor<float> x{i}Local = inQueueX{i}.AllocTensor<float>();\n"
        f"        AscendC::DataCopy(x{i}Local, xGm{i}[progress * TILE_LENGTH], TILE_LENGTH);\n"
        f"        inQueueX{i}.EnQue(x{i}Local);"
        for i in range(n_inputs)
    )
    deq_lines = "\n".join(
        f"        AscendC::LocalTensor<float> x{i}Local = inQueueX{i}.DeQue<float>();"
        for i in range(n_inputs)
    )
    free_lines = "\n".join(
        f"        inQueueX{i}.FreeTensor(x{i}Local);" for i in range(n_inputs)
    )
    tmp_alloc = "\n".join(
        f"        AscendC::LocalTensor<float> t{i} = tmpBuf{i}.Get<float>();"
        for i in range(n_tmp)
    )
    compute_lines = "\n".join("        " + x for x in vec)

    gm_params = ", ".join(f"GM_ADDR x{i}" for i in range(n_inputs))
    init_sets = "\n".join(
        f"        xGm{i}.SetGlobalBuffer((__gm__ float*)x{i} + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);"
        for i in range(n_inputs)
    )
    pipe_bufs = "\n".join(
        f"        pipe.InitBuffer(inQueueX{i}, BUFFER_NUM, TILE_LENGTH * sizeof(float));"
        for i in range(n_inputs)
    )
    tmp_inits = "\n".join(
        f"        pipe.InitBuffer(tmpBuf{i}, TILE_LENGTH * sizeof(float));" for i in range(n_tmp)
    )
    queues = "\n".join(
        f"    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX{i};"
        for i in range(n_inputs)
    )
    gms = "\n".join(f"    AscendC::GlobalTensor<float> xGm{i};" for i in range(n_inputs))
    tmps = "\n".join(
        f"    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf{i};" for i in range(n_tmp)
    )
    kernel_args = ", ".join(f"GM_ADDR x{i}" for i in range(n_inputs))

    src = f'''/**
 * {name} · 由易道算子 Agent v4 自动生成（AscendC 设备侧 kernel）
 * 多核 tiling + 三段式流水 + 双缓冲；可由 ccec 编译在 AI Core 运行。
 * 向量指令序列：{vec}
 */
#include "kernel_operator.h"

constexpr int32_t TOTAL_LENGTH = {total_length};
constexpr int32_t USE_CORE_NUM = {use_core_num};
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;
constexpr int32_t TILE_NUM = {tile_num};
constexpr int32_t BUFFER_NUM = {buffer_num};
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM;

class Kernel{name} {{
public:
    __aicore__ inline Kernel{name}() {{}}
    __aicore__ inline void Init({gm_params}, GM_ADDR y) {{
{init_sets}
        yGm.SetGlobalBuffer((__gm__ float*)y + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
{pipe_bufs}
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(float));
{tmp_inits}
    }}
    __aicore__ inline void Process() {{
        int32_t loopCount = TILE_NUM * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {{ CopyIn(i); Compute(i); CopyOut(i); }}
    }}
private:
    __aicore__ inline void CopyIn(int32_t progress) {{
{copyin_lines}
    }}
    __aicore__ inline void Compute(int32_t progress) {{
{deq_lines}
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
{tmp_alloc}
{compute_lines}
        outQueueY.EnQue<float>(yLocal);
{free_lines}
    }}
    __aicore__ inline void CopyOut(int32_t progress) {{
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[progress * TILE_LENGTH], yLocal, TILE_LENGTH);
        outQueueY.FreeTensor(yLocal);
    }}
    AscendC::TPipe pipe;
{queues}
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
{gms}
    AscendC::GlobalTensor<float> yGm;
{tmps}
}};

extern "C" __global__ __aicore__ void {name}({kernel_args}, GM_ADDR y) {{
    Kernel{name} op;
    op.Init({", ".join(f"x{i}" for i in range(n_inputs))}, y);
    op.Process();
}}
'''
    return DeviceKernel(name=name, source=src, op=name, n_inputs=n_inputs,
                        vector_calls=tuple(vec))
