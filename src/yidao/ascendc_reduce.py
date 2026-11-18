"""AscendC 设备侧 kernel 生成器 —— 行级归约族（reduce / softmax / layernorm）。

与 ascendc_device.py（逐元素族）互补：本模块处理含 Reduce 节点的表达式树。
核心是把一棵树编译成「逐行程序」：一行 COLS 个元素载入 UB 后，按后序执行
一串向量指令，Reduce 用 WholeReduceSum/Max/Min 归约成标量，标量参与后续
运算时用 Duplicate 广播回整行，最后写回一行（或每行一个标量）。

范式（真机 910B / CANN9 已验证）：
  - ROWS x COLS 输入，COLS=64=1 repeat，8 核各处理 ROWS/8 行
  - DataCopy 整行 -> Compute(向量指令) -> DataCopy 整行/标量
  - WholeReduceSum/Max/Min(dst, src, COLS, 1, 1, 1, 1) 把一整行归约到 dst[0]
  - GetValue(0) 取标量 + Duplicate(dst, scalar, COLS) 广播
"""

from __future__ import annotations

from dataclasses import dataclass

from .expr import Elem, Expr, Input, Reduce, Scalar, walk
from .compile import REWRITES, apply_rewrites

# 数值稳定不变量内建：设备 softmax 自动减最大值，不溢出（无需调用方记得传）
_AUTO_REWRITES = [REWRITES["数值稳定(减最大值)"]]


@dataclass(frozen=True)
class RowKernel:
    name: str
    source: str
    n_inputs: int
    scalar_out: bool          # True: 每行输出一个标量；False: 每行输出 COLS 个
    vector_calls: tuple[str, ...]


# 整行->整行的向量 API（与 ascendc_device 的 elementwise 表一致）
_UNARY = {
    "exp": "Exp({d}, {a}, COLS);",
    "tanh": "Tanh({d}, {a}, COLS);",
    "relu": "Maxs({d}, {a}, (float)0.0, COLS);",
    "neg": "Muls({d}, {a}, (float)-1.0, COLS);",
}
_BINARY = {
    "add": "Add({d}, {a}, {b}, COLS);",
    "sub": "Sub({d}, {a}, {b}, COLS);",
    "mul": "Mul({d}, {a}, {b}, COLS);",
    "div": "Div({d}, {a}, {b}, COLS);",
    "maximum": "Max({d}, {a}, {b}, COLS);",
}
_REDUCE_API = {"sum": "WholeReduceSum", "mean": "WholeReduceSum",
              "max": "WholeReduceMax", "min": "WholeReduceMin"}


class _Planner:
    """把表达式树规划成逐行指令序列。

    每个中间值是一个「槽」：
      - ("row", name)    : 长度 COLS 的 LocalTensor
      - ("scalar", name) : 归约标量（LocalTensor[0]）
      - ("const", float) : 编译期常量
    """

    def __init__(self) -> None:
        self.instrs: list[str] = []
        self._rowcnt = 0
        self._scalarcnt = 0
        self._svcnt = 0          # 取标量值的局部 float 编号（独立于标量槽）
        self._tmpbufs = 0        # 行临时 TBuf 数
        self._scalar_bufs = 0    # 标量 TBuf 数

    def new_row(self) -> str:
        s = f"t{self._rowcnt}"
        self._rowcnt += 1
        self._tmpbufs += 1
        return s

    def new_scalar(self) -> str:
        s = f"s{self._scalarcnt}"
        self._scalarcnt += 1
        self._scalar_bufs += 1
        return s

    def _as_row(self, v: tuple) -> str:
        """把一个槽变成整行 LocalTensor 名字（标量/常量 -> Duplicate 广播）。"""
        kind, ref = v
        if kind == "row":
            return ref
        if kind == "const":
            d = self.new_row()
            self.instrs.append(f"Duplicate({d}, (float){ref!r}, COLS);")
            return d
        if kind == "scalar":
            # ref 形如 s0Local；先 GetValue 再 Duplicate
            d = self.new_row()
            sv = f"sv{self._svcnt}"
            self._svcnt += 1
            self.instrs.append(f"float {sv} = {ref}.GetValue(0);")
            self.instrs.append(f"Duplicate({d}, {sv}, COLS);")
            return d
        raise ValueError(f"未知槽类型 {kind}")

    def emit(self, node: Expr) -> tuple[str, object]:
        if isinstance(node, Input):
            return ("row", f"x{node.i}Local")
        if isinstance(node, Scalar):
            return ("const", float(node.v))
        if isinstance(node, Reduce):
            if node.axis not in (-1, None):
                raise ValueError("设备归约仅支持末轴 axis=-1")
            a = self._as_row(self.emit(node.x))
            s = self.new_scalar()
            api = _REDUCE_API.get(node.op)
            if api is None:
                raise ValueError(f"设备后端不支持归约 op：{node.op}")
            self.instrs.append(f"{api}<float>({s}Local, {a}, (int32_t)COLS, 1, 1, 1, 1);")
            if node.op == "mean":
                self.instrs.append(f"Muls({s}Local, {s}Local, (float)(1.0f/COLS), 1);")
            return ("scalar", f"{s}Local")
        if isinstance(node, Elem):
            if node.op == "sigmoid":
                a = self._as_row(self.emit(node.args[0]))
                d = self.new_row()
                e = self.new_row()
                self.instrs.append(f"Muls({e}, {a}, (float)-1.0, COLS);")
                self.instrs.append(f"Exp({e}, {e}, COLS);")
                self.instrs.append(f"Adds({e}, {e}, (float)1.0, COLS);")
                self.instrs.append(f"Reciprocal({d}, {e}, COLS);")
                return ("row", d)
            if node.op == "pow":
                a = self._as_row(self.emit(node.args[0]))
                if isinstance(node.args[1], Scalar) and node.args[1].v == 0.5:
                    d = self.new_row()
                    self.instrs.append(f"Sqrt({d}, {a}, COLS);")
                    return ("row", d)
                raise ValueError("设备后端 pow 仅支持指数 0.5")
            if node.op in _UNARY:
                a = self._as_row(self.emit(node.args[0]))
                d = self.new_row()
                self.instrs.append(_UNARY[node.op].format(d=d, a=a))
                return ("row", d)
            if node.op in _BINARY:
                a = self._as_row(self.emit(node.args[0]))
                b = self._as_row(self.emit(node.args[1]))
                d = self.new_row()
                self.instrs.append(_BINARY[node.op].format(d=d, a=a, b=b))
                return ("row", d)
            raise ValueError(f"设备后端不支持 op：{node.op}")
        raise ValueError(f"设备后端不支持节点：{type(node).__name__}")


def compile_reduce_kernel(name: str, tree: Expr, *, rows: int = 64, cols: int = 64,
                          core_num: int = 8) -> RowKernel:
    """把含 Reduce 的表达式树编译成行级 AscendC kernel 源码。

    要求 COLS 为 64 的倍数（一个 repeat = 64 float），输入单张量 float32。
    """
    if cols % 64 != 0:
        raise ValueError("设备归约要求 cols 为 64 的倍数")
    # 不变量内建：自动施加数值稳定重写
    tree = apply_rewrites(tree, _AUTO_REWRITES)
    n_inputs = max(n.i for n in walk(tree) if isinstance(n, Input)) + 1

    planner = _Planner()
    root = planner.emit(tree)
    scalar_out = root[0] == "scalar" or root[0] == "const"

    # 结果搬到输出槽
    if root[0] == "row":
        planner.instrs.append(f"Adds(yLocal, {root[1]}, (float)0.0, COLS);")
    elif root[0] == "scalar":
        planner.instrs.append(f"Duplicate(yLocal, {root[1]}.GetValue(0), COLS);")
    else:  # const
        planner.instrs.append(f"Duplicate(yLocal, (float){root[1]!r}, COLS);")

    instrs = planner.instrs
    n_tmpbuf = planner._tmpbufs
    n_scalar = planner._scalar_bufs

    rows_per_core = rows // core_num
    # 标量输出：去掉末尾把结果搬 yLocal 的指令（改用 SetValue(r,...)）；行输出全保留
    compute_body = "\n".join("            " + x for x in (instrs[:-1] if scalar_out else instrs))

    tmp_decls = "\n".join(
        f"        AscendC::LocalTensor<float> t{i} = tmpBuf{i}.Get<float>();"
        for i in range(n_tmpbuf)
    )
    scalar_decls = "\n".join(
        f"        AscendC::LocalTensor<float> s{i}Local = sBuf{i}.Get<float>();"
        for i in range(n_scalar)
    )
    tmp_inits = "\n".join(
        f"        pipe.InitBuffer(tmpBuf{i}, COLS * sizeof(float));" for i in range(n_tmpbuf)
    )
    scalar_inits = "\n".join(
        f"        pipe.InitBuffer(sBuf{i}, 64);" for i in range(n_scalar)
    )
    tmp_members = "\n".join(
        f"    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf{i};" for i in range(n_tmpbuf)
    )
    scalar_members = "\n".join(
        f"    AscendC::TBuf<AscendC::TPosition::VECCALC> sBuf{i};" for i in range(n_scalar)
    )

    if scalar_out:
        # 标量输出 kernel：循环里每行归约 -> yLocal.SetValue(r,...)，循环外一次写回 ROWS_PER_CORE 个
        if root[0] == "scalar":
            tail_val = f"{root[1]}.GetValue(0)"
        else:  # const
            tail_val = f"(float){root[1]!r}"
        compute_method = f'''    __aicore__ inline void Compute() {{
        AscendC::LocalTensor<float> yLocal = yAcc.Get<float>();
        for (int32_t r = 0; r < ROWS_PER_CORE; r++) {{
            CopyIn(r);
            AscendC::LocalTensor<float> x0Local = inQueueX.DeQue<float>();
{scalar_decls}
{tmp_decls}
{compute_body}
            yLocal.SetValue(r, {tail_val});
            inQueueX.FreeTensor(x0Local);
        }}
    }}'''
        out_queue_block = ""
        copyout_method = '''    __aicore__ inline void CopyOut() {
        AscendC::LocalTensor<float> yLocal = yAcc.Get<float>();
        AscendC::DataCopy(yGm, yLocal, ROWS_PER_CORE);
    }'''
        y_acc_init = "        pipe.InitBuffer(yAcc, ROWS_PER_CORE * sizeof(float));"
        y_acc_member = "    AscendC::TBuf<AscendC::TPosition::VECCALC> yAcc;"
        process = "        Compute(); CopyOut();"
    else:
        compute_method = f'''    __aicore__ inline void Compute(int32_t r) {{
        AscendC::LocalTensor<float> x0Local = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
{scalar_decls}
{tmp_decls}
{compute_body}
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(x0Local);
    }}'''
        out_queue_block = "        pipe.InitBuffer(outQueueY, 1, COLS * sizeof(float));"
        copyout_method = f'''    __aicore__ inline void CopyOut(int32_t r) {{
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[r * COLS], yLocal, COLS);
        outQueueY.FreeTensor(yLocal);
    }}'''
        y_acc_init = ""
        y_acc_member = ""
        process = "        for (int32_t r = 0; r < ROWS_PER_CORE; r++) { CopyIn(r); Compute(r); CopyOut(r); }"
        in_y_init = ""

    copyin_method = '''    __aicore__ inline void CopyIn(int32_t r) {
        AscendC::LocalTensor<float> x0Local = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(x0Local, xGm[r * COLS], COLS);
        inQueueX.EnQue(x0Local);
    }'''

    out_queue_member = "" if scalar_out else "    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueY;"

    src = f'''/**
 * {name} · 由易道算子 Agent v4 自动生成（AscendC 行级归约 kernel）
 * 数值稳定 softmax / 归约族通用范式：COPY-IN -> VECTOR+REDUCE -> COPY-OUT
 * 行级归约 WholeReduceSum/Max + 标量广播 Duplicate/GetValue
 */
#include "kernel_operator.h"

constexpr int32_t ROWS = {rows};
constexpr int32_t COLS = {cols};
constexpr int32_t CORE_NUM = {core_num};
constexpr int32_t ROWS_PER_CORE = ROWS / CORE_NUM;

class Kernel{name} {{
public:
    __aicore__ inline Kernel{name}() {{}}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {{
        xGm.SetGlobalBuffer((__gm__ float*)x + AscendC::GetBlockIdx() * ROWS_PER_CORE * COLS, ROWS_PER_CORE * COLS);
        yGm.SetGlobalBuffer((__gm__ float*)y + AscendC::GetBlockIdx() * ROWS_PER_CORE * ({"1" if scalar_out else "COLS"}), ROWS_PER_CORE * ({"1" if scalar_out else "COLS"}));
        pipe.InitBuffer(inQueueX, 1, COLS * sizeof(float));
{out_queue_block}
{y_acc_init}
{tmp_inits}
{scalar_inits}
    }}
    __aicore__ inline void Process() {{
{process}
    }}
private:
{copyin_method}
{compute_method}
{copyout_method}
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX;
{out_queue_member}
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
{tmp_members}
{scalar_members}
{y_acc_member}
}};

extern "C" __global__ __aicore__ void {name}(GM_ADDR x, GM_ADDR y) {{
    Kernel{name} op; op.Init(x, y); op.Process();
}}
'''
    return RowKernel(name=name, source=src, n_inputs=n_inputs,
                     scalar_out=scalar_out, vector_calls=tuple(instrs))
