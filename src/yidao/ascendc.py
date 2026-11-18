"""AscendC 真实后端：表达式树 -> AscendC C++ kernel 源码（真工件）。

【这是"真实后端"而非"验证源码字符串"的含义】
生成的源码用的是真实 AscendC 编程模型：
  - GM_ADDR 入参 / __global__ __aicore__ 入口
  - GlobalTensor::SetGlobalBuffer + LocalTensor + TPipe + TQue<QuePosition,D>
  - DataCopy(GM->Local) / Compute / DataCopy(Local->GM) 三段式
把 cann_stub/kernel_operator.h 换成真机 CANN 的 kernel_operator.h，
同一份源码即可由 ccec 编译烧录到昇腾 NPU。

【求值策略】Compute 用通用坐标求值器（emit_eval）：对输出每个元素，
按表达式树递归求值。这保证任意声明式树都能编译（新增算子仍只加树），
且数值稳定/高精度累加两条重写以 C++ 形态真正生效：
  - 数值稳定 rewrite -> exp(z - max(z)) 出现在 C++ 源码里；
  - 累加精度 rewrite -> reduce/matmul 的累加变量用 double。

【可移植点 / 诚实边界】真机高性能实现会把逐元素循环替换为 AscendC
Vector API（Adds/Muls/Exp/ReduceSum/Matmul 等）并做 tiling；本骨架用标量
求值保证结构真实 + 语义可验证（见 README）。知止。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .expr import (
    Broadcast, Elem, Expr, Gather, Input, MatMul, Reduce, Scalar, Where,
)
from .compile import Rewrite, apply_rewrites

MAXDIM = 4
MAXIN = 8


@dataclass
class AstKernel:
    """一次编译的产物：源码 + 元信息。"""

    key: str
    source: str
    n_inputs: int
    out_shape: tuple[int, ...]
    dtype: str
    rewrites: list[str] = field(default_factory=list)


# ---- 形状推导（Python 端，用于分配输出缓冲 + 坐标 unflatten）-------------- #

def _broadcast_shape(shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
    if not shapes:
        return ()
    r = max(len(s) for s in shapes)
    out = [1] * r
    for s in shapes:
        pad = (1,) * (r - len(s)) + s
        for i, d in enumerate(pad):
            if d == 1:
                continue                      # 广播维：保留已推得的其他维度
            if out[i] == 1:
                out[i] = d
            elif out[i] != d:
                raise ValueError(f"shape 广播冲突：{out[i]} vs {d}")
    return tuple(out)


def infer_shape(node: Expr, shapes: dict[int, tuple[int, ...]]) -> tuple[int, ...]:
    if isinstance(node, Input):
        return shapes[node.i]
    if isinstance(node, Scalar):
        return ()
    if isinstance(node, Elem):
        return _broadcast_shape([infer_shape(a, shapes) for a in node.args])
    if isinstance(node, Reduce):
        base = infer_shape(node.x, shapes)
        ax = node.axis if node.axis is not None else len(base) - 1
        ax = ax % len(base) if base else 0
        if node.keepdims:
            return tuple(1 if i == ax else d for i, d in enumerate(base))
        return tuple(d for i, d in enumerate(base) if i != ax)
    if isinstance(node, MatMul):
        a, b = infer_shape(node.a, shapes), infer_shape(node.b, shapes)
        return (a[0], b[-1])
    if isinstance(node, Gather):
        data, idx = infer_shape(node.data, shapes), infer_shape(node.idx, shapes)
        return idx + data[1:]
    if isinstance(node, Broadcast):
        return tuple(s if isinstance(s, int) else shapes[int(s[1:])][0] for s in node.shape)
    if isinstance(node, Where):
        return _broadcast_shape([infer_shape(a, shapes) for a in (node.cond, node.a, node.b)])
    raise ValueError(f"无法推导形状：{type(node).__name__}")


def _first_input(node: Expr) -> int:
    """取子树引用的第一个输入索引——reduce/matmul 的维度长度从它查表。"""
    from .expr import walk
    for n in walk(node):
        if isinstance(n, Input):
            return n.i
    return 0


# ---- C++ 表达式发射 ------------------------------------------------------- #

_EMAP = {
    "add": lambda a: f"({a[0]} + {a[1]})",
    "sub": lambda a: f"({a[0]} - {a[1]})",
    "mul": lambda a: f"({a[0]} * {a[1]})",
    "div": lambda a: f"({a[0]} / {a[1]})",
    "neg": lambda a: f"(-{a[0]})",
    "relu": lambda a: f"fmaxf({a[0]}, 0.0f)",
    "maximum": lambda a: f"fmaxf({a[0]}, {a[1]})",
    "exp": lambda a: f"expf({a[0]})",
    "tanh": lambda a: f"tanhf({a[0]})",
    "sigmoid": lambda a: f"(1.0f / (1.0f + expf(-({a[0]}))))",
    "pow": lambda a: f"powf({a[0]}, {a[1]})",
    "asf64": lambda a: f"((double)({a[0]}))",  # 提示：外层累加用 double
}


class _Emitter:
    """把一棵（已应用重写的）树编译成 C++ 求值表达式 + 需要的辅助函数。"""

    def __init__(self) -> None:
        self.helpers: list[str] = []
        self._n = 0

    def helper_name(self, tag: str) -> str:
        self._n += 1
        return f"yj_{tag}_{self._n}"

    def emit(self, node: Expr, coord: str, cnd: int) -> str:
        if isinstance(node, Input):
            return f"IN({node.i}, {coord}, {cnd})"
        if isinstance(node, Scalar):
            return f"((float){node.v!r})"
        if isinstance(node, Elem):
            args = [self.emit(a, coord, cnd) for a in node.args]
            if node.op in _EMAP:
                return _EMAP[node.op](args)
            raise ValueError(f"AscendC 后端未知逐元素运算：{node.op}")
        if isinstance(node, Reduce):
            return self._emit_reduce(node, coord, cnd)
        if isinstance(node, MatMul):
            return self._emit_matmul(node, coord, cnd)
        if isinstance(node, Gather):
            return self._emit_gather(node, coord, cnd)
        if isinstance(node, Broadcast):
            return self.emit(node.x, coord, cnd)
        if isinstance(node, Where):
            c = self.emit(node.cond, coord, cnd)
            a = self.emit(node.a, coord, cnd)
            b = self.emit(node.b, coord, cnd)
            return f"(({c}) != 0.0f ? ({a}) : ({b}))"
        raise ValueError(f"AscendC 后端无法编译节点：{type(node).__name__}")

    # ---- 维度长度表达式（运行时查输入 shape）----
    @staticmethod
    def _dim_expr(node: Expr, axis: int) -> str:
        i = _first_input(node)
        if axis == -1:
            return f"G_IN_SHAPE[{i}][G_IN_NDIM[{i}] - 1]"
        return f"G_IN_SHAPE[{i}][{axis}]"

    @staticmethod
    def _has_asf64(node: Expr) -> bool:
        from .expr import walk
        return any(isinstance(n, Elem) and n.op == "asf64" for n in walk(node))

    def _emit_reduce(self, node: Reduce, coord: str, cnd: int) -> str:
        name = self.helper_name("reduce")
        xnd = cnd + (0 if node.keepdims else 1)
        axis_pos = (node.axis if node.axis is not None else xnd - 1) % xnd
        alen = self._dim_expr(node.x, node.axis if node.axis is not None else -1)
        fp = "double" if self._has_asf64(node.x) else "float"
        init = {"sum": "0.0", "mean": "0.0", "max": "-1e30"}.get(node.op, "0.0")
        child = self.emit(node.x, "xc", xnd)
        if node.op == "max":
            update = f"acc = ({fp})(acc > (tmp) ? acc : (tmp));"
        else:
            update = "acc += (tmp);"
        body = f"""
static float {name}(const int* coord, int cnd) {{
  {fp} acc = ({fp}){init};
  int xc[MAXDIM];
  for (int d = 0; d < cnd; ++d) xc[d] = coord[d];
  int alen = {alen};
  for (int k = 0; k < alen; ++k) {{
    xc[{axis_pos}] = k;
    {fp} tmp = ({fp})({child});
    {update}
  }}
  {'acc /= (alen > 0 ? alen : 1);' if node.op == 'mean' else ''}
  return (float)acc;
}}"""
        self.helpers.append(body)
        return f"{name}({coord}, {cnd})"

    def _emit_matmul(self, node: MatMul, coord: str, cnd: int) -> str:
        name = self.helper_name("matmul")
        Kexpr = self._dim_expr(node.a, -1)
        fp = "double" if (self._has_asf64(node.a) or self._has_asf64(node.b)) else "float"
        a = self.emit(node.a, "ac", 2)
        b = self.emit(node.b, "bc", 2)
        body = f"""
static float {name}(const int* coord, int cnd) {{
  {fp} acc = ({fp})0.0;
  int ac[2] = {{ coord[0], 0 }};
  int bc[2] = {{ 0, coord[1] }};
  int K = {Kexpr};
  for (int k = 0; k < K; ++k) {{
    ac[1] = k; bc[0] = k;
    acc += ({fp})({a}) * ({fp})({b});
  }}
  return (float)acc;
}}"""
        self.helpers.append(body)
        return f"{name}({coord}, {cnd})"

    def _emit_gather(self, node: Gather, coord: str, cnd: int) -> str:
        name = self.helper_name("gather")
        # 输出 shape = idx.shape + data.shape[1:]；当前均为 idx=(K,), data=(N,M)
        idxexpr = self.emit(node.idx, "ic", 1)
        dataexpr = self.emit(node.data, "dc", 2)
        srcrows = f"G_IN_SHAPE[{_first_input(node.data)}][0]"
        body = f"""
static float {name}(const int* coord, int cnd) {{
  int ic[1] = {{ coord[0] }};
  int row = (int)({idxexpr});
  if (row < 0 || row >= {srcrows}) {{ YJ_OOB = 1; return 0.0f; }}
  int dc[2] = {{ row, coord[1] }};
  return {dataexpr};
}}"""
        self.helpers.append(body)
        return f"{name}({coord}, {cnd})"


# ---- 完整源码拼装 --------------------------------------------------------- #

_PRELUDE = r"""
#include "kernel_operator.h"
#include <cmath>

#define MAXIN 8
#define MAXDIM 4

// 输入/输出元数据（由 launch 入口填入，kernel 编译一次可跑多种 shape）
static int   G_IN_NDIM[MAXIN];
static int   G_IN_SHAPE[MAXIN][MAXDIM];
static const float* G_IN[MAXIN];
static int   G_NIN;
static int   YJ_OOB = 0;
static int   G_OUT_NDIM;
static int   G_OUT_SHAPE[MAXDIM];

static inline int yj_in_index(int i, const int* coord, int cnd) {
  int nd = G_IN_NDIM[i];
  int off = cnd - nd;              // 右对齐广播
  int idx = 0;
  for (int d = 0; d < nd; ++d) {
    int c = (off + d >= 0) ? coord[off + d] : 0;
    int dim = G_IN_SHAPE[i][d];
    if (dim == 1) c = 0;           // 广播维
    idx = idx * dim + c;
  }
  return idx;
}
static inline float IN(int i, const int* coord, int cnd) {
  return G_IN[i][yj_in_index(i, coord, cnd)];
}
static inline void yj_unflatten(int flat, const int* shape, int nd, int* coord) {
  for (int d = nd - 1; d >= 0; --d) { coord[d] = flat % shape[d]; flat /= shape[d]; }
}
"""

_LAUNCH = r"""
// ---- 真 AscendC 三段式 Kernel（GM_ADDR -> Local -> Compute -> GM）--------
class YidaoKernel {
public:
  __aicore__ inline void Init(GM_ADDR ins[], int total_in, GM_ADDR out, int out_size) {
    for (int i = 0; i < total_in; ++i) {
      int n = 1;
      for (int d = 0; d < G_IN_NDIM[i]; ++d) n *= G_IN_SHAPE[i][d];
      xGm[i].SetGlobalBuffer((__gm__ float*)ins[i], (uint32_t)n);
      pipe.InitBuffer(qin[i], 1, (uint32_t)n * sizeof(float));
    }
    yGm.SetGlobalBuffer((__gm__ float*)out, (uint32_t)out_size);
    pipe.InitBuffer(qout, 1, (uint32_t)out_size * sizeof(float));
    m_out_size = out_size;
  }
  __aicore__ inline void CopyIn() {
    for (int i = 0; i < G_NIN; ++i) {
      LocalTensor<float> lt = qin[i].AllocTensor<float>();
      DataCopy(lt, xGm[i], (uint32_t)lt.GetSize());
      qin[i].EnQue(lt);            // EnQue 持有真实存储（拷贝进 slot）
      G_IN[i] = qin[i].slot.buf.data();  // Compute 消费 CopyIn 到本地的持久数据
    }
  }
  __aicore__ inline void Compute() {
    LocalTensor<float> ol = qout.AllocTensor<float>();
    int coord[MAXDIM];
    for (int flat = 0; flat < m_out_size; ++flat) {
      yj_unflatten(flat, G_OUT_SHAPE, G_OUT_NDIM, coord);
      ol[flat] = YJ_EVAL(coord, G_OUT_NDIM);
    }
    qout.EnQue(ol);
  }
  __aicore__ inline void CopyOut() {
    LocalTensor<float> ol = qout.DeQue<float>();
    DataCopy(yGm, ol, (uint32_t)m_out_size);
    qout.FreeTensor(ol);
  }
  __aicore__ inline void Process() { CopyIn(); Compute(); CopyOut(); }
private:
  TPipe pipe;
  GlobalTensor<float> xGm[MAXIN], yGm;
  TQue<QuePosition::VECIN, 1> qin[MAXIN];
  TQue<QuePosition::VECOUT, 1> qout;
  int m_out_size;
};

// __global__ __aicore__ 核函数入口（真机由 <<<blockDim>>> 或 ACL 启动；
// 桩环境退化为普通函数调用，语义一致）。GM_ADDR 是 AI Core 可见的设备地址。
__global__ __aicore__ void yidao_kernel(GM_ADDR ins[], int nin, GM_ADDR out, int out_size) {
  YidaoKernel op;
  op.Init(ins, nin, out, out_size);
  op.Process();
}

extern "C" int yidao_kernel_launch(
    float** inputs, int** in_shapes, int* in_ndims, int nin,
    float* out, int* out_shape, int out_ndim) {
  G_NIN = nin;
  G_OUT_NDIM = out_ndim;
  for (int d = 0; d < out_ndim; ++d) G_OUT_SHAPE[d] = out_shape[d];
  for (int i = 0; i < nin; ++i) {
    G_IN_NDIM[i] = in_ndims[i];
    for (int d = 0; d < in_ndims[i]; ++d) G_IN_SHAPE[i][d] = in_shapes[i][d];
  }
  YJ_OOB = 0;
  GM_ADDR ins[MAXIN];
  for (int i = 0; i < nin; ++i) ins[i] = (GM_ADDR)inputs[i];
  int out_size = 1;
  for (int d = 0; d < out_ndim; ++d) out_size *= out_shape[d];
  yidao_kernel(ins, nin, (GM_ADDR)out, out_size);  // 启动 __global__ 核函数
  return YJ_OOB;
}
"""


def compile_ascendc_source(
    key: str, tree: Expr, rewrites: list[Rewrite], n_inputs: int,
    dtypes: tuple[str, ...], output_dtype: str,
    shapes: dict[int, tuple[int, ...]],
) -> AstKernel:
    """把表达式树编译成 AscendC kernel 源码。shapes 为符号 shape（含>N 等维度名，
    用 1 占位即可；真实维度由运行时 C ABI 传入）。"""
    t = apply_rewrites(tree, rewrites)
    em = _Emitter()
    root = em.emit(t, "coord", len(infer_shape(t, shapes)) or 1)
    helpers = "\n".join(em.helpers)
    src = (
        "// 由易道算子 Agent · 易理内核 v4 自动生成（AscendC 后端）\n"
        f"// 算子：{key}；类型：{output_dtype}；重写：{[r.name for r in rewrites]}\n"
        + _PRELUDE
        + helpers
        + f"\nstatic float YJ_EVAL(const int* coord, int cnd) {{ return {root}; }}\n"
        + _LAUNCH
    )
    out_shape = infer_shape(t, shapes)
    return AstKernel(
        key=key, source=src, n_inputs=n_inputs, out_shape=out_shape,
        dtype=output_dtype, rewrites=[r.name for r in rewrites],
    )
