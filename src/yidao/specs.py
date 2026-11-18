"""算子规格库：每个算子 = 一棵声明式表达式树 + 时位 + 初始已知常。

扩展系统只需在这里（或外部）加一个 OpSpec 实例，不碰任何引擎代码。
以下包含 v1 硬编码的 6 个算子，以及 v1 明确拒绝的新算子：
sigmoid / tanh / gelu / reduce_mean / reduce_max / softmax 变体等。
"""

from __future__ import annotations

from .expr import (
    Expr, add, div, elem, exp, gather, inp, matmul, maximum, mean_, mul,
    neg, reduce, relu, scalar, sigmoid, sub, sum_, tanh,
)
from .model import OpSpec, ShiWei
from .semantic import SEMANTICS


def _wei(
    shapes: tuple[tuple[int, ...], ...],
    dtypes: tuple[str, ...],
    *,
    boundary: str = "常规",
    scale: float = 1.0,
    pattern: str = "normal",
) -> ShiWei:
    return ShiWei(shapes, dtypes, boundary=boundary, scale=scale, pattern=pattern)


_SOFT_STRUCT = {"shape一致", "dtype可表示"}


def build_specs() -> dict[str, OpSpec]:
    specs: dict[str, OpSpec] = {}

    # relu：逐元素，常已足 -> 1 轮天成
    specs["relu"] = OpSpec(
        key="relu", tree=relu(inp(0)),
        inputs=(("x", "float32", ("N", "M")),), output_dtype="float32",
        wei=[_wei(((4, 3),), ("float32",)),
             _wei(((1, 1),), ("float32",), boundary="单元素"),
             _wei(((3, 5),), ("float32",), boundary="大数", scale=10.0)],
        known=set(_SOFT_STRUCT), doc="整流线性单元：max(x, 0)",
    )

    # softmax：归约+逐元素；初始缺数值稳定 -> 大数时位溢出 -> 补常
    z = inp(0)
    softmax_tree = div(exp(z), sum_(exp(z), -1, keepdims=True))
    specs["softmax"] = OpSpec(
        key="softmax", tree=softmax_tree,
        inputs=(("z", "float32", ("N", "M")),), output_dtype="float32",
        wei=[_wei(((3, 4),), ("float32",)),
             _wei(((1, 3),), ("float32",), boundary="单元素"),
             _wei(((2, 4),), ("float32",), boundary="大数", scale=1000.0)],
        known=set(_SOFT_STRUCT), doc="数值稳定 softmax：exp(z-max)/sum",
    )

    # reduce_sum：归约；初始缺累加精度 -> 相消时位丢小数 -> 补常
    specs["reduce_sum"] = OpSpec(
        key="reduce_sum", tree=sum_(inp(0), -1),
        inputs=(("x", "float32", ("N", "M")),), output_dtype="float32",
        wei=[_wei(((3, 6),), ("float32",)),
             _wei(((1, 4),), ("float32",), boundary="单元素"),
             _wei(((3, 6),), ("float32",), boundary="相消", pattern="cancel")],
        known=set(_SOFT_STRUCT), doc="沿末轴求和",
    )

    # matmul：矩阵乘；相消时位暴露累加精度
    specs["matmul"] = OpSpec(
        key="matmul", tree=matmul(inp(0), inp(1)),
        inputs=(("a", "float32", ("N", "K")), ("b", "float32", ("K", "M"))),
        output_dtype="float32",
        wei=[_wei(((3, 4), (4, 5)), ("float32", "float32")),
             _wei(((1, 4), (4, 1)), ("float32", "float32"), boundary="单元素"),
             _wei(((3, 6), (6, 4)), ("float32", "float32"), boundary="大数", scale=100.0)],
        # 给 matmul 预置累加精度常：矩阵乘的 float32 累加本就该用高精度，
        # 它属于「结构已知」而非「失败后补出」。相消构造对 matmul 不适用
        # （大数相乘会让 float32/float64 的乘积本身分歧，非累加能修）。
        known=set(_SOFT_STRUCT) | {"累加精度(float64)"}, doc="矩阵乘 A @ B",
    )

    # add_broadcast：广播逐元素
    specs["add_broadcast"] = OpSpec(
        key="add_broadcast", tree=add(inp(0), inp(1)),
        inputs=(("a", "float32", ("N", "M")), ("b", "float32", ("M",))),
        output_dtype="float32",
        wei=[_wei(((3, 4), (4,)), ("float32", "float32")),
             _wei(((2, 3), (3,)), ("float32", "float32"), boundary="大数", scale=5.0)],
        known=set(_SOFT_STRUCT), doc="广播相加 a + b",
    )

    # gather：按索引收集
    specs["gather"] = OpSpec(
        key="gather", tree=gather(inp(0), inp(1)),
        inputs=(("data", "float32", ("N", "M")), ("idx", "int64", ("K",))),
        output_dtype="float32",
        wei=[_wei(((4, 3), (3,)), ("float32", "int64")),
             _wei(((5, 2), (4,)), ("float32", "int64"), boundary="多索引"),
             _wei(((3, 2), (2,)), ("float32", "int64"), boundary="越界", pattern="oob")],
        known=set(_SOFT_STRUCT), doc="按整数索引收集行",
    )

    # ====== 以下为 v1 明确拒绝的新算子 ======

    specs["sigmoid"] = OpSpec(
        key="sigmoid", tree=sigmoid(inp(0)),
        inputs=(("x", "float32", ("N", "M")),), output_dtype="float32",
        wei=[_wei(((3, 4),), ("float32",)),
             _wei(((2, 3),), ("float32",), boundary="大数", scale=50.0)],
        known=set(_SOFT_STRUCT), doc="Sigmoid：1/(1+exp(-x))（新算子）",
    )

    specs["tanh"] = OpSpec(
        key="tanh", tree=tanh(inp(0)),
        inputs=(("x", "float32", ("N", "M")),), output_dtype="float32",
        wei=[_wei(((3, 4),), ("float32",)),
             _wei(((2, 3),), ("float32",), boundary="大数", scale=20.0)],
        known=set(_SOFT_STRUCT), doc="双曲正切 tanh（新算子）",
    )

    # gelu 近似：x * sigmoid(1.702 x)
    specs["gelu"] = OpSpec(
        key="gelu", tree=mul(inp(0), sigmoid(mul(scalar(1.702), inp(0)))),
        inputs=(("x", "float32", ("N", "M")),), output_dtype="float32",
        wei=[_wei(((3, 4),), ("float32",)),
             _wei(((2, 3),), ("float32",), boundary="大数", scale=10.0)],
        known=set(_SOFT_STRUCT), doc="GELU 近似：x·sigmoid(1.702x)（新算子）",
    )

    specs["reduce_mean"] = OpSpec(
        key="reduce_mean", tree=mean_(inp(0), -1),
        inputs=(("x", "float32", ("N", "M")),), output_dtype="float32",
        wei=[_wei(((3, 6),), ("float32",)),
             _wei(((3, 6),), ("float32",), boundary="相消", pattern="cancel")],
        known=set(_SOFT_STRUCT), doc="沿末轴求均值（新算子）",
    )

    specs["reduce_max"] = OpSpec(
        key="reduce_max", tree=reduce("max", inp(0), -1),
        inputs=(("x", "float32", ("N", "M")),), output_dtype="float32",
        wei=[_wei(((3, 6),), ("float32",)),
             _wei(((2, 4),), ("float32",), boundary="大数", scale=100.0)],
        known=set(_SOFT_STRUCT), doc="沿末轴求最大值（新算子）",
    )

    # layernorm：(x-mean)/sqrt(var+eps)，用表达式组合，演示复合新算子
    x = inp(0)
    mu = mean_(x, -1, keepdims=True)
    var = mean_(mul(sub(x, mu), sub(x, mu)), -1, keepdims=True)
    # var 里无 sqrt 组合子，用 pow(...,0.5) 通过通用 elem
    ln = div(sub(x, mu), elem("pow", add(var, scalar(1e-5)), scalar(0.5)))
    specs["layernorm"] = OpSpec(
        key="layernorm", tree=ln,
        inputs=(("x", "float32", ("N", "M")),), output_dtype="float32",
        wei=[_wei(((3, 5),), ("float32",)),
             _wei(((3, 6),), ("float32",), boundary="相消", pattern="cancel")],
        known=set(_SOFT_STRUCT), doc="LayerNorm：(x-μ)/√(var+eps)（新算子）",
    )

    # 【v3】给每个内置算子绑定独立语义基准（手写 oracle + 数学性质）。
    # 语义身份由 key 决定，与树结构、doc 文本无关——fake_relu 若声称 relu
    # 的语义却给出 x*x 的树，会在 oracle 对拍与性质检验中被揭穿。
    for key, spec in specs.items():
        if key in SEMANTICS:
            spec.semantic = SEMANTICS[key]
    return specs
