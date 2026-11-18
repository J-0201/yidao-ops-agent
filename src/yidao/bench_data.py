"""官方评测集（裁判方资产）——为内置算子提供 EvalContract。

评测集约定：每个算子一份契约 = proto（原型）+ golden（参考实现）+ cases（用例）。
本模块代表【评测方】，与算子库 specs.py（选手方）物理分离、互不 import：
  - specs.py 定义「怎么实现」（表达式树 + 选手自报的常）；
  - bench_data.py 定义「对不对、测哪些」（golden + 用例），选手不可见、不可改。

golden 复用 semantic.py 的手写 NumPy 原语（它们本就与 expr/compile 零依赖，
是独立可信实现），但【注册与裁判权】归评测集——这正是与 v3-v6 的本质区别：
过去 oracle 是选手自己塞进 OpSpec 的；现在由评测集在裁判时强制注入。
"""

from __future__ import annotations

import numpy as np

from .evalbench import EvalCase, EvalContract, EvalProto, EvalRegistry
from .semantic import (
    SEMANTICS, _gelu_oracle, _layernorm_oracle, _relu_props,
    _reduce_props, _sigmoid_oracle, _softmax_props,
)


def _f32(shape, *, boundary="常规", scale=1.0, pattern="normal") -> EvalCase:
    return EvalCase(shape, ("float32",) * len(shape), boundary=boundary,
                    scale=scale, pattern=pattern)


def build_registry() -> EvalRegistry:
    """构造内置评测集。返回完整注册好的裁判注册表。"""
    reg = EvalRegistry()

    def add(key, proto, golden, cases, props=()):
        reg.register(EvalContract(
            key=key, proto=proto, golden=golden, cases=cases,
            properties=tuple(props), source="official-bench-v1",
        ))

    P = EvalProto
    # ---- 单输入逐元素 / 归约 ----
    add("relu",
        P((("x", "float32", ("N", "M")),), "float32"),
        SEMANTICS["relu"].oracle,
        [_f32(((4, 3),)), _f32(((1, 1),), boundary="单元素"),
         _f32(((3, 5),), boundary="大数", scale=10.0)],
        _relu_props())

    add("sigmoid",
        P((("x", "float32", ("N", "M")),), "float32"),
        _sigmoid_oracle,
        [_f32(((3, 4),)), _f32(((2, 3),), boundary="大数", scale=50.0)])

    add("tanh",
        P((("x", "float32", ("N", "M")),), "float32"),
        lambda x: np.tanh(x),
        [_f32(((3, 4),)), _f32(((2, 3),), boundary="大数", scale=20.0)])

    add("gelu",
        P((("x", "float32", ("N", "M")),), "float32"),
        _gelu_oracle,
        [_f32(((3, 4),)), _f32(((2, 3),), boundary="大数", scale=10.0)])

    add("softmax",
        P((("z", "float32", ("N", "M")),), "float32"),
        SEMANTICS["softmax"].oracle,
        [_f32(((3, 4),)), _f32(((1, 3),), boundary="单元素"),
         _f32(((2, 4),), boundary="大数", scale=1000.0)],
        _softmax_props())

    add("reduce_sum",
        P((("x", "float32", ("N", "M")),), "float32"),
        SEMANTICS["reduce_sum"].oracle,
        [_f32(((3, 6),)), _f32(((1, 4),), boundary="单元素"),
         _f32(((3, 6),), boundary="相消", pattern="cancel")],
        _reduce_props("sum"))

    add("reduce_mean",
        P((("x", "float32", ("N", "M")),), "float32"),
        SEMANTICS["reduce_mean"].oracle,
        [_f32(((3, 6),)), _f32(((3, 6),), boundary="相消", pattern="cancel")],
        _reduce_props("mean"))

    add("reduce_max",
        P((("x", "float32", ("N", "M")),), "float32"),
        SEMANTICS["reduce_max"].oracle,
        [_f32(((3, 6),)), _f32(((2, 4),), boundary="大数", scale=100.0)],
        _reduce_props("max"))

    add("layernorm",
        P((("x", "float32", ("N", "M")),), "float32"),
        _layernorm_oracle,
        [_f32(((3, 5),)), _f32(((3, 6),), boundary="相消", pattern="cancel")])

    # ---- 双输入 ----
    add("matmul",
        P((("a", "float32", ("N", "K")), ("b", "float32", ("K", "M"))), "float32"),
        SEMANTICS["matmul"].oracle,
        [EvalCase(((3, 4), (4, 5)), ("float32", "float32")),
         EvalCase(((1, 4), (4, 1)), ("float32", "float32"), boundary="单元素"),
         EvalCase(((3, 6), (6, 4)), ("float32", "float32"), boundary="大数", scale=100.0)])

    add("add_broadcast",
        P((("a", "float32", ("N", "M")), ("b", "float32", ("M",))), "float32"),
        SEMANTICS["add_broadcast"].oracle,
        [EvalCase(((3, 4), (4,)), ("float32", "float32")),
         EvalCase(((2, 3), (3,)), ("float32", "float32"), boundary="大数", scale=5.0)])

    add("gather",
        P((("data", "float32", ("N", "M")), ("idx", "int64", ("K",))), "float32"),
        SEMANTICS["gather"].oracle,
        [EvalCase(((4, 3), (3,)), ("float32", "int64")),
         EvalCase(((5, 2), (4,)), ("float32", "int64"), boundary="多索引"),
         EvalCase(((3, 2), (2,)), ("float32", "int64"), boundary="越界", pattern="oob")])

    return reg
