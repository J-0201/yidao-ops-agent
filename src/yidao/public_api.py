"""公共扩展 API（v5）——让「通用性」真正可用，而非只内置 12 算子。

第三方评估对 v2/v3 的一条持续边界：
  「oracle 与性质只覆盖内置 12 算子；外部注册的新算子若不提供 SemanticSpec，
    自动降级为忠实度验证——通用性与语义可信度之间存在取舍。」

v4 虽然抽象出多后端，但新增一个"带可信语义基准"的算子仍要手工拼
SemanticSpec(gold, properties)、OpSpec、ShiWei……门槛高，于是外部算子
往往不带基准、落回降级。v5 提供一站式公共 API 把这条门槛降到最低：

  - 性质构造器：nonnegative / between / row_sums_to_one / monotonic /
    zero_for_nonpositive / identity_for_positive / zero_mean / unit_variance /
    shape_matches / finite —— 覆盖绝大多数逐元素与归约算子的意图；
  - make_semantic(name, oracle, properties=...) —— 从权威函数 + 性质造基准；
  - make_op(...) —— 从表达式树 + 权威函数一站式造带基准的 OpSpec；
  - YiDaoAgent.register_op(...) —— 直接注册并完成开发。

这些构造器与 semantic.py 一样【不 import expr/compile 的求值路径】，
性质只读 (out, inputs) 的数值事实。降低门槛不等于降低严谨：外部算子若
不提供 oracle，依然诚实降级——本模块只是让"愿意提供"变得容易。
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .model import OpSpec, ShiWei
from .semantic import Property, SemanticSpec


# ---- 公共性质构造器 ------------------------------------------------------ #

def nonnegative() -> Property:
    """输出全体非负。"""
    return Property("非负性", lambda out, inp: (
        bool(np.all(out >= 0)), f"存在负值 {float(np.min(out)):.3g}"
    ))


def between(lo: float, hi: float, strict: bool = False) -> Property:
    """输出落在 [lo, hi]（strict=True 时为开区间）。"""
    def check(out, inp):
        if strict:
            ok = bool(np.all(out > lo) and np.all(out < hi))
        else:
            ok = bool(np.all(out >= lo) and np.all(out <= hi))
        return ok, f"超出区间：min={float(np.min(out)):.3g} max={float(np.max(out)):.3g}"
    return Property(f"值域[{'(' if strict else '['}{lo},{hi}{')' if strict else ']'}]", check)


def finite() -> Property:
    """输出全部有限（无 NaN/Inf）。"""
    return Property("有限性", lambda out, inp: (
        bool(np.all(np.isfinite(out))), "存在 NaN/Inf"
    ))


def row_sums_to_one(atol: float = 1e-4) -> Property:
    """沿末轴归一（行和为 1）。"""
    def check(out, inp):
        s = out.astype(np.float64).sum(axis=-1)
        return bool(np.allclose(s, 1.0, atol=atol, rtol=atol)), f"行和偏离1: {np.round(s.ravel()[:3], 4).tolist()}"
    return Property("归一化(行和为1)", check)


def monotonic(increasing: bool = True, axis: int = -1) -> Property:
    """沿轴单调（递增或递减）。"""
    def check(out, inp):
        d = np.diff(out.astype(np.float64), axis=axis)
        ok = bool(np.all(d >= -1e-6)) if increasing else bool(np.all(d <= 1e-6))
        return ok, "单调性被破坏"
    return Property("单调递增" if increasing else "单调递减", check)


def zero_for_nonpositive(atol: float = 1e-6) -> Property:
    """输入 ≤0 处输出为 0（ReLU 族）。"""
    def check(out, inp):
        mask = np.asarray(inp[0]) <= 0
        if not np.any(mask):
            return True, ""
        return bool(np.all(np.abs(out[mask]) <= atol)), "x<=0 处 out≠0"
    return Property("非正输入归零", check)


def identity_for_positive(atol: float = 1e-5) -> Property:
    """输入 >0 处输出等于输入（ReLU 族）。"""
    def check(out, inp):
        x = np.asarray(inp[0])
        mask = x > 0
        if not np.any(mask):
            return True, ""
        return bool(np.allclose(out[mask], x[mask], atol=atol)), "x>0 处 out≠x"
    return Property("正半轴恒等", check)


def zero_mean(atol: float = 1e-3) -> Property:
    """沿末轴零均值（LayerNorm 族）。"""
    def check(out, inp):
        m = out.astype(np.float64).mean(axis=-1)
        return bool(np.allclose(m, 0.0, atol=atol)), f"均值偏离0: {np.round(m.ravel()[:3], 4).tolist()}"
    return Property("零均值", check)


def unit_variance(atol: float = 1e-3) -> Property:
    """沿末轴单位方差（LayerNorm 族）。"""
    def check(out, inp):
        v = out.astype(np.float64).var(axis=-1)
        return bool(np.allclose(v, 1.0, atol=atol)), f"方差偏离1: {np.round(v.ravel()[:3], 4).tolist()}"
    return Property("单位方差", check)


PROPERTY_BUILDERS: dict[str, Callable[..., Property]] = {
    "nonnegative": nonnegative,
    "between": between,
    "finite": finite,
    "row_sums_to_one": row_sums_to_one,
    "monotonic": monotonic,
    "zero_for_nonpositive": zero_for_nonpositive,
    "identity_for_positive": identity_for_positive,
    "zero_mean": zero_mean,
    "unit_variance": unit_variance,
}


# ---- 基准 / 算子一站式构造 ------------------------------------------------ #

def make_semantic(
    name: str,
    oracle: Callable[..., np.ndarray],
    properties: Sequence[Property] = (),
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> SemanticSpec:
    """从权威函数 + 性质构造独立语义基准。oracle 应只用手写 NumPy 数学原语。"""
    return SemanticSpec(name=name, oracle=oracle, properties=tuple(properties), atol=atol, rtol=rtol)


def make_op(
    key: str,
    tree,
    inputs: tuple[tuple[str, str, tuple[str, ...]], ...],
    output_dtype: str,
    *,
    oracle: Callable[..., np.ndarray] | None = None,
    properties: Sequence[Property] = (),
    wei: list[ShiWei] | None = None,
    known: set[str] | None = None,
    doc: str = "",
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> OpSpec:
    """一站式构造带独立语义基准的算子规格。

    oracle=None 时语义基准缺省，算子将诚实降级为「仅验忠实度」。
    wei=None 时按 inputs 自动生成两个常规时位（小 shape + 大数 shape），
    让外部算子开箱即可进入观复闭环。
    """
    semantic = make_semantic(key, oracle, properties, atol=atol, rtol=rtol) if oracle else None
    if wei is None:
        shapes = []
        dtypes = []
        for _name, dtype, sym in inputs:
            shape = tuple(4 if isinstance(d, str) else d for d in sym)
            shapes.append(shape)
            dtypes.append(dtype)
        wei = [ShiWei(tuple(shapes), tuple(dtypes)),
               ShiWei(tuple(shapes), tuple(dtypes), boundary="大数", scale=10.0)]
    return OpSpec(
        key=key, tree=tree, inputs=inputs, output_dtype=output_dtype,
        wei=wei, known=known or set(), doc=doc, semantic=semantic,
    )
