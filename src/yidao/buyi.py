"""不易层（④ 常）：不变量目录。每条不变量绑定一个症状与一条树重写规则。

v1 的谓词在这里保留：`holds(ctx)` 在真实输出上判定。重写规则名指向
compile.REWRITES——内建即把规则应用到表达式树，违规结构在树上写不出。

不变量按筋骨组织必备集：观复只在「本筋骨必备但尚未激活」的常里补。
"""

from __future__ import annotations

import numpy as np

from .model import BuYi, Symptom


def _finite(out: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(out)))


def _close(ctx: dict) -> bool:
    out, ref = ctx["out"], ctx["ref"]
    if not np.all(np.isfinite(out)):  # 非有限由数值稳定常专职管，此处弃权
        return True
    return bool(np.allclose(
        out.astype(np.float64), ref.astype(np.float64),
        atol=ctx.get("atol", 1e-5), rtol=ctx.get("rtol", 1e-5),
    ))


def catalog() -> dict[str, BuYi]:
    return {
        "数值稳定(减最大值)": BuYi(
            name="数值稳定(减最大值)", level="numeric",
            builtin="exp 前先减末轴最大值（keepdims），指数恒≤0 不溢出",
            symptom=Symptom.NONFINITE, rewrite_name="数值稳定(减最大值)",
            predicate=lambda c: _finite(c["out"]),
        ),
        "累加精度(float64)": BuYi(
            name="累加精度(float64)", level="numeric",
            builtin="归约累加提升到 float64，数值相消时不丢小数",
            symptom=Symptom.PRECISION, rewrite_name="累加精度(float64)",
            predicate=_close,
        ),
        "shape一致": BuYi(
            name="shape一致", level="struct",
            builtin="输出 shape 由表达式树静态推导并对齐参考",
            symptom=Symptom.SHAPE,
            predicate=lambda c: c["out"].shape == c["ref"].shape,
        ),
        "dtype可表示": BuYi(
            name="dtype可表示", level="struct",
            builtin="输出回写声明 dtype",
            symptom=Symptom.DTYPE,
            predicate=lambda c: str(c["out"].dtype) == c["spec"].output_dtype,
        ),
    }


# 每种筋骨必备的常（简易层据此在生成前校验，观复据此限定补常范围）
REQUIRED: dict[str, list[str]] = {
    "QIAN": ["shape一致", "dtype可表示"],
    "KUN": ["shape一致", "dtype可表示", "累加精度(float64)"],
    "ZHEN": ["shape一致", "dtype可表示", "累加精度(float64)"],
    "XUN": ["shape一致", "dtype可表示"],
    "KAN": ["shape一致", "dtype可表示"],
    "LI": ["shape一致", "dtype可表示", "数值稳定(减最大值)"],
}


def symptom_to_const() -> dict[str, str]:
    """症状 -> 能根治它的常。由不变量目录的 symptom 字段导出，非手写映射。"""
    return {inv.symptom: name for name, inv in catalog().items() if inv.symptom}
