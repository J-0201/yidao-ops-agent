"""症状诊断：从 out/ref 的真实差异读出症状（数据事实，无人工标注）。

v1 的抗体谱（heals 由谓词行为导出）与注入器自动合成是漂亮的机制，但 v2 里
不变量本身就是「针对某症状的树重写规则」——重写规则绑定症状是设计事实
（减最大值治非有限、float64 治精度），无需再行为反推。保留的是诊断的严谨：
只读 out/ref/inp 的数值事实，不查注释、不查字典 key。
"""

from __future__ import annotations

import numpy as np

from .model import Symptom

ATOL = 1e-5
RTOL = 1e-5


def diagnose(
    out: np.ndarray,
    ref: np.ndarray,
    inp: list[np.ndarray],
    atol: float = ATOL,
    rtol: float = RTOL,
) -> set[str]:
    """从真实输出 vs 参考的差异诊断症状。错误实现无处遁形。"""
    syms: set[str] = set()
    nonfinite = not bool(np.all(np.isfinite(out)))
    if nonfinite:
        syms.add(Symptom.NONFINITE)
    if out.shape != ref.shape:
        syms.add(Symptom.SHAPE)
    elif str(out.dtype) != str(ref.dtype):
        syms.add(Symptom.DTYPE)
    elif not nonfinite and not np.allclose(
        out.astype(np.float64), ref.astype(np.float64),
        atol=atol, rtol=rtol, equal_nan=False,
    ):
        syms.add(Symptom.PRECISION)
    # 索引输入越界
    if inp:
        data = inp[0]
        for x in inp:
            if x.dtype.kind in "iu" and x.size:
                if x.min() < 0 or x.max() >= data.shape[0]:
                    syms.add(Symptom.INDEX)
                    break
    return syms
