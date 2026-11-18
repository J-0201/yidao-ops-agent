"""语义基准层（v3 的核心突破）——独立于规格树的「意图」可执行化。

【v2 的根本边界】v2 的验证是 kernel=compile(树) vs ref=interpret(同一棵树)。
两条路径虽独立实现，却消费同一棵树——于是只能证明「实现忠实于树」，
无法证明「树符合目标算子的数学意图」。fake_relu：树写成 x*x、doc 声称
ReLU，两条路径一致 -> 判天成。这是同义反复。

【v3 的解法】引入 SemanticSpec，它由两部分构成，都【不 import expr / compile】：
  1. oracle    —— 手写权威实现，用 NumPy 数学原语直接算，与规格树无任何
                  代码依赖（唯一联系是数学等价）。它代表「金标准语义」。
  2. properties—— 数学性质谓词（非负、归一、恒等、零均值…），是意图的
                  另一重、独立于 oracle 的表达。

OpSpec 必须显式携带 semantic（声明要实现哪个语义）；doc 文本不再承诺语义。
验证时：kernel 对拍 oracle（语义判据）+ 直接验性质（意图判据）+ 解释器
对拍（实现忠实度，降为辅证）。fake_relu 绑了 ReLU 的 semantic 后，x*x
与 oracle 不符、且违反恒等/归零性质 -> 判悔，且补常无法修复（问题在树，
不在常）-> 诚实停手未天成。

【诚实边界】oracle 与性质由人工声明，是金标准的终点，无法自我证明其正确；
但它们把「意图」从不可验证的 doc 文本，变成可执行、独立于树的对象——
这正是评估要求的「独立于规格树的可信语义基准」。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class Property:
    """一条数学性质：对 (输出, 输入列表) 判 (是否成立, 失败说明)。"""

    name: str
    check: Callable[[np.ndarray, list[np.ndarray]], tuple[bool, str]]


@dataclass(frozen=True)
class SemanticSpec:
    """语义基准：手写 oracle + 数学性质。与表达式树零代码依赖。"""

    name: str
    oracle: Callable[..., np.ndarray]
    properties: tuple[Property, ...] = ()
    atol: float = 1e-4
    rtol: float = 1e-4

    def gold(self, inputs: list[np.ndarray]) -> np.ndarray:
        """金标准输出（float64），独立于任何编译/解释路径。"""
        return np.asarray(self.oracle(*inputs), dtype=np.float64)

    def judge(self, out: np.ndarray, inputs: list[np.ndarray]) -> tuple[bool, list[str], np.ndarray]:
        """返回 (语义是否成立, 失败说明列表, 金标准输出)。

        失败说明区分「语义偏离」(oracle 对拍不过) 与「性质违反」(性质谓词不过)。
        二者都计入失败；oracle 对拍失败还给出可用于补常的症状数据。
        """
        gold = self.gold(inputs)
        fails: list[str] = []
        # dtype 对齐：gold 回写到 kernel 的声明 dtype 再比（dtype 由结构常管）
        gold_cast = gold.astype(out.dtype)
        if not _close(out, gold_cast, self.atol, self.rtol):
            fails.append("语义偏离：输出与权威实现(oracle)不符")
        for p in self.properties:
            try:
                ok, msg = p.check(out, inputs)
                if not ok:
                    fails.append(f"性质违反·{p.name}：{msg}")
            except Exception as exc:  # 性质检验自身异常视为违反
                fails.append(f"性质违反·{p.name}：检验异常 {type(exc).__name__}")
        return (not fails), fails, gold_cast


def _close(a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> bool:
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.allclose(a.astype(np.float64), b.astype(np.float64),
                            atol=atol, rtol=rtol))


# ---- 通用性质构造器（与具体算子无关的可复用意图表达）---------------------- #

def _nonnegative(name: str = "非负性") -> Property:
    return Property(name, lambda out, inp: (
        bool(np.all(out >= 0)), f"存在负值 {float(np.min(out)):.3g}"
    ))


def _row_sums_to_one() -> Property:
    def check(out: np.ndarray, inp: list[np.ndarray]) -> tuple[bool, str]:
        s = out.astype(np.float64).sum(axis=-1)
        ok = bool(np.allclose(s, 1.0, atol=1e-4, rtol=1e-4))
        return ok, f"行和偏离1: {np.round(s.ravel()[:3], 4).tolist()}"
    return Property("归一化(行和为1)", check)


def _range(lo: float, hi: float, strict: bool = True) -> Property:
    def check(out: np.ndarray, inp: list[np.ndarray]) -> tuple[bool, str]:
        if strict:
            ok = bool(np.all(out > lo) and np.all(out < hi))
        else:
            ok = bool(np.all(out >= lo) and np.all(out <= hi))
        return ok, f"超出({lo},{hi})：min={float(np.min(out)):.3g} max={float(np.max(out)):.3g}"
    return Property(f"值域[{lo},{hi}]" + ("" if strict else "闭"), check)


# ---- oracle：全部为手写 NumPy 数学原语，不引用 expr/compile --------------- #

def _softmax_oracle(z: np.ndarray) -> np.ndarray:
    # 稳定 log-sum-exp，独立于「树重写 exp(z-max)」的代码路径
    z64 = np.asarray(z, dtype=np.float64)
    m = np.max(z64, axis=-1, keepdims=True)
    e = np.exp(z64 - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def _sigmoid_oracle(x: np.ndarray) -> np.ndarray:
    # 用 tanh 形式，独立于树的 1/(1+exp(-x)) 展开
    return 0.5 * (1.0 + np.tanh(np.asarray(x, dtype=np.float64) / 2.0))


def _gelu_oracle(x: np.ndarray) -> np.ndarray:
    # 与树 x·sigmoid(1.702x) 数学等价，但 sigmoid 用 tanh 路径
    x64 = np.asarray(x, dtype=np.float64)
    return x64 * 0.5 * (1.0 + np.tanh(1.702 * x64 / 2.0))


def _layernorm_oracle(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    mu = np.mean(x64, axis=-1, keepdims=True)
    var = np.mean((x64 - mu) ** 2, axis=-1, keepdims=True)
    return (x64 - mu) / np.sqrt(var + 1e-5)


def _relu_props() -> tuple[Property, ...]:
    def identity(out, inp):
        x = inp[0]
        mask = x > 0
        if not np.any(mask):
            return True, ""
        return bool(np.allclose(out[mask], x[mask], atol=1e-5)), "x>0 处 out≠x"
    def zero(out, inp):
        x = inp[0]
        mask = x <= 0
        if not np.any(mask):
            return True, ""
        return bool(np.all(out[mask] == 0)), "x<=0 处 out≠0"
    return (_nonnegative(), Property("正半轴恒等", identity), Property("负半轴归零", zero))


def _softmax_props() -> tuple[Property, ...]:
    def order(out, inp):
        x = np.asarray(inp[0])
        # 每行最大处仍是最大（保序）
        ok = bool(np.all(np.argmax(out, axis=-1) == np.argmax(x, axis=-1)))
        return ok, "argmax 未保持"
    return (_nonnegative(), _row_sums_to_one(), Property("保序(argmax)", order))


def _reduce_props(op: str) -> tuple[Property, ...]:
    if op == "max":
        def check(out, inp):
            x = np.asarray(inp[0])
            return bool(np.all(out <= np.max(x, axis=-1) + 1e-5)), "超出原数组最大值"
        return (Property("上界性", check),)
    # sum / mean：结果落在 [n*min, n*max] 的合理范围（mean 不乘 n）
    def bounded(out, inp):
        x = np.asarray(inp[0], dtype=np.float64)
        n = x.shape[-1]
        lo, hi = np.min(x, axis=-1), np.max(x, axis=-1)
        if op == "sum":
            lo, hi = lo * n, hi * n
        tol = 1e-4 * max(1.0, float(np.max(np.abs(hi))))
        return bool(np.all(out >= lo - tol) and np.all(out <= hi + tol)), "结果超出 [min,max] 边界"
    return (Property("边界性", bounded),)


def _layernorm_props() -> tuple[Property, ...]:
    def zero_mean(out, inp):
        m = out.astype(np.float64).mean(axis=-1)
        return bool(np.allclose(m, 0.0, atol=1e-3)), f"均值偏离0: {np.round(m.ravel()[:3],4).tolist()}"
    def unit_var(out, inp):
        v = out.astype(np.float64).var(axis=-1)
        return bool(np.allclose(v, 1.0, atol=1e-3)), f"方差偏离1: {np.round(v.ravel()[:3],4).tolist()}"
    return (Property("零均值", zero_mean), Property("单位方差", unit_var))


SEMANTICS: dict[str, SemanticSpec] = {
    "relu": SemanticSpec("relu", lambda x: np.maximum(x, 0), _relu_props()),
    "softmax": SemanticSpec("softmax", _softmax_oracle, _softmax_props()),
    # oracle 内部强制 float64 累加：金标准必须在高精度下算，否则相消时
    # float32 输入会让 oracle 自身丢精度，失去「黄金」资格。
    "reduce_sum": SemanticSpec("reduce_sum", lambda x: np.add.reduce(np.asarray(x, dtype=np.float64), axis=-1), _reduce_props("sum")),
    "reduce_mean": SemanticSpec("reduce_mean", lambda x: np.add.reduce(np.asarray(x, dtype=np.float64), axis=-1) / x.shape[-1], _reduce_props("mean")),
    "reduce_max": SemanticSpec("reduce_max", lambda x: np.maximum.reduce(np.asarray(x, dtype=np.float64), axis=-1), _reduce_props("max")),
    "matmul": SemanticSpec("matmul", lambda a, b: np.einsum(
        "ik,kj->ij", np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))),
    "add_broadcast": SemanticSpec("add_broadcast", lambda a, b: a + b),
    "gather": SemanticSpec("gather", lambda data, idx: np.take(data, idx, axis=0)),
    # 饱和到端点(0/1、-1/1)是 float32 舍入的正常现象，用闭区间
    "sigmoid": SemanticSpec("sigmoid", _sigmoid_oracle, (_range(0.0, 1.0, strict=False),)),
    "tanh": SemanticSpec("tanh", lambda x: np.tanh(x), (_range(-1.0, 1.0, strict=False),)),
    "gelu": SemanticSpec("gelu", _gelu_oracle),
    "layernorm": SemanticSpec("layernorm", _layernorm_oracle, _layernorm_props()),
}


def get_semantic(name: str) -> SemanticSpec | None:
    return SEMANTICS.get(name)
