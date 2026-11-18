"""属性随机测试层（v5）——把「覆盖」从固定种子推进到随机性质对拍。

【为什么需要这一层】第三方评估对 v3/v4 反复点名的一条边界是：
  「覆盖靠固定时位 + 固定种子（20261117），未触及的输入区域错误仍可能漏网」。
确定性验证内核 ShengHuDong.verify 用固定种子是可复现性的需要，不能改。
于是 v5 增加这一层——不改内核，而是在更大、更随机的输入空间上做
「三方对拍」(property-based randomized differential testing)：

    numpy kernel  ──┐
    独立 oracle     ├─ 三方一致 ⇒ 该算子在这一随机输入区域可信
    解释器 interpret┘

随机维度有三：
  1. seed          —— 多个随机种子，摆脱单一 20261117；
  2. shape         —— 每轮为符号维度 (N/M/K) 随机取新值，摆脱固定 shape；
  3. distribution  —— normal / uniform / large / near_zero / zeros / cancel
                      多种数值分布，主动探测溢出、相消、退化边界。

这仍不是形式化证明（评估承认的固有上限），但把「外验的覆盖广度」实质性
提高了一个数量级，且完全自动化、可复用、对外部新算子同样生效。

【诚实边界】随机测试只能证伪不能证明；它扩大了发现错误的面，但不声称
穷尽。容差按 float32 输出设定（覆盖层用于抓结构性错误，非精度微差）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .buyi import REQUIRED, catalog
from .compile import interpret
from .jianyi import JianYi
from .model import BuYi, OpSpec
from .shenghudong import active_rewrites, build_kernel_fn


# ---- 输入分布：主动探测不同数值区域 -------------------------------------- #

DISTRIBUTIONS = ("normal", "uniform", "large", "near_zero", "zeros", "cancel")


def _sample_float(dist: str, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    if dist == "normal":
        return rng.normal(0.0, 1.0, size=shape)
    if dist == "uniform":
        return rng.uniform(-2.0, 2.0, size=shape)
    if dist == "large":
        return rng.normal(0.0, 1e3, size=shape)
    if dist == "near_zero":
        return rng.normal(0.0, 1e-6, size=shape)
    if dist == "zeros":
        return np.zeros(shape)
    if dist == "cancel":
        arr = rng.normal(0.0, 1.0, size=shape)
        if arr.shape[-1] >= 2:
            arr[..., 0] = 1e8
            arr[..., -1] = -1e8
        return arr
    raise ValueError(f"未知分布：{dist}")


def _symbols_of(spec: OpSpec) -> list[str]:
    seen: list[str] = []
    for _name, _dtype, sym in spec.inputs:
        for d in sym:
            if isinstance(d, str) and d not in seen:
                seen.append(d)
    return seen


def _random_shapes(
    spec: OpSpec, rng: np.random.Generator, lo: int = 2, hi: int = 8
) -> tuple[tuple[int, ...], ...]:
    """为每个符号维度随机取值，再按各输入的符号表达式拼出真实 shape。

    同一 spec 的所有输入共用符号映射——matmul 的 (N,K)@(K,M)、
    add_broadcast 的 (N,M)+(M,) 因此自动保持维度和广播一致。
    """
    sym_val = {s: int(rng.integers(lo, hi + 1)) for s in _symbols_of(spec)}
    shapes = []
    for _name, _dtype, sym in spec.inputs:
        shapes.append(tuple(sym_val[d] if isinstance(d, str) else d for d in sym))
    return tuple(shapes)


def _sample_inputs(
    spec: OpSpec, shapes: tuple[tuple[int, ...], ...], dist: str, rng: np.random.Generator
) -> list[np.ndarray]:
    """按时位 shape + 分布采样；整数（索引）输入单独处理并保证不越界。"""
    # 索引输入的合法上界 = data 输入第 0 维
    data_rows = shapes[0][0] if shapes else 2
    xs: list[np.ndarray] = []
    for (name, dtype, _sym), shape in zip(spec.inputs, shapes):
        dt = np.dtype(dtype)
        if dt.kind in "iu":
            xs.append(rng.integers(0, data_rows, size=shape).astype(dt))
        else:
            xs.append(_sample_float(dist, shape, rng).astype(dt))
    return xs


def make_full_active(spec: OpSpec) -> dict[str, BuYi]:
    """构造「全内建」不变量集：声明的 known + 本筋骨必备 + 所有带重写的数值保护。

    代表算子最终交付形态（所有该内建的都内建了）。属性测试验证的正是这个
    最终形态在随机输入下是否恒对。外部新算子无需手工装配即可受测。
    """
    cat = catalog()
    active: dict[str, BuYi] = {n: cat[n] for n in spec.known if n in cat}
    for n in REQUIRED[JianYi().identify(spec).key]:
        if n in cat:
            active[n] = cat[n]
    for n, inv in cat.items():
        if inv.rewrite_name:
            active[n] = inv
    return active


# ---- 对拍结果 ------------------------------------------------------------ #

@dataclass
class CoverageCase:
    seed: int
    distribution: str
    shapes: tuple[tuple[int, ...], ...]
    passed: bool
    reason: str = ""


@dataclass
class CoverageReport:
    key: str
    cases: list[CoverageCase] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def rate(self) -> float:
        return self.passed / max(self.total, 1)

    @property
    def distributions(self) -> list[str]:
        return sorted({c.distribution for c in self.cases})

    @property
    def n_shapes(self) -> int:
        return len({c.shapes for c in self.cases})

    @property
    def ok(self) -> bool:
        return self.total > 0 and self.passed == self.total


# ---- 主入口 -------------------------------------------------------------- #

def coverage_check(
    spec: OpSpec,
    active: dict[str, BuYi] | None = None,
    *,
    seeds: tuple[int, ...] = (1, 7, 20261117, 99991),
    n_shapes: int = 3,
    distributions: tuple[str, ...] = DISTRIBUTIONS,
    backend: str = "numpy",
    runtime=None,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> CoverageReport:
    """对一个算子做属性随机对拍，返回结构化覆盖报告。

    对每个 (seed, shape 变体, distribution) 组合：编译 kernel 输出 vs 独立 oracle
    金标准 vs 解释器参考，三方一致且形状/dtype 正确即通过。
    """
    active = active if active is not None else make_full_active(spec)
    _source, kernel_fn = build_kernel_fn(spec, active, backend, runtime)
    rewrites = active_rewrites(active)
    report = CoverageReport(key=spec.key)

    for seed in seeds:
        rng = np.random.default_rng(seed)
        for _ in range(n_shapes):
            shapes = _random_shapes(spec, rng)
            for dist in distributions:
                xs = _sample_inputs(spec, shapes, dist, rng)
                reason = ""
                try:
                    with np.errstate(all="ignore"):
                        out = np.asarray(kernel_fn(*xs))
                        ref = interpret(spec.tree, xs, rewrites)
                    ok = True
                    if out.shape != ref.shape:
                        ok, reason = False, f"shape {out.shape} != ref {ref.shape}"
                    if ok and str(out.dtype) != spec.output_dtype:
                        ok, reason = False, f"dtype {out.dtype} != {spec.output_dtype}"
                    if ok:
                        ref_c = ref.astype(out.dtype)
                        if not np.allclose(out.astype(np.float64), ref_c.astype(np.float64), atol=atol, rtol=rtol, equal_nan=True):
                            ok, reason = False, f"忠实度偏离 interpret（{dist}）"
                        # 语义基准存在时才做独立 oracle 对拍——无基准者只验忠实度，
                        # 绝不拿 interpret 自己当 gold 自我对拍（那会同义反复）。
                        if ok and spec.semantic is not None:
                            gold = spec.semantic.gold(xs)
                            gold_c = gold.astype(out.dtype)
                            if not np.allclose(out.astype(np.float64), gold_c.astype(np.float64), atol=atol, rtol=rtol, equal_nan=True):
                                ok, reason = False, f"语义偏离 oracle（{dist}）"
                except Exception as exc:
                    ok, reason = False, f"执行异常 {type(exc).__name__}: {exc}"
                report.cases.append(CoverageCase(seed, dist, shapes, ok, reason))
    return report


def coverage_all(
    specs: dict[str, OpSpec], **kwargs
) -> dict[str, CoverageReport]:
    """对规格库全部算子跑覆盖测试。runtime 复用（若 ascendc 后端）。"""
    runtime = kwargs.pop("runtime", None)
    out: dict[str, CoverageReport] = {}
    for key, spec in specs.items():
        out[key] = coverage_check(spec, runtime=runtime, **kwargs)
    return out


def format_report(reports: dict[str, CoverageReport]) -> str:
    """人类可读的覆盖报告。"""
    lines = ["== 属性随机覆盖报告（v5）=="]
    total = passed = 0
    for key, r in reports.items():
        total += r.total
        passed += r.passed
        flag = "OK" if r.ok else "FAIL"
        lines.append(
            f"  [{flag}] {key:<14} {r.passed}/{r.total} "
            f"(seed×{len(set(c.seed for c in r.cases))} "
            f"shape×{r.n_shapes} dist×{len(r.distributions)})"
        )
        for c in r.cases:
            if not c.passed:
                lines.append(f"        - seed={c.seed} {c.distribution} {c.shapes}: {c.reason}")
    lines.append(f"  合计 {passed}/{total} 通过")
    return "\n".join(lines)
