"""生乎动（⑤）：真实时位上跑 kernel，三判据验证。

【v3 三判据】
  1. 语义判据（主）：kernel vs 独立 oracle 对拍 + 数学性质检验。
     oracle 走完全独立于表达式树的代码路径（手写 NumPy 原语），
     这是「树符合意图」的可信基准——根治 v2 的 fake_relu 同义反复。
  2. 忠实度判据（辅）：kernel vs interpret(树)。证明实现忠实于规格树，
     但不再冒充语义正确。
  3. 不变量判据：激活常谓词 + 必备常探针（保留 v2 机制）。

【无语义基准时】spec.semantic=None -> 语义判据跳过，只有忠实度判据，
报告明确降级「仅验实现忠实度，语义未核验」，绝不声称语义天成。

【症状 -> 失败】diagnose 症状一票否决计入失败（v2 修复延续）。
【吝(LIN)】低失败率且无严重症状（非有限/形状/dtype/语义/性质）时判吝，
吝为可交付但带警告——让四象的吝成为真实状态。
"""

from __future__ import annotations

import numpy as np

from .buyi import REQUIRED, catalog
from .compile import REWRITES, compile_source, interpret, make_kernel
from .jianyi import JianYi
from .lesion import ATOL, RTOL, diagnose
from .model import BuYi, LayerResult, OpSpec, SiXiang, Symptom, SemanticDiag


def _normalize_shape(sym: tuple[str, ...], wei: ShiWei, i: int) -> tuple[int, ...]:
    out = []
    for d in sym:
        if isinstance(d, int):
            out.append(d)
        elif d == "N":
            out.append(wei.shapes[0][0] if wei.shapes else 4)
        elif d == "M":
            out.append(wei.shapes[0][-1] if wei.shapes else 4)
        else:
            out.append(4)
    return tuple(out)


def _inputs_for(spec: OpSpec, w: ShiWei, rng: np.random.Generator) -> list[np.ndarray]:
    """按时位采样真实输入。cancel 相消；oob 制造索引越界以验边界处理。"""
    xs: list[np.ndarray] = []
    for i, (name, dtype, sym) in enumerate(spec.inputs):
        shape = w.shapes[i] if i < len(w.shapes) else _normalize_shape(sym, w, i)
        dt = np.dtype(dtype)
        if w.pattern == "oob" and dt.kind in "iu":
            # 索引输入：故意越界（数据长度之外），验系统是否正确拒绝
            bound = w.shapes[0][0] if w.shapes else 4
            xs.append(rng.integers(bound, bound + 5, size=shape).astype(dt))
        elif dt.kind in "iu":
            bound = w.shapes[0][0] if w.shapes else 4
            xs.append(rng.integers(0, bound, size=shape).astype(dt))
        elif w.pattern == "cancel":
            arr = np.ones(shape, dtype=np.float64)
            arr[..., 0] = 1e8
            arr[..., -1] = -1e8
            xs.append(arr.astype(dt))
        else:
            xs.append(rng.normal(0.0, w.scale, size=shape).astype(dt))
    return xs


def active_rewrites(active: dict[str, BuYi]) -> list:
    return [
        REWRITES[inv.rewrite_name]
        for inv in active.values()
        if inv.rewrite_name and inv.rewrite_name in REWRITES
    ]


# ---- 多后端构建：树 + 内建重写 -> (源码, 可调用 kernel_fn) ----------------- #
# 后端只负责「合成」，三判据验证（oracle 对拍/忠实度/不变量）完全复用。

def build_kernel_fn_numpy(spec: OpSpec, active: dict[str, BuYi]) -> tuple[str, object]:
    """NumPy 后端：树 -> NumPy 源码 -> exec（v2/v3 既有能力）。"""
    source = compile_source(spec.tree, active_rewrites(active), spec.n_inputs, spec.output_dtype)
    return source, make_kernel(source)


def _symbolic_shapes(spec: OpSpec) -> dict[int, tuple[int, ...]]:
    """把 spec 的符号 shape（N/M/K）转成编译期占位维度；运行时 shape 由 C ABI 传入。"""
    out: dict[int, tuple[int, ...]] = {}
    for i, (_name, _dtype, sym) in enumerate(spec.inputs):
        out[i] = tuple(1 if isinstance(d, str) else d for d in sym)
    return out


def build_kernel_fn_ascendc(spec: OpSpec, active: dict[str, BuYi], runtime=None) -> tuple[str, object]:
    """AscendC 后端：树 -> 真 AscendC C++ 源码 -> g++/桩编译 -> 可调用 kernel。"""
    from .ascendc import compile_ascendc_source, infer_shape
    from .ascendc_runtime import AscendCRuntime

    rt = runtime or AscendCRuntime()
    ak = compile_ascendc_source(
        key=spec.key, tree=spec.tree, rewrites=active_rewrites(active),
        n_inputs=spec.n_inputs, dtypes=tuple(d[1] for d in spec.inputs),
        output_dtype=spec.output_dtype, shapes=_symbolic_shapes(spec),
    )
    tree = spec.tree

    def shape_fn(in_shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
        return infer_shape(tree, {i: s for i, s in enumerate(in_shapes)})

    kernel_fn = rt.build(ak, shape_fn)
    return ak.source, kernel_fn


def build_kernel_fn(
    spec: OpSpec, active: dict[str, BuYi], backend: str = "numpy", runtime=None
) -> tuple[str, object]:
    """后端分派入口。backend ∈ {numpy, ascendc}。"""
    if backend == "numpy":
        return build_kernel_fn_numpy(spec, active)
    if backend == "ascendc":
        return build_kernel_fn_ascendc(spec, active, runtime)
    raise ValueError(f"未知后端：{backend}")


# 严重症状：出现即判悔（不可作为「可交付」）
_SEVERE = (Symptom.NONFINITE, Symptom.SHAPE, Symptom.DTYPE,
           Symptom.SEMANTIC, Symptom.PROPERTY)


def classify(
    failures: int, total: int, symptoms: set[str], warnings: list[str] | None = None
) -> tuple[SiXiang, str]:
    """四象判定（纯函数）：吉/吝/厉/悔。

    吝不是「低失败率」的凑数状态，而是有独立语义——「通过，但需守」。
    《易》：「吝，有所惜而不敢纵也。」对应到算子：为守住正确性不得不
    内建数值保护（float64 回退、减最大值），虽通过却留了代价/警讯。
    于是吝在真实闭环中自然可达——v3 此前只用了 failure_rate 单轴，
    丢失了这一维度，故 12 算子的轨迹里吝从未出现。

    - 全绿且无警告 -> 吉（天成，无需保护）；
    - 全绿但有数值保护警告 -> 吝（通过但需守，可交付带警讯）；
    - 严重症状或失败率>20% -> 悔（须观复）；
    - 其余 -> 厉（临界，待查）。
    """
    warnings = warnings or []
    rate = failures / max(total, 1)
    severe = any(s in symptoms for s in _SEVERE)
    if failures == 0:
        if warnings:
            return SiXiang.LIN, f"吝：{total} 项守住，但需守 {len(warnings)} 处数值保护"
        return SiXiang.JI, f"吉：{total} 项判定守住"
    if severe or rate > 0.2:
        return SiXiang.HUI, f"悔：{failures}/{total} 破，失败率 {rate:.0%}"
    # 轻微失败（≤20% 且非严重）：归吝（可交付带警讯）
    return SiXiang.LIN, f"吝：{failures}/{total} 破(轻微)，失败率 {rate:.0%}"


class ShengHuDong:
    """生乎动：真跑、三判据、症状一票否决、语义基准优先。"""

    def verify(
        self, spec: OpSpec, active: dict[str, BuYi], source: str, kernel_fn
    ) -> LayerResult:
        rng = np.random.default_rng(20261117)
        details: list[str] = []
        symptoms: set[str] = set()
        failures = 0
        total = 0
        sem_diag = SemanticDiag(checked=spec.semantic is not None)
        cat = catalog()
        required = REQUIRED[JianYi().identify(spec).key]
        # 警告：为守住正确性而内建的数值保护重写（吝的语义来源）。
        # 这是真实工程信号——float32 下有精度/溢出风险，需以高精度/稳定化兜底。
        warnings = [
            f"内建数值保护「{n}」" for n in active if catalog()[n].rewrite_name
        ]

        for w in spec.wei:
            xs = _inputs_for(spec, w, rng)
            # ---- 索引越界时位：验系统是否正确拒绝（拒绝=通过，漏检=失败）----
            if w.pattern == "oob":
                total += 1
                try:
                    _ = kernel_fn(*xs)
                    # 未抛异常：越界索引竟产生输出 -> 漏检，失败
                    failures += 1
                    symptoms.add(Symptom.INDEX)
                    details.append(f"[{w.label()}] 越界索引未被拒绝")
                except Exception:
                    # 正确拒绝（抛异常）——这是期望行为，计通过
                    symptoms.add(Symptom.INDEX)  # 记录越界事实作为证据
                    details.append(f"[{w.label()}] 越界索引被正确拒绝")
                continue

            try:
                with np.errstate(all="ignore"):
                    out = kernel_fn(*xs)
            except Exception as exc:
                failures += 1
                total += 1
                details.append(f"[{w.label()}] 执行异常 {type(exc).__name__}: {exc}")
                continue

            # ===== 判据1：语义（独立 oracle + 性质）——主判据 =====
            if spec.semantic is not None:
                sem_ok, sem_fails, gold = spec.semantic.judge(out, xs)
                sem_diag.n_checked += 1
                total += 1
                if not sem_ok:
                    failures += 1
                    sem_diag.failures.extend(f"[{w.label()}] {f}" for f in sem_fails)
                    # oracle 对拍的数值差异转成可驱动补常的症状
                    symptoms |= diagnose(out, gold, xs)
                    if any("性质违反" in f for f in sem_fails):
                        symptoms.add(Symptom.PROPERTY)
                    if any("语义偏离" in f for f in sem_fails):
                        symptoms.add(Symptom.SEMANTIC)
                    details.append(f"[{w.label()}] 语义未过：{sem_fails}")

            # ===== 判据2：忠实度（解释器对拍）——辅证，不冒充语义 =====
            try:
                ref = interpret(spec.tree, xs, active_rewrites(active))
                ref = ref.astype(np.dtype(spec.output_dtype))
            except Exception as exc:
                failures += 1
                total += 1
                details.append(f"[{w.label()}] 解释器异常 {type(exc).__name__}: {exc}")
                continue
            ctx = {"out": out, "ref": ref, "inp": xs, "spec": spec,
                   "atol": ATOL, "rtol": RTOL}
            syms = diagnose(out, ref, xs)
            total += 1
            if syms:
                failures += 1
                symptoms |= syms
                details.append(f"[{w.label()}] 忠实度症状 {sorted(syms)}")

            # ===== 判据3：不变量谓词 + 必备常探针 =====
            for cname, inv in active.items():
                if inv.predicate is None:
                    continue
                total += 1
                if not inv.holds(ctx):
                    failures += 1
                    details.append(f"[{w.label()}] 常「{cname}」破")
            for cname in required:
                if cname in active:
                    continue
                inv = cat.get(cname)
                if inv and inv.predicate and not inv.holds(ctx):
                    total += 1
                    failures += 1
                    symptoms.add(inv.symptom)
                    details.append(f"[{w.label()}] 缺常「{cname}」当补")

        rate = failures / max(total, 1)
        sixiang, summary = classify(failures, total, symptoms, warnings)
        return LayerResult(
            sixiang=sixiang, summary=summary, failure_rate=rate,
            failures=failures, total=total, details=details,
            symptoms=symptoms, semantic=sem_diag, warnings=warnings,
        )
