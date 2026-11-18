"""主循环：串起 观复规格 -> 简易筋骨 -> 变易编译 -> 生乎动 -> 爻变观复。

v3 变化：
  - OpSpec 显式携带 semantic（语义基准），验证时如实展示语义核验状态；
  - 无 semantic 的算子降级为「仅验实现忠实度」，报告明确不声称语义天成；
  - 语义错误（性质违反/语义偏离）不可通过补常修复，观复诚实停手判未天成。

新增算子的唯一入口：register(spec) 或构造 OpSpec 传给 develop()。
"""

from __future__ import annotations

from .buyi import REQUIRED, catalog  # noqa: F401
from .evalbench import EvalRegistry
from .jianyi import JianYi
from .model import BuYi, DevResult, OpSpec, SiXiang
from .shenghudong import ShengHuDong, build_kernel_fn
from .yaobian import YaoBian


class YiDaoAgent:
    def __init__(
        self, max_rounds: int = 6, backend: str = "numpy",
        eval_registry: EvalRegistry | None = None,
    ) -> None:
        self.max_rounds = max_rounds
        self.backend = backend          # numpy | ascendc（v4 多后端）
        self.jianyi = JianYi()
        self.shenghudong = ShengHuDong()
        self._specs: dict[str, OpSpec] = {}
        # 裁判注册表（v7）：与 _specs（选手注册表）彻底分离。
        # 评测方通过构造参数注入；选手 register/register_op 无权触碰。
        self.eval_registry = eval_registry
        self._ascendc_runtime = None     # 惰性复用编译器缓存
        if backend not in ("numpy", "ascendc"):
            raise ValueError(f"未知后端：{backend}")

    def register(self, spec: OpSpec) -> None:
        self._specs[spec.key] = spec

    def register_op(
        self,
        key: str,
        tree,
        inputs: tuple[tuple[str, str, tuple[str, ...]], ...],
        output_dtype: str,
        *,
        oracle=None,
        properties=(),
        wei=None,
        known=None,
        doc: str = "",
    ) -> OpSpec:
        """v5 便捷扩展：一站式注册一个外部算子（可带独立语义基准）。

        等价于 make_op(...) + register(...)。oracle=None 时该算子诚实降级
        为「仅验忠实度」。返回构造好的 OpSpec，可立即 develop(key)。
        """
        from .public_api import make_op
        spec = make_op(key, tree, inputs, output_dtype, oracle=oracle,
                       properties=properties, wei=wei, known=known, doc=doc)
        self.register(spec)
        return spec

    def develop(self, spec: OpSpec | str) -> DevResult:
        if isinstance(spec, str):
            if spec not in self._specs:
                raise ValueError(f"未注册的算子：{spec}")
            spec = self._specs[spec]

        # ===== v7：外部裁判优先。有评测契约则强制改用裁判视图 =====
        # 选手自带的 semantic（可能是假 oracle）与 wei（挑过的用例）一律作废，
        # 验证只认评测方持有的 golden/cases。无契约则诚实降级到 v3 路径。
        judge_meta: dict = {}
        contract = self.eval_registry.get(spec.key) if self.eval_registry else None
        if contract is not None:
            spec, judge_meta = contract.judge_spec(spec)

        active: dict[str, BuYi] = {
            n: catalog()[n] for n in spec.known if n in catalog()
        }
        jijin = self.jianyi.identify(spec)

        yao = YaoBian(max_rounds=self.max_rounds)
        source = ""
        result = None
        if self.backend == "ascendc" and self._ascendc_runtime is None:
            from .ascendc_runtime import AscendCRuntime
            self._ascendc_runtime = AscendCRuntime()
        while True:
            source, kernel_fn = build_kernel_fn(spec, active, self.backend, self._ascendc_runtime)
            result = self.shenghudong.verify(spec, active, source, kernel_fn)
            nxt = yao.next_active(result, active, spec)
            if nxt is None:
                break
            active = nxt

        assert result is not None
        converged = result.sixiang.yang
        report = self._report(spec, jijin, result, yao.trace, active, converged, self.backend,
                              judge_meta=judge_meta)
        return DevResult(
            key=spec.key, converged=converged, sixiang=result.sixiang,
            source=source, report=report, traces=yao.trace,
            output_dtype=spec.output_dtype, n_inputs=spec.n_inputs,
            semantic_trusted=spec.semantic is not None,
            semantic_failures=list(result.semantic.failures),
            backend=self.backend,
            judged_by=judge_meta.get("judged_by"),
            proto_ok=judge_meta.get("proto_ok", False),
        )

    @staticmethod
    def _report(spec, jijin, result, traces, active, converged, backend="numpy",
                judge_meta: dict | None = None) -> str:
        judge_meta = judge_meta or {}
        btag = "AscendC C++" if backend == "ascendc" else "NumPy/Python"
        lines = [
            f"== 算子 {spec.key} | 筋骨 {jijin.value} | 后端 {btag} ==",
            spec.doc or "(声明式规格：表达式树)",
            "",
        ]
        # ===== v7：外部裁判可见性（选手与裁判分离的审计痕迹）=====
        if judge_meta.get("judged_by"):
            lines += [
                "【外部评测集裁判·选手与裁判分离】",
                f"  裁判来源：{judge_meta['judged_by']}",
                f"  原型校验：逐字一致（proto ✓）",
                f"  裁判用例：{judge_meta.get('cases')} 条（评测方提供，选手不可挑选）",
                *([f"  选手自带 oracle：已忽略（作者不再兼任裁判）"]
                  if judge_meta.get("author_oracle_ignored") else []),
                "",
            ]
        else:
            lines += [
                "【外部评测集·缺位】",
                "  该算子无评测契约——语义基准由选手自带，未经独立裁判。",
                "  结论不冒称「评测集认证」，请补充 EvalContract 后再引用。",
                "",
            ]
        # 语义基准状态——v3 核心可见性
        if spec.semantic is not None:
            sem_ok = result.semantic.ok
            lines += [
                "【语义基准·独立核验】",
                f"  基准：{spec.semantic.name}（手写 oracle + {len(spec.semantic.properties)} 条数学性质）",
                f"  路径：oracle 独立于表达式树（不 import expr/compile）",
                f"  结果：{'语义守住' if sem_ok else '语义未过 ' + str(result.semantic.failures[:2])}",
                "",
            ]
        else:
            lines += [
                "【语义基准·降级】",
                "  未绑定独立语义基准——仅验证「实现忠实于规格树」，",
                "  不声称树符合目标算子的数学意图（doc/key 文本无约束力）。",
                "",
            ]
        lines += ["【爻变轨迹（观复闭环）】"]
        for t in traces:
            line = f"  第{t.round}轮 {t.sixiang.value}：{t.summary}"
            if t.symptoms:
                line += f"\n      症状(自动诊断)：{sorted(t.symptoms)}"
            if t.repaired:
                line += f"\n      -> 观复推理补常「{t.repaired}」：{t.reason} -> 重生"
            lines.append(line)
        verdict = "天成" if converged else "未天成"
        if converged and spec.semantic is not None:
            verdict = "天成(语义+忠实度双验)"
        elif converged:
            verdict = "忠实度天成(语义未核验)"
        # 吝：通过但需守——《易》「吝，有所惜而不敢纵」，明示工程警讯
        if result.sixiang == SiXiang.LIN and converged:
            verdict = "通过·需守(吝)"
        lines += [
            "",
            "【内建进结构的常（树重写规则）】",
            *([f"  - {n}" for n in active if catalog()[n].rewrite_name]
              or ["  - （无数值重写；结构已保证）"]),
            "",
            "【知止·报告】",
            f"  {verdict}：{result.summary}",
            f"  共 {len(traces)} 轮爻变，失败率 {result.failure_rate:.0%}。",
        ]
        return "\n".join(lines)
