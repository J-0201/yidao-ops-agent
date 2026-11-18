"""爻变（⑥ 观复）：破常 -> 补常 -> 重生 的真实闭环。

从症状自动推理该补哪条常：症状 -> symptom_to_const() 映射（由不变量目录
的 symptom 字段导出，非题级人工标注）-> 在「本筋骨必备但尚未激活」的常里
选。补的是常（树重写规则），回到编译层重生，不是就地打补丁。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .buyi import REQUIRED, catalog, symptom_to_const
from .jianyi import JianYi
from .model import BuYi, CycleTrace, LayerResult, OpSpec, SiXiang


@dataclass
class YaoBian:
    max_rounds: int = 6
    trace: list[CycleTrace] = field(default_factory=list)

    def next_active(
        self, result: LayerResult, active: dict[str, BuYi], spec: OpSpec
    ) -> dict[str, BuYi] | None:
        """吉/吝 -> None（收手）；厉/悔 -> 补常后的新 active（回到编译层重生）。"""
        rnd = len(self.trace) + 1
        self.trace.append(CycleTrace(
            round=rnd, sixiang=result.sixiang, summary=result.summary,
            symptoms=set(result.symptoms),
        ))
        if result.sixiang.yang:
            return None
        if rnd >= self.max_rounds:
            return None  # 达上限仍未收敛 -> 诚实停手，判未天成

        picked = self.infer(result.symptoms, active, spec)
        if picked is None:
            return None  # 无人能治 -> 无法收敛，诚实停手
        self.trace[-1].repaired = picked
        self.trace[-1].reason = (
            f"症状 {sorted(result.symptoms)} 命中常「{picked}」的治愈能力"
        )
        nxt = dict(active)
        nxt[picked] = catalog()[picked]
        return nxt

    def infer(
        self, symptoms: set[str], active: dict[str, BuYi], spec: OpSpec
    ) -> str | None:
        """证据驱动的确定性推理：在必备但缺失的常里找症状能治者。"""
        if not symptoms:
            return None
        required = REQUIRED[JianYi().identify(spec).key]
        missing = [n for n in required if n not in active and n in catalog()]
        smap = symptom_to_const()
        for sym in symptoms:
            cand = smap.get(sym)
            if cand in missing:
                return cand
        return None
