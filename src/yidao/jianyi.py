"""简易层（③ 依理）：从表达式树的【结构】推导筋骨——不是字符串匹配。

v1 硬伤：`"softmax" in text` / `"@" in text` 子串判断，可被 `math_def` 里
无关文本欺骗。v2 直接看 OpSpec.tree 的节点类型与形状关系，这是结构事实，
无法用注释伪造。
"""

from __future__ import annotations

from .expr import Elem, Expr, Gather, Input, MatMul, Reduce, count_ops, walk
from .model import JinGu, OpSpec


class JianYi:
    """依理：从表达式树结构推导筋骨 + 下刀路径。"""

    def identify(self, spec: OpSpec) -> JinGu:
        nodes = walk(spec.tree)
        ops = count_ops(spec.tree)
        has = lambda t: any(isinstance(n, t) for n in nodes)

        # 优先按最强的结构信号判定
        if has(MatMul):
            return JinGu.ZHEN
        if has(Gather):
            return JinGu.KAN
        # 归约 + 逐元素exp/除法 = 混合（softmax/layernorm 类：先归约后逐元素）
        n_reduce = ops.get("Reduce", 0)
        n_elem = ops.get("Elem", 0)
        if n_reduce > 0 and n_elem > 0:
            return JinGu.LI
        if n_reduce > 0:
            return JinGu.KUN
        if has(Elem) and n_elem > 1:
            # 多输入逐元素且存在两个不同 Input -> 广播（如 add_broadcast）
            inputs_used = {n.i for n in nodes if isinstance(n, Input)}
            if len(inputs_used) >= 2:
                return JinGu.XUN
            return JinGu.QIAN
        return JinGu.QIAN

    def path(self, jijin: JinGu) -> list[str]:
        return {
            JinGu.QIAN: ["tile 切分", "向量化", "边界 mask"],
            JinGu.KUN: ["选归约轴", "warp/block reduce", "累加精度", "空轴处理"],
            JinGu.ZHEN: ["tiling 分块", "内维匹配校验", "高精度累加", "边界 mask"],
            JinGu.XUN: ["右轴对齐", "扩张维 mask", "输出 shape 推导"],
            JinGu.KAN: ["索引边界校验", "重复索引独立取值", "边界 mask"],
            JinGu.LI: ["先归约后逐元素", "中间态放置", "数值稳定变换"],
        }[jijin]
