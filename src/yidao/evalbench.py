"""外部评测集裁判层（采用「选手与裁判分离」的评测契约设计）。

【v3–v6 的根本漏洞】
  SemanticSpec 由算子作者自己绑定：OpSpec.semantic = 作者传进来的 oracle，
  register_op(oracle=...) 允许作者自带「权威实现」。于是作者既是选手又是裁判。
  实测攻击：声称 ReLU、树写成 x*x 的 fake_relu，只要绑一个同样算 x*x 的
  假 oracle，v3–v6 全部判「天成」——因为系统验证的是「实现忠于作者给的 oracle」。
  这不是算子 bug，是制度 bug：裁判权交给了选手。

【v7 解法：评测集独立于算子注册】
  引入 EvalContract（评测契约），评测契约三件套：
    - proto  : 算子原型（输入 dtype / 输出 dtype / 输入个数 / 从属类别），
               选手的 OpSpec 必须与它【逐字一致】，不一致直接拒收（硬伤判定）。
    - golden : 裁判方持有的权威参考实现，选手【不可注入、不可覆盖】。
    - cases  : 裁判方提供的测试时位与采样分布，选手不能自定义「只挑好过的用例」。
  裁判权归属：EvalRegistry 由评测方独立注册；agent.register(spec) 只登记选手。
  develop() 时，若该 key 有评测契约，则【强制】用契约的 golden 与 cases 验证，
  选手 self.semantic 被降级为「自检参考」，不参与裁判，也无法篡改结论。

【兼容降级】
  - 无评测契约的算子：回退到 v3 语义基准路径（保留 register_op 自带 oracle），
    但报告必须标注「未经外部评测集裁判」——诚实降级，不冒称可信。
  - golden / cases 仍由人工编写，是可信终点，无法自我证明（与外部评测集同构）。

本模块不 import expr / compile 的求值路径；golden 用手写 NumPy 数学原语。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .model import OpSpec, ShiWei
from .semantic import Property, SemanticSpec


class ContractError(ValueError):
    """评测契约违规：proto 与算子原型不一致，或契约本身非法。"""


@dataclass(frozen=True)
class EvalProto:
    """算子原型（对标 proto.yaml）。选手的 OpSpec 必须逐字匹配。"""

    inputs: tuple[tuple[str, str, tuple[str, ...]], ...]  # (name, dtype, shape符号)
    output_dtype: str

    def assert_matches(self, spec: OpSpec) -> list[str]:
        """逐字校验选手规格与评测原型；返回不匹配项（空=一致）。"""
        errs: list[str] = []
        if len(spec.inputs) != len(self.inputs):
            errs.append(f"输入个数 {len(spec.inputs)}≠原型 {len(self.inputs)}")
        for i, ((sname, sdtype, sshape), (pname, pdtype, pshape)) in enumerate(
            zip(spec.inputs, self.inputs)
        ):
            if sname != pname:
                errs.append(f"输入{i}名 '{sname}'≠原型 '{pname}'")
            if sdtype != pdtype:
                errs.append(f"输入{i}dtype '{sdtype}'≠原型 '{pdtype}'")
            if tuple(sshape) != tuple(pshape):
                errs.append(f"输入{i}shape符号 {tuple(sshape)}≠原型 {tuple(pshape)}")
        if spec.output_dtype != self.output_dtype:
            errs.append(f"输出dtype '{spec.output_dtype}'≠原型 '{self.output_dtype}'")
        return errs


@dataclass(frozen=True)
class EvalCase:
    """一条裁判用例：时位 + 采样分布（对标 cases.yaml 的一条）。"""

    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    boundary: str = "常规"
    scale: float = 1.0
    pattern: str = "normal"  # normal | cancel | oob

    def to_shiwei(self) -> ShiWei:
        return ShiWei(self.shapes, self.dtypes, boundary=self.boundary,
                      scale=self.scale, pattern=self.pattern)


@dataclass(frozen=True)
class EvalContract:
    """评测契约：proto + golden + cases + 性质，由【评测方】持有。"""

    key: str
    proto: EvalProto
    golden: Callable[..., np.ndarray]           # 裁判权威实现，float64 语义
    cases: tuple[EvalCase, ...]
    properties: tuple[Property, ...] = ()
    atol: float = 1e-4
    rtol: float = 1e-4
    source: str = "external-bench"             # 评测集来源标识

    def __post_init__(self) -> None:
        if not self.cases:
            raise ContractError(f"评测契约 {self.key} 无任何用例")
        for c in self.cases:
            if len(c.shapes) != len(self.proto.inputs):
                raise ContractError(
                    f"评测契约 {self.key} 用例 shape 个数 {len(c.shapes)}"
                    f"≠原型输入数 {len(self.proto.inputs)}")

    def check_proto(self, spec: OpSpec) -> None:
        """选手规格必须与原型逐字一致，否则拒收（硬伤级判定）。"""
        errs = self.proto.assert_matches(spec)
        if errs:
            raise ContractError(
                f"算子 '{spec.key}' 与评测原型不一致（评测拒绝注册）：" + "; ".join(errs))

    def semantic(self) -> SemanticSpec:
        """把裁判 golden 包装成 SemanticSpec，供现有验证链路消费。"""
        return SemanticSpec(
            name=f"{self.source}:{self.key}",
            oracle=self.golden,
            properties=self.properties,
            atol=self.atol, rtol=self.rtol,
        )

    def judge_spec(self, spec: OpSpec) -> tuple[OpSpec, dict[str, Any]]:
        """生成「裁判视图」规格：用契约的 golden/cases 覆盖选手自带内容。

        关键：选手的 spec.semantic（可能是假 oracle）被【丢弃】；
        spec.wei（选手挑选的时位）被【丢弃】，改用裁判 cases。
        返回的 spec 副本仅供本轮验证，不修改注册表里的原始选手规格。
        """
        self.check_proto(spec)
        judged = OpSpec(
            key=spec.key, tree=spec.tree, inputs=spec.inputs,
            output_dtype=spec.output_dtype,
            wei=[c.to_shiwei() for c in self.cases],
            known=set(spec.known), doc=spec.doc,
            semantic=self.semantic(),
        )
        meta = {
            "judged_by": self.source,
            "proto_ok": True,
            "cases": len(self.cases),
            "author_oracle_ignored": spec.semantic is not None,
        }
        return judged, meta


class EvalRegistry:
    """评测集注册表——裁判方持有，与 YiDaoAgent 的选手注册表完全分离。"""

    def __init__(self) -> None:
        self._contracts: dict[str, EvalContract] = {}

    def register(self, contract: EvalContract) -> None:
        """注册评测契约。评测方专属接口；选手无法通过这些接口篡改 golden。"""
        if contract.key in self._contracts:
            raise ContractError(f"评测契约 {contract.key} 已存在，拒绝覆盖")
        self._contracts[contract.key] = contract

    def get(self, key: str) -> EvalContract | None:
        return self._contracts.get(key)

    def has(self, key: str) -> bool:
        return key in self._contracts

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._contracts))

    def __len__(self) -> int:
        return len(self._contracts)
