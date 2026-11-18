"""核心数据模型（v3）。

在 v2 基础上引入独立语义基准：
  - SemanticSpec  语义基准：手写 oracle + 数学性质（见 semantic.py）
  - OpSpec        增加 semantic 字段——「意图」从 doc 文本升级为可执行对象
  - SemanticDiag  语义诊断层结果（语义判据 + 性质判据）
  - LayerResult   增加 semantic_ok / semantic_fails 字段

易理骨架不变：阴阳/四象/时位/筋骨/不易。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TYPE_CHECKING

from .expr import Expr

if TYPE_CHECKING:
    from .semantic import SemanticSpec


class Polarity(Enum):
    YANG = "阳"
    YIN = "阴"

    def flip(self) -> "Polarity":
        return Polarity.YIN if self is Polarity.YANG else Polarity.YANG


class SiXiang(Enum):
    """四象：吉吝厉悔。yang 者可交付，yin 者须观复。"""

    JI = "吉"
    LIN = "吝"
    LI = "厉"
    HUI = "悔"

    @property
    def yang(self) -> bool:
        return self in (SiXiang.JI, SiXiang.LIN)

    @property
    def polarity(self) -> Polarity:
        return Polarity.YANG if self.yang else Polarity.YIN


@dataclass(frozen=True)
class ShiWei:
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    backend: str = "numpy"
    boundary: str = "常规"
    scale: float = 1.0
    pattern: str = "normal"  # normal | cancel（numerical cancellation）| oob（索引越界）

    def label(self) -> str:
        return f"shape={self.shapes} dtype={self.dtypes[0]} 边界={self.boundary} scale={self.scale:g}"


class JinGu(Enum):
    """筋骨：从表达式树结构推导，取八卦为助记，不用于占筮。"""

    QIAN = "乾·逐元素"
    KUN = "坤·归约"
    ZHEN = "震·矩阵乘"
    XUN = "巽·广播"
    KAN = "坎·收集"
    LI = "离·混合"

    @property
    def key(self) -> str:
        return self.name


@dataclass
class BuYi:
    """不易：一条不变量。两条腿——谓词（验）+ 重写规则（内建）。"""

    name: str
    level: str  # struct | numeric
    builtin: str
    symptom: str
    rewrite_name: str | None = None
    predicate: Callable[[dict[str, Any]], bool] | None = None

    def holds(self, ctx: dict[str, Any]) -> bool:
        return True if self.predicate is None else bool(self.predicate(ctx))


class Symptom:
    NONFINITE = "非有限值"
    SHAPE = "形状不符"
    DTYPE = "dtype不符"
    PRECISION = "精度偏差"
    INDEX = "索引越界"
    SEMANTIC = "语义偏离"       # 输出与 oracle 不符
    PROPERTY = "性质违反"       # 数学性质谓词不过
    ALL = (NONFINITE, SHAPE, DTYPE, PRECISION, INDEX, SEMANTIC, PROPERTY)


@dataclass
class OpSpec:
    """算子规格：声明式。新增算子 = 提供一个实例，不碰引擎。

    semantic 是「意图」的可执行声明：
      - 绑定 SemanticSpec -> 验证会做 oracle 对拍 + 性质检验（语义可核验）；
      - semantic=None    -> 诚实降级，仅验证实现忠实度，报告明确不声称语义正确。
    doc 只是人读说明，不再承诺语义（文本无约束力，正是 v2 的病灶）。
    """

    key: str
    tree: Expr
    inputs: tuple[tuple[str, str, tuple[str, ...]], ...]
    output_dtype: str
    wei: list[ShiWei] = field(default_factory=list)
    known: set[str] = field(default_factory=set)
    doc: str = ""
    semantic: "SemanticSpec | None" = None

    @property
    def n_inputs(self) -> int:
        return len(self.inputs)

    @property
    def semantic_trusted(self) -> bool:
        return self.semantic is not None


@dataclass
class SemanticDiag:
    """语义诊断汇总：跨全部时位的 oracle/性质结果。"""

    checked: bool = False           # 是否绑定了语义基准
    failures: list[str] = field(default_factory=list)
    n_checked: int = 0

    @property
    def ok(self) -> bool:
        return not self.failures


@dataclass
class LayerResult:
    sixiang: SiXiang
    summary: str
    failure_rate: float = 0.0
    failures: int = 0
    total: int = 0
    details: list[str] = field(default_factory=list)
    symptoms: set[str] = field(default_factory=set)
    semantic: SemanticDiag = field(default_factory=SemanticDiag)
    warnings: list[str] = field(default_factory=list)  # 通过但需守：内建数值保护的代价


@dataclass
class CycleTrace:
    round: int
    sixiang: SiXiang
    summary: str
    symptoms: set[str] = field(default_factory=set)
    repaired: str | None = None
    reason: str = ""


@dataclass
class DevResult:
    key: str
    converged: bool
    sixiang: SiXiang
    source: str
    report: str
    traces: list[CycleTrace]
    output_dtype: str
    n_inputs: int
    semantic_trusted: bool = False
    semantic_failures: list[str] = field(default_factory=list)
    backend: str = "numpy"          # numpy | ascendc（v4 多后端）
    judged_by: str | None = None    # v7：外部评测集来源；None=未经裁判
    proto_ok: bool = False          # v7：是否已通过原型逐字校验

    @property
    def source_language(self) -> str:
        return "AscendC C++" if self.backend == "ascendc" else "NumPy/Python"

    @property
    def has_real_source(self) -> bool:
        bad = ("...", "pass  #", "TODO", "f(x)")
        return bool(self.source.strip()) and not any(b in self.source for b in bad)
