"""通用中间表示：算子 = 一棵由组合子构成的表达式树（Expr）。

v1 的病根是「6 处硬编码 + 关键词路由」：每个算子写死一个 Python 函数、
一个参考实现、一套骨架。v2 把算子统一表达为组合子树的**数据**，于是：

  - 新增算子 = 用组合子拼一棵树（声明式规格），不碰引擎源码；
  - 编译器把树 -> NumPy 源码（变易层，真 exec）；
  - 解释器用另一条路径对同一棵树求值（独立外验，不复用编译器代码）；
  - 筋骨从树的**结构**推导（简易层），不再是 math_def 子串匹配；
  - 不变量 = 对树的**重写规则**（不易层），内建即改树，违规结构写不出。

组合子（combinator）全集：
  Input(i)              第 i 个输入张量
  Scalar(v)             标量常量
  Elem(op, *args)       逐元素运算：add/sub/mul/div/neg/exp/maximum/relu/tanh/sigmoid...
  Reduce(op, x, axis)   归约：sum/max/mean，沿 axis
  MatMul(a, b)          矩阵乘
  Gather(data, idx)     按整数索引收集
  Broadcast(x, shape)   显式广播到 shape
  StableSoftmax(x)      复合组合子：数值稳定 softmax（归约+逐元素，演示复合）

每个组合子是不可变的 frozen dataclass，树可被安全地重写/遍历。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ---- 组合子树 ------------------------------------------------------------ #

@dataclass(frozen=True)
class Expr:
    """所有组合子的基类。不可变，可重写。"""

    def children(self) -> tuple["Expr", ...]:
        return ()

    def map(self, f: Callable[["Expr"], "Expr"]) -> "Expr":
        """子树自底向上重写：先重写孩子，再由 f 处理本节点。"""
        return f(self.rebuild(tuple(c.map(f) for c in self.children())))

    def rebuild(self, children: tuple["Expr", ...]) -> "Expr":
        return self


@dataclass(frozen=True)
class Input(Expr):
    i: int = 0

    def label(self) -> str:
        return f"x{self.i}"


@dataclass(frozen=True)
class Scalar(Expr):
    v: float = 1.0


@dataclass(frozen=True)
class Elem(Expr):
    """逐元素运算。op ∈ {add,sub,mul,div,neg,exp,maximum,relu,tanh,sigmoid,...}"""

    op: str
    args: tuple[Expr, ...]

    def children(self) -> tuple[Expr, ...]:
        return self.args

    def rebuild(self, children: tuple[Expr, ...]) -> Expr:
        return Elem(self.op, children)


@dataclass(frozen=True)
class Reduce(Expr):
    """归约。op ∈ {sum, max, mean}。axis=None 表示全归约。

    keepdims=True 保留被归约轴（供 softmax 减最大值时广播用）。
    """

    op: str
    x: Expr
    axis: int | None = -1
    keepdims: bool = False

    def children(self) -> tuple[Expr, ...]:
        return (self.x,)

    def rebuild(self, children: tuple[Expr, ...]) -> Expr:
        return Reduce(self.op, children[0], self.axis, self.keepdims)


@dataclass(frozen=True)
class MatMul(Expr):
    a: Expr
    b: Expr

    def children(self) -> tuple[Expr, ...]:
        return (self.a, self.b)

    def rebuild(self, children: tuple[Expr, ...]) -> Expr:
        return MatMul(children[0], children[1])


@dataclass(frozen=True)
class Gather(Expr):
    data: Expr
    idx: Expr

    def children(self) -> tuple[Expr, ...]:
        return (self.data, self.idx)

    def rebuild(self, children: tuple[Expr, ...]) -> Expr:
        return Gather(children[0], children[1])


@dataclass(frozen=True)
class Broadcast(Expr):
    x: Expr
    shape: tuple[int | str, ...]

    def children(self) -> tuple[Expr, ...]:
        return (self.x,)

    def rebuild(self, children: tuple[Expr, ...]) -> Expr:
        return Broadcast(children[0], self.shape)


@dataclass(frozen=True)
class Where(Expr):
    """逐元素条件选择：cond ? a : b（用于分段稳定实现，如 sigmoid/tanh）。"""

    cond: Expr
    a: Expr
    b: Expr

    def children(self) -> tuple[Expr, ...]:
        return (self.cond, self.a, self.b)

    def rebuild(self, children: tuple[Expr, ...]) -> Expr:
        return Where(children[0], children[1], children[2])


# ---- 组合子构造器（声明式规格的「词汇表」）-------------------------------- #

def inp(i: int = 0) -> Input:
    return Input(i)


def scalar(v: float) -> Scalar:
    return Scalar(v)


def elem(op: str, *args: Expr) -> Elem:
    return Elem(op, tuple(args))


def add(a: Expr, b: Expr) -> Elem:
    return Elem("add", (a, b))


def sub(a: Expr, b: Expr) -> Elem:
    return Elem("sub", (a, b))


def mul(a: Expr, b: Expr) -> Elem:
    return Elem("mul", (a, b))


def div(a: Expr, b: Expr) -> Elem:
    return Elem("div", (a, b))


def neg(a: Expr) -> Elem:
    return Elem("neg", (a,))


def exp(a: Expr) -> Elem:
    return Elem("exp", (a,))


def maximum(a: Expr, b: Expr) -> Elem:
    return Elem("maximum", (a, b))


def relu(a: Expr) -> Elem:
    return Elem("relu", (a,))


def tanh(a: Expr) -> Elem:
    return Elem("tanh", (a,))


def sigmoid(a: Expr) -> Elem:
    return Elem("sigmoid", (a,))


def reduce(op: str, x: Expr, axis: int | None = -1, keepdims: bool = False) -> Reduce:
    return Reduce(op, x, axis, keepdims)


def sum_(x: Expr, axis: int | None = -1, keepdims: bool = False) -> Reduce:
    return Reduce("sum", x, axis, keepdims)


def max_(x: Expr, axis: int | None = -1, keepdims: bool = False) -> Reduce:
    return Reduce("max", x, axis, keepdims)


def mean_(x: Expr, axis: int | None = -1, keepdims: bool = False) -> Reduce:
    return Reduce("mean", x, axis, keepdims)


def matmul(a: Expr, b: Expr) -> MatMul:
    return MatMul(a, b)


def gather(data: Expr, idx: Expr) -> Gather:
    return Gather(data, idx)


def broadcast(x: Expr, shape: tuple[int | str, ...]) -> Broadcast:
    return Broadcast(x, shape)


def where(cond: Expr, a: Expr, b: Expr) -> Where:
    return Where(cond, a, b)


# ---- 树工具 -------------------------------------------------------------- #

def walk(e: Expr) -> list[Expr]:
    """前序遍历全部节点。"""
    out = [e]
    for c in e.children():
        out.extend(walk(c))
    return out


def count_ops(e: Expr) -> dict[str, int]:
    """统计各节点类型——筋骨推导与复杂度度量都基于此结构事实。"""
    d: dict[str, int] = {}
    for node in walk(e):
        k = type(node).__name__
        d[k] = d.get(k, 0) + 1
        if isinstance(node, Elem):
            d["Elem:" + node.op] = d.get("Elem:" + node.op, 0) + 1
        if isinstance(node, Reduce):
            d["Reduce:" + node.op] = d.get("Reduce:" + node.op, 0) + 1
    return d


def depth(e: Expr) -> int:
    cs = e.children()
    return 1 + max((depth(c) for c in cs), default=0)
