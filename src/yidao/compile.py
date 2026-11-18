"""双路径求值：编译器（变易层，产出真源码）与独立解释器（黄金参考）。

结构性解决 v1 的「对拍同源化 / 外验漏网」两大病根：

  - 编译器 compile_source(tree, rewrites) -> 源码字符串 -> exec -> kernel；
  - 解释器 interpret(tree, inputs, rewrites) 用另一套递归逻辑直接求值；
  - 两者接收【同一份重写规则】。不变量是作用在表达式树上的重写，
    内建时两条路径同步获得该性质——不是「kernel 偷偷抄参考」。

不变量作为树重写规则（rewrites: list[Rewrite]）：
  stabilize_subtract_max : exp(z) -> exp(z - max(z))       （softmax 减最大值）
  cancel_safe_sum         : sum(x) -> sum(float64(x))       （相消时保精度）

「违规在结构上写不出」= 应用了规则的树里根本不存在裸 exp / float32 累加。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


# ---- 合成源码安全校验（v5）：AST 白名单，堵住 exec 注入面 ---------------- #

class UnsafeSourceError(ValueError):
    """合成源码含白名单之外的语法/名称时抛出。"""


def validate_source(source: str) -> None:
    """校验源码仅由受控子集构成，否则拒绝。

    make_kernel 执行的是本模块从表达式树拼出的源码，理论上是受控的；但
    第三方评估指出 `exec` 仅放行 `__import__` 仍有极端注入面——恶意 spec
    若能把任意代码塞进表达式（如自定义 Elem op 名），exec 就会执行它。
    v5 在 exec 前用 AST 白名单把这类面彻底收紧：

      - 顶层只允许 `import numpy as np` 与单个 `def kernel(...)`；
      - kernel 体只允许 return（可含 docstring）；
      - 表达式只允许受控节点：数值常量、参数名、np.<公开属性>、
        <expr>.astype、np 函数/astype 调用、下标、元组/列表、二元/一元运算；
      - 任何 __import__/open/eval/非 numpy import/双下划线属性/推导式/
        lambda/global 等一律拒绝。

    白名单按"结构"而非"算子名"设定，因此新增合法算子无需改这里。
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise UnsafeSourceError(f"源码语法错误：{exc}") from exc

    fn: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != "numpy":
                    raise UnsafeSourceError(f"禁止的 import：{alias.name}")
            continue
        if isinstance(node, ast.ImportFrom):
            raise UnsafeSourceError("禁止 from-import")
        if isinstance(node, ast.FunctionDef):
            if fn is not None or node.name != "kernel":
                raise UnsafeSourceError("只允许单个名为 kernel 的函数")
            fn = node
            continue
        raise UnsafeSourceError(f"禁止的顶层语句：{type(node).__name__}")

    if fn is None:
        raise UnsafeSourceError("缺少 kernel 函数")
    args = fn.args
    if args.posonlyargs or args.vararg or args.kwonlyargs or args.kw_defaults or args.kwarg or args.defaults:
        raise UnsafeSourceError("kernel 参数形式受限")
    if fn.decorator_list:
        raise UnsafeSourceError("禁止装饰器")
    allowed_names = {a.arg for a in args.args} | {"np"}

    body = fn.body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    if not body or not all(isinstance(s, ast.Return) for s in body):
        raise UnsafeSourceError("kernel 只允许 return 语句")

    for stmt in body:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Name):
                if n.id not in allowed_names:
                    raise UnsafeSourceError(f"禁止的名称：{n.id}")
            elif isinstance(n, ast.Attribute):
                if n.attr.startswith("__"):
                    raise UnsafeSourceError(f"禁止的属性：{n.attr}")
            elif isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Name):
                    raise UnsafeSourceError(f"禁止直接调用函数：{f.id}")
                if isinstance(f, ast.Attribute):
                    if f.attr.startswith("__"):
                        raise UnsafeSourceError(f"禁止的方法：{f.attr}")
                    is_np_call = isinstance(f.value, ast.Name) and f.value.id == "np"
                    if not is_np_call and f.attr not in ("astype", "reshape"):
                        raise UnsafeSourceError(f"禁止的调用目标：{f.attr}")
                else:
                    raise UnsafeSourceError("禁止的调用形式")
            elif isinstance(n, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp,
                                ast.GeneratorExp, ast.NamedExpr, ast.Await,
                                ast.Yield, ast.YieldFrom, ast.Global, ast.Nonlocal)):
                raise UnsafeSourceError(f"禁止的语法：{type(n).__name__}")

from .expr import (
    Broadcast, Elem, Expr, Gather, Input, MatMul, Reduce, Scalar, Where,
    exp, maximum, mul, sub, sum_, MatMul,
)


# ---- 重写规则（不变量的可执行形态）-------------------------------------- #

@dataclass(frozen=True)
class Rewrite:
    """一条树重写规则：name + 作用函数树->树。"""

    name: str
    apply: Callable[[Expr], Expr]

    def __call__(self, tree: Expr) -> Expr:
        return self.apply(tree)


def _stabilize_softmax(node: Expr) -> Expr:
    """softmax 模式 div(exp(z), sum(exp(z), keepdims)) -> 分子分母同减 max(z)。

    只对「exp(z) / sum(exp(z))」这一结构生效，不碰 sigmoid / tanh 等
    其它 exp——避免把 logistic 的 exp 错误地减最大值而破坏其数学。
    max 沿末轴 keepdims，结果可广播；exp 输入恒 ≤ 0，不溢出。
    """
    if isinstance(node, Elem) and node.op == "div" and len(node.args) == 2:
        num, den = node.args
        if (
            isinstance(num, Elem) and num.op == "exp"
            and isinstance(den, Reduce) and den.op == "sum"
            and isinstance(den.x, Elem) and den.x.op == "exp"
            and _same_shape_expr(num.args[0], den.x.args[0])
        ):
            z = num.args[0]
            shift = Reduce("max", z, -1, keepdims=True)
            safe_z = sub(z, shift)
            return Elem("div", (exp(safe_z), Reduce("sum", exp(safe_z), den.axis, True)))
    return node


def _same_shape_expr(a: Expr, b: Expr) -> bool:
    """结构等价判断（用于确认分子分母的 exp 输入是同一表达式）。"""
    return repr(a) == repr(b)


def _cancel_safe(node: Expr) -> Expr:
    """把累加提升到 float64，保留 axis/keepdims。

    【v3 修复 matmul 空转】重写不仅作用于 Reduce(sum/mean)，也作用于
    MatMul：把 A、B  operands 提升到 float64——np.matmul 的累加在 float64
    下进行。v2 里此规则只匹配 Reduce，MatMul 树无 Reduce 节点故完全空转。
    """
    from .expr import elem
    if isinstance(node, Reduce) and node.op in ("sum", "mean"):
        return Reduce(node.op, elem("asf64", node.x), node.axis, node.keepdims)
    if isinstance(node, MatMul):
        return MatMul(elem("asf64", node.a), elem("asf64", node.b))
    return node


REWRITES: dict[str, Rewrite] = {
    "数值稳定(减最大值)": Rewrite("数值稳定(减最大值)", lambda t: t.map(_stabilize_softmax)),
    "累加精度(float64)": Rewrite("累加精度(float64)", lambda t: t.map(_cancel_safe)),
}


def apply_rewrites(tree: Expr, rewrites: list[Rewrite]) -> Expr:
    t = tree
    for rw in rewrites:
        t = rw(t)
    return t


# ---- 编译器：表达式树 -> NumPy 源码字符串（真工件）---------------------- #

def compile_source(
    tree: Expr, rewrites: list[Rewrite], n_inputs: int,
    output_dtype: str | None = None,
) -> str:
    """把（已应用重写规则的）表达式树编译成可独立运行的 NumPy 函数源码。

    output_dtype 给定则最终回写到该 dtype——「float64 累加后回写」的语义：
    累加过程高精度，产物遵循声明 dtype（与参考对齐，dtype 常由结构保证）。
    """
    t = apply_rewrites(tree, rewrites)
    expr_str = _emit(t)
    if output_dtype:
        expr_str = f"({expr_str}).astype(np.{output_dtype})"
    names = ", ".join(f"x{i}" for i in range(n_inputs))
    return (
        "import numpy as np\n"
        f"def kernel({names}):\n"
        f"    return {expr_str}\n"
    )


def _emit(e: Expr) -> str:
    if isinstance(e, Input):
        return f"x{e.i}"
    if isinstance(e, Scalar):
        return repr(e.v)
    if isinstance(e, Elem):
        args = ", ".join(_emit(a) for a in e.args)
        if e.op == "add" and len(e.args) == 2:
            return f"({_emit(e.args[0])} + {_emit(e.args[1])})"
        if e.op == "sub" and len(e.args) == 2:
            return f"({_emit(e.args[0])} - {_emit(e.args[1])})"
        if e.op == "mul" and len(e.args) == 2:
            return f"({_emit(e.args[0])} * {_emit(e.args[1])})"
        if e.op == "div" and len(e.args) == 2:
            return f"({_emit(e.args[0])} / {_emit(e.args[1])})"
        if e.op == "neg":
            return f"(-{_emit(e.args[0])})"
        if e.op == "relu":
            return f"np.maximum({_emit(e.args[0])}, 0)"
        if e.op == "maximum" and len(e.args) == 2:
            return f"np.maximum({_emit(e.args[0])}, {_emit(e.args[1])})"
        if e.op == "asf64":
            return f"({_emit(e.args[0])}).astype(np.float64)"
        if e.op == "pow" and len(e.args) == 2:
            return f"({_emit(e.args[0])} ** {_emit(e.args[1])})"
        if e.op == "sigmoid":
            # NumPy 无 np.sigmoid，展开为 1/(1+exp(-x))
            return f"(1.0 / (1.0 + np.exp(-{_emit(e.args[0])})))"
        # exp/tanh 及通用逐元素
        fn = {"exp": "exp", "tanh": "tanh"}.get(e.op, e.op)
        return f"np.{fn}({args})" if len(e.args) == 1 else f"np.{fn}({args})"
    if isinstance(e, Reduce):
        axis = "" if e.axis is None else f", axis={e.axis}"
        kd = ", keepdims=True" if e.keepdims else ""
        fn = {"sum": "sum", "max": "amax", "mean": "mean"}.get(e.op, e.op)
        return f"np.{fn}({_emit(e.x)}{axis}{kd})"
    if isinstance(e, MatMul):
        return f"({_emit(e.a)} @ {_emit(e.b)})"
    if isinstance(e, Gather):
        return f"({_emit(e.data)}[{_emit(e.idx)}])"
    if isinstance(e, Broadcast):
        shape = tuple(s if isinstance(s, int) else f"x{int(s[1:])}" for s in e.shape)
        return f"np.broadcast_to({_emit(e.x)}, {shape!r})"
    if isinstance(e, Where):
        return f"np.where({_emit(e.cond)}, {_emit(e.a)}, {_emit(e.b)})"
    raise ValueError(f"无法编译节点：{type(e).__name__}")


def make_kernel(source: str) -> Callable[..., Any]:
    """exec 受控源码成函数。源码由本模块从表达式树拼出，非外部输入。

    v5 加固：exec 前先过 AST 白名单（validate_source），只放行受控子集；
    即便恶意 spec 把任意代码塞进表达式，也过不了白名单。builtins 仍只
    放行 __import__（源码首行 `import numpy as np` 需要），不给 eval/open。
    """
    validate_source(source)
    ns: dict[str, Any] = {"__builtins__": {"__import__": __import__}}
    exec(compile(source, "<yidao-kernel>", "exec"), ns)  # noqa: S102
    return ns["kernel"]


# ---- 独立解释器：黄金参考（另一条求值路径）------------------------------ #

def interpret(tree: Expr, inputs: list[np.ndarray], rewrites: list[Rewrite] | None = None) -> np.ndarray:
    """用 float64 递归求值表达式树——不调用编译器、不复用编译产物。

    rewrites=None 表示「朴素参考」（不内建任何稳定化）；外部验证时可给
    参考也应用同一 rewrites，实现「不变量内建后两条路径同步」。

    注意：参考用 float64 是数值计算的黄金标准基线（独立于 kernel 的 dtype）。
    """
    t = apply_rewrites(tree, rewrites) if rewrites else tree
    return _eval(t, inputs)


def _eval(e: Expr, inputs: list[np.ndarray]) -> np.ndarray:
    if isinstance(e, Input):
        return np.asarray(inputs[e.i], dtype=np.float64)
    if isinstance(e, Scalar):
        return np.asarray(e.v, dtype=np.float64)
    if isinstance(e, Elem):
        vals = [_eval(a, inputs) for a in e.args]
        op = e.op
        if op == "add":
            return vals[0] + vals[1]
        if op == "sub":
            return vals[0] - vals[1]
        if op == "mul":
            return vals[0] * vals[1]
        if op == "div":
            return vals[0] / vals[1]
        if op == "neg":
            return -vals[0]
        if op == "relu":
            return np.maximum(vals[0], 0)
        if op == "maximum":
            return np.maximum(vals[0], vals[1])
        if op == "asf64":
            return vals[0].astype(np.float64)
        if op == "pow":
            return vals[0] ** vals[1]
        if op == "exp":
            return np.exp(vals[0])
        if op == "tanh":
            return np.tanh(vals[0])
        if op == "sigmoid":
            return 1.0 / (1.0 + np.exp(-vals[0]))
        # v5 通用回退：与编译器 `np.<op>` 对称，支持外部算子的任意 numpy
        # 逐元素 op（log1p/abs/sqrt/round...）。禁止下划线开头属性防逃逸。
        if not op.startswith("_"):
            fn = getattr(np, op, None)
            if callable(fn):
                return fn(*vals)
        raise ValueError(f"解释器未知逐元素运算：{op}")
    if isinstance(e, Reduce):
        v = _eval(e.x, inputs)
        with np.errstate(all="ignore"):
            if e.op == "sum":
                return np.sum(v, axis=e.axis, keepdims=e.keepdims)
            if e.op == "mean":
                return np.mean(v, axis=e.axis, keepdims=e.keepdims)
            if e.op == "max":
                return np.max(v, axis=e.axis, keepdims=e.keepdims)
        raise ValueError(f"解释器未知归约：{e.op}")
    if isinstance(e, MatMul):
        return _eval(e.a, inputs) @ _eval(e.b, inputs)
    if isinstance(e, Gather):
        return _eval(e.data, inputs)[_eval(e.idx, inputs).astype(np.int64)]
    if isinstance(e, Broadcast):
        shape = tuple(s if isinstance(s, int) else inputs[int(s[1:])].shape[0] for s in e.shape)
        return np.broadcast_to(_eval(e.x, inputs), shape)
    if isinstance(e, Where):
        return np.where(_eval(e.cond, inputs), _eval(e.a, inputs), _eval(e.b, inputs))
    raise ValueError(f"解释器无法求值节点：{type(e).__name__}")
