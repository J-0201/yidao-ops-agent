"""可执行测试（无 pytest 依赖）——v3 独立语义基准版。

针对第三方评估对 v2 的每条残余指控给出回归测试：
  1)【根本】语义正确性闭环：fake_relu（doc/key 声称 ReLU、树为 x*x）被判未天成；
  2) oracle 路径独立：semantic.py 不 import expr/compile；
  3) oracle 自身是黄金：float64 累加，相消时位不错判合法算子；
  4)【中等】matmul 累加精度重写真正生效（不再空转）；
  5)【中等】吝(LIN) 是真实可达状态；
  6)【中等】索引越界由 oob 时位可靠触发；
  7) 无语义基准的算子诚实降级，不声称语义天成；
  8) 全部 12 算子收敛 + v2 已有能力（错误 kernel 判悔、筋骨结构推导、闭环）。
"""

from __future__ import annotations

import ast
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from yidao import JinGu, YiDaoAgent
from yidao.buyi import catalog
from yidao.compile import REWRITES, apply_rewrites, compile_source, interpret, make_kernel
from yidao.expr import div, exp, inp, mul, relu, sum_
from yidao.jianyi import JianYi
from yidao.model import OpSpec, ShiWei, SiXiang, Symptom
from yidao.semantic import SEMANTICS
from yidao.shenghudong import ShengHuDong, classify
from yidao.specs import build_specs


def test_all_specs_converge_with_semantics() -> None:
    """全部算子语义核验通过且收敛。"""
    agent = YiDaoAgent()
    for key, spec in build_specs().items():
        r = agent.develop(spec)
        assert r.converged, f"{key} 未收敛：{r.report}"
        assert r.semantic_trusted, f"{key} 未绑定语义基准"
        assert r.semantic_failures == [], f"{key} 语义未过：{r.semantic_failures[:1]}"
        assert r.has_real_source, f"{key} 源码是占位符"


def test_fake_relu_rejected_by_semantics() -> None:
    """【根本突破】fake_relu：doc/key 声称 ReLU，树写 x*x —— 必须判未天成。

    v2 里它判「吉·天成」（编译器与解释器同源，一致即放行）。
    v3 用独立 oracle + 数学性质揭穿：语义偏离 + 正半轴恒等违反 + 负半轴归零违反。
    """
    fake = OpSpec(
        key="relu", tree=mul(inp(0), inp(0)),
        inputs=(("x", "float32", ("N", "M")),), output_dtype="float32",
        wei=[ShiWei(((3, 4),), ("float32",))],
        known={"shape一致", "dtype可表示"},
        doc="整流线性单元 ReLU：max(x, 0)",  # 文本声称（无约束力）
        semantic=SEMANTICS["relu"],           # 绑定语义基准
    )
    r = YiDaoAgent().develop(fake)
    assert not r.converged, "fake_relu 竟被判可交付——语义闭环失效"
    assert r.sixiang == SiXiang.HUI
    joined = " ".join(r.semantic_failures)
    assert "语义偏离" in joined
    assert "性质违反" in joined and ("恒等" in joined or "归零" in joined)


def test_semantic_path_independent() -> None:
    """oracle 路径独立：semantic.py 不 import expr/compile（AST 精确核验）。"""
    import yidao.semantic as sm
    tree = ast.parse(open(sm.__file__).read())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.endswith("expr") or node.module.endswith("compile"):
                bad.append(node.module)
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.endswith("expr") or a.name.endswith("compile"):
                    bad.append(a.name)
    assert not bad, f"语义基准与表达式树存在代码依赖：{bad}"


def test_oracle_is_golden_on_cancel() -> None:
    """oracle 自身是黄金：相消时位用 float64 累加，不误判合法算子。"""
    sem = SEMANTICS["reduce_sum"]
    cancel = np.array([[1e8, 1, 1, 1, 1, -1e8]], dtype=np.float32)
    gold = sem.gold([cancel])
    # 正确和 = 4（中间四个 1）；float32 累加会丢成 0，float64 oracle 应得 4
    assert np.allclose(gold.astype(np.float64), [4.0], atol=1e-4), \
        f"oracle 相消时位丢了精度：{gold}（gold 必须是 float64 累加）"


def test_matmul_rewrite_not_noop() -> None:
    """【中等】matmul 累加精度重写真正生效：作用于 MatMul 节点。"""
    spec = build_specs()["matmul"]
    rw = REWRITES["累加精度(float64)"]
    src_plain = compile_source(spec.tree, [], 2, "float32")
    src_rw = compile_source(spec.tree, [rw], 2, "float32")
    assert src_plain != src_rw, "matmul 重写仍是空转（源码未变）"
    assert "float64" in src_rw, "matmul 提升 float64 未织进源码"
    fn = make_kernel(src_rw)
    a = np.random.default_rng(1).normal(size=(3, 4)).astype("float32")
    b = np.random.default_rng(2).normal(size=(4, 5)).astype("float32")
    assert np.allclose(fn(a, b), a.astype("f8") @ b.astype("f8"), atol=1e-4)


def test_lin_is_reachable() -> None:
    """吝(LIN) 判定纯函数：吉/吝/厉/悔 四分支。

    v3.1 关键：吝的独立语义是「通过但需守」——全绿且有数值保护警告 -> 吝。
    """
    # 全绿无警告 -> 吉
    assert classify(0, 20, set(), [])[0] == SiXiang.JI
    # 全绿但有数值保护警告 -> 吝（通过但需守）
    lin, summary = classify(0, 20, set(), ["内建数值保护·float64"])
    assert lin == SiXiang.LIN, f"通过带警告应判吝，实得 {lin}: {summary}"
    assert lin.yang, "吝属阳（可交付但带警告）"
    # 轻微失败(5%)也可归吝
    assert classify(1, 20, {Symptom.PRECISION})[0] == SiXiang.LIN
    # 严重症状即使失败率低 -> 悔
    assert classify(1, 20, {Symptom.NONFINITE})[0] == SiXiang.HUI
    assert classify(1, 20, {Symptom.PROPERTY})[0] == SiXiang.HUI
    # 高失败率 -> 悔
    assert classify(5, 20, {Symptom.PRECISION})[0] == SiXiang.HUI


def test_lin_appears_in_real_closed_loop() -> None:
    """【端到端】吝在真实 develop 闭环中首次可达（评估 v3 的残余指控）。

    评估实测：v3 的 12 算子 develop 轨迹只有吉/悔，吝从未真实出现。
    修复：恢复吝的「通过但需守」语义——内建数值保护重写即警告。
      - 内建数值保护的算子（softmax/reduce_sum/matmul）终判吝；
      - 无数值保护的算子（relu）终判吉；
      - 吝仍 converged=True（可交付），且语义基准守住（非空转警告）。
    """
    specs = build_specs()
    agent = YiDaoAgent()
    results = {k: agent.develop(sp) for k, sp in specs.items()}

    # 内建数值保护的算子 -> 吝（真实闭环中可达）
    for key in ("softmax", "reduce_sum", "reduce_mean", "matmul"):
        r = results[key]
        assert r.sixiang == SiXiang.LIN, f"{key} 应判吝，实得 {r.sixiang.value}"
        assert r.converged, f"{key} 吝应可交付"
        assert r.semantic_failures == [], f"{key} 吝不应掩盖语义失败"
        # 警告落在最后一轮的 LayerResult 证据里（通过报告可见）
        assert "数值保护" in r.report or "需守" in r.report, f"{key} 报告未体现警告"

    # 无数值保护的算子 -> 吉
    for key in ("relu", "add_broadcast", "gather", "sigmoid", "tanh"):
        assert results[key].sixiang == SiXiang.JI, f"{key} 应判吉，实得 {results[key].sixiang.value}"

    # 全局四象分布：吉/吝/悔 三种真实出现（厉为临界态，无需强求）
    dist = Counter(t.sixiang.value for r in results.values() for t in r.traces)
    assert dist["吉"] > 0 and dist["吝"] > 0 and dist["悔"] > 0, f"四象分布异常：{dist}"


def test_index_oob_triggered() -> None:
    """【中等】索引越界由专用 oob 时位可靠触发（症状集含索引越界）。"""
    spec = build_specs()["gather"]
    r = YiDaoAgent().develop(spec)
    all_syms = set()
    for t in r.traces:
        all_syms |= t.symptoms
    assert Symptom.INDEX in all_syms, f"oob 时位未触发索引越界症状：{all_syms}"


def test_no_semantic_honest_downgrade() -> None:
    """无语义基准的算子诚实降级：converged 但 semantic_trusted=False，报告不声称语义天成。"""
    fake = OpSpec(
        key="unverified", tree=mul(inp(0), inp(0)),
        inputs=(("x", "float32", ("N", "M")),), output_dtype="float32",
        wei=[ShiWei(((3, 4),), ("float32",))],
        known={"shape一致", "dtype可表示"}, semantic=None,
    )
    r = YiDaoAgent().develop(fake)
    assert r.converged, "无基准算子也应能编译收敛"
    assert not r.semantic_trusted
    assert "降级" in r.report and "不声称" in r.report
    assert "语义天成" not in r.report.replace("不声称树符合", "")  # 不出现肯定的语义天成


def test_wrong_kernel_still_rejected() -> None:
    """v2 致命漏洞修复在 v3 延续：恒返回 0 的错误 kernel 判悔。"""
    spec = build_specs()["relu"]
    active = {n: catalog()[n] for n in spec.known}
    def bad_kernel(x):
        return np.zeros_like(x)
    result = ShengHuDong().verify(spec, active, "bad", bad_kernel)
    assert not result.sixiang.yang, "错误 kernel 被判可交付"
    assert Symptom.PRECISION in result.symptoms
    assert result.failures > 0


def test_jijin_from_structure() -> None:
    """筋骨从树结构推导，doc 文本无法欺骗。"""
    specs = build_specs()
    jy = JianYi()
    assert jy.identify(specs["matmul"]) == JinGu.ZHEN
    assert jy.identify(specs["softmax"]) == JinGu.LI
    s = specs["matmul"]
    s.doc = "softmax softmax softmax"
    assert jy.identify(s) == JinGu.ZHEN


def test_softmax_closed_loop() -> None:
    """softmax 闭环：补数值稳定常，源码内建减最大值。"""
    r = YiDaoAgent().develop(build_specs()["softmax"])
    assert any(t.repaired == "数值稳定(减最大值)" for t in r.traces)
    assert "amax" in r.source
    fn = make_kernel(r.source)
    with np.errstate(all="ignore"):
        out = fn(np.array([[1000.0, 1001.0, 1002.0]], dtype="float32"))
    assert np.all(np.isfinite(out))


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n全部 {len(fns)} 项测试通过。")


if __name__ == "__main__":
    _run_all()
