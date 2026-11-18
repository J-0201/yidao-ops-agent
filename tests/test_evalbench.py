"""外部评测集裁判层的对抗测试 + 回归测试。

核心命题：把「裁判权」从选手手里夺走后，v3-v6 的语义蒙混漏洞被根治。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from yidao import YiDaoAgent, OpSpec, ShiWei, EvalRegistry, EvalContract, \
    EvalCase, EvalProto, build_registry, ContractError
from yidao.expr import mul, inp
from yidao.specs import build_specs


def test_v6_attack_succeeds_without_judge():
    """复现 v3-v6 漏洞：无外部裁判时，假 oracle 让 fake_relu 蒙混过关。"""
    fake = OpSpec(
        key="relu", tree=mul(inp(0), inp(0)),  # 树算 x*x，根本不是 ReLU
        inputs=(("x", "float32", ("N", "M")),), output_dtype="float32",
        wei=[ShiWei(((3, 4),), ("float32",))],
        known={"shape一致", "dtype可表示"},
        semantic=__import__("yidao.semantic", fromlist=["SemanticSpec"]).SemanticSpec(
            "fake", lambda x: np.asarray(x) ** 2),  # 选手自带假 oracle：算 x*x
    )
    r = YiDaoAgent().develop(fake)  # 无任何评测集（v6 行为）
    x = np.array([[-2.0, 3.0]], dtype=np.float32)
    out = __import__("yidao.compile", fromlist=["make_kernel"]).make_kernel(r.source)(x)
    # 漏洞成立：x*x = [4,9] 被判天成，而真 ReLU 应是 [0,3]
    assert r.converged, "复现失败：无裁判时本应蒙混过关"
    assert np.allclose(out.ravel(), [4.0, 9.0]), out
    assert r.judged_by is None, "无裁判却被标记裁判"
    print(f"  [漏洞复现] v6 行为：fake_relu 输出 {out.ravel()} 判{r.sixiang.value}（应[0,3]）")


def test_v7_attack_blocked_by_judge():
    """v7：同一个 fake_relu，接入官方评测集后必须被当场揭穿、判未天成。"""
    fake = OpSpec(
        key="relu", tree=mul(inp(0), inp(0)),  # x*x
        inputs=(("x", "float32", ("N", "M")),), output_dtype="float32",
        wei=[ShiWei(((3, 4),), ("float32",))],  # 选手只挑了最简单的时位
        known={"shape一致", "dtype可表示"},
        semantic=__import__("yidao.semantic", fromlist=["SemanticSpec"]).SemanticSpec(
            "fake", lambda x: np.asarray(x) ** 2),  # 仍塞假 oracle
    )
    agent = YiDaoAgent(eval_registry=build_registry())  # 裁判方注入
    r = agent.develop(fake)
    assert r.judged_by == "official-bench-v1", r.judged_by
    assert r.proto_ok is True
    assert not r.converged, f"漏洞未堵：fake_relu 竟被判{r.sixiang.value}"
    assert r.sixiang.value == "悔", r.sixiang.value
    assert any("语义" in f or "归零" in f or "恒等" in f for f in r.semantic_failures), \
        r.semantic_failures
    assert "已忽略" in r.report and "外部评测集" in r.report, "报告未体现裁判分离"
    print(f"  [漏洞已堵] v7 行为：fake_relu 判{r.sixiang.value}，语义失败 {len(r.semantic_failures)} 条，假 oracle 已忽略")


def test_proto_mismatch_rejected():
    """选手规格与评测原型不一致（如偷偷改 dtype/shape/输入名）→ 评测拒收。"""
    bad = OpSpec(
        key="relu", tree=mul(inp(0), inp(0)),
        inputs=(("x", "float64", ("N", "M")),),  # dtype 与原型 float32 不符
        output_dtype="float32", wei=[ShiWei(((3, 4),), ("float64",))],
    )
    agent = YiDaoAgent(eval_registry=build_registry())
    try:
        agent.develop(bad)
        assert False, "proto 不符却未被拒收"
    except ContractError as e:
        assert "dtype" in str(e) and "原型" in str(e), str(e)
        print(f"  [原型校验] 拒收：{str(e)[:60]}")


def test_cases_are_judge_controlled():
    """选手挑选的 wei 被裁判 cases 覆盖：报告中的用例数应等于裁判用例数。"""
    spec = OpSpec(
        key="relu", tree=__import__("yidao.expr", fromlist=["relu"]).relu(inp(0)),
        inputs=(("x", "float32", ("N", "M")),), output_dtype="float32",
        wei=[ShiWei(((99, 99),), ("float32",))],  # 选手只给 1 个超大简单时位
        known={"shape一致", "dtype可表示"},
    )
    contract = build_registry().get("relu")
    judged, meta = contract.judge_spec(spec)
    assert len(judged.wei) == len(contract.cases) == 3, meta
    assert judged.wei[0].shapes == ((4, 3),), judged.wei[0]
    assert meta["cases"] == 3
    print(f"  [用例归属] 选手 1 个时位 → 裁判覆盖为 {meta['cases']} 个")


def test_official_specs_pass_under_judge():
    """回归：12 个内置算子接入评测集后仍全部通过（无误伤）。"""
    reg = build_registry()
    specs = build_specs()
    agent = YiDaoAgent(eval_registry=reg)
    passed = []
    for key, spec in specs.items():
        r = agent.develop(spec)
        assert r.judged_by == "official-bench-v1", key
        assert r.proto_ok, key
        passed.append((key, r.sixiang.value, r.converged))
    assert len(passed) == 12, passed
    assert all(c for _, _, c in passed), passed
    print(f"  [回归] {len(passed)}/12 算子在评测集下全部通过：{dict((k, s) for k, s, _ in passed)}")


def test_no_contract_honest_downgrade():
    """无评测契约的外部算子：回退 v3 路径，但报告明确标注未经裁判。"""
    from yidao.public_api import make_op
    from yidao.expr import add
    spec = make_op("ext_add", add(inp(0), inp(1)),
                   (("a", "float32", ("N", "M")), ("b", "float32", ("N", "M"))),
                   "float32", oracle=lambda a, b: a + b,
                   known={"shape一致", "dtype可表示"})
    r = YiDaoAgent().develop(spec)  # 无裁判注册表
    assert r.judged_by is None
    assert "缺位" in r.report and "未经独立裁判" in r.report
    print("  [降级] 无评测集时报告明确标注「未经独立裁判」")


if __name__ == "__main__":
    tests = [
        test_v6_attack_succeeds_without_judge,
        test_v7_attack_blocked_by_judge,
        test_proto_mismatch_rejected,
        test_cases_are_judge_controlled,
        test_official_specs_pass_under_judge,
        test_no_contract_honest_downgrade,
    ]
    for t in tests:
        print(f"\n▶ {t.__name__}")
        t()
    print("\n✅ 全部对抗 + 回归测试通过")
