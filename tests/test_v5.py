"""v5 增量测试：属性随机覆盖 + 语义构造器 + 安全 exec。"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from yidao import YiDaoAgent
from yidao.specs import build_specs
from yidao import coverage as cov
from yidao import (make_op, make_semantic, nonnegative, between,
                    row_sums_to_one, zero_mean, unit_variance)
from yidao.expr import relu, inp


PASS = []

def check(name, cond):
    assert cond, f"FAIL: {name}"
    PASS.append(name)
    print("  PASS", name)


# ---- 属性随机覆盖 ----
def test_coverage_all_specs():
    specs = build_specs()
    reports = cov.coverage_all(specs, seeds=(1, 7, 20261117), n_shapes=3)
    bad = {k: r for k, r in reports.items() if not r.ok}
    assert not bad, "覆盖未过：" + str({k: (r.passed, r.total, [c.reason for c in r.cases if not c.passed][:2]) for k, r in bad.items()})
    total = sum(r.total for r in reports.values())
    assert total >= 12 * 3 * 3 * len(cov.DISTRIBUTIONS), total
    check(f"覆盖层：12 算子 × {total // 12} 随机用例全过", True)


def test_coverage_uses_multiple_shapes_and_seeds():
    r = cov.coverage_check(build_specs()["relu"], seeds=(1, 2, 3), n_shapes=4)
    seeds = set(c.seed for c in r.cases)
    shapes = set(c.shapes for c in r.cases)
    check(f"覆盖层确实用多 seed({len(seeds)}) 多 shape({len(shapes)})",
          len(seeds) == 3 and len(shapes) >= 4)


# ---- 语义构造器 / 外部算子一站式注册 ----
def test_public_property_builders():
    out = np.array([-1.0, 0.0, 2.0])
    ok, _ = nonnegative().check(out, [out])
    assert not ok, "nonnegative 应抓到负值"
    ok, _ = between(0, 1).check(np.array([0.5, 1.5]), [None])
    assert not ok, "between 应抓到越界"
    ok, _ = row_sums_to_one().check(np.array([[0.7, 0.2]]), [None])
    assert not ok, "row_sums_to_one 应抓到未归一"
    check("公共性质构造器行为正确", True)


def test_make_op_and_register_external():
    # 外部新算子：softplus(x)=log(1+exp(x))，用公共 API 一站式造带 oracle + 性质的规格
    from yidao.expr import elem
    softplus_tree = elem("log1p", elem("exp", inp(0)))  # 语义=log(1+exp(x))
    spec = make_op(
        "softplus_ext", softplus_tree,
        (("x", "float32", ("N", "M")),), "float32",
        oracle=lambda x: np.logaddexp(0.0, np.asarray(x, dtype=np.float64)),
        properties=(nonnegative(),),
        doc="外部注册算子：softplus",
    )
    assert spec.semantic is not None and spec.semantic.name == "softplus_ext"
    check("make_op 造出带独立基准的外部算子", True)

    agent = YiDaoAgent()
    r = agent.develop(spec)
    assert r.converged and r.sixiang.value in ("吉", "吝"), r.report
    check(f"外部算子 softplus 收敛（{r.sixiang.value}，语义可信={r.semantic_trusted}）", True)


def test_external_op_no_oracle_honest_downgrade():
    from yidao.expr import tanh
    agent = YiDaoAgent()
    agent.register_op("tanh_ext", tanh(inp(0)), (("x", "float32", ("N", "M")),), "float32")
    r = agent.develop("tanh_ext")
    assert not r.semantic_trusted, "无 oracle 应降级"
    check("无 oracle 外部算子诚实降级（语义未核验）", True)


# ---- 安全 exec ----
def test_safe_exec_rejects_malicious_source():
    from yidao.compile import validate_source, UnsafeSourceError
    validate_source("import numpy as np\ndef kernel(x0):\n    return np.maximum(x0, 0)\n")
    check("合法合成源码通过 AST 白名单", True)
    for bad in (
        "import os\ndef kernel(x0):\n    return os.system('ls')\n",
        "def kernel(x0):\n    return __import__('os').system('ls')\n",
        "def kernel(x0):\n    return open('/etc/passwd').read()\n",
    ):
        try:
            validate_source(bad)
        except UnsafeSourceError:
            continue
        raise AssertionError(f"恶意源码未被拦截：{bad[:40]}")
    check("恶意源码（import os / __import__ / open）被拦截", True)


if __name__ == "__main__":
    print("=== v5 增量测试 ===")
    test_coverage_all_specs()
    test_coverage_uses_multiple_shapes_and_seeds()
    test_public_property_builders()
    test_make_op_and_register_external()
    test_external_op_no_oracle_honest_downgrade()
    test_safe_exec_rejects_malicious_source()
    print(f"\n全部 {len(PASS)} 项测试通过。")
