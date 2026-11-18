"""演示：选手与裁判分离如何根治语义蒙混漏洞。

运行：python3.11 -m examples.evalbench_demo
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from yidao import (
    YiDaoAgent, OpSpec, ShiWei, SemanticSpec, build_registry, ContractError,
)
from yidao.expr import mul, inp
from yidao.compile import make_kernel


def build_fake_relu():
    """恶意选手：声称 ReLU，树写成 x*x，并自带一个算 x*x 的假 oracle。"""
    return OpSpec(
        key="relu", tree=mul(inp(0), inp(0)),
        inputs=(("x", "float32", ("N", "M")),), output_dtype="float32",
        wei=[ShiWei(((3, 4),), ("float32",))],
        known={"shape一致", "dtype可表示"},
        semantic=SemanticSpec("假基准", lambda x: np.asarray(x) ** 2),
    )


def predict(r, x):
    return make_kernel(r.source)(x).ravel()


def main():
    x = np.array([[-2.0, 3.0]], dtype=np.float32)
    print("真 ReLU 应为 [0, 3]；恶意 kernel 实际算 x*x = [4, 9]\n")

    # ---- 场景一：没有外部裁判（= v3-v6 旧行为）----
    print("=" * 62)
    print("场景一 · 无外部评测集（v3-v6 旧机制）")
    print("=" * 62)
    r_old = YiDaoAgent().develop(build_fake_relu())
    print(f"  判定：{r_old.sixiang.value} / converged={r_old.converged}")
    print(f"  实际输出：{predict(r_old, x)}   ← 错误却被接受")
    print("  ⚠ 作者自带的假 oracle 充当裁判 —— 选手自己给自己打分。\n")

    # ---- 场景二：接入外部评测集（v7）----
    print("=" * 62)
    print("场景二 · 接入官方评测集（v7 选手与裁判分离）")
    print("=" * 62)
    r_new = YiDaoAgent(eval_registry=build_registry()).develop(build_fake_relu())
    print(f"  判定：{r_new.sixiang.value} / converged={r_new.converged}")
    print(f"  裁判来源：{r_new.judged_by}  原型校验：{r_new.proto_ok}")
    print(f"  语义失败 {len(r_new.semantic_failures)} 条，前 2 条：")
    for f in r_new.semantic_failures[:2]:
        print("    -", f[:78])
    print("  ✓ 假 oracle 被忽略，评测集 golden 当场揭穿 x*x ≠ ReLU。\n")

    # ---- 场景三：原型不一致 → 评测拒收 ----
    print("=" * 62)
    print("场景三 · 选手偷改 dtype，原型逐字校验拒收")
    print("=" * 62)
    bad = OpSpec(key="relu", tree=mul(inp(0), inp(0)),
                 inputs=(("x", "float64", ("N", "M")),), output_dtype="float32",
                 wei=[ShiWei(((3, 4),), ("float64",))])
    try:
        YiDaoAgent(eval_registry=build_registry()).develop(bad)
        print("  ✗ 未被拒收（异常）")
    except ContractError as e:
        print(f"  ✓ 评测拒收：{str(e)[:90]}")

    # ---- 场景四：正常算子不受误伤 ----
    print("\n" + "=" * 62)
    print("场景四 · 12 个正常算子在评测集下全部回归通过")
    print("=" * 62)
    from yidao.specs import build_specs
    reg = build_registry()
    agent = YiDaoAgent(eval_registry=reg)
    ok = sum(1 for k, s in build_specs().items() if agent.develop(s).converged)
    print(f"  {ok}/12 通过；每个算子均由评测集 golden 裁判，非选手自报。")


if __name__ == "__main__":
    main()
