"""可执行测试（无 pytest 依赖）——v4 多后端 + AscendC 真实后端版。

覆盖：
  1) AscendC 后端源码是真 AscendC 形态（GM_ADDR/GlobalTensor/TPipe/TQue/三段式）；
  2) kernel 被真正 g++/桩编译、加载、执行（非字符串检查）；
  3) 不变量重写真织进 C++ 源码并生效（相消时位 float32=0 / AscendC=4）；
  4) 12 算子在 AscendC 后端全部收敛且语义通过；
  5) 两后端语义等价（NumPy vs AscendC 输出对拍 oracle 一致）；
  6) 索引越界真被 kernel 拒绝；
  7) v3 全部能力回归（fake_relu 揭穿、吝可达、oracle 独立、错误 kernel 判悔等）。
"""

from __future__ import annotations

import ast
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from yidao import JinGu, YiDaoAgent
from yidao.ascendc import compile_ascendc_source, infer_shape
from yidao.ascendc_runtime import AscendCRuntime
from yidao.buyi import catalog
from yidao.compile import REWRITES
from yidao.expr import inp, mul, relu, sum_
from yidao.jianyi import JianYi
from yidao.model import OpSpec, ShiWei, SiXiang, Symptom
from yidao.semantic import SEMANTICS
from yidao.shenghudong import ShengHuDong
from yidao.specs import build_specs


def test_ascendc_source_is_real_kernel() -> None:
    """源码是真 AscendC 形态，不是占位/字符串把戏。"""
    ak = compile_ascendc_source("relu", relu(inp(0)), [], 1, ("float32",), "float32", {0: (4, 3)})
    s = ak.source
    for token in ("GM_ADDR", "__global__", "__aicore__", "GlobalTensor",
                  "LocalTensor", "TPipe", "TQue", "DataCopy", "SetGlobalBuffer",
                  "CopyIn", "Compute", "CopyOut", "yidao_kernel_launch"):
        assert token in s, f"AscendC 源码缺少关键要素：{token}"
    assert "..." not in s and "TODO" not in s


def test_ascendc_kernel_really_compiles_and_runs() -> None:
    """kernel 被真正编译、加载、执行；输出与 oracle 对拍。"""
    ak = compile_ascendc_source("relu", relu(inp(0)), [], 1, ("float32",), "float32", {0: (4, 3)})
    rt = AscendCRuntime()
    fn = rt.build(ak, lambda shp: shp[0])
    x = np.random.default_rng(7).normal(size=(4, 3)).astype("float32")
    out = fn(x)
    assert out.shape == x.shape
    assert np.allclose(out, np.maximum(x, 0), atol=1e-6), "AscendC kernel 数值不对"


def test_rewrites_woven_into_cpp_and_effective() -> None:
    """不变量重写真织进 C++，且相消时位真生效。"""
    tree = sum_(inp(0), -1)
    plain = compile_ascendc_source("rs", tree, [], 1, ("float32",), "float32", {0: (1, 6)})
    rw = compile_ascendc_source("rs", tree, [REWRITES["累加精度(float64)"]], 1, ("float32",), "float32", {0: (1, 6)})
    assert "float acc" in plain.source and "double acc" in rw.source, "float64 重写未织进 C++"
    fn = AscendCRuntime().build(rw, lambda s: (s[0][0],))
    x = np.array([[1e8, 1, 1, 1, 1, -1e8]], dtype="float32")
    assert np.allclose(fn(x).astype("f8"), [4.0], atol=1e-4), "AscendC float64 累加未生效"


def test_all_specs_converge_on_ascendc() -> None:
    """12 算子在 AscendC 后端全部收敛、语义通过、源码为真 C++。"""
    agent = YiDaoAgent(backend="ascendc")
    for key, spec in build_specs().items():
        r = agent.develop(spec)
        assert r.converged, f"{key} 未收敛：{r.report}"
        assert r.backend == "ascendc" and r.source_language == "AscendC C++"
        assert r.semantic_failures == [], f"{key} 语义未过：{r.semantic_failures[:1]}"
        assert "GM_ADDR" in r.source, f"{key} 源码不是 AscendC"
        assert r.has_real_source, f"{key} 源码是占位符"


def test_two_backends_semantically_equivalent() -> None:
    """NumPy 与 AscendC 两后端对同一算子输出一致（各自对拍 oracle 已过）。"""
    specs = build_specs()
    na, aa = YiDaoAgent(backend="numpy"), YiDaoAgent(backend="ascendc")
    for key in ("relu", "sigmoid", "reduce_sum", "matmul", "softmax", "layernorm"):
        rn, ra = na.develop(specs[key]), aa.develop(specs[key])
        assert rn.sixiang == ra.sixiang, f"{key} 两后端四象不一致：{rn.sixiang.value} vs {ra.sixiang.value}"
        assert rn.semantic_failures == [] and ra.semantic_failures == []


def test_oob_rejected_by_ascendc_kernel() -> None:
    """索引越界真被 kernel 检测到并拒绝。"""
    from yidao.expr import gather
    ak = compile_ascendc_source("gather", gather(inp(0), inp(1)), [], 2,
                                ("float32", "int64"), "float32", {0: (4, 3), 1: (2,)})
    fn = AscendCRuntime().build(ak, lambda s: (s[1][0], s[0][1]))
    data = np.arange(12, dtype="float32").reshape(4, 3)
    try:
        fn(data, np.array([1, 99], dtype="int64"))
        raise AssertionError("AscendC kernel 未拒绝越界索引")
    except IndexError:
        pass


# ---- v3 能力回归（确保多后端改造没破坏既有能力）-------------------------- #

def test_fake_relu_rejected_both_backends() -> None:
    """fake_relu 在两个后端都被语义基准揭穿。"""
    fake = OpSpec(
        key="relu", tree=mul(inp(0), inp(0)),
        inputs=(("x", "float32", ("N", "M")),), output_dtype="float32",
        wei=[ShiWei(((3, 4),), ("float32",))],
        known={"shape一致", "dtype可表示"},
        doc="整流线性单元 ReLU：max(x, 0)", semantic=SEMANTICS["relu"],
    )
    for backend in ("numpy", "ascendc"):
        r = YiDaoAgent(backend=backend).develop(fake)
        assert not r.converged and r.sixiang == SiXiang.HUI, f"{backend} 未揭穿 fake_relu"
        joined = " ".join(r.semantic_failures)
        assert "语义偏离" in joined and "性质违反" in joined


def test_semantic_path_independent() -> None:
    import yidao.semantic as sm
    tree = ast.parse(open(sm.__file__).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.endswith(("expr", "compile")), "语义基准与树存在代码依赖"


def test_lin_reachable_ascendc() -> None:
    agent = YiDaoAgent(backend="ascendc")
    results = {k: agent.develop(sp) for k, sp in build_specs().items()}
    for key in ("softmax", "reduce_sum", "matmul"):
        assert results[key].sixiang == SiXiang.LIN, f"{key} 应判吝"
    dist = Counter(r.sixiang.value for r in results.values())
    assert dist["吉"] > 0 and dist["吝"] > 0


def test_device_kernel_generator_elementwise() -> None:
    """设备侧 kernel 生成器：产出符合 AI Core 范式的真 AscendC 源码。"""
    from yidao.expr import add, inp, relu
    from yidao.ascendc_device import compile_device_kernel
    rk = compile_device_kernel("relu_custom", relu(inp(0)))
    ak = compile_device_kernel("add_custom", add(inp(0), inp(1)))
    for dk, vec in ((rk, "Maxs"), (ak, "Add")):
        s = dk.source
        for tok in ("__global__", "__aicore__", "GM_ADDR", "GlobalTensor",
                    "LocalTensor", "TPipe", "TQue", "DataCopy",
                    "GetBlockIdx", "TILE_LENGTH", vec):
            assert tok in s, f"设备 kernel 缺少 {tok}"
    # 复合算子（sigmoid）需要临时 buffer，应生成 TBuf + 多条向量指令
    # 【v8 精度修复】sigmoid 必须用 fp32 Div，绝不用 vrec 近似指令 Reciprocal：
    #   Reciprocal 底层 fp16 近似使 1/1=0.998047，MERE 卡在 9.6e-4 超阈 8×；
    #   Div 底层 vdiv fp32 精确除法把 MERE 压到 2.5e-8（真机 昇腾 NPU 复核）。
    from yidao.expr import sigmoid
    sk = compile_device_kernel("sigmoid_custom", sigmoid(inp(0)))
    assert "TBuf" in sk.source and "Exp" in sk.source and "Div(" in sk.source
    assert "Duplicate(" in sk.source          # 分子常量 1 的 fp32 张量
    assert "Reciprocal" not in sk.source, "sigmoid 不得再用 vrec 近似指令"
    assert rk.n_inputs == 1 and ak.n_inputs == 2
    # 与已在昇腾 NPU 上验证过的源码形态一致
    assert "Maxs(yLocal, x0Local, (float)0.0, TILE_LENGTH);" in rk.source
    assert "Add(yLocal, x0Local, x1Local, TILE_LENGTH);" in ak.source


def test_device_kernel_generator_reduce_family() -> None:
    """行级归约 kernel 生成器：reduce/softmax/layernorm 的设备侧源码形态。"""
    import re
    from yidao.specs import build_specs
    from yidao.ascendc_reduce import compile_reduce_kernel
    sp = build_specs()
    core_tokens = ("__global__", "__aicore__", "GM_ADDR", "GlobalTensor",
                   "LocalTensor", "TPipe", "TQue", "DataCopy", "GetBlockIdx",
                   "ROWS_PER_CORE", "COLS")
    # 标量输出族
    for key, api in (("reduce_sum", "WholeReduceSum"),
                     ("reduce_max", "WholeReduceMax"),
                     ("reduce_mean", "WholeReduceSum")):
        rk = compile_reduce_kernel(key + "_custom", sp[key].tree)
        assert rk.scalar_out is True
        for tok in core_tokens + (api, "SetValue"):
            assert tok in rk.source, f"{key} 缺少 {tok}"
    # 行输出族 + 数值稳定内建（softmax 自动生成 exp(x-max)）
    sm = compile_reduce_kernel("softmax_custom", sp["softmax"].tree)
    assert sm.scalar_out is False
    for tok in core_tokens + ("WholeReduceMax", "WholeReduceSum", "Duplicate", "Sub", "Exp"):
        assert tok in sm.source, f"softmax 缺少 {tok}"
    # layernorm：多归约 + Sqrt + 广播
    ln = compile_reduce_kernel("layernorm_custom", sp["layernorm"].tree)
    for tok in ("Sqrt", "Duplicate", "Sub", "Mul", "WholeReduceSum", "Div"):
        assert tok in ln.source, f"layernorm 缺少 {tok}"
    # 所有缓冲区的引用都必须先声明（防止编号错位）
    for rk in (sm, ln):
        s = rk.source
        decl_s = set(re.findall(r"LocalTensor<float> (s\d+Local)", s))
        use_s = set(re.findall(r"\b(s\d+Local)\b", s))
        decl_t = set(re.findall(r"LocalTensor<float> (t\d+) ", s))
        use_t = set(re.findall(r"\b(t\d+)\b", s))
        assert use_s <= decl_s and use_t <= decl_t, "缓冲区声明与引用不一致"


def test_wrong_kernel_still_rejected() -> None:
    spec = build_specs()["relu"]
    active = {n: catalog()[n] for n in spec.known}
    result = ShengHuDong().verify(spec, active, "bad", lambda x: np.zeros_like(x))
    assert not result.sixiang.yang and result.failures > 0


def test_jijin_from_structure() -> None:
    specs = build_specs()
    jy = JianYi()
    assert jy.identify(specs["matmul"]) == JinGu.ZHEN
    specs["matmul"].doc = "softmax softmax"
    assert jy.identify(specs["matmul"]) == JinGu.ZHEN


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n全部 {len(fns)} 项测试通过。")


if __name__ == "__main__":
    _run_all()
