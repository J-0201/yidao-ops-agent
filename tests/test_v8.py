"""v8 增量测试：sigmoid 精度修复（vrec→vdiv）。

实测数据（昇腾 NPU / CANN，shape[8,2048], [-2,2], seed 20261117,
torch.sigmoid 等价 float64 golden, MERE 阈值 2^-13）：
  - v7 Reciprocal(vrec fp16 近似)：MERE=9.56e-04  FAIL（超阈 8×）
  - v8 Div(vdiv fp32 精确除法)：  MERE=2.52e-08  PASS（余量 ~4800×）
  - 边界 x=10: v7=0.998047(错) → v8=0.999955(真值)
本测试在 host 固化「生成源码形态」与「数值口径」。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from yidao.expr import sigmoid, inp, mul, scalar
from yidao.ascendc_device import compile_device_kernel
from yidao.semantic import SEMANTICS, _sigmoid_oracle
from yidao.bench_data import build_registry

PASS = []


def check(name, cond):
    assert cond, f"FAIL: {name}"
    PASS.append(name)
    print("  PASS", name)


def test_sigmoid_kernel_uses_f32_div():
    """生成的 sigmoid 设备 kernel 用 Div，不含 Reciprocal。"""
    k = compile_device_kernel("sigmoid_v8", sigmoid(inp(0)), total_length=8 * 2048)
    lines = [l.strip() for l in k.source.splitlines()]
    div_lines = [l for l in lines if l.startswith("Div(")]
    check("sigmoid 含 fp32 Div 指令", len(div_lines) == 1)
    check("sigmoid 无 Reciprocal(vrec 近似)", "Reciprocal" not in k.source)
    check("sigmoid 含分子常量 1 (Duplicate)", any(l.startswith("Duplicate(") for l in lines))
    check("sigmoid 仍用 vexp (Exp)", any(l.startswith("Exp(") for l in lines))
    check("sigmoid 含 End 输出搬运", any(l.startswith("Adds(yLocal") for l in lines))


def test_gelu_kernel_also_fixed():
    """gelu = x·sigmoid(1.702x) 内的 sigmoid 同样修复。"""
    tree = mul(inp(0), sigmoid(mul(scalar(1.702), inp(0))))
    k = compile_device_kernel("gelu_v8", tree, total_length=8 * 2048)
    check("gelu 设备 kernel 无 Reciprocal", "Reciprocal" not in k.source)
    check("gelu 设备 kernel 用 Div", "Div(" in k.source)


def test_host_numerics_div_beats_threshold():
    """host 模拟：vdir fp32 除法口径下 MERE 远低于 2^-13。"""
    rng = np.random.RandomState(20261117)
    x = rng.uniform(-2, 2, 8 * 2048).astype(np.float32)
    gold = 1.0 / (1.0 + np.exp(-x.astype(np.float64)))
    # 等价于 vdiv：float32 除法结果
    out = (np.float32(1.0) / (np.float32(1.0) + np.exp(-x))).astype(np.float32)
    rel = np.abs(out.astype(np.float64) - gold) / np.maximum(np.abs(gold), 1e-12)
    mere, mare = float(rel.mean()), float(rel.max())
    th = 2 ** -13
    check(f"Div 口径 MERE={mere:.2e} < 2^-13", mere < th)
    check(f"Div 口径 MARE={mare:.2e} < 10*2^-13", mare < 10 * th)

    # 对照：模拟 vrec fp16 近似（截断到 fp16 网格）——必须超阈
    rec = (1.0 / (np.float32(1.0) + np.exp(-x))).astype(np.float16).astype(np.float32)
    rel_bad = np.abs(rec.astype(np.float64) - gold) / np.maximum(np.abs(gold), 1e-12)
    check("vrec(fp16) 对照口径 MERE 超阈（修复必要性）", rel_bad.mean() > th)


def test_semantics_and_bench_contract_unchanged():
    """修复只动设备实现，不动语义 oracle 与评测契约（裁判仍是 float64 tanh）。"""
    oracle = SEMANTICS["sigmoid"].oracle
    x = np.array([-100., -10., 0., 10., 100.], dtype=np.float32)
    g = oracle(x)
    check("sigmoid oracle 边界单调且饱和到[0,1]", g[0] < 1e-3 and g[-1] > 0.999
          and np.all(np.diff(g) > 0))
    reg = build_registry()
    c = reg.get("sigmoid")
    check("评测集 sigmoid 契约存在且用裁判 golden", c is not None and c.golden is _sigmoid_oracle)
