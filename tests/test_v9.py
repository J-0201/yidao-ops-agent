"""v9 增量测试：设备 kernel tiling 性能优化（TILE128→TILE256）。

实测数据（昇腾 NPU / CANN，shape[8,2048], ACL event 纯 kernel 计时,
10 轮交替采样）：
  - v8 TILE_NUM=8 (TILE=128, 16 循环): 4.204 us  (CV 0.9%)
  - v9 TILE_NUM=4 (TILE=256,  8 循环): 2.993 us  (CV 3.7%)  快 32.8%
  - 参考实现 (TILE=256, 8 循环): 2.972 us  (CV 2.2%)
  - TILE=512 (4 循环): 均值 2.43us 但 CV 8.9%（抖动大），不取为默认。
  - 全部变体精度 MERE=2.52e-08（Div 修复保持）。

结论：性能差距根因是 tile 粒度（循环次数），不是 DataCopyPad/指令选择。
本测试在 host 固化「默认 tiling」「Div 修复不回退」「精度语义不变」。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from yidao.expr import sigmoid, inp, add
from yidao.ascendc_device import compile_device_kernel

PASS = []


def check(name, cond):
    assert cond, f"FAIL: {name}"
    PASS.append(name)
    print("  PASS", name)


def test_default_tiling_is_tile256():
    """默认 tile_num=4 → TILE_LENGTH=256，8 次循环（更优的 tile 性能点）。"""
    k = compile_device_kernel("sigmoid_v9", sigmoid(inp(0)))
    check("默认 TILE_NUM=4", "TILE_NUM = 4;" in k.source)
    check("BUFFER_NUM 仍为 2（双缓冲保持）", "BUFFER_NUM = 2;" in k.source)
    # 精度修复不能被 tiling 改动带崩
    check("无 Reciprocal（v8 精度修复不回退）", "Reciprocal" not in k.source)
    check("仍有 Div", "Div(" in k.source)


def test_explicit_tiling_override_still_works():
    """显式传 tile_num 仍可覆盖默认（性能实验/特殊 shape 用）。"""
    k8 = compile_device_kernel("s8", sigmoid(inp(0)), tile_num=8)
    k4 = compile_device_kernel("s4", sigmoid(inp(0)), tile_num=4)
    check("显式 tile_num=8 生效", "TILE_NUM = 8;" in k8.source)
    check("显式 tile_num=4 生效", "TILE_NUM = 4;" in k4.source)
    # 不同 tiling 不影响向量指令序列
    v8 = [l for l in k8.source.splitlines() if l.strip().startswith("Div(")]
    v4 = [l for l in k4.source.splitlines() if l.strip().startswith("Div(")]
    check("tiling 不改变向量指令", v8 == v4 and len(v4) == 1)


def test_multi_input_kernel_tiling():
    """多输入算子（add）同样应用新 tiling，且指令正确。"""
    k = compile_device_kernel("add_v9", add(inp(0), inp(1)))
    check("add 默认 TILE_NUM=4", "TILE_NUM = 4;" in k.source)
    check("add 用 Add 指令", any(l.strip().startswith("Add(") for l in k.source.splitlines()))
    check("add 双输入", k.n_inputs == 2)


if __name__ == "__main__":
    for fn in [test_default_tiling_is_tile256, test_explicit_tiling_override_still_works,
               test_multi_input_kernel_tiling]:
        print("==", fn.__name__)
        fn()
    print(f"\n全部 {len(PASS)} 项通过")
