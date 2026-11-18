"""v6 增量测试：Gather / 动态 tiling / MatMul 设备生成器。"""

import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from yidao import (compile_gather_kernel, compile_dynamic_kernel,
                    compile_matmul_kernel)
from yidao.expr import relu, add, inp

PASS = []


def check(name, cond):
    assert cond, f"FAIL: {name}"
    PASS.append(name)
    print("  PASS", name)


# ---- 生成器产出真 kernel 源码 ----
def test_gather_kernel_source():
    k = compile_gather_kernel("gather_custom", rows=64, cols=64, k=64, core_num=8)
    assert "__global__ __aicore__" in k.source
    assert "GetValue(I)" in k.source            # 模板常量索引（规避变址坑）
    assert "if constexpr (I < K_PER_CORE)" in k.source
    assert "DataCopy" in k.source and "idxBuf" in k.source
    check("Gather 设备 kernel 源码含模板展开+查表搬运", True)


def test_dynamic_kernel_source():
    r = compile_dynamic_kernel("dyn_relu", relu(inp(0)), total_length=1000, core_num=8)
    a = compile_dynamic_kernel("dyn_add", add(inp(0), inp(1)), total_length=777, core_num=8)
    for k in (r, a):
        assert "constexpr int32_t BLOCK" in k.source and "32B 对齐" in k.source
        assert "TOTAL_LENGTH - start" in k.source   # 尾块 count
        assert "int32_t cc = ((count + 7) / 8) * 8" in k.source  # 32B 对齐
    check("动态 kernel 源码含对齐分块+尾块处理", True)


def test_matmul_kernel_source():
    k = compile_matmul_kernel("matmul_custom", m=32, k=32, n=32, core_num=8)
    assert "__global__ __aicore__" in k.source
    assert "WholeReduceSum" in k.source and "AscendC::Mul" in k.source
    assert "DotCols<C + 1>" in k.source            # 模板展开规避变址
    assert "非 Cube" in k.source                    # 诚实标注
    check("MatMul 设备 kernel 源码含向量点积+模板展开+诚实标注", True)


def test_generator_param_validation():
    for fn, kw in [
        (compile_gather_kernel, dict(k=63, core_num=8)),       # k 不整除
        (compile_matmul_kernel, dict(m=30, k=32, n=32)),       # m 不整除核
        (compile_matmul_kernel, dict(m=32, k=30, n=32)),       # k 非 8 倍数
    ]:
        try:
            fn("bad", **kw)
        except ValueError:
            continue
        raise AssertionError(f"参数校验未拦截：{fn.__name__} {kw}")
    check("生成器参数校验（整除/对齐）生效", True)


# ---- 真机证据本地复核（binary 随包交付，可离线复现）----

if __name__ == "__main__":
    print("=== v6 增量测试 ===")
    test_gather_kernel_source()
    test_dynamic_kernel_source()
    test_matmul_kernel_source()
    test_generator_param_validation()
    print(f"\n全部 {len(PASS)} 项测试通过。")
