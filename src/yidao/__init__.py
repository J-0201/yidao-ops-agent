"""易道算子 Agent · 易理内核 v6 —— 完整设备侧覆盖版（补齐 MatMul/Gather/动态 tiling）。

v4 在 v3 独立语义基准之上，把「变易·代码合成」抽象为可插拔后端：
  - numpy 后端：树 -> NumPy 源码 -> exec（保留全部既有能力）；
  - ascendc 后端：树 -> 真 AscendC C++ kernel（GM_ADDR/GlobalTensor/
    LocalTensor/TPipe/TQue/三段式）-> g++/CANN 桩编译 -> 真跑 -> 与 oracle 对拍。

v5 补齐三处成熟度边界：coverage 属性随机对拍、public_api 公共扩展、安全 exec。
v6 把 v5 文档列为「下一步」的三类设备 kernel 全部补齐并真机验证：
  - ascendc_gather：按索引收集行（模板展开规避变址读坑），真机 err=0；
  - ascendc_dynamic：任意长度 + 非整 repeat 尾块（32B 对齐分块），真机 err=0；
  - ascendc_matmul：向量化点积累加 MatMul（非 Cube，诚实标注），真机 rel=3e-7。

三判据验证层与后端无关，两个后端共用同一套语义基准与观复闭环。
易理为内核，易象为皮肤；凡不能写成可执行约束者不进内核。
"""

__version__ = "1.0.0"

from .agent import YiDaoAgent
from .expr import Expr
from .model import BuYi, DevResult, JinGu, OpSpec, Polarity, ShiWei, SiXiang, Symptom
from .semantic import Property, SemanticSpec, get_semantic
from .ascendc import compile_ascendc_source, infer_shape, AstKernel
from .ascendc_runtime import AscendCRuntime
from .ascendc_device import compile_device_kernel, DeviceKernel
from .ascendc_reduce import compile_reduce_kernel, RowKernel
from .ascendc_gather import compile_gather_kernel, GatherKernel
from .ascendc_dynamic import compile_dynamic_kernel, DynamicKernel
from .ascendc_matmul import compile_matmul_kernel, MatMulKernel
from . import semantic
from . import coverage
from .evalbench import (
    EvalRegistry, EvalContract, EvalProto, EvalCase, ContractError,
)
from .bench_data import build_registry
from .public_api import (
    make_semantic, make_op, nonnegative, between, finite, row_sums_to_one,
    monotonic, zero_for_nonpositive, identity_for_positive, zero_mean,
    unit_variance, PROPERTY_BUILDERS,
)

__all__ = [
    "__version__",
    "YiDaoAgent", "Expr", "BuYi", "DevResult", "JinGu", "OpSpec",
    "Polarity", "ShiWei", "SiXiang", "Symptom",
    "Property", "SemanticSpec", "get_semantic", "semantic",
    "compile_ascendc_source", "infer_shape", "AstKernel", "AscendCRuntime",
    "compile_device_kernel", "DeviceKernel",
    "compile_reduce_kernel", "RowKernel",
    "compile_gather_kernel", "GatherKernel",
    "compile_dynamic_kernel", "DynamicKernel",
    "compile_matmul_kernel", "MatMulKernel",
    "coverage", "make_semantic", "make_op",
    "EvalRegistry", "EvalContract", "EvalProto", "EvalCase", "ContractError", "build_registry",
    "nonnegative", "between", "finite", "row_sums_to_one", "monotonic",
    "zero_for_nonpositive", "identity_for_positive", "zero_mean",
    "unit_variance", "PROPERTY_BUILDERS",
]
