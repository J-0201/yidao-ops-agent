"""AscendC runtime：把生成的 C++ kernel 源码用 g++ 编译成 .so，
通过 ctypes 桥接 numpy 数组，产出与 numpy 后端同签名的可调用 kernel。

这是「真实后端」的执行侧证据：源码不是摆设，它被真正编译、加载、执行，
并在 shenghudong 中与独立 oracle 对拍。沙箱用 CANN 桩 + g++；
有 CANN 的机器上把 stub 换成真 kernel_operator.h + ccec 即可烧录 NPU。
"""

from __future__ import annotations

import ctypes
import hashlib
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .ascendc import AstKernel

_STUB = Path(__file__).parent / "cann_stub"


class AscendCRuntime:
    def __init__(self, cxx: str = "g++", opt: str = "-O2", cache: bool = True) -> None:
        self.cxx = cxx
        self.opt = opt
        self.cache = cache
        self._cache: dict[str, Callable[..., Any]] = {}
        self._dir = tempfile.mkdtemp(prefix="yidao_ascendc_")
        if not (_STUB / "kernel_operator.h").exists():
            raise FileNotFoundError(f"CANN 桩头文件缺失：{_STUB / 'kernel_operator.h'}")

    def build(
        self,
        ak: AstKernel,
        shape_fn: Callable[[list[tuple[int, ...]]], tuple[int, ...]],
    ) -> Callable[..., np.ndarray]:
        """编译 kernel 源码 -> .so -> 返回 numpy 可调用函数。

        shape_fn：由实际输入 shape 推导输出 shape（输出 shape 随时位变化，
        同一 .so 复用，故必须调用期推导，不能编译期写死）。
        """
        digest = hashlib.sha1(ak.source.encode()).hexdigest()[:12]
        if self.cache and digest in self._cache:
            return self._cache[digest]
        cpp = Path(self._dir) / f"kernel_{digest}.cpp"
        so = Path(self._dir) / f"kernel_{digest}.so"
        cpp.write_text(ak.source, encoding="utf-8")
        cmd = [
            self.cxx, self.opt, "-shared", "-fPIC", "-std=c++17",
            f"-I{_STUB}", str(cpp), "-o", str(so),
        ]
        import subprocess
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(
                f"AscendC kernel 编译失败：\n{proc.stderr}\n--- 源码 ---\n{ak.source}"
            )
        lib = ctypes.CDLL(str(so))
        fn = lib.yidao_kernel_launch
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        out_dtype = np.dtype(ak.dtype)

        def kernel(*inputs: np.ndarray) -> np.ndarray:
            ins = [np.ascontiguousarray(x, dtype=np.float32) for x in inputs]
            n = len(ins)
            in_ptrs = (ctypes.POINTER(ctypes.c_float) * n)(
                *[x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for x in ins]
            )
            shape_arrays = []
            for x in ins:
                shp = np.array(x.shape, dtype=np.int32)
                shape_arrays.append(shp.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
            in_shapes = (ctypes.POINTER(ctypes.c_int) * n)(*shape_arrays)
            in_ndims = (ctypes.c_int * n)(*[x.ndim for x in ins])
            out_shape = shape_fn([x.shape for x in ins])
            out = np.empty(out_shape, dtype=np.float32)
            out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            out_shape_arr = (ctypes.c_int * len(out_shape))(*out_shape)
            status = fn(in_ptrs, in_shapes, in_ndims, n, out_ptr, out_shape_arr, len(out_shape))
            if status != 0:
                raise IndexError("AscendC kernel 检测到索引越界（YJ_OOB）")
            return out.astype(out_dtype)

        if self.cache:
            self._cache[digest] = kernel
        return kernel
