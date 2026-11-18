# 易道（YiDao）

**以易理为方法论内核的算子开发 Agent** —— 把"写出能跑的算子代码"升级为"写出语义可信、可被独立裁判验证的算子代码"。

易道把算子开发抽象为一套可执行的约束系统：

- **易象**（声明式表达式树）：用组合子描述算子意图，而非手写过程式代码；
- **变易**（多后端代码合成）：同一棵树可编译为 NumPy 源码或真 AscendC C++ kernel；
- **简易**（筋骨/补常）：从已知不变量出发，迭代补全缺失约束直到收敛；
- **观复**（三判据验证）：语义基准对拍 + 数学性质检验 + 实现忠实度，三者独立；
- **爻变**（选手与裁判分离）：评测契约 golden/cases 由裁判方持有，选手不可注入、不可覆盖。

> 凡不能写成可执行约束者，不进内核。

---

## 特性

| 能力 | 说明 |
|---|---|
| 声明式表达式树 | `inp` / `relu` / `sigmoid` / `sum_` / `mul` 等组合子描述算子，树可被遍历、重写、编译 |
| 双后端合成 | NumPy 后端（可直接 `exec`）与 AscendC 后端（真 AI Core kernel：GM_ADDR/TPipe/TQue/三段式流水） |
| 独立语义基准 | `SemanticSpec`：权威 oracle + 数学性质；与表达式树零依赖，避免"自己验证自己" |
| 选手与裁判分离 | `EvalContract`（proto + golden + cases）+ `EvalRegistry`；有契约时强制覆盖选手自带基准 |
| 安全执行 | `compile_source` 静态检查 + 受限命名空间 `exec`，拒绝危险调用 |
| 属性随机对拍 | 多 seed、多 shape 的差分测试，输出结构化覆盖报告 |
| 公共扩展 API | `make_op` / `make_semantic` + 性质构造器，第三方算子一站式接入 |

内置 12 个算子的评测契约：`relu / sigmoid / softmax / gelu / add / mul / matmul / reduce_sum / reduce_max / reduce_mean / layernorm / gather`。

---

## 安装

```bash
pip install .
# 或开发模式
pip install -e ".[test]"
```

要求 Python ≥ 3.10，唯一运行时依赖 `numpy`。AscendC 后端需另行安装 CANN 工具链；NumPy 后端零依赖。

---

## 快速开始

### 1. 开发一个内置算子（带独立裁判）

```python
from yidao import YiDaoAgent, build_registry
from yidao.specs import build_specs
from yidao.compile import make_kernel
import numpy as np

spec = build_specs()["relu"]
agent = YiDaoAgent(eval_registry=build_registry())
result = agent.develop(spec)

print(result.converged)     # True
print(result.judged_by)     # official-bench-v1
print(result.proto_ok)      # True —— 通过原型逐字校验

kernel = make_kernel(result.source)
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
print(kernel(x))            # [0. 0. 0. 1. 2.]
```

### 2. 扩展一个新算子（自带可信语义基准）

```python
from yidao import YiDaoAgent, OpSpec, make_semantic
from yidao.expr import sigmoid, inp

spec = OpSpec(
    key="my_sigmoid",
    tree=sigmoid(inp(0)),
    inputs=(("x", "float32", ("N",)),),
    output_dtype="float32",
    known={"输入形状一致", "输出 dtype 可表示"},
    semantic=make_semantic("my_sigmoid", lambda x: 1 / (1 + np.exp(-x))),
)
result = YiDaoAgent().develop(spec)
print(result.converged, result.semantic_trusted)
```

### 3. 选手与裁判分离：封堵"自评分"漏洞

```bash
python -m examples.evalbench_demo
```

该示例演示：一个声称 ReLU、实际把树写成 `x*x` 并自带假 oracle 的"恶意选手"，在无裁判时被误判为正确；接入 `EvalRegistry` 后，裁判方 golden 强制覆盖假 oracle，攻击被当场揭穿。

---

## 项目结构

```
yidao/
├── src/yidao/
│   ├── agent.py            # 主循环：规格→简易→变易→生乎动→爻变
│   ├── expr.py / compile.py# 表达式树 / 编译与安全 exec
│   ├── semantic.py         # 独立语义基准（oracle + 性质）
│   ├── evalbench.py        # 评测契约与裁判注册表
│   ├── bench_data.py       # 内置 12 算子评测集（裁判方资产）
│   ├── specs.py / model.py # 算子规格与数据模型
│   ├── buyi.py / jianyi.py / shenghudong.py / yaobian.py / lesion.py  # 易理方法论内核
│   ├── ascendc*.py         # AscendC 后端（声明式 / 行归约 / Gather / 动态 tiling / MatMul / 设备 kernel）
│   ├── coverage.py         # 属性随机对拍
│   ├── public_api.py       # 公共扩展 API
│   └── cann_stub/          # CANN 头文件桩（无 CANN 环境也可编译验证）
├── tests/                  # 全量回归测试
└── examples/               # 可运行示例
```

---

## 测试

```bash
pytest tests/ -q
```

或运行统一入口（含属性覆盖报告）：

```bash
python run_all.py
```

---

## 版本说明

开源首发版本为 `1.0.0`。代码注释中保留的 `v3`–`v9` 标记是项目内部迭代沿革的设计记录，用于说明各能力的引入背景，不影响当前 API。

## 许可证

Apache License 2.0 —— 详见 [LICENSE](LICENSE)。
