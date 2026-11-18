#!/usr/bin/env python3
"""统一验证入口：一条命令跑完全部回归 + 增量 + 覆盖报告。

用法：python3.11 run_all.py
退出码 0 表示全绿；任一测试失败则非 0。
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

TESTS = [
    "tests/test_v3.py",
    "tests/test_v4.py",
    "tests/test_v5.py",
    "tests/test_v6.py",
    "tests/test_v8.py",
    "tests/test_v9.py",
]


def main() -> int:
    print("=" * 64)
    print(" 易道算子 Agent —— 全量验证")
    print("=" * 64)
    failed: list[str] = []
    for t in TESTS:
        print(f"\n>>> {t}")
        r = subprocess.run([sys.executable, os.path.join(ROOT, t)], cwd=ROOT)
        if r.returncode != 0:
            failed.append(t)

    # 属性随机覆盖报告（作为独立证据打印，不改变退出码——覆盖率不是通过门槛）
    print("\n>>> 属性随机覆盖 (coverage)")
    from yidao import coverage
    from yidao.specs import build_specs
    reports = coverage.coverage_all(build_specs(), seeds=(1, 7, 20261117), n_shapes=3)
    print(coverage.format_report(reports))
    if not all(r.ok for r in reports.values()):
        failed.append("coverage")

    print("\n" + "=" * 64)
    if failed:
        print(f" 存在失败：{failed}")
        return 1
    print(" 全部测试通过（v3 + v4 + v5 + v6 + 属性覆盖）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
