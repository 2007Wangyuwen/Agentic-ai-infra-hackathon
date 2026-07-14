# 原创性声明

本项目为 C1 赛道 AEC IR 编译器，全部代码由参赛者独立开发。

## 编译流程

```text
PTX 9.3 文本 → 解析 → Lowering → 优化（O0–O3）→ 调度 → 寄存器分配 → 128-bit 编码 → .aecbin
```

## 原创实现

| 模块 | 说明 |
|------|------|
| `aec_parser.py` | PTX 9.3 标量子集手写解析器 |
| `aec_ir.py` | 自研 IR（SSA + CFG） |
| `aec_passes.py` | PTX→AEC lowering + 全优化流水线 |
| `aec_regalloc.py` | Linear-Scan 寄存器分配器 |
| `aec_schedule.py` | 分段 dual-issue 指令调度器 |
| `aec_backend.py` | 128-bit 指令编码与二进制生成 |

## 外部代码

无。全部使用 Python 标准库，无第三方依赖。
