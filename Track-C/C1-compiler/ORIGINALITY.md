# 原创性声明

本项目为本队针对 Track-C C1 赛题独立设计、实现、调试和维护的 AEC IR 编译器。

项目开发过程中使用了 Cursor/大语言模型作为代码编写、问题定位、测试设计和文档整理的辅助工具。所有设计决策、代码集成、调试验证与最终提交均由参赛队员完成，参赛队员能够解释并维护提交代码。

## 原创实现

| 模块 | 说明 |
|------|------|
| `aec_parser.py` | PTX 9.3 标量子集解析器 |
| `aec_ir.py` | 项目内部 IR 与 CFG 数据结构 |
| `aec_passes.py` | PTX→AEC lowering 与优化流水线 |
| `aec_regalloc.py` | Linear-Scan 寄存器分配 |
| `aec_schedule.py` | 分段依赖感知调度 |
| `aec_backend.py` | 128-bit 指令编码与二进制生成 |

## 第三方依赖

编译器运行时仅使用 Python 标准库，无额外第三方 Python 依赖。

官方提供的测试数据、CModel、Runtime Starter Kit、设备库和固定 Kernel Image 仅用于接口适配与测试，不属于本队原创实现。
