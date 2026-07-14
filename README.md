# Agentic AI Infrastructure Hackathon — Track C

> **Agentic4Systems Summer School 2026 — Track 03: Agentic AI Infra**
>
> 编译器 &middot; Runtime &middot; 算子调度 — AEC GPGPU 全软件栈

## 赛道概览

Track C 聚焦 **AEC GPGPU 的软件栈**，从编译器到底层驱动再到高层算子调度：

```
C1 编译器        -> PTX 风格 IR -> AEC ISA 机器码
C2 Runtime       -> libaec.so + 虚拟设备驱动
C3 算子调度      -> ONNX 模型 -> AEC GPGPU 推理
```

| 赛题 | 原始满分 | 评分构成 |
|------|:---:|------|
| C1 AEC IR 编译器 | 100 | 正确性 50 + 性能 35 + 鲁棒 5 + Agent 10 |
| C2 主机侧驱动 | 100 | Runtime 30 + 计算库 30 + Driver 20 + Agent 20 |
| C3 算子调度 | 100 | 图解析 10 + 分解 15 + 融合 15 + 内存 10 + 端到端 50 |

**赛道总分 = (C1 + C2 + C3) / 3**，每道题归一化到 100 分。

## 当前状态

| 赛题 | 本地测试 | 公开评分 | 目标等级 |
|------|:---:|:---:|:---:|
| C1 | 72/72 ✅ | 待服务器验证 | Excellent |
| C2 | Agent 正确 ✅ | 88/100 (public) | Excellent |
| C3 | 全协议通过 ✅ | 待服务器验证 | S 级 |

## 目录结构

```
Track-C/
├── C1-compiler/                     # AEC IR 编译器
│   ├── compiler/                    # 编译器核心
│   │   ├── aec-cc                   # CLI 入口
│   │   ├── aec_ir.py                # IR 数据结构 (SSA/CFG)
│   │   ├── aec_parser.py            # PTX 9.3 解析器
│   │   ├── aec_passes.py            # Lowering + O0-O3 优化流水线
│   │   ├── aec_regalloc.py          # Linear-Scan 寄存器分配
│   │   ├── aec_schedule.py          # 分段 dual-issue 调度
│   │   ├── aec_backend.py           # 128-bit 编码 + .aecbin 输出
│   │   └── aec_tmul.py              # TMUL lowering (占位)
│   ├── disassembler/aec-objdump     # 反汇编器
│   ├── agent/run_agent              # 自动优化 Agent
│   ├── golden_test/                 # 评测模拟器 + 鲁棒性测试
│   └── README.md                    # C1 详细文档
│
├── C2-runtime/                      # 主机侧 Runtime
│   ├── src/aec_runtime.cpp          # Runtime 实现 (~880 行)
│   ├── agents/                      # Excellent 阶段 Agent
│   │   ├── dma_agent.py             # DMA 策略 Agent
│   │   └── kernel_agent.py          # Kernel 选择 Agent
│   ├── explanations/                # 实现文档 (中文)
│   └── README.md                    # C2 详细文档
│
├── C3-scheduler/                    # 算子调度与模型部署
│   └── solution/
│       ├── infer_worker.py           # C3.5 持久化 Worker (评测入口)
│       ├── infer.py                  # standalone CLI (本地调试)
│       ├── export_dag.py             # C3.1 ONNX -> DAG JSON
│       ├── check_output.py           # 本地精度/准确率验证
│       ├── test_scheduler.py         # C3.2/C3.3/C3.4 单元测试
│       ├── selfcheck_worker.py       # Worker 协议自测
│       ├── run_tests.py              # C3 全模块集成测试
│       └── scheduler/                # 调度器核心
│           ├── graph.py              # GraphIR + GraphNode
│           ├── strategy.py           # 算子分解 + 多精度路由
│           ├── graph_passes.py       # 算子融合 passes
│           └── memory.py             # 内存池 + 生命周期 + 执行计划
│
├── run_all_etests.py                 # C1+C2+C3 全部本地测试
├── run_server_tests.sh               # Linux 服务器一键测试脚本
└── .gitignore
```

## 快速开始

### 本地测试 (Windows / macOS / Linux)

```bash
cd Track-C
python run_all_etests.py
```

在所有平台上用合成数据测试 C1 编译/反汇编/优化、C2 Agent、C3 全模块。

### 服务器测试 (Linux)

```bash
# 1. 上传代码到服务器
# 2. 运行全部测试
cd Track-C
bash run_server_tests.sh all

# 或分项运行
bash run_server_tests.sh c1
bash run_server_tests.sh c2
bash run_server_tests.sh c3
```

输出保存到 `server_test_output.txt`。

## 各赛题命令

### C1 — AEC IR 编译器

```bash
# 编译
python compiler/aec-cc input.ptx -O2 -o output.aecbin
# 反汇编
python disassembler/aec-objdump output.aecbin
# 评测模拟
python golden_test/eval_sim.py
# 鲁棒性测试
python golden_test/robustness_test.py
```

### C2 — Host Runtime

```bash
cd C2-runtime/starter-kit
make -j2
LD_LIBRARY_PATH=$PWD/lib python3 grader/public_grade.py --submission . --profile public
```

### C3 — 算子调度

```bash
cd C3-scheduler/solution
pip install -r requirements.txt

# C3.1 图解析
python export_dag.py --onnx {onnx} --output {output}

# C3.5 推理 (评测使用 worker 协议)
python infer_worker.py

# 本地快速推理
python infer.py --onnx model.onnx --input input_dir --output output_dir --batch-size 256

# 精度验证
python check_output.py --pred output_dir --golden golden_dir --labels labels.npy
```

## BigFormer 策略 (C3)

BigFormer fp32 权重约 18 GB，超过 H200 虚拟化后的 16 GB 显存。Worker 自动检测大模型并执行 FP16 转换：

1. 尝试 `onnxruntime.transformers.float16` 转换 → 权重 ~9 GB
2. GPU 推理 + 输出 cast 回 FP32
3. 转换失败则回退 CPU 执行

BigFormer 占 C3.5 性能分的 **80%**。

## 测试环境

| 项目 | 本地 | 评测服务器 |
|------|------|------|
| OS | Windows | Linux 6.8.0 |
| Python | 3.14.0 | 3.12.3 |
| CUDA | 无 | 12.8, H200 |
| ONNX | 1.22.0 | 1.22.0 |
| ONNX Runtime | 1.27.0 | 1.27.0 |
| CuPy | — | 14.1.1 |
