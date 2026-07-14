# Agentic AI Infrastructure Hackathon — Track C

> **Agentic4Systems Summer School 2026 — Track 03: Agentic AI Infra**
>
> 编译器 · Runtime · 算子调度 — AEC GPGPU 全软件栈

## 目录结构

```
Track-C/
├── C1-compiler/          # AEC IR 编译器
├── C2-runtime/           # 主机侧 Runtime
├── C3-scheduler/         # 算子调度与模型部署
├── run_all_etests.py     # C1+C2+C3 全部本地测试
└── run_server_tests.sh   # Linux 服务器一键测试脚本
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
cd C1-compiler

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


## CModel 验证（2026-07-14）

5 个公开测试用例在 `aec-precise-linux-x86_64` 上全部通过：

| Case | AEC 指令 | Steps | 状态 |
|------|:---:|---:|:---:|
| T1 `basic_lowering` | 24 | 786,432 | done |
| T2 `scalar_optimization` | 26 | 425,984 | done |
| T3 `memory_reuse` | 30 | 491,520 | done |
| T4 `register_scheduling` | 38 | 622,592 | done |
| T5 `scalar_gemm` | 52 | 612,864 | done |

## 测试环境

| 项目 | 本地 | 评测服务器 |
|------|------|------|
| OS | Windows | Linux 6.8.0 |
| Python | 3.14.0 | 3.12.3 |
| CUDA | 无 | 12.8, H200 |
| ONNX | 1.22.0 | 1.22.0 |
| ONNX Runtime | 1.27.0 | 1.27.0 |
| CuPy | — | 14.1.1 |
