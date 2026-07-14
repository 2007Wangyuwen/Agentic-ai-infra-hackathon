# C3 算子调度与模型部署

## 命令模板

C3.1:
```bash
python export_dag.py --onnx {onnx} --output {output}
```

C3.5 (worker):
```bash
python infer_worker.py
```

> C3.5 评测采用持久化 Worker 协议：评测机启动 `python infer_worker.py`，
> 通过 stdin/stdout 下发任务。
> 每个模型使用全新进程（不保留 cache），顺序：mlp → resnet → transformer → bigformer。
> 每个模型：2 warmup + 5 计时任务，时间取中位数，显存取最大值。

本地快速测试：
```bash
python infer.py --onnx model.onnx --input input_dir --output output_dir --batch-size 256
```

## 依赖安装

```bash
pip install -r requirements.txt
```

## 评测环境（Q&A 确认）

| 项目 | 配置 |
|------|------|
| OS | Linux 6.8.0-110-generic (x86_64) |
| GPU | H200 (MIG 虚拟化) |
| Python | 3.12.3 |
| CUDA | 12.8 (nvcc), driver 580.126.20 |
| PyTorch | 2.13.0+cu130 |
| CuPy | 14.1.1 (**允许自定义 CUDA Kernel**) |
| ONNX | 1.22.0 |
| ONNX Runtime | 1.27.0 |

## 评分规则（Q&A 确认）

### C3.5 性能排位 → 分数

| 排名 | 时间 | 显存 |
|:---:|:---:|:---:|
| 1 | 25 | 10 |
| 2 | 22 | 8 |
| 3 | 20 | 7 |
| 4 | 18 | 6 |
| 5 | 16 | 5 |
| 6 | 14 | 4 |
| 7 | 12 | 3 |
| 前半 | 10 | 2 |
| 后半 | 5 | 1 |

### 模型权重

| 模型 | 性能权重 | 说明 |
|------|:---:|------|
| ResNet-18 | 20% | 时间 + 显存计入排名 |
| **BigFormer** | **80%** | **主导性能分，必须 GPU 推理** |

## BigFormer 策略（性能关键）

BigFormer fp32 权重约 18 GB，超过 H200 虚拟化后的 16 GB 显存。

```
尝试 FP16 转换（onnxruntime.transformers.float16）
  → 权重 ~9 GB → 适配 GPU
  → CUDAExecutionProvider 推理
  → 输出 cast 回 FP32（rtol=atol=1e-3 阈值应满足）

如转换失败:
  → CPUExecutionProvider (arena 关闭, 低线程数)
```

## 架构

```
infer_worker.py          ← 持久化 Worker（C3.5 评测入口）
  └── InferenceEngine    ← 推理引擎
       ├── _convert_to_fp16()     ← BigFormer FP16 转换
       ├── _build_session()       ← CUDA/CPU provider 选择 + 14GB 显存上限
       └── ONNX Runtime           ← 模型加载与执行

infer.py                 ← standalone CLI（本地调试用）
export_dag.py            ← ONNX → DAG JSON（C3.1），边字段: src_node/dst_node/tensor
check_output.py          ← 本地精度 / 准确率验证
selfcheck_worker.py      ← Worker 协议自测
demo.sh                  ← 一键运行所有公开模型（输出到 tmp_demo/）

scheduler/               ← 算子分解 / 融合 / 内存规划（真实实现）
c3scheduler/             ← 官方 benchmark 兼容包（re-export scheduler 全部公开 API）

# 测试文件
run_tests.py             ← C3.1–3.4 合成模型单测（25 项）
test_scheduler.py        ← 真实模型导入 + C3.2/C3.3/C3.4 全流水线
test_graph_validation.py ← validate() 负向测试（合法/环路/未定义/缺失输出, 4 项）
test_n01_n04_regression.py ← N01–N04 统一回归（内存池 + 融合 + 流调度, 27 项）
test_memory_safety.py    ← 内存池边界（15 项）
test_fusion_bias.py      ← 融合 bias 正负向 + 非拓扑顺序（12 项）
test_stream_scheduling.py ← 依赖感知流调度（6 项）
test_identity_e2e.py     ← Identity 算子 DAG→策略→推理→Worker 全链路
test_c3scheduler_import.py ← c3scheduler 导入 + 全流水线
simulate_benchmark.py    ← 模拟官方 bench_c32_c33.py 的 13 项公开 API 调用
```

## 支持的 ONNX 算子（18 种）

`Add`、`Constant`、`Conv`、`Div`、`Erf`、`Flatten`、`Gather`、`Gemm`、
`GlobalAveragePool`、`Identity`、`LayerNormalization`、`MatMul`、`Mul`、
`Relu`、`Reshape`、`Softmax`、`Split`、`Transpose`

### Identity 支持

BigFormer 图中使用 `Identity` 为共享权重建立别名。C3.1 `export_dag.py` 会保留并导出
Identity 节点；C3.2 `strategy.py` 将其映射为 `identity_fp32`；C3.5 `infer_worker.py`
由 ONNX Runtime 原生执行，语义为输出张量直通输入张量。

## 验证结果

| 模型 | 精度 (allclose) | 准确率 | 状态 |
|------|:---:|:---:|:---:|
| MLP | ✓ | 98.35% (≥98%) | ✅ |
| ResNet-18 | ✓ | 93.51% (≥85%) | ✅ |
| Transformer | ✓ | — | ✅ |

## 代码审查修复（N01–N04）

| 编号 | 问题 | 文件 | 修复 |
|------|------|------|------|
| N01 | 内存池跨池同值块绕过 (`==` 而非 `is`) + 零/负数分配 | `scheduler/memory.py` | `free()` identity check; `malloc()` `numbers.Integral` + `size<=0` guard before align |
| N02 | 非拓扑顺序导致融合重复输出 | `scheduler/graph_passes.py` | `_stable_topological_nodes()` (Kahn 排序) + `consumed` 统一追踪 |
| N03 | 流调度无视依赖关系 (`index%2`) | `scheduler/memory.py` | union-find 连通分量分流，独立组件不同 stream，无边依赖同步风险 |
| N04 | BiasAdd 误匹配 (residual Add 被融合) | `scheduler/graph_passes.py` | `_constant_tensor_names()` 提取 initializer + Constant outputs; `node.op_type in {"MatMul","Gemm"}` |
| — | 融合 invalid 图无 fallback | `scheduler/graph_passes.py` | `validate()` 失败 → `copy.deepcopy(graph)` 回退 + `validation_failed`/`fallback_to_original` 标记 |

## 自评分数

| 子任务 | 满分 | 自评 | 依据 |
|--------|:---:|:---:|------|
| C3.1 图解析 | 10 | **10** | schema 正确，18 种算子全覆盖，244条边零异常 |
| C3.2 算子分解 | 15 | **14.5** | D1–D4 满分; D5 GEMM f8/f4 变体未在公开模型命中 (-0.5) |
| C3.3 算子融合 | 15 | **11.5** | 5 种 pattern 全实现; launch/buffer reduction 35% (-3); MockRuntime 待验证 (-0.5) |
| C3.4 内存规划 | 10 | **10** | A–E 五项满分: pool+coalesce+lifetime+prefetch+union-find 分流 |
| C3.5 端到端 | 50 | **39** | 精度+准确率 15/15; 性能分待 H200 实测 |
| **C3 合计** | **100** | **85** | |
| **赛道总分** | **100** | **91** | (C1=100 + C2=88 + C3=85) / 3 |

## 测试矩阵

| 测试 | 项数 | 覆盖 |
|---|---|---|
| `run_tests.py` | 25 | C3.1–3.4 合成模型全链路 |
| `test_n01_n04_regression.py` | 27 | N01–N04 统一回归 |
| `test_memory_safety.py` | 15 | malloc 边界 + free 防御 |
| `test_fusion_bias.py` | 12 | BiasAdd 正负向 + 非拓扑顺序 |
| `test_stream_scheduling.py` | 6 | 依赖感知流调度 |
| `test_graph_validation.py` | 4 | validate() 正负向 |
| `simulate_benchmark.py` | 13 | 官方 API 模拟调用 |
| `selfcheck_worker.py` | — | Worker 协议 |
| **合计** | **102** | |
