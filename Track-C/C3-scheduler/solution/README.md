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
export_dag.py            ← ONNX → DAG JSON（C3.1）
check_output.py          ← 本地精度 / 准确率验证
test_scheduler.py        ← C3.2 / C3.3 / C3.4 单元测试
scheduler/               ← 算子分解 / 融合 / 内存规划
```

## 验证结果

| 模型 | 精度 (allclose) | 准确率 | 状态 |
|------|:---:|:---:|:---:|
| MLP | ✓ | 98.35% (≥98%) | ✅ |
| ResNet-18 | ✓ | 93.51% (≥85%) | ✅ |
| Transformer | ✓ | — | ✅ |
