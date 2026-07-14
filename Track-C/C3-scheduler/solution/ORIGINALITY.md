# C3 原创性与依赖声明

## 来源与边界

- 官方赛题材料：ONNX 模型、测试数据、spec 文档仅作为评测契约使用
- C3.1 图解析：`export_dag.py` 为本提交自行实现
- C3.5 推理：`infer_worker.py` 与 `infer.py` 为本提交自行实现
- C3.2/C3.3/C3.4 算子调度：`scheduler/` 模块为团队协作实现

## 第三方依赖

| 包 | 版本 | 用途 |
|---|---|---|
| numpy | 2.5.1 | 数组操作 |
| onnx | 1.22.0 | ONNX 模型加载与解析 |
| onnxruntime | 1.27.0 | 推理后端 |

## AI 辅助披露

开发过程中使用了大语言模型辅助阅读赛题、梳理实现计划、编写和检查代码。
团队成员能够解释并维护最终代码；代码逻辑均可由 C3 官方 spec 推导。

代码审查中发现的边缘问题已根据审查建议修复：

- **N01** (`scheduler/memory.py`): 内存池 `free()` 改用 `is` 身份比较防止跨池同值块绕过；`malloc()` 增加 `numbers.Integral` + `size<=0` 守卫（对齐前）。
- **N02** (`scheduler/graph_passes.py`): 融合器入口加入 `_stable_topological_nodes()` (Kahn 排序)，防止非拓扑节点列表导致重复输出。`consumed` set 统一追踪所有 pattern。
- **N03** (`scheduler/memory.py`): 流调度从 `index%2` 改为 union-find 连通分量分流——有数据依赖的同组件共享 stream，独立组件使用不同 stream，无跨流依赖风险。
- **N04** (`scheduler/graph_passes.py`): `_is_add_bias` 增加 `_constant_tensor_names()` 提取 initializer + Constant 节点输出；residual Add 的图输入不被误判为 bias。入口恢复 `node.op_type in {"MatMul", "Gemm"}`。
- **Pipeline fallback**: `GraphPassPipeline.run()` 在 `validate()` 失败时 `copy.deepcopy` 原图回退，并记录 `validation_failed`/`fallback_to_original` 标记。

## 合规说明

- 未修改官方测试数据、golden 输出或评测脚本
- 推理输出由通用算法逻辑产生，未根据测试用例 ID 或输入 hash 硬编码答案
- 未使用预计算的推理结果
- `c3scheduler/` 为官方 benchmark 导入目标的兼容代理包，保证 `from c3scheduler import X` 与 `from scheduler import X` 返回完全相同对象
