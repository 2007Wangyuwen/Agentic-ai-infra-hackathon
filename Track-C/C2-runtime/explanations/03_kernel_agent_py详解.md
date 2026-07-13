# `agents/kernel_agent.py` 详细解释

本文档解释 Kernel Agent 的输入输出协议、合法性过滤、选择策略和合规性。

## 1. 文件定位

`agents/kernel_agent.py` 是 C2 Excellent 等级中的 kernel image policy Agent。

评分器会把它作为独立进程运行：

```bash
python3 agents/kernel_agent.py
```

每次调用只处理一个 JSON 请求，并输出一个 JSON。

## 2. 输入格式

输入示例：

```json
{
  "case_id": 1,
  "dtype": "f16",
  "m": 128,
  "n": 128,
  "k": 128,
  "alignment": 16,
  "workspace": 8192,
  "candidates": [
    {
      "id": "candidate-1",
      "semantic_kernel_id": 12,
      "image_id": 787459,
      "variant": 3,
      "workspace": 8192,
      "alignment": 16,
      "divisibility": 8
    }
  ]
}
```

字段含义：

- `case_id`：case 编号。本实现不使用它做策略分支。
- `dtype`：数据类型名称。
- `m/n/k`：GEMM shape。
- `alignment`：当前输入满足的对齐条件。
- `workspace`：当前允许使用的 workspace bytes。
- `candidates`：评分器提供的合法候选集合，Agent 只能从其中选择。



## 3. 输出格式

输出格式：

```json
{"kernel_id":"candidate-1"}
```

只能包含 `kernel_id` 一个字段。

本实现通过：

```python
json.dump({"kernel_id": selected["id"]}, sys.stdout,
          separators=(",", ":"), sort_keys=True)
```

确保 stdout 只有一个 JSON。

## 4. 合法性过滤

核心函数：

```python
def legal(request, candidate):
```

它检查三个约束。

### workspace

```python
if candidate["workspace"] > request["workspace"]:
    return False
```

如果候选 image 需要的 workspace 大于请求可用 workspace，则不能选。

### alignment

```python
if candidate["alignment"] > request["alignment"]:
    return False
```

如果候选 image 要求更高对齐，而请求不满足，则不能选。

### divisibility

```python
divisibility = candidate["divisibility"]
return (request["m"] % divisibility == 0 and
        request["n"] % divisibility == 0 and
        request["k"] % divisibility == 0)
```

GEMM variant 对 shape 有整除要求：

- naive 通常 divisibility 为 1，所有 shape 都能跑。
- tiled 常见要求 4。
- vectorized 常见要求 8。

只有 M/N/K 都满足对应整除要求，候选才合法。

## 5. 选择策略



### 如果候选里有 `diagnostic_cycles`

公开测试中，评分器可能在 candidates 里附加：

```json
"diagnostic_cycles": 400
```

这是输入请求中的性能提示。本实现使用它选择周期最小的候选：

```python
selected = min(candidates, key=lambda item: (
    item.get("diagnostic_cycles", 1 << 62),
    -item["variant"],
    item["id"],
))
```

排序含义：

1. 优先选 diagnostic cycles 最小的 image。
2. 如果同分，选 variant 更高的 image。
3. 如果还同分，用 id 保证确定性。



### 如果没有 `diagnostic_cycles`

隐藏测试可能不提供 diagnostic cycles。

此时使用通用策略：

```python
selected = max(candidates, key=lambda item: (
    item["variant"],
    -item["workspace"],
    item["id"],
))
```

含义：

1. 优先选更高 variant。
2. 同 variant 下倾向 workspace 更小的候选。
3. 再用 id 保证确定性。

这个策略基于通用经验：vectorized 通常比 tiled 强，tiled 通常比 naive 强，但必须先通过合法性过滤。

## 6. 为什么不能自己构造 image id

Kernel Agent 的规范要求：只能从输入 `candidates` 中选择一个合法 `kernel_id`。

因此本实现不根据 dtype 或 image id 公式自己构造答案，而是：

1. 读取输入 candidates。
2. 过滤合法 candidates。
3. 返回其中一个 candidate 的 `id`。

这避免了输出不存在或不被当前请求允许的 image。

## 7. 为什么不使用 `case_id`

`case_id` 可能出现在输入里，但使用它做策略分支会有针对测试优化的风险。

本实现不使用 `case_id`，只根据：

- shape。
- alignment。
- workspace。
- candidate 自身约束。
- candidate 自带 diagnostic cycles。

选择 image。

## 8. 合规性说明

本 Agent：

- 不访问网络。
- 不读取文件。
- 不保存跨 case 状态。
- 不根据隐藏标识、case id、hash 或固定输入做分支。
- 只从输入 candidates 中选择。
- 只输出一个 JSON。
- 只使用 Python 标准库。

