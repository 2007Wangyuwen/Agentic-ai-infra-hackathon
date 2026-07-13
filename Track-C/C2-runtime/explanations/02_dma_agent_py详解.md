# `agents/dma_agent.py` 详细解释

本文档解释 DMA Agent 的输入输出协议、策略公式、代码结构和合规性。

## 1. 文件定位

`agents/dma_agent.py` 是 C2 Excellent 等级中的 DMA policy Agent。

它不是 Runtime 的一部分，也不链接 `libaec.so`。评分器会把它作为独立进程启动：

```bash
python3 agents/dma_agent.py
```

每次调用：

1. 从 stdin 读取一个 JSON。
2. 计算策略。
3. 向 stdout 输出一个 JSON。
4. 进程结束。

Agent 必须无状态，不能在 case 间保存信息，也不能访问网络或评分器隐藏文件。

## 2. 输入格式

输入示例：

```json
{
  "case_id": 1,
  "direction": "h2d",
  "bytes": 4096,
  "alignment": 64,
  "registered": true,
  "concurrency": 2
}
```

字段含义：

- `case_id`：case 编号。本实现不使用它做分支。
- `direction`：传输方向，`h2d` 或 `d2h`。
- `bytes`：传输字节数。
- `alignment`：host/device 传输对齐信息。
- `registered`：host range 是否已经注册。
- `concurrency`：可利用的并发度。

## 3. 输出格式

输出示例：

```json
{
  "channel": 0,
  "chunk_bytes": 65536,
  "queue_depth": 2,
  "use_zero_copy": true
}
```

字段合法值：

- `channel`：0 或 1。
- `chunk_bytes`：4096、65536、1048576。
- `queue_depth`：1、2、4、8。
- `use_zero_copy`：布尔值，只有 `registered == true` 时才能为 true。

本实现输出时使用：

```python
json.dump(best[1], sys.stdout, separators=(",", ":"), sort_keys=True)
```

这样 stdout 只有一个 JSON，没有额外日志。

## 4. 周期公式

代码中的 `cycles(request, action)` 实现了赛题公开 DMA 虚拟周期模型：

```text
setup + ceil(ceil(bytes / 32) / parallelism)
      + 24 * (ceil(bytes / chunk_bytes) - 1)
      + alignment_penalty
```

各部分解释：

### setup

```python
setup = 45 if action["use_zero_copy"] else 100
```

如果使用 zero-copy，setup 成本为 45；否则为 100。

zero-copy 只有在输入 `registered == true` 时合法。

### payload

```python
payload = math.ceil(request["bytes"] / 32)
```

以 32 bytes 为一个基本传输单位。

### parallelism

```python
parallelism = min(action["queue_depth"], request["concurrency"], 2)
```

并行度受三个因素限制：

- Agent 选择的 queue depth。
- 请求给出的 concurrency。
- 设备最多两个 DMA 通道。

### chunk penalty

```python
24 * (chunks - 1)
```

chunk 越多，额外开销越多。大传输通常倾向更大的 `chunk_bytes`。

### alignment penalty

```python
alignment_penalty = 0 if request["alignment"] >= 64 else 13
```

低于 64 字节对齐会有额外惩罚。

## 5. 策略实现

核心逻辑：

```python
for chunk_bytes in (4096, 65536, 1048576):
    for queue_depth in (1, 2, 4, 8):
        for use_zero_copy in (False, True):
            ...
```

因为合法动作空间很小，所以直接枚举所有方案。

对每个方案：

1. 检查 zero-copy 合法性。
2. 计算虚拟周期。
3. 用确定性 key 选择最优方案。

## 6. 为什么不使用 `direction` 和 `case_id`

### 不使用 `case_id`

`case_id` 不能用于针对性优化。根据竞赛诚信要求，不能根据测试编号直接生成答案。

本实现完全不读取 `case_id` 来决定策略。

### 基本不使用 `direction`

公开 DMA 周期公式对 H2D 和 D2H 是对称的，因此方向不影响周期选择。保留读取整个 request 是为了符合输入协议，但策略本身基于公开公式。

## 7. tie-break 规则

代码使用：

```python
key = (score, chunk_bytes, queue_depth, not use_zero_copy)
```

含义：

1. 首先选最小 score。
2. 同分时选更小 chunk。
3. 再同分选更小 queue depth。
4. 再同分时优先 `use_zero_copy == true`。

tie-break 的目的只是让输出确定，避免同一个输入在不同 Python 环境中得到不同合法输出。

## 8. 合规性说明

本 Agent：

- 不访问网络。
- 不读取文件。
- 不保存跨 case 状态。
- 不根据 case id、隐藏标识、hash 或固定输入做分支。
- 不输出日志。
- 只使用 Python 标准库。
- 对所有合法输入按同一公式枚举。

## 9. 可解释回答模板

如果被问“DMA Agent 怎么工作的”，可以这样回答：

```text
它读取一次 DMA 请求后，枚举 chunk_bytes、queue_depth、zero_copy 的所有合法组合。
对每个组合使用赛题文档给出的虚拟周期公式计算成本，然后选择周期最小的方案。
zero_copy 只有 registered 输入为 true 时才允许。策略不使用 case_id，也不保存历史状态。
```

