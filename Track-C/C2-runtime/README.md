# Track-C C2 Runtime

本目录保存 Track-C / C2 主机侧 Runtime 的队伍实现源码与解释文档。

注意：这里不是完整官方 starter-kit，也不包含最终比赛提交物。正式构建与验证仍需要使用官方 C2 starter-kit，因为官方 starter-kit 中包含公共头文件、固定 kernel image、受控设备库和 public grader。

## 目录结构

```text
Track-C/C2-runtime/
├── src/
│   └── aec_runtime.cpp
├── agents/
│   ├── dma_agent.py
│   └── kernel_agent.py
├── explanations/
│   ├── 00_实现总览.md
│   ├── 01_aec_runtime_cpp详解.md
│   ├── 02_dma_agent_py详解.md
│   ├── 03_kernel_agent_py详解.md
│   └── 04_原创性与提交说明.md
├── ORIGINALITY.md
└── README.md
```

## 文件说明

- `src/aec_runtime.cpp`：C2 Runtime 主体实现。编译后生成 `libaec.so`。
- `agents/dma_agent.py`：DMA policy Agent，按公开周期公式枚举合法策略。
- `agents/kernel_agent.py`：Kernel image policy Agent，只在输入 candidates 中选择合法 image。
- `explanations/`：实现解释、答辩说明、提交说明和合规说明。
- `ORIGINALITY.md`：原创性、依赖和 AI 辅助披露草稿。



## 如何放入官方 starter-kit

假设官方 starter-kit 位于：

```text
Track-C/C2-runtime/starter-kit/
```

将本目录文件覆盖到官方 starter-kit 对应位置：

```bash
cp src/aec_runtime.cpp <starter-kit>/src/aec_runtime.cpp
cp agents/dma_agent.py <starter-kit>/agents/dma_agent.py
cp agents/kernel_agent.py <starter-kit>/agents/kernel_agent.py
```

## 构建

进入官方 starter-kit 根目录后执行：

```bash
make clean
make -j2
```

构建产物：

```text
libaec.so
```

`libaec.so` 是构建产物，不建议提交到 GitHub 源码仓库。

## 本地验证

在官方 starter-kit 根目录执行：

```bash
LD_LIBRARY_PATH=$PWD/lib python3 grader/public_grade.py --submission . --profile public
```

如果验证 `submission/` 目录：

```bash
LD_LIBRARY_PATH=$PWD/lib python3 grader/public_grade.py --submission submission --profile public
```

当前公开验证结果：

```text
AEC score: 88.000000/100 — Good
16/16 public cases passed
```

说明：public profile 不计算 hidden performance，所以 R401/R402 在公开报告中各显示 `4.0/10` 是正常现象。

## 生成正式提交目录

在官方 starter-kit 根目录执行：

```bash
rm -rf submission
mkdir -p submission/agents
cp libaec.so submission/
cp agents/dma_agent.py submission/agents/
cp agents/kernel_agent.py submission/agents/
```

最终比赛平台提交目录应为：

```text
submission/
├── libaec.so
└── agents/
    ├── dma_agent.py
    └── kernel_agent.py
```



## 合规说明

本实现遵循以下原则：

- 不修改官方头文件、设备库、kernel image、grader、schema 或测试定义。
- Runtime 计算接口通过官方 Device ABI resolve fixed image 并 submit 执行，不在 Host 侧直接计算结果。
- Agent 策略基于通用输入字段和候选约束，不根据隐藏测试标识、case id、输入 hash 或固定数据生成答案。
- 使用大语言模型辅助开发已在 `ORIGINALITY.md` 中披露。

