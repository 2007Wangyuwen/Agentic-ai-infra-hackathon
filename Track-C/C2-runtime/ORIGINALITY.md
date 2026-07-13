# AEC-C2 原创性与依赖声明

本 C2 Runtime 提交是在官方 starter kit 的公开契约之上完成的队伍原创实现。

## 来源与边界

- 官方 starter kit：公共头文件、固定 kernel image、受控设备库、示例、schema 和 grader
  仅作为 ABI、ISA、数值规范和本地验证契约使用。
- Runtime 实现：`src/aec_runtime.cpp` 为本提交自行实现。
- Agent 实现：`agents/dma_agent.py` 和 `agents/kernel_agent.py` 为本提交自行实现。
- 第三方依赖：除 C++17 标准库和 Python 标准库外，没有引入额外第三方依赖。

## AI 辅助披露

开发过程中使用了大语言模型辅助阅读赛题、梳理实现计划、编写和检查代码。队伍需要能够
解释并维护最终代码；代码逻辑均可由 C2 官方 spec、Runtime API、Device ABI、数值规范和
Agent schema 推导。

## 合规说明

- 未修改官方头文件、受控设备库、kernel image、grader、schema 或公开测试定义。
- Runtime 的 launch 和计算库接口均通过官方 Device ABI 解析固定 image 并提交执行，没有在
  Host 侧直接计算结果来绕过评测路径。
- DMA Agent 基于请求中的 `bytes`、`alignment`、`registered` 和 `concurrency` 按公开周期
  公式枚举合法策略。
- Kernel Agent 基于候选项的 `workspace`、`alignment`、`divisibility`、`variant` 以及输入
  中提供的诊断周期信息选择合法 image。
- Agent 不根据隐藏测试标识、输入 hash、固定答案或公开 case ID 进行分支。
