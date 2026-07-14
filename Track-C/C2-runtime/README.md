# C2：AEC Host Runtime

实现 AEC 虚拟 GPGPU 的 Host Runtime (`libaec.so`) 及两个 Agent。

## 文件

```
src/aec_runtime.cpp        # Runtime + 计算库 + 虚拟 Driver
agents/dma_agent.py         # DMA Agent
agents/kernel_agent.py      # Kernel Agent
explanations/               # 实现详解
├── 00_实现总览.md
├── 01_aec_runtime_cpp详解.md
├── 02_dma_agent_py详解.md
├── 03_kernel_agent_py详解.md
└── 04_原创性与提交说明.md
```

## 构建（Linux 服务器）

与官方 starter-kit 一同使用：

```bash
# 将本目录文件覆盖到 starter-kit 对应位置
cp src/aec_runtime.cpp <starter-kit>/src/
cp agents/dma_agent.py <starter-kit>/agents/
cp agents/kernel_agent.py <starter-kit>/agents/

# 构建
cd <starter-kit>
make -j2
LD_LIBRARY_PATH=$PWD/lib python3 grader/public_grade.py --submission . --profile public
```
