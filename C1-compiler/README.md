# C1 AEC IR 编译器

## 目录结构

```
C1-compiler/
├── compiler/
│   └── aec-cc           ← PTX-to-AEC ISA 编译器
├── disassembler/
│   └── aec-objdump      ← AEC binary 反汇编器
├── agent/
│   └── run_agent        ← 自动优化循环 Agent
└── README.md
```

## 命令模板

```bash
python agent/run_agent
```
