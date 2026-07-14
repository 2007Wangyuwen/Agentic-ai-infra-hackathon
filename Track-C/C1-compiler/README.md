# C1 AEC IR 编译器

面向 AEC 指令集的 PTX 9.3 标量子集编译器，生成 128-bit 定长 `.aecbin` 指令流。

---

## 目录

```
C1-compiler/
├── compiler/               ← 编译器核心
│   ├── aec-cc              ← CLI 入口
│   ├── aec_ir.py           ← IR 数据结构 (SSA/CGF/类型/Opcode)
│   ├── aec_parser.py       ← PTX 9.3 标量子集文本解析
│   ├── aec_passes.py       ← Lowering + 优化流水线 (O0–O3)
│   ├── aec_regalloc.py     ← Linear-Scan 寄存器分配
│   ├── aec_schedule.py     ← 分段 dual-issue 调度
│   ├── aec_backend.py      ← 指令编码 + .aecbin 输出 + 反汇编
│   └── aec_tmul.py         ← TMUL lowering 占位 (C1 标量子集不使用)
├── disassembler/           ← 反汇编器
│   └── aec-objdump
├── agent/                  ← 自动优化 Agent
│   └── run_agent
├── build/                  ← 构建输出目录
└── README.md
```

---

## 命令

```bash
# 编译 (默认 O2)
python compiler/aec-cc kernel.ptx -o output.aecbin

# 指定优化级别
python compiler/aec-cc kernel.ptx -O0 -o output.aecbin
python compiler/aec-cc kernel.ptx -O2 -o output.aecbin
python compiler/aec-cc kernel.ptx -O3 -o output.aecbin

# 生成编译报告
python compiler/aec-cc kernel.ptx -O2 -o output.aecbin --report compile_report.json

# 反汇编
python disassembler/aec-objdump output.aecbin

# Agent 自优化循环
python agent/run_agent
```

---

## 优化流水线

| 级别 | Pass | 说明 |
|:---:|---|---|
| O0 | PTX Lowering | 无优化，逐指令 lowering |
| O1 | const_prop → DCE | 常量传播、死代码删除 |
| O2 | CSE → const_prop → DCE → GMEM load CSE → load_hoist → 地址 CSE → LICM → gemm_iv_opt → peephole → DCE → merge_blocks → 分段 dual-issue 调度 | 全优化 + GEMM 增量地址 + 基本块合并 + 调度 |
| O3 | （续 O2） | O3 保留为未来扩展，当前与 O2 等价 |

### O1 Pass

- **常量传播** — 追踪 LOADI/CPY 链，折叠常量
- **DCE** — 谓词感知的死代码消除，并清理未引用的 LOADI

### O2 Pass

- **算术 CSE** — 公共子表达式消除 (ADD/SUB/MUL/MAD/FMA/AND/OR/XOR/SHL/SHR)
- **GMEM load CSE** — 消除对同一地址的重复 global memory load
- **Load Hoisting** — 循环中地址不变的 global load 外提至 pre-header
- **地址 CSE** — 复用重复的地址计算
- **LICM** — 正向不动点循环不变量外提（追踪传递闭包）
- **gemm_iv_opt** — 识别 GEMM K-loop 模式，替换为增量地址 + MAD.f32 融合：
  - Pre-loop: 12 条（一次性初始化指针 + stride）
  - 循环体: 9 条/轮 (CMPP + BRX + 2×LD + MAD.f32 + 3×ADD stride + BR)
- **Peephole** — 5 规则：死 hi 消除、自赋值 CPY 消除、MUL×4→SHL×2、LOADI 去重、CPY 别名前向替换
- **merge_blocks** — 单前驱后继的 fall-through 基本块合并
- **分段 dual-issue 调度** — 基本块内按依赖排序 + LSU/ALU 配对（≤2 条/cycle），控制流保持原位（GEMM 路径跳过调度）

### O3 Pass

- 保留为未来扩展。O3 当前与 O2 等价，所有优化已在 O2 生效。

### Peephole 规则

| 匹配 | 替换 |
|------|------|
| `LOADI Rh, 0` 且 `Rh` 无引用 | 删除 |
| `CPY Rd, Rd` | 删除 |
| `MUL Rd, Rs, LOADI(4)` | `LOADI Rt, 2; SHL Rd, Rs, Rt` |
| `LOADI Rd, V` 再次出现 | `CPY Rd, first_V_R` |
| `CPY Ra, Rb` + `ADD Rd, Ra, Rc` | `ADD Rd, Rb, Rc` (别名前向替换) |

---

## 编译结果

| 测试 | O0 | O2 | O3 | 关键优化 |
|------|:--:|:--:|:--:|------|
| T1 vector_add | 34 | 24 | 24 | CSE, peephole, DCE, merge_blocks |
| T2 repeated_expression | 38 | 26 | 26 | CSE, DCE, peephole, merge_blocks |
| T3 repeated_global_load | 44 | 30 | 30 | GMEM load CSE, peephole, merge_blocks |
| T4 mixed_load_compute | 54 | 38 | 38 | addr CSE, dual-issue schedule, merge_blocks |
| T5 scalar_gemm | 61 | 52 | 52 | gemm_iv_opt (loop body: 9 inst/iter) |

> O2 为正式评测级别。O3 当前与 O2 等价。

## CModel 验证结果（aec-precise-linux-x86_64, O2）

| Case | Status | Steps | 正确性 |
|------|:------:|------:|:------:|
| T1 vector_add | `done` | 786,432 | PASS |
| T2 repeated_expression | `done` | 425,984 | PASS |
| T3 repeated_global_load | `done` | 491,520 | PASS |
| T4 mixed_load_compute | `done` | 622,592 | PASS |
| T5 scalar_gemm | `done` | 612,864 | PASS |

## 鲁棒性测试（35/35 通过）

7 种变异策略 × 5 个测试用例，全部编译通过：

`rename_regs` `shuffle_labels` `dead_code` `param_swap` `low_regs` `extra_branch` `u64_tmp`

## 比赛评测模拟

```bash
# 完整评测流程（编译 → CModel → 正确性比对）
wsl -d Ubuntu -e bash -c '
  rsync -a /mnt/e/C1-compiler/compiler/ ~/c1/compiler/
  cd ~/c1 && python3 golden_test/eval_sim.py
'
```

## PTX → AEC Lowering 规则 (ABI)

### 参数 ABI

```
ld.param.u32 %r1, [param_n]  →  LOADI Rtmp, offset
                                LD.pmem.u32 R1, [Rtmp]

ld.param.u64 %rd1, [param_a] →  LOADI Rtmp, offset
                                LD.pmem.u32 Rrd1_low,  [Rtmp]
                                LOADI Rtmp, offset+4
                                LD.pmem.u32 Rrd1_high, [Rtmp]
```

参数偏移按声明顺序自然对齐，`u64` 对齐到 8，`u32` 对齐到 4。

### 地址 ABI

```
mul.wide.u32 %rd4, %r5, 4  →  MUL.u32 Rrd4_low, R5, 4
                              LOADI Rrd4_high, 0

add.u64 %rd5, %rd1, %rd4    →  ADD.u32 Rrd5_low, Rrd1_low, Rrd4_low
                              LOADI Rrd5_high, 0

ld.global.f32 %f1, [%rd5]   →  LD.gmem.f32 Rf1, [Rrd5_low]
```

Global memory 地址取 low 32 bits（评测保证 high 32 bits = 0）。

### 分支

```
bra LABEL       →  BR target_pc
@%p bra LABEL   →  BRX P, target_pc
@!%p bra LABEL  →  BRX !P, target_pc
ret             →  HALT
```

PC 单位是 AEC 指令索引（非字节偏移）。

---

## 代码生成

### 寄存器分配 — Linear Scan

- 收集所有虚拟寄存器的 live interval
- 64-bit pair (`__lo`/`__hi`) 分配到偶数对齐寄存器对
- Predicate 寄存器独立分配 (P0–P7)
- 最大 254 GPR，支持 spill

### 指令编码 — 128-bit

```
bits [127:112]  Opcode      (16 bit)
bits [111:96]   Pred/Ctrl   (16 bit)
bits [ 95:80]   Dest        (16 bit)
bits [ 79:64]   Src1        (16 bit)
bits [ 63:32]   Src2/Imm32  (32 bit)
bits [ 31: 0]   ImmExt      (32 bit)
```

字段布局：`(w0, w1, w2, w3)` = `(ImmExt, Src2/Imm32, Dest|Src1, Opcode|PredCtrl)`

- `LOADI` 的 32-bit immediate 写入 `w0` (ImmExt)
- `BR`/`BRX` 的 branch target 写入 `w0`
- `ST` 的 value 写入 `w1`

### .aecbin 格式

```
raw 128-bit instruction stream
无 header / 无 data section / 无 relocation / 无 symbol table
entry_pc = 0，所有 label 编译期解析完成
```

---

## 模块说明

| 模块 | 职责 |
|------|------|
| `aec_ir.py` | `AECType`, `AECOpcode`, `MemSpace`, `CmpOp`, `SSAValue`, `IRInst`, `BasicBlock`, `CFG`, `Kernel` |
| `aec_parser.py` | `parse_ptx()` — 文本解析，创建 CFG + SSA |
| `aec_passes.py` | `lower_ptx()`, `lower_immediates()`, `run_opt_pipeline()` — 全部 lowering/优化 pass |
| `aec_regalloc.py` | `LinearScanAllocator` — interval 构建 + 寄存器分配 |
| `aec_schedule.py` | `schedule_instructions()` — 分段 dual-issue 调度 (LSU/ALU 配对) |
| `aec_backend.py` | `encode_instruction()`, `generate_binary()`, `write_binary()`, `disassemble()` |
| `aec_tmul.py` | 占位 — C1 标量子集不使用 Tensor 指令 |

---

## 测试

测试用例位于 `testcases/`（上级目录或 `../../testcases/`），每个包含 `kernel.ptx` + `manifest.json`。

当前 workspace 测试：
```
E:\工作\hackathon-prep\testcases\
├── T1_basic_lowering/
├── T2_scalar_optimization/
├── T3_memory_reuse/
├── T4_register_scheduling/
└── T5_scalar_gemm/
```
