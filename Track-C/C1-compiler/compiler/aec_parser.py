"""PTX 9.3 restricted scalar subset parser."""
import re
from typing import Optional, List
from aec_ir import *

REG_PREFIX_TYPE = {
    "p": "pred",
    "r": "u32",
    "s": "s32",
    "rd": "u64",
    "b": "b32",
    "bd": "b64",
    "f": "f32",
}


def parse_ptx(text: str) -> Kernel:
    kernel = Kernel()
    cfg = CFG()
    current_block = None
    block_order: List[str] = []

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith((".version", ".target", ".address_size")):
            continue

        m = re.match(r"\.visible\s+\.entry\s+(\w+)\s*\(", line)
        if m:
            kernel.name = m.group(1)
            blk = BasicBlock(name="entry")
            cfg.blocks["entry"] = blk
            cfg.entry = "entry"
            block_order.append("entry")
            current_block = blk
            continue

        m = re.match(r"\.param\s+\.(\w+)\s+(\w+)", line)
        if m:
            kernel.params.append((m.group(2), m.group(1)))
            continue

        m = re.match(r"\.reg\s+\.(\w+)\s+(%[\w]+)<(\d+)>", line)
        if m:
            ptx_type, reg_base = m.group(1), m.group(2)
            for i in range(int(m.group(3))):
                kernel.reg_types[f"{reg_base}{i}"] = ptx_type
            continue

        if line in ("(", ")", "{", "}", ";"):
            continue

        m = re.match(r"^(\w+):\s*$", line)
        if m:
            label = m.group(1)
            if label not in cfg.blocks:
                blk = BasicBlock(name=label)
                cfg.blocks[label] = blk
                block_order.append(label)
            else:
                blk = cfg.blocks[label]
            if current_block and label not in current_block.successors:
                current_block.successors.append(label)
            blk.predecessors.append(current_block.name if current_block else "entry")
            current_block = blk
            continue

        inst = _parse_instruction(line, kernel.reg_types)
        if inst and current_block:
            current_block.add(inst)

    cfg.block_order = block_order
    kernel.cfg = cfg
    return kernel


def _vreg(name: str, reg_types: Dict[str, str]) -> SSAValue:
    ptx_type = reg_types.get(name, "u32")
    if name.startswith("%rd") or name.startswith("%bd"):
        ptx_type = reg_types.get(name, "u64" if name.startswith("%rd") else "b64")
    is_u64 = ptx_type in ("u64", "b64")
    return SSAValue(
        name=name,
        ptx_type=ptx_type,
        aec_type=PTX_TO_AEC.get(ptx_type, AECType.U32),
        is_u64=is_u64,
        high_name=f"{name}__hi" if is_u64 else "",
    )


def _parse_instruction(line: str, reg_types: Dict[str, str]) -> Optional[IRInst]:
    line = line.rstrip(";").strip()
    if not line:
        return None

    pred_val = None
    neg = False
    remaining = line
    if line.startswith("@"):
        m = re.match(r"@(!?)(%\w+)", line)
        if m:
            neg = m.group(1) == "!"
            pred_val = _vreg(m.group(2), reg_types)
            pred_val.ptx_type = "pred"
            remaining = line[m.end():].strip()

    parts = remaining.replace(",", " ").split()
    if not parts:
        return None

    raw_op = parts[0]
    clean_args = [a.strip("[]") for a in parts[1:] if a.strip("[]")]
    operands: List[SSAValue] = []
    for a in clean_args:
        if a in SPECIAL_NAMES:
            operands.append(SSAValue(name=a, ptx_type="u32", is_special=True, special_reg=SPECIAL_NAMES[a]))
        elif a.startswith("%"):
            operands.append(_vreg(a, reg_types))
        elif re.match(r"^-?\d+$", a) or a.lower().startswith(("0x", "0d", "0f")):
            operands.append(SSAValue(name=f"#const_{a}", is_const=True, const_val=_parse_imm(a), ptx_type="u32"))
        else:
            operands.append(SSAValue(name=a, ptx_type="unknown"))

    op_parts = raw_op.split(".")
    op_base = op_parts[0]
    suffixes = [s for s in op_parts[1:] if s not in ("lo", "wide", "rn", "global", "param")]
    ptx_type = "u32"
    for s in reversed(op_parts[1:]):
        if s in PTX_TO_AEC:
            ptx_type = s
            break
    aec_type = PTX_TO_AEC.get(ptx_type, AECType.U32)
    is_wide = "wide" in op_parts

    inst = IRInst()
    inst.pred = pred_val
    inst.neg_pred = neg

    if op_base == "add":
        inst.opcode = AECOpcode.ADD
        inst.aec_type = aec_type
        inst.dst, inst.src1 = operands[0], operands[1]
        inst.src2 = operands[2] if len(operands) > 2 else None
    elif op_base == "sub":
        inst.opcode = AECOpcode.SUB
        inst.aec_type = aec_type
        inst.dst, inst.src1 = operands[0], operands[1]
        inst.src2 = operands[2] if len(operands) > 2 else None
    elif op_base == "mul":
        inst.opcode = AECOpcode.MUL
        inst.aec_type = AECType.U32 if is_wide or ptx_type == "u32" else aec_type
        inst.dst, inst.src1 = operands[0], operands[1]
        inst.src2 = operands[2] if len(operands) > 2 else None
        inst.aec_type = AECType.B64 if is_wide else inst.aec_type
    elif op_base == "mad":
        inst.opcode = AECOpcode.MAD
        inst.aec_type = aec_type
        inst.dst, inst.src1, inst.src2 = operands[0], operands[1], operands[2]
        inst.src3 = operands[3] if len(operands) > 3 else None
    elif op_base == "fma":
        inst.opcode = AECOpcode.FMA
        inst.aec_type = AECType.F32
        inst.dst, inst.src1, inst.src2 = operands[0], operands[1], operands[2]
        inst.src3 = operands[3] if len(operands) > 3 else None
    elif op_base == "and":
        inst.opcode = AECOpcode.AND
        inst.aec_type = AECType.B32
        inst.dst, inst.src1, inst.src2 = operands[0], operands[1], operands[2]
    elif op_base == "or":
        inst.opcode = AECOpcode.OR
        inst.aec_type = AECType.B32
        inst.dst, inst.src1, inst.src2 = operands[0], operands[1], operands[2]
    elif op_base == "xor":
        inst.opcode = AECOpcode.XOR
        inst.aec_type = AECType.B32
        inst.dst, inst.src1, inst.src2 = operands[0], operands[1], operands[2]
    elif op_base == "shr":
        inst.opcode = AECOpcode.SHR
        inst.aec_type = AECType.U32
        inst.dst, inst.src1 = operands[0], operands[1]
        inst.src2 = operands[2] if len(operands) > 2 else None
    elif op_base == "shl":
        inst.opcode = AECOpcode.SHL
        inst.aec_type = AECType.U32
        inst.dst, inst.src1 = operands[0], operands[1]
        inst.src2 = operands[2] if len(operands) > 2 else None
    elif op_base == "mov":
        inst.opcode = AECOpcode.CPY
        inst.aec_type = aec_type
        inst.dst, inst.src1 = operands[0], operands[1]
    elif op_base == "ld":
        inst.opcode = AECOpcode.LD
        inst.aec_type = aec_type
        inst.dst = operands[0]
        inst.addr_reg = operands[1] if len(operands) > 1 else None
        if "param" in raw_op:
            inst.mem_space = MemSpace.PMEM
            inst.src1 = operands[1] if len(operands) > 1 else None
        else:
            inst.mem_space = MemSpace.GMEM
            inst.src1 = operands[1] if len(operands) > 1 else None
    elif op_base == "st":
        inst.opcode = AECOpcode.ST
        inst.aec_type = aec_type
        inst.mem_space = MemSpace.GMEM
        inst.addr_reg = operands[0] if operands else None
        inst.src1 = operands[0] if operands else None
        inst.src2 = operands[1] if len(operands) > 1 else None
    elif op_base == "setp":
        inst.opcode = AECOpcode.CMPP
        inst.aec_type = AECType.U32
        inst.dst = operands[0]
        inst.src1, inst.src2 = operands[1], operands[2] if len(operands) > 2 else None
        inst.cmp_op = _parse_cmp(suffixes)
    elif op_base == "bra":
        inst.is_branch = True
        inst.is_terminator = True
        inst.branch_target = operands[0].name if operands else ""
        inst.opcode = AECOpcode.BRX if pred_val else AECOpcode.BR
    elif op_base == "ret":
        inst.opcode = AECOpcode.HALT
        inst.is_terminator = True
    else:
        return None
    return inst


def _parse_cmp(suffixes) -> CmpOp:
    for s in suffixes:
        if s == "eq":
            return CmpOp.EQ
        if s == "ne":
            return CmpOp.NE
        if s == "lt":
            return CmpOp.LT
        if s == "le":
            return CmpOp.LE
        if s == "gt":
            return CmpOp.GT
        if s == "ge":
            return CmpOp.GE
    return CmpOp.EQ


def _parse_imm(s: str) -> int:
    neg = s.startswith("-")
    s = s.lstrip("-").lower()
    if s.startswith("0f"):
        val = int(s[2:], 16)
    elif s.startswith("0x"):
        val = int(s[2:], 16)
    elif s.startswith("0d"):
        val = int(s[2:], 10)
    else:
        val = int(s) if s.isdigit() else 0
    return -val if neg else val
