"""Backend: register allocation, encoding, raw .aecbin output, disassembler."""
import struct
from typing import List, Dict, Optional, Tuple
from aec_ir import *

TYPE_SHIFT = 3
SUBOP_SHIFT = 8
SPACE_SHIFT = 11
PRED_NEG = 1 << 14
PRED_EN = 1 << 15
MAX_REG = 254


class RegAllocator:
    def __init__(self):
        self.assigned: Dict[str, int] = {}
        self.next_reg = 0
        self.next_pred = 0

    def allocate(self, instructions: List[IRInst]):
        for inst in instructions:
            if inst.opcode == AECOpcode.CMPP:
                self._assign_pred(inst.dst)
                self._assign(inst.src1)
                self._assign(inst.src2)
                continue
            if inst.opcode == AECOpcode.BRX and inst.pred:
                self._assign_pred(inst.pred)
                continue
            self._assign(inst.dst)
            self._assign(inst.src1)
            self._assign(inst.src2)
            self._assign(inst.src3)

    def _align_even(self):
        if self.next_reg % 2:
            self.next_reg += 1

    def _assign(self, val: Optional[SSAValue]):
        if val is None or val.is_const or val.is_special:
            return
        key = val.name
        if key in self.assigned:
            return
        if val.is_u64 or key.endswith("__hi"):
            return
        if key.endswith("__lo"):
            base = key[:-4]
            if base not in self.assigned:
                self._align_even()
                self.assigned[base] = self.next_reg
                self.assigned[f"{base}__lo"] = self.next_reg
                self.assigned[f"{base}__hi"] = self.next_reg + 1
                self.next_reg += 2
            return
        if val.ptx_type in ("u64", "b64"):
            self._align_even()
            self.assigned[key] = self.next_reg
            self.assigned[f"{key}__lo"] = self.next_reg
            self.assigned[f"{key}__hi"] = self.next_reg + 1
            self.next_reg += 2
            return
        if self.next_reg <= MAX_REG:
            self.assigned[key] = self.next_reg
            self.next_reg += 1

    def _assign_pred(self, val: Optional[SSAValue]):
        if val is None or val.name in self.assigned:
            return
        self.assigned[val.name] = self.next_pred
        self.next_pred = (self.next_pred + 1) % 8

    def reg_of(self, val: Optional[SSAValue]) -> int:
        if val is None:
            return 0
        if val.is_special:
            return val.special_reg
        if val.is_const:
            return 0
        if val.is_u64:
            return self.assigned.get(val.low_key(), self.assigned.get(val.name, 0))
        if val.name.endswith("__hi"):
            return self.assigned.get(val.name, 0)
        if val.name.endswith("__lo"):
            return self.assigned.get(val.name, 0)
        return self.assigned.get(val.name, 0)

    def pred_of(self, val: Optional[SSAValue]) -> int:
        if val is None:
            return 0
        return self.assigned.get(val.name, 0)


def _pred_ctrl(inst: IRInst, alloc: RegAllocator) -> int:
    ctrl = 0
    ctrl |= (inst.aec_type.value & 0xF) << TYPE_SHIFT
    if inst.opcode in (AECOpcode.CMPP,):
        ctrl |= (inst.cmp_op.value & 0x7) << SUBOP_SHIFT
    if inst.opcode in (AECOpcode.LD, AECOpcode.ST):
        ctrl |= (inst.mem_space.value & 0x7) << SPACE_SHIFT
    if inst.opcode == AECOpcode.BRX:
        ctrl |= alloc.pred_of(inst.pred) & 0x7
        ctrl |= PRED_EN
        if inst.neg_pred:
            ctrl |= PRED_NEG
    return ctrl


def encode_instruction(inst: IRInst, alloc: RegAllocator, branch_target: int = 0) -> bytes:
    opcode = inst.opcode.value
    pred_ctrl = _pred_ctrl(inst, alloc)

    dst = 0
    src1 = 0
    src2 = 0
    w0 = 0
    w1 = 0

    if inst.opcode == AECOpcode.CMPP:
        dst = alloc.pred_of(inst.dst)
        src1 = alloc.reg_of(inst.src1)
        src2 = alloc.reg_of(inst.src2)
        w1 = src2
    elif inst.opcode == AECOpcode.LOADI:
        dst = alloc.reg_of(inst.dst)
        w0 = inst.immediate & 0xFFFFFFFF
    elif inst.opcode == AECOpcode.LOADI64:
        dst = alloc.reg_of(inst.dst)
        w0 = inst.immediate & 0xFFFFFFFF
        w1 = (inst.immediate >> 32) & 0xFFFFFFFF
    elif inst.opcode == AECOpcode.BR:
        w0 = branch_target & 0xFFFFFFFF
    elif inst.opcode == AECOpcode.BRX:
        w0 = branch_target & 0xFFFFFFFF
    elif inst.opcode in (AECOpcode.HALT,):
        pass
    elif inst.opcode == AECOpcode.ST:
        src1 = alloc.reg_of(inst.src1)
        src2 = alloc.reg_of(inst.src2)
        w1 = src2
    elif inst.opcode in (AECOpcode.MAD, AECOpcode.FMA):
        dst = alloc.reg_of(inst.dst)
        src1 = alloc.reg_of(inst.src1)
        src2 = alloc.reg_of(inst.src2)
        w1 = src2
        w0 = alloc.reg_of(inst.src3)
    elif inst.opcode == AECOpcode.LD:
        dst = alloc.reg_of(inst.dst)
        src1 = alloc.reg_of(inst.src1)
    else:
        dst = alloc.reg_of(inst.dst)
        src1 = alloc.reg_of(inst.src1)
        src2 = alloc.reg_of(inst.src2)
        w1 = src2

    w2 = ((dst & 0xFFFF) << 16) | (src1 & 0xFFFF)
    w3 = ((opcode & 0xFFFF) << 16) | (pred_ctrl & 0xFFFF)
    return struct.pack("<IIII", w0, w1, w2, w3)


def resolve_branches(instructions: List[IRInst]) -> Dict[str, int]:
    block_starts: Dict[str, int] = {}
    for pc, inst in enumerate(instructions):
        if inst.block and inst.block not in block_starts:
            block_starts[inst.block] = pc
    return block_starts


def generate_binary(instructions: List[IRInst], alloc: RegAllocator, block_starts: Dict[str, int]) -> List[bytes]:
    binary: List[bytes] = []
    for i, inst in enumerate(instructions):
        target = 0
        if inst.is_branch:
            target = block_starts.get(inst.branch_target, 0)
        binary.append(encode_instruction(inst, alloc, target))
    return binary


def write_binary(binary: List[bytes], output_path: str):
    with open(output_path, "wb") as f:
        for chunk in binary:
            f.write(chunk)


OP_NAMES = {v.value: k for k, v in AECOpcode.__members__.items()}
T_NAMES = {v.value: k for k, v in AECType.__members__.items()}
MS_NAMES = {0: "gmem", 1: "smem", 2: "cmem", 3: "lmem", 4: "pmem"}


def disassemble(input_path: str) -> str:
    with open(input_path, "rb") as f:
        data = f.read()
    if not data:
        return "Error: empty file\n"
    if len(data) % 16:
        return "Error: size not multiple of 16\n"
    ic = len(data) // 16
    lines = [f"; AEC raw stream: {ic} instructions", ""]
    for i in range(ic):
        off = i * 16
        w0, w1, w2, w3 = struct.unpack_from("<IIII", data, off)
        opcode = (w3 >> 16) & 0xFFFF
        pred_ctrl = w3 & 0xFFFF
        dst = (w2 >> 16) & 0xFFFF
        src1 = w2 & 0xFFFF
        src2 = w1 & 0xFFFF
        op_name = OP_NAMES.get(opcode, f"OP_{opcode:04X}")
        atype = (pred_ctrl >> TYPE_SHIFT) & 0xF
        type_name = T_NAMES.get(atype, "?").lower()
        subop = (pred_ctrl >> SUBOP_SHIFT) & 0x7
        space = (pred_ctrl >> SPACE_SHIFT) & 0x7
        pred = pred_ctrl & 0x7
        pred_neg = bool(pred_ctrl & PRED_NEG)

        if opcode == AECOpcode.LOADI.value:
            lines.append(f"  LOADI.{type_name} R{dst}, 0x{w0:08X}")
            continue
        if opcode == AECOpcode.BR.value:
            lines.append(f"  BR 0x{w0:08X}")
            continue
        if opcode == AECOpcode.BRX.value:
            p = f"!P{pred}" if pred_neg else f"P{pred}"
            lines.append(f"  BRX {p}, 0x{w0:08X}")
            continue
        if opcode == AECOpcode.HALT.value:
            lines.append("  HALT")
            continue
        if opcode == AECOpcode.CMPP.value:
            cmp_names = ["eq", "ne", "lt", "le", "gt", "ge"]
            cmp_s = cmp_names[subop] if subop < len(cmp_names) else str(subop)
            lines.append(f"  CMPP.{cmp_s}.u32 P{dst}, R{src1}, R{src2}")
            continue
        if opcode == AECOpcode.ST.value:
            ms = MS_NAMES.get(space, "gmem")
            lines.append(f"  ST.{ms}.{type_name} [R{src1}], R{src2}")
            continue
        if opcode == AECOpcode.LD.value:
            ms = MS_NAMES.get(space, "gmem")
            lines.append(f"  LD.{ms}.{type_name} R{dst}, [R{src1}]")
            continue
        if opcode in (AECOpcode.MAD.value, AECOpcode.FMA.value):
            lines.append(f"  {op_name}.{type_name} R{dst}, R{src1}, R{src2}, R{w0}")
            continue
        if src1 >= 0x0100 and src1 <= 0x0123:
            lines.append(f"  {op_name}.{type_name} R{dst}, {SPECIAL_REVERSE.get(src1, f'SR_{src1:04X}')}")
        else:
            ops = f"R{dst}, R{src1}"
            if opcode in (AECOpcode.ADD.value, AECOpcode.SUB.value, AECOpcode.MUL.value,
                            AECOpcode.MAD.value, AECOpcode.FMA.value, AECOpcode.AND.value,
                            AECOpcode.OR.value, AECOpcode.XOR.value, AECOpcode.SHL.value,
                            AECOpcode.SHR.value):
                ops += f", R{src2}"
            lines.append(f"  {op_name}.{type_name} {ops}")
    return "\n".join(lines) + "\n"
