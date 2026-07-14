"""AEC IR: SSA values, instructions, basic blocks, CFG."""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import IntEnum

# C1 type codes (spec section 5.3)
class AECType(IntEnum):
    B32 = 0
    B64 = 1
    U32 = 2
    S32 = 3
    F32 = 8
    NONE = 0xF

class AECOpcode(IntEnum):
    ADD = 0x0001
    SUB = 0x0002
    MUL = 0x0003
    MAD = 0x0004
    FMA = 0x0005
    AND = 0x0010
    OR = 0x0011
    XOR = 0x0012
    SHL = 0x0014
    SHR = 0x0015
    CMPP = 0x0021
    LD = 0x0030
    ST = 0x0031
    BR = 0x0040
    BRX = 0x0041
    HALT = 0x0045
    CPY = 0x0054
    LOADI = 0x0055
    LOADI64 = 0x0056

class MemSpace(IntEnum):
    GMEM = 0
    SMEM = 1
    CMEM = 2
    LMEM = 3
    PMEM = 4

class CmpOp(IntEnum):
    EQ = 0
    NE = 1
    LT = 2
    LE = 3
    GT = 4
    GE = 5

PTX_TO_AEC = {
    "b32": AECType.B32,
    "b64": AECType.B64,
    "u32": AECType.U32,
    "s32": AECType.S32,
    "u64": AECType.B64,
    "f32": AECType.F32,
    "pred": AECType.U32,
}

PARAM_SIZE = {"u32": 4, "s32": 4, "b32": 4, "f32": 4, "u64": 8, "b64": 8}
PARAM_ALIGN = {"u32": 4, "s32": 4, "b32": 4, "f32": 4, "u64": 8, "b64": 8}

SPECIAL_NAMES = {
    "%tid.x": 0x0100, "%ntid.x": 0x0101, "%ctaid.x": 0x0102, "%nctaid.x": 0x0103,
    "%laneid": 0x0104,
    "%tid.y": 0x0110, "%ntid.y": 0x0111, "%ctaid.y": 0x0112, "%nctaid.y": 0x0113,
    "%tid.z": 0x0120, "%ntid.z": 0x0121, "%ctaid.z": 0x0122, "%nctaid.z": 0x0123,
}
SPECIAL_REVERSE = {v: k for k, v in SPECIAL_NAMES.items()}

U64_TYPES = frozenset({"u64", "b64", "rd", "bd"})


@dataclass
class SSAValue:
    name: str
    ptx_type: str = "u32"
    aec_type: AECType = AECType.U32
    is_const: bool = False
    const_val: int = 0
    is_special: bool = False
    special_reg: int = 0
    is_u64: bool = False
    high_name: str = ""

    def low_key(self) -> str:
        return self.name if not self.is_u64 else f"{self.name}__lo"

    def high_key(self) -> str:
        return self.high_name if self.high_name else f"{self.name}__hi"


@dataclass
class IRInst:
    opcode: AECOpcode = AECOpcode.ADD
    aec_type: AECType = AECType.NONE
    dst: Optional["SSAValue"] = None
    src1: Optional["SSAValue"] = None
    src2: Optional["SSAValue"] = None
    src3: Optional["SSAValue"] = None
    pred: Optional["SSAValue"] = None
    neg_pred: bool = False
    mem_space: MemSpace = MemSpace.GMEM
    cmp_op: CmpOp = CmpOp.EQ
    immediate: int = 0
    branch_target: str = ""
    is_branch: bool = False
    is_terminator: bool = False
    addr_reg: Optional["SSAValue"] = None
    block: str = ""


@dataclass
class BasicBlock:
    name: str
    instructions: List[IRInst] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)

    def add(self, inst: IRInst):
        self.instructions.append(inst)


@dataclass
class CFG:
    blocks: Dict[str, BasicBlock] = field(default_factory=dict)
    entry: str = ""
    block_order: List[str] = field(default_factory=list)

    def linearize(self) -> List[IRInst]:
        result: List[IRInst] = []
        order = self.block_order or self._default_order()
        for name in order:
            if name in self.blocks:
                for inst in self.blocks[name].instructions:
                    inst.block = name
                    result.append(inst)
        return result

    def _default_order(self) -> List[str]:
        result: List[str] = []
        seen = set()
        bname = self.entry
        while bname and bname in self.blocks and bname not in seen:
            seen.add(bname)
            result.append(bname)
            succs = [s for s in self.blocks[bname].successors if s in self.blocks]
            bname = succs[0] if succs else None
        for name in self.blocks:
            if name not in seen:
                result.append(name)
        return result


@dataclass
class Kernel:
    name: str = ""
    params: List[Tuple[str, str]] = field(default_factory=list)
    reg_types: Dict[str, str] = field(default_factory=dict)
    cfg: CFG = field(default_factory=CFG)

    def param_offsets(self) -> Dict[str, int]:
        offsets: Dict[str, int] = {}
        off = 0
        for name, ptype in self.params:
            align = PARAM_ALIGN.get(ptype, 4)
            off = (off + align - 1) // align * align
            offsets[name] = off
            off += PARAM_SIZE.get(ptype, 4)
        return offsets

    def param_block_size(self) -> int:
        if not self.params:
            return 0
        off = 0
        for _, ptype in self.params:
            align = PARAM_ALIGN.get(ptype, 4)
            off = (off + align - 1) // align * align
            off += PARAM_SIZE.get(ptype, 4)
        return (off + 7) // 8 * 8

    def all_instructions(self) -> List[IRInst]:
        return self.cfg.linearize()
