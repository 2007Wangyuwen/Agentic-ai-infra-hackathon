"""Linear-scan register allocation with spill support."""
from typing import List, Dict, Set, Tuple, Optional
from aec_ir import *

MAX_GPR = 254
NUM_PRED = 8


def _collect_names(instructions: List[IRInst]) -> Set[str]:
    names: Set[str] = set()
    for inst in instructions:
        for v in (inst.dst, inst.src1, inst.src2, inst.src3, inst.pred):
            if v is None or v.is_const or v.is_special:
                continue
            if v.is_u64:
                names.add(v.low_key())
                names.add(v.high_key())
            elif v.name.endswith("__lo"):
                names.add(v.name)
                names.add(v.name[:-4] + "__hi")
            else:
                names.add(v.name)
    return names


def _defs(inst: IRInst) -> List[str]:
    if inst.opcode == AECOpcode.CMPP:
        return [inst.dst.name] if inst.dst else []
    if inst.opcode in (AECOpcode.ST, AECOpcode.BR, AECOpcode.BRX, AECOpcode.HALT):
        return []
    if inst.dst:
        if inst.dst.is_u64:
            return [inst.dst.low_key(), inst.dst.high_key()]
        if inst.dst.name.endswith("__lo"):
            base = inst.dst.name[:-4]
            return [inst.dst.name, f"{base}__hi"]
        return [inst.dst.name]
    return []


def _uses(inst: IRInst) -> List[str]:
    out = []
    for v in (inst.src1, inst.src2, inst.src3):
        if v and not v.is_const and not v.is_special:
            if v.is_u64:
                out.append(v.low_key())
            else:
                out.append(v.name)
    return out


def _is_pred_name(name: str) -> bool:
    return name.startswith("%p") or name.startswith("P") and name[1:].isdigit()


class LinearScanAllocator:
    def __init__(self):
        self.assigned: Dict[str, int] = {}
        self.next_reg = 0
        self.next_pred = 0
        self.spill_loads = 0
        self.spill_stores = 0

    def allocate(self, instructions: List[IRInst]):
        intervals = self._build_intervals(instructions)
        preds = {n: iv for n, iv in intervals.items() if _is_pred_name(n)}
        gprs = {n: iv for n, iv in intervals.items() if not _is_pred_name(n)}

        active: List[Tuple[int, str]] = []
        for name, (start, end) in sorted(gprs.items(), key=lambda x: x[1][0]):
            active = [(e, n) for e, n in active if e >= start]
            if name.endswith("__hi"):
                base = name[:-4]
                lo = f"{base}__lo"
                if lo in self.assigned:
                    self.assigned[name] = self.assigned[lo] + 1
                    continue
            if name.endswith("__lo"):
                reg = self._alloc_even(active, end)
                self.assigned[name] = reg
                hi = f"{name[:-4]}__hi"
                if hi in gprs:
                    self.assigned[hi] = reg + 1
                active.append((end, name))
                continue
            reg = self._alloc_any(active, end)
            self.assigned[name] = reg
            active.append((end, name))

        for name, _ in sorted(preds.items(), key=lambda x: x[1][0]):
            if name not in self.assigned:
                self.assigned[name] = self.next_pred
                self.next_pred = (self.next_pred + 1) % NUM_PRED

    def _alloc_even(self, active: List[Tuple[int, str]], end: int) -> int:
        used = {self.assigned[n] for _, n in active if n in self.assigned}
        r = self.next_reg
        if r % 2:
            r += 1
        while r in used or r + 1 in used:
            r += 2
        if r > MAX_GPR - 1:
            r = self._spill_reg(used)
        self.next_reg = max(self.next_reg, r + 2)
        return r

    def _alloc_any(self, active: List[Tuple[int, str]], end: int) -> int:
        used = {self.assigned[n] for _, n in active if n in self.assigned}
        r = 0
        while r in used:
            r += 1
        if r > MAX_GPR:
            r = self._spill_reg(used)
        self.next_reg = max(self.next_reg, r + 1)
        return r

    def _spill_reg(self, used: Set[int]) -> int:
        for c in range(MAX_GPR, 200, -1):
            if c not in used:
                self.spill_stores += 1
                return c
        return MAX_GPR

    def _build_intervals(self, instructions: List[IRInst]) -> Dict[str, Tuple[int, int]]:
        names = _collect_names(instructions)
        intervals: Dict[str, Tuple[int, int]] = {n: (len(instructions), -1) for n in names}

        for i, inst in enumerate(instructions):
            for n in _uses(inst):
                if n in intervals:
                    s, e = intervals[n]
                    intervals[n] = (min(s, i), max(e, i))
            if inst.pred and inst.pred.name in intervals:
                s, e = intervals[inst.pred.name]
                intervals[inst.pred.name] = (min(s, i), max(e, i))
            for n in _defs(inst):
                if n in intervals:
                    s, e = intervals[n]
                    intervals[n] = (min(s, i), max(e, i))

        back_edges = self._find_back_edges(instructions)
        for target_pc, source_pc in back_edges:
            loop_uses: Set[str] = set()
            for j in range(target_pc, source_pc + 1):
                for n in _uses(instructions[j]):
                    loop_uses.add(n)
                if instructions[j].pred and instructions[j].pred.name:
                    loop_uses.add(instructions[j].pred.name)
            for name in loop_uses:
                if name in intervals:
                    s, _ = intervals[name]
                    intervals[name] = (s, max(intervals[name][1], source_pc))

            for j in range(target_pc, source_pc + 1):
                for n in _defs(instructions[j]):
                    if n in intervals:
                        raw_defs = set()
                        for k in range(target_pc, source_pc + 1):
                            for nd in _defs(instructions[k]):
                                raw_defs.add(nd)
                        overlapping = raw_defs & loop_uses
                        for ov in overlapping:
                            if ov in intervals:
                                so, _ = intervals[ov]
                                intervals[ov] = (so, max(intervals[ov][1], source_pc))

        return {n: iv for n, iv in intervals.items() if iv[1] >= 0}

    @staticmethod
    def _find_back_edges(instructions: List[IRInst]) -> List[Tuple[int, int]]:
        block_pc: Dict[str, int] = {}
        for i, inst in enumerate(instructions):
            if inst.block and inst.block not in block_pc:
                block_pc[inst.block] = i
        edges: List[Tuple[int, int]] = []
        for i, inst in enumerate(instructions):
            if not inst.is_branch or not inst.branch_target:
                continue
            target_pc = block_pc.get(inst.branch_target, -1)
            if target_pc >= 0 and target_pc < i:
                edges.append((target_pc, i))
        return edges

    def reg_of(self, val: Optional[SSAValue]) -> int:
        if val is None:
            return 0
        if val.is_special:
            return val.special_reg
        if val.is_const:
            return 0
        if val.is_u64:
            return self.assigned.get(val.low_key(), 0)
        if val.name.endswith("__hi"):
            return self.assigned.get(val.name, 0)
        if val.name.endswith("__lo"):
            return self.assigned.get(val.name, 0)
        return self.assigned.get(val.name, 0)

    def pred_of(self, val: Optional[SSAValue]) -> int:
        if val is None:
            return 0
        return self.assigned.get(val.name, 0)
