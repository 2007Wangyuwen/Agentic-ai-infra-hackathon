"""Dual-issue scheduling: per-segment list scheduler with FU pairing."""
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from aec_ir import *


def _uses(inst: IRInst) -> List[str]:
    names = []
    for v in (inst.src1, inst.src2, inst.src3):
        if v and not v.is_const and not v.is_special:
            names.append(v.name)
    return names


def _defs(inst: IRInst) -> List[str]:
    if inst.opcode == AECOpcode.CMPP:
        return [inst.dst.name] if inst.dst else []
    if inst.opcode in (AECOpcode.ST, AECOpcode.BR, AECOpcode.BRX, AECOpcode.HALT):
        return []
    if inst.dst:
        return [inst.dst.name]
    return []


def _is_fixed(inst: IRInst) -> bool:
    return inst.opcode in (AECOpcode.BR, AECOpcode.BRX, AECOpcode.HALT, AECOpcode.CMPP, AECOpcode.ST)


_LSU_OPS = frozenset({AECOpcode.LD, AECOpcode.ST})
_CFU_OPS = frozenset({AECOpcode.BR, AECOpcode.BRX, AECOpcode.HALT})


def _fu_class(op: AECOpcode) -> int:
    """0 = LSU, 1 = ALU, 2 = CFU."""
    if op in _LSU_OPS:
        return 0
    if op in _CFU_OPS:
        return 2
    return 1


def _schedule_segment_dual(insts: List[IRInst]) -> List[IRInst]:
    n = len(insts)
    if n <= 1:
        return insts

    preds: Dict[int, Set[int]] = {i: set() for i in range(n)}
    last_def: Dict[str, int] = {}

    for i, inst in enumerate(insts):
        for name in _uses(inst):
            if name in last_def:
                preds[i].add(last_def[name])
        for name in _defs(inst):
            last_def[name] = i

    remaining = set(range(n))
    ready = {i for i in range(n) if not preds[i]}
    scheduled: List[int] = []

    def prio(i: int) -> Tuple[int, int]:
        op = insts[i].opcode
        if op == AECOpcode.LD and insts[i].mem_space == MemSpace.GMEM:
            return (100, -i)
        if op == AECOpcode.LD:
            return (80, -i)
        if op in (AECOpcode.LOADI, AECOpcode.CPY):
            return (60, -i)
        return (10, -i)

    while remaining:
        if not ready:
            min_idx = min(remaining)
            ready = {min_idx}

        candidates = sorted(ready, key=lambda i: prio(i), reverse=True)
        first = candidates[0]

        scheduled.append(first)
        ready.discard(first)
        remaining.discard(first)

        for j in list(remaining):
            preds[j].discard(first)
            if not preds[j]:
                ready.add(j)

        if not ready:
            continue

        paired = None
        for c in sorted(ready, key=lambda i: prio(i), reverse=True):
            if _fu_class(insts[c].opcode) == _fu_class(insts[first].opcode):
                continue
            defs_f = set(_defs(insts[first]))
            uses_f = set(_uses(insts[first]))
            defs_c = set(_defs(insts[c]))
            uses_c = set(_uses(insts[c]))
            if (defs_f & uses_c) or (defs_c & uses_f) or (defs_f & defs_c):
                continue
            paired = c
            break

        if paired is not None:
            scheduled.append(paired)
            ready.discard(paired)
            remaining.discard(paired)
            for j in list(remaining):
                preds[j].discard(paired)
                if not preds[j]:
                    ready.add(j)

    return [insts[i] for i in scheduled]


_schedule_segment = _schedule_segment_dual


def schedule_block(insts: List[IRInst]) -> List[IRInst]:
    result: List[IRInst] = []
    segment: List[IRInst] = []
    for inst in insts:
        if _is_fixed(inst):
            if segment:
                result.extend(_schedule_segment(segment))
                segment = []
            result.append(inst)
        else:
            segment.append(inst)
    if segment:
        result.extend(_schedule_segment(segment))
    return result


def schedule_instructions(instructions: List[IRInst]) -> List[IRInst]:
    blocks: Dict[str, List[IRInst]] = defaultdict(list)
    order: List[str] = []
    for inst in instructions:
        b = inst.block or "entry"
        if b not in blocks:
            order.append(b)
        blocks[b].append(inst)
    result: List[IRInst] = []
    for b in order:
        result.extend(schedule_block(blocks[b]))
    return result
