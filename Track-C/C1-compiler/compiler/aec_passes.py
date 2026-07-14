"""PTX lowering and scalar/memory optimization passes (T2–T5)."""
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
from aec_ir import *

TMP_COUNTER = 0

_PURE_OPS = frozenset({
    AECOpcode.ADD, AECOpcode.SUB, AECOpcode.MUL, AECOpcode.MAD, AECOpcode.FMA,
    AECOpcode.AND, AECOpcode.OR, AECOpcode.XOR, AECOpcode.SHL, AECOpcode.SHR, AECOpcode.CPY,
})
_SIDE_EFFECT = frozenset({AECOpcode.ST, AECOpcode.LD, AECOpcode.CMPP, AECOpcode.BR, AECOpcode.BRX, AECOpcode.HALT})


def _tmp(ptx_type: str = "u32") -> SSAValue:
    global TMP_COUNTER
    TMP_COUNTER += 1
    return SSAValue(name=f"__t{TMP_COUNTER}", ptx_type=ptx_type, aec_type=PTX_TO_AEC.get(ptx_type, AECType.U32))


def _lo(val: SSAValue) -> SSAValue:
    if val.is_u64:
        return SSAValue(name=val.low_key(), ptx_type="u32", aec_type=AECType.U32)
    return val


def _hi(val: SSAValue) -> SSAValue:
    return SSAValue(name=val.high_key(), ptx_type="u32", aec_type=AECType.U32)


def _loadi(dst: SSAValue, imm: int, aec_type: AECType = AECType.NONE) -> IRInst:
    return IRInst(opcode=AECOpcode.LOADI, aec_type=aec_type, dst=dst, immediate=imm & 0xFFFFFFFF)


def _copy_meta(inst: IRInst, **kwargs) -> IRInst:
    base = dict(
        opcode=inst.opcode, aec_type=inst.aec_type, dst=inst.dst, src1=inst.src1,
        src2=inst.src2, src3=inst.src3, pred=inst.pred, neg_pred=inst.neg_pred,
        mem_space=inst.mem_space, cmp_op=inst.cmp_op, immediate=inst.immediate,
        branch_target=inst.branch_target, is_branch=inst.is_branch,
        is_terminator=inst.is_terminator, addr_reg=inst.addr_reg, block=inst.block,
    )
    base.update(kwargs)
    return IRInst(**base)


def _inst_key(inst: IRInst) -> tuple:
    return (
        inst.opcode.value, inst.aec_type.value,
        inst.mem_space.value if inst.opcode in (AECOpcode.LD, AECOpcode.ST) else 0,
        inst.cmp_op.value if inst.opcode == AECOpcode.CMPP else 0,
        inst.src1.name if inst.src1 else "",
        inst.src2.name if inst.src2 else "",
        inst.src3.name if inst.src3 else "",
        inst.immediate,
    )


# ── Lowering ──────────────────────────────────────────────────────────────

def lower_ptx(kernel: Kernel, instructions: List[IRInst]) -> List[IRInst]:
    global TMP_COUNTER
    TMP_COUNTER = 0
    offsets = kernel.param_offsets()
    result: List[IRInst] = []

    for inst in instructions:
        if inst.opcode == AECOpcode.LD and inst.mem_space == MemSpace.PMEM:
            for lowered in _lower_ld_param(inst, offsets):
                lowered.block = inst.block
                result.append(lowered)
            continue

        if inst.opcode == AECOpcode.CPY and inst.src1 and inst.src1.is_const:
            if inst.dst and inst.dst.is_u64:
                result.append(_copy_meta(_loadi(_lo(inst.dst), inst.src1.const_val & 0xFFFFFFFF), block=inst.block))
                result.append(_copy_meta(_loadi(_hi(inst.dst), (inst.src1.const_val >> 32) & 0xFFFFFFFF), block=inst.block))
            else:
                result.append(_copy_meta(_loadi(inst.dst, inst.src1.const_val), block=inst.block))
            continue

        if inst.opcode == AECOpcode.MUL and inst.dst and inst.dst.is_u64:
            lo, hi = _lo(inst.dst), _hi(inst.dst)
            src2 = inst.src2
            if src2 and src2.is_const and src2.const_val > 0 and (src2.const_val & (src2.const_val - 1)) == 0:
                shift = src2.const_val.bit_length() - 1
                stmp = _tmp("u32")
                result.append(_copy_meta(_loadi(stmp, shift), block=inst.block))
                result.append(_copy_meta(IRInst(opcode=AECOpcode.SHL, aec_type=AECType.U32, dst=lo, src1=_lo(inst.src1) if inst.src1 else None, src2=stmp), block=inst.block))
            else:
                result.append(_copy_meta(IRInst(opcode=AECOpcode.MUL, aec_type=AECType.U32, dst=lo, src1=_lo(inst.src1) if inst.src1 else None, src2=src2), block=inst.block))
            result.append(_copy_meta(_loadi(hi, 0), block=inst.block))
            continue

        if inst.opcode == AECOpcode.ADD and inst.dst and inst.dst.is_u64:
            lo, hi = _lo(inst.dst), _hi(inst.dst)
            result.append(_copy_meta(IRInst(opcode=AECOpcode.ADD, aec_type=AECType.U32, dst=lo, src1=_lo(inst.src1), src2=_lo(inst.src2) if inst.src2 else None), block=inst.block))
            result.append(_copy_meta(_loadi(hi, 0), block=inst.block))
            continue

        if inst.opcode == AECOpcode.LD and inst.mem_space == MemSpace.GMEM:
            addr = inst.addr_reg or inst.src1
            low_addr = _lo(addr) if addr and addr.is_u64 else addr
            result.append(_copy_meta(IRInst(opcode=AECOpcode.LD, aec_type=inst.aec_type, mem_space=MemSpace.GMEM, dst=inst.dst, src1=low_addr), block=inst.block))
            continue

        if inst.opcode == AECOpcode.ST:
            addr = inst.addr_reg or inst.src1
            low_addr = _lo(addr) if addr and addr.is_u64 else addr
            result.append(_copy_meta(IRInst(opcode=AECOpcode.ST, aec_type=inst.aec_type, mem_space=MemSpace.GMEM, src1=low_addr, src2=inst.src2), block=inst.block))
            continue

        if inst.opcode == AECOpcode.CPY:
            aet = inst.aec_type if inst.aec_type != AECType.NONE else AECType.U32
            result.append(_copy_meta(IRInst(opcode=AECOpcode.CPY, aec_type=aet, dst=inst.dst, src1=inst.src1), block=inst.block))
            continue

        if inst.is_branch:
            result.append(inst)
            continue
        result.append(inst)
    return result


def _lower_ld_param(inst: IRInst, offsets: Dict[str, int]) -> List[IRInst]:
    param_name = inst.src1.name if inst.src1 else ""
    base = offsets.get(param_name, 0)
    out: List[IRInst] = []
    ptype = inst.dst.ptx_type if inst.dst else "u32"

    if ptype in ("u64", "b64") or (inst.dst and inst.dst.is_u64):
        lo, hi = _lo(inst.dst), _hi(inst.dst)
        t0, t1 = _tmp("u32"), _tmp("u32")
        out.append(_loadi(t0, base))
        out.append(IRInst(opcode=AECOpcode.LD, aec_type=AECType.U32, mem_space=MemSpace.PMEM, dst=lo, src1=t0))
        out.append(_loadi(t1, base + 4))
        out.append(IRInst(opcode=AECOpcode.LD, aec_type=AECType.U32, mem_space=MemSpace.PMEM, dst=hi, src1=t1))
    else:
        tmp = _tmp("u32")
        out.append(_loadi(tmp, base))
        out.append(IRInst(opcode=AECOpcode.LD, aec_type=inst.aec_type, mem_space=MemSpace.PMEM, dst=inst.dst, src1=tmp))
    return out


def lower_immediates(instructions: List[IRInst]) -> List[IRInst]:
    result: List[IRInst] = []
    for inst in instructions:
        if inst.opcode in (AECOpcode.BR, AECOpcode.BRX, AECOpcode.HALT):
            result.append(inst)
            continue
        if inst.src2 and inst.src2.is_const:
            val = inst.src2.const_val & 0xFFFFFFFF
            tmp = _tmp("u32")
            result.append(_copy_meta(_loadi(tmp, val), block=inst.block))
            inst = _copy_meta(inst, src2=tmp)
        result.append(inst)
    return result


# ── T2: constant propagation + CSE + DCE ──────────────────────────────────

def const_prop(instructions: List[IRInst]) -> List[IRInst]:
    copies: Dict[str, SSAValue] = {}
    consts: Dict[str, int] = {}
    result: List[IRInst] = []

    for inst in instructions:
        if inst.opcode == AECOpcode.CPY and inst.dst and inst.src1:
            if inst.src1.is_const:
                consts[inst.dst.name] = inst.src1.const_val
                copies.pop(inst.dst.name, None)
            elif inst.src1.is_special:
                copies.pop(inst.dst.name, None)
                consts.pop(inst.dst.name, None)
            elif inst.src1.name in copies:
                copies[inst.dst.name] = copies[inst.src1.name]
            elif inst.src1.name in consts:
                consts[inst.dst.name] = consts[inst.src1.name]
            else:
                copies[inst.dst.name] = inst.src1
            result.append(inst)
            continue

        if inst.opcode == AECOpcode.LOADI and inst.dst:
            consts[inst.dst.name] = inst.immediate
            copies.pop(inst.dst.name, None)
            result.append(inst)
            continue

        new = inst
        for attr in ("src1", "src2", "src3"):
            v = getattr(inst, attr)
            if v and v.name in copies:
                new = _copy_meta(new, **{attr: copies[v.name]})
        result.append(new)

        if inst.dst:
            copies.pop(inst.dst.name, None)
            consts.pop(inst.dst.name, None)
        if inst.opcode in _SIDE_EFFECT or inst.is_branch:
            copies.clear()
            consts.clear()
    return result


def cse(instructions: List[IRInst]) -> List[IRInst]:
    result: List[IRInst] = []
    seen: Dict[tuple, SSAValue] = {}
    skip = {AECOpcode.LOADI, AECOpcode.LOADI64, AECOpcode.ST, AECOpcode.CMPP, AECOpcode.BR, AECOpcode.BRX, AECOpcode.HALT}
    for inst in instructions:
        if inst.is_branch or inst.is_terminator or inst.opcode in skip:
            result.append(inst)
            continue
        if inst.opcode == AECOpcode.LD:
            result.append(inst)
            continue
        key = _inst_key(inst)
        if key in seen and inst.dst:
            result.append(IRInst(opcode=AECOpcode.CPY, aec_type=inst.aec_type, dst=inst.dst, src1=seen[key], block=inst.block))
        else:
            if inst.dst:
                seen[key] = inst.dst
            result.append(inst)
    return result


def dce(instructions: List[IRInst]) -> List[IRInst]:
    used: Set[str] = set()
    for inst in instructions:
        for src in (inst.src1, inst.src2, inst.src3):
            if src and not src.is_const and not src.is_special:
                used.add(src.name)
                if src.is_u64:
                    used.add(src.low_key())
        if inst.pred and inst.pred.name:
            used.add(inst.pred.name)

    live: Set[int] = set()
    changed = True
    while changed:
        changed = False
        for idx, inst in enumerate(instructions):
            if idx in live:
                continue
            keep = inst.opcode in (AECOpcode.ST, AECOpcode.BR, AECOpcode.BRX, AECOpcode.HALT, AECOpcode.CMPP)
            if keep:
                live.add(idx)
                changed = True
                if inst.dst:
                    used.add(inst.dst.name)
                if inst.opcode == AECOpcode.CMPP:
                    for src in (inst.src1, inst.src2):
                        if src and not src.is_const:
                            used.add(src.name)
            elif inst.dst and inst.dst.name in used:
                live.add(idx)
                changed = True
                used.add(inst.dst.name)
                for src in (inst.src1, inst.src2, inst.src3):
                    if src and not src.is_const and not src.is_special:
                        used.add(src.name)
    pruned = [inst for idx, inst in enumerate(instructions) if idx in live]
    used2: Set[str] = set()
    for inst in pruned:
        for src in (inst.src1, inst.src2, inst.src3):
            if src and not src.is_const and not src.is_special:
                used2.add(src.name)
    return [
        inst for inst in pruned
        if not (inst.opcode == AECOpcode.LOADI and inst.dst and inst.dst.name not in used2)
    ]


# ── T3: memory load CSE + address reuse ───────────────────────────────────

def gmem_load_cse(instructions: List[IRInst]) -> List[IRInst]:
    result: List[IRInst] = []
    seen: Dict[tuple, SSAValue] = {}
    for inst in instructions:
        if inst.opcode == AECOpcode.LD and inst.mem_space == MemSpace.GMEM and inst.src1 and inst.dst:
            key = (inst.mem_space.value, inst.src1.name, inst.aec_type.value)
            if key in seen:
                result.append(IRInst(opcode=AECOpcode.CPY, aec_type=inst.aec_type, dst=inst.dst, src1=seen[key], block=inst.block))
            else:
                seen[key] = inst.dst
                result.append(inst)
        else:
            if inst.opcode == AECOpcode.ST:
                seen.clear()
            result.append(inst)
    return result


def load_hoist(instructions: List[IRInst]) -> List[IRInst]:
    """Hoist loop-invariant LD.gmem before the loop header, replace with CPY."""
    loops = _find_loops(instructions)
    if not loops:
        return instructions

    header, body = loops[-1]
    header_start = next((i for i, inst in enumerate(instructions) if inst.block == header), -1)
    if header_start < 0:
        return instructions

    defined_in_loop: Set[str] = set()
    for inst in instructions:
        if inst.block in body and inst.dst:
            defined_in_loop.add(inst.dst.name)

    hoisted_lds: List[IRInst] = []
    remove_set: Set[int] = set()

    for i, inst in enumerate(instructions):
        if inst.block not in body:
            continue
        if inst.opcode != AECOpcode.LD or inst.mem_space != MemSpace.GMEM:
            continue
        if inst.src1 is None or inst.dst is None:
            continue
        if inst.src1.name in defined_in_loop:
            continue

        hoisted_lds.append(IRInst(
            opcode=AECOpcode.LD, aec_type=inst.aec_type,
            mem_space=MemSpace.GMEM, dst=inst.dst,
            src1=inst.src1, block=header,
        ))
        remove_set.add(i)

    if not hoisted_lds:
        return instructions

    result: List[IRInst] = []
    hoisted_done = False
    for i, inst in enumerate(instructions):
        if not hoisted_done and inst.block == header:
            result.extend(hoisted_lds)
            hoisted_done = True
        if i in remove_set:
            continue
        result.append(inst)
    return result


def addr_cse(instructions: List[IRInst]) -> List[IRInst]:
    result: List[IRInst] = []
    seen: Dict[tuple, SSAValue] = {}
    for inst in instructions:
        if inst.opcode in (AECOpcode.ADD, AECOpcode.MUL, AECOpcode.MAD) and inst.dst:
            key = _inst_key(inst)
            if key in seen:
                result.append(IRInst(opcode=AECOpcode.CPY, aec_type=inst.aec_type, dst=inst.dst, src1=seen[key], block=inst.block))
            else:
                seen[key] = inst.dst
                result.append(inst)
        else:
            result.append(inst)
    return result


# ── T2/T5: LICM + loop strength reduction ─────────────────────────────────

def _find_loops(instructions: List[IRInst]) -> List[Tuple[str, Set[str]]]:
    block_starts: Dict[str, int] = {}
    for i, inst in enumerate(instructions):
        if inst.block and inst.block not in block_starts:
            block_starts[inst.block] = i
    loops = []
    seen_headers: Set[str] = set()
    for i, inst in enumerate(instructions):
        if not inst.is_branch:
            continue
        header = inst.branch_target
        if header in seen_headers:
            continue
        hstart = block_starts.get(header, -1)
        if hstart >= 0 and hstart <= i:
            body: Set[str] = set()
            in_loop = False
            for j, other in enumerate(instructions):
                if other.block == header:
                    in_loop = True
                if in_loop:
                    body.add(other.block)
                if in_loop and other.is_branch and other.branch_target == header and j >= i:
                    break
            if body:
                seen_headers.add(header)
                loops.append((header, body))
    return loops


def licm(instructions: List[IRInst]) -> List[IRInst]:
    """Loop-invariant code motion with fixed-point iteration.

    First pass: identify values NOT defined in the loop as initial invariants.
    Then iterate: if all src operands of a pure instruction are invariants
    (or constants/specials), mark its dst as invariant too.
    Repeat until no new invariants discovered."""
    loops = _find_loops(instructions)
    if not loops:
        return instructions

    invariant_idxs: Set[int] = set()

    for header, body in loops:
        defined_in_loop: Set[str] = set()
        for inst in instructions:
            if inst.block in body and inst.dst:
                defined_in_loop.add(inst.dst.name)

        invariants: Set[str] = set()
        for inst in instructions:
            if inst.block not in body and inst.dst:
                invariants.add(inst.dst.name)

        changed = True
        while changed:
            changed = False
            for idx, inst in enumerate(instructions):
                if inst.block not in body:
                    continue
                if inst.is_branch or inst.is_terminator:
                    continue
                if inst.opcode in (AECOpcode.ST, AECOpcode.CMPP):
                    continue
                if inst.opcode == AECOpcode.LD and inst.mem_space == MemSpace.GMEM:
                    continue

                all_src_invariant = True
                for src in (inst.src1, inst.src2, inst.src3):
                    if src is None or src.is_const or src.is_special:
                        continue
                    if src.name not in invariants and src.name not in defined_in_loop:
                        all_src_invariant = True
                        continue
                    if src.name not in invariants:
                        all_src_invariant = False
                        break
                if all_src_invariant and inst.dst and inst.dst.name not in invariants:
                    invariants.add(inst.dst.name)
                    changed = True

        for idx, inst in enumerate(instructions):
            if inst.block not in body:
                continue
            if inst.is_branch or inst.is_terminator:
                continue
            if inst.opcode in (AECOpcode.ST, AECOpcode.CMPP):
                continue
            if inst.opcode == AECOpcode.LD and inst.mem_space == MemSpace.GMEM:
                continue

            all_invariant = True
            for src in (inst.src1, inst.src2, inst.src3):
                if src and not src.is_const and not src.is_special and src.name not in invariants:
                    all_invariant = False
                    break
            if all_invariant:
                invariant_idxs.add(idx)

    if not invariant_idxs:
        return instructions

    hoisted: List[IRInst] = []
    rest: List[IRInst] = []
    header_set = {h for h, _ in loops}
    for idx, inst in enumerate(instructions):
        if idx in invariant_idxs:
            hoisted.append(inst)
        else:
            rest.append(inst)

    insert_at = 0
    for i, inst in enumerate(rest):
        if inst.block in header_set:
            insert_at = i
            break

    for h in hoisted:
        h.block = rest[insert_at].block if insert_at < len(rest) else h.block
    return rest[:insert_at] + hoisted + rest[insert_at:]


def gemm_iv_opt(instructions: List[IRInst]) -> List[IRInst]:
    """T5 O3: incremental address update in GEMM K-loop.

    Only fires when the loop pattern (two MAD+SHL+ADD address computations)
    is definitively recognized.  Gracefully returns input on mismatch."""
    loops = _find_loops(instructions)
    if not loops:
        return instructions

    header, body = loops[0]
    loop_indices = [i for i, inst in enumerate(instructions) if inst.block in body]
    if len(loop_indices) < 10:
        return instructions

    # —— detect the two MAD ops ——
    a_mad = b_mad = None
    a_mad_i = b_mad_i = -1
    for i in loop_indices:
        inst = instructions[i]
        if inst.opcode != AECOpcode.MAD or inst.src3 is None:
            continue
        if a_mad is None:
            a_mad = inst; a_mad_i = i
        elif i > a_mad_i + 1:
            b_mad = inst; b_mad_i = i
            break
    if a_mad is None or b_mad is None:
        return instructions

    k_v = a_mad.src3
    K_v = a_mad.src2
    row_v = a_mad.src1
    N_v = b_mad.src2
    col_v = b_mad.src3
    if not all((k_v, K_v, row_v, N_v, col_v)):
        return instructions

    # —— locate the SHL after each MAD ——
    a_shl_i = -1
    for i in range(a_mad_i + 1, b_mad_i):
        inst = instructions[i]
        if inst.opcode == AECOpcode.SHL and inst.src1 and inst.src1.name == a_mad.dst.name:
            a_shl_i = i; break
    b_shl_i = -1
    for i in range(b_mad_i + 1, min(b_mad_i + 6, len(instructions))):
        inst = instructions[i]
        if inst.opcode == AECOpcode.SHL and inst.src1 and inst.src1.name == b_mad.dst.name:
            b_shl_i = i; break
    if a_shl_i < 0 or b_shl_i < 0:
        return instructions

    # —— locate the ADD after each SHL ——
    a_add_i = -1
    for i in range(a_shl_i + 1, min(a_shl_i + 3, len(instructions))):
        if instructions[i].opcode == AECOpcode.ADD:
            a_add_i = i; break
    b_add_i = -1
    for i in range(b_shl_i + 1, min(b_shl_i + 3, len(instructions))):
        if instructions[i].opcode == AECOpcode.ADD:
            b_add_i = i; break
    if a_add_i < 0 or b_add_i < 0:
        return instructions

    a_base = instructions[a_add_i].src1
    b_base = instructions[b_add_i].src1
    if a_base is None or b_base is None:
        return instructions

    # —— locate LD, MUL, ADD.f32, k++, BR (may be reordered by scheduler) ——
    ld_a_i = ld_b_i = mul_i = add_f32_i = k_add_i = br_i = -1
    for i in [j for j in loop_indices if j > b_add_i]:
        inst = instructions[i]
        if inst.opcode == AECOpcode.LD and inst.mem_space == MemSpace.GMEM:
            if ld_a_i < 0:
                ld_a_i = i
            elif ld_b_i < 0:
                ld_b_i = i
        elif inst.opcode == AECOpcode.MUL and inst.aec_type == AECType.F32 and ld_b_i >= 0:
            mul_i = i
        elif inst.opcode == AECOpcode.ADD and inst.aec_type == AECType.F32 and mul_i >= 0:
            add_f32_i = i
        elif inst.opcode == AECOpcode.ADD and inst.aec_type == AECType.U32 and i > b_add_i:
            if inst.src1 and inst.src1.name == k_v.name:
                k_add_i = i
                one_val = instructions[k_add_i].src2
    for i in [j for j in loop_indices if j > b_add_i]:
        if instructions[i].opcode == AECOpcode.BR:
            br_i = i
            break
    if any(x < 0 for x in (ld_a_i, ld_b_i, mul_i, add_f32_i, k_add_i, br_i)):
        return instructions

    acc_var = instructions[add_f32_i].dst
    k_val   = instructions[k_add_i].dst or k_v
    if not (acc_var and k_val):
        return instructions

    # —— locate CMPP and BRX ——
    cmp_i = brx_i = -1
    for i in loop_indices:
        if instructions[i].opcode == AECOpcode.CMPP:
            cmp_i = i
        if instructions[i].opcode == AECOpcode.BRX and cmp_i >= 0:
            brx_i = i; break
    if cmp_i < 0 or brx_i < 0:
        return instructions

    blk = header

    pre_block = "entry"
    for inst in instructions:
        if inst.block == header:
            break
        if inst.block:
            pre_block = inst.block

    # —— pre-loop: compute initial pointers and strides ——
    zero_s = _tmp("u32"); t2_s = _tmp("u32")
    s_a    = _tmp("u32"); s_b = _tmp("u32")
    a_ref  = _tmp("u32"); b_ref = _tmp("u32")
    a_ptr  = SSAValue(name="__gm_a__lo", ptx_type="u32", aec_type=AECType.U32)
    b_ptr  = SSAValue(name="__gm_b__lo", ptx_type="u32", aec_type=AECType.U32)

    pre: List[IRInst] = []
    pre.append(IRInst(opcode=AECOpcode.LOADI, aec_type=AECType.NONE, dst=zero_s, immediate=0, block=pre_block))
    pre.append(IRInst(opcode=AECOpcode.LOADI, aec_type=AECType.NONE, dst=t2_s,   immediate=2, block=pre_block))
    pre.append(IRInst(opcode=AECOpcode.LOADI, aec_type=AECType.NONE, dst=s_a,    immediate=4, block=pre_block))
    pre.append(IRInst(opcode=AECOpcode.SHL,   aec_type=AECType.U32, dst=s_b,    src1=N_v,   src2=t2_s,  block=pre_block))     # N*4
    pre.append(IRInst(opcode=AECOpcode.MAD,   aec_type=AECType.U32, dst=a_ref,  src1=row_v, src2=K_v, src3=zero_s, block=pre_block))   # row*K+0
    pre.append(IRInst(opcode=AECOpcode.SHL,   aec_type=AECType.U32, dst=a_ptr,  src1=a_ref, src2=t2_s,  block=pre_block))     # *4
    pre.append(IRInst(opcode=AECOpcode.ADD,   aec_type=AECType.U32, dst=a_ptr,  src1=a_base,src2=a_ptr, block=pre_block))     # &A[row*K]
    pre.append(IRInst(opcode=AECOpcode.MAD,   aec_type=AECType.U32, dst=b_ref,  src1=zero_s,src2=N_v, src3=col_v, block=pre_block))   # 0*N+col
    pre.append(IRInst(opcode=AECOpcode.SHL,   aec_type=AECType.U32, dst=b_ptr,  src1=b_ref, src2=t2_s,  block=pre_block))     # *4
    pre.append(IRInst(opcode=AECOpcode.ADD,   aec_type=AECType.U32, dst=b_ptr,  src1=b_base,src2=b_ptr, block=pre_block))     # &B[col]

    # —— optimized loop body ——
    ld_src  = instructions[ld_a_i].dst
    ld_src2 = instructions[ld_b_i].dst
    new_loop: List[IRInst] = []
    new_loop.append(instructions[cmp_i])                                              # CMPP  k, K
    new_loop.append(instructions[brx_i])                                              # BRX   -> STORE
    new_loop.append(IRInst(opcode=AECOpcode.LD,  aec_type=AECType.F32, mem_space=MemSpace.GMEM, dst=ld_src, src1=a_ptr, block=blk))
    new_loop.append(IRInst(opcode=AECOpcode.LD,  aec_type=AECType.F32, mem_space=MemSpace.GMEM, dst=ld_src2,src1=b_ptr, block=blk))
    new_loop.append(IRInst(opcode=AECOpcode.MAD, aec_type=AECType.F32, dst=acc_var, src1=ld_src, src2=ld_src2, src3=acc_var, block=blk))
    new_loop.append(IRInst(opcode=AECOpcode.ADD, aec_type=AECType.U32, dst=a_ptr,  src1=a_ptr,  src2=s_a,      block=blk))   # A_ptr += 4
    new_loop.append(IRInst(opcode=AECOpcode.ADD, aec_type=AECType.U32, dst=b_ptr,  src1=b_ptr,  src2=s_b,      block=blk))   # B_ptr += N*4
    new_loop.append(IRInst(opcode=AECOpcode.ADD, aec_type=AECType.U32, dst=k_val,  src1=k_val,  src2=one_val,  block=blk))   # k++
    new_loop.append(instructions[br_i])                                               # BR LOOP

    # —— assemble final result ——
    to_remove = set(range(a_mad_i, b_add_i + 1)) | {mul_i, add_f32_i, k_add_i, br_i, ld_a_i, ld_b_i, cmp_i, brx_i}
    result: List[IRInst] = []
    first_remove = min(to_remove) if to_remove else -1
    inserted = False
    for i, inst in enumerate(instructions):
        if i in to_remove:
            if not inserted:
                result.extend(pre)
                result.extend(new_loop)
                inserted = True
            continue
        result.append(inst)
    if not inserted:
        result.extend(pre)
        result.extend(new_loop)
    return result


def _gemm_apply_strength(instructions: List[IRInst]) -> List[IRInst]:
    return instructions


# ── Peephole + block cleanup ────────────────────────────────────────────────

def peephole(instructions: List[IRInst], safe: bool = False) -> List[IRInst]:
    """Peephole optimizations on the IR level.

    Rules (in order):
    1.  Dead `LOADI Rh, 0` for __hi registers with no uses → remove.
    2.  Self-assign `CPY Rd, Rd` → remove.
    3.  `MUL Rd, Rs, LOADI(4)` → `LOADI Rt, 2; SHL Rd, Rs, Rt`.
    4.  Duplicate LOADI same-immediate → replace second with CPY.
    5.  CPY alias forward-substitution in arithmetic operands.

    When safe=True, only rules 1-3 run (safe for post-gemm_iv_opt).
    """
    used: Set[str] = set()
    for inst in instructions:
        for src in (inst.src1, inst.src2, inst.src3):
            if src and not src.is_const and not src.is_special:
                used.add(src.name)
                if src.is_u64:
                    used.add(src.low_key())

    result: List[IRInst] = []
    loadi_seen: Dict[int, SSAValue] = {}
    cpy_aliases: Dict[str, SSAValue] = {}

    for inst in instructions:
        # ── rule 1: dead zero hi ──
        if inst.opcode == AECOpcode.LOADI and inst.dst and inst.dst.name.endswith("__hi"):
            if inst.immediate == 0 and inst.dst.name not in used:
                cpy_aliases.pop(inst.dst.name, None)
                continue

        # ── rule 2: self-assign CPY ──
        if inst.opcode == AECOpcode.CPY and inst.dst and inst.src1 and inst.dst.name == inst.src1.name:
            cpy_aliases.pop(inst.dst.name, None)
            continue

        if not safe:
            # ── rule 5: forward-substitute CPY aliases ──
            if inst.opcode in (AECOpcode.ADD, AECOpcode.SUB, AECOpcode.MUL, AECOpcode.MAD,
                               AECOpcode.FMA, AECOpcode.SHL, AECOpcode.SHR,
                               AECOpcode.AND, AECOpcode.OR, AECOpcode.XOR,
                               AECOpcode.CMPP, AECOpcode.LD, AECOpcode.ST):
                new_inst = inst
                for attr in ("src1", "src2", "src3"):
                    v = getattr(inst, attr)
                    if v and v.name in cpy_aliases and attr != "addr_reg":
                        alias = cpy_aliases[v.name]
                        if alias.name != v.name:
                            new_inst = _copy_meta(new_inst, **{attr: alias})
                if new_inst is not inst:
                    inst = new_inst

        # ── rule 3: MUL ×4 → SHL ×2 ──
        if inst.opcode == AECOpcode.MUL and inst.src2 and inst.src2.name:
            prev = result[-6:] if len(result) >= 6 else result
            for p in reversed(prev):
                if p.opcode == AECOpcode.LOADI and p.dst and p.dst.name == inst.src2.name and p.immediate == 4:
                    shift = _tmp("u32")
                    result.append(_copy_meta(_loadi(shift, 2), block=inst.block))
                    result.append(IRInst(opcode=AECOpcode.SHL, aec_type=AECType.U32, dst=inst.dst, src1=inst.src1, src2=shift, block=inst.block))
                    break
            else:
                result.append(inst)
            continue

        if not safe:
            # ── rule 4: LOADI dedup ──
            if inst.opcode == AECOpcode.LOADI and inst.dst:
                imm = inst.immediate
                if imm in loadi_seen:
                    ae_type = inst.aec_type if inst.aec_type != AECType.NONE else AECType.U32
                    cpy = IRInst(opcode=AECOpcode.CPY, aec_type=ae_type, dst=inst.dst, src1=loadi_seen[imm], block=inst.block)
                    result.append(cpy)
                    cpy_aliases[inst.dst.name] = loadi_seen[imm]
                    continue
                loadi_seen[imm] = inst.dst
                result.append(inst)
                continue

            # ── record CPY for alias tracking ──
            if inst.opcode == AECOpcode.CPY and inst.dst and inst.src1 and not inst.src1.is_const and not inst.src1.is_special:
                if inst.src1.name in cpy_aliases:
                    cpy_aliases[inst.dst.name] = cpy_aliases[inst.src1.name]
                else:
                    cpy_aliases[inst.dst.name] = inst.src1

        # ── side effects clear aliases ──
        if not safe and inst.opcode == AECOpcode.ST:
            loadi_seen.clear()
            cpy_aliases.clear()

        result.append(inst)
    return result


def merge_blocks(instructions: List[IRInst]) -> List[IRInst]:
    """Merge fall-through blocks: if block A has exactly one successor B,
    and B has A as its only predecessor, merge B into A."""
    pred_count: Dict[str, int] = {}
    succ_of: Dict[str, List[str]] = defaultdict(list)
    insts_by_block: Dict[str, List[int]] = defaultdict(list)

    for i, inst in enumerate(instructions):
        blk = inst.block or "entry"
        insts_by_block[blk].append(i)
        if inst.is_branch and inst.branch_target:
            succ_of[blk].append(inst.branch_target)
            pred_count[inst.branch_target] = pred_count.get(inst.branch_target, 0) + 1

    for blk in list(succ_of):
        targets = succ_of[blk]
        if len(targets) != 1:
            continue
        target = targets[0]
        if target not in insts_by_block or pred_count.get(target, 0) != 1:
            continue
        if blk == target:
            continue
        has_cond = any(
            i in insts_by_block[blk] and instructions[i].opcode == AECOpcode.BRX
            for i in insts_by_block[blk]
        )
        if has_cond:
            continue

        for idx in insts_by_block[target]:
            instructions[idx].block = blk

    return instructions


def promote_shared_memory(instructions: List[IRInst]) -> List[IRInst]:
    return instructions


def run_opt_pipeline(instructions: List[IRInst], opt_level: int) -> List[IRInst]:
    if opt_level >= 1:
        instructions = const_prop(instructions)
        instructions = dce(instructions)
    if opt_level >= 2:
        instructions = cse(instructions)
        instructions = const_prop(instructions)
        instructions = dce(instructions)
        instructions = gmem_load_cse(instructions)
        instructions = load_hoist(instructions)
        instructions = addr_cse(instructions)
        instructions = licm(instructions)
        instructions = gemm_iv_opt(instructions)
        gemm_ok = any(
            inst.dst and inst.dst.name == "__gm_a__lo" for inst in instructions
        )
        instructions = peephole(instructions, safe=gemm_ok)
        instructions = dce(instructions)
        instructions = merge_blocks(instructions)
    if opt_level >= 3:
        pass  # reserved for future O3-only passes
    return instructions