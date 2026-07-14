#!/usr/bin/env python3
"""Generate CModel inputs for T1-T5 public cases."""
import random, struct
from pathlib import Path

def f32(x): return struct.unpack("<f", struct.pack("<f", x))[0]
def pack_f32(values): return struct.pack("<" + "f" * len(values), *values)
def rand_f32(numel, seed):
    rng = random.Random(seed)
    return [f32(rng.uniform(-1.0, 1.0)) for _ in range(numel)]
def zero_f32(numel): return [0.0] * numel
def u32(value): return struct.pack("<I", value & 0xffffffff)
def u64(value): return struct.pack("<Q", value & 0xffffffffffffffff)

def write_case(case, buffers, pmem_entries):
    out = Path("/tmp/c1_inputs") / case
    out.mkdir(parents=True, exist_ok=True)
    for name, values in buffers.items():
        (out / f"input_{name}.bin").write_bytes(pack_f32(values))
    pmem = bytearray()
    for offset, data in pmem_entries:
        needed = offset + len(data)
        if len(pmem) < needed: pmem.extend(b"\x00" * (needed - len(pmem)))
        pmem[offset:offset+len(data)] = data
    size = (len(pmem) + 7) & ~7
    if len(pmem) < size: pmem.extend(b"\x00" * (size - len(pmem)))
    (out / "pmem.bin").write_bytes(pmem)

write_case("T1_basic_lowering",
    {"a": rand_f32(1048576, 1), "b": rand_f32(1048576, 2), "c": zero_f32(1048576)},
    [(0, u64(256)), (8, u64(4194560)), (16, u64(8388864)), (24, u32(1048576))])
write_case("T2_scalar_optimization",
    {"x": rand_f32(524288, 3), "y": rand_f32(524288, 4), "out": zero_f32(524288)},
    [(0, u64(256)), (8, u64(2097408)), (16, u64(4194560)), (24, u32(524288))])
write_case("T3_memory_reuse",
    {"x": rand_f32(524288, 5), "y": rand_f32(524288, 6), "z": rand_f32(524288, 7), "out": zero_f32(524288)},
    [(0, u64(256)), (8, u64(2097408)), (16, u64(4194560)), (24, u64(6291712)), (32, u32(524288))])
write_case("T4_register_scheduling",
    {"a": rand_f32(524288, 8), "b": rand_f32(524288, 9), "c": rand_f32(524288, 10), "d": rand_f32(524288, 11), "out": zero_f32(524288)},
    [(0, u64(256)), (8, u64(2097408)), (16, u64(4194560)), (24, u64(6291712)), (32, u64(8388864)), (40, u32(524288))])
write_case("T5_scalar_gemm",
    {"A": rand_f32(16384, 12), "B": rand_f32(16384, 13), "C": zero_f32(16384)},
    [(0, u64(256)), (8, u64(65792)), (16, u64(131328)), (24, u32(128)), (28, u32(128)), (32, u32(128))])
print("Inputs generated: T1-T5")
