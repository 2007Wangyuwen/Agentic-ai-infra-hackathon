"""Golden test input generator.

Reads testcase manifests and produces deterministic binary input files
(pmem.bin + input_<buffer>.bin) for all 5 public cases.

Usage:
    python golden_test/input_gen.py                        # write to /tmp/c1_inputs/
    python golden_test/input_gen.py --outdir ./test_data   # write to custom dir
"""
import json
import os
import random
import struct
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def f32_to_raw(x: float) -> bytes:
    return struct.pack("<f", x)


def pack_f32(values: List[float]) -> bytes:
    return struct.pack("<" + "f" * len(values), *values)


def unpack_f32(data: bytes) -> List[float]:
    n = len(data) // 4
    return list(struct.unpack("<" + "f" * n, data))


def rand_f32(numel: int, seed: int) -> List[float]:
    """Deterministic f32 uniform random in [-1.0, 1.0]."""
    rng = random.Random(seed)
    vals = []
    for _ in range(numel):
        x = rng.uniform(-1.0, 1.0)
        vals.append(struct.unpack("<f", struct.pack("<f", x))[0])
    return vals


def zero_f32(numel: int) -> List[float]:
    return [0.0] * numel


def u32(value: int) -> bytes:
    return struct.pack("<I", value & 0xFFFFFFFF)


def u64(value: int) -> bytes:
    return struct.pack("<Q", value & 0xFFFFFFFFFFFFFFFF)


# --- Manifest-driven address computation ---

def get_numel(buf: Dict) -> int:
    """Extract element count from buffer definition."""
    if "numel" in buf:
        return buf["numel"]
    if "shape" in buf:
        import math
        return math.prod(buf["shape"])
    raise KeyError("Buffer must have 'numel' or 'shape'")


def compute_gmem_layout(
    buffers: Dict,
    base_addr: int = 256,
    align: int = 256,
) -> Dict[str, Tuple[int, int]]:
    """Assign GMEM byte addresses to buffers in manifest order.

    Returns {buffer_name: (byte_addr, byte_size)}.
    """
    buf_names = list(buffers.keys())
    layout = {}
    addr = base_addr
    for name in buf_names:
        buf = buffers[name]
        numel = get_numel(buf)
        size = numel * 4  # all f32
        layout[name] = (addr, size)
        addr += size
        addr = ((addr + align - 1) // align) * align
    return layout


def build_pmem_entries(
    params: List[Dict],
    layout: Dict[str, Tuple[int, int]],
) -> List[Tuple[int, bytes]]:
    """Build (offset, bytes) entries for the parameter memory block."""
    entries = []
    offset = 0
    for p in params:
        if p["kind"] == "gmem_ptr":
            buf_name = p["buffer"]
            addr, _ = layout[buf_name]
            entries.append((offset, u64(addr)))
            offset += 8
        elif p["kind"] == "value":
            if p["type"] in ("u64", "b64"):
                entries.append((offset, u64(p["value"])))
                offset += 8
            elif p["type"] in ("u32", "s32", "b32", "f32"):
                entries.append((offset, u32(p["value"])))
                offset += 4
    return entries


def pack_pmem(entries: List[Tuple[int, bytes]]) -> bytes:
    max_offset = max(offset + len(data) for offset, data in entries) if entries else 0
    size = (max_offset + 7) & ~7  # 8-byte aligned
    pmem = bytearray(size)
    for offset, data in entries:
        pmem[offset : offset + len(data)] = data
    return bytes(pmem)


def load_manifest(case_dir: str) -> Dict:
    path = os.path.join(case_dir, "manifest.json")
    with open(path) as f:
        return json.load(f)


def generate_case(
    manifest: Dict,
    outdir: str,
    case_name: str,
) -> Dict:
    """Generate all input files for one test case.

    Returns info dict with layout and buffer data for downstream use.
    """
    out = Path(outdir) / case_name
    out.mkdir(parents=True, exist_ok=True)

    buffers_def = manifest["buffers"]
    params = manifest["params"]

    layout = compute_gmem_layout(buffers_def)

    # Generate buffer data
    buf_data = {}
    for name, buf in buffers_def.items():
        numel = get_numel(buf)
        if buf["init"] == "rand_uniform":
            vals = rand_f32(numel, buf["seed"])
        elif buf["init"] == "zero":
            vals = zero_f32(numel)
        else:
            raise ValueError(f"Unknown buffer init type: {buf['init']}")
        bin_path = out / f"input_{name}.bin"
        bin_path.write_bytes(pack_f32(vals))
        buf_data[name] = vals

    # Generate pmem.bin
    pmem_entries = build_pmem_entries(params, layout)
    pmem_path = out / "pmem.bin"
    pmem_path.write_bytes(pack_pmem(pmem_entries))

    print(f"[gen] {case_name}: {len(buffers_def)} buffers, {len(params)} params → {out}")

    return {
        "case": case_name,
        "layout": layout,
        "buf_data": buf_data,
        "params": pmem_entries,
        "outdir": str(out),
    }


# --- Test case catalog ---

TESTCASES = [
    "T1_basic_lowering",
    "T2_scalar_optimization",
    "T3_memory_reuse",
    "T4_register_scheduling",
    "T5_scalar_gemm",
]


def generate_all(testcases_dir: str, outdir: str) -> Dict[str, Dict]:
    results = {}
    for name in TESTCASES:
        case_dir = os.path.join(testcases_dir, name)
        manifest = load_manifest(case_dir)
        results[name] = generate_case(manifest, outdir, name)
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate golden test inputs from manifests")
    parser.add_argument(
        "--testcases", default=None,
        help="Path to testcases directory (default: ../testcases relative to this file)",
    )
    parser.add_argument(
        "--outdir", default="/tmp/c1_inputs",
        help="Output directory (default: /tmp/c1_inputs)",
    )
    args = parser.parse_args()

    if args.testcases is None:
        args.testcases = str(Path(__file__).resolve().parent.parent / "testcases")

    if not os.path.isdir(args.testcases):
        print(f"ERROR: testcases directory not found: {args.testcases}", file=sys.stderr)
        sys.exit(1)

    results = generate_all(args.testcases, args.outdir)
    print(f"\n[gen] Done. {len(results)} cases generated → {args.outdir}")


if __name__ == "__main__":
    main()
