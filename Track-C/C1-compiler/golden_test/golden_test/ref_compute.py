"""Golden reference output computer.

Computes expected output buffers for all 5 test cases using the formulas
in manifest.json. Serves as the ground truth for correctness verification.

T1-T4: elementwise formula evaluation in pure Python.
T5: matrix multiplication C = A @ B.

Usage:
    python golden_test/ref_compute.py                        # read inputs from /tmp/c1_inputs/
    python golden_test/ref_compute.py --indir ./test_data    # read from custom dir
    python golden_test/ref_compute.py --outdir ./ref_outputs # write to custom dir
"""
import json
import os
import struct
import sys
import argparse
from pathlib import Path

# Import shared utilities from input_gen
from input_gen import (
    load_manifest, compute_gmem_layout, rand_f32, zero_f32,
    pack_f32, unpack_f32, get_numel, TESTCASES,
)


def compute_elementwise(
    formula: str,
    output_name: str,
    buffers: dict,
    numel: int,
) -> list:
    """Evaluate an elementwise formula against loaded buffer data.

    Supported patterns from public cases:
      - c[i] = a[i] + b[i]                            (T1)
      - out[i] = (x[i] + y[i]) * (x[i] + y[i]) + x[i] (T2)
      - out[i] = x[i] * y[i] + x[i] * z[i]            (T3)
      - out[i] = (a[i]+b[i])*(c[i]-d[i]) + (a[i]*c[i])*(b[i]+d[i]) (T4)

    Formula format: "output[i] = expression" — only the RHS expression is evaluated.
    Uses compile() for performance — O(1) per element vs O(parse) with eval().
    """
    # Split formula into output[i] = expression
    if " = " in formula:
        rhs = formula.split(" = ", 1)[1].strip()
    else:
        rhs = formula.strip()

    # Build data arrays as dict of lists
    ns = {name: data for name, data in buffers.items()}

    # Replace buf_name[i] with ns['buf_name'][i]
    buf_names = sorted(ns.keys(), key=len, reverse=True)
    code = rhs
    for name in buf_names:
        code = code.replace(f"{name}[i]", f"ns['{name}'][i]")

    # Compile once, then evaluate per element
    try:
        compiled = compile(code, "formula", "eval")
    except Exception as e:
        print(f"[ref] ERROR compiling formula: {formula}", file=sys.stderr)
        print(f"[ref] Error: {e}", file=sys.stderr)
        raise

    result = [0.0] * numel
    for i in range(numel):
        result[i] = eval(compiled, {"ns": ns, "i": i})
    return result


def compute_matmul(
    formula: str,
    output_name: str,
    buffers: dict,
    shape: list,
) -> list:
    """Compute matrix multiplication C = A @ B.

    Assumes row-major layout. shape = [M, N] for output.
    A: M×K, B: K×N, C: M×N — all implicit from shape and the fact that A cols = B rows.
    """
    a_name, b_name, c_name = None, None, None
    if "A @ B" in formula:
        a_name = "A"
        b_name = "B"
        c_name = "C"
    elif "a @ b" in formula:
        a_name = "a"
        b_name = "b"
        c_name = "c"

    if not a_name:
        raise ValueError(f"Unsupported matmul formula: {formula}")

    M, N = shape[0], shape[1]
    # K is inferred: len(buffer) / N for B (B is K×N), or len(buffer) / M for A (A is M×K)
    # Since both A and B have the same numel (16384 for 128×128), K = N = M
    # General inference: K = len(A) / M (since A is M×K row-major)
    K = len(buffers[a_name]) // M

    A = buffers[a_name]
    B = buffers[b_name]

    result = [0.0] * (M * N)
    for m in range(M):
        for n in range(N):
            acc = 0.0
            for k in range(K):
                acc += A[m * K + k] * B[k * N + n]
            result[m * N + n] = acc

    return result


def compute_reference_for_case(
    manifest: dict,
    buf_data: dict,
    case_name: str,
    outdir: str,
) -> str:
    """Compute reference output for one case, write to file, return path."""
    check = manifest["check"]
    check_type = check["type"]
    output_name = check["output"]
    formula = check.get("formula", "")
    atol = check.get("atol", 1e-6)
    rtol = check.get("rtol", 1e-6)

    output_buf = manifest["buffers"][output_name]
    numel = get_numel(output_buf)

    if check_type == "elementwise":
        ref_vals = compute_elementwise(formula, output_name, buf_data, numel)
    elif check_type == "matmul":
        shape = output_buf.get("shape", [numel, 1])
        ref_vals = compute_matmul(formula, output_name, buf_data, shape)
    else:
        raise ValueError(f"Unknown check type: {check_type}")

    out_path = os.path.join(outdir, f"{case_name}_ref_{output_name}.bin")
    with open(out_path, "wb") as f:
        f.write(pack_f32(ref_vals))

    print(f"[ref] {case_name}: {check_type} '{formula}' → {len(ref_vals)} elems, atol={atol}, rtol={rtol}")
    return out_path


def compute_all(
    testcases_dir: str,
    input_data_dir: str,
    ref_outdir: str,
) -> dict:
    """Compute reference outputs for all test cases."""
    results = {}
    Path(ref_outdir).mkdir(parents=True, exist_ok=True)

    for name in TESTCASES:
        case_dir = os.path.join(testcases_dir, name)
        manifest = load_manifest(case_dir)

        # Build buffer data using the same deterministic seeds
        buffers_def = manifest["buffers"]
        buf_data = {}
        for buf_name, buf in buffers_def.items():
            numel = get_numel(buf)
            if buf["init"] == "rand_uniform":
                buf_data[buf_name] = rand_f32(numel, buf["seed"])
            elif buf["init"] == "zero":
                buf_data[buf_name] = zero_f32(numel)

        ref_path = compute_reference_for_case(manifest, buf_data, name, ref_outdir)
        results[name] = {
            "ref_path": ref_path,
            "output_buffer": manifest["check"]["output"],
            "atol": manifest["check"]["atol"],
            "rtol": manifest["check"]["rtol"],
            "numel": get_numel(manifest["buffers"][manifest["check"]["output"]]),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute golden reference outputs")
    parser.add_argument(
        "--testcases", default=None,
        help="Path to testcases directory",
    )
    parser.add_argument(
        "--indir", default="/tmp/c1_inputs",
        help="Directory containing generated input buffers (used for layout only)",
    )
    parser.add_argument(
        "--outdir", default="/tmp/c1_ref_outputs",
        help="Directory to write reference output .bin files",
    )
    args = parser.parse_args()

    if args.testcases is None:
        args.testcases = str(Path(__file__).resolve().parent.parent / "testcases")

    if not os.path.isdir(args.testcases):
        print(f"ERROR: testcases directory not found: {args.testcases}", file=sys.stderr)
        sys.exit(1)

    results = compute_all(args.testcases, args.indir, args.outdir)
    print(f"\n[ref] Done. {len(results)} reference outputs → {args.outdir}")


if __name__ == "__main__":
    main()
