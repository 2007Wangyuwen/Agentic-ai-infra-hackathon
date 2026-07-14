"""CModel output correctness checker.

Compares CModel-dumped output buffers against reference outputs with
elementwise tolerance (atol/rtol). Produces a JSON report.

Usage:
    python golden_test/checker.py                              # default paths
    python golden_test/checker.py --dump-dir ./dumps --ref-dir ./refs
"""
import json
import os
import struct
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def load_f32_bin(path: str) -> List[float]:
    with open(path, "rb") as f:
        data = f.read()
    n = len(data) // 4
    if n * 4 != len(data):
        raise ValueError(f"File size not multiple of 4: {path}")
    return list(struct.unpack("<" + "f" * n, data))


def compare_elementwise(
    actual: List[float],
    expected: List[float],
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> Dict:
    """Compare two float arrays elementwise with absolute + relative tolerance."""
    if len(actual) != len(expected):
        return {
            "pass": False,
            "reason": f"length mismatch: actual={len(actual)}, expected={len(expected)}",
            "max_err": float("inf"),
            "max_err_idx": -1,
            "failed": len(actual),
            "total": len(expected),
            "first_failures": [],
        }

    max_err = 0.0
    max_err_idx = -1
    failed = 0
    first_failures = []

    for i in range(len(expected)):
        ae = actual[i]
        ee = expected[i]

        # Handle NaN — both NaN is a match, one NaN is a fail
        if ae != ae and ee != ee:
            continue
        if ae != ae or ee != ee:
            err = float("inf")
        else:
            err = abs(ae - ee)
            threshold = atol + rtol * max(abs(ae), abs(ee))
            if err <= threshold:
                continue

        failed += 1
        if err > max_err:
            max_err = err
            max_err_idx = i
        if len(first_failures) < 10:
            first_failures.append({
                "index": i,
                "actual": ae,
                "expected": ee,
                "diff": err if err != float("inf") else "NaN",
            })

    return {
        "pass": failed == 0,
        "reason": "",
        "max_err": max_err if max_err != float("inf") else "NaN",
        "max_err_idx": max_err_idx,
        "failed": failed,
        "total": len(expected),
        "first_failures": first_failures,
    }


# --- Test case configuration ---

CASE_CONFIGS = {
    "T1_basic_lowering":      {"output_buf": "c",   "atol": 1e-6, "rtol": 1e-6},
    "T2_scalar_optimization":  {"output_buf": "out", "atol": 1e-5, "rtol": 1e-5},
    "T3_memory_reuse":         {"output_buf": "out", "atol": 1e-5, "rtol": 1e-5},
    "T4_register_scheduling":  {"output_buf": "out", "atol": 1e-5, "rtol": 1e-5},
    "T5_scalar_gemm":          {"output_buf": "C",   "atol": 1e-4, "rtol": 1e-4},
}

TESTCASES_ORDER = [
    "T1_basic_lowering",
    "T2_scalar_optimization",
    "T3_memory_reuse",
    "T4_register_scheduling",
    "T5_scalar_gemm",
]


def check_all(
    dump_dir: str,
    ref_dir: str,
    opt_level: str = "O2",
) -> Tuple[Dict, bool]:
    """Run all checks and return (report, all_pass)."""
    results = {}
    all_pass = True

    for name in TESTCASES_ORDER:
        cfg = CASE_CONFIGS[name]
        output_buf = cfg["output_buf"]

        dump_path = os.path.join(dump_dir, f"{name}_dump_{output_buf}.bin")
        ref_path = os.path.join(ref_dir, f"{name}_ref_{output_buf}.bin")

        if not os.path.isfile(dump_path):
            results[name] = {
                "pass": False,
                "reason": f"dump file not found: {dump_path}",
                "max_err": None,
                "failed": -1,
                "total": -1,
            }
            all_pass = False
            continue

        if not os.path.isfile(ref_path):
            results[name] = {
                "pass": False,
                "reason": f"reference file not found: {ref_path}",
                "max_err": None,
                "failed": -1,
                "total": -1,
            }
            all_pass = False
            continue

        try:
            actual = load_f32_bin(dump_path)
            expected = load_f32_bin(ref_path)
            cmp = compare_elementwise(actual, expected, cfg["atol"], cfg["rtol"])
            results[name] = cmp
            if not cmp["pass"]:
                all_pass = False
        except Exception as e:
            results[name] = {
                "pass": False,
                "reason": str(e),
                "max_err": None,
                "failed": -1,
                "total": -1,
            }
            all_pass = False

    # Build summary
    passed = sum(1 for r in results.values() if r["pass"])
    failed_cases = sum(1 for r in results.values() if not r["pass"])

    report = {
        "pass": all_pass,
        "total_cases": len(TESTCASES_ORDER),
        "passed_cases": passed,
        "failed_cases": failed_cases,
        "opt_level": opt_level,
        "cases": results,
    }

    return report, all_pass


def main():
    parser = argparse.ArgumentParser(description="Check CModel output correctness")
    parser.add_argument(
        "--dump-dir", default="/tmp/c1_dumps",
        help="Directory containing dumped output buffers from CModel",
    )
    parser.add_argument(
        "--ref-dir", default="/tmp/c1_ref_outputs",
        help="Directory containing reference output .bin files",
    )
    parser.add_argument(
        "--opt", default="O2",
        help="Optimization level label for report (default: O2)",
    )
    parser.add_argument(
        "--report", default=None,
        help="Path to write JSON report (default: stdout only)",
    )
    args = parser.parse_args()

    report, all_pass = check_all(args.dump_dir, args.ref_dir, args.opt)

    # Print summary to stdout
    for name in TESTCASES_ORDER:
        r = report["cases"].get(name, {})
        status = "PASS" if r.get("pass") else "FAIL"
        reason = r.get("reason", "")
        failed = r.get("failed", "?")
        total = r.get("total", "?")
        max_err = r.get("max_err", "?")
        if isinstance(max_err, float):
            max_err_str = f"{max_err:.6e}"
        else:
            max_err_str = str(max_err)

        if r.get("pass"):
            print(f"  [{status}] {name}: {total} elements OK")
        elif reason:
            print(f"  [{status}] {name}: {reason}")
        else:
            print(f"  [{status}] {name}: {failed}/{total} mismatches, max_err={max_err_str}")

    print(f"\nSummary: {report['passed_cases']}/{report['total_cases']} cases passed")

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.report}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
