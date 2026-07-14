#!/usr/bin/env python3
"""
Robustness stress test: generates PTX variants and verifies compilation.

Variant strategies:
  1. rename_regs   — swap register prefixes (%r→%rr, %f→%ff, %p→%pp, %rd→%rdd)
  2. shuffle_labels — prefix all block labels
  3. dead_code     — inject a dead basic block
  4. param_swap    — swap first and last .param declarations
  5. low_regs      — halve GPR declaration count to stress spill
  6. extra_branch  — insert a predicated branch block to stress CFG
  7. u64_tmp       — inject unused .u64 temporary moves to stress u64 lowering

Usage:
    python golden_test/robustness_test.py
    python golden_test/robustness_test.py --case T5_scalar_gemm
"""
import os, re, random, subprocess, tempfile, json, sys, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
TESTCASES = PROJECT / "testcases"
COMPILER = PROJECT / "compiler" / "aec-cc"

ALL_CASES = [
    "T1_basic_lowering", "T2_scalar_optimization",
    "T3_memory_reuse", "T4_register_scheduling", "T5_scalar_gemm",
]

# =========================================================================
# Variant generators
# =========================================================================

def v_rename(ptx):
    """Rename register prefixes: %r→%rr, %f→%ff, %p→%pp, %rd→%rdd."""
    for a, b in [(r"%r", "%rr"), (r"%f", "%ff"), (r"%p", "%pp"), (r"%rd", "%rdd")]:
        ptx = re.sub(a + r"(\d+)", b + r"\1", ptx)
    return ptx


def v_labels(ptx):
    """Prefix all block labels with B_."""
    return re.sub(r"(?<!\w)([A-Z][A-Z_0-9]*):", r"B_\1:", ptx)


def v_dead(ptx):
    """Insert a dead basic block after kernel body entry."""
    dead = "\nDEAD_BLOCK:\n    mov.u32 %r99, 0;\n    add.u32 %r98, %r99, %r99;\n"
    lines = ptx.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == "{":
            lines.insert(i + 1, dead)
            break
    return "\n".join(lines)


def v_paramswap(ptx):
    """Swap first and last .param in entry declaration."""
    m = re.search(r"\.visible \.entry (\w+)\((.+?)\)", ptx, re.DOTALL)
    if not m:
        return ptx
    params = [p.strip() for p in m.group(2).split(",")]
    if len(params) >= 2:
        params[0], params[-1] = params[-1], params[0]
    newp = ", ".join(params)
    return ptx[: m.start(2)] + newp + ptx[m.end(2):]


def v_low_regs(ptx):
    """Halve GPR declarations to stress register spill."""
    # .reg .u32 %r<16> → .reg .u32 %r<6>
    ptx = re.sub(r"(\.reg \.u32\s+%r)<\d+>", r"\1<6>", ptx)
    ptx = re.sub(r"(\.reg \.f32\s+%f)<\d+>", r"\1<6>", ptx)
    ptx = re.sub(r"(\.reg \.u64\s+%rd)<\d+>", r"\1<4>", ptx)
    ptx = re.sub(r"(\.reg \.pred\s+%p)<\d+>", r"\1<3>", ptx)
    return ptx


def v_extra_branch(ptx):
    """Insert a predicated-branch block before ret to stress CFG."""
    extra = (
        "\n    setp.lt.u32 %p0, %r0, 0;\n"
        "    @%p0 bra SKIP_BLK;\n"
        "EXTRA_BLOCK:\n"
        "    add.u32 %r0, %r0, 0;\n"
        "SKIP_BLK:\n"
    )
    # Insert before 'ret;' in the body (not inside comments)
    ptx = re.sub(r"(\n\s*ret\s*;)", extra + r"\1", ptx, count=1)
    return ptx


def v_u64_tmp(ptx):
    """Inject unused .u64 register moves to stress u64 lowering."""
    extra = (
        "\n    .reg .u64 %rd99;\n"
        "    mov.u64 %rd99, 0;\n"
    )
    # Insert after other .reg declarations, before first label
    lines = ptx.split("\n")
    insert_at = len(lines)
    for i, line in enumerate(lines):
        if line.strip() and not line.strip().startswith(".reg") and not line.strip().startswith("{"):
            insert_at = i - 1
            break
    lines[insert_at:insert_at] = extra.strip().split("\n")
    return "\n".join(lines)


# =========================================================================
# Variant catalog
# =========================================================================

VARIANTS = [
    ("rename_regs",   v_rename),
    ("shuffle_labels", v_labels),
    ("dead_code",     v_dead),
    ("param_swap",    v_paramswap),
    ("low_regs",      v_low_regs),
    ("extra_branch",  v_extra_branch),
    ("u64_tmp",       v_u64_tmp),
]


def run_variant(vname, vfunc, orig):
    """Compile one variant and return result dict."""
    try:
        vptx = vfunc(orig)
    except Exception as e:
        return {"status": "gen_fail", "error": str(e)[:100]}

    wd = Path(tempfile.mkdtemp(prefix="c1r_"))
    try:
        (wd / "kernel.ptx").write_text(vptx, encoding="utf-8")
        r = subprocess.run(
            [sys.executable, str(COMPILER), str(wd / "kernel.ptx"), "-O2",
             "-o", str(wd / "o.aecbin"), "--report", str(wd / "r.json")],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            return {"status": "compile_fail", "error": r.stderr[:200]}
        rep = json.loads((wd / "r.json").read_text())
        return {
            "status": rep.get("status", "?"),
            "inst": rep.get("num_aec_instructions", 0),
            "regs": rep.get("num_physical_registers", 0),
            "preds": rep.get("num_predicates", 0),
            "spills": rep.get("spills", {}),
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "crash", "error": str(e)[:200]}
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="C1 robustness stress test")
    parser.add_argument("--case", default=None, help="Test a single case (default: all)")
    parser.add_argument("--variant", default=None, help="Test a single variant (default: all)")
    args = parser.parse_args()

    cases = [args.case] if args.case else ALL_CASES
    if args.variant:
        variants = [(vn, vf) for vn, vf in VARIANTS if vn == args.variant]
        if not variants:
            print(f"Unknown variant: {args.variant}")
            return 1
    else:
        variants = VARIANTS

    passed, failed = 0, 0
    for case in cases:
        ptx_path = TESTCASES / case / "kernel.ptx"
        if not ptx_path.exists():
            print(f"\n[SKIP] {case}: PTX not found")
            continue
        orig = ptx_path.read_text(encoding="utf-8")
        print(f"\n=== {case} ===")
        for vn, vf in variants:
            r = run_variant(vn, vf, orig)
            ok = r["status"] == "ok"
            passed += ok
            failed += not ok
            icon = "OK" if ok else "FAIL"
            if ok:
                spills = r.get("spills", {})
                sp = f" spill={spills.get('stores',0)}" if spills.get("stores", 0) > 0 else ""
                detail = f"  {r['inst']} inst {r['regs']} regs{sp}"
            else:
                detail = f"  {r.get('error','?')[:60]}"
            print(f"  [{icon}] {vn:<18} {detail}")

    total = passed + failed
    print(f"\n{'='*50}")
    print(f"  Summary: {passed}/{total} variants passed ({failed} failed)")
    print(f"{'='*50}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
