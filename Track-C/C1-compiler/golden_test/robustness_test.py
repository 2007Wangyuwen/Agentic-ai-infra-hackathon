#!/usr/bin/env python3
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


def v_rename(ptx):
    for a, b in [(r"%r", "%rr"), (r"%f", "%ff"), (r"%p", "%pp"), (r"%rd", "%rdd")]:
        ptx = re.sub(a + r"(\d+)", b + r"\1", ptx)
    return ptx


def v_labels(ptx):
    return re.sub(r"(?<!\w)([A-Z][A-Z_0-9]*):", r"B_\1:", ptx)


def v_dead(ptx):
    dead = "\nDEAD_BLOCK:\n    mov.u32 %r99, 0;\n    add.u32 %r98, %r99, %r99;\n"
    lines = ptx.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == "{":
            lines.insert(i + 1, dead)
            break
    return "\n".join(lines)


def v_paramswap(ptx):
    m = re.search(r"\.visible \.entry (\w+)\((.+?)\)", ptx, re.DOTALL)
    if not m:
        return ptx
    params = [p.strip() for p in m.group(2).split(",")]
    if len(params) >= 2:
        params[0], params[-1] = params[-1], params[0]
    newp = ", ".join(params)
    return ptx[: m.start(2)] + newp + ptx[m.end(2):]


VARIANTS = [
    ("rename_regs", v_rename),
    ("shuffle_labels", v_labels),
    ("dead_code", v_dead),
    ("param_swap", v_paramswap),
]


def run_variant(vname, vfunc, orig):
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
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "crash", "error": str(e)[:200]}
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=None)
    args = parser.parse_args()

    cases = [args.case] if args.case else ALL_CASES
    passed, failed = 0, 0
    for case in cases:
        ptx_path = TESTCASES / case / "kernel.ptx"
        orig = ptx_path.read_text(encoding="utf-8")
        print(f"\n=== {case} ===")
        for vn, vf in VARIANTS:
            r = run_variant(vn, vf, orig)
            ok = r["status"] == "ok"
            passed += ok
            failed += not ok
            icon = "OK" if ok else "FAIL"
            detail = f"  {r.get('inst','?')} inst {r.get('regs','?')} regs" if ok else f"  {r.get('error','?')[:60]}"
            print(f"  [{icon}] {vn:<18} {detail}")

    print(f"\n{'='*40}")
    print(f"  Summary: {passed}/{passed + failed} variants passed")
    print(f"{'='*40}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
