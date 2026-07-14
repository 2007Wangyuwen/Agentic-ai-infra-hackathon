#!/usr/bin/env python3
"""Golden Test Orchestrator — cross-platform entry point.

On Linux/macOS: runs the full pipeline (compile → CModel → verify).
On Windows:     generates inputs + references, prints a ready-to-copy
                portable test bundle for transfer to a Linux/macOS machine.

Usage:
    # Full pipeline (Linux/macOS):
    python golden_test/run_all.py

    # Windows — generate portable bundle for Linux/macOS:
    python golden_test/run_all.py --bundle ./c1_test_bundle

    # Custom options:
    python golden_test/run_all.py -O O0,O2,O3
    python golden_test/run_all.py --cmodel ./path/to/aec-precise
    python golden_test/run_all.py --compile-only
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import platform
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
TESTCASES = PROJECT / "testcases"

ALL_CASES = [
    "T1_basic_lowering",
    "T2_scalar_optimization",
    "T3_memory_reuse",
    "T4_register_scheduling",
    "T5_scalar_gemm",
]

IS_WINDOWS = platform.system() == "Windows"


def compile_cases(testcases_dir, build_dir, opt_levels):
    """Compile all test cases at specified optimization levels."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "aec_cc", str(PROJECT / "compiler" / "aec-cc")
    )
    aec_cc = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(PROJECT / "compiler"))
    spec.loader.exec_module(aec_cc)

    compiled = {}
    for case_name in ALL_CASES:
        ptx_path = testcases_dir / case_name / "kernel.ptx"
        if not ptx_path.exists():
            print(f"  [WARN] PTX not found: {ptx_path} — skipping", file=sys.stderr)
            continue
        for level in opt_levels:
            label = f"{case_name}_{level}"
            out_path = build_dir / f"{case_name}_{level}.aecbin"
            print(f"  Compiling {case_name} @ {level} ...")
            try:
                aec_cc.compile_ptx(str(ptx_path), str(out_path), level)
                compiled[label] = str(out_path)
                print(f"    OK: {out_path}")
            except Exception as e:
                print(f"    FAIL: {e}", file=sys.stderr)
    return compiled


def create_portable_bundle(out_dir, opt_levels):
    """Create a portable test bundle for transfer to a Linux/macOS machine.

    The bundle contains:
      - compiled .aecbin files (built on current machine)
      - generated inputs from input_gen.py
      - reference outputs from ref_compute.py
      - run_all.sh script for CModel execution
      - checker.py for verification
    """
    bundle = Path(out_dir)
    bundle.mkdir(parents=True, exist_ok=True)

    bundle_inputs = bundle / "inputs"
    bundle_build = bundle / "build"
    bundle_ref = bundle / "ref_outputs"
    bundle_scripts = bundle / "scripts"

    for d in [bundle_inputs, bundle_build, bundle_ref, bundle_scripts]:
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: compile
    print("[bundle] Step 1: Compiling ...")
    compiled = compile_cases(TESTCASES, bundle_build, opt_levels)

    # Step 2: generate inputs
    print("[bundle] Step 2: Generating inputs ...")
    from input_gen import generate_all as gen_inputs
    gen_inputs(str(TESTCASES), str(bundle_inputs))

    # Step 3: compute reference outputs
    print("[bundle] Step 3: Computing reference outputs ...")
    from ref_compute import compute_all as compute_refs
    compute_refs(str(TESTCASES), str(bundle_inputs), str(bundle_ref))

    # Step 4: copy scripts
    print("[bundle] Step 4: Copying scripts ...")
    shutil.copy2(HERE / "run_all.sh", bundle_scripts / "run_all.sh")
    shutil.copy2(HERE / "checker.py", bundle_scripts / "checker.py")
    shutil.copy2(HERE / "input_gen.py", bundle_scripts / "input_gen.py")
    shutil.copy2(HERE / "ref_compute.py", bundle_scripts / "ref_compute.py")
    (bundle_scripts / "__init__.py").write_text("# bundle scripts\n")

    # Identify compiled files per case
    t1 = [f for f in compiled if "T1_" in f]
    t2 = [f for f in compiled if "T2_" in f]
    t3 = [f for f in compiled if "T3_" in f]
    t4 = [f for f in compiled if "T4_" in f]
    t5 = [f for f in compiled if "T5_" in f]

    config = {
        "description": "C1 AEC Compiler — Portable Golden Test Bundle",
        "instructions": (
            "Copy this directory to a Linux or macOS machine with aec-precise CModel.\n"
            "Then run:\n"
            "  cd scripts && python3 checker.py --dump-dir ../dumps --ref-dir ../ref_outputs\n"
            "Or use the full pipeline:\n"
            "  bash scripts/run_all.sh --cmodel /path/to/aec-precise -O O2\n"
        ),
        "cases": {
            "T1_basic_lowering": {
                "build": t1,
                "inputs": str(bundle_inputs / "T1_basic_lowering"),
                "ref": str(bundle_ref / "T1_basic_lowering_ref_c.bin"),
            },
            "T2_scalar_optimization": {
                "build": t2,
                "inputs": str(bundle_inputs / "T2_scalar_optimization"),
                "ref": str(bundle_ref / "T2_scalar_optimization_ref_out.bin"),
            },
            "T3_memory_reuse": {
                "build": t3,
                "inputs": str(bundle_inputs / "T3_memory_reuse"),
                "ref": str(bundle_ref / "T3_memory_reuse_ref_out.bin"),
            },
            "T4_register_scheduling": {
                "build": t4,
                "inputs": str(bundle_inputs / "T4_register_scheduling"),
                "ref": str(bundle_ref / "T4_register_scheduling_ref_out.bin"),
            },
            "T5_scalar_gemm": {
                "build": t5,
                "inputs": str(bundle_inputs / "T5_scalar_gemm"),
                "ref": str(bundle_ref / "T5_scalar_gemm_ref_C.bin"),
            },
        },
    }

    with open(bundle / "bundle_config.json", "w") as f:
        json.dump(config, f, indent=2)

    readme_lines = [
        "C1 Golden Test Bundle",
        "=====================",
        "",
        f"Generated on {platform.system()} {platform.machine()}",
        "",
        "Contents:",
        f"  build/          — Compiled .aecbin files ({','.join(opt_levels)})",
        "  inputs/         — Deterministic test inputs (pmem.bin + input_*.bin)",
        "  ref_outputs/    — Reference (golden) output buffers",
        "  scripts/        — CModel runner + checker scripts",
        "",
        "Usage on Linux/macOS:",
        "  1. Open a terminal in this directory",
        "  2. Run CModel against your compiled binaries:",
        "       bash scripts/run_all.sh --cmodel /path/to/aec-precise -O O2",
        "  3. Or use pre-compiled binaries from build/:",
        "       bash scripts/run_all.sh --compile-only",
        "",
        "  4. Verify against reference outputs:",
        "       cd scripts",
        "       python3 checker.py --dump-dir ../dumps --ref-dir ../ref_outputs",
        "",
        f"Bundle created for opt levels: {','.join(opt_levels)}",
    ]
    (bundle / "README.md").write_text("\n".join(readme_lines))

    print(f"\n[bundle] Ready → {bundle}")
    print(f"  {len(compiled)} binaries compiled")
    print(f"  Copy this directory to a Linux/macOS machine and run scripts/run_all.sh")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-platform golden test orchestrator for AEC compiler"
    )
    parser.add_argument(
        "-O", default="O2",
        help="Comma-separated optimization levels (default: O2)",
    )
    parser.add_argument(
        "--cmodel", default="./simple-gpgpu/cmodel/precise/aec-precise",
        help="Path to aec-precise CModel binary",
    )
    parser.add_argument(
        "--bundle", default=None,
        help="Create a portable test bundle at this path (for Windows users)",
    )
    parser.add_argument(
        "--compile-only", action="store_true",
        help="Compile only, skip CModel execution",
    )
    parser.add_argument(
        "--inputs-only", action="store_true",
        help="Generate inputs and references only, skip compilation",
    )
    parser.add_argument(
        "--outdir", default="/tmp/c1_rundir",
        help="Output directory for the full pipeline",
    )
    args = parser.parse_args()

    opt_levels = [lvl.strip() for lvl in args.O.split(",") if lvl.strip()]

    # Mode: inputs-only
    if args.inputs_only:
        print("[run_all.py] Inputs-only mode")
        from input_gen import generate_all as gen_inputs
        indir = args.outdir if args.bundle is None else str(Path(args.bundle) / "inputs")
        gen_inputs(str(TESTCASES), indir)
        from ref_compute import compute_all as compute_refs
        refdir = args.outdir + "/ref_outputs" if args.bundle is None else str(Path(args.bundle) / "ref_outputs")
        compute_refs(str(TESTCASES), indir, refdir)
        print(f"[run_all.py] Inputs → {indir}")
        print(f"[run_all.py] References → {refdir}")
        return

    # Mode: bundle (cross-platform portable)
    if args.bundle:
        create_portable_bundle(args.bundle, opt_levels)
        return

    # Mode: Windows — no CModel available
    if IS_WINDOWS:
        print("[run_all.py] Windows detected — CModel cannot run locally.")
        print("[run_all.py] Use --bundle to create a portable test bundle:")
        print(f"  python golden_test/run_all.py --bundle ./c1_test_bundle -O {','.join(opt_levels)}")
        print("[run_all.py] Then copy c1_test_bundle/ to a Linux/macOS machine.")
        sys.exit(1)

    # Mode: Linux/macOS — delegate to shell script
    script = HERE / "run_all.sh"
    cmd = [
        "bash", str(script),
        "-O", ",".join(opt_levels),
        "--cmodel", args.cmodel,
    ]
    print(f"[run_all.py] Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
