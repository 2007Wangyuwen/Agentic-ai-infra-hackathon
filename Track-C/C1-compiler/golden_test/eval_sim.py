#!/usr/bin/env python3
"""C1 competition evaluation simulator — exact pipeline per scoring.md."""
import subprocess, json, os, sys, struct, re, tempfile, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
COMPILER = PROJECT / "compiler" / "aec-cc"
CMODEL = PROJECT / "GoldenModel" / "bin" / "aec-precise-linux-x86_64"
TESTCASES = PROJECT / "testcases"

CASES = [
    ("T1_basic_lowering",      "4096,1,1", "256,1,1",  12591104, 8388864,  4194304, "c",
     [("pmem", 0, "pmem.bin"), ("gmem", 256, "input_a.bin"), ("gmem", 4194560, "input_b.bin"), ("gmem", 8388864, "input_c.bin")]),
    ("T2_scalar_optimization",  "2048,1,1", "256,1,1",   6299648, 4194560,  2097152, "out",
     [("pmem", 0, "pmem.bin"), ("gmem", 256, "input_x.bin"), ("gmem", 2097408, "input_y.bin"), ("gmem", 4194560, "input_out.bin")]),
    ("T3_memory_reuse",         "2048,1,1", "256,1,1",   8396800, 6291712,  2097152, "out",
     [("pmem", 0, "pmem.bin"), ("gmem", 256, "input_x.bin"), ("gmem", 2097408, "input_y.bin"), ("gmem", 4194560, "input_z.bin"), ("gmem", 6291712, "input_out.bin")]),
    ("T4_register_scheduling",  "2048,1,1", "256,1,1",  10493952, 8388864,  2097152, "out",
     [("pmem", 0, "pmem.bin"), ("gmem", 256, "input_a.bin"), ("gmem", 2097408, "input_b.bin"), ("gmem", 4194560, "input_c.bin"), ("gmem", 6291712, "input_d.bin"), ("gmem", 8388864, "input_out.bin")]),
    ("T5_scalar_gemm",          "8,8,1",    "16,16,1",    204800, 131328,    65536, "C",
     [("pmem", 0, "pmem.bin"), ("gmem", 256, "input_A.bin"), ("gmem", 65792, "input_B.bin"), ("gmem", 131328, "input_C.bin")]),
]

def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120, **kw)

# === Step 1: Generate inputs ===
print("=" * 60)
print("  Phase 1: 生成输入数据")
print("=" * 60)
sys.path.insert(0, str(HERE))
from input_gen import generate_all
outdir = Path("/tmp/c1_eval")
outdir.mkdir(parents=True, exist_ok=True)
(tmp := outdir / "inputs").mkdir(exist_ok=True)
(tmp2 := outdir / "build").mkdir(exist_ok=True)
(tmp3 := outdir / "dumps").mkdir(exist_ok=True)
(tmp4 := outdir / "ref").mkdir(exist_ok=True)
(tmp5 := outdir / "reports").mkdir(exist_ok=True)
IN = outdir / "inputs"
BLD = outdir / "build"
DMP = outdir / "dumps"
REF = outdir / "ref"
RPT = outdir / "reports"
generate_all(str(TESTCASES), str(IN))

# === Step 2: Compile (exactly as evaluator runs) ===
print("\n" + "=" * 60)
print("  Phase 2: 编译 (aec-cc {ptx} -O2 -o {output} --report {report})")
print("=" * 60)

reports = {}
for cfg in CASES:
    case, grid, block, gmem, daddr, dsize, outbuf, loads = cfg
    ptx = str(TESTCASES / case / "kernel.ptx")
    out = str(BLD / f"{case}_O2.aecbin")
    rpt = str(RPT / f"{case}_report.json")
    r = run([sys.executable, str(COMPILER), ptx, "-O2", "-o", out, "--report", rpt])
    if r.returncode != 0:
        print(f"  FAIL {case}: {r.stderr[:100]}")
        reports[case] = {"status": "compile_fail"}
        continue
    rep = json.loads(open(rpt).read())
    reports[case] = rep
    print(f"  {case}: PTX={rep['num_ptx_instructions']} -> AEC={rep['num_aec_instructions']}  GPRs={rep['num_physical_registers']}  Preds={rep['num_predicates']}  Spills={rep['spills']}")

# === Step 3: CModel execution ===
print("\n" + "=" * 60)
print("  Phase 3: CModel 执行")
print("=" * 60)

results = {}
for cfg in CASES:
    case, grid, block, gmem, daddr, dsize, outbuf, loads = cfg
    binary = str(BLD / f"{case}_O2.aecbin")
    dump_file = str(DMP / f"{case}_dump_{outbuf}.bin")
    cmd = [
        str(CMODEL),
        "--program", binary,
        "--grid", grid,
        "--block", block,
        "--gmem-size", str(gmem),
        "--cmem-size", "65536",
        "--pmem-size", "65536",
        "--smem-size", "65536",
        "--lmem-size", "4096",
        "--max-steps", "50000000",
        "--dump", f"{daddr}:{dsize}:{dump_file}",
    ]
    for mem, addr, fname in loads:
        cmd.extend(["--load", f"{mem}:{addr}:{str(IN / case / fname)}"])
    r = run(cmd)
    output = r.stdout.strip() + r.stderr.strip()
    # Parse JSON status line
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                js = json.loads(line)
                status = js.get("status", "?")
                steps = js.get("steps", 0)
                err = js.get("error", "")
                results[case] = {"status": status, "steps": steps, "error": err}
                print(f"  {case}: status={status}  steps={steps}" + (f"  error={err}" if err else ""))
                break
            except: pass
    else:
        results[case] = {"status": "fail", "steps": 0, "error": output[:100]}
        print(f"  {case}: FAIL (no JSON) {output[:80]}")

# === Step 4: Compute reference ===
print("\n" + "=" * 60)
print("  Phase 4: 标准答案计算")
print("=" * 60)
from ref_compute import compute_all as compute_refs
compute_refs(str(TESTCASES), str(IN), str(REF))

# === Step 5: Correctness check ===
print("\n" + "=" * 60)
print("  Phase 5: 正确性比对")
print("=" * 60)
from checker import check_all as check
check_report, all_pass = check(str(DMP), str(REF), "O2")

# === Final report ===
print("\n" + "=" * 60)
print("  评测结果汇总")
print("=" * 60)
print()
print(f"  {'Case':<30} {'AEC':>5} {'Steps':>10} {'Correctness':>12}")
print(f"  {'-'*30} {'-'*5} {'-'*10} {'-'*12}")
for cfg in CASES:
    case = cfg[0]
    aec = reports.get(case, {}).get("num_aec_instructions", "?")
    r = results.get(case, {})
    cc = check_report.get("cases", {}).get(case, {})
    c_ok = cc.get("pass", False)
    print(f"  {case:<30} {str(aec):>5} {r.get('steps',0):>10} {'PASS' if c_ok else 'FAIL':>12}")
passed = sum(1 for cfg in CASES if check_report["cases"].get(cfg[0], {}).get("pass"))
print(f"\n  正确性: {passed}/{len(CASES)} 通过")
notdone = sum(1 for cfg in CASES if results.get(cfg[0],{}).get("status")!="done")
if notdone:
    print(f"  异常: {notdone} case CModel 未正常结束")
print(f"\n  诊断指标:")
for cfg in CASES:
    case = cfg[0]
    rep = reports.get(case, {})
    if rep:
        sp = rep.get("spills", {})
        print(f"    {case:<30} ptx={rep.get('num_ptx_instructions','?'):>3} aes={rep.get('num_aec_instructions','?'):>3} gprs={rep.get('num_physical_registers','?'):>3} preds={rep.get('num_predicates','?'):>2} spill={sp.get('stores',0)}")
print()
print(f"  评测命令模板: python compiler/aec-cc {{ptx}} -O2 -o {{output}} --report {{report}}")
print()
print("=" * 60)
print(f"  状态: {'全部通过' if passed==5 else f'{passed}/5 通过'}")
print("=" * 60)
