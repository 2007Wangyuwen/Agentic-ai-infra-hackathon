#!/bin/bash
# =============================================================================
# Golden Test: Full Pipeline Runner (Linux/macOS)
#
# Compiles all 5 test cases with aec-cc, runs them through aec-precise CModel,
# and dumps output buffers for correctness verification.
#
# Usage:
#   ./golden_test/run_all.sh                                    # default: O2
#   ./golden_test/run_all.sh -O O0                              # O0 only
#   ./golden_test/run_all.sh -O O0,O2,O3                        # all levels
#   ./golden_test/run_all.sh --cmodel ./path/to/aec-precise     # custom CModel
#   ./golden_test/run_all.sh --compile-only                     # compile only, no CModel
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

AEC_CC="${PROJECT_DIR}/compiler/aec-cc"
CMODEL="./simple-gpgpu/cmodel/precise/aec-precise"
TESTCASES_DIR="${PROJECT_DIR}/testcases"
OUTDIR="/tmp/c1_rundir"
INPUTDIR="${OUTDIR}/inputs"
DUMPDIR="${OUTDIR}/dumps"
BUILDDIR="${OUTDIR}/build"
REPORTDIR="${OUTDIR}/reports"

OPT_LEVELS="O2"
COMPILE_ONLY=false

usage() {
    cat <<'HELP'
Usage: run_all.sh [options]

Options:
  -O LEVELS         Comma-separated opt levels (default: O2)
  --cmodel PATH     Path to aec-precise binary
  --compile-only    Compile only, do not run CModel
  -h, --help        Show this help
HELP
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -O) OPT_LEVELS="$2"; shift 2 ;;
        --cmodel) CMODEL="$2"; shift 2 ;;
        --compile-only) COMPILE_ONLY=true; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown: $1"; usage ;;
    esac
done

mkdir -p "$OUTDIR" "$INPUTDIR" "$DUMPDIR" "$BUILDDIR" "$REPORTDIR"

if [[ "$COMPILE_ONLY" != "true" ]]; then
    if [[ ! -x "$CMODEL" ]]; then
        if [[ -x "${PROJECT_DIR}/GoldenModel/bin/aec-precise-linux-x86_64" ]]; then
            CMODEL="${PROJECT_DIR}/GoldenModel/bin/aec-precise-linux-x86_64"
        elif [[ -x "${PROJECT_DIR}/GoldenModel/bin/aec-precise-macos-arm64" ]]; then
            CMODEL="${PROJECT_DIR}/GoldenModel/bin/aec-precise-macos-arm64"
        else
            echo "ERROR: CModel not found. Use: --cmodel /path/to/aec-precise"
            exit 1
        fi
    fi
    echo "[run_all] CModel: $CMODEL"
fi
echo "[run_all] Opt levels: $OPT_LEVELS"
echo "[run_all] Output: $OUTDIR"

# Step 1: Generate inputs
echo ""
echo "=== Step 1: Generate Inputs ==="
python3 "${SCRIPT_DIR}/input_gen.py" --testcases "$TESTCASES_DIR" --outdir "$INPUTDIR"

# Step 2: Compile
echo ""
echo "=== Step 2: Compile ==="
IFS=',' read -ra LEVELS <<< "$OPT_LEVELS"
declare -A COMPILED_BINS

for CASE in T1_basic_lowering T2_scalar_optimization T3_memory_reuse T4_register_scheduling T5_scalar_gemm; do
    PTX="${TESTCASES_DIR}/${CASE}/kernel.ptx"
    for LEVEL in "${LEVELS[@]}"; do
        BIN="${BUILDDIR}/${CASE}_${LEVEL}.aecbin"
        echo "  ${CASE} @ ${LEVEL}"
        python3 "$AEC_CC" "$PTX" "-${LEVEL}" -o "$BIN"
        COMPILED_BINS["${CASE}_${LEVEL}"]="$BIN"
    done
done

if [[ "$COMPILE_ONLY" == "true" ]]; then
    echo "[run_all] Compile-only — done. Binaries → $BUILDDIR"
    exit 0
fi

# Step 3: Run CModel
echo ""
echo "=== Step 3: Run CModel ==="

run_cmodel() {
    local label="$1" program="$2" case_name="$3" grid="$4" block="$5"
    local gmem="$6" max_steps="${7:-50000000}" dump_addr="$8" dump_size="$9" dump_name="${10}"
    shift 10

    local json_log="${REPORTDIR}/${label}.json"
    local dump_file="${DUMPDIR}/${dump_name}"

    echo "  ${label} ..."
    set +e
    "$CMODEL" \
        --program "$program" --grid "$grid" --block "$block" \
        --gmem-size "$gmem" --cmem-size 65536 --pmem-size 65536 \
        --smem-size 65536 --lmem-size 4096 --max-steps "$max_steps" \
        --dump "${dump_addr}:${dump_size}:${dump_file}" \
        "$@" > "$json_log" 2>&1
    local rc=$?
    set -e

    local status steps
    status=$(python3 -c "import json; d=json.load(open('$json_log')); print(d.get('status','?'))" 2>/dev/null || echo "?")
    steps=$(python3 -c "import json; d=json.load(open('$json_log')); print(d.get('steps',-1))" 2>/dev/null || echo "-1")
    printf "    status=%-8s  steps=%-8s  exit=%d\n" "$status" "$steps" "$rc"
}

for CASE in T1_basic_lowering T2_scalar_optimization T3_memory_reuse T4_register_scheduling T5_scalar_gemm; do
    for LEVEL in "${LEVELS[@]}"; do
        BIN="${COMPILED_BINS[${CASE}_${LEVEL}]}"
        IN="${INPUTDIR}/${CASE}"

        case "$CASE" in
            T1_basic_lowering)
                run_cmodel "${CASE}_${LEVEL}" "$BIN" "$CASE" \
                    "4096,1,1" "256,1,1" "12591104" "50000000" \
                    "8388864" "4194304" "T1_basic_lowering_dump_c.bin" \
                    --load "pmem:0:${IN}/pmem.bin" \
                    --load "gmem:256:${IN}/input_a.bin" \
                    --load "gmem:4194560:${IN}/input_b.bin" \
                    --load "gmem:8388864:${IN}/input_c.bin"
                ;;
            T2_scalar_optimization)
                run_cmodel "${CASE}_${LEVEL}" "$BIN" "$CASE" \
                    "2048,1,1" "256,1,1" "6299648" "50000000" \
                    "4194560" "2097152" "T2_scalar_optimization_dump_out.bin" \
                    --load "pmem:0:${IN}/pmem.bin" \
                    --load "gmem:256:${IN}/input_x.bin" \
                    --load "gmem:2097408:${IN}/input_y.bin" \
                    --load "gmem:4194560:${IN}/input_out.bin"
                ;;
            T3_memory_reuse)
                run_cmodel "${CASE}_${LEVEL}" "$BIN" "$CASE" \
                    "2048,1,1" "256,1,1" "8396800" "50000000" \
                    "6291712" "2097152" "T3_memory_reuse_dump_out.bin" \
                    --load "pmem:0:${IN}/pmem.bin" \
                    --load "gmem:256:${IN}/input_x.bin" \
                    --load "gmem:2097408:${IN}/input_y.bin" \
                    --load "gmem:4194560:${IN}/input_z.bin" \
                    --load "gmem:6291712:${IN}/input_out.bin"
                ;;
            T4_register_scheduling)
                run_cmodel "${CASE}_${LEVEL}" "$BIN" "$CASE" \
                    "2048,1,1" "256,1,1" "10493952" "50000000" \
                    "8388864" "2097152" "T4_register_scheduling_dump_out.bin" \
                    --load "pmem:0:${IN}/pmem.bin" \
                    --load "gmem:256:${IN}/input_a.bin" \
                    --load "gmem:2097408:${IN}/input_b.bin" \
                    --load "gmem:4194560:${IN}/input_c.bin" \
                    --load "gmem:6291712:${IN}/input_d.bin" \
                    --load "gmem:8388864:${IN}/input_out.bin"
                ;;
            T5_scalar_gemm)
                run_cmodel "${CASE}_${LEVEL}" "$BIN" "$CASE" \
                    "8,8,1" "16,16,1" "204800" "50000000" \
                    "131328" "65536" "T5_scalar_gemm_dump_C.bin" \
                    --load "pmem:0:${IN}/pmem.bin" \
                    --load "gmem:256:${IN}/input_A.bin" \
                    --load "gmem:65792:${IN}/input_B.bin" \
                    --load "gmem:131328:${IN}/input_C.bin"
                ;;
        esac
    done
done

# Step 4: Compute reference outputs
echo ""
echo "=== Step 4: Reference Outputs ==="
REFDIR="${OUTDIR}/ref_outputs"
python3 "${SCRIPT_DIR}/ref_compute.py" --testcases "$TESTCASES_DIR" --outdir "$REFDIR"

# Step 5: Check correctness
echo ""
echo "=== Step 5: Verify Correctness ==="
for LEVEL in "${LEVELS[@]}"; do
    echo ""
    echo "--- ${LEVEL} ---"
    python3 "${SCRIPT_DIR}/checker.py" \
        --dump-dir "$DUMPDIR" --ref-dir "$REFDIR" \
        --opt "$LEVEL" --report "${REPORTDIR}/correctness_${LEVEL}.json" || true
done

# Summary
echo ""
echo "============================================="
echo "  Golden Test Complete"
echo "============================================="
echo "  Build   → ${BUILDDIR}"
echo "  Inputs  → ${INPUTDIR}"
echo "  Dumps   → ${DUMPDIR}"
echo "  Ref     → ${REFDIR}"
echo "  Reports → ${REPORTDIR}"
echo ""

echo "--- Steps Summary ---"
for LEVEL in "${LEVELS[@]}"; do
    echo "  ${LEVEL}:"
    for CASE in T1_basic_lowering T2_scalar_optimization T3_memory_reuse T4_register_scheduling T5_scalar_gemm; do
        log="${REPORTDIR}/${CASE}_${LEVEL}.json"
        if [[ -f "$log" ]]; then
            python3 -c "
import json
d = json.load(open('$log'))
print(f'    {d.get(\"_case\",\"?\"):30s}  status={d.get(\"status\",\"?\"):8s}  steps={d.get(\"steps\",-1)}')
" 2>/dev/null || true
        fi
    done
done
