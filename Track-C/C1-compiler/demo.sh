#!/bin/bash
set -e

echo "=== C1 AEC Compiler Self-Test ==="
echo ""

REPO="$(cd "$(dirname "$0")" && pwd)"
TESTCASES="${REPO}/testcases"
BUILDDIR="/tmp/c1_demo_build"

mkdir -p "$BUILDDIR"

for CASE in T1_basic_lowering T2_scalar_optimization T3_memory_reuse T4_register_scheduling T5_scalar_gemm; do
    PTX="${TESTCASES}/${CASE}/kernel.ptx"
    OUT="${BUILDDIR}/${CASE}.aecbin"
    echo "  Compiling ${CASE} ..."
    python3 "${REPO}/compiler/aec-cc" "$PTX" -O2 -o "$OUT" --report "${BUILDDIR}/${CASE}_report.json"
    echo "    $(python3 -c "import os;print(os.path.getsize('$OUT'))") bytes"
done

echo ""
echo "=== Disassembly Check ==="
python3 "${REPO}/disassembler/aec-objdump" "${BUILDDIR}/T1_basic_lowering.aecbin" | head -5

echo ""
echo "=== All 5 cases compiled OK ==="
echo ""
echo "To run CModel verification (requires aec-precise on Linux/macOS):"
echo "  bash ${REPO}/golden_test/run_all.sh --cmodel ./aec-precise -O O2"
