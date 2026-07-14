#!/bin/bash
set -e
# =============================================================================
# 完整比赛评测模拟 —— 严格按 scoring.md / spec.md 执行
# =============================================================================

rsync -a --exclude=__pycache__ --exclude="*.pyc" --exclude=mig25 \
  /mnt/e/C1-compiler/compiler/ /home/annmarie/c1/compiler/
chmod +x /home/annmarie/c1/GoldenModel/bin/aec-precise-linux-x86_64
cd /home/annmarie/c1

OUTDIR=/tmp/c1_eval
mkdir -p ${OUTDIR}/{build,reports,inputs,dumps,ref}

echo "=========================================="
echo "  C1 比赛评测模拟"
echo "=========================================="

# ──── Phase A: 编译（评测机实际执行） ────
echo ""
echo "--- Phase A: 编译 (compiler/aec-cc {ptx} -O2 -o {output} --report {report}) ---"
echo ""

declare -A PTX_INSTS
declare -A AEC_INSTS
declare -A GPR_COUNT
declare -A PRED_COUNT
declare -A SPILLS

for CASE in T1_basic_lowering T2_scalar_optimization T3_memory_reuse T4_register_scheduling T5_scalar_gemm; do
  PTX="testcases/${CASE}/kernel.ptx"
  OUT="${OUTDIR}/build/${CASE}_O2.aecbin"
  REPORT="${OUTDIR}/reports/${CASE}_report.json"

  echo "  > compiler/aec-cc ${PTX} -O2 -o ${OUT} --report ${REPORT}"
  python3 compiler/aec-cc "$PTX" -O2 -o "$OUT" --report "$REPORT"

  R=$(python3 -c "import json;r=json.load(open('$REPORT'));print(r['num_ptx_instructions'],r['num_aec_instructions'],r['num_physical_registers'],r['num_predicates'],r['spills']['stores'],r['spills']['loads'])")
  read PTX_I AEC_I GPRS PREDS SP_S SP_L <<< "$R"
  PTX_INSTS[$CASE]=$PTX_I
  AEC_INSTS[$CASE]=$AEC_I
  GPR_COUNT[$CASE]=$GPRS
  PRED_COUNT[$CASE]=$PREDS
  SPILLS[$CASE]="${SP_S}:${SP_L}"
  echo "    PTX=${PTX_I}  AEC=${AEC_I}  GPRs=${GPRS}  Preds=${PREDS}  Spills=${SP_S}/${SP_L}"
done

# ──── Phase B: CModel 执行 + 正确性比对 ────
echo ""
echo "--- Phase B: CModel 执行 + 正确性 ---"
echo ""

python3 golden_test/input_gen.py --testcases testcases --outdir ${OUTDIR}/inputs
python3 golden_test/ref_compute.py --testcases testcases --outdir ${OUTDIR}/ref

CMODEL=./GoldenModel/bin/aec-precise-linux-x86_64
IN=${OUTDIR}/inputs
BLD=${OUTDIR}/build
DMP=${OUTDIR}/dumps

declare -A STEPS
declare -A CORRECT

run_case() {
  local LABEL=$1 PROG=$2 GRID=$3 BLOCK=$4 GMEM=$5 DADDR=$6 DSIZE=$7 DFILE=$8
  shift 8
  local RESULT
  RESULT=$($CMODEL --program "$PROG" --grid "$GRID" --block "$BLOCK" --gmem-size "$GMEM" \
    --cmem-size 65536 --pmem-size 65536 --smem-size 65536 --lmem-size 4096 \
    --max-steps 50000000 "$@" --dump "${DADDR}:${DSIZE}:${DFILE}" 2>&1)
  echo "    $LABEL: $RESULT"
  echo "$RESULT" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['status'],d['steps'])" 2>/dev/null
}

echo "  T1 vector_add"
run_case "T1" ${BLD}/T1_basic_lowering_O2.aecbin "4096,1,1" "256,1,1" "12591104" "8388864" "4194304" ${DMP}/T1_basic_lowering_dump_c.bin \
  --load "pmem:0:${IN}/T1_basic_lowering/pmem.bin" \
  --load "gmem:256:${IN}/T1_basic_lowering/input_a.bin" \
  --load "gmem:4194560:${IN}/T1_basic_lowering/input_b.bin" \
  --load "gmem:8388864:${IN}/T1_basic_lowering/input_c.bin" > /dev/null
read STATUS STEP <<< $(tail -1 /dev/stdin 2>/dev/null || echo "? 0")
STEPS[T1]=$STEP; [[ "$STATUS" == "done" ]] && CORRECT[T1]=1 || CORRECT[T1]=0

echo "  T2 scalar_optimization"
run_case "T2" ${BLD}/T2_scalar_optimization_O2.aecbin "2048,1,1" "256,1,1" "6299648" "4194560" "2097152" ${DMP}/T2_scalar_optimization_dump_out.bin \
  --load "pmem:0:${IN}/T2_scalar_optimization/pmem.bin" \
  --load "gmem:256:${IN}/T2_scalar_optimization/input_x.bin" \
  --load "gmem:2097408:${IN}/T2_scalar_optimization/input_y.bin" \
  --load "gmem:4194560:${IN}/T2_scalar_optimization/input_out.bin" > /dev/null
read STATUS STEP <<< $(tail -1 /dev/stdin 2>/dev/null || echo "? 0")
STEPS[T2]=$STEP; [[ "$STATUS" == "done" ]] && CORRECT[T2]=1 || CORRECT[T2]=0

echo "  T3 memory_reuse"
run_case "T3" ${BLD}/T3_memory_reuse_O2.aecbin "2048,1,1" "256,1,1" "8396800" "6291712" "2097152" ${DMP}/T3_memory_reuse_dump_out.bin \
  --load "pmem:0:${IN}/T3_memory_reuse/pmem.bin" \
  --load "gmem:256:${IN}/T3_memory_reuse/input_x.bin" \
  --load "gmem:2097408:${IN}/T3_memory_reuse/input_y.bin" \
  --load "gmem:4194560:${IN}/T3_memory_reuse/input_z.bin" \
  --load "gmem:6291712:${IN}/T3_memory_reuse/input_out.bin" > /dev/null
read STATUS STEP <<< $(tail -1 /dev/stdin 2>/dev/null || echo "? 0")
STEPS[T3]=$STEP; [[ "$STATUS" == "done" ]] && CORRECT[T3]=1 || CORRECT[T3]=0

echo "  T4 register_scheduling"
run_case "T4" ${BLD}/T4_register_scheduling_O2.aecbin "2048,1,1" "256,1,1" "10493952" "8388864" "2097152" ${DMP}/T4_register_scheduling_dump_out.bin \
  --load "pmem:0:${IN}/T4_register_scheduling/pmem.bin" \
  --load "gmem:256:${IN}/T4_register_scheduling/input_a.bin" \
  --load "gmem:2097408:${IN}/T4_register_scheduling/input_b.bin" \
  --load "gmem:4194560:${IN}/T4_register_scheduling/input_c.bin" \
  --load "gmem:6291712:${IN}/T4_register_scheduling/input_d.bin" \
  --load "gmem:8388864:${IN}/T4_register_scheduling/input_out.bin" > /dev/null
read STATUS STEP <<< $(tail -1 /dev/stdin 2>/dev/null || echo "? 0")
STEPS[T4]=$STEP; [[ "$STATUS" == "done" ]] && CORRECT[T4]=1 || CORRECT[T4]=0

echo "  T5 scalar_gemm"
run_case "T5" ${BLD}/T5_scalar_gemm_O2.aecbin "8,8,1" "16,16,1" "204800" "131328" "65536" ${DMP}/T5_scalar_gemm_dump_C.bin \
  --load "pmem:0:${IN}/T5_scalar_gemm/pmem.bin" \
  --load "gmem:256:${IN}/T5_scalar_gemm/input_A.bin" \
  --load "gmem:65792:${IN}/T5_scalar_gemm/input_B.bin" \
  --load "gmem:131328:${IN}/T5_scalar_gemm/input_C.bin" > /dev/null
read STATUS STEP <<< $(tail -1 /dev/stdin 2>/dev/null || echo "? 0")
STEPS[T5]=$STEP; [[ "$STATUS" == "done" ]] && CORRECT[T5]=1 || CORRECT[T5]=0

# ──── Phase C: 正确性比对 ────
echo ""
echo "--- Phase C: 正确性比对 ---"
python3 golden_test/checker.py --dump-dir ${OUTDIR}/dumps --ref-dir ${OUTDIR}/ref --opt O2

# ──── 评分报告 ────
echo ""
echo "=========================================="
echo "  评测结果汇总"
echo "=========================================="
echo ""
printf "  %-30s %6s %8s %12s %10s\n" "Case" "AEC" "Steps" "Correctness" "Status"
printf "  %-30s %6s %8s %12s %10s\n" "------------------------------" "---" "------" "-----------" "--------"

for CASE in T1 T2 T3 T4 T5; do
  case "$CASE" in
    T1) FULL="T1_basic_lowering" ;;
    T2) FULL="T2_scalar_optimization" ;;
    T3) FULL="T3_memory_reuse" ;;
    T4) FULL="T4_register_scheduling" ;;
    T5) FULL="T5_scalar_gemm" ;;
  esac
  C=${CORRECT[$CASE]:-0}
  if [ "$C" == "1" ]; then
    COR_STR="PASS"
  else
    COR_STR="FAIL"
  fi
  printf "  %-30s %6s %8s %12s\n" "$FULL" "${AEC_INSTS[$FULL]}" "${STEPS[$CASE]}" "$COR_STR"
done

echo ""
PASSED=$(( ${CORRECT[T1]:-0} + ${CORRECT[T2]:-0} + ${CORRECT[T3]:-0} + ${CORRECT[T4]:-0} + ${CORRECT[T5]:-0} ))
echo "  正确性: ${PASSED}/5 通过"
echo ""
echo "  诊断指标:"
for CASE in T1_basic_lowering T2_scalar_optimization T3_memory_reuse T4_register_scheduling T5_scalar_gemm; do
  echo "    ${CASE}: PTX=${PTX_INSTS[$CASE]} AEC=${AEC_INSTS[$CASE]} GPRs=${GPR_COUNT[$CASE]} Preds=${PRED_COUNT[$CASE]} Spills=${SPILLS[$CASE]}"
done
echo ""
echo "  提交命令模板:"
echo "    python compiler/aec-cc {ptx} -O2 -o {output} --report {report}"
echo ""
echo "=========================================="
echo "  评测模拟完成"
echo "=========================================="
