#!/bin/bash
set -e
rsync -a --exclude=__pycache__ --exclude="*.pyc" --exclude=mig25 /mnt/e/C1-compiler/compiler/ /home/annmarie/c1/compiler/
chmod +x /home/annmarie/c1/GoldenModel/bin/aec-precise-linux-x86_64
cd /home/annmarie/c1
mkdir -p /tmp/c1_run/inputs /tmp/c1_run/build /tmp/c1_run/dumps /tmp/c1_run/ref

echo "=== Gen ==="
python3 golden_test/input_gen.py --testcases testcases --outdir /tmp/c1_run/inputs

for C in T1_basic_lowering T2_scalar_optimization T3_memory_reuse T4_register_scheduling T5_scalar_gemm; do
  python3 compiler/aec-cc "testcases/${C}/kernel.ptx" -O2 -o "/tmp/c1_run/build/${C}_O2.aecbin"
done

CMODEL=./GoldenModel/bin/aec-precise-linux-x86_64
IN=/tmp/c1_run/inputs
BLD=/tmp/c1_run/build
DMP=/tmp/c1_run/dumps

echo "=== T1 ==="
$CMODEL --program ${BLD}/T1_basic_lowering_O2.aecbin --grid 4096,1,1 --block 256,1,1 --gmem-size 12591104 --max-steps 50000000 --load pmem:0:${IN}/T1_basic_lowering/pmem.bin --load gmem:256:${IN}/T1_basic_lowering/input_a.bin --load gmem:4194560:${IN}/T1_basic_lowering/input_b.bin --load gmem:8388864:${IN}/T1_basic_lowering/input_c.bin --dump 8388864:4194304:${DMP}/T1_basic_lowering_dump_c.bin

echo "=== T2 ==="
$CMODEL --program ${BLD}/T2_scalar_optimization_O2.aecbin --grid 2048,1,1 --block 256,1,1 --gmem-size 6299648 --max-steps 50000000 --load pmem:0:${IN}/T2_scalar_optimization/pmem.bin --load gmem:256:${IN}/T2_scalar_optimization/input_x.bin --load gmem:2097408:${IN}/T2_scalar_optimization/input_y.bin --load gmem:4194560:${IN}/T2_scalar_optimization/input_out.bin --dump 4194560:2097152:${DMP}/T2_scalar_optimization_dump_out.bin

echo "=== T3 ==="
$CMODEL --program ${BLD}/T3_memory_reuse_O2.aecbin --grid 2048,1,1 --block 256,1,1 --gmem-size 8396800 --max-steps 50000000 --load pmem:0:${IN}/T3_memory_reuse/pmem.bin --load gmem:256:${IN}/T3_memory_reuse/input_x.bin --load gmem:2097408:${IN}/T3_memory_reuse/input_y.bin --load gmem:4194560:${IN}/T3_memory_reuse/input_z.bin --load gmem:6291712:${IN}/T3_memory_reuse/input_out.bin --dump 6291712:2097152:${DMP}/T3_memory_reuse_dump_out.bin

echo "=== T4 ==="
$CMODEL --program ${BLD}/T4_register_scheduling_O2.aecbin --grid 2048,1,1 --block 256,1,1 --gmem-size 10493952 --max-steps 50000000 --load pmem:0:${IN}/T4_register_scheduling/pmem.bin --load gmem:256:${IN}/T4_register_scheduling/input_a.bin --load gmem:2097408:${IN}/T4_register_scheduling/input_b.bin --load gmem:4194560:${IN}/T4_register_scheduling/input_c.bin --load gmem:6291712:${IN}/T4_register_scheduling/input_d.bin --load gmem:8388864:${IN}/T4_register_scheduling/input_out.bin --dump 8388864:2097152:${DMP}/T4_register_scheduling_dump_out.bin

echo "=== T5 ==="
$CMODEL --program ${BLD}/T5_scalar_gemm_O2.aecbin --grid 8,8,1 --block 16,16,1 --gmem-size 204800 --max-steps 50000000 --load pmem:0:${IN}/T5_scalar_gemm/pmem.bin --load gmem:256:${IN}/T5_scalar_gemm/input_A.bin --load gmem:65792:${IN}/T5_scalar_gemm/input_B.bin --load gmem:131328:${IN}/T5_scalar_gemm/input_C.bin --dump 131328:65536:${DMP}/T5_scalar_gemm_dump_C.bin

echo "=== Ref ==="
python3 golden_test/ref_compute.py --testcases testcases --outdir /tmp/c1_run/ref

echo "=== Check ==="
python3 golden_test/checker.py --dump-dir /tmp/c1_run/dumps --ref-dir /tmp/c1_run/ref --opt O2

echo "=== DONE ==="
