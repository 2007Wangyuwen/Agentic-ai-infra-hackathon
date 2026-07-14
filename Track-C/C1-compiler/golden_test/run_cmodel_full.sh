#!/bin/bash
# Full C1 CModel test: generate inputs -> compile -> run T1-T5
python3 /mnt/e/Agentic-ai-infra-hackathon/Track-C/C1-compiler/golden_test/gen_inputs.py
TC=/mnt/e/Agentic-ai-infra-hackathon/Track-C/C1-compiler/testcases
CC=/mnt/e/Agentic-ai-infra-hackathon/Track-C/C1-compiler/compiler/aec-cc
CM=/mnt/e/Agentic-ai-infra-hackathon/aec-precise-linux-x86_64
chmod +x $CM

# Compile all
for case in T1_basic_lowering T2_scalar_optimization T3_memory_reuse T4_register_scheduling T5_scalar_gemm; do
    python3 $CC $TC/$case/kernel.ptx -O2 -o /tmp/${case}_O2.aecbin --report /tmp/${case}_report.json
done

# T1
echo -n "T1: "
$CM --program /tmp/T1_basic_lowering_O2.aecbin --grid 4096,1,1 --block 256,1,1 --gmem-size 12591104 --cmem-size 65536 --pmem-size 65536 --smem-size 65536 --lmem-size 4096 --max-steps 50000000 --load pmem:0:/tmp/c1_inputs/T1_basic_lowering/pmem.bin --load gmem:256:/tmp/c1_inputs/T1_basic_lowering/input_a.bin --load gmem:4194560:/tmp/c1_inputs/T1_basic_lowering/input_b.bin --load gmem:8388864:/tmp/c1_inputs/T1_basic_lowering/input_c.bin --dump 8388864:4194304:/tmp/T1_dump.bin 2>&1 | grep -o '\{"status".*\}'

# T2
echo -n "T2: "
$CM --program /tmp/T2_scalar_optimization_O2.aecbin --grid 2048,1,1 --block 256,1,1 --gmem-size 6299648 --cmem-size 65536 --pmem-size 65536 --smem-size 65536 --lmem-size 4096 --max-steps 50000000 --load pmem:0:/tmp/c1_inputs/T2_scalar_optimization/pmem.bin --load gmem:256:/tmp/c1_inputs/T2_scalar_optimization/input_x.bin --load gmem:2097408:/tmp/c1_inputs/T2_scalar_optimization/input_y.bin --load gmem:4194560:/tmp/c1_inputs/T2_scalar_optimization/input_out.bin --dump 4194560:2097152:/tmp/T2_dump.bin 2>&1 | grep -o '\{"status".*\}'

# T3
echo -n "T3: "
$CM --program /tmp/T3_memory_reuse_O2.aecbin --grid 2048,1,1 --block 256,1,1 --gmem-size 8396800 --cmem-size 65536 --pmem-size 65536 --smem-size 65536 --lmem-size 4096 --max-steps 50000000 --load pmem:0:/tmp/c1_inputs/T3_memory_reuse/pmem.bin --load gmem:256:/tmp/c1_inputs/T3_memory_reuse/input_x.bin --load gmem:2097408:/tmp/c1_inputs/T3_memory_reuse/input_y.bin --load gmem:4194560:/tmp/c1_inputs/T3_memory_reuse/input_z.bin --load gmem:6291712:/tmp/c1_inputs/T3_memory_reuse/input_out.bin --dump 6291712:2097152:/tmp/T3_dump.bin 2>&1 | grep -o '\{"status".*\}'

# T4
echo -n "T4: "
$CM --program /tmp/T4_register_scheduling_O2.aecbin --grid 2048,1,1 --block 256,1,1 --gmem-size 10493952 --cmem-size 65536 --pmem-size 65536 --smem-size 65536 --lmem-size 4096 --max-steps 50000000 --load pmem:0:/tmp/c1_inputs/T4_register_scheduling/pmem.bin --load gmem:256:/tmp/c1_inputs/T4_register_scheduling/input_a.bin --load gmem:2097408:/tmp/c1_inputs/T4_register_scheduling/input_b.bin --load gmem:4194560:/tmp/c1_inputs/T4_register_scheduling/input_c.bin --load gmem:6291712:/tmp/c1_inputs/T4_register_scheduling/input_d.bin --load gmem:8388864:/tmp/c1_inputs/T4_register_scheduling/input_out.bin --dump 8388864:2097152:/tmp/T4_dump.bin 2>&1 | grep -o '\{"status".*\}'

# T5
echo -n "T5: "
$CM --program /tmp/T5_scalar_gemm_O2.aecbin --grid 8,8,1 --block 16,16,1 --gmem-size 204800 --cmem-size 65536 --pmem-size 65536 --smem-size 65536 --lmem-size 4096 --max-steps 50000000 --load pmem:0:/tmp/c1_inputs/T5_scalar_gemm/pmem.bin --load gmem:256:/tmp/c1_inputs/T5_scalar_gemm/input_A.bin --load gmem:65792:/tmp/c1_inputs/T5_scalar_gemm/input_B.bin --load gmem:131328:/tmp/c1_inputs/T5_scalar_gemm/input_C.bin --dump 131328:65536:/tmp/T5_dump.bin 2>&1 | grep -o '\{"status".*\}'

echo "DONE"
