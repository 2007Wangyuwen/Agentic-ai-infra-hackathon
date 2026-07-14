#!/usr/bin/env python3
"""Minimal replica of the official benchmark's import surface.

Simulates exactly how bench_c32_c33.py calls our public APIs,
based on the published contract in spec.md (C3.2/C3.3 sections).

This file does NOT modify the original benchmark — it validates that
our c3scheduler package satisfies every documented import.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# ═══════════════════════════════════════════════════════════════
# Exact imports used by the official benchmark (per spec.md)
# ═══════════════════════════════════════════════════════════════
from c3scheduler.graph import import_onnx_graph, GraphIR, GraphNode
from c3scheduler.strategy import strategy, hardware
from c3scheduler.graph_passes import GraphPassPipeline

ok = 0; total = 0

def check(name, passed, detail=""):
    global ok, total
    total += 1
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {name}" + (f"  {detail}" if detail else ""))
    if passed: ok += 1


# ── build a synthetic MLP ONNX in memory ──
import onnx
from onnx import helper, TensorProto

inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 1, 28, 28])
out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", 10])
w1 = helper.make_tensor("w1", TensorProto.FLOAT, [784, 128], [0.01] * 100352)
b1 = helper.make_tensor("b1", TensorProto.FLOAT, [128], [0.0] * 128)
w2 = helper.make_tensor("w2", TensorProto.FLOAT, [128, 10], [0.01] * 1280)
b2 = helper.make_tensor("b2", TensorProto.FLOAT, [10], [0.0] * 10)
nodes = [
    helper.make_node("Flatten", ["input"], ["f"], name="flatten", axis=1),
    helper.make_node("Gemm", ["f", "w1", "b1"], ["o1"], name="fc1"),
    helper.make_node("Relu", ["o1"], ["r"], name="relu"),
    helper.make_node("Gemm", ["r", "w2", "b2"], ["logits"], name="fc2"),
]
g = helper.make_graph(nodes, "mlp", [inp], [out], initializer=[w1, b1, w2, b2])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
model_path = "/tmp/_bench_import_mlp.onnx"
onnx.save(m, model_path)


# ═══════════════════════════════════════════════════════════════
# C3.2: Decomposition — exact spec.md contract
# ═══════════════════════════════════════════════════════════════

# (1) import_onnx_graph → GraphIR (the "original operator DAG")
graph = import_onnx_graph(model_path)
check("import_onnx_graph returns GraphIR", isinstance(graph, GraphIR))

# (2) strategy.select_precision(node, graph) → PrecisionProfile
precisions = []
for node in graph.nodes:
    prec = strategy.select_precision(node, graph)
    assert prec.precision in hardware.supported_precisions(), \
        f"{node.op_type}: {prec.precision} not in hardware"
    precisions.append(prec.precision)
check("select_precision all in hardware", len(precisions) == len(graph.nodes))

# (3) strategy.decompose(node, graph, precision) → List[KernelSpecRef]
#     Expect Softmax/LayerNorm to produce intermediate tensors
total_kernels = 0
intermediate_count = 0
for node in graph.nodes:
    prec = strategy.select_precision(node, graph)
    kernels = strategy.decompose(node, graph, prec)
    total_kernels += len(kernels)
    # intermediate = KernelSpecRef.outputs \ node.outputs
    node_outs = set(node.outputs)
    for k in kernels:
        kern_outs = set(k.outputs)
        if kern_outs - node_outs:
            intermediate_count += 1
check("decompose produced kernels", total_kernels >= len(graph.nodes),
      f"kernels={total_kernels} nodes={len(graph.nodes)}")

# (4) strategy.tune_kernel(ref, precision, problem_size) → KernelTuningParams
all_tuning_valid = True
for node in graph.nodes:
    prec = strategy.select_precision(node, graph)
    kernels = strategy.decompose(node, graph, prec)
    for k in kernels:
        tp = strategy.tune_kernel(k, prec, 1024)
        if not (0 < tp.block_x <= hardware.max_threads_per_block):
            all_tuning_valid = False
        if not (tp.grid_x > 0):
            all_tuning_valid = False
        if not (tp.smem_bytes <= hardware.smem_bytes or tp.smem_bytes == -1):
            all_tuning_valid = False
check("tune_kernel validity", all_tuning_valid,
      f"block_x ok / grid_x>0 / smem within budget")

# (5) hardware.supported_precisions()
spp = hardware.supported_precisions()
check("hardware.supported_precisions", len(spp) >= 2, f"precisions={spp}")


# ═══════════════════════════════════════════════════════════════
# C3.3: Fusion — exact spec.md contract
# ═══════════════════════════════════════════════════════════════

pipeline = GraphPassPipeline(enable_fusion=True)
optimized, pass_results = pipeline.run(graph)

# Must return: pass_results["Fusion"]["stats"]["fusion_log"]
fusion_log = pass_results["Fusion"]["stats"]["fusion_log"]
check("pipeline returns fusion_log", isinstance(fusion_log, list))

# Each entry must have: pattern, nodes, fused_node
for entry in fusion_log:
    check(f"fusion entry keys: {entry['pattern']}",
          set(entry.keys()) >= {"pattern", "nodes", "fused_node"})

# graph.validate() must pass after fusion
check("optimized graph validates", optimized.validate())

# graph.outputs / graph.inputs preserved
check("graph outputs preserved", bool(optimized.outputs))
check("graph inputs preserved", bool(optimized.inputs))

# Optimized graph nodes <= original nodes
check("fusion reduced or same nodes", len(optimized.nodes) <= len(graph.nodes),
      f"{len(graph.nodes)} -> {len(optimized.nodes)}")


# ═══════════════════════════════════════════════════════════════
# C3.4: Memory — public API check
# ═══════════════════════════════════════════════════════════════
from c3scheduler.memory import analyze_lifetimes, build_execution_plan, DeviceMemoryPool

lt = analyze_lifetimes(optimized)
check("analyze_lifetimes returns dict", isinstance(lt, dict), f"count={len(lt)}")

plan = build_execution_plan(optimized)
check("build_execution_plan returns list", isinstance(plan, list), f"steps={len(plan)}")

pool = DeviceMemoryPool()
b1 = pool.malloc(4096)
pool.free(b1)
check("DeviceMemoryPool alloc+free works", True)


# ── report ──
print(f"\n{'=' * 50}")
print(f"  Summary: {ok}/{total} PASSED")
print(f"{'=' * 50}")

sys.exit(0 if ok == total else 1)
