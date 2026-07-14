#!/usr/bin/env python3
"""Quick per-module tests — C3.1, C3.2, C3.3, C3.4 (no subprocess)."""
import json, os, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import onnx
from onnx import helper, TensorProto
from scheduler import GraphIR, GraphNode, strategy, GraphPassPipeline
from scheduler.memory import DeviceMemoryPool, analyze_lifetimes, build_execution_plan

ok = 0
total = 0
def check(name, cond, detail=""):
    global ok, total
    total += 1
    if cond:
        print(f"  [PASS] {name}" + (f"  {detail}" if detail else ""))
        ok += 1
    else:
        print(f"  [FAIL] {name}  {detail}")

# ── C3.1: ONNX -> DAG JSON ──
print("="*50)
print("C3.1 export_dag.py")
print("="*50)
tmp = tempfile.mkdtemp(prefix="c3t_")
sys.path.insert(0, str(HERE))
import importlib.util
spec = importlib.util.spec_from_file_location("export_dag", str(HERE / "export_dag.py"))
ed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ed)

# Make tiny MLP ONNX
inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 1, 28, 28])
out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", 10])
w1 = helper.make_tensor("w1", TensorProto.FLOAT, [784, 128], [0.01]*(784*128))
b1 = helper.make_tensor("b1", TensorProto.FLOAT, [128], [0.0]*128)
n1 = helper.make_node("Flatten", ["input"], ["f"], name="flatten", axis=1)
n2 = helper.make_node("Gemm", ["f", "w1", "b1"], ["o1"], name="fc1")
n3 = helper.make_node("Relu", ["o1"], ["r"], name="relu")
n4 = helper.make_node("Gemm", ["r", "w2", "b2"], ["logits"], name="fc2")
w2 = helper.make_tensor("w2", TensorProto.FLOAT, [128, 10], [0.01]*(128*10))
b2 = helper.make_tensor("b2", TensorProto.FLOAT, [10], [0.0]*10)
graph = helper.make_graph([n1,n2,n3,n4],"mlp",[inp],[out],initializer=[w1,b1,w2,b2])
model = helper.make_model(graph, producer_name="c3t")
model.opset_import[0].version = 20
mp = os.path.join(tmp, "m.onnx")
onnx.save(model, mp)

dag = ed.parse_onnx(mp)
check("format_version", dag.get("format_version") == "1.0")
check("graph_inputs", len(dag["graph_inputs"]) == 1)
check("graph_outputs", len(dag["graph_outputs"]) == 1)
check("num_nodes", dag["num_nodes"] == 4)
check("num_edges", dag["num_edges"] == 3)
check("nodes have op_type", all("op_type" in n for n in dag["nodes"]))
check("edge keys", all(set(e.keys()) >= {"src","dst","tensor"} for e in dag["edges"]))

# ── C3.2 Decomposition ──
print("\n" + "="*50)
print("C3.2 Decomposition + Tuning")
print("="*50)
gir = GraphIR(inputs=["input"], outputs=["logits"], nodes=[
    GraphNode("flatten","Flatten",["input"],["f"]),
    GraphNode("fc1","Gemm",["f","w1","b1"],["o1"]),
    GraphNode("relu","Relu",["o1"],["r"]),
    GraphNode("fc2","Gemm",["r","w2","b2"],["logits"]),
])

precs = set()
total_kernels = 0
all_tuning_ok = True
for node in gir.nodes:
    prec = strategy.select_precision(node, gir)
    precs.add(prec.precision)
    kernels = strategy.decompose(node, gir, prec)
    total_kernels += len(kernels)
    for k in kernels:
        tuning = strategy.tune_kernel(k, prec, 1024)
        if not (0 < tuning.block_x <= 256):
            all_tuning_ok = False
        if not (tuning.grid_x > 0):
            all_tuning_ok = False
        if not (tuning.smem_bytes <= 48*1024 or tuning.smem_bytes == -1):
            all_tuning_ok = False

check("precisions used", len(precs) >= 2, f"precs={precs}")
check("total kernels", total_kernels >= 4, f"kernels={total_kernels}")
check("tuning validity", all_tuning_ok)
check("Gemm → matmul", any("matmul_fp" in k.name for node in gir.nodes
      for k in strategy.decompose(node, gir, strategy.select_precision(node, gir))))

# Test Softmax decomposition
sgir = GraphIR(inputs=["x"], outputs=["y"], nodes=[
    GraphNode("sm","Softmax",["x"],["y"]),
])
sprec = strategy.select_precision(sgir.nodes[0], sgir)
sk = strategy.decompose(sgir.nodes[0], sgir, sprec)
check("Softmax fp32", sprec.precision == "fp32")
check("Softmax 4-kernel", len(sk) == 4, f"got {len(sk)} kernels: {[k.name for k in sk]}")

# LayerNorm test
lgir = GraphIR(inputs=["x","w","b"], outputs=["y"], nodes=[
    GraphNode("ln","LayerNormalization",["x","w","b"],["y"]),
])
lprec = strategy.select_precision(lgir.nodes[0], lgir)
lk = strategy.decompose(lgir.nodes[0], lgir, lprec)
check("LayerNorm fp32", lprec.precision == "fp32")
check("LayerNorm 6-kernel", len(lk) == 6, f"got {len(lk)} kernels")

# C3.3 Fusion
print("\n" + "="*50)
print("C3.3 Fusion")
print("="*50)

# Test: MatMul -> AddBias
fgir = GraphIR(inputs=["a","b","c"], outputs=["y"], nodes=[
    GraphNode("m","MatMul",["a","b"],["m_out"]),
    GraphNode("add","Add",["m_out","c"],["y"]),
])
pipeline = GraphPassPipeline(enable_fusion=True)
opt, stats = pipeline.run(fgir)
log = stats["Fusion"]["stats"]["fusion_log"]
check("FusedMatMulBias", len(log) == 1 and log[0]["pattern"] == "FusedMatMulBias",
      f"fusions={log}")
check("nodes reduced", len(opt.nodes) < len(fgir.nodes),
      f"{len(fgir.nodes)} → {len(opt.nodes)}")

# Test: elementwise chain
ew_gir = GraphIR(inputs=["x"], outputs=["y"], nodes=[
    GraphNode("add","Add",["x","c1"],["a1"]),
    GraphNode("mul","Mul",["a1","c2"],["a2"]),
    GraphNode("relu","Relu",["a2"],["y"]),
])
opt2, stats2 = pipeline.run(ew_gir)
log2 = stats2["Fusion"]["stats"]["fusion_log"]
check("FusedEWChain", len(log2) >= 1, f"fusions={log2}")

# C3.4 Memory
print("\n" + "="*50)
print("C3.4 Memory")
print("="*50)

lifetimes = analyze_lifetimes(opt)
check("lifetimes tracked", len(lifetimes) > 0, f"count={len(lifetimes)}")

# Test memory pool
pool = DeviceMemoryPool()
b1 = pool.malloc(4096)
check("pool alloc", b1.size >= 4096)
b2 = pool.malloc(8192)
check("pool alloc 2", b2.size >= 8192)
pool.free(b1)
b3 = pool.malloc(2048)
check("pool reuse smaller", b3.offset < b2.offset,  # reused b1's slot
      f"offset={b3.offset}, capacity={pool.capacity}")
pool.free(b2)
pool.free(b3)
check("pool coalesced", pool.capacity > 0 and all(blk.free for blk in pool.blocks),
      f"blocks={[(blk.offset, blk.size, blk.free) for blk in pool.blocks]}")

plan = build_execution_plan(opt)
check("execution plan", len(plan) > 0, f"steps={len(plan)}")
check("has compute streams", any(
    s.stream.startswith("compute_") for s in plan
), f"streams={set(s.stream for s in plan)}")

# ── summary ──
print(f"\n{'='*50}")
print(f"Summary: {ok}/{total} passed")
print(f"{'='*50}")
if ok == total:
    print("ALL PASSED!")
else:
    print(f"{total - ok} FAILED")
print(f"\nTest artifacts at: {tmp}")
