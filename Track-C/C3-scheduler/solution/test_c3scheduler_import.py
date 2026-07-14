#!/usr/bin/env python3
"""Verify c3scheduler import_onnx_graph with a structure-identical MLP ONNX.

Official models not available locally — synthetic equivalent tested instead.
"""
import onnx, sys
from onnx import helper, TensorProto
from pathlib import Path
from c3scheduler.graph import import_onnx_graph
from c3scheduler.strategy import strategy
from c3scheduler.graph_passes import GraphPassPipeline
from c3scheduler.memory import analyze_lifetimes, build_execution_plan

# ── Build MLP ONNX (Flatten → Gemm → Relu → Gemm) ──
inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 1, 28, 28])
out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", 10])
w1 = helper.make_tensor("w1", TensorProto.FLOAT, [784, 128], [0.01] * 100352)
b1 = helper.make_tensor("b1", TensorProto.FLOAT, [128], [0.0] * 128)
w2 = helper.make_tensor("w2", TensorProto.FLOAT, [128, 10], [0.01] * 1280)
b2 = helper.make_tensor("b2", TensorProto.FLOAT, [10], [0.0] * 10)
n1 = helper.make_node("Flatten", ["input"], ["f"], name="flatten", axis=1)
n2 = helper.make_node("Gemm", ["f", "w1", "b1"], ["o1"], name="fc1")
n3 = helper.make_node("Relu", ["o1"], ["r"], name="relu")
n4 = helper.make_node("Gemm", ["r", "w2", "b2"], ["logits"], name="fc2")
g = helper.make_graph([n1,n2,n3,n4], "mlp", [inp], [out], initializer=[w1,b1,w2,b2])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
p = Path("/tmp/_c3_import_mlp.onnx")
onnx.save(m, str(p))

# ── Test import_onnx_graph from c3scheduler ──
gir = import_onnx_graph(p)
assert len(gir.nodes) == 4, f"Expected 4 nodes, got {len(gir.nodes)}"
assert gir.validate()
print(f"MLP: {len(gir.nodes)} nodes  validate=True   OK")

# ── Full C3.2 / C3.3 / C3.4 pipeline ──
for node in gir.nodes:
    prec = strategy.select_precision(node, gir)
    kernels = strategy.decompose(node, gir, prec)
    assert len(kernels) > 0, f"{node.op_type}: no kernels"

pipe = GraphPassPipeline(enable_fusion=True)
opt, stats = pipe.run(gir)
patterns = [r["pattern"] for r in stats["Fusion"]["stats"]["fusion_log"]]
print(f"C3.3 Fusion: {len(gir.nodes)} -> {len(opt.nodes)}  patterns={patterns}")

lt = analyze_lifetimes(opt)
plan = build_execution_plan(opt)
print(f"C3.4 Memory: {len(lt)} lifetimes  {len(plan)} plan steps  streams={set(s.stream for s in plan)}")

print()
print("c3scheduler MLP full pipeline: PASSED")
