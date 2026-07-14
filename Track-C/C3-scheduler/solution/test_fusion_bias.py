#!/usr/bin/env python3
"""Fusion bias detection boundary tests — C3.3 scoring requirement."""

from c3scheduler.graph import GraphIR, GraphNode
from c3scheduler.graph_passes import GraphPassPipeline


ok = 0
total = 0


def check(name: str, passed: bool, detail: str = "") -> None:
    global ok, total
    total += 1
    tag = "PASS" if passed else "FAIL"
    d = f"  {detail}" if detail else ""
    print(f"  [{tag}] {name}{d}")
    if passed:
        ok += 1


def _run_fusion(graph: GraphIR):
    p = GraphPassPipeline(enable_fusion=True)
    opt, stats = p.run(graph)
    log = stats["Fusion"]["stats"]["fusion_log"]
    fused = any(r["pattern"] == "FusedMatMulBias" for r in log)
    return opt, fused


# ═══════════════════════════════════════════════════════════════
# Test 1: initializer bias — SHOULD fuse
# ═══════════════════════════════════════════════════════════════
g = GraphIR(
    inputs=["x", "w", "bias"],
    outputs=["y"],
    nodes=[
        GraphNode("mm", "MatMul", ["x", "w"], ["mm_out"]),
        GraphNode("add", "Add", ["mm_out", "bias"], ["y"]),
    ],
    initializers={"w", "bias"},
)
opt, fused = _run_fusion(g)
check("initializer bias fused", fused,
      f"nodes={len(g.nodes)}->{len(opt.nodes)}")
check("initializer bias validate", opt.validate())

# ═══════════════════════════════════════════════════════════════
# Test 2: Constant-node bias — SHOULD fuse (e.g. attention scale)
# ═══════════════════════════════════════════════════════════════
g2 = GraphIR(
    inputs=["x", "w"],
    outputs=["y"],
    nodes=[
        GraphNode("cst", "Constant", [], ["cst"], attributes={"value": 1.0}),
        GraphNode("mm", "MatMul", ["x", "w"], ["mm_out"]),
        GraphNode("add", "Add", ["mm_out", "cst"], ["y"]),
    ],
    initializers={"w"},
)
opt2, fused2 = _run_fusion(g2)
check("Constant-node bias fused", fused2,
      f"nodes={len(g2.nodes)}->{len(opt2.nodes)}")
check("Constant-node bias validate", opt2.validate())

# ═══════════════════════════════════════════════════════════════
# Test 3: residual Add — should NOT fuse
# ═══════════════════════════════════════════════════════════════
g3 = GraphIR(
    inputs=["x", "w", "residual"],
    outputs=["y"],
    nodes=[
        GraphNode("mm", "MatMul", ["x", "w"], ["mm_out"]),
        GraphNode("add", "Add", ["mm_out", "residual"], ["y"]),
    ],
    initializers={"w"},
)
opt3, fused3 = _run_fusion(g3)
check("residual Add NOT fused", not fused3,
      f"fused={fused3} nodes={len(g3.nodes)}->{len(opt3.nodes)}")
check("residual Add validate", opt3.validate())
check("residual MatMul preserved",
      any(n.op_type == "MatMul" for n in opt3.nodes))
check("residual Add preserved",
      any(n.op_type == "Add" for n in opt3.nodes))

# ═══════════════════════════════════════════════════════════════
# Test 4: N02 — non-topological order must not duplicate outputs
# ═══════════════════════════════════════════════════════════════
g4 = GraphIR(
    inputs=["x", "w", "bias"],
    outputs=["y"],
    nodes=[
        GraphNode("add", "Add", ["mm_out", "bias"], ["y"]),   # consumer FIRST
        GraphNode("mm", "MatMul", ["x", "w"], ["mm_out"]),     # producer SECOND
    ],
    initializers={"w", "bias"},
)
assert g4.validate(), "graph should be a valid DAG"
opt4, stats4 = GraphPassPipeline(enable_fusion=True).run(g4)
log4 = stats4["Fusion"]["stats"]["fusion_log"]
fused4 = any(r["pattern"] == "FusedMatMulBias" for r in log4)
check("non-topo: still fused MatMul+Bias", fused4,
      f"fusions={[r['pattern'] for r in log4]}")
check("non-topo: validate", opt4.validate())
# Gather all output tensors from optimized nodes
all_outputs = []
for n in opt4.nodes:
    all_outputs.extend(n.outputs)
# No duplicate outputs
unique_ok = len(all_outputs) == len(set(all_outputs))
check("non-topo: no duplicate outputs", unique_ok,
      f"outputs={all_outputs}")
# Exactly one "y" output
y_count = sum(1 for o in all_outputs if o == "y")
check("non-topo: single 'y' output", y_count == 1,
      f"y_count={y_count}")


# ── summary ──
print(f"\nSummary: {ok}/{total} passed")
if ok == total:
    print("ALL PASSED!")
else:
    raise SystemExit(1)
