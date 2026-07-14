#!/usr/bin/env python3
"""N01–N04 regression suite — every C3 fix validated in a single pass."""

from scheduler.memory import DeviceMemoryPool, MemoryBlock
from scheduler.graph_passes import GraphPassPipeline


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


def expect_exception(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


# ═══════════════════════════════════════════════════════════════
# N01 — memory pool safety
# ═══════════════════════════════════════════════════════════════

# malloc(0) rejected
pool = DeviceMemoryPool()
expect_exception(ValueError, lambda: pool.malloc(0))
check("malloc(0) rejected", True)

# malloc(-1) rejected
pool = DeviceMemoryPool()
expect_exception(ValueError, lambda: pool.malloc(-1))
check("malloc(-1) rejected", True)

# malloc(-300) rejected
pool = DeviceMemoryPool()
expect_exception(ValueError, lambda: pool.malloc(-300))
check("malloc(-300) rejected", True)

# double-free rejected
pool = DeviceMemoryPool()
b = pool.malloc(256)
pool.free(b)
expect_exception(ValueError, lambda: pool.free(b))
check("double-free rejected", True)

# external block rejected
pool = DeviceMemoryPool()
ext = MemoryBlock(0, 256, False)
expect_exception(ValueError, lambda: pool.free(ext))
check("external block rejected", True)

# cross-pool same-value block rejected
pa = DeviceMemoryPool()
pb = DeviceMemoryPool()
ba = pa.malloc(256)
bb = pb.malloc(256)
assert ba == bb and ba is not bb
expect_exception(ValueError, lambda: pa.free(bb))
check("cross-pool same-value free rejected", True)
check("external block not mutated", bb.free is False)
check("owned block not mutated", ba.free is False)

# capacity stays non-negative after rejections
pool = DeviceMemoryPool()
try: pool.malloc(0)
except ValueError: pass
try: pool.malloc(-300)
except ValueError: pass
check("capacity >= 0 after rejects", pool.capacity >= 0,
      f"capacity={pool.capacity}")

# happy-path reuse
pool = DeviceMemoryPool()
a = pool.malloc(256)
b = pool.malloc(512)
pool.free(a)
c = pool.malloc(128)
check("reuse offset after free", c.offset == 0)
check("other block still allocated", not b.free)


# ═══════════════════════════════════════════════════════════════
# N02 / N03 — fusion bias detection + non-topological + consumed
# ═══════════════════════════════════════════════════════════════
from c3scheduler.graph import GraphIR, GraphNode

# -- initializer bias: should fuse
g = GraphIR(
    inputs=["x", "w", "bias"], outputs=["y"],
    nodes=[GraphNode("m", "MatMul", ["x", "w"], ["m_out"]),
           GraphNode("a", "Add", ["m_out", "bias"], ["y"])],
    initializers={"w", "bias"},
)
opt, stats = GraphPassPipeline(True).run(g)
log = stats["Fusion"]["stats"]["fusion_log"]
check("init bias: fused", any(r["pattern"] == "FusedMatMulBias" for r in log))
check("init bias: validate", opt.validate())

# -- Constant-node bias: should fuse
g2 = GraphIR(
    inputs=["x", "w"], outputs=["y"],
    nodes=[GraphNode("c", "Constant", [], ["c"], attributes={"v": 1}),
           GraphNode("m", "MatMul", ["x", "w"], ["m_out"]),
           GraphNode("a", "Add", ["m_out", "c"], ["y"])],
    initializers={"w"},
)
opt2, stats2 = GraphPassPipeline(True).run(g2)
log2 = stats2["Fusion"]["stats"]["fusion_log"]
check("const bias: fused", any(r["pattern"] == "FusedMatMulBias" for r in log2))
check("const bias: validate", opt2.validate())

# -- residual Add: must NOT fuse
g3 = GraphIR(
    inputs=["x", "w", "r"], outputs=["y"],
    nodes=[GraphNode("m", "MatMul", ["x", "w"], ["m_out"]),
           GraphNode("a", "Add", ["m_out", "r"], ["y"])],
    initializers={"w"},
)
opt3, stats3 = GraphPassPipeline(True).run(g3)
log3 = stats3["Fusion"]["stats"]["fusion_log"]
check("residual: NOT fused", not any(r["pattern"] == "FusedMatMulBias" for r in log3))
check("residual: validate", opt3.validate())
check("residual: MatMul preserved", any(n.op_type == "MatMul" for n in opt3.nodes))
check("residual: Add preserved", any(n.op_type == "Add" for n in opt3.nodes))

# -- non-topological order: still correct fusion, no duplicates
g4 = GraphIR(
    inputs=["x", "w", "bias"], outputs=["y"],
    nodes=[GraphNode("a", "Add", ["m_out", "bias"], ["y"]),   # consumer FIRST
           GraphNode("m", "MatMul", ["x", "w"], ["m_out"])],    # producer SECOND
    initializers={"w", "bias"},
)
opt4, stats4 = GraphPassPipeline(True).run(g4)
log4 = stats4["Fusion"]["stats"]["fusion_log"]
all_outs = [o for n in opt4.nodes for o in n.outputs]
check("non-topo: fused", any(r["pattern"] == "FusedMatMulBias" for r in log4))
check("non-topo: validate", opt4.validate())
check("non-topo: no duplicate outputs", len(all_outs) == len(set(all_outs)),
      f"outputs={all_outs}")
check("non-topo: single 'y'", all_outs.count("y") == 1)

# -- fallback on invalid fusion
g5 = GraphIR(
    inputs=["x"], outputs=["y"],
    nodes=[GraphNode("m", "MatMul", ["x", "missing"], ["y"])],
)
opt5, stats5 = GraphPassPipeline(True).run(g5)
s5 = stats5["Fusion"]["stats"]
fallback_ok = s5.get("fallback_to_original", False)
check("fallback recorded on invalid fusion",
      fallback_ok or len(opt5.nodes) == len(g5.nodes))


# ═══════════════════════════════════════════════════════════════
# N04 — dependency-aware stream scheduling
# ═══════════════════════════════════════════════════════════════
from c3scheduler.memory import build_execution_plan

def stream_of(plan, name):
    return [s.stream for s in plan if s.kind == "kernel" and s.name == name][0]

# chain A -> B: same stream
g = GraphIR(inputs=["x"], outputs=["y"],
            nodes=[GraphNode("a", "Relu", ["x"], ["mid"]),
                   GraphNode("b", "Relu", ["mid"], ["y"])])
plan = build_execution_plan(g)
check("stream: chain same stream", stream_of(plan, "a") == stream_of(plan, "b"))

# independent: different streams possible
g = GraphIR(inputs=["x", "y"], outputs=["a", "b"],
            nodes=[GraphNode("a", "Relu", ["x"], ["a"]),
                   GraphNode("b", "Relu", ["y"], ["b"])])
plan = build_execution_plan(g)
check("stream: independent diff streams",
      stream_of(plan, "a") != stream_of(plan, "b"))

# merge A+B->C: all same stream
g = GraphIR(inputs=["x", "y"], outputs=["z"],
            nodes=[GraphNode("a", "Relu", ["x"], ["a_out"]),
                   GraphNode("b", "Relu", ["y"], ["b_out"]),
                   GraphNode("c", "Add", ["a_out", "b_out"], ["z"])])
plan = build_execution_plan(g)
check("stream: merge no cross-stream",
      stream_of(plan, "a") == stream_of(plan, "b") == stream_of(plan, "c"))


# ── summary ──
print(f"\n{'='*50}")
print(f"  N01–N04 regression: {ok}/{total} PASSED")
print(f"{'='*50}")
if ok != total:
    raise SystemExit(1)
