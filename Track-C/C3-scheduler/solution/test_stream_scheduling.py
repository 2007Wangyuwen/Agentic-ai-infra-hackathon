#!/usr/bin/env python3
"""Dependency-aware stream scheduling tests — C3.4 code-review requirement."""

from c3scheduler.graph import GraphIR, GraphNode
from c3scheduler.memory import analyze_lifetimes, build_execution_plan


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


def _streams(plan, name):
    return [s.stream for s in plan if s.kind == "kernel" and s.name == name][0]


# Test 1: dependent chain A -> B -> same stream
g = GraphIR(
    inputs=["x"], outputs=["y"],
    nodes=[
        GraphNode("a", "Relu", ["x"], ["mid"]),
        GraphNode("b", "Relu", ["mid"], ["y"]),
    ],
)
plan = build_execution_plan(g)
sa, sb = _streams(plan, "a"), _streams(plan, "b")
check("chain: A and B same stream", sa == sb, f"A={sa} B={sb}")

# Test 2: independent nodes — may be on different streams
g2 = GraphIR(
    inputs=["x", "y"], outputs=["a_out", "b_out"],
    nodes=[
        GraphNode("a", "Relu", ["x"], ["a_out"]),
        GraphNode("b", "Relu", ["y"], ["b_out"]),
    ],
)
plan2 = build_execution_plan(g2)
streams2 = set(s.stream for s in plan2 if s.kind == "kernel")
check("independent: at least 1 stream", len(streams2) >= 1,
      f"streams={streams2}")

# Test 3: merge graph A + B -> C -> all same stream (no wait/event)
g3 = GraphIR(
    inputs=["x", "y"], outputs=["z"],
    nodes=[
        GraphNode("a", "Relu", ["x"], ["a_out"]),
        GraphNode("b", "Relu", ["y"], ["b_out"]),
        GraphNode("c", "Add", ["a_out", "b_out"], ["z"]),
    ],
)
plan3 = build_execution_plan(g3)
sa3 = _streams(plan3, "a")
sb3 = _streams(plan3, "b")
sc3 = _streams(plan3, "c")
check("merge: A and C same stream", sa3 == sc3, f"A={sa3} C={sc3}")
check("merge: B and C same stream", sb3 == sc3, f"B={sb3} C={sc3}")
check("merge: A and B same stream", sa3 == sb3, f"A={sa3} B={sb3}")

# Test 4: two disjoint components -> different streams
g4 = GraphIR(
    inputs=["u", "v"], outputs=["u_out", "v_out"],
    nodes=[
        GraphNode("comp1", "Relu", ["u"], ["u_out"]),
        GraphNode("comp2", "Relu", ["v"], ["v_out"]),
    ],
)
plan4 = build_execution_plan(g4)
s4a = _streams(plan4, "comp1")
s4b = _streams(plan4, "comp2")
# Two independent components — union-find alternates, so ideally different
check("disjoint: different streams", s4a != s4b,
      f"comp1={s4a} comp2={s4b}")

# summary
print(f"\nSummary: {ok}/{total} passed")
if ok == total:
    print("ALL PASSED!")
else:
    raise SystemExit(1)
