#!/usr/bin/env python3
"""Negative-tests for GraphIR.validate() — C3.3 fusion scoring requirement."""

from scheduler.graph import GraphIR, GraphNode

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


# ═════════════════════════════════════════════════════════════════
# Test 1: valid DAG  — input → A → B → output
# ═════════════════════════════════════════════════════════════════
g = GraphIR(
    inputs=["input"],
    outputs=["out"],
    nodes=[
        GraphNode("a", "Add", ["input", "w"], ["mid"]),
        GraphNode("b", "Relu", ["mid"], ["out"]),
    ],
    initializers={"w"},
)
check("valid DAG", g.validate())

# ═════════════════════════════════════════════════════════════════
# Test 2: cycle  — A consumes B_out, B consumes A_out
# ═════════════════════════════════════════════════════════════════
g_cycle = GraphIR(
    inputs=["x"],
    outputs=["y"],
    nodes=[
        GraphNode("a", "Add", ["x", "b_out"], ["a_out"]),
        GraphNode("b", "Add", ["a_out"], ["b_out", "y"]),
    ],
    initializers=set(),
)
check("cycle rejected", not g_cycle.validate())

# ═════════════════════════════════════════════════════════════════
# Test 3: undefined tensor  — A consumes "missing_tensor"
# ═════════════════════════════════════════════════════════════════
g_missing = GraphIR(
    inputs=["input"],
    outputs=["out"],
    nodes=[
        GraphNode("a", "Identity", ["missing_tensor"], ["out"]),
    ],
    initializers=set(),
)
check("undefined tensor rejected", not g_missing.validate())

# ═════════════════════════════════════════════════════════════════
# Test 4: graph output never produced
# ═════════════════════════════════════════════════════════════════
g_unproduced_out = GraphIR(
    inputs=["x"],
    outputs=["y"],         # "y" is never produced by any node
    nodes=[
        GraphNode("a", "Relu", ["x"], ["mid"]),
    ],
    initializers=set(),
)
check("missing graph output rejected", not g_unproduced_out.validate())

# ── summary ──
print(f"\nSummary: {ok}/{total} passed")
if ok == total:
    print("ALL PASSED!")
