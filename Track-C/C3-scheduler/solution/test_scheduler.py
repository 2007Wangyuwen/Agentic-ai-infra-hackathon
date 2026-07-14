#!/usr/bin/env python3
"""Test C3.2/C3.3/C3.4 modules with real ONNX models."""

import json
from pathlib import Path

import onnx
from c3scheduler.graph import import_onnx_graph, GraphIR, GraphNode
from c3scheduler.strategy import strategy
from c3scheduler.graph_passes import GraphPassPipeline
from c3scheduler.memory import analyze_lifetimes, build_execution_plan


SOLUTION_DIR = Path(__file__).resolve().parent
TC_DIR = SOLUTION_DIR.parent / "testcases" / "release_to_competitors"
MODELS_DIR = TC_DIR / "models"


def test_model(model_path: str, label: str):
    print(f"\n{'='*60}")
    print(f"Testing: {label} ({model_path})")
    print(f"{'='*60}")

    graph = import_onnx_graph(model_path)
    print(f"[Graph] {len(graph.nodes)} nodes")

    # C3.2: Decomposition
    print(f"\nC3.2: Decomposition")
    total_kernels = 0
    for node in graph.nodes[:5]:  # show first 5
        prec = strategy.select_precision(node, graph)
        kernels = strategy.decompose(node, graph, prec)
        tuning = strategy.tune_kernel(kernels[0], prec, 1024)
        print(f"  {node.op_type:20s} -> prec={prec.precision}, kernels={len(kernels)}, block={tuning.block_x}, grid={tuning.grid_x}")
        total_kernels += len(kernels)
    for node in graph.nodes[5:]:
        prec = strategy.select_precision(node, graph)
        kernels = strategy.decompose(node, graph, prec)
        total_kernels += len(kernels)
    print(f"  ... total kernels after decomposition: {total_kernels}")

    # C3.3: Fusion
    print(f"\nC3.3: Fusion")
    pipeline = GraphPassPipeline(enable_fusion=True)
    optimized, stats = pipeline.run(graph)
    log = stats["Fusion"]["stats"]["fusion_log"]
    print(f"  Before: {len(graph.nodes)} nodes -> After: {len(optimized.nodes)} nodes")
    print(f"  Fusions applied: {len(log)}")
    for entry in log[:5]:
        print(f"    {entry['pattern']}: {entry['nodes']}")

    # C3.4: Memory planning
    print(f"\nC3.4: Memory Planning")
    lifetimes = analyze_lifetimes(optimized)
    plan = build_execution_plan(optimized)
    print(f"  Tensor lifetimes tracked: {len(lifetimes)}")
    print(f"  Execution plan steps: {len(plan)}")
    print(f"  Streams used: {set(s.stream for s in plan)}")
    print(f"  Weight preloads: {sum(1 for s in plan if s.kind == 'h2d_weight')}")

    print(f"\n[OK] {label} passed all stages")


if __name__ == "__main__":
    test_model(str(MODELS_DIR / "mlp_v1.onnx"), "MLP")
    test_model(str(MODELS_DIR / "resnet_v1.onnx"), "ResNet-18")
    test_model(str(MODELS_DIR / "transformer_v1.onnx"), "Transformer")
    print(f"\n{'='*60}")
    print("ALL MODELS PASSED C3.2 + C3.3 + C3.4")
    print(f"{'='*60}")
