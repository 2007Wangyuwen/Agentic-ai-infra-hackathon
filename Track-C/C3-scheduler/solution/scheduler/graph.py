"""Graph IR representation for C3."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Union

import onnx
from onnx import helper


def _onnx_attribute_value(attr):
    """Generic ONNX attribute → Python value, covering all types."""
    value = helper.get_attribute_value(attr)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def import_onnx_graph(model_path: Union[str, Path]) -> "GraphIR":
    """Parse an ONNX model into a GraphIR.  Called by the official benchmark.

    Extracts inputs (excluding initializers), outputs, nodes with their
    op_type / inputs / outputs / attributes, and preserves kernel_shape /
    strides for Conv2d decomposition.
    """
    model = onnx.load(str(model_path))
    graph = model.graph
    initializer_names = {init.name for init in graph.initializer}
    inputs = [x.name for x in graph.input if x.name not in initializer_names]
    outputs = [x.name for x in graph.output]
    nodes = []
    for idx, node in enumerate(graph.node):
        name = node.name if node.name else f"{node.op_type}_{idx}"
        attrs = {attr.name: _onnx_attribute_value(attr) for attr in node.attribute}
        nodes.append(GraphNode(
            name=name, op_type=node.op_type,
            inputs=list(node.input), outputs=list(node.output),
            attributes=attrs,
        ))
    return GraphIR(inputs=inputs, outputs=outputs, nodes=nodes, initializers=initializer_names)


# ═══════════════════════════════════════════════════════════════════
# Graph IR data structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GraphNode:
    name: str
    op_type: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    attributes: Dict = field(default_factory=dict)


@dataclass
class GraphIR:
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    nodes: List[GraphNode] = field(default_factory=list)
    initializers: Set[str] = field(default_factory=set)

    # ── helpers ───────────────────────────────────────────────────

    def consumer_map(self) -> Dict[str, List[GraphNode]]:
        """Build tensor → consumer nodes mapping."""
        consumers: Dict[str, List[GraphNode]] = {}
        for node in self.nodes:
            for tensor in node.inputs:
                consumers.setdefault(tensor, []).append(node)
        return consumers

    # ── validation ────────────────────────────────────────────────

    @staticmethod
    def _is_resolvable(tensor: str, graph_inputs: set, initializers: set,
                       producer: dict) -> bool:
        if not tensor:
            return True          # empty string — ONNX optional input
        if tensor in graph_inputs:
            return True
        if tensor in initializers:
            return True
        return tensor in producer

    def validate(self) -> bool:
        """Full structural validation: uniqueness, single-producer,
        resolvable references, and acyclic dependencies.

        Called by the official benchmark during C3.3 fusion scoring.
        """
        nodes = list(self.nodes)

        # ── 1. Node names must be unique ──
        node_names = [node.name for node in nodes]
        if len(node_names) != len(set(node_names)):
            return False

        graph_inputs: set = {t for t in self.inputs if t}
        initializers: set = getattr(self, "initializers", set())

        # ── 2. Each tensor has at most one producer ──
        producer: Dict[str, str] = {}
        for node in nodes:
            for tensor in node.outputs:
                if not tensor:
                    continue
                if tensor in producer:
                    return False
                producer[tensor] = node.name

        # ── 3. Every consumed tensor must be resolvable ──
        for node in nodes:
            for tensor in node.inputs:
                if not tensor:
                    continue
                if not self._is_resolvable(tensor, graph_inputs,
                                           initializers, producer):
                    return False

        for tensor in self.outputs:
            if not tensor:
                continue
            if not self._is_resolvable(tensor, graph_inputs,
                                       initializers, producer):
                return False

        # ── 4. Build dependency graph + Kahn topological sort ──
        successors: Dict[str, set] = {node.name: set() for node in nodes}
        indegree: Dict[str, int] = {node.name: 0 for node in nodes}

        for node in nodes:
            deps: set = set()
            for tensor in node.inputs:
                src = producer.get(tensor)
                if src is None:
                    continue
                if src == node.name:            # self-loop
                    return False
                deps.add(src)
            for src in deps:
                successors[src].add(node.name)
                indegree[node.name] += 1

        ready = [name for name, d in indegree.items() if d == 0]
        visited = 0
        while ready:
            cur = ready.pop()
            visited += 1
            for nxt in successors[cur]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    ready.append(nxt)

        return visited == len(nodes)
