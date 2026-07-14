"""Graph optimization and fusion pass pipeline for C3.3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .graph import GraphIR, GraphNode


@dataclass
class FusionRecord:
    pattern: str
    nodes: List[str]
    fused_node: str


@dataclass
class GraphPassPipeline:
    enable_fusion: bool = True

    def run(self, graph: GraphIR):
        if not self.enable_fusion:
            return graph, {"Fusion": {"stats": {"fusion_log": []}}}
        optimized, log = fuse_graph(graph)

        result_stats: dict = {"fusion_log": [record.__dict__ for record in log]}
        if not optimized.validate():
            import copy
            result_stats["validation_failed"] = True
            result_stats["fallback_to_original"] = True
            optimized = copy.deepcopy(graph)
        return optimized, {"Fusion": {"stats": result_stats}}


def _stable_topological_nodes(graph: GraphIR) -> List[GraphNode]:
    """Return graph nodes in stable topological order (producer before consumer).

    Necessary because *fuse_graph* scans linearly and relies on the invariant
    that a producer appears before its consumer.  A valid DAG whose node list
    is not topologically sorted can cause duplicate-fusion artifacts.
    """
    nodes = list(graph.nodes)
    node_by = {n.name: n for n in nodes}
    orig_idx = {n.name: i for i, n in enumerate(nodes)}

    # Build producer map
    producer: Dict[str, str] = {}
    for n in nodes:
        for t in n.outputs:
            if t:
                producer[t] = n.name

    # Build dependency graph
    successors: Dict[str, set] = {n.name: set() for n in nodes}
    indegree: Dict[str, int] = {n.name: 0 for n in nodes}
    for n in nodes:
        deps: set = set()
        for t in n.inputs:
            src = producer.get(t)
            if src is not None and src != n.name:
                deps.add(src)
        for src in deps:
            if n.name not in successors[src]:
                successors[src].add(n.name)
                indegree[n.name] += 1

    # Kahn with stable tie-breaking (original index)
    ready = sorted((name for name, d in indegree.items() if d == 0),
                   key=orig_idx.get)
    result: List[GraphNode] = []
    while ready:
        cur = ready.pop(0)
        result.append(node_by[cur])
        new_ready = []
        for nxt in successors[cur]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                new_ready.append(nxt)
        ready.extend(sorted(new_ready, key=orig_idx.get))

    if len(result) != len(nodes):
        raise ValueError("fuse_graph: cyclic dependency detected")
    return result


def fuse_graph(graph: GraphIR) -> tuple[GraphIR, List[FusionRecord]]:
    nodes = _stable_topological_nodes(graph)
    consumers = graph.consumer_map()
    new_nodes: List[GraphNode] = []
    fusion_log: List[FusionRecord] = []
    consumed: set[str] = set()
    index = 0
    while index < len(nodes):
        node = nodes[index]
        if node.name in consumed:
            index += 1
            continue
        next_node = _single_consumer_next(node, consumers)
        if node.op_type in {"MatMul", "Gemm"} and next_node and _is_add_bias(next_node, node, graph):
            fused = _make_fused("FusedMatMulBias", [node, next_node])
            new_nodes.append(fused)
            fusion_log.append(FusionRecord("FusedMatMulBias", [node.name, next_node.name], fused.name))
            consumed.add(node.name)
            consumed.add(next_node.name)
        elif node.op_type == "Conv" and next_node and next_node.op_type == "Relu":
            fused = _make_fused("FusedConvRelu", [node, next_node])
            new_nodes.append(fused)
            fusion_log.append(FusionRecord("FusedConvRelu", [node.name, next_node.name], fused.name))
            consumed.add(node.name)
            consumed.add(next_node.name)
        elif node.op_type == "Conv" and next_node and next_node.op_type == "BatchNormalization":
            fused = _make_fused("FusedConv2dBatchNorm", [node, next_node])
            new_nodes.append(fused)
            fusion_log.append(FusionRecord("FusedConv2dBatchNorm", [node.name, next_node.name], fused.name))
            consumed.add(node.name)
            consumed.add(next_node.name)
        elif node.op_type == "Softmax" and next_node and next_node.op_type == "Dropout":
            fused = _make_fused("FusedSoftmaxDropout", [node, next_node])
            new_nodes.append(fused)
            fusion_log.append(FusionRecord("FusedSoftmaxDropout", [node.name, next_node.name], fused.name))
            consumed.add(node.name)
            consumed.add(next_node.name)
        elif node.op_type == "Add" and next_node and next_node.op_type == "LayerNormalization":
            fused = _make_fused("FusedResidualNorm", [node, next_node])
            new_nodes.append(fused)
            fusion_log.append(FusionRecord("FusedResidualNorm", [node.name, next_node.name], fused.name))
            consumed.add(node.name)
            consumed.add(next_node.name)
        else:
            chain = _elementwise_chain(nodes, index, consumers)
            if len(chain) >= 2:
                fused = _make_fused("FusedEWChain", chain)
                new_nodes.append(fused)
                fusion_log.append(FusionRecord("FusedEWChain", [n.name for n in chain], fused.name))
                consumed.update(n.name for n in chain)
            else:
                new_nodes.append(node)
        index += 1
    return GraphIR(inputs=list(graph.inputs), outputs=list(graph.outputs), nodes=new_nodes,
                   initializers=getattr(graph, "initializers", set())), fusion_log


def _single_consumer_next(node: GraphNode, consumers: Dict[str, List[GraphNode]]):
    if not node.outputs:
        return None
    uses = consumers.get(node.outputs[0], [])
    return uses[0] if len(uses) == 1 else None


def _constant_tensor_names(graph: GraphIR) -> set[str]:
    """Collect every tensor that is a constant (initializer or Constant op output)."""
    names: set[str] = set()

    # ONNX initializers (weights, biases)
    initializers = getattr(graph, "initializers", None)
    if isinstance(initializers, dict):
        names.update(initializers.keys())
    elif initializers is not None:
        for value in initializers:
            name = getattr(value, "name", None) if not isinstance(value, str) else value
            if name:
                names.add(name)

    # Constant op nodes (e.g. attention scale factor)
    for node in graph.nodes:
        if node.op_type == "Constant":
            names.update(o for o in node.outputs if o)

    return names


def _is_add_bias(add_node: GraphNode, mm_node: GraphNode, graph: GraphIR) -> bool:
    """True if *add_node* is a bias addition consuming *mm_node*'s output.

    The Add must have exactly 2 inputs, one from the MatMul/Gemm and the
    other from the constant-tensor set (initializer or Constant op).
    A residual Add whose second operand is a graph input or intermediate
    tensor is rejected.
    """
    if add_node.op_type != "Add" or len(add_node.inputs) != 2:
        return False

    mm_out = set(mm_node.outputs)
    other_inputs = [t for t in add_node.inputs if t not in mm_out]
    if not other_inputs:
        return False

    constants = _constant_tensor_names(graph)
    return other_inputs[0] in constants


def _elementwise_chain(nodes: List[GraphNode], start: int, consumers: Dict[str, List[GraphNode]]) -> List[GraphNode]:
    elementwise = {"Add", "Mul", "Div", "Relu", "Erf"}
    chain = []
    current = nodes[start]
    while current.op_type in elementwise and len(chain) < 5:
        chain.append(current)
        nxt = _single_consumer_next(current, consumers)
        if nxt is None or nxt.op_type not in elementwise:
            break
        current = nxt
    return chain


def _make_fused(pattern: str, nodes: List[GraphNode]) -> GraphNode:
    first, last = nodes[0], nodes[-1]
    internal = {tensor for node in nodes[:-1] for tensor in node.outputs}
    inputs = []
    for node in nodes:
        for tensor in node.inputs:
            if tensor not in internal and tensor not in inputs:
                inputs.append(tensor)
    return GraphNode(
        name=f"{pattern}::{'+'.join(node.name for node in nodes)}",
        op_type=pattern,
        inputs=inputs or list(first.inputs),
        outputs=list(last.outputs),
        attributes={"fused_ops": [node.op_type for node in nodes]},
    )

