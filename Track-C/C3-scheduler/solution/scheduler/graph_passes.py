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
        optimized.validate()
        return optimized, {"Fusion": {"stats": {"fusion_log": [record.__dict__ for record in log]}}}


def fuse_graph(graph: GraphIR) -> tuple[GraphIR, List[FusionRecord]]:
    consumers = graph.consumer_map()
    new_nodes: List[GraphNode] = []
    fusion_log: List[FusionRecord] = []
    skip = set()
    index = 0
    while index < len(graph.nodes):
        node = graph.nodes[index]
        if node.name in skip:
            index += 1
            continue
        next_node = _single_consumer_next(node, consumers)
        if node.op_type in {"MatMul", "Gemm"} and next_node and _is_add_bias(next_node):
            fused = _make_fused("FusedMatMulBias", [node, next_node])
            new_nodes.append(fused)
            fusion_log.append(FusionRecord("FusedMatMulBias", [node.name, next_node.name], fused.name))
            skip.add(next_node.name)
        elif node.op_type == "Conv" and next_node and next_node.op_type == "BatchNormalization":
            fused = _make_fused("FusedConv2dBatchNorm", [node, next_node])
            new_nodes.append(fused)
            fusion_log.append(FusionRecord("FusedConv2dBatchNorm", [node.name, next_node.name], fused.name))
            skip.add(next_node.name)
        elif node.op_type == "Softmax" and next_node and next_node.op_type == "Dropout":
            fused = _make_fused("FusedSoftmaxDropout", [node, next_node])
            new_nodes.append(fused)
            fusion_log.append(FusionRecord("FusedSoftmaxDropout", [node.name, next_node.name], fused.name))
            skip.add(next_node.name)
        elif node.op_type == "Add" and next_node and next_node.op_type == "LayerNormalization":
            fused = _make_fused("FusedResidualNorm", [node, next_node])
            new_nodes.append(fused)
            fusion_log.append(FusionRecord("FusedResidualNorm", [node.name, next_node.name], fused.name))
            skip.add(next_node.name)
        else:
            chain = _elementwise_chain(graph.nodes, index, consumers)
            if len(chain) >= 2:
                fused = _make_fused("FusedEWChain", chain)
                new_nodes.append(fused)
                fusion_log.append(FusionRecord("FusedEWChain", [n.name for n in chain], fused.name))
                skip.update(n.name for n in chain[1:])
            else:
                new_nodes.append(node)
        index += 1
    return GraphIR(inputs=list(graph.inputs), outputs=list(graph.outputs), nodes=new_nodes), fusion_log


def _single_consumer_next(node: GraphNode, consumers: Dict[str, List[GraphNode]]):
    if not node.outputs:
        return None
    uses = consumers.get(node.outputs[0], [])
    return uses[0] if len(uses) == 1 else None


def _is_add_bias(node: GraphNode) -> bool:
    return node.op_type == "Add" and len(node.inputs) == 2


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

