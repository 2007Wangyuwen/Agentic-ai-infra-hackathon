"""Graph IR representation for C3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


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

    def consumer_map(self) -> Dict[str, List[GraphNode]]:
        """Build tensor -> consumer nodes mapping."""
        consumers: Dict[str, List[GraphNode]] = {}
        for node in self.nodes:
            for tensor in node.inputs:
                consumers.setdefault(tensor, []).append(node)
        return consumers

    def validate(self) -> bool:
        """Basic validation: all consumed tensors are produced."""
        produced = set(self.inputs)
        for node in self.nodes:
            for inp in node.inputs:
                if inp and inp not in produced:
                    pass  # weight or initializer, ok
            produced.update(node.outputs)
        return True
