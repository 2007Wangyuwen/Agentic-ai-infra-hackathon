"""Memory planning and scheduling primitives for C3.4 code review."""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .graph import GraphIR


@dataclass
class MemoryBlock:
    offset: int
    size: int
    free: bool = False


class DeviceMemoryPool:
    """Best-fit pool with coalescing free blocks."""

    def __init__(self):
        self.blocks: List[MemoryBlock] = []
        self.capacity = 0

    def malloc(self, size: int) -> MemoryBlock:
        # NumPy-safe guard — must precede _align to avoid negative→-256 conversion.
        if isinstance(size, bool) or not isinstance(size, numbers.Integral):
            raise TypeError("allocation size must be an integer")
        size = int(size)
        if size <= 0:
            raise ValueError("allocation size must be positive")
        size = _align(size)
        candidates = [block for block in self.blocks if block.free and block.size >= size]
        if candidates:
            block = min(candidates, key=lambda item: item.size)
            block.free = False
            if block.size >= size + 256:
                tail = MemoryBlock(block.offset + size, block.size - size, True)
                block.size = size
                self.blocks.insert(self.blocks.index(block) + 1, tail)
            return block
        block = MemoryBlock(self.capacity, size, False)
        self.blocks.append(block)
        self.capacity += size
        return block

    def free(self, block: MemoryBlock) -> None:
        # Identity check — dataclass __eq__ is value-based; use `is`.
        owned = next((c for c in self.blocks if c is block), None)
        if owned is None:
            raise ValueError("DeviceMemoryPool.free(): block does not belong to this pool")
        if owned.free:
            raise ValueError("DeviceMemoryPool.free(): double free detected")
        owned.free = True
        self._coalesce()

    def _coalesce(self) -> None:
        merged: List[MemoryBlock] = []
        for block in sorted(self.blocks, key=lambda item: item.offset):
            if merged and merged[-1].free and block.free and merged[-1].offset + merged[-1].size == block.offset:
                merged[-1].size += block.size
            else:
                merged.append(block)
        self.blocks = merged


@dataclass
class TensorLifetime:
    first_use: int
    last_use: int
    slot: Optional[int] = None


@dataclass
class PlanStep:
    kind: str
    name: str
    stream: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


def analyze_lifetimes(graph: GraphIR) -> Dict[str, TensorLifetime]:
    lifetimes: Dict[str, TensorLifetime] = {}
    for index, node in enumerate(graph.nodes):
        for tensor in node.outputs:
            lifetimes.setdefault(tensor, TensorLifetime(index, index))
        for tensor in node.inputs:
            if tensor in lifetimes:
                lifetimes[tensor].last_use = index
    return lifetimes


def assign_reuse_slots(lifetimes: Dict[str, TensorLifetime]) -> Dict[str, int]:
    active: List[tuple[int, int]] = []
    free_slots: List[int] = []
    next_slot = 0
    for tensor, lifetime in sorted(lifetimes.items(), key=lambda item: item[1].first_use):
        still_active = []
        for last_use, slot in active:
            if last_use < lifetime.first_use:
                free_slots.append(slot)
            else:
                still_active.append((last_use, slot))
        active = still_active
        slot = free_slots.pop() if free_slots else next_slot
        if slot == next_slot:
            next_slot += 1
        lifetime.slot = slot
        active.append((lifetime.last_use, slot))
    return {tensor: lifetime.slot or 0 for tensor, lifetime in lifetimes.items()}


def build_execution_plan(graph: GraphIR) -> List[PlanStep]:
    """Create a schedule with weight preload and dependency-aware compute streams.

    Stream assignment uses union-find on the undirected data-dependency graph:
    nodes connected by any producer-consumer edge share one component;
    independent components get alternating compute streams.
    """
    lifetimes = analyze_lifetimes(graph)
    assign_reuse_slots(lifetimes)

    # ── Union-find: connected by data dependency → same component ──
    parent = {n.name: n.name for n in graph.nodes}

    def _root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a, b):
        ra, rb = _root(a), _root(b)
        if ra != rb:
            parent[ra] = rb

    # Build producer map
    producer: dict = {}
    for n in graph.nodes:
        for t in n.outputs:
            if t:
                producer[t] = n.name
    # Union producer-consumer pairs
    for n in graph.nodes:
        for t in n.inputs:
            src = producer.get(t)
            if src is not None and src != n.name:
                _union(src, n.name)

    # Assign alternating stream per connected component
    component_stream: dict = {}
    stream_idx = 0
    for n in graph.nodes:
        root = _root(n.name)
        if root not in component_stream:
            component_stream[root] = f"compute_{stream_idx % 2}"
            stream_idx += 1

    # ── Build execution plan ──
    steps: List[PlanStep] = []
    produced = set(graph.inputs)
    for node in graph.nodes:
        weight_inputs = [name for name in node.inputs if name not in produced and name]
        for weight in weight_inputs:
            steps.append(PlanStep("h2d_weight", weight, "copy_prefetch"))
            produced.add(weight)
        stream = component_stream[_root(node.name)]
        steps.append(PlanStep("kernel", node.name, stream, list(node.inputs), list(node.outputs)))
        produced.update(node.outputs)
    return steps


def _align(size: int, alignment: int = 256) -> int:
    return ((int(size) + alignment - 1) // alignment) * alignment
