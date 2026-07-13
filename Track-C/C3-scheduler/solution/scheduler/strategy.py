"""Kernel decomposition and tuning strategy for C3.2."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from .graph import GraphIR, GraphNode


@dataclass(frozen=True)
class PrecisionProfile:
    precision: str
    accumulator: str = "fp32"


@dataclass(frozen=True)
class KernelSpecRef:
    name: str
    inputs: List[str]
    outputs: List[str]


@dataclass(frozen=True)
class KernelTuningParams:
    block_x: int
    grid_x: int
    smem_bytes: int


@dataclass
class HardwareProfile:
    max_threads_per_block: int = 256
    smem_bytes: int = 48 * 1024
    precisions: List[str] = field(default_factory=lambda: ["fp32", "fp16", "fp8", "fp4"])

    def supported_precisions(self) -> List[str]:
        return list(self.precisions)


class DecompositionStrategy:
    def __init__(self, hardware: HardwareProfile | None = None):
        self.hardware = hardware or HardwareProfile()
        self._intermediate_id = 0

    def select_precision(self, node: GraphNode, graph: GraphIR) -> PrecisionProfile:
        sensitive = {
            "Softmax",
            "LayerNormalization",
            "BatchNormalization",
            "ReduceMax",
            "ReduceSum",
            "ReduceMean",
        }
        if node.op_type in sensitive:
            return PrecisionProfile("fp32")
        if node.op_type in {"Conv"}:
            return PrecisionProfile("fp16")
        if node.op_type in {"MatMul", "Gemm"}:
            return PrecisionProfile("fp16")
        if node.op_type in {"Relu", "Add", "Mul", "Div"}:
            return PrecisionProfile("fp8")
        if node.op_type in {"Flatten", "Reshape", "Transpose", "Split", "Gather", "Constant"}:
            return PrecisionProfile("fp32")
        return PrecisionProfile("fp32")

    def decompose(self, node: GraphNode, graph: GraphIR, precision: PrecisionProfile) -> List[KernelSpecRef]:
        p = precision.precision
        if node.op_type in {"Gemm", "MatMul"}:
            return [KernelSpecRef(f"matmul_{p}", node.inputs, node.outputs)]
        if node.op_type == "Conv":
            kernel = self._conv_kernel_name(node, graph, p)
            inter = self._intermediate()
            return [
                KernelSpecRef(kernel, node.inputs, [inter]),
                KernelSpecRef(f"conv_epilogue_{p}", [inter], node.outputs),
            ]
        if node.op_type == "Softmax":
            a, b, c = self._intermediate(), self._intermediate(), self._intermediate()
            x = node.inputs
            return [
                KernelSpecRef("reduce_max_fp32", x, [a]),
                KernelSpecRef("exp_fp32", [x[0], a], [b]),
                KernelSpecRef("reduce_sum_fp32", [b], [c]),
                KernelSpecRef("div_fp32", [b, c], node.outputs),
            ]
        if node.op_type == "LayerNormalization":
            mean, centered, sq, var = (
                self._intermediate(),
                self._intermediate(),
                self._intermediate(),
                self._intermediate(),
            )
            return [
                KernelSpecRef("reduce_mean_fp32", node.inputs[:1], [mean]),
                KernelSpecRef("sub_fp32", [node.inputs[0], mean], [centered]),
                KernelSpecRef("mul_fp32", [centered, centered], [sq]),
                KernelSpecRef("reduce_mean_fp32", [sq], [var]),
                KernelSpecRef("sqrt_fp32", [var], [self._intermediate()]),
                KernelSpecRef("layernorm_affine_fp32", node.inputs + [mean, var], node.outputs),
            ]
        if node.op_type in {"Relu", "Add", "Mul", "Div", "Erf"}:
            return [KernelSpecRef(f"elementwise_{node.op_type.lower()}_{p}", node.inputs, node.outputs)]
        if node.op_type in {"Flatten", "Reshape", "Transpose", "Split", "Gather", "GlobalAveragePool", "Constant"}:
            return [KernelSpecRef(f"{node.op_type.lower()}_{p}", node.inputs, node.outputs)]
        return [KernelSpecRef(f"generic_{node.op_type.lower()}_{p}", node.inputs, node.outputs)]

    def tune_kernel(
        self,
        ref: KernelSpecRef,
        precision: PrecisionProfile,
        problem_size: int | tuple[int, ...] | None = None,
    ) -> KernelTuningParams:
        elements = _problem_elements(problem_size)
        if ref.name.startswith("matmul"):
            block = 256
            smem = 16 * 1024
        elif ref.name.startswith("winograd") or ref.name.startswith("im2col"):
            block = 128
            smem = 24 * 1024
        elif ref.name.startswith("reduce"):
            block = 256
            smem = 4 * 1024
        else:
            block = 128
            smem = 0
        block = min(block, self.hardware.max_threads_per_block)
        grid = max(1, math.ceil(elements / block))
        smem = smem if smem <= self.hardware.smem_bytes else -1
        return KernelTuningParams(block_x=block, grid_x=grid, smem_bytes=smem)

    def _conv_kernel_name(self, node: GraphNode, graph: GraphIR, precision: str) -> str:
        kernel_shape = node.attributes.get("kernel_shape")
        strides = node.attributes.get("strides", [1, 1])
        if kernel_shape == [3, 3] and strides == [1, 1]:
            return f"winograd_forward_{precision}"
        return f"im2col_{precision}"

    def _intermediate(self) -> str:
        self._intermediate_id += 1
        return f"__c3_inter_{self._intermediate_id}__"


def _problem_elements(problem_size: int | tuple[int, ...] | None) -> int:
    if problem_size is None:
        return 1024
    if isinstance(problem_size, int):
        return max(1, problem_size)
    total = 1
    for value in problem_size:
        total *= max(1, int(value))
    return total


hardware = HardwareProfile()
strategy = DecompositionStrategy(hardware)
