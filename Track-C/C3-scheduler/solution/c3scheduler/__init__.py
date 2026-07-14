"""Compatibility package — re-exports scheduler under the expected name."""
from scheduler.graph import GraphIR, GraphNode, import_onnx_graph
from scheduler.strategy import DecompositionStrategy, HardwareProfile, strategy, hardware
from scheduler.graph_passes import GraphPassPipeline
from scheduler.memory import DeviceMemoryPool, analyze_lifetimes, build_execution_plan
