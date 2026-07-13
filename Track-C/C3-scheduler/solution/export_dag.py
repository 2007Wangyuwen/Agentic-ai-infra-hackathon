#!/usr/bin/env python3
import argparse
import json
import onnx
from onnx import TensorProto


def dtype_to_str(elem_type):
    try:
        return TensorProto.DataType.Name(elem_type)
    except Exception:
        return str(elem_type)


def parse_onnx(model_path):
    model = onnx.load(model_path)
    graph = model.graph

    initializer_names = set(init.name for init in graph.initializer)

    graph_inputs = []
    for x in graph.input:
        if x.name not in initializer_names:
            t = x.type.tensor_type
            shape = []
            if t.HasField("shape"):
                for d in t.shape.dim:
                    if d.dim_param:
                        shape.append(d.dim_param)
                    elif d.dim_value:
                        shape.append(int(d.dim_value))
                    else:
                        shape.append("unknown")
            graph_inputs.append({
                "name": x.name,
                "dtype": dtype_to_str(t.elem_type),
                "shape": shape,
            })

    graph_outputs = []
    for x in graph.output:
        t = x.type.tensor_type
        shape = []
        if t.HasField("shape"):
            for d in t.shape.dim:
                if d.dim_param:
                    shape.append(d.dim_param)
                elif d.dim_value:
                    shape.append(int(d.dim_value))
                else:
                    shape.append("unknown")
        graph_outputs.append({
            "name": x.name,
            "dtype": dtype_to_str(t.elem_type),
            "shape": shape,
        })

    nodes = []
    producer = {}

    for idx, node in enumerate(graph.node):
        name = node.name if node.name else f"{node.op_type}_{idx}"
        nodes.append({
            "name": name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
        })
        for out in node.output:
            producer[out] = name

    edges = []
    for idx, node in enumerate(graph.node):
        dst = node.name if node.name else f"{node.op_type}_{idx}"
        for inp in node.input:
            if inp in producer:
                edges.append({"src": producer[inp], "dst": dst, "tensor": inp})

    return {
        "format_version": "1.0",
        "graph_inputs": graph_inputs,
        "graph_outputs": graph_outputs,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    dag = parse_onnx(args.onnx)
    with open(args.output, "w") as f:
        json.dump(dag, f, indent=2)
    print(f"[OK] {args.output}: {dag['num_nodes']} nodes, {dag['num_edges']} edges")


if __name__ == "__main__":
    main()
