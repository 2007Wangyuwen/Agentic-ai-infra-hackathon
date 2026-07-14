#!/usr/bin/env python3
"""Debug C3.5 standalone inference on synthetic model (fixed opset)."""
import json, os, sys, tempfile, time, shutil
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import onnx
from onnx import helper, TensorProto
from infer_worker import InferenceEngine

# ── create tiny model ──
tmp = tempfile.mkdtemp(prefix="c3d_")
print(f"Test dir: {tmp}")

inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 10])
out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", 5])
w = helper.make_tensor("w", TensorProto.FLOAT, [10, 5], [0.01] * 50)
b = helper.make_tensor("b", TensorProto.FLOAT, [5], [0.0] * 5)
n = helper.make_node("Gemm", ["input", "w", "b"], ["logits"], name="gemm")
g = helper.make_graph([n], "tiny", [inp], [out], initializer=[w, b])

# Fix: set opset to 20 (compatible with onnxruntime 1.27.0)
model = helper.make_model(g, producer_name="c3d")
model.opset_import[0].version = 20

onnx_path = os.path.join(tmp, "tiny.onnx")
onnx.save(model, onnx_path)

# ── input data ──
input_dir = os.path.join(tmp, "input")
os.makedirs(input_dir)
data = np.random.randn(32, 10).astype(np.float32)
np.save(os.path.join(input_dir, "input.npy"), data)
with open(os.path.join(input_dir, "manifest.json"), "w") as f:
    json.dump({"tensors": [{"name": "input", "file": "input.npy",
              "dtype": "float32", "shape": [32, 10]}]}, f)

output_dir = os.path.join(tmp, "output")

# ── run inference ──
print("Creating engine...")
engine = InferenceEngine()
print("Running infer...")
t0 = time.perf_counter()
result = engine.infer(onnx_path, input_dir, output_dir, batch_size=16)
elapsed = time.perf_counter() - t0

print(f"Result: {result}")
print(f"Time: {elapsed:.1f}s")

out_npy = os.path.join(output_dir, "logits.npy")
if os.path.exists(out_npy):
    logits = np.load(out_npy)
    print(f"Output shape: {logits.shape}, dtype: {logits.dtype}")
    assert logits.shape == (32, 5), f"Bad shape: {logits.shape}"
    assert logits.dtype == np.float32, f"Bad dtype: {logits.dtype}"
    print("Output OK!")
else:
    print("FAIL: no output file")

shutil.rmtree(output_dir, ignore_errors=True)
print(f"\nArtifacts at: {tmp}")
print("SUCCESS — InferenceEngine works!")
