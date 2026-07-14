#!/usr/bin/env python3
"""End-to-end Identity operator test: DAG export + inference + numerical verification."""
import json, sys
import numpy as np
import onnx
from onnx import helper, TensorProto
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TMP = Path("/tmp/_c3_identity_test")
TMP.mkdir(exist_ok=True)
INP = TMP / "input"
INP.mkdir(exist_ok=True)
OUT = TMP / "output"

# ── Build Identity ONNX ──
x_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 4])
y_info = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", 4])
node = helper.make_node("Identity", ["input"], ["logits"], name="identity_test")
graph = helper.make_graph([node], "identity_graph", [x_info], [y_info])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
onnx.checker.check_model(model)
onnx_path = TMP / "identity.onnx"
onnx.save(model, str(onnx_path))

# ── Input data ──
x = np.arange(32, dtype=np.float32).reshape(8, 4)
np.save(INP / "input.npy", x)
with open(INP / "manifest.json", "w") as f:
    json.dump({"tensors": [{"name": "input", "file": "input.npy",
               "dtype": "float32", "shape": list(x.shape)}]}, f)

# ── C3.1: DAG export ──
import subprocess
r = subprocess.run([sys.executable, str(HERE / "export_dag.py"),
                    "--onnx", str(onnx_path), "--output", str(TMP / "dag.json")],
                   capture_output=True, text=True, timeout=10)
dag = json.loads((TMP / "dag.json").read_text())
assert dag["nodes"][0]["op_type"] == "Identity", "dag missing Identity"
assert all(set(e) == {"src_node", "dst_node", "tensor"} for e in dag["edges"])
print(f"[C3.1] DAG: {dag['nodes'][0]['op_type']}  edges schema OK")

# ── C3.2: strategy decomposition ──
from c3scheduler.graph import import_onnx_graph
from c3scheduler.strategy import strategy
gir = import_onnx_graph(onnx_path)
n = gir.nodes[0]
prec = strategy.select_precision(n, gir)
kernels = strategy.decompose(n, gir, prec)
tuning = strategy.tune_kernel(kernels[0], prec, 1024)
assert prec.precision == "fp32", f"bad precision: {prec.precision}"
assert kernels[0].name == "identity_fp32", f"bad kernel: {kernels[0].name}"
assert tuning.block_x > 0 and tuning.grid_x > 0
print(f"[C3.2] Strategy: prec={prec.precision}  kernel={kernels[0].name}  "
      f"block={tuning.block_x}  grid={tuning.grid_x}")

# ── C3.5: standalone inference ──
from infer_worker import InferenceEngine
engine = InferenceEngine()
result = engine.infer(str(onnx_path), str(INP), str(OUT), batch_size=4)
assert result["status"] == "ok", f"inference failed: {result}"
assert result["samples"] == 8
y = np.load(OUT / "logits.npy")
assert y.dtype == np.float32
assert y.shape == x.shape
assert np.array_equal(x, y)
print(f"[C3.5] Inference: input={x.shape}  output={y.shape}  "
      f"dtype={y.dtype}  array_equal={np.array_equal(x, y)}  "
      f"allclose={np.allclose(x, y, rtol=1e-3, atol=1e-3)}")

# ── C3.5: worker protocol ──
import subprocess, threading

def read_line(s, timeout=60):
    r = [None]
    def _f(): r.append(s.readline())
    th = threading.Thread(target=_f, daemon=True); th.start(); th.join(timeout)
    return r[-1] if len(r) > 1 else None

proc = subprocess.Popen([sys.executable, str(HERE / "infer_worker.py")],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
ready = read_line(proc.stdout, 60)
assert ready and ready.strip() == "READY", f"bad READY: {ready}"
wout = TMP / "wout"
task = json.dumps({"onnx": str(onnx_path), "input": str(INP), "output": str(wout), "batch_size": 4})
proc.stdin.write(task + "\n"); proc.stdin.flush()
rline = read_line(proc.stdout, 120)
res = json.loads(rline.strip())
assert res.get("status") == "ok", f"worker failed: {res}"
wy = np.load(wout / "logits.npy")
assert np.array_equal(x, wy)
print(f"[C3.5] Worker: status=ok  samples={res['samples']}  array_equal={np.array_equal(x, wy)}")

proc.stdin.write('{"cmd":"exit"}\n'); proc.stdin.flush()
try: proc.wait(15)
except: proc.kill(); proc.wait()

print("\nIDENTITY E2E: ALL PASSED")
