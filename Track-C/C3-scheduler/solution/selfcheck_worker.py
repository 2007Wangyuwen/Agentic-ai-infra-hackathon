#!/usr/bin/env python3
"""C3.5 worker protocol selfcheck — standalone, with timeouts."""
import json, os, sys, tempfile, time, threading
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent

# Create synthetic ONNX model
import onnx
from onnx import helper, TensorProto
import numpy as np

tmp = tempfile.mkdtemp(prefix="c3w_")
print(f"Temp dir: {tmp}")

# MLP ONNX
inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 1, 28, 28])
out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", 10])
w1 = helper.make_tensor("w1", TensorProto.FLOAT, [784, 128], [0.01] * 100352)
b1 = helper.make_tensor("b1", TensorProto.FLOAT, [128], [0.0] * 128)
w2 = helper.make_tensor("w2", TensorProto.FLOAT, [128, 10], [0.01] * 1280)
b2 = helper.make_tensor("b2", TensorProto.FLOAT, [10], [0.0] * 10)
n1 = helper.make_node("Flatten", ["input"], ["f"], name="flatten", axis=1)
n2 = helper.make_node("Gemm", ["f", "w1", "b1"], ["o1"], name="fc1")
n3 = helper.make_node("Relu", ["o1"], ["r"], name="relu")
n4 = helper.make_node("Gemm", ["r", "w2", "b2"], ["logits"], name="fc2")
graph = helper.make_graph([n1, n2, n3, n4], "mlp", [inp], [out],
                          initializer=[w1, b1, w2, b2])
onnx_path = os.path.join(tmp, "mlp.onnx")
model = helper.make_model(graph, producer_name="c3t")
model.opset_import[0].version = 20  # compat with onnxruntime 1.27.0
onnx.save(model, onnx_path)

# Input data
input_dir = os.path.join(tmp, "input")
os.makedirs(input_dir)
data = np.random.randn(128, 1, 28, 28).astype(np.float32)
np.save(os.path.join(input_dir, "input.npy"), data)
with open(os.path.join(input_dir, "manifest.json"), "w") as f:
    json.dump({"tensors": [{"name": "input", "file": "input.npy",
              "dtype": "float32", "shape": [128, 1, 28, 28]}]}, f)

output_dir = os.path.join(tmp, "output")

# ── launch worker ──
print("\nLaunching worker...")
proc = subprocess.Popen(
    [sys.executable, str(HERE / "infer_worker.py")],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    stderr=subprocess.PIPE, text=True,
)

def read_line_with_timeout(stream, timeout=60):
    """Read a line with timeout using threading."""
    result = [None]
    def target():
        result[0] = stream.readline()
    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None  # timeout
    return result[0]

errors = 0

# 1. Wait for READY
print("Waiting for READY...", end=" ")
sys.stdout.flush()
ready = read_line_with_timeout(proc.stdout, timeout=90)
if ready and ready.strip() == "READY":
    print("OK")
else:
    print(f"FAIL: got {repr(ready)}")
    # Check stderr for clues
    try:
        proc.kill()
    except:
        pass
    proc.wait(timeout=5)
    stderr_output = proc.stderr.read()
    print(f"Stderr:\n{stderr_output[:500]}")
    errors += 1

if errors == 0:
    # 2. Send task
    task = json.dumps({"onnx": onnx_path, "input": input_dir,
                       "output": output_dir, "batch_size": 64})
    print(f"Sending task: {task[:80]}...")
    proc.stdin.write(task + "\n")
    proc.stdin.flush()

    # 3. Wait for result
    print("Waiting for result...", end=" ")
    sys.stdout.flush()
    result_line = read_line_with_timeout(proc.stdout, timeout=120)
    if result_line:
        try:
            result = json.loads(result_line.strip())
            if result.get("status") == "ok":
                print(f"OK  samples={result.get('samples')}")
            else:
                print(f"FAIL: {result}")
                errors += 1
        except json.JSONDecodeError:
            print(f"FAIL: invalid JSON: {result_line[:100]}")
            errors += 1
    else:
        print("FAIL: timeout")
        errors += 1

    if errors == 0:
        # 4. Check output files
        out_mani = os.path.join(output_dir, "manifest.json")
        out_npy = os.path.join(output_dir, "logits.npy")
        if os.path.exists(out_mani) and os.path.exists(out_npy):
            logits = np.load(out_npy)
            print(f"Output: shape={logits.shape}, dtype={logits.dtype}")
            print("Output files: OK")
            # Check correctness: expect model output shape [128, 10]
            if logits.shape == (128, 10) and logits.dtype == np.float32:
                print("Output shape/dtype: OK")
            else:
                print(f"FAIL: expected (128,10) float32, got {logits.shape} {logits.dtype}")
                errors += 1
        else:
            print(f"FAIL: output files missing")
            errors += 1

        # 5. Send exit
        print("Sending exit...")
        proc.stdin.write('{"cmd":"exit"}\n')
        proc.stdin.flush()

        # 6. Wait for clean exit
        try:
            proc.wait(timeout=15)
            print(f"Exit code: {proc.returncode}  {'OK' if proc.returncode == 0 else 'FAIL'}")
            if proc.returncode != 0:
                errors += 1
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print("FAIL: worker did not exit")
            errors += 1

# Cleanup stderr read
if proc.poll() is None:
    proc.kill()
    proc.wait()
stderr_output = proc.stderr.read()
if stderr_output:
    print(f"\n--- Worker stderr ---")
    for line in stderr_output.splitlines()[-20:]:
        print(f"  {line}")

# ── summary ──
print(f"\n{'='*50}")
if errors == 0:
    print("C3.5 Worker Protocol: ALL PASSED")
else:
    print(f"C3.5 Worker Protocol: {errors} ERROR(S)")
print(f"{'='*50}")

import shutil
shutil.rmtree(output_dir, ignore_errors=True)
print(f"\nTest artifacts at: {tmp}")
sys.exit(errors)
