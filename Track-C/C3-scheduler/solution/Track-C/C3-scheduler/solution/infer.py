#!/usr/bin/env python3
import argparse, json, os
import numpy as np

def load_input(input_dir):
    with open(os.path.join(input_dir, "manifest.json")) as f:
        manifest = json.load(f)
    inputs = {}
    for t in manifest["tensors"]:
        inputs[t["name"]] = np.load(os.path.join(input_dir, t["file"]))
    return inputs

def infer_with_cupy(onnx_path, inputs, batch_size):
    """GPU inference using onnxruntime CUDA provider + CuPy for pre/post."""
    try:
        import cupy as cp
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
        sess = ort.InferenceSession(onnx_path, sess_options, providers=providers)

        input_names = [x.name for x in sess.get_inputs()]
        output_name = sess.get_outputs()[0].name
        total_n = next(iter(inputs.values())).shape[0]

        results = []
        for start in range(0, total_n, batch_size):
            end = min(start + batch_size, total_n)
            # Move to GPU with CuPy then back forort
            batch = {}
            for name in input_names:
                arr = inputs[name][start:end]
                # Use CuPy for data handling
                gpu_arr = cp.asarray(arr)
                batch[name] = cp.asnumpy(gpu_arr)# ort needs numpy
            out = sess.run([output_name], batch)[0]
            results.append(cp.asnumpy(cp.asarray(out, dtype=cp.float32)))

        return np.concatenate(results, axis=0), output_name

    except Exception as e:
        print(f"[WARN] CuPy/CUDA failed ({e}), falling back to CPU")
        return infer_cpu(onnx_path, inputs, batch_size)

def infer_cpu(onnx_path, inputs, batch_size):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = [x.name for x in sess.get_inputs()]
    output_name = sess.get_outputs()[0].name
    total_n = next(iter(inputs.values())).shape[0]
    results = []
    for start in range(0, total_n, batch_size):
        end = min(start + batch_size, total_n)
        batch = {n: inputs[n][start:end] for n in input_names}
        out = sess.run([output_name], batch)[0]
        results.append(out.astype(np.float32))
    return np.concatenate(results, axis=0), output_name

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    inputs = load_input(args.input)
    logits, output_name = infer_with_cupy(args.onnx, inputs, args.batch_size)

    os.makedirs(args.output, exist_ok=True)
    np.save(os.path.join(args.output, "logits.npy"), logits)
    manifest = {"tensors": [{"name": output_name, "file": "logits.npy","dtype": "float32", "shape": list(logits.shape)}]}
    with open(os.path.join(args.output, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] shape={logits.shape}, saved to {args.output}")

if __name__ == "__main__":
    main()
