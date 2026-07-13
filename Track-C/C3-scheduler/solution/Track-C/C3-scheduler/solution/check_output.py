#!/usr/bin/env python3
import argparse, json, os
import numpy as np

def load_tensor(d):
    with open(os.path.join(d, "manifest.json")) as f:
        m = json.load(f)
    t = m["tensors"][0]
    return np.load(os.path.join(d, t["file"]))

p = argparse.ArgumentParser()
p.add_argument("--pred", required=True)
p.add_argument("--golden", required=True)
p.add_argument("--labels", default=None)
args = p.parse_args()

pred = load_tensor(args.pred)
golden = load_tensor(args.golden)
ok = np.allclose(pred, golden, rtol=1e-3, atol=1e-3)
print(f"allclose: {ok}, max_diff: {np.max(np.abs(pred - golden)):.6f}")
if args.labels:
    labels = np.load(args.labels)
    acc = (pred.argmax(-1) == labels).mean()
    print(f"accuracy: {acc:.4f}")
