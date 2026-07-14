#!/usr/bin/env python3
"""C3.5 本地精度/准确率验证工具。

用法:
  python check_output.py --pred <out_dir> --golden <golden_dir> [--labels labels.npy]
  python check_output.py --pred <out_dir> --golden <golden_dir>     # 仅精度 (Transformer / BigFormer)
"""
import argparse
import json
import os

import numpy as np


def load_tensor(d: str) -> np.ndarray:
    with open(os.path.join(d, "manifest.json"), encoding="utf-8") as f:
        m = json.load(f)
    t = m["tensors"][0]
    return np.load(os.path.join(d, t["file"]))


def main() -> None:
    p = argparse.ArgumentParser(description="C3.5 output validator")
    p.add_argument("--pred", required=True)
    p.add_argument("--golden", required=True)
    p.add_argument("--labels", default=None)
    args = p.parse_args()

    pred = load_tensor(args.pred)
    golden = load_tensor(args.golden)

    max_diff = float(np.max(np.abs(pred - golden)))
    allclose = bool(np.allclose(pred, golden, rtol=1e-3, atol=1e-3))

    print(f"allclose: {allclose}, max_diff: {max_diff:.6f}")
    if not allclose:
        print(f"  [WARN] max_diff {max_diff:.6f} exceeds threshold 1e-3")

    if args.labels:
        labels = np.load(args.labels)
        acc = float((pred.argmax(-1) == labels).mean())
        print(f"accuracy: {acc:.4f}")
    else:
        print("accuracy: n/a (no labels file provided)")


if __name__ == "__main__":
    main()
