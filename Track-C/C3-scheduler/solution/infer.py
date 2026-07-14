#!/usr/bin/env python3
"""C3.5 standalone inference runner — local debugging convenience.

For the evaluator, use the persistent worker:  python infer_worker.py
"""
import argparse
import time

from infer_worker import InferenceEngine


def main() -> None:
    p = argparse.ArgumentParser(description="C3.5 AEC inference (standalone)")
    p.add_argument("--onnx", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    t0 = time.perf_counter()
    engine = InferenceEngine()
    result = engine.infer(args.onnx, args.input, args.output, args.batch_size)
    elapsed = time.perf_counter() - t0

    print(f"[OK] status={result['status']}  samples={result['samples']}  "
          f"time={elapsed:.1f}s  saved to {args.output}")


if __name__ == "__main__":
    main()
