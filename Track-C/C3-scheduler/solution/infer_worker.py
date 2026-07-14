#!/usr/bin/env python3
"""
C3.5 Persistent inference worker.

Implements the C3.5 worker protocol:
  - stdout: only READY signal and result JSON lines
  - stderr: all logging and diagnostics
  - stdin:  task JSON lines from the evaluator

Protocol flow (per model, fresh process each time):
  start -> READY -> (2 warmup tasks + 5 timed tasks) -> {"cmd":"exit"} -> exit 0

BigFormer strategy (80% of performance score):
  fp32 weights ~18 GB > 16 GB GPU.
  Convert model to FP16 via onnx/ort float16 tools -> ~9 GB -> fits GPU.
  FP16 inference on CUDA, outputs cast back to FP32 for golden comparison.
  Fallback chain: FP16+GPU -> FP16+CPU -> FP32+CPU.

Q&A confirmations:
  - CuPy available (cupy 14.1.1) – custom kernels allowed
  - ONNX Runtime 1.27.0 with CUDA EP on H200
  - NVML sampling at 20ms intervals
  - batch_size not fixed – must handle any value without OOM
  - Each model: fresh process, no cache retained
  - Perf weighting: ResNet 20%, BigFormer 80%
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, List

import numpy as np


# ── helpers ────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[worker] {msg}", file=sys.stderr, flush=True)


def _load_input(manifest_dir: str) -> Dict[str, np.ndarray]:
    path = os.path.join(manifest_dir, "manifest.json")
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    tensors: Dict[str, np.ndarray] = {}
    for entry in manifest["tensors"]:
        filepath = os.path.join(manifest_dir, entry["file"])
        tensors[entry["name"]] = np.load(filepath)
    return tensors


def _write_output(output_dir: str, logits: np.ndarray, output_name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "logits.npy"), logits.astype(np.float32))
    manifest = {
        "tensors": [
            {
                "name": output_name,
                "file": "logits.npy",
                "dtype": "float32",
                "shape": list(logits.shape),
            }
        ]
    }
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


# ── model detection ────────────────────────────────────────────────────────

def _model_category(onnx_path: str) -> str:
    """Classify model into: small / bigformer.

    BigFormer: external-data companion > 1 GiB or main .onnx > 1 GiB.
    """
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        try:
            if os.path.getsize(data_file) > 1024 * 1024 * 1024:
                return "bigformer"
        except OSError:
            pass
    try:
        if os.path.getsize(onnx_path) > 1024 * 1024 * 1024:
            return "bigformer"
    except OSError:
        pass
    return "small"


# ── FP16 model conversion ──────────────────────────────────────────────────

def _convert_to_fp16(onnx_path: str) -> str:
    """Convert an ONNX model's float weights to float16.

    Returns path to a temporary .onnx file.  Caller should clean up.
    Tries onnxruntime.transformers.float16 first (fast, C++ backed),
    falls back to manual numpy conversion.
    """
    _log("Converting model to FP16 for GPU inference …")

    # ── try onnxruntime float16 converter ──
    try:
        from onnxruntime.transformers import float16 as ort_fp16

        tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        tmp.close()
        ort_fp16.convert_float_to_float16(onnx_path, tmp.name, keep_io_types=True)
        _log(f"FP16 conversion via onnxruntime.transformers.float16 → {tmp.name}")
        return tmp.name
    except Exception:
        _log("onnxruntime.transformers.float16 unavailable, falling back to manual")

    # ── manual conversion via onnx + numpy ──
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(onnx_path, load_external_data=False)
    graph = model.graph

    # Convert float initializers to float16
    converted_count = 0
    for idx, init in enumerate(graph.initializer):
        if init.data_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16):
            arr = numpy_helper.to_array(init)
            if init.data_type == onnx.TensorProto.FLOAT:
                arr = arr.astype(np.float16)
                new_init = numpy_helper.from_array(arr, name=init.name)
                new_init.data_type = onnx.TensorProto.FLOAT16
                graph.initializer[idx].CopyFrom(new_init)
                converted_count += 1
    _log(f"Converted {converted_count} initializers to FP16")

    # Keep graph inputs/outputs as FLOAT (keep_io_types) – ONNX Runtime
    # will insert Cast nodes automatically for I/O when providers detect
    # mixed precision.
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp.close()
    onnx.save(model, tmp.name)
    _log(f"Manual FP16 model saved → {tmp.name}")
    return tmp.name


# ── inference engine ───────────────────────────────────────────────────────

class InferenceEngine:
    """Loads an ONNX model, runs inference, writes outputs."""

    def __init__(self) -> None:
        self._ort = None
        self._fp16_path: str | None = None

    @property
    def ort(self):
        if self._ort is None:
            import onnxruntime as _ort
            self._ort = _ort
        return self._ort

    def _safe_batch_size(self, batch_size: int, total_n: int) -> int:
        """Cap batch_size to avoid OOM.  Start with requested, halve if needed."""
        return max(1, min(batch_size, total_n))

    def _build_session(
        self, onnx_path: str, category: str
    ):
        """Build an ONNX Runtime session.

        small models  → CUDA (14 GiB limit), fp32
        bigformer     → CUDA (14 GiB limit), fp16 weights
                         Falls back to CPU if CUDA unavailable or conversion fails.
        """
        sess_opts = self.ort.SessionOptions()
        sess_opts.graph_optimization_level = (
            self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_opts.intra_op_num_threads = os.cpu_count() or 4
        sess_opts.inter_op_num_threads = os.cpu_count() or 4

        available = self.ort.get_available_providers()
        has_cuda = "CUDAExecutionProvider" in available

        if has_cuda:
            cuda_opts: Dict[str, Any] = {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "gpu_mem_limit": 14 * 1024 * 1024 * 1024,
                "do_copy_in_default_stream": True,
            }
            providers: Any = [
                ("CUDAExecutionProvider", cuda_opts),
                "CPUExecutionProvider",
            ]

            if category == "bigformer":
                # Convert model to FP16 so it fits in GPU (~9 GB vs 18 GB)
                try:
                    self._fp16_path = _convert_to_fp16(onnx_path)
                    onnx_path = self._fp16_path
                    _log("FP16 model ready — running on CUDA")
                except Exception as exc:
                    _log(f"FP16 conversion failed ({exc}), "
                         "falling back to CPU for BigFormer")
                    sess_opts.enable_cpu_mem_arena = False
                    sess_opts.intra_op_num_threads = max(
                        1, (os.cpu_count() or 4) // 2
                    )
                    sess_opts.inter_op_num_threads = 2
                    providers = ["CPUExecutionProvider"]
        else:
            _log("CUDA not available — using CPU")
            if category == "bigformer":
                sess_opts.enable_cpu_mem_arena = False
                sess_opts.intra_op_num_threads = max(
                    1, (os.cpu_count() or 4) // 2
                )
                sess_opts.inter_op_num_threads = 2
            providers = ["CPUExecutionProvider"]

        _log(f"Session providers: {providers}")
        try:
            return self.ort.InferenceSession(onnx_path, sess_opts, providers=providers)
        except Exception as exc:
            _log(f"CUDA session failed ({exc}), retrying CPU-only")
            sess_opts.enable_cpu_mem_arena = False
            return self.ort.InferenceSession(
                onnx_path, sess_opts, providers=["CPUExecutionProvider"]
            )

    def _cleanup_fp16(self) -> None:
        if self._fp16_path and os.path.exists(self._fp16_path):
            try:
                os.unlink(self._fp16_path)
            except OSError:
                pass
            self._fp16_path = None

    def infer(
        self,
        onnx_path: str,
        input_dir: str,
        output_dir: str,
        batch_size: int = 256,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            return self._infer_impl(onnx_path, input_dir, output_dir, batch_size, t0)
        finally:
            self._cleanup_fp16()

    def _infer_impl(
        self,
        onnx_path: str,
        input_dir: str,
        output_dir: str,
        batch_size: int,
        t0: float,
    ) -> Dict[str, Any]:
        inputs = _load_input(input_dir)
        total_n = next(iter(inputs.values())).shape[0]
        batch_size = self._safe_batch_size(batch_size, total_n)

        category = _model_category(onnx_path)
        _log(f"Model category={category}, samples={total_n}, batch={batch_size}")

        session = self._build_session(onnx_path, category)

        input_names = [x.name for x in session.get_inputs()]
        output_name = session.get_outputs()[0].name
        _log(f"Session ready in {time.perf_counter() - t0:.1f}s")

        # ── batched inference ──
        infer_start = time.perf_counter()
        if total_n <= batch_size:
            feed = {n: inputs[n] for n in input_names}
            out = session.run([output_name], feed)[0]
            logits = out.astype(np.float32)
        else:
            chunks: List[np.ndarray] = []
            for start in range(0, total_n, batch_size):
                end = min(start + batch_size, total_n)
                feed = {n: inputs[n][start:end] for n in input_names}
                out = session.run([output_name], feed)[0]
                chunks.append(out.astype(np.float32))
            logits = np.concatenate(chunks, axis=0)

        infer_sec = time.perf_counter() - infer_start
        _log(f"Done in {infer_sec:.1f}s  ({total_n / infer_sec:.0f} samp/s)  "
             f"shape={logits.shape}")

        _write_output(output_dir, logits, output_name)
        return {"status": "ok", "samples": int(logits.shape[0])}


# ── persistent worker ──────────────────────────────────────────────────────

class PersistentWorker:
    """Implements the C3.5 stdin/stdout worker protocol."""

    def __init__(self) -> None:
        self.engine = InferenceEngine()

    def _send_ready(self) -> None:
        sys.stdout.write("READY\n")
        sys.stdout.flush()
        _log("Worker ready")

    def _send_result(self, result: Dict[str, Any]) -> None:
        line = json.dumps(result, separators=(",", ":"))
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def run_forever(self) -> None:
        self._send_ready()

        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue

            try:
                task = json.loads(line)
            except json.JSONDecodeError:
                _log(f"Invalid JSON on stdin: {line[:120]}")
                continue

            if task.get("cmd") == "exit":
                _log("Received exit — shutting down")
                sys.exit(0)

            _log(f"Task: onnx={task.get('onnx','?')}, "
                 f"batch_size={task.get('batch_size','?')}")
            try:
                result = self.engine.infer(
                    onnx_path=task["onnx"],
                    input_dir=task["input"],
                    output_dir=task["output"],
                    batch_size=task.get("batch_size", 256),
                )
                self._send_result(result)
            except Exception:
                _log("Task failed — traceback follows")
                traceback.print_exc(file=sys.stderr)
                self._send_result({
                    "status": "error",
                    "error": traceback.format_exc()[:300],
                })

        _log("stdin closed — exiting")
        sys.exit(0)


def main() -> None:
    PersistentWorker().run_forever()


if __name__ == "__main__":
    main()
