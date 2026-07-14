#!/bin/bash
set -e

OUT=tmp_demo
mkdir -p $OUT

echo "=== 婵€娲昏櫄鎷熺幆澧?==="
source .venv/bin/activate

echo ""
echo "=== C3.1: ONNX 鍥捐В鏋?==="
python export_dag.py --onnx ../testcases/release_to_competitors/models/mlp_v1.onnx --output $OUT/mlp_dag.json
python export_dag.py --onnx ../testcases/release_to_competitors/models/resnet_v1.onnx --output $OUT/resnet_dag.json
python export_dag.py --onnx ../testcases/release_to_competitors/models/transformer_v1.onnx --output $OUT/transformer_dag.json

echo ""
echo "=== C3.2/C3.3/C3.4: Scheduler 妯″潡娴嬭瘯 ==="
python test_scheduler.py

echo ""
echo "=== C3.5: MLP 鎺ㄧ悊 (standalone) ==="
python infer.py --onnx ../testcases/release_to_competitors/models/mlp_v1.onnx \
  --input ../testcases/release_to_competitors/testdata/c35/mlp_v1/input \
  --output $OUT/out_mlp --batch-size 256

echo ""
echo "=== C3.5: ResNet 鎺ㄧ悊 (standalone) ==="
python infer.py --onnx ../testcases/release_to_competitors/models/resnet_v1.onnx \
  --input ../testcases/release_to_competitors/testdata/c35/resnet_v1/input \
  --output $OUT/out_resnet --batch-size 128

echo ""
echo "=== C3.5: Transformer 鎺ㄧ悊 (standalone) ==="
python infer.py --onnx ../testcases/release_to_competitors/models/transformer_v1.onnx \
  --input ../testcases/release_to_competitors/testdata/c35/transformer_v1/input \
  --output $OUT/out_transformer --batch-size 64

echo ""
echo "=== 绮惧害楠岃瘉 ==="
python check_output.py --pred $OUT/out_mlp \
  --golden ../testcases/release_to_competitors/testdata/c35/mlp_v1/golden \
  --labels ../testcases/release_to_competitors/testdata/c35/mlp_v1/labels.npy

python check_output.py --pred $OUT/out_resnet \
  --golden ../testcases/release_to_competitors/testdata/c35/resnet_v1/golden \
  --labels ../testcases/release_to_competitors/testdata/c35/resnet_v1/labels.npy

python check_output.py --pred $OUT/out_transformer \
  --golden ../testcases/release_to_competitors/testdata/c35/transformer_v1/golden

echo ""
echo "=== C3.5: Worker 鍗忚鑷祴 ==="
echo "鎻愮ず: 鐢ㄨ祫鏂欏寘涓殑 selfcheck_worker.py 楠岃瘉 worker 鍗忚:"
echo "  python selfcheck_worker.py --worker 'python infer_worker.py' --models mlp_v1 resnet_v1"
echo ""

echo "=== 鍏ㄩ儴瀹屾垚 ==="
