#!/bin/bash
set -e

echo "=== 激活虚拟环境 ==="
source .venv/bin/activate

echo ""
echo "=== C3.1: ONNX 图解析 ==="
python export_dag.py --onnx ../testcases/release_to_competitors/models/mlp_v1.onnx --output mlp_dag.json
python export_dag.py --onnx ../testcases/release_to_competitors/models/resnet_v1.onnx --output resnet_dag.json
python export_dag.py --onnx ../testcases/release_to_competitors/models/transformer_v1.onnx --output transformer_dag.json

echo ""
echo "=== C3.5: MLP 推理 ==="
python infer.py --onnx ../testcases/release_to_competitors/models/mlp_v1.onnx \
  --input ../testcases/release_to_competitors/testdata/c35/mlp_v1/input \
  --output ./out_mlp --batch-size 256

echo ""
echo "=== C3.5: ResNet 推理 ==="
python infer.py --onnx ../testcases/release_to_competitors/models/resnet_v1.onnx \
  --input ../testcases/release_to_competitors/testdata/c35/resnet_v1/input \
  --output ./out_resnet --batch-size 128

echo ""
echo "=== C3.5: Transformer 推理 ==="
python infer.py --onnx ../testcases/release_to_competitors/models/transformer_v1.onnx \
  --input ../testcases/release_to_competitors/testdata/c35/transformer_v1/input \
  --output ./out_transformer --batch-size 64

echo ""
echo "=== 精度验证 ==="
python check_output.py --pred ./out_mlp \
  --golden ../testcases/release_to_competitors/testdata/c35/mlp_v1/golden \
  --labels ../testcases/release_to_competitors/testdata/c35/mlp_v1/labels.npy

python check_output.py --pred ./out_resnet \
  --golden ../testcases/release_to_competitors/testdata/c35/resnet_v1/golden \
  --labels ../testcases/release_to_competitors/testdata/c35/resnet_v1/labels.npy

python check_output.py --pred ./out_transformer \
  --golden ../testcases/release_to_competitors/testdata/c35/transformer_v1/golden

echo ""
echo "=== 全部完成 ==="
