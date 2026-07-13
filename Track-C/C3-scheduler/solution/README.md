# C3 算子调度与模型部署

## 命令模板

C3.1:
```bash
python export_dag.py --onnx {onnx} --output {output}
```

C3.5:
```bash
python infer.py --onnx {onnx} --input {input} --output {output} --batch-size 256
```

## 依赖安装

```bash
pip install -r requirements.txt
```

## 验证结果

- MLP: accuracy=98.35%
- ResNet-18: accuracy=93.51%
- Transformer: allclose=True
