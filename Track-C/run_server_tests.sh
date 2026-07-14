#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Track-C 服务器测试脚本
# 用法: bash run_server_tests.sh [c1|c2|c3|all]
# 所有输出保存到: ./server_test_output.txt
# 复制该文件贴回给 AI 分析
# ═══════════════════════════════════════════════════════════════════════════
set -o pipefail

OUTPUT_FILE="./server_test_output.txt"
exec > >(tee "$OUTPUT_FILE") 2>&1

echo "============================================"
echo "  Track-C 服务器测试"
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  主机: $(hostname)"
echo "  系统: $(uname -a)"
echo "  Python: $(python3 --version 2>&1)"
echo "  NCC: $(nvcc --version 2>&1 | head -1)"
echo "============================================"

TARGET="${1:-all}"

# ─────────────────────────────────────────────────────────────────
# C1: AEC IR 编译器
# ─────────────────────────────────────────────────────────────────
test_c1() {
    echo ""
    echo "============================================"
    echo "  C1: AEC IR 编译器"
    echo "============================================"

    cd "$SCRIPT_DIR/C1-compiler" || return 1

    # 检查 CModel 是否存在
    C1_CMODEL="./GoldenModel/bin/aec-precise-linux-x86_64"
    if [ ! -f "$C1_CMODEL" ]; then
        echo "[C1] WARN: CModel 未找到: $(pwd)/$C1_CMODEL"
        echo "[C1] 跳过 CModel 执行，仅做编译测试"
    fi

    # 1.1 编译测试 (如果有 testcases)
    if [ -d "./testcases" ]; then
        echo ""
        echo "--- C1.1: 编译公开测试用例 ---"
        for dir in ./testcases/T*_*; do
            if [ -f "$dir/kernel.ptx" ]; then
                name=$(basename "$dir")
                for lvl in O0 O1 O2 O3; do
                    python3 compiler/aec-cc "$dir/kernel.ptx" "-$lvl" -o "/tmp/c1_${name}_${lvl}.aecbin" --report "/tmp/c1_${name}_${lvl}.json" 2>&1
                done
            fi
        done
        echo "[C1.1] 编译完成"
    else
        echo "[C1.1] testcases/ 目录不存在，跳过"
    fi

    # 1.2 CModel 执行 + 正确性验证
    if [ -f "$C1_CMODEL" ] && [ -d "./testcases" ]; then
        echo ""
        echo "--- C1.2: CModel 执行 + 正确性 ---"
        python3 golden_test/eval_sim.py 2>&1
        echo "[C1.2] CModel 测试完成（上方为结果）"
    fi

    # 1.3 鲁棒性测试
    if [ -d "./testcases" ]; then
        echo ""
        echo "--- C1.3: 鲁棒性测试 ---"
        python3 golden_test/robustness_test.py 2>&1
        echo "[C1.3] 鲁棒性测试完成（上方为结果）"
    fi

    echo "[C1] 全部完成"
}

# ─────────────────────────────────────────────────────────────────
# C2: 主机侧 Runtime
# ─────────────────────────────────────────────────────────────────
test_c2() {
    echo ""
    echo "============================================"
    echo "  C2: Host Runtime"
    echo "============================================"

    KIT="$SCRIPT_DIR/C2-runtime/starter-kit"
    if [ ! -d "$KIT" ]; then
        echo "[C2] starter-kit 目录不存在: $KIT"
        return 1
    fi

    cd "$KIT" || return 1

    # 2.1 覆盖我们的实现
    echo "[C2] 覆盖实现文件..."
    cp "$SCRIPT_DIR/C2-runtime/src/aec_runtime.cpp" src/aec_runtime.cpp 2>/dev/null && echo "  src/aec_runtime.cpp OK"
    mkdir -p agents
    cp "$SCRIPT_DIR/C2-runtime/agents/dma_agent.py" agents/dma_agent.py 2>/dev/null && echo "  agents/dma_agent.py OK"
    cp "$SCRIPT_DIR/C2-runtime/agents/kernel_agent.py" agents/kernel_agent.py 2>/dev/null && echo "  agents/kernel_agent.py OK"

    # 2.2 构建
    echo ""
    echo "[C2] 构建 libaec.so ..."
    make clean 2>/dev/null
    make -j2 2>&1
    if [ ! -f "libaec.so" ]; then
        echo "[C2] ERROR: libaec.so 构建失败"
        return 1
    fi
    echo "[C2] libaec.so 构建成功"

    # 2.3 公开评分
    echo ""
    echo "--- C2.3: 公开评分 (public) ---"
    LD_LIBRARY_PATH=$PWD/lib python3 grader/public_grade.py --submission . --profile public 2>&1

    # 2.4 完整评分 (含 hidden performance)
    echo ""
    echo "--- C2.4: 完整评分 (full, 含 hidden) ---"
    LD_LIBRARY_PATH=$PWD/lib python3 grader/public_grade.py --submission . --profile full 2>&1

    echo "[C2] 全部完成"
}

# ─────────────────────────────────────────────────────────────────
# C3: 算子调度与模型部署
# ─────────────────────────────────────────────────────────────────
test_c3() {
    echo ""
    echo "============================================"
    echo "  C3: Operator Scheduler"
    echo "============================================"

    C3_SOL="$SCRIPT_DIR/C3-scheduler/solution"
    cd "$C3_SOL" || return 1

    # 3.0 安装依赖
    echo "[C3] 安装依赖..."
    pip install -r requirements.txt -q 2>&1 | tail -1

    # 3.1 图解析
    echo ""
    echo "--- C3.1: 图解析 ---"
    MODELS="../testcases/release_to_competitors/models"
    if [ -d "$MODELS" ]; then
        for model in mlp_v1 resnet_v1 transformer_v1; do
            if [ -f "$MODELS/${model}.onnx" ]; then
                echo "  Parsing $model..."
                python3 export_dag.py --onnx "$MODELS/${model}.onnx" --output "/tmp/c3_${model}_dag.json" 2>&1
            fi
        done
    else
        echo "  模型目录不存在: $MODELS"
    fi

    # 3.2 / 3.3 / 3.4 Scheduler 模块
    echo ""
    echo "--- C3.2/C3.3/C3.4: Scheduler 模块 ---"
    if [ -f "test_scheduler.py" ] && [ -d "$MODELS" ]; then
        python3 test_scheduler.py 2>&1
    else
        echo "  test_scheduler.py 或模型目录不存在，跳过"
    fi

    # 3.5 Worker 协议自测
    echo ""
    echo "--- C3.5.1: Worker 协议自测 ---"
    SELFCHEK="../testcases/release_to_competitors/selfcheck_worker.py"
    if [ -f "$SELFCHEK" ]; then
        python3 "$SELFCHEK" --worker "python3 $C3_SOL/infer_worker.py" --models mlp_v1 resnet_v1 transformer_v1 2>&1
    else
        echo "  selfcheck_worker.py 不存在: $SELFCHEK"
        echo "  手动测试:"
        echo "  echo 'READY test' | python3 infer_worker.py"
    fi

    # 3.5 端到端精度验证
    echo ""
    echo "--- C3.5.2: 端到端精度验证 ---"
    TESTDATA="../testcases/release_to_competitors/testdata/c35"
    if [ -d "$TESTDATA" ]; then
        for model in mlp_v1 resnet_v1 transformer_v1; do
            local ONNX="$MODELS/${model}.onnx"
            local INP="$TESTDATA/${model}/input"
            local GOLDEN="$TESTDATA/${model}/golden"
            local LABELS="$TESTDATA/${model}/labels.npy"
            local OUT="/tmp/c3_out_${model}"

            if [ -f "$ONNX" ] && [ -d "$INP" ] && [ -d "$GOLDEN" ]; then
                echo ""
                echo "  >>> $model <<<"
                python3 infer.py --onnx "$ONNX" --input "$INP" --output "$OUT" --batch-size 256 2>&1

                LABEL_ARG=""
                [ -f "$LABELS" ] && LABEL_ARG="--labels $LABELS"
                python3 check_output.py --pred "$OUT" --golden "$GOLDEN" $LABEL_ARG 2>&1
            fi
        done
    else
        echo "  测试数据目录不存在: $TESTDATA"
    fi

    # BigFormer 检测
    echo ""
    echo "--- C3.5.3: BigFormer 检测 ---"
    local BF_ONNX="$MODELS/bigformer_v1.onnx"
    if [ -f "$BF_ONNX" ]; then
        local BF_SIZE=$(stat --printf="%s" "$BF_ONNX" 2>/dev/null || ls -l "$BF_ONNX" | awk '{print $5}')
        echo "  BigFormer 模型存在, 大小: $BF_SIZE bytes"
        if [ -f "${BF_ONNX}.data" ]; then
            local BF_DATA_SIZE=$(stat --printf="%s" "${BF_ONNX}.data" 2>/dev/null || ls -l "${BF_ONNX}.data" | awk '{print $5}')
            echo "  BigFormer 外部数据文件存在, 大小: $BF_DATA_SIZE bytes"
        fi

        # 测试 FP16 转换是否成功
        echo "  测试 FP16 转换..."
        python3 -c "
import onnx, sys
from onnx import helper
try:
    from onnxruntime.transformers import float16 as ort_fp16
    print('  onnxruntime.transformers.float16 可用')
    model = onnx.load('$BF_ONNX')
    print(f'  模型加载成功, opset_imports={[o.domain+str(o.version) for o in model.opset_import]}')
except Exception as e:
    print(f'  FP16 模块不可用或加载失败: {e}')
" 2>&1
    else
        echo "  BigFormer 模型不存在: $BF_ONNX"
    fi

    echo "[C3] 全部完成"
}

# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
# 获取脚本所在目录（Track-C 根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "测试目标: $TARGET"
echo "Track-C 目录: $SCRIPT_DIR"
echo ""

case "$TARGET" in
    c1)  test_c1 ;;
    c2)  test_c2 ;;
    c3)  test_c3 ;;
    all)
        test_c1
        test_c2
        test_c3
        ;;
    *)
        echo "用法: bash run_server_tests.sh [c1|c2|c3|all]"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  全部测试完成"
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  完整输出已保存到: $OUTPUT_FILE"
echo "============================================"
