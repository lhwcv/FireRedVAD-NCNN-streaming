#!/bin/bash
# FireRedVAD 四者对比脚本
# 对比：PyTorch vs ONNX vs NCNN Python vs NCNN C++
# 使用方法：./compare_four.sh [wav_file]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$BASE_DIR/build"
MODELS_DIR="$BASE_DIR/models"
OUTPUT_DIR="$BASE_DIR/output"

WAV_FILE="${1:-$BASE_DIR/test/hello_zh.wav}"

echo "=========================================="
echo "FireRedVAD 四者对比"
echo "=========================================="
echo "Audio: $WAV_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行 NCNN C++ 获取原始数据
echo "Step 1: Running NCNN C++..."
NCNN_CPP_OUTPUT="$OUTPUT_DIR/ncnn_cpp_raw.txt"

if [ ! -f "$BUILD_DIR/test_vad_stream" ]; then
    echo "Error: test_vad_stream not found. Run build.sh first."
    exit 1
fi

cd "$BUILD_DIR"
LD_LIBRARY_PATH=/opt/ncnn-20260113/build/src:$LD_LIBRARY_PATH \
./test_vad_stream \
    "$MODELS_DIR/firered_vad_packed_cache.ncnn.param" \
    "$MODELS_DIR/firered_vad_packed_cache.ncnn.bin" \
    "$MODELS_DIR/cmvn_means.bin" \
    "$MODELS_DIR/cmvn_istd.bin" \
    "$WAV_FILE" 2>&1 | grep "Frame" > "$NCNN_CPP_OUTPUT"

echo "  NCNN C++ output saved to: $NCNN_CPP_OUTPUT"
echo "  Frames: $(wc -l < "$NCNN_CPP_OUTPUT")"
echo ""

# 运行 Python 四者对比
echo "Step 2: Running Python comparison..."
python3 "$SCRIPT_DIR/compare_four_streaming.py" "$WAV_FILE" "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "对比完成！"
echo "=========================================="
echo "图表：$OUTPUT_DIR/compare_four.png"
echo "数据：$OUTPUT_DIR/compare_four.txt"
echo ""
