#!/bin/bash
# FireRedVAD NCNN Stream-VAD 一键构建脚本
# 使用方法：./build.sh [clean|build|install|test|all]
# 默认使用官方 Stream-VAD 模型（流式专用）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
INSTALL_DIR="$SCRIPT_DIR/install"
NCNN_ROOT="${NCNN_ROOT:-/opt/ncnn-20260113}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    echo "FireRedVAD NCNN 构建脚本"
    echo ""
    echo "使用方法：$0 [command]"
    echo ""
    echo "Commands:"
    echo "  clean    清理构建目录"
    echo "  build    构建项目（默认）"
    echo "  install  安装到 install/ 目录"
    echo "  test     运行测试（需要 hello_zh.wav）"
    echo "  all      完整流程：clean -> build -> install -> test"
    echo "  help     显示帮助信息"
    echo ""
    echo "环境变量:"
    echo "  NCNN_ROOT  NCNN 安装路径（默认：/opt/ncnn-20260113）"
    echo ""
}

clean() {
    print_info "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
    rm -rf "$INSTALL_DIR"
    print_info "Clean completed"
}

build() {
    print_info "Building FireRedVAD NCNN (Stream-VAD model)..."
    
    if [ ! -d "$BUILD_DIR" ]; then
        mkdir -p "$BUILD_DIR"
    fi
    
    cd "$BUILD_DIR"
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DNCNN_ROOT="$NCNN_ROOT"
    
    make -j$(nproc)
    
    print_info "Build completed successfully!"
    print_info "Binaries: $BUILD_DIR/"
    print_info "  - libfirered_vad_stream.so (Stream-VAD C API)"
    print_info "  - test_vad_stream (Stream-VAD test executable)"
}

install() {
    print_info "Installing to $INSTALL_DIR..."
    
    if [ ! -d "$BUILD_DIR" ]; then
        print_error "Build directory not found. Run 'build' first."
        exit 1
    fi
    
    cd "$BUILD_DIR"
    make install
    
    print_info "Installation completed!"
    print_info "Installed to: $INSTALL_DIR"
    print_info "  - include/: Header files"
    print_info "  - lib/: Libraries"
    print_info "  - bin/: Executables"
}

test() {
    print_info "Running tests..."
    
    TEST_WAV="$SCRIPT_DIR/test/hello_zh.wav"
    
    if [ ! -f "$TEST_WAV" ]; then
        print_warn "Test audio not found: $TEST_WAV"
        print_warn "Please provide a test WAV file (16kHz, mono)"
        exit 1
    fi
    
    if [ ! -f "$BUILD_DIR/test_vad_stream" ]; then
        print_error "Test executable not found. Run 'build' first."
        exit 1
    fi
    
    # 查找模型文件（Stream-VAD 版本）
    MODEL_PARAM="$SCRIPT_DIR/models/firered_vad_packed_cache_stream.ncnn.param"
    MODEL_BIN="$SCRIPT_DIR/models/firered_vad_packed_cache_stream.ncnn.bin"
    CMVN_MEANS="$SCRIPT_DIR/models/cmvn_means_stream.bin"
    CMVN_ISTD="$SCRIPT_DIR/models/cmvn_istd_stream.bin"
    
    if [ ! -f "$MODEL_PARAM" ] || [ ! -f "$MODEL_BIN" ]; then
        print_error "Model files not found in $SCRIPT_DIR/models/"
        print_error "Please run convert scripts first or download pretrained models"
        exit 1
    fi
    
    cd "$BUILD_DIR"
    
    export LD_LIBRARY_PATH="$NCNN_ROOT/build/src:$LD_LIBRARY_PATH"
    
    print_info "Running test_vad_stream (Stream-VAD model)..."
    ./test_vad_stream \
        "$MODEL_PARAM" \
        "$MODEL_BIN" \
        "$CMVN_MEANS" \
        "$CMVN_ISTD" \
        "$TEST_WAV"
    
    print_info "Test completed!"
}

all() {
    print_info "Running full build process..."
    echo ""
    clean
    echo ""
    build
    echo ""
    install
    echo ""
    test
    echo ""
    print_info "All steps completed successfully!"
}

# 主逻辑
case "${1:-build}" in
    clean)
        clean
        ;;
    build)
        build
        ;;
    install)
        install
        ;;
    test)
        test
        ;;
    all)
        all
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
