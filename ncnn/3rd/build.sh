#!/usr/bin/env sh


# Minimal NCNN builder (host, static)
# Usage: ./3rd/build.sh [version]
# Default version: 20260113

VER="${1:-20260113}"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
UNAME_S="$(uname -s)"
case "$UNAME_S" in
  Darwin) PLATFORM="MacOS"; OS_CMAKE_OPTS="-DCMAKE_OSX_DEPLOYMENT_TARGET=10.13" ;;
  Linux)  PLATFORM="Linux";  OS_CMAKE_OPTS="" ;;
  *)      PLATFORM="Linux";  OS_CMAKE_OPTS=""; echo "[ncnn] Unknown OS, defaulting to Linux prebuild layout";;
esac
SRC_DIR="$BASE_DIR/source/ncnn-$VER"
ZIP="$BASE_DIR/source/ncnn-$VER.zip"
URL="https://github.com/Tencent/ncnn/archive/refs/tags/$VER.zip"
BUILD_DIR="$BASE_DIR/ncnn_build"
INSTALL_PREFIX="$BASE_DIR/prebuild/$PLATFORM/ncnn"

mkdir -p "$BASE_DIR/source"

if [ ! -d "$SRC_DIR" ]; then
  echo "[ncnn] Source not found: $SRC_DIR"
  if [ ! -f "$ZIP" ]; then
    echo "[ncnn] Downloading $URL -> $ZIP (using curl/wget if available)"
    if command -v curl >/dev/null 2>&1; then
      curl -fL "$URL" -o "$ZIP" || { echo "Download failed; place zip at $ZIP"; exit 1; }
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$ZIP" "$URL" || { echo "Download failed; place zip at $ZIP"; exit 1; }
    else
      echo "No curl/wget found. Manually download to: $ZIP"; exit 1
    fi
  fi
  echo "[ncnn] Unzipping $ZIP"
  unzip -q -o "$ZIP" -d "$BASE_DIR/source"
fi

rm -rf "$BUILD_DIR"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DNCNN_VULKAN=OFF \
  -DNCNN_PIXEL=ON \
  -DNCNN_BUILD_TOOLS=OFF \
  -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_TESTS=OFF \
  -DNCNN_OPENMP=OFF \
  -DNCNN_BENCHMARK=OFF \
  -DNCNN_SSE2=OFF \
  -DNCNN_AVX2=OFF \
  ${OS_CMAKE_OPTS} \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"

cmake --build "$BUILD_DIR" -- -j"${JOBS:-$(getconf _NPROCESSORS_ONLN || echo 4)}"
cmake --install "$BUILD_DIR"

echo "[ncnn] Installed to: $INSTALL_PREFIX"
echo "[ncnn] Library:      $INSTALL_PREFIX/lib/libncnn.a"
echo "[ncnn] Headers:      $INSTALL_PREFIX/include/"
echo "[ncnn] Set NCNN_ROOT to install prefix for runtime CMake:"
echo "        export NCNN_ROOT=\"$INSTALL_PREFIX\""
